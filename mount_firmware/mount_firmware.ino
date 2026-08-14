// ============================================================
// Dual-stepper controller for ESP32 (TMC2209 STEP/DIR)
// + Fixed 1/64 microstepping (configured by hardware wiring)
// + Bluetooth Classic SPP via BluetoothSerial
//
// Commands (newline-terminated; CR/LF/CRLF accepted):
//   PING
//   ENABLE 0|1
//   MS 64                            (legacy compatibility; no-op)
//   MS AZ|ALT 64                     (legacy compatibility; no-op)
//   STOP
//   MOVE A|B FWD|REV steps delay_us [SMOOTH|DIRECT]
//   STATUS                              (reports MS=64 MSFIXED=1)
//   DEBUG 0|1                       (toggle ALIVE heartbeat)
//
// YOUR PCB pinout (ESP32 30-pin):
//   EN  (common): GPIO21  (LOW=enabled)
//   AZ:  STEP=GPIO33  DIR=GPIO25
//   ALT: STEP=GPIO26  DIR=GPIO27
//
// Notes (Mac terminal cleanliness):
// - Responses are forced to CRLF ("\r\n") regardless of terminal.
// - READY is sent ONLY when SPP channel is actually opened.
// - ALIVE is optional (DEBUG 1) and is suppressed while you are typing.
// ============================================================

#include <Arduino.h>
#include "BluetoothSerial.h"
#include "esp_spp_api.h"
#include "driver/gpio.h"

BluetoothSerial SerialBT;

// --- Pins ---
static const uint8_t STEP_A = 33;  // AZ STEP
static const uint8_t DIR_A  = 25;  // AZ DIR
static const uint8_t STEP_B = 26;  // ALT STEP
static const uint8_t DIR_B  = 27;  // ALT DIR

static const uint8_t EN_PIN  = 21; // LOW=enabled (common)

static const uint16_t STEP_PULSE_US = 3;

// MOVE acceleration profile. A GoTo may request a short period for a long
// slew, but starting/stopping at that cadence excites the telescope structure.
// Start slowly and use a symmetric S-curve in step frequency.  Short moves
// stay inside a gentle triangular profile; long slews keep their requested
// high speed only in the central cruise section.
static const float MOVE_MAX_RATE_STEPS_S = 12000.0f;
static const float MOVE_SMOOTH_START_RATE_STEPS_S = 400.0f;
static const float MOVE_SMOOTH_MAX_ACCEL_STEPS_S2 = 4000.0f;
static const float SMOOTHERSTEP_MAX_DERIVATIVE = 1.875f;

static bool g_enabled = false;

// MOVE scheduler (non-blocking)
static volatile long g_moveRemA = 0;
static volatile long g_moveRemB = 0;
static long g_moveTotalA = 0;
static long g_moveTotalB = 0;
static uint32_t moveTargetPerA_us = 0;
static uint32_t moveTargetPerB_us = 0;
static uint32_t movePerA_us  = 0;
static uint32_t movePerB_us  = 0;
static uint32_t moveNextA_us = 0;
static uint32_t moveNextB_us = 0;
static bool moveSmoothA = true;
static bool moveSmoothB = true;

static const uint16_t FIXED_MICROSTEPS = 64;

// BT state / debug
static volatile bool g_btConnected = false;
static bool g_debugAlive = false;     // default OFF (clean terminal)
static uint32_t g_lastRxMs = 0;       // last received byte time (ms)

static inline void setHighDrive(uint8_t pin) {
  // Increase output drive strength for long traces / noisy loads.
  gpio_set_drive_capability((gpio_num_t)pin, GPIO_DRIVE_CAP_3);
}

static inline void pulseStep(uint8_t pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(pin, LOW);
}

static inline void setEnable(bool on) {
  g_enabled = on;
  digitalWrite(EN_PIN, on ? LOW : HIGH);
}

static inline void clearMovePlans() {
  g_moveRemA = 0;
  g_moveRemB = 0;
  g_moveTotalA = 0;
  g_moveTotalB = 0;
  moveTargetPerA_us = 0;
  moveTargetPerB_us = 0;
  movePerA_us = 0;
  movePerB_us = 0;
  moveNextA_us = 0;
  moveNextB_us = 0;
}

static uint32_t profiledMovePeriodUs(
  uint32_t requestedPer_us,
  long total,
  long remaining,
  bool smoothProfile
) {
  const uint32_t safeMinPeriod_us = (uint32_t)ceilf(1000000.0f / MOVE_MAX_RATE_STEPS_S);
  const uint32_t targetPer_us = max(requestedPer_us, safeMinPeriod_us);
  if (targetPer_us == 0 || total <= 0 || remaining <= 0) return targetPer_us;
  if (!smoothProfile) return targetPer_us;

  const float targetRate = 1000000.0f / (float)targetPer_us;
  const float startRate = min(
    targetRate,
    MOVE_SMOOTH_START_RATE_STEPS_S
  );
  if (targetRate <= startRate) return targetPer_us;

  const long completed = max(0L, total - remaining);
  const long stoppingEdge = max(0L, remaining - 1L);
  const float edgeSteps = (float)min(completed, stoppingEdge);
  const float halfMoveSteps = max(1.0f, (float)total / 2.0f);
  const float idealRampSteps = (
    (targetRate * targetRate - startRate * startRate)
    * SMOOTHERSTEP_MAX_DERIVATIVE
    / (2.0f * MOVE_SMOOTH_MAX_ACCEL_STEPS_S2)
  );
  const float rampSteps = max(1.0f, min(halfMoveSteps, idealRampSteps));
  if (edgeSteps >= rampSteps) return targetPer_us;

  const float peakRateSq = min(
    targetRate * targetRate,
    startRate * startRate
      + 2.0f * MOVE_SMOOTH_MAX_ACCEL_STEPS_S2
      * rampSteps / SMOOTHERSTEP_MAX_DERIVATIVE
  );
  const float x = edgeSteps / rampSteps;
  // Smootherstep: continuous jerk and zero acceleration at both endpoints.
  const float smoother = x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
  const float rate = sqrtf(
    startRate * startRate + (peakRateSq - startRate * startRate) * smoother
  );
  if (!(rate > 0.0f)) return (uint32_t)ceilf(1000000.0f / startRate);
  const uint32_t period_us = (uint32_t)roundf(1000000.0f / rate);
  return max(safeMinPeriod_us, period_us);
}

static void startMoveNonBlocking(
  char axis,
  bool fwd,
  long steps,
  long delay_us,
  bool smoothProfile
) {
  if (steps <= 0) return;
  if (delay_us < 0) delay_us = 0;

  uint32_t per_us = (uint32_t)(max(1L, delay_us + (long)STEP_PULSE_US));
  uint32_t now = micros();

  if (axis == 'A') {
    digitalWrite(DIR_A, fwd ? HIGH : LOW);
    g_moveRemA = steps;
    g_moveTotalA = steps;
    moveSmoothA = smoothProfile;
    moveTargetPerA_us = per_us;
    movePerA_us = profiledMovePeriodUs(per_us, steps, steps, moveSmoothA);
    moveNextA_us = now;
  } else { // 'B'
    digitalWrite(DIR_B, fwd ? HIGH : LOW);
    g_moveRemB = steps;
    g_moveTotalB = steps;
    moveSmoothB = smoothProfile;
    moveTargetPerB_us = per_us;
    movePerB_us = profiledMovePeriodUs(per_us, steps, steps, moveSmoothB);
    moveNextB_us = now;
  }
}

// ---- BT output helpers (force CRLF) ----
static inline void btPrintCRLF(const char* s) {
  if (!g_btConnected) return;
  SerialBT.print(s);
  SerialBT.print("\r\n");
}

static void replyBT(const String& s) {
  if (!g_btConnected) return;
  SerialBT.print(s);
  SerialBT.print("\r\n");
}

// ---- Line reader (accept CR/LF/CRLF) ----
static bool readLineBT(String &outLine) {
  static String line;
  while (SerialBT.available()) {
    char c = (char)SerialBT.read();
    g_lastRxMs = millis();

    if (c == '\r' || c == '\n') {
      if (line.length() == 0) continue;  // swallow empty from CRLF
      outLine = line;
      line = "";
      outLine.trim();
      return true;
    }

    line += c;
    if (line.length() > 200) { line = ""; outLine = ""; return true; }
  }
  return false;
}

// ---- SPP callback: READY only when RFCOMM actually opens ----
void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
  if (event == ESP_SPP_SRV_OPEN_EVT) {
    g_btConnected = true;
    btPrintCRLF("READY");
  } else if (event == ESP_SPP_CLOSE_EVT) {
    g_btConnected = false;
    clearMovePlans();
    setEnable(false);
  }
}

void setup() {
  pinMode(STEP_A, OUTPUT);
  pinMode(DIR_A, OUTPUT);
  pinMode(STEP_B, OUTPUT);
  pinMode(DIR_B, OUTPUT);

  pinMode(EN_PIN, OUTPUT);

  setHighDrive(STEP_A);
  setHighDrive(DIR_A);
  setHighDrive(STEP_B);
  setHighDrive(DIR_B);
  setHighDrive(EN_PIN);

  digitalWrite(STEP_A, LOW);
  digitalWrite(STEP_B, LOW);
  digitalWrite(DIR_A, LOW);
  digitalWrite(DIR_B, LOW);

  setEnable(false);
  delay(2);

  SerialBT.register_callback(btCallback);
  SerialBT.begin("AstroPanoptes-ESP32");
}

void loop() {
  // Heartbeat opcional (DEBUG), evita intercalar con escritura:
  // - manda ALIVE cada 2s
  // - pero si hubo RX en los últimos 500 ms, no imprime
  static uint32_t tAlive = 0;
  if (g_btConnected && g_debugAlive && (millis() - tAlive) > 2000) {
    tAlive = millis();
    if (millis() - g_lastRxMs > 500) {
      btPrintCRLF("ALIVE");
    }
  }

  // --- move scheduler ---
  if (g_enabled) {
    uint32_t now = micros();

    if (g_moveRemA > 0 && movePerA_us > 0 && (int32_t)(now - moveNextA_us) >= 0) {
      const uint32_t stepStarted_us = micros();
      pulseStep(STEP_A);
      g_moveRemA -= 1;
      if (g_moveRemA <= 0) {
        g_moveRemA = 0;
        movePerA_us = 0;
      } else {
        movePerA_us = profiledMovePeriodUs(moveTargetPerA_us, g_moveTotalA, g_moveRemA, moveSmoothA);
        // Schedule from the pulse start so movePerA_us is the true
        // step-to-step period. Using micros() after the pulse silently added
        // STEP_PULSE_US a second time and made host duration estimates drift.
        moveNextA_us = stepStarted_us + movePerA_us;
      }
    }

    if (g_moveRemB > 0 && movePerB_us > 0 && (int32_t)(now - moveNextB_us) >= 0) {
      const uint32_t stepStarted_us = micros();
      pulseStep(STEP_B);
      g_moveRemB -= 1;
      if (g_moveRemB <= 0) {
        g_moveRemB = 0;
        movePerB_us = 0;
      } else {
        movePerB_us = profiledMovePeriodUs(moveTargetPerB_us, g_moveTotalB, g_moveRemB, moveSmoothB);
        moveNextB_us = stepStarted_us + movePerB_us;
      }
    }
  }

  // --- commands over BT ---
  String cmd;
  if (!readLineBT(cmd)) return;
  if (cmd.length() == 0) return;

  char buf[220];
  cmd.toCharArray(buf, sizeof(buf));
  char *tok = strtok(buf, " ");
  if (!tok) return;

  if (!strcmp(tok, "PING")) {
    btPrintCRLF("READY");
    return;
  }

  if (!strcmp(tok, "DEBUG")) {
    char *a = strtok(NULL, " ");
    int on = a ? atoi(a) : 0;
    g_debugAlive = (on != 0);
    replyBT(String("OK DEBUG ") + (g_debugAlive ? "1" : "0"));
    return;
  }

  if (!strcmp(tok, "ENABLE")) {
    char *a = strtok(NULL, " ");
    int on = a ? atoi(a) : 0;
    bool enableOn = (on != 0);
    if (enableOn) {
      bool wasEnabled = g_enabled;
      if (!wasEnabled) {
        setEnable(true);
        delay(2);
      }
    } else {
      setEnable(false);
      clearMovePlans();
    }
    btPrintCRLF("OK");
    return;
  }

  if (!strcmp(tok, "STOP")) {
    clearMovePlans();
    btPrintCRLF("OK");
    return;
  }

  if (!strcmp(tok, "MS")) {
    char *a1 = strtok(NULL, " ");
    if (!a1) { btPrintCRLF("ERR"); return; }

    uint16_t ms = 0;
    if (!strcmp(a1, "AZ") || !strcmp(a1, "ALT")) {
      char *a2 = strtok(NULL, " ");
      ms = a2 ? (uint16_t)atoi(a2) : 0;
    } else {
      ms = (uint16_t)atoi(a1);
    }

    if (ms != FIXED_MICROSTEPS) {
      replyBT(String("ERR MS_FIXED ") + String((uint16_t)FIXED_MICROSTEPS));
      return;
    }
    replyBT(String("OK MS_FIXED ") + String((uint16_t)FIXED_MICROSTEPS));
    return;
  }

  if (!strcmp(tok, "MOVE")) {
    char *ax = strtok(NULL, " ");
    char *dr = strtok(NULL, " ");
    char *st = strtok(NULL, " ");
    char *du = strtok(NULL, " ");
    char *pf = strtok(NULL, " ");

    if (!ax || !dr || !st || !du) { btPrintCRLF("ERR"); return; }
    char axis = ax[0];
    if (!(axis == 'A' || axis == 'B')) { btPrintCRLF("ERR"); return; }

    bool fwd = (!strcmp(dr, "FWD"));
    if (!(fwd || !strcmp(dr, "REV"))) { btPrintCRLF("ERR"); return; }

    long steps = atol(st);
    long delay_us = atol(du);
    if (delay_us < 0) delay_us = 0;
    bool smoothProfile = true;
    if (pf && !strcmp(pf, "DIRECT")) {
      smoothProfile = false;
    } else if (pf && strcmp(pf, "SMOOTH")) {
      btPrintCRLF("ERR PROFILE");
      return;
    }

    startMoveNonBlocking(axis, fwd, steps, delay_us, smoothProfile);
    btPrintCRLF("OK");
    return;
  }

  if (!strcmp(tok, "STATUS")) {
    String s;
    s.reserve(110);
    s += "EN=";    s += (g_enabled ? "1" : "0");
    s += " MS=";   s += String((uint16_t)FIXED_MICROSTEPS);
    s += " MSFIXED=1";
    s += " MOVEPROFILES=1";
    s += " MOVE="; s += String((long)g_moveRemA); s += ","; s += String((long)g_moveRemB);
    s += " PROFILE="; s += (moveSmoothA ? "SMOOTH" : "DIRECT");
    s += ","; s += (moveSmoothB ? "SMOOTH" : "DIRECT");
    s += " BT=";   s += (g_btConnected ? "1" : "0");
    s += " DBG=";  s += (g_debugAlive ? "1" : "0");
    replyBT(s);
    return;
  }

  btPrintCRLF("ERR");
}
