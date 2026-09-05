// ============================================================
// Triple-stepper controller for ESP32 (TMC2209 STEP/DIR)
// + Fixed 1/64 microstepping (configured by hardware wiring)
// + Bluetooth Classic SPP via BluetoothSerial
//
// Axes:
//   A = AZ    (mount azimuth)
//   B = ALT   (mount altitude)
//   C = FOCUS (focuser, drives the telescope focus knob)
//
// Commands (newline-terminated; CR/LF/CRLF accepted):
//   PING
//   ENABLE 0|1
//   MS 64                            (legacy compatibility; no-op)
//   MS AZ|ALT 64                     (legacy compatibility; no-op)
//   STOP                             (all axes)
//   STOP A|B|C                       (single axis)
//   MOVE A|B|C FWD|REV steps delay_us [SMOOTH|DIRECT]
//   TEST A|B|C FWD|REV steps delay_us [SMOOTH|DIRECT]
//   STATUS                              (reports MS=64 MSFIXED=1 AXES=3)
//   DEBUG 0|1                       (toggle ALIVE heartbeat)
//
// MOVE vs TEST
//   MOVE is the observing command and stays clamped to MOVE_MAX_RATE_STEPS_S,
//   which protects the telescope's 45:1 drive from being commanded past what it
//   can actually follow. TEST runs the identical scheduler with that clamp
//   lifted, for bench-testing other motors whose reduction allows much higher
//   step rates. Both accept SMOOTH (S-curve accel/brake) and DIRECT (constant
//   rate). Never use TEST on the telescope drive.
//
// YOUR PCB pinout (ESP32 30-pin):
//   EN  (common): GPIO21  (LOW=enabled)
//   AZ:    STEP=GPIO33  DIR=GPIO25
//   ALT:   STEP=GPIO26  DIR=GPIO27
//   FOCUS: STEP=GPIO14  DIR=GPIO13     <-- ADJUST to your CNC-shield wiring
//
// If you change the focuser pins, avoid GPIO 34/35/36/39 (input-only, they
// cannot drive STEP or DIR at all) and the strapping pins 0/2/4/5/12/15. GPIO12
// in particular selects the flash voltage at reset: a driver input holding it
// high stops the board from booting, and that failure looks like a dead ESP32
// rather than a wiring mistake.
// ============================================================

#include <Arduino.h>
#include "BluetoothSerial.h"
#include "esp_spp_api.h"
#include "driver/gpio.h"

BluetoothSerial SerialBT;

// --- Axis indices ---
enum AxisId { AX_A = 0, AX_B = 1, AX_C = 2, AX_COUNT = 3 };

// --- Pins (index by AxisId) ---
// The focuser is the third driver socket on the CNC shield. Change STEP/DIR
// below to match how that socket is wired to the ESP32.
static const uint8_t STEP_PIN[AX_COUNT] = { 33, 26, 14 };
static const uint8_t DIR_PIN [AX_COUNT] = { 25, 27, 13 };

static const uint8_t EN_PIN  = 21; // LOW=enabled (common to all drivers)

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

// Absolute ceiling for TEST. Not a drive-safety limit but a physical one: the
// scheduler cannot emit pulses faster than the loop can service them, and the
// STEP pulse itself takes STEP_PULSE_US.
static const float TEST_MAX_RATE_STEPS_S = 200000.0f;

static bool g_enabled = false;

// MOVE scheduler (non-blocking), per axis
static volatile long g_moveRem[AX_COUNT]   = { 0, 0, 0 };
static long g_moveTotal[AX_COUNT]          = { 0, 0, 0 };
static uint32_t moveTargetPer_us[AX_COUNT] = { 0, 0, 0 };
static uint32_t movePer_us[AX_COUNT]       = { 0, 0, 0 };
static uint32_t moveNext_us[AX_COUNT]      = { 0, 0, 0 };
static bool moveSmooth[AX_COUNT]           = { true, true, true };
// Set by TEST: run this axis with the observing speed clamp lifted.
static bool moveUnlimited[AX_COUNT]        = { false, false, false };

static const uint16_t FIXED_MICROSTEPS = 64;

// BT state / debug
static volatile bool g_btConnected = false;
static bool g_debugAlive = false;     // default OFF (clean terminal)
static uint32_t g_lastRxMs = 0;       // last received byte time (ms)

static inline void setHighDrive(uint8_t pin) {
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

static inline bool axisFromChar(char c, int *out) {
  if (c == 'A') { *out = AX_A; return true; }
  if (c == 'B') { *out = AX_B; return true; }
  if (c == 'C') { *out = AX_C; return true; }
  return false;
}

static void clearMovePlan(int ax) {
  g_moveRem[ax] = 0;
  g_moveTotal[ax] = 0;
  moveTargetPer_us[ax] = 0;
  movePer_us[ax] = 0;
  moveNext_us[ax] = 0;
  moveUnlimited[ax] = false;
}

static inline void clearMovePlans() {
  for (int ax = 0; ax < AX_COUNT; ++ax) clearMovePlan(ax);
}

static uint32_t profiledMovePeriodUs(
  uint32_t requestedPer_us,
  long total,
  long remaining,
  bool smoothProfile,
  bool unlimited
) {
  const float maxRate = unlimited ? TEST_MAX_RATE_STEPS_S : MOVE_MAX_RATE_STEPS_S;
  const uint32_t safeMinPeriod_us = (uint32_t)ceilf(1000000.0f / maxRate);
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
  int ax,
  bool fwd,
  long steps,
  long delay_us,
  bool smoothProfile,
  bool unlimited
) {
  if (steps <= 0) return;
  if (delay_us < 0) delay_us = 0;
  if (ax < 0 || ax >= AX_COUNT) return;

  uint32_t per_us = (uint32_t)(max(1L, delay_us + (long)STEP_PULSE_US));
  uint32_t now = micros();

  digitalWrite(DIR_PIN[ax], fwd ? HIGH : LOW);
  g_moveRem[ax] = steps;
  g_moveTotal[ax] = steps;
  moveSmooth[ax] = smoothProfile;
  moveUnlimited[ax] = unlimited;
  moveTargetPer_us[ax] = per_us;
  movePer_us[ax] = profiledMovePeriodUs(per_us, steps, steps, smoothProfile, unlimited);
  moveNext_us[ax] = now;
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
  for (int ax = 0; ax < AX_COUNT; ++ax) {
    pinMode(STEP_PIN[ax], OUTPUT);
    pinMode(DIR_PIN[ax], OUTPUT);
    // Increase output drive strength for long traces / noisy loads.
    setHighDrive(STEP_PIN[ax]);
    setHighDrive(DIR_PIN[ax]);
    digitalWrite(STEP_PIN[ax], LOW);
    digitalWrite(DIR_PIN[ax], LOW);
  }

  pinMode(EN_PIN, OUTPUT);
  setHighDrive(EN_PIN);

  setEnable(false);
  delay(2);

  SerialBT.register_callback(btCallback);
  SerialBT.begin("AstroPanoptes-ESP32");
}

// Shared by MOVE and TEST: they differ only in whether the observing speed
// clamp applies.
static void handleMoveCommand(bool unlimited) {
  char *ax = strtok(NULL, " ");
  char *dr = strtok(NULL, " ");
  char *st = strtok(NULL, " ");
  char *du = strtok(NULL, " ");
  char *pf = strtok(NULL, " ");

  if (!ax || !dr || !st || !du) { btPrintCRLF("ERR"); return; }
  int axis = -1;
  if (!axisFromChar(ax[0], &axis)) { btPrintCRLF("ERR"); return; }

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

  startMoveNonBlocking(axis, fwd, steps, delay_us, smoothProfile, unlimited);
  btPrintCRLF("OK");
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

  // --- move scheduler (all axes) ---
  if (g_enabled) {
    uint32_t now = micros();
    for (int ax = 0; ax < AX_COUNT; ++ax) {
      if (g_moveRem[ax] > 0 && movePer_us[ax] > 0 && (int32_t)(now - moveNext_us[ax]) >= 0) {
        const uint32_t stepStarted_us = micros();
        pulseStep(STEP_PIN[ax]);
        g_moveRem[ax] -= 1;
        if (g_moveRem[ax] <= 0) {
          g_moveRem[ax] = 0;
          movePer_us[ax] = 0;
          moveUnlimited[ax] = false;
        } else {
          movePer_us[ax] = profiledMovePeriodUs(
            moveTargetPer_us[ax], g_moveTotal[ax], g_moveRem[ax],
            moveSmooth[ax], moveUnlimited[ax]
          );
          // Schedule from the pulse start so movePer_us is the true
          // step-to-step period. Using micros() after the pulse silently added
          // STEP_PULSE_US a second time and made host duration estimates drift.
          moveNext_us[ax] = stepStarted_us + movePer_us[ax];
        }
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
    // STOP alone halts everything; STOP <axis> halts just that one, so the
    // focuser can be interrupted without aborting a slew (and vice versa).
    char *a = strtok(NULL, " ");
    if (a) {
      int axis = -1;
      if (!axisFromChar(a[0], &axis)) { btPrintCRLF("ERR"); return; }
      clearMovePlan(axis);
    } else {
      clearMovePlans();
    }
    btPrintCRLF("OK");
    return;
  }

  if (!strcmp(tok, "MS")) {
    char *a1 = strtok(NULL, " ");
    if (!a1) { btPrintCRLF("ERR"); return; }

    uint16_t ms = 0;
    if (!strcmp(a1, "AZ") || !strcmp(a1, "ALT") || !strcmp(a1, "FOCUS")) {
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
    handleMoveCommand(false);
    return;
  }

  if (!strcmp(tok, "TEST")) {
    // Bench testing for motors with a different reduction: same scheduler,
    // same profiles, but without the telescope's speed clamp.
    handleMoveCommand(true);
    return;
  }

  if (!strcmp(tok, "STATUS")) {
    String s;
    s.reserve(180);
    s += "EN=";    s += (g_enabled ? "1" : "0");
    s += " MS=";   s += String((uint16_t)FIXED_MICROSTEPS);
    s += " MSFIXED=1";
    s += " MOVEPROFILES=1";
    s += " AXES=3";
    s += " FOCUS=1";
    s += " TESTMODE=1";
    s += " MOVE=";
    for (int ax = 0; ax < AX_COUNT; ++ax) {
      if (ax) s += ",";
      s += String((long)g_moveRem[ax]);
    }
    s += " PROFILE=";
    for (int ax = 0; ax < AX_COUNT; ++ax) {
      if (ax) s += ",";
      s += (moveSmooth[ax] ? "SMOOTH" : "DIRECT");
    }
    s += " BT=";   s += (g_btConnected ? "1" : "0");
    s += " DBG=";  s += (g_debugAlive ? "1" : "0");
    replyBT(s);
    return;
  }

  btPrintCRLF("ERR");
}
