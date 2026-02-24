// ============================================================
// Dual-stepper controller for ESP32 (TMC2209 STEP/DIR)
// + Microstepping via COMMON MS pins (standalone)
// + Bluetooth Classic SPP via BluetoothSerial
//
// Commands (newline-terminated; CR/LF/CRLF accepted):
//   PING
//   ENABLE 0|1
//   MS <8|16|32|64>                 (common -> affects both drivers)
//   MS AZ <8|16|32|64>              (accepted; same as MS <ms>)
//   MS ALT <8|16|32|64>             (accepted; same as MS <ms>)
//   STOP
//   MOVE A|B FWD|REV steps delay_us
//   STATUS                              (adds PINMS=<ms1>,<ms2>)
//   DEBUG 0|1                       (toggle ALIVE heartbeat)
//
// YOUR PCB pinout (ESP32 30-pin):
//   EN  (common): GPIO21  (LOW=enabled)
//   MS1 (common): GPIO19
//   MS2 (common): GPIO18
//   AZ:  STEP=GPIO33  DIR=GPIO25
//   ALT: STEP=GPIO26  DIR=GPIO27
//
// Microstep table (your driver board mapping):
//   LOW/LOW   -> 1/8
//   HIGH/HIGH -> 1/16
//   HIGH/LOW  -> 1/32
//   LOW/HIGH  -> 1/64
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
static const uint8_t MS1_PIN = 19; // common
static const uint8_t MS2_PIN = 18; // common

static const uint16_t STEP_PULSE_US = 3;

static bool g_enabled = false;

// MOVE scheduler (non-blocking)
static volatile long g_moveRemA = 0;
static volatile long g_moveRemB = 0;
static uint32_t movePerA_us  = 0;
static uint32_t movePerB_us  = 0;
static uint32_t moveNextA_us = 0;
static uint32_t moveNextB_us = 0;

static volatile uint16_t g_ms_common = 64;

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

static bool setMSPinsCommon(uint16_t ms) {
  switch (ms) {
    case 8:
      digitalWrite(MS1_PIN, LOW);
      digitalWrite(MS2_PIN, LOW);
      g_ms_common = 8;
      return true;
    case 16:
      digitalWrite(MS1_PIN, HIGH);
      digitalWrite(MS2_PIN, HIGH);
      g_ms_common = 16;
      return true;
    case 32:
      digitalWrite(MS1_PIN, HIGH);
      digitalWrite(MS2_PIN, LOW);
      g_ms_common = 32;
      return true;
    case 64:
      digitalWrite(MS1_PIN, LOW);
      digitalWrite(MS2_PIN, HIGH);
      g_ms_common = 64;
      return true;
    default:
      return false;
  }
}

// Apply common MS pins. If relatchEnabled=true and drivers are enabled,
// briefly disable -> apply MS -> enable so external drivers can re-sample.
static bool applyMSCommon(uint16_t ms, bool relatchEnabled) {
  bool wasEnabled = g_enabled;
  if (relatchEnabled && wasEnabled) {
    setEnable(false);
    delay(2);
  }

  if (!setMSPinsCommon(ms)) {
    // restore previous enable state if needed
    if (relatchEnabled && wasEnabled) {
      setEnable(true);
      delay(2);
    }
    return false;
  }

  // Let MS levels settle on board traces/driver input.
  delayMicroseconds(300);

  if (relatchEnabled && wasEnabled) {
    setEnable(true);
    delay(2);
  }
  return true;
}

static inline void clearMovePlans() {
  g_moveRemA = 0;
  g_moveRemB = 0;
  movePerA_us = 0;
  movePerB_us = 0;
  moveNextA_us = 0;
  moveNextB_us = 0;
}

static void startMoveNonBlocking(char axis, bool fwd, long steps, long delay_us) {
  if (steps <= 0) return;
  if (delay_us < 0) delay_us = 0;

  uint32_t per_us = (uint32_t)(max(1L, delay_us + (long)STEP_PULSE_US));
  uint32_t now = micros();

  if (axis == 'A') {
    digitalWrite(DIR_A, fwd ? HIGH : LOW);
    g_moveRemA = steps;
    movePerA_us = per_us;
    moveNextA_us = now;
  } else { // 'B'
    digitalWrite(DIR_B, fwd ? HIGH : LOW);
    g_moveRemB = steps;
    movePerB_us = per_us;
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
  pinMode(MS1_PIN, OUTPUT);
  pinMode(MS2_PIN, OUTPUT);

  setHighDrive(STEP_A);
  setHighDrive(DIR_A);
  setHighDrive(STEP_B);
  setHighDrive(DIR_B);
  setHighDrive(EN_PIN);
  setHighDrive(MS1_PIN);
  setHighDrive(MS2_PIN);

  digitalWrite(STEP_A, LOW);
  digitalWrite(STEP_B, LOW);
  digitalWrite(DIR_A, LOW);
  digitalWrite(DIR_B, LOW);

  setEnable(false);
  delay(2);
  applyMSCommon(64, false);
  // Second pass helps boards that need extra settle time at boot.
  delay(2);
  applyMSCommon(64, false);

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
      pulseStep(STEP_A);
      g_moveRemA -= 1;
      moveNextA_us += movePerA_us;
      if (g_moveRemA <= 0) { g_moveRemA = 0; movePerA_us = 0; }
    }

    if (g_moveRemB > 0 && movePerB_us > 0 && (int32_t)(now - moveNextB_us) >= 0) {
      pulseStep(STEP_B);
      g_moveRemB -= 1;
      moveNextB_us += movePerB_us;
      if (g_moveRemB <= 0) { g_moveRemB = 0; movePerB_us = 0; }
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
      // Re-apply current MS before enabling to avoid stale latched value.
      if (!applyMSCommon((uint16_t)g_ms_common, wasEnabled)) { btPrintCRLF("ERR"); return; }
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

    if (!applyMSCommon(ms, true)) { btPrintCRLF("ERR"); return; }
    replyBT(String("OK MS ") + String((uint16_t)g_ms_common));
    return;
  }

  if (!strcmp(tok, "MOVE")) {
    char *ax = strtok(NULL, " ");
    char *dr = strtok(NULL, " ");
    char *st = strtok(NULL, " ");
    char *du = strtok(NULL, " ");

    if (!ax || !dr || !st || !du) { btPrintCRLF("ERR"); return; }
    char axis = ax[0];
    if (!(axis == 'A' || axis == 'B')) { btPrintCRLF("ERR"); return; }

    bool fwd = (!strcmp(dr, "FWD"));
    if (!(fwd || !strcmp(dr, "REV"))) { btPrintCRLF("ERR"); return; }

    long steps = atol(st);
    long delay_us = atol(du);
    if (delay_us < 0) delay_us = 0;

    startMoveNonBlocking(axis, fwd, steps, delay_us);
    btPrintCRLF("OK");
    return;
  }

  if (!strcmp(tok, "STATUS")) {
    String s;
    s.reserve(110);
    s += "EN=";    s += (g_enabled ? "1" : "0");
    s += " MS=";   s += String((uint16_t)g_ms_common);
    s += " PINMS=";
    s += String(digitalRead(MS1_PIN) ? 1 : 0);
    s += ",";
    s += String(digitalRead(MS2_PIN) ? 1 : 0);
    s += " MOVE="; s += String((long)g_moveRemA); s += ","; s += String((long)g_moveRemB);
    s += " BT=";   s += (g_btConnected ? "1" : "0");
    s += " DBG=";  s += (g_debugAlive ? "1" : "0");
    replyBT(s);
    return;
  }

  btPrintCRLF("ERR");
}
