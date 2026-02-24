// ============================================================
// Smooth dual-stepper controller for ESP32 (TMC2209 STEP/DIR)
// + Microstepping via MS pins (standalone, COMMON MS1/MS2)
// + Bluetooth SPP serial (BluetoothSerial)
//
// Commands (newline-terminated):
//   PING
//   ENABLE 0|1
//   MS <8|16|32|64>                 (set both axes; common MS pins anyway)
//   MS AZ <8|16|32|64>              (accepted; same as MS <ms> because shared pins)
//   MS ALT <8|16|32|64>             (accepted; same as MS <ms> because shared pins)
//   STOP                            (cancela planes MOVE)
//   MOVE A|B FWD|REV steps delay_us (non-blocking per-axis microstep move)
//   STATUS
//
// Pinout (YOUR PCB):
//   EN (common):  GPIO21  (LOW=enabled)
//   MS1(common):  GPIO19
//   MS2(common):  GPIO18
//   AZ:  STEP=GPIO33  DIR=GPIO25
//   ALT: STEP=GPIO26  DIR=GPIO27
//
// Microstep table (as per your drivers):
//   (LOW / LOW)  -> 1/8
//   (HIGH/HIGH)  -> 1/16
//   (HIGH/LOW)   -> 1/32
//   (LOW /HIGH)  -> 1/64
// ============================================================

#include <Arduino.h>
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// ---------------- Pin mapping (ESP32) ----------------
static const uint8_t STEP_A = 33;  // AZ
static const uint8_t DIR_A  = 25;

static const uint8_t STEP_B = 26;  // ALT
static const uint8_t DIR_B  = 27;

static const uint8_t EN_PIN = 21;  // LOW = enabled

// Common microstep pins (shared for both drivers)
static const uint8_t MS1_PIN = 19;
static const uint8_t MS2_PIN = 18;

// Pulse width for STEP pin (us) - ESP32 can do short pulses, keep conservative
static const uint16_t STEP_PULSE_US = 3;

static bool g_enabled = false;

// MOVE scheduler (non-blocking, one plan per axis)
static volatile long g_moveRemA = 0;
static volatile long g_moveRemB = 0;
static uint32_t movePerA_us  = 0;
static uint32_t movePerB_us  = 0;
static uint32_t moveNextA_us = 0;
static uint32_t moveNextB_us = 0;

// Current microsteps per full-step (8/16/32/64)
static volatile uint16_t g_ms_common = 64;

// ---------- small helpers ----------
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

  // Preserve "delay_us between pulses" semantics while accounting for pulse width.
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

// ---------- line reader that works for BOTH USB Serial and Bluetooth ----------
static bool readLineAny(String &outLine) {
  static String line;

  auto pump = [&](Stream &s) -> bool {
    while (s.available()) {
      char c = (char)s.read();
      if (c == '\r') continue;
      if (c == '\n') {
        outLine = line;
        line = "";
        outLine.trim();
        return true;
      }
      line += c;
      if (line.length() > 200) { // safety
        line = "";
        outLine = "";
        return true; // treat as "got a line" but empty => ignored by parser
      }
    }
    return false;
  };

  // Prefer BT if connected, but accept USB serial too
  if (SerialBT.hasClient()) {
    if (pump(SerialBT)) return true;
  }
  if (pump(Serial)) return true;

  return false;
}

static void replyBoth(const String &s) {
  Serial.println(s);
  if (SerialBT.hasClient()) SerialBT.println(s);
}

void setup() {
  // IO init
  pinMode(STEP_A, OUTPUT);
  pinMode(DIR_A, OUTPUT);
  pinMode(STEP_B, OUTPUT);
  pinMode(DIR_B, OUTPUT);

  pinMode(EN_PIN, OUTPUT);
  pinMode(MS1_PIN, OUTPUT);
  pinMode(MS2_PIN, OUTPUT);

  digitalWrite(STEP_A, LOW);
  digitalWrite(STEP_B, LOW);
  digitalWrite(DIR_A, LOW);
  digitalWrite(DIR_B, LOW);

  setEnable(false);

  // default microsteps
  setMSPinsCommon(64);

  // USB serial
  Serial.begin(115200);

  // Bluetooth SPP
  // Device name shown in phone/PC BT scan:
  SerialBT.begin("AstroPanoptes-ESP32");  // change if you want

  replyBoth("READY");
}

void loop() {
  // ---- MOVE plans (parallel by axis) ----
  if (g_enabled) {
    uint32_t now = micros();

    if (g_moveRemA > 0 && movePerA_us > 0 && (int32_t)(now - moveNextA_us) >= 0) {
      pulseStep(STEP_A);
      g_moveRemA -= 1;
      moveNextA_us += movePerA_us;
      if (g_moveRemA <= 0) {
        g_moveRemA = 0;
        movePerA_us = 0;
      }
    }

    if (g_moveRemB > 0 && movePerB_us > 0 && (int32_t)(now - moveNextB_us) >= 0) {
      pulseStep(STEP_B);
      g_moveRemB -= 1;
      moveNextB_us += movePerB_us;
      if (g_moveRemB <= 0) {
        g_moveRemB = 0;
        movePerB_us = 0;
      }
    }
  }

  // ---- Command handling ----
  String cmd;
  if (!readLineAny(cmd)) return;
  if (cmd.length() == 0) return;

  char buf[220];
  cmd.toCharArray(buf, sizeof(buf));

  char *tok = strtok(buf, " ");
  if (!tok) return;

  if (!strcmp(tok, "PING")) {
    replyBoth("READY");
    return;
  }

  if (!strcmp(tok, "ENABLE")) {
    char *a = strtok(NULL, " ");
    int on = a ? atoi(a) : 0;
    setEnable(on != 0);
    if (!g_enabled) clearMovePlans();
    replyBoth("OK");
    return;
  }

  if (!strcmp(tok, "STOP")) {
    clearMovePlans();
    replyBoth("OK");
    return;
  }

  if (!strcmp(tok, "MS")) {
    // MS <ms> | MS AZ <ms> | MS ALT <ms>
    // NOTE: because MS pins are COMMON, all forms map to the same setting.
    char *a1 = strtok(NULL, " ");
    if (!a1) { replyBoth("ERR"); return; }

    uint16_t ms = 0;

    if (!strcmp(a1, "AZ") || !strcmp(a1, "ALT")) {
      char *a2 = strtok(NULL, " ");
      ms = a2 ? (uint16_t)atoi(a2) : 0;
    } else {
      ms = (uint16_t)atoi(a1);
    }

    if (!setMSPinsCommon(ms)) { replyBoth("ERR"); return; }
    replyBoth(String("OK MS ") + String(g_ms_common));
    return;
  }

  if (!strcmp(tok, "MOVE")) {
    // MOVE A|B FWD|REV steps delay_us
    char *ax = strtok(NULL, " ");
    char *dr = strtok(NULL, " ");
    char *st = strtok(NULL, " ");
    char *du = strtok(NULL, " ");

    if (!ax || !dr || !st || !du) { replyBoth("ERR"); return; }

    char axis = ax[0];
    if (!(axis == 'A' || axis == 'B')) { replyBoth("ERR"); return; }

    bool fwd = (!strcmp(dr, "FWD"));
    if (!(fwd || !strcmp(dr, "REV"))) { replyBoth("ERR"); return; }

    long steps = atol(st);
    long delay_us = atol(du);
    if (delay_us < 0) delay_us = 0;

    startMoveNonBlocking(axis, fwd, steps, delay_us);
    replyBoth("OK");
    return;
  }

  if (!strcmp(tok, "STATUS")) {
    String s;
    s.reserve(80);
    s += "EN=";
    s += (g_enabled ? "1" : "0");
    s += " MS=";
    s += String(g_ms_common);
    s += " MOVE=";
    s += String((long)g_moveRemA);
    s += ",";
    s += String((long)g_moveRemB);
    replyBoth(s);
    return;
  }

  replyBoth("ERR");
}