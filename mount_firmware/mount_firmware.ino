// ============================================================
// Smooth dual-stepper controller for Arduino UNO (TMC2209 STEP/DIR)
// + Microstepping via MS pins (standalone)
// Commands (newline-terminated):
//   PING
//   ENABLE 0|1
//   MS <8|16|32|64>                 (set both axes)
//   MS AZ <8|16|32|64>
//   MS ALT <8|16|32|64>
//   STOP                            (cancela planes MOVE)
//   MOVE A|B FWD|REV steps delay_us (non-blocking per-axis microstep move)
//   STATUS
//
// Pinout:
//   AZ:  STEP=2 DIR=4  MS1=8  MS2=9
//   ALT: STEP=3 DIR=5  MS1=10 MS2=11
//   EN:  6 (LOW=enabled)
//
// Microstep table (as per user's drivers):
//   (LOW / LOW)  -> 1/8
//   (HIGH/HIGH)  -> 1/16
//   (HIGH/LOW)   -> 1/32
//   (LOW /HIGH)  -> 1/64
// ============================================================

#include <Arduino.h>

const uint8_t STEP_A = 2;   // AZ
const uint8_t DIR_A  = 4;
const uint8_t STEP_B = 3;   // ALT
const uint8_t DIR_B  = 5;

const uint8_t EN_PIN = 6;   // LOW = enabled

// Microstep select pins
const uint8_t AZ_MS1 = 8;
const uint8_t AZ_MS2 = 9;
const uint8_t ALT_MS1 = 10;
const uint8_t ALT_MS2 = 11;

// pulse width for STEP pin (us)
const uint16_t STEP_PULSE_US = 3;

bool g_enabled = false;

// MOVE scheduler (non-blocking, one plan per axis)
volatile long g_moveRemA = 0;
volatile long g_moveRemB = 0;
uint32_t movePerA_us = 0;
uint32_t movePerB_us = 0;
uint32_t moveNextA_us = 0;
uint32_t moveNextB_us = 0;

// Current microsteps per full-step (8/16/32/64)
volatile uint16_t g_ms_az = 64;
volatile uint16_t g_ms_alt = 64;

static inline void pulseStep(uint8_t pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(pin, LOW);
}

static inline void setEnable(bool on) {
  g_enabled = on;
  digitalWrite(EN_PIN, on ? LOW : HIGH);
}

static bool setMSPins(uint8_t ms1_pin, uint8_t ms2_pin, uint16_t ms) {
  // Mapping:
  // 1/8  -> LOW, LOW
  // 1/16 -> HIGH, HIGH
  // 1/32 -> HIGH, LOW
  // 1/64 -> LOW, HIGH
  switch (ms) {
    case 8:
      digitalWrite(ms1_pin, LOW);
      digitalWrite(ms2_pin, LOW);
      return true;
    case 16:
      digitalWrite(ms1_pin, HIGH);
      digitalWrite(ms2_pin, HIGH);
      return true;
    case 32:
      digitalWrite(ms1_pin, HIGH);
      digitalWrite(ms2_pin, LOW);
      return true;
    case 64:
      digitalWrite(ms1_pin, LOW);
      digitalWrite(ms2_pin, HIGH);
      return true;
    default:
      return false;
  }
}

static bool setMicrostepsAZ(uint16_t ms) {
  if (!setMSPins(AZ_MS1, AZ_MS2, ms)) return false;
  g_ms_az = ms;
  return true;
}

static bool setMicrostepsALT(uint16_t ms) {
  if (!setMSPins(ALT_MS1, ALT_MS2, ms)) return false;
  g_ms_alt = ms;
  return true;
}

void clearMovePlans() {
  g_moveRemA = 0;
  g_moveRemB = 0;
  movePerA_us = 0;
  movePerB_us = 0;
  moveNextA_us = 0;
  moveNextB_us = 0;
}

void startMoveNonBlocking(char axis, bool fwd, long steps, long delay_us) {
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
  } else {
    digitalWrite(DIR_B, fwd ? HIGH : LOW);
    g_moveRemB = steps;
    movePerB_us = per_us;
    moveNextB_us = now;
  }
}

String readLine() {
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      String out = line;
      line = "";
      out.trim();
      return out;
    }
    line += c;
    if (line.length() > 200) { // safety
      line = "";
      return "";
    }
  }
  return "";
}

void setup() {
  pinMode(STEP_A, OUTPUT);
  pinMode(DIR_A, OUTPUT);
  pinMode(STEP_B, OUTPUT);
  pinMode(DIR_B, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  pinMode(AZ_MS1, OUTPUT);
  pinMode(AZ_MS2, OUTPUT);
  pinMode(ALT_MS1, OUTPUT);
  pinMode(ALT_MS2, OUTPUT);

  digitalWrite(STEP_A, LOW);
  digitalWrite(STEP_B, LOW);
  digitalWrite(DIR_A, LOW);
  digitalWrite(DIR_B, LOW);

  setEnable(false);

  // default microsteps
  setMicrostepsAZ(64);
  setMicrostepsALT(64);

  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("READY");
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
  String cmd = readLine();
  if (cmd.length() == 0) return;

  char buf[220];
  cmd.toCharArray(buf, sizeof(buf));
  char *tok = strtok(buf, " ");
  if (!tok) return;

  if (!strcmp(tok, "PING")) {
    Serial.println("READY");
    return;
  }

  if (!strcmp(tok, "ENABLE")) {
    char *a = strtok(NULL, " ");
    int on = a ? atoi(a) : 0;
    setEnable(on != 0);
    if (!g_enabled) {
      clearMovePlans();
    }
    Serial.println("OK");
    return;
  }

  if (!strcmp(tok, "STOP")) {
    clearMovePlans();
    Serial.println("OK");
    return;
  }

  if (!strcmp(tok, "MS")) {
    // MS <ms> | MS AZ <ms> | MS ALT <ms>
    char *a1 = strtok(NULL, " ");
    if (!a1) { Serial.println("ERR"); return; }

    bool ok = false;

    if (!strcmp(a1, "AZ") || !strcmp(a1, "ALT")) {
      char *a2 = strtok(NULL, " ");
      uint16_t ms = a2 ? (uint16_t)atoi(a2) : 0;
      if (!strcmp(a1, "AZ")) ok = setMicrostepsAZ(ms);
      else                   ok = setMicrostepsALT(ms);
    } else {
      uint16_t ms = (uint16_t)atoi(a1);
      ok = setMicrostepsAZ(ms) && setMicrostepsALT(ms);
    }

    if (!ok) { Serial.println("ERR"); return; }
    Serial.print("OK MS ");
    Serial.print(g_ms_az);
    Serial.print(" ");
    Serial.println(g_ms_alt);
    return;
  }

  if (!strcmp(tok, "MOVE")) {
    // MOVE A|B FWD|REV steps delay_us
    char *ax = strtok(NULL, " ");
    char *dr = strtok(NULL, " ");
    char *st = strtok(NULL, " ");
    char *du = strtok(NULL, " ");

    if (!ax || !dr || !st || !du) { Serial.println("ERR"); return; }
    char axis = ax[0];
    bool fwd = (!strcmp(dr, "FWD"));

    long steps = atol(st);
    long delay_us = atol(du);
    if (delay_us < 0) delay_us = 0;

    startMoveNonBlocking(axis, fwd, steps, delay_us);

    Serial.println("OK");
    return;
  }

  if (!strcmp(tok, "STATUS")) {
    Serial.print("EN=");
    Serial.print(g_enabled ? 1 : 0);
    Serial.print(" MS=");
    Serial.print(g_ms_az);
    Serial.print(",");
    Serial.print(g_ms_alt);
    Serial.print(" MOVE=");
    Serial.print(g_moveRemA);
    Serial.print(",");
    Serial.println(g_moveRemB);
    return;
  }

  Serial.println("ERR");
}
