# mount_arduino.py
from __future__ import annotations

import json
import math
import os
import plistlib
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Dict

import serial
import serial.tools.list_ports

from ap_types import Axis, MountStatus, TrackingMode, TrackingStatus
from protocols import StatePublisherProtocol
from logging_utils import log_error, log_info
from workers import BaseWorker


FIXED_MICROSTEPS = 64
SUPPORTED_MICROSTEPS: tuple[int, ...] = (FIXED_MICROSTEPS,)

STEP_PULSE_US = 3
FIRMWARE_MOVE_MAX_RATE_STEPS_S = 12_000.0
FIRMWARE_MOVE_SMOOTH_START_RATE_STEPS_S = 400.0
FIRMWARE_MOVE_SMOOTH_MAX_ACCEL_STEPS_S2 = 4_000.0
_SMOOTHERSTEP_MAX_DERIVATIVE = 1.875


def normalize_move_profile(profile: str) -> str:
    value = str(profile or "smooth").strip().lower()
    if value not in {"smooth", "direct"}:
        raise ValueError("move profile must be 'smooth' or 'direct'")
    return value


def firmware_move_period_us(
    target_delay_us: int,
    total_steps: int,
    remaining_steps: int,
    profile: str = "smooth",
) -> int:
    """Mirror the selectable, rate-limited firmware MOVE profile."""
    move_profile = normalize_move_profile(profile)
    requested_period = max(1, int(target_delay_us) + int(STEP_PULSE_US))
    safe_min_period = int(math.ceil(1.0e6 / FIRMWARE_MOVE_MAX_RATE_STEPS_S))
    target_period = max(requested_period, safe_min_period)
    total = max(0, int(total_steps))
    remaining = max(0, int(remaining_steps))
    if total <= 0 or remaining <= 0 or move_profile == "direct":
        return target_period

    target_rate = 1.0e6 / float(target_period)
    start_rate = min(
        target_rate,
        float(FIRMWARE_MOVE_SMOOTH_START_RATE_STEPS_S),
    )
    if target_rate <= start_rate:
        return target_period

    completed = max(0, total - remaining)
    stopping_edge = max(0, remaining - 1)
    edge_steps = float(min(completed, stopping_edge))
    half_move_steps = max(1.0, float(total) / 2.0)
    ideal_ramp_steps = (
        (target_rate * target_rate - start_rate * start_rate)
        * _SMOOTHERSTEP_MAX_DERIVATIVE
        / (2.0 * FIRMWARE_MOVE_SMOOTH_MAX_ACCEL_STEPS_S2)
    )
    ramp_steps = max(1.0, min(half_move_steps, ideal_ramp_steps))
    if edge_steps >= ramp_steps:
        return target_period

    # Smootherstep gives zero acceleration at both ends. Interpolating rate²
    # makes the peak physical acceleration analytically bounded.
    peak_rate_sq = min(
        target_rate * target_rate,
        start_rate * start_rate
        + (
            2.0
            * FIRMWARE_MOVE_SMOOTH_MAX_ACCEL_STEPS_S2
            * ramp_steps
            / _SMOOTHERSTEP_MAX_DERIVATIVE
        ),
    )
    x = edge_steps / ramp_steps
    smoother = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    rate = math.sqrt(
        start_rate * start_rate
        + (peak_rate_sq - start_rate * start_rate) * smoother
    )
    if not (rate > 0.0):
        rate = start_rate
    return max(safe_min_period, int(round(1.0e6 / rate)))


def estimate_firmware_move_duration_s(
    steps: int,
    min_delay_us: int,
    profile: str = "smooth",
) -> float:
    """Estimate one firmware MOVE using the selected profile."""
    total = max(0, int(steps))
    if total <= 0:
        return 0.0
    move_profile = normalize_move_profile(profile)
    if move_profile == "direct":
        period_sum_us = total * firmware_move_period_us(
            min_delay_us, total, total, profile=move_profile
        )
    else:
        requested_period = max(1, int(min_delay_us) + int(STEP_PULSE_US))
        safe_min_period = int(
            math.ceil(1.0e6 / FIRMWARE_MOVE_MAX_RATE_STEPS_S)
        )
        target_period = max(requested_period, safe_min_period)
        target_rate = 1.0e6 / float(target_period)
        start_rate = min(
            target_rate,
            float(FIRMWARE_MOVE_SMOOTH_START_RATE_STEPS_S),
        )
        ideal_ramp_steps = max(
            1.0,
            (
                (target_rate * target_rate - start_rate * start_rate)
                * _SMOOTHERSTEP_MAX_DERIVATIVE
                / (2.0 * FIRMWARE_MOVE_SMOOTH_MAX_ACCEL_STEPS_S2)
            ),
        )
        ramp_count = min(total // 2, int(math.ceil(ideal_ramp_steps)))
        edge_sum_us = sum(
            firmware_move_period_us(
                min_delay_us,
                total,
                total - completed,
                profile=move_profile,
            )
            for completed in range(ramp_count)
        )
        center_count = total - (2 * ramp_count)
        center_period = firmware_move_period_us(
            min_delay_us,
            total,
            total - ramp_count,
            profile=move_profile,
        )
        period_sum_us = (2 * edge_sum_us) + (center_count * center_period)
    return float(period_sum_us / 1.0e6)


# =========================
# Utilities
# =========================

def list_serial_ports() -> list[str]:
    ports: list[str] = []
    for p in serial.tools.list_ports.comports():
        dev = str(getattr(p, "device", "") or "").strip()
        if dev:
            ports.append(dev)
    return sorted(set(ports))


def _safe_lower(s: str) -> str:
    try:
        return (s or "").lower()
    except Exception as exc:
        log_error(None, "Mount: failed to normalize string", exc, throttle_s=10.0, throttle_key="mount_safe_lower")
        return ""


def _port_hint_text(port_info: object) -> str:
    fields = [
        getattr(port_info, "device", ""),
        getattr(port_info, "description", ""),
        getattr(port_info, "manufacturer", ""),
        getattr(port_info, "product", ""),
        getattr(port_info, "interface", ""),
        getattr(port_info, "hwid", ""),
    ]
    return " ".join(str(v or "") for v in fields).strip().lower()


def _is_auto_port_value(port: str) -> bool:
    return (str(port or "").strip().upper() in {"", "AUTO"})


def resolve_mount_port(port: str) -> str:
    """
    Resolves the serial port path.
    - explicit path -> use as-is
    - AUTO/empty    -> pick best candidate (prefers AstroPanoptes-ESP32 over generic)
    """
    requested = str(port or "").strip()
    if not _is_auto_port_value(requested):
        return requested

    candidates: list[tuple[int, str]] = []
    for p in serial.tools.list_ports.comports():
        dev = str(getattr(p, "device", "") or "").strip()
        if not dev:
            continue
        hint = _port_hint_text(p)
        score = 0
        if "astropanoptes-esp32" in hint:
            score += 100
        if "astropanoptes" in hint:
            score += 70
        if "esp32" in hint:
            score += 40
        # Never treat macOS' generic incoming RFCOMM endpoint, unrelated
        # headsets, or debug ports as the telescope. AUTO must wait for a
        # positively identified AstroPanoptes port instead of guessing.
        if score > 0:
            candidates.append((score, dev))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_dev = candidates[0]
    return best_dev if best_score > 0 else ""


def resolve_common_microsteps(az_div: int, alt_div: int, *, default_ms: int = 64) -> int:
    """Return the hardware-wired microstep divisor (always 1/64)."""
    _ = az_div, alt_div, default_ms
    return int(FIXED_MICROSTEPS)


def _axis_to_fw(axis: Axis) -> str:
    # Firmware: A=AZ, B=ALT
    return "A" if axis == Axis.AZ else "B"


def _dir_to_fw(direction: int) -> str:
    # Firmware: FWD/REV
    return "FWD" if direction >= 0 else "REV"


# =========================
# Config + Controller (protocol exact)
# =========================

@dataclass
class ArduinoConfig:
    port: str = "AUTO"
    baud: int = 115200

    # timeouts serial
    timeout_s: float = 0.10
    write_timeout_s: float = 0.25

    # ESP32 Bluetooth SPP usually does not need long waits on open.
    connect_sleep_s: float = 0.60

    # send behavior
    flush_on_send: bool = True
    reset_input_on_send: bool = False

    # handshake on connect
    handshake_attempts: int = 4
    open_attempts: int = 3
    open_retry_delay_s: float = 1.0

    # reconnect behavior
    allow_reconnect: bool = True

    # macOS Bluetooth stale-session workaround (uses `blueutil`)
    bt_forget_before_connect: bool = True
    bt_device_name: str = "AstroPanoptes-ESP32"
    bt_forget_retry_s: float = 0.80
    bt_pair_after_forget: bool = True
    bt_inquiry_s: float = 6.0
    bt_serial_port_wait_s: float = 20.0


class ArduinoController:
    """
    Controlador thread-safe para ESP32 (Bluetooth SPP exposed as serial port).

    Protocolo (newline):
      PING                  -> READY
      ENABLE 0|1            -> OK
      STOP                  -> OK
      MS 64                 -> OK MS_FIXED 64 (legacy no-op)
      MS AZ|ALT 64          -> OK MS_FIXED 64 (legacy no-op)
      MOVE A|B FWD|REV steps delay_us -> OK
      STATUS                -> EN=... MS=64 MSFIXED=1 MOVE=... BT=... DBG=...
    """

    def __init__(self, cfg: ArduinoConfig):
        self.cfg = cfg
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._move_profiles_supported: Optional[bool] = None

    def _blueutil_path(self) -> str:
        found = str(shutil.which("blueutil") or "")
        if found:
            return found

        # Finder-launched .app bundles do not normally inherit Homebrew's PATH.
        for candidate in ("/opt/homebrew/bin/blueutil", "/usr/local/bin/blueutil"):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return ""

    def _blueutil_run(self, *args: str, timeout_s: float = 10.0) -> tuple[int, str, str]:
        bin_path = self._blueutil_path()
        if not bin_path:
            return (127, "", "blueutil not found")
        try:
            p = subprocess.run(
                [bin_path, *args],
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
            return (int(p.returncode), str(p.stdout or "").strip(), str(p.stderr or "").strip())
        except Exception as exc:
            return (1, "", str(exc))

    @staticmethod
    def _blueutil_device_id(row: object) -> str:
        if not isinstance(row, dict):
            return ""
        return str(row.get("address", "") or row.get("id", "") or "").strip()

    def _blueutil_find_device_in(self, *args: str) -> str:
        name = str(self.cfg.bt_device_name or "").strip()
        if not name:
            return ""

        timeout_s = 15.0
        if "--inquiry" in args:
            try:
                inquiry_index = args.index("--inquiry")
                inquiry_s = float(args[inquiry_index + 1])
            except (ValueError, IndexError, TypeError):
                inquiry_s = float(self.cfg.bt_inquiry_s)
            # blueutil's advertised inquiry duration excludes the later name
            # lookup pass, which can add several seconds on macOS.
            timeout_s = max(20.0, inquiry_s + 20.0)

        rc, out, err = self._blueutil_run(*args, "--format", "json", timeout_s=timeout_s)
        if rc != 0 or not out:
            if err:
                log_info(
                    None,
                    f"Mount: BT device lookup hint ({err})",
                    throttle_s=2.0,
                    throttle_key="mount_blueutil_lookup_hint",
                )
            return ""

        try:
            rows = json.loads(out)
        except Exception as exc:
            log_error(
                None,
                "Mount: failed to parse blueutil device JSON",
                exc,
                throttle_s=10.0,
                throttle_key="mount_blueutil_json",
            )
            return ""

        if not isinstance(rows, list):
            return ""
        wanted = name.casefold()
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_name = str(row.get("name", "") or "").strip()
            if candidate_name.casefold() == wanted:
                return self._blueutil_device_id(row)
        return ""

    def _blueutil_get_device_id(self) -> str:
        """
        Returns BT device ID (address if possible) for the configured mount device name.
        """
        name = str(self.cfg.bt_device_name or "").strip()
        if not name:
            return ""

        if re.fullmatch(r"[0-9A-Fa-f]{2}(?:[:-]?[0-9A-Fa-f]{2}){5}", name):
            return name

        device_id = self._blueutil_find_device_in("--paired")
        if device_id:
            return device_id

        device_id = self._macos_persistent_device_id()
        if device_id:
            return device_id

        # Once macOS has forgotten the device, name lookup only works after an
        # inquiry. Keep the discovered address because pairing by name is not
        # reliable for unpaired devices on recent macOS releases.
        inquiry_s = max(1, int(round(float(self.cfg.bt_inquiry_s))))
        return self._blueutil_find_device_in("--inquiry", str(inquiry_s))

    def _macos_persistent_device_id(self) -> str:
        """Recover an RFCOMM device address kept by macOS after unpairing."""
        name = str(self.cfg.bt_device_name or "").strip().casefold()
        if not name:
            return ""

        plist_path = "/Library/Preferences/com.apple.Bluetooth.plist"
        try:
            with open(plist_path, "rb") as stream:
                preferences = plistlib.load(stream)
        except (OSError, plistlib.InvalidFileException):
            return ""

        ports = preferences.get("PersistentPorts", {})
        if not isinstance(ports, dict):
            return ""
        for address, details in ports.items():
            if not isinstance(details, dict):
                continue
            bsd_name = str(details.get("BSDName", "") or "").strip().casefold()
            if bsd_name == name:
                return str(address or "").strip()
        return ""

    def _refresh_bt_pairing(self) -> tuple[bool, str]:
        """
        macOS workaround:
        disconnect + unpair + pair + connect via blueutil to avoid stale RFCOMM sessions.
        """
        if not bool(self.cfg.bt_forget_before_connect):
            return (True, "")
        if not self._blueutil_path():
            return (False, "blueutil no encontrado; instala con 'brew install blueutil'")
        device_id = self._blueutil_get_device_id()
        if not device_id:
            return (False, f"dispositivo Bluetooth '{self.cfg.bt_device_name}' no encontrado")

        rc_disc, _o_disc, e_disc = self._blueutil_run("--disconnect", device_id, timeout_s=4.0)
        if rc_disc != 0 and e_disc:
            log_info(None, f"Mount: BT disconnect hint ({e_disc})", throttle_s=2.0, throttle_key="mount_bt_disconnect_hint")
        else:
            self._blueutil_run("--wait-disconnect", device_id, "5", timeout_s=6.0)

        rc_unpair, _o_unpair, e_unpair = self._blueutil_run("--unpair", device_id, timeout_s=8.0)
        if rc_unpair != 0 and e_unpair:
            log_info(None, f"Mount: BT unpair hint ({e_unpair})", throttle_s=2.0, throttle_key="mount_bt_unpair_hint")

        if float(self.cfg.bt_forget_retry_s) > 0.0:
            time.sleep(float(self.cfg.bt_forget_retry_s))

        if bool(self.cfg.bt_pair_after_forget):
            rc_pair, _o_pair, e_pair = self._blueutil_run("--pair", device_id, timeout_s=30.0)
            if rc_pair != 0:
                message = e_pair or "pair failed"
                log_info(None, f"Mount: BT pair hint ({message})", throttle_s=2.0, throttle_key="mount_bt_pair_hint")
                return (False, f"no se pudo emparejar la montura: {message}")

        rc_conn, _o_conn, e_conn = self._blueutil_run("--connect", device_id, timeout_s=20.0)
        if rc_conn != 0:
            message = e_conn or "connect failed"
            log_info(None, f"Mount: BT connect hint ({message})", throttle_s=2.0, throttle_key="mount_bt_connect_hint")
            return (False, f"no se pudo conectar la montura por Bluetooth: {message}")

        rc_wait, _o_wait, e_wait = self._blueutil_run("--wait-connect", device_id, "15", timeout_s=16.0)
        if rc_wait != 0:
            message = e_wait or "connection timeout"
            return (False, f"macOS no confirmó la conexión Bluetooth: {message}")

        if float(self.cfg.bt_forget_retry_s) > 0.0:
            time.sleep(float(self.cfg.bt_forget_retry_s))
        return (True, "")

    def _wait_for_mount_port(self) -> str:
        target_port = resolve_mount_port(self.cfg.port)
        if target_port or not _is_auto_port_value(self.cfg.port):
            return target_port

        deadline = time.monotonic() + max(0.0, float(self.cfg.bt_serial_port_wait_s))
        while time.monotonic() < deadline:
            time.sleep(0.20)
            target_port = resolve_mount_port(self.cfg.port)
            if target_port:
                return target_port
        return ""

    @property
    def is_connected(self) -> bool:
        ser = self._ser
        return (ser is not None) and bool(getattr(ser, "is_open", True))

    # ----------------------------
    # Connection lifecycle
    # ----------------------------

    def connect(self) -> str:
        """
        Abre el puerto y hace handshake:
          - resuelve puerto si cfg.port=AUTO
          - drena salida inicial
          - PING
          - ENABLE 1
          - STOP
        """
        if self.is_connected:
            return f"Mount ya conectado ({self.cfg.port})"

        # The Bluetooth SPP serial port may not exist while the device is
        # forgotten. Recreate the OS-level pairing first, then resolve it.
        bt_ok, bt_error = self._refresh_bt_pairing()
        target_port = self._wait_for_mount_port()
        if not target_port:
            available = ", ".join(list_serial_ports()) or "none"
            detail = f"; Bluetooth={bt_error}" if (not bt_ok and bt_error) else ""
            return f"Mount error al conectar (no serial port found; available={available}{detail})"

        open_attempts = max(1, int(self.cfg.open_attempts))
        last_error = "Mount error al conectar (unknown)"

        for open_try in range(open_attempts):
            with self._lock:
                if self.is_connected:
                    return f"Mount ya conectado ({self.cfg.port})"

                # cerrar previo si existe
                try:
                    if self._ser is not None:
                        self._ser.close()
                except Exception as exc:
                    log_error(None, "Mount: failed to close existing serial connection", exc, throttle_s=5.0, throttle_key="mount_close_existing")
                self._ser = None

                try:
                    ser = serial.Serial(
                        target_port,
                        int(self.cfg.baud),
                        timeout=float(self.cfg.timeout_s),
                        write_timeout=float(self.cfg.write_timeout_s),
                    )
                    if float(self.cfg.connect_sleep_s) > 0.0:
                        time.sleep(float(self.cfg.connect_sleep_s))

                    # limpiar buffers
                    try:
                        ser.reset_input_buffer()
                        ser.reset_output_buffer()
                    except Exception as exc:
                        log_error(None, "Mount: failed to reset serial buffers", exc, throttle_s=5.0, throttle_key="mount_reset_buffers")

                    self._ser = ser
                    self._move_profiles_supported = None
                    self.cfg.port = target_port
                except Exception as e:
                    self._ser = None
                    log_error(None, "Mount: failed to connect", e, throttle_s=5.0, throttle_key="mount_connect")
                    last_error = f"Mount error al conectar ({e})"
                    continue

            # fuera del lock: usar send()
            _ = self._drain_lines(max_lines=40, max_time_s=0.50)

            attempts = max(1, int(self.cfg.handshake_attempts))
            pong = ""
            for i in range(attempts):
                pong = self.ping()
                if pong:
                    break
                # Give RFCOMM a short extra window when the channel has just opened.
                if i < (attempts - 1):
                    time.sleep(0.20)
            if not pong:
                last_error = f"Mount error al conectar (sin respuesta a PING en {self.cfg.port})"
                self.close()
            else:
                ok1 = self.enable(True)
                if not ok1:
                    ok1 = self.enable(True)
                ok2 = self.stop()
                if not ok2:
                    ok2 = self.stop()
                if ok1 and ok2 and self.is_connected:
                    return (
                        f"Mount conectado en {self.cfg.port} "
                        f"(PING={pong or 'NO-RESP'} ENABLE={ok1 or 'NO-RESP'} STOP={ok2 or 'NO-RESP'})"
                    )
                last_error = (
                    f"Mount error al conectar (handshake incompleto en {self.cfg.port}; "
                    f"PING={pong or 'NO-RESP'} ENABLE={ok1 or 'NO-RESP'} STOP={ok2 or 'NO-RESP'})"
                )
                self.close()

            if open_try < (open_attempts - 1):
                time.sleep(max(0.0, float(self.cfg.open_retry_delay_s)))

        return last_error

    def _close_serial_locked(self, ser: Optional[serial.Serial]) -> None:
        try:
            if ser is not None:
                ser.close()
        except Exception as exc:
            log_error(
                None,
                "Mount: failed to close serial connection",
                exc,
                throttle_s=5.0,
                throttle_key="mount_close_locked",
            )
        self._ser = None
        self._move_profiles_supported = None

    def close(self) -> None:
        with self._lock:
            self._close_serial_locked(self._ser)

    def _ensure_connected(self) -> bool:
        if self.is_connected:
            return True
        if not bool(self.cfg.allow_reconnect):
            return False
        msg = self.connect()
        return "conectado" in _safe_lower(msg)

    # ----------------------------
    # Low-level I/O
    # ----------------------------

    def _drain_lines(self, max_lines: int = 10, max_time_s: float = 0.05) -> List[str]:
        """
        Lee y descarta líneas disponibles por un tiempo acotado.
        Útil para evitar backlog de respuestas.
        """
        if not self.is_connected:
            return []

        lines: List[str] = []
        t0 = time.time()

        with self._lock:
            ser = self._ser
            if ser is None:
                return []

            old_timeout = getattr(ser, "timeout", None)
            try:
                ser.timeout = 0.0  # no bloqueante
                while len(lines) < int(max_lines) and (time.time() - t0) < float(max_time_s):
                    try:
                        b = ser.readline()
                    except Exception as exc:
                        log_error(None, "Mount: failed to read serial line", exc, throttle_s=5.0, throttle_key="mount_readline")
                        break
                    if not b:
                        break
                    s = b.decode(errors="ignore").strip()
                    if s:
                        lines.append(s)
            finally:
                try:
                    if old_timeout is not None:
                        ser.timeout = old_timeout
                except Exception as exc:
                    log_error(None, "Mount: failed to restore serial timeout", exc, throttle_s=5.0, throttle_key="mount_timeout_restore")

        return lines

    @staticmethod
    def _response_matches_command(cmd: str, line: str) -> bool:
        """Reject delayed/asynchronous lines that belong to another command."""
        command = str(cmd or "").strip().upper()
        response = str(line or "").strip().upper()
        if not response:
            return False
        if response.startswith("ERR"):
            return True
        verb = command.split(" ", 1)[0]
        if verb == "PING":
            return response == "READY"
        if verb == "STATUS":
            return response.startswith("EN=") and "MOVE=" in response
        if verb == "MS":
            return response.startswith("OK MS")
        if verb == "DEBUG":
            return response.startswith("OK DEBUG")
        if verb in {"ENABLE", "STOP", "MOVE"}:
            return response == "OK"
        return True

    def send(self, cmd: str, timeout_s: float = 0.20, *, reset_input: Optional[bool] = None) -> str:
        """
        Envía un comando y espera 1 línea de respuesta (bloqueante hasta timeout_s).
        """
        cmd = (cmd or "").strip()
        if not cmd:
            return ""
        if not self._ensure_connected():
            raise RuntimeError("Mount not connected")

        if reset_input is None:
            reset_input = bool(self.cfg.reset_input_on_send)

        with self._lock:
            ser = self._ser
            if ser is None or not bool(getattr(ser, "is_open", True)):
                raise RuntimeError("Mount serial not open")

            try:
                if reset_input:
                    try:
                        ser.reset_input_buffer()
                    except Exception as exc:
                        log_error(None, "Mount: failed to reset input buffer", exc, throttle_s=5.0, throttle_key="mount_reset_input")
                        self._close_serial_locked(ser)
                        raise serial.SerialException("input buffer reset failed") from exc

                ser.write((cmd + "\n").encode("ascii", errors="ignore"))
                if self.cfg.flush_on_send:
                    try:
                        ser.flush()
                    except Exception as exc:
                        log_error(None, "Mount: failed to flush serial buffer", exc, throttle_s=5.0, throttle_key="mount_flush")

                t0 = time.time()
                while True:
                    try:
                        line = ser.readline().decode(errors="ignore").strip()
                    except Exception as exc:
                        log_error(None, "Mount: failed to read serial response", exc, throttle_s=5.0, throttle_key="mount_read_response")
                        line = ""
                    if line and self._response_matches_command(cmd, line):
                        return line
                    if (time.time() - t0) > float(timeout_s):
                        return ""
            except Exception as exc:
                log_error(None, "Mount: send failed", exc, throttle_s=5.0, throttle_key="mount_send")
                self._close_serial_locked(ser)
                raise

    # ----------------------------
    # High-level commands
    # ----------------------------

    def ping(self) -> str:
        return self.send("PING", timeout_s=0.80, reset_input=False)

    def enable(self, on: bool) -> str:
        return self.send(f"ENABLE {1 if on else 0}", timeout_s=0.80, reset_input=False)

    def stop(self) -> str:
        return self.send("STOP", timeout_s=0.80, reset_input=False)

    def move(
        self,
        axis: str,
        direction: str,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> str:
        axis = (axis or "").strip().upper()
        direction = (direction or "").strip().upper()

        if axis not in ("A", "B"):
            axis = "A"
        if direction not in ("FWD", "REV"):
            direction = "FWD"

        steps_i = max(0, int(steps))
        delay_i = max(0, int(delay_us))
        profile_i = normalize_move_profile(profile).upper()
        if self._move_profiles_supported is not True:
            status = self.status().upper()
            self._move_profiles_supported = "MOVEPROFILES=1" in status
        if not self._move_profiles_supported:
            if profile_i != "DIRECT":
                raise RuntimeError(
                    "firmware does not advertise MOVEPROFILES=1; flash the updated "
                    "firmware before using smooth moves"
                )
            # Legacy firmware implements the same constant-cadence move used
            # by the DIRECT profile, but accepts no trailing profile token.
            est_s = estimate_firmware_move_duration_s(
                steps_i,
                delay_i,
                profile="direct",
            )
            timeout_s = max(3.50, est_s + 1.5)
            return self.send(
                f"MOVE {axis} {direction} {steps_i} {delay_i}",
                timeout_s=float(timeout_s),
            )
        # Works with both firmware variants:
        # - legacy blocking MOVE (needs long timeout);
        # - new non-blocking MOVE (returns immediately anyway).
        est_s = estimate_firmware_move_duration_s(
            steps_i,
            delay_i,
            profile=profile_i.lower(),
        )
        timeout_s = max(3.50, est_s + 1.5)
        return self.send(
            f"MOVE {axis} {direction} {steps_i} {delay_i} {profile_i}",
            timeout_s=float(timeout_s),
        )

    def status(self) -> str:
        return self.send("STATUS", timeout_s=0.50, reset_input=False)

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        if int(az_div) != FIXED_MICROSTEPS or int(alt_div) != FIXED_MICROSTEPS:
            raise ValueError(
                f"Microstepping is hardware-fixed at 1/{FIXED_MICROSTEPS}; "
                f"requested AZ=1/{int(az_div)} ALT=1/{int(alt_div)}"
            )
        # Compatibility handshake: new firmware treats this as a no-op while
        # older installed firmware still receives the known-safe value.
        return self.send(f"MS {FIXED_MICROSTEPS}", timeout_s=0.60)


# =========================
# App-facing mount wrapper
# =========================

class ArduinoMount:
    """
    Wrapper de alto nivel para la app, alineado con el contrato:

      - connect(port, baud)
      - disconnect()
      - stop()
      - set_microsteps(64, 64)              -> legacy fixed-MS handshake
      - move_steps(axis, direction, steps, delay_us) -> MOVE ...
    """

    def __init__(self, cfg: Optional[ArduinoConfig] = None) -> None:
        self.cfg = cfg or ArduinoConfig()
        self.ctrl = ArduinoController(self.cfg)

    def connect(self, port: str, baud: int = 115200) -> str:
        self.cfg.port = str(port)
        self.cfg.baud = int(baud)
        msg = self.ctrl.connect()
        if "error" in _safe_lower(msg):
            log_error(None, f"Mount: connect failed ({msg})", throttle_s=5.0, throttle_key="mount_connect_result")
        else:
            log_info(None, f"Mount: connect result ({msg})", throttle_s=2.0, throttle_key="mount_connect_result_ok")
        return msg

    def disconnect(self) -> None:
        try:
            if self.ctrl.is_connected:
                self.stop()
        except Exception as exc:
            log_error(None, "Mount: stop failed during disconnect", exc, throttle_s=2.0, throttle_key="mount_stop_on_disconnect")
        finally:
            self.ctrl.close()

    def is_connected(self) -> bool:
        return self.ctrl.is_connected

    def stop(self) -> str:
        return self.ctrl.stop()

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        return self.ctrl.set_microsteps(int(az_div), int(alt_div))

    @staticmethod
    def _estimate_move_duration_s(
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> float:
        return estimate_firmware_move_duration_s(
            int(steps), int(delay_us), profile=profile
        )

    def move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        *,
        profile: str = "smooth",
        blocking: bool = True,
        stop_before_move: bool = True,
    ) -> str:
        """
        Movimiento manual determinista (tu modo preferido):
          MOVE A|B FWD|REV steps delay_us

        axis: Axis.AZ / Axis.ALT
        direction: -1 o +1  (>=0 => FWD, <0 => REV)
        blocking:
          - True: espera duración estimada del movimiento.
          - False: retorna inmediatamente tras enviar MOVE.
        stop_before_move:
          - True: envía STOP antes de MOVE (cancela cualquier movimiento previo).
          - False: no envía STOP; permite solapar MOVEs en ejes distintos.
        """
        # Validación mínima
        if int(steps) <= 0:
            return ""
        if int(delay_us) <= 0:
            return ""

        ax = _axis_to_fw(axis)
        dr = _dir_to_fw(int(direction))

        if bool(stop_before_move):
            # Seguridad: detener planes de MOVE en progreso antes de iniciar el nuevo.
            # (El AppRunner ya hace stop() antes de llamar, pero aquí es idempotente.)
            try:
                self.ctrl.stop()
            except Exception as exc:
                log_error(
                    None,
                    "Mount: failed to stop before move",
                    exc,
                    throttle_s=5.0,
                    throttle_key="mount_stop_before_move",
                )

        move_profile = normalize_move_profile(profile)
        resp = self.ctrl.move(
            ax,
            dr,
            int(steps),
            int(delay_us),
            profile=move_profile,
        )
        if bool(blocking):
            wait_s = self._estimate_move_duration_s(
                int(steps),
                int(delay_us),
                profile=move_profile,
            )
            # Small safety margin for serial/firmware jitter.
            if wait_s > 0.0:
                time.sleep(wait_s + 0.05)
        return resp


class MountMoveWorker(BaseWorker):
    """
    Background worker for manual mount moves.

    Dependencies injected:
      - get_mount(): returns ArduinoMount or None
      - note_manual_move(axis, direction, steps)
      - publish_state(patch)
    """

    def __init__(
        self,
        *,
        get_mount: Callable[[], Optional["ArduinoMount"]],
        note_manual_move: Callable[[Axis, int, int], None],
        get_last_direction: Optional[Callable[[Axis], int]] = None,
        set_last_direction: Optional[Callable[[Axis, int], None]] = None,
        get_backlash_steps: Optional[Callable[[Axis], int]] = None,
        publish_state: StatePublisherProtocol,
        operation_finished: Optional[Callable[[], None]] = None,
        out_log: Any = None,
    ) -> None:
        super().__init__(name="MountMoveWorker")
        self._get_mount = get_mount
        self._note_manual_move = note_manual_move
        self._get_last_direction = get_last_direction
        self._set_last_direction = set_last_direction
        self._get_backlash_steps = get_backlash_steps
        self._publish_state = publish_state
        self._operation_finished = operation_finished
        self._out_log = out_log

    def request(
        self,
        *,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> None:
        super().request(
            axis=axis,
            direction=int(direction),
            steps=int(steps),
            delay_us=int(delay_us),
            profile=normalize_move_profile(profile),
        )

    def _handle_request(self, request: Dict[str, Any]) -> None:
        try:
            mount = self._get_mount()
            if mount is None or not mount.is_connected():
                return
            axis = request["axis"]
            direction = int(request["direction"])
            steps = int(request["steps"])
            delay_us = int(request["delay_us"])
            profile = normalize_move_profile(str(request.get("profile", "smooth")))
            new_direction = +1 if direction >= 0 else -1
            previous_direction = (
                int(self._get_last_direction(axis))
                if self._get_last_direction is not None
                else 0
            )
            backlash_steps = (
                max(0, int(self._get_backlash_steps(axis)))
                if self._get_backlash_steps is not None
                else 0
            )
            if (
                previous_direction in (-1, +1)
                and previous_direction != new_direction
                and backlash_steps > 0
            ):
                mount.move_steps(
                    axis=axis,
                    direction=new_direction,
                    steps=backlash_steps,
                    delay_us=delay_us,
                    profile=profile,
                    blocking=True,
                    stop_before_move=False,
                )
                if self._set_last_direction is not None:
                    self._set_last_direction(axis, new_direction)
            mount.move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=delay_us,
                profile=profile,
                # One MOVE is important: the loaded firmware accelerates and
                # brakes each command internally. Splitting it would create
                # repeated slowdowns between chunks.
                blocking=True,
                stop_before_move=False,
            )

            self._note_manual_move(axis, direction, steps)
        except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
            self._publish_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "manual move failed"},
                    "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE},
                }
            )
            log_error(self._out_log, "Mount: MOVE steps failed", exc)
        finally:
            if self._operation_finished is not None:
                self._operation_finished()
