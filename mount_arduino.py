# mount_arduino.py
from __future__ import annotations

import json
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


SUPPORTED_MICROSTEPS: tuple[int, ...] = (8, 16, 32, 64)


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
        if "bluetooth" in hint or "spp" in hint:
            score += 20
        if dev.startswith("/dev/cu."):
            score += 5
        candidates.append((score, dev))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_dev = candidates[0]
    if best_score > 0:
        return best_dev

    # Fallback when no ESP32-like metadata is exposed by the OS.
    for _score, dev in candidates:
        if dev.startswith("/dev/cu."):
            return dev
    return best_dev


def resolve_common_microsteps(az_div: int, alt_div: int, *, default_ms: int = 64) -> int:
    """
    ESP32 firmware uses COMMON MS1/MS2 pins:
    both axes always end up with the same microstep divider.
    """
    az = int(az_div)
    alt = int(alt_div)
    if az in SUPPORTED_MICROSTEPS and alt in SUPPORTED_MICROSTEPS:
        return az
    if az in SUPPORTED_MICROSTEPS:
        return az
    if alt in SUPPORTED_MICROSTEPS:
        return alt
    return int(default_ms)


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


class ArduinoController:
    """
    Controlador thread-safe para ESP32 (Bluetooth SPP exposed as serial port).

    Protocolo (newline):
      PING                  -> READY
      ENABLE 0|1            -> OK
      STOP                  -> OK
      MS <8|16|32|64>       -> OK MS <common>
      MS AZ <...>           -> OK MS <common> (accepted, but common pins)
      MS ALT <...>          -> OK MS <common> (accepted, but common pins)
      MOVE A|B FWD|REV steps delay_us -> OK
      STATUS                -> EN=... MS=... MOVE=... BT=... DBG=...
    """

    def __init__(self, cfg: ArduinoConfig):
        self.cfg = cfg
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    def _blueutil_path(self) -> str:
        return str(shutil.which("blueutil") or "")

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

    def _blueutil_get_device_id(self) -> str:
        """
        Returns BT device ID (address if possible) for the configured mount device name.
        """
        name = str(self.cfg.bt_device_name or "").strip()
        if not name:
            return ""

        rc, out, _err = self._blueutil_run("--paired", "--format", "json", timeout_s=6.0)
        if rc == 0 and out:
            try:
                rows = json.loads(out)
                if isinstance(rows, list):
                    for row in rows:
                        nm = str((row or {}).get("name", "")).strip()
                        if nm == name:
                            addr = str((row or {}).get("address", "")).strip()
                            return addr or name
            except Exception:
                pass
        return name

    def _refresh_bt_pairing(self) -> None:
        """
        macOS workaround:
        disconnect + unpair + pair + connect via blueutil to avoid stale RFCOMM sessions.
        """
        if not bool(self.cfg.bt_forget_before_connect):
            return
        if not self._blueutil_path():
            return
        device_id = self._blueutil_get_device_id()
        if not device_id:
            return

        rc_disc, _o_disc, e_disc = self._blueutil_run("--disconnect", device_id, timeout_s=6.0)
        if rc_disc != 0 and e_disc:
            log_info(None, f"Mount: BT disconnect hint ({e_disc})", throttle_s=2.0, throttle_key="mount_bt_disconnect_hint")

        rc_unpair, _o_unpair, e_unpair = self._blueutil_run("--unpair", device_id, timeout_s=8.0)
        if rc_unpair != 0 and e_unpair:
            log_info(None, f"Mount: BT unpair hint ({e_unpair})", throttle_s=2.0, throttle_key="mount_bt_unpair_hint")

        if float(self.cfg.bt_forget_retry_s) > 0.0:
            time.sleep(float(self.cfg.bt_forget_retry_s))

        if bool(self.cfg.bt_pair_after_forget):
            rc_pair, _o_pair, e_pair = self._blueutil_run("--pair", device_id, timeout_s=12.0)
            if rc_pair != 0:
                log_info(None, f"Mount: BT pair hint ({e_pair or 'pair failed'})", throttle_s=2.0, throttle_key="mount_bt_pair_hint")

        rc_conn, _o_conn, e_conn = self._blueutil_run("--connect", device_id, timeout_s=10.0)
        if rc_conn != 0 and e_conn:
            log_info(None, f"Mount: BT connect hint ({e_conn})", throttle_s=2.0, throttle_key="mount_bt_connect_hint")

        if float(self.cfg.bt_forget_retry_s) > 0.0:
            time.sleep(float(self.cfg.bt_forget_retry_s))

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
        target_port = resolve_mount_port(self.cfg.port)
        if not target_port:
            available = ", ".join(list_serial_ports()) or "none"
            return f"Mount error al conectar (no serial port found; available={available})"

        self._refresh_bt_pairing()

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
                    if line:
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

    def move(self, axis: str, direction: str, steps: int, delay_us: int) -> str:
        axis = (axis or "").strip().upper()
        direction = (direction or "").strip().upper()

        if axis not in ("A", "B"):
            axis = "A"
        if direction not in ("FWD", "REV"):
            direction = "FWD"

        steps_i = max(0, int(steps))
        delay_i = max(0, int(delay_us))
        # Works with both firmware variants:
        # - legacy blocking MOVE (needs long timeout);
        # - new non-blocking MOVE (returns immediately anyway).
        est_s = (float(steps_i) * 2.0 * float(delay_i)) / 1.0e6
        timeout_s = max(3.50, est_s + 1.5)
        return self.send(f"MOVE {axis} {direction} {steps_i} {delay_i}", timeout_s=float(timeout_s))

    def status(self) -> str:
        return self.send("STATUS", timeout_s=0.50, reset_input=False)

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        ms = resolve_common_microsteps(int(az_div), int(alt_div), default_ms=64)
        if int(az_div) != int(alt_div):
            log_info(
                None,
                f"Mount: MS common pins active; forcing AZ={int(az_div)} ALT={int(alt_div)} -> {ms}",
                throttle_s=2.0,
                throttle_key="mount_ms_common_force",
            )
        if ms not in SUPPORTED_MICROSTEPS:
            raise ValueError(f"Unsupported microsteps value: {ms} (supported: {SUPPORTED_MICROSTEPS})")
        return self.send(f"MS {ms}", timeout_s=0.60)


# =========================
# App-facing mount wrapper
# =========================

class ArduinoMount:
    """
    Wrapper de alto nivel para la app, alineado con el contrato:

      - connect(port, baud)
      - disconnect()
      - stop()
      - set_microsteps(az_div, alt_div)     -> MS ...
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
    def _estimate_move_duration_s(steps: int, delay_us: int) -> float:
        steps_i = max(0, int(steps))
        delay_i = max(0, int(delay_us))
        # Firmware MOVE cadence is approximately one step every (delay_us + pulse_us).
        pulse_us = 3.0
        return float(steps_i) * ((float(delay_i) + pulse_us) / 1.0e6)

    def move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        *,
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

        resp = self.ctrl.move(ax, dr, int(steps), int(delay_us))
        if bool(blocking):
            wait_s = self._estimate_move_duration_s(int(steps), int(delay_us))
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
        publish_state: StatePublisherProtocol,
        out_log: Any = None,
    ) -> None:
        super().__init__(name="MountMoveWorker")
        self._get_mount = get_mount
        self._note_manual_move = note_manual_move
        self._publish_state = publish_state
        self._out_log = out_log

    def request(self, *, axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        super().request(
            axis=axis,
            direction=int(direction),
            steps=int(steps),
            delay_us=int(delay_us),
        )

    def _handle_request(self, request: Dict[str, Any]) -> None:
        mount = self._get_mount()
        if mount is None or not mount.is_connected():
            return
        axis = request["axis"]
        direction = int(request["direction"])
        steps = int(request["steps"])
        delay_us = int(request["delay_us"])

        try:
            mount.move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=delay_us,
                # Keep manual commands serialized at the app layer.
                # Firmware keeps only one active plan per axis, so sending a
                # second MOVE too early can replace the previous one.
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
