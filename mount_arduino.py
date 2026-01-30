# mount_arduino.py
from __future__ import annotations

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


# =========================
# Utilities
# =========================

def list_serial_ports() -> list[str]:
    ports: list[str] = []
    for p in serial.tools.list_ports.comports():
        if p.device:
            ports.append(p.device)
    return ports


def _safe_lower(s: str) -> str:
    try:
        return (s or "").lower()
    except Exception as exc:
        log_error(None, "Mount: failed to normalize string", exc, throttle_s=10.0, throttle_key="mount_safe_lower")
        return ""


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
    port: str = ""
    baud: int = 115200

    # timeouts serial
    timeout_s: float = 0.10
    write_timeout_s: float = 0.25

    # Arduino reset timing on open (UNO)
    connect_sleep_s: float = 1.8

    # send behavior
    flush_on_send: bool = True
    reset_input_on_send: bool = True

    # RATE performance
    rate_read_timeout_s: float = 0.01
    rate_reset_input: bool = True

    # reconnect behavior
    allow_reconnect: bool = True


class ArduinoController:
    """
    Controlador thread-safe para Arduino UNO con tu firmware actual.

    Protocolo (newline):
      PING                  -> READY
      ENABLE 0|1            -> OK
      STOP                  -> OK
      RATE vA vB            -> OK RATE <vA> <vB>
      MS <8|16|32|64>       -> OK MS <az> <alt>
      MS AZ <...>           -> OK MS <az> <alt>
      MS ALT <...>          -> OK MS <az> <alt>
      MOVE A|B FWD|REV steps delay_us -> OK
      STATUS                -> EN=... RATE=... MS=...
    """

    def __init__(self, cfg: ArduinoConfig):
        self.cfg = cfg
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

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
          - drena salida inicial
          - PING
          - ENABLE 1
          - STOP
        """
        with self._lock:
            if self.is_connected:
                return f"Arduino ya conectado ({self.cfg.port})"

            # cerrar previo si existe
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception as exc:
                log_error(None, "Mount: failed to close existing serial connection", exc, throttle_s=5.0, throttle_key="mount_close_existing")
            self._ser = None

            try:
                ser = serial.Serial(
                    self.cfg.port,
                    int(self.cfg.baud),
                    timeout=float(self.cfg.timeout_s),
                    write_timeout=float(self.cfg.write_timeout_s),
                )
                # esperar reset típico UNO al abrir puerto
                time.sleep(float(self.cfg.connect_sleep_s))

                # limpiar buffers
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception as exc:
                    log_error(None, "Mount: failed to reset serial buffers", exc, throttle_s=5.0, throttle_key="mount_reset_buffers")

                self._ser = ser
            except Exception as e:
                self._ser = None
                log_error(None, "Mount: failed to connect", e, throttle_s=5.0, throttle_key="mount_connect")
                return f"Arduino error al conectar ({e})"

        # fuera del lock: usar send()
        _ = self._drain_lines(max_lines=20, max_time_s=0.25)

        pong = self.ping()
        ok1 = self.enable(True)
        ok2 = self.stop()

        if self.is_connected:
            return (
                f"Arduino conectado en {self.cfg.port} "
                f"(PING={pong or 'NO-RESP'} ENABLE={ok1 or 'NO-RESP'} STOP={ok2 or 'NO-RESP'})"
            )
        return f"Arduino conectado en {self.cfg.port} (pero puerto no quedó abierto)"

    def close(self) -> None:
        with self._lock:
            ser = self._ser
            self._ser = None
            try:
                if ser is not None:
                    ser.close()
            except Exception as exc:
                log_error(None, "Mount: failed to close serial connection", exc, throttle_s=5.0, throttle_key="mount_close")

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
        Útil para evitar backlog de respuestas (especialmente con RATE).
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
                try:
                    ser.close()
                except Exception as exc:
                    log_error(None, "Mount: failed to close serial after send error", exc, throttle_s=5.0, throttle_key="mount_close_after_send")
                self._ser = None
                raise

    def send_fast(self, cmd: str, *, reset_input: bool = True, read_timeout_s: float = 0.01) -> str:
        """
        Versión para alta frecuencia (RATE):
        - escribe el comando
        - opcionalmente descarta backlog antes
        - intenta leer 1 línea con timeout muy corto (para no bloquear)
        """
        cmd = (cmd or "").strip()
        if not cmd:
            return ""
        if not self._ensure_connected():
            raise RuntimeError("Mount not connected")

        with self._lock:
            ser = self._ser
            if ser is None or not bool(getattr(ser, "is_open", True)):
                raise RuntimeError("Mount serial not open")

            try:
                if reset_input:
                    try:
                        ser.reset_input_buffer()
                    except Exception as exc:
                        log_error(None, "Mount: failed to reset input buffer (fast)", exc, throttle_s=5.0, throttle_key="mount_reset_input_fast")

                ser.write((cmd + "\n").encode("ascii", errors="ignore"))
                if self.cfg.flush_on_send:
                    try:
                        ser.flush()
                    except Exception as exc:
                        log_error(None, "Mount: failed to flush serial buffer (fast)", exc, throttle_s=5.0, throttle_key="mount_flush_fast")

                old_timeout = getattr(ser, "timeout", None)
                try:
                    ser.timeout = float(read_timeout_s)
                    line = ser.readline().decode(errors="ignore").strip()
                finally:
                    try:
                        if old_timeout is not None:
                            ser.timeout = old_timeout
                    except Exception as exc:
                        log_error(None, "Mount: failed to restore serial timeout (fast)", exc, throttle_s=5.0, throttle_key="mount_timeout_restore_fast")

                return line or ""
            except Exception as exc:
                log_error(None, "Mount: send_fast failed", exc, throttle_s=5.0, throttle_key="mount_send_fast")
                try:
                    ser.close()
                except Exception as exc:
                    log_error(None, "Mount: failed to close serial after send_fast error", exc, throttle_s=5.0, throttle_key="mount_close_after_send_fast")
                self._ser = None
                raise

    # ----------------------------
    # High-level commands
    # ----------------------------

    def ping(self) -> str:
        return self.send("PING", timeout_s=0.30)

    def enable(self, on: bool) -> str:
        return self.send(f"ENABLE {1 if on else 0}", timeout_s=0.30)

    def stop(self) -> str:
        return self.send("STOP", timeout_s=0.30)

    def rate(self, v_az: float, v_alt: float) -> str:
        """
        RATE a alta frecuencia: usar send_fast.
        Firmware responde "OK RATE ...", pero aquí no bloqueamos esperando.
        """
        return self.send_fast(
            f"RATE {float(v_az):.3f} {float(v_alt):.3f}",
            reset_input=bool(self.cfg.rate_reset_input),
            read_timeout_s=float(self.cfg.rate_read_timeout_s),
        )

    def move(self, axis: str, direction: str, steps: int, delay_us: int) -> str:
        axis = (axis or "").strip().upper()
        direction = (direction or "").strip().upper()

        if axis not in ("A", "B"):
            axis = "A"
        if direction not in ("FWD", "REV"):
            direction = "FWD"

        steps_i = max(0, int(steps))
        delay_i = max(0, int(delay_us))
        # MOVE es blocking en firmware; timeout debe escalar con la duración esperada
        # Aproximación: cada microstep hace HIGH+LOW con delay_us => ~2*delay_us por paso
        est_s = (float(steps_i) * 2.0 * float(delay_i)) / 1.0e6
        timeout_s = max(3.50, est_s + 1.5)
        return self.send(f"MOVE {axis} {direction} {steps_i} {delay_i}", timeout_s=float(timeout_s))

    def status(self) -> str:
        return self.send("STATUS", timeout_s=0.30)

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        az = int(az_div)
        alt = int(alt_div)

        if az == alt:
            return self.send(f"MS {az}", timeout_s=0.60)

        r1 = self.send(f"MS AZ {az}", timeout_s=0.60)
        r2 = self.send(f"MS ALT {alt}", timeout_s=0.60)
        return r2 or r1


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
        self.stop()
        self.ctrl.close()

    def is_connected(self) -> bool:
        return self.ctrl.is_connected

    def stop(self) -> str:
        return self.ctrl.stop()

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        return self.ctrl.set_microsteps(int(az_div), int(alt_div))

    def move_steps(self, axis: Axis, direction: int, steps: int, delay_us: int) -> str:
        """
        Movimiento manual determinista (tu modo preferido):
          MOVE A|B FWD|REV steps delay_us

        axis: Axis.AZ / Axis.ALT
        direction: -1 o +1  (>=0 => FWD, <0 => REV)
        """
        # Validación mínima
        if int(steps) <= 0:
            return ""
        if int(delay_us) <= 0:
            return ""

        ax = _axis_to_fw(axis)
        dr = _dir_to_fw(int(direction))

        # Seguridad: si algo estaba en RATE/continuous, detener antes.
        # (El AppRunner ya hace stop() antes de llamar, pero aquí es idempotente.)
        try:
            self.ctrl.stop()
        except Exception as exc:
            log_error(None, "Mount: failed to stop before move", exc, throttle_s=5.0, throttle_key="mount_stop_before_move")

        return self.ctrl.move(ax, dr, int(steps), int(delay_us))

    def rate(self, v_az: float, v_alt: float) -> str:
        return self.ctrl.rate(float(v_az), float(v_alt))


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
            mount.stop()
            mount.move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=delay_us,
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
