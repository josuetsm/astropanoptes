# logging_utils.py
"""Eventos de la app: una linea por evento, en el formato de `console_format`.

Los eventos van por stderr. stdout queda para la conversacion de la consola
(comando, resultado, detalle), de modo que un script -- o un agente -- puede
separar lo que respondio a su comando de lo que paso mientras tanto.

Las trazas completas de excepciones no van a la consola: alli solo se ve el tipo
y el mensaje, en la misma linea del evento. La traza entera se escribe en el
fichero de log de la sesion, si lo hay, que es donde sirve.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO

from console_format import event_line

_GLOBAL_LOG_SINK_LOCK = threading.Lock()
_GLOBAL_LOG_SINK: Optional[Callable[[str], None]] = None

_LOG_FILE_LOCK = threading.Lock()
_LOG_FILE: Optional[TextIO] = None
_LOG_FILE_PATH: Optional[Path] = None


def set_global_log_sink(sink: Optional[Callable[[str], None]]) -> None:
    """Destino comun para los eventos sin destino propio (la consola de la GUI)."""
    global _GLOBAL_LOG_SINK
    with _GLOBAL_LOG_SINK_LOCK:
        _GLOBAL_LOG_SINK = sink


def set_log_file(path: Optional[Path | str]) -> Optional[Path]:
    """Abre (o cierra, con None) el fichero de log de la sesion.

    Recibe todos los eventos y, a diferencia de la consola, las trazas completas.
    """
    global _LOG_FILE, _LOG_FILE_PATH
    with _LOG_FILE_LOCK:
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.close()
            except OSError:
                pass
        _LOG_FILE, _LOG_FILE_PATH = None, None
        if path is None:
            return None
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = target.open("a", encoding="utf-8", buffering=1)
        _LOG_FILE_PATH = target
        return target


def log_file_path() -> Optional[Path]:
    return _LOG_FILE_PATH


_THROTTLE_LOCK = threading.Lock()
_THROTTLE_STATE: Dict[str, float] = {}


def _should_log(throttle_key: str, throttle_s: Optional[float]) -> bool:
    if throttle_s is None:
        return True
    now = time.monotonic()
    with _THROTTLE_LOCK:
        last = _THROTTLE_STATE.get(throttle_key, None)
        if last is not None and (now - last) < float(throttle_s):
            return False
        _THROTTLE_STATE[throttle_key] = now
    return True


def format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _exc_summary(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _to_file(line: str, exc: Optional[BaseException] = None) -> None:
    with _LOG_FILE_LOCK:
        if _LOG_FILE is None:
            return
        try:
            _LOG_FILE.write(line + "\n")
            if exc is not None:
                _LOG_FILE.write(format_exc(exc))
        except OSError:
            pass


def append_to_output(out: Any, msg: str) -> None:
    """Escribe en un destino sin romper la app si falla."""
    if out is None:
        return
    try:
        if hasattr(out, "append") and callable(getattr(out, "append")):
            out.append(msg)
            return
        if hasattr(out, "write") and callable(getattr(out, "write")):
            out.write(msg + "\n")
            if hasattr(out, "flush"):
                out.flush()
            return
        with out:
            print(msg)
    except Exception:
        sys.stderr.write(event_line("error", "app: no se pudo escribir un evento en su destino") + "\n")
        sys.stderr.flush()


def _write_console(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _emit(level: str, out: Optional[Any], msg: str, exc: Optional[BaseException]) -> None:
    text = msg if exc is None else f"{msg}  ({_exc_summary(exc)})"
    line = event_line(level, text)
    _to_file(line, exc)
    if out is not None:
        append_to_output(out, line)
        return
    with _GLOBAL_LOG_SINK_LOCK:
        sink = _GLOBAL_LOG_SINK
    if sink is not None:
        try:
            sink(line)
            return
        except Exception:
            pass
    _write_console(line)


def _debug_on_console() -> bool:
    return os.environ.get("ASTROPANOPTES_DEBUG", "").strip().lower() in {"1", "true", "yes", "si", "sí"}


def log_debug(
    out: Optional[Any],
    msg: str,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    """Detalle interno: va al fichero de log de la sesion, no a la consola.

    Los volcados por cuadro (fuentes detectadas, configuracion completa de un
    procedimiento) sirven para depurar despues, pero en la consola tapaban los
    eventos que importan: un autocal escribia diez lineas solo de fuentes por
    cuadro. Con ASTROPANOPTES_DEBUG=1 vuelven a la consola.
    """
    if not _should_log(throttle_key or msg, throttle_s):
        return
    line = event_line("debug", msg)
    _to_file(line)
    if _debug_on_console():
        if out is not None:
            append_to_output(out, line)
        else:
            _write_console(line)


def log_info(
    out: Optional[Any],
    msg: str,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    if _should_log(throttle_key or msg, throttle_s):
        _emit("info", out, msg, None)


def log_warning(
    out: Optional[Any],
    msg: str,
    exc: Optional[BaseException] = None,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    if _should_log(throttle_key or msg, throttle_s):
        _emit("warning", out, msg, exc)


def log_error(
    out: Optional[Any],
    msg: str,
    exc: Optional[BaseException] = None,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    if _should_log(throttle_key or msg, throttle_s):
        _emit("error", out, msg, exc)
