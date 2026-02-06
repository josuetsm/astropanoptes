# logging_utils.py
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import traceback
from typing import Optional, Dict, Callable

if importlib.util.find_spec("ipywidgets") is not None:
    import ipywidgets as W
else:
    W = None  # type: ignore


_START_TIME = time.monotonic()

_GLOBAL_LOG_SINK_LOCK = threading.Lock()
_GLOBAL_LOG_SINK: Optional[Callable[[str], None]] = None


def set_global_log_sink(sink: Optional[Callable[[str], None]]) -> None:
    """
    Register a global sink for log lines (used when out is None).
    The sink must be a callable that accepts a single string line.
    """
    global _GLOBAL_LOG_SINK
    with _GLOBAL_LOG_SINK_LOCK:
        _GLOBAL_LOG_SINK = sink


def _ts() -> str:
    # timestamp monotónico y consistente para logs (HH:MM:SS.mmm)
    elapsed = time.monotonic() - _START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = float(elapsed % 60.0)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


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


def append_to_output(out: "W.Output", msg: str) -> None:
    """
    Escribe en un ipywidgets.Output sin romper el notebook si falla.
    """
    if out is None:
        return
    try:
        if hasattr(out, "append") and callable(getattr(out, "append")):
            out.append(msg)
            return
        if hasattr(out, "write") and callable(getattr(out, "write")):
            out.write(msg + "\n")
            return
        with out:
            print(msg)
    except Exception:
        sys.stderr.write(f"[{_ts()}][Logging][ERROR] Failed to write to output widget.\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()


def _format_line(level: str, msg: str) -> str:
    thread_name = threading.current_thread().name
    return f"[{_ts()}][{thread_name}][{level}] {msg}"


def _write_console(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log_info(
    out: Optional["W.Output"],
    msg: str,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    key = throttle_key or msg
    if not _should_log(key, throttle_s):
        return
    line = _format_line("INFO", msg)
    if out is not None:
        append_to_output(out, line)
        return
    sink = None
    with _GLOBAL_LOG_SINK_LOCK:
        sink = _GLOBAL_LOG_SINK
    if sink is not None:
        try:
            sink(line)
            return
        except Exception:
            sys.stderr.write(f"[{_ts()}][Logging][ERROR] Failed to write to global sink.\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
    _write_console(line)


def log_error(
    out: Optional["W.Output"],
    msg: str,
    exc: Optional[BaseException] = None,
    *,
    throttle_s: Optional[float] = None,
    throttle_key: Optional[str] = None,
) -> None:
    key = throttle_key or msg
    if not _should_log(key, throttle_s):
        return
    line = _format_line("ERROR", msg)
    if out is not None:
        append_to_output(out, line)
        if exc is not None:
            append_to_output(out, format_exc(exc))
        return
    sink = None
    with _GLOBAL_LOG_SINK_LOCK:
        sink = _GLOBAL_LOG_SINK
    if sink is not None:
        try:
            sink(line)
            if exc is not None:
                sink(format_exc(exc))
            return
        except Exception:
            sys.stderr.write(f"[{_ts()}][Logging][ERROR] Failed to write to global sink.\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
    _write_console(line)
    if exc is not None:
        _write_console(format_exc(exc))
