# logging_utils.py
from __future__ import annotations

import time
import traceback
from typing import Optional

try:
    import ipywidgets as W
except Exception:
    W = None  # type: ignore


def _ts() -> str:
    # timestamp simple y consistente
    return f"{time.time():.1f}s"


def format_exc(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def append_to_output(out: "W.Output", msg: str) -> None:
    """
    Escribe en un ipywidgets.Output sin romper el notebook si falla.
    """
    if out is None:
        return
    try:
        with out:
            print(msg)
    except Exception:
        pass


def log_info(out: Optional["W.Output"], msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    if out is not None:
        append_to_output(out, line)
    else:
        print(line)


def log_error(out: Optional["W.Output"], msg: str, exc: Optional[BaseException] = None) -> None:
    line = f"[{_ts()}] ERROR: {msg}"
    if out is not None:
        append_to_output(out, line)
        if exc is not None:
            append_to_output(out, format_exc(exc))
    else:
        print(line)
        if exc is not None:
            print(format_exc(exc))
