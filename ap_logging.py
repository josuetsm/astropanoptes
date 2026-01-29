# ap_logging.py
from __future__ import annotations

from typing import Optional

from logging_utils import log_error, log_info


def get_logger_prefix(prefix: str) -> str:
    """Return a normalized prefix for log messages."""
    return str(prefix).strip()


def info(prefix: str, msg: str) -> None:
    log_info(None, f"{get_logger_prefix(prefix)}: {msg}")


def error(prefix: str, msg: str, exc: Optional[BaseException] = None) -> None:
    log_error(None, f"{get_logger_prefix(prefix)}: {msg}", exc)
