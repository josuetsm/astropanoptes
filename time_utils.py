# time_utils.py
from __future__ import annotations

from time import monotonic


def monotonic_s() -> float:
    return float(monotonic())
