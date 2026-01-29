# math_utils.py
from __future__ import annotations

from typing import Sequence

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def wrap_deg_180(x: float) -> float:
    y = (float(x) + 180.0) % 360.0 - 180.0
    if y <= -180.0:
        y += 360.0
    return float(y)


def wrap_deg_360(x: float) -> float:
    y = float(x) % 360.0
    if y < 0.0:
        y += 360.0
    return float(y)


def as_array2(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError("expected a 2-vector")
    return a
