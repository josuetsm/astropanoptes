# goto_calibration.py
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def fit_J_from_samples(
    calib_steps: Iterable[np.ndarray],
    calib_daltaz: Iterable[np.ndarray],
    *,
    min_samples: int = 3,
    ridge: float = 1e-12,
) -> Optional[np.ndarray]:
    steps_list = list(calib_steps)
    daltaz_list = list(calib_daltaz)
    if len(steps_list) < int(min_samples):
        return None

    S = np.stack(steps_list, axis=0)  # (N,2)
    D = np.stack(daltaz_list, axis=0)  # (N,2)

    if ridge > 0:
        lam = float(ridge)
        S_aug = np.vstack([S, math.sqrt(lam) * np.eye(2)])
        D_aug = np.vstack([D, np.zeros((2, 2), dtype=np.float64)])
    else:
        S_aug, D_aug = S, D

    B, *_ = np.linalg.lstsq(S_aug, D_aug, rcond=None)
    J_new = B.T

    if not np.all(np.isfinite(J_new)):
        return None

    return J_new.astype(np.float64)
