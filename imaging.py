# imaging.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2


_BAYER_CV2 = {
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
}
_BAYER_CV2_EA = {
    "RGGB": cv2.COLOR_BayerRG2RGB_EA,
    "BGGR": cv2.COLOR_BayerBG2RGB_EA,
    "GRBG": cv2.COLOR_BayerGR2RGB_EA,
    "GBRG": cv2.COLOR_BayerGB2RGB_EA,
}


def ensure_raw16_bayer(frame: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a RAW16 Bayer frame.

    Returns a 2D uint16 array (H,W). Accepts (H,W) or (H,W,1).
    """
    arr = np.asarray(frame)
    if arr.dtype != np.uint16:
        raise TypeError(f"raw16 must be uint16, got {arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"raw16 must have shape (H,W) or (H,W,1), got {arr.shape}")


def debayer_cv2(raw: np.ndarray, pattern: str = "RGGB", edge_aware: bool = False) -> np.ndarray:
    p = (pattern or "RGGB").upper()
    code = (_BAYER_CV2_EA if edge_aware else _BAYER_CV2).get(p, cv2.COLOR_BayerRG2RGB)
    return cv2.cvtColor(raw, code)


def median_prefilter_raw16(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    base = cv2.medianBlur(img, ksize)
    return base.astype(np.float32)


def half_to_full_shift(dx_half: float, dy_half: float) -> Tuple[float, float]:
    return 2.0 * float(dx_half), 2.0 * float(dy_half)


def half_affine_to_full(M_half: np.ndarray) -> np.ndarray:
    M = np.asarray(M_half, dtype=np.float32)
    if M.shape != (2, 3):
        raise ValueError(f"Expected affine matrix shape (2,3), got {M.shape}")
    S = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    S_inv = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    M3 = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    M_full = S @ M3 @ S_inv
    return M_full[:2, :]


def warp_rgb16(rgb16: np.ndarray, M: np.ndarray, dsize: Tuple[int, int] | None = None) -> np.ndarray:
    rgb = np.asarray(rgb16)
    if rgb.dtype != np.uint16:
        raise TypeError(f"warp_rgb16 espera uint16, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"warp_rgb16 espera shape (H,W,3), got {rgb.shape}")
    mat = np.asarray(M, dtype=np.float32)
    if mat.shape != (2, 3):
        raise ValueError(f"warp_rgb16 espera matriz affine 2x3, got {mat.shape}")
    h, w = rgb.shape[:2]
    out_size = dsize if dsize is not None else (w, h)
    warped = cv2.warpAffine(
        rgb,
        mat,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped.astype(np.uint16, copy=False)


__all__ = [
    "ensure_raw16_bayer",
    "debayer_cv2",
    "median_prefilter_raw16",
    "warp_rgb16",
    "half_to_full_shift",
    "half_affine_to_full",
]
