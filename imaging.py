# imaging.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

from hotpixels import apply_hotpixel_mask_replace


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


def bayer_green_u8_from_u16(u16: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """
    Extrae una aproximación rápida del canal verde desde Bayer RAW16, sin debayer completo.

    Retorna una imagen u8 submuestreada 2x, usando el promedio de los dos píxeles verdes
    del mosaico 2x2 (cuando aplica).

    bayer_pattern: "RGGB" | "BGGR" | "GRBG" | "GBRG"
    """
    if u16.dtype != np.uint16:
        raise ValueError("bayer_green_u8_from_u16 espera uint16")

    p = bayer_pattern.upper().strip()

    # Coordenadas de verdes en una celda 2x2:
    # RGGB: R G / G B  -> verdes en (0,1) y (1,0)
    # BGGR: B G / G R  -> verdes en (0,1) y (1,0)
    # GRBG: G R / B G  -> verdes en (0,0) y (1,1)
    # GBRG: G B / R G  -> verdes en (0,0) y (1,1)
    if p in ("RGGB", "BGGR"):
        g1 = u16[0::2, 1::2]
        g2 = u16[1::2, 0::2]
    elif p in ("GRBG", "GBRG"):
        g1 = u16[0::2, 0::2]
        g2 = u16[1::2, 1::2]
    else:
        # fallback conservador
        g1 = u16[0::2, 1::2]
        g2 = u16[1::2, 0::2]

    # promedio en 16-bit (evita overflow usando uint32)
    g = ((g1.astype(np.uint32) + g2.astype(np.uint32)) // 2).astype(np.uint16)
    return (g >> 8).astype(np.uint8, copy=False)


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
    "apply_hotpixel_mask_replace",
    "bayer_green_u8_from_u16",
    "debayer_cv2",
    "warp_rgb16",
    "half_to_full_shift",
    "half_affine_to_full",
]
