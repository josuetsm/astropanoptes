# imaging.py
from __future__ import annotations

from typing import Tuple

import math
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


def estimate_sensor_drift_from_stack(
    stack: np.ndarray,
    *,
    fps: float = 10.0,
    window: int = 60,
    median_k: int = 3,
    smooth_k: int = 20,
    vmax_px_s: float = 30.0,
    margin_px: float = 10.0,
    max_shift_cap: int = 200,
    profile_q: float | None = None,     # None => max(); e.g. 99.7 => percentile
    use_subpixel: bool = False,         # recomendado False (rapido)
    return_per_window: bool = False,
) -> dict:
    """
    Estima deriva promedio en el sensor (vx, vy) en px/s a partir de un stack (N,H,W).

    Metodo:
      - median blur (igual que hotpixels)
      - colapsa a perfiles 1D: dy => max/percentil sobre x; dx => max/percentil sobre y
      - suaviza perfil con moving-average
      - correlacion cruzada 1D con gating por max_shift(window)
      - (opcional) refinamiento parabolico subpixel en el pico

    Convencion:
      - vx > 0: movimiento hacia +x (derecha)
      - vy > 0: movimiento hacia +y (abajo en array). Si tu interpretas +y arriba,
        simplemente invierte el signo al usarlo.

    Notas:
      - window=60 es el baseline principal de robustez.
      - vmax_px_s=30.0 asume sensor quieto (gating agresivo).
    """
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError(f"stack must be (N,H,W), got {arr.shape}")
    if arr.dtype != np.uint16:
        raise TypeError(f"stack must be uint16, got {arr.dtype}")

    n, h, w = arr.shape
    if n <= int(window):
        raise ValueError(f"need n > window, got n={n} window={window}")
    if float(fps) <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps}")

    dt = 1.0 / float(fps)
    smooth_k = int(max(1, smooth_k))
    kernel = np.ones(smooth_k, dtype=np.float64) / float(smooth_k)

    # max shift derivado (funcion de window)
    max_shift = int(math.ceil(vmax_px_s * (float(window) / float(fps)) + margin_px))
    max_shift = int(min(max_shift, max_shift_cap))

    def profile_1d(img_u16: np.ndarray, which: str) -> np.ndarray:
        img = median_prefilter_raw16(img_u16, ksize=int(median_k))

        if which == "dy":
            if profile_q is None:
                p = img.max(axis=1).astype(np.float64, copy=False)
            else:
                p = np.percentile(img, profile_q, axis=1).astype(np.float64, copy=False)
        elif which == "dx":
            if profile_q is None:
                p = img.max(axis=0).astype(np.float64, copy=False)
            else:
                p = np.percentile(img, profile_q, axis=0).astype(np.float64, copy=False)
        else:
            raise ValueError("which must be 'dx' or 'dy'")

        return np.convolve(p, kernel, mode="same")

    def subpixel_peak_parabola(y_m1, y0, y_p1) -> float:
        denom = (y_m1 - 2.0 * y0 + y_p1)
        if abs(denom) < 1e-15:
            return 0.0
        return 0.5 * (y_m1 - y_p1) / denom

    def shift_1d_gated(p0: np.ndarray, p1: np.ndarray) -> float:
        a = p1 - p1.mean()
        b = p0 - p0.mean()

        corr = np.correlate(a, b, mode="full")
        L = len(p1)
        shifts = np.arange(-(L - 1), (L - 1) + 1)

        mask = (shifts >= -max_shift) & (shifts <= max_shift)
        corr_m = corr.copy()
        corr_m[~mask] = -np.inf

        i0 = int(np.argmax(corr_m))
        shift_int = i0 - (L - 1)

        if not use_subpixel:
            return float(shift_int)

        if 1 <= i0 < (len(corr) - 1):
            delta = subpixel_peak_parabola(corr[i0 - 1], corr[i0], corr[i0 + 1])
        else:
            delta = 0.0

        return float(shift_int + delta)

    # precompute 1D profiles (ahorra recomputo por ventana)
    pdy_all = [profile_1d(arr[i], "dy") for i in range(n)]
    pdx_all = [profile_1d(arr[i], "dx") for i in range(n)]

    vx_list = []
    vy_list = []

    for i in range(0, n - int(window)):
        j = i + int(window)

        dy = shift_1d_gated(pdy_all[i], pdy_all[j])
        dx = shift_1d_gated(pdx_all[i], pdx_all[j])

        vx = dx / (float(window) * dt)
        vy = dy / (float(window) * dt)

        vx_list.append(vx)
        vy_list.append(vy)

    vx_arr = np.asarray(vx_list, dtype=np.float64)
    vy_arr = np.asarray(vy_list, dtype=np.float64)

    out = {
        "vx_mean": float(vx_arr.mean()),
        "vy_mean": float(vy_arr.mean()),
        "vx_std": float(vx_arr.std(ddof=1)) if vx_arr.size > 1 else 0.0,
        "vy_std": float(vy_arr.std(ddof=1)) if vy_arr.size > 1 else 0.0,
        "n_frames": int(n),
        "n_windows": int(vx_arr.size),
        "params": {
            "fps": float(fps),
            "window": int(window),
            "median_k": int(median_k),
            "smooth_k": int(smooth_k),
            "vmax_px_s": float(vmax_px_s),
            "margin_px": float(margin_px),
            "max_shift_cap": int(max_shift_cap),
            "max_shift_used": int(max_shift),
            "profile_q": profile_q,
            "use_subpixel": bool(use_subpixel),
        },
    }

    if return_per_window:
        out["vx_per_window"] = vx_arr
        out["vy_per_window"] = vy_arr

    return out


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
    "estimate_sensor_drift_from_stack",
    "warp_rgb16",
    "half_to_full_shift",
    "half_affine_to_full",
]
