# preview.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2


def stretch_to_u8(img: np.ndarray, plo: float = 1.0, phi: float = 99.0, gamma: float = 1.0) -> np.ndarray:
    """
    Stretch robusto para preview. Soporta (H,W) o (H,W,3) float32.
    """
    x = img.astype(np.float32, copy=False)

    if x.ndim == 2:
        lo, hi = np.percentile(x, (plo, phi))
        y = (x - lo) / max(1e-6, (hi - lo))
        y = np.clip(y, 0.0, 1.0)
        if gamma != 1.0:
            y = y ** (1.0 / gamma)
        return (y * 255.0).astype(np.uint8)

    out = np.empty_like(x, dtype=np.uint8)
    for c in range(3):
        xc = x[..., c]
        lo, hi = np.percentile(xc, (plo, phi))
        y = (xc - lo) / max(1e-6, (hi - lo))
        y = np.clip(y, 0.0, 1.0)
        if gamma != 1.0:
            y = y ** (1.0 / gamma)
        out[..., c] = (y * 255.0).astype(np.uint8)
    return out


def to_u8_preview(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen u8/u16 a u8 para preview de forma barata.

    - Si img es uint8: se retorna tal cual.
    - Si img es uint16: se usa (img >> 8) para mapear 0..65535 -> 0..255.
      (Equivalente a tomar el byte alto; muy rápido).
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        # shift produce uint16; casteo final a uint8
        return (img >> 8).astype(np.uint8, copy=False)
    # fallback: convertir con clip
    x = img.astype(np.float32, copy=False)
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def stretch_fast_u8(
    u8: np.ndarray,
    plo: float = 5.0,
    phi: float = 99.5,
    sample_stride: int = 4,
) -> np.ndarray:
    """
    Stretch rápido para preview (u8 -> u8), usando percentiles sobre un submuestreo.

    - sample_stride controla el costo de percentil:
        4 => usa 1/16 de los píxeles (aprox).
    """
    if u8.dtype != np.uint8:
        raise ValueError("stretch_fast_u8 espera uint8")

    if u8.size == 0:
        return u8

    if sample_stride > 1 and u8.shape[0] >= 64 and u8.shape[1] >= 64:
        samp = u8[::sample_stride, ::sample_stride]
    else:
        samp = u8

    lo = float(np.percentile(samp, plo))
    hi = float(np.percentile(samp, phi))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 255.0

    if hi <= lo + 1.0:
        hi = lo + 1.0

    # mapeo lineal: (x - lo) * 255/(hi-lo)
    y = (u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
    y = np.clip(y, 0.0, 255.0)
    return y.astype(np.uint8)


def encode_jpeg(u8: np.ndarray, quality: int = 75) -> bytes:
    """
    Encode JPEG rápido para ipywidgets.Image.

    - u8: HxW o HxWx3 uint8.
    - quality: 1..100 (típico 60-85)
    """
    if u8.dtype != np.uint8:
        raise ValueError("encode_jpeg espera uint8")

    q = int(quality)
    if q < 1:
        q = 1
    if q > 100:
        q = 100

    ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return b""
    return buf.tobytes()


def make_preview_jpeg(
    img: np.ndarray,
    plo: float = 5.0,
    phi: float = 99.5,
    jpeg_quality: int = 75,
    sample_stride: int = 4,
) -> Tuple[bytes, np.ndarray]:
    """
    Pipeline compacto para preview:
      img(u16/u8) -> u8 -> stretch -> jpeg

    Retorna:
      (jpeg_bytes, u8_preview_used)
    """
    u8 = to_u8_preview(img)

    u8s = stretch_fast_u8(u8, plo=plo, phi=phi, sample_stride=sample_stride)
    jpg = encode_jpeg(u8s, quality=jpeg_quality)
    return jpg, u8s


__all__ = [
    "stretch_to_u8",
    "to_u8_preview",
    "stretch_fast_u8",
    "encode_jpeg",
    "make_preview_jpeg",
]
