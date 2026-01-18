# imaging.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import cv2


def downsample_u16(u16: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample barato para uint16. Para preview/tracking, lo más barato y estable
    es stride slicing cuando factor es 2, 3, 4, etc.

    Nota: esto NO hace anti-alias. Es intencional (performance first).
    """
    if factor <= 1:
        return u16
    return u16[::factor, ::factor]


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
    ds: int = 2,
    plo: float = 5.0,
    phi: float = 99.5,
    jpeg_quality: int = 75,
    sample_stride: int = 4,
) -> Tuple[bytes, np.ndarray]:
    """
    Pipeline compacto para preview:
      img(u16/u8) -> u8 -> downsample por stride -> stretch -> jpeg

    Retorna:
      (jpeg_bytes, u8_preview_used)
    """
    u8 = to_u8_preview(img)

    if ds > 1:
        u8 = u8[::ds, ::ds]

    u8s = stretch_fast_u8(u8, plo=plo, phi=phi, sample_stride=sample_stride)
    jpg = encode_jpeg(u8s, quality=jpeg_quality)
    return jpg, u8s


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

