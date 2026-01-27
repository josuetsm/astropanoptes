"""Hot pixel detection and masking utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional, List

import cv2
import numpy as np


_UINT16_MAX = np.iinfo(np.uint16).max


def _ensure_2d(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}.")
    return img


def hotpix_prefilter_base(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median prefilter for hot pixel detection.

    Args:
        img: 2D image array (uint8, uint16, or float).
        ksize: Median blur kernel size (odd integer).

    Returns:
        Median-filtered image in float32.
    """
    img = _ensure_2d(np.asarray(img))
    if img.dtype == np.uint8:
        img_u16 = img.astype(np.uint16) * 257
    elif img.dtype == np.uint16:
        img_u16 = img
    elif np.issubdtype(img.dtype, np.floating):
        img_u16 = np.clip(np.rint(img), 0, _UINT16_MAX).astype(np.uint16)
    else:
        raise TypeError(f"Unsupported dtype: {img.dtype}.")

    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize must be a positive odd integer.")

    base = cv2.medianBlur(img_u16, ksize)
    return base.astype(np.float32)


def _mad(data: np.ndarray) -> float:
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return float(mad)


def build_hotpixel_mask(
    frames_u16: Iterable[np.ndarray],
    ksize: int = 3,
    thr_k: float = 8.0,
    min_hits_frac: float = 0.7,
    max_component_area: int = 4,
) -> np.ndarray:
    """Build a hot-pixel mask from a sequence of frames.

    Args:
        frames_u16: Iterable of 2D uint16 frames.
        ksize: Median blur kernel size.
        thr_k: Threshold multiplier for MAD-based thresholding.
        min_hits_frac: Minimum fraction of frames that must trigger a hit.
        max_component_area: Maximum connected component area to keep.

    Returns:
        Boolean mask of hot pixels (True for hot pixels).
    """
    frames = [np.asarray(frame) for frame in frames_u16]
    if not frames:
        raise ValueError("frames_u16 must contain at least one frame.")

    frames = [_ensure_2d(frame) for frame in frames]
    if any(frame.dtype != np.uint16 for frame in frames):
        raise TypeError("All frames must be uint16.")

    hits = np.zeros(frames[0].shape, dtype=np.uint16)

    for frame in frames:
        base = hotpix_prefilter_base(frame, ksize=ksize)
        spike = frame.astype(np.float32) - base
        median = float(np.median(spike))
        mad = _mad(spike)
        thr = median + thr_k * 1.4826 * mad
        hits += spike > thr

    hits_frac = hits.astype(np.float32) / float(len(frames))
    mask = hits_frac >= float(min_hits_frac)

    if max_component_area is not None and max_component_area > 0:
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num_labels > 1:
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > max_component_area:
                    mask[labels == label] = False

    return mask


def build_hotpixel_mask_temporal(
    frames_u16: Iterable[np.ndarray],
    *,
    abs_percentile: float = 99.9,
    var_percentile: float = 10.0,
    max_component_area: int = 4,
) -> np.ndarray:
    """Build a hot-pixel mask from a short temporal sequence.

    A hot pixel is defined as persistently bright (high temporal median)
    and stable (low temporal MAD).

    Args:
        frames_u16: Iterable of 2D uint16 frames.
        abs_percentile: Percentile over per-pixel median to set the absolute threshold.
        var_percentile: Percentile over per-pixel MAD to set the stability threshold.
        max_component_area: Maximum connected component area to keep.

    Returns:
        Boolean mask of hot pixels (True for hot pixels).
    """
    frames = [np.asarray(frame) for frame in frames_u16]
    if not frames:
        raise ValueError("frames_u16 must contain at least one frame.")

    frames = [_ensure_2d(frame) for frame in frames]
    if any(frame.dtype != np.uint16 for frame in frames):
        raise TypeError("All frames must be uint16.")

    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)

    thr_abs = float(np.percentile(median, float(abs_percentile)))
    thr_var = float(np.percentile(mad, float(var_percentile)))

    mask = (median >= thr_abs) & (mad <= thr_var)

    if max_component_area is not None and max_component_area > 0:
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num_labels > 1:
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > max_component_area:
                    mask[labels == label] = False

    return mask


def save_hotpixel_mask(mask: np.ndarray, meta: Dict[str, Any], path_base: str) -> Tuple[Path, Path]:
    """Save a hot-pixel mask and metadata to .npy and .json files.

    Args:
        mask: 2D boolean mask.
        meta: Metadata dictionary.
        path_base: Base path without extension.

    Returns:
        Tuple of (npy_path, json_path).
    """
    mask = _ensure_2d(np.asarray(mask)).astype(bool)
    base = Path(path_base)
    npy_path = base.with_suffix(".npy")
    json_path = base.with_suffix(".json")

    meta_out = dict(meta or {})
    meta_out.setdefault("shape", list(mask.shape))
    meta_out.setdefault("roi", None)
    meta_out.setdefault("exp_ms", None)
    meta_out.setdefault("gain", None)
    meta_out.setdefault("fmt", None)
    meta_out.setdefault("camera_model", None)
    meta_out.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    meta_out.setdefault("thr_k", None)
    meta_out.setdefault("min_hits_frac", None)

    np.save(npy_path, mask)
    json_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True))

    return npy_path, json_path


def load_hotpixel_mask(path_base: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a hot-pixel mask and metadata from .npy and .json files.

    Args:
        path_base: Base path without extension.

    Returns:
        Tuple of (mask, metadata).
    """
    base = Path(path_base)
    npy_path = base.with_suffix(".npy")
    json_path = base.with_suffix(".json")

    mask = np.load(npy_path).astype(bool)
    meta = json.loads(json_path.read_text())
    return mask, meta


def apply_hotpixel_mask_replace(img_u16: np.ndarray, mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Replace hot pixels with median-filtered values.

    Args:
        img_u16: 2D uint16 image.
        mask: 2D boolean mask.
        ksize: Median blur kernel size.

    Returns:
        Image with hot pixels replaced.
    """
    img = _ensure_2d(np.asarray(img_u16))
    if img.dtype != np.uint16:
        raise TypeError("img_u16 must be uint16.")

    mask_arr = _ensure_2d(np.asarray(mask)).astype(bool)
    if mask_arr.shape != img.shape:
        raise ValueError("mask shape must match img_u16 shape.")

    base = hotpix_prefilter_base(img, ksize=ksize)
    replacement = np.clip(np.rint(base), 0, _UINT16_MAX).astype(np.uint16)
    out = img.copy()
    out[mask_arr] = replacement[mask_arr]
    return out


def hotpixel_weight_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a hot-pixel boolean mask into a weighting mask.

    Args:
        mask: 2D boolean mask of hot pixels.

    Returns:
        Float32 weight mask with 0 for hot pixels and 1 elsewhere.
    """
    mask_arr = _ensure_2d(np.asarray(mask)).astype(bool)
    weights = np.ones_like(mask_arr, dtype=np.float32)
    weights[mask_arr] = 0.0
    return weights


def _bayer_parity_offsets(window: int = 5) -> List[Tuple[int, int]]:
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be a positive odd integer.")
    radius = window // 2
    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if (dx % 2 == 0) and (dy % 2 == 0):
                offsets.append((dy, dx))
    return offsets


def apply_hotpixel_correction(
    raw16: np.ndarray,
    mask: Optional[np.ndarray],
    bayer_pattern: str,
    *,
    window: int = 5,
) -> np.ndarray:
    """Correct hot pixels using same-color Bayer neighbors.

    Args:
        raw16: 2D uint16 Bayer image.
        mask: 2D boolean mask of hot pixels (True for hot pixels).
        bayer_pattern: Bayer pattern string (RGGB/BGGR/GRBG/GBRG).
        window: Neighborhood window size (odd integer).

    Returns:
        Corrected uint16 image.
    """
    img = _ensure_2d(np.asarray(raw16))
    if img.dtype != np.uint16:
        raise TypeError("raw16 must be uint16.")

    if mask is None:
        return img.copy()

    mask_arr = _ensure_2d(np.asarray(mask)).astype(bool)
    if mask_arr.shape != img.shape:
        raise ValueError("mask shape must match raw16 shape.")

    hot_idx = np.argwhere(mask_arr)
    if hot_idx.size == 0:
        return img.copy()

    out = img.copy()
    offsets = _bayer_parity_offsets(window=window)

    h, w = img.shape
    for y, x in hot_idx:
        vals = []
        y0 = int(y)
        x0 = int(x)
        parity_y = y0 % 2
        parity_x = x0 % 2
        for dy, dx in offsets:
            yy = y0 + dy
            xx = x0 + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue
            if (yy % 2) != parity_y or (xx % 2) != parity_x:
                continue
            if mask_arr[yy, xx]:
                continue
            vals.append(int(img[yy, xx]))
        if vals:
            out[y0, x0] = np.uint16(int(np.median(vals)))
    return out
