"""Hot pixel detection and masking utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
