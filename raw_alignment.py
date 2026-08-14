from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from imaging import ensure_raw16_bayer


@dataclass(frozen=True)
class RawAlignmentSignature:
    """Compact representation used to register RAW16 frames.

    The broad profiles provide sub-pixel precision.  The detail profiles remove
    the slowly varying envelope of the per-axis maxima, which prevents two
    unrelated noise frames from looking like a high-confidence match.
    """

    shape: Tuple[int, int]
    broad_x: np.ndarray
    broad_y: np.ndarray
    detail_x: np.ndarray
    detail_y: np.ndarray
    feature_count: int

    @property
    def has_signal(self) -> bool:
        return bool(
            np.linalg.norm(self.detail_x) > 1e-6
            and np.linalg.norm(self.detail_y) > 1e-6
        )


@dataclass(frozen=True)
class RawAlignmentResult:
    ok: bool
    dx: float = 0.0
    dy: float = 0.0
    response: float = 0.0
    broad_response: float = 0.0
    detail_response: float = 0.0
    reason: str = "low_confidence"


def _odd_ksize(value: int, *, minimum: int = 1) -> int:
    k = max(int(value), int(minimum))
    return k if (k % 2) == 1 else k + 1


def _smooth_1d(profile: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float32).reshape(1, -1)
    k = max(1, min(int(width), int(values.shape[1])))
    if k <= 1:
        return values.reshape(-1).copy()
    return cv2.blur(values, (k, 1), borderType=cv2.BORDER_REFLECT).reshape(-1)


def _axis_profiles(
    profile: np.ndarray, *, smooth_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    smooth_width = max(1, int(smooth_k))
    broad = _smooth_1d(profile, smooth_width)

    # Stars occupy only a few pixels along either axis.  A short smoothing pass
    # retains them, while the much wider pass models gradients and the common
    # envelope of read noise extrema.
    detail_width = max(1, int(round(smooth_width / 4.0)))
    detail_base_width = max(9, (4 * smooth_width) + 1)
    detail_base_width = min(detail_base_width, int(np.asarray(profile).size))
    detail_narrow = _smooth_1d(profile, detail_width)
    detail_background = _smooth_1d(detail_narrow, detail_base_width)
    detail = np.maximum(detail_narrow - detail_background, 0.0)
    return broad.astype(np.float32, copy=False), detail.astype(np.float32, copy=False)


def _count_profile_peaks(profile: np.ndarray) -> int:
    values = np.asarray(profile, dtype=np.float32)
    if values.size < 3 or not np.any(values > 0.0):
        return 0
    positive = values[values > 0.0]
    baseline = float(np.median(positive)) if positive.size else 0.0
    mad = float(np.median(np.abs(positive - baseline))) if positive.size else 0.0
    threshold = max(baseline + 4.0 * max(mad, 1e-6), 0.08 * float(values.max()))
    peaks = (
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= threshold)
    )
    return int(np.count_nonzero(peaks))


def build_raw_alignment_signature(
    raw16: np.ndarray,
    *,
    median_k: int = 3,
    smooth_k: int = 30,
) -> RawAlignmentSignature:
    """Build an O(H*W) alignment signature directly from Bayer RAW16 data."""

    raw = ensure_raw16_bayer(raw16)
    median_width = _odd_ksize(median_k, minimum=1)
    # OpenCV only supports large median kernels for 8-bit images.  RAW16 uses
    # the optimized 3x3/5x5 path; larger configured values are safely capped.
    median_width = min(median_width, 5)
    filtered = raw if median_width <= 1 else cv2.medianBlur(raw, median_width)

    profile_x = filtered.max(axis=0).astype(np.float32, copy=False)
    profile_y = filtered.max(axis=1).astype(np.float32, copy=False)
    broad_x, detail_x = _axis_profiles(profile_x, smooth_k=int(smooth_k))
    broad_y, detail_y = _axis_profiles(profile_y, smooth_k=int(smooth_k))
    feature_count = min(_count_profile_peaks(detail_x), _count_profile_peaks(detail_y))

    return RawAlignmentSignature(
        shape=(int(raw.shape[0]), int(raw.shape[1])),
        broad_x=broad_x,
        broad_y=broad_y,
        detail_x=detail_x,
        detail_y=detail_y,
        feature_count=int(feature_count),
    )


def _shift_1d_centered(
    ref_profile: np.ndarray,
    cur_profile: np.ndarray,
    *,
    center: float,
    max_shift: int,
    subpixel: bool,
) -> Tuple[float, float]:
    ref = np.asarray(ref_profile, dtype=np.float64)
    cur = np.asarray(cur_profile, dtype=np.float64)
    if ref.shape != cur.shape or ref.ndim != 1 or ref.size == 0:
        return 0.0, 0.0

    ref = ref - float(ref.mean())
    cur = cur - float(cur.mean())
    norm = float(np.linalg.norm(ref) * np.linalg.norm(cur))
    if not np.isfinite(norm) or norm <= 1e-12:
        return 0.0, 0.0

    length = int(cur.size)
    center_use = float(center) if np.isfinite(center) else 0.0
    radius = max(1, int(max_shift))
    shift_min = max(-(length - 1), int(np.ceil(center_use - radius)))
    shift_max = min(length - 1, int(np.floor(center_use + radius)))
    if shift_min > shift_max:
        return 0.0, 0.0

    # Only evaluate the requested search window.  A full correlation does
    # roughly W^2 work even though tracking normally searches just a few dozen
    # pixels around its motion prediction.
    shifts = np.arange(shift_min, shift_max + 1, dtype=np.int32)
    corr = np.empty(shifts.size, dtype=np.float64)
    for index, shift_value in enumerate(shifts):
        shift = int(shift_value)
        if shift < 0:
            corr[index] = float(np.dot(cur[:shift], ref[-shift:]))
        elif shift > 0:
            corr[index] = float(np.dot(cur[shift:], ref[:-shift]))
        else:
            corr[index] = float(np.dot(cur, ref))

    peak_index = int(np.argmax(corr))
    if not np.isfinite(corr[peak_index]):
        return 0.0, 0.0
    shift_int = int(shifts[peak_index])
    response = float(np.clip(float(corr[peak_index]) / norm, 0.0, 1.0))

    delta = 0.0
    if subpixel and 1 <= peak_index < (corr.size - 1):
        left = float(corr[peak_index - 1])
        peak = float(corr[peak_index])
        right = float(corr[peak_index + 1])
        denominator = left - (2.0 * peak) + right
        if (
            all(np.isfinite(v) for v in (left, peak, right, denominator))
            and abs(denominator) > 1e-12
        ):
            candidate = 0.5 * (left - right) / denominator
            if np.isfinite(candidate) and abs(candidate) <= 1.0:
                delta = float(candidate)

    shift = float(shift_int + delta)
    return (shift if np.isfinite(shift) else float(shift_int)), response


def estimate_raw_translation(
    reference: RawAlignmentSignature,
    current: RawAlignmentSignature,
    *,
    center_dx: float = 0.0,
    center_dy: float = 0.0,
    search_radius_px: float = 50.0,
    max_displacement_px: Optional[float] = None,
    min_response: float = 0.25,
    use_subpixel: bool = True,
    max_profile_disagreement_px: float = 3.0,
) -> RawAlignmentResult:
    """Estimate current-frame displacement relative to a reference signature."""

    if reference.shape != current.shape:
        return RawAlignmentResult(ok=False, reason="shape_changed")
    if not reference.has_signal or not current.has_signal:
        return RawAlignmentResult(ok=False, reason="no_signal")

    radius = max(1, int(round(float(search_radius_px))))
    broad_dx, broad_rx = _shift_1d_centered(
        reference.broad_x,
        current.broad_x,
        center=float(center_dx),
        max_shift=radius,
        subpixel=bool(use_subpixel),
    )
    broad_dy, broad_ry = _shift_1d_centered(
        reference.broad_y,
        current.broad_y,
        center=float(center_dy),
        max_shift=radius,
        subpixel=bool(use_subpixel),
    )
    detail_dx, detail_rx = _shift_1d_centered(
        reference.detail_x,
        current.detail_x,
        center=float(center_dx),
        max_shift=radius,
        subpixel=bool(use_subpixel),
    )
    detail_dy, detail_ry = _shift_1d_centered(
        reference.detail_y,
        current.detail_y,
        center=float(center_dy),
        max_shift=radius,
        subpixel=bool(use_subpixel),
    )

    broad_response = float(min(broad_rx, broad_ry))
    detail_response = float(min(detail_rx, detail_ry))
    response = float(min(broad_response, detail_response))
    # The high-pass detail profiles retain the narrow stellar peaks and give a
    # less biased sub-pixel position.  Broad profiles remain an independent
    # confidence/consistency guard.
    dx = float(detail_dx)
    dy = float(detail_dy)

    if not all(np.isfinite(v) for v in (dx, dy, response)):
        return RawAlignmentResult(ok=False, reason="non_finite")

    displacement = float(np.hypot(dx, dy))
    if max_displacement_px is not None and displacement > float(max_displacement_px):
        return RawAlignmentResult(
            ok=False,
            dx=dx,
            dy=dy,
            response=response,
            broad_response=broad_response,
            detail_response=detail_response,
            reason="shift_out_of_range",
        )

    disagreement = float(
        np.hypot(broad_dx - detail_dx, broad_dy - detail_dy)
    )
    if disagreement > max(0.5, float(max_profile_disagreement_px)):
        return RawAlignmentResult(
            ok=False,
            dx=dx,
            dy=dy,
            response=response,
            broad_response=broad_response,
            detail_response=detail_response,
            reason="ambiguous_alignment",
        )

    if response < float(min_response):
        return RawAlignmentResult(
            ok=False,
            dx=dx,
            dy=dy,
            response=response,
            broad_response=broad_response,
            detail_response=detail_response,
            reason="low_confidence",
        )

    return RawAlignmentResult(
        ok=True,
        dx=dx,
        dy=dy,
        response=response,
        broad_response=broad_response,
        detail_response=detail_response,
        reason="ok",
    )


__all__ = [
    "RawAlignmentResult",
    "RawAlignmentSignature",
    "build_raw_alignment_signature",
    "estimate_raw_translation",
]
