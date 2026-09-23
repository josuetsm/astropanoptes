from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
import sep
from imaging import ensure_raw16_bayer, median_prefilter_raw16


_SEP_EXTRACT_LOCK = threading.Lock()
_SEP_PIXSTACK = 300_000
_SEP_SUB_OBJECTS = 1024          # SEP's own default


def _is_pixstack_error(exc: BaseException) -> bool:
    text = str(exc).strip().lower()
    return "pixel buffer full" in text or "pixstack" in text


def _is_deblend_error(exc: BaseException) -> bool:
    """SEP's *other* capacity limit, separate from the pixel stack.

    A frame full of hot pixels or a mosaic with a ragged border can push the
    deblender past its sub-object limit. Unlike the pixstack error this one used
    to escape uncaught, and since detection is the first step of a solve, it
    failed the whole plate solve rather than degrading.
    """
    text = str(exc).strip().lower()
    return "deblend" in text and ("overflow" in text or "sub-object" in text or "limit" in text)


def _extract_with_crowding_recovery(
    img_det: np.ndarray,
    *,
    threshold: float,
    minarea: int,
) -> np.ndarray:
    """Retry a crowded SEP extraction without letting it kill a worker.

    A normal stellar image succeeds on the first pass.  Bright terrestrial or
    strongly structured frames can exceed SEP's 300k active-pixel stack; for
    those frames, progressively retain only the strongest sources.  If even
    that is insufficient, grow the global stack to the image size and make one
    final bounded attempt.
    """
    global _SEP_PIXSTACK, _SEP_SUB_OBJECTS
    last_error: Optional[BaseException] = None
    threshold_factors = (1.0, 1.5, 2.25, 3.5)
    with _SEP_EXTRACT_LOCK:
        for factor in threshold_factors:
            try:
                return sep.extract(
                    img_det,
                    float(threshold) * float(factor),
                    minarea=int(minarea),
                )
            except Exception as exc:
                if _is_deblend_error(exc):
                    # Raise the deblender's ceiling once and retry the same
                    # threshold: the frame is legitimately crowded, not wrong.
                    if _SEP_SUB_OBJECTS < 8192:
                        _SEP_SUB_OBJECTS = 8192
                        sep.set_sub_object_limit(_SEP_SUB_OBJECTS)
                        try:
                            return sep.extract(
                                img_det,
                                float(threshold) * float(factor),
                                minarea=int(minarea),
                            )
                        except Exception as exc2:
                            if not (_is_pixstack_error(exc2) or _is_deblend_error(exc2)):
                                raise
                            last_error = exc2
                            continue
                    last_error = exc
                    continue
                if not _is_pixstack_error(exc):
                    raise
                last_error = exc

        required = max(300_000, int(img_det.size) + 1024)
        if required > int(_SEP_PIXSTACK):
            sep.set_extract_pixstack(int(required))
            _SEP_PIXSTACK = int(required)
        try:
            # The previous retries increased the threshold only to reduce the
            # active-pixel count while using the old global pixstack. Once the
            # stack is large enough, retry the requested threshold so dense
            # but valid stellar fields do not silently lose faint sources.
            return sep.extract(
                img_det,
                float(threshold),
                minarea=int(minarea),
            )
        except Exception as exc:
            if not (_is_pixstack_error(exc) or _is_deblend_error(exc)):
                raise
            last_error = exc

        # Last resort: pick the threshold from the pixel distribution instead of
        # from the background RMS.
        #
        # Scaling the RMS assumes the frame is mostly sky. When it is not — a
        # daylight or badly-lit frame where a third of the pixels sit above the
        # background and the RMS collapses to a fraction of an ADU — every
        # multiple of it still selects most of the image, and the deblender
        # drowns no matter how high the ceiling is raised. A high percentile is
        # bounded by construction: it keeps a known small fraction of pixels.
        for pct in (99.9, 99.99):
            try:
                return sep.extract(
                    img_det,
                    float(np.percentile(img_det, pct)),
                    minarea=int(minarea),
                )
            except Exception as exc:
                if not (_is_pixstack_error(exc) or _is_deblend_error(exc)):
                    raise
                last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("SEP extraction failed without an error")


_BAD_PIXEL_MAP: Optional[np.ndarray] = None
_BAD_PIXEL_MAP_TRIED = False
_BAD_PIXEL_PATH = "calibration_frames/bad_pixel_map.npy"


def load_bad_pixel_map(path: Optional[str] = None) -> Optional[np.ndarray]:
    """Boolean map of known-defective sensor pixels, or None if unavailable.

    Built by ``scripts/pixel_diagnostics.py`` from dark frames and repeated
    sky detections. Cached after the first read; the file rarely changes and
    detection runs on every frame.
    """
    global _BAD_PIXEL_MAP, _BAD_PIXEL_MAP_TRIED
    if path is not None:
        try:
            return np.asarray(np.load(path)).astype(bool)
        except Exception:
            return None
    if _BAD_PIXEL_MAP_TRIED:
        return _BAD_PIXEL_MAP
    _BAD_PIXEL_MAP_TRIED = True
    try:
        _BAD_PIXEL_MAP = np.asarray(np.load(_BAD_PIXEL_PATH)).astype(bool)
    except Exception:
        _BAD_PIXEL_MAP = None
    return _BAD_PIXEL_MAP


def set_bad_pixel_map(mask: Optional[np.ndarray]) -> None:
    """Override the cached map (None disables masking)."""
    global _BAD_PIXEL_MAP, _BAD_PIXEL_MAP_TRIED
    _BAD_PIXEL_MAP = None if mask is None else np.asarray(mask).astype(bool)
    _BAD_PIXEL_MAP_TRIED = True


def repair_bad_pixels(raw16: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Replace defective pixels with a local median of their neighbours.

    A hot pixel is a single bright sample sitting on the sky, which is exactly
    what a faint star looks like to a detector. Left alone they enter the
    brightest-N list the plate solver works from and waste triplet trials on
    geometry that no catalogue can match. Substituting the local median keeps
    the frame's statistics intact, and because the map only covers ~0.03% of
    the sensor no real signal is at risk.

    Only same-shape masks are applied, so a ROI or binning change simply skips
    the repair instead of corrupting the frame.
    """
    if mask is None:
        mask = load_bad_pixel_map()
    if mask is None or mask.shape != raw16.shape or not mask.any():
        return raw16
    import cv2

    out = raw16.copy()
    # 5x5 median: wide enough that adjacent defects do not feed each other.
    med = cv2.medianBlur(raw16, 5)
    out[mask] = med[mask]
    return out


def sep_detect_from_raw16(
    raw16: np.ndarray,
    *,
    sep_bw: int,
    sep_bh: int,
    sep_thresh_sigma: float,
    sep_minarea: int,
    max_sources: Optional[int] = None,
    repair_defects: bool = True,
) -> Tuple[np.ndarray, sep.Background, np.ndarray, np.ndarray]:
    """
    Detect sources from a RAW16 Bayer frame using SEP.

    Args:
        raw16: uint16 2D array (RAW16 Bayer).
        sep_bw: Background box width.
        sep_bh: Background box height.
        sep_thresh_sigma: Threshold multiplier for global RMS.
        sep_minarea: Minimum source area.
        max_sources: Optional cap on number of detections (sorted by flux desc).

    Returns:
        img_det: float32 detection image (background-subtracted, >=0).
        bkg: sep.Background object.
        objects: structured array of detected objects.
        obj_xy: (N,2) float64 array of x,y positions.
    """
    raw = ensure_raw16_bayer(raw16)
    if repair_defects:
        raw = repair_bad_pixels(raw)
    img_med = median_prefilter_raw16(raw, ksize=3)

    bkg = sep.Background(img_med, bw=int(sep_bw), bh=int(sep_bh))
    img_sub = img_med - bkg.back()
    img_det = np.maximum(img_sub, 0.0, out=img_sub)

    thresh = float(sep_thresh_sigma) * float(bkg.globalrms)
    objects = _extract_with_crowding_recovery(
        img_det,
        threshold=thresh,
        minarea=int(sep_minarea),
    )

    if objects is None or len(objects) == 0:
        obj_xy = np.zeros((0, 2), dtype=np.float64)
        empty_objects = np.zeros((0,), dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")])
        return img_det.astype(np.float32, copy=False), bkg, empty_objects, obj_xy

    order = np.argsort(-objects["flux"].astype(np.float64))
    objects = objects[order]

    if max_sources is not None:
        n_use = min(int(max_sources), len(objects))
        objects = objects[:n_use]

    x = objects["x"].astype(np.float64)
    y = objects["y"].astype(np.float64)
    obj_xy = np.column_stack([x, y])

    return img_det.astype(np.float32, copy=False), bkg, objects, obj_xy


def estimate_shift_from_objects(
    ref_xy: np.ndarray,
    cur_xy: np.ndarray,
    *,
    max_shift_px: float,
) -> Tuple[float, float, float, int]:
    """
    Estimate translation to align cur_xy onto ref_xy using nearest-neighbor shifts.

    Returns (dx, dy, resp, n_matches) where shifting cur_xy by (dx, dy) best aligns to ref_xy.
    resp is the match ratio in [0,1].
    """
    ref = np.asarray(ref_xy, dtype=np.float64)
    cur = np.asarray(cur_xy, dtype=np.float64)

    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"ref_xy must have shape (N,2), got {ref.shape}")
    if cur.ndim != 2 or cur.shape[1] != 2:
        raise ValueError(f"cur_xy must have shape (N,2), got {cur.shape}")

    if ref.size == 0 or cur.size == 0:
        return 0.0, 0.0, 0.0, 0

    diff = ref[None, :, :] - cur[:, None, :]
    dist2 = np.sum(diff ** 2, axis=2)
    nn_idx = np.argmin(dist2, axis=1)
    min_dist = np.sqrt(dist2[np.arange(cur.shape[0]), nn_idx])

    max_shift = float(max_shift_px)
    good = min_dist <= max_shift
    if not np.any(good):
        return 0.0, 0.0, 0.0, 0

    shifts = diff[np.arange(cur.shape[0]), nn_idx]
    shifts = shifts[good]
    dx = float(np.median(shifts[:, 0]))
    dy = float(np.median(shifts[:, 1]))
    matches = int(shifts.shape[0])
    denom = float(max(1, min(ref.shape[0], cur.shape[0])))
    resp = float(matches / denom)
    return dx, dy, resp, matches


__all__ = ["sep_detect_from_raw16", "estimate_shift_from_objects"]
