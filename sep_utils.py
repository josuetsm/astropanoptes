from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
import sep
from imaging import ensure_raw16_bayer, median_prefilter_raw16


_SEP_EXTRACT_LOCK = threading.Lock()
_SEP_PIXSTACK = 300_000


def _is_pixstack_error(exc: BaseException) -> bool:
    text = str(exc).strip().lower()
    return "pixel buffer full" in text or "pixstack" in text


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
    global _SEP_PIXSTACK
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
            if not _is_pixstack_error(exc):
                raise
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("SEP extraction failed without an error")


def sep_detect_from_raw16(
    raw16: np.ndarray,
    *,
    sep_bw: int,
    sep_bh: int,
    sep_thresh_sigma: float,
    sep_minarea: int,
    max_sources: Optional[int] = None,
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
