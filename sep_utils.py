from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import sep

from imaging import ensure_raw16_bayer


def sep_detect_from_raw16(
    raw16_hp: np.ndarray,
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
        raw16_hp: uint16 2D array (hotpixel-corrected Bayer).
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
    raw = ensure_raw16_bayer(raw16_hp)
    img = raw.astype(np.float32, copy=False)

    bkg = sep.Background(img, bw=int(sep_bw), bh=int(sep_bh))
    img_sub = img - bkg.back()
    img_det = np.maximum(img_sub, 0.0, out=img_sub)

    thresh = float(sep_thresh_sigma) * float(bkg.globalrms)
    objects = sep.extract(img_det, thresh, minarea=int(sep_minarea))

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


__all__ = ["sep_detect_from_raw16"]
