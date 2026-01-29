# tracking_preprocess.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from imaging import ensure_raw16_bayer
from sep_utils import sep_detect_from_raw16


def build_tracking_observations(
    raw16: np.ndarray,
    *,
    sep_cfg: Any,
    prefilter_ksize: int = 3,
    max_sources: Optional[int] = None,
    compute_saturation: bool = True,
) -> Dict[str, Any]:
    """
    Build minimal tracking observations from a RAW16 frame.

    Returns a dict with:
      - obj_xy: (N,2) float64
      - star_count: int
      - saturation_frac: float
      - img_det: float32 detection image
    """
    raw = ensure_raw16_bayer(raw16)
    img_det, _bkg, _objects, obj_xy = sep_detect_from_raw16(
        raw,
        sep_bw=int(sep_cfg.bw),
        sep_bh=int(sep_cfg.bh),
        sep_thresh_sigma=float(sep_cfg.thresh_sigma),
        sep_minarea=int(sep_cfg.minarea),
        max_sources=max_sources,
        prefilter_ksize=prefilter_ksize,
    )
    star_count = int(obj_xy.shape[0])
    saturation_frac = 0.0
    if compute_saturation:
        max_val = np.iinfo(raw.dtype).max
        saturation_frac = float(np.mean(raw >= max_val))
    return {
        "obj_xy": obj_xy,
        "star_count": star_count,
        "saturation_frac": saturation_frac,
        "img_det": img_det,
    }
