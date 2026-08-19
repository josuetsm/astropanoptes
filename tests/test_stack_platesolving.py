from __future__ import annotations

import numpy as np

from config import AppConfig
from stacking import (
    LiveMosaicStackerGray,
    StackingWorker,
    _bayer_to_gray_code,
    _bayer_to_rgb_code,
)


def _frame(h=64, w=80, *, bg=1500, stars=((20, 30), (45, 55)), shift=(0, 0)):
    """Synthetic Bayer-ish frame: light-polluted background plus point stars."""
    img = np.full((h, w), bg, dtype=np.uint16)
    for y, x in stars:
        yy, xx = y + shift[0], x + shift[1]
        if 1 <= yy < h - 1 and 1 <= xx < w - 1:
            img[yy - 1:yy + 2, xx - 1:xx + 2] = 9000
            img[yy, xx] = 30000
    return img


def _stacker(**kw):
    params = dict(
        resp_min=0.0,
        align_median_k=3,
        smooth_k=8,
        max_shift_px=40.0,
        use_subpixel=True,
        drizzle_scale=1.0,
        color_mode="mono",
        bayer_to_gray_code=_bayer_to_gray_code("RGGB"),
        bayer_to_rgb_code=_bayer_to_rgb_code("RGGB"),
        preview_log_vmin=5.0,
    )
    params.update(kw)
    return LiveMosaicStackerGray(**params)


def test_preview_keeps_light_polluted_background_dark() -> None:
    """Regression: a fixed log black point washed the preview out.

    With a bright sky background (log1p(1500) = 7.3) the old fixed
    ``preview_log_vmin`` of 5.0 sat *below* the background, so the sky itself
    rendered mid-grey and the whole preview looked luminous.
    """
    st = _stacker()
    st.add_frame(_frame(bg=1500), t_unix=1000.0)
    prev = st.get_preview_u8()

    assert prev is not None
    bg_level = float(np.median(prev))
    assert bg_level < 60.0, f"fondo demasiado claro: {bg_level}"
    # las estrellas siguen destacando
    assert int(prev.max()) > 200


def test_preview_ignores_empty_canvas_border() -> None:
    """The growing mosaic canvas must not drag the stretch statistics."""
    st = _stacker()
    st.add_frame(_frame(bg=1500), t_unix=1000.0)
    st.add_frame(_frame(bg=1500, shift=(6, 9)), t_unix=1001.0)
    prev = st.get_preview_u8()
    assert prev is not None
    # el canvas crecio y las zonas sin datos quedan en negro puro
    if st.canvas_h > st.frame_h or st.canvas_w > st.frame_w:
        assert int(prev.min()) == 0
    bg_level = float(np.median(prev))
    assert bg_level < 60.0


def test_stack_records_reference_time_not_latest() -> None:
    """Frames are aligned onto the first one, so that is the stack's epoch."""
    st = _stacker()
    st.add_frame(_frame(), t_unix=1000.0)
    st.add_frame(_frame(shift=(2, 3)), t_unix=1060.0)
    assert st.ref_time_unix == 1000.0
    assert st.last_time_unix == 1060.0


def test_get_stack_for_solve_exposes_geometry_and_epoch() -> None:
    cfg = AppConfig()
    cfg.stacking.resp_min = 0.0
    worker = StackingWorker(cfg)
    worker.engine.configure_from_cfg()
    worker.engine.enabled = True
    eng = worker.engine._live_gray
    assert eng is not None
    eng.add_frame(_frame(), t_unix=2000.0)
    eng.add_frame(_frame(shift=(4, 5)), t_unix=2030.0)

    info = worker.get_stack_for_solve()
    assert info is not None
    assert info["obstime_unix"] == 2000.0          # epoca = referencia, no "ahora"
    assert info["frames"] >= 1
    assert info["image"].ndim == 2                  # el solver espera (H,W)
    assert info["image"].dtype == np.uint16
    assert info["drizzle_scale"] >= 1.0
    assert info["canvas"][0] >= info["frame_shape"][0]


def test_get_stack_for_solve_returns_none_when_empty() -> None:
    cfg = AppConfig()
    worker = StackingWorker(cfg)
    assert worker.get_stack_for_solve() is None
