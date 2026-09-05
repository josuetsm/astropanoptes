from __future__ import annotations

import threading

import numpy as np
import pytest

from config import AppConfig
from stacking import (
    LiveMosaicStackerGray,
    StackEngine,
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


def _solve_info(st):
    """Stack listo para resolver, sin construir un worker (y sus hilos)."""
    engine = StackEngine.__new__(StackEngine)      # sin __init__: sin hilos
    engine._live_gray = st
    engine._stack_lock = threading.RLock()
    return engine.get_stack_for_solve()


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


def test_solve_image_is_centred_on_the_reference_frame() -> None:
    """The mosaic's centre must land on the pointing, without losing signal.

    Plate solving is told the image is centred near the requested target, and
    that target is where the telescope pointed for the *reference* frame. With
    the mount parked the canvas grows away from it, so the raw mosaic's centre
    drifts off the pointing by half the accumulated drift.
    """
    st = _stacker()
    st.add_frame(_frame(), t_unix=1000.0)
    for i in range(1, 7):                      # deriva sostenida en una diagonal
        st.add_frame(_frame(shift=(3 * i, 4 * i)), t_unix=1000.0 + i)

    # el canvas efectivamente crecio en la direccion de la deriva
    assert st.canvas_h > st.frame_h or st.canvas_w > st.frame_w

    ref_cx = st.ref_origin_x + st.frame_w * 0.5
    ref_cy = st.ref_origin_y + st.frame_h * 0.5
    raw_cx, raw_cy = st.canvas_w * 0.5, st.canvas_h * 0.5
    # el centro crudo del mosaico NO coincide con el del frame de referencia
    assert abs(raw_cx - ref_cx) > 1.0 or abs(raw_cy - ref_cy) > 1.0

    info = _solve_info(st)
    assert info is not None

    sh, sw = info["solve_shape"]
    # tras el relleno, el centro de la imagen SI cae sobre el de referencia
    assert abs(sw * 0.5 - ref_cx) <= 1.0
    assert abs(sh * 0.5 - ref_cy) <= 1.0
    # y no se descarto nada: la imagen entregada contiene todo el mosaico
    assert sh >= st.canvas_h and sw >= st.canvas_w
    assert info["image"].shape == (sh, sw)


def test_recentring_preserves_all_stacked_signal() -> None:
    """Re-centring pads, never crops: no well-exposed pixel may be lost."""
    st = _stacker()
    st.add_frame(_frame(), t_unix=1000.0)
    for i in range(1, 5):
        st.add_frame(_frame(shift=(4 * i, 2 * i)), t_unix=1000.0 + i)

    raw = st.get_mean_u16()
    wgt = st.wgt
    assert wgt is not None
    solid = wgt >= 0.85 * float(wgt.max())

    info = _solve_info(st)
    assert info is not None
    img = info["image"]
    ox, oy = info["pad_offset_xy"]

    # la imagen entregada contiene el mosaico completo...
    assert img.shape[0] >= raw.shape[0] and img.shape[1] >= raw.shape[1]
    inner = img[oy:oy + raw.shape[0], ox:ox + raw.shape[1]]
    # ...y cada pixel bien expuesto llega intacto (solo se aplana el borde)
    np.testing.assert_array_equal(inner[solid], raw[solid])
    assert float(np.sum(inner[solid].astype(np.float64))) == pytest.approx(
        float(np.sum(raw[solid].astype(np.float64))), rel=1e-9
    )


def test_ragged_mosaic_border_is_flattened_for_the_detector() -> None:
    """The drifted canvas edge must not reach SEP as bright sources.

    On a real drifted stack, 12 of the 30 brightest "detections" were border
    steps rather than stars, which pushed real stars out of the list the solver
    uses; suppressing them took that solve from 4 inliers to 14.
    """
    st = _stacker()
    st.add_frame(_frame(), t_unix=1000.0)
    for i in range(1, 6):
        st.add_frame(_frame(shift=(5 * i, 6 * i)), t_unix=1000.0 + i)

    raw = st.get_mean_u16()
    wgt = st.wgt
    assert wgt is not None
    partial = wgt < 0.85 * float(wgt.max())
    assert partial.any(), "el canvas deberia tener zona parcialmente cubierta"

    info = _solve_info(st)
    assert info is not None
    img = info["image"]

    # el relleno es asimetrico (centra la referencia), asi que uso su offset
    ox, oy = info["pad_offset_xy"]
    inner = img[oy:oy + raw.shape[0], ox:ox + raw.shape[1]]

    # las zonas parcialmente cubiertas quedan al nivel del cielo, sin escalones
    vals = inner[partial]
    assert vals.size
    assert float(vals.std()) < 1.0, "el borde sigue teniendo estructura detectable"

    # y las zonas bien expuestas conservan su contenido intacto
    solid = ~partial
    np.testing.assert_array_equal(inner[solid], raw[solid])


def test_stack_source_never_widens_the_search_radius() -> None:
    """Regression: widening the radius for the mosaic made solves crawl.

    search_radius_deg decides how much catalog is loaded and searched, so its
    cost grows roughly as the square. A mosaic covering more sky is a reason to
    accept a center further from the target, not a reason to load more catalog:
    the pointing uncertainty is unchanged by stacking.
    """
    from app_runner import AppRunner

    cfg = AppConfig()
    cfg.stacking.resp_min = 0.0
    runner = AppRunner(cfg)
    try:
        eng = runner._stacking.engine
        eng.configure_from_cfg()
        eng.enabled = True
        assert eng._live_gray is not None
        eng._live_gray.add_frame(_frame(), t_unix=1000.0)
        eng._live_gray.add_frame(_frame(shift=(8, 11)), t_unix=1030.0)

        base_radius = float(AppConfig().platesolving.search_radius_deg)
        runner._platesolving_source = "stack"
        got = runner._get_platesolving_cfg_snapshot()

        assert float(got.search_radius_deg) == base_radius
        # el margen de aceptacion sí puede crecer: es una comprobacion barata
        assert float(got.max_center_offset_margin_deg) >= 0.0
    finally:
        runner.stop()
