from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import AppConfig
from goto import GoToWorker
from sep_utils import sep_detect_from_raw16


def _objects(rows: list[tuple[float, float, float]]) -> np.ndarray:
    out = np.zeros(len(rows), dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")])
    for idx, (x, y, flux) in enumerate(rows):
        out[idx] = (x, y, flux)
    return out


def test_sep_retries_pixel_stack_overflow_with_a_stronger_threshold() -> None:
    raw = np.zeros((40, 60), dtype=np.uint16)
    fake_bkg = SimpleNamespace(
        globalrms=2.0,
        back=lambda: np.zeros(raw.shape, dtype=np.float32),
    )
    detected = _objects([(12.0, 18.0, 900.0)])

    with (
        patch("sep_utils.median_prefilter_raw16", return_value=raw.astype(np.float32)),
        patch("sep_utils.sep.Background", return_value=fake_bkg),
        patch(
            "sep_utils.sep.extract",
            side_effect=[
                Exception("internal pixel buffer full: pixstack reached"),
                detected,
            ],
        ) as extract,
    ):
        _img, _bkg, objects, xy = sep_detect_from_raw16(
            raw,
            sep_bw=16,
            sep_bh=16,
            sep_thresh_sigma=3.0,
            sep_minarea=5,
            max_sources=10,
        )

    assert extract.call_count == 2
    assert float(extract.call_args_list[0].args[1]) == 6.0
    assert float(extract.call_args_list[1].args[1]) == 9.0
    assert len(objects) == 1
    np.testing.assert_allclose(xy, [[12.0, 18.0]])


def test_sep_restores_requested_threshold_after_growing_pixel_stack() -> None:
    raw = np.zeros((40, 60), dtype=np.uint16)
    fake_bkg = SimpleNamespace(
        globalrms=2.0,
        back=lambda: np.zeros(raw.shape, dtype=np.float32),
    )
    detected = _objects([(12.0, 18.0, 900.0)])
    overflow = Exception("internal pixel buffer full: pixstack reached")

    with (
        patch("sep_utils.median_prefilter_raw16", return_value=raw.astype(np.float32)),
        patch("sep_utils.sep.Background", return_value=fake_bkg),
        patch("sep_utils.sep.set_extract_pixstack") as set_pixstack,
        patch(
            "sep_utils.sep.extract",
            side_effect=[overflow, overflow, overflow, overflow, detected],
        ) as extract,
        patch("sep_utils._SEP_PIXSTACK", 1),
    ):
        _img, _bkg, objects, _xy = sep_detect_from_raw16(
            raw,
            sep_bw=16,
            sep_bh=16,
            sep_thresh_sigma=3.0,
            sep_minarea=5,
            max_sources=10,
        )

    assert len(objects) == 1
    set_pixstack.assert_called_once()
    assert extract.call_count == 5
    assert [float(call.args[1]) for call in extract.call_args_list] == [6.0, 9.0, 13.5, 21.0, 6.0]


def test_autocal_does_not_treat_quantized_sparse_frame_as_overexposed() -> None:
    cfg = AppConfig()
    img_det = np.ones((30, 30), dtype=np.float32) * 20.0
    bkg = SimpleNamespace(globalrms=1.0)
    objects = _objects([(15.0, 15.0, 1000.0)])
    dummy = SimpleNamespace(
        _get_sep_cfg=lambda: cfg.sep,
        _get_platesolving_cfg=lambda: cfg.platesolving,
        _out_log=None,
    )

    with patch(
        "goto.sep_detect_from_raw16",
        return_value=(img_det, bkg, objects, np.array([[15.0, 15.0]], dtype=np.float64)),
    ):
        _xy, star_count, _sat, _sources = GoToWorker._autocal_detect(
            dummy,
            np.zeros((30, 30), dtype=np.uint16),
        )

    assert star_count == 1


def test_autocal_treats_high_active_area_with_near_cap_sources_as_overexposed() -> None:
    cfg = AppConfig()
    img_det = np.ones((30, 30), dtype=np.float32) * 20.0
    bkg = SimpleNamespace(globalrms=1.0)
    rows = [
        (float(i % 30), float((i // 30) % 30), float(1000 - i))
        for i in range(200)
    ]
    objects = _objects(rows)
    obj_xy = np.column_stack((objects["x"], objects["y"]))
    dummy = SimpleNamespace(
        _get_sep_cfg=lambda: cfg.sep,
        _get_platesolving_cfg=lambda: cfg.platesolving,
        _out_log=None,
    )

    with patch(
        "goto.sep_detect_from_raw16",
        return_value=(img_det, bkg, objects, obj_xy),
    ):
        _xy, star_count, _sat, _sources = GoToWorker._autocal_detect(
            dummy,
            np.zeros((30, 30), dtype=np.uint16),
        )

    assert star_count == int(cfg.platesolving.max_det) + 1


def test_goto_never_changes_operator_exposure_or_gain() -> None:
    changes: list[tuple[str, float]] = []
    camera_cfg = SimpleNamespace(exp_ms=300.0, gain=600)
    dummy = SimpleNamespace(
        _get_camera_cfg=lambda: camera_cfg,
        _apply_camera_param=lambda name, value: changes.append((str(name), float(value))),
    )

    changed = GoToWorker._autocal_adjust_exposure(
        dummy,
        star_count=0,
        saturation_frac=0.5,
        target_min=3,
        target_max=200,
        sat_max=0.01,
        exp_min_ms=20.0,
        exp_max_ms=1200.0,
        exp_step=1.5,
        gain_min=0,
        gain_max=600,
        gain_step=50,
        settle_s=0.0,
    )

    assert not changed
    assert changes == []
