from __future__ import annotations

from unittest.mock import patch

import numpy as np
from astropy.time import Time

from ap_types import Axis
from app_runner import AppRunner
from config import AppConfig
from goto import MountKinematics
from platesolving import ObserverConfig
from simulation import SimulatedCameraStream, SimulationState, restore_iers_after_demo
from tracking import (
    estimate_shift_from_source_matches,
    make_tracking_state,
    tracking_set_params,
    tracking_step,
)


EMPTY_OBJECTS = np.zeros((0,), dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")])


def _objects(points: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros((len(points),), dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")])
    for i, (x, y) in enumerate(points):
        out["x"][i] = x
        out["y"][i] = y
        out["flux"][i] = float(len(points) - i) * 1000.0
    return out


def test_tracking_rejects_noise_correlation_when_sep_has_no_detections() -> None:
    state = make_tracking_state()
    raw0 = np.zeros((32, 32), dtype=np.uint16)
    raw1 = np.ones((32, 32), dtype=np.uint16)

    with (
        patch("tracking.sep_detect_from_raw16", return_value=(None, None, EMPTY_OBJECTS, np.zeros((0, 2)))),
        patch("tracking.estimate_shift_from_phase_alignment", return_value=(3.0, 0.0, 1.0)),
        patch("tracking.estimate_shift_from_profile_alignment", return_value=(3.0, 0.0, 1.0)),
    ):
        tracking_step(state, raw0, now_t=0.0, tracking_enabled=True)
        out = tracking_step(state, raw1, now_t=0.1, tracking_enabled=True)

    assert not out.ok
    assert out.n_det == 0
    assert out.rate_az == 0.0
    assert out.rate_alt == 0.0


def test_tracking_bad_frame_keeps_last_good_incremental_reference() -> None:
    state = make_tracking_state()
    raw0 = np.zeros((16, 16), dtype=np.uint16)
    raw1 = np.ones((16, 16), dtype=np.uint16) * 100
    objs = _objects([(5.0, 5.0), (10.0, 8.0), (12.0, 12.0)])

    with (
        patch("tracking.sep_detect_from_raw16", return_value=(None, None, objs, np.zeros((3, 2)))),
        patch("tracking.estimate_shift_from_phase_alignment", return_value=(20.0, 0.0, 0.0)),
        patch("tracking.estimate_shift_from_profile_alignment", return_value=(20.0, 0.0, 0.0)),
        patch("tracking.estimate_shift_from_source_matches", return_value=(20.0, 0.0, 0.0, 0)),
    ):
        tracking_step(state, raw0, now_t=0.0, tracking_enabled=True)
        out = tracking_step(state, raw1, now_t=0.1, tracking_enabled=True)

    assert not out.ok
    assert state.prev_t == 0.0
    assert state.prev_align_u16 is not None
    assert np.array_equal(state.prev_align_u16, raw0)
    assert state.last_dx_inc == 0.0
    assert state.last_dy_inc == 0.0


def test_tracking_invalid_measurement_does_not_report_a_fake_zero_velocity() -> None:
    state = make_tracking_state()
    raw = np.zeros((16, 16), dtype=np.uint16)
    objs0 = _objects([(5.0, 5.0), (10.0, 8.0), (12.0, 12.0)])
    objs1 = _objects([(6.0, 5.0), (11.0, 8.0), (13.0, 12.0)])
    detections = iter([objs0, objs1, EMPTY_OBJECTS])

    with (
        patch(
            "tracking.sep_detect_from_raw16",
            side_effect=lambda *args, **kwargs: (None, None, next(detections), np.zeros((0, 2))),
        ),
        patch("tracking.estimate_shift_from_phase_alignment", return_value=(1.0, 0.0, 1.0)),
        patch("tracking.estimate_shift_from_profile_alignment", return_value=(1.0, 0.0, 1.0)),
    ):
        first = tracking_step(state, raw, now_t=0.0, tracking_enabled=False)
        good = tracking_step(state, raw, now_t=0.1, tracking_enabled=False)
        bad = tracking_step(state, raw, now_t=0.2, tracking_enabled=False)

    assert not first.ok
    assert first.measurement_reason == "initializing"
    assert good.ok
    assert good.vx > 0.0
    assert not bad.ok
    assert bad.measurement_reason == "no_sources"
    assert bad.vx == good.vx
    assert state.prev_t == 0.1


def test_tracking_recovers_incremental_lock_from_target_keyframe() -> None:
    state = make_tracking_state()
    raw = np.zeros((16, 16), dtype=np.uint16)
    objs0 = _objects([(5.0, 5.0), (10.0, 8.0), (12.0, 12.0)])
    objs1 = _objects([(6.0, 5.0), (11.0, 8.0), (13.0, 12.0)])
    objs2 = _objects([(7.0, 5.0), (12.0, 8.0), (14.0, 12.0)])
    detections = iter([objs0, objs1, objs2])

    with (
        patch(
            "tracking.sep_detect_from_raw16",
            side_effect=lambda *args, **kwargs: (None, None, next(detections), np.zeros((0, 2))),
        ),
        patch(
            "tracking.estimate_shift_from_phase_alignment",
            side_effect=[(1.0, 0.0, 1.0), (20.0, 0.0, 0.0), (2.0, 0.0, 1.0)],
        ),
        patch(
            "tracking.estimate_shift_from_profile_alignment",
            side_effect=[(1.0, 0.0, 1.0), (20.0, 0.0, 0.0), (2.0, 0.0, 1.0)],
        ),
        patch(
            "tracking.estimate_shift_from_source_matches",
            side_effect=[(1.0, 0.0, 1.0, 3), (20.0, 0.0, 0.0, 0), (2.0, 0.0, 1.0, 3)],
        ),
    ):
        tracking_step(state, raw, now_t=0.0, tracking_enabled=False)
        good = tracking_step(state, raw, now_t=0.1, tracking_enabled=False)
        recovered = tracking_step(state, raw, now_t=0.2, tracking_enabled=False)

    assert good.ok
    assert recovered.ok
    assert recovered.measurement_reason == "keyframe_recovery"
    assert recovered.measurement_source.startswith("keyframe:")
    assert recovered.x_hat == 2.0
    assert state.prev_t == 0.2


def test_source_matching_is_one_to_one_in_dense_fields() -> None:
    ref = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    cur = np.array([[1.0, 0.0], [1.2, 0.0], [11.0, 0.0]], dtype=np.float64)

    dx, dy, resp, matches = estimate_shift_from_source_matches(
        ref,
        cur,
        center_dx=1.0,
        center_dy=0.0,
        max_shift_px=2.0,
        min_sources=2,
    )

    assert matches == 2
    assert resp == 1.0
    assert dx == 1.0
    assert dy == 0.0


def test_unmeasured_calibration_prior_never_drives_mount() -> None:
    state = make_tracking_state()
    assert not state.auto.ok
    assert state.auto.A_pinv is None


def test_real_alignment_rejects_noise_frame_and_reacquires_original_target() -> None:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:128, :128]
    base_f = np.full((128, 128), 700.0, dtype=np.float64)
    for x, y, amplitude in [
        (22, 25, 28_000),
        (55, 42, 22_000),
        (94, 31, 18_000),
        (31, 91, 25_000),
        (82, 84, 16_000),
        (108, 103, 21_000),
    ]:
        base_f += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.6**2))
    base = np.clip(base_f + rng.normal(0.0, 12.0, base_f.shape), 0, 65_535).astype(np.uint16)

    state = make_tracking_state()
    tracking_set_params(
        state,
        sep_bw=16,
        sep_bh=16,
        sep_thresh_sigma=2.5,
        sep_minarea=3,
        sep_min_sources=3,
        resp_min=0.06,
    )

    outputs = []
    for i, shift in enumerate([0, 1, None, 3, 4]):
        if shift is None:
            frame = np.clip(700.0 + rng.normal(0.0, 12.0, base.shape), 0, 65_535).astype(np.uint16)
        else:
            frame = np.pad(base[:, : 128 - shift], ((0, 0), (shift, 0)))
        outputs.append(tracking_step(state, frame, now_t=0.1 * i, tracking_enabled=False))

    assert not outputs[0].ok
    assert outputs[1].ok
    assert 0.8 <= outputs[1].dx <= 1.2
    assert not outputs[2].ok
    assert outputs[2].vx == outputs[1].vx
    assert outputs[3].ok
    assert 1.5 <= outputs[3].dx <= 2.5
    assert outputs[4].ok
    assert abs(outputs[4].x_hat - 4.0) <= 0.25


def test_tracking_rls_ignores_small_applied_rates() -> None:
    state = make_tracking_state()
    state.lock_conf = 1.0
    raw0 = np.zeros((16, 16), dtype=np.uint16)
    raw1 = np.zeros((16, 16), dtype=np.uint16)
    objs0 = _objects([(5.0, 5.0), (10.0, 8.0), (12.0, 12.0)])
    objs1 = _objects([(5.2, 5.0), (10.2, 8.0), (12.2, 12.0)])

    calls = {"n": 0}

    def _detect(raw, **kwargs):
        return (None, None, objs0 if calls["n"] == 0 else objs1, np.zeros((3, 2)))

    with (
        patch("tracking.sep_detect_from_raw16", side_effect=_detect),
        patch("tracking.estimate_shift_from_phase_alignment", return_value=(0.2, 0.0, 1.0)),
        patch("tracking.estimate_shift_from_profile_alignment", return_value=(0.2, 0.0, 1.0)),
        patch("tracking.auto_rls_update") as mocked_rls,
    ):
        calls["n"] = 0
        tracking_step(state, raw0, now_t=0.0, tracking_enabled=True)
        calls["n"] = 1
        tracking_step(
            state,
            raw1,
            now_t=0.1,
            tracking_enabled=True,
            applied_rate_az=1.0,
            applied_rate_alt=0.0,
        )

    mocked_rls.assert_not_called()


def test_demo_synthetic_catalog_places_stars_inside_frame() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 321
    cfg.simulation.frame_w = 240
    cfg.simulation.frame_h = 180
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)
    try:
        center = state.center_icrs(observer=ObserverConfig(), obstime=Time.now())
        catalog = stream._synthetic_catalog(center, radius_deg=1.2)
        img = np.zeros((180, 240), dtype=np.float32)
        drawn = stream._draw_catalog_stars(img, catalog, center, Time.now())
    finally:
        restore_iers_after_demo(None)

    assert len(catalog) >= 80
    assert drawn > 20
    assert float(img.max()) > 0.0


def test_runner_seeds_tracking_calibration_from_demo_geometry() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 123
    runner = AppRunner(cfg)
    try:
        runner._ensure_simulation_state()
        assert runner._tracking_seed_calibration_from_pointing()
        assert runner._tracking_state.auto.ok
        assert runner._tracking_state.auto.src == "geometry"
        assert abs(float(runner._tracking_state.auto.detA)) > 1e-6
    finally:
        runner.stop()


def test_tracking_rate_accounts_exact_emitted_steps_with_fractional_accumulator() -> None:
    class _FakeMount:
        def __init__(self) -> None:
            self.moves: list[tuple[Axis, int]] = []

        def move_steps(
            self,
            *,
            axis: Axis,
            direction: int,
            steps: int,
            delay_us: int,
            blocking: bool,
            stop_before_move: bool,
        ) -> None:
            del delay_us, blocking, stop_before_move
            self.moves.append((axis, int(direction) * int(steps)))

        def stop(self) -> None:
            return None

    runner = AppRunner(AppConfig())
    fake_mount = _FakeMount()
    runner._mount = fake_mount
    runner._rate_emul_last_t = 0.0
    clock = {"t": 0.25}

    try:
        with patch("app_runner._perf", side_effect=lambda: float(clock["t"])):
            first = runner._tracking_rate_safe(10.0, 0.0)
            clock["t"] = 0.5
            second = runner._tracking_rate_safe(10.0, 0.0)
    finally:
        runner._mount = None
        runner.stop()

    assert first == (2, 0)
    assert second == (3, 0)
    assert sum(steps for axis, steps in fake_mount.moves if axis == Axis.AZ) == 5
    np.testing.assert_allclose(runner._goto.model.steps_est, [5.0, 0.0])
