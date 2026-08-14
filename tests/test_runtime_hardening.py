from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy.time import Time

from actions import Action, ActionType
from ap_types import Axis, Frame, GotoStatus
from goto import platesolving_center_to_altaz_deg
from platesolving import PlatesolvingResult
from app_runner import AppRunner
from config import AppConfig
from workers import BaseWorker


def test_stacking_basename_uses_wall_time_not_monotonic_epoch() -> None:
    runner = AppRunner(AppConfig())
    raw = np.zeros((4, 4), dtype=np.uint16)
    frame = Frame(raw=raw, t_capture=12345.0, meta={"t_wall": 1_700_000_000.0})
    runner._cam_stream = SimpleNamespace(latest=lambda: frame)

    basename = runner._stacking_capture_basename("stack")

    assert "19700101" not in basename
    assert basename.startswith("stack_")


def test_stacking_basename_ignores_monotonic_capture_as_wall_time() -> None:
    runner = AppRunner(AppConfig())
    raw = np.zeros((4, 4), dtype=np.uint16)
    frame = Frame(raw=raw, t_capture=12345.0, meta={})
    runner._cam_stream = SimpleNamespace(latest=lambda: frame)

    basename = runner._stacking_capture_basename("stack")

    assert "19700101" not in basename


def test_platesolving_live_frame_is_copied_from_ring_buffer() -> None:
    runner = AppRunner(AppConfig())
    raw = np.zeros((4, 4), dtype=np.uint16)
    frame = Frame(raw=raw, t_capture=time.perf_counter(), meta={"t_wall": time.time()})
    runner._cam_stream = SimpleNamespace(latest=lambda: frame)

    out = runner._get_live_frame_for_platesolving()

    assert out is not None
    out[0, 0] = 65535
    assert raw[0, 0] == 0


def test_failed_platesolve_clears_cached_success_used_by_sync() -> None:
    runner = AppRunner(AppConfig())
    success = SimpleNamespace(success=True)
    rejected = SimpleNamespace(success=False)

    runner._publish_platesolving_state({"platesolving_result": success})
    assert runner._last_platesolving_result is success

    runner._publish_platesolving_state({"platesolving_result": rejected})
    assert runner._last_platesolving_result is None


def test_goto_worker_solution_is_not_enqueued_for_duplicate_sample_validation() -> None:
    runner = AppRunner(AppConfig())
    solution = _successful_plate_solution()

    runner._publish_platesolving_state(
        {
            "platesolving_result": solution,
            "platesolving_result_handled": True,
        }
    )

    assert runner._last_platesolving_result is solution
    assert runner._actions.empty()


def test_emergency_stop_invalidates_sync_when_motion_is_active() -> None:
    runner = AppRunner(AppConfig())
    stopped = []
    runner._mount = SimpleNamespace(stop=lambda: stopped.append(True))
    runner._goto.model.synced = True
    runner._manual_move_active_until_s["az"] = time.perf_counter() + 10.0

    runner._handle_action(
        Action(ActionType.MOUNT_STOP, {}, time.perf_counter())
    )

    state = runner.get_state()
    assert stopped
    assert not runner._goto.model.synced
    assert not state.goto.synced
    assert not state.goto.pointing_valid
    assert state.goto.status == GotoStatus.CANCELLED
    assert state.goto.reason == "STOP_POSITION_UNKNOWN"


def test_goto_cancel_invalidates_sync_when_motion_is_active() -> None:
    runner = AppRunner(AppConfig())
    stopped = []
    runner._mount = SimpleNamespace(stop=lambda: stopped.append(True))
    runner._goto.model.synced = True
    runner._manual_move_active_until_s["alt"] = time.perf_counter() + 10.0

    runner._handle_goto_action(ActionType.GOTO_CANCEL, {})

    state = runner.get_state()
    assert stopped
    assert not runner._goto.model.synced
    assert not state.goto.synced
    assert not state.goto.pointing_valid
    assert state.goto.status == GotoStatus.CANCELLED
    assert state.goto.reason == "CANCEL_POSITION_UNKNOWN"


def test_temporal_detection_parameters_keep_ten_frame_minimum() -> None:
    runner = AppRunner(AppConfig())

    runner._handle_platesolving_action(
        ActionType.PLATESOLVING_SET_PARAMS,
        {"temporal_min_hits": 4, "temporal_window_frames": 6},
    )

    assert runner.cfg.platesolving.temporal_min_hits == 10
    assert runner.cfg.platesolving.temporal_window_frames == 10


def test_operation_counters_distinguish_start_and_finish() -> None:
    runner = AppRunner(AppConfig())

    runner._publish_platesolving_state(
        {"platesolving": {"busy": True, "status": "RUNNING"}}
    )
    active = runner.get_operation_counters()["platesolving"]
    assert active == {"started": 1, "finished": 0}

    runner._publish_platesolving_state(
        {"platesolving": {"busy": False, "status": "FAIL"}}
    )
    finished = runner.get_operation_counters()["platesolving"]
    assert finished == {"started": 1, "finished": 1}


def _successful_plate_solution() -> PlatesolvingResult:
    return PlatesolvingResult(
        success=True,
        status="OK",
        theta_deg=12.0,
        dx_px=0.0,
        dy_px=0.0,
        response=1.0,
        scale_arcsec_per_px=2.0,
        R_2x2=((1.0, 0.0), (0.0, 1.0)),
        t_arcsec=(0.0, 0.0),
        n_inliers=20,
        rms_arcsec=1.0,
        rms_px=0.5,
        center_ra_deg=120.0,
        center_dec_deg=-30.0,
        overlay=[],
        guides=[],
        metrics={},
        obstime_unix=1_789_000_000.0,
    )


def test_plate_solution_is_accepted_automatically_after_quality_checks() -> None:
    runner = AppRunner(AppConfig())
    solution = _successful_plate_solution()
    runner._last_platesolving_result = solution

    runner._handle_goto_action(ActionType.GOTO_VALIDATE_SAMPLE, {"result": solution})

    state = runner.get_state()
    expected = platesolving_center_to_altaz_deg(
        solution.center_ra_deg,
        solution.center_dec_deg,
        observer=runner._platesolving_observer,
        obstime=Time(solution.obstime_unix, format="unix", scale="utc"),
    )
    assert state.goto.manual_samples == 1
    assert state.goto.sample_last_ok
    assert state.goto.sample_last_reason == "SAMPLE_ACCEPTED_AUTOMATICALLY"
    # Automatic sample validation must not consume the same trustworthy
    # solution before the operator can use `mount sync`.
    assert runner._last_platesolving_result is solution
    np.testing.assert_allclose(runner._goto.model._manual_az_alt_abs[-1], expected, atol=0.02)
    assert runner._goto.model.synced
    np.testing.assert_allclose(
        runner._goto.model.ref_steps,
        runner._goto.model.steps_est,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        runner._goto.model.predict_az_alt_deg(),
        expected,
        rtol=0.0,
        atol=1e-9,
    )


def test_bad_plate_solution_is_rejected_automatically() -> None:
    runner = AppRunner(AppConfig())
    solution = _successful_plate_solution()
    solution = PlatesolvingResult(
        **{
            **solution.__dict__,
            "n_inliers": 2,
            "rms_px": 8.0,
        }
    )

    runner._handle_goto_action(ActionType.GOTO_VALIDATE_SAMPLE, {"result": solution})

    assert runner.get_state().goto.manual_samples == 0
    assert not runner.get_state().goto.sample_last_ok
    assert runner.get_state().goto.sample_last_reason == "SAMPLE_INSUFFICIENT_INLIERS"
    assert runner._last_platesolving_result is None


def test_second_incompatible_axis_sample_cannot_rewrite_nominal_scale() -> None:
    runner = AppRunner(AppConfig())
    first = _successful_plate_solution()
    runner._goto.model.add_manual_sample(
        np.array([130.0, 36.0]),
        roll_deg=-1.0,
    )
    runner._goto.model.note_manual_move(Axis.ALT, 1, 80)
    second = PlatesolvingResult(
        **{
            **first.__dict__,
            "center_ra_deg": 120.0,
            "center_dec_deg": -30.0,
        }
    )
    measured = np.array([130.008, 36.258], dtype=np.float64)

    with (
        patch("app_runner.platesolving_center_to_altaz_deg", return_value=measured),
        patch("app_runner.platesolving_roll_sample_deg", return_value=-1.1),
    ):
        runner._handle_goto_action(ActionType.GOTO_VALIDATE_SAMPLE, {"result": second})

    assert not runner.get_state().goto.sample_last_ok
    assert runner.get_state().goto.sample_last_reason == "SAMPLE_MOTION_MISMATCH"
    assert runner._goto.model.kin.gear_reduction_alt == pytest.approx(45.0)
    assert runner._goto.model.kin.microsteps_alt == 64
    assert runner._goto.model.J_deg_per_step[1, 1] == pytest.approx(0.000625)


def test_runner_initial_state_publishes_configured_backlash() -> None:
    cfg = AppConfig()
    cfg.goto.backlash_steps_az = 3
    cfg.goto.backlash_steps_alt = 11

    runner = AppRunner(cfg)
    state = runner.get_state()

    assert state.goto.backlash_steps_az == 3
    assert state.goto.backlash_steps_alt == 11


def test_base_worker_logs_and_survives_unhandled_request_error() -> None:
    class FlakyWorker(BaseWorker):
        def __init__(self) -> None:
            super().__init__(name="FlakyWorker", idle_sleep_s=0.005)
            self.handled: list[str] = []

        def _handle_request(self, request):
            value = str(request["value"])
            if value == "boom":
                raise RuntimeError("boom")
            self.handled.append(value)

    worker = FlakyWorker()
    try:
        worker.request(value="boom")
        time.sleep(0.05)
        worker.request(value="ok")

        deadline = time.time() + 1.0
        while time.time() < deadline and worker.handled != ["ok"]:
            time.sleep(0.01)

        assert worker.handled == ["ok"]
    finally:
        worker.stop()
        worker.join(timeout=1.0)
