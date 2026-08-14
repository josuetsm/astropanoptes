from __future__ import annotations

import threading
import time

import numpy as np

from ap_types import Frame
import app_runner as app_runner_module
from actions import ActionType
from app_runner import AppRunner
from config import AppConfig
from tracking import TrackingOutput
from workers import SaveWorker


class _FrameStream:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def latest(self) -> Frame:
        return self.frame


def test_camera_params_are_applied_with_one_capture_restart() -> None:
    runner = AppRunner(AppConfig())
    restarts: list[str] = []
    runner._restart_camera_stream_if_active = lambda *, reason: restarts.append(reason)
    try:
        runner._apply_camera_params(
            {
                "exp_ms": 250.0,
                "gain": 320,
                "offset": 24,
                "preview_jpeg_quality": 80,
            }
        )

        assert len(restarts) == 1
        assert "exp_ms" in restarts[0]
        assert "gain" in restarts[0]
        assert "offset" in restarts[0]
        assert "preview_jpeg_quality" not in restarts[0]
        assert runner.cfg.camera.exp_ms == 250.0
        assert runner.cfg.camera.gain == 320
        assert runner.cfg.camera.offset == 24

        runner._apply_camera_params({"preview_stretch_plo": 2.0})
        assert len(restarts) == 1
    finally:
        runner.stop()


def test_tracking_params_entrypoint_queues_an_action() -> None:
    runner = AppRunner(AppConfig())
    try:
        runner.request_tracking_params(resp_min=0.2)
        action = runner._actions.get_nowait()
        assert action.type == ActionType.TRACKING_SET_PARAMS
        assert action.payload == {"resp_min": 0.2}
    finally:
        runner.stop()


def test_stacking_enqueues_each_camera_sequence_once() -> None:
    runner = AppRunner(AppConfig())
    raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
    stream = _FrameStream(Frame(raw=raw, t_capture=1.0, meta={"seq": 7}))
    submitted: list[np.ndarray] = []
    original_enqueue = runner._stacking.enqueue_frame
    runner._cam_stream = stream
    runner._stacking_enabled = True
    runner._stacking.enqueue_frame = lambda frame, t=None: submitted.append(frame)
    try:
        assert runner._maybe_enqueue_stacking_frame()
        assert not runner._maybe_enqueue_stacking_frame()
        stream.frame.meta["seq"] = 8
        assert runner._maybe_enqueue_stacking_frame()
        assert len(submitted) == 2
        assert submitted[0] is not raw
    finally:
        runner._stacking.enqueue_frame = original_enqueue
        runner._cam_stream = None
        runner._stacking_enabled = False
        runner.stop()


def test_astronomy_updates_use_their_own_cadences() -> None:
    cfg = AppConfig()
    cfg.pointing_hz = 2.0
    cfg.tracking.sidereal_ff_update_hz = 2.0
    runner = AppRunner(cfg)
    pointing_calls: list[float] = []
    feedforward_calls: list[float] = []
    runner._update_goto_pointing_state = lambda: pointing_calls.append(1.0)

    def _feedforward(*, now_t=None):
        feedforward_calls.append(float(now_t))
        return 1.0, 2.0, True

    runner._tracking_feedforward_rate = _feedforward
    try:
        assert runner._maybe_update_goto_pointing_state(now=10.0)
        assert not runner._maybe_update_goto_pointing_state(now=10.1)
        assert runner._maybe_update_goto_pointing_state(now=10.5)
        assert len(pointing_calls) == 2

        assert runner._cached_tracking_feedforward_rate(now_t=100.0) == (1.0, 2.0, True)
        assert runner._cached_tracking_feedforward_rate(now_t=100.1) == (1.0, 2.0, True)
        assert runner._cached_tracking_feedforward_rate(now_t=100.5) == (1.0, 2.0, True)
        assert feedforward_calls == [100.0, 100.5]
    finally:
        runner.stop()


def test_preview_submission_coalesces_duplicate_sequences() -> None:
    runner = AppRunner(AppConfig())
    raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
    stream = _FrameStream(Frame(raw=raw, t_capture=1.0, meta={"seq": 3}))
    requests: list[dict] = []
    runner._cam_stream = stream
    runner._preview_worker.request = lambda **payload: requests.append(payload)
    try:
        runner._t_last_preview = 0.0
        runner._maybe_update_preview()
        runner._t_last_preview = 0.0
        runner._maybe_update_preview()
        assert len(requests) == 1

        stream.frame.meta["seq"] = 4
        runner._t_last_preview = 0.0
        runner._maybe_update_preview()
        assert len(requests) == 2
        assert requests[0]["raw16"] is not raw
    finally:
        runner._cam_stream = None
        runner.stop()


def test_tracking_worker_publishes_results_off_the_control_thread(monkeypatch) -> None:
    runner = AppRunner(AppConfig())
    worker_threads: list[str] = []

    def _tracking_step(*_args, **_kwargs) -> TrackingOutput:
        worker_threads.append(threading.current_thread().name)
        return TrackingOutput(
            ok=True,
            mode="TRACK",
            resp=1.0,
            dx=0.0,
            dy=0.0,
            vx=0.0,
            vy=0.0,
            abs_resp=1.0,
            x_hat=0.0,
            y_hat=0.0,
            rate_az=4.0,
            rate_alt=5.0,
            calib_src="auto",
            detA=1.0,
            n_det=12,
        )

    monkeypatch.setattr(app_runner_module, "tracking_step", _tracking_step)
    try:
        runner._submit_tracking_frame(
            raw16=np.zeros((8, 8), dtype=np.uint16),
            frame_token=1.0,
            frame_t=1.0,
            tracking_enabled=True,
        )
        deadline = time.monotonic() + 2.0
        output = None
        while time.monotonic() < deadline:
            output, error = runner._tracking_result_snapshot()
            if output is not None or error is not None:
                break
            time.sleep(0.01)

        assert error is None
        assert output is not None
        assert output.rate_az == 4.0
        assert worker_threads == ["TrackingWorker"]
    finally:
        runner.stop()


def test_save_worker_runs_disk_work_off_the_calling_thread() -> None:
    finished = threading.Event()
    callback_threads: list[str] = []

    def _save(_request) -> None:
        callback_threads.append(threading.current_thread().name)
        finished.set()

    worker = SaveWorker(_save)
    worker.request(out_dir="unused")
    try:
        assert finished.wait(timeout=2.0)
        assert callback_threads == ["SaveWorker"]
    finally:
        worker.stop(timeout=2.0)


def test_loop_profiler_reports_percentiles() -> None:
    runner = AppRunner(AppConfig())
    try:
        runner._record_loop_performance({"actions_ms": 1.0, "total_ms": 10.0})
        runner._record_loop_performance({"actions_ms": 3.0, "total_ms": 30.0})
        metrics = runner.get_performance_metrics()
        assert metrics["sample_count"] == 2
        assert metrics["sections"]["actions_ms"]["p50"] == 2.0
        assert metrics["sections"]["total_ms"]["p99"] > 29.0
    finally:
        runner.stop()
