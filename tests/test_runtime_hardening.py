from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from ap_types import Frame
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
