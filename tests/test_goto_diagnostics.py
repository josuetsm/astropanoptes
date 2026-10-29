from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.time import Time

from ap_types import GotoStatus
from config import AppConfig
from goto import GoToController, GoToModel, GoToWorker, _AutocalFrame
from goto_diagnostics import DiagnosticSession, raw16_statistics
from platesolving import (
    ObserverConfig,
    PlatesolvingResult,
    PlatesolvingWorker,
    TemporalDetections,
)


def _solution(*, status: str, success: bool, obstime: Time, x: float) -> PlatesolvingResult:
    return PlatesolvingResult(
        success=success,
        status=status,
        theta_deg=0.0,
        dx_px=0.0,
        dy_px=0.0,
        response=1.0,
        scale_arcsec_per_px=1.0,
        R_2x2=((1.0, 0.0), (0.0, 1.0)),
        t_arcsec=(0.0, 0.0),
        n_inliers=8,
        rms_arcsec=0.3,
        rms_px=0.3,
        center_ra_deg=120.0,
        center_dec_deg=-30.0,
        overlay=[],
        guides=[],
        metrics={"marker": float(x)},
        obstime_unix=float(obstime.unix),
    )


def _worker(tmp_path, *, cfg: AppConfig | None = None) -> GoToWorker:
    cfg = cfg or AppConfig()
    cfg.goto.diagnostics_enabled = True
    cfg.goto.diagnostics_dir = str(tmp_path)
    cfg.platesolving.diagnostics_enabled = True
    cfg.platesolving.diagnostics_dir = str(tmp_path)
    controller = GoToController(model=GoToModel())
    return GoToWorker(
        goto_controller=controller,
        get_state=lambda: SimpleNamespace(camera=SimpleNamespace(connected=True, roll_deg=0.0)),
        publish_state=lambda patch: None,
        get_frame=lambda: None,
        get_goto_cfg=lambda: cfg.goto,
        get_mount_cfg=lambda: cfg.mount,
        get_sep_cfg=lambda: cfg.sep,
        get_camera_cfg=lambda: cfg.camera,
        get_platesolving_cfg=lambda: cfg.platesolving,
        get_observer=lambda: ObserverConfig(),
        apply_camera_param=lambda name, value: None,
        pause_tracking=lambda: False,
        resume_tracking=lambda: None,
        pause_stacking=lambda: False,
        resume_stacking=lambda: None,
        rate_mount=lambda az, alt: None,
        move_steps=lambda axis, direction, steps, delay_us: None,
        stop_mount=lambda: None,
    )


def test_diagnostic_session_persists_raw_stack_and_timeline(tmp_path) -> None:
    session = DiagnosticSession(
        root_dir=str(tmp_path),
        operation="autocal",
        enabled=True,
        context={"target": "M42"},
    )
    raw0 = np.arange(24, dtype=np.uint16).reshape(4, 6)
    raw1 = raw0 + np.uint16(10)

    raw_path = session.save_raw("platesolve", raw0, metadata={"seq": 7})
    stack_path = session.save_raw_stack(
        "drift",
        [raw0, raw1],
        frame_metadata=[{"t": 1.0}, {"t": 2.0}],
    )
    session.record("goto_plan", steps=np.array([120, -80]))
    session.close("OK", final_error_arcsec=2.5)

    assert raw_path is not None
    assert stack_path is not None
    np.testing.assert_array_equal(np.load(raw_path), raw0)
    with np.load(stack_path) as saved:
        np.testing.assert_array_equal(saved["raw16"], np.stack([raw0, raw1]))

    manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OK"
    assert [item["kind"] for item in manifest["artifacts"]] == ["raw16", "raw16_stack"]
    assert manifest["artifacts"][0]["frame"]["sha256"] == raw16_statistics(raw0)["sha256"]
    stages = [json.loads(line)["stage"] for line in (session.path / "timeline.jsonl").read_text().splitlines()]
    assert "goto_plan" in stages
    assert stages[-1] == "session_finished"




def test_explicit_platesolve_persists_the_exact_solver_input(tmp_path) -> None:
    cfg = AppConfig()
    cfg.platesolving.diagnostics_enabled = True
    cfg.platesolving.diagnostics_dir = str(tmp_path)
    raw = np.arange(80, dtype=np.uint16).reshape(8, 10)
    result = _solution(
        status="OK",
        success=True,
        obstime=Time("2026-08-11T01:00:00", scale="utc"),
        x=1.0,
    )
    published: list[dict] = []
    worker = PlatesolvingWorker(
        get_frame=lambda: raw,
        get_cfg=lambda: cfg.platesolving,
        get_sep_cfg=lambda: cfg.sep,
        get_observer=lambda: ObserverConfig(),
        publish_state=lambda patch: published.append(patch),
    )

    with patch.object(worker, "_solve_or_verify", return_value=(result, raw.copy())):
        worker._handle_request({"target": "M42"})

    sessions = list(tmp_path.glob("*_platesolve_*"))
    assert len(sessions) == 1
    manifest = json.loads((sessions[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OK"
    saved_raw = sessions[0] / manifest["artifacts"][0]["path"]
    np.testing.assert_array_equal(np.load(saved_raw), raw)
    debug_patches = [
        patch_["platesolving"]["debug_info"]
        for patch_ in published
        if "platesolving" in patch_ and patch_["platesolving"].get("debug_info")
    ]
    assert debug_patches[-1]["diagnostics_dir"] == str(sessions[0].resolve())


def test_explicit_platesolve_waits_for_frame_after_camera_restart(tmp_path) -> None:
    cfg = AppConfig()
    cfg.platesolving.diagnostics_enabled = False
    cfg.platesolving.frame_wait_timeout_s = 0.2
    raw = np.arange(80, dtype=np.uint16).reshape(8, 10)
    result = _solution(
        status="OK",
        success=True,
        obstime=Time("2026-08-11T01:00:00", scale="utc"),
        x=1.0,
    )
    frames = iter([None, None, raw])
    worker = PlatesolvingWorker(
        get_frame=lambda: next(frames, raw),
        get_cfg=lambda: cfg.platesolving,
        get_sep_cfg=lambda: cfg.sep,
        get_observer=lambda: ObserverConfig(),
        publish_state=lambda patch: None,
    )

    with patch.object(worker, "_solve_or_verify", return_value=(result, raw.copy())) as solve:
        worker._handle_request({"target": "M42"})

    assert solve.call_count == 1


def test_explicit_platesolve_cancellation_stops_before_solver(tmp_path) -> None:
    cfg = AppConfig()
    cfg.platesolving.diagnostics_enabled = False
    raw = np.arange(80, dtype=np.uint16).reshape(8, 10)
    published: list[dict] = []
    worker = PlatesolvingWorker(
        get_frame=lambda: raw,
        get_cfg=lambda: cfg.platesolving,
        get_sep_cfg=lambda: cfg.sep,
        get_observer=lambda: ObserverConfig(),
        publish_state=lambda patch: published.append(patch),
    )
    worker.cancel_current()

    with patch.object(worker, "_solve_or_verify") as solve:
        worker._handle_request({"target": "M42"})

    assert solve.call_count == 0
    final = [
        p["platesolving"]
        for p in published
        if "platesolving" in p and p["platesolving"].get("busy") is False
    ][-1]
    assert final["reason"] == "CANCELLED"


def test_explicit_platesolve_total_timeout_stops_before_solver(tmp_path) -> None:
    cfg = AppConfig()
    cfg.platesolving.diagnostics_enabled = False
    cfg.platesolving.total_timeout_s = 0.001
    raw = np.arange(80, dtype=np.uint16).reshape(8, 10)
    published: list[dict] = []
    worker = PlatesolvingWorker(
        get_frame=lambda: raw,
        get_cfg=lambda: cfg.platesolving,
        get_sep_cfg=lambda: cfg.sep,
        get_observer=lambda: ObserverConfig(),
        publish_state=lambda patch: published.append(patch),
    )

    def collect(*_args, **_kwargs):
        worker._solve_deadline = 0.0
        worker._raise_if_aborted()

    with (
        patch.object(worker, "_collect_temporal_detections", side_effect=collect),
        patch.object(worker, "_solve_or_verify") as solve,
    ):
        worker._handle_request({"target": "M42"})

    assert solve.call_count == 0
    final = [
        p["platesolving"]
        for p in published
        if "platesolving" in p and p["platesolving"].get("busy") is False
    ][-1]
    assert final["reason"] == "TIMEOUT"




def test_goto_worker_writes_model_and_planning_diagnostics(tmp_path) -> None:
    cfg = AppConfig()
    cfg.goto.diagnostics_enabled = True
    cfg.goto.diagnostics_dir = str(tmp_path)
    worker = _worker(tmp_path, cfg=cfg)
    worker._goto.model.synced = True

    def _goto_blocking(target, **kwargs):
        kwargs["diagnostics"].record("goto_plan_iteration", planned_steps=[12, -4])
        return SimpleNamespace(
            ok=True,
            status="OK_MODEL",
            iters=1,
            err_norm_arcsec=lambda: 5.0,
        )

    worker._goto.goto_blocking = _goto_blocking
    worker._handle_request({"kind": "goto", "target": "M42", "params": {}})

    sessions = list(tmp_path.glob("*_goto_*"))
    assert len(sessions) == 1
    manifest = json.loads((sessions[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OK_MODEL"
    assert manifest["summary"]["model_after"]["synced"] is True
    stages = [event["stage"] for event in manifest["events"]]
    assert "operation_started" in stages
    assert "goto_plan_iteration" in stages
