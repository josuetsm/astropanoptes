# app_runner.py
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
import datetime as _dt

from dataclasses import dataclass, replace
from typing import Optional, Any, Dict, List, Tuple, Sequence

import cv2
import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ap_types import SystemState, Axis, Frame
from config import AppConfig, PlatesolvingConfig, SepConfig
from actions import Action, ActionType
from logging_utils import log_info, log_error

from camera_poa import POACameraDevice, CameraStream
from imaging import ensure_raw16_bayer
from preview import make_preview_jpeg, encode_jpeg, stretch_to_u8
from mount_arduino import ArduinoMount

from tracking import (
    auto_reset,
    auto_set_from_A,
    calib_reset,
    calib_set_A_micro,
    make_tracking_state,
    tracking_step,
    tracking_set_params,
)
from stacking import StackingWorker

from sep_utils import sep_detect_from_raw16, estimate_shift_from_objects

from platesolving import (
    ObserverConfig,
    PlatesolvingWorker,
    save_gaia_auth,
    load_gaia_auth,
)

from goto import GoToController, GoToConfig as GoToRuntimeConfig, GoToModel, MountKinematics, GoToWorker
from mount_arduino import MountMoveWorker


def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()



def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return "(none)"
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


class AppRunner:
    """
    Orquestador principal (runtime).

    Responsabilidades:
    - Mantener la cámara capturando a máxima FPS (CameraStream).
    - Ejecutar un loop estable (control_hz) que:
        - aplica actions
        - actualiza SystemState
        - genera preview JPEG a view_hz
        - ejecuta tracking y envía RATE a la montura si tracking está ON
        - encola frames a stacking si stacking está ON
    - Ejecutar plate solving bajo demanda en thread dedicado (no bloquea loop).
    - Exponer getters thread-safe para UI.

    Regla: UI no toca cámara/montura directamente; todo va por actions.
    """

    def __init__(self, cfg: AppConfig, out_log=None) -> None:
        self.default_cfg = cfg
        self.cfg = replace(cfg)
        self.cfg.camera = replace(cfg.camera)
        self.cfg.preview = replace(cfg.preview)
        self.cfg.mount = replace(cfg.mount)
        self.cfg.tracking = replace(cfg.tracking)
        self.cfg.stacking = replace(cfg.stacking)
        self.cfg.goto = replace(cfg.goto)
        self.cfg.platesolving = replace(cfg.platesolving)
        self.out_log = out_log

        self._actions: "queue.Queue[Action]" = queue.Queue()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        # Subsystems
        self._cam_dev: Optional[POACameraDevice] = None
        self._cam_stream: Optional[CameraStream] = None
        self._mount: Optional[ArduinoMount] = None

        # Tracking subsystem
        self._tracking_state = make_tracking_state()

        # Stacking subsystem
        self._stacking = StackingWorker(self.cfg)
        self._stacking_enabled = bool(self.cfg.stacking.enabled_init)

        # Platesolving subsystem
        self._platesolving_cfg_lock = threading.Lock()
        self._platesolving_last_auto_t = 0.0
        self._platesolving_auto_target: str = ""

        # Config platesolving (runtime copy, actualizable desde UI por action)
        self._platesolving_observer = ObserverConfig()  # Santiago por default en tu platesolving.py
        self._platesolving_worker = PlatesolvingWorker(
            get_frame=self._get_live_frame_for_platesolving,
            get_cfg=self._get_platesolving_cfg_snapshot,
            get_sep_cfg=self._get_sep_cfg_snapshot,
            get_observer=lambda: self._platesolving_observer,
            publish_state=self._publish_platesolving_state,
            out_log=self.out_log,
        )

        # GoTo subsystem (no bloquea loop)
        kin = MountKinematics(
            motor_full_steps_per_rev=200,
            microsteps_az=int(self.cfg.mount.ms_az),
            microsteps_alt=int(self.cfg.mount.ms_alt),
            motor_pulley_teeth=20,
            ring_radius_m_az=0.24,
            ring_radius_m_alt=0.235,
        )
        goto_cfg = GoToRuntimeConfig(
            observer=self._platesolving_observer,
            sep=self.cfg.sep,
            alt_min_deg=float(self.cfg.goto.alt_min_deg),
            alt_max_deg=float(self.cfg.goto.alt_max_deg),
            tol_arcsec=float(self.cfg.goto.tol_arcsec),
            max_iters=int(self.cfg.goto.max_iters),
            gain=float(self.cfg.goto.gain),
            max_step_per_iter=int(self.cfg.goto.max_step_per_iter),
            slew_delay_us_az=int(self.cfg.goto.slew_delay_us),
            slew_delay_us_alt=int(self.cfg.goto.slew_delay_us),
            settle_s=float(self.cfg.goto.settle_s),
            stages=int(self.cfg.goto.stages),
            platesolving_feedback=bool(self.cfg.goto.platesolving_feedback),
        )
        self._goto = GoToController(cfg=goto_cfg, model=GoToModel(kin=kin))
        self._last_platesolving_result: Optional[Any] = None
        self._mount_move_worker = MountMoveWorker(
            get_mount=lambda: self._mount,
            note_manual_move=self._goto.model.note_manual_move,
            publish_state=self._set_state_safe,
            out_log=self.out_log,
        )
        self._goto_worker = GoToWorker(
            goto_controller=self._goto,
            get_state=self.get_state,
            publish_state=self._publish_platesolving_state,
            get_frame=self._get_latest_frame,
            get_goto_cfg=self._get_goto_cfg_snapshot,
            get_mount_cfg=self._get_mount_cfg_snapshot,
            get_sep_cfg=self._get_sep_cfg_snapshot,
            get_camera_cfg=self._get_camera_cfg_snapshot,
            get_platesolving_cfg=self._get_platesolving_cfg_snapshot,
            get_observer=lambda: self._platesolving_observer,
            apply_camera_param=self._apply_camera_param,
            pause_tracking=self._pause_tracking_for_goto,
            resume_tracking=self._resume_tracking_after_goto,
            pause_stacking=self._pause_stacking_for_goto,
            resume_stacking=self._resume_stacking_after_goto,
            move_steps=self._goto_move_steps,
            stop_mount=self._mount_stop,
            out_log=self.out_log,
        )
        self._tracking_calib_cols: Dict[str, np.ndarray] = {}
        self._tracking_bootstrap = {
            "active": False,
            "phase": "IDLE",
            "t_phase": 0.0,
            "t_set": 0.0,
            "samples": [],
            "v_base": None,
            "v_az": None,
            "v_alt": None,
        }

        # State + outputs (thread-safe)
        self._state = SystemState()
        self._state_lock = threading.Lock()

        self._latest_preview_jpeg: Optional[bytes] = None
        self._preview_lock = threading.Lock()

        # Preview stats
        self._t_last_preview = 0.0
        self._t_fps_view0 = _perf()
        self._n_view = 0

        # Control loop stats
        self._t_fps_loop0 = _perf()
        self._n_loop = 0

        # Parámetros de overlay en vivo (SEP)
        self._live_sep_overlay_enabled = False
        self._live_sep_params = {
            "sep_bw": int(self.cfg.sep.bw),
            "sep_bh": int(self.cfg.sep.bh),
            "sep_thresh_sigma": float(self.cfg.sep.thresh_sigma),
            "sep_minarea": int(self.cfg.sep.minarea),
            "max_det": int(self.cfg.platesolving.max_det),
        }

        # Estado inicial
        self._set_state_safe(camera_status="DISCONNECTED", camera_connected=False)
        self._set_state_safe(mount_status="DISCONNECTED", mount_connected=False)

        # Tracking fields (si existen)
        self._set_state_safe(
            tracking_enabled=False,
            tracking_mode="IDLE",
            tracking_resp=0.0,
            tracking_dx=0.0,
            tracking_dy=0.0,
            tracking_vx=0.0,
            tracking_vy=0.0,
            tracking_abs_resp=0.0,
            tracking_x_hat=0.0,
            tracking_y_hat=0.0,
            tracking_rate_az=0.0,
            tracking_rate_alt=0.0,
            tracking_calib_src="none",
            tracking_detA=0.0,
        )

        # Stacking fields (si existen)
        self._set_state_safe(
            stacking_enabled=self._stacking_enabled,
            stacking_mode="RUNNING" if self._stacking_enabled else "IDLE",
            stacking_status="ON" if self._stacking_enabled else "OFF",
            stacking_on=self._stacking_enabled,
        )

        # Platesolving fields (si existen)
        self._set_state_safe(
            platesolving_status="IDLE",
            platesolving_busy=False,
            platesolving_last_ok=False,
            platesolving_theta_deg=0.0,
            platesolving_dx_px=0.0,
            platesolving_dy_px=0.0,
            platesolving_resp=0.0,
            platesolving_n_inliers=0,
            platesolving_rms_px=0.0,
            platesolving_overlay=[],
            platesolving_guides=[],
            platesolving_debug_jpeg=None,
            platesolving_debug_info=None,
            platesolving_center_ra_deg=0.0,
            platesolving_center_dec_deg=0.0,
        )

        # GoTo fields
        self._set_state_safe(
            goto_busy=False,
            goto_status="IDLE",
            goto_synced=False,
            goto_last_error_arcsec=0.0,
            goto_J00=float(self._goto.model.J_deg_per_step[0,0]),
            goto_J01=float(self._goto.model.J_deg_per_step[0,1]),
            goto_J10=float(self._goto.model.J_deg_per_step[1,0]),
            goto_J11=float(self._goto.model.J_deg_per_step[1,1]),
        )

    # -------------------------
    # Platesolving config copy
    # -------------------------
    def _copy_platesolving_config(self, cfg: PlatesolvingConfig) -> PlatesolvingConfig:
        """
        Devuelve una copia de PlatesolvingConfig para evitar aliasing con defaults.
        """
        return replace(cfg)

    def _get_platesolving_cfg_snapshot(self) -> PlatesolvingConfig:
        with self._platesolving_cfg_lock:
            return self._copy_platesolving_config(self.cfg.platesolving)

    def _get_sep_cfg_snapshot(self) -> SepConfig:
        return replace(self.cfg.sep)

    def _get_camera_cfg_snapshot(self):
        return replace(self.cfg.camera)

    def _get_mount_cfg_snapshot(self):
        return replace(self.cfg.mount)

    def _get_goto_cfg_snapshot(self):
        return replace(self.cfg.goto)

    def _get_latest_frame(self) -> Optional[Frame]:
        if self._cam_stream is None:
            return None
        return self._cam_stream.latest()

    def _get_fps_capture(self) -> float:
        if self._cam_stream is None:
            return 0.0
        st = self._cam_stream.stats()
        return float(st.get("fps_capture", 0.0))

    def _get_live_frame_for_platesolving(self) -> Optional[np.ndarray]:
        if self._cam_stream is None:
            return None
        fr = self._cam_stream.latest()
        if fr is None:
            return None
        return fr.raw

    def _publish_platesolving_state(self, **kwargs: Any) -> None:
        result = kwargs.pop("platesolving_result", None)
        self._set_state_safe(**kwargs)
        if result is not None and bool(getattr(result, "success", False)):
            self._last_platesolving_result = result

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self) -> None:
        if self._thr is not None:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="AppRunner", daemon=True)
        self._thr.start()
        if self._stacking_enabled:
            self._stacking.start()
            log_info(self.out_log, "Stacking: worker started")
        log_info(self.out_log, "Runner: started")

    def stop(self) -> None:
        self._stop.set()

        # detener platesolving worker si existe
        self._platesolving_worker.stop()

        # detener GoTo worker
        self._goto_worker.stop()
        self._goto_worker.join(timeout=2.0)

        # detener mount move worker
        self._mount_move_worker.stop()
        self._mount_move_worker.join(timeout=2.0)

        thr = self._thr
        if thr is not None:
            thr.join(timeout=2.0)
        self._thr = None

        self._shutdown_camera()
        self._shutdown_mount()
        try:
            self._stacking.stop()
        except Exception as exc:
            log_error(self.out_log, "Stacking: stop failed", exc)

        log_info(self.out_log, "Runner: stopped")

    # -------------------------
    # UI entrypoints
    # -------------------------
    def enqueue(self, action: Action) -> None:
        self._actions.put(action)

    def get_state(self) -> SystemState:
        with self._state_lock:
            s = self._state
            return SystemState(**s.__dict__)

    def get_latest_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._latest_preview_jpeg

    # -------------------------
    # Internal helpers
    # -------------------------
    def _set_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def _set_state_safe(self, **kwargs: Any) -> None:
        """
        Setea solo atributos existentes en SystemState (para no romper si aún
        no agregaste campos).
        """
        with self._state_lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)

    def _get_tracking_enabled(self) -> bool:
        with self._state_lock:
            return bool(getattr(self._state, "tracking_enabled", False))

    def _tracking_keyframe_reset(self) -> None:
        try:
            self._tracking_state.key_obj_xy = "PENDING"
        except Exception as exc:
            log_error(self.out_log, "Tracking: failed to reset keyframe", exc)

    def _is_manual_move_active(self) -> bool:
        return bool(self._mount_move_worker.is_busy())

    def _mount_rate_safe(self, az: float, alt: float) -> None:
        if self._mount is None:
            return
        if self._is_manual_move_active():
            return
        try:
            self._mount.rate(float(az), float(alt))
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False, tracking_enabled=False, tracking_mode="IDLE")
            log_error(
                self.out_log,
                "Mount: RATE failed",
                exc,
                throttle_s=2.0,
                throttle_key="mount_rate",
            )

    def _goto_move_steps(self, axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        if self._mount is None:
            raise RuntimeError("mount not connected")
        self._mount.move_steps(axis, direction, steps, delay_us)

    def _pause_tracking_for_goto(self) -> bool:
        was_tracking = self._get_tracking_enabled()
        if was_tracking:
            self._set_state_safe(tracking_enabled=False)
            self._mount_rate_safe(0.0, 0.0)
        return was_tracking

    def _resume_tracking_after_goto(self) -> None:
        self._set_state_safe(tracking_enabled=True)
        self._tracking_keyframe_reset()

    def _pause_stacking_for_goto(self) -> bool:
        was_stacking = bool(self._stacking_enabled)
        if was_stacking:
            self._stacking_enabled = False
            self._set_state_safe(stacking_enabled=False, stacking_mode="IDLE", stacking_status="OFF", stacking_on=False)
        return was_stacking

    def _resume_stacking_after_goto(self) -> None:
        self._stacking_enabled = True
        self._stacking.start()
        self._set_state_safe(stacking_enabled=True, stacking_mode="RUNNING", stacking_status="ON", stacking_on=True)

    def _tracking_capture_objects(
        self,
        *,
        wait_for_seq: Optional[int] = None,
        timeout_s: float = 1.5,
    ) -> Optional[Tuple[Frame, np.ndarray]]:
        if self._cam_stream is None:
            return None

        t0 = _perf()
        while (_perf() - t0) < float(timeout_s):
            fr = self._cam_stream.latest()
            if fr is None:
                time.sleep(0.01)
                continue
            if wait_for_seq is not None and int(fr.seq) == int(wait_for_seq):
                time.sleep(0.005)
                continue

            raw16 = ensure_raw16_bayer(fr.raw)
            _, _, _, obj_xy = sep_detect_from_raw16(
                raw16,
                sep_bw=int(self.cfg.sep.bw),
                sep_bh=int(self.cfg.sep.bh),
                sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
                sep_minarea=int(self.cfg.sep.minarea),
                max_sources=None,
            )
            return fr, obj_xy

        return None

    def _tracking_pause_for_calib(self) -> bool:
        was_tracking = self._get_tracking_enabled()
        if was_tracking:
            self._set_state_safe(tracking_enabled=False)
            self._mount_rate_safe(0.0, 0.0)
            self._tracking_keyframe_reset()
        return was_tracking

    def _tracking_resume_after_calib(self, was_tracking: bool) -> None:
        if was_tracking:
            self._set_state_safe(tracking_enabled=True)
            self._tracking_keyframe_reset()

    def _tracking_calibrate_axis(self, axis: Axis) -> Optional[np.ndarray]:
        if self._cam_stream is None or self._mount is None:
            log_info(self.out_log, "Tracking: calibration skipped (camera/mount inactive)")
            return None
        if not self._mount.is_connected():
            log_info(self.out_log, "Tracking: calibration skipped (mount disconnected)")
            return None

        cfg = self._tracking_state.cfg.calib
        steps = max(1, int(cfg.cal_steps_init))
        max_steps = max(steps, int(cfg.cal_steps_max))
        delay_us = int(cfg.cal_delay_us)
        max_shift_px = max(
            float(self._tracking_state.cfg.keyframe.abs_max_px),
            float(self._tracking_state.cfg.rate.max_shift_per_frame_px),
            float(cfg.cal_target_px_max) * 2.0,
        )

        for attempt in range(int(cfg.cal_try_max)):
            before = self._tracking_capture_objects(timeout_s=1.5)
            if before is None:
                log_info(self.out_log, "Tracking: calibration failed (no frame)")
                return None
            fr0, obj0 = before
            if obj0.size == 0:
                log_info(self.out_log, "Tracking: calibration failed (no stars)")
                return None

            if self._mount_move_worker.is_busy():
                log_info(self.out_log, "Tracking: calibration skipped (mount busy)")
                return None
            try:
                self._mount.stop()
                self._mount.move_steps(axis, direction=1, steps=steps, delay_us=delay_us)
            except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
                log_error(self.out_log, "Tracking: calibration move failed", exc)
                return None

            after = self._tracking_capture_objects(wait_for_seq=fr0.seq, timeout_s=2.0)

            try:
                self._mount.move_steps(axis, direction=-1, steps=steps, delay_us=delay_us)
            except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
                log_error(self.out_log, "Tracking: calibration move-back failed", exc)

            if after is None:
                log_info(self.out_log, "Tracking: calibration failed (no post-move frame)")
                return None
            _, obj1 = after
            if obj1.size == 0:
                log_info(self.out_log, "Tracking: calibration failed (no post-move stars)")
                return None

            dx, dy, resp, matches = estimate_shift_from_objects(
                obj0,
                obj1,
                max_shift_px=max_shift_px,
            )
            mag = float(np.hypot(dx, dy))

            if float(resp) < float(cfg.cal_resp_min) or matches <= 0:
                log_info(self.out_log, f"Tracking: calibration low response (resp={resp:.3f})")
                return None

            if mag < float(cfg.cal_target_px_min) and steps < max_steps:
                steps = min(max_steps, steps * 2)
                log_info(self.out_log, f"Tracking: calibration retry (shift={mag:.2f}px, steps={steps})")
                continue

            if mag > float(cfg.cal_target_px_max) and steps > 1:
                steps = max(1, int(round(steps / 2)))
                log_info(self.out_log, f"Tracking: calibration retry (shift={mag:.2f}px, steps={steps})")
                continue

            if steps <= 0:
                raise ValueError("calibration steps must be positive")

            col = np.array([dx / float(steps), dy / float(steps)], dtype=np.float64)
            log_info(self.out_log, f"Tracking: calibration axis={axis.value} col={col}")
            return col

        log_info(self.out_log, "Tracking: calibration failed (max retries)")
        return None

    def _tracking_apply_calib_cols(self) -> bool:
        az_col = self._tracking_calib_cols.get("az")
        alt_col = self._tracking_calib_cols.get("alt")
        if az_col is None or alt_col is None:
            return False

        A = np.column_stack([az_col, alt_col]).astype(np.float64, copy=False)
        calib_set_A_micro(self._tracking_state, A, src="manual")
        self._set_state_safe(
            tracking_calib_src="manual",
            tracking_detA=float(self._tracking_state.cal_det),
        )
        log_info(self.out_log, f"Tracking: calibration applied A={A}")
        return True

    def _tracking_calib_reset(self) -> None:
        calib_reset(self._tracking_state)
        self._tracking_state.cfg.calib.calib_A = None
        self._tracking_state.cfg.calib.calib_b = None
        self._tracking_calib_cols.clear()
        self._set_state_safe(tracking_calib_src="none", tracking_detA=0.0)

    def _tracking_bootstrap_reset(self) -> None:
        boot = self._tracking_bootstrap
        boot["active"] = False
        boot["phase"] = "IDLE"
        boot["t_phase"] = 0.0
        boot["t_set"] = 0.0
        boot["samples"] = []
        boot["v_base"] = None
        boot["v_az"] = None
        boot["v_alt"] = None

    def _tracking_bootstrap_start(self, now_t: float) -> None:
        cfg = self._tracking_state.cfg.autoboost
        if not cfg.enabled:
            return
        if self._mount is None or not self._mount.is_connected():
            return
        boot = self._tracking_bootstrap
        boot["active"] = True
        boot["phase"] = "BASE"
        boot["t_phase"] = float(now_t)
        boot["t_set"] = float(now_t)
        boot["samples"] = []
        boot["v_base"] = None
        boot["v_az"] = None
        boot["v_alt"] = None
        self._mount_rate_safe(0.0, 0.0)
        log_info(self.out_log, "Tracking: bootstrap start")

    def _tracking_bootstrap_collect(self, vx: float, vy: float, resp_ok: bool) -> None:
        if not resp_ok:
            return
        boot = self._tracking_bootstrap
        if not boot["active"]:
            return
        boot["samples"].append((float(vx), float(vy)))

    def _tracking_bootstrap_finish(self, v_base: np.ndarray, v_az: np.ndarray, v_alt: np.ndarray) -> None:
        cfg = self._tracking_state.cfg.autoboost
        rate = float(cfg.rate)
        col_az = (v_az - v_base) / rate
        col_alt = (v_alt - v_base) / rate
        A = np.column_stack([col_az, col_alt]).astype(np.float64, copy=False)
        auto_set_from_A(self._tracking_state, A_micro=A, b_pxps=v_base, src="boot")
        self._set_state_safe(
            tracking_calib_src="boot",
            tracking_detA=float(self._tracking_state.auto.detA),
        )
        log_info(self.out_log, f"Tracking: bootstrap ok A={A}")
        self._tracking_bootstrap_reset()

    def _tracking_bootstrap_step(self, now_t: float) -> None:
        boot = self._tracking_bootstrap
        if not boot["active"]:
            return

        cfg = self._tracking_state.cfg.autoboost
        phase = str(boot["phase"])
        phase_dur = float(cfg.base_s if phase == "BASE" else cfg.axis_s)
        if (float(now_t) - float(boot["t_set"])) < float(cfg.settle_s):
            return

        if (float(now_t) - float(boot["t_phase"])) < phase_dur:
            return

        samples = np.array(boot["samples"], dtype=np.float64) if boot["samples"] else None
        if samples is None or samples.shape[0] < int(cfg.min_samples):
            log_info(self.out_log, f"Tracking: bootstrap retry (phase={phase}, samples={len(boot['samples'])})")
            boot["phase"] = "BASE"
            boot["t_phase"] = float(now_t)
            boot["t_set"] = float(now_t)
            boot["samples"] = []
            self._mount_rate_safe(0.0, 0.0)
            return

        v_mean = samples.mean(axis=0)
        if phase == "BASE":
            boot["v_base"] = v_mean
            boot["phase"] = "AZ"
            boot["t_phase"] = float(now_t)
            boot["t_set"] = float(now_t)
            boot["samples"] = []
            self._mount_rate_safe(float(cfg.rate), 0.0)
            log_info(self.out_log, f"Tracking: bootstrap base ok v0={v_mean}")
            return

        if phase == "AZ":
            boot["v_az"] = v_mean
            boot["phase"] = "ALT"
            boot["t_phase"] = float(now_t)
            boot["t_set"] = float(now_t)
            boot["samples"] = []
            self._mount_rate_safe(0.0, float(cfg.rate))
            log_info(self.out_log, f"Tracking: bootstrap az ok v1={v_mean}")
            return

        if phase == "ALT":
            boot["v_alt"] = v_mean
            self._mount_rate_safe(0.0, 0.0)
            v_base = boot["v_base"]
            v_az = boot["v_az"]
            v_alt = boot["v_alt"]
            if v_base is None or v_az is None or v_alt is None:
                raise ValueError("bootstrap state incomplete")
            self._tracking_bootstrap_finish(
                np.asarray(v_base, dtype=np.float64),
                np.asarray(v_az, dtype=np.float64),
                np.asarray(v_alt, dtype=np.float64),
            )

    def _tracking_bootstrap_calibration(self) -> None:
        was_tracking = self._tracking_pause_for_calib()

        az_col = self._tracking_calibrate_axis(Axis.AZ)
        if az_col is not None:
            self._tracking_calib_cols["az"] = az_col

        alt_col = self._tracking_calibrate_axis(Axis.ALT)
        if alt_col is not None:
            self._tracking_calib_cols["alt"] = alt_col

        if not self._tracking_apply_calib_cols():
            log_info(self.out_log, "Tracking: bootstrap calibration incomplete")

        self._tracking_resume_after_calib(was_tracking)

    # -------------------------
    # Camera
    # -------------------------
    def _shutdown_camera(self) -> None:
        if self._cam_stream is not None:
            try:
                self._cam_stream.stop()
            except Exception as exc:
                log_error(self.out_log, "Camera: stream stop failed", exc)
            self._cam_stream = None

        if self._cam_dev is not None:
            try:
                self._cam_dev.close()
            except Exception as exc:
                log_error(self.out_log, "Camera: device close failed", exc)
            self._cam_dev = None

        self._set_state_safe(
            camera_connected=False,
            camera_status="DISCONNECTED",
            fps_capture=0.0,
        )

    def _connect_camera(self, camera_index: int) -> None:
        self._shutdown_camera()
        self._set_state_safe(camera_status="CONNECTING", camera_connected=False)

        try:
            dev = POACameraDevice()
            info = dev.open(camera_index)

            stream = CameraStream(ring=3)
            stream.start(dev, self.cfg.camera, self.cfg.preview)

            self._cam_dev = dev
            self._cam_stream = stream

            self._set_state_safe(
                camera_connected=True,
                camera_status="OK",
            )
            log_info(
                self.out_log,
                f"Camera: connected id={info.camera_id} model={info.model} sensor={info.sensor} "
                f"usb3={info.is_usb3} bayer={info.bayer_pattern} max={info.max_w}x{info.max_h}",
            )
        except Exception as exc:
            self._shutdown_camera()
            self._set_state_safe(camera_connected=False, camera_status="ERR")
            log_error(self.out_log, "Camera: connect failed (is it open in another app?)", exc)

    def _apply_camera_param(self, name: str, value: Any) -> None:
        n = (name or "").strip()

        if n in ("exp_ms", "exposure_ms"):
            self.cfg.camera.exp_ms = float(value)
        elif n in ("gain",):
            self.cfg.camera.gain = int(value)
        elif n in ("auto_gain",):
            self.cfg.camera.auto_gain = bool(value)
        elif n in ("img_format",):
            self.cfg.camera.img_format = str(value)
        elif n in ("use_roi",):
            self.cfg.camera.use_roi = bool(value)
        elif n in ("roi_x",):
            self.cfg.camera.roi_x = int(value)
        elif n in ("roi_y",):
            self.cfg.camera.roi_y = int(value)
        elif n in ("roi_w",):
            self.cfg.camera.roi_w = int(value)
        elif n in ("roi_h",):
            self.cfg.camera.roi_h = int(value)
        elif n in ("binning", "bin_hw"):
            self.cfg.camera.binning = int(value)
        elif n in ("preview_view_hz",):
            self.cfg.preview.view_hz = float(value)
        elif n in ("preview_jpeg_quality",):
            self.cfg.preview.jpeg_quality = int(value)
        elif n in ("preview_stretch_plo",):
            self.cfg.preview.stretch_plo = float(value)
        elif n in ("preview_stretch_phi",):
            self.cfg.preview.stretch_phi = float(value)
        else:
            log_info(self.out_log, f"Camera: param ignorado (no soportado aún): {n}={value}")
            return

        self._restart_camera_stream_if_active(reason=f"{n} change")

    def _restart_camera_stream_if_active(self, *, reason: str) -> None:
        if self._cam_dev is None or self._cam_stream is None:
            return
        try:
            cam_index = int(self.cfg.camera.camera_index)
            log_info(self.out_log, f"Camera: reconfigure (restart stream) due to {reason}")
            self._connect_camera(cam_index)
        except Exception as exc:
            self._set_state_safe(camera_status="ERR")
            log_error(self.out_log, "Camera: failed to apply config (restart)", exc)

    def _reset_camera_defaults(self) -> None:
        self.cfg.camera = replace(self.default_cfg.camera)
        self._restart_camera_stream_if_active(reason="camera defaults reset")

    def _reset_preview_defaults(self) -> None:
        self.cfg.preview = replace(self.default_cfg.preview)
        self._restart_camera_stream_if_active(reason="preview defaults reset")

    def _reset_mount_defaults(self) -> None:
        self.cfg.mount = replace(self.default_cfg.mount)
        if self._mount is not None and self._mount.is_connected():
            self._mount_set_microsteps(self.cfg.mount.ms_az, self.cfg.mount.ms_alt)
        self._goto.model.kin.microsteps_az = int(self.cfg.mount.ms_az)
        self._goto.model.kin.microsteps_alt = int(self.cfg.mount.ms_alt)
        self._goto.model.init_from_mechanics()
        self._set_state_safe(
            goto_J00=float(self._goto.model.J_deg_per_step[0, 0]),
            goto_J01=float(self._goto.model.J_deg_per_step[0, 1]),
            goto_J10=float(self._goto.model.J_deg_per_step[1, 0]),
            goto_J11=float(self._goto.model.J_deg_per_step[1, 1]),
        )

    def _reset_tracking_defaults(self) -> None:
        self.cfg.tracking = replace(self.default_cfg.tracking)
        tracking_set_params(
            self._tracking_state,
            sigma_hp=self.cfg.tracking.sigma_hp,
            resp_min=self.cfg.tracking.resp_min,
        )
        self._tracking_keyframe_reset()

    def _reset_stacking_defaults(self) -> None:
        self.cfg.stacking = replace(self.default_cfg.stacking)
        self._stacking.engine.configure_from_cfg()

    def _reset_platesolving_defaults(self) -> None:
        with self._platesolving_cfg_lock:
            self.cfg.platesolving = replace(self.default_cfg.platesolving)
        self._live_sep_params = {
            "sep_bw": int(self.cfg.sep.bw),
            "sep_bh": int(self.cfg.sep.bh),
            "sep_thresh_sigma": float(self.cfg.sep.thresh_sigma),
            "sep_minarea": int(self.cfg.sep.minarea),
            "max_det": int(self.cfg.platesolving.max_det),
        }

    def _maybe_update_preview(self) -> None:
        if self._cam_stream is None:
            return

        view_hz = float(self.cfg.preview.view_hz)
        if view_hz <= 0.1:
            view_hz = 0.1

        now = _perf()
        if (now - self._t_last_preview) < (1.0 / view_hz):
            return

        fr = self._cam_stream.latest()
        if fr is None:
            return

        try:
            overlay_enabled = bool(self._live_sep_overlay_enabled)

            raw16 = ensure_raw16_bayer(fr.raw)
            raw16_work = raw16

            if overlay_enabled:
                _, u8_preview = make_preview_jpeg(
                    raw16_work,
                    plo=float(self.cfg.preview.stretch_plo),
                    phi=float(self.cfg.preview.stretch_phi),
                    jpeg_quality=int(self.cfg.preview.jpeg_quality),
                    sample_stride=4,
                )
                u8_preview = self._apply_live_sep_overlay(raw16_work, u8_preview)
                jpg = encode_jpeg(u8_preview, quality=int(self.cfg.preview.jpeg_quality))
            else:
                jpg, _ = make_preview_jpeg(
                    raw16_work,
                    plo=float(self.cfg.preview.stretch_plo),
                    phi=float(self.cfg.preview.stretch_phi),
                    jpeg_quality=int(self.cfg.preview.jpeg_quality),
                    sample_stride=4,
                )

            with self._preview_lock:
                self._latest_preview_jpeg = jpg

            self._t_last_preview = now

            self._n_view += 1
            if (now - self._t_fps_view0) >= 1.0:
                fps_view = self._n_view / (now - self._t_fps_view0)
                self._t_fps_view0 = now
                self._n_view = 0
                self._set_state_safe(fps_view=float(fps_view))

        except Exception as exc:
            log_error(self.out_log, "Preview: failed", exc)

    def _apply_live_sep_overlay(self, raw16: np.ndarray, u8_preview: np.ndarray) -> np.ndarray:
        try:
            params = dict(self._live_sep_params)
            _, _, _, obj_xy = sep_detect_from_raw16(
                raw16,
                sep_bw=int(params.get("sep_bw", 64)),
                sep_bh=int(params.get("sep_bh", 64)),
                sep_thresh_sigma=float(params.get("sep_thresh_sigma", 3.0)),
                sep_minarea=int(params.get("sep_minarea", 5)),
                max_sources=int(params.get("max_det", 250)),
            )

            if obj_xy is None or len(obj_xy) == 0:
                return u8_preview

            if u8_preview.ndim == 2:
                img = cv2.cvtColor(u8_preview, cv2.COLOR_GRAY2BGR)
            else:
                img = u8_preview.copy()

            h, w = img.shape[:2]
            for x, y in obj_xy:
                ix = int(round(float(x)))
                iy = int(round(float(y)))
                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    continue
                cv2.circle(img, (ix, iy), 6, (0, 255, 255), 1, lineType=cv2.LINE_AA)

            return img
        except Exception as exc:
            log_error(self.out_log, "Live SEP: overlay failed", exc, throttle_s=2.0, throttle_key="live_sep_overlay")
            return u8_preview

    # -------------------------
    # Mount
    # -------------------------
    def _shutdown_mount(self) -> None:
        if self._mount is not None:
            try:
                self._mount.disconnect()
            except Exception as exc:
                log_error(self.out_log, "Mount: disconnect failed", exc)
        self._mount = None
        self._set_state_safe(mount_connected=False, mount_status="DISCONNECTED")

    def _connect_mount(self, port: str, baudrate: int) -> None:
        self._shutdown_mount()
        self._set_state_safe(mount_status="CONNECTING", mount_connected=False)

        try:
            m = ArduinoMount()
            msg = m.connect(port=str(port), baud=int(baudrate))
            self._mount = m
            self._set_state_safe(mount_connected=True, mount_status="OK")
            log_info(self.out_log, f"Mount: connected ({msg})")
        except Exception as exc:
            self._shutdown_mount()
            self._set_state_safe(mount_connected=False, mount_status="ERR")
            log_error(self.out_log, "Mount: connect failed", exc)

    def _mount_stop(self) -> None:
        if self._mount is None:
            return
        try:
            self._mount.stop()
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False, tracking_enabled=False, tracking_mode="IDLE")
            log_error(self.out_log, "Mount: STOP failed", exc)

    def _mount_set_microsteps(self, az_div: int, alt_div: int) -> None:
        if self._mount is None or not self._mount.is_connected():
            return
        try:
            self._mount.stop()
            self._mount.set_microsteps(int(az_div), int(alt_div))
            self._goto.model.set_microsteps(int(az_div), int(alt_div))
            log_info(self.out_log, f"Mount: MS set (AZ={int(az_div)} ALT={int(alt_div)})")
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False, tracking_enabled=False, tracking_mode="IDLE")
            log_error(self.out_log, "Mount: MS failed", exc)

    def _mount_move_steps(self, axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        if self._mount is None or not self._mount.is_connected():
            return
        if self._mount_move_worker.is_busy():
            log_info(self.out_log, "Mount: MOVE ignored; previous move still running")
            return
        self._mount_move_worker.request(
            axis=axis,
            direction=direction,
            steps=steps,
            delay_us=delay_us,
        )

    # -------------------------
    # Platesolving (worker)
    # -------------------------
    def _platesolving_request(self, *, target: Any) -> None:
        """
        Encola un request para platesolving. Si hay uno pendiente, se reemplaza.
        """
        self._platesolving_worker.request(target=target)

    def _maybe_autosolve(self) -> None:
        cfg = self._get_platesolving_cfg_snapshot()
        if not bool(cfg.auto_solve):
            return
        target = str(self._platesolving_auto_target or "").strip()
        if not target:
            return
        st = self.get_state()
        if bool(getattr(st, "platesolving_busy", False)):
            return
        now = _perf()
        if (now - float(self._platesolving_last_auto_t)) < max(2.0, float(cfg.solve_every_s)):
            return
        self._platesolving_request(target=target)
        self._platesolving_last_auto_t = float(now)

    # -------------------------
    # Stacking save helper
    # -------------------------
    def _save_stacking(self, out_dir: str, basename: str, fmt: str) -> None:
        """
        Assemble the current mosaic of stacked tiles and save it to disk.

        Two files are produced:
          - a raw floating-point numpy array (.npy) capturing the full
            dynamic range of the mosaic;
          - a stretched PNG image (.png) for quick viewing.
        The output directory is created if necessary.  Errors are logged but
        otherwise ignored.

        Parameters
        ----------
        out_dir : str
            Directory in which to save the files.
        basename : str
            Base name for the output files; suffixes `_raw.npy` and
            `_stretch.png` will be appended.
        fmt : str
            Ignored; included for API compatibility.
        """
        eng = self._stacking.engine
        try:
            if eng.canvas is None or eng.canvas.num_tiles() == 0:
                log_info(self.out_log, "Stacking: save skipped (no data)")
                return

            canvas = eng.canvas
            tile_size = canvas.tile_size
            keys = list(canvas.tiles.keys())
            txs = [k[0] for k in keys]
            tys = [k[1] for k in keys]
            tx_min, tx_max = min(txs), max(txs)
            ty_min, ty_max = min(tys), max(tys)
            width = (tx_max - tx_min + 1) * tile_size
            height = (ty_max - ty_min + 1) * tile_size
            if eng.color_mode == "mono":
                out = np.zeros((height, width), dtype=np.float32)
                wgt = np.zeros((height, width), dtype=np.float32)
            else:
                out = np.zeros((height, width, 3), dtype=np.float32)
                wgt = np.zeros((height, width), dtype=np.float32)
            for (tx, ty), tile in canvas.tiles.items():
                x0 = (tx - tx_min) * tile_size
                y0 = (ty - ty_min) * tile_size
                tile_sum = tile.sum.astype(np.float32, copy=False)
                tile_w = tile.w.astype(np.float32, copy=False)
                if eng.color_mode == "mono":
                    out[y0 : y0 + tile_size, x0 : x0 + tile_size] += tile_sum
                    wgt[y0 : y0 + tile_size, x0 : x0 + tile_size] += tile_w
                else:
                    out[y0 : y0 + tile_size, x0 : x0 + tile_size] += tile_sum
                    wgt[y0 : y0 + tile_size, x0 : x0 + tile_size] += tile_w
            if eng.color_mode == "mono":
                mask = wgt > 0
                out[mask] = out[mask] / wgt[mask]
            else:
                mask = wgt > 0
                for c in range(3):
                    out[..., c][mask] = out[..., c][mask] / wgt[mask]
            # Create output directory
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # Save raw
            raw_path = os.path.join(out_dir, f"{basename}_raw.npy")
            np.save(raw_path, out)
            # Save stretched image
            u8 = stretch_to_u8(out)
            stretch_path = os.path.join(out_dir, f"{basename}_stretch.png")
            if eng.color_mode == "mono":
                cv2.imwrite(stretch_path, u8)
            else:
                cv2.imwrite(stretch_path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
            log_info(self.out_log, f"Stacking: saved raw to {raw_path} and stretch to {stretch_path}")
        except Exception as exc:
            log_error(self.out_log, "Stacking: save failed", exc)


    # -------------------------
    # Main loop
    # -------------------------
    def _run(self) -> None:
        dt_target = 1.0 / max(1.0, float(self.cfg.control_hz))
        t_last = _perf()

        while not self._stop.is_set():
            t0 = _perf()

            # 1) actions
            self._drain_actions(max_n=50)

            # 2) stats capture
            if self._cam_stream is not None:
                st = self._cam_stream.stats()
                self._set_state_safe(fps_capture=float(st.get("fps_capture", 0.0)))

            # 2b) tracking
            tracking_on = self._get_tracking_enabled()
            if tracking_on and (self._cam_stream is not None) and (self._mount is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    # Tracking en RAW16 + SEP
                    meta = dict(getattr(fr, "meta", {}) or {})
                    raw16 = ensure_raw16_bayer(fr.raw)
                    _, _, _, obj_xy = sep_detect_from_raw16(
                        raw16,
                        sep_bw=int(self.cfg.sep.bw),
                        sep_bh=int(self.cfg.sep.bh),
                        sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
                        sep_minarea=int(self.cfg.sep.minarea),
                        max_sources=None,
                    )

                    calib_ready = bool(self._tracking_state.cal_A_pinv is not None) or bool(
                        self._tracking_state.auto.ok and self._tracking_state.auto.A_pinv is not None
                    )
                    boot_active = bool(self._tracking_bootstrap.get("active", False))
                    if (not boot_active) and (not calib_ready):
                        self._tracking_bootstrap_start(_now_s())
                        boot_active = bool(self._tracking_bootstrap.get("active", False))

                    try:
                        out = tracking_step(
                            self._tracking_state,
                            obj_xy,
                            now_t=_now_s(),
                            tracking_enabled=bool(tracking_on and (not boot_active)),
                        )
                    except TypeError as exc:
                        log_error(self.out_log, "Tracking: falling back to legacy tracking_step signature", exc, throttle_s=60.0, throttle_key="tracking_step_signature")
                        out = tracking_step(self._tracking_state, img_det)

                    if boot_active:
                        self._tracking_bootstrap_collect(float(out.vx), float(out.vy), resp_ok=bool(out.ok))
                        self._tracking_bootstrap_step(_now_s())
                    elif not self._is_manual_move_active():
                        try:
                            self._mount.rate(float(out.rate_az), float(out.rate_alt))
                        except Exception as exc:
                            self._set_state_safe(mount_status="ERR", mount_connected=False, tracking_enabled=False, tracking_mode="IDLE")
                            log_error(self.out_log, "Tracking: mount.rate failed", exc, throttle_s=2.0, throttle_key="tracking_mount_rate")

                    self._set_state_safe(
                        tracking_mode=str(out.mode),
                        tracking_resp=float(out.resp),
                        tracking_dx=float(out.dx),
                        tracking_dy=float(out.dy),
                        tracking_vx=float(out.vx),
                        tracking_vy=float(out.vy),
                        tracking_abs_resp=float(out.abs_resp),
                        tracking_x_hat=float(out.x_hat),
                        tracking_y_hat=float(out.y_hat),
                        tracking_rate_az=float(out.rate_az),
                        tracking_rate_alt=float(out.rate_alt),
                        tracking_calib_src=str(out.calib_src),
                        tracking_detA=float(out.detA),
                    )
            else:
                if self._mount is not None:
                    self._mount_rate_safe(0.0, 0.0)
                self._tracking_bootstrap_reset()
                self._set_state_safe(
                    tracking_mode="IDLE",
                    tracking_rate_az=0.0,
                    tracking_rate_alt=0.0,
                )

            # 2c) stacking
            if self._stacking_enabled and (self._cam_stream is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    raw16 = ensure_raw16_bayer(fr.raw)
                    self._stacking.enqueue_frame(raw16.copy(), t=_now_s())

            # 2d) publish stacking metrics
            m = self._stacking.engine.metrics
            self._set_state_safe(
                stacking_enabled=bool(m.enabled),
                stacking_mode="RUNNING" if m.enabled else "IDLE",
                stacking_status="ON" if m.enabled else "OFF",
                stacking_on=bool(m.enabled),
                stacking_fps=float(getattr(m, "stacking_fps", 0.0)),
                stacking_tiles_used=int(getattr(m, "tiles_used", 0)),
                stacking_tiles_evicted=int(getattr(m, "tiles_evicted", 0)),
                stacking_frames_in=int(getattr(m, "frames_in", 0)),
                stacking_frames_used=int(getattr(m, "frames_used", 0)),
                stacking_frames_dropped=int(getattr(m, "frames_dropped", 0)),
                stacking_frames_rejected=int(getattr(m, "frames_rejected", 0)),
                stacking_last_resp=float(getattr(m, "last_resp", 0.0)),
                stacking_last_dx=float(getattr(m, "last_dx", 0.0)),
                stacking_last_dy=float(getattr(m, "last_dy", 0.0)),
                stacking_last_theta_deg=float(getattr(m, "last_theta_deg", 0.0)),
                stacking_preview_jpeg=self._stacking.engine.get_preview_jpeg(),
            )

            # 2e) platesolving autosolve scheduling (if enabled)
            self._maybe_autosolve()

            # 3) preview
            self._maybe_update_preview()

            # 4) loop stats
            t1 = _perf()
            frame_ms = (t1 - t0) * 1000.0
            self._set_state_safe(frame_ms=float(frame_ms))

            self._n_loop += 1
            if (t1 - self._t_fps_loop0) >= 1.0:
                fps_loop = self._n_loop / (t1 - self._t_fps_loop0)
                self._t_fps_loop0 = t1
                self._n_loop = 0
                self._set_state_safe(fps_control_loop=float(fps_loop))

            # 5) sleep
            now = _perf()
            elapsed = now - t_last
            t_last = now
            slack = dt_target - elapsed
            if slack > 0:
                time.sleep(slack)

    # -------------------------
    # Actions
    # -------------------------
    def _drain_actions(self, max_n: int = 50) -> None:
        for _ in range(max_n):
            try:
                act = self._actions.get_nowait()
            except queue.Empty:
                return

            try:
                self._handle_action(act)
            except Exception as exc:
                if act.type in (ActionType.CAMERA_CONNECT, ActionType.CAMERA_SET_PARAM):
                    self._set_state_safe(camera_status="ERR", camera_connected=False)

                if act.type in (
                    ActionType.MOUNT_CONNECT,
                    ActionType.MOUNT_STOP,
                    ActionType.MOUNT_SET_MICROSTEPS,
                    ActionType.MOUNT_MOVE_STEPS,
                    ActionType.TRACKING_START,
                    ActionType.TRACKING_STOP,
                    ActionType.TRACKING_SET_PARAMS,
                    ActionType.TRACKING_CALIB_AZ,
                    ActionType.TRACKING_CALIB_ALT,
                    ActionType.TRACKING_CALIB_RESET,
                    ActionType.TRACKING_AUTO_RESET,
                    ActionType.TRACKING_BOOTSTRAP,
                    ActionType.STACKING_START,
                    ActionType.STACKING_STOP,
                    ActionType.STACKING_RESET,
                    ActionType.STACKING_SET_PARAMS,
                    ActionType.PLATESOLVING_RUN,
                    ActionType.PLATESOLVING_SET_PARAMS,
                    ActionType.GOTO_AUTOCALIBRATE,
                ):
                    if act.type in (
                        ActionType.MOUNT_CONNECT,
                        ActionType.MOUNT_STOP,
                        ActionType.MOUNT_SET_MICROSTEPS,
                        ActionType.MOUNT_MOVE_STEPS,
                    ):
                        self._set_state_safe(mount_status="ERR", mount_connected=False)
                        self._set_state_safe(tracking_enabled=False, tracking_mode="IDLE")

                log_error(self.out_log, f"Action failed: {act.type}", exc)

    def _handle_action(self, act: Action) -> None:
        t = act.type
        p = act.payload

        # ---- Camera ----
        if t == ActionType.CAMERA_CONNECT:
            idx = int(p.get("camera_index", 0))
            self.cfg.camera.camera_index = idx
            self._connect_camera(idx)
            return

        if t == ActionType.CAMERA_DISCONNECT:
            self._shutdown_camera()
            log_info(self.out_log, "Camera: disconnected")
            return

        if t == ActionType.CAMERA_SET_PARAM:
            name = str(p.get("name", ""))
            value = p.get("value", None)
            self._apply_camera_param(name, value)
            return

        if t == ActionType.RESET_CAMERA_DEFAULTS:
            self._reset_camera_defaults()
            log_info(self.out_log, "Camera: RESET_DEFAULTS")
            return

        if t == ActionType.RESET_PREVIEW_DEFAULTS:
            self._reset_preview_defaults()
            log_info(self.out_log, "Preview: RESET_DEFAULTS")
            return

        # ---- Mount ----
        if t == ActionType.MOUNT_CONNECT:
            port = str(p.get("port", ""))
            baud = int(p.get("baudrate", 115200))
            self._connect_mount(port, baud)
            return

        if t == ActionType.MOUNT_DISCONNECT:
            self._shutdown_mount()
            log_info(self.out_log, "Mount: disconnected")
            return

        if t == ActionType.MOUNT_STOP:
            self._mount_stop()
            self._tracking_keyframe_reset()
            return

        if t == ActionType.MOUNT_SET_MICROSTEPS:
            az_div = int(p.get("az_div", 64))
            alt_div = int(p.get("alt_div", 64))
            self._mount_set_microsteps(az_div, alt_div)
            return

        if t == ActionType.MOUNT_MOVE_STEPS:
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            steps = int(p.get("steps", 600))
            delay_us = int(p.get("delay_us", 1800))
            self._mount_move_steps(axis, direction, steps, delay_us)
            self._tracking_keyframe_reset()
            return

        if t == ActionType.RESET_MOUNT_DEFAULTS:
            self._reset_mount_defaults()
            log_info(self.out_log, "Mount: RESET_DEFAULTS")
            return

        # ---- Tracking ----
        if t == ActionType.TRACKING_START:
            self._set_state_safe(tracking_enabled=True)
            self._mount_rate_safe(0.0, 0.0)
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: START")
            return

        if t == ActionType.TRACKING_STOP:
            self._set_state_safe(tracking_enabled=False)
            self._mount_rate_safe(0.0, 0.0)
            log_info(self.out_log, "Tracking: STOP")
            return

        if t == ActionType.TRACKING_SET_PARAMS:
            if isinstance(p, dict):
                tracking_set_params(self._tracking_state, **p)
                log_info(self.out_log, f"Tracking: SET_PARAMS {_format_params(p)}")
            return

        if t == ActionType.RESET_TRACKING_DEFAULTS:
            self._reset_tracking_defaults()
            log_info(self.out_log, "Tracking: RESET_DEFAULTS")
            return

        if t == ActionType.TRACKING_CALIB_RESET:
            self._tracking_calib_reset()
            self._tracking_bootstrap_reset()
            log_info(self.out_log, "Tracking: CALIB_RESET")
            return

        if t == ActionType.TRACKING_AUTO_RESET:
            auto_reset(self._tracking_state, src="none")
            self._tracking_bootstrap_reset()
            log_info(self.out_log, "Tracking: AUTO_RESET")
            return

        if t == ActionType.TRACKING_CALIB_AZ:
            was_tracking = self._tracking_pause_for_calib()
            az_col = self._tracking_calibrate_axis(Axis.AZ)
            if az_col is not None:
                self._tracking_calib_cols["az"] = az_col
                self._tracking_apply_calib_cols()
            self._tracking_resume_after_calib(was_tracking)
            return

        if t == ActionType.TRACKING_CALIB_ALT:
            was_tracking = self._tracking_pause_for_calib()
            alt_col = self._tracking_calibrate_axis(Axis.ALT)
            if alt_col is not None:
                self._tracking_calib_cols["alt"] = alt_col
                self._tracking_apply_calib_cols()
            self._tracking_resume_after_calib(was_tracking)
            return

        if t == ActionType.TRACKING_BOOTSTRAP:
            self._tracking_bootstrap_start(_now_s())
            return

        # ---- Stacking ----
        if t == ActionType.STACKING_START:
            self._stacking_enabled = True
            self._stacking.start()
            self._set_state_safe(stacking_enabled=True, stacking_mode="RUNNING", stacking_status="ON", stacking_on=True)
            log_info(self.out_log, "Stacking: START")
            return

        if t == ActionType.STACKING_STOP:
            self._stacking_enabled = False
            self._stacking.stop()
            self._set_state_safe(stacking_enabled=False, stacking_mode="IDLE", stacking_status="OFF", stacking_on=False)
            log_info(self.out_log, "Stacking: STOP")
            return

        if t == ActionType.STACKING_RESET:
            self._stacking.reset()
            log_info(self.out_log, "Stacking: RESET")
            return

        if t == ActionType.STACKING_SET_PARAMS:
            if isinstance(p, dict):
                self._stacking.set_params(**p)
                log_info(self.out_log, f"Stacking: SET_PARAMS {list(p.keys())}")
            return

        if t == ActionType.RESET_STACKING_DEFAULTS:
            self._reset_stacking_defaults()
            log_info(self.out_log, "Stacking: RESET_DEFAULTS")
            return

        # Save stacked mosaic (raw + stretch)
        if t == ActionType.STACKING_SAVE:
            # Payload should contain out_dir, basename, fmt; defaults provided
            if isinstance(p, dict):
                out_dir = str(p.get("out_dir", "stack_output"))
                basename = str(p.get("basename", "stack"))
                fmt = str(p.get("fmt", "png"))
                self._save_stacking(out_dir, basename, fmt)
            else:
                # Fallback to default directory and timestamp
                out_dir = "stack_output"
                basename = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                fmt = "png"
                self._save_stacking(out_dir, basename, fmt)
            return

        # ---- Platesolving ----
        if t == ActionType.PLATESOLVING_SET_PARAMS:
            # Permite actualizar PlatesolvingConfig desde UI sin reimportar
            # Ej: {'pixel_size_m': 2.9e-6, 'focal_m': 0.9, 'gmax': 14.5, ...}
            if isinstance(p, dict):
                payload = dict(p)
                if "auto_target" in payload:
                    self._platesolving_auto_target = str(payload.pop("auto_target") or "")

                # Rebuild dataclass con campos existentes
                with self._platesolving_cfg_lock:
                    d = dict(self.cfg.platesolving.__dict__)
                    for k, v in payload.items():
                        if k in d:
                            d[k] = v
                    self.cfg.platesolving = PlatesolvingConfig(**d)
                if payload:
                    log_info(self.out_log, "Platesolving: params updated")
            return

        if t == ActionType.RESET_PLATESOLVING_DEFAULTS:
            self._reset_platesolving_defaults()
            log_info(self.out_log, "Platesolving: RESET_DEFAULTS")
            return

        # ---- Live SEP overlay ----
        if t == ActionType.LIVE_SEP_SET_PARAMS:
            if isinstance(p, dict):
                enabled = p.get("enabled", self._live_sep_overlay_enabled)
                self._live_sep_overlay_enabled = bool(enabled)
                for key in ("sep_bw", "sep_bh", "sep_thresh_sigma", "sep_minarea", "max_det"):
                    if key in p:
                        self._live_sep_params[key] = p.get(key)
                if "sep_bw" in p:
                    self.cfg.sep.bw = int(p.get("sep_bw"))
                if "sep_bh" in p:
                    self.cfg.sep.bh = int(p.get("sep_bh"))
                if "sep_thresh_sigma" in p:
                    self.cfg.sep.thresh_sigma = float(p.get("sep_thresh_sigma"))
                if "sep_minarea" in p:
                    self.cfg.sep.minarea = int(p.get("sep_minarea"))
                self._goto.cfg.sep = self.cfg.sep
                log_info(self.out_log, "Live SEP: params updated")
            return


        if t == ActionType.PLATESOLVING_RUN:
            # Payload esperado:
            #  - target: str|tuple|dict (ver platesolving.py)
            #  - (opcional) gaia_username / gaia_password (persistir)
            target = p.get("target", None)

            # Si vienen credenciales, persistirlas
            user = str(p.get("gaia_username", "")).strip()
            pw = str(p.get("gaia_password", "")).strip()
            if user and pw:
                save_gaia_auth(user, pw)
                log_info(self.out_log, "Platesolving: Gaia credentials saved")

            self._platesolving_request(target=target)
            log_info(self.out_log, "Platesolving: RUN source=live")
            return

        # ---- GoTo ----
        if t == ActionType.MOUNT_SYNC:
            # Sync usando el último platesolving OK
            sol = getattr(self, '_last_platesolving_result', None)
            if sol is None or not bool(getattr(sol, 'success', False)):
                log_info(self.out_log, 'GoTo: sync failed (no successful platesolving cached)')
                self._set_state_safe(goto_synced=False, goto_status='SYNC_ERR')
                return
            ok = False
            try:
                ok = bool(self._goto.sync_from_platesolving(sol))
            except Exception as exc:
                log_error(self.out_log, 'GoTo: sync exception', exc)
            self._set_state_safe(goto_synced=bool(ok), goto_status='SYNC_OK' if ok else 'SYNC_ERR')
            log_info(self.out_log, f"GoTo: sync {'OK' if ok else 'ERR'}")
            return

        if t == ActionType.MOUNT_GOTO:
            target = p.get('target', {})
            self._goto_worker.request(kind='goto', target=target, params=p)
            return

        if t == ActionType.GOTO_CALIBRATE:
            params = p.get('params', {})
            self._goto_worker.request(kind='calibrate', target=None, params=params)
            return

        if t == ActionType.GOTO_AUTOCALIBRATE:
            params = p.get('params', {})
            self._goto_worker.request(kind='autocal', target=None, params=params)
            return

        if t == ActionType.GOTO_CANCEL:
            self._goto_worker.cancel()
            self._mount_stop()
            self._set_state_safe(goto_busy=False, goto_status='CANCELLED')
            return

        # ---- Otros ----        # ---- Otros ----
        log_info(self.out_log, f"Unknown or unhandled action type: {t}")
