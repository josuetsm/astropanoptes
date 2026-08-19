# app_runner.py
from __future__ import annotations

import os
import json
import queue
import collections
import threading
import time
from pathlib import Path
import datetime as _dt

from dataclasses import replace
from typing import Optional, Any, Callable, Dict, List, Tuple, Sequence

import cv2
import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from astropy.utils import iers

from ap_types import (
    AppState,
    Axis,
    CameraStatus,
    Frame,
    GotoStatus,
    MountStatus,
    PlatesolvingStatus,
    StackingStatus,
    TrackingMode,
    TrackingStatus,
)
from config import AppConfig, PlatesolvingConfig, SepConfig
from actions import (
    Action,
    ActionType,
    camera_connect,
    camera_disconnect,
    camera_record_raw,
    camera_stop_record_raw,
    camera_set_param,
    camera_set_params,
    goto_autocalibrate,
    goto_estimate_roll,
    goto_calibrate,
    goto_fit_model,
    goto_reset,
    goto_restore_last_log,
    goto_cancel,
    goto_list_samples,
    goto_prune_outliers,
    goto_validate_sample,
    expected_stars_set_params,
    live_sep_set_params,
    mount_connect,
    mount_disconnect,
    mount_goto,
    mount_move_steps,
    mount_set_microsteps,
    mount_stop,
    mount_sync,
    platesolving_download_current_field,
    platesolving_run,
    platesolving_set_params,
    stacking_reset,
    stacking_save,
    stacking_set_params,
    stacking_start,
    stacking_stop,
    tracking_set_params as tracking_set_params_action,
    tracking_start,
    tracking_stop,
)
from logging_utils import log_info, log_error
from gaia_cache import (
    GaiaCacheMissError,
    bright_healpix_cone_with_mag,
    gaia_healpix_cone_with_mag,
    gaia_healpix_coverage,
)

from camera_poa import POACameraDevice, CameraStream
from imaging import ensure_raw16_bayer
from preview import make_preview_jpeg, encode_jpeg
from transmission_error import (
    TransmissionErrorCollector,
    gain_from_tracking_matrix,
)
from mount_arduino import (
    ArduinoMount,
    estimate_firmware_move_duration_s,
    resolve_common_microsteps,
)
from simulation import SimulatedCameraStream, SimulatedMount, SimulationState, restore_iers_after_demo

from tracking import (
    TrackingOutput,
    auto_reset,
    make_tracking_state,
    reset_tracker,
    tracking_step,
    tracking_set_params,
)
from stacking import StackingWorker

from sep_utils import sep_detect_from_raw16

from platesolving import (
    ObserverConfig,
    PlatesolvingWorker,
    expected_field_rotation_deg,
    parse_target_to_icrs,
    project_catalog_to_pixels,
    save_gaia_auth,
    load_gaia_auth,
)

from goto import (
    GoToController,
    GoToConfig as GoToRuntimeConfig,
    GoToModel,
    MountKinematics,
    GoToWorker,
    _platesolving_result_obstime,
    platesolving_center_to_altaz_deg,
    platesolving_roll_sample_deg,
    roll_axis_distance_deg,
)
from mount_arduino import MountMoveWorker
from workers import BaseWorker, SaveWorker


def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)



def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return "(none)"
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _axis_sign_from_invert(invert_flag: bool) -> int:
    # invert=True means user-facing +direction maps to opposite physical motion.
    # Use the same sign in kinematics so model Alt/Az evolves in physical direction.
    return -1 if bool(invert_flag) else +1


class _CoalescingCallbackWorker(BaseWorker):
    def __init__(self, *, name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        super().__init__(name=name, idle_sleep_s=0.005)
        self._callback = callback

    def _handle_request(self, request: Dict[str, Any]) -> None:
        self._callback(request)


class AppRunner:
    """
    Orquestador principal (runtime).

    Responsabilidades:
    - Mantener la cámara capturando a máxima FPS (CameraStream).
    - Ejecutar un loop estable (control_hz) que:
        - aplica actions
        - actualiza AppState
        - genera preview JPEG a view_hz
        - ejecuta tracking y corrige con MOVE discreto si tracking está ON
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
        self.cfg.simulation = replace(cfg.simulation)
        common_ms = resolve_common_microsteps(int(self.cfg.mount.ms_az), int(self.cfg.mount.ms_alt), default_ms=64)
        self.cfg.mount.ms_az = int(common_ms)
        self.cfg.mount.ms_alt = int(common_ms)
        self.out_log = out_log

        self._actions: "queue.Queue[Action]" = queue.Queue()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._raw_record_lock = threading.Lock()
        self._raw_record_active = False
        self._raw_record_thread: Optional[threading.Thread] = None
        self._raw_record_stop = threading.Event()

        # Subsystems
        self._cam_dev: Optional[POACameraDevice] = None
        self._cam_stream: Optional[Any] = None
        self._mount: Optional[Any] = None
        self._simulation_state: Optional[SimulationState] = None

        # Tracking subsystem
        self._tracking_state_lock = threading.RLock()
        self._tracking_result_lock = threading.Lock()
        self._tracking_generation = 0
        self._tracking_worker_error: Optional[Exception] = None
        self._tracking_state = make_tracking_state()
        tracking_set_params(
            self._tracking_state,
            resp_min=self.cfg.tracking.resp_min,
            align_median_k=int(self.cfg.stacking.align_median_k),
            align_smooth_k=int(self.cfg.stacking.smooth_k),
            align_max_shift_px=float(self.cfg.stacking.max_shift_px),
            align_use_subpixel=bool(self.cfg.stacking.use_subpixel),
        )

        # Stacking subsystem
        self._stacking = StackingWorker(self.cfg)
        self._stacking_enabled = bool(self.cfg.stacking.enabled_init)
        self._save_worker = SaveWorker(self._handle_save_request)

        # Platesolving subsystem
        self._platesolving_cfg_lock = threading.Lock()
        self._platesolving_last_auto_t = 0.0
        self._platesolving_auto_target: str = ""
        self._gaia_download_lock = threading.Lock()
        self._gaia_download_thread: Optional[threading.Thread] = None
        self._gaia_coverage_lock = threading.Lock()
        self._gaia_coverage_thread: Optional[threading.Thread] = None
        self._gaia_coverage_pending = False
        self._gaia_coverage_version = 0
        self._gaia_coverage_cache: Optional[Dict[str, object]] = None
        self._gaia_coverage_error: Optional[str] = None

        # Config platesolving (runtime copy, actualizable desde UI por action)
        self._platesolving_observer = ObserverConfig()  # Estación Central por defecto.
        # "live" = último frame de la cámara; "stack" = mosaico acumulado.
        self._platesolving_source: str = "live"
        self._transmission_collector: Optional[TransmissionErrorCollector] = None
        self._transmission_gain_ref: Optional[np.ndarray] = None
        self._platesolving_stack_info: Optional[Dict[str, Any]] = None
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
            gear_reduction_az=45.0,
            gear_reduction_alt=45.0,
            axis_sign_az=+1,
            axis_sign_alt=_axis_sign_from_invert(self.cfg.mount.invert_alt),
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
            slew_min_delay_us=int(self.cfg.goto.slew_min_delay_us),
            slew_full_speed_distance_deg=float(self.cfg.goto.slew_full_speed_distance_deg),
            max_unfitted_goto_deg=float(self.cfg.goto.max_unfitted_goto_deg),
            max_goto_distance_deg=float(self.cfg.goto.max_goto_distance_deg),
            settle_s=float(self.cfg.goto.settle_s),
            backlash_steps_az=int(self.cfg.goto.backlash_steps_az),
            backlash_steps_alt=int(self.cfg.goto.backlash_steps_alt),
            stages=int(self.cfg.goto.stages),
            platesolving_feedback=bool(self.cfg.goto.platesolving_feedback),
        )
        self._goto = GoToController(cfg=goto_cfg, model=GoToModel(kin=kin))
        self._last_platesolving_result: Optional[Any] = None
        self._mount_move_worker = MountMoveWorker(
            get_mount=lambda: self._mount,
            note_manual_move=self._goto.model.note_manual_move,
            get_last_direction=self._goto.model.last_move_direction,
            set_last_direction=self._goto.model.set_last_move_direction,
            get_backlash_steps=lambda axis: (
                self._goto.model.backlash_steps_az
                if axis == Axis.AZ
                else self._goto.model.backlash_steps_alt
            ),
            publish_state=self._update_state,
            operation_finished=lambda: self._finish_operation("mount_move"),
            out_log=self.out_log,
        )
        self._manual_move_active_until_s: Dict[str, float] = {
            Axis.AZ.value: 0.0,
            Axis.ALT.value: 0.0,
        }
        self._rate_emul_lock = threading.Lock()
        self._rate_emul_last_t: Optional[float] = None
        self._rate_emul_acc_az: float = 0.0
        self._rate_emul_acc_alt: float = 0.0
        self._rate_emul_active: bool = False
        self._tracking_last_frame_token: Optional[float] = None
        self._tracking_last_output: Optional[Any] = None
        self._tracking_last_cmd_az: float = 0.0
        self._tracking_last_cmd_alt: float = 0.0
        self._tracking_ff_hold_az: float = 0.0
        self._tracking_ff_hold_alt: float = 0.0
        self._tracking_ff_last_valid_t: Optional[float] = None
        self._tracking_ff_last_compute_t: Optional[float] = None
        self._tracking_ff_cached: Tuple[float, float, bool] = (0.0, 0.0, False)
        self._stacking_last_frame_token: Optional[float] = None
        self._tracking_worker = _CoalescingCallbackWorker(
            name="TrackingWorker",
            callback=self._process_tracking_request,
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
            rate_mount=self._mount_rate_safe,
            move_steps=self._goto_move_steps,
            stop_mount=self._mount_stop,
            out_log=self.out_log,
        )
        self._operation_lock = threading.Lock()
        self._operation_started = {
            "platesolving": 0,
            "goto": 0,
            "mount_move": 0,
            "camera_record": 0,
        }
        self._operation_finished = {
            "platesolving": 0,
            "goto": 0,
            "mount_move": 0,
            "camera_record": 0,
        }
        # State + outputs (thread-safe)
        self._state = AppState()
        self._state_lock = threading.Lock()

        self._latest_preview_jpeg: Optional[bytes] = None
        self._preview_lock = threading.Lock()
        self._preview_config_lock = threading.RLock()
        self._preview_generation = 0
        self._preview_last_frame_token: Optional[float] = None
        self._preview_worker = _CoalescingCallbackWorker(
            name="PreviewWorker",
            callback=self._render_preview_request,
        )

        # Preview stats
        self._t_last_preview = 0.0
        self._t_last_pointing = 0.0
        self._t_last_state_publish = 0.0
        self._t_fps_view0 = _perf()
        self._n_view = 0

        # Control loop stats
        self._t_fps_loop0 = _perf()
        self._n_loop = 0
        profile_samples = max(300, int(max(1.0, float(self.cfg.control_hz)) * 30.0))
        self._performance_lock = threading.Lock()
        self._performance_samples = {
            name: collections.deque(maxlen=profile_samples)
            for name in (
                "actions_ms",
                "state_ms",
                "tracking_ms",
                "stacking_ms",
                "autosolve_ms",
                "pointing_ms",
                "preview_ms",
                "total_ms",
            )
        }
        self._last_stall_log_t = 0.0

        # Parámetros de overlay en vivo (SEP)
        self._live_sep_overlay_enabled = False
        self._live_sep_params = {
            "sep_bw": int(self.cfg.sep.bw),
            "sep_bh": int(self.cfg.sep.bh),
            "sep_thresh_sigma": float(self.cfg.sep.thresh_sigma),
            "sep_minarea": int(self.cfg.sep.minarea),
            "max_det": int(self.cfg.platesolving.max_det),
        }
        self._expected_stars_overlay_enabled = False
        self._expected_stars_mag_limit = float(self.cfg.preview.expected_stars_mag_limit)
        self._expected_stars_max = int(self.cfg.preview.expected_stars_max)
        self._expected_stars_catalog: Optional[Any] = None
        self._expected_stars_catalog_center: Optional[SkyCoord] = None
        self._expected_stars_catalog_radius_deg = 0.0
        self._expected_stars_catalog_source = ""

        # Estado inicial
        self._update_state(
            {
                "camera": {"status": CameraStatus.DISCONNECTED, "connected": False},
                "mount": {"status": MountStatus.DISCONNECTED, "connected": False},
            }
        )

        self._update_state(
            {
                "tracking": {
                    "enabled": False,
                    "status": TrackingStatus.OFF,
                    "mode": TrackingMode.IDLE,
                    "ff_enabled": bool(self.cfg.tracking.sidereal_ff_enabled),
                    "ff_ready": False,
                    "resp": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "abs_resp": 0.0,
                    "rate_az": 0.0,
                    "rate_alt": 0.0,
                    "rate_fb_az": 0.0,
                    "rate_fb_alt": 0.0,
                    "rate_ff_az": 0.0,
                    "rate_ff_alt": 0.0,
                    "calib_src": "none",
                    "calib_det": 0.0,
                    "n_det": 0,
                    "measurement_valid": False,
                    "measurement_reason": "off",
                    "measurement_source": "none",
                    "error_x_px": 0.0,
                    "error_y_px": 0.0,
                    "error_px": 0.0,
                    "lock_conf": 0.0,
                    "fail_count": 0,
                },
                "stacking": {
                    "enabled": self._stacking_enabled,
                    "status": StackingStatus.RUNNING if self._stacking_enabled else StackingStatus.OFF,
                },
                "platesolving": {
                    "status": PlatesolvingStatus.IDLE,
                    "busy": False,
                    "last_ok": False,
                    "theta_deg": 0.0,
                    "dx_px": 0.0,
                    "dy_px": 0.0,
                    "resp": 0.0,
                    "n_inliers": 0,
                    "rms_px": 0.0,
                    "overlay": [],
                    "guides": [],
                    "debug_jpeg": None,
                    "debug_info": None,
                    "center_ra_deg": 0.0,
                    "center_dec_deg": 0.0,
                },
                "goto": {
                    "busy": False,
                    "status": GotoStatus.IDLE,
                    "synced": False,
                    "pointing_valid": False,
                    "pointing_az_deg": 0.0,
                    "pointing_alt_deg": 0.0,
                    "pointing_ra_deg": 0.0,
                    "pointing_dec_deg": 0.0,
                    "last_error_arcsec": 0.0,
                    "J00": float(self._goto.model.J_deg_per_step[0, 0]),
                    "J01": float(self._goto.model.J_deg_per_step[0, 1]),
                    "J10": float(self._goto.model.J_deg_per_step[1, 0]),
                    "J11": float(self._goto.model.J_deg_per_step[1, 1]),
                    "periodic_error_az_deg": float(
                        np.linalg.norm(self._goto.model.periodic_coeff_deg[0, :])
                    ),
                    "periodic_error_alt_deg": float(
                        np.linalg.norm(self._goto.model.periodic_coeff_deg[1, :])
                    ),
                    "periodic_model_samples": int(self._goto.model.periodic_model_samples),
                    "last_direction_az": int(self._goto.model.last_move_direction_az),
                    "last_direction_alt": int(self._goto.model.last_move_direction_alt),
                    "backlash_steps_az": int(self._goto.model.backlash_steps_az),
                    "backlash_steps_alt": int(self._goto.model.backlash_steps_alt),
                    "expected_stars_overlay_enabled": False,
                    "expected_stars_overlay_count": 0,
                    "expected_stars_overlay_source": "",
                    "expected_stars_overlay_reason": None,
                },
            }
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
            cfg = self._copy_platesolving_config(self.cfg.platesolving)
        if str(getattr(self, "_platesolving_source", "live")).lower() != "stack":
            return cfg
        # Solving the live mosaic: drizzle subdivides pixels, so the angular
        # scale per pixel shrinks by that factor. Scaling the focal length is
        # equivalent and keeps every scale check downstream consistent.
        info = self._stacking.get_stack_for_solve() if self._stacking is not None else None
        scale = float((info or {}).get("drizzle_scale", 1.0) or 1.0)
        if np.isfinite(scale) and scale > 1.0:
            cfg.focal_m = float(cfg.focal_m) * scale
        # A mosaic covers more sky than one frame; widen the cone accordingly so
        # a fitted center near the mosaic edge is not rejected as out of range.
        if info is not None:
            ch, cw = info.get("canvas", (0, 0))
            fh, fw = info.get("frame_shape", (0, 0))
            if fh > 0 and fw > 0 and ch >= fh and cw >= fw:
                grow = max(float(ch) / float(fh), float(cw) / float(fw))
                radius = getattr(cfg, "search_radius_deg", None)
                if radius is not None and np.isfinite(grow) and grow > 1.0:
                    cfg.search_radius_deg = float(radius) * min(grow, 3.0)
        return cfg

    def _get_sep_cfg_snapshot(self) -> SepConfig:
        return replace(self.cfg.sep)

    def _get_camera_cfg_snapshot(self):
        return replace(self.cfg.camera)

    def _get_mount_cfg_snapshot(self):
        return replace(self.cfg.mount)

    def _get_goto_cfg_snapshot(self):
        return replace(self.cfg.goto)

    def set_simulation_enabled(self, enabled: bool) -> None:
        enabled_b = bool(enabled)
        self.cfg.simulation.enabled = enabled_b
        self._release_simulation_if_idle()
        if self._cam_stream is not None or self._mount is not None:
            log_info(self.out_log, "Simulation: mode changed; reconnect camera/mount to apply it")
        else:
            log_info(self.out_log, f"Simulation: {'enabled' if enabled_b else 'disabled'}")

    def _simulation_enabled(self) -> bool:
        return bool(getattr(self.cfg.simulation, "enabled", False))

    def _ensure_simulation_state(self) -> SimulationState:
        if self._simulation_state is None:
            self._simulation_state = SimulationState(
                cfg=self.cfg.simulation,
                kin=self._goto.model.kin,
                out_log=self.out_log,
            )
        return self._simulation_state

    def _release_simulation_if_idle(self) -> None:
        if self._simulation_enabled():
            return
        if self._cam_stream is not None or self._mount is not None:
            return
        self._simulation_state = None
        restore_iers_after_demo(self.out_log)

    def _get_latest_frame(self) -> Optional[Frame]:
        if self._cam_stream is None:
            return None
        return self._cam_stream.latest()

    def _frame_seq(self, fr: Frame) -> Optional[int]:
        seq = fr.meta.get("seq")
        if seq is None:
            return None
        return int(seq)

    def _frame_mono_t(self, fr: Frame) -> float:
        for key in ("t_capture_mono", "t_mono"):
            try:
                value = float(fr.meta.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                return float(value)
        try:
            value = float(fr.t_capture)
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            return float(value)
        return float(_perf())

    def _frame_token(self, fr: Frame) -> float:
        seq = self._frame_seq(fr)
        return float(seq) if seq is not None else float(self._frame_mono_t(fr))

    def _frame_wall_t(self, fr: Frame) -> Optional[float]:
        for key in ("t_wall", "capture_time_unix", "unix_time"):
            try:
                value = float(fr.meta.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if np.isfinite(value) and value > 0.0:
                return float(value)

        try:
            value = float(fr.t_capture)
        except Exception:
            value = float("nan")
        # Backwards compatibility for synthetic/tests frames that still put
        # wall-clock seconds in t_capture. perf_counter values must not be used
        # as Unix timestamps.
        if np.isfinite(value) and value > 946684800.0:  # 2000-01-01 UTC
            return float(value)
        return None

    def _tracking_mode_from_output(self, mode: str) -> TrackingMode:
        try:
            return TrackingMode(str(mode))
        except ValueError:
            return TrackingMode.IDLE

    def _clip_tracking_rate_pair(self, az: float, alt: float) -> Tuple[float, float]:
        az_v = float(az) if np.isfinite(float(az)) else 0.0
        alt_v = float(alt) if np.isfinite(float(alt)) else 0.0
        rate_max = float(getattr(self.cfg.mount, "rate_max", 0.0))
        if np.isfinite(rate_max) and rate_max > 0.0:
            az_v = max(-rate_max, min(rate_max, az_v))
            alt_v = max(-rate_max, min(rate_max, alt_v))
        return float(az_v), float(alt_v)

    def _tracking_pointing_altaz(self) -> Optional[Tuple[float, float]]:
        az_alt = None
        try:
            az_alt = self._goto.model.current_az_alt_deg()
        except Exception:
            az_alt = None

        if az_alt is not None and len(az_alt) >= 2:
            az = float(az_alt[0]) % 360.0
            alt = float(np.clip(float(az_alt[1]), -90.0, 90.0))
            if np.isfinite(az) and np.isfinite(alt):
                return float(az), float(alt)

        st = self.get_state()
        if bool(st.goto.pointing_valid):
            az = float(st.goto.pointing_az_deg) % 360.0
            alt = float(np.clip(float(st.goto.pointing_alt_deg), -90.0, 90.0))
            if np.isfinite(az) and np.isfinite(alt):
                return float(az), float(alt)
        return None

    def _tracking_feedforward_rate(self, *, now_t: Optional[float] = None) -> Tuple[float, float, bool]:
        t_eval_unix = float(now_t) if (now_t is not None and np.isfinite(float(now_t))) else float(_now_s())
        if not bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)):
            self._reset_tracking_feedforward_cache()
            return 0.0, 0.0, False

        hold_s = float(getattr(self.cfg.tracking, "sidereal_ff_hold_s", 8.0))
        if (not np.isfinite(hold_s)) or hold_s < 0.0:
            hold_s = 0.0
        slew_per_s = float(getattr(self.cfg.tracking, "sidereal_ff_slew_per_s", 120.0))
        if (not np.isfinite(slew_per_s)) or slew_per_s <= 0.0:
            slew_per_s = 0.0

        pointing = self._tracking_pointing_altaz()
        if pointing is not None:
            az_deg, alt_deg = pointing

            dt_s = float(getattr(self.cfg.tracking, "sidereal_ff_dt_s", 1.0))
            if (not np.isfinite(dt_s)) or dt_s <= 1e-3:
                dt_s = 1.0
            cond_max = float(getattr(self.cfg.tracking, "sidereal_ff_cond_max", 5_000.0))
            if (not np.isfinite(cond_max)) or cond_max <= 1.0:
                cond_max = 5_000.0
            gain = float(getattr(self.cfg.tracking, "sidereal_ff_gain", 1.0))
            if not np.isfinite(gain):
                gain = 1.0

            t_eval = Time(float(t_eval_unix), format="unix", scale="utc")
            rate_ff = self._goto.model.sidereal_step_rate_deg_s(
                az_deg=float(az_deg),
                alt_deg=float(alt_deg),
                observer=self._platesolving_observer,
                obstime=t_eval,
                dt_s=float(dt_s),
                cond_max=float(cond_max),
            )
            if rate_ff is not None:
                az_ff = float(rate_ff[0]) * float(gain)
                alt_ff = float(rate_ff[1]) * float(gain)
                az_ff, alt_ff = self._clip_tracking_rate_pair(az_ff, alt_ff)

                if self._tracking_ff_last_valid_t is not None and slew_per_s > 0.0:
                    dt_ff = max(0.0, float(t_eval_unix - float(self._tracking_ff_last_valid_t)))
                    max_delta = float(slew_per_s) * float(dt_ff)
                    if max_delta > 0.0:
                        az_ff = float(self._tracking_ff_hold_az) + max(-max_delta, min(max_delta, float(az_ff - self._tracking_ff_hold_az)))
                        alt_ff = float(self._tracking_ff_hold_alt) + max(-max_delta, min(max_delta, float(alt_ff - self._tracking_ff_hold_alt)))

                self._tracking_ff_hold_az = float(az_ff)
                self._tracking_ff_hold_alt = float(alt_ff)
                self._tracking_ff_last_valid_t = float(t_eval_unix)
                return float(az_ff), float(alt_ff), True

        if self._tracking_ff_last_valid_t is not None:
            dt_hold = float(t_eval_unix - float(self._tracking_ff_last_valid_t))
            if dt_hold <= float(hold_s):
                az_ff, alt_ff = self._clip_tracking_rate_pair(float(self._tracking_ff_hold_az), float(self._tracking_ff_hold_alt))
                return float(az_ff), float(alt_ff), True

        self._reset_tracking_feedforward_cache()
        return 0.0, 0.0, False

    def _cached_tracking_feedforward_rate(
        self,
        *,
        now_t: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """Refresh the astronomical model slowly; the fast loop reuses it."""
        t_eval = float(now_t) if now_t is not None else float(_now_s())
        if not bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)):
            self._reset_tracking_feedforward_cache()
            return 0.0, 0.0, False

        update_hz = _finite_float(
            getattr(self.cfg.tracking, "sidereal_ff_update_hz", 2.0),
            2.0,
        )
        update_hz = max(0.1, min(20.0, float(update_hz)))
        due = (
            self._tracking_ff_last_compute_t is None
            or t_eval < float(self._tracking_ff_last_compute_t)
            or (t_eval - float(self._tracking_ff_last_compute_t)) >= (1.0 / update_hz)
        )
        if due:
            self._tracking_ff_cached = self._tracking_feedforward_rate(now_t=t_eval)
            self._tracking_ff_last_compute_t = float(t_eval)
        return self._tracking_ff_cached

    def _tracking_seed_calibration_from_pointing(self) -> bool:
        try:
            if self._simulation_state is not None:
                az_deg, alt_deg = self._simulation_state.snapshot_altaz()
            else:
                pointing = self._tracking_pointing_altaz()
                if pointing is None:
                    return False
                az_deg, alt_deg = pointing

            obstime = Time.now()
            true_altaz_observer = replace(
                self._platesolving_observer,
                refraction_enable=False,
            )
            center = parse_target_to_icrs(
                {"az_deg": float(az_deg), "alt_deg": float(alt_deg)},
                observer=true_altaz_observer,
                obstime=obstime,
            ).icrs

            scale = 206265.0 * float(self.cfg.platesolving.pixel_size_m) / float(self.cfg.platesolving.focal_m)
            if (not np.isfinite(scale)) or scale <= 1e-9:
                return False

            theta = expected_field_rotation_deg(
                float(center.ra.deg),
                float(center.dec.deg),
                observer=self._platesolving_observer,
                obstime=obstime,
                roll_offset_deg=float(self.cfg.camera.roll_deg),
                az_step_deg=float(getattr(self.cfg.platesolving, "rotation_prior_az_step_deg", 0.05)),
            )
            if theta is None or not np.isfinite(float(theta)):
                theta = 0.0
            th = np.deg2rad(float(theta))
            R = np.array(
                [[float(np.cos(th)), -float(np.sin(th))], [float(np.sin(th)), float(np.cos(th))]],
                dtype=np.float64,
            )

            J = np.asarray(self._goto.model.J_deg_per_step, dtype=np.float64).reshape(2, 2)
            A = np.zeros((2, 2), dtype=np.float64)
            for col in range(2):
                az2 = float((float(az_deg) + float(J[0, col])) % 360.0)
                alt2 = float(np.clip(float(alt_deg) + float(J[1, col]), -89.0, 89.0))
                shifted = parse_target_to_icrs(
                    {"az_deg": az2, "alt_deg": alt2},
                    observer=true_altaz_observer,
                    obstime=obstime,
                ).icrs
                # Observed star displacement when the telescope center moves by
                # one positive step on this axis.
                off = center.transform_to(shifted.skyoffset_frame())
                q_arcsec = np.array(
                    [
                        float(off.lon.to_value(u.arcsec)),
                        float(off.lat.to_value(u.arcsec)),
                    ],
                    dtype=np.float64,
                )
                A[:, col] = (q_arcsec / float(scale)) @ R

            det = float(np.linalg.det(A))
            if (not np.isfinite(det)) or abs(det) < 1e-6:
                return False
            theta_cal = np.column_stack([A, np.zeros(2, dtype=np.float64)])
            with self._tracking_state_lock:
                auto_reset(self._tracking_state, src="geometry", theta=theta_cal)
            log_info(
                self.out_log,
                (
                    "Tracking: geometry calibration seeded "
                    f"A=[[{A[0,0]:+.4f},{A[0,1]:+.4f}],"
                    f"[{A[1,0]:+.4f},{A[1,1]:+.4f}]]"
                ),
            )
            return True
        except Exception as exc:
            log_error(self.out_log, "Tracking: failed to seed geometry calibration", exc, throttle_s=5.0, throttle_key="tracking_seed_calib")
            return False

    def _get_fps_capture(self) -> float:
        if self._cam_stream is None:
            return 0.0
        st = self._cam_stream.stats()
        return float(st.get("fps_capture", 0.0))

    def _get_live_frame_for_platesolving(self) -> Optional[np.ndarray]:
        if str(getattr(self, "_platesolving_source", "live")).lower() == "stack":
            stacked = self._get_stack_frame_for_platesolving()
            if stacked is not None:
                return stacked
            log_info(
                self.out_log,
                "Platesolving: stack source empty, falling back to live frame",
            )
        if self._cam_stream is None:
            return None
        fr = self._cam_stream.latest()
        if fr is None:
            return None
        try:
            return ensure_raw16_bayer(fr.raw).copy()
        except Exception as exc:
            log_error(
                self.out_log,
                "Platesolving: live frame copy failed",
                exc,
                throttle_s=2.0,
                throttle_key="platesolving_frame_copy",
            )
            return None

    def _get_stack_frame_for_platesolving(self) -> Optional[np.ndarray]:
        """Live mosaic as solver input.

        Stacking short exposures is preferable to a long one under heavy light
        pollution: stars stay point-like rather than trailing, faint stars
        emerge from the accumulated signal, and the mosaic spans more sky than
        a single frame. The detector then sees many more usable sources, which
        is exactly what the triplet search needs.
        """
        if self._stacking is None:
            return None
        try:
            info = self._stacking.get_stack_for_solve()
        except Exception as exc:
            log_error(
                self.out_log,
                "Platesolving: stack snapshot failed",
                exc,
                throttle_s=5.0,
                throttle_key="platesolving_stack_snapshot",
            )
            return None
        if not info:
            return None
        img = info.get("image")
        if img is None or int(info.get("frames", 0)) <= 0:
            return None
        self._platesolving_stack_info = info
        log_info(
            self.out_log,
            f"Platesolving: using live stack ({int(info['frames'])} frames, "
            f"canvas {info['canvas'][1]}x{info['canvas'][0]})",
        )
        return np.ascontiguousarray(img)

    def _publish_platesolving_state(self, patch: Dict[str, Dict[str, Any]]) -> None:
        with self._operation_lock:
            for name in ("platesolving", "goto"):
                section = patch.get(name, {})
                if not isinstance(section, dict) or "busy" not in section:
                    continue
                if bool(section.get("busy")):
                    if str(getattr(section.get("status"), "value", section.get("status"))) == "RUNNING":
                        self._operation_started[name] += 1
                else:
                    self._operation_finished[name] = self._operation_started[name]
        result = patch.pop("platesolving_result", None)
        # GoToWorker consumes its own plate solutions (continuity check,
        # manual sample/sync and calibration). Re-enqueuing the same result in
        # the runner used to store every AutoCal solution twice and biased the
        # pointing-model fit. Explicit PlatesolvingWorker results remain
        # automatically validated here.
        result_handled = bool(patch.pop("platesolving_result_handled", False))
        self._update_state(patch)
        if result is not None:
            # Never leave an older successful solve available for Sync after
            # the newest attempt was rejected (including motion-continuity
            # rejection). A stale solution is more dangerous than no solution.
            self._last_platesolving_result = (
                result if bool(getattr(result, "success", False)) else None
            )
            if self._last_platesolving_result is not None and not result_handled:
                # The operator decides when to solve; the program decides if
                # the resulting sample is trustworthy enough for the model.
                self.enqueue(goto_validate_sample(result))
            else:
                result_status = str(getattr(result, "status", "PLATESOLVING_FAILED"))
                self._update_state(
                    {
                        "goto": {
                            "sample_last_ok": False,
                            "sample_last_reason": result_status,
                        }
                    }
                )

    def _goto_pointing_snapshot(self) -> Optional[Dict[str, float]]:
        az_alt = self._goto.model.current_az_alt_deg()
        if az_alt is None:
            return None

        az = float(az_alt[0]) % 360.0
        alt = float(np.clip(float(az_alt[1]), -90.0, 90.0))
        if not np.isfinite(az) or not np.isfinite(alt):
            return None

        coord_icrs = parse_target_to_icrs(
            {"az_deg": az, "alt_deg": alt},
            observer=self._platesolving_observer,
            obstime=Time.now(),
        ).icrs
        ra = float(coord_icrs.ra.deg) % 360.0
        dec = float(coord_icrs.dec.deg)
        if not np.isfinite(ra) or not np.isfinite(dec):
            return None

        return {
            "az_deg": az,
            "alt_deg": alt,
            "ra_deg": ra,
            "dec_deg": dec,
        }

    def _update_goto_pointing_state(self) -> None:
        try:
            p = self._goto_pointing_snapshot()
        except Exception as exc:
            log_error(
                self.out_log,
                "GoTo: live pointing update failed",
                exc,
                throttle_s=2.0,
                throttle_key="goto_pointing_update",
            )
            p = None

        if p is None:
            self._update_state({"goto": {"pointing_valid": False}})
            return

        self._update_state(
            {
                "goto": {
                    "pointing_valid": True,
                    "pointing_az_deg": float(p["az_deg"]),
                    "pointing_alt_deg": float(p["alt_deg"]),
                    "pointing_ra_deg": float(p["ra_deg"]),
                    "pointing_dec_deg": float(p["dec_deg"]),
                }
            }
        )

    def _maybe_update_goto_pointing_state(self, *, now: Optional[float] = None) -> bool:
        t_now = float(_perf() if now is None else now)
        update_hz = max(
            0.1,
            min(20.0, _finite_float(getattr(self.cfg, "pointing_hz", 2.0), 2.0)),
        )
        if (t_now - float(self._t_last_pointing)) < (1.0 / update_hz):
            return False
        self._t_last_pointing = t_now
        self._update_goto_pointing_state()
        return True

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
        with self._gaia_download_lock:
            gaia_thr = self._gaia_download_thread
        if gaia_thr is not None and gaia_thr.is_alive():
            gaia_thr.join(timeout=2.0)
            if gaia_thr.is_alive():
                log_error(
                    self.out_log,
                    "Gaia download: thread did not stop within timeout",
                    RuntimeError("gaia download still running"),
                )

        with self._gaia_coverage_lock:
            coverage_thr = self._gaia_coverage_thread
        if coverage_thr is not None and coverage_thr.is_alive():
            coverage_thr.join(timeout=2.0)
            if coverage_thr.is_alive():
                log_error(
                    self.out_log,
                    "Gaia coverage: worker did not stop within timeout",
                    RuntimeError("Gaia coverage still running"),
                )

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

        self._preview_worker.stop()
        self._preview_worker.join(timeout=2.0)
        self._tracking_worker.stop()
        self._tracking_worker.join(timeout=2.0)
        self._save_worker.stop(timeout=5.0)

        with self._raw_record_lock:
            raw_thr = self._raw_record_thread
        self._raw_record_stop.set()
        if raw_thr is not None and raw_thr.is_alive():
            raw_thr.join(timeout=2.0)
            if raw_thr.is_alive():
                log_error(self.out_log, "Raw record: thread did not stop within timeout", RuntimeError("raw recording still running"))

        self._shutdown_camera()
        self._shutdown_mount()
        self._simulation_state = None
        restore_iers_after_demo(self.out_log)
        try:
            self._stacking.stop()
        except Exception as exc:
            log_error(self.out_log, "Stacking: stop failed", exc)
        try:
            self._stacking.shutdown()
        except Exception as exc:
            log_error(self.out_log, "Stacking: shutdown failed", exc)

        log_info(self.out_log, "Runner: stopped")

    # -------------------------
    # UI entrypoints
    # -------------------------
    def enqueue(self, action: Action) -> None:
        self._actions.put(action)

    def get_state(self) -> AppState:
        with self._state_lock:
            return self._state.snapshot()

    def get_latest_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._latest_preview_jpeg

    def get_gaia_coverage(self) -> Dict[str, object]:
        cfg = self._get_platesolving_cfg_snapshot()
        center_icrs: Optional[SkyCoord] = None
        radius_deg: Optional[float] = None
        source: Optional[str] = None
        try:
            center_icrs, source = self._current_field_center_icrs()
            radius_deg = self._current_field_download_radius_deg()
        except Exception:
            pass

        coverage = gaia_healpix_coverage(
            cfg=cfg,
            center_icrs=center_icrs,
            radius_deg=radius_deg,
        )
        obstime = Time.now()
        tile_icrs = SkyCoord(
            ra=np.asarray(coverage["tile_ra_deg"], dtype=np.float64) * u.deg,
            dec=np.asarray(coverage["tile_dec_deg"], dtype=np.float64) * u.deg,
            frame="icrs",
        )
        with (
            iers.conf.set_temp("auto_download", False),
            iers.conf.set_temp("auto_max_age", None),
        ):
            altaz_frame = AltAz(
                obstime=obstime,
                location=self._platesolving_observer.location(),
            )
            tile_altaz = tile_icrs.transform_to(altaz_frame)
            coverage["tile_az_deg"] = np.asarray(tile_altaz.az.deg, dtype=np.float64)
            coverage["tile_alt_deg"] = np.asarray(tile_altaz.alt.deg, dtype=np.float64)
            if center_icrs is not None:
                center_altaz = center_icrs.transform_to(altaz_frame)
                coverage["center_az_deg"] = float(center_altaz.az.deg) % 360.0
                coverage["center_alt_deg"] = float(center_altaz.alt.deg)
            else:
                coverage["center_az_deg"] = None
                coverage["center_alt_deg"] = None

        coverage["projection_time_utc"] = str(obstime.utc.isot)
        coverage["observer_lat_deg"] = float(self._platesolving_observer.lat_deg)
        coverage["observer_lon_deg"] = float(self._platesolving_observer.lon_deg)
        coverage["observer_height_m"] = float(self._platesolving_observer.height_m)
        coverage["field_source"] = source
        return coverage

    def request_gaia_coverage_refresh(self) -> bool:
        """Schedule the expensive coverage projection outside the Qt thread."""
        with self._gaia_coverage_lock:
            if self._gaia_coverage_pending:
                return False
            self._gaia_coverage_pending = True

        def _worker() -> None:
            coverage: Optional[Dict[str, object]] = None
            error: Optional[str] = None
            try:
                coverage = self.get_gaia_coverage()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log_error(self.out_log, "Gaia coverage: refresh failed", exc)
            finally:
                with self._gaia_coverage_lock:
                    if coverage is not None:
                        self._gaia_coverage_cache = coverage
                        self._gaia_coverage_version += 1
                    self._gaia_coverage_error = error
                    self._gaia_coverage_pending = False
                    self._gaia_coverage_thread = None

        thread = threading.Thread(target=_worker, name="GaiaCoverage", daemon=True)
        with self._gaia_coverage_lock:
            self._gaia_coverage_thread = thread
        thread.start()
        return True

    def get_gaia_coverage_snapshot(self) -> Dict[str, object]:
        with self._gaia_coverage_lock:
            return {
                "version": int(self._gaia_coverage_version),
                "pending": bool(self._gaia_coverage_pending),
                "error": self._gaia_coverage_error,
                "coverage": self._gaia_coverage_cache,
            }

    def get_operation_counters(self) -> Dict[str, Dict[str, int]]:
        """Monotonic operation counters used by terminal automation.

        Unlike polling ``busy == false``, these counters cannot report success
        before a queued worker request has actually started.
        """
        with self._operation_lock:
            return {
                name: {
                    "started": int(self._operation_started[name]),
                    "finished": int(self._operation_finished[name]),
                }
                for name in self._operation_started
            }

    def _record_loop_performance(self, sections: Dict[str, float]) -> None:
        with self._performance_lock:
            for name, value in sections.items():
                samples = self._performance_samples.get(name)
                if samples is not None:
                    samples.append(float(value))

        total_ms = float(sections.get("total_ms", 0.0))
        now = _perf()
        if total_ms > 50.0 and (now - self._last_stall_log_t) >= 2.0:
            self._last_stall_log_t = now
            breakdown = ", ".join(
                f"{name.removesuffix('_ms')}={float(value):.1f}ms"
                for name, value in sections.items()
                if name != "total_ms"
            )
            log_info(
                self.out_log,
                f"Runner: slow loop total={total_ms:.1f}ms ({breakdown})",
            )

    def get_performance_metrics(self) -> Dict[str, object]:
        with self._performance_lock:
            copied = {
                name: np.asarray(tuple(values), dtype=np.float64)
                for name, values in self._performance_samples.items()
            }

        sections: Dict[str, Dict[str, float]] = {}
        sample_count = 0
        for name, values in copied.items():
            if values.size == 0:
                continue
            sample_count = max(sample_count, int(values.size))
            p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
            sections[name] = {
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
                "max": float(np.max(values)),
            }
        return {
            "sample_count": int(sample_count),
            "window_s": 30.0,
            "sections": sections,
        }

    def _start_operation(self, name: str) -> None:
        with self._operation_lock:
            self._operation_started[name] += 1

    def _finish_operation(self, name: str) -> None:
        with self._operation_lock:
            self._operation_finished[name] = self._operation_started[name]

    def request_camera_connect(self, camera_index: int) -> None:
        self.enqueue(camera_connect(camera_index))

    def request_camera_disconnect(self) -> None:
        self.enqueue(camera_disconnect())

    def request_camera_param(self, name: str, value: Any) -> None:
        self.enqueue(camera_set_param(name, value))

    def request_camera_params(self, params: Dict[str, Any]) -> None:
        self.enqueue(camera_set_params(params))

    def request_camera_record_raw(
        self,
        *,
        duration_s: Optional[float] = 20.0,
        out_dir: str = "raw_output",
        basename: Optional[str] = None,
    ) -> None:
        self.enqueue(camera_record_raw(duration_s=duration_s, out_dir=out_dir, basename=basename))

    def request_camera_stop_record_raw(self) -> None:
        self.enqueue(camera_stop_record_raw())

    def camera_recording_active(self) -> bool:
        with self._raw_record_lock:
            return bool(self._raw_record_active)

    def request_mount_connect(self, port: str, baudrate: int) -> None:
        self.enqueue(mount_connect(port, baudrate))

    def request_mount_disconnect(self) -> None:
        self.enqueue(mount_disconnect())

    def request_mount_set_microsteps(self, az_div: int, alt_div: int) -> None:
        self.enqueue(mount_set_microsteps(az_div=az_div, alt_div=alt_div))

    def request_mount_move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> None:
        self.enqueue(
            mount_move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=delay_us,
                profile=profile,
            )
        )

    def request_mount_stop(self) -> None:
        self.enqueue(mount_stop())

    def cancel_platesolving(self) -> None:
        """Cooperatively cancel the currently running explicit plate solve."""
        self._platesolving_worker.cancel_current()

    def request_mount_sync(self) -> None:
        self.enqueue(mount_sync())

    def request_mount_goto(self, target: Any, **kwargs: Any) -> None:
        self.enqueue(mount_goto(target, **kwargs))

    def request_goto_calibrate(self, params: Dict[str, Any]) -> None:
        self.enqueue(goto_calibrate(params))

    def request_goto_autocalibrate(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_autocalibrate(params))

    def request_goto_estimate_roll(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_estimate_roll(params))

    def request_goto_fit_model(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_fit_model(params))

    def request_goto_list_samples(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_list_samples(params))

    def request_goto_prune_outliers(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_prune_outliers(params))

    def request_goto_reset(self) -> None:
        self.enqueue(goto_reset())

    def request_goto_restore_last_log(self) -> None:
        self.enqueue(goto_restore_last_log())

    def request_goto_cancel(self) -> None:
        self.enqueue(goto_cancel())

    def request_tracking_start(self) -> None:
        self.enqueue(tracking_start())

    def request_tracking_stop(self) -> None:
        self.enqueue(tracking_stop())

    def request_tracking_params(self, **kwargs: Any) -> None:
        self.enqueue(tracking_set_params_action(**kwargs))

    def request_stacking_start(self) -> None:
        self.enqueue(stacking_start())

    def request_stacking_stop(self) -> None:
        self.enqueue(stacking_stop())

    def request_stacking_reset(self) -> None:
        self.enqueue(stacking_reset())

    def request_stacking_params(self, **kwargs: Any) -> None:
        self.enqueue(stacking_set_params(**kwargs))

    def request_stacking_save(self, **kwargs: Any) -> None:
        self.enqueue(stacking_save(**kwargs))

    def request_platesolving_run(self, target: Any) -> None:
        self.enqueue(platesolving_run(target=target))

    def request_platesolving_download_current_field(self, radius_deg: Optional[float] = None) -> None:
        self.enqueue(platesolving_download_current_field(radius_deg=radius_deg))

    def request_platesolving_params(self, **kwargs: Any) -> None:
        self.enqueue(platesolving_set_params(**kwargs))

    def request_live_sep_params(self, **kwargs: Any) -> None:
        self.enqueue(live_sep_set_params(**kwargs))

    def request_expected_stars_params(self, **kwargs: Any) -> None:
        self.enqueue(expected_stars_set_params(**kwargs))

    # -------------------------
    # Internal helpers
    # -------------------------
    def _update_state(self, patch: Dict[str, Dict[str, Any]]) -> None:
        with self._state_lock:
            self._state.update(patch)

    def _get_tracking_enabled(self) -> bool:
        with self._state_lock:
            return bool(self._state.tracking.enabled)

    def _invalidate_tracking_pipeline(self) -> None:
        with self._tracking_result_lock:
            self._tracking_generation += 1
            self._tracking_last_frame_token = None
            self._tracking_last_output = None
            self._tracking_worker_error = None

    def _submit_tracking_frame(
        self,
        *,
        raw16: np.ndarray,
        frame_token: float,
        frame_t: float,
        tracking_enabled: bool,
    ) -> None:
        with self._tracking_result_lock:
            generation = int(self._tracking_generation)
            self._tracking_last_frame_token = float(frame_token)
        self._tracking_worker.request(
            generation=generation,
            raw16=raw16,
            frame_t=float(frame_t),
            tracking_enabled=bool(tracking_enabled),
            applied_rate_az=float(self._tracking_last_cmd_az),
            applied_rate_alt=float(self._tracking_last_cmd_alt),
        )

    def _process_tracking_request(self, request: Dict[str, Any]) -> None:
        generation = int(request.get("generation", -1))
        try:
            with self._tracking_state_lock:
                out = tracking_step(
                    self._tracking_state,
                    ensure_raw16_bayer(request["raw16"]),
                    now_t=float(request["frame_t"]),
                    tracking_enabled=bool(request.get("tracking_enabled", True)),
                    applied_rate_az=float(request.get("applied_rate_az", 0.0)),
                    applied_rate_alt=float(request.get("applied_rate_alt", 0.0)),
                )
        except Exception as exc:
            with self._tracking_result_lock:
                if generation == int(self._tracking_generation):
                    self._tracking_worker_error = exc
            return

        with self._tracking_result_lock:
            if generation != int(self._tracking_generation):
                return
            self._tracking_last_output = out
            self._tracking_worker_error = None

    def _tracking_result_snapshot(self) -> Tuple[Optional[Any], Optional[Exception]]:
        with self._tracking_result_lock:
            return self._tracking_last_output, self._tracking_worker_error

    def _publish_tracking_output(
        self,
        out: TrackingOutput,
        *,
        tracking_on: bool,
        ff_ready: bool,
        rate_cmd_az: float,
        rate_cmd_alt: float,
        rate_fb_az: float,
        rate_fb_alt: float,
        rate_ff_az: float,
        rate_ff_alt: float,
    ) -> None:
        tracking_mode = self._tracking_mode_from_output(out.mode)
        self._update_state(
            {
                "tracking": {
                    "enabled": bool(tracking_on),
                    "status": TrackingStatus.RUNNING,
                    "mode": tracking_mode,
                    "resp": float(out.resp),
                    "dx": float(out.dx),
                    "dy": float(out.dy),
                    "vx": float(out.vx),
                    "vy": float(out.vy),
                    "abs_resp": float(out.abs_resp),
                    "ff_enabled": bool(
                        getattr(self.cfg.tracking, "sidereal_ff_enabled", True)
                    ),
                    "ff_ready": bool(ff_ready),
                    "rate_az": float(rate_cmd_az),
                    "rate_alt": float(rate_cmd_alt),
                    "rate_fb_az": float(rate_fb_az),
                    "rate_fb_alt": float(rate_fb_alt),
                    "rate_ff_az": float(rate_ff_az),
                    "rate_ff_alt": float(rate_ff_alt),
                    "calib_src": str(out.calib_src),
                    "calib_det": float(out.detA),
                    "n_det": int(out.n_det),
                    "measurement_valid": bool(out.ok),
                    "measurement_reason": str(out.measurement_reason),
                    "measurement_source": str(out.measurement_source),
                    "error_x_px": float(out.x_hat),
                    "error_y_px": float(out.y_hat),
                    "error_px": float(np.hypot(out.x_hat, out.y_hat)),
                    "lock_conf": float(out.lock_conf),
                    "fail_count": int(out.fail_count),
                    "last_error": (
                        f"measurement invalid: {out.measurement_reason}"
                        if (not out.ok and int(out.fail_count) >= 2)
                        else (
                            "feedback calibration unavailable"
                            if (out.ok and str(out.calib_src) == "none")
                            else None
                        )
                    ),
                }
            }
        )

    def _publish_tracking_off(self) -> None:
        self._update_state(
            {
                "tracking": {
                    "enabled": False,
                    "status": TrackingStatus.OFF,
                    "mode": TrackingMode.IDLE,
                    "ff_enabled": bool(
                        getattr(self.cfg.tracking, "sidereal_ff_enabled", True)
                    ),
                    "ff_ready": False,
                    "rate_az": 0.0,
                    "rate_alt": 0.0,
                    "rate_fb_az": 0.0,
                    "rate_fb_alt": 0.0,
                    "rate_ff_az": 0.0,
                    "rate_ff_alt": 0.0,
                    "n_det": 0,
                    "measurement_valid": False,
                    "measurement_reason": "off",
                    "measurement_source": "none",
                    "error_x_px": 0.0,
                    "error_y_px": 0.0,
                    "error_px": 0.0,
                    "lock_conf": 0.0,
                    "fail_count": 0,
                    "last_error": None,
                }
            }
        )

    def _tracking_keyframe_reset(self) -> None:
        try:
            # Reset both the absolute target and the incremental reference.
            # Keeping the old incremental frame after a GoTo/manual movement
            # made the first new frame look like a tracking jump and could send
            # the controller toward the previous field.
            with self._tracking_state_lock:
                reset_tracker(self._tracking_state, now_t=float(_perf()), mode="STABILIZE")
            self._invalidate_tracking_pipeline()
        except Exception as exc:
            log_error(self.out_log, "Tracking: failed to reset keyframe", exc)

    def _reset_tracking_feedforward_cache(self) -> None:
        self._tracking_ff_hold_az = 0.0
        self._tracking_ff_hold_alt = 0.0
        self._tracking_ff_last_valid_t = None
        self._tracking_ff_last_compute_t = None
        self._tracking_ff_cached = (0.0, 0.0, False)

    def _estimate_manual_move_runtime_s(
        self,
        *,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> float:
        return estimate_firmware_move_duration_s(
            int(steps), int(delay_us), profile=profile
        )

    def _mark_manual_move_active(
        self,
        *,
        axis: Axis,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> None:
        est_s = self._estimate_manual_move_runtime_s(
            steps=steps,
            delay_us=delay_us,
            profile=profile,
        )
        until = _perf() + est_s + 0.05
        key = axis.value
        prev = float(self._manual_move_active_until_s.get(key, 0.0))
        self._manual_move_active_until_s[key] = max(prev, until)

    def _clear_manual_move_activity(self) -> None:
        self._manual_move_active_until_s[Axis.AZ.value] = 0.0
        self._manual_move_active_until_s[Axis.ALT.value] = 0.0

    def _reset_rate_emulation(self) -> None:
        with self._rate_emul_lock:
            self._rate_emul_last_t = _perf()
            self._rate_emul_acc_az = 0.0
            self._rate_emul_acc_alt = 0.0
            self._rate_emul_active = False

    def _is_manual_move_active(self) -> bool:
        if self._mount_move_worker.is_busy():
            return True
        now = _perf()
        return now < max(
            float(self._manual_move_active_until_s.get(Axis.AZ.value, 0.0)),
            float(self._manual_move_active_until_s.get(Axis.ALT.value, 0.0)),
        )

    def _rate_to_delay_us(self, rate_steps_s: float, *, axis: Axis) -> int:
        rate_abs = abs(float(rate_steps_s))
        if rate_abs <= 1e-9:
            base = int(self.cfg.mount.slew_delay_us_az if axis == Axis.AZ else self.cfg.mount.slew_delay_us_alt)
            return max(1, base)
        # Firmware cadence ~= 1e6 / (delay_us + pulse_us), pulse_us ~= 3.
        delay = int(round((1.0e6 / rate_abs) - 3.0))
        return max(1, min(delay, 50000))

    def _send_move_steps_direct(self, *, axis: Axis, signed_steps: int, delay_us: int) -> None:
        if self._mount is None or signed_steps == 0:
            return
        direction = +1 if int(signed_steps) >= 0 else -1
        steps = abs(int(signed_steps))
        try:
            self._mount.move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=int(delay_us),
                profile="direct",
                blocking=False,
                stop_before_move=False,
            )
        except TypeError as exc:
            # Keep rate emulation compatible with lightweight/legacy mount
            # adapters that predate the optional movement profile.
            if "profile" not in str(exc):
                raise
            self._mount.move_steps(
                axis=axis,
                direction=direction,
                steps=steps,
                delay_us=int(delay_us),
                blocking=False,
                stop_before_move=False,
            )

    def _mount_rate_safe(self, az: float, alt: float) -> tuple[int, int]:
        if self._mount is None:
            return 0, 0
        with self._rate_emul_lock:
            now = _perf()
            if self._rate_emul_last_t is None:
                self._rate_emul_last_t = now
            dt = float(now - float(self._rate_emul_last_t))
            if not np.isfinite(dt) or dt < 0.0:
                dt = 0.0
            self._rate_emul_last_t = now

            az_cmd = float(az)
            alt_cmd = float(alt)
            is_stop = (abs(az_cmd) <= 1e-9) and (abs(alt_cmd) <= 1e-9)

            if is_stop:
                self._rate_emul_acc_az = 0.0
                self._rate_emul_acc_alt = 0.0
                was_active = bool(self._rate_emul_active)
                self._rate_emul_active = False
                if not was_active:
                    return 0, 0
                try:
                    self._mount.stop()
                except Exception as exc:
                    self._update_state(
                        {
                            "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "MOVE stop failed"},
                            "tracking": {
                                "enabled": False,
                                "status": TrackingStatus.OFF,
                                "mode": TrackingMode.IDLE,
                                "last_error": "mount STOP failed",
                            },
                        }
                    )
                    log_error(self.out_log, "Mount: STOP failed (rate emulation)", exc, throttle_s=2.0, throttle_key="mount_stop_rate_emul")
                return 0, 0

            if self._is_manual_move_active():
                return 0, 0

            if dt > 0.0:
                self._rate_emul_acc_az += az_cmd * dt
                self._rate_emul_acc_alt += alt_cmd * dt

            step_az = int(np.trunc(self._rate_emul_acc_az))
            step_alt = int(np.trunc(self._rate_emul_acc_alt))
            step_az = int(max(-400, min(400, step_az)))
            step_alt = int(max(-400, min(400, step_alt)))

            self._rate_emul_acc_az -= float(step_az)
            self._rate_emul_acc_alt -= float(step_alt)
            self._rate_emul_active = True

            if step_az == 0 and step_alt == 0:
                return 0, 0

            delay_az = self._rate_to_delay_us(abs(az_cmd), axis=Axis.AZ)
            delay_alt = self._rate_to_delay_us(abs(alt_cmd), axis=Axis.ALT)

        try:
            if step_az != 0:
                self._send_move_steps_direct(axis=Axis.AZ, signed_steps=step_az, delay_us=delay_az)
            if step_alt != 0:
                self._send_move_steps_direct(axis=Axis.ALT, signed_steps=step_alt, delay_us=delay_alt)
            return step_az, step_alt
        except Exception as exc:
            self._update_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "MOVE command failed"},
                    "tracking": {
                        "enabled": False,
                        "status": TrackingStatus.OFF,
                        "mode": TrackingMode.IDLE,
                        "last_error": "mount MOVE failed",
                    },
                }
            )
            log_error(self.out_log, "Mount: MOVE rate-emulation failed", exc, throttle_s=2.0, throttle_key="mount_move_rate_emul")
            return 0, 0

    def _ensure_transmission_collector(self) -> Optional[TransmissionErrorCollector]:
        if self._transmission_collector is not None:
            return self._transmission_collector
        try:
            kin = self._goto.model.kin
            self._transmission_collector = TransmissionErrorCollector(
                period_steps=(
                    float(kin.transmission_error_period_steps(Axis.AZ)),
                    float(kin.transmission_error_period_steps(Axis.ALT)),
                ),
                deg_per_step=(
                    float(kin.deg_per_step(Axis.AZ)),
                    float(kin.deg_per_step(Axis.ALT)),
                ),
            )
        except Exception as exc:
            log_error(
                self.out_log,
                "Transmission: collector init failed",
                exc,
                throttle_s=30.0,
                throttle_key="transmission_collector_init",
            )
            return None
        return self._transmission_collector

    def _observe_transmission_error(self, out: Any) -> None:
        """Feed the cycloidal error learner from the tracking loop.

        The tracker's RLS already measures the real px/µstep response many
        times per second. Its variation with lobe phase *is* the transmission
        error, so recording it here gives thousands of free samples instead of
        the eight hard-won plate solves the manual fit needs.
        """
        if not bool(getattr(out, "measurement_valid", False)):
            return
        if float(getattr(out, "lock_conf", 0.0)) < 0.7:
            return
        A = getattr(out, "A", None)
        if A is None:
            A = getattr(getattr(self._tracking_state, "auto", None), "A", None)
        if A is None:
            return
        collector = self._ensure_transmission_collector()
        if collector is None:
            return
        try:
            norms = gain_from_tracking_matrix(A)
            if norms is None:
                return
            if self._transmission_gain_ref is None:
                self._transmission_gain_ref = np.asarray(norms, dtype=np.float64)
                return
            ref = self._transmission_gain_ref
            # Reference tracks slowly so a genuine drift in plate scale (focus,
            # altitude) does not masquerade as a transmission error.
            self._transmission_gain_ref = 0.999 * ref + 0.001 * np.asarray(norms)
            collector.observe(
                steps=self._goto.model.steps_est,
                gain=np.asarray(norms, dtype=np.float64) / ref,
            )
        except Exception as exc:
            log_error(
                self.out_log,
                "Transmission: observe failed",
                exc,
                throttle_s=30.0,
                throttle_key="transmission_observe",
            )

    def get_transmission_error_report(self) -> Dict[str, Any]:
        """Coverage and current estimate of the learned transmission error."""
        collector = self._transmission_collector
        if collector is None:
            return {"samples": 0, "coverage": {"az": 0.0, "alt": 0.0, "min": 0.0}, "fitted": False}
        report: Dict[str, Any] = {
            "samples": int(collector.samples),
            "rejected": int(collector.rejected),
            "coverage": collector.coverage(),
            "fitted": False,
        }
        result = collector.fit()
        if result is not None:
            coeff, detail = result
            report["fitted"] = True
            report["coeff_deg"] = coeff.tolist()
            report.update({k: float(v) for k, v in detail.items()})
        return report

    def apply_learned_transmission_error(self) -> bool:
        """Push the learned periodic error into the pointing model."""
        collector = self._transmission_collector
        if collector is None:
            return False
        result = collector.fit()
        if result is None:
            log_info(self.out_log, "Transmission: cobertura de fase insuficiente todavía")
            return False
        coeff, detail = result
        try:
            self._goto.model.periodic_coeff_deg = self._goto.model.safe_periodic_coeff_for_prediction(coeff)
            self._goto.model.periodic_model_samples = int(collector.samples)
        except Exception as exc:
            log_error(self.out_log, "Transmission: apply failed", exc)
            return False
        log_info(
            self.out_log,
            "Transmission: modelo periódico actualizado desde tracking "
            f"({int(collector.samples)} muestras, "
            f"cobertura az={detail.get('az_coverage', 0.0):.2f} alt={detail.get('alt_coverage', 0.0):.2f})",
        )
        return True

    def _tracking_rate_safe(self, az: float, alt: float) -> tuple[int, int]:
        moved_steps = self._mount_rate_safe(float(az), float(alt))
        self._goto.model.note_emitted_rate_steps(moved_steps)
        return moved_steps

    def _goto_move_steps(self, axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        if self._mount is None:
            raise RuntimeError("mount not connected")
        self._mount.move_steps(
            axis,
            direction,
            steps,
            delay_us,
            profile="direct",
            blocking=False,
            stop_before_move=False,
        )

    def _pause_tracking_for_goto(self) -> bool:
        was_tracking = self._get_tracking_enabled()
        if was_tracking:
            self._update_state(
                {
                    "tracking": {
                        "enabled": False,
                        "status": TrackingStatus.PAUSED,
                        "ff_enabled": bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)),
                        "ff_ready": False,
                    }
                }
            )
            self._mount_rate_safe(0.0, 0.0)
        return was_tracking

    def _resume_tracking_after_goto(self) -> None:
        self._update_state(
            {
                "tracking": {
                    "enabled": True,
                    "status": TrackingStatus.RUNNING,
                    "ff_enabled": bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)),
                    "ff_ready": False,
                }
            }
        )
        self._tracking_keyframe_reset()

    def _pause_stacking_for_goto(self) -> bool:
        was_stacking = bool(self._stacking_enabled)
        if was_stacking:
            self._stacking_enabled = False
            self._stacking.stop()
            self._update_state({"stacking": {"enabled": False, "status": StackingStatus.OFF}})
        return was_stacking

    def _resume_stacking_after_goto(self) -> None:
        self._stacking_last_frame_token = None
        self._stacking_enabled = True
        self._stacking.start()
        self._update_state({"stacking": {"enabled": True, "status": StackingStatus.RUNNING}})

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
            self._invalidate_tracking_pipeline()
            self._tracking_last_cmd_az = 0.0
            self._tracking_last_cmd_alt = 0.0
            self._stacking_last_frame_token = None
            self._reset_tracking_feedforward_cache()

        self._invalidate_preview_pipeline()
        with self._preview_lock:
            self._latest_preview_jpeg = None

        if self._cam_dev is not None:
            try:
                self._cam_dev.close()
            except Exception as exc:
                log_error(self.out_log, "Camera: device close failed", exc)
            self._cam_dev = None

        self._update_state(
            {
                "camera": {
                    "connected": False,
                    "status": CameraStatus.DISCONNECTED,
                    "fps_capture": 0.0,
                }
            }
        )
        self._release_simulation_if_idle()

    def _connect_camera(self, camera_index: int) -> None:
        self._shutdown_camera()
        self._update_state({"camera": {"status": CameraStatus.CONNECTING, "connected": False}})

        try:
            if self._simulation_enabled():
                sim_state = self._ensure_simulation_state()
                stream = SimulatedCameraStream(
                    state=sim_state,
                    cfg=self.cfg,
                    observer=self._platesolving_observer,
                    out_log=self.out_log,
                )
                stream.start()
                self._cam_dev = None
                self._cam_stream = stream
                snap = sim_state.snapshot()
                self._update_state(
                    {
                        "camera": {
                            "connected": True,
                            "status": CameraStatus.OK,
                            "last_error": None,
                            "roll_deg": float(self.cfg.camera.roll_deg)
                            + float(snap["camera_roll_error_deg"]),
                        }
                    }
                )
                log_info(
                    self.out_log,
                    (
                        "Camera: connected in DEMO mode "
                        f"frame={int(self.cfg.simulation.frame_w)}x{int(self.cfg.simulation.frame_h)} "
                        f"roll_error={float(snap['camera_roll_error_deg']):+.3f} deg"
                    ),
                )
                return

            dev = POACameraDevice()
            info = dev.open(camera_index)

            stream = CameraStream(ring=3)
            stream.start(dev, self.cfg.camera, self.cfg.preview)

            self._cam_dev = dev
            self._cam_stream = stream

            self._update_state({"camera": {"connected": True, "status": CameraStatus.OK, "last_error": None}})
            log_info(
                self.out_log,
                f"Camera: connected id={info.camera_id} model={info.model} sensor={info.sensor} "
                f"usb3={info.is_usb3} bayer={info.bayer_pattern} max={info.max_w}x{info.max_h}",
            )
        except Exception as exc:
            self._shutdown_camera()
            self._update_state(
                {
                    "camera": {
                        "connected": False,
                        "status": CameraStatus.ERROR,
                        "last_error": "connect failed",
                    }
                }
            )
            log_error(self.out_log, "Camera: connect failed (is it open in another app?)", exc)

    def _set_camera_param_value(self, name: str, value: Any) -> Optional[bool]:
        """Apply one runtime value and report whether capture must restart."""
        n = (name or "").strip()

        if n in ("exp_ms", "exposure_ms"):
            self.cfg.camera.exp_ms = float(value)
        elif n in ("gain",):
            self.cfg.camera.gain = int(value)
        elif n in ("offset", "black_level"):
            self.cfg.camera.offset = int(value)
        elif n in ("auto_gain",):
            self.cfg.camera.auto_gain = bool(value)
        elif n in ("gamma",):
            self.cfg.camera.gamma = float(value)
            return False
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
            return False
        elif n in ("preview_jpeg_quality",):
            self.cfg.preview.jpeg_quality = int(value)
            return False
        elif n in ("preview_stretch_plo",):
            self.cfg.preview.stretch_plo = float(value)
            return False
        elif n in ("preview_stretch_phi",):
            self.cfg.preview.stretch_phi = float(value)
            return False
        elif n in ("roll_deg", "camera_roll_deg"):
            roll = float(value)
            if not np.isfinite(roll):
                log_info(self.out_log, f"Camera: roll inválido ignorado: {value}")
                return None
            self.cfg.camera.roll_deg = float(roll)
            self.cfg.platesolving.rotation_prior_roll_offset_deg = float(roll)
            self._update_state({"camera": {"roll_deg": float(roll)}})
            return False
        else:
            log_info(self.out_log, f"Camera: param ignorado (no soportado aún): {n}={value}")
            return None

        return True

    def _apply_camera_params(self, params: Dict[str, Any]) -> None:
        restart_names: List[str] = []
        applied_names: List[str] = []
        for name, value in dict(params).items():
            needs_restart = self._set_camera_param_value(str(name), value)
            if needs_restart is None:
                continue
            applied_names.append(str(name))
            if needs_restart:
                restart_names.append(str(name))

        if restart_names:
            self._restart_camera_stream_if_active(
                reason=f"batched change: {', '.join(restart_names)}"
            )
        if applied_names:
            self._invalidate_preview_pipeline()
            log_info(self.out_log, f"Camera: SET_PARAMS {', '.join(applied_names)}")

    def _apply_camera_param(self, name: str, value: Any) -> None:
        self._apply_camera_params({str(name): value})

    def _restart_camera_stream_if_active(self, *, reason: str) -> None:
        if self._cam_stream is None:
            return
        try:
            cam_index = int(self.cfg.camera.camera_index)
            log_info(self.out_log, f"Camera: reconfigure (restart stream) due to {reason}")
            self._connect_camera(cam_index)
        except Exception as exc:
            self._update_state({"camera": {"status": CameraStatus.ERROR, "last_error": "reconfigure failed"}})
            log_error(self.out_log, "Camera: failed to apply config (restart)", exc)

    def _reset_camera_defaults(self) -> None:
        self.cfg.camera = replace(self.default_cfg.camera)
        self._update_state({"camera": {"roll_deg": float(self.cfg.camera.roll_deg)}})
        self._restart_camera_stream_if_active(reason="camera defaults reset")

    def _reset_preview_defaults(self) -> None:
        self.cfg.preview = replace(self.default_cfg.preview)
        self._expected_stars_mag_limit = float(self.cfg.preview.expected_stars_mag_limit)
        self._expected_stars_max = int(self.cfg.preview.expected_stars_max)
        self._invalidate_expected_stars_catalog()
        self._invalidate_preview_pipeline()

    def _reset_mount_defaults(self) -> None:
        self.cfg.mount = replace(self.default_cfg.mount)
        if self._mount is not None and self._mount.is_connected():
            self._mount_set_microsteps(self.cfg.mount.ms_az, self.cfg.mount.ms_alt)
        self._goto.model.kin.microsteps_az = int(self.cfg.mount.ms_az)
        self._goto.model.kin.microsteps_alt = int(self.cfg.mount.ms_alt)
        self._goto.model.kin.axis_sign_az = +1
        self._goto.model.kin.axis_sign_alt = _axis_sign_from_invert(self.cfg.mount.invert_alt)
        self._goto.model.init_from_mechanics()
        fit_report = self._goto.model.model_fit_report()
        self._update_state(
            {
                "goto": {
                    "J00": float(self._goto.model.J_deg_per_step[0, 0]),
                    "J01": float(self._goto.model.J_deg_per_step[0, 1]),
                    "J10": float(self._goto.model.J_deg_per_step[1, 0]),
                    "J11": float(self._goto.model.J_deg_per_step[1, 1]),
                    "J00_err": float(fit_report["J00_err"]),
                    "J01_err": float(fit_report["J01_err"]),
                    "J10_err": float(fit_report["J10_err"]),
                    "J11_err": float(fit_report["J11_err"]),
                    "model_non_orthogonality_deg": float(fit_report["model_non_orthogonality_deg"]),
                    "model_non_orthogonality_err_deg": float(fit_report["model_non_orthogonality_err_deg"]),
                    "model_camera_roll_deg": float(fit_report["model_roll_deg"]),
                    "model_camera_roll_err_deg": float(fit_report["model_roll_err_deg"]),
                    "model_camera_roll_samples": int(fit_report["model_roll_samples"]),
                    "model_pitch_deg": float(fit_report["model_pitch_deg"]),
                    "model_pitch_err_deg": float(fit_report["model_pitch_err_deg"]),
                    "model_yaw_deg": float(fit_report["model_yaw_deg"]),
                    "model_yaw_err_deg": float(fit_report["model_yaw_err_deg"]),
                    "model_fit_samples": int(fit_report["model_fit_samples"]),
                    "model_fit_rms_az_deg": float(fit_report["model_fit_rms_az_deg"]),
                    "model_fit_rms_alt_deg": float(fit_report["model_fit_rms_alt_deg"]),
                    "model_fit_rms_arcsec": float(fit_report["model_fit_rms_arcsec"]),
                    "periodic_error_az_deg": float(fit_report["periodic_error_az_deg"]),
                    "periodic_error_alt_deg": float(fit_report["periodic_error_alt_deg"]),
                    "periodic_model_samples": int(fit_report["periodic_model_samples"]),
                    "last_direction_az": int(self._goto.model.last_move_direction_az),
                    "last_direction_alt": int(self._goto.model.last_move_direction_alt),
                    "backlash_steps_az": int(self._goto.model.backlash_steps_az),
                    "backlash_steps_alt": int(self._goto.model.backlash_steps_alt),
                }
            }
        )

    def _reset_tracking_defaults(self) -> None:
        self.cfg.tracking = replace(self.default_cfg.tracking)
        with self._tracking_state_lock:
            tracking_set_params(
                self._tracking_state,
                resp_min=self.cfg.tracking.resp_min,
                align_median_k=int(self.cfg.stacking.align_median_k),
                align_smooth_k=int(self.cfg.stacking.smooth_k),
                align_max_shift_px=float(self.cfg.stacking.max_shift_px),
                align_use_subpixel=bool(self.cfg.stacking.use_subpixel),
            )
        self._update_state(
            {
                "tracking": {
                    "ff_enabled": bool(self.cfg.tracking.sidereal_ff_enabled),
                    "ff_ready": False,
                    "rate_ff_az": 0.0,
                    "rate_ff_alt": 0.0,
                }
            }
        )
        self._reset_tracking_feedforward_cache()
        self._tracking_keyframe_reset()

    def _reset_stacking_defaults(self) -> None:
        self.cfg.stacking = replace(self.default_cfg.stacking)
        self._stacking.configure_from_cfg()
        with self._tracking_state_lock:
            tracking_set_params(
                self._tracking_state,
                align_median_k=int(self.cfg.stacking.align_median_k),
                align_smooth_k=int(self.cfg.stacking.smooth_k),
                align_max_shift_px=float(self.cfg.stacking.max_shift_px),
                align_use_subpixel=bool(self.cfg.stacking.use_subpixel),
            )

    def _reset_platesolving_defaults(self) -> None:
        with self._platesolving_cfg_lock:
            self.cfg.platesolving = replace(self.default_cfg.platesolving)
            self._platesolving_observer = ObserverConfig()
            self._goto.cfg.observer = self._platesolving_observer
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

        view_hz = max(0.1, float(self.cfg.preview.view_hz))

        now = _perf()
        if (now - self._t_last_preview) < (1.0 / view_hz):
            return

        fr = self._cam_stream.latest()
        if fr is None:
            return

        token = self._frame_token(fr)
        if (
            self._preview_last_frame_token is not None
            and token == float(self._preview_last_frame_token)
        ):
            return

        try:
            raw16 = ensure_raw16_bayer(fr.raw).copy()
            with self._preview_config_lock:
                generation = int(self._preview_generation)
            self._preview_worker.request(
                generation=generation,
                token=float(token),
                raw16=raw16,
                wall_t=self._frame_wall_t(fr),
            )
            self._preview_last_frame_token = float(token)
            self._t_last_preview = now
        except Exception as exc:
            log_error(self.out_log, "Preview: enqueue failed", exc)

    def _render_preview_request(self, request: Dict[str, Any]) -> None:
        try:
            raw16_work = ensure_raw16_bayer(request["raw16"])
            with self._preview_config_lock:
                generation = int(request.get("generation", -1))
                if generation != int(self._preview_generation):
                    return

            state_snapshot = self.get_state()
            platesolving_overlay = list(
                getattr(state_snapshot.platesolving, "overlay", None) or []
            )
            overlay_enabled = bool(
                self._live_sep_overlay_enabled
                or self._expected_stars_overlay_enabled
                or platesolving_overlay
            )

            if overlay_enabled:
                _, u8_preview = make_preview_jpeg(
                    raw16_work,
                    plo=float(self.cfg.preview.stretch_plo),
                    phi=float(self.cfg.preview.stretch_phi),
                    jpeg_quality=int(self.cfg.preview.jpeg_quality),
                    sample_stride=4,
                    gamma=float(self.cfg.camera.gamma),
                )
                if self._live_sep_overlay_enabled:
                    u8_preview = self._apply_live_sep_overlay(raw16_work, u8_preview)
                if platesolving_overlay:
                    u8_preview = self._apply_platesolving_overlay(
                        u8_preview,
                        platesolving_overlay,
                    )
                if self._expected_stars_overlay_enabled:
                    wall_t = request.get("wall_t")
                    obstime = (
                        Time(float(wall_t), format="unix", scale="utc")
                        if wall_t is not None
                        else Time.now()
                    )
                    u8_preview = self._apply_expected_stars_overlay(
                        raw16_work,
                        u8_preview,
                        obstime=obstime,
                    )
                jpg = encode_jpeg(
                    u8_preview,
                    quality=int(self.cfg.preview.jpeg_quality),
                )
            else:
                jpg, _ = make_preview_jpeg(
                    raw16_work,
                    plo=float(self.cfg.preview.stretch_plo),
                    phi=float(self.cfg.preview.stretch_phi),
                    jpeg_quality=int(self.cfg.preview.jpeg_quality),
                    sample_stride=4,
                    gamma=float(self.cfg.camera.gamma),
                )

            with self._preview_config_lock:
                if generation != int(self._preview_generation):
                    return

            with self._preview_lock:
                self._latest_preview_jpeg = jpg

            now = _perf()
            self._n_view += 1
            if (now - self._t_fps_view0) >= 1.0:
                fps_view = self._n_view / (now - self._t_fps_view0)
                self._t_fps_view0 = now
                self._n_view = 0
                self._update_state({"camera": {"fps_view": float(fps_view)}})
        except Exception as exc:
            log_error(self.out_log, "Preview: failed", exc)

    def _invalidate_preview_pipeline(self) -> None:
        with self._preview_config_lock:
            self._preview_generation += 1
            self._preview_last_frame_token = None

    def _maybe_enqueue_stacking_frame(self) -> bool:
        if not self._stacking_enabled or self._cam_stream is None:
            return False
        fr = self._cam_stream.latest()
        if fr is None:
            return False

        token = self._frame_token(fr)
        if (
            self._stacking_last_frame_token is not None
            and token == float(self._stacking_last_frame_token)
        ):
            return False

        raw16 = ensure_raw16_bayer(fr.raw)
        self._stacking.enqueue_frame(raw16.copy(), t=_now_s())
        self._stacking_last_frame_token = float(token)
        return True

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

    def _apply_platesolving_overlay(
        self,
        u8_preview: np.ndarray,
        overlay: Sequence[Any],
    ) -> np.ndarray:
        """Draw persistent detections and the exact plate-solving seeds."""
        if u8_preview.ndim == 2:
            img = cv2.cvtColor(u8_preview, cv2.COLOR_GRAY2BGR)
        else:
            img = u8_preview.copy()
        h, w = img.shape[:2]
        colors = {
            "det": (255, 0, 0),
            "det_persistent": (255, 255, 0),
            "seed": (255, 0, 255),
            "match": (0, 255, 0),
            "guide": (0, 0, 255),
        }
        for item in overlay:
            if isinstance(item, dict):
                x_raw = item.get("x", 0.0)
                y_raw = item.get("y", 0.0)
                kind = str(item.get("kind", "det"))
                label = item.get("label", None)
            else:
                x_raw = getattr(item, "x", 0.0)
                y_raw = getattr(item, "y", 0.0)
                kind = str(getattr(item, "kind", "det"))
                label = getattr(item, "label", None)
            x = int(round(float(x_raw)))
            y = int(round(float(y_raw)))
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            color = colors.get(kind, (255, 255, 0))
            radius = 12 if kind == "seed" else 9 if kind in {"match", "guide"} else 7
            thickness = 2 if kind == "seed" else 1
            cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
            if kind == "seed":
                cv2.line(img, (x - 5, y), (x + 5, y), color, 1, lineType=cv2.LINE_AA)
                cv2.line(img, (x, y - 5), (x, y + 5), color, 1, lineType=cv2.LINE_AA)
            if kind in {"seed", "guide"} and label:
                cv2.putText(
                    img,
                    str(label),
                    (x + radius + 2, max(12, y - radius)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )
        return img

    def _invalidate_expected_stars_catalog(self) -> None:
        self._expected_stars_catalog = None
        self._expected_stars_catalog_center = None
        self._expected_stars_catalog_radius_deg = 0.0
        self._expected_stars_catalog_source = ""

    def _set_expected_stars_status(
        self,
        *,
        count: int = 0,
        source: str = "",
        reason: Optional[str] = None,
    ) -> None:
        self._update_state(
            {
                "goto": {
                    "expected_stars_overlay_enabled": bool(
                        self._expected_stars_overlay_enabled
                    ),
                    "expected_stars_overlay_count": int(max(0, count)),
                    "expected_stars_overlay_source": str(source),
                    "expected_stars_overlay_reason": reason,
                }
            }
        )

    def _expected_stars_model_center(self, *, obstime: Time) -> Optional[SkyCoord]:
        model = self._goto.model
        if int(getattr(model, "model_fit_samples", 0)) <= 0:
            return None
        if not bool(getattr(model, "synced", False)):
            return None
        az_alt = model.predict_az_alt_deg()
        if az_alt is None:
            return None
        az = float(az_alt[0]) % 360.0
        alt = float(np.clip(float(az_alt[1]), -90.0, 90.0))
        if not np.isfinite(az) or not np.isfinite(alt):
            return None
        with (
            iers.conf.set_temp("auto_download", False),
            iers.conf.set_temp("auto_max_age", None),
        ):
            true_altaz_observer = replace(
                self._platesolving_observer,
                refraction_enable=False,
            )
            return parse_target_to_icrs(
                {"az_deg": az, "alt_deg": alt},
                observer=true_altaz_observer,
                obstime=obstime,
            ).icrs

    def _load_expected_stars_catalog(
        self,
        *,
        center_icrs: SkyCoord,
        radius_deg: float,
    ) -> Any:
        cached_center = self._expected_stars_catalog_center
        cached_radius = float(self._expected_stars_catalog_radius_deg)
        if (
            self._expected_stars_catalog is not None
            and cached_center is not None
            and cached_radius >= radius_deg
            and float(cached_center.separation(center_icrs).deg)
            <= max(0.01, cached_radius - radius_deg)
        ):
            return self._expected_stars_catalog

        query_radius = max(0.20, float(radius_deg) * 1.8)
        cfg = replace(
            self._get_platesolving_cfg_snapshot(),
            download_missing_tiles=False,
        )
        source = "Gaia + Hipparcos/Tycho-2"
        try:
            tab = gaia_healpix_cone_with_mag(
                center_icrs=center_icrs,
                radius_deg=query_radius,
                cfg=cfg,
                verbose=False,
            )
        except GaiaCacheMissError:
            tab = bright_healpix_cone_with_mag(
                center_icrs=center_icrs,
                radius_deg=query_radius,
                cfg=cfg,
                mag_limit=self._expected_stars_mag_limit,
            )
            source = "Hipparcos/Tycho-2"

        frame = tab.to_pandas() if hasattr(tab, "to_pandas") else tab
        self._expected_stars_catalog = frame
        self._expected_stars_catalog_center = center_icrs
        self._expected_stars_catalog_radius_deg = query_radius
        self._expected_stars_catalog_source = source
        return frame

    def _apply_expected_stars_overlay(
        self,
        raw16: np.ndarray,
        u8_preview: np.ndarray,
        *,
        obstime: Time,
    ) -> np.ndarray:
        try:
            center = self._expected_stars_model_center(obstime=obstime)
            if center is None:
                self._set_expected_stars_status(reason="Se requiere un fit GoTo sincronizado")
                return u8_preview

            h, w = raw16.shape[:2]
            scale = (
                206265.0
                * float(self.cfg.platesolving.pixel_size_m)
                / float(self.cfg.platesolving.focal_m)
            )
            if not np.isfinite(scale) or scale <= 0.0:
                self._set_expected_stars_status(reason="Escala óptica inválida")
                return u8_preview
            radius_deg = (
                1.08 * 0.5 * float(np.hypot(w, h)) * scale / 3600.0
            )
            catalog = self._load_expected_stars_catalog(
                center_icrs=center,
                radius_deg=radius_deg,
            )
            if len(catalog) == 0:
                self._set_expected_stars_status(
                    source=self._expected_stars_catalog_source,
                    reason="Catálogo local vacío para este campo",
                )
                return u8_preview

            mags = np.asarray(catalog["phot_g_mean_mag"], dtype=np.float64)
            keep_mag = np.isfinite(mags) & (mags <= float(self._expected_stars_mag_limit))
            if not np.any(keep_mag):
                self._set_expected_stars_status(
                    source=self._expected_stars_catalog_source,
                    reason="Sin estrellas dentro del límite de magnitud",
                )
                return u8_preview
            subset = catalog.loc[keep_mag].reset_index(drop=True)
            mags = np.asarray(subset["phot_g_mean_mag"], dtype=np.float64)
            coords = SkyCoord(
                ra=np.asarray(subset["ra"], dtype=np.float64) * u.deg,
                dec=np.asarray(subset["dec"], dtype=np.float64) * u.deg,
                frame="icrs",
            )

            model = self._goto.model
            fitted_roll = float(getattr(model, "model_roll_deg", float("nan")))
            if int(getattr(model, "model_roll_samples", 0)) <= 0 or not np.isfinite(fitted_roll):
                fitted_roll = float(self.cfg.camera.roll_deg)
            with (
                iers.conf.set_temp("auto_download", False),
                iers.conf.set_temp("auto_max_age", None),
            ):
                theta = expected_field_rotation_deg(
                    float(center.ra.deg),
                    float(center.dec.deg),
                    observer=self._platesolving_observer,
                    obstime=obstime,
                    roll_offset_deg=fitted_roll,
                    az_step_deg=float(
                        getattr(self.cfg.platesolving, "rotation_prior_az_step_deg", 0.05)
                    ),
                )
            if theta is None or not np.isfinite(float(theta)):
                theta = 0.0
            pixels = project_catalog_to_pixels(
                coords,
                center_icrs=center,
                scale_arcsec_per_px=scale,
                theta_deg=float(theta),
                image_width=w,
                image_height=h,
            )
            in_view = (
                (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] < float(w))
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] < float(h))
            )
            pixels = pixels[in_view]
            mags = mags[in_view]
            if len(pixels) > int(self._expected_stars_max):
                order = np.argsort(mags)[: int(self._expected_stars_max)]
                pixels = pixels[order]
                mags = mags[order]

            if u8_preview.ndim == 2:
                img = cv2.cvtColor(u8_preview, cv2.COLOR_GRAY2BGR)
            else:
                img = u8_preview.copy()

            color = (255, 0, 255)
            for (x, y), mag in zip(pixels, mags):
                ix = int(round(float(x)))
                iy = int(round(float(y)))
                radius = int(np.clip(round(7.0 - 0.25 * float(mag)), 3, 9))
                cv2.circle(img, (ix, iy), radius, color, 1, lineType=cv2.LINE_AA)
                cv2.line(img, (ix - 2, iy), (ix + 2, iy), color, 1, cv2.LINE_AA)
                cv2.line(img, (ix, iy - 2), (ix, iy + 2), color, 1, cv2.LINE_AA)

            cx = int(round(w * 0.5))
            cy = int(round(h * 0.5))
            cv2.drawMarker(
                img,
                (cx, cy),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Modelo: {len(pixels)} estrellas",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
            self._set_expected_stars_status(
                count=len(pixels),
                source=self._expected_stars_catalog_source,
                reason=None,
            )
            return img
        except Exception as exc:
            log_error(
                self.out_log,
                "Expected stars overlay failed",
                exc,
                throttle_s=2.0,
                throttle_key="expected_stars_overlay",
            )
            self._set_expected_stars_status(reason=type(exc).__name__)
            return u8_preview

    # -------------------------
    # Mount
    # -------------------------
    def _shutdown_mount(self) -> None:
        self._reset_rate_emulation()
        self._clear_manual_move_activity()
        if self._mount is not None:
            try:
                self._mount.disconnect()
            except Exception as exc:
                log_error(self.out_log, "Mount: disconnect failed", exc)
        self._mount = None
        self._update_state({"mount": {"connected": False, "status": MountStatus.DISCONNECTED}})
        self._release_simulation_if_idle()

    def _connect_mount(self, port: str, baudrate: int) -> None:
        self._shutdown_mount()
        self._update_state({"mount": {"status": MountStatus.CONNECTING, "connected": False}})

        try:
            if self._simulation_enabled():
                sim_state = self._ensure_simulation_state()
                m = SimulatedMount(sim_state, out_log=self.out_log)
                msg = m.connect(port=str(port or "DEMO"), baud=int(baudrate))
                self._mount = m
                self._mount_set_microsteps(self.cfg.mount.ms_az, self.cfg.mount.ms_alt)
                snap = sim_state.snapshot()
                self._update_state({"mount": {"connected": True, "status": MountStatus.OK, "last_error": None}})
                log_info(
                    self.out_log,
                    (
                        f"Mount: connected in DEMO mode ({msg}); "
                        f"truth Az/Alt={float(snap['az_deg']):.3f}/{float(snap['alt_deg']):.3f} deg"
                    ),
                )
                return

            m = ArduinoMount()
            msg = m.connect(port=str(port), baud=int(baudrate))
            if "error" in str(msg).lower():
                self._shutdown_mount()
                self._update_state({"mount": {"connected": False, "status": MountStatus.ERROR, "last_error": str(msg)}})
                log_error(self.out_log, f"Mount: connect failed ({msg})")
                return
            self._mount = m
            # Ensure microstep settings are applied on every connect so manual speed is consistent.
            self._mount_set_microsteps(self.cfg.mount.ms_az, self.cfg.mount.ms_alt)
            self._update_state({"mount": {"connected": True, "status": MountStatus.OK, "last_error": None}})
            log_info(self.out_log, f"Mount: connected ({msg})")
        except Exception as exc:
            self._shutdown_mount()
            self._update_state({"mount": {"connected": False, "status": MountStatus.ERROR, "last_error": "connect failed"}})
            log_error(self.out_log, "Mount: connect failed", exc)

    def _mount_stop(self) -> None:
        self._reset_rate_emulation()
        self._clear_manual_move_activity()
        if self._mount is None:
            return
        try:
            self._mount.stop()
        except Exception as exc:
            self._update_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "stop failed"},
                    "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE},
                }
            )
            log_error(self.out_log, "Mount: STOP failed", exc)

    def _mount_set_microsteps(self, az_div: int, alt_div: int) -> None:
        if self._mount is None or not self._mount.is_connected():
            return
        if int(az_div) != 64 or int(alt_div) != 64:
            log_error(
                self.out_log,
                "Mount: ignored microstep change; hardware is fixed at 1/64",
                ValueError(
                    f"requested AZ=1/{int(az_div)} ALT=1/{int(alt_div)}"
                ),
            )
            return
        try:
            common_ms = resolve_common_microsteps(int(az_div), int(alt_div), default_ms=64)
            self._mount.stop()
            ms_resp = str(self._mount.set_microsteps(int(common_ms), int(common_ms)) or "").strip().upper()
            if not ms_resp.startswith("OK"):
                raise RuntimeError(f"invalid MS response: {ms_resp or 'NO-RESP'}")
            self._goto.model.set_microsteps(int(common_ms), int(common_ms))
            self.cfg.mount.ms_az = int(common_ms)
            self.cfg.mount.ms_alt = int(common_ms)
            log_info(self.out_log, "Mount: fixed microstepping confirmed (1/64)")
        except Exception as exc:
            self._update_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "set microsteps failed"},
                    "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE},
                }
            )
            log_error(self.out_log, "Mount: MS failed", exc)

    def _mount_move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> None:
        if self._mount is None or not self._mount.is_connected():
            return
        self._mark_manual_move_active(
            axis=axis,
            steps=int(steps),
            delay_us=int(delay_us),
            profile=profile,
        )
        self._start_operation("mount_move")
        self._mount_move_worker.request(
            axis=axis,
            direction=direction,
            steps=steps,
            delay_us=delay_us,
            profile=profile,
        )

    # -------------------------
    # Platesolving (worker)
    # -------------------------
    def _platesolving_request(self, *, target: Any) -> None:
        """
        Encola un request para platesolving. Si hay uno pendiente, se reemplaza.
        """
        obstime_unix = None
        if str(getattr(self, "_platesolving_source", "live")).lower() == "stack":
            info = self._stacking.get_stack_for_solve() if self._stacking is not None else None
            if info:
                obstime_unix = info.get("obstime_unix")
        self._platesolving_worker.request(target=target, obstime_unix=obstime_unix)

    def _current_field_center_icrs(self) -> Tuple[SkyCoord, str]:
        if self._simulation_enabled():
            sim_state = self._ensure_simulation_state()
            center = sim_state.center_icrs(observer=self._platesolving_observer, obstime=Time.now()).icrs
            return center, "simulation"

        st = self.get_state()
        if bool(st.platesolving.last_ok):
            ra = float(st.platesolving.center_ra_deg)
            dec = float(st.platesolving.center_dec_deg)
            if np.isfinite(ra) and np.isfinite(dec):
                return SkyCoord(ra=(ra % 360.0) * u.deg, dec=dec * u.deg, frame="icrs"), "platesolving"

        if bool(st.goto.pointing_valid):
            ra = float(st.goto.pointing_ra_deg)
            dec = float(st.goto.pointing_dec_deg)
            if np.isfinite(ra) and np.isfinite(dec):
                return SkyCoord(ra=(ra % 360.0) * u.deg, dec=dec * u.deg, frame="icrs"), "goto"

        raise RuntimeError("no current field is available")

    def _current_field_download_radius_deg(self, requested_radius_deg: Optional[float] = None) -> float:
        if requested_radius_deg is not None:
            requested = _finite_float(requested_radius_deg, 0.0)
            if requested > 0.0:
                return float(requested)

        ps_cfg = self._get_platesolving_cfg_snapshot()
        candidates: List[float] = []
        search_radius = getattr(ps_cfg, "search_radius_deg", None)
        if search_radius is not None:
            search_radius_f = _finite_float(search_radius, 0.0)
            if search_radius_f > 0.0:
                candidates.append(float(search_radius_f))

        if self._simulation_enabled():
            sim_cfg = self.cfg.simulation
            candidates.append(_finite_float(getattr(sim_cfg, "catalog_radius_deg", 1.2), 1.2))
            cam_cfg = self.cfg.camera
            if bool(getattr(cam_cfg, "use_roi", False)):
                w = int(max(32, int(getattr(cam_cfg, "roi_w", getattr(sim_cfg, "frame_w", 1280)))))
                h = int(max(32, int(getattr(cam_cfg, "roi_h", getattr(sim_cfg, "frame_h", 720)))))
            else:
                w = int(max(32, int(getattr(sim_cfg, "frame_w", 1280))))
                h = int(max(32, int(getattr(sim_cfg, "frame_h", 720))))
            pixel_size_m = _finite_float(getattr(ps_cfg, "pixel_size_m", 2.9e-6), 2.9e-6)
            focal_m = _finite_float(getattr(ps_cfg, "focal_m", 0.9), 0.9)
            if focal_m <= 0.0:
                focal_m = 0.9
            scale_arcsec_px = float(206265.0 * pixel_size_m / focal_m)
            fov_radius_deg = float(np.hypot(w, h) * 0.5 * scale_arcsec_px / 3600.0)
            candidates.append(float(fov_radius_deg * 1.8))

        radius = max(candidates) if candidates else 1.0
        return float(max(0.01, radius))

    def _request_gaia_current_field_download(self, radius_deg: Optional[float] = None) -> None:
        with self._gaia_download_lock:
            current = self._gaia_download_thread
            if current is not None and current.is_alive():
                log_info(self.out_log, "Gaia download: already running")
                return

        if self._platesolving_worker.is_busy():
            log_info(self.out_log, "Gaia download: skipped because plate solving is busy")
            return

        try:
            center_icrs, source = self._current_field_center_icrs()
            radius = self._current_field_download_radius_deg(radius_deg)
            cfg = self._get_platesolving_cfg_snapshot()
            cfg.download_missing_tiles = True
        except Exception as exc:
            self._update_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": "NO_CURRENT_FIELD",
                        "debug_info": {"status": "NO_CURRENT_FIELD"},
                    }
                }
            )
            log_error(self.out_log, "Gaia download: failed to resolve current field", exc)
            return

        def _worker() -> None:
            current_thread = threading.current_thread()
            debug_base = {
                "status": "GAIA_DOWNLOAD_RUNNING",
                "source": source,
                "radius_deg": float(radius),
                "center_ra_deg": float(center_icrs.ra.deg),
                "center_dec_deg": float(center_icrs.dec.deg),
            }
            self._update_state(
                {
                    "platesolving": {
                        "busy": True,
                        "status": PlatesolvingStatus.RUNNING,
                        "reason": None,
                        "debug_info": dict(debug_base),
                    }
                }
            )
            try:
                t0 = _perf()
                tab = gaia_healpix_cone_with_mag(
                    center_icrs=center_icrs,
                    radius_deg=float(radius),
                    cfg=cfg,
                    auth=load_gaia_auth(),
                    verbose=True,
                )
                rows = int(len(tab))
                elapsed_s = float(_perf() - t0)

                stream = self._cam_stream
                if stream is not None and hasattr(stream, "invalidate_catalog_cache"):
                    stream.invalidate_catalog_cache()
                    self._update_state({"camera": {"last_error": None}})

                debug_ok = dict(debug_base)
                debug_ok.update({"status": "GAIA_DOWNLOAD_OK", "rows": rows, "elapsed_s": elapsed_s})
                self._update_state(
                    {
                        "platesolving": {
                            "busy": False,
                            "status": PlatesolvingStatus.OK,
                            "reason": None,
                            "debug_info": debug_ok,
                        }
                    }
                )
                log_info(
                    self.out_log,
                    (
                        "Gaia download: OK "
                        f"source={source} center=({center_icrs.ra.deg:.5f},{center_icrs.dec.deg:.5f}) "
                        f"radius={radius:.3f}deg rows={rows} elapsed={elapsed_s:.1f}s"
                    ),
                )
            except Exception as exc:
                debug_fail = dict(debug_base)
                debug_fail.update({"status": "GAIA_DOWNLOAD_FAILED", "error": type(exc).__name__})
                self._update_state(
                    {
                        "platesolving": {
                            "busy": False,
                            "status": PlatesolvingStatus.FAIL,
                            "reason": "GAIA_DOWNLOAD_FAILED",
                            "debug_info": debug_fail,
                        }
                    }
                )
                log_error(self.out_log, "Gaia download: failed", exc)
            finally:
                with self._gaia_download_lock:
                    if self._gaia_download_thread is current_thread:
                        self._gaia_download_thread = None

        thr = threading.Thread(target=_worker, name="GaiaCurrentFieldDownload", daemon=True)
        with self._gaia_download_lock:
            self._gaia_download_thread = thr
        thr.start()

    def _maybe_autosolve(self) -> None:
        cfg = self._get_platesolving_cfg_snapshot()
        if not bool(cfg.auto_solve):
            return
        target = str(self._platesolving_auto_target or "").strip()
        if not target:
            return
        st = self.get_state()
        if bool(st.platesolving.busy):
            return
        now = _perf()
        if (now - float(self._platesolving_last_auto_t)) < max(2.0, float(cfg.solve_every_s)):
            return
        self._platesolving_request(target=target)
        self._platesolving_last_auto_t = float(now)

    # -------------------------
    # Stacking save helper
    # -------------------------
    def _stacking_capture_basename(self, basename: str) -> str:
        prefix = str(basename).strip() or "stack"
        safe_prefix = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in prefix)
        safe_prefix = safe_prefix.strip("_") or "stack"

        capture_dt = _dt.datetime.now()
        fr = self._get_latest_frame()
        if fr is not None:
            t_wall = self._frame_wall_t(fr)
            if t_wall is not None:
                try:
                    capture_dt = _dt.datetime.fromtimestamp(float(t_wall))
                except (OSError, OverflowError, ValueError) as exc:
                    log_error(
                        self.out_log,
                        "Stacking: invalid frame wall timestamp; using current time",
                        exc,
                        throttle_s=5.0,
                        throttle_key="stacking_capture_timestamp",
                    )

        az = float("nan")
        alt = float("nan")
        try:
            az_alt = self._goto.model.current_az_alt_deg()
        except Exception as exc:
            log_error(
                self.out_log,
                "Stacking: failed to read current pointing for output name",
                exc,
                throttle_s=10.0,
                throttle_key="stacking_capture_pointing",
            )
            az_alt = None

        if az_alt is not None and len(az_alt) >= 2:
            try:
                az = float(az_alt[0]) % 360.0
                alt = float(np.clip(float(az_alt[1]), -90.0, 90.0))
            except Exception as exc:
                log_error(
                    self.out_log,
                    "Stacking: invalid current pointing for output name",
                    exc,
                    throttle_s=10.0,
                    throttle_key="stacking_capture_pointing_invalid",
                )
                az = float("nan")
                alt = float("nan")

        if (not np.isfinite(az)) or (not np.isfinite(alt)):
            st = self.get_state()
            if bool(getattr(st.goto, "pointing_valid", False)):
                az = float(st.goto.pointing_az_deg) % 360.0
                alt = float(np.clip(float(st.goto.pointing_alt_deg), -90.0, 90.0))

        def _coord_token(value: float, *, signed: bool) -> str:
            if not np.isfinite(value):
                return "NA"
            if signed:
                sign = "m" if value < 0.0 else "p"
                return f"{sign}{abs(value):05.2f}".replace(".", "p")
            return f"{value:06.2f}".replace(".", "p")

        stamp = capture_dt.strftime("%Y%m%d_%H%M%S")
        az_txt = _coord_token(az, signed=False)
        alt_txt = _coord_token(alt, signed=True)
        return f"{safe_prefix}_{stamp}_az{az_txt}_alt{alt_txt}"

    def _save_stacking(self, out_dir: str, basename: str, fmt: str) -> None:
        """
        Save current live-stacking mosaic to disk.

        Two files are produced:
          - a raw floating-point numpy array (.npy) with stacked mean;
          - a logarithmic stretched PNG image (.png) in uint16.
        The output directory is created if necessary.  Errors are logged but
        otherwise ignored.

        Parameters
        ----------
        out_dir : str
            Directory in which to save the files.
        basename : str
            Prefix used for the output files.  The final file name also
            includes capture timestamp and pointing az/alt.
        fmt : str
            Ignored; included for API compatibility.
        """
        eng = self._stacking
        try:
            raw, _ = eng.get_stack_snapshot(mean_dtype=np.float32, wgt_dtype=np.float32)
            if raw is None:
                log_info(self.out_log, "Stacking: save skipped (no data)")
                return

            # Create output directory
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._update_state({"stacking": {"status": StackingStatus.ERROR, "last_error": "save failed"}})
                log_error(self.out_log, f"Stacking: failed to create output directory {out_dir}", exc)
                return

            final_basename = self._stacking_capture_basename(basename)

            # Save raw stack
            raw_path = os.path.join(out_dir, f"{final_basename}_raw.npy")
            np.save(raw_path, raw)

            # Save logarithmic PNG (uint16)
            img = np.log(raw.astype(np.float64) + 1.0)
            vmin = float(img.mean() - 0.1)
            vmax = float(img.max())
            denom = (vmax - vmin) if (vmax > vmin) else 1.0
            img_clip = np.clip(img, vmin, vmax)
            img_u16 = ((img_clip - vmin) * (65535.0 / denom)).astype(np.uint16)

            png_path = os.path.join(out_dir, f"{final_basename}.png")
            if img_u16.ndim == 2:
                ok_png = cv2.imwrite(png_path, img_u16)
            else:
                ok_png = cv2.imwrite(png_path, cv2.cvtColor(img_u16, cv2.COLOR_RGB2BGR))
            if not ok_png:
                raise RuntimeError(f"cv2.imwrite failed for {png_path}")

            log_info(self.out_log, f"Stacking: saved raw to {raw_path} and png to {png_path}")
        except Exception as exc:
            self._update_state({"stacking": {"status": StackingStatus.ERROR, "last_error": "save failed"}})
            log_error(self.out_log, "Stacking: save failed", exc)

    def _handle_save_request(self, request: Dict[str, Any]) -> None:
        self._save_stacking(
            str(request.get("out_dir", "stack_output")),
            str(request.get("basename", "stack")),
            str(request.get("fmt", "png")),
        )

    # -------------------------
    # Raw recording helper
    # -------------------------
    def _start_raw_recording(
        self,
        *,
        duration_s: Optional[float],
        out_dir: str,
        basename: Optional[str] = None,
    ) -> None:
        if self._cam_stream is None:
            log_info(self.out_log, "Raw record: skipped (camera not connected)")
            return

        with self._raw_record_lock:
            if self._raw_record_active:
                log_info(self.out_log, "Raw record: already running")
                return
            self._raw_record_active = True
            self._raw_record_stop.clear()
        self._start_operation("camera_record")

        duration_s = None if duration_s is None else max(0.1, float(duration_s))
        if not basename:
            basename = _dt.datetime.now().strftime("raw_%Y%m%d_%H%M%S")

        def _worker() -> None:
            frames: list[np.ndarray] = []
            frame_metadata: list[dict[str, Any]] = []
            try:
                stream = self._cam_stream
                if stream is None:
                    log_info(self.out_log, "Raw record: aborted (camera disconnected)")
                    return

                t0 = _perf()
                last_token: Optional[float] = None

                if duration_s is None:
                    log_info(self.out_log, "Raw record: started; waiting for Stop")
                else:
                    log_info(self.out_log, f"Raw record: start duration={duration_s:.1f}s")
                while (
                    not self._raw_record_stop.is_set()
                    and not self._stop.is_set()
                    and (duration_s is None or (_perf() - t0) < duration_s)
                ):
                    fr = stream.latest()
                    if fr is None:
                        time.sleep(0.001)
                        continue
                    seq = self._frame_seq(fr)
                    token = float(seq) if seq is not None else float(fr.t_capture)
                    if last_token is not None and token == last_token:
                        time.sleep(0.0005)
                        continue
                    last_token = token
                    try:
                        raw16 = ensure_raw16_bayer(fr.raw)
                    except Exception as exc:
                        log_error(
                            self.out_log,
                            "Raw record: frame is not RAW16 Bayer; saving numpy view fallback",
                            exc,
                            throttle_s=5.0,
                            throttle_key="raw_record_frame_format",
                        )
                        raw16 = np.asarray(fr.raw)
                    frames.append(raw16.copy())
                    frame_metadata.append(
                        {
                            "index": int(len(frames) - 1),
                            "seq": int(seq) if seq is not None else None,
                            "t_capture_s": float(fr.t_capture),
                            "t_wall_unix": self._frame_wall_t(fr),
                        }
                    )

                if not frames:
                    log_info(self.out_log, "Raw record: no frames captured")
                    return

                try:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    log_error(self.out_log, f"Raw record: failed to create output directory {out_dir}", exc)
                    return
                out_path = os.path.join(out_dir, f"{basename}.npy")
                stack = np.stack(frames, axis=0)
                np.save(out_path, stack)
                metadata_path = str(Path(out_path).with_suffix(".json"))
                with open(metadata_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "raw_file": str(Path(out_path).resolve()),
                            "shape": [int(value) for value in stack.shape],
                            "dtype": str(stack.dtype),
                            "duration_requested_s": None if duration_s is None else float(duration_s),
                            "duration_actual_s": float(max(0.0, _perf() - t0)),
                            "stopped_by_user": bool(self._raw_record_stop.is_set()),
                            "camera": {
                                "exp_ms": float(self.cfg.camera.exp_ms),
                                "gain": int(self.cfg.camera.gain),
                                "offset": int(self.cfg.camera.offset),
                            },
                            "frames": frame_metadata,
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                log_info(
                    self.out_log,
                    f"Raw record: saved {stack.shape[0]} frames to {out_path} "
                    f"with timing metadata {metadata_path} "
                    f"(shape={stack.shape}, dtype={stack.dtype})",
                )
            except Exception as exc:
                log_error(self.out_log, "Raw record: failed", exc)
            finally:
                with self._raw_record_lock:
                    self._raw_record_active = False
                    self._raw_record_thread = None
                self._finish_operation("camera_record")

        self._raw_record_thread = threading.Thread(target=_worker, name="RawRecord", daemon=True)
        self._raw_record_thread.start()

    def _stop_raw_recording(self) -> None:
        with self._raw_record_lock:
            active = bool(self._raw_record_active)
        if active:
            self._raw_record_stop.set()
            log_info(self.out_log, "Raw record: stop requested")


    # -------------------------
    # Main loop
    # -------------------------
    def _run(self) -> None:
        dt_target = 1.0 / max(1.0, float(self.cfg.control_hz))
        t_last = _perf()

        while not self._stop.is_set():
            t0 = _perf()
            perf_sections: Dict[str, float] = {}
            section_t = t0
            state_publish_hz = max(
                0.5,
                min(
                    60.0,
                    _finite_float(getattr(self.cfg, "state_publish_hz", 10.0), 10.0),
                ),
            )
            publish_state = (
                t0 - float(self._t_last_state_publish)
            ) >= (1.0 / state_publish_hz)
            if publish_state:
                self._t_last_state_publish = t0

            # 1) actions
            self._drain_actions(max_n=50)
            section_end = _perf()
            perf_sections["actions_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 2) stats capture
            if publish_state and self._cam_stream is not None:
                st = self._cam_stream.stats()
                camera_patch = {"fps_capture": float(st.get("fps_capture", 0.0))}
                if "last_error" in st:
                    camera_patch["last_error"] = st.get("last_error")
                self._update_state({"camera": camera_patch})
            section_end = _perf()
            perf_sections["state_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 2b) tracking
            tracking_on = self._get_tracking_enabled()
            if tracking_on and (self._cam_stream is not None) and (self._mount is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    frame_t = self._frame_mono_t(fr)
                    frame_token = self._frame_token(fr)
                    with self._tracking_result_lock:
                        last_token = self._tracking_last_frame_token
                    is_new_frame = (
                        last_token is None
                        or frame_token != float(last_token)
                    )

                    if is_new_frame:
                        try:
                            self._submit_tracking_frame(
                                raw16=ensure_raw16_bayer(fr.raw).copy(),
                                frame_token=float(frame_token),
                                frame_t=float(frame_t),
                                tracking_enabled=bool(tracking_on),
                            )
                        except Exception as exc:
                            with self._tracking_result_lock:
                                self._tracking_worker_error = exc

                    out, tracking_error = self._tracking_result_snapshot()
                    if tracking_error is not None:
                        self._invalidate_tracking_pipeline()
                        self._update_state(
                            {
                                "tracking": {
                                    "enabled": False,
                                    "status": TrackingStatus.ERROR,
                                    "mode": TrackingMode.IDLE,
                                    "last_error": "tracking step failed",
                                }
                            }
                        )
                        try:
                            self._mount_rate_safe(0.0, 0.0)
                        except Exception as stop_exc:
                            log_error(
                                self.out_log,
                                "Tracking: failed to stop mount after tracking error",
                                stop_exc,
                                throttle_s=2.0,
                                throttle_key="tracking_stop_after_error",
                            )
                        log_error(
                            self.out_log,
                            "Tracking: step failed",
                            tracking_error,
                            throttle_s=2.0,
                            throttle_key="tracking_step",
                        )
                        continue

                    if out is None:
                        out = TrackingOutput(
                            ok=False,
                            mode="IDLE",
                            resp=0.0,
                            dx=0.0,
                            dy=0.0,
                            vx=0.0,
                            vy=0.0,
                            abs_resp=0.0,
                            x_hat=0.0,
                            y_hat=0.0,
                            rate_az=0.0,
                            rate_alt=0.0,
                            calib_src="none",
                            detA=0.0,
                            n_det=0,
                            measurement_reason="processing",
                            measurement_source="none",
                        )

                    self._observe_transmission_error(out)

                    rate_fb_az = float(out.rate_az)
                    rate_fb_alt = float(out.rate_alt)
                    rate_ff_az = 0.0
                    rate_ff_alt = 0.0
                    ff_ready = False
                    if bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)):
                        rate_ff_az, rate_ff_alt, ff_ready = self._cached_tracking_feedforward_rate(
                            now_t=float(_now_s())
                        )

                    rate_cmd_az = float(rate_fb_az + rate_ff_az)
                    rate_cmd_alt = float(rate_fb_alt + rate_ff_alt)
                    rate_cmd_az, rate_cmd_alt = self._clip_tracking_rate_pair(rate_cmd_az, rate_cmd_alt)

                    try:
                        self._tracking_rate_safe(float(rate_cmd_az), float(rate_cmd_alt))
                        self._tracking_last_cmd_az = float(rate_cmd_az)
                        self._tracking_last_cmd_alt = float(rate_cmd_alt)
                    except Exception as exc:
                        self._tracking_last_cmd_az = 0.0
                        self._tracking_last_cmd_alt = 0.0
                        self._update_state(
                            {
                                "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "tracking move failed"},
                                "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE, "last_error": "mount MOVE failed", "n_det": int(out.n_det)},
                            }
                        )
                        log_error(self.out_log, "Tracking: mount MOVE failed", exc, throttle_s=2.0, throttle_key="tracking_mount_move")

                    if publish_state:
                        self._publish_tracking_output(
                            out,
                            tracking_on=bool(tracking_on),
                            ff_ready=bool(ff_ready),
                            rate_cmd_az=float(rate_cmd_az),
                            rate_cmd_alt=float(rate_cmd_alt),
                            rate_fb_az=float(rate_fb_az),
                            rate_fb_alt=float(rate_fb_alt),
                            rate_ff_az=float(rate_ff_az),
                            rate_ff_alt=float(rate_ff_alt),
                        )
            else:
                goto_busy = bool(self.get_state().goto.busy)
                if self._mount is not None and not goto_busy:
                    self._mount_rate_safe(0.0, 0.0)
                self._tracking_last_cmd_az = 0.0
                self._tracking_last_cmd_alt = 0.0
                if publish_state:
                    self._publish_tracking_off()

            section_end = _perf()
            perf_sections["tracking_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 2c) stacking
            if self._stacking_enabled and (self._cam_stream is not None):
                try:
                    self._maybe_enqueue_stacking_frame()
                except Exception as exc:
                    self._update_state({"stacking": {"last_error": "enqueue failed"}})
                    log_error(
                        self.out_log,
                        "Stacking: enqueue failed",
                        exc,
                        throttle_s=2.0,
                        throttle_key="stacking_enqueue",
                    )

            # 2d) publish stacking metrics
            if publish_state:
                m = self._stacking.metrics
                self._update_state(
                    {
                        "stacking": {
                            "enabled": bool(m.enabled),
                            "status": StackingStatus.RUNNING if m.enabled else StackingStatus.OFF,
                            "fps": float(getattr(m, "stacking_fps", 0.0)),
                            "tiles_used": int(getattr(m, "tiles_used", 0)),
                            "tiles_evicted": int(getattr(m, "tiles_evicted", 0)),
                            "frames_in": int(getattr(m, "frames_in", 0)),
                            "frames_used": int(getattr(m, "frames_used", 0)),
                            "frames_dropped": int(getattr(m, "frames_dropped", 0)),
                            "frames_rejected": int(getattr(m, "frames_rejected", 0)),
                            "last_resp": float(getattr(m, "last_resp", 0.0)),
                            "last_dx": float(getattr(m, "last_dx", 0.0)),
                            "last_dy": float(getattr(m, "last_dy", 0.0)),
                            "last_theta_deg": float(getattr(m, "last_theta_deg", 0.0)),
                            "preview_jpeg": self._stacking.get_preview_jpeg(),
                        }
                    }
                )
            section_end = _perf()
            perf_sections["stacking_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 2e) platesolving autosolve scheduling (if enabled)
            self._maybe_autosolve()
            section_end = _perf()
            perf_sections["autosolve_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 2f) live pointing readout (Az/Alt + RA/Dec updated with current time)
            self._maybe_update_goto_pointing_state(now=t0)
            section_end = _perf()
            perf_sections["pointing_ms"] = (section_end - section_t) * 1000.0
            section_t = section_end

            # 3) preview
            self._maybe_update_preview()
            section_end = _perf()
            perf_sections["preview_ms"] = (section_end - section_t) * 1000.0

            # 4) loop stats
            t1 = _perf()
            frame_ms = (t1 - t0) * 1000.0
            perf_sections["total_ms"] = frame_ms
            self._record_loop_performance(perf_sections)
            if publish_state:
                self._update_state({"camera": {"frame_ms": float(frame_ms)}})

            self._n_loop += 1
            if (t1 - self._t_fps_loop0) >= 1.0:
                fps_loop = self._n_loop / (t1 - self._t_fps_loop0)
                self._t_fps_loop0 = t1
                self._n_loop = 0
                self._update_state({"camera": {"fps_control_loop": float(fps_loop)}})

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
                if act.type in (
                    ActionType.CAMERA_CONNECT,
                    ActionType.CAMERA_SET_PARAM,
                    ActionType.CAMERA_SET_PARAMS,
                ):
                    self._update_state({"camera": {"status": CameraStatus.ERROR, "connected": False, "last_error": "action failed"}})

                if act.type in (
                    ActionType.STACKING_START,
                    ActionType.STACKING_STOP,
                    ActionType.STACKING_RESET,
                    ActionType.STACKING_SET_PARAMS,
                    ActionType.STACKING_SAVE,
                    ActionType.RESET_STACKING_DEFAULTS,
                ):
                    self._update_state({"stacking": {"status": StackingStatus.ERROR, "last_error": "action failed"}})

                if act.type in (
                    ActionType.PLATESOLVING_RUN,
                    ActionType.PLATESOLVING_SET_PARAMS,
                    ActionType.PLATESOLVING_DOWNLOAD_CURRENT_FIELD,
                    ActionType.RESET_PLATESOLVING_DEFAULTS,
                    ActionType.LIVE_SEP_SET_PARAMS,
                ):
                    self._update_state(
                        {
                            "platesolving": {
                                "busy": False,
                                "status": PlatesolvingStatus.FAIL,
                                "reason": "ACTION_FAILED",
                                "last_ok": False,
                            }
                        }
                    )

                if act.type in (
                    ActionType.MOUNT_SYNC,
                    ActionType.MOUNT_GOTO,
                    ActionType.GOTO_CALIBRATE,
                    ActionType.GOTO_AUTOCALIBRATE,
                    ActionType.GOTO_ESTIMATE_ROLL,
                    ActionType.GOTO_FIT_MODEL,
                    ActionType.GOTO_RESET,
                    ActionType.GOTO_CANCEL,
                    ActionType.GOTO_LIST_SAMPLES,
                    ActionType.GOTO_PRUNE_OUTLIERS,
                    ActionType.GOTO_RESTORE_LAST_LOG,
                ):
                    self._update_state({"goto": {"busy": False, "status": GotoStatus.FAIL, "reason": "ACTION_FAILED"}})

                if act.type in (
                    ActionType.MOUNT_CONNECT,
                    ActionType.MOUNT_STOP,
                    ActionType.MOUNT_SET_MICROSTEPS,
                    ActionType.MOUNT_MOVE_STEPS,
                    ActionType.TRACKING_START,
                    ActionType.TRACKING_STOP,
                    ActionType.TRACKING_SET_PARAMS,
                    ActionType.TRACKING_KEYFRAME_RESET,
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
                        self._update_state(
                            {
                                "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "action failed"},
                                "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE},
                            }
                        )

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

        if t == ActionType.CAMERA_SET_PARAMS:
            self._apply_camera_params(p if isinstance(p, dict) else {})
            return

        if t == ActionType.CAMERA_RECORD_RAW:
            if isinstance(p, dict):
                raw_duration = p.get("duration_s", 20.0)
                duration_s = None if raw_duration is None else float(raw_duration)
                out_dir = str(p.get("out_dir", "raw_output"))
                basename = p.get("basename", None)
                self._start_raw_recording(duration_s=duration_s, out_dir=out_dir, basename=basename)
            return

        if t == ActionType.CAMERA_STOP_RECORD_RAW:
            self._stop_raw_recording()
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
            self._platesolving_worker.cancel_current()
            move_was_active = bool(
                self._mount_move_worker.is_busy()
                or self._is_manual_move_active()
                or self.get_state().goto.busy
                or self._get_tracking_enabled()
            )
            self._goto_worker.cancel()
            self._reset_tracking_feedforward_cache()
            self._update_state(
                {
                    "tracking": {
                        "enabled": False,
                        "status": TrackingStatus.OFF,
                        "mode": TrackingMode.IDLE,
                        "measurement_valid": False,
                        "measurement_reason": "emergency_stop",
                    },
                    "goto": {
                        "busy": False,
                        "status": GotoStatus.CANCELLED,
                        "reason": (
                            "STOP_POSITION_UNKNOWN"
                            if move_was_active
                            else "STOPPED"
                        ),
                    },
                }
            )
            if move_was_active:
                # Firmware STOP does not report how many steps of an in-flight
                # MOVE were completed. Keeping the old sync would make the
                # next GoTo unsafe.
                self._goto.model.synced = False
                self._update_state(
                    {"goto": {"synced": False, "pointing_valid": False}}
                )
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
            profile = str(p.get("profile", "smooth"))
            self._mount_move_steps(axis, direction, steps, delay_us, profile)
            self._tracking_keyframe_reset()
            return

        if t == ActionType.RESET_MOUNT_DEFAULTS:
            self._reset_mount_defaults()
            log_info(self.out_log, "Mount: RESET_DEFAULTS")
            return

        if self._handle_tracking_action(t, p):
            return
        if self._handle_stacking_action(t, p):
            return
        if self._handle_platesolving_action(t, p):
            return
        if self._handle_goto_action(t, p):
            return

        # ---- Otros ----
        log_info(self.out_log, f"Unknown or unhandled action type: {t}")

    def _handle_tracking_action(self, t: ActionType, p: Dict[str, Any]) -> bool:
        if t == ActionType.TRACKING_START:
            seeded = self._tracking_seed_calibration_from_pointing()
            with self._tracking_state_lock:
                auto_ready = bool(
                    self._tracking_state.auto.ok
                    and self._tracking_state.auto.A_pinv is not None
                )
                if (not seeded) and (not auto_ready):
                    auto_reset(self._tracking_state, src="auto")
            self._invalidate_tracking_pipeline()
            self._tracking_last_cmd_az = 0.0
            self._tracking_last_cmd_alt = 0.0
            self._reset_tracking_feedforward_cache()
            self._update_state(
                {
                    "tracking": {
                        "enabled": True,
                        "status": TrackingStatus.RUNNING,
                        "mode": TrackingMode.IDLE,
                        "ff_enabled": bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)),
                        "ff_ready": False,
                        "measurement_valid": False,
                        "measurement_reason": "initializing",
                        "measurement_source": "none",
                        "lock_conf": 0.0,
                        "fail_count": 0,
                        "last_error": None,
                    }
                }
            )
            self._reset_rate_emulation()
            self._mount_rate_safe(0.0, 0.0)
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: START")
            return True

        if t == ActionType.TRACKING_STOP:
            self._invalidate_tracking_pipeline()
            self._reset_tracking_feedforward_cache()
            self._update_state(
                {
                    "tracking": {
                        "enabled": False,
                        "status": TrackingStatus.OFF,
                        "mode": TrackingMode.IDLE,
                        "ff_enabled": bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)),
                        "ff_ready": False,
                        "rate_ff_az": 0.0,
                        "rate_ff_alt": 0.0,
                        "rate_fb_az": 0.0,
                        "rate_fb_alt": 0.0,
                        "measurement_valid": False,
                        "measurement_reason": "off",
                        "measurement_source": "none",
                        "lock_conf": 0.0,
                        "fail_count": 0,
                        "last_error": None,
                    }
                }
            )
            self._reset_rate_emulation()
            self._mount_rate_safe(0.0, 0.0)
            log_info(self.out_log, "Tracking: STOP")
            return True

        if t == ActionType.TRACKING_KEYFRAME_RESET:
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: KEYFRAME_RESET")
            return True

        if t == ActionType.TRACKING_SET_PARAMS:
            if isinstance(p, dict):
                updates = dict(p)
                tracking_updates = dict(updates)
                ff_update_requested = any(
                    str(key).startswith("sidereal_ff_") or str(key).startswith("ff_")
                    for key in updates
                )

                ff_enabled = updates.get("sidereal_ff_enabled", updates.get("ff_enabled", None))
                if ff_enabled is not None:
                    self.cfg.tracking.sidereal_ff_enabled = bool(ff_enabled)
                    tracking_updates.pop("sidereal_ff_enabled", None)
                    tracking_updates.pop("ff_enabled", None)
                    if not bool(self.cfg.tracking.sidereal_ff_enabled):
                        self._reset_tracking_feedforward_cache()

                ff_update_hz = updates.get(
                    "sidereal_ff_update_hz",
                    updates.get("ff_update_hz", None),
                )
                if ff_update_hz is not None:
                    v = _finite_float(ff_update_hz, 2.0)
                    self.cfg.tracking.sidereal_ff_update_hz = float(
                        max(0.1, min(20.0, v))
                    )
                    tracking_updates.pop("sidereal_ff_update_hz", None)
                    tracking_updates.pop("ff_update_hz", None)

                ff_gain = updates.get("sidereal_ff_gain", updates.get("ff_gain", None))
                if ff_gain is not None:
                    v = float(ff_gain)
                    self.cfg.tracking.sidereal_ff_gain = float(v if np.isfinite(v) else 1.0)
                    tracking_updates.pop("sidereal_ff_gain", None)
                    tracking_updates.pop("ff_gain", None)

                ff_dt_s = updates.get("sidereal_ff_dt_s", updates.get("ff_dt_s", None))
                if ff_dt_s is not None:
                    v = float(ff_dt_s)
                    if (not np.isfinite(v)) or v <= 1e-3:
                        v = 1.0
                    self.cfg.tracking.sidereal_ff_dt_s = float(v)
                    tracking_updates.pop("sidereal_ff_dt_s", None)
                    tracking_updates.pop("ff_dt_s", None)

                ff_cond_max = updates.get("sidereal_ff_cond_max", updates.get("ff_cond_max", None))
                if ff_cond_max is not None:
                    v = float(ff_cond_max)
                    if (not np.isfinite(v)) or v <= 1.0:
                        v = 5_000.0
                    self.cfg.tracking.sidereal_ff_cond_max = float(v)
                    tracking_updates.pop("sidereal_ff_cond_max", None)
                    tracking_updates.pop("ff_cond_max", None)

                ff_hold_s = updates.get("sidereal_ff_hold_s", updates.get("ff_hold_s", None))
                if ff_hold_s is not None:
                    v = float(ff_hold_s)
                    if (not np.isfinite(v)) or v < 0.0:
                        v = 8.0
                    self.cfg.tracking.sidereal_ff_hold_s = float(v)
                    tracking_updates.pop("sidereal_ff_hold_s", None)
                    tracking_updates.pop("ff_hold_s", None)

                ff_slew_per_s = updates.get("sidereal_ff_slew_per_s", updates.get("ff_slew_per_s", None))
                if ff_slew_per_s is not None:
                    v = float(ff_slew_per_s)
                    if (not np.isfinite(v)) or v <= 0.0:
                        v = 120.0
                    self.cfg.tracking.sidereal_ff_slew_per_s = float(v)
                    tracking_updates.pop("sidereal_ff_slew_per_s", None)
                    tracking_updates.pop("ff_slew_per_s", None)

                if tracking_updates:
                    with self._tracking_state_lock:
                        tracking_set_params(self._tracking_state, **tracking_updates)
                if ff_update_requested:
                    self._reset_tracking_feedforward_cache()
                self._update_state(
                    {
                        "tracking": {
                            "ff_enabled": bool(self.cfg.tracking.sidereal_ff_enabled),
                        }
                    }
                )
                log_info(self.out_log, f"Tracking: SET_PARAMS {_format_params(updates)}")
            return True

        if t == ActionType.RESET_TRACKING_DEFAULTS:
            self._reset_tracking_defaults()
            log_info(self.out_log, "Tracking: RESET_DEFAULTS")
            return True

        if t == ActionType.TRACKING_CALIB_RESET:
            with self._tracking_state_lock:
                auto_reset(self._tracking_state, src="auto")
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: CALIB_RESET (autocal only)")
            return True

        if t == ActionType.TRACKING_AUTO_RESET:
            with self._tracking_state_lock:
                auto_reset(self._tracking_state, src="auto")
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: AUTO_RESET")
            return True

        if t == ActionType.TRACKING_CALIB_AZ:
            log_info(self.out_log, "Tracking: CALIB_AZ ignored (autocalibration only)")
            return True

        if t == ActionType.TRACKING_CALIB_ALT:
            log_info(self.out_log, "Tracking: CALIB_ALT ignored (autocalibration only)")
            return True

        if t == ActionType.TRACKING_BOOTSTRAP:
            log_info(self.out_log, "Tracking: BOOTSTRAP ignored (autocalibration only)")
            return True

        return False

    def _handle_stacking_action(self, t: ActionType, p: Dict[str, Any]) -> bool:
        if t == ActionType.STACKING_START:
            self._stacking_last_frame_token = None
            self._stacking_enabled = True
            self._stacking.start()
            self._update_state({"stacking": {"enabled": True, "status": StackingStatus.RUNNING, "last_error": None}})
            log_info(self.out_log, "Stacking: START")
            return True

        if t == ActionType.STACKING_STOP:
            self._stacking_last_frame_token = None
            self._stacking_enabled = False
            self._stacking.stop()
            self._update_state({"stacking": {"enabled": False, "status": StackingStatus.OFF, "last_error": None}})
            log_info(self.out_log, "Stacking: STOP")
            return True

        if t == ActionType.STACKING_RESET:
            self._stacking_last_frame_token = None
            self._stacking.reset()
            self._update_state({"stacking": {"last_error": None}})
            log_info(self.out_log, "Stacking: RESET")
            return True

        if t == ActionType.STACKING_SET_PARAMS:
            if isinstance(p, dict):
                self._stacking.set_params(**p)
                align_updates = {}
                if "align_median_k" in p:
                    align_updates["align_median_k"] = int(self.cfg.stacking.align_median_k)
                if "smooth_k" in p:
                    align_updates["align_smooth_k"] = int(self.cfg.stacking.smooth_k)
                if "max_shift_px" in p:
                    align_updates["align_max_shift_px"] = float(self.cfg.stacking.max_shift_px)
                if "use_subpixel" in p:
                    align_updates["align_use_subpixel"] = bool(self.cfg.stacking.use_subpixel)
                if align_updates:
                    with self._tracking_state_lock:
                        tracking_set_params(self._tracking_state, **align_updates)
                log_info(self.out_log, f"Stacking: SET_PARAMS {list(p.keys())}")
            return True

        if t == ActionType.RESET_STACKING_DEFAULTS:
            self._reset_stacking_defaults()
            log_info(self.out_log, "Stacking: RESET_DEFAULTS")
            return True

        if t == ActionType.STACKING_SAVE:
            if isinstance(p, dict):
                out_dir = str(p.get("out_dir", "stack_output"))
                basename = str(p.get("basename", "stack"))
                fmt = str(p.get("fmt", "png"))
                self._save_worker.request(
                    out_dir=out_dir,
                    basename=basename,
                    fmt=fmt,
                )
            else:
                self._save_worker.request(
                    out_dir="stack_output",
                    basename="stack",
                    fmt="png",
                )
            log_info(self.out_log, "Stacking: save queued")
            return True

        return False

    def _handle_platesolving_action(self, t: ActionType, p: Dict[str, Any]) -> bool:
        if t == ActionType.PLATESOLVING_SET_PARAMS:
            if isinstance(p, dict):
                payload = dict(p)
                if "auto_target" in payload:
                    self._platesolving_auto_target = str(payload.pop("auto_target") or "")
                if "source" in payload:
                    source = str(payload.pop("source") or "live").strip().lower()
                    if source not in {"live", "stack"}:
                        log_info(self.out_log, f"Platesolving: fuente inválida {source!r} (use live|stack)")
                    else:
                        self._platesolving_source = source
                        log_info(self.out_log, f"Platesolving: fuente = {source}")

                observer_payload = {}
                observer_keymap = {
                    "observer_lat_deg": "lat_deg",
                    "observer_lon_deg": "lon_deg",
                    "observer_height_m": "height_m",
                    "observer_refraction_enable": "refraction_enable",
                    "observer_refraction_P_hPa": "refraction_P_hPa",
                    "observer_refraction_T_C": "refraction_T_C",
                }
                for key, observer_field in observer_keymap.items():
                    if key in payload:
                        observer_payload[observer_field] = payload.pop(key)

                with self._platesolving_cfg_lock:
                    if "temporal_min_hits" in payload:
                        payload["temporal_min_hits"] = max(
                            10, int(payload["temporal_min_hits"])
                        )
                    if "temporal_window_frames" in payload:
                        payload["temporal_window_frames"] = max(
                            10, int(payload["temporal_window_frames"])
                        )
                    d = dict(self.cfg.platesolving.__dict__)
                    for k, v in payload.items():
                        if k in d:
                            d[k] = v
                    d["temporal_window_frames"] = max(
                        int(d.get("temporal_min_hits", 10)),
                        int(d.get("temporal_window_frames", 12)),
                    )
                    self.cfg.platesolving = PlatesolvingConfig(**d)
                    if observer_payload:
                        obs = dict(self._platesolving_observer.__dict__)
                        try:
                            if "lat_deg" in observer_payload:
                                obs["lat_deg"] = float(observer_payload["lat_deg"])
                            if "lon_deg" in observer_payload:
                                obs["lon_deg"] = float(observer_payload["lon_deg"])
                            if "height_m" in observer_payload:
                                obs["height_m"] = float(observer_payload["height_m"])
                            if "refraction_enable" in observer_payload:
                                obs["refraction_enable"] = bool(observer_payload["refraction_enable"])
                            if "refraction_P_hPa" in observer_payload:
                                obs["refraction_P_hPa"] = float(observer_payload["refraction_P_hPa"])
                            if "refraction_T_C" in observer_payload:
                                obs["refraction_T_C"] = float(observer_payload["refraction_T_C"])
                            self._platesolving_observer = ObserverConfig(**obs)
                            self._goto.cfg.observer = self._platesolving_observer
                        except Exception as exc:
                            log_error(self.out_log, "Platesolving: observer params rejected", exc)
                if payload:
                    log_info(self.out_log, "Platesolving: params updated")
                if observer_payload:
                    log_info(self.out_log, "Platesolving: observer updated")
            return True

        if t == ActionType.RESET_PLATESOLVING_DEFAULTS:
            self._reset_platesolving_defaults()
            log_info(self.out_log, "Platesolving: RESET_DEFAULTS")
            return True

        if t == ActionType.PLATESOLVING_DOWNLOAD_CURRENT_FIELD:
            radius = p.get("radius_deg", None)
            self._request_gaia_current_field_download(radius_deg=radius)
            return True

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
                if "max_det" in p:
                    self.cfg.platesolving.max_det = int(p.get("max_det"))
                    self._live_sep_params["max_det"] = int(p.get("max_det"))
                self._goto.cfg.sep = self.cfg.sep
                self._invalidate_preview_pipeline()
                log_info(self.out_log, "Live SEP: params updated")
            return True

        if t == ActionType.EXPECTED_STARS_SET_PARAMS:
            if isinstance(p, dict):
                enabled = bool(p.get("enabled", self._expected_stars_overlay_enabled))
                mag_limit = _finite_float(
                    p.get("mag_limit", self._expected_stars_mag_limit),
                    self._expected_stars_mag_limit,
                )
                max_stars = int(p.get("max_stars", self._expected_stars_max))
                mag_limit = float(np.clip(mag_limit, -2.0, self.cfg.platesolving.gmax))
                max_stars = int(np.clip(max_stars, 1, 5000))
                params_changed = (
                    mag_limit != self._expected_stars_mag_limit
                    or max_stars != self._expected_stars_max
                )
                self._expected_stars_overlay_enabled = enabled
                self._expected_stars_mag_limit = mag_limit
                self._expected_stars_max = max_stars
                self.cfg.preview.expected_stars_mag_limit = mag_limit
                self.cfg.preview.expected_stars_max = max_stars
                if params_changed:
                    self._invalidate_expected_stars_catalog()
                reason = None
                if enabled and int(getattr(self._goto.model, "model_fit_samples", 0)) <= 0:
                    reason = "Se requiere un fit GoTo"
                self._set_expected_stars_status(reason=reason)
                self._invalidate_preview_pipeline()
                log_info(
                    self.out_log,
                    "Expected stars overlay: "
                    f"{'ON' if enabled else 'OFF'} "
                    f"mag<={mag_limit:.1f} max={max_stars}",
                )
            return True

        if t == ActionType.PLATESOLVING_RUN:
            target = p.get("target", None)
            user = str(p.get("gaia_username", "")).strip()
            pw = str(p.get("gaia_password", "")).strip()
            if user and pw:
                save_gaia_auth(user, pw)
                log_info(self.out_log, "Platesolving: Gaia credentials saved")
            self._last_platesolving_result = None
            self._update_state(
                {
                    "goto": {
                        "sample_last_ok": False,
                        "sample_last_reason": "SOLVING",
                    }
                }
            )
            self._platesolving_request(target=target)
            log_info(self.out_log, "Platesolving: RUN source=live")
            return True

        return False

    def _handle_goto_action(self, t: ActionType, p: Dict[str, Any]) -> bool:
        if t == ActionType.GOTO_VALIDATE_SAMPLE:
            sol = p.get("result", None)
            if sol is None or not bool(getattr(sol, "success", False)):
                self._update_state(
                    {
                        "goto": {
                            "status": GotoStatus.FAIL,
                            "reason": "SAMPLE_NO_VALID_SOLUTION",
                            "sample_last_ok": False,
                            "sample_last_reason": "SAMPLE_NO_VALID_SOLUTION",
                        }
                    }
                )
                log_info(self.out_log, "GoTo: automatic sample validation rejected (no valid solution)")
                return True
            try:
                ps_cfg = self._get_platesolving_cfg_snapshot()
                n_inliers = int(getattr(sol, "n_inliers", 0))
                rms_px = float(getattr(sol, "rms_px", float("inf")))
                min_inliers = max(1, int(getattr(ps_cfg, "min_inliers", 1)))
                max_rms_px = float(getattr(ps_cfg, "max_rms_px", 0.0))
                quality_reason: Optional[str] = None
                if n_inliers < min_inliers:
                    quality_reason = "SAMPLE_INSUFFICIENT_INLIERS"
                elif not np.isfinite(rms_px):
                    quality_reason = "SAMPLE_INVALID_RMS"
                elif max_rms_px > 0.0 and rms_px > max_rms_px:
                    quality_reason = "SAMPLE_RMS_TOO_HIGH"

                solve_obstime = _platesolving_result_obstime(sol)
                az_alt = platesolving_center_to_altaz_deg(
                    float(sol.center_ra_deg),
                    float(sol.center_dec_deg),
                    observer=self._platesolving_observer,
                    obstime=solve_obstime,
                )
                roll = platesolving_roll_sample_deg(
                    sol,
                    observer=self._platesolving_observer,
                    obstime=solve_obstime,
                )
                if not np.isfinite(roll):
                    quality_reason = quality_reason or "SAMPLE_INVALID_ROLL"

                continuity = self._goto.model.manual_sample_continuity_report(
                    az_alt,
                    roll_deg=roll,
                )
                if not bool(continuity.get("motion_ok", False)):
                    quality_reason = quality_reason or "SAMPLE_MOTION_MISMATCH"
                if not bool(continuity.get("roll_ok", False)):
                    quality_reason = quality_reason or "SAMPLE_ROLL_MISMATCH"

                # Estimar Roll establishes an independent axis orientation.
                # Apply it as a first-sample guard, modulo 180 degrees because
                # a plate axis has no arrow direction.
                model_roll_samples = int(getattr(self._goto.model, "model_roll_samples", 0))
                model_roll_deg = float(getattr(self._goto.model, "model_roll_deg", float("nan")))
                roll_tolerance = max(
                    0.0,
                    float(getattr(self._goto.model, "manual_sample_roll_tolerance_deg", 12.0)),
                )
                roll_axis_error = float("nan")
                if model_roll_samples > 0 and np.isfinite(model_roll_deg) and np.isfinite(roll):
                    roll_axis_error = roll_axis_distance_deg(roll, model_roll_deg)
                    if roll_tolerance > 0.0 and roll_axis_error > roll_tolerance:
                        quality_reason = quality_reason or "SAMPLE_ESTIMATED_ROLL_MISMATCH"

                if quality_reason is not None:
                    self._last_platesolving_result = None
                    self._update_state(
                        {
                            "goto": {
                                "status": GotoStatus.FAIL,
                                "reason": quality_reason,
                                "sample_last_ok": False,
                                "sample_last_reason": quality_reason,
                                "sample_last_az_deg": float(az_alt[0]),
                                "sample_last_alt_deg": float(az_alt[1]),
                                "sample_last_roll_deg": float(roll),
                            },
                            "platesolving": {
                                "last_ok": False,
                                "reason": quality_reason,
                            },
                        }
                    )
                    log_info(
                        self.out_log,
                        "GoTo: sample automatically rejected "
                        f"reason={quality_reason} inliers={n_inliers}/{min_inliers} "
                        f"rms={rms_px:.3f}/{max_rms_px:.3f}px "
                        f"motion={float(continuity.get('observed_motion_deg', float('nan'))):.4f}deg "
                        f"limit={float(continuity.get('motion_limit_deg', float('nan'))):.4f}deg "
                        f"roll_jump={float(continuity.get('roll_jump_deg', float('nan'))):.3f}deg "
                        f"roll_est_error={roll_axis_error:.3f}deg",
                    )
                    return True
                debug_info = dict(self.get_state().platesolving.debug_info or {})
                source = str(debug_info.get("diagnostics_dir", "") or "")
                n_samples = self._goto.model.add_manual_sample(
                    az_alt,
                    roll_deg=roll,
                    source=source or None,
                )
                # A verified plate solve is the best available absolute
                # pointing measurement. Rebase the existing fitted model at
                # the current emitted-step counter so waypoint errors cannot
                # accumulate into the next GoTo. This preserves J, fit
                # diagnostics, and all manual samples.
                self._goto.model.sync_from_world_az_alt(az_alt)
                self._update_state(
                    {
                        "goto": {
                            "J00": float(self._goto.model.J_deg_per_step[0, 0]),
                            "J01": float(self._goto.model.J_deg_per_step[0, 1]),
                            "J10": float(self._goto.model.J_deg_per_step[1, 0]),
                            "J11": float(self._goto.model.J_deg_per_step[1, 1]),
                            "manual_samples": int(n_samples),
                            "status": GotoStatus.OK,
                            "reason": None,
                            "sample_last_ok": True,
                            "sample_last_reason": "SAMPLE_ACCEPTED_AUTOMATICALLY",
                            "sample_last_az_deg": float(az_alt[0]),
                            "sample_last_alt_deg": float(az_alt[1]),
                            "sample_last_roll_deg": float(roll),
                        },
                    }
                )
                log_info(
                    self.out_log,
                    "GoTo: sample automatically accepted "
                    f"n={n_samples} inliers={n_inliers} rms={rms_px:.3f}px "
                    f"az={float(az_alt[0]):.6f} alt={float(az_alt[1]):.6f} roll={float(roll):+.3f}",
                )
            except Exception as exc:
                self._update_state(
                    {
                        "goto": {
                            "status": GotoStatus.FAIL,
                            "reason": "SAMPLE_VALIDATION_FAILED",
                            "sample_last_ok": False,
                            "sample_last_reason": "SAMPLE_VALIDATION_FAILED",
                        }
                    }
                )
                log_error(self.out_log, "GoTo: automatic sample validation failed", exc)
            return True

        if t == ActionType.MOUNT_SYNC:
            manual_az = p.get("az_deg")
            manual_alt = p.get("alt_deg")
            if manual_az is not None and manual_alt is not None:
                az_alt = np.asarray([float(manual_az), float(manual_alt)], dtype=np.float64)
                ok = bool(
                    np.all(np.isfinite(az_alt))
                    and 0.0 <= float(az_alt[0]) <= 360.0
                    and -10.0 <= float(az_alt[1]) <= 90.0
                    and self._goto.model.sync_from_world_az_alt(az_alt)
                )
                self._update_state(
                    {
                        "goto": {
                            "synced": bool(ok),
                            "status": GotoStatus.OK if ok else GotoStatus.FAIL,
                            "reason": None if ok else "SYNC_INVALID_ALTAZ",
                        }
                    }
                )
                log_info(
                    self.out_log,
                    "GoTo: explicit AltAz sync "
                    f"{'OK' if ok else 'ERR'} az={float(az_alt[0]):.6f} "
                    f"alt={float(az_alt[1]):.6f}",
                )
                return True
            sol = getattr(self, "_last_platesolving_result", None)
            if sol is None or not bool(getattr(sol, "success", False)):
                log_info(self.out_log, "GoTo: sync failed (no successful platesolving cached)")
                self._update_state(
                    {
                        "goto": {
                            "synced": False,
                            "status": GotoStatus.FAIL,
                            "reason": "SYNC_NO_SOLUTION",
                        }
                    }
                )
                return True
            ok = False
            try:
                ok = bool(self._goto.sync_from_platesolving(sol))
            except Exception as exc:
                log_error(self.out_log, "GoTo: sync exception", exc)
            self._update_state(
                {
                    "goto": {
                        "synced": bool(ok),
                        "status": GotoStatus.OK if ok else GotoStatus.FAIL,
                        "reason": None if ok else "SYNC_FAILED",
                    }
                }
            )
            log_info(self.out_log, f"GoTo: sync {'OK' if ok else 'ERR'}")
            return True

        if t == ActionType.MOUNT_GOTO:
            target = p.get("target", {})
            self._goto_worker.request(kind="goto", target=target, params=p)
            return True

        if t == ActionType.GOTO_CALIBRATE:
            params = p.get("params", {})
            self._goto_worker.request(kind="calibrate", target=None, params=params)
            return True

        if t == ActionType.GOTO_AUTOCALIBRATE:
            params = p.get("params", {})
            self._goto_worker.request(kind="autocal", target=None, params=params)
            return True

        if t == ActionType.GOTO_ESTIMATE_ROLL:
            params = p.get("params", {})
            self._goto_worker.request(kind="roll", target=None, params=params)
            return True

        if t == ActionType.GOTO_FIT_MODEL:
            params = p.get("params", {})
            self._goto_worker.request(kind="fit_model", target=None, params=params)
            return True

        if t == ActionType.GOTO_LIST_SAMPLES:
            params = p.get("params", {})
            self._goto_worker.request(kind="list_samples", target=None, params=params)
            return True

        if t == ActionType.GOTO_PRUNE_OUTLIERS:
            params = p.get("params", {})
            self._goto_worker.request(kind="prune_outliers", target=None, params=params)
            return True

        if t == ActionType.GOTO_RESET:
            self._last_platesolving_result = None
            self._update_state(
                {
                    "goto": {"sample_last_ok": False, "sample_last_reason": None},
                    "platesolving": {"last_ok": False, "reason": None},
                }
            )
            self._goto_worker.request(kind="reset", target=None, params={})
            return True

        if t == ActionType.GOTO_RESTORE_LAST_LOG:
            self._goto_worker.request(kind="restore_last_log", target=None, params={})
            return True

        if t == ActionType.GOTO_CANCEL:
            move_was_active = bool(
                self._goto_worker.is_busy()
                or self._mount_move_worker.is_busy()
                or self._is_manual_move_active()
                or self.get_state().goto.busy
            )
            self._goto_worker.cancel()
            self._mount_stop()
            if move_was_active:
                self._goto.model.synced = False
            self._update_state(
                {
                    "goto": {
                        "busy": False,
                        "status": GotoStatus.CANCELLED,
                        "reason": (
                            "CANCEL_POSITION_UNKNOWN"
                            if move_was_active
                            else "CANCELLED"
                        ),
                        "synced": bool(self._goto.model.synced),
                        "pointing_valid": bool(self._goto.model.synced),
                    }
                }
            )
            return True

        return False
