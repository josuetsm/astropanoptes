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
    camera_set_param,
    goto_autocalibrate,
    goto_calibrate,
    goto_fit_model,
    goto_cancel,
    live_sep_set_params,
    mount_connect,
    mount_disconnect,
    mount_goto,
    mount_move_steps,
    mount_set_microsteps,
    mount_stop,
    mount_sync,
    platesolving_run,
    platesolving_set_params,
    stacking_reset,
    stacking_save,
    stacking_start,
    stacking_stop,
    tracking_set_params,
    tracking_start,
    tracking_stop,
)
from logging_utils import log_info, log_error

from camera_poa import POACameraDevice, CameraStream
from imaging import ensure_raw16_bayer
from preview import make_preview_jpeg, encode_jpeg, stretch_to_u8
from mount_arduino import ArduinoMount

from tracking import (
    auto_reset,
    make_tracking_state,
    tracking_step,
    tracking_set_params,
)
from stacking import StackingWorker

from sep_utils import sep_detect_from_raw16

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
        - actualiza AppState
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
        self._raw_record_lock = threading.Lock()
        self._raw_record_active = False
        self._raw_record_thread: Optional[threading.Thread] = None

        # Subsystems
        self._cam_dev: Optional[POACameraDevice] = None
        self._cam_stream: Optional[CameraStream] = None
        self._mount: Optional[ArduinoMount] = None

        # Tracking subsystem
        self._tracking_state = make_tracking_state()
        tracking_set_params(
            self._tracking_state,
            resp_min=self.cfg.tracking.resp_min,
            sep_bw=int(self.cfg.sep.bw),
            sep_bh=int(self.cfg.sep.bh),
            sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
            sep_minarea=int(self.cfg.sep.minarea),
            sep_max_sources=int(self.cfg.platesolving.max_det),
        )

        # Stacking subsystem
        self._stacking = StackingWorker(self.cfg)
        self._stacking_enabled = bool(self.cfg.stacking.enabled_init)

        # Platesolving subsystem
        self._platesolving_cfg_lock = threading.Lock()
        self._platesolving_last_auto_t = 0.0
        self._platesolving_auto_target: str = ""

        # Config platesolving (runtime copy, actualizable desde UI por action)
        self._platesolving_observer = ObserverConfig()  # Algarrobo por default en tu platesolving.py
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
            publish_state=self._update_state,
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
            rate_mount=self._mount_rate_safe,
            move_steps=self._goto_move_steps,
            stop_mount=self._mount_stop,
            out_log=self.out_log,
        )
        # State + outputs (thread-safe)
        self._state = AppState()
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
                    "resp": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "abs_resp": 0.0,
                    "rate_az": 0.0,
                    "rate_alt": 0.0,
                    "calib_src": "none",
                    "calib_det": 0.0,
                    "n_det": 0,
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
                    "last_error_arcsec": 0.0,
                    "J00": float(self._goto.model.J_deg_per_step[0, 0]),
                    "J01": float(self._goto.model.J_deg_per_step[0, 1]),
                    "J10": float(self._goto.model.J_deg_per_step[1, 0]),
                    "J11": float(self._goto.model.J_deg_per_step[1, 1]),
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

    def _frame_seq(self, fr: Frame) -> Optional[int]:
        seq = fr.meta.get("seq")
        if seq is None:
            return None
        return int(seq)

    def _tracking_mode_from_output(self, mode: str) -> TrackingMode:
        try:
            return TrackingMode(str(mode))
        except ValueError:
            return TrackingMode.IDLE

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

    def _publish_platesolving_state(self, patch: Dict[str, Dict[str, Any]]) -> None:
        result = patch.pop("platesolving_result", None)
        self._update_state(patch)
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

    def get_state(self) -> AppState:
        with self._state_lock:
            return self._state.snapshot()

    def get_latest_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._latest_preview_jpeg

    def request_camera_connect(self, camera_index: int) -> None:
        self.enqueue(camera_connect(camera_index))

    def request_camera_disconnect(self) -> None:
        self.enqueue(camera_disconnect())

    def request_camera_param(self, name: str, value: Any) -> None:
        self.enqueue(camera_set_param(name, value))

    def request_camera_record_raw(
        self,
        *,
        duration_s: float = 20.0,
        out_dir: str = "raw_output",
        basename: Optional[str] = None,
    ) -> None:
        self.enqueue(camera_record_raw(duration_s=duration_s, out_dir=out_dir, basename=basename))

    def request_mount_connect(self, port: str, baudrate: int) -> None:
        self.enqueue(mount_connect(port, baudrate))

    def request_mount_disconnect(self) -> None:
        self.enqueue(mount_disconnect())

    def request_mount_set_microsteps(self, az_div: int, alt_div: int) -> None:
        self.enqueue(mount_set_microsteps(az_div=az_div, alt_div=alt_div))

    def request_mount_move_steps(self, axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        self.enqueue(mount_move_steps(axis=axis, direction=direction, steps=steps, delay_us=delay_us))

    def request_mount_stop(self) -> None:
        self.enqueue(mount_stop())

    def request_mount_sync(self) -> None:
        self.enqueue(mount_sync())

    def request_mount_goto(self, target: Any, **kwargs: Any) -> None:
        self.enqueue(mount_goto(target, **kwargs))

    def request_goto_calibrate(self, params: Dict[str, Any]) -> None:
        self.enqueue(goto_calibrate(params))

    def request_goto_autocalibrate(self) -> None:
        self.enqueue(goto_autocalibrate())

    def request_goto_fit_model(self, params: Dict[str, Any] | None = None) -> None:
        self.enqueue(goto_fit_model(params))

    def request_goto_cancel(self) -> None:
        self.enqueue(goto_cancel())

    def request_tracking_start(self) -> None:
        self.enqueue(tracking_start())

    def request_tracking_stop(self) -> None:
        self.enqueue(tracking_stop())

    def request_tracking_params(self, **kwargs: Any) -> None:
        self.enqueue(tracking_set_params(**kwargs))

    def request_stacking_start(self) -> None:
        self.enqueue(stacking_start())

    def request_stacking_stop(self) -> None:
        self.enqueue(stacking_stop())

    def request_stacking_reset(self) -> None:
        self.enqueue(stacking_reset())

    def request_stacking_save(self, **kwargs: Any) -> None:
        self.enqueue(stacking_save(**kwargs))

    def request_platesolving_run(self, target: str) -> None:
        self.enqueue(platesolving_run(target=target))

    def request_platesolving_params(self, **kwargs: Any) -> None:
        self.enqueue(platesolving_set_params(**kwargs))

    def request_live_sep_params(self, **kwargs: Any) -> None:
        self.enqueue(live_sep_set_params(**kwargs))

    # -------------------------
    # Internal helpers
    # -------------------------
    def _update_state(self, patch: Dict[str, Dict[str, Any]]) -> None:
        with self._state_lock:
            self._state.update(patch)

    def _get_tracking_enabled(self) -> bool:
        with self._state_lock:
            return bool(self._state.tracking.enabled)

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
            self._update_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "RATE failed"},
                    "tracking": {
                        "enabled": False,
                        "status": TrackingStatus.OFF,
                        "mode": TrackingMode.IDLE,
                        "last_error": "mount RATE failed",
                    },
                }
            )
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
            self._update_state({"tracking": {"enabled": False, "status": TrackingStatus.PAUSED}})
            self._mount_rate_safe(0.0, 0.0)
        return was_tracking

    def _resume_tracking_after_goto(self) -> None:
        self._update_state({"tracking": {"enabled": True, "status": TrackingStatus.RUNNING}})
        self._tracking_keyframe_reset()

    def _pause_stacking_for_goto(self) -> bool:
        was_stacking = bool(self._stacking_enabled)
        if was_stacking:
            self._stacking_enabled = False
            self._update_state({"stacking": {"enabled": False, "status": StackingStatus.OFF}})
        return was_stacking

    def _resume_stacking_after_goto(self) -> None:
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

    def _connect_camera(self, camera_index: int) -> None:
        self._shutdown_camera()
        self._update_state({"camera": {"status": CameraStatus.CONNECTING, "connected": False}})

        try:
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
            self._update_state({"camera": {"status": CameraStatus.ERROR, "last_error": "reconfigure failed"}})
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
        self._update_state(
            {
                "goto": {
                    "J00": float(self._goto.model.J_deg_per_step[0, 0]),
                    "J01": float(self._goto.model.J_deg_per_step[0, 1]),
                    "J10": float(self._goto.model.J_deg_per_step[1, 0]),
                    "J11": float(self._goto.model.J_deg_per_step[1, 1]),
                }
            }
        )

    def _reset_tracking_defaults(self) -> None:
        self.cfg.tracking = replace(self.default_cfg.tracking)
        tracking_set_params(
            self._tracking_state,
            resp_min=self.cfg.tracking.resp_min,
            sep_bw=int(self.cfg.sep.bw),
            sep_bh=int(self.cfg.sep.bh),
            sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
            sep_minarea=int(self.cfg.sep.minarea),
            sep_max_sources=int(self.cfg.platesolving.max_det),
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
                self._update_state({"camera": {"fps_view": float(fps_view)}})

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
        self._update_state({"mount": {"connected": False, "status": MountStatus.DISCONNECTED}})

    def _connect_mount(self, port: str, baudrate: int) -> None:
        self._shutdown_mount()
        self._update_state({"mount": {"status": MountStatus.CONNECTING, "connected": False}})

        try:
            m = ArduinoMount()
            msg = m.connect(port=str(port), baud=int(baudrate))
            self._mount = m
            self._update_state({"mount": {"connected": True, "status": MountStatus.OK, "last_error": None}})
            # Ensure microstep settings are applied on every connect so manual speed is consistent.
            self._mount_set_microsteps(self.cfg.mount.ms_az, self.cfg.mount.ms_alt)
            log_info(self.out_log, f"Mount: connected ({msg})")
        except Exception as exc:
            self._shutdown_mount()
            self._update_state({"mount": {"connected": False, "status": MountStatus.ERROR, "last_error": "connect failed"}})
            log_error(self.out_log, "Mount: connect failed", exc)

    def _mount_stop(self) -> None:
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
        try:
            self._mount.stop()
            self._mount.set_microsteps(int(az_div), int(alt_div))
            self._goto.model.set_microsteps(int(az_div), int(alt_div))
            log_info(self.out_log, f"Mount: MS set (AZ={int(az_div)} ALT={int(alt_div)})")
        except Exception as exc:
            self._update_state(
                {
                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "set microsteps failed"},
                    "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE},
                }
            )
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
    # Raw recording helper
    # -------------------------
    def _start_raw_recording(
        self,
        *,
        duration_s: float,
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

        duration_s = max(0.1, float(duration_s))
        if not basename:
            basename = _dt.datetime.now().strftime("raw_%Y%m%d_%H%M%S")

        def _worker() -> None:
            frames: list[np.ndarray] = []
            try:
                stream = self._cam_stream
                if stream is None:
                    log_info(self.out_log, "Raw record: aborted (camera disconnected)")
                    return

                t0 = _perf()
                last_token: Optional[float] = None

                log_info(self.out_log, f"Raw record: start duration={duration_s:.1f}s")
                while (_perf() - t0) < duration_s and not self._stop.is_set():
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
                    except Exception:
                        raw16 = np.asarray(fr.raw)
                    frames.append(raw16.copy())

                if not frames:
                    log_info(self.out_log, "Raw record: no frames captured")
                    return

                try:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                out_path = os.path.join(out_dir, f"{basename}.npy")
                stack = np.stack(frames, axis=0)
                np.save(out_path, stack)
                log_info(
                    self.out_log,
                    f"Raw record: saved {stack.shape[0]} frames to {out_path} (shape={stack.shape}, dtype={stack.dtype})",
                )
            except Exception as exc:
                log_error(self.out_log, "Raw record: failed", exc)
            finally:
                with self._raw_record_lock:
                    self._raw_record_active = False
                    self._raw_record_thread = None

        self._raw_record_thread = threading.Thread(target=_worker, name="RawRecord", daemon=True)
        self._raw_record_thread.start()


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
                self._update_state({"camera": {"fps_capture": float(st.get("fps_capture", 0.0))}})

            # 2b) tracking
            tracking_on = self._get_tracking_enabled()
            if tracking_on and (self._cam_stream is not None) and (self._mount is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    # Tracking en RAW16 + SEP
                    raw16 = ensure_raw16_bayer(fr.raw)
                    out = tracking_step(
                        self._tracking_state,
                        raw16,
                        now_t=_now_s(),
                        tracking_enabled=bool(tracking_on),
                    )

                    if not self._is_manual_move_active():
                        try:
                            self._mount.rate(float(out.rate_az), float(out.rate_alt))
                        except Exception as exc:
                            self._update_state(
                                {
                                    "mount": {"status": MountStatus.ERROR, "connected": False, "last_error": "tracking rate failed"},
                                    "tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE, "last_error": "mount.rate failed", "n_det": int(out.n_det)},
                                }
                            )
                            log_error(self.out_log, "Tracking: mount.rate failed", exc, throttle_s=2.0, throttle_key="tracking_mount_rate")

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
                                "rate_az": float(out.rate_az),
                                "rate_alt": float(out.rate_alt),
                                "calib_src": str(out.calib_src),
                                "calib_det": float(out.detA),
                                "n_det": int(out.n_det),
                                "last_error": "SEP: no detections" if int(out.n_det) == 0 else None,
                            }
                        }
                    )
            else:
                goto_busy = bool(self.get_state().goto.busy)
                if self._mount is not None and not goto_busy:
                    self._mount_rate_safe(0.0, 0.0)
                self._update_state(
                    {
                        "tracking": {
                            "enabled": False,
                            "status": TrackingStatus.OFF,
                            "mode": TrackingMode.IDLE,
                            "rate_az": 0.0,
                            "rate_alt": 0.0,
                            "n_det": 0,
                            "last_error": None,
                        }
                    }
                )

            # 2c) stacking
            if self._stacking_enabled and (self._cam_stream is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    raw16 = ensure_raw16_bayer(fr.raw)
                    self._stacking.enqueue_frame(raw16.copy(), t=_now_s())

            # 2d) publish stacking metrics
            m = self._stacking.engine.metrics
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
                        "preview_jpeg": self._stacking.engine.get_preview_jpeg(),
                    }
                }
            )

            # 2e) platesolving autosolve scheduling (if enabled)
            self._maybe_autosolve()

            # 3) preview
            self._maybe_update_preview()

            # 4) loop stats
            t1 = _perf()
            frame_ms = (t1 - t0) * 1000.0
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
                if act.type in (ActionType.CAMERA_CONNECT, ActionType.CAMERA_SET_PARAM):
                    self._update_state({"camera": {"status": CameraStatus.ERROR, "connected": False, "last_error": "action failed"}})

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

        if t == ActionType.CAMERA_RECORD_RAW:
            if isinstance(p, dict):
                duration_s = float(p.get("duration_s", 20.0))
                out_dir = str(p.get("out_dir", "raw_output"))
                basename = p.get("basename", None)
                self._start_raw_recording(duration_s=duration_s, out_dir=out_dir, basename=basename)
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
            if not self._tracking_state.auto.ok or self._tracking_state.auto.A_pinv is None:
                auto_reset(self._tracking_state, src="auto")
            self._update_state({"tracking": {"enabled": True, "status": TrackingStatus.RUNNING, "mode": TrackingMode.IDLE}})
            self._mount_rate_safe(0.0, 0.0)
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: START")
            return

        if t == ActionType.TRACKING_STOP:
            self._update_state({"tracking": {"enabled": False, "status": TrackingStatus.OFF, "mode": TrackingMode.IDLE}})
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
            auto_reset(self._tracking_state, src="auto")
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: CALIB_RESET (autocal only)")
            return

        if t == ActionType.TRACKING_AUTO_RESET:
            auto_reset(self._tracking_state, src="auto")
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: AUTO_RESET")
            return

        if t == ActionType.TRACKING_CALIB_AZ:
            log_info(self.out_log, "Tracking: CALIB_AZ ignored (autocalibration only)")
            return

        if t == ActionType.TRACKING_CALIB_ALT:
            log_info(self.out_log, "Tracking: CALIB_ALT ignored (autocalibration only)")
            return

        if t == ActionType.TRACKING_BOOTSTRAP:
            log_info(self.out_log, "Tracking: BOOTSTRAP ignored (autocalibration only)")
            return

        # ---- Stacking ----
        if t == ActionType.STACKING_START:
            self._stacking_enabled = True
            self._stacking.start()
            self._update_state({"stacking": {"enabled": True, "status": StackingStatus.RUNNING}})
            log_info(self.out_log, "Stacking: START")
            return

        if t == ActionType.STACKING_STOP:
            self._stacking_enabled = False
            self._stacking.stop()
            self._update_state({"stacking": {"enabled": False, "status": StackingStatus.OFF}})
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
                if "max_det" in p:
                    self.cfg.platesolving.max_det = int(p.get("max_det"))
                    self._live_sep_params["max_det"] = int(p.get("max_det"))
                self._goto.cfg.sep = self.cfg.sep
                tracking_set_params(
                    self._tracking_state,
                    sep_bw=int(self.cfg.sep.bw),
                    sep_bh=int(self.cfg.sep.bh),
                    sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
                    sep_minarea=int(self.cfg.sep.minarea),
                    sep_max_sources=int(self.cfg.platesolving.max_det),
                )
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
                self._update_state(
                    {
                        "goto": {
                            "synced": False,
                            "status": GotoStatus.FAIL,
                            "reason": "SYNC_NO_SOLUTION",
                        }
                    }
                )
                return
            ok = False
            try:
                ok = bool(self._goto.sync_from_platesolving(sol))
            except Exception as exc:
                log_error(self.out_log, 'GoTo: sync exception', exc)
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

        if t == ActionType.GOTO_FIT_MODEL:
            params = p.get('params', {})
            self._goto_worker.request(kind='fit_model', target=None, params=params)
            return

        if t == ActionType.GOTO_CANCEL:
            self._goto_worker.cancel()
            self._mount_stop()
            self._update_state({"goto": {"busy": False, "status": GotoStatus.CANCELLED, "reason": "CANCELLED"}})
            return

        # ---- Otros ----        # ---- Otros ----
        log_info(self.out_log, f"Unknown or unhandled action type: {t}")
