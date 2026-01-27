# app_runner.py
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
import json
import re
import datetime as _dt

from dataclasses import replace
from typing import Optional, Any, Dict, List

import cv2
import numpy as np

from ap_types import SystemState, Axis, Frame
from config import AppConfig
from actions import Action, ActionType
from logging_utils import log_info, log_error

from camera_poa import POACameraDevice, CameraStream
from imaging import ensure_raw16_bayer
from preview import make_preview_jpeg, encode_jpeg, stretch_to_u8
from mount_arduino import ArduinoMount

from tracking import make_tracking_state, tracking_step, tracking_set_params
from stacking import StackingWorker

from hotpixels import (
    apply_hotpixel_correction,
    build_hotpixel_mask_temporal,
    load_hotpixel_mask,
    save_hotpixel_mask,
)
from sep_utils import sep_detect_from_raw16

from platesolve import (
    PlatesolveConfig,
    ObserverConfig,
    platesolve_from_frame,
    save_gaia_auth,
    load_gaia_auth,
)

from goto import GoToController, GoToConfig, GoToModel, MountKinematics


def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()



def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:80] if s else "target"


def _apply_hotpix_if_available(
    raw16: np.ndarray,
    *,
    hotpix_mask: Optional[np.ndarray],
    bayer_pattern: str,
) -> np.ndarray:
    if hotpix_mask is None:
        return raw16
    if hotpix_mask.shape != raw16.shape:
        raise ValueError(f"Hotpixel mask shape {hotpix_mask.shape} does not match frame shape {raw16.shape}.")
    return apply_hotpixel_correction(raw16, hotpix_mask, bayer_pattern)


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
        self.cfg.hotpixels = replace(cfg.hotpixels)
        self.cfg.platesolve = replace(cfg.platesolve)
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

        # Platesolve subsystem (thread dedicado)
        self._platesolve_lock = threading.Lock()
        self._platesolve_cfg_lock = threading.Lock()
        self._platesolve_thr: Optional[threading.Thread] = None
        self._platesolve_cancel = threading.Event()
        self._platesolve_pending: Optional[Dict[str, Any]] = None
        self._platesolve_last_auto_t = 0.0
        self._platesolve_auto_target: str = ""

        # Hotpixel calibration (thread dedicado)
        self._hotpix_lock = threading.Lock()
        self._hotpix_thr: Optional[threading.Thread] = None
        self._hotpix_cancel = threading.Event()
        self._hotpix_mask: Optional[np.ndarray] = None
        self._hotpix_meta: Optional[Dict[str, Any]] = None
        self._load_hotpix_mask_if_available(self.cfg.hotpixels.mask_path_base)

        # Config platesolve (runtime copy, actualizable desde UI por action)
        self._platesolve_observer = ObserverConfig()  # Santiago por default en tu platesolve.py

        # GoTo subsystem (no bloquea loop)
        kin = MountKinematics(
            motor_full_steps_per_rev=200,
            microsteps_az=int(self.cfg.mount.ms_az),
            microsteps_alt=int(self.cfg.mount.ms_alt),
            motor_pulley_teeth=20,
            ring_radius_m_az=0.24,
            ring_radius_m_alt=0.235,
        )
        self._goto = GoToController(cfg=GoToConfig(observer=self._platesolve_observer, sep=self.cfg.sep), model=GoToModel(kin=kin))
        self._goto_lock = threading.Lock()
        self._goto_thr: Optional[threading.Thread] = None
        self._goto_cancel = threading.Event()
        self._goto_pending: Optional[Dict[str, Any]] = None
        self._last_platesolve_result: Optional[Any] = None

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
            "max_det": int(self.cfg.platesolve.max_det),
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

        # Platesolve fields (si existen)
        self._set_state_safe(
            platesolve_status="IDLE",
            platesolve_busy=False,
            platesolve_last_ok=False,
            platesolve_theta_deg=0.0,
            platesolve_dx_px=0.0,
            platesolve_dy_px=0.0,
            platesolve_resp=0.0,
            platesolve_n_inliers=0,
            platesolve_rms_px=0.0,
            platesolve_overlay=[],
            platesolve_guides=[],
            platesolve_debug_jpeg=None,
            platesolve_debug_info=None,
            platesolve_center_ra_deg=0.0,
            platesolve_center_dec_deg=0.0,
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
    # Platesolve config copy
    # -------------------------
    def _copy_platesolve_config(self, cfg: PlatesolveConfig) -> PlatesolveConfig:
        """
        Devuelve una copia de PlatesolveConfig para evitar aliasing con defaults.
        """
        return replace(cfg)

    def _get_platesolve_cfg_snapshot(self) -> PlatesolveConfig:
        with self._platesolve_cfg_lock:
            return self._copy_platesolve_config(self.cfg.platesolve)

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

        # detener platesolve thread si existe
        self._platesolve_cancel.set()
        thr_ps = None
        with self._platesolve_lock:
            thr_ps = self._platesolve_thr
        if thr_ps is not None:
            thr_ps.join(timeout=2.0)
        with self._platesolve_lock:
            self._platesolve_thr = None
            self._platesolve_pending = None

        # detener hotpixel calibration thread si existe
        self._hotpix_cancel.set()
        thr_hp = None
        with self._hotpix_lock:
            thr_hp = self._hotpix_thr
        if thr_hp is not None:
            thr_hp.join(timeout=2.0)
        with self._hotpix_lock:
            self._hotpix_thr = None

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
            self._tracking_state.key_reg = "PENDING"
        except Exception as exc:
            log_error(self.out_log, "Tracking: failed to reset keyframe", exc)

    def _mount_rate_safe(self, az: float, alt: float) -> None:
        if self._mount is None:
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
        with self._goto_lock:
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

    def _reset_hotpixels_defaults(self) -> None:
        self.cfg.hotpixels = replace(self.default_cfg.hotpixels)
        self._load_hotpix_mask_if_available(self.cfg.hotpixels.mask_path_base)

    def _reset_platesolve_defaults(self) -> None:
        with self._platesolve_cfg_lock:
            self.cfg.platesolve = replace(self.default_cfg.platesolve)
        self._live_sep_params = {
            "sep_bw": int(self.cfg.sep.bw),
            "sep_bh": int(self.cfg.sep.bh),
            "sep_thresh_sigma": float(self.cfg.sep.thresh_sigma),
            "sep_minarea": int(self.cfg.sep.minarea),
            "max_det": int(self.cfg.platesolve.max_det),
        }

    def _load_hotpix_mask_if_available(self, path_base: str) -> None:
        try:
            mask, meta = load_hotpixel_mask(path_base)
        except FileNotFoundError:
            self._hotpix_mask = None
            self._hotpix_meta = None
            return
        except Exception as exc:
            log_error(self.out_log, "Hotpix: failed to load mask", exc, throttle_s=5.0, throttle_key="hotpix_load")
            self._hotpix_mask = None
            self._hotpix_meta = None
            return

        if mask.ndim != 2:
            log_info(self.out_log, f"Hotpix: invalid mask shape {mask.shape}")
            self._hotpix_mask = None
            self._hotpix_meta = None
            return

        self._hotpix_mask = mask
        self._hotpix_meta = meta

    def _apply_hotpix(self, raw16: np.ndarray, bayer_pattern: str) -> np.ndarray:
        try:
            return _apply_hotpix_if_available(raw16, hotpix_mask=self._hotpix_mask, bayer_pattern=bayer_pattern)
        except ValueError as exc:
            log_error(self.out_log, "Hotpix: mask mismatch; disabling hotpixel correction", exc, throttle_s=5.0, throttle_key="hotpix_apply")
            self._hotpix_mask = None
            self._hotpix_meta = None
            return raw16

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
            meta = getattr(fr, "meta", {}) or {}
            meta_dict = meta if isinstance(meta, dict) else {}
            bayer = meta_dict.get("bayer_pattern", "RGGB")
            raw16_hp = self._apply_hotpix(raw16, bayer)

            if overlay_enabled:
                _, u8_preview = make_preview_jpeg(
                    raw16_hp,
                    plo=float(self.cfg.preview.stretch_plo),
                    phi=float(self.cfg.preview.stretch_phi),
                    jpeg_quality=int(self.cfg.preview.jpeg_quality),
                    sample_stride=4,
                )
                u8_preview = self._apply_live_sep_overlay(raw16_hp, u8_preview)
                jpg = encode_jpeg(u8_preview, quality=int(self.cfg.preview.jpeg_quality))
            else:
                jpg, _ = make_preview_jpeg(
                    raw16_hp,
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

    def _apply_live_sep_overlay(self, raw16_hp: np.ndarray, u8_preview: np.ndarray) -> np.ndarray:
        try:
            params = dict(self._live_sep_params)
            _, _, _, obj_xy = sep_detect_from_raw16(
                raw16_hp,
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
        try:
            self._mount.stop()
            self._mount.move_steps(
                axis=axis,
                direction=int(direction),
                steps=int(steps),
                delay_us=int(delay_us),
            )
            self._goto.model.note_manual_move(axis, int(direction), int(steps))
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False, tracking_enabled=False, tracking_mode="IDLE")
            log_error(self.out_log, "Mount: MOVE steps failed", exc)

    # -------------------------
    # Platesolve (thread worker)
    # -------------------------
    def _platesolve_start_worker_if_needed(self) -> None:
        with self._platesolve_lock:
            if self._platesolve_thr is not None and self._platesolve_thr.is_alive():
                return
            self._platesolve_cancel.clear()
            self._platesolve_thr = threading.Thread(
                target=self._platesolve_worker,
                name="PlatesolveWorker",
                daemon=True,
            )
            self._platesolve_thr.start()

    def _render_platesolve_debug_jpeg(
        self,
        frame: Optional[np.ndarray],
        overlay: Optional[List[Any]],
    ) -> Optional[bytes]:
        if frame is None:
            return None
        gray = frame
        if getattr(gray, "ndim", 0) == 3:
            if gray.shape[2] == 1:
                gray = gray[:, :, 0]
            else:
                gray = gray[:, :, :3].astype(np.float32).mean(axis=2)
        gray = np.asarray(gray, dtype=np.float32)
        if gray.ndim != 2:
            return None

        p1, p99 = np.percentile(gray, [1.0, 99.0])
        if p99 <= p1:
            p1 = float(gray.min()) if gray.size else 0.0
            p99 = float(gray.max()) if gray.size else 1.0
        scale = 255.0 / max(1e-6, float(p99 - p1))
        u8 = np.clip((gray - p1) * scale, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

        if overlay:
            h, w = img.shape[:2]
            colors = {
                "det": (255, 0, 0),
                "match": (0, 255, 0),
                "guide": (0, 0, 255),
            }
            for item in overlay:
                x = int(round(float(getattr(item, "x", 0.0))))
                y = int(round(float(getattr(item, "y", 0.0))))
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                kind = str(getattr(item, "kind", "det"))
                color = colors.get(kind, (255, 255, 0))
                radius = 10 if kind == "guide" else 8 if kind == "match" else 7
                cv2.circle(img, (x, y), radius, color, 1, lineType=cv2.LINE_AA)
                label = getattr(item, "label", None)
                if kind == "guide" and label:
                    cv2.putText(
                        img,
                        str(label),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return bytes(buf.tobytes())

    def _build_platesolve_debug_info(self, result: Any) -> Dict[str, Any]:
        metrics = dict(getattr(result, "metrics", {}) or {})
        info = {
            "status": str(getattr(result, "status", "")),
            "response": float(getattr(result, "response", 0.0)),
            "n_det": metrics.get("n_det"),
            "gaia_rows": metrics.get("gaia_rows"),
            "n_inliers": int(getattr(result, "n_inliers", 0)),
            "rms_px": float(getattr(result, "rms_px", 0.0)),
            "theta_deg": float(getattr(result, "theta_deg", 0.0)),
            "dx_px": float(getattr(result, "dx_px", 0.0)),
            "dy_px": float(getattr(result, "dy_px", 0.0)),
            "radius_deg": metrics.get("radius_deg"),
            "scale_arcsec_per_px": float(getattr(result, "scale_arcsec_per_px", metrics.get("scale_arcsec_per_px", 0.0))),
        }
        return info

    def _platesolve_worker(self) -> None:
        """
        Worker que ejecuta plate solving sin bloquear el loop principal.
        Toma requests desde self._platesolve_pending (la última gana).
    
        Además, en cada solve guarda un "snapshot" reproducible en disco:
          - raw (exacto desde la cámara) + meta + config + target/source
          - u8_view (si existe) para inspección rápida
          - debug_jpeg + debug_info/result para reproducir el diagnóstico
        """
        def _dump_dir() -> Path:
            d = Path("platesolve_dumps")
            d.mkdir(parents=True, exist_ok=True)
            return d
    
        def _dump_snapshot(
            *,
            source: str,
            target: Any,
            raw: np.ndarray,
            fmt: str,
            meta: Dict[str, Any],
            u8_view: Optional[np.ndarray],
            cfg: PlatesolveConfig,
            observer: ObserverConfig,
            extra: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            """
            Guarda:
              - *_raw.npy (exacto)
              - *_u8.npy (opcional)
              - *_meta.json (metadatos + cfg/observer + request)
            Devuelve base path (sin extensión) o None.
            """
            try:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base = _dump_dir() / f"{ts}_{str(source)}_{_safe_slug(str(target))}"
    
                raw_c = np.ascontiguousarray(raw)
                np.save(str(base) + "_raw.npy", raw_c)
    
                if u8_view is not None:
                    np.save(str(base) + "_u8.npy", np.ascontiguousarray(u8_view))
    
                info = {
                    "ts": ts,
                    "source": str(source),
                    "target": target,
                    "fmt": str(fmt),
                    "shape": list(raw_c.shape),
                    "dtype": str(raw_c.dtype),
                    "meta": meta or {},
                    "platesolve_cfg": dict(getattr(cfg, "__dict__", {}) or {}),
                    "observer": dict(getattr(observer, "__dict__", {}) or {}),
                }
                if extra:
                    info["extra"] = extra
    
                with open(str(base) + "_meta.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
    
                return str(base)
            except (OSError, ValueError, TypeError) as exc:
                log_error(self.out_log, "Platesolve: failed to dump snapshot", exc)
                return None
    
        while not self._stop.is_set() and not self._platesolve_cancel.is_set():
            req: Optional[Dict[str, Any]] = None
            with self._platesolve_lock:
                req = self._platesolve_pending
                self._platesolve_pending = None
    
            if req is None:
                time.sleep(0.05)
                continue
    
            # Marcar busy
            self._set_state_safe(
                platesolve_busy=True,
                platesolve_status="RUNNING",
                platesolve_debug_jpeg=None,
                platesolve_debug_info=None,
            )
    
            dump_base: Optional[str] = None
    
            try:
                target = req.get("target", None)
    
                if target is None:
                    self._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_TARGET",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_TARGET"},
                    )
                    continue
    
                # -------------------------
                # Obtener frame + snapshot (solo live)
                # -------------------------
                if self._cam_stream is None:
                    self._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_CAMERA",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_CAMERA"},
                    )
                    continue

                fr = self._cam_stream.latest()
                if fr is None:
                    self._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_FRAME",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_FRAME"},
                    )
                    continue

                # RAW exacto + meta para reproducir mañana
                fmt = str(getattr(fr, "fmt", "") or "RAW16")
                meta = dict(getattr(fr, "meta", {}) or {})
                raw_in = np.ascontiguousarray(fr.raw)
                u8_in = np.ascontiguousarray(fr.u8_view) if getattr(fr, "u8_view", None) is not None else None

                platesolve_cfg = self._get_platesolve_cfg_snapshot()

                dump_base = _dump_snapshot(
                    source="live",
                    target=target,
                    raw=raw_in,
                    fmt=fmt,
                    meta=meta,
                    u8_view=u8_in,
                    cfg=platesolve_cfg,
                    observer=self._platesolve_observer,
                )

                # Frame para solver: RAW16 + hotpixel correction
                bayer = str((meta or {}).get("bayer_pattern", "RGGB"))
                raw16 = ensure_raw16_bayer(raw_in)
                frame = self._apply_hotpix(raw16, bayer)

                # Debug de stats de entrada (opcional)
                debug_stats = bool(getattr(platesolve_cfg, "debug_input_stats", False))

                def _stats(a: np.ndarray, name: str) -> None:
                    if not debug_stats:
                        return
                    a = np.asarray(a)
                    log_info(self.out_log, f"[{name}] shape={a.shape} dtype={a.dtype} C={a.flags['C_CONTIGUOUS']}")
                    if a.size == 0:
                        log_info(self.out_log, "  EMPTY")
                        return
                    if a.ndim == 1:
                        log_info(self.out_log, f"  1D buffer: min={a.min()} max={a.max()} mean={a.mean():.3g}")
                        return
                    flat = a.reshape(-1)
                    p = np.percentile(flat, [0, 1, 5, 50, 95, 99, 100])
                    log_info(self.out_log, f"  min/p1/p5/p50/p95/p99/max = {p}")
                    log_info(self.out_log, f"  mean={flat.mean():.3g} std={flat.std():.3g}")
                    if a.dtype == np.uint16:
                        log_info(self.out_log, f"  sat65535={np.mean(flat == 65535):.4f}")
                    if a.dtype == np.uint8:
                        log_info(self.out_log, f"  sat255={np.mean(flat == 255):.4f}")

                _stats(raw_in, "fr.raw")
                if hasattr(fr, "u8_view") and fr.u8_view is not None:
                    _stats(fr.u8_view, "fr.u8_view")
                _stats(frame, "frame(raw16_hp)")

                result = platesolve_from_frame(
                    frame,
                    target=target,
                    cfg=platesolve_cfg,
                    sep_cfg=self.cfg.sep,
                    observer=self._platesolve_observer,
                    progress_cb=None,
                )
    
                # -------------------------
                # Debug outputs (jpeg + info)
                # -------------------------
                debug_jpeg = self._render_platesolve_debug_jpeg(
                    frame,
                    list(getattr(result, "overlay", []) or []),
                )
                debug_info = self._build_platesolve_debug_info(result)
    
                # Guardar debug outputs junto al snapshot (si existe dump_base)
                if dump_base:
                    try:
                        if debug_jpeg:
                            with open(dump_base + "_debug.jpg", "wb") as f:
                                f.write(debug_jpeg)
                        with open(dump_base + "_result.json", "w", encoding="utf-8") as f:
                            json.dump(debug_info, f, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        log_error(self.out_log, "Platesolve: failed to dump debug outputs", exc)
    
                # Publicar resultado (si existen campos)
                self._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status=getattr(result, "status", "UNKNOWN"),
                    platesolve_last_ok=bool(getattr(result, "success", False)),
                    platesolve_theta_deg=float(getattr(result, "theta_deg", 0.0)),
                    platesolve_dx_px=float(getattr(result, "dx_px", 0.0)),
                    platesolve_dy_px=float(getattr(result, "dy_px", 0.0)),
                    platesolve_resp=float(getattr(result, "response", 0.0)),
                    platesolve_n_inliers=int(getattr(result, "n_inliers", 0)),
                    platesolve_rms_px=float(getattr(result, "rms_px", 0.0)),
                    platesolve_overlay=list(getattr(result, "overlay", []) or []),
                    platesolve_guides=list(getattr(result, "guides", []) or []),
                    platesolve_debug_jpeg=debug_jpeg,
                    platesolve_debug_info=debug_info,
                    platesolve_center_ra_deg=float(getattr(result, "center_ra_deg", 0.0)),
                    platesolve_center_dec_deg=float(getattr(result, "center_dec_deg", 0.0)),
                )
    
                # Cache para GoTo sync/calibrate (solo si OK)
                if bool(getattr(result, "success", False)):
                    self._last_platesolve_result = result
    
            except Exception as exc:
                self._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status="ERR_EXCEPTION",
                    platesolve_last_ok=False,
                )
                log_error(self.out_log, "Platesolve: failed", exc)


    def _platesolve_request(self, *, target: Any) -> None:
        """
        Encola un request para platesolve. Si hay uno pendiente, se reemplaza.
        """
        with self._platesolve_lock:
            self._platesolve_pending = {"target": target}
        self._platesolve_start_worker_if_needed()

    def _maybe_autosolve(self) -> None:
        cfg = self._get_platesolve_cfg_snapshot()
        if not bool(cfg.auto_solve):
            return
        target = str(self._platesolve_auto_target or "").strip()
        if not target:
            return
        st = self.get_state()
        if bool(getattr(st, "platesolve_busy", False)):
            return
        now = _perf()
        if (now - float(self._platesolve_last_auto_t)) < max(2.0, float(cfg.solve_every_s)):
            return
        self._platesolve_request(target=target)
        self._platesolve_last_auto_t = float(now)

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
    # Hotpixel calibration
    # -------------------------
    def _hotpix_start_worker_if_needed(
        self,
        *,
        n_frames: int,
        abs_percentile: float,
        var_percentile: float,
        max_component_area: int,
        out_path_base: str,
    ) -> None:
        with self._hotpix_lock:
            if self._hotpix_thr is not None and self._hotpix_thr.is_alive():
                log_info(self.out_log, "Hotpix: calibration already running")
                return
            self._hotpix_cancel.clear()
            self._hotpix_thr = threading.Thread(
                target=self._hotpix_worker,
                name="hotpix-calibration",
                daemon=True,
                kwargs={
                    "n_frames": n_frames,
                    "abs_percentile": abs_percentile,
                    "var_percentile": var_percentile,
                    "max_component_area": max_component_area,
                    "out_path_base": out_path_base,
                },
            )
            self._hotpix_thr.start()

    def _hotpix_worker(
        self,
        *,
        n_frames: int,
        abs_percentile: float,
        var_percentile: float,
        max_component_area: int,
        out_path_base: str,
    ) -> None:
        if self._cam_stream is None:
            log_info(self.out_log, "Hotpix: camera stream not available")
            return

        frames: List[np.ndarray] = []
        last_seq = -1
        cam_stats = self._cam_stream.stats()
        fps_capture = float(cam_stats.get("fps_capture", 0.0))
        timeout_s = max(5.0, (n_frames / max(1.0, fps_capture)) * 3.0)
        deadline = _perf() + timeout_s
        last_frame: Optional[Frame] = None

        while len(frames) < n_frames and _perf() < deadline:
            if self._stop.is_set() or self._hotpix_cancel.is_set():
                return
            fr = self._cam_stream.latest()
            if fr is None or fr.seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = fr.seq
            last_frame = fr

            raw16 = ensure_raw16_bayer(fr.raw)
            frames.append(raw16.copy())

        if len(frames) < n_frames:
            log_info(self.out_log, f"Hotpix: calibration timed out ({len(frames)}/{n_frames} frames)")
            return

        try:
            mask = build_hotpixel_mask_temporal(
                frames,
                abs_percentile=float(abs_percentile),
                var_percentile=float(var_percentile),
                max_component_area=int(max_component_area),
            )
        except Exception as exc:
            log_error(self.out_log, "Hotpix: failed to build mask", exc)
            return

        cam_cfg = self.cfg.camera
        frame_meta = last_frame.meta if last_frame and last_frame.meta else {}
        roi = frame_meta.get("roi")

        meta = {
            "roi": list(roi) if roi is not None else None,
            "exp_ms": float(getattr(cam_cfg, "exp_ms", 0.0)),
            "gain": int(getattr(cam_cfg, "gain", 0)),
            "fmt": str(getattr(last_frame, "fmt", "")) if last_frame is not None else "",
            "camera_model": frame_meta.get("camera_model"),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "abs_percentile": float(abs_percentile),
            "var_percentile": float(var_percentile),
            "max_component_area": int(max_component_area),
            "n_frames": int(n_frames),
        }

        try:
            npy_path, json_path = save_hotpixel_mask(mask, meta, out_path_base)
            self._hotpix_mask = mask
            self._hotpix_meta = meta
            log_info(
                self.out_log,
                f"Hotpix: mask saved ({npy_path.name}, {json_path.name})",
            )
        except Exception as exc:
            log_error(self.out_log, "Hotpix: failed to save mask", exc)


    # -------------------------
    # GoTo worker
    # -------------------------
    def _goto_start_worker_if_needed(self) -> None:
        if self._goto_thr is not None and self._goto_thr.is_alive():
            return
        self._goto_cancel.clear()
        self._goto_thr = threading.Thread(target=self._goto_worker, name="GoToWorker", daemon=True)
        self._goto_thr.start()

    def _goto_request(self, *, kind: str, target: Any, params: Dict[str, Any]) -> None:
        with self._goto_lock:
            self._goto_pending = {"kind": str(kind), "target": target, "params": dict(params)}
        self._goto_start_worker_if_needed()

    def _goto_worker(self) -> None:
        """Ejecuta GoTo/Calibración en background para no bloquear el control loop."""
        while not self._stop.is_set():
            with self._goto_lock:
                req = self._goto_pending
                self._goto_pending = None

            if req is None:
                time.sleep(0.05)
                continue

            kind = str(req.get("kind", "goto"))
            target = req.get("target", None)
            params = dict(req.get("params", {}) or {})

            # Latch current modes
            was_tracking = bool(getattr(self.get_state(), "tracking_enabled", False))
            was_stacking = bool(self._stacking_enabled)

            # Pause tracking/stacking during operation
            if was_tracking:
                try:
                    self._set_state_safe(tracking_enabled=False)
                    self._mount_rate_safe(0.0, 0.0)
                except Exception as exc:
                    log_error(self.out_log, "GoTo: failed to pause tracking", exc)
            if was_stacking:
                try:
                    self._stacking_enabled = False
                    self._set_state_safe(stacking_enabled=False, stacking_mode="IDLE", stacking_status="OFF", stacking_on=False)
                except Exception as exc:
                    log_error(self.out_log, "GoTo: failed to pause stacking", exc)

            self._set_state_safe(goto_busy=True, goto_status=kind.upper())
            self._goto_cancel.clear()

            def should_cancel() -> bool:
                return self._goto_cancel.is_set() or self._stop.is_set()

            def get_live_frame():
                if self._cam_stream is None:
                    return None
                fr = self._cam_stream.latest()
                if fr is None:
                    return None
                meta = dict(getattr(fr, "meta", {}) or {})
                bayer = str(meta.get("bayer_pattern", "RGGB"))
                raw16 = ensure_raw16_bayer(fr.raw)
                return self._apply_hotpix(raw16, bayer)

            def move_steps(axis: Axis, direction: int, steps: int, delay_us: int):
                if self._mount is None:
                    raise RuntimeError("mount not connected")
                return self._mount.move_steps(axis, direction, steps, delay_us)

            def stop():
                if self._mount is not None:
                    return self._mount.stop()
                return None

            platesolve_cfg = self._get_platesolve_cfg_snapshot()

            try:
                if kind == "goto":
                    # Expect target dict from UI/actions
                    # params may include: delay_us, tol_arcsec, max_iters, max_step_deg, max_step_per_iter, gain
                    delay_us = int(params.get("delay_us", 1800))
                    tol_arcsec = float(params.get("tol_arcsec", 10.0))
                    max_iters = int(params.get("max_iters", 8))
                    gain = float(params.get("gain", 0.9))
                    max_step_per_iter = self._goto.cfg.max_step_per_iter
                    if "max_step_per_iter" in params:
                        max_step_per_iter = int(params.get("max_step_per_iter"))
                    else:
                        max_step_deg = float(params.get("max_step_deg", 5.0))
                        j_matrix = self._goto.model.J_deg_per_step
                        max_abs_deg_per_step = float(np.max(np.abs(j_matrix))) if j_matrix is not None and j_matrix.size else 0.0
                        if max_abs_deg_per_step > 0.0:
                            max_step_per_iter = int(max(1, round(max_step_deg / max_abs_deg_per_step)))

                    self._goto.cfg = replace(
                        self._goto.cfg,
                        tol_arcsec=tol_arcsec,
                        max_iters=max_iters,
                        gain=gain,
                        max_step_per_iter=max_step_per_iter,
                    )

                    ok, last_err = self._goto.goto_blocking(
                        target,
                        get_live_frame=get_live_frame,
                        move_steps=move_steps,
                        stop=stop,
                        platesolve_cfg=platesolve_cfg,
                        delay_us=delay_us,
                        should_cancel=should_cancel,
                        status_cb=lambda s: self._set_state_safe(goto_status=str(s)),
                        error_cb=lambda e: self._set_state_safe(goto_last_error_arcsec=float(e)),
                    )
                    self._set_state_safe(goto_last_error_arcsec=float(last_err), goto_status="GOTO_OK" if ok else "GOTO_ERR")

                elif kind == "calibrate":
                    # Calibrate requires a prior sync/platesolve; we use the current live frame + small dithers
                    if "n_samples" not in params and "samples" in params:
                        params["n_samples"] = params.get("samples")
                    if "step_unit" not in params and "units" in params:
                        params["step_unit"] = params.get("units")
                    if "step_magnitudes" not in params:
                        step_small = params.get("step_small", None)
                        step_big = params.get("step_big", None)
                        if step_small is not None or step_big is not None:
                            mags_legacy = []
                            if step_small is not None:
                                mags_legacy.append(float(step_small))
                            if step_big is not None:
                                mags_legacy.append(float(step_big))
                            params["step_magnitudes"] = mags_legacy

                    delay_us = int(params.get("delay_us", 1800))
                    n_samples = int(params.get("n_samples", 12))
                    unit = str(params.get("step_unit", "deg"))
                    mags = params.get("step_magnitudes") or [1.0, 5.0]

                    self._goto.cfg = replace(
                        self._goto.cfg,
                        slew_delay_us_az=delay_us,
                        slew_delay_us_alt=delay_us,
                    )
                    step_magnitudes_steps = mags if unit == "steps" else None
                    step_magnitudes_deg = mags if unit != "steps" else (1.0, 5.0)

                    self._goto.calibrate_blocking(
                        get_live_frame=get_live_frame,
                        move_steps=move_steps,
                        stop=stop,
                        platesolve_cfg=platesolve_cfg,
                        step_magnitudes_deg=step_magnitudes_deg,
                        step_magnitudes_steps=step_magnitudes_steps,
                        samples_per_mag=n_samples,
                    )
                    self._set_state_safe(goto_status="CAL_OK")

                else:
                    self._set_state_safe(goto_status=f"ERR_KIND_{kind}")

                # Update J in state (best effort)
                try:
                    J = self._goto.model.J_deg_per_step
                    self._set_state_safe(
                        goto_J00=float(J[0, 0]),
                        goto_J01=float(J[0, 1]),
                        goto_J10=float(J[1, 0]),
                        goto_J11=float(J[1, 1]),
                        goto_synced=bool(getattr(self._goto.model, "synced", False)),
                    )
                except Exception as exc:
                    log_error(self.out_log, "GoTo: failed to update J matrix", exc)

            except Exception as exc:
                log_error(self.out_log, f"GoTo worker failed ({kind})", exc)
                self._set_state_safe(goto_status="ERR_EXCEPTION")

            finally:
                # Restore stacking/tracking
                if was_stacking:
                    try:
                        self._stacking_enabled = True
                        self._stacking.start()
                        self._set_state_safe(stacking_enabled=True, stacking_mode="RUNNING", stacking_status="ON", stacking_on=True)
                    except Exception as exc:
                        log_error(self.out_log, "GoTo: failed to resume stacking", exc)

                if was_tracking:
                    try:
                        self._set_state_safe(tracking_enabled=True)
                        self._tracking_keyframe_reset()
                    except Exception as exc:
                        log_error(self.out_log, "GoTo: failed to resume tracking", exc)

                self._set_state_safe(goto_busy=False)

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
                    # Tracking en RAW16 (hotpixel-corrected) + SEP
                    meta = dict(getattr(fr, "meta", {}) or {})
                    bayer = str(meta.get("bayer_pattern", "RGGB"))
                    raw16 = ensure_raw16_bayer(fr.raw)
                    raw16_hp = self._apply_hotpix(raw16, bayer)
                    img_det, _, _, _ = sep_detect_from_raw16(
                        raw16_hp,
                        sep_bw=int(self.cfg.sep.bw),
                        sep_bh=int(self.cfg.sep.bh),
                        sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
                        sep_minarea=int(self.cfg.sep.minarea),
                        max_sources=None,
                    )

                    try:
                        out = tracking_step(
                            self._tracking_state,
                            img_det,
                            now_t=_now_s(),
                            tracking_enabled=True,
                        )
                    except TypeError as exc:
                        log_error(self.out_log, "Tracking: falling back to legacy tracking_step signature", exc, throttle_s=60.0, throttle_key="tracking_step_signature")
                        out = tracking_step(self._tracking_state, img_det)

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
                self._set_state_safe(
                    tracking_mode="IDLE",
                    tracking_rate_az=0.0,
                    tracking_rate_alt=0.0,
                )

            # 2c) stacking
            if self._stacking_enabled and (self._cam_stream is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    meta = dict(getattr(fr, "meta", {}) or {})
                    bayer = str(meta.get("bayer_pattern", "RGGB"))
                    raw16 = ensure_raw16_bayer(fr.raw)
                    raw16_hp = self._apply_hotpix(raw16, bayer)
                    self._stacking.enqueue_frame(raw16_hp.copy(), t=_now_s())

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

            # 2e) platesolve autosolve scheduling (if enabled)
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
                    ActionType.MOUNT_NUDGE,
                    ActionType.MOUNT_START_CONTINUOUS,
                    ActionType.MOUNT_STOP,
                    ActionType.MOUNT_SET_MICROSTEPS,
                    ActionType.MOUNT_MOVE_STEPS,
                    ActionType.TRACKING_START,
                    ActionType.TRACKING_STOP,
                    ActionType.TRACKING_SET_PARAMS,
                    ActionType.STACKING_START,
                    ActionType.STACKING_STOP,
                    ActionType.STACKING_RESET,
                    ActionType.STACKING_SET_PARAMS,
                    ActionType.PLATESOLVE_RUN,
                    ActionType.PLATESOLVE_SET_PARAMS,
                ):
                    if act.type in (
                        ActionType.MOUNT_CONNECT,
                        ActionType.MOUNT_NUDGE,
                        ActionType.MOUNT_START_CONTINUOUS,
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
                log_info(self.out_log, f"Tracking: SET_PARAMS {list(p.keys())}")
            return

        if t == ActionType.RESET_TRACKING_DEFAULTS:
            self._reset_tracking_defaults()
            log_info(self.out_log, "Tracking: RESET_DEFAULTS")
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

        if t == ActionType.HOTPIX_CALIBRATE:
            if self._cam_stream is None:
                log_info(self.out_log, "Hotpix: calibration skipped (camera stream inactive)")
                return
            if self._get_tracking_enabled():
                self._set_state_safe(tracking_enabled=False, tracking_mode="IDLE")
                self._mount_rate_safe(0.0, 0.0)
            n_frames = int(p.get("n_frames", self.cfg.hotpixels.calib_frames))
            abs_percentile = float(p.get("abs_percentile", self.cfg.hotpixels.calib_abs_percentile))
            var_percentile = float(p.get("var_percentile", self.cfg.hotpixels.calib_var_percentile))
            max_component_area = int(p.get("max_component_area", self.cfg.hotpixels.max_component_area))
            out_path_base = str(p.get("out_path_base", self.cfg.hotpixels.mask_path_base))
            self._hotpix_start_worker_if_needed(
                n_frames=n_frames,
                abs_percentile=abs_percentile,
                var_percentile=var_percentile,
                max_component_area=max_component_area,
                out_path_base=out_path_base,
            )
            return

        if t == ActionType.RESET_HOTPIXELS_DEFAULTS:
            self._reset_hotpixels_defaults()
            log_info(self.out_log, "Hotpix: RESET_DEFAULTS")
            return

        # ---- Platesolve ----
        if t == ActionType.PLATESOLVE_SET_PARAMS:
            # Permite actualizar PlatesolveConfig desde UI sin reimportar
            # Ej: {'pixel_size_m': 2.9e-6, 'focal_m': 0.9, 'gmax': 14.5, ...}
            if isinstance(p, dict):
                payload = dict(p)
                if "auto_target" in payload:
                    self._platesolve_auto_target = str(payload.pop("auto_target") or "")

                # Rebuild dataclass con campos existentes
                with self._platesolve_cfg_lock:
                    d = dict(self.cfg.platesolve.__dict__)
                    for k, v in payload.items():
                        if k in d:
                            d[k] = v
                    self.cfg.platesolve = PlatesolveConfig(**d)
                log_info(self.out_log, f"Platesolve: SET_PARAMS {list(payload.keys())}")
            return

        if t == ActionType.RESET_PLATESOLVE_DEFAULTS:
            self._reset_platesolve_defaults()
            log_info(self.out_log, "Platesolve: RESET_DEFAULTS")
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


        if t == ActionType.PLATESOLVE_RUN:
            # Payload esperado:
            #  - target: str|tuple|dict (ver platesolve.py)
            #  - (opcional) gaia_username / gaia_password (persistir)
            target = p.get("target", None)

            # Si vienen credenciales, persistirlas
            user = str(p.get("gaia_username", "")).strip()
            pw = str(p.get("gaia_password", "")).strip()
            if user and pw:
                save_gaia_auth(user, pw)
                log_info(self.out_log, "Platesolve: Gaia credentials saved")

            self._platesolve_request(target=target)
            log_info(self.out_log, "Platesolve: RUN source=live")
            return

        # ---- GoTo ----
        if t == ActionType.MOUNT_SYNC:
            # Sync usando el último platesolve OK
            sol = getattr(self, '_last_platesolve_result', None)
            if sol is None or not bool(getattr(sol, 'success', False)):
                log_info(self.out_log, 'GoTo: sync failed (no successful platesolve cached)')
                self._set_state_safe(goto_synced=False, goto_status='SYNC_ERR')
                return
            ok = False
            try:
                ok = bool(self._goto.sync_from_platesolve(sol))
            except Exception as exc:
                log_error(self.out_log, 'GoTo: sync exception', exc)
            self._set_state_safe(goto_synced=bool(ok), goto_status='SYNC_OK' if ok else 'SYNC_ERR')
            return

        if t == ActionType.MOUNT_GOTO:
            target = p.get('target', {})
            self._goto_request(kind='goto', target=target, params=p)
            return

        if t == ActionType.GOTO_CALIBRATE:
            params = p.get('params', {})
            self._goto_request(kind='calibrate', target=None, params=params)
            return

        if t == ActionType.GOTO_CANCEL:
            self._goto_cancel.set()
            try:
                self._mount_stop()
            except Exception as exc:
                log_error(self.out_log, "GoTo: cancel mount stop failed", exc)
            self._set_state_safe(goto_busy=False, goto_status='CANCELLED')
            return

        # ---- Otros ----        # ---- Otros ----
        log_info(self.out_log, f"Unknown or unhandled action type: {t}")
