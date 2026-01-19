# app_runner.py
from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Any, Dict, List

from ap_types import SystemState, Axis
from config import AppConfig, CameraConfig, PreviewConfig
from actions import Action, ActionType
from logging_utils import log_info, log_error

from camera_poa import POACameraDevice, CameraStream
from imaging import make_preview_jpeg
from mount_arduino import ArduinoMount

from tracking import make_tracking_state, tracking_step, tracking_set_params
from stacking import StackingWorker

# Platesolve (nuevo)
# Se asume que platesolve.py está al mismo nivel que app_runner.py
from platesolve import (
    PlatesolveConfig,
    ObserverConfig,
    platesolve_from_live,
    platesolve_from_stack,
    save_gaia_auth,
    load_gaia_auth,
)


def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()


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
        self.cfg = cfg
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
        self._stacking = StackingWorker(cfg)
        self._stacking_enabled = bool(cfg.stacking.enabled_init)

        # Platesolve subsystem (thread dedicado)
        self._platesolve_lock = threading.Lock()
        self._platesolve_thr: Optional[threading.Thread] = None
        self._platesolve_cancel = threading.Event()
        self._platesolve_pending: Optional[Dict[str, Any]] = None

        # Config platesolve (se puede setear desde UI por action)
        self._platesolve_cfg = self._build_default_platesolve_config(cfg)
        self._platesolve_observer = ObserverConfig()  # Santiago por default en tu platesolve.py

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

        # Cache de parámetros “pendientes”
        self._pending_camera_cfg: CameraConfig = cfg.camera
        self._pending_preview_cfg: PreviewConfig = cfg.preview

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
            platesolve_center_ra_deg=0.0,
            platesolve_center_dec_deg=0.0,
        )

    # -------------------------
    # Platesolve config default
    # -------------------------
    def _build_default_platesolve_config(self, cfg: AppConfig) -> PlatesolveConfig:
        """
        Construye un PlatesolveConfig razonable.

        Nota: aquí NO invento campos de AppConfig. Si en tu AppConfig existe
        cfg.platesolve.<...>, puedes mapearlos aquí. Si no, quedan defaults.

        Lo único realmente crítico es pixel_size_m y focal_m.
        Si no están en cfg, deja valores conservadores y los seteas desde UI.
        """
        # Valores por defecto: AJÚSTALOS según tu hardware real o setéalos por UI.
        pixel_size_m = getattr(getattr(cfg, "platesolve", None), "pixel_size_m", None)
        focal_m = getattr(getattr(cfg, "platesolve", None), "focal_m", None)

        if pixel_size_m is None:
            # Ejemplo típico: 2.9 um
            pixel_size_m = 2.9e-6
        if focal_m is None:
            # Ejemplo típico: 900 mm
            focal_m = 0.9

        return PlatesolveConfig(
            pixel_size_m=float(pixel_size_m),
            focal_m=float(focal_m),
            # resto se queda en defaults del dataclass
        )

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

        thr = self._thr
        if thr is not None:
            thr.join(timeout=2.0)
        self._thr = None

        self._shutdown_camera()
        self._shutdown_mount()
        try:
            self._stacking.stop()
        except Exception:
            pass

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
        except Exception:
            pass

    def _mount_rate_safe(self, az: float, alt: float) -> None:
        if self._mount is None:
            return
        try:
            self._mount.rate(float(az), float(alt))
        except Exception:
            pass

    # -------------------------
    # Camera
    # -------------------------
    def _shutdown_camera(self) -> None:
        if self._cam_stream is not None:
            try:
                self._cam_stream.stop()
            except Exception:
                pass
            self._cam_stream = None

        if self._cam_dev is not None:
            try:
                self._cam_dev.close()
            except Exception:
                pass
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
            stream.start(dev, self._pending_camera_cfg, self._pending_preview_cfg)

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
            self._pending_camera_cfg.exp_ms = float(value)
        elif n in ("gain",):
            self._pending_camera_cfg.gain = int(value)
        elif n in ("auto_gain",):
            self._pending_camera_cfg.auto_gain = bool(value)
        elif n in ("img_format",):
            self._pending_camera_cfg.img_format = str(value)
        elif n in ("use_roi",):
            self._pending_camera_cfg.use_roi = bool(value)
        elif n in ("roi_x",):
            self._pending_camera_cfg.roi_x = int(value)
        elif n in ("roi_y",):
            self._pending_camera_cfg.roi_y = int(value)
        elif n in ("roi_w",):
            self._pending_camera_cfg.roi_w = int(value)
        elif n in ("roi_h",):
            self._pending_camera_cfg.roi_h = int(value)
        elif n in ("binning", "bin_hw"):
            self._pending_camera_cfg.binning = int(value)
        elif n in ("preview_view_hz",):
            self._pending_preview_cfg.view_hz = float(value)
        elif n in ("preview_ds",):
            self._pending_preview_cfg.ds = int(value)
        elif n in ("preview_jpeg_quality",):
            self._pending_preview_cfg.jpeg_quality = int(value)
        elif n in ("preview_stretch_plo",):
            self._pending_preview_cfg.stretch_plo = float(value)
        elif n in ("preview_stretch_phi",):
            self._pending_preview_cfg.stretch_phi = float(value)
        else:
            log_info(self.out_log, f"Camera: param ignorado (no soportado aún): {n}={value}")
            return

        if self._cam_dev is not None and self._cam_stream is not None:
            try:
                cam_index = int(self._pending_camera_cfg.camera_index)
                log_info(self.out_log, f"Camera: reconfigure (restart stream) due to {n} change")
                self._connect_camera(cam_index)
            except Exception as exc:
                self._set_state_safe(camera_status="ERR")
                log_error(self.out_log, "Camera: failed to apply param (restart)", exc)

    def _maybe_update_preview(self) -> None:
        if self._cam_stream is None:
            return

        view_hz = float(self._pending_preview_cfg.view_hz)
        if view_hz <= 0.1:
            view_hz = 0.1

        now = _perf()
        if (now - self._t_last_preview) < (1.0 / view_hz):
            return

        fr = self._cam_stream.latest()
        if fr is None:
            return

        try:
            u = fr.u8_view

            if u.ndim == 3 and u.shape[2] == 3:
                ds = int(self._pending_preview_cfg.ds)
                u2 = u[::ds, ::ds, :] if ds > 1 else u
                from imaging import encode_jpeg  # evita ciclos
                jpg = encode_jpeg(u2, quality=int(self._pending_preview_cfg.jpeg_quality))
            else:
                jpg, _ = make_preview_jpeg(
                    u,
                    ds=int(self._pending_preview_cfg.ds),
                    plo=float(self._pending_preview_cfg.stretch_plo),
                    phi=float(self._pending_preview_cfg.stretch_phi),
                    jpeg_quality=int(self._pending_preview_cfg.jpeg_quality),
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

    # -------------------------
    # Mount
    # -------------------------
    def _shutdown_mount(self) -> None:
        if self._mount is not None:
            try:
                self._mount.disconnect()
            except Exception:
                pass
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
            self._set_state_safe(mount_status="ERR", mount_connected=False)
            log_error(self.out_log, "Mount: STOP failed", exc)

    def _mount_set_microsteps(self, az_div: int, alt_div: int) -> None:
        if self._mount is None or not self._mount.is_connected():
            return
        try:
            self._mount.stop()
            self._mount.set_microsteps(int(az_div), int(alt_div))
            log_info(self.out_log, f"Mount: MS set (AZ={int(az_div)} ALT={int(alt_div)})")
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False)
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
        except Exception as exc:
            self._set_state_safe(mount_status="ERR", mount_connected=False)
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

    def _platesolve_worker(self) -> None:
        """
        Worker que ejecuta plate solving sin bloquear el loop principal.
        Toma requests desde self._platesolve_pending (la última gana).
        """
        while not self._stop.is_set() and not self._platesolve_cancel.is_set():
            req: Optional[Dict[str, Any]] = None
            with self._platesolve_lock:
                req = self._platesolve_pending
                self._platesolve_pending = None

            if req is None:
                time.sleep(0.05)
                continue

            # Marcar busy
            self._set_state_safe(platesolve_busy=True, platesolve_status="RUNNING")

            try:
                source = str(req.get("source", "live")).lower().strip()
                target = req.get("target", None)

                if target is None:
                    self._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_TARGET",
                        platesolve_last_ok=False,
                    )
                    continue

                if source == "stack":
                    stack = self._stacking.engine.get_latest_stack()  # debes exponerlo si no existe
                    if stack is None:
                        self._set_state_safe(
                            platesolve_busy=False,
                            platesolve_status="ERR_NO_STACK",
                            platesolve_last_ok=False,
                        )
                        continue
                    frame = stack
                    result = platesolve_from_stack(
                        frame,
                        target=target,
                        cfg=self._platesolve_cfg,
                        observer=self._platesolve_observer,
                        progress_cb=None,
                    )
                else:
                    if self._cam_stream is None:
                        self._set_state_safe(
                            platesolve_busy=False,
                            platesolve_status="ERR_NO_CAMERA",
                            platesolve_last_ok=False,
                        )
                        continue
                    fr = self._cam_stream.latest()
                    if fr is None:
                        self._set_state_safe(
                            platesolve_busy=False,
                            platesolve_status="ERR_NO_FRAME",
                            platesolve_last_ok=False,
                        )
                        continue

                    # Elegimos u8_view o u16; platesolve soporta RGB/BGR/gray -> convierte a gray
                    frame = fr.u8_view
                    result = platesolve_from_live(
                        frame,
                        target=target,
                        cfg=self._platesolve_cfg,
                        observer=self._platesolve_observer,
                        progress_cb=None,
                    )

                # Publicar resultado (si existen campos)
                self._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status=result.status,
                    platesolve_last_ok=bool(result.success),
                    platesolve_theta_deg=float(result.theta_deg),
                    platesolve_dx_px=float(result.dx_px),
                    platesolve_dy_px=float(result.dy_px),
                    platesolve_resp=float(result.response),
                    platesolve_n_inliers=int(result.n_inliers),
                    platesolve_rms_px=float(result.rms_px),
                    platesolve_overlay=list(result.overlay),
                    platesolve_guides=list(result.guides),
                    platesolve_center_ra_deg=float(result.center_ra_deg),
                    platesolve_center_dec_deg=float(result.center_dec_deg),
                )

            except Exception as exc:
                self._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status="ERR_EXCEPTION",
                    platesolve_last_ok=False,
                )
                log_error(self.out_log, "Platesolve: failed", exc)

    def _platesolve_request(self, *, source: str, target: Any) -> None:
        """
        Encola un request para platesolve. Si hay uno pendiente, se reemplaza.
        """
        with self._platesolve_lock:
            self._platesolve_pending = {"source": str(source), "target": target}
        self._platesolve_start_worker_if_needed()

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
                    if hasattr(fr, "u16") and fr.u16 is not None:
                        frame_u16 = fr.u16
                    else:
                        u8 = fr.u8_view
                        frame_u16 = (u8.astype("uint16") * 257)

                    try:
                        out = tracking_step(
                            self._tracking_state,
                            frame_u16,
                            now_t=_now_s(),
                            tracking_enabled=True,
                        )
                    except TypeError:
                        out = tracking_step(self._tracking_state, frame_u16)

                    try:
                        self._mount.rate(float(out.rate_az), float(out.rate_alt))
                    except Exception as exc:
                        log_error(self.out_log, "Tracking: mount.rate failed", exc)

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
                    if getattr(fr, "raw", None) is not None:
                        raw = fr.raw
                        fmt = str(fr.fmt or "RAW16").upper()
                        if "RAW" in fmt and "16" in fmt:
                            fmt = "RAW16"
                        elif "RAW" in fmt and "8" in fmt:
                            fmt = "RAW8"
                        elif "RGB" in fmt:
                            fmt = "RGB24"
                        elif "MONO" in fmt and "8" in fmt:
                            fmt = "MONO8"
                        elif "MONO" in fmt and "16" in fmt:
                            fmt = "MONO16"
                    else:
                        u8 = fr.u8_view
                        raw = (u8.astype("uint16") * 257)
                        fmt = "MONO16"

                    self._stacking.enqueue_frame(raw.copy(), fmt=fmt, t=_now_s())

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

                log_error(self.out_log, f"Action failed: {act.type}", exc)

    def _handle_action(self, act: Action) -> None:
        t = act.type
        p = act.payload

        # ---- Camera ----
        if t == ActionType.CAMERA_CONNECT:
            idx = int(p.get("camera_index", 0))
            self._pending_camera_cfg.camera_index = idx
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

        # ---- Platesolve ----
        if t == ActionType.PLATESOLVE_SET_PARAMS:
            # Permite actualizar PlatesolveConfig desde UI sin reimportar
            # Ej: {'pixel_size_m': 2.9e-6, 'focal_m': 0.9, 'gmax': 14.5, ...}
            if isinstance(p, dict):
                # Rebuild dataclass con campos existentes
                d = dict(self._platesolve_cfg.__dict__)
                d.update(p)
                self._platesolve_cfg = PlatesolveConfig(**d)
                log_info(self.out_log, f"Platesolve: SET_PARAMS {list(p.keys())}")
            return

        if t == ActionType.PLATESOLVE_RUN:
            # Payload esperado:
            #  - source: "live" o "stack"
            #  - target: str|tuple|dict (ver platesolve.py)
            #  - (opcional) gaia_username / gaia_password (persistir)
            source = str(p.get("source", "live"))
            target = p.get("target", None)

            # Si vienen credenciales, persistirlas
            user = str(p.get("gaia_username", "")).strip()
            pw = str(p.get("gaia_password", "")).strip()
            if user and pw:
                save_gaia_auth(user, pw)
                log_info(self.out_log, "Platesolve: Gaia credentials saved")

            self._platesolve_request(source=source, target=target)
            log_info(self.out_log, f"Platesolve: RUN source={source}")
            return

        # ---- Otros ----
        log_info(self.out_log, f"Unknown or unhandled action type: {t}")