# app_runner.py
from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Any

from ap_types import SystemState, Axis
from config import AppConfig, CameraConfig, PreviewConfig
from actions import Action, ActionType
from logging_utils import log_info, log_error

from camera_poa import POACameraDevice, CameraStream
from imaging import make_preview_jpeg

from mount_arduino import ArduinoMount

from tracking import make_tracking_state, tracking_step, tracking_set_params
from stacking import StackingWorker


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
        - aplica acciones (connect, set params, etc.)
        - actualiza SystemState
        - genera preview JPEG a view_hz
        - (NUEVO) ejecuta tracking y envía RATE a la montura si tracking está ON
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

        # Tracking subsystem (NUEVO)
        self._tracking_state = make_tracking_state()  # usa defaults de tracking.py

        # Stacking subsystem (NUEVO)
        self._stacking = StackingWorker(cfg)
        self._stacking_enabled = bool(cfg.stacking.enabled_init)

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

        # Inicializar estado por defecto si existen campos
        self._set_state_safe(camera_status="DISCONNECTED", camera_connected=False)
        self._set_state_safe(mount_status="DISCONNECTED", mount_connected=False)

        # Tracking flags (si SystemState los tiene; si no, no rompe)
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

        self._set_state_safe(
            stacking_enabled=self._stacking_enabled,
            stacking_mode="RUNNING" if self._stacking_enabled else "IDLE",
            stacking_status="ON" if self._stacking_enabled else "OFF",
            stacking_on=self._stacking_enabled,
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
        # Tu tracking.py ya soporta esto: si key_reg es "PENDING", resetea al próximo frame.
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
            # no escalamos a ERR aquí; el loop ya lo verá si persiste
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

            # Política producto: stacking sin binning => fuerza en CameraStream.start()
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
            # Se guarda, pero CameraStream.start() forzará bin=1 para stacking (por ahora)
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

        # aplicar si está conectada (versión simple: reinicio)
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
                if ds > 1:
                    u2 = u[::ds, ::ds, :]
                else:
                    u2 = u
                from imaging import encode_jpeg  # import local para evitar ciclos
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

            # FPS view
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
            # política: para cambiar MS, paramos primero (como en tu script)
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
            # política: antes de MOVE, detener rate (equivalente a RATE 0 0)
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
    # Main loop
    # -------------------------
    def _run(self) -> None:
        dt_target = 1.0 / max(1.0, float(self.cfg.control_hz))

        t_last = _perf()

        while not self._stop.is_set():
            t0 = _perf()

            # 1) Procesar acciones sin bloquear el loop
            self._drain_actions(max_n=50)

            # 2) Actualizar stats de captura (sin bloquear)
            if self._cam_stream is not None:
                st = self._cam_stream.stats()
                self._set_state_safe(fps_capture=float(st.get("fps_capture", 0.0)))

            # 2b) TRACKING (NUEVO): corre solo si tracking_enabled y hay cam+mount
            tracking_on = self._get_tracking_enabled()
            if tracking_on and (self._cam_stream is not None) and (self._mount is not None):
                fr = self._cam_stream.latest()
                if fr is not None:
                    # Preferimos u16 si existe; fallback: elevar u8 -> u16
                    if hasattr(fr, "u16") and fr.u16 is not None:
                        frame_u16 = fr.u16
                    else:
                        u8 = fr.u8_view
                        frame_u16 = (u8.astype("uint16") * 257)

                    try:
                        out = tracking_step(
                            self._tracking_state,
                            frame_u16,
                            now_t=_now_s(),            # IMPORTANTE: tu tracking.py usa now_t
                            tracking_enabled=True,
                        )
                    except TypeError:
                        # En caso de mismatch por versión (defensivo)
                        out = tracking_step(self._tracking_state, frame_u16)

                    # Enviar RATE resultante (si tracking_step no tiene calib, rate=0)
                    try:
                        self._mount.rate(float(out.rate_az), float(out.rate_alt))
                    except Exception as exc:
                        log_error(self.out_log, "Tracking: mount.rate failed", exc)

                    # Publicar métricas (solo si SystemState tiene campos)
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
                # Tracking OFF (o faltan subsystems): asegurar RATE 0 0 (evita drift)
                if self._mount is not None:
                    self._mount_rate_safe(0.0, 0.0)
                self._set_state_safe(
                    tracking_mode="IDLE",
                    tracking_rate_az=0.0,
                    tracking_rate_alt=0.0,
                )

            # 2c) STACKING: enqueue frame for worker (non-blocking)
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

            # 2d) Publish stacking metrics to SystemState
            m = self._stacking.engine.metrics
            self._set_state_safe(
                stacking_enabled=bool(m.enabled),
                stacking_mode="RUNNING" if m.enabled else "IDLE",
                stacking_status="ON" if m.enabled else "OFF",
                stacking_on=bool(m.enabled),
                stacking_fps=float(m.stacking_fps),
                stacking_tiles_used=int(m.tiles_used),
                stacking_tiles_evicted=int(m.tiles_evicted),
                stacking_frames_in=int(m.frames_in),
                stacking_frames_used=int(m.frames_used),
                stacking_frames_dropped=int(m.frames_dropped),
                stacking_frames_rejected=int(m.frames_rejected),
                stacking_last_resp=float(m.last_resp),
                stacking_last_dx=float(m.last_dx),
                stacking_last_dy=float(m.last_dy),
                stacking_last_theta_deg=float(m.last_theta_deg),
                stacking_preview_jpeg=self._stacking.engine.get_preview_jpeg(),
            )

            # 3) Actualizar preview (throttled)
            self._maybe_update_preview()

            # 4) Stats del loop
            t1 = _perf()
            frame_ms = (t1 - t0) * 1000.0
            self._set_state_safe(frame_ms=float(frame_ms))

            self._n_loop += 1
            if (t1 - self._t_fps_loop0) >= 1.0:
                fps_loop = self._n_loop / (t1 - self._t_fps_loop0)
                self._t_fps_loop0 = t1
                self._n_loop = 0
                self._set_state_safe(fps_control_loop=float(fps_loop))

            # 5) Sleep para mantener dt_target
            now = _perf()
            elapsed = now - t_last
            t_last = now

            slack = dt_target - elapsed
            if slack > 0:
                time.sleep(slack)

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
                    # NUEVO
                    ActionType.MOUNT_SET_MICROSTEPS,
                    ActionType.MOUNT_MOVE_STEPS,
                    # tracking
                    ActionType.TRACKING_START,
                    ActionType.TRACKING_STOP,
                    ActionType.TRACKING_SET_PARAMS,
                ):
                    # mount err solo si aplica; tracking actions no deberían marcar mount err
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
            # Igual que tu script antiguo: STOP implica que el próximo frame debe fijar keyframe nuevo.
            self._tracking_keyframe_reset()
            return

        # NUEVO: MS
        if t == ActionType.MOUNT_SET_MICROSTEPS:
            az_div = int(p.get("az_div", 64))
            alt_div = int(p.get("alt_div", 64))
            self._mount_set_microsteps(az_div, alt_div)
            return

        # NUEVO: MOVE steps
        if t == ActionType.MOUNT_MOVE_STEPS:
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            steps = int(p.get("steps", 600))
            delay_us = int(p.get("delay_us", 1800))
            self._mount_move_steps(axis, direction, steps, delay_us)

            # IMPORTANTE: después de un MOVE manual, resetea keyframe para no “pelear” con el salto.
            self._tracking_keyframe_reset()
            return

        # (legacy / compat)
        if t == ActionType.MOUNT_NUDGE:
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            rate = float(p.get("rate", 0.0))
            duration_ms = int(p.get("duration_ms", 250))
            self._mount_nudge(axis, direction, rate, duration_ms)

            # Nudge también debe resetear keyframe (mismo motivo)
            self._tracking_keyframe_reset()
            return

        if t == ActionType.MOUNT_START_CONTINUOUS:
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            rate = float(p.get("rate", 0.0))
            self._mount_start_continuous(axis, direction, rate)

            # empezar movimiento continuo invalida keyframe
            self._tracking_keyframe_reset()
            return

        # ---- Tracking (YA IMPLEMENTADO) ----
        if t == ActionType.TRACKING_START:
            # habilita tracking
            self._set_state_safe(tracking_enabled=True)
            # fuerza RATE 0 0 al activar y keyframe pending (como script)
            self._mount_rate_safe(0.0, 0.0)
            self._tracking_keyframe_reset()
            log_info(self.out_log, "Tracking: START")
            return

        if t == ActionType.TRACKING_STOP:
            # deshabilita tracking
            self._set_state_safe(tracking_enabled=False)
            # seguridad: parar montura
            self._mount_rate_safe(0.0, 0.0)
            log_info(self.out_log, "Tracking: STOP")
            return

        if t == ActionType.TRACKING_SET_PARAMS:
            # payload dict con kwargs para tracking_set_params(state, **kwargs)
            if isinstance(p, dict):
                tracking_set_params(self._tracking_state, **p)
                log_info(self.out_log, f"Tracking: SET_PARAMS {list(p.keys())}")
            return

        # ---- Stacking (NUEVO) ----
        if t == ActionType.STACKING_START:
            self._stacking_enabled = True
            self._stacking.start()
            self._set_state_safe(
                stacking_enabled=True,
                stacking_mode="RUNNING",
                stacking_status="ON",
                stacking_on=True,
            )
            log_info(self.out_log, "Stacking: START")
            return

        if t == ActionType.STACKING_STOP:
            self._stacking_enabled = False
            self._stacking.stop()
            self._set_state_safe(
                stacking_enabled=False,
                stacking_mode="IDLE",
                stacking_status="OFF",
                stacking_on=False,
            )
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

        # ---- Otros módulos (por ahora no implementados aquí) ----
        if t in (
            ActionType.STACKING_SAVE,
            ActionType.PLATESOLVE_RUN,
            ActionType.PLATESOLVE_SET_PARAMS,
            ActionType.MOUNT_SYNC,
            ActionType.MOUNT_GOTO,
        ):
            log_info(self.out_log, f"Action received (not implemented yet): {t}")
            return

        log_info(self.out_log, f"Unknown action type: {t}")
