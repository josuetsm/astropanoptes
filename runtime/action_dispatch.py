from __future__ import annotations

import datetime as _dt
import queue
from typing import Any, Dict

import serial

from actions import Action, ActionType
from ap_types import Axis
from logging_utils import log_info, log_error

from tracking import auto_reset, tracking_set_params
from platesolve import PlatesolveConfig, save_gaia_auth


def _now_s() -> float:
    import time

    return time.time()


def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return "(none)"
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


class ActionDispatcher:
    def __init__(self, runner, logger) -> None:
        self._runner = runner
        self._logger = logger

    def drain(self, max_n: int = 50) -> None:
        for _ in range(max_n):
            try:
                act = self._runner._actions.get_nowait()
            except queue.Empty:
                return

            try:
                self.handle(act)
            except Exception as exc:
                if act.type in (ActionType.CAMERA_CONNECT, ActionType.CAMERA_SET_PARAM):
                    self._runner._set_state_safe(camera_status="ERR", camera_connected=False)

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
                    ActionType.TRACKING_CALIB_AZ,
                    ActionType.TRACKING_CALIB_ALT,
                    ActionType.TRACKING_CALIB_RESET,
                    ActionType.TRACKING_AUTO_RESET,
                    ActionType.TRACKING_BOOTSTRAP,
                    ActionType.STACKING_START,
                    ActionType.STACKING_STOP,
                    ActionType.STACKING_RESET,
                    ActionType.STACKING_SET_PARAMS,
                    ActionType.PLATESOLVE_RUN,
                    ActionType.PLATESOLVE_SET_PARAMS,
                    ActionType.GOTO_AUTOCALIBRATE,
                ):
                    if act.type in (
                        ActionType.MOUNT_CONNECT,
                        ActionType.MOUNT_NUDGE,
                        ActionType.MOUNT_START_CONTINUOUS,
                        ActionType.MOUNT_STOP,
                        ActionType.MOUNT_SET_MICROSTEPS,
                        ActionType.MOUNT_MOVE_STEPS,
                    ):
                        self._runner._set_state_safe(mount_status="ERR", mount_connected=False)
                        self._runner._set_state_safe(tracking_enabled=False, tracking_mode="IDLE")

                log_error(self._logger, f"Action failed: {act.type}", exc)

    def handle(self, act: Action) -> None:
        t = act.type
        p = act.payload

        # ---- Camera ----
        if t == ActionType.CAMERA_CONNECT:
            idx = int(p.get("camera_index", 0))
            self._runner.cfg.camera.camera_index = idx
            self._runner._connect_camera(idx)
            return

        if t == ActionType.CAMERA_DISCONNECT:
            self._runner._shutdown_camera()
            log_info(self._logger, "Camera: disconnected")
            return

        if t == ActionType.CAMERA_SET_PARAM:
            name = str(p.get("name", ""))
            value = p.get("value", None)
            self._runner._apply_camera_param(name, value)
            return

        if t == ActionType.RESET_CAMERA_DEFAULTS:
            self._runner._reset_camera_defaults()
            log_info(self._logger, "Camera: RESET_DEFAULTS")
            return

        if t == ActionType.RESET_PREVIEW_DEFAULTS:
            self._runner._reset_preview_defaults()
            log_info(self._logger, "Preview: RESET_DEFAULTS")
            return

        # ---- Mount ----
        if t == ActionType.MOUNT_CONNECT:
            port = str(p.get("port", ""))
            baud = int(p.get("baudrate", 115200))
            self._runner._connect_mount(port, baud)
            return

        if t == ActionType.MOUNT_DISCONNECT:
            self._runner._shutdown_mount()
            log_info(self._logger, "Mount: disconnected")
            return

        if t == ActionType.MOUNT_NUDGE:
            if self._runner._mount is None or not self._runner._mount.is_connected():
                return
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            rate = float(p.get("rate", 0.0))
            duration_ms = int(p.get("duration_ms", 0))
            try:
                self._runner._mount.nudge(axis, direction, rate, duration_ms)
                self._runner._tracking_keyframe_reset()
            except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
                self._runner._set_state_safe(
                    mount_status="ERR",
                    mount_connected=False,
                    tracking_enabled=False,
                    tracking_mode="IDLE",
                )
                log_error(self._logger, "Mount: NUDGE failed", exc)
            return

        if t == ActionType.MOUNT_START_CONTINUOUS:
            if self._runner._mount is None or not self._runner._mount.is_connected():
                return
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            rate = float(p.get("rate", 0.0))
            try:
                self._runner._mount.start_continuous(axis, direction, rate)
                self._runner._tracking_keyframe_reset()
            except (RuntimeError, ValueError, OSError, serial.SerialException) as exc:
                self._runner._set_state_safe(
                    mount_status="ERR",
                    mount_connected=False,
                    tracking_enabled=False,
                    tracking_mode="IDLE",
                )
                log_error(self._logger, "Mount: START_CONTINUOUS failed", exc)
            return

        if t == ActionType.MOUNT_STOP:
            self._runner._mount_stop()
            self._runner._tracking_keyframe_reset()
            return

        if t == ActionType.MOUNT_SET_MICROSTEPS:
            az_div = int(p.get("az_div", 64))
            alt_div = int(p.get("alt_div", 64))
            self._runner._mount_set_microsteps(az_div, alt_div)
            return

        if t == ActionType.MOUNT_MOVE_STEPS:
            axis = Axis(str(p.get("axis", Axis.AZ.value)))
            direction = int(p.get("direction", 1))
            steps = int(p.get("steps", 600))
            delay_us = int(p.get("delay_us", 1800))
            self._runner._mount_move_steps(axis, direction, steps, delay_us)
            self._runner._tracking_keyframe_reset()
            return

        if t == ActionType.RESET_MOUNT_DEFAULTS:
            self._runner._reset_mount_defaults()
            log_info(self._logger, "Mount: RESET_DEFAULTS")
            return

        # ---- Tracking ----
        if t == ActionType.TRACKING_START:
            self._runner._set_state_safe(tracking_enabled=True)
            self._runner._mount_rate_safe(0.0, 0.0)
            self._runner._tracking_keyframe_reset()
            log_info(self._logger, "Tracking: START")
            return

        if t == ActionType.TRACKING_STOP:
            self._runner._set_state_safe(tracking_enabled=False)
            self._runner._mount_rate_safe(0.0, 0.0)
            log_info(self._logger, "Tracking: STOP")
            return

        if t == ActionType.TRACKING_SET_PARAMS:
            if isinstance(p, dict):
                tracking_set_params(self._runner._tracking_state, **p)
                log_info(self._logger, f"Tracking: SET_PARAMS {_format_params(p)}")
            return

        if t == ActionType.RESET_TRACKING_DEFAULTS:
            self._runner._reset_tracking_defaults()
            log_info(self._logger, "Tracking: RESET_DEFAULTS")
            return

        if t == ActionType.TRACKING_KEYFRAME_RESET:
            self._runner._tracking_keyframe_reset()
            return

        if t == ActionType.TRACKING_CALIB_RESET:
            self._runner._tracking_calib_reset()
            self._runner._tracking_bootstrap_reset()
            log_info(self._logger, "Tracking: CALIB_RESET")
            return

        if t == ActionType.TRACKING_AUTO_RESET:
            auto_reset(self._runner._tracking_state, src="none")
            self._runner._tracking_bootstrap_reset()
            log_info(self._logger, "Tracking: AUTO_RESET")
            return

        if t == ActionType.TRACKING_CALIB_AZ:
            was_tracking = self._runner._tracking_pause_for_calib()
            az_col = self._runner._tracking_calibrate_axis(Axis.AZ)
            if az_col is not None:
                self._runner._tracking_calib_cols["az"] = az_col
                self._runner._tracking_apply_calib_cols()
            self._runner._tracking_resume_after_calib(was_tracking)
            return

        if t == ActionType.TRACKING_CALIB_ALT:
            was_tracking = self._runner._tracking_pause_for_calib()
            alt_col = self._runner._tracking_calibrate_axis(Axis.ALT)
            if alt_col is not None:
                self._runner._tracking_calib_cols["alt"] = alt_col
                self._runner._tracking_apply_calib_cols()
            self._runner._tracking_resume_after_calib(was_tracking)
            return

        if t == ActionType.TRACKING_BOOTSTRAP:
            self._runner._tracking_bootstrap_start(_now_s())
            return

        # ---- Stacking ----
        if t == ActionType.STACKING_START:
            self._runner._stacking_enabled = True
            self._runner._stacking.start()
            self._runner._set_state_safe(
                stacking_enabled=True,
                stacking_mode="RUNNING",
                stacking_status="ON",
                stacking_on=True,
            )
            log_info(self._logger, "Stacking: START")
            return

        if t == ActionType.STACKING_STOP:
            self._runner._stacking_enabled = False
            self._runner._stacking.stop()
            self._runner._set_state_safe(
                stacking_enabled=False,
                stacking_mode="IDLE",
                stacking_status="OFF",
                stacking_on=False,
            )
            log_info(self._logger, "Stacking: STOP")
            return

        if t == ActionType.STACKING_RESET:
            self._runner._stacking.reset()
            log_info(self._logger, "Stacking: RESET")
            return

        if t == ActionType.STACKING_SET_PARAMS:
            if isinstance(p, dict):
                self._runner._stacking.set_params(**p)
                log_info(self._logger, f"Stacking: SET_PARAMS {list(p.keys())}")
            return

        if t == ActionType.RESET_STACKING_DEFAULTS:
            self._runner._reset_stacking_defaults()
            log_info(self._logger, "Stacking: RESET_DEFAULTS")
            return

        # Save stacked mosaic (raw + stretch)
        if t == ActionType.STACKING_SAVE:
            # Payload should contain out_dir, basename, fmt; defaults provided
            if isinstance(p, dict):
                out_dir = str(p.get("out_dir", "stack_output"))
                basename = str(p.get("basename", "stack"))
                fmt = str(p.get("fmt", "png"))
                self._runner._save_stacking(out_dir, basename, fmt)
            else:
                # Fallback to default directory and timestamp
                out_dir = "stack_output"
                basename = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                fmt = "png"
                self._runner._save_stacking(out_dir, basename, fmt)
            return

        # ---- Hotpixels ----
        if t == ActionType.HOTPIX_CALIBRATE:
            if self._runner._cam_stream is None:
                log_info(self._logger, "Hotpix: calibration skipped (camera stream inactive)")
                return
            if self._runner._get_tracking_enabled():
                self._runner._set_state_safe(tracking_enabled=False, tracking_mode="IDLE")
                self._runner._mount_rate_safe(0.0, 0.0)
            n_frames = int(p.get("n_frames", self._runner.cfg.hotpixels.calib_frames))
            abs_percentile = float(p.get("abs_percentile", self._runner.cfg.hotpixels.calib_abs_percentile))
            var_percentile = float(p.get("var_percentile", self._runner.cfg.hotpixels.calib_var_percentile))
            max_component_area = int(p.get("max_component_area", self._runner.cfg.hotpixels.max_component_area))
            out_path_base = str(p.get("out_path_base", self._runner.cfg.hotpixels.mask_path_base))
            self._runner._hotpix_start_worker_if_needed(
                n_frames=n_frames,
                abs_percentile=abs_percentile,
                var_percentile=var_percentile,
                max_component_area=max_component_area,
                out_path_base=out_path_base,
            )
            return

        if t == ActionType.RESET_HOTPIXELS_DEFAULTS:
            self._runner._reset_hotpixels_defaults()
            log_info(self._logger, "Hotpix: RESET_DEFAULTS")
            return

        # ---- Platesolve ----
        if t == ActionType.PLATESOLVE_SET_PARAMS:
            # Permite actualizar PlatesolveConfig desde UI sin reimportar
            # Ej: {'pixel_size_m': 2.9e-6, 'focal_m': 0.9, 'gmax': 14.5, ...}
            if isinstance(p, dict):
                payload = dict(p)
                if "auto_target" in payload:
                    self._runner._platesolve.set_auto_target(str(payload.pop("auto_target") or ""))

                # Rebuild dataclass con campos existentes
                with self._runner._platesolve.cfg_lock:
                    d = dict(self._runner.cfg.platesolve.__dict__)
                    for k, v in payload.items():
                        if k in d:
                            d[k] = v
                    self._runner.cfg.platesolve = PlatesolveConfig(**d)
                if payload:
                    log_info(self._logger, "Platesolve: params updated")
            return

        if t == ActionType.RESET_PLATESOLVE_DEFAULTS:
            self._runner._reset_platesolve_defaults()
            log_info(self._logger, "Platesolve: RESET_DEFAULTS")
            return

        # ---- Live SEP overlay ----
        if t == ActionType.LIVE_SEP_SET_PARAMS:
            if isinstance(p, dict):
                enabled = p.get("enabled", self._runner._live_sep_overlay_enabled)
                self._runner._live_sep_overlay_enabled = bool(enabled)
                for key in ("sep_bw", "sep_bh", "sep_thresh_sigma", "sep_minarea", "max_det"):
                    if key in p:
                        self._runner._live_sep_params[key] = p.get(key)
                if "sep_bw" in p:
                    self._runner.cfg.sep.bw = int(p.get("sep_bw"))
                if "sep_bh" in p:
                    self._runner.cfg.sep.bh = int(p.get("sep_bh"))
                if "sep_thresh_sigma" in p:
                    self._runner.cfg.sep.thresh_sigma = float(p.get("sep_thresh_sigma"))
                if "sep_minarea" in p:
                    self._runner.cfg.sep.minarea = int(p.get("sep_minarea"))
                self._runner._goto.cfg.sep = self._runner.cfg.sep
                log_info(self._logger, "Live SEP: params updated")
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
                log_info(self._logger, "Platesolve: Gaia credentials saved")

            self._runner._platesolve.request(target=target)
            log_info(self._logger, "Platesolve: RUN source=live")
            return

        # ---- GoTo ----
        if t == ActionType.MOUNT_SYNC:
            # Sync usando el último platesolve OK
            sol = self._runner._platesolve.get_last_result()
            if sol is None or not bool(getattr(sol, "success", False)):
                log_info(self._logger, "GoTo: sync failed (no successful platesolve cached)")
                self._runner._set_state_safe(goto_synced=False, goto_status="SYNC_ERR")
                return
            ok = False
            try:
                ok = bool(self._runner._goto.sync_from_platesolve(sol))
            except Exception as exc:
                log_error(self._logger, "GoTo: sync exception", exc)
            self._runner._set_state_safe(goto_synced=bool(ok), goto_status="SYNC_OK" if ok else "SYNC_ERR")
            log_info(self._logger, f"GoTo: sync {'OK' if ok else 'ERR'}")
            return

        if t == ActionType.MOUNT_GOTO:
            target = p.get("target", {})
            self._runner._goto_request(kind="goto", target=target, params=p)
            return

        if t == ActionType.GOTO_CALIBRATE:
            params = p.get("params", {})
            self._runner._goto_request(kind="calibrate", target=None, params=params)
            return

        if t == ActionType.GOTO_AUTOCALIBRATE:
            params = p.get("params", {})
            self._runner._goto_request(kind="autocal", target=None, params=params)
            return

        if t == ActionType.GOTO_CANCEL:
            self._runner._goto_cancel.set()
            try:
                self._runner._mount_stop()
            except Exception as exc:
                log_error(self._logger, "GoTo: cancel mount stop failed", exc)
            self._runner._set_state_safe(goto_busy=False, goto_status="CANCELLED")
            return

        # ---- Otros ----
        log_info(self._logger, f"Unknown or unhandled action type: {t}")
