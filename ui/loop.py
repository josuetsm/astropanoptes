from __future__ import annotations

from dataclasses import dataclass
import asyncio
import time
from typing import Dict

from app_runner import AppRunner

from ui.layout.camera import CameraPanelHandles, LiveViewHandles, TopBarHandles
from ui.layout.mount import GotoPanelHandles, ManualMountHandles, MountPanelHandles
from ui.layout.tracking import TrackingPanelHandles
from ui.layout.stacking import StackingPanelHandles
from ui.layout.platesolving import PlatesolvingPanelHandles
from ui.utils.debounce import DebouncedCall
from ui.utils.guard import RenderGuard
from ui.utils.widgets import safe_set, set_html, set_image_bytes


@dataclass
class UIDebouncers:
    camera_params: Dict[str, DebouncedCall]
    live_sep: DebouncedCall
    platesolving_params: DebouncedCall
    microsteps: DebouncedCall


@dataclass
class UIHandles:
    top_bar: TopBarHandles
    live_view: LiveViewHandles
    camera_panel: CameraPanelHandles
    mount_panel: MountPanelHandles
    manual_mount: ManualMountHandles
    goto_panel: GotoPanelHandles
    tracking_panel: TrackingPanelHandles
    stacking_panel: StackingPanelHandles
    platesolving_panel: PlatesolvingPanelHandles


class UILoop:
    def __init__(
        self,
        runner: AppRunner,
        handles: "UIHandles",
        guard: RenderGuard,
        debouncers: UIDebouncers,
        max_hz: float = 10.0,
    ) -> None:
        self.runner = runner
        self.handles = handles
        self.guard = guard
        self.debouncers = debouncers
        self.max_hz = max(0.5, float(max_hz))
        self._handle: asyncio.TimerHandle | None = None
        self._stopped = True
        self._loop: asyncio.AbstractEventLoop | None = None
        self._image_cache: Dict[str, int] = {}

    def start(self) -> None:
        if self._handle is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._stopped = False
        self._schedule()

    def stop(self) -> None:
        self._stopped = True
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _schedule(self) -> None:
        if self._loop is None:
            return
        interval = 1.0 / self.max_hz
        self._handle = self._loop.call_later(interval, self._run_once)

    def _run_once(self) -> None:
        if self._stopped:
            return
        self.tick()
        self._schedule()

    def tick(self) -> None:
        now = time.monotonic()
        for debouncer in self.debouncers.camera_params.values():
            debouncer.maybe_fire(now)
        self.debouncers.live_sep.maybe_fire(now)
        self.debouncers.platesolving_params.maybe_fire(now)
        self.debouncers.microsteps.maybe_fire(now)

        state = self.runner.get_state()
        top_bar = self.handles.top_bar
        live_view = self.handles.live_view
        manual_mount = self.handles.manual_mount
        tracking_panel = self.handles.tracking_panel
        stacking_panel = self.handles.stacking_panel
        platesolving_panel = self.handles.platesolving_panel
        goto_panel = self.handles.goto_panel

        cam_connected = bool(state.camera.connected)
        mount_connected = bool(state.mount.connected)
        stacking_running = bool(state.stacking.enabled)
        tracking_running = bool(state.tracking.enabled)

        set_html(top_bar.status_camera, f"Camera: <b>{state.camera.status.value}</b>", self.guard)
        set_html(top_bar.status_mount, f"Mount: <b>{state.mount.status.value}</b>", self.guard)

        if tracking_running:
            mode = str(state.tracking.mode.value)
            resp = float(state.tracking.resp)
            set_html(top_bar.status_tracking, f"Tracking: <b>{mode}</b> (resp={resp:.3f})", self.guard)
        else:
            set_html(top_bar.status_tracking, "Tracking: <b>OFF</b>", self.guard)

        mode = str(state.stacking.status.value)
        fps = float(state.stacking.fps)
        set_html(top_bar.status_stacking, f"Stacking: <b>{mode}</b> ({fps:.2f} fps)", self.guard)

        safe_set(
            top_bar.fps_label,
            "value",
            (
                f"FPS cap: {state.camera.fps_capture:.2f} | view: {state.camera.fps_view:.2f} | "
                f"loop: {state.camera.fps_control_loop:.2f}"
            ),
            self.guard,
        )
        safe_set(top_bar.frame_ms_label, "value", f"frame_ms: {state.camera.frame_ms:.2f}", self.guard)

        if tracking_running:
            mode = str(state.tracking.mode.value)
            resp = float(state.tracking.resp)
            dx = float(state.tracking.dx)
            dy = float(state.tracking.dy)
            raz = float(state.tracking.rate_az)
            ralt = float(state.tracking.rate_alt)
            n_det = int(state.tracking.n_det)
            sep_txt = "SEP=0 (no detections)" if n_det == 0 else f"SEP={n_det}"
            set_html(
                tracking_panel.info,
                (
                    f"<b>Tracking</b>: {mode} | {sep_txt} | resp={resp:.3f} | dx={dx:+.2f} dy={dy:+.2f} | "
                    f"RATE=({raz:+.1f}, {ralt:+.1f})"
                ),
                self.guard,
            )
        else:
            set_html(tracking_panel.info, "Tracking: idle", self.guard)

        ps_status = str(state.platesolving.status.value)
        ps_busy = bool(state.platesolving.busy)
        ps_ok = bool(state.platesolving.last_ok)
        ra = float(state.platesolving.center_ra_deg)
        dec = float(state.platesolving.center_dec_deg)
        th = float(state.platesolving.theta_deg)
        dx = float(state.platesolving.dx_px)
        dy = float(state.platesolving.dy_px)
        resp = float(state.platesolving.resp)
        nin = int(state.platesolving.n_inliers)
        rms = float(state.platesolving.rms_px)
        set_html(
            platesolving_panel.status,
            (
                f"<b>Platesolving</b>: {ps_status} | busy={ps_busy} | ok={ps_ok} | "
                f"RA={ra:.6f} Dec={dec:.6f} | theta={th:+.2f}° dx={dx:+.2f} dy={dy:+.2f} | "
                f"resp={resp:.3f} inliers={nin} rms={rms:.2f}px"
            ),
            self.guard,
        )

        debug_info = dict(state.platesolving.debug_info or {})
        if debug_info:
            ordered = [
                "status",
                "response",
                "n_det",
                "gaia_rows",
                "n_inliers",
                "rms_px",
                "theta_deg",
                "dx_px",
                "dy_px",
                "radius_deg",
                "scale_arcsec_per_px",
            ]

            def fmt_value(val) -> str:
                if isinstance(val, float):
                    return f"{val:.4g}"
                return str(val)

            lines = []
            for key in ordered:
                if key not in debug_info:
                    continue
                val = debug_info.get(key)
                if val is None:
                    continue
                lines.append(f"<li><b>{key}</b>: {fmt_value(val)}</li>")
            if lines:
                set_html(platesolving_panel.debug_html, "<ul>" + "".join(lines) + "</ul>", self.guard)
            else:
                set_html(platesolving_panel.debug_html, "", self.guard)
        else:
            set_html(platesolving_panel.debug_html, "", self.guard)

        if bool(top_bar.tracking_toggle.value) != tracking_running:
            safe_set(top_bar.tracking_toggle, "value", tracking_running, self.guard)
        if bool(top_bar.stacking_toggle.value) != stacking_running:
            safe_set(top_bar.stacking_toggle, "value", stacking_running, self.guard)

        set_image_bytes(
            live_view.image,
            self.runner.get_latest_preview_jpeg(),
            self.guard,
            self._image_cache,
            "live",
        )
        set_image_bytes(
            stacking_panel.image,
            state.stacking.preview_jpeg,
            self.guard,
            self._image_cache,
            "stack",
        )
        set_image_bytes(
            platesolving_panel.image,
            state.platesolving.debug_jpeg,
            self.guard,
            self._image_cache,
            "platesolving",
        )

        safe_set(top_bar.connect_camera, "disabled", cam_connected, self.guard)
        safe_set(top_bar.disconnect_camera, "disabled", not cam_connected, self.guard)
        safe_set(top_bar.connect_mount, "disabled", mount_connected, self.guard)
        safe_set(top_bar.disconnect_mount, "disabled", not mount_connected, self.guard)

        safe_set(manual_mount.ms_az, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.ms_alt, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.steps_az, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.delay_az, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.steps_alt, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.delay_alt, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.btn_az_left, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.btn_az_right, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.btn_alt_up, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.btn_alt_down, "disabled", not mount_connected, self.guard)
        safe_set(manual_mount.btn_stop, "disabled", False, self.guard)

        safe_set(top_bar.tracking_toggle, "disabled", not (cam_connected and mount_connected), self.guard)
        safe_set(tracking_panel.btn_start, "disabled", not (cam_connected and mount_connected), self.guard)
        safe_set(tracking_panel.btn_stop, "disabled", not tracking_running, self.guard)

        safe_set(top_bar.stacking_toggle, "disabled", not cam_connected, self.guard)
        safe_set(stacking_panel.btn_start, "disabled", not cam_connected, self.guard)
        safe_set(stacking_panel.btn_stop, "disabled", not stacking_running, self.guard)
        safe_set(stacking_panel.btn_reset, "disabled", not stacking_running, self.guard)
        safe_set(top_bar.save_stack, "disabled", not stacking_running, self.guard)

        safe_set(platesolving_panel.btn_solve, "disabled", ps_busy, self.guard)
        safe_set(platesolving_panel.search_radius_deg, "disabled", not bool(platesolving_panel.use_radius.value), self.guard)

        safe_set(self.handles.mount_panel.connect_btn, "disabled", mount_connected, self.guard)
        safe_set(self.handles.mount_panel.disconnect_btn, "disabled", not mount_connected, self.guard)

        goto_status = str(state.goto.status.value)
        goto_busy = bool(state.goto.busy)
        goto_synced = bool(state.goto.synced)
        goto_err = float(state.goto.last_error_arcsec)
        j00 = float(state.goto.J00)
        j01 = float(state.goto.J01)
        j10 = float(state.goto.J10)
        j11 = float(state.goto.J11)
        set_html(
            goto_panel.status,
            (
                f"<b>GoTo</b>: {goto_status} | busy={goto_busy} | synced={goto_synced} | "
                f"err={goto_err:.1f}\" | J=[[{j00:.6g},{j01:.6g}],[{j10:.6g},{j11:.6g}]]"
            ),
            self.guard,
        )
