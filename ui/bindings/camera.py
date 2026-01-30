from __future__ import annotations

from typing import Dict

from app_runner import AppRunner

from ui.layout.camera import CameraPanelHandles, LiveViewHandles, TopBarHandles
from ui.utils.debounce import DebouncedCall
from ui.utils.guard import RenderGuard


def bind_camera(
    top_bar: TopBarHandles,
    live_view: LiveViewHandles,
    camera_panel: CameraPanelHandles,
    runner: AppRunner,
    guard: RenderGuard,
    camera_param_debouncers: Dict[str, DebouncedCall],
    live_sep_debouncer: DebouncedCall,
) -> None:
    def on_connect_camera(_btn) -> None:
        runner.request_camera_connect(int(camera_panel.camera_id.value))

    def on_disconnect_camera(_btn) -> None:
        runner.request_camera_disconnect()

    top_bar.connect_camera.on_click(on_connect_camera)
    top_bar.disconnect_camera.on_click(on_disconnect_camera)

    def on_camera_param(name: str, value) -> None:
        if guard.active:
            return
        debouncer = camera_param_debouncers[name]
        debouncer.trigger(lambda: runner.request_camera_param(name, value))

    camera_panel.exp_ms.observe(lambda change: on_camera_param("exp_ms", int(change["new"])), names="value")
    camera_panel.gain.observe(lambda change: on_camera_param("gain", int(change["new"])), names="value")
    camera_panel.auto_gain.observe(lambda change: on_camera_param("auto_gain", bool(change["new"])), names="value")
    camera_panel.view_hz.observe(
        lambda change: on_camera_param("preview_view_hz", float(change["new"])),
        names="value",
    )
    camera_panel.jpeg_q.observe(
        lambda change: on_camera_param("preview_jpeg_quality", int(change["new"])),
        names="value",
    )
    camera_panel.stretch_plo.observe(
        lambda change: on_camera_param("preview_stretch_plo", float(change["new"])),
        names="value",
    )
    camera_panel.stretch_phi.observe(
        lambda change: on_camera_param("preview_stretch_phi", float(change["new"])),
        names="value",
    )

    def on_live_sep_change(_change=None) -> None:
        if guard.active:
            return

        def send() -> None:
            runner.request_live_sep_params(
                enabled=bool(live_view.sep_toggle.value),
                sep_bw=int(live_view.sep_bw.value),
                sep_bh=int(live_view.sep_bh.value),
                sep_thresh_sigma=float(live_view.sep_sigma.value),
                sep_minarea=int(live_view.sep_minarea.value),
                max_det=int(live_view.sep_max_det.value),
            )

        live_sep_debouncer.trigger(send)

    live_view.sep_toggle.observe(on_live_sep_change, names="value")
    live_view.sep_bw.observe(on_live_sep_change, names="value")
    live_view.sep_bh.observe(on_live_sep_change, names="value")
    live_view.sep_sigma.observe(on_live_sep_change, names="value")
    live_view.sep_minarea.observe(on_live_sep_change, names="value")
    live_view.sep_max_det.observe(on_live_sep_change, names="value")
