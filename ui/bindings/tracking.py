from __future__ import annotations

from app_runner import AppRunner

from ui.layout.camera import TopBarHandles
from ui.layout.tracking import TrackingPanelHandles
from ui.utils.guard import RenderGuard


def bind_tracking(
    top_bar: TopBarHandles,
    tracking_panel: TrackingPanelHandles,
    runner: AppRunner,
    guard: RenderGuard,
) -> None:
    def send_track_params() -> None:
        runner.request_tracking_params(
            resp_min=float(tracking_panel.resp_min.value),
        )

    def on_tracking_toggle(change) -> None:
        if guard.active:
            return
        on = bool(change["new"])
        current = bool(runner.get_state().tracking.enabled)
        if on == current:
            return
        if on:
            send_track_params()
            runner.request_tracking_start()
        else:
            runner.request_tracking_stop()

    top_bar.tracking_toggle.observe(on_tracking_toggle, names="value")

    tracking_panel.resp_min.observe(lambda _change: send_track_params(), names="value")

    tracking_panel.btn_start.on_click(lambda _btn: (send_track_params(), runner.request_tracking_start()))
    tracking_panel.btn_stop.on_click(lambda _btn: runner.request_tracking_stop())
