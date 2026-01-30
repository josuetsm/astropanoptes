from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W
from IPython.display import display

from app_runner import AppRunner
from ui.bindings.camera import bind_camera
from ui.bindings.mount import bind_mount
from ui.bindings.tracking import bind_tracking
from ui.bindings.stacking import bind_stacking
from ui.bindings.platesolving import bind_platesolving
from ui.layout.camera import build_camera_panel, build_live_view, build_top_bar
from ui.layout.debug import build_debug_panel
from ui.layout.mount import build_goto_panel, build_manual_mount, build_mount_panel
from ui.layout.tracking import build_tracking_panel
from ui.layout.stacking import build_stacking_panel
from ui.layout.platesolving import build_platesolving_panel
from ui.loop import UIDebouncers, UIHandles, UILoop
from ui.utils.debounce import DebouncedCall
from ui.utils.guard import RenderGuard


@dataclass
class UI:
    root: W.Widget
    handles: UIHandles
    loop: UILoop


def _build_ui(runner: AppRunner) -> UI:
    cfg = runner.cfg

    top_bar_widget, top_bar_handles = build_top_bar()
    live_view_widget, live_view_handles = build_live_view(cfg)
    manual_mount_widget, manual_mount_handles = build_manual_mount(cfg)
    camera_panel_widget, camera_panel_handles = build_camera_panel(cfg)
    mount_panel_widget, mount_panel_handles = build_mount_panel(cfg)
    tracking_panel_widget, tracking_panel_handles = build_tracking_panel(cfg)
    stacking_panel_widget, stacking_panel_handles = build_stacking_panel()
    platesolving_panel_widget, platesolving_panel_handles = build_platesolving_panel(cfg)
    goto_panel_widget, goto_panel_handles = build_goto_panel(cfg)
    debug_panel_widget, debug_panel_handles = build_debug_panel()

    runner.out_log = debug_panel_handles.log_output

    tabs = W.Tab(
        children=[
            camera_panel_widget,
            mount_panel_widget,
            tracking_panel_widget,
            stacking_panel_widget,
            platesolving_panel_widget,
            goto_panel_widget,
            debug_panel_widget,
        ]
    )
    for i, name in enumerate([
        "Camera",
        "Mount",
        "Tracking",
        "Stacking",
        "Platesolving",
        "GoTo",
        "Logs",
    ]):
        tabs.set_title(i, name)

    root = W.VBox([top_bar_widget, live_view_widget, manual_mount_widget, tabs], layout=W.Layout(gap="10px"))

    guard = RenderGuard()
    debouncers = UIDebouncers(
        camera_params={
            "exp_ms": DebouncedCall(0.3),
            "gain": DebouncedCall(0.3),
            "auto_gain": DebouncedCall(0.3),
            "preview_view_hz": DebouncedCall(0.3),
            "preview_jpeg_quality": DebouncedCall(0.3),
            "preview_stretch_plo": DebouncedCall(0.3),
            "preview_stretch_phi": DebouncedCall(0.3),
        },
        live_sep=DebouncedCall(0.3),
        platesolving_params=DebouncedCall(0.8),
        microsteps=DebouncedCall(0.3),
    )

    handles = UIHandles(
        top_bar=top_bar_handles,
        live_view=live_view_handles,
        camera_panel=camera_panel_handles,
        mount_panel=mount_panel_handles,
        manual_mount=manual_mount_handles,
        goto_panel=goto_panel_handles,
        tracking_panel=tracking_panel_handles,
        stacking_panel=stacking_panel_handles,
        platesolving_panel=platesolving_panel_handles,
    )

    bind_camera(
        top_bar_handles,
        live_view_handles,
        camera_panel_handles,
        runner,
        guard,
        debouncers.camera_params,
        debouncers.live_sep,
    )
    bind_mount(
        top_bar_handles,
        mount_panel_handles,
        manual_mount_handles,
        goto_panel_handles,
        runner,
        guard,
        debouncers.microsteps,
    )
    bind_tracking(top_bar_handles, tracking_panel_handles, runner, guard)
    bind_stacking(top_bar_handles, stacking_panel_handles, runner, guard)
    bind_platesolving(platesolving_panel_handles, runner, guard, debouncers.platesolving_params)

    ui_loop = UILoop(runner, handles, guard, debouncers, max_hz=10.0)
    return UI(root=root, handles=handles, loop=ui_loop)


def build_ui(runner: AppRunner) -> W.Widget:
    return _build_ui(runner).root


def show_ui(runner: AppRunner, *, start_loops: bool = True, ui_hz: float = 10.0) -> tuple[UI, UILoop]:
    ui = _build_ui(runner)
    ui.loop.max_hz = float(ui_hz)
    display(ui.root)
    if start_loops:
        ui.loop.start()
    return ui, ui.loop
