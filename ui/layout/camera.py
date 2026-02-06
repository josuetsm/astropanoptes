from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W

from config import AppConfig


@dataclass
class TopBarHandles:
    status_camera: W.HTML
    status_mount: W.HTML
    status_tracking: W.HTML
    status_stacking: W.HTML
    fps_label: W.Label
    frame_ms_label: W.Label
    connect_camera: W.Button
    disconnect_camera: W.Button
    connect_mount: W.Button
    disconnect_mount: W.Button
    stop_mount: W.Button
    tracking_toggle: W.ToggleButton
    stacking_toggle: W.ToggleButton
    save_stack: W.Button
    record_raw20: W.Button


@dataclass
class LiveViewHandles:
    image: W.Image
    sep_toggle: W.ToggleButton
    sep_bw: W.BoundedIntText
    sep_bh: W.BoundedIntText
    sep_sigma: W.BoundedFloatText
    sep_minarea: W.BoundedIntText
    sep_max_det: W.BoundedIntText


@dataclass
class CameraPanelHandles:
    camera_id: W.Dropdown
    exp_ms: W.BoundedIntText
    gain: W.BoundedIntText
    auto_gain: W.Checkbox
    img_format: W.HTML
    view_hz: W.BoundedFloatText
    jpeg_q: W.BoundedIntText
    stretch_plo: W.BoundedFloatText
    stretch_phi: W.BoundedFloatText


def build_top_bar() -> tuple[W.Widget, TopBarHandles]:
    status_camera = W.HTML(value="Camera: <b>DISCONNECTED</b>")
    status_mount = W.HTML(value="Mount: <b>DISCONNECTED</b>")
    status_tracking = W.HTML(value="Tracking: <b>OFF</b>")
    status_stacking = W.HTML(value="Stacking: <b>OFF</b>")

    fps_label = W.Label(value="FPS cap: 0.0 | view: 0.0 | loop: 0.0")
    frame_ms_label = W.Label(value="frame_ms: 0.0")

    connect_camera = W.Button(description="Connect Camera", button_style="success")
    disconnect_camera = W.Button(description="Disconnect Camera", button_style="")
    connect_mount = W.Button(description="Connect Mount", button_style="success")
    disconnect_mount = W.Button(description="Disconnect Mount", button_style="")
    stop_mount = W.Button(description="STOP Mount", button_style="danger")

    tracking_toggle = W.ToggleButton(description="Tracking", value=False, disabled=False)
    stacking_toggle = W.ToggleButton(description="Stacking", value=False, disabled=False)
    save_stack = W.Button(description="Save Stack", disabled=True)
    record_raw20 = W.Button(description="Record 20s RAW", button_style="warning")

    top_left = W.VBox([status_camera, status_mount, status_tracking, status_stacking])
    top_mid = W.VBox([fps_label, frame_ms_label])
    top_right = W.HBox(
        [
            connect_camera,
            disconnect_camera,
            connect_mount,
            disconnect_mount,
            stop_mount,
            tracking_toggle,
            stacking_toggle,
            save_stack,
            record_raw20,
        ],
        layout=W.Layout(flex_flow="row wrap", align_items="center"),
    )
    widget = W.HBox([top_left, top_mid, top_right], layout=W.Layout(justify_content="space-between"))
    handles = TopBarHandles(
        status_camera=status_camera,
        status_mount=status_mount,
        status_tracking=status_tracking,
        status_stacking=status_stacking,
        fps_label=fps_label,
        frame_ms_label=frame_ms_label,
        connect_camera=connect_camera,
        disconnect_camera=disconnect_camera,
        connect_mount=connect_mount,
        disconnect_mount=disconnect_mount,
        stop_mount=stop_mount,
        tracking_toggle=tracking_toggle,
        stacking_toggle=stacking_toggle,
        save_stack=save_stack,
        record_raw20=record_raw20,
    )
    return widget, handles


def build_live_view(cfg: AppConfig) -> tuple[W.Widget, LiveViewHandles]:
    sep_cfg = cfg.sep
    platesolving_cfg = cfg.platesolving

    image = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="500px"))

    sep_toggle = W.ToggleButton(
        description="SEP Overlay",
        value=False,
        disabled=False,
        layout=W.Layout(width="140px"),
    )
    sep_bw = W.BoundedIntText(
        description="sep_bw",
        value=int(sep_cfg.bw),
        min=4,
        max=512,
        step=1,
        layout=W.Layout(width="180px"),
    )
    sep_bh = W.BoundedIntText(
        description="sep_bh",
        value=int(sep_cfg.bh),
        min=4,
        max=512,
        step=1,
        layout=W.Layout(width="180px"),
    )
    sep_sigma = W.BoundedFloatText(
        description="sep_sigma",
        value=float(sep_cfg.thresh_sigma),
        min=0.1,
        max=20.0,
        step=0.1,
        layout=W.Layout(width="200px"),
    )
    sep_minarea = W.BoundedIntText(
        description="sep_minarea",
        value=int(sep_cfg.minarea),
        min=1,
        max=500,
        step=1,
        layout=W.Layout(width="220px"),
    )
    sep_max_det = W.BoundedIntText(
        description="max_det",
        value=int(platesolving_cfg.max_det),
        min=1,
        max=5000,
        step=5,
        layout=W.Layout(width="200px"),
    )

    overlay_controls = W.VBox(
        [
            W.HTML("<b>Live SEP Overlay</b>"),
            W.HBox([sep_toggle, sep_max_det]),
            W.HBox([sep_bw, sep_bh, sep_sigma, sep_minarea]),
        ],
        layout=W.Layout(border="1px solid #eee", padding="6px", gap="6px", max_width="980px"),
    )

    widget = W.VBox([image, overlay_controls])
    handles = LiveViewHandles(
        image=image,
        sep_toggle=sep_toggle,
        sep_bw=sep_bw,
        sep_bh=sep_bh,
        sep_sigma=sep_sigma,
        sep_minarea=sep_minarea,
        sep_max_det=sep_max_det,
    )
    return widget, handles


def build_camera_panel(cfg: AppConfig) -> tuple[W.Widget, CameraPanelHandles]:
    camera_cfg = cfg.camera
    preview_cfg = cfg.preview

    camera_id = W.Dropdown(
        options=[("0", 0)],
        value=int(camera_cfg.camera_index),
        description="Camera",
        layout=W.Layout(width="220px"),
    )

    exp_ms = W.BoundedIntText(
        value=int(camera_cfg.exp_ms),
        min=1,
        max=60000,
        step=1,
        description="Exp (ms)",
        layout=W.Layout(width="240px"),
    )

    gain = W.BoundedIntText(
        value=int(camera_cfg.gain),
        min=0,
        max=500,
        step=1,
        description="Gain",
        layout=W.Layout(width="220px"),
    )

    auto_gain = W.Checkbox(value=bool(camera_cfg.auto_gain), description="Auto Gain")

    img_format = W.HTML(
        value="<b>Format</b>: RAW16 (fixed)",
        layout=W.Layout(width="240px"),
    )

    view_hz = W.BoundedFloatText(
        value=float(preview_cfg.view_hz),
        min=0.5,
        max=60.0,
        step=0.5,
        description="View Hz",
        layout=W.Layout(width="240px"),
    )
    jpeg_q = W.BoundedIntText(
        value=int(preview_cfg.jpeg_quality),
        min=10,
        max=100,
        step=1,
        description="JPEG Q",
        layout=W.Layout(width="240px"),
    )

    stretch_plo = W.BoundedFloatText(
        value=float(preview_cfg.stretch_plo),
        min=0.0,
        max=30.0,
        step=0.5,
        description="Pctl Lo",
        layout=W.Layout(width="240px"),
    )
    stretch_phi = W.BoundedFloatText(
        value=float(preview_cfg.stretch_phi),
        min=70.0,
        max=100.0,
        step=0.1,
        description="Pctl Hi",
        layout=W.Layout(width="240px"),
    )

    cam_grid = W.VBox(
        [
            W.HBox([camera_id, img_format]),
            W.HBox([exp_ms, gain, auto_gain]),
            W.HBox([view_hz, jpeg_q]),
            W.HBox([stretch_plo, stretch_phi]),
        ]
    )
    widget = W.VBox([W.HTML("<b>Camera</b>"), cam_grid])
    handles = CameraPanelHandles(
        camera_id=camera_id,
        exp_ms=exp_ms,
        gain=gain,
        auto_gain=auto_gain,
        img_format=img_format,
        view_hz=view_hz,
        jpeg_q=jpeg_q,
        stretch_plo=stretch_plo,
        stretch_phi=stretch_phi,
    )
    return widget, handles
