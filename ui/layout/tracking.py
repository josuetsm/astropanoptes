from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W

from config import AppConfig


@dataclass
class TrackingPanelHandles:
    resp_min: W.BoundedFloatText
    min_sources: W.BoundedIntText
    btn_start: W.Button
    btn_stop: W.Button
    info: W.HTML


def build_tracking_panel(cfg: AppConfig) -> tuple[W.Widget, TrackingPanelHandles]:
    tracking_cfg = cfg.tracking

    resp_min = W.BoundedFloatText(
        description="resp_min",
        value=float(tracking_cfg.resp_min),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="260px"),
    )
    min_sources = W.BoundedIntText(
        description="min_sources",
        value=1,
        min=1,
        max=50,
        step=1,
        layout=W.Layout(width="260px"),
    )

    btn_start = W.Button(description="Start", button_style="success", layout=W.Layout(width="110px"))
    btn_stop = W.Button(description="Stop", button_style="warning", layout=W.Layout(width="110px"))

    info = W.HTML(value="Tracking: idle")

    widget = W.VBox(
        [
            W.HTML("<b>Tracking</b>"),
            W.HBox([btn_start, btn_stop]),
            W.HBox([resp_min, min_sources]),
            info,
        ]
    )

    handles = TrackingPanelHandles(
        resp_min=resp_min,
        min_sources=min_sources,
        btn_start=btn_start,
        btn_stop=btn_stop,
        info=info,
    )
    return widget, handles
