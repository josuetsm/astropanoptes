from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W


@dataclass
class StackingPanelHandles:
    image: W.Image
    btn_reset: W.Button
    btn_start: W.Button
    btn_stop: W.Button


def build_stacking_panel() -> tuple[W.Widget, StackingPanelHandles]:
    image = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    btn_reset = W.Button(description="Reset Stack", button_style="warning", layout=W.Layout(width="140px"))
    btn_start = W.Button(description="Start", button_style="success", layout=W.Layout(width="110px"))
    btn_stop = W.Button(description="Stop", button_style="warning", layout=W.Layout(width="110px"))

    widget = W.VBox(
        [
            W.HTML("<b>Stacking</b>"),
            W.HBox([btn_start, btn_stop, btn_reset]),
            image,
        ]
    )
    handles = StackingPanelHandles(
        image=image,
        btn_reset=btn_reset,
        btn_start=btn_start,
        btn_stop=btn_stop,
    )
    return widget, handles
