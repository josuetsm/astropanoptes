from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W


@dataclass
class DebugPanelHandles:
    log_output: W.Output


def build_debug_panel() -> tuple[W.Widget, DebugPanelHandles]:
    log_output = W.Output(layout=W.Layout(border="1px solid #ddd", height="180px", overflow="auto"))
    widget = W.VBox([W.HTML("<b>Logs</b>"), log_output])
    return widget, DebugPanelHandles(log_output=log_output)
