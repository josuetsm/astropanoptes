from __future__ import annotations

import time

from app_runner import AppRunner

from ui.layout.camera import TopBarHandles
from ui.layout.stacking import StackingPanelHandles
from ui.utils.guard import RenderGuard


def bind_stacking(
    top_bar: TopBarHandles,
    stacking_panel: StackingPanelHandles,
    runner: AppRunner,
    guard: RenderGuard,
) -> None:
    def on_stacking_toggle(change) -> None:
        if guard.active:
            return
        on = bool(change["new"])
        current = bool(runner.get_state().stacking.enabled)
        if on == current:
            return
        if on:
            runner.request_stacking_start()
        else:
            runner.request_stacking_stop()

    top_bar.stacking_toggle.observe(on_stacking_toggle, names="value")

    stacking_panel.btn_start.on_click(lambda _btn: runner.request_stacking_start())
    stacking_panel.btn_stop.on_click(lambda _btn: runner.request_stacking_stop())
    stacking_panel.btn_reset.on_click(lambda _btn: runner.request_stacking_reset())

    def on_save_stack(_btn) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = "stack_output"
        runner.request_stacking_save(out_dir=out_dir, basename=ts, fmt="png")

    top_bar.save_stack.on_click(on_save_stack)
