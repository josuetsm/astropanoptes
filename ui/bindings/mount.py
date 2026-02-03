from __future__ import annotations

from app_runner import AppRunner
from ap_types import Axis

from ui.layout.mount import GotoPanelHandles, ManualMountHandles, MountPanelHandles
from ui.layout.camera import TopBarHandles
from ui.utils.debounce import DebouncedCall
from ui.utils.guard import RenderGuard


def bind_mount(
    top_bar: TopBarHandles,
    mount_panel: MountPanelHandles,
    manual_mount: ManualMountHandles,
    goto_panel: GotoPanelHandles,
    runner: AppRunner,
    guard: RenderGuard,
    microstep_debouncer: DebouncedCall,
) -> None:
    def enqueue_mount_connect() -> None:
        port = str(mount_panel.serial_port.value).strip()
        baud = int(mount_panel.baudrate.value)
        runner.request_mount_connect(port, baud)

    def enqueue_mount_disconnect() -> None:
        runner.request_mount_disconnect()

    def on_connect_mount(_btn) -> None:
        enqueue_mount_connect()

    def on_disconnect_mount(_btn) -> None:
        enqueue_mount_disconnect()

    top_bar.connect_mount.on_click(on_connect_mount)
    top_bar.disconnect_mount.on_click(on_disconnect_mount)
    mount_panel.connect_btn.on_click(on_connect_mount)
    mount_panel.disconnect_btn.on_click(on_disconnect_mount)

    def send_microsteps() -> None:
        runner.request_mount_set_microsteps(
            az_div=int(manual_mount.ms_az.value),
            alt_div=int(manual_mount.ms_alt.value),
        )

    def on_microstep_change(_change=None) -> None:
        if guard.active:
            return
        microstep_debouncer.trigger(send_microsteps)

    manual_mount.ms_az.observe(on_microstep_change, names="value")
    manual_mount.ms_alt.observe(on_microstep_change, names="value")

    def on_ramp_change(_change=None) -> None:
        if guard.active:
            return
        cfg = runner.cfg.mount
        cfg.manual_ramp_enable = bool(manual_mount.ramp_enable.value)
        cfg.manual_ramp_frac = float(manual_mount.ramp_frac.value)
        cfg.manual_ramp_min_steps = int(manual_mount.ramp_min_steps.value)
        cfg.manual_ramp_start_delay_scale = float(manual_mount.ramp_start_scale.value)
        cfg.manual_ramp_segments = int(manual_mount.ramp_segments.value)

    for widget in [
        manual_mount.ramp_enable,
        manual_mount.ramp_frac,
        manual_mount.ramp_min_steps,
        manual_mount.ramp_start_scale,
        manual_mount.ramp_segments,
    ]:
        widget.observe(on_ramp_change, names="value")

    def enqueue_move(axis: Axis, direction: int, steps: int, delay_us: int) -> None:
        if steps <= 0:
            return
        if delay_us <= 0:
            return
        on_ramp_change()
        runner.request_mount_move_steps(axis=axis, direction=direction, steps=steps, delay_us=delay_us)

    manual_mount.btn_az_left.on_click(
        lambda _btn: enqueue_move(Axis.AZ, -1, int(manual_mount.steps_az.value), int(manual_mount.delay_az.value))
    )
    manual_mount.btn_az_right.on_click(
        lambda _btn: enqueue_move(Axis.AZ, +1, int(manual_mount.steps_az.value), int(manual_mount.delay_az.value))
    )
    manual_mount.btn_alt_up.on_click(
        lambda _btn: enqueue_move(Axis.ALT, +1, int(manual_mount.steps_alt.value), int(manual_mount.delay_alt.value))
    )
    manual_mount.btn_alt_down.on_click(
        lambda _btn: enqueue_move(Axis.ALT, -1, int(manual_mount.steps_alt.value), int(manual_mount.delay_alt.value))
    )
    manual_mount.btn_stop.on_click(lambda _btn: runner.request_mount_stop())
    top_bar.stop_mount.on_click(lambda _btn: runner.request_mount_stop())

    def goto_mode_changed(change) -> None:
        if guard.active:
            return
        mode = str(change["new"])
        if mode == "name":
            goto_panel.target_box.children = [goto_panel.box_name]
        elif mode == "planet":
            goto_panel.target_box.children = [goto_panel.box_planet]
        elif mode == "radec":
            goto_panel.target_box.children = [goto_panel.box_radec]
        else:
            goto_panel.target_box.children = [goto_panel.box_altaz]

    goto_panel.mode.observe(goto_mode_changed, names="value")

    def build_goto_target():
        mode = str(goto_panel.mode.value)
        if mode == "name":
            return str(goto_panel.name.value).strip()
        if mode == "planet":
            return str(goto_panel.planet.value).strip()
        if mode == "radec":
            return {"ra_deg": float(goto_panel.ra.value), "dec_deg": float(goto_panel.dec.value)}
        return {"az_deg": float(goto_panel.az.value), "alt_deg": float(goto_panel.alt.value)}

    def enqueue_goto_sync() -> None:
        runner.request_mount_sync()

    def enqueue_goto_run() -> None:
        target = build_goto_target()
        runner.request_mount_goto(
            target,
            tol_arcsec=float(goto_panel.tol_arcsec.value),
            stages=int(goto_panel.stages.value),
            gain=float(goto_panel.gain.value),
            settle_s=float(goto_panel.settle_s.value),
            delay_us=int(goto_panel.delay_us.value),
            platesolving_feedback=bool(goto_panel.feedback.value),
        )

    def enqueue_goto_calib() -> None:
        params = {
            "n_samples": int(goto_panel.calib_samples.value),
            "max_radius_deg": float(goto_panel.calib_radius.value),
            "delay_us": int(goto_panel.delay_us.value),
        }
        runner.request_goto_calibrate(params)

    def enqueue_goto_autocal() -> None:
        runner.request_goto_autocalibrate()

    def enqueue_goto_cancel() -> None:
        runner.request_goto_cancel()

    goto_panel.btn_sync.on_click(lambda _btn: enqueue_goto_sync())
    goto_panel.btn_goto.on_click(lambda _btn: enqueue_goto_run())
    goto_panel.btn_calib.on_click(lambda _btn: enqueue_goto_calib())
    goto_panel.btn_autocal.on_click(lambda _btn: enqueue_goto_autocal())
    goto_panel.btn_cancel.on_click(lambda _btn: enqueue_goto_cancel())
