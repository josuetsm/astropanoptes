from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W

from config import AppConfig


@dataclass
class MountPanelHandles:
    serial_port: W.Text
    baudrate: W.BoundedIntText
    connect_btn: W.Button
    disconnect_btn: W.Button


@dataclass
class ManualMountHandles:
    ms_az: W.Dropdown
    ms_alt: W.Dropdown
    steps_az: W.BoundedIntText
    delay_az: W.BoundedIntText
    steps_alt: W.BoundedIntText
    delay_alt: W.BoundedIntText
    ramp_enable: W.Checkbox
    ramp_frac: W.BoundedFloatText
    ramp_min_steps: W.BoundedIntText
    ramp_start_scale: W.BoundedFloatText
    ramp_segments: W.BoundedIntText
    btn_az_left: W.Button
    btn_az_right: W.Button
    btn_alt_up: W.Button
    btn_alt_down: W.Button
    btn_stop: W.Button


@dataclass
class GotoPanelHandles:
    status: W.HTML
    mode: W.Dropdown
    name: W.Text
    planet: W.Dropdown
    ra: W.BoundedFloatText
    dec: W.BoundedFloatText
    az: W.BoundedFloatText
    alt: W.BoundedFloatText
    tol_arcsec: W.BoundedFloatText
    stages: W.BoundedIntText
    gain: W.BoundedFloatText
    settle_s: W.BoundedFloatText
    feedback: W.Checkbox
    calib_samples: W.BoundedIntText
    calib_radius: W.BoundedFloatText
    delay_us: W.BoundedIntText
    btn_sync: W.Button
    btn_goto: W.Button
    btn_calib: W.Button
    btn_autocal: W.Button
    btn_cancel: W.Button
    target_box: W.VBox
    box_name: W.HBox
    box_planet: W.HBox
    box_radec: W.HBox
    box_altaz: W.HBox


def build_mount_panel(cfg: AppConfig) -> tuple[W.Widget, MountPanelHandles]:
    mount_cfg = cfg.mount
    serial_port = W.Text(value=str(mount_cfg.port), description="Port", layout=W.Layout(width="520px"))
    baudrate = W.BoundedIntText(
        value=int(mount_cfg.baudrate),
        min=9600,
        max=2000000,
        step=9600,
        description="Baud",
        layout=W.Layout(width="260px"),
    )

    connect_btn = W.Button(description="Connect", button_style="success")
    disconnect_btn = W.Button(description="Disconnect")

    widget = W.VBox(
        [
            W.HTML("<b>Mount</b>"),
            W.HBox([serial_port, baudrate]),
            W.HBox([connect_btn, disconnect_btn]),
            W.HTML("<small>Tip: si no conecta, verifica que el puerto no esté abierto en otro programa.</small>"),
        ]
    )

    handles = MountPanelHandles(
        serial_port=serial_port,
        baudrate=baudrate,
        connect_btn=connect_btn,
        disconnect_btn=disconnect_btn,
    )
    return widget, handles


def build_manual_mount(cfg: AppConfig) -> tuple[W.Widget, ManualMountHandles]:
    mount_cfg = cfg.mount
    ms_opts = [8, 16, 32, 64]

    ms_az = W.Dropdown(
        options=ms_opts,
        value=int(mount_cfg.ms_az),
        description="MS AZ",
        layout=W.Layout(width="170px"),
    )
    ms_alt = W.Dropdown(
        options=ms_opts,
        value=int(mount_cfg.ms_alt),
        description="MS ALT",
        layout=W.Layout(width="170px"),
    )

    steps_az = W.BoundedIntText(
        value=int(mount_cfg.slew_steps_az),
        min=1,
        max=500000,
        step=10,
        description="AZ steps",
        layout=W.Layout(width="180px"),
    )
    delay_az = W.BoundedIntText(
        value=int(mount_cfg.slew_delay_us_az),
        min=50,
        max=200000,
        step=50,
        description="AZ delay (µs)",
        layout=W.Layout(width="220px"),
    )

    steps_alt = W.BoundedIntText(
        value=int(mount_cfg.slew_steps_alt),
        min=1,
        max=500000,
        step=10,
        description="ALT steps",
        layout=W.Layout(width="190px"),
    )
    delay_alt = W.BoundedIntText(
        value=int(mount_cfg.slew_delay_us_alt),
        min=50,
        max=200000,
        step=50,
        description="ALT delay (µs)",
        layout=W.Layout(width="230px"),
    )

    ramp_enable = W.Checkbox(
        value=bool(getattr(mount_cfg, "manual_ramp_enable", True)),
        description="Ramp enable",
        indent=False,
    )
    ramp_frac = W.BoundedFloatText(
        value=float(getattr(mount_cfg, "manual_ramp_frac", 0.2)),
        min=0.0,
        max=0.8,
        step=0.05,
        description="Ramp frac",
        layout=W.Layout(width="180px"),
    )
    ramp_min_steps = W.BoundedIntText(
        value=int(getattr(mount_cfg, "manual_ramp_min_steps", 120)),
        min=0,
        max=500000,
        step=10,
        description="Min steps",
        layout=W.Layout(width="180px"),
    )
    ramp_start_scale = W.BoundedFloatText(
        value=float(getattr(mount_cfg, "manual_ramp_start_delay_scale", 2.0)),
        min=1.0,
        max=6.0,
        step=0.1,
        description="Start x",
        layout=W.Layout(width="160px"),
    )
    ramp_segments = W.BoundedIntText(
        value=int(getattr(mount_cfg, "manual_ramp_segments", 8)),
        min=1,
        max=32,
        step=1,
        description="Segments",
        layout=W.Layout(width="170px"),
    )

    btn_az_left = W.Button(description="AZ ←", layout=W.Layout(width="80px"))
    btn_az_right = W.Button(description="AZ →", layout=W.Layout(width="80px"))
    btn_alt_up = W.Button(description="ALT ↑", layout=W.Layout(width="80px"))
    btn_alt_down = W.Button(description="ALT ↓", layout=W.Layout(width="80px"))
    btn_stop = W.Button(description="STOP", button_style="danger", layout=W.Layout(width="90px"))

    widget = W.VBox(
        [
            W.HTML("<b>Manual Mount Control</b>"),
            W.HBox([ms_az, ms_alt]),
            W.HBox(
                [
                    W.VBox(
                        [
                            W.HTML("<b>AZ</b>"),
                            steps_az,
                            delay_az,
                            W.HBox([btn_az_left, btn_az_right]),
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                    W.VBox(
                        [
                            W.HTML("<b>ALT</b>"),
                            steps_alt,
                            delay_alt,
                            W.HBox([btn_alt_up, btn_alt_down]),
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                    W.VBox(
                        [
                            W.HTML("<b>Safety</b>"),
                            btn_stop,
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                    W.VBox(
                        [
                            W.HTML("<b>Rampa</b>"),
                            ramp_enable,
                            ramp_frac,
                            ramp_min_steps,
                            ramp_start_scale,
                            ramp_segments,
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                ]
            ),
        ],
        layout=W.Layout(border="1px solid #eee", padding="8px", gap="6px"),
    )

    handles = ManualMountHandles(
        ms_az=ms_az,
        ms_alt=ms_alt,
        steps_az=steps_az,
        delay_az=delay_az,
        steps_alt=steps_alt,
        delay_alt=delay_alt,
        ramp_enable=ramp_enable,
        ramp_frac=ramp_frac,
        ramp_min_steps=ramp_min_steps,
        ramp_start_scale=ramp_start_scale,
        ramp_segments=ramp_segments,
        btn_az_left=btn_az_left,
        btn_az_right=btn_az_right,
        btn_alt_up=btn_alt_up,
        btn_alt_down=btn_alt_down,
        btn_stop=btn_stop,
    )
    return widget, handles


def build_goto_panel(cfg: AppConfig) -> tuple[W.Widget, GotoPanelHandles]:
    goto_cfg = cfg.goto
    status = W.HTML("<b>GoTo</b>: IDLE")

    mode = W.Dropdown(
        options=[
            ("Objeto / nombre", "name"),
            ("Planeta", "planet"),
            ("RA/DEC", "radec"),
            ("Alt/Az", "altaz"),
        ],
        value="name",
        description="Modo:",
        disabled=False,
        layout=W.Layout(width="260px"),
    )

    name = W.Text(value="", description="Obj:", layout=W.Layout(width="360px"))
    planet = W.Dropdown(
        options=["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"],
        value="mars",
        description="Planeta:",
        layout=W.Layout(width="260px"),
    )

    ra = W.BoundedFloatText(
        value=0.0,
        min=0.0,
        max=360.0,
        step=0.1,
        description="RA°:",
        layout=W.Layout(width="200px"),
    )
    dec = W.BoundedFloatText(
        value=0.0,
        min=-90.0,
        max=90.0,
        step=0.1,
        description="Dec°:",
        layout=W.Layout(width="200px"),
    )
    az = W.BoundedFloatText(
        value=0.0,
        min=0.0,
        max=360.0,
        step=0.1,
        description="Az°:",
        layout=W.Layout(width="200px"),
    )
    alt = W.BoundedFloatText(
        value=45.0,
        min=0.0,
        max=90.0,
        step=0.1,
        description="Alt°:",
        layout=W.Layout(width="200px"),
    )

    tol_arcsec = W.BoundedFloatText(
        value=float(goto_cfg.tol_arcsec),
        min=0.5,
        max=3600.0,
        step=0.5,
        description="Tol (arcsec):",
        layout=W.Layout(width="220px"),
    )
    stages = W.BoundedIntText(
        value=int(goto_cfg.stages),
        min=1,
        max=50,
        step=1,
        description="Etapas:",
        layout=W.Layout(width="160px"),
    )
    gain = W.BoundedFloatText(
        value=float(goto_cfg.gain),
        min=0.1,
        max=2.0,
        step=0.05,
        description="Gain:",
        layout=W.Layout(width="160px"),
    )
    settle_s = W.BoundedFloatText(
        value=float(goto_cfg.settle_s),
        min=0.0,
        max=10.0,
        step=0.05,
        description="Settle(s):",
        layout=W.Layout(width="170px"),
    )
    feedback = W.Checkbox(value=bool(goto_cfg.platesolving_feedback), description="Feedback platesolving", indent=False)

    calib_samples = W.BoundedIntText(
        value=int(goto_cfg.calib_samples),
        min=1,
        max=80,
        step=1,
        description="Muestras:",
        layout=W.Layout(width="180px"),
    )
    calib_radius = W.BoundedFloatText(
        value=float(goto_cfg.calib_max_radius_deg),
        min=0.1,
        max=60.0,
        step=0.1,
        description="Rango°:",
        layout=W.Layout(width="180px"),
    )

    delay_us = W.BoundedIntText(
        value=int(goto_cfg.slew_delay_us),
        min=50,
        max=50000,
        step=50,
        description="delay_us:",
        layout=W.Layout(width="200px"),
    )

    btn_sync = W.Button(description="Sync", button_style="info", layout=W.Layout(width="100px"))
    btn_goto = W.Button(description="GoTo", button_style="success", layout=W.Layout(width="100px"))
    btn_calib = W.Button(description="Calibrate", button_style="warning", layout=W.Layout(width="120px"))
    btn_autocal = W.Button(description="AutoCalibrate", button_style="warning", layout=W.Layout(width="140px"))
    btn_cancel = W.Button(description="Cancel", button_style="danger", layout=W.Layout(width="110px"))

    box_name = W.HBox([name])
    box_planet = W.HBox([planet])
    box_radec = W.HBox([ra, dec])
    box_altaz = W.HBox([az, alt])
    target_box = W.VBox([box_name])

    box_buttons = W.HBox([btn_sync, btn_goto, btn_calib, btn_autocal, btn_cancel])
    box_params = W.HBox([tol_arcsec, stages, gain, settle_s, feedback])
    box_calib = W.HBox([calib_samples, calib_radius])
    box_delay = W.HBox([delay_us])

    widget = W.VBox(
        [
            status,
            W.HBox([mode]),
            target_box,
            box_params,
            box_delay,
            box_buttons,
            W.HTML("<hr/>"),
            W.HTML("<b>Calibración</b> (muestras aleatorias)"),
            box_calib,
        ]
    )

    handles = GotoPanelHandles(
        status=status,
        mode=mode,
        name=name,
        planet=planet,
        ra=ra,
        dec=dec,
        az=az,
        alt=alt,
        tol_arcsec=tol_arcsec,
        stages=stages,
        gain=gain,
        settle_s=settle_s,
        feedback=feedback,
        calib_samples=calib_samples,
        calib_radius=calib_radius,
        delay_us=delay_us,
        btn_sync=btn_sync,
        btn_goto=btn_goto,
        btn_calib=btn_calib,
        btn_autocal=btn_autocal,
        btn_cancel=btn_cancel,
        target_box=target_box,
        box_name=box_name,
        box_planet=box_planet,
        box_radec=box_radec,
        box_altaz=box_altaz,
    )
    return widget, handles
