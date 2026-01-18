# ui.py
from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional

import ipywidgets as W
from IPython.display import display

from config import AppConfig
from app_runner import AppRunner
from actions import (
    camera_connect,
    camera_disconnect,
    camera_set_param,
    mount_connect,
    mount_disconnect,
    mount_stop,
    mount_set_microsteps,   # NEW
    mount_move_steps,       # NEW
    tracking_start,
    tracking_stop,
    tracking_set_params,
    stacking_start,
    stacking_stop,
)
from ap_types import Axis
from logging_utils import log_info


def _clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        v = lo
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def _clamp_float(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = lo
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def build_ui(cfg: AppConfig, runner: AppRunner) -> Dict[str, Any]:
    """
    Construye la UI y retorna un dict con:
      - "root": widget raíz
      - "widgets": dict de widgets por id (w_*)
      - "log_out": Output widget para logs
    """
    # -------------------------
    # Logs
    # -------------------------
    w_out_log = W.Output(layout=W.Layout(border="1px solid #ddd", height="180px", overflow="auto"))
    runner.out_log = w_out_log  # conectar runner a este output

    # -------------------------
    # Top Bar (siempre visible)
    # -------------------------
    w_status_camera = W.HTML(value="Camera: <b>DISCONNECTED</b>")
    w_status_mount = W.HTML(value="Mount: <b>DISCONNECTED</b>")
    w_status_tracking = W.HTML(value="Tracking: <b>OFF</b>")
    w_status_stacking = W.HTML(value="Stacking: <b>OFF</b>")

    w_lbl_fps = W.Label(value="FPS cap: 0.0 | view: 0.0 | loop: 0.0")
    w_lbl_frame_ms = W.Label(value="frame_ms: 0.0")

    w_btn_connect_camera = W.Button(description="Connect Camera", button_style="success")
    w_btn_disconnect_camera = W.Button(description="Disconnect Camera", button_style="")

    w_btn_connect_mount = W.Button(description="Connect Mount", button_style="success")
    w_btn_disconnect_mount = W.Button(description="Disconnect Mount", button_style="")

    # Tracking ya NO es placeholder
    w_btn_tracking_toggle = W.ToggleButton(description="Tracking", value=False, disabled=False)

    w_btn_stacking_toggle = W.ToggleButton(description="Stacking", value=False, disabled=False)
    w_btn_save_quick = W.Button(description="Save Stack", disabled=True)

    top_left = W.VBox([w_status_camera, w_status_mount, w_status_tracking, w_status_stacking])
    top_mid = W.VBox([w_lbl_fps, w_lbl_frame_ms])
    top_right = W.HBox(
        [
            w_btn_connect_camera,
            w_btn_disconnect_camera,
            w_btn_connect_mount,
            w_btn_disconnect_mount,
            w_btn_tracking_toggle,
            w_btn_stacking_toggle,
            w_btn_save_quick,
        ],
        layout=W.Layout(flex_flow="row wrap", align_items="center"),
    )
    w_top_bar = W.HBox([top_left, top_mid, top_right], layout=W.Layout(justify_content="space-between"))

    # -------------------------
    # Live View (siempre visible)
    # -------------------------
    w_img_live = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    w_live_box = W.VBox([w_img_live])

    # -------------------------
    # Manual Mount Control (siempre visible)  [MOVE + microsteps + delay_us]
    # -------------------------
    ms_opts = [8, 16, 32, 64]

    w_dd_ms_az = W.Dropdown(
        options=ms_opts,
        value=int(getattr(cfg.mount, "ms_az", 64)),
        description="MS AZ",
        layout=W.Layout(width="170px"),
    )
    w_dd_ms_alt = W.Dropdown(
        options=ms_opts,
        value=int(getattr(cfg.mount, "ms_alt", 64)),
        description="MS ALT",
        layout=W.Layout(width="170px"),
    )
    w_btn_apply_ms = W.Button(description="Apply MS", layout=W.Layout(width="110px"))

    w_steps_az = W.BoundedIntText(
        value=_clamp_int(getattr(cfg.mount, "slew_steps_az", 600), 1, 500000),
        min=1,
        max=500000,
        step=10,
        description="AZ steps",
        layout=W.Layout(width="180px"),
    )
    w_delay_az = W.BoundedIntText(
        value=_clamp_int(getattr(cfg.mount, "slew_delay_us_az", 1800), 50, 200000),
        min=50,
        max=200000,
        step=50,
        description="AZ delay (µs)",
        layout=W.Layout(width="220px"),
    )

    w_steps_alt = W.BoundedIntText(
        value=_clamp_int(getattr(cfg.mount, "slew_steps_alt", 600), 1, 500000),
        min=1,
        max=500000,
        step=10,
        description="ALT steps",
        layout=W.Layout(width="190px"),
    )
    w_delay_alt = W.BoundedIntText(
        value=_clamp_int(getattr(cfg.mount, "slew_delay_us_alt", 1800), 50, 200000),
        min=50,
        max=200000,
        step=50,
        description="ALT delay (µs)",
        layout=W.Layout(width="230px"),
    )

    # Botones manuales (como tu script)
    w_btn_az_left = W.Button(description="AZ ←", layout=W.Layout(width="80px"))
    w_btn_az_right = W.Button(description="AZ →", layout=W.Layout(width="80px"))
    w_btn_alt_up = W.Button(description="ALT ↑", layout=W.Layout(width="80px"))
    w_btn_alt_down = W.Button(description="ALT ↓", layout=W.Layout(width="80px"))
    w_btn_stop = W.Button(description="STOP", button_style="danger", layout=W.Layout(width="90px"))

    w_mount_controls = W.VBox(
        [
            W.HTML("<b>Manual Mount Control</b>"),
            W.HBox([w_dd_ms_az, w_dd_ms_alt, w_btn_apply_ms]),
            W.HBox(
                [
                    W.VBox(
                        [
                            W.HTML("<b>AZ</b>"),
                            w_steps_az,
                            w_delay_az,
                            W.HBox([w_btn_az_left, w_btn_az_right]),
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                    W.VBox(
                        [
                            W.HTML("<b>ALT</b>"),
                            w_steps_alt,
                            w_delay_alt,
                            W.HBox([w_btn_alt_up, w_btn_alt_down]),
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                    W.VBox(
                        [
                            W.HTML("<b>Safety</b>"),
                            w_btn_stop,
                        ],
                        layout=W.Layout(border="1px solid #eee", padding="6px"),
                    ),
                ]
            ),
        ],
        layout=W.Layout(border="1px solid #eee", padding="8px", gap="6px"),
    )

    # -------------------------
    # Camera Tab
    # -------------------------
    w_dd_camera_id = W.Dropdown(
        options=[("0", 0)],
        value=int(cfg.camera.camera_index),
        description="Camera",
        layout=W.Layout(width="220px"),
    )

    w_bi_exp_ms = W.BoundedIntText(
        value=_clamp_int(cfg.camera.exp_ms, 1, 60000),
        min=1,
        max=60000,
        step=1,
        description="Exp (ms)",
        layout=W.Layout(width="240px"),
    )

    w_bi_gain = W.BoundedIntText(
        value=_clamp_int(cfg.camera.gain, 0, 500),
        min=0,
        max=500,
        step=1,
        description="Gain",
        layout=W.Layout(width="220px"),
    )

    w_cb_auto_gain = W.Checkbox(value=bool(cfg.camera.auto_gain), description="Auto Gain")

    w_dd_img_format = W.Dropdown(
        options=[("RAW16", "RAW16"), ("RAW8", "RAW8"), ("RGB24", "RGB24"), ("MONO8", "MONO8")],
        value=str(cfg.camera.img_format),
        description="Format",
        layout=W.Layout(width="240px"),
    )

    # Preview controls (mantengo aquí)
    w_bt_view_hz = W.BoundedFloatText(
        value=float(cfg.preview.view_hz),
        min=0.5,
        max=60.0,
        step=0.5,
        description="View Hz",
        layout=W.Layout(width="240px"),
    )
    w_dd_ds = W.Dropdown(
        options=[("1x", 1), ("2x", 2), ("3x", 3), ("4x", 4)],
        value=int(cfg.preview.ds),
        description="DS",
        layout=W.Layout(width="200px"),
    )
    w_bi_jpeg_q = W.BoundedIntText(
        value=int(cfg.preview.jpeg_quality),
        min=10,
        max=100,
        step=1,
        description="JPEG Q",
        layout=W.Layout(width="240px"),
    )

    w_bt_plo = W.BoundedFloatText(
        value=float(cfg.preview.stretch_plo),
        min=0.0,
        max=30.0,
        step=0.5,
        description="Pctl Lo",
        layout=W.Layout(width="240px"),
    )
    w_bt_phi = W.BoundedFloatText(
        value=float(cfg.preview.stretch_phi),
        min=70.0,
        max=100.0,
        step=0.1,
        description="Pctl Hi",
        layout=W.Layout(width="240px"),
    )

    cam_grid = W.VBox(
        [
            W.HBox([w_dd_camera_id, w_dd_img_format]),
            W.HBox([w_bi_exp_ms, w_bi_gain, w_cb_auto_gain]),
            W.HBox([w_bt_view_hz, w_dd_ds, w_bi_jpeg_q]),
            W.HBox([w_bt_plo, w_bt_phi]),
        ]
    )
    w_tab_camera = W.VBox([W.HTML("<b>Camera</b>"), cam_grid])

    # -------------------------
    # Mount Tab (real)
    # -------------------------
    default_port = str(getattr(cfg.mount, "port", "/dev/cu.usbserial-1130"))
    default_baud = int(getattr(cfg.mount, "baud", 115200))

    w_txt_serial_port = W.Text(value=default_port, description="Port", layout=W.Layout(width="520px"))
    w_bi_baudrate = W.BoundedIntText(
        value=_clamp_int(default_baud, 9600, 2000000),
        min=9600,
        max=2000000,
        step=9600,
        description="Baud",
        layout=W.Layout(width="260px"),
    )

    w_btn_mount_connect_tab = W.Button(description="Connect", button_style="success")
    w_btn_mount_disconnect_tab = W.Button(description="Disconnect")

    w_tab_mount = W.VBox(
        [
            W.HTML("<b>Mount</b>"),
            W.HBox([w_txt_serial_port, w_bi_baudrate]),
            W.HBox([w_btn_mount_connect_tab, w_btn_mount_disconnect_tab]),
            W.HTML("<small>Tip: si no conecta, verifica que el puerto no esté abierto en otro programa.</small>"),
        ]
    )

    # ============================================================
    # TAB TRACKING (NUEVO)  [TextFloat / TextInt, sin sliders]
    # ============================================================
    w_dd_track_mode = W.Dropdown(
        description="Mode",
        options=["AUTO", "STARS", "PLANET"],
        value="AUTO",
        layout=W.Layout(width="200px"),
    )

    # parámetros mínimos (ajusta rangos si quieres)
    w_tf_sigma_hp = W.BoundedFloatText(
        description="sigma_hp",
        value=float(getattr(getattr(cfg, "tracking", object()), "sigma_hp", 10.0)),
        min=0.5,
        max=300.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    w_tf_resp_min = W.BoundedFloatText(
        description="resp_min",
        value=float(getattr(getattr(cfg, "tracking", object()), "resp_min", 0.06)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="260px"),
    )

    # botones redundantes en tab (útiles aunque tengas toggle arriba)
    w_btn_track_start = W.Button(description="Start", button_style="success", layout=W.Layout(width="110px"))
    w_btn_track_stop = W.Button(description="Stop", button_style="warning", layout=W.Layout(width="110px"))

    w_lbl_track_info = W.HTML(value="Tracking: idle")

    def _send_track_params(_=None):
        runner.enqueue(
            tracking_set_params(
                {
                    "mode": str(w_dd_track_mode.value),
                    "sigma_hp": float(w_tf_sigma_hp.value),
                    "resp_min": float(w_tf_resp_min.value),
                }
            )
        )

    def _on_track_start(_btn):
        _send_track_params()
        runner.enqueue(tracking_start())
        # refleja en toggle
        try:
            w_btn_tracking_toggle.value = True
        except Exception:
            pass

    def _on_track_stop(_btn):
        runner.enqueue(tracking_stop())
        try:
            w_btn_tracking_toggle.value = False
        except Exception:
            pass

    w_btn_track_start.on_click(_on_track_start)
    w_btn_track_stop.on_click(_on_track_stop)

    w_dd_track_mode.observe(_send_track_params, names="value")
    w_tf_sigma_hp.observe(_send_track_params, names="value")
    w_tf_resp_min.observe(_send_track_params, names="value")

    w_tab_tracking = W.VBox(
        [
            W.HTML("<b>Tracking</b>"),
            W.HBox([w_btn_track_start, w_btn_track_stop, w_dd_track_mode]),
            W.HBox([w_tf_sigma_hp, w_tf_resp_min]),
            w_lbl_track_info,
        ]
    )

    # -------------------------
    # Placeholder tabs
    # -------------------------
    w_img_stack = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    w_tab_stacking = W.VBox([W.HTML("<b>Stacking</b>"), w_img_stack])
    w_tab_platesolve = W.VBox([W.HTML("<b>PlateSolve</b> (coming soon)")])
    w_tab_goto = W.VBox([W.HTML("<b>GoTo</b> (coming soon)")])

    w_tab_logs = W.VBox([W.HTML("<b>Logs</b>"), w_out_log])

    w_tabs = W.Tab(children=[w_tab_camera, w_tab_mount, w_tab_tracking, w_tab_stacking, w_tab_platesolve, w_tab_goto, w_tab_logs])
    for i, name in enumerate(["Camera", "Mount", "Tracking", "Stacking", "PlateSolve", "GoTo", "Logs"]):
        w_tabs.set_title(i, name)

    # -------------------------
    # Root layout
    # -------------------------
    root = W.VBox([w_top_bar, w_live_box, w_mount_controls, w_tabs], layout=W.Layout(gap="10px"))

    # -------------------------
    # Bindings: Camera buttons
    # -------------------------
    def _on_connect_camera(_btn):
        idx = int(w_dd_camera_id.value)
        w_status_camera.value = "Camera: <b>CONNECTING...</b>"
        runner.enqueue(camera_connect(idx))

    def _on_disconnect_camera(_btn):
        runner.enqueue(camera_disconnect())

    w_btn_connect_camera.on_click(_on_connect_camera)
    w_btn_disconnect_camera.on_click(_on_disconnect_camera)

    # -------------------------
    # Bindings: Camera params
    # -------------------------
    def _on_exp(change):
        runner.enqueue(camera_set_param("exp_ms", int(change["new"])))

    def _on_gain(change):
        runner.enqueue(camera_set_param("gain", int(change["new"])))

    def _on_auto_gain(change):
        runner.enqueue(camera_set_param("auto_gain", bool(change["new"])))

    def _on_img_format(change):
        runner.enqueue(camera_set_param("img_format", str(change["new"])))

    def _on_view_hz(change):
        runner.enqueue(camera_set_param("preview_view_hz", float(change["new"])))

    def _on_ds(change):
        runner.enqueue(camera_set_param("preview_ds", int(change["new"])))

    def _on_jpeg_q(change):
        runner.enqueue(camera_set_param("preview_jpeg_quality", int(change["new"])))

    def _on_plo(change):
        runner.enqueue(camera_set_param("preview_stretch_plo", float(change["new"])))

    def _on_phi(change):
        runner.enqueue(camera_set_param("preview_stretch_phi", float(change["new"])))

    w_bi_exp_ms.observe(_on_exp, names="value")
    w_bi_gain.observe(_on_gain, names="value")
    w_cb_auto_gain.observe(_on_auto_gain, names="value")
    w_dd_img_format.observe(_on_img_format, names="value")

    w_bt_view_hz.observe(_on_view_hz, names="value")
    w_dd_ds.observe(_on_ds, names="value")
    w_bi_jpeg_q.observe(_on_jpeg_q, names="value")
    w_bt_plo.observe(_on_plo, names="value")
    w_bt_phi.observe(_on_phi, names="value")

    # -------------------------
    # Bindings: Mount connect/disconnect
    # -------------------------
    def _enqueue_mount_connect():
        port = str(w_txt_serial_port.value).strip()
        baud = int(w_bi_baudrate.value)
        runner.enqueue(mount_connect(port, baud))

    def _enqueue_mount_disconnect():
        runner.enqueue(mount_disconnect())

    def _on_connect_mount(_btn):
        _enqueue_mount_connect()

    def _on_disconnect_mount(_btn):
        _enqueue_mount_disconnect()

    w_btn_connect_mount.on_click(_on_connect_mount)
    w_btn_disconnect_mount.on_click(_on_disconnect_mount)
    w_btn_mount_connect_tab.on_click(_on_connect_mount)
    w_btn_mount_disconnect_tab.on_click(_on_disconnect_mount)

    # -------------------------
    # Bindings: Manual mount (MOVE steps/delay + MS)
    # -------------------------
    def _enqueue_apply_ms():
        az_div = int(w_dd_ms_az.value)
        alt_div = int(w_dd_ms_alt.value)
        runner.enqueue(mount_set_microsteps(az_div=az_div, alt_div=alt_div))

    def _enqueue_move(axis: Axis, direction: int, steps: int, delay_us: int):
        if steps <= 0 or delay_us <= 0:
            log_info(w_out_log, f"Manual MOVE: invalid params steps={steps} delay_us={delay_us}")
            return
        runner.enqueue(mount_move_steps(axis=axis, direction=direction, steps=steps, delay_us=delay_us))

    def _on_apply_ms(_btn):
        _enqueue_apply_ms()

    def _on_az_left(_btn):
        _enqueue_move(Axis.AZ, -1, int(w_steps_az.value), int(w_delay_az.value))

    def _on_az_right(_btn):
        _enqueue_move(Axis.AZ, +1, int(w_steps_az.value), int(w_delay_az.value))

    def _on_alt_up(_btn):
        _enqueue_move(Axis.ALT, +1, int(w_steps_alt.value), int(w_delay_alt.value))

    def _on_alt_down(_btn):
        _enqueue_move(Axis.ALT, -1, int(w_steps_alt.value), int(w_delay_alt.value))

    def _on_stop(_btn):
        runner.enqueue(mount_stop())

    w_btn_apply_ms.on_click(_on_apply_ms)
    w_btn_az_left.on_click(_on_az_left)
    w_btn_az_right.on_click(_on_az_right)
    w_btn_alt_up.on_click(_on_alt_up)
    w_btn_alt_down.on_click(_on_alt_down)
    w_btn_stop.on_click(_on_stop)

    # -------------------------
    # Bindings: Tracking toggle (Top Bar)
    # -------------------------
    def _on_tracking_toggle(change):
        on = bool(change["new"])
        if on:
            # empujar params actuales antes de iniciar
            _send_track_params()
            runner.enqueue(tracking_start())
        else:
            runner.enqueue(tracking_stop())

    w_btn_tracking_toggle.observe(_on_tracking_toggle, names="value")

    # -------------------------
    # Bindings: Stacking toggle (Top Bar)
    # -------------------------
    def _on_stacking_toggle(change):
        on = bool(change["new"])
        if on:
            runner.enqueue(stacking_start())
        else:
            runner.enqueue(stacking_stop())

    w_btn_stacking_toggle.observe(_on_stacking_toggle, names="value")

    widgets = {
        # top bar
        "w_status_camera": w_status_camera,
        "w_status_mount": w_status_mount,
        "w_status_tracking": w_status_tracking,
        "w_status_stacking": w_status_stacking,
        "w_lbl_fps": w_lbl_fps,
        "w_lbl_frame_ms": w_lbl_frame_ms,
        "w_btn_connect_camera": w_btn_connect_camera,
        "w_btn_disconnect_camera": w_btn_disconnect_camera,
        "w_btn_connect_mount": w_btn_connect_mount,
        "w_btn_disconnect_mount": w_btn_disconnect_mount,
        "w_btn_tracking_toggle": w_btn_tracking_toggle,
        "w_btn_stacking_toggle": w_btn_stacking_toggle,
        # live
        "w_img_live": w_img_live,
        # stacking tab
        "w_img_stack": w_img_stack,
        # manual mount control
        "w_dd_ms_az": w_dd_ms_az,
        "w_dd_ms_alt": w_dd_ms_alt,
        "w_btn_apply_ms": w_btn_apply_ms,
        "w_steps_az": w_steps_az,
        "w_delay_az": w_delay_az,
        "w_steps_alt": w_steps_alt,
        "w_delay_alt": w_delay_alt,
        "w_btn_az_left": w_btn_az_left,
        "w_btn_az_right": w_btn_az_right,
        "w_btn_alt_up": w_btn_alt_up,
        "w_btn_alt_down": w_btn_alt_down,
        "w_btn_stop": w_btn_stop,
        # camera tab
        "w_dd_camera_id": w_dd_camera_id,
        "w_bi_exp_ms": w_bi_exp_ms,
        "w_bi_gain": w_bi_gain,
        "w_cb_auto_gain": w_cb_auto_gain,
        "w_dd_img_format": w_dd_img_format,
        "w_bt_view_hz": w_bt_view_hz,
        "w_dd_ds": w_dd_ds,
        "w_bi_jpeg_q": w_bi_jpeg_q,
        "w_bt_plo": w_bt_plo,
        "w_bt_phi": w_bt_phi,
        # mount tab
        "w_txt_serial_port": w_txt_serial_port,
        "w_bi_baudrate": w_bi_baudrate,
        "w_btn_mount_connect_tab": w_btn_mount_connect_tab,
        "w_btn_mount_disconnect_tab": w_btn_mount_disconnect_tab,
        # tracking tab widgets
        "w_dd_track_mode": w_dd_track_mode,
        "w_tf_sigma_hp": w_tf_sigma_hp,
        "w_tf_resp_min": w_tf_resp_min,
        "w_btn_track_start": w_btn_track_start,
        "w_btn_track_stop": w_btn_track_stop,
        "w_lbl_track_info": w_lbl_track_info,
        # logs
        "w_out_log": w_out_log,
        "w_tabs": w_tabs,
        "root": root,
    }

    return {"root": root, "widgets": widgets, "log_out": w_out_log}


# -------------------------
# UI update loop
# -------------------------
class UILoop:
    def __init__(self, runner: AppRunner, widgets: Dict[str, Any], max_hz: float = 10.0) -> None:
        self.runner = runner
        self.widgets = widgets
        self.max_hz = max(0.5, float(max_hz))
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thr is not None:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="UILoop", daemon=True)
        self._thr.start()
        log_info(self.widgets.get("w_out_log"), "UI loop: started")

    def stop(self) -> None:
        self._stop.set()
        thr = self._thr
        if thr is not None:
            thr.join(timeout=1.0)
        self._thr = None
        log_info(self.widgets.get("w_out_log"), "UI loop: stopped")

    def _run(self) -> None:
        dt = 1.0 / self.max_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self.tick()
            t1 = time.perf_counter()
            sleep = dt - (t1 - t0)
            if sleep > 0:
                time.sleep(sleep)

    def tick(self) -> None:
        st = self.runner.get_state()

        # status labels
        self.widgets["w_status_camera"].value = f"Camera: <b>{st.camera_status}</b>"
        self.widgets["w_status_mount"].value = f"Mount: <b>{st.mount_status}</b>"

        # Tracking (robusto a ausencia de campos)
        tracking_enabled = bool(getattr(st, "tracking_enabled", False))
        if tracking_enabled:
            mode = str(getattr(st, "tracking_mode", "ON"))
            resp = float(getattr(st, "tracking_resp", 0.0))
            self.widgets["w_status_tracking"].value = f"Tracking: <b>{mode}</b> (resp={resp:.3f})"
        else:
            self.widgets["w_status_tracking"].value = "Tracking: <b>OFF</b>"

        # Stacking placeholder
        if hasattr(st, "stacking_mode"):
            mode = str(getattr(st, "stacking_mode", "IDLE"))
            fps = float(getattr(st, "stacking_fps", 0.0))
            self.widgets["w_status_stacking"].value = f"Stacking: <b>{mode}</b> ({fps:.2f} fps)"
        else:
            self.widgets["w_status_stacking"].value = "Stacking: <b>OFF</b>"

        self.widgets["w_lbl_fps"].value = (
            f"FPS cap: {st.fps_capture:.2f} | view: {st.fps_view:.2f} | loop: {st.fps_control_loop:.2f}"
        )
        self.widgets["w_lbl_frame_ms"].value = f"frame_ms: {st.frame_ms:.2f}"

        # tracking info panel (si existe)
        if "w_lbl_track_info" in self.widgets:
            if tracking_enabled:
                mode = str(getattr(st, "tracking_mode", "ON"))
                resp = float(getattr(st, "tracking_resp", 0.0))
                dx = float(getattr(st, "tracking_dx", 0.0))
                dy = float(getattr(st, "tracking_dy", 0.0))
                raz = float(getattr(st, "tracking_rate_az", 0.0))
                ralt = float(getattr(st, "tracking_rate_alt", 0.0))
                self.widgets["w_lbl_track_info"].value = (
                    f"<b>Tracking</b>: {mode} | resp={resp:.3f} | "
                    f"dx={dx:+.2f} dy={dy:+.2f} | RATE=({raz:+.1f}, {ralt:+.1f})"
                )
            else:
                self.widgets["w_lbl_track_info"].value = "Tracking: idle"

        # opcional: mantener toggle en sync si cambia por fuera
        if "w_btn_tracking_toggle" in self.widgets:
            try:
                btn = self.widgets["w_btn_tracking_toggle"]
                if bool(btn.value) != tracking_enabled:
                    # OJO: esto evita loops visuales, pero no re-enlaza observe; es suficiente
                    btn.value = tracking_enabled
            except Exception:
                pass
        if "w_btn_stacking_toggle" in self.widgets:
            try:
                btn = self.widgets["w_btn_stacking_toggle"]
                stacking_enabled = bool(getattr(st, "stacking_enabled", False))
                if bool(btn.value) != stacking_enabled:
                    btn.value = stacking_enabled
            except Exception:
                pass

        jpg = self.runner.get_latest_preview_jpeg()
        if jpg:
            self.widgets["w_img_live"].value = jpg

        stack_jpg = getattr(st, "stacking_preview_jpeg", None)
        if stack_jpg and "w_img_stack" in self.widgets:
            self.widgets["w_img_stack"].value = stack_jpg


def show_ui(cfg: AppConfig, runner: AppRunner, *, start_loops: bool = True, ui_hz: float = 10.0):
    built = build_ui(cfg, runner)
    display(built["root"])

    ui_loop = UILoop(runner, built["widgets"], max_hz=float(ui_hz))
    if start_loops:
        ui_loop.start()

    return built, ui_loop
