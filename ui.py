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
    mount_set_microsteps,
    mount_move_steps,
    mount_sync,
    mount_goto,
    goto_calibrate,
    goto_cancel,
    tracking_start,
    tracking_stop,
    tracking_set_params,
    stacking_start,
    stacking_stop,
    stacking_reset,
    stacking_save,
    hotpix_calibrate,
    platesolve_run,
    platesolve_set_params,
    live_sep_set_params,
)
from ap_types import Axis
from logging_utils import log_info, log_error


def _clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception as exc:
        log_error(None, f"UI: failed to parse int input {x!r}", exc, throttle_s=5.0, throttle_key="ui_clamp_int")
        v = lo
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def _clamp_float(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception as exc:
        log_error(None, f"UI: failed to parse float input {x!r}", exc, throttle_s=5.0, throttle_key="ui_clamp_float")
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
    platesolve_cfg = cfg.platesolve
    sep_cfg = cfg.sep
    mount_cfg = cfg.mount
    tracking_cfg = cfg.tracking

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
    w_btn_stop_mount = W.Button(description="STOP Mount", button_style="danger")

    # Tracking ya NO es placeholder
    w_btn_tracking_toggle = W.ToggleButton(description="Tracking", value=False, disabled=False)

    w_btn_stacking_toggle = W.ToggleButton(description="Stacking", value=False, disabled=False)
    w_btn_save_quick = W.Button(description="Save Stack", disabled=True)
    # The Save Stack button will trigger saving the current stacked mosaic in both
    # raw (floating point) and stretched (uint8) formats.  See the handler
    # below for implementation.  It remains disabled until stacking is running.

    top_left = W.VBox([w_status_camera, w_status_mount, w_status_tracking, w_status_stacking])
    top_mid = W.VBox([w_lbl_fps, w_lbl_frame_ms])
    top_right = W.HBox(
        [
            w_btn_connect_camera,
            w_btn_disconnect_camera,
            w_btn_connect_mount,
            w_btn_disconnect_mount,
            w_btn_stop_mount,
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
    w_img_live = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="500px"))

    w_tb_live_sep = W.ToggleButton(
        description="SEP Overlay",
        value=False,
        disabled=False,
        layout=W.Layout(width="140px"),
    )
    w_bi_live_sep_bw = W.BoundedIntText(
        description="sep_bw",
        value=int(sep_cfg.bw),
        min=4,
        max=512,
        step=1,
        layout=W.Layout(width="180px"),
    )
    w_bi_live_sep_bh = W.BoundedIntText(
        description="sep_bh",
        value=int(sep_cfg.bh),
        min=4,
        max=512,
        step=1,
        layout=W.Layout(width="180px"),
    )
    w_tf_live_sep_sigma = W.BoundedFloatText(
        description="sep_sigma",
        value=float(sep_cfg.thresh_sigma),
        min=0.1,
        max=20.0,
        step=0.1,
        layout=W.Layout(width="200px"),
    )
    w_bi_live_sep_minarea = W.BoundedIntText(
        description="sep_minarea",
        value=int(sep_cfg.minarea),
        min=1,
        max=500,
        step=1,
        layout=W.Layout(width="220px"),
    )
    w_bi_live_sep_max_det = W.BoundedIntText(
        description="max_det",
        value=int(platesolve_cfg.max_det),
        min=1,
        max=5000,
        step=5,
        layout=W.Layout(width="200px"),
    )

    def _send_live_sep_params(_=None) -> None:
        runner.enqueue(
            live_sep_set_params(
                enabled=bool(w_tb_live_sep.value),
                sep_bw=int(w_bi_live_sep_bw.value),
                sep_bh=int(w_bi_live_sep_bh.value),
                sep_thresh_sigma=float(w_tf_live_sep_sigma.value),
                sep_minarea=int(w_bi_live_sep_minarea.value),
                max_det=int(w_bi_live_sep_max_det.value),
            )
        )

    # Observers for SEP overlay are attached via the debounce handler defined below.
    # See _debounce_live_sep for details.

    w_live_overlay_controls = W.VBox(
        [
            W.HTML("<b>Live SEP Overlay</b>"),
            W.HBox([w_tb_live_sep, w_bi_live_sep_max_det]),
            W.HBox([w_bi_live_sep_bw, w_bi_live_sep_bh, w_tf_live_sep_sigma, w_bi_live_sep_minarea]),
        ],
        layout=W.Layout(border="1px solid #eee", padding="6px", gap="6px", max_width="980px"),
    )

    w_live_box = W.VBox([w_img_live, w_live_overlay_controls])

    # -------------------------
    # Manual Mount Control (siempre visible)  [MOVE + microsteps + delay_us]
    # -------------------------
    ms_opts = [8, 16, 32, 64]

    w_dd_ms_az = W.Dropdown(
        options=ms_opts,
        value=int(mount_cfg.ms_az),
        description="MS AZ",
        layout=W.Layout(width="170px"),
    )
    w_dd_ms_alt = W.Dropdown(
        options=ms_opts,
        value=int(mount_cfg.ms_alt),
        description="MS ALT",
        layout=W.Layout(width="170px"),
    )
    w_btn_apply_ms = W.Button(description="Apply MS", layout=W.Layout(width="110px"))
    # Hide the Apply MS button; microstep divisions will be applied automatically via debounce
    w_btn_apply_ms.layout.display = "none"

    w_steps_az = W.BoundedIntText(
        value=_clamp_int(mount_cfg.slew_steps_az, 1, 500000),
        min=1,
        max=500000,
        step=10,
        description="AZ steps",
        layout=W.Layout(width="180px"),
    )
    w_delay_az = W.BoundedIntText(
        value=_clamp_int(mount_cfg.slew_delay_us_az, 50, 200000),
        min=50,
        max=200000,
        step=50,
        description="AZ delay (µs)",
        layout=W.Layout(width="220px"),
    )

    w_steps_alt = W.BoundedIntText(
        value=_clamp_int(mount_cfg.slew_steps_alt, 1, 500000),
        min=1,
        max=500000,
        step=10,
        description="ALT steps",
        layout=W.Layout(width="190px"),
    )
    w_delay_alt = W.BoundedIntText(
        value=_clamp_int(mount_cfg.slew_delay_us_alt, 50, 200000),
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

    # -------------------------------------------------------------------------
    # Debounce logic for microstep changes.
    #
    # We wait briefly (0.3s) after a change to either MS AZ or MS ALT before
    # sending a mount_set_microsteps action to the runner. Each subsequent change
    # within the debounce interval resets the timer, ensuring we only send the
    # final selected values.
    _ms_timer: Optional[threading.Timer] = None  # type: ignore

    def _debounce_set_ms(change: Any = None) -> None:
        nonlocal _ms_timer
        # cancel any existing timer
        if _ms_timer is not None:
            try:
                _ms_timer.cancel()
            except Exception:
                pass
        # capture current values
        az_val = int(w_dd_ms_az.value)
        alt_val = int(w_dd_ms_alt.value)

        def _send_ms() -> None:
            runner.enqueue(mount_set_microsteps(az_div=az_val, alt_div=alt_val))

        # create and start new timer
        _ms_timer = threading.Timer(0.3, _send_ms)
        _ms_timer.start()

    w_dd_ms_az.observe(_debounce_set_ms, names="value")
    w_dd_ms_alt.observe(_debounce_set_ms, names="value")

    # -------------------------------------------------------------------------
    # Debounce logic for Live SEP overlay parameters.
    #
    # Changing any of the SEP overlay controls rapidly can enqueue a large number
    # of live_sep_set_params actions.  To prevent flooding the runner, wait
    # briefly (0.3s) after the last change before sending the action.  Each
    # subsequent change within the debounce window resets the timer.
    _live_sep_timer: Optional[threading.Timer] = None  # type: ignore

    def _debounce_live_sep(change: Any = None) -> None:
        nonlocal _live_sep_timer
        # Cancel any existing timer
        if _live_sep_timer is not None:
            try:
                _live_sep_timer.cancel()
            except Exception:
                pass
        # Capture current values
        enabled = bool(w_tb_live_sep.value)
        sep_bw = int(w_bi_live_sep_bw.value)
        sep_bh = int(w_bi_live_sep_bh.value)
        sep_thresh_sigma = float(w_tf_live_sep_sigma.value)
        sep_minarea = int(w_bi_live_sep_minarea.value)
        max_det = int(w_bi_live_sep_max_det.value)

        def _send_sep() -> None:
            runner.enqueue(
                live_sep_set_params(
                    enabled=enabled,
                    sep_bw=sep_bw,
                    sep_bh=sep_bh,
                    sep_thresh_sigma=sep_thresh_sigma,
                    sep_minarea=sep_minarea,
                    max_det=max_det,
                )
            )

        _live_sep_timer = threading.Timer(0.3, _send_sep)
        _live_sep_timer.start()

    # Observe changes on SEP overlay controls using the debounce handler
    w_tb_live_sep.observe(_debounce_live_sep, names="value")
    w_bi_live_sep_bw.observe(_debounce_live_sep, names="value")
    w_bi_live_sep_bh.observe(_debounce_live_sep, names="value")
    w_tf_live_sep_sigma.observe(_debounce_live_sep, names="value")
    w_bi_live_sep_minarea.observe(_debounce_live_sep, names="value")
    w_bi_live_sep_max_det.observe(_debounce_live_sep, names="value")

    # -------------------------------------------------------------------------
    # Debounce logic for PlateSolve parameters.
    #
    # The platesolve config is large; changing any field enqueues an action.  To
    # avoid spamming the runner, collect changes and apply them after 0.8s of
    # inactivity.  The existing _ps_send_params function is used to build the
    # payload; the timer simply invokes that function.
    _ps_timer: Optional[threading.Timer] = None  # type: ignore

    def _debounce_ps_params(change: Any = None) -> None:
        nonlocal _ps_timer
        if _ps_timer is not None:
            _ps_timer.cancel()

        def _send_ps() -> None:
            # Call the existing helper to enqueue the parameters.  We ignore the
            # change argument because _ps_send_params reads the current widget
            # values.
            _ps_send_params()

        _ps_timer = threading.Timer(0.8, _send_ps)
        _ps_timer.start()


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

    # The camera stream always uses RAW16 format internally.  Instead of allowing
    # the user to change the image format (which could break the pipeline),
    # present a fixed label.  See imaging.ensure_raw16_bayer() for details.
    w_lbl_img_format = W.HTML(
        value="<b>Format</b>: RAW16 (fixed)",
        layout=W.Layout(width="240px")
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
            # Always display the fixed RAW16 format; do not allow changing it.
            W.HBox([w_dd_camera_id, w_lbl_img_format]),
            W.HBox([w_bi_exp_ms, w_bi_gain, w_cb_auto_gain]),
            W.HBox([w_bt_view_hz, w_bi_jpeg_q]),
            W.HBox([w_bt_plo, w_bt_phi]),
        ]
    )
    w_tab_camera = W.VBox([W.HTML("<b>Camera</b>"), cam_grid])

    # -------------------------
    # Mount Tab (real)
    # -------------------------
    default_port = str(mount_cfg.port)
    default_baud = int(mount_cfg.baudrate)

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
        value=float(tracking_cfg.sigma_hp),
        min=0.5,
        max=300.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    w_tf_resp_min = W.BoundedFloatText(
        description="resp_min",
        value=float(tracking_cfg.resp_min),
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
        try:
            if not bool(w_btn_tracking_toggle.value):
                w_btn_tracking_toggle.value = True
        except Exception as exc:
            log_error(w_out_log, "UI: failed to update tracking toggle (start)", exc, throttle_s=5.0, throttle_key="ui_track_toggle_start")

    def _on_track_stop(_btn):
        try:
            if bool(w_btn_tracking_toggle.value):
                w_btn_tracking_toggle.value = False
        except Exception as exc:
            log_error(w_out_log, "UI: failed to update tracking toggle (stop)", exc, throttle_s=5.0, throttle_key="ui_track_toggle_stop")

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
    # Stacking tab (existing)
    # -------------------------
    w_img_stack = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    w_btn_stack_reset = W.Button(description="Reset Stack", button_style="warning", layout=W.Layout(width="140px"))
    # Stacking control buttons: replicate Start and Stop inside the tab for symmetry
    # with the tracking tab.  These simply toggle the global stacking toggle on
    # the top bar; they do not send actions directly.  Reset remains available
    # separately.
    w_btn_stack_start = W.Button(description="Start", button_style="success", layout=W.Layout(width="110px"))
    w_btn_stack_stop = W.Button(description="Stop", button_style="warning", layout=W.Layout(width="110px"))
    w_btn_hotpix_calib = W.Button(description="Calibrar Hot Pixels", button_style="info", layout=W.Layout(width="200px"))
    # Bindings for stacking buttons in the tab
    def _on_stack_start(_btn):
        try:
            if not bool(w_btn_stacking_toggle.value):
                w_btn_stacking_toggle.value = True
        except Exception as exc:
            log_error(w_out_log, "UI: failed to update stacking toggle (start)", exc, throttle_s=5.0, throttle_key="ui_stack_toggle_start")

    def _on_stack_stop(_btn):
        try:
            if bool(w_btn_stacking_toggle.value):
                w_btn_stacking_toggle.value = False
        except Exception as exc:
            log_error(w_out_log, "UI: failed to update stacking toggle (stop)", exc, throttle_s=5.0, throttle_key="ui_stack_toggle_stop")

    w_btn_stack_start.on_click(_on_stack_start)
    w_btn_stack_stop.on_click(_on_stack_stop)

    def _on_hotpix_calib(_btn):
        hp_cfg = cfg.hotpixels
        runner.enqueue(
            hotpix_calibrate(
                n_frames=int(hp_cfg.calib_frames),
                abs_percentile=float(hp_cfg.calib_abs_percentile),
                var_percentile=float(hp_cfg.calib_var_percentile),
                max_component_area=int(hp_cfg.max_component_area),
                out_path_base=str(hp_cfg.mask_path_base),
            )
        )

    w_btn_hotpix_calib.on_click(_on_hotpix_calib)

    w_tab_stacking = W.VBox([
        W.HTML("<b>Stacking</b>"),
        W.HBox([w_btn_stack_start, w_btn_stack_stop, w_btn_stack_reset, w_btn_hotpix_calib]),
        w_img_stack
    ])

    # ============================================================
    # TAB PLATESOLVE (integrado)
    # ============================================================
    # Instrument params -> se convierten a SI para PlatesolveConfig
    w_tf_ps_focal_mm = W.BoundedFloatText(
        description="focal (mm)",
        value=float(platesolve_cfg.focal_m) * 1000.0,
        min=10.0,
        max=50000.0,
        step=1.0,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_pixel_um = W.BoundedFloatText(
        description="pixel (µm)",
        value=float(platesolve_cfg.pixel_size_m) * 1e6,
        min=0.5,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    w_bi_ps_binning = W.BoundedIntText(
        description="binning",
        value=1,
        min=1,
        max=8,
        step=1,
        layout=W.Layout(width="200px"),
    )

    w_bi_ps_binning.disabled = True  # CameraStream fuerza binning=1; evitamos inconsistencia

    # Solver params (subconjunto razonable)
    w_bi_ps_max_det = W.BoundedIntText(
        description="max_det",
        value=int(platesolve_cfg.max_det),
        min=20,
        max=2000,
        step=10,
        layout=W.Layout(width="220px"),
    )
    w_tf_ps_det_sigma = W.BoundedFloatText(
        description="det_sigma",
        value=float(platesolve_cfg.det_thresh_sigma),
        min=0.5,
        max=50.0,
        step=0.5,
        layout=W.Layout(width="220px"),
    )
    w_bi_ps_minarea = W.BoundedIntText(
        description="minarea",
        value=int(platesolve_cfg.det_minarea),
        min=1,
        max=200,
        step=1,
        layout=W.Layout(width="220px"),
    )
    w_tf_ps_point_sigma = W.BoundedFloatText(
        description="point_sigma",
        value=float(platesolve_cfg.point_sigma),
        min=0.2,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )
    w_tf_ps_gmax = W.BoundedFloatText(
        description="gmax",
        value=float(platesolve_cfg.gmax),
        min=6.0,
        max=20.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )
    w_cb_ps_use_radius = W.Checkbox(
        description="use search_radius_deg",
        value=platesolve_cfg.search_radius_deg is not None,
    )
    w_tf_ps_search_radius_deg = W.BoundedFloatText(
        description="search_radius_deg",
        value=float(platesolve_cfg.search_radius_deg or 2.0),
        min=0.1,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_search_radius_factor = W.BoundedFloatText(
        description="search_radius_factor",
        value=float(platesolve_cfg.search_radius_factor),
        min=0.5,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_theta_step = W.BoundedFloatText(
        description="theta_step (deg)",
        value=float(platesolve_cfg.theta_step_deg),
        min=0.5,
        max=60.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_theta_refine_span = W.BoundedFloatText(
        description="theta_refine_span",
        value=float(platesolve_cfg.theta_refine_span_deg),
        min=0.5,
        max=60.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_theta_refine_step = W.BoundedFloatText(
        description="theta_refine_step",
        value=float(platesolve_cfg.theta_refine_step_deg),
        min=0.1,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    w_tf_ps_match_max = W.BoundedFloatText(
        description="match_max_px",
        value=float(platesolve_cfg.match_max_px),
        min=0.5,
        max=25.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    w_bi_ps_min_inliers = W.BoundedIntText(
        description="min_inliers",
        value=int(platesolve_cfg.min_inliers),
        min=1,
        max=200,
        step=1,
        layout=W.Layout(width="260px"),
    )
    w_bi_ps_guide_n = W.BoundedIntText(
        description="guide_n",
        value=int(platesolve_cfg.guide_n),
        min=0,
        max=20,
        step=1,
        layout=W.Layout(width="220px"),
    )
    w_tf_ps_simbad_radius_arcsec = W.BoundedFloatText(
        description="simbad_radius\"",
        value=float(platesolve_cfg.simbad_radius_arcsec),
        min=0.1,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )

    # Controls
    w_txt_ps_target = W.Text(
        description="target",
        value="",
        placeholder="Ej: 'M42' | '12.5 5.3' (RA/Dec) | ...",
        layout=W.Layout(width="740px"),
    )
    w_btn_ps_solve = W.Button(description="Solve", button_style="success", layout=W.Layout(width="120px"))

    w_tb_ps_auto = W.ToggleButton(
        description="Auto",
        value=bool(platesolve_cfg.auto_solve),
        disabled=False,
        layout=W.Layout(width="110px"),
    )
    w_tf_ps_every_s = W.BoundedFloatText(
        description="every (s)",
        value=float(platesolve_cfg.solve_every_s),
        min=2.0,
        max=600.0,
        step=1.0,
        layout=W.Layout(width="220px"),
    )

    w_lbl_ps_status = W.HTML(value="PlateSolve: idle")
    w_img_platesolve = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    w_html_platesolve = W.HTML(value="")

    def _ps_send_params(_=None) -> None:
        # Conversión:
        # pixel_size_m debe incorporar binning si el solver trabaja en pixels "binned"
        pixel_size_m = float(w_tf_ps_pixel_um.value) * 1e-6  # binning fijo=1 en CameraStream
        focal_m = float(w_tf_ps_focal_mm.value) / 1000.0

        params = {
            "pixel_size_m": float(pixel_size_m),
            "focal_m": float(focal_m),
            "max_det": int(w_bi_ps_max_det.value),
            "det_thresh_sigma": float(w_tf_ps_det_sigma.value),
            "det_minarea": int(w_bi_ps_minarea.value),
            "point_sigma": float(w_tf_ps_point_sigma.value),
            "gmax": float(w_tf_ps_gmax.value),
            "search_radius_deg": float(w_tf_ps_search_radius_deg.value) if w_cb_ps_use_radius.value else None,
            "search_radius_factor": float(w_tf_ps_search_radius_factor.value),
            "theta_step_deg": float(w_tf_ps_theta_step.value),
            "theta_refine_span_deg": float(w_tf_ps_theta_refine_span.value),
            "theta_refine_step_deg": float(w_tf_ps_theta_refine_step.value),
            "match_max_px": float(w_tf_ps_match_max.value),
            "min_inliers": int(w_bi_ps_min_inliers.value),
            "guide_n": int(w_bi_ps_guide_n.value),
            "simbad_radius_arcsec": float(w_tf_ps_simbad_radius_arcsec.value),
            "auto_solve": bool(w_tb_ps_auto.value),
            "solve_every_s": float(w_tf_ps_every_s.value),
            "auto_target": str(w_txt_ps_target.value),
        }
        runner.enqueue(platesolve_set_params(**params))

    def _ps_request_once() -> None:
        target = str(w_txt_ps_target.value).strip()
        if not target:
            log_info(w_out_log, "PlateSolve: missing target")
            return
        runner.enqueue(platesolve_run(target=target))

    def _on_ps_solve(_btn):
        _ps_request_once()

    w_btn_ps_solve.on_click(_on_ps_solve)

    # Observers -> update runner cfg live
    # Use the debounce handler for all platesolve parameter changes.  This
    # prevents flooding the runner when multiple parameters are edited in
    # sequence.  When the user finishes editing, the parameters are applied
    # together after a brief delay.
    for _w in [
        w_tf_ps_focal_mm,
        w_tf_ps_pixel_um,
        w_bi_ps_binning,
        w_bi_ps_max_det,
        w_tf_ps_det_sigma,
        w_bi_ps_minarea,
        w_tf_ps_point_sigma,
        w_tf_ps_gmax,
        w_cb_ps_use_radius,
        w_tf_ps_search_radius_deg,
        w_tf_ps_search_radius_factor,
        w_tf_ps_theta_step,
        w_tf_ps_theta_refine_span,
        w_tf_ps_theta_refine_step,
        w_tf_ps_match_max,
        w_bi_ps_min_inliers,
        w_bi_ps_guide_n,
        w_tf_ps_simbad_radius_arcsec,
    ]:
        _w.observe(_debounce_ps_params, names="value")
    w_tb_ps_auto.observe(_debounce_ps_params, names="value")
    w_tf_ps_every_s.observe(_debounce_ps_params, names="value")
    w_txt_ps_target.observe(_debounce_ps_params, names="value")

    w_tab_platesolve = W.VBox(
        [
            W.HTML("<b>PlateSolve</b>"),
            W.HBox([w_btn_ps_solve, w_tb_ps_auto, w_tf_ps_every_s]),
            w_txt_ps_target,
            W.HTML("<b>Instrument</b>"),
            W.HBox([w_tf_ps_focal_mm, w_tf_ps_pixel_um, w_bi_ps_binning]),
            W.HTML("<b>Solver</b>"),
            W.HBox([w_bi_ps_max_det, w_tf_ps_det_sigma, w_bi_ps_minarea, w_tf_ps_point_sigma]),
            W.HBox([w_tf_ps_gmax, w_cb_ps_use_radius, w_tf_ps_search_radius_deg, w_tf_ps_search_radius_factor]),
            W.HBox([w_tf_ps_theta_step, w_tf_ps_theta_refine_span, w_tf_ps_theta_refine_step]),
            W.HBox([w_tf_ps_match_max, w_bi_ps_min_inliers, w_bi_ps_guide_n, w_tf_ps_simbad_radius_arcsec]),
            w_lbl_ps_status,
            w_html_platesolve,
            w_img_platesolve,
        ],
        layout=W.Layout(border="1px solid #eee", padding="8px", gap="6px"),
    )

    # -------------------------
    # Placeholder tabs
    # -------------------------
    # -------------------------
    # GoTo tab
    # -------------------------
    w_lbl_goto_status = W.HTML("<b>GoTo</b>: IDLE")

    w_dd_goto_mode = W.Dropdown(
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

    w_txt_goto_name = W.Text(value="", description="Obj:", layout=W.Layout(width="360px"))
    w_dd_goto_planet = W.Dropdown(
        options=["moon","mercury","venus","mars","jupiter","saturn","uranus","neptune"],
        value="mars",
        description="Planeta:",
        layout=W.Layout(width="260px"),
    )
    # Use bounded inputs for RA/Dec/Az/Alt to prevent out-of-range values.  RA and
    # Azimuth wrap around 0–360°, declination is limited to ±90°, and altitude
    # between 0° (horizon) and 90° (zenith).
    w_tf_goto_ra = W.BoundedFloatText(value=0.0, min=0.0, max=360.0, step=0.1, description="RA°:", layout=W.Layout(width="200px"))
    w_tf_goto_dec = W.BoundedFloatText(value=0.0, min=-90.0, max=90.0, step=0.1, description="Dec°:", layout=W.Layout(width="200px"))
    w_tf_goto_az = W.BoundedFloatText(value=0.0, min=0.0, max=360.0, step=0.1, description="Az°:", layout=W.Layout(width="200px"))
    w_tf_goto_alt = W.BoundedFloatText(value=45.0, min=0.0, max=90.0, step=0.1, description="Alt°:", layout=W.Layout(width="200px"))

    w_bt_goto_tol = W.BoundedFloatText(value=10.0, min=0.5, max=3600.0, step=0.5, description="Tol (arcsec):", layout=W.Layout(width="220px"))
    w_bi_goto_max_iters = W.BoundedIntText(value=6, min=1, max=50, step=1, description="Iters:", layout=W.Layout(width="160px"))
    w_bt_goto_gain = W.BoundedFloatText(value=0.85, min=0.1, max=2.0, step=0.05, description="Gain:", layout=W.Layout(width="160px"))
    w_bt_goto_settle_s = W.BoundedFloatText(value=0.25, min=0.0, max=10.0, step=0.05, description="Settle(s):", layout=W.Layout(width="170px"))

    # Calibración
    w_bi_calib_samples = W.BoundedIntText(value=8, min=2, max=80, step=1, description="Muestras:", layout=W.Layout(width="180px"))
    w_dd_calib_units = W.Dropdown(options=[("Grados", "deg"), ("Pasos", "steps")], value="deg", description="Unidad:", layout=W.Layout(width="190px"))
    w_bt_calib_small = W.BoundedFloatText(value=1.0, min=0.1, max=30.0, step=0.1, description="Paso 1:", layout=W.Layout(width="160px"))
    w_bt_calib_big = W.BoundedFloatText(value=5.0, min=0.1, max=60.0, step=0.1, description="Paso 2:", layout=W.Layout(width="160px"))

    # Reusa delays de la pestaña mount/manual
    w_bi_goto_delay_us = W.BoundedIntText(value=1800, min=50, max=50000, step=50, description="delay_us:", layout=W.Layout(width="200px"))

    w_btn_goto_sync = W.Button(description="Sync", button_style="info", layout=W.Layout(width="100px"))
    w_btn_goto_run = W.Button(description="GoTo", button_style="success", layout=W.Layout(width="100px"))
    w_btn_goto_calib = W.Button(description="Calibrate", button_style="warning", layout=W.Layout(width="120px"))
    w_btn_goto_cancel = W.Button(description="Cancel", button_style="danger", layout=W.Layout(width="110px"))

    w_box_goto_name = W.HBox([w_txt_goto_name])
    w_box_goto_planet = W.HBox([w_dd_goto_planet])
    w_box_goto_radec = W.HBox([w_tf_goto_ra, w_tf_goto_dec])
    w_box_goto_altaz = W.HBox([w_tf_goto_az, w_tf_goto_alt])

    w_box_goto_target = W.VBox([w_box_goto_name])

    def _goto_mode_changed(change):
        m = str(change["new"])
        if m == "name":
            w_box_goto_target.children = [w_box_goto_name]
        elif m == "planet":
            w_box_goto_target.children = [w_box_goto_planet]
        elif m == "radec":
            w_box_goto_target.children = [w_box_goto_radec]
        else:
            w_box_goto_target.children = [w_box_goto_altaz]

    w_dd_goto_mode.observe(_goto_mode_changed, names="value")

    def _build_goto_target() -> Any:
        m = str(w_dd_goto_mode.value)
        if m == "name":
            return str(w_txt_goto_name.value).strip()
        if m == "planet":
            return str(w_dd_goto_planet.value).strip()
        if m == "radec":
            return {"ra_deg": float(w_tf_goto_ra.value), "dec_deg": float(w_tf_goto_dec.value)}
        return {"az_deg": float(w_tf_goto_az.value), "alt_deg": float(w_tf_goto_alt.value)}

    def _enqueue_goto_sync():
        runner.enqueue(mount_sync())

    def _enqueue_goto_run():
        target = _build_goto_target()
        runner.enqueue(
            mount_goto(
                target,
                tol_arcsec=float(w_bt_goto_tol.value),
                max_iters=int(w_bi_goto_max_iters.value),
                gain=float(w_bt_goto_gain.value),
                settle_s=float(w_bt_goto_settle_s.value),
                delay_us=int(w_bi_goto_delay_us.value),
            )
        )

    def _enqueue_goto_calib():
        params = {
            "n_samples": int(w_bi_calib_samples.value),
            "step_unit": str(w_dd_calib_units.value),
            "step_magnitudes": [float(w_bt_calib_small.value), float(w_bt_calib_big.value)],
            "delay_us": int(w_bi_goto_delay_us.value),
        }
        runner.enqueue(goto_calibrate(params))

    def _enqueue_goto_cancel():
        runner.enqueue(goto_cancel())

    w_btn_goto_sync.on_click(lambda _: _enqueue_goto_sync())
    w_btn_goto_run.on_click(lambda _: _enqueue_goto_run())
    w_btn_goto_calib.on_click(lambda _: _enqueue_goto_calib())
    w_btn_goto_cancel.on_click(lambda _: _enqueue_goto_cancel())

    w_box_goto_buttons = W.HBox([w_btn_goto_sync, w_btn_goto_run, w_btn_goto_calib, w_btn_goto_cancel])
    w_box_goto_params = W.HBox([w_bt_goto_tol, w_bi_goto_max_iters, w_bt_goto_gain, w_bt_goto_settle_s])
    w_box_goto_calib = W.HBox([w_bi_calib_samples, w_dd_calib_units, w_bt_calib_small, w_bt_calib_big])
    w_box_goto_delay = W.HBox([w_bi_goto_delay_us])

    w_tab_goto = W.VBox([
        w_lbl_goto_status,
        W.HBox([w_dd_goto_mode]),
        w_box_goto_target,
        w_box_goto_params,
        w_box_goto_delay,
        w_box_goto_buttons,
        W.HTML("<hr/>"),
        W.HTML("<b>Calibración</b> (muestras aleatorias)") ,
        w_box_goto_calib,
    ])
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
    # -------------------------------------------------------------------------
    # Camera parameter debounce.  Changing camera exposure, gain or preview
    # settings forces the runner to reconnect the camera.  To avoid multiple
    # reconnections while editing values (e.g. when typing), each parameter
    # update is debounced: only the last change within 0.3s triggers a
    # camera_set_param action.
    _cam_param_timers: Dict[str, threading.Timer] = {}

    def _debounce_camera_param(name: str, value: Any) -> None:
        """Send camera_set_param(name, value) after 0.3s of inactivity."""
        # cancel existing timer for this parameter
        t = _cam_param_timers.get(name)
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

        def _send() -> None:
            try:
                runner.enqueue(camera_set_param(name, value))
            except Exception as exc:
                log_error(w_out_log, f"UI: failed to enqueue camera param {name}", exc, throttle_s=5.0, throttle_key=f"ui_cam_{name}")

        timer = threading.Timer(0.3, _send)
        _cam_param_timers[name] = timer
        timer.start()

    def _on_exp(change):
        _debounce_camera_param("exp_ms", int(change["new"]))

    def _on_gain(change):
        _debounce_camera_param("gain", int(change["new"]))

    def _on_auto_gain(change):
        _debounce_camera_param("auto_gain", bool(change["new"]))

    # No handler for image format: the format is fixed to RAW16.  The corresponding dropdown
    # has been removed.

    def _on_view_hz(change):
        _debounce_camera_param("preview_view_hz", float(change["new"]))

    def _on_jpeg_q(change):
        _debounce_camera_param("preview_jpeg_quality", int(change["new"]))

    def _on_plo(change):
        _debounce_camera_param("preview_stretch_plo", float(change["new"]))

    def _on_phi(change):
        _debounce_camera_param("preview_stretch_phi", float(change["new"]))

    w_bi_exp_ms.observe(_on_exp, names="value")
    w_bi_gain.observe(_on_gain, names="value")
    w_cb_auto_gain.observe(_on_auto_gain, names="value")
    # There is no observer for image format; see notes above.

    w_bt_view_hz.observe(_on_view_hz, names="value")
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
    # Bindings: Manual mount (MOVE steps/delay)
    # -------------------------
    # Note: microstep divisions (MS AZ / MS ALT) are applied automatically via the debounce
    # handler defined above. This section binds only movement and stop actions.
    def _enqueue_move(axis: Axis, direction: int, steps: int, delay_us: int):
        if steps <= 0 or delay_us <= 0:
            log_info(w_out_log, f"Manual MOVE: invalid params steps={steps} delay_us={delay_us}")
            return
        runner.enqueue(mount_move_steps(axis=axis, direction=direction, steps=steps, delay_us=delay_us))

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

    def _on_stop_top(_btn):
        runner.enqueue(mount_stop())

    w_btn_az_left.on_click(_on_az_left)
    w_btn_az_right.on_click(_on_az_right)
    w_btn_alt_up.on_click(_on_alt_up)
    w_btn_alt_down.on_click(_on_alt_down)
    w_btn_stop.on_click(_on_stop)
    w_btn_stop_mount.on_click(_on_stop_top)

    # -------------------------
    # Bindings: Tracking toggle (Top Bar)
    # -------------------------
    def _on_tracking_toggle(change):
        on = bool(change["new"])
        current = bool(getattr(runner.get_state(), "tracking_enabled", False))
        if on == current:
            return
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
        current = bool(getattr(runner.get_state(), "stacking_enabled", False))
        if on == current:
            return
        if on:
            runner.enqueue(stacking_start())
        else:
            runner.enqueue(stacking_stop())

    w_btn_stacking_toggle.observe(_on_stacking_toggle, names="value")

    def _on_stacking_reset(_btn):
        runner.enqueue(stacking_reset())

    w_btn_stack_reset.on_click(_on_stacking_reset)

    # -------------------------
    # Binding: Save Stack
    # -------------------------
    def _on_save_stack(_btn):
        """Save the current stacked mosaic to disk.

        The output directory is fixed to 'stack_output' relative to the current
        working directory.  A timestamp is used as the basename to avoid
        collisions.  The backend will produce both a raw .npy file and a
        stretched PNG.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = "stack_output"
        runner.enqueue(stacking_save(out_dir=out_dir, basename=ts, fmt="png"))

    w_btn_save_quick.on_click(_on_save_stack)

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
        "w_btn_stop_mount": w_btn_stop_mount,
        "w_btn_tracking_toggle": w_btn_tracking_toggle,
        "w_btn_stacking_toggle": w_btn_stacking_toggle,
        "w_btn_save_quick": w_btn_save_quick,
        # live
        "w_img_live": w_img_live,
        "w_tb_live_sep": w_tb_live_sep,
        "w_bi_live_sep_bw": w_bi_live_sep_bw,
        "w_bi_live_sep_bh": w_bi_live_sep_bh,
        "w_tf_live_sep_sigma": w_tf_live_sep_sigma,
        "w_bi_live_sep_minarea": w_bi_live_sep_minarea,
        "w_bi_live_sep_max_det": w_bi_live_sep_max_det,
        # stacking tab
        "w_img_stack": w_img_stack,
        "w_btn_stack_reset": w_btn_stack_reset,
        "w_btn_stack_start": w_btn_stack_start,
        "w_btn_stack_stop": w_btn_stack_stop,
        "w_btn_hotpix_calib": w_btn_hotpix_calib,
        # platesolve tab
        "w_txt_ps_target": w_txt_ps_target,
        "w_btn_ps_solve": w_btn_ps_solve,
        "w_tb_ps_auto": w_tb_ps_auto,
        "w_tf_ps_every_s": w_tf_ps_every_s,
        "w_lbl_ps_status": w_lbl_ps_status,
        "w_img_platesolve": w_img_platesolve,
        "w_html_platesolve": w_html_platesolve,
        "w_tf_ps_focal_mm": w_tf_ps_focal_mm,
        "w_tf_ps_pixel_um": w_tf_ps_pixel_um,
        "w_bi_ps_binning": w_bi_ps_binning,
        "w_bi_ps_max_det": w_bi_ps_max_det,
        "w_tf_ps_det_sigma": w_tf_ps_det_sigma,
        "w_bi_ps_minarea": w_bi_ps_minarea,
        "w_tf_ps_point_sigma": w_tf_ps_point_sigma,
        "w_tf_ps_gmax": w_tf_ps_gmax,
        "w_cb_ps_use_radius": w_cb_ps_use_radius,
        "w_tf_ps_search_radius_deg": w_tf_ps_search_radius_deg,
        "w_tf_ps_search_radius_factor": w_tf_ps_search_radius_factor,
        "w_tf_ps_theta_step": w_tf_ps_theta_step,
        "w_tf_ps_theta_refine_span": w_tf_ps_theta_refine_span,
        "w_tf_ps_theta_refine_step": w_tf_ps_theta_refine_step,
        "w_tf_ps_match_max": w_tf_ps_match_max,
        "w_bi_ps_min_inliers": w_bi_ps_min_inliers,
        "w_bi_ps_guide_n": w_bi_ps_guide_n,
        "w_tf_ps_simbad_radius_arcsec": w_tf_ps_simbad_radius_arcsec,
        # goto tab
        "w_lbl_goto_status": w_lbl_goto_status,
        # manual mount control
        "w_dd_ms_az": w_dd_ms_az,
        "w_dd_ms_alt": w_dd_ms_alt,
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
        # image format is fixed to RAW16; provide label instead of dropdown
        "w_lbl_img_format": w_lbl_img_format,
        "w_bt_view_hz": w_bt_view_hz,
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

        # Determine connectivity and running states for gating controls
        cam_connected = bool(getattr(st, "camera_connected", False))
        mount_connected = bool(getattr(st, "mount_connected", False))
        stacking_running = bool(getattr(st, "stacking_enabled", False))
        tracking_running = bool(getattr(st, "tracking_enabled", False))

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
        # GoTo status (si existe)
        if "w_lbl_goto_status" in self.widgets:
            busy = bool(getattr(st, "goto_busy", False))
            status = str(getattr(st, "goto_status", "IDLE"))
            synced = bool(getattr(st, "goto_synced", False))
            err_as = float(getattr(st, "goto_last_error_arcsec", 0.0))
            J00 = float(getattr(st, "goto_J00", 0.0))
            J01 = float(getattr(st, "goto_J01", 0.0))
            J10 = float(getattr(st, "goto_J10", 0.0))
            J11 = float(getattr(st, "goto_J11", 0.0))
            self.widgets["w_lbl_goto_status"].value = (
                f"<b>GoTo</b>: {status} | busy={busy} | synced={synced} | "
                f"err={err_as:.1f}\" | J=[[{J00:.6g},{J01:.6g}],[{J10:.6g},{J11:.6g}]]"
            )


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

        # Platesolve status (si existe)
        if "w_lbl_ps_status" in self.widgets:
            ps_status = str(getattr(st, "platesolve_status", "IDLE"))
            ps_busy = bool(getattr(st, "platesolve_busy", False))
            ps_ok = bool(getattr(st, "platesolve_last_ok", False))
            ra = float(getattr(st, "platesolve_center_ra_deg", 0.0))
            dec = float(getattr(st, "platesolve_center_dec_deg", 0.0))
            th = float(getattr(st, "platesolve_theta_deg", 0.0))
            dx = float(getattr(st, "platesolve_dx_px", 0.0))
            dy = float(getattr(st, "platesolve_dy_px", 0.0))
            resp = float(getattr(st, "platesolve_resp", 0.0))
            nin = int(getattr(st, "platesolve_n_inliers", 0))
            rms = float(getattr(st, "platesolve_rms_px", 0.0))

            self.widgets["w_lbl_ps_status"].value = (
                f"<b>PlateSolve</b>: {ps_status} | busy={ps_busy} | ok={ps_ok} | "
                f"RA={ra:.6f} Dec={dec:.6f} | "
                f"theta={th:+.2f}° dx={dx:+.2f} dy={dy:+.2f} | "
                f"resp={resp:.3f} inliers={nin} rms={rms:.2f}px"
            )
        if "w_html_platesolve" in self.widgets:
            debug_info = dict(getattr(st, "platesolve_debug_info", {}) or {})
            if debug_info:
                ordered = [
                    "status",
                    "response",
                    "n_det",
                    "gaia_rows",
                    "n_inliers",
                    "rms_px",
                    "theta_deg",
                    "dx_px",
                    "dy_px",
                    "radius_deg",
                    "scale_arcsec_per_px",
                ]

                def _fmt_value(val: Any) -> str:
                    if isinstance(val, float):
                        return f"{val:.4g}"
                    return str(val)

                lines = []
                for key in ordered:
                    if key not in debug_info:
                        continue
                    val = debug_info.get(key)
                    if val is None:
                        continue
                    lines.append(f"<li><b>{key}</b>: {_fmt_value(val)}</li>")
                if lines:
                    self.widgets["w_html_platesolve"].value = "<ul>" + "".join(lines) + "</ul>"
                else:
                    self.widgets["w_html_platesolve"].value = ""
            else:
                self.widgets["w_html_platesolve"].value = ""

        # opcional: mantener toggle en sync si cambia por fuera
        if "w_btn_tracking_toggle" in self.widgets:
            try:
                btn = self.widgets["w_btn_tracking_toggle"]
                if bool(btn.value) != tracking_enabled:
                    btn.value = tracking_enabled
            except Exception as exc:
                log_error(w_out_log, "UI: failed to sync tracking toggle", exc, throttle_s=5.0, throttle_key="ui_sync_tracking_toggle")
        if "w_btn_stacking_toggle" in self.widgets:
            try:
                btn = self.widgets["w_btn_stacking_toggle"]
                stacking_enabled = bool(getattr(st, "stacking_enabled", False))
                if bool(btn.value) != stacking_enabled:
                    btn.value = stacking_enabled
            except Exception as exc:
                log_error(w_out_log, "UI: failed to sync stacking toggle", exc, throttle_s=5.0, throttle_key="ui_sync_stacking_toggle")

        jpg = self.runner.get_latest_preview_jpeg()
        if jpg:
            self.widgets["w_img_live"].value = jpg

        stack_jpg = getattr(st, "stacking_preview_jpeg", None)
        if stack_jpg and "w_img_stack" in self.widgets:
            self.widgets["w_img_stack"].value = stack_jpg

        ps_jpg = getattr(st, "platesolve_debug_jpeg", None)
        if ps_jpg and "w_img_platesolve" in self.widgets:
            self.widgets["w_img_platesolve"].value = ps_jpg

        # -------------------------------------------------------------------
        # Enable/disable UI controls based on current state
        # Camera connect/disconnect
        if "w_btn_connect_camera" in self.widgets:
            self.widgets["w_btn_connect_camera"].disabled = cam_connected
        if "w_btn_disconnect_camera" in self.widgets:
            self.widgets["w_btn_disconnect_camera"].disabled = not cam_connected
        # Mount connect/disconnect (both top bar and tab)
        if "w_btn_connect_mount" in self.widgets:
            self.widgets["w_btn_connect_mount"].disabled = mount_connected
        if "w_btn_disconnect_mount" in self.widgets:
            self.widgets["w_btn_disconnect_mount"].disabled = not mount_connected
        if "w_btn_mount_connect_tab" in self.widgets:
            self.widgets["w_btn_mount_connect_tab"].disabled = mount_connected
        if "w_btn_mount_disconnect_tab" in self.widgets:
            self.widgets["w_btn_mount_disconnect_tab"].disabled = not mount_connected

        # Manual mount controls: disable if mount is not connected
        manual_keys = [
            "w_dd_ms_az",
            "w_dd_ms_alt",
            "w_steps_az",
            "w_delay_az",
            "w_steps_alt",
            "w_delay_alt",
            "w_btn_az_left",
            "w_btn_az_right",
            "w_btn_alt_up",
            "w_btn_alt_down",
            "w_btn_stop",
        ]
        for k in manual_keys:
            if k in self.widgets:
                # Always allow STOP (for safety) regardless of mount connection
                if k in ("w_btn_stop",):
                    self.widgets[k].disabled = False
                else:
                    self.widgets[k].disabled = not mount_connected

        # Tracking controls: disable toggle if camera or mount not connected
        if "w_btn_tracking_toggle" in self.widgets:
            self.widgets["w_btn_tracking_toggle"].disabled = not (cam_connected and mount_connected)
        # Tracking tab buttons
        if "w_btn_track_start" in self.widgets:
            self.widgets["w_btn_track_start"].disabled = not (cam_connected and mount_connected)
        if "w_btn_track_stop" in self.widgets:
            self.widgets["w_btn_track_stop"].disabled = not tracking_running

        # Stacking controls: disable toggle and tab buttons when camera not connected
        if "w_btn_stacking_toggle" in self.widgets:
            self.widgets["w_btn_stacking_toggle"].disabled = not cam_connected
        if "w_btn_stack_start" in self.widgets:
            self.widgets["w_btn_stack_start"].disabled = not cam_connected
        if "w_btn_stack_stop" in self.widgets:
            # Enable stop only if stacking is currently running
            self.widgets["w_btn_stack_stop"].disabled = not stacking_running
        if "w_btn_stack_reset" in self.widgets:
            # Allow resetting only if stacking is running (has state)
            self.widgets["w_btn_stack_reset"].disabled = not stacking_running
        if "w_btn_hotpix_calib" in self.widgets:
            self.widgets["w_btn_hotpix_calib"].disabled = not cam_connected

        # Save Stack button: enable only when stacking is running.  When disabled,
        # clicking has no effect.  The button is visible in the top bar.
        if "w_btn_save_quick" in self.widgets:
            self.widgets["w_btn_save_quick"].disabled = not stacking_running


def show_ui(cfg: AppConfig, runner: AppRunner, *, start_loops: bool = True, ui_hz: float = 10.0):
    built = build_ui(cfg, runner)
    display(built["root"])

    ui_loop = UILoop(runner, built["widgets"], max_hz=float(ui_hz))
    if start_loops:
        ui_loop.start()

    return built, ui_loop
