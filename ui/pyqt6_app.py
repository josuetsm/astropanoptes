from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from actions import tracking_keyframe_reset
from ap_types import Axis
from app_runner import AppRunner
from config import AppConfig
from logging_utils import set_global_log_sink
from ui.tabs_mixin import ModulesTabsMixin, STACKING_DRIZZLE_SCALES


class QtLogSink(QObject):
    message = pyqtSignal(str)

    def __init__(self, append_cb) -> None:
        super().__init__()
        self.message.connect(append_cb)

    def __call__(self, msg: str) -> None:
        self.message.emit(msg)


@dataclass
class Detection:
    x: float
    y: float
    a: float = 3.0
    b: float = 3.0
    theta: float = 0.0
    flux: float = 0.0


@dataclass
class OverlayToggles:
    show_detections: bool = True
    show_seeds: bool = True
    show_seed_edges: bool = True
    show_drift: bool = True
    show_crosshair: bool = True
    show_text: bool = False


@dataclass
class DriftInfo:
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0


@dataclass
class PlatesolveSummary:
    running: bool = False
    status: str = "idle"
    inliers: int = 0
    rms_px: float = 0.0
    theta_deg: float = 0.0
    dx_px: float = 0.0
    dy_px: float = 0.0
    center_ra: str = "--"
    center_dec: str = "--"


class OverlayRenderer:
    def __init__(self) -> None:
        self.pen_det = QPen(QColor(80, 200, 255), 1)
        self.pen_seed = QPen(QColor(255, 210, 80), 2)
        self.pen_seed_edge = QPen(QColor(255, 210, 80, 160), 1)
        self.pen_drift = QPen(QColor(120, 255, 120), 2)
        self.pen_cross = QPen(QColor(200, 200, 200, 140), 1)

        self.font = QFont()
        self.font.setPointSize(10)

    def render(
        self,
        base: QImage,
        *,
        toggles: OverlayToggles,
        detections: Optional[Sequence[Detection]] = None,
        n_seeds: int = 3,
        drift: Optional[DriftInfo] = None,
    ) -> QImage:
        if base.isNull():
            return base

        out = base.copy()
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(self.font)

        width = out.width()
        height = out.height()
        cx = width * 0.5
        cy = height * 0.5

        det_list = list(detections) if detections is not None else []

        if toggles.show_crosshair:
            painter.setPen(self.pen_cross)
            painter.drawLine(int(cx) - 18, int(cy), int(cx) + 18, int(cy))
            painter.drawLine(int(cx), int(cy) - 18, int(cx), int(cy) + 18)
            painter.drawEllipse(QPointF(cx, cy), 12.0, 12.0)

        if toggles.show_detections and det_list:
            painter.setPen(self.pen_det)
            for det in det_list:
                self._draw_ellipse(painter, det)
                painter.drawEllipse(QPointF(det.x, det.y), 1.5, 1.5)

        seeds: list[Detection] = []
        if det_list and n_seeds > 0:
            seeds = sorted(det_list, key=lambda det: det.flux, reverse=True)[: int(n_seeds)]

        if toggles.show_seeds and seeds:
            painter.setPen(self.pen_seed)
            for i, det in enumerate(seeds):
                painter.drawEllipse(QPointF(det.x, det.y), 10.0, 10.0)
                painter.drawText(QPointF(det.x + 12.0, det.y - 10.0), f"S{i + 1}")

        if toggles.show_seed_edges and len(seeds) >= 2:
            painter.setPen(self.pen_seed_edge)
            for i in range(len(seeds) - 1):
                painter.drawLine(QPointF(seeds[i].x, seeds[i].y), QPointF(seeds[i + 1].x, seeds[i + 1].y))
            if len(seeds) >= 3:
                painter.drawLine(QPointF(seeds[-1].x, seeds[-1].y), QPointF(seeds[0].x, seeds[0].y))

        if toggles.show_drift and drift is not None:
            vx = float(drift.vx_px_s)
            vy = float(drift.vy_px_s)

            scale = 1.0
            maxlen = 120.0
            length = math.hypot(vx, vy) * scale
            if length > maxlen and length > 1e-6:
                scale *= maxlen / length

            origin = QPointF(40.0, height - 40.0)
            tip = QPointF(origin.x() + vx * scale, origin.y() + vy * scale)

            painter.setPen(self.pen_drift)
            painter.drawLine(origin, tip)
            self._draw_arrow_head(painter, origin, tip, size=10.0)

        painter.end()
        return out

    def _draw_ellipse(self, painter: QPainter, det: Detection) -> None:
        painter.save()
        painter.translate(det.x, det.y)
        painter.rotate(math.degrees(det.theta))
        rect = QRectF(-det.a, -det.b, 2.0 * det.a, 2.0 * det.b)
        painter.drawEllipse(rect)
        painter.restore()

    def _draw_arrow_head(self, painter: QPainter, start: QPointF, end: QPointF, *, size: float = 10.0) -> None:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        tip = end
        left = QPointF(tip.x() - ux * size + px * (size * 0.45), tip.y() - uy * size + py * (size * 0.45))
        right = QPointF(tip.x() - ux * size - px * (size * 0.45), tip.y() - uy * size - py * (size * 0.45))
        painter.drawLine(tip, left)
        painter.drawLine(tip, right)


class Chip(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.set_mode("neutral")

    def set_mode(self, mode: str) -> None:
        if mode == "green":
            self.setStyleSheet(
                "QLabel { padding:4px 10px; border-radius:10px;"
                "background:#16321a; border:1px solid #2f6b38; color:#e8ffe8; }"
            )
        elif mode == "red":
            self.setStyleSheet(
                "QLabel { padding:4px 10px; border-radius:10px;"
                "background:#3a1515; border:1px solid #7a2a2a; color:#ffecec; }"
            )
        elif mode == "active":
            self.setStyleSheet(
                "QLabel { padding:4px 10px; border-radius:10px;"
                "background:#2b2b2b; border:1px solid #5a5a5a; color:#f0f0f0; }"
            )
        else:
            self.setStyleSheet(
                "QLabel { padding:4px 10px; border-radius:10px;"
                "background:#1b1b1b; border:1px solid #444; color:#ddd; }"
            )


class AstroPanoptesWindow(ModulesTabsMixin, QMainWindow):
    def __init__(self, runner: AppRunner, cfg: AppConfig) -> None:
        super().__init__()
        self.runner = runner
        self.cfg = cfg
        self._ps_outputs_enabled = False

        self.setWindowTitle("AstroPanoptes")
        self.resize(1280, 760)

        self.od_enabled = False

        self.overlay_toggles = OverlayToggles(
            show_detections=True,
            show_seeds=True,
            show_seed_edges=True,
            show_drift=True,
            show_crosshair=True,
            show_text=False,
        )
        self.renderer = OverlayRenderer()

        self.base_w = 960
        self.base_h = 600
        self._rng = np.random.default_rng(7)
        self._stars = self._init_star_catalog(n=120)

        self._build_central()
        self._log_sink = QtLogSink(self._log)
        set_global_log_sink(self._log_sink)
        self._build_top_toolbar()
        self._build_docks()
        self._build_menu()

        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(100)
        self._frame_timer.timeout.connect(self._render_frame)
        self._frame_timer.start()

        self._t_ms = 0.0
        self._log("PyQt6 UI ready.")

    def closeEvent(self, event) -> None:  # noqa: N802
        set_global_log_sink(None)
        self.runner.stop()
        super().closeEvent(event)

    def _build_central(self) -> None:
        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)

        self.live_view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.live_view.setMinimumSize(QSize(640, 420))
        self.live_view.setStyleSheet(
            "QLabel { background:#0f0f0f; border:1px solid #2a2a2a; border-radius:10px; }"
        )

        self.stacked_view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.stacked_view.setMinimumSize(QSize(640, 420))
        self.stacked_view.setStyleSheet(
            "QLabel { background:#0f0f0f; border:1px solid #2a2a2a; border-radius:10px; }"
        )

        wrap_live = QWidget()
        live_layout = QVBoxLayout(wrap_live)
        live_layout.setContentsMargins(12, 12, 12, 12)
        live_layout.addWidget(self.live_view)

        wrap_stack = QWidget()
        stack_layout = QVBoxLayout(wrap_stack)
        stack_layout.setContentsMargins(12, 12, 12, 12)
        stack_layout.addWidget(self.stacked_view)

        self.view_tabs.addTab(wrap_live, "Live")
        self.view_tabs.addTab(wrap_stack, "Stacked")

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.logs_frame = QGroupBox("Logs")
        logs_layout = QVBoxLayout(self.logs_frame)
        logs_layout.setContentsMargins(10, 10, 10, 10)
        logs_layout.addWidget(self.log)

        self.central_split = QSplitter(Qt.Orientation.Vertical)
        self.central_split.addWidget(self.view_tabs)
        self.central_split.addWidget(self.logs_frame)
        self.central_split.setCollapsible(0, False)
        self.central_split.setCollapsible(1, True)
        self.central_split.setSizes([800, 240])

        self.setCentralWidget(self.central_split)

    def _build_top_toolbar(self) -> None:
        self.tb_top = QToolBar("Top Bar", self)
        self.tb_top.setMovable(True)
        self.tb_top.setFloatable(True)
        self.tb_top.setAllowedAreas(
            Qt.ToolBarArea.TopToolBarArea
            | Qt.ToolBarArea.BottomToolBarArea
            | Qt.ToolBarArea.LeftToolBarArea
            | Qt.ToolBarArea.RightToolBarArea
        )

        top = QFrame()
        top.setFrameShape(QFrame.Shape.StyledPanel)
        top.setStyleSheet("QFrame { border:1px solid #2a2a2a; border-radius:10px; background:#161616; }")

        self.btn_connect_camera = QPushButton("Connect camera")
        self.btn_disconnect_camera = QPushButton("Disconnect camera")
        self.btn_connect_mount = QPushButton("Connect mount")
        self.btn_disconnect_mount = QPushButton("Disconnect mount")
        self.btn_connect_camera.clicked.connect(self._connect_camera)
        self.btn_disconnect_camera.clicked.connect(self._disconnect_camera)
        self.btn_connect_mount.clicked.connect(self._connect_mount)
        self.btn_disconnect_mount.clicked.connect(self._disconnect_mount)

        self.ch_cam = Chip("Camera")
        self.ch_mount = Chip("Mount")
        self.ch_sync = Chip("Sync")
        self.ch_tracking = Chip("Tracking")
        self.ch_stacking = Chip("Stacking")
        self.ch_od = Chip("Object Detection")
        self.ch_ps = Chip("Plate Solving")
        self.ch_goto = Chip("GoTo")

        self.lbl_fps = QLabel("FPS cap/max: --/--/--")
        self.lbl_drift = QLabel("drift vx/vy: --/-- px/s")
        self.lbl_coords = QLabel("RA/Dec: -- -- | Az/Alt: -- --")
        self.lbl_errors = QLabel("Errors: none")
        self.lbl_errors.setStyleSheet(
            "QLabel { padding:4px 10px; border:1px solid #5a2a2a; border-radius:10px; "
            "background:#1b0f0f; color:#ffdada; }"
        )
        for label in (self.lbl_fps, self.lbl_drift, self.lbl_coords):
            label.setStyleSheet(
                "QLabel { padding:4px 10px; border:1px solid #3a3a3a; border-radius:10px; "
                "background:#121212; color:#ddd; }"
            )

        row = QHBoxLayout()
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        row.addWidget(self.btn_connect_camera)
        row.addWidget(self.btn_disconnect_camera)
        row.addWidget(self.btn_connect_mount)
        row.addWidget(self.btn_disconnect_mount)
        row.addSpacing(8)
        for widget in [
            self.ch_cam,
            self.ch_mount,
            self.ch_sync,
            self.ch_tracking,
            self.ch_stacking,
            self.ch_od,
            self.ch_ps,
            self.ch_goto,
        ]:
            row.addWidget(widget)
        row.addStretch(1)

        metrics = QHBoxLayout()
        metrics.setContentsMargins(10, 0, 10, 8)
        metrics.setSpacing(8)
        metrics.addWidget(self.lbl_fps)
        metrics.addWidget(self.lbl_drift)
        metrics.addWidget(self.lbl_coords, stretch=1)
        metrics.addWidget(self.lbl_errors)

        layout = QVBoxLayout(top)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(row)
        layout.addLayout(metrics)

        self.tb_top.addWidget(top)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tb_top)

        self._update_chips_from_state(self.runner.get_state())

    def _build_docks(self) -> None:
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )

        modules_panel = self._build_modules_tabs()
        self.dock_modules = QDockWidget("Modules", self)
        self.dock_modules.setWidget(modules_panel)
        self.dock_modules.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_modules)

        manual_wrap = QWidget()
        mv = QVBoxLayout(manual_wrap)
        mv.setContentsMargins(10, 10, 10, 10)
        mv.addWidget(self._build_manual_mount_panel())

        self.dock_manual = QDockWidget("Manual Controls", self)
        self.dock_manual.setWidget(manual_wrap)
        self.dock_manual.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_manual)
        self.splitDockWidget(self.dock_modules, self.dock_manual, Qt.Orientation.Vertical)

        self.resizeDocks([self.dock_modules, self.dock_manual], [700, 300], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock_modules, self.dock_manual], [420, 420], Qt.Orientation.Horizontal)

    def _build_menu(self) -> None:
        menu_view = self.menuBar().addMenu("View")

        menu_view.addAction(self.dock_modules.toggleViewAction())
        menu_view.addAction(self.dock_manual.toggleViewAction())

        act_top = self.tb_top.toggleViewAction()
        act_top.setText("Top Bar")
        menu_view.addAction(act_top)

        self.act_logs = QAction("Logs", self, checkable=True)
        self.act_logs.setChecked(True)
        self.act_logs.toggled.connect(self.logs_frame.setVisible)
        menu_view.addAction(self.act_logs)

    def _build_manual_mount_panel(self) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { border:1px solid #2a2a2a; border-radius:10px; background:#141414; }"
            "QLabel { color:#ddd; }"
        )

        self.sb_steps = QSpinBox()
        self.sb_steps.setRange(1, 2_000_000)
        self.sb_steps.setValue(2000)

        self.sb_delay = QSpinBox()
        self.sb_delay.setRange(50, 200_000)
        self.sb_delay.setValue(1000)
        self.sb_delay.setSuffix(" µs")

        toprow = QHBoxLayout()
        toprow.addWidget(QLabel("steps:"))
        toprow.addWidget(self.sb_steps)
        toprow.addSpacing(12)
        toprow.addWidget(QLabel("delay_us:"))
        toprow.addWidget(self.sb_delay)
        toprow.addStretch(1)

        def mk_btn(text: str, danger: bool = False) -> QToolButton:
            button = QToolButton()
            button.setText(text)
            button.setFixedSize(QSize(36, 36))
            if danger:
                button.setStyleSheet(
                    "QToolButton { border-radius:10px; background:#7a2a2a; border:1px solid #7a2a2a; "
                    "color:#fff; font-weight:800; }"
                    "QToolButton:hover { background:#6a2424; border:1px solid #6a2424; }"
                )
            else:
                button.setStyleSheet(
                    "QToolButton { border-radius:10px; background:#1f1f1f; border:1px solid #3a3a3a; "
                    "color:#e8e8e8; }"
                    "QToolButton:hover { background:#262626; border:1px solid #5a5a5a; }"
                )
            return button

        self.b_up = mk_btn("▲")
        self.b_down = mk_btn("▼")
        self.b_left = mk_btn("◀")
        self.b_right = mk_btn("▶")
        self.b_stop = mk_btn("■", danger=True)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        grid.addWidget(self.b_up, 0, 1)
        grid.addWidget(self.b_left, 1, 0)
        grid.addWidget(self.b_stop, 1, 1)
        grid.addWidget(self.b_right, 1, 2)
        grid.addWidget(self.b_down, 2, 1)

        self.b_up.clicked.connect(lambda: self._manual_move(Axis.ALT, 1))
        self.b_down.clicked.connect(lambda: self._manual_move(Axis.ALT, -1))
        self.b_left.clicked.connect(lambda: self._manual_move(Axis.AZ, -1))
        self.b_right.clicked.connect(lambda: self._manual_move(Axis.AZ, 1))
        self.b_stop.clicked.connect(self._manual_stop)

        dwrap = QHBoxLayout()
        dwrap.addStretch(1)
        dwrap.addLayout(grid)
        dwrap.addStretch(1)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addLayout(toprow)
        layout.addLayout(dwrap)

        return card

    # Module tabs and per-tab helpers are provided by `ui.tabs_mixin.ModulesTabsMixin`.

    def _camera_apply(self) -> None:
        exp_ms = float(self.ds_exp_ms.value())
        gain = int(self.sb_gain.value())
        self.runner.request_camera_param("exp_ms", exp_ms)
        self.runner.request_camera_param("gain", gain)
        self._log(f"[camera] apply exposure={exp_ms:.1f} ms gain={gain}")

    def _camera_record_raw(self) -> None:
        if not bool(self.runner.get_state().camera.connected):
            self._log("[camera] Record RAW skipped (camera not connected)")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        basename = f"raw_{ts}"
        self.runner.request_camera_record_raw(duration_s=20.0, out_dir="raw_output", basename=basename)
        self._log(f"[camera] Record 20s RAW -> raw_output/{basename}.npy")

    def _connect_camera(self) -> None:
        self.runner.request_camera_connect(self.cfg.camera.camera_index)
        self._log("[top] Connect camera")

    def _disconnect_camera(self) -> None:
        self.runner.request_camera_disconnect()
        self._log("[top] Disconnect camera")

    def _connect_mount(self) -> None:
        self.runner.request_mount_connect(self.cfg.mount.port, self.cfg.mount.baudrate)
        self._log("[top] Connect mount")

    def _disconnect_mount(self) -> None:
        self.runner.request_mount_disconnect()
        self._log("[top] Disconnect mount")

    def _od_start(self) -> None:
        self.od_enabled = True
        self.runner.request_live_sep_params(
            enabled=True,
            sep_minarea=int(self.sb_od_minarea.value()),
            sep_thresh_sigma=float(self.ds_od_sigma.value()),
            max_det=int(self.sb_od_maxdet.value()),
            sep_bw=int(self.sb_od_bw.value()),
            sep_bh=int(self.sb_od_bh.value()),
        )
        self._log("[od] Start")

    def _od_stop(self) -> None:
        self.od_enabled = False
        self.runner.request_live_sep_params(enabled=False)
        self._log("[od] Stop")

    def _tracking_start(self) -> None:
        self.runner.request_tracking_start()
        self._log("[tracking] Start")

    def _tracking_stop(self) -> None:
        self.runner.request_tracking_stop()
        self._log("[tracking] Stop")

    def _tracking_reset(self) -> None:
        self.runner.enqueue(tracking_keyframe_reset())
        self._log("[tracking] Reset")

    def _stacking_start(self) -> None:
        self.runner.request_stacking_start()
        self._log("[stacking] Start")

    def _stacking_stop(self) -> None:
        self.runner.request_stacking_stop()
        self._log("[stacking] Stop")

    def _stacking_reset(self) -> None:
        self.runner.request_stacking_reset()
        self._log("[stacking] Reset")

    def _stacking_save(self) -> None:
        self.runner.request_stacking_save(out_dir="stack_output", basename="stack", fmt="png")
        self._log("[stacking] Save Stack -> stack_output/stack_YYYYMMDD_HHMMSS_az123p45_altp67p89.{raw.npy,png}")

    def _stacking_color_toggled(self, checked: bool) -> None:
        color_mode = "rgb" if bool(checked) else "mono"
        self.runner.request_stacking_params(color_mode=color_mode, bayer_pattern="RGGB")
        self._log(f"[stacking] color_mode={color_mode} (stack RGB RGGB, alignment mono)")

    def _stacking_drizzle_changed(self, _: int) -> None:
        data = self.dd_st_drizzle.currentData()
        try:
            drizzle_scale = float(data)
        except Exception:
            drizzle_scale = 2.0
        if drizzle_scale not in STACKING_DRIZZLE_SCALES:
            drizzle_scale = 2.0
        self.runner.request_stacking_params(drizzle_scale=drizzle_scale)
        self._log(f"[stacking] drizzle=x{int(drizzle_scale)}")

    def _platesolve_start(self) -> None:
        if not hasattr(self, "ed_ps_target"):
            self._log("[plate solving] tab removed; use GoTo panel controls")
            return
        target = self.ed_ps_target.text().strip()
        if not target:
            self._log("[plate solving] target empty; ignored")
            return
        self.runner.request_platesolving_params(
            search_radius_deg=float(self.ds_ps_radius.value()),
            search_radius_factor=float(self.ds_ps_radius_factor.value()),
            max_det=int(self.sb_ps_maxdet.value()),
            N_det=int(self.sb_ps_ndet.value()),
            N_seed=int(self.sb_ps_nseeds.value()),
            min_inliers=int(self.sb_ps_mininl.value()),
            det_thresh_sigma=float(self.ds_ps_det_sigma.value()),
            det_minarea=int(self.sb_ps_minarea.value()),
            point_sigma=float(self.ds_ps_point_sigma.value()),
            gmax=float(self.ds_ps_gmax.value()),
            match_max_px=float(self.ds_ps_match_max.value()),
            match_tol_arcsec=float(self.ds_ps_match_tol.value()),
            pred_margin_arcsec=float(self.ds_ps_pred_margin.value()),
            theta_step_deg=float(self.ds_ps_theta_step.value()),
            theta_refine_span_deg=float(self.ds_ps_theta_refine_span.value()),
            theta_refine_step_deg=float(self.ds_ps_theta_refine_step.value()),
            triplet_tol_arcsec=float(self.ds_ps_triplet_tol.value()),
            triplet_sigma_arcsec=float(self.ds_ps_triplet_sigma.value()),
            triplet_max_trials=int(self.sb_ps_triplet_trials.value()),
            max_i_scan=int(self.sb_ps_max_i_scan.value()),
            guide_n=int(self.sb_ps_guide_n.value()),
            simbad_radius_arcsec=float(self.ds_ps_simbad.value()),
            rotation_prior_roll_offset_deg=float(self.runner.get_state().camera.roll_deg),
        )
        self.runner.request_platesolving_run(target=target)
        self._log(f"[plate solving] Solve target={target}")

    def _goto_start(self) -> None:
        target = self._build_goto_target()
        if target is None:
            self._log("[goto] missing target; ignored")
            return
        params = {
            "platesolving_feedback": bool(self.cb_fb.isChecked()),
            "stages": int(self.sb_stages.value()),
            "N_seed": int(self.sb_goto_ps_nseeds.value()),
            "min_inliers": int(self.sb_goto_ps_mininl.value()),
        }
        self.runner.request_mount_goto(target, **params)
        self._log(
            f"[goto] GoTo target={target} "
            f"(N_seed={params['N_seed']} min_inliers={params['min_inliers']})"
        )

    def _goto_cancel(self) -> None:
        self.runner.request_goto_cancel()
        self._log("[goto] Cancel")

    def _autocalibrate(self) -> None:
        ps_mode = self._autocal_ps_mode_value()
        params = {
            "autocal_solve_radius_deg": float(self.ds_autocal_ps_radius.value()),
            "autocal_solve_gmax": float(self.ds_autocal_ps_gmax.value()),
            "N_seed": int(self.sb_goto_ps_nseeds.value()),
            "min_inliers": int(self.sb_goto_ps_mininl.value()),
            "autocal_ps_mode": ps_mode,
        }
        if ps_mode == "manual_altaz":
            params["autocal_ps_target"] = {
                "az_deg": float(self.ds_autocal_ps_manual_az.value()),
                "alt_deg": float(self.ds_autocal_ps_manual_alt.value()),
            }
        self.runner.request_goto_autocalibrate(params)
        target_txt = ""
        if "autocal_ps_target" in params:
            target_txt = f", target={params['autocal_ps_target']}"
        self._log(
            "[goto] AutoCalibrate "
            f"(radius={params['autocal_solve_radius_deg']:.2f}deg, gmax={params['autocal_solve_gmax']:.2f}, "
            f"N_seed={params['N_seed']}, min_inliers={params['min_inliers']}, "
            f"ps_mode={params['autocal_ps_mode']}{target_txt})"
        )

    def _goto_estimate_roll(self) -> None:
        self.runner.request_goto_estimate_roll()
        self._log("[goto] Estimar Roll")

    def _goto_fit_model(self) -> None:
        self.runner.request_goto_fit_model()
        self._log("[goto] Fit GoTo Model")

    def _goto_list_samples(self) -> None:
        self.runner.request_goto_list_samples()
        self._log("[goto] Listar muestras manuales")

    def _goto_prune_outliers(self) -> None:
        self.runner.request_goto_prune_outliers()
        self._log("[goto] Eliminar muestras outliers")

    def _goto_restore_last_log(self) -> None:
        self.runner.request_goto_restore_last_log()
        self._log("[goto] Cargar ultimo registro desde CSV (backup)")

    def _goto_reset(self) -> None:
        self.runner.request_goto_reset()
        self._log("[goto] Reset manual samples + sync")

    def _home(self) -> None:
        target = {
            "az_deg": 0.0,
            "alt_deg": float(self.cfg.goto.alt_min_deg),
        }
        self.runner.request_mount_goto(target)
        self._log("[goto] Home -> alt/az safe default")

    def _manual_move(self, axis: Axis, direction: int) -> None:
        steps = int(self.sb_steps.value())
        delay_us = int(self.sb_delay.value())
        if axis == Axis.AZ and self.cfg.mount.invert_az:
            direction *= -1
        if axis == Axis.ALT and self.cfg.mount.invert_alt:
            direction *= -1
        self.runner.request_mount_move_steps(axis, direction, steps, delay_us)
        self._log(f"[manual] move axis={axis.value} direction={direction} steps={steps} delay_us={delay_us}")

    def _manual_stop(self) -> None:
        self.runner.request_mount_stop()
        self._log("[manual] STOP")

    def _on_tick(self) -> None:
        self._t_ms += 100.0
        state = self.runner.get_state()

        fps_max = max(0.1, 1000.0 / max(0.1, float(self.ds_exp_ms.value())))
        self.lbl_fps.setText(
            f"FPS cap/max: {state.camera.fps_capture:.2f}/"
            f"{fps_max:.2f}"
        )
        self.lbl_drift.setText(f"drift vx/vy: {state.tracking.vx:.2f}/{state.tracking.vy:.2f} px/s")
        if hasattr(self, "lbl_goto_samples"):
            self.lbl_goto_samples.setText(str(getattr(state.goto, "manual_samples", 0)))

        if state.goto.pointing_valid:
            ra_str = self._format_ra_deg(state.goto.pointing_ra_deg)
            dec_str = self._format_dec_deg(state.goto.pointing_dec_deg)
            az_str = self._format_az_deg(state.goto.pointing_az_deg)
            alt_str = self._format_alt_deg(state.goto.pointing_alt_deg)
            self.lbl_coords.setText(f"RA/Dec: {ra_str} {dec_str} | Az/Alt: {az_str} {alt_str}")
        elif state.goto.synced or state.platesolving.last_ok:
            ra_str = self._format_ra_deg(state.platesolving.center_ra_deg)
            dec_str = self._format_dec_deg(state.platesolving.center_dec_deg)
            self.lbl_coords.setText(f"RA/Dec: {ra_str} {dec_str} | Az/Alt: -- --")
        else:
            self.lbl_coords.setText("RA/Dec: -- -- | Az/Alt: -- --")

        self._update_chips_from_state(state)
        self._update_ps_outputs(state)
        self._update_error_banner(state)

    def _render_frame(self) -> None:
        preview = self.runner.get_latest_preview_jpeg()
        live_pix = self._pixmap_from_jpeg(preview)
        if live_pix is None:
            live_pix = QPixmap.fromImage(self._render_mock_frame())

        stack_preview = self.runner.get_state().stacking.preview_jpeg
        stack_pix = self._pixmap_from_jpeg(stack_preview)
        if stack_pix is None:
            stack_pix = live_pix

        self.live_view.setPixmap(
            live_pix.scaled(
                self.live_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.stacked_view.setPixmap(
            stack_pix.scaled(
                self.stacked_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _render_mock_frame(self) -> QImage:
        img = QImage(self.base_w, self.base_h, QImage.Format.Format_RGB32)
        img.fill(0x101014)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        dx = math.cos(self._t_ms / 1000.0) * 0.5
        dy = math.sin(self._t_ms / 1000.0) * 0.5

        for (x, y, amp) in self._stars:
            xx = (x + dx) % self.base_w
            yy = (y + dy) % self.base_h
            c = int(min(255, 120 + amp))
            painter.setPen(QPen(QColor(c, c, c), 1))
            painter.drawPoint(int(xx), int(yy))

        painter.end()

        return self.renderer.render(
            img,
            toggles=self.overlay_toggles,
            detections=[],
            n_seeds=0,
            drift=DriftInfo(0.0, 0.0),
        )

    def _pixmap_from_jpeg(self, data: Optional[bytes]) -> Optional[QPixmap]:
        if not data:
            return None
        image = QImage.fromData(data)
        if image.isNull():
            return None
        return QPixmap.fromImage(image)

    def _update_ps_outputs(self, state) -> None:
        if not self._ps_outputs_enabled:
            return
        self.lbl_ps_status.setText(state.platesolving.status.value)
        self.lbl_ps_inliers.setText(str(state.platesolving.n_inliers))
        self.lbl_ps_rms.setText(f"{state.platesolving.rms_px:.2f}")
        if state.platesolving.last_ok:
            center_ra = self._format_ra_deg(state.platesolving.center_ra_deg)
            center_dec = self._format_dec_deg(state.platesolving.center_dec_deg)
            self.lbl_ps_center.setText(f"{center_ra} / {center_dec}")
        else:
            self.lbl_ps_center.setText("-- / --")
        self.lbl_ps_theta.setText(f"{state.platesolving.theta_deg:.2f}")
        self.lbl_ps_dxdy.setText(f"{state.platesolving.dx_px:.1f} / {state.platesolving.dy_px:.1f}")

    def _update_chips_from_state(self, state) -> None:
        self.ch_cam.set_mode("green" if state.camera.connected else "red")
        self.ch_mount.set_mode("green" if state.mount.connected else "red")
        self.ch_sync.set_mode("green" if state.goto.synced else "red")
        self.ch_tracking.set_mode("active" if state.tracking.enabled else "neutral")
        self.ch_stacking.set_mode("active" if state.stacking.enabled else "neutral")
        self.ch_od.set_mode("active" if self.od_enabled else "neutral")
        self.ch_ps.set_mode("active" if state.platesolving.busy else "neutral")
        self.ch_goto.set_mode("active" if state.goto.busy else "neutral")

    def _update_error_banner(self, state) -> None:
        errors = []
        if state.camera.last_error:
            errors.append(f"camera: {state.camera.last_error}")
        if state.mount.last_error:
            errors.append(f"mount: {state.mount.last_error}")
        if state.tracking.last_error:
            errors.append(f"tracking: {state.tracking.last_error}")
        if state.stacking.last_error:
            errors.append(f"stacking: {state.stacking.last_error}")
        if state.platesolving.reason:
            errors.append(f"platesolving: {state.platesolving.reason}")
        if state.goto.reason:
            errors.append(f"goto: {state.goto.reason}")
        if errors:
            self.lbl_errors.setText(f"Errors: {' | '.join(errors)}")
        else:
            self.lbl_errors.setText("Errors: none")

    def _format_ra_deg(self, ra_deg: float) -> str:
        total_seconds = ra_deg * 240.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    def _format_dec_deg(self, dec_deg: float) -> str:
        sign = "+" if dec_deg >= 0 else "-"
        dec_abs = abs(dec_deg)
        degrees = int(dec_abs)
        minutes = int((dec_abs - degrees) * 60)
        seconds = (dec_abs - degrees - minutes / 60) * 3600
        return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:05.2f}"

    def _format_az_deg(self, az_deg: float) -> str:
        return f"{(float(az_deg) % 360.0):06.2f}°"

    def _format_alt_deg(self, alt_deg: float) -> str:
        return f"{float(alt_deg):+06.2f}°"

    def _build_goto_target(self) -> Optional[object]:
        mode = self.dd_goto_mode.currentText()
        if mode.startswith("name"):
            value = self.ed_goto_name.text().strip()
            return value or None
        if mode.startswith("planet"):
            return self.dd_goto_planet.currentText().strip()
        if mode == "radec":
            if self.dd_radec_fmt.currentText() == "deg":
                return {"ra_deg": float(self.ds_ra.value()), "dec_deg": float(self.ds_dec.value())}
            ra = self.ed_ra_hms.text().strip()
            dec = self.ed_dec_dms.text().strip()
            if not ra or not dec:
                return None
            return f"{ra} {dec}"
        return {"az_deg": float(self.ds_az.value()), "alt_deg": float(self.ds_alt.value())}

    def _init_star_catalog(self, n: int = 100) -> list[tuple[float, float, float]]:
        stars: list[tuple[float, float, float]] = []
        for _ in range(n):
            x = self._rng.uniform(0, self.base_w)
            y = self._rng.uniform(0, self.base_h)
            amp = self._rng.uniform(0, 140)
            stars.append((x, y, amp))
        stars.sort(key=lambda t: t[2], reverse=True)
        return stars

    def _log(self, msg: str) -> None:
        self.log.append(msg)


@dataclass
class UI:
    app: QApplication
    window: AstroPanoptesWindow
    runner: AppRunner


def build_ui(runner: Optional[AppRunner] = None, *, cfg: Optional[AppConfig] = None) -> AstroPanoptesWindow:
    if runner is None:
        cfg = cfg or AppConfig()
        runner = AppRunner(cfg)
        window = AstroPanoptesWindow(runner, cfg)
        runner.start()
        return window
    cfg = cfg or runner.cfg
    return AstroPanoptesWindow(runner, cfg)


def show_ui(*, start_app: bool = True, start_runner: bool = True) -> UI:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    cfg = AppConfig()
    runner = AppRunner(cfg)
    window = AstroPanoptesWindow(runner, cfg)
    if start_runner:
        runner.start()
    window.showMaximized()

    if start_app and created:
        app.exec()

    return UI(app=app, window=window, runner=runner)


def main() -> None:
    show_ui()


__all__ = [
    "AstroPanoptesWindow",
    "UI",
    "build_ui",
    "show_ui",
    "main",
]


if __name__ == "__main__":
    main()
