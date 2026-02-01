from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
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
class SyncInfo:
    synced: bool = False
    ra_str: str = "--"
    dec_str: str = "--"
    alt_str: str = "--"
    az_str: str = "--"


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

        seeds = []
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


class AstroPanoptesWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AstroPanoptes")
        self.resize(1280, 760)

        self.cam_connected = False
        self.mount_connected = False
        self.synced = False

        self.tracking_on = False
        self.stacking_on = False
        self.od_on = False
        self.platesolve_running = False
        self.goto_running = False

        self.vx = 0.0
        self.vy = 0.0
        self.fps_view = 0.0

        self.sync_info = SyncInfo(False, "--", "--", "--", "--")
        self.ps = PlatesolveSummary(False, "idle", 0, 0.0, 0.0, 0.0, 0.0, "--", "--")

        self.exp_ms = 20.0
        self.gain = 100

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
        self._build_top_toolbar()
        self._build_docks()
        self._build_menu()

        self._tick = QTimer(self)
        self._tick.setInterval(50)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(100)
        self._frame_timer.timeout.connect(self._render_frame)
        self._frame_timer.start()

        self._t_ms = 0.0
        self._log("Demo loaded.")

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

        self.btn_connect_all = QPushButton("Connect all")
        self.btn_disconnect_all = QPushButton("Disconnect all")
        self.btn_connect_all.clicked.connect(self._connect_all)
        self.btn_disconnect_all.clicked.connect(self._disconnect_all)

        self.ch_cam = Chip("Camera")
        self.ch_mount = Chip("Mount")
        self.ch_sync = Chip("Sync")
        self.ch_tracking = Chip("Tracking")
        self.ch_stacking = Chip("Stacking")
        self.ch_od = Chip("Object Detection")
        self.ch_ps = Chip("Plate Solving")
        self.ch_goto = Chip("GoTo")

        self.lbl_fps = QLabel("FPS view/max: --/--")
        self.lbl_drift = QLabel("drift vx/vy: --/-- px/s")
        self.lbl_coords = QLabel("RA/Dec: -- -- | Alt/Az: -- --")
        for label in (self.lbl_fps, self.lbl_drift, self.lbl_coords):
            label.setStyleSheet(
                "QLabel { padding:4px 10px; border:1px solid #3a3a3a; border-radius:10px; "
                "background:#121212; color:#ddd; }"
            )

        row = QHBoxLayout()
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        row.addWidget(self.btn_connect_all)
        row.addWidget(self.btn_disconnect_all)
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

        layout = QVBoxLayout(top)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(row)
        layout.addLayout(metrics)

        self.tb_top.addWidget(top)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tb_top)

        self._update_chips()

    def _build_docks(self) -> None:
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )

        self.modules_tabs = self._build_modules_tabs()
        self.dock_modules = QDockWidget("Modules", self)
        self.dock_modules.setWidget(self.modules_tabs)
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

        self.b_up.clicked.connect(lambda: self._manual_move("alt+"))
        self.b_down.clicked.connect(lambda: self._manual_move("alt-"))
        self.b_left.clicked.connect(lambda: self._manual_move("az-"))
        self.b_right.clicked.connect(lambda: self._manual_move("az+"))
        self.b_stop.clicked.connect(lambda: self._log("[manual] STOP"))

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

    def _build_modules_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._tab_camera(), "Camera")
        tabs.addTab(self._tab_tracking(), "Tracking")
        tabs.addTab(self._tab_stacking(), "Stacking")
        tabs.addTab(self._tab_od(), "Object Detection")
        tabs.addTab(self._tab_platesolve(), "Plate Solving")
        tabs.addTab(self._tab_goto(), "GoTo")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(tabs)

        self.modules_tabs = tabs
        return wrap

    def _tab_camera(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Camera")
        form = QFormLayout()

        self.ds_exp_ms = QDoubleSpinBox()
        self.ds_exp_ms.setRange(0.1, 10_000.0)
        self.ds_exp_ms.setDecimals(1)
        self.ds_exp_ms.setValue(self.exp_ms)
        self.ds_exp_ms.setSuffix(" ms")

        self.sb_gain = QSpinBox()
        self.sb_gain.setRange(0, 6000)
        self.sb_gain.setValue(self.gain)

        self.btn_apply_cam = QPushButton("Apply")
        self.btn_apply_cam.clicked.connect(self._camera_apply)

        form.addRow("Exposure:", self.ds_exp_ms)
        form.addRow("Gain:", self.sb_gain)
        form.addRow(self.btn_apply_cam)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget

    def _tab_tracking(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Tracking")
        form = QFormLayout()

        self.btn_tr_start = QPushButton("Start")
        self.btn_tr_stop = QPushButton("Stop")
        self.btn_tr_reset = QPushButton("Reset")

        row = QHBoxLayout()
        row.addWidget(self.btn_tr_start)
        row.addWidget(self.btn_tr_stop)
        row.addWidget(self.btn_tr_reset)
        row.addStretch(1)

        self.btn_tr_start.clicked.connect(lambda: self._set_tracking(True))
        self.btn_tr_stop.clicked.connect(lambda: self._set_tracking(False))
        self.btn_tr_reset.clicked.connect(lambda: self._log("[tracking] Reset"))

        form.addRow(row)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget

    def _tab_stacking(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Stacking")
        form = QFormLayout()

        self.btn_st_start = QPushButton("Start")
        self.btn_st_stop = QPushButton("Stop")
        self.btn_st_reset = QPushButton("Reset")
        self.btn_st_save = QPushButton("Save Stack")

        row = QHBoxLayout()
        for button in [self.btn_st_start, self.btn_st_stop, self.btn_st_reset, self.btn_st_save]:
            row.addWidget(button)
        row.addStretch(1)

        self.btn_st_start.clicked.connect(lambda: self._set_stacking(True))
        self.btn_st_stop.clicked.connect(lambda: self._set_stacking(False))
        self.btn_st_reset.clicked.connect(lambda: self._log("[stacking] Reset"))
        self.btn_st_save.clicked.connect(lambda: self._log("[stacking] Save Stack"))

        form.addRow(row)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget

    def _tab_od(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Object Detection")
        form = QFormLayout()

        self.btn_od_start = QPushButton("Start")
        self.btn_od_stop = QPushButton("Stop")

        row = QHBoxLayout()
        row.addWidget(self.btn_od_start)
        row.addWidget(self.btn_od_stop)
        row.addStretch(1)

        self.btn_od_start.clicked.connect(self._od_start)
        self.btn_od_stop.clicked.connect(self._od_stop)

        self.sb_od_minarea = QSpinBox()
        self.sb_od_minarea.setRange(1, 500)
        self.sb_od_minarea.setValue(5)

        self.ds_od_sigma = QDoubleSpinBox()
        self.ds_od_sigma.setRange(0.1, 20.0)
        self.ds_od_sigma.setValue(3.0)

        self.sb_od_maxdet = QSpinBox()
        self.sb_od_maxdet.setRange(1, 5000)
        self.sb_od_maxdet.setValue(500)

        self.sb_od_bw = QSpinBox()
        self.sb_od_bw.setRange(4, 512)
        self.sb_od_bw.setValue(64)

        self.sb_od_bh = QSpinBox()
        self.sb_od_bh.setRange(4, 512)
        self.sb_od_bh.setValue(64)

        form.addRow(row)
        form.addRow("minarea:", self.sb_od_minarea)
        form.addRow("thresh_sigma:", self.ds_od_sigma)
        form.addRow("max_det:", self.sb_od_maxdet)
        form.addRow("bw:", self.sb_od_bw)
        form.addRow("bh:", self.sb_od_bh)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)
        return widget

    def _tab_platesolve(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Plate Solving")
        form = QFormLayout()

        self.ed_ps_target = QLineEdit()
        self.ed_ps_target.setPlaceholderText("target: name(SIMBAD) | RA/Dec | Alt/Az")

        self.ds_ps_radius = QDoubleSpinBox()
        self.ds_ps_radius.setRange(0.1, 30.0)
        self.ds_ps_radius.setValue(1.0)
        self.ds_ps_radius.setSuffix(" deg")

        self.sb_ps_maxdet = QSpinBox()
        self.sb_ps_maxdet.setRange(3, 200)
        self.sb_ps_maxdet.setValue(7)

        self.sb_ps_nseeds = QSpinBox()
        self.sb_ps_nseeds.setRange(0, 10)
        self.sb_ps_nseeds.setValue(3)

        self.sb_ps_mininl = QSpinBox()
        self.sb_ps_mininl.setRange(1, 100)
        self.sb_ps_mininl.setValue(4)

        self.btn_ps_solve = QPushButton("Solve")
        self.btn_ps_solve.clicked.connect(self._platesolve_start)

        form.addRow("target:", self.ed_ps_target)
        form.addRow("search radius:", self.ds_ps_radius)
        form.addRow("max detections:", self.sb_ps_maxdet)
        form.addRow("n_seeds:", self.sb_ps_nseeds)
        form.addRow("min inliers:", self.sb_ps_mininl)
        form.addRow(self.btn_ps_solve)

        out = QGroupBox("Outputs")
        out_form = QFormLayout()
        self.lbl_ps_status = QLabel("idle")
        self.lbl_ps_inliers = QLabel("0")
        self.lbl_ps_rms = QLabel("0.0")
        self.lbl_ps_center = QLabel("-- / --")
        self.lbl_ps_theta = QLabel("0.0")
        self.lbl_ps_dxdy = QLabel("0.0 / 0.0")
        out_form.addRow("status:", self.lbl_ps_status)
        out_form.addRow("inliers:", self.lbl_ps_inliers)
        out_form.addRow("rms_px:", self.lbl_ps_rms)
        out_form.addRow("center ra/dec:", self.lbl_ps_center)
        out_form.addRow("theta_deg:", self.lbl_ps_theta)
        out_form.addRow("dx/dy px:", self.lbl_ps_dxdy)
        out.setLayout(out_form)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addWidget(out)
        layout.addStretch(1)
        return widget

    def _tab_goto(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("GoTo")
        form = QFormLayout()

        self.dd_goto_mode = QComboBox()
        self.dd_goto_mode.addItems(["name (SIMBAD)", "planet/moon", "radec", "altaz"])

        self.ed_goto_name = QLineEdit()
        self.ed_goto_name.setPlaceholderText("Object name (SIMBAD)")

        self.dd_goto_planet = QComboBox()
        self.dd_goto_planet.addItems(
            ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
        )

        self.ds_ra = QDoubleSpinBox()
        self.ds_ra.setRange(0.0, 360.0)
        self.ds_ra.setDecimals(6)

        self.ds_dec = QDoubleSpinBox()
        self.ds_dec.setRange(-90.0, 90.0)
        self.ds_dec.setDecimals(6)

        self.ed_ra_hms = QLineEdit()
        self.ed_ra_hms.setPlaceholderText("RA HH:MM:SS(.s)")

        self.ed_dec_dms = QLineEdit()
        self.ed_dec_dms.setPlaceholderText("Dec ±DD:MM:SS")

        self.dd_radec_fmt = QComboBox()
        self.dd_radec_fmt.addItems(["deg", "HMS/DMS"])

        self.ds_az = QDoubleSpinBox()
        self.ds_az.setRange(0.0, 360.0)
        self.ds_az.setDecimals(6)

        self.ds_alt = QDoubleSpinBox()
        self.ds_alt.setRange(0.0, 90.0)
        self.ds_alt.setDecimals(6)

        self.tgt_frame = QFrame()
        self.tgt_v = QVBoxLayout(self.tgt_frame)
        self.tgt_v.setContentsMargins(0, 0, 0, 0)
        self.tgt_v.setSpacing(6)

        self.cb_fb = QCheckBox("Platesolve feedback")
        self.sb_stages = QSpinBox()
        self.sb_stages.setRange(0, 20)
        self.sb_stages.setValue(0)
        self.sb_stages.setEnabled(False)
        self.cb_fb.toggled.connect(self.sb_stages.setEnabled)

        rowfb = QHBoxLayout()
        rowfb.addWidget(self.cb_fb)
        rowfb.addSpacing(10)
        rowfb.addWidget(QLabel("Stages:"))
        rowfb.addWidget(self.sb_stages)
        rowfb.addStretch(1)

        self.btn_goto = QPushButton("GoTo")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_autocal = QPushButton("AutoCalibrate")
        self.btn_home = QPushButton("Home")

        rowb = QHBoxLayout()
        for button in [self.btn_goto, self.btn_cancel, self.btn_autocal, self.btn_home]:
            rowb.addWidget(button)
        rowb.addStretch(1)

        self.btn_goto.clicked.connect(self._goto_start)
        self.btn_cancel.clicked.connect(self._goto_cancel)
        self.btn_autocal.clicked.connect(self._autocalibrate)
        self.btn_home.clicked.connect(self._home)

        form.addRow("mode:", self.dd_goto_mode)
        form.addRow("target:", self.tgt_frame)
        form.addRow(rowfb)
        form.addRow(rowb)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)

        self.dd_goto_mode.currentIndexChanged.connect(self._goto_mode_switch)
        self.dd_radec_fmt.currentIndexChanged.connect(self._goto_mode_switch)
        self._goto_mode_switch()
        return widget

    def _goto_mode_switch(self) -> None:
        while self.tgt_v.count():
            item = self.tgt_v.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        mode = self.dd_goto_mode.currentText()
        if mode.startswith("name"):
            self.tgt_v.addWidget(self.ed_goto_name)
        elif mode.startswith("planet"):
            self.tgt_v.addWidget(self.dd_goto_planet)
        elif mode == "radec":
            self.tgt_v.addWidget(self.dd_radec_fmt)
            if self.dd_radec_fmt.currentText() == "deg":
                row = QHBoxLayout()
                row.addWidget(QLabel("RA°"))
                row.addWidget(self.ds_ra)
                row.addSpacing(10)
                row.addWidget(QLabel("Dec°"))
                row.addWidget(self.ds_dec)
                row.addStretch(1)
                wrap = QFrame()
                wrap_layout = QHBoxLayout(wrap)
                wrap_layout.setContentsMargins(0, 0, 0, 0)
                wrap_layout.addLayout(row)
                self.tgt_v.addWidget(wrap)
            else:
                row = QHBoxLayout()
                row.addWidget(QLabel("RA"))
                row.addWidget(self.ed_ra_hms)
                row.addSpacing(10)
                row.addWidget(QLabel("Dec"))
                row.addWidget(self.ed_dec_dms)
                row.addStretch(1)
                wrap = QFrame()
                wrap_layout = QHBoxLayout(wrap)
                wrap_layout.setContentsMargins(0, 0, 0, 0)
                wrap_layout.addLayout(row)
                self.tgt_v.addWidget(wrap)
        else:
            row = QHBoxLayout()
            row.addWidget(QLabel("Az°"))
            row.addWidget(self.ds_az)
            row.addSpacing(10)
            row.addWidget(QLabel("Alt°"))
            row.addWidget(self.ds_alt)
            row.addStretch(1)
            wrap = QFrame()
            wrap_layout = QHBoxLayout(wrap)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.addLayout(row)
            self.tgt_v.addWidget(wrap)

    def _camera_apply(self) -> None:
        self.exp_ms = float(self.ds_exp_ms.value())
        self.gain = int(self.sb_gain.value())
        self._log(f"[camera] apply exposure={self.exp_ms:.1f} ms gain={self.gain}")

    def _connect_all(self) -> None:
        self.cam_connected = True
        self.mount_connected = True
        self._log("[top] Connect all")
        self._update_chips()

    def _disconnect_all(self) -> None:
        self.cam_connected = False
        self.mount_connected = False
        self.synced = False
        self.sync_info = SyncInfo(False, "--", "--", "--", "--")
        self._log("[top] Disconnect all")
        self._update_chips()

    def _od_start(self) -> None:
        self.od_on = True
        self._log("[od] Start")
        self._update_chips()

    def _od_stop(self) -> None:
        self.od_on = False
        self._log("[od] Stop")
        self._update_chips()

    def _set_tracking(self, on: bool) -> None:
        self.tracking_on = on
        self._log(f"[tracking] {'Start' if on else 'Stop'}")
        self._update_chips()

    def _set_stacking(self, on: bool) -> None:
        self.stacking_on = on
        self._log(f"[stacking] {'Start' if on else 'Stop'}")
        self._update_chips()

    def _platesolve_start(self) -> None:
        if self.platesolve_running:
            return
        self.platesolve_running = True
        self.ps.running = True
        self.ps.status = "running"
        self._log("[plate solving] Solve (mock) started")
        self._update_chips()
        QTimer.singleShot(2000, self._platesolve_finish)

    def _platesolve_finish(self) -> None:
        self.platesolve_running = False
        self.ps.running = False
        self.ps.status = "ok"
        self.ps.inliers = random.randint(4, 15)
        self.ps.rms_px = random.random() * 1.5
        self.ps.theta_deg = random.uniform(-5, 5)
        self.ps.dx_px = random.uniform(-50, 50)
        self.ps.dy_px = random.uniform(-50, 50)
        self.ps.center_ra = "05:35:17.3"
        self.ps.center_dec = "-05:23:28"
        self._log("[plate solving] finished OK (mock)")
        self._update_ps_outputs()
        self._update_chips()

    def _goto_start(self) -> None:
        if self.goto_running:
            return
        self.goto_running = True
        self._log(
            f"[goto] GoTo started (mock) feedback={self.cb_fb.isChecked()} stages={self.sb_stages.value()}"
        )
        self._update_chips()
        QTimer.singleShot(2500, self._goto_finish)

    def _goto_finish(self) -> None:
        self.goto_running = False
        self._log("[goto] GoTo finished (mock)")
        self._update_chips()

    def _goto_cancel(self) -> None:
        self.goto_running = False
        self._log("[goto] Cancel (mock)")
        self._update_chips()

    def _autocalibrate(self) -> None:
        self.synced = True
        self.sync_info = SyncInfo(
            True,
            ra_str="05:35:17.3",
            dec_str="-05:23:28",
            alt_str="45.12°",
            az_str="132.80°",
        )
        self._log("[goto] AutoCalibrate -> Sync=True (mock)")
        self._update_chips()

    def _home(self) -> None:
        self._log("[goto] Home -> startup pose (mock)")

    def _manual_move(self, direction: str) -> None:
        steps = int(self.sb_steps.value())
        delay_us = int(self.sb_delay.value())
        self._log(f"[manual] move dir={direction} steps={steps} delay_us={delay_us} (mock enqueue)")

    def _on_tick(self) -> None:
        self._t_ms += 50.0
        t = self._t_ms / 1000.0

        self.vx = 15.0 * math.cos(t * 0.7) + 5.0 * math.sin(t * 1.4)
        self.vy = 8.0 * math.sin(t * 0.5)

        fps_max = max(0.1, 1000.0 / max(0.1, float(self.exp_ms)))
        self.lbl_fps.setText(f"FPS view/max: {self.fps_view:.2f}/{fps_max:.2f}")
        self.lbl_drift.setText(f"drift vx/vy: {self.vx:.2f}/{self.vy:.2f} px/s")

        if self.synced:
            self.lbl_coords.setText(
                f"RA/Dec: {self.sync_info.ra_str} {self.sync_info.dec_str} | "
                f"Alt/Az: {self.sync_info.alt_str} {self.sync_info.az_str}"
            )
        else:
            self.lbl_coords.setText("RA/Dec: -- -- | Alt/Az: -- --")

    def _render_frame(self) -> None:
        img = QImage(self.base_w, self.base_h, QImage.Format.Format_RGB32)
        img.fill(0x101014)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        dx = self.vx * 0.02
        dy = self.vy * 0.02

        for (x, y, amp) in self._stars:
            xx = (x + dx) % self.base_w
            yy = (y + dy) % self.base_h
            c = int(min(255, 120 + amp))
            painter.setPen(QPen(QColor(c, c, c), 1))
            painter.drawPoint(int(xx), int(yy))

        painter.end()

        dets: List[Detection] = []
        if self.od_on:
            for i in range(min(80, len(self._stars))):
                x, y, amp = self._stars[i]
                jx = x + random.uniform(-1.2, 1.2)
                jy = y + random.uniform(-1.2, 1.2)
                flux = float(amp) + random.random() * 20.0
                a = 3.0 + random.random() * 2.0
                b = 3.0 + random.random() * 2.0
                th = random.uniform(-0.5, 0.5)
                dets.append(Detection(jx, jy, a, b, th, flux))
            dets = sorted(dets, key=lambda det: det.flux, reverse=True)[: int(self.sb_od_maxdet.value())]

        self.fps_view = 1000.0 / max(1.0, float(self._frame_timer.interval()))
        n_seeds = int(self.sb_ps_nseeds.value())

        out = self.renderer.render(
            img,
            toggles=self.overlay_toggles,
            detections=dets,
            n_seeds=n_seeds,
            drift=DriftInfo(self.vx, self.vy),
        )

        pix = QPixmap.fromImage(out)
        self.live_view.setPixmap(
            pix.scaled(
                self.live_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.stacked_view.setPixmap(
            pix.scaled(
                self.stacked_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _init_star_catalog(self, n: int = 100) -> list[tuple[float, float, float]]:
        stars: list[tuple[float, float, float]] = []
        for _ in range(n):
            x = self._rng.uniform(0, self.base_w)
            y = self._rng.uniform(0, self.base_h)
            amp = self._rng.uniform(0, 140)
            stars.append((x, y, amp))
        stars.sort(key=lambda t: t[2], reverse=True)
        return stars

    def _update_ps_outputs(self) -> None:
        self.lbl_ps_status.setText(self.ps.status)
        self.lbl_ps_inliers.setText(str(self.ps.inliers))
        self.lbl_ps_rms.setText(f"{self.ps.rms_px:.2f}")
        self.lbl_ps_center.setText(f"{self.ps.center_ra} / {self.ps.center_dec}")
        self.lbl_ps_theta.setText(f"{self.ps.theta_deg:.2f}")
        self.lbl_ps_dxdy.setText(f"{self.ps.dx_px:.1f} / {self.ps.dy_px:.1f}")

    def _update_chips(self) -> None:
        self.ch_cam.set_mode("green" if self.cam_connected else "red")
        self.ch_mount.set_mode("green" if self.mount_connected else "red")
        self.ch_sync.set_mode("green" if self.synced else "red")
        self.ch_tracking.set_mode("active" if self.tracking_on else "neutral")
        self.ch_stacking.set_mode("active" if self.stacking_on else "neutral")
        self.ch_od.set_mode("active" if self.od_on else "neutral")
        self.ch_ps.set_mode("active" if self.platesolve_running else "neutral")
        self.ch_goto.set_mode("active" if self.goto_running else "neutral")

    def _log(self, msg: str) -> None:
        self.log.append(msg)


@dataclass
class UI:
    app: QApplication
    window: AstroPanoptesWindow


def build_ui() -> AstroPanoptesWindow:
    return AstroPanoptesWindow()


def show_ui(*, start_app: bool = True) -> UI:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    window = AstroPanoptesWindow()
    window.showMaximized()

    if start_app and created:
        app.exec()

    return UI(app=app, window=window)


__all__ = [
    "AstroPanoptesWindow",
    "UI",
    "build_ui",
    "show_ui",
]
