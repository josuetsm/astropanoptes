from __future__ import annotations

import datetime as _dt
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


OBSERVER_PRESETS = (
    ("San Carlos", {"lat_deg": -36.4248, "lon_deg": -71.9580, "height_m": 161.0}),
    ("Algarrobo", {"lat_deg": -33.3667, "lon_deg": -71.6667, "height_m": 28.0}),
    ("Estación Central (Santiago)", {"lat_deg": -33.4569, "lon_deg": -70.6990, "height_m": 520.0}),
)
BARLOW_FACTORS = (1, 2, 3, 4, 5)
STACKING_DRIZZLE_SCALES = (1.0, 2.0, 3.0)

if TYPE_CHECKING:
    from ui.pyqt6_app import AstroPanoptesWindow


def _set_option_tooltip(widget: QWidget, text: str) -> None:
    widget.setToolTip(text)
    widget.setToolTipDuration(12_000)
    for child in widget.findChildren(QLineEdit):
        child.setToolTip(text)
        child.setToolTipDuration(12_000)


def _option_label(text: str, tooltip: str) -> QLabel:
    label = QLabel(text)
    _set_option_tooltip(label, tooltip)
    return label


def _add_option_row(form: QFormLayout, label: str, widget: QWidget, tooltip: str) -> None:
    _set_option_tooltip(widget, tooltip)
    form.addRow(_option_label(label, tooltip), widget)


class GaiaCoverageMap(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._coverage: dict[str, object] = {}
        self.setMinimumHeight(260)
        self.setToolTip("Mapa horizontal local: azimut 0°=N, 90°=E; altitud respecto del horizonte")

    def set_coverage(self, coverage: dict[str, object]) -> None:
        self._coverage = dict(coverage)
        self.update()

    @staticmethod
    def _map_point(plot: QRectF, az_deg: float, alt_deg: float) -> QPointF:
        x = plot.left() + ((float(az_deg) % 360.0) / 360.0) * plot.width()
        y = plot.top() + ((90.0 - float(alt_deg)) / 180.0) * plot.height()
        return QPointF(x, y)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101317"))

        plot = QRectF(self.rect()).adjusted(42.0, 20.0, -18.0, -32.0)
        if plot.width() <= 0.0 or plot.height() <= 0.0:
            painter.end()
            return

        horizon_y = plot.top() + plot.height() * 0.5
        painter.fillRect(
            QRectF(plot.left(), horizon_y, plot.width(), plot.bottom() - horizon_y),
            QColor("#17191d"),
        )

        painter.setPen(QPen(QColor("#303944"), 1))
        painter.drawRect(plot)
        for az in (0.0, 90.0, 180.0, 270.0, 360.0):
            x = plot.left() + (az / 360.0) * plot.width()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        for alt in (-60.0, -30.0, 0.0, 30.0, 60.0):
            y = plot.top() + ((90.0 - alt) / 180.0) * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        painter.setPen(QPen(QColor("#8f9aa6"), 2))
        painter.drawLine(QPointF(plot.left(), horizon_y), QPointF(plot.right(), horizon_y))

        painter.setPen(QColor("#89939e"))
        for az, label in (
            (0.0, "N 0°"),
            (90.0, "E 90°"),
            (180.0, "S 180°"),
            (270.0, "W 270°"),
            (360.0, "N 360°"),
        ):
            x = plot.left() + (az / 360.0) * plot.width()
            painter.drawText(QPointF(x - 18.0, plot.bottom() + 18.0), label)
        for alt in (-60.0, 0.0, 60.0):
            y = plot.top() + ((90.0 - alt) / 180.0) * plot.height()
            painter.drawText(QPointF(4.0, y + 4.0), f"{alt:+.0f}°")

        az_values = np.asarray(self._coverage.get("tile_az_deg", []), dtype=np.float64)
        alt_values = np.asarray(self._coverage.get("tile_alt_deg", []), dtype=np.float64)
        cached = {int(pix) for pix in self._coverage.get("cached_tiles", [])}
        required = {int(pix) for pix in self._coverage.get("field_required_tiles", [])}

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#26303a"))
        for az, alt in zip(az_values, alt_values):
            point = self._map_point(plot, float(az), float(alt))
            painter.drawEllipse(point, 0.8, 0.8)

        tile_radius = max(1.6, min(4.0, plot.width() / 360.0 * 1.7))
        for pix in cached:
            if pix < 0 or pix >= len(az_values):
                continue
            color = QColor("#2aa876") if alt_values[pix] >= 0.0 else QColor("#245b48")
            painter.setBrush(color)
            point = self._map_point(plot, az_values[pix], alt_values[pix])
            painter.drawEllipse(point, tile_radius, tile_radius)

        for pix in required:
            if pix < 0 or pix >= len(az_values):
                continue
            color = QColor("#73e2a7") if pix in cached else QColor("#ff6b6b")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            point = self._map_point(plot, az_values[pix], alt_values[pix])
            painter.drawEllipse(point, tile_radius + 3.0, tile_radius + 3.0)

        center_az = self._coverage.get("center_az_deg")
        center_alt = self._coverage.get("center_alt_deg")
        radius_deg = self._coverage.get("field_radius_deg")
        if center_az is not None and center_alt is not None:
            point = self._map_point(plot, float(center_az), float(center_alt))
            painter.setPen(QPen(QColor("#ffd166"), 2))
            painter.drawLine(QPointF(point.x() - 6.0, point.y()), QPointF(point.x() + 6.0, point.y()))
            painter.drawLine(QPointF(point.x(), point.y() - 6.0), QPointF(point.x(), point.y() + 6.0))

            if radius_deg is not None and float(radius_deg) > 0.0:
                cos_alt = max(0.15, abs(math.cos(math.radians(float(center_alt)))))
                rx = float(radius_deg) / cos_alt / 360.0 * plot.width()
                ry = float(radius_deg) / 180.0 * plot.height()
                painter.setPen(QPen(QColor("#ffd166"), 1, Qt.PenStyle.DashLine))
                painter.drawEllipse(point, rx, ry)

        painter.end()


class GaiaTabMixin:
    def _tab_gaia(self: "AstroPanoptesWindow") -> QWidget:
        self.gaia_tab = QWidget()
        layout = QVBoxLayout(self.gaia_tab)
        layout.setSpacing(10)

        summary = QGroupBox("Cobertura Gaia local")
        form = QFormLayout(summary)

        self.lbl_gaia_field_status = QLabel("Campo actual: sin posición")
        self.lbl_gaia_field_status.setStyleSheet(
            "QLabel { padding:6px 10px; border-radius:8px; "
            "background:#24282d; border:1px solid #444b53; color:#e5e7eb; font-weight:600; }"
        )
        self.lbl_gaia_tiles = QLabel("--")
        self.lbl_gaia_area = QLabel("--")
        self.lbl_gaia_disk = QLabel("--")
        self.lbl_gaia_field = QLabel("--")
        self.lbl_gaia_center = QLabel("--")
        self.lbl_gaia_projection = QLabel("--")
        self.lbl_gaia_config = QLabel("--")
        self.lbl_gaia_cache_dir = QLabel("--")
        self.lbl_gaia_cache_dir.setWordWrap(True)

        self.btn_gaia_refresh = QPushButton("Actualizar cobertura")
        self.btn_gaia_download = QPushButton("Descargar campo actual")
        self.btn_gaia_refresh.clicked.connect(lambda: self._refresh_gaia_coverage(force=True))
        self.btn_gaia_download.clicked.connect(self._gaia_download_current_field)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_gaia_refresh)
        actions.addWidget(self.btn_gaia_download)
        actions.addStretch(1)

        form.addRow(self.lbl_gaia_field_status)
        form.addRow("Teselas en caché:", self.lbl_gaia_tiles)
        form.addRow("Área del cielo:", self.lbl_gaia_area)
        form.addRow("Tamaño en disco:", self.lbl_gaia_disk)
        form.addRow("Campo actual:", self.lbl_gaia_field)
        form.addRow("Az/Alt / radio:", self.lbl_gaia_center)
        form.addRow("Proyección local:", self.lbl_gaia_projection)
        form.addRow("Configuración:", self.lbl_gaia_config)
        form.addRow("Directorio:", self.lbl_gaia_cache_dir)
        form.addRow(actions)

        self.gaia_coverage_map = GaiaCoverageMap()
        legend = QLabel(
            "<span style='color:#2aa876'>●</span> tesela cacheada &nbsp;&nbsp; "
            "<span style='color:#73e2a7'>○</span> requerida y disponible &nbsp;&nbsp; "
            "<span style='color:#ff6b6b'>○</span> requerida y faltante &nbsp;&nbsp; "
            "<span style='color:#ffd166'>＋</span> campo actual"
        )
        legend.setTextFormat(Qt.TextFormat.RichText)

        layout.addWidget(summary)
        layout.addWidget(self.gaia_coverage_map, stretch=1)
        layout.addWidget(legend)
        self._gaia_last_refresh_t = 0.0
        self._gaia_last_download_status = None
        return self.gaia_tab

    @staticmethod
    def _gaia_format_bytes(value: object) -> str:
        size = float(value or 0.0)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                break
            size /= 1024.0
        return f"{size:.1f} {unit}"

    def _gaia_download_current_field(self: "AstroPanoptesWindow") -> None:
        self._download_gaia_current_field()
        self.lbl_gaia_field_status.setText("Campo actual: descarga solicitada")

    def _gaia_tab_selected(self: "AstroPanoptesWindow", index: int) -> None:
        is_gaia = self.modules_tabs.widget(index) is self.gaia_tab
        dock_manual = getattr(self, "dock_manual", None)
        if dock_manual is not None:
            if is_gaia:
                manual_visible = not bool(dock_manual.isHidden())
                self._manual_visible_before_gaia = manual_visible
                self._manual_hidden_for_gaia = manual_visible
                if manual_visible:
                    dock_manual.setVisible(False)
            elif bool(getattr(self, "_manual_hidden_for_gaia", False)):
                dock_manual.setVisible(
                    bool(getattr(self, "_manual_visible_before_gaia", True))
                )
                self._manual_hidden_for_gaia = False

        if is_gaia:
            self._refresh_gaia_coverage(force=True)

    def _gaia_maybe_refresh(self: "AstroPanoptesWindow", state) -> None:
        if not hasattr(self, "gaia_tab") or self.modules_tabs.currentWidget() is not self.gaia_tab:
            return
        debug = state.platesolving.debug_info or {}
        download_status = debug.get("status") if isinstance(debug, dict) else None
        force = (
            download_status != self._gaia_last_download_status
            and download_status in {"GAIA_DOWNLOAD_OK", "GAIA_DOWNLOAD_FAILED"}
        )
        self._gaia_last_download_status = download_status
        self.btn_gaia_download.setEnabled(not bool(state.platesolving.busy))
        self._refresh_gaia_coverage(force=force)

    def _refresh_gaia_coverage(self: "AstroPanoptesWindow", *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._gaia_last_refresh_t) < 5.0:
            return
        self._gaia_last_refresh_t = now
        try:
            coverage = self.runner.get_gaia_coverage()
        except Exception as exc:
            self.lbl_gaia_field_status.setText(f"No se pudo inspeccionar el caché: {type(exc).__name__}")
            self.lbl_gaia_field_status.setStyleSheet(
                "QLabel { padding:6px 10px; border-radius:8px; "
                "background:#3a1515; border:1px solid #7a2a2a; color:#ffecec; font-weight:600; }"
            )
            self._log(f"[gaia] coverage inspection failed: {exc}")
            return

        cached_count = int(coverage.get("cached_tile_count", 0))
        total_tiles = int(coverage.get("total_tiles", 0))
        fraction = float(coverage.get("coverage_fraction", 0.0))
        area = float(coverage.get("covered_area_sq_deg", 0.0))
        required = list(coverage.get("field_required_tiles", []))
        field_cached = list(coverage.get("field_cached_tiles", []))
        field_missing = list(coverage.get("field_missing_tiles", []))

        tiles_text = f"{cached_count:,} / {total_tiles:,} ({fraction * 100.0:.2f}%)"
        if bool(coverage.get("bright_catalog_enabled", False)):
            tiles_text += (
                f" · Gaia {int(coverage.get('gaia_cached_tile_count', 0)):,}"
                f" · Hip/Tycho {int(coverage.get('bright_cached_tile_count', 0)):,}"
            )
        self.lbl_gaia_tiles.setText(tiles_text)
        self.lbl_gaia_area.setText(f"{area:,.1f} deg²")
        disk_text = self._gaia_format_bytes(coverage.get("cached_bytes", 0))
        newest = coverage.get("newest_mtime")
        if newest is not None:
            updated = _dt.datetime.fromtimestamp(float(newest)).strftime("%Y-%m-%d %H:%M")
            disk_text += f" · última tesela {updated}"
        self.lbl_gaia_disk.setText(disk_text)

        if required:
            field_fraction = len(field_cached) / len(required)
            self.lbl_gaia_field.setText(
                f"{len(field_cached)} / {len(required)} teselas ({field_fraction * 100.0:.0f}%)"
                + (f" · faltan {len(field_missing)}" if field_missing else "")
            )
            if field_missing:
                self.lbl_gaia_field_status.setText("Campo actual: cobertura incompleta")
                status_style = (
                    "background:#3a1515; border:1px solid #7a2a2a; color:#ffecec;"
                )
            else:
                self.lbl_gaia_field_status.setText("Campo actual: cubierto")
                status_style = (
                    "background:#16321a; border:1px solid #2f6b38; color:#e8ffe8;"
                )
        else:
            self.lbl_gaia_field.setText("Sin posición actual")
            self.lbl_gaia_field_status.setText("Campo actual: sin posición")
            status_style = "background:#24282d; border:1px solid #444b53; color:#e5e7eb;"
        self.lbl_gaia_field_status.setStyleSheet(
            f"QLabel {{ padding:6px 10px; border-radius:8px; {status_style} font-weight:600; }}"
        )

        center_az = coverage.get("center_az_deg")
        center_alt = coverage.get("center_alt_deg")
        radius_deg = coverage.get("field_radius_deg")
        source = coverage.get("field_source")
        if center_az is None or center_alt is None or radius_deg is None:
            self.lbl_gaia_center.setText("--")
        else:
            source_text = f" · {source}" if source else ""
            self.lbl_gaia_center.setText(
                f"Az {float(center_az):.2f}° · Alt {float(center_alt):+.2f}° · "
                f"r={float(radius_deg):.3f}°{source_text}"
            )

        projection_time = str(coverage.get("projection_time_utc", "--")).replace("T", " ")
        self.lbl_gaia_projection.setText(
            f"{projection_time} UTC · "
            f"{float(coverage.get('observer_lat_deg', 0.0)):+.4f}°, "
            f"{float(coverage.get('observer_lon_deg', 0.0)):+.4f}°"
        )
        catalog_text = (
            f"{coverage.get('table_name', '--')} · G≤{float(coverage.get('gmax', 0.0)):.1f}"
        )
        if bool(coverage.get("bright_catalog_enabled", False)):
            catalog_text += (
                f" + Hipparcos/Tycho-2 V≤{float(coverage.get('gmax', 0.0)):.1f}"
            )
        self.lbl_gaia_config.setText(
            f"{catalog_text} · NSIDE {int(coverage.get('nside', 0))} · "
            f"{coverage.get('order', '--')}"
        )
        self.lbl_gaia_cache_dir.setText(str(coverage.get("cache_dir", "--")))
        self.gaia_coverage_map.set_coverage(coverage)


class ObserverTabMixin:
    def _tab_observer(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Configuración del observador")
        form = QFormLayout()

        self.dd_obs_site = QComboBox()
        for site_name, site_data in OBSERVER_PRESETS:
            self.dd_obs_site.addItem(site_name, site_data)
        self.dd_obs_site.currentIndexChanged.connect(self._observer_site_changed)

        self.lbl_obs_site_coords = QLabel("--")

        self.ds_obs_focal_mm = QDoubleSpinBox()
        self.ds_obs_focal_mm.setRange(50.0, 10_000.0)
        self.ds_obs_focal_mm.setDecimals(1)
        self.ds_obs_focal_mm.setValue(float(self.cfg.platesolving.focal_m) * 1_000.0)
        self.ds_obs_focal_mm.setSuffix(" mm")
        self.ds_obs_focal_mm.valueChanged.connect(self._observer_refresh_effective_focal)

        self.ds_obs_pixel_um = QDoubleSpinBox()
        self.ds_obs_pixel_um.setRange(0.1, 20.0)
        self.ds_obs_pixel_um.setDecimals(1)
        self.ds_obs_pixel_um.setSingleStep(0.1)
        self.ds_obs_pixel_um.setValue(round(float(self.cfg.platesolving.pixel_size_m) * 1e6, 1))
        self.ds_obs_pixel_um.setSuffix(" µm")

        self.dd_obs_barlow = QComboBox()
        for factor in BARLOW_FACTORS:
            label = "x1 (sin barlow)" if factor == 1 else f"x{factor}"
            self.dd_obs_barlow.addItem(label, factor)
        self.dd_obs_barlow.currentIndexChanged.connect(self._observer_refresh_effective_focal)

        self.lbl_obs_effective_focal = QLabel("--")

        self.cb_obs_rot_prior = QCheckBox("Prior de rotación (plate solving)")
        self.cb_obs_rot_prior.setChecked(
            bool(getattr(self.cfg.platesolving, "rotation_prior_enable", True))
        )

        self.ds_obs_rot_tol = QDoubleSpinBox()
        self.ds_obs_rot_tol.setRange(1.0, 180.0)
        self.ds_obs_rot_tol.setDecimals(1)
        self.ds_obs_rot_tol.setValue(
            float(getattr(self.cfg.platesolving, "rotation_prior_tol_deg", 45.0))
        )
        self.ds_obs_rot_tol.setSuffix(" deg")

        self.btn_obs_apply = QPushButton("Apply")
        _set_option_tooltip(
            self.btn_obs_apply,
            "Aplica ubicación, escala óptica y prior de rotación al plate solving y al modelo de apuntado.",
        )
        self.btn_obs_apply.clicked.connect(self._observer_apply)

        _add_option_row(
            form,
            "Ubicación:",
            self.dd_obs_site,
            "Sitio del observador usado para convertir entre RA/Dec y Az/Alt.",
        )
        form.addRow("Lat/Lon/Alt:", self.lbl_obs_site_coords)
        _add_option_row(
            form,
            "Focal:",
            self.ds_obs_focal_mm,
            "Distancia focal base del telescopio, sin multiplicador Barlow.",
        )
        _add_option_row(
            form,
            "Tamaño píxel:",
            self.ds_obs_pixel_um,
            "Tamaño físico del píxel del sensor. Afecta la escala angular por píxel.",
        )
        _add_option_row(
            form,
            "Barlow:",
            self.dd_obs_barlow,
            "Multiplicador óptico aplicado a la focal base para calcular la focal efectiva.",
        )
        form.addRow("Focal efectiva:", self.lbl_obs_effective_focal)
        _set_option_tooltip(
            self.cb_obs_rot_prior,
            "Usa la orientación esperada de la cámara/montura como prior para acelerar y estabilizar plate solving.",
        )
        form.addRow(self.cb_obs_rot_prior)
        _add_option_row(
            form,
            "Tolerancia rotación:",
            self.ds_obs_rot_tol,
            "Margen angular permitido alrededor del prior de rotación durante plate solving.",
        )
        form.addRow(self.btn_obs_apply)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)

        self._observer_site_changed()
        self._observer_refresh_effective_focal()
        return widget

    def _observer_site_data(self: "AstroPanoptesWindow") -> dict[str, float]:
        data = self.dd_obs_site.currentData()
        if isinstance(data, dict):
            try:
                return {
                    "lat_deg": float(data["lat_deg"]),
                    "lon_deg": float(data["lon_deg"]),
                    "height_m": float(data["height_m"]),
                }
            except (KeyError, TypeError, ValueError):
                pass
        fallback = OBSERVER_PRESETS[0][1]
        return {
            "lat_deg": float(fallback["lat_deg"]),
            "lon_deg": float(fallback["lon_deg"]),
            "height_m": float(fallback["height_m"]),
        }

    def _observer_barlow_factor(self: "AstroPanoptesWindow") -> int:
        try:
            factor = int(self.dd_obs_barlow.currentData())
        except (TypeError, ValueError):
            factor = 1
        return max(1, min(5, factor))

    def _observer_site_changed(self: "AstroPanoptesWindow", *_args) -> None:
        site = self._observer_site_data()
        self.lbl_obs_site_coords.setText(
            f"{site['lat_deg']:.4f}°, {site['lon_deg']:.4f}°, {site['height_m']:.0f} m"
        )

    def _observer_refresh_effective_focal(self: "AstroPanoptesWindow", *_args) -> None:
        base_focal_mm = float(self.ds_obs_focal_mm.value())
        factor = self._observer_barlow_factor()
        effective_focal_mm = base_focal_mm * factor
        self.lbl_obs_effective_focal.setText(f"{effective_focal_mm:.1f} mm")

    def _observer_apply(self: "AstroPanoptesWindow") -> None:
        site = self._observer_site_data()
        base_focal_mm = float(self.ds_obs_focal_mm.value())
        barlow_factor = self._observer_barlow_factor()
        effective_focal_mm = base_focal_mm * barlow_factor
        effective_focal_m = effective_focal_mm / 1_000.0
        pixel_um = round(float(self.ds_obs_pixel_um.value()), 1)
        pixel_size_m = pixel_um * 1e-6
        rotation_prior_enable = bool(self.cb_obs_rot_prior.isChecked())
        rotation_prior_tol_deg = float(self.ds_obs_rot_tol.value())
        roll_offset_deg = float(self.runner.get_state().camera.roll_deg)

        self.runner.request_platesolving_params(
            focal_m=effective_focal_m,
            pixel_size_m=pixel_size_m,
            observer_lat_deg=float(site["lat_deg"]),
            observer_lon_deg=float(site["lon_deg"]),
            observer_height_m=float(site["height_m"]),
            rotation_prior_enable=rotation_prior_enable,
            rotation_prior_tol_deg=rotation_prior_tol_deg,
            rotation_prior_roll_offset_deg=roll_offset_deg,
        )
        self.cfg.platesolving.focal_m = effective_focal_m
        self.cfg.platesolving.pixel_size_m = pixel_size_m
        self.cfg.platesolving.rotation_prior_enable = rotation_prior_enable
        self.cfg.platesolving.rotation_prior_tol_deg = rotation_prior_tol_deg
        self.cfg.platesolving.rotation_prior_roll_offset_deg = roll_offset_deg

        self._log(
            "[observer] apply "
            f"site={self.dd_obs_site.currentText()} lat={site['lat_deg']:.4f} lon={site['lon_deg']:.4f} "
            f"alt={site['height_m']:.0f}m focal={base_focal_mm:.1f}mm barlow=x{barlow_factor} "
            f"effective={effective_focal_mm:.1f}mm pixel={pixel_um:.1f}um "
            f"rot_prior={int(rotation_prior_enable)} tol={rotation_prior_tol_deg:.1f}deg roll={roll_offset_deg:+.2f}deg"
        )


class CameraTabMixin:
    def _tab_camera(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Camera")
        form = QFormLayout()

        self.ds_exp_ms = QDoubleSpinBox()
        self.ds_exp_ms.setRange(0.1, 10_000.0)
        self.ds_exp_ms.setDecimals(1)
        self.ds_exp_ms.setValue(self.cfg.camera.exp_ms)
        self.ds_exp_ms.setSuffix(" ms")

        self.sb_gain = QSpinBox()
        self.sb_gain.setRange(0, 6000)
        self.sb_gain.setValue(self.cfg.camera.gain)

        self.sb_offset = QSpinBox()
        self.sb_offset.setRange(0, 500)
        self.sb_offset.setValue(self.cfg.camera.offset)

        self.btn_apply_cam = QPushButton("Apply")
        _set_option_tooltip(
            self.btn_apply_cam,
            "Aplica exposición, ganancia y offset a la cámara activa.",
        )
        self.btn_apply_cam.clicked.connect(self._camera_apply)
        self.btn_record_raw = QPushButton("Record 20s RAW (.npy)")
        _set_option_tooltip(
            self.btn_record_raw,
            "Graba 20 segundos de frames RAW en raw_output para diagnóstico o análisis offline.",
        )
        self.btn_record_raw.clicked.connect(self._camera_record_raw)

        _add_option_row(
            form,
            "Exposure:",
            self.ds_exp_ms,
            "Tiempo de exposición por frame. Exposiciones más largas capturan más señal y bajan el FPS máximo.",
        )
        _add_option_row(
            form,
            "Gain:",
            self.sb_gain,
            "Ganancia electrónica de la cámara. Más ganancia aumenta señal aparente y ruido.",
        )
        _add_option_row(
            form,
            "Offset:",
            self.sb_offset,
            "Nivel negro que evita recortar el ruido en cero. Para gain 360, la Mars-C requiere aproximadamente 350.",
        )
        form.addRow(self.btn_apply_cam)
        form.addRow(self.btn_record_raw)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget


class TrackingTabMixin:
    def _tab_tracking(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        self.btn_tr_start = QPushButton("Start")
        self.btn_tr_stop = QPushButton("Stop")
        self.btn_tr_reset = QPushButton("Reset keyframe")
        self.btn_tr_apply = QPushButton("Apply")
        _set_option_tooltip(self.btn_tr_start, "Activa tracking y comienza a enviar correcciones de velocidad a la montura.")
        _set_option_tooltip(self.btn_tr_stop, "Detiene tracking y deja la montura sin correcciones de tracking.")
        _set_option_tooltip(self.btn_tr_reset, "Reinicia la referencia visual usada para medir deriva desde el frame actual.")
        _set_option_tooltip(self.btn_tr_apply, "Aplica los parámetros visibles de tracking sin iniciar ni detener el tracking.")

        actions = QHBoxLayout()
        actions.addWidget(self.btn_tr_start)
        actions.addWidget(self.btn_tr_stop)
        actions.addWidget(self.btn_tr_reset)
        actions.addWidget(self.btn_tr_apply)
        actions.addStretch(1)

        self.btn_tr_start.clicked.connect(self._tracking_start)
        self.btn_tr_stop.clicked.connect(self._tracking_stop)
        self.btn_tr_reset.clicked.connect(self._tracking_reset)
        self.btn_tr_apply.clicked.connect(self._tracking_apply)

        self.ds_tr_resp_min = QDoubleSpinBox()
        self.ds_tr_resp_min.setRange(0.0, 1.0)
        self.ds_tr_resp_min.setDecimals(3)
        self.ds_tr_resp_min.setSingleStep(0.01)
        self.ds_tr_resp_min.setValue(float(self.cfg.tracking.resp_min))

        self.cb_tr_ff = QCheckBox("Feed-forward sideral")
        self.cb_tr_ff.setChecked(bool(getattr(self.cfg.tracking, "sidereal_ff_enabled", True)))

        self.ds_tr_ff_gain = QDoubleSpinBox()
        self.ds_tr_ff_gain.setRange(0.0, 5.0)
        self.ds_tr_ff_gain.setDecimals(3)
        self.ds_tr_ff_gain.setSingleStep(0.05)
        self.ds_tr_ff_gain.setValue(float(getattr(self.cfg.tracking, "sidereal_ff_gain", 1.0)))

        self.ds_tr_ff_dt = QDoubleSpinBox()
        self.ds_tr_ff_dt.setRange(0.01, 30.0)
        self.ds_tr_ff_dt.setDecimals(2)
        self.ds_tr_ff_dt.setSingleStep(0.1)
        self.ds_tr_ff_dt.setSuffix(" s")
        self.ds_tr_ff_dt.setValue(float(getattr(self.cfg.tracking, "sidereal_ff_dt_s", 1.0)))

        self.ds_tr_ff_cond = QDoubleSpinBox()
        self.ds_tr_ff_cond.setRange(1.0, 1_000_000.0)
        self.ds_tr_ff_cond.setDecimals(0)
        self.ds_tr_ff_cond.setSingleStep(100.0)
        self.ds_tr_ff_cond.setValue(float(getattr(self.cfg.tracking, "sidereal_ff_cond_max", 5_000.0)))

        self.ds_tr_ff_hold = QDoubleSpinBox()
        self.ds_tr_ff_hold.setRange(0.0, 120.0)
        self.ds_tr_ff_hold.setDecimals(1)
        self.ds_tr_ff_hold.setSingleStep(0.5)
        self.ds_tr_ff_hold.setSuffix(" s")
        self.ds_tr_ff_hold.setValue(float(getattr(self.cfg.tracking, "sidereal_ff_hold_s", 8.0)))

        self.ds_tr_ff_slew = QDoubleSpinBox()
        self.ds_tr_ff_slew.setRange(1.0, 10_000.0)
        self.ds_tr_ff_slew.setDecimals(1)
        self.ds_tr_ff_slew.setSingleStep(10.0)
        self.ds_tr_ff_slew.setSuffix(" steps/s²")
        self.ds_tr_ff_slew.setValue(float(getattr(self.cfg.tracking, "sidereal_ff_slew_per_s", 120.0)))

        self.sb_tr_sep_minarea = QSpinBox()
        self.sb_tr_sep_minarea.setRange(1, 500)
        self.sb_tr_sep_minarea.setValue(int(self.cfg.sep.minarea))

        self.ds_tr_sep_sigma = QDoubleSpinBox()
        self.ds_tr_sep_sigma.setRange(0.1, 20.0)
        self.ds_tr_sep_sigma.setDecimals(2)
        self.ds_tr_sep_sigma.setSingleStep(0.1)
        self.ds_tr_sep_sigma.setValue(float(self.cfg.sep.thresh_sigma))

        self.sb_tr_sep_max_sources = QSpinBox()
        self.sb_tr_sep_max_sources.setRange(1, 5000)
        self.sb_tr_sep_max_sources.setValue(int(self.cfg.platesolving.max_det))

        self.sb_tr_sep_min_sources = QSpinBox()
        self.sb_tr_sep_min_sources.setRange(1, 100)
        self.sb_tr_sep_min_sources.setValue(3)

        self.sb_tr_sep_bw = QSpinBox()
        self.sb_tr_sep_bw.setRange(4, 512)
        self.sb_tr_sep_bw.setValue(int(self.cfg.sep.bw))

        self.sb_tr_sep_bh = QSpinBox()
        self.sb_tr_sep_bh.setRange(4, 512)
        self.sb_tr_sep_bh.setValue(int(self.cfg.sep.bh))

        control_box = QGroupBox("Control")
        control_form = QFormLayout(control_box)
        _add_option_row(
            control_form,
            "resp_min:",
            self.ds_tr_resp_min,
            "Respuesta mínima para aceptar la medición de alineación. Más alto rechaza frames dudosos; más bajo tolera señal débil.",
        )
        _set_option_tooltip(
            self.cb_tr_ff,
            "Activa una corrección anticipada por movimiento sideral usando el modelo de apuntado actual.",
        )
        control_form.addRow(self.cb_tr_ff)
        _add_option_row(
            control_form,
            "FF gain:",
            self.ds_tr_ff_gain,
            "Multiplicador de la velocidad feed-forward. 1.0 usa la predicción completa; valores menores la suavizan.",
        )
        _add_option_row(
            control_form,
            "FF dt:",
            self.ds_tr_ff_dt,
            "Intervalo usado para estimar la deriva sideral futura desde el modelo de apuntado.",
        )
        _add_option_row(
            control_form,
            "FF cond max:",
            self.ds_tr_ff_cond,
            "Condición máxima permitida para la geometría del modelo. Si se supera, el feed-forward se considera poco confiable.",
        )
        _add_option_row(
            control_form,
            "FF hold:",
            self.ds_tr_ff_hold,
            "Tiempo durante el cual se conserva la última velocidad feed-forward válida si el modelo queda temporalmente sin geometría confiable.",
        )
        _add_option_row(
            control_form,
            "FF slew:",
            self.ds_tr_ff_slew,
            "Límite de cambio por segundo de la velocidad feed-forward para evitar saltos bruscos en la montura.",
        )

        sep_box = QGroupBox("Detección SEP")
        sep_form = QFormLayout(sep_box)
        _add_option_row(
            sep_form,
            "minarea:",
            self.sb_tr_sep_minarea,
            "Cantidad mínima de píxeles conectados sobre el umbral para aceptar una fuente.",
        )
        _add_option_row(
            sep_form,
            "thresh_sigma:",
            self.ds_tr_sep_sigma,
            "Umbral de detección en sigmas sobre el fondo local. Más alto detecta menos fuentes, pero más limpias.",
        )
        _add_option_row(
            sep_form,
            "max sources:",
            self.sb_tr_sep_max_sources,
            "Máximo de fuentes detectadas que se usan para medir deriva y emparejar movimientos.",
        )
        _add_option_row(
            sep_form,
            "min sources:",
            self.sb_tr_sep_min_sources,
            "Mínimo de fuentes requeridas para confiar en una medición de tracking.",
        )
        _add_option_row(
            sep_form,
            "bw:",
            self.sb_tr_sep_bw,
            "Ancho de la malla de fondo usada por SEP para estimar el fondo local.",
        )
        _add_option_row(
            sep_form,
            "bh:",
            self.sb_tr_sep_bh,
            "Alto de la malla de fondo usada por SEP para estimar el fondo local.",
        )

        columns = QHBoxLayout()
        columns.addWidget(control_box, stretch=1)
        columns.addWidget(sep_box, stretch=1)

        layout.addLayout(actions)
        layout.addLayout(columns)
        layout.addStretch(1)
        return widget


class StackingTabMixin:
    def _tab_stacking(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        self.btn_st_start = QPushButton("Start")
        self.btn_st_stop = QPushButton("Stop")
        self.btn_st_reset = QPushButton("Reset")
        self.btn_st_save = QPushButton("Save Stack")
        self.btn_st_apply = QPushButton("Apply")
        _set_option_tooltip(self.btn_st_start, "Inicia el apilado en vivo de los frames entrantes.")
        _set_option_tooltip(self.btn_st_stop, "Pausa el apilado en vivo sin borrar el stack acumulado.")
        _set_option_tooltip(self.btn_st_reset, "Borra el stack acumulado y reinicia las estadísticas de apilado.")
        _set_option_tooltip(self.btn_st_save, "Guarda el stack actual en stack_output como RAW numpy y PNG.")
        _set_option_tooltip(self.btn_st_apply, "Aplica los parámetros visibles de stacking y reconfigura el motor de apilado.")
        self.cb_st_color = QCheckBox("Stacking a color (RGB)")
        self.cb_st_color.setChecked(str(self.cfg.stacking.color_mode).lower() == "rgb")
        self.cb_st_color.toggled.connect(self._stacking_color_toggled)
        self.dd_st_drizzle = QComboBox()
        self.dd_st_drizzle.addItem("x1 (off)", 1.0)
        self.dd_st_drizzle.addItem("x2", 2.0)
        self.dd_st_drizzle.addItem("x3", 3.0)
        drizzle_cfg = float(getattr(self.cfg.stacking, "drizzle_scale", 1.0))
        if drizzle_cfg >= 2.5:
            drizzle_idx = 2
        elif drizzle_cfg >= 1.5:
            drizzle_idx = 1
        else:
            drizzle_idx = 0
        self.dd_st_drizzle.setCurrentIndex(drizzle_idx)
        self.dd_st_drizzle.currentIndexChanged.connect(self._stacking_drizzle_changed)

        self.dd_st_bayer = QComboBox()
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
            self.dd_st_bayer.addItem(pattern, pattern)
        bayer = str(getattr(self.cfg.stacking, "bayer_pattern", "RGGB")).upper()
        idx = self.dd_st_bayer.findText(bayer)
        self.dd_st_bayer.setCurrentIndex(max(0, idx))

        self.sb_st_batch = QSpinBox()
        self.sb_st_batch.setRange(1, 200)
        self.sb_st_batch.setValue(int(self.cfg.stacking.batch_size))

        self.sb_st_max_queue = QSpinBox()
        self.sb_st_max_queue.setRange(1, 1000)
        self.sb_st_max_queue.setValue(int(self.cfg.stacking.max_queue))

        self.sb_st_align_median = QSpinBox()
        self.sb_st_align_median.setRange(1, 31)
        self.sb_st_align_median.setSingleStep(2)
        self.sb_st_align_median.setValue(int(self.cfg.stacking.align_median_k))

        self.sb_st_smooth = QSpinBox()
        self.sb_st_smooth.setRange(1, 300)
        self.sb_st_smooth.setValue(int(self.cfg.stacking.smooth_k))

        self.sb_st_max_shift = QSpinBox()
        self.sb_st_max_shift.setRange(1, 500)
        self.sb_st_max_shift.setSuffix(" px")
        self.sb_st_max_shift.setValue(int(self.cfg.stacking.max_shift_px))

        self.cb_st_subpixel = QCheckBox("Subpixel alignment")
        self.cb_st_subpixel.setChecked(bool(self.cfg.stacking.use_subpixel))

        self.ds_st_preview_hz = QDoubleSpinBox()
        self.ds_st_preview_hz.setRange(0.1, 30.0)
        self.ds_st_preview_hz.setDecimals(1)
        self.ds_st_preview_hz.setSingleStep(0.5)
        self.ds_st_preview_hz.setSuffix(" Hz")
        self.ds_st_preview_hz.setValue(float(self.cfg.stacking.preview_hz))

        self.ds_st_preview_vmin = QDoubleSpinBox()
        self.ds_st_preview_vmin.setRange(0.0, 65_535.0)
        self.ds_st_preview_vmin.setDecimals(1)
        self.ds_st_preview_vmin.setSingleStep(1.0)
        self.ds_st_preview_vmin.setValue(float(self.cfg.stacking.preview_log_vmin))

        actions = QHBoxLayout()
        for button in [self.btn_st_start, self.btn_st_stop, self.btn_st_reset, self.btn_st_save, self.btn_st_apply]:
            actions.addWidget(button)
        actions.addStretch(1)

        self.btn_st_start.clicked.connect(self._stacking_start)
        self.btn_st_stop.clicked.connect(self._stacking_stop)
        self.btn_st_reset.clicked.connect(self._stacking_reset)
        self.btn_st_save.clicked.connect(self._stacking_save)
        self.btn_st_apply.clicked.connect(self._stacking_apply)

        stack_box = QGroupBox("Stack")
        stack_form = QFormLayout(stack_box)
        _add_option_row(
            stack_form,
            "Drizzle:",
            self.dd_st_drizzle,
            "Escala de salida del apilado. x1 conserva tamaño nativo; x2/x3 aumentan resolución a costa de memoria y CPU.",
        )
        _add_option_row(
            stack_form,
            "Bayer:",
            self.dd_st_bayer,
            "Patrón del mosaico Bayer del sensor. Debe coincidir con la cámara para que el color RGB salga correcto.",
        )
        _set_option_tooltip(
            self.cb_st_color,
            "Apila en RGB usando el patrón Bayer seleccionado. Si está apagado, el stack se mantiene monocromo.",
        )
        stack_form.addRow(self.cb_st_color)
        _add_option_row(
            stack_form,
            "Batch size:",
            self.sb_st_batch,
            "Cantidad de frames que procesa el worker por ciclo. Más alto puede rendir mejor, pero agrega latencia.",
        )
        _add_option_row(
            stack_form,
            "Max queue:",
            self.sb_st_max_queue,
            "Máximo de frames esperando en la cola. Más grande tolera ráfagas, pero puede acumular frames viejos.",
        )

        align_box = QGroupBox("Alineación y preview")
        align_form = QFormLayout(align_box)
        _set_option_tooltip(
            self.cb_st_subpixel,
            "Permite estimar desplazamientos fraccionales de píxel durante la alineación del stack.",
        )
        align_form.addRow(self.cb_st_subpixel)
        _add_option_row(
            align_form,
            "Median k:",
            self.sb_st_align_median,
            "Tamaño del filtro mediano previo a la alineación. Debe ser impar; ayuda a remover píxeles calientes y ruido impulsivo.",
        )
        _add_option_row(
            align_form,
            "Smooth k:",
            self.sb_st_smooth,
            "Suavizado de perfiles usado para estimar desplazamiento. Más alto estabiliza, pero responde menos a cambios finos.",
        )
        _add_option_row(
            align_form,
            "Max shift:",
            self.sb_st_max_shift,
            "Desplazamiento máximo aceptado entre frames. Si se supera, el frame puede rechazarse para evitar contaminar el stack.",
        )
        _add_option_row(
            align_form,
            "Preview Hz:",
            self.ds_st_preview_hz,
            "Frecuencia de actualización de la vista apilada. Más alta consume más CPU.",
        )
        _add_option_row(
            align_form,
            "Preview vmin:",
            self.ds_st_preview_vmin,
            "Piso de brillo usado para el estiramiento logarítmico del preview del stack.",
        )

        columns = QHBoxLayout()
        columns.addWidget(stack_box, stretch=1)
        columns.addWidget(align_box, stretch=1)

        layout.addLayout(actions)
        layout.addLayout(columns)
        layout.addStretch(1)
        return widget


class ObjectDetectionTabMixin:
    def _tab_od(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Object Detection")
        form = QFormLayout()

        self.btn_od_start = QPushButton("Start")
        self.btn_od_stop = QPushButton("Stop")
        _set_option_tooltip(self.btn_od_start, "Activa el overlay de detección SEP sobre la vista live.")
        _set_option_tooltip(self.btn_od_stop, "Desactiva el overlay de detección SEP sobre la vista live.")

        row = QHBoxLayout()
        row.addWidget(self.btn_od_start)
        row.addWidget(self.btn_od_stop)
        row.addStretch(1)

        self.btn_od_start.clicked.connect(self._od_start)
        self.btn_od_stop.clicked.connect(self._od_stop)

        self.sb_od_minarea = QSpinBox()
        self.sb_od_minarea.setRange(1, 500)
        self.sb_od_minarea.setValue(self.cfg.sep.minarea)

        self.ds_od_sigma = QDoubleSpinBox()
        self.ds_od_sigma.setRange(0.1, 20.0)
        self.ds_od_sigma.setValue(self.cfg.sep.thresh_sigma)

        self.sb_od_maxdet = QSpinBox()
        self.sb_od_maxdet.setRange(1, 5000)
        self.sb_od_maxdet.setValue(self.cfg.platesolving.max_det)

        self.sb_od_bw = QSpinBox()
        self.sb_od_bw.setRange(4, 512)
        self.sb_od_bw.setValue(self.cfg.sep.bw)

        self.sb_od_bh = QSpinBox()
        self.sb_od_bh.setRange(4, 512)
        self.sb_od_bh.setValue(self.cfg.sep.bh)

        form.addRow(row)
        _add_option_row(
            form,
            "minarea:",
            self.sb_od_minarea,
            "Cantidad mínima de píxeles conectados sobre el umbral para aceptar una detección.",
        )
        _add_option_row(
            form,
            "thresh_sigma:",
            self.ds_od_sigma,
            "Umbral de detección en sigmas sobre el fondo local. Más alto reduce falsos positivos.",
        )
        _add_option_row(
            form,
            "max_det:",
            self.sb_od_maxdet,
            "Cantidad máxima de detecciones que se muestran y procesan en el overlay.",
        )
        _add_option_row(
            form,
            "bw:",
            self.sb_od_bw,
            "Ancho de la malla de fondo usada por SEP para detección live.",
        )
        _add_option_row(
            form,
            "bh:",
            self.sb_od_bh,
            "Alto de la malla de fondo usada por SEP para detección live.",
        )

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)
        return widget


class GoToTabMixin:
    def _tab_goto(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("GoTo")
        form = QFormLayout()

        self.dd_goto_mode = QComboBox()
        self.dd_goto_mode.addItems(["name (SIMBAD)", "planet/moon", "radec", "altaz"])
        _set_option_tooltip(
            self.dd_goto_mode,
            "Tipo de objetivo para GoTo: nombre SIMBAD, planeta/luna, coordenadas RA/Dec o coordenadas Az/Alt.",
        )

        self.ed_goto_name = QLineEdit()
        self.ed_goto_name.setPlaceholderText("Object name (SIMBAD)")
        _set_option_tooltip(
            self.ed_goto_name,
            "Nombre astronómico que se resolverá con SIMBAD, por ejemplo M42, Vega o Saturn Nebula.",
        )

        self.dd_goto_planet = QComboBox()
        self.dd_goto_planet.addItems(
            ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
        )
        _set_option_tooltip(
            self.dd_goto_planet,
            "Objeto del sistema solar calculado para la hora y ubicación actuales.",
        )

        self.ds_ra = QDoubleSpinBox()
        self.ds_ra.setRange(0.0, 360.0)
        self.ds_ra.setDecimals(6)
        _set_option_tooltip(self.ds_ra, "Ascensión recta del objetivo en grados ICRS/J2000.")

        self.ds_dec = QDoubleSpinBox()
        self.ds_dec.setRange(-90.0, 90.0)
        self.ds_dec.setDecimals(6)
        _set_option_tooltip(self.ds_dec, "Declinación del objetivo en grados ICRS/J2000.")

        self.ed_ra_hms = QLineEdit()
        self.ed_ra_hms.setPlaceholderText("RA HH:MM:SS(.s)")
        _set_option_tooltip(self.ed_ra_hms, "Ascensión recta en formato horas:minutos:segundos.")

        self.ed_dec_dms = QLineEdit()
        self.ed_dec_dms.setPlaceholderText("Dec ±DD:MM:SS")
        _set_option_tooltip(self.ed_dec_dms, "Declinación en formato grados:minutos:segundos con signo.")

        self.dd_radec_fmt = QComboBox()
        self.dd_radec_fmt.addItems(["deg", "HMS/DMS"])
        _set_option_tooltip(self.dd_radec_fmt, "Formato de entrada para coordenadas RA/Dec.")

        self.ds_az = QDoubleSpinBox()
        self.ds_az.setRange(0.0, 360.0)
        self.ds_az.setDecimals(6)
        _set_option_tooltip(self.ds_az, "Azimut local del objetivo, en grados desde el norte hacia el este.")

        self.ds_alt = QDoubleSpinBox()
        self.ds_alt.setRange(0.0, 90.0)
        self.ds_alt.setDecimals(6)
        _set_option_tooltip(self.ds_alt, "Altitud local del objetivo sobre el horizonte.")

        self.tgt_frame = QFrame()
        _set_option_tooltip(self.tgt_frame, "Campos del objetivo según el modo de GoTo seleccionado.")
        self.tgt_v = QVBoxLayout(self.tgt_frame)
        self.tgt_v.setContentsMargins(0, 0, 0, 0)
        self.tgt_v.setSpacing(6)

        self.radec_deg_frame = QFrame()
        row_radec_deg = QHBoxLayout(self.radec_deg_frame)
        row_radec_deg.setContentsMargins(0, 0, 0, 0)
        row_radec_deg.addWidget(QLabel("RA°"))
        row_radec_deg.addWidget(self.ds_ra)
        row_radec_deg.addSpacing(10)
        row_radec_deg.addWidget(QLabel("Dec°"))
        row_radec_deg.addWidget(self.ds_dec)
        row_radec_deg.addStretch(1)

        self.radec_hms_frame = QFrame()
        row_radec_hms = QHBoxLayout(self.radec_hms_frame)
        row_radec_hms.setContentsMargins(0, 0, 0, 0)
        row_radec_hms.addWidget(QLabel("RA"))
        row_radec_hms.addWidget(self.ed_ra_hms)
        row_radec_hms.addSpacing(10)
        row_radec_hms.addWidget(QLabel("Dec"))
        row_radec_hms.addWidget(self.ed_dec_dms)
        row_radec_hms.addStretch(1)

        self.altaz_frame = QFrame()
        row_altaz = QHBoxLayout(self.altaz_frame)
        row_altaz.setContentsMargins(0, 0, 0, 0)
        row_altaz.addWidget(QLabel("Az°"))
        row_altaz.addWidget(self.ds_az)
        row_altaz.addSpacing(10)
        row_altaz.addWidget(QLabel("Alt°"))
        row_altaz.addWidget(self.ds_alt)
        row_altaz.addStretch(1)

        self.sb_goto_ps_nseeds = QSpinBox()
        self.sb_goto_ps_nseeds.setRange(0, 10)
        self.sb_goto_ps_nseeds.setValue(self.cfg.platesolving.N_seed)
        _set_option_tooltip(
            self.sb_goto_ps_nseeds,
            "Cantidad de estrellas semilla usadas por Plate Solving. Más semillas puede mejorar robustez y costo.",
        )

        self.sb_goto_ps_mininl = QSpinBox()
        self.sb_goto_ps_mininl.setRange(1, 100)
        self.sb_goto_ps_mininl.setValue(self.cfg.platesolving.min_inliers)
        _set_option_tooltip(
            self.sb_goto_ps_mininl,
            "Mínimo de coincidencias requeridas para aceptar una solución de plate solving.",
        )

        self.ds_goto_ps_radius = QDoubleSpinBox()
        self.ds_goto_ps_radius.setRange(0.1, 30.0)
        self.ds_goto_ps_radius.setDecimals(2)
        self.ds_goto_ps_radius.setValue(
            float(self.cfg.platesolving.search_radius_deg or 1.0)
        )
        self.ds_goto_ps_radius.setSuffix(" deg")
        _set_option_tooltip(
            self.ds_goto_ps_radius,
            "Radio alrededor del Az/Alt aproximado donde Plate Solving buscará la solución.",
        )

        self.ds_goto_ps_gmax = QDoubleSpinBox()
        self.ds_goto_ps_gmax.setRange(6.0, 20.0)
        self.ds_goto_ps_gmax.setDecimals(2)
        self.ds_goto_ps_gmax.setValue(float(self.cfg.platesolving.gmax))
        _set_option_tooltip(
            self.ds_goto_ps_gmax,
            "Magnitud límite del catálogo usado por plate solving. Mayor valor incluye estrellas más débiles.",
        )

        self.ds_goto_ps_az = QDoubleSpinBox()
        self.ds_goto_ps_az.setRange(0.0, 360.0)
        self.ds_goto_ps_az.setDecimals(6)
        _set_option_tooltip(self.ds_goto_ps_az, "Azimut aproximado del centro del campo que se resolverá.")

        self.ds_goto_ps_alt = QDoubleSpinBox()
        self.ds_goto_ps_alt.setRange(-10.0, 90.0)
        self.ds_goto_ps_alt.setDecimals(6)
        _set_option_tooltip(self.ds_goto_ps_alt, "Altitud aproximada del centro del campo que se resolverá.")

        self.dd_platesolve_mode = QComboBox()
        self.dd_platesolve_mode.addItem("Deriva", "drift")
        self.dd_platesolve_mode.addItem("Alt/Az (manual)", "manual_altaz")
        self.dd_platesolve_mode.addItem("Alt/Az (registrado)", "current_altaz")
        _set_option_tooltip(
            self.dd_platesolve_mode,
            "Origen de la posición aproximada: deriva medida, coordenadas Alt/Az ingresadas manualmente o el Alt/Az registrado por el modelo.",
        )

        self.platesolve_target_frame = QFrame()
        row_manual = QHBoxLayout(self.platesolve_target_frame)
        row_manual.setContentsMargins(0, 0, 0, 0)
        row_manual.addWidget(QLabel("Az°"))
        row_manual.addWidget(self.ds_goto_ps_az)
        row_manual.addSpacing(10)
        row_manual.addWidget(QLabel("Alt°"))
        row_manual.addWidget(self.ds_goto_ps_alt)
        row_manual.addStretch(1)

        rowfb = QHBoxLayout()
        rowfb.addWidget(QLabel("Plate Solving radius:"))
        rowfb.addWidget(self.ds_goto_ps_radius)
        rowfb.addSpacing(8)
        rowfb.addWidget(QLabel("gmax:"))
        rowfb.addWidget(self.ds_goto_ps_gmax)
        rowfb.addStretch(1)

        rowps = QHBoxLayout()
        rowps.addWidget(QLabel("N seeds:"))
        rowps.addWidget(self.sb_goto_ps_nseeds)
        rowps.addSpacing(12)
        rowps.addWidget(QLabel("Min inliers:"))
        rowps.addWidget(self.sb_goto_ps_mininl)
        rowps.addStretch(1)

        self.btn_goto = QPushButton("GoTo")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_platesolve = QPushButton("Plate Solving")
        self.btn_roll = QPushButton("Estimar Roll")
        self.btn_fit_model = QPushButton("Fit GoTo Model")
        self.btn_fit_model.setEnabled(False)
        self.btn_list_samples = QPushButton("Listar Muestras")
        self.btn_prune_outliers = QPushButton("Eliminar Outliers")
        self.btn_restore_last_log = QPushButton("Cargar Último Registro")
        self.btn_reset_goto = QPushButton("Reset")
        self.btn_home = QPushButton("Home")
        self.cb_expected_stars = QCheckBox("Estrellas esperadas según modelo")
        _set_option_tooltip(self.btn_goto, "Mueve la montura hacia el objetivo usando el modelo GoTo actual.")
        _set_option_tooltip(self.btn_cancel, "Cancela la operación GoTo en curso.")
        _set_option_tooltip(self.btn_platesolve, "Resuelve el cuadro vivo y agrega la muestra sólo si pasa automáticamente las validaciones de match, RMS, movimiento y roll; no cambia parámetros de Cámara.")
        _set_option_tooltip(self.btn_roll, "Estima y aplica la orientación del eje +Az en la imagen sin cambiar exposición ni ganancia.")
        _set_option_tooltip(self.btn_fit_model, "Ajusta el modelo GoTo usando las muestras manuales registradas.")
        _set_option_tooltip(self.btn_list_samples, "Muestra en el log las muestras manuales disponibles para el ajuste.")
        _set_option_tooltip(self.btn_prune_outliers, "Elimina muestras que degradan el ajuste del modelo GoTo.")
        _set_option_tooltip(self.btn_restore_last_log, "Carga el último respaldo CSV de muestras manuales del modelo GoTo.")
        _set_option_tooltip(self.btn_reset_goto, "Borra sincronización, muestras manuales y estado del modelo GoTo.")
        _set_option_tooltip(self.btn_home, "Mueve la montura a una posición segura predeterminada.")
        _set_option_tooltip(
            self.cb_expected_stars,
            "Muestra en el preview estrellas proyectadas según el modelo GoTo ajustado.",
        )
        self.cb_expected_stars.setChecked(False)
        self.cb_expected_stars.setEnabled(False)
        self.ds_expected_stars_mag = QDoubleSpinBox()
        self.ds_expected_stars_mag.setRange(-2.0, float(self.cfg.platesolving.gmax))
        self.ds_expected_stars_mag.setDecimals(1)
        self.ds_expected_stars_mag.setValue(
            float(self.cfg.preview.expected_stars_mag_limit)
        )
        self.ds_expected_stars_mag.setPrefix("mag≤")
        _set_option_tooltip(
            self.ds_expected_stars_mag,
            "Magnitud máxima de estrellas esperadas que se dibujan en el overlay del modelo.",
        )
        self.sb_expected_stars_max = QSpinBox()
        self.sb_expected_stars_max.setRange(1, 5000)
        self.sb_expected_stars_max.setValue(int(self.cfg.preview.expected_stars_max))
        self.sb_expected_stars_max.setPrefix("máx ")
        _set_option_tooltip(
            self.sb_expected_stars_max,
            "Cantidad máxima de estrellas esperadas que se proyectan en el preview.",
        )
        self.lbl_expected_stars = QLabel("Requiere Fit GoTo Model")

        rowb_top = QHBoxLayout()
        for button in [
            self.btn_goto,
            self.btn_cancel,
            self.btn_platesolve,
            self.btn_roll,
            self.btn_home,
        ]:
            rowb_top.addWidget(button)
        rowb_top.addStretch(1)

        rowb_bottom = QHBoxLayout()
        for button in [
            self.btn_fit_model,
            self.btn_list_samples,
            self.btn_prune_outliers,
            self.btn_restore_last_log,
            self.btn_reset_goto,
        ]:
            rowb_bottom.addWidget(button)
        rowb_bottom.addStretch(1)

        rowb = QVBoxLayout()
        rowb.addLayout(rowb_top)
        rowb.addLayout(rowb_bottom)

        self.btn_goto.clicked.connect(self._goto_start)
        self.btn_cancel.clicked.connect(self._goto_cancel)
        self.btn_platesolve.clicked.connect(self._goto_platesolve)
        self.btn_roll.clicked.connect(self._goto_estimate_roll)
        self.btn_fit_model.clicked.connect(self._goto_fit_model)
        self.btn_list_samples.clicked.connect(self._goto_list_samples)
        self.btn_prune_outliers.clicked.connect(self._goto_prune_outliers)
        self.btn_restore_last_log.clicked.connect(self._goto_restore_last_log)
        self.btn_reset_goto.clicked.connect(self._goto_reset)
        self.btn_home.clicked.connect(self._home)
        self.cb_expected_stars.toggled.connect(self._expected_stars_params_changed)
        self.ds_expected_stars_mag.valueChanged.connect(
            self._expected_stars_params_changed
        )
        self.sb_expected_stars_max.valueChanged.connect(
            self._expected_stars_params_changed
        )
        self.dd_platesolve_mode.currentIndexChanged.connect(
            self._platesolve_mode_switch
        )

        self.lbl_goto_samples = QLabel("0")

        _add_option_row(
            form,
            "mode:",
            self.dd_goto_mode,
            "Tipo de objetivo y formato de entrada para la orden GoTo.",
        )
        form.addRow(_option_label("target:", "Campos del objetivo según el modo seleccionado."), self.tgt_frame)
        form.addRow(rowfb)
        _add_option_row(
            form,
            "Plate Solving modo:",
            self.dd_platesolve_mode,
            "Selecciona deriva, Alt/Az manual o el Alt/Az registrado como punto de partida para Plate Solving.",
        )
        form.addRow("Plate Solving centro:", self.platesolve_target_frame)
        form.addRow(rowps)
        form.addRow(rowb)
        form.addRow("manual samples:", self.lbl_goto_samples)
        expected_row = QHBoxLayout()
        expected_row.addWidget(self.cb_expected_stars)
        expected_row.addWidget(self.ds_expected_stars_mag)
        expected_row.addWidget(self.sb_expected_stars_max)
        expected_row.addStretch(1)
        form.addRow("Overlay modelo:", expected_row)
        form.addRow("Estado overlay:", self.lbl_expected_stars)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)

        self.dd_goto_mode.currentIndexChanged.connect(self._goto_mode_switch)
        self.dd_radec_fmt.currentIndexChanged.connect(self._goto_mode_switch)
        self._goto_mode_switch()
        self._platesolve_mode_switch()
        return widget

    def _platesolve_mode_value(self: "AstroPanoptesWindow") -> str:
        return str(self.dd_platesolve_mode.currentData() or "drift")

    def _platesolve_mode_switch(self: "AstroPanoptesWindow") -> None:
        manual = self._platesolve_mode_value() == "manual_altaz"
        self.platesolve_target_frame.setVisible(manual)
        self.ds_goto_ps_az.setEnabled(manual)
        self.ds_goto_ps_alt.setEnabled(manual)

    def _expected_stars_params_changed(self: "AstroPanoptesWindow", *_args) -> None:
        self.runner.request_expected_stars_params(
            enabled=bool(self.cb_expected_stars.isChecked()),
            mag_limit=float(self.ds_expected_stars_mag.value()),
            max_stars=int(self.sb_expected_stars_max.value()),
        )

    def _update_expected_stars_controls(self: "AstroPanoptesWindow", state) -> None:
        ready = bool(
            int(getattr(state.goto, "model_fit_samples", 0)) > 0
            and bool(getattr(state.goto, "synced", False))
        )
        self.cb_expected_stars.setEnabled(ready)
        self.ds_expected_stars_mag.setEnabled(ready)
        self.sb_expected_stars_max.setEnabled(ready)

        if not ready:
            self.lbl_expected_stars.setText("Requiere Fit GoTo Model sincronizado")
            return
        reason = getattr(state.goto, "expected_stars_overlay_reason", None)
        if reason:
            self.lbl_expected_stars.setText(str(reason))
            return
        if bool(getattr(state.goto, "expected_stars_overlay_enabled", False)):
            count = int(getattr(state.goto, "expected_stars_overlay_count", 0))
            source = str(getattr(state.goto, "expected_stars_overlay_source", "") or "")
            suffix = f" · {source}" if source else ""
            self.lbl_expected_stars.setText(f"{count} estrellas proyectadas{suffix}")
        else:
            self.lbl_expected_stars.setText("Desactivado")

    def _goto_mode_switch(self: "AstroPanoptesWindow") -> None:
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
                self.tgt_v.addWidget(self.radec_deg_frame)
            else:
                self.tgt_v.addWidget(self.radec_hms_frame)
        else:
            self.tgt_v.addWidget(self.altaz_frame)

class ModulesTabsMixin(
    ObserverTabMixin,
    CameraTabMixin,
    TrackingTabMixin,
    StackingTabMixin,
    ObjectDetectionTabMixin,
    GoToTabMixin,
    GaiaTabMixin,
):
    def _build_modules_tabs(self: "AstroPanoptesWindow") -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._tab_observer(), "Observador")
        tabs.addTab(self._tab_camera(), "Camera")
        tabs.addTab(self._tab_tracking(), "Tracking")
        tabs.addTab(self._tab_stacking(), "Stacking")
        tabs.addTab(self._tab_od(), "Object Detection")
        tabs.addTab(self._tab_goto(), "GoTo")
        tabs.setTabToolTip(0, "Ubicación, escala óptica y prior de rotación para plate solving.")
        tabs.setTabToolTip(1, "Exposición, ganancia y captura RAW de diagnóstico.")
        tabs.setTabToolTip(2, "Control de tracking, feed-forward sideral y detección SEP para deriva.")
        tabs.setTabToolTip(3, "Live stacking, color, drizzle, alineación y preview del stack.")
        tabs.setTabToolTip(4, "Overlay de detección SEP sobre la vista live.")
        tabs.setTabToolTip(5, "Plate Solving y toma manual de muestras, ajuste del modelo GoTo y estrellas esperadas.")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(tabs)

        self.modules_tabs = tabs
        return wrap
