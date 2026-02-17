from __future__ import annotations

from typing import TYPE_CHECKING

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
    ("Algarrobo", {"lat_deg": -33.3667, "lon_deg": -71.6667, "height_m": 28.0}),
    ("Estación Central (Santiago)", {"lat_deg": -33.4569, "lon_deg": -70.6990, "height_m": 520.0}),
)
BARLOW_FACTORS = (1, 2, 3, 4, 5)
STACKING_DRIZZLE_SCALES = (1.0, 2.0, 3.0)

if TYPE_CHECKING:
    from ui.pyqt6_app import AstroPanoptesWindow


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
        self.btn_obs_apply.clicked.connect(self._observer_apply)

        form.addRow("Ubicación:", self.dd_obs_site)
        form.addRow("Lat/Lon/Alt:", self.lbl_obs_site_coords)
        form.addRow("Focal:", self.ds_obs_focal_mm)
        form.addRow("Tamaño píxel:", self.ds_obs_pixel_um)
        form.addRow("Barlow:", self.dd_obs_barlow)
        form.addRow("Focal efectiva:", self.lbl_obs_effective_focal)
        form.addRow(self.cb_obs_rot_prior)
        form.addRow("Tolerancia rotación:", self.ds_obs_rot_tol)
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

        self.btn_apply_cam = QPushButton("Apply")
        self.btn_apply_cam.clicked.connect(self._camera_apply)
        self.btn_record_raw = QPushButton("Record 20s RAW (.npy)")
        self.btn_record_raw.clicked.connect(self._camera_record_raw)

        form.addRow("Exposure:", self.ds_exp_ms)
        form.addRow("Gain:", self.sb_gain)
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

        self.btn_tr_start.clicked.connect(self._tracking_start)
        self.btn_tr_stop.clicked.connect(self._tracking_stop)
        self.btn_tr_reset.clicked.connect(self._tracking_reset)

        form.addRow(row)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget


class StackingTabMixin:
    def _tab_stacking(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        box = QGroupBox("Stacking")
        form = QFormLayout()

        self.btn_st_start = QPushButton("Start")
        self.btn_st_stop = QPushButton("Stop")
        self.btn_st_reset = QPushButton("Reset")
        self.btn_st_save = QPushButton("Save Stack")
        self.cb_st_color = QCheckBox("Stacking a color (RGB, Bayer RGGB)")
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

        row = QHBoxLayout()
        for button in [self.btn_st_start, self.btn_st_stop, self.btn_st_reset, self.btn_st_save]:
            row.addWidget(button)
        row.addStretch(1)

        self.btn_st_start.clicked.connect(self._stacking_start)
        self.btn_st_stop.clicked.connect(self._stacking_stop)
        self.btn_st_reset.clicked.connect(self._stacking_reset)
        self.btn_st_save.clicked.connect(self._stacking_save)

        form.addRow(row)
        form.addRow("Drizzle:", self.dd_st_drizzle)
        form.addRow(self.cb_st_color)
        box.setLayout(form)

        layout.addWidget(box)
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
        form.addRow("minarea:", self.sb_od_minarea)
        form.addRow("thresh_sigma:", self.ds_od_sigma)
        form.addRow("max_det:", self.sb_od_maxdet)
        form.addRow("bw:", self.sb_od_bw)
        form.addRow("bh:", self.sb_od_bh)

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

        self.cb_fb = QCheckBox("Platesolve feedback")
        self.sb_stages = QSpinBox()
        self.sb_stages.setRange(0, 20)
        self.sb_stages.setValue(self.cfg.goto.stages)
        self.sb_stages.setEnabled(self.cfg.goto.platesolving_feedback)
        self.cb_fb.setChecked(self.cfg.goto.platesolving_feedback)
        self.cb_fb.toggled.connect(self.sb_stages.setEnabled)

        self.sb_goto_ps_nseeds = QSpinBox()
        self.sb_goto_ps_nseeds.setRange(0, 10)
        self.sb_goto_ps_nseeds.setValue(self.cfg.platesolving.N_seed)

        self.sb_goto_ps_mininl = QSpinBox()
        self.sb_goto_ps_mininl.setRange(1, 100)
        self.sb_goto_ps_mininl.setValue(self.cfg.platesolving.min_inliers)

        self.ds_autocal_ps_radius = QDoubleSpinBox()
        self.ds_autocal_ps_radius.setRange(0.1, 30.0)
        self.ds_autocal_ps_radius.setDecimals(2)
        self.ds_autocal_ps_radius.setValue(
            float(self.cfg.platesolving.search_radius_deg or 1.0)
        )
        self.ds_autocal_ps_radius.setSuffix(" deg")

        self.ds_autocal_ps_gmax = QDoubleSpinBox()
        self.ds_autocal_ps_gmax.setRange(6.0, 20.0)
        self.ds_autocal_ps_gmax.setDecimals(2)
        self.ds_autocal_ps_gmax.setValue(float(self.cfg.platesolving.gmax))

        self.dd_autocal_ps_mode = QComboBox()
        self.dd_autocal_ps_mode.addItems(
            ["deriva (actual)", "alt/az manual", "alt/az actual (registrado)"]
        )

        self.ds_autocal_ps_manual_az = QDoubleSpinBox()
        self.ds_autocal_ps_manual_az.setRange(0.0, 360.0)
        self.ds_autocal_ps_manual_az.setDecimals(6)

        self.ds_autocal_ps_manual_alt = QDoubleSpinBox()
        self.ds_autocal_ps_manual_alt.setRange(0.0, 90.0)
        self.ds_autocal_ps_manual_alt.setDecimals(6)

        self.autocal_manual_frame = QFrame()
        row_manual = QHBoxLayout(self.autocal_manual_frame)
        row_manual.setContentsMargins(0, 0, 0, 0)
        row_manual.addWidget(QLabel("Az°"))
        row_manual.addWidget(self.ds_autocal_ps_manual_az)
        row_manual.addSpacing(10)
        row_manual.addWidget(QLabel("Alt°"))
        row_manual.addWidget(self.ds_autocal_ps_manual_alt)
        row_manual.addStretch(1)

        rowfb = QHBoxLayout()
        rowfb.addWidget(self.cb_fb)
        rowfb.addSpacing(10)
        rowfb.addWidget(QLabel("Stages:"))
        rowfb.addWidget(self.sb_stages)
        rowfb.addSpacing(12)
        rowfb.addWidget(QLabel("AutoCal radius:"))
        rowfb.addWidget(self.ds_autocal_ps_radius)
        rowfb.addSpacing(8)
        rowfb.addWidget(QLabel("AutoCal gmax:"))
        rowfb.addWidget(self.ds_autocal_ps_gmax)
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
        self.btn_autocal = QPushButton("Platesolving")
        self.btn_roll = QPushButton("Estimar Roll")
        self.btn_fit_model = QPushButton("Fit GoTo Model")
        self.btn_list_samples = QPushButton("Listar Muestras")
        self.btn_prune_outliers = QPushButton("Eliminar Outliers")
        self.btn_restore_last_log = QPushButton("Cargar Último Registro")
        self.btn_reset_goto = QPushButton("Reset")
        self.btn_home = QPushButton("Home")

        rowb_top = QHBoxLayout()
        for button in [
            self.btn_goto,
            self.btn_cancel,
            self.btn_autocal,
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
        self.btn_autocal.clicked.connect(self._autocalibrate)
        self.btn_roll.clicked.connect(self._goto_estimate_roll)
        self.btn_fit_model.clicked.connect(self._goto_fit_model)
        self.btn_list_samples.clicked.connect(self._goto_list_samples)
        self.btn_prune_outliers.clicked.connect(self._goto_prune_outliers)
        self.btn_restore_last_log.clicked.connect(self._goto_restore_last_log)
        self.btn_reset_goto.clicked.connect(self._goto_reset)
        self.btn_home.clicked.connect(self._home)

        self.lbl_goto_samples = QLabel("0")

        form.addRow("mode:", self.dd_goto_mode)
        form.addRow("target:", self.tgt_frame)
        form.addRow(rowfb)
        form.addRow("AutoCal PS mode:", self.dd_autocal_ps_mode)
        form.addRow("AutoCal manual:", self.autocal_manual_frame)
        form.addRow(rowps)
        form.addRow(rowb)
        form.addRow("manual samples:", self.lbl_goto_samples)

        box.setLayout(form)
        layout.addWidget(box)
        layout.addStretch(1)

        self.dd_goto_mode.currentIndexChanged.connect(self._goto_mode_switch)
        self.dd_radec_fmt.currentIndexChanged.connect(self._goto_mode_switch)
        self.dd_autocal_ps_mode.currentIndexChanged.connect(self._autocal_ps_mode_switch)
        self._goto_mode_switch()
        self._autocal_ps_mode_switch()
        return widget

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

    def _autocal_ps_mode_value(self: "AstroPanoptesWindow") -> str:
        mode = self.dd_autocal_ps_mode.currentText()
        if mode.startswith("alt/az manual"):
            return "manual_altaz"
        if mode.startswith("alt/az actual"):
            return "current_altaz"
        return "drift"

    def _autocal_ps_mode_switch(self: "AstroPanoptesWindow") -> None:
        manual = self._autocal_ps_mode_value() == "manual_altaz"
        self.autocal_manual_frame.setVisible(manual)
        self.ds_autocal_ps_manual_az.setEnabled(manual)
        self.ds_autocal_ps_manual_alt.setEnabled(manual)


class ModulesTabsMixin(
    ObserverTabMixin,
    CameraTabMixin,
    TrackingTabMixin,
    StackingTabMixin,
    ObjectDetectionTabMixin,
    GoToTabMixin,
):
    def _build_modules_tabs(self: "AstroPanoptesWindow") -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._tab_observer(), "Observador")
        tabs.addTab(self._tab_camera(), "Camera")
        tabs.addTab(self._tab_tracking(), "Tracking")
        tabs.addTab(self._tab_stacking(), "Stacking")
        tabs.addTab(self._tab_od(), "Object Detection")
        tabs.addTab(self._tab_goto(), "GoTo")

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(tabs)

        self.modules_tabs = tabs
        return wrap
