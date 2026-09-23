from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
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
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


OBSERVER_PRESETS = (
    ("Estación Central (Santiago)", {"lat_deg": -33.4569, "lon_deg": -70.6990, "height_m": 520.0}),
    ("San Carlos", {"lat_deg": -36.4248, "lon_deg": -71.9580, "height_m": 161.0}),
    ("Algarrobo", {"lat_deg": -33.3667, "lon_deg": -71.6667, "height_m": 28.0}),
)
# Los barlows que existen de verdad en este equipo. Ofrecer factores que no se
# tienen solo invita a dejar la escala mal puesta, y una escala equivocada hace
# que el plate solving busque un campo que no es -- sin decir por que falla.
BARLOW_FACTORS = (1, 2, 5)
# Cada barlow cambia el tiro optico, asi que cada uno tiene su posicion de foco:
# son los presets que vale la pena tener a un clic.
FOCUS_PRESET_NAMES = ("x1", "x2", "x5")
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
        return factor if factor in BARLOW_FACTORS else 1

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

        self.ds_gamma = QDoubleSpinBox()
        self.ds_gamma.setRange(0.1, 5.0)
        self.ds_gamma.setDecimals(2)
        self.ds_gamma.setSingleStep(0.05)
        self.ds_gamma.setValue(float(getattr(self.cfg.camera, "gamma", 1.0)))

        self.btn_apply_cam = QPushButton("Apply")
        _set_option_tooltip(
            self.btn_apply_cam,
            "Aplica exposición, ganancia, offset y gamma a la cámara activa.",
        )
        self.btn_apply_cam.clicked.connect(self._camera_apply)
        self.btn_record_raw = QPushButton("Start recording")
        _set_option_tooltip(
            self.btn_record_raw,
            "Comienza a guardar frames RAW en raw_output. Pulsa Stop recording cuando quieras terminar.",
        )
        self.btn_record_raw.clicked.connect(self._camera_record_raw)
        self.btn_stop_record_raw = QPushButton("Stop recording")
        self.btn_stop_record_raw.setEnabled(False)
        _set_option_tooltip(
            self.btn_stop_record_raw,
            "Detiene la grabación y guarda inmediatamente los frames capturados.",
        )
        self.btn_stop_record_raw.clicked.connect(self._camera_stop_record_raw)

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
        _add_option_row(
            form,
            "Gamma:",
            self.ds_gamma,
            "Curva de tono aplicada solo al preview/visor (no altera el RAW guardado). >1 aclara tonos medios, <1 los oscurece.",
        )
        form.addRow(self.btn_apply_cam)
        record_actions = QHBoxLayout()
        record_actions.addWidget(self.btn_record_raw)
        record_actions.addWidget(self.btn_stop_record_raw)
        form.addRow(record_actions)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget


class FocuserTabMixin:
    def _tab_focuser(self: "AstroPanoptesWindow") -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        cfg = self.cfg.focuser

        # ---- Manual ----
        manual_box = QGroupBox("Enfoque manual")
        manual_form = QFormLayout()

        self.sb_focus_step = QSpinBox()
        self.sb_focus_step.setRange(1, 100_000)
        self.sb_focus_step.setValue(int(cfg.step_size))

        self.sb_focus_delay = QSpinBox()
        self.sb_focus_delay.setRange(10, 200_000)
        self.sb_focus_delay.setValue(int(cfg.delay_us))
        self.sb_focus_delay.setSuffix(" µs")

        self.dd_focus_profile = QComboBox()
        self.dd_focus_profile.addItem("Suave (curva S)", "smooth")
        self.dd_focus_profile.addItem("Directo (sin rampa)", "direct")
        index = self.dd_focus_profile.findData(str(cfg.profile))
        if index >= 0:
            self.dd_focus_profile.setCurrentIndex(index)

        self.cb_focus_invert = QCheckBox("Invertir sentido")
        self.cb_focus_invert.setChecked(bool(cfg.invert))

        self.sb_focus_backlash = QSpinBox()
        self.sb_focus_backlash.setRange(0, 20_000)
        self.sb_focus_backlash.setValue(int(cfg.backlash_steps))

        self.sb_focus_travel = QSpinBox()
        self.sb_focus_travel.setRange(0, 1_000_000)
        self.sb_focus_travel.setValue(int(cfg.max_travel_steps))

        self.btn_focus_near = QPushButton("Acercar  ▲")
        self.btn_focus_far = QPushButton("Alejar  ▼")
        self.btn_focus_stop = QPushButton("Parar")
        self.btn_focus_zero = QPushButton("Poner a cero")
        _set_option_tooltip(
            self.btn_focus_near,
            "Mueve el enfocador un paso en el sentido de acercar el foco.",
        )
        _set_option_tooltip(
            self.btn_focus_far,
            "Mueve el enfocador un paso en el sentido de alejar el foco.",
        )
        _set_option_tooltip(
            self.btn_focus_stop,
            "Detiene el enfocador y cancela la búsqueda automática, sin tocar la montura.",
        )
        _set_option_tooltip(
            self.btn_focus_zero,
            "Declara la posición actual como el cero de esta sesión. El enfocador "
            "no tiene encoder: la posición sólo tiene sentido dentro de la sesión.",
        )
        self.btn_focus_near.clicked.connect(lambda: self._focus_move(+1))
        self.btn_focus_far.clicked.connect(lambda: self._focus_move(-1))
        self.btn_focus_stop.clicked.connect(self._focus_cancel)
        self.btn_focus_zero.clicked.connect(self._focus_zero)

        _add_option_row(
            manual_form,
            "Paso:",
            self.sb_focus_step,
            "Microsteps enviados por cada pulsación de acercar/alejar.",
        )
        _add_option_row(
            manual_form,
            "Delay:",
            self.sb_focus_delay,
            "Retardo mínimo entre microsteps del enfocador: fija su velocidad.",
        )
        _add_option_row(
            manual_form,
            "Movimiento:",
            self.dd_focus_profile,
            "Suave acelera y frena con una curva S; directo aplica de inmediato la velocidad indicada.",
        )
        _add_option_row(
            manual_form,
            "Sentido:",
            self.cb_focus_invert,
            "Marca esto si «acercar» aleja el foco: depende de cómo quedó montado el motor sobre la ruedita.",
        )
        _add_option_row(
            manual_form,
            "Juego (backlash):",
            self.sb_focus_backlash,
            "Microsteps que el acople se traga al invertir el sentido. Se envían "
            "antes del movimiento útil y no cuentan como recorrido.",
        )
        _add_option_row(
            manual_form,
            "Recorrido máximo:",
            self.sb_focus_travel,
            "Tope de seguridad respecto al cero de sesión. El recorrido útil del "
            "enfocador es corto y forzarlo contra el tope es lo único que puede romperlo. 0 lo desactiva.",
        )

        move_row = QHBoxLayout()
        move_row.addWidget(self.btn_focus_near)
        move_row.addWidget(self.btn_focus_far)
        move_row.addWidget(self.btn_focus_stop)
        manual_form.addRow(move_row)
        manual_form.addRow(self.btn_focus_zero)
        manual_box.setLayout(manual_form)

        # ---- Automático ----
        auto_box = QGroupBox("Búsqueda automática")
        auto_form = QFormLayout()

        self.sb_focus_coarse_step = QSpinBox()
        self.sb_focus_coarse_step.setRange(1, 100_000)
        self.sb_focus_coarse_step.setValue(int(cfg.autofocus_coarse_step))

        self.sb_focus_coarse_points = QSpinBox()
        self.sb_focus_coarse_points.setRange(3, 99)
        self.sb_focus_coarse_points.setValue(int(cfg.autofocus_coarse_points))

        self.sb_focus_fine_step = QSpinBox()
        self.sb_focus_fine_step.setRange(1, 100_000)
        self.sb_focus_fine_step.setValue(int(cfg.autofocus_fine_step))

        self.sb_focus_fine_points = QSpinBox()
        self.sb_focus_fine_points.setRange(3, 99)
        self.sb_focus_fine_points.setValue(int(cfg.autofocus_fine_points))

        self.ds_focus_settle = QDoubleSpinBox()
        self.ds_focus_settle.setRange(0.0, 30.0)
        self.ds_focus_settle.setDecimals(2)
        self.ds_focus_settle.setSingleStep(0.1)
        self.ds_focus_settle.setValue(float(cfg.autofocus_settle_s))
        self.ds_focus_settle.setSuffix(" s")

        self.sb_focus_frames = QSpinBox()
        self.sb_focus_frames.setRange(1, 30)
        self.sb_focus_frames.setValue(int(cfg.autofocus_frames))

        self.btn_focus_auto = QPushButton("Buscar mejor foco")
        _set_option_tooltip(
            self.btn_focus_auto,
            "Barre el rango grueso, localiza el máximo de nitidez y lo afina. "
            "Necesita la cámara conectada y capturando.",
        )
        self.btn_focus_auto.clicked.connect(self._focus_autofocus)

        _add_option_row(
            auto_form,
            "Paso grueso:",
            self.sb_focus_coarse_step,
            "Separación entre medidas del primer barrido. Debe ser lo bastante "
            "grande para cruzar la zona de foco en pocos puntos.",
        )
        _add_option_row(
            auto_form,
            "Puntos gruesos:",
            self.sb_focus_coarse_points,
            "Medidas del primer barrido, centradas en la posición actual. Si el "
            "máximo cae en un extremo, el barrido se extiende solo.",
        )
        _add_option_row(
            auto_form,
            "Paso fino:",
            self.sb_focus_fine_step,
            "Separación del segundo barrido alrededor del máximo grueso. Marca la precisión final.",
        )
        _add_option_row(
            auto_form,
            "Puntos finos:",
            self.sb_focus_fine_points,
            "Medidas del segundo barrido.",
        )
        _add_option_row(
            auto_form,
            "Asentamiento:",
            self.ds_focus_settle,
            "Espera tras cada movimiento antes de medir, para que el tubo deje de vibrar.",
        )
        _add_option_row(
            auto_form,
            "Frames por punto:",
            self.sb_focus_frames,
            "Se toma la mediana de la nitidez de estos frames. Más frames "
            "amortiguan el seeing, que en una sola toma puede superar la "
            "diferencia entre dos posiciones vecinas.",
        )
        auto_form.addRow(self.btn_focus_auto)
        auto_box.setLayout(auto_form)

        # ---- Presets por barlow ----
        preset_box = QGroupBox("Posiciones guardadas")
        preset_form = QFormLayout()

        self.btn_focus_home = QPushButton("Homing (retraer hasta el tope)")
        _set_option_tooltip(
            self.btn_focus_home,
            "Retrae el enfocador más allá de su recorrido. El piñón tiene dientes "
            "rotos en ese extremo, así que patina sin forzar nada y queda siempre "
            "en el mismo sitio físico. Es el origen que hace que una posición "
            "guardada signifique lo mismo mañana.",
        )
        self.btn_focus_home.clicked.connect(self._focus_home)
        preset_form.addRow(self.btn_focus_home)

        self.btn_focus_goto_preset = {}
        self.btn_focus_save_preset = {}
        for name in FOCUS_PRESET_NAMES:
            row = QHBoxLayout()
            go = QPushButton(f"Ir a {name}")
            save = QPushButton("Guardar aquí")
            _set_option_tooltip(
                go, f"Mueve el enfocador a la posición guardada para el barlow {name}."
            )
            _set_option_tooltip(
                save,
                f"Guarda la posición actual como el foco del barlow {name}. "
                "Hazlo con la sesión homed, o sólo valdrá hasta que cierres la app.",
            )
            go.clicked.connect(lambda _checked=False, n=name: self._focus_preset(n))
            save.clicked.connect(lambda _checked=False, n=name: self._focus_save_preset(n))
            row.addWidget(go)
            row.addWidget(save)
            self.btn_focus_goto_preset[name] = go
            self.btn_focus_save_preset[name] = save
            preset_form.addRow(f"Barlow {name}:", row)

        preset_box.setLayout(preset_form)

        self.btn_focus_apply = QPushButton("Aplicar parámetros")
        _set_option_tooltip(
            self.btn_focus_apply,
            "Guarda estos valores en la configuración del enfocador. Los botones "
            "manuales y la búsqueda automática ya envían lo que está en pantalla.",
        )
        self.btn_focus_apply.clicked.connect(self._focus_apply)

        self.lbl_focus_status = QLabel("Enfocador: sin montura conectada")
        self.lbl_focus_status.setWordWrap(True)
        self.lbl_focus_status.setStyleSheet("color:#bbb;")

        layout.addWidget(manual_box)
        layout.addWidget(preset_box)
        layout.addWidget(auto_box)
        layout.addWidget(self.btn_focus_apply)
        layout.addWidget(self.lbl_focus_status)
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

        layout.addLayout(actions)
        layout.addWidget(control_box)
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

        # --- Fuente de imagen y verificacion ---
        self.dd_ps_source = QComboBox()
        self.dd_ps_source.addItem("Cuadro vivo", "live")
        self.dd_ps_source.addItem("Mosaico apilado", "stack")
        _set_option_tooltip(
            self.dd_ps_source,
            "Imagen que resuelve el solver. Con cielo contaminado, el mosaico apilado suele "
            "ganar: exposiciones cortas mantienen las estrellas puntuales en vez de dejar "
            "trazas, la señal acumulada saca estrellas más débiles, y el mosaico cubre más "
            "cielo que un cuadro suelto. Requiere el apilado en marcha.",
        )

        self.ds_ps_verify_tol = QDoubleSpinBox()
        self.ds_ps_verify_tol.setRange(1.0, 600.0)
        self.ds_ps_verify_tol.setDecimals(1)
        self.ds_ps_verify_tol.setValue(
            float(getattr(self.cfg.platesolving, "verify_pointing_tol_arcsec", 30.0))
        )
        self.ds_ps_verify_tol.setSuffix(" ″")
        _set_option_tooltip(
            self.ds_ps_verify_tol,
            "Diferencia máxima de apuntado admitida al verificar un solve nuevo contra el "
            "anterior ya confirmado. Si se supera, se descarta el atajo y se rehace el "
            "solve completo.",
        )

        self.ds_ps_verify_roll = QDoubleSpinBox()
        self.ds_ps_verify_roll.setRange(0.1, 45.0)
        self.ds_ps_verify_roll.setDecimals(2)
        self.ds_ps_verify_roll.setValue(
            float(getattr(self.cfg.platesolving, "verify_roll_tol_deg", 3.0))
        )
        self.ds_ps_verify_roll.setSuffix(" deg")
        _set_option_tooltip(
            self.ds_ps_verify_roll,
            "Diferencia máxima de giro de campo admitida en esa misma verificación.",
        )

        self.sb_ps_min_validation = QSpinBox()
        self.sb_ps_min_validation.setRange(0, 50)
        self.sb_ps_min_validation.setValue(
            int(getattr(self.cfg.platesolving, "min_validation_inliers", 2))
        )
        _set_option_tooltip(
            self.sb_ps_min_validation,
            "Coincidencias exigidas más allá del triplete semilla. Una tripleta aporta 3 por "
            "construcción y siempre encaja consigo misma, así que un solve de 3 inliers no "
            "prueba nada por bajo que sea su rms. Ésta es la red de seguridad principal: "
            "bajarla a 0 deja pasar coincidencias falsas contra el catálogo.",
        )

        self.cb_ps_temporal = QCheckBox("Confirmación temporal de fuentes")
        self.cb_ps_temporal.setChecked(
            bool(getattr(self.cfg.platesolving, "temporal_detection_enabled", True))
        )
        _set_option_tooltip(
            self.cb_ps_temporal,
            "Exige que una fuente persista en varios cuadros antes de usarla, lo que descarta "
            "píxeles calientes y rayos cósmicos. Con exposiciones largas la deriva sideral "
            "puede descorrelacionar las fuentes y dejar el campo sin detecciones; y sobre el "
            "mosaico apilado no aporta nada, porque el apilado ya promedia varios cuadros.",
        )

        rowsrc = QHBoxLayout()
        rowsrc.addWidget(QLabel("Fuente:"))
        rowsrc.addWidget(self.dd_ps_source)
        rowsrc.addSpacing(12)
        rowsrc.addWidget(QLabel("Validación:"))
        rowsrc.addWidget(self.sb_ps_min_validation)
        rowsrc.addStretch(1)

        rowcons = QHBoxLayout()
        rowcons.addWidget(QLabel("Tol. apuntado:"))
        rowcons.addWidget(self.ds_ps_verify_tol)
        rowcons.addSpacing(12)
        rowcons.addWidget(QLabel("Tol. giro:"))
        rowcons.addWidget(self.ds_ps_verify_roll)
        rowcons.addStretch(1)

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
        form.addRow(rowsrc)
        form.addRow(rowcons)
        form.addRow(self.cb_ps_temporal)
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
    FocuserTabMixin,
    TrackingTabMixin,
    StackingTabMixin,
    GoToTabMixin,
):
    def _build_modules_tabs(self: "AstroPanoptesWindow") -> QWidget:
        tabs = QTabWidget()
        pages = (
            (
                self._tab_observer(),
                "Observador",
                "Ubicación, escala óptica y prior de rotación para plate solving.",
            ),
            (
                self._tab_camera(),
                "Camera",
                "Exposición, ganancia y captura RAW de diagnóstico.",
            ),
            (
                self._tab_focuser(),
                "Enfoque",
                "Enfocador motorizado: acercar/alejar manual y búsqueda automática del mejor foco.",
            ),
            (
                self._tab_tracking(),
                "Tracking",
                "Control de tracking y alineación RAW16 directa con feed-forward sideral.",
            ),
            (
                self._tab_stacking(),
                "Stacking",
                "Live stacking, color, drizzle, alineación y preview del stack.",
            ),
            (
                self._tab_goto(),
                "GoTo",
                "Plate Solving y toma manual de muestras, ajuste del modelo GoTo y estrellas esperadas.",
            ),
        )
        for page, title, tooltip in pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(page)
            tabs.setTabToolTip(tabs.addTab(scroll, title), tooltip)

        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(tabs)

        self.modules_tabs = tabs
        return wrap
