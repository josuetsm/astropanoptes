import json
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
)

from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from app_runner import AppRunner
from ap_types import Axis, CameraStatus, MountStatus
from config import AppConfig
from ui.pyqt6_app import AstroPanoptesWindow


def _key_event(key: Qt.Key) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


class UiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # Ruta propia por test: si no, cerrar la ventana pisaría los ajustes
        # reales del usuario en ~/.astropanoptes/settings.json.
        self._tmp = tempfile.TemporaryDirectory()
        self.settings_path = Path(self._tmp.name) / "settings.json"
        self.cfg = AppConfig()
        self.runner = AppRunner(self.cfg)
        self.window = AstroPanoptesWindow(
            self.runner, self.cfg, settings_path=self.settings_path
        )

    def tearDown(self) -> None:
        self.window.close()
        self._tmp.cleanup()

    def test_plate_solving_is_available_in_goto_panel(self) -> None:
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        self.assertIn("GoTo", labels)
        self.assertNotIn("Gaia", labels)
        self.assertNotIn("Plate Solving", labels)
        self.assertEqual(self.window.btn_platesolve.text(), "Plate Solving")
        self.assertEqual(
            [
                self.window.dd_platesolve_mode.itemText(i)
                for i in range(self.window.dd_platesolve_mode.count())
            ],
            ["Deriva", "Alt/Az (manual)", "Alt/Az (registrado)"],
        )
        self.assertFalse(hasattr(self.window, "btn_autocal"))
        self.assertFalse(hasattr(self.window, "btn_accept_sample"))
        self.assertFalse(hasattr(self.window, "btn_reject_sample"))

    def test_plate_solving_manual_altaz_submits_operator_target(self) -> None:
        calls: list[dict] = []
        self.runner.request_goto_autocalibrate = lambda params: calls.append(dict(params))
        self.window.dd_platesolve_mode.setCurrentText("Alt/Az (manual)")
        self.window.ds_goto_ps_az.setValue(123.456)
        self.window.ds_goto_ps_alt.setValue(42.25)

        self.window.btn_platesolve.click()

        self.assertTrue(calls)
        self.assertEqual(calls[-1]["autocal_ps_mode"], "manual_altaz")
        self.assertEqual(
            calls[-1]["autocal_ps_target"],
            {"az_deg": 123.456, "alt_deg": 42.25},
        )
        self.assertNotIn("exp_ms", calls[-1])
        self.assertNotIn("gain", calls[-1])

    def test_plate_solving_non_manual_modes_use_no_operator_target(self) -> None:
        calls: list[dict] = []
        self.runner.request_goto_autocalibrate = lambda params: calls.append(dict(params))

        for label, value in (
            ("Deriva", "drift"),
            ("Alt/Az (registrado)", "current_altaz"),
        ):
            self.window.dd_platesolve_mode.setCurrentText(label)
            self.window.btn_platesolve.click()
            self.assertEqual(calls[-1]["autocal_ps_mode"], value)
            self.assertNotIn("autocal_ps_target", calls[-1])

    def test_on_tick_without_plate_solving_widgets(self) -> None:
        self.assertFalse(getattr(self.window, "_ps_outputs_enabled", True))
        self.window._on_tick()

    def test_window_minimum_size_fits_available_desktop(self) -> None:
        available = self._app.primaryScreen().availableGeometry()

        self.window.show()
        self._app.processEvents()

        minimum = self.window.minimumSizeHint()
        self.assertLessEqual(minimum.width(), available.width())
        self.assertLessEqual(minimum.height(), available.height())

        self.window.resize(available.size())
        self._app.processEvents()
        self.assertLessEqual(self.window.width(), available.width())
        self.assertLessEqual(self.window.height(), available.height())

        status_labels = (
            self.window.lbl_fps,
            self.window.lbl_drift,
            self.window.lbl_coords,
            self.window.lbl_errors,
        )
        for left, right in zip(status_labels, status_labels[1:]):
            self.assertLess(left.geometry().right(), right.geometry().left())

    def test_module_panels_scroll_instead_of_growing_window(self) -> None:
        pages = [
            self.window.modules_tabs.widget(index)
            for index in range(self.window.modules_tabs.count())
        ]

        self.assertTrue(pages)
        self.assertTrue(all(isinstance(page, QScrollArea) for page in pages))

    def test_long_error_is_bounded_and_kept_in_tooltip(self) -> None:
        long_error = "camera transport timeout " * 250
        self.runner._update_state({"camera": {"last_error": long_error}})

        self.window._update_error_banner(self.runner.get_state())
        self.window.show()
        self._app.processEvents()

        self.assertLessEqual(len(self.window.lbl_errors.text()), 90)
        self.assertIn(long_error, self.window.lbl_errors.toolTip())
        available_width = self._app.primaryScreen().availableGeometry().width()
        self.assertLessEqual(self.window.minimumSizeHint().width(), available_width)

    def test_toolbar_has_one_toggle_button_per_device(self) -> None:
        self.assertFalse(hasattr(self.window, "btn_connect_camera"))
        self.assertFalse(hasattr(self.window, "btn_disconnect_camera"))
        self.assertFalse(hasattr(self.window, "btn_connect_mount"))
        self.assertFalse(hasattr(self.window, "btn_disconnect_mount"))
        self.assertEqual(self.window.btn_camera_connection.text(), "Connect camera")
        self.assertEqual(self.window.btn_mount_connection.text(), "Connect mount")

    def test_manual_controls_select_smooth_or_direct_move_profile(self) -> None:
        calls: list[tuple] = []
        self.runner.request_mount_move_steps = lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )
        self.window.sb_steps.setValue(1234)
        self.window.sb_delay.setValue(567)

        self.assertEqual(self.window.sb_delay.minimum(), 10)
        self.assertEqual(self.window.dd_manual_move_profile.currentData(), "smooth")
        self.window._manual_move(Axis.ALT, -1)
        self.assertEqual(calls[-1][0], (Axis.ALT, -1, 1234, 567))
        self.assertEqual(calls[-1][1], {"profile": "smooth"})

        self.window.dd_manual_move_profile.setCurrentIndex(1)
        self.window._manual_move(Axis.AZ, 1)
        self.assertEqual(calls[-1][0], (Axis.AZ, 1, 1234, 567))
        self.assertEqual(calls[-1][1], {"profile": "direct"})

    def test_camera_connection_button_toggles_from_runner_state(self) -> None:
        calls: list[str] = []
        self.runner.request_camera_connect = lambda _index: calls.append("connect")
        self.runner.request_camera_disconnect = lambda: calls.append("disconnect")

        self.window.btn_camera_connection.click()
        self.assertEqual(calls, ["connect"])
        self.assertFalse(self.window.btn_camera_connection.isEnabled())

        self.runner._update_state(
            {"camera": {"connected": True, "status": CameraStatus.OK}}
        )
        self.window._update_chips_from_state(self.runner.get_state())
        self.assertTrue(self.window.btn_camera_connection.isEnabled())
        self.assertEqual(self.window.btn_camera_connection.text(), "Disconnect camera")

        self.window.btn_camera_connection.click()
        self.assertEqual(calls, ["connect", "disconnect"])

    def test_camera_apply_batches_settings_and_log_is_bounded(self) -> None:
        calls: list[dict] = []
        self.runner.request_camera_params = lambda params: calls.append(dict(params))
        self.window.ds_exp_ms.setValue(250.0)
        self.window.sb_gain.setValue(320)
        self.window.sb_offset.setValue(24)
        self.window.ds_gamma.setValue(1.2)

        self.window._camera_apply()

        self.assertEqual(
            calls,
            [{"exp_ms": 250.0, "gain": 320, "offset": 24, "gamma": 1.2}],
        )
        self.assertEqual(self.window.log.document().maximumBlockCount(), 3000)

    def test_mount_connection_button_toggles_from_runner_state(self) -> None:
        calls: list[str] = []
        self.runner.request_mount_connect = lambda _port, _baud: calls.append("connect")
        self.runner.request_mount_disconnect = lambda: calls.append("disconnect")

        self.window.btn_mount_connection.click()
        self.assertEqual(calls, ["connect"])
        self.assertFalse(self.window.btn_mount_connection.isEnabled())

        self.runner._update_state(
            {"mount": {"connected": True, "status": MountStatus.OK}}
        )
        self.window._update_chips_from_state(self.runner.get_state())
        self.assertTrue(self.window.btn_mount_connection.isEnabled())
        self.assertEqual(self.window.btn_mount_connection.text(), "Disconnect mount")

        self.window.btn_mount_connection.click()
        self.assertEqual(calls, ["connect", "disconnect"])

    def test_expected_stars_overlay_is_enabled_after_model_fit(self) -> None:
        calls: list[dict] = []
        self.runner.request_expected_stars_params = lambda **kwargs: calls.append(dict(kwargs))

        self.assertFalse(self.window.cb_expected_stars.isEnabled())
        self.runner._update_state(
            {
                "goto": {
                    "model_fit_samples": 4,
                    "synced": True,
                }
            }
        )
        self.window._on_tick()
        self.assertTrue(self.window.cb_expected_stars.isEnabled())

        self.window.cb_expected_stars.click()
        self.assertTrue(calls)
        self.assertTrue(calls[-1]["enabled"])
        self.assertEqual(calls[-1]["mag_limit"], 15.0)

    def test_download_gaia_field_button_is_removed(self) -> None:
        self.assertFalse(hasattr(self.window, "btn_download_gaia"))

    def test_gaia_coverage_panel_is_removed(self) -> None:
        for attribute in ("gaia_tab", "gaia_coverage_map", "btn_gaia_refresh"):
            self.assertFalse(hasattr(self.window, attribute), attribute)
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        self.assertNotIn("Gaia", labels)

    def test_object_detection_tab_is_removed(self) -> None:
        for attribute in ("sb_od_minarea", "ds_od_sigma", "btn_od_start", "ch_od"):
            self.assertFalse(hasattr(self.window, attribute), attribute)
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        self.assertNotIn("Object Detection", labels)

    def test_every_module_tab_has_a_tooltip(self) -> None:
        tabs = self.window.modules_tabs
        for index in range(tabs.count()):
            self.assertTrue(tabs.tabToolTip(index), tabs.tabText(index))

    def _wait_for_console(self, predicate, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            QCoreApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _wait_until(self, predicate, timeout_s: float = 10.0) -> bool:
        return self._wait_for_console(predicate, timeout_s)

    def test_console_replaces_the_logs_panel(self) -> None:
        self.assertIs(self.window.log, self.window.console.view)
        self.assertEqual("Consola", self.window.console_frame.title())
        self.assertTrue(self.window.act_console.isChecked())

    def test_console_echoes_and_runs_a_command_against_the_session(self) -> None:
        console = self.window.console
        console.input.setText("status")
        console.input.returnPressed.emit()

        self.assertIn("> status", console.view.toPlainText())
        ran = self._wait_for_console(
            lambda: "camera" in console.view.toPlainText().lower()
        )
        self.assertTrue(ran, console.view.toPlainText())

    def test_console_reports_unknown_commands_without_crashing(self) -> None:
        console = self.window.console
        console.run_command("comando-que-no-existe")
        failed = self._wait_for_console(
            lambda: "CLI ERROR" in console.view.toPlainText()
        )
        self.assertTrue(failed, console.view.toPlainText())

    def test_console_input_keeps_command_history(self) -> None:
        entry = self.window.console.input
        for line in ("tracking start", "stacking start"):
            entry.setText(line)
            entry.returnPressed.emit()

        entry.keyPressEvent(_key_event(Qt.Key.Key_Up))
        self.assertEqual("stacking start", entry.text())
        entry.keyPressEvent(_key_event(Qt.Key.Key_Up))
        self.assertEqual("tracking start", entry.text())
        entry.keyPressEvent(_key_event(Qt.Key.Key_Down))
        self.assertEqual("stacking start", entry.text())

    def test_settings_survive_a_restart(self) -> None:
        # Los ajustes se leen de `runner.cfg`, así que el runner debe estar
        # corriendo para que la cola de acciones se drene antes de guardar.
        self.runner.start()
        self.addCleanup(self.runner.stop)
        self.window.dd_obs_site.setCurrentIndex(1)
        self.window.ds_obs_focal_mm.setValue(1200.0)
        self.window.dd_obs_barlow.setCurrentIndex(1)
        self.window.ds_exp_ms.setValue(250.0)
        self.window.sb_gain.setValue(320)
        self.window.ds_gamma.setValue(1.4)
        self.window.cb_st_color.setChecked(True)
        self.window.dd_st_drizzle.setCurrentIndex(1)
        self.window._camera_apply()
        self.assertTrue(
            self._wait_until(lambda: self.runner.cfg.camera.exp_ms == 250.0),
            "la cámara no aplicó los parámetros a tiempo",
        )

        site = self.window.dd_obs_site.currentText()
        self.window.close()
        self.assertTrue(self.settings_path.exists())

        runner = AppRunner(AppConfig())
        window = AstroPanoptesWindow(
            runner, AppConfig(), settings_path=self.settings_path
        )
        try:
            self.assertEqual(site, window.dd_obs_site.currentText())
            self.assertAlmostEqual(1200.0, window.ds_obs_focal_mm.value())
            self.assertEqual(2, window._observer_barlow_factor())
            self.assertAlmostEqual(250.0, window.ds_exp_ms.value())
            self.assertEqual(320, window.sb_gain.value())
            self.assertAlmostEqual(1.4, window.ds_gamma.value())
            self.assertTrue(window.cb_st_color.isChecked())
            self.assertEqual(2.0, window.dd_st_drizzle.currentData())
        finally:
            window.close()

    def test_restored_settings_reach_the_running_session(self) -> None:
        self.runner.start()
        self.addCleanup(self.runner.stop)
        self.window.ds_exp_ms.setValue(180.0)
        self.window.sb_gain.setValue(210)
        self.window._camera_apply()
        self.assertTrue(
            self._wait_until(lambda: self.runner.cfg.camera.gain == 210),
            "la cámara no aplicó los parámetros a tiempo",
        )
        self.window.close()

        runner = AppRunner(AppConfig())
        calls: list[dict] = []
        runner.request_camera_params = lambda params: calls.append(dict(params))
        window = AstroPanoptesWindow(
            runner, AppConfig(), settings_path=self.settings_path
        )
        try:
            self.assertTrue(calls)
            self.assertAlmostEqual(180.0, calls[-1]["exp_ms"])
            self.assertEqual(210, calls[-1]["gain"])
        finally:
            window.close()

    def test_unusable_settings_file_does_not_block_startup(self) -> None:
        for payload in ("{ not json", json.dumps({"version": 999, "camera": {"gain": 1}})):
            self.settings_path.write_text(payload, encoding="utf-8")
            runner = AppRunner(AppConfig())
            cfg = AppConfig()
            window = AstroPanoptesWindow(
                runner, cfg, settings_path=self.settings_path
            )
            try:
                self.assertEqual(cfg.camera.gain, window.sb_gain.value())
            finally:
                window.close()

    def test_tracking_tab_applies_exposed_parameters(self) -> None:
        calls: list[dict] = []
        self.runner.request_tracking_params = lambda **kwargs: calls.append(dict(kwargs))

        self.window.ds_tr_resp_min.setValue(0.123)
        self.window.cb_tr_ff.setChecked(False)
        self.window.ds_tr_ff_gain.setValue(0.75)
        self.window.ds_tr_ff_dt.setValue(2.5)
        self.window.ds_tr_ff_cond.setValue(1234.0)
        self.window.ds_tr_ff_hold.setValue(4.5)
        self.window.ds_tr_ff_slew.setValue(88.0)

        self.window.btn_tr_apply.click()

        self.assertTrue(calls)
        params = calls[-1]
        self.assertAlmostEqual(params["resp_min"], 0.123, places=3)
        self.assertFalse(params["sidereal_ff_enabled"])
        self.assertAlmostEqual(params["sidereal_ff_gain"], 0.75)
        self.assertAlmostEqual(params["sidereal_ff_dt_s"], 2.5)
        self.assertEqual([], [key for key in params if key.startswith("sep_")])

    def test_stacking_tab_applies_exposed_parameters(self) -> None:
        calls: list[dict] = []
        self.runner.request_stacking_params = lambda **kwargs: calls.append(dict(kwargs))

        self.window.cb_st_color.setChecked(True)
        self.window.dd_st_bayer.setCurrentText("BGGR")
        self.window.dd_st_drizzle.setCurrentIndex(1)
        self.window.sb_st_batch.setValue(4)
        self.window.sb_st_max_queue.setValue(16)
        self.window.sb_st_align_median.setValue(6)
        self.window.sb_st_smooth.setValue(12)
        self.window.sb_st_max_shift.setValue(42)
        self.window.cb_st_subpixel.setChecked(False)
        self.window.ds_st_preview_hz.setValue(2.5)
        self.window.ds_st_preview_vmin.setValue(9.5)

        self.window.btn_st_apply.click()

        self.assertTrue(calls)
        params = calls[-1]
        self.assertEqual(params["color_mode"], "rgb")
        self.assertEqual(params["bayer_pattern"], "BGGR")
        self.assertEqual(params["drizzle_scale"], 2.0)
        self.assertEqual(params["batch_size"], 4)
        self.assertEqual(params["max_queue"], 16)
        self.assertEqual(params["align_median_k"], 7)
        self.assertEqual(params["smooth_k"], 12)
        self.assertEqual(params["max_shift_px"], 42)
        self.assertFalse(params["use_subpixel"])
        self.assertAlmostEqual(params["preview_hz"], 2.5)
        self.assertAlmostEqual(params["preview_log_vmin"], 9.5)

    def test_tracking_and_stacking_options_have_tooltips(self) -> None:
        self.assertIn("Respuesta mínima", self.window.ds_tr_resp_min.toolTip())
        self.assertIn("movimiento sideral", self.window.cb_tr_ff.toolTip())
        self.assertIn("mosaico Bayer", self.window.dd_st_bayer.toolTip())
        self.assertIn("desplazamientos fraccionales", self.window.cb_st_subpixel.toolTip())

        labels = [
            label
            for label in self.window.findChildren(QLabel)
            if label.text() == "resp_min:"
        ]
        self.assertTrue(labels)
        self.assertIn("Respuesta mínima", labels[0].toolTip())

    def test_interactive_controls_have_tooltips(self) -> None:
        missing: list[tuple[str, str]] = []
        classes = (
            QPushButton,
            QCheckBox,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QLineEdit,
            QToolButton,
        )
        for widget_cls in classes:
            for widget in self.window.findChildren(widget_cls):
                text = widget.text() if hasattr(widget, "text") else ""
                name = widget.objectName() or text or widget_cls.__name__
                internal_qt_widget = name.startswith("qt_") or name in {
                    "ScrollLeftButton",
                    "ScrollRightButton",
                }
                if internal_qt_widget:
                    continue
                if not widget.toolTip():
                    missing.append((widget_cls.__name__, name))

        self.assertEqual([], missing)

    def test_goto_uses_model_without_platesolving_parameters(self) -> None:
        calls: list[tuple[object, dict]] = []

        def _capture(target, **kwargs):
            calls.append((target, kwargs))

        self.runner.request_mount_goto = _capture

        self.window.dd_goto_mode.setCurrentText("name (SIMBAD)")
        self.window.ed_goto_name.setText("M42")

        self.window._goto_start()

        self.assertEqual(len(calls), 1)
        target, params = calls[0]
        self.assertEqual(target, "M42")
        self.assertEqual(params, {})
        self.assertFalse(hasattr(self.window, "cb_fb"))
        self.assertFalse(hasattr(self.window, "sb_stages"))

    def test_goto_mode_switch_keeps_target_widgets_alive(self) -> None:
        self.window.dd_goto_mode.setCurrentText("altaz")
        self.window.ds_az.setValue(123.456)
        self.window.ds_alt.setValue(45.678)

        self.window.dd_goto_mode.setCurrentText("radec")
        self.window.dd_radec_fmt.setCurrentText("deg")
        self.window.ds_ra.setValue(210.123456)
        self.window.ds_dec.setValue(-12.654321)

        self.window.dd_radec_fmt.setCurrentText("HMS/DMS")
        self.window.ed_ra_hms.setText("12:34:56")
        self.window.ed_dec_dms.setText("-12:34:56")

        self.window.dd_goto_mode.setCurrentText("name (SIMBAD)")
        self.window.dd_goto_mode.setCurrentText("altaz")
        self.window.dd_goto_mode.setCurrentText("radec")
        self.window.dd_radec_fmt.setCurrentText("deg")

        self.assertAlmostEqual(self.window.ds_az.value(), 123.456, places=6)
        self.assertAlmostEqual(self.window.ds_alt.value(), 45.678, places=6)
        self.assertAlmostEqual(self.window.ds_ra.value(), 210.123456, places=6)
        self.assertAlmostEqual(self.window.ds_dec.value(), -12.654321, places=6)
        self.assertEqual(self.window.ed_ra_hms.text(), "12:34:56")
        self.assertEqual(self.window.ed_dec_dms.text(), "-12:34:56")
