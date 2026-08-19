import os
import unittest

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

from app_runner import AppRunner
from ap_types import Axis, CameraStatus, MountStatus
from config import AppConfig
from ui.pyqt6_app import AstroPanoptesWindow


class UiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.cfg = AppConfig()
        self.runner = AppRunner(self.cfg)
        self.window = AstroPanoptesWindow(self.runner, self.cfg)

    def tearDown(self) -> None:
        self.window.close()

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
        self.window.sb_tr_sep_minarea.setValue(7)
        self.window.ds_tr_sep_sigma.setValue(4.25)
        self.window.sb_tr_sep_max_sources.setValue(123)
        self.window.sb_tr_sep_min_sources.setValue(3)
        self.window.sb_tr_sep_bw.setValue(32)
        self.window.sb_tr_sep_bh.setValue(48)

        self.window.btn_tr_apply.click()

        self.assertTrue(calls)
        params = calls[-1]
        self.assertAlmostEqual(params["resp_min"], 0.123, places=3)
        self.assertFalse(params["sidereal_ff_enabled"])
        self.assertAlmostEqual(params["sidereal_ff_gain"], 0.75)
        self.assertAlmostEqual(params["sidereal_ff_dt_s"], 2.5)
        self.assertEqual(params["sep_minarea"], 7)
        self.assertAlmostEqual(params["sep_thresh_sigma"], 4.25)
        self.assertEqual(params["sep_max_sources"], 123)
        self.assertEqual(params["sep_min_sources"], 3)
        self.assertEqual(params["sep_bw"], 32)
        self.assertEqual(params["sep_bh"], 48)

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
        self.assertIn("malla de fondo", self.window.sb_tr_sep_bw.toolTip())
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
