import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app_runner import AppRunner
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

    def test_modules_tabs_without_plate_solving(self) -> None:
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        self.assertIn("GoTo", labels)
        self.assertNotIn("Plate Solving", labels)

    def test_on_tick_without_plate_solving_widgets(self) -> None:
        self.assertFalse(getattr(self.window, "_ps_outputs_enabled", True))
        self.window._on_tick()

    def test_goto_uses_embedded_platesolving_controls(self) -> None:
        calls: list[tuple[object, dict]] = []

        def _capture(target, **kwargs):
            calls.append((target, kwargs))

        self.runner.request_mount_goto = _capture

        self.window.dd_goto_mode.setCurrentText("name (SIMBAD)")
        self.window.ed_goto_name.setText("M42")
        self.window.sb_goto_ps_nseeds.setValue(4)
        self.window.sb_goto_ps_mininl.setValue(7)
        self.window.cb_fb.setChecked(True)
        self.window.sb_stages.setValue(3)

        self.window._goto_start()

        self.assertEqual(len(calls), 1)
        target, params = calls[0]
        self.assertEqual(target, "M42")
        self.assertEqual(int(params["N_seed"]), 4)
        self.assertEqual(int(params["min_inliers"]), 7)
        self.assertEqual(int(params["stages"]), 3)
        self.assertTrue(bool(params["platesolving_feedback"]))

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
