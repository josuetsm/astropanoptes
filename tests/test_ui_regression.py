import os
import unittest

import numpy as np

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
        self.assertIn("Gaia", labels)
        self.assertNotIn("Plate Solving", labels)

    def test_on_tick_without_plate_solving_widgets(self) -> None:
        self.assertFalse(getattr(self.window, "_ps_outputs_enabled", True))
        self.window._on_tick()

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

    def test_download_gaia_field_button_calls_runner(self) -> None:
        calls: list[dict] = []

        def _capture(**kwargs):
            calls.append(dict(kwargs))

        self.runner.request_platesolving_download_current_field = _capture

        self.window.btn_download_gaia.click()

        self.assertEqual(len(calls), 1)

    def test_gaia_panel_displays_cache_and_current_field_coverage(self) -> None:
        self.runner.get_gaia_coverage = lambda: {
            "cache_dir": "/tmp/gaia",
            "table_name": "gaiadr3.gaia_source",
            "gmax": 15.0,
            "nside": 1,
            "order": "ring",
            "total_tiles": 12,
            "cached_tiles": [4],
            "cached_tile_count": 1,
            "coverage_fraction": 1.0 / 12.0,
            "covered_area_sq_deg": 3437.75,
            "cached_bytes": 2048,
            "newest_mtime": None,
            "tile_az_deg": np.linspace(0.0, 330.0, 12),
            "tile_alt_deg": np.linspace(-60.0, 60.0, 12),
            "field_available": False,
            "field_required_tiles": [4, 5],
            "field_cached_tiles": [4],
            "field_missing_tiles": [5],
            "field_radius_deg": 2.0,
            "center_az_deg": 120.0,
            "center_alt_deg": 30.0,
            "projection_time_utc": "2026-06-05T12:00:00.000",
            "observer_lat_deg": -33.3667,
            "observer_lon_deg": -71.6667,
            "field_source": "simulation",
        }
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        self.window.modules_tabs.setCurrentIndex(labels.index("Gaia"))

        self.assertIn("1 / 12", self.window.lbl_gaia_tiles.text())
        self.assertIn("1 / 2", self.window.lbl_gaia_field.text())
        self.assertIn("incompleta", self.window.lbl_gaia_field_status.text())
        self.assertIn("Az 120.00", self.window.lbl_gaia_center.text())
        self.assertIn("Alt +30.00", self.window.lbl_gaia_center.text())
        self.assertIn("simulation", self.window.lbl_gaia_center.text())

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

    def test_gaia_panel_temporarily_hides_manual_controls(self) -> None:
        labels = [
            self.window.modules_tabs.tabText(i)
            for i in range(self.window.modules_tabs.count())
        ]
        goto_index = labels.index("GoTo")
        gaia_index = labels.index("Gaia")
        self.window.modules_tabs.setCurrentIndex(goto_index)
        self.window.dock_manual.show()
        self.assertFalse(self.window.dock_manual.isHidden())

        self.window.modules_tabs.setCurrentIndex(gaia_index)
        self.assertTrue(self.window.dock_manual.isHidden())

        self.window.modules_tabs.setCurrentIndex(goto_index)
        self.assertFalse(self.window.dock_manual.isHidden())

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
