from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from app_runner import AppRunner
from config import AppConfig
from ui.pyqt6_app import AstroPanoptesWindow


class PlatesolvingControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # Ruta propia: al cerrar, la ventana guarda ajustes, y con la ruta por
        # defecto este test pisaria ~/.astropanoptes/settings.json del usuario.
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = AppConfig()
        self.runner = AppRunner(self.cfg)
        self.window = AstroPanoptesWindow(
            self.runner,
            self.cfg,
            settings_path=Path(self._tmp.name) / "settings.json",
        )

    def tearDown(self) -> None:
        self.window.close()
        self._tmp.cleanup()

    def test_image_source_and_verification_are_exposed(self) -> None:
        """These drive whether a solve succeeds at all, so they belong in the UI."""
        for name in (
            "dd_ps_source",
            "ds_ps_verify_tol",
            "ds_ps_verify_roll",
            "sb_ps_min_validation",
            "cb_ps_temporal",
        ):
            self.assertTrue(hasattr(self.window, name), f"falta el control {name}")

    def test_source_offers_live_and_stack(self) -> None:
        values = [
            self.window.dd_ps_source.itemData(i)
            for i in range(self.window.dd_ps_source.count())
        ]
        self.assertEqual(values, ["live", "stack"])

    def test_controls_reach_the_solver(self) -> None:
        sent: list[dict] = []
        self.runner.request_platesolving_params = lambda **kw: sent.append(dict(kw))
        self.runner.request_goto_autocalibrate = lambda _p: None

        self.window.dd_ps_source.setCurrentIndex(1)          # stack
        self.window.ds_ps_verify_tol.setValue(45.0)
        self.window.ds_ps_verify_roll.setValue(5.0)
        self.window.sb_ps_min_validation.setValue(0)
        self.window.cb_ps_temporal.setChecked(False)

        self.window._goto_platesolve()

        self.assertTrue(sent, "no se enviaron parámetros al solver")
        params = sent[0]
        self.assertEqual(params["source"], "stack")
        self.assertEqual(params["verify_pointing_tol_arcsec"], 45.0)
        self.assertEqual(params["verify_roll_tol_deg"], 5.0)
        self.assertEqual(params["min_validation_inliers"], 0)
        self.assertFalse(params["temporal_detection_enabled"])

    def test_defaults_mirror_the_config(self) -> None:
        ps = self.cfg.platesolving
        self.assertEqual(
            self.window.sb_ps_min_validation.value(), int(ps.min_validation_inliers)
        )
        self.assertEqual(
            self.window.cb_ps_temporal.isChecked(), bool(ps.temporal_detection_enabled)
        )
        self.assertEqual(self.window.dd_ps_source.currentData(), "live")


if __name__ == "__main__":
    unittest.main()
