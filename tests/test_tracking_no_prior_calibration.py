"""El seguimiento no puede exigir calibrar antes de empezar.

La camara va montada derecha y las reducciones de los ejes se conocen, asi que
hay escala de partida de sobra para empezar a seguir y dejar que la
autocalibracion RLS la afine sobre la marcha.

Hasta ahora, si el modelo GoTo no sabia donde apuntaba -app recien abierta, o
despues de un ``goto reset``- la siembra por geometria fallaba y el estado se
quedaba con ``A_pinv=None``. El lazo de control comprueba justo eso antes de
mandar velocidades, asi que el seguimiento se activaba y no movia nada.
"""
from __future__ import annotations

import numpy as np
import pytest

from ap_types import Axis
from config import AppConfig


def _runner(monkeypatch, *, pointing=None):
    """AppRunner con lo justo para probar la siembra de calibracion."""
    import app_runner as ar
    from goto import GoToModel
    from tracking import make_tracking_state

    r = ar.AppRunner.__new__(ar.AppRunner)
    r.cfg = AppConfig()
    r.out_log = None
    model = GoToModel()
    model.init_from_mechanics()
    r._goto = type("_G", (), {"model": model})()
    r._tracking_state = make_tracking_state()
    monkeypatch.setattr(
        ar.AppRunner, "_tracking_pointing_altaz", lambda self: pointing, raising=False
    )
    return r


class TestNominalCalibration:
    def test_it_exists_without_knowing_where_the_mount_points(self, monkeypatch) -> None:
        r = _runner(monkeypatch, pointing=None)
        theta = r._tracking_nominal_calibration()
        assert theta is not None
        assert theta.shape == (2, 3)
        assert np.all(np.isfinite(theta))

    def test_it_is_invertible_so_the_loop_can_actually_command(self, monkeypatch) -> None:
        r = _runner(monkeypatch, pointing=None)
        theta = r._tracking_nominal_calibration()
        assert abs(float(np.linalg.det(theta[:, :2]))) > 1e-6

    def test_each_axis_carries_its_own_reduction(self, monkeypatch) -> None:
        """Altitud va a 90.5:1 y azimut a 45:1: no pueden salir iguales."""
        r = _runner(monkeypatch, pointing=None)
        theta = r._tracking_nominal_calibration()
        kin = r._goto.model.kin
        scale = 206265.0 * r.cfg.platesolving.pixel_size_m / r.cfg.platesolving.focal_m
        px_per_step_az = abs(float(kin.deg_per_step(Axis.AZ))) * 3600.0 / scale
        px_per_step_alt = abs(float(kin.deg_per_step(Axis.ALT))) * 3600.0 / scale
        assert np.linalg.norm(theta[:, 0]) == pytest.approx(px_per_step_az, rel=1e-6)
        assert np.linalg.norm(theta[:, 1]) == pytest.approx(px_per_step_alt, rel=1e-6)
        assert px_per_step_alt < px_per_step_az

    def test_stars_move_opposite_to_the_tube(self, monkeypatch) -> None:
        r = _runner(monkeypatch, pointing=None)
        r.cfg.camera.roll_deg = 0.0
        theta = r._tracking_nominal_calibration()
        j = r._goto.model.J_deg_per_step
        # Un paso positivo de cada eje corre el campo en sentido contrario.
        assert np.sign(theta[0, 0]) == -np.sign(j[0, 0])
        assert np.sign(theta[1, 1]) == -np.sign(j[1, 1])

    def test_an_unknown_altitude_never_underestimates_the_response(
        self, monkeypatch
    ) -> None:
        """Sobreestimar la respuesta corrige despacio; subestimarla se pasa."""
        blind = _runner(monkeypatch, pointing=None)._tracking_nominal_calibration()
        for alt in (10.0, 45.0, 70.0):
            known = _runner(
                monkeypatch, pointing=(120.0, alt)
            )._tracking_nominal_calibration()
            assert np.linalg.norm(blind[:, 0]) >= np.linalg.norm(known[:, 0]) - 1e-12

    def test_the_camera_roll_rotates_the_mapping(self, monkeypatch) -> None:
        straight = _runner(monkeypatch, pointing=None)
        straight.cfg.camera.roll_deg = 0.0
        a0 = straight._tracking_nominal_calibration()

        tilted = _runner(monkeypatch, pointing=None)
        tilted.cfg.camera.roll_deg = 30.0
        a30 = tilted._tracking_nominal_calibration()

        assert not np.allclose(a0[:, :2], a30[:, :2])
        # Girar la camara no cambia cuanto se mueve el campo, solo hacia donde.
        for col in (0, 1):
            assert np.linalg.norm(a0[:, col]) == pytest.approx(
                np.linalg.norm(a30[:, col]), rel=1e-9
            )

    def test_a_broken_plate_scale_yields_no_calibration(self, monkeypatch) -> None:
        r = _runner(monkeypatch, pointing=None)
        r.cfg.platesolving.focal_m = 0.0
        assert r._tracking_nominal_calibration() is None


class TestTrackingStartsWithoutAnyCalibration:
    """La prueba que importa: arrancar en frio y que el lazo pueda mandar."""

    def test_start_leaves_the_loop_able_to_command(self) -> None:
        from actions import ActionType
        from app_runner import AppRunner

        runner = AppRunner(AppConfig())
        # Modelo GoTo virgen: no sabe donde apunta, que es el caso que fallaba.
        assert not runner._goto.model.synced

        runner._handle_tracking_action(ActionType.TRACKING_START, {})

        auto = runner._tracking_state.auto
        assert auto.ok, "el seguimiento arranco sin calibracion utilizable"
        assert auto.A_pinv is not None, "sin A_pinv el lazo no manda nada"
        assert auto.src == "nominal"

    def test_the_nominal_seed_does_not_overwrite_a_measured_one(self) -> None:
        from actions import ActionType
        from app_runner import AppRunner
        from tracking import auto_reset

        runner = AppRunner(AppConfig())
        measured = np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.0]], dtype=np.float64)
        auto_reset(runner._tracking_state, src="rls", theta=measured)
        assert runner._tracking_state.auto.ok

        runner._handle_tracking_action(ActionType.TRACKING_START, {})

        assert runner._tracking_state.auto.src != "nominal"
        np.testing.assert_allclose(runner._tracking_state.auto.A, measured[:, :2])
