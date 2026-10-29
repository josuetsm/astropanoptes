"""La primera muestra tras un reset no puede entrar sin ninguna comprobacion.

Un triplete siempre encaja consigo mismo, asi que un solve de 3 inliers es
barato de falsear contra un catalogo grande y encima presume de rms bajisimo
justo por ser un ajuste exacto. En cielo real se colo uno con rms 0.15 px,
apuntado a grados del verdadero y roll de -170 grados donde todas las muestras
buenas daban -4.3.

Lo que lo dejo pasar fue el reset: ``reset_manual_samples_and_sync`` borraba
tambien el roll de camara, y sin referencia de roll la primera muestra se
aceptaba a ciegas. El roll es mecanico -como esta montada la camara en el
enfocador-, no cambia porque se resetee el modelo de apuntado.
"""
from __future__ import annotations

import numpy as np

from goto import GoToModel


def _model_with_roll(roll_deg: float) -> GoToModel:
    m = GoToModel()
    m.model_roll_deg = float(roll_deg)
    m.model_roll_samples = 4
    return m


class TestRollSurvivesAReset:
    def test_reset_keeps_the_camera_roll(self) -> None:
        m = _model_with_roll(-4.3)
        m.reset_manual_samples_and_sync()
        assert m.model_roll_samples > 0
        assert m.model_roll_deg == -4.3

    def test_reset_still_clears_the_pointing_model(self) -> None:
        """Conservar el roll no puede significar conservar el apuntado."""
        m = _model_with_roll(-4.3)
        m.model_fit_samples = 5
        m.synced = True
        m.reset_manual_samples_and_sync()
        assert m.model_fit_samples == 0
        assert not m.synced


class TestFirstSampleAfterAReset:
    def test_a_wild_roll_is_rejected(self) -> None:
        """El caso real: roll de -169.9 contra -4.3, 165 grados de salto."""
        m = _model_with_roll(-4.3)
        m.reset_manual_samples_and_sync()
        report = m.manual_sample_continuity_report(
            np.array([99.08, 50.12]), roll_deg=-169.93
        )
        assert not report["ok"]
        assert not report["roll_ok"]

    def test_a_consistent_roll_is_accepted(self) -> None:
        m = _model_with_roll(-4.3)
        m.reset_manual_samples_and_sync()
        report = m.manual_sample_continuity_report(
            np.array([98.13, 51.38]), roll_deg=-4.28
        )
        assert report["ok"]
        assert report["roll_ok"]

    def test_the_roll_axis_has_no_direction(self) -> None:
        """172 grados es -8 en un eje sin flecha: 4 grados de -4.3, no 176."""
        m = _model_with_roll(-4.3)
        m.reset_manual_samples_and_sync()
        report = m.manual_sample_continuity_report(
            np.array([96.30, 57.86]), roll_deg=172.06
        )
        assert report["roll_ok"]

    def test_without_any_roll_reference_nothing_is_claimed(self) -> None:
        """Un modelo recien creado no tiene con que comparar, y no inventa."""
        m = GoToModel()
        report = m.manual_sample_continuity_report(
            np.array([98.13, 51.38]), roll_deg=-4.28
        )
        assert report["ok"]
        assert not report["has_reference"]

    def test_a_sample_without_roll_is_not_blocked(self) -> None:
        m = _model_with_roll(-4.3)
        m.reset_manual_samples_and_sync()
        report = m.manual_sample_continuity_report(np.array([98.13, 51.38]))
        assert report["ok"]
