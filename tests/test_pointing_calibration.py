"""El contrato de la orquestacion: saber que falta antes de intentarlo.

El proceso anterior no permitia preguntar "¿me sirve lo que llevo?". Se lanzaba
la calibracion y el resultado era un codigo que no decia que faltaba, de ahi las
rafagas de autocal cada diez segundos y los ocho reset en los registros.
"""
from __future__ import annotations

import pytest

from pointing.calibration import CalibrationSession
from pointing.estimators.roll import RollOutcome
from pointing.estimators.tilt import GeometryOutcome
from pointing.model import PointingModel, apply_geometry
from pointing.observation import (
    AdmissionReason,
    ObservationKind,
    PointingObservation,
    SolveQuality,
)

TRUTH = dict(tilt_north_deg=1.1, tilt_west_deg=-0.6, npae_deg=0.5, collimation_deg=0.25)


def _obs(az_nominal, alt_nominal, *, roll=-3.5, admitted=True, steps=None):
    """Observacion sintetica coherente con TRUTH."""
    world_az, world_alt = apply_geometry(az_nominal, alt_nominal, **TRUTH)
    kin = PointingModel().kin
    from ap_types import Axis
    steps = steps or (
        az_nominal / kin.deg_per_step(Axis.AZ),
        alt_nominal / kin.deg_per_step(Axis.ALT),
    )
    return PointingObservation(
        t_unix=1789522125.0,
        kind=ObservationKind.PLATE_SOLVE,
        steps=steps,
        load_dir=(1, 1),
        az_deg=world_az,
        alt_deg=world_alt,
        sigma_arcsec=2.0,
        quality=SolveQuality(n_detections=40, n_inliers=20, validation_inliers=8,
                             rms_inliers_arcsec=0.8),
        roll_deg=roll,
        admitted=admitted,
        reason=AdmissionReason.ADMITTED if admitted else AdmissionReason.LOW_CONFIDENCE,
    )


def _well_spread_session() -> CalibrationSession:
    session = CalibrationSession()
    for az, alt in [(10.0, 30.0), (75.0, 55.0), (140.0, 25.0), (200.0, 68.0),
                    (260.0, 40.0), (320.0, 62.0), (45.0, 72.0), (175.0, 45.0)]:
        session.add(_obs(az, alt))
    return session


def test_a_fresh_session_says_nothing_is_measured_yet() -> None:
    status = CalibrationSession().status()

    assert status.n_admitted == 0
    assert not status.ready_to_point
    assert all(not t.measured for t in status.terms)
    geometry = next(t for t in status.terms if t.name == "geometry")
    assert not geometry.measurable
    assert "hacen falta" in geometry.detail and "hay 0" in geometry.detail


def test_the_status_names_the_direction_that_is_missing() -> None:
    """Lo que habria ahorrado las rafagas de autocal: saber que falta.

    Con azimut estrecho, el estado tiene que decir que falta recorrido en azimut,
    no devolver un codigo y dejar al operador adivinando.
    """
    session = CalibrationSession()
    for i in range(8):
        session.add(_obs(100.0 + i * 4.0, 30.0 + i * 5.0))

    geometry = next(t for t in session.status().terms if t.name == "geometry")

    assert not geometry.measurable
    assert "azimut" in geometry.detail


def test_rejected_observations_are_counted_but_not_used() -> None:
    session = _well_spread_session()
    session.add(_obs(90.0, 50.0, admitted=False))

    status = session.status()

    assert status.n_admitted == 8
    assert status.n_rejected == 1


def test_geometry_becomes_measurable_once_the_sky_is_covered() -> None:
    status = _well_spread_session().status()

    geometry = next(t for t in status.terms if t.name == "geometry")
    assert geometry.measurable
    assert not geometry.measured, "medible no es lo mismo que medido"


def test_fitting_installs_the_three_geometric_terms_together() -> None:
    """O entran los tres o no entra ninguno: son un solo ajuste."""
    session = _well_spread_session()

    outcome = session.fit_geometry_terms()

    assert outcome is GeometryOutcome.OK
    terms = session.model.terms
    assert terms["base_tilt"].measured
    assert terms["axis_non_perpendicularity"].measured
    assert terms["collimation"].measured
    assert terms["base_tilt"].value[0] == pytest.approx(TRUTH["tilt_north_deg"], abs=0.05)
    assert terms["collimation"].value[0] == pytest.approx(TRUTH["collimation_deg"], abs=0.05)


def test_a_refused_fit_installs_nothing_at_all() -> None:
    """Quedarse con parte de un ajuste rechazado es recortarlo por otra via."""
    session = CalibrationSession()
    for i in range(8):
        session.add(_obs(100.0 + i * 4.0, 30.0 + i * 5.0))

    outcome = session.fit_geometry_terms()

    assert outcome is GeometryOutcome.ILL_CONDITIONED
    terms = session.model.terms
    assert not terms["base_tilt"].measured
    assert not terms["axis_non_perpendicularity"].measured
    assert not terms["collimation"].measured


def test_after_fitting_the_session_reports_itself_ready_to_point() -> None:
    session = _well_spread_session()
    session.fit_geometry_terms()

    assert session.status().ready_to_point


def test_camera_roll_is_fitted_from_the_same_observations() -> None:
    """No necesita procedimiento propio: sale de los plate solves que ya hay."""
    session = _well_spread_session()

    outcome = session.fit_camera_roll()

    assert outcome is RollOutcome.OK
    assert session.model.terms["camera_roll"].measured
    assert session.model.terms["camera_roll"].value[0] == pytest.approx(-3.5, abs=0.1)


def test_the_two_hardware_terms_say_where_they_come_from() -> None:
    """Backlash y rizado no salen de estas observaciones, y el estado lo dice.

    Cada uno nombra su fuente y el eje en que domina, que es la asimetria de esta
    montura: cicloidal impreso en azimut, planetario comprado en altitud.
    """
    status = _well_spread_session().status()

    backlash = next(t for t in status.terms if t.name == "backlash_steps")
    ripple = next(t for t in status.terms if t.name == "transmission_error_deg")

    assert not backlash.measurable and "inversion de sentido" in backlash.detail
    assert "altitud" in backlash.detail
    assert not ripple.measurable and "tracking" in ripple.detail
    assert "azimut" in ripple.detail


def test_the_report_reads_like_a_checklist() -> None:
    session = _well_spread_session()
    session.fit_geometry_terms()

    text = session.status().describe()

    assert "observaciones: 8 admitidas" in text
    assert "[medido " in text
    assert "geometry" in text and "backlash_steps" in text
