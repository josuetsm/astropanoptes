"""Contrato de los dos terminos que se miden sin mover la montura a proposito.

El roll sale de los propios plate solves; el rizado, del tracking, que barre la
fase gratis durante una sesion normal. Ninguno de los dos necesita un
procedimiento dedicado, y esa es justamente la razon de que se midan.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pointing.estimators.roll import RollOutcome, estimate_roll, wrap_axis_deg
from pointing.estimators.transmission import RippleOutcome, accept_ripple
from pointing.terms import AXIS_BOUNDS

# ------------------------------------------------------------------------ roll


def test_recovers_a_steady_camera_angle() -> None:
    est = estimate_roll([-3.5, -3.4, -3.6, -3.5, -3.45, -3.55])

    assert est.outcome is RollOutcome.OK
    assert est.deg == pytest.approx(-3.5, abs=0.05)
    assert est.n_obs == 6
    assert est.sigma_deg < 0.2


def test_the_180_degree_branch_is_the_same_camera_angle() -> None:
    """Una camara girada media vuelta encuadra el mismo campo.

    Tratar 179 y 1 como 178 grados de diferencia fue un error real: hay varios
    commits seguidos de "corregir estimacion de roll" en el historial.
    """
    assert wrap_axis_deg(179.0) == pytest.approx(-1.0)
    assert wrap_axis_deg(-179.0) == pytest.approx(1.0)
    assert wrap_axis_deg(90.0) == pytest.approx(-90.0)

    est = estimate_roll([179.5, -0.5, 179.8, 0.2, -0.3, 179.6])

    assert est.outcome is RollOutcome.OK
    assert abs(wrap_axis_deg(est.deg)) < 0.5
    assert est.scatter_deg < 1.0


def test_scatter_too_wide_is_its_own_answer() -> None:
    """La camara no se mueve sola: si los rolls bailan, falla otra cosa.

    Promediar igualmente daria un numero que no describe nada y lo instalaria en
    el modelo como si fuera una medida.
    """
    est = estimate_roll([-3.5, 2.0, -9.0, 4.5, -1.0, 7.5])

    assert est.outcome is RollOutcome.INCONSISTENT
    assert "rotacion de campo" in est.detail
    assert est.scatter_deg > 1.5


def test_too_few_samples_does_not_pretend() -> None:
    est = estimate_roll([-3.5, -3.4])

    assert est.outcome is RollOutcome.NOT_ENOUGH_DATA
    assert est.n_obs == 2


def test_non_finite_samples_are_dropped_not_propagated() -> None:
    est = estimate_roll([-3.5, float("nan"), -3.45, -3.55, -3.5, -3.48, float("inf")])

    assert est.outcome is RollOutcome.OK
    assert est.n_obs == 5


# ---------------------------------------------------------------------- rizado


def _fit(az_sin, az_cos, alt_sin, alt_cos, *, coverage=0.8):
    coeff = np.array([[az_sin, az_cos], [alt_sin, alt_cos]], dtype=np.float64)
    report = {"az_coverage": coverage, "alt_coverage": coverage, "samples": 2400.0}
    return coeff, report


BOUNDS = AXIS_BOUNDS["transmission_error_deg"]


def test_a_well_covered_fit_is_accepted_and_flattened_for_the_term() -> None:
    est = accept_ripple(_fit(0.12, -0.05, 0.004, 0.002), bounds_deg=BOUNDS, n_samples=2400)

    assert est.outcome is RippleOutcome.OK
    assert est.coefficients == (0.12, -0.05, 0.004, 0.002)
    assert est.amplitude_deg[0] == pytest.approx(math.hypot(0.12, 0.05))
    assert est.amplitude_deg[1] == pytest.approx(math.hypot(0.004, 0.002))


def test_without_phase_coverage_it_waits_instead_of_fitting() -> None:
    """El tracking completa la fase solo; no hace falta forzar nada."""
    est = accept_ripple(_fit(0.12, -0.05, 0.004, 0.002, coverage=0.3), bounds_deg=BOUNDS)

    assert est.outcome is RippleOutcome.NOT_ENOUGH_PHASE
    assert "el tracking la completa solo" in est.detail


def test_a_collector_that_could_not_fit_is_not_an_error() -> None:
    est = accept_ripple(None, bounds_deg=BOUNDS, coverage={"min": 0.2})

    assert est.outcome is RippleOutcome.NOT_ENOUGH_PHASE
    assert est.coefficients == (0.0, 0.0, 0.0, 0.0)


def test_the_planetary_axis_refuses_a_cycloidal_sized_ripple() -> None:
    """Un planetario comprado apenas riza; 0.15 deg en altitud es medida mala.

    Con una cota comun para los dos ejes esto pasaria inadvertido, porque en
    azimut ese mismo valor es perfectamente normal.
    """
    est = accept_ripple(_fit(0.12, -0.05, 0.15, 0.02), bounds_deg=BOUNDS)

    assert est.outcome is RippleOutcome.OUT_OF_BOUNDS
    assert "altitud" in est.detail
    # y el valor sigue visible para diagnosticar
    assert est.amplitude_deg[1] > BOUNDS[1]


def test_the_same_amplitude_is_fine_on_the_cycloidal_axis() -> None:
    """La asimetria, vista desde el otro lado: lo que en alt es absurdo, en az no."""
    est = accept_ripple(_fit(0.15, 0.02, 0.004, 0.001), bounds_deg=BOUNDS)

    assert est.outcome is RippleOutcome.OK
    assert est.amplitude_deg[0] > BOUNDS[1], "supera la cota de altitud"
    assert est.amplitude_deg[0] < BOUNDS[0], "pero cabe de sobra en la de azimut"


def test_the_real_collector_round_trips_through_the_gate() -> None:
    """Integracion con el modulo que ya existia, sin reimplementar su ajuste."""
    from pointing.kinematics import MountKinematics
    from ap_types import Axis
    from transmission_error import TransmissionErrorCollector

    kin = MountKinematics()
    period = [kin.transmission_error_period_steps(Axis.AZ),
              kin.transmission_error_period_steps(Axis.ALT)]
    k = [kin.deg_per_step(Axis.AZ), kin.deg_per_step(Axis.ALT)]
    collector = TransmissionErrorCollector(period_steps=period, deg_per_step=k)

    # un rizado sintetico en azimut, barrido en fase como lo haria el tracking
    c_sin = 0.10
    for i in range(4000):
        steps = i * (period[0] / 400.0)
        phase = 2.0 * math.pi * steps / period[0]
        gain_az = 1.0 + (2.0 * math.pi / (period[0] * k[0])) * c_sin * math.cos(phase)
        collector.observe(steps=[steps, steps], gain=[gain_az, 1.0])

    est = accept_ripple(
        collector.fit(), bounds_deg=BOUNDS, coverage=collector.coverage(),
        n_samples=collector.samples,
    )

    assert est.outcome is RippleOutcome.OK, est.detail
    assert est.coefficients[0] == pytest.approx(c_sin, rel=0.15)
