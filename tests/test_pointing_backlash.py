"""El contrato de la medida de backlash.

El eje de altitud de esta montura es un planetario comprado y tiene juego de
verdad; el de azimut es un cicloidal impreso y casi no tiene. El modelo anterior
trataba el backlash como una constante de configuracion igual para ambos
(`backlash_steps_alt=10`, que son 11 arcsec) y no lo media nunca. Estos tests
fijan que ahora se mide, que la cota es por eje, y que una medida que no se
repite se reporta en vez de promediarse.
"""
from __future__ import annotations

import pytest

from ap_types import Axis
from pointing.estimators.backlash import (
    BacklashLeg,
    BacklashOutcome,
    estimate_backlash,
)
from pointing.kinematics import MountKinematics

KIN = MountKinematics()
K_ALT = abs(KIN.deg_per_step(Axis.ALT))       # 0.00031077 deg/paso
K_AZ = abs(KIN.deg_per_step(Axis.AZ))         # 0.000625 deg/paso
CYCLE = KIN.transmission_error_period_steps(Axis.ALT)  # 12800

# Cotas por eje: el planetario admite grados de juego, el cicloidal no.
BOUND_ALT_STEPS = 4000.0   # ~1.24 deg
BOUND_AZ_STEPS = 200.0     # ~0.125 deg
SIGMA = 0.003              # ~11 arcsec de un plate solve aceptado


def _legs(
    *,
    axis: Axis,
    k: float,
    n_steps: float,
    backlash_steps: float,
    n_references: int = 2,
    reversal_backlashes=None,
):
    """Tramos sinteticos: los de referencia recorren todo, los de inversion menos."""
    legs = [
        BacklashLeg(
            axis=axis, d_steps=n_steps, angle_deg=k * n_steps, sigma_deg=SIGMA, reverses=False
        )
        for _ in range(n_references)
    ]
    values = reversal_backlashes if reversal_backlashes is not None else [backlash_steps] * 2
    for slack in values:
        legs.append(
            BacklashLeg(
                axis=axis,
                d_steps=-n_steps,
                angle_deg=k * (n_steps - slack),
                sigma_deg=SIGMA,
                reverses=True,
            )
        )
    return legs


def _estimate(legs, *, axis=Axis.ALT, bound=BOUND_ALT_STEPS, max_periodic=0.02):
    return estimate_backlash(
        legs,
        axis=axis,
        cycle_period_steps=CYCLE,
        max_periodic_error_deg=max_periodic,
        bound_steps=bound,
    )


def test_recovers_the_slack_of_a_purchased_planetary_gearbox() -> None:
    """30 arcmin de juego en altitud son ~1600 pasos, y hay que verlos."""
    legs = _legs(axis=Axis.ALT, k=K_ALT, n_steps=CYCLE, backlash_steps=1600.0)

    est = _estimate(legs)

    assert est.outcome is BacklashOutcome.OK
    assert est.steps == pytest.approx(1600.0, abs=25.0)
    assert est.arcsec == pytest.approx(1600.0 * K_ALT * 3600.0, rel=0.05)
    assert est.n_reversals == 2 and est.n_references == 2


def test_the_default_of_ten_steps_is_nowhere_near_what_it_measures() -> None:
    """Regresion del valor que estaba en config: 10 pasos son 11 arcsec.

    Si el eje tiene un juego realista, la medida tiene que salir dos ordenes de
    magnitud por encima del default, no cerca de el.
    """
    legs = _legs(axis=Axis.ALT, k=K_ALT, n_steps=CYCLE, backlash_steps=1600.0)

    est = _estimate(legs)

    assert est.steps > 100.0 * 10.0
    assert est.arcsec > 30.0 * 60.0 * 0.5  # bastante mas que los 11 arcsec del default


def test_measurement_does_not_lean_on_the_nominal_scale() -> None:
    """El juego es un deficit relativo, asi que un error de escala se cancela.

    Si la escala real difiere un 10% de la nominal, la holgura medida tiene que
    seguir saliendo correcta: distinguir escala de juego es trabajo de
    mechanics_check, no de aqui.
    """
    legs = _legs(axis=Axis.ALT, k=K_ALT * 1.10, n_steps=CYCLE, backlash_steps=1200.0)

    est = _estimate(legs)

    assert est.outcome is BacklashOutcome.OK
    assert est.steps == pytest.approx(1200.0, abs=25.0)
    assert est.deg_per_step_measured == pytest.approx(K_ALT * 1.10, rel=1e-6)


def test_repeats_that_disagree_are_reported_not_averaged() -> None:
    """Un juego que no se repite no es ruido: es algo suelto en el tren."""
    legs = _legs(
        axis=Axis.ALT, k=K_ALT, n_steps=CYCLE, backlash_steps=0.0,
        reversal_backlashes=[900.0, 2500.0],
    )

    est = _estimate(legs)

    assert est.outcome is BacklashOutcome.INCONSISTENT
    assert "no es repetible" in est.detail
    # las dos medidas siguen visibles para poder mirarlas
    assert len(est.per_reversal_steps) == 2
    assert max(est.per_reversal_steps) - min(est.per_reversal_steps) > 1000.0


def test_a_cycloidal_axis_rejects_planetary_sized_slack() -> None:
    """La cota es por eje. El cicloidal impreso de azimut no puede tener 1600 pasos.

    Un valor asi en ese eje significa que la medida esta mal, no que el eje tenga
    ese juego, y guardarlo recortado a la cota seria guardar una mentira.
    """
    legs = _legs(axis=Axis.AZ, k=K_AZ, n_steps=CYCLE, backlash_steps=1600.0)

    est = _estimate(legs, axis=Axis.AZ, bound=BOUND_AZ_STEPS, max_periodic=0.25)

    assert est.outcome is BacklashOutcome.OUT_OF_BOUNDS
    assert "cota" in est.detail


def test_small_slack_on_the_cycloidal_axis_is_accepted() -> None:
    """El cicloidal si tiene algo de juego, solo que poco."""
    legs = _legs(axis=Axis.AZ, k=K_AZ, n_steps=CYCLE, backlash_steps=60.0)

    est = _estimate(legs, axis=Axis.AZ, bound=BOUND_AZ_STEPS, max_periodic=0.25)

    assert est.outcome is BacklashOutcome.OK
    assert est.steps == pytest.approx(60.0, abs=30.0)


def test_without_a_reference_leg_there_is_nothing_to_compare_against() -> None:
    legs = [
        BacklashLeg(axis=Axis.ALT, d_steps=-CYCLE, angle_deg=K_ALT * (CYCLE - 1600.0),
                    sigma_deg=SIGMA, reverses=True)
        for _ in range(2)
    ]

    est = _estimate(legs)

    assert est.outcome is BacklashOutcome.NOT_ENOUGH_DATA
    assert est.n_references == 0


def test_one_reversal_is_not_a_repeatable_measurement() -> None:
    legs = _legs(
        axis=Axis.ALT, k=K_ALT, n_steps=CYCLE, backlash_steps=1600.0,
        reversal_backlashes=[1600.0],
    )

    est = _estimate(legs)

    assert est.outcome is BacklashOutcome.NOT_ENOUGH_DATA
    assert est.n_reversals == 1


def test_no_slack_at_all_is_a_valid_answer() -> None:
    """Un eje sin juego tiene que poder decir cero, no un residuo positivo."""
    legs = _legs(axis=Axis.AZ, k=K_AZ, n_steps=CYCLE, backlash_steps=0.0)

    est = _estimate(legs, axis=Axis.AZ, bound=BOUND_AZ_STEPS, max_periodic=0.25)

    assert est.outcome is BacklashOutcome.OK
    assert est.steps == pytest.approx(0.0, abs=30.0)
