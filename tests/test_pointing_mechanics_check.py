"""El contrato de la verificacion de mecanica.

La pregunta que responde este estimador es discreta: que reductor hay montado.
Estos tests fijan las tres respuestas que importan -- confirma, contradice, o se
niega a decidir -- y sobre todo la tercera, porque responder mal a esta pregunta
es lo que produjo el 78% de fallos de ajuste del modelo anterior.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from ap_types import Axis
from pointing.estimators.mechanics_check import (
    AxisMove,
    MechanicsOutcome,
    axis_angle_between_positions_deg,
    verify_reduction,
)
from pointing.kinematics import MountKinematics

MICROSTEPS = 64
MOTOR_FULL_STEPS = 200
CYCLE = float(MOTOR_FULL_STEPS * MICROSTEPS)  # 12800, una vuelta de motor
CANDIDATES = (45.0, 90.5)


def _deg_per_step(ratio: float) -> float:
    return 360.0 / (MOTOR_FULL_STEPS * MICROSTEPS * ratio)


def _moves(true_ratio: float, *, d_steps: float, n: int, sigma_deg: float = 0.0006):
    """Movimientos sinteticos sin sesgo ciclolidal (fase que cancela)."""
    k = _deg_per_step(true_ratio)
    return [
        AxisMove(axis=Axis.ALT, d_steps=d_steps, angle_deg=k * d_steps, sigma_deg=sigma_deg)
        for _ in range(n)
    ]


def _verify(moves, *, configured: float):
    return verify_reduction(
        moves,
        axis=Axis.ALT,
        configured_ratio=configured,
        candidate_ratios=CANDIDATES,
        microsteps=MICROSTEPS,
        motor_full_steps_per_rev=MOTOR_FULL_STEPS,
        cycle_period_steps=CYCLE,
    )


def test_full_cycle_move_identifies_45_to_1_decisively() -> None:
    verdict = _verify(_moves(45.0, d_steps=CYCLE, n=2), configured=45.0)

    assert verdict.outcome is MechanicsOutcome.CONFIRMED
    assert verdict.accepted_ratio == 45.0
    assert verdict.ok
    # la separacion entre hipotesis debe ser aplastante, no ajustada
    assert verdict.chi2[90.5] - verdict.chi2[45.0] > 1e5
    assert verdict.measured_deg_per_step == pytest.approx(_deg_per_step(45.0), rel=1e-6)


def test_full_cycle_move_identifies_90_5_to_1_decisively() -> None:
    verdict = _verify(_moves(90.5, d_steps=CYCLE, n=2), configured=90.5)

    assert verdict.outcome is MechanicsOutcome.CONFIRMED
    assert verdict.accepted_ratio == 90.5
    assert verdict.chi2[45.0] - verdict.chi2[90.5] > 1e5


def test_hardware_that_disagrees_with_config_is_contradicted_not_absorbed() -> None:
    """El caso real: se cambio el reductor y la constante se quedo atras.

    El modelo viejo habria ajustado J hacia el valor medido y lo habria recortado
    al 10% de la mecanica declarada, quedandose a medio camino entre las dos
    verdades. Aqui tiene que decir cual es, y que no es la configurada.
    """
    verdict = _verify(_moves(90.5, d_steps=CYCLE, n=2), configured=45.0)

    assert verdict.outcome is MechanicsOutcome.CONTRADICTED
    assert verdict.accepted_ratio == 90.5
    assert not verdict.ok
    assert "45" in verdict.detail


def test_short_move_declines_to_decide_instead_of_guessing() -> None:
    """Regresion del error real: los movimientos de calibracion eran del 14%.

    A esa fraccion de ciclo el error de transmision no se cancela, y lo que queda
    es un sesgo acotado que no se promedia. Con pocos movimientos asi, la
    respuesta honesta es que no se puede decidir.
    """
    short = 0.14 * CYCLE
    verdict = _verify(_moves(90.5, d_steps=short, n=2), configured=90.5)

    assert verdict.outcome is MechanicsOutcome.UNDETERMINED
    assert verdict.accepted_ratio is None
    assert verdict.min_cycle_fraction == pytest.approx(0.14, abs=1e-9)
    assert "ciclo completo" in verdict.detail


def test_enough_short_moves_do_accumulate_to_a_decision() -> None:
    """Muchos movimientos cortos a fases distintas si separan las hipotesis.

    El sesgo entra como incertidumbre por movimiento, asi que la evidencia se
    acumula: la puerta es la separacion en chi2, no una regla sobre la longitud.
    """
    verdict = _verify(_moves(90.5, d_steps=0.14 * CYCLE, n=10), configured=90.5)

    assert verdict.outcome is MechanicsOutcome.CONFIRMED
    assert verdict.accepted_ratio == 90.5


def test_a_reduction_matching_neither_candidate_is_reported_not_forced() -> None:
    """Un tercer reductor no debe caer en el candidato mas cercano."""
    moves = _moves(67.0, d_steps=CYCLE, n=3, sigma_deg=0.0006)
    verdict = _verify(moves, configured=45.0)

    assert verdict.outcome is MechanicsOutcome.UNMODELLED
    assert verdict.accepted_ratio is None
    assert verdict.measured_deg_per_step == pytest.approx(_deg_per_step(67.0), rel=1e-3)
    assert verdict.measured_ratio == pytest.approx(67.0, rel=1e-3)


def test_too_few_moves_is_its_own_answer() -> None:
    verdict = _verify(_moves(45.0, d_steps=CYCLE, n=1), configured=45.0)

    assert verdict.outcome is MechanicsOutcome.NOT_ENOUGH_DATA
    assert verdict.n_moves == 1


def test_altitude_rotation_is_read_as_great_circle_separation() -> None:
    """Girar en altitud mueve el eje optico por un circulo maximo, exacto."""
    angle = axis_angle_between_positions_deg(
        axis=Axis.ALT, az0_deg=120.0, alt0_deg=30.0, az1_deg=120.0, alt1_deg=34.0
    )
    assert angle == pytest.approx(4.0, abs=1e-9)


def test_azimuth_rotation_undoes_the_cosine_compression() -> None:
    """Girar en azimut a 60 deg de altura mueve el cielo la mitad del angulo.

    Sin deshacer ese cos(alt), un giro de azimut a altura alta se leeria como la
    mitad de los grados y elegiria el reductor equivocado por un factor 2 -- que
    es exactamente la confusion que este estimador existe para evitar.
    """
    angle = axis_angle_between_positions_deg(
        axis=Axis.AZ, az0_deg=100.0, alt0_deg=60.0, az1_deg=108.0, alt1_deg=60.0
    )
    assert angle == pytest.approx(8.0, abs=1e-6)

    naive_separation = 2.0 * math.degrees(
        math.asin(math.sin(math.radians(4.0)) * math.cos(math.radians(60.0)))
    )
    assert naive_separation == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------------------
# Golden: las muestras reales de Josue, congeladas en tests/data
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "data" / "goto_manual_samples_2026-08.csv"


def _alt_moves_from_fixture(day: str, *, single_axis: bool):
    """Movimientos de altitud de una noche, sacados de muestras consecutivas.

    ``single_axis`` separa los pares en que solo se movio altitud de los que
    movieron los dos ejes a la vez. Esa distincion no es cosmetica: el angulo de
    un eje solo se lee de dos posiciones del cielo si el otro eje no se movio.
    """
    rows = list(csv.DictReader(FIXTURE.open()))
    out = []
    prev = None
    for row in rows:
        try:
            cur = (
                float(row["sample_idx"]),
                float(row["steps_az"]),
                float(row["steps_alt"]),
                float(row["az_deg"]),
                float(row["alt_deg"]),
                (row.get("ts_utc") or "")[:10],
            )
        except (KeyError, ValueError):
            prev = None
            continue
        if prev is not None and cur[0] > prev[0] and cur[5] == day:
            d_az, d_alt = cur[1] - prev[1], cur[2] - prev[2]
            pure = abs(d_az) < 100.0
            if abs(d_alt) >= 1500.0 and pure == single_axis:
                out.append(
                    AxisMove(
                        axis=Axis.ALT,
                        d_steps=d_alt,
                        angle_deg=axis_angle_between_positions_deg(
                            axis=Axis.ALT,
                            az0_deg=prev[3], alt0_deg=prev[4],
                            az1_deg=cur[3], alt1_deg=cur[4],
                        ),
                        sigma_deg=0.01,
                    )
                )
        prev = cur
    return out


def test_real_single_axis_moves_confirm_the_reducer_of_that_night() -> None:
    """Los dos movimientos de altitud pura del 1 de septiembre dicen 45:1.

    Esa noche la configuracion tambien decia 45:1, y el verificador lo confirma
    con los datos crudos del usuario, sin ajustar nada.
    """
    moves = _alt_moves_from_fixture("2026-09-01", single_axis=True)
    assert len(moves) == 2, "la fixture trae exactamente dos movimientos de eje puro esa noche"

    confirmed = _verify(moves, configured=45.0)
    assert confirmed.outcome is MechanicsOutcome.CONFIRMED, confirmed.detail
    assert confirmed.accepted_ratio == 45.0
    assert confirmed.measured_deg_per_step == pytest.approx(0.000649, rel=0.02)

    # y contra la constante de hoy, los mismos datos la contradicen: el reductor
    # de altitud se cambio despues de esa noche
    assert _verify(moves, configured=90.5).outcome is MechanicsOutcome.CONTRADICTED


def test_one_move_is_not_enough_however_good_it_looks() -> None:
    """La noche del 16 solo dejo un movimiento de altitud puro."""
    moves = _alt_moves_from_fixture("2026-09-16", single_axis=True)
    assert len(moves) == 1

    verdict = _verify(moves, configured=90.5)
    assert verdict.outcome is MechanicsOutcome.NOT_ENOUGH_DATA
    assert verdict.accepted_ratio is None


def test_mixed_axis_moves_are_refused_instead_of_answered_wrong() -> None:
    """El hallazgo incomodo: en 7 noches casi ningun movimiento fue de un solo eje.

    Cuando los dos ejes se mueven a la vez, la separacion entre las dos
    posiciones del cielo no es el angulo del eje de altitud, y el numero que sale
    no corresponde a ningun reductor. El verificador tiene que negarse, que es lo
    que el proceso anterior nunca hizo: ajustaba J contra estos mismos pares y se
    quedaba con el resultado recortado a la cota.
    """
    mixed = _alt_moves_from_fixture("2026-09-01", single_axis=False)
    assert len(mixed) >= 5, "la mayoria de los pares reales movieron los dos ejes"

    verdict = _verify(mixed, configured=45.0)
    assert verdict.outcome is MechanicsOutcome.UNMODELLED
    assert verdict.accepted_ratio is None
    assert math.isfinite(verdict.measured_deg_per_step)


def test_kinematics_cycle_period_is_one_motor_revolution() -> None:
    """El periodo que usa el verificador sale de la cinematica, no de una constante suelta."""
    kin = MountKinematics()
    assert kin.transmission_error_period_steps(Axis.AZ) == pytest.approx(CYCLE)
    assert kin.transmission_error_period_steps(Axis.ALT) == pytest.approx(CYCLE)
