"""Cuanto juego tiene el tren de engranajes, medido en vez de supuesto.

El backlash es el unico termino del modelo que hoy no se estima nunca: vive como
constante de configuracion (`backlash_steps_az=0`, `backlash_steps_alt=10`) y se
materializa como pulsos no contados al invertir sentido. Diez pasos en altitud
son 11 arcsec, y el eje de altitud de esta montura es un planetario comprado,
cuyo juego tipico en la salida son 15-60 arcmin: entre 800 y 3200 pasos. Es
decir, el modelo puede estar perdiendo cerca de un grado en cada inversion y
contandolo como movimiento real.

La asimetria importa y es al reves en cada eje:

    AZ   cicloidal impreso   -> mucho error de transmision, poco juego
    ALT  planetario comprado -> poco error de transmision, mucho juego

asi que este procedimiento es sobre todo para altitud, y el de error de
transmision sobre todo para azimut.

El experimento
--------------
Un tramo que invierte el sentido recorre menos que uno que no lo invierte,
porque los primeros pasos solo consumen la holgura. La diferencia *es* el juego:

    solve -> +N -> solve -> -N -> solve -> -N -> solve
             (a)          (b)           (c)

(b) invierte y (c) no, ambos en el mismo sentido: el deficit de (b) frente a (c)
mide la holgura sin depender de la escala nominal. Se repite para que dos
medidas independientes puedan contradecirse -- y si se contradicen, eso no es
ruido que promediar, es un acoplamiento suelto, y se reporta como tal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from ap_types import Axis

__all__ = ["BacklashLeg", "BacklashOutcome", "BacklashEstimate", "estimate_backlash"]


class BacklashOutcome(str, Enum):
    """Como acabo la medida."""

    OK = "BACKLASH_OK"
    """Las repeticiones concuerdan y el valor cae dentro de lo fisicamente posible."""

    INCONSISTENT = "BACKLASH_INCONSISTENT"
    """Las repeticiones no concuerdan: el juego no es repetible, hay algo suelto."""

    OUT_OF_BOUNDS = "BACKLASH_OUT_OF_BOUNDS"
    """El valor excede lo que el eje puede tener; no se guarda recortado."""

    NOT_ENOUGH_DATA = "BACKLASH_NOT_ENOUGH_DATA"
    """Faltan tramos con inversion, o tramos de referencia sin ella."""


@dataclass(frozen=True)
class BacklashLeg:
    """Un tramo de un solo eje entre dos plate solves aceptados.

    ``reverses`` dice si este tramo cambio el sentido de carga del tren respecto
    del anterior. Es el dato que convierte el experimento en una medida: sin el,
    todos los tramos son iguales y la holgura es invisible.
    """

    axis: Axis
    d_steps: float
    angle_deg: float
    sigma_deg: float
    reverses: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.d_steps) or self.d_steps == 0.0:
            raise ValueError("d_steps must be a non-zero finite number")
        if not math.isfinite(self.angle_deg) or self.angle_deg < 0.0:
            raise ValueError("angle_deg is a magnitude and must be finite and non-negative")
        if not math.isfinite(self.sigma_deg) or self.sigma_deg <= 0.0:
            raise ValueError("sigma_deg must be positive and finite")


@dataclass(frozen=True)
class BacklashEstimate:
    """El juego del eje, con lo necesario para discutirlo."""

    axis: Axis
    outcome: BacklashOutcome
    steps: float
    sigma_steps: float
    arcsec: float
    n_reversals: int
    n_references: int
    per_reversal_steps: List[float] = field(default_factory=list)
    deg_per_step_measured: float = float("nan")
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is BacklashOutcome.OK


def estimate_backlash(
    legs: Sequence[BacklashLeg],
    *,
    axis: Axis,
    cycle_period_steps: float,
    max_periodic_error_deg: float,
    bound_steps: float,
    min_reversals: int = 2,
    consistency_sigmas: float = 3.0,
) -> BacklashEstimate:
    """Mide el juego como deficit de los tramos que invierten sentido.

    La referencia es la escala *medida* en los tramos sin inversion, no la
    nominal: el juego es un deficit relativo al recorrido normal de ese mismo
    tren esa misma noche, asi que cualquier discrepancia de escala se cancela en
    vez de contaminar. Si la escala nominal esta mal, eso lo dice
    `mechanics_check`, que es donde corresponde.

    ``bound_steps`` es por eje: el cicloidal de azimut no puede tener el juego
    que tiene el planetario de altitud, y un valor fuera de rango se rechaza en
    vez de guardarse recortado.
    """
    usable = [leg for leg in legs if leg.axis == axis]
    references = [leg for leg in usable if not leg.reverses]
    reversals = [leg for leg in usable if leg.reverses]

    common = dict(axis=axis, n_reversals=len(reversals), n_references=len(references))

    if len(references) < 1 or len(reversals) < int(min_reversals):
        return BacklashEstimate(
            outcome=BacklashOutcome.NOT_ENOUGH_DATA,
            steps=float("nan"),
            sigma_steps=float("nan"),
            arcsec=float("nan"),
            detail=(
                f"se necesitan >=1 tramo de referencia y >={int(min_reversals)} con inversion; "
                f"hay {len(references)} y {len(reversals)}"
            ),
            **common,
        )

    period = abs(float(cycle_period_steps))
    ref_sigma = _effective_sigma(references, period, max_periodic_error_deg)
    ref_steps = np.array([abs(leg.d_steps) for leg in references], dtype=np.float64)
    ref_angles = np.array([leg.angle_deg for leg in references], dtype=np.float64)
    ref_w = 1.0 / ref_sigma**2
    denom = float(np.sum(ref_w * ref_steps**2))
    if denom <= 0.0:
        return BacklashEstimate(
            outcome=BacklashOutcome.NOT_ENOUGH_DATA,
            steps=float("nan"), sigma_steps=float("nan"), arcsec=float("nan"),
            detail="los tramos de referencia no recorren nada", **common,
        )
    k_measured = float(np.sum(ref_w * ref_steps * ref_angles) / denom)
    k_sigma = float(np.sqrt(1.0 / denom))
    if k_measured <= 0.0:
        return BacklashEstimate(
            outcome=BacklashOutcome.NOT_ENOUGH_DATA,
            steps=float("nan"), sigma_steps=float("nan"), arcsec=float("nan"),
            detail="la escala medida en los tramos de referencia no es positiva", **common,
        )

    rev_sigma = _effective_sigma(reversals, period, max_periodic_error_deg)
    per_reversal: List[float] = []
    per_sigma: List[float] = []
    for leg, sigma in zip(reversals, rev_sigma):
        expected = k_measured * abs(leg.d_steps)
        deficit_deg = expected - leg.angle_deg
        per_reversal.append(deficit_deg / k_measured)
        # la incertidumbre del deficit lleva la del tramo y la de la escala
        var = sigma**2 + (abs(leg.d_steps) * k_sigma) ** 2
        per_sigma.append(float(np.sqrt(var)) / k_measured)

    values = np.array(per_reversal, dtype=np.float64)
    sigmas = np.array(per_sigma, dtype=np.float64)
    w = 1.0 / sigmas**2
    steps = float(np.sum(w * values) / np.sum(w))
    sigma_steps = float(np.sqrt(1.0 / np.sum(w)))

    common = dict(
        **common,
        per_reversal_steps=[float(v) for v in values],
        deg_per_step_measured=k_measured,
    )
    arcsec = abs(steps) * k_measured * 3600.0

    # Repeticiones que no concuerdan no son ruido: un tren con holgura repetible
    # da el mismo deficit cada vez. Si no lo da, hay algo suelto y promediarlo
    # inventaria un numero que no describe nada.
    spread = float(np.max(values) - np.min(values)) if values.size > 1 else 0.0
    spread_sigma = float(np.sqrt(np.sum(sigmas**2)))
    if values.size > 1 and spread > float(consistency_sigmas) * spread_sigma:
        return BacklashEstimate(
            outcome=BacklashOutcome.INCONSISTENT,
            steps=steps, sigma_steps=sigma_steps, arcsec=arcsec,
            detail=(
                f"las {values.size} repeticiones difieren en {spread:.0f} pasos "
                f"(>{float(consistency_sigmas):.0f} sigma): el juego no es repetible, "
                "revisa si hay algo suelto en el tren"
            ),
            **common,
        )

    if steps < -float(consistency_sigmas) * sigma_steps or abs(steps) > abs(float(bound_steps)):
        return BacklashEstimate(
            outcome=BacklashOutcome.OUT_OF_BOUNDS,
            steps=steps, sigma_steps=sigma_steps, arcsec=arcsec,
            detail=(
                f"{steps:.0f} pasos ({arcsec / 60.0:.1f} arcmin) fuera de lo posible "
                f"para este eje (cota {abs(float(bound_steps)):.0f} pasos)"
            ),
            **common,
        )

    steps = max(0.0, steps)
    return BacklashEstimate(
        outcome=BacklashOutcome.OK,
        steps=steps,
        sigma_steps=sigma_steps,
        arcsec=abs(steps) * k_measured * 3600.0,
        detail=(
            f"{steps:.0f} +/- {sigma_steps:.0f} pasos "
            f"({abs(steps) * k_measured * 60.0:.1f} arcmin de juego)"
        ),
        **common,
    )


def _effective_sigma(
    legs: Sequence[BacklashLeg], period: float, max_periodic_error_deg: float
) -> np.ndarray:
    """Incertidumbre del tramo, con el sesgo del rizado de transmision dentro.

    El termino periodico se cancela sobre un ciclo entero y no sobre una
    fraccion, asi que un tramo corto arrastra un sesgo acotado que no se
    promedia. En altitud es pequeno -- el planetario apenas riza -- pero en
    azimut domina, y por eso entra aqui y no como una regla sobre la longitud.
    """
    d_steps = np.array([abs(leg.d_steps) for leg in legs], dtype=np.float64)
    solve = np.array([leg.sigma_deg for leg in legs], dtype=np.float64)
    fractions = d_steps / period if period > 0.0 else np.ones_like(d_steps)
    bias = 2.0 * float(max_periodic_error_deg) * np.abs(np.sin(np.pi * fractions))
    return np.sqrt(solve**2 + bias**2)
