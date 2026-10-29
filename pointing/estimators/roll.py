"""El giro de la camara respecto de los ejes de la montura.

Hoy este angulo existe dos veces: `cfg.camera.roll_deg`, que usa el tracking para
construir su semilla, y `model_roll_deg` dentro del modelo de apuntado. Dos
numeros para un mismo angulo fisico es una forma segura de que se separen. Aqui
es un termino como los demas, con una sola procedencia.

Es un angulo de *eje*, no de vector: la camara girada 180 grados encuadra el
mismo campo, asi que todo se hace modulo 180. Comparar 179 con 1 como si
estuvieran a 178 grados fue un error real que costo varios intentos de arreglo en
el codigo anterior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

import numpy as np

__all__ = ["RollOutcome", "RollEstimate", "estimate_roll", "wrap_axis_deg"]


class RollOutcome(str, Enum):
    OK = "ROLL_OK"
    NOT_ENOUGH_DATA = "ROLL_NOT_ENOUGH_DATA"
    INCONSISTENT = "ROLL_INCONSISTENT"
    """La dispersion es demasiado alta: no es la camara moviendose, es el modelo
    de rotacion de campo el que no cuadra."""


@dataclass(frozen=True)
class RollEstimate:
    outcome: RollOutcome
    deg: float = float("nan")
    sigma_deg: float = float("nan")
    n_obs: int = 0
    scatter_deg: float = float("nan")
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is RollOutcome.OK


def wrap_axis_deg(angle_deg: float) -> float:
    """A [-90, 90): la camara girada media vuelta encuadra lo mismo."""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def estimate_roll(
    roll_samples_deg: Sequence[float],
    *,
    min_samples: int = 5,
    max_scatter_deg: float = 1.5,
) -> RollEstimate:
    """Media circular modulo 180 de los rolls observados.

    Cada muestra ya viene con la rotacion de campo esperada descontada, asi que
    lo que queda es el angulo mecanico de la camara, que no deberia cambiar en
    toda la sesion. Si cambia, lo que falla es la prediccion de rotacion de
    campo, no la camara: por eso una dispersion alta es su propio resultado y no
    se promedia igualmente.
    """
    values = [wrap_axis_deg(v) for v in roll_samples_deg if math.isfinite(float(v))]
    n = len(values)
    if n < int(min_samples):
        return RollEstimate(
            outcome=RollOutcome.NOT_ENOUGH_DATA,
            n_obs=n,
            detail=f"hacen falta {int(min_samples)} muestras de roll, hay {n}",
        )

    # Se dobla el angulo para que el modulo 180 se convierta en un circulo
    # completo, se promedia como vector, y se vuelve a la mitad.
    doubled = np.radians(2.0 * np.asarray(values, dtype=np.float64))
    mean_vector = np.array([np.mean(np.cos(doubled)), np.mean(np.sin(doubled))])
    resultant = float(np.hypot(*mean_vector))
    mean_deg = wrap_axis_deg(0.5 * math.degrees(math.atan2(mean_vector[1], mean_vector[0])))

    if resultant <= 1e-12:
        scatter = 90.0
    else:
        scatter = 0.5 * math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant))))
    sigma = scatter / math.sqrt(n)

    if scatter > float(max_scatter_deg):
        return RollEstimate(
            outcome=RollOutcome.INCONSISTENT,
            deg=mean_deg,
            sigma_deg=sigma,
            n_obs=n,
            scatter_deg=scatter,
            detail=(
                f"dispersion {scatter:.2f} deg (maximo {float(max_scatter_deg):.2f}): "
                "la camara no se mueve sola, asi que lo que no cuadra es la "
                "rotacion de campo predicha"
            ),
        )

    return RollEstimate(
        outcome=RollOutcome.OK,
        deg=mean_deg,
        sigma_deg=sigma,
        n_obs=n,
        scatter_deg=scatter,
        detail=f"{mean_deg:+.2f} +/- {sigma:.2f} deg sobre {n} muestras",
    )
