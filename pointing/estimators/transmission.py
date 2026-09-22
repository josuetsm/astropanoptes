"""El rizado del reductor, aprendido del tracking y no de plate solves.

`transmission_error.py` ya resuelve la parte dificil y su docstring explica por
que: ajustar este termino por plate solves exige ocho medidas con cobertura de
fase, cada una de varios minutos bajo cielo real, y "en la practica ese ajuste
casi nunca llega a hacerse". El tracking, en cambio, estima continuamente la
ganancia del reductor y barre la fase gratis durante una sesion normal.

Este modulo no reimplementa ese ajuste: lo envuelve con la decision de aceptarlo
o no, y con las cotas por eje. La asimetria importa aqui mas que en ningun otro
termino -- azimut lleva un cicloidal impreso que riza mucho, altitud un
planetario comprado que apenas riza -- asi que un coeficiente grande en altitud
no es un reductor con caracter, es una medida mala.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["RippleOutcome", "RippleEstimate", "accept_ripple"]


class RippleOutcome(str, Enum):
    OK = "RIPPLE_OK"
    NOT_ENOUGH_PHASE = "RIPPLE_NOT_ENOUGH_PHASE"
    """Falta barrer fase: el primer armonico no esta determinado."""

    OUT_OF_BOUNDS = "RIPPLE_OUT_OF_BOUNDS"
    """La amplitud excede lo que ese reductor puede rizar."""


@dataclass(frozen=True)
class RippleEstimate:
    """Coeficientes (sin, cos) por eje, aplanados como en `terms.py`."""

    outcome: RippleOutcome
    coefficients: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    amplitude_deg: Tuple[float, float] = (0.0, 0.0)
    coverage: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is RippleOutcome.OK


def accept_ripple(
    fit_result,
    *,
    bounds_deg: Tuple[float, float],
    coverage: Optional[Dict[str, float]] = None,
    n_samples: int = 0,
    min_coverage: float = 0.6,
) -> RippleEstimate:
    """Decide si el ajuste del colector es utilizable, y lo pasa a terminos.

    ``fit_result`` es lo que devuelve `TransmissionErrorCollector.fit()`: una
    tupla (coeficientes 2x2, informe) o None cuando falta cobertura. Se separa la
    aceptacion del ajuste para que la regla de admision sea legible y testeable
    sin montar un colector entero.
    """
    cov = dict(coverage or {})
    if fit_result is None:
        return RippleEstimate(
            outcome=RippleOutcome.NOT_ENOUGH_PHASE,
            coverage=cov,
            n_samples=int(n_samples),
            detail=(
                "el colector no pudo ajustar: falta barrer fase del lobulo "
                f"(cobertura minima {cov.get('min', float('nan')):.2f})"
            ),
        )

    coeff, report = fit_result
    coeff = np.asarray(coeff, dtype=np.float64).reshape(2, 2)
    cov = {**report, **cov} if report else cov

    worst_coverage = float(cov.get("min", min(cov.get("az_coverage", 1.0), cov.get("alt_coverage", 1.0))))
    if math.isfinite(worst_coverage) and worst_coverage < float(min_coverage):
        return RippleEstimate(
            outcome=RippleOutcome.NOT_ENOUGH_PHASE,
            coverage=cov,
            n_samples=int(n_samples),
            detail=(
                f"cobertura de fase {worst_coverage:.2f} < {float(min_coverage):.2f}: "
                "sigue observando, el tracking la completa solo"
            ),
        )

    amplitude = (
        float(math.hypot(coeff[0, 0], coeff[0, 1])),
        float(math.hypot(coeff[1, 0], coeff[1, 1])),
    )
    flat = (float(coeff[0, 0]), float(coeff[0, 1]), float(coeff[1, 0]), float(coeff[1, 1]))

    for idx, name in ((0, "azimut"), (1, "altitud")):
        if amplitude[idx] > abs(float(bounds_deg[idx])):
            return RippleEstimate(
                outcome=RippleOutcome.OUT_OF_BOUNDS,
                coefficients=flat,
                amplitude_deg=amplitude,
                coverage=cov,
                n_samples=int(n_samples),
                detail=(
                    f"amplitud en {name} {amplitude[idx]:.3f} deg > "
                    f"{abs(float(bounds_deg[idx])):.3f}: mas de lo que ese reductor puede rizar"
                ),
            )

    return RippleEstimate(
        outcome=RippleOutcome.OK,
        coefficients=flat,
        amplitude_deg=amplitude,
        coverage=cov,
        n_samples=int(n_samples),
        detail=(
            f"amplitud az={amplitude[0]:.3f} deg alt={amplitude[1]:.3f} deg "
            f"con cobertura {worst_coverage:.2f}"
        ),
    )
