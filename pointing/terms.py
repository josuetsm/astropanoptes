"""Los terminos del modelo de error: con nombre, unidad, cota y procedencia.

El modelo anterior tenia un solo parametro ajustable de verdad -- una matriz J de
2x2 -- y dentro de ella se mezclaban cosas que no son la misma: la reduccion
mecanica (que se conoce exactamente), la inclinacion de la base, la
no-perpendicularidad de los ejes y el rizado del reductor. Ajustarlas juntas las
hacia indistinguibles, y por eso el ajuste no era idempotente y acababa recortado
contra su propia cota.

Aqui cada efecto fisico es un termino con su nombre, en las unidades en que se
mide, con la cota que la mecanica permite y con la procedencia del numero. Un
termino que toca su cota **invalida el ajuste**: no se guarda recortado, porque
un valor en el borde es una medida que fallo, no una medida conservadora.

La asimetria de esta montura
----------------------------
Los dos ejes no comparten mecanica, y sus errores dominantes son opuestos:

    AZ   cicloidal impreso en 3D  -> riza mucho, cierra casi sin juego
    ALT  planetario comprado      -> apenas riza, tiene el juego en el engrane

Por eso las cotas son *por eje* y no globales. Una cota comun obligaria a
elegir entre rechazar el juego real de altitud o admitir en azimut un juego que
ese reductor no puede tener.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

__all__ = [
    "Provenance",
    "Term",
    "TermSet",
    "AXIS_BOUNDS",
    "default_terms",
]


class Provenance(str, Enum):
    """De donde salio el numero. Un termino nunca es solo su valor."""

    NOMINAL = "nominal"
    """De la mecanica o la configuracion. Nunca se midio."""

    PLATE_SOLVE = "plate_solve"
    """De observaciones astrometricas admitidas."""

    TRACKING = "tracking"
    """Del lazo de seguimiento, que barre la fase gratis durante la sesion."""

    OPERATOR = "operator"
    """Introducido o confirmado a mano."""


@dataclass(frozen=True)
class Term:
    """Un efecto fisico del modelo, con todo lo necesario para juzgarlo."""

    name: str
    value: np.ndarray
    sigma: np.ndarray
    unit: str
    bound: np.ndarray
    provenance: Provenance = Provenance.NOMINAL
    n_obs: int = 0
    t_updated: float = 0.0

    def __post_init__(self) -> None:
        value = np.atleast_1d(np.asarray(self.value, dtype=np.float64))
        sigma = np.atleast_1d(np.asarray(self.sigma, dtype=np.float64))
        bound = np.atleast_1d(np.asarray(self.bound, dtype=np.float64))
        if sigma.shape != value.shape:
            raise ValueError(f"{self.name}: sigma {sigma.shape} no cuadra con value {value.shape}")
        if bound.shape != value.shape:
            raise ValueError(f"{self.name}: bound {bound.shape} no cuadra con value {value.shape}")
        if np.any(sigma < 0.0):
            raise ValueError(f"{self.name}: sigma no puede ser negativa")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "bound", bound)

    @property
    def measured(self) -> bool:
        """Si alguien lo midio, o sigue siendo el valor de partida."""
        return self.provenance is not Provenance.NOMINAL and self.n_obs > 0

    @property
    def within_bounds(self) -> bool:
        return bool(np.all(np.isfinite(self.value)) and np.all(np.abs(self.value) <= self.bound))

    @property
    def at_bound(self) -> np.ndarray:
        """Que componentes estan pegadas al borde de lo fisicamente posible."""
        return np.abs(self.value) >= self.bound * (1.0 - 1e-9)

    def updated(
        self,
        value,
        *,
        sigma,
        provenance: Provenance,
        n_obs: int,
        t_unix: Optional[float] = None,
    ) -> "Term":
        """Un termino nuevo con el valor medido, o un error si no cabe.

        Rechazar en vez de recortar es deliberado. El modelo anterior proyectaba
        el ajuste dentro de la envolvente mecanica y lo guardaba, asi que un
        ajuste malo se volvia indistinguible de uno conservador y se arrastraba
        toda la noche.
        """
        candidate = replace(
            self,
            value=np.atleast_1d(np.asarray(value, dtype=np.float64)),
            sigma=np.atleast_1d(np.asarray(sigma, dtype=np.float64)),
            provenance=provenance,
            n_obs=int(n_obs),
            t_updated=float(time.time() if t_unix is None else t_unix),
        )
        if not candidate.within_bounds:
            raise ValueError(
                f"{self.name}: {np.asarray(candidate.value).tolist()} {self.unit} "
                f"excede la cota {np.asarray(self.bound).tolist()}; "
                "una medida fuera de lo posible se rechaza, no se recorta"
            )
        return candidate

    def describe(self) -> str:
        vals = ", ".join(f"{v:+.6g}" for v in np.atleast_1d(self.value))
        sigs = ", ".join(f"{v:.3g}" for v in np.atleast_1d(self.sigma))
        origin = self.provenance.value if self.measured else "sin medir"
        return f"{self.name}=[{vals}] +/-[{sigs}] {self.unit} ({origin}, n={self.n_obs})"


# Cotas por eje, en el orden (AZ, ALT). Cada una dice que es fisicamente posible
# para *este* reductor, no para un eje generico.
AXIS_BOUNDS: Dict[str, Tuple[float, float]] = {
    # El cicloidal impreso cierra casi sin juego; el planetario comprado trae
    # entre 15 y 60 arcmin en la salida, que a 0.00031 deg/paso son 800-3200
    # microsteps. 4000 deja margen sin admitir un disparate.
    "backlash_steps": (200.0, 4000.0),
    # Y al reves para el rizado: el cicloidal impreso es el que riza.
    "transmission_error_deg": (0.25, 0.03),
}


def default_terms() -> "TermSet":
    """El modelo de partida: mecanica conocida, desviaciones sin medir.

    Todo arranca en cero con procedencia NOMINAL, que es la verdad: al abrir la
    app nadie ha medido nada todavia. Un termino en cero y sin medir no estropea
    el apuntado -- simplemente no lo corrige -- y se distingue de uno medido que
    resulto ser cero.
    """
    bl_az, bl_alt = AXIS_BOUNDS["backlash_steps"]
    te_az, te_alt = AXIS_BOUNDS["transmission_error_deg"]
    return TermSet(
        {
            "base_tilt": Term(
                name="base_tilt",
                value=np.zeros(2),
                sigma=np.zeros(2),
                unit="deg",
                bound=np.array([2.0, 2.0]),
            ),
            "axis_non_perpendicularity": Term(
                name="axis_non_perpendicularity",
                value=np.zeros(1),
                sigma=np.zeros(1),
                unit="deg",
                bound=np.array([2.0]),
            ),
            "collimation": Term(
                name="collimation",
                value=np.zeros(1),
                sigma=np.zeros(1),
                unit="deg",
                bound=np.array([1.0]),
            ),
            "camera_roll": Term(
                name="camera_roll",
                value=np.zeros(1),
                sigma=np.zeros(1),
                unit="deg",
                bound=np.array([180.0]),
            ),
            "backlash_steps": Term(
                name="backlash_steps",
                value=np.zeros(2),
                sigma=np.zeros(2),
                unit="steps",
                bound=np.array([bl_az, bl_alt]),
            ),
            "transmission_error_deg": Term(
                name="transmission_error_deg",
                # (sin, cos) por eje, aplanado como (az_sin, az_cos, alt_sin, alt_cos)
                value=np.zeros(4),
                sigma=np.zeros(4),
                unit="deg",
                bound=np.array([te_az, te_az, te_alt, te_alt]),
            ),
        }
    )


@dataclass
class TermSet:
    """Los terminos del modelo, accesibles por nombre."""

    terms: Dict[str, Term] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Term:
        return self.terms[name]

    def __contains__(self, name: str) -> bool:
        return name in self.terms

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms.values())

    def set(self, term: Term) -> None:
        self.terms[term.name] = term

    @property
    def all_within_bounds(self) -> bool:
        return all(term.within_bounds for term in self.terms.values())

    def measured_names(self) -> Tuple[str, ...]:
        return tuple(sorted(name for name, term in self.terms.items() if term.measured))

    def unmeasured_names(self) -> Tuple[str, ...]:
        return tuple(sorted(name for name, term in self.terms.items() if not term.measured))

    def describe(self) -> str:
        return "\n".join(self.terms[name].describe() for name in sorted(self.terms))
