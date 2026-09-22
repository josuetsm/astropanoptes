"""Que se puede medir ahora, y que falta para poder medir lo demas.

Esta es la pieza que habria ahorrado 109 corridas a ciegas. El proceso anterior
no tenia forma de preguntar "¿me sirve lo que llevo?": se lanzaba la calibracion,
se esperaba un minuto, y el resultado era `ERR_SAMPLE_CONTINUITY` o
`MODEL_OUTSIDE_MECHANICAL_LIMITS` sin decir que le faltaba. De ahi las rafagas de
autocal cada diez segundos y los ocho `reset` en los registros.

Aqui el estado de la calibracion es una pregunta barata que se puede hacer en
cualquier momento, y la respuesta dice que direccion falta -- "hace falta una
muestra por encima de 60 grados de altura" -- en vez de un codigo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ap_types import Axis
from pointing.estimators.roll import RollOutcome, estimate_roll
from pointing.estimators.tilt import (
    GeometryOutcome,
    GeometrySample,
    fit_geometry,
)
from pointing.kinematics import MountKinematics
from pointing.model import PointingModel
from pointing.observation import ObservationKind, PointingObservation
from pointing.terms import Provenance

__all__ = ["TermStatus", "CalibrationStatus", "CalibrationSession"]


@dataclass(frozen=True)
class TermStatus:
    """Si un termino se puede medir con lo que hay, y si no, que falta."""

    name: str
    measurable: bool
    measured: bool
    detail: str = ""

    def describe(self) -> str:
        mark = "medido" if self.measured else ("listo" if self.measurable else "falta")
        return f"[{mark:7}] {self.name}: {self.detail}" if self.detail else f"[{mark:7}] {self.name}"


@dataclass(frozen=True)
class CalibrationStatus:
    """Retrato de la calibracion en un instante."""

    n_admitted: int
    n_rejected: int
    terms: Tuple[TermStatus, ...] = ()

    @property
    def ready_to_point(self) -> bool:
        """Con la geometria medida ya se apunta; el resto refina."""
        return any(t.name == "geometry" and t.measured for t in self.terms)

    def describe(self) -> str:
        head = f"observaciones: {self.n_admitted} admitidas, {self.n_rejected} rechazadas"
        return head + "\n" + "\n".join(t.describe() for t in self.terms)


@dataclass
class CalibrationSession:
    """Ata las observaciones de la noche a los terminos del modelo.

    No mueve nada ni habla con la camara: recibe observaciones ya admitidas y
    decide que se puede estimar con ellas. Separarlo de la adquisicion es lo que
    permite probarlo sin montura y razonar sobre el proceso sin leer el worker.
    """

    model: PointingModel = field(default_factory=PointingModel)
    observations: List[PointingObservation] = field(default_factory=list)

    # ----------------------------------------------------------------- entrada

    def add(self, observation: PointingObservation) -> None:
        self.observations.append(observation)

    def extend(self, observations: Sequence[PointingObservation]) -> None:
        self.observations.extend(observations)

    @property
    def admitted(self) -> List[PointingObservation]:
        return [o for o in self.observations if o.admitted]

    # ------------------------------------------------------------------ estado

    def status(self) -> CalibrationStatus:
        """Que se puede medir ahora mismo, y que le falta a lo que no."""
        admitted = self.admitted
        rejected = len(self.observations) - len(admitted)

        geometry_probe = fit_geometry(self._geometry_samples(admitted))
        geometry_ready = geometry_probe.outcome not in (
            GeometryOutcome.NOT_ENOUGH_DATA,
            GeometryOutcome.ILL_CONDITIONED,
        )
        roll_probe = estimate_roll([o.roll_deg for o in admitted if o.roll_deg is not None])

        terms = (
            TermStatus(
                name="geometry",
                measurable=geometry_ready,
                measured=self.model.terms["base_tilt"].measured,
                detail=geometry_probe.detail or self._geometry_hint(admitted),
            ),
            TermStatus(
                name="camera_roll",
                measurable=roll_probe.outcome is not RollOutcome.NOT_ENOUGH_DATA,
                measured=self.model.terms["camera_roll"].measured,
                detail=roll_probe.detail,
            ),
            TermStatus(
                name="backlash_steps",
                measurable=False,
                measured=self.model.terms["backlash_steps"].measured,
                detail=(
                    "necesita su propio experimento de inversion de sentido; "
                    "en altitud es el termino dominante"
                ),
            ),
            TermStatus(
                name="transmission_error_deg",
                measurable=False,
                measured=self.model.terms["transmission_error_deg"].measured,
                detail="lo aprende el tracking mientras observas; en azimut es el dominante",
            ),
        )
        return CalibrationStatus(n_admitted=len(admitted), n_rejected=rejected, terms=terms)

    # ------------------------------------------------------------------ ajustes

    def fit_geometry_terms(self) -> GeometryOutcome:
        """Ajusta inclinacion, no-perpendicularidad y colimacion, si se puede.

        Instala los tres a la vez o ninguno: son un solo ajuste, y quedarse con
        parte de un resultado rechazado es como recortarlo contra la cota.
        """
        result = fit_geometry(self._geometry_samples(self.admitted))
        if not result.ok:
            return result.outcome

        terms = self.model.terms
        terms.set(terms["base_tilt"].updated(
            list(result.base_tilt_deg),
            sigma=[result.sigma["tilt_north"], result.sigma["tilt_west"]],
            provenance=Provenance.PLATE_SOLVE,
            n_obs=result.n_samples,
        ))
        terms.set(terms["axis_non_perpendicularity"].updated(
            [result.npae_deg], sigma=[result.sigma["npae"]],
            provenance=Provenance.PLATE_SOLVE, n_obs=result.n_samples,
        ))
        terms.set(terms["collimation"].updated(
            [result.collimation_deg], sigma=[result.sigma["collimation"]],
            provenance=Provenance.PLATE_SOLVE, n_obs=result.n_samples,
        ))
        return result.outcome

    def fit_camera_roll(self) -> RollOutcome:
        rolls = [o.roll_deg for o in self.admitted if o.roll_deg is not None]
        result = estimate_roll(rolls)
        if not result.ok:
            return result.outcome
        terms = self.model.terms
        terms.set(terms["camera_roll"].updated(
            [result.deg], sigma=[result.sigma_deg],
            provenance=Provenance.PLATE_SOLVE, n_obs=result.n_obs,
        ))
        return result.outcome

    # ------------------------------------------------------------------ interno

    def _geometry_samples(self, admitted: Sequence[PointingObservation]) -> List[GeometrySample]:
        """Observaciones reducidas a nominal-vs-observado, en grados de eje."""
        out: List[GeometrySample] = []
        kin = self.model.kin
        reference = self.model.reference
        for obs in admitted:
            if obs.kind is not ObservationKind.PLATE_SOLVE:
                continue
            if reference is None:
                # Sin referencia se usa la primera observacion como origen: los
                # offsets de indice son parametros del ajuste, asi que da igual
                # donde se ponga el cero.
                reference_steps = admitted[0].steps
                base_az, base_alt = admitted[0].az_deg, admitted[0].alt_deg
            else:
                reference_steps = reference.steps
                base_az, base_alt = reference.az_axis_deg, reference.alt_axis_deg
            nominal_az = base_az + kin.deg_per_step(Axis.AZ) * (obs.steps[0] - reference_steps[0])
            nominal_alt = base_alt + kin.deg_per_step(Axis.ALT) * (obs.steps[1] - reference_steps[1])
            out.append(
                GeometrySample(
                    az_nominal_deg=nominal_az % 360.0,
                    alt_nominal_deg=nominal_alt,
                    az_obs_deg=obs.az_deg,
                    alt_obs_deg=obs.alt_deg,
                    sigma_deg=max(1e-6, obs.sigma_arcsec / 3600.0),
                )
            )
        return out

    @staticmethod
    def _geometry_hint(admitted: Sequence[PointingObservation]) -> str:
        if not admitted:
            return "sin observaciones admitidas todavia"
        alts = [o.alt_deg for o in admitted]
        return f"{len(admitted)} observaciones, alturas {min(alts):.0f}-{max(alts):.0f} deg"
