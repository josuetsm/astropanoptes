"""Una observacion de apuntado, y la puerta que decide si sirve.

Este es el cambio con mas efecto medido de todo el rediseño, y no estaba en el
diagnostico inicial. Sobre 109 corridas reales de calibracion:

    aceptadas            43   mediana 5 inliers,  2 de validacion, 0.39 deg del prior
    rechazadas por geom. 42   mediana 3 inliers,  0 de validacion, 2.16 deg del prior
    fallo el solver      24   mediana 3 inliers,  0 de validacion

Las rechazadas no tenian ninguna confirmacion independiente: tres inliers son
los tres vertices del triplete semilla, que encaja consigo mismo por
construccion. La guardia geometrica estaba haciendo bien su trabajo, pero lo
hacia por el motivo equivocado -- decidia por movimiento lo que se decide por
astrometria. Y al reves: 14 de las 43 aceptadas tenian un inlier de validacion o
ninguno, y no deberian haber entrado.

De ahi que la admision sea astrometrica primero y geometrica despues, que las dos
razones se distingan en el registro, y que una observacion rechazada tambien se
guarde: son los datos con los que se afina la puerta.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "ObservationKind",
    "SolveQuality",
    "PointingObservation",
    "Admission",
    "AdmissionReason",
    "admit",
    "ObservationStore",
]

SCHEMA_VERSION = 1


class ObservationKind(str, Enum):
    PLATE_SOLVE = "plate_solve"
    """Posicion absoluta del cielo, por astrometria."""

    DRIFT = "drift"
    """Apuntado deducido de la deriva sideral, sin catalogo ni modelo previo."""


class AdmissionReason(str, Enum):
    """Por que entro o no entro. Cada una con su payload numerico."""

    ADMITTED = "OBS_ADMITTED"

    LOW_CONFIDENCE = "OBS_LOW_CONFIDENCE"
    """La astrometria no se confirma a si misma: sin inliers de validacion."""

    PRIOR_MISMATCH_MOTION = "OBS_PRIOR_MISMATCH_MOTION"
    """El cielo se movio distinto de lo que dicen los pasos comandados."""

    PRIOR_MISMATCH_ROLL = "OBS_PRIOR_MISMATCH_ROLL"
    """El giro de campo salto mas de lo que puede saltar."""

    MOUNT_MOVING = "OBS_MOUNT_MOVING"
    """La montura se movio durante la exposicion."""


@dataclass(frozen=True)
class SolveQuality:
    """Lo que la astrometria sabe sobre su propia respuesta.

    ``validation_inliers`` es el discriminador: estrellas que confirman la
    hipotesis *aparte* de las tres que la definieron. Un solve de tres inliers no
    prueba nada por bajo que sea su rms; de hecho el rms sale ridiculamente bajo
    justo por ser un ajuste exacto.
    """

    n_detections: int = 0
    n_inliers: int = 0
    validation_inliers: int = 0
    rms_inliers_arcsec: float = float("inf")
    arcsec_per_px: float = float("nan")
    offset_from_prior_deg: float = float("nan")

    def confidence(self) -> float:
        """0..1, creciente en la evidencia independiente. Solo para ordenar."""
        if self.validation_inliers <= 0 or not math.isfinite(self.rms_inliers_arcsec):
            return 0.0
        by_validation = min(1.0, self.validation_inliers / 8.0)
        by_rms = 1.0 / (1.0 + max(0.0, self.rms_inliers_arcsec) / 2.0)
        return float(by_validation * by_rms)


@dataclass(frozen=True)
class PointingObservation:
    """Donde estaba apuntando la montura, y como de seguro se sabe.

    ``steps`` son microsteps absolutos comandados, y ``load_dir`` el sentido de
    carga del tren en cada eje en el momento de la captura: sin el, el backlash
    no puede ser un termino del modelo, solo una constante.
    """

    t_unix: float
    kind: ObservationKind
    steps: Tuple[float, float]
    load_dir: Tuple[int, int]
    az_deg: float
    alt_deg: float
    sigma_arcsec: float
    quality: SolveQuality = field(default_factory=SolveQuality)
    roll_deg: Optional[float] = None
    roll_sigma_deg: Optional[float] = None
    source: str = ""
    admitted: bool = False
    reason: AdmissionReason = AdmissionReason.ADMITTED

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["reason"] = self.reason.value
        data["steps"] = list(self.steps)
        data["load_dir"] = list(self.load_dir)
        data["schema"] = SCHEMA_VERSION
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PointingObservation":
        quality = SolveQuality(**(data.get("quality") or {}))
        return cls(
            t_unix=float(data["t_unix"]),
            kind=ObservationKind(data["kind"]),
            steps=tuple(float(v) for v in data["steps"]),  # type: ignore[arg-type]
            load_dir=tuple(int(v) for v in data["load_dir"]),  # type: ignore[arg-type]
            az_deg=float(data["az_deg"]),
            alt_deg=float(data["alt_deg"]),
            sigma_arcsec=float(data["sigma_arcsec"]),
            quality=quality,
            roll_deg=data.get("roll_deg"),
            roll_sigma_deg=data.get("roll_sigma_deg"),
            source=str(data.get("source", "")),
            admitted=bool(data.get("admitted", False)),
            reason=AdmissionReason(data.get("reason", AdmissionReason.ADMITTED.value)),
        )


@dataclass(frozen=True)
class Admission:
    """El veredicto, con los numeros que lo sostienen.

    El payload existe porque en 255 sesiones no se podia saber cual de las dos
    mitades de la guardia habia saltado: se calculaban movimiento observado,
    esperado, limite y salto de roll, y se tiraban detras de un unico string.
    """

    reason: AdmissionReason
    detail: str = ""
    observed_motion_deg: float = float("nan")
    expected_motion_deg: float = float("nan")
    motion_limit_deg: float = float("nan")
    roll_jump_deg: float = float("nan")
    roll_limit_deg: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.reason is AdmissionReason.ADMITTED


def admit(
    quality: SolveQuality,
    *,
    observed_motion_deg: float = float("nan"),
    expected_motion_deg: float = float("nan"),
    motion_limit_deg: float = float("nan"),
    roll_jump_deg: float = float("nan"),
    roll_limit_deg: float = float("nan"),
    mount_moved_during_exposure: bool = False,
    min_validation_inliers: int = 2,
    strong_inliers: int = 6,
    strong_rms_arcsec: float = 1.5,
) -> Admission:
    """Decide si una observacion entra al modelo, y dice exactamente por que.

    El orden importa. Primero la astrometria, porque un solve sin confirmacion
    independiente no es una medida de nada y rechazarlo por geometria oculta la
    causa real. Despues la continuidad contra los pasos comandados, que sigue
    siendo util pero como segunda opinion.
    """
    if mount_moved_during_exposure:
        return Admission(
            reason=AdmissionReason.MOUNT_MOVING,
            detail="la montura se movio durante la exposicion",
            observed_motion_deg=observed_motion_deg,
            expected_motion_deg=expected_motion_deg,
            motion_limit_deg=motion_limit_deg,
            roll_jump_deg=roll_jump_deg,
            roll_limit_deg=roll_limit_deg,
        )

    confirmed = quality.validation_inliers >= int(min_validation_inliers)
    strong = (
        quality.n_inliers >= int(strong_inliers)
        and math.isfinite(quality.rms_inliers_arcsec)
        and quality.rms_inliers_arcsec <= float(strong_rms_arcsec)
    )
    payload = dict(
        observed_motion_deg=observed_motion_deg,
        expected_motion_deg=expected_motion_deg,
        motion_limit_deg=motion_limit_deg,
        roll_jump_deg=roll_jump_deg,
        roll_limit_deg=roll_limit_deg,
    )

    if not (confirmed or strong):
        return Admission(
            reason=AdmissionReason.LOW_CONFIDENCE,
            detail=(
                f"{quality.validation_inliers} inliers de validacion "
                f"(hacen falta {int(min_validation_inliers)}), "
                f"{quality.n_inliers} inliers, rms {quality.rms_inliers_arcsec:.2f} arcsec"
            ),
            **payload,
        )

    if (
        math.isfinite(roll_jump_deg)
        and math.isfinite(roll_limit_deg)
        and abs(roll_jump_deg) > abs(roll_limit_deg)
    ):
        return Admission(
            reason=AdmissionReason.PRIOR_MISMATCH_ROLL,
            detail=(
                f"el giro de campo salto {roll_jump_deg:+.2f} deg, "
                f"limite {abs(roll_limit_deg):.2f}"
            ),
            **payload,
        )

    if (
        math.isfinite(observed_motion_deg)
        and math.isfinite(expected_motion_deg)
        and math.isfinite(motion_limit_deg)
        and abs(observed_motion_deg - expected_motion_deg) > abs(motion_limit_deg)
    ):
        return Admission(
            reason=AdmissionReason.PRIOR_MISMATCH_MOTION,
            detail=(
                f"el cielo se movio {observed_motion_deg:.3f} deg y los pasos dicen "
                f"{expected_motion_deg:.3f}, limite {abs(motion_limit_deg):.3f}"
            ),
            **payload,
        )

    return Admission(reason=AdmissionReason.ADMITTED, detail="", **payload)


class ObservationStore:
    """Un JSONL por noche. Se lee al arrancar, no cuando alguien se acuerda.

    El modelo anterior solo restauraba su estado con un comando manual, asi que
    cerrar la app perdia una noche de solves ganados a pulso. Aqui la noche en
    curso se carga sola.
    """

    def __init__(self, root: Path | str = "stack_output/pointing") -> None:
        self.root = Path(root)

    def path_for(self, t_unix: Optional[float] = None) -> Path:
        """Fichero de la noche a la que pertenece ese instante.

        La noche va de mediodia a mediodia local, para que una sesion que cruza
        la medianoche siga siendo la misma noche.
        """
        stamp = time.localtime(time.time() if t_unix is None else float(t_unix))
        day = time.mktime(stamp) - (12.0 * 3600.0)
        night = time.localtime(day)
        return self.root / f"observations_{time.strftime('%Y%m%d', night)}.jsonl"

    def append(self, observation: PointingObservation) -> Path:
        path = self.path_for(observation.t_unix)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(observation.to_json(), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return path

    def load(self, t_unix: Optional[float] = None) -> List[PointingObservation]:
        """Todas las observaciones de esa noche, admitidas y rechazadas."""
        path = self.path_for(t_unix)
        if not path.exists():
            return []
        out: List[PointingObservation] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(PointingObservation.from_json(json.loads(raw)))
            except (ValueError, KeyError, TypeError):
                # Una linea corrupta no puede costar la noche entera.
                continue
        return out

    def load_admitted(self, t_unix: Optional[float] = None) -> List[PointingObservation]:
        return [obs for obs in self.load(t_unix) if obs.admitted]

    def load_current_night(self) -> List[PointingObservation]:
        return self.load(None)
