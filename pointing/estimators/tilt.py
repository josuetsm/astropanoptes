"""Los tres terminos geometricos, ajustados a la vez y no por turnos.

Aqui es donde muere el esquema anterior. El codigo viejo ajustaba J contra
posiciones del mundo sin aplicar la rotacion, y despues ajustaba la rotacion
*dada* esa J. Pero una inclinacion de la base hace que un paso de altitud cambie
el azimut del mundo, y eso caia en `J[0,1]`; la rotacion explicaba exactamente lo
mismo. Los dos estimadores se turnaban sobre un unico grado de libertad, que es
lo que producia los ajustes no idempotentes y el `DEGENERATE_MODEL`.

La salida es que los cuatro efectos son *linealmente* separables si se escriben
con su dependencia real de la altura, que es distinta en cada uno:

    IA, IE   offsets de indice        constantes
    AN, AW   inclinacion del eje az   sin/cos del azimut, y tan(alt) en azimut
    NPAE     no-perpendicularidad     tan(alt)
    CA       colimacion               sec(alt)

Seis parametros, todos lineales, un solo minimos-cuadrados ponderado. Los
offsets de indice entran como parametros de estorbo y salen por la puerta de
atras: los absorbe la sincronizacion, que es su sitio.

Lo que hace falta para que sea resoluble
----------------------------------------
Las formas anteriores solo se distinguen si las observaciones las excitan. Una
inclinacion es un seno del azimut: sin recorrido en azimut, su magnitud y su
direccion se confunden entre si. tan(alt) y sec(alt) casi coinciden a baja
altura: sin una muestra alta, la no-perpendicularidad y la colimacion son la
misma columna. Por eso el estimador comprueba la geometria de los datos *antes*
de resolver y, si no llega, dice que direccion falta en vez de devolver un
numero que no significa nada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["GeometrySample", "GeometryOutcome", "GeometryFit", "fit_geometry"]


class GeometryOutcome(str, Enum):
    OK = "GEOM_OK"
    NOT_ENOUGH_DATA = "GEOM_NOT_ENOUGH_DATA"
    ILL_CONDITIONED = "GEOM_ILL_CONDITIONED"
    """Los datos no excitan alguna de las formas. El detalle dice cual falta."""

    OUT_OF_BOUNDS = "GEOM_OUT_OF_BOUNDS"
    """Algun parametro sale de lo fisicamente posible: se rechaza, no se recorta."""

    UNSTABLE = "GEOM_UNSTABLE"
    """Una sola observacion decide el resultado; es un outlier con palanca."""

    POOR_FIT = "GEOM_POOR_FIT"
    """Los residuos no bajan: el modelo no describe estas observaciones."""


@dataclass(frozen=True)
class GeometrySample:
    """Una observacion admitida, ya reducida a lo que el ajuste necesita.

    ``az_nominal_deg``/``alt_nominal_deg`` son donde la mecanica dice que apunta
    (pasos por grados-por-paso, sin ninguna correccion geometrica), y
    ``az_obs_deg``/``alt_obs_deg`` donde la astrometria dice que apunta de
    verdad. La diferencia es lo que los terminos tienen que explicar.
    """

    az_nominal_deg: float
    alt_nominal_deg: float
    az_obs_deg: float
    alt_obs_deg: float
    sigma_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_deg) or self.sigma_deg <= 0.0:
            raise ValueError("sigma_deg must be positive and finite")


@dataclass(frozen=True)
class GeometryFit:
    """El resultado, con lo necesario para creerselo o no."""

    outcome: GeometryOutcome
    base_tilt_deg: Tuple[float, float] = (0.0, 0.0)
    npae_deg: float = 0.0
    collimation_deg: float = 0.0
    index_offset_deg: Tuple[float, float] = (0.0, 0.0)
    sigma: Dict[str, float] = field(default_factory=dict)
    rms_arcsec: float = float("nan")
    rms_before_arcsec: float = float("nan")
    n_samples: int = 0
    condition_number: float = float("nan")
    az_span_deg: float = 0.0
    alt_span_deg: float = 0.0
    max_alt_deg: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is GeometryOutcome.OK


def fit_geometry(
    samples: Sequence[GeometrySample],
    *,
    bounds: Optional[Dict[str, float]] = None,
    min_samples: int = 6,
    min_az_span_deg: float = 120.0,
    min_alt_span_deg: float = 25.0,
    min_high_alt_deg: float = 60.0,
    max_condition: float = 100.0,
    max_rms_arcsec: float = 60.0,
    max_leverage_sigmas: float = 2.0,
) -> GeometryFit:
    """Ajusta inclinacion, no-perpendicularidad y colimacion de una vez.

    Comprueba primero si los datos pueden sostener el ajuste, resuelve despues, y
    verifica por leave-one-out que ninguna observacion sola decide el resultado.
    """
    limits = {"base_tilt": 2.0, "npae": 2.0, "collimation": 1.0}
    limits.update(bounds or {})

    n = len(samples)
    if n < int(min_samples):
        return GeometryFit(
            outcome=GeometryOutcome.NOT_ENOUGH_DATA,
            n_samples=n,
            detail=f"hacen falta {int(min_samples)} observaciones admitidas, hay {n}",
        )

    az = np.array([s.az_nominal_deg for s in samples], dtype=np.float64)
    alt = np.array([s.alt_nominal_deg for s in samples], dtype=np.float64)
    az_span = _angular_span_deg(az)
    alt_span = float(np.max(alt) - np.min(alt))
    max_alt = float(np.max(alt))

    geometry = dict(az_span_deg=az_span, alt_span_deg=alt_span, max_alt_deg=max_alt, n_samples=n)

    missing = []
    if az_span < float(min_az_span_deg):
        missing.append(
            f"recorrido en azimut {az_span:.0f} deg (hacen falta {float(min_az_span_deg):.0f}): "
            "una inclinacion es un seno del azimut y sin recorrido su magnitud y su "
            "direccion se confunden"
        )
    if alt_span < float(min_alt_span_deg):
        missing.append(
            f"recorrido en altura {alt_span:.0f} deg (hacen falta {float(min_alt_span_deg):.0f})"
        )
    if max_alt < float(min_high_alt_deg):
        missing.append(
            f"ninguna muestra por encima de {float(min_high_alt_deg):.0f} deg de altura: "
            "sin eso tan(alt) y sec(alt) son la misma columna y la "
            "no-perpendicularidad no se separa de la colimacion"
        )
    if missing:
        return GeometryFit(
            outcome=GeometryOutcome.ILL_CONDITIONED,
            detail="; ".join(missing),
            **geometry,
        )

    design, target, weights, sky_scale = _build_system(samples)
    condition = _collinearity(design, weights)
    geometry["condition_number"] = condition
    if not math.isfinite(condition) or condition > float(max_condition):
        return GeometryFit(
            outcome=GeometryOutcome.ILL_CONDITIONED,
            detail=(
                f"numero de condicion {condition:.0f} (maximo {float(max_condition):.0f}): "
                "las observaciones estan demasiado alineadas para separar los terminos"
            ),
            **geometry,
        )

    params, covariance, offset = _solve_iterating(samples)
    design, target, weights, sky_scale = _build_system(samples, offset)
    residual = target - design @ params
    # en arcsec sobre el cielo, que es la unidad en que se juzga un apuntado
    rms_arcsec = float(np.sqrt(np.mean((residual * sky_scale) ** 2)) * 3600.0)
    rms_before = float(np.sqrt(np.mean((target * sky_scale) ** 2)) * 3600.0)

    ia, ie, an, aw, npae, ca = (float(v) for v in params)
    ia, ie = ia + float(offset[0]), ie + float(offset[1])
    sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    named_sigma = {
        "index_az": float(sigmas[0]), "index_alt": float(sigmas[1]),
        "tilt_north": float(sigmas[2]), "tilt_west": float(sigmas[3]),
        "npae": float(sigmas[4]), "collimation": float(sigmas[5]),
    }
    common = dict(
        base_tilt_deg=(an, aw),
        npae_deg=npae,
        collimation_deg=ca,
        index_offset_deg=(ia, ie),
        sigma=named_sigma,
        rms_arcsec=rms_arcsec,
        rms_before_arcsec=rms_before,
        **geometry,
    )

    over = []
    if math.hypot(an, aw) > limits["base_tilt"]:
        over.append(f"inclinacion {math.hypot(an, aw):.2f} deg > {limits['base_tilt']:.2f}")
    if abs(npae) > limits["npae"]:
        over.append(f"no-perpendicularidad {npae:.2f} deg > {limits['npae']:.2f}")
    if abs(ca) > limits["collimation"]:
        over.append(f"colimacion {ca:.2f} deg > {limits['collimation']:.2f}")
    if over:
        return GeometryFit(
            outcome=GeometryOutcome.OUT_OF_BOUNDS,
            detail="; ".join(over) + "; se rechaza en vez de recortarse",
            **common,
        )

    if math.isfinite(rms_arcsec) and rms_arcsec > float(max_rms_arcsec):
        return GeometryFit(
            outcome=GeometryOutcome.POOR_FIT,
            detail=(
                f"rms {rms_arcsec:.0f} arcsec (maximo {float(max_rms_arcsec):.0f}); "
                f"antes del ajuste era {rms_before:.0f}"
            ),
            **common,
        )

    worst_name, worst_shift = _leverage(samples, params, sigmas)
    if worst_shift > float(max_leverage_sigmas):
        return GeometryFit(
            outcome=GeometryOutcome.UNSTABLE,
            detail=(
                f"quitar una sola observacion mueve {worst_name} {worst_shift:.1f} sigmas: "
                "el ajuste lo decide un outlier con palanca, no el conjunto"
            ),
            **common,
        )

    return GeometryFit(
        outcome=GeometryOutcome.OK,
        detail=(
            f"inclinacion=({an:+.3f},{aw:+.3f}) npae={npae:+.3f} colimacion={ca:+.3f} deg; "
            f"rms {rms_before:.0f} -> {rms_arcsec:.0f} arcsec"
        ),
        **common,
    )


def _collinearity(design: np.ndarray, weights: np.ndarray) -> float:
    """Cuanto se parecen entre si las columnas, en una escala comparable.

    Se normaliza cada columna antes de medir para que el numero hable de
    colinealidad y no de unidades: si dependiera de la escala, cambiar las sigmas
    moveria el umbral sin que la geometria de los datos hubiera cambiado.

    Medido sobre repartos sinteticos: uno excelente da 36, uno bueno 66, uno sin
    muestras altas 237 y uno con azimut estrecho 298. La colinealidad residual de
    un reparto bueno es intrinseca -- tan(alt) y sec(alt) se parecen -- asi que el
    umbral separa "resoluble" de "degenerado", no "perfecto" de "bueno".
    """
    scaled = design * weights[:, None]
    norms = np.linalg.norm(scaled, axis=0)
    norms[norms == 0.0] = 1.0
    return float(np.linalg.cond(scaled / norms))


def _build_system(
    samples: Sequence[GeometrySample],
    offset_deg: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matriz de diseno, residuos objetivo, pesos y factor a grados de cielo.

    El peso y el factor geometrico son cosas distintas y por eso viajan
    separados: el peso dice cuanto vale una observacion en el ajuste, el factor
    convierte un residuo en azimut a error sobre el cielo para poder reportarlo
    en arcsec.

    El residuo en azimut se pondera por cos(alt) para convertirlo en error sobre
    el cielo: un grado de azimut cerca del cenit es mucho menos que un grado
    abajo, y sin eso las muestras altas dominarian el ajuste sin merecerlo.
    """
    rows: List[np.ndarray] = []
    targets: List[float] = []
    weights: List[float] = []
    sky_scale: List[float] = []
    for s in samples:
        # La base se evalua en los angulos de eje, que no se conocen hasta saber
        # donde esta el cero: por eso se corrige con el offset de la pasada
        # anterior. Sin esa correccion, tan(alt) y sec(alt) se evaluan un grado
        # fuera y el residuo resultante supera de largo el sigma de un solve.
        az = math.radians(s.az_nominal_deg + offset_deg[0])
        alt = math.radians(s.alt_nominal_deg + offset_deg[1])
        tan_alt = math.tan(alt)
        cos_alt = math.cos(alt)
        sec_alt = 1.0 / cos_alt if abs(cos_alt) > 1e-9 else 1e9

        rows.append(np.array([1.0, 0.0, -math.sin(az) * tan_alt, -math.cos(az) * tan_alt,
                              tan_alt, sec_alt]))
        targets.append(_wrap180(s.az_obs_deg - s.az_nominal_deg - offset_deg[0]))
        weights.append(cos_alt / s.sigma_deg)
        sky_scale.append(cos_alt)

        rows.append(np.array([0.0, 1.0, math.cos(az), -math.sin(az), 0.0, 0.0]))
        targets.append(s.alt_obs_deg - s.alt_nominal_deg - offset_deg[1])
        weights.append(1.0 / s.sigma_deg)
        sky_scale.append(1.0)

    return (
        np.vstack(rows),
        np.array(targets, dtype=np.float64),
        np.array(weights, dtype=np.float64),
        np.array(sky_scale, dtype=np.float64),
    )


def _solve(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    w = weights[:, None]
    a = design * w
    b = target * weights
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    normal = a.T @ a
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        covariance = np.full((6, 6), np.inf)
    return params, covariance


def _solve_iterating(
    samples: Sequence[GeometrySample], passes: int = 3
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Resuelve corrigiendo donde se evalua la base, no solo los residuos.

    Los offsets de indice dicen cuanto estaba desplazado el nominal, asi que
    reinyectarlos y repetir deja la base evaluada en los angulos de eje reales.
    Converge en dos pasadas; se hacen tres.
    """
    offset = (0.0, 0.0)
    params = np.zeros(6)
    covariance = np.zeros((6, 6))
    for _ in range(int(passes)):
        design, target, weights, _ = _build_system(samples, offset)
        params, covariance = _solve(design, target, weights)
        offset = (offset[0] + float(params[0]), offset[1] + float(params[1]))
    return params, covariance, offset


def _leverage(
    samples: Sequence[GeometrySample], params: np.ndarray, sigmas: np.ndarray
) -> Tuple[str, float]:
    """Cuanto mueve el resultado quitar la observacion mas influyente.

    Es la comprobacion que el codigo anterior hacia por fuerza bruta, probando
    cada muestra como referencia. Aqui es una pregunta directa: si el ajuste lo
    decide un solo punto, no es un ajuste.
    """
    names = ("index_az", "index_alt", "tilt_north", "tilt_west", "npae", "collimation")
    worst_name, worst = names[0], 0.0
    for drop in range(len(samples)):
        subset = [s for i, s in enumerate(samples) if i != drop]
        if len(subset) < 5:
            continue
        try:
            trial, _, _ = _solve_iterating(subset)
        except np.linalg.LinAlgError:
            continue
        for idx, name in enumerate(names):
            sigma = float(sigmas[idx])
            if not math.isfinite(sigma) or sigma <= 0.0:
                continue
            shift = abs(float(trial[idx]) - float(params[idx])) / sigma
            if shift > worst:
                worst_name, worst = name, shift
    return worst_name, worst


def _angular_span_deg(angles_deg: np.ndarray) -> float:
    """Recorrido en azimut, contando que 350 y 10 estan a 20 grados.

    El hueco mas grande entre muestras consecutivas alrededor del circulo dice
    cuanto del circulo *no* se cubrio; lo que queda es el recorrido util.
    """
    if angles_deg.size < 2:
        return 0.0
    ordered = np.sort(np.asarray(angles_deg, dtype=np.float64) % 360.0)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    return float(360.0 - np.max(gaps))


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0
