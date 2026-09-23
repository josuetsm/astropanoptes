"""El mapa de pasos a cielo, sin ninguna matriz libre.

    pasos --(cinematica: deg/paso fijo, nunca ajustado)--> angulos de eje nominales
          --(+ rizado de transmision, + backlash segun sentido)--> angulos reales
          --(inclinacion, no-perpendicularidad, colimacion)--> AltAz del mundo

El modelo anterior metia todo esto en una J de 2x2 ajustada libremente y luego la
recortaba al 10% de la mecanica. Eso hacia dos cosas malas a la vez: ajustaba la
reduccion, que se conoce exactamente, y mezclaba en los mismos cuatro numeros
efectos que dependen de la altura de formas distintas -- la inclinacion entra
como sin/cos del azimut, la no-perpendicularidad como tan(alt) y la colimacion
como sec(alt). Una matriz constante no puede representar tan(alt), asi que el
codigo terminaba ensanchando su tolerancia por 1/cos(alt) para no rechazar datos
buenos. Esa tolerancia era el sintoma; los terminos con nombre son la cura.

Los terminos geometricos son los clasicos de una montura alt-az:

    AN, AW   inclinacion del eje de azimut (norte, oeste)
    NPAE     el eje de altitud no es perpendicular al de azimut
    CA       colimacion: el eje optico no es perpendicular al de altitud

Los offsets de indice (donde esta el cero de cada eje) no son terminos: los
absorbe la sincronizacion, que es su sitio.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np

from ap_types import Axis
from pointing.kinematics import MountKinematics
from pointing.sky import sidereal_world_rate_deg_s
from pointing.terms import Provenance, Term, TermSet, default_terms

__all__ = ["PointingReference", "PointingModel", "apply_geometry", "remove_geometry"]

_MAX_ALT_FOR_TAN_DEG = 89.0
"""Cerca del cenit tan(alt) diverge y el modelo deja de tener sentido numerico."""


def apply_geometry(
    az_axis_deg: float,
    alt_axis_deg: float,
    *,
    tilt_north_deg: float,
    tilt_west_deg: float,
    npae_deg: float,
    collimation_deg: float,
) -> Tuple[float, float]:
    """De angulos de eje a AltAz del mundo, aplicando los terminos geometricos.

    Es la forma estandar de un modelo de apuntado alt-az. Cada termino tiene una
    dependencia distinta con la altura, que es justo lo que los hace separables
    con observaciones bien repartidas -- y lo que los hacia indistinguibles
    dentro de una matriz constante.
    """
    az = math.radians(float(az_axis_deg))
    alt = math.radians(float(np.clip(float(alt_axis_deg), -_MAX_ALT_FOR_TAN_DEG, _MAX_ALT_FOR_TAN_DEG)))
    tan_alt = math.tan(alt)
    cos_alt = math.cos(alt)
    if abs(cos_alt) < 1e-9:
        cos_alt = math.copysign(1e-9, cos_alt or 1.0)

    d_az = (
        -float(tilt_north_deg) * math.sin(az) * tan_alt
        - float(tilt_west_deg) * math.cos(az) * tan_alt
        + float(npae_deg) * tan_alt
        + float(collimation_deg) / cos_alt
    )
    d_alt = float(tilt_north_deg) * math.cos(az) - float(tilt_west_deg) * math.sin(az)

    return (float(az_axis_deg) + d_az) % 360.0, float(alt_axis_deg) + d_alt


def remove_geometry(
    az_world_deg: float,
    alt_world_deg: float,
    *,
    tilt_north_deg: float,
    tilt_west_deg: float,
    npae_deg: float,
    collimation_deg: float,
    tolerance_deg: float = 1e-9,
    max_iterations: int = 40,
) -> Tuple[float, float]:
    """La inversa de :func:`apply_geometry`, por iteracion de punto fijo.

    Se itera hasta converger y no un numero fijo de veces: cerca del cenit la
    correccion crece como tan(alt) y sec(alt), asi que a 78 grados con terminos
    realistas son varios grados y unas pocas pasadas no bastan. El criterio es
    el residuo, que es lo que de verdad importa.
    """
    az, alt = float(az_world_deg), float(alt_world_deg)
    for _ in range(int(max_iterations)):
        az_try, alt_try = apply_geometry(
            az, alt,
            tilt_north_deg=tilt_north_deg, tilt_west_deg=tilt_west_deg,
            npae_deg=npae_deg, collimation_deg=collimation_deg,
        )
        d_az = _wrap180(float(az_world_deg) - az_try)
        d_alt = float(alt_world_deg) - alt_try
        az = (az + d_az) % 360.0
        alt = alt + d_alt
        if abs(d_az) < float(tolerance_deg) and abs(d_alt) < float(tolerance_deg):
            break
    return az, alt


@dataclass(frozen=True)
class PointingReference:
    """Donde estaba la montura la ultima vez que se supo de verdad.

    Es lo que absorbe los offsets de indice: no hace falta saber donde esta el
    cero mecanico de cada eje, solo una posicion del cielo con sus pasos.
    """

    steps: Tuple[float, float]
    az_axis_deg: float
    alt_axis_deg: float
    t_unix: float = 0.0


@dataclass
class PointingModel:
    """Estado de apuntado: mecanica conocida, desviaciones medidas, referencia."""

    kin: MountKinematics = field(default_factory=MountKinematics)
    terms: TermSet = field(default_factory=default_terms)
    reference: Optional[PointingReference] = None
    steps: Tuple[float, float] = (0.0, 0.0)
    load_dir: Tuple[int, int] = (0, 0)

    # ---------------------------------------------------------------- estado

    @property
    def synced(self) -> bool:
        """Sin referencia no se sabe donde apunta, por muy medido que este todo."""
        return self.reference is not None

    @synced.setter
    def synced(self, value: bool) -> None:
        """Solo para invalidar: apagar la sincronia es soltar la referencia.

        Encenderla a mano no significa nada -- sin referencia no hay de donde
        sacar el apuntado -- asi que ponerla a True se ignora.
        """
        if not value:
            self.reference = None

    def sync(self, *, az_world_deg: float, alt_world_deg: float, t_unix: float = 0.0) -> None:
        """Fija la referencia desde una posicion del mundo observada."""
        az_axis, alt_axis = remove_geometry(
            az_world_deg, alt_world_deg, **self._geometry_kwargs()
        )
        self.reference = PointingReference(
            steps=tuple(float(v) for v in self.steps),  # type: ignore[arg-type]
            az_axis_deg=az_axis,
            alt_axis_deg=alt_axis,
            t_unix=float(t_unix),
        )

    def note_steps(self, steps: Sequence[float], *, load_dir: Optional[Sequence[int]] = None) -> None:
        self.steps = (float(steps[0]), float(steps[1]))
        if load_dir is not None:
            self.load_dir = (int(np.sign(load_dir[0])), int(np.sign(load_dir[1])))

    # ------------------------------------------------------------ mapa directo

    def axis_angles_deg(
        self,
        steps: Optional[Sequence[float]] = None,
        *,
        load_dir: Optional[Sequence[int]] = None,
    ) -> Tuple[float, float]:
        """Angulos reales de los ejes para un contador de pasos dado."""
        if self.reference is None:
            raise RuntimeError("no hay referencia: sincroniza antes de predecir")
        s = self.steps if steps is None else (float(steps[0]), float(steps[1]))
        d = self.load_dir if load_dir is None else (int(np.sign(load_dir[0])), int(np.sign(load_dir[1])))

        out = []
        for idx, axis in enumerate((Axis.AZ, Axis.ALT)):
            k = self.kin.deg_per_step(axis)
            base = (self.reference.az_axis_deg if idx == 0 else self.reference.alt_axis_deg)
            travelled = k * (s[idx] - self.reference.steps[idx])
            ripple = self._ripple_deg(idx, s[idx]) - self._ripple_deg(idx, self.reference.steps[idx])
            slack = self._backlash_offset_deg(idx, d[idx]) - self._backlash_offset_deg(
                idx, self._reference_load_dir(idx)
            )
            out.append(base + travelled + ripple + slack)
        return float(out[0]) % 360.0, float(out[1])

    def world_from_steps(
        self,
        steps: Optional[Sequence[float]] = None,
        *,
        load_dir: Optional[Sequence[int]] = None,
    ) -> Tuple[float, float]:
        """AltAz del mundo al que apunta la montura con esos pasos."""
        az_axis, alt_axis = self.axis_angles_deg(steps, load_dir=load_dir)
        return apply_geometry(az_axis, alt_axis, **self._geometry_kwargs())

    def current_world_deg(self) -> Optional[Tuple[float, float]]:
        if self.reference is None:
            return None
        return self.world_from_steps()

    # ------------------------------------------------------------ mapa inverso

    def steps_for_world(
        self,
        az_world_deg: float,
        alt_world_deg: float,
        *,
        load_dir: Optional[Sequence[int]] = None,
        iterations: int = 6,
    ) -> Tuple[float, float]:
        """Pasos absolutos que hay que comandar para apuntar ahi.

        El rizado de transmision hace la relacion no lineal, asi que se resuelve
        por iteracion: la correccion es pequena y converge sola.
        """
        if self.reference is None:
            raise RuntimeError("no hay referencia: sincroniza antes de apuntar")
        az_axis, alt_axis = remove_geometry(
            az_world_deg, alt_world_deg, **self._geometry_kwargs()
        )
        d = self.load_dir if load_dir is None else (int(np.sign(load_dir[0])), int(np.sign(load_dir[1])))

        target = (az_axis, alt_axis)
        guess = list(self.reference.steps)
        for idx, axis in enumerate((Axis.AZ, Axis.ALT)):
            k = self.kin.deg_per_step(axis)
            base = (self.reference.az_axis_deg if idx == 0 else self.reference.alt_axis_deg)
            slack = self._backlash_offset_deg(idx, d[idx]) - self._backlash_offset_deg(
                idx, self._reference_load_dir(idx)
            )
            ref_ripple = self._ripple_deg(idx, self.reference.steps[idx])
            wanted = target[idx] - base - slack
            if idx == 0:
                wanted = _wrap180(wanted)
            step = self.reference.steps[idx] + wanted / k
            for _ in range(int(iterations)):
                residual = wanted - (
                    k * (step - self.reference.steps[idx])
                    + self._ripple_deg(idx, step)
                    - ref_ripple
                )
                step += residual / k
            guess[idx] = float(step)
        return float(guess[0]), float(guess[1])

    # ------------------------------------------------------------------ interno

    def _geometry_kwargs(self) -> dict:
        tilt = self.terms["base_tilt"].value
        return dict(
            tilt_north_deg=float(tilt[0]),
            tilt_west_deg=float(tilt[1]),
            npae_deg=float(self.terms["axis_non_perpendicularity"].value[0]),
            collimation_deg=float(self.terms["collimation"].value[0]),
        )

    def _ripple_deg(self, axis_index: int, steps: float) -> float:
        """Primer armonico del error de transmision en ese eje."""
        coeff = self.terms["transmission_error_deg"].value
        c_sin, c_cos = float(coeff[2 * axis_index]), float(coeff[2 * axis_index + 1])
        if c_sin == 0.0 and c_cos == 0.0:
            return 0.0
        axis = Axis.AZ if axis_index == 0 else Axis.ALT
        period = self.kin.transmission_error_period_steps(axis)
        phase = 2.0 * math.pi * float(steps) / period
        return c_sin * math.sin(phase) + c_cos * math.cos(phase)

    def _backlash_offset_deg(self, axis_index: int, direction: int) -> float:
        """Donde se apoya el eje dentro de su holgura, segun el sentido de carga.

        Moverse en un sentido apoya el flanco de ese lado, asi que el eje queda a
        media holgura de un lado o del otro. La diferencia entre ambos extremos
        es el juego entero, que es lo que se pierde al invertir.
        """
        slack_steps = float(self.terms["backlash_steps"].value[axis_index])
        if slack_steps == 0.0 or direction == 0:
            return 0.0
        axis = Axis.AZ if axis_index == 0 else Axis.ALT
        k = self.kin.deg_per_step(axis)
        return -0.5 * k * slack_steps * float(np.sign(direction))

    def _reference_load_dir(self, axis_index: int) -> int:
        # La referencia se tomo con el tren cargado en el sentido de entonces; si
        # no se registro, se asume el actual y el termino se cancela.
        return int(self.load_dir[axis_index])

    # ------------------------------------------------------------- jacobiana

    def world_jacobian(
        self,
        steps: Optional[Sequence[float]] = None,
        *,
        delta_steps: float = 200.0,
    ) -> np.ndarray:
        """d(AltAz del mundo) / d(pasos), 2x2, por diferencia central.

        Numerica y no analitica a proposito: asi incluye sola la inclinacion, la
        no-perpendicularidad, la colimacion y el rizado, sin que nadie tenga que
        derivarlas a mano ni acordarse de anadir la siguiente. El paso de 200
        microsteps es lo bastante grande para no sufrir cancelacion y lo bastante
        pequeno para que la curvatura no importe.
        """
        s = list(self.steps if steps is None else [float(steps[0]), float(steps[1])])
        jac = np.zeros((2, 2), dtype=np.float64)
        for axis_index in range(2):
            plus, minus = list(s), list(s)
            plus[axis_index] += float(delta_steps)
            minus[axis_index] -= float(delta_steps)
            az_p, alt_p = self.world_from_steps(plus)
            az_m, alt_m = self.world_from_steps(minus)
            jac[0, axis_index] = _wrap180(az_p - az_m) / (2.0 * float(delta_steps))
            jac[1, axis_index] = (alt_p - alt_m) / (2.0 * float(delta_steps))
        return jac

    def step_rate_for_world_rate(
        self,
        world_rate_deg_s: Sequence[float],
        steps: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Pasos por segundo que producen esa velocidad sobre el cielo."""
        v = np.asarray(world_rate_deg_s, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(v)):
            return None
        jac = self.world_jacobian(steps)
        try:
            rate = np.linalg.solve(jac, v)
        except np.linalg.LinAlgError:
            rate, *_ = np.linalg.lstsq(jac, v, rcond=None)
        return rate if np.all(np.isfinite(rate)) else None


    # ============================================================
    # Fachada de compatibilidad
    # ============================================================
    #
    # `app_runner` lee 23 miembros del modelo anterior. Reproducirlos aqui es lo
    # que permite cambiar el modelo en un commit pequeno y reversible, en vez de
    # tocar el runner y el modelo a la vez. Es codigo de transicion: cuando el
    # runner hable en terminos de `pointing`, esta seccion se borra entera.

    @property
    def J_deg_per_step(self) -> np.ndarray:
        """La matriz mecanica, diagonal y fija. Ya no se ajusta nadie.

        El tracking la usa solo como semilla de su propia estimacion por RLS, que
        es un lazo cerrado y reaprende la respuesta real de todos modos. Para
        cualquier otra cosa esta `world_jacobian`, que incluye los terminos
        geometricos.
        """
        return np.array(
            [[self.kin.deg_per_step(Axis.AZ), 0.0], [0.0, self.kin.deg_per_step(Axis.ALT)]],
            dtype=np.float64,
        )

    def safe_J_for_prediction(self) -> np.ndarray:
        return self.J_deg_per_step

    def mechanical_J(self) -> np.ndarray:
        return self.J_deg_per_step

    def init_from_mechanics(self) -> None:
        """Sin efecto: la mecanica es el punto de partida y no se abandona nunca."""

    def set_microsteps(self, az_div: int, alt_div: int) -> None:
        self.kin.microsteps_az = int(az_div)
        self.kin.microsteps_alt = int(alt_div)

    @property
    def steps_est(self) -> np.ndarray:
        return np.array(self.steps, dtype=np.float64)

    @steps_est.setter
    def steps_est(self, value: Sequence[float]) -> None:
        self.steps = (float(value[0]), float(value[1]))

    @property
    def backlash_steps_az(self) -> int:
        return int(round(float(self.terms["backlash_steps"].value[0])))

    @property
    def backlash_steps_alt(self) -> int:
        return int(round(float(self.terms["backlash_steps"].value[1])))

    @property
    def periodic_coeff_deg(self) -> np.ndarray:
        return np.asarray(self.terms["transmission_error_deg"].value, dtype=np.float64).reshape(2, 2)

    @periodic_coeff_deg.setter
    def periodic_coeff_deg(self, value) -> None:
        flat = np.asarray(value, dtype=np.float64).reshape(4)
        term = self.terms["transmission_error_deg"]
        self.terms.set(term.updated(
            flat, sigma=term.sigma, provenance=Provenance.TRACKING, n_obs=max(1, term.n_obs)
        ))

    @property
    def periodic_model_samples(self) -> int:
        return int(self.terms["transmission_error_deg"].n_obs)

    @periodic_model_samples.setter
    def periodic_model_samples(self, value: int) -> None:
        term = self.terms["transmission_error_deg"]
        self.terms.set(term.updated(
            term.value, sigma=term.sigma, provenance=term.provenance, n_obs=int(value)
        ))

    def safe_periodic_coeff_for_prediction(self, coeff=None) -> np.ndarray:
        """Acota el rizado a lo que cada reductor puede rizar. Por eje.

        Azimut lleva un cicloidal impreso y altitud un planetario comprado: la
        misma amplitud es normal en uno y una medida mala en el otro.
        """
        candidate = self.periodic_coeff_deg if coeff is None else np.asarray(coeff, dtype=np.float64)
        candidate = candidate.reshape(2, 2).copy()
        if not np.all(np.isfinite(candidate)):
            return np.zeros((2, 2), dtype=np.float64)
        bound = np.asarray(self.terms["transmission_error_deg"].bound, dtype=np.float64).reshape(2, 2)
        return np.clip(candidate, -bound, bound)

    @property
    def model_fit_samples(self) -> int:
        return int(self.terms["base_tilt"].n_obs)

    @property
    def model_roll_deg(self) -> float:
        return float(self.terms["camera_roll"].value[0])

    @property
    def model_roll_samples(self) -> int:
        return int(self.terms["camera_roll"].n_obs)

    @property
    def last_move_direction_az(self) -> int:
        return int(self.load_dir[0])

    @property
    def last_move_direction_alt(self) -> int:
        return int(self.load_dir[1])

    def last_move_direction(self, axis: Axis) -> int:
        return int(self.load_dir[0 if axis == Axis.AZ else 1])

    def set_last_move_direction(self, axis: Axis, direction: int) -> None:
        idx = 0 if axis == Axis.AZ else 1
        load = list(self.load_dir)
        load[idx] = int(np.sign(direction))
        self.load_dir = (int(load[0]), int(load[1]))

    def note_manual_move(self, axis: Axis, direction: int, steps: int) -> None:
        idx = 0 if axis == Axis.AZ else 1
        moved = list(self.steps)
        moved[idx] += float(int(np.sign(direction)) * abs(int(steps)))
        self.steps = (float(moved[0]), float(moved[1]))
        if steps:
            self.set_last_move_direction(axis, direction)

    def note_emitted_rate_steps(self, dsteps: Sequence[float]) -> None:
        delta = np.asarray(dsteps, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(delta)):
            return
        self.steps = (self.steps[0] + float(delta[0]), self.steps[1] + float(delta[1]))
        for idx, axis in enumerate((Axis.AZ, Axis.ALT)):
            if abs(float(delta[idx])) > 0.0:
                self.set_last_move_direction(axis, int(np.sign(float(delta[idx]))))

    def current_az_alt_deg(self) -> Optional[np.ndarray]:
        where = self.current_world_deg()
        return None if where is None else np.array(where, dtype=np.float64)

    def predict_az_alt_deg(self, *, from_ref: bool = False) -> np.ndarray:
        if self.reference is None:
            return np.array([float("nan"), float("nan")], dtype=np.float64)
        steps = self.reference.steps if from_ref else self.steps
        return np.array(self.world_from_steps(steps), dtype=np.float64)

    def sync_from_world_az_alt(self, az_alt_world_deg: Sequence[float]) -> bool:
        values = np.asarray(az_alt_world_deg, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(values)):
            return False
        self.sync(az_world_deg=float(values[0]), alt_world_deg=float(values[1]), t_unix=time.time())
        return True

    def sidereal_step_rate_deg_s(
        self,
        *,
        az_deg: float,
        alt_deg: float,
        observer,
        obstime=None,
        dt_s: float = 1.0,
        cond_max: float = 1e6,
    ) -> Optional[np.ndarray]:
        """Pasos por segundo para seguir el cielo desde ese apuntado.

        La velocidad sobre el horizonte sale de `pointing.sky`; convertirla a
        pasos usa la jacobiana numerica del modelo, que ya lleva dentro la
        inclinacion, la no-perpendicularidad, la colimacion y el rizado.
        """
        world_rate = sidereal_world_rate_deg_s(
            az_deg=az_deg, alt_deg=alt_deg,
            location=observer.location(), obstime=obstime, dt_s=dt_s,
        )
        if world_rate is None:
            return None
        if self.reference is None:
            jac = self.J_deg_per_step
            try:
                rate = np.linalg.solve(jac, world_rate)
            except np.linalg.LinAlgError:
                return None
            return rate if np.all(np.isfinite(rate)) else None
        jac = self.world_jacobian()
        if float(np.linalg.cond(jac)) > float(cond_max):
            return None
        return self.step_rate_for_world_rate(world_rate)

    def model_fit_report(self) -> dict:
        """Lo que hay medido, en forma de diccionario plano."""
        report = {
            "model_fit_samples": self.model_fit_samples,
            "model_roll_deg": self.model_roll_deg,
            "model_roll_samples": self.model_roll_samples,
            "periodic_model_samples": self.periodic_model_samples,
            "backlash_steps_az": self.backlash_steps_az,
            "backlash_steps_alt": self.backlash_steps_alt,
            "synced": self.synced,
        }
        for term in self.terms:
            report[f"term_{term.name}"] = [float(v) for v in np.atleast_1d(term.value)]
            report[f"term_{term.name}_provenance"] = term.provenance.value
        return report

    # -------------------------------------------------------------- diagnostico

    def describe(self) -> str:
        head = (
            f"referencia={'si' if self.synced else 'NO'} "
            f"pasos=({self.steps[0]:.0f},{self.steps[1]:.0f}) "
            f"sentido={self.load_dir}"
        )
        return head + "\n" + self.terms.describe()


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0
