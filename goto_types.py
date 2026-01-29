# goto_types.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ap_types import Axis
from config import SepConfig
from logging_utils import log_error
from platesolve import ObserverConfig, PlatesolveResult

from goto_calibration import fit_J_from_samples
from goto_math import as_array2, wrap_deg_180, wrap_deg_360


TargetType = Union[
    "SkyCoord",
    Tuple[float, float],
    Tuple[str, str],
    str,
    Dict[str, Any],
]


@dataclass
class MountKinematics:
    """Mechanical parameters used to compute an initial steps/deg model."""

    motor_full_steps_per_rev: int = 200
    microsteps_az: int = 64
    microsteps_alt: int = 64

    motor_pulley_teeth: int = 20
    belt_pitch_m: float = 0.002

    ring_radius_m_az: float = 0.24
    ring_radius_m_alt: float = 0.235

    axis_sign_az: int = +1
    axis_sign_alt: int = +1

    def ring_teeth(self, axis: Axis) -> float:
        r = float(self.ring_radius_m_az if axis == Axis.AZ else self.ring_radius_m_alt)
        return float((2.0 * math.pi * r) / float(self.belt_pitch_m))

    def microsteps_per_motor_rev(self, axis: Axis) -> int:
        ms = int(self.microsteps_az if axis == Axis.AZ else self.microsteps_alt)
        return int(self.motor_full_steps_per_rev) * ms

    def steps_per_axis_rev(self, axis: Axis) -> float:
        """Microsteps per full 360° axis revolution."""
        mu = float(self.microsteps_per_motor_rev(axis))
        ratio = float(self.ring_teeth(axis)) / float(self.motor_pulley_teeth)
        return float(mu * ratio)

    def steps_per_deg(self, axis: Axis) -> float:
        return float(self.steps_per_axis_rev(axis) / 360.0)

    def deg_per_step(self, axis: Axis) -> float:
        spd = float(self.steps_per_deg(axis))
        if spd <= 0:
            raise ValueError("invalid steps_per_deg")
        sign = int(self.axis_sign_az if axis == Axis.AZ else self.axis_sign_alt)
        sign = +1 if sign >= 0 else -1
        return float(sign / spd)


@dataclass
class GoToModel:
    """Internal pointing model."""

    kin: MountKinematics = field(default_factory=MountKinematics)

    J_deg_per_step: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))

    synced: bool = False
    ref_steps: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ref_az_alt_deg: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    steps_est: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    last_solve_az_alt_deg: Optional[np.ndarray] = None
    last_solve_time: float = 0.0

    _calib_steps: List[np.ndarray] = field(default_factory=list, repr=False)
    _calib_daltaz: List[np.ndarray] = field(default_factory=list, repr=False)

    def init_from_mechanics(self) -> None:
        """Initialize J from the mechanical model (diagonal, no coupling)."""
        dps_az = self.kin.deg_per_step(Axis.AZ)
        dps_alt = self.kin.deg_per_step(Axis.ALT)
        self.J_deg_per_step = np.array(
            [[dps_az, 0.0], [0.0, dps_alt]],
            dtype=np.float64,
        )

    def set_microsteps(self, az_div: int, alt_div: int) -> None:
        self.kin.microsteps_az = int(az_div)
        self.kin.microsteps_alt = int(alt_div)
        base = self.J_deg_per_step.copy()
        self.init_from_mechanics()
        self.J_deg_per_step[0, 1] = float(base[0, 1])
        self.J_deg_per_step[1, 0] = float(base[1, 0])

    def note_manual_move(self, axis: Axis, direction: int, steps: int) -> None:
        """Update step counter when the app executes a MOVE."""
        s = float(abs(int(steps)))
        s *= +1.0 if int(direction) >= 0 else -1.0
        if axis == Axis.AZ:
            self.steps_est[0] += s
        else:
            self.steps_est[1] += s

    def predict_az_alt_deg(self, *, from_ref: bool = False) -> np.ndarray:
        """Predict current mount AZ/ALT from the model + steps."""
        if from_ref or (not self.synced):
            return self.ref_az_alt_deg.copy()
        dsteps = self.steps_est - self.ref_steps
        daltaz = self.J_deg_per_step @ dsteps
        az = wrap_deg_360(self.ref_az_alt_deg[0] + float(daltaz[0]))
        alt = float(self.ref_az_alt_deg[1] + float(daltaz[1]))
        return np.array([az, alt], dtype=np.float64)

    def current_az_alt_deg(self) -> Optional[np.ndarray]:
        """Best estimate of current mount AZ/ALT."""
        if self.last_solve_az_alt_deg is not None:
            return self.last_solve_az_alt_deg.copy()
        if not self.synced:
            return None
        return self.predict_az_alt_deg()

    def apply_plate_solve(self, az_alt_deg: np.ndarray) -> bool:
        """Update last solve and reconcile steps_est with the solved AltAz."""
        az_alt_deg = as_array2(az_alt_deg)
        self.last_solve_az_alt_deg = az_alt_deg.copy()
        self.last_solve_time = time.time()

        if not self.synced:
            return False

        daltaz = np.array(
            [
                wrap_deg_180(float(az_alt_deg[0]) - float(self.ref_az_alt_deg[0])),
                float(az_alt_deg[1]) - float(self.ref_az_alt_deg[1]),
            ],
            dtype=np.float64,
        )

        try:
            dsteps, *_ = np.linalg.lstsq(self.J_deg_per_step, daltaz, rcond=None)
        except np.linalg.LinAlgError as exc:
            log_error(None, "GoTo: failed to reconcile steps from plate-solve", exc, throttle_s=5.0, throttle_key="goto_steps_reconcile")
            return False

        if not np.all(np.isfinite(dsteps)):
            log_error(None, "GoTo: non-finite steps from plate-solve reconciliation", None, throttle_s=5.0, throttle_key="goto_steps_reconcile_nan")
            return False

        self.steps_est = self.ref_steps + dsteps
        return True

    def add_calibration_sample(self, dsteps: np.ndarray, daltaz_deg: np.ndarray) -> None:
        self._calib_steps.append(as_array2(dsteps))
        self._calib_daltaz.append(as_array2(daltaz_deg))

    def fit_J_from_samples(self, *, min_samples: int = 3, ridge: float = 1e-12) -> bool:
        """Least squares fit of J using accumulated calibration samples."""
        J_new = fit_J_from_samples(self._calib_steps, self._calib_daltaz, min_samples=min_samples, ridge=ridge)
        if J_new is None:
            return False
        self.J_deg_per_step = J_new
        return True


@dataclass
class GoToConfig:
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    sep: SepConfig = field(default_factory=SepConfig)

    alt_min_deg: float = 10.0
    alt_max_deg: float = 90.0

    tol_arcsec: float = 10.0

    max_iters: int = 8
    gain: float = 0.85
    max_step_per_iter: int = 150000
    stages: int = 1
    platesolve_feedback: bool = True

    slew_delay_us_az: int = 1200
    slew_delay_us_alt: int = 1200

    settle_s: float = 0.25

    platesolve_radius_deg_seq: Tuple[Optional[float], ...] = (1.0, 2.5, 5.0)

    solve_near_predicted: bool = True


@dataclass
class GoToStatus:
    ok: bool = False
    status: str = "IDLE"

    iters: int = 0
    err_az_arcsec: float = 0.0
    err_alt_arcsec: float = 0.0

    last_solution: Optional[PlatesolveResult] = None

    def err_norm_arcsec(self) -> float:
        return float(math.hypot(float(self.err_az_arcsec), float(self.err_alt_arcsec)))
