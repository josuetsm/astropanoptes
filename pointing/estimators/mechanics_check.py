"""Which gear reduction is installed? A discrete question, not a regression.

The reduction of a cycloidal drive is exact by construction: teeth do not slip,
so degrees-per-step is not a free parameter and fitting it was the bug this
package exists to remove. What a configuration *can* get wrong is which reducer
is bolted to the axis. Between 45:1 and 90.5:1 there is a factor of two, and a
factor of two is settled by a hypothesis test, not by least squares that lands
somewhere in between and gets clipped to a bound.

This happened for real. `gear_reduction_alt` went from 45.0 to 90.5 when the
altitude reducer was replaced, and the saved samples show the change cleanly:
0.000639 deg/step before, 0.000339 after. Nothing warned; the constant and the
hardware simply had to be kept in sync by hand. This check is what notices.

Why a whole transmission cycle
------------------------------
The cycloidal transmission error is a first harmonic in step count with period
`kin.transmission_error_period_steps` (one motor revolution). Over a whole
number of cycles it cancels exactly, so the move measures the mean reduction and
nothing else. Over a fraction of a cycle it does not cancel, and the leftover is
a *bias*, not noise -- averaging more short moves does not remove it. Short moves
are still usable here, because the hypotheses differ by 2x and the bias is
bounded, but the bias is propagated into the uncertainty so that a move too short
to decide reports that it cannot decide instead of guessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np

from ap_types import Axis

__all__ = [
    "AxisMove",
    "MechanicsOutcome",
    "MechanicsVerdict",
    "axis_angle_between_positions_deg",
    "verify_reduction",
]


class MechanicsOutcome(str, Enum):
    """Why the check ended the way it did."""

    CONFIRMED = "MECH_CONFIRMED"
    """One candidate fits decisively, and it is the configured one."""

    CONTRADICTED = "MECH_CONTRADICTED"
    """One candidate fits decisively, and it is *not* the configured one."""

    UNDETERMINED = "MECH_UNDETERMINED"
    """No candidate wins by the required margin. Move further, or a whole cycle."""

    UNMODELLED = "MECH_UNMODELLED"
    """The best candidate still does not describe the data: it is neither."""

    NOT_ENOUGH_DATA = "MECH_NOT_ENOUGH_DATA"
    """Fewer usable moves than required."""


@dataclass(frozen=True)
class AxisMove:
    """One single-axis move bracketed by two accepted plate solves.

    ``angle_deg`` is the rotation of the *axis*, already recovered from the two
    sky positions by :func:`axis_angle_between_positions_deg`, and carries the
    same sign as ``d_steps``. ``sigma_deg`` is the 1-sigma of that angle from
    the two solves alone; the transmission bias is added by the verifier, which
    is the only place that knows the cycle period.
    """

    axis: Axis
    d_steps: float
    angle_deg: float
    sigma_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.d_steps) or self.d_steps == 0.0:
            raise ValueError("d_steps must be a non-zero finite number")
        if not math.isfinite(self.angle_deg):
            raise ValueError("angle_deg must be finite")
        if not math.isfinite(self.sigma_deg) or self.sigma_deg <= 0.0:
            raise ValueError("sigma_deg must be positive and finite")


@dataclass(frozen=True)
class MechanicsVerdict:
    """The answer, with every number needed to argue with it."""

    axis: Axis
    outcome: MechanicsOutcome
    configured_ratio: float
    accepted_ratio: Optional[float]
    measured_deg_per_step: float
    measured_sigma: float
    measured_ratio: float = float("nan")
    """Reduction implied by the measurement alone, whatever the candidates are.

    Reported so that a reducer matching *neither* candidate is visible as a
    number instead of being forced into the nearest one.
    """
    chi2: Dict[float, float] = field(default_factory=dict)
    n_moves: int = 0
    min_cycle_fraction: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True only when the hardware matches the configuration."""
        return self.outcome is MechanicsOutcome.CONFIRMED


def axis_angle_between_positions_deg(
    *,
    axis: Axis,
    az0_deg: float,
    alt0_deg: float,
    az1_deg: float,
    alt1_deg: float,
) -> float:
    """Rotation of the axis implied by two solved sky positions, in degrees.

    Rotating about the ALT axis carries the optical axis along a great circle by
    exactly the axis angle, so the angular separation between the two positions
    *is* the answer. Rotating about the AZ axis carries it along a small circle
    of radius ``cos(alt)``, so the separation is compressed by that factor and
    has to be undone.

    Both forms ignore the base tilt, which is bounded at ~2 deg and enters here
    at second order. That is immaterial against hypotheses a factor of two
    apart, and this function is not used for anything finer.
    """
    p0 = _unit_vector(az0_deg, alt0_deg)
    p1 = _unit_vector(az1_deg, alt1_deg)
    sep_rad = float(np.arccos(float(np.clip(np.dot(p0, p1), -1.0, 1.0))))

    if axis == Axis.ALT:
        return float(np.degrees(sep_rad))

    cos_alt = math.cos(math.radians(0.5 * (float(alt0_deg) + float(alt1_deg))))
    if cos_alt <= 1e-6:
        raise ValueError("azimuth moves carry no information at the zenith")
    ratio = math.sin(0.5 * sep_rad) / cos_alt
    if ratio > 1.0:
        # Geometrically impossible for a pure azimuth rotation; the solves or the
        # altitude disagree. Report the separation rather than an arcsin domain
        # error, and let the chi2 test reject it.
        return float(np.degrees(sep_rad))
    return float(np.degrees(2.0 * math.asin(ratio)))


def verify_reduction(
    moves: Sequence[AxisMove],
    *,
    axis: Axis,
    configured_ratio: float,
    candidate_ratios: Sequence[float],
    microsteps: int,
    motor_full_steps_per_rev: int,
    cycle_period_steps: float,
    max_periodic_error_deg: float = 0.25,
    min_moves: int = 2,
    chi2_margin: float = 25.0,
    max_reduced_chi2: float = 9.0,
) -> MechanicsVerdict:
    """Decide which candidate reduction the moves are consistent with.

    ``chi2_margin`` of 25 is odds of order 1e5 between the two best candidates:
    this either answers unambiguously or declines to answer. ``max_reduced_chi2``
    of 9 is 3 sigma per point, and guards the case the margin cannot see -- that
    the winner is merely the least bad of two wrong hypotheses.
    """
    usable = [m for m in moves if m.axis == axis]
    ratios = [float(r) for r in candidate_ratios if float(r) > 0.0]
    if not ratios:
        raise ValueError("at least one candidate reduction is required")

    if len(usable) < int(min_moves):
        return MechanicsVerdict(
            axis=axis,
            outcome=MechanicsOutcome.NOT_ENOUGH_DATA,
            configured_ratio=float(configured_ratio),
            accepted_ratio=None,
            measured_deg_per_step=float("nan"),
            measured_sigma=float("nan"),
            n_moves=len(usable),
            detail=f"se necesitan {int(min_moves)} movimientos en {axis.value}, hay {len(usable)}",
        )

    d_steps = np.array([m.d_steps for m in usable], dtype=np.float64)
    angles = np.array([m.angle_deg for m in usable], dtype=np.float64)

    # The periodic term cancels over a whole cycle and not otherwise. What is
    # left over is a bias bounded by the peak-to-peak of the harmonic across the
    # phase span the move covers, so a whole-cycle move costs nothing and a short
    # one pays for itself in uncertainty.
    period = abs(float(cycle_period_steps))
    fractions = np.abs(d_steps) / period if period > 0.0 else np.ones_like(d_steps)
    bias = 2.0 * float(max_periodic_error_deg) * np.abs(np.sin(np.pi * fractions))
    sigma_solve = np.array([m.sigma_deg for m in usable], dtype=np.float64)
    sigma = np.sqrt(sigma_solve**2 + bias**2)

    chi2: Dict[float, float] = {}
    for ratio in ratios:
        k = _deg_per_step(ratio, microsteps, motor_full_steps_per_rev)
        # Compare magnitudes: which way a positive step turns the axis is a sign
        # convention living in the kinematics, and a reduction hypothesis says
        # nothing about it.
        residual = np.abs(angles) - k * np.abs(d_steps)
        chi2[ratio] = float(np.sum((residual / sigma) ** 2))

    weights = 1.0 / sigma**2
    denom = float(np.sum(weights * d_steps**2))
    measured = float(np.sum(weights * np.abs(d_steps) * np.abs(angles)) / denom) if denom > 0 else float("nan")
    measured_sigma = float(np.sqrt(1.0 / denom)) if denom > 0 else float("nan")

    ordered = sorted(chi2.items(), key=lambda kv: kv[1])
    best_ratio, best_chi2 = ordered[0]
    dof = max(1, len(usable) - 1)

    common = dict(
        axis=axis,
        configured_ratio=float(configured_ratio),
        measured_deg_per_step=measured,
        measured_sigma=measured_sigma,
        chi2=chi2,
        measured_ratio=(
            360.0 / (float(motor_full_steps_per_rev) * float(microsteps) * measured)
            if measured > 0.0 and math.isfinite(measured)
            else float("nan")
        ),
        n_moves=len(usable),
        min_cycle_fraction=float(np.min(fractions)),
    )

    if best_chi2 / dof > float(max_reduced_chi2):
        return MechanicsVerdict(
            outcome=MechanicsOutcome.UNMODELLED,
            accepted_ratio=None,
            detail=(
                f"ningun candidato describe los datos: chi2/dof={best_chi2 / dof:.1f}; "
                f"medido {measured:.8f} deg/paso"
            ),
            **common,
        )

    if len(ordered) > 1 and (ordered[1][1] - best_chi2) < float(chi2_margin):
        return MechanicsVerdict(
            outcome=MechanicsOutcome.UNDETERMINED,
            accepted_ratio=None,
            detail=(
                f"{ordered[0][0]:g}:1 y {ordered[1][0]:g}:1 no se separan "
                f"(dchi2={ordered[1][1] - best_chi2:.1f} < {float(chi2_margin):.0f}); "
                f"mueve un ciclo completo ({period:.0f} pasos)"
            ),
            **common,
        )

    confirmed = math.isclose(best_ratio, float(configured_ratio), rel_tol=1e-9)
    return MechanicsVerdict(
        outcome=MechanicsOutcome.CONFIRMED if confirmed else MechanicsOutcome.CONTRADICTED,
        accepted_ratio=best_ratio,
        detail=(
            f"{best_ratio:g}:1"
            + ("" if confirmed else f", pero la configuracion dice {float(configured_ratio):g}:1")
            + f"; medido {measured:.8f} deg/paso"
        ),
        **common,
    )


def _deg_per_step(ratio: float, microsteps: int, motor_full_steps_per_rev: int) -> float:
    steps_per_axis_rev = float(motor_full_steps_per_rev) * float(microsteps) * float(ratio)
    return 360.0 / steps_per_axis_rev


def _unit_vector(az_deg: float, alt_deg: float) -> np.ndarray:
    az = math.radians(float(az_deg))
    alt = math.radians(float(alt_deg))
    return np.array(
        [math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)],
        dtype=np.float64,
    )
