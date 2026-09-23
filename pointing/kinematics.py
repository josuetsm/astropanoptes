"""Mechanical constants of the mount: microsteps in, degrees out.

Nothing here is fitted. The gear reduction is exact by construction, so the
degrees-per-step of each axis is known a priori and the pointing model takes it
as given. What a reduction can be *wrong* about is which reduction is installed,
and that is a discrete question answered by `pointing.estimators.mechanics_check`,
not a continuous parameter to regress.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ap_types import Axis

__all__ = ["MountKinematics"]


@dataclass
class MountKinematics:
    """Mechanical parameters used to compute an initial steps/deg model."""

    # Stepper
    motor_full_steps_per_rev: int = 200

    # Microstepping dividers (what the firmware sets on MS pins: 8/16/32/64)
    microsteps_az: int = 64
    microsteps_alt: int = 64

    # Belt / pulleys
    motor_pulley_teeth: int = 20
    belt_pitch_m: float = 0.002  # GT2

    # Direct mechanical reduction. 45.0 means motor:axis = 45:1.
    # Los dos ejes no son iguales: altitud lleva 90.5:1, asi que su paso
    # nominal es la mitad que el de azimut (1.12" frente a 2.25" a 1/64).
    gear_reduction_az: float | None = 45.0
    gear_reduction_alt: float | None = 90.5

    # Un reductor cicloidal repite su error de transmision de primer orden una
    # vez por vuelta de motor, o sea tantos ciclos por vuelta de salida como
    # indique su reduccion. Va como float justamente porque 90.5 no es entero:
    # redondear a 90 correria el periodo un 0.55% y desfasaria el modelo
    # periodico a las pocas vueltas.
    transmission_lobes_az: float = 45.0
    transmission_lobes_alt: float = 90.5

    # Ring radii (meters), used only when gear_reduction_* is None.
    ring_radius_m_az: float = 0.24
    ring_radius_m_alt: float = 0.235

    # Optional sign convention adjustments (because FWD/REV wiring might invert)
    # +1 means: positive steps => increasing AZ/ALT in degrees.
    axis_sign_az: int = +1
    axis_sign_alt: int = +1

    def ring_teeth(self, axis: Axis) -> float:
        r = float(self.ring_radius_m_az if axis == Axis.AZ else self.ring_radius_m_alt)
        return float((2.0 * math.pi * r) / float(self.belt_pitch_m))

    def gear_reduction(self, axis: Axis) -> float:
        explicit = self.gear_reduction_az if axis == Axis.AZ else self.gear_reduction_alt
        if explicit is not None:
            ratio = float(explicit)
        else:
            ratio = float(self.ring_teeth(axis)) / float(self.motor_pulley_teeth)
        if ratio <= 0.0:
            raise ValueError("invalid gear_reduction")
        return float(ratio)

    def microsteps_per_motor_rev(self, axis: Axis) -> int:
        ms = int(self.microsteps_az if axis == Axis.AZ else self.microsteps_alt)
        return int(self.motor_full_steps_per_rev) * ms

    def steps_per_axis_rev(self, axis: Axis) -> float:
        """Microsteps per full 360° axis revolution."""
        mu = float(self.microsteps_per_motor_rev(axis))
        return float(mu * self.gear_reduction(axis))

    def steps_per_deg(self, axis: Axis) -> float:
        return float(self.steps_per_axis_rev(axis) / 360.0)

    def deg_per_step(self, axis: Axis) -> float:
        spd = float(self.steps_per_deg(axis))
        if spd <= 0:
            raise ValueError("invalid steps_per_deg")
        sign = int(self.axis_sign_az if axis == Axis.AZ else self.axis_sign_alt)
        sign = +1 if sign >= 0 else -1
        return float(sign / spd)

    def transmission_error_period_steps(self, axis: Axis) -> float:
        lobes = float(
            self.transmission_lobes_az
            if axis == Axis.AZ
            else self.transmission_lobes_alt
        )
        if not math.isfinite(lobes) or lobes <= 0.0:
            raise ValueError("invalid transmission_lobes")
        return float(self.steps_per_axis_rev(axis) / lobes)
