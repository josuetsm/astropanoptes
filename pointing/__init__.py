"""Pointing model for an alt-az mount without absolute encoders.

The mechanics are known, not fitted: the gear reduction fixes how many degrees
an axis turns per microstep, exactly, because teeth do not slip. What is
genuinely unknown is how the mount sits in the world and how it deviates from
that ideal, and each of those deviations is a named term with a unit, a bound
and a procedure that measures it.
"""
from __future__ import annotations

from pointing.kinematics import MountKinematics

__all__ = ["MountKinematics"]
