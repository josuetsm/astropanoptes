# goto.py
# -*- coding: utf-8 -*-
"""GoTo + calibration facade for Astropanoptes (Alt-Az, no absolute encoders).

This module keeps the public API stable while delegating implementation to
smaller submodules for easier maintenance and debugging.
"""
from __future__ import annotations

from goto_controller import (
    GoToController,
    pick_bright_start_star,
    resolve_target_icrs,
    make_default_goto_controller_for_your_mount,
)
from goto_math import (
    as_array2,
    clamp,
    icrs_to_altaz_deg,
    norm2,
    now_time,
    platesolve_center_to_altaz_deg,
    wrap_deg_180,
    wrap_deg_360,
)
from goto_types import (
    GoToConfig,
    GoToModel,
    GoToStatus,
    MountKinematics,
    TargetType,
)

__all__ = [
    "GoToController",
    "GoToConfig",
    "GoToModel",
    "GoToStatus",
    "MountKinematics",
    "TargetType",
    "pick_bright_start_star",
    "resolve_target_icrs",
    "icrs_to_altaz_deg",
    "platesolve_center_to_altaz_deg",
    "wrap_deg_180",
    "wrap_deg_360",
    "clamp",
    "norm2",
    "as_array2",
    "now_time",
    "make_default_goto_controller_for_your_mount",
]
