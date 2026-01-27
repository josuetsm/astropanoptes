# goto.py
# -*- coding: utf-8 -*-
"""GoTo + calibration for Astropanoptes (Alt-Az, no absolute encoders).

This module is intentionally *self-contained*: it does not import AppRunner.
AppRunner (or any orchestrator) should provide callbacks for:
  - get_live_frame(): -> np.ndarray (uint16 RAW16 Bayer; platesolve will use SEP)
  - move_steps(axis: Axis, direction: int, steps: int, delay_us: int) -> None/str
  - stop() -> None/str
  - (optional) set_tracking_enabled(bool) + tracking_keyframe_reset()

Core idea
---------
We keep an internal estimate of commanded motor steps since the last sync:
  s = [s_az, s_alt]^T

and a local linear kinematic map between *step deltas* and *AltAz deltas*:
  d(altaz_deg) = J_deg_per_step @ dsteps

J starts from mechanics (diagonal) and is refined by calibration using
plate-solves after randomized dithers (least squares fit).

A GoTo is done as a closed-loop:
  1) estimate current mount AltAz (from last solve, otherwise from model)
  2) compute desired target AltAz (at current time/location)
  3) convert error (deg) -> correction steps via inv(J)
  4) move (MOVE blocking, per axis)
  5) plate-solve near the predicted center to measure the new AltAz
  6) iterate until tolerance (default 10 arcsec) or max iters

Notes
-----
- Your firmware's MOVE command is blocking and temporarily zeros RATE internally.
  We still recommend disabling tracking (AppRunner tracking loop) during GoTo,
  then re-enabling and resetting keyframe once arrived.
- Because the mount can rotate 360° in AZ, we always choose the shortest AZ
  error (wrap to [-180, +180]).
- ALT is constrained to a safe range (default 10..90 deg).

"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ap_types import Axis
from config import SepConfig

# We reuse target parsing & observer from your plate-solver module.
from platesolve import (
    ObserverConfig,
    PlatesolveConfig,
    PlatesolveResult,
    platesolve_from_frame,
    parse_target_to_icrs,
)
from logging_utils import log_error, log_info

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time


# ============================================================
# Types
# ============================================================

TargetType = Union[
    SkyCoord,
    Tuple[float, float],
    Tuple[str, str],
    str,
    Dict[str, Any],
]


# ============================================================
# Helpers
# ============================================================

def _wrap_deg_180(x: float) -> float:
    """Wrap degrees to (-180, 180]."""
    y = (float(x) + 180.0) % 360.0 - 180.0
    # put -180 at +180 for consistency
    if y <= -180.0:
        y += 360.0
    return float(y)


def _wrap_deg_360(x: float) -> float:
    """Wrap degrees to [0, 360)."""
    y = float(x) % 360.0
    if y < 0.0:
        y += 360.0
    return float(y)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _norm2(a: np.ndarray) -> float:
    return float(np.sqrt(float(np.sum(a * a))))


def _as_array2(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError("expected a 2-vector")
    return a


def _now_time() -> Time:
    # astropy Time uses UTC by default
    return Time.now()


_BRIGHT_START_STARS: Tuple[Dict[str, float | str], ...] = (
    {"name": "Sirius", "ra_deg": 101.28715533, "dec_deg": -16.71611586, "gmag": -1.46},
    {"name": "Canopus", "ra_deg": 95.987877, "dec_deg": -52.695661, "gmag": -0.74},
    {"name": "Arcturus", "ra_deg": 213.915300, "dec_deg": 19.182409, "gmag": -0.05},
    {"name": "Vega", "ra_deg": 279.234735, "dec_deg": 38.783689, "gmag": 0.03},
    {"name": "Capella", "ra_deg": 79.172328, "dec_deg": 45.997991, "gmag": 0.08},
    {"name": "Rigel", "ra_deg": 78.634467, "dec_deg": -8.201638, "gmag": 0.12},
    {"name": "Procyon", "ra_deg": 114.825493, "dec_deg": 5.224993, "gmag": 0.38},
    {"name": "Betelgeuse", "ra_deg": 88.792939, "dec_deg": 7.407064, "gmag": 0.50},
    {"name": "Aldebaran", "ra_deg": 68.980163, "dec_deg": 16.509302, "gmag": 0.86},
    {"name": "Antares", "ra_deg": 247.351917, "dec_deg": -26.432003, "gmag": 0.96},
    {"name": "Spica", "ra_deg": 201.298248, "dec_deg": -11.161323, "gmag": 0.98},
    {"name": "Fomalhaut", "ra_deg": 344.412750, "dec_deg": -29.621837, "gmag": 1.16},
    {"name": "Achernar", "ra_deg": 24.428600, "dec_deg": -57.236800, "gmag": 0.46},
    {"name": "Acrux", "ra_deg": 186.649563, "dec_deg": -63.099093, "gmag": 0.77},
)


def pick_bright_start_star(
    observer: ObserverConfig,
    obstime: Optional[Time],
    *,
    min_alt_deg: float = 15.0,
) -> Optional[Dict[str, float | str]]:
    """Pick a bright, currently visible star to use for the first sync."""
    if obstime is None:
        obstime = _now_time()

    altaz_frame = AltAz(obstime=obstime, location=observer.location())
    candidates: List[Dict[str, float | str]] = []
    fallback: List[Dict[str, float | str]] = []

    for star in _BRIGHT_START_STARS:
        coord = SkyCoord(
            ra=float(star["ra_deg"]) * u.deg,
            dec=float(star["dec_deg"]) * u.deg,
            frame="icrs",
        )
        altaz = coord.transform_to(altaz_frame)
        alt_deg = float(altaz.alt.deg)
        az_deg = float(altaz.az.deg)
        payload: Dict[str, float | str] = {
            "name": str(star["name"]),
            "ra_deg": float(star["ra_deg"]),
            "dec_deg": float(star["dec_deg"]),
            "gmag": float(star["gmag"]),
            "alt_deg": alt_deg,
            "az_deg": az_deg,
        }
        if alt_deg > 0.0:
            fallback.append(payload)
        if alt_deg >= float(min_alt_deg):
            candidates.append(payload)

    if candidates:
        return max(candidates, key=lambda item: float(item["alt_deg"]))
    if fallback:
        return max(fallback, key=lambda item: float(item["alt_deg"]))
    return None


# ============================================================
# Kinematics + model
# ============================================================

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

    # Ring radii (meters)
    ring_radius_m_az: float = 0.24
    ring_radius_m_alt: float = 0.235

    # Optional sign convention adjustments (because FWD/REV wiring might invert)
    # +1 means: positive steps => increasing AZ/ALT in degrees.
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
    """Internal pointing model.

    Coordinates:
      - Steps are *commanded* microsteps from firmware MOVE (per axis).
      - Angles are mount AltAz in degrees.

    Mapping:
      d_altaz = J_deg_per_step @ d_steps
      where d_altaz = [d_az_deg, d_alt_deg]^T and d_steps = [d_az, d_alt]^T.
    """

    kin: MountKinematics = field(default_factory=MountKinematics)

    # J (2x2): deg per step
    J_deg_per_step: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))

    # Reference (sync)
    synced: bool = False
    ref_steps: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ref_az_alt_deg: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))  # [az, alt]

    # Current estimated step counter (relative, but we store absolute in same units as ref_steps)
    steps_est: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    # Last successful plate-solve (used as the best estimate of mount AZ/ALT)
    last_solve_az_alt_deg: Optional[np.ndarray] = None
    last_solve_time: float = 0.0

    # Calibration samples (for updating J)
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
        # Keep cross-coupling terms, but rescale the diagonal baseline.
        # If you prefer, you can call init_from_mechanics() to reset fully.
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
        """Predict current mount AZ/ALT from the model + steps.

        If from_ref=True, returns ref_az_alt_deg.
        """
        if from_ref or (not self.synced):
            return self.ref_az_alt_deg.copy()
        dsteps = self.steps_est - self.ref_steps
        daltaz = self.J_deg_per_step @ dsteps
        az = _wrap_deg_360(self.ref_az_alt_deg[0] + float(daltaz[0]))
        alt = float(self.ref_az_alt_deg[1] + float(daltaz[1]))
        return np.array([az, alt], dtype=np.float64)

    def current_az_alt_deg(self) -> Optional[np.ndarray]:
        """Best estimate of current mount AZ/ALT.

        Prefers last successful plate-solve, otherwise the model prediction.
        """
        if self.last_solve_az_alt_deg is not None:
            return self.last_solve_az_alt_deg.copy()
        if not self.synced:
            return None
        return self.predict_az_alt_deg()

    def add_calibration_sample(self, dsteps: np.ndarray, daltaz_deg: np.ndarray) -> None:
        self._calib_steps.append(_as_array2(dsteps))
        self._calib_daltaz.append(_as_array2(daltaz_deg))

    def fit_J_from_samples(self, *, min_samples: int = 3, ridge: float = 1e-12) -> bool:
        """Least squares fit of J using accumulated calibration samples.

        We solve D = S @ B and set J = B^T (so that d = J @ s).

        Returns True if an update was applied.
        """
        if len(self._calib_steps) < int(min_samples):
            return False
        S = np.stack(self._calib_steps, axis=0)  # (N,2)
        D = np.stack(self._calib_daltaz, axis=0)  # (N,2)

        # Ridge-regularized least squares: minimize ||S B - D||^2 + ridge||B||^2
        # Implemented by augmenting S and D.
        if ridge > 0:
            lam = float(ridge)
            S_aug = np.vstack([S, math.sqrt(lam) * np.eye(2)])
            D_aug = np.vstack([D, np.zeros((2, 2), dtype=np.float64)])
        else:
            S_aug, D_aug = S, D

        B, *_ = np.linalg.lstsq(S_aug, D_aug, rcond=None)
        J_new = B.T

        # sanity: avoid singular / crazy values
        if not np.all(np.isfinite(J_new)):
            return False

        self.J_deg_per_step = J_new.astype(np.float64)
        return True


# ============================================================
# GoTo config + status
# ============================================================

@dataclass
class GoToConfig:
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    sep: SepConfig = field(default_factory=SepConfig)

    # Safe operating window
    alt_min_deg: float = 10.0
    alt_max_deg: float = 90.0

    # GoTo tolerance
    tol_arcsec: float = 10.0

    # Closed-loop parameters
    max_iters: int = 8
    gain: float = 0.85
    max_step_per_iter: int = 150000  # hard clamp (microsteps)

    # MOVE speed (blocking). delay_us ~ 1e6 / microsteps_per_s.
    slew_delay_us_az: int = 1200
    slew_delay_us_alt: int = 1200

    settle_s: float = 0.25

    # Platesolve retry strategy (expands search radius)
    # None => use cfg.search_radius_deg (or its default estimate)
    platesolve_radius_deg_seq: Tuple[Optional[float], ...] = (1.0, 2.5, 5.0)

    # After each correction iteration, solve near:
    #   - predicted center (recommended)
    #   - or directly at target
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


# ============================================================
# Target resolution
# ============================================================

_PLANET_NAMES = {
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "moon",
}


def _looks_like_planet_target(target: TargetType) -> Optional[str]:
    if isinstance(target, dict):
        for k in ("planet", "body"):
            if k in target:
                name = str(target[k]).strip().lower()
                if name in _PLANET_NAMES:
                    return name
    if isinstance(target, str):
        name = target.strip().lower()
        if name in _PLANET_NAMES:
            return name
    return None


def resolve_target_icrs(
    target: TargetType,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> SkyCoord:
    """Resolve supported target representations to an ICRS SkyCoord.

    Supported:
      - Anything supported by platesolve.parse_target_to_icrs (name, ra/dec, alt/az dict)
      - Planets (and Moon) by name

    Planet resolution uses astropy's built-in solar system ephemeris.
    """
    if obstime is None:
        obstime = _now_time()

    planet = _looks_like_planet_target(target)
    if planet is not None:
        loc = observer.location()
        with solar_system_ephemeris.set("builtin"):
            c = get_body(planet, obstime, loc)
        return c.icrs

    # Delegate everything else to the plate-solver's parser (includes AltAz dict).
    return parse_target_to_icrs(
        target,
        observer=observer,
        obstime=obstime,
        progress_cb=None,
        simbad_retries=2,
        simbad_backoff_s=0.4,
    ).icrs


def icrs_to_altaz_deg(
    coord_icrs: SkyCoord,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    if obstime is None:
        obstime = _now_time()
    loc = observer.location()
    altaz = coord_icrs.transform_to(AltAz(obstime=obstime, location=loc))
    az = _wrap_deg_360(float(altaz.az.deg))
    alt = float(altaz.alt.deg)
    return np.array([az, alt], dtype=np.float64)


def platesolve_center_to_altaz_deg(
    ra_deg: float,
    dec_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    return icrs_to_altaz_deg(c, observer=observer, obstime=obstime)


# ============================================================
# GoTo controller
# ============================================================

MoveStepsFn = Callable[[Axis, int, int, int], Any]
StopFn = Callable[[], Any]
GetFrameFn = Callable[[], Optional[np.ndarray]]


@dataclass
class GoToController:
    cfg: GoToConfig = field(default_factory=GoToConfig)
    model: GoToModel = field(default_factory=GoToModel)

    def __post_init__(self) -> None:
        # Ensure model has a reasonable initial J
        if self.model.J_deg_per_step is None or self.model.J_deg_per_step.shape != (2, 2):
            self.model.J_deg_per_step = np.eye(2, dtype=np.float64)
        # If it is still identity (common default), initialize from mechanics
        if np.allclose(self.model.J_deg_per_step, np.eye(2)):
            self.model.init_from_mechanics()

    # -------------------------
    # Sync
    # -------------------------

    def sync_from_platesolve(self, sol: PlatesolveResult, *, obstime: Optional[Time] = None) -> bool:
        """Set the mount's absolute AZ/ALT reference using a plate-solve."""
        if not bool(getattr(sol, "success", False)):
            return False

        az_alt = platesolve_center_to_altaz_deg(
            float(sol.center_ra_deg),
            float(sol.center_dec_deg),
            observer=self.cfg.observer,
            obstime=obstime,
        )

        self.model.synced = True
        self.model.ref_steps = self.model.steps_est.copy()
        self.model.ref_az_alt_deg = az_alt.copy()
        self.model.last_solve_az_alt_deg = az_alt.copy()
        self.model.last_solve_time = time.time()
        return True

    # -------------------------
    # Platesolve helper
    # -------------------------

    def _platesolve_live(
        self,
        *,
        get_live_frame: GetFrameFn,
        target_for_solver: TargetType,
        platesolve_cfg: PlatesolveConfig,
        obstime: Optional[Time] = None,
    ) -> PlatesolveResult:
        if obstime is None:
            obstime = _now_time()

        frame = get_live_frame()
        if frame is None:
            return PlatesolveResult(
                success=False,
                status="ERR_NO_FRAME",
                theta_deg=0.0,
                dx_px=0.0,
                dy_px=0.0,
                response=0.0,
                n_inliers=0,
                rms_px=float("inf"),
                center_ra_deg=0.0,
                center_dec_deg=0.0,
                scale_arcsec_per_px=0.0,
                R_2x2=((1.0, 0.0), (0.0, 1.0)),
                t_arcsec=(0.0, 0.0),
                rms_arcsec=float("inf"),
                overlay=[],
                guides=[],
                metrics={"err": 1.0},
            )

        last: Optional[PlatesolveResult] = None
        for rad in self.cfg.platesolve_radius_deg_seq:
            cfg2 = platesolve_cfg
            if rad is not None:
                try:
                    cfg2 = replace(platesolve_cfg, search_radius_deg=float(rad))
                except Exception as exc:
                    log_info(
                        None,
                        f"GoTo: failed to apply platesolve radius override ({rad}); using default",
                        throttle_s=5.0,
                        throttle_key="goto_radius_fallback",
                    )
                    log_error(
                        None,
                        "GoTo: platesolve config override failed",
                        exc,
                        throttle_s=5.0,
                        throttle_key="goto_radius_fallback_exc",
                    )
                    cfg2 = platesolve_cfg

            res = platesolve_from_frame(
                frame,
                target=target_for_solver,
                cfg=cfg2,
                sep_cfg=self.cfg.sep,
                observer=self.cfg.observer,
                obstime=obstime,
                progress_cb=None,
            )
            last = res
            if bool(getattr(res, "success", False)):
                return res

        # give last attempt
        assert last is not None
        return last

    # -------------------------
    # GoTo (blocking)
    # -------------------------

    def goto_blocking(
        self,
        target: TargetType,
        *,
        get_live_frame: GetFrameFn,
        platesolve_cfg: PlatesolveConfig,
        move_steps: MoveStepsFn,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        obstime: Optional[Time] = None,
    ) -> GoToStatus:
        """Closed-loop GoTo (blocking).

        Intended to be executed in a dedicated thread by AppRunner.
        """
        st = GoToStatus(ok=False, status="RUNNING")

        if not self.model.synced:
            st.status = "ERR_NOT_SYNCED"
            return st

        # Disable tracking while slewing
        was_tracking = False
        if tracking_pause is not None:
            try:
                tracking_pause(True)
                was_tracking = True
            except Exception as exc:
                log_error(None, "GoTo: failed to pause tracking", exc)

        try:
            # Resolve target once to ICRS; we will recompute AltAz each iter.
            if obstime is None:
                obstime = _now_time()

            target_icrs = resolve_target_icrs(target, observer=self.cfg.observer, obstime=obstime)

            # Check visibility / safe altitude.
            altaz_now = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)
            if not (self.cfg.alt_min_deg <= float(altaz_now[1]) <= self.cfg.alt_max_deg):
                st.status = "ERR_TARGET_OUT_OF_RANGE"
                return st

            # Iterate corrections
            for it in range(int(self.cfg.max_iters)):
                st.iters = it + 1
                obstime = _now_time()

                # target altaz at current time
                altaz_tgt = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)

                # current mount altaz best estimate
                altaz_cur = self.model.current_az_alt_deg()
                if altaz_cur is None:
                    st.status = "ERR_NO_CURRENT"
                    return st

                # error in degrees (shortest az)
                daz = _wrap_deg_180(float(altaz_tgt[0]) - float(altaz_cur[0]))
                dalt = float(altaz_tgt[1]) - float(altaz_cur[1])

                st.err_az_arcsec = float(daz * 3600.0)
                st.err_alt_arcsec = float(dalt * 3600.0)

                if (abs(st.err_az_arcsec) <= float(self.cfg.tol_arcsec)) and (
                    abs(st.err_alt_arcsec) <= float(self.cfg.tol_arcsec)
                ):
                    st.ok = True
                    st.status = "OK"
                    return st

                # Convert error -> steps using inverse J.
                d_altaz_vec = np.array([daz, dalt], dtype=np.float64)
                J = self.model.J_deg_per_step
                try:
                    invJ = np.linalg.inv(J)
                except np.linalg.LinAlgError as exc:
                    log_error(None, "GoTo: singular J matrix during solve", exc, throttle_s=5.0, throttle_key="goto_invJ")
                    st.status = "ERR_SINGULAR_MODEL"
                    return st

                dsteps = invJ @ d_altaz_vec

                # Apply gain and clamp.
                dsteps *= float(self.cfg.gain)

                # Hard clamp per iteration (infinity norm)
                dsteps = np.clip(
                    dsteps,
                    -float(self.cfg.max_step_per_iter),
                    +float(self.cfg.max_step_per_iter),
                )

                # Predict after move to enforce ALT bounds.
                # (We clamp ALT delta if needed. AZ is free.)
                pred_after = altaz_cur.copy()
                pred_after[0] = _wrap_deg_360(float(pred_after[0]) + float((J @ dsteps)[0]))
                pred_after[1] = float(pred_after[1]) + float((J @ dsteps)[1])

                if pred_after[1] < float(self.cfg.alt_min_deg) or pred_after[1] > float(self.cfg.alt_max_deg):
                    # Scale down ALT component only.
                    # Equivalent to scaling dsteps along the column that affects ALT.
                    # For robustness, just linearly scale dsteps to bring ALT into range.
                    alt_target = _clamp(pred_after[1], self.cfg.alt_min_deg, self.cfg.alt_max_deg)
                    delta_alt_allowed = float(alt_target - float(altaz_cur[1]))

                    # Solve for a scale alpha on dsteps such that ALT change matches allowed.
                    dalt_pred = float((J @ dsteps)[1])
                    if abs(dalt_pred) > 1e-12:
                        alpha = float(delta_alt_allowed / dalt_pred)
                        alpha = _clamp(alpha, -1.0, 1.0)
                        dsteps *= alpha

                # Execute movement (blocking) per axis.
                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed before move", exc)

                self._exec_steps(move_steps, Axis.AZ, float(dsteps[0]), delay_us=int(self.cfg.slew_delay_us_az))
                self._exec_steps(move_steps, Axis.ALT, float(dsteps[1]), delay_us=int(self.cfg.slew_delay_us_alt))

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed after move", exc)

                # Settle
                time.sleep(max(0.0, float(self.cfg.settle_s)))

                # Plate-solve to update absolute mount AZ/ALT.
                if self.cfg.solve_near_predicted:
                    # Use predicted center to keep solve radius small.
                    altaz_pred = self.model.predict_az_alt_deg()
                    target_for_solver: TargetType = {"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])}
                else:
                    target_for_solver = target

                sol = self._platesolve_live(
                    get_live_frame=get_live_frame,
                    target_for_solver=target_for_solver,
                    platesolve_cfg=platesolve_cfg,
                    obstime=obstime,
                )
                st.last_solution = sol

                if bool(getattr(sol, "success", False)):
                    az_alt_new = platesolve_center_to_altaz_deg(
                        float(sol.center_ra_deg),
                        float(sol.center_dec_deg),
                        observer=self.cfg.observer,
                        obstime=obstime,
                    )
                    self.model.last_solve_az_alt_deg = az_alt_new
                    self.model.last_solve_time = time.time()
                else:
                    # If solve fails, fall back to model prediction but keep iterating.
                    # You can also choose to abort here.
                    self.model.last_solve_az_alt_deg = self.model.predict_az_alt_deg()
                    self.model.last_solve_time = time.time()

            st.status = "ERR_MAX_ITERS"
            return st

        finally:
            # Restore tracking
            if was_tracking and tracking_pause is not None:
                try:
                    tracking_pause(False)
                except Exception as exc:
                    log_error(None, "GoTo: failed to resume tracking", exc)
                if tracking_keyframe_reset is not None:
                    try:
                        tracking_keyframe_reset()
                    except Exception as exc:
                        log_error(None, "GoTo: failed to reset tracking keyframe", exc)

    def _exec_steps(self, move_steps: MoveStepsFn, axis: Axis, signed_steps: float, *, delay_us: int) -> None:
        s = int(round(float(signed_steps)))
        if s == 0:
            return
        direction = +1 if s >= 0 else -1
        steps = abs(s)

        # Update model counter first (best effort even if move fails)
        self.model.note_manual_move(axis, direction, steps)

        # Perform the actual move
        move_steps(axis, direction, steps, int(delay_us))

    # -------------------------
    # Calibration (blocking)
    # -------------------------

    def calibrate_blocking(
        self,
        *,
        get_live_frame: GetFrameFn,
        platesolve_cfg: PlatesolveConfig,
        move_steps: MoveStepsFn,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        step_magnitudes_deg: Sequence[float] = (1.0, 5.0, 10.0),
        step_magnitudes_steps: Optional[Sequence[int]] = None,
        samples_per_mag: int = 1,
        obstime: Optional[Time] = None,
    ) -> Dict[str, Any]:
        """Refine the model J (including cross-coupling) via randomized dithers.

        Preconditions:
          - You should have synced once with a successful plate-solve.

        Procedure:
          - For each magnitude in step_magnitudes_deg:
              * choose a random direction in (AZ,ALT)
              * convert to steps using current diagonal scale
              * move
              * plate-solve near predicted center
              * measure delta AltAz
              * add sample
          - Fit J via least squares

        Returns a dict with summary + fitted matrix.
        """
        out: Dict[str, Any] = {
            "ok": False,
            "n_samples": 0,
            "J_deg_per_step": None,
            "status": "RUNNING",
        }

        if not self.model.synced:
            out["status"] = "ERR_NOT_SYNCED"
            return out

        # Disable tracking while calibrating
        was_tracking = False
        if tracking_pause is not None:
            try:
                tracking_pause(True)
                was_tracking = True
            except Exception as exc:
                log_error(None, "GoTo: failed to pause tracking (calibration)", exc)

        try:
            if obstime is None:
                obstime = _now_time()

            # Need a starting solve to define a baseline altaz.
            altaz0 = self.model.current_az_alt_deg()
            if altaz0 is None:
                out["status"] = "ERR_NO_CURRENT"
                return out

            # Ensure we have a recent solve; if not, do one near prediction.
            # (This keeps calibration stable if you manually moved without a new solve.)
            if self.model.last_solve_az_alt_deg is None:
                altaz_pred = self.model.predict_az_alt_deg()
                sol0 = self._platesolve_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolve_cfg=platesolve_cfg,
                    obstime=obstime,
                )
                if not bool(getattr(sol0, "success", False)):
                    out["status"] = "ERR_PLATESOLVE_BASE"
                    return out
                altaz0 = platesolve_center_to_altaz_deg(
                    float(sol0.center_ra_deg),
                    float(sol0.center_dec_deg),
                    observer=self.cfg.observer,
                    obstime=obstime,
                )
                self.model.last_solve_az_alt_deg = altaz0

            # Run samples
            use_steps = (step_magnitudes_steps is not None) and (len(step_magnitudes_steps) > 0)
            mags: Sequence[Union[float, int]] = step_magnitudes_steps if use_steps else step_magnitudes_deg

            for mag in mags:
                for _ in range(int(samples_per_mag)):
                    # Random direction
                    ang = random.uniform(0.0, 2.0 * math.pi)

                    J = self.model.J_deg_per_step

                    if use_steps:
                        # magnitude directly in step-space (what you asked for)
                        mag_steps = float(mag)
                        dsteps = np.array(
                            [mag_steps * math.cos(ang), mag_steps * math.sin(ang)],
                            dtype=np.float64,
                        )
                    else:
                        # magnitude in degrees (converted to steps using the current model)
                        mag_deg = float(mag)
                        daz_deg = mag_deg * math.cos(ang)
                        dalt_deg = mag_deg * math.sin(ang)

                        try:
                            invJ = np.linalg.inv(J)
                        except np.linalg.LinAlgError as exc:
                            log_error(None, "GoTo: singular J matrix during calibration; resetting mechanics", exc, throttle_s=5.0, throttle_key="goto_calib_invJ")
                            # fall back to diagonal mechanics
                            self.model.init_from_mechanics()
                            J = self.model.J_deg_per_step
                            invJ = np.linalg.inv(J)

                        dsteps = invJ @ np.array([daz_deg, dalt_deg], dtype=np.float64)

                    # Commanded steps are integers; use the same for prediction + sampling
                    dsteps = np.array([float(int(round(dsteps[0]))), float(int(round(dsteps[1])))], dtype=np.float64)
                    if int(dsteps[0]) == 0 and int(dsteps[1]) == 0:
                        continue

                    # Predict and enforce ALT safe range by flipping ALT sign if needed
                    altaz_cur = self.model.current_az_alt_deg()
                    if altaz_cur is None:
                        out["status"] = "ERR_NO_CURRENT"
                        return out

                    pred_after = altaz_cur.copy()
                    pred_after[0] = _wrap_deg_360(float(pred_after[0]) + float((J @ dsteps)[0]))
                    pred_after[1] = float(pred_after[1]) + float((J @ dsteps)[1])
                    if pred_after[1] < float(self.cfg.alt_min_deg) or pred_after[1] > float(self.cfg.alt_max_deg):
                        # flip the ALT component
                        dsteps[1] *= -1.0
                        pred_after[1] = float(altaz_cur[1]) + float((J @ dsteps)[1])
                        pred_after[1] = _clamp(pred_after[1], self.cfg.alt_min_deg, self.cfg.alt_max_deg)

                    if stop is not None:
                        try:
                            stop()
                        except Exception as exc:
                            log_error(None, "GoTo: stop failed before calibration move", exc)

                    # Move
                    self._exec_steps(move_steps, Axis.AZ, float(dsteps[0]), delay_us=int(self.cfg.slew_delay_us_az))
                    self._exec_steps(move_steps, Axis.ALT, float(dsteps[1]), delay_us=int(self.cfg.slew_delay_us_alt))

                    if stop is not None:
                        try:
                            stop()
                        except Exception as exc:
                            log_error(None, "GoTo: stop failed after calibration move", exc)

                    time.sleep(max(0.0, float(self.cfg.settle_s)))

                    # Plate-solve near predicted center (recommended)
                    altaz_pred = self.model.predict_az_alt_deg()
                    sol = self._platesolve_live(
                        get_live_frame=get_live_frame,
                        target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                        platesolve_cfg=platesolve_cfg,
                        obstime=_now_time(),
                    )
                    if not bool(getattr(sol, "success", False)):
                        # skip sample
                        continue

                    altaz_new = platesolve_center_to_altaz_deg(
                        float(sol.center_ra_deg),
                        float(sol.center_dec_deg),
                        observer=self.cfg.observer,
                        obstime=_now_time(),
                    )

                    # Measured delta (wrap az)
                    daltaz_meas = np.array(
                        [
                            _wrap_deg_180(float(altaz_new[0]) - float(altaz_cur[0])),
                            float(altaz_new[1]) - float(altaz_cur[1]),
                        ],
                        dtype=np.float64,
                    )

                    # Measured step delta (what we commanded this sample)
                    dsteps_meas = np.array([float(dsteps[0]), float(dsteps[1])], dtype=np.float64)

                    self.model.add_calibration_sample(dsteps_meas, daltaz_meas)
                    self.model.last_solve_az_alt_deg = altaz_new
                    self.model.last_solve_time = time.time()

            # Fit
            ok = self.model.fit_J_from_samples(min_samples=3)
            out["ok"] = bool(ok)
            out["n_samples"] = int(len(self.model._calib_steps))
            out["J_deg_per_step"] = self.model.J_deg_per_step.copy().tolist()
            out["status"] = "OK" if ok else "ERR_INSUFFICIENT_SAMPLES"
            return out

        finally:
            # Restore tracking
            if was_tracking and tracking_pause is not None:
                try:
                    tracking_pause(False)
                except Exception as exc:
                    log_error(None, "GoTo: failed to resume tracking (calibration)", exc)
                if tracking_keyframe_reset is not None:
                    try:
                        tracking_keyframe_reset()
                    except Exception as exc:
                        log_error(None, "GoTo: failed to reset tracking keyframe (calibration)", exc)


# ============================================================
# Convenience: initial model builder (for your exact mount)
# ============================================================

def make_default_goto_controller_for_your_mount() -> GoToController:
    """Factory using the mechanical parameters you provided.

    AZ: 20T motor pulley -> GT2 ring radius 24 cm
    ALT: 20T motor pulley -> GT2 ring radius 23.5 cm
    Microstepping defaults to 1/64.
    """
    kin = MountKinematics(
        motor_full_steps_per_rev=200,
        microsteps_az=64,
        microsteps_alt=64,
        motor_pulley_teeth=20,
        belt_pitch_m=0.002,
        ring_radius_m_az=0.24,
        ring_radius_m_alt=0.235,
        axis_sign_az=+1,
        axis_sign_alt=+1,
    )
    model = GoToModel(kin=kin)
    model.init_from_mechanics()
    cfg = GoToConfig(observer=ObserverConfig(lat_deg=-33.4489, lon_deg=-70.6693, height_m=520.0))
    return GoToController(cfg=cfg, model=model)
