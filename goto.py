# goto.py
# -*- coding: utf-8 -*-
"""GoTo + calibration for Astropanoptes (Alt-Az, no absolute encoders).

This module is intentionally *self-contained*: it does not import AppRunner.
AppRunner (or any orchestrator) should provide callbacks for:
  - get_live_frame(): -> np.ndarray (uint16 RAW16 Bayer; platesolving will use SEP)
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
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ap_types import (
    AppState,
    Axis,
    Frame,
    GotoAutocalStatus,
    GotoStatus,
    PlatesolvingStatus,
)
from protocols import StatePublisherProtocol
from config import SepConfig, PlatesolvingConfig, MountConfig, CameraConfig

# We reuse target parsing & observer from your plate-solver module.
from platesolving import (
    ObserverConfig,
    PlatesolvingResult,
    parse_target_to_icrs,
    solve_plate,
    _build_platesolving_debug_info,
    _render_platesolving_debug_jpeg,
)
from logging_utils import log_error, log_info
from workers import BaseWorker
from imaging import ensure_raw16_bayer
from sep_utils import sep_detect_from_raw16, estimate_shift_from_objects

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

    def apply_plate_solve(self, az_alt_deg: np.ndarray) -> bool:
        """Update last solve and reconcile steps_est with the solved AltAz.

        Returns True if steps_est was updated from the solve.
        """
        az_alt_deg = _as_array2(az_alt_deg)
        self.last_solve_az_alt_deg = az_alt_deg.copy()
        self.last_solve_time = time.time()

        if not self.synced:
            return False

        daltaz = np.array(
            [
                _wrap_deg_180(float(az_alt_deg[0]) - float(self.ref_az_alt_deg[0])),
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
    stages: int = 1
    platesolving_feedback: bool = True

    # MOVE speed (blocking). delay_us ~ 1e6 / microsteps_per_s.
    slew_delay_us_az: int = 1200
    slew_delay_us_alt: int = 1200

    settle_s: float = 0.25

    # Platesolving retry strategy (expands search radius)
    # None => use cfg.search_radius_deg (or its default estimate)
    platesolving_radius_deg_seq: Tuple[Optional[float], ...] = (1.0, 2.5, 5.0)

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

    last_solution: Optional[PlatesolvingResult] = None

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
      - Anything supported by platesolving.parse_target_to_icrs (name, ra/dec, alt/az dict)
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


def platesolving_center_to_altaz_deg(
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

    def sync_from_platesolving(self, sol: PlatesolvingResult, *, obstime: Optional[Time] = None) -> bool:
        """Set the mount's absolute AZ/ALT reference using a plate-solve."""
        if not bool(getattr(sol, "success", False)):
            return False

        az_alt = platesolving_center_to_altaz_deg(
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
    # Platesolving helper
    # -------------------------

    def _platesolving_live(
        self,
        *,
        get_live_frame: GetFrameFn,
        target_for_solver: TargetType,
        platesolving_cfg: PlatesolvingConfig,
        radius_deg_seq: Optional[Tuple[Optional[float], ...]] = None,
        obstime: Optional[Time] = None,
    ) -> PlatesolvingResult:
        if obstime is None:
            obstime = _now_time()

        frame = get_live_frame()
        if frame is None:
            return PlatesolvingResult(
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

        last: Optional[PlatesolvingResult] = None
        radius_seq = radius_deg_seq if radius_deg_seq is not None else self.cfg.platesolving_radius_deg_seq
        for rad in radius_seq:
            cfg2 = platesolving_cfg
            if rad is not None:
                try:
                    cfg2 = replace(platesolving_cfg, search_radius_deg=float(rad))
                except Exception as exc:
                    log_info(
                        None,
                        f"GoTo: failed to apply platesolving radius override ({rad}); using default",
                        throttle_s=5.0,
                        throttle_key="goto_radius_fallback",
                    )
                    log_error(
                        None,
                        "GoTo: platesolving config override failed",
                        exc,
                        throttle_s=5.0,
                        throttle_key="goto_radius_fallback_exc",
                    )
                    cfg2 = platesolving_cfg

            res = solve_plate(
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
        platesolving_cfg: PlatesolvingConfig,
        move_steps: MoveStepsFn,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        stages: int = 1,
        platesolving_feedback: bool = True,
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

        stages = max(1, int(stages))
        use_platesolving_feedback = bool(platesolving_feedback)

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
            for it in range(stages):
                st.iters = it + 1
                obstime = _now_time()

                # target altaz at current time
                altaz_tgt = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)

                # current mount altaz best estimate
                if use_platesolving_feedback:
                    altaz_cur = self.model.current_az_alt_deg()
                else:
                    altaz_cur = self.model.predict_az_alt_deg()
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

                remaining = max(1, stages - it)
                stage_scale = 1.0 / float(remaining)
                dsteps *= stage_scale

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

                if use_platesolving_feedback:
                    sol = self._platesolving_live(
                        get_live_frame=get_live_frame,
                        target_for_solver=target_for_solver,
                        platesolving_cfg=platesolving_cfg,
                        obstime=obstime,
                    )
                    st.last_solution = sol

                    if bool(getattr(sol, "success", False)):
                        az_alt_new = platesolving_center_to_altaz_deg(
                            float(sol.center_ra_deg),
                            float(sol.center_dec_deg),
                            observer=self.cfg.observer,
                            obstime=obstime,
                        )
                        self.model.apply_plate_solve(az_alt_new)
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
        platesolving_cfg: PlatesolvingConfig,
        move_steps: MoveStepsFn,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        n_samples: int = 3,
        max_radius_deg: float = 1.0,
        obstime: Optional[Time] = None,
    ) -> Dict[str, Any]:
        """Refine the model J (including cross-coupling) via randomized dithers.

        Preconditions:
          - You should have synced once with a successful plate-solve.

        Procedure:
          - For each sample:
              * choose a random direction and radius within max_radius_deg
              * convert to steps using current J
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

        calib_platesolving_cfg = replace(
            platesolving_cfg,
            search_radius_deg=1.0,
            gmax=15.0,
            nside=16,
        )

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
                sol0 = self._platesolving_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolving_cfg=calib_platesolving_cfg,
                    radius_deg_seq=(1.0,),
                    obstime=obstime,
                )
                if not bool(getattr(sol0, "success", False)):
                    out["status"] = "ERR_PLATESOLVING_BASE"
                    return out
                altaz0 = platesolving_center_to_altaz_deg(
                    float(sol0.center_ra_deg),
                    float(sol0.center_dec_deg),
                    observer=self.cfg.observer,
                    obstime=obstime,
                )
                self.model.apply_plate_solve(altaz0)

            # Run samples
            max_radius = float(max_radius_deg)
            if max_radius <= 0.0:
                out["status"] = "ERR_BAD_RADIUS"
                return out
            total_samples = int(max(1, n_samples))

            for _ in range(total_samples):
                # Random direction + radius (uniform over area)
                ang = random.uniform(0.0, 2.0 * math.pi)
                radius = math.sqrt(random.random()) * max_radius

                daz_deg = radius * math.cos(ang)
                dalt_deg = radius * math.sin(ang)

                J = self.model.J_deg_per_step
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
                sol = self._platesolving_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolving_cfg=calib_platesolving_cfg,
                    radius_deg_seq=(1.0,),
                    obstime=_now_time(),
                )
                if not bool(getattr(sol, "success", False)):
                    # skip sample
                    continue

                altaz_new = platesolving_center_to_altaz_deg(
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
                self.model.apply_plate_solve(altaz_new)

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
# Worker (threaded orchestration)
# ============================================================

def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()


@dataclass(frozen=True)
class _AutocalFrame:
    raw16: np.ndarray
    t_capture: float
    obj_xy: np.ndarray
    star_count: int
    saturation_frac: float
    top_sources: Tuple[Tuple[float, float, float], ...]


@dataclass(frozen=True)
class _AutocalJResult:
    col: Optional[np.ndarray]
    ok_count: int
    resp_low: int
    missing_frames: int


class GoToWorker(BaseWorker):
    """
    Background worker for GoTo / calibration / autocalibration.

    Dependencies injected:
      - get_state(): AppState snapshot
      - publish_state(patch): publish UI state
      - get_frame(): latest Frame (or None)
      - get_goto_cfg(): GoToConfig snapshot
      - get_mount_cfg(): MountConfig snapshot
      - get_sep_cfg(): SepConfig snapshot
      - get_camera_cfg(): CameraConfig snapshot
      - get_platesolving_cfg(): PlatesolvingConfig snapshot
      - get_observer(): ObserverConfig snapshot
      - apply_camera_param(name, value): set camera parameter
      - pause_tracking()/resume_tracking()
      - pause_stacking()/resume_stacking()
      - rate_mount(az_rate, alt_rate)
      - move_steps(axis, direction, steps, delay_us)
      - stop_mount()
    """

    def __init__(
        self,
        *,
        goto_controller: GoToController,
        get_state: Callable[[], AppState],
        publish_state: StatePublisherProtocol,
        get_frame: Callable[[], Optional[Frame]],
        get_goto_cfg: Callable[[], GoToConfig],
        get_mount_cfg: Callable[[], MountConfig],
        get_sep_cfg: Callable[[], SepConfig],
        get_camera_cfg: Callable[[], CameraConfig],
        get_platesolving_cfg: Callable[[], PlatesolvingConfig],
        get_observer: Callable[[], ObserverConfig],
        apply_camera_param: Callable[[str, Any], None],
        pause_tracking: Callable[[], bool],
        resume_tracking: Callable[[], None],
        pause_stacking: Callable[[], bool],
        resume_stacking: Callable[[], None],
        rate_mount: Callable[[float, float], Any],
        move_steps: Callable[[Axis, int, int, int], Any],
        stop_mount: Callable[[], Any],
        out_log: Any = None,
    ) -> None:
        super().__init__(name="GoToWorker")
        self._goto = goto_controller
        self._get_state = get_state
        self._publish_state = publish_state
        self._get_frame = get_frame
        self._get_goto_cfg = get_goto_cfg
        self._get_mount_cfg = get_mount_cfg
        self._get_sep_cfg = get_sep_cfg
        self._get_camera_cfg = get_camera_cfg
        self._get_platesolving_cfg = get_platesolving_cfg
        self._get_observer = get_observer
        self._apply_camera_param = apply_camera_param
        self._pause_tracking = pause_tracking
        self._resume_tracking = resume_tracking
        self._pause_stacking = pause_stacking
        self._resume_stacking = resume_stacking
        self._rate_mount = rate_mount
        self._move_steps = move_steps
        self._stop_mount = stop_mount
        self._out_log = out_log
        self._op_cancel = threading.Event()

    def request(self, *, kind: str, target: Any, params: Dict[str, Any]) -> None:
        super().request(kind=str(kind), target=target, params=dict(params))

    def cancel(self) -> None:
        self._op_cancel.set()

    def _get_live_raw16(self) -> Optional[np.ndarray]:
        fr = self._get_frame()
        if fr is None:
            return None
        return ensure_raw16_bayer(fr.raw)

    def _frame_seq(self, fr: Frame) -> Optional[int]:
        seq = fr.meta.get("seq")
        if seq is None:
            return None
        return int(seq)

    def _autocal_detect(
        self, raw16: np.ndarray
    ) -> Tuple[np.ndarray, int, float, Tuple[Tuple[float, float, float], ...]]:
        sep_cfg = self._get_sep_cfg()
        platesolving_cfg = self._get_platesolving_cfg()
        img_det, _bkg, objects, obj_xy = sep_detect_from_raw16(
            raw16,
            sep_bw=int(sep_cfg.bw),
            sep_bh=int(sep_cfg.bh),
            sep_thresh_sigma=float(sep_cfg.thresh_sigma),
            sep_minarea=int(sep_cfg.minarea),
            max_sources=int(platesolving_cfg.max_det),
        )
        _ = img_det
        star_count = int(obj_xy.shape[0])
        max_val = np.iinfo(raw16.dtype).max
        saturation_frac = float(np.mean(raw16 >= max_val))
        top_sources: Tuple[Tuple[float, float, float], ...] = ()
        if objects is not None and len(objects) > 0:
            n_use = min(3, len(objects))
            xs = objects["x"][:n_use].astype(np.float64)
            ys = objects["y"][:n_use].astype(np.float64)
            fluxes = objects["flux"][:n_use].astype(np.float64)
            top_sources = tuple(
                (float(xs[i]), float(ys[i]), float(fluxes[i])) for i in range(n_use)
            )
        return obj_xy, star_count, saturation_frac, top_sources

    def _format_autocal_sources(
        self, sources: Sequence[Tuple[float, float, float]]
    ) -> str:
        if not sources:
            return "[]"
        parts = [f"{flux:.1f}@({x:.1f},{y:.1f})" for x, y, flux in sources]
        return "[" + ", ".join(parts) + "]"

    def _autocal_capture_frames(
        self,
        *,
        n_frames: int,
        timeout_s: float,
        min_dt_s: float = 0.0,
        skip_frames: int = 0,
    ) -> List[_AutocalFrame]:
        frames: List[_AutocalFrame] = []
        deadline = _perf() + float(timeout_s)
        last_seq: Optional[int] = None
        last_capture_t: Optional[float] = None
        min_dt_s = max(0.0, float(min_dt_s))
        skip_remaining = max(0, int(skip_frames))
        while len(frames) < int(n_frames) and _perf() < deadline:
            if self._op_cancel.is_set():
                break
            fr = self._get_frame()
            if fr is None:
                time.sleep(0.01)
                continue
            seq = self._frame_seq(fr)
            if last_seq is not None and seq is not None and int(seq) == last_seq:
                time.sleep(0.005)
                continue
            if seq is not None:
                last_seq = int(seq)
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            raw16 = ensure_raw16_bayer(fr.raw).copy()
            obj_xy, star_count, saturation_frac, top_sources = self._autocal_detect(raw16)
            t_capture = float(getattr(fr, "t_capture", _now_s()))
            if last_capture_t is not None and (t_capture - last_capture_t) < min_dt_s:
                time.sleep(0.005)
                continue
            frames.append(
                _AutocalFrame(
                    raw16=raw16,
                    t_capture=t_capture,
                    obj_xy=obj_xy,
                    star_count=star_count,
                    saturation_frac=saturation_frac,
                    top_sources=top_sources,
                )
            )
            last_capture_t = t_capture
        return frames

    def _autocal_adjust_exposure(
        self,
        *,
        star_count: int,
        saturation_frac: float,
        target_min: int,
        target_max: int,
        sat_max: float,
        exp_min_ms: float,
        exp_max_ms: float,
        exp_step: float,
        gain_min: int,
        gain_max: int,
        gain_step: int,
        settle_s: float,
    ) -> bool:
        cam_cfg = self._get_camera_cfg()
        exp_ms = float(getattr(cam_cfg, "exp_ms", 0.0))
        gain = int(getattr(cam_cfg, "gain", 0))

        need_more = int(star_count) < int(target_min)
        need_less = int(star_count) > int(target_max) or float(saturation_frac) > float(sat_max)

        if not need_more and not need_less:
            return False

        if need_more:
            if exp_ms < float(exp_max_ms):
                new_exp = min(float(exp_max_ms), exp_ms * float(exp_step))
                if new_exp != exp_ms:
                    self._apply_camera_param("exp_ms", new_exp)
                    time.sleep(float(settle_s))
                    return True
            if gain < int(gain_max):
                new_gain = min(int(gain_max), gain + int(gain_step))
                if new_gain != gain:
                    self._apply_camera_param("gain", new_gain)
                    time.sleep(float(settle_s))
                    return True
        else:
            if gain > int(gain_min):
                new_gain = max(int(gain_min), gain - int(gain_step))
                if new_gain != gain:
                    self._apply_camera_param("gain", new_gain)
                    time.sleep(float(settle_s))
                    return True
            if exp_ms > float(exp_min_ms):
                new_exp = max(float(exp_min_ms), exp_ms / float(exp_step))
                if new_exp != exp_ms:
                    self._apply_camera_param("exp_ms", new_exp)
                    time.sleep(float(settle_s))
                    return True
        return False

    def _autocal_exposure_in_range(
        self,
        *,
        star_count: int,
        saturation_frac: float,
        target_min: int,
        target_max: int,
        sat_max: float,
    ) -> bool:
        if int(star_count) < int(target_min):
            return False
        if int(star_count) > int(target_max):
            return False
        if float(saturation_frac) > float(sat_max):
            return False
        return True

    def _autocal_estimate_drift(
        self,
        frames: List[_AutocalFrame],
        *,
        dt_min_s: float,
        dt_max_s: float,
        max_pairs: int,
        max_shift_px: float,
        min_resp: float,
    ) -> Optional[np.ndarray]:
        if len(frames) < 2:
            log_info(
                self._out_log,
                f"GoTo: AutoCal drift needs >=2 frames, got {len(frames)}",
            )
            return None
        if len(frames) - 1 > int(max_pairs):
            idx = np.linspace(0, len(frames) - 1, int(max_pairs) + 1).round().astype(int)
            frames = [frames[i] for i in idx]

        ref = frames[0]
        v_list: List[np.ndarray] = []
        resp_list: List[float] = []
        resp_low = 0
        used = 0
        for fr in frames[1:]:
            dt = float(fr.t_capture - ref.t_capture)
            if not (float(dt_min_s) <= dt <= float(dt_max_s)):
                continue
            dx, dy, resp, _n = estimate_shift_from_objects(
                ref.obj_xy,
                fr.obj_xy,
                max_shift_px=float(max_shift_px),
            )
            used += 1
            if float(resp) < float(min_resp):
                resp_low += 1
                continue
            v = np.array([-dx / dt, -dy / dt], dtype=np.float64)
            v_list.append(v)
            resp_list.append(float(resp))

        if len(v_list) < 2:
            resp_min = min(resp_list) if resp_list else 0.0
            resp_med = float(np.median(resp_list)) if resp_list else 0.0
            resp_max = max(resp_list) if resp_list else 0.0
            log_info(
                self._out_log,
                "GoTo: AutoCal drift insufficient valid pairs "
                f"valid={len(v_list)} total={used} resp_low={resp_low} "
                f"resp_stats=[{resp_min:.3f},{resp_med:.3f},{resp_max:.3f}]",
            )
            return None
        drift = np.median(np.stack(v_list, axis=0), axis=0)
        log_info(
            self._out_log,
            "GoTo: AutoCal drift estimate "
            f"vx={float(drift[0]):.3f}px/s vy={float(drift[1]):.3f}px/s "
            f"valid_pairs={len(v_list)}",
        )
        return drift

    def _autocal_pick_best_frame(
        self,
        frames: Sequence[_AutocalFrame],
    ) -> Optional[_AutocalFrame]:
        if not frames:
            return None
        return max(
            frames,
            key=lambda fr: (int(fr.star_count), -float(fr.saturation_frac)),
        )

    def _autocal_axis_rates(self, axis: Axis, rate: float) -> Tuple[float, float]:
        if axis == Axis.AZ:
            return float(rate), 0.0
        return 0.0, float(rate)

    def _autocal_rate_ramp(
        self,
        *,
        axis: Axis,
        start_rate: float,
        end_rate: float,
        ramp_s: float,
        ramp_hz: float,
    ) -> None:
        if self._rate_mount is None:
            return
        ramp_s = max(0.0, float(ramp_s))
        ramp_hz = max(1.0, float(ramp_hz))
        if ramp_s <= 0.0:
            az_rate, alt_rate = self._autocal_axis_rates(axis, float(end_rate))
            self._rate_mount(az_rate, alt_rate)
            return
        steps = max(1, int(round(ramp_s * ramp_hz)))
        for i in range(1, steps + 1):
            if self._op_cancel.is_set():
                break
            f = float(i) / float(steps)
            rate = float(start_rate) + (float(end_rate) - float(start_rate)) * f
            az_rate, alt_rate = self._autocal_axis_rates(axis, rate)
            self._rate_mount(az_rate, alt_rate)
            time.sleep(1.0 / float(ramp_hz))

    def _autocal_axis_rate_scan(
        self,
        *,
        axis: Axis,
        rate_steps_s: float,
        ramp_s: float,
        ramp_hz: float,
        plateau_s: float,
        plateau_frames: int,
        plateau_min_dt_s: float,
        plateau_skip_frames: int,
        drift_pix: np.ndarray,
        max_shift_px: float,
        min_resp: float,
    ) -> _AutocalJResult:
        if self._rate_mount is None:
            return _AutocalJResult(col=None, ok_count=0, resp_low=0, missing_frames=1)

        plateau_frames = max(2, int(plateau_frames))
        plateau_s = max(0.1, float(plateau_s))

        self._autocal_rate_ramp(
            axis=axis,
            start_rate=0.0,
            end_rate=float(rate_steps_s),
            ramp_s=ramp_s,
            ramp_hz=ramp_hz,
        )
        if self._op_cancel.is_set():
            self._autocal_rate_ramp(
                axis=axis,
                start_rate=float(rate_steps_s),
                end_rate=0.0,
                ramp_s=ramp_s,
                ramp_hz=ramp_hz,
            )
            return _AutocalJResult(col=None, ok_count=0, resp_low=0, missing_frames=1)

        frames = self._autocal_capture_frames(
            n_frames=int(plateau_frames),
            timeout_s=float(plateau_s),
            min_dt_s=float(plateau_min_dt_s),
            skip_frames=int(plateau_skip_frames),
        )

        self._autocal_rate_ramp(
            axis=axis,
            start_rate=float(rate_steps_s),
            end_rate=0.0,
            ramp_s=ramp_s,
            ramp_hz=ramp_hz,
        )
        if self._rate_mount is not None:
            self._rate_mount(0.0, 0.0)

        if len(frames) < 2:
            return _AutocalJResult(col=None, ok_count=0, resp_low=0, missing_frames=1)

        base = frames[0]
        cols: List[np.ndarray] = []
        resp_low = 0
        for fr in frames[1:]:
            dt = float(fr.t_capture - base.t_capture)
            if dt <= 0.0:
                continue
            dx, dy, resp, _n = estimate_shift_from_objects(
                base.obj_xy,
                fr.obj_xy,
                max_shift_px=float(max_shift_px),
            )
            if float(resp) < float(min_resp):
                resp_low += 1
                continue
            dp = np.array([-dx, -dy], dtype=np.float64) - drift_pix * dt
            steps = float(rate_steps_s) * dt
            if steps == 0.0:
                continue
            cols.append(dp / steps)

        if not cols:
            log_info(
                self._out_log,
                "GoTo: AutoCal J axis scan "
                f"axis={axis.name} rate={float(rate_steps_s):.2f} "
                f"frames={len(frames)} resp_low={resp_low} ok=0",
            )
            return _AutocalJResult(col=None, ok_count=0, resp_low=resp_low, missing_frames=0)

        col = np.median(np.stack(cols, axis=0), axis=0)
        log_info(
            self._out_log,
            "GoTo: AutoCal J axis scan "
            f"axis={axis.name} rate={float(rate_steps_s):.2f} "
            f"frames={len(frames)} resp_low={resp_low} ok={len(cols)}",
        )
        return _AutocalJResult(col=col, ok_count=len(cols), resp_low=resp_low, missing_frames=0)

    def _goto_autocalibrate_blocking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": False,
            "status": "RUNNING",
            "drift_pix": None,
            "J_pix_per_step": None,
            "pointing_estimate": None,
            "platesolving_result": None,
        }

        st = self._get_state()
        if not bool(st.camera.connected):
            out["status"] = "ERR_NO_CAMERA"
            return out
        if not bool(st.mount.connected):
            out["status"] = "ERR_NO_MOUNT"
            return out

        goto_cfg = self._get_goto_cfg()
        mount_cfg = self._get_mount_cfg()
        platesolving_cfg = self._get_platesolving_cfg()
        sep_cfg = self._get_sep_cfg()
        observer = self._get_observer()

        target_star_min = int(params.get("target_star_min", 3))
        target_star_max = int(params.get("target_star_max", 200))
        target_sat_max = float(params.get("target_sat_max", 0.01))
        exp_min_ms = float(params.get("exp_min_ms", 20.0))
        exp_max_ms = float(params.get("exp_max_ms", 1200.0))
        exp_step = float(params.get("exp_step", 1.5))
        gain_min = int(params.get("gain_min", 0))
        gain_max = int(params.get("gain_max", 600))
        gain_step = int(params.get("gain_step", 50))
        settle_s = float(params.get("settle_s", goto_cfg.settle_s))
        tune_attempts = int(params.get("tune_attempts", 5))
        tune_settle_s = float(params.get("tune_settle_s", 0.4))

        drift_frames = int(params.get("drift_frames", 10))
        drift_dt_min = float(params.get("drift_dt_min_s", 1.0))
        drift_dt_max = float(params.get("drift_dt_max_s", 10.0))
        drift_pairs = int(params.get("drift_pairs", 20))
        drift_max_shift_px = float(params.get("drift_max_shift_px", 25.0))
        drift_min_resp = float(params.get("drift_min_resp", 0.2))
        drift_capture_timeout_s = float(params.get("drift_capture_timeout_s", 12.0))

        jcal_rate_scale = float(params.get("jcal_rate_scale", 1.0))
        jcal_ramp_s = float(params.get("jcal_ramp_s", 0.6))
        jcal_ramp_hz = float(params.get("jcal_ramp_hz", 25.0))
        jcal_plateau_s = float(params.get("jcal_plateau_s", 2.0))
        jcal_plateau_frames = int(params.get("jcal_plateau_frames", 4))
        jcal_plateau_min_dt_s = float(params.get("jcal_plateau_min_dt_s", 0.2))
        jcal_plateau_skip_frames = int(params.get("jcal_plateau_skip_frames", 1))
        jcal_probe_s = float(params.get("jcal_probe_s", 0.5))
        jcal_probe_scale = float(params.get("jcal_probe_scale", 0.35))
        jcal_max_shift_px = float(params.get("jcal_max_shift_px", 30.0))
        jcal_min_resp = float(params.get("jcal_min_resp", 0.2))

        solve_attempts = int(params.get("solve_attempts", 5))
        jitter_deg = float(params.get("solve_jitter_deg", 0.2))

        if self._rate_mount is None:
            out["status"] = "ERR_NO_RATE"
            return out
        if jcal_rate_scale <= 0.0:
            out["status"] = "ERR_JCAL_PARAMS"
            return out
        if jcal_ramp_s < 0.0 or jcal_ramp_hz <= 0.0:
            out["status"] = "ERR_JCAL_PARAMS"
            return out
        if jcal_plateau_s <= 0.0 or jcal_plateau_frames < 2:
            out["status"] = "ERR_JCAL_PARAMS"
            return out
        if jcal_plateau_min_dt_s < 0.0 or jcal_plateau_skip_frames < 0:
            out["status"] = "ERR_JCAL_PARAMS"
            return out
        if jcal_probe_s < 0.0 or jcal_probe_scale < 0.0:
            out["status"] = "ERR_JCAL_PARAMS"
            return out
        if drift_capture_timeout_s <= 0.0:
            out["status"] = "ERR_DRIFT_PARAMS"
            return out

        log_info(
            self._out_log,
            "GoTo: AutoCal config "
            f"stars=[{target_star_min},{target_star_max}] sat_max={target_sat_max:.3f} "
            f"exp_ms=[{exp_min_ms:.1f},{exp_max_ms:.1f}] gain=[{gain_min},{gain_max}] "
            f"drift_frames={drift_frames} drift_dt=[{drift_dt_min:.2f},{drift_dt_max:.2f}] "
            f"drift_pairs={drift_pairs} drift_min_resp={drift_min_resp:.2f} "
            f"drift_capture_timeout_s={drift_capture_timeout_s:.2f} "
            f"jcal_rate_scale={jcal_rate_scale:.2f} "
            f"jcal_ramp_s={jcal_ramp_s:.2f} ramp_hz={jcal_ramp_hz:.1f} "
            f"jcal_plateau_s={jcal_plateau_s:.2f} frames={jcal_plateau_frames} "
            f"plateau_min_dt_s={jcal_plateau_min_dt_s:.2f} skip_frames={jcal_plateau_skip_frames} "
            f"jcal_probe_s={jcal_probe_s:.2f} probe_scale={jcal_probe_scale:.2f} "
            f"jcal_min_resp={jcal_min_resp:.2f} jcal_max_shift_px={jcal_max_shift_px:.1f}",
        )

        plate_scale_rad = float(platesolving_cfg.pixel_size_m) / float(platesolving_cfg.focal_m)
        if not np.isfinite(plate_scale_rad) or plate_scale_rad <= 0.0:
            out["status"] = "ERR_BAD_PLATE_SCALE"
            return out

        self._publish_state(
            {"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "EXPOSURE_TUNE"}}
        )
        tuned = False
        for _ in range(int(tune_attempts)):
            if self._op_cancel.is_set():
                out["status"] = "CANCELLED"
                return out
            frames = self._autocal_capture_frames(n_frames=1, timeout_s=1.5)
            if not frames:
                continue
            fr = frames[0]
            changed = self._autocal_adjust_exposure(
                star_count=fr.star_count,
                saturation_frac=fr.saturation_frac,
                target_min=target_star_min,
                target_max=target_star_max,
                sat_max=target_sat_max,
                exp_min_ms=exp_min_ms,
                exp_max_ms=exp_max_ms,
                exp_step=exp_step,
                gain_min=gain_min,
                gain_max=gain_max,
                gain_step=gain_step,
                settle_s=tune_settle_s,
            )
            if not changed:
                tuned = True
                break
        if not tuned:
            frames = self._autocal_capture_frames(n_frames=1, timeout_s=1.5)
            if frames:
                fr = frames[0]
                tuned = self._autocal_exposure_in_range(
                    star_count=fr.star_count,
                    saturation_frac=fr.saturation_frac,
                    target_min=target_star_min,
                    target_max=target_star_max,
                    sat_max=target_sat_max,
                )
        if not tuned:
            out["status"] = "ERR_EXPOSURE_TUNE"
            log_info(self._out_log, "GoTo: AutoCal exposure tune failed")
            return out
        cam_cfg = self._get_camera_cfg()
        log_info(
            self._out_log,
            "GoTo: AutoCal exposure tuned "
            f"exp_ms={float(getattr(cam_cfg, 'exp_ms', 0.0)):.2f} "
            f"gain={int(getattr(cam_cfg, 'gain', 0))}",
        )

        self._publish_state({"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "DRIFT"}})
        drift_frames_list = self._autocal_capture_frames(
            n_frames=drift_frames,
            timeout_s=drift_capture_timeout_s,
            min_dt_s=drift_dt_min,
        )
        if len(drift_frames_list) < 2:
            out["status"] = "ERR_DRIFT_FRAMES"
            log_info(
                self._out_log,
                f"GoTo: AutoCal drift capture insufficient frames ({len(drift_frames_list)})",
            )
            return out
        star_counts = [fr.star_count for fr in drift_frames_list]
        sat_fracs = [fr.saturation_frac for fr in drift_frames_list]
        log_info(
            self._out_log,
            "GoTo: AutoCal drift frames "
            f"count={len(drift_frames_list)} "
            f"stars=[{min(star_counts)},{int(np.median(star_counts))},{max(star_counts)}] "
            f"sat=[{min(sat_fracs):.3f},{float(np.median(sat_fracs)):.3f},{max(sat_fracs):.3f}]",
        )
        if len(drift_frames_list) > 1:
            ref_t = float(drift_frames_list[0].t_capture)
            for idx, fr in enumerate(drift_frames_list):
                dt = float(fr.t_capture - ref_t)
                log_info(
                    self._out_log,
                    "GoTo: AutoCal drift frame sources "
                    f"idx={idx} dt={dt:.3f}s stars={fr.star_count} "
                    f"top3={self._format_autocal_sources(fr.top_sources)}",
                )
        drift_pix = self._autocal_estimate_drift(
            drift_frames_list,
            dt_min_s=drift_dt_min,
            dt_max_s=drift_dt_max,
            max_pairs=drift_pairs,
            max_shift_px=drift_max_shift_px,
            min_resp=drift_min_resp,
        )
        if drift_pix is None:
            out["status"] = "ERR_DRIFT"
            log_info(self._out_log, "GoTo: AutoCal drift estimate failed")
            return out
        out["drift_pix"] = drift_pix
        self._publish_state(
            {
                "goto": {
                    "autocal_drift_px_s_x": float(drift_pix[0]),
                    "autocal_drift_px_s_y": float(drift_pix[1]),
                }
            }
        )

        self._publish_state({"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "JCAL"}})
        drift_norm = float(np.linalg.norm(drift_pix))
        if drift_norm <= 0.0:
            out["status"] = "ERR_JCAL_RATE"
            return out

        steps_per_rad_az = float(self._goto.model.kin.steps_per_deg(Axis.AZ)) * (180.0 / math.pi)
        steps_per_rad_alt = float(self._goto.model.kin.steps_per_deg(Axis.ALT)) * (180.0 / math.pi)
        v_rad_mag = drift_norm * float(plate_scale_rad)
        rate_az_mag = v_rad_mag * steps_per_rad_az * float(jcal_rate_scale)
        rate_alt_mag = v_rad_mag * steps_per_rad_alt * float(jcal_rate_scale)

        rate_max = float(getattr(mount_cfg, "rate_max", 0.0))
        if rate_max > 0.0:
            rate_az_mag = min(rate_az_mag, rate_max)
            rate_alt_mag = min(rate_alt_mag, rate_max)

        if rate_az_mag <= 0.0 or rate_alt_mag <= 0.0:
            out["status"] = "ERR_JCAL_RATE"
            return out

        def _px_per_step(axis: Axis) -> float:
            dps = abs(float(self._goto.model.kin.deg_per_step(axis)))
            rad_per_step = dps * (math.pi / 180.0)
            return float(rad_per_step / plate_scale_rad)

        def _expected_shift(rate_steps_s: float, duration_s: float, axis: Axis) -> float:
            return abs(float(rate_steps_s)) * float(duration_s) * _px_per_step(axis)

        drift_dir = drift_pix / drift_norm

        def _pick_rate_sign(axis: Axis, rate_mag: float, probe_s: float, probe_scale: float) -> float:
            if probe_s <= 0.0 or probe_scale <= 0.0:
                return 1.0
            probe_rate = float(rate_mag) * float(probe_scale)
            if probe_rate <= 0.0:
                return 1.0
            probe_frames = max(2, int(round(float(jcal_plateau_frames) * 0.5)))
            probe_shift = max(
                float(jcal_max_shift_px),
                _expected_shift(probe_rate, probe_s, axis) * 1.2,
            )
            probe = self._autocal_axis_rate_scan(
                axis=axis,
                rate_steps_s=probe_rate,
                ramp_s=jcal_ramp_s,
                ramp_hz=jcal_ramp_hz,
                plateau_s=probe_s,
                plateau_frames=probe_frames,
                plateau_min_dt_s=jcal_plateau_min_dt_s,
                plateau_skip_frames=jcal_plateau_skip_frames,
                drift_pix=drift_pix,
                max_shift_px=probe_shift,
                min_resp=jcal_min_resp,
            )
            if probe.col is None:
                return 1.0
            return 1.0 if float(np.dot(probe.col, drift_dir)) >= 0.0 else -1.0

        max_shift_az = max(
            float(jcal_max_shift_px),
            _expected_shift(rate_az_mag, jcal_plateau_s, Axis.AZ) * 1.2,
        )
        max_shift_alt = max(
            float(jcal_max_shift_px),
            _expected_shift(rate_alt_mag, jcal_plateau_s, Axis.ALT) * 1.2,
        )

        sign_az = _pick_rate_sign(Axis.AZ, rate_az_mag, jcal_probe_s, jcal_probe_scale)
        sign_alt = _pick_rate_sign(Axis.ALT, rate_alt_mag, jcal_probe_s, jcal_probe_scale)

        col_az_res = self._autocal_axis_rate_scan(
            axis=Axis.AZ,
            rate_steps_s=sign_az * rate_az_mag,
            ramp_s=jcal_ramp_s,
            ramp_hz=jcal_ramp_hz,
            plateau_s=jcal_plateau_s,
            plateau_frames=jcal_plateau_frames,
            plateau_min_dt_s=jcal_plateau_min_dt_s,
            plateau_skip_frames=jcal_plateau_skip_frames,
            drift_pix=drift_pix,
            max_shift_px=max_shift_az,
            min_resp=jcal_min_resp,
        )
        if col_az_res.col is None:
            out["status"] = "ERR_JCAL_AZ"
            return out

        col_alt_res = self._autocal_axis_rate_scan(
            axis=Axis.ALT,
            rate_steps_s=sign_alt * rate_alt_mag,
            ramp_s=jcal_ramp_s,
            ramp_hz=jcal_ramp_hz,
            plateau_s=jcal_plateau_s,
            plateau_frames=jcal_plateau_frames,
            plateau_min_dt_s=jcal_plateau_min_dt_s,
            plateau_skip_frames=jcal_plateau_skip_frames,
            drift_pix=drift_pix,
            max_shift_px=max_shift_alt,
            min_resp=jcal_min_resp,
        )
        if col_alt_res.col is None:
            out["status"] = "ERR_JCAL_ALT"
            return out

        J_pix = np.column_stack([col_az_res.col, col_alt_res.col])
        out["J_pix_per_step"] = J_pix
        self._publish_state(
            {
                "goto": {
                    "autocal_J_pix_per_step_00": float(J_pix[0, 0]),
                    "autocal_J_pix_per_step_01": float(J_pix[0, 1]),
                    "autocal_J_pix_per_step_10": float(J_pix[1, 0]),
                    "autocal_J_pix_per_step_11": float(J_pix[1, 1]),
                }
            }
        )

        self._publish_state(
            {"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "POINTING_SOLVE"}}
        )
        e_az = J_pix[:, 0]
        e_alt = J_pix[:, 1]
        n_az = float(np.linalg.norm(e_az))
        n_alt = float(np.linalg.norm(e_alt))
        if n_az <= 0.0 or n_alt <= 0.0:
            out["status"] = "ERR_JCAL_NORM"
            return out

        e_az_hat = e_az / n_az
        e_alt_hat = e_alt / n_alt

        v_az_pix = float(np.dot(drift_pix, e_az_hat))
        v_alt_pix = float(np.dot(drift_pix, e_alt_hat))
        v_obs = np.array([v_az_pix * plate_scale_rad, v_alt_pix * plate_scale_rad], dtype=np.float64)

        def _wrap_deg_180(x: float) -> float:
            y = (float(x) + 180.0) % 360.0 - 180.0
            if y <= -180.0:
                y += 360.0
            return float(y)

        def _wrap_deg_360(x: float) -> float:
            y = float(x) % 360.0
            if y < 0.0:
                y += 360.0
            return float(y)

        def _predict_rate(az_deg: float, alt_deg: float, t0: Time, dt_s: float = 1.0) -> np.ndarray:
            loc = observer.location()
            altaz0 = AltAz(az=float(az_deg) * u.deg, alt=float(alt_deg) * u.deg, obstime=t0, location=loc)
            coord = SkyCoord(altaz0)
            altaz1 = coord.transform_to(AltAz(obstime=t0 + float(dt_s) * u.s, location=loc))
            daz = _wrap_deg_180(float(altaz1.az.deg) - float(az_deg))
            dalt = float(altaz1.alt.deg) - float(alt_deg)
            return np.array(
                [
                    np.deg2rad(daz) / float(dt_s),
                    np.deg2rad(dalt) / float(dt_s),
                ],
                dtype=np.float64,
            )

        t_ref = Time.now()
        seeds = params.get(
            "pointing_seeds",
            [
                (90.0, 45.0),
                (270.0, 45.0),
                (60.0, 35.0),
                (300.0, 35.0),
            ],
        )
        best = None
        best_res = float("inf")
        for seed in seeds:
            if self._op_cancel.is_set():
                out["status"] = "CANCELLED"
                return out
            az = float(seed[0])
            alt = float(seed[1])
            for _ in range(8):
                pred = _predict_rate(az, alt, t_ref)
                resid = pred - v_obs
                if float(np.linalg.norm(resid)) < 1e-7:
                    break
                delta = 0.1
                pred_az = _predict_rate(az + delta, alt, t_ref)
                pred_alt = _predict_rate(az, alt + delta, t_ref)
                J = np.column_stack([(pred_az - pred) / delta, (pred_alt - pred) / delta])
                if np.linalg.matrix_rank(J) < 2:
                    break
                step = np.linalg.solve(J, -resid)
                step = np.clip(step, -5.0, 5.0)
                az = _wrap_deg_360(az + float(step[0]))
                alt = float(alt + float(step[1]))
                alt = float(np.clip(alt, goto_cfg.alt_min_deg, goto_cfg.alt_max_deg))
            resid_norm = float(np.linalg.norm(_predict_rate(az, alt, t_ref) - v_obs))
            if resid_norm < best_res:
                best_res = resid_norm
                best = (az, alt)

        if best is None:
            out["status"] = "ERR_POINTING_SOLVE"
            return out

        az_hat, alt_hat = best
        dist_to_0 = min(az_hat, 360.0 - az_hat)
        dist_to_180 = abs(az_hat - 180.0)
        if dist_to_0 < 15.0 or dist_to_180 < 15.0:
            out["status"] = "ERR_DEGENERATE_AZ"
            return out

        out["pointing_estimate"] = {"az_deg": az_hat, "alt_deg": alt_hat, "radius_deg": 1.0}
        self._publish_state(
            {
                "goto": {
                    "autocal_az_deg": float(az_hat),
                    "autocal_alt_deg": float(alt_hat),
                    "autocal_radius_deg": 1.0,
                }
            }
        )

        self._publish_state(
            {"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "PLATESOLVING_LOOP"}}
        )
        v_deg_s = np.rad2deg(v_obs)
        jitter_seq = [
            (0.0, 0.0),
            (jitter_deg, 0.0),
            (-jitter_deg, 0.0),
            (0.0, jitter_deg),
            (0.0, -jitter_deg),
        ]
        platesolving_result = None
        attempts = 0
        while attempts < int(solve_attempts):
            if self._op_cancel.is_set():
                out["status"] = "CANCELLED"
                return out
            frames = self._autocal_capture_frames(n_frames=1, timeout_s=1.5)
            if not frames:
                attempts += 1
                continue
            fr = frames[0]
            changed = self._autocal_adjust_exposure(
                star_count=fr.star_count,
                saturation_frac=fr.saturation_frac,
                target_min=target_star_min,
                target_max=target_star_max,
                sat_max=target_sat_max,
                exp_min_ms=exp_min_ms,
                exp_max_ms=exp_max_ms,
                exp_step=exp_step,
                gain_min=gain_min,
                gain_max=gain_max,
                gain_step=gain_step,
                settle_s=tune_settle_s,
            )
            if changed:
                continue

            t_now = Time.now()
            dt_s = float((t_now - t_ref).to_value(u.s))
            az_c = _wrap_deg_360(az_hat + float(v_deg_s[0]) * dt_s)
            alt_c = float(alt_hat + float(v_deg_s[1]) * dt_s)
            alt_c = float(np.clip(alt_c, goto_cfg.alt_min_deg, goto_cfg.alt_max_deg))

            jitter = jitter_seq[attempts % len(jitter_seq)]
            target = {"az_deg": float(az_c + jitter[0]), "alt_deg": float(alt_c + jitter[1])}

            ps_cfg = replace(platesolving_cfg, search_radius_deg=1.0)
            platesolving_result = solve_plate(
                fr.raw16,
                target=target,
                cfg=ps_cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=t_now,
                progress_cb=None,
            )
            attempts += 1

            debug_jpeg = _render_platesolving_debug_jpeg(
                fr.raw16,
                list(getattr(platesolving_result, "overlay", []) or []),
            )
            debug_info = _build_platesolving_debug_info(platesolving_result)

            ps_ok = bool(getattr(platesolving_result, "success", False))
            ps_reason = str(getattr(platesolving_result, "status", "UNKNOWN"))
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.OK if ps_ok else PlatesolvingStatus.FAIL,
                        "reason": None if ps_ok else ps_reason,
                        "last_ok": ps_ok,
                        "theta_deg": float(getattr(platesolving_result, "theta_deg", 0.0)),
                        "dx_px": float(getattr(platesolving_result, "dx_px", 0.0)),
                        "dy_px": float(getattr(platesolving_result, "dy_px", 0.0)),
                        "resp": float(getattr(platesolving_result, "response", 0.0)),
                        "n_inliers": int(getattr(platesolving_result, "n_inliers", 0)),
                        "rms_px": float(getattr(platesolving_result, "rms_px", 0.0)),
                        "overlay": list(getattr(platesolving_result, "overlay", []) or []),
                        "guides": list(getattr(platesolving_result, "guides", []) or []),
                        "debug_jpeg": debug_jpeg,
                        "debug_info": debug_info,
                        "center_ra_deg": float(getattr(platesolving_result, "center_ra_deg", 0.0)),
                        "center_dec_deg": float(getattr(platesolving_result, "center_dec_deg", 0.0)),
                    }
                }
            )

            if ps_ok:
                self._publish_state({"platesolving_result": platesolving_result})
                break

        if platesolving_result is None or not bool(getattr(platesolving_result, "success", False)):
            out["status"] = "ERR_PLATESOLVING"
            out["platesolving_result"] = platesolving_result
            return out

        out["platesolving_result"] = platesolving_result
        self._publish_state({"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "SYNC"}})
        ok_sync = bool(self._goto.sync_from_platesolving(platesolving_result))
        self._publish_state(
            {
                "goto": {
                    "synced": ok_sync,
                    "status": GotoStatus.OK if ok_sync else GotoStatus.FAIL,
                    "reason": None if ok_sync else "SYNC_FAILED",
                }
            }
        )
        if not ok_sync:
            out["status"] = "ERR_SYNC"
            return out

        self._publish_state(
            {"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "CALIBRATE_J"}}
        )

        def _get_live_frame_for_calib() -> Optional[np.ndarray]:
            return self._get_live_raw16()

        calib_out = self._goto.calibrate_blocking(
            get_live_frame=_get_live_frame_for_calib,
            move_steps=self._move_steps,
            stop=self._stop_mount,
            platesolving_cfg=platesolving_cfg,
            n_samples=int(params.get("calib_samples", goto_cfg.calib_samples)),
            max_radius_deg=float(params.get("calib_max_radius_deg", goto_cfg.calib_max_radius_deg)),
        )
        if not bool(calib_out.get("ok", False)):
            out["status"] = "ERR_CALIBRATE_J"
            return out

        out["ok"] = True
        out["status"] = "OK"
        self._publish_state(
            {
                "goto": {
                    "autocal_last_ok": True,
                    "autocal_status": GotoAutocalStatus.OK,
                    "autocal_reason": "READY",
                }
            }
        )
        return out

    def _publish_j_matrix_state(self) -> None:
        J = getattr(self._goto.model, "J_deg_per_step", None)
        if J is None or getattr(J, "shape", None) != (2, 2):
            log_error(self._out_log, "GoTo: J matrix unavailable or invalid", ValueError("invalid J matrix"))
            return
        self._publish_state(
            {
                "goto": {
                    "J00": float(J[0, 0]),
                    "J01": float(J[0, 1]),
                    "J10": float(J[1, 0]),
                    "J11": float(J[1, 1]),
                    "synced": bool(getattr(self._goto.model, "synced", False)),
                }
            }
        )

    def _handle_request(self, request: Dict[str, Any]) -> None:
        kind = str(request.get("kind", "goto"))
        target = request.get("target", None)
        params = dict(request.get("params", {}) or {})

        was_tracking = self._pause_tracking()
        was_stacking = self._pause_stacking()

        self._publish_state(
            {"goto": {"busy": True, "status": GotoStatus.RUNNING, "reason": str(kind)}}
        )
        self._op_cancel.clear()

        goto_cfg = self._get_goto_cfg()
        platesolving_cfg = self._get_platesolving_cfg()

        try:
            if kind == "goto":
                delay_us = int(params.get("delay_us", goto_cfg.slew_delay_us))
                tol_arcsec = float(params.get("tol_arcsec", goto_cfg.tol_arcsec))
                stages = int(params.get("stages", goto_cfg.stages))
                platesolving_feedback = bool(params.get("platesolving_feedback", goto_cfg.platesolving_feedback))
                gain = float(params.get("gain", goto_cfg.gain))
                max_step_per_iter = int(goto_cfg.max_step_per_iter)
                if "max_step_per_iter" in params:
                    max_step_per_iter = int(params.get("max_step_per_iter"))
                else:
                    max_step_deg = float(params.get("max_step_deg", 5.0))
                    j_matrix = self._goto.model.J_deg_per_step
                    max_abs_deg_per_step = float(np.max(np.abs(j_matrix))) if j_matrix is not None and j_matrix.size else 0.0
                    if max_abs_deg_per_step > 0.0:
                        max_step_per_iter = int(max(1, round(max_step_deg / max_abs_deg_per_step)))

                self._goto.cfg = replace(
                    self._goto.cfg,
                    tol_arcsec=tol_arcsec,
                    max_iters=stages,
                    gain=gain,
                    max_step_per_iter=max_step_per_iter,
                    slew_delay_us_az=delay_us,
                    slew_delay_us_alt=delay_us,
                    stages=stages,
                    platesolving_feedback=platesolving_feedback,
                )

                status = self._goto.goto_blocking(
                    target,
                    get_live_frame=self._get_live_raw16,
                    move_steps=self._move_steps,
                    stop=self._stop_mount,
                    platesolving_cfg=platesolving_cfg,
                    stages=stages,
                    platesolving_feedback=platesolving_feedback,
                )
                err_norm = float(status.err_norm_arcsec())
                self._publish_state(
                    {
                        "goto": {
                            "last_error_arcsec": err_norm,
                            "status": GotoStatus.OK if status.ok else GotoStatus.FAIL,
                            "reason": str(status.status),
                        }
                    }
                )
                if status.ok:
                    log_info(
                        self._out_log,
                        f"GoTo: OK status={status.status} iters={status.iters} err_arcsec={err_norm:.2f}",
                    )
                else:
                    log_info(
                        self._out_log,
                        f"GoTo: ERR status={status.status} iters={status.iters} err_arcsec={err_norm:.2f}",
                    )

            elif kind == "calibrate":
                if "n_samples" not in params and "samples" in params:
                    params["n_samples"] = params.get("samples")
                if "max_radius_deg" not in params and "radius_deg" in params:
                    params["max_radius_deg"] = params.get("radius_deg")

                delay_us = int(params.get("delay_us", goto_cfg.slew_delay_us))
                n_samples = int(params.get("n_samples", goto_cfg.calib_samples))
                max_radius_deg = float(params.get("max_radius_deg", goto_cfg.calib_max_radius_deg))

                self._goto.cfg = replace(
                    self._goto.cfg,
                    slew_delay_us_az=delay_us,
                    slew_delay_us_alt=delay_us,
                )

                calib_out = self._goto.calibrate_blocking(
                    get_live_frame=self._get_live_raw16,
                    move_steps=self._move_steps,
                    stop=self._stop_mount,
                    platesolving_cfg=platesolving_cfg,
                    n_samples=n_samples,
                    max_radius_deg=max_radius_deg,
                )
                calib_ok = bool(calib_out.get("ok", False))
                calib_status = str(calib_out.get("status", "UNKNOWN"))
                calib_samples = int(calib_out.get("n_samples", 0))
                self._publish_state(
                    {
                        "goto": {
                            "status": GotoStatus.OK if calib_ok else GotoStatus.FAIL,
                            "reason": f"CALIBRATE_{calib_status}",
                        }
                    }
                )
                log_info(
                    self._out_log,
                    f"GoTo: CALIBRATE status={calib_status} ok={calib_ok} samples={calib_samples}",
                )

            elif kind == "autocal":
                self._publish_state({"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "RUNNING"}})
                autocal_out = self._goto_autocalibrate_blocking(params)
                autocal_ok = bool(autocal_out.get("ok", False))
                autocal_reason = "READY" if autocal_ok else str(autocal_out.get("status", "UNKNOWN"))
                self._publish_state(
                    {
                        "goto": {
                            "status": GotoStatus.OK if autocal_ok else GotoStatus.FAIL,
                            "reason": "AUTOCAL",
                            "autocal_last_ok": autocal_ok,
                            "autocal_status": GotoAutocalStatus.OK if autocal_ok else GotoAutocalStatus.FAIL,
                            "autocal_reason": autocal_reason,
                        }
                    }
                )
                log_info(
                    self._out_log,
                    f"GoTo: AUTOCAL status={autocal_out.get('status')} ok={autocal_ok}",
                )

            else:
                self._publish_state(
                    {"goto": {"status": GotoStatus.FAIL, "reason": f"ERR_KIND_{kind}"}}
                )

            self._publish_j_matrix_state()

        except (RuntimeError, ValueError, TypeError) as exc:
            log_error(self._out_log, f"GoTo worker failed ({kind})", exc)
            self._publish_state({"goto": {"status": GotoStatus.FAIL, "reason": "EXCEPTION"}})

        finally:
            if was_stacking:
                self._resume_stacking()
            if was_tracking:
                self._resume_tracking()
            self._publish_state({"goto": {"busy": False}})


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
