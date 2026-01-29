# goto_controller.py
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time

from ap_types import Axis
from logging_utils import log_error, log_info
from platesolve import (
    ObserverConfig,
    PlatesolveConfig,
    PlatesolveResult,
    platesolve_sweep,
    parse_target_to_icrs,
)

from goto_math import (
    clamp,
    icrs_to_altaz_deg,
    now_time,
    platesolve_center_to_altaz_deg,
    wrap_deg_180,
    wrap_deg_360,
)
from goto_stages import plan_stage_move
from goto_types import GoToConfig, GoToModel, GoToStatus, MountKinematics, TargetType


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
        obstime = now_time()

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
    """Resolve supported target representations to an ICRS SkyCoord."""
    if obstime is None:
        obstime = now_time()

    planet = _looks_like_planet_target(target)
    if planet is not None:
        loc = observer.location()
        with solar_system_ephemeris.set("builtin"):
            c = get_body(planet, obstime, loc)
        return c.icrs

    return parse_target_to_icrs(
        target,
        observer=observer,
        obstime=obstime,
    ).icrs


MoveStepsFn = Callable[[Axis, int, int, int], Any]
StopFn = Callable[[], Any]
GetFrameFn = Callable[[], Optional[np.ndarray]]


@dataclass
class GoToController:
    cfg: GoToConfig = field(default_factory=GoToConfig)
    model: GoToModel = field(default_factory=GoToModel)

    def __post_init__(self) -> None:
        if self.model.J_deg_per_step is None or self.model.J_deg_per_step.shape != (2, 2):
            self.model.J_deg_per_step = np.eye(2, dtype=np.float64)
        if np.allclose(self.model.J_deg_per_step, np.eye(2)):
            self.model.init_from_mechanics()

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

    def _platesolve_live(
        self,
        *,
        get_live_frame: GetFrameFn,
        target_for_solver: TargetType,
        platesolve_cfg: PlatesolveConfig,
        radius_deg_seq: Optional[Tuple[Optional[float], ...]] = None,
        obstime: Optional[Time] = None,
    ) -> PlatesolveResult:
        if obstime is None:
            obstime = now_time()

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
        radius_seq = radius_deg_seq if radius_deg_seq is not None else self.cfg.platesolve_radius_deg_seq
        for rad in radius_seq:
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

            res = platesolve_sweep(
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

        assert last is not None
        return last

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
        stages: int = 1,
        platesolve_feedback: bool = True,
        obstime: Optional[Time] = None,
    ) -> GoToStatus:
        """Closed-loop GoTo (blocking)."""
        st = GoToStatus(ok=False, status="RUNNING")

        if not self.model.synced:
            st.status = "ERR_NOT_SYNCED"
            return st

        was_tracking = False
        if tracking_pause is not None:
            try:
                tracking_pause(True)
                was_tracking = True
            except Exception as exc:
                log_error(None, "GoTo: failed to pause tracking", exc)

        stages = max(1, int(stages))
        use_platesolve_feedback = bool(platesolve_feedback)

        try:
            if obstime is None:
                obstime = now_time()

            target_icrs = resolve_target_icrs(target, observer=self.cfg.observer, obstime=obstime)

            altaz_now = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)
            if not (self.cfg.alt_min_deg <= float(altaz_now[1]) <= self.cfg.alt_max_deg):
                st.status = "ERR_TARGET_OUT_OF_RANGE"
                return st

            for it in range(stages):
                st.iters = it + 1
                obstime = now_time()

                altaz_tgt = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)

                if use_platesolve_feedback:
                    altaz_cur = self.model.current_az_alt_deg()
                else:
                    altaz_cur = self.model.predict_az_alt_deg()
                if altaz_cur is None:
                    st.status = "ERR_NO_CURRENT"
                    return st

                try:
                    plan = plan_stage_move(
                        self.cfg,
                        self.model,
                        altaz_tgt=altaz_tgt,
                        altaz_cur=altaz_cur,
                        stage_index=it,
                        stages=stages,
                    )
                except np.linalg.LinAlgError as exc:
                    log_error(None, "GoTo: singular J matrix during solve", exc, throttle_s=5.0, throttle_key="goto_invJ")
                    st.status = "ERR_SINGULAR_MODEL"
                    return st

                st.err_az_arcsec = plan.err_az_arcsec
                st.err_alt_arcsec = plan.err_alt_arcsec

                if plan.done:
                    st.ok = True
                    st.status = "OK"
                    return st

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed before move", exc)

                self._exec_steps(move_steps, Axis.AZ, float(plan.dsteps[0]), delay_us=int(self.cfg.slew_delay_us_az))
                self._exec_steps(move_steps, Axis.ALT, float(plan.dsteps[1]), delay_us=int(self.cfg.slew_delay_us_alt))

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed after move", exc)

                time.sleep(max(0.0, float(self.cfg.settle_s)))

                if self.cfg.solve_near_predicted:
                    altaz_pred = self.model.predict_az_alt_deg()
                    target_for_solver: TargetType = {"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])}
                else:
                    target_for_solver = target

                if use_platesolve_feedback:
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
                        self.model.apply_plate_solve(az_alt_new)
                    else:
                        self.model.last_solve_az_alt_deg = self.model.predict_az_alt_deg()
                        self.model.last_solve_time = time.time()

            st.status = "ERR_MAX_ITERS"
            return st

        finally:
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

        self.model.note_manual_move(axis, direction, steps)

        move_steps(axis, direction, steps, int(delay_us))

    def calibrate_blocking(
        self,
        *,
        get_live_frame: GetFrameFn,
        platesolve_cfg: PlatesolveConfig,
        move_steps: MoveStepsFn,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        n_samples: int = 3,
        max_radius_deg: float = 1.0,
        obstime: Optional[Time] = None,
    ) -> Dict[str, Any]:
        """Refine the model J (including cross-coupling) via randomized dithers."""
        out: Dict[str, Any] = {
            "ok": False,
            "n_samples": 0,
            "J_deg_per_step": None,
            "status": "RUNNING",
        }

        if not self.model.synced:
            out["status"] = "ERR_NOT_SYNCED"
            return out

        calib_platesolve_cfg = replace(
            platesolve_cfg,
            search_radius_deg=1.0,
            gmax=15.0,
            nside=16,
        )

        was_tracking = False
        if tracking_pause is not None:
            try:
                tracking_pause(True)
                was_tracking = True
            except Exception as exc:
                log_error(None, "GoTo: failed to pause tracking (calibration)", exc)

        try:
            if obstime is None:
                obstime = now_time()

            altaz0 = self.model.current_az_alt_deg()
            if altaz0 is None:
                out["status"] = "ERR_NO_CURRENT"
                return out

            if self.model.last_solve_az_alt_deg is None:
                altaz_pred = self.model.predict_az_alt_deg()
                sol0 = self._platesolve_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolve_cfg=calib_platesolve_cfg,
                    radius_deg_seq=(1.0,),
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
                self.model.apply_plate_solve(altaz0)

            max_radius = float(max_radius_deg)
            if max_radius <= 0.0:
                out["status"] = "ERR_BAD_RADIUS"
                return out
            total_samples = int(max(1, n_samples))

            for _ in range(total_samples):
                ang = random.uniform(0.0, 2.0 * math.pi)
                radius = math.sqrt(random.random()) * max_radius

                daz_deg = radius * math.cos(ang)
                dalt_deg = radius * math.sin(ang)

                J = self.model.J_deg_per_step
                try:
                    invJ = np.linalg.inv(J)
                except np.linalg.LinAlgError as exc:
                    log_error(None, "GoTo: singular J matrix during calibration; resetting mechanics", exc, throttle_s=5.0, throttle_key="goto_calib_invJ")
                    self.model.init_from_mechanics()
                    J = self.model.J_deg_per_step
                    invJ = np.linalg.inv(J)

                dsteps = invJ @ np.array([daz_deg, dalt_deg], dtype=np.float64)

                dsteps = np.array([float(int(round(dsteps[0]))), float(int(round(dsteps[1])))], dtype=np.float64)
                if int(dsteps[0]) == 0 and int(dsteps[1]) == 0:
                    continue

                altaz_cur = self.model.current_az_alt_deg()
                if altaz_cur is None:
                    out["status"] = "ERR_NO_CURRENT"
                    return out

                pred_after = altaz_cur.copy()
                pred_after[0] = wrap_deg_360(float(pred_after[0]) + float((J @ dsteps)[0]))
                pred_after[1] = float(pred_after[1]) + float((J @ dsteps)[1])
                if pred_after[1] < float(self.cfg.alt_min_deg) or pred_after[1] > float(self.cfg.alt_max_deg):
                    dsteps[1] *= -1.0
                    pred_after[1] = float(altaz_cur[1]) + float((J @ dsteps)[1])
                    pred_after[1] = clamp(pred_after[1], self.cfg.alt_min_deg, self.cfg.alt_max_deg)

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed before calibration move", exc)

                self._exec_steps(move_steps, Axis.AZ, float(dsteps[0]), delay_us=int(self.cfg.slew_delay_us_az))
                self._exec_steps(move_steps, Axis.ALT, float(dsteps[1]), delay_us=int(self.cfg.slew_delay_us_alt))

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed after calibration move", exc)

                time.sleep(max(0.0, float(self.cfg.settle_s)))

                altaz_pred = self.model.predict_az_alt_deg()
                sol = self._platesolve_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolve_cfg=calib_platesolve_cfg,
                    radius_deg_seq=(1.0,),
                    obstime=now_time(),
                )
                if not bool(getattr(sol, "success", False)):
                    continue

                altaz_new = platesolve_center_to_altaz_deg(
                    float(sol.center_ra_deg),
                    float(sol.center_dec_deg),
                    observer=self.cfg.observer,
                    obstime=now_time(),
                )

                daltaz_meas = np.array(
                    [
                        wrap_deg_180(float(altaz_new[0]) - float(altaz_cur[0])),
                        float(altaz_new[1]) - float(altaz_cur[1]),
                    ],
                    dtype=np.float64,
                )

                dsteps_meas = np.array([float(dsteps[0]), float(dsteps[1])], dtype=np.float64)

                self.model.add_calibration_sample(dsteps_meas, daltaz_meas)
                self.model.apply_plate_solve(altaz_new)

            ok = self.model.fit_J_from_samples(min_samples=3)
            out["ok"] = bool(ok)
            out["n_samples"] = int(len(self.model._calib_steps))
            out["J_deg_per_step"] = self.model.J_deg_per_step.copy().tolist()
            out["status"] = "OK" if ok else "ERR_INSUFFICIENT_SAMPLES"
            return out

        finally:
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


def make_default_goto_controller_for_your_mount() -> GoToController:
    """Factory using the mechanical parameters you provided."""
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
