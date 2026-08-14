from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils import iers

from ap_types import Axis, Frame
from config import AppConfig, SimulationConfig
from gaia_cache import GaiaCacheMissError, gaia_healpix_cone_with_mag, load_gaia_auth
from goto import MountKinematics
from logging_utils import log_error, log_info
from mount_arduino import estimate_firmware_move_duration_s, resolve_common_microsteps
from platesolving import ObserverConfig, expected_field_rotation_deg, parse_target_to_icrs


_IERS_OFFLINE_CONFIGURED = False
_IERS_OFFLINE_LOCK = threading.Lock()
_IERS_PREVIOUS_CONFIG: Optional[Tuple[Any, Any]] = None


def _perf() -> float:
    return time.perf_counter()


def _now_s() -> float:
    return time.time()


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _configure_iers_for_demo(out_log: Any = None) -> None:
    global _IERS_OFFLINE_CONFIGURED, _IERS_PREVIOUS_CONFIG
    with _IERS_OFFLINE_LOCK:
        if _IERS_OFFLINE_CONFIGURED:
            return
        try:
            _IERS_PREVIOUS_CONFIG = (iers.conf.auto_download, iers.conf.auto_max_age)
            iers.conf.auto_download = False
            iers.conf.auto_max_age = None
            _IERS_OFFLINE_CONFIGURED = True
            log_info(out_log, "Simulation: Astropy IERS configured for offline demo mode")
        except Exception as exc:
            log_error(out_log, "Simulation: failed to configure offline IERS mode", exc)


def restore_iers_after_demo(out_log: Any = None) -> None:
    global _IERS_OFFLINE_CONFIGURED, _IERS_PREVIOUS_CONFIG
    with _IERS_OFFLINE_LOCK:
        if not _IERS_OFFLINE_CONFIGURED:
            return
        if _IERS_PREVIOUS_CONFIG is None:
            _IERS_OFFLINE_CONFIGURED = False
            return
        try:
            iers.conf.auto_download = _IERS_PREVIOUS_CONFIG[0]
            iers.conf.auto_max_age = _IERS_PREVIOUS_CONFIG[1]
            _IERS_OFFLINE_CONFIGURED = False
            _IERS_PREVIOUS_CONFIG = None
            log_info(out_log, "Simulation: Astropy IERS settings restored")
        except Exception as exc:
            log_error(out_log, "Simulation: failed to restore Astropy IERS settings", exc)


class SimulationState:
    """Physical truth for demo mode.

    The app's GoTo model remains independent. The simulated mount applies a
    slightly perturbed mechanical matrix, so plate-solving calibration has real
    residuals to measure and fit.
    """

    def __init__(
        self,
        *,
        cfg: SimulationConfig,
        kin: MountKinematics,
        out_log: Any = None,
    ) -> None:
        _configure_iers_for_demo(out_log)
        self.cfg = replace(cfg)
        seed = int(cfg.seed) if cfg.seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._connected_mount = False

        tilt_limit = abs(_finite_float(cfg.random_mount_tilt_deg, 1.0))
        roll_limit = abs(_finite_float(cfg.random_camera_roll_deg, 1.0))
        self.mount_tilt_az_deg = float(self._rng.uniform(-tilt_limit, tilt_limit))
        self.mount_tilt_alt_deg = float(self._rng.uniform(-tilt_limit, tilt_limit))
        self.camera_roll_error_deg = float(self._rng.uniform(-roll_limit, roll_limit))

        # Keep perturbation factors stable across microstep changes.
        self._scale_az = float(1.0 + self._rng.uniform(-0.004, 0.004))
        self._scale_alt = float(1.0 + self._rng.uniform(-0.004, 0.004))
        self._coupling_az_from_alt = float(np.tan(np.deg2rad(self.mount_tilt_az_deg)))
        self._coupling_alt_from_az = float(np.tan(np.deg2rad(self.mount_tilt_alt_deg)))

        self.kin = replace(kin)
        self._true_J_deg_per_step = self._build_true_J()
        self._steps_true = np.zeros(2, dtype=np.float64)

        self._az_deg = (float(cfg.initial_az_deg) + self.mount_tilt_az_deg) % 360.0
        self._alt_deg = float(np.clip(float(cfg.initial_alt_deg) + self.mount_tilt_alt_deg, 1.0, 89.0))

        log_info(
            out_log,
            (
                "Simulation: seeded demo mode "
                f"seed={self.seed} mount_tilt=({self.mount_tilt_az_deg:+.3f},"
                f"{self.mount_tilt_alt_deg:+.3f}) deg "
                f"camera_roll_error={self.camera_roll_error_deg:+.3f} deg"
            ),
        )

    def _build_true_J(self) -> np.ndarray:
        dps_az = float(self.kin.deg_per_step(Axis.AZ))
        dps_alt = float(self.kin.deg_per_step(Axis.ALT))
        return np.array(
            [
                [dps_az * self._scale_az, dps_alt * self._coupling_az_from_alt],
                [dps_az * self._coupling_alt_from_az, dps_alt * self._scale_alt],
            ],
            dtype=np.float64,
        )

    @property
    def true_J_deg_per_step(self) -> np.ndarray:
        with self._lock:
            return self._true_J_deg_per_step.copy()

    def set_connected_mount(self, connected: bool) -> None:
        with self._lock:
            self._connected_mount = bool(connected)

    def is_mount_connected(self) -> bool:
        with self._lock:
            return bool(self._connected_mount)

    def set_microsteps(self, az_div: int, alt_div: int) -> None:
        common_ms = resolve_common_microsteps(int(az_div), int(alt_div), default_ms=64)
        with self._lock:
            self.kin.microsteps_az = int(common_ms)
            self.kin.microsteps_alt = int(common_ms)
            self._true_J_deg_per_step = self._build_true_J()

    def apply_move(self, axis: Axis, direction: int, steps: int) -> Tuple[float, float]:
        steps_abs = abs(int(steps))
        if steps_abs <= 0:
            return self.snapshot_altaz()
        signed = float(steps_abs) * (+1.0 if int(direction) >= 0 else -1.0)
        d_steps = np.zeros(2, dtype=np.float64)
        if axis == Axis.AZ:
            d_steps[0] = signed
        else:
            d_steps[1] = signed

        with self._lock:
            d_altaz = self._true_J_deg_per_step @ d_steps
            self._steps_true += d_steps
            self._az_deg = float((self._az_deg + float(d_altaz[0])) % 360.0)
            self._alt_deg = float(np.clip(self._alt_deg + float(d_altaz[1]), 1.0, 89.0))
            return float(self._az_deg), float(self._alt_deg)

    def snapshot_altaz(self) -> Tuple[float, float]:
        with self._lock:
            return float(self._az_deg), float(self._alt_deg)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "az_deg": float(self._az_deg),
                "alt_deg": float(self._alt_deg),
                "steps_az": float(self._steps_true[0]),
                "steps_alt": float(self._steps_true[1]),
                "mount_tilt_az_deg": float(self.mount_tilt_az_deg),
                "mount_tilt_alt_deg": float(self.mount_tilt_alt_deg),
                "camera_roll_error_deg": float(self.camera_roll_error_deg),
            }

    def center_icrs(self, *, observer: ObserverConfig, obstime: Time) -> SkyCoord:
        az_deg, alt_deg = self.snapshot_altaz()
        true_altaz_observer = replace(observer, refraction_enable=False)
        return parse_target_to_icrs(
            {"az_deg": float(az_deg), "alt_deg": float(alt_deg)},
            observer=true_altaz_observer,
            obstime=obstime,
        ).icrs


class SimulatedMount:
    def __init__(self, state: SimulationState, *, out_log: Any = None) -> None:
        self._state = state
        self._out_log = out_log

    def connect(self, port: str = "DEMO", baud: int = 115200) -> str:
        self._state.set_connected_mount(True)
        return f"OK SIM mount port={port or 'DEMO'} baud={int(baud)}"

    def disconnect(self) -> None:
        try:
            self.stop()
        finally:
            self._state.set_connected_mount(False)

    def is_connected(self) -> bool:
        return self._state.is_mount_connected()

    def stop(self) -> str:
        return "OK SIM STOP"

    def set_microsteps(self, az_div: int, alt_div: int) -> str:
        self._state.set_microsteps(int(az_div), int(alt_div))
        return "OK SIM MS"

    @staticmethod
    def _estimate_move_duration_s(
        steps: int,
        delay_us: int,
        profile: str = "smooth",
    ) -> float:
        return estimate_firmware_move_duration_s(
            int(steps), int(delay_us), profile=profile
        )

    def move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        *,
        profile: str = "smooth",
        blocking: bool = True,
        stop_before_move: bool = True,
    ) -> str:
        if not self.is_connected():
            raise RuntimeError("simulated mount not connected")
        if int(steps) <= 0:
            return "OK SIM MOVE skipped"
        self._state.apply_move(axis=axis, direction=int(direction), steps=int(steps))
        if bool(blocking):
            wait_s = self._estimate_move_duration_s(
                int(steps), int(delay_us), profile=profile
            )
            if wait_s > 0.0:
                time.sleep(min(wait_s + 0.02, 0.25))
        return "OK SIM MOVE"


class SimulatedCameraStream:
    def __init__(
        self,
        *,
        state: SimulationState,
        cfg: AppConfig,
        observer: ObserverConfig,
        out_log: Any = None,
    ) -> None:
        self._state = state
        self._cfg = cfg
        self._observer = observer
        self._out_log = out_log
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._seq = 0
        self._fps_capture = 0.0
        self._dropped = 0
        self._rng = np.random.default_rng(int(state.seed) ^ 0xA57A)
        self._catalog_center: Optional[SkyCoord] = None
        self._catalog_df: Optional[pd.DataFrame] = None
        self._catalog_is_synthetic = False
        self._catalog_source = "none"
        self._last_error: Optional[str] = None
        self._catalog_lock = threading.Lock()

    def start(self) -> None:
        if self._thr is not None:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="SimCameraStream", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        thr = self._thr
        if thr is not None:
            thr.join(timeout=2.0)
            if thr.is_alive():
                log_error(self._out_log, "Simulation camera: thread did not stop within timeout")
        self._thr = None
        with self._lock:
            self._latest = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def stats(self) -> Dict[str, Any]:
        with self._catalog_lock:
            catalog_source = str(self._catalog_source)
            last_error = self._last_error
        return {
            "fps_capture": float(self._fps_capture),
            "dropped": int(self._dropped),
            "seq": int(self._seq),
            "catalog_source": catalog_source,
            "last_error": last_error,
        }

    def invalidate_catalog_cache(self) -> None:
        with self._catalog_lock:
            self._catalog_center = None
            self._catalog_df = None
            self._catalog_is_synthetic = False
            self._catalog_source = "none"
            self._last_error = None

    def _run(self) -> None:
        fps = _finite_float(getattr(self._cfg.simulation, "fps", 8.0), 8.0)
        period_s = 1.0 / max(0.2, fps)
        n = 0
        t_fps = _perf()
        next_t = _perf()

        while not self._stop.is_set():
            t0 = _perf()
            t_wall = _now_s()
            try:
                frame = self._render_frame(t_capture=t0, t_wall=t_wall)
                with self._lock:
                    self._latest = frame
                    self._seq += 1
                n += 1
                now = _perf()
                if (now - t_fps) >= 1.0:
                    self._fps_capture = float(n / max(1e-6, now - t_fps))
                    n = 0
                    t_fps = now
            except Exception as exc:
                log_error(
                    self._out_log,
                    "Simulation camera: frame render failed",
                    exc,
                    throttle_s=2.0,
                    throttle_key="simulation_render_failed",
                )

            next_t += period_s
            sleep_s = max(0.0, next_t - _perf())
            if sleep_s > 0.0:
                self._stop.wait(timeout=sleep_s)
            elif (_perf() - next_t) > 2.0:
                next_t = _perf()

    def _frame_shape(self) -> Tuple[int, int]:
        cam_cfg = self._cfg.camera
        sim_cfg = self._cfg.simulation
        if bool(getattr(cam_cfg, "use_roi", False)):
            w = int(max(32, int(getattr(cam_cfg, "roi_w", sim_cfg.frame_w))))
            h = int(max(32, int(getattr(cam_cfg, "roi_h", sim_cfg.frame_h))))
        else:
            w = int(max(32, int(getattr(sim_cfg, "frame_w", 1280))))
            h = int(max(32, int(getattr(sim_cfg, "frame_h", 720))))
        return w, h

    def _arcsec_per_px(self) -> float:
        ps_cfg = self._cfg.platesolving
        pixel_size_m = _finite_float(getattr(ps_cfg, "pixel_size_m", 2.9e-6), 2.9e-6)
        focal_m = _finite_float(getattr(ps_cfg, "focal_m", 0.9), 0.9)
        if focal_m <= 0.0:
            focal_m = 0.9
        return float(206265.0 * pixel_size_m / focal_m)

    def _catalog_radius_deg(self, w: int, h: int) -> float:
        sim_cfg = self._cfg.simulation
        scale = self._arcsec_per_px()
        fov_radius = float(np.hypot(w, h) * 0.5 * scale / 3600.0)
        return float(max(fov_radius * 1.8, _finite_float(sim_cfg.catalog_radius_deg, 1.2)))

    @staticmethod
    def _empty_catalog() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_id": pd.Series(dtype="int64"),
                "ra": pd.Series(dtype="float64"),
                "dec": pd.Series(dtype="float64"),
                "phot_g_mean_mag": pd.Series(dtype="float64"),
            }
        )

    def _get_catalog(self, center_icrs: SkyCoord, radius_deg: float) -> Tuple[pd.DataFrame, str]:
        sim_cfg = self._cfg.simulation
        margin_deg = abs(_finite_float(getattr(sim_cfg, "catalog_reload_margin_deg", 0.35), 0.35))
        with self._catalog_lock:
            if self._catalog_center is not None and self._catalog_df is not None:
                sep_deg = float(self._catalog_center.separation(center_icrs).deg)
                if sep_deg <= max(0.02, radius_deg * margin_deg):
                    return self._catalog_df.copy(), str(self._catalog_source)

        last_error: Optional[str] = None
        try:
            gaia_cfg = replace(self._cfg.platesolving)
            gaia_cfg.download_missing_tiles = False
            tab = gaia_healpix_cone_with_mag(
                center_icrs=center_icrs,
                radius_deg=float(radius_deg),
                cfg=gaia_cfg,
                auth=load_gaia_auth(),
                verbose=False,
            )
            if isinstance(tab, pd.DataFrame):
                df = tab
            else:
                df = tab.to_pandas()
            want = [c for c in ["source_id", "ra", "dec", "phot_g_mean_mag"] if c in df.columns]
            df = df.loc[:, want].dropna(subset=["ra", "dec"]).reset_index(drop=True)
            source = "gaia"
            if df.empty:
                last_error = "Gaia cache returned 0 stars for the current demo field"
                log_info(
                    self._out_log,
                    f"Simulation camera: Gaia catalog is empty radius={radius_deg:.2f} deg",
                    throttle_s=10.0,
                    throttle_key="simulation_gaia_empty",
                )
            else:
                log_info(
                    self._out_log,
                    f"Simulation camera: Gaia catalog loaded rows={len(df)} radius={radius_deg:.2f} deg",
                    throttle_s=10.0,
                    throttle_key="simulation_gaia_loaded",
                )
        except GaiaCacheMissError as exc:
            missing_count = len(getattr(exc, "missing_paths", []))
            if bool(getattr(sim_cfg, "allow_synthetic_fallback", False)):
                log_info(
                    self._out_log,
                    (
                        "Simulation camera: Gaia cache missing for current field; "
                        "using opt-in synthetic fallback"
                    ),
                    throttle_s=10.0,
                    throttle_key="simulation_gaia_cache_miss_synthetic",
                )
                df = self._synthetic_catalog(center_icrs, radius_deg)
                source = "synthetic"
                last_error = f"Gaia cache missing ({missing_count} tiles); using synthetic fallback"
            else:
                log_info(
                    self._out_log,
                    (
                        "Simulation camera: Gaia cache missing for current field; "
                        "rendering no stars until those Gaia tiles exist"
                    ),
                    throttle_s=10.0,
                    throttle_key="simulation_gaia_cache_miss",
                )
                df = self._empty_catalog()
                source = "gaia_missing"
                last_error = f"Gaia cache missing for current demo field ({missing_count} tiles)"
        except Exception as exc:
            log_error(
                self._out_log,
                "Simulation camera: Gaia catalog load failed",
                exc,
                throttle_s=10.0,
                throttle_key="simulation_gaia_load_failed",
            )
            if bool(getattr(sim_cfg, "allow_synthetic_fallback", False)):
                df = self._synthetic_catalog(center_icrs, radius_deg)
                source = "synthetic"
                last_error = "Gaia catalog load failed; using synthetic fallback"
            else:
                df = self._empty_catalog()
                source = "gaia_error"
                last_error = "Gaia catalog load failed for demo field"

        with self._catalog_lock:
            self._catalog_center = center_icrs
            self._catalog_df = df.copy()
            self._catalog_is_synthetic = source == "synthetic"
            self._catalog_source = str(source)
            self._last_error = last_error
        return df, str(source)

    def _synthetic_catalog(self, center_icrs: SkyCoord, radius_deg: float) -> pd.DataFrame:
        key = int((float(center_icrs.ra.deg) * 1000.0) + (float(center_icrs.dec.deg) * 997.0))
        rng = np.random.default_rng((int(self._state.seed) ^ key) & 0xFFFFFFFF)
        n = int(max(80, min(260, getattr(self._cfg.simulation, "max_render_stars", 240))))

        w, h = self._frame_shape()
        fov_radius_arcsec = float(np.hypot(w, h) * 0.5 * self._arcsec_per_px())
        draw_radius_arcsec = min(float(radius_deg) * 3600.0, max(60.0, fov_radius_arcsec * 1.15))
        # Bias most synthetic stars into the visible FOV so demo mode remains
        # useful even before Gaia tiles have been cached locally.
        r = np.sqrt(rng.uniform(0.0, 1.0, n)) * float(draw_radius_arcsec)
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        lon = r * np.cos(phi)
        lat = r * np.sin(phi)
        off_frame = center_icrs.skyoffset_frame()
        coords = SkyCoord(lon=lon * u.arcsec, lat=lat * u.arcsec, frame=off_frame).icrs
        mags = rng.uniform(8.5, 15.0, n)
        return pd.DataFrame(
            {
                "source_id": np.arange(n, dtype=np.int64),
                "ra": coords.ra.deg,
                "dec": coords.dec.deg,
                "phot_g_mean_mag": mags,
            }
        )

    def _render_frame(self, *, t_capture: float, t_wall: float) -> Frame:
        w, h = self._frame_shape()
        obstime = Time(float(t_wall), format="unix", scale="utc")
        center_icrs = self._state.center_icrs(observer=self._observer, obstime=obstime)
        radius_deg = self._catalog_radius_deg(w, h)
        catalog, catalog_source_raw = self._get_catalog(center_icrs, radius_deg)
        if isinstance(catalog_source_raw, bool):
            catalog_source = "synthetic" if catalog_source_raw else "gaia"
        else:
            catalog_source = str(catalog_source_raw or "unknown")

        sim_cfg = self._cfg.simulation
        background = _finite_float(getattr(sim_cfg, "background_adu", 700.0), 700.0)
        noise = max(0.0, _finite_float(getattr(sim_cfg, "noise_adu", 18.0), 18.0))
        img = self._rng.normal(background, noise, size=(h, w)).astype(np.float32)

        stars_drawn = self._draw_catalog_stars(img, catalog, center_icrs, obstime)
        np.clip(img, 0.0, 65535.0, out=img)

        snap = self._state.snapshot()
        meta: Dict[str, Any] = {
            "seq": int(self._seq),
            "simulated": True,
            "t_wall": float(t_wall),
            "t_capture_mono": float(t_capture),
            "t_mono": float(t_capture),
            "exp_ms": float(getattr(self._cfg.camera, "exp_ms", 0.0)),
            "gain": int(getattr(self._cfg.camera, "gain", 0)),
            "binning": int(getattr(self._cfg.camera, "binning", 1)),
            "width": int(w),
            "height": int(h),
            "center_ra_deg": float(center_icrs.ra.deg),
            "center_dec_deg": float(center_icrs.dec.deg),
            "sim_az_deg": float(snap["az_deg"]),
            "sim_alt_deg": float(snap["alt_deg"]),
            "sim_mount_tilt_az_deg": float(snap["mount_tilt_az_deg"]),
            "sim_mount_tilt_alt_deg": float(snap["mount_tilt_alt_deg"]),
            "sim_camera_roll_error_deg": float(snap["camera_roll_error_deg"]),
            "sim_catalog_rows": int(len(catalog)),
            "sim_stars_drawn": int(stars_drawn),
            "sim_catalog_source": catalog_source,
        }
        return Frame(raw=img.astype(np.uint16), t_capture=float(t_capture), meta=meta)

    def _draw_catalog_stars(
        self,
        img: np.ndarray,
        catalog: pd.DataFrame,
        center_icrs: SkyCoord,
        obstime: Time,
    ) -> int:
        if catalog.empty:
            return 0

        h, w = img.shape[:2]
        coords = SkyCoord(
            ra=np.asarray(catalog["ra"], dtype=np.float64) * u.deg,
            dec=np.asarray(catalog["dec"], dtype=np.float64) * u.deg,
            frame="icrs",
        )
        off = coords.transform_to(center_icrs.skyoffset_frame())
        q_arcsec = np.column_stack(
            [
                np.asarray(off.lon.to_value(u.arcsec), dtype=np.float64),
                np.asarray(off.lat.to_value(u.arcsec), dtype=np.float64),
            ]
        )

        theta = expected_field_rotation_deg(
            float(center_icrs.ra.deg),
            float(center_icrs.dec.deg),
            observer=self._observer,
            obstime=obstime,
            roll_offset_deg=float(getattr(self._cfg.camera, "roll_deg", 0.0))
            + float(self._state.camera_roll_error_deg),
            az_step_deg=float(getattr(self._cfg.platesolving, "rotation_prior_az_step_deg", 0.05)),
        )
        if theta is None or not np.isfinite(float(theta)):
            theta = 0.0
        R = _rotation_matrix(float(theta))
        scale = self._arcsec_per_px()
        center_px = np.array([w * 0.5, h * 0.5], dtype=np.float64)
        px = (q_arcsec / float(scale)) @ R + center_px

        margin = 6.0
        keep = (
            (px[:, 0] >= -margin)
            & (px[:, 0] < (w + margin))
            & (px[:, 1] >= -margin)
            & (px[:, 1] < (h + margin))
        )
        if not np.any(keep):
            return 0

        px = px[keep]
        if "phot_g_mean_mag" in catalog.columns:
            mags = np.asarray(catalog.loc[keep, "phot_g_mean_mag"], dtype=np.float64)
        else:
            mags = np.full(px.shape[0], 13.0, dtype=np.float64)
        finite = np.isfinite(mags)
        mags[~finite] = 15.0

        order = np.argsort(mags)
        max_stars = int(max(1, getattr(self._cfg.simulation, "max_render_stars", 240)))
        order = order[:max_stars]
        px = px[order]
        mags = mags[order]

        base_sigma = max(0.45, _finite_float(getattr(self._cfg.simulation, "star_sigma_px", 1.25), 1.25))
        base_flux = _finite_float(getattr(self._cfg.simulation, "star_flux_adu", 18000.0), 18000.0)
        exp_scale = max(0.05, _finite_float(getattr(self._cfg.camera, "exp_ms", 100.0), 100.0) / 100.0)
        gain_scale = max(0.2, _finite_float(getattr(self._cfg.camera, "gain", 360.0), 360.0) / 360.0)

        drawn = 0
        for (x, y), mag in zip(px, mags):
            mag_f = float(np.clip(float(mag), 3.0, 20.0))
            mag_scale = float(10.0 ** (-0.4 * (mag_f - 12.0)))
            bright = float(np.clip((12.5 - mag_f) / 5.0, 0.0, 1.0))
            faint = float(np.clip((mag_f - 13.0) / 4.0, 0.0, 1.0))

            # Real PSF size is mostly optical, but bright stars look wider in
            # raw frames because their wings and saturation stay above noise.
            core_sigma = float(np.clip(base_sigma * (0.82 + 0.62 * bright - 0.14 * faint), 0.42, 3.2))
            halo_sigma = float(np.clip(core_sigma * (2.2 + 2.4 * bright), core_sigma + 0.4, 14.0))
            halo_frac = float(0.018 + 0.11 * bright)
            radius = int(max(2, min(26, np.ceil(3.3 * halo_sigma if bright > 0.05 else 3.2 * core_sigma))))

            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if xi < -radius or yi < -radius or xi >= w + radius or yi >= h + radius:
                continue

            amp = float(base_flux * exp_scale * np.sqrt(gain_scale) * mag_scale)
            amp = float(np.clip(amp, 90.0, 62000.0))

            x0 = max(0, xi - radius)
            x1 = min(w, xi + radius + 1)
            y0 = max(0, yi - radius)
            y1 = min(h, yi + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r2 = (xx - float(x)) ** 2 + (yy - float(y)) ** 2
            patch = amp * np.exp(-0.5 * (r2 / (core_sigma * core_sigma)))
            if bright > 0.02:
                halo_amp = float(np.clip(amp * halo_frac, 8.0, 8500.0))
                patch += halo_amp * np.exp(-0.5 * (r2 / (halo_sigma * halo_sigma)))
            if amp > 52000.0:
                spike_amp = float((amp - 52000.0) * (0.018 + 0.035 * bright))
                if spike_amp > 0.0:
                    spike_sigma = max(0.65, core_sigma * 0.55)
                    spike_len = max(3.0, halo_sigma * 1.6)
                    dx = xx - float(x)
                    dy = yy - float(y)
                    patch += spike_amp * np.exp(-0.5 * (dy * dy) / (spike_sigma * spike_sigma)) * np.exp(
                        -np.abs(dx) / spike_len
                    )
                    patch += spike_amp * np.exp(-0.5 * (dx * dx) / (spike_sigma * spike_sigma)) * np.exp(
                        -np.abs(dy) / spike_len
                    )
            img[y0:y1, x0:x1] += patch.astype(np.float32)
            drawn += 1

        return int(drawn)
