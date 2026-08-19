# platesolving.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import cv2

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time

from sklearn.neighbors import KDTree
from itertools import combinations, permutations

from logging_utils import log_error, log_info
from ap_types import PlatesolvingStatus
from protocols import StatePublisherProtocol
from workers import BaseWorker
from imaging import ensure_raw16_bayer
from goto_diagnostics import DiagnosticSession
from sep_utils import estimate_shift_from_objects, sep_detect_from_raw16
from config import PlatesolvingConfig, SepConfig

# IMPORTANT: all Gaia/cache/auth logic must live in gaia_cache.py
import gaia_cache as gc

load_gaia_auth = gc.load_gaia_auth
save_gaia_auth = gc.save_gaia_auth


# ============================================================
# Public API surface
# ============================================================

__all__ = [
    "ObserverConfig",
    "OverlayItem",
    "GuideStar",
    "PlatesolvingResult",
    "TargetParseError",
    "expected_field_rotation_deg",
    "TemporalDetections",
    "detect_persistent_sep_objects",
    "project_catalog_to_pixels",
    "verify_plate_from_prior",
    "platesolving_solutions_consistent",
    "solve_plate",
    "PlatesolvingWorker",
    "pixel_to_radec",
    "load_gaia_auth",
    "save_gaia_auth",
]


ProgressCB = Callable[[str, Dict[str, Any]], None]


# ============================================================
# Errors
# ============================================================

class PlatesolvingError(RuntimeError):
    pass


class TargetParseError(PlatesolvingError):
    pass


# ============================================================
# Config / Data types
# ============================================================

@dataclass(frozen=True)
class ObserverConfig:
    """
    Default observer: Estación Central, Santiago, Chile (approximate).
    Used for AltAz -> ICRS conversion.
    """
    lat_deg: float = -33.4569
    lon_deg: float = -70.6990
    height_m: float = 520.0
    refraction_enable: bool = True
    refraction_P_hPa: float = 1013.25
    refraction_T_C: float = 15.0

    def location(self) -> EarthLocation:
        return EarthLocation(
            lat=self.lat_deg * u.deg,
            lon=self.lon_deg * u.deg,
            height=self.height_m * u.m,
        )


def _R_true_to_app_deg(h_true_deg: float, *, P_hPa: float, T_C: float) -> float:
    """
    Refraction R(h_true) in degrees, where: h_app = h_true + R(h_true).
    Bennett-style approximation; avoid near/below horizon for pointing model.
    """
    if h_true_deg <= -1.0:
        return 0.0

    x = math.radians(h_true_deg + 10.3 / (h_true_deg + 5.11))
    R_arcmin = 1.02 / math.tan(x)

    T_K = T_C + 273.15
    R_arcmin *= (P_hPa / 1010.0) * (283.0 / T_K)
    return R_arcmin / 60.0


def _unrefract_app_to_true(
    h_app_deg: float,
    *,
    P_hPa: float = 1013.25,
    T_C: float = 15.0,
    iters: int = 8,
) -> float:
    """Solve h_app = h_true + R(h_true) for h_true."""
    h_app = float(np.clip(float(h_app_deg), -90.0, 90.0))
    # In our Bennett branch model, refraction is 0 for h_true <= -1 deg.
    # Keep sub-horizon apparent altitudes invertible in that regime.
    if h_app <= -1.0:
        return float(h_app)

    h_hi = 89.999999
    h = min(h_hi, max(-1.0, h_app))
    for _ in range(int(iters)):
        R = _R_true_to_app_deg(h, P_hPa=P_hPa, T_C=T_C)
        f = (h + R) - h_app

        eps = 1e-3
        R2 = _R_true_to_app_deg(h + eps, P_hPa=P_hPa, T_C=T_C)
        df = 1.0 + (R2 - R) / eps
        if abs(df) < 1e-12:
            break

        step = f / df
        h -= step
        if h > h_hi:
            h = h_hi
        if h < -1.0:
            h = -1.0
        if abs(step) < 1e-7:
            break
    return float(h)


@dataclass(frozen=True)
class OverlayItem:
    x: float
    y: float
    kind: str                  # "det", "match", "guide"
    label: Optional[str] = None


@dataclass(frozen=True)
class TemporalDetections:
    """Sources confirmed over a drift-compensated RAW16 frame window."""

    reference_frame: np.ndarray
    xy: np.ndarray
    flux: np.ndarray
    hits: np.ndarray
    frame_count: int
    required_hits: int
    drift_xy: Tuple[Tuple[float, float], ...]
    drift_failures: int = 0


@dataclass(frozen=True)
class GuideStar:
    name: str
    ra_deg: float
    dec_deg: float
    gmag: float
    x: float
    y: float


@dataclass(frozen=True)
class PlatesolvingResult:
    success: bool
    status: str
    theta_deg: float
    dx_px: float
    dy_px: float
    response: float

    # similarity / plate model (close to your notebook logic)
    # scale (arcsec/px), rotation matrix (2x2), translation (2,)
    scale_arcsec_per_px: float
    R_2x2: Tuple[Tuple[float, float], Tuple[float, float]]
    t_arcsec: Tuple[float, float]  # translation in TAN arcsec space

    n_inliers: int
    rms_arcsec: float
    rms_px: float

    center_ra_deg: float
    center_dec_deg: float

    overlay: List[OverlayItem]
    guides: List[GuideStar]
    metrics: Dict[str, float]
    obstime_unix: float = float("nan")


# ============================================================
# Target parsing (ICRS/J2000, AltAz, name via gaia_cache/simbad if you keep it there)
# ============================================================

TargetType = Union[
    SkyCoord,
    Tuple[float, float],
    Tuple[str, str],
    str,
    Dict[str, Any],
]


def parse_target_to_icrs(
    target: TargetType,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time],
) -> SkyCoord:
    if obstime is None:
        obstime = Time.now()

    if isinstance(target, SkyCoord):
        return target.icrs

    if isinstance(target, dict):
        # ICRS degrees
        if ("ra_deg" in target and "dec_deg" in target) or ("ra" in target and "dec" in target):
            ra = float(target.get("ra_deg", target.get("ra")))
            dec = float(target.get("dec_deg", target.get("dec")))
            return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        # AltAz degrees
        if "alt_deg" in target and "az_deg" in target:
            alt = float(target["alt_deg"])
            az = float(target["az_deg"])
            t = obstime
            if target.get("obstime"):
                t = Time(str(target["obstime"]))
            loc = observer.location()
            if bool(getattr(observer, "refraction_enable", False)):
                alt = _unrefract_app_to_true(
                    alt,
                    P_hPa=float(getattr(observer, "refraction_P_hPa", 1013.25)),
                    T_C=float(getattr(observer, "refraction_T_C", 15.0)),
                )
            altaz = AltAz(alt=alt * u.deg, az=az * u.deg, obstime=t, location=loc)
            return SkyCoord(altaz).icrs

        raise TargetParseError(f"Unrecognized dict target keys: {list(target.keys())}")

    if isinstance(target, (tuple, list)) and len(target) == 2:
        a, b = target[0], target[1]
        return SkyCoord(ra=float(a) * u.deg, dec=float(b) * u.deg, frame="icrs")

    if isinstance(target, str):
        s = target.strip()
        if not s:
            raise TargetParseError("Empty target string.")

        # If it contains letters, treat as name. Delegate to gaia_cache resolver.
        if any(ch.isalpha() for ch in s):
            return gc.resolve_name_to_icrs(s).icrs

        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            ra_s, dec_s = parts[0], parts[1]
            if ":" in ra_s:
                return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
            return SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")

        raise TargetParseError(f"Could not parse target string: {s}")

    raise TargetParseError(f"Unsupported target type: {type(target).__name__}")


_ICRS_FRAME = ICRS()


def _ensure_icrs(coord: SkyCoord, *, label: str) -> SkyCoord:
    if not coord.frame.is_equivalent_frame(_ICRS_FRAME):
        log_info(None, f"Platesolving: normalizing {label} from frame {coord.frame} to ICRS")
    return coord.icrs


def _icrs_to_altaz_app_deg(
    ra_deg: float,
    dec_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> Tuple[float, float]:
    if obstime is None:
        obstime = Time.now()
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    altaz = c.transform_to(AltAz(obstime=obstime, location=observer.location()))
    az = float(altaz.az.deg) % 360.0
    alt_true = float(altaz.alt.deg)
    alt = alt_true
    if bool(getattr(observer, "refraction_enable", False)):
        alt = alt_true + _R_true_to_app_deg(
            alt_true,
            P_hPa=float(getattr(observer, "refraction_P_hPa", 1013.25)),
            T_C=float(getattr(observer, "refraction_T_C", 15.0)),
        )
    return float(az), float(alt)


def _wrap_deg_180(angle_deg: float) -> float:
    return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)


def _angle_distance_deg(a_deg: float, b_deg: float) -> float:
    return float(abs(_wrap_deg_180(float(a_deg) - float(b_deg))))


def expected_field_rotation_deg(
    ra_deg: float,
    dec_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
    roll_offset_deg: float = 0.0,
    az_step_deg: float = 0.05,
) -> Optional[float]:
    """
    Estimate expected image theta (deg) for a given ICRS center and observer.

    theta convention matches solve_plate:
      theta = atan2(R[1,0], R[0,0]) where R maps image xy -> tangent (east, north).

    We compute local +Az direction in the tangent plane and subtract roll_offset_deg,
    where roll_offset_deg is the orientation of +Az in image coordinates.
    """
    if obstime is None:
        obstime = Time.now()

    center_icrs = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    az_deg, alt_deg = _icrs_to_altaz_app_deg(
        float(ra_deg),
        float(dec_deg),
        observer=observer,
        obstime=obstime,
    )

    step = abs(float(az_step_deg))
    if not np.isfinite(step) or step < 1e-5:
        step = 0.05

    try:
        az_next = (float(az_deg) + step) % 360.0
        next_icrs = parse_target_to_icrs(
            {"az_deg": float(az_next), "alt_deg": float(alt_deg)},
            observer=observer,
            obstime=obstime,
        ).icrs
        off_frame = center_icrs.skyoffset_frame()
        next_off = next_icrs.transform_to(off_frame)
        du = float(next_off.lon.to_value(u.arcsec))
        dv = float(next_off.lat.to_value(u.arcsec))
    except (RuntimeError, ValueError, TypeError):
        return None

    if (not np.isfinite(du)) or (not np.isfinite(dv)):
        return None
    if math.hypot(du, dv) < 1e-6:
        return None

    az_axis_theta = float(np.degrees(np.arctan2(dv, du)))
    theta_expected = _wrap_deg_180(az_axis_theta - float(roll_offset_deg))
    return float(theta_expected)


# ============================================================
# Image: SEP detection (closer to your notebook logic)
# ============================================================

def stretch01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def detect_sep_objects(
    raw16: np.ndarray,
    *,
    sep_bw: int,
    sep_bh: int,
    sep_thresh_sigma: float,
    sep_minarea: int,
    max_sources: int,
    progress_cb: Optional[ProgressCB],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      disp      : stretch image [0..1]
      obj_xy    : (N,2) xy in pixels sorted by flux desc (top max_sources)
      obj_flux  : (N,) flux sorted desc
    """
    if progress_cb:
        progress_cb("detect:start", {})

    raw = ensure_raw16_bayer(raw16)
    disp = stretch01(raw)

    img_det, bkg, objects, obj_xy = sep_detect_from_raw16(
        raw,
        sep_bw=sep_bw,
        sep_bh=sep_bh,
        sep_thresh_sigma=sep_thresh_sigma,
        sep_minarea=sep_minarea,
        max_sources=max_sources,
    )

    if objects is None or len(objects) == 0:
        if progress_cb:
            progress_cb("detect:empty", {"thresh": float(sep_thresh_sigma), "globalrms": float(bkg.globalrms)})
        return disp.astype(np.float32), np.zeros((0, 2), np.float64), np.zeros((0,), np.float64)

    flux = objects["flux"].astype(np.float64)

    if progress_cb:
        progress_cb("detect:done", {"n": int(len(obj_xy)), "thresh": float(sep_thresh_sigma), "globalrms": float(bkg.globalrms)})

    return disp.astype(np.float32), obj_xy, flux


def _one_to_one_pixel_matches(
    predicted_xy: np.ndarray,
    observed_xy: np.ndarray,
    *,
    radius_px: float,
) -> List[Tuple[int, int]]:
    """Greedily assign the closest unique observations to predicted tracks."""
    predicted = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    observed = np.asarray(observed_xy, dtype=np.float64).reshape(-1, 2)
    if predicted.shape[0] == 0 or observed.shape[0] == 0:
        return []

    dist2 = np.sum(
        np.square(predicted[:, None, :] - observed[None, :, :]),
        axis=2,
    )
    candidates = np.argwhere(dist2 <= float(radius_px) ** 2)
    if candidates.size == 0:
        return []
    order = np.argsort(dist2[candidates[:, 0], candidates[:, 1]])
    used_tracks: set[int] = set()
    used_objects: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for idx in order:
        track_idx = int(candidates[int(idx), 0])
        object_idx = int(candidates[int(idx), 1])
        if track_idx in used_tracks or object_idx in used_objects:
            continue
        used_tracks.add(track_idx)
        used_objects.add(object_idx)
        matches.append((track_idx, object_idx))
    return matches


def detect_persistent_sep_objects(
    frames: List[np.ndarray] | Tuple[np.ndarray, ...],
    *,
    sep_bw: int,
    sep_bh: int,
    sep_thresh_sigma: float,
    sep_minarea: int,
    max_sources: int,
    min_hits: int = 10,
    match_radius_px: float = 4.0,
    max_drift_per_frame_px: float = 32.0,
    min_drift_response: float = 0.05,
    progress_cb: Optional[ProgressCB] = None,
) -> TemporalDetections:
    """Keep only sources seen repeatedly after compensating stellar drift.

    Every frame uses the same native-RAW16 SEP pipeline as ``solve_plate``.
    Tracks are propagated into the newest frame using a robust global
    translation estimated between consecutive detections, then matched
    one-to-one within ``match_radius_px``. Coordinates returned by this
    function therefore belong to the last frame, which is also returned as
    ``reference_frame`` for solving and display.
    """
    raw_frames = [ensure_raw16_bayer(frame).copy() for frame in frames]
    if not raw_frames:
        raise ValueError("at least one temporal detection frame is required")

    required_hits = max(1, int(min_hits))
    detections: List[Tuple[np.ndarray, np.ndarray]] = []
    for frame_idx, raw16 in enumerate(raw_frames, start=1):
        _disp, xy, flux = detect_sep_objects(
            raw16,
            sep_bw=int(sep_bw),
            sep_bh=int(sep_bh),
            sep_thresh_sigma=float(sep_thresh_sigma),
            sep_minarea=int(sep_minarea),
            max_sources=int(max_sources),
            progress_cb=None,
        )
        detections.append(
            (
                np.asarray(xy, dtype=np.float64).reshape(-1, 2),
                np.asarray(flux, dtype=np.float64).reshape(-1),
            )
        )
        if progress_cb:
            progress_cb(
                "detect:temporal:frame",
                {
                    "frame": int(frame_idx),
                    "frames": int(len(raw_frames)),
                    "n": int(xy.shape[0]),
                },
            )

    first_xy, first_flux = detections[0]
    tracks: List[Dict[str, Any]] = [
        {
            "xy": np.asarray(point, dtype=np.float64).copy(),
            "hits": 1,
            "flux": [float(first_flux[idx])],
        }
        for idx, point in enumerate(first_xy)
    ]
    previous_xy = first_xy
    drift_xy: List[Tuple[float, float]] = [(0.0, 0.0)]
    drift_failures = 0
    cumulative_x = 0.0
    cumulative_y = 0.0

    for frame_idx in range(1, len(detections)):
        current_xy, current_flux = detections[frame_idx]
        dx, dy, response, _matches = estimate_shift_from_objects(
            previous_xy,
            current_xy,
            max_shift_px=max(0.5, float(max_drift_per_frame_px)),
        )
        if (
            not np.isfinite(dx)
            or not np.isfinite(dy)
            or float(response) < max(0.0, float(min_drift_response))
        ):
            dx = 0.0
            dy = 0.0
            drift_failures += 1
        cumulative_x -= float(dx)
        cumulative_y -= float(dy)
        drift_xy.append((float(cumulative_x), float(cumulative_y)))

        # estimate_shift_from_objects returns the shift applied to current
        # points to align them to previous points. Inverting it propagates a
        # previous track into current-frame coordinates.
        predicted = np.asarray(
            [track["xy"] for track in tracks],
            dtype=np.float64,
        ).reshape(-1, 2)
        if predicted.size:
            predicted[:, 0] -= float(dx)
            predicted[:, 1] -= float(dy)
        frame_matches = _one_to_one_pixel_matches(
            predicted,
            current_xy,
            radius_px=max(0.5, float(match_radius_px)),
        )
        matched_objects: set[int] = set()
        matched_tracks: set[int] = set()
        for track_idx, object_idx in frame_matches:
            tracks[track_idx]["xy"] = current_xy[object_idx].copy()
            tracks[track_idx]["hits"] = int(tracks[track_idx]["hits"]) + 1
            tracks[track_idx]["flux"].append(float(current_flux[object_idx]))
            matched_tracks.add(int(track_idx))
            matched_objects.add(int(object_idx))
        for track_idx, point in enumerate(predicted):
            if track_idx not in matched_tracks:
                tracks[track_idx]["xy"] = np.asarray(point, dtype=np.float64).copy()
        for object_idx, point in enumerate(current_xy):
            if object_idx in matched_objects:
                continue
            tracks.append(
                {
                    "xy": np.asarray(point, dtype=np.float64).copy(),
                    "hits": 1,
                    "flux": [float(current_flux[object_idx])],
                }
            )
        previous_xy = current_xy

    confirmed = [track for track in tracks if int(track["hits"]) >= required_hits]
    reference_frame = raw_frames[-1]

    # A strict hit count is excellent for rejecting hot pixels, but real
    # undersampled stars can split/merge between adjacent SEP detections.  In
    # that case a genuine star may accumulate several shorter tracks.  Build a
    # drift-aligned temporal median, detect on that higher-SNR image, and keep
    # only stack detections supported by a track in multiple native frames.
    # This preserves the temporal rejection while avoiding the four-source
    # collapse seen with the Mars-C data.
    if len(raw_frames) >= 3 and drift_failures < len(raw_frames) - 1:
        cumulative = np.asarray(drift_xy, dtype=np.float64).reshape(-1, 2)
        final_shift = cumulative[-1]
        aligned_frames: List[np.ndarray] = []
        height, width = raw_frames[-1].shape
        for raw16, position in zip(raw_frames, cumulative):
            delta = final_shift - position
            transform = np.array(
                [[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]],
                dtype=np.float32,
            )
            aligned_frames.append(
                cv2.warpAffine(
                    raw16,
                    transform,
                    (int(width), int(height)),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            )
        aligned_median = np.median(np.stack(aligned_frames, axis=0), axis=0)
        reference_frame = np.clip(aligned_median, 0.0, 65535.0).astype(np.uint16)
        _disp, stack_xy, stack_flux = detect_sep_objects(
            reference_frame,
            sep_bw=int(sep_bw),
            sep_bh=int(sep_bh),
            sep_thresh_sigma=float(sep_thresh_sigma),
            sep_minarea=int(sep_minarea),
            max_sources=int(max_sources),
            progress_cb=None,
        )
        support_hits = max(3, int(math.ceil(0.20 * len(raw_frames))))
        supported_tracks = [
            track for track in tracks if int(track["hits"]) >= support_hits
        ]
        if supported_tracks and stack_xy.shape[0] > 0:
            supported_xy = np.asarray(
                [track["xy"] for track in supported_tracks], dtype=np.float64
            ).reshape(-1, 2)
            stack_matches = _one_to_one_pixel_matches(
                stack_xy,
                supported_xy,
                radius_px=max(6.0, float(match_radius_px) * 1.5),
            )
            aligned_confirmed: List[Dict[str, Any]] = []
            for stack_idx, supported_idx in stack_matches:
                source_track = supported_tracks[supported_idx]
                aligned_confirmed.append(
                    {
                        "xy": np.asarray(stack_xy[stack_idx], dtype=np.float64),
                        "hits": int(source_track["hits"]),
                        "flux": [float(stack_flux[stack_idx])],
                    }
                )
            if len(aligned_confirmed) > len(confirmed):
                confirmed = aligned_confirmed
            if progress_cb:
                progress_cb(
                    "detect:temporal:aligned",
                    {
                        "stack_sources": int(stack_xy.shape[0]),
                        "supported_tracks": int(len(supported_tracks)),
                        "confirmed": int(len(aligned_confirmed)),
                        "support_hits": int(support_hits),
                    },
                )
    confirmed.sort(
        key=lambda track: float(np.median(np.asarray(track["flux"], dtype=np.float64))),
        reverse=True,
    )
    if confirmed:
        xy_out = np.stack([np.asarray(track["xy"], dtype=np.float64) for track in confirmed])
        flux_out = np.asarray(
            [float(np.median(np.asarray(track["flux"], dtype=np.float64))) for track in confirmed],
            dtype=np.float64,
        )
        hits_out = np.asarray([int(track["hits"]) for track in confirmed], dtype=np.int32)
    else:
        xy_out = np.zeros((0, 2), dtype=np.float64)
        flux_out = np.zeros((0,), dtype=np.float64)
        hits_out = np.zeros((0,), dtype=np.int32)

    if progress_cb:
        progress_cb(
            "detect:temporal:done",
            {
                "frames": int(len(raw_frames)),
                "required_hits": int(required_hits),
                "confirmed": int(xy_out.shape[0]),
                "drift_failures": int(drift_failures),
            },
        )
    return TemporalDetections(
        reference_frame=reference_frame,
        xy=xy_out,
        flux=flux_out,
        hits=hits_out,
        frame_count=int(len(raw_frames)),
        required_hits=int(required_hits),
        drift_xy=tuple(drift_xy),
        drift_failures=int(drift_failures),
    )


def _detection_overlay(
    xy: np.ndarray,
    *,
    seed_count: int,
    hits: Optional[np.ndarray] = None,
    frame_count: int = 1,
) -> List[OverlayItem]:
    overlay: List[OverlayItem] = []
    positions = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    hit_values = None if hits is None else np.asarray(hits).reshape(-1)
    for idx, (x, y) in enumerate(positions):
        if idx < int(seed_count):
            hit_label = ""
            if hit_values is not None and idx < hit_values.size:
                hit_label = f" {int(hit_values[idx])}/{int(frame_count)}"
            overlay.append(OverlayItem(float(x), float(y), "seed", f"S{idx + 1}{hit_label}"))
        else:
            label = None
            if hit_values is not None and idx < hit_values.size:
                label = f"{int(hit_values[idx])}/{int(frame_count)}"
            overlay.append(OverlayItem(float(x), float(y), "det_persistent", label))
    return overlay


# ============================================================
# Spherical helpers (your notebook logic)
# ============================================================

def unitvec_from_radec(ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float64)


def chord_radius(theta_rad: float) -> float:
    return float(2.0 * np.sin(float(theta_rad) / 2.0))


def annulus_candidates(
    tree: KDTree,
    V: np.ndarray,
    center_idx: int,
    theta_arcsec: float,
    tol_arcsec: float,
) -> np.ndarray:
    theta_min = ((float(theta_arcsec) - float(tol_arcsec)) * u.arcsec).to(u.rad).value
    theta_max = ((float(theta_arcsec) + float(tol_arcsec)) * u.arcsec).to(u.rad).value
    theta_min = max(float(theta_min), 0.0)

    r_max = chord_radius(theta_max)
    r_min = chord_radius(theta_min)

    idxs = tree.query_radius(V[center_idx:center_idx + 1], r=r_max, return_distance=False)[0]
    if idxs.size == 0:
        return idxs

    dots = V[idxs] @ V[center_idx]
    chord2 = 2.0 - 2.0 * dots
    return idxs[chord2 >= (r_min * r_min)]


def sorted_sides_arcsec_from_pixels(xy3: np.ndarray, arcsec_per_pixel: float) -> np.ndarray:
    (x1, y1), (x2, y2), (x3, y3) = xy3
    d12 = np.hypot(x2 - x1, y2 - y1) * float(arcsec_per_pixel)
    d23 = np.hypot(x3 - x2, y3 - y2) * float(arcsec_per_pixel)
    d31 = np.hypot(x1 - x3, y1 - y3) * float(arcsec_per_pixel)
    return np.sort(np.array([d12, d23, d31], dtype=np.float64))


def sorted_sides_arcsec_from_coords(coords: SkyCoord, i: int, j: int, k: int) -> np.ndarray:
    s1 = coords[i].separation(coords[j]).to_value(u.arcsec)
    s2 = coords[j].separation(coords[k]).to_value(u.arcsec)
    s3 = coords[k].separation(coords[i]).to_value(u.arcsec)
    return np.sort(np.array([s1, s2, s3], dtype=np.float64))


def triplet_score(img_sides: np.ndarray, cat_sides: np.ndarray, sigma_arcsec: float) -> Tuple[float, float]:
    errs = cat_sides - img_sides
    score = float(np.sum((errs / float(sigma_arcsec)) ** 2))
    err_max = float(np.max(np.abs(errs)))
    return score, err_max


def sph_centroid(skycoords: SkyCoord) -> SkyCoord:
    ra = skycoords.ra.to_value(u.rad)
    dec = skycoords.dec.to_value(u.rad)
    V = unitvec_from_radec(ra, dec)
    m = V.mean(axis=0)
    n = float(np.linalg.norm(m))
    if n <= 0:
        # fallback to first coord
        return SkyCoord(ra=skycoords[0].ra, dec=skycoords[0].dec, frame="icrs")
    m /= n
    dec0 = float(np.arcsin(m[2]))
    ra0 = float(np.arctan2(m[1], m[0]) % (2 * np.pi))
    return SkyCoord(ra=ra0 * u.rad, dec=dec0 * u.rad, frame="icrs")


def procrustes_similarity(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    2D similarity: Q ≈ s*(P @ R.T) + t
    P: (N,2) px ; Q: (N,2) arcsec
    Returns (s, R, t, rms) with det(R)=+1 enforced (no reflection)
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    varP = float(np.sum(Pc ** 2))
    s = float(np.sum(S) / (varP + 1e-12))
    t = Q.mean(axis=0) - (s * (P.mean(axis=0) @ R.T))

    Qhat = (s * (P @ R.T)) + t
    rms = float(np.sqrt(np.mean(np.sum((Q - Qhat) ** 2, axis=1))))
    return s, R, t, rms


def best_assignment_similarity(img_pts3: np.ndarray, cat_pts3: np.ndarray) -> Dict[str, Any]:
    best = None
    for perm in permutations(range(3)):
        Q = cat_pts3[list(perm)]
        s, R, t, rms = procrustes_similarity(img_pts3, Q)
        cand = (rms, perm, s, R, t)
        if best is None or cand[0] < best[0]:
            best = cand
    rms, perm, s, R, t = best
    return {"rms": float(rms), "perm": perm, "s": float(s), "R": R, "t": t}


def apply_similarity(Pxy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (float(s) * (Pxy @ R.T)) + t


def inverse_similarity(Qxy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return ((Qxy - t) / float(s)) @ R


def project_catalog_to_pixels(
    coords: SkyCoord,
    *,
    center_icrs: SkyCoord,
    scale_arcsec_per_px: float,
    theta_deg: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project catalog coordinates onto a centered camera frame."""
    scale = float(scale_arcsec_per_px)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_arcsec_per_px must be positive")
    center = center_icrs.icrs
    catalog = coords.icrs
    offset = catalog.transform_to(center.skyoffset_frame())
    q_arcsec = np.column_stack(
        [
            np.asarray(offset.lon.to_value(u.arcsec), dtype=np.float64),
            np.asarray(offset.lat.to_value(u.arcsec), dtype=np.float64),
        ]
    )
    theta = np.deg2rad(float(theta_deg))
    rotation = np.array(
        [
            [float(np.cos(theta)), -float(np.sin(theta))],
            [float(np.sin(theta)), float(np.cos(theta))],
        ],
        dtype=np.float64,
    )
    center_px = np.array(
        [float(image_width) * 0.5, float(image_height) * 0.5],
        dtype=np.float64,
    )
    return (q_arcsec / scale) @ rotation + center_px


def one_to_one_match(pred_xy: np.ndarray, cat_xy: np.ndarray, radius_arcsec: float) -> List[Tuple[int, int, float]]:
    """
    Greedy 1–1 matching by distance (no duplicates).
    pred_xy: (Nd,2) in arcsec
    cat_xy : (Ng,2) in arcsec
    Returns list of (det_idx, cat_idx, dist_arcsec)
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    cat_xy = np.asarray(cat_xy, dtype=np.float64)

    Nd = int(pred_xy.shape[0])
    Ng = int(cat_xy.shape[0])
    if Nd == 0 or Ng == 0:
        return []

    tree = KDTree(cat_xy, leaf_size=40, metric="euclidean")
    ind, dist = tree.query_radius(
        pred_xy,
        r=float(radius_arcsec),
        return_distance=True,
        sort_results=True
    )

    edges: List[Tuple[float, int, int]] = []
    for det_i, (cats, ds) in enumerate(zip(ind, dist)):
        if len(cats) == 0:
            continue
        for c, d in zip(cats, ds):
            edges.append((float(d), int(det_i), int(c)))

    if not edges:
        return []

    edges.sort(key=lambda x: x[0])

    used_det = np.zeros(Nd, dtype=bool)
    used_cat = np.zeros(Ng, dtype=bool)

    matches: List[Tuple[int, int, float]] = []
    for d, det_i, cat_i in edges:
        if used_det[det_i] or used_cat[cat_i]:
            continue
        used_det[det_i] = True
        used_cat[cat_i] = True
        matches.append((det_i, cat_i, float(d)))

    return matches


# ============================================================
# Guides (delegated: optional, depends on gaia_cache providing a name resolver)
# ============================================================

def select_guide_star_indices(df_gaia: pd.DataFrame, n: int) -> List[int]:
    n = int(max(1, n))
    if "phot_g_mean_mag" in df_gaia.columns and len(df_gaia) > 0:
        idx = np.argsort(df_gaia["phot_g_mean_mag"].to_numpy(np.float64))
        return [int(i) for i in idx[: min(n, len(idx))].tolist()]
    return [int(i) for i in range(min(n, len(df_gaia)))]


def build_guides_from_solution(
    df_gaia: pd.DataFrame,
    guide_idx: List[int],
    *,
    center_icrs: SkyCoord,
    s_arcsec_per_px: float,
    R: np.ndarray,
    t_arcsec: np.ndarray,
    cfg: PlatesolvingConfig,
    progress_cb: Optional[ProgressCB],
) -> List[GuideStar]:
    """
    Converts selected Gaia stars to pixel positions using the inverse similarity.
    Naming is delegated to gaia_cache if it provides resolve_coord_name().
    """
    guides: List[GuideStar] = []
    if len(df_gaia) == 0:
        return guides

    coords = SkyCoord(
        ra=np.asarray(df_gaia["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(df_gaia["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )

    # Gaia to TAN arcsec around center
    d_lon, d_lat = center_icrs.spherical_offsets_to(coords)
    cat_all = np.column_stack([d_lon.to_value(u.arcsec), d_lat.to_value(u.arcsec)])

    for i in guide_idx:
        ra = float(df_gaia.at[i, "ra"])
        dec = float(df_gaia.at[i, "dec"])
        gmag = float(df_gaia.at[i, "phot_g_mean_mag"]) if "phot_g_mean_mag" in df_gaia.columns else float("nan")

        # arcsec -> px (in the fitted pixel grid)
        px = inverse_similarity(cat_all[i:i + 1], float(s_arcsec_per_px), R, t_arcsec)[0]
        x_ds, y_ds = float(px[0]), float(px[1])

        # convert to full-res pixels if desired by app; for overlay we keep DS coords
        name = "GAIA"
        if hasattr(gc, "resolve_coord_name"):
            try:
                name = str(gc.resolve_coord_name(SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"), cfg=cfg))
            except (RuntimeError, ValueError, TypeError) as exc:
                log_error(None, "Platesolving: failed to resolve Gaia coord name", exc)
                name = "GAIA"

        guides.append(GuideStar(name=name, ra_deg=ra, dec_deg=dec, gmag=gmag, x=x_ds, y=y_ds))

    return guides


# ============================================================
# Pixel -> RA/Dec helpers (from similarity fit + TAN center)
# ============================================================

def pixel_to_radec(
    x_px: float,
    y_px: float,
    *,
    center_icrs: SkyCoord,
    s_arcsec_per_px: float,
    R: np.ndarray,
    t_arcsec: np.ndarray,
) -> SkyCoord:
    """
    Given a pixel position (in the same pixel coordinate system used to fit),
    return ICRS SkyCoord using the inverse similarity:
      px -> arcsec offsets -> SkyCoord offset frame -> ICRS
    """
    P = np.array([[float(x_px), float(y_px)]], dtype=np.float64)
    # forward similarity: px -> arcsec
    Q_arcsec = apply_similarity(P, float(s_arcsec_per_px), R, t_arcsec)[0]
    u_as, v_as = float(Q_arcsec[0]), float(Q_arcsec[1])

    off_frame = center_icrs.skyoffset_frame()
    c_off = SkyCoord(lon=u_as * u.arcsec, lat=v_as * u.arcsec, frame=off_frame)
    return c_off.icrs


# ============================================================
# Main solver: Triplet candidates + 1–1 inliers (your notebook logic)
# ============================================================

def _gaia_load_df(
    center_icrs: SkyCoord,
    radius_deg: float,
    *,
    cfg: PlatesolvingConfig,
    gaia_auth: Optional[Tuple[str, str]],
    progress_cb: Optional[ProgressCB],
) -> pd.DataFrame:
    """
    Thin adapter: Gaia/cache/auth is owned by gaia_cache.py.
    Expect it to return a DataFrame or an Astropy Table convertible to DF.
    """
    if gaia_auth is None and hasattr(gc, "load_gaia_auth"):
        gaia_auth = gc.load_gaia_auth()

    out = gc.gaia_healpix_cone_with_mag(
        center_icrs=center_icrs,
        radius_deg=float(radius_deg),
        cfg=cfg,
        auth=gaia_auth,
        progress_cb=progress_cb,
    )

    if isinstance(out, pd.DataFrame):
        df = out
    else:
        df = out.to_pandas()

    want = [c for c in ["source_id", "ra", "dec", "phot_g_mean_mag"] if c in df.columns]
    df = df.loc[:, want].dropna(subset=["ra", "dec"]).reset_index(drop=True)
    return df


def _configured_plate_scale_arcsec_per_px(cfg: PlatesolvingConfig) -> float:
    if hasattr(cfg, "pixel_size_m") and hasattr(cfg, "focal_m"):
        return float(206265.0 * float(cfg.pixel_size_m) / float(cfg.focal_m))
    if hasattr(cfg, "pixel_um") and hasattr(cfg, "focal_mm"):
        return float(206265.0 * (float(cfg.pixel_um) * 1e-3) / float(cfg.focal_mm))
    return float(getattr(cfg, "arcsec_per_px", 1.0))


def _result_obstime(result: PlatesolvingResult, fallback: Optional[Time] = None) -> Time:
    ts = float(getattr(result, "obstime_unix", float("nan")))
    if np.isfinite(ts) and ts > 0.0:
        return Time(ts, format="unix", scale="utc")
    return fallback if fallback is not None else Time.now()


def _prior_projection_candidates(
    prior: PlatesolvingResult,
    *,
    observer: ObserverConfig,
    obstime: Time,
) -> List[Tuple[str, SkyCoord, float]]:
    """Predict the next field for both an idle and a tracking Alt-Az mount."""
    prior_center = SkyCoord(
        ra=float(prior.center_ra_deg) * u.deg,
        dec=float(prior.center_dec_deg) * u.deg,
        frame="icrs",
    )
    prior_time = _result_obstime(prior, obstime)
    theta_prior = float(prior.theta_deg)

    # Camera roll is stable in the mount frame.  Recover it from the first
    # solution, then recompute field rotation at the time of the new frame.
    axis_theta_prior = expected_field_rotation_deg(
        float(prior_center.ra.deg),
        float(prior_center.dec.deg),
        observer=observer,
        obstime=prior_time,
        roll_offset_deg=0.0,
    )
    roll_offset = 0.0
    if axis_theta_prior is not None and np.isfinite(float(axis_theta_prior)):
        roll_offset = _wrap_deg_180(float(axis_theta_prior) - theta_prior)

    candidates: List[Tuple[str, SkyCoord, float]] = []
    try:
        az_deg, alt_deg = _icrs_to_altaz_app_deg(
            float(prior_center.ra.deg),
            float(prior_center.dec.deg),
            observer=observer,
            obstime=prior_time,
        )
        idle_center = parse_target_to_icrs(
            {"az_deg": float(az_deg), "alt_deg": float(alt_deg)},
            observer=observer,
            obstime=obstime,
        ).icrs
        idle_theta = expected_field_rotation_deg(
            float(idle_center.ra.deg),
            float(idle_center.dec.deg),
            observer=observer,
            obstime=obstime,
            roll_offset_deg=float(roll_offset),
        )
        candidates.append(
            (
                "idle_altaz",
                idle_center,
                theta_prior if idle_theta is None else float(idle_theta),
            )
        )
    except (RuntimeError, ValueError, TypeError):
        pass

    # Tracking keeps the ICRS center approximately fixed.  Keeping this second
    # hypothesis makes the fast verifier safe for normal, tracked observations
    # without broadening the positional matching tolerance.
    candidates.append(("tracking_icrs", prior_center, theta_prior))
    return candidates


def verify_plate_from_prior(
    frame: np.ndarray,
    *,
    prior: PlatesolvingResult,
    target: Optional[TargetType],
    cfg: PlatesolvingConfig,
    sep_cfg: Optional[SepConfig] = None,
    observer: ObserverConfig = ObserverConfig(),
    obstime: Optional[Time] = None,
    gaia_auth: Optional[Tuple[str, str]] = None,
    progress_cb: Optional[ProgressCB] = None,
    temporal_detections: Optional[TemporalDetections] = None,
) -> PlatesolvingResult:
    """Verify/refine a prior WCS without repeating the triplet search.

    Gaia is projected from the previous solution using the elapsed time.  A
    fresh SEP detection must independently reproduce at least ``min_inliers``
    one-to-one catalog matches; the similarity is then refit from those stars.
    """
    if obstime is None:
        obstime = Time.now()
    raw16 = ensure_raw16_bayer(
        temporal_detections.reference_frame
        if temporal_detections is not None
        else frame
    )
    h, w = raw16.shape[:2]
    sep_cfg = sep_cfg or SepConfig()
    if temporal_detections is None:
        _disp, img_xy, _img_flux = detect_sep_objects(
            raw16,
            sep_bw=int(sep_cfg.bw),
            sep_bh=int(sep_cfg.bh),
            sep_thresh_sigma=float(sep_cfg.thresh_sigma),
            sep_minarea=int(sep_cfg.minarea),
            max_sources=int(getattr(cfg, "max_det", 200)),
            progress_cb=progress_cb,
        )
        temporal_hits = None
        temporal_frame_count = 1
    else:
        img_xy = np.asarray(temporal_detections.xy, dtype=np.float64).copy()
        _img_flux = np.asarray(temporal_detections.flux, dtype=np.float64).copy()
        temporal_hits = np.asarray(temporal_detections.hits, dtype=np.int32).copy()
        temporal_frame_count = int(temporal_detections.frame_count)
    if temporal_detections is None:
        temporal_metrics: Dict[str, float] = {}
    else:
        temporal_metrics = {
            "temporal_frames": float(temporal_detections.frame_count),
            "temporal_required_hits": float(temporal_detections.required_hits),
            "temporal_confirmed": float(img_xy.shape[0]),
            "temporal_drift_failures": float(temporal_detections.drift_failures),
        }
    N_det = int(getattr(cfg, "N_det", getattr(cfg, "match_n_det", 30)))
    if N_det > 0:
        img_xy = img_xy[: min(N_det, img_xy.shape[0])]
        if temporal_hits is not None:
            temporal_hits = temporal_hits[: img_xy.shape[0]]
    overlay = _detection_overlay(
        img_xy,
        seed_count=0,
        hits=temporal_hits,
        frame_count=temporal_frame_count,
    )

    def _failure(status: str, metrics: Optional[Dict[str, float]] = None) -> PlatesolvingResult:
        return replace(
            prior,
            success=False,
            status=str(status),
            response=0.0,
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            overlay=overlay,
            guides=[],
            metrics={**temporal_metrics, **dict(metrics or {})},
            obstime_unix=float(obstime.unix),
        )

    # This used to be hardcoded to max(6, ...), silently ignoring a lower
    # cfg.min_inliers. That defeats the whole point of the multi-frame
    # consensus check (initial_consensus_count) as a substitute safety net in
    # genuinely star-poor fields (heavy light pollution, narrow FoV): with a
    # low per-frame floor here, a real single-star or two-star field could
    # never be confirmed at all, no matter how many frames agreed.
    min_inliers = max(1, int(getattr(cfg, "min_inliers", 6)))
    if img_xy.shape[0] < min_inliers:
        return _failure("FAST_PRIOR_NOT_ENOUGH_DETECTIONS", {"n_det": float(img_xy.shape[0])})

    expected_scale = _configured_plate_scale_arcsec_per_px(cfg)
    prior_scale = float(prior.scale_arcsec_per_px)
    if not np.isfinite(prior_scale) or prior_scale <= 0.0:
        prior_scale = expected_scale
    prior_age_s = abs(float(obstime.unix) - float(_result_obstime(prior, obstime).unix))
    max_prior_age_s = max(0.1, float(getattr(cfg, "fast_prior_max_age_s", 60.0)))
    if not np.isfinite(prior_age_s) or prior_age_s > max_prior_age_s:
        return _failure(
            "FAST_PRIOR_EXPIRED",
            {"fast_prior_age_s": float(prior_age_s), "fast_prior_max_age_s": float(max_prior_age_s)},
        )
    radius_deg = float(getattr(cfg, "search_radius_deg", None) or 1.0)
    candidates = _prior_projection_candidates(prior, observer=observer, obstime=obstime)
    if not candidates:
        return _failure("FAST_PRIOR_NO_PREDICTION")

    # One cone covers both the idle and tracking predictions over the short
    # confirmation interval, so cached Gaia data is loaded only once.
    query_center = candidates[0][1]
    extra_radius = max(
        float(query_center.separation(center).deg) for _, center, _ in candidates
    )
    try:
        gaia_df = _gaia_load_df(
            query_center,
            float(radius_deg + extra_radius),
            cfg=cfg,
            gaia_auth=gaia_auth,
            progress_cb=progress_cb,
        )
    except gc.NeedGaiaAuthError as exc:
        return _failure("NEED_GAIA_AUTH", {"missing_tiles": float(getattr(exc, "missing_tiles", 0))})
    except gc.GaiaCacheMissError as exc:
        return _failure("GAIA_CACHE_MISS", {"missing": float(len(getattr(exc, "missing_paths", [])))})
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        log_error(None, "Platesolving: fast-prior Gaia load failed", exc)
        return _failure("GAIA_LOAD_ERROR", {"err": 1.0})

    if len(gaia_df) < min_inliers:
        return _failure("GAIA_TOO_SMALL", {"gaia_rows": float(len(gaia_df))})
    coords = SkyCoord(
        ra=np.asarray(gaia_df["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(gaia_df["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )

    target_center: Optional[SkyCoord] = None
    if target is not None:
        try:
            target_center = parse_target_to_icrs(
                target,
                observer=observer,
                obstime=obstime,
            ).icrs
        except (TargetParseError, ValueError, TypeError):
            return _failure("TARGET_PARSE_ERROR")

    initial_radius_px = max(1.0, float(getattr(cfg, "fast_prior_match_radius_px", 24.0)))
    final_match_px = float(getattr(cfg, "match_max_px", 3.5))
    if not np.isfinite(final_match_px) or final_match_px <= 0.0:
        final_match_px = 3.5
    final_match_arcsec = max(1e-6, final_match_px * expected_scale)
    scale_tol_frac = max(0.0, float(getattr(cfg, "scale_tol_frac", 0.0)))
    max_rms_px = float(getattr(cfg, "max_rms_px", 0.0))
    max_center_shift_px = max(1.0, float(getattr(cfg, "fast_prior_center_shift_px", 64.0)))
    rotation_tol_deg = max(0.1, float(getattr(cfg, "fast_prior_rotation_tol_deg", 5.0)))

    best: Optional[Dict[str, Any]] = None
    best_attempt: Dict[str, float] = {}
    for mode, predicted_center, predicted_theta in candidates:
        projected_px = project_catalog_to_pixels(
            coords,
            center_icrs=predicted_center,
            scale_arcsec_per_px=prior_scale,
            theta_deg=float(predicted_theta),
            image_width=int(w),
            image_height=int(h),
        )
        margin = initial_radius_px
        keep = (
            (projected_px[:, 0] >= -margin)
            & (projected_px[:, 0] <= float(w) + margin)
            & (projected_px[:, 1] >= -margin)
            & (projected_px[:, 1] <= float(h) + margin)
        )
        cat_idx_map = np.flatnonzero(keep)
        initial = one_to_one_match(
            img_xy,
            projected_px[keep],
            radius_arcsec=initial_radius_px,
        )
        matches: List[Tuple[int, int, float]] = [
            (int(det_i), int(cat_idx_map[cat_i]), float(dist_px * prior_scale))
            for det_i, cat_i, dist_px in initial
        ]
        if len(matches) < min_inliers:
            best_attempt[f"{mode}_initial_inliers"] = float(len(matches))
            continue

        catalog_offsets = coords.transform_to(predicted_center.skyoffset_frame())
        cat_arcsec = np.column_stack(
            [
                catalog_offsets.lon.to_value(u.arcsec),
                catalog_offsets.lat.to_value(u.arcsec),
            ]
        )
        fit: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        for _ in range(2):
            det_idx = np.asarray([m[0] for m in matches], dtype=np.int64)
            cat_idx = np.asarray([m[1] for m in matches], dtype=np.int64)
            if det_idx.size < min_inliers:
                break
            s_fit, R_fit, t_fit, _ = procrustes_similarity(
                img_xy[det_idx],
                cat_arcsec[cat_idx],
            )
            fit = (float(s_fit), np.asarray(R_fit), np.asarray(t_fit))
            pred_arcsec = apply_similarity(img_xy, s_fit, R_fit, t_fit)
            matches = one_to_one_match(
                pred_arcsec,
                cat_arcsec,
                radius_arcsec=final_match_arcsec,
            )
        if fit is None or len(matches) < min_inliers:
            best_attempt[f"{mode}_refined_inliers"] = float(len(matches))
            continue

        det_idx = np.asarray([m[0] for m in matches], dtype=np.int64)
        cat_idx = np.asarray([m[1] for m in matches], dtype=np.int64)
        s_fit, R_fit, t_fit, _ = procrustes_similarity(
            img_xy[det_idx],
            cat_arcsec[cat_idx],
        )
        pred_arcsec = apply_similarity(img_xy, s_fit, R_fit, t_fit)
        matches = one_to_one_match(pred_arcsec, cat_arcsec, radius_arcsec=final_match_arcsec)
        if len(matches) < min_inliers:
            continue
        residuals = np.asarray([m[2] for m in matches], dtype=np.float64)
        rms_arcsec = float(np.sqrt(np.mean(residuals * residuals)))
        rms_px = float(rms_arcsec / max(1e-9, float(s_fit)))
        solution_center = pixel_to_radec(
            float(w) * 0.5,
            float(h) * 0.5,
            center_icrs=predicted_center,
            s_arcsec_per_px=float(s_fit),
            R=R_fit,
            t_arcsec=t_fit,
        ).icrs
        theta_fit = float(np.degrees(np.arctan2(R_fit[1, 0], R_fit[0, 0])))
        scale_err = abs(float(s_fit) - expected_scale) / max(1e-9, expected_scale)
        center_shift_px = float(
            solution_center.separation(predicted_center).to_value(u.arcsec)
            / max(1e-9, float(s_fit))
        )
        rotation_err = _angle_distance_deg(theta_fit, float(predicted_theta))
        target_offset_deg = 0.0
        target_ok = True
        max_target_offset_deg = float("inf")
        if target_center is not None:
            target_offset_deg = float(solution_center.separation(target_center).deg)
            center_factor = float(getattr(cfg, "max_center_offset_factor", 0.0))
            if np.isfinite(center_factor) and center_factor > 0.0:
                max_target_offset_deg = (
                    radius_deg * center_factor
                    + max(0.0, float(getattr(cfg, "max_center_offset_margin_deg", 0.0)))
                )
                target_ok = bool(target_offset_deg <= max_target_offset_deg)
        valid = bool(
            np.isfinite(s_fit)
            and s_fit > 0.0
            and (scale_tol_frac <= 0.0 or scale_err <= scale_tol_frac)
            and (max_rms_px <= 0.0 or rms_px <= max_rms_px)
            and center_shift_px <= max_center_shift_px
            and rotation_err <= rotation_tol_deg
            and target_ok
        )
        best_attempt.update(
            {
                f"{mode}_refined_inliers": float(len(matches)),
                f"{mode}_rms_px": float(rms_px),
                f"{mode}_center_shift_px": float(center_shift_px),
                f"{mode}_rotation_err_deg": float(rotation_err),
            }
        )
        if not valid:
            continue
        candidate = {
            "mode": mode,
            "center_ref": predicted_center,
            "solution_center": solution_center,
            "s": float(s_fit),
            "R": np.asarray(R_fit, dtype=np.float64),
            "t": np.asarray(t_fit, dtype=np.float64),
            "matches": matches,
            "cat_arcsec": cat_arcsec,
            "rms_arcsec": rms_arcsec,
            "rms_px": rms_px,
            "theta": theta_fit,
            "scale_err": scale_err,
            "center_shift_px": center_shift_px,
            "rotation_err": rotation_err,
            "target_offset_deg": target_offset_deg,
            "max_target_offset_deg": max_target_offset_deg,
        }
        if best is None or (len(matches), -rms_px) > (len(best["matches"]), -best["rms_px"]):
            best = candidate

    if best is None:
        metrics = {
            "n_det": float(img_xy.shape[0]),
            "gaia_rows": float(len(gaia_df)),
            "fast_prior": 1.0,
            **best_attempt,
        }
        return _failure("FAST_PRIOR_VALIDATION_FAILED", metrics)

    R = best["R"]
    s = float(best["s"])
    t_arcsec = best["t"]
    solution_center = best["solution_center"]
    cat_px = inverse_similarity(best["cat_arcsec"], s, R, t_arcsec)
    for det_idx, cat_idx, _ in best["matches"]:
        ix, iy = img_xy[int(det_idx)]
        gx, gy = cat_px[int(cat_idx)]
        overlay.append(OverlayItem(float(ix), float(iy), "match", None))
        overlay.append(OverlayItem(float(gx), float(gy), "match", None))

    offset_reference = target_center if target_center is not None else best["center_ref"]
    reference_offset = offset_reference.transform_to(solution_center.skyoffset_frame())
    offset_arcsec = np.array(
        [
            reference_offset.lon.to_value(u.arcsec),
            reference_offset.lat.to_value(u.arcsec),
        ],
        dtype=np.float64,
    )
    offset_px = (offset_arcsec / max(1e-9, s)) @ R
    metrics = {
        "n_det": float(img_xy.shape[0]),
        "gaia_rows": float(len(gaia_df)),
        "n_inliers": float(len(best["matches"])),
        "validation_inliers": float(len(best["matches"])),
        "rms_inliers_arcsec": float(best["rms_arcsec"]),
        "rms_px": float(best["rms_px"]),
        "max_rms_px": float(max_rms_px),
        "scale_arcsec_per_px": float(s),
        "scale_err_frac": float(best["scale_err"]),
        "target_offset_deg": float(best["target_offset_deg"]),
        "max_center_offset_deg": float(best["max_target_offset_deg"]),
        "fast_prior": 1.0,
        "fast_prior_idle": 1.0 if best["mode"] == "idle_altaz" else 0.0,
        "prior_center_shift_px": float(best["center_shift_px"]),
        "prior_rotation_err_deg": float(best["rotation_err"]),
        **temporal_metrics,
    }
    return PlatesolvingResult(
        success=True,
        status="OK_FAST_PRIOR",
        theta_deg=float(best["theta"]),
        dx_px=float(offset_px[0]),
        dy_px=float(offset_px[1]),
        response=float(len(best["matches"])) / max(1.0, float(best["rms_px"])),
        scale_arcsec_per_px=float(s),
        R_2x2=((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1]))),
        t_arcsec=(float(t_arcsec[0]), float(t_arcsec[1])),
        n_inliers=int(len(best["matches"])),
        rms_arcsec=float(best["rms_arcsec"]),
        rms_px=float(best["rms_px"]),
        center_ra_deg=float(solution_center.ra.deg),
        center_dec_deg=float(solution_center.dec.deg),
        overlay=overlay,
        guides=[],
        metrics=metrics,
        obstime_unix=float(obstime.unix),
    )


def platesolving_solutions_consistent(
    reference: PlatesolvingResult,
    candidate: PlatesolvingResult,
    *,
    observer: ObserverConfig,
    pointing_tol_arcsec: float = 30.0,
    scale_tol_frac: float = 0.02,
    roll_tol_deg: float = 3.0,
) -> Dict[str, Any]:
    """Compare solutions in the mount frame, compensating sidereal drift."""
    if not bool(reference.success) or not bool(candidate.success):
        return {"ok": False, "pointing_arcsec": float("inf"), "scale_frac": float("inf"), "roll_deg": float("inf")}

    def _mount_vector(result: PlatesolvingResult) -> Tuple[np.ndarray, float]:
        result_time = _result_obstime(result)
        az_deg, alt_deg = _icrs_to_altaz_app_deg(
            float(result.center_ra_deg),
            float(result.center_dec_deg),
            observer=observer,
            obstime=result_time,
        )
        az = math.radians(az_deg)
        alt = math.radians(alt_deg)
        vector = np.array(
            [math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)],
            dtype=np.float64,
        )
        axis_theta = expected_field_rotation_deg(
            float(result.center_ra_deg),
            float(result.center_dec_deg),
            observer=observer,
            obstime=result_time,
            roll_offset_deg=0.0,
        )
        roll = float("nan") if axis_theta is None else _wrap_deg_180(float(axis_theta) - float(result.theta_deg))
        return vector, roll

    tracking_hypothesis = bool(
        float((getattr(candidate, "metrics", {}) or {}).get("fast_prior", 0.0)) > 0.5
        and float((getattr(candidate, "metrics", {}) or {}).get("fast_prior_idle", 1.0)) < 0.5
    )
    try:
        v0, roll0 = _mount_vector(reference)
        v1, roll1 = _mount_vector(candidate)
        if tracking_hypothesis:
            c0 = SkyCoord(
                ra=float(reference.center_ra_deg) * u.deg,
                dec=float(reference.center_dec_deg) * u.deg,
                frame="icrs",
            )
            c1 = SkyCoord(
                ra=float(candidate.center_ra_deg) * u.deg,
                dec=float(candidate.center_dec_deg) * u.deg,
                frame="icrs",
            )
            pointing_arcsec = float(c0.separation(c1).to_value(u.arcsec))
        else:
            dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
            pointing_arcsec = float(math.degrees(math.acos(dot)) * 3600.0)
    except (RuntimeError, ValueError, TypeError):
        pointing_arcsec = float("inf")
        roll0 = float("nan")
        roll1 = float("nan")
    scale0 = float(reference.scale_arcsec_per_px)
    scale1 = float(candidate.scale_arcsec_per_px)
    scale_frac = abs(scale1 - scale0) / max(1e-9, abs(scale0))
    roll_delta = (
        _angle_distance_deg(roll1, roll0)
        if np.isfinite(roll0) and np.isfinite(roll1)
        else float("inf")
    )
    ok = bool(
        np.isfinite(pointing_arcsec)
        and pointing_arcsec <= float(pointing_tol_arcsec)
        and np.isfinite(scale_frac)
        and scale_frac <= float(scale_tol_frac)
        and np.isfinite(roll_delta)
        and roll_delta <= float(roll_tol_deg)
    )
    return {
        "ok": ok,
        "pointing_arcsec": float(pointing_arcsec),
        "scale_frac": float(scale_frac),
        "roll_deg": float(roll_delta),
        "tracking_hypothesis": bool(tracking_hypothesis),
    }


def solve_plate(
    frame: np.ndarray,
    *,
    target: TargetType,
    cfg: PlatesolvingConfig,
    sep_cfg: Optional[SepConfig] = None,
    observer: ObserverConfig = ObserverConfig(),
    obstime: Optional[Time] = None,
    gaia_auth: Optional[Tuple[str, str]] = None,
    progress_cb: Optional[ProgressCB] = None,
    temporal_detections: Optional[TemporalDetections] = None,
) -> PlatesolvingResult:
    if obstime is None:
        obstime = Time.now()

    # 1) Parse target -> ICRS center
    try:
        center_icrs = parse_target_to_icrs(target, observer=observer, obstime=obstime)
    except (TargetParseError, ValueError, TypeError) as exc:
        log_error(None, "Platesolving: target parse failed", exc)
        return PlatesolvingResult(
            success=False,
            status="TARGET_PARSE_ERROR",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=0.0,
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=0.0,
            center_dec_deg=0.0,
            overlay=[],
            guides=[],
            metrics={},
        )
    center_icrs = _ensure_icrs(center_icrs, label="center")

    rotation_prior_enable = bool(getattr(cfg, "rotation_prior_enable", False))
    rotation_prior_tol_deg = float(getattr(cfg, "rotation_prior_tol_deg", 45.0))
    rotation_prior_roll_offset_deg = float(getattr(cfg, "rotation_prior_roll_offset_deg", 0.0))
    rotation_prior_az_step_deg = float(getattr(cfg, "rotation_prior_az_step_deg", 0.05))
    scale_tol_frac = float(getattr(cfg, "scale_tol_frac", 0.0))

    expected_theta_deg: Optional[float] = None
    if rotation_prior_enable and np.isfinite(rotation_prior_tol_deg) and rotation_prior_tol_deg > 0.0:
        expected_theta_deg = expected_field_rotation_deg(
            float(center_icrs.ra.deg),
            float(center_icrs.dec.deg),
            observer=observer,
            obstime=obstime,
            roll_offset_deg=rotation_prior_roll_offset_deg,
            az_step_deg=rotation_prior_az_step_deg,
        )

    # 2) Prepare frame (RAW16)
    raw16 = ensure_raw16_bayer(
        temporal_detections.reference_frame
        if temporal_detections is not None
        else frame
    )
    h, w = raw16.shape[:2]
    sep_cfg = sep_cfg or SepConfig()

    # 3) Detect stars (SEP)
    if temporal_detections is None:
        disp, img_xy_all, img_flux_all = detect_sep_objects(
            raw16,
            sep_bw=int(sep_cfg.bw),
            sep_bh=int(sep_cfg.bh),
            sep_thresh_sigma=float(sep_cfg.thresh_sigma),
            sep_minarea=int(sep_cfg.minarea),
            max_sources=int(getattr(cfg, "max_det", 200)),
            progress_cb=progress_cb,
        )
        temporal_hits = None
        temporal_frame_count = 1
        temporal_metrics: Dict[str, float] = {}
    else:
        disp = stretch01(raw16)
        img_xy_all = np.asarray(temporal_detections.xy, dtype=np.float64).copy()
        img_flux_all = np.asarray(temporal_detections.flux, dtype=np.float64).copy()
        temporal_hits = np.asarray(temporal_detections.hits, dtype=np.int32).copy()
        temporal_frame_count = int(temporal_detections.frame_count)
        drift = np.asarray(temporal_detections.drift_xy, dtype=np.float64).reshape(-1, 2)
        max_drift = float(np.max(np.linalg.norm(drift, axis=1))) if drift.size else 0.0
        temporal_metrics = {
            "temporal_frames": float(temporal_detections.frame_count),
            "temporal_required_hits": float(temporal_detections.required_hits),
            "temporal_confirmed": float(img_xy_all.shape[0]),
            "temporal_drift_failures": float(temporal_detections.drift_failures),
            "temporal_max_drift_px": float(max_drift),
        }

    # SEP sorts detections by flux. A clipped star touching the sensor edge
    # can acquire a badly biased flux and centroid, become one of the three
    # mandatory seeds, and leave the solver with a single false triplet. Keep
    # those detections for corroboration but stably move interior sources to
    # the front of the seed pool whenever at least three are available.
    seed_pool_count = int(img_xy_all.shape[0])
    seed_edge_excluded = 0
    seed_edge_margin_px = max(
        0.0, float(getattr(cfg, "seed_edge_margin_px", 0.0))
    )
    if seed_edge_margin_px > 0.0 and img_xy_all.shape[0] >= 3:
        interior = (
            (img_xy_all[:, 0] >= seed_edge_margin_px)
            & (img_xy_all[:, 0] <= (float(w - 1) - seed_edge_margin_px))
            & (img_xy_all[:, 1] >= seed_edge_margin_px)
            & (img_xy_all[:, 1] <= (float(h - 1) - seed_edge_margin_px))
        )
        interior_idx = np.flatnonzero(interior)
        if interior_idx.size >= 3:
            edge_idx = np.flatnonzero(~interior)
            order = np.concatenate([interior_idx, edge_idx])
            img_xy_all = img_xy_all[order]
            img_flux_all = img_flux_all[order]
            if temporal_hits is not None:
                temporal_hits = temporal_hits[order]
            seed_pool_count = int(interior_idx.size)
            seed_edge_excluded = int(edge_idx.size)
    temporal_metrics["seed_edge_excluded"] = float(seed_edge_excluded)

    N_det = int(getattr(cfg, "N_det", getattr(cfg, "match_n_det", 30)))
    N_seed = int(getattr(cfg, "N_seed", getattr(cfg, "match_n_seed", 3)))
    n_solver_detections = (
        min(N_det, img_xy_all.shape[0])
        if N_det > 0
        else img_xy_all.shape[0]
    )
    N_seed_eff = min(
        int(max(3, N_seed)),
        int(n_solver_detections),
        int(seed_pool_count),
    )
    overlay = _detection_overlay(
        img_xy_all,
        seed_count=N_seed_eff,
        hits=temporal_hits,
        frame_count=temporal_frame_count,
    )

    if img_xy_all.shape[0] < 3:
        return PlatesolvingResult(
            success=False,
            status=(
                "NOT_ENOUGH_PERSISTENT_DETECTIONS"
                if temporal_detections is not None
                else "NOT_ENOUGH_DETECTIONS"
            ),
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=0.0,
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"n_det": float(img_xy_all.shape[0]), **temporal_metrics},
        )

    # 4) Plate scale (arcsec/px) at full-res
    # pixel_size_m expected in cfg; if not, allow pixel_um + focal_mm fallback
    if hasattr(cfg, "pixel_size_m") and hasattr(cfg, "focal_m"):
        arcsec_per_px = 206265.0 * (float(cfg.pixel_size_m)) / float(cfg.focal_m)
    elif hasattr(cfg, "pixel_um") and hasattr(cfg, "focal_mm"):
        arcsec_per_px = 206265.0 * (float(cfg.pixel_um) * 1e-3) / float(cfg.focal_mm)
    else:
        # last resort: require explicit
        arcsec_per_px = float(getattr(cfg, "arcsec_per_px", 1.0))

    # 5) Gaia radius: prefer cfg.search_radius_deg else estimate from FOV
    def _estimate_radius_deg() -> float:
        diag_px = float(np.hypot(w, h))
        factor = float(getattr(cfg, "search_radius_factor", 1.15))
        radius_as = factor * (diag_px / 2.0) * float(arcsec_per_px)
        return float(max(0.4, radius_as / 3600.0))

    radius_deg = float(getattr(cfg, "search_radius_deg", None) or _estimate_radius_deg())

    # 6) Load Gaia from gaia_cache.py
    try:
        if progress_cb:
            progress_cb("gaia:load:start", {"radius_deg": float(radius_deg), "gmax": float(getattr(cfg, "gmax", 15.0))})

        gaia_df = _gaia_load_df(center_icrs, radius_deg, cfg=cfg, gaia_auth=gaia_auth, progress_cb=progress_cb)

    except gc.NeedGaiaAuthError as e:
        return PlatesolvingResult(
            success=False,
            status="NEED_GAIA_AUTH",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"missing_tiles": float(getattr(e, "missing_tiles", 0))},
        )

    except gc.GaiaCacheMissError as e:
        missing_paths = getattr(e, "missing_paths", [])
        return PlatesolvingResult(
            success=False,
            status="GAIA_CACHE_MISS",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"missing": float(len(missing_paths))},
        )

    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        log_error(None, "Platesolving: Gaia load failed", exc)
        return PlatesolvingResult(
            success=False,
            status="GAIA_LOAD_ERROR",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"err": 1.0},
        )

    if len(gaia_df) < int(max(8, 3 * img_xy_all.shape[0])):
        return PlatesolvingResult(
            success=False,
            status="GAIA_TOO_SMALL",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"gaia_rows": float(len(gaia_df))},
        )

    # 7) Build SkyCoord + 3D KDTree on unit vectors
    coords = SkyCoord(
        ra=np.asarray(gaia_df["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(gaia_df["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    V = unitvec_from_radec(coords.ra.to_value(u.rad), coords.dec.to_value(u.rad))
    tree3 = KDTree(V, leaf_size=40, metric="euclidean")

    # 8) Seeds / validation subsets
    img_xy_all = img_xy_all[: min(N_det, img_xy_all.shape[0])] if N_det > 0 else img_xy_all
    img_flux_all = img_flux_all[: img_xy_all.shape[0]]
    if temporal_hits is not None:
        temporal_hits = temporal_hits[: img_xy_all.shape[0]]

    seed_pool_count = min(int(seed_pool_count), int(img_xy_all.shape[0]))
    N_seed_eff = min(
        int(max(3, N_seed)),
        int(img_xy_all.shape[0]),
        int(seed_pool_count),
    )
    img_xy_seed = img_xy_all[:N_seed_eff]

    if img_xy_seed.shape[0] < 3:
        return PlatesolvingResult(
            success=False,
            status="NOT_ENOUGH_SEEDS",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"n_seed": float(img_xy_seed.shape[0])},
        )

    # 9) Triplets in image seeds
    img_triplets: List[Tuple[int, int, int, np.ndarray]] = []
    for (a, b, c) in combinations(range(img_xy_seed.shape[0]), 3):
        sides = sorted_sides_arcsec_from_pixels(img_xy_seed[[a, b, c]], arcsec_per_pixel=arcsec_per_px)
        img_triplets.append((a, b, c, sides))

    # 10) Candidate generation via annuli on 3D KDTree
    tol_arcsec_pairs = float(getattr(cfg, "tol_arcsec_pairs", getattr(cfg, "triplet_tol_arcsec", 3.0)))
    sigma_arcsec = float(getattr(cfg, "sigma_arcsec", getattr(cfg, "triplet_sigma_arcsec", 0.6)))
    max_trials = int(getattr(cfg, "max_trials", getattr(cfg, "triplet_max_trials", 500)))

    candidates: List[Dict[str, Any]] = []
    if progress_cb:
        progress_cb("platesolving:triplets:start", {"n_triplets": int(len(img_triplets))})

    # Performance safeguard: cap how many Gaia "i" centers we scan per triplet
    # (otherwise O(N^2) can explode at high gmax/radius)
    max_i_scan = int(getattr(cfg, "max_i_scan", 2000))
    i_scan = np.arange(V.shape[0])
    if V.shape[0] > max_i_scan:
        # bias towards bright Gaia if available
        if "phot_g_mean_mag" in gaia_df.columns:
            mags = np.asarray(gaia_df["phot_g_mean_mag"], dtype=np.float64)
            i_scan = np.argsort(mags)[:max_i_scan]
        else:
            i_scan = i_scan[:max_i_scan]

    for triplet_idx, (a, b, c, img_sides) in enumerate(img_triplets):
        if progress_cb and triplet_idx % 8 == 0:
            progress_cb(
                "platesolving:checkpoint",
                {"phase": "triplets", "triplet": int(triplet_idx)},
            )
        d1, d2, d3 = float(img_sides[0]), float(img_sides[1]), float(img_sides[2])

        theta_min = ((d3 - tol_arcsec_pairs) * u.arcsec).to(u.rad).value
        theta_max = ((d3 + tol_arcsec_pairs) * u.arcsec).to(u.rad).value
        theta_min = max(float(theta_min), 0.0)

        r_max = chord_radius(theta_max)
        r_min = chord_radius(theta_min)

        for i_position, i in enumerate(i_scan):
            if progress_cb and i_position % 64 == 0:
                progress_cb(
                    "platesolving:checkpoint",
                    {
                        "phase": "catalog_scan",
                        "triplet": int(triplet_idx),
                        "catalog_index": int(i_position),
                    },
                )
            nbrs = tree3.query_radius(V[i:i + 1], r=r_max, return_distance=False)[0]
            nbrs = nbrs[nbrs > i]  # avoid duplicate pairs
            if nbrs.size == 0:
                continue

            dots = V[nbrs] @ V[i]
            chord2 = 2.0 - 2.0 * dots
            nbrs = nbrs[chord2 >= (r_min * r_min)]
            if nbrs.size == 0:
                continue

            for j in nbrs:
                candA = np.intersect1d(
                    annulus_candidates(tree3, V, int(i), d1, tol_arcsec_pairs),
                    annulus_candidates(tree3, V, int(j), d2, tol_arcsec_pairs),
                    assume_unique=False
                )
                candB = np.intersect1d(
                    annulus_candidates(tree3, V, int(i), d2, tol_arcsec_pairs),
                    annulus_candidates(tree3, V, int(j), d1, tol_arcsec_pairs),
                    assume_unique=False
                )
                ks = np.union1d(candA, candB)
                if ks.size == 0:
                    continue
                ks = ks[(ks != i) & (ks != j)]
                if ks.size == 0:
                    continue

                for k in ks:
                    cat_sides = sorted_sides_arcsec_from_coords(coords, int(i), int(j), int(k))
                    score, err_max = triplet_score(img_sides, cat_sides, sigma_arcsec=sigma_arcsec)
                    if err_max <= tol_arcsec_pairs:
                        candidates.append({
                            "score": float(score),
                            "err_max": float(err_max),
                            "img_triplet": (int(a), int(b), int(c)),
                            "gaia_idx": (int(i), int(j), int(k)),
                            "gaia_source_id": (
                                int(gaia_df["source_id"].iloc[int(i)]) if "source_id" in gaia_df.columns else int(i),
                                int(gaia_df["source_id"].iloc[int(j)]) if "source_id" in gaia_df.columns else int(j),
                                int(gaia_df["source_id"].iloc[int(k)]) if "source_id" in gaia_df.columns else int(k),
                            ),
                        })

    candidates.sort(key=lambda d: d["score"])

    if progress_cb:
        progress_cb("platesolving:triplets:candidates", {"n_candidates": int(len(candidates))})

    if len(candidates) == 0:
        return PlatesolvingResult(
            success=False,
            status="NO_TRIPLET_CANDIDATES",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"n_candidates": 0.0},
        )

    # 11) Validate top candidates with 1–1 inliers
    match_tol_arcsec = float(getattr(cfg, "match_tol_arcsec", 5.0))
    match_max_px = float(getattr(cfg, "match_max_px", 0.0))
    if np.isfinite(match_max_px) and match_max_px > 0.0:
        # ``match_max_px`` used to be exposed in the UI but ignored by the
        # solver. Clamp the angular tolerance with it so the configured
        # full-resolution centroid tolerance actually takes effect.
        match_tol_arcsec = min(match_tol_arcsec, match_max_px * float(arcsec_per_px))
    if not np.isfinite(match_tol_arcsec) or match_tol_arcsec <= 0.0:
        match_tol_arcsec = max(1e-6, 3.5 * float(arcsec_per_px))
    pred_margin_arcsec = float(getattr(cfg, "pred_margin_arcsec", match_tol_arcsec + 20.0))

    def evaluate_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
        a, b, c = cand["img_triplet"]
        i, j, k = cand["gaia_idx"]

        img_tri = img_xy_seed[[a, b, c]]
        tri_coords = coords[[i, j, k]]
        center0 = sph_centroid(tri_coords)

        # Gaia triplet in TAN plane (arcsec)
        d_lon, d_lat = center0.spherical_offsets_to(tri_coords)
        cat_tri = np.column_stack([d_lon.to_value(u.arcsec), d_lat.to_value(u.arcsec)])

        fit = best_assignment_similarity(img_tri, cat_tri)

        # ALL Gaia into TAN plane
        d_lon_all, d_lat_all = center0.spherical_offsets_to(coords)
        cat_all = np.column_stack([d_lon_all.to_value(u.arcsec), d_lat_all.to_value(u.arcsec)])

        # ALL detections -> TAN via similarity
        pred_all = apply_similarity(img_xy_all, fit["s"], fit["R"], fit["t"])

        # crop Gaia around predicted bbox
        xmin, ymin = pred_all.min(axis=0) - pred_margin_arcsec
        xmax, ymax = pred_all.max(axis=0) + pred_margin_arcsec
        keep = (
            (cat_all[:, 0] >= xmin) & (cat_all[:, 0] <= xmax) &
            (cat_all[:, 1] >= ymin) & (cat_all[:, 1] <= ymax)
        )
        cat_f = cat_all[keep]
        cat_idx_map = np.flatnonzero(keep)

        matches_local = one_to_one_match(pred_all, cat_f, radius_arcsec=match_tol_arcsec)
        inliers = [(int(det_i), int(cat_idx_map[cat_i]), float(d)) for (det_i, cat_i, d) in matches_local]

        # Refit with every corroborating match, then rematch. The old path
        # kept the exact transform from only three seed stars, so a chance
        # triplet could remain artificially perfect while the rest of the
        # field was only loosely consistent.
        for _ in range(2):
            if len(inliers) < 3:
                break
            det_idx = np.asarray([m[0] for m in inliers], dtype=np.int64)
            cat_idx = np.asarray([m[1] for m in inliers], dtype=np.int64)
            s_refined, R_refined, t_refined, _ = procrustes_similarity(
                img_xy_all[det_idx],
                cat_all[cat_idx],
            )
            if (
                (not np.isfinite(s_refined))
                or s_refined <= 0.0
                or (not np.all(np.isfinite(R_refined)))
                or (not np.all(np.isfinite(t_refined)))
            ):
                break
            fit = {
                **fit,
                "s": float(s_refined),
                "R": np.asarray(R_refined, dtype=np.float64),
                "t": np.asarray(t_refined, dtype=np.float64),
            }
            pred_all = apply_similarity(img_xy_all, fit["s"], fit["R"], fit["t"])
            inliers = one_to_one_match(pred_all, cat_all, radius_arcsec=match_tol_arcsec)

        num_inliers = int(len(inliers))
        rms_inliers = float(np.sqrt(np.mean([d * d for (_, _, d) in inliers]))) if num_inliers > 0 else float("inf")
        seed_det = {int(a), int(b), int(c)}
        validation_inliers = int(sum(1 for det_i, _, _ in inliers if int(det_i) not in seed_det))
        R_fit = np.asarray(fit["R"], dtype=np.float64)
        theta_fit_deg = float(np.degrees(np.arctan2(R_fit[1, 0], R_fit[0, 0])))

        rotation_err_deg: Optional[float] = None
        rotation_ok = True
        if expected_theta_deg is not None:
            rotation_err_deg = _angle_distance_deg(theta_fit_deg, expected_theta_deg)
            rotation_ok = bool(rotation_err_deg <= float(rotation_prior_tol_deg))

        # Sanity check the fitted scale against the known optics. The
        # triangle-side tolerance used to generate candidates is an absolute
        # arcsec value, so it is comparatively loose for short/tight seed
        # triangles and can admit candidates whose implied plate scale is
        # far from what the instrument actually delivers.
        scale_err_frac: Optional[float] = None
        scale_ok = True
        if scale_tol_frac > 0.0 and np.isfinite(arcsec_per_px) and arcsec_per_px > 0.0:
            s_fit = float(fit["s"])
            scale_err_frac = abs(s_fit - arcsec_per_px) / float(arcsec_per_px)
            scale_ok = bool(np.isfinite(scale_err_frac) and scale_err_frac <= scale_tol_frac)

        return {
            "num_inliers": num_inliers,
            "validation_inliers": validation_inliers,
            "rms_inliers": rms_inliers,
            "fit": fit,
            "center": center0,
            "candidate": cand,
            "inliers": inliers,
            "theta_deg": theta_fit_deg,
            "rotation_err_deg": rotation_err_deg,
            "rotation_ok": rotation_ok,
            "scale_err_frac": scale_err_frac,
            "scale_ok": scale_ok,
        }

    to_eval = candidates[: min(len(candidates), int(max_trials))]

    best = None
    rotation_rejected = 0
    scale_rejected = 0
    best_rotation_err_deg: Optional[float] = None
    best_scale_err_frac: Optional[float] = None
    if progress_cb:
        progress_cb("platesolving:validate:start", {"n_eval": int(len(to_eval))})

    for candidate_idx, cand in enumerate(to_eval):
        if progress_cb and candidate_idx % 16 == 0:
            progress_cb(
                "platesolving:checkpoint",
                {"phase": "validation", "candidate": int(candidate_idx)},
            )
        ev = evaluate_candidate(cand)
        s_err = ev.get("scale_err_frac", None)
        if isinstance(s_err, (float, int)) and np.isfinite(float(s_err)):
            s_err_f = float(s_err)
            if best_scale_err_frac is None or s_err_f < best_scale_err_frac:
                best_scale_err_frac = s_err_f
        if not bool(ev.get("scale_ok", True)):
            scale_rejected += 1
            continue
        if expected_theta_deg is not None:
            err = ev.get("rotation_err_deg", None)
            if isinstance(err, (float, int)) and np.isfinite(float(err)):
                err_f = float(err)
                if best_rotation_err_deg is None or err_f < best_rotation_err_deg:
                    best_rotation_err_deg = err_f
            if not bool(ev.get("rotation_ok", False)):
                rotation_rejected += 1
                continue
        if best is None:
            best = ev
            continue
        cur = (ev["num_inliers"], -ev["rms_inliers"], -ev["candidate"]["score"])
        bst = (best["num_inliers"], -best["rms_inliers"], -best["candidate"]["score"])
        if cur > bst:
            best = ev

    if best is None:
        if scale_tol_frac > 0.0 and scale_rejected > 0:
            metrics = {
                "n_eval": float(len(to_eval)),
                "scale_expected_arcsec_per_px": float(arcsec_per_px),
                "scale_tol_frac": float(scale_tol_frac),
                "scale_rejected": float(scale_rejected),
            }
            if best_scale_err_frac is not None:
                metrics["scale_best_err_frac"] = float(best_scale_err_frac)
            return PlatesolvingResult(
                success=False,
                status="NO_SCALE_MATCH",
                theta_deg=0.0,
                dx_px=0.0,
                dy_px=0.0,
                response=0.0,
                scale_arcsec_per_px=float(arcsec_per_px),
                R_2x2=((1.0, 0.0), (0.0, 1.0)),
                t_arcsec=(0.0, 0.0),
                n_inliers=0,
                rms_arcsec=float("inf"),
                rms_px=float("inf"),
                center_ra_deg=float(center_icrs.ra.deg),
                center_dec_deg=float(center_icrs.dec.deg),
                overlay=overlay,
                guides=[],
                metrics=metrics,
            )
        if expected_theta_deg is not None and rotation_rejected > 0:
            metrics = {
                "n_eval": float(len(to_eval)),
                "rotation_expected_deg": float(expected_theta_deg),
                "rotation_tol_deg": float(rotation_prior_tol_deg),
                "rotation_rejected": float(rotation_rejected),
            }
            if best_rotation_err_deg is not None:
                metrics["rotation_best_err_deg"] = float(best_rotation_err_deg)
            return PlatesolvingResult(
                success=False,
                status="NO_ROTATION_MATCH",
                theta_deg=0.0,
                dx_px=0.0,
                dy_px=0.0,
                response=0.0,
                scale_arcsec_per_px=float(arcsec_per_px),
                R_2x2=((1.0, 0.0), (0.0, 1.0)),
                t_arcsec=(0.0, 0.0),
                n_inliers=0,
                rms_arcsec=float("inf"),
                rms_px=float("inf"),
                center_ra_deg=float(center_icrs.ra.deg),
                center_dec_deg=float(center_icrs.dec.deg),
                overlay=overlay,
                guides=[],
                metrics=metrics,
            )
        return PlatesolvingResult(
            success=False,
            status="VALIDATION_FAILED",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            overlay=overlay,
            guides=[],
            metrics={"n_eval": float(len(to_eval))},
        )

    # 12) Final overlays (inliers + Gaia points in view)
    best_center: SkyCoord = _ensure_icrs(best["center"], label="best_center")
    best_center_icrs = SkyCoord(ra=best_center.icrs.ra, dec=best_center.icrs.dec, frame="icrs")
    center_icrs_ref = SkyCoord(ra=center_icrs.icrs.ra, dec=center_icrs.icrs.dec, frame="icrs")
    best_fit: Dict[str, Any] = best["fit"]
    best_inliers: List[Tuple[int, int, float]] = best["inliers"]

    # Gaia all into best TAN arcsec
    d_lon_all, d_lat_all = best_center_icrs.spherical_offsets_to(coords)
    gaia_xy_arcsec = np.column_stack([d_lon_all.to_value(u.arcsec), d_lat_all.to_value(u.arcsec)])

    s = float(best_fit["s"])
    R = np.asarray(best_fit["R"], dtype=np.float64)
    t_arcsec = np.asarray(best_fit["t"], dtype=np.float64)
    solution_center = pixel_to_radec(
        float(w) * 0.5,
        float(h) * 0.5,
        center_icrs=best_center_icrs,
        s_arcsec_per_px=s,
        R=R,
        t_arcsec=t_arcsec,
    ).icrs
    solution_center_icrs = SkyCoord(
        ra=solution_center.ra,
        dec=solution_center.dec,
        frame="icrs",
    )

    # convert Gaia arcsec -> pixels for overlay
    gaia_xy_px = inverse_similarity(gaia_xy_arcsec, s, R, t_arcsec)

    # mark matches in overlay
    for det_idx, cat_idx, dist in best_inliers:
        ix, iy = img_xy_all[det_idx]
        gx, gy = gaia_xy_px[cat_idx]
        overlay.append(OverlayItem(float(ix), float(iy), "match", None))
        # optionally show Gaia match point too as "match" with label
        overlay.append(OverlayItem(float(gx), float(gy), "match", None))

    # 13) Guides (optional)
    guides: List[GuideStar] = []
    guide_n = int(getattr(cfg, "guide_n", 3))
    if guide_n > 0:
        gi = select_guide_star_indices(gaia_df, guide_n)
        guides = build_guides_from_solution(
            gaia_df,
            gi,
            center_icrs=best_center_icrs,
            s_arcsec_per_px=s,
            R=R,
            t_arcsec=t_arcsec,
            cfg=cfg,
            progress_cb=progress_cb,
        )
        for g in guides:
            overlay.append(OverlayItem(float(g.x), float(g.y), "guide", str(g.name)))

    # 14) Success criterion
    min_inliers = int(getattr(cfg, "min_inliers", 3))
    min_validation_inliers = max(0, int(getattr(cfg, "min_validation_inliers", 0)))
    success_inliers = bool(best["num_inliers"] >= min_inliers)
    success_validation = bool(best.get("validation_inliers", 0) >= min_validation_inliers)

    offset_lon, offset_lat = solution_center_icrs.spherical_offsets_to(center_icrs_ref)
    offset_arcsec = np.array([offset_lon.to_value(u.arcsec), offset_lat.to_value(u.arcsec)], dtype=np.float64)
    offset_px = (offset_arcsec / max(1e-9, float(s))) @ R
    theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    rms_px = float(best["rms_inliers"] / max(1e-9, float(s)))
    max_rms_px = float(getattr(cfg, "max_rms_px", 0.0))
    rms_ok = bool(
        (not np.isfinite(max_rms_px))
        or max_rms_px <= 0.0
        or (np.isfinite(rms_px) and rms_px <= max_rms_px)
    )
    target_offset_deg = float(solution_center_icrs.separation(center_icrs_ref).deg)
    max_center_offset_factor = float(getattr(cfg, "max_center_offset_factor", 0.0))
    max_center_offset_margin_deg = max(
        0.0,
        float(getattr(cfg, "max_center_offset_margin_deg", 0.0)),
    )
    max_center_offset_deg = float("inf")
    if np.isfinite(max_center_offset_factor) and max_center_offset_factor > 0.0:
        max_center_offset_deg = (
            float(radius_deg) * max_center_offset_factor + max_center_offset_margin_deg
        )
    center_ok = bool(
        np.isfinite(target_offset_deg)
        and target_offset_deg <= max_center_offset_deg
    )
    response = float(best["num_inliers"]) / max(1.0, rms_px)
    rotation_err_deg: Optional[float] = None
    rotation_ok = True
    if expected_theta_deg is not None:
        rotation_err_deg = _angle_distance_deg(theta_deg, expected_theta_deg)
        rotation_ok = bool(rotation_err_deg <= float(rotation_prior_tol_deg))
    success = bool(
        success_inliers
        and success_validation
        and rms_ok
        and center_ok
        and rotation_ok
    )
    status = "OK"
    if not success_inliers:
        status = "LOW_INLIERS"
    elif not success_validation:
        status = "LOW_VALIDATION_INLIERS"
    elif not rms_ok:
        status = "HIGH_RMS"
    elif not center_ok:
        status = "CENTER_OUT_OF_RANGE"
    elif not rotation_ok:
        status = "ROTATION_MISMATCH"

    metrics = {
        "n_det": float(img_xy_all.shape[0]),
        "n_seed": float(img_xy_seed.shape[0]),
        "gaia_rows": float(len(gaia_df)),
        "radius_deg": float(radius_deg),
        "arcsec_per_px": float(arcsec_per_px),
        "triplet_score": float(best["candidate"]["score"]),
        "triplet_err_max": float(best["candidate"]["err_max"]),
        "max_trials": float(len(to_eval)),
        "n_inliers": float(best["num_inliers"]),
        "validation_inliers": float(best.get("validation_inliers", 0)),
        "min_validation_inliers": float(min_validation_inliers),
        "rms_inliers_arcsec": float(best["rms_inliers"]),
        "rms_px": float(rms_px),
        "max_rms_px": float(max_rms_px),
        "scale_arcsec_per_px": float(s),
        "match_tol_arcsec": float(match_tol_arcsec),
        "match_max_px": float(match_max_px),
        "target_offset_deg": float(target_offset_deg),
        "max_center_offset_deg": float(max_center_offset_deg),
        **temporal_metrics,
    }
    if expected_theta_deg is not None:
        metrics["rotation_expected_deg"] = float(expected_theta_deg)
        metrics["rotation_tol_deg"] = float(rotation_prior_tol_deg)
        metrics["rotation_rejected"] = float(rotation_rejected)
        if rotation_err_deg is not None:
            metrics["rotation_err_deg"] = float(rotation_err_deg)
        if best_rotation_err_deg is not None:
            metrics["rotation_best_err_deg"] = float(best_rotation_err_deg)

    return PlatesolvingResult(
        success=success,
        status=status,
        theta_deg=theta_deg,
        dx_px=float(offset_px[0]),
        dy_px=float(offset_px[1]),
        response=response,
        scale_arcsec_per_px=float(s),
        R_2x2=((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1]))),
        t_arcsec=(float(t_arcsec[0]), float(t_arcsec[1])),
        n_inliers=int(best["num_inliers"]),
        rms_arcsec=float(best["rms_inliers"]),
        rms_px=rms_px,
        center_ra_deg=float(solution_center_icrs.ra.deg),
        center_dec_deg=float(solution_center_icrs.dec.deg),
        overlay=overlay,
        guides=guides,
        metrics=metrics,
        obstime_unix=float(obstime.unix),
    )


def _render_platesolving_debug_jpeg(
    frame: Optional[np.ndarray],
    overlay: Optional[List[Any]],
) -> Optional[bytes]:
    if frame is None:
        return None
    gray = frame
    if getattr(gray, "ndim", 0) == 3:
        if gray.shape[2] == 1:
            gray = gray[:, :, 0]
        else:
            gray = gray[:, :, :3].astype(np.float32).mean(axis=2)
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        return None

    p1, p99 = np.percentile(gray, [1.0, 99.0])
    if p99 <= p1:
        p1 = float(gray.min()) if gray.size else 0.0
        p99 = float(gray.max()) if gray.size else 1.0
    scale = 255.0 / max(1e-6, float(p99 - p1))
    u8 = np.clip((gray - p1) * scale, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    if overlay:
        h, w = img.shape[:2]
        colors = {
            "det": (255, 0, 0),
            "det_persistent": (255, 255, 0),
            "seed": (255, 0, 255),
            "match": (0, 255, 0),
            "guide": (0, 0, 255),
        }
        for item in overlay:
            x = int(round(float(getattr(item, "x", 0.0))))
            y = int(round(float(getattr(item, "y", 0.0))))
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            kind = str(getattr(item, "kind", "det"))
            color = colors.get(kind, (255, 255, 0))
            radius = 11 if kind == "seed" else 10 if kind == "guide" else 8 if kind == "match" else 7
            thickness = 2 if kind == "seed" else 1
            cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
            label = getattr(item, "label", None)
            if kind in {"guide", "seed"} and label:
                cv2.putText(
                    img,
                    str(label),
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )

    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except cv2.error as exc:
        log_error(None, "Platesolving: failed to encode debug JPEG", exc)
        return None
    if not ok:
        return None
    return bytes(buf.tobytes())


def _build_platesolving_debug_info(result: Any) -> Dict[str, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    info = {
        "status": str(getattr(result, "status", "")),
        "response": float(getattr(result, "response", 0.0)),
        "n_det": metrics.get("n_det"),
        "gaia_rows": metrics.get("gaia_rows"),
        "n_inliers": int(getattr(result, "n_inliers", 0)),
        "rms_px": float(getattr(result, "rms_px", 0.0)),
        "theta_deg": float(getattr(result, "theta_deg", 0.0)),
        "dx_px": float(getattr(result, "dx_px", 0.0)),
        "dy_px": float(getattr(result, "dy_px", 0.0)),
        "radius_deg": metrics.get("radius_deg"),
        "scale_arcsec_per_px": float(getattr(result, "scale_arcsec_per_px", metrics.get("scale_arcsec_per_px", 0.0))),
        "rotation_expected_deg": metrics.get("rotation_expected_deg"),
        "rotation_err_deg": metrics.get("rotation_err_deg"),
        "rotation_tol_deg": metrics.get("rotation_tol_deg"),
        "rotation_rejected": metrics.get("rotation_rejected"),
        "validation_inliers": metrics.get("validation_inliers"),
        "min_validation_inliers": metrics.get("min_validation_inliers"),
        "max_rms_px": metrics.get("max_rms_px"),
        "target_offset_deg": metrics.get("target_offset_deg"),
        "max_center_offset_deg": metrics.get("max_center_offset_deg"),
        "match_tol_arcsec": metrics.get("match_tol_arcsec"),
        "fast_prior": metrics.get("fast_prior"),
        "continuous_prior": metrics.get("continuous_prior"),
        "consensus_count": metrics.get("consensus_count"),
        "consensus_requested": metrics.get("consensus_requested"),
        "consensus_pointing_arcsec": metrics.get("consensus_pointing_arcsec"),
    }
    return info


_GAIA_FAILURE_STATUSES = {
    "NEED_GAIA_AUTH",
    "GAIA_CACHE_MISS",
    "GAIA_LOAD_ERROR",
    "GAIA_TOO_SMALL",
}
_GENERIC_GAIA_ERROR_REASON = "GAIA_ERROR"
_GENERIC_PLATESOLVING_ERROR_REASON = "PLATESOLVING_ERROR"


class _PlatesolvingAbort(RuntimeError):
    def __init__(self, status: str) -> None:
        super().__init__(str(status))
        self.status = str(status)


def _public_platesolving_reason(status: str) -> str:
    st = str(status or "").strip().upper()
    if st in _GAIA_FAILURE_STATUSES:
        return _GENERIC_GAIA_ERROR_REASON
    if st in {"", "EXCEPTION"}:
        return _GENERIC_PLATESOLVING_ERROR_REASON
    return st


class PlatesolvingWorker(BaseWorker):
    """
    Worker de plate solving desacoplado del AppRunner.

    Dependencias inyectadas:
      - get_frame(): devuelve el frame actual (numpy array)
      - get_cfg(): devuelve PlatesolvingConfig
      - get_sep_cfg(): devuelve SepConfig
      - get_observer(): devuelve ObserverConfig
      - publish_state(patch): publica estado/resultados a la UI
    """

    def __init__(
        self,
        *,
        get_frame: Callable[[], Optional[np.ndarray]],
        get_cfg: Callable[[], PlatesolvingConfig],
        get_sep_cfg: Callable[[], SepConfig],
        get_observer: Callable[[], ObserverConfig],
        publish_state: StatePublisherProtocol,
        out_log: Any = None,
    ) -> None:
        super().__init__(name="PlatesolvingWorker")
        self._get_frame = get_frame
        self._get_cfg = get_cfg
        self._get_sep_cfg = get_sep_cfg
        self._get_observer = get_observer
        self._publish_state = publish_state
        self._out_log = out_log
        self._last_confirmed_result: Optional[PlatesolvingResult] = None
        self._last_confirmed_frame: Optional[np.ndarray] = None
        self._request_cancel = threading.Event()
        self._solve_deadline = float("inf")

    def request(self, *, target: Any, obstime_unix: Optional[float] = None) -> None:
        self._request_cancel.clear()
        super().request(target=target, obstime_unix=obstime_unix)

    def cancel_current(self) -> None:
        self._request_cancel.set()

    def _abort_status(self) -> Optional[str]:
        if self._cancel.is_set() or self._request_cancel.is_set():
            return "CANCELLED"
        if time.perf_counter() >= float(self._solve_deadline):
            return "TIMEOUT"
        return None

    def _raise_if_aborted(self) -> None:
        status = self._abort_status()
        if status is not None:
            raise _PlatesolvingAbort(status)

    def stop(self) -> None:
        self._request_cancel.set()
        super().stop()
        self.join(timeout=1.0)

    def _log_input_stats(self, arr: np.ndarray, name: str, enabled: bool) -> None:
        if not enabled:
            return
        a = np.asarray(arr)
        log_info(self._out_log, f"[{name}] shape={a.shape} dtype={a.dtype} C={a.flags['C_CONTIGUOUS']}")
        if a.size == 0:
            log_info(self._out_log, "  EMPTY")
            return
        if a.ndim == 1:
            log_info(self._out_log, f"  1D buffer: min={a.min()} max={a.max()} mean={a.mean():.3g}")
            return
        flat = a.reshape(-1)
        p = np.percentile(flat, [0, 1, 5, 50, 95, 99, 100])
        log_info(self._out_log, f"  min/p1/p5/p50/p95/p99/max = {p}")
        log_info(self._out_log, f"  mean={flat.mean():.3g} std={flat.std():.3g}")
        if a.dtype == np.uint16:
            log_info(self._out_log, f"  sat65535={np.mean(flat == 65535):.4f}")
        if a.dtype == np.uint8:
            log_info(self._out_log, f"  sat255={np.mean(flat == 255):.4f}")

    def _wait_for_distinct_frame(
        self,
        previous: np.ndarray,
        *,
        timeout_s: float,
    ) -> Optional[Tuple[np.ndarray, Time]]:
        deadline = time.perf_counter() + max(0.1, float(timeout_s))
        while time.perf_counter() < deadline:
            if self._abort_status() is not None:
                return None
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                raw16 = ensure_raw16_bayer(frame).copy()
            except (ValueError, TypeError):
                time.sleep(0.01)
                continue
            if raw16.shape != previous.shape or not np.array_equal(raw16, previous):
                return raw16, Time.now()
            time.sleep(0.01)
        return None

    def _wait_for_frame(self, *, timeout_s: float) -> Optional[np.ndarray]:
        """Wait for a valid RAW16 frame after camera connect/reconfigure."""
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while time.perf_counter() <= deadline:
            frame = self._get_frame()
            if frame is not None:
                try:
                    return ensure_raw16_bayer(frame).copy()
                except (ValueError, TypeError):
                    pass
            if self._abort_status() is not None:
                return None
            time.sleep(0.01)
        return None

    def _collect_temporal_detections(
        self,
        initial_frame: np.ndarray,
        *,
        cfg: PlatesolvingConfig,
        sep_cfg: SepConfig,
        diagnostics: Optional[DiagnosticSession] = None,
    ) -> Tuple[Optional[TemporalDetections], np.ndarray, Time]:
        """Collect a distinct-frame window and confirm persistent sources."""
        initial = ensure_raw16_bayer(initial_frame).copy()
        if not bool(getattr(cfg, "temporal_detection_enabled", True)):
            return None, initial, Time.now()

        min_hits = max(10, int(getattr(cfg, "temporal_min_hits", 10)))
        window = max(min_hits, int(getattr(cfg, "temporal_window_frames", 12)))
        timeout_s = max(
            0.5,
            float(getattr(cfg, "temporal_detection_timeout_s", 8.0)),
        )
        frames: List[np.ndarray] = [initial]
        previous = initial
        latest_time = Time.now()
        deadline = time.perf_counter() + timeout_s
        while len(frames) < window:
            self._raise_if_aborted()
            remaining = float(deadline - time.perf_counter())
            if remaining <= 0.0:
                break
            fresh = self._wait_for_distinct_frame(
                previous,
                timeout_s=min(remaining, max(0.15, remaining / (window - len(frames)))),
            )
            if fresh is None:
                continue
            current, current_time = fresh
            frames.append(current)
            previous = current
            latest_time = current_time

        detections = detect_persistent_sep_objects(
            frames,
            sep_bw=int(sep_cfg.bw),
            sep_bh=int(sep_cfg.bh),
            sep_thresh_sigma=float(sep_cfg.thresh_sigma),
            sep_minarea=int(sep_cfg.minarea),
            max_sources=int(getattr(cfg, "max_det", 200)),
            min_hits=min_hits,
            match_radius_px=float(getattr(cfg, "temporal_match_radius_px", 4.0)),
            max_drift_per_frame_px=float(
                getattr(cfg, "temporal_max_drift_per_frame_px", 32.0)
            ),
            min_drift_response=float(
                getattr(cfg, "temporal_min_drift_response", 0.05)
            ),
            progress_cb=(
                (lambda stage, payload: diagnostics.record(stage, **dict(payload or {})))
                if diagnostics is not None
                else None
            ),
        )
        N_det = int(getattr(cfg, "N_det", getattr(cfg, "match_n_det", 30)))
        n_solver = min(N_det, detections.xy.shape[0]) if N_det > 0 else detections.xy.shape[0]
        N_seed = int(getattr(cfg, "N_seed", getattr(cfg, "match_n_seed", 3)))
        seed_count = min(max(3, N_seed), int(n_solver))
        overlay = _detection_overlay(
            detections.xy,
            seed_count=seed_count,
            hits=detections.hits,
            frame_count=detections.frame_count,
        )
        self._publish_state(
            {
                "platesolving": {
                    "overlay": overlay,
                    "debug_info": {
                        "status": "TEMPORAL_DETECTION_READY",
                        "temporal_frames": int(detections.frame_count),
                        "temporal_required_hits": int(detections.required_hits),
                        "temporal_confirmed": int(detections.xy.shape[0]),
                        "temporal_seeds": int(seed_count),
                    },
                }
            }
        )
        log_info(
            self._out_log,
            "Platesolving: temporal detection "
            f"frames={detections.frame_count}/{window} "
            f"required_hits={detections.required_hits} "
            f"confirmed={detections.xy.shape[0]} seeds={seed_count} "
            f"drift_failures={detections.drift_failures}",
        )
        return detections, detections.reference_frame, latest_time

    def _confirm_initial_solution(
        self,
        first: PlatesolvingResult,
        first_frame: np.ndarray,
        *,
        target: Any,
        cfg: PlatesolvingConfig,
        sep_cfg: SepConfig,
        observer: ObserverConfig,
        diagnostics: Optional[DiagnosticSession] = None,
    ) -> Tuple[PlatesolvingResult, np.ndarray]:
        requested = max(1, int(getattr(cfg, "initial_consensus_count", 3)))
        if requested <= 1 or not bool(first.success):
            return first, first_frame
        timeout_s = max(0.2, float(getattr(cfg, "initial_consensus_timeout_s", 8.0)))
        per_frame_timeout = timeout_s / float(max(1, requested - 1))
        accepted: List[PlatesolvingResult] = [first]
        previous_frame = first_frame
        prior = first
        max_pointing = 0.0
        max_scale = 0.0
        max_roll = 0.0

        for confirmation_idx in range(2, requested + 1):
            self._raise_if_aborted()
            fresh = self._wait_for_distinct_frame(previous_frame, timeout_s=per_frame_timeout)
            if fresh is None:
                metrics = dict(getattr(prior, "metrics", {}) or {})
                metrics.update(
                    {
                        "consensus_count": float(len(accepted)),
                        "consensus_requested": float(requested),
                    }
                )
                return (
                    replace(
                        prior,
                        success=False,
                        status="INITIAL_CONSENSUS_NO_NEW_FRAME",
                        metrics=metrics,
                    ),
                    previous_frame,
                )
            current_frame, current_time = fresh
            confirmation_temporal: Optional[TemporalDetections] = None
            if bool(getattr(cfg, "temporal_detection_enabled", True)):
                (
                    confirmation_temporal,
                    current_frame,
                    current_time,
                ) = self._collect_temporal_detections(
                    current_frame,
                    cfg=cfg,
                    sep_cfg=sep_cfg,
                    diagnostics=diagnostics,
                )
            if diagnostics is not None:
                diagnostics.save_raw(
                    f"consensus_{confirmation_idx}",
                    current_frame,
                    metadata={
                        "obstime_unix": float(current_time.unix),
                        "temporal_frames": (
                            int(confirmation_temporal.frame_count)
                            if confirmation_temporal is not None
                            else 1
                        ),
                        "temporal_confirmed": (
                            int(confirmation_temporal.xy.shape[0])
                            if confirmation_temporal is not None
                            else None
                        ),
                    },
                )
            verified = verify_plate_from_prior(
                current_frame,
                prior=prior,
                target=target,
                cfg=cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=current_time,
                progress_cb=None,
                temporal_detections=confirmation_temporal,
            )
            self._raise_if_aborted()
            consistency = platesolving_solutions_consistent(
                first,
                verified,
                observer=observer,
                pointing_tol_arcsec=float(getattr(cfg, "consensus_pointing_tol_arcsec", 30.0)),
                scale_tol_frac=float(getattr(cfg, "consensus_scale_tol_frac", 0.02)),
                roll_tol_deg=float(getattr(cfg, "consensus_roll_tol_deg", 3.0)),
            )
            if diagnostics is not None:
                diagnostics.record(
                    "platesolving_consensus",
                    confirmation_index=int(confirmation_idx),
                    result=verified,
                    consistency=consistency,
                )
            if not bool(verified.success) or not bool(consistency.get("ok", False)):
                metrics = dict(getattr(verified, "metrics", {}) or {})
                metrics.update(
                    {
                        "consensus_count": float(len(accepted)),
                        "consensus_requested": float(requested),
                        "consensus_pointing_arcsec": float(consistency.get("pointing_arcsec", float("inf"))),
                        "consensus_scale_frac": float(consistency.get("scale_frac", float("inf"))),
                        "consensus_roll_deg": float(consistency.get("roll_deg", float("inf"))),
                    }
                )
                return (
                    replace(
                        verified,
                        success=False,
                        status="INITIAL_CONSENSUS_MISMATCH",
                        metrics=metrics,
                    ),
                    current_frame,
                )
            max_pointing = max(max_pointing, float(consistency["pointing_arcsec"]))
            max_scale = max(max_scale, float(consistency["scale_frac"]))
            max_roll = max(max_roll, float(consistency["roll_deg"]))
            accepted.append(verified)
            prior = verified
            previous_frame = current_frame
            log_info(
                self._out_log,
                "Platesolving: fast independent confirmation "
                f"{confirmation_idx}/{requested} inliers={verified.n_inliers} "
                f"rms_px={verified.rms_px:.3f} mount_delta={float(consistency['pointing_arcsec']):.2f}arcsec",
            )

        metrics = dict(getattr(prior, "metrics", {}) or {})
        metrics.update(
            {
                "consensus_count": float(len(accepted)),
                "consensus_requested": float(requested),
                "consensus_pointing_arcsec": float(max_pointing),
                "consensus_scale_frac": float(max_scale),
                "consensus_roll_deg": float(max_roll),
            }
        )
        return (
            replace(
                prior,
                status="OK_CONSENSUS",
                overlay=list(first.overlay),
                guides=list(first.guides),
                metrics=metrics,
            ),
            first_frame,
        )

    def _solve_or_verify(
        self,
        raw16: np.ndarray,
        *,
        target: Any,
        cfg: PlatesolvingConfig,
        sep_cfg: SepConfig,
        observer: ObserverConfig,
        obstime: Time,
        diagnostics: Optional[DiagnosticSession] = None,
        temporal_detections: Optional[TemporalDetections] = None,
    ) -> Tuple[PlatesolvingResult, np.ndarray]:
        def _progress(stage: str, payload: Dict[str, Any]) -> None:
            self._raise_if_aborted()
            if diagnostics is not None and str(stage) != "platesolving:checkpoint":
                diagnostics.record(str(stage), **dict(payload or {}))

        prior = self._last_confirmed_result
        if prior is not None:
            if (
                temporal_detections is None
                and
                self._last_confirmed_frame is not None
                and raw16.shape == self._last_confirmed_frame.shape
                and np.array_equal(raw16, self._last_confirmed_frame)
            ):
                fresh = self._wait_for_distinct_frame(
                    raw16,
                    timeout_s=float(getattr(cfg, "initial_consensus_timeout_s", 8.0)),
                )
                if fresh is not None:
                    raw16, obstime = fresh
                    if diagnostics is not None:
                        diagnostics.save_raw(
                            "continuous_fresh_input",
                            raw16,
                            metadata={"obstime_unix": float(obstime.unix)},
                        )
            fast = verify_plate_from_prior(
                raw16,
                prior=prior,
                target=target,
                cfg=cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=obstime,
                progress_cb=_progress,
                temporal_detections=temporal_detections,
            )
            consistency = platesolving_solutions_consistent(
                prior,
                fast,
                observer=observer,
                pointing_tol_arcsec=float(getattr(cfg, "consensus_pointing_tol_arcsec", 30.0)),
                scale_tol_frac=float(getattr(cfg, "consensus_scale_tol_frac", 0.02)),
                roll_tol_deg=float(getattr(cfg, "consensus_roll_tol_deg", 3.0)),
            )
            if diagnostics is not None:
                diagnostics.record(
                    "platesolving_fast_prior",
                    result=fast,
                    consistency=consistency,
                )
            if bool(fast.success) and bool(consistency.get("ok", False)):
                metrics = dict(getattr(fast, "metrics", {}) or {})
                metrics.update(
                    {
                        "continuous_prior": 1.0,
                        "consensus_count": float(
                            max(3, int((getattr(prior, "metrics", {}) or {}).get("consensus_count", 3)))
                        ),
                        "consensus_pointing_arcsec": float(consistency["pointing_arcsec"]),
                    }
                )
                return replace(fast, status="OK_FAST_CONTINUOUS", metrics=metrics), raw16
            log_info(
                self._out_log,
                "Platesolving: previous field no longer continuous; running full solve and new consensus",
            )

        full = solve_plate(
            raw16,
            target=target,
            cfg=cfg,
            sep_cfg=sep_cfg,
            observer=observer,
            obstime=obstime,
            progress_cb=_progress,
            temporal_detections=temporal_detections,
        )
        if diagnostics is not None:
            diagnostics.record("platesolving_full_solve", result=full)
        return self._confirm_initial_solution(
            full,
            raw16,
            target=target,
            cfg=cfg,
            sep_cfg=sep_cfg,
            observer=observer,
            diagnostics=diagnostics,
        )

    def _handle_request(self, request: Dict[str, Any]) -> None:
        diagnostics: Optional[DiagnosticSession] = None
        self._publish_state(
            {
                "platesolving": {
                    "busy": True,
                    "status": PlatesolvingStatus.RUNNING,
                    "reason": None,
                    "overlay": [],
                    "debug_jpeg": None,
                    "debug_info": None,
                }
            }
        )

        target = request.get("target", None)
        if target is None:
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": "NO_TARGET",
                        "last_ok": False,
                        "debug_jpeg": None,
                        "debug_info": {"status": "NO_TARGET"},
                    }
                }
            )
            log_error(self._out_log, "Platesolving: ERR_NO_TARGET")
            return

        cfg = self._get_cfg()
        total_timeout_s = max(1.0, float(getattr(cfg, "total_timeout_s", 120.0)))
        self._solve_deadline = time.perf_counter() + total_timeout_s
        raw16 = self._wait_for_frame(
            timeout_s=float(getattr(cfg, "frame_wait_timeout_s", 3.0))
        )
        if raw16 is None:
            abort_status = self._abort_status()
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": abort_status or "NO_FRAME",
                        "last_ok": False,
                        "debug_jpeg": None,
                        "debug_info": {"status": abort_status or "NO_FRAME"},
                    }
                }
            )
            if abort_status is not None:
                log_info(self._out_log, f"Platesolving: {abort_status}")
            else:
                log_error(self._out_log, "Platesolving: ERR_NO_FRAME")
            return

        try:
            sep_cfg = self._get_sep_cfg()
            observer = self._get_observer()
            # A stacked mosaic shows the sky at its *reference* frame time, not
            # now: frames are aligned onto that first frame. Using Time.now()
            # would offset the fitted center by the sidereal drift accumulated
            # while stacking, which for a minute-long stack is several arcmin.
            requested_time = request.get("obstime_unix")
            if requested_time is not None and np.isfinite(float(requested_time)):
                obstime = Time(float(requested_time), format="unix", scale="utc")
            else:
                obstime = Time.now()
            # Keep the first native frame available.  Temporal confirmation is
            # a useful hot-pixel rejection layer, but it must not make a field
            # unsolvable when the mount/camera motion estimate is unreliable
            # (for example, faint stars with a changing SEP source count).
            initial_raw16 = raw16.copy()
            debug_stats = bool(getattr(cfg, "debug_input_stats", False))
            self._log_input_stats(raw16, "frame(raw16)", debug_stats)
            diagnostics = DiagnosticSession(
                root_dir=str(getattr(cfg, "diagnostics_dir", "stack_output/goto_diagnostics")),
                operation="platesolve",
                enabled=bool(getattr(cfg, "diagnostics_enabled", True)),
                context={
                    "target": target,
                    "platesolving_config": cfg,
                    "sep_config": sep_cfg,
                    "observer": observer,
                },
                out_log=self._out_log,
            )
            diagnostics.save_raw(
                "platesolve_input",
                raw16,
                metadata={"obstime_unix": float(obstime.unix)},
            )
            temporal_detections, raw16, obstime = self._collect_temporal_detections(
                raw16,
                cfg=cfg,
                sep_cfg=sep_cfg,
                diagnostics=diagnostics,
            )
            if (
                temporal_detections is not None
                and temporal_detections.xy.shape[0] < 3
            ):
                # Fall back to the original frame rather than the aligned
                # temporal median, which can be nearly flat when drift
                # estimation failed.  The full plate solver still requires a
                # geometrically consistent Gaia match before reporting OK.
                diagnostics.record(
                    "platesolve_temporal_fallback",
                    confirmed=int(temporal_detections.xy.shape[0]),
                    drift_failures=int(temporal_detections.drift_failures),
                    reason="TEMPORAL_CONFIRMATION_EMPTY",
                )
                log_info(
                    self._out_log,
                    "Platesolving: temporal confirmation empty; "
                    "falling back to the original RAW16 frame",
                )
                temporal_detections = None
                raw16 = initial_raw16
            if temporal_detections is not None:
                diagnostics.save_raw(
                    "platesolve_temporal_reference",
                    raw16,
                    metadata={
                        "obstime_unix": float(obstime.unix),
                        "frames": int(temporal_detections.frame_count),
                        "required_hits": int(temporal_detections.required_hits),
                        "confirmed": int(temporal_detections.xy.shape[0]),
                    },
                )
        except _PlatesolvingAbort as exc:
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": exc.status,
                        "last_ok": False,
                        "debug_info": {"status": exc.status},
                    }
                }
            )
            if diagnostics is not None:
                diagnostics.close(exc.status)
            log_info(self._out_log, f"Platesolving: {exc.status}")
            return
        except Exception as exc:
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": _GENERIC_PLATESOLVING_ERROR_REASON,
                        "last_ok": False,
                        "debug_jpeg": None,
                        "debug_info": {"status": "FRAME_OR_CONFIG_ERROR"},
                    }
                }
            )
            log_error(self._out_log, "Platesolving: frame/config preparation failed", exc)
            if diagnostics is not None:
                diagnostics.close("FRAME_OR_CONFIG_ERROR", error=repr(exc))
            return

        try:
            result, result_frame = self._solve_or_verify(
                raw16,
                target=target,
                cfg=cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=obstime,
                diagnostics=diagnostics,
                temporal_detections=temporal_detections,
            )
        except _PlatesolvingAbort as exc:
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": exc.status,
                        "last_ok": False,
                    }
                }
            )
            if diagnostics is not None:
                diagnostics.close(exc.status)
            log_info(self._out_log, f"Platesolving: {exc.status}")
            return
        except Exception as exc:
            self._publish_state(
                {
                    "platesolving": {
                        "busy": False,
                        "status": PlatesolvingStatus.FAIL,
                        "reason": _GENERIC_PLATESOLVING_ERROR_REASON,
                        "last_ok": False,
                    }
                }
            )
            log_error(self._out_log, "Platesolving: failed", exc)
            if diagnostics is not None:
                diagnostics.close("EXCEPTION", error=repr(exc))
            return

        if bool(getattr(result, "success", False)):
            self._last_confirmed_result = result
            self._last_confirmed_frame = result_frame.copy()
        else:
            self._last_confirmed_result = None
            self._last_confirmed_frame = None

        result_ok = bool(getattr(result, "success", False))
        result_status = str(getattr(result, "status", "UNKNOWN"))
        public_reason = _public_platesolving_reason(result_status)
        try:
            debug_jpeg = _render_platesolving_debug_jpeg(
                result_frame,
                list(getattr(result, "overlay", []) or []),
            )
            debug_info = _build_platesolving_debug_info(result)
            if diagnostics is not None and diagnostics.path_str is not None:
                debug_info["diagnostics_dir"] = diagnostics.path_str
        except Exception as exc:
            debug_jpeg = None
            debug_info = {"status": result_status, "debug_error": type(exc).__name__}
            log_error(self._out_log, "Platesolving: debug output failed", exc)

        self._publish_state(
            {
                "platesolving": {
                    "busy": False,
                    "status": PlatesolvingStatus.OK if result_ok else PlatesolvingStatus.FAIL,
                    "reason": None if result_ok else public_reason,
                    "last_ok": result_ok,
                    "theta_deg": float(getattr(result, "theta_deg", 0.0)),
                    "dx_px": float(getattr(result, "dx_px", 0.0)),
                    "dy_px": float(getattr(result, "dy_px", 0.0)),
                    "resp": float(getattr(result, "response", 0.0)),
                    "n_inliers": int(getattr(result, "n_inliers", 0)),
                    "rms_px": float(getattr(result, "rms_px", 0.0)),
                    "overlay": list(getattr(result, "overlay", []) or []),
                    "guides": list(getattr(result, "guides", []) or []),
                    "debug_jpeg": debug_jpeg,
                    "debug_info": debug_info,
                    "center_ra_deg": float(getattr(result, "center_ra_deg", 0.0)),
                    "center_dec_deg": float(getattr(result, "center_dec_deg", 0.0)),
                }
            }
        )
        self._publish_state({"platesolving_result": result})
        if diagnostics is not None:
            diagnostics.close(
                result_status,
                success=result_ok,
                result=result,
                result_frame_artifact=(
                    diagnostics.save_raw(
                        "platesolve_result_frame",
                        result_frame,
                        metadata={"status": result_status},
                    )
                    if not np.array_equal(raw16, result_frame)
                    else None
                ),
            )

        status = str(getattr(result, "status", "UNKNOWN"))
        success = bool(getattr(result, "success", False))
        resp = float(getattr(result, "response", 0.0))
        n_inliers = int(getattr(result, "n_inliers", 0))
        rms_px = float(getattr(result, "rms_px", 0.0))
        if success:
            metrics = dict(getattr(result, "metrics", {}) or {})
            az_match, alt_match = _icrs_to_altaz_app_deg(
                float(getattr(result, "center_ra_deg", 0.0)),
                float(getattr(result, "center_dec_deg", 0.0)),
                observer=observer,
                obstime=_result_obstime(result, fallback=obstime),
            )
            log_info(
                self._out_log,
                "Platesolving: OK "
                f"status={status} resp={resp:.3g} inliers={n_inliers} rms_px={rms_px:.3g} "
                f"valid={int(metrics.get('validation_inliers', 0) or 0)} "
                f"scale={float(getattr(result, 'scale_arcsec_per_px', 0.0)):.4f}arcsec/px "
                f"target_offset={float(metrics.get('target_offset_deg', float('nan'))):.4f}deg "
                f"match_az={az_match:.4f}deg match_alt={alt_match:.4f}deg",
            )
        else:
            if public_reason != status:
                log_error(
                    self._out_log,
                    f"Platesolving: ERR reason={public_reason} status={status} "
                    f"resp={resp:.3g} inliers={n_inliers} rms_px={rms_px:.3g}",
                )
            else:
                log_error(
                    self._out_log,
                    f"Platesolving: ERR status={status} resp={resp:.3g} inliers={n_inliers} rms_px={rms_px:.3g}",
                )
