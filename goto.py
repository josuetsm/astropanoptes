# goto.py
# -*- coding: utf-8 -*-
"""GoTo + calibration for Astropanoptes (Alt-Az, no absolute encoders).

This module is intentionally *self-contained*: it does not import AppRunner.
AppRunner (or any orchestrator) should provide callbacks for:
  - get_live_frame(): -> np.ndarray (uint16 RAW16 Bayer; platesolving will use SEP)
  - move_steps(axis: Axis, direction: int, steps: int, delay_us: int) -> None/str
  - rate_mount(az_rate_steps_s: float, alt_rate_steps_s: float) -> None/str (optional)
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

A GoTo uses the fitted model without plate-solving feedback:
  1) estimate current mount AltAz from the model
  2) predict target AltAz at the expected slew completion time
  3) convert the full error to motor steps via inv(J)
  4) execute one parallel AZ/ALT MOVE

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

import csv
import math
import os
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
from gaia_cache import BRIGHT_STAR_SUPPLEMENT

# We reuse target parsing & observer from your plate-solver module.
from platesolving import (
    ObserverConfig,
    PlatesolvingResult,
    expected_field_rotation_deg,
    parse_target_to_icrs,
    platesolving_solutions_consistent,
    solve_plate,
    verify_plate_from_prior,
    _build_platesolving_debug_info,
    _render_platesolving_debug_jpeg,
)
from logging_utils import log_error, log_info
from mount_arduino import estimate_firmware_move_duration_s
from workers import BaseWorker
from imaging import ensure_raw16_bayer, estimate_sensor_drift_from_stack
from goto_diagnostics import DiagnosticSession
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

_GOTO_CSV_LOG_LOCK = threading.Lock()


def _autocal_frame_is_crowded(
    *,
    active_fraction: float,
    star_count: int,
    max_sources: int,
    crowded_limit: float = 0.08,
) -> bool:
    """Return whether a detection frame is genuinely source-crowded.

    Player One RAW16 data is quantized in coarse ADU steps.  After the median
    prefilter SEP can report a very small ``globalrms`` (often 1 ADU), making a
    harmless quantization step look like a large above-threshold pixel area.
    Active area alone must therefore not drive exposure/gain downward.  Treat
    the frame as crowded only when the source list is also close to its cap.
    """
    if not np.isfinite(float(active_fraction)):
        return False
    cap = max(1, int(max_sources))
    crowded_sources = max(20, int(math.ceil(0.75 * float(cap))))
    return bool(
        float(active_fraction) > float(crowded_limit)
        and int(star_count) >= crowded_sources
    )


def _autocal_should_tune_exposure(
    params: Dict[str, Any],
    *,
    autocal_ps_mode: str,
) -> bool:
    """Return False: GoTo operations never own camera exposure or gain.

    Exposure and gain are operator settings from the Camera panel.  Keeping
    this policy in the worker (and not only in the UI) also protects legacy or
    scripted AutoCal requests from silently changing them.
    """
    _ = params, autocal_ps_mode
    return False

_GOTO_MANUAL_SAMPLE_CSV_FIELDS = [
    "ts_unix",
    "ts_utc",
    "sample_idx",
    "steps_az",
    "steps_alt",
    "az_deg",
    "alt_deg",
    "roll_deg",
    "synced",
    "ref_steps_az",
    "ref_steps_alt",
    "ref_az_mount_deg",
    "ref_alt_mount_deg",
    "last_direction_az",
    "last_direction_alt",
    "backlash_steps_az",
    "backlash_steps_alt",
    "R00",
    "R01",
    "R02",
    "R10",
    "R11",
    "R12",
    "R20",
    "R21",
    "R22",
]

_GOTO_MODEL_FIT_CSV_FIELDS = [
    "ts_unix",
    "ts_utc",
    "fit_kind",
    "ok",
    "reason",
    "min_samples",
    "ridge",
    "total_samples",
    "used_samples",
    "outliers",
    "J00",
    "J01",
    "J10",
    "J11",
    "J00_err",
    "J01_err",
    "J10_err",
    "J11_err",
    "model_fit_rms_az_deg",
    "model_fit_rms_alt_deg",
    "model_fit_rms_arcsec",
    "model_non_orthogonality_deg",
    "model_non_orthogonality_err_deg",
    "model_roll_deg",
    "model_roll_err_deg",
    "model_roll_samples",
    "model_pitch_deg",
    "model_pitch_err_deg",
    "model_yaw_deg",
    "model_yaw_err_deg",
    "periodic_az_sin_deg",
    "periodic_az_cos_deg",
    "periodic_alt_sin_deg",
    "periodic_alt_cos_deg",
    "periodic_model_samples",
    "R00",
    "R01",
    "R02",
    "R10",
    "R11",
    "R12",
    "R20",
    "R21",
    "R22",
]


def _goto_logs_dir() -> str:
    v = str(os.environ.get("ASTROPANOPTES_GOTO_LOG_DIR", "")).strip()
    if v:
        return v
    return os.path.join("stack_output", "goto_logs")


def _utc_str(ts_unix: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts_unix)))


def _append_csv_log_row(filename: str, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    try:
        log_dir = _goto_logs_dir()
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        with _GOTO_CSV_LOG_LOCK:
            expected_fields = list(fieldnames)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, "r", newline="", encoding="utf-8") as existing:
                    reader = csv.DictReader(existing)
                    existing_fields = list(reader.fieldnames or [])
                    old_rows = list(reader) if existing_fields != expected_fields else []
                if existing_fields != expected_fields:
                    # Evolve operational logs without corrupting older rows.
                    # Missing new columns are left blank; unknown legacy
                    # columns are intentionally ignored by DictWriter.
                    tmp_path = path + ".schema.tmp"
                    with open(tmp_path, "w", newline="", encoding="utf-8") as migrated:
                        writer = csv.DictWriter(
                            migrated,
                            fieldnames=expected_fields,
                            extrasaction="ignore",
                        )
                        writer.writeheader()
                        writer.writerows(old_rows)
                    os.replace(tmp_path, path)
            write_header = (not os.path.exists(path)) or (os.path.getsize(path) <= 0)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=expected_fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(dict(row))
    except Exception as exc:
        key = f"goto_csv_log_{str(filename).replace('.', '_')}"
        log_error(None, f"GoTo: failed to append CSV log ({filename})", exc, throttle_s=5.0, throttle_key=key)


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


def _circular_mean_deg(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("expected at least one angle")
    r = np.deg2rad(v)
    s = float(np.mean(np.sin(r)))
    c = float(np.mean(np.cos(r)))
    if abs(s) < 1e-15 and abs(c) < 1e-15:
        return 0.0
    return _wrap_deg_360(math.degrees(math.atan2(s, c)))


def _circular_std_deg(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.0
    r = np.deg2rad(v)
    s = float(np.mean(np.sin(r)))
    c = float(np.mean(np.cos(r)))
    R = float(math.hypot(s, c))
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(R)))))


def _non_orthogonality_deg_from_params(j00: float, j01: float, j10: float, j11: float) -> float:
    c0 = np.array([float(j00), float(j10)], dtype=np.float64)
    c1 = np.array([float(j01), float(j11)], dtype=np.float64)
    n0 = float(np.linalg.norm(c0))
    n1 = float(np.linalg.norm(c1))
    if n0 <= 1e-18 or n1 <= 1e-18:
        return 0.0
    cosang = float(np.clip(float(np.dot(c0, c1)) / (n0 * n1), -1.0, 1.0))
    ang = float(np.degrees(math.acos(cosang)))
    return float(ang - 90.0)


def _non_orthogonality_deg_from_J(J: np.ndarray) -> float:
    A = np.asarray(J, dtype=np.float64)
    if A.shape != (2, 2):
        return 0.0
    return _non_orthogonality_deg_from_params(A[0, 0], A[0, 1], A[1, 0], A[1, 1])


def _altaz_deg_to_unit_vec(az_deg: float, alt_deg: float) -> np.ndarray:
    """AltAz (deg) -> ENU unit vector [x_east, y_north, z_up]."""
    az = math.radians(float(az_deg))
    alt = math.radians(float(alt_deg))
    c_alt = math.cos(alt)
    x = c_alt * math.sin(az)
    y = c_alt * math.cos(az)
    z = math.sin(alt)
    return np.array([x, y, z], dtype=np.float64)


def _unit_vec_to_altaz_deg(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(vv))
    if (not np.isfinite(n)) or n <= 0.0:
        raise ValueError("invalid vector norm")
    vv = vv / n
    x = float(vv[0])
    y = float(vv[1])
    z = _clamp(float(vv[2]), -1.0, 1.0)
    alt_deg = math.degrees(math.asin(z))
    az_deg = _wrap_deg_360(math.degrees(math.atan2(x, y)))
    return np.array([az_deg, alt_deg], dtype=np.float64)


def _coerce_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Return a finite right-handed orthonormal 3x3 rotation matrix."""
    A = np.asarray(R, dtype=np.float64)
    if A.shape != (3, 3) or (not np.all(np.isfinite(A))):
        return np.eye(3, dtype=np.float64)
    try:
        U, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    Rn = U @ Vt
    if float(np.linalg.det(Rn)) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    if not np.all(np.isfinite(Rn)):
        return np.eye(3, dtype=np.float64)
    return Rn.astype(np.float64)


def _rotate_altaz_deg(az_alt_deg: np.ndarray, R: np.ndarray) -> np.ndarray:
    aa = _as_array2(az_alt_deg)
    v = _altaz_deg_to_unit_vec(float(aa[0]), float(aa[1]))
    vr = np.asarray(R, dtype=np.float64) @ v
    return _unit_vec_to_altaz_deg(vr)


def _fit_rotation_kabsch(src_unit: np.ndarray, dst_unit: np.ndarray) -> Optional[np.ndarray]:
    """Best-fit rotation R with dst ~= R @ src for unit vectors."""
    src = np.asarray(src_unit, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_unit, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or int(src.shape[0]) < 3:
        return None
    finite = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
    src = src[finite]
    dst = dst[finite]
    if int(src.shape[0]) < 3:
        return None

    H = src.T @ dst
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if float(np.linalg.det(R)) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    if not np.all(np.isfinite(R)):
        return None
    return _coerce_rotation_matrix(R)


def _rotation_rotvec_deg(R: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector in degrees (ENU axes)."""
    A = _coerce_rotation_matrix(R)
    tr = float(np.trace(A))
    c = _clamp(0.5 * (tr - 1.0), -1.0, 1.0)
    ang = float(math.acos(c))
    if ang <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    s = float(math.sin(ang))
    if abs(s) <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            float(A[2, 1] - A[1, 2]),
            float(A[0, 2] - A[2, 0]),
            float(A[1, 0] - A[0, 1]),
        ],
        dtype=np.float64,
    ) / (2.0 * s)
    rv = axis * ang
    return np.degrees(rv).astype(np.float64)


def _rotvec_deg_to_rotation_matrix(rotvec_deg: np.ndarray) -> np.ndarray:
    """Rotation matrix from axis-angle rotvec in degrees (ENU axes)."""
    rv_deg = np.asarray(rotvec_deg, dtype=np.float64).reshape(3,)
    if not np.all(np.isfinite(rv_deg)):
        return np.eye(3, dtype=np.float64)
    rv = np.deg2rad(rv_deg)
    ang = float(np.linalg.norm(rv))
    if ang <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / ang
    kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    I = np.eye(3, dtype=np.float64)
    R = I + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)
    return _coerce_rotation_matrix(R)


def _limit_rotation_tilt_ns_oe_deg(R: np.ndarray, *, max_tilt_deg: float) -> np.ndarray:
    """
    Clamp NS/OE tilt (ENU rotvec x/y components) to a hard limit.

    We keep the z component (azimuth encoder zero offset) unconstrained.
    """
    R0 = _coerce_rotation_matrix(R)
    lim = float(max_tilt_deg)
    if (not np.isfinite(lim)) or lim <= 0.0:
        return R0
    rv = _rotation_rotvec_deg(R0)
    rv_limited = rv.copy()
    rv_limited[0] = _clamp(float(rv_limited[0]), -lim, +lim)  # tilt NS
    rv_limited[1] = _clamp(float(rv_limited[1]), -lim, +lim)  # tilt OE
    return _rotvec_deg_to_rotation_matrix(rv_limited)


def _flatten_points(xy: np.ndarray) -> np.ndarray:
    P = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return P


def _peak_score_from_hist(counts: np.ndarray, *, topk: int = 6) -> float:
    c = counts.astype(np.float64, copy=False)
    N = float(np.sum(c))
    if N <= 0.0:
        return 0.0
    nb = int(c.size)
    if nb <= 0:
        return 0.0
    expected = N / float(nb)
    excess = np.maximum(c - expected, 0.0)
    k = int(min(max(1, int(topk)), nb))
    top = np.partition(excess, -k)[-k:]
    return float(np.sum(top) / N)


def _angle_sweep_best_direction(
    xy: np.ndarray,
    *,
    deg_min: float = 0.0,
    deg_max: float = 180.0,
    deg_step: float = 0.1,
    bin_width_px: float = 2.0,
    topk_bins_for_score: int = 6,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Returns best dict plus (degs, scores).
    """
    P = _flatten_points(xy)
    if P.size == 0:
        raise ValueError("no points for angle sweep")
    P0 = P - np.median(P, axis=0, keepdims=True)
    degs = np.arange(float(deg_min), float(deg_max), float(deg_step), dtype=np.float64)
    if degs.size == 0:
        raise ValueError("empty angle grid")
    scores = np.zeros_like(degs)

    best: Dict[str, Any] = {
        "deg": None,
        "u": None,
        "v": None,
        "s": None,
        "score": -1.0,
        "hist": None,
        "edges": None,
    }

    rmax = float(np.max(np.linalg.norm(P0, axis=1)))
    if not np.isfinite(rmax) or rmax < 1e-6:
        raise ValueError("degenerate point cloud")

    bw = float(bin_width_px)
    if not np.isfinite(bw) or bw <= 0.0:
        bw = 1.0

    s_min = -rmax
    s_max = +rmax
    nbins = int(np.ceil((s_max - s_min) / bw))
    nbins = max(nbins, 16)

    for i, deg in enumerate(degs):
        th = np.deg2rad(deg)
        u = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
        v = np.array([-np.sin(th), np.cos(th)], dtype=np.float64)
        s = P0 @ v
        counts, edges = np.histogram(s, bins=nbins, range=(s_min, s_max))
        score = _peak_score_from_hist(counts, topk=topk_bins_for_score)
        scores[i] = float(score)
        if float(score) > float(best["score"]):
            best.update(
                {
                    "deg": float(deg),
                    "u": u,
                    "v": v,
                    "s": s,
                    "score": float(score),
                    "hist": counts,
                    "edges": edges,
                }
            )

    return best, degs, scores


def _robust_line_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Theil-Sen slope (median of pairwise slopes).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 2:
        return 0.0
    slopes: List[np.ndarray] = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        m = dx != 0
        if np.any(m):
            slopes.append((y[i + 1 :][m] - y[i]) / dx[m])
    if not slopes:
        return 0.0
    slopes_cat = np.concatenate(slopes, axis=0)
    return float(np.median(slopes_cat))


def _drift_to_az_alt(
    vx: float,
    vy: float,
    *,
    phi_deg: float,
    omega_arcsec_s: float,
    scale_arcsec_per_px: float,
    dedup_tol_deg: float = 1e-3,
    sort_by_forward_err: bool = True,
) -> List[Tuple[float, float]]:
    """
    Closed form inversion from drift (vx, vy) to (az, alt).

    Conventions:
      - +x right (east), +y up
      - omega in arcsec/s
      - scale in arcsec/px
    Returns 0, 1 or 2 solutions.
    """
    phi = math.radians(float(phi_deg))
    C = math.cos(phi)
    S = math.sin(phi)
    if abs(C) < 1e-12:
        log_error(None, "GoTo: drift_to_az_alt degenerate (cos(phi) ~ 0)")
        return []

    omega = float(omega_arcsec_s)
    scale = float(scale_arcsec_per_px)
    if omega <= 0.0 or scale <= 0.0:
        log_error(None, "GoTo: drift_to_az_alt invalid omega/scale")
        return []

    u = (float(vy) * scale) / (omega * C)
    q = (float(vx) * scale) / omega

    if abs(u) > 1.0 + 1e-9:
        log_error(None, "GoTo: drift_to_az_alt no physical solution (|sin(az)| > 1)")
        return []
    u = float(max(-1.0, min(1.0, u)))

    c0 = math.sqrt(max(0.0, 1.0 - u * u))
    cos_az_candidates = (+c0, -c0)

    sols: List[Tuple[float, float]] = []
    for c in cos_az_candidates:
        az = math.atan2(u, c)
        az_deg = float(math.degrees(az) % 360.0)

        K = C * c
        R2 = K * K + S * S
        if q * q > R2 + 1e-12:
            continue
        if abs(S) < 1e-12:
            continue

        sqrt_term = abs(S) * math.sqrt(max(0.0, R2 - q * q))
        for sh in ((-q * K + sqrt_term) / R2, (-q * K - sqrt_term) / R2):
            if sh < -1e-9 or sh > 1.0 + 1e-9:
                continue
            sh = float(max(0.0, min(1.0, sh)))
            ch = (q + K * sh) / S
            if ch < -1e-9:
                continue
            ch = float(max(0.0, min(1.0, ch)))
            alt_deg = float(math.degrees(math.atan2(sh, ch)))
            if 0.0 <= alt_deg <= 90.0:
                sols.append((az_deg, alt_deg))

    if not sols:
        log_error(None, "GoTo: drift_to_az_alt no physical solution")
        return []

    def _circ_dist_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    def _forward_err(az_deg: float, alt_deg: float) -> float:
        az = math.radians(az_deg)
        alt = math.radians(alt_deg)
        d_alt = omega * C * math.sin(az)
        d_x = omega * (S * math.cos(alt) - C * math.sin(alt) * math.cos(az))
        vx2 = d_x / scale
        vy2 = d_alt / scale
        return (vx2 - float(vx)) ** 2 + (vy2 - float(vy)) ** 2

    uniq: List[Tuple[float, float]] = []
    for az_deg, alt_deg in sols:
        is_new = True
        for az2, alt2 in uniq:
            if _circ_dist_deg(az_deg, az2) <= float(dedup_tol_deg) and abs(alt_deg - alt2) <= float(dedup_tol_deg):
                is_new = False
                break
        if is_new:
            uniq.append((az_deg, alt_deg))

    if sort_by_forward_err and len(uniq) > 1:
        uniq.sort(key=lambda p: (_forward_err(p[0], p[1]), p[0], p[1]))
    return uniq


def _R_true_to_app_deg(h_true_deg: float, *, P_hPa: float, T_C: float) -> float:
    """
    Refraction R(h_true) in degrees, where: h_app = h_true + R(h_true).

    Bennett-style approximation (reasonable for h_true >= ~5°; avoid near horizon).
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


def _dRdh_true(h_true_deg: float, *, P_hPa: float, T_C: float) -> float:
    """Derivative dR/dh (deg/deg) by finite difference."""
    eps = 1e-3
    R1 = _R_true_to_app_deg(h_true_deg, P_hPa=P_hPa, T_C=T_C)
    R2 = _R_true_to_app_deg(h_true_deg + eps, P_hPa=P_hPa, T_C=T_C)
    return float((R2 - R1) / eps)


def _drift_to_az_alt_refracted(
    vx: float,
    vy: float,
    *,
    phi_deg: float,
    omega_arcsec_s: float,
    scale_arcsec_per_px: float,
    P_hPa: float = 1013.25,
    T_C: float = 15.0,
    max_iter: int = 12,
    lm_lambda: float = 1e-2,
    dedup_tol_deg: float = 1e-3,
    sort_by_forward_err: bool = True,
) -> List[Tuple[float, float]]:
    """
    Refraction-aware inversion from drift (vx, vy) to (az, alt_app).

    Returns 0, 1 or 2 solutions; altitude is apparent (refracted) to match
    parse_target_to_icrs and the GoTo model's AltAz convention.
    """
    phi = math.radians(float(phi_deg))
    C = math.cos(phi)
    S = math.sin(phi)
    if abs(C) < 1e-12:
        log_error(None, "GoTo: drift_to_az_alt_refracted degenerate (cos(phi) ~ 0)")
        return []

    omega = float(omega_arcsec_s)
    scale = float(scale_arcsec_per_px)
    if omega <= 0.0 or scale <= 0.0:
        log_error(None, "GoTo: drift_to_az_alt_refracted invalid omega/scale")
        return []

    def _wrap360(a_deg: float) -> float:
        return float(a_deg) % 360.0

    def _circ_dist_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    def _forward_refracted(az_deg: float, alt_true_deg: float) -> Tuple[float, float]:
        az = math.radians(float(az_deg))
        h_true = float(alt_true_deg)

        d_alt_true = omega * C * math.sin(az)
        d_x_true = omega * (S * math.cos(math.radians(h_true)) - C * math.sin(math.radians(h_true)) * math.cos(az))

        R = _R_true_to_app_deg(h_true, P_hPa=float(P_hPa), T_C=float(T_C))
        dR = _dRdh_true(h_true, P_hPa=float(P_hPa), T_C=float(T_C))
        h_app = h_true + R

        d_alt_app = (1.0 + dR) * d_alt_true

        ch_true = math.cos(math.radians(h_true))
        ch_app = math.cos(math.radians(h_app))
        if ch_true < 1e-8:
            scale_x = 1.0
        else:
            scale_x = ch_app / ch_true
        d_x_app = scale_x * d_x_true

        vx_pred = d_x_app / scale
        vy_pred = d_alt_app / scale
        return vx_pred, vy_pred

    def _sse(az_deg: float, alt_true_deg: float) -> float:
        vx2, vy2 = _forward_refracted(az_deg, alt_true_deg)
        return (vx2 - float(vx)) ** 2 + (vy2 - float(vy)) ** 2

    def _gauss_newton_lm(az0: float, alt0: float) -> Tuple[float, float, float]:
        az = _wrap360(float(az0))
        alt = float(alt0)
        lam = float(lm_lambda)

        for _ in range(int(max_iter)):
            if alt > 89.9:
                alt = 89.9
            if alt < -0.5:
                alt = -0.5

            vx0, vy0 = _forward_refracted(az, alt)
            r0x = vx0 - float(vx)
            r0y = vy0 - float(vy)

            da = 1e-3
            dh = 1e-3
            vx_a, vy_a = _forward_refracted(_wrap360(az + da), alt)
            vx_h, vy_h = _forward_refracted(az, alt + dh)

            j00 = (vx_a - vx0) / da
            j10 = (vy_a - vy0) / da
            j01 = (vx_h - vx0) / dh
            j11 = (vy_h - vy0) / dh

            a00 = j00 * j00 + j10 * j10 + lam
            a01 = j00 * j01 + j10 * j11
            a11 = j01 * j01 + j11 * j11 + lam
            b0 = -(j00 * r0x + j10 * r0y)
            b1 = -(j01 * r0x + j11 * r0y)

            det = a00 * a11 - a01 * a01
            if abs(det) < 1e-18:
                break

            dp0 = (b0 * a11 - b1 * a01) / det
            dp1 = (-b0 * a01 + b1 * a00) / det

            az_try = _wrap360(az + dp0)
            alt_try = alt + dp1

            s0 = r0x * r0x + r0y * r0y
            s1 = _sse(az_try, alt_try)
            if s1 < s0:
                az, alt = az_try, alt_try
                lam *= 0.5
                if abs(dp0) < 1e-6 and abs(dp1) < 1e-6:
                    break
            else:
                lam *= 5.0

        return az, alt, _sse(az, alt)

    base = _drift_to_az_alt(
        float(vx),
        float(vy),
        phi_deg=float(phi_deg),
        omega_arcsec_s=float(omega_arcsec_s),
        scale_arcsec_per_px=float(scale_arcsec_per_px),
        dedup_tol_deg=float(dedup_tol_deg),
        sort_by_forward_err=False,
    )
    if not base:
        log_error(None, "GoTo: drift_to_az_alt_refracted no base solutions")
        return []

    seeds_true: List[Tuple[float, float]] = []
    for az_seed, alt_app_seed in base:
        alt_true_seed = _unrefract_app_to_true(alt_app_seed, P_hPa=float(P_hPa), T_C=float(T_C))
        seeds_true.append((az_seed, alt_true_seed))

    refined: List[Tuple[float, float, float]] = []
    for az0, alt0 in seeds_true:
        refined.append(_gauss_newton_lm(az0, alt0))

    sols: List[Tuple[float, float]] = []
    for az_deg, alt_true_deg, _ in refined:
        alt_app_deg = alt_true_deg + _R_true_to_app_deg(
            alt_true_deg,
            P_hPa=float(P_hPa),
            T_C=float(T_C),
        )
        if np.isfinite(alt_app_deg) and -0.5 <= alt_app_deg <= 90.0:
            sols.append((_wrap360(az_deg), float(alt_app_deg)))

    if not sols:
        log_error(None, "GoTo: drift_to_az_alt_refracted no solutions after refinement")
        return []

    uniq: List[Tuple[float, float]] = []
    for az_deg, alt_deg in sols:
        is_new = True
        for az2, alt2 in uniq:
            if _circ_dist_deg(az_deg, az2) <= float(dedup_tol_deg) and abs(alt_deg - alt2) <= float(dedup_tol_deg):
                is_new = False
                break
        if is_new:
            uniq.append((az_deg, alt_deg))

    if sort_by_forward_err and len(uniq) > 1:
        uniq.sort(key=lambda p: (_sse(p[0], p[1]), p[0], p[1]))
    return uniq


def _apply_roll_to_drift(v: np.ndarray, roll_deg: float) -> np.ndarray:
    """
    Rotate drift vector by -roll so +x aligns with az-axis.

    For drift-axis alignment we only need orientation modulo 180 deg, so we
    first map roll to [-90, +90). This avoids accidental full-vector flips when
    roll is reported in the opposite 180-deg branch.
    """
    if not np.isfinite(roll_deg):
        return v
    roll_axis_deg = _roll_axis_equivalent_deg(float(roll_deg))
    r = math.radians(float(roll_axis_deg))
    if abs(r) < 1e-12:
        return v
    c = math.cos(r)
    s = math.sin(r)
    vx = float(v[0])
    vy = float(v[1])
    return np.array([c * vx + s * vy, -s * vx + c * vy], dtype=np.float64)


def _predict_horizontal_tangent_rate(
    az_deg: float,
    alt_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Time,
    dt_s: float = 1.0,
) -> np.ndarray:
    """Sidereal motion in the local orthonormal (AZ tangent, ALT) basis.

    The first component is ``cos(alt) * dAz/dt``.  Using bare ``dAz/dt``
    against an image displacement is geometrically wrong away from the
    horizon and was able to drive the JCal inverse to an altitude limit.
    """
    dt = max(0.05, float(dt_s))
    az0 = _wrap_deg_360(float(az_deg))
    alt0 = float(alt_deg)
    frame0 = AltAz(
        az=az0 * u.deg,
        alt=alt0 * u.deg,
        obstime=obstime,
        location=observer.location(),
    )
    fixed_coord = SkyCoord(frame0)
    frame1 = fixed_coord.transform_to(
        AltAz(obstime=obstime + dt * u.s, location=observer.location())
    )
    daz_rad_s = math.radians(
        _wrap_deg_180(float(frame1.az.deg) - az0)
    ) / dt
    dalt_rad_s = math.radians(float(frame1.alt.deg) - alt0) / dt
    return np.array(
        [math.cos(math.radians(alt0)) * daz_rad_s, dalt_rad_s],
        dtype=np.float64,
    )


def _solve_jcal_pointing(
    drift_pix_s: np.ndarray,
    J_pix_per_step: np.ndarray,
    *,
    plate_scale_rad_per_px: float,
    observer: ObserverConfig,
    obstime: Time,
    alt_min_deg: float,
    alt_max_deg: float,
    axis_sign_az: int = 1,
    axis_sign_alt: int = 1,
    seeds: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Infer local pointing from drift and the two measured motor axes.

    ``J_pix_per_step`` describes image motion caused by positive motor steps.
    Its column magnitudes are irrelevant here, but both (slightly
    non-orthogonal) measured directions matter.  Solving the 2x2 basis avoids
    double-counting one component as the previous pair of dot products did.
    """
    drift = np.asarray(drift_pix_s, dtype=np.float64).reshape(2,)
    J_pix = np.asarray(J_pix_per_step, dtype=np.float64).reshape(2, 2)
    scale = float(plate_scale_rad_per_px)
    if not np.all(np.isfinite(drift)) or not np.all(np.isfinite(J_pix)):
        return {"ok": False, "status": "NONFINITE_INPUT"}
    if not np.isfinite(scale) or scale <= 0.0:
        return {"ok": False, "status": "INVALID_PLATE_SCALE"}

    norms = np.linalg.norm(J_pix, axis=0)
    if np.any(norms <= 1e-12):
        return {"ok": False, "status": "DEGENERATE_AXIS"}
    axes = J_pix / norms[None, :]
    cond = float(np.linalg.cond(axes))
    if not np.isfinite(cond) or cond > 20.0:
        return {"ok": False, "status": "ILL_CONDITIONED_AXES", "condition": cond}

    coeff_pix_s = np.linalg.solve(axes, drift)
    signs = np.array(
        [1.0 if int(axis_sign_az) >= 0 else -1.0,
         1.0 if int(axis_sign_alt) >= 0 else -1.0],
        dtype=np.float64,
    )
    # Positive mount motion makes a fixed star move in the opposite image
    # direction.  Convert the image-basis coefficients back to natural local
    # tangent components.
    observed_tangent = -signs * coeff_pix_s * scale
    observed_norm = float(np.linalg.norm(observed_tangent))
    if not np.isfinite(observed_norm) or observed_norm <= 1e-12:
        return {"ok": False, "status": "ZERO_DRIFT"}

    seed_values = list(seeds or ())
    if not seed_values:
        seed_values = [
            (az, alt)
            for alt in (20.0, 35.0, 50.0, 65.0, 78.0)
            for az in np.arange(22.5, 360.0, 45.0)
        ]

    candidates: List[Dict[str, float]] = []
    lo = float(alt_min_deg)
    hi = float(alt_max_deg)
    for seed_az, seed_alt in seed_values:
        az = _wrap_deg_360(float(seed_az))
        alt = float(np.clip(float(seed_alt), lo, hi))
        hit_boundary = False
        for _ in range(16):
            pred = _predict_horizontal_tangent_rate(
                az, alt, observer=observer, obstime=obstime
            )
            resid = pred - observed_tangent
            if float(np.linalg.norm(resid)) <= max(2e-9, observed_norm * 1e-4):
                break
            delta_deg = 0.05
            pred_az = _predict_horizontal_tangent_rate(
                az + delta_deg, alt, observer=observer, obstime=obstime
            )
            pred_alt = _predict_horizontal_tangent_rate(
                az, min(hi, alt + delta_deg), observer=observer, obstime=obstime
            )
            jac = np.column_stack(
                [(pred_az - pred) / delta_deg, (pred_alt - pred) / delta_deg]
            )
            if np.linalg.matrix_rank(jac) < 2:
                break
            step = np.linalg.lstsq(jac, -resid, rcond=None)[0]
            step = np.clip(step, -8.0, 8.0)
            az = _wrap_deg_360(az + float(step[0]))
            unclipped_alt = alt + float(step[1])
            if unclipped_alt < lo or unclipped_alt > hi:
                hit_boundary = True
            alt = float(np.clip(unclipped_alt, lo, hi))

        pred = _predict_horizontal_tangent_rate(
            az, alt, observer=observer, obstime=obstime
        )
        resid_norm = float(np.linalg.norm(pred - observed_tangent))
        if hit_boundary and (abs(alt - lo) < 1e-6 or abs(alt - hi) < 1e-6):
            continue
        if any(
            abs(_wrap_deg_180(az - item["az_deg"])) < 0.02
            and abs(alt - item["alt_deg"]) < 0.02
            for item in candidates
        ):
            continue
        candidates.append(
            {"az_deg": float(az), "alt_deg": float(alt), "residual_rad_s": resid_norm}
        )

    candidates.sort(key=lambda item: float(item["residual_rad_s"]))
    if not candidates:
        return {
            "ok": False,
            "status": "NO_INTERIOR_SOLUTION",
            "condition": cond,
            "coeff_pix_s": coeff_pix_s,
            "observed_tangent_rad_s": observed_tangent,
        }
    best = candidates[0]
    max_residual = max(1.0e-6, 0.15 * observed_norm)
    if float(best["residual_rad_s"]) > max_residual:
        return {
            "ok": False,
            "status": "HIGH_RESIDUAL",
            "condition": cond,
            "coeff_pix_s": coeff_pix_s,
            "observed_tangent_rad_s": observed_tangent,
            "residual_rad_s": float(best["residual_rad_s"]),
            "max_residual_rad_s": float(max_residual),
            "candidates": candidates,
        }

    coordinate_rate = _predict_horizontal_tangent_rate(
        float(best["az_deg"]),
        float(best["alt_deg"]),
        observer=observer,
        obstime=obstime,
    ).copy()
    cos_alt = math.cos(math.radians(float(best["alt_deg"])))
    if abs(cos_alt) <= 1e-6:
        return {"ok": False, "status": "ZENITH_DEGENERACY"}
    coordinate_rate[0] /= cos_alt
    return {
        "ok": True,
        "status": "OK",
        "az_deg": float(best["az_deg"]),
        "alt_deg": float(best["alt_deg"]),
        "condition": cond,
        "coeff_pix_s": coeff_pix_s,
        "observed_tangent_rad_s": observed_tangent,
        "coordinate_rate_rad_s": coordinate_rate,
        "residual_rad_s": float(best["residual_rad_s"]),
        "candidates": candidates,
    }


def _roll_deg_from_drift_delta(dv: np.ndarray, slew_rate_steps_s: float) -> float:
    """
    Convert drift delta from roll estimation into camera roll (deg).

    roll is defined as the orientation of the +AZ axis in image coords (+x right, +y up).
    If the induced AZ slew was negative, dv points to -AZ and must be rotated 180 deg.
    """
    d = np.asarray(dv, dtype=np.float64).reshape(2,)
    if not np.all(np.isfinite(d)):
        raise ValueError("drift delta must be finite")

    roll_deg = float(math.degrees(math.atan2(float(d[1]), float(d[0]))))
    if np.isfinite(slew_rate_steps_s) and float(slew_rate_steps_s) < 0.0:
        roll_deg = _wrap_deg_180(roll_deg + 180.0)
    else:
        roll_deg = _wrap_deg_180(roll_deg)
    return float(roll_deg)


def _roll_axis_equivalent_deg(roll_deg: float) -> float:
    """
    Canonical roll branch for axis alignment only (direction-agnostic).

    Returns angle in [-90, +90), equivalent to roll modulo 180 deg.
    """
    r = _wrap_deg_180(float(roll_deg))
    if r >= 90.0:
        r -= 180.0
    return float(r)


def roll_axis_distance_deg(a_deg: float, b_deg: float) -> float:
    """Smallest angular separation between unoriented camera axes."""
    return abs(_roll_axis_equivalent_deg(float(a_deg) - float(b_deg)))


def _roll_equivalent_near_reference_deg(roll_deg: float, ref_deg: float) -> float:
    """
    Pick the equivalent roll branch (r or r+180) closest to a reference angle.

    This prevents sporadic 180-deg branch flips from propagating to consumers
    that expect a temporally stable camera roll estimate.
    """
    r0 = _wrap_deg_180(float(roll_deg))
    if not np.isfinite(ref_deg):
        return float(r0)
    r1 = _wrap_deg_180(float(r0) + 180.0)
    d0 = abs(_wrap_deg_180(float(r0) - float(ref_deg)))
    d1 = abs(_wrap_deg_180(float(r1) - float(ref_deg)))
    if d1 < d0:
        return float(r1)
    return float(r0)


def _now_time() -> Time:
    # astropy Time uses UTC by default
    return Time.now()


_BRIGHT_START_STARS = BRIGHT_STAR_SUPPLEMENT


def pick_bright_start_star(
    observer: ObserverConfig,
    obstime: Optional[Time],
    *,
    min_alt_deg: float = 15.0,
) -> Optional[Dict[str, float | str]]:
    """Pick a bright, currently visible star to use for the first sync."""
    if obstime is None:
        obstime = _now_time()

    candidates: List[Dict[str, float | str]] = []
    fallback: List[Dict[str, float | str]] = []

    for star in _BRIGHT_START_STARS:
        coord = SkyCoord(
            ra=float(star["ra_deg"]) * u.deg,
            dec=float(star["dec_deg"]) * u.deg,
            frame="icrs",
        )
        altaz = icrs_to_altaz_deg(coord, observer=observer, obstime=obstime)
        az_deg = float(altaz[0])
        alt_deg = float(altaz[1])
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

    # Direct mechanical reduction. 45.0 means motor:axis = 45:1.
    gear_reduction_az: float | None = 45.0
    gear_reduction_alt: float | None = 45.0

    # A 45-lobe cycloidal reducer repeats its first-order transmission error
    # once per motor revolution: 45 cycles per full output revolution.
    transmission_lobes_az: int = 45
    transmission_lobes_alt: int = 45

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
        lobes = int(
            self.transmission_lobes_az
            if axis == Axis.AZ
            else self.transmission_lobes_alt
        )
        if lobes <= 0:
            raise ValueError("invalid transmission_lobes")
        return float(self.steps_per_axis_rev(axis) / float(lobes))


@dataclass
class GoToModel:
    """Internal pointing model.

    Coordinates:
      - Steps are *commanded* microsteps from firmware MOVE (per axis).
      - Angles are mount AltAz in degrees.

    Mapping:
      d_altaz = J_deg_per_step @ d_steps
      where d_altaz = [d_az_deg, d_alt_deg]^T and d_steps = [d_az, d_alt]^T.

    Global correction:
      AltAz_world ~= R_mount_to_world * AltAz_mount_model (on unit vectors).
    """

    kin: MountKinematics = field(default_factory=MountKinematics)

    # J (2x2): deg per step
    J_deg_per_step: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))

    # Reference (sync)
    synced: bool = False
    ref_steps: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    # Reference in mount-model frame (before global spherical correction).
    ref_az_alt_deg: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))  # [az, alt]

    # Global all-sky correction: world ~= R_mount_to_world * mount_model.
    R_mount_to_world: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    # Hard limit for mount base tilts (NS/OE components of the global rotation).
    max_tilt_ns_oe_deg: float = 2.0

    # The fitted step matrix is only a small correction around the known 45:1
    # mechanics. In particular, a fit must never reverse either configured
    # axis. Coupling is limited because large sky rotations belong in
    # R_mount_to_world, not in the motor scale matrix.
    max_step_scale_deviation_frac: float = 0.10
    max_step_axis_coupling_frac: float = 0.05

    # A fitted axis needs enough physical travel to distinguish scale from
    # plate-solve noise/backlash.  At the default 45:1 and 1/64 microstepping,
    # 0.25 deg corresponds to 400 microsteps.
    min_fit_axis_span_deg: float = 0.25

    # Even a mechanically plausible matrix is not accepted when the samples
    # disagree by more than this after the global rotation fit.
    max_model_fit_rms_arcsec: float = 120.0

    # Robust-fit thresholds. Plate-solving residuals below the floor are kept;
    # above it, a 3-MAD rule rejects samples more aggressively than the old
    # 4.5-MAD policy.
    fit_outlier_sigma: float = 3.0
    fit_outlier_floor_arcsec: float = 10.0

    # First-order cycloidal transmission-error model. The nominal ratio stays
    # 45:1; these bounded periodic offsets account for local acceleration and
    # deceleration of the output within each lobe cycle.
    # Eight points leave useful residual degrees of freedom after fitting the
    # two global step columns plus sine/cosine. Six points proved too easy to
    # overfit when plate solutions carried unrelated offsets.
    min_periodic_model_samples: int = 8
    min_periodic_phase_span_frac: float = 0.25
    max_periodic_error_deg: float = 0.25
    periodic_coeff_deg: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    periodic_model_samples: int = 0

    # Reject a new manual plate-solve before it enters the fit when its jump
    # is incompatible with the microsteps emitted since the previous sample.
    # The allowance is intentionally conservative to tolerate backlash and a
    # still-imperfect mechanical model while blocking multi-degree false
    # solves after a few hundred microsteps.
    manual_sample_motion_factor: float = 3.0
    manual_sample_motion_margin_deg: float = 0.05
    manual_sample_roll_tolerance_deg: float = 12.0

    # Current estimated step counter (relative, but we store absolute in same units as ref_steps)
    steps_est: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    # Last successful plate-solve (used as the best estimate of mount AZ/ALT)
    last_solve_az_alt_deg: Optional[np.ndarray] = None
    last_solve_steps_est: Optional[np.ndarray] = None
    last_solve_time: float = 0.0

    # Calibration samples (for updating J)
    _calib_steps: List[np.ndarray] = field(default_factory=list, repr=False)
    _calib_daltaz: List[np.ndarray] = field(default_factory=list, repr=False)

    # Manual calibration samples (absolute measurements)
    _manual_steps_abs: List[np.ndarray] = field(default_factory=list, repr=False)
    _manual_az_alt_abs: List[np.ndarray] = field(default_factory=list, repr=False)
    _manual_roll_deg_abs: List[float] = field(default_factory=list, repr=False)
    # Path of the diagnostic session that produced each live sample. Empty
    # strings identify restored/legacy samples without raw provenance.
    _manual_source_abs: List[str] = field(default_factory=list, repr=False)
    _manual_fit_inlier_mask: Optional[np.ndarray] = field(default=None, repr=False)

    # History of commanded steps (AZ, ALT) during the session
    steps_history: List[np.ndarray] = field(default_factory=list, repr=False)

    # Last physical load direction for each gear train. This is intentionally
    # separate from steps_est: backlash take-up pulses move the motor but not
    # the modeled optical axis.
    last_move_direction_az: int = 0
    last_move_direction_alt: int = 0
    backlash_steps_az: int = 0
    backlash_steps_alt: int = 10

    # Model-fit report fields (published to state).
    J00_err: float = 0.0
    J01_err: float = 0.0
    J10_err: float = 0.0
    J11_err: float = 0.0
    model_non_orthogonality_deg: float = 0.0
    model_non_orthogonality_err_deg: float = 0.0
    model_roll_deg: float = 0.0
    model_roll_err_deg: float = 0.0
    model_roll_samples: int = 0
    model_pitch_deg: float = 0.0
    model_pitch_err_deg: float = 0.0
    model_yaw_deg: float = 0.0
    model_yaw_err_deg: float = 0.0
    model_fit_samples: int = 0
    model_fit_rms_az_deg: float = 0.0
    model_fit_rms_alt_deg: float = 0.0
    model_fit_rms_arcsec: float = 0.0
    last_fit_reason: str = "NOT_FITTED"
    # Cycles of transmission error spanned by the samples of the last fit.
    last_fit_phase_coverage: Dict[str, float] = field(default_factory=dict)

    def mechanical_J(self) -> np.ndarray:
        """Return the signed diagonal matrix implied by the mount mechanics."""
        return np.array(
            [
                [float(self.kin.deg_per_step(Axis.AZ)), 0.0],
                [0.0, float(self.kin.deg_per_step(Axis.ALT))],
            ],
            dtype=np.float64,
        )

    def manual_phase_coverage(self, sample_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fraction of a transmission-error cycle spanned by the samples, per axis.

        J is the *mean* scale; the cycloidal transmission error is what makes
        the locally measured scale wander (up to max_periodic_error*2pi/period,
        about 20% with the defaults). So a fit built from moves much shorter
        than one lobe measures the local slope of that error, not the mean
        scale, and can land far outside the mechanical envelope even when the
        gearing is perfect. Reporting the coverage turns an opaque
        "outside mechanical limits" rejection into an actionable one.
        """
        out = {"az": 0.0, "alt": 0.0, "min": 0.0}
        if not self._manual_steps_abs:
            return out
        S = np.stack(self._manual_steps_abs, axis=0).astype(np.float64)
        if sample_mask is not None:
            m = np.asarray(sample_mask, dtype=bool).reshape(-1)
            if m.size == S.shape[0] and m.any():
                S = S[m, :]
        if S.shape[0] < 2:
            return out
        for axis_idx, (axis, key) in enumerate(((Axis.AZ, "az"), (Axis.ALT, "alt"))):
            period = float(self.kin.transmission_error_period_steps(axis))
            if not np.isfinite(period) or period <= 0.0:
                continue
            span_steps = float(np.ptp(S[:, axis_idx]))
            out[key] = float(min(span_steps / period, 10.0))
        out["min"] = float(min(out["az"], out["alt"]))
        return out

    def _coupling_limit_factors(self) -> Tuple[float, float]:
        """Geometric correction of the cross-coupling budget with altitude.

        The coupling limit is meant to bound the physical non-orthogonality of
        the axes, but J is expressed in degrees of *azimuth* per step, and near
        the zenith one degree of azimuth is a tiny angle on the sky (cos alt).
        A fixed limit on the raw J entry therefore becomes absurdly strict up
        high: a real 1.6 deg non-orthogonality measured at alt 78 deg reads as
        0.10 in J terms and gets rejected against a 0.05 budget, even though on
        the sky it is only 0.029.

        Returns (factor_for_J01, factor_for_J10):
          - J[0,1] is azimuth per alt-step  -> its sky angle is J01*cos(alt),
            so the budget on J01 scales as 1/cos(alt).
          - J[1,0] is altitude per az-step  -> an az step moves cos(alt) on the
            sky, so the budget on J10 scales as cos(alt).
        """
        alt = float(np.clip(float(self.ref_az_alt_deg[1]), -89.0, 89.0))
        cos_alt = float(np.cos(np.deg2rad(alt)))
        if (not np.isfinite(cos_alt)) or cos_alt <= 1e-6:
            cos_alt = 1e-6
        # Bounded so a near-zenith reference cannot open the budget without end.
        widen = float(np.clip(1.0 / cos_alt, 1.0, 6.0))
        narrow = float(np.clip(cos_alt, 1.0 / 6.0, 1.0))
        return widen, narrow

    def constrain_J_to_mechanics(self, candidate: np.ndarray) -> np.ndarray:
        """Project a fitted J into a safe neighborhood of the mechanical J."""
        mechanical = self.mechanical_J()
        J = np.asarray(candidate, dtype=np.float64).copy()
        if J.shape != (2, 2) or not np.all(np.isfinite(J)):
            return mechanical

        scale_frac = _clamp(float(self.max_step_scale_deviation_frac), 0.0, 0.95)
        coupling_frac = _clamp(float(self.max_step_axis_coupling_frac), 0.0, 0.50)
        widen, narrow = self._coupling_limit_factors()
        # column 0 drives J[1,0] (alt per az-step); column 1 drives J[0,1].
        cross_gain = (narrow, widen)
        for col in range(2):
            diag = col
            cross = 1 - col
            mech_diag = float(mechanical[diag, col])
            mech_abs = abs(mech_diag)
            mech_sign = +1.0 if mech_diag >= 0.0 else -1.0
            fitted_along_sign = float(J[diag, col]) * mech_sign
            fitted_along_sign = _clamp(
                fitted_along_sign,
                mech_abs * (1.0 - scale_frac),
                mech_abs * (1.0 + scale_frac),
            )
            J[diag, col] = mech_sign * fitted_along_sign
            cross_limit = coupling_frac * mech_abs * float(cross_gain[col])
            J[cross, col] = _clamp(float(J[cross, col]), -cross_limit, cross_limit)

        return J

    def is_J_within_mechanical_limits(self, candidate: np.ndarray) -> bool:
        """Return whether ``candidate`` is already inside the 45:1 envelope.

        This is intentionally separate from ``constrain_J_to_mechanics``:
        constraining is a final safety net for prediction, while a newly fitted
        or restored model outside the envelope must be rejected rather than
        silently accepted at a clipped boundary.
        """
        J = np.asarray(candidate, dtype=np.float64)
        if J.shape != (2, 2) or not np.all(np.isfinite(J)):
            return False
        constrained = self.constrain_J_to_mechanics(J)
        mechanical_scale = float(np.max(np.abs(self.mechanical_J())))
        atol = max(1e-15, mechanical_scale * 1e-9)
        return bool(np.allclose(J, constrained, rtol=0.0, atol=atol))

    def safe_J_for_prediction(self) -> np.ndarray:
        """Return a finite, mechanically bounded matrix for non-moving math."""
        return self.constrain_J_to_mechanics(self.J_deg_per_step)

    def safe_periodic_coeff_for_prediction(
        self,
        coeff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        candidate = np.asarray(
            self.periodic_coeff_deg if coeff is None else coeff,
            dtype=np.float64,
        )
        if candidate.shape != (2, 2) or not np.all(np.isfinite(candidate)):
            return np.zeros((2, 2), dtype=np.float64)
        out = candidate.copy()
        amplitude_limit = max(0.0, float(self.max_periodic_error_deg))
        for axis_idx in range(2):
            amplitude = float(np.linalg.norm(out[axis_idx, :]))
            if amplitude_limit <= 0.0:
                out[axis_idx, :] = 0.0
            elif amplitude > amplitude_limit and amplitude > 0.0:
                out[axis_idx, :] *= amplitude_limit / amplitude
        return out

    def _periodic_offset_deg(
        self,
        steps_abs: Sequence[float],
        *,
        coeff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        steps = _as_array2(steps_abs)
        c = self.safe_periodic_coeff_for_prediction(coeff)
        out = np.zeros(2, dtype=np.float64)
        for axis_idx, axis in enumerate((Axis.AZ, Axis.ALT)):
            period = float(self.kin.transmission_error_period_steps(axis))
            phase = 2.0 * math.pi * float(steps[axis_idx]) / period
            out[axis_idx] = (
                float(c[axis_idx, 0]) * math.sin(phase)
                + float(c[axis_idx, 1]) * math.cos(phase)
            )
        return out

    def mount_delta_for_steps(
        self,
        steps_from: Sequence[float],
        steps_to: Sequence[float],
        *,
        J_model: Optional[np.ndarray] = None,
        periodic_coeff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        start = _as_array2(steps_from)
        end = _as_array2(steps_to)
        J = self.safe_J_for_prediction() if J_model is None else self.constrain_J_to_mechanics(J_model)
        linear = J @ (end - start)
        periodic = self._periodic_offset_deg(end, coeff=periodic_coeff) - self._periodic_offset_deg(start, coeff=periodic_coeff)
        return np.asarray(linear + periodic, dtype=np.float64)

    def solve_step_delta_for_mount_delta(
        self,
        mount_delta_deg: Sequence[float],
        *,
        steps_from: Optional[Sequence[float]] = None,
        max_iters: int = 6,
    ) -> np.ndarray:
        """Invert the bounded nonlinear transmission model with Newton steps."""
        target = _as_array2(mount_delta_deg)
        start = self.steps_est.copy() if steps_from is None else _as_array2(steps_from)
        J = self.safe_J_for_prediction()
        try:
            delta = np.linalg.solve(J, target)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, target, rcond=None)
        delta = np.asarray(delta, dtype=np.float64).reshape(2,)
        if self.periodic_model_samples < int(self.min_periodic_model_samples):
            return delta

        for _ in range(max(1, int(max_iters))):
            predicted = self.mount_delta_for_steps(start, start + delta)
            residual = target - predicted
            if float(np.linalg.norm(residual)) <= 1e-7:
                break
            jac = np.zeros((2, 2), dtype=np.float64)
            eps = 8.0
            for axis_idx in range(2):
                perturb = np.zeros(2, dtype=np.float64)
                perturb[axis_idx] = eps
                plus = self.mount_delta_for_steps(start, start + delta + perturb)
                minus = self.mount_delta_for_steps(start, start + delta - perturb)
                jac[:, axis_idx] = (plus - minus) / (2.0 * eps)
            try:
                correction = np.linalg.solve(jac, residual)
            except np.linalg.LinAlgError:
                correction, *_ = np.linalg.lstsq(jac, residual, rcond=None)
            if not np.all(np.isfinite(correction)):
                break
            delta += np.asarray(correction, dtype=np.float64)
        return delta

    def init_from_mechanics(self) -> None:
        """Initialize J from the mechanical model (diagonal, no coupling)."""
        self.J_deg_per_step = self.mechanical_J()
        self.J00_err = 0.0
        self.J01_err = 0.0
        self.J10_err = 0.0
        self.J11_err = 0.0
        self.model_non_orthogonality_deg = _non_orthogonality_deg_from_J(self.J_deg_per_step)
        self.model_non_orthogonality_err_deg = 0.0
        self.model_pitch_deg = 0.0
        self.model_pitch_err_deg = 0.0
        self.model_yaw_deg = 0.0
        self.model_yaw_err_deg = 0.0
        self.model_fit_samples = 0
        self.model_fit_rms_az_deg = 0.0
        self.model_fit_rms_alt_deg = 0.0
        self.model_fit_rms_arcsec = 0.0
        self.periodic_coeff_deg = np.zeros((2, 2), dtype=np.float64)
        self.periodic_model_samples = 0

    def set_microsteps(self, az_div: int, alt_div: int) -> None:
        """Validate the hardware-wired 1/64 setting without rebasing state."""
        if int(az_div) != 64 or int(alt_div) != 64:
            raise ValueError(
                f"microstepping is hardware-fixed at 1/64; "
                f"requested AZ=1/{int(az_div)} ALT=1/{int(alt_div)}"
            )
        self.kin.microsteps_az = 64
        self.kin.microsteps_alt = 64

    def note_manual_move(self, axis: Axis, direction: int, steps: int) -> None:
        """Update step counter when the app executes a MOVE."""
        s = float(abs(int(steps)))
        s *= +1.0 if int(direction) >= 0 else -1.0
        if axis == Axis.AZ:
            self.steps_est[0] += s
            self.last_move_direction_az = +1 if int(direction) >= 0 else -1
        else:
            self.steps_est[1] += s
            self.last_move_direction_alt = +1 if int(direction) >= 0 else -1
        d_az = s if axis == Axis.AZ else 0.0
        d_alt = s if axis == Axis.ALT else 0.0
        self.steps_history.append(np.array([d_az, d_alt], dtype=np.float64))

    def note_emitted_rate_steps(self, dsteps: Sequence[float]) -> None:
        """Account for integer steps actually emitted by RATE emulation."""
        moved = _as_array2(dsteps)
        if not np.all(np.isfinite(moved)):
            return
        self.steps_est += moved
        if abs(float(moved[0])) >= 1.0:
            self.last_move_direction_az = +1 if float(moved[0]) > 0.0 else -1
        if abs(float(moved[1])) >= 1.0:
            self.last_move_direction_alt = +1 if float(moved[1]) > 0.0 else -1

    def last_move_direction(self, axis: Axis) -> int:
        value = self.last_move_direction_az if axis == Axis.AZ else self.last_move_direction_alt
        if int(value) in (-1, +1):
            return int(value)
        return 0

    def set_last_move_direction(self, axis: Axis, direction: int) -> None:
        value = +1 if int(direction) >= 0 else -1
        if axis == Axis.AZ:
            self.last_move_direction_az = value
        else:
            self.last_move_direction_alt = value

    def _csv_rotation_values(self) -> Dict[str, float]:
        R = self._rotation_mount_to_world()
        return {
            "R00": float(R[0, 0]),
            "R01": float(R[0, 1]),
            "R02": float(R[0, 2]),
            "R10": float(R[1, 0]),
            "R11": float(R[1, 1]),
            "R12": float(R[1, 2]),
            "R20": float(R[2, 0]),
            "R21": float(R[2, 1]),
            "R22": float(R[2, 2]),
        }

    def _log_manual_sample_csv(self, *, sample_idx: int, az_alt_world: np.ndarray, roll_sample: Optional[float]) -> None:
        ts_unix = float(self.last_solve_time if self.last_solve_time > 0.0 else time.time())
        az_alt = _as_array2(az_alt_world)
        roll_out = float(roll_sample) if roll_sample is not None and np.isfinite(float(roll_sample)) else ""
        row: Dict[str, Any] = {
            "ts_unix": ts_unix,
            "ts_utc": _utc_str(ts_unix),
            "sample_idx": int(sample_idx),
            "steps_az": float(self.steps_est[0]),
            "steps_alt": float(self.steps_est[1]),
            "az_deg": float(az_alt[0]),
            "alt_deg": float(az_alt[1]),
            "roll_deg": roll_out,
            "synced": int(bool(self.synced)),
            "ref_steps_az": float(self.ref_steps[0]),
            "ref_steps_alt": float(self.ref_steps[1]),
            "ref_az_mount_deg": float(self.ref_az_alt_deg[0]),
            "ref_alt_mount_deg": float(self.ref_az_alt_deg[1]),
            "last_direction_az": int(self.last_move_direction_az),
            "last_direction_alt": int(self.last_move_direction_alt),
            "backlash_steps_az": int(max(0, self.backlash_steps_az)),
            "backlash_steps_alt": int(max(0, self.backlash_steps_alt)),
        }
        row.update(self._csv_rotation_values())
        _append_csv_log_row("goto_manual_samples.csv", _GOTO_MANUAL_SAMPLE_CSV_FIELDS, row)

    def _log_fit_csv(
        self,
        *,
        fit_kind: str,
        ok: bool,
        reason: str,
        min_samples: int,
        ridge: float,
        total_samples: int,
        used_samples: int,
    ) -> None:
        ts_unix = float(time.time())
        rep = self.model_fit_report()
        row: Dict[str, Any] = {
            "ts_unix": ts_unix,
            "ts_utc": _utc_str(ts_unix),
            "fit_kind": str(fit_kind),
            "ok": int(bool(ok)),
            "reason": str(reason),
            "min_samples": int(min_samples),
            "ridge": float(ridge),
            "total_samples": int(total_samples),
            "used_samples": int(used_samples),
            "outliers": int(max(0, int(total_samples) - int(used_samples))),
            "J00": float(self.J_deg_per_step[0, 0]),
            "J01": float(self.J_deg_per_step[0, 1]),
            "J10": float(self.J_deg_per_step[1, 0]),
            "J11": float(self.J_deg_per_step[1, 1]),
            "J00_err": float(rep["J00_err"]),
            "J01_err": float(rep["J01_err"]),
            "J10_err": float(rep["J10_err"]),
            "J11_err": float(rep["J11_err"]),
            "model_fit_rms_az_deg": float(rep["model_fit_rms_az_deg"]),
            "model_fit_rms_alt_deg": float(rep["model_fit_rms_alt_deg"]),
            "model_fit_rms_arcsec": float(rep["model_fit_rms_arcsec"]),
            "model_non_orthogonality_deg": float(rep["model_non_orthogonality_deg"]),
            "model_non_orthogonality_err_deg": float(rep["model_non_orthogonality_err_deg"]),
            "model_roll_deg": float(rep["model_roll_deg"]),
            "model_roll_err_deg": float(rep["model_roll_err_deg"]),
            "model_roll_samples": int(rep["model_roll_samples"]),
            "model_pitch_deg": float(rep["model_pitch_deg"]),
            "model_pitch_err_deg": float(rep["model_pitch_err_deg"]),
            "model_yaw_deg": float(rep["model_yaw_deg"]),
            "model_yaw_err_deg": float(rep["model_yaw_err_deg"]),
            "periodic_az_sin_deg": float(self.periodic_coeff_deg[0, 0]),
            "periodic_az_cos_deg": float(self.periodic_coeff_deg[0, 1]),
            "periodic_alt_sin_deg": float(self.periodic_coeff_deg[1, 0]),
            "periodic_alt_cos_deg": float(self.periodic_coeff_deg[1, 1]),
            "periodic_model_samples": int(self.periodic_model_samples),
            "periodic_error_az_deg": float(np.linalg.norm(self.periodic_coeff_deg[0, :])),
            "periodic_error_alt_deg": float(np.linalg.norm(self.periodic_coeff_deg[1, :])),
            "last_direction_az": int(self.last_move_direction_az),
            "last_direction_alt": int(self.last_move_direction_alt),
            "backlash_steps_az": int(self.backlash_steps_az),
            "backlash_steps_alt": int(self.backlash_steps_alt),
        }
        row.update(self._csv_rotation_values())
        _append_csv_log_row("goto_model_fit_log.csv", _GOTO_MODEL_FIT_CSV_FIELDS, row)

    def restore_from_latest_logs(self, *, log_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Restore GoTo model state from the latest persisted CSV logs.

        Manual samples are restored from the last contiguous sample_idx block
        in ``goto_manual_samples.csv`` (last session-like segment).
        Model/J state is restored from the latest valid row in
        ``goto_model_fit_log.csv``.
        """

        def _as_float(value: Any, default: float = float("nan")) -> float:
            try:
                s = str(value).strip()
                if s == "":
                    return float(default)
                return float(s)
            except Exception:
                return float(default)

        def _as_int(value: Any, default: int = 0) -> int:
            v = _as_float(value, float(default))
            if not np.isfinite(v):
                return int(default)
            return int(round(float(v)))

        def _as_bool(value: Any, default: bool = False) -> bool:
            s = str(value).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off", ""):
                return False
            try:
                return bool(int(float(s)))
            except Exception:
                return bool(default)

        def _read_rows(path: str) -> List[Dict[str, str]]:
            if not os.path.exists(path):
                return []
            rows: List[Dict[str, str]] = []
            with _GOTO_CSV_LOG_LOCK:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row:
                            rows.append(dict(row))
            return rows

        def _rotation_from_row(row: Dict[str, Any]) -> Optional[np.ndarray]:
            vals = [
                _as_float(row.get("R00", float("nan"))),
                _as_float(row.get("R01", float("nan"))),
                _as_float(row.get("R02", float("nan"))),
                _as_float(row.get("R10", float("nan"))),
                _as_float(row.get("R11", float("nan"))),
                _as_float(row.get("R12", float("nan"))),
                _as_float(row.get("R20", float("nan"))),
                _as_float(row.get("R21", float("nan"))),
                _as_float(row.get("R22", float("nan"))),
            ]
            if not all(np.isfinite(v) for v in vals):
                return None
            R = np.asarray(vals, dtype=np.float64).reshape(3, 3)
            return _coerce_rotation_matrix(R)

        base_dir = str(log_dir).strip() if log_dir is not None else _goto_logs_dir()
        manual_path = os.path.join(base_dir, "goto_manual_samples.csv")
        fit_path = os.path.join(base_dir, "goto_model_fit_log.csv")

        manual_rows = _read_rows(manual_path)
        fit_rows = _read_rows(fit_path)
        if not manual_rows and not fit_rows:
            return {
                "ok": False,
                "status": "NO_LOGS",
                "manual_samples": 0,
                "synced": bool(self.synced),
                "camera_roll_deg": float("nan"),
            }

        manual_entries: List[Dict[str, Any]] = []
        for row in manual_rows:
            sample_idx = _as_int(row.get("sample_idx", -1), -1)
            steps_az = _as_float(row.get("steps_az", float("nan")))
            steps_alt = _as_float(row.get("steps_alt", float("nan")))
            az_deg = _as_float(row.get("az_deg", float("nan")))
            alt_deg = _as_float(row.get("alt_deg", float("nan")))
            if sample_idx <= 0:
                continue
            if not all(np.isfinite(v) for v in (steps_az, steps_alt, az_deg, alt_deg)):
                continue
            roll_deg = _as_float(row.get("roll_deg", float("nan")))
            ref_steps_az = _as_float(row.get("ref_steps_az", 0.0), 0.0)
            ref_steps_alt = _as_float(row.get("ref_steps_alt", 0.0), 0.0)
            ref_az_mount = _as_float(row.get("ref_az_mount_deg", 0.0), 0.0)
            ref_alt_mount = _as_float(row.get("ref_alt_mount_deg", 0.0), 0.0)
            ts_unix = _as_float(row.get("ts_unix", float("nan")))
            manual_entries.append(
                {
                    "sample_idx": int(sample_idx),
                    "steps": np.array([steps_az, steps_alt], dtype=np.float64),
                    "az_alt": np.array([_wrap_deg_360(az_deg), alt_deg], dtype=np.float64),
                    "roll_deg": float(roll_deg),
                    "synced": bool(_as_bool(row.get("synced", 0), False)),
                    "ref_steps": np.array([ref_steps_az, ref_steps_alt], dtype=np.float64),
                    "ref_az_alt_mount": np.array([_wrap_deg_360(ref_az_mount), ref_alt_mount], dtype=np.float64),
                    "ts_unix": float(ts_unix) if np.isfinite(ts_unix) else float("nan"),
                    "last_direction_az": _as_int(row.get("last_direction_az", 0), 0),
                    "last_direction_alt": _as_int(row.get("last_direction_alt", 0), 0),
                    "backlash_steps_az": _as_int(row.get("backlash_steps_az", self.backlash_steps_az), self.backlash_steps_az),
                    "backlash_steps_alt": _as_int(row.get("backlash_steps_alt", self.backlash_steps_alt), self.backlash_steps_alt),
                    "R": _rotation_from_row(row),
                }
            )

        # Keep only the latest contiguous block by sample_idx (last session-like block).
        manual_session: List[Dict[str, Any]] = []
        if manual_entries:
            expected = int(manual_entries[-1]["sample_idx"])
            for entry in reversed(manual_entries):
                if int(entry["sample_idx"]) != expected:
                    break
                manual_session.append(entry)
                if expected <= 1:
                    break
                expected -= 1
            manual_session.reverse()

        manual_session_start_ts = float("nan")
        if manual_session:
            manual_session_start_ts = float(manual_session[0].get("ts_unix", float("nan")))

        fit_candidates: List[Dict[str, Any]] = []
        for row in reversed(fit_rows):
            # Failed fit attempts also persist the current matrix for
            # diagnostics.  They are not valid restore anchors; keep looking
            # for the latest successful fit (legacy rows without ``ok`` are
            # treated as successful).
            if not _as_bool(row.get("ok", 1), True):
                continue
            J00 = _as_float(row.get("J00", float("nan")))
            J01 = _as_float(row.get("J01", float("nan")))
            J10 = _as_float(row.get("J10", float("nan")))
            J11 = _as_float(row.get("J11", float("nan")))
            if not all(np.isfinite(v) for v in (J00, J01, J10, J11)):
                continue
            J_logged = np.array([[J00, J01], [J10, J11]], dtype=np.float64)
            ts_unix = _as_float(row.get("ts_unix", float("nan")))
            belongs_to_manual_session = bool(
                manual_session
                and np.isfinite(manual_session_start_ts)
                and np.isfinite(ts_unix)
                and float(ts_unix) >= float(manual_session_start_ts)
            )
            if manual_session and not belongs_to_manual_session:
                # A fit from an earlier session must never be attached to the
                # latest contiguous sample block merely because its nominal
                # gearbox happens to match the startup configuration.
                continue
            used_samples = _as_int(
                row.get("used_samples", row.get("model_fit_samples", 0)), 0
            )
            total_samples = _as_int(row.get("total_samples", used_samples), used_samples)
            if belongs_to_manual_session and (
                used_samples <= 0
                or used_samples > len(manual_session)
                or total_samples > len(manual_session)
            ):
                continue

            if not self.is_J_within_mechanical_limits(J_logged):
                # Motor, reducer and microstep values are physical
                # configuration, never inferred fit parameters.  Keep old
                # anomalous rows for diagnostics but do not restore them.
                continue
            fit_candidates.append({
                "J": J_logged,
                "J00_err": _as_float(row.get("J00_err", 0.0), 0.0),
                "J01_err": _as_float(row.get("J01_err", 0.0), 0.0),
                "J10_err": _as_float(row.get("J10_err", 0.0), 0.0),
                "J11_err": _as_float(row.get("J11_err", 0.0), 0.0),
                "model_fit_rms_az_deg": _as_float(row.get("model_fit_rms_az_deg", 0.0), 0.0),
                "model_fit_rms_alt_deg": _as_float(row.get("model_fit_rms_alt_deg", 0.0), 0.0),
                "model_fit_rms_arcsec": _as_float(row.get("model_fit_rms_arcsec", 0.0), 0.0),
                "model_non_orthogonality_deg": _as_float(row.get("model_non_orthogonality_deg", 0.0), 0.0),
                "model_non_orthogonality_err_deg": _as_float(row.get("model_non_orthogonality_err_deg", 0.0), 0.0),
                "model_roll_deg": _as_float(row.get("model_roll_deg", 0.0), 0.0),
                "model_roll_err_deg": _as_float(row.get("model_roll_err_deg", 0.0), 0.0),
                "model_roll_samples": _as_int(row.get("model_roll_samples", 0), 0),
                "model_pitch_deg": _as_float(row.get("model_pitch_deg", 0.0), 0.0),
                "model_pitch_err_deg": _as_float(row.get("model_pitch_err_deg", 0.0), 0.0),
                "model_yaw_deg": _as_float(row.get("model_yaw_deg", 0.0), 0.0),
                "model_yaw_err_deg": _as_float(row.get("model_yaw_err_deg", 0.0), 0.0),
                "periodic_coeff_deg": np.array(
                    [
                        [
                            _as_float(row.get("periodic_az_sin_deg", 0.0), 0.0),
                            _as_float(row.get("periodic_az_cos_deg", 0.0), 0.0),
                        ],
                        [
                            _as_float(row.get("periodic_alt_sin_deg", 0.0), 0.0),
                            _as_float(row.get("periodic_alt_cos_deg", 0.0), 0.0),
                        ],
                    ],
                    dtype=np.float64,
                ),
                "periodic_model_samples": _as_int(row.get("periodic_model_samples", 0), 0),
                "model_fit_samples": int(used_samples),
                "ts_unix": float(ts_unix) if np.isfinite(ts_unix) else float("nan"),
                "R": _rotation_from_row(row),
            })

        fit_entry: Optional[Dict[str, Any]] = None
        if fit_candidates:
            finite_rms = [
                float(e["model_fit_rms_arcsec"])
                for e in fit_candidates
                if np.isfinite(float(e["model_fit_rms_arcsec"]))
                and float(e["model_fit_rms_arcsec"]) >= 0.0
            ]
            if finite_rms:
                best_rms = min(finite_rms)
                # Prefer the newest fit among statistically comparable
                # candidates, but never replace a clean fit with a much worse
                # one merely because a backlash-corrupted sample was added.
                rms_limit = max(best_rms + 10.0, best_rms * 1.5)
                acceptable = [
                    e
                    for e in fit_candidates
                    if np.isfinite(float(e["model_fit_rms_arcsec"]))
                    and float(e["model_fit_rms_arcsec"]) <= rms_limit
                ]
            else:
                acceptable = list(fit_candidates)
            fit_entry = max(
                acceptable,
                key=lambda e: (
                    int(e.get("model_fit_samples", 0)),
                    float(e.get("ts_unix", float("-inf"))),
                ),
            )

        if not manual_session and fit_entry is None:
            return {
                "ok": False,
                "status": "NO_VALID_ROWS",
                "manual_samples": 0,
                "synced": bool(self.synced),
                "camera_roll_deg": float("nan"),
            }

        prev_steps = self.steps_est.copy()

        self._calib_steps.clear()
        self._calib_daltaz.clear()
        self._manual_steps_abs.clear()
        self._manual_az_alt_abs.clear()
        self._manual_roll_deg_abs.clear()
        self._manual_source_abs.clear()
        self._manual_fit_inlier_mask = None
        self.steps_history.clear()

        self.synced = False
        self.ref_steps = prev_steps.copy()
        self.ref_az_alt_deg = np.zeros(2, dtype=np.float64)
        self.last_solve_az_alt_deg = None
        self.last_solve_steps_est = None
        self.last_solve_time = 0.0

        self.J00_err = 0.0
        self.J01_err = 0.0
        self.J10_err = 0.0
        self.J11_err = 0.0
        self.model_non_orthogonality_deg = _non_orthogonality_deg_from_J(self.J_deg_per_step)
        self.model_non_orthogonality_err_deg = 0.0
        self.model_roll_deg = 0.0
        self.model_roll_err_deg = 0.0
        self.model_roll_samples = 0
        self.model_pitch_deg = 0.0
        self.model_pitch_err_deg = 0.0
        self.model_yaw_deg = 0.0
        self.model_yaw_err_deg = 0.0
        self.model_fit_samples = 0
        self.model_fit_rms_az_deg = 0.0
        self.model_fit_rms_alt_deg = 0.0
        self.model_fit_rms_arcsec = 0.0

        latest_rot_ts = float("-inf")
        latest_rot = np.eye(3, dtype=np.float64)

        if manual_session:
            self._manual_steps_abs = [np.asarray(e["steps"], dtype=np.float64).copy() for e in manual_session]
            self._manual_az_alt_abs = [np.asarray(e["az_alt"], dtype=np.float64).copy() for e in manual_session]
            self._manual_roll_deg_abs = [float(e["roll_deg"]) for e in manual_session]
            self._manual_source_abs = ["" for _ in manual_session]

            last = manual_session[-1]
            self.steps_est = np.asarray(last["steps"], dtype=np.float64).copy()
            self.last_move_direction_az = int(last.get("last_direction_az", 0))
            self.last_move_direction_alt = int(last.get("last_direction_alt", 0))
            self.backlash_steps_az = max(0, int(last.get("backlash_steps_az", self.backlash_steps_az)))
            self.backlash_steps_alt = max(0, int(last.get("backlash_steps_alt", self.backlash_steps_alt)))
            if self.last_move_direction_az not in (-1, +1):
                self.last_move_direction_az = 0
            if self.last_move_direction_alt not in (-1, +1):
                self.last_move_direction_alt = 0
            if len(manual_session) >= 2:
                delta_last = np.asarray(manual_session[-1]["steps"], dtype=np.float64) - np.asarray(manual_session[-2]["steps"], dtype=np.float64)
                if self.last_move_direction_az == 0 and abs(float(delta_last[0])) >= 1.0:
                    self.last_move_direction_az = +1 if float(delta_last[0]) > 0.0 else -1
                if self.last_move_direction_alt == 0 and abs(float(delta_last[1])) >= 1.0:
                    self.last_move_direction_alt = +1 if float(delta_last[1]) > 0.0 else -1
            self.last_solve_steps_est = self.steps_est.copy()
            self.last_solve_az_alt_deg = np.asarray(last["az_alt"], dtype=np.float64).copy()
            ts = float(last["ts_unix"])
            self.last_solve_time = ts if np.isfinite(ts) else 0.0
            self.synced = bool(last["synced"])
            if self.synced:
                self.ref_steps = np.asarray(last["ref_steps"], dtype=np.float64).copy()
                self.ref_az_alt_deg = np.asarray(last["ref_az_alt_mount"], dtype=np.float64).copy()
            else:
                self.ref_steps = self.steps_est.copy()
                self.ref_az_alt_deg = np.zeros(2, dtype=np.float64)

            R_last = last.get("R", None)
            if R_last is not None:
                latest_rot = np.asarray(R_last, dtype=np.float64)
                latest_rot_ts = ts if np.isfinite(ts) else float("-inf")
        else:
            self.steps_est = prev_steps.copy()
            self.ref_steps = self.steps_est.copy()

        if fit_entry is not None:
            self.J_deg_per_step = self.constrain_J_to_mechanics(
                np.asarray(fit_entry["J"], dtype=np.float64)
            )
            self.J00_err = float(fit_entry["J00_err"])
            self.J01_err = float(fit_entry["J01_err"])
            self.J10_err = float(fit_entry["J10_err"])
            self.J11_err = float(fit_entry["J11_err"])
            self.model_non_orthogonality_deg = float(fit_entry["model_non_orthogonality_deg"])
            self.model_non_orthogonality_err_deg = float(fit_entry["model_non_orthogonality_err_deg"])
            self.model_roll_deg = float(fit_entry["model_roll_deg"])
            self.model_roll_err_deg = float(fit_entry["model_roll_err_deg"])
            self.model_roll_samples = int(fit_entry["model_roll_samples"])
            self.model_pitch_deg = float(fit_entry["model_pitch_deg"])
            self.model_pitch_err_deg = float(fit_entry["model_pitch_err_deg"])
            self.model_yaw_deg = float(fit_entry["model_yaw_deg"])
            self.model_yaw_err_deg = float(fit_entry["model_yaw_err_deg"])
            self.model_fit_samples = int(max(0, int(fit_entry["model_fit_samples"])))
            self.model_fit_rms_az_deg = float(fit_entry["model_fit_rms_az_deg"])
            self.model_fit_rms_alt_deg = float(fit_entry["model_fit_rms_alt_deg"])
            self.model_fit_rms_arcsec = float(fit_entry["model_fit_rms_arcsec"])
            self.periodic_coeff_deg = self.safe_periodic_coeff_for_prediction(
                fit_entry.get("periodic_coeff_deg", np.zeros((2, 2), dtype=np.float64))
            )
            self.periodic_model_samples = max(
                0, int(fit_entry.get("periodic_model_samples", 0))
            )

            R_fit = fit_entry.get("R", None)
            ts_fit = float(fit_entry["ts_unix"])
            if R_fit is not None and (not np.isfinite(latest_rot_ts) or ts_fit >= latest_rot_ts):
                latest_rot = np.asarray(R_fit, dtype=np.float64)

        self.R_mount_to_world = _limit_rotation_tilt_ns_oe_deg(
            _coerce_rotation_matrix(latest_rot),
            max_tilt_deg=float(self.max_tilt_ns_oe_deg),
        )

        # Manual samples are absolute plate-solving measurements.  A normal
        # calibration session records them before Fit Model performs its final
        # sync, so their CSV rows commonly have synced=0 even though the fit is
        # valid.  Restoring such a session must recreate the same final sync;
        # otherwise GoTo immediately fails with ERR_NOT_SYNCED after restart.
        synced_from_manual = False
        if manual_session and fit_entry is not None and not bool(self.synced):
            synced_from_manual = bool(self.sync_from_latest_manual_sample())

        camera_roll_deg = float("nan")
        if fit_entry is not None:
            cand = float(fit_entry.get("model_roll_deg", float("nan")))
            if np.isfinite(cand):
                camera_roll_deg = _wrap_deg_180(cand)
        if (not np.isfinite(camera_roll_deg)) and manual_session:
            for entry in reversed(manual_session):
                cand = float(entry.get("roll_deg", float("nan")))
                if np.isfinite(cand):
                    camera_roll_deg = _wrap_deg_180(cand)
                    break

        return {
            "ok": True,
            "status": "OK",
            "manual_samples": int(len(self._manual_steps_abs)),
            "synced": bool(self.synced),
            "synced_from_manual": bool(synced_from_manual),
            "loaded_manual": bool(manual_session),
            "loaded_fit": bool(fit_entry is not None),
            "camera_roll_deg": float(camera_roll_deg),
            "manual_path": str(manual_path),
            "fit_path": str(fit_path),
        }

    def _rotation_mount_to_world(self) -> np.ndarray:
        self.R_mount_to_world = _limit_rotation_tilt_ns_oe_deg(
            _coerce_rotation_matrix(self.R_mount_to_world),
            max_tilt_deg=float(self.max_tilt_ns_oe_deg),
        )
        return self.R_mount_to_world

    def _mount_to_world_altaz(self, az_alt_mount_deg: np.ndarray) -> np.ndarray:
        return _rotate_altaz_deg(_as_array2(az_alt_mount_deg), self._rotation_mount_to_world())

    def _world_to_mount_altaz(self, az_alt_world_deg: np.ndarray) -> np.ndarray:
        R = self._rotation_mount_to_world()
        return _rotate_altaz_deg(_as_array2(az_alt_world_deg), R.T)

    def sync_from_world_az_alt(self, az_alt_world_deg: np.ndarray) -> bool:
        """Sync reference from a world AltAz measurement."""
        ref_world = _as_array2(az_alt_world_deg)
        ref_mount = self._world_to_mount_altaz(ref_world)

        self.synced = True
        self.ref_steps = self.steps_est.copy()
        self.ref_az_alt_deg = ref_mount.copy()
        self.last_solve_az_alt_deg = ref_world.copy()
        self.last_solve_steps_est = self.steps_est.copy()
        self.last_solve_time = time.time()
        return True

    def predict_az_alt_deg(self, *, from_ref: bool = False) -> np.ndarray:
        """Predict current mount AZ/ALT from the model + steps.

        If from_ref=True, returns the world AltAz corresponding to ref_az_alt_deg.
        """
        if from_ref or (not self.synced):
            return self._mount_to_world_altaz(self.ref_az_alt_deg)
        dsteps = self.steps_est - self.ref_steps
        daltaz = self.mount_delta_for_steps(self.ref_steps, self.steps_est)
        az_mount = _wrap_deg_360(float(self.ref_az_alt_deg[0]) + float(daltaz[0]))
        alt_mount = float(self.ref_az_alt_deg[1] + float(daltaz[1]))
        return self._mount_to_world_altaz(np.array([az_mount, alt_mount], dtype=np.float64))

    def current_az_alt_deg(self) -> Optional[np.ndarray]:
        """Best estimate of current mount AZ/ALT.

        Prefers last successful plate-solve, otherwise the model prediction.
        """
        if self.last_solve_az_alt_deg is not None:
            if self.last_solve_steps_est is not None:
                dsteps = self.steps_est - self.last_solve_steps_est
                daltaz = self.mount_delta_for_steps(
                    self.last_solve_steps_est,
                    self.steps_est,
                )
                last_mount = self._world_to_mount_altaz(self.last_solve_az_alt_deg)
                az_mount = _wrap_deg_360(float(last_mount[0]) + float(daltaz[0]))
                alt_mount = float(last_mount[1]) + float(daltaz[1])
                return self._mount_to_world_altaz(np.array([az_mount, alt_mount], dtype=np.float64))
            return self.last_solve_az_alt_deg.copy()
        if not self.synced:
            return None
        return self.predict_az_alt_deg()

    def _world_deg_per_step_matrix(
        self,
        *,
        az_deg: float,
        alt_deg: float,
        deriv_step: float = 64.0,
    ) -> Optional[np.ndarray]:
        J = self.safe_J_for_prediction()

        az0 = _wrap_deg_360(float(az_deg))
        alt0 = float(np.clip(float(alt_deg), -89.5, 89.5))
        if (not np.isfinite(az0)) or (not np.isfinite(alt0)):
            return None

        base_world = np.array([az0, alt0], dtype=np.float64)
        base_mount = self._world_to_mount_altaz(base_world)
        eps = max(1.0, abs(float(deriv_step)))
        cols: List[np.ndarray] = []

        for axis_idx in range(2):
            dsteps = np.zeros(2, dtype=np.float64)
            dsteps[axis_idx] = eps
            dmount = self.mount_delta_for_steps(
                self.steps_est,
                self.steps_est + dsteps,
                J_model=J,
            )

            mount_p = np.array(
                [
                    _wrap_deg_360(float(base_mount[0]) + float(dmount[0])),
                    float(np.clip(float(base_mount[1]) + float(dmount[1]), -89.5, 89.5)),
                ],
                dtype=np.float64,
            )
            mount_m = np.array(
                [
                    _wrap_deg_360(float(base_mount[0]) - float(dmount[0])),
                    float(np.clip(float(base_mount[1]) - float(dmount[1]), -89.5, 89.5)),
                ],
                dtype=np.float64,
            )

            world_p = self._mount_to_world_altaz(mount_p)
            world_m = self._mount_to_world_altaz(mount_m)
            daz = _wrap_deg_180(float(world_p[0]) - float(world_m[0]))
            dalt = float(world_p[1]) - float(world_m[1])
            cols.append(np.array([daz / (2.0 * eps), dalt / (2.0 * eps)], dtype=np.float64))

        J_world_step = np.column_stack(cols)
        if J_world_step.shape != (2, 2) or (not np.all(np.isfinite(J_world_step))):
            return None
        return J_world_step

    def world_altaz_rate_to_step_rate_deg_s(
        self,
        *,
        az_deg: float,
        alt_deg: float,
        world_rate_deg_s: Sequence[float],
        deriv_step: float = 64.0,
        cond_max: float = 5_000.0,
    ) -> Optional[np.ndarray]:
        v_world = _as_array2(world_rate_deg_s)
        if not np.all(np.isfinite(v_world)):
            return None

        J_world_step = self._world_deg_per_step_matrix(
            az_deg=float(az_deg),
            alt_deg=float(alt_deg),
            deriv_step=float(deriv_step),
        )
        if J_world_step is None:
            return None

        try:
            cond = float(np.linalg.cond(J_world_step))
        except np.linalg.LinAlgError:
            return None
        if (not np.isfinite(cond)) or cond > float(cond_max):
            return None

        try:
            rate_steps = np.linalg.solve(J_world_step, v_world)
        except np.linalg.LinAlgError:
            rate_steps, *_ = np.linalg.lstsq(J_world_step, v_world, rcond=None)

        if not np.all(np.isfinite(rate_steps)):
            return None
        return np.asarray(rate_steps, dtype=np.float64).reshape(2,)

    def sidereal_world_rate_deg_s(
        self,
        *,
        az_deg: float,
        alt_deg: float,
        observer: ObserverConfig,
        obstime: Optional[Time] = None,
        dt_s: float = 1.0,
    ) -> Optional[np.ndarray]:
        dt = float(dt_s)
        if (not np.isfinite(dt)) or dt <= 1e-6:
            return None

        az0 = _wrap_deg_360(float(az_deg))
        alt0 = float(np.clip(float(alt_deg), -89.5, 89.5))
        if (not np.isfinite(az0)) or (not np.isfinite(alt0)):
            return None

        t0 = obstime if obstime is not None else _now_time()
        try:
            coord_icrs = resolve_target_icrs(
                {"az_deg": az0, "alt_deg": alt0},
                observer=observer,
                obstime=t0,
            )
            altaz_1 = icrs_to_altaz_deg(
                coord_icrs,
                observer=observer,
                obstime=t0 + dt * u.s,
            )
        except Exception as exc:
            log_error(None, "GoTo: sidereal world-rate prediction failed", exc, throttle_s=5.0, throttle_key="goto_sidereal_world_rate")
            return None

        daz = _wrap_deg_180(float(altaz_1[0]) - az0)
        dalt = float(altaz_1[1]) - alt0
        rate = np.array([daz / dt, dalt / dt], dtype=np.float64)
        if not np.all(np.isfinite(rate)):
            return None
        return rate

    def sidereal_step_rate_deg_s(
        self,
        *,
        az_deg: float,
        alt_deg: float,
        observer: ObserverConfig,
        obstime: Optional[Time] = None,
        dt_s: float = 1.0,
        deriv_step: float = 64.0,
        cond_max: float = 5_000.0,
    ) -> Optional[np.ndarray]:
        world_rate = self.sidereal_world_rate_deg_s(
            az_deg=float(az_deg),
            alt_deg=float(alt_deg),
            observer=observer,
            obstime=obstime,
            dt_s=float(dt_s),
        )
        if world_rate is None:
            return None
        return self.world_altaz_rate_to_step_rate_deg_s(
            az_deg=float(az_deg),
            alt_deg=float(alt_deg),
            world_rate_deg_s=world_rate,
            deriv_step=float(deriv_step),
            cond_max=float(cond_max),
        )

    def apply_plate_solve(self, az_alt_deg: np.ndarray) -> bool:
        """Update last solve and reconcile steps_est with the solved AltAz.

        Returns True if steps_est was updated from the solve.
        """
        az_alt_world = _as_array2(az_alt_deg)
        updated_steps = False

        if self.synced:
            az_alt_mount = self._world_to_mount_altaz(az_alt_world)
            daltaz = np.array(
                [
                    _wrap_deg_180(float(az_alt_mount[0]) - float(self.ref_az_alt_deg[0])),
                    float(az_alt_mount[1]) - float(self.ref_az_alt_deg[1]),
                ],
                dtype=np.float64,
            )

            try:
                dsteps = self.solve_step_delta_for_mount_delta(
                    daltaz,
                    steps_from=self.ref_steps,
                )
            except np.linalg.LinAlgError as exc:
                log_error(None, "GoTo: failed to reconcile steps from plate-solve", exc, throttle_s=5.0, throttle_key="goto_steps_reconcile")
                dsteps = None

            if dsteps is not None:
                if not np.all(np.isfinite(dsteps)):
                    log_error(None, "GoTo: non-finite steps from plate-solve reconciliation", None, throttle_s=5.0, throttle_key="goto_steps_reconcile_nan")
                else:
                    self.steps_est = self.ref_steps + dsteps
                    updated_steps = True

        self.last_solve_az_alt_deg = az_alt_world.copy()
        self.last_solve_steps_est = self.steps_est.copy()
        self.last_solve_time = time.time()
        return bool(updated_steps)

    def add_calibration_sample(self, dsteps: np.ndarray, daltaz_deg: np.ndarray) -> None:
        self._calib_steps.append(_as_array2(dsteps))
        self._calib_daltaz.append(_as_array2(daltaz_deg))

    def fit_J_from_samples(self, *, min_samples: int = 3, ridge: float = 1e-12) -> bool:
        """Least squares fit of J using accumulated calibration samples.

        We solve D = S @ B and set J = B^T (so that d = J @ s).

        Returns True if an update was applied.
        """
        self.last_fit_reason = "RUNNING"
        if len(self._calib_steps) < int(min_samples):
            self.last_fit_reason = "INSUFFICIENT_SAMPLES"
            self._log_fit_csv(
                fit_kind="calibration",
                ok=False,
                reason="INSUFFICIENT_SAMPLES",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(len(self._calib_steps)),
                used_samples=0,
            )
            return False
        S_all = np.stack(self._calib_steps, axis=0).astype(np.float64)  # (N,2)
        D_all = np.stack(self._calib_daltaz, axis=0).astype(np.float64)  # (N,2)
        n_all = int(S_all.shape[0])

        def _solve_with_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
            idx = np.flatnonzero(mask)
            n_use = int(idx.size)
            if n_use < int(min_samples):
                return None

            S = S_all[idx, :]
            D = D_all[idx, :]

            mechanical = self.mechanical_J()
            span_steps = np.ptp(S, axis=0)
            span_deg = np.array(
                [
                    abs(float(mechanical[0, 0])) * float(span_steps[0]),
                    abs(float(mechanical[1, 1])) * float(span_steps[1]),
                ],
                dtype=np.float64,
            )
            min_span_deg = max(0.0, float(self.min_fit_axis_span_deg))
            if np.any(~np.isfinite(span_deg)) or np.any(span_deg < min_span_deg):
                return None

            # Ridge-regularized least squares: minimize ||S B - D||^2 + ridge||B||^2
            # Implemented by augmenting S and D.
            if ridge > 0:
                lam = float(ridge)
                S_aug = np.vstack([S, math.sqrt(lam) * np.eye(2)])
                D_aug = np.vstack([D, np.zeros((2, 2), dtype=np.float64)])
            else:
                S_aug, D_aug = S, D

            B, *_ = np.linalg.lstsq(S_aug, D_aug, rcond=None)
            J_unconstrained = np.asarray(B.T, dtype=np.float64).copy()
            J_new = self.constrain_J_to_mechanics(J_unconstrained)
            B = J_new.T

            if not np.all(np.isfinite(J_new)):
                return None
            det_J = float(np.linalg.det(J_new))
            if (not np.isfinite(det_J)) or abs(det_J) < 1e-12:
                return None
            try:
                cond_J = float(np.linalg.cond(J_new))
            except np.linalg.LinAlgError:
                return None
            if (not np.isfinite(cond_J)) or cond_J > 1e10:
                return None

            pred_use = S @ B
            res_use = D - pred_use
            pred_all = S_all @ B
            res_all = D_all - pred_all
            return {
                "mask": mask.copy(),
                "n_use": n_use,
                "J_new": J_new,
                "J_unconstrained": J_unconstrained,
                "res_use": res_use,
                "res_all": res_all,
            }

        mask = np.ones(n_all, dtype=bool)
        fit = _solve_with_mask(mask)
        if fit is None:
            self.last_fit_reason = "DEGENERATE_MODEL"
            self._log_fit_csv(
                fit_kind="calibration",
                ok=False,
                reason="DEGENERATE_MODEL",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n_all),
                used_samples=0,
            )
            return False

        # Robust outlier rejection on total residual norm using MAD scale.
        min_keep = max(int(min_samples), 3)
        if n_all >= max(min_keep + 2, 5):
            for _ in range(3):
                res_all = np.asarray(fit["res_all"], dtype=np.float64)
                res_norm = np.hypot(res_all[:, 0], res_all[:, 1])
                finite = np.isfinite(res_norm)
                if int(np.sum(finite)) < min_keep:
                    break

                med = float(np.median(res_norm[finite]))
                mad = float(np.median(np.abs(res_norm[finite] - med)))
                sigma = float(1.4826 * mad)
                floor = float(max(0.0, self.fit_outlier_floor_arcsec) / 3600.0)
                thr = med + float(max(0.0, self.fit_outlier_sigma)) * sigma
                thr = max(floor, float(thr if np.isfinite(thr) else 0.0))

                new_mask = finite & (res_norm <= thr)
                if int(np.sum(new_mask)) < min_keep:
                    idx_f = np.flatnonzero(finite)
                    order = idx_f[np.argsort(res_norm[idx_f])]
                    keep = order[:min_keep]
                    new_mask = np.zeros_like(mask, dtype=bool)
                    new_mask[keep] = True

                if np.array_equal(new_mask, mask):
                    # Fallback: if one sample is clearly separated, prune the worst.
                    idx_f = np.flatnonzero(finite & mask)
                    if int(idx_f.size) <= min_keep:
                        break
                    worst = int(idx_f[np.argmax(res_norm[idx_f])])
                    worst_res = float(res_norm[worst])
                    med_ref = max(float(med), floor)
                    if not (np.isfinite(worst_res) and worst_res > max(3.0 * floor, 2.0 * med_ref)):
                        break
                    new_mask = mask.copy()
                    new_mask[worst] = False

                fit_new = _solve_with_mask(new_mask)
                if fit_new is None:
                    break
                mask = new_mask
                fit = fit_new

        if not self.is_J_within_mechanical_limits(fit["J_unconstrained"]):
            # Distinguish "the gearing is wrong" from "the samples are too
            # short to measure the mean scale". With moves well under one
            # cycloidal lobe the fit reads the local slope of the transmission
            # error, which alone can exceed the envelope on perfect hardware.
            coverage = self.manual_phase_coverage(mask)
            short_travel = float(coverage.get("min", 0.0)) < 0.5
            reason = (
                "MODEL_FIT_PHASE_COVERAGE_TOO_SHORT"
                if short_travel
                else "MODEL_OUTSIDE_MECHANICAL_LIMITS"
            )
            self.last_fit_reason = reason
            self.last_fit_phase_coverage = dict(coverage)
            self._log_fit_csv(
                fit_kind="calibration",
                ok=False,
                reason=reason,
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n_all),
                used_samples=int(fit["n_use"]),
            )
            return False

        res = np.asarray(fit["res_use"], dtype=np.float64)
        fit_rms_arcsec = float(
            np.sqrt(np.mean(np.square(res[:, 0]) + np.square(res[:, 1]))) * 3600.0
        )
        max_rms = max(0.0, float(self.max_model_fit_rms_arcsec))
        if (
            not np.isfinite(fit_rms_arcsec)
            or (max_rms > 0.0 and fit_rms_arcsec > max_rms)
        ):
            self.last_fit_reason = "FIT_RMS_TOO_HIGH"
            self._log_fit_csv(
                fit_kind="calibration",
                ok=False,
                reason="FIT_RMS_TOO_HIGH",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n_all),
                used_samples=int(fit["n_use"]),
            )
            return False

        self.J_deg_per_step = np.asarray(fit["J_new"], dtype=np.float64).copy()
        self.J00_err = 0.0
        self.J01_err = 0.0
        self.J10_err = 0.0
        self.J11_err = 0.0
        self.model_non_orthogonality_deg = _non_orthogonality_deg_from_J(self.J_deg_per_step)
        self.model_non_orthogonality_err_deg = 0.0
        self.model_fit_samples = int(fit["n_use"])
        self.model_fit_rms_az_deg = float(np.sqrt(np.mean(np.square(res[:, 0]))))
        self.model_fit_rms_alt_deg = float(np.sqrt(np.mean(np.square(res[:, 1]))))
        self.model_fit_rms_arcsec = fit_rms_arcsec
        n_out = int(n_all - int(fit["n_use"]))
        if n_out > 0:
            log_info(
                None,
                f"GoTo: calibration fit rejected outliers={n_out}/{n_all}",
                throttle_s=0.2,
                throttle_key="goto_fit_calib_outliers",
            )
        self._log_fit_csv(
            fit_kind="calibration",
            ok=True,
            reason="OK",
            min_samples=int(min_samples),
            ridge=float(ridge),
            total_samples=int(n_all),
            used_samples=int(fit["n_use"]),
        )
        self.last_fit_reason = "OK"
        return True

    def add_manual_sample(
        self,
        az_alt_deg: np.ndarray,
        *,
        theta_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
        source: Optional[str] = None,
    ) -> int:
        """Store an absolute (steps, AltAz) sample from a manual plate-solve."""
        az_alt = _as_array2(az_alt_deg)
        self._manual_steps_abs.append(self.steps_est.copy())
        self._manual_az_alt_abs.append(az_alt.copy())
        # Backward-compatible argument: theta_deg interpreted as a roll sample (deg).
        roll_sample = roll_deg if roll_deg is not None else theta_deg
        roll_store = float("nan")
        if roll_sample is not None:
            roll = float(roll_sample)
            if np.isfinite(roll):
                roll_store = _wrap_deg_180(roll)
        # Keep one roll slot per sample (NaN when missing) so index-based pruning is safe.
        self._manual_roll_deg_abs.append(float(roll_store))
        self._manual_source_abs.append(str(source or ""))
        n_samples = int(len(self._manual_steps_abs))
        # Preserve which older samples supported the accepted model. The new
        # sample starts as unvalidated and is admitted by the next robust fit
        # only if its residual is compatible.
        if self._manual_fit_inlier_mask is not None:
            old_mask = np.asarray(self._manual_fit_inlier_mask, dtype=bool).reshape(-1)
            if old_mask.size == n_samples - 1:
                self._manual_fit_inlier_mask = np.concatenate(
                    [old_mask, np.array([False], dtype=bool)]
                )
            else:
                self._manual_fit_inlier_mask = None
        self.last_solve_az_alt_deg = az_alt.copy()
        self.last_solve_steps_est = self.steps_est.copy()
        self.last_solve_time = time.time()
        self._log_manual_sample_csv(sample_idx=n_samples, az_alt_world=az_alt, roll_sample=roll_sample)
        return n_samples

    def manual_sample_continuity_report(
        self,
        az_alt_deg: np.ndarray,
        *,
        roll_deg: Optional[float] = None,
        reference_steps: Optional[np.ndarray] = None,
        reference_az_alt_deg: Optional[np.ndarray] = None,
        reference_roll_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check a plate-solve against the preceding physical mount motion.

        The check is deliberately separate from ``add_manual_sample`` so
        synthetic/offline model-fitting tests and explicit log restoration can
        still load arbitrary samples. Live calibration paths call this before
        mutating the model.
        """
        candidate_world = _as_array2(az_alt_deg)

        if reference_steps is None or reference_az_alt_deg is None:
            if not self._manual_steps_abs or not self._manual_az_alt_abs:
                return {
                    "ok": True,
                    "has_reference": False,
                    "motion_ok": True,
                    "roll_ok": True,
                }
            ref_steps = _as_array2(self._manual_steps_abs[-1])
            ref_world = _as_array2(self._manual_az_alt_abs[-1])
            if reference_roll_deg is None and self._manual_roll_deg_abs:
                roll_prev = float(self._manual_roll_deg_abs[-1])
                if np.isfinite(roll_prev):
                    reference_roll_deg = roll_prev
        else:
            ref_steps = _as_array2(reference_steps)
            ref_world = _as_array2(reference_az_alt_deg)

        dsteps = self.steps_est.copy() - ref_steps
        ref_mount = self._world_to_mount_altaz(ref_world)
        candidate_mount = self._world_to_mount_altaz(candidate_world)
        observed_delta = np.array(
            [
                _wrap_deg_180(float(candidate_mount[0]) - float(ref_mount[0])),
                float(candidate_mount[1]) - float(ref_mount[1]),
            ],
            dtype=np.float64,
        )

        mechanical_delta = self.mechanical_J() @ dsteps
        fitted_delta = self.constrain_J_to_mechanics(self.J_deg_per_step) @ dsteps
        expected_motion_deg = max(
            float(np.linalg.norm(mechanical_delta)),
            float(np.linalg.norm(fitted_delta)),
        )
        observed_motion_deg = float(np.linalg.norm(observed_delta))
        motion_limit_deg = (
            max(0.0, float(self.manual_sample_motion_margin_deg))
            + max(0.0, float(self.manual_sample_motion_factor)) * expected_motion_deg
        )
        motion_ok = bool(
            np.isfinite(observed_motion_deg)
            and observed_motion_deg <= motion_limit_deg
        )

        roll_jump_deg = float("nan")
        roll_ok = True
        if (
            roll_deg is not None
            and reference_roll_deg is not None
            and np.isfinite(float(roll_deg))
            and np.isfinite(float(reference_roll_deg))
        ):
            roll_jump_deg = roll_axis_distance_deg(
                float(roll_deg),
                float(reference_roll_deg),
            )
            roll_tol = max(0.0, float(self.manual_sample_roll_tolerance_deg))
            roll_ok = bool(roll_tol <= 0.0 or roll_jump_deg <= roll_tol)

        return {
            "ok": bool(motion_ok and roll_ok),
            "has_reference": True,
            "motion_ok": bool(motion_ok),
            "roll_ok": bool(roll_ok),
            "dsteps_az": float(dsteps[0]),
            "dsteps_alt": float(dsteps[1]),
            "observed_daz_deg": float(observed_delta[0]),
            "observed_dalt_deg": float(observed_delta[1]),
            "observed_motion_deg": float(observed_motion_deg),
            "expected_motion_deg": float(expected_motion_deg),
            "motion_limit_deg": float(motion_limit_deg),
            "roll_jump_deg": float(roll_jump_deg),
            "roll_tolerance_deg": float(self.manual_sample_roll_tolerance_deg),
        }

    def bootstrap_axis_scale_from_manual_pair(
        self,
        az_alt_deg: np.ndarray,
        *,
        max_cross_fraction: float = 0.15,
    ) -> Dict[str, Any]:
        """Diagnose a single-axis sample without changing physical mechanics.

        This compatibility/reporting hook intentionally never rewrites motor
        steps, reducer ratio or microstepping from a single plate-solved move.
        Residual transmission error belongs in the bounded multi-sample fit.
        """
        if int(self.model_fit_samples) > 0:
            return {"ok": False, "status": "MODEL_ALREADY_FITTED"}
        if not self._manual_steps_abs or not self._manual_az_alt_abs:
            return {"ok": False, "status": "NO_REFERENCE_SAMPLE"}

        ref_steps = _as_array2(self._manual_steps_abs[-1])
        ref_world = _as_array2(self._manual_az_alt_abs[-1])
        dsteps = self.steps_est.copy() - ref_steps
        active = np.flatnonzero(np.abs(dsteps) >= 1.0)
        if int(active.size) != 1:
            return {
                "ok": False,
                "status": "MOVE_NOT_SINGLE_AXIS",
                "dsteps": dsteps,
            }
        axis_idx = int(active[0])
        cross_idx = 1 - axis_idx
        if abs(float(dsteps[cross_idx])) > 0.02 * abs(float(dsteps[axis_idx])):
            return {"ok": False, "status": "CROSS_AXIS_STEPS", "dsteps": dsteps}

        candidate_world = _as_array2(az_alt_deg)
        ref_mount = self._world_to_mount_altaz(ref_world)
        candidate_mount = self._world_to_mount_altaz(candidate_world)
        observed_delta = np.array(
            [
                _wrap_deg_180(float(candidate_mount[0]) - float(ref_mount[0])),
                float(candidate_mount[1]) - float(ref_mount[1]),
            ],
            dtype=np.float64,
        )
        primary_delta = float(observed_delta[axis_idx])
        if not np.isfinite(primary_delta) or abs(primary_delta) < 0.02:
            return {"ok": False, "status": "MOTION_TOO_SMALL"}
        cross_fraction = abs(float(observed_delta[cross_idx])) / abs(primary_delta)
        if not np.isfinite(cross_fraction) or cross_fraction > max(0.0, float(max_cross_fraction)):
            return {
                "ok": False,
                "status": "CROSS_MOTION_TOO_LARGE",
                "cross_fraction": cross_fraction,
            }

        signed_deg_per_step = primary_delta / float(dsteps[axis_idx])
        abs_deg_per_step = abs(signed_deg_per_step)
        # Broad physical bounds: 18k..36M command steps per full revolution.
        if not np.isfinite(abs_deg_per_step) or not (1e-5 <= abs_deg_per_step <= 0.02):
            return {
                "ok": False,
                "status": "IMPLAUSIBLE_SCALE",
                "deg_per_step": signed_deg_per_step,
            }

        axis = Axis.AZ if axis_idx == 0 else Axis.ALT
        nominal_deg_per_step = float(self.mechanical_J()[axis_idx, axis_idx])
        deviation_fraction = (
            abs(signed_deg_per_step - nominal_deg_per_step)
            / max(abs(nominal_deg_per_step), 1e-15)
        )
        return {
            "ok": False,
            "status": "NOMINAL_KINEMATICS_FIXED",
            "axis": axis.value,
            "deg_per_step": float(signed_deg_per_step),
            "effective_steps_per_deg": float(1.0 / abs_deg_per_step),
            "nominal_deg_per_step": nominal_deg_per_step,
            "deviation_fraction": float(deviation_fraction),
            "cross_fraction": float(cross_fraction),
            "observed_delta": observed_delta,
            "dsteps": dsteps,
        }

    def sync_from_latest_manual_sample(self) -> bool:
        """Set absolute reference from the latest manual (steps, AltAz) sample."""
        if not self._manual_steps_abs or not self._manual_az_alt_abs:
            return False

        ref_steps = _as_array2(self._manual_steps_abs[-1])
        ref_az_alt_world = _as_array2(self._manual_az_alt_abs[-1])
        ref_az_alt_mount = self._world_to_mount_altaz(ref_az_alt_world)

        self.synced = True
        self.ref_steps = ref_steps.copy()
        self.ref_az_alt_deg = ref_az_alt_mount.copy()
        self.last_solve_az_alt_deg = ref_az_alt_world.copy()
        self.last_solve_steps_est = ref_steps.copy()
        self.last_solve_time = time.time()
        return True

    def reset_manual_samples_and_sync(self) -> None:
        """Drop manual sample history and clear current sync/solve anchor."""
        self._manual_steps_abs.clear()
        self._manual_az_alt_abs.clear()
        self._manual_roll_deg_abs.clear()
        self._manual_source_abs.clear()
        self._manual_fit_inlier_mask = None

        self.synced = False
        self.ref_steps = self.steps_est.copy()
        self.ref_az_alt_deg = np.zeros(2, dtype=np.float64)
        self.last_solve_az_alt_deg = None
        self.last_solve_steps_est = None
        self.last_solve_time = 0.0
        self.R_mount_to_world = np.eye(3, dtype=np.float64)

        self.model_roll_deg = 0.0
        self.model_roll_err_deg = 0.0
        self.model_roll_samples = 0
        self.model_pitch_deg = 0.0
        self.model_pitch_err_deg = 0.0
        self.model_yaw_deg = 0.0
        self.model_yaw_err_deg = 0.0
        self.model_fit_samples = 0
        self.model_fit_rms_az_deg = 0.0
        self.model_fit_rms_alt_deg = 0.0
        self.model_fit_rms_arcsec = 0.0
        self.periodic_coeff_deg = np.zeros((2, 2), dtype=np.float64)
        self.periodic_model_samples = 0

    def _manual_reference_index_from_steps(self, S_abs: np.ndarray, idx: np.ndarray) -> int:
        if int(idx.size) <= 1:
            return int(idx[0]) if int(idx.size) == 1 else 0
        S_use = S_abs[idx, :]
        dmat = np.linalg.norm(S_use[:, None, :] - S_use[None, :, :], axis=2)
        med_d = np.median(dmat, axis=1)
        return int(idx[int(np.argmin(med_d))])

    def _fit_rotation_from_manual_samples(
        self,
        *,
        S_abs: np.ndarray,
        A_world: np.ndarray,
        sample_mask: np.ndarray,
        J_model: np.ndarray,
        periodic_coeff: Optional[np.ndarray] = None,
        max_iter: int = 6,
    ) -> np.ndarray:
        idx = np.flatnonzero(np.asarray(sample_mask, dtype=bool))
        if int(idx.size) < 3:
            return self._rotation_mount_to_world()

        J = np.asarray(J_model, dtype=np.float64)
        if J.shape != (2, 2) or (not np.all(np.isfinite(J))):
            return self._rotation_mount_to_world()

        ref_idx = self._manual_reference_index_from_steps(S_abs, idx)
        s_ref = np.asarray(S_abs[ref_idx, :], dtype=np.float64).reshape(2,)
        S_rel = np.asarray(S_abs[idx, :] - s_ref[None, :], dtype=np.float64)
        A_world_use = np.asarray(A_world[idx, :], dtype=np.float64)

        # Recompute from a fixed baseline. Starting from the previous fit makes
        # repeated fits with identical samples accumulate a spurious rotation.
        R_est = np.eye(3, dtype=np.float64)
        for _ in range(max(1, int(max_iter))):
            a_ref_mount = _rotate_altaz_deg(A_world[ref_idx, :], R_est.T)

            A_mount_pred = np.zeros((int(idx.size), 2), dtype=np.float64)
            for i in range(int(idx.size)):
                d_mount = self.mount_delta_for_steps(
                    s_ref,
                    S_abs[int(idx[i]), :],
                    J_model=J,
                    periodic_coeff=periodic_coeff,
                )
                A_mount_pred[i, 0] = _wrap_deg_360(float(a_ref_mount[0]) + float(d_mount[0]))
                A_mount_pred[i, 1] = float(a_ref_mount[1]) + float(d_mount[1])

            src = np.stack(
                [_altaz_deg_to_unit_vec(float(a[0]), float(a[1])) for a in A_mount_pred],
                axis=0,
            )
            dst = np.stack(
                [_altaz_deg_to_unit_vec(float(a[0]), float(a[1])) for a in A_world_use],
                axis=0,
            )
            R_fit = _fit_rotation_kabsch(src, dst)
            if R_fit is None:
                break
            R_fit = _limit_rotation_tilt_ns_oe_deg(
                R_fit,
                max_tilt_deg=float(self.max_tilt_ns_oe_deg),
            )

            d_rot = _rotation_rotvec_deg(R_fit @ R_est.T)
            R_est = R_fit
            if float(np.linalg.norm(d_rot)) < 1e-7:
                break

        return _coerce_rotation_matrix(R_est)

    def _manual_residuals_against_model(
        self,
        *,
        J_model: np.ndarray,
        R_mount_to_world: np.ndarray,
        reference_mask: Optional[np.ndarray] = None,
        periodic_coeff: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        n = int(len(self._manual_steps_abs))
        if n <= 0:
            return {
                "ref_idx": 0,
                "res_az_deg": np.zeros(0, dtype=np.float64),
                "res_alt_deg": np.zeros(0, dtype=np.float64),
                "res_norm_deg": np.zeros(0, dtype=np.float64),
                "threshold_deg": float("nan"),
                "suggested_outlier_mask": np.zeros(0, dtype=bool),
            }

        J = np.asarray(J_model, dtype=np.float64)
        if J.shape != (2, 2) or (not np.all(np.isfinite(J))):
            return None
        R = _coerce_rotation_matrix(R_mount_to_world)

        S_abs = np.stack(self._manual_steps_abs, axis=0).astype(np.float64)
        A_world = np.stack(self._manual_az_alt_abs, axis=0).astype(np.float64)

        idx_all = np.arange(n, dtype=int)
        idx_ref = idx_all
        if reference_mask is not None:
            ref_mask = np.asarray(reference_mask, dtype=bool).reshape(-1)
            if int(ref_mask.size) == n:
                idx_ref_cand = np.flatnonzero(ref_mask)
                if int(idx_ref_cand.size) > 0:
                    idx_ref = idx_ref_cand.astype(int)
        ref_idx = self._manual_reference_index_from_steps(S_abs, idx_ref)
        s_ref = S_abs[ref_idx, :]

        A_mount = np.zeros((n, 2), dtype=np.float64)
        R_inv = R.T
        for i in range(n):
            A_mount[i, :] = _rotate_altaz_deg(A_world[i, :], R_inv)

        a_ref_mount = A_mount[ref_idx, :]
        coeff = self.periodic_coeff_deg if periodic_coeff is None else periodic_coeff
        d_mount = np.stack(
            [
                self.mount_delta_for_steps(
                    s_ref,
                    S_abs[i, :],
                    J_model=J,
                    periodic_coeff=coeff,
                )
                for i in range(n)
            ],
            axis=0,
        )

        A_world_pred = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            az_m = _wrap_deg_360(float(a_ref_mount[0]) + float(d_mount[i, 0]))
            alt_m = float(a_ref_mount[1] + float(d_mount[i, 1]))
            A_world_pred[i, :] = _rotate_altaz_deg(np.array([az_m, alt_m], dtype=np.float64), R)

        res_az = np.array(
            [_wrap_deg_180(float(A_world[i, 0]) - float(A_world_pred[i, 0])) for i in range(n)],
            dtype=np.float64,
        )
        res_alt = (A_world[:, 1] - A_world_pred[:, 1]).astype(np.float64)
        res_norm = np.hypot(res_az, res_alt).astype(np.float64)

        finite = np.isfinite(res_norm)
        if int(np.sum(finite)) <= 0:
            threshold = float("nan")
            out_mask = np.zeros(n, dtype=bool)
        else:
            med = float(np.median(res_norm[finite]))
            mad = float(np.median(np.abs(res_norm[finite] - med)))
            sigma = float(1.4826 * mad)
            floor = float(max(0.0, self.fit_outlier_floor_arcsec) / 3600.0)
            threshold = max(
                floor,
                float(
                    med + float(max(0.0, self.fit_outlier_sigma)) * sigma
                    if np.isfinite(sigma)
                    else floor
                ),
            )
            out_mask = finite & (res_norm > float(threshold))

        return {
            "ref_idx": int(ref_idx),
            "res_az_deg": res_az,
            "res_alt_deg": res_alt,
            "res_norm_deg": res_norm,
            "threshold_deg": float(threshold),
            "suggested_outlier_mask": out_mask.astype(bool),
        }

    def _fit_periodic_transmission_error(
        self,
        *,
        J_model: np.ndarray,
        R_mount_to_world: np.ndarray,
        sample_mask: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Fit one bounded sine/cosine transmission-error cycle per axis."""
        n = len(self._manual_steps_abs)
        mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if n < int(self.min_periodic_model_samples) or mask.size != n:
            return np.zeros((2, 2), dtype=np.float64), 0
        idx = np.flatnonzero(mask)
        if idx.size < int(self.min_periodic_model_samples):
            return np.zeros((2, 2), dtype=np.float64), 0

        S = np.stack(self._manual_steps_abs, axis=0).astype(np.float64)
        A_world = np.stack(self._manual_az_alt_abs, axis=0).astype(np.float64)
        A_mount = np.stack(
            [
                _rotate_altaz_deg(a, _coerce_rotation_matrix(R_mount_to_world).T)
                for a in A_world
            ],
            axis=0,
        )
        ref_idx = self._manual_reference_index_from_steps(S, idx)
        coeff = np.zeros((2, 2), dtype=np.float64)
        fitted_axes = 0
        for axis_idx, axis in enumerate((Axis.AZ, Axis.ALT)):
            period = float(self.kin.transmission_error_period_steps(axis))
            phase = 2.0 * math.pi * S[:, axis_idx] / period
            phase_selected = np.unwrap(phase[idx])
            phase_span = float(np.ptp(phase_selected)) if phase_selected.size else 0.0
            if phase_span < 2.0 * math.pi * float(self.min_periodic_phase_span_frac):
                continue
            phase_ref = float(phase[ref_idx])
            X = np.column_stack(
                (
                    np.sin(phase) - math.sin(phase_ref),
                    np.cos(phase) - math.cos(phase_ref),
                )
            )
            if float(np.linalg.cond(X[idx, :])) > 1e4:
                continue
            if axis_idx == 0:
                observed = np.array(
                    [
                        _wrap_deg_180(float(A_mount[i, 0]) - float(A_mount[ref_idx, 0]))
                        for i in range(n)
                    ],
                    dtype=np.float64,
                )
            else:
                observed = A_mount[:, 1] - float(A_mount[ref_idx, 1])
            linear = (S - S[ref_idx, :]) @ np.asarray(J_model, dtype=np.float64).T
            y = observed - linear[:, axis_idx]
            beta, *_ = np.linalg.lstsq(X[idx, :], y[idx], rcond=None)
            if np.all(np.isfinite(beta)):
                coeff[axis_idx, :] = beta
                fitted_axes += 1
        if fitted_axes <= 0:
            return np.zeros((2, 2), dtype=np.float64), 0
        return self.safe_periodic_coeff_for_prediction(coeff), int(idx.size)

    def manual_samples_deviation_report(self, *, sort_by_deviation: bool = True) -> List[Dict[str, Any]]:
        """Return per-sample residuals against the current model."""
        n = int(len(self._manual_steps_abs))
        if n <= 0:
            return []

        J = np.asarray(self.J_deg_per_step, dtype=np.float64)
        if J.shape != (2, 2) or (not np.all(np.isfinite(J))):
            J = np.array(
                [
                    [float(self.kin.deg_per_step(Axis.AZ)), 0.0],
                    [0.0, float(self.kin.deg_per_step(Axis.ALT))],
                ],
                dtype=np.float64,
            )

        resid = self._manual_residuals_against_model(
            J_model=J,
            R_mount_to_world=self._rotation_mount_to_world(),
            reference_mask=self._manual_fit_inlier_mask,
        )
        if resid is None:
            return []

        S_abs = np.stack(self._manual_steps_abs, axis=0).astype(np.float64)
        A_world = np.stack(self._manual_az_alt_abs, axis=0).astype(np.float64)
        res_az = np.asarray(resid["res_az_deg"], dtype=np.float64)
        res_alt = np.asarray(resid["res_alt_deg"], dtype=np.float64)
        res_norm = np.asarray(resid["res_norm_deg"], dtype=np.float64)
        out_mask = np.asarray(resid["suggested_outlier_mask"], dtype=bool)
        threshold = float(resid["threshold_deg"])
        ref_idx = int(resid["ref_idx"])

        order = np.arange(n, dtype=int)
        if bool(sort_by_deviation):
            order = np.argsort(res_norm)[::-1]

        report: List[Dict[str, Any]] = []
        for rank, i in enumerate(order, start=1):
            report.append(
                {
                    "sample_idx": int(i),
                    "rank_deviation": int(rank),
                    "is_ref_idx": bool(int(i) == ref_idx),
                    "steps_az": float(S_abs[i, 0]),
                    "steps_alt": float(S_abs[i, 1]),
                    "az_deg": float(A_world[i, 0]),
                    "alt_deg": float(A_world[i, 1]),
                    "dev_az_deg": float(res_az[i]),
                    "dev_alt_deg": float(res_alt[i]),
                    "dev_deg": float(res_norm[i]),
                    "dev_arcsec": float(res_norm[i] * 3600.0),
                    "outlier_suggested": bool(out_mask[i]),
                    "threshold_deg": float(threshold),
                    "threshold_arcsec": float(threshold * 3600.0 if np.isfinite(threshold) else float("nan")),
                }
            )
        return report

    def prune_manual_outliers(self, *, min_samples: int = 3, ridge: float = 1e-12) -> Dict[str, Any]:
        """Remove manual samples whose residuals are outliers vs current model."""
        n_before = int(len(self._manual_steps_abs))
        out: Dict[str, Any] = {
            "ok": False,
            "status": "RUNNING",
            "n_before": n_before,
            "n_after": n_before,
            "removed_indices": [],
            "removed_count": 0,
            "fit_before_ok": False,
            "fit_after_ok": False,
            "threshold_arcsec": float("nan"),
        }

        if n_before <= 0:
            out["status"] = "ERR_NO_SAMPLES"
            return out
        if n_before <= int(min_samples):
            out["status"] = "ERR_INSUFFICIENT_MARGIN"
            return out

        # Inspect residuals against the currently accepted model first. A bad
        # new sample must not be allowed to inflate/refit the model before we
        # decide whether it is an outlier.
        report = self.manual_samples_deviation_report(sort_by_deviation=True)
        if not report:
            out["status"] = "ERR_NO_REPORT"
            return out
        out["fit_before_ok"] = bool(self.model_fit_samples >= int(min_samples))

        threshold_arcsec = float(report[0].get("threshold_arcsec", float("nan")))
        out["threshold_arcsec"] = threshold_arcsec

        suggested = [int(r["sample_idx"]) for r in report if bool(r.get("outlier_suggested", False))]
        if not suggested:
            out["status"] = "NO_OUTLIERS"
            return out

        max_remove = max(0, n_before - int(min_samples))
        if max_remove <= 0:
            out["status"] = "ERR_INSUFFICIENT_MARGIN"
            return out

        selected = suggested[:max_remove]
        selected_set = set(int(i) for i in selected)
        remove_desc = sorted(selected_set, reverse=True)

        n_roll = int(len(self._manual_roll_deg_abs))
        roll_aligned = (n_roll == n_before)
        source_aligned = int(len(self._manual_source_abs)) == n_before

        for idx in remove_desc:
            if 0 <= int(idx) < len(self._manual_steps_abs):
                del self._manual_steps_abs[int(idx)]
            if 0 <= int(idx) < len(self._manual_az_alt_abs):
                del self._manual_az_alt_abs[int(idx)]
            if roll_aligned and 0 <= int(idx) < len(self._manual_roll_deg_abs):
                del self._manual_roll_deg_abs[int(idx)]
            if source_aligned and 0 <= int(idx) < len(self._manual_source_abs):
                del self._manual_source_abs[int(idx)]

        # Sample set changed; previous fit mask is no longer aligned.
        self._manual_fit_inlier_mask = None

        if not roll_aligned:
            # Legacy sessions may have compressed roll history without per-sample alignment.
            self._manual_roll_deg_abs.clear()
        if not source_aligned:
            self._manual_source_abs.clear()

        n_after = int(len(self._manual_steps_abs))
        out["n_after"] = n_after
        out["removed_indices"] = [int(i) for i in sorted(selected_set)]
        out["removed_count"] = int(len(selected_set))

        fit_after_ok = False
        if n_after >= int(min_samples):
            fit_after_ok = bool(self.fit_J_from_manual_samples(min_samples=int(min_samples), ridge=float(ridge)))
        out["fit_after_ok"] = fit_after_ok

        out["ok"] = bool(len(selected_set) > 0)
        out["status"] = "OK" if out["ok"] else "NO_OUTLIERS"
        return out

    def fit_J_from_manual_samples(self, *, min_samples: int = 3, ridge: float = 1e-12) -> bool:
        """Fit J from manual samples and estimate global mount->world rotation.

        J captures step-domain anisotropy/coupling (AZ vs ALT microsteps are distinct).
        R_mount_to_world is estimated with Wahba/Kabsch and constrained with hard
        NS/OE tilt limits (|tilt| <= max_tilt_ns_oe_deg).
        """
        previous_inlier_mask = (
            None
            if self._manual_fit_inlier_mask is None
            else np.asarray(self._manual_fit_inlier_mask, dtype=bool).copy()
        )
        previous_fit_samples = int(self.model_fit_samples)
        previous_fit_rms_arcsec = float(self.model_fit_rms_arcsec)
        self.last_fit_reason = "RUNNING"

        if len(self._manual_steps_abs) < int(min_samples):
            self.last_fit_reason = "INSUFFICIENT_SAMPLES"
            self._log_fit_csv(
                fit_kind="manual",
                ok=False,
                reason="INSUFFICIENT_SAMPLES",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(len(self._manual_steps_abs)),
                used_samples=0,
            )
            return False

        S_abs = np.stack(self._manual_steps_abs, axis=0).astype(np.float64)  # (N,2)
        A_abs = np.stack(self._manual_az_alt_abs, axis=0).astype(np.float64)  # (N,2) [az, alt] in deg
        n = int(S_abs.shape[0])
        if n < int(min_samples):
            self.last_fit_reason = "INSUFFICIENT_SAMPLES"
            self._log_fit_csv(
                fit_kind="manual",
                ok=False,
                reason="INSUFFICIENT_SAMPLES",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n),
                used_samples=0,
            )
            return False

        ref_world_before = self.predict_az_alt_deg(from_ref=True) if bool(self.synced) else None

        J_prev = self.constrain_J_to_mechanics(
            np.asarray(self.J_deg_per_step, dtype=np.float64)
        )

        def _solve_with_mask(mask: np.ndarray) -> Optional[Dict[str, Any]]:
            idx = np.flatnonzero(mask)
            n_use = int(idx.size)
            if n_use < int(min_samples):
                return None

            best_fit: Optional[Dict[str, Any]] = None
            best_obj = float("inf")

            # Try all masked samples as potential references.
            # This avoids anchoring the model to a high-leverage outlier.
            for ref_idx in idx.tolist():
                s_ref = S_abs[int(ref_idx), :]
                a_ref = A_abs[int(ref_idx), :]
                S_rel = S_abs - s_ref  # (N,2)
                d_az = np.array(
                    [_wrap_deg_180(float(A_abs[i, 0]) - float(a_ref[0])) for i in range(n)],
                    dtype=np.float64,
                )
                d_alt = (A_abs[:, 1] - float(a_ref[1])).astype(np.float64)
                D = np.column_stack((d_az, d_alt)).astype(np.float64)  # (N,2)

                S_use = S_rel[idx, :]
                D_use = D[idx, :]

                # Solve only for axes with meaningful physical travel. Tiny
                # moves otherwise let plate-solve noise masquerade as a new
                # gearbox scale.
                mechanical = self.mechanical_J()
                col_span_steps = np.ptp(S_use, axis=0)
                col_span_deg = np.array(
                    [
                        abs(float(mechanical[0, 0])) * float(col_span_steps[0]),
                        abs(float(mechanical[1, 1])) * float(col_span_steps[1]),
                    ],
                    dtype=np.float64,
                )
                min_span_deg = max(0.0, float(self.min_fit_axis_span_deg))
                active_cols = [
                    j for j in range(2)
                    if np.isfinite(col_span_deg[j])
                    and float(col_span_deg[j]) >= min_span_deg
                ]
                if not active_cols:
                    continue
                S_active = S_use[:, active_cols]
                p = int(S_active.shape[1])
                if n_use < max(int(min_samples), p + 1):
                    continue

                lam = max(0.0, float(ridge))
                B_active = np.zeros((p, 2), dtype=np.float64)
                periodic_seed = np.zeros((2, 2), dtype=np.float64)
                std_by_output: List[np.ndarray] = []

                # Fit the long-term scale/coupling and the bounded cycloidal
                # term in the same regression. Fitting J first would let a
                # partial transmission-error cycle masquerade as a different
                # gearbox ratio, which is precisely what the fixed 45:1
                # mechanical baseline must prevent.
                solve_ok = True
                for output_idx, axis in enumerate((Axis.AZ, Axis.ALT)):
                    columns: List[np.ndarray] = [
                        np.asarray(S_active[:, j], dtype=np.float64)
                        for j in range(p)
                    ]
                    periodic_enabled = False
                    if (
                        output_idx in active_cols
                        and n_use >= max(int(self.min_periodic_model_samples), p + 3)
                    ):
                        period = float(self.kin.transmission_error_period_steps(axis))
                        phase = 2.0 * math.pi * S_abs[:, output_idx] / period
                        phase_span = float(np.ptp(phase[idx])) if idx.size else 0.0
                        phase_ref = float(phase[int(ref_idx)])
                        periodic_columns = (
                            np.sin(phase) - math.sin(phase_ref),
                            np.cos(phase) - math.cos(phase_ref),
                        )
                        periodic_design = np.column_stack(periodic_columns)[idx, :]
                        if (
                            phase_span
                            >= 2.0 * math.pi * float(self.min_periodic_phase_span_frac)
                            and np.all(np.isfinite(periodic_design))
                            and float(np.linalg.cond(periodic_design)) <= 1e4
                        ):
                            columns.extend(
                                [periodic_design[:, 0], periodic_design[:, 1]]
                            )
                            periodic_enabled = True

                    X = np.column_stack(columns).astype(np.float64)
                    if n_use < int(X.shape[1]) + 1 or not np.all(np.isfinite(X)):
                        solve_ok = False
                        break

                    # Steps and sine/cosine columns differ by several orders
                    # of magnitude. Column normalization keeps least-squares
                    # conditioning stable without changing the fitted units.
                    scale = np.sqrt(np.mean(np.square(X), axis=0))
                    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-15):
                        solve_ok = False
                        break
                    Xn = X / scale[None, :]
                    q = int(Xn.shape[1])
                    if lam > 0.0:
                        X_aug = np.vstack((Xn, math.sqrt(lam) * np.eye(q)))
                        y_aug = np.concatenate(
                            (D_use[:, output_idx], np.zeros(q, dtype=np.float64))
                        )
                    else:
                        X_aug = Xn
                        y_aug = D_use[:, output_idx]
                    beta_n, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
                    beta = beta_n / scale
                    if not np.all(np.isfinite(beta)):
                        solve_ok = False
                        break

                    B_active[:, output_idx] = beta[:p]
                    if periodic_enabled:
                        periodic_seed[output_idx, :] = beta[p : p + 2]

                    pred_output = X @ beta
                    output_residual = D_use[:, output_idx] - pred_output
                    dof_output = max(1, n_use - q)
                    sigma2 = float(
                        np.dot(output_residual, output_residual) / float(dof_output)
                    )
                    XtXn = (Xn.T @ Xn) + (lam * np.eye(q))
                    try:
                        cov_n = np.linalg.inv(XtXn)
                    except np.linalg.LinAlgError:
                        cov_n = np.linalg.pinv(XtXn)
                    std_beta = (
                        np.sqrt(np.maximum(np.diag(cov_n) * sigma2, 0.0)) / scale
                    )
                    std_by_output.append(std_beta[:p])

                if not solve_ok:
                    continue
                periodic_seed = self.safe_periodic_coeff_for_prediction(periodic_seed)

                J_unconstrained = J_prev.copy()
                for ridx, cidx in enumerate(active_cols):
                    J_unconstrained[0, cidx] = float(B_active[ridx, 0])
                    J_unconstrained[1, cidx] = float(B_active[ridx, 1])

                J_new = self.constrain_J_to_mechanics(J_unconstrained)

                # If a non-excited column was already invalid/near-zero, restore mechanical baseline.
                if 0 not in active_cols and float(np.linalg.norm(J_new[:, 0])) < 1e-12:
                    J_new[:, 0] = np.array([float(self.kin.deg_per_step(Axis.AZ)), 0.0], dtype=np.float64)
                if 1 not in active_cols and float(np.linalg.norm(J_new[:, 1])) < 1e-12:
                    J_new[:, 1] = np.array([0.0, float(self.kin.deg_per_step(Axis.ALT))], dtype=np.float64)

                if not np.all(np.isfinite(J_new)):
                    continue
                det_J = float(np.linalg.det(J_new))
                if (not np.isfinite(det_J)) or abs(det_J) < 1e-12:
                    continue
                try:
                    cond_J = float(np.linalg.cond(J_new))
                except np.linalg.LinAlgError:
                    continue
                if (not np.isfinite(cond_J)) or cond_J > 1e10:
                    continue

                periodic_delta = np.stack(
                    [
                        self._periodic_offset_deg(S_abs[i, :], coeff=periodic_seed)
                        - self._periodic_offset_deg(s_ref, coeff=periodic_seed)
                        for i in range(n)
                    ],
                    axis=0,
                )
                pred_use = (S_use @ J_new.T) + periodic_delta[idx, :]
                res_az_use = np.array(
                    [_wrap_deg_180(float(D_use[i, 0]) - float(pred_use[i, 0])) for i in range(n_use)],
                    dtype=np.float64,
                )
                res_alt_use = (D_use[:, 1] - pred_use[:, 1]).astype(np.float64)
                std_beta_az_active = std_by_output[0]
                std_beta_alt_active = std_by_output[1]

                pred_all = (S_rel @ J_new.T) + periodic_delta
                res_az_all = np.array(
                    [_wrap_deg_180(float(D[i, 0]) - float(pred_all[i, 0])) for i in range(n)],
                    dtype=np.float64,
                )
                res_alt_all = (D[:, 1] - pred_all[:, 1]).astype(np.float64)

                res_norm_use = np.hypot(res_az_use, res_alt_use)
                finite_use = np.isfinite(res_norm_use)
                if int(np.sum(finite_use)) <= 0:
                    continue
                obj = float(np.mean(np.square(res_norm_use[finite_use])))
                if (not np.isfinite(obj)) or obj >= best_obj:
                    continue

                best_obj = obj
                best_fit = {
                    "mask": mask.copy(),
                    "n_use": n_use,
                    "p": p,
                    "ref_idx": int(ref_idx),
                    "active_cols": active_cols,
                    "J_new": J_new,
                    "J_unconstrained": J_unconstrained,
                    "periodic_coeff": periodic_seed,
                    "axis_span_deg": col_span_deg,
                    "res_az_use": res_az_use,
                    "res_alt_use": res_alt_use,
                    "res_az_all": res_az_all,
                    "res_alt_all": res_alt_all,
                    "std_beta_az_active": std_beta_az_active,
                    "std_beta_alt_active": std_beta_alt_active,
                }

            return best_fit

        mask = np.ones(n, dtype=bool)
        # When a clean model already exists, use it as an independent guard
        # against a newly added false solve/backlash-corrupted sample. This is
        # evaluated before the candidate fit can move toward that sample.
        if previous_fit_samples >= max(int(min_samples), 5):
            prior_resid = self._manual_residuals_against_model(
                J_model=J_prev,
                R_mount_to_world=self._rotation_mount_to_world(),
                reference_mask=previous_inlier_mask,
            )
            if prior_resid is not None:
                suggested = np.asarray(
                    prior_resid["suggested_outlier_mask"], dtype=bool
                ).reshape(-1)
                prior_norm = np.asarray(prior_resid["res_norm_deg"], dtype=np.float64)
                guarded = np.isfinite(prior_norm) & ~suggested
                if guarded.size == n and int(np.sum(guarded)) >= int(min_samples):
                    mask = guarded
        fit = _solve_with_mask(mask)
        if fit is None:
            self.last_fit_reason = "DEGENERATE_MODEL"
            self._log_fit_csv(
                fit_kind="manual",
                ok=False,
                reason="DEGENERATE_MODEL",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n),
                used_samples=0,
            )
            return False

        # Robust outlier rejection on total residual norm using MAD scale.
        min_keep = max(int(min_samples), 3)
        if n >= max(min_keep + 2, 5):
            for _ in range(3):
                res_norm = np.hypot(fit["res_az_all"], fit["res_alt_all"])
                finite = np.isfinite(res_norm)
                need = max(min_keep, int(fit["p"]) + 1)
                if int(np.sum(finite)) < need:
                    break

                med = float(np.median(res_norm[finite]))
                mad = float(np.median(np.abs(res_norm[finite] - med)))
                sigma = float(1.4826 * mad)
                floor = float(max(0.0, self.fit_outlier_floor_arcsec) / 3600.0)
                thr = med + float(max(0.0, self.fit_outlier_sigma)) * sigma
                thr = max(floor, float(thr if np.isfinite(thr) else 0.0))

                new_mask = finite & (res_norm <= thr)

                if int(np.sum(new_mask)) < need:
                    idx_f = np.flatnonzero(finite)
                    order = idx_f[np.argsort(res_norm[idx_f])]
                    keep = order[:need]
                    new_mask = np.zeros_like(mask, dtype=bool)
                    new_mask[keep] = True

                if np.array_equal(new_mask, mask):
                    # Fallback: if one sample is clearly separated, prune it.
                    idx_f = np.flatnonzero(finite & mask)
                    if int(np.sum(mask)) <= need or int(idx_f.size) == 0:
                        break
                    med_ref = max(float(med), floor)
                    thr_fallback = max(3.0 * floor, 2.0 * med_ref)
                    order = idx_f[np.argsort(res_norm[idx_f])[::-1]]
                    accepted = False
                    for cand in order:
                        cand_res = float(res_norm[int(cand)])
                        if (not np.isfinite(cand_res)) or cand_res <= thr_fallback:
                            break
                        trial_mask = mask.copy()
                        trial_mask[int(cand)] = False
                        fit_trial = _solve_with_mask(trial_mask)
                        if fit_trial is None:
                            continue
                        mask = trial_mask
                        fit = fit_trial
                        accepted = True
                        break
                    if not accepted:
                        break
                    continue

                fit_new = _solve_with_mask(new_mask)
                if fit_new is None:
                    break
                mask = new_mask
                fit = fit_new

        # If robust-MAD kept all samples, try a leave-one-out fallback and accept
        # only if it yields a clearly better fit. This catches high-leverage outliers.
        if int(fit["n_use"]) == n and n > int(min_keep):
            res_norm_all = np.hypot(fit["res_az_all"], fit["res_alt_all"]).astype(np.float64)
            base_use = np.hypot(fit["res_az_use"], fit["res_alt_use"]).astype(np.float64)
            base_rms = float(np.sqrt(np.mean(np.square(base_use)))) if int(base_use.size) > 0 else float("inf")
            med_all = (
                float(np.median(res_norm_all[np.isfinite(res_norm_all)]))
                if int(np.sum(np.isfinite(res_norm_all))) > 0
                else float("inf")
            )
            floor = float(max(0.0, self.fit_outlier_floor_arcsec) / 3600.0)

            best: Optional[Tuple[int, Dict[str, Any], float]] = None
            for cand in range(n):
                trial_mask = np.ones(n, dtype=bool)
                trial_mask[cand] = False
                fit_trial = _solve_with_mask(trial_mask)
                if fit_trial is None:
                    continue
                trial_use = np.hypot(fit_trial["res_az_use"], fit_trial["res_alt_use"]).astype(np.float64)
                if int(trial_use.size) == 0:
                    continue
                trial_rms = float(np.sqrt(np.mean(np.square(trial_use))))
                if (best is None) or (trial_rms < float(best[2])):
                    best = (cand, fit_trial, trial_rms)

            if best is not None and np.isfinite(base_rms):
                cand_idx, fit_trial, trial_rms = best
                cand_res = (
                    float(res_norm_all[int(cand_idx)])
                    if int(cand_idx) < int(res_norm_all.size)
                    else 0.0
                )
                med_ref = max(float(med_all), floor)
                if (
                    np.isfinite(cand_res)
                    and trial_rms < 0.95 * base_rms
                    and cand_res > max(3.0 * floor, 2.0 * med_ref)
                ):
                    fit = fit_trial

        if not self.is_J_within_mechanical_limits(fit["J_unconstrained"]):
            self.last_fit_reason = "MODEL_OUTSIDE_MECHANICAL_LIMITS"
            self._log_fit_csv(
                fit_kind="manual",
                ok=False,
                reason="MODEL_OUTSIDE_MECHANICAL_LIMITS",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n),
                used_samples=int(fit["n_use"]),
            )
            return False

        J_candidate = np.asarray(fit["J_new"], dtype=np.float64).copy()
        periodic_seed = self.safe_periodic_coeff_for_prediction(
            fit.get("periodic_coeff", np.zeros((2, 2), dtype=np.float64))
        )
        R_candidate = self._fit_rotation_from_manual_samples(
            S_abs=S_abs,
            A_world=A_abs,
            sample_mask=np.asarray(fit["mask"], dtype=bool),
            J_model=J_candidate,
            periodic_coeff=periodic_seed,
        )
        R_candidate = _limit_rotation_tilt_ns_oe_deg(
            _coerce_rotation_matrix(R_candidate),
            max_tilt_deg=float(self.max_tilt_ns_oe_deg),
        )

        periodic_candidate, periodic_samples = self._fit_periodic_transmission_error(
            J_model=J_candidate,
            R_mount_to_world=R_candidate,
            sample_mask=np.asarray(fit["mask"], dtype=bool),
        )
        R_candidate = self._fit_rotation_from_manual_samples(
            S_abs=S_abs,
            A_world=A_abs,
            sample_mask=np.asarray(fit["mask"], dtype=bool),
            J_model=J_candidate,
            periodic_coeff=periodic_candidate,
        )
        R_candidate = _limit_rotation_tilt_ns_oe_deg(
            _coerce_rotation_matrix(R_candidate),
            max_tilt_deg=float(self.max_tilt_ns_oe_deg),
        )

        res_az = np.asarray(fit["res_az_use"], dtype=np.float64)
        res_alt = np.asarray(fit["res_alt_use"], dtype=np.float64)
        resid_model = self._manual_residuals_against_model(
            J_model=J_candidate,
            R_mount_to_world=R_candidate,
            reference_mask=np.asarray(fit["mask"], dtype=bool),
            periodic_coeff=periodic_candidate,
        )
        if resid_model is not None:
            idx_use = np.flatnonzero(np.asarray(fit["mask"], dtype=bool))
            res_az_all = np.asarray(resid_model["res_az_deg"], dtype=np.float64)
            res_alt_all = np.asarray(resid_model["res_alt_deg"], dtype=np.float64)
            if int(idx_use.size) > 0 and int(res_az_all.size) >= int(np.max(idx_use) + 1):
                res_az = res_az_all[idx_use]
                res_alt = res_alt_all[idx_use]

        fit_rms_arcsec = float(
            np.sqrt(np.mean(np.square(res_az) + np.square(res_alt))) * 3600.0
        )
        max_rms = max(0.0, float(self.max_model_fit_rms_arcsec))
        if (
            not np.isfinite(fit_rms_arcsec)
            or (max_rms > 0.0 and fit_rms_arcsec > max_rms)
        ):
            self.last_fit_reason = "FIT_RMS_TOO_HIGH"
            self._log_fit_csv(
                fit_kind="manual",
                ok=False,
                reason="FIT_RMS_TOO_HIGH",
                min_samples=int(min_samples),
                ridge=float(ridge),
                total_samples=int(n),
                used_samples=int(fit["n_use"]),
            )
            return False

        if (
            previous_fit_samples >= max(int(min_samples), 5)
            and np.isfinite(previous_fit_rms_arcsec)
            and previous_fit_rms_arcsec > 0.0
        ):
            regression_limit = max(
                previous_fit_rms_arcsec + 10.0,
                previous_fit_rms_arcsec * 1.5,
            )
            if fit_rms_arcsec > regression_limit:
                self._manual_fit_inlier_mask = previous_inlier_mask
                self.last_fit_reason = "FIT_REGRESSION"
                self._log_fit_csv(
                    fit_kind="manual",
                    ok=False,
                    reason="FIT_REGRESSION",
                    min_samples=int(min_samples),
                    ridge=float(ridge),
                    total_samples=int(n),
                    used_samples=int(fit["n_use"]),
                )
                return False

        self.J_deg_per_step = J_candidate
        self.R_mount_to_world = R_candidate
        self.periodic_coeff_deg = periodic_candidate
        self.periodic_model_samples = int(periodic_samples)
        if ref_world_before is not None and np.all(np.isfinite(ref_world_before)):
            self.ref_az_alt_deg = self._world_to_mount_altaz(ref_world_before)

        self.model_fit_samples = int(fit["n_use"])
        self._manual_fit_inlier_mask = np.asarray(fit["mask"], dtype=bool).copy()
        self.model_fit_rms_az_deg = float(np.sqrt(np.mean(np.square(res_az))))
        self.model_fit_rms_alt_deg = float(np.sqrt(np.mean(np.square(res_alt))))
        self.model_fit_rms_arcsec = fit_rms_arcsec

        n_out = int(n - int(fit["n_use"]))
        if n_out > 0:
            log_info(
                None,
                f"GoTo: manual fit rejected outliers={n_out}/{n}",
                throttle_s=0.2,
                throttle_key="goto_fit_manual_outliers",
            )

        rv_deg = _rotation_rotvec_deg(self.R_mount_to_world)
        # Report NS/OE tilts from ENU rotvec x/y components.
        self.model_pitch_deg = float(rv_deg[0])
        self.model_yaw_deg = float(rv_deg[1])
        self.model_yaw_err_deg = 0.0
        self.model_pitch_err_deg = 0.0

        # Start with conservative defaults for non-fitted columns.
        self.J00_err = 0.0
        self.J01_err = 0.0
        self.J10_err = 0.0
        self.J11_err = 0.0
        active_cols = list(fit["active_cols"])
        std_beta_az_active = np.asarray(fit["std_beta_az_active"], dtype=np.float64)
        std_beta_alt_active = np.asarray(fit["std_beta_alt_active"], dtype=np.float64)
        for ridx, cidx in enumerate(active_cols):
            if cidx == 0:
                self.J00_err = float(std_beta_az_active[ridx])
                self.J10_err = float(std_beta_alt_active[ridx])
            else:
                self.J01_err = float(std_beta_az_active[ridx])
                self.J11_err = float(std_beta_alt_active[ridx])

        self.model_non_orthogonality_deg = _non_orthogonality_deg_from_J(self.J_deg_per_step)
        self.model_non_orthogonality_err_deg = 0.0
        j_params = np.array(
            [
                float(self.J_deg_per_step[0, 0]),
                float(self.J_deg_per_step[0, 1]),
                float(self.J_deg_per_step[1, 0]),
                float(self.J_deg_per_step[1, 1]),
            ],
            dtype=np.float64,
        )
        j_vars = np.array(
            [
                float(self.J00_err * self.J00_err),
                float(self.J01_err * self.J01_err),
                float(self.J10_err * self.J10_err),
                float(self.J11_err * self.J11_err),
            ],
            dtype=np.float64,
        )
        if np.all(np.isfinite(j_params)) and np.all(np.isfinite(j_vars)):
            grad = np.zeros(4, dtype=np.float64)
            for i in range(4):
                eps = max(1e-12, 1e-6 * abs(float(j_params[i])))
                p_hi = j_params.copy()
                p_lo = j_params.copy()
                p_hi[i] += eps
                p_lo[i] -= eps
                f_hi = _non_orthogonality_deg_from_params(*p_hi.tolist())
                f_lo = _non_orthogonality_deg_from_params(*p_lo.tolist())
                if np.isfinite(f_hi) and np.isfinite(f_lo):
                    grad[i] = float((f_hi - f_lo) / (2.0 * eps))
            v_nonorth = float(np.sum(np.square(grad) * j_vars))
            if np.isfinite(v_nonorth) and v_nonorth >= 0.0:
                self.model_non_orthogonality_err_deg = float(math.sqrt(v_nonorth))

        th_all = np.asarray(self._manual_roll_deg_abs, dtype=np.float64).reshape(-1)
        roll_mask = np.asarray(fit["mask"], dtype=bool).reshape(-1)
        if roll_mask.size == th_all.size:
            th = th_all[roll_mask]
        else:
            th = th_all
        th = th[np.isfinite(th)]
        self.model_roll_samples = int(th.size)
        self.model_roll_deg = 0.0
        self.model_roll_err_deg = 0.0
        if th.size >= 1:
            self.model_roll_deg = _wrap_deg_180(_circular_mean_deg(th))
            if th.size >= 2:
                roll_std = _circular_std_deg(th)
                self.model_roll_err_deg = float(roll_std / math.sqrt(float(th.size)))
            else:
                self.model_roll_err_deg = 0.0

        self._log_fit_csv(
            fit_kind="manual",
            ok=True,
            reason="OK",
            min_samples=int(min_samples),
            ridge=float(ridge),
            total_samples=int(n),
            used_samples=int(fit["n_use"]),
        )
        self.last_fit_reason = "OK"
        return True

    def model_fit_report(self) -> Dict[str, Any]:
        return {
            "J00_err": float(self.J00_err),
            "J01_err": float(self.J01_err),
            "J10_err": float(self.J10_err),
            "J11_err": float(self.J11_err),
            "model_non_orthogonality_deg": float(self.model_non_orthogonality_deg),
            "model_non_orthogonality_err_deg": float(self.model_non_orthogonality_err_deg),
            "model_roll_deg": float(self.model_roll_deg),
            "model_roll_err_deg": float(self.model_roll_err_deg),
            "model_roll_samples": int(self.model_roll_samples),
            "model_pitch_deg": float(self.model_pitch_deg),
            "model_pitch_err_deg": float(self.model_pitch_err_deg),
            "model_yaw_deg": float(self.model_yaw_deg),
            "model_yaw_err_deg": float(self.model_yaw_err_deg),
            "periodic_az_sin_deg": float(self.periodic_coeff_deg[0, 0]),
            "periodic_az_cos_deg": float(self.periodic_coeff_deg[0, 1]),
            "periodic_alt_sin_deg": float(self.periodic_coeff_deg[1, 0]),
            "periodic_alt_cos_deg": float(self.periodic_coeff_deg[1, 1]),
            "periodic_model_samples": int(self.periodic_model_samples),
            "periodic_error_az_deg": float(np.linalg.norm(self.periodic_coeff_deg[0, :])),
            "periodic_error_alt_deg": float(np.linalg.norm(self.periodic_coeff_deg[1, :])),
            "last_direction_az": int(self.last_move_direction_az),
            "last_direction_alt": int(self.last_move_direction_alt),
            "backlash_steps_az": int(self.backlash_steps_az),
            "backlash_steps_alt": int(self.backlash_steps_alt),
            "model_fit_samples": int(self.model_fit_samples),
            "model_fit_rms_az_deg": float(self.model_fit_rms_az_deg),
            "model_fit_rms_alt_deg": float(self.model_fit_rms_alt_deg),
            "model_fit_rms_arcsec": float(self.model_fit_rms_arcsec),
            "last_fit_reason": str(self.last_fit_reason),
        }


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

    # Model-only GoTo parameters
    max_iters: int = 1
    gain: float = 1.0
    max_step_per_iter: int = 0
    stages: int = 1
    platesolving_feedback: bool = False

    # MOVE speed (blocking). delay_us ~ 1e6 / microsteps_per_s.
    slew_delay_us_az: int = 1200
    slew_delay_us_alt: int = 1200
    # Adaptive slew safety. Smaller delays are faster; the loaded firmware
    # treats this floor as the maximum speed and ramps around it.
    slew_min_delay_us: int = 400
    slew_full_speed_distance_deg: float = 20.0
    max_unfitted_goto_deg: float = 3.0
    max_goto_distance_deg: float = 10.0

    settle_s: float = 0.25

    # Extra physical pulses used only to take up drivetrain slack after a
    # direction reversal. They are deliberately excluded from ``steps_est``:
    # the measured mount position does not change while backlash is consumed.
    backlash_steps_az: int = 0
    backlash_steps_alt: int = 10

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


def _observer_without_refraction(observer: ObserverConfig) -> ObserverConfig:
    try:
        return replace(observer, refraction_enable=False)
    except Exception:
        return observer


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
    observer_no_refract = _observer_without_refraction(observer)
    return parse_target_to_icrs(
        target,
        observer=observer_no_refract,
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
    alt_true = float(altaz.alt.deg)
    # GoTo model uses true AltAz (no atmospheric refraction correction).
    return np.array([az, alt_true], dtype=np.float64)


def platesolving_center_to_altaz_deg(
    ra_deg: float,
    dec_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    return icrs_to_altaz_deg(c, observer=observer, obstime=obstime)


def _platesolving_result_obstime(
    sol: PlatesolvingResult,
    *,
    fallback: Optional[Time] = None,
) -> Time:
    obstime_unix = float(getattr(sol, "obstime_unix", float("nan")))
    if np.isfinite(obstime_unix) and obstime_unix > 0.0:
        return Time(obstime_unix, format="unix", scale="utc")
    return fallback if fallback is not None else _now_time()


def platesolving_roll_sample_deg(
    sol: PlatesolvingResult,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> float:
    """Camera +AZ-axis roll represented by a plate-solving solution."""
    theta = float(getattr(sol, "theta_deg", float("nan")))
    ra = float(getattr(sol, "center_ra_deg", float("nan")))
    dec = float(getattr(sol, "center_dec_deg", float("nan")))
    if (not np.isfinite(theta)) or (not np.isfinite(ra)) or (not np.isfinite(dec)):
        return float("nan")
    t_eval = obstime if obstime is not None else _platesolving_result_obstime(sol)
    az_axis_theta = expected_field_rotation_deg(
        ra,
        dec,
        observer=observer,
        obstime=t_eval,
        roll_offset_deg=0.0,
    )
    if az_axis_theta is None or (not np.isfinite(az_axis_theta)):
        return float("nan")
    return _wrap_deg_180(float(az_axis_theta) - float(theta))


# ============================================================
# GoTo controller
# ============================================================

MoveStepsFn = Callable[[Axis, int, int, int], Any]
RateMountFn = Callable[[float, float], Any]
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
        self.model.backlash_steps_az = max(0, int(self.cfg.backlash_steps_az))
        self.model.backlash_steps_alt = max(0, int(self.cfg.backlash_steps_alt))

    # -------------------------
    # Sync
    # -------------------------

    def sync_from_platesolving(self, sol: PlatesolvingResult, *, obstime: Optional[Time] = None) -> bool:
        """Set the mount's absolute AZ/ALT reference using a plate-solve."""
        if not bool(getattr(sol, "success", False)):
            return False
        solve_obstime = _platesolving_result_obstime(sol, fallback=obstime)

        az_alt = platesolving_center_to_altaz_deg(
            float(sol.center_ra_deg),
            float(sol.center_dec_deg),
            observer=self.cfg.observer,
            obstime=solve_obstime,
        )

        return bool(self.model.sync_from_world_az_alt(az_alt))

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
        diagnostics: Optional[DiagnosticSession] = None,
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

        raw16 = ensure_raw16_bayer(frame).copy()
        if diagnostics is not None:
            diagnostics.save_raw(
                "calibration_platesolve_input",
                raw16,
                metadata={
                    "target": target_for_solver,
                    "obstime_unix": float(obstime.unix),
                    "radius_deg_seq": radius_deg_seq,
                },
            )

        def _progress(stage: str, payload: Dict[str, Any]) -> None:
            if diagnostics is not None:
                diagnostics.record(str(stage), **dict(payload or {}))

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
                raw16,
                target=target_for_solver,
                cfg=cfg2,
                sep_cfg=self.cfg.sep,
                observer=_observer_without_refraction(self.cfg.observer),
                obstime=obstime,
                progress_cb=_progress,
            )
            res = replace(res, obstime_unix=float(obstime.unix))
            if diagnostics is not None:
                diagnostics.record(
                    "calibration_platesolve_attempt",
                    search_radius_deg=rad,
                    result=res,
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
        rate_mount: Optional[RateMountFn] = None,
        stop: Optional[StopFn] = None,
        tracking_pause: Optional[Callable[[bool], Any]] = None,
        tracking_keyframe_reset: Optional[Callable[[], Any]] = None,
        stages: int = 1,
        platesolving_feedback: bool = False,
        obstime: Optional[Time] = None,
        diagnostics: Optional[DiagnosticSession] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> GoToStatus:
        """Execute one model-predicted GoTo MOVE without plate-solving feedback."""
        st = GoToStatus(ok=False, status="RUNNING")
        # Retained for API compatibility with callers shared with calibration.
        _ = get_live_frame, platesolving_cfg, rate_mount, stages, platesolving_feedback

        if not self.model.synced:
            st.status = "ERR_NOT_SYNCED"
            if diagnostics is not None:
                diagnostics.record("goto_rejected", reason=st.status)
            return st

        if cancel_requested is not None and bool(cancel_requested()):
            st.status = "CANCELLED"
            if diagnostics is not None:
                diagnostics.record("goto_rejected", reason=st.status)
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
            if obstime is None:
                obstime = _now_time()

            try:
                target_icrs = resolve_target_icrs(
                    target,
                    observer=self.cfg.observer,
                    obstime=obstime,
                )
            except Exception as exc:
                log_error(None, f"GoTo: failed to resolve target {target!r}", exc)
                st.status = "ERR_TARGET_RESOLVE"
                if diagnostics is not None:
                    diagnostics.record("goto_rejected", reason=st.status, error=repr(exc))
                return st

            # Check visibility / safe altitude.
            altaz_now = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=obstime)
            if diagnostics is not None:
                diagnostics.record(
                    "goto_target_resolved",
                    target=target,
                    target_icrs={
                        "ra_deg": float(target_icrs.ra.deg),
                        "dec_deg": float(target_icrs.dec.deg),
                    },
                    target_altaz_command_time=altaz_now,
                    command_obstime_unix=float(obstime.unix),
                )
            if not (self.cfg.alt_min_deg <= float(altaz_now[1]) <= self.cfg.alt_max_deg):
                st.status = "ERR_TARGET_OUT_OF_RANGE"
                if diagnostics is not None:
                    diagnostics.record("goto_rejected", reason=st.status, target_altaz=altaz_now)
                return st

            altaz_cur = self.model.predict_az_alt_deg()
            if altaz_cur is None:
                st.status = "ERR_NO_CURRENT"
                if diagnostics is not None:
                    diagnostics.record("goto_rejected", reason=st.status)
                return st

            initial_daz = _wrap_deg_180(float(altaz_now[0]) - float(altaz_cur[0]))
            initial_dalt = float(altaz_now[1]) - float(altaz_cur[1])
            initial_distance_deg = float(math.hypot(initial_daz, initial_dalt))
            max_distance_deg = max(0.0, float(self.cfg.max_goto_distance_deg))
            if max_distance_deg > 0.0 and initial_distance_deg > max_distance_deg:
                st.status = "ERR_GOTO_DISTANCE_LIMIT"
                if diagnostics is not None:
                    diagnostics.record(
                        "goto_rejected",
                        reason=st.status,
                        distance_deg=initial_distance_deg,
                        limit_deg=max_distance_deg,
                    )
                return st

            max_unfitted_deg = max(0.0, float(self.cfg.max_unfitted_goto_deg))
            if (
                int(getattr(self.model, "model_fit_samples", 0)) < 3
                and max_unfitted_deg > 0.0
                and initial_distance_deg > max_unfitted_deg
            ):
                st.status = "ERR_MODEL_NOT_FITTED_FOR_DISTANCE"
                if diagnostics is not None:
                    diagnostics.record(
                        "goto_rejected",
                        reason=st.status,
                        distance_deg=initial_distance_deg,
                        limit_deg=max_unfitted_deg,
                        model_fit_samples=int(getattr(self.model, "model_fit_samples", 0)),
                    )
                return st

            J_motion = np.asarray(self.model.J_deg_per_step, dtype=np.float64)
            if not self.model.is_J_within_mechanical_limits(J_motion):
                log_error(
                    None,
                    "GoTo: refusing MOVE with model outside 45:1 mechanical limits",
                    None,
                    throttle_s=5.0,
                    throttle_key="goto_unsafe_model",
                )
                st.status = "ERR_UNSAFE_MODEL"
                if diagnostics is not None:
                    diagnostics.record(
                        "goto_rejected",
                        reason=st.status,
                        J_deg_per_step=J_motion,
                        mechanical_J=self.model.mechanical_J(),
                    )
                return st
            J_motion = self.model.constrain_J_to_mechanics(J_motion)
            try:
                invJ = np.linalg.inv(J_motion)
            except np.linalg.LinAlgError as exc:
                log_error(
                    None,
                    "GoTo: singular J matrix during solve",
                    exc,
                    throttle_s=5.0,
                    throttle_key="goto_invJ",
                )
                st.status = "ERR_SINGULAR_MODEL"
                if diagnostics is not None:
                    diagnostics.record("goto_rejected", reason=st.status, error=repr(exc))
                return st

            command_time = _now_time()
            arrival_time = command_time
            dsteps = np.zeros(2, dtype=np.float64)
            delay_us_az = int(self.cfg.slew_delay_us_az)
            delay_us_alt = int(self.cfg.slew_delay_us_alt)
            daz = 0.0
            dalt = 0.0

            # Fixed-point estimate: target Alt/Az changes while a long slew runs.
            for _plan_iter in range(4):
                if cancel_requested is not None and bool(cancel_requested()):
                    st.status = "CANCELLED"
                    if diagnostics is not None:
                        diagnostics.record("goto_rejected", reason=st.status)
                    return st
                altaz_tgt = icrs_to_altaz_deg(
                    target_icrs,
                    observer=self.cfg.observer,
                    obstime=arrival_time,
                )
                if not (
                    self.cfg.alt_min_deg
                    <= float(altaz_tgt[1])
                    <= self.cfg.alt_max_deg
                ):
                    st.status = "ERR_TARGET_OUT_OF_RANGE"
                    if diagnostics is not None:
                        diagnostics.record(
                            "goto_rejected",
                            reason=st.status,
                            plan_iteration=int(_plan_iter + 1),
                            target_altaz=altaz_tgt,
                        )
                    return st

                daz = _wrap_deg_180(float(altaz_tgt[0]) - float(altaz_cur[0]))
                dalt = float(altaz_tgt[1]) - float(altaz_cur[1])
                altaz_tgt_mount = self.model._world_to_mount_altaz(altaz_tgt)
                altaz_cur_mount = self.model._world_to_mount_altaz(altaz_cur)
                d_altaz_vec = np.array(
                    [
                        _wrap_deg_180(float(altaz_tgt_mount[0]) - float(altaz_cur_mount[0])),
                        float(altaz_tgt_mount[1]) - float(altaz_cur_mount[1]),
                    ],
                    dtype=np.float64,
                )
                dsteps = self.model.solve_step_delta_for_mount_delta(
                    d_altaz_vec,
                    steps_from=self.model.steps_est,
                )
                dsteps *= float(self.cfg.gain)

                max_step_per_iter = int(self.cfg.max_step_per_iter)
                if max_step_per_iter > 0:
                    dsteps = np.clip(
                        dsteps,
                        -float(max_step_per_iter),
                        +float(max_step_per_iter),
                    )

                pred_after_mount = altaz_cur_mount.copy()
                d_mount_pred = self.model.mount_delta_for_steps(
                    self.model.steps_est,
                    self.model.steps_est + dsteps,
                    J_model=J_motion,
                )
                pred_after_mount[0] = _wrap_deg_360(float(pred_after_mount[0]) + float(d_mount_pred[0]))
                pred_after_mount[1] = float(pred_after_mount[1]) + float(d_mount_pred[1])
                pred_after = self.model._mount_to_world_altaz(pred_after_mount)

                if pred_after[1] < float(self.cfg.alt_min_deg) or pred_after[1] > float(self.cfg.alt_max_deg):
                    alt_target = _clamp(pred_after[1], self.cfg.alt_min_deg, self.cfg.alt_max_deg)
                    delta_alt_allowed = float(alt_target - float(altaz_cur[1]))
                    dalt_pred = float(pred_after[1] - float(altaz_cur[1]))
                    if abs(dalt_pred) > 1e-12:
                        alpha = float(delta_alt_allowed / dalt_pred)
                        alpha = _clamp(alpha, -1.0, 1.0)
                        dsteps *= alpha

                err_distance_deg = float(math.hypot(float(daz), float(dalt)))
                delay_us_az = self._adaptive_slew_delay_us(
                    err_distance_deg,
                    int(self.cfg.slew_delay_us_az),
                    min_delay_us=int(self.cfg.slew_min_delay_us),
                    full_speed_distance_deg=float(self.cfg.slew_full_speed_distance_deg),
                )
                delay_us_alt = self._adaptive_slew_delay_us(
                    err_distance_deg,
                    int(self.cfg.slew_delay_us_alt),
                    min_delay_us=int(self.cfg.slew_min_delay_us),
                    full_speed_distance_deg=float(self.cfg.slew_full_speed_distance_deg),
                )

                duration_s = max(
                    self._estimate_move_duration_s(
                        abs(int(round(float(dsteps[0])))),
                        int(delay_us_az),
                    ),
                    self._estimate_move_duration_s(
                        abs(int(round(float(dsteps[1])))),
                        int(delay_us_alt),
                    ),
                )
                next_arrival = command_time + (
                    duration_s + max(0.0, float(self.cfg.settle_s))
                ) * u.s
                if diagnostics is not None:
                    diagnostics.record(
                        "goto_plan_iteration",
                        iteration=int(_plan_iter + 1),
                        current_altaz_world=altaz_cur,
                        target_altaz_world=altaz_tgt,
                        current_altaz_mount=altaz_cur_mount,
                        target_altaz_mount=altaz_tgt_mount,
                        error_altaz_mount_deg=d_altaz_vec,
                        planned_steps=dsteps,
                        predicted_after_world=pred_after,
                        delay_us_az=int(delay_us_az),
                        delay_us_alt=int(delay_us_alt),
                        estimated_move_duration_s=float(duration_s),
                        estimated_arrival_unix=float(next_arrival.unix),
                        J_deg_per_step=J_motion,
                        inverse_J_step_per_deg=invJ,
                    )
                if abs(float((next_arrival - arrival_time).to_value(u.s))) < 0.01:
                    arrival_time = next_arrival
                    break
                arrival_time = next_arrival

            st.err_az_arcsec = float(daz * 3600.0)
            st.err_alt_arcsec = float(dalt * 3600.0)
            if (abs(st.err_az_arcsec) <= float(self.cfg.tol_arcsec)) and (
                abs(st.err_alt_arcsec) <= float(self.cfg.tol_arcsec)
            ):
                st.ok = True
                st.status = "OK"
                if diagnostics is not None:
                    diagnostics.record(
                        "goto_no_move_needed",
                        error_az_arcsec=st.err_az_arcsec,
                        error_alt_arcsec=st.err_alt_arcsec,
                    )
                return st

            st.iters = 1
            log_info(
                None,
                "GoTo: model move "
                f"dist={float(math.hypot(daz, dalt)):.3f}deg "
                f"dsteps=[{int(round(float(dsteps[0]))):+d},{int(round(float(dsteps[1]))):+d}] "
                f"delay_us=[AZ={int(delay_us_az)},ALT={int(delay_us_alt)}] "
                f"eta={float((arrival_time - command_time).to_value(u.s)):.2f}s",
                throttle_s=0.02,
                throttle_key="goto_model_move",
            )
            if cancel_requested is not None and bool(cancel_requested()):
                st.status = "CANCELLED"
                if diagnostics is not None:
                    diagnostics.record("goto_rejected", reason=st.status)
                return st
            self._exec_steps_parallel(
                move_steps,
                dsteps_az=float(dsteps[0]),
                dsteps_alt=float(dsteps[1]),
                delay_us_az=int(delay_us_az),
                delay_us_alt=int(delay_us_alt),
                stop=stop,
            )
            if diagnostics is not None:
                diagnostics.record(
                    "goto_move_completed",
                    commanded_steps={
                        "az": int(round(float(dsteps[0]))),
                        "alt": int(round(float(dsteps[1]))),
                    },
                    delay_us={"az": int(delay_us_az), "alt": int(delay_us_alt)},
                    model_steps_est=self.model.steps_est,
                )
            time.sleep(max(0.0, float(self.cfg.settle_s)))

            final_time = _now_time()
            final_tgt = icrs_to_altaz_deg(target_icrs, observer=self.cfg.observer, obstime=final_time)
            final_cur = self.model.predict_az_alt_deg()
            if final_cur is not None:
                daz_f = _wrap_deg_180(float(final_tgt[0]) - float(final_cur[0]))
                dalt_f = float(final_tgt[1]) - float(final_cur[1])
                st.err_az_arcsec = float(daz_f * 3600.0)
                st.err_alt_arcsec = float(dalt_f * 3600.0)
            st.ok = True
            within_tolerance = (
                abs(st.err_az_arcsec) <= float(self.cfg.tol_arcsec)
                and abs(st.err_alt_arcsec) <= float(self.cfg.tol_arcsec)
            )
            st.status = "OK" if within_tolerance else "OK_MODEL"
            if diagnostics is not None:
                diagnostics.record(
                    "goto_model_verification",
                    status=st.status,
                    target_altaz=final_tgt,
                    model_altaz=final_cur,
                    error_az_arcsec=st.err_az_arcsec,
                    error_alt_arcsec=st.err_alt_arcsec,
                    within_tolerance=within_tolerance,
                )
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

        self._take_up_backlash_if_reversed(
            move_steps,
            axis=axis,
            direction=direction,
            delay_us=int(delay_us),
        )

        # Update model counter first (best effort even if move fails)
        self.model.note_manual_move(axis, direction, steps)

        # Perform the actual move
        move_steps(axis, direction, steps, int(delay_us))
        # MOVE may be dispatched asynchronously by the runtime callback.
        # Wait estimated move duration so callers that expect blocking semantics
        # (calibration, dithers, etc.) remain deterministic.
        wait_s = self._estimate_move_duration_s(steps, int(delay_us))
        if wait_s > 0.0:
            time.sleep(wait_s + 0.02)

    @staticmethod
    def _estimate_move_duration_s(steps: int, delay_us: int) -> float:
        return estimate_firmware_move_duration_s(int(steps), int(delay_us))

    def _exec_steps_parallel(
        self,
        move_steps: MoveStepsFn,
        *,
        dsteps_az: float,
        dsteps_alt: float,
        delay_us_az: int,
        delay_us_alt: int,
        stop: Optional[StopFn] = None,
    ) -> None:
        s_az = int(round(float(dsteps_az)))
        s_alt = int(round(float(dsteps_alt)))
        if s_az == 0 and s_alt == 0:
            return

        if stop is not None:
            try:
                stop()
            except Exception as exc:
                log_error(None, "GoTo: stop failed before parallel move", exc)

        if s_az != 0:
            dir_az = +1 if s_az >= 0 else -1
            self._take_up_backlash_if_reversed(
                move_steps,
                axis=Axis.AZ,
                direction=dir_az,
                delay_us=int(delay_us_az),
            )
            self.model.note_manual_move(Axis.AZ, dir_az, abs(s_az))
            move_steps(Axis.AZ, dir_az, abs(s_az), int(delay_us_az))

        if s_alt != 0:
            dir_alt = +1 if s_alt >= 0 else -1
            self._take_up_backlash_if_reversed(
                move_steps,
                axis=Axis.ALT,
                direction=dir_alt,
                delay_us=int(delay_us_alt),
            )
            self.model.note_manual_move(Axis.ALT, dir_alt, abs(s_alt))
            move_steps(Axis.ALT, dir_alt, abs(s_alt), int(delay_us_alt))

        wait_az = self._estimate_move_duration_s(abs(s_az), int(delay_us_az))
        wait_alt = self._estimate_move_duration_s(abs(s_alt), int(delay_us_alt))
        wait_s = max(wait_az, wait_alt)
        if wait_s > 0.0:
            time.sleep(wait_s + 0.02)

    def _take_up_backlash_if_reversed(
        self,
        move_steps: MoveStepsFn,
        *,
        axis: Axis,
        direction: int,
        delay_us: int,
    ) -> None:
        """Emit uncounted take-up pulses on a known direction reversal."""
        new_direction = +1 if int(direction) >= 0 else -1
        previous = self.model.last_move_direction(axis)
        if previous == 0 and len(self.model._manual_steps_abs) >= 2:
            axis_idx = 0 if axis == Axis.AZ else 1
            for idx in range(len(self.model._manual_steps_abs) - 1, 0, -1):
                delta = float(
                    self.model._manual_steps_abs[idx][axis_idx]
                    - self.model._manual_steps_abs[idx - 1][axis_idx]
                )
                if abs(delta) >= 1.0:
                    previous = +1 if delta > 0.0 else -1
                    break
        takeup = int(
            self.model.backlash_steps_az
            if axis == Axis.AZ
            else self.model.backlash_steps_alt
        )
        if previous == 0 or int(previous) == new_direction or takeup <= 0:
            return
        move_steps(axis, new_direction, takeup, int(delay_us))
        self.model.set_last_move_direction(axis, new_direction)
        wait_s = self._estimate_move_duration_s(takeup, int(delay_us))
        if wait_s > 0.0:
            time.sleep(wait_s + 0.02)

    def _adaptive_slew_delay_us(
        self,
        distance_deg: float,
        base_delay_us: int,
        *,
        min_delay_us: int = 400,
        full_speed_distance_deg: float = 20.0,
    ) -> int:
        """Map angular distance to MOVE delay (bigger distance => smaller delay)."""
        base_i = max(1, int(base_delay_us))
        min_i = max(1, min(int(min_delay_us), base_i))
        dist = max(0.0, float(distance_deg))
        if float(full_speed_distance_deg) <= 1e-9:
            return int(min_i)
        alpha = _clamp(dist / float(full_speed_distance_deg), 0.0, 1.0)
        delay = float(base_i) - alpha * float(base_i - min_i)
        return int(max(min_i, round(delay)))

    def _delay_us_to_rate_steps_s(self, delay_us: int) -> float:
        d = max(1, int(delay_us))
        # MOVE uses roughly HIGH+LOW delays, i.e. ~2*delay_us per microstep.
        return float(1.0e6 / (2.0 * float(d)))

    def _exec_rate_vector_move(
        self,
        rate_mount: RateMountFn,
        dsteps: np.ndarray,
        *,
        delay_us_az: int,
        delay_us_alt: int,
        stop: Optional[StopFn] = None,
    ) -> None:
        s_az = int(round(float(dsteps[0])))
        s_alt = int(round(float(dsteps[1])))
        if s_az == 0 and s_alt == 0:
            return

        # Update model counters first (same policy as _exec_steps).
        if s_az != 0:
            self.model.note_manual_move(Axis.AZ, +1 if s_az >= 0 else -1, abs(s_az))
        if s_alt != 0:
            self.model.note_manual_move(Axis.ALT, +1 if s_alt >= 0 else -1, abs(s_alt))

        max_rate_az = max(1e-3, self._delay_us_to_rate_steps_s(int(delay_us_az)))
        max_rate_alt = max(1e-3, self._delay_us_to_rate_steps_s(int(delay_us_alt)))
        t_az = abs(float(s_az)) / max_rate_az if s_az != 0 else 0.0
        t_alt = abs(float(s_alt)) / max_rate_alt if s_alt != 0 else 0.0
        duration_s = max(t_az, t_alt)
        if duration_s <= 0.0:
            return

        az_rate = _clamp(float(s_az) / duration_s, -max_rate_az, +max_rate_az)
        alt_rate = _clamp(float(s_alt) / duration_s, -max_rate_alt, +max_rate_alt)
        log_info(
            None,
            "GoTo: rate move "
            f"dsteps=[{s_az:+d},{s_alt:+d}] "
            f"rates=[{az_rate:+.2f},{alt_rate:+.2f}] steps/s "
            f"dur={duration_s:.3f}s",
            throttle_s=0.02,
            throttle_key="goto_rate_move",
        )

        if stop is not None:
            try:
                stop()
            except Exception as exc:
                log_error(None, "GoTo: stop failed before rate move", exc)

        t0 = time.perf_counter()
        try:
            rate_mount(float(az_rate), float(alt_rate))
            while True:
                dt = float(time.perf_counter() - t0)
                rem = float(duration_s - dt)
                if rem <= 0.0:
                    break
                time.sleep(min(0.02, rem))
        finally:
            try:
                rate_mount(0.0, 0.0)
            except Exception as exc:
                log_error(None, "GoTo: failed to stop rate move", exc)
            if stop is not None:
                try:
                    stop()
                except Exception as exc:
                    log_error(None, "GoTo: stop failed after rate move", exc)

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
        diagnostics: Optional[DiagnosticSession] = None,
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
                    diagnostics=diagnostics,
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

                daz_mount_deg = radius * math.cos(ang)
                dalt_mount_deg = radius * math.sin(ang)

                J = np.asarray(self.model.J_deg_per_step, dtype=np.float64)
                if not self.model.is_J_within_mechanical_limits(J):
                    log_error(
                        None,
                        "GoTo: calibration model outside 45:1 limits; resetting mechanics",
                        None,
                        throttle_s=5.0,
                        throttle_key="goto_calib_unsafe_J",
                    )
                    self.model.init_from_mechanics()
                    J = self.model.J_deg_per_step
                try:
                    invJ = np.linalg.inv(J)
                except np.linalg.LinAlgError as exc:
                    log_error(None, "GoTo: singular J matrix during calibration; resetting mechanics", exc, throttle_s=5.0, throttle_key="goto_calib_invJ")
                    # fall back to diagonal mechanics
                    self.model.init_from_mechanics()
                    J = self.model.J_deg_per_step
                    invJ = np.linalg.inv(J)

                dsteps = invJ @ np.array([daz_mount_deg, dalt_mount_deg], dtype=np.float64)

                # Commanded steps are integers; use the same for prediction + sampling
                dsteps = np.array([float(int(round(dsteps[0]))), float(int(round(dsteps[1])))], dtype=np.float64)
                if int(dsteps[0]) == 0 and int(dsteps[1]) == 0:
                    continue

                # Predict and enforce ALT safe range by flipping ALT sign if needed
                altaz_cur = self.model.current_az_alt_deg()
                if altaz_cur is None:
                    out["status"] = "ERR_NO_CURRENT"
                    return out
                altaz_cur_mount = self.model._world_to_mount_altaz(altaz_cur)

                pred_after_mount = altaz_cur_mount.copy()
                d_mount_pred = J @ dsteps
                pred_after_mount[0] = _wrap_deg_360(float(pred_after_mount[0]) + float(d_mount_pred[0]))
                pred_after_mount[1] = float(pred_after_mount[1]) + float(d_mount_pred[1])
                pred_after = self.model._mount_to_world_altaz(pred_after_mount)
                if pred_after[1] < float(self.cfg.alt_min_deg) or pred_after[1] > float(self.cfg.alt_max_deg):
                    # flip the ALT component
                    dsteps[1] *= -1.0
                    d_mount_pred = J @ dsteps
                    pred_after_mount[0] = _wrap_deg_360(float(altaz_cur_mount[0]) + float(d_mount_pred[0]))
                    pred_after_mount[1] = float(altaz_cur_mount[1]) + float(d_mount_pred[1])
                    pred_after = self.model._mount_to_world_altaz(pred_after_mount)

                if stop is not None:
                    try:
                        stop()
                    except Exception as exc:
                        log_error(None, "GoTo: stop failed before calibration move", exc)

                # Move
                steps_before = self.model.steps_est.copy()
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
                solve_obstime = _now_time()
                sol = self._platesolving_live(
                    get_live_frame=get_live_frame,
                    target_for_solver={"az_deg": float(altaz_pred[0]), "alt_deg": float(altaz_pred[1])},
                    platesolving_cfg=calib_platesolving_cfg,
                    radius_deg_seq=(1.0,),
                    obstime=solve_obstime,
                    diagnostics=diagnostics,
                )
                if not bool(getattr(sol, "success", False)):
                    # skip sample
                    continue

                altaz_new = platesolving_center_to_altaz_deg(
                    float(sol.center_ra_deg),
                    float(sol.center_dec_deg),
                    observer=self.cfg.observer,
                    obstime=_platesolving_result_obstime(sol, fallback=solve_obstime),
                )
                continuity = self.model.manual_sample_continuity_report(
                    altaz_new,
                    reference_steps=steps_before,
                    reference_az_alt_deg=altaz_cur,
                )
                if not bool(continuity.get("ok", False)):
                    log_error(
                        None,
                        "GoTo: calibration rejected implausible plate-solve "
                        f"dsteps=[{float(continuity.get('dsteps_az', 0.0)):+.0f},"
                        f"{float(continuity.get('dsteps_alt', 0.0)):+.0f}] "
                        f"motion={float(continuity.get('observed_motion_deg', float('nan'))):.4f}deg "
                        f"limit={float(continuity.get('motion_limit_deg', float('nan'))):.4f}deg",
                    )
                    continue

                # Measured mount-frame delta (J is modeled in mount frame).
                altaz_new_mount = self.model._world_to_mount_altaz(altaz_new)
                daltaz_meas_mount = np.array(
                    [
                        _wrap_deg_180(float(altaz_new_mount[0]) - float(altaz_cur_mount[0])),
                        float(altaz_new_mount[1]) - float(altaz_cur_mount[1]),
                    ],
                    dtype=np.float64,
                )

                # Measured step delta (what we commanded this sample)
                dsteps_meas = np.array([float(dsteps[0]), float(dsteps[1])], dtype=np.float64)

                self.model.add_calibration_sample(dsteps_meas, daltaz_meas_mount)
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
    t_wall: float
    t_mono: float
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
        self._rate_steps_capture: Optional[np.ndarray] = None
        self._initial_solution_confirmed = False
        self._diagnostics: Optional[DiagnosticSession] = None

    def request(self, *, kind: str, target: Any, params: Dict[str, Any]) -> None:
        # Clear cancellation for the new request before it becomes visible to
        # the worker. If STOP arrives after this point, its cancellation must
        # remain set; clearing it inside _handle_request created a race where
        # an emergency stop just before startup was silently ignored.
        self._op_cancel.clear()
        super().request(kind=str(kind), target=target, params=dict(params))

    def cancel(self) -> None:
        self._op_cancel.set()

    def _command_mount_rate(self, az_rate: float, alt_rate: float) -> Any:
        if self._rate_mount is None:
            return None
        result = self._rate_mount(float(az_rate), float(alt_rate))
        capture = self._rate_steps_capture
        if capture is not None and result is not None:
            try:
                moved = np.asarray(result, dtype=np.float64).reshape(-1)
                if moved.size == 2 and np.all(np.isfinite(moved)):
                    capture += moved
            except (TypeError, ValueError):
                pass
        return result

    def _finish_rate_step_capture(self) -> np.ndarray:
        moved = self._rate_steps_capture
        self._rate_steps_capture = None
        if moved is None:
            return np.zeros(2, dtype=np.float64)
        return np.asarray(moved, dtype=np.float64).reshape(2).copy()

    def _apply_rate_steps_to_model(self, moved_steps: np.ndarray) -> None:
        moved = np.rint(np.asarray(moved_steps, dtype=np.float64).reshape(2))
        self._goto.model.note_emitted_rate_steps(moved)

    def _diagnostics_record(self, stage: str, **payload: Any) -> None:
        diagnostics = self._diagnostics
        if diagnostics is not None:
            diagnostics.record(stage, **payload)

    def _diagnostics_model_snapshot(self) -> Dict[str, Any]:
        model = self._goto.model
        return {
            "synced": bool(model.synced),
            "steps_est": np.asarray(model.steps_est, dtype=np.float64),
            "ref_steps": np.asarray(model.ref_steps, dtype=np.float64),
            "ref_az_alt_mount_deg": np.asarray(model.ref_az_alt_deg, dtype=np.float64),
            "current_az_alt_world_deg": model.current_az_alt_deg(),
            "J_deg_per_step": np.asarray(model.J_deg_per_step, dtype=np.float64),
            "mechanical_J_deg_per_step": model.mechanical_J(),
            "R_mount_to_world": np.asarray(model.R_mount_to_world, dtype=np.float64),
            "manual_steps_abs": list(model._manual_steps_abs),
            "manual_az_alt_abs": list(model._manual_az_alt_abs),
            "manual_roll_deg_abs": list(model._manual_roll_deg_abs),
            "manual_source_abs": list(model._manual_source_abs),
            "manual_fit_inlier_mask": model._manual_fit_inlier_mask,
            "fit_report": model.model_fit_report(),
        }

    def _diagnostics_save_live_frame(self, stage: str) -> Optional[str]:
        diagnostics = self._diagnostics
        if diagnostics is None:
            return None
        frame = self._get_frame()
        if frame is None:
            diagnostics.record("live_frame_missing", requested_stage=str(stage))
            return None
        try:
            raw16 = ensure_raw16_bayer(frame.raw).copy()
            return diagnostics.save_raw(
                str(stage),
                raw16,
                metadata={
                    "frame_t_capture": getattr(frame, "t_capture", None),
                    "frame_meta": dict(getattr(frame, "meta", {}) or {}),
                    "camera_config": self._get_camera_cfg(),
                },
            )
        except Exception as exc:
            diagnostics.record(
                "live_frame_save_failed",
                requested_stage=str(stage),
                error=repr(exc),
            )
            return None

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

    def _autocal_frame_time_s(self, fr: _AutocalFrame) -> float:
        t_mono = float(getattr(fr, "t_mono", float("nan")))
        if np.isfinite(t_mono):
            return t_mono
        t_capture = float(getattr(fr, "t_capture", float("nan")))
        if np.isfinite(t_capture):
            return t_capture
        return float(getattr(fr, "t_wall", _now_s()))

    def _autocal_frame_obstime(self, fr: _AutocalFrame) -> Time:
        t_wall = float(getattr(fr, "t_wall", float("nan")))
        if np.isfinite(t_wall) and t_wall > 0.0:
            return Time(t_wall, format="unix", scale="utc")
        return Time.now()

    def _autocal_detect(
        self, raw16: np.ndarray
    ) -> Tuple[np.ndarray, int, float, Tuple[Tuple[float, float, float], ...]]:
        sep_cfg = self._get_sep_cfg()
        platesolving_cfg = self._get_platesolving_cfg()
        img_det, bkg, objects, obj_xy = sep_detect_from_raw16(
            raw16,
            sep_bw=int(sep_cfg.bw),
            sep_bh=int(sep_cfg.bh),
            sep_thresh_sigma=float(sep_cfg.thresh_sigma),
            sep_minarea=int(sep_cfg.minarea),
            max_sources=int(platesolving_cfg.max_det),
        )
        star_count = int(obj_xy.shape[0])
        # RAW16 quantization can make SEP's global RMS unrealistically small.
        # Do not reduce exposure from active area alone: require the extracted
        # source list to be close to its configured cap as corroboration.
        threshold = float(sep_cfg.thresh_sigma) * float(bkg.globalrms)
        active_fraction = float(np.mean(np.asarray(img_det) > threshold))
        crowded_limit = 0.08
        max_sources = int(platesolving_cfg.max_det)
        if _autocal_frame_is_crowded(
            active_fraction=active_fraction,
            star_count=star_count,
            max_sources=max_sources,
            crowded_limit=crowded_limit,
        ):
            star_count = max(star_count, int(platesolving_cfg.max_det) + 1)
            log_info(
                self._out_log,
                "GoTo: AutoCal crowded detection frame "
                f"active_pixels={100.0 * active_fraction:.1f}% -> reducing exposure/gain",
                throttle_s=2.0,
                throttle_key="goto_autocal_crowded_detection",
            )
        elif np.isfinite(active_fraction) and active_fraction > crowded_limit:
            log_info(
                self._out_log,
                "GoTo: AutoCal quantized/noisy background "
                f"active_pixels={100.0 * active_fraction:.1f}% "
                f"sources={star_count}/{max_sources}; preserving exposure/gain",
                throttle_s=2.0,
                throttle_key="goto_autocal_quantized_background",
            )
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
        min_usable_frames: int = 0,
        min_usable_sources: int = 1,
        rate_hold_axis: Optional[Axis] = None,
        rate_hold_steps_s: float = 0.0,
        rate_hold_hz: float = 20.0,
        diagnostic_stage: Optional[str] = None,
    ) -> List[_AutocalFrame]:
        frames: List[_AutocalFrame] = []
        deadline = _perf() + float(timeout_s)
        last_frame_token: Optional[Tuple[str, int]] = None
        last_capture_t: Optional[float] = None
        usable = 0
        min_dt_s = max(0.0, float(min_dt_s))
        skip_remaining = max(0, int(skip_frames))
        min_usable_frames = max(0, int(min_usable_frames))
        min_usable_sources = max(1, int(min_usable_sources))
        rate_hold_enabled = bool(
            (self._rate_mount is not None)
            and (rate_hold_axis is not None)
            and np.isfinite(float(rate_hold_steps_s))
            and (abs(float(rate_hold_steps_s)) > 1.0e-9)
        )
        rate_hold_axis_resolved: Optional[Axis] = None
        if rate_hold_enabled:
            rate_hold_axis_resolved = Axis(rate_hold_axis)
        rate_hold_period_s = 1.0 / max(1.0, float(rate_hold_hz))
        next_rate_hold_t = _perf()
        while (
            (len(frames) < int(n_frames) or (min_usable_frames > 0 and usable < min_usable_frames))
            and _perf() < deadline
        ):
            if self._op_cancel.is_set():
                break
            if rate_hold_enabled and (rate_hold_axis_resolved is not None) and _perf() >= next_rate_hold_t:
                try:
                    az_rate, alt_rate = self._autocal_axis_rates(rate_hold_axis_resolved, float(rate_hold_steps_s))
                    self._command_mount_rate(az_rate, alt_rate)
                except Exception as exc:
                    log_error(
                        self._out_log,
                        "GoTo: failed to maintain autocal axis rate",
                        exc,
                        throttle_s=2.0,
                        throttle_key="goto_autocal_rate_hold",
                    )
                finally:
                    next_rate_hold_t = _perf() + rate_hold_period_s
            fr = self._get_frame()
            if fr is None:
                time.sleep(0.01)
                continue
            seq = self._frame_seq(fr)
            frame_token = (
                ("seq", int(seq)) if seq is not None else ("object", int(id(fr)))
            )
            if last_frame_token is not None and frame_token == last_frame_token:
                time.sleep(0.005)
                continue
            last_frame_token = frame_token
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            raw16 = ensure_raw16_bayer(fr.raw).copy()
            try:
                obj_xy, star_count, saturation_frac, top_sources = self._autocal_detect(raw16)
            except Exception as exc:
                log_error(
                    self._out_log,
                    "GoTo: AutoCal frame detection failed; waiting for a new frame",
                    exc,
                    throttle_s=2.0,
                    throttle_key="goto_autocal_frame_detection",
                )
                time.sleep(0.01)
                continue
            try:
                t_capture = float(getattr(fr, "t_capture", float("nan")))
            except Exception:
                t_capture = float("nan")
            try:
                t_wall = float(fr.meta.get("t_wall", _now_s()))
            except Exception:
                t_wall = float(_now_s())
            try:
                t_mono = float(fr.meta.get("t_capture_mono", t_capture))
            except Exception:
                t_mono = t_capture
            if not np.isfinite(t_mono):
                t_mono = float(_perf())
            frame_t = t_mono
            if last_capture_t is not None and (frame_t - last_capture_t) < min_dt_s:
                time.sleep(0.005)
                continue
            frames.append(
                _AutocalFrame(
                    raw16=raw16,
                    t_capture=t_capture,
                    t_wall=t_wall,
                    t_mono=t_mono,
                    obj_xy=obj_xy,
                    star_count=star_count,
                    saturation_frac=saturation_frac,
                    top_sources=top_sources,
                )
            )
            last_capture_t = frame_t
            if obj_xy.shape[0] >= min_usable_sources:
                usable += 1
        diagnostics = self._diagnostics
        if diagnostics is not None and frames and diagnostic_stage:
            diagnostics.save_raw_stack(
                str(diagnostic_stage),
                [fr.raw16 for fr in frames],
                frame_metadata=[
                    {
                        "t_capture": fr.t_capture,
                        "t_wall": fr.t_wall,
                        "t_mono": fr.t_mono,
                        "star_count": fr.star_count,
                        "saturation_fraction": fr.saturation_frac,
                        "top_sources": fr.top_sources,
                    }
                    for fr in frames
                ],
                metadata={
                    "requested_frames": int(n_frames),
                    "captured_frames": int(len(frames)),
                    "min_usable_frames": int(min_usable_frames),
                    "min_usable_sources": int(min_usable_sources),
                    "rate_hold_axis": rate_hold_axis,
                    "rate_hold_steps_s": float(rate_hold_steps_s),
                },
            )
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
        """Preserve the exposure/gain chosen by the operator.

        This legacy helper intentionally remains as a no-op so old callers
        cannot take ownership of camera capture settings.
        """
        cam_cfg = self._get_camera_cfg()
        exp_ms = float(getattr(cam_cfg, "exp_ms", 0.0))
        gain = int(getattr(cam_cfg, "gain", 0))
        record_diagnostic = getattr(self, "_diagnostics_record", lambda _stage, **_payload: None)
        record_diagnostic(
            "exposure_tune_decision",
            action="preserve_operator_settings",
            star_count=int(star_count),
            saturation_fraction=float(saturation_frac),
            exposure_ms=float(exp_ms),
            gain=int(gain),
        )
        _ = target_min, target_max, sat_max, exp_min_ms, exp_max_ms, exp_step
        _ = gain_min, gain_max, gain_step, settle_s
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

    def _autocal_estimate_drift_stack(
        self,
        frames: List[_AutocalFrame],
        *,
        window: int,
        median_k: int,
        smooth_k: int,
        vmax_px_s: float,
        margin_px: float,
        max_shift_cap: int,
        profile_q: Optional[float],
        use_subpixel: bool,
    ) -> Optional[np.ndarray]:
        n = int(len(frames))
        if n <= int(window):
            log_error(
                self._out_log,
                f"GoTo: AutoCal drift stack needs > window frames (n={n} window={int(window)})",
            )
            return None

        t = np.asarray([self._autocal_frame_time_s(fr) for fr in frames], dtype=np.float64)
        order = np.argsort(t)
        t = t[order]
        frames = [frames[i] for i in order]

        dt = np.diff(t)
        dt = dt[dt > 0.0]
        dt_med: Optional[float] = None
        if dt.size > 0:
            dt_med = float(np.median(dt))
            if not np.isfinite(dt_med) or dt_med <= 0.0:
                dt_med = None
        if dt_med is None:
            log_error(self._out_log, "GoTo: AutoCal drift stack missing valid frame capture timestamps")
            return None

        try:
            stack = np.stack([fr.raw16 for fr in frames], axis=0)
        except Exception as exc:
            log_error(self._out_log, "GoTo: AutoCal drift stack build failed", exc)
            return None

        try:
            out = estimate_sensor_drift_from_stack(
                stack,
                frame_times_s=t,
                window=int(window),
                median_k=int(median_k),
                smooth_k=int(smooth_k),
                vmax_px_s=float(vmax_px_s),
                margin_px=float(margin_px),
                max_shift_cap=int(max_shift_cap),
                profile_q=profile_q,
                use_subpixel=bool(use_subpixel),
                return_per_window=False,
            )
        except Exception as exc:
            log_error(self._out_log, "GoTo: AutoCal drift stack failed", exc)
            return None

        vx = float(out.get("vx_mean", 0.0))
        vy = float(out.get("vy_mean", 0.0))
        if not np.isfinite(vx) or not np.isfinite(vy):
            log_error(self._out_log, "GoTo: AutoCal drift stack produced non-finite velocity")
            return None

        # estimator uses +y down; autocal uses +y up
        v = np.array([vx, -vy], dtype=np.float64)

        log_info(
            self._out_log,
            "GoTo: AutoCal drift stack "
            f"v=[{float(v[0]):.3f},{float(v[1]):.3f}] "
            f"dt_med={float(dt_med or 0.0):.3f}s dt_span={float(t[-1] - t[0]):.3f}s "
            f"window={int(window)} n={n} "
            f"vx_std={float(out.get('vx_std', 0.0)):.3f} vy_std={float(out.get('vy_std', 0.0)):.3f}",
        )
        self._diagnostics_record(
            "drift_estimate_stack",
            velocity_px_s_xy_up=v,
            estimator_output=out,
            frame_times_s=t,
            frame_count=n,
            window=int(window),
        )
        return v

    def _autocal_estimate_drift_line_sweep(
        self,
        frames: List[_AutocalFrame],
        *,
        min_frames: int,
        min_duration_s: float,
        min_sources: int,
        topk_sources: int,
        deg_step: float,
        bin_width_px: float,
        topk_bins_for_score: int,
        use_theil_sen: bool,
        max_shift_px: float,
        min_resp: float,
    ) -> Optional[np.ndarray]:
        if len(frames) < int(min_frames):
            log_error(
                self._out_log,
                f"GoTo: AutoCal drift line sweep needs >= {int(min_frames)} frames, got {len(frames)}",
            )
            return None

        per_frame: List[np.ndarray] = []
        t_list: List[float] = []
        min_sources = max(1, int(min_sources))
        for fr in frames:
            n = int(fr.obj_xy.shape[0])
            if n < min_sources:
                continue
            k = int(min(int(topk_sources), n))
            if k <= 0:
                continue
            per_frame.append(fr.obj_xy[:k].astype(np.float64, copy=False))
            t_list.append(self._autocal_frame_time_s(fr))

        if len(per_frame) < int(min_frames):
            log_error(
                self._out_log,
                f"GoTo: AutoCal drift line sweep insufficient usable frames "
                f"usable={len(per_frame)} min={int(min_frames)}",
            )
            return None

        t = np.asarray(t_list, dtype=np.float64)
        if t.size < 2:
            log_error(self._out_log, "GoTo: AutoCal drift line sweep needs >=2 timestamps")
            return None
        order = np.argsort(t)
        t = t[order]
        per_frame = [per_frame[i] for i in order]

        diffs = np.diff(t)
        diffs = diffs[diffs > 0.0]
        dt_med: Optional[float] = None
        if diffs.size > 0:
            dt_med = float(np.median(diffs))
            if not np.isfinite(dt_med) or dt_med <= 0.0:
                dt_med = None

        duration_s = float(t[-1] - t[0])
        if not np.isfinite(duration_s) or duration_s < float(min_duration_s):
            log_error(
                self._out_log,
                "GoTo: AutoCal drift line sweep duration too short "
                f"dt={duration_s:.3f}s min={float(min_duration_s):.3f}s",
            )
            return None

        xy_all = np.vstack(per_frame)
        try:
            best, _degs, _scores = _angle_sweep_best_direction(
                xy_all,
                deg_min=0.0,
                deg_max=180.0,
                deg_step=float(deg_step),
                bin_width_px=float(bin_width_px),
                topk_bins_for_score=int(topk_bins_for_score),
            )
        except Exception as exc:
            log_error(
                self._out_log,
                "GoTo: AutoCal drift line sweep failed",
                exc,
            )
            return None

        u = best.get("u", None)
        if u is None:
            log_error(self._out_log, "GoTo: AutoCal drift line sweep missing direction vector")
            return None
        u = np.asarray(u, dtype=np.float64).reshape(2,)

        # Estimate drift speed using flux-order matching of top-k sources.
        vels: List[np.ndarray] = []
        total_pairs = 0
        for i in range(1, len(per_frame)):
            dt = float(t[i] - t[i - 1])
            if dt <= 0.0:
                continue
            total_pairs += 1
            ref = per_frame[i - 1]
            cur = per_frame[i]
            n_ref = int(ref.shape[0])
            n_cur = int(cur.shape[0])
            n = min(n_ref, n_cur)
            if n < int(min_sources):
                continue
            shifts = ref[:n] - cur[:n]
            mags = np.hypot(shifts[:, 0], shifts[:, 1])
            good = mags <= float(max_shift_px)
            if not np.any(good):
                continue
            shifts = shifts[good]
            dx = float(np.median(shifts[:, 0]))
            dy = float(np.median(shifts[:, 1]))
            resp = float(int(np.count_nonzero(good)) / n)
            if float(resp) < float(min_resp):
                continue
            # vels in image coords (+x right, +y down)
            vels.append(np.array([-dx / dt, -dy / dt], dtype=np.float64))

        if vels:
            v_arr = np.stack(vels, axis=0)
            speed = float(np.median(v_arr @ u))
            v_img = speed * u
            v = np.array([float(v_img[0]), -float(v_img[1])], dtype=np.float64)  # to +y up
            if np.all(np.isfinite(v)):
                log_info(
                    self._out_log,
                    "GoTo: AutoCal drift line sweep "
                    f"deg={float(best.get('deg', 0.0)):.2f} score={float(best.get('score', 0.0)):.3f} "
                    f"v=[{float(v[0]):.3f},{float(v[1]):.3f}] "
                    f"dt_med={float(dt_med or 0.0):.3f}s pairs={len(vels)}/{total_pairs} "
                    f"frames={len(per_frame)} pts={xy_all.shape[0]}",
                )
                self._diagnostics_record(
                    "drift_estimate_line_sweep",
                    mode="matched_velocity",
                    velocity_px_s_xy_up=v,
                    direction=best,
                    frame_times_s=t,
                    usable_frame_count=len(per_frame),
                    matched_pairs=len(vels),
                )
                return v

        # Fallback: project positions onto drift axis and fit slope.
        t_rel = t - t[0]
        t_f = np.array([np.median(p @ u) for p in per_frame], dtype=np.float64)
        if not np.all(np.isfinite(t_f)):
            log_error(self._out_log, "GoTo: AutoCal drift line sweep invalid projection values")
            return None

        if bool(use_theil_sen):
            slope = _robust_line_fit_slope(t_rel, t_f)
        else:
            A = np.column_stack([t_rel, np.ones_like(t_rel)])
            slope, _b = np.linalg.lstsq(A, t_f, rcond=None)[0]

        v_img = float(slope) * u
        v = np.array([float(v_img[0]), -float(v_img[1])], dtype=np.float64)  # to +y up
        if not np.all(np.isfinite(v)):
            log_error(self._out_log, "GoTo: AutoCal drift line sweep produced non-finite velocity")
            return None

        log_info(
            self._out_log,
            "GoTo: AutoCal drift line sweep (proj) "
            f"deg={float(best.get('deg', 0.0)):.2f} score={float(best.get('score', 0.0)):.3f} "
            f"slope={float(slope):.6f} v=[{float(v[0]):.3f},{float(v[1]):.3f}] "
            f"frames={len(per_frame)} pts={xy_all.shape[0]}",
        )
        self._diagnostics_record(
            "drift_estimate_line_sweep",
            mode="projection_fit",
            velocity_px_s_xy_up=v,
            direction=best,
            slope=float(slope),
            frame_times_s=t,
            usable_frame_count=len(per_frame),
        )
        return v

    def _goto_estimate_roll_blocking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": False,
            "status": "RUNNING",
            "roll_deg": None,
        }

        st = self._get_state()
        if not bool(st.camera.connected):
            out["status"] = "ERR_NO_CAMERA"
            log_error(self._out_log, "GoTo: Roll estimate failed (no camera)")
            return out
        if not bool(st.mount.connected):
            out["status"] = "ERR_NO_MOUNT"
            log_error(self._out_log, "GoTo: Roll estimate failed (no mount)")
            return out
        if self._rate_mount is None:
            out["status"] = "ERR_NO_RATE"
            log_error(self._out_log, "GoTo: Roll estimate failed (rate unavailable)")
            return out

        try:
            self._command_mount_rate(0.0, 0.0)
        except Exception as exc:
            log_error(self._out_log, "GoTo: Roll estimate failed to stop mount", exc)

        mount_cfg = self._get_mount_cfg()

        roll_window = int(params.get("roll_window", 60))
        roll_frames = int(params.get("roll_frames", max(roll_window + 5, 80)))
        roll_dt_min = float(params.get("roll_dt_min_s", 0.0))
        roll_timeout_s = float(params.get("roll_capture_timeout_s", 18.0))
        roll_min_sources = int(params.get("roll_min_sources", 1))

        roll_rate_steps_s = float(
            params.get("roll_rate_steps_s", min(10.0, float(getattr(mount_cfg, "default_rate", 80.0))))
        )
        roll_rate_steps_s = abs(float(roll_rate_steps_s))
        roll_rate_max_steps_s = float(params.get("roll_rate_max_steps_s", 10.0))
        roll_rate_min_steps_s = float(params.get("roll_rate_min_steps_s", 2.5))
        roll_rate_backoff = float(params.get("roll_rate_backoff", 0.70))
        roll_rate_attempts = int(params.get("roll_rate_attempts", 4))
        roll_follow_vx = bool(params.get("roll_follow_vx", True))
        roll_ramp_s = float(params.get("roll_ramp_s", 0.6))
        roll_ramp_hz = float(params.get("roll_ramp_hz", 25.0))
        roll_settle_s = float(params.get("roll_settle_s", 0.2))
        roll_timeout_slow_scale_max = float(params.get("roll_capture_timeout_slow_scale_max", 3.0))

        roll_stack_median_k = int(params.get("roll_stack_median_k", 3))
        roll_stack_smooth_k = int(params.get("roll_stack_smooth_k", 20))
        # Roll estimation runs while slewing; allow significantly larger drift than "sensor quiet" defaults.
        roll_stack_vmax_px_s = float(params.get("roll_stack_vmax_px_s", 120.0))
        roll_stack_margin_px = float(params.get("roll_stack_margin_px", 10.0))
        roll_stack_max_shift_cap = int(params.get("roll_stack_max_shift_cap", 600))
        roll_stack_profile_q = params.get("roll_stack_profile_q", None)
        roll_stack_use_subpixel = bool(params.get("roll_stack_use_subpixel", False))
        roll_use_stack = bool(params.get("roll_use_stack", True))
        roll_use_line = bool(params.get("roll_use_line", False))
        roll_line_max_shift_px = float(params.get("roll_line_max_shift_px", 120.0))
        roll_max_rel_std = float(params.get("roll_max_rel_std", 0.35))
        roll_max_speed_px_s = float(params.get("roll_max_speed_px_s", max(120.0, roll_stack_vmax_px_s)))
        roll_guard_prev_axis = bool(params.get("roll_guard_prev_axis", True))
        roll_max_axis_jump_deg = float(params.get("roll_max_axis_jump_deg", 35.0))

        if roll_window < 1:
            roll_window = 1
        if roll_stack_profile_q is not None:
            try:
                roll_stack_profile_q = float(roll_stack_profile_q)
                if not np.isfinite(roll_stack_profile_q):
                    roll_stack_profile_q = None
            except (TypeError, ValueError):
                roll_stack_profile_q = None
        if roll_rate_steps_s <= 0.0:
            out["status"] = "ERR_BAD_RATE"
            log_error(self._out_log, "GoTo: Roll estimate failed (rate <= 0)")
            return out
        if np.isfinite(roll_rate_max_steps_s) and roll_rate_max_steps_s > 0.0:
            roll_rate_steps_s = min(float(roll_rate_steps_s), float(roll_rate_max_steps_s))
        if not np.isfinite(roll_rate_min_steps_s) or roll_rate_min_steps_s <= 0.0:
            roll_rate_min_steps_s = 1.0
        if roll_rate_steps_s < roll_rate_min_steps_s:
            roll_rate_steps_s = float(roll_rate_min_steps_s)
        if not np.isfinite(roll_rate_backoff) or not (0.0 < roll_rate_backoff < 1.0):
            roll_rate_backoff = 0.70
        if roll_rate_attempts < 1:
            roll_rate_attempts = 1
        if (not np.isfinite(roll_timeout_slow_scale_max)) or roll_timeout_slow_scale_max < 1.0:
            roll_timeout_slow_scale_max = 1.0
        if not np.isfinite(roll_max_speed_px_s) or roll_max_speed_px_s <= 0.0:
            out["status"] = "ERR_BAD_SPEED_LIMIT"
            log_error(self._out_log, "GoTo: Roll estimate failed (invalid roll_max_speed_px_s)")
            return out
        if (not np.isfinite(roll_max_axis_jump_deg)) or roll_max_axis_jump_deg <= 0.0:
            roll_guard_prev_axis = False
        if (not roll_use_stack) and (not roll_use_line):
            out["status"] = "ERR_ROLL_METHOD"
            log_error(self._out_log, "GoTo: Roll estimate failed (all methods disabled)")
            return out

        def _capture_timeout_for_rate(rate_steps_s: float) -> float:
            timeout_s = float(roll_timeout_s)
            if timeout_s <= 0.0:
                return timeout_s
            rate_abs = abs(float(rate_steps_s))
            if (not np.isfinite(rate_abs)) or rate_abs <= 1e-9:
                return timeout_s
            scale = float(roll_rate_steps_s) / rate_abs
            if not np.isfinite(scale) or scale < 1.0:
                scale = 1.0
            scale = min(float(roll_timeout_slow_scale_max), float(scale))
            return float(timeout_s * scale)

        def _estimate_drift(frames: List[_AutocalFrame], label: str) -> Optional[np.ndarray]:
            v = None
            source = "none"
            if roll_use_stack:
                try:
                    t = np.asarray([self._autocal_frame_time_s(fr) for fr in frames], dtype=np.float64)
                    order = np.argsort(t)
                    t = t[order]
                    frames_sorted = [frames[i] for i in order]
                    stack = np.stack([fr.raw16 for fr in frames_sorted], axis=0)
                    drift_out = estimate_sensor_drift_from_stack(
                        stack,
                        frame_times_s=t,
                        window=int(roll_window),
                        median_k=int(roll_stack_median_k),
                        smooth_k=int(roll_stack_smooth_k),
                        vmax_px_s=float(roll_stack_vmax_px_s),
                        margin_px=float(roll_stack_margin_px),
                        max_shift_cap=int(roll_stack_max_shift_cap),
                        profile_q=roll_stack_profile_q,
                        use_subpixel=bool(roll_stack_use_subpixel),
                        return_per_window=False,
                    )
                    vx = float(drift_out.get("vx_mean", 0.0))
                    vy = float(drift_out.get("vy_mean", 0.0))
                    vx_std = float(drift_out.get("vx_std", 0.0))
                    vy_std = float(drift_out.get("vy_std", 0.0))
                    v = np.array([vx, -vy], dtype=np.float64)
                    speed = float(np.hypot(v[0], v[1]))
                    rel_std = float(max(vx_std, vy_std) / max(speed, 1e-6))
                    if not np.all(np.isfinite(v)):
                        log_error(self._out_log, f"GoTo: Roll {label} stack drift non-finite")
                        v = None
                    elif rel_std > float(roll_max_rel_std):
                        log_error(
                            self._out_log,
                            "GoTo: Roll stack drift unstable "
                            f"label={label} rel_std={rel_std:.3f} "
                            f"vx_std={vx_std:.3f} vy_std={vy_std:.3f} speed={speed:.3f}",
                        )
                        v = None
                    else:
                        source = "stack"
                except Exception as exc:
                    log_error(self._out_log, f"GoTo: Roll {label} stack drift failed", exc)
                    v = None
            if v is None and roll_use_line:
                v = self._autocal_estimate_drift_line_sweep(
                    frames,
                    min_frames=max(2, int(min(roll_frames, 10))),
                    min_duration_s=0.0,
                    min_sources=max(1, int(roll_min_sources)),
                    topk_sources=4,
                    deg_step=0.1,
                    bin_width_px=2.0,
                    topk_bins_for_score=4,
                    use_theil_sen=True,
                    max_shift_px=float(roll_line_max_shift_px),
                    min_resp=0.1,
                )
                if v is not None:
                    source = "line"
            if v is not None:
                speed = float(np.hypot(float(v[0]), float(v[1])))
                if speed > float(roll_max_speed_px_s):
                    log_error(
                        self._out_log,
                        "GoTo: Roll drift speed too high "
                        f"label={label} source={source} speed={speed:.3f} "
                        f"limit={float(roll_max_speed_px_s):.3f}",
                    )
                    v = None
            return v

        log_info(
            self._out_log,
            "GoTo: Roll estimate config "
            f"frames={roll_frames} window={roll_window} "
            f"rate_steps_s={roll_rate_steps_s:.1f} ramp_s={roll_ramp_s:.2f} "
            f"use_stack={int(roll_use_stack)} use_line={int(roll_use_line)} "
            f"max_rel_std={roll_max_rel_std:.3f} max_speed={roll_max_speed_px_s:.1f} "
            f"follow_vx={int(roll_follow_vx)} attempts={roll_rate_attempts} backoff={roll_rate_backoff:.2f} "
            f"capture_timeout_s={roll_timeout_s:.2f} timeout_scale_max={roll_timeout_slow_scale_max:.2f}",
        )

        frames0 = self._autocal_capture_frames(
            n_frames=int(roll_frames),
            timeout_s=float(roll_timeout_s),
            min_dt_s=float(roll_dt_min),
            min_usable_frames=max(0, int(roll_frames)),
            min_usable_sources=max(1, int(roll_min_sources)),
        )
        if len(frames0) < max(2, int(roll_frames)):
            out["status"] = "ERR_DRIFT0_FRAMES"
            log_error(
                self._out_log,
                f"GoTo: Roll estimate insufficient baseline frames ({len(frames0)}/{roll_frames})",
            )
            return out

        v0 = _estimate_drift(frames0, "baseline")
        if v0 is None:
            out["status"] = "ERR_DRIFT0"
            log_error(self._out_log, "GoTo: Roll estimate failed (baseline drift)")
            return out

        slew_sign = 1.0
        if roll_follow_vx and abs(float(v0[0])) > 1e-6:
            slew_sign = 1.0 if float(v0[0]) >= 0.0 else -1.0
        log_info(
            self._out_log,
            "GoTo: Roll slew direction "
            f"v0x={float(v0[0]):+.3f} sign={int(slew_sign):+d}",
        )

        v1: Optional[np.ndarray] = None
        frames1: List[_AutocalFrame] = []
        last_frames_count = 0
        used_slew_rate = 0.0
        rate_try = float(roll_rate_steps_s)
        for attempt_idx in range(int(roll_rate_attempts)):
            rate_signed = float(slew_sign * rate_try)
            attempt_timeout_s = _capture_timeout_for_rate(rate_try)
            log_info(
                self._out_log,
                "GoTo: Roll slew attempt "
                f"{attempt_idx + 1}/{int(roll_rate_attempts)} rate_steps_s={rate_signed:+.2f} "
                f"capture_timeout_s={attempt_timeout_s:.2f}",
            )

            frames_try: List[_AutocalFrame] = []
            self._rate_steps_capture = np.zeros(2, dtype=np.float64)
            try:
                self._autocal_rate_ramp(
                    axis=Axis.AZ,
                    start_rate=0.0,
                    end_rate=rate_signed,
                    ramp_s=roll_ramp_s,
                    ramp_hz=roll_ramp_hz,
                )
                if roll_settle_s > 0.0:
                    time.sleep(float(roll_settle_s))
                frames_try = self._autocal_capture_frames(
                    n_frames=int(roll_frames),
                    timeout_s=float(attempt_timeout_s),
                    min_dt_s=float(roll_dt_min),
                    min_usable_frames=max(0, int(roll_frames)),
                    min_usable_sources=max(1, int(roll_min_sources)),
                    rate_hold_axis=Axis.AZ,
                    rate_hold_steps_s=float(rate_signed),
                    rate_hold_hz=float(roll_ramp_hz),
                )
            finally:
                try:
                    self._autocal_rate_ramp(
                        axis=Axis.AZ,
                        start_rate=rate_signed,
                        end_rate=0.0,
                        ramp_s=roll_ramp_s,
                        ramp_hz=roll_ramp_hz,
                    )
                    if self._rate_mount is not None:
                        self._command_mount_rate(0.0, 0.0)
                finally:
                    moved_steps = self._finish_rate_step_capture()
                    self._apply_rate_steps_to_model(moved_steps)
                    if np.any(np.abs(moved_steps) >= 0.5):
                        log_info(
                            self._out_log,
                            "GoTo: Roll slew accounted model steps "
                            f"az={int(round(float(moved_steps[0]))):+d} "
                            f"alt={int(round(float(moved_steps[1]))):+d}",
                        )

            last_frames_count = len(frames_try)
            if len(frames_try) < max(2, int(roll_frames)):
                log_error(
                    self._out_log,
                    "GoTo: Roll slew attempt insufficient frames "
                    f"attempt={attempt_idx + 1} frames={len(frames_try)}/{roll_frames}",
                )
            else:
                v_try = _estimate_drift(frames_try, f"slew#{attempt_idx + 1}")
                if v_try is not None:
                    v1 = v_try
                    frames1 = frames_try
                    used_slew_rate = rate_signed
                    break

            if attempt_idx + 1 < int(roll_rate_attempts):
                next_rate = float(rate_try) * float(roll_rate_backoff)
                if next_rate < float(roll_rate_min_steps_s):
                    if rate_try <= float(roll_rate_min_steps_s) + 1e-9:
                        log_error(
                            self._out_log,
                            "GoTo: Roll slew retry aborted (already at min rate) "
                            f"rate={rate_try:.2f} min={float(roll_rate_min_steps_s):.2f}",
                        )
                        break
                    next_rate = float(roll_rate_min_steps_s)
                log_info(
                    self._out_log,
                    "GoTo: Roll slew retry with lower rate "
                    f"{rate_try:.2f} -> {next_rate:.2f}",
                )
                rate_try = next_rate

        if v1 is None:
            if last_frames_count < max(2, int(roll_frames)):
                out["status"] = "ERR_DRIFT1_FRAMES"
            else:
                out["status"] = "ERR_DRIFT1"
            log_error(
                self._out_log,
                "GoTo: Roll estimate failed (slew drift) "
                f"attempts={int(roll_rate_attempts)}",
            )
            return out

        dv = np.array([float(v1[0]) - float(v0[0]), float(v1[1]) - float(v0[1])], dtype=np.float64)
        if not np.all(np.isfinite(dv)):
            out["status"] = "ERR_DRIFT_DELTA"
            log_error(self._out_log, "GoTo: Roll estimate failed (non-finite delta)")
            return out

        dv_mag = float(np.hypot(dv[0], dv[1]))
        min_dv = float(params.get("roll_min_delta_px_s", 0.05))
        if dv_mag < min_dv:
            out["status"] = "ERR_DRIFT_DELTA_SMALL"
            log_error(
                self._out_log,
                f"GoTo: Roll estimate failed (delta too small: {dv_mag:.3f} < {min_dv:.3f})",
            )
            return out

        roll_raw_deg = float(math.degrees(math.atan2(float(dv[1]), float(dv[0]))))
        invert_az = bool(getattr(mount_cfg, "invert_az", False))
        slew_rate_for_roll = float(used_slew_rate)
        if invert_az:
            # If AZ direction is inverted at mount level, command sign is opposite to
            # physical +AZ/-AZ motion. Use physical sign for roll orientation.
            slew_rate_for_roll = -slew_rate_for_roll
        roll_branch_deg = _roll_deg_from_drift_delta(dv, slew_rate_for_roll)
        try:
            prev_roll_deg = float(getattr(self._get_state().camera, "roll_deg", float("nan")))
        except Exception:
            prev_roll_deg = float("nan")
        prev_roll_ref_deg = _roll_axis_equivalent_deg(prev_roll_deg) if np.isfinite(prev_roll_deg) else float("nan")
        roll_deg = _roll_equivalent_near_reference_deg(roll_branch_deg, prev_roll_ref_deg)
        roll_axis_deg = _roll_axis_equivalent_deg(roll_deg)
        axis_jump_deg = float("nan")
        if roll_guard_prev_axis and np.isfinite(prev_roll_ref_deg):
            axis_jump_deg = abs(_wrap_deg_180(float(roll_axis_deg) - float(prev_roll_ref_deg)))
            if axis_jump_deg > float(roll_max_axis_jump_deg):
                out["status"] = "ERR_ROLL_AXIS_JUMP"
                out["roll_deg"] = roll_deg
                out["roll_deg_raw"] = roll_branch_deg
                out["roll_axis_deg"] = roll_axis_deg
                out["roll_axis_jump_deg"] = axis_jump_deg
                log_error(
                    self._out_log,
                    "GoTo: Roll estimate rejected (axis jump too large) "
                    f"jump={axis_jump_deg:.3f}deg max={float(roll_max_axis_jump_deg):.3f}deg "
                    f"prev_ref={prev_roll_ref_deg:+.3f}deg roll_axis={roll_axis_deg:+.3f}deg",
                )
                return out
        out["ok"] = True
        out["status"] = "OK"
        out["roll_deg"] = roll_deg
        out["roll_deg_raw"] = roll_branch_deg
        out["roll_axis_deg"] = roll_axis_deg

        self._goto.model.model_roll_deg = _wrap_deg_180(float(roll_deg))
        self._goto.model.model_roll_err_deg = 0.0
        self._goto.model.model_roll_samples = 1
        self._apply_camera_param("roll_deg", float(roll_deg))
        self._publish_state(
            {
                "goto": {
                    "model_camera_roll_deg": float(self._goto.model.model_roll_deg),
                    "model_camera_roll_err_deg": float(self._goto.model.model_roll_err_deg),
                    "model_camera_roll_samples": int(self._goto.model.model_roll_samples),
                }
            }
        )
        log_info(
            self._out_log,
            "GoTo: Roll estimate OK "
            f"slew_rate_cmd={used_slew_rate:+.2f} slew_rate_eff={slew_rate_for_roll:+.2f} invert_az={int(invert_az)} "
            f"roll_raw={roll_raw_deg:+.3f}deg roll_branch={roll_branch_deg:+.3f}deg "
            f"roll_prev={prev_roll_deg:+.3f}deg roll_ref={prev_roll_ref_deg:+.3f}deg "
            f"roll={roll_deg:+.3f}deg roll_axis={roll_axis_deg:+.3f}deg "
            f"axis_jump={axis_jump_deg:+.3f}deg guard={int(roll_guard_prev_axis)} "
            f"v0=[{float(v0[0]):.3f},{float(v0[1]):.3f}] "
            f"v1=[{float(v1[0]):.3f},{float(v1[1]):.3f}] "
            f"dv=[{float(dv[0]):.3f},{float(dv[1]):.3f}]",
        )
        return out

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

    def _autocal_run_platesolve(
        self,
        raw16: np.ndarray,
        *,
        target: Any,
        platesolving_cfg: PlatesolvingConfig,
        sep_cfg: SepConfig,
        observer: ObserverConfig,
        obstime: Time,
        solve_radius_deg: Optional[float] = None,
        solve_gmax: Optional[float] = None,
    ) -> PlatesolvingResult:
        ps_kwargs: Dict[str, Any] = {}
        if solve_radius_deg is not None:
            ps_kwargs["search_radius_deg"] = float(solve_radius_deg)
        if solve_gmax is not None:
            ps_kwargs["gmax"] = float(solve_gmax)
        ps_cfg = replace(platesolving_cfg, **ps_kwargs) if ps_kwargs else platesolving_cfg
        result_frame = np.asarray(raw16)
        diagnostics = self._diagnostics
        if diagnostics is not None:
            diagnostics.save_raw(
                "autocal_platesolve_input",
                result_frame,
                metadata={
                    "target": target,
                    "obstime_unix": float(obstime.unix),
                    "platesolving_config": ps_cfg,
                    "sep_config": sep_cfg,
                    "observer": observer,
                },
            )
        def _progress(stage: str, payload: Dict[str, Any]) -> None:
            self._diagnostics_record(str(stage), **dict(payload or {}))

        result = solve_plate(
            raw16,
            target=target,
            cfg=ps_cfg,
            sep_cfg=sep_cfg,
            observer=observer,
            obstime=obstime,
            progress_cb=_progress,
        )
        result = replace(result, obstime_unix=float(obstime.unix))
        self._diagnostics_record("autocal_platesolve_full", result=result)
        if bool(result.success) and not bool(self._initial_solution_confirmed):
            result, result_frame = self._autocal_confirm_initial_solution(
                result,
                first_frame=result_frame,
                target=target,
                platesolving_cfg=ps_cfg,
                sep_cfg=sep_cfg,
                observer=observer,
            )
            if bool(result.success):
                self._initial_solution_confirmed = True
        debug_jpeg = _render_platesolving_debug_jpeg(
            result_frame,
            list(getattr(result, "overlay", []) or []),
        )
        debug_info = _build_platesolving_debug_info(result)
        if diagnostics is not None and diagnostics.path_str is not None:
            debug_info["diagnostics_dir"] = diagnostics.path_str

        ps_ok = bool(getattr(result, "success", False))
        ps_reason = str(getattr(result, "status", "UNKNOWN"))
        self._diagnostics_record(
            "autocal_platesolve_result",
            success=ps_ok,
            status=ps_reason,
            result=result,
        )
        self._publish_state(
            {
                "platesolving": {
                    "busy": False,
                    "status": PlatesolvingStatus.OK if ps_ok else PlatesolvingStatus.FAIL,
                    "reason": None if ps_ok else ps_reason,
                    "last_ok": ps_ok,
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
        if ps_ok:
            metrics = dict(getattr(result, "metrics", {}) or {})
            az_match, alt_match = platesolving_center_to_altaz_deg(
                float(getattr(result, "center_ra_deg", 0.0)),
                float(getattr(result, "center_dec_deg", 0.0)),
                observer=observer,
                obstime=_platesolving_result_obstime(result, fallback=obstime),
            )
            log_info(
                self._out_log,
                "Platesolving: OK "
                f"status={ps_reason} inliers={int(getattr(result, 'n_inliers', 0))} "
                f"valid={int(metrics.get('validation_inliers', 0) or 0)} "
                f"rms_px={float(getattr(result, 'rms_px', float('nan'))):.3f} "
                f"scale={float(getattr(result, 'scale_arcsec_per_px', 0.0)):.4f}arcsec/px "
                f"target_offset={float(metrics.get('target_offset_deg', float('nan'))):.4f}deg "
                f"match_az={float(az_match):.4f}deg "
                f"match_alt={float(alt_match):.4f}deg",
            )
            self._publish_state(
                {
                    "platesolving_result": result,
                    # This worker performs its own continuity validation and
                    # stores/syncs the solution in the enclosing operation.
                    "platesolving_result_handled": True,
                }
            )
        else:
            metrics = dict(getattr(result, "metrics", {}) or {})
            log_info(
                self._out_log,
                f"Platesolving: ERR status={ps_reason} "
                f"inliers={int(getattr(result, 'n_inliers', 0))} "
                f"valid={int(metrics.get('validation_inliers', 0) or 0)} "
                f"rms_px={float(getattr(result, 'rms_px', float('nan'))):.3f} "
                f"target_offset={float(metrics.get('target_offset_deg', float('nan'))):.4f}deg",
            )
        return result

    def _autocal_confirm_initial_solution(
        self,
        first: PlatesolvingResult,
        *,
        first_frame: np.ndarray,
        target: Any,
        platesolving_cfg: PlatesolvingConfig,
        sep_cfg: SepConfig,
        observer: ObserverConfig,
    ) -> Tuple[PlatesolvingResult, np.ndarray]:
        requested = max(1, int(getattr(platesolving_cfg, "initial_consensus_count", 3)))
        if requested <= 1:
            return first, first_frame
        timeout_s = max(
            0.2,
            float(getattr(platesolving_cfg, "initial_consensus_timeout_s", 8.0)),
        )
        # Skip the frame currently exposed by latest(): it may be the one used
        # by the expensive first solve.  The returned frames are sequence-unique.
        frames = self._autocal_capture_frames(
            n_frames=requested - 1,
            timeout_s=timeout_s,
            skip_frames=1,
            diagnostic_stage="autocal_platesolve_consensus",
        )
        if len(frames) < requested - 1:
            metrics = dict(getattr(first, "metrics", {}) or {})
            metrics.update(
                {
                    "consensus_count": 1.0,
                    "consensus_requested": float(requested),
                }
            )
            return (
                replace(
                    first,
                    success=False,
                    status="INITIAL_CONSENSUS_NO_NEW_FRAME",
                    metrics=metrics,
                ),
                first_frame,
            )

        prior = first
        result_frame = first_frame
        max_pointing = 0.0
        max_scale = 0.0
        max_roll = 0.0
        for idx, frame in enumerate(frames, start=2):
            frame_time = self._autocal_frame_obstime(frame)
            verified = verify_plate_from_prior(
                frame.raw16,
                prior=prior,
                target=target,
                cfg=platesolving_cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=frame_time,
                progress_cb=None,
            )
            consistency = platesolving_solutions_consistent(
                first,
                verified,
                observer=observer,
                pointing_tol_arcsec=float(
                    getattr(platesolving_cfg, "consensus_pointing_tol_arcsec", 30.0)
                ),
                scale_tol_frac=float(
                    getattr(platesolving_cfg, "consensus_scale_tol_frac", 0.02)
                ),
                roll_tol_deg=float(
                    getattr(platesolving_cfg, "consensus_roll_tol_deg", 3.0)
                ),
            )
            self._diagnostics_record(
                "autocal_platesolve_consensus",
                confirmation_index=int(idx),
                result=verified,
                consistency=consistency,
            )
            if not bool(verified.success) or not bool(consistency.get("ok", False)):
                metrics = dict(getattr(verified, "metrics", {}) or {})
                metrics.update(
                    {
                        "consensus_count": float(idx - 1),
                        "consensus_requested": float(requested),
                        "consensus_pointing_arcsec": float(
                            consistency.get("pointing_arcsec", float("inf"))
                        ),
                        "consensus_scale_frac": float(
                            consistency.get("scale_frac", float("inf"))
                        ),
                        "consensus_roll_deg": float(
                            consistency.get("roll_deg", float("inf"))
                        ),
                    }
                )
                log_error(
                    self._out_log,
                    "Platesolving: initial independent confirmation rejected "
                    f"frame={idx}/{requested} status={verified.status} "
                    f"mount_delta={float(consistency.get('pointing_arcsec', float('inf'))):.2f}arcsec",
                )
                return (
                    replace(
                        verified,
                        success=False,
                        status="INITIAL_CONSENSUS_MISMATCH",
                        metrics=metrics,
                    ),
                    frame.raw16,
                )
            max_pointing = max(max_pointing, float(consistency["pointing_arcsec"]))
            max_scale = max(max_scale, float(consistency["scale_frac"]))
            max_roll = max(max_roll, float(consistency["roll_deg"]))
            prior = verified
            result_frame = frame.raw16
            log_info(
                self._out_log,
                "Platesolving: fast independent confirmation "
                f"{idx}/{requested} inliers={verified.n_inliers} rms_px={verified.rms_px:.3f} "
                f"mount_delta={float(consistency['pointing_arcsec']):.2f}arcsec",
            )

        metrics = dict(getattr(prior, "metrics", {}) or {})
        metrics.update(
            {
                "consensus_count": float(requested),
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
                guides=list(first.guides),
                metrics=metrics,
            ),
            result_frame,
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
            self._command_mount_rate(az_rate, alt_rate)
            return
        steps = max(1, int(round(ramp_s * ramp_hz)))
        for i in range(1, steps + 1):
            if self._op_cancel.is_set():
                break
            f = float(i) / float(steps)
            rate = float(start_rate) + (float(end_rate) - float(start_rate)) * f
            az_rate, alt_rate = self._autocal_axis_rates(axis, rate)
            self._command_mount_rate(az_rate, alt_rate)
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
            rate_hold_axis=axis,
            rate_hold_steps_s=float(rate_steps_s),
            rate_hold_hz=float(ramp_hz),
        )

        self._autocal_rate_ramp(
            axis=axis,
            start_rate=float(rate_steps_s),
            end_rate=0.0,
            ramp_s=ramp_s,
            ramp_hz=ramp_hz,
        )
        if self._rate_mount is not None:
            self._command_mount_rate(0.0, 0.0)

        if len(frames) < 2:
            return _AutocalJResult(col=None, ok_count=0, resp_low=0, missing_frames=1)

        base = frames[0]
        cols: List[np.ndarray] = []
        resp_low = 0
        for fr in frames[1:]:
            base_t = self._autocal_frame_time_s(base)
            fr_t = self._autocal_frame_time_s(fr)
            dt = float(fr_t - base_t)
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
            # Convert shifts to +y up to match drift_pix convention.
            dp = np.array([-dx, dy], dtype=np.float64) - drift_pix * dt
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

    def _goto_calibrate_right_scan_blocking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": False,
            "status": "RUNNING",
            "steps_done": 0,
            "solves_ok": 0,
            "fit_updates": 0,
            "n_samples": 0,
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
        observer = _observer_without_refraction(self._get_observer())

        ps_overrides: Dict[str, Any] = {}
        if "N_seed" in params:
            ps_overrides["N_seed"] = int(params.get("N_seed"))
        if "min_inliers" in params:
            ps_overrides["min_inliers"] = int(params.get("min_inliers"))
        if ps_overrides:
            try:
                platesolving_cfg = replace(platesolving_cfg, **ps_overrides)
            except Exception as exc:
                log_error(
                    self._out_log,
                    f"GoTo: invalid right-scan platesolving overrides ({ps_overrides})",
                    exc,
                )

        scan_steps = int(params.get("scan_steps", 10))
        scan_step_microsteps = int(params.get("scan_step_microsteps", 300))
        scan_ps_radius_deg = float(params.get("scan_ps_radius_deg", 0.5))
        scan_ps_gmax = float(params.get("scan_ps_gmax", getattr(platesolving_cfg, "gmax", 15.0)))
        scan_theta_tol_deg = float(params.get("scan_theta_tol_deg", 20.0))
        scan_fit_min_samples = int(params.get("scan_fit_min_samples", 3))
        scan_fit_ridge = float(params.get("scan_fit_ridge", 1e-12))
        scan_sync_latest = bool(params.get("scan_sync_latest", True))
        scan_direction = str(params.get("scan_direction", "right")).strip().lower()
        delay_us = int(
            params.get(
                "delay_us",
                getattr(goto_cfg, "slew_delay_us", getattr(goto_cfg, "slew_delay_us_az", 1200)),
            )
        )
        settle_s = float(params.get("settle_s", goto_cfg.settle_s))

        if scan_steps < 1:
            out["status"] = "ERR_SCAN_STEPS"
            return out
        if scan_step_microsteps < 1:
            out["status"] = "ERR_SCAN_STEP_SIZE"
            return out
        if not np.isfinite(scan_ps_radius_deg) or scan_ps_radius_deg <= 0.0:
            out["status"] = "ERR_SCAN_PS_RADIUS"
            return out
        if not np.isfinite(scan_ps_gmax) or scan_ps_gmax <= 0.0:
            out["status"] = "ERR_SCAN_PS_GMAX"
            return out
        if not np.isfinite(scan_theta_tol_deg) or scan_theta_tol_deg <= 0.0:
            out["status"] = "ERR_SCAN_THETA_TOL"
            return out
        if scan_fit_min_samples < 2:
            scan_fit_min_samples = 2
        if not np.isfinite(scan_fit_ridge) or scan_fit_ridge < 0.0:
            scan_fit_ridge = 1e-12
        if delay_us < 50:
            delay_us = 50
        if settle_s < 0.0:
            settle_s = 0.0

        scan_axis: Optional[Axis] = None
        move_dir = 0
        if scan_direction in ("right", "+", "az+"):
            scan_axis = Axis.AZ
            move_dir = +1
        elif scan_direction in ("left", "-", "az-"):
            scan_axis = Axis.AZ
            move_dir = -1
        elif scan_direction in ("up", "alt+"):
            scan_axis = Axis.ALT
            move_dir = +1
        elif scan_direction in ("down", "alt-"):
            scan_axis = Axis.ALT
            move_dir = -1
        else:
            out["status"] = "ERR_SCAN_DIRECTION"
            return out

        if scan_axis == Axis.AZ and bool(getattr(mount_cfg, "invert_az", False)):
            move_dir *= -1
        if scan_axis == Axis.ALT and bool(getattr(mount_cfg, "invert_alt", False)):
            move_dir *= -1
        signed_step = float(move_dir * scan_step_microsteps)

        def _coerce_altaz(raw_target: Any) -> Optional[Tuple[float, float]]:
            try:
                if isinstance(raw_target, dict):
                    if "az_deg" in raw_target and "alt_deg" in raw_target:
                        az = float(raw_target.get("az_deg"))
                        alt = float(raw_target.get("alt_deg"))
                    elif "az" in raw_target and "alt" in raw_target:
                        az = float(raw_target.get("az"))
                        alt = float(raw_target.get("alt"))
                    else:
                        return None
                else:
                    arr = np.asarray(raw_target, dtype=np.float64).reshape(-1)
                    if arr.size < 2:
                        return None
                    az = float(arr[0])
                    alt = float(arr[1])
            except Exception:
                return None
            if not np.isfinite(az) or not np.isfinite(alt):
                return None
            az = _wrap_deg_360(az)
            alt = float(np.clip(alt, goto_cfg.alt_min_deg, goto_cfg.alt_max_deg))
            return (az, alt)

        def _current_altaz() -> Optional[Tuple[float, float]]:
            model_altaz = self._goto.model.current_az_alt_deg()
            if model_altaz is not None:
                parsed = _coerce_altaz(model_altaz)
                if parsed is not None:
                    return parsed
            st_now = self._get_state()
            if bool(getattr(st_now.goto, "pointing_valid", False)):
                parsed = _coerce_altaz(
                    {
                        "az_deg": float(getattr(st_now.goto, "pointing_az_deg", 0.0)),
                        "alt_deg": float(getattr(st_now.goto, "pointing_alt_deg", 0.0)),
                    }
                )
                if parsed is not None:
                    return parsed
            if bool(getattr(st_now.platesolving, "last_ok", False)):
                try:
                    az_alt = platesolving_center_to_altaz_deg(
                        float(getattr(st_now.platesolving, "center_ra_deg", 0.0)),
                        float(getattr(st_now.platesolving, "center_dec_deg", 0.0)),
                        observer=observer,
                        obstime=Time.now(),
                    )
                    parsed = _coerce_altaz(az_alt)
                    if parsed is not None:
                        return parsed
                except Exception as exc:
                    log_error(self._out_log, "GoTo: right-scan failed to decode last platesolve center", exc)
            if bool(getattr(self._goto.model, "synced", False)):
                try:
                    parsed = _coerce_altaz(self._goto.model.predict_az_alt_deg())
                    if parsed is not None:
                        return parsed
                except Exception as exc:
                    log_error(
                        self._out_log,
                        "GoTo: right-scan failed to predict model AltAz",
                        exc,
                        throttle_s=5.0,
                        throttle_key="goto_right_scan_model_altaz",
                    )
            return None

        def _theta_dist_mod180(a_deg: float, b_deg: float) -> float:
            d = abs((float(a_deg) - float(b_deg)) % 180.0)
            return float(min(d, 180.0 - d))

        theta_ref: Optional[float] = None
        continuity_steps: Optional[np.ndarray] = None
        continuity_altaz: Optional[np.ndarray] = None
        continuity_roll: Optional[float] = None

        def _solve_near_altaz(az_deg: float, alt_deg: float, *, label: str, step_idx: int) -> Optional[PlatesolvingResult]:
            if self._op_cancel.is_set():
                out["status"] = "CANCELLED"
                return None
            frames = self._autocal_capture_frames(n_frames=1, timeout_s=1.5)
            if not frames:
                out["status"] = f"ERR_SCAN_NO_FRAME_STEP_{step_idx}"
                return None
            frame = frames[0]
            obstime = self._autocal_frame_obstime(frame)
            try:
                target_icrs = parse_target_to_icrs(
                    {"az_deg": float(az_deg), "alt_deg": float(alt_deg)},
                    observer=observer,
                    obstime=obstime,
                ).icrs
                target = (float(target_icrs.ra.deg), float(target_icrs.dec.deg))
            except Exception as exc:
                out["status"] = f"ERR_SCAN_TARGET_STEP_{step_idx}"
                log_error(self._out_log, "GoTo: right-scan failed to transform AltAz -> ICRS", exc)
                return None

            log_info(
                self._out_log,
                "GoTo: right-scan platesolve "
                f"{label} step={step_idx} target_az={float(az_deg):.3f} target_alt={float(alt_deg):.3f} "
                f"target_ra={float(target[0]):.6f} target_dec={float(target[1]):.6f} "
                f"radius={scan_ps_radius_deg:.2f}",
            )
            result = self._autocal_run_platesolve(
                frame.raw16,
                target=target,
                platesolving_cfg=platesolving_cfg,
                sep_cfg=sep_cfg,
                observer=observer,
                obstime=obstime,
                solve_radius_deg=scan_ps_radius_deg,
                solve_gmax=scan_ps_gmax,
            )
            if not bool(getattr(result, "success", False)):
                out["status"] = f"ERR_SCAN_PLATESOLVING_STEP_{step_idx}"
                return None
            return result

        def _consume_solution(
            result: PlatesolvingResult,
            *,
            step_idx: int,
            update_model: bool = True,
        ) -> bool:
            nonlocal theta_ref, continuity_steps, continuity_altaz, continuity_roll
            theta = float(getattr(result, "theta_deg", float("nan")))
            if not np.isfinite(theta):
                out["status"] = f"ERR_SCAN_THETA_STEP_{step_idx}"
                return False
            if theta_ref is None:
                theta_ref = theta
            else:
                dtheta = _theta_dist_mod180(theta, float(theta_ref))
                if dtheta > float(scan_theta_tol_deg):
                    out["status"] = f"ERR_SCAN_THETA_INCONSISTENT_STEP_{step_idx}"
                    log_error(
                        self._out_log,
                        "GoTo: right-scan theta inconsistent "
                        f"step={step_idx} theta={theta:.3f} theta_ref={float(theta_ref):.3f} dtheta={dtheta:.3f}",
                    )
                    return False

            solve_obstime = _platesolving_result_obstime(result)
            az_alt = platesolving_center_to_altaz_deg(
                float(result.center_ra_deg),
                float(result.center_dec_deg),
                observer=observer,
                obstime=solve_obstime,
            )
            roll_sample = self._roll_sample_from_solution(
                result,
                observer=observer,
                obstime=solve_obstime,
            )
            if continuity_steps is None or continuity_altaz is None:
                continuity = {"ok": True, "has_reference": False}
            else:
                continuity = self._check_manual_sample_continuity(
                    az_alt,
                    roll_deg=roll_sample,
                    context=f"right-scan step={step_idx}",
                    reference_steps=continuity_steps,
                    reference_az_alt_deg=continuity_altaz,
                    reference_roll_deg=continuity_roll,
                )
            if not bool(continuity.get("ok", False)):
                self._invalidate_platesolving_after_continuity_rejection(result)
                out["status"] = f"ERR_SCAN_CONTINUITY_STEP_{step_idx}"
                return False

            continuity_steps = self._goto.model.steps_est.copy()
            continuity_altaz = np.asarray(az_alt, dtype=np.float64).copy()
            continuity_roll = float(roll_sample) if np.isfinite(roll_sample) else None

            if not update_model:
                return True

            n_samples = int(
                self._goto.model.add_manual_sample(
                    az_alt,
                    roll_deg=roll_sample,
                    source=(self._diagnostics.path_str if self._diagnostics is not None else None),
                )
            )
            out["n_samples"] = n_samples
            out["solves_ok"] = int(out["solves_ok"]) + 1
            self._publish_state(
                {
                    "goto": {
                        "manual_samples": n_samples,
                        "autocal_az_deg": float(az_alt[0]),
                        "autocal_alt_deg": float(az_alt[1]),
                        "autocal_radius_deg": float(scan_ps_radius_deg),
                    }
                }
            )

            fit_ok = bool(
                self._goto.model.fit_J_from_manual_samples(
                    min_samples=int(max(2, scan_fit_min_samples)),
                    ridge=float(scan_fit_ridge),
                )
            )
            if fit_ok:
                out["fit_updates"] = int(out["fit_updates"]) + 1
                if scan_sync_latest:
                    _ = bool(self._goto.model.sync_from_latest_manual_sample())
                self._publish_j_matrix_state()
                self._publish_state({"goto": {"synced": bool(getattr(self._goto.model, "synced", False))}})
                self._log_model_fit_state(prefix="GoTo: right-scan fit update")
            return True

        log_info(
            self._out_log,
            "GoTo: right-scan calibration start "
            f"steps={scan_steps} microsteps={scan_step_microsteps} axis={scan_axis.value} dir={scan_direction} "
            f"cmd_dir={move_dir:+d} ps_radius={scan_ps_radius_deg:.2f} theta_tol={scan_theta_tol_deg:.1f}",
        )

        altaz0 = _current_altaz()
        if altaz0 is None:
            out["status"] = "ERR_SCAN_NO_CURRENT"
            return out

        self._publish_state({"goto": {"status": GotoStatus.RUNNING, "reason": "CALIBRATE_RIGHT_SCAN_BASE"}})
        base_result = _solve_near_altaz(float(altaz0[0]), float(altaz0[1]), label="base", step_idx=0)
        if base_result is None:
            return out
        if not _consume_solution(base_result, step_idx=0, update_model=False):
            return out

        for step_idx in range(1, int(scan_steps) + 1):
            if self._op_cancel.is_set():
                out["status"] = "CANCELLED"
                return out

            self._publish_state(
                {"goto": {"status": GotoStatus.RUNNING, "reason": f"CALIBRATE_RIGHT_SCAN_MOVE_{step_idx}"}}
            )
            self._exec_steps(
                self._move_steps,
                scan_axis,
                signed_steps=signed_step,
                delay_us=int(delay_us),
            )
            try:
                self._stop_mount()
            except Exception as exc:
                log_error(
                    self._out_log,
                    "GoTo: right-scan stop failed after scan move",
                    exc,
                    throttle_s=5.0,
                    throttle_key="goto_right_scan_stop_after_move",
                )
            if settle_s > 0.0:
                time.sleep(float(settle_s))

            altaz_est = _current_altaz()
            if altaz_est is None:
                out["status"] = f"ERR_SCAN_NO_CURRENT_STEP_{step_idx}"
                return out

            self._publish_state(
                {"goto": {"status": GotoStatus.RUNNING, "reason": f"CALIBRATE_RIGHT_SCAN_SOLVE_{step_idx}"}}
            )
            result = _solve_near_altaz(float(altaz_est[0]), float(altaz_est[1]), label="scan", step_idx=step_idx)
            if result is None:
                return out
            if not _consume_solution(result, step_idx=step_idx):
                return out
            out["steps_done"] = int(step_idx)

        out["ok"] = True
        out["status"] = "OK"
        self._publish_state(
            {
                "goto": {
                    "status": GotoStatus.OK,
                    "reason": None,
                    "autocal_last_ok": True,
                    "autocal_status": GotoAutocalStatus.OK,
                    "autocal_reason": "RIGHT_SCAN_READY",
                    "synced": bool(getattr(self._goto.model, "synced", False)),
                }
            }
        )
        log_info(
            self._out_log,
            "GoTo: right-scan calibration OK "
            f"steps_done={out['steps_done']} solves_ok={out['solves_ok']} "
            f"fit_updates={out['fit_updates']} samples={out['n_samples']}",
        )
        return out

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
        observer = _observer_without_refraction(self._get_observer())

        ps_overrides: Dict[str, Any] = {}
        if "N_seed" in params:
            ps_overrides["N_seed"] = int(params.get("N_seed"))
        if "min_inliers" in params:
            ps_overrides["min_inliers"] = int(params.get("min_inliers"))
        if ps_overrides:
            try:
                platesolving_cfg = replace(platesolving_cfg, **ps_overrides)
            except Exception as exc:
                log_error(
                    self._out_log,
                    f"GoTo: invalid autocal platesolving overrides ({ps_overrides})",
                    exc,
                )

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
        drift_capture_timeout_s = float(params.get("drift_capture_timeout_s", 12.0))
        drift_line_topk_sources = int(params.get("drift_line_topk_sources", 4))
        drift_line_min_sources = int(params.get("drift_line_min_sources", 1))
        drift_line_min_frames = int(params.get("drift_line_min_frames", 10))
        drift_line_min_duration_s = float(params.get("drift_line_min_duration_s", 5.0))
        drift_line_deg_step = float(params.get("drift_line_deg_step", 0.1))
        drift_line_bin_width_px = float(params.get("drift_line_bin_width_px", 2.0))
        drift_line_topk_bins = int(params.get("drift_line_topk_bins", 4))
        drift_line_use_theil_sen = bool(params.get("drift_line_use_theil_sen", True))
        drift_line_max_shift_px = float(params.get("drift_line_max_shift_px", 50.0))
        drift_line_min_resp = float(params.get("drift_line_min_resp", 0.1))
        drift_stack_enable = bool(params.get("drift_stack_enable", True))
        drift_stack_window = int(params.get("drift_stack_window", 60))
        drift_stack_median_k = int(params.get("drift_stack_median_k", 3))
        drift_stack_smooth_k = int(params.get("drift_stack_smooth_k", 20))
        drift_stack_vmax_px_s = float(params.get("drift_stack_vmax_px_s", 30.0))
        drift_stack_margin_px = float(params.get("drift_stack_margin_px", 10.0))
        drift_stack_max_shift_cap = int(params.get("drift_stack_max_shift_cap", 200))
        drift_stack_profile_q = params.get("drift_stack_profile_q", None)
        drift_stack_use_subpixel = bool(params.get("drift_stack_use_subpixel", False))
        drift_refract_enable = False
        pointing_method = str(params.get("pointing_method", "horiz_drift")).strip().lower()
        drift_pointing_omega = float(
            params.get(
                "drift_pointing_omega_arcsec_s",
                params.get("drift_pointing_omega_deg_s", 15.041),
            )
        )
        drift_capture_timeout_eff = float(drift_capture_timeout_s)
        if drift_line_min_frames > 0 and drift_dt_min > 0.0:
            drift_capture_timeout_eff += float(drift_dt_min) * 2.0

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
        autocal_solve_radius_deg = float(
            params.get("autocal_solve_radius_deg", getattr(platesolving_cfg, "search_radius_deg", 1.0) or 1.0)
        )
        autocal_solve_gmax = float(params.get("autocal_solve_gmax", getattr(platesolving_cfg, "gmax", 15.0)))
        autocal_ps_mode_raw = str(params.get("autocal_ps_mode", "drift")).strip().lower()
        if autocal_ps_mode_raw in ("manual", "manual_altaz", "manual-altaz"):
            autocal_ps_mode = "manual_altaz"
        elif autocal_ps_mode_raw in ("current", "current_altaz", "current-altaz", "live", "registered"):
            autocal_ps_mode = "current_altaz"
        else:
            autocal_ps_mode = "drift"
        autocal_ps_target_raw = params.get("autocal_ps_target", None)
        manual_only = bool(params.get("manual_only", True))
        tune_exposure = _autocal_should_tune_exposure(
            params,
            autocal_ps_mode=autocal_ps_mode,
        )

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
        if autocal_solve_radius_deg <= 0.0 or not np.isfinite(autocal_solve_radius_deg):
            out["status"] = "ERR_AUTOCAL_PS_PARAMS"
            return out
        if autocal_solve_gmax <= 0.0 or not np.isfinite(autocal_solve_gmax):
            out["status"] = "ERR_AUTOCAL_PS_PARAMS"
            return out

        if drift_stack_window < 1:
            drift_stack_window = 1
        if drift_stack_profile_q is not None:
            try:
                drift_stack_profile_q = float(drift_stack_profile_q)
                if not np.isfinite(drift_stack_profile_q):
                    drift_stack_profile_q = None
            except (TypeError, ValueError):
                drift_stack_profile_q = None

        log_info(
            self._out_log,
            "GoTo: AutoCal config "
            f"stars=[{target_star_min},{target_star_max}] sat_max={target_sat_max:.3f} "
            f"exp_ms=[{exp_min_ms:.1f},{exp_max_ms:.1f}] gain=[{gain_min},{gain_max}] "
            f"drift_frames={drift_frames} drift_dt_min={drift_dt_min:.2f} "
            f"drift_capture_timeout_s={drift_capture_timeout_eff:.2f} "
            f"drift_line_topk={drift_line_topk_sources} min_sources={drift_line_min_sources} "
            f"drift_line_min_frames={drift_line_min_frames} min_dt={drift_line_min_duration_s:.2f} "
            f"drift_line_deg_step={drift_line_deg_step:.2f} bin_width={drift_line_bin_width_px:.2f} "
            f"drift_line_topk_bins={drift_line_topk_bins} "
            f"drift_line_max_shift_px={drift_line_max_shift_px:.1f} min_resp={drift_line_min_resp:.2f} "
            "drift_timebase=capture_ts "
            f"drift_stack_enable={int(bool(drift_stack_enable))} window={int(drift_stack_window)} "
            f"vmax={drift_stack_vmax_px_s:.1f} "
            f"drift_refract_enable={int(bool(drift_refract_enable))} "
            f"autocal_ps_radius_deg={autocal_solve_radius_deg:.2f} autocal_ps_gmax={autocal_solve_gmax:.2f} "
            f"autocal_ps_mode={autocal_ps_mode} "
            f"pointing_method={pointing_method} omega_arcsec_s={drift_pointing_omega:.3f} "
            f"jcal_rate_scale={jcal_rate_scale:.2f} "
            f"jcal_ramp_s={jcal_ramp_s:.2f} ramp_hz={jcal_ramp_hz:.1f} "
            f"jcal_plateau_s={jcal_plateau_s:.2f} frames={jcal_plateau_frames} "
            f"plateau_min_dt_s={jcal_plateau_min_dt_s:.2f} skip_frames={jcal_plateau_skip_frames} "
            f"jcal_probe_s={jcal_probe_s:.2f} probe_scale={jcal_probe_scale:.2f} "
            f"jcal_min_resp={jcal_min_resp:.2f} jcal_max_shift_px={jcal_max_shift_px:.1f}",
        )
        if drift_stack_enable and drift_frames <= drift_stack_window:
            log_info(
                self._out_log,
                "GoTo: AutoCal drift stack will be skipped "
                f"(drift_frames={drift_frames} window={drift_stack_window})",
            )

        plate_scale_rad = float(platesolving_cfg.pixel_size_m) / float(platesolving_cfg.focal_m)
        if not np.isfinite(plate_scale_rad) or plate_scale_rad <= 0.0:
            out["status"] = "ERR_BAD_PLATE_SCALE"
            return out

        tuned = not bool(tune_exposure)
        if tune_exposure:
            self._publish_state(
                {"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "EXPOSURE_TUNE"}}
            )
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
        else:
            log_info(
                self._out_log,
                "GoTo: manual plate solve preserving operator exposure/gain",
            )
        cam_cfg = self._get_camera_cfg()
        log_info(
            self._out_log,
            f"GoTo: AutoCal exposure {'tuned' if tune_exposure else 'preserved'} "
            f"exp_ms={float(getattr(cam_cfg, 'exp_ms', 0.0)):.2f} "
            f"gain={int(getattr(cam_cfg, 'gain', 0))}",
        )

        def _coerce_altaz_target(raw_target: Any) -> Optional[Tuple[float, float]]:
            az: Optional[float] = None
            alt: Optional[float] = None
            if isinstance(raw_target, dict):
                if "az_deg" in raw_target and "alt_deg" in raw_target:
                    az = float(raw_target.get("az_deg"))
                    alt = float(raw_target.get("alt_deg"))
                elif "az" in raw_target and "alt" in raw_target:
                    az = float(raw_target.get("az"))
                    alt = float(raw_target.get("alt"))
            elif isinstance(raw_target, (tuple, list)) and len(raw_target) >= 2:
                az = float(raw_target[0])
                alt = float(raw_target[1])
            if az is None or alt is None:
                return None
            if not np.isfinite(az) or not np.isfinite(alt):
                return None
            az = _wrap_deg_360(float(az))
            alt = float(np.clip(float(alt), goto_cfg.alt_min_deg, goto_cfg.alt_max_deg))
            return (az, alt)

        def _resolve_current_altaz_target() -> Optional[Tuple[float, float]]:
            try:
                model_altaz = self._goto.model.current_az_alt_deg()
            except Exception:
                model_altaz = None
            if model_altaz is not None:
                coerced = _coerce_altaz_target(model_altaz)
                if coerced is not None:
                    return coerced
            st_now = self._get_state()
            if bool(getattr(st_now.goto, "pointing_valid", False)):
                return _coerce_altaz_target(
                    {
                        "az_deg": float(getattr(st_now.goto, "pointing_az_deg", 0.0)),
                        "alt_deg": float(getattr(st_now.goto, "pointing_alt_deg", 0.0)),
                    }
                )
            return None

        if autocal_ps_mode != "drift":
            if autocal_ps_mode == "manual_altaz":
                target_base = _coerce_altaz_target(autocal_ps_target_raw)
            else:
                target_base = _resolve_current_altaz_target()
            if target_base is None:
                out["status"] = "ERR_AUTOCAL_PS_TARGET"
                log_error(
                    self._out_log,
                    f"GoTo: AutoCal invalid target for mode={autocal_ps_mode}",
                )
                return out

            az_hat = float(target_base[0])
            alt_hat = float(target_base[1])
            out["pointing_estimate"] = {"az_deg": az_hat, "alt_deg": alt_hat, "radius_deg": 1.0}
            self._publish_state(
                {
                    "goto": {
                        "autocal_az_deg": az_hat,
                        "autocal_alt_deg": alt_hat,
                        "autocal_radius_deg": 1.0,
                        "autocal_status": GotoAutocalStatus.RUNNING,
                        "autocal_reason": "PLATESOLVING_LOOP",
                    }
                }
            )

            jitter_seq = [
                (0.0, 0.0),
                (jitter_deg, 0.0),
                (-jitter_deg, 0.0),
                (0.0, jitter_deg),
                (0.0, -jitter_deg),
            ]
            platesolving_result: Optional[PlatesolvingResult] = None
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
                changed = False
                if tune_exposure:
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

                solve_obstime = self._autocal_frame_obstime(fr)
                jitter = jitter_seq[attempts % len(jitter_seq)]
                target = {
                    "az_deg": _wrap_deg_360(az_hat + float(jitter[0])),
                    "alt_deg": float(np.clip(alt_hat + float(jitter[1]), goto_cfg.alt_min_deg, goto_cfg.alt_max_deg)),
                }
                if attempts == 0:
                    log_info(
                        self._out_log,
                        "GoTo: AutoCal platesolve first target "
                        f"mode={autocal_ps_mode} "
                        f"az={float(target['az_deg']):.3f}deg "
                        f"alt={float(target['alt_deg']):.3f}deg "
                        f"radius={1.0:.3f}deg "
                        f"jitter=({float(jitter[0]):.3f},{float(jitter[1]):.3f})",
                    )
                platesolving_result = self._autocal_run_platesolve(
                    fr.raw16,
                    target=target,
                    platesolving_cfg=platesolving_cfg,
                    sep_cfg=sep_cfg,
                    observer=observer,
                    obstime=solve_obstime,
                    solve_radius_deg=autocal_solve_radius_deg,
                    solve_gmax=autocal_solve_gmax,
                )
                attempts += 1
                if bool(getattr(platesolving_result, "success", False)):
                    break

            if platesolving_result is None or not bool(getattr(platesolving_result, "success", False)):
                out["status"] = "ERR_PLATESOLVING"
                out["platesolving_result"] = platesolving_result
                return out

            out["platesolving_result"] = platesolving_result

            if manual_only:
                solve_obstime = _platesolving_result_obstime(platesolving_result)
                az_alt = platesolving_center_to_altaz_deg(
                    float(platesolving_result.center_ra_deg),
                    float(platesolving_result.center_dec_deg),
                    observer=observer,
                    obstime=solve_obstime,
                )
                roll_sample = self._roll_sample_from_solution(
                    platesolving_result,
                    observer=observer,
                    obstime=solve_obstime,
                )
                continuity = self._check_manual_sample_continuity(
                    az_alt,
                    roll_deg=roll_sample,
                    context=f"autocal mode={autocal_ps_mode}",
                )
                if not bool(continuity.get("ok", False)):
                    out["platesolving_result"] = self._invalidate_platesolving_after_continuity_rejection(
                        platesolving_result
                    )
                    out["status"] = "ERR_SAMPLE_CONTINUITY"
                    return out
                n_samples = int(
                    self._goto.model.add_manual_sample(
                        az_alt,
                        roll_deg=roll_sample,
                        source=(self._diagnostics.path_str if self._diagnostics is not None else None),
                    )
                )
                self._publish_state(
                    {
                        "goto": {
                            "status": GotoStatus.OK,
                            "reason": None,
                            "autocal_last_ok": True,
                            "autocal_status": GotoAutocalStatus.OK,
                            "autocal_reason": "MANUAL_SAMPLE",
                            "manual_samples": n_samples,
                        }
                    }
                )
                out["ok"] = True
                out["status"] = "OK_MANUAL_SAMPLE"
                out["manual_samples"] = n_samples
                return out

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

        self._publish_state({"goto": {"autocal_status": GotoAutocalStatus.RUNNING, "autocal_reason": "DRIFT"}})
        drift_frames_list = self._autocal_capture_frames(
            n_frames=drift_frames,
            timeout_s=drift_capture_timeout_eff,
            min_dt_s=drift_dt_min,
            min_usable_frames=drift_line_min_frames,
            min_usable_sources=drift_line_min_sources,
            diagnostic_stage="autocal_drift_frames",
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
            ref_t = self._autocal_frame_time_s(drift_frames_list[0])
            for idx, fr in enumerate(drift_frames_list):
                fr_t = self._autocal_frame_time_s(fr)
                dt = float(fr_t - ref_t)
                log_info(
                    self._out_log,
                    "GoTo: AutoCal drift frame sources "
                    f"idx={idx} dt={dt:.3f}s stars={fr.star_count} "
                    f"top3={self._format_autocal_sources(fr.top_sources)}",
                )
        best_drift_frame = self._autocal_pick_best_frame(drift_frames_list)

        drift_pix: Optional[np.ndarray] = None
        drift_method = "line_sweep"
        # The stack estimator needs more captured frames than its window.
        # AutoCal already reports up-front when the requested capture cannot
        # satisfy that requirement, so do not call it merely to emit a false
        # error before falling back to the line-sweep estimator.
        if drift_stack_enable and len(drift_frames_list) > drift_stack_window:
            drift_pix = self._autocal_estimate_drift_stack(
                drift_frames_list,
                window=drift_stack_window,
                median_k=drift_stack_median_k,
                smooth_k=drift_stack_smooth_k,
                vmax_px_s=drift_stack_vmax_px_s,
                margin_px=drift_stack_margin_px,
                max_shift_cap=drift_stack_max_shift_cap,
                profile_q=drift_stack_profile_q,
                use_subpixel=drift_stack_use_subpixel,
            )
            if drift_pix is not None:
                drift_method = "stack"
        if drift_pix is None:
            drift_pix = self._autocal_estimate_drift_line_sweep(
                drift_frames_list,
                min_frames=drift_line_min_frames,
                min_duration_s=drift_line_min_duration_s,
                min_sources=drift_line_min_sources,
                topk_sources=drift_line_topk_sources,
                deg_step=drift_line_deg_step,
                bin_width_px=drift_line_bin_width_px,
                topk_bins_for_score=drift_line_topk_bins,
                use_theil_sen=drift_line_use_theil_sen,
                max_shift_px=drift_line_max_shift_px,
                min_resp=drift_line_min_resp,
            )
        if drift_pix is None:
            out["status"] = "ERR_DRIFT"
            log_error(self._out_log, "GoTo: AutoCal drift estimate failed")
            return out
        drift_pix_raw = np.asarray(drift_pix, dtype=np.float64)
        roll_deg = 0.0
        try:
            roll_deg = float(getattr(self._get_state().camera, "roll_deg", 0.0))
        except Exception:
            roll_deg = 0.0
        drift_pix_corr = _apply_roll_to_drift(drift_pix_raw, roll_deg)

        use_horizontal = pointing_method in (
            "horiz_drift",
            "horizontal",
            "horiz",
            "drift",
            "drift_horiz",
            "horizontal_drift",
            "auto",
        )
        use_roll_corr = bool(use_horizontal and np.isfinite(roll_deg) and abs(roll_deg) > 1e-9)
        drift_pix_use = drift_pix_corr if use_roll_corr else drift_pix_raw
        drift_pix = np.asarray(drift_pix_use, dtype=np.float64)

        log_info(
            self._out_log,
            "GoTo: AutoCal drift vectors "
            f"raw=[{float(drift_pix_raw[0]):.3f},{float(drift_pix_raw[1]):.3f}] "
            f"corr=[{float(drift_pix_corr[0]):.3f},{float(drift_pix_corr[1]):.3f}] "
            f"roll_deg={float(roll_deg):+.3f} use_roll={int(use_roll_corr)}",
        )
        self._diagnostics_record(
            "autocal_drift_vectors",
            method=drift_method,
            raw_velocity_px_s_xy_up=drift_pix_raw,
            roll_corrected_velocity_px_s_xy_up=drift_pix_corr,
            selected_velocity_px_s_xy_up=drift_pix,
            camera_roll_deg=float(roll_deg),
            roll_correction_used=bool(use_roll_corr),
        )

        out["drift_pix"] = drift_pix
        out["drift_pix_raw"] = drift_pix_raw
        out["drift_method"] = drift_method
        self._publish_state(
            {
                "goto": {
                    "autocal_drift_px_s_x": float(drift_pix[0]),
                    "autocal_drift_px_s_y": float(drift_pix[1]),
                }
            }
        )
        platesolving_result = None
        az_hat = None
        alt_hat = None

        if use_horizontal and best_drift_frame is not None:
            arcsec_per_px = float(np.rad2deg(plate_scale_rad) * 3600.0)

            def _solve_drift_to_azalt(v_xy: np.ndarray) -> List[Tuple[float, float]]:
                vx = float(v_xy[0])
                vy_up = float(v_xy[1])
                return _drift_to_az_alt(
                    vx,
                    vy_up,
                    phi_deg=float(getattr(observer, "lat_deg", 0.0)),
                    omega_arcsec_s=float(drift_pointing_omega),
                    scale_arcsec_per_px=arcsec_per_px,
                )

            def _solve_drift_to_azalt_with_fallback(
                v_xy: np.ndarray,
            ) -> Tuple[str, np.ndarray, List[Tuple[float, float]]]:
                v = np.asarray(v_xy, dtype=np.float64).reshape(2,)
                vx = float(v[0])
                vy = float(v[1])
                variants: List[Tuple[str, np.ndarray]] = [
                    ("direct", np.array([vx, vy], dtype=np.float64)),
                    # Retry with alternate axis conventions when +y(up) does not
                    # match camera coordinates or readout orientation.
                    ("y_flip", np.array([vx, -vy], dtype=np.float64)),
                    ("x_flip", np.array([-vx, vy], dtype=np.float64)),
                    ("xy_flip", np.array([-vx, -vy], dtype=np.float64)),
                    ("swap_xy", np.array([vy, vx], dtype=np.float64)),
                    ("swap_xy_y_flip", np.array([vy, -vx], dtype=np.float64)),
                    ("swap_xy_x_flip", np.array([-vy, vx], dtype=np.float64)),
                    ("swap_xy_xy_flip", np.array([-vy, -vx], dtype=np.float64)),
                ]
                tried: List[np.ndarray] = []
                for label, vv in variants:
                    if any(np.allclose(vv, prev, atol=1e-9, rtol=0.0) for prev in tried):
                        continue
                    tried.append(vv)
                    sols = _solve_drift_to_azalt(vv)
                    if sols:
                        return label, vv, sols
                return "direct", v, []

            t_wall = float(getattr(best_drift_frame, "t_wall", 0.0))
            obstime = Time(t_wall, format="unix") if t_wall > 0.0 else Time.now()

            drift_trials: List[Tuple[str, np.ndarray]] = []
            if use_roll_corr:
                drift_trials.append(("roll", np.asarray(drift_pix_corr, dtype=np.float64)))
            drift_trials.append(("raw", np.asarray(drift_pix_raw, dtype=np.float64)))

            prev_trial_vec: Optional[np.ndarray] = None
            for trial_label, trial_vec in drift_trials:
                if prev_trial_vec is not None and np.allclose(trial_vec, prev_trial_vec, atol=1e-9, rtol=0.0):
                    continue
                prev_trial_vec = trial_vec

                if trial_label == "raw" and use_roll_corr:
                    log_error(
                        self._out_log,
                        "GoTo: AutoCal drift az/alt candidates none/failed with roll-corrected drift; retrying raw drift",
                    )

                conv_label, trial_vec_eff, sols = _solve_drift_to_azalt_with_fallback(trial_vec)
                mode_label = trial_label if conv_label == "direct" else f"{trial_label}:{conv_label}"
                if conv_label != "direct":
                    log_info(
                        self._out_log,
                        "GoTo: AutoCal drift az/alt fallback convention "
                        f"mode={mode_label} vec=[{float(trial_vec_eff[0]):.3f},{float(trial_vec_eff[1]):.3f}]",
                    )
                if not sols:
                    log_error(
                        self._out_log,
                        f"GoTo: AutoCal drift az/alt candidates: none (mode={mode_label})",
                    )
                    continue

                sol_txt = ", ".join(f"az={az_deg:.3f} alt={alt_deg:.3f}" for az_deg, alt_deg in sols)
                log_info(
                    self._out_log,
                    f"GoTo: AutoCal drift az/alt candidates (mode={mode_label}): {sol_txt}",
                )

                candidates: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
                for az_deg, alt_deg in sols:
                    try:
                        coord = parse_target_to_icrs(
                            {"az_deg": float(az_deg), "alt_deg": float(alt_deg)},
                            observer=observer,
                            obstime=obstime,
                        )
                        target = (float(coord.ra.deg), float(coord.dec.deg))
                        candidates.append(((float(az_deg), float(alt_deg)), target))
                    except Exception as exc:
                        log_info(self._out_log, f"GoTo: AutoCal drift target parse failed: {exc}")

                if not candidates:
                    log_error(
                        self._out_log,
                        f"GoTo: AutoCal drift target parse produced no candidates (mode={mode_label})",
                    )
                    continue

                drift_pix = np.asarray(trial_vec_eff, dtype=np.float64)
                out["drift_pix"] = drift_pix
                self._publish_state(
                    {
                        "goto": {
                            "autocal_drift_px_s_x": float(drift_pix[0]),
                            "autocal_drift_px_s_y": float(drift_pix[1]),
                        }
                    }
                )

                self._publish_state(
                    {
                        "goto": {
                            "autocal_status": GotoAutocalStatus.RUNNING,
                            "autocal_reason": "PLATESOLVING_LOOP",
                        }
                    }
                )

                for idx, (azalt, target) in enumerate(candidates):
                    if self._op_cancel.is_set():
                        out["status"] = "CANCELLED"
                        return out
                    log_info(
                        self._out_log,
                        "GoTo: AutoCal platesolve (drift) "
                        f"mode={mode_label} cand={idx} az={azalt[0]:.3f} alt={azalt[1]:.3f}",
                    )
                    platesolving_result = self._autocal_run_platesolve(
                        best_drift_frame.raw16,
                        target=target,
                        platesolving_cfg=platesolving_cfg,
                        sep_cfg=sep_cfg,
                        observer=observer,
                        obstime=obstime,
                        solve_radius_deg=autocal_solve_radius_deg,
                        solve_gmax=autocal_solve_gmax,
                    )
                    if bool(getattr(platesolving_result, "success", False)):
                        az_hat, alt_hat = float(azalt[0]), float(azalt[1])
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
                        break
                if platesolving_result is not None and bool(getattr(platesolving_result, "success", False)):
                    break
        if use_horizontal:
            if platesolving_result is None or not bool(getattr(platesolving_result, "success", False)):
                out["status"] = "ERR_PLATESOLVING"
                out["platesolving_result"] = platesolving_result
                log_info(self._out_log, "GoTo: AutoCal platesolving failed (horizontal drift)")
                return out

        if not use_horizontal:
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
            t_ref = Time.now()
            pointing_report = _solve_jcal_pointing(
                drift_pix,
                J_pix,
                plate_scale_rad_per_px=plate_scale_rad,
                observer=observer,
                obstime=t_ref,
                alt_min_deg=float(goto_cfg.alt_min_deg),
                alt_max_deg=float(goto_cfg.alt_max_deg),
                axis_sign_az=int(self._goto.model.kin.axis_sign_az),
                axis_sign_alt=int(self._goto.model.kin.axis_sign_alt),
                seeds=params.get("pointing_seeds"),
            )
            out["pointing_solver"] = pointing_report
            if not bool(pointing_report.get("ok", False)):
                out["status"] = "ERR_POINTING_SOLVE"
                log_error(
                    self._out_log,
                    "GoTo: AutoCal JCal pointing solve rejected "
                    f"status={pointing_report.get('status')} "
                    f"resid={pointing_report.get('residual_rad_s')}",
                )
                return out
            coeff_pix_s = np.asarray(pointing_report["coeff_pix_s"], dtype=np.float64)
            tangent_obs = np.asarray(
                pointing_report["observed_tangent_rad_s"], dtype=np.float64
            )
            v_obs = np.asarray(
                pointing_report["coordinate_rate_rad_s"], dtype=np.float64
            )
            log_info(
                self._out_log,
                "GoTo: AutoCal drift basis solve "
                f"coeff_az={float(coeff_pix_s[0]):.3f}px/s "
                f"coeff_alt={float(coeff_pix_s[1]):.3f}px/s "
                f"e_az_hat=[{float(e_az_hat[0]):.3f},{float(e_az_hat[1]):.3f}] "
                f"e_alt_hat=[{float(e_alt_hat[0]):.3f},{float(e_alt_hat[1]):.3f}] "
                f"tangent_rad_s=[{float(tangent_obs[0]):.6e},{float(tangent_obs[1]):.6e}]",
            )

        if platesolving_result is None or not bool(getattr(platesolving_result, "success", False)):
            az_hat = float(pointing_report["az_deg"])
            alt_hat = float(pointing_report["alt_deg"])
            best_res = float(pointing_report["residual_rad_s"])
            dist_to_0 = min(az_hat, 360.0 - az_hat)
            dist_to_180 = abs(az_hat - 180.0)
            if dist_to_0 < 15.0 or dist_to_180 < 15.0:
                out["status"] = "ERR_DEGENERATE_AZ"
                return out

            log_info(
                self._out_log,
                "GoTo: AutoCal pointing estimate "
                f"az={float(az_hat):.3f}deg alt={float(alt_hat):.3f}deg "
                f"resid={float(best_res):.3e} v_obs_rad_s=[{float(v_obs[0]):.6e},{float(v_obs[1]):.6e}]",
            )
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

                solve_obstime = self._autocal_frame_obstime(fr)
                dt_s = float((solve_obstime - t_ref).to_value(u.s))
                az_c = _wrap_deg_360(az_hat + float(v_deg_s[0]) * dt_s)
                alt_c = float(alt_hat + float(v_deg_s[1]) * dt_s)
                alt_c = float(np.clip(alt_c, goto_cfg.alt_min_deg, goto_cfg.alt_max_deg))

                jitter = jitter_seq[attempts % len(jitter_seq)]
                target = {"az_deg": float(az_c + jitter[0]), "alt_deg": float(alt_c + jitter[1])}

                if attempts == 0:
                    log_info(
                        self._out_log,
                        "GoTo: AutoCal platesolve first target "
                        f"az={float(target['az_deg']):.3f}deg "
                        f"alt={float(target['alt_deg']):.3f}deg "
                        f"radius={1.0:.3f}deg "
                        f"jitter=({float(jitter[0]):.3f},{float(jitter[1]):.3f}) "
                        f"dt={dt_s:.3f}s",
                    )
                platesolving_result = self._autocal_run_platesolve(
                    fr.raw16,
                    target=target,
                    platesolving_cfg=platesolving_cfg,
                    sep_cfg=sep_cfg,
                    observer=observer,
                    obstime=solve_obstime,
                    solve_radius_deg=autocal_solve_radius_deg,
                    solve_gmax=autocal_solve_gmax,
                )
                attempts += 1
                if bool(getattr(platesolving_result, "success", False)):
                    break

        if platesolving_result is None or not bool(getattr(platesolving_result, "success", False)):
            out["status"] = "ERR_PLATESOLVING"
            out["platesolving_result"] = platesolving_result
            return out

        out["platesolving_result"] = platesolving_result

        manual_only = bool(params.get("manual_only", True))
        if manual_only:
            solve_obstime = _platesolving_result_obstime(platesolving_result)
            az_alt = platesolving_center_to_altaz_deg(
                float(platesolving_result.center_ra_deg),
                float(platesolving_result.center_dec_deg),
                observer=observer,
                obstime=solve_obstime,
            )
            roll_sample = self._roll_sample_from_solution(
                platesolving_result,
                observer=observer,
                obstime=solve_obstime,
            )
            continuity = self._check_manual_sample_continuity(
                az_alt,
                roll_deg=roll_sample,
                context="autocal drift",
            )
            if not bool(continuity.get("ok", False)):
                out["platesolving_result"] = self._invalidate_platesolving_after_continuity_rejection(
                    platesolving_result
                )
                out["status"] = "ERR_SAMPLE_CONTINUITY"
                return out
            n_samples = int(
                self._goto.model.add_manual_sample(
                    az_alt,
                    roll_deg=roll_sample,
                    source=(self._diagnostics.path_str if self._diagnostics is not None else None),
                )
            )
            self._publish_state(
                {
                    "goto": {
                        "status": GotoStatus.OK,
                        "reason": None,
                        "autocal_last_ok": True,
                        "autocal_status": GotoAutocalStatus.OK,
                        "autocal_reason": "MANUAL_SAMPLE",
                        "manual_samples": n_samples,
                    }
                }
            )
            out["ok"] = True
            out["status"] = "OK_MANUAL_SAMPLE"
            out["manual_samples"] = n_samples
            return out

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

    def _roll_sample_from_solution(
        self,
        result: PlatesolvingResult,
        *,
        observer: ObserverConfig,
        obstime: Optional[Time] = None,
    ) -> float:
        return platesolving_roll_sample_deg(
            result,
            observer=observer,
            obstime=obstime,
        )

    def _check_manual_sample_continuity(
        self,
        az_alt_deg: np.ndarray,
        *,
        roll_deg: Optional[float],
        context: str,
        reference_steps: Optional[np.ndarray] = None,
        reference_az_alt_deg: Optional[np.ndarray] = None,
        reference_roll_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        report = self._goto.model.manual_sample_continuity_report(
            az_alt_deg,
            roll_deg=roll_deg,
            reference_steps=reference_steps,
            reference_az_alt_deg=reference_az_alt_deg,
            reference_roll_deg=reference_roll_deg,
        )
        if not bool(report.get("ok", False)):
            log_error(
                self._out_log,
                "GoTo: rejected implausible plate-solve sample "
                f"context={context} "
                f"dsteps=[{float(report.get('dsteps_az', 0.0)):+.0f},"
                f"{float(report.get('dsteps_alt', 0.0)):+.0f}] "
                f"motion={float(report.get('observed_motion_deg', float('nan'))):.4f}deg "
                f"limit={float(report.get('motion_limit_deg', float('nan'))):.4f}deg "
                f"roll_jump={float(report.get('roll_jump_deg', float('nan'))):.3f}deg "
                f"roll_limit={float(report.get('roll_tolerance_deg', float('nan'))):.3f}deg",
            )
        return report

    def _invalidate_platesolving_after_continuity_rejection(
        self,
        result: PlatesolvingResult,
    ) -> PlatesolvingResult:
        rejected = replace(
            result,
            success=False,
            status="MOTION_CONTINUITY_MISMATCH",
        )
        self._publish_state(
            {
                "platesolving": {
                    "busy": False,
                    "status": PlatesolvingStatus.FAIL,
                    "reason": "MOTION_CONTINUITY_MISMATCH",
                    "last_ok": False,
                },
                # Clear the cached success as well; otherwise a later manual
                # Sync could reuse the solution that was just rejected.
                "platesolving_result": rejected,
                "platesolving_result_handled": True,
            }
        )
        return rejected

    def _publish_j_matrix_state(self) -> None:
        model = self._goto.model
        J = getattr(model, "J_deg_per_step", None)
        if J is None or getattr(J, "shape", None) != (2, 2):
            log_error(self._out_log, "GoTo: J matrix unavailable or invalid", ValueError("invalid J matrix"))
            return
        fit_report = model.model_fit_report()
        self._publish_state(
            {
                "goto": {
                    "J00": float(J[0, 0]),
                    "J01": float(J[0, 1]),
                    "J10": float(J[1, 0]),
                    "J11": float(J[1, 1]),
                    "J00_err": float(fit_report["J00_err"]),
                    "J01_err": float(fit_report["J01_err"]),
                    "J10_err": float(fit_report["J10_err"]),
                    "J11_err": float(fit_report["J11_err"]),
                    "model_non_orthogonality_deg": float(fit_report["model_non_orthogonality_deg"]),
                    "model_non_orthogonality_err_deg": float(fit_report["model_non_orthogonality_err_deg"]),
                    "model_camera_roll_deg": float(fit_report["model_roll_deg"]),
                    "model_camera_roll_err_deg": float(fit_report["model_roll_err_deg"]),
                    "model_camera_roll_samples": int(fit_report["model_roll_samples"]),
                    "model_pitch_deg": float(fit_report["model_pitch_deg"]),
                    "model_pitch_err_deg": float(fit_report["model_pitch_err_deg"]),
                    "model_yaw_deg": float(fit_report["model_yaw_deg"]),
                    "model_yaw_err_deg": float(fit_report["model_yaw_err_deg"]),
                    "model_fit_samples": int(fit_report["model_fit_samples"]),
                    "model_fit_rms_az_deg": float(fit_report["model_fit_rms_az_deg"]),
                    "model_fit_rms_alt_deg": float(fit_report["model_fit_rms_alt_deg"]),
                    "model_fit_rms_arcsec": float(fit_report["model_fit_rms_arcsec"]),
                    "periodic_error_az_deg": float(fit_report["periodic_error_az_deg"]),
                    "periodic_error_alt_deg": float(fit_report["periodic_error_alt_deg"]),
                    "periodic_model_samples": int(fit_report["periodic_model_samples"]),
                    "last_direction_az": int(fit_report["last_direction_az"]),
                    "last_direction_alt": int(fit_report["last_direction_alt"]),
                    "backlash_steps_az": int(fit_report["backlash_steps_az"]),
                    "backlash_steps_alt": int(fit_report["backlash_steps_alt"]),
                    "synced": bool(getattr(model, "synced", False)),
                }
            }
        )

    def _log_model_fit_state(self, *, prefix: str) -> None:
        rep = self._goto.model.model_fit_report()
        J = getattr(self._goto.model, "J_deg_per_step", np.zeros((2, 2), dtype=np.float64))
        log_info(
            self._out_log,
            f"{prefix} "
            f"samples={int(rep['model_fit_samples'])} "
            f"J=[[{float(J[0, 0]):+.6e},{float(J[0, 1]):+.6e}],"
            f"[{float(J[1, 0]):+.6e},{float(J[1, 1]):+.6e}]] "
            f"dJ=[[{float(rep['J00_err']):.3e},{float(rep['J01_err']):.3e}],"
            f"[{float(rep['J10_err']):.3e},{float(rep['J11_err']):.3e}]] "
            f"nonorth={float(rep['model_non_orthogonality_deg']):+.4f}±{float(rep['model_non_orthogonality_err_deg']):.4f}deg "
            f"roll={float(rep['model_roll_deg']):+.3f}±{float(rep['model_roll_err_deg']):.3f}deg "
            f"rms=[az={float(rep['model_fit_rms_az_deg']):.4f}deg "
            f"alt={float(rep['model_fit_rms_alt_deg']):.4f}deg "
            f"tot={float(rep['model_fit_rms_arcsec']):.2f}arcsec]",
        )

    def _pointing_snapshot_from_model(self) -> Optional[Dict[str, float]]:
        az_alt = self._goto.model.current_az_alt_deg()
        if az_alt is None:
            return None
        az = float(az_alt[0]) % 360.0
        alt = float(np.clip(float(az_alt[1]), -90.0, 90.0))
        if (not np.isfinite(az)) or (not np.isfinite(alt)):
            return None
        coord_icrs = parse_target_to_icrs(
            {"az_deg": az, "alt_deg": alt},
            observer=self._get_observer(),
            obstime=Time.now(),
        ).icrs
        ra = float(coord_icrs.ra.deg) % 360.0
        dec = float(coord_icrs.dec.deg)
        if (not np.isfinite(ra)) or (not np.isfinite(dec)):
            return None
        return {
            "az_deg": az,
            "alt_deg": alt,
            "ra_deg": ra,
            "dec_deg": dec,
        }

    def _handle_request(self, request: Dict[str, Any]) -> None:
        kind = str(request.get("kind", "goto"))
        target = request.get("target", None)
        params = dict(request.get("params", {}) or {})

        goto_cfg = self._get_goto_cfg()
        platesolving_cfg = self._get_platesolving_cfg()
        diagnostics_dir = str(
            getattr(
                goto_cfg,
                "diagnostics_dir",
                getattr(platesolving_cfg, "diagnostics_dir", "stack_output/goto_diagnostics"),
            )
        )
        diagnostics_enabled = bool(
            getattr(
                goto_cfg,
                "diagnostics_enabled",
                getattr(platesolving_cfg, "diagnostics_enabled", False),
            )
        )
        self._diagnostics = DiagnosticSession(
            root_dir=diagnostics_dir,
            operation=kind,
            enabled=diagnostics_enabled,
            context={
                "target": target,
                "params": params,
                "goto_config": goto_cfg,
                "platesolving_config": platesolving_cfg,
                "sep_config": self._get_sep_cfg(),
                "camera_config": self._get_camera_cfg(),
                "mount_config": self._get_mount_cfg(),
                "observer": self._get_observer(),
                "model_before": self._diagnostics_model_snapshot(),
            },
            out_log=self._out_log,
        )
        diagnostics_path = self._diagnostics.path_str
        if diagnostics_path is not None:
            self._publish_state({"goto": {"diagnostics_dir": diagnostics_path}})
        diagnostic_status = "UNKNOWN"

        was_tracking = self._pause_tracking()
        was_stacking = self._pause_stacking()

        self._publish_state(
            {"goto": {"busy": True, "status": GotoStatus.RUNNING, "reason": str(kind)}}
        )
        self._initial_solution_confirmed = False
        self._diagnostics_record(
            "operation_started",
            tracking_was_enabled=bool(was_tracking),
            stacking_was_enabled=bool(was_stacking),
        )

        try:
            if self._op_cancel.is_set():
                diagnostic_status = "CANCELLED_BEFORE_START"
                self._publish_state(
                    {
                        "goto": {
                            "status": GotoStatus.CANCELLED,
                            "reason": "CANCELLED_BEFORE_START",
                        }
                    }
                )
                return
            if kind == "goto":
                self._diagnostics_save_live_frame("goto_before_move")
                delay_us = int(params.get("delay_us", goto_cfg.slew_delay_us))
                tol_arcsec = float(params.get("tol_arcsec", goto_cfg.tol_arcsec))
                gain = float(params.get("gain", goto_cfg.gain))
                if "max_step_per_iter" in params:
                    max_step_per_iter = int(params.get("max_step_per_iter"))
                elif "max_step_deg" in params:
                    max_step_deg = float(params.get("max_step_deg", 5.0))
                    j_matrix = self._goto.model.J_deg_per_step
                    max_abs_deg_per_step = float(np.max(np.abs(j_matrix))) if j_matrix is not None and j_matrix.size else 0.0
                    if max_abs_deg_per_step > 0.0:
                        max_step_per_iter = int(max(1, round(max_step_deg / max_abs_deg_per_step)))
                    else:
                        max_step_per_iter = 0
                else:
                    max_step_per_iter = int(goto_cfg.max_step_per_iter)

                self._goto.cfg = replace(
                    self._goto.cfg,
                    tol_arcsec=tol_arcsec,
                    max_iters=1,
                    gain=gain,
                    max_step_per_iter=max_step_per_iter,
                    slew_delay_us_az=delay_us,
                    slew_delay_us_alt=delay_us,
                    stages=1,
                    platesolving_feedback=False,
                )

                status = self._goto.goto_blocking(
                    target,
                    get_live_frame=self._get_live_raw16,
                    move_steps=self._move_steps,
                    stop=self._stop_mount,
                    platesolving_cfg=platesolving_cfg,
                    stages=1,
                    platesolving_feedback=False,
                    diagnostics=self._diagnostics,
                    cancel_requested=self._op_cancel.is_set,
                )
                self._diagnostics_save_live_frame("goto_after_move")
                diagnostic_status = str(status.status)
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
                strategy = str(params.get("strategy", "default")).strip().lower()

                if strategy in ("right_scan", "scan_right", "right", "direction_scan", "dir_scan"):
                    calib_out = self._goto_calibrate_right_scan_blocking(params)
                    calib_ok = bool(calib_out.get("ok", False))
                    calib_status = str(calib_out.get("status", "UNKNOWN"))
                    diagnostic_status = f"CALIBRATE_RIGHT_SCAN_{calib_status}"
                    self._diagnostics_record(
                        "calibration_result",
                        strategy="right_scan",
                        result=calib_out,
                        model_after=self._diagnostics_model_snapshot(),
                    )
                    calib_samples = int(calib_out.get("n_samples", 0))
                    self._publish_state(
                        {
                            "goto": {
                                "status": GotoStatus.OK if calib_ok else GotoStatus.FAIL,
                                "reason": f"CALIBRATE_RIGHT_SCAN_{calib_status}",
                            }
                        }
                    )
                    log_info(
                        self._out_log,
                        "GoTo: CALIBRATE_RIGHT_SCAN "
                        f"status={calib_status} ok={calib_ok} "
                        f"steps_done={int(calib_out.get('steps_done', 0))} "
                        f"solves_ok={int(calib_out.get('solves_ok', 0))} "
                        f"samples={calib_samples}",
                    )
                    self._publish_j_matrix_state()
                    return

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
                    diagnostics=self._diagnostics,
                )
                calib_ok = bool(calib_out.get("ok", False))
                calib_status = str(calib_out.get("status", "UNKNOWN"))
                diagnostic_status = f"CALIBRATE_{calib_status}"
                self._diagnostics_record(
                    "calibration_result",
                    strategy="default",
                    result=calib_out,
                    model_after=self._diagnostics_model_snapshot(),
                )
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
                autocal_status = str(autocal_out.get("status", "UNKNOWN"))
                diagnostic_status = f"AUTOCAL_{autocal_status}"
                self._diagnostics_record(
                    "autocal_result",
                    result=autocal_out,
                    model_after=self._diagnostics_model_snapshot(),
                )
                if autocal_ok:
                    autocal_reason = "MANUAL_SAMPLE" if autocal_status == "OK_MANUAL_SAMPLE" else "READY"
                else:
                    autocal_reason = autocal_status
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

            elif kind == "roll":
                roll_out = self._goto_estimate_roll_blocking(params)
                roll_ok = bool(roll_out.get("ok", False))
                roll_status = str(roll_out.get("status", "UNKNOWN"))
                diagnostic_status = f"ROLL_{roll_status}"
                self._diagnostics_record(
                    "roll_result",
                    result=roll_out,
                    model_after=self._diagnostics_model_snapshot(),
                )
                self._publish_state(
                    {
                        "goto": {
                            "status": GotoStatus.OK if roll_ok else GotoStatus.FAIL,
                            "reason": f"ROLL_{roll_status}",
                        }
                    }
                )
                log_info(
                    self._out_log,
                    f"GoTo: ROLL_ESTIMATE status={roll_status} ok={roll_ok}",
                )

            elif kind == "fit_model":
                min_samples = int(params.get("min_samples", 3))
                ridge = float(params.get("ridge", 1e-12))
                self._diagnostics_record(
                    "fit_model_input",
                    min_samples=int(min_samples),
                    ridge=float(ridge),
                    model=self._diagnostics_model_snapshot(),
                    deviation_report=self._goto.model.manual_samples_deviation_report(
                        sort_by_deviation=False
                    ),
                )
                ok = bool(self._goto.model.fit_J_from_manual_samples(min_samples=min_samples, ridge=ridge))
                n_manual = int(len(getattr(self._goto.model, "_manual_steps_abs", [])))
                synced_from_manual = False
                if ok and not bool(getattr(self._goto.model, "synced", False)):
                    synced_from_manual = bool(self._goto.model.sync_from_latest_manual_sample())
                self._publish_j_matrix_state()
                fit_reason = str(getattr(self._goto.model, "last_fit_reason", "DEGENERATE_MODEL"))
                diagnostic_status = "FIT_MODEL_OK" if ok else f"FIT_MODEL_ERR_{fit_reason}"
                if not ok and fit_reason == "MODEL_FIT_PHASE_COVERAGE_TOO_SHORT":
                    cov = dict(getattr(self._goto.model, "last_fit_phase_coverage", {}) or {})
                    period_az = int(self._goto.model.kin.transmission_error_period_steps(Axis.AZ))
                    period_alt = int(self._goto.model.kin.transmission_error_period_steps(Axis.ALT))
                    log_info(
                        self._out_log,
                        "GoTo fit: recorrido insuficiente para medir la escala media "
                        f"(az={cov.get('az', 0.0):.2f} ciclos, alt={cov.get('alt', 0.0):.2f} ciclos). "
                        "El error de transmisión ciclodial domina en movimientos cortos; "
                        f"usa desplazamientos de al menos un ciclo completo "
                        f"({period_az} pasos en az, {period_alt} en alt).",
                    )
                self._diagnostics_record(
                    "fit_model_result",
                    success=ok,
                    reason=fit_reason,
                    synced_from_manual=bool(synced_from_manual),
                    model=self._diagnostics_model_snapshot(),
                    deviation_report=self._goto.model.manual_samples_deviation_report(
                        sort_by_deviation=False
                    ),
                )
                fail_reason = None if ok else f"ERR_{fit_reason}"
                self._publish_state(
                    {
                        "goto": {
                            "synced": bool(getattr(self._goto.model, "synced", False)),
                            "status": GotoStatus.OK if ok else GotoStatus.FAIL,
                            "reason": None if ok else fail_reason,
                        }
                    }
                )
                log_info(
                    self._out_log,
                    f"GoTo: FIT_MODEL ok={ok} min_samples={min_samples} "
                    f"manual_samples={n_manual} "
                    f"synced={bool(getattr(self._goto.model, 'synced', False))} "
                    f"synced_from_manual={synced_from_manual}",
                )
                if ok:
                    self._log_model_fit_state(prefix="GoTo: FIT_MODEL report")

            elif kind == "list_samples":
                report = self._goto.model.manual_samples_deviation_report(sort_by_deviation=True)
                n_manual = int(len(getattr(self._goto.model, "_manual_steps_abs", [])))
                diagnostic_status = "LIST_SAMPLES"
                self._diagnostics_record("manual_samples_report", samples=report)
                if not report:
                    self._publish_state(
                        {
                            "goto": {
                                "manual_samples": int(n_manual),
                                "status": GotoStatus.OK,
                                "reason": "LIST_SAMPLES_EMPTY",
                            }
                        }
                    )
                    log_info(self._out_log, "GoTo: LIST_SAMPLES no manual samples")
                else:
                    thr_arcsec = float(report[0].get("threshold_arcsec", float("nan")))
                    log_info(
                        self._out_log,
                        f"GoTo: LIST_SAMPLES n={n_manual} threshold={thr_arcsec:.2f}arcsec (sorted by deviation)",
                    )
                    for row in report:
                        idx = int(row.get("sample_idx", -1))
                        rank = int(row.get("rank_deviation", 0))
                        dev_arcsec = float(row.get("dev_arcsec", float("nan")))
                        dev_az_arcsec = float(row.get("dev_az_deg", float("nan"))) * 3600.0
                        dev_alt_arcsec = float(row.get("dev_alt_deg", float("nan"))) * 3600.0
                        out_tag = "OUTLIER" if bool(row.get("outlier_suggested", False)) else "inlier"
                        ref_tag = " ref" if bool(row.get("is_ref_idx", False)) else ""
                        log_info(
                            self._out_log,
                            "GoTo: SAMPLE "
                            f"idx={idx} rank={rank} {out_tag}{ref_tag} "
                            f"dev={dev_arcsec:.2f}arcsec "
                            f"(az={dev_az_arcsec:+.2f}, alt={dev_alt_arcsec:+.2f}) "
                            f"steps=[{float(row.get('steps_az', 0.0)):.0f},{float(row.get('steps_alt', 0.0)):.0f}] "
                            f"altaz=[{float(row.get('az_deg', 0.0)):.4f},{float(row.get('alt_deg', 0.0)):.4f}]",
                        )
                    self._publish_state(
                        {
                            "goto": {
                                "manual_samples": int(n_manual),
                                "status": GotoStatus.OK,
                                "reason": "LIST_SAMPLES",
                            }
                        }
                    )

            elif kind == "prune_outliers":
                min_samples = int(params.get("min_samples", 3))
                ridge = float(params.get("ridge", 1e-12))
                out = self._goto.model.prune_manual_outliers(min_samples=min_samples, ridge=ridge)
                n_after = int(out.get("n_after", len(getattr(self._goto.model, "_manual_steps_abs", []))))
                removed = int(out.get("removed_count", 0))
                removed_indices = list(out.get("removed_indices", []))
                prune_ok = bool(out.get("ok", False))
                prune_status = str(out.get("status", "UNKNOWN"))
                diagnostic_status = f"PRUNE_OUTLIERS_{prune_status}"
                synced_from_manual = False
                if prune_ok and (not bool(getattr(self._goto.model, "synced", False))) and n_after > 0:
                    synced_from_manual = bool(self._goto.model.sync_from_latest_manual_sample())
                self._publish_state(
                    {
                        "goto": {
                            "manual_samples": int(n_after),
                            "synced": bool(getattr(self._goto.model, "synced", False)),
                            "status": GotoStatus.OK if prune_status in ("OK", "NO_OUTLIERS") else GotoStatus.FAIL,
                            "reason": f"PRUNE_OUTLIERS_{prune_status}",
                        }
                    }
                )
                log_info(
                    self._out_log,
                    "GoTo: PRUNE_OUTLIERS "
                    f"status={prune_status} ok={prune_ok} "
                    f"removed={removed} indices={removed_indices} "
                    f"n_before={int(out.get('n_before', 0))} n_after={n_after} "
                    f"threshold={float(out.get('threshold_arcsec', float('nan'))):.2f}arcsec "
                    f"fit_before_ok={bool(out.get('fit_before_ok', False))} "
                    f"fit_after_ok={bool(out.get('fit_after_ok', False))} "
                    f"synced_from_manual={synced_from_manual}",
                )
                if prune_ok:
                    self._log_model_fit_state(prefix="GoTo: PRUNE_OUTLIERS report")
                self._diagnostics_record(
                    "prune_outliers_result",
                    result=out,
                    model_after=self._diagnostics_model_snapshot(),
                )

            elif kind == "restore_last_log":
                model = self._goto.model
                out = model.restore_from_latest_logs()
                ok = bool(out.get("ok", False))
                status = str(out.get("status", "UNKNOWN"))
                diagnostic_status = f"RESTORE_{status}"
                n_manual = int(out.get("manual_samples", len(getattr(model, "_manual_steps_abs", []))))
                camera_roll = float(out.get("camera_roll_deg", float("nan")))
                if np.isfinite(camera_roll):
                    try:
                        self._apply_camera_param("roll_deg", float(camera_roll))
                    except Exception as exc:
                        log_error(self._out_log, "GoTo: failed to apply restored camera roll", exc)
                        self._publish_state({"camera": {"roll_deg": float(camera_roll)}})

                pointing = None
                try:
                    pointing = self._pointing_snapshot_from_model()
                except Exception as exc:
                    log_error(self._out_log, "GoTo: failed to compute restored pointing", exc)

                goto_patch: Dict[str, Any] = {
                    "manual_samples": int(n_manual),
                    "synced": bool(getattr(model, "synced", False)),
                    "status": GotoStatus.OK if ok else GotoStatus.FAIL,
                    "reason": f"RESTORE_{status}",
                }
                if pointing is not None:
                    goto_patch.update(
                        {
                            "pointing_valid": True,
                            "pointing_az_deg": float(pointing["az_deg"]),
                            "pointing_alt_deg": float(pointing["alt_deg"]),
                            "pointing_ra_deg": float(pointing["ra_deg"]),
                            "pointing_dec_deg": float(pointing["dec_deg"]),
                        }
                    )
                elif ok:
                    goto_patch["pointing_valid"] = False
                self._publish_state(
                    {
                        "goto": goto_patch,
                    }
                )
                log_info(
                    self._out_log,
                    "GoTo: RESTORE_LAST_LOG "
                    f"status={status} ok={ok} "
                    f"manual_samples={n_manual} "
                    f"synced={bool(getattr(model, 'synced', False))} "
                    f"camera_roll={camera_roll:+.3f}deg "
                    f"loaded_manual={bool(out.get('loaded_manual', False))} "
                    f"loaded_fit={bool(out.get('loaded_fit', False))}",
                )
                if ok:
                    self._log_model_fit_state(prefix="GoTo: RESTORE_LAST_LOG report")

            elif kind == "reset":
                model = self._goto.model
                n_manual_prev = int(len(getattr(model, "_manual_steps_abs", [])))
                was_synced = bool(getattr(model, "synced", False))
                model.reset_manual_samples_and_sync()
                diagnostic_status = "RESET_OK"
                self._publish_state(
                    {
                        "goto": {
                            "manual_samples": 0,
                            "sample_last_ok": False,
                            "sample_last_reason": None,
                            "synced": False,
                            "pointing_valid": False,
                            "pointing_az_deg": 0.0,
                            "pointing_alt_deg": 0.0,
                            "pointing_ra_deg": 0.0,
                            "pointing_dec_deg": 0.0,
                            "last_error_arcsec": 0.0,
                            "autocal_status": GotoAutocalStatus.IDLE,
                            "autocal_reason": None,
                            "autocal_last_ok": False,
                            "status": GotoStatus.OK,
                            "reason": None,
                        }
                    }
                )
                log_info(
                    self._out_log,
                    f"GoTo: RESET ok=True manual_samples={n_manual_prev}->0 synced={was_synced}->False",
                )

            else:
                diagnostic_status = f"ERR_KIND_{kind}"
                self._publish_state(
                    {"goto": {"status": GotoStatus.FAIL, "reason": f"ERR_KIND_{kind}"}}
                )

            self._publish_j_matrix_state()

        except (RuntimeError, ValueError, TypeError) as exc:
            diagnostic_status = "EXCEPTION"
            log_error(self._out_log, f"GoTo worker failed ({kind})", exc)
            self._diagnostics_record("operation_exception", error=repr(exc))
            self._publish_state({"goto": {"status": GotoStatus.FAIL, "reason": "EXCEPTION"}})

        finally:
            if was_stacking:
                self._resume_stacking()
            if was_tracking:
                self._resume_tracking()
            self._publish_state({"goto": {"busy": False}})
            diagnostics = self._diagnostics
            self._diagnostics = None
            if diagnostics is not None:
                diagnostics.close(
                    diagnostic_status,
                    cancelled=bool(self._op_cancel.is_set()),
                    tracking_restored=bool(was_tracking),
                    stacking_restored=bool(was_stacking),
                    model_after=self._diagnostics_model_snapshot(),
                )


# ============================================================
# Convenience: initial model builder (for your exact mount)
# ============================================================

def make_default_goto_controller_for_your_mount() -> GoToController:
    """Factory using the mechanical parameters you provided.

    AZ/ALT: 45:1 motor-to-axis mechanical reduction.
    Microstepping defaults to 1/64.
    """
    kin = MountKinematics(
        motor_full_steps_per_rev=200,
        microsteps_az=64,
        microsteps_alt=64,
        gear_reduction_az=45.0,
        gear_reduction_alt=45.0,
        axis_sign_az=+1,
        axis_sign_alt=+1,
    )
    model = GoToModel(kin=kin)
    model.init_from_mechanics()
    cfg = GoToConfig(observer=ObserverConfig())
    return GoToController(cfg=cfg, model=model)
