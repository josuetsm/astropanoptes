# tracking.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

import numpy as np
from logging_utils import log_error
from raw_alignment import (
    RawAlignmentSignature,
    build_raw_alignment_signature,
    estimate_raw_translation,
)

try:
    import cv2
except Exception:  # pragma: no cover - runtime fallback for limited envs
    cv2 = None


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class KeyframeConfig:
    abs_corr_every_s: float = 2.5
    abs_resp_min: float = 0.08
    abs_max_px: float = 140.0
    abs_blend_beta: float = 0.35
    keyframe_refresh_px: float = 2.5


@dataclass
class PIConfig:
    kp: float = 0.20
    ki: float = 0.015
    kd: float = 0.00
    eint_clamp: float = 400.0


@dataclass
class RateLimiterConfig:
    rate_max: float = 300.0
    rate_slew_per_update: float = 50.0
    update_s: float = 0.5
    observe_s: float = 5.0
    fail_reset_n: int = 12
    max_shift_per_frame_px: float = 25.0
    min_meas_dt_s: float = 0.03
    max_meas_v_px_s: float = 200.0
    max_shift_scale_min: float = 0.20
    lock_warmup_frames: int = 12
    lock_drop_decay: float = 0.80
    fb_max_frac: float = 0.60
    eint_decay_on_bad: float = 0.96
    rls_min_lock_conf: float = 0.70
    rls_min_rate_steps_s: float = 50.0
    source_resp_min: float = 0.35
    source_profile_disagree_px: float = 3.0


@dataclass
class AlignmentConfig:
    median_k: int = 3
    smooth_k: int = 30
    max_shift_px: float = 50.0
    use_subpixel: bool = True


@dataclass
class CalibrationConfig:
    """
    Modelo: v_pxps = A * u + b

    - u: [u_az, u_alt] en µsteps/s (señal de control; AppRunner la discretiza a MOVE)
    - v: [vx, vy] en px/s (medido directamente desde perfiles RAW16)
    """
    lambda_dls: float = 0.05               # DLS para pinv


@dataclass
class AutoCalConfig:
    rls_forget: float = 0.990
    P0: float = 2000.0
    min_det: float = 1e-4
    max_cond: float = 250.0


@dataclass
class TrackingConfig:
    resp_min: float = 0.25
    align: AlignmentConfig = field(default_factory=AlignmentConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    pi: PIConfig = field(default_factory=PIConfig)
    rate: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    calib: CalibrationConfig = field(default_factory=CalibrationConfig)
    autocal: AutoCalConfig = field(default_factory=AutoCalConfig)


# ============================================================
# State dataclasses
# ============================================================

@dataclass
class AutoCalState:
    ok: bool = False
    src: str = "none"   # none|auto|rls

    theta: Optional[np.ndarray] = None  # 2x3: [A|b]
    P: Optional[np.ndarray] = None      # 3x3

    A: Optional[np.ndarray] = None      # 2x2
    b: Optional[np.ndarray] = None      # 2
    A_pinv: Optional[np.ndarray] = None # 2x2

    detA: float = 0.0
    condA: float = 0.0
    last_upd_t: Optional[float] = None


@dataclass
class TrackingState:
    cfg: TrackingConfig = field(default_factory=TrackingConfig)

    # incremental tracking
    prev_signature: Optional[RawAlignmentSignature] = None
    prev_t: Optional[float] = None
    fail: int = 0
    last_dx_inc: float = 0.0
    last_dy_inc: float = 0.0

    # filtered velocity estimate (px/s)
    vpx: float = 0.0
    vpy: float = 0.0
    vx_inst: float = 0.0
    vy_inst: float = 0.0
    resp_inc: float = 0.0
    lock_conf: float = 0.0

    # keyframe & absolute correction
    key_signature: Optional[RawAlignmentSignature] = None
    key_t: Optional[float] = None
    x_hat: float = 0.0
    y_hat: float = 0.0
    abs_last_t: Optional[float] = None
    abs_resp_last: float = 0.0

    # PI integral
    eint_x: float = 0.0
    eint_y: float = 0.0

    # output rates (µsteps/s)
    rate_az: float = 0.0
    rate_alt: float = 0.0

    # mode
    current_mode: str = "IDLE"   # IDLE|STABILIZE|TRACK
    t_mode: Optional[float] = None

    # autocal state
    auto: AutoCalState = field(default_factory=AutoCalState)

    # last used calibration source for control
    calib_src_last: str = "none"  # none|auto


@dataclass
class TrackingOutput:
    ok: bool
    mode: str
    resp: float
    dx: float
    dy: float
    vx: float
    vy: float
    abs_resp: float
    x_hat: float
    y_hat: float
    rate_az: float
    rate_alt: float
    calib_src: str
    detA: float
    n_det: int
    measurement_reason: str = ""
    measurement_source: str = "none"
    lock_conf: float = 0.0
    fail_count: int = 0


@dataclass(frozen=True)
class _AlignmentMeasurement:
    ok: bool
    dx: float = 0.0
    dy: float = 0.0
    resp: float = 0.0
    source: str = "none"
    reason: str = "low_confidence"


# ============================================================
# Small helpers
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _ensure_raw16_bayer(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint16:
        raise TypeError(f"raw16 must be uint16, got {arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"raw16 must have shape (H,W) or (H,W,1), got {arr.shape}")


def rate_ramp(cur: float, target: float, max_delta: float) -> float:
    d = target - cur
    d = clamp(d, -max_delta, +max_delta)
    return cur + d


def compute_A_pinv_dls(A: np.ndarray, lam: float) -> np.ndarray:
    """
    DLS pseudo-inverse: (A^T A + lam^2 I)^(-1) A^T
    """
    A = np.asarray(A, dtype=np.float64).reshape(2, 2)
    AtA = A.T @ A
    I = np.eye(2, dtype=np.float64)
    M = AtA + (float(lam) * float(lam)) * I
    return np.linalg.inv(M) @ A.T


def _odd_ksize(v: int, *, minimum: int = 1) -> int:
    k = max(int(v), int(minimum))
    if (k % 2) == 0:
        k += 1
    return k


def _smooth_kernel(k: int) -> np.ndarray:
    x = np.ones(max(1, int(k)), dtype=np.float64)
    return x / x.sum()


def _median_blur_u16(img_u16: np.ndarray, *, median_k: int) -> np.ndarray:
    img = np.asarray(img_u16, dtype=np.uint16)
    k = _odd_ksize(median_k, minimum=1)
    if k <= 1:
        return img
    if cv2 is None:
        return img
    return cv2.medianBlur(img, k)


def _profile_1d(img_u16: np.ndarray, which: str, *, median_k: int, kernel: np.ndarray) -> np.ndarray:
    img = _median_blur_u16(img_u16, median_k=int(median_k))
    if which == "dx":
        p = img.max(axis=0).astype(np.float64, copy=False)
    elif which == "dy":
        p = img.max(axis=1).astype(np.float64, copy=False)
    else:
        raise ValueError("which must be dx or dy")
    return np.convolve(p, kernel, mode="same")


def _shift_1d_centered(
    p_ref: np.ndarray,
    p_cur: np.ndarray,
    *,
    center: float,
    max_shift: int,
    subpixel: bool,
) -> Tuple[float, float]:
    if not np.isfinite(center):
        center = 0.0

    a = np.asarray(p_cur, dtype=np.float64)
    b = np.asarray(p_ref, dtype=np.float64)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))

    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if (not np.isfinite(norm)) or norm <= 1e-12:
        return 0.0, 0.0

    corr = np.correlate(a, b, mode="full")
    if corr.size == 0 or (not np.any(np.isfinite(corr))):
        return 0.0, 0.0

    L = len(a)
    shifts = np.arange(-(L - 1), L, dtype=np.float64)
    lo = float(center) - float(max_shift)
    hi = float(center) + float(max_shift)
    mask = (shifts >= lo) & (shifts <= hi)
    if np.any(mask):
        corr[~mask] = -np.inf

    i0 = int(np.argmax(corr))
    if not np.isfinite(corr[i0]):
        return 0.0, 0.0
    shift_int = i0 - (L - 1)
    resp = clamp(float(corr[i0]) / norm, 0.0, 1.0)

    if not subpixel:
        return float(shift_int), float(resp)

    delta = 0.0
    if 1 <= i0 < (len(corr) - 1):
        y0, y1, y2 = float(corr[i0 - 1]), float(corr[i0]), float(corr[i0 + 1])
        if np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2):
            denom = y0 - 2.0 * y1 + y2
            if np.isfinite(denom) and abs(denom) > 1e-12:
                cand = 0.5 * (y0 - y2) / denom
                if np.isfinite(cand):
                    delta = cand

    out = float(shift_int + delta)
    if not np.isfinite(out):
        out = float(shift_int)
    return out, float(resp)


def estimate_shift_from_profile_alignment(
    ref_u16: np.ndarray,
    cur_u16: np.ndarray,
    *,
    center_dx: float,
    center_dy: float,
    max_shift_px: float,
    median_k: int,
    smooth_k: int,
    use_subpixel: bool,
) -> Tuple[float, float, float]:
    ref = _ensure_raw16_bayer(ref_u16)
    cur = _ensure_raw16_bayer(cur_u16)
    if ref.shape != cur.shape:
        return 0.0, 0.0, 0.0

    kernel = _smooth_kernel(int(smooth_k))
    max_shift = max(1, int(round(float(max_shift_px))))

    dx, rx = _shift_1d_centered(
        _profile_1d(ref, "dx", median_k=int(median_k), kernel=kernel),
        _profile_1d(cur, "dx", median_k=int(median_k), kernel=kernel),
        center=float(center_dx),
        max_shift=max_shift,
        subpixel=bool(use_subpixel),
    )
    dy, ry = _shift_1d_centered(
        _profile_1d(ref, "dy", median_k=int(median_k), kernel=kernel),
        _profile_1d(cur, "dy", median_k=int(median_k), kernel=kernel),
        center=float(center_dy),
        max_shift=max_shift,
        subpixel=bool(use_subpixel),
    )

    if not np.isfinite(dx):
        dx = 0.0
    if not np.isfinite(dy):
        dy = 0.0

    resp = float(min(rx, ry))
    if not np.isfinite(resp):
        resp = 0.0
    return float(dx), float(dy), float(resp)


def estimate_shift_from_phase_alignment(
    ref_u16: np.ndarray,
    cur_u16: np.ndarray,
    *,
    max_shift_px: float,
    median_k: int,
) -> Tuple[float, float, float]:
    ref = _ensure_raw16_bayer(ref_u16)
    cur = _ensure_raw16_bayer(cur_u16)
    if ref.shape != cur.shape or cv2 is None:
        return 0.0, 0.0, 0.0

    ref_f = _median_blur_u16(ref, median_k=int(median_k)).astype(np.float32, copy=False)
    cur_f = _median_blur_u16(cur, median_k=int(median_k)).astype(np.float32, copy=False)
    ref_f = ref_f - float(np.mean(ref_f))
    cur_f = cur_f - float(np.mean(cur_f))

    if float(np.linalg.norm(ref_f)) <= 1e-6 or float(np.linalg.norm(cur_f)) <= 1e-6:
        return 0.0, 0.0, 0.0

    try:
        h, w = ref_f.shape[:2]
        window = cv2.createHanningWindow((int(w), int(h)), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ref_f, cur_f, window)
    except Exception as exc:
        log_error(None, "Tracking: phase correlation failed", exc, throttle_s=2.0, throttle_key="tracking_phase_corr")
        return 0.0, 0.0, 0.0

    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(resp):
        return 0.0, 0.0, 0.0
    mag = float(np.hypot(float(dx), float(dy)))
    if mag > float(max_shift_px):
        return float(dx), float(dy), 0.0
    return float(dx), float(dy), clamp(float(resp), 0.0, 1.0)


def _extract_obj_xy_and_flux(obj_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts SEP structured arrays and returns (xy, flux).
    """
    if isinstance(obj_xy, np.ndarray) and obj_xy.dtype.names is not None:
        if "x" not in obj_xy.dtype.names or "y" not in obj_xy.dtype.names:
            raise ValueError("SEP objects must include 'x' and 'y' fields")
        x = obj_xy["x"].astype(np.float64)
        y = obj_xy["y"].astype(np.float64)
        xy = np.column_stack([x, y])
        if "flux" not in obj_xy.dtype.names:
            raise ValueError("SEP objects must include 'flux' field")
        flux = obj_xy["flux"].astype(np.float64)
        return xy, flux

    raise ValueError("obj_xy must be a SEP structured array with fields x,y,flux")


def estimate_shift_from_flux_match(
    ref_xy: np.ndarray,
    cur_xy: np.ndarray,
    *,
    max_shift_px: float,
    min_sources: int,
) -> Tuple[float, float, float, int]:
    """
    Estimate translation to align cur_xy onto ref_xy by matching sources by flux order.
    Assumes inputs are already sorted brightest->dimmest.
    Returns (dx, dy, resp, n_used).
    """
    ref = np.asarray(ref_xy, dtype=np.float64)
    cur = np.asarray(cur_xy, dtype=np.float64)

    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"ref_xy must have shape (N,2), got {ref.shape}")
    if cur.ndim != 2 or cur.shape[1] != 2:
        raise ValueError(f"cur_xy must have shape (N,2), got {cur.shape}")

    n_ref = int(ref.shape[0])
    n_cur = int(cur.shape[0])
    n_min = int(min_sources)
    if n_ref < n_min or n_cur < n_min:
        return 0.0, 0.0, 0.0, 0

    n = min(n_ref, n_cur)
    shifts = ref[:n] - cur[:n]
    mags = np.hypot(shifts[:, 0], shifts[:, 1])
    good = mags <= float(max_shift_px)
    if not np.any(good):
        return 0.0, 0.0, 0.0, 0

    shifts = shifts[good]
    dx = float(shifts[:, 0].mean())
    dy = float(shifts[:, 1].mean())

    mag = float(np.hypot(dx, dy))
    if not np.isfinite(mag):
        return 0.0, 0.0, 0.0, 0

    resp = float(int(np.count_nonzero(good)) / n)
    return dx, dy, resp, int(np.count_nonzero(good))


def estimate_shift_from_source_matches(
    ref_xy: np.ndarray,
    cur_xy: np.ndarray,
    *,
    center_dx: float,
    center_dy: float,
    max_shift_px: float,
    min_sources: int,
) -> Tuple[float, float, float, int]:
    """
    Estimate current-frame displacement relative to the reference sources.

    sep_utils.estimate_shift_from_objects returns the translation that would
    move the current points onto the reference points. Tracking wants the
    observed displacement of the current frame relative to the reference, so
    the sign is inverted here.
    """
    ref = np.asarray(ref_xy, dtype=np.float64)
    cur = np.asarray(cur_xy, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 2 or cur.ndim != 2 or cur.shape[1] != 2:
        raise ValueError("source arrays must have shape (N,2)")
    if ref.size == 0 or cur.size == 0:
        return 0.0, 0.0, 0.0, 0

    center = np.array([float(center_dx), float(center_dy)], dtype=np.float64)
    if not np.all(np.isfinite(center)):
        center = np.zeros(2, dtype=np.float64)

    # Candidate displacement for each cur/ref pair is cur - ref.  Match after
    # removing the predicted displacement, and only retain mutual nearest
    # neighbours.  The old one-way matcher allowed several current sources to
    # claim the same reference star, which could report a high response while
    # silently jumping to a neighbouring star in a dense field.
    d = cur[:, None, :] - ref[None, :, :]
    err = d - center.reshape(1, 1, 2)
    dist2 = np.sum(err * err, axis=2)
    cur_to_ref = np.argmin(dist2, axis=1)
    ref_to_cur = np.argmin(dist2, axis=0)
    cur_idx = np.arange(cur.shape[0])
    mutual = ref_to_cur[cur_to_ref] == cur_idx
    nn_dist = np.sqrt(dist2[cur_idx, cur_to_ref])
    good = mutual & (nn_dist <= float(max_shift_px))
    if not np.any(good):
        return 0.0, 0.0, 0.0, 0

    shifts = d[cur_idx, cur_to_ref][good]
    initial_matches = int(shifts.shape[0])
    if initial_matches < int(min_sources):
        return 0.0, 0.0, 0.0, initial_matches

    # Reject inconsistent pairs around the robust common translation.  This
    # keeps cosmic rays, variable detections and stars entering/leaving the
    # frame from pulling the measured target displacement away.
    center_shift = np.median(shifts, axis=0)
    residual = np.hypot(shifts[:, 0] - center_shift[0], shifts[:, 1] - center_shift[1])
    residual_med = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_med)))
    residual_limit = max(0.75, residual_med + 3.5 * max(residual_mad, 0.10))
    inliers = residual <= residual_limit
    shifts = shifts[inliers]
    matches = int(shifts.shape[0])
    if matches < int(min_sources):
        return 0.0, 0.0, 0.0, int(matches)

    dx = float(np.median(shifts[:, 0]))
    dy = float(np.median(shifts[:, 1]))
    denom = float(max(1, min(ref.shape[0], cur.shape[0])))
    resp = clamp(float(matches / denom), 0.0, 1.0)
    return dx, dy, resp, matches


def _choose_consistent_alignment(
    candidates: list[tuple[str, float, float, float]],
    *,
    center_dx: float,
    center_dy: float,
    agree_px: float,
    lock_conf: float,
) -> _AlignmentMeasurement:
    if not candidates:
        return _AlignmentMeasurement(ok=False, reason="low_confidence")

    center = np.array([float(center_dx), float(center_dy)], dtype=np.float64)
    if not np.all(np.isfinite(center)):
        center = np.zeros(2, dtype=np.float64)
    agree = max(0.5, float(agree_px))

    # Prefer an estimator cluster over any isolated high response.  In
    # particular this prevents a repeated 1-D profile or a neighbouring star
    # from winning merely because it is close to the previous value.
    best_cluster: list[tuple[str, float, float, float]] = []
    best_key: tuple[float, float, float] | None = None
    for item in candidates:
        cluster = [
            other
            for other in candidates
            if float(np.hypot(item[1] - other[1], item[2] - other[2])) <= agree
        ]
        xy = np.array([[c[1], c[2]] for c in cluster], dtype=np.float64)
        cluster_center = np.median(xy, axis=0)
        key = (
            float(len(cluster)),
            float(sum(c[3] for c in cluster)),
            -float(np.linalg.norm(cluster_center - center)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_cluster = cluster

    if len(best_cluster) >= 2:
        reliability = {"phase": 1.0, "source": 1.15, "profile": 0.75}
        weights = np.array(
            [max(1e-3, float(c[3])) * reliability.get(c[0], 1.0) for c in best_cluster],
            dtype=np.float64,
        )
        xy = np.array([[c[1], c[2]] for c in best_cluster], dtype=np.float64)
        def _weighted_median(values: np.ndarray) -> float:
            order = np.argsort(values)
            values_sorted = values[order]
            weights_sorted = weights[order]
            cutoff = 0.5 * float(np.sum(weights_sorted))
            idx = int(np.searchsorted(np.cumsum(weights_sorted), cutoff, side="left"))
            return float(values_sorted[min(idx, len(values_sorted) - 1)])

        fused = np.array(
            [_weighted_median(xy[:, 0]), _weighted_median(xy[:, 1])],
            dtype=np.float64,
        )
        resp = clamp(float(np.average([c[3] for c in best_cluster], weights=weights)), 0.0, 1.0)
        names = "+".join(sorted({c[0] for c in best_cluster}))
        return _AlignmentMeasurement(
            ok=True,
            dx=float(fused[0]),
            dy=float(fused[1]),
            resp=float(resp),
            source=names,
            reason="ok",
        )

    # A single estimator is still useful in a sparse field.  Once locked, do
    # not accept an isolated estimate that also contradicts the motion model.
    priority = {"source": 2, "phase": 1, "profile": 0}
    ordered = sorted(
        candidates,
        key=lambda c: (
            float(np.hypot(c[1] - center[0], c[2] - center[1])),
            -float(c[3]),
            -priority.get(c[0], 0),
        ),
    )
    chosen = ordered[0]
    distance = float(np.hypot(chosen[1] - center[0], chosen[2] - center[1]))
    # A 1-D profile can correlate strongly even for unrelated noise because it
    # collapses an entire image axis.  It is therefore a guard/consensus signal,
    # never a sufficient target measurement on its own.
    single_min_resp = {"phase": 0.15, "source": 0.45, "profile": float("inf")}
    if float(chosen[3]) < single_min_resp.get(str(chosen[0]), 0.25):
        return _AlignmentMeasurement(
            ok=False,
            dx=float(chosen[1]),
            dy=float(chosen[2]),
            resp=float(chosen[3]),
            source=str(chosen[0]),
            reason="low_confidence",
        )
    if len(candidates) > 1 and float(lock_conf) >= 0.25 and distance > (2.0 * agree):
        return _AlignmentMeasurement(
            ok=False,
            dx=float(chosen[1]),
            dy=float(chosen[2]),
            resp=float(chosen[3]),
            source=str(chosen[0]),
            reason="estimator_disagreement",
        )
    return _AlignmentMeasurement(
        ok=True,
        dx=float(chosen[1]),
        dy=float(chosen[2]),
        resp=float(chosen[3]),
        source=str(chosen[0]),
        reason="ok",
    )


def _estimate_alignment(
    state: TrackingState,
    *,
    reference: RawAlignmentSignature,
    current: RawAlignmentSignature,
    center_dx: float,
    center_dy: float,
    search_radius_px: float,
    max_displacement_px: float,
) -> _AlignmentMeasurement:
    result = estimate_raw_translation(
        reference,
        current,
        center_dx=float(center_dx),
        center_dy=float(center_dy),
        search_radius_px=float(search_radius_px),
        max_displacement_px=float(max_displacement_px),
        min_response=float(state.cfg.resp_min),
        use_subpixel=bool(state.cfg.align.use_subpixel),
        max_profile_disagreement_px=float(state.cfg.rate.source_profile_disagree_px),
    )
    return _AlignmentMeasurement(
        ok=bool(result.ok),
        dx=float(result.dx),
        dy=float(result.dy),
        resp=float(result.response),
        source="raw_profile",
        reason=str(result.reason),
    )


# ============================================================
# AutoCal (RLS)
# ============================================================

def auto_reset(state: TrackingState, *, src: str = "none", theta: Optional[np.ndarray] = None) -> None:
    cfg = state.cfg.autocal
    a = state.auto
    a.ok = False
    a.src = src
    has_calibration = theta is not None

    if theta is None:
        # Prior used to initialise RLS only.  It is not a measured calibration
        # and must never be allowed to drive the mount by itself.
        theta = np.array([[0.20, 0.00, 0.0],
                          [0.00, 0.10, 0.0]], dtype=np.float64)

    a.theta = np.asarray(theta, dtype=np.float64).reshape(2, 3)
    a.P = float(cfg.P0) * np.eye(3, dtype=np.float64)

    a.A = a.theta[:, :2].copy()
    a.b = a.theta[:, 2].copy()
    a.A_pinv = None

    a.detA = float(np.linalg.det(a.A))
    try:
        a.condA = float(np.linalg.cond(a.A))
    except Exception as exc:
        log_error(None, "Tracking: failed to compute condition number", exc, throttle_s=10.0, throttle_key="tracking_cond")
        a.condA = 1e9
    a.last_upd_t = None

    _auto_recompute_pinv(state)
    if not has_calibration:
        a.A_pinv = None
        a.ok = False


def _auto_recompute_pinv(state: TrackingState) -> None:
    cfg = state.cfg.autocal
    a = state.auto

    if a.A is None:
        a.A_pinv = None
        a.ok = False
        return

    det = float(np.linalg.det(a.A))
    a.detA = det
    if (not np.isfinite(det)) or abs(det) < float(cfg.min_det):
        a.A_pinv = None
        a.ok = False
        return

    try:
        cond = float(np.linalg.cond(a.A))
    except Exception as exc:
        log_error(None, "Tracking: failed to compute condition number (auto)", exc, throttle_s=10.0, throttle_key="tracking_cond_auto")
        cond = 1e9
    a.condA = cond

    lam = float(state.cfg.calib.lambda_dls)
    lam_eff = max(lam, 0.15) if cond > float(cfg.max_cond) else lam

    try:
        a.A_pinv = compute_A_pinv_dls(a.A, lam_eff)
        a.ok = True
    except Exception as exc:
        log_error(None, "Tracking: failed to compute A_pinv (auto)", exc, throttle_s=10.0, throttle_key="tracking_pinv_auto")
        a.A_pinv = None
        a.ok = False


def auto_rls_update(state: TrackingState, *, u_az: float, u_alt: float, vx: float, vy: float, now_t: float) -> None:
    cfg = state.cfg.autocal
    if (not np.isfinite(vx)) or (not np.isfinite(vy)):
        return

    a = state.auto
    if a.theta is None or a.P is None:
        auto_reset(state, src="rls")

    theta = a.theta
    P = a.P

    phi = np.array([float(u_az), float(u_alt), 1.0], dtype=np.float64).reshape(3, 1)
    lam = float(cfg.rls_forget)

    denom = lam + float((phi.T @ P @ phi)[0, 0])
    if denom <= 1e-9 or (not np.isfinite(denom)):
        return

    K = (P @ phi) / denom
    y = np.array([float(vx), float(vy)], dtype=np.float64).reshape(2, 1)
    y_hat = theta @ phi
    err = y - y_hat

    theta_new = theta + (err @ K.T)
    P_new = (P - (K @ (phi.T @ P))) / lam

    a.theta = theta_new
    a.P = P_new
    a.A = theta_new[:, :2].copy()
    a.b = theta_new[:, 2].copy()
    a.last_upd_t = float(now_t)
    a.src = "rls"
    _auto_recompute_pinv(state)


def _get_A_pinv_use(state: TrackingState) -> Tuple[Optional[np.ndarray], np.ndarray, str, float]:
    """
    retorna (A_pinv, b, src, detA)
    """
    # solo autocalibración automática
    if state.auto.ok and state.auto.A_pinv is not None:
        b = np.asarray(state.auto.b if state.auto.b is not None else np.zeros(2), dtype=np.float64).reshape(2,)
        src = str(state.auto.src or "auto")
        return state.auto.A_pinv, b, src, float(state.auto.detA)

    # none
    b = np.zeros(2, dtype=np.float64)
    return None, b, "none", 0.0


# ============================================================
# Public API
# ============================================================

def make_tracking_state(cfg: Optional[TrackingConfig] = None) -> TrackingState:
    st = TrackingState(cfg=cfg or TrackingConfig())
    auto_reset(st, src="none")
    st.current_mode = "IDLE"
    return st


def reset_tracker(state: TrackingState, *, now_t: float, mode: str = "STABILIZE") -> None:
    state.prev_signature = None
    state.prev_t = None
    state.fail = 0
    state.last_dx_inc = 0.0
    state.last_dy_inc = 0.0
    state.vpx = 0.0
    state.vpy = 0.0
    state.vx_inst = 0.0
    state.vy_inst = 0.0
    state.resp_inc = 0.0
    state.lock_conf = 0.0
    state.rate_az = 0.0
    state.rate_alt = 0.0

    state.current_mode = str(mode)
    state.t_mode = float(now_t)

    state.key_signature = None
    state.key_t = None
    state.x_hat = 0.0
    state.y_hat = 0.0
    state.eint_x = 0.0
    state.eint_y = 0.0
    state.abs_last_t = None
    state.abs_resp_last = 0.0


def reset_keyframe(
    state: TrackingState,
    *,
    signature: Optional[RawAlignmentSignature],
    now_t: float,
) -> None:
    state.key_signature = signature
    state.key_t = float(now_t)
    state.x_hat = 0.0
    state.y_hat = 0.0
    state.eint_x = 0.0
    state.eint_y = 0.0
    state.abs_last_t = float(now_t)
    state.abs_resp_last = 0.0


def tracking_set_params(state: TrackingState, **kwargs: Any) -> None:
    """
    Actualiza config de tracking.
    """
    cfg = state.cfg

    for k, v in kwargs.items():
        try:
            if k == "resp_min":
                cfg.resp_min = float(v)
            elif k in ("align_median_k", "median_k"):
                cfg.align.median_k = int(v)
            elif k in ("align_smooth_k", "smooth_k"):
                cfg.align.smooth_k = int(v)
            elif k in ("align_max_shift_px", "max_shift_px"):
                cfg.align.max_shift_px = float(v)
            elif k in ("align_use_subpixel", "use_subpixel"):
                cfg.align.use_subpixel = bool(v)
            elif k == "kp":
                cfg.pi.kp = float(v)
            elif k == "ki":
                cfg.pi.ki = float(v)
            elif k == "kd":
                cfg.pi.kd = float(v)
            elif k == "calib_lambda_dls":
                cfg.calib.lambda_dls = float(v)
                _auto_recompute_pinv(state)
            elif k == "rls_forget":
                cfg.autocal.rls_forget = float(v)
            elif k in {
                "sep_bw",
                "sep_bh",
                "sep_thresh_sigma",
                "sep_minarea",
                "sep_max_sources",
                "sep_min_sources",
            }:
                # Accepted as no-op for old saved CLI/UI configurations.
                continue
            elif k in ("min_meas_dt_s", "rate_min_meas_dt_s"):
                cfg.rate.min_meas_dt_s = float(v)
            elif k in ("max_meas_v_px_s", "rate_max_meas_v_px_s"):
                cfg.rate.max_meas_v_px_s = float(v)
            elif k in ("max_shift_scale_min", "rate_max_shift_scale_min"):
                cfg.rate.max_shift_scale_min = float(v)
            elif k in ("lock_warmup_frames", "rate_lock_warmup_frames"):
                cfg.rate.lock_warmup_frames = int(v)
            elif k in ("lock_drop_decay", "rate_lock_drop_decay"):
                cfg.rate.lock_drop_decay = float(v)
            elif k in ("fb_max_frac", "rate_fb_max_frac"):
                cfg.rate.fb_max_frac = float(v)
            elif k in ("eint_decay_on_bad", "rate_eint_decay_on_bad"):
                cfg.rate.eint_decay_on_bad = float(v)
            elif k in ("rls_min_lock_conf", "rate_rls_min_lock_conf"):
                cfg.rate.rls_min_lock_conf = float(v)
            elif k in ("rls_min_rate_steps_s", "rate_rls_min_rate_steps_s"):
                cfg.rate.rls_min_rate_steps_s = float(v)
            elif k in ("source_resp_min", "rate_source_resp_min"):
                cfg.rate.source_resp_min = float(v)
            elif k in ("source_profile_disagree_px", "rate_source_profile_disagree_px"):
                cfg.rate.source_profile_disagree_px = float(v)
            else:
                log_error(None, f"Tracking: unknown param {k}", ValueError(f"unknown param {k}"), throttle_s=5.0, throttle_key=f"tracking_param_unknown_{k}")
        except Exception as exc:
            log_error(None, f"Tracking: failed to apply param {k}", exc, throttle_s=5.0, throttle_key=f"tracking_param_{k}")
            continue


def tracking_step(
    state: TrackingState,
    raw16: np.ndarray,
    *,
    now_t: float,
    tracking_enabled: bool = True,
    applied_rate_az: Optional[float] = None,
    applied_rate_alt: Optional[float] = None,
) -> TrackingOutput:
    """
    Un paso de tracking puro (sin tocar hardware):
    - Construye una firma compacta directamente desde RAW16 Bayer.
    - Alinea perfiles 1D de precisión y valida contra perfiles de detalle.
      Se usa tanto para incremental (v) como para corrección absoluta contra keyframe.
    - Si tracking_enabled y hay A_pinv (auto), computa targets de velocidad (µsteps/s) (pero NO envía).
      AppRunner discretiza esa velocidad en comandos MOVE al Arduino.

    raw16 debe ser un frame Bayer RAW16 uint16.
    """
    raw_align = _ensure_raw16_bayer(raw16)
    signature = build_raw_alignment_signature(
        raw_align,
        median_k=int(state.cfg.align.median_k),
        smooth_k=int(state.cfg.align.smooth_k),
    )
    n_det = int(signature.feature_count)

    if state.key_signature is None and signature.has_signal:
        reset_keyframe(state, signature=signature, now_t=now_t)

    # first frame
    if state.prev_signature is None or state.prev_t is None:
        state.prev_signature = signature
        state.prev_t = now_t
        return TrackingOutput(
            ok=False,
            mode=state.current_mode,
            resp=0.0,
            dx=0.0,
            dy=0.0,
            vx=0.0,
            vy=0.0,
            abs_resp=float(state.abs_resp_last),
            x_hat=float(state.x_hat),
            y_hat=float(state.y_hat),
            rate_az=float(state.rate_az),
            rate_alt=float(state.rate_alt),
            calib_src="none",
            detA=0.0,
            n_det=int(n_det),
            measurement_reason="initializing" if signature.has_signal else "no_signal",
            measurement_source="none",
            lock_conf=float(state.lock_conf),
            fail_count=int(state.fail),
        )

    if not state.prev_signature.has_signal and signature.has_signal:
        state.prev_signature = signature
        state.prev_t = now_t
        state.last_dx_inc = 0.0
        state.last_dy_inc = 0.0
        return TrackingOutput(
            ok=False,
            mode=state.current_mode,
            resp=0.0,
            dx=0.0,
            dy=0.0,
            vx=float(state.vx_inst),
            vy=float(state.vy_inst),
            abs_resp=float(state.abs_resp_last),
            x_hat=float(state.x_hat),
            y_hat=float(state.y_hat),
            rate_az=float(state.rate_az),
            rate_alt=float(state.rate_alt),
            calib_src=str(state.calib_src_last),
            detA=float(state.auto.detA if state.auto.ok else 0.0),
            n_det=int(n_det),
            measurement_reason="initializing",
            measurement_source="none",
            lock_conf=float(state.lock_conf),
            fail_count=int(state.fail),
        )

    dt = float(now_t - float(state.prev_t))
    if dt <= 1e-6:
        dt = 1e-6
    min_meas_dt = float(state.cfg.rate.min_meas_dt_s)
    if (not np.isfinite(min_meas_dt)) or min_meas_dt <= 1e-6:
        min_meas_dt = 0.03
    dt_meas = max(dt, min_meas_dt)

    shift_scale_min = float(state.cfg.rate.max_shift_scale_min)
    if (not np.isfinite(shift_scale_min)) or shift_scale_min <= 0.0:
        shift_scale_min = 0.20
    max_shift_inc = float(state.cfg.rate.max_shift_per_frame_px) * clamp(
        dt / float(state.cfg.rate.update_s),
        shift_scale_min,
        4.0,
    )
    max_shift_inc = min(max_shift_inc, float(state.cfg.align.max_shift_px))

    # Predict displacement from the last valid instantaneous velocity.  Since
    # prev_t is deliberately kept on bad frames, this prediction and the search
    # radius grow with the actual gap and allow reacquisition after a short
    # dropout without changing the target reference.
    if float(state.lock_conf) > 0.0 and np.isfinite(state.vx_inst) and np.isfinite(state.vy_inst):
        center_dx = float(state.vx_inst) * dt
        center_dy = float(state.vy_inst) * dt
    else:
        center_dx = float(state.last_dx_inc)
        center_dy = float(state.last_dy_inc)

    inc = _estimate_alignment(
        state,
        reference=state.prev_signature,
        current=signature,
        center_dx=float(center_dx),
        center_dy=float(center_dy),
        search_radius_px=float(max_shift_inc),
        max_displacement_px=float(max_shift_inc),
    )
    dx_inc = float(inc.dx)
    dy_inc = float(inc.dy)
    resp_inc = float(inc.resp)
    good_inc = bool(inc.ok)
    measurement_reason = str(inc.reason)
    measurement_source = str(inc.source)

    # Periodically validate accumulated error against the original target
    # keyframe.  Also attempt it immediately on an incremental failure; a valid
    # absolute match can recover lock without ever adopting the bad frame as the
    # new target.
    abs_measurement: Optional[_AlignmentMeasurement] = None
    abs_due = state.abs_last_t is None or (
        (now_t - float(state.abs_last_t)) >= float(state.cfg.keyframe.abs_corr_every_s)
    )
    if state.key_signature is not None and (abs_due or not good_inc):
        abs_search = min(
            float(state.cfg.keyframe.abs_max_px),
            max(float(max_shift_inc), 2.0 * float(state.cfg.rate.source_profile_disagree_px)),
        )
        abs_measurement = _estimate_alignment(
            state,
            reference=state.key_signature,
            current=signature,
            center_dx=float(state.x_hat + center_dx),
            center_dy=float(state.y_hat + center_dy),
            search_radius_px=float(abs_search),
            max_displacement_px=float(state.cfg.keyframe.abs_max_px),
        )
        state.abs_last_t = now_t
        state.abs_resp_last = float(abs_measurement.resp)

        if not good_inc and abs_measurement.ok:
            recovery_dx = float(abs_measurement.dx - state.x_hat)
            recovery_dy = float(abs_measurement.dy - state.y_hat)
            recovery_mag = float(np.hypot(recovery_dx, recovery_dy))
            if np.isfinite(recovery_mag) and recovery_mag <= float(max_shift_inc):
                dx_inc = recovery_dx
                dy_inc = recovery_dy
                resp_inc = float(abs_measurement.resp)
                good_inc = True
                measurement_reason = "keyframe_recovery"
                measurement_source = f"keyframe:{abs_measurement.source}"

    state.resp_inc = float(resp_inc)
    meas_clipped = False

    if good_inc:
        state.fail = 0
        state.last_dx_inc = float(dx_inc)
        state.last_dy_inc = float(dy_inc)
        state.x_hat += float(dx_inc)
        state.y_hat += float(dy_inc)

        vx_raw = float(dx_inc) / dt_meas
        vy_raw = float(dy_inc) / dt_meas
        vmax = float(state.cfg.rate.max_meas_v_px_s)
        if (not np.isfinite(vmax)) or vmax <= 1e-6:
            vmax = 200.0
        speed_raw = float(np.hypot(vx_raw, vy_raw))
        if speed_raw > vmax:
            s = float(vmax / max(speed_raw, 1e-9))
            vx = float(vx_raw * s)
            vy = float(vy_raw * s)
            meas_clipped = True
        else:
            vx = float(vx_raw)
            vy = float(vy_raw)
        state.vx_inst = vx
        state.vy_inst = vy

        # EMA (como tu script)
        a = 0.18
        state.vpx = (1.0 - a) * state.vpx + a * vx
        state.vpy = (1.0 - a) * state.vpy + a * vy
    else:
        state.fail += 1
        decay = clamp(float(state.cfg.rate.lock_drop_decay), 0.0, 1.0)
        state.vpx *= decay
        state.vpy *= decay

    warmup_frames = max(1, int(state.cfg.rate.lock_warmup_frames))
    lock_drop = clamp(float(state.cfg.rate.lock_drop_decay), 0.0, 1.0)
    if good_inc:
        state.lock_conf = clamp(float(state.lock_conf) + (1.0 / float(warmup_frames)), 0.0, 1.0)
    else:
        state.lock_conf = clamp(float(state.lock_conf) * lock_drop, 0.0, 1.0)

    if good_inc:
        state.prev_signature = signature
        state.prev_t = now_t

    # fail reset
    if state.fail >= int(state.cfg.rate.fail_reset_n):
        failed_frames = int(state.fail)
        state.rate_az = 0.0
        state.rate_alt = 0.0
        reset_tracker(state, now_t=now_t, mode="STABILIZE")
        return TrackingOutput(
            ok=False,
            mode=state.current_mode,
            resp=float(resp_inc),
            dx=float(dx_inc),
            dy=float(dy_inc),
            vx=float(state.vx_inst),
            vy=float(state.vy_inst),
            abs_resp=float(state.abs_resp_last),
            x_hat=float(state.x_hat),
            y_hat=float(state.y_hat),
            rate_az=float(state.rate_az),
            rate_alt=float(state.rate_alt),
            calib_src="none",
            detA=0.0,
            n_det=int(n_det),
            measurement_reason="lost_lock",
            measurement_source=str(measurement_source),
            lock_conf=0.0,
            fail_count=int(failed_frames),
        )

    if good_inc and abs_measurement is not None and abs_measurement.ok:
        mag_abs = float(np.hypot(abs_measurement.dx, abs_measurement.dy))
        if (
            float(abs_measurement.resp) >= float(state.cfg.keyframe.abs_resp_min)
            and mag_abs <= float(state.cfg.keyframe.abs_max_px)
            and np.isfinite(mag_abs)
        ):
            beta = float(state.cfg.keyframe.abs_blend_beta)
            state.x_hat = (1.0 - beta) * state.x_hat + beta * float(abs_measurement.dx)
            state.y_hat = (1.0 - beta) * state.y_hat + beta * float(abs_measurement.dy)

    # control (compute rates) - only if tracking_enabled
    calib_pinv, b_use, src, detA = _get_A_pinv_use(state)
    state.calib_src_last = src

    if tracking_enabled and calib_pinv is not None:
        # PI over position error x_hat/y_hat (como tu script)
        ex = float(state.x_hat)
        ey = float(state.y_hat)

        upd = float(state.cfg.rate.update_s)
        dt_ctrl = clamp(dt, 0.0, max(4.0 * upd, 0.05))
        if good_inc:
            state.eint_x = clamp(state.eint_x + ex * dt_ctrl, -float(state.cfg.pi.eint_clamp), +float(state.cfg.pi.eint_clamp))
            state.eint_y = clamp(state.eint_y + ey * dt_ctrl, -float(state.cfg.pi.eint_clamp), +float(state.cfg.pi.eint_clamp))
        else:
            eint_decay = clamp(float(state.cfg.rate.eint_decay_on_bad), 0.0, 1.0)
            state.eint_x *= eint_decay
            state.eint_y *= eint_decay

        Kp = float(state.cfg.pi.kp)
        Ki = float(state.cfg.pi.ki)
        Kd = float(state.cfg.pi.kd)

        vx_d = float(state.vpx)
        vy_d = float(state.vpy)

        if good_inc:
            v_cmd_x = -(Kp * ex + Ki * state.eint_x + Kd * vx_d)
            v_cmd_y = -(Kp * ey + Ki * state.eint_y + Kd * vy_d)
        else:
            v_cmd_x = 0.0
            v_cmd_y = 0.0

        v_target = np.array([[v_cmd_x - float(b_use[0])],
                             [v_cmd_y - float(b_use[1])]], dtype=np.float64)
        u_dot = (calib_pinv @ v_target).reshape(-1)

        rate_fb_max = float(state.cfg.rate.rate_max) * clamp(float(state.cfg.rate.fb_max_frac), 0.05, 1.0)
        lock_gain = clamp(float(state.lock_conf), 0.0, 1.0)
        rate_az_t = clamp(float(u_dot[0]), -rate_fb_max, +rate_fb_max) * lock_gain
        rate_alt_t = clamp(float(u_dot[1]), -rate_fb_max, +rate_fb_max) * lock_gain

        slew_scale = clamp(dt_ctrl / max(upd, 1e-6), 0.1, 4.0)
        slew_per_step = float(state.cfg.rate.rate_slew_per_update) * slew_scale
        state.rate_az = rate_ramp(float(state.rate_az), rate_az_t, slew_per_step)
        state.rate_alt = rate_ramp(float(state.rate_alt), rate_alt_t, slew_per_step)

        rls_min_lock = clamp(float(state.cfg.rate.rls_min_lock_conf), 0.0, 1.0)
        if good_inc and (not meas_clipped) and (float(state.lock_conf) >= rls_min_lock):
            if applied_rate_az is None:
                u_rls_az = float(state.rate_az)
            else:
                u_rls_az = float(applied_rate_az)
            if applied_rate_alt is None:
                u_rls_alt = float(state.rate_alt)
            else:
                u_rls_alt = float(applied_rate_alt)
            min_rls_rate = float(state.cfg.rate.rls_min_rate_steps_s)
            if (not np.isfinite(min_rls_rate)) or min_rls_rate < 0.0:
                min_rls_rate = 0.0
            rls_rate_mag = float(np.hypot(u_rls_az, u_rls_alt))
        else:
            u_rls_az = 0.0
            u_rls_alt = 0.0
            min_rls_rate = float("inf")
            rls_rate_mag = 0.0

        if (
            good_inc
            and (not meas_clipped)
            and (float(state.lock_conf) >= rls_min_lock)
            and rls_rate_mag >= min_rls_rate
        ):
            auto_rls_update(
                state,
                u_az=float(u_rls_az),
                u_alt=float(u_rls_alt),
                vx=float(state.vx_inst),
                vy=float(state.vy_inst),
                now_t=float(now_t),
            )

        # keyframe refresh when stable
        e_mag = float(np.hypot(ex, ey))
        if (e_mag <= float(state.cfg.keyframe.keyframe_refresh_px)) and (float(state.abs_resp_last) >= float(state.cfg.keyframe.abs_resp_min)):
            reset_keyframe(state, signature=signature, now_t=now_t)

    else:
        # no calib -> hold rates at 0
        state.rate_az = 0.0
        state.rate_alt = 0.0

    return TrackingOutput(
        ok=bool(good_inc),
        mode=str(state.current_mode),
        resp=float(resp_inc),
        dx=float(dx_inc),
        dy=float(dy_inc),
        vx=float(state.vx_inst),
        vy=float(state.vy_inst),
        abs_resp=float(state.abs_resp_last),
        x_hat=float(state.x_hat),
        y_hat=float(state.y_hat),
        rate_az=float(state.rate_az),
        rate_alt=float(state.rate_alt),
        calib_src=str(src),
        detA=float(detA),
        n_det=int(n_det),
        measurement_reason=str(measurement_reason),
        measurement_source=str(measurement_source),
        lock_conf=float(state.lock_conf),
        fail_count=int(state.fail),
    )
