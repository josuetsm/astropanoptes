# tracking.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

import numpy as np
from logging_utils import log_error
from sep_utils import sep_detect_from_raw16

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


@dataclass
class SepTrackingConfig:
    bw: int = 64
    bh: int = 64
    thresh_sigma: float = 3.0
    minarea: int = 5
    max_sources: int = 50
    min_sources: int = 1


@dataclass
class AlignmentConfig:
    median_k: int = 3
    smooth_k: int = 30
    max_shift_px: float = 50.0
    use_subpixel: bool = False


@dataclass
class CalibrationConfig:
    """
    Modelo: v_pxps = A * u + b

    - u: [u_az, u_alt] en µsteps/s (tal como envías RATE)
    - v: [vx, vy] en px/s (medido por SEP: centroides/flux)
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
    resp_min: float = 0.06
    align: AlignmentConfig = field(default_factory=AlignmentConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    pi: PIConfig = field(default_factory=PIConfig)
    rate: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    sep_track: SepTrackingConfig = field(default_factory=SepTrackingConfig)
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
    prev_obj_xy: Optional[np.ndarray] = None
    prev_align_u16: Optional[np.ndarray] = None
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

    # keyframe & absolute correction
    key_obj_xy: Optional[np.ndarray] = None
    key_align_u16: Optional[np.ndarray] = None
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


# ============================================================
# AutoCal (RLS)
# ============================================================

def auto_reset(state: TrackingState, *, src: str = "none", theta: Optional[np.ndarray] = None) -> None:
    cfg = state.cfg.autocal
    a = state.auto
    a.ok = False
    a.src = src

    if theta is None:
        # inicial razonable (no crítica)
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
        return state.auto.A_pinv, b, "auto", float(state.auto.detA)

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
    state.prev_obj_xy = None
    state.prev_align_u16 = None
    state.prev_t = None
    state.fail = 0
    state.last_dx_inc = 0.0
    state.last_dy_inc = 0.0
    state.vpx = 0.0
    state.vpy = 0.0
    state.vx_inst = 0.0
    state.vy_inst = 0.0
    state.resp_inc = 0.0
    state.rate_az = 0.0
    state.rate_alt = 0.0

    state.current_mode = str(mode)
    state.t_mode = float(now_t)

    state.key_obj_xy = None
    state.key_align_u16 = None
    state.key_t = None
    state.x_hat = 0.0
    state.y_hat = 0.0
    state.eint_x = 0.0
    state.eint_y = 0.0
    state.abs_last_t = None
    state.abs_resp_last = 0.0


def reset_keyframe(
    state: TrackingState,
    obj_xy: Optional[np.ndarray],
    *,
    align_u16: Optional[np.ndarray],
    now_t: float,
) -> None:
    state.key_obj_xy = obj_xy
    if align_u16 is None:
        state.key_align_u16 = None
    else:
        state.key_align_u16 = np.asarray(align_u16, dtype=np.uint16).copy()
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
            elif k == "sep_bw":
                cfg.sep_track.bw = int(v)
            elif k == "sep_bh":
                cfg.sep_track.bh = int(v)
            elif k == "sep_thresh_sigma":
                cfg.sep_track.thresh_sigma = float(v)
            elif k == "sep_minarea":
                cfg.sep_track.minarea = int(v)
            elif k == "sep_max_sources":
                cfg.sep_track.max_sources = int(v)
            elif k == "sep_min_sources":
                cfg.sep_track.min_sources = int(v)
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
) -> TrackingOutput:
    """
    Un paso de tracking puro (sin tocar hardware):
    - SEP sobre RAW16 con median blur k=3 para hotpixels.
    - Alineación por perfiles 1D (misma estrategia que stacking):
      max por eje + suavizado + correlación cruzada centrada.
      Se usa tanto para incremental (v) como para corrección absoluta contra keyframe.
    - Si tracking_enabled y hay A_pinv (auto), computa RATE targets (pero NO envía).
      AppRunner es quien envía RATE al Arduino.

    raw16 debe ser un frame Bayer RAW16 uint16.
    """
    raw_align = _ensure_raw16_bayer(raw16)

    cfg_sep = state.cfg.sep_track
    _, _, objects, _ = sep_detect_from_raw16(
        raw_align,
        sep_bw=int(cfg_sep.bw),
        sep_bh=int(cfg_sep.bh),
        sep_thresh_sigma=float(cfg_sep.thresh_sigma),
        sep_minarea=int(cfg_sep.minarea),
        max_sources=int(cfg_sep.max_sources),
    )

    resp_min = float(state.cfg.resp_min)

    obj_xy, _ = _extract_obj_xy_and_flux(objects)
    n_det = int(obj_xy.shape[0])

    if (not state.auto.ok) or state.auto.A_pinv is None:
        auto_reset(state, src="auto")

    # keyframe init/pending
    if state.key_obj_xy is None:
        reset_keyframe(state, obj_xy, align_u16=raw_align, now_t=now_t)
    elif isinstance(state.key_obj_xy, str) and state.key_obj_xy == "PENDING":
        reset_keyframe(state, obj_xy, align_u16=raw_align, now_t=now_t)

    # first frame
    if state.prev_align_u16 is None or state.prev_t is None:
        state.prev_obj_xy = obj_xy
        state.prev_align_u16 = raw_align.copy()
        state.prev_t = now_t
        return TrackingOutput(
            ok=True,
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
        )

    dt = float(now_t - float(state.prev_t))
    if dt <= 1e-6:
        dt = 1e-6

    max_shift_inc = float(state.cfg.rate.max_shift_per_frame_px) * clamp(
        dt / float(state.cfg.rate.update_s),
        1.0,
        4.0,
    )
    max_shift_inc = min(max_shift_inc, float(state.cfg.align.max_shift_px))

    # Vector de deriva (px) por correlación de perfiles, centrada en el último shift.
    dx_inc, dy_inc, resp_inc = estimate_shift_from_profile_alignment(
        state.prev_align_u16,
        raw_align,
        center_dx=float(state.last_dx_inc),
        center_dy=float(state.last_dy_inc),
        max_shift_px=float(max_shift_inc),
        median_k=int(state.cfg.align.median_k),
        smooth_k=int(state.cfg.align.smooth_k),
        use_subpixel=bool(state.cfg.align.use_subpixel),
    )
    state.last_dx_inc = float(dx_inc)
    state.last_dy_inc = float(dy_inc)
    mag_inc = float(np.hypot(dx_inc, dy_inc))

    good_inc = (
        float(resp_inc) >= resp_min
        and mag_inc <= max_shift_inc
        and np.isfinite(mag_inc)
    )

    state.resp_inc = float(resp_inc)

    if good_inc:
        state.fail = 0
        state.x_hat += float(dx_inc)
        state.y_hat += float(dy_inc)

        vx = float(dx_inc) / dt
        vy = float(dy_inc) / dt
        state.vx_inst = vx
        state.vy_inst = vy

        # EMA (como tu script)
        a = 0.18
        state.vpx = (1.0 - a) * state.vpx + a * vx
        state.vpy = (1.0 - a) * state.vpy + a * vy
    else:
        state.fail += 1

    if good_inc:
        state.prev_obj_xy = obj_xy
        state.prev_align_u16 = raw_align.copy()
        state.prev_t = now_t

    # fail reset
    if state.fail >= int(state.cfg.rate.fail_reset_n):
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
        )

    # ABS correction against keyframe
    if state.key_align_u16 is not None:
        if (state.abs_last_t is None) or ((now_t - float(state.abs_last_t)) >= float(state.cfg.keyframe.abs_corr_every_s)):
            dx_abs, dy_abs, resp_abs = estimate_shift_from_profile_alignment(
                state.key_align_u16,
                raw_align,
                center_dx=float(state.x_hat),
                center_dy=float(state.y_hat),
                max_shift_px=float(state.cfg.keyframe.abs_max_px),
                median_k=int(state.cfg.align.median_k),
                smooth_k=int(state.cfg.align.smooth_k),
                use_subpixel=bool(state.cfg.align.use_subpixel),
            )
            state.abs_last_t = now_t
            state.abs_resp_last = float(resp_abs)
            mag_abs = float(np.hypot(dx_abs, dy_abs))

            if (
                resp_abs >= float(state.cfg.keyframe.abs_resp_min)
                and mag_abs <= float(state.cfg.keyframe.abs_max_px)
                and np.isfinite(mag_abs)
            ):
                beta = float(state.cfg.keyframe.abs_blend_beta)
                state.x_hat = (1.0 - beta) * state.x_hat + beta * float(dx_abs)
                state.y_hat = (1.0 - beta) * state.y_hat + beta * float(dy_abs)

    # control (compute rates) - only if tracking_enabled
    calib_pinv, b_use, src, detA = _get_A_pinv_use(state)
    state.calib_src_last = src

    if tracking_enabled and calib_pinv is not None:
        # PI over position error x_hat/y_hat (como tu script)
        ex = float(state.x_hat)
        ey = float(state.y_hat)

        upd = float(state.cfg.rate.update_s)
        state.eint_x = clamp(state.eint_x + ex * upd, -float(state.cfg.pi.eint_clamp), +float(state.cfg.pi.eint_clamp))
        state.eint_y = clamp(state.eint_y + ey * upd, -float(state.cfg.pi.eint_clamp), +float(state.cfg.pi.eint_clamp))

        Kp = float(state.cfg.pi.kp)
        Ki = float(state.cfg.pi.ki)
        Kd = float(state.cfg.pi.kd)

        vx_d = float(state.vpx)
        vy_d = float(state.vpy)

        v_cmd_x = -(Kp * ex + Ki * state.eint_x + Kd * vx_d)
        v_cmd_y = -(Kp * ey + Ki * state.eint_y + Kd * vy_d)

        v_target = np.array([[v_cmd_x - float(b_use[0])],
                             [v_cmd_y - float(b_use[1])]], dtype=np.float64)
        u_dot = (calib_pinv @ v_target).reshape(-1)

        rate_az_t = clamp(float(u_dot[0]), -float(state.cfg.rate.rate_max), +float(state.cfg.rate.rate_max))
        rate_alt_t = clamp(float(u_dot[1]), -float(state.cfg.rate.rate_max), +float(state.cfg.rate.rate_max))

        state.rate_az = rate_ramp(float(state.rate_az), rate_az_t, float(state.cfg.rate.rate_slew_per_update))
        state.rate_alt = rate_ramp(float(state.rate_alt), rate_alt_t, float(state.cfg.rate.rate_slew_per_update))

        if good_inc:
            auto_rls_update(
                state,
                u_az=float(state.rate_az),
                u_alt=float(state.rate_alt),
                vx=float(state.vx_inst),
                vy=float(state.vy_inst),
                now_t=float(now_t),
            )

        # keyframe refresh when stable
        e_mag = float(np.hypot(ex, ey))
        if (e_mag <= float(state.cfg.keyframe.keyframe_refresh_px)) and (float(state.abs_resp_last) >= float(state.cfg.keyframe.abs_resp_min)):
            reset_keyframe(state, obj_xy, align_u16=raw_align, now_t=now_t)

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
    )
