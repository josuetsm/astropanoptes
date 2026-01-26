# tracking.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
from hotpixels import hotpix_prefilter_base
from logging_utils import log_error


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class TrackingPreprocConfig:
    subtract_bg_ema: bool = True
    bg_ema_alpha: float = 0.03
    sigma_hp: float = 10.0
    sigma_smooth: float = 2.0
    bright_percentile: float = 99.3


@dataclass
class HotPixelsConfig:
    base_ksize: int = 3


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
class CalibrationConfig:
    """
    Modelo: v_pxps = A * u + b

    - u: [u_az, u_alt] en µsteps/s (tal como envías RATE)
    - v: [vx, vy] en px/s (medido por phase correlation)
    """
    calib_A: Optional[np.ndarray] = None   # 2x2 (px/s)/(µstep/s)
    calib_b: Optional[np.ndarray] = None   # 2 (px/s)
    lambda_dls: float = 0.05               # DLS para pinv

    # manual dither defaults (como tu script)
    cal_try_max: int = 6
    cal_steps_init: int = 6       # en microsteps (MOVE)
    cal_steps_max: int = 140
    cal_delay_us: int = 5000
    cal_target_px_min: float = 1.0
    cal_target_px_max: float = 5.0
    cal_resp_min: float = 0.08


@dataclass
class AutoCalConfig:
    enabled: bool = True
    rls_forget: float = 0.990
    P0: float = 2000.0
    min_det: float = 1e-4
    max_cond: float = 250.0


@dataclass
class AutoBoostConfig:
    enabled: bool = True
    rate: float = 25.0
    base_s: float = 2.0
    axis_s: float = 2.0
    settle_s: float = 0.6
    min_samples: int = 8


@dataclass
class TrackingConfig:
    preproc: TrackingPreprocConfig = field(default_factory=TrackingPreprocConfig)
    hotpixels: HotPixelsConfig = field(default_factory=HotPixelsConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    pi: PIConfig = field(default_factory=PIConfig)
    rate: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    calib: CalibrationConfig = field(default_factory=CalibrationConfig)
    autocal: AutoCalConfig = field(default_factory=AutoCalConfig)
    autoboost: AutoBoostConfig = field(default_factory=AutoBoostConfig)


# ============================================================
# State dataclasses
# ============================================================

@dataclass
class AutoCalState:
    ok: bool = False
    src: str = "none"   # none|manual_init|boot|rls

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

    # preproc background
    bg_ema: Optional[np.ndarray] = None

    # incremental tracking
    prev_reg: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    fail: int = 0

    # filtered velocity estimate (px/s)
    vpx: float = 0.0
    vpy: float = 0.0
    vx_inst: float = 0.0
    vy_inst: float = 0.0
    resp_inc: float = 0.0

    # keyframe & absolute correction
    key_reg: Optional[np.ndarray] = None
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
    current_mode: str = "IDLE"   # IDLE|STABILIZE|TRACK|AUTOBOOST
    t_mode: Optional[float] = None

    # manual calibration columns (px/fullstep) then A_micro in calib cache
    cal_az_full: Optional[np.ndarray] = None  # shape (2,)
    cal_alt_full: Optional[np.ndarray] = None # shape (2,)
    cal_A_micro: Optional[np.ndarray] = None  # 2x2 (px/s)/(µstep/s) [aquí µstep/s, no fullstep]
    cal_A_pinv: Optional[np.ndarray] = None   # 2x2
    cal_det: float = 0.0

    # autocal state
    auto: AutoCalState = field(default_factory=AutoCalState)

    # last used calibration source for control
    calib_src_last: str = "none"  # none|manual|auto


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


# ============================================================
# Small helpers
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


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


# ============================================================
# Image preproc + phase correlation
# ============================================================

def preprocess_for_phasecorr(frame_u16: np.ndarray, state: TrackingState, *, update_bg: bool = True) -> np.ndarray:
    cfg = state.cfg
    pre = cfg.preproc
    x = hotpix_prefilter_base(frame_u16, ksize=cfg.hotpixels.base_ksize)

    if pre.subtract_bg_ema:
        if state.bg_ema is None:
            state.bg_ema = x.copy()
        else:
            if update_bg:
                a = float(pre.bg_ema_alpha)
                state.bg_ema = (1.0 - a) * state.bg_ema + a * x
        x = x - state.bg_ema

    if pre.sigma_hp and pre.sigma_hp > 0:
        low = cv2.GaussianBlur(x, (0, 0), float(pre.sigma_hp))
        x = x - low

    x = np.maximum(x, 0.0)

    if pre.sigma_smooth and pre.sigma_smooth > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(pre.sigma_smooth))

    samp = x[::4, ::4] if (x.shape[0] > 64 and x.shape[1] > 64) else x
    thr = float(np.percentile(samp, float(pre.bright_percentile)))
    mask = x >= thr

    reg = np.zeros_like(x, dtype=np.float32)
    if np.any(mask):
        vals = x[mask]
        m = float(vals.mean())
        s = float(vals.std()) + 1e-6
        reg[mask] = (vals - m) / s

    return reg.astype(np.float32)


def warp_translate(img: np.ndarray, dx: float, dy: float, *, is_mask: bool = False) -> np.ndarray:
    H, W = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def phasecorr_delta(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float, float]:
    """
    devuelve dx,dy tal que warp(cur, dx,dy) ~ ref
    """
    H, W = ref.shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    shift, resp = cv2.phaseCorrelate(ref * win, cur * win)
    dx, dy = shift
    return (-float(dx), -float(dy), float(resp))


def pyramid_phasecorr_delta(ref: np.ndarray, cur: np.ndarray, levels: int = 3) -> Tuple[float, float, float]:
    if levels <= 1:
        return phasecorr_delta(ref, cur)

    ref_p = [ref]
    cur_p = [cur]
    for _ in range(levels - 1):
        ref_p.append(cv2.pyrDown(ref_p[-1]))
        cur_p.append(cv2.pyrDown(cur_p[-1]))

    dx_tot = 0.0
    dy_tot = 0.0
    resp_last = 0.0
    for lvl in reversed(range(levels)):
        if lvl != levels - 1:
            dx_tot *= 2.0
            dy_tot *= 2.0
        r = ref_p[lvl]
        c = cur_p[lvl]
        c_w = warp_translate(c, dx_tot, dy_tot)
        dx, dy, resp = phasecorr_delta(r, c_w)
        dx_tot += dx
        dy_tot += dy
        resp_last = resp
    return dx_tot, dy_tot, resp_last


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


def auto_set_from_A(state: TrackingState, *, A_micro: np.ndarray, b_pxps: Optional[np.ndarray] = None, src: str = "boot") -> None:
    A_micro = np.asarray(A_micro, dtype=np.float64).reshape(2, 2)
    if b_pxps is None:
        b_pxps = np.zeros(2, dtype=np.float64)
    b_pxps = np.asarray(b_pxps, dtype=np.float64).reshape(2,)

    theta = np.concatenate([A_micro, b_pxps.reshape(2, 1)], axis=1)  # 2x3
    auto_reset(state, src=src, theta=theta)
    state.auto.src = src


def auto_rls_update(state: TrackingState, *, u_az: float, u_alt: float, vx: float, vy: float) -> None:
    cfg = state.cfg.autocal
    if (not cfg.enabled) or (not np.isfinite(vx)) or (not np.isfinite(vy)):
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
    a.last_upd_t = time.time()
    a.src = "rls"
    _auto_recompute_pinv(state)


# ============================================================
# Manual calibration cache (A from measured columns)
# ============================================================

def calib_reset(state: TrackingState) -> None:
    state.cal_az_full = None
    state.cal_alt_full = None
    state.cal_A_micro = None
    state.cal_A_pinv = None
    state.cal_det = 0.0


def calib_set_column_fullstep(state: TrackingState, axis: str, col_full: np.ndarray, *, ms_div: int) -> None:
    """
    axis: "AZ" or "ALT"
    col_full: px/fullstep (shape (2,))
    ms_div: microstep divisor (p.ej. 64) para convertir a px/µstep (A_micro)
    """
    col_full = np.asarray(col_full, dtype=np.float64).reshape(2,)
    if axis.upper() == "AZ":
        state.cal_az_full = col_full
    else:
        state.cal_alt_full = col_full

    if state.cal_az_full is None or state.cal_alt_full is None:
        state.cal_A_micro = None
        state.cal_A_pinv = None
        state.cal_det = 0.0
        return

    # construir A_micro con columnas (px/µstep) a partir de px/fullstep y ms_div
    # Nota: en tu firmware/convención, RATE está en µsteps/s, por lo que A debe ser px/s por µstep/s.
    # col_full (px/fullstep) / ms_div = px/µstep.
    # A_micro col0 = AZ, col1 = ALT.
    # ms_div de AZ/ALT puede ser distinto; por eso lo hacemos por separado en el AppRunner,
    # y aquí asumimos que col_full ya fue dividido por su ms_div *antes* o pasamos ms_div por columna.
    # Para mantener API clara, AppRunner llamará calib_set_A_micro directamente.
    pass


def calib_set_A_micro(state: TrackingState, A_micro: np.ndarray, *, src: str = "manual") -> None:
    A = np.asarray(A_micro, dtype=np.float64).reshape(2, 2)
    state.cal_A_micro = A
    state.cal_det = float(np.linalg.det(A))

    # manual overrides -> también setea cfg.calib.calib_A
    state.cfg.calib.calib_A = A.copy()

    try:
        pinv = compute_A_pinv_dls(A, float(state.cfg.calib.lambda_dls))
    except Exception as exc:
        log_error(None, "Tracking: failed to compute A_pinv (manual)", exc, throttle_s=10.0, throttle_key="tracking_pinv_manual")
        pinv = None

    state.cal_A_pinv = pinv
    state.cfg.calib.calib_b = np.asarray(state.cfg.calib.calib_b if state.cfg.calib.calib_b is not None else np.zeros(2), dtype=np.float64)

    # opcional: seed autocal
    if state.cfg.autocal.enabled:
        auto_set_from_A(state, A_micro=A, b_pxps=np.array([0.0, 0.0], dtype=np.float64), src="manual_init")
        state.auto.src = "manual_init"


def _get_A_pinv_use(state: TrackingState) -> Tuple[Optional[np.ndarray], np.ndarray, str, float]:
    """
    retorna (A_pinv, b, src, detA)
    """
    # manual primero
    if state.cal_A_pinv is not None and state.cal_A_micro is not None:
        b = np.asarray(state.cfg.calib.calib_b if state.cfg.calib.calib_b is not None else np.zeros(2), dtype=np.float64).reshape(2,)
        return state.cal_A_pinv, b, "manual", float(state.cal_det)

    # luego auto
    if state.auto.ok and state.auto.A_pinv is not None:
        b = np.asarray(state.auto.b if state.auto.b is not None else np.zeros(2), dtype=np.float64).reshape(2,)
        return state.auto.A_pinv, b, ("boot" if state.auto.src == "boot" else "auto"), float(state.auto.detA)

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


def reset_tracker(state: TrackingState, mode: str = "STABILIZE") -> None:
    state.prev_reg = None
    state.prev_t = None
    state.fail = 0
    state.vpx = 0.0
    state.vpy = 0.0
    state.vx_inst = 0.0
    state.vy_inst = 0.0
    state.resp_inc = 0.0
    state.rate_az = 0.0
    state.rate_alt = 0.0

    state.current_mode = str(mode)
    state.t_mode = time.time()

    state.key_reg = None
    state.key_t = None
    state.x_hat = 0.0
    state.y_hat = 0.0
    state.eint_x = 0.0
    state.eint_y = 0.0
    state.abs_last_t = None
    state.abs_resp_last = 0.0


def reset_keyframe(state: TrackingState, reg_now: Optional[np.ndarray]) -> None:
    state.key_reg = reg_now
    state.key_t = time.time()
    state.x_hat = 0.0
    state.y_hat = 0.0
    state.eint_x = 0.0
    state.eint_y = 0.0
    state.abs_last_t = time.time()
    state.abs_resp_last = 0.0


def tracking_set_params(state: TrackingState, **kwargs: Any) -> None:
    """
    Actualiza config de tracking (robusto a kwargs extra).
    """
    cfg = state.cfg

    for k, v in kwargs.items():
        try:
            if k == "sigma_hp":
                cfg.preproc.sigma_hp = float(v)
            elif k == "sigma_smooth":
                cfg.preproc.sigma_smooth = float(v)
            elif k == "bright_percentile":
                cfg.preproc.bright_percentile = float(v)
            elif k == "resp_min":
                cfg.rate.max_shift_per_frame_px = float(cfg.rate.max_shift_per_frame_px)  # noop
                # resp_min vive en control loop (tracking_step)
                state.cfg.preproc  # keep
                state.cfg.__dict__  # noop
                # guardamos en un campo "virtual" por compat con tu UI
                setattr(cfg, "_resp_min", float(v))
            elif k == "kp":
                cfg.pi.kp = float(v)
            elif k == "ki":
                cfg.pi.ki = float(v)
            elif k == "kd":
                cfg.pi.kd = float(v)
            elif k == "calib_A":
                arr = np.asarray(v, dtype=np.float64).reshape(2, 2)
                cfg.calib.calib_A = arr
                # invalidar cache manual explícita si el usuario setea A directo
                state.cal_A_micro = arr.copy()
                try:
                    state.cal_A_pinv = compute_A_pinv_dls(arr, float(cfg.calib.lambda_dls))
                    state.cal_det = float(np.linalg.det(arr))
                except Exception as exc:
                    log_error(None, "Tracking: failed to update manual calibration A_pinv", exc, throttle_s=10.0, throttle_key="tracking_calib_A")
                    state.cal_A_pinv = None
                    state.cal_det = 0.0
            elif k == "calib_b":
                cfg.calib.calib_b = np.asarray(v, dtype=np.float64).reshape(2,)
            elif k == "calib_lambda_dls":
                cfg.calib.lambda_dls = float(v)
                # recompute pinv (manual y auto)
                if state.cal_A_micro is not None:
                    try:
                        state.cal_A_pinv = compute_A_pinv_dls(state.cal_A_micro, float(cfg.calib.lambda_dls))
                    except Exception as exc:
                        log_error(None, "Tracking: failed to recompute manual calibration pinv", exc, throttle_s=10.0, throttle_key="tracking_calib_lambda")
                        state.cal_A_pinv = None
                _auto_recompute_pinv(state)
            elif k == "autocal_enabled":
                cfg.autocal.enabled = bool(v)
            elif k == "rls_forget":
                cfg.autocal.rls_forget = float(v)
            elif k == "autoboost_enabled":
                cfg.autoboost.enabled = bool(v)
            elif k == "autoboost_rate":
                cfg.autoboost.rate = float(v)
            else:
                # ignore extras
                pass
        except Exception as exc:
            log_error(None, f"Tracking: failed to apply param {k}", exc, throttle_s=5.0, throttle_key=f"tracking_param_{k}")
            continue


def tracking_step(
    state: TrackingState,
    frame_u16_for_tracking: np.ndarray,
    *,
    now_t: Optional[float] = None,
    tracking_enabled: bool = True,
) -> TrackingOutput:
    """
    Un paso de tracking puro (sin tocar hardware):
    - Preproc + phasecorr incremental (v) + keyframe abs correction (x_hat/y_hat).
    - Si tracking_enabled y hay A_pinv (manual o auto), computa RATE targets (pero NO envía).
      AppRunner es quien envía RATE al Arduino.
    """
    if now_t is None:
        now_t = time.time()

    # resp_min configurable desde UI
    resp_min = float(getattr(state.cfg, "_resp_min", 0.06))

    reg = preprocess_for_phasecorr(frame_u16_for_tracking, state, update_bg=True)

    # keyframe init/pending
    if state.key_reg is None:
        reset_keyframe(state, reg)
    elif isinstance(state.key_reg, str) and state.key_reg == "PENDING":
        reset_keyframe(state, reg)

    # first frame
    if state.prev_reg is None or state.prev_t is None:
        state.prev_reg = reg
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
        )

    dt = float(now_t - float(state.prev_t))
    if dt <= 1e-6:
        dt = 1e-6

    dx_inc, dy_inc, resp_inc = phasecorr_delta(state.prev_reg, reg)
    mag_inc = float(np.hypot(dx_inc, dy_inc))

    good_inc = (
        float(resp_inc) >= resp_min
        and mag_inc <= float(state.cfg.rate.max_shift_per_frame_px)
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

    state.prev_reg = reg
    state.prev_t = now_t

    # fail reset
    if state.fail >= int(state.cfg.rate.fail_reset_n):
        state.rate_az = 0.0
        state.rate_alt = 0.0
        reset_tracker(state, mode="STABILIZE")
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
        )

    # ABS correction against keyframe
    if isinstance(state.key_reg, np.ndarray):
        if (state.abs_last_t is None) or ((now_t - float(state.abs_last_t)) >= float(state.cfg.keyframe.abs_corr_every_s)):
            dx_abs, dy_abs, resp_abs = pyramid_phasecorr_delta(state.key_reg, reg, levels=3)
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

        # keyframe refresh when stable
        e_mag = float(np.hypot(ex, ey))
        if (e_mag <= float(state.cfg.keyframe.keyframe_refresh_px)) and (float(state.abs_resp_last) >= float(state.cfg.keyframe.abs_resp_min)):
            reset_keyframe(state, reg)

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
    )
