# ap_types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import numpy as np


class Axis(str, Enum):
    AZ = "az"
    ALT = "alt"


class DisplayMode(str, Enum):
    RAW = "Raw"
    AUTOSTRETCH = "AutoStretch"
    LOGSTRETCH = "LogStretch"
    HISTEQ = "HistogramEq"


@dataclass(frozen=True)
class Frame:
    """
    Frame lógico del sistema.

    - raw: frame full-res (uint16 Bayer) para todo el pipeline geométrico
    - u8_view: vista opcional para preview legacy (si existe)
    """
    t_capture: float
    seq: int
    w: int
    h: int
    fmt: str

    raw: np.ndarray
    u8_view: Optional[np.ndarray] = None

    meta: Optional[Dict[str, Any]] = None  # exp_ms, gain, binning, roi, etc.


@dataclass
class SystemState:
    # connectivity
    camera_connected: bool = False
    mount_connected: bool = False

    # runtime flags
    tracking_on: bool = False
    stacking_on: bool = False

    # status strings (UI-friendly)
    camera_status: str = "DISCONNECTED"   # DISCONNECTED|OK|ERR
    mount_status: str = "DISCONNECTED"    # DISCONNECTED|OK|ERR
    tracking_status: str = "OFF"          # OFF|ON|PAUSED
    stacking_status: str = "OFF"          # OFF|ON|SAVING

    # perf metrics
    fps_capture: float = 0.0
    fps_view: float = 0.0
    fps_control_loop: float = 0.0
    frame_ms: float = 0.0

    # tracking metrics (optional)
    tracking_enabled: bool = False
    tracking_mode: str = "IDLE"
    tracking_resp: float = 0.0
    tracking_dx: float = 0.0
    tracking_dy: float = 0.0
    tracking_rate_az: float = 0.0
    tracking_rate_alt: float = 0.0
    tracking_vx: float = 0.0
    tracking_vy: float = 0.0
    tracking_abs_resp: float = 0.0

    # calib / auto-cal / bootstrap status for UI
    calib_manual_ok: bool = False
    calib_auto_ok: bool = False
    calib_src: str = "none"        # none|manual|auto|boot
    calib_det: float = 0.0
    calib_ms_az: int = 0
    calib_ms_alt: int = 0
    bootstrap_active: bool = False
    bootstrap_phase: str = "IDLE"

    # stacking metrics (optional)
    stack_frames: int = 0
    stack_snr: float = 0.0
    stack_runtime_s: float = 0.0

    stacking_enabled: bool = False
    stacking_mode: str = "IDLE"
    stacking_fps: float = 0.0
    stacking_tiles_used: int = 0
    stacking_tiles_evicted: int = 0
    stacking_frames_in: int = 0
    stacking_frames_used: int = 0
    stacking_frames_dropped: int = 0
    stacking_frames_rejected: int = 0
    stacking_last_resp: float = 0.0
    stacking_last_dx: float = 0.0
    stacking_last_dy: float = 0.0
    stacking_last_theta_deg: float = 0.0
    stacking_preview_jpeg: Optional[bytes] = None

    # platesolve metrics (optional)
    platesolve_status: str = "IDLE"
    platesolve_busy: bool = False
    platesolve_last_ok: bool = False
    platesolve_theta_deg: float = 0.0
    platesolve_dx_px: float = 0.0
    platesolve_dy_px: float = 0.0
    platesolve_resp: float = 0.0
    platesolve_n_inliers: int = 0
    platesolve_rms_px: float = 0.0
    platesolve_overlay: Any = None
    platesolve_guides: Any = None
    platesolve_debug_jpeg: Optional[bytes] = None
    platesolve_debug_info: Optional[Dict[str, Any]] = None
    platesolve_center_ra_deg: float = 0.0
    platesolve_center_dec_deg: float = 0.0

    # goto metrics (optional)
    goto_busy: bool = False
    goto_status: str = "IDLE"
    goto_synced: bool = False
    goto_last_error_arcsec: float = 0.0
    goto_J00: float = 0.0
    goto_J01: float = 0.0
    goto_J10: float = 0.0
    goto_J11: float = 0.0
