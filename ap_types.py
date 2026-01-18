# ap_types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import numpy as np


class Axis(str, Enum):
    AZ = "az"
    ALT = "alt"


class TrackSource(str, Enum):
    LUMA = "luma"
    GREEN = "green"
    FULL = "full"


class DisplayMode(str, Enum):
    RAW = "Raw"
    AUTOSTRETCH = "AutoStretch"
    LOGSTRETCH = "LogStretch"
    HISTEQ = "HistogramEq"


@dataclass(frozen=True)
class Frame:
    """
    Frame lógico del sistema.

    - u8_view: imagen barata (uint8) para preview/tracking (H x W)
    - raw: frame full-res (p.ej. uint16 o uint8) opcional para stacking/platesolve
    """
    t_capture: float
    seq: int
    w: int
    h: int
    fmt: str

    u8_view: np.ndarray
    raw: Optional[np.ndarray] = None

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

    # platesolve metrics (optional)
    solve_status: str = "IDLE"  # IDLE|RUNNING|OK|ERR
    ra_dec: Tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    scale_arcsec_px: float = 0.0
