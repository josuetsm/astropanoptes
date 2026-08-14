# ap_types.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class Axis(str, Enum):
    AZ = "az"
    ALT = "alt"


class DisplayMode(str, Enum):
    RAW = "Raw"
    AUTOSTRETCH = "AutoStretch"
    LOGSTRETCH = "LogStretch"
    HISTEQ = "HistogramEq"


class CameraStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    OK = "OK"
    ERROR = "ERROR"
    CONNECTING = "CONNECTING"


class MountStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    OK = "OK"
    ERROR = "ERROR"
    CONNECTING = "CONNECTING"


class TrackingStatus(str, Enum):
    OFF = "OFF"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class TrackingMode(str, Enum):
    IDLE = "IDLE"
    STABILIZE = "STABILIZE"
    TRACK = "TRACK"


class StackingStatus(str, Enum):
    OFF = "OFF"
    RUNNING = "RUNNING"
    SAVING = "SAVING"
    ERROR = "ERROR"


class PlatesolvingStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    OK = "OK"
    FAIL = "FAIL"


class GotoStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    OK = "OK"
    FAIL = "FAIL"
    CANCELLED = "CANCELLED"


class GotoAutocalStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    OK = "OK"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Frame:
    """
    Frame lógico del sistema.

    - raw: frame full-res (uint16 Bayer) para todo el pipeline geométrico
    - t_capture: timestamp monotónico (`time.perf_counter`) para deltas/control
    - meta["t_wall"]: timestamp Unix (`time.time`) para nombres, logs y astrometría
    - u8_view: vista opcional para preview
    - meta: metadatos varios (seq, exp_ms, gain, binning, roi, etc.)
    """

    raw: np.ndarray
    t_capture: float
    meta: Dict[str, Any] = field(default_factory=dict)
    u8_view: Optional[np.ndarray] = None


@dataclass
class CameraState:
    connected: bool = False
    status: CameraStatus = CameraStatus.DISCONNECTED
    fps_capture: float = 0.0
    fps_view: float = 0.0
    fps_control_loop: float = 0.0
    frame_ms: float = 0.0
    roll_deg: float = 0.0
    last_error: Optional[str] = None


@dataclass
class MountState:
    connected: bool = False
    status: MountStatus = MountStatus.DISCONNECTED
    last_error: Optional[str] = None


@dataclass
class TrackingState:
    enabled: bool = False
    status: TrackingStatus = TrackingStatus.OFF
    mode: TrackingMode = TrackingMode.IDLE
    ff_enabled: bool = False
    ff_ready: bool = False
    resp: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    rate_az: float = 0.0
    rate_alt: float = 0.0
    rate_fb_az: float = 0.0
    rate_fb_alt: float = 0.0
    rate_ff_az: float = 0.0
    rate_ff_alt: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    abs_resp: float = 0.0
    calib_manual_ok: bool = False
    calib_auto_ok: bool = False
    calib_src: str = "none"
    calib_det: float = 0.0
    n_det: int = 0
    measurement_valid: bool = False
    measurement_reason: str = "off"
    measurement_source: str = "none"
    error_x_px: float = 0.0
    error_y_px: float = 0.0
    error_px: float = 0.0
    lock_conf: float = 0.0
    fail_count: int = 0
    calib_ms_az: int = 0
    calib_ms_alt: int = 0
    bootstrap_active: bool = False
    bootstrap_phase: TrackingMode | str = TrackingMode.IDLE
    last_error: Optional[str] = None


@dataclass
class StackingState:
    enabled: bool = False
    status: StackingStatus = StackingStatus.OFF
    fps: float = 0.0
    tiles_used: int = 0
    tiles_evicted: int = 0
    frames_in: int = 0
    frames_used: int = 0
    frames_dropped: int = 0
    frames_rejected: int = 0
    last_resp: float = 0.0
    last_dx: float = 0.0
    last_dy: float = 0.0
    last_theta_deg: float = 0.0
    preview_jpeg: Optional[bytes] = None
    stack_frames: int = 0
    stack_snr: float = 0.0
    stack_runtime_s: float = 0.0
    last_error: Optional[str] = None


@dataclass
class PlatesolvingState:
    status: PlatesolvingStatus = PlatesolvingStatus.IDLE
    reason: Optional[str] = None
    last_ok: bool = False
    theta_deg: float = 0.0
    dx_px: float = 0.0
    dy_px: float = 0.0
    resp: float = 0.0
    n_inliers: int = 0
    rms_px: float = 0.0
    overlay: Any = None
    guides: Any = None
    debug_jpeg: Optional[bytes] = None
    debug_info: Optional[Dict[str, Any]] = None
    center_ra_deg: float = 0.0
    center_dec_deg: float = 0.0
    busy: bool = False


@dataclass
class GotoState:
    status: GotoStatus = GotoStatus.IDLE
    reason: Optional[str] = None
    diagnostics_dir: Optional[str] = None
    busy: bool = False
    synced: bool = False
    pointing_valid: bool = False
    pointing_az_deg: float = 0.0
    pointing_alt_deg: float = 0.0
    pointing_ra_deg: float = 0.0
    pointing_dec_deg: float = 0.0
    last_error_arcsec: float = 0.0
    manual_samples: int = 0
    sample_last_ok: bool = False
    sample_last_reason: Optional[str] = None
    sample_last_az_deg: float = 0.0
    sample_last_alt_deg: float = 0.0
    sample_last_roll_deg: float = 0.0
    J00: float = 0.0
    J01: float = 0.0
    J10: float = 0.0
    J11: float = 0.0
    J00_err: float = 0.0
    J01_err: float = 0.0
    J10_err: float = 0.0
    J11_err: float = 0.0
    model_non_orthogonality_deg: float = 0.0
    model_non_orthogonality_err_deg: float = 0.0
    model_camera_roll_deg: float = 0.0
    model_camera_roll_err_deg: float = 0.0
    model_camera_roll_samples: int = 0
    model_pitch_deg: float = 0.0
    model_pitch_err_deg: float = 0.0
    model_yaw_deg: float = 0.0
    model_yaw_err_deg: float = 0.0
    model_fit_samples: int = 0
    model_fit_rms_az_deg: float = 0.0
    model_fit_rms_alt_deg: float = 0.0
    model_fit_rms_arcsec: float = 0.0
    periodic_error_az_deg: float = 0.0
    periodic_error_alt_deg: float = 0.0
    periodic_model_samples: int = 0
    last_direction_az: int = 0
    last_direction_alt: int = 0
    backlash_steps_az: int = 0
    backlash_steps_alt: int = 0
    expected_stars_overlay_enabled: bool = False
    expected_stars_overlay_count: int = 0
    expected_stars_overlay_source: str = ""
    expected_stars_overlay_reason: Optional[str] = None
    autocal_status: GotoAutocalStatus = GotoAutocalStatus.IDLE
    autocal_reason: Optional[str] = None
    autocal_last_ok: bool = False
    autocal_drift_px_s_x: float = 0.0
    autocal_drift_px_s_y: float = 0.0
    autocal_J_pix_per_step_00: float = 0.0
    autocal_J_pix_per_step_01: float = 0.0
    autocal_J_pix_per_step_10: float = 0.0
    autocal_J_pix_per_step_11: float = 0.0
    autocal_az_deg: float = 0.0
    autocal_alt_deg: float = 0.0
    autocal_radius_deg: float = 0.0


@dataclass
class AppState:
    camera: CameraState = field(default_factory=CameraState)
    mount: MountState = field(default_factory=MountState)
    tracking: TrackingState = field(default_factory=TrackingState)
    stacking: StackingState = field(default_factory=StackingState)
    platesolving: PlatesolvingState = field(default_factory=PlatesolvingState)
    goto: GotoState = field(default_factory=GotoState)

    def update(self, patch: Dict[str, Dict[str, Any]]) -> None:
        for section, updates in patch.items():
            if not hasattr(self, section):
                raise KeyError(f"unknown AppState section: {section}")
            sub = getattr(self, section)
            if not isinstance(updates, dict):
                raise TypeError(f"state patch for {section} must be dict")
            for key, value in updates.items():
                if not hasattr(sub, key):
                    raise KeyError(f"unknown field {section}.{key}")
                setattr(sub, key, value)

    def snapshot(self) -> "AppState":
        return AppState(
            camera=replace(self.camera),
            mount=replace(self.mount),
            tracking=replace(self.tracking),
            stacking=replace(self.stacking),
            platesolving=replace(self.platesolving),
            goto=replace(self.goto),
        )
