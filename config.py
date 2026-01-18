# config.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    camera_index: int = 0

    # ROI
    use_roi: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 1944
    roi_h: int = 1096

    # binning + format
    binning: int = 1
    img_format: str = "RAW16"  # recomendado para stacking a color

    # exposure/gain
    exp_ms: float = 1.0
    gain: int = 360
    auto_gain: bool = True

    # camera gamma (si aplica en SDK)
    gamma: float = 1.0

    # debayer (solo para preview/stacking si lo usas)
    debayer: str = "SDK default"  # Off (Mono) | RGGB | BGGR | GRBG | GBRG | SDK default


@dataclass
class PreviewConfig:
    # refresco visual (NO afecta captura)
    view_hz: float = 10.0

    # downsample barato por stride
    ds: int = 2

    # JPEG encode
    jpeg_quality: int = 75

    # stretch por percentiles (u8)
    stretch_plo: float = 5.0
    stretch_phi: float = 99.5

    # sleep para polling de ready (si aplica)
    ready_sleep_s: float = 0.0005


@dataclass
class MountConfig:
    rate_max: float = 600.0
    default_rate: float = 80.0
    default_nudge_ms: int = 250

    invert_az: bool = False
    invert_alt: bool = False

    baudrate: int = 115200


@dataclass
class TrackingConfig:
    track_source: str = "green"  # luma|green|full
    track_method: str = "PyramidPhaseCorr"
    track_downsample: int = 2


@dataclass
class StackingConfig:
    stack_align_source: str = "Use Tracking Delta"
    stack_resolution: str = "Full-res"


@dataclass
class PlateSolveConfig:
    focal_mm: float = 900.0
    pixel_um: float = 2.9
    binning: int = 1
    auto_solve: bool = False
    solve_every_s: float = 15.0

#@dataclass
#class GoToConfig:
    # complete
    


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    mount: MountConfig = field(default_factory=MountConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    platesolve: PlateSolveConfig = field(default_factory=PlateSolveConfig)
#    goto: GoToConfig = field(default_factory=GoToConfig)
    
    control_hz: float = 120.0

    log_to_file: bool = False
    log_path: str = "./astropanoptes.log"
