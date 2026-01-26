# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    exp_ms: float = 100.0
    gain: int = 360
    auto_gain: bool = False

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
    enabled_init: bool = False

    # batching / queue
    batch_size: int = 10
    max_queue: int = 80  # >= batch_size*4 recomendado

    # alignment
    align_downsample: int = 4
    resp_min: float = 0.08
    outlier_k_mad: float = 3.0

    # drizzle / mosaic
    drizzle_scale: float = 2.0     # 1.5,2.0,2.5,3.0
    pixfrac: float = 0.8
    tile_size_out: int = 512
    max_tiles: int = 64

    # color
    color_mode: Literal["mono", "rgb"] = "mono"
    bayer_pattern: str = "RGGB"

    # preview
    preview_hz: float = 1.0

    # backend
    backend: Literal["cpu"] = "cpu"


@dataclass
class PlatesolveConfig:
    # Instrument (SI)
    pixel_size_m: float = 2.9e-6
    focal_m: float = 0.9

    # App-level control
    auto_solve: bool = False
    solve_every_s: float = 15.0

    # Debug
    debug_input_stats: bool = False

    # Image processing
    downsample: int = 2
    max_det: int = 250
    det_thresh_sigma: float = 3.5
    det_minarea: int = 5
    point_sigma: float = 1.2  # sigma for gaussian blur of point-maps
    sep_bw: int = 64
    sep_bh: int = 64
    sep_thresh_sigma: float = 3.5
    sep_minarea: int = 5

    # Gaia cache + query
    cache_dir: str = "~/.cache/gaia_cones"
    table_name: str = "gaiadr3.gaia_source"
    columns: tuple[str, ...] = ("source_id", "ra", "dec", "phot_g_mean_mag")
    gmax: float = 17.0
    nside: int = 64
    order: str = "ring"
    prefer_parquet: bool = True
    row_limit: int = -1
    retries: int = 3
    backoff_s: float = 3.0

    # Solve (Option C)
    theta_step_deg: float = 15.0
    theta_refine_step_deg: float = 3.0
    theta_refine_span_deg: float = 12.0
    triplet_tol_arcsec: float = 3.0
    triplet_sigma_arcsec: float = 0.6
    triplet_max_trials: int = 500
    max_i_scan: int = 2000

    # Matching
    match_max_px: float = 3.5  # in downsampled pixels
    match_tol_arcsec: float = 5.0
    pred_margin_arcsec: float = 25.0
    min_inliers: int = 3
    N_det: int = 30
    N_seed: int = 3

    # Search area (Gaia cone radius)
    search_radius_deg: float | None = 1.0
    search_radius_factor: float = 1.4  # radius ~= factor * (diag/2)

    # Download missing tiles
    download_missing_tiles: bool = True

    # Guides / labeling
    guide_n: int = 3
    simbad_radius_arcsec: float = 2.0
    simbad_retries: int = 3
    simbad_backoff_s: float = 0.6

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
    platesolve: PlatesolveConfig = field(default_factory=PlatesolveConfig)
#    goto: GoToConfig = field(default_factory=GoToConfig)
    
    control_hz: float = 120.0

    log_to_file: bool = False
    log_path: str = "./astropanoptes.log"
