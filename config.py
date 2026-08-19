# config.py
from __future__ import annotations

import os
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

    # exposure/gain/black level.  The Mars-C preset for its low-read-noise
    # region (gain 380) uses offset 350; gain 360 with the factory offset 12
    # clips a large part of a dark frame at the digital floor.
    exp_ms: float = 100.0
    gain: int = 360
    offset: int = 350
    auto_gain: bool = False

    # camera gamma (si aplica en SDK)
    gamma: float = 1.0

    # camera roll (deg). 0 => +x aligned with az-axis (east)
    roll_deg: float = 0.0

    # debayer (solo para preview/stacking si lo usas)
    debayer: str = "SDK default"  # Off (Mono) | RGGB | BGGR | GRBG | GBRG | SDK default


@dataclass
class PreviewConfig:
    # refresco visual (NO afecta captura)
    view_hz: float = 10.0

    # JPEG encode
    jpeg_quality: int = 75

    # stretch por percentiles (u8)
    stretch_plo: float = 5.0
    stretch_phi: float = 99.5

    # sleep para polling de ready (si aplica)
    ready_sleep_s: float = 0.0005

    # Optional live overlay projected from the fitted GoTo pointing model.
    expected_stars_mag_limit: float = 15.0
    expected_stars_max: int = 300


@dataclass
class MountConfig:
    # Use AUTO for ESP32 Bluetooth SPP auto-discovery (recommended on macOS).
    # You can override with ASTROPANOPTES_MOUNT_PORT, e.g. /dev/cu.AstroPanoptes-ESP32
    port: str = os.environ.get("ASTROPANOPTES_MOUNT_PORT", "AUTO")
    baudrate: int = int(os.environ.get("ASTROPANOPTES_MOUNT_BAUDRATE", "115200"))

    rate_max: float = 600.0
    default_rate: float = 80.0

    invert_az: bool = False
    invert_alt: bool = False
    ms_az: int = 64
    ms_alt: int = 64
    slew_steps_az: int = 600
    slew_steps_alt: int = 600
    slew_delay_us_az: int = 1800
    slew_delay_us_alt: int = 1800


@dataclass
class TrackingConfig:
    resp_min: float = 0.25
    sidereal_ff_enabled: bool = True
    sidereal_ff_update_hz: float = 2.0
    sidereal_ff_gain: float = 1.0
    sidereal_ff_dt_s: float = 1.0
    sidereal_ff_cond_max: float = 5_000.0
    sidereal_ff_hold_s: float = 8.0
    sidereal_ff_slew_per_s: float = 120.0


@dataclass
class StackingConfig:
    enabled_init: bool = False

    # batching / queue
    batch_size: int = 10
    max_queue: int = 80  # >= batch_size*4 recomendado

    # alignment
    resp_min: float = 0.25
    outlier_k_mad: float = 3.0
    align_median_k: int = 3
    smooth_k: int = 30
    max_shift_px: int = 50
    use_subpixel: bool = True

    # drizzle / mosaic
    drizzle_scale: float = 1.0     # UI presets: 1.0 (off), 2.0, 3.0
    pixfrac: float = 0.8
    tile_size_out: int = 512
    max_tiles: int = 64

    # color
    color_mode: Literal["mono", "rgb"] = "mono"
    bayer_pattern: str = "RGGB"

    # preview
    preview_hz: float = 1.0
    preview_log_vmin: float = 5.0

    # backend
    backend: Literal["cpu"] = "cpu"


@dataclass
class SepConfig:
    minarea: int = 5
    bw: int = 64
    bh: int = 64
    thresh_sigma: float = 3.0


@dataclass
class PlatesolvingConfig:
    # Instrument (SI)
    pixel_size_m: float = 2.9e-6
    focal_m: float = 0.9
    # Off by default: real sessions saw the roll estimate confused with the
    # drift-correction orientation, producing a wrong rotation prior.
    rotation_prior_enable: bool = False
    rotation_prior_tol_deg: float = 45.0
    rotation_prior_roll_offset_deg: float = 0.0
    rotation_prior_az_step_deg: float = 0.05

    # Sanity check against known optics: a candidate triplet match is rejected
    # if its fitted plate scale deviates from pixel_size_m/focal_m by more
    # than this fraction. Real matches recover the true scale to well under
    # 1%; a wrong triplet (accepted by the loose side-length tolerance for
    # short/tight triangles) typically implies a scale far outside this band.
    scale_tol_frac: float = 0.04

    # The first trustworthy position is established from several genuinely
    # different camera frames: only the first frame performs the expensive
    # triplet search, and the following frames verify the projected Gaia field
    # using the previous WCS and the expected sidereal drift. This is the real
    # safety net for a light-polluted sky like Santiago's, where a single
    # frame often only has 3-4 real stars (min_inliers below has to be set low
    # to match): a lone low-inlier solve is cheap to fake by coincidence
    # against a large catalog, but reproducing the *same* pointing/scale/roll
    # across independent frames by chance is not. Requiring only 1 (i.e. no
    # confirmation at all) defeats that net entirely and was what let a false
    # 3-star match through during real observing.
    initial_consensus_count: int = 3
    initial_consensus_timeout_s: float = 20.0
    consensus_pointing_tol_arcsec: float = 30.0
    consensus_scale_tol_frac: float = 0.02
    consensus_roll_tol_deg: float = 3.0
    fast_prior_match_radius_px: float = 24.0
    fast_prior_center_shift_px: float = 64.0
    fast_prior_rotation_tol_deg: float = 5.0
    fast_prior_max_age_s: float = 60.0

    # App-level control
    auto_solve: bool = False
    solve_every_s: float = 15.0

    # Debug
    debug_input_stats: bool = False
    # Persist the exact RAW16 frame(s), configuration and solver result for
    # each explicit solve. Set ASTROPANOPTES_DIAGNOSTICS=0 to disable at run time.
    diagnostics_enabled: bool = True
    diagnostics_dir: str = "stack_output/goto_diagnostics"

    # Image processing
    max_det: int = 250
    det_thresh_sigma: float = 3.5
    det_minarea: int = 5
    point_sigma: float = 1.2  # sigma for gaussian blur of point-maps

    # Temporal source confirmation. SEP still runs on each native RAW16 frame
    # (median 3x3 + local background subtraction), but a source only reaches
    # the plate solver after being tracked through at least 10 frames. The
    # tracker estimates and removes frame-to-frame stellar drift first.
    temporal_detection_enabled: bool = True
    # Camera reconfiguration restarts the stream asynchronously.  Explicit
    # solves wait briefly for the first new RAW frame instead of failing with
    # NO_FRAME during that normal transition.
    frame_wait_timeout_s: float = 3.0
    # Hard wall-clock budget for an explicit solve, including temporal frame
    # collection and independent consensus. Cooperative checkpoints stop the
    # expensive catalog search without killing its worker thread.
    # Real observing sessions are weather-limited; a shorter budget matched
    # what actually worked without stalling on a closing sky.
    total_timeout_s: float = 75.0
    temporal_window_frames: int = 12
    temporal_min_hits: int = 10
    # Must cover at least ten distinct frames even at long exposure. The old
    # 8 s default made the configured 10-hit requirement impossible at
    # exposures around 1 s or longer.
    temporal_detection_timeout_s: float = 20.0
    temporal_match_radius_px: float = 4.0
    temporal_max_drift_per_frame_px: float = 32.0
    temporal_min_drift_response: float = 0.05

    # Gaia cache + query
    cache_dir: str = "~/.cache/gaia_cones"
    table_name: str = "gaiadr3.gaia_source"
    columns: tuple[str, ...] = ("source_id", "ra", "dec", "phot_g_mean_mag")
    gmax: float = 15.0
    nside: int = 16
    order: str = "ring"
    # Same reasoning as download_missing_tiles: keep solves offline by default.
    bright_catalog_enabled: bool = False
    bright_catalog_margin_deg: float = 0.15
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
    # Dense real fields (e.g. Milky Way) needed a wider search budget than
    # these defaults to reliably find a valid triplet.
    triplet_max_trials: int = 1500
    max_i_scan: int = 5000

    # Matching
    match_max_px: float = 3.5  # in full-res pixels
    match_tol_arcsec: float = 5.0
    pred_margin_arcsec: float = 25.0
    # A triplet hypothesis already contributes up to three matches by
    # construction, so 3 is the structural floor: a field with only 3 real
    # stars (common under Santiago's light pollution, in this instrument's
    # narrow ~0.36x0.20 deg FoV) can never produce a "validation" inlier
    # beyond the seed triplet, no matter how good the detector is. Guarding
    # against a false 3-star coincidence is initial_consensus_count's job
    # (above), not this single-frame count's.
    min_inliers: int = 3
    min_validation_inliers: int = 0
    max_rms_px: float = 2.5
    # The search radius is also the declared pointing uncertainty. A fitted
    # optical center outside that cone is not a valid answer for the request.
    # Set the factor to 0 to disable this guard for special offline uses.
    max_center_offset_factor: float = 1.0
    max_center_offset_margin_deg: float = 0.0
    N_det: int = 30
    N_seed: int = 8
    # Clipped stars can have severely biased flux/centroids and used to
    # dominate the brightest-three seed triplet. They remain available as
    # validation detections, but interior detections are preferred as seeds.
    seed_edge_margin_px: float = 8.0

    # Search area (Gaia cone radius)
    search_radius_deg: float | None = 3.0
    search_radius_factor: float = 1.4  # radius ~= factor * (diag/2)

    # Download missing tiles. Off by default: the full local Gaia cache
    # (mag <= gmax) already covers real sessions, and downloads over a
    # residential link were the main source of stalls in practice.
    download_missing_tiles: bool = False

    # Guides / labeling
    guide_n: int = 3
    simbad_radius_arcsec: float = 2.0
    simbad_retries: int = 3
    simbad_backoff_s: float = 0.6

@dataclass
class GoToConfig:
    # GoTo defaults
    tol_arcsec: float = 10.0
    max_iters: int = 1
    gain: float = 1.0
    settle_s: float = 0.25
    # Measured physical slack consumed when an axis reverses. These pulses do
    # not represent sky motion and therefore are not added to the model's
    # commanded-position counter.
    backlash_steps_az: int = 0
    backlash_steps_alt: int = 10
    max_step_per_iter: int = 0
    slew_delay_us: int = 1800
    # Hard lower bound for adaptive GoTo delays. Smaller delays mean more
    # speed. The loaded firmware accelerates and brakes around this target, so
    # it remains the maximum-speed limit without needing a firmware update.
    slew_min_delay_us: int = 400
    slew_full_speed_distance_deg: float = 20.0
    # Until a model has been fitted from independent plate-solve samples, only
    # short verification moves are allowed. A sync establishes position, not
    # motor direction/scale.
    max_unfitted_goto_deg: float = 3.0
    # A single model-only command should never produce an unexpectedly large
    # slew. Longer routes must be split into verified stops or explicitly
    # override this value in an advanced request.
    max_goto_distance_deg: float = 10.0
    stages: int = 1
    platesolving_feedback: bool = False

    # One self-contained directory per GoTo/AutoCal/model-fit operation. It
    # links pre-slew model state, SEP/plate-solving raws, drift stacks and the
    # emitted motor commands in an append-only timeline.
    diagnostics_enabled: bool = True
    diagnostics_dir: str = "stack_output/goto_diagnostics"

    # Safe operating window
    alt_min_deg: float = 10.0
    alt_max_deg: float = 90.0

    # Calibration defaults (random samples within radius)
    calib_samples: int = 3
    calib_max_radius_deg: float = 1.0


@dataclass
class SimulationConfig:
    # Demo mode replaces the physical camera/mount with deterministic simulators.
    enabled: bool = False
    seed: int | None = None

    # Initial nominal pointing. The true pointing starts with a small random
    # mechanical offset around this position.
    initial_az_deg: float = 180.0
    initial_alt_deg: float = 45.0
    random_mount_tilt_deg: float = 1.0
    random_camera_roll_deg: float = 1.0

    # Simulated camera frame. A smaller frame keeps the demo responsive while
    # preserving the same plate scale convention used by platesolving.py.
    frame_w: int = 1280
    frame_h: int = 720
    fps: float = 8.0
    background_adu: float = 700.0
    noise_adu: float = 18.0
    star_sigma_px: float = 1.25
    star_flux_adu: float = 18000.0
    max_render_stars: int = 240

    # Gaia catalog reuse. Missing cache tiles are not downloaded by the camera
    # simulator; the real plate solver can still download them if configured.
    # Keep synthetic fallback opt-in so demo sessions do not hide missing Gaia data.
    catalog_radius_deg: float = 1.2
    catalog_reload_margin_deg: float = 0.35
    allow_synthetic_fallback: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    mount: MountConfig = field(default_factory=MountConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    sep: SepConfig = field(default_factory=SepConfig)
    platesolving: PlatesolvingConfig = field(default_factory=PlatesolvingConfig)
    goto: GoToConfig = field(default_factory=GoToConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    control_hz: float = 120.0
    state_publish_hz: float = 10.0
    pointing_hz: float = 2.0

    log_to_file: bool = False
    log_path: str = "./astropanoptes.log"
