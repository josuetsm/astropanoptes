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

    # Retardo maximo entre microsteps al emular una velocidad de seguimiento.
    #
    # Es tentador subirlo para que los pasos salgan repartidos al ritmo pedido
    # en vez de en rafagas -- suena mejor. Pero un MOVE nuevo sobre un eje
    # sobrescribe el plan pendiente en el firmware, asi que un lote pausado se
    # queda a medias cuando llega la orden siguiente. El acumulador ya descuenta
    # esos pasos como ejecutados, el control se queda corto sin saberlo, y la
    # realimentacion sube pidiendo cada vez mas: se probo con 500 ms y el
    # resultado fue que la imagen daba tirones visibles a cada correccion.
    #
    # Repartir los pasos de verdad exige emitir por ventanas, sincronizado con
    # la duracion del lote, no solo alargar el retardo.
    rate_emul_max_delay_us: int = 50_000


@dataclass
class FocuserConfig:
    """Enfocador: tercer motor del CNC shield sobre la ruedita de foco.

    No hay encoder ni final de carrera, asi que todo es relativo y limitado por
    ``max_travel_steps``: el recorrido util del enfocador es corto y forzarlo
    contra el tope es la unica forma real de romperlo.
    """
    enabled: bool = True

    # Movimiento manual (un toque de "acercar"/"alejar").
    step_size: int = 200
    delay_us: int = 900
    profile: str = "smooth"
    invert: bool = False
    backlash_steps: int = 0

    # Busqueda automatica. El barrido grueso localiza la V, el fino la afina.
    autofocus_coarse_step: int = 400
    autofocus_coarse_points: int = 9
    autofocus_fine_step: int = 100
    autofocus_fine_points: int = 7
    autofocus_settle_s: float = 0.8
    autofocus_frames: int = 3

    # --- Busqueda guiada por presets ---
    # Con una posicion conocida no hace falta barrer todo el recorrido: basta
    # una ventana alrededor. Su ancho sale de la dispersion medida en este
    # equipo -- el foco cambia de noche a noche con la temperatura -- y no de
    # una constante. Si el prior falla, se cae a la busqueda general.
    autofocus_prior_enabled: bool = True
    autofocus_prior_window_steps: int = 800      # ventana minima sin historial
    autofocus_prior_window_sigma: float = 4.0    # k * dispersion observada
    autofocus_prior_max_window_steps: int = 6000
    autofocus_prior_points: int = 7
    # Cuan cerca hay que estar de un preset para asumir que es el que esta puesto.
    autofocus_prior_match_steps: int = 1500
    history_max: int = 20

    # Tope de seguridad para el recorrido acumulado desde el cero de sesion.
    # Solo se usa mientras NO hay homing: sin origen conocido no se puede hacer
    # mejor que limitar simetricamente alrededor de donde estaba al arrancar.
    max_travel_steps: int = 20000

    # --- Homing aproximado ---
    # No hay final de carrera, pero el pinon tiene dientes rotos: al retraer del
    # todo sigue girando en banda sin mover nada. Eso da un cero mecanico
    # repetible, que es justo lo que hace que una posicion guardada signifique
    # algo de una sesion a otra.
    home_travel_steps: int = 32000       # recorrido util (2.5 vueltas a 1/64)
    home_overshoot_steps: int = 3000     # extra para entrar en la zona de patinaje
    home_extend_is_positive: bool = True # +1 extiende, -1 retrae hacia el cero
    presets_path: str = "calibration_frames/focus_presets.json"


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

    # Fallos consecutivos de tracking_step tolerados antes de rendirse. Un frame
    # malo suelto (una alineacion que revienta, una racha de viento) no puede
    # apagar la sesion: se reinicia la referencia y se sigue. Lo que no puede es
    # reintentar para siempre si el fallo es permanente.
    max_consecutive_step_failures: int = 5

    # Muestreo a disco de cada ciclo de tracking (error medido, tasas
    # comandadas, fuente de calibracion, etc.) para poder debuggear o analizar
    # una sesion despues, sin depender de lo que haya quedado en pantalla.
    # Desacoplado de state_publish_hz: la UI puede refrescar rapido sin que el
    # CSV crezca a ese mismo ritmo en sesiones de varias horas.
    log_enabled: bool = True
    log_hz: float = 1.0


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

    # Tolerancias con las que se compara un solve nuevo contra el anterior ya
    # confirmado (camino rapido de verificacion). No son un consenso: el
    # consenso inicial sobre varios fotogramas independientes se elimino porque
    # la estrategia es resolver sobre el mosaico apilado, que es *una* imagen.
    # Pedirle fotogramas distintos a un mosaico no confirma nada; con el
    # apilado parado ni siquiera termina. Lo que ocupa su lugar como red de
    # seguridad es min_validation_inliers, mas abajo.
    fresh_frame_timeout_s: float = 20.0
    verify_pointing_tol_arcsec: float = 30.0
    verify_scale_tol_frac: float = 0.02
    verify_roll_tol_deg: float = 3.0
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
    # collection. Cooperative checkpoints stop the expensive catalog search
    # without killing its worker thread.
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

    # Catalog depth relative to what the image resolves. The cone is loaded at
    # gmax, but only its brightest stars can ever match: the solver compares the
    # N_det brightest detections, so stars fainter than the image reaches just
    # multiply spurious triplets and cost (the search is ~quadratic in catalog
    # size). The cap is a density: factor * N_det stars per field-of-view worth
    # of sky, so it follows the search radius and the FOV instead of being a
    # fixed magnitude. Measured on a real field, 1.5 keeps the solve well above
    # min_inliers while cutting a 94 s solve to a few seconds. Set to 0 to load
    # the cone at full depth.
    catalog_density_factor: float = 1.5

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
    # Un triplete aporta tres emparejamientos por construccion: siempre encaja
    # consigo mismo. Por eso un solve de 3 inliers no es evidencia de nada por
    # bajo que sea su rms; de hecho el rms sale ridiculamente bajo justo por
    # ser un ajuste exacto (se vieron matches falsos con rms de 0.15 px, roll
    # disparatado y apuntado a grados del real). Lo que convierte un solve en
    # creible son los inliers de *validacion*: estrellas que confirman la
    # hipotesis aparte de las tres que la definieron.
    #
    # Esta es la red de seguridad que sustituye al consenso multi-fotograma,
    # ahora que se resuelve sobre el mosaico apilado. El mosaico llega mas
    # profundo que un fotograma suelto, asi que un acierto real confirma
    # varias estrellas de sobra: los solves buenos de campo real dieron 7-9
    # inliers. Un campo demasiado pobre para llegar aqui no es un campo que
    # convenga creerse.
    min_inliers: int = 5
    min_validation_inliers: int = 2
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
    # Axis non-orthogonality. A real session measured ~1.6 deg on this mount,
    # so the demo draws from a band that actually contains that value.
    random_mount_tilt_deg: float = 2.0
    # Camera roll. Measured around -3.5 deg in a real session; a +/-1 deg demo
    # made the rotation prior look far easier than it is.
    random_camera_roll_deg: float = 4.0

    # Cycloidal transmission error: the dominant mechanical effect on this
    # mount. Tantos lobulos por vuelta de salida como reduccion tenga el eje,
    # asi que el periodo es siempre una vuelta de motor: 12800 microsteps a
    # 1/64, que son 8 deg de salida en azimut (45:1) y 4 deg en altitud
    # (90.5:1). Its amplitude
    # is drawn in degrees of output; the resulting swing in *locally measured*
    # scale is amplitude*2*pi/period, about 20% at the top of this band, which
    # is what made short calibration moves read 87% of nominal while a
    # full-cycle move read 100.5%.
    #
    # Los dos ejes no comparten mecanica y sus errores dominantes son opuestos:
    # azimut lleva un cicloidal *impreso en 3D*, que casi no tiene juego pero
    # riza mucho; altitud lleva un planetario comprado, que apenas riza pero
    # tiene el juego concentrado en el engrane. Sortear el mismo rango para los
    # dos hacia que la simulacion no se pareciera a esta montura en particular,
    # y un test cerrado contra ella no probaba lo que hacia falta.
    transmission_error_deg_min_az: float = 0.08
    transmission_error_deg_max_az: float = 0.25
    transmission_error_deg_min_alt: float = 0.002
    transmission_error_deg_max_alt: float = 0.02

    # Backlash consumed when an axis reverses, in microsteps. These pulses move
    # the motor but not the sky, so a fit that ignores them mismodels every
    # direction change.
    # Por eje, por la misma razon: el cicloidal impreso de azimut cierra casi
    # sin juego, mientras el planetario de altitud trae el tipico de un
    # reductor comercial -- del orden de 15-60 arcmin en la salida, que a
    # 0.00031 deg/paso son entre 800 y 3200 microsteps.
    backlash_steps_min_az: int = 0
    backlash_steps_max_az: int = 40
    backlash_steps_min_alt: int = 800
    backlash_steps_max_alt: int = 3200

    # Foco. El demo arranca enfocado por defecto: cualquier desenfoque de
    # arranque ensancharia las PSF de todas las demas demos (plate solving,
    # tracking, stacking) y cambiaria resultados que no tienen que ver con el
    # enfocador. Subiendo esto, el foco real se sortea dentro de
    # +-focus_best_offset_steps y la busqueda automatica tiene algo que
    # encontrar; con 0 la curva es plana y no prueba nada.
    focus_best_offset_steps: int = 0
    focus_blur_px_per_step: float = 0.010
    focus_max_defocus_sigma_px: float = 12.0

    # Simulated camera frame. Matches the real Mars-C sensor (roi_w/roi_h in
    # CameraConfig) so the demo's field of view is the same as the physical
    # camera's, not an arbitrarily smaller crop.
    frame_w: int = 1944
    frame_h: int = 1096
    fps: float = 8.0
    background_adu: float = 700.0
    noise_adu: float = 18.0
    # Seeing-limited PSF. A real Santiago session measured 5.3" FWHM, which at
    # 0.66"/px is sigma ~3.4 px. The old 1.25 px default modelled a ~2" night
    # that this site does not have, making plate solving look far easier than
    # it is: stars were compact and well separated at any exposure.
    star_sigma_px: float = 3.0
    # Brightness of a G=12 star, in ADU, at the reference exposure/gain.
    # Deliberately *not* a "visible stars" knob: there is no separate
    # magnitude or star-count cutoff anywhere in the renderer. A star's peak
    # amplitude is compared against the same sky-background + read noise every
    # other part of this file already computes, exactly like a real sensor;
    # whether it rises above that floor is what decides if it is seen, and it
    # fades in continuously as exposure, gain or sky brightness change instead
    # of blinking on/off at a threshold.
    #   sigma_total   = sqrt(background_adu + noise_adu**2) = sqrt(700+18**2) = 32 ADU
    #   thresh_sigma  = 3.0, same detection threshold SepConfig uses for real frames
    #   target        = a G~12.2 star (after ~0.5 mag of extinction at 45 deg
    #                   altitude) sits right at that 3-sigma edge -- picked so
    #                   that, integrated over the real camera FOV above, the
    #                   local Gaia density around the demo's default pointing
    #                   yields ~3-4 stars per frame, matching what a real
    #                   Santiago session sees (see README.md, "Confianza en
    #                   cielos contaminados").
    #   star_flux_adu = thresh_sigma * sigma_total / 10**(-0.4*(12.2-12)) = 96/0.832 = 115
    star_flux_adu: float = 115.0
    # Safety cap on how many catalog stars get a patch drawn per frame; not a
    # visibility mechanism (see star_flux_adu above). Sized well above what
    # the real FOV actually turns up so it never binds in practice.
    max_render_stars: int = 240

    # --- Light-polluted sky (Santiago) ---
    # Frame-to-frame seeing wander, as a fraction of star_sigma_px. Turbulence
    # is what makes lucky-imaging selection worth doing at all.
    seeing_jitter_frac: float = 0.25
    # Sky brightness gradient across the frame, as a fraction of background.
    # Urban skyglow is never flat, and a flat background hides how much a
    # local-background estimator actually matters.
    sky_gradient_frac: float = 0.12
    # Photon (shot) noise on the sky background. Under heavy skyglow this
    # dominates read noise and, together with star_flux_adu above, is what
    # buries faint stars -- there is no separate magnitude cutoff.
    shot_noise_enabled: bool = True
    # Atmospheric extinction in magnitudes per airmass. Stars low on the
    # horizon genuinely fade; without this the demo solves near the horizon as
    # easily as at zenith. This lowers a star's effective brightness before
    # the background/noise comparison above, so it is one more continuous
    # contributor to "fading into the sky", not a cutoff of its own.
    extinction_mag_per_airmass: float = 0.35
    # Faint cutoff: stars dimmer than this are simply not recorded from the
    # city. Raise it to simulate a darker site.
    limiting_magnitude: float = 13.5

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
    focuser: FocuserConfig = field(default_factory=FocuserConfig)
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
