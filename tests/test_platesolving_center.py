from unittest.mock import patch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import pandas as pd

from config import AppConfig
from platesolving import (
    ObserverConfig,
    PlatesolvingResult,
    expected_field_rotation_deg,
    detect_persistent_sep_objects,
    platesolving_solutions_consistent,
    project_catalog_to_pixels,
    solve_plate,
    verify_plate_from_prior,
)


def test_temporal_aligned_median_recovers_fragmented_star_tracks() -> None:
    frame_count = 12
    height, width = 180, 240
    frames = [np.full((height, width), 1000 + i, dtype=np.uint16) for i in range(frame_count)]
    base = np.array(
        [[40.0, 35.0], [85.0, 55.0], [130.0, 90.0], [175.0, 65.0], [205.0, 130.0]],
        dtype=np.float64,
    )
    drift = np.array([1.25, -2.0], dtype=np.float64)
    call_index = 0

    def fake_detect(_frame, **_kwargs):
        nonlocal call_index
        if call_index < frame_count:
            idx = call_index
            xy = base + drift * idx
            # Each source disappears often enough that no track reaches the
            # strict 10/12 threshold, while remaining a real repeated source.
            keep = np.array(
                [((idx + source_idx) % 3) != 0 for source_idx in range(len(base))]
            )
            xy = xy[keep]
        else:
            xy = base + drift * (frame_count - 1)
        call_index += 1
        flux = np.linspace(5000.0, 1000.0, len(xy), dtype=np.float64)
        return np.zeros((height, width), dtype=np.float32), xy, flux

    with patch("platesolving.detect_sep_objects", side_effect=fake_detect):
        detections = detect_persistent_sep_objects(
            frames,
            sep_bw=32,
            sep_bh=32,
            sep_thresh_sigma=3.0,
            sep_minarea=5,
            max_sources=30,
            min_hits=10,
            match_radius_px=4.0,
            max_drift_per_frame_px=8.0,
            min_drift_response=0.0,
        )

    assert detections.xy.shape[0] == len(base)
    assert int(np.max(detections.hits)) < 10
    assert detections.reference_frame.dtype == np.uint16


def _prior_result(center: SkyCoord, *, theta_deg: float, obstime: Time) -> PlatesolvingResult:
    theta = np.deg2rad(theta_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    return PlatesolvingResult(
        success=True,
        status="OK",
        theta_deg=float(theta_deg),
        dx_px=0.0,
        dy_px=0.0,
        response=10.0,
        scale_arcsec_per_px=1.0,
        R_2x2=((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1]))),
        t_arcsec=(0.0, 0.0),
        n_inliers=8,
        rms_arcsec=0.2,
        rms_px=0.2,
        center_ra_deg=float(center.ra.deg),
        center_dec_deg=float(center.dec.deg),
        overlay=[],
        guides=[],
        metrics={},
        obstime_unix=float(obstime.unix),
    )


def test_fast_prior_verification_refits_six_catalog_stars_without_triplets() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 12
    cfg.min_inliers = 6
    cfg.rotation_prior_enable = False
    width, height = 500, 360
    obstime = Time("2026-06-08T01:00:00", scale="utc")
    center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    offsets = np.array(
        [
            [-160.0, -90.0], [-60.0, 105.0], [70.0, -115.0],
            [155.0, 85.0], [-205.0, 40.0], [210.0, -35.0],
            [-25.0, -145.0], [35.0, 150.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 13.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords,
        center_icrs=center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )
    prior = _prior_result(center, theta_deg=17.0, obstime=obstime)

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.linspace(8000.0, 1000.0, len(detections)),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = verify_plate_from_prior(
            np.zeros((height, width), dtype=np.uint16),
            prior=prior,
            target=center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=obstime,
        )

    assert result.success
    assert result.status == "OK_FAST_PRIOR"
    assert result.n_inliers == len(detections)
    assert float(result.metrics["fast_prior"]) == 1.0
    solved = SkyCoord(result.center_ra_deg * u.deg, result.center_dec_deg * u.deg)
    assert float(solved.separation(center).arcsec) < 0.2


def test_fast_prior_verification_rejects_a_far_false_first_solution() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 12
    cfg.min_inliers = 6
    cfg.rotation_prior_enable = False
    width, height = 500, 360
    obstime = Time("2026-06-08T01:00:00", scale="utc")
    true_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    false_center = SkyCoord(
        lon=0.5 * u.deg,
        lat=0.0 * u.deg,
        frame=true_center.skyoffset_frame(),
    ).icrs
    offsets = np.array(
        [[-160.0, -90.0], [-60.0, 105.0], [70.0, -115.0], [155.0, 85.0], [-205.0, 40.0], [210.0, -35.0]],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 13.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords,
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )
    prior = _prior_result(false_center, theta_deg=17.0, obstime=obstime)
    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.linspace(6000.0, 1000.0, len(detections)),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = verify_plate_from_prior(
            np.zeros((height, width), dtype=np.uint16),
            prior=prior,
            target=true_center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=obstime,
        )

    assert not result.success
    assert result.status == "FAST_PRIOR_VALIDATION_FAILED"


def test_solution_consensus_handles_continuous_tracking_in_icrs() -> None:
    observer = ObserverConfig()
    t0 = Time("2026-06-08T01:00:00", scale="utc")
    t1 = Time("2026-06-08T01:00:30", scale="utc")
    center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    axis0 = expected_field_rotation_deg(
        float(center.ra.deg), float(center.dec.deg), observer=observer, obstime=t0
    )
    axis1 = expected_field_rotation_deg(
        float(center.ra.deg), float(center.dec.deg), observer=observer, obstime=t1
    )
    assert axis0 is not None and axis1 is not None
    roll_deg = 11.0
    reference = _prior_result(center, theta_deg=float(axis0) - roll_deg, obstime=t0)
    candidate = _prior_result(center, theta_deg=float(axis1) - roll_deg, obstime=t1)
    candidate = PlatesolvingResult(
        **{
            **candidate.__dict__,
            "metrics": {"fast_prior": 1.0, "fast_prior_idle": 0.0},
        }
    )

    report = platesolving_solutions_consistent(
        reference,
        candidate,
        observer=observer,
        pointing_tol_arcsec=30.0,
        scale_tol_frac=0.02,
        roll_tol_deg=3.0,
    )

    assert report["ok"]
    assert report["tracking_hypothesis"]
    assert float(report["pointing_arcsec"]) < 0.1


def test_solve_plate_returns_optical_axis_not_triplet_centroid() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 3
    cfg.N_seed = 3
    cfg.min_inliers = 3
    cfg.min_validation_inliers = 0
    cfg.max_trials = 500
    cfg.max_i_scan = 100
    cfg.rotation_prior_enable = False

    width, height = 400, 300
    true_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    target = SkyCoord(
        lon=240.0 * u.arcsec,
        lat=-150.0 * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    offsets = np.array(
        [
            [-82.0, -51.0],
            [96.0, -18.0],
            [21.0, 103.0],
            [-310.0, 205.0],
            [285.0, 240.0],
            [-260.0, -270.0],
            [330.0, -160.0],
            [145.0, 315.0],
            [-350.0, 45.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 14.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords[:3],
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.array([3000.0, 2000.0, 1000.0], dtype=np.float64),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=target,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-06-08T01:00:00", scale="utc"),
        )

    solved_center = SkyCoord(
        ra=result.center_ra_deg * u.deg,
        dec=result.center_dec_deg * u.deg,
        frame="icrs",
    )
    assert result.success
    assert float(solved_center.separation(true_center).arcsec) < 0.2
    assert np.isfinite(result.obstime_unix)
    assert abs(float(result.metrics["match_tol_arcsec"]) - 3.5) < 1e-9


def test_solve_plate_does_not_use_clipped_edge_detection_as_seed() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 4
    cfg.N_seed = 3
    cfg.min_inliers = 3
    cfg.min_validation_inliers = 0
    cfg.max_trials = 500
    cfg.max_i_scan = 100
    cfg.rotation_prior_enable = False
    cfg.seed_edge_margin_px = 8.0

    width, height = 400, 300
    true_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    offsets = np.array(
        [
            [-82.0, -51.0],
            [96.0, -18.0],
            [21.0, 103.0],
            [-310.0, 205.0],
            [285.0, 240.0],
            [-260.0, -270.0],
            [330.0, -160.0],
            [145.0, 315.0],
            [-350.0, 45.0],
            [360.0, 260.0],
            [-370.0, -220.0],
            [250.0, -340.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 14.0, len(coords)),
        }
    )
    real_detections = project_catalog_to_pixels(
        coords[:3],
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )
    detections = np.vstack(
        [np.array([[2.0, 150.0]], dtype=np.float64), real_detections]
    )

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.array([1.0e7, 3000.0, 2000.0, 1000.0], dtype=np.float64),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=true_center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-06-08T01:00:00", scale="utc"),
        )

    assert result.success
    assert result.n_inliers == 3
    assert result.metrics["seed_edge_excluded"] == 1.0


def test_solve_plate_rejects_candidate_whose_scale_disagrees_with_known_optics() -> None:
    """
    The triangle-side tolerance used to accept a candidate triplet is an
    absolute arcsec value, so it is comparatively loose for short/tight seed
    triangles: a candidate can pass it while implying a plate scale far from
    what the instrument's focal length/pixel size actually deliver. This
    reproduces the working fixture above but tightens scale_tol_frac so a
    would-be "success" is required to instead be rejected as NO_SCALE_MATCH,
    proving the guard is wired into the real solve_plate() matching path.
    """
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 3
    cfg.N_seed = 3
    cfg.min_inliers = 3
    cfg.min_validation_inliers = 0
    cfg.max_trials = 500
    cfg.max_i_scan = 100
    cfg.rotation_prior_enable = False
    cfg.scale_tol_frac = 1e-10

    width, height = 400, 300
    true_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    target = SkyCoord(
        lon=240.0 * u.arcsec,
        lat=-150.0 * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    offsets = np.array(
        [
            [-82.0, -51.0],
            [96.0, -18.0],
            [21.0, 103.0],
            [-310.0, 205.0],
            [285.0, 240.0],
            [-260.0, -270.0],
            [330.0, -160.0],
            [145.0, 315.0],
            [-350.0, 45.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 14.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords[:3],
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.array([3000.0, 2000.0, 1000.0], dtype=np.float64),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=target,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-06-08T01:00:00", scale="utc"),
        )

    assert not result.success
    assert result.status == "NO_SCALE_MATCH"


def test_solve_plate_requires_matches_independent_from_seed_triplet() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 4
    cfg.N_seed = 3
    cfg.min_inliers = 4
    cfg.min_validation_inliers = 3
    cfg.max_trials = 500
    cfg.max_i_scan = 100
    cfg.rotation_prior_enable = False

    width, height = 400, 300
    true_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    offsets = np.array(
        [
            [-82.0, -51.0],
            [96.0, -18.0],
            [21.0, 103.0],
            [-140.0, 125.0],
            [-310.0, 205.0],
            [285.0, 240.0],
            [-260.0, -270.0],
            [330.0, -160.0],
            [145.0, 315.0],
            [-350.0, 45.0],
            [370.0, 80.0],
            [-90.0, -360.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 14.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords[:4],
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.linspace(4000.0, 1000.0, len(detections)),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=true_center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-06-08T01:00:00", scale="utc"),
        )

    assert not result.success
    assert result.status == "LOW_VALIDATION_INLIERS"
    assert int(result.metrics["validation_inliers"]) == 1


def test_solve_plate_rejects_center_outside_requested_search_cone() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 6
    cfg.N_seed = 3
    cfg.min_inliers = 6
    cfg.min_validation_inliers = 3
    cfg.max_trials = 500
    cfg.max_i_scan = 100
    cfg.rotation_prior_enable = False

    width, height = 400, 300
    target_center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    true_center = SkyCoord(
        lon=1.5 * u.deg,
        lat=0.0 * u.deg,
        frame=target_center.skyoffset_frame(),
    ).icrs
    offsets = np.array(
        [
            [-82.0, -51.0], [96.0, -18.0], [21.0, 103.0],
            [-140.0, 125.0], [160.0, 130.0], [-155.0, -120.0],
            [-310.0, 205.0], [285.0, 240.0], [-260.0, -270.0],
            [330.0, -160.0], [145.0, 315.0], [-350.0, 45.0],
            [370.0, 80.0], [-90.0, -360.0], [410.0, -220.0],
            [-430.0, -190.0], [250.0, -390.0], [-275.0, 380.0],
        ],
        dtype=np.float64,
    )
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec,
        lat=offsets[:, 1] * u.arcsec,
        frame=true_center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(len(coords), dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 14.0, len(coords)),
        }
    )
    detections = project_catalog_to_pixels(
        coords[:6],
        center_icrs=true_center,
        scale_arcsec_per_px=1.0,
        theta_deg=17.0,
        image_width=width,
        image_height=height,
    )

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(
                np.zeros((height, width), dtype=np.float32),
                detections,
                np.linspace(6000.0, 1000.0, len(detections)),
            ),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=target_center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-06-08T01:00:00", scale="utc"),
        )

    assert not result.success
    assert result.status == "CENTER_OUT_OF_RANGE"
    assert float(result.metrics["target_offset_deg"]) > 1.4
