from unittest.mock import patch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import pandas as pd

from config import AppConfig
from platesolving import ObserverConfig, project_catalog_to_pixels, solve_plate


def test_solve_plate_returns_optical_axis_not_triplet_centroid() -> None:
    cfg = AppConfig().platesolving
    cfg.pixel_size_m = 1.0e-6
    cfg.focal_m = 0.206265
    cfg.search_radius_deg = 1.0
    cfg.N_det = 3
    cfg.N_seed = 3
    cfg.min_inliers = 3
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

