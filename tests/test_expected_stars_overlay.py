from unittest.mock import patch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils import iers
import numpy as np

from app_runner import AppRunner
from config import AppConfig
from goto import platesolving_center_to_altaz_deg
from platesolving import project_catalog_to_pixels


def test_catalog_projection_places_center_at_image_center() -> None:
    center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    offset = SkyCoord(
        lon=[0.0, 10.0] * u.arcsec,
        lat=[0.0, 0.0] * u.arcsec,
        frame=center.skyoffset_frame(),
    ).icrs

    pixels = project_catalog_to_pixels(
        offset,
        center_icrs=center,
        scale_arcsec_per_px=1.0,
        theta_deg=0.0,
        image_width=300,
        image_height=200,
    )

    np.testing.assert_allclose(pixels[0], [150.0, 100.0], atol=1e-8)
    np.testing.assert_allclose(pixels[1], [160.0, 100.0], atol=1e-6)


def test_expected_stars_overlay_uses_fitted_model_center() -> None:
    cfg = AppConfig()
    cfg.platesolving.pixel_size_m = 4.0e-6
    cfg.platesolving.focal_m = 0.8
    runner = AppRunner(cfg)
    model = runner._goto.model
    model.synced = True
    model.ref_steps = model.steps_est.copy()
    model.ref_az_alt_deg = np.array([180.0, 45.0], dtype=np.float64)
    model.model_fit_samples = 4
    model.model_roll_samples = 1
    model.model_roll_deg = 0.0
    runner._expected_stars_overlay_enabled = True
    obstime = Time("2026-06-07T02:00:00", scale="utc")
    center = runner._expected_stars_model_center(obstime=obstime)
    assert center is not None
    catalog = Table(
        {
            "source_id": np.array([-1], dtype=np.int64),
            "ra": [float(center.ra.deg)],
            "dec": [float(center.dec.deg)],
            "phot_g_mean_mag": [5.0],
        }
    )

    raw = np.zeros((200, 300), dtype=np.uint16)
    preview = np.zeros((200, 300), dtype=np.uint8)
    try:
        with patch("app_runner.gaia_healpix_cone_with_mag", return_value=catalog):
            out = runner._apply_expected_stars_overlay(
                raw,
                preview,
                obstime=obstime,
            )
        state = runner.get_state()
    finally:
        runner.stop()

    assert out.shape == (200, 300, 3)
    assert int(out[100, 150, 0]) > 200
    assert int(out[100, 150, 1]) < 80
    assert int(out[100, 150, 2]) > 200
    assert state.goto.expected_stars_overlay_count == 1
    assert state.goto.expected_stars_overlay_source == "Gaia + Hipparcos/Tycho-2"


def test_expected_stars_model_center_preserves_true_altaz_with_refraction_enabled() -> None:
    runner = AppRunner(AppConfig())
    model = runner._goto.model
    model.synced = True
    model.ref_steps = model.steps_est.copy()
    model.ref_az_alt_deg = np.array([210.0, 15.0], dtype=np.float64)
    model.model_fit_samples = 4
    obstime = Time("2026-06-07T02:00:00", scale="utc")

    try:
        assert runner._platesolving_observer.refraction_enable
        center = runner._expected_stars_model_center(obstime=obstime)
        assert center is not None
        with (
            iers.conf.set_temp("auto_download", False),
            iers.conf.set_temp("auto_max_age", None),
        ):
            recovered = platesolving_center_to_altaz_deg(
                float(center.ra.deg),
                float(center.dec.deg),
                observer=runner._platesolving_observer,
                obstime=obstime,
            )
    finally:
        runner.stop()

    np.testing.assert_allclose(recovered, [210.0, 15.0], atol=1e-8)
