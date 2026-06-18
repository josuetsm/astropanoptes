from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from ap_types import Axis, PlatesolvingStatus
from app_runner import AppRunner
from config import AppConfig
from gaia_cache import GaiaCacheMissError, add_bright_star_supplement
from goto import MountKinematics, platesolving_center_to_altaz_deg
from platesolving import ObserverConfig
from simulation import SimulatedCameraStream, SimulatedMount, SimulationState


def _catalog_near(center: SkyCoord, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    radius_arcsec = 160.0
    r = np.sqrt(rng.uniform(0.0, 1.0, n)) * radius_arcsec
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    off = center.skyoffset_frame()
    coords = SkyCoord(
        lon=(r * np.cos(phi)) * u.arcsec,
        lat=(r * np.sin(phi)) * u.arcsec,
        frame=off,
    ).icrs
    return pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(9.5, 14.5, n),
        }
    )


def _patched_catalog(self, center_icrs, radius_deg):
    return _catalog_near(center_icrs), False


def _wait_for_gaia_download(runner: AppRunner, timeout_s: float = 2.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        with runner._gaia_download_lock:
            thr = runner._gaia_download_thread
        if thr is None or not thr.is_alive():
            return
        time.sleep(0.02)
    raise AssertionError("Gaia download thread did not finish")


def test_simulation_initial_offsets_are_bounded() -> None:
    cfg = AppConfig()
    cfg.simulation.seed = 42
    cfg.simulation.initial_az_deg = 123.0
    cfg.simulation.initial_alt_deg = 44.0
    cfg.simulation.random_mount_tilt_deg = 1.0
    cfg.simulation.random_camera_roll_deg = 1.0

    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    snap = state.snapshot()

    az_delta = ((float(snap["az_deg"]) - 123.0 + 180.0) % 360.0) - 180.0
    alt_delta = float(snap["alt_deg"]) - 44.0
    assert abs(az_delta) <= 1.0
    assert abs(alt_delta) <= 1.0
    assert abs(float(snap["camera_roll_error_deg"])) <= 1.0


def test_simulation_center_icrs_round_trips_true_altaz_without_refraction_bias() -> None:
    cfg = AppConfig()
    cfg.simulation.seed = 43
    cfg.simulation.initial_az_deg = 210.0
    cfg.simulation.initial_alt_deg = 15.0
    cfg.simulation.random_mount_tilt_deg = 0.0
    observer = ObserverConfig(refraction_enable=True)
    obstime = Time("2026-06-08T01:00:00", scale="utc")
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)

    center = state.center_icrs(observer=observer, obstime=obstime)
    round_trip = platesolving_center_to_altaz_deg(
        float(center.ra.deg),
        float(center.dec.deg),
        observer=observer,
        obstime=obstime,
    )

    np.testing.assert_allclose(round_trip, [210.0, 15.0], atol=1e-8)


def test_simulated_mount_move_changes_physical_truth() -> None:
    cfg = AppConfig()
    cfg.simulation.seed = 7
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    mount = SimulatedMount(state, out_log=None)
    mount.connect(port="DEMO", baud=115200)
    before = state.snapshot_altaz()

    mount.move_steps(Axis.AZ, direction=1, steps=5000, delay_us=1, blocking=False)
    after = state.snapshot_altaz()

    assert after[0] != before[0]
    assert mount.is_connected()


def test_simulated_camera_stream_produces_raw16_gaia_like_frame() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 100
    cfg.simulation.frame_w = 240
    cfg.simulation.frame_h = 180
    cfg.simulation.fps = 15.0
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)

    with patch.object(SimulatedCameraStream, "_get_catalog", _patched_catalog):
        try:
            stream.start()
            deadline = time.time() + 2.0
            frame = None
            while time.time() < deadline:
                frame = stream.latest()
                if frame is not None:
                    break
                time.sleep(0.02)
        finally:
            stream.stop()

    assert frame is not None
    assert frame.raw.dtype == np.uint16
    assert frame.raw.shape == (180, 240)
    assert bool(frame.meta["simulated"])
    assert int(frame.meta["sim_stars_drawn"]) > 0
    assert int(frame.raw.max()) > int(frame.raw.min())


def test_demo_requires_gaia_cache_by_default() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 101
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)
    center = SkyCoord(ra=10.0 * u.deg, dec=-20.0 * u.deg, frame="icrs")

    with patch(
        "simulation.gaia_healpix_cone_with_mag",
        side_effect=GaiaCacheMissError([Path("/tmp/missing-gaia-tile.parquet")]),
    ):
        catalog, source = stream._get_catalog(center, radius_deg=1.2)

    stats = stream.stats()
    assert str(source) == "gaia_missing"
    assert catalog.empty
    assert stats["catalog_source"] == "gaia_missing"
    assert "Gaia cache missing" in str(stats["last_error"])
    assert not cfg.simulation.allow_synthetic_fallback


def test_demo_synthetic_catalog_is_opt_in_when_gaia_cache_is_missing() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 102
    cfg.simulation.allow_synthetic_fallback = True
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)
    center = SkyCoord(ra=10.0 * u.deg, dec=-20.0 * u.deg, frame="icrs")

    with patch(
        "simulation.gaia_healpix_cone_with_mag",
        side_effect=GaiaCacheMissError([Path("/tmp/missing-gaia-tile.parquet")]),
    ):
        catalog, source = stream._get_catalog(center, radius_deg=1.2)

    stats = stream.stats()
    assert str(source) == "synthetic"
    assert len(catalog) >= 80
    assert stats["catalog_source"] == "synthetic"
    assert "synthetic fallback" in str(stats["last_error"])


def test_demo_star_magnitude_changes_apparent_size() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 103
    cfg.simulation.frame_w = 220
    cfg.simulation.frame_h = 140
    cfg.simulation.star_sigma_px = 1.25
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)
    center = SkyCoord(ra=10.0 * u.deg, dec=-20.0 * u.deg, frame="icrs")
    scale = stream._arcsec_per_px()
    off = center.skyoffset_frame()
    bright = SkyCoord(lon=(-35.0 * scale) * u.arcsec, lat=0.0 * u.arcsec, frame=off).icrs
    faint = SkyCoord(lon=(35.0 * scale) * u.arcsec, lat=0.0 * u.arcsec, frame=off).icrs
    catalog = pd.DataFrame(
        {
            "source_id": [1, 2],
            "ra": [bright.ra.deg, faint.ra.deg],
            "dec": [bright.dec.deg, faint.dec.deg],
            "phot_g_mean_mag": [8.0, 14.5],
        }
    )
    img = np.zeros((140, 220), dtype=np.float32)

    with patch("simulation.expected_field_rotation_deg", return_value=0.0):
        drawn = stream._draw_catalog_stars(img, catalog, center, Time.now())

    def _weighted_radius(cx: int, cy: int, half: int = 24) -> float:
        patch_img = img[cy - half : cy + half + 1, cx - half : cx + half + 1].astype(np.float64)
        yy, xx = np.mgrid[cy - half : cy + half + 1, cx - half : cx + half + 1]
        weight = np.maximum(patch_img, 0.0)
        return float(np.sqrt(np.sum(weight * ((xx - cx) ** 2 + (yy - cy) ** 2)) / np.sum(weight)))

    assert drawn == 2
    assert _weighted_radius(75, 70) > _weighted_radius(145, 70) * 1.8
    assert float(img[70, 75]) > float(img[70, 145])


def test_canopus_is_supplemented_and_rendered_at_field_center() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 104
    cfg.simulation.random_mount_tilt_deg = 0.0
    cfg.simulation.random_camera_roll_deg = 0.0
    state = SimulationState(cfg=cfg.simulation, kin=MountKinematics(), out_log=None)
    stream = SimulatedCameraStream(state=state, cfg=cfg, observer=ObserverConfig(), out_log=None)
    canopus = SkyCoord(ra=95.987877 * u.deg, dec=-52.695661 * u.deg, frame="icrs")
    faint_neighbor = SkyCoord(
        lon=76.0 * u.arcsec,
        lat=0.0 * u.arcsec,
        frame=canopus.skyoffset_frame(),
    ).icrs
    gaia_only = Table(
        {
            "source_id": np.array([5500822146529182592], dtype=np.int64),
            "ra": [float(faint_neighbor.ra.deg)],
            "dec": [float(faint_neighbor.dec.deg)],
            "phot_g_mean_mag": [14.32],
        }
    )

    supplemented = add_bright_star_supplement(
        gaia_only,
        center_icrs=canopus,
        radius_deg=1.2,
        gmax=15.0,
    )
    catalog = supplemented.to_pandas()
    coords = SkyCoord(
        ra=np.asarray(catalog["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(catalog["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    sep_arcsec = np.asarray(coords.separation(canopus).arcsec, dtype=np.float64)
    canopus_row = catalog.iloc[int(np.argmin(sep_arcsec))]

    img = np.zeros((cfg.simulation.frame_h, cfg.simulation.frame_w), dtype=np.float32)
    with patch("simulation.expected_field_rotation_deg", return_value=0.0):
        drawn = stream._draw_catalog_stars(img, catalog, canopus, Time.now())

    cy = cfg.simulation.frame_h // 2
    cx = cfg.simulation.frame_w // 2
    assert len(catalog) == 2
    assert float(np.min(sep_arcsec)) < 0.01
    assert float(canopus_row["phot_g_mean_mag"]) == -0.74
    assert drawn == 2
    assert float(img[cy, cx]) > 60_000.0


def test_app_runner_demo_connects_without_camera_or_mount_hardware() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 200
    cfg.simulation.frame_w = 200
    cfg.simulation.frame_h = 150
    cfg.simulation.fps = 12.0
    runner = AppRunner(cfg)

    with patch.object(SimulatedCameraStream, "_get_catalog", _patched_catalog):
        try:
            runner._connect_mount("DEMO", 115200)
            runner._connect_camera(0)
            deadline = time.time() + 2.0
            frame = None
            while time.time() < deadline:
                frame = runner._get_latest_frame()
                if frame is not None:
                    break
                time.sleep(0.02)

            st = runner.get_state()
            assert st.mount.connected
            assert st.camera.connected
            assert frame is not None
            assert frame.raw.shape == (150, 200)
            assert bool(frame.meta["simulated"])
        finally:
            runner.stop()


def test_runner_downloads_current_demo_field_to_gaia_cache() -> None:
    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 300
    runner = AppRunner(cfg)
    gaia_rows = pd.DataFrame(
        {
            "source_id": [1],
            "ra": [10.0],
            "dec": [-20.0],
            "phot_g_mean_mag": [12.0],
        }
    )

    try:
        with patch("app_runner.gaia_healpix_cone_with_mag", return_value=gaia_rows) as mocked:
            runner._request_gaia_current_field_download()
            _wait_for_gaia_download(runner)

        assert mocked.call_count == 1
        kwargs = mocked.call_args.kwargs
        assert bool(kwargs["cfg"].download_missing_tiles)
        assert float(kwargs["radius_deg"]) >= float(cfg.simulation.catalog_radius_deg)
        state = runner.get_state()
        assert state.platesolving.status == PlatesolvingStatus.OK
        assert state.platesolving.debug_info is not None
        assert state.platesolving.debug_info["status"] == "GAIA_DOWNLOAD_OK"
        assert int(state.platesolving.debug_info["rows"]) == 1
    finally:
        runner.stop()


def test_runner_download_current_field_requires_known_pointing() -> None:
    runner = AppRunner(AppConfig())
    try:
        runner._request_gaia_current_field_download()
        state = runner.get_state()
        assert state.platesolving.status == PlatesolvingStatus.FAIL
        assert state.platesolving.reason == "NO_CURRENT_FIELD"
    finally:
        runner.stop()
