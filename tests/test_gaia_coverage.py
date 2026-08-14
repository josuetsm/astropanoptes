from pathlib import Path
from unittest.mock import patch

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np

from app_runner import AppRunner
from config import AppConfig
from gaia_cache import (
    _bright_tile_cache_key,
    _cache_key,
    _normalize_hipparcos,
    _normalize_tycho2,
    _path_for_in,
    gaia_healpix_cone_with_mag,
    gaia_healpix_coverage,
    merge_bright_catalog,
)


def test_gaia_healpix_coverage_reports_cached_and_required_tiles(tmp_path: Path) -> None:
    cfg = AppConfig().platesolving
    cfg.cache_dir = str(tmp_path)
    cfg.nside = 1

    empty = gaia_healpix_coverage(cfg=cfg)
    assert int(empty["total_tiles"]) == 12
    assert int(empty["cached_tile_count"]) == 0

    pix = 4
    hexkey = _cache_key(kind="healpix_tile", payload={
        "table": cfg.table_name,
        "nside": cfg.nside,
        "order": cfg.order,
        "pix": pix,
        "gmax": cfg.gmax,
        "columns": list(cfg.columns),
    })
    tile_path = _path_for_in(tmp_path.resolve(), hexkey, cfg.prefer_parquet)
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_path.write_bytes(b"cached-gaia-tile")
    bright_key = _bright_tile_cache_key(
        nside=cfg.nside,
        order=cfg.order,
        pix=pix,
        vmax=cfg.gmax,
    )
    bright_path = _path_for_in(tmp_path.resolve(), bright_key, cfg.prefer_parquet)
    bright_path.parent.mkdir(parents=True, exist_ok=True)
    bright_path.write_bytes(b"cached-bright-tile")

    center = SkyCoord(
        ra=float(empty["tile_ra_deg"][pix]) * u.deg,
        dec=float(empty["tile_dec_deg"][pix]) * u.deg,
        frame="icrs",
    )
    coverage = gaia_healpix_coverage(cfg=cfg, center_icrs=center, radius_deg=0.01)

    assert int(coverage["cached_tile_count"]) == 1
    assert coverage["cached_tiles"] == [pix]
    assert coverage["field_required_tiles"] == [pix]
    assert coverage["field_cached_tiles"] == [pix]
    assert coverage["field_missing_tiles"] == []
    assert bool(coverage["field_available"])
    assert int(coverage["gaia_cached_tile_count"]) == 1
    assert int(coverage["bright_cached_tile_count"]) == 1
    assert int(coverage["cached_bytes"]) == (
        len(b"cached-gaia-tile") + len(b"cached-bright-tile")
    )


def test_bright_catalog_replaces_saturated_gaia_match() -> None:
    hip_raw = Table({
        "hip": [30438],
        "ra": [95.98787789671694],
        "de": [-52.69571786807983],
        "pmra": [19.93],
        "pmde": [23.24],
        "vmag": [-0.62],
    })
    hip = _normalize_hipparcos(hip_raw, vmax=15.0)
    gaia = Table({
        "source_id": np.array([5500822146529182592], dtype=np.int64),
        "ra": np.asarray(hip["ra"], dtype=np.float64),
        "dec": np.asarray(hip["dec"], dtype=np.float64),
        "phot_g_mean_mag": [14.32],
    })

    merged = merge_bright_catalog(gaia, hip)

    assert len(merged) == 1
    assert int(merged["source_id"][0]) < 0
    assert float(merged["phot_g_mean_mag"][0]) == -0.62


def test_tycho2_uses_documented_bt_vt_to_v_transform() -> None:
    raw = Table({
        "id_tycho": np.array([123456789], dtype=np.int64),
        "ra_mdeg": [120.0],
        "de_mdeg": [-30.0],
        "pm_ra": [0.0],
        "pm_de": [0.0],
        "bt_mag": [8.0],
        "vt_mag": [7.0],
    })

    normalized = _normalize_tycho2(raw, vmax=15.0)

    assert len(normalized) == 1
    assert int(normalized["source_id"][0]) < 0
    assert abs(float(normalized["phot_g_mean_mag"][0]) - 6.91) < 1e-12


def test_healpix_catalog_downloads_and_merges_bright_tile(tmp_path: Path) -> None:
    cfg = AppConfig().platesolving
    cfg.cache_dir = str(tmp_path)
    cfg.nside = 16
    cfg.bright_catalog_margin_deg = 0.0
    center = SkyCoord(ra=95.987877 * u.deg, dec=-52.695661 * u.deg, frame="icrs")
    gaia = Table({
        "source_id": np.array([5500822146529182592], dtype=np.int64),
        "ra": [float(center.ra.deg)],
        "dec": [float(center.dec.deg)],
        "phot_g_mean_mag": [14.32],
    })
    bright = Table({
        "source_id": np.array([-1000000030438], dtype=np.int64),
        "ra": [float(center.ra.deg)],
        "dec": [float(center.dec.deg)],
        "phot_g_mean_mag": [-0.62],
    })

    with (
        patch("gaia_cache._query_healpix_tile_async", return_value=gaia),
        patch("gaia_cache._query_bright_catalog_tile_async", return_value=bright),
    ):
        result = gaia_healpix_cone_with_mag(
            center_icrs=center,
            radius_deg=0.01,
            cfg=cfg,
            verbose=False,
        )

    assert len(result) == 1
    assert int(result["source_id"][0]) < 0
    coverage = gaia_healpix_coverage(cfg=cfg, center_icrs=center, radius_deg=0.01)
    assert bool(coverage["field_available"])


def test_runner_projects_gaia_coverage_to_current_altaz() -> None:
    cfg = AppConfig()
    runner = AppRunner(cfg)
    center = SkyCoord(ra=120.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")
    cached = {
        "tile_ra_deg": np.array([0.0, 90.0, 180.0]),
        "tile_dec_deg": np.array([-30.0, 0.0, 30.0]),
    }

    try:
        with (
            patch("app_runner.gaia_healpix_coverage", return_value=cached),
            patch.object(runner, "_current_field_center_icrs", return_value=(center, "test")),
            patch.object(runner, "_current_field_download_radius_deg", return_value=1.5),
        ):
            coverage = runner.get_gaia_coverage()
    finally:
        runner.stop()

    tile_az = np.asarray(coverage["tile_az_deg"], dtype=np.float64)
    tile_alt = np.asarray(coverage["tile_alt_deg"], dtype=np.float64)
    assert tile_az.shape == (3,)
    assert tile_alt.shape == (3,)
    assert np.all(np.isfinite(tile_az))
    assert np.all(np.isfinite(tile_alt))
    assert np.all((tile_az >= 0.0) & (tile_az < 360.0))
    assert np.all((tile_alt >= -90.0) & (tile_alt <= 90.0))
    assert 0.0 <= float(coverage["center_az_deg"]) < 360.0
    assert -90.0 <= float(coverage["center_alt_deg"]) <= 90.0
    assert coverage["field_source"] == "test"
    assert "projection_time_utc" in coverage
