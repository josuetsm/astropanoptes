from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.neighbors import KDTree

import pandas as pd
from unittest.mock import patch

from astropy.time import Time

from config import AppConfig
from platesolving import (
    ObserverConfig,
    _limit_catalog_depth,
    annulus_candidates,
    annulus_candidates_batch,
    project_catalog_to_pixels,
    solve_plate,
    sorted_sides_arcsec_from_coords,
    sorted_sides_arcsec_from_vectors,
)


def _sky(n=400, seed=3):
    """Random unit vectors plus the matching SkyCoord, as the solver builds them."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    V = np.column_stack(
        [
            np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra)),
            np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra)),
            np.sin(np.deg2rad(dec)),
        ]
    )
    return c, np.ascontiguousarray(V)


def test_vector_sides_match_astropy_separations() -> None:
    """The fast path must reproduce astropy's geometry, not approximate it."""
    coords, V = _sky()
    rng = np.random.default_rng(11)
    for _ in range(60):
        i, j, k = rng.choice(len(coords), size=3, replace=False)
        slow = sorted_sides_arcsec_from_coords(coords, int(i), int(j), int(k))
        fast = sorted_sides_arcsec_from_vectors(V, int(i), int(j), int(k))
        # arcsec-level agreement over separations of many degrees
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-3)


def test_vector_sides_handle_coincident_points() -> None:
    """A zero-length side must not produce NaN through arccos rounding."""
    _, V = _sky(n=10)
    V[1] = V[0]
    out = sorted_sides_arcsec_from_vectors(V, 0, 1, 2)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-6)


def test_batched_annuli_match_per_index_calls() -> None:
    """Batching is only safe if it returns exactly the per-call result."""
    _, V = _sky(n=300, seed=5)
    tree = KDTree(V, leaf_size=40, metric="euclidean")
    centres = np.array([0, 7, 25, 100, 299], dtype=np.int64)
    theta, tol = 1800.0, 30.0

    batch = annulus_candidates_batch(tree, V, centres, theta, tol)
    assert len(batch) == len(centres)
    for pos, idx in enumerate(centres):
        one = annulus_candidates(tree, V, int(idx), theta, tol)
        np.testing.assert_array_equal(np.sort(batch[pos]), np.sort(one))


def test_batched_annuli_preserve_empty_results() -> None:
    _, V = _sky(n=120, seed=9)
    tree = KDTree(V, leaf_size=40, metric="euclidean")
    centres = np.array([0, 1, 2], dtype=np.int64)
    # an absurdly narrow annulus should select nothing for most centres
    batch = annulus_candidates_batch(tree, V, centres, 1.0, 0.001)
    for pos, idx in enumerate(centres):
        one = annulus_candidates(tree, V, int(idx), 1.0, 0.001)
        assert batch[pos].size == one.size


def _catalog(n, *, mag_lo=6.0, mag_hi=16.0):
    return pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "ra": np.linspace(10.0, 11.0, n),
            "dec": np.linspace(-5.0, -4.0, n),
            "phot_g_mean_mag": np.linspace(mag_lo, mag_hi, n),
        }
    )


def test_catalog_depth_cap_keeps_the_brightest_at_the_image_density() -> None:
    """A cone far deeper than the image only feeds the search noise and cost.

    The solver compares the N_det brightest detections, so the catalogue is cut
    to that same density over the cone. Measured on a real field, this took a
    94 s solve to a few seconds while landing on the same centre and rotation.
    """
    cfg = AppConfig().platesolving
    cfg.N_det = 30
    cfg.catalog_density_factor = 1.5
    df = _catalog(40_000)

    out = _limit_catalog_depth(
        df, cfg=cfg, arcsec_per_px=0.6646, width=2522, height=2704, radius_deg=3.0
    )

    field_deg2 = (2522 * 0.6646 / 3600.0) * (2704 * 0.6646 / 3600.0)
    expected = int(np.ceil(1.5 * 30 * (np.pi * 9.0) / field_deg2))
    assert len(out) == expected
    assert len(out) < len(df)
    # se queda con las mas brillantes, no con un trozo cualquiera
    assert float(out["phot_g_mean_mag"].max()) <= float(
        df["phot_g_mean_mag"].nsmallest(expected).max()
    )


def test_catalog_depth_cap_leaves_a_catalog_that_already_fits() -> None:
    """Sin margen que recortar no debe tocar nada (ni con factor 0)."""
    cfg = AppConfig().platesolving
    cfg.N_det = 30
    small = _catalog(300)

    # cono del tamano del campo: no hay exceso que recortar
    kept = _limit_catalog_depth(
        small, cfg=cfg, arcsec_per_px=0.6646, width=2522, height=2704, radius_deg=0.2
    )
    assert kept is small

    # factor 0 desactiva el tope por completo
    cfg.catalog_density_factor = 0.0
    assert _limit_catalog_depth(
        _catalog(40_000), cfg=cfg, arcsec_per_px=0.6646,
        width=2522, height=2704, radius_deg=3.0,
    ).shape[0] == 40_000


def test_gaia_size_guard_counts_the_detections_the_solver_uses() -> None:
    """El guard media contra max_det (250) aunque el solver solo use N_det (30).

    Eso exigia 750 estrellas de catalogo para una busqueda que nunca mira mas
    alla de las 30 detecciones mas brillantes, y rechazaba con GAIA_TOO_SMALL
    campos perfectamente resolubles.
    """
    center = SkyCoord(ra=120.0 * u.deg, dec=-25.0 * u.deg, frame="icrs")
    rng = np.random.default_rng(11)
    n_cat, width, height, scale = 120, 600, 600, 1.0
    offsets = rng.uniform(-240.0, 240.0, size=(n_cat, 2))
    coords = SkyCoord(
        lon=offsets[:, 0] * u.arcsec, lat=offsets[:, 1] * u.arcsec,
        frame=center.skyoffset_frame(),
    ).icrs
    catalog = pd.DataFrame(
        {
            "source_id": np.arange(n_cat, dtype=np.int64),
            "ra": coords.ra.deg,
            "dec": coords.dec.deg,
            "phot_g_mean_mag": np.linspace(8.0, 13.0, n_cat),
        }
    )
    # 60 detecciones: mas que N_det=30, y 3*60=180 > 120 disparaba el guard
    det = project_catalog_to_pixels(
        coords[:60], center_icrs=center, scale_arcsec_per_px=scale,
        theta_deg=0.0, image_width=width, image_height=height,
    )
    flux = np.linspace(9000.0, 1000.0, len(det))

    cfg = AppConfig().platesolving
    cfg.search_radius_deg = 0.2          # cono ~ campo: sin recorte de catalogo
    cfg.temporal_detection_enabled = False
    cfg.focal_m = 206265.0 * cfg.pixel_size_m / scale
    cfg.max_center_offset_factor = 0.0

    with (
        patch(
            "platesolving.detect_sep_objects",
            return_value=(np.zeros((height, width), dtype=np.float32), det, flux),
        ),
        patch("platesolving._gaia_load_df", return_value=catalog),
    ):
        result = solve_plate(
            np.zeros((height, width), dtype=np.uint16),
            target=center,
            cfg=cfg,
            observer=ObserverConfig(),
            obstime=Time("2026-09-16T01:28:45", scale="utc"),
        )

    assert result.status != "GAIA_TOO_SMALL"
    assert result.success
