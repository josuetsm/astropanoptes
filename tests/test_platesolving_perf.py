from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.neighbors import KDTree

from platesolving import (
    annulus_candidates,
    annulus_candidates_batch,
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
