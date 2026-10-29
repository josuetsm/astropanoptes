from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from scripts.combine_raw_stacks import (
    _largest_valid_rectangle,
    _nearest_accepted_position,
)
from scripts.stack_raw_recordings import _combine_common_field, _common_bounds
from scripts.solve_saved_stack import _to_time, load_sidecar, load_weight


def test_largest_valid_rectangle_removes_partial_coverage_corners() -> None:
    mask = np.ones((6, 8), dtype=bool)
    mask[:2, 6:] = False
    mask[4:, 6:] = False

    left, top, right, bottom = _largest_valid_rectangle(mask)

    assert (left, top, right, bottom) == (0, 0, 6, 6)
    assert np.all(mask[top:bottom, left:right])


def test_nearest_accepted_position_skips_rejected_midpoint() -> None:
    metadata = {
        "positions_native_px": [
            [0.0, 0.0],
            [1.0, -1.0],
            None,
            [3.0, -3.0],
            [4.0, -4.0],
        ]
    }

    frame_index, position = _nearest_accepted_position(metadata)

    assert frame_index == 1
    np.testing.assert_allclose(position, np.array([1.0, -1.0]))


def test_cfa_reconstruction_has_no_bayer_checkerboard() -> None:
    pattern_offsets = {
        "RGGB": ((0, 0, 1000), (0, 1, 2000), (1, 0, 2000), (1, 1, 3000)),
        "BGGR": ((0, 0, 3000), (0, 1, 2000), (1, 0, 2000), (1, 1, 1000)),
        "GRBG": ((0, 0, 2000), (0, 1, 1000), (1, 0, 3000), (1, 1, 2000)),
        "GBRG": ((0, 0, 2000), (0, 1, 3000), (1, 0, 1000), (1, 1, 2000)),
    }

    for pattern, samples in pattern_offsets.items():
        raw = np.zeros((1, 16, 20), dtype=np.uint16)
        for row_offset, column_offset, value in samples:
            raw[0, row_offset::2, column_offset::2] = value
        positions = [(0.0, 0.0)]
        bounds = _common_bounds(
            positions,
            native_h=raw.shape[1],
            native_w=raw.shape[2],
            scale=2,
        )

        rgb = _combine_common_field(
            raw,
            positions,
            bayer_pattern=pattern,
            scale=2,
            bounds=bounds,
        )

        assert rgb.shape == (24, 32, 3)
        expected = np.array([1000, 2000, 3000], dtype=np.uint16)
        np.testing.assert_array_equal(rgb, np.broadcast_to(expected, rgb.shape))


@pytest.fixture
def santiago_tz():
    """Fija la zona a Chile continental para la duracion del test."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Santiago"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_saved_stack_epoch_follows_daylight_saving(santiago_tz) -> None:
    """El nombre lleva hora local, y Chile cambia de huso en septiembre.

    Con el desfase fijo de -4 que habia, una captura de mediados de septiembre
    se fechaba una hora tarde. Una hora son 15 grados de angulo horario: el
    solver buscaba el campo a 15 grados de donde estaba y solo encontraba
    coincidencias falsas.
    """
    invierno = _to_time("20260831", "220736")   # UTC-4
    verano = _to_time("20260915", "222845")     # UTC-3 (DST)

    assert invierno.iso.startswith("2026-09-01 02:07:36")
    assert verano.iso.startswith("2026-09-16 01:28:45")


def test_sidecar_and_weight_are_read_from_the_saved_stack(tmp_path) -> None:
    """Lo que escribe "stacking save" es lo que el solve offline debe leer."""
    raw_path = tmp_path / "stack_20260915_222845_az258p87_altp39p41_raw.npy"
    np.save(raw_path, np.zeros((4, 4), dtype=np.uint16))
    wgt = np.full((4, 4), 2.0, dtype=np.float32)
    np.save(tmp_path / "stack_20260915_222845_az258p87_altp39p41_wgt.npy", wgt)
    meta = {"obstime_unix": 1789522125.0, "drizzle_scale": 2.0, "pointing_az_deg": 258.87}
    (tmp_path / "stack_20260915_222845_az258p87_altp39p41.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    assert load_sidecar(raw_path) == meta
    np.testing.assert_array_equal(load_weight(raw_path), wgt)


def test_missing_sidecar_is_not_an_error(tmp_path) -> None:
    """Los stacks anteriores al sidecar siguen resolviendose por el nombre."""
    raw_path = tmp_path / "stack_20260831_220736_az098p13_altp51p38_raw.npy"
    np.save(raw_path, np.zeros((4, 4), dtype=np.uint16))

    assert load_sidecar(raw_path) == {}
    assert load_weight(raw_path) is None
