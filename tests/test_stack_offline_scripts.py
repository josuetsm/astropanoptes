from __future__ import annotations

import numpy as np

from scripts.combine_raw_stacks import (
    _largest_valid_rectangle,
    _nearest_accepted_position,
)
from scripts.stack_raw_recordings import _combine_common_field, _common_bounds


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
