from __future__ import annotations

import numpy as np

from scripts.combine_raw_stacks import (
    _largest_valid_rectangle,
    _nearest_accepted_position,
)


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
