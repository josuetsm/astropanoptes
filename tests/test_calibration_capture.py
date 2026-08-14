from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from calibration_capture import (
    CalibrationRequest,
    combine_frames,
    create_session_directory,
    frame_statistics,
    infer_raw_shift,
)


def test_infer_raw_shift_detects_left_aligned_12_bit_data() -> None:
    native = np.arange(256, dtype=np.uint16).reshape(16, 16)
    raw = native << 4

    shift, aligned_fraction = infer_raw_shift(raw, 12)

    assert shift == 4
    assert aligned_fraction == 1.0


def test_infer_raw_shift_keeps_right_aligned_data() -> None:
    raw = np.arange(1, 4097, dtype=np.uint16).reshape(64, 64) & np.uint16(4095)

    shift, aligned_fraction = infer_raw_shift(raw, 12)

    assert shift == 0
    assert aligned_fraction < 0.999


def test_frame_statistics_reports_native_adu_not_raw16_container_values() -> None:
    raw = np.array([[0, 16], [32, 65520]], dtype=np.uint16)

    stats = frame_statistics(raw, bit_depth=12, raw_shift=4)

    assert stats["native_mean_adu"] == pytest.approx((0 + 1 + 2 + 4095) / 4)
    assert stats["floor_fraction"] == 0.25
    assert stats["saturation_fraction"] == 0.25
    assert stats["raw_max"] == 65520


def test_frame_statistics_treats_mars_c_4094_adu_as_saturated() -> None:
    raw = np.full((4, 4), 4094 << 4, dtype=np.uint16)

    stats = frame_statistics(raw, bit_depth=12, raw_shift=4)

    assert stats["saturation_level_adu"] == 4094.0
    assert stats["saturation_fraction"] == 1.0


@pytest.mark.parametrize("method, expected", [("median", 30), ("mean", 40)])
def test_combine_frames_uses_requested_method(
    tmp_path: Path,
    method: str,
    expected: int,
) -> None:
    paths = []
    for index, value in enumerate((10, 30, 80), start=1):
        path = tmp_path / f"frame_{index}.npy"
        np.save(path, np.full((5, 7), value, dtype=np.uint16), allow_pickle=False)
        paths.append(path)

    master = combine_frames(paths, method=method, chunk_rows=2)

    assert master.dtype == np.uint16
    assert master.shape == (5, 7)
    assert np.all(master == expected)


def test_create_session_directory_never_overwrites_an_existing_series(tmp_path: Path) -> None:
    request = CalibrationRequest(kind="dark", exposure_ms=100.0, gain=360, offset=350)
    timestamp = datetime(2026, 8, 11, 3, 4, 5).astimezone()

    session = create_session_directory(tmp_path, request, now=timestamp)

    assert session.name == "20260811_030405_dark_100ms_gain360_offset350"
    assert (session / "frames").is_dir()
    with pytest.raises(FileExistsError):
        create_session_directory(tmp_path, request, now=timestamp)


@pytest.mark.parametrize(
    "capture_request",
    [
        CalibrationRequest(kind="dark", exposure_ms=0),
        CalibrationRequest(kind="dark", frames=0),
        CalibrationRequest(kind="dark", warmup_frames=-1),
    ],
)
def test_request_rejects_invalid_capture_parameters(capture_request: CalibrationRequest) -> None:
    with pytest.raises(ValueError):
        capture_request.validate()
