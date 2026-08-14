from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from raw_alignment import build_raw_alignment_signature, estimate_raw_translation


def _stellar_raw(*, height: int = 256, width: int = 384, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width), 700.0, dtype=np.float64)
    for _ in range(28):
        x = float(rng.uniform(12, width - 12))
        y = float(rng.uniform(12, height - 12))
        amplitude = float(rng.uniform(8_000, 42_000))
        sigma = float(rng.uniform(1.1, 2.2))
        image += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    image += rng.normal(0.0, 22.0, image.shape)
    return np.clip(image, 0.0, 65_535.0).astype(np.uint16)


def _translate(raw: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        raw,
        matrix,
        (raw.shape[1], raw.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def test_raw_alignment_recovers_subpixel_translation() -> None:
    reference = _stellar_raw()
    current = _translate(reference, 3.25, -2.5)

    result = estimate_raw_translation(
        build_raw_alignment_signature(reference),
        build_raw_alignment_signature(current),
        search_radius_px=12,
        min_response=0.25,
        use_subpixel=True,
    )

    assert result.ok
    assert result.response > 0.75
    assert abs(result.dx - 3.25) < 0.35
    assert abs(result.dy + 2.5) < 0.35


def test_raw_alignment_rejects_unrelated_noise() -> None:
    rng = np.random.default_rng(18)
    reference = np.clip(700.0 + rng.normal(0.0, 30.0, (256, 384)), 0, 65_535).astype(
        np.uint16
    )
    current = np.clip(700.0 + rng.normal(0.0, 30.0, (256, 384)), 0, 65_535).astype(
        np.uint16
    )

    result = estimate_raw_translation(
        build_raw_alignment_signature(reference),
        build_raw_alignment_signature(current),
        search_radius_px=20,
        min_response=0.25,
    )

    assert not result.ok
    assert result.reason in {"low_confidence", "ambiguous_alignment"}


def test_saved_twenty_second_raw_recording_aligns_without_sep() -> None:
    path = Path("raw_output/raw_20260217_014952.npy")
    if not path.exists():
        pytest.skip("the optional 20-second RAW16 recording is not present")

    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    results = []
    for index in range(0, 50, 5):
        reference = build_raw_alignment_signature(frames[index])
        current = build_raw_alignment_signature(frames[index + 1])
        results.append(
            estimate_raw_translation(
                reference,
                current,
                search_radius_px=50,
                min_response=0.25,
            )
        )

    assert sum(result.ok for result in results) >= 9
    assert float(np.median([result.response for result in results])) > 0.80
