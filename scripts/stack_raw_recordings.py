#!/usr/bin/env python3
"""Create memory-efficient color drizzle stacks from recorded RAW sequences."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from stacking import LiveMosaicStackerGray, _bayer_to_gray_code, _bayer_to_rgb_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align RAW recordings at native resolution, then make a fixed-size "
            "color stack at the requested drizzle scale."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Recorded RAW .npy files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("stack_output/raw_drizzle")
    )
    parser.add_argument("--scale", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument(
        "--bayer-pattern",
        default="RGGB",
        choices=("RGGB", "BGGR", "GRBG", "GBRG"),
    )
    parser.add_argument("--min-response", type=float, default=0.25)
    parser.add_argument(
        "--max-shift",
        type=int,
        default=50,
        help="Search radius in native sensor pixels",
    )
    parser.add_argument("--smooth-k", type=int, default=30)
    parser.add_argument("--asinh-strength", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_progress(total: int, every: int = 25) -> Iterable[int]:
    for index in range(total):
        if index == 0 or (index + 1) % every == 0 or index + 1 == total:
            print(f"  frame {index + 1:>3}/{total}", flush=True)
        yield index


def _alignment_positions(
    raw_frames: np.ndarray,
    *,
    bayer_pattern: str,
    min_response: float,
    max_shift: int,
    smooth_k: int,
) -> tuple[list[tuple[float, float] | None], list[float]]:
    aligner = LiveMosaicStackerGray(
        color_mode="mono",
        resp_min=min_response,
        align_median_k=3,
        smooth_k=smooth_k,
        max_shift_px=max_shift,
        use_subpixel=True,
        preview_log_vmin=5.0,
        drizzle_scale=1.0,
        bayer_to_gray_code=_bayer_to_gray_code(bayer_pattern),
        bayer_to_rgb_code=_bayer_to_rgb_code(bayer_pattern),
    )
    positions: list[tuple[float, float] | None] = []
    responses: list[float] = []
    world_x = 0.0
    world_y = 0.0

    print("  alineando a resolución nativa", flush=True)
    for index in _iter_progress(len(raw_frames)):
        accepted = aligner.add_frame(raw_frames[index])
        responses.append(float(aligner.last_resp))
        if not accepted:
            positions.append(None)
            continue
        if index > 0:
            world_x -= float(aligner.last_dx)
            world_y -= float(aligner.last_dy)
        positions.append((world_x, world_y))

    return positions, responses


def _common_bounds(
    positions: list[tuple[float, float] | None],
    *,
    native_h: int,
    native_w: int,
    scale: int,
) -> tuple[int, int, int, int]:
    valid = [position for position in positions if position is not None]
    if not valid:
        raise RuntimeError("No frame passed the alignment quality threshold")

    output_h = native_h * scale
    output_w = native_w * scale
    # Two output pixels of safety avoid interpolation against an image border.
    margin = 2
    left = int(math.ceil(max(x * scale for x, _ in valid))) + margin
    top = int(math.ceil(max(y * scale for _, y in valid))) + margin
    right = int(math.floor(min(x * scale + output_w for x, _ in valid))) - margin
    bottom = int(math.floor(min(y * scale + output_h for _, y in valid))) - margin
    if right <= left or bottom <= top:
        raise RuntimeError("Accepted frames do not share a common field of view")
    return left, top, right, bottom


def _combine_common_field(
    raw_frames: np.ndarray,
    positions: list[tuple[float, float] | None],
    *,
    bayer_pattern: str,
    scale: int,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    rgb_code = _bayer_to_rgb_code(bayer_pattern)
    accepted_indices = [
        index for index, position in enumerate(positions) if position is not None
    ]

    print(
        f"  combinando {len(accepted_indices)} frames en campo común "
        f"{width}x{height}",
        flush=True,
    )
    for progress_index, frame_index in enumerate(accepted_indices):
        show_progress = (
            progress_index == 0
            or (progress_index + 1) % 25 == 0
            or progress_index + 1 == len(accepted_indices)
        )
        if show_progress:
            print(
                f"  frame útil {progress_index + 1:>3}/{len(accepted_indices)}",
                flush=True,
            )

        position = positions[frame_index]
        assert position is not None
        world_x, world_y = position
        rgb = cv2.cvtColor(raw_frames[frame_index], rgb_code)

        # Match cv2.resize's pixel-center convention while applying scale and
        # the measured subpixel translation in a single allocation.
        center_offset = (float(scale) - 1.0) * 0.5
        matrix = np.array(
            [
                [float(scale), 0.0, world_x * scale - left + center_offset],
                [0.0, float(scale), world_y * scale - top + center_offset],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        accumulator += warped

    accumulator /= float(len(accepted_indices))
    return np.clip(accumulator, 0.0, 65535.0).astype(np.uint16)


def _stretch_rgb(
    rgb_u16: np.ndarray, *, strength: float
) -> tuple[np.ndarray, dict[str, object]]:
    # Estimate display parameters from a sparse sample to keep peak memory low.
    sample = rgb_u16[::8, ::8].astype(np.float32)
    raw_sample_luma = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    luma_q16, luma_q50, luma_q84, luma_q99, luma_q999, luma_q9998 = np.percentile(
        raw_sample_luma, (16.0, 50.0, 84.0, 99.0, 99.9, 99.98)
    )
    background_width = max(float(luma_q99 - luma_q50), 1.0)
    stellar_contrast = float((luma_q9998 - luma_q50) / background_width)
    background_noise = max(float(luma_q84 - luma_q16) * 0.5, 0.5)
    stellar_density_snr = float((luma_q999 - luma_q50) / background_noise)
    low_signal = stellar_contrast < 20.0 or stellar_density_snr < 25.0
    p16 = np.percentile(sample, 16.0, axis=(0, 1))
    median = np.percentile(sample, 50.0, axis=(0, 1))
    background_sigma = median - p16
    if low_signal:
        black = median + 0.75 * background_sigma
    else:
        black = median - 1.5 * background_sigma
    black = np.maximum(0.0, black).astype(np.float32)

    sample -= black
    np.maximum(sample, 0.0, out=sample)
    sample_luma = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    white = max(float(np.percentile(sample_luma, 99.98)), 1.0)

    image = rgb_u16.astype(np.float32)
    image -= black
    np.maximum(image, 0.0, out=image)
    image /= white

    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    strength = 4.0 if low_signal else max(float(strength), 0.1)
    mapped_luma = np.arcsinh(strength * luma) / np.arcsinh(strength)
    gain = np.divide(mapped_luma, luma, out=np.zeros_like(luma), where=luma > 1e-7)
    image *= gain[..., None]

    # A restrained saturation lift makes stellar colors visible without
    # changing neutral background pixels.
    display_luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image -= display_luma[..., None]
    image *= 0.65 if low_signal else 1.15
    image += display_luma[..., None]
    np.clip(image, 0.0, 1.0, out=image)
    # Low-signal stacks need a gentler gamma or the display transform turns
    # background read noise into false color detail.
    display_gamma = 0.9 if low_signal else 0.5
    np.power(image, display_gamma, out=image)
    stretched = (image * 65535.0 + 0.5).astype(np.uint16)
    stats = {
        "black_rgb": [float(value) for value in black],
        "white_luma": white,
        "asinh_strength": strength,
        "display_gamma": display_gamma,
        "stellar_contrast": stellar_contrast,
        "stellar_density_snr": stellar_density_snr,
        "quality": "low_signal" if low_signal else "good",
    }
    return stretched, stats


def _save_preview(
    path: Path, stretched_rgb: np.ndarray, *, max_width: int = 1800
) -> None:
    height, width = stretched_rgb.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        preview = cv2.resize(
            stretched_rgb,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = stretched_rgb
    preview_u8 = (preview / 257.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def stack_recording(path: Path, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    raw_frames = np.load(path, mmap_mode="r")
    if raw_frames.ndim != 3 or raw_frames.dtype != np.uint16:
        raise ValueError(
            f"{path}: expected (frames, height, width) uint16, got "
            f"{raw_frames.shape} {raw_frames.dtype}"
        )

    stem = f"{path.stem}_drizzle{args.scale}x_rgb"
    linear_path = args.output_dir / f"{stem}_linear.npy"
    png_path = args.output_dir / f"{stem}.png"
    full_jpeg_path = args.output_dir / f"{stem}_full.jpg"
    preview_path = args.output_dir / f"{stem}_preview.jpg"
    metadata_path = args.output_dir / f"{stem}.json"
    outputs = (linear_path, png_path, full_jpeg_path, preview_path, metadata_path)
    if not args.overwrite and any(output.exists() for output in outputs):
        raise FileExistsError(f"Output already exists for {path}; use --overwrite")

    print(f"\n{path}", flush=True)
    positions, responses = _alignment_positions(
        raw_frames,
        bayer_pattern=args.bayer_pattern,
        min_response=args.min_response,
        max_shift=args.max_shift,
        smooth_k=args.smooth_k,
    )
    native_h, native_w = raw_frames.shape[1:]
    bounds = _common_bounds(
        positions,
        native_h=native_h,
        native_w=native_w,
        scale=args.scale,
    )
    linear = _combine_common_field(
        raw_frames,
        positions,
        bayer_pattern=args.bayer_pattern,
        scale=args.scale,
        bounds=bounds,
    )

    np.save(linear_path, linear, allow_pickle=False)
    stretched, stretch_stats = _stretch_rgb(linear, strength=args.asinh_strength)
    cv2.imwrite(str(png_path), cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR))
    full_jpeg_u8 = (stretched / 257.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(full_jpeg_path),
        cv2.cvtColor(full_jpeg_u8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 96],
    )
    _save_preview(preview_path, stretched)

    accepted = sum(position is not None for position in positions)
    accepted_responses = [
        response
        for position, response in zip(positions, responses)
        if position is not None
    ]
    metadata = {
        "source": str(path),
        "source_shape": list(raw_frames.shape),
        "source_dtype": str(raw_frames.dtype),
        "bayer_pattern": args.bayer_pattern,
        "drizzle_scale": args.scale,
        "alignment_resolution": "native",
        "alignment_algorithm": "raw_alignment_signature",
        "min_response": args.min_response,
        "max_shift_native_px": args.max_shift,
        "frames_total": int(len(raw_frames)),
        "frames_used": accepted,
        "frames_rejected": int(len(raw_frames) - accepted),
        "median_response": float(np.median(accepted_responses)),
        "minimum_accepted_response": float(min(accepted_responses)),
        "common_field_bounds_drizzled": list(bounds),
        "output_shape": list(linear.shape),
        "positions_native_px": [
            list(position) if position is not None else None for position in positions
        ],
        "stretch": stretch_stats,
        "full_resolution_files": {
            "png_16bit": str(png_path),
            "jpeg_8bit": str(full_jpeg_path),
            "linear_npy": str(linear_path),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(
        f"  listo: {accepted}/{len(raw_frames)} frames, "
        f"{linear.shape[1]}x{linear.shape[0]}, "
        f"respuesta mediana {metadata['median_response']:.3f}, {elapsed:.1f}s",
        flush=True,
    )
    if stretch_stats["quality"] == "low_signal":
        print(
            "  aviso: señal estelar débil; se guardó con estirado conservador "
            "y no se recomienda como selección final",
            flush=True,
        )
    print(f"  {png_path}", flush=True)

    del stretched, linear, raw_frames
    gc.collect()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        stack_recording(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
