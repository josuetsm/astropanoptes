#!/usr/bin/env python3
"""Register and photometrically combine per-recording RAW drizzle stacks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from raw_alignment import build_raw_alignment_signature, estimate_raw_translation
from scripts.stack_raw_recordings import _save_preview, _stretch_rgb


@dataclass
class StackInput:
    metadata_path: Path
    metadata: dict[str, object]
    linear_path: Path
    shape: tuple[int, int, int]
    frame_index: int
    position_native: np.ndarray
    bounds_left_top: np.ndarray
    translation: np.ndarray
    response: float
    background_rgb: np.ndarray
    noise_luma: float
    photometric_ratio: float = 1.0
    information_weight: float = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register multiple per-recording linear drizzle stacks and combine "
            "them using photometric/inverse-noise weighting."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Per-recording JSON metadata"
    )
    parser.add_argument("--name", required=True, help="Output filename stem")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("stack_output/raw_drizzle_x3")
    )
    parser.add_argument("--coverage-fraction", type=float, default=0.55)
    parser.add_argument("--cross-max-shift", type=int, default=1000)
    parser.add_argument("--cross-min-response", type=float, default=0.50)
    parser.add_argument("--asinh-strength", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _nearest_accepted_position(metadata: dict[str, object]) -> tuple[int, np.ndarray]:
    positions = metadata["positions_native_px"]
    if not isinstance(positions, list) or not positions:
        raise ValueError("metadata has no frame positions")
    midpoint = len(positions) // 2
    candidates = [
        (abs(index - midpoint), index)
        for index, position in enumerate(positions)
        if position is not None
    ]
    if not candidates:
        raise ValueError("metadata has no accepted frame")
    frame_index = min(candidates)[1]
    return frame_index, np.asarray(positions[frame_index], dtype=np.float64)


def _linear_path_for(metadata_path: Path) -> Path:
    name = metadata_path.name
    if not name.endswith(".json"):
        raise ValueError(f"expected JSON metadata, got {metadata_path}")
    return metadata_path.with_name(name[:-5] + "_linear.npy")


def _image_stats(linear_path: Path) -> tuple[np.ndarray, float]:
    image = np.load(linear_path, mmap_mode="r", allow_pickle=False)
    sample = image[::8, ::8].astype(np.float32)
    background_rgb = np.percentile(sample, 50.0, axis=(0, 1)).astype(np.float32)
    luma = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    p16, p84 = np.percentile(luma, (16.0, 84.0))
    noise = max(float(p84 - p16) * 0.5, 0.5)
    return background_rgb, noise


def _load_inputs(paths: list[Path]) -> list[StackInput]:
    loaded: list[StackInput] = []
    for metadata_path in paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        linear_path = _linear_path_for(metadata_path)
        image = np.load(linear_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint16:
            raise ValueError(f"{linear_path}: expected RGB uint16 stack")
        frame_index, position = _nearest_accepted_position(metadata)
        bounds = np.asarray(
            metadata["common_field_bounds_drizzled"][:2], dtype=np.float64
        )
        background_rgb, noise = _image_stats(linear_path)
        loaded.append(
            StackInput(
                metadata_path=metadata_path,
                metadata=metadata,
                linear_path=linear_path,
                shape=tuple(int(value) for value in image.shape),
                frame_index=frame_index,
                position_native=position,
                bounds_left_top=bounds,
                translation=np.zeros(2, dtype=np.float64),
                response=1.0,
                background_rgb=background_rgb,
                noise_luma=noise,
            )
        )
    return loaded


def _register_inputs(
    stacks: list[StackInput], *, max_shift: int, min_response: float
) -> None:
    anchor = stacks[0]
    anchor_raw = np.load(
        str(anchor.metadata["source"]), mmap_mode="r", allow_pickle=False
    )
    anchor_signature = build_raw_alignment_signature(anchor_raw[anchor.frame_index])
    scale = float(anchor.metadata["drizzle_scale"])

    for stack in stacks:
        raw_frames = np.load(
            str(stack.metadata["source"]), mmap_mode="r", allow_pickle=False
        )
        signature = build_raw_alignment_signature(raw_frames[stack.frame_index])
        alignment = estimate_raw_translation(
            anchor_signature,
            signature,
            search_radius_px=float(max_shift),
            min_response=float(min_response),
            use_subpixel=True,
            max_profile_disagreement_px=6.0,
        )
        if not alignment.ok:
            raise RuntimeError(
                f"{stack.metadata_path}: cross-recording alignment rejected "
                f"({alignment.reason}, response={alignment.response:.3f})"
            )
        displacement = np.array([alignment.dx, alignment.dy], dtype=np.float64)
        delta = (
            scale
            * (
                stack.position_native
                - anchor.position_native
                + displacement
            )
            + anchor.bounds_left_top
            - stack.bounds_left_top
        )
        # A current-stack pixel q corresponds to anchor coordinates q - delta.
        stack.translation = -delta
        stack.response = float(alignment.response)
        print(
            f"  {Path(str(stack.metadata['source'])).name}: "
            f"response={stack.response:.3f} "
            f"translation=({stack.translation[0]:.1f}, {stack.translation[1]:.1f})",
            flush=True,
        )


def _luma_sample(linear_path: Path, stride: int) -> np.ndarray:
    image = np.load(linear_path, mmap_mode="r", allow_pickle=False)
    sample = image[::stride, ::stride].astype(np.float32)
    return cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)


def _fit_photometry(stacks: list[StackInput]) -> None:
    anchor = stacks[0]
    stride = 6
    anchor_luma = _luma_sample(anchor.linear_path, stride)
    anchor_background = float(
        0.299 * anchor.background_rgb[0]
        + 0.587 * anchor.background_rgb[1]
        + 0.114 * anchor.background_rgb[2]
    )
    anchor_signal = anchor_luma - anchor_background

    for stack in stacks:
        current_luma = _luma_sample(stack.linear_path, stride)
        current_background = float(
            0.299 * stack.background_rgb[0]
            + 0.587 * stack.background_rgb[1]
            + 0.114 * stack.background_rgb[2]
        )
        matrix = np.array(
            [
                [1.0, 0.0, stack.translation[0] / stride],
                [0.0, 1.0, stack.translation[1] / stride],
            ],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            current_luma,
            matrix,
            (anchor_luma.shape[1], anchor_luma.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        coverage = cv2.warpAffine(
            np.ones(current_luma.shape, dtype=np.uint8),
            matrix,
            (anchor_luma.shape[1], anchor_luma.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        current_signal = aligned - current_background
        valid = (
            coverage
            & (anchor_signal > max(8.0 * anchor.noise_luma, 30.0))
            & (anchor_signal < 20_000.0)
            & (current_signal > max(3.0 * stack.noise_luma, 5.0))
            & (current_signal < 20_000.0)
        )
        ratios = np.divide(
            current_signal[valid],
            anchor_signal[valid],
        )
        ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        if ratios.size < 20:
            raise RuntimeError(
                f"{stack.metadata_path}: only {ratios.size} photometric samples"
            )
        lo, hi = np.percentile(ratios, (15.0, 85.0))
        trimmed = ratios[(ratios >= lo) & (ratios <= hi)]
        ratio = max(float(np.median(trimmed)), 1e-4)
        gain = 1.0 / ratio
        normalized_noise = float(stack.noise_luma) * gain
        weight = (float(anchor.noise_luma) / max(normalized_noise, 1e-6)) ** 2
        stack.photometric_ratio = ratio
        stack.information_weight = float(np.clip(weight, 1e-4, 4.0))
        print(
            f"    photometry ratio={ratio:.4f} weight={stack.information_weight:.4f} "
            f"samples={ratios.size}",
            flush=True,
        )


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return left, top, right, bottom for the largest all-true rectangle."""
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, width, height)
    for row in range(height):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current_height = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current_height:
                start_index, bar_height = stack.pop()
                area = bar_height * (column - start_index)
                if area > best_area:
                    best_area = area
                    best = (
                        start_index,
                        row - bar_height + 1,
                        column,
                        row + 1,
                    )
                start = start_index
            if current_height > 0 and (
                not stack or stack[-1][1] < current_height
            ):
                stack.append((start, current_height))
    if best_area <= 0:
        raise RuntimeError("coverage mask has no valid rectangle")
    return best


def _combine(
    stacks: list[StackInput], *, coverage_fraction: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    min_x = int(math.floor(min(stack.translation[0] for stack in stacks)))
    min_y = int(math.floor(min(stack.translation[1] for stack in stacks)))
    max_x = int(
        math.ceil(max(stack.translation[0] + stack.shape[1] for stack in stacks))
    )
    max_y = int(
        math.ceil(max(stack.translation[1] + stack.shape[0] for stack in stacks))
    )
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for stack in stacks:
        image = np.load(stack.linear_path, mmap_mode="r", allow_pickle=False)
        signal = image.astype(np.float32)
        signal -= stack.background_rgb
        gain = 1.0 / stack.photometric_ratio
        weighted_gain = gain * stack.information_weight

        placement = stack.translation - np.array([min_x, min_y], dtype=np.float64)
        x0 = int(math.floor(placement[0]))
        y0 = int(math.floor(placement[1]))
        fx = float(placement[0] - x0)
        fy = float(placement[1] - y0)
        h, w = image.shape[:2]
        if abs(fx) > 1e-4 or abs(fy) > 1e-4:
            matrix = np.array([[1.0, 0.0, fx], [0.0, 1.0, fy]], dtype=np.float32)
            signal = cv2.warpAffine(
                signal,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = cv2.warpAffine(
                np.ones((h, w), dtype=np.float32),
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            mask = np.ones((h, w), dtype=np.float32)

        signal *= weighted_gain
        accumulator[y0 : y0 + h, x0 : x0 + w] += signal
        weight_map[y0 : y0 + h, x0 : x0 + w] += (
            mask * stack.information_weight
        )

    mean_signal = np.zeros_like(accumulator, dtype=np.float32)
    np.divide(
        accumulator,
        weight_map[..., None],
        out=mean_signal,
        where=weight_map[..., None] > 0.0,
    )
    positive_weights = weight_map[weight_map > 0.0]
    max_coverage = float(np.percentile(positive_weights, 99.9))
    threshold = max(float(coverage_fraction), 0.0) * max_coverage
    valid = weight_map >= threshold
    left, top, right, bottom = _largest_valid_rectangle(valid)

    background = stacks[0].background_rgb.astype(np.float32)
    combined = mean_signal[top:bottom, left:right]
    combined += background
    linear = np.clip(combined, 0.0, 65535.0).astype(np.uint16)
    coverage = weight_map[top:bottom, left:right].astype(np.float32, copy=False)
    return linear, coverage, (min_x + left, min_y + top, min_x + right, min_y + bottom)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / args.name
    linear_path = output_stem.with_name(output_stem.name + "_linear.npy")
    coverage_path = output_stem.with_name(output_stem.name + "_coverage.npy")
    png_path = output_stem.with_suffix(".png")
    full_jpeg_path = output_stem.with_name(output_stem.name + "_full.jpg")
    preview_path = output_stem.with_name(output_stem.name + "_preview.jpg")
    metadata_path = output_stem.with_suffix(".json")
    outputs = (
        linear_path,
        coverage_path,
        png_path,
        full_jpeg_path,
        preview_path,
        metadata_path,
    )
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("combined output exists; use --overwrite")

    stacks = _load_inputs(args.inputs)
    scale_values = {float(stack.metadata["drizzle_scale"]) for stack in stacks}
    if len(scale_values) != 1:
        raise ValueError("all input stacks must use the same drizzle scale")
    print("Registrando grabaciones:", flush=True)
    _register_inputs(
        stacks,
        max_shift=args.cross_max_shift,
        min_response=args.cross_min_response,
    )
    print("Normalizando señal y ruido:", flush=True)
    _fit_photometry(stacks)
    print("Combinando campos:", flush=True)
    linear, coverage, crop_bounds = _combine(
        stacks, coverage_fraction=args.coverage_fraction
    )
    np.save(linear_path, linear, allow_pickle=False)
    np.save(coverage_path, coverage, allow_pickle=False)
    stretched, stretch_stats = _stretch_rgb(linear, strength=args.asinh_strength)
    cv2.imwrite(str(png_path), cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR))
    full_jpeg_u8 = (stretched / 257.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(full_jpeg_path),
        cv2.cvtColor(full_jpeg_u8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 96],
    )
    _save_preview(preview_path, stretched)

    metadata = {
        "name": args.name,
        "drizzle_scale": next(iter(scale_values)),
        "frames_total": int(
            sum(int(stack.metadata["frames_total"]) for stack in stacks)
        ),
        "frames_used": int(sum(int(stack.metadata["frames_used"]) for stack in stacks)),
        "recordings": [
            {
                "source": stack.metadata["source"],
                "frames_used": stack.metadata["frames_used"],
                "cross_response": stack.response,
                "translation_drizzled_px": stack.translation.tolist(),
                "photometric_ratio_to_anchor": stack.photometric_ratio,
                "information_weight": stack.information_weight,
            }
            for stack in stacks
        ],
        "coverage_fraction": float(args.coverage_fraction),
        "crop_bounds_anchor_px": list(crop_bounds),
        "output_shape": list(linear.shape),
        "stretch": stretch_stats,
        "full_resolution_files": {
            "png_16bit": str(png_path),
            "jpeg_8bit": str(full_jpeg_path),
            "linear_npy": str(linear_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Listo: {metadata['frames_used']}/{metadata['frames_total']} frames, "
        f"{linear.shape[1]}x{linear.shape[0]} -> {png_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
