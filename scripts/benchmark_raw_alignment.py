from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raw_alignment import (  # noqa: E402
    build_raw_alignment_signature,
    estimate_raw_translation,
)
from sep_utils import sep_detect_from_raw16  # noqa: E402


def _median_ms(values: list[float]) -> str:
    return f"{statistics.median(values):.2f}" if values else "ERROR"


def benchmark_recording(path: Path, *, samples: int, stride: int) -> dict[str, object]:
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    pair_indices = list(range(0, min(len(frames) - 1, samples * stride), stride))
    alignment_ms: list[float] = []
    sep_ms: list[float] = []
    responses: list[float] = []
    accepted = 0
    sep_errors = 0

    for index in pair_indices:
        reference_raw = np.array(frames[index], copy=True)
        current_raw = np.array(frames[index + 1], copy=True)
        reference = build_raw_alignment_signature(reference_raw)

        started = time.perf_counter()
        current = build_raw_alignment_signature(current_raw)
        result = estimate_raw_translation(
            reference,
            current,
            search_radius_px=50,
            min_response=0.25,
        )
        alignment_ms.append(1_000.0 * (time.perf_counter() - started))
        responses.append(float(result.response))
        accepted += int(result.ok)

        started = time.perf_counter()
        try:
            sep_detect_from_raw16(
                current_raw,
                sep_bw=64,
                sep_bh=64,
                sep_thresh_sigma=3.0,
                sep_minarea=5,
                max_sources=50,
            )
        except Exception:
            sep_errors += 1
        else:
            sep_ms.append(1_000.0 * (time.perf_counter() - started))

    return {
        "file": path.name,
        "pairs": len(pair_indices),
        "accepted": accepted,
        "response": statistics.median(responses) if responses else 0.0,
        "alignment_ms": alignment_ms,
        "sep_ms": sep_ms,
        "sep_errors": sep_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct RAW16 alignment with SEP on saved 20-second recordings."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()

    paths = args.paths or sorted(Path("raw_output").glob("*.npy"))
    if not paths:
        parser.error("no RAW recordings found")

    all_alignment_ms: list[float] = []
    all_sep_ms: list[float] = []
    for path in paths:
        report = benchmark_recording(
            path,
            samples=max(1, int(args.samples)),
            stride=max(1, int(args.stride)),
        )
        alignment_ms = report["alignment_ms"]
        sep_ms = report["sep_ms"]
        assert isinstance(alignment_ms, list)
        assert isinstance(sep_ms, list)
        all_alignment_ms.extend(alignment_ms)
        all_sep_ms.extend(sep_ms)
        print(
            f"{report['file']}: accepted={report['accepted']}/{report['pairs']} "
            f"response={report['response']:.3f} "
            f"raw_align_ms={_median_ms(alignment_ms)} "
            f"sep_ms={_median_ms(sep_ms)} sep_errors={report['sep_errors']}"
        )

    print(
        f"TOTAL: raw_align_ms={_median_ms(all_alignment_ms)} "
        f"sep_ms={_median_ms(all_sep_ms)}"
    )
    if args.require_speedup and all_sep_ms:
        return int(statistics.median(all_alignment_ms) >= statistics.median(all_sep_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
