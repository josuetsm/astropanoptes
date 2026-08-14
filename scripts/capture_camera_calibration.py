#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_capture import CalibrationRequest, capture_calibration_series
from config import CameraConfig


def _parser() -> argparse.ArgumentParser:
    defaults = CameraConfig()
    parser = argparse.ArgumentParser(
        description="Captura una serie dark o blank RAW16 y genera su master.",
    )
    parser.add_argument(
        "kind",
        choices=("dark", "blank"),
        help="dark: sensor tapado; blank: óptica destapada apuntando al campo de referencia vacío.",
    )
    parser.add_argument("--exposure-ms", type=float, default=defaults.exp_ms)
    parser.add_argument("--gain", type=int, default=defaults.gain)
    parser.add_argument("--offset", type=int, default=defaults.offset)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index)
    parser.add_argument("--combine", choices=("median", "mean"), default="median")
    parser.add_argument("--output", type=Path, default=Path("calibration_frames"))
    parser.add_argument(
        "--yes",
        action="store_true",
        help="No pedir confirmación de la condición óptica.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = CalibrationRequest(
        kind=args.kind,
        exposure_ms=args.exposure_ms,
        gain=args.gain,
        offset=args.offset,
        frames=args.frames,
        warmup_frames=args.warmup_frames,
        camera_index=args.camera_index,
        combine=args.combine,
    )
    request.validate()

    if not args.yes:
        if request.kind == "dark":
            condition = "Tapa completamente el sensor o telescopio."
        else:
            condition = "Destapa la óptica y apunta al campo de referencia blank."
        print(condition)
        print(
            f"Se capturarán {request.frames} frames a {request.exposure_ms:g} ms, "
            f"gain {request.gain}, offset {request.offset}."
        )
        try:
            input("Presiona Enter cuando esté listo (Ctrl-C para cancelar)... ")
        except (EOFError, KeyboardInterrupt):
            print("\nCaptura cancelada; no se abrió la cámara.", file=sys.stderr)
            return 130

    session = capture_calibration_series(request, output_root=args.output)
    print(f"Serie terminada: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
