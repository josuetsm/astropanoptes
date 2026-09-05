#!/usr/bin/env python3
"""Resolver por plate solving un stack ya guardado en disco.

Sirve para comprobar el pipeline sin telescopio: los ficheros que guarda
``stacking save`` llevan en el nombre el Az/Alt del apuntado y la marca de
tiempo local, que es exactamente lo que el solver necesita como punto de
partida y como época.

Uso:
    python scripts/solve_saved_stack.py stack_output/*_raw.npy
    python scripts/solve_saved_stack.py --az 98.3 --alt 55.6 archivo_raw.npy
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig                      # noqa: E402
from platesolving import ObserverConfig, solve_plate  # noqa: E402

# stack_YYYYMMDD_HHMMSS_az###p##_alt[p|m]##p##_raw.npy
_NAME_RE = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6})_az(?P<az>[0-9p]+)_alt(?P<altsign>[pm])(?P<alt>[0-9p]+)"
)
# Chile continental en agosto: UTC-4.
_LOCAL_UTC_OFFSET_H = -4.0


def _num(text: str) -> float:
    return float(text.replace("p", "."))


def parse_name(path: Path) -> Tuple[Optional[float], Optional[float], Optional[Time]]:
    m = _NAME_RE.search(path.name)
    if not m:
        # sin coordenadas en el nombre; intenta al menos la marca de tiempo
        m2 = re.search(r"(?P<date>\d{8})_(?P<time>\d{6})", path.name)
        if not m2:
            return None, None, None
        return None, None, _to_time(m2.group("date"), m2.group("time"))
    az = _num(m.group("az"))
    alt = _num(m.group("alt"))
    if m.group("altsign") == "m":
        alt = -alt
    return az, alt, _to_time(m.group("date"), m.group("time"))


def _to_time(date_s: str, time_s: str) -> Time:
    naive = dt.datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    return Time(naive - dt.timedelta(hours=_LOCAL_UTC_OFFSET_H), scale="utc")


def load_stack(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(arr.max())
    if peak > 65535.0 and peak > 0:
        arr = arr * (65535.0 / peak)
    return np.clip(arr, 0.0, 65535.0).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--az", type=float, default=None, help="sobreescribe el Az del nombre")
    ap.add_argument("--alt", type=float, default=None, help="sobreescribe el Alt del nombre")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--min-inliers", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    print(f"{'archivo':46} {'objetivo':>16} {'t':>7}  resultado")
    print("-" * 96)
    for path in args.files:
        az, alt, obstime = parse_name(path)
        if args.az is not None:
            az = args.az
        if args.alt is not None:
            alt = args.alt
        label = path.name[:44]
        if az is None or alt is None:
            print(f"{label:46} {'sin Az/Alt':>16} {'-':>7}  omitido (usa --az/--alt)")
            continue

        img = load_stack(path)
        cfg = AppConfig().platesolving
        cfg.search_radius_deg = float(args.radius)
        cfg.min_inliers = int(args.min_inliers)
        cfg.min_validation_inliers = max(0, int(args.min_inliers) - 3)
        cfg.rotation_prior_enable = False
        cfg.temporal_detection_enabled = False
        cfg.total_timeout_s = float(args.timeout)
        cfg.download_missing_tiles = False

        t0 = time.perf_counter()
        try:
            res = solve_plate(
                img,
                target={"az_deg": float(az), "alt_deg": float(alt)},
                cfg=cfg,
                observer=ObserverConfig(),
                obstime=obstime,
            )
        except Exception as exc:  # catálogo ausente, etc.
            print(f"{label:46} {f'{az:.1f}/{alt:.1f}':>16} {'-':>7}  ERROR {type(exc).__name__}: {exc}")
            continue
        dt_s = time.perf_counter() - t0

        if res.success:
            detail = (
                f"OK  inliers={res.n_inliers:<3} rms={res.rms_px:.2f}px  "
                f"RA={res.center_ra_deg:.3f} Dec={res.center_dec_deg:.3f} rot={res.theta_deg:+.1f}"
            )
        else:
            detail = f"{res.status}  inliers={res.n_inliers}"
        print(f"{label:46} {f'{az:.1f}/{alt:.1f}':>16} {dt_s:6.1f}s  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
