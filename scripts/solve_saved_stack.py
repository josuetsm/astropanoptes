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
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig                      # noqa: E402
from platesolving import ObserverConfig, solve_plate  # noqa: E402
from stacking import flatten_mosaic_border        # noqa: E402

# stack_YYYYMMDD_HHMMSS_az###p##_alt[p|m]##p##_raw.npy
_NAME_RE = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6})_az(?P<az>[0-9p]+)_alt(?P<altsign>[pm])(?P<alt>[0-9p]+)"
)
# El nombre lleva la hora *local* del equipo que capturo el stack. Interpretarla
# con la zona local del sistema es lo unico que acierta todo el ano: Chile pasa
# a horario de verano el primer domingo de septiembre, y el desfase fijo de -4
# que habia aqui desplazaba la epoca una hora en cuanto empezaba el DST. Una
# hora son 15 grados de angulo horario, asi que el solver buscaba el campo a 15
# grados de donde estaba y solo encontraba coincidencias falsas de 3 inliers.


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
    local = naive.astimezone()  # zona (y DST) del sistema para esa fecha
    return Time(local.astimezone(dt.timezone.utc).replace(tzinfo=None), scale="utc")


def _companion(path: Path, suffix: str) -> Path:
    """Ruta del fichero hermano: <base>_raw.npy -> <base><suffix>."""
    name = path.name
    stem = name[: -len("_raw.npy")] if name.endswith("_raw.npy") else path.stem
    return path.with_name(stem + suffix)


def load_sidecar(path: Path) -> dict:
    """Metadatos guardados junto al stack; {} si el stack es anterior a ellos."""
    meta_path = _companion(path, ".json")
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_weight(path: Path) -> Optional[np.ndarray]:
    wgt_path = _companion(path, "_wgt.npy")
    if not wgt_path.exists():
        return None
    try:
        return np.asarray(np.load(wgt_path), dtype=np.float32)
    except (OSError, ValueError):
        return None


def coverage_weight(img: np.ndarray) -> np.ndarray:
    """Peso aproximado para los stacks guardados antes de que hubiera sidecar.

    El mosaico deja a cero lo que ningun fotograma cubrio, asi que la zona
    cubierta se recupera del propio stack. El anillo del borde si esta cubierto,
    solo que por menos fotogramas, y es justo el que forma la escalera que el
    detector lee como estrellas; erosionar unos pixeles lo deja fuera igual que
    haria el umbral de cobertura del mapa real. Un stack sin borde (todo
    cubierto) queda intacto.
    """
    covered = (np.asarray(img) > 0).astype(np.uint8)
    if covered.all():
        return np.ones(covered.shape, dtype=np.float32)
    kernel = np.ones((9, 9), np.uint8)
    return cv2.erode(covered, kernel, iterations=1).astype(np.float32)


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

        # El sidecar que escribe "stacking save" manda sobre el nombre: lleva la
        # epoca en UTC y el drizzle, que el nombre no puede expresar.
        meta = load_sidecar(path)
        obstime_unix = meta.get("obstime_unix")
        if obstime_unix is not None:
            obstime = Time(float(obstime_unix), format="unix", scale="utc")
        if meta.get("pointing_az_deg") is not None and args.az is None:
            az = float(meta["pointing_az_deg"])
        if meta.get("pointing_alt_deg") is not None and args.alt is None:
            alt = float(meta["pointing_alt_deg"])

        # Sin aplanar el borde dentado, los escalones del mosaico son las
        # "estrellas" mas brillantes del cuadro y el solve no llega a ninguna
        # parte: medido sobre un stack real, 26 de las 30 detecciones mas
        # brillantes eran borde, y aplanarlo lo llevo de 3 inliers a 16.
        wgt = load_weight(path)
        if wgt is None or wgt.shape != img.shape:
            wgt = coverage_weight(img)
        img, _sky = flatten_mosaic_border(img, wgt)

        cfg = AppConfig().platesolving
        cfg.search_radius_deg = float(args.radius)
        # El drizzle subdivide pixeles, asi que la escala angular por pixel baja
        # en ese factor. Escalar la focal es equivalente, igual que en la app.
        drizzle = float(meta.get("drizzle_scale", 1.0) or 1.0)
        if np.isfinite(drizzle) and drizzle > 1.0:
            cfg.focal_m = float(cfg.focal_m) * drizzle
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
