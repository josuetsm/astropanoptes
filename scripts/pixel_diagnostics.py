#!/usr/bin/env python3
"""Diagnóstico de píxeles defectuosos del sensor.

Un píxel defectuoso está fijo en el sensor; una estrella no. Esa es la única
diferencia que hace falta para separarlos, y es la base de todo lo que hay aquí.

Dos fuentes de evidencia, que se combinan:

* **Master dark** (sin luz): cualquier píxel muy por encima del nivel de
  oscuridad es defectuoso por definición. Es la evidencia más limpia, pero solo
  ve los que están calientes a la temperatura y exposición de esa toma.
* **Frames de cielo de apuntados distintos**: un píxel que aparece brillante en
  la misma coordenada del sensor en tomas de campos diferentes no puede ser una
  estrella. Esto atrapa además los intermitentes que un único dark no revela.

El resultado es un mapa booleano que se puede aplicar a cualquier frame antes de
la detección de fuentes, evitando que el plate solving gaste tripletas en
píxeles muertos.

Uso:
    python scripts/pixel_diagnostics.py                 # informe
    python scripts/pixel_diagnostics.py --save mapa.npy # y guarda el mapa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DARK = ROOT / "calibration_frames/20260811_153352_dark_100ms_gain360_offset350/master_dark.npy"


def _as2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a.mean(axis=0) if a.shape[0] < a.shape[-1] else a.mean(axis=2)
    return np.asarray(a, dtype=np.float64)


def hot_from_dark(dark_path: Path, sigma: float = 8.0) -> Tuple[np.ndarray, dict]:
    """Píxeles calientes según un master dark."""
    dark = _as2d(np.load(dark_path, mmap_mode="r"))
    med = float(np.median(dark))
    # MAD: robusta frente a los propios píxeles calientes que buscamos
    mad = float(np.median(np.abs(dark - med)))
    sd = 1.4826 * mad if mad > 0 else float(dark.std())
    thr = med + sigma * sd
    mask = dark > thr
    return mask, {
        "shape": dark.shape,
        "median": med,
        "sigma_robusta": sd,
        "umbral": thr,
        "n": int(mask.sum()),
        "max": float(dark.max()),
    }


def _frame_iter(path: Path, max_frames: int = 3):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        yield np.asarray(arr, dtype=np.float64)
        return
    n = min(int(arr.shape[0]), max_frames)
    for i in np.linspace(0, arr.shape[0] - 1, n).astype(int):
        yield np.asarray(arr[i], dtype=np.float64)


def hot_from_sky(paths: List[Path], sigma: float = 6.0, min_hits: int = 3):
    """Píxeles brillantes que se repiten en la misma coordenada del sensor.

    Cada *archivo* cuenta como una sola observación, no cada frame: los frames
    de una misma grabación miran el mismo campo, así que sus estrellas caen en
    los mismos píxeles y se contarían como defectos. Dentro de cada archivo se
    usa la mediana temporal, que además elimina rayos cósmicos y deja en pie
    justo lo que persiste: estrellas de ese campo y defectos del sensor.

    Comparando después entre archivos de apuntados distintos, las estrellas se
    mueven y los defectos no.
    """
    counts: Optional[np.ndarray] = None
    used = 0
    for p in paths:
        try:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim == 3:
                n = int(arr.shape[0])
                idx = np.linspace(0, n - 1, min(n, 9)).astype(int)
                stack = np.stack([np.asarray(arr[i], dtype=np.float32) for i in idx])
                frame = np.median(stack, axis=0).astype(np.float64)
            else:
                frame = np.asarray(arr, dtype=np.float64)
        except Exception:
            continue
        if counts is None:
            counts = np.zeros(frame.shape, dtype=np.int32)
        elif frame.shape != counts.shape:
            continue
        med = float(np.median(frame))
        mad = float(np.median(np.abs(frame - med)))
        sd = 1.4826 * mad if mad > 0 else float(frame.std())
        counts += (frame > med + sigma * sd).astype(np.int32)
        used += 1
    if counts is None:
        return None, {"tomas": 0}
    return counts >= int(min_hits), {"tomas": used, "min_hits": int(min_hits)}


def describe(mask: np.ndarray) -> dict:
    """Agrupa los defectos en píxeles sueltos, pares y racimos."""
    import cv2

    n, _lab, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    sizes = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    singles = sum(1 for s in sizes if s == 1)
    pairs = sum(1 for s in sizes if s == 2)
    clusters = sum(1 for s in sizes if s >= 3)
    worst = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), int(cent[i][0]), int(cent[i][1])) for i in range(1, n)),
        reverse=True,
    )[:8]
    return {
        "defectos": int(n - 1),
        "pixeles": int(mask.sum()),
        "sueltos": singles,
        "pares": pairs,
        "racimos": clusters,
        "mayores": worst,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dark", type=Path, default=DEFAULT_DARK)
    ap.add_argument("--sky", nargs="*", type=Path, default=None)
    ap.add_argument("--dark-sigma", type=float, default=8.0)
    ap.add_argument("--sky-sigma", type=float, default=6.0)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 74)
    print("DIAGNOSTICO DE PIXELES DEFECTUOSOS")
    print("=" * 74)

    dark_mask = None
    if args.dark and args.dark.exists():
        dark_mask, info = hot_from_dark(args.dark, args.dark_sigma)
        print(f"\n[1] MASTER DARK  {args.dark.name}")
        print(f"    sensor {info['shape'][1]}x{info['shape'][0]}   nivel={info['median']:.1f} ADU"
              f"   sigma={info['sigma_robusta']:.2f}")
        print(f"    umbral {args.dark_sigma:g} sigma = {info['umbral']:.1f} ADU   pico={info['max']:.0f} ADU")
        d = describe(dark_mask)
        print(f"    -> {d['pixeles']} pixeles calientes en {d['defectos']} defectos "
              f"({d['sueltos']} sueltos, {d['pares']} pares, {d['racimos']} racimos)")
        frac = 100.0 * d["pixeles"] / dark_mask.size
        print(f"       {frac:.4f}% del sensor")
        if d["mayores"]:
            print("       mayores:  " + "  ".join(f"{a}px@({x},{y})" for a, x, y in d["mayores"][:5]))
    else:
        print(f"\n[1] MASTER DARK  no encontrado en {args.dark}")

    sky_mask = None
    if args.sky:
        sky_mask, sinfo = hot_from_sky(args.sky, args.sky_sigma, args.min_hits)
        print(f"\n[2] CIELO  {sinfo['tomas']} tomas de apuntados distintos")
        if sky_mask is not None:
            s = describe(sky_mask)
            print(f"    -> {s['pixeles']} pixeles brillantes repetidos en >= {args.min_hits} tomas")
            print(f"       {s['defectos']} defectos ({s['sueltos']} sueltos, {s['racimos']} racimos)")

    if dark_mask is not None and sky_mask is not None:
        both = dark_mask & sky_mask
        only_sky = sky_mask & ~dark_mask
        print("\n[3] COMBINADO")
        print(f"    en dark y cielo : {int(both.sum())}  (confirmados)")
        print(f"    solo en cielo   : {int(only_sky.sum())}  (intermitentes; el dark no los ve)")
        final = dark_mask | sky_mask
    else:
        final = dark_mask if dark_mask is not None else sky_mask

    if final is not None:
        print(f"\nMAPA FINAL: {int(final.sum())} pixeles a enmascarar "
              f"({100.0*final.sum()/final.size:.4f}% del sensor)")
        if args.save:
            np.save(args.save, final)
            print(f"guardado -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
