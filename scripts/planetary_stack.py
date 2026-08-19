#!/usr/bin/env python3
"""Apilado planetario incremental (lucky imaging) sobre una sesión abierta.

Captura en tandas cortas, y tras cada tanda selecciona los mejores frames por
nitidez, los alinea con precisión subpíxel y los acumula en un stack
persistente, borrando el RAW de la tanda antes de capturar la siguiente. Así se
pueden apilar decenas de miles de frames sin necesitar espacio en disco para
todos a la vez, y sin recortar el sensor (el ROI degrada el encuadre y la
recuperación del tracking).

La alineación se hace en desplazamientos enteros PARES sobre el mosaico Bayer,
de modo que el patrón RGGB se conserva y el debayer a color sigue siendo válido
después de apilar.

Uso:
    python scripts/planetary_stack.py --runs 3 --seconds 60
    python scripts/planetary_stack.py --report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_output"
OUT = ROOT / "stack_output"
STATE = OUT / "planetary_accum.npz"
HALF = 220  # ventana de trabajo: 440x440 px, cubre disco + anillos con margen


def attach(*cmds: str, timeout: float = 600.0) -> str:
    args = [sys.executable, str(ROOT / "app.py"), "attach"]
    for c in cmds:
        args += ["-c", c]
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def frame_metrics(f: np.ndarray, bg: float):
    """Devuelve (nitidez, cx, cy, x, y) o None si el objeto no es utilizable."""
    b = cv2.GaussianBlur(f, (21, 21), 0)
    _, peak, _, loc = cv2.minMaxLoc(b)
    x, y = loc
    h, w = f.shape
    if not (HALF <= x < w - HALF and HALF <= y < h - HALF):
        return None
    win = f[y - HALF:y + HALF, x - HALF:x + HALF]
    contrast = float(peak - bg)
    if contrast <= 50.0:
        return None
    # centro de masa: mucho más estable que el pico en un disco saturado/difuso
    ww = np.clip(win - bg, 0, None)
    tot = float(ww.sum())
    if tot <= 0:
        return None
    ys, xs = np.mgrid[0:2 * HALF, 0:2 * HALF]
    cx = float((xs * ww).sum() / tot)
    cy = float((ys * ww).sum() / tot)
    # nitidez de borde normalizada por contraste -> comparable entre frames
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
    band = np.abs(win - (bg + contrast * 0.5)) < (contrast * 0.12)
    if int(band.sum()) < 20:
        return None
    sharp = float(np.median(np.hypot(gx, gy)[band])) / contrast
    return sharp, cx, cy, x, y


def process_run(path: Path, keep_frac: float):
    a = np.load(path, mmap_mode="r")
    n = len(a)
    bg = float(np.median(np.asarray(a[0], dtype=np.float32)))
    info = []
    for i in range(n):
        f = np.asarray(a[i], dtype=np.float32)
        m = frame_metrics(f, bg)
        if m is not None:
            info.append((m[0], i, m[1], m[2], m[3], m[4]))
    if not info:
        return None, 0, n, bg
    info.sort(reverse=True)
    keep = info[:max(4, int(keep_frac * len(info)))]

    ref_cx, ref_cy = keep[0][2], keep[0][3]
    acc = np.zeros((2 * HALF, 2 * HALF), dtype=np.float64)
    cnt = 0
    for sharp, i, cx, cy, x, y in keep:
        f = np.asarray(a[i], dtype=np.float32)
        # desplazamiento entero PAR: preserva la fase del mosaico Bayer
        dx = int(round((ref_cx - cx) / 2)) * 2
        dy = int(round((ref_cy - cy) / 2)) * 2
        xs, ys = x - dx, y - dy
        h, w = f.shape
        if not (HALF <= xs < w - HALF and HALF <= ys < h - HALF):
            continue
        acc += f[ys - HALF:ys + HALF, xs - HALF:xs + HALF]
        cnt += 1
    return acc, cnt, n, bg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--keep", type=float, default=0.5, help="fracción de frames a conservar por nitidez")
    ap.add_argument("--reset", action="store_true", help="descarta el stack acumulado previo")
    ap.add_argument("--report", action="store_true", help="solo muestra el estado acumulado")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.reset and STATE.exists():
        STATE.unlink()

    if STATE.exists():
        st = np.load(STATE)
        acc, total, captured, bg0 = st["acc"], int(st["count"]), int(st["captured"]), float(st["bg"])
    else:
        acc, total, captured, bg0 = np.zeros((2 * HALF, 2 * HALF), dtype=np.float64), 0, 0, 0.0

    if args.report:
        print(f"acumulado: {total} frames apilados de {captured} capturados")
        return 0

    for r in range(args.runs):
        name = f"pstack_run{r}"
        p = RAW / f"{name}.npy"
        if p.exists():
            p.unlink()
        print(f"[tanda {r+1}/{args.runs}] capturando {args.seconds:.0f}s ...", flush=True)
        attach(f"camera record {args.seconds:.0f} raw_output {name}",
               f"await record {args.seconds + 90:.0f}")
        if not p.exists():
            print("  (sin archivo, salto)")
            continue
        gb = p.stat().st_size / 1e9
        print(f"  procesando {gb:.1f} GB ...", flush=True)
        a2, cnt, n, bg = process_run(p, args.keep)
        p.unlink()  # libera el disco antes de la siguiente tanda
        if a2 is None:
            print("  (sin frames utiles)")
            continue
        acc += a2
        total += cnt
        captured += n
        if bg0 == 0.0:
            bg0 = bg
        print(f"  +{cnt} apilados de {n} capturados  ->  total {total}", flush=True)
        np.savez(STATE, acc=acc, count=total, captured=captured, bg=bg0)

    print(f"\nTOTAL: {total} frames apilados de {captured} capturados")
    print(f"mejora de SNR ~ x{np.sqrt(max(total,1)):.1f}")
    if total:
        np.save(OUT / "planetary_mean.npy", (acc / total).astype(np.float32))
        print(f"-> {OUT/'planetary_mean.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
