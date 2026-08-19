#!/usr/bin/env python3
"""Asistente de enfoque en vivo para objetos planetarios.

Captura el preview de una sesión ya abierta (vía ``python app.py attach``) y
muestra una métrica de nitidez actualizada, para poder ajustar el foco
observando si el número sube o baja.

La métrica es la nitidez de borde: la mediana del gradiente sobre los píxeles
del borde del disco, normalizada por el contraste del objeto. A diferencia de
la varianza del laplaciano global, no depende del brillo total, de modo que
sigue siendo comparable aunque cambien exposición o ganancia mientras se
enfoca.

Uso:
    python scripts/focus_assist.py            # hasta Ctrl-C
    python scripts/focus_assist.py --seconds 120
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
IMG = ROOT / "terminal_output" / "images" / "_focus.jpg"


def grab() -> np.ndarray | None:
    subprocess.run(
        [sys.executable, str(ROOT / "app.py"), "attach", "-c", "image live _focus"],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    img = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)
    return img


def sharpness(img: np.ndarray) -> tuple[float, float, tuple[int, int]]:
    """Devuelve (nitidez_borde, diametro_px, centro)."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (21, 21), 0)
    _, peak, _, loc = cv2.minMaxLoc(blur)
    bg = float(np.median(f))
    # Recorte generoso alrededor del objeto.
    r = 260
    x, y = loc
    win = f[max(0, y - r): y + r, max(0, x - r): x + r]
    if win.size == 0:
        return 0.0, 0.0, loc

    contrast = float(win.max() - bg)
    if contrast <= 1.0:
        return 0.0, 0.0, loc

    # Borde del disco = píxeles cerca de media altura.
    half = bg + contrast * 0.5
    band = np.abs(win - half) < (contrast * 0.12)
    if int(band.sum()) < 20:
        return 0.0, 0.0, loc

    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    edge = float(np.median(grad[band])) / contrast * 100.0

    mask = win > half
    ys, xs = np.nonzero(mask)
    diam = float(max(xs.max() - xs.min(), ys.max() - ys.min())) if len(xs) else 0.0
    return edge, diam, loc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=0.0, help="duración; 0 = hasta Ctrl-C")
    args = ap.parse_args()

    print("Asistente de enfoque. Gira el foco despacio y observa la NITIDEZ.")
    print("Sube = mejor foco. El diametro deberia ACHICARSE al enfocar.\n")
    print(f"{'nitidez':>9}  {'diam.px':>8}  barra")

    best = 0.0
    t0 = time.time()
    try:
        while True:
            img = grab()
            if img is None:
                time.sleep(0.5)
                continue
            edge, diam, _ = sharpness(img)
            best = max(best, edge)
            bar = "#" * int(min(50, edge / max(best, 1e-6) * 50))
            flag = "  <-- MEJOR" if edge >= best - 1e-9 and edge > 0 else ""
            print(f"{edge:9.2f}  {diam:8.0f}  {bar}{flag}")
            if args.seconds and (time.time() - t0) > args.seconds:
                break
    except KeyboardInterrupt:
        print("\nlisto.")
    print(f"\nmejor nitidez observada: {best:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
