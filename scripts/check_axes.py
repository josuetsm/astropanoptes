#!/usr/bin/env python3
"""Comprueba que cada eje de la montura mueve de verdad el campo.

Un TMC2209 en STEP/DIR no tiene realimentacion: el firmware emite los pulsos y
responde OK aunque al otro lado no haya motor. Un cable suelto, un driver sin
corriente o un Vref a cero son indistinguibles de un eje sano desde el software,
y se comen la noche entera sin que nada lo delate -- la app sigue informando que
los movimientos se completan.

La unica prueba real es mirar el cielo: si el eje se mueve, el campo se desplaza.
Esto manda un movimiento conocido a cada eje y mide cuanto se corrio la imagen con
el mismo alineador que usa el tracking.

Pero el cielo se mueve solo: sin seguimiento, la deriva sideral corre el campo
22 px/s a x1 y 90 px/s a x5. En los segundos que dura la prueba eso puede superar
al propio movimiento comandado, y un eje muerto pareceria vivo. Por eso se mide
primero una referencia -- el mismo tiempo, sin mover nada -- y cada eje se juzga
por cuanto se desplazo *de mas* respecto a esa deriva.

Correrlo al empezar la sesion cuesta medio minuto.

Uso:
    python scripts/check_axes.py                  # con la GUI abierta
    python scripts/check_axes.py --steps 600
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_server import DEFAULT_SOCKET_PATH, RESPONSE_TERMINATOR  # noqa: E402
from raw_alignment import (  # noqa: E402
    build_raw_alignment_signature,
    estimate_raw_translation,
)


# Fraccion del desplazamiento esperado por debajo de la cual se considera que el
# eje no responde. Generosa a proposito: la escala configurada puede estar mal y
# el juego mecanico se come parte del primer movimiento. Lo que se quiere separar
# es "se mueve" de "no se mueve en absoluto", no medir la calibracion.
DEAD_FRACTION = 0.25
# Por encima de esto el eje se mueve, pero mucho menos de lo previsto: casi
# siempre significa que la escala optica configurada no es la que hay puesta.
SUSPECT_FRACTION = 0.60
# Por encima de esto el eje mueve mucho MAS de lo previsto, que casi siempre
# significa lo mismo al reves: la escala configurada es mas gruesa que la real
# -- tipicamente el barlow puesto no es el seleccionado en la pestana Observador.
# Ese error es tan silencioso como un cable suelto: el plate solving busca a una
# escala equivocada y falla sin decir por que.
SCALE_FRACTION = 2.0


class Console:
    """Conexion al socket de control de la app abierta."""

    def __init__(self, path: Path) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(path))
        self.stream = self.sock.makefile("rwb")

    def cmd(self, line: str, timeout: float = 180.0) -> str:
        self.sock.settimeout(timeout)
        self.stream.write((line + "\n").encode())
        self.stream.flush()
        out = []
        while True:
            raw = self.stream.readline()
            if not raw:
                break
            text = raw.decode(errors="replace").rstrip("\r\n")
            if text == RESPONSE_TERMINATOR:
                break
            out.append(text)
        return "\n".join(out)

    def close(self) -> None:
        try:
            self.stream.close()
            self.sock.close()
        except OSError:
            pass


@dataclass
class AxisResult:
    axis: str
    steps: int
    expected_px: float
    measured_px: float
    ok_measure: bool
    reason: str
    drift_px: float = 0.0

    @property
    def excess_px(self) -> float:
        """Desplazamiento atribuible al motor, descontada la deriva del cielo."""
        return max(0.0, self.measured_px - self.drift_px)

    @property
    def fraction(self) -> float:
        if self.expected_px <= 0.0:
            return 0.0
        return self.excess_px / self.expected_px

    @property
    def verdict(self) -> str:
        if not self.ok_measure:
            return "SIN MEDIDA"
        if self.fraction < DEAD_FRACTION:
            return "NO RESPONDE"
        if self.fraction < SUSPECT_FRACTION:
            return "SOSPECHOSO"
        if self.fraction > SCALE_FRACTION:
            return "ESCALA MAL"
        return "OK"


def grab(console: Console, tag: str = "_axischeck") -> Optional[np.ndarray]:
    """Frame RAW16 actual, promediado en el tiempo.

    Se graba RAW en vez de leer el JPEG del preview: el preview va estirado y en
    8 bits, y el alineador -- el mismo del tracking -- espera RAW16. Con el
    preview la correlacion sale ambigua aunque el campo se haya movido.
    """
    console.cmd(f"camera record 1 raw_output {tag}", timeout=60)
    console.cmd("await record 60", timeout=70)
    path = ROOT / "raw_output" / f"{tag}.npy"
    try:
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    except (OSError, ValueError):
        return None
    frame = np.median(arr, axis=0) if arr.ndim == 3 else arr
    return np.ascontiguousarray(frame.astype(np.uint16))


def measure_shift(before: np.ndarray, after: np.ndarray, radius_px: float,
                  expected_px: float = 0.0):
    _ = expected_px
    """Desplazamiento entre dos frames, en pixeles.

    Se desactiva la validacion cruzada entre el perfil grueso y el fino. En el
    tracking tiene todo el sentido: un enganche erroneo manda la montura a
    perseguir algo que no es, y ahi conviene descartar la medida antes que
    arriesgarse. Aqui el salto es grande y deliberado, el perfil fino deja de
    localizar a esa escala, y exigir que ambos coincidan descarta medidas
    perfectamente buenas -- se comprobo con un movimiento conocido de 328 px que
    el alineador media 325.6 px y aun asi lo marcaba como ambiguo.

    Lo que sustituye a esa validacion es mas directo: el resultado se compara
    con el desplazamiento esperado y con la deriva medida sin mover. Una medida
    disparatada no puede pasar por buena.
    """
    ref = build_raw_alignment_signature(before, median_k=3, smooth_k=30)
    cur = build_raw_alignment_signature(after, median_k=3, smooth_k=30)
    result = estimate_raw_translation(
        ref,
        cur,
        search_radius_px=float(radius_px),
        max_displacement_px=float(radius_px) * 1.5,
        # Umbral de confianza bajo a proposito: no hace falta precision, solo
        # saber si el campo se movio. El contraste contra la deriva medida y
        # contra el desplazamiento esperado es lo que descarta una medida mala.
        min_response=0.08,
        max_profile_disagreement_px=float("inf"),
    )
    if not result.ok:
        return 0.0, False, str(result.reason)
    return float(math.hypot(result.dx, result.dy)), True, "ok"


# Fraccion del alto del sensor que debe desplazarse el campo en la prueba. Ni
# tan poco que se confunda con el juego mecanico, ni tanto que no quede solape
# entre las dos imagenes -- sin solape no hay nada que correlacionar.
TARGET_SHIFT_FRACTION = 0.30
# Suelo en pasos: por debajo, el juego del tren se come el movimiento entero y un
# eje sano parece muerto. A escalas finas manda este suelo, no la fraccion.
MIN_TEST_STEPS = 25
# Pasos que se mandan y se descartan antes de medir, para dejar el juego tomado.
BACKLASH_TAKEUP_STEPS = 40


def test_steps_for_scale(
    arcsec_per_px: float, deg_per_step: float, frame_h: int = 1096
) -> int:
    """Cuantos pasos mover para que el campo se corra una fraccion util."""
    target_px = TARGET_SHIFT_FRACTION * float(frame_h)
    arcsec = target_px * float(arcsec_per_px)
    steps = arcsec / (float(deg_per_step) * 3600.0)
    return int(max(MIN_TEST_STEPS, min(2000, round(steps))))


def measure_drift(console: Console, seconds: float, radius_px: float) -> float:
    """Deriva del cielo en px/s, sin mover la montura.

    Devuelve una *tasa*, no un desplazamiento: cada prueba de eje dura lo que
    dura su propio movimiento, que depende de los pasos y del retardo y puede
    ser mucho mas larga que esta sonda. Restar un desplazamiento fijo medido en
    otra ventana descuenta de menos y le acredita al motor una deriva que no
    movio el, que es como un eje muerto llegaba a puntuar "OK".
    """
    before = grab(console, "_axisdrift")
    t0 = time.monotonic()
    time.sleep(max(0.0, seconds))
    after = grab(console, "_axisdrift")
    elapsed = max(1e-6, time.monotonic() - t0)
    if before is None or after is None:
        return 0.0
    measured, ok, _reason = measure_shift(before, after, radius_px, expected_px=radius_px)
    return (measured / elapsed) if ok else 0.0


def check_axis(
    console: Console,
    axis: str,
    steps: int,
    delay_us: int,
    arcsec_per_px: float,
    deg_per_step: float,
    settle_s: float,
    drift_rate_px_s: float = 0.0,
    cos_alt: float = 1.0,
) -> AxisResult:
    # En alt-az, un movimiento de azimut desplaza el campo Delta_az * cos(alt),
    # no Delta_az: cerca del cenit el mismo numero de pasos mueve mucho menos
    # cielo. Sin esta correccion un eje de azimut perfectamente sano puntua bajo
    # a altura alta, y a partir de ~75 grados daria "no responde".
    expected_px = (deg_per_step * steps * 3600.0 * cos_alt) / max(arcsec_per_px, 1e-9)

    # Tomar el juego primero, en el mismo sentido que la medida: si no, el
    # backlash se come parte del movimiento y un eje sano puntua bajo.
    console.cmd(f"mount move {axis} 1 {BACKLASH_TAKEUP_STEPS} {delay_us} direct")
    console.cmd("await mount 200", timeout=210)
    time.sleep(settle_s)

    # La deriva se descuenta sobre el tiempo real transcurrido entre estas dos
    # tomas, que es la ventana en la que el cielo pudo correrse.
    before = grab(console)
    t_before = time.monotonic()
    console.cmd(f"mount move {axis} 1 {steps} {delay_us} direct")
    console.cmd("await mount 200", timeout=210)
    time.sleep(settle_s)
    after = grab(console)
    drift_px = max(0.0, float(drift_rate_px_s)) * max(0.0, time.monotonic() - t_before)

    # Volver siempre, haya salido lo que haya salido: la comprobacion no debe
    # dejar el telescopio apuntando a otro sitio.
    console.cmd(f"mount move {axis} -1 {steps + BACKLASH_TAKEUP_STEPS} {delay_us} direct")
    console.cmd("await mount 200", timeout=210)

    if before is None or after is None:
        return AxisResult(axis, steps, expected_px, 0.0, False, "sin_imagen", 0.0)

    radius = min(expected_px * 1.6 + 20.0, 0.9 * float(before.shape[0]))
    measured, ok, reason = measure_shift(before, after, radius_px=radius,
                                         expected_px=expected_px)
    return AxisResult(axis, steps, expected_px, measured, ok, reason, drift_px)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)
    parser.add_argument("--steps", type=int, default=None,
                        help="pasos de prueba; por defecto se calculan de la escala")
    parser.add_argument("--delay-us", type=int, default=1800)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--alt-deg", type=float, default=None,
                        help="altitud aproximada de apuntado (afecta solo al azimut)")
    args = parser.parse_args()

    try:
        console = Console(Path(args.socket).expanduser())
    except OSError:
        print(f"No hay ninguna app abierta en {args.socket}; abre la GUI primero.")
        return 2

    try:
        if console.cmd("get mount.connected").strip().lower() != "true":
            print("La montura no esta conectada.")
            return 2
        if console.cmd("get camera.connected").strip().lower() != "true":
            print("La camara no esta conectada: sin imagen no hay forma de saber "
                  "si los ejes mueven algo.")
            return 2

        # La escala vive en la configuracion, no en el estado publicado.
        import json

        ps = json.loads(console.cmd("config platesolving"))
        focal_m = float(ps["focal_m"])
        pixel_m = float(ps["pixel_size_m"])
        arcsec_per_px = math.degrees(pixel_m / focal_m) * 3600.0

        from ap_types import Axis
        from goto import MountKinematics

        kin = MountKinematics()
        print(f"escala configurada: {arcsec_per_px:.3f} \"/px  (focal {focal_m:.2f} m)")
        print("prueba: movimiento de ida y vuelta por eje, "
              "dimensionado para correr el campo ~"
              f"{TARGET_SHIFT_FRACTION*100:.0f}% del sensor\n")

        # Tasa de deriva. Cada eje descuenta despues segun lo que dure su
        # propia prueba, no segun lo que duro esta sonda.
        probe_seconds = 2.0 * args.settle + 2.5
        drift_rate = measure_drift(console, probe_seconds, radius_px=500.0)
        print(f"deriva del cielo: {drift_rate:.1f} px/s "
              f"(sonda de {probe_seconds:.1f} s)\n")

        # Altitud de apuntado: solo afecta al eje de azimut.
        alt_deg = args.alt_deg
        source = "indicada"
        if alt_deg is None:
            state_alt = console.cmd("get goto.pointing_alt_deg").strip()
            valid = console.cmd("get goto.pointing_valid").strip().lower() == "true"
            try:
                alt_deg, source = (float(state_alt), "del modelo") if valid else (45.0, "supuesta")
            except ValueError:
                alt_deg, source = 45.0, "supuesta"
        cos_alt = max(0.05, math.cos(math.radians(float(alt_deg))))
        print(f"altitud {source}: {float(alt_deg):.1f} deg  ->  el azimut mueve "
              f"cos(alt) = {cos_alt:.2f} del cielo por paso")
        if float(alt_deg) > 70.0:
            print("  (aviso: cerca del cenit la prueba de azimut pierde sensibilidad)")

        results = []
        for axis, enum in (("az", Axis.AZ), ("alt", Axis.ALT)):
            dps = float(kin.deg_per_step(enum))
            steps = args.steps or test_steps_for_scale(arcsec_per_px, dps)
            res = check_axis(
                console,
                axis,
                steps,
                args.delay_us,
                arcsec_per_px,
                dps,
                args.settle,
                drift_rate,
                cos_alt if axis == "az" else 1.0,
            )
            results.append(res)
            print(
                f"  {axis.upper():4s} {steps:4d} pasos   esperado {res.expected_px:7.1f} px   "
                f"medido {res.measured_px:7.1f} px   "
                f"neto {res.excess_px:7.1f} px ({res.fraction*100:5.1f}%)   {res.verdict}"
                + ("" if res.ok_measure else f"  [{res.reason}]")
            )

        print()
        dead = [r for r in results if r.verdict == "NO RESPONDE"]
        blind = [r for r in results if r.verdict == "SIN MEDIDA"]
        weak = [r for r in results if r.verdict == "SOSPECHOSO"]
        scale = [r for r in results if r.verdict == "ESCALA MAL"]
        if dead:
            for r in dead:
                print(f"El eje {r.axis.upper()} recibe los pulsos pero no mueve nada.")
            print("Revisa el cable del motor, que el driver tenga corriente y el Vref.")
            return 1
        if blind:
            print("No se pudo medir el desplazamiento: hace falta un campo con algo "
                  "que seguir. Apunta a una zona con estrellas y repite.")
            return 3
        if scale:
            factor = float(np.median([r.fraction for r in scale]))
            print(f"Los ejes mueven ~{factor:.1f}x mas cielo del previsto: la escala "
                  "optica configurada no es la que hay montada.")
            print(f"En la pestana Observador, la focal efectiva deberia ser unas "
                  f"{factor:.1f} veces la actual (¿barlow correcto?).")
            return 1
        if weak:
            for r in weak:
                print(f"El eje {r.axis.upper()} mueve, pero {r.fraction*100:.0f}% de lo "
                      "previsto: casi siempre es la escala optica mal configurada "
                      "(¿barlow correcto en la pestana Observador?).")
            return 1
        print("Los dos ejes mueven el campo como se espera.")
        return 0
    finally:
        console.close()
        for leftover in (ROOT / "raw_output").glob("_axis*.npy"):
            leftover.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
