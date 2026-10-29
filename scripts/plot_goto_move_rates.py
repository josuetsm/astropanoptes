#!/usr/bin/env python3
"""Reconstruir las tasas de movimiento reales usadas en sesiones de GoTo.

A diferencia de tracking_samples.csv (que recien se empezo a guardar), los
diagnosticos de goto.py llevan guardandose desde fines de agosto en
stack_output/goto_diagnostics/<timestamp>_goto_<id>/timeline.jsonl. Cada MOVE
que de verdad se ejecuto en la montura real deja un evento
``goto_move_completed`` con los pasos comandados por eje y el delay_us usado,
que es exactamente lo que fija la tasa del motor: rate_steps_s = 1e6/delay_us.

Uso:
    python scripts/plot_goto_move_rates.py
    python scripts/plot_goto_move_rates.py --diagnostics-dir stack_output/goto_diagnostics --out informe/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAGNOSTICS_DIR = ROOT / "stack_output" / "goto_diagnostics"
DEFAULT_OUT_DIR = DEFAULT_DIAGNOSTICS_DIR / "plots"

# Sesiones reales distintas (dos noches de observacion, por ejemplo) no deben
# graficarse pegadas: mas separacion que esto entre dos MOVE es otra sesion.
_MAX_GAP_S = 3600.0


def _load_move_events(diagnostics_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(diagnostics_dir / "*_goto_*" / "timeline.jsonl"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("stage") != "goto_move_completed":
                        continue
                    steps = d.get("commanded_steps") or {}
                    delay = d.get("delay_us") or {}
                    az_steps = float(steps.get("az", 0.0))
                    alt_steps = float(steps.get("alt", 0.0))
                    delay_az = float(delay.get("az", 0.0)) or float("nan")
                    delay_alt = float(delay.get("alt", 0.0)) or float("nan")
                    rate_az = 1.0e6 / delay_az if np.isfinite(delay_az) and delay_az > 0 else float("nan")
                    rate_alt = 1.0e6 / delay_alt if np.isfinite(delay_alt) and delay_alt > 0 else float("nan")
                    max_rate = max(abs(rate_az) if np.isfinite(rate_az) else 0.0, abs(rate_alt) if np.isfinite(rate_alt) else 0.0)
                    duration_s = max(abs(az_steps), abs(alt_steps)) / max_rate if max_rate > 0 else 0.0
                    events.append(
                        {
                            "source": path,
                            "ts_unix": float(d.get("ts_unix", 0.0)),
                            "az_steps": az_steps,
                            "alt_steps": alt_steps,
                            "rate_az": rate_az,
                            "rate_alt": rate_alt,
                            "duration_s": duration_s,
                        }
                    )
        except Exception as exc:
            print(f"aviso: no se pudo leer {path}: {exc}", file=sys.stderr)
    events.sort(key=lambda e: e["ts_unix"])
    return events


def _split_into_sessions(events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    sessions: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    prev_t: Optional[float] = None
    for ev in events:
        t = ev["ts_unix"]
        if prev_t is not None and (t - prev_t) > _MAX_GAP_S:
            if current:
                sessions.append(current)
            current = []
        current.append(ev)
        prev_t = t
    if current:
        sessions.append(current)
    return sessions


def _plot_session(events: List[Dict[str, Any]], *, title: str, out_path: Path) -> None:
    times = [np.datetime64(int(ev["ts_unix"] * 1000), "ms") for ev in events]
    rate_az = np.array([ev["rate_az"] for ev in events])
    rate_alt = np.array([ev["rate_alt"] for ev in events])
    az_steps = np.array([ev["az_steps"] for ev in events])
    alt_steps = np.array([ev["alt_steps"] for ev in events])
    duration_s = np.array([ev["duration_s"] for ev in events])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    sizes = 20.0 + 4.0 * np.sqrt(np.abs(az_steps) + np.abs(alt_steps))
    ax.scatter(times, rate_az, s=sizes, color="tab:blue", alpha=0.7, label="tasa az")
    ax.scatter(times, rate_alt, s=sizes, color="tab:orange", alpha=0.5, marker="x", label="tasa alt")
    ax.set_ylabel("tasa real del MOVE\n(steps/s = 1e6/delay_us)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.bar(times, az_steps, width=0.0015, color="tab:blue", alpha=0.7, label="pasos az")
    ax.bar(times, alt_steps, width=0.0015, color="tab:orange", alpha=0.5, label="pasos alt")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel("pasos comandados\npor MOVE")
    ax.set_xlabel("hora real (UTC)")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    print(
        f"{out_path}: {len(events)} MOVE reales, "
        f"duracion total en movimiento={duration_s.sum():.1f}s, "
        f"tasa az min/max={np.nanmin(rate_az):.0f}/{np.nanmax(rate_az):.0f} steps/s"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    events = _load_move_events(args.diagnostics_dir)
    if not events:
        print("No se encontraron eventos goto_move_completed reales.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    sessions = _split_into_sessions(events)
    for idx, session in enumerate(sessions):
        t0 = np.datetime64(int(session[0]["ts_unix"] * 1000), "ms")
        suffix = f"_session{idx}" if len(sessions) > 1 else ""
        out_path = args.out / f"goto_move_rates{suffix}.png"
        _plot_session(session, title=f"Tasas de MOVE reales - sesion desde {t0}", out_path=out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
