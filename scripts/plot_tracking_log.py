#!/usr/bin/env python3
"""Graficar que tan bien sigue el tracking, desde tracking_samples.csv.

Por cada sesion de tracking (filas separadas por mas de 30 s son sesiones
distintas) se genera una figura con:

- el error de seguimiento en X/Y de la imagen, en segundos de arco: es
  ``error_x_px``/``error_y_px``, lo que el lazo intenta llevar a cero;
- la velocidad de cada eje: feedforward sideral, comandada total y la que de
  verdad llego a la montura (``move_steps`` entre filas / dt);
- la nube de error X-Y, el histograma de |error| y un resumen numerico.

Las metricas usan solo mediciones validas (``ok=1``) y descartan los primeros
``--settle-s`` segundos, que son la convergencia inicial.

Uso:
    python scripts/plot_tracking_log.py
    python scripts/plot_tracking_log.py --min-duration 120 --out informe/
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "stack_output" / "tracking_logs"
# Fuera del directorio de logs: si los PNG cayeran ahi, el glob por defecto
# (tracking_samples.csv*) los tomaria como otra sesion.
DEFAULT_OUT_DIR = DEFAULT_LOG_DIR / "plots"
DEFAULT_SCALE = 206.265 * 2.9 / 900.0  # "/px: pixel 2.9 um, focal 900 mm

_MAX_GAP_S = 30.0
_LOCAL_TZ = dt.timezone(dt.timedelta(hours=-3))


def _f(row: Dict[str, Any], key: str) -> float:
    try:
        s = str(row.get(key, "")).strip()
        return float(s) if s else float("nan")
    except ValueError:
        return float("nan")


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f) if row]


def _find_default_logs() -> List[Path]:
    if not DEFAULT_LOG_DIR.exists():
        return []
    return [p for p in sorted(DEFAULT_LOG_DIR.glob("tracking_samples.csv*")) if p.is_file()]


def _split_into_sessions(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    sessions: List[List[Dict[str, Any]]] = []
    prev_t: Optional[float] = None
    for row in rows:
        t = _f(row, "ts_unix")
        if not np.isfinite(t):
            continue
        if prev_t is None or (t - prev_t) > _MAX_GAP_S:
            sessions.append([])
        sessions[-1].append(row)
        prev_t = t
    return sessions


def _delivered_rate(rows: List[Dict[str, Any]], t: np.ndarray, key: str) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    if key not in rows[0]:
        return out
    for i in range(1, len(rows)):
        d = t[i] - t[i - 1]
        if d > 0:
            out[i] = _f(rows[i], key) / d
    return out


def _rolling(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = w.mean()
    return out


def _metrics(ex: np.ndarray, ey: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    r = np.hypot(ex, ey)[valid]
    return {
        "n": float(valid.sum()),
        "rms_x": float(np.sqrt(np.mean(ex[valid] ** 2))) if valid.any() else float("nan"),
        "rms_y": float(np.sqrt(np.mean(ey[valid] ** 2))) if valid.any() else float("nan"),
        "rms": float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan"),
        "p50": float(np.percentile(r, 50)) if r.size else float("nan"),
        "p90": float(np.percentile(r, 90)) if r.size else float("nan"),
        "within_2": float(np.mean(r <= 2.0)) if r.size else float("nan"),
        "within_5": float(np.mean(r <= 5.0)) if r.size else float("nan"),
    }


def _convergence_s(t: np.ndarray, r: np.ndarray, ok: np.ndarray, *, window_s: float = 30.0, thresh: float = 5.0) -> float:
    """Primer instante en que la mediana movil de |error| baja de ``thresh``."""
    for i in range(len(t)):
        w = ok & (t > t[i] - window_s) & (t <= t[i])
        if t[i] >= window_s and w.sum() >= 5 and np.median(r[w]) < thresh:
            return float(t[i])
    return float(t[-1])


def _session_arrays(rows: List[Dict[str, Any]], scale: float) -> Dict[str, np.ndarray]:
    t_abs = np.array([_f(r, "ts_unix") for r in rows])
    ex = np.array([_f(r, "error_x_px") for r in rows]) * scale
    ey = np.array([_f(r, "error_y_px") for r in rows]) * scale
    return {
        "t_abs": t_abs,
        "t": t_abs - t_abs[0],
        "ok": np.array([str(r.get("ok", "")).strip() == "1" for r in rows]),
        "ex": ex,
        "ey": ey,
        "r": np.hypot(ex, ey),
    }


def _plot_session(rows: List[Dict[str, Any]], *, scale: float, settle_s: Optional[float], out_path: Path) -> Dict[str, Any]:
    a = _session_arrays(rows, scale)
    t_abs, t, ok, ex, ey = a["t_abs"], a["t"], a["ok"], a["ex"], a["ey"]
    tm = t / 60.0
    converged = True
    if settle_s is None:
        settle_s = _convergence_s(t, a["r"], ok)
        converged = settle_s < t[-1]
    settled = ok & (t >= settle_s) if converged else ok
    m = _metrics(ex, ey, settled)

    rates = {}
    for axis in ("az", "alt"):
        rates[axis] = {
            "ff": np.array([_f(r, f"rate_ff_{axis}") for r in rows]),
            "cmd": np.array([_f(r, f"rate_cmd_{axis}") for r in rows]),
            "real": _delivered_rate(rows, t, f"move_steps_{axis}"),
        }
    cmd_tot = {a: np.nansum(np.abs(rates[a]["cmd"][1:] * np.diff(t))) for a in rates}
    real_tot = {a: np.nansum(np.abs([_f(r, f"move_steps_{a}") for r in rows[1:]])) for a in rates}

    start_local = dt.datetime.fromtimestamp(t_abs[0], _LOCAL_TZ)
    alts = np.array([_f(r, "pointing_alt_deg") for r in rows])
    azs = np.array([_f(r, "pointing_az_deg") for r in rows])
    where = (
        f"az {np.nanmedian(azs):.0f}° alt {np.nanmedian(alts):.0f}°"
        if np.isfinite(alts).any() and np.isfinite(azs).any()
        else "apuntado desconocido"
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 2.2, 1.6], hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.axhspan(-2, 2, color="#2ca02c", alpha=0.08, label="±2\"")
    ax.plot(tm, np.where(ok, ex, np.nan), lw=0.9, color="tab:blue", label="error X imagen")
    ax.plot(tm, np.where(ok, ey, np.nan), lw=0.9, color="tab:orange", label="error Y imagen")
    bad = ~ok
    if bad.any():
        ax.plot(tm[bad], np.zeros(bad.sum()), "|", color="tab:red", ms=10, label="medicion invalida")
    if converged and settle_s > 0:
        ax.axvline(settle_s / 60.0, color="grey", ls=":", lw=1)
    lim = max(5.0, np.nanpercentile(np.abs(np.r_[ex[ok], ey[ok]]), 99) * 1.2) if ok.any() else 10.0
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("error (\")")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.set_title(
        f"Tracking {start_local:%Y-%m-%d %H:%M} local — {t[-1]/60:.1f} min, {where} — "
        f"{'tras converger' if converged else 'sin converger'}: RMS {m['rms']:.2f}\", mediana {m['p50']:.2f}\"",
        fontsize=11,
    )

    for row_i, axis in ((1, "az"), (2, "alt")):
        ax = fig.add_subplot(gs[row_i, 0:2])
        rr = rates[axis]
        ax.plot(tm, rr["ff"], "--", color="grey", lw=1.2, label="feedforward sideral")
        ax.plot(tm, rr["cmd"], color="tab:blue" if axis == "az" else "tab:orange", lw=1.0, label="comandada")
        ax.plot(tm, rr["real"], ".", ms=2.5, color="black", alpha=0.35, label="entregada (por fila)")
        ax.plot(tm, _rolling(rr["real"], 10), color="black", lw=1.2, label="entregada (media 10 s)")
        ax.set_ylabel(f"vel. {axis} (pasos/s)")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ratio = real_tot[axis] / cmd_tot[axis] if cmd_tot[axis] > 0 else float("nan")
        ax.text(0.01, 0.92, f"entregado/comandado: {ratio:.1%}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        if row_i == 2:
            ax.set_xlabel("minutos desde el inicio")

    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(ex[settled], ey[settled], c=tm[settled], s=6, cmap="viridis")
    for radius, style in ((m["p50"], "-"), (m["p90"], "--")):
        if np.isfinite(radius):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls=style, color="tab:red"))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("X (\")")
    ax.set_ylabel("Y (\")")
    ax.set_title("nube de error (circulos: p50 y p90)", fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="min")

    ax = fig.add_subplot(gs[1, 2])
    r = np.hypot(ex, ey)[settled]
    if r.size:
        ax.hist(r, bins=40, color="tab:purple", alpha=0.8)
        for q, style in ((m["p50"], "-"), (m["p90"], "--")):
            ax.axvline(q, color="tab:red", ls=style)
    ax.set_xlabel("|error| (\")")
    ax.set_ylabel("muestras")
    ax.set_title("distribucion de |error|", fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    lines = [
        f"converge a los {settle_s:.0f} s (linea punteada)" if converged else "NO converge: metricas sobre toda la sesion",
        f"muestras validas desde ahi: {int(m['n'])} de {len(rows)}",
        f"mediciones invalidas: {bad.sum()} ({bad.mean():.0%})",
        "",
        f"RMS total: {m['rms']:.2f}\"",
        f"RMS X / Y: {m['rms_x']:.2f}\" / {m['rms_y']:.2f}\"",
        f"mediana |error|: {m['p50']:.2f}\"",
        f"p90 |error|: {m['p90']:.2f}\"",
        f"tiempo dentro de 2\": {m['within_2']:.0%}",
        f"tiempo dentro de 5\": {m['within_5']:.0%}",
        "",
        f"escala: {scale:.3f}\"/px",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9.5)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "start": start_local,
        "minutes": t[-1] / 60.0,
        "settle_s": float(settle_s) if converged else float("nan"),
        "arrays": a,
        "bad_frac": float(bad.mean()),
        "ratio_az": real_tot["az"] / cmd_tot["az"] if cmd_tot["az"] > 0 else float("nan"),
        "ratio_alt": real_tot["alt"] / cmd_tot["alt"] if cmd_tot["alt"] > 0 else float("nan"),
        **m,
    }


def _plot_comparison(summaries: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(summaries):
        a = s["arrays"]
        r = np.where(a["ok"], a["r"], np.nan)
        # mediana movil de 15 s: deja ver la tendencia sin el ruido por fila
        smooth = np.array([
            np.nanmedian(r[(a["t"] > ti - 15) & (a["t"] <= ti)]) if np.isfinite(r[(a["t"] > ti - 15) & (a["t"] <= ti)]).any() else np.nan
            for ti in a["t"]
        ])
        label = f"{s['start']:%d-%m %H:%M} ({s['minutes']:.0f} min) — mediana {s['p50']:.1f}\""
        ax.plot(a["t"] / 60.0, smooth, color=cmap(i), lw=1.6, label=label)
    ax.set_yscale("log")
    ax.axhspan(0.1, 2.0, color="#2ca02c", alpha=0.08)
    ax.axhline(2.7, color="grey", ls=":", lw=1)
    ax.text(0.99, 2.9, "seeing medido (~2.7\")", color="grey", fontsize=8, ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylim(0.3, 2000)
    ax.set_xlabel("minutos desde el inicio de la sesion")
    ax.set_ylabel("|error| (\"), mediana movil 15 s")
    ax.set_title("Calidad del tracking por sesion (escala log)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="*", type=Path, help="por defecto, todos los de stack_output/tracking_logs/")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-duration", type=float, default=60.0, help="ignora sesiones mas cortas (s)")
    parser.add_argument("--settle-s", type=float, default=None,
                        help="segundos excluidos de las metricas (por defecto: hasta que converge)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="segundos de arco por pixel")
    parser.add_argument("--since", type=str, default=None, help="solo sesiones desde esta fecha local (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    since = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=_LOCAL_TZ) if args.since else None

    csv_files = args.csv_files or _find_default_logs()
    if not csv_files:
        print("No se encontraron logs de tracking.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in csv_files:
        for session in _split_into_sessions(_read_rows(path)):
            dur = _f(session[-1], "ts_unix") - _f(session[0], "ts_unix")
            if dur < args.min_duration:
                continue
            start = dt.datetime.fromtimestamp(_f(session[0], "ts_unix"), _LOCAL_TZ)
            if since is not None and start < since:
                continue
            out_path = args.out / f"tracking_{start:%Y%m%d_%H%M%S}.png"
            s = _plot_session(session, scale=args.scale, settle_s=args.settle_s, out_path=out_path)
            s["png"] = out_path
            summaries.append(s)

    if not summaries:
        print(f"Ninguna sesion dura al menos {args.min_duration:.0f} s.", file=sys.stderr)
        return 1
    comparison = args.out / "tracking_comparacion.png"
    _plot_comparison(summaries, comparison)
    print(f"{'inicio (local)':17s} {'min':>5s} {'conv s':>6s} {'RMS':>6s} {'RMS X':>6s} {'RMS Y':>6s} {'p50':>6s} {'p90':>6s} {'<2\"':>5s} {'<5\"':>5s} {'inval':>6s} {'entr/cmd az':>11s} {'alt':>6s}")
    for s in summaries:
        print(
            f"{s['start']:%Y-%m-%d %H:%M}  {s['minutes']:5.1f} {('no' if not np.isfinite(s['settle_s']) else f"{s['settle_s']:.0f}"):>6s} {s['rms']:6.2f} {s['rms_x']:6.2f} {s['rms_y']:6.2f} "
            f"{s['p50']:6.2f} {s['p90']:6.2f} {s['within_2']:5.0%} {s['within_5']:5.0%} {s['bad_frac']:6.0%} "
            f"{s['ratio_az']:11.0%} {s['ratio_alt']:6.0%}"
        )
    for s in summaries:
        print(s["png"])
    print(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
