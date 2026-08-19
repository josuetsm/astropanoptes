# transmission_error.py
"""Aprender el error de transmisión cicloidal desde el tracking.

El modelo de apuntado ya representa el error de transmisión como un primer
armónico por eje (``GoToModel.periodic_coeff_deg``), pero para ajustarlo exige
al menos ocho plate solves manuales con suficiente cobertura de fase. Cada
plate solve toma minutos bajo cielo real, así que en la práctica ese ajuste casi
nunca llega a hacerse.

Mientras tanto, el tracking visual estima continuamente por mínimos cuadrados
recursivos la matriz ``A`` que relaciona la velocidad comandada (µsteps/s) con
la velocidad medida en la imagen (px/s). La *ganancia* de esa matriz es
justamente la respuesta local del reductor: sube y baja con la fase del lóbulo
cicloidal. Registrando (fase, ganancia) durante una sesión normal se obtienen
miles de muestras gratis, sin mover la montura a propósito ni resolver placas.

La relación entre ambas descripciones es una derivada. Si el desplazamiento
angular extra vale

    offset(s) = C_sin·sin(2πs/P) + C_cos·cos(2πs/P)

entonces la ganancia relativa observada por el tracking es

    g(s) = 1 + (1/k)·d(offset)/ds
         = 1 + (2π/(P·k))·(C_sin·cos(2πs/P) − C_cos·sin(2πs/P))

con ``k`` la escala nominal en grados por paso y ``P`` el período en pasos. Se
ajusta ``g`` contra la fase y se invierte para recuperar ``C_sin``/``C_cos``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class _AxisBins:
    """Medias por bin de fase, robustas a las ráfagas de ruido del RLS."""

    n_bins: int
    total: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.total = np.zeros(int(self.n_bins), dtype=np.float64)
        self.count = np.zeros(int(self.n_bins), dtype=np.int64)

    def add(self, phase: float, value: float) -> None:
        idx = int(phase / (2.0 * math.pi) * self.n_bins) % self.n_bins
        self.total[idx] += float(value)
        self.count[idx] += 1

    def reset(self) -> None:
        self.total[:] = 0.0
        self.count[:] = 0

    def populated(self, min_per_bin: int) -> np.ndarray:
        return self.count >= int(min_per_bin)

    def means(self) -> np.ndarray:
        out = np.zeros_like(self.total)
        np.divide(self.total, self.count, out=out, where=self.count > 0)
        return out


class TransmissionErrorCollector:
    """Acumula ganancia observada vs fase del lóbulo y ajusta el primer armónico.

    Se alimenta con la posición absoluta en pasos y la ganancia relativa medida
    por el tracking (1.0 = exactamente la escala nominal). No mueve nada ni
    depende del plate solving.
    """

    def __init__(
        self,
        *,
        period_steps: Sequence[float],
        deg_per_step: Sequence[float],
        n_bins: int = 24,
        min_samples_per_bin: int = 8,
        min_populated_frac: float = 0.6,
        max_gain_deviation: float = 0.5,
    ) -> None:
        self.period_steps = np.asarray(period_steps, dtype=np.float64).reshape(2)
        self.deg_per_step = np.asarray(deg_per_step, dtype=np.float64).reshape(2)
        self.n_bins = max(6, int(n_bins))
        self.min_samples_per_bin = max(1, int(min_samples_per_bin))
        self.min_populated_frac = float(np.clip(min_populated_frac, 0.1, 1.0))
        # Una ganancia absurda viene de una medición mala del tracking (nube,
        # pérdida de enganche), no del reductor: se descarta antes de acumular.
        self.max_gain_deviation = float(max(0.05, max_gain_deviation))
        self._bins = [_AxisBins(self.n_bins), _AxisBins(self.n_bins)]
        self._seen = 0
        self._rejected = 0

    @property
    def samples(self) -> int:
        return int(self._seen)

    @property
    def rejected(self) -> int:
        return int(self._rejected)

    def reset(self) -> None:
        for b in self._bins:
            b.reset()
        self._seen = 0
        self._rejected = 0

    def observe(self, *, steps: Sequence[float], gain: Sequence[float]) -> bool:
        s = np.asarray(steps, dtype=np.float64).reshape(2)
        g = np.asarray(gain, dtype=np.float64).reshape(2)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(g))):
            self._rejected += 1
            return False
        if np.any(np.abs(g - 1.0) > self.max_gain_deviation):
            self._rejected += 1
            return False
        for axis in range(2):
            period = float(self.period_steps[axis])
            if not np.isfinite(period) or period <= 0.0:
                continue
            phase = (2.0 * math.pi * float(s[axis]) / period) % (2.0 * math.pi)
            self._bins[axis].add(phase, float(g[axis]))
        self._seen += 1
        return True

    def coverage(self) -> Dict[str, float]:
        """Fracción de bins de fase con datos suficientes, por eje."""
        out: Dict[str, float] = {}
        for axis, key in ((0, "az"), (1, "alt")):
            pop = self._bins[axis].populated(self.min_samples_per_bin)
            out[key] = float(np.count_nonzero(pop)) / float(self.n_bins)
        out["min"] = float(min(out["az"], out["alt"]))
        return out

    def fit(self) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Devuelve (coeficientes 2x2 en grados, informe) o None si falta cobertura.

        Las columnas del resultado son (sin, cos), en el mismo orden y unidades
        que ``GoToModel.periodic_coeff_deg``.
        """
        coeff = np.zeros((2, 2), dtype=np.float64)
        report: Dict[str, float] = {}
        fitted = 0
        for axis, key in ((0, "az"), (1, "alt")):
            bins = self._bins[axis]
            pop = bins.populated(self.min_samples_per_bin)
            frac = float(np.count_nonzero(pop)) / float(self.n_bins)
            report[f"{key}_coverage"] = frac
            if frac < self.min_populated_frac:
                continue
            period = float(self.period_steps[axis])
            k = float(self.deg_per_step[axis])
            if (not np.isfinite(period)) or period <= 0.0 or (not np.isfinite(k)) or k == 0.0:
                continue

            centers = (np.arange(self.n_bins) + 0.5) / self.n_bins * 2.0 * math.pi
            means = bins.means()
            phi = centers[pop]
            g = means[pop]
            # g(phi) = c0 + a*cos(phi) + b*sin(phi)
            X = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
            try:
                beta, *_ = np.linalg.lstsq(X, g, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(beta)):
                continue
            _c0, a, b = (float(beta[0]), float(beta[1]), float(beta[2]))
            # invertir la derivada: a = 2*pi*C_sin/(P*k), b = -2*pi*C_cos/(P*k)
            scale = period * k / (2.0 * math.pi)
            coeff[axis, 0] = a * scale        # C_sin
            coeff[axis, 1] = -b * scale       # C_cos
            report[f"{key}_amplitude_deg"] = float(math.hypot(coeff[axis, 0], coeff[axis, 1]))
            fitted += 1

        if fitted <= 0:
            return None
        report["axes_fitted"] = float(fitted)
        report["samples"] = float(self._seen)
        return coeff, report


def gain_from_tracking_matrix(
    A: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Ganancia relativa por eje a partir de la matriz ``A`` del tracking.

    ``A`` mapea µsteps/s a px/s. Su norma por columna es la respuesta del eje;
    normalizada contra una referencia da la variación relativa, que es lo único
    que interesa aquí (la escala absoluta en px/paso depende de la placa y del
    ángulo, no del reductor).
    """
    M = np.asarray(A, dtype=np.float64)
    if M.shape != (2, 2) or not np.all(np.isfinite(M)):
        return None
    norms = np.linalg.norm(M, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    if reference is None:
        return norms
    ref = np.asarray(reference, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(ref)) or np.any(ref <= 0.0):
        return None
    return norms / ref


__all__ = ["TransmissionErrorCollector", "gain_from_tracking_matrix"]
