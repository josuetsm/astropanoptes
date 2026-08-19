from __future__ import annotations

import math

import numpy as np

from transmission_error import TransmissionErrorCollector, gain_from_tracking_matrix

PERIOD = 12800.0          # un lóbulo cicloidal, 45 lóbulos por vuelta
DEG_PER_STEP = 1.0 / 1600.0


def _collector(**kw) -> TransmissionErrorCollector:
    params = dict(
        period_steps=(PERIOD, PERIOD),
        deg_per_step=(DEG_PER_STEP, DEG_PER_STEP),
        n_bins=24,
        min_samples_per_bin=2,
        min_populated_frac=0.6,
    )
    params.update(kw)
    return TransmissionErrorCollector(**params)


def _gain_for(steps: float, c_sin: float, c_cos: float) -> float:
    """Ganancia relativa que produciría un offset C_sin·sin + C_cos·cos."""
    phase = 2.0 * math.pi * steps / PERIOD
    d_offset = (2.0 * math.pi / PERIOD) * (c_sin * math.cos(phase) - c_cos * math.sin(phase))
    return 1.0 + d_offset / DEG_PER_STEP


def test_recovers_a_known_transmission_error() -> None:
    """Inyecta un error conocido y comprueba que se recupera."""
    c_sin, c_cos = 0.06, -0.03          # grados
    col = _collector()
    for s in np.linspace(0.0, 3.0 * PERIOD, 900):
        g = _gain_for(float(s), c_sin, c_cos)
        col.observe(steps=(s, s), gain=(g, g))

    out = col.fit()
    assert out is not None
    coeff, report = out
    assert report["axes_fitted"] == 2.0
    np.testing.assert_allclose(coeff[0], [c_sin, c_cos], atol=2e-3)
    np.testing.assert_allclose(coeff[1], [c_sin, c_cos], atol=2e-3)


def test_recovers_error_despite_measurement_noise() -> None:
    """El RLS del tracking es ruidoso; el promediado por bins debe absorberlo."""
    rng = np.random.default_rng(7)
    c_sin, c_cos = 0.05, 0.02
    col = _collector(min_samples_per_bin=6)
    for s in rng.uniform(0.0, 4.0 * PERIOD, 4000):
        g = _gain_for(float(s), c_sin, c_cos) + rng.normal(0.0, 0.05)
        col.observe(steps=(s, s), gain=(g, g))

    out = col.fit()
    assert out is not None
    coeff, _ = out
    amp_true = math.hypot(c_sin, c_cos)
    amp_fit = math.hypot(coeff[0, 0], coeff[0, 1])
    assert abs(amp_fit - amp_true) < 0.2 * amp_true


def test_refuses_to_fit_without_phase_coverage() -> None:
    """Sin recorrer el lóbulo no hay nada que ajustar: mejor no inventar."""
    col = _collector()
    for s in np.linspace(0.0, PERIOD * 0.1, 200):     # solo 10% del ciclo
        g = _gain_for(float(s), 0.05, 0.0)
        col.observe(steps=(s, s), gain=(g, g))
    assert col.fit() is None
    assert col.coverage()["min"] < 0.6


def test_rejects_implausible_gain_samples() -> None:
    """Una nube o una pérdida de enganche no debe entrar al modelo."""
    col = _collector()
    assert col.observe(steps=(0.0, 0.0), gain=(1.02, 0.98))
    assert not col.observe(steps=(0.0, 0.0), gain=(9.0, 1.0))
    assert not col.observe(steps=(0.0, 0.0), gain=(float("nan"), 1.0))
    assert col.samples == 1
    assert col.rejected == 2


def test_flat_response_yields_negligible_coefficients() -> None:
    """Un reductor perfecto no debe producir un error periódico inventado."""
    col = _collector()
    for s in np.linspace(0.0, 2.0 * PERIOD, 600):
        col.observe(steps=(s, s), gain=(1.0, 1.0))
    out = col.fit()
    assert out is not None
    coeff, _ = out
    assert float(np.max(np.abs(coeff))) < 1e-6


def test_gain_from_tracking_matrix_normalises_against_reference() -> None:
    A = np.diag([2.0, 4.0])
    np.testing.assert_allclose(gain_from_tracking_matrix(A), [2.0, 4.0])
    np.testing.assert_allclose(
        gain_from_tracking_matrix(A, reference=[2.0, 5.0]), [1.0, 0.8]
    )
    assert gain_from_tracking_matrix(np.full((2, 2), np.nan)) is None
    assert gain_from_tracking_matrix(np.zeros((2, 2))) is None
