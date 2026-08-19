from __future__ import annotations

import numpy as np

from ap_types import Axis
from goto import GoToModel


def _model(alt_deg: float = 0.0) -> GoToModel:
    m = GoToModel()
    m.init_from_mechanics()
    m.ref_az_alt_deg = np.array([0.0, float(alt_deg)], dtype=np.float64)
    return m


def test_coupling_budget_is_geometric_not_raw_azimuth() -> None:
    """A real non-orthogonality must not be rejected just for being near zenith.

    J[0,1] is degrees of azimuth per alt-step, and near the zenith one degree
    of azimuth spans a much smaller angle on the sky. Judging the raw entry
    against a fixed budget makes the same physical tilt pass low and fail high.
    """
    mech = _model().mechanical_J()
    tilt_deg = 1.6                       # non-orthogonality actually measured
    sky_coupling = np.sin(np.deg2rad(tilt_deg))

    for alt in (0.0, 78.0):
        m = _model(alt)
        J = mech.copy()
        # same physical tilt, expressed in azimuth degrees at this altitude
        J[0, 1] = sky_coupling / np.cos(np.deg2rad(alt)) * mech[1, 1]
        assert m.is_J_within_mechanical_limits(J), (
            f"tilt de {tilt_deg} deg rechazado a alt={alt}"
        )


def test_coupling_still_rejects_a_genuinely_large_tilt() -> None:
    """The widened budget must not become a blank cheque."""
    m = _model(78.0)
    J = m.mechanical_J().copy()
    J[0, 1] = np.sin(np.deg2rad(25.0)) / np.cos(np.deg2rad(78.0)) * J[1, 1]
    assert not m.is_J_within_mechanical_limits(J)


def test_scale_envelope_is_unchanged() -> None:
    """J is the mean scale; its +/-10% envelope stays tight."""
    m = _model()
    mech = m.mechanical_J()
    assert m.is_J_within_mechanical_limits(mech)
    too_far = mech.copy()
    too_far[0, 0] *= 1.11
    assert not m.is_J_within_mechanical_limits(too_far)
    reversed_alt = mech.copy()
    reversed_alt[1, 1] *= -1.0
    assert not m.is_J_within_mechanical_limits(reversed_alt)


def test_phase_coverage_reports_cycles_spanned() -> None:
    m = _model()
    period = float(m.kin.transmission_error_period_steps(Axis.ALT))
    m._manual_steps_abs = [
        np.array([0.0, 0.0]),
        np.array([0.0, period * 0.25]),
        np.array([0.0, period * 0.5]),
    ]
    cov = m.manual_phase_coverage()
    assert cov["alt"] == 0.5
    assert cov["az"] == 0.0
    assert cov["min"] == 0.0


def test_phase_coverage_distinguishes_short_travel_from_bad_gearing() -> None:
    """This is the case that blocked a real session.

    Moves of 400 and 1600 steps against a 12800-step lobe measured the local
    slope of the transmission error (87% of nominal), not the mean scale. A
    full-cycle move on the same hardware measured 100.5%.
    """
    m = _model()
    period = float(m.kin.transmission_error_period_steps(Axis.ALT))
    m._manual_steps_abs = [
        np.array([0.0, 0.0]),
        np.array([0.0, 400.0]),
        np.array([0.0, 2000.0]),
    ]
    assert m.manual_phase_coverage()["alt"] < 0.5      # -> recorrido insuficiente

    m._manual_steps_abs = [
        np.array([0.0, 0.0]),
        np.array([0.0, period * 0.6]),
        np.array([0.0, period]),
    ]
    assert m.manual_phase_coverage()["alt"] >= 1.0     # -> escala media medible
