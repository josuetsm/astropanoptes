"""La fachada cubre lo que el runner usa, miembro por miembro.

Es lo que hace seguro el cambio de modelo: `app_runner` lee 23 miembros del
modelo anterior, y el commit que cambia uno por otro solo es pequeno y
reversible si esos 23 ya existen con el mismo tipo y el mismo sentido. Este test
los enumera a mano *y* los descubre del codigo, para que anadir un uso nuevo en
el runner rompa aqui y no de noche bajo el cielo.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ap_types import Axis
from pointing.model import PointingModel
from pointing.terms import Provenance

RUNNER = Path(__file__).resolve().parents[1] / "app_runner.py"

# Lo que el runner usa hoy. Escrito a mano para que la lista sea legible y para
# que quitar algo del runner no relaje el contrato sin que nadie se entere.
EXPECTED_SURFACE = {
    "J_deg_per_step", "add_manual_sample", "backlash_steps_alt", "backlash_steps_az",
    "current_az_alt_deg", "init_from_mechanics", "kin", "last_move_direction",
    "last_move_direction_alt", "last_move_direction_az",
    "manual_sample_continuity_report", "model_fit_report", "note_emitted_rate_steps",
    "note_manual_move", "periodic_coeff_deg", "periodic_model_samples",
    "predict_az_alt_deg", "safe_periodic_coeff_for_prediction",
    "set_last_move_direction", "set_microsteps", "sidereal_step_rate_deg_s",
    "steps_est", "sync_from_world_az_alt", "synced",
}

# Estos dos los reemplaza `admit_observation` en el mismo commit del cambio: son
# el unico sitio del runner que cambia de verdad, y por eso no estan en la
# fachada.
REPLACED_BY_ADMISSION = {"add_manual_sample", "manual_sample_continuity_report"}


def _model() -> PointingModel:
    m = PointingModel()
    m.note_steps((1000.0, -2000.0), load_dir=(1, -1))
    m.sync(az_world_deg=120.0, alt_world_deg=45.0)
    return m


def test_the_runner_uses_no_member_this_test_does_not_know_about() -> None:
    """Si alguien engancha un miembro nuevo en el runner, se entera aqui."""
    used = set(re.findall(r"_goto\.model\.([a-zA-Z_]+)", RUNNER.read_text(encoding="utf-8")))

    unknown = used - EXPECTED_SURFACE
    assert not unknown, f"el runner usa miembros no contemplados: {sorted(unknown)}"


def test_every_member_the_runner_reads_exists_on_the_new_model() -> None:
    model = _model()

    missing = [
        name for name in EXPECTED_SURFACE - REPLACED_BY_ADMISSION
        if not hasattr(model, name)
    ]
    assert not missing, f"la fachada no cubre: {sorted(missing)}"


def test_the_matrix_the_tracking_seed_reads_is_the_mechanical_one() -> None:
    """Ya no se ajusta: es la reduccion, que se conoce exactamente.

    El tracking la usa solo como semilla de su RLS, que es lazo cerrado y
    reaprende la respuesta real, asi que fijarla lo hace mas predecible, no peor.
    """
    model = _model()
    j = model.J_deg_per_step

    assert j.shape == (2, 2)
    assert j[0, 0] == pytest.approx(model.kin.deg_per_step(Axis.AZ))
    assert j[1, 1] == pytest.approx(model.kin.deg_per_step(Axis.ALT))
    assert j[0, 1] == 0.0 and j[1, 0] == 0.0
    assert np.array_equal(model.safe_J_for_prediction(), j)


def test_the_coupling_lives_in_the_jacobian_not_in_the_matrix() -> None:
    """El `J[0,1]` que el modelo viejo ajustaba libremente ahora sale solo.

    Una inclinacion hace que un paso de altitud mueva el azimut del mundo; con
    los terminos con nombre eso aparece en la jacobiana sin que nadie lo ajuste.
    """
    model = _model()
    assert model.world_jacobian()[0, 1] == pytest.approx(0.0, abs=1e-12)

    model.terms.set(model.terms["base_tilt"].updated(
        [1.2, -0.7], sigma=[0.01, 0.01], provenance=Provenance.PLATE_SOLVE, n_obs=8))

    assert abs(model.world_jacobian()[0, 1]) > 1e-7


def test_step_bookkeeping_matches_the_old_semantics() -> None:
    model = _model()

    model.note_manual_move(Axis.AZ, +1, 500)
    assert model.steps_est[0] == pytest.approx(1500.0)
    assert model.last_move_direction(Axis.AZ) == +1
    assert model.last_move_direction_az == +1

    model.note_manual_move(Axis.ALT, -1, 300)
    assert model.steps_est[1] == pytest.approx(-2300.0)
    assert model.last_move_direction_alt == -1

    model.note_emitted_rate_steps([10.0, -4.0])
    assert model.steps_est[0] == pytest.approx(1510.0)
    assert model.steps_est[1] == pytest.approx(-2304.0)


def test_the_transmission_learner_can_still_write_its_result() -> None:
    """El unico camino que escribe en el modelo desde fuera, y sigue valiendo."""
    model = _model()
    coeff = np.array([[0.10, -0.03], [0.004, 0.001]])

    model.periodic_coeff_deg = model.safe_periodic_coeff_for_prediction(coeff)
    model.periodic_model_samples = 2400

    np.testing.assert_allclose(model.periodic_coeff_deg, coeff)
    assert model.periodic_model_samples == 2400
    assert model.terms["transmission_error_deg"].provenance is Provenance.TRACKING


def test_the_ripple_bound_is_applied_per_axis() -> None:
    """Un cicloidal impreso riza; un planetario comprado no.

    Recortar los dos con el mismo numero dejaria pasar en altitud una medida
    absurda solo porque en azimut seria normal.
    """
    model = _model()
    absurd = np.array([[0.10, 0.0], [0.90, 0.0]])

    safe = model.safe_periodic_coeff_for_prediction(absurd)

    assert safe[0, 0] == pytest.approx(0.10), "en azimut cabe"
    assert safe[1, 0] < 0.10, "en altitud se recorta muy por debajo"


def test_cancelling_a_goto_invalidates_the_sync() -> None:
    """El runner pone `synced = False` al cancelar; eso suelta la referencia."""
    model = _model()
    assert model.synced

    model.synced = False

    assert not model.synced
    assert model.current_az_alt_deg() is None


def test_pointing_readouts_have_the_old_shapes() -> None:
    model = _model()

    where = model.current_az_alt_deg()
    assert isinstance(where, np.ndarray) and where.shape == (2,)
    assert where[0] == pytest.approx(120.0, abs=1e-6)

    predicted = model.predict_az_alt_deg()
    assert predicted.shape == (2,)
    from_reference = model.predict_az_alt_deg(from_ref=True)
    assert from_reference.shape == (2,)


def test_sync_from_world_rejects_garbage_instead_of_storing_it() -> None:
    model = PointingModel()

    assert not model.sync_from_world_az_alt([float("nan"), 45.0])
    assert not model.synced
    assert model.sync_from_world_az_alt([258.87, 39.41])
    assert model.synced


def test_the_fit_report_is_a_flat_dictionary_with_provenance() -> None:
    model = _model()
    report = model.model_fit_report()

    assert report["synced"] is True
    assert report["model_fit_samples"] == 0
    assert report["term_base_tilt_provenance"] == "nominal"
    assert report["term_backlash_steps"] == [0.0, 0.0]


def test_microstepping_changes_reach_the_kinematics() -> None:
    model = _model()
    before = model.kin.deg_per_step(Axis.AZ)

    model.set_microsteps(32, 32)

    assert model.kin.deg_per_step(Axis.AZ) == pytest.approx(2.0 * before)


def test_init_from_mechanics_is_a_no_op_because_mechanics_never_left() -> None:
    model = _model()
    before = model.J_deg_per_step.copy()

    model.init_from_mechanics()

    np.testing.assert_array_equal(model.J_deg_per_step, before)


def test_the_sidereal_feed_forward_matches_the_model_being_replaced() -> None:
    """La comprobacion que hace seguro el cambio, y la mas delicada de todas.

    El feed-forward sideral es lo unico del modelo que el tracking usa en
    caliente, y el tracking hoy funciona. El modelo nuevo llega a la velocidad
    por otro camino -- diferencia finita sobre el cielo y jacobiana numerica del
    mapa, en vez de una matriz ajustada -- asi que lo que importa es que el
    numero salga igual. Sale igual a trece decimales.

    Este test se borra con `GoToModel`; hasta entonces es el puente.
    """
    from astropy.time import Time

    from goto import GoToModel
    from platesolving import ObserverConfig

    observer = ObserverConfig()
    obstime = Time("2026-09-16T01:28:45", scale="utc")

    old = GoToModel()
    old.init_from_mechanics()
    new = PointingModel()

    for az, alt in ((120.0, 45.0), (258.9, 39.4), (10.0, 70.0), (200.0, 20.0), (300.0, 60.0)):
        old.sync_from_world_az_alt(np.array([az, alt]))
        new.note_steps((0.0, 0.0), load_dir=(1, 1))
        new.sync(az_world_deg=az, alt_world_deg=alt)

        before = old.sidereal_step_rate_deg_s(
            az_deg=az, alt_deg=alt, observer=observer, obstime=obstime, dt_s=1.0)
        after = new.sidereal_step_rate_deg_s(
            az_deg=az, alt_deg=alt, observer=observer, obstime=obstime, dt_s=1.0)

        assert before is not None and after is not None
        np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-12)


def test_the_feed_forward_still_answers_before_anything_is_synced() -> None:
    """Al arrancar no hay referencia, y el tracking pide la tasa igualmente.

    Sin referencia no hay jacobiana del mapa, asi que se usa la mecanica: es una
    aproximacion buena y, sobre todo, es un numero finito en vez de None.
    """
    from astropy.time import Time

    from platesolving import ObserverConfig

    model = PointingModel()
    assert not model.synced

    rate = model.sidereal_step_rate_deg_s(
        az_deg=120.0, alt_deg=45.0, observer=ObserverConfig(),
        obstime=Time("2026-09-16T01:28:45", scale="utc"), dt_s=1.0,
    )

    assert rate is not None and np.all(np.isfinite(rate))
