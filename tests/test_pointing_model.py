"""El contrato del mapa de pasos a cielo.

Lo que se comprueba aqui es que el modelo es invertible, que cada termino
geometrico tiene la dependencia con la altura que le toca -- que es lo que los
hace separables y lo que una matriz constante no podia representar -- y que el
backlash entra segun el sentido de carga y no como una constante.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ap_types import Axis
from pointing.kinematics import MountKinematics
from pointing.model import (
    PointingModel,
    PointingReference,
    apply_geometry,
    remove_geometry,
)
from pointing.terms import Provenance, default_terms

ARCSEC = 1.0 / 3600.0


def _az_gap(a: float, b: float) -> float:
    """Distancia angular entre dos azimuts, sin que el corte en 360 estorbe."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _model(**geometry) -> PointingModel:
    m = PointingModel(kin=MountKinematics(), terms=default_terms())
    if geometry:
        tilt = [geometry.get("tilt_north_deg", 0.0), geometry.get("tilt_west_deg", 0.0)]
        m.terms.set(m.terms["base_tilt"].updated(
            tilt, sigma=[0.01, 0.01], provenance=Provenance.PLATE_SOLVE, n_obs=8))
        m.terms.set(m.terms["axis_non_perpendicularity"].updated(
            [geometry.get("npae_deg", 0.0)], sigma=[0.01],
            provenance=Provenance.PLATE_SOLVE, n_obs=8))
        m.terms.set(m.terms["collimation"].updated(
            [geometry.get("collimation_deg", 0.0)], sigma=[0.01],
            provenance=Provenance.PLATE_SOLVE, n_obs=8))
    m.note_steps((0.0, 0.0), load_dir=(1, 1))
    m.sync(az_world_deg=120.0, alt_world_deg=45.0)
    return m


# --------------------------------------------------------------------- geometria


def test_geometry_round_trips_to_under_an_arcsecond() -> None:
    """Ida y vuelta con todos los terminos activos."""
    kwargs = dict(tilt_north_deg=1.2, tilt_west_deg=-0.7, npae_deg=0.9, collimation_deg=0.4)

    for az in (0.0, 73.0, 180.0, 291.0):
        for alt in (12.0, 35.0, 62.0, 78.0):
            world = apply_geometry(az, alt, **kwargs)
            back = remove_geometry(*world, **kwargs)
            # el azimut da la vuelta: 359.9999 y 0.0 son el mismo sitio
            assert _az_gap(back[0], az) < 0.5 * ARCSEC
            assert back[1] == pytest.approx(alt, abs=0.5 * ARCSEC)


def test_a_model_with_no_measured_terms_is_the_identity() -> None:
    """Sin nada medido, el modelo no corrige: no inventa correcciones."""
    for az, alt in ((10.0, 20.0), (250.0, 70.0)):
        assert apply_geometry(
            az, alt, tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=0.0
        ) == (az % 360.0, alt)


def test_base_tilt_is_a_sinusoid_in_azimuth_and_flat_in_altitude() -> None:
    """La firma de una inclinacion: seno del azimut en la altura.

    Es lo que permite separarla de los otros dos terminos con observaciones
    repartidas en azimut, y por eso el estimador exige 120 grados de recorrido.
    """
    kwargs = dict(tilt_north_deg=1.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=0.0)

    d_alt = [apply_geometry(az, 45.0, **kwargs)[1] - 45.0 for az in (0.0, 90.0, 180.0, 270.0)]

    assert d_alt[0] == pytest.approx(+1.0, abs=1e-9)   # cos(0)
    assert d_alt[1] == pytest.approx(0.0, abs=1e-9)    # cos(90)
    assert d_alt[2] == pytest.approx(-1.0, abs=1e-9)   # cos(180)
    assert d_alt[3] == pytest.approx(0.0, abs=1e-9)


def test_non_perpendicularity_grows_as_tan_of_altitude() -> None:
    """La dependencia que una matriz constante no puede representar.

    De ahi que el modelo viejo tuviera que ensanchar su tolerancia por
    1/cos(alt) para no rechazar datos buenos: estaba metiendo un tan(alt) en
    cuatro numeros constantes.
    """
    kwargs = dict(tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=1.0, collimation_deg=0.0)

    low = apply_geometry(100.0, 10.0, **kwargs)[0] - 100.0
    high = apply_geometry(100.0, 70.0, **kwargs)[0] - 100.0

    assert low == pytest.approx(math.tan(math.radians(10.0)), abs=1e-9)
    assert high == pytest.approx(math.tan(math.radians(70.0)), abs=1e-9)
    assert high > 15.0 * low, "tan(70) es mas de quince veces tan(10)"


def test_collimation_grows_as_secant_of_altitude() -> None:
    """Y esta como 1/cos(alt): distinta de la anterior, luego separable."""
    kwargs = dict(tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=1.0)

    low = apply_geometry(100.0, 10.0, **kwargs)[0] - 100.0
    high = apply_geometry(100.0, 70.0, **kwargs)[0] - 100.0

    assert low == pytest.approx(1.0 / math.cos(math.radians(10.0)), abs=1e-9)
    assert high == pytest.approx(1.0 / math.cos(math.radians(70.0)), abs=1e-9)


def test_the_two_altitude_dependent_terms_are_not_the_same_shape() -> None:
    """tan(alt) y sec(alt) se separan, pero hace falta llegar arriba.

    A baja altura casi coinciden; solo por encima de 60 grados se distinguen de
    verdad, que es por lo que el estimador exige una muestra alta.
    """
    npae = dict(tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=1.0, collimation_deg=0.0)
    coll = dict(tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=1.0)

    def gap(alt: float) -> float:
        return abs(
            (apply_geometry(100.0, alt, **npae)[0] - 100.0)
            - (apply_geometry(100.0, alt, **coll)[0] - 100.0)
        )

    assert gap(70.0) < gap(20.0), "arriba las dos formas convergen en magnitud"
    assert gap(20.0) > 0.5, "abajo se distinguen bien"


# ------------------------------------------------------------------ pasos a cielo


def test_steps_and_world_round_trip() -> None:
    model = _model(tilt_north_deg=0.8, tilt_west_deg=-0.5, npae_deg=0.6, collimation_deg=0.3)

    for target in ((123.0, 40.0), (118.0, 55.0), (130.0, 33.0)):
        steps = model.steps_for_world(*target)
        back = model.world_from_steps(steps)
        assert _az_gap(back[0], target[0]) < 2.0 * ARCSEC
        assert back[1] == pytest.approx(target[1], abs=2.0 * ARCSEC)


def test_the_scale_comes_from_the_gear_ratio_and_is_never_fitted() -> None:
    """Mover N pasos recorre exactamente lo que dice la reduccion."""
    model = _model()
    kin = MountKinematics()

    az0, alt0 = model.axis_angles_deg((0.0, 0.0))
    az1, _ = model.axis_angles_deg((1000.0, 0.0))
    _, alt1 = model.axis_angles_deg((0.0, 1000.0))

    assert az1 - az0 == pytest.approx(1000.0 * kin.deg_per_step(Axis.AZ), abs=1e-9)
    assert alt1 - alt0 == pytest.approx(1000.0 * kin.deg_per_step(Axis.ALT), abs=1e-9)


def test_reversing_direction_loses_the_slack() -> None:
    """El backlash es estado, no constante: depende del sentido de carga.

    En altitud, con un planetario comprado, ese juego son cientos de pasos y se
    pierde entero en cada inversion. El modelo viejo no lo representaba en
    absoluto: lo emitia como pulsos no contados y seguia.
    """
    model = _model()
    slack = 1600.0
    model.terms.set(model.terms["backlash_steps"].updated(
        [0.0, slack], sigma=[1.0, 20.0], provenance=Provenance.PLATE_SOLVE, n_obs=3))

    forward = model.axis_angles_deg((0.0, 5000.0), load_dir=(1, +1))[1]
    reversed_ = model.axis_angles_deg((0.0, 5000.0), load_dir=(1, -1))[1]

    k = MountKinematics().deg_per_step(Axis.ALT)
    assert abs(forward - reversed_) == pytest.approx(abs(k * slack), rel=1e-9)


def test_the_ripple_is_periodic_in_one_motor_revolution() -> None:
    """El rizado se repite cada vuelta de motor y se cancela sobre un ciclo entero.

    Es la razon de que los movimientos de calibracion tengan que ser de ciclo
    completo, y la que hacia que los del 14% leyeran la pendiente local.
    """
    model = _model()
    model.terms.set(model.terms["transmission_error_deg"].updated(
        [0.20, 0.0, 0.0, 0.0], sigma=[0.01] * 4, provenance=Provenance.TRACKING, n_obs=2000))

    period = MountKinematics().transmission_error_period_steps(Axis.AZ)
    k = MountKinematics().deg_per_step(Axis.AZ)

    whole = model.axis_angles_deg((period, 0.0))[0] - model.axis_angles_deg((0.0, 0.0))[0]
    assert whole == pytest.approx(k * period, abs=1e-9), "sobre un ciclo el rizado se cancela"

    quarter = model.axis_angles_deg((period / 4.0, 0.0))[0] - model.axis_angles_deg((0.0, 0.0))[0]
    assert abs(quarter - k * period / 4.0) == pytest.approx(0.20, abs=1e-9)


def test_the_inverse_accounts_for_the_ripple() -> None:
    """Apuntar con rizado activo tiene que llegar igual de cerca."""
    model = _model()
    model.terms.set(model.terms["transmission_error_deg"].updated(
        [0.18, -0.06, 0.004, 0.001], sigma=[0.01] * 4,
        provenance=Provenance.TRACKING, n_obs=2000))

    for target in ((125.0, 50.0), (112.0, 38.0)):
        steps = model.steps_for_world(*target)
        back = model.world_from_steps(steps)
        assert _az_gap(back[0], target[0]) < 2.0 * ARCSEC
        assert back[1] == pytest.approx(target[1], abs=2.0 * ARCSEC)


def test_predicting_without_a_reference_is_an_error_not_a_guess() -> None:
    """Sin sincronizar no se sabe donde apunta, y eso no se disimula."""
    model = PointingModel()

    assert not model.synced
    assert model.current_world_deg() is None
    with pytest.raises(RuntimeError, match="sincroniza"):
        model.axis_angles_deg((0.0, 0.0))
    with pytest.raises(RuntimeError, match="sincroniza"):
        model.steps_for_world(120.0, 45.0)


def test_sync_puts_the_current_steps_on_the_observed_sky() -> None:
    model = PointingModel()
    model.note_steps((4321.0, -765.0), load_dir=(1, -1))
    model.sync(az_world_deg=258.87, alt_world_deg=39.41, t_unix=1e9)

    assert model.synced
    where = model.current_world_deg()
    assert where is not None
    assert _az_gap(where[0], 258.87) < ARCSEC
    assert where[1] == pytest.approx(39.41, abs=ARCSEC)
    assert model.reference is not None and model.reference.steps == (4321.0, -765.0)


def test_sync_survives_a_tilted_mount() -> None:
    """La referencia se guarda en angulos de eje, no de cielo.

    Si se guardara en cielo, cambiar la inclinacion medida movería el apuntado
    de sitio sin que la montura se hubiera movido.
    """
    model = _model(tilt_north_deg=1.5, tilt_west_deg=-0.9, npae_deg=0.7, collimation_deg=0.2)
    where = model.current_world_deg()

    assert where is not None
    assert _az_gap(where[0], 120.0) < ARCSEC
    assert where[1] == pytest.approx(45.0, abs=ARCSEC)
