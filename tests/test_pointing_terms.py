"""El contrato de los terminos del modelo de error.

Dos cosas que el modelo anterior no hacia y que aqui son el contrato: un valor
fuera de lo fisicamente posible se rechaza en vez de recortarse, y las cotas son
por eje, porque los dos ejes de esta montura no comparten mecanica.
"""
from __future__ import annotations

import numpy as np
import pytest

from pointing.terms import AXIS_BOUNDS, Provenance, Term, TermSet, default_terms


def test_a_fresh_model_has_measured_nothing() -> None:
    """Al abrir la app nadie ha medido nada, y eso tiene que notarse.

    Un termino en cero sin medir no corrige nada, que es lo correcto; lo que no
    puede es hacerse pasar por una medida que dio cero.
    """
    terms = default_terms()

    assert terms.measured_names() == ()
    assert set(terms.unmeasured_names()) == {
        "base_tilt",
        "axis_non_perpendicularity",
        "collimation",
        "camera_roll",
        "backlash_steps",
        "transmission_error_deg",
    }
    assert all(t.provenance is Provenance.NOMINAL for t in terms)
    assert terms.all_within_bounds


def test_bounds_are_per_axis_because_the_axes_are_not_the_same_machine() -> None:
    """Azimut es cicloidal impreso, altitud es planetario comprado.

    El juego vive en altitud y el rizado en azimut, asi que una cota comun
    obligaria a elegir entre rechazar el juego real de altitud o admitir en
    azimut uno que ese reductor no puede tener.
    """
    backlash_az, backlash_alt = AXIS_BOUNDS["backlash_steps"]
    ripple_az, ripple_alt = AXIS_BOUNDS["transmission_error_deg"]

    assert backlash_alt > 10.0 * backlash_az, "alt admite un orden de magnitud mas de juego"
    assert backlash_alt >= 3200.0, "el planetario llega a 60 arcmin de juego"
    assert backlash_az <= 200.0, "el cicloidal impreso cierra casi sin juego"

    assert ripple_az > ripple_alt, "el cicloidal es el que riza"
    assert ripple_az >= 0.25


def test_a_measurement_outside_the_bound_is_refused_not_clipped() -> None:
    """La diferencia con el modelo viejo, que proyectaba y guardaba.

    Un ajuste recortado contra su cota se vuelve indistinguible de uno bueno y
    se arrastra toda la noche. Aqui falla ruidosamente.
    """
    terms = default_terms()
    backlash = terms["backlash_steps"]

    with pytest.raises(ValueError, match="se rechaza, no se recorta"):
        backlash.updated(
            [50.0, 9000.0],  # 9000 pasos en altitud no los tiene ningun planetario
            sigma=[5.0, 50.0],
            provenance=Provenance.PLATE_SOLVE,
            n_obs=4,
        )

    # y el termino original queda intacto
    assert terms["backlash_steps"].n_obs == 0
    assert np.all(terms["backlash_steps"].value == 0.0)


def test_a_plausible_measurement_is_accepted_and_carries_its_provenance() -> None:
    terms = default_terms()

    measured = terms["backlash_steps"].updated(
        [12.0, 1600.0], sigma=[8.0, 40.0], provenance=Provenance.PLATE_SOLVE, n_obs=3, t_unix=1e9
    )
    terms.set(measured)

    assert measured.measured
    assert measured.provenance is Provenance.PLATE_SOLVE
    assert measured.n_obs == 3
    assert measured.t_updated == 1e9
    assert terms.measured_names() == ("backlash_steps",)
    assert "backlash_steps" not in terms.unmeasured_names()


def test_a_value_sitting_exactly_on_the_bound_is_flagged() -> None:
    """Pegado a la cota no es lo mismo que dentro: es una medida sospechosa."""
    terms = default_terms()
    bound_alt = AXIS_BOUNDS["backlash_steps"][1]

    edge = terms["backlash_steps"].updated(
        [0.0, bound_alt], sigma=[1.0, 1.0], provenance=Provenance.PLATE_SOLVE, n_obs=2
    )

    assert edge.within_bounds  # cabe, justo
    assert bool(edge.at_bound[1]), "la componente de altitud esta en el borde"
    assert not bool(edge.at_bound[0])


def test_transmission_error_has_a_sine_and_cosine_per_axis() -> None:
    """Cuatro numeros, no dos: cada eje lleva su primer armonico completo."""
    terms = default_terms()
    ripple = terms["transmission_error_deg"]

    assert ripple.value.shape == (4,)
    az_bound, alt_bound = AXIS_BOUNDS["transmission_error_deg"]
    np.testing.assert_allclose(ripple.bound, [az_bound, az_bound, alt_bound, alt_bound])


def test_tracking_is_a_first_class_provenance_for_the_ripple() -> None:
    """El rizado se aprende del tracking, que barre la fase gratis toda la noche.

    Ajustarlo por plate solves exige ocho medidas con cobertura de fase y en la
    practica no llegaba a hacerse nunca; de ahi que la procedencia sea parte del
    termino y no una nota al margen.
    """
    terms = default_terms()

    learned = terms["transmission_error_deg"].updated(
        [0.12, -0.04, 0.005, 0.002],
        sigma=[0.01, 0.01, 0.002, 0.002],
        provenance=Provenance.TRACKING,
        n_obs=2400,
    )

    assert learned.provenance is Provenance.TRACKING
    assert learned.n_obs == 2400
    assert learned.measured


def test_shape_mismatches_are_caught_at_construction() -> None:
    with pytest.raises(ValueError, match="no cuadra"):
        Term(name="x", value=np.zeros(2), sigma=np.zeros(3), unit="deg", bound=np.ones(2))
    with pytest.raises(ValueError, match="no cuadra"):
        Term(name="x", value=np.zeros(2), sigma=np.zeros(2), unit="deg", bound=np.ones(3))
    with pytest.raises(ValueError, match="sigma no puede ser negativa"):
        Term(name="x", value=np.zeros(2), sigma=np.array([-1.0, 0.0]), unit="deg", bound=np.ones(2))


def test_describe_says_whether_a_number_was_measured() -> None:
    terms = default_terms()
    assert "sin medir" in terms["camera_roll"].describe()

    terms.set(
        terms["camera_roll"].updated(
            [-3.5], sigma=[0.4], provenance=Provenance.PLATE_SOLVE, n_obs=7
        )
    )
    line = terms["camera_roll"].describe()
    assert "plate_solve" in line and "n=7" in line and "sin medir" not in line


def test_the_whole_set_reports_whether_anything_is_out_of_bounds() -> None:
    terms = default_terms()
    assert terms.all_within_bounds

    # un termino construido a mano fuera de cota (no via updated) se detecta igual
    terms.set(
        Term(
            name="collimation",
            value=np.array([5.0]),
            sigma=np.array([0.1]),
            unit="deg",
            bound=np.array([1.0]),
            provenance=Provenance.PLATE_SOLVE,
            n_obs=3,
        )
    )
    assert not terms.all_within_bounds


class TestTermSetIsAPlainContainer:
    """Sin magia: nombres dentro, terminos fuera."""

    def test_membership_and_iteration(self) -> None:
        terms = default_terms()
        assert "base_tilt" in terms
        assert "no_existe" not in terms
        assert len(list(terms)) == 6

    def test_set_replaces_by_name(self) -> None:
        terms = TermSet()
        t = Term(name="a", value=np.zeros(1), sigma=np.zeros(1), unit="deg", bound=np.ones(1))
        terms.set(t)
        terms.set(t.updated([0.5], sigma=[0.1], provenance=Provenance.OPERATOR, n_obs=1))
        assert len(list(terms)) == 1
        assert terms["a"].value[0] == 0.5
