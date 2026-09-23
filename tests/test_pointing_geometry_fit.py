"""El contrato del ajuste geometrico.

Los tres terminos se ajustan juntos porque por separado son el mismo grado de
libertad: el codigo anterior alternaba entre una matriz J y una rotacion, y cada
uno explicaba lo que el otro acababa de explicar. Estos tests fijan que ahora se
recuperan de verdad, y que cuando los datos no pueden sostenerlos el estimador
dice que direccion falta en vez de devolver un numero.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pointing.estimators.tilt import (
    GeometryOutcome,
    GeometrySample,
    fit_geometry,
)
from pointing.model import apply_geometry

TRUE = dict(tilt_north_deg=1.2, tilt_west_deg=-0.7, npae_deg=0.6, collimation_deg=0.3)
SIGMA = 0.0006  # ~2 arcsec


def _samples(pointings, *, truth=None, sigma=SIGMA, seed=11, index=(0.0, 0.0)):
    """Observaciones sinteticas: la mecanica dice una cosa y el cielo otra."""
    truth = TRUE if truth is None else truth
    rng = np.random.default_rng(seed)
    out = []
    for az, alt in pointings:
        # El offset de indice desplaza el *angulo de eje*, no la posicion del
        # cielo: dice donde esta el cero mecanico, asi que entra antes de la
        # geometria y no despues.
        world_az, world_alt = apply_geometry(az + index[0], alt + index[1], **truth)
        out.append(
            GeometrySample(
                az_nominal_deg=az,
                alt_nominal_deg=alt,
                az_obs_deg=world_az + rng.normal(0.0, sigma),
                alt_obs_deg=world_alt + rng.normal(0.0, sigma),
                sigma_deg=sigma,
            )
        )
    return out


def _well_spread():
    """Reparto que si excita las cuatro formas: azimut completo y una alta."""
    return [
        (10.0, 30.0), (75.0, 55.0), (140.0, 25.0), (200.0, 68.0),
        (260.0, 40.0), (320.0, 62.0), (45.0, 72.0), (175.0, 45.0),
    ]


def test_recovers_all_three_terms_from_well_spread_observations() -> None:
    fit = fit_geometry(_samples(_well_spread()))

    assert fit.outcome is GeometryOutcome.OK, fit.detail
    assert fit.base_tilt_deg[0] == pytest.approx(TRUE["tilt_north_deg"], abs=0.05)
    assert fit.base_tilt_deg[1] == pytest.approx(TRUE["tilt_west_deg"], abs=0.05)
    assert fit.npae_deg == pytest.approx(TRUE["npae_deg"], abs=0.05)
    assert fit.collimation_deg == pytest.approx(TRUE["collimation_deg"], abs=0.05)


def test_the_fit_actually_reduces_the_residual() -> None:
    """Un ajuste que no baja los residuos no ha explicado nada."""
    fit = fit_geometry(_samples(_well_spread()))

    assert fit.ok
    assert fit.rms_arcsec < 0.1 * fit.rms_before_arcsec
    assert fit.rms_arcsec < 10.0


def test_index_offsets_are_absorbed_and_do_not_contaminate_the_terms() -> None:
    """Donde esta el cero de cada eje no es un termino: lo fija la sincronizacion.

    Entra como parametro de estorbo para que no se reparta entre los demas.
    """
    fit = fit_geometry(_samples(_well_spread(), index=(3.0, -1.5)))

    assert fit.ok, fit.detail
    assert fit.index_offset_deg[0] == pytest.approx(3.0, abs=0.05)
    assert fit.index_offset_deg[1] == pytest.approx(-1.5, abs=0.05)
    # y los terminos fisicos salen igual de bien que sin offset
    assert fit.npae_deg == pytest.approx(TRUE["npae_deg"], abs=0.05)
    assert fit.collimation_deg == pytest.approx(TRUE["collimation_deg"], abs=0.05)


def test_without_azimuth_travel_the_tilt_is_not_identifiable() -> None:
    """Una inclinacion es un seno del azimut; sin recorrido no se separa.

    El error tiene que nombrar la direccion que falta, no ser un codigo pelado.
    """
    narrow = [(100.0 + i * 4.0, 30.0 + i * 6.0) for i in range(8)]

    fit = fit_geometry(_samples(narrow))

    assert fit.outcome is GeometryOutcome.ILL_CONDITIONED
    assert "azimut" in fit.detail
    assert fit.az_span_deg < 120.0


def test_without_a_high_sample_the_two_altitude_terms_collapse() -> None:
    """tan(alt) y sec(alt) casi coinciden abajo: hace falta subir para separarlos."""
    low = [(az, alt) for az, alt in _well_spread() if alt < 50.0]
    low += [(30.0, 20.0), (210.0, 35.0), (300.0, 45.0), (150.0, 40.0)]

    fit = fit_geometry(_samples(low))

    assert fit.outcome is GeometryOutcome.ILL_CONDITIONED
    assert "no-perpendicularidad" in fit.detail and "colimacion" in fit.detail
    assert fit.max_alt_deg < 60.0


def test_too_few_observations_is_its_own_answer() -> None:
    fit = fit_geometry(_samples(_well_spread()[:4]))

    assert fit.outcome is GeometryOutcome.NOT_ENOUGH_DATA
    assert fit.n_samples == 4


def test_a_physically_impossible_result_is_refused_not_clipped() -> None:
    """Como en los terminos: pegarlo a la cota lo vuelve indistinguible de bueno."""
    absurd = dict(tilt_north_deg=6.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=0.0)

    fit = fit_geometry(_samples(_well_spread(), truth=absurd))

    assert fit.outcome is GeometryOutcome.OUT_OF_BOUNDS
    assert "se rechaza en vez de recortarse" in fit.detail
    # el valor sigue visible para poder diagnosticarlo
    assert fit.base_tilt_deg[0] > 2.0


@pytest.mark.parametrize(
    "offset_deg, expected",
    [
        (0.02, GeometryOutcome.UNSTABLE),      # 72 arcsec: solo se ve por palanca
        (0.10, GeometryOutcome.UNSTABLE),
        (0.30, GeometryOutcome.POOR_FIT),      # ya no cabe en los residuos
        (0.80, GeometryOutcome.OUT_OF_BOUNDS), # empuja un termino fuera de lo posible
    ],
)
def test_a_single_bad_observation_is_caught_whatever_its_size(offset_deg, expected) -> None:
    """Tres guardias para tres regimenes, y ninguno deja pasar el outlier.

    El mas interesante es el pequeno: 72 arcsec no mueven el rms lo suficiente
    para notarse, pero si deciden el ajuste. Esa es la comprobacion que el codigo
    anterior hacia por fuerza bruta, probando cada muestra como referencia.
    """
    samples = _samples(_well_spread())
    bad = samples[3]
    samples[3] = GeometrySample(
        az_nominal_deg=bad.az_nominal_deg,
        alt_nominal_deg=bad.alt_nominal_deg,
        az_obs_deg=bad.az_obs_deg + offset_deg,
        alt_obs_deg=bad.alt_obs_deg - 0.7 * offset_deg,
        sigma_deg=bad.sigma_deg,
    )

    fit = fit_geometry(samples)

    assert fit.outcome is expected, fit.detail
    assert not fit.ok


def test_the_leverage_check_names_the_parameter_it_moves() -> None:
    """Saber que se rompio importa tanto como saber que se rompio algo."""
    samples = _samples(_well_spread())
    bad = samples[3]
    samples[3] = GeometrySample(
        az_nominal_deg=bad.az_nominal_deg,
        alt_nominal_deg=bad.alt_nominal_deg,
        az_obs_deg=bad.az_obs_deg + 0.05,
        alt_obs_deg=bad.alt_obs_deg - 0.035,
        sigma_deg=bad.sigma_deg,
    )

    fit = fit_geometry(samples)

    assert fit.outcome is GeometryOutcome.UNSTABLE
    assert "palanca" in fit.detail
    assert any(name in fit.detail for name in ("tilt_north", "tilt_west", "npae", "collimation"))


def test_a_perfect_mount_fits_to_zero() -> None:
    """Sin errores geometricos, los cuatro terminos tienen que salir cero."""
    ideal = dict(tilt_north_deg=0.0, tilt_west_deg=0.0, npae_deg=0.0, collimation_deg=0.0)

    fit = fit_geometry(_samples(_well_spread(), truth=ideal))

    assert fit.ok, fit.detail
    assert abs(fit.base_tilt_deg[0]) < 0.02
    assert abs(fit.base_tilt_deg[1]) < 0.02
    assert abs(fit.npae_deg) < 0.02
    assert abs(fit.collimation_deg) < 0.02


def test_azimuth_span_counts_the_wrap_around() -> None:
    """350 y 10 grados estan a 20 grados, no a 340."""
    wrapped = [(350.0, 30.0), (5.0, 65.0), (20.0, 40.0), (355.0, 70.0),
               (10.0, 25.0), (0.0, 55.0)]

    fit = fit_geometry(_samples(wrapped))

    assert fit.az_span_deg < 40.0, "el recorrido real es de unos 30 grados"
    assert fit.outcome is GeometryOutcome.ILL_CONDITIONED


def test_the_uncertainty_of_each_term_is_reported_by_name() -> None:
    fit = fit_geometry(_samples(_well_spread()))

    assert fit.ok
    assert set(fit.sigma) == {
        "index_az", "index_alt", "tilt_north", "tilt_west", "npae", "collimation"
    }
    assert all(v > 0.0 and math.isfinite(v) for v in fit.sigma.values())
