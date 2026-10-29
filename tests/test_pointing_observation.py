"""El contrato de la admision de observaciones.

Sobre 109 corridas reales, el 61% no produjo medida utilizable, y no por el
modelo sino por la calidad de la observacion. Estos tests fijan que la puerta
decide por astrometria, que las dos razones de rechazo se distinguen en el
registro, y que lo rechazado tambien se guarda.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pointing.observation import (
    Admission,
    AdmissionReason,
    ObservationKind,
    ObservationStore,
    PointingObservation,
    SolveQuality,
    admit,
)


def _quality(**kw) -> SolveQuality:
    base = dict(
        n_detections=30,
        n_inliers=12,
        validation_inliers=6,
        rms_inliers_arcsec=0.9,
        arcsec_per_px=0.665,
        offset_from_prior_deg=0.2,
    )
    base.update(kw)
    return SolveQuality(**base)


def test_a_well_confirmed_solve_is_admitted() -> None:
    verdict = admit(_quality(), observed_motion_deg=1.0, expected_motion_deg=1.01,
                    motion_limit_deg=0.1)

    assert verdict.ok
    assert verdict.reason is AdmissionReason.ADMITTED


def test_a_solve_with_no_independent_confirmation_is_refused() -> None:
    """Tres inliers son los tres vertices del triplete: encaja consigo mismo.

    Es la mediana exacta de las 42 muestras que el proceso viejo rechazaba por
    geometria. Aqui se rechazan por lo que de verdad les pasa.
    """
    verdict = admit(_quality(n_inliers=3, validation_inliers=0, rms_inliers_arcsec=0.5))

    assert verdict.reason is AdmissionReason.LOW_CONFIDENCE
    assert "validacion" in verdict.detail


def test_a_ridiculously_low_rms_does_not_rescue_an_unconfirmed_solve() -> None:
    """El rms bajo es sintoma del ajuste exacto, no prueba de acierto.

    Se vieron coincidencias falsas con rms de 0.15 px, roll disparatado y
    apuntado a grados del real.
    """
    verdict = admit(_quality(n_inliers=3, validation_inliers=0, rms_inliers_arcsec=0.05))

    assert verdict.reason is AdmissionReason.LOW_CONFIDENCE


def test_many_inliers_with_tight_rms_pass_without_validation_stars() -> None:
    """Un campo pobre pero coherente sigue siendo admisible."""
    verdict = admit(_quality(n_inliers=8, validation_inliers=1, rms_inliers_arcsec=0.8))

    assert verdict.ok


def test_motion_and_roll_rejections_are_distinguishable_from_the_record_alone() -> None:
    """En 255 sesiones no se podia saber cual de las dos mitades habia saltado.

    Se calculaban los cuatro numeros y se tiraban detras de un unico string.
    """
    by_motion = admit(
        _quality(), observed_motion_deg=2.16, expected_motion_deg=0.30, motion_limit_deg=0.20
    )
    by_roll = admit(
        _quality(), observed_motion_deg=0.30, expected_motion_deg=0.30, motion_limit_deg=0.20,
        roll_jump_deg=9.0, roll_limit_deg=3.0,
    )

    assert by_motion.reason is AdmissionReason.PRIOR_MISMATCH_MOTION
    assert by_roll.reason is AdmissionReason.PRIOR_MISMATCH_ROLL
    assert by_motion.reason is not by_roll.reason

    # y los numeros viajan con el veredicto, no solo dentro del texto
    assert by_motion.observed_motion_deg == pytest.approx(2.16)
    assert by_motion.expected_motion_deg == pytest.approx(0.30)
    assert by_motion.motion_limit_deg == pytest.approx(0.20)
    assert by_roll.roll_jump_deg == pytest.approx(9.0)
    assert by_roll.roll_limit_deg == pytest.approx(3.0)


def test_astrometry_is_judged_before_geometry() -> None:
    """Un solve sin confirmar y ademas descolocado se rechaza por lo primero.

    Si se reportara el desajuste geometrico se ocultaria la causa real, que es
    que la astrometria no se sostiene.
    """
    verdict = admit(
        _quality(n_inliers=3, validation_inliers=0),
        observed_motion_deg=2.16, expected_motion_deg=0.30, motion_limit_deg=0.20,
    )

    assert verdict.reason is AdmissionReason.LOW_CONFIDENCE
    # pero los numeros de la geometria siguen registrados para poder mirarlos
    assert verdict.observed_motion_deg == pytest.approx(2.16)


def test_a_moving_mount_invalidates_the_exposure_whatever_the_solve_says() -> None:
    verdict = admit(_quality(), mount_moved_during_exposure=True)

    assert verdict.reason is AdmissionReason.MOUNT_MOVING


def test_missing_geometry_does_not_reject_by_itself() -> None:
    """La primera observacion de la noche no tiene prior contra el que comparar."""
    verdict = admit(_quality())

    assert verdict.ok


def test_confidence_orders_solves_by_independent_evidence() -> None:
    weak = _quality(validation_inliers=0)
    fair = _quality(validation_inliers=2, rms_inliers_arcsec=1.2)
    strong = _quality(validation_inliers=8, rms_inliers_arcsec=0.4)

    assert weak.confidence() == 0.0
    assert fair.confidence() < strong.confidence()
    assert 0.0 < strong.confidence() <= 1.0


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def _observation(**kw) -> PointingObservation:
    base = dict(
        t_unix=1789522125.0,
        kind=ObservationKind.PLATE_SOLVE,
        steps=(1234.0, -5678.0),
        load_dir=(1, -1),
        az_deg=258.87,
        alt_deg=39.41,
        sigma_arcsec=1.8,
        quality=_quality(),
        roll_deg=-3.5,
        roll_sigma_deg=0.4,
        source="stack_output/goto_diagnostics/x",
        admitted=True,
        reason=AdmissionReason.ADMITTED,
    )
    base.update(kw)
    return PointingObservation(**base)


def test_an_observation_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path)
    original = _observation()

    store.append(original)
    loaded = store.load(original.t_unix)

    assert len(loaded) == 1
    assert loaded[0] == original


def test_rejected_observations_are_kept_flagged(tmp_path: Path) -> None:
    """Son los datos con los que se afina la puerta; tirarlos es tirar la evidencia."""
    store = ObservationStore(tmp_path)
    store.append(_observation(admitted=True))
    store.append(
        _observation(admitted=False, reason=AdmissionReason.LOW_CONFIDENCE,
                     quality=_quality(validation_inliers=0))
    )

    everything = store.load(1789522125.0)
    admitted = store.load_admitted(1789522125.0)

    assert len(everything) == 2
    assert len(admitted) == 1
    assert everything[1].reason is AdmissionReason.LOW_CONFIDENCE


def test_a_session_across_midnight_stays_one_night(tmp_path: Path) -> None:
    """La noche va de mediodia a mediodia: 23:50 y 00:10 son la misma sesion."""
    store = ObservationStore(tmp_path)
    before = time.mktime(time.struct_time((2026, 9, 15, 23, 50, 0, 0, 0, -1)))
    after = time.mktime(time.struct_time((2026, 9, 16, 0, 10, 0, 0, 0, -1)))

    assert store.path_for(before) == store.path_for(after)


def test_a_corrupt_line_does_not_cost_the_whole_night(tmp_path: Path) -> None:
    store = ObservationStore(tmp_path)
    store.append(_observation())
    path = store.path_for(1789522125.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{esto no es json\n")
    store.append(_observation(az_deg=100.0))

    loaded = store.load(1789522125.0)

    assert len(loaded) == 2
    assert loaded[1].az_deg == pytest.approx(100.0)


def test_an_empty_night_is_not_an_error(tmp_path: Path) -> None:
    assert ObservationStore(tmp_path).load(1789522125.0) == []


def test_the_record_is_plain_json_anyone_can_read(tmp_path: Path) -> None:
    """Un JSONL versionado, no un CSV con esquema migrado a mano."""
    store = ObservationStore(tmp_path)
    store.append(_observation())

    line = store.path_for(1789522125.0).read_text(encoding="utf-8").splitlines()[0]
    data = json.loads(line)

    assert data["schema"] == 1
    assert data["kind"] == "plate_solve"
    assert data["steps"] == [1234.0, -5678.0]
    assert data["load_dir"] == [1, -1]
    assert data["quality"]["validation_inliers"] == 6


# ---------------------------------------------------------------------------
# Golden: la puerta contra las 109 corridas reales
# ---------------------------------------------------------------------------

QUALITY_FIXTURE = Path(__file__).parent / "data" / "autocal_solve_quality_2026-09.json"


def _real_runs():
    data = json.loads(QUALITY_FIXTURE.read_text(encoding="utf-8"))
    for run in data["runs"]:
        rms = run["rms_inliers_arcsec"]
        yield run["legacy_status"], SolveQuality(
            n_detections=run["n_detections"],
            n_inliers=run["n_inliers"],
            validation_inliers=run["validation_inliers"],
            rms_inliers_arcsec=float("inf") if rms is None else float(rms),
            offset_from_prior_deg=(
                float("nan") if run["target_offset_deg"] is None else run["target_offset_deg"]
            ),
        )


def test_the_gate_keeps_the_solid_two_thirds_of_what_the_old_one_accepted() -> None:
    """De las 43 que el proceso viejo admitio, 14 tenian <=1 inlier de validacion.

    No deberian haber entrado: son las que alimentaban los fits con rms alto y
    las que luego habia que podar como outliers.
    """
    accepted = [q for status, q in _real_runs() if "OK" in status]
    assert len(accepted) == 43

    admitted = [q for q in accepted if admit(q).ok]
    assert len(admitted) == 29
    assert all(admit(q).reason is AdmissionReason.LOW_CONFIDENCE
               for q in accepted if not admit(q).ok)


def test_the_gate_rejects_the_false_solves_for_an_astrometric_reason() -> None:
    """Las 42 que la guardia geometrica rechazaba: casi todas no se sostenian solas.

    La geometria acertaba al rechazarlas, pero por el motivo equivocado. Aqui
    caen por lo que de verdad les pasa, que es no tener confirmacion
    independiente.
    """
    rejected = [q for status, q in _real_runs() if "CONTINUITY" in status]
    assert len(rejected) == 42

    by_astrometry = [q for q in rejected if admit(q).reason is AdmissionReason.LOW_CONFIDENCE]
    assert len(by_astrometry) == 37


def test_a_handful_of_good_solves_were_rejected_by_geometry_alone() -> None:
    """Cinco tenian astrometria solida y aun asi la geometria las tiro.

    Esas son el caso en que la guardia se equivocaba: si la astrometria se
    confirma sola, lo sospechoso es el contador de pasos contra el que se
    comparo, no el solve. Por eso la continuidad es segunda opinion y no la
    primera.
    """
    rejected = [q for status, q in _real_runs() if "CONTINUITY" in status]

    would_admit = [q for q in rejected if admit(q).ok]
    assert len(would_admit) == 5
    assert all(q.validation_inliers >= 2 or q.n_inliers >= 6 for q in would_admit)


def test_every_solver_failure_stays_rejected() -> None:
    failed = [q for status, q in _real_runs() if "ERR_PLATESOLVING" in status]
    assert len(failed) == 24
    assert not any(admit(q).ok for q in failed)
