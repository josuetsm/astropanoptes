import numpy as np
import pytest
from astropy.time import Time
from ap_types import Axis

from goto import (
    GoToConfig,
    GoToController,
    GoToModel,
    _apply_roll_to_drift,
    _autocal_frame_is_crowded,
    _autocal_should_tune_exposure,
    _drift_to_az_alt,
    _predict_horizontal_tangent_rate,
    _solve_jcal_pointing,
    roll_axis_distance_deg,
)
from platesolving import ObserverConfig


@pytest.fixture(autouse=True)
def _isolate_goto_csv_logs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ASTROPANOPTES_GOTO_LOG_DIR", str(tmp_path / "goto_logs"))


def test_manual_fit_is_idempotent_with_noisy_samples() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    j_true = np.array(
        [[6.87e-4, 5.8e-6], [4.6e-6, 6.82e-4]],
        dtype=np.float64,
    )
    base_steps = np.array([40_000.0, -40_000.0], dtype=np.float64)
    base_altaz = np.array([210.0, 15.0], dtype=np.float64)
    deltas = np.array(
        [
            [-1800.0, -1200.0],
            [-900.0, 700.0],
            [-250.0, -400.0],
            [0.0, 0.0],
            [450.0, 550.0],
            [1200.0, -650.0],
            [2100.0, 1400.0],
        ],
        dtype=np.float64,
    )
    noise_arcsec = np.array(
        [
            [1.0, -2.0],
            [-1.5, 0.5],
            [2.0, 1.0],
            [-0.5, -1.0],
            [1.0, 1.5],
            [-2.0, 0.0],
            [0.5, -0.5],
        ],
        dtype=np.float64,
    )

    for dsteps, noise in zip(deltas, noise_arcsec):
        model.steps_est = base_steps + dsteps
        altaz = base_altaz + (j_true @ dsteps) + (noise / 3600.0)
        model.add_manual_sample(altaz, roll_deg=-0.13)

    assert model.fit_J_from_manual_samples(min_samples=5)
    j_first = model.J_deg_per_step.copy()
    r_first = model.R_mount_to_world.copy()
    rms_first = float(model.model_fit_rms_arcsec)

    assert model.fit_J_from_manual_samples(min_samples=5)
    np.testing.assert_allclose(model.J_deg_per_step, j_first, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(model.R_mount_to_world, r_first, rtol=0.0, atol=1e-12)
    assert abs(float(model.model_fit_rms_arcsec) - rms_first) < 1e-9


def test_manual_sample_continuity_rejects_multi_degree_jump_after_200_steps() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.steps_est = np.array([0.0, 0.0], dtype=np.float64)
    model.add_manual_sample(np.array([35.0, 69.5], dtype=np.float64), roll_deg=-5.0)

    model.steps_est = np.array([200.0, 0.0], dtype=np.float64)
    bad = model.manual_sample_continuity_report(
        np.array([42.0, 67.4], dtype=np.float64),
        roll_deg=32.0,
    )
    assert not bool(bad["ok"])
    assert not bool(bad["motion_ok"])
    assert not bool(bad["roll_ok"])
    assert float(bad["observed_motion_deg"]) > 2.0
    assert float(bad["motion_limit_deg"]) < 0.6

    plausible = model.manual_sample_continuity_report(
        np.array([35.12, 69.5], dtype=np.float64),
        roll_deg=-4.5,
    )
    assert bool(plausible["ok"])
    assert bool(plausible["motion_ok"])
    assert bool(plausible["roll_ok"])


def test_45_to_1_model_envelope_is_tight_and_signed() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    mechanical = model.mechanical_J()

    np.testing.assert_allclose(
        mechanical,
        np.diag([1.0 / 1600.0, 1.0 / 1600.0]),
        rtol=0.0,
        atol=1e-15,
    )
    assert model.is_J_within_mechanical_limits(mechanical)

    too_far = mechanical.copy()
    too_far[0, 0] *= 1.11
    assert not model.is_J_within_mechanical_limits(too_far)

    reversed_alt = mechanical.copy()
    reversed_alt[1, 1] *= -1.0
    assert not model.is_J_within_mechanical_limits(reversed_alt)

    coupled = mechanical.copy()
    coupled[1, 0] = 0.051 * mechanical[0, 0]
    assert not model.is_J_within_mechanical_limits(coupled)


def test_45_lobe_period_is_one_motor_revolution_at_fixed_microstepping() -> None:
    model = GoToModel()
    model.init_from_mechanics()

    assert model.kin.transmission_error_period_steps(Axis.AZ) == pytest.approx(12_800.0)
    assert model.kin.transmission_error_period_steps(Axis.ALT) == pytest.approx(12_800.0)


def test_periodic_transmission_model_keeps_global_45_to_1_and_inverts_locally() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.periodic_coeff_deg = np.array(
        [[0.08, -0.03], [0.04, 0.02]], dtype=np.float64
    )
    model.periodic_model_samples = 8
    start = np.array([1200.0, 2500.0], dtype=np.float64)
    intended_steps = np.array([3100.0, -1800.0], dtype=np.float64)
    desired = model.mount_delta_for_steps(start, start + intended_steps)

    recovered = model.solve_step_delta_for_mount_delta(desired, steps_from=start)

    np.testing.assert_allclose(recovered, intended_steps, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(
        model.mechanical_J(),
        np.diag([1.0 / 1600.0, 1.0 / 1600.0]),
        rtol=0.0,
        atol=1e-15,
    )


def test_manual_fit_separates_cycloidal_error_from_global_45_to_1_scale() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    mechanical = model.mechanical_J().copy()
    truth_coeff = np.array(
        [[0.045, -0.020], [0.030, 0.015]], dtype=np.float64
    )
    base_steps = np.array([18_000.0, -9_000.0], dtype=np.float64)
    base_altaz = np.array([185.0, 42.0], dtype=np.float64)
    deltas = np.array(
        [
            [-5200.0, -4100.0],
            [-3900.0, -2500.0],
            [-2600.0, 800.0],
            [-1300.0, 2700.0],
            [0.0, 0.0],
            [1300.0, -3100.0],
            [2600.0, -1400.0],
            [3900.0, 1900.0],
            [5200.0, 3900.0],
            [6400.0, 5100.0],
        ],
        dtype=np.float64,
    )

    for dsteps in deltas:
        absolute_steps = base_steps + dsteps
        linear = mechanical @ dsteps
        periodic = (
            model._periodic_offset_deg(absolute_steps, coeff=truth_coeff)
            - model._periodic_offset_deg(base_steps, coeff=truth_coeff)
        )
        model.steps_est = absolute_steps.copy()
        model.add_manual_sample(base_altaz + linear + periodic, roll_deg=-1.2)

    assert model.fit_J_from_manual_samples(min_samples=6)
    np.testing.assert_allclose(
        model.J_deg_per_step,
        mechanical,
        rtol=0.0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        model.periodic_coeff_deg,
        truth_coeff,
        rtol=0.0,
        atol=2e-3,
    )
    assert model.periodic_model_samples >= 8


def test_bad_manual_fit_is_rejected_without_mutating_previous_model() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    previous = model.J_deg_per_step.copy()
    bad_j = previous.copy()
    bad_j[0, 0] *= 1.30

    steps = np.array(
        [
            [-1200.0, -900.0],
            [-700.0, 500.0],
            [-200.0, -350.0],
            [300.0, 400.0],
            [850.0, -450.0],
            [1300.0, 800.0],
        ],
        dtype=np.float64,
    )
    for dsteps in steps:
        model.steps_est = dsteps.copy()
        model.add_manual_sample(
            np.array([210.0, 45.0], dtype=np.float64) + (bad_j @ dsteps),
            roll_deg=0.0,
        )

    assert not model.fit_J_from_manual_samples(min_samples=5)
    assert model.last_fit_reason == "MODEL_OUTSIDE_MECHANICAL_LIMITS"
    np.testing.assert_allclose(model.J_deg_per_step, previous, rtol=0.0, atol=0.0)


def test_unsafe_logged_fit_is_not_restored() -> None:
    source = GoToModel()
    source.init_from_mechanics()
    source.J_deg_per_step[0, 0] *= 1.25
    source._log_fit_csv(
        fit_kind="manual",
        ok=True,
        reason="OK",
        min_samples=3,
        ridge=1e-12,
        total_samples=3,
        used_samples=3,
    )

    restored = GoToModel()
    restored.init_from_mechanics()
    before = restored.J_deg_per_step.copy()
    result = restored.restore_from_latest_logs()

    assert not bool(result["ok"])
    assert result["status"] == "NO_VALID_ROWS"
    np.testing.assert_allclose(restored.J_deg_per_step, before, rtol=0.0, atol=0.0)


def test_restore_rejects_fit_that_conflicts_with_nominal_kinematics() -> None:
    source = GoToModel()
    source.init_from_mechanics()
    for idx, (steps, altaz) in enumerate(
        [
            ((0.0, 0.0), (130.54, 36.40)),
            ((60.0, 0.0), (130.82, 36.40)),
            ((60.0, 80.0), (130.82, 36.76)),
        ]
    ):
        source.steps_est = np.asarray(steps, dtype=np.float64)
        source.add_manual_sample(np.asarray(altaz, dtype=np.float64), roll_deg=-1.2)

    effective_j = np.array(
        [[0.00455, 0.000037], [0.000092, 0.00443]], dtype=np.float64
    )
    source.J_deg_per_step = effective_j.copy()
    source.model_fit_samples = 3
    source.model_roll_samples = 3
    source.model_roll_deg = -1.2
    source._log_fit_csv(
        fit_kind="manual",
        ok=True,
        reason="OK",
        min_samples=3,
        ridge=1e-12,
        total_samples=3,
        used_samples=3,
    )

    restored = GoToModel()
    restored.init_from_mechanics()
    result = restored.restore_from_latest_logs()

    assert bool(result["ok"])
    assert not bool(result["loaded_fit"])
    assert restored.model_fit_samples == 0
    assert restored.kin.gear_reduction_az == pytest.approx(45.0)
    assert restored.kin.gear_reduction_alt == pytest.approx(45.0)
    assert restored.kin.microsteps_az == 64
    assert restored.kin.microsteps_alt == 64


def test_restore_does_not_attach_fit_from_older_manual_session() -> None:
    old = GoToModel()
    old.init_from_mechanics()
    old.steps_est = np.array([0.0, 0.0])
    old.add_manual_sample(np.array([200.0, 20.0]), roll_deg=12.0)
    old.model_fit_samples = 1
    old.model_roll_samples = 1
    old.model_roll_deg = 12.0
    old._log_fit_csv(
        fit_kind="manual",
        ok=True,
        reason="OK",
        min_samples=1,
        ridge=1e-12,
        total_samples=1,
        used_samples=1,
    )

    latest = GoToModel()
    latest.init_from_mechanics()
    latest.steps_est = np.array([10.0, 20.0])
    latest.add_manual_sample(np.array([130.0, 36.0]), roll_deg=-1.0)
    latest.steps_est = np.array([30.0, 20.0])
    latest.add_manual_sample(np.array([130.1, 36.0]), roll_deg=-1.1)

    restored = GoToModel()
    restored.init_from_mechanics()
    result = restored.restore_from_latest_logs()

    assert bool(result["ok"])
    assert not bool(result["loaded_fit"])
    assert restored.model_fit_samples == 0
    assert float(result["camera_roll_deg"]) == pytest.approx(-1.1)


def test_backlash_takeup_on_restored_direction_is_not_counted_as_motion() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.steps_est = np.array([0.0, 0.0])
    model.add_manual_sample(np.array([130.0, 36.0]), roll_deg=-1.0)
    model.steps_est = np.array([0.0, 20.0])
    model.add_manual_sample(np.array([130.0, 36.09]), roll_deg=-1.0)
    controller = GoToController(
        cfg=GoToConfig(backlash_steps_az=0, backlash_steps_alt=10),
        model=model,
    )
    emitted = []

    controller._exec_steps_parallel(
        lambda axis, direction, steps, delay: emitted.append(
            (axis, direction, steps, delay)
        ),
        dsteps_az=0.0,
        dsteps_alt=-5.0,
        delay_us_az=1,
        delay_us_alt=1,
    )

    assert emitted == [
        (Axis.ALT, -1, 10, 1),
        (Axis.ALT, -1, 5, 1),
    ]
    np.testing.assert_allclose(model.steps_est, np.array([0.0, 15.0]))


def test_backlash_direction_and_amount_survive_log_restore() -> None:
    source = GoToModel()
    source.init_from_mechanics()
    source.backlash_steps_az = 3
    source.backlash_steps_alt = 12
    source.note_manual_move(Axis.AZ, -1, 40)
    source.note_manual_move(Axis.ALT, 1, 25)
    source.add_manual_sample(np.array([130.0, 36.0]), roll_deg=-1.0)

    restored = GoToModel()
    restored.init_from_mechanics()
    result = restored.restore_from_latest_logs()

    assert result["ok"]
    assert restored.last_move_direction(Axis.AZ) == -1
    assert restored.last_move_direction(Axis.ALT) == 1
    assert restored.backlash_steps_az == 3
    assert restored.backlash_steps_alt == 12


def test_single_axis_exec_applies_backlash_takeup_on_reversal() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.last_move_direction_alt = 1
    controller = GoToController(
        cfg=GoToConfig(backlash_steps_alt=10),
        model=model,
    )
    emitted = []

    controller._exec_steps(
        lambda axis, direction, steps, delay: emitted.append(
            (axis, direction, steps, delay)
        ),
        Axis.ALT,
        -5,
        delay_us=1,
    )

    assert emitted == [
        (Axis.ALT, -1, 10, 1),
        (Axis.ALT, -1, 5, 1),
    ]
    assert model.last_move_direction(Axis.ALT) == -1
    np.testing.assert_allclose(model.steps_est, np.array([0.0, -5.0]))


def test_high_rms_manual_fit_is_rejected_without_mutating_model() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    previous = model.J_deg_per_step.copy()
    steps = np.array(
        [
            [-16000.0, -12000.0],
            [-9000.0, 7000.0],
            [-3000.0, -5000.0],
            [3000.0, 5000.0],
            [9000.0, -7000.0],
            [16000.0, 12000.0],
        ],
        dtype=np.float64,
    )
    offsets = np.array(
        [
            [0.08, -0.06],
            [-0.08, 0.06],
            [0.07, 0.07],
            [-0.07, -0.07],
            [0.06, -0.08],
            [-0.06, 0.08],
        ],
        dtype=np.float64,
    )
    for dsteps, offset in zip(steps, offsets):
        model.steps_est = dsteps.copy()
        model.add_manual_sample(
            np.array([180.0, 50.0], dtype=np.float64)
            + (previous @ dsteps)
            + offset,
            roll_deg=0.0,
        )

    assert not model.fit_J_from_manual_samples(min_samples=5)
    assert model.last_fit_reason == "FIT_RMS_TOO_HIGH"
    np.testing.assert_allclose(model.J_deg_per_step, previous, rtol=0.0, atol=0.0)


def test_microstep_change_is_rejected_because_hardware_is_fixed() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.steps_est = np.array([3200.0, -1600.0], dtype=np.float64)
    model.sync_from_world_az_alt(np.array([200.0, 40.0], dtype=np.float64))
    model.steps_est += np.array([800.0, 400.0], dtype=np.float64)
    before_steps = model.steps_est.copy()
    before_ref = model.ref_steps.copy()
    before_j = model.J_deg_per_step.copy()

    with pytest.raises(ValueError, match="hardware-fixed"):
        model.set_microsteps(32, 32)

    np.testing.assert_allclose(model.steps_est, before_steps, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.ref_steps, before_ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.J_deg_per_step, before_j, rtol=0.0, atol=0.0)
    model.set_microsteps(64, 64)
    np.testing.assert_allclose(model.steps_est, before_steps, rtol=0.0, atol=0.0)


def test_quantized_active_pixels_do_not_force_exposure_down_with_sparse_sources() -> None:
    assert not _autocal_frame_is_crowded(
        active_fraction=0.113,
        star_count=39,
        max_sources=250,
    )
    assert _autocal_frame_is_crowded(
        active_fraction=0.113,
        star_count=240,
        max_sources=250,
    )


def test_manual_plate_solve_preserves_operator_exposure_by_default() -> None:
    assert not _autocal_should_tune_exposure({}, autocal_ps_mode="manual_altaz")
    assert not _autocal_should_tune_exposure({}, autocal_ps_mode="current_altaz")
    assert not _autocal_should_tune_exposure({}, autocal_ps_mode="drift")
    assert not _autocal_should_tune_exposure(
        {"tune_exposure": True},
        autocal_ps_mode="manual_altaz",
    )


@pytest.mark.parametrize(
    ("az_deg", "alt_deg"),
    [(20.0, 20.0), (90.0, 45.0), (150.0, 30.0), (220.0, 50.0), (300.0, 70.0)],
)
def test_drift_altaz_inversion_contains_forward_solution(az_deg: float, alt_deg: float) -> None:
    phi_deg = -33.3667
    omega = 15.041
    scale = 2.1
    phi = np.deg2rad(phi_deg)
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    vx = omega * (np.sin(phi) * np.cos(alt) - np.cos(phi) * np.sin(alt) * np.cos(az)) / scale
    vy = omega * np.cos(phi) * np.sin(az) / scale

    candidates = _drift_to_az_alt(
        float(vx),
        float(vy),
        phi_deg=phi_deg,
        omega_arcsec_s=omega,
        scale_arcsec_per_px=scale,
    )

    assert any(
        abs(((candidate_az - az_deg + 180.0) % 360.0) - 180.0) < 1e-6
        and abs(candidate_alt - alt_deg) < 1e-6
        for candidate_az, candidate_alt in candidates
    )


def test_roll_correction_recovers_az_axis_drift_components() -> None:
    drift_az_alt_axes = np.array([3.25, -1.75], dtype=np.float64)
    roll_deg = 37.0
    r = np.deg2rad(roll_deg)
    camera_drift = np.array(
        [
            np.cos(r) * drift_az_alt_axes[0] - np.sin(r) * drift_az_alt_axes[1],
            np.sin(r) * drift_az_alt_axes[0] + np.cos(r) * drift_az_alt_axes[1],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        _apply_roll_to_drift(camera_drift, roll_deg),
        drift_az_alt_axes,
        rtol=0.0,
        atol=1e-12,
    )


def test_roll_axis_comparison_treats_180_degree_branch_as_equivalent() -> None:
    assert roll_axis_distance_deg(-174.0, 6.0) == pytest.approx(0.0, abs=1e-12)
    assert roll_axis_distance_deg(79.0, -89.0) == pytest.approx(12.0, abs=1e-12)


def test_jcal_pointing_uses_horizontal_tangent_geometry() -> None:
    observer = ObserverConfig()
    obstime = Time("2026-08-12T10:03:59", scale="utc")
    true_az = 135.0
    true_alt = 45.0
    scale = np.deg2rad(0.6646 / 3600.0)
    tangent_rate = _predict_horizontal_tangent_rate(
        true_az,
        true_alt,
        observer=observer,
        obstime=obstime,
    )
    roll = np.deg2rad(4.0)
    axes = np.array(
        [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    coeff = -tangent_rate / scale
    drift = axes @ coeff
    J_pix = axes @ np.diag([4.2, 0.45])

    report = _solve_jcal_pointing(
        drift,
        J_pix,
        plate_scale_rad_per_px=scale,
        observer=observer,
        obstime=obstime,
        alt_min_deg=10.0,
        alt_max_deg=85.0,
        axis_sign_az=1,
        axis_sign_alt=1,
        seeds=[(132.0, 42.0)],
    )

    assert report["ok"]
    assert float(report["az_deg"]) == pytest.approx(true_az, abs=0.02)
    assert float(report["alt_deg"]) == pytest.approx(true_alt, abs=0.02)
    assert float(report["residual_rad_s"]) < 1e-8


def test_jcal_pointing_rejects_a_boundary_clip_as_a_solution() -> None:
    report = _solve_jcal_pointing(
        np.array([200.0, 200.0]),
        np.eye(2),
        plate_scale_rad_per_px=np.deg2rad(0.6646 / 3600.0),
        observer=ObserverConfig(),
        obstime=Time("2026-08-12T10:03:59", scale="utc"),
        alt_min_deg=10.0,
        alt_max_deg=85.0,
        seeds=[(90.0, 10.0), (270.0, 85.0)],
    )

    assert not report["ok"]
    assert report["status"] in {"NO_INTERIOR_SOLUTION", "HIGH_RESIDUAL"}


def test_single_axis_pair_never_rewrites_nominal_kinematics() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    model.add_manual_sample(np.array([129.70, 36.13]), roll_deg=-1.1)
    model.note_manual_move(Axis.AZ, 1, 100)

    report = model.bootstrap_axis_scale_from_manual_pair(
        np.array([130.15, 36.139]),
    )

    assert not report["ok"]
    assert report["status"] == "NOMINAL_KINEMATICS_FIXED"
    assert report["axis"] == "az"
    assert float(report["deg_per_step"]) == pytest.approx(0.0045, rel=1e-9)
    assert float(report["effective_steps_per_deg"]) == pytest.approx(222.2222, rel=1e-5)
    assert model.kin.gear_reduction_az == pytest.approx(45.0, rel=1e-9)
    assert model.kin.microsteps_az == 64
    assert model.kin.motor_full_steps_per_rev == 200
    assert model.J_deg_per_step[0, 0] == pytest.approx(0.000625, rel=1e-9)
