import numpy as np

from goto import GoToModel


def test_manual_fit_is_idempotent_with_noisy_samples() -> None:
    model = GoToModel()
    model.init_from_mechanics()
    j_true = np.array(
        [[7.47e-4, 5.8e-6], [4.6e-6, 7.62e-4]],
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
