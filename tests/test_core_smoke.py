import unittest

import numpy as np
import pandas as pd

from config import AppConfig
from goto import GoToModel, MountKinematics, _roll_deg_from_drift_delta
from platesolving import select_guide_star_indices
from stacking import StackEngine
from tracking import make_tracking_state, tracking_step, tracking_set_params


class CoreSmokeTests(unittest.TestCase):
    def test_tracking_step_smoke(self) -> None:
        state = make_tracking_state()
        tracking_set_params(
            state,
            resp_min=0.0,
            sep_bw=16,
            sep_bh=16,
            sep_thresh_sigma=0.5,
            sep_minarea=1,
            sep_max_sources=50,
        )
        raw = np.zeros((32, 32), dtype=np.uint16)
        raw[8:11, 8:11] = 60000
        raw[20:23, 20:23] = 50000
        out = tracking_step(state, raw, now_t=0.0, tracking_enabled=False)
        self.assertTrue(out.ok)

        raw2 = np.zeros((32, 32), dtype=np.uint16)
        raw2[9:12, 9:12] = 60000
        raw2[21:24, 21:24] = 50000
        out2 = tracking_step(state, raw2, now_t=1.0, tracking_enabled=False)
        self.assertIsNotNone(out2)

    def test_stacking_engine_smoke(self) -> None:
        cfg = AppConfig()
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()
        raw = np.zeros((64, 64), dtype=np.uint16)
        engine.step_batch([{"raw16": raw, "t": 0.0}])
        engine.stop()
        self.assertIsNotNone(engine.metrics)

    def test_platesolving_guides_smoke(self) -> None:
        df = pd.DataFrame({"phot_g_mean_mag": [10.0, 11.0, 9.0]})
        idx = select_guide_star_indices(df, 2)
        self.assertEqual(idx, [2, 0])

    def test_goto_model_smoke(self) -> None:
        kin = MountKinematics(
            motor_full_steps_per_rev=200,
            microsteps_az=64,
            microsteps_alt=64,
            motor_pulley_teeth=20,
            ring_radius_m_az=0.24,
            ring_radius_m_alt=0.235,
        )
        model = GoToModel(kin=kin)
        model.init_from_mechanics()
        self.assertEqual(model.J_deg_per_step.shape, (2, 2))

    def test_goto_model_manual_fit_reports_params_and_errors(self) -> None:
        model = GoToModel()
        j_true = np.array([[0.0120, 0.0018], [0.0007, 0.0085]], dtype=np.float64)
        base_steps = np.array([5200.0, -2100.0], dtype=np.float64)
        base_az_alt = np.array([121.5, 38.25], dtype=np.float64)
        theta_true = 17.0
        rng = np.random.default_rng(7)

        deltas = np.array(
            [
                [-1200.0, -800.0],
                [-800.0, 600.0],
                [-200.0, -300.0],
                [300.0, 400.0],
                [900.0, -500.0],
                [1400.0, 900.0],
            ],
            dtype=np.float64,
        )
        for d_az, d_alt in deltas:
            d_steps = np.array([d_az, d_alt], dtype=np.float64)
            model.steps_est = base_steps + d_steps
            d_altaz = j_true @ d_steps
            az = float(base_az_alt[0] + d_altaz[0])
            alt = float(base_az_alt[1] + d_altaz[1])
            theta = theta_true + float(rng.normal(0.0, 0.15))
            model.add_manual_sample(np.array([az % 360.0, alt], dtype=np.float64), theta_deg=theta)

        ok = model.fit_J_from_manual_samples(min_samples=6, ridge=1e-9)
        self.assertTrue(ok)
        self.assertEqual(model.model_fit_samples, 6)
        self.assertEqual(model.model_roll_samples, 6)
        self.assertTrue(np.all(np.isfinite(model.J_deg_per_step)))
        self.assertAlmostEqual(model.J_deg_per_step[0, 0], j_true[0, 0], places=5)
        self.assertAlmostEqual(model.J_deg_per_step[0, 1], j_true[0, 1], places=5)
        self.assertAlmostEqual(model.J_deg_per_step[1, 0], j_true[1, 0], places=5)
        self.assertAlmostEqual(model.J_deg_per_step[1, 1], j_true[1, 1], places=5)
        self.assertTrue(np.isfinite(model.model_non_orthogonality_deg))
        self.assertTrue(np.isfinite(model.model_non_orthogonality_err_deg))
        self.assertTrue(np.isfinite(model.model_roll_deg))
        self.assertTrue(np.isfinite(model.model_roll_err_deg))
        self.assertEqual(model.model_yaw_deg, 0.0)
        self.assertEqual(model.model_yaw_err_deg, 0.0)
        self.assertEqual(model.model_pitch_deg, 0.0)
        self.assertEqual(model.model_pitch_err_deg, 0.0)
        self.assertTrue(np.isfinite(model.model_fit_rms_arcsec))
        self.assertTrue(np.isfinite(model.J00_err))
        self.assertTrue(np.isfinite(model.J11_err))

    def test_roll_deg_from_drift_delta_respects_slew_sign(self) -> None:
        # +30 deg with positive slew => +30
        dv_pos = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))], dtype=np.float64)
        roll_pos = _roll_deg_from_drift_delta(dv_pos, 25.0)
        self.assertAlmostEqual(roll_pos, 30.0, places=6)

        # +30 deg with negative slew flips measured vector, should still return +30
        dv_neg = -dv_pos
        roll_neg = _roll_deg_from_drift_delta(dv_neg, -25.0)
        self.assertAlmostEqual(roll_neg, 30.0, places=6)

        # -30 deg with negative slew should recover -30 (not +150)
        dv_left = np.array([np.cos(np.deg2rad(150.0)), np.sin(np.deg2rad(150.0))], dtype=np.float64)
        roll_left = _roll_deg_from_drift_delta(dv_left, -20.0)
        self.assertAlmostEqual(roll_left, -30.0, places=6)


if __name__ == "__main__":
    unittest.main()
