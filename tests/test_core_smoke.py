import unittest
import time
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils import iers

from config import AppConfig
from goto import (
    GoToConfig,
    GoToController,
    GoToModel,
    MountKinematics,
    _roll_deg_from_drift_delta,
    _roll_equivalent_near_reference_deg,
    _rotvec_deg_to_rotation_matrix,
    _rotate_altaz_deg,
    icrs_to_altaz_deg,
)
from platesolving import ObserverConfig, expected_field_rotation_deg, select_guide_star_indices
from stacking import StackEngine
from tracking import make_tracking_state, tracking_step, tracking_set_params


class CoreSmokeTests(unittest.TestCase):
    @staticmethod
    def _wrap_deg_180(angle_deg: float) -> float:
        return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)

    @classmethod
    def _theta_az_from_parallactic(
        cls,
        ra_deg: float,
        dec_deg: float,
        *,
        observer: ObserverConfig,
        obstime: Time,
    ) -> float:
        lst_deg = float(obstime.sidereal_time("apparent", longitude=observer.location().lon).deg)
        H_deg = cls._wrap_deg_180(lst_deg - float(ra_deg))
        H = np.deg2rad(H_deg)
        phi = np.deg2rad(float(observer.lat_deg))
        dec = np.deg2rad(float(dec_deg))
        q_deg = float(
            np.degrees(
                np.arctan2(
                    np.sin(H),
                    (np.tan(phi) * np.cos(dec)) - (np.sin(dec) * np.cos(H)),
                )
            )
        )
        return cls._wrap_deg_180(180.0 - q_deg)

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

    def test_tracking_pi_integrator_uses_frame_dt(self) -> None:
        state = make_tracking_state()
        state.auto.ok = True
        state.auto.A_pinv = np.eye(2, dtype=np.float64)
        state.auto.b = np.zeros(2, dtype=np.float64)
        raw = np.zeros((32, 32), dtype=np.uint16)
        objects = np.array([(10.0, 10.0, 1000.0)], dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")])

        with (
            patch("tracking.sep_detect_from_raw16", return_value=(None, None, objects, None)),
            patch("tracking.estimate_shift_from_profile_alignment", return_value=(1.0, 0.0, 1.0)),
            patch("tracking.auto_rls_update", return_value=None),
        ):
            tracking_step(state, raw, now_t=0.0, tracking_enabled=True)
            out = tracking_step(state, raw, now_t=0.1, tracking_enabled=True)

        self.assertTrue(np.isfinite(float(out.rate_az)))
        self.assertAlmostEqual(float(state.eint_x), 0.1, places=2)
        self.assertAlmostEqual(float(state.eint_y), 0.0, places=4)

    def test_stacking_engine_smoke(self) -> None:
        cfg = AppConfig()
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()
        raw = np.zeros((64, 64), dtype=np.uint16)
        engine.step_batch([{"raw16": raw, "t": 0.0}])
        engine.stop()
        self.assertIsNotNone(engine.metrics)

    def test_stacking_engine_rgb_mode_uses_rggb(self) -> None:
        cfg = AppConfig()
        cfg.stacking.color_mode = "rgb"
        cfg.stacking.bayer_pattern = "RGGB"
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()

        raw = np.zeros((32, 32), dtype=np.uint16)
        raw[0::2, 0::2] = 1000  # R
        raw[0::2, 1::2] = 2000  # G
        raw[1::2, 0::2] = 2000  # G
        raw[1::2, 1::2] = 3000  # B

        engine.step_batch([{"raw16": raw, "t": 0.0}])
        mean = engine.get_stack_mean(out_dtype=np.uint16)
        engine.stop()

        self.assertIsNotNone(mean)
        if mean is None:
            self.fail("mean stack should not be None in rgb mode")
        self.assertEqual(mean.ndim, 3)
        self.assertEqual(mean.shape[2], 3)

        r_mean = float(np.mean(mean[..., 0]))
        g_mean = float(np.mean(mean[..., 1]))
        b_mean = float(np.mean(mean[..., 2]))
        self.assertLess(r_mean, g_mean)
        self.assertLess(g_mean, b_mean)

    def test_stacking_engine_drizzle_x2_scales_mono_output(self) -> None:
        cfg = AppConfig()
        cfg.stacking.color_mode = "mono"
        cfg.stacking.drizzle_scale = 2.0
        cfg.stacking.bayer_pattern = "RGGB"
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()

        raw = np.zeros((24, 40), dtype=np.uint16)
        raw[10:14, 18:22] = 42000

        engine.step_batch([{"raw16": raw, "t": 0.0}])
        mean = engine.get_stack_mean(out_dtype=np.uint16)
        engine.stop()

        self.assertIsNotNone(mean)
        if mean is None:
            self.fail("mean stack should not be None in mono drizzle mode")
        self.assertEqual(mean.ndim, 2)
        self.assertEqual(mean.shape, (48, 80))

    def test_stacking_engine_drizzle_x1_keeps_native_size(self) -> None:
        cfg = AppConfig()
        cfg.stacking.color_mode = "mono"
        cfg.stacking.drizzle_scale = 1.0
        cfg.stacking.bayer_pattern = "RGGB"
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()

        raw = np.zeros((24, 40), dtype=np.uint16)
        raw[8:16, 16:24] = 30000

        engine.step_batch([{"raw16": raw, "t": 0.0}])
        mean = engine.get_stack_mean(out_dtype=np.uint16)
        engine.stop()

        self.assertIsNotNone(mean)
        if mean is None:
            self.fail("mean stack should not be None in drizzle off mode")
        self.assertEqual(mean.ndim, 2)
        self.assertEqual(mean.shape, (24, 40))

    def test_stacking_engine_drizzle_x3_scales_rgb_output(self) -> None:
        cfg = AppConfig()
        cfg.stacking.color_mode = "rgb"
        cfg.stacking.drizzle_scale = 3.0
        cfg.stacking.bayer_pattern = "RGGB"
        engine = StackEngine(cfg)
        engine.configure_from_cfg()
        engine.start()

        raw = np.zeros((24, 40), dtype=np.uint16)
        raw[0::2, 0::2] = 1000
        raw[0::2, 1::2] = 2000
        raw[1::2, 0::2] = 2000
        raw[1::2, 1::2] = 3000

        engine.step_batch([{"raw16": raw, "t": 0.0}])
        mean = engine.get_stack_mean(out_dtype=np.uint16)
        engine.stop()

        self.assertIsNotNone(mean)
        if mean is None:
            self.fail("mean stack should not be None in rgb drizzle mode")
        self.assertEqual(mean.ndim, 3)
        self.assertEqual(mean.shape, (72, 120, 3))

    def test_platesolving_guides_smoke(self) -> None:
        df = pd.DataFrame({"phot_g_mean_mag": [10.0, 11.0, 9.0]})
        idx = select_guide_star_indices(df, 2)
        self.assertEqual(idx, [2, 0])

    def test_expected_field_rotation_returns_finite(self) -> None:
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        observer = ObserverConfig(
            lat_deg=-33.3667,
            lon_deg=-71.6667,
            height_m=28.0,
            refraction_enable=False,
        )
        obstime = Time("2026-01-15T00:00:00", scale="utc")

        theta = expected_field_rotation_deg(
            120.0,
            -20.0,
            observer=observer,
            obstime=obstime,
            roll_offset_deg=0.0,
        )
        self.assertIsNotNone(theta)
        self.assertTrue(np.isfinite(float(theta)))

    def test_expected_field_rotation_matches_parallactic_relation(self) -> None:
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        observer = ObserverConfig(
            lat_deg=-33.3667,
            lon_deg=-71.6667,
            height_m=28.0,
            refraction_enable=False,
        )
        cases = (
            (120.0, -20.0, "2026-01-15T00:00:00"),
            (45.0, 10.0, "2026-02-01T06:00:00"),
            (300.0, -45.0, "2026-03-20T03:30:00"),
            (210.0, 30.0, "2026-04-10T12:00:00"),
        )

        for ra_deg, dec_deg, iso in cases:
            obstime = Time(iso, scale="utc")
            theta = expected_field_rotation_deg(
                ra_deg,
                dec_deg,
                observer=observer,
                obstime=obstime,
                roll_offset_deg=0.0,
            )
            self.assertIsNotNone(theta)
            if theta is None:
                self.fail("expected field rotation must be finite for nominal targets")
            theta_ref = self._theta_az_from_parallactic(
                ra_deg,
                dec_deg,
                observer=observer,
                obstime=obstime,
            )
            err_deg = abs(self._wrap_deg_180(float(theta) - float(theta_ref)))
            self.assertLess(err_deg, 1.0)

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

    def test_roll_equivalent_near_reference_prefers_continuous_branch(self) -> None:
        # Same axis orientation as +6 deg, but in opposite 180-deg branch.
        self.assertAlmostEqual(_roll_equivalent_near_reference_deg(-174.0, 0.0), 6.0, places=6)
        # If previous roll is near -180 branch, keep that branch.
        self.assertAlmostEqual(_roll_equivalent_near_reference_deg(6.0, -170.0), -174.0, places=6)
        # Exact branch tie keeps canonical wrapped branch.
        self.assertAlmostEqual(_roll_equivalent_near_reference_deg(45.0, -45.0), 45.0, places=6)

    def test_goto_model_manual_fit_keeps_unexcited_axis_column(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        j_before = model.J_deg_per_step.copy()

        base_steps = np.array([1000.0, 500.0], dtype=np.float64)
        base_az_alt = np.array([220.0, 58.0], dtype=np.float64)
        j_true_az_col = np.array([0.0012, -0.00005], dtype=np.float64)

        for d_az in np.array([-600.0, -300.0, 0.0, 350.0, 700.0], dtype=np.float64):
            d_steps = np.array([d_az, 0.0], dtype=np.float64)  # no ALT excitation
            model.steps_est = base_steps + d_steps
            d_altaz = j_true_az_col * d_az
            az = float(base_az_alt[0] + d_altaz[0])
            alt = float(base_az_alt[1] + d_altaz[1])
            model.add_manual_sample(np.array([az % 360.0, alt], dtype=np.float64), theta_deg=0.0)

        ok = model.fit_J_from_manual_samples(min_samples=3, ridge=1e-9)
        self.assertTrue(ok)
        self.assertGreater(abs(float(np.linalg.det(model.J_deg_per_step))), 1e-12)
        self.assertAlmostEqual(model.J_deg_per_step[0, 1], j_before[0, 1], places=12)
        self.assertAlmostEqual(model.J_deg_per_step[1, 1], j_before[1, 1], places=12)

    def test_goto_model_reset_manual_samples_and_sync(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        model.steps_est = np.array([1234.0, -567.0], dtype=np.float64)
        model.add_manual_sample(np.array([210.0, 42.0], dtype=np.float64), theta_deg=12.5)
        self.assertTrue(model.sync_from_latest_manual_sample())
        model.model_roll_deg = 5.0
        model.model_roll_err_deg = 1.0
        model.model_roll_samples = 1
        model.model_fit_samples = 3
        model.model_fit_rms_az_deg = 0.1
        model.model_fit_rms_alt_deg = 0.2
        model.model_fit_rms_arcsec = 100.0

        model.reset_manual_samples_and_sync()

        self.assertFalse(model.synced)
        self.assertEqual(len(model._manual_steps_abs), 0)
        self.assertEqual(len(model._manual_az_alt_abs), 0)
        self.assertEqual(len(model._manual_roll_deg_abs), 0)
        self.assertIsNone(model.last_solve_az_alt_deg)
        self.assertIsNone(model.last_solve_steps_est)
        self.assertEqual(model.last_solve_time, 0.0)
        np.testing.assert_allclose(model.ref_steps, model.steps_est)
        np.testing.assert_allclose(model.ref_az_alt_deg, np.zeros(2, dtype=np.float64))
        self.assertEqual(model.model_roll_deg, 0.0)
        self.assertEqual(model.model_roll_err_deg, 0.0)
        self.assertEqual(model.model_roll_samples, 0)
        self.assertEqual(model.model_fit_samples, 0)
        self.assertEqual(model.model_fit_rms_az_deg, 0.0)
        self.assertEqual(model.model_fit_rms_alt_deg, 0.0)
        self.assertEqual(model.model_fit_rms_arcsec, 0.0)

    def test_goto_model_restore_last_log_recovers_latest_state(self) -> None:
        prev_log_dir = os.environ.get("ASTROPANOPTES_GOTO_LOG_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = tmpdir

                source_model = GoToModel()
                source_model.init_from_mechanics()

                source_model.steps_est = np.array([1000.0, 2000.0], dtype=np.float64)
                source_model.add_manual_sample(np.array([120.0, 45.0], dtype=np.float64), theta_deg=5.0)
                self.assertTrue(source_model.sync_from_latest_manual_sample())

                source_model.steps_est = np.array([1350.0, 1780.0], dtype=np.float64)
                source_model.add_manual_sample(np.array([121.2, 44.7], dtype=np.float64), theta_deg=6.0)

                J_restore = np.array([[0.00123, 0.00011], [-0.00007, 0.00101]], dtype=np.float64)
                R_restore = _rotvec_deg_to_rotation_matrix(np.array([0.3, -0.2, 0.7], dtype=np.float64))

                source_model.J_deg_per_step = J_restore.copy()
                source_model.J00_err = 1.0e-6
                source_model.J01_err = 2.0e-6
                source_model.J10_err = 3.0e-6
                source_model.J11_err = 4.0e-6
                source_model.model_non_orthogonality_deg = 0.12
                source_model.model_non_orthogonality_err_deg = 0.03
                source_model.model_roll_deg = 6.5
                source_model.model_roll_err_deg = 0.4
                source_model.model_roll_samples = 2
                source_model.model_pitch_deg = 0.15
                source_model.model_pitch_err_deg = 0.02
                source_model.model_yaw_deg = -0.10
                source_model.model_yaw_err_deg = 0.02
                source_model.model_fit_samples = 2
                source_model.model_fit_rms_az_deg = 0.010
                source_model.model_fit_rms_alt_deg = 0.020
                source_model.model_fit_rms_arcsec = 22.0
                source_model.R_mount_to_world = R_restore.copy()
                source_model._log_fit_csv(
                    fit_kind="manual",
                    ok=True,
                    reason="OK",
                    min_samples=2,
                    ridge=1e-12,
                    total_samples=2,
                    used_samples=2,
                )

                restored = GoToModel()
                restored.init_from_mechanics()
                out = restored.restore_from_latest_logs()

                self.assertTrue(bool(out.get("ok", False)))
                self.assertEqual(str(out.get("status")), "OK")
                self.assertEqual(int(out.get("manual_samples", -1)), 2)
                self.assertAlmostEqual(float(out.get("camera_roll_deg", float("nan"))), 6.5, places=9)
                self.assertTrue(restored.synced)
                self.assertEqual(len(restored._manual_steps_abs), 2)
                np.testing.assert_allclose(
                    restored.ref_steps,
                    np.array([1000.0, 2000.0], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    restored.steps_est,
                    np.array([1350.0, 1780.0], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(restored.J_deg_per_step, J_restore, rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(restored.R_mount_to_world, R_restore, rtol=0.0, atol=1e-12)
                self.assertAlmostEqual(restored.model_fit_rms_arcsec, 22.0, places=9)
        finally:
            if prev_log_dir is None:
                os.environ.pop("ASTROPANOPTES_GOTO_LOG_DIR", None)
            else:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = prev_log_dir

    def test_goto_model_restore_last_log_handles_no_logs(self) -> None:
        prev_log_dir = os.environ.get("ASTROPANOPTES_GOTO_LOG_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = tmpdir
                model = GoToModel()
                model.init_from_mechanics()
                out = model.restore_from_latest_logs()
                self.assertFalse(bool(out.get("ok", False)))
                self.assertEqual(str(out.get("status")), "NO_LOGS")
                self.assertFalse(np.isfinite(float(out.get("camera_roll_deg", float("nan")))))
        finally:
            if prev_log_dir is None:
                os.environ.pop("ASTROPANOPTES_GOTO_LOG_DIR", None)
            else:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = prev_log_dir

    def test_goto_model_restore_last_log_uses_latest_manual_roll_when_no_fit(self) -> None:
        prev_log_dir = os.environ.get("ASTROPANOPTES_GOTO_LOG_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = tmpdir

                source_model = GoToModel()
                source_model.init_from_mechanics()
                source_model.steps_est = np.array([10.0, 20.0], dtype=np.float64)
                source_model.add_manual_sample(np.array([100.0, 30.0], dtype=np.float64), theta_deg=11.0)
                source_model.steps_est = np.array([15.0, 22.0], dtype=np.float64)
                source_model.add_manual_sample(np.array([100.2, 30.1], dtype=np.float64), theta_deg=12.0)

                restored = GoToModel()
                restored.init_from_mechanics()
                out = restored.restore_from_latest_logs()
                self.assertTrue(bool(out.get("ok", False)))
                self.assertAlmostEqual(float(out.get("camera_roll_deg", float("nan"))), 12.0, places=9)
        finally:
            if prev_log_dir is None:
                os.environ.pop("ASTROPANOPTES_GOTO_LOG_DIR", None)
            else:
                os.environ["ASTROPANOPTES_GOTO_LOG_DIR"] = prev_log_dir

    def test_goto_model_manual_fit_rejects_outlier_sample(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        j_true = np.array([[0.0013, 0.0004], [-0.0002, 0.0011]], dtype=np.float64)
        base_steps = np.array([4200.0, -1700.0], dtype=np.float64)
        base_az_alt = np.array([223.0, 58.5], dtype=np.float64)
        deltas = np.array(
            [
                [-1200.0, -700.0],
                [-700.0, 500.0],
                [-250.0, -300.0],
                [300.0, 250.0],
                [650.0, -450.0],
                [900.0, 700.0],
                [1200.0, 900.0],  # outlier injected here
            ],
            dtype=np.float64,
        )

        for i, (d_az, d_alt) in enumerate(deltas):
            d_steps = np.array([d_az, d_alt], dtype=np.float64)
            model.steps_est = base_steps + d_steps
            d_altaz = j_true @ d_steps
            az = float(base_az_alt[0] + d_altaz[0])
            alt = float(base_az_alt[1] + d_altaz[1])
            if i == len(deltas) - 1:
                az += 8.0
                alt -= 5.0
            model.add_manual_sample(np.array([az % 360.0, alt], dtype=np.float64), theta_deg=0.0)

        ok = model.fit_J_from_manual_samples(min_samples=5, ridge=1e-9)
        self.assertTrue(ok)
        self.assertGreaterEqual(model.model_fit_samples, 5)
        self.assertLess(model.model_fit_samples, len(deltas))
        self.assertAlmostEqual(model.J_deg_per_step[0, 0], j_true[0, 0], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[0, 1], j_true[0, 1], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[1, 0], j_true[1, 0], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[1, 1], j_true[1, 1], places=4)

    def test_goto_model_manual_fit_rejects_central_reference_outlier(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        j_true = np.array([[0.0012, 0.0003], [-0.0002, 0.0010]], dtype=np.float64)
        base_steps = np.array([0.0, 0.0], dtype=np.float64)
        base_az_alt = np.array([210.0, 40.0], dtype=np.float64)
        deltas = np.array(
            [
                [0.0, 0.0],  # central outlier candidate (high leverage reference)
                [10000.0, 0.0],
                [-10000.0, 0.0],
                [0.0, 10000.0],
                [0.0, -10000.0],
                [10000.0, 10000.0],
            ],
            dtype=np.float64,
        )

        for i, (d_az, d_alt) in enumerate(deltas):
            d_steps = np.array([d_az, d_alt], dtype=np.float64)
            model.steps_est = base_steps + d_steps
            d_altaz = j_true @ d_steps
            az = float(base_az_alt[0] + d_altaz[0])
            alt = float(base_az_alt[1] + d_altaz[1])
            if i == 0:
                az += 2.0
                alt -= 1.0
            model.add_manual_sample(np.array([az % 360.0, alt], dtype=np.float64), theta_deg=0.0)

        ok = model.fit_J_from_manual_samples(min_samples=5, ridge=1e-9)
        self.assertTrue(ok)
        self.assertEqual(model.model_fit_samples, 5)

        report = model.manual_samples_deviation_report(sort_by_deviation=True)
        self.assertEqual(len(report), len(deltas))
        outliers = [row for row in report if bool(row.get("outlier_suggested", False))]
        self.assertEqual(len(outliers), 1)
        self.assertEqual(int(outliers[0]["sample_idx"]), 0)

        inlier_devs = [float(row["dev_arcsec"]) for row in report if not bool(row.get("outlier_suggested", False))]
        self.assertGreater(len(inlier_devs), 0)
        self.assertLess(max(inlier_devs), 5.0)

    def test_goto_model_calibration_fit_rejects_outlier_sample(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        j_true = np.array([[0.0011, 0.0003], [-0.0001, 0.0010]], dtype=np.float64)
        steps = np.array(
            [
                [-900.0, -500.0],
                [-600.0, 400.0],
                [-200.0, -300.0],
                [250.0, 300.0],
                [700.0, -450.0],
                [1100.0, 800.0],
                [1300.0, 900.0],  # outlier injected here
            ],
            dtype=np.float64,
        )

        for i, s in enumerate(steps):
            d_altaz = j_true @ s
            if i == len(steps) - 1:
                d_altaz = d_altaz + np.array([6.0, -4.0], dtype=np.float64)
            model.add_calibration_sample(s, d_altaz)

        ok = model.fit_J_from_samples(min_samples=4, ridge=1e-9)
        self.assertTrue(ok)
        self.assertGreaterEqual(model.model_fit_samples, 4)
        self.assertLess(model.model_fit_samples, len(steps))
        self.assertAlmostEqual(model.J_deg_per_step[0, 0], j_true[0, 0], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[0, 1], j_true[0, 1], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[1, 0], j_true[1, 0], places=4)
        self.assertAlmostEqual(model.J_deg_per_step[1, 1], j_true[1, 1], places=4)

    def test_goto_parallel_move_dispatches_both_axes(self) -> None:
        ctrl = GoToController(cfg=GoToConfig(), model=GoToModel())
        calls: list[tuple[str, int, int, int, float]] = []

        def _fake_move(axis, direction, steps, delay_us):
            calls.append((axis.value, int(direction), int(steps), int(delay_us), time.perf_counter()))

        t0 = time.perf_counter()
        ctrl._exec_steps_parallel(
            _fake_move,
            dsteps_az=120.0,
            dsteps_alt=-90.0,
            delay_us_az=60,
            delay_us_alt=80,
            stop=None,
        )
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "az")
        self.assertEqual(calls[1][0], "alt")
        self.assertLess(abs(calls[1][4] - calls[0][4]), 0.01)
        self.assertGreater(elapsed, 0.01)

    def test_goto_model_manual_fit_rotation_tilt_is_limited(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        model.max_tilt_ns_oe_deg = 2.0

        j_true = np.array([[0.0011, 0.0002], [-0.0001, 0.0010]], dtype=np.float64)
        base_steps = np.array([2400.0, -800.0], dtype=np.float64)
        base_mount = np.array([170.0, 42.0], dtype=np.float64)
        # Intentionally above tilt limits; fit must clamp x/y to +/-2 deg.
        r_true = _rotvec_deg_to_rotation_matrix(np.array([5.0, -4.0, 12.0], dtype=np.float64))
        deltas = np.array(
            [
                [-900.0, -500.0],
                [-600.0, 400.0],
                [-200.0, -300.0],
                [250.0, 300.0],
                [700.0, -450.0],
                [1100.0, 800.0],
            ],
            dtype=np.float64,
        )

        for d_az, d_alt in deltas:
            d_steps = np.array([d_az, d_alt], dtype=np.float64)
            model.steps_est = base_steps + d_steps
            d_mount = j_true @ d_steps
            az_mount = float(base_mount[0] + d_mount[0]) % 360.0
            alt_mount = float(base_mount[1] + d_mount[1])
            world = _rotate_altaz_deg(np.array([az_mount, alt_mount], dtype=np.float64), r_true)
            model.add_manual_sample(world, theta_deg=0.0)

        ok = model.fit_J_from_manual_samples(min_samples=5, ridge=1e-9)
        self.assertTrue(ok)
        self.assertLessEqual(abs(float(model.model_pitch_deg)), 2.000001)
        self.assertLessEqual(abs(float(model.model_yaw_deg)), 2.000001)

    def test_goto_icrs_to_altaz_ignores_refraction_flag(self) -> None:
        coord = SkyCoord(ra=210.0, dec=-20.0, unit="deg", frame="icrs")
        t = Time("2026-02-13T03:00:00", format="isot", scale="utc")
        obs_no = ObserverConfig(
            lat_deg=-30.0,
            lon_deg=-70.0,
            height_m=1000.0,
            refraction_enable=False,
            refraction_P_hPa=850.0,
            refraction_T_C=5.0,
        )
        obs_yes = ObserverConfig(
            lat_deg=-30.0,
            lon_deg=-70.0,
            height_m=1000.0,
            refraction_enable=True,
            refraction_P_hPa=850.0,
            refraction_T_C=5.0,
        )
        altaz_no = icrs_to_altaz_deg(coord, observer=obs_no, obstime=t)
        altaz_yes = icrs_to_altaz_deg(coord, observer=obs_yes, obstime=t)
        self.assertAlmostEqual(float(altaz_no[0]), float(altaz_yes[0]), places=9)
        self.assertAlmostEqual(float(altaz_no[1]), float(altaz_yes[1]), places=9)

    def test_goto_model_sidereal_step_rate_is_finite_and_consistent(self) -> None:
        model = GoToModel()
        model.init_from_mechanics()
        observer = ObserverConfig(lat_deg=-30.0, lon_deg=-70.0, height_m=1000.0)
        obstime = Time("2026-02-13T03:00:00", format="isot", scale="utc")
        az_deg = 140.0
        alt_deg = 45.0

        world_rate = model.sidereal_world_rate_deg_s(
            az_deg=az_deg,
            alt_deg=alt_deg,
            observer=observer,
            obstime=obstime,
            dt_s=1.0,
        )
        self.assertIsNotNone(world_rate)
        if world_rate is None:
            self.fail("world sidereal rate should be available")

        step_rate = model.world_altaz_rate_to_step_rate_deg_s(
            az_deg=az_deg,
            alt_deg=alt_deg,
            world_rate_deg_s=world_rate,
            cond_max=1.0e6,
        )
        self.assertIsNotNone(step_rate)
        if step_rate is None:
            self.fail("step sidereal rate should be available")
        self.assertTrue(np.all(np.isfinite(step_rate)))
        self.assertLess(float(np.max(np.abs(step_rate))), 2000.0)

        J_world_step = model._world_deg_per_step_matrix(az_deg=az_deg, alt_deg=alt_deg)
        self.assertIsNotNone(J_world_step)
        if J_world_step is None:
            self.fail("world/step Jacobian should be available")
        world_rate_reconstructed = np.asarray(J_world_step, dtype=np.float64) @ np.asarray(step_rate, dtype=np.float64)
        np.testing.assert_allclose(world_rate_reconstructed, np.asarray(world_rate, dtype=np.float64), rtol=0.02, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
