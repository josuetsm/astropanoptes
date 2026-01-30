import unittest

import numpy as np
import pandas as pd

from config import AppConfig
from goto import GoToModel, MountKinematics
from platesolving import select_guide_star_indices
from stacking import StackEngine
from tracking import make_tracking_state, tracking_step


class CoreSmokeTests(unittest.TestCase):
    def test_tracking_step_smoke(self) -> None:
        state = make_tracking_state()
        obj = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        out = tracking_step(state, obj, now_t=0.0, tracking_enabled=False)
        self.assertTrue(out.ok)

        out2 = tracking_step(state, obj + 0.5, now_t=1.0, tracking_enabled=False)
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


if __name__ == "__main__":
    unittest.main()
