# goto_stages.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from goto_math import clamp, wrap_deg_180, wrap_deg_360
from goto_types import GoToConfig, GoToModel


@dataclass
class GoToStagePlan:
    dsteps: np.ndarray
    err_az_arcsec: float
    err_alt_arcsec: float
    done: bool


def plan_stage_move(
    cfg: GoToConfig,
    model: GoToModel,
    *,
    altaz_tgt: np.ndarray,
    altaz_cur: np.ndarray,
    stage_index: int,
    stages: int,
) -> GoToStagePlan:
    daz = wrap_deg_180(float(altaz_tgt[0]) - float(altaz_cur[0]))
    dalt = float(altaz_tgt[1]) - float(altaz_cur[1])

    err_az_arcsec = float(daz * 3600.0)
    err_alt_arcsec = float(dalt * 3600.0)

    if (abs(err_az_arcsec) <= float(cfg.tol_arcsec)) and (
        abs(err_alt_arcsec) <= float(cfg.tol_arcsec)
    ):
        return GoToStagePlan(
            dsteps=np.zeros(2, dtype=np.float64),
            err_az_arcsec=err_az_arcsec,
            err_alt_arcsec=err_alt_arcsec,
            done=True,
        )

    d_altaz_vec = np.array([daz, dalt], dtype=np.float64)
    J = model.J_deg_per_step
    invJ = np.linalg.inv(J)
    dsteps = invJ @ d_altaz_vec

    remaining = max(1, int(stages) - int(stage_index))
    stage_scale = 1.0 / float(remaining)
    dsteps *= stage_scale

    dsteps *= float(cfg.gain)

    dsteps = np.clip(
        dsteps,
        -float(cfg.max_step_per_iter),
        +float(cfg.max_step_per_iter),
    )

    pred_after = altaz_cur.copy()
    pred_after[0] = wrap_deg_360(float(pred_after[0]) + float((J @ dsteps)[0]))
    pred_after[1] = float(pred_after[1]) + float((J @ dsteps)[1])

    if pred_after[1] < float(cfg.alt_min_deg) or pred_after[1] > float(cfg.alt_max_deg):
        alt_target = clamp(pred_after[1], cfg.alt_min_deg, cfg.alt_max_deg)
        delta_alt_allowed = float(alt_target - float(altaz_cur[1]))
        dalt_pred = float((J @ dsteps)[1])
        if abs(dalt_pred) > 1e-12:
            alpha = float(delta_alt_allowed / dalt_pred)
            alpha = clamp(alpha, -1.0, 1.0)
            dsteps *= alpha

    return GoToStagePlan(
        dsteps=dsteps,
        err_az_arcsec=err_az_arcsec,
        err_alt_arcsec=err_alt_arcsec,
        done=False,
    )
