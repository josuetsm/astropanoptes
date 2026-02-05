from __future__ import annotations

from app_runner import AppRunner
from logging_utils import log_info

from ui.layout.platesolving import PlatesolvingPanelHandles
from ui.utils.debounce import DebouncedCall
from ui.utils.guard import RenderGuard


def bind_platesolving(
    platesolving_panel: PlatesolvingPanelHandles,
    runner: AppRunner,
    guard: RenderGuard,
    params_debouncer: DebouncedCall,
) -> None:
    def build_params() -> dict[str, object]:
        pixel_size_m = float(platesolving_panel.pixel_um.value) * 1e-6
        focal_m = float(platesolving_panel.focal_mm.value) / 1000.0
        return {
            "pixel_size_m": float(pixel_size_m),
            "focal_m": float(focal_m),
            "max_det": int(platesolving_panel.max_det.value),
            "N_det": int(platesolving_panel.n_det.value),
            "N_seed": int(platesolving_panel.n_seed.value),
            "det_thresh_sigma": float(platesolving_panel.det_sigma.value),
            "det_minarea": int(platesolving_panel.minarea.value),
            "point_sigma": float(platesolving_panel.point_sigma.value),
            "gmax": float(platesolving_panel.gmax.value),
            "search_radius_deg": float(platesolving_panel.search_radius_deg.value)
            if platesolving_panel.use_radius.value
            else None,
            "search_radius_factor": float(platesolving_panel.search_radius_factor.value),
            "theta_step_deg": float(platesolving_panel.theta_step.value),
            "theta_refine_span_deg": float(platesolving_panel.theta_refine_span.value),
            "theta_refine_step_deg": float(platesolving_panel.theta_refine_step.value),
            "match_max_px": float(platesolving_panel.match_max.value),
            "match_tol_arcsec": float(platesolving_panel.match_tol_arcsec.value),
            "pred_margin_arcsec": float(platesolving_panel.pred_margin_arcsec.value),
            "triplet_tol_arcsec": float(platesolving_panel.triplet_tol_arcsec.value),
            "triplet_sigma_arcsec": float(platesolving_panel.triplet_sigma_arcsec.value),
            "triplet_max_trials": int(platesolving_panel.triplet_max_trials.value),
            "max_i_scan": int(platesolving_panel.max_i_scan.value),
            "min_inliers": int(platesolving_panel.min_inliers.value),
            "guide_n": int(platesolving_panel.guide_n.value),
            "simbad_radius_arcsec": float(platesolving_panel.simbad_radius.value),
            "auto_solve": bool(platesolving_panel.auto_toggle.value),
            "solve_every_s": float(platesolving_panel.every_s.value),
            "auto_target": str(platesolving_panel.target.value),
        }

    def on_params_change(_change=None) -> None:
        if guard.active:
            return
        params_debouncer.trigger(lambda: runner.request_platesolving_params(**build_params()))

    for widget in [
        platesolving_panel.focal_mm,
        platesolving_panel.pixel_um,
        platesolving_panel.binning,
        platesolving_panel.max_det,
        platesolving_panel.n_det,
        platesolving_panel.n_seed,
        platesolving_panel.det_sigma,
        platesolving_panel.minarea,
        platesolving_panel.point_sigma,
        platesolving_panel.gmax,
        platesolving_panel.use_radius,
        platesolving_panel.search_radius_deg,
        platesolving_panel.search_radius_factor,
        platesolving_panel.theta_step,
        platesolving_panel.theta_refine_span,
        platesolving_panel.theta_refine_step,
        platesolving_panel.match_max,
        platesolving_panel.match_tol_arcsec,
        platesolving_panel.pred_margin_arcsec,
        platesolving_panel.triplet_tol_arcsec,
        platesolving_panel.triplet_sigma_arcsec,
        platesolving_panel.triplet_max_trials,
        platesolving_panel.max_i_scan,
        platesolving_panel.min_inliers,
        platesolving_panel.guide_n,
        platesolving_panel.simbad_radius,
        platesolving_panel.auto_toggle,
        platesolving_panel.every_s,
        platesolving_panel.target,
    ]:
        widget.observe(on_params_change, names="value")

    def request_once() -> None:
        target = str(platesolving_panel.target.value).strip()
        if not target:
            log_info(runner.out_log, "Platesolving: missing target")
            return
        runner.request_platesolving_run(target=target)

    platesolving_panel.btn_solve.on_click(lambda _btn: request_once())
