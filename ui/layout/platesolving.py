from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as W

from config import AppConfig


@dataclass
class PlatesolvingPanelHandles:
    target: W.Text
    btn_solve: W.Button
    auto_toggle: W.ToggleButton
    every_s: W.BoundedFloatText
    status: W.HTML
    image: W.Image
    debug_html: W.HTML
    focal_mm: W.BoundedFloatText
    pixel_um: W.BoundedFloatText
    binning: W.BoundedIntText
    max_det: W.BoundedIntText
    n_det: W.BoundedIntText
    n_seed: W.BoundedIntText
    det_sigma: W.BoundedFloatText
    minarea: W.BoundedIntText
    point_sigma: W.BoundedFloatText
    gmax: W.BoundedFloatText
    use_radius: W.Checkbox
    search_radius_deg: W.BoundedFloatText
    search_radius_factor: W.BoundedFloatText
    theta_step: W.BoundedFloatText
    theta_refine_span: W.BoundedFloatText
    theta_refine_step: W.BoundedFloatText
    match_max: W.BoundedFloatText
    match_tol_arcsec: W.BoundedFloatText
    pred_margin_arcsec: W.BoundedFloatText
    triplet_tol_arcsec: W.BoundedFloatText
    triplet_sigma_arcsec: W.BoundedFloatText
    triplet_max_trials: W.BoundedIntText
    max_i_scan: W.BoundedIntText
    min_inliers: W.BoundedIntText
    guide_n: W.BoundedIntText
    simbad_radius: W.BoundedFloatText


def build_platesolving_panel(cfg: AppConfig) -> tuple[W.Widget, PlatesolvingPanelHandles]:
    platesolving_cfg = cfg.platesolving

    focal_mm = W.BoundedFloatText(
        description="focal (mm)",
        value=float(platesolving_cfg.focal_m) * 1000.0,
        min=10.0,
        max=50000.0,
        step=1.0,
        layout=W.Layout(width="260px"),
    )
    pixel_um = W.BoundedFloatText(
        description="pixel (µm)",
        value=float(platesolving_cfg.pixel_size_m) * 1e6,
        min=0.5,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    binning = W.BoundedIntText(
        description="binning",
        value=1,
        min=1,
        max=8,
        step=1,
        layout=W.Layout(width="200px"),
    )
    binning.disabled = True

    max_det = W.BoundedIntText(
        description="max_det",
        value=int(platesolving_cfg.max_det),
        min=20,
        max=2000,
        step=10,
        layout=W.Layout(width="220px"),
    )
    n_det = W.BoundedIntText(
        description="N_det",
        value=int(platesolving_cfg.N_det),
        min=1,
        max=2000,
        step=1,
        layout=W.Layout(width="220px"),
    )
    n_seed = W.BoundedIntText(
        description="N_seed",
        value=int(platesolving_cfg.N_seed),
        min=0,
        max=50,
        step=1,
        layout=W.Layout(width="220px"),
    )
    det_sigma = W.BoundedFloatText(
        description="det_sigma",
        value=float(platesolving_cfg.det_thresh_sigma),
        min=0.5,
        max=50.0,
        step=0.5,
        layout=W.Layout(width="220px"),
    )
    minarea = W.BoundedIntText(
        description="minarea",
        value=int(platesolving_cfg.det_minarea),
        min=1,
        max=200,
        step=1,
        layout=W.Layout(width="220px"),
    )
    point_sigma = W.BoundedFloatText(
        description="point_sigma",
        value=float(platesolving_cfg.point_sigma),
        min=0.2,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )
    gmax = W.BoundedFloatText(
        description="gmax",
        value=float(platesolving_cfg.gmax),
        min=6.0,
        max=20.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )
    use_radius = W.Checkbox(
        description="use search_radius_deg",
        value=platesolving_cfg.search_radius_deg is not None,
    )
    search_radius_deg = W.BoundedFloatText(
        description="search_radius_deg",
        value=float(platesolving_cfg.search_radius_deg or 2.0),
        min=0.1,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    search_radius_factor = W.BoundedFloatText(
        description="search_radius_factor",
        value=float(platesolving_cfg.search_radius_factor),
        min=0.5,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    theta_step = W.BoundedFloatText(
        description="theta_step (deg)",
        value=float(platesolving_cfg.theta_step_deg),
        min=0.5,
        max=60.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    theta_refine_span = W.BoundedFloatText(
        description="theta_refine_span",
        value=float(platesolving_cfg.theta_refine_span_deg),
        min=0.5,
        max=60.0,
        step=0.5,
        layout=W.Layout(width="260px"),
    )
    theta_refine_step = W.BoundedFloatText(
        description="theta_refine_step",
        value=float(platesolving_cfg.theta_refine_step_deg),
        min=0.1,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    match_max = W.BoundedFloatText(
        description="match_max_px",
        value=float(platesolving_cfg.match_max_px),
        min=0.5,
        max=25.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    match_tol_arcsec = W.BoundedFloatText(
        description="match_tol_arcsec",
        value=float(platesolving_cfg.match_tol_arcsec),
        min=0.1,
        max=60.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    pred_margin_arcsec = W.BoundedFloatText(
        description="pred_margin_arcsec",
        value=float(platesolving_cfg.pred_margin_arcsec),
        min=0.0,
        max=300.0,
        step=1.0,
        layout=W.Layout(width="260px"),
    )
    triplet_tol_arcsec = W.BoundedFloatText(
        description="triplet_tol_arcsec",
        value=float(platesolving_cfg.triplet_tol_arcsec),
        min=0.1,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    triplet_sigma_arcsec = W.BoundedFloatText(
        description="triplet_sigma_arcsec",
        value=float(platesolving_cfg.triplet_sigma_arcsec),
        min=0.1,
        max=10.0,
        step=0.1,
        layout=W.Layout(width="260px"),
    )
    triplet_max_trials = W.BoundedIntText(
        description="triplet_max_trials",
        value=int(platesolving_cfg.triplet_max_trials),
        min=10,
        max=20000,
        step=10,
        layout=W.Layout(width="260px"),
    )
    max_i_scan = W.BoundedIntText(
        description="max_i_scan",
        value=int(platesolving_cfg.max_i_scan),
        min=10,
        max=20000,
        step=10,
        layout=W.Layout(width="260px"),
    )
    min_inliers = W.BoundedIntText(
        description="min_inliers",
        value=int(platesolving_cfg.min_inliers),
        min=1,
        max=200,
        step=1,
        layout=W.Layout(width="260px"),
    )
    guide_n = W.BoundedIntText(
        description="guide_n",
        value=int(platesolving_cfg.guide_n),
        min=0,
        max=20,
        step=1,
        layout=W.Layout(width="220px"),
    )
    simbad_radius = W.BoundedFloatText(
        description='simbad_radius"',
        value=float(platesolving_cfg.simbad_radius_arcsec),
        min=0.1,
        max=30.0,
        step=0.1,
        layout=W.Layout(width="220px"),
    )

    target = W.Text(
        description="target",
        value="",
        placeholder="Ej: 'M42' | '12.5 5.3' (RA/Dec) | ...",
        layout=W.Layout(width="740px"),
    )
    btn_solve = W.Button(description="Solve", button_style="success", layout=W.Layout(width="120px"))

    auto_toggle = W.ToggleButton(
        description="Auto",
        value=bool(platesolving_cfg.auto_solve),
        disabled=False,
        layout=W.Layout(width="110px"),
    )
    every_s = W.BoundedFloatText(
        description="every (s)",
        value=float(platesolving_cfg.solve_every_s),
        min=2.0,
        max=600.0,
        step=1.0,
        layout=W.Layout(width="220px"),
    )

    status = W.HTML(value="Platesolving: idle")
    image = W.Image(format="jpeg", layout=W.Layout(width="100%", max_width="980px"))
    debug_html = W.HTML(value="")

    widget = W.VBox(
        [
            W.HTML("<b>Platesolving</b>"),
            W.HBox([btn_solve, auto_toggle, every_s]),
            target,
            W.HTML("<b>Instrument</b>"),
            W.HBox([focal_mm, pixel_um, binning]),
            W.HTML("<b>Solver</b>"),
            W.HBox([max_det, n_det, n_seed, det_sigma]),
            W.HBox([minarea, point_sigma, gmax, use_radius]),
            W.HBox([search_radius_deg, search_radius_factor, theta_step, theta_refine_span]),
            W.HBox([theta_refine_step, match_max, match_tol_arcsec, pred_margin_arcsec]),
            W.HBox([triplet_tol_arcsec, triplet_sigma_arcsec, triplet_max_trials, max_i_scan]),
            W.HBox([min_inliers, guide_n, simbad_radius]),
            status,
            debug_html,
            image,
        ],
        layout=W.Layout(border="1px solid #eee", padding="8px", gap="6px"),
    )

    handles = PlatesolvingPanelHandles(
        target=target,
        btn_solve=btn_solve,
        auto_toggle=auto_toggle,
        every_s=every_s,
        status=status,
        image=image,
        debug_html=debug_html,
        focal_mm=focal_mm,
        pixel_um=pixel_um,
        binning=binning,
        max_det=max_det,
        n_det=n_det,
        n_seed=n_seed,
        det_sigma=det_sigma,
        minarea=minarea,
        point_sigma=point_sigma,
        gmax=gmax,
        use_radius=use_radius,
        search_radius_deg=search_radius_deg,
        search_radius_factor=search_radius_factor,
        theta_step=theta_step,
        theta_refine_span=theta_refine_span,
        theta_refine_step=theta_refine_step,
        match_max=match_max,
        match_tol_arcsec=match_tol_arcsec,
        pred_margin_arcsec=pred_margin_arcsec,
        triplet_tol_arcsec=triplet_tol_arcsec,
        triplet_sigma_arcsec=triplet_sigma_arcsec,
        triplet_max_trials=triplet_max_trials,
        max_i_scan=max_i_scan,
        min_inliers=min_inliers,
        guide_n=guide_n,
        simbad_radius=simbad_radius,
    )
    return widget, handles
