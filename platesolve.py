# platesolve.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import cv2
import sep

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

from sklearn.neighbors import KDTree
from itertools import combinations, permutations

from logging_utils import log_error
from config import PlatesolveConfig

# IMPORTANT: all Gaia/cache/auth logic must live in gaia_cache.py
import gaia_cache as gc


# ============================================================
# Public API surface
# ============================================================

__all__ = [
    "ObserverConfig",
    "PlatesolveConfig",
    "OverlayItem",
    "GuideStar",
    "PlatesolveResult",
    "TargetParseError",
    "platesolve_sweep",
    "platesolve_from_live",
    "platesolve_from_stack",
    "pixel_to_radec",
]


ProgressCB = Callable[[str, Dict[str, Any]], None]


# ============================================================
# Errors
# ============================================================

class PlatesolveError(RuntimeError):
    pass


class TargetParseError(PlatesolveError):
    pass


# ============================================================
# Config / Data types
# ============================================================

@dataclass(frozen=True)
class ObserverConfig:
    """
    Default observer: Santiago, Chile (approx).
    Used for AltAz -> ICRS conversion.
    """
    lat_deg: float = -33.4489
    lon_deg: float = -70.6693
    height_m: float = 520.0

    def location(self) -> EarthLocation:
        return EarthLocation(
            lat=self.lat_deg * u.deg,
            lon=self.lon_deg * u.deg,
            height=self.height_m * u.m,
        )


@dataclass(frozen=True)
class OverlayItem:
    x: float
    y: float
    kind: str                  # "det", "match", "guide"
    label: Optional[str] = None


@dataclass(frozen=True)
class GuideStar:
    name: str
    ra_deg: float
    dec_deg: float
    gmag: float
    x: float
    y: float


@dataclass(frozen=True)
class PlatesolveResult:
    success: bool
    status: str
    theta_deg: float
    dx_px: float
    dy_px: float
    response: float

    # similarity / plate model (close to your notebook logic)
    # scale (arcsec/px), rotation matrix (2x2), translation (2,)
    scale_arcsec_per_px: float
    R_2x2: Tuple[Tuple[float, float], Tuple[float, float]]
    t_arcsec: Tuple[float, float]  # translation in TAN arcsec space

    n_inliers: int
    rms_arcsec: float
    rms_px: float

    center_ra_deg: float
    center_dec_deg: float

    downsample: int

    overlay: List[OverlayItem]
    guides: List[GuideStar]
    metrics: Dict[str, float]


# ============================================================
# Target parsing (ICRS/J2000, AltAz, name via gaia_cache/simbad if you keep it there)
# ============================================================

TargetType = Union[
    SkyCoord,
    Tuple[float, float],
    Tuple[str, str],
    str,
    Dict[str, Any],
]


def parse_target_to_icrs(
    target: TargetType,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time],
) -> SkyCoord:
    if obstime is None:
        obstime = Time.now()

    if isinstance(target, SkyCoord):
        return target.icrs

    if isinstance(target, dict):
        # ICRS degrees
        if ("ra_deg" in target and "dec_deg" in target) or ("ra" in target and "dec" in target):
            ra = float(target.get("ra_deg", target.get("ra")))
            dec = float(target.get("dec_deg", target.get("dec")))
            return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        # AltAz degrees
        if "alt_deg" in target and "az_deg" in target:
            alt = float(target["alt_deg"])
            az = float(target["az_deg"])
            t = obstime
            if target.get("obstime"):
                t = Time(str(target["obstime"]))
            loc = observer.location()
            altaz = AltAz(alt=alt * u.deg, az=az * u.deg, obstime=t, location=loc)
            return SkyCoord(altaz).icrs

        raise TargetParseError(f"Unrecognized dict target keys: {list(target.keys())}")

    if isinstance(target, (tuple, list)) and len(target) == 2:
        a, b = target[0], target[1]
        return SkyCoord(ra=float(a) * u.deg, dec=float(b) * u.deg, frame="icrs")

    if isinstance(target, str):
        s = target.strip()
        if not s:
            raise TargetParseError("Empty target string.")

        # If it contains letters, treat as name. Delegate to gaia_cache if it provides a resolver.
        if any(ch.isalpha() for ch in s):
            if hasattr(gc, "resolve_name_to_icrs"):
                return gc.resolve_name_to_icrs(s).icrs
            raise TargetParseError("Name targets require gaia_cache.resolve_name_to_icrs(name).")

        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            ra_s, dec_s = parts[0], parts[1]
            if ":" in ra_s:
                return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
            return SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")

        raise TargetParseError(f"Could not parse target string: {s}")

    raise TargetParseError(f"Unsupported target type: {type(target).__name__}")


# ============================================================
# Image: SEP detection (closer to your notebook logic)
# ============================================================

def _to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    f = int(max(1, factor))
    if f == 1:
        return img
    return np.ascontiguousarray(img[::f, ::f])


def stretch01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def median3_u16(img: np.ndarray) -> np.ndarray:
    """cv2.medianBlur requires uint8/uint16"""
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img_u = img
    else:
        # be conservative: if float, clip to uint16 range
        if np.issubdtype(img.dtype, np.floating):
            x = np.asarray(img)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.clip(x, 0.0, 65535.0)
            img_u = x.astype(np.uint16)
        else:
            img_u = img.astype(np.uint16)
    return cv2.medianBlur(img_u, 3).astype(np.float32)


def detect_sep_objects(
    raw_gray: np.ndarray,
    *,
    sep_bw: int,
    sep_bh: int,
    sep_thresh_sigma: float,
    sep_minarea: int,
    max_sources: int,
    progress_cb: Optional[ProgressCB],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      disp      : stretch image [0..1]
      obj_xy    : (N,2) xy in pixels sorted by flux desc (top max_sources)
      obj_flux  : (N,) flux sorted desc
    """
    if progress_cb:
        progress_cb("detect:start", {})

    raw = np.asarray(raw_gray)
    disp = stretch01(raw)

    img_med = median3_u16(raw)
    bkg = sep.Background(img_med, bw=int(sep_bw), bh=int(sep_bh))
    img_sub = img_med - bkg.back()
    img_det = np.maximum(img_sub, 0.0)

    thresh = float(sep_thresh_sigma) * float(bkg.globalrms)
    objects = sep.extract(img_det, thresh, minarea=int(sep_minarea))

    if objects is None or len(objects) == 0:
        if progress_cb:
            progress_cb("detect:empty", {"thresh": float(thresh), "globalrms": float(bkg.globalrms)})
        return disp.astype(np.float32), np.zeros((0, 2), np.float64), np.zeros((0,), np.float64)

    # sort by flux desc
    order = np.argsort(-objects["flux"].astype(np.float64))
    objects = objects[order]

    # cap sources
    n_use = min(int(max_sources), len(objects))
    objects = objects[:n_use]

    x = objects["x"].astype(np.float64)
    y = objects["y"].astype(np.float64)
    flux = objects["flux"].astype(np.float64)

    if progress_cb:
        progress_cb("detect:done", {"n": int(n_use), "thresh": float(thresh), "globalrms": float(bkg.globalrms)})

    return disp.astype(np.float32), np.column_stack([x, y]), flux


# ============================================================
# Spherical helpers (your notebook logic)
# ============================================================

def unitvec_from_radec(ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ]).astype(np.float64)


def chord_radius(theta_rad: float) -> float:
    return float(2.0 * np.sin(float(theta_rad) / 2.0))


def annulus_candidates(
    tree: KDTree,
    V: np.ndarray,
    center_idx: int,
    theta_arcsec: float,
    tol_arcsec: float,
) -> np.ndarray:
    theta_min = ((float(theta_arcsec) - float(tol_arcsec)) * u.arcsec).to(u.rad).value
    theta_max = ((float(theta_arcsec) + float(tol_arcsec)) * u.arcsec).to(u.rad).value
    theta_min = max(float(theta_min), 0.0)

    r_max = chord_radius(theta_max)
    r_min = chord_radius(theta_min)

    idxs = tree.query_radius(V[center_idx:center_idx + 1], r=r_max, return_distance=False)[0]
    if idxs.size == 0:
        return idxs

    dots = V[idxs] @ V[center_idx]
    chord2 = 2.0 - 2.0 * dots
    return idxs[chord2 >= (r_min * r_min)]


def sorted_sides_arcsec_from_pixels(xy3: np.ndarray, arcsec_per_pixel: float) -> np.ndarray:
    (x1, y1), (x2, y2), (x3, y3) = xy3
    d12 = np.hypot(x2 - x1, y2 - y1) * float(arcsec_per_pixel)
    d23 = np.hypot(x3 - x2, y3 - y2) * float(arcsec_per_pixel)
    d31 = np.hypot(x1 - x3, y1 - y3) * float(arcsec_per_pixel)
    return np.sort(np.array([d12, d23, d31], dtype=np.float64))


def sorted_sides_arcsec_from_coords(coords: SkyCoord, i: int, j: int, k: int) -> np.ndarray:
    s1 = coords[i].separation(coords[j]).to_value(u.arcsec)
    s2 = coords[j].separation(coords[k]).to_value(u.arcsec)
    s3 = coords[k].separation(coords[i]).to_value(u.arcsec)
    return np.sort(np.array([s1, s2, s3], dtype=np.float64))


def triplet_score(img_sides: np.ndarray, cat_sides: np.ndarray, sigma_arcsec: float) -> Tuple[float, float]:
    errs = cat_sides - img_sides
    score = float(np.sum((errs / float(sigma_arcsec)) ** 2))
    err_max = float(np.max(np.abs(errs)))
    return score, err_max


def sph_centroid(skycoords: SkyCoord) -> SkyCoord:
    ra = skycoords.ra.to_value(u.rad)
    dec = skycoords.dec.to_value(u.rad)
    V = unitvec_from_radec(ra, dec)
    m = V.mean(axis=0)
    n = float(np.linalg.norm(m))
    if n <= 0:
        # fallback to first coord
        return SkyCoord(ra=skycoords[0].ra, dec=skycoords[0].dec, frame="icrs")
    m /= n
    dec0 = float(np.arcsin(m[2]))
    ra0 = float(np.arctan2(m[1], m[0]) % (2 * np.pi))
    return SkyCoord(ra=ra0 * u.rad, dec=dec0 * u.rad, frame="icrs")


def procrustes_similarity(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    2D similarity: Q ≈ s*(P @ R.T) + t
    P: (N,2) px ; Q: (N,2) arcsec
    Returns (s, R, t, rms) with det(R)=+1 enforced (no reflection)
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    varP = float(np.sum(Pc ** 2))
    s = float(np.sum(S) / (varP + 1e-12))
    t = Q.mean(axis=0) - (s * (P.mean(axis=0) @ R.T))

    Qhat = (s * (P @ R.T)) + t
    rms = float(np.sqrt(np.mean(np.sum((Q - Qhat) ** 2, axis=1))))
    return s, R, t, rms


def best_assignment_similarity(img_pts3: np.ndarray, cat_pts3: np.ndarray) -> Dict[str, Any]:
    best = None
    for perm in permutations(range(3)):
        Q = cat_pts3[list(perm)]
        s, R, t, rms = procrustes_similarity(img_pts3, Q)
        cand = (rms, perm, s, R, t)
        if best is None or cand[0] < best[0]:
            best = cand
    rms, perm, s, R, t = best
    return {"rms": float(rms), "perm": perm, "s": float(s), "R": R, "t": t}


def apply_similarity(Pxy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (float(s) * (Pxy @ R.T)) + t


def inverse_similarity(Qxy: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return ((Qxy - t) / float(s)) @ R


def one_to_one_match(pred_xy: np.ndarray, cat_xy: np.ndarray, radius_arcsec: float) -> List[Tuple[int, int, float]]:
    """
    Greedy 1–1 matching by distance (no duplicates).
    pred_xy: (Nd,2) in arcsec
    cat_xy : (Ng,2) in arcsec
    Returns list of (det_idx, cat_idx, dist_arcsec)
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    cat_xy = np.asarray(cat_xy, dtype=np.float64)

    Nd = int(pred_xy.shape[0])
    Ng = int(cat_xy.shape[0])
    if Nd == 0 or Ng == 0:
        return []

    tree = KDTree(cat_xy, leaf_size=40, metric="euclidean")
    ind, dist = tree.query_radius(
        pred_xy,
        r=float(radius_arcsec),
        return_distance=True,
        sort_results=True
    )

    edges: List[Tuple[float, int, int]] = []
    for det_i, (cats, ds) in enumerate(zip(ind, dist)):
        if len(cats) == 0:
            continue
        for c, d in zip(cats, ds):
            edges.append((float(d), int(det_i), int(c)))

    if not edges:
        return []

    edges.sort(key=lambda x: x[0])

    used_det = np.zeros(Nd, dtype=bool)
    used_cat = np.zeros(Ng, dtype=bool)

    matches: List[Tuple[int, int, float]] = []
    for d, det_i, cat_i in edges:
        if used_det[det_i] or used_cat[cat_i]:
            continue
        used_det[det_i] = True
        used_cat[cat_i] = True
        matches.append((det_i, cat_i, float(d)))

    return matches


# ============================================================
# Guides (delegated: optional, depends on gaia_cache providing a name resolver)
# ============================================================

def select_guide_star_indices(df_gaia: pd.DataFrame, n: int) -> List[int]:
    n = int(max(1, n))
    if "phot_g_mean_mag" in df_gaia.columns and len(df_gaia) > 0:
        idx = np.argsort(df_gaia["phot_g_mean_mag"].to_numpy(np.float64))
        return [int(i) for i in idx[: min(n, len(idx))].tolist()]
    return [int(i) for i in range(min(n, len(df_gaia)))]


def build_guides_from_solution(
    df_gaia: pd.DataFrame,
    guide_idx: List[int],
    *,
    center_icrs: SkyCoord,
    s_arcsec_per_px: float,
    R: np.ndarray,
    t_arcsec: np.ndarray,
    downsample: int,
    cfg: PlatesolveConfig,
    progress_cb: Optional[ProgressCB],
) -> List[GuideStar]:
    """
    Converts selected Gaia stars to pixel positions using the inverse similarity.
    Naming is delegated to gaia_cache if it provides resolve_coord_name().
    """
    guides: List[GuideStar] = []
    if len(df_gaia) == 0:
        return guides

    coords = SkyCoord(
        ra=np.asarray(df_gaia["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(df_gaia["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )

    # Gaia to TAN arcsec around center
    d_lon, d_lat = center_icrs.spherical_offsets_to(coords)
    cat_all = np.column_stack([d_lon.to_value(u.arcsec), d_lat.to_value(u.arcsec)])

    for i in guide_idx:
        ra = float(df_gaia.at[i, "ra"])
        dec = float(df_gaia.at[i, "dec"])
        gmag = float(df_gaia.at[i, "phot_g_mean_mag"]) if "phot_g_mean_mag" in df_gaia.columns else float("nan")

        # arcsec -> px (in downsampled pixel grid)
        px = inverse_similarity(cat_all[i:i + 1], float(s_arcsec_per_px), R, t_arcsec)[0]
        x_ds, y_ds = float(px[0]), float(px[1])

        # convert to full-res pixels if desired by app; for overlay we keep DS coords
        name = "GAIA"
        if hasattr(gc, "resolve_coord_name"):
            try:
                name = str(gc.resolve_coord_name(SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"), cfg=cfg))
            except Exception:
                name = "GAIA"

        guides.append(GuideStar(name=name, ra_deg=ra, dec_deg=dec, gmag=gmag, x=x_ds, y=y_ds))

    return guides


# ============================================================
# Pixel -> RA/Dec helpers (from similarity fit + TAN center)
# ============================================================

def pixel_to_radec(
    x_px: float,
    y_px: float,
    *,
    center_icrs: SkyCoord,
    s_arcsec_per_px: float,
    R: np.ndarray,
    t_arcsec: np.ndarray,
) -> SkyCoord:
    """
    Given a pixel position (in the same pixel coordinate system used to fit, i.e. downsampled),
    return ICRS SkyCoord using the inverse similarity:
      px -> arcsec offsets -> SkyCoord offset frame -> ICRS
    """
    P = np.array([[float(x_px), float(y_px)]], dtype=np.float64)
    # forward similarity: px -> arcsec
    Q_arcsec = apply_similarity(P, float(s_arcsec_per_px), R, t_arcsec)[0]
    u_as, v_as = float(Q_arcsec[0]), float(Q_arcsec[1])

    off_frame = center_icrs.skyoffset_frame()
    c_off = SkyCoord(lon=u_as * u.arcsec, lat=v_as * u.arcsec, frame=off_frame)
    return c_off.icrs


# ============================================================
# Main solver: Triplet candidates + 1–1 inliers (your notebook logic)
# ============================================================

def _gaia_load_df(
    center_icrs: SkyCoord,
    radius_deg: float,
    *,
    cfg: PlatesolveConfig,
    gaia_auth: Optional[Tuple[str, str]],
    progress_cb: Optional[ProgressCB],
) -> pd.DataFrame:
    """
    Thin adapter: Gaia/cache/auth is owned by gaia_cache.py.
    Expect it to return a DataFrame or an Astropy Table convertible to DF.
    """
    if gaia_auth is None and hasattr(gc, "load_gaia_auth"):
        gaia_auth = gc.load_gaia_auth()

    out = gc.gaia_healpix_cone_with_mag(
        center_icrs=center_icrs,
        radius_deg=float(radius_deg),
        cfg=cfg,
        auth=gaia_auth,
        progress_cb=progress_cb,
    )

    if isinstance(out, pd.DataFrame):
        df = out
    else:
        df = out.to_pandas()

    want = [c for c in ["source_id", "ra", "dec", "phot_g_mean_mag"] if c in df.columns]
    df = df.loc[:, want].dropna(subset=["ra", "dec"]).reset_index(drop=True)
    return df


def platesolve_sweep(
    frame: np.ndarray,
    *,
    target: TargetType,
    cfg: PlatesolveConfig,
    observer: ObserverConfig = ObserverConfig(),
    obstime: Optional[Time] = None,
    source: str = "live",
    gaia_auth: Optional[Tuple[str, str]] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> PlatesolveResult:
    if obstime is None:
        obstime = Time.now()

    # 1) Parse target -> ICRS center
    try:
        center_icrs = parse_target_to_icrs(target, observer=observer, obstime=obstime)
    except Exception as exc:
        log_error(None, "Platesolve: target parse failed", exc)
        return PlatesolveResult(
            success=False,
            status="TARGET_PARSE_ERROR",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=0.0,
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=0.0,
            center_dec_deg=0.0,
            downsample=int(getattr(cfg, "downsample", 1)),
            overlay=[],
            guides=[],
            metrics={"source": 1.0},
        )

    # 2) Prepare frame (gray + downsample)
    gray = _to_gray(frame)
    ds = int(max(1, getattr(cfg, "downsample", 1)))
    gray_ds = _downsample(gray, ds)
    h, w = gray_ds.shape[:2]

    # 3) Detect stars (SEP)
    disp, img_xy_all, img_flux_all = detect_sep_objects(
        gray_ds,
        sep_bw=int(getattr(cfg, "sep_bw", 64)),
        sep_bh=int(getattr(cfg, "sep_bh", 64)),
        sep_thresh_sigma=float(getattr(cfg, "sep_thresh_sigma", getattr(cfg, "det_thresh_sigma", 3.0))),
        sep_minarea=int(getattr(cfg, "sep_minarea", getattr(cfg, "det_minarea", 5))),
        max_sources=int(getattr(cfg, "max_det", 200)),
        progress_cb=progress_cb,
    )

    overlay: List[OverlayItem] = [OverlayItem(float(x), float(y), "det", None) for x, y in img_xy_all]

    if img_xy_all.shape[0] < 3:
        return PlatesolveResult(
            success=False,
            status="NOT_ENOUGH_DETECTIONS",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=0.0,
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"n_det": float(img_xy_all.shape[0])},
        )

    # 4) Plate scale (arcsec/px) at full-res, then adjust for downsample
    # pixel_size_m expected in cfg; if not, allow pixel_um + focal_mm fallback
    if hasattr(cfg, "pixel_size_m") and hasattr(cfg, "focal_m"):
        arcsec_per_px = 206265.0 * (float(cfg.pixel_size_m)) / float(cfg.focal_m)
    elif hasattr(cfg, "pixel_um") and hasattr(cfg, "focal_mm"):
        arcsec_per_px = 206265.0 * (float(cfg.pixel_um) * 1e-3) / float(cfg.focal_mm)
    else:
        # last resort: require explicit
        arcsec_per_px = float(getattr(cfg, "arcsec_per_px", 1.0))

    arcsec_per_px_ds = float(arcsec_per_px) * float(ds)

    # 5) Gaia radius: prefer cfg.search_radius_deg else estimate from FOV
    def _estimate_radius_deg() -> float:
        diag_px = float(np.hypot(w, h))
        factor = float(getattr(cfg, "search_radius_factor", 1.15))
        radius_as = factor * (diag_px / 2.0) * float(arcsec_per_px_ds)
        return float(max(0.4, radius_as / 3600.0))

    radius_deg = float(getattr(cfg, "search_radius_deg", None) or _estimate_radius_deg())

    # 6) Load Gaia from gaia_cache.py
    try:
        if progress_cb:
            progress_cb("gaia:load:start", {"radius_deg": float(radius_deg), "gmax": float(getattr(cfg, "gmax", 15.0)), "source": source})

        gaia_df = _gaia_load_df(center_icrs, radius_deg, cfg=cfg, gaia_auth=gaia_auth, progress_cb=progress_cb)

    except gc.NeedGaiaAuthError as e:
        return PlatesolveResult(
            success=False,
            status="NEED_GAIA_AUTH",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"missing_tiles": float(getattr(e, "missing_tiles", 0))},
        )

    except gc.GaiaCacheMissError as e:
        missing_paths = getattr(e, "missing_paths", [])
        return PlatesolveResult(
            success=False,
            status="GAIA_CACHE_MISS",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"missing": float(len(missing_paths))},
        )

    except Exception as exc:
        log_error(None, "Platesolve: Gaia load failed", exc)
        return PlatesolveResult(
            success=False,
            status="GAIA_LOAD_ERROR",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"err": 1.0},
        )

    if len(gaia_df) < int(max(8, 3 * img_xy_all.shape[0])):
        return PlatesolveResult(
            success=False,
            status="GAIA_TOO_SMALL",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"gaia_rows": float(len(gaia_df))},
        )

    # 7) Build SkyCoord + 3D KDTree on unit vectors
    coords = SkyCoord(
        ra=np.asarray(gaia_df["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(gaia_df["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    V = unitvec_from_radec(coords.ra.to_value(u.rad), coords.dec.to_value(u.rad))
    tree3 = KDTree(V, leaf_size=40, metric="euclidean")

    # 8) Seeds / validation subsets
    N_det = int(getattr(cfg, "N_det", getattr(cfg, "match_n_det", 30)))
    N_seed = int(getattr(cfg, "N_seed", getattr(cfg, "match_n_seed", 3)))

    img_xy_all = img_xy_all[: min(N_det, img_xy_all.shape[0])] if N_det > 0 else img_xy_all
    img_flux_all = img_flux_all[: img_xy_all.shape[0]]

    N_seed_eff = min(int(max(3, N_seed)), img_xy_all.shape[0])
    img_xy_seed = img_xy_all[:N_seed_eff]

    if img_xy_seed.shape[0] < 3:
        return PlatesolveResult(
            success=False,
            status="NOT_ENOUGH_SEEDS",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"n_seed": float(img_xy_seed.shape[0])},
        )

    # 9) Triplets in image seeds
    img_triplets: List[Tuple[int, int, int, np.ndarray]] = []
    for (a, b, c) in combinations(range(img_xy_seed.shape[0]), 3):
        sides = sorted_sides_arcsec_from_pixels(img_xy_seed[[a, b, c]], arcsec_per_pixel=arcsec_per_px_ds)
        img_triplets.append((a, b, c, sides))

    # 10) Candidate generation via annuli on 3D KDTree
    tol_arcsec_pairs = float(getattr(cfg, "tol_arcsec_pairs", getattr(cfg, "triplet_tol_arcsec", 3.0)))
    sigma_arcsec = float(getattr(cfg, "sigma_arcsec", getattr(cfg, "triplet_sigma_arcsec", 0.6)))
    max_trials = int(getattr(cfg, "max_trials", getattr(cfg, "triplet_max_trials", 500)))

    candidates: List[Dict[str, Any]] = []
    if progress_cb:
        progress_cb("platesolve:triplets:start", {"n_triplets": int(len(img_triplets))})

    # Performance safeguard: cap how many Gaia "i" centers we scan per triplet
    # (otherwise O(N^2) can explode at high gmax/radius)
    max_i_scan = int(getattr(cfg, "max_i_scan", 2000))
    i_scan = np.arange(V.shape[0])
    if V.shape[0] > max_i_scan:
        # bias towards bright Gaia if available
        if "phot_g_mean_mag" in gaia_df.columns:
            mags = np.asarray(gaia_df["phot_g_mean_mag"], dtype=np.float64)
            i_scan = np.argsort(mags)[:max_i_scan]
        else:
            i_scan = i_scan[:max_i_scan]

    for (a, b, c, img_sides) in img_triplets:
        d1, d2, d3 = float(img_sides[0]), float(img_sides[1]), float(img_sides[2])

        theta_min = ((d3 - tol_arcsec_pairs) * u.arcsec).to(u.rad).value
        theta_max = ((d3 + tol_arcsec_pairs) * u.arcsec).to(u.rad).value
        theta_min = max(float(theta_min), 0.0)

        r_max = chord_radius(theta_max)
        r_min = chord_radius(theta_min)

        for i in i_scan:
            nbrs = tree3.query_radius(V[i:i + 1], r=r_max, return_distance=False)[0]
            nbrs = nbrs[nbrs > i]  # avoid duplicate pairs
            if nbrs.size == 0:
                continue

            dots = V[nbrs] @ V[i]
            chord2 = 2.0 - 2.0 * dots
            nbrs = nbrs[chord2 >= (r_min * r_min)]
            if nbrs.size == 0:
                continue

            for j in nbrs:
                candA = np.intersect1d(
                    annulus_candidates(tree3, V, int(i), d1, tol_arcsec_pairs),
                    annulus_candidates(tree3, V, int(j), d2, tol_arcsec_pairs),
                    assume_unique=False
                )
                candB = np.intersect1d(
                    annulus_candidates(tree3, V, int(i), d2, tol_arcsec_pairs),
                    annulus_candidates(tree3, V, int(j), d1, tol_arcsec_pairs),
                    assume_unique=False
                )
                ks = np.union1d(candA, candB)
                if ks.size == 0:
                    continue
                ks = ks[(ks != i) & (ks != j)]
                if ks.size == 0:
                    continue

                for k in ks:
                    cat_sides = sorted_sides_arcsec_from_coords(coords, int(i), int(j), int(k))
                    score, err_max = triplet_score(img_sides, cat_sides, sigma_arcsec=sigma_arcsec)
                    if err_max <= tol_arcsec_pairs:
                        candidates.append({
                            "score": float(score),
                            "err_max": float(err_max),
                            "img_triplet": (int(a), int(b), int(c)),
                            "gaia_idx": (int(i), int(j), int(k)),
                            "gaia_source_id": (
                                int(gaia_df["source_id"].iloc[int(i)]) if "source_id" in gaia_df.columns else int(i),
                                int(gaia_df["source_id"].iloc[int(j)]) if "source_id" in gaia_df.columns else int(j),
                                int(gaia_df["source_id"].iloc[int(k)]) if "source_id" in gaia_df.columns else int(k),
                            ),
                        })

    candidates.sort(key=lambda d: d["score"])

    if progress_cb:
        progress_cb("platesolve:triplets:candidates", {"n_candidates": int(len(candidates))})

    if len(candidates) == 0:
        return PlatesolveResult(
            success=False,
            status="NO_TRIPLET_CANDIDATES",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"n_candidates": 0.0},
        )

    # 11) Validate top candidates with 1–1 inliers
    match_tol_arcsec = float(getattr(cfg, "match_tol_arcsec", 5.0))
    pred_margin_arcsec = float(getattr(cfg, "pred_margin_arcsec", match_tol_arcsec + 20.0))

    def evaluate_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
        a, b, c = cand["img_triplet"]
        i, j, k = cand["gaia_idx"]

        img_tri = img_xy_seed[[a, b, c]]
        tri_coords = coords[[i, j, k]]
        center0 = sph_centroid(tri_coords)

        # Gaia triplet in TAN plane (arcsec)
        d_lon, d_lat = center0.spherical_offsets_to(tri_coords)
        cat_tri = np.column_stack([d_lon.to_value(u.arcsec), d_lat.to_value(u.arcsec)])

        fit = best_assignment_similarity(img_tri, cat_tri)

        # ALL Gaia into TAN plane
        d_lon_all, d_lat_all = center0.spherical_offsets_to(coords)
        cat_all = np.column_stack([d_lon_all.to_value(u.arcsec), d_lat_all.to_value(u.arcsec)])

        # ALL detections -> TAN via similarity
        pred_all = apply_similarity(img_xy_all, fit["s"], fit["R"], fit["t"])

        # crop Gaia around predicted bbox
        xmin, ymin = pred_all.min(axis=0) - pred_margin_arcsec
        xmax, ymax = pred_all.max(axis=0) + pred_margin_arcsec
        keep = (
            (cat_all[:, 0] >= xmin) & (cat_all[:, 0] <= xmax) &
            (cat_all[:, 1] >= ymin) & (cat_all[:, 1] <= ymax)
        )
        cat_f = cat_all[keep]
        cat_idx_map = np.flatnonzero(keep)

        matches_local = one_to_one_match(pred_all, cat_f, radius_arcsec=match_tol_arcsec)
        inliers = [(int(det_i), int(cat_idx_map[cat_i]), float(d)) for (det_i, cat_i, d) in matches_local]

        num_inliers = int(len(inliers))
        rms_inliers = float(np.sqrt(np.mean([d * d for (_, _, d) in inliers]))) if num_inliers > 0 else float("inf")

        return {
            "num_inliers": num_inliers,
            "rms_inliers": rms_inliers,
            "fit": fit,
            "center": center0,
            "candidate": cand,
            "inliers": inliers,
        }

    to_eval = candidates[: min(len(candidates), int(max_trials))]

    best = None
    if progress_cb:
        progress_cb("platesolve:validate:start", {"n_eval": int(len(to_eval))})

    for cand in to_eval:
        ev = evaluate_candidate(cand)
        if best is None:
            best = ev
            continue
        cur = (ev["num_inliers"], -ev["rms_inliers"], -ev["candidate"]["score"])
        bst = (best["num_inliers"], -best["rms_inliers"], -best["candidate"]["score"])
        if cur > bst:
            best = ev

    if best is None:
        return PlatesolveResult(
            success=False,
            status="VALIDATION_FAILED",
            theta_deg=0.0,
            dx_px=0.0,
            dy_px=0.0,
            response=0.0,
            scale_arcsec_per_px=float(arcsec_per_px_ds),
            R_2x2=((1.0, 0.0), (0.0, 1.0)),
            t_arcsec=(0.0, 0.0),
            n_inliers=0,
            rms_arcsec=float("inf"),
            rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"n_eval": float(len(to_eval))},
        )

    # 12) Final overlays (inliers + Gaia points in view)
    best_center: SkyCoord = best["center"]
    best_fit: Dict[str, Any] = best["fit"]
    best_inliers: List[Tuple[int, int, float]] = best["inliers"]

    # Gaia all into best TAN arcsec
    d_lon_all, d_lat_all = best_center.spherical_offsets_to(coords)
    gaia_xy_arcsec = np.column_stack([d_lon_all.to_value(u.arcsec), d_lat_all.to_value(u.arcsec)])

    s = float(best_fit["s"])
    R = np.asarray(best_fit["R"], dtype=np.float64)
    t_arcsec = np.asarray(best_fit["t"], dtype=np.float64)

    # convert Gaia arcsec -> pixels (DS) for overlay
    gaia_xy_px = inverse_similarity(gaia_xy_arcsec, s, R, t_arcsec)

    # mark matches in overlay
    for det_idx, cat_idx, dist in best_inliers:
        ix, iy = img_xy_all[det_idx]
        gx, gy = gaia_xy_px[cat_idx]
        overlay.append(OverlayItem(float(ix), float(iy), "match", None))
        # optionally show Gaia match point too as "match" with label
        overlay.append(OverlayItem(float(gx), float(gy), "match", None))

    # 13) Guides (optional)
    guides: List[GuideStar] = []
    guide_n = int(getattr(cfg, "guide_n", 3))
    if guide_n > 0:
        gi = select_guide_star_indices(gaia_df, guide_n)
        guides = build_guides_from_solution(
            gaia_df,
            gi,
            center_icrs=best_center,
            s_arcsec_per_px=s,
            R=R,
            t_arcsec=t_arcsec,
            downsample=ds,
            cfg=cfg,
            progress_cb=progress_cb,
        )
        for g in guides:
            overlay.append(OverlayItem(float(g.x), float(g.y), "guide", str(g.name)))

    # 14) Success criterion
    min_inliers = int(getattr(cfg, "min_inliers", 3))
    success = bool(best["num_inliers"] >= min_inliers)

    offset_lon, offset_lat = best_center.spherical_offsets_to(center_icrs)
    offset_arcsec = np.array([offset_lon.to_value(u.arcsec), offset_lat.to_value(u.arcsec)], dtype=np.float64)
    offset_px = (offset_arcsec / max(1e-9, float(s))) @ R
    theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    rms_px = float(best["rms_inliers"] / max(1e-9, float(s)))
    response = float(best["num_inliers"]) / max(1.0, rms_px)

    metrics = {
        "source": 1.0 if source == "live" else 2.0,
        "n_det": float(img_xy_all.shape[0]),
        "n_seed": float(img_xy_seed.shape[0]),
        "gaia_rows": float(len(gaia_df)),
        "radius_deg": float(radius_deg),
        "arcsec_per_px_ds": float(arcsec_per_px_ds),
        "triplet_score": float(best["candidate"]["score"]),
        "triplet_err_max": float(best["candidate"]["err_max"]),
        "max_trials": float(len(to_eval)),
        "n_inliers": float(best["num_inliers"]),
        "rms_inliers_arcsec": float(best["rms_inliers"]),
        "scale_arcsec_per_px": float(s),
        "downsample": float(ds),
    }

    return PlatesolveResult(
        success=success,
        status="OK" if success else "LOW_INLIERS",
        theta_deg=theta_deg,
        dx_px=float(offset_px[0]),
        dy_px=float(offset_px[1]),
        response=response,
        scale_arcsec_per_px=float(s),
        R_2x2=((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1]))),
        t_arcsec=(float(t_arcsec[0]), float(t_arcsec[1])),
        n_inliers=int(best["num_inliers"]),
        rms_arcsec=float(best["rms_inliers"]),
        rms_px=rms_px,
        center_ra_deg=float(best_center.ra.deg),
        center_dec_deg=float(best_center.dec.deg),
        downsample=int(ds),
        overlay=overlay,
        guides=guides,
        metrics=metrics,
    )


# ============================================================
# Convenience wrappers for app_runner.py
# ============================================================

def platesolve_from_live(
    frame: np.ndarray,
    *,
    target: TargetType,
    cfg: PlatesolveConfig,
    observer: ObserverConfig = ObserverConfig(),
    obstime: Optional[Time] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> PlatesolveResult:
    return platesolve_sweep(
        frame,
        target=target,
        cfg=cfg,
        observer=observer,
        obstime=obstime,
        source="live",
        gaia_auth=None,
        progress_cb=progress_cb,
    )


def platesolve_from_stack(
    stack_frame: np.ndarray,
    *,
    target: TargetType,
    cfg: PlatesolveConfig,
    observer: ObserverConfig = ObserverConfig(),
    obstime: Optional[Time] = None,
    progress_cb: Optional[ProgressCB] = None,
) -> PlatesolveResult:
    return platesolve_sweep(
        stack_frame,
        target=target,
        cfg=cfg,
        observer=observer,
        obstime=obstime,
        source="stack",
        gaia_auth=None,
        progress_cb=progress_cb,
    )
