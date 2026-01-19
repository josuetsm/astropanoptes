# platesolve.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import cv2
import sep

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.time import Time
from astropy.table import Table, vstack

from scipy.spatial import cKDTree

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy_healpix import HEALPix
from logging_utils import log_error


# ============================================================
# Public API surface
# ============================================================

__all__ = [
    "ObserverConfig",
    "PlatesolveConfig",
    "OverlayItem",
    "GuideStar",
    "PlatesolveResult",
    "NeedGaiaAuthError",
    "GaiaCacheMissError",
    "TargetParseError",
    "load_gaia_auth",
    "save_gaia_auth",
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


class NeedGaiaAuthError(PlatesolveError):
    def __init__(self, missing_tiles: int):
        super().__init__(f"Missing Gaia tiles in cache: {missing_tiles}. Gaia credentials required to download.")
        self.missing_tiles = int(missing_tiles)


class GaiaCacheMissError(PlatesolveError):
    def __init__(self, missing_paths: List[Path]):
        super().__init__(f"Missing Gaia tiles in cache: {len(missing_paths)}")
        self.missing_paths = list(missing_paths)


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
class PlatesolveConfig:
    # Instrument
    pixel_size_m: float
    focal_m: float

    # Image processing
    downsample: int = 2
    max_det: int = 250
    det_thresh_sigma: float = 6.0
    det_minarea: int = 5
    point_sigma: float = 1.2  # sigma for gaussian blur of point-maps

    # Gaia cache + query
    cache_dir: str = "~/.cache/gaia_cones"
    table_name: str = "gaiadr3.gaia_source"
    columns: Tuple[str, ...] = ("source_id", "ra", "dec", "phot_g_mean_mag")
    gmax: float = 14.5
    nside: int = 64
    order: str = "ring"
    prefer_parquet: bool = True
    row_limit: int = -1
    retries: int = 3
    backoff_s: float = 3.0

    # Solve (Option C)
    theta_step_deg: float = 15.0
    theta_refine_step_deg: float = 3.0
    theta_refine_span_deg: float = 12.0

    # Matching
    match_max_px: float = 3.5  # in downsampled pixels
    min_inliers: int = 10

    # Search area (Gaia cone radius)
    search_radius_deg: Optional[float] = None
    search_radius_factor: float = 1.4  # radius ~= factor * (diag/2)

    # Download missing tiles
    download_missing_tiles: bool = True

    # Guides / labeling
    guide_n: int = 3
    simbad_radius_arcsec: float = 2.0
    simbad_retries: int = 3
    simbad_backoff_s: float = 0.6


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

    n_inliers: int
    rms_px: float

    center_ra_deg: float
    center_dec_deg: float

    scale_arcsec_per_px: float
    downsample: int

    overlay: List[OverlayItem]
    guides: List[GuideStar]
    metrics: Dict[str, float]


# ============================================================
# File utils (atomic json + permissions)
# ============================================================

def _chmod_600(p: Path) -> None:
    try:
        os.chmod(p, 0o600)
    except Exception as exc:
        log_error(None, f"Platesolve: failed to chmod 600 ({p})", exc, throttle_s=10.0, throttle_key="platesolve_chmod")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _chmod_600(tmp)
    tmp.replace(path)


# ============================================================
# Gaia auth (persistent file)
# ============================================================

def _gaia_auth_path() -> Path:
    p = Path("~/.config/astropanoptes/gaia_auth.json").expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_gaia_auth() -> Optional[Tuple[str, str]]:
    p = _gaia_auth_path()
    if not p.exists():
        return None
    try:
        data = _load_json(p)
        user = str(data.get("username", "")).strip()
        pw = str(data.get("password", "")).strip()
        if user and pw:
            return (user, pw)
    except Exception as exc:
        log_error(None, "Platesolve: failed to load Gaia auth", exc, throttle_s=10.0, throttle_key="platesolve_gaia_auth_load")
        return None
    return None


def save_gaia_auth(username: str, password: str) -> None:
    p = _gaia_auth_path()
    _atomic_write_json(p, {"username": username, "password": password, "ts": time.time()})
    _chmod_600(p)


# ============================================================
# SIMBAD cache + resolvers (always used for guides & name targets)
# ============================================================

def _simbad_cache_path() -> Path:
    p = Path("~/.cache/astropanoptes/simbad_cache.json").expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _simbad_key_radec(ra_deg: float, dec_deg: float, prec_arcsec: float = 1.0) -> str:
    step = prec_arcsec / 3600.0
    ra_q = round(float(ra_deg) / step) * step
    dec_q = round(float(dec_deg) / step) * step
    return f"radec:{ra_q:.8f},{dec_q:.8f}"


def _simbad_key_obj(name: str) -> str:
    return f"obj:{name.strip().lower()}"


def simbad_coord_for_object_name(
    name: str,
    *,
    retries: int,
    backoff_s: float,
    progress_cb: Optional[ProgressCB] = None,
) -> SkyCoord:
    name = str(name).strip()
    if not name:
        raise TargetParseError("Empty object name.")

    cache_path = _simbad_cache_path()
    cache = _load_json(cache_path)
    key = _simbad_key_obj(name)

    if key in cache and "ra_deg" in cache[key] and "dec_deg" in cache[key]:
        return SkyCoord(cache[key]["ra_deg"] * u.deg, cache[key]["dec_deg"] * u.deg, frame="icrs")

    sim = Simbad()
    last_err: Optional[Exception] = None

    for attempt in range(1, int(retries) + 1):
        try:
            if progress_cb:
                progress_cb("simbad:query_object", {"attempt": attempt, "name": name})
            tab = sim.query_object(name)
            if tab is None or len(tab) == 0:
                raise TargetParseError(f"SIMBAD did not find: {name}")

            ra_raw = tab["ra"][0]
            dec_raw = tab["dec"][0]
            ra_s = ra_raw.decode("utf-8") if isinstance(ra_raw, (bytes, bytearray)) else str(ra_raw)
            dec_s = dec_raw.decode("utf-8") if isinstance(dec_raw, (bytes, bytearray)) else str(dec_raw)

            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")

            cache[key] = {"ra_deg": float(c.ra.deg), "dec_deg": float(c.dec.deg), "ts": time.time()}
            _atomic_write_json(cache_path, cache)
            return c
        except Exception as e:
            last_err = e
            time.sleep(float(backoff_s) * attempt)

    raise TargetParseError(f"SIMBAD name resolve failed: {name}. Last error: {last_err}")


def simbad_name_for_coord(
    coord: SkyCoord,
    *,
    radius_arcsec: float,
    retries: int,
    backoff_s: float,
    progress_cb: Optional[ProgressCB] = None,
) -> str:
    cache_path = _simbad_cache_path()
    cache = _load_json(cache_path)
    key = _simbad_key_radec(coord.ra.deg, coord.dec.deg, prec_arcsec=1.0)

    if key in cache and "name" in cache[key]:
        return str(cache[key]["name"])

    sim = Simbad()
    last_err: Optional[Exception] = None

    for attempt in range(1, int(retries) + 1):
        try:
            if progress_cb:
                progress_cb("simbad:query_region", {"attempt": attempt})
            tab = sim.query_region(coord, radius=float(radius_arcsec) * u.arcsec)

            if tab is None or len(tab) == 0:
                name = "SIMBAD:UNKNOWN"
            else:
                raw = tab["MAIN_ID"][0]
                name = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

            cache[key] = {"name": name, "ts": time.time()}
            _atomic_write_json(cache_path, cache)
            return name
        except Exception as e:
            last_err = e
            time.sleep(float(backoff_s) * attempt)

    # Keep deterministic behavior even on failure
    if last_err is not None:
        return "SIMBAD:ERROR"
    return "SIMBAD:UNKNOWN"


# ============================================================
# Target parsing (ICRS/J2000, AltAz, name)
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
    progress_cb: Optional[ProgressCB],
    simbad_retries: int,
    simbad_backoff_s: float,
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
        if isinstance(a, str) or isinstance(b, str):
            return SkyCoord(str(a), str(b), unit=(u.hourangle, u.deg), frame="icrs")
        return SkyCoord(ra=float(a) * u.deg, dec=float(b) * u.deg, frame="icrs")

    if isinstance(target, str):
        s = target.strip()
        if not s:
            raise TargetParseError("Empty target string.")

        # Name => SIMBAD (always)
        if any(ch.isalpha() for ch in s):
            return simbad_coord_for_object_name(
                s,
                retries=simbad_retries,
                backoff_s=simbad_backoff_s,
                progress_cb=progress_cb,
            ).icrs

        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            ra_s, dec_s = parts[0], parts[1]
            if ":" in ra_s:
                return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
            return SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")

        raise TargetParseError(f"Could not parse target string: {s}")

    raise TargetParseError(f"Unsupported target type: {type(target).__name__}")


# ============================================================
# Gaia HEALPix cache (integrated)
# ============================================================

_DEFAULT_CACHE_DIR = Path(os.environ.get("GAIA_CONE_CACHE_DIR", "~/.cache/gaia_cones")).expanduser().resolve()


def set_cache_dir(path: Union[str, Path]) -> None:
    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = Path(path).expanduser().resolve()
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tile_path(hexkey: str, prefer_parquet: bool) -> Path:
    ext = "parquet" if prefer_parquet else "ecsv"
    return _DEFAULT_CACHE_DIR.joinpath(hexkey[:2], hexkey[2:4], f"{hexkey}.{ext}")


def _save_table(tab: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    if path.suffix.lower() == ".parquet":
        tab.to_pandas().to_parquet(tmp, index=False)
    else:
        tab.write(tmp, format="ascii.ecsv", overwrite=True, fast_writer=False)

    tmp.replace(path)


def _load_table(path: Path) -> Table:
    if path.suffix.lower() == ".parquet":
        return Table.from_pandas(pd.read_parquet(path))
    return Table.read(path, format="ascii.ecsv")


def _adql_polygon_from_skycoord(poly: SkyCoord) -> str:
    lon = np.asarray(poly.ra.deg).ravel()
    lat = np.asarray(poly.dec.deg).ravel()
    pairs = ", ".join(f"{float(l):.10f},{float(b):.10f}" for l, b in zip(lon, lat))
    return f"POLYGON('ICRS', {pairs})"


def _query_polygon_tile(
    *,
    table_name: str,
    columns: Sequence[str],
    gmax: float,
    poly_sky: SkyCoord,
    row_limit: int,
    retries: int,
    backoff_s: float,
    progress_cb: Optional[ProgressCB],
) -> Table:
    cols_sql = ", ".join(columns)
    poly_adql = _adql_polygon_from_skycoord(poly_sky)

    query = f"""
    SELECT {cols_sql}
    FROM {table_name}
    WHERE phot_g_mean_mag <= {gmax}
      AND 1=CONTAINS(POINT('ICRS', ra, dec), {poly_adql})
    """

    old_rl = Gaia.ROW_LIMIT
    Gaia.ROW_LIMIT = int(row_limit)
    try:
        last_err: Optional[Exception] = None
        for attempt in range(1, int(retries) + 1):
            try:
                if progress_cb:
                    progress_cb("gaia:tile_query", {"attempt": attempt})
                job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=False)
                return job.get_results()
            except Exception as e:
                last_err = e
                if attempt == retries:
                    raise
                time.sleep(float(backoff_s) * attempt)
        raise last_err if last_err else RuntimeError("Gaia query failed")
    finally:
        Gaia.ROW_LIMIT = old_rl


def gaia_healpix_cone_with_mag(
    center_icrs: SkyCoord,
    radius_deg: float,
    *,
    gmax: float,
    nside: int,
    order: str,
    table_name: str,
    columns: Sequence[str],
    auth: Optional[Tuple[str, str]],
    row_limit: int,
    prefer_parquet: bool,
    retries: int,
    backoff_s: float,
    progress_cb: Optional[ProgressCB] = None,
) -> Table:
    """
    HEALPix mosaic around center. Cache per tile.
    If auth is provided, performs a single Gaia.login/logout wrapping missing-tile queries.
    """
    center = center_icrs.icrs
    hp = HEALPix(nside=int(nside), order=str(order), frame=center.frame)
    pix_indices = hp.cone_search_skycoord(center, Angle(float(radius_deg), u.deg))

    if progress_cb:
        progress_cb("gaia:mosaic", {"tiles": int(len(pix_indices)), "nside": int(nside), "order": str(order)})

    did_login = False
    try:
        if auth:
            if progress_cb:
                progress_cb("gaia:login", {})
            Gaia.login(user=auth[0], password=auth[1])
            did_login = True

        parts: List[Table] = []

        for k, pix in enumerate(pix_indices, start=1):
            poly = hp.boundaries_skycoord(pix, step=1)

            hexkey = _cache_key(
                "healpix_tile",
                {
                    "table": table_name,
                    "nside": int(nside),
                    "order": str(order),
                    "pix": int(pix),
                    "gmax": float(gmax),
                    "columns": list(columns),
                },
            )
            path = _tile_path(hexkey, prefer_parquet=prefer_parquet)

            if path.exists():
                tab = _load_table(path)
            else:
                if progress_cb:
                    progress_cb("gaia:tile_miss", {"k": k, "pix": int(pix)})
                tab = _query_polygon_tile(
                    table_name=table_name,
                    columns=columns,
                    gmax=float(gmax),
                    poly_sky=poly,
                    row_limit=int(row_limit),
                    retries=int(retries),
                    backoff_s=float(backoff_s),
                    progress_cb=progress_cb,
                )
                _save_table(tab, path)

            parts.append(tab)

    finally:
        if did_login:
            if progress_cb:
                progress_cb("gaia:logout", {})
            try:
                Gaia.logout()
            except Exception as exc:
                log_error(None, "Platesolve: Gaia logout failed", exc, throttle_s=10.0, throttle_key="platesolve_gaia_logout")

    if not parts:
        return Table(names=list(columns))

    full = vstack(parts, join_type="outer", metadata_conflicts="silent")

    # Dedup by source_id
    if "source_id" in full.colnames:
        df = full.to_pandas()
        df = df.drop_duplicates(subset=["source_id"])
        full = Table.from_pandas(df)

    # Crop to exact circle
    sc = SkyCoord(full["ra"] * u.deg, full["dec"] * u.deg, frame="icrs")
    sep_deg = sc.separation(center).deg
    full = full[np.asarray(sep_deg) <= float(radius_deg)]

    return full


def preflight_missing_gaia_tiles(
    center_icrs: SkyCoord,
    radius_deg: float,
    *,
    table_name: str,
    columns: Sequence[str],
    gmax: float,
    nside: int,
    order: str,
    prefer_parquet: bool,
) -> List[Path]:
    center = center_icrs.icrs
    hp = HEALPix(nside=int(nside), order=str(order), frame=center.frame)
    pix_indices = hp.cone_search_skycoord(center, float(radius_deg) * u.deg)

    missing: List[Path] = []
    for pix in np.asarray(pix_indices, dtype=np.int64).tolist():
        hexkey = _cache_key(
            "healpix_tile",
            {
                "table": table_name,
                "nside": int(nside),
                "order": str(order),
                "pix": int(pix),
                "gmax": float(gmax),
                "columns": list(columns),
            },
        )
        path = _tile_path(hexkey, prefer_parquet=prefer_parquet)
        if not path.exists():
            missing.append(path)
    return missing


def load_gaia_df_with_optional_download(
    center_icrs: SkyCoord,
    radius_deg: float,
    *,
    cfg: PlatesolveConfig,
    gaia_auth: Optional[Tuple[str, str]],
    progress_cb: Optional[ProgressCB],
) -> pd.DataFrame:
    set_cache_dir(cfg.cache_dir)

    missing = preflight_missing_gaia_tiles(
        center_icrs,
        float(radius_deg),
        table_name=cfg.table_name,
        columns=cfg.columns,
        gmax=cfg.gmax,
        nside=cfg.nside,
        order=cfg.order,
        prefer_parquet=cfg.prefer_parquet,
    )

    if progress_cb:
        progress_cb("gaia:preflight", {"missing": int(len(missing)), "radius_deg": float(radius_deg)})

    if missing:
        if not cfg.download_missing_tiles:
            raise GaiaCacheMissError(missing)
        if gaia_auth is None:
            raise NeedGaiaAuthError(missing_tiles=len(missing))

        tab = gaia_healpix_cone_with_mag(
            center_icrs=center_icrs,
            radius_deg=float(radius_deg),
            gmax=float(cfg.gmax),
            nside=int(cfg.nside),
            order=str(cfg.order),
            table_name=str(cfg.table_name),
            columns=tuple(cfg.columns),
            auth=gaia_auth,
            row_limit=int(cfg.row_limit),
            prefer_parquet=bool(cfg.prefer_parquet),
            retries=int(cfg.retries),
            backoff_s=float(cfg.backoff_s),
            progress_cb=progress_cb,
        )
    else:
        # IMPORTANT: auth=None ensures no login overhead
        tab = gaia_healpix_cone_with_mag(
            center_icrs=center_icrs,
            radius_deg=float(radius_deg),
            gmax=float(cfg.gmax),
            nside=int(cfg.nside),
            order=str(cfg.order),
            table_name=str(cfg.table_name),
            columns=tuple(cfg.columns),
            auth=None,
            row_limit=int(cfg.row_limit),
            prefer_parquet=bool(cfg.prefer_parquet),
            retries=int(cfg.retries),
            backoff_s=float(cfg.backoff_s),
            progress_cb=progress_cb,
        )

    df = tab.to_pandas()
    keep = [c for c in ["source_id", "ra", "dec", "phot_g_mean_mag"] if c in df.columns]
    df = df.loc[:, keep].dropna(subset=["ra", "dec"]).reset_index(drop=True)
    return df


# ============================================================
# Image: SEP detection
# ============================================================

def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    f = int(max(1, factor))
    if f == 1:
        return img
    return img[::f, ::f]


def detect_stars_sep(
    img: np.ndarray,
    *,
    max_sources: int,
    thresh_sigma: float,
    minarea: int,
    progress_cb: Optional[ProgressCB],
) -> Tuple[np.ndarray, np.ndarray]:
    if progress_cb:
        progress_cb("detect:start", {})

    data = _to_gray_f32(img)
    bkg = sep.Background(data)
    sub = data - bkg
    thresh = float(thresh_sigma) * float(bkg.globalrms)

    objs = sep.extract(sub, thresh, minarea=int(minarea))
    if objs is None or len(objs) == 0:
        if progress_cb:
            progress_cb("detect:empty", {})
        return np.zeros((0, 2), np.float64), np.zeros((0,), np.float64)

    flux = objs["flux"].astype(np.float64)
    x = objs["x"].astype(np.float64)
    y = objs["y"].astype(np.float64)

    idx = np.argsort(-flux)[: int(max_sources)]
    xy = np.column_stack([x[idx], y[idx]])

    if progress_cb:
        progress_cb("detect:done", {"n": int(len(idx)), "thresh": float(thresh)})

    return xy, flux[idx]


# ============================================================
# Projection + point maps
# ============================================================

def scale_arcsec_per_px(pixel_size_m: float, focal_m: float) -> float:
    return (206265.0 * float(pixel_size_m)) / float(focal_m)


def gaia_offsets_arcsec(center: SkyCoord, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    off_frame = center.skyoffset_frame()
    off = sc.transform_to(off_frame)
    u_as = off.lon.to(u.arcsec).value
    v_as = off.lat.to(u.arcsec).value
    return u_as.astype(np.float64), v_as.astype(np.float64)


def rotate_uv(u_as: np.ndarray, v_as: np.ndarray, theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.deg2rad(float(theta_deg))
    c, s = float(np.cos(t)), float(np.sin(t))
    return (c * u_as - s * v_as), (s * u_as + c * v_as)


def uv_to_xy_px(
    u_as: np.ndarray,
    v_as: np.ndarray,
    *,
    scale_as_per_px: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = cx + (u_as / float(scale_as_per_px))
    y = cy - (v_as / float(scale_as_per_px))
    return x, y


def mag_to_weight(mag: np.ndarray) -> np.ndarray:
    m = np.asarray(mag, dtype=np.float64)
    m0 = float(np.nanmin(m)) if np.isfinite(m).any() else 0.0
    w = 10.0 ** (-0.4 * (m - m0))
    w = w / (np.max(w) + 1e-12)
    return w.astype(np.float32)


def render_points_map(
    shape_hw: Tuple[int, int],
    xy: np.ndarray,
    *,
    sigma: float,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out = np.zeros((h, w), np.float32)

    if xy is None or len(xy) == 0:
        return out

    pts = np.asarray(xy, dtype=np.float64)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)

    if weights is None:
        out[ys, xs] = 1.0
    else:
        ww = np.asarray(weights, dtype=np.float32)
        if ww.shape[0] != xs.shape[0]:
            ww = np.ones((xs.shape[0],), np.float32)
        out[ys, xs] = np.maximum(out[ys, xs], ww)

    k = int(max(3, (int(6 * float(sigma)) | 1)))
    out = cv2.GaussianBlur(out, (k, k), float(sigma))
    return out


def estimate_search_radius_deg(h: int, w: int, *, scale_as_per_px: float, factor: float) -> float:
    diag_px = float(np.hypot(w, h))
    radius_as = float(factor) * (diag_px / 2.0) * float(scale_as_per_px)
    radius_deg = radius_as / 3600.0
    return float(max(0.4, radius_deg))


# ============================================================
# Guides (3 brightest in search area; SIMBAD always)
# ============================================================

def select_guide_star_indices(df_gaia: pd.DataFrame, n: int) -> List[int]:
    n = int(max(1, n))
    if "phot_g_mean_mag" in df_gaia.columns:
        idx = np.argsort(df_gaia["phot_g_mean_mag"].to_numpy(np.float64))
        return [int(i) for i in idx[: min(n, len(idx))].tolist()]
    return [int(i) for i in range(min(n, len(df_gaia)))]


def build_guides(
    df_gaia: pd.DataFrame,
    guide_idx: List[int],
    *,
    center_icrs: SkyCoord,
    theta_deg: float,
    dx_px: float,
    dy_px: float,
    scale_as_per_px: float,
    cx: float,
    cy: float,
    cfg: PlatesolveConfig,
    progress_cb: Optional[ProgressCB],
) -> List[GuideStar]:
    guides: List[GuideStar] = []
    for i in guide_idx:
        ra = float(df_gaia.at[i, "ra"])
        dec = float(df_gaia.at[i, "dec"])
        gmag = float(df_gaia.at[i, "phot_g_mean_mag"]) if "phot_g_mean_mag" in df_gaia.columns else float("nan")

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        name = simbad_name_for_coord(
            coord,
            radius_arcsec=float(cfg.simbad_radius_arcsec),
            retries=int(cfg.simbad_retries),
            backoff_s=float(cfg.simbad_backoff_s),
            progress_cb=progress_cb,
        )

        u_as, v_as = gaia_offsets_arcsec(center_icrs, np.array([ra], np.float64), np.array([dec], np.float64))
        ur, vr = rotate_uv(u_as, v_as, float(theta_deg))
        x, y = uv_to_xy_px(ur, vr, scale_as_per_px=float(scale_as_per_px), cx=float(cx), cy=float(cy))
        x = float(x[0] + float(dx_px))
        y = float(y[0] + float(dy_px))

        guides.append(GuideStar(name=str(name), ra_deg=ra, dec_deg=dec, gmag=gmag, x=x, y=y))

    return guides


# ============================================================
# Pixel -> RA/Dec helpers
# ============================================================

def pixel_to_sky_offset_arcsec(
    x_px: float,
    y_px: float,
    *,
    cx: float,
    cy: float,
    theta_deg: float,
    dx_px: float,
    dy_px: float,
    scale_arcsec_per_px: float,
) -> Tuple[float, float]:
    # Undo shift
    x = float(x_px) - float(dx_px)
    y = float(y_px) - float(dy_px)

    # In rotated frame (arcsec)
    u_rot = (x - float(cx)) * float(scale_arcsec_per_px)
    v_rot = (float(cy) - y) * float(scale_arcsec_per_px)

    # Undo rotation
    t = np.deg2rad(float(theta_deg))
    c, s = float(np.cos(t)), float(np.sin(t))
    u_as = c * u_rot + s * v_rot
    v_as = -s * u_rot + c * v_rot
    return float(u_as), float(v_as)


def pixel_to_radec(
    x_px: float,
    y_px: float,
    *,
    center_icrs: SkyCoord,
    cx: float,
    cy: float,
    theta_deg: float,
    dx_px: float,
    dy_px: float,
    scale_arcsec_per_px: float,
) -> SkyCoord:
    u_as, v_as = pixel_to_sky_offset_arcsec(
        x_px,
        y_px,
        cx=cx,
        cy=cy,
        theta_deg=theta_deg,
        dx_px=dx_px,
        dy_px=dy_px,
        scale_arcsec_per_px=scale_arcsec_per_px,
    )
    off_frame = center_icrs.skyoffset_frame()
    c_off = SkyCoord(lon=u_as * u.arcsec, lat=v_as * u.arcsec, frame=off_frame)
    return c_off.icrs


# ============================================================
# Main solver (Option C): sweep theta + phase correlation
# ============================================================

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
        center_icrs = parse_target_to_icrs(
            target,
            observer=observer,
            obstime=obstime,
            progress_cb=progress_cb,
            simbad_retries=int(cfg.simbad_retries),
            simbad_backoff_s=float(cfg.simbad_backoff_s),
        )
    except Exception as exc:
        log_error(None, "Platesolve: target parse failed", exc)
        return PlatesolveResult(
            success=False,
            status="TARGET_PARSE_ERROR",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=0.0, center_dec_deg=0.0,
            scale_arcsec_per_px=0.0, downsample=int(cfg.downsample),
            overlay=[],
            guides=[],
            metrics={"source": 1.0},
        )

    # 2) Prepare working frame (gray + downsample)
    gray = _to_gray_f32(frame)
    ds = int(max(1, cfg.downsample))
    gray_ds = _downsample(gray, ds)
    h, w = gray_ds.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    scale0 = scale_arcsec_per_px(cfg.pixel_size_m, cfg.focal_m)
    scale_ds = float(scale0) * float(ds)

    # 3) SEP detections
    det_xy, det_flux = detect_stars_sep(
        gray_ds,
        max_sources=int(cfg.max_det),
        thresh_sigma=float(cfg.det_thresh_sigma),
        minarea=int(cfg.det_minarea),
        progress_cb=progress_cb,
    )

    overlay: List[OverlayItem] = [OverlayItem(float(x), float(y), "det", None) for x, y in det_xy]

    if len(det_xy) < max(1, int(cfg.min_inliers)):
        return PlatesolveResult(
            success=False,
            status="NOT_ENOUGH_DETECTIONS",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"n_det": float(len(det_xy))},
        )

    # 4) Search radius for Gaia
    if cfg.search_radius_deg is None:
        radius_deg = estimate_search_radius_deg(h, w, scale_as_per_px=scale_ds, factor=float(cfg.search_radius_factor))
    else:
        radius_deg = float(cfg.search_radius_deg)

    # 5) Gaia load (cache + download if missing)
    if gaia_auth is None:
        gaia_auth = load_gaia_auth()

    try:
        if progress_cb:
            progress_cb("gaia:load:start", {"radius_deg": float(radius_deg), "gmax": float(cfg.gmax), "source": source})

        gaia_df = load_gaia_df_with_optional_download(
            center_icrs,
            float(radius_deg),
            cfg=cfg,
            gaia_auth=gaia_auth,
            progress_cb=progress_cb,
        )
    except NeedGaiaAuthError as e:
        return PlatesolveResult(
            success=False,
            status="NEED_GAIA_AUTH",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"missing_tiles": float(e.missing_tiles)},
        )
    except GaiaCacheMissError as e:
        return PlatesolveResult(
            success=False,
            status="GAIA_CACHE_MISS",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"missing": float(len(e.missing_paths))},
        )
    except Exception as exc:
        log_error(None, "Platesolve: Gaia load failed", exc)
        return PlatesolveResult(
            success=False,
            status="GAIA_LOAD_ERROR",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"err": 1.0},
        )

    if len(gaia_df) < 30:
        return PlatesolveResult(
            success=False,
            status="GAIA_TOO_SMALL",
            theta_deg=0.0, dx_px=0.0, dy_px=0.0, response=0.0,
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"gaia_rows": float(len(gaia_df))},
        )

    # 6) Gaia offsets; select bright subset for correlation
    ra_all = gaia_df["ra"].to_numpy(np.float64)
    dec_all = gaia_df["dec"].to_numpy(np.float64)
    u_as_all, v_as_all = gaia_offsets_arcsec(center_icrs, ra_all, dec_all)

    if "phot_g_mean_mag" in gaia_df.columns:
        mags = gaia_df["phot_g_mean_mag"].to_numpy(np.float64)
        idx_bright = np.argsort(mags)[: min(600, len(mags))]
        weights_gaia = mag_to_weight(mags[idx_bright])
    else:
        idx_bright = np.arange(min(600, len(gaia_df)))
        weights_gaia = None

    u_sel = u_as_all[idx_bright]
    v_sel = v_as_all[idx_bright]

    # 7) Build maps
    det_map = render_points_map((h, w), det_xy, sigma=float(cfg.point_sigma), weights=det_flux.astype(np.float32))
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def eval_theta(theta_deg: float) -> Tuple[float, float, float]:
        ur, vr = rotate_uv(u_sel, v_sel, theta_deg)
        gx, gy = uv_to_xy_px(ur, vr, scale_as_per_px=scale_ds, cx=cx, cy=cy)
        gxy = np.column_stack([gx, gy])
        gaia_map = render_points_map((h, w), gxy, sigma=float(cfg.point_sigma), weights=weights_gaia)
        (dx, dy), resp = cv2.phaseCorrelate(det_map, gaia_map, hann)
        return float(dx), float(dy), float(resp)

    # 8) Coarse sweep
    if progress_cb:
        progress_cb("platesolve:sweep:start", {"theta_step_deg": float(cfg.theta_step_deg)})

    best_theta = 0.0
    best_dx = 0.0
    best_dy = 0.0
    best_resp = -1.0

    for th in np.arange(0.0, 360.0, float(cfg.theta_step_deg)):
        dx, dy, resp = eval_theta(float(th))
        if resp > best_resp:
            best_resp, best_theta, best_dx, best_dy = float(resp), float(th), float(dx), float(dy)

    # 9) Refine
    span = float(cfg.theta_refine_span_deg)
    step = float(cfg.theta_refine_step_deg)
    for th in np.arange(best_theta - span, best_theta + span + 1e-9, step):
        dx, dy, resp = eval_theta(float(th))
        if resp > best_resp:
            best_resp, best_theta, best_dx, best_dy = float(resp), float(th), float(dx), float(dy)

    if progress_cb:
        progress_cb("platesolve:sweep:best", {"theta_deg": best_theta, "dx": best_dx, "dy": best_dy, "resp": best_resp})

    # 10) Final matching (inliers)
    ur_all, vr_all = rotate_uv(u_as_all, v_as_all, best_theta)
    gx_all, gy_all = uv_to_xy_px(ur_all, vr_all, scale_as_per_px=scale_ds, cx=cx, cy=cy)
    gx_all = gx_all + best_dx
    gy_all = gy_all + best_dy

    inside = (gx_all >= 0) & (gx_all < w) & (gy_all >= 0) & (gy_all < h)
    g_idx = np.where(inside)[0]
    if len(g_idx) == 0:
        return PlatesolveResult(
            success=False,
            status="NO_GAIA_IN_FRAME",
            theta_deg=float(best_theta), dx_px=float(best_dx), dy_px=float(best_dy), response=float(best_resp),
            n_inliers=0, rms_px=float("inf"),
            center_ra_deg=float(center_icrs.ra.deg),
            center_dec_deg=float(center_icrs.dec.deg),
            scale_arcsec_per_px=float(scale_ds),
            downsample=int(ds),
            overlay=overlay,
            guides=[],
            metrics={"resp": float(best_resp)},
        )

    gxy = np.column_stack([gx_all[g_idx], gy_all[g_idx]])

    det_tree = cKDTree(det_xy.astype(np.float64))
    dists, nn = det_tree.query(gxy.astype(np.float64), k=1, workers=-1)

    pairs = [(float(dists[i]), int(nn[i]), int(g_idx[i])) for i in range(len(g_idx)) if float(dists[i]) <= float(cfg.match_max_px)]
    pairs.sort(key=lambda t: t[0])

    used_det = set()
    used_gaia = set()
    matches: List[Tuple[int, int, float]] = []
    for d, idet, iga in pairs:
        if idet in used_det or iga in used_gaia:
            continue
        used_det.add(idet)
        used_gaia.add(iga)
        matches.append((idet, iga, d))

    n_in = int(len(matches))
    rms = float(np.sqrt(np.mean([d * d for (_, _, d) in matches]))) if n_in > 0 else float("inf")

    for idet, _, _ in matches:
        x, y = det_xy[idet]
        overlay.append(OverlayItem(float(x), float(y), "match", None))

    # 11) Guides: 3 brightest in the SEARCH AREA (always SIMBAD)
    guide_idx = select_guide_star_indices(gaia_df, int(cfg.guide_n))
    guides = build_guides(
        gaia_df,
        guide_idx,
        center_icrs=center_icrs,
        theta_deg=float(best_theta),
        dx_px=float(best_dx),
        dy_px=float(best_dy),
        scale_as_per_px=float(scale_ds),
        cx=float(cx),
        cy=float(cy),
        cfg=cfg,
        progress_cb=progress_cb,
    )

    for g in guides:
        overlay.append(OverlayItem(float(g.x), float(g.y), "guide", str(g.name)))

    success = bool(n_in >= int(cfg.min_inliers))

    metrics = {
        "resp": float(best_resp),
        "n_det": float(len(det_xy)),
        "gaia_rows": float(len(gaia_df)),
        "n_inliers": float(n_in),
        "rms_px": float(rms),
        "theta_deg": float(best_theta),
        "dx_px": float(best_dx),
        "dy_px": float(best_dy),
        "radius_deg": float(radius_deg),
        "scale_arcsec_per_px": float(scale_ds),
        "downsample": float(ds),
    }

    return PlatesolveResult(
        success=bool(success),
        status="OK" if success else "LOW_INLIERS",
        theta_deg=float(best_theta),
        dx_px=float(best_dx),
        dy_px=float(best_dy),
        response=float(best_resp),
        n_inliers=int(n_in),
        rms_px=float(rms),
        center_ra_deg=float(center_icrs.ra.deg),
        center_dec_deg=float(center_icrs.dec.deg),
        scale_arcsec_per_px=float(scale_ds),
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
