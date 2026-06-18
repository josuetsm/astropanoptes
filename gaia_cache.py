# -*- coding: utf-8 -*-
"""
gaia_cache.py — Consultas Gaia (astroquery) con caché y mosaico HEALPix (async, login único).

Funciones
---------
- normalize_input(target): acepta nombre SIMBAD, sexagesimal ("HH:MM:SS ±DD:MM:SS"),
  (ra_deg, dec_deg) o dict {'ra','dec'} en grados. Devuelve SkyCoord (ICRS).
- gaia_cone_with_mag(target, radius, *, gmax=15, ...): cono único, filtro en servidor G<=gmax,
  caché por parámetros. Usa launch_job_async(background=False) para evitar límites de tiempo del sync.
- gaia_healpix_cone_with_mag(target, radius, *, gmax=15, nside=16, ...): mosaico HEALPix con filtro
  en servidor por tesela (polígono), caché por tesela, ensamblaje deduplicado y recorte al círculo.
  Hace login UNA sola vez para todo el mosaico y usa launch_job_async(background=False) por tesela.

Notas de logging (cambio solicitado)
------------------------------------
- En gaia_healpix_cone_with_mag, NO se imprime nada si todas las teselas están en caché.
- Solo se loggea (y se hace login) si falta al menos una tesela.

Requisitos
----------
- astroquery, astropy
- astropy-healpix (para función HEALPix)
- (opcional) pyarrow para parquet

Licencia: MIT
"""
from __future__ import annotations

import os
import json
import hashlib
import math
import time
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List, Dict

import numpy as np
from erfa import ErfaWarning
from logging_utils import log_error, log_info
from astroquery.gaia import Gaia
from astroquery.utils.tap import TapPlus
from astropy.coordinates import SkyCoord, Angle, ICRS
import astropy.units as u
from astropy.table import Table, vstack
from astropy.time import Time

from astropy_healpix import HEALPix

try:
    import pyarrow  # type: ignore  # noqa: F401
    _HAS_PARQUET = True
except Exception as exc:
    log_error(None, "Gaia cache: failed to import pyarrow; parquet disabled", exc)
    _HAS_PARQUET = False

class GaiaCacheMissError(RuntimeError):
    """Raised when required Gaia cache tiles are missing and downloads are disabled."""

    def __init__(self, missing_paths: Sequence[Path], missing_tiles: Optional[Sequence[int]] = None) -> None:
        self.missing_paths = list(missing_paths)
        self.missing_tiles = list(missing_tiles) if missing_tiles is not None else []
        super().__init__(f"Missing Gaia cache tiles: {len(self.missing_paths)}")


class NeedGaiaAuthError(RuntimeError):
    """Raised when Gaia authentication is required to proceed."""

    def __init__(self, missing_tiles: Optional[Sequence[int]] = None) -> None:
        self.missing_tiles = list(missing_tiles) if missing_tiles is not None else []
        super().__init__("Gaia authentication required to download missing tiles.")


# -------------------------
# Config caché y defaults
# -------------------------
_DEFAULT_CACHE_DIR = Path(os.environ.get("GAIA_CONE_CACHE_DIR", "~/.cache/gaia_cones")).expanduser()
DEFAULT_TABLE = "gaiadr3.gaia_source"
DEFAULT_COLUMNS = ("source_id", "ra", "dec", "phot_g_mean_mag")
BRIGHT_CATALOG_TABLES = ("public.hipparcos", "public.tycho2")
BRIGHT_CATALOG_VERSION = 1
BRIGHT_CATALOG_MATCH_ARCSEC = 2.0
BRIGHT_CATALOG_EPOCH = Time(2016.0, format="jyear")
_HIPPARCOS_SOURCE_ID_BASE = 1_000_000_000_000
_TYCHO2_SOURCE_ID_BASE = 2_000_000_000_000

# Gaia DR3 omits or has unreliable astrometry/photometry for some saturated
# naked-eye stars. These entries keep simulation and plate solving consistent
# without altering the cached Gaia tiles.
BRIGHT_STAR_SUPPLEMENT: Tuple[Dict[str, float | str], ...] = (
    {"name": "Sirius", "ra_deg": 101.28715533, "dec_deg": -16.71611586, "gmag": -1.46},
    {"name": "Canopus", "ra_deg": 95.987877, "dec_deg": -52.695661, "gmag": -0.74},
    {"name": "Arcturus", "ra_deg": 213.915300, "dec_deg": 19.182409, "gmag": -0.05},
    {"name": "Vega", "ra_deg": 279.234735, "dec_deg": 38.783689, "gmag": 0.03},
    {"name": "Capella", "ra_deg": 79.172328, "dec_deg": 45.997991, "gmag": 0.08},
    {"name": "Rigel", "ra_deg": 78.634467, "dec_deg": -8.201638, "gmag": 0.12},
    {"name": "Procyon", "ra_deg": 114.825493, "dec_deg": 5.224993, "gmag": 0.38},
    {"name": "Betelgeuse", "ra_deg": 88.792939, "dec_deg": 7.407064, "gmag": 0.50},
    {"name": "Aldebaran", "ra_deg": 68.980163, "dec_deg": 16.509302, "gmag": 0.86},
    {"name": "Antares", "ra_deg": 247.351917, "dec_deg": -26.432003, "gmag": 0.96},
    {"name": "Spica", "ra_deg": 201.298248, "dec_deg": -11.161323, "gmag": 0.98},
    {"name": "Fomalhaut", "ra_deg": 344.412750, "dec_deg": -29.621837, "gmag": 1.16},
    {"name": "Achernar", "ra_deg": 24.428600, "dec_deg": -57.236800, "gmag": 0.46},
    {"name": "Acrux", "ra_deg": 186.649563, "dec_deg": -63.099093, "gmag": 0.77},
)

_DEFAULT_AUTH_PATH = Path(os.environ.get(
    "GAIA_AUTH_FILE",
    "~/.config/astropanoptes/gaia_auth.json",
)).expanduser()
_REPO_ROOT = Path(__file__).resolve().parent


def _gaia_auth_path(auth_file: Optional[Union[str, Path]] = None) -> Path:
    if auth_file:
        return Path(auth_file).expanduser()
    env_path = os.environ.get("GAIA_AUTH_FILE")
    if env_path:
        return Path(env_path).expanduser()
    return _DEFAULT_AUTH_PATH


def _gaia_env_user_pass() -> Tuple[Optional[str], Optional[str]]:
    user = os.environ.get("GAIA_USER") or os.environ.get("GAIA_USERNAME")
    password = os.environ.get("GAIA_PASS") or os.environ.get("GAIA_PASSWORD")
    return user, password


def _repo_relpath(path: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return None


def _ensure_gitignore_entry(rel_path: Path) -> None:
    gitignore = _REPO_ROOT / ".gitignore"
    entry = rel_path.as_posix()
    if gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8").splitlines()
        if entry in existing:
            return
    with gitignore.open("a", encoding="utf-8") as handle:
        if gitignore.stat().st_size > 0:
            handle.write("\n")
        handle.write(f"{entry}\n")


def load_gaia_auth(auth_file: Optional[Union[str, Path]] = None) -> Optional[Tuple[str, str]]:
    """
    Load Gaia credentials from environment variables or an optional JSON file.
    Environment variables take precedence over file content.
    """
    user, password = _gaia_env_user_pass()
    if user and password:
        return user, password

    path = _gaia_auth_path(auth_file)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    user = data.get("user") or data.get("username")
    password = data.get("password")
    if user and password:
        return str(user), str(password)
    return None


def save_gaia_auth(
    user: str,
    password: str,
    auth_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save Gaia credentials to JSON outside the repo by default.
    If the target path is inside the repo, add it to .gitignore.
    """
    path = _gaia_auth_path(auth_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"user": str(user), "password": str(password)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        log_error(None, "Gaia cache: failed to chmod auth file", exc)

    rel_path = _repo_relpath(path)
    if rel_path is not None:
        _ensure_gitignore_entry(rel_path)
    return path


def resolve_name_to_icrs(name: str) -> SkyCoord:
    """
    Resolve a target name to ICRS.
    Supports 'Gaia DR2 <source_id>' via gaiadr2.gaia_source, otherwise falls back to SIMBAD.
    """
    match = re.match(r"^\s*Gaia\s*DR2\s+(\d+)\s*$", name, re.IGNORECASE)
    if match:
        source_id = int(match.group(1))
        query = f"SELECT ra, dec FROM gaiadr2.gaia_source WHERE source_id = {source_id}"
        job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=False)
        results = job.get_results()
        if len(results) < 1:
            raise ValueError(f"Gaia DR2 source not found: {source_id}")
        ra = float(results["ra"][0])
        dec = float(results["dec"][0])
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    return SkyCoord.from_name(name).icrs


# -------------------------
# Normalización de inputs
# -------------------------
def normalize_input(target) -> SkyCoord:
    """
    Acepta:
      - 'Ankaa' → SIMBAD (SkyCoord.from_name)
      - '00:26:14.8 -39:39:00.7' → sexagesimal
      - (ra_deg, dec_deg) o [ra_deg, dec_deg] → grados
      - {'ra': 6.5, 'dec': -39.6} → grados
    Devuelve SkyCoord(ICRS).
    """
    if isinstance(target, SkyCoord):
        return target.icrs

    if isinstance(target, str):
        if any(ch.isalpha() for ch in target):
            return SkyCoord.from_name(target)
        ra_str, dec_str = target.split()
        if ":" in ra_str:
            return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        return SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg, frame="icrs")

    if isinstance(target, (tuple, list)) and len(target) == 2:
        ra, dec = target
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")

    if isinstance(target, dict) and {"ra", "dec"} <= target.keys():
        return SkyCoord(float(target["ra"]) * u.deg, float(target["dec"]) * u.deg, frame="icrs")

    raise ValueError(f"Formato de target no reconocido: {target}")


# -------------------------
# Utilidades de caché
# -------------------------
def set_cache_dir(path: Union[str, Path]) -> None:
    """Cambia el directorio base de caché en tiempo de ejecución."""
    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = Path(path).expanduser().resolve()
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(*, kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _path_for(hexkey: str, prefer_parquet: bool) -> Path:
    return _path_for_in(_DEFAULT_CACHE_DIR, hexkey, prefer_parquet)


def _path_for_in(cache_dir: Path, hexkey: str, prefer_parquet: bool) -> Path:
    ext = "parquet" if (prefer_parquet and _HAS_PARQUET) else "ecsv"
    return cache_dir.joinpath(hexkey[:2], hexkey[2:4], f"{hexkey}.{ext}")


def _bright_tile_cache_key(*, nside: int, order: str, pix: int, vmax: float) -> str:
    return _cache_key(kind="bright_catalog_tile", payload={
        "version": BRIGHT_CATALOG_VERSION,
        "tables": list(BRIGHT_CATALOG_TABLES),
        "nside": int(nside),
        "order": str(order),
        "pix": int(pix),
        "vmax": float(vmax),
    })


def _save_table(tab: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        import pandas as pd
        tab.to_pandas().to_parquet(path, index=False)
    else:
        tab.write(path, format="ascii.ecsv", overwrite=True, fast_writer=False)


def _load_table(path: Path) -> Table:
    if path.suffix.lower() == ".parquet":
        import pandas as pd
        return Table.from_pandas(pd.read_parquet(path))
    return Table.read(path, format="ascii.ecsv")


def add_bright_star_supplement(
    tab: Table,
    *,
    center_icrs: SkyCoord,
    radius_deg: float,
    gmax: float,
) -> Table:
    """Add known saturated bright stars missing from the Gaia result."""
    required = {"source_id", "ra", "dec", "phot_g_mean_mag"}
    if not required.issubset(tab.colnames):
        return tab

    center = normalize_input(center_icrs)
    radius = float(radius_deg)
    if not math.isfinite(radius) or radius <= 0.0:
        return tab

    if len(tab) > 0:
        existing_coords = SkyCoord(
            ra=np.asarray(tab["ra"], dtype=np.float64) * u.deg,
            dec=np.asarray(tab["dec"], dtype=np.float64) * u.deg,
            frame="icrs",
        )
        existing_mags = np.asarray(tab["phot_g_mean_mag"], dtype=np.float64)
    else:
        existing_coords = None
        existing_mags = np.empty(0, dtype=np.float64)

    rows: List[Tuple[int, float, float, float]] = []
    for index, star in enumerate(BRIGHT_STAR_SUPPLEMENT):
        mag = float(star["gmag"])
        if mag > float(gmax):
            continue
        coord = SkyCoord(
            ra=float(star["ra_deg"]) * u.deg,
            dec=float(star["dec_deg"]) * u.deg,
            frame="icrs",
        )
        if float(coord.separation(center).deg) > radius:
            continue

        has_bright_match = False
        if existing_coords is not None:
            sep_arcsec = np.asarray(existing_coords.separation(coord).arcsec, dtype=np.float64)
            has_bright_match = bool(
                np.any((sep_arcsec <= 2.0) & np.isfinite(existing_mags) & (existing_mags <= 3.0))
            )
        if has_bright_match:
            continue

        rows.append(
            (
                -(index + 1),
                float(star["ra_deg"]),
                float(star["dec_deg"]),
                mag,
            )
        )

    if not rows:
        return tab

    base = tab.copy()
    base["source_id"] = np.asarray(base["source_id"], dtype=np.int64)
    supplement = Table(
        rows=rows,
        names=("source_id", "ra", "dec", "phot_g_mean_mag"),
        dtype=("int64", "float64", "float64", "float64"),
    )
    return vstack([base, supplement], join_type="exact", metadata_conflicts="silent")


def _finite_float_column(tab: Table, name: str, default: float = np.nan) -> np.ndarray:
    """Return an Astropy column as finite-friendly float64 values."""
    if name not in tab.colnames:
        return np.full(len(tab), default, dtype=np.float64)
    col = tab[name]
    if hasattr(col, "filled"):
        col = col.filled(default)
    return np.asarray(col, dtype=np.float64)


def _propagate_catalog_positions(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pm_ra_masyr: np.ndarray,
    pm_dec_masyr: np.ndarray,
    *,
    source_epoch_jyear: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate ICRS positions to the Gaia DR3 reference epoch (J2016.0)."""
    pm_ra = np.where(np.isfinite(pm_ra_masyr), pm_ra_masyr, 0.0)
    pm_dec = np.where(np.isfinite(pm_dec_masyr), pm_dec_masyr, 0.0)
    coords = SkyCoord(
        ra=np.asarray(ra_deg, dtype=np.float64) * u.deg,
        dec=np.asarray(dec_deg, dtype=np.float64) * u.deg,
        pm_ra_cosdec=pm_ra * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
        frame="icrs",
        obstime=Time(source_epoch_jyear, format="jyear"),
    )
    with warnings.catch_warnings():
        # No distance is available in both source tables. ERFA correctly
        # assumes a large distance; tangential propagation remains valid.
        warnings.simplefilter("ignore", ErfaWarning)
        propagated = coords.apply_space_motion(new_obstime=BRIGHT_CATALOG_EPOCH)
    return (
        np.asarray(propagated.ra.deg, dtype=np.float64),
        np.asarray(propagated.dec.deg, dtype=np.float64),
    )


def _normalize_hipparcos(tab: Table, *, vmax: float) -> Table:
    if len(tab) == 0:
        return Table(names=list(DEFAULT_COLUMNS), dtype=("int64", "float64", "float64", "float64"))

    mag = _finite_float_column(tab, "vmag")
    ra_source = _finite_float_column(tab, "ra")
    dec_source = _finite_float_column(tab, "de")
    keep = (
        np.isfinite(mag)
        & np.isfinite(ra_source)
        & np.isfinite(dec_source)
        & (mag <= float(vmax))
    )
    if not np.any(keep):
        return Table(names=list(DEFAULT_COLUMNS), dtype=("int64", "float64", "float64", "float64"))

    hip = np.asarray(tab["hip"][keep], dtype=np.int64)
    ra, dec = _propagate_catalog_positions(
        ra_source[keep],
        dec_source[keep],
        _finite_float_column(tab, "pmra", 0.0)[keep],
        _finite_float_column(tab, "pmde", 0.0)[keep],
        source_epoch_jyear=1991.25,
    )
    return Table(
        {
            "source_id": -(_HIPPARCOS_SOURCE_ID_BASE + hip),
            "ra": ra,
            "dec": dec,
            # Johnson V is used as a brightness proxy when Gaia G is absent.
            "phot_g_mean_mag": mag[keep],
        }
    )


def _normalize_tycho2(tab: Table, *, vmax: float) -> Table:
    if len(tab) == 0:
        return Table(names=list(DEFAULT_COLUMNS), dtype=("int64", "float64", "float64", "float64"))

    vt = _finite_float_column(tab, "vt_mag")
    bt = _finite_float_column(tab, "bt_mag")
    ra_source = _finite_float_column(tab, "ra_mdeg")
    dec_source = _finite_float_column(tab, "de_mdeg")
    both = np.isfinite(vt) & np.isfinite(bt)
    mag = np.where(both, vt - 0.090 * (bt - vt), np.where(np.isfinite(vt), vt, bt))
    keep = (
        np.isfinite(mag)
        & np.isfinite(ra_source)
        & np.isfinite(dec_source)
        & (mag <= float(vmax))
    )
    if not np.any(keep):
        return Table(names=list(DEFAULT_COLUMNS), dtype=("int64", "float64", "float64", "float64"))

    tycho_id = np.asarray(tab["id_tycho"][keep], dtype=np.int64)
    ra, dec = _propagate_catalog_positions(
        ra_source[keep],
        dec_source[keep],
        _finite_float_column(tab, "pm_ra", 0.0)[keep],
        _finite_float_column(tab, "pm_de", 0.0)[keep],
        source_epoch_jyear=2000.0,
    )
    return Table(
        {
            "source_id": -(_TYCHO2_SOURCE_ID_BASE + tycho_id),
            "ra": ra,
            "dec": dec,
            # Tycho BT/VT are transformed to approximate Johnson V.
            "phot_g_mean_mag": mag[keep],
        }
    )


def merge_bright_catalog(primary: Table, bright: Table) -> Table:
    """
    Merge Hipparcos/Tycho rows into Gaia without duplicating valid Gaia sources.

    A nearby Gaia row is retained unless its magnitude is more than two
    magnitudes fainter than the external bright-catalog value, which is the
    characteristic failure mode for saturated stars.
    """
    required = set(DEFAULT_COLUMNS)
    if len(bright) == 0 or not required.issubset(primary.colnames) or not required.issubset(bright.colnames):
        return primary
    if len(primary) == 0:
        return bright.copy()

    base = primary.copy()
    base["source_id"] = np.asarray(base["source_id"], dtype=np.int64)
    supplement = bright.copy()
    supplement["source_id"] = np.asarray(supplement["source_id"], dtype=np.int64)

    base_coords = SkyCoord(
        ra=_finite_float_column(base, "ra") * u.deg,
        dec=_finite_float_column(base, "dec") * u.deg,
        frame="icrs",
    )
    bright_coords = SkyCoord(
        ra=_finite_float_column(supplement, "ra") * u.deg,
        dec=_finite_float_column(supplement, "dec") * u.deg,
        frame="icrs",
    )
    nearest, sep2d, _ = bright_coords.match_to_catalog_sky(base_coords)
    base_mag = _finite_float_column(base, "phot_g_mean_mag")
    bright_mag = _finite_float_column(supplement, "phot_g_mean_mag")
    close = np.asarray(sep2d.arcsec <= BRIGHT_CATALOG_MATCH_ARCSEC, dtype=bool)
    valid_gaia = close & np.isfinite(base_mag[nearest]) & (
        base_mag[nearest] <= bright_mag + 2.0
    )

    replace_base = np.zeros(len(base), dtype=bool)
    for index in np.asarray(nearest[close & ~valid_gaia], dtype=np.int64):
        replace_base[int(index)] = True
    add_bright = ~valid_gaia
    if not np.any(add_bright):
        return base

    return vstack(
        [base[~replace_base], supplement[add_bright]],
        join_type="outer",
        metadata_conflicts="silent",
    )


def gaia_healpix_coverage(
    *,
    cfg=None,
    center_icrs: Optional[SkyCoord] = None,
    radius_deg: Optional[float] = None,
) -> Dict[str, object]:
    """
    Inspect the local Gaia HEALPix cache without loading tables or using the network.

    Coverage is reported for the exact cache configuration in ``cfg``. If a
    center and radius are provided, the result also describes whether every
    tile required by that field is available.
    """
    _ensure_healpix_available()

    cache_dir = _DEFAULT_CACHE_DIR
    table_name = DEFAULT_TABLE
    columns: Sequence[str] = DEFAULT_COLUMNS
    gmax = 15.0
    nside = 16
    order = "ring"
    bright_catalog_enabled = True
    bright_catalog_margin_deg = 0.15
    prefer_parquet = True
    if cfg is not None:
        cache_dir = Path(getattr(cfg, "cache_dir", cache_dir)).expanduser().resolve()
        table_name = str(getattr(cfg, "table_name", table_name))
        columns = tuple(getattr(cfg, "columns", columns))
        gmax = float(getattr(cfg, "gmax", gmax))
        nside = int(getattr(cfg, "nside", nside))
        order = str(getattr(cfg, "order", order))
        bright_catalog_enabled = bool(getattr(cfg, "bright_catalog_enabled", bright_catalog_enabled))
        bright_catalog_margin_deg = float(
            getattr(cfg, "bright_catalog_margin_deg", bright_catalog_margin_deg)
        )
        prefer_parquet = bool(getattr(cfg, "prefer_parquet", prefer_parquet))

    hp = HEALPix(nside=nside, order=order, frame=ICRS())
    pix_indices = np.arange(hp.npix, dtype=np.int64)
    centers = hp.healpix_to_skycoord(pix_indices)

    gaia_cached_tiles: List[int] = []
    bright_cached_tiles: List[int] = []
    cached_bytes = 0
    newest_mtime: Optional[float] = None
    for pix in pix_indices:
        pix_i = int(pix)
        hexkey = _cache_key(kind="healpix_tile", payload={
            "table": table_name,
            "nside": nside,
            "order": order,
            "pix": pix_i,
            "gmax": gmax,
            "columns": list(columns),
        })
        path = _path_for_in(cache_dir, hexkey, prefer_parquet)
        paths = [("gaia", path)]
        if bright_catalog_enabled:
            bright_key = _bright_tile_cache_key(
                nside=nside,
                order=order,
                pix=pix_i,
                vmax=gmax,
            )
            paths.append(("bright", _path_for_in(cache_dir, bright_key, prefer_parquet)))

        for catalog_kind, catalog_path in paths:
            try:
                stat = catalog_path.stat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                log_error(
                    None,
                    f"Gaia coverage: failed to inspect cache tile {catalog_path}",
                    exc,
                    throttle_s=10.0,
                    throttle_key=f"gaia_coverage_stat_{catalog_kind}_{pix_i}",
                )
                continue
            if catalog_kind == "gaia":
                gaia_cached_tiles.append(pix_i)
            else:
                bright_cached_tiles.append(pix_i)
            cached_bytes += int(stat.st_size)
            if newest_mtime is None or float(stat.st_mtime) > newest_mtime:
                newest_mtime = float(stat.st_mtime)

    if bright_catalog_enabled:
        cached_tiles = sorted(set(gaia_cached_tiles) & set(bright_cached_tiles))
    else:
        cached_tiles = gaia_cached_tiles

    required_tiles: List[int] = []
    bright_required_tiles: List[int] = []
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None
    field_radius_deg: Optional[float] = None
    if center_icrs is not None and radius_deg is not None:
        center = normalize_input(center_icrs)
        field_radius_deg = float(radius_deg)
        if math.isfinite(field_radius_deg) and field_radius_deg > 0.0:
            required_tiles = [
                int(pix)
                for pix in hp.cone_search_skycoord(center, Angle(field_radius_deg, u.deg))
            ]
            if bright_catalog_enabled:
                bright_required_tiles = [
                    int(pix)
                    for pix in hp.cone_search_skycoord(
                        center,
                        Angle(field_radius_deg + max(0.0, bright_catalog_margin_deg), u.deg),
                    )
                ]
            center_ra_deg = float(center.ra.deg) % 360.0
            center_dec_deg = float(center.dec.deg)

    gaia_cached_set = set(gaia_cached_tiles)
    bright_cached_set = set(bright_cached_tiles)
    field_cached_tiles = [pix for pix in required_tiles if pix in gaia_cached_set]
    field_missing_tiles = [pix for pix in required_tiles if pix not in gaia_cached_set]
    if bright_catalog_enabled:
        field_cached_tiles = [
            pix for pix in required_tiles
            if pix in gaia_cached_set and pix in bright_cached_set
        ]
        field_missing_tiles = sorted(
            set(pix for pix in required_tiles if pix not in gaia_cached_set)
            | set(pix for pix in bright_required_tiles if pix not in bright_cached_set)
        )
    total_tiles = int(hp.npix)
    tile_area_sq_deg = float(4.0 * math.pi * (180.0 / math.pi) ** 2 / total_tiles)

    return {
        "cache_dir": str(cache_dir),
        "table_name": table_name,
        "columns": tuple(columns),
        "gmax": gmax,
        "nside": nside,
        "order": order,
        "total_tiles": total_tiles,
        "cached_tiles": cached_tiles,
        "cached_tile_count": len(cached_tiles),
        "gaia_cached_tiles": gaia_cached_tiles,
        "gaia_cached_tile_count": len(gaia_cached_tiles),
        "bright_catalog_enabled": bright_catalog_enabled,
        "bright_cached_tiles": bright_cached_tiles,
        "bright_cached_tile_count": len(bright_cached_tiles),
        "bright_catalog_tables": BRIGHT_CATALOG_TABLES,
        "coverage_fraction": float(len(cached_tiles) / total_tiles),
        "covered_area_sq_deg": float(len(cached_tiles) * tile_area_sq_deg),
        "tile_area_sq_deg": tile_area_sq_deg,
        "cached_bytes": cached_bytes,
        "newest_mtime": newest_mtime,
        "tile_ra_deg": np.asarray(centers.ra.deg, dtype=np.float64),
        "tile_dec_deg": np.asarray(centers.dec.deg, dtype=np.float64),
        "field_available": bool(required_tiles) and not field_missing_tiles,
        "field_required_tiles": required_tiles,
        "field_bright_required_tiles": bright_required_tiles,
        "field_cached_tiles": field_cached_tiles,
        "field_missing_tiles": field_missing_tiles,
        "field_radius_deg": field_radius_deg,
        "center_ra_deg": center_ra_deg,
        "center_dec_deg": center_dec_deg,
    }


def _tap_login_only(user: str, password: str) -> None:
    """
    Login only against Gaia TAP endpoint.

    Using Gaia.login() can hang while trying the separate Gaia data server.
    """
    TapPlus.login(Gaia, user=user, password=password, verbose=False)


def _tap_logout_only() -> None:
    """Logout only from Gaia TAP endpoint."""
    TapPlus.logout(Gaia, verbose=False)


# -------------------------
# Cono único (async)
# -------------------------
def gaia_cone_with_mag(
    target,
    radius: Union[float, u.Quantity],
    *,
    gmax: float = 15.0,
    table_name: str = DEFAULT_TABLE,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    auth: Optional[Tuple[str, str]] = None,
    row_limit: int = -1,
    prefer_parquet: bool = True,
    retries: int = 3,
    backoff_s: float = 3.0,
    verbose: bool = True,
) -> Table:
    """
    Cone search con filtro 'phot_g_mean_mag <= gmax' en el servidor (ADQL),
    caché por parámetros. Usa launch_job_async(background=False).
    """
    center = normalize_input(target)
    ra_deg, dec_deg = center.ra.deg, center.dec.deg
    radius_deg = (radius.to_value(u.deg) if isinstance(radius, u.Quantity) else float(radius))

    hexkey = _cache_key(kind="cone", payload={
        "table": table_name, "ra": round(ra_deg, 8), "dec": round(dec_deg, 8),
        "radius": round(radius_deg, 8), "gmax": float(gmax), "columns": list(columns)
    })
    path = _path_for(hexkey, prefer_parquet)
    if path.exists():
        if verbose:
            log_info(None, f"[gaia_cache] HIT {path}")
        return add_bright_star_supplement(
            _load_table(path),
            center_icrs=center,
            radius_deg=radius_deg,
            gmax=gmax,
        )

    Gaia.ROW_LIMIT = row_limit
    cols_sql = ", ".join(columns)
    query = f"""
    SELECT {cols_sql}
    FROM {table_name}
    WHERE phot_g_mean_mag <= {gmax}
      AND 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
          )
    """

    did_login = False
    try:
        if auth:
            if verbose:
                log_info(None, "[gaia_cache] Login TAP-only al Gaia Archive...")
            _tap_login_only(auth[0], auth[1])
            did_login = True

        for attempt in range(1, retries + 1):
            try:
                job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=verbose)
                tab = job.get_results()
                break
            except Exception as e:
                if attempt == retries:
                    raise
                if verbose:
                    log_info(None, f"[gaia_cache] retry {attempt}: {type(e).__name__} -> {e}")
                log_error(None, f"Gaia cache: query retry {attempt} failed", e)
                time.sleep(backoff_s * attempt)

    finally:
        if did_login:
            if verbose:
                log_info(None, "[gaia_cache] Logout TAP-only del Gaia Archive.")
            try:
                _tap_logout_only()
            except Exception as exc:
                log_error(None, "Gaia cache: logout failed", exc)

    # dedup por source_id
    if "source_id" in tab.colnames:
        try:
            import pandas as pd
            tab = Table.from_pandas(tab.to_pandas().drop_duplicates(subset=["source_id"]))
        except Exception as exc:
            log_error(None, "Gaia cache: pandas dedup failed; falling back to python", exc)
            seen = set()
            keep = []
            for i, sid in enumerate(tab["source_id"]):
                sid = int(sid)
                if sid not in seen:
                    seen.add(sid)
                    keep.append(i)
            tab = tab[keep]

    _save_table(tab, path)
    if verbose:
        log_info(None, f"[gaia_cache] MISS -> saved {len(tab)} rows to {path}")
    return add_bright_star_supplement(
        tab,
        center_icrs=center,
        radius_deg=radius_deg,
        gmax=gmax,
    )


# -------------------------
# HEALPix helpers
# -------------------------
def _ensure_healpix_available() -> None:
    if HEALPix is None:
        raise ImportError("astropy-healpix no está disponible. Instala 'astropy-healpix'.")


def _adql_polygon_from_skycoord(poly: SkyCoord) -> str:
    """
    Convierte vértices SkyCoord a ADQL POLYGON('ICRS', lon1,lat1, ..., lonN,latN).
    Acepta arrays con cualquier forma; se aplana.
    """
    import numpy as np
    lon = np.asarray(poly.ra.deg).ravel()
    lat = np.asarray(poly.dec.deg).ravel()
    pairs = ", ".join(f"{float(lon_i):.10f},{float(lat_i):.10f}" for lon_i, lat_i in zip(lon, lat))
    return f"POLYGON('ICRS', {pairs})"


def _query_healpix_tile_async(
    *,
    table_name: str,
    columns: Sequence[str],
    gmax: float,
    poly_sky: SkyCoord,
    row_limit: int,
    retries: int,
    backoff_s: float,
    verbose: bool,
) -> Table:
    cols_sql = ", ".join(columns)
    poly_adql = _adql_polygon_from_skycoord(poly_sky)
    query = f"""
    SELECT {cols_sql}
    FROM {table_name}
    WHERE phot_g_mean_mag <= {gmax}
      AND 1=CONTAINS(POINT('ICRS', ra, dec), {poly_adql})
    """
    Gaia.ROW_LIMIT = row_limit

    for attempt in range(1, retries + 1):
        try:
            job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=verbose)
            return job.get_results()
        except Exception as e:
            if attempt == retries:
                raise
            if verbose:
                log_info(None, f"[gaia_healpix] retry {attempt}: {type(e).__name__} -> {e}")
            log_error(None, f"Gaia healpix: query retry {attempt} failed", e)
            time.sleep(backoff_s * attempt)


def _launch_catalog_query(
    query: str,
    *,
    row_limit: int,
    retries: int,
    backoff_s: float,
    verbose: bool,
    label: str,
) -> Table:
    Gaia.ROW_LIMIT = row_limit
    for attempt in range(1, retries + 1):
        try:
            job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=verbose)
            return job.get_results()
        except Exception as exc:
            if attempt == retries:
                raise
            if verbose:
                log_info(None, f"[{label}] retry {attempt}: {type(exc).__name__} -> {exc}")
            log_error(None, f"{label}: query retry {attempt} failed", exc)
            time.sleep(backoff_s * attempt)
    raise RuntimeError(f"{label}: query did not return")


def _query_bright_catalog_tile_async(
    *,
    vmax: float,
    poly_sky: SkyCoord,
    row_limit: int,
    retries: int,
    backoff_s: float,
    verbose: bool,
) -> Table:
    """Download and normalize Hipparcos plus Tycho-2 for one HEALPix tile."""
    poly_adql = _adql_polygon_from_skycoord(poly_sky)
    hip_query = f"""
    SELECT hip, ra, de, pmra, pmde, vmag
    FROM public.hipparcos
    WHERE vmag <= {float(vmax)}
      AND 1=CONTAINS(POINT('ICRS', ra, de), {poly_adql})
    """
    # Query all Tycho-2 rows in the tile. Some entries have only BT, so a
    # server-side VT cut would discard potentially bright red sources.
    tycho_query = f"""
    SELECT id_tycho, ra_mdeg, de_mdeg, pm_ra, pm_de, bt_mag, vt_mag
    FROM public.tycho2
    WHERE 1=CONTAINS(POINT('ICRS', ra_mdeg, de_mdeg), {poly_adql})
    """
    hip = _normalize_hipparcos(
        _launch_catalog_query(
            hip_query,
            row_limit=row_limit,
            retries=retries,
            backoff_s=backoff_s,
            verbose=verbose,
            label="hipparcos_healpix",
        ),
        vmax=vmax,
    )
    tycho = _normalize_tycho2(
        _launch_catalog_query(
            tycho_query,
            row_limit=row_limit,
            retries=retries,
            backoff_s=backoff_s,
            verbose=verbose,
            label="tycho2_healpix",
        ),
        vmax=vmax,
    )
    return merge_bright_catalog(hip, tycho)


def bright_healpix_cone_with_mag(
    *,
    center_icrs: SkyCoord,
    radius_deg: float,
    cfg=None,
    mag_limit: Optional[float] = None,
) -> Table:
    """Load a local-only Hipparcos/Tycho-2 cone from the bright tile cache."""
    _ensure_healpix_available()
    center = normalize_input(center_icrs)
    radius = float(radius_deg)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_deg must be positive")

    cache_dir = _DEFAULT_CACHE_DIR
    nside = 16
    order = "ring"
    cache_vmax = 15.0
    prefer_parquet = True
    margin_deg = 0.15
    if cfg is not None:
        cache_dir = Path(getattr(cfg, "cache_dir", cache_dir)).expanduser().resolve()
        nside = int(getattr(cfg, "nside", nside))
        order = str(getattr(cfg, "order", order))
        cache_vmax = float(getattr(cfg, "gmax", cache_vmax))
        prefer_parquet = bool(getattr(cfg, "prefer_parquet", prefer_parquet))
        margin_deg = float(getattr(cfg, "bright_catalog_margin_deg", margin_deg))

    hp = HEALPix(nside=nside, order=order, frame=ICRS())
    pixels = hp.cone_search_skycoord(
        center,
        Angle(radius + max(0.0, margin_deg), u.deg),
    )
    parts: List[Table] = []
    missing_paths: List[Path] = []
    missing_tiles: List[int] = []
    for pix in pixels:
        pix_i = int(pix)
        key = _bright_tile_cache_key(
            nside=nside,
            order=order,
            pix=pix_i,
            vmax=cache_vmax,
        )
        path = _path_for_in(cache_dir, key, prefer_parquet)
        if not path.exists():
            missing_paths.append(path)
            missing_tiles.append(pix_i)
            continue
        parts.append(_load_table(path))

    if missing_paths:
        raise GaiaCacheMissError(missing_paths, missing_tiles=missing_tiles)
    if not parts:
        return Table(
            names=list(DEFAULT_COLUMNS),
            dtype=("int64", "float64", "float64", "float64"),
        )

    full = vstack(parts, join_type="exact", metadata_conflicts="silent")
    if len(full) == 0:
        return full
    _, unique_indices = np.unique(
        np.asarray(full["source_id"], dtype=np.int64),
        return_index=True,
    )
    full = full[np.sort(unique_indices)]
    coords = SkyCoord(
        ra=np.asarray(full["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(full["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    keep = np.asarray(coords.separation(center).deg <= radius, dtype=bool)
    limit = cache_vmax if mag_limit is None else min(float(mag_limit), cache_vmax)
    mags = _finite_float_column(full, "phot_g_mean_mag")
    keep &= np.isfinite(mags) & (mags <= limit)
    return full[keep]


# -------------------------
# Mosaico HEALPix (async, login único)
# -------------------------
def gaia_healpix_cone_with_mag(
    target=None,
    radius: Optional[Union[float, u.Quantity]] = None,
    *,
    center_icrs: Optional[SkyCoord] = None,
    radius_deg: Optional[float] = None,
    cfg=None,
    progress_cb=None,
    gmax: float = 15.0,
    nside: int = 16,
    order: str = "ring",
    table_name: str = DEFAULT_TABLE,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    auth: Optional[Tuple[str, str]] = None,
    row_limit: int = -1,
    prefer_parquet: bool = True,
    retries: int = 3,
    backoff_s: float = 3.0,
    verbose: bool = True,
) -> Table:
    """
    Mosaico HEALPix de Gaia G<=gmax complementado por Hipparcos/Tycho V<=gmax.

    Logging:
      - Si TODAS las teselas están en caché: no imprime nada (aunque verbose=True).
      - Si falta al menos una tesela: imprime progreso, login/logout y resumen final (si verbose=True).
    """
    _ensure_healpix_available()

    if cfg is not None:
        cache_dir = getattr(cfg, "cache_dir", None)
        if cache_dir:
            set_cache_dir(cache_dir)
        table_name = getattr(cfg, "table_name", table_name)
        columns = getattr(cfg, "columns", columns)
        gmax = float(getattr(cfg, "gmax", gmax))
        nside = int(getattr(cfg, "nside", nside))
        order = getattr(cfg, "order", order)
        bright_catalog_enabled = bool(getattr(cfg, "bright_catalog_enabled", True))
        bright_catalog_margin_deg = float(getattr(cfg, "bright_catalog_margin_deg", 0.15))
        prefer_parquet = bool(getattr(cfg, "prefer_parquet", prefer_parquet))
        row_limit = int(getattr(cfg, "row_limit", row_limit))
        retries = int(getattr(cfg, "retries", retries))
        backoff_s = float(getattr(cfg, "backoff_s", backoff_s))
        download_missing_tiles = bool(getattr(cfg, "download_missing_tiles", True))
    else:
        download_missing_tiles = True
        bright_catalog_enabled = True
        bright_catalog_margin_deg = 0.15

    if center_icrs is not None:
        center = normalize_input(center_icrs)
    elif target is not None:
        center = normalize_input(target)
    else:
        raise ValueError("gaia_healpix_cone_with_mag: missing target/center_icrs")

    if radius_deg is None:
        if radius is None:
            raise ValueError("gaia_healpix_cone_with_mag: missing radius/radius_deg")
        radius_deg = (radius.to_value(u.deg) if isinstance(radius, u.Quantity) else float(radius))
    else:
        radius_deg = float(radius_deg)

    hp = HEALPix(nside=nside, order=order, frame=center.frame)
    pix_indices = hp.cone_search_skycoord(center, Angle(radius_deg, u.deg))
    if bright_catalog_enabled:
        bright_pix_indices = hp.cone_search_skycoord(
            center,
            Angle(radius_deg + max(0.0, bright_catalog_margin_deg), u.deg),
        )
    else:
        bright_pix_indices = np.asarray([], dtype=np.int64)

    # --- Pre-chequeo de caché: decide si se hará login y si se loggeará ---
    missing: List[int] = []
    bright_missing: List[int] = []
    cache_paths: Dict[int, Path] = {}
    bright_cache_paths: Dict[int, Path] = {}
    for pix in pix_indices:
        pix_i = int(pix)
        hexkey = _cache_key(kind="healpix_tile", payload={
            "table": table_name, "nside": int(nside), "order": str(order),
            "pix": pix_i, "gmax": float(gmax), "columns": list(columns)
        })
        path = _path_for(hexkey, prefer_parquet)
        cache_paths[pix_i] = path
        if not path.exists():
            missing.append(pix_i)

    for pix in bright_pix_indices:
        pix_i = int(pix)
        hexkey = _bright_tile_cache_key(
            nside=nside,
            order=order,
            pix=pix_i,
            vmax=gmax,
        )
        path = _path_for(hexkey, prefer_parquet)
        bright_cache_paths[pix_i] = path
        if not path.exists():
            bright_missing.append(pix_i)

    need_download = bool(missing or bright_missing)

    if need_download and not download_missing_tiles:
        missing_paths = [cache_paths[pix] for pix in missing]
        missing_paths.extend(bright_cache_paths[pix] for pix in bright_missing)
        raise GaiaCacheMissError(
            missing_paths,
            missing_tiles=sorted(set(missing) | set(bright_missing)),
        )

    if verbose and need_download:
        log_info(
            None,
            f"[gaia_healpix] nside={nside}, Gaia tiles={len(pix_indices)}, "
            f"bright tiles={len(bright_pix_indices)}",
        )
        if progress_cb:
            progress_cb("gaia:healpix:start", {
                "tiles": float(len(pix_indices)),
                "missing": float(len(missing) + len(bright_missing)),
            })

    did_login = False
    try:
        # Login SOLO si hay algo que descargar (y auth provisto)
        if auth and need_download:
            if verbose:
                log_info(
                    None,
                    "[gaia_healpix] Login TAP-only único al Gaia Archive... "
                    f"(missing tiles={len(missing) + len(bright_missing)})",
                )
            _tap_login_only(auth[0], auth[1])
            did_login = True
        elif need_download and auth is None and getattr(Gaia, "login", None) is not None:
            # allow anonymous downloads; only error out if caller explicitly wants auth
            pass

        parts: List[Table] = []
        bright_parts: List[Table] = []
        for i, pix in enumerate(pix_indices, 1):
            pix_i = int(pix)
            path = cache_paths[pix_i]

            if path.exists():
                # Si hay descargas, puede ser útil indicar el directorio base una vez
                if verbose and need_download and i == 1:
                    log_info(None, f"[gaia_healpix] HIT first tile -> {path.parent}")
                tab = _load_table(path)
            else:
                if verbose:
                    # Ojo: este bloque solo corre si need_download=True (por definición)
                    log_info(None, f"[gaia_healpix] Query tile {i}/{len(pix_indices)} (pix={pix_i})")
                if progress_cb:
                    progress_cb("gaia:healpix:tile", {"tile": float(i), "tiles": float(len(pix_indices)), "pix": float(pix_i)})
                poly = hp.boundaries_skycoord(pix, step=1)
                tab = _query_healpix_tile_async(
                    table_name=table_name,
                    columns=columns,
                    gmax=gmax,
                    poly_sky=poly,
                    row_limit=row_limit,
                    retries=retries,
                    backoff_s=backoff_s,
                    verbose=False,  # evita logs internos por tile
                )
                _save_table(tab, path)

            parts.append(tab)

        for i, pix in enumerate(bright_pix_indices, 1):
            pix_i = int(pix)
            path = bright_cache_paths[pix_i]
            if path.exists():
                tab = _load_table(path)
            else:
                if verbose:
                    log_info(
                        None,
                        f"[gaia_healpix] Query bright tile {i}/{len(bright_pix_indices)} "
                        f"(pix={pix_i})",
                    )
                poly = hp.boundaries_skycoord(pix, step=1)
                tab = _query_bright_catalog_tile_async(
                    vmax=gmax,
                    poly_sky=poly,
                    row_limit=row_limit,
                    retries=retries,
                    backoff_s=backoff_s,
                    verbose=False,
                )
                _save_table(tab, path)
            bright_parts.append(tab)

    finally:
        if did_login:
            if verbose:
                log_info(None, "[gaia_healpix] Logout TAP-only del Gaia Archive.")
            try:
                _tap_logout_only()
            except Exception as exc:
                log_error(None, "Gaia healpix: logout failed", exc)

    if parts:
        full = vstack(parts, join_type="outer", metadata_conflicts="silent")
    else:
        full = Table(
            names=list(DEFAULT_COLUMNS),
            dtype=("int64", "float64", "float64", "float64"),
        )

    # Deduplicación por source_id
    if "source_id" in full.colnames:
        try:
            import pandas as pd
            full = Table.from_pandas(full.to_pandas().drop_duplicates(subset=["source_id"]))
        except Exception as exc:
            log_error(None, "Gaia healpix: pandas dedup failed; falling back to python", exc)
            seen = set()
            keep = []
            for j, sid in enumerate(full["source_id"]):
                sid = int(sid)
                if sid not in seen:
                    seen.add(sid)
                    keep.append(j)
            full = full[keep]

    # Recorte fino al círculo exacto
    if len(full) > 0:
        sc = SkyCoord(full["ra"] * u.deg, full["dec"] * u.deg, frame="icrs")
        sep = sc.separation(center).deg
        full = full[sep <= radius_deg]

    if bright_parts:
        bright_full = vstack(bright_parts, join_type="exact", metadata_conflicts="silent")
        if len(bright_full) > 0:
            _, unique_indices = np.unique(
                np.asarray(bright_full["source_id"], dtype=np.int64),
                return_index=True,
            )
            bright_full = bright_full[np.sort(unique_indices)]
            bright_coords = SkyCoord(
                bright_full["ra"] * u.deg,
                bright_full["dec"] * u.deg,
                frame="icrs",
            )
            bright_full = bright_full[bright_coords.separation(center).deg <= radius_deg]
            full = merge_bright_catalog(full, bright_full)

    full = add_bright_star_supplement(
        full,
        center_icrs=center,
        radius_deg=radius_deg,
        gmax=gmax,
    )

    if verbose and need_download:
        log_info(None, f"[gaia_healpix] Final combined rows (limit={gmax}): {len(full)}")

    return full
