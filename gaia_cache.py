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
import time
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List, Dict

from logging_utils import log_error
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.table import Table, vstack

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
    except ValueError as exc:
        log_error(None, "Gaia cache: failed to resolve repo-relative path", exc)
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
    ext = "parquet" if (prefer_parquet and _HAS_PARQUET) else "ecsv"
    return _DEFAULT_CACHE_DIR.joinpath(hexkey[:2], hexkey[2:4], f"{hexkey}.{ext}")


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
            print(f"[gaia_cache] HIT {path}")
        return _load_table(path)

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
                print("[gaia_cache] Login al Gaia Archive…")
            Gaia.login(user=auth[0], password=auth[1])
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
                    print(f"[gaia_cache] retry {attempt}: {type(e).__name__} -> {e}")
                log_error(None, f"Gaia cache: query retry {attempt} failed", e)
                time.sleep(backoff_s * attempt)

    finally:
        if did_login:
            if verbose:
                print("[gaia_cache] Logout del Gaia Archive.")
            try:
                Gaia.logout()
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
        print(f"[gaia_cache] MISS -> saved {len(tab)} rows to {path}")
    return tab


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
                print(f"[gaia_healpix] retry {attempt}: {type(e).__name__} -> {e}")
            log_error(None, f"Gaia healpix: query retry {attempt} failed", e)
            time.sleep(backoff_s * attempt)


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
    HEALPix mosaico con filtro 'phot_g_mean_mag <= gmax'.

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
        prefer_parquet = bool(getattr(cfg, "prefer_parquet", prefer_parquet))
        row_limit = int(getattr(cfg, "row_limit", row_limit))
        retries = int(getattr(cfg, "retries", retries))
        backoff_s = float(getattr(cfg, "backoff_s", backoff_s))
        download_missing_tiles = bool(getattr(cfg, "download_missing_tiles", True))
    else:
        download_missing_tiles = True

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

    # --- Pre-chequeo de caché: decide si se hará login y si se loggeará ---
    missing: List[int] = []
    cache_paths: Dict[int, Path] = {}
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

    need_download = (len(missing) > 0)

    if need_download and not download_missing_tiles:
        missing_paths = [cache_paths[pix] for pix in missing]
        raise GaiaCacheMissError(missing_paths, missing_tiles=missing)

    if verbose and need_download:
        print(f"[gaia_healpix] nside={nside}, tiles={len(pix_indices)}")
        if progress_cb:
            progress_cb("gaia:healpix:start", {"tiles": float(len(pix_indices)), "missing": float(len(missing))})

    did_login = False
    try:
        # Login SOLO si hay algo que descargar (y auth provisto)
        if auth and need_download:
            if verbose:
                print(f"[gaia_healpix] Login único al Gaia Archive… (missing tiles={len(missing)})")
            Gaia.login(user=auth[0], password=auth[1])
            did_login = True
        elif need_download and auth is None and getattr(Gaia, "login", None) is not None:
            # allow anonymous downloads; only error out if caller explicitly wants auth
            pass

        parts: List[Table] = []
        for i, pix in enumerate(pix_indices, 1):
            pix_i = int(pix)
            path = cache_paths[pix_i]

            if path.exists():
                # Si hay descargas, puede ser útil indicar el directorio base una vez
                if verbose and need_download and i == 1:
                    print(f"[gaia_healpix] HIT first tile -> {path.parent}")
                tab = _load_table(path)
            else:
                if verbose:
                    # Ojo: este bloque solo corre si need_download=True (por definición)
                    print(f"[gaia_healpix] Query tile {i}/{len(pix_indices)} (pix={pix_i})")
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

    finally:
        if did_login:
            if verbose:
                print("[gaia_healpix] Logout del Gaia Archive.")
            try:
                Gaia.logout()
            except Exception as exc:
                log_error(None, "Gaia healpix: logout failed", exc)

    if not parts:
        return Table(names=list(columns), dtype=[float] * len(columns))

    full = vstack(parts, join_type="outer", metadata_conflicts="silent")

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
    sc = SkyCoord(full["ra"] * u.deg, full["dec"] * u.deg, frame="icrs")
    sep = sc.separation(center).deg
    full = full[sep <= radius_deg]

    if verbose and need_download:
        print(f"[gaia_healpix] Final rows (G<={gmax}): {len(full)}")

    return full
