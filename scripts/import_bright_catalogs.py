#!/usr/bin/env python3
"""Download Hipparcos/Tycho-2 from CDS and build the local HEALPix cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
from astropy_healpix import HEALPix
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import AppConfig  # noqa: E402
from gaia_cache import (  # noqa: E402
    DEFAULT_COLUMNS,
    _bright_tile_cache_key,
    _normalize_hipparcos,
    _normalize_tycho2,
    _path_for_in,
    _save_table,
    merge_bright_catalog,
)


HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat"
TYCHO2_URL_TEMPLATE = "https://cdsarc.cds.unistra.fr/ftp/I/259/tyc2.dat.{part:02d}.gz"
HIPPARCOS_EXPECTED_ROWS = 118_218
TYCHO2_EXPECTED_ROWS = 2_539_913

HIPPARCOS_COLSPECS = (
    (8, 14),    # HIP
    (41, 46),   # Vmag
    (51, 63),   # RAdeg
    (64, 76),   # DEdeg
    (87, 95),   # pmRA
    (96, 104),  # pmDE
)
HIPPARCOS_NAMES = ("hip", "vmag", "ra", "de", "pmra", "pmde")

TYCHO2_COLSPECS = (
    (0, 4),     # TYC1
    (5, 10),    # TYC2
    (11, 12),   # TYC3
    (15, 27),   # RAmdeg
    (28, 40),   # DEmdeg
    (41, 48),   # pmRA
    (49, 56),   # pmDE
    (110, 116), # BTmag
    (123, 129), # VTmag
)
TYCHO2_NAMES = (
    "tyc1",
    "tyc2",
    "tyc3",
    "ra_mdeg",
    "de_mdeg",
    "pm_ra",
    "pm_de",
    "bt_mag",
    "vt_mag",
)


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    path: Path


def _empty_catalog() -> Table:
    return Table(
        names=list(DEFAULT_COLUMNS),
        dtype=("int64", "float64", "float64", "float64"),
    )


def _download_one(spec: DownloadSpec, *, timeout_s: float = 60.0) -> Path:
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    remote_size = None
    try:
        response = requests.head(spec.url, allow_redirects=True, timeout=timeout_s)
        response.raise_for_status()
        if response.headers.get("Content-Length"):
            remote_size = int(response.headers["Content-Length"])
    except requests.RequestException:
        pass

    if spec.path.exists() and remote_size is not None and spec.path.stat().st_size == remote_size:
        return spec.path

    partial = spec.path.with_name(f"{spec.path.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    mode = "ab" if offset else "wb"

    with requests.get(
        spec.url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            mode = "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if remote_size is not None and partial.stat().st_size != remote_size:
        raise RuntimeError(
            f"Incomplete download for {spec.url}: "
            f"{partial.stat().st_size} of {remote_size} bytes"
        )
    os.replace(partial, spec.path)
    return spec.path


def download_catalogs(raw_dir: Path, *, workers: int = 4) -> tuple[Path, list[Path]]:
    specs = [
        DownloadSpec(HIPPARCOS_URL, raw_dir / "hip_main.dat"),
        *[
            DownloadSpec(
                TYCHO2_URL_TEMPLATE.format(part=part),
                raw_dir / f"tyc2.dat.{part:02d}.gz",
            )
            for part in range(20)
        ],
    ]
    completed: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(_download_one, spec): spec for spec in specs}
        for index, future in enumerate(as_completed(futures), 1):
            path = future.result()
            completed.append(path)
            print(f"[download] {index:02d}/{len(specs)} {path.name}", flush=True)

    hip_path = raw_dir / "hip_main.dat"
    tycho_paths = [raw_dir / f"tyc2.dat.{part:02d}.gz" for part in range(20)]
    return hip_path, tycho_paths


def _numeric_frame(path: Path, *, colspecs, names, compression=None) -> pd.DataFrame:
    frame = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        header=None,
        compression=compression,
    )
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def parse_hipparcos(path: Path, *, vmax: float) -> tuple[Table, int]:
    frame = _numeric_frame(
        path,
        colspecs=HIPPARCOS_COLSPECS,
        names=HIPPARCOS_NAMES,
    )
    raw_rows = len(frame)
    tab = Table.from_pandas(frame)
    return _normalize_hipparcos(tab, vmax=vmax), raw_rows


def _tycho_numeric_id(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["tyc1"].to_numpy(np.int64) * 1_000_000
        + frame["tyc2"].to_numpy(np.int64) * 10
        + frame["tyc3"].to_numpy(np.int64)
    )


def parse_tycho2(paths: Iterable[Path], *, vmax: float) -> tuple[Table, int]:
    parts: list[Table] = []
    raw_rows = 0
    paths = list(paths)
    for index, path in enumerate(paths, 1):
        frame = _numeric_frame(
            path,
            colspecs=TYCHO2_COLSPECS,
            names=TYCHO2_NAMES,
            compression="gzip",
        )
        raw_rows += len(frame)
        valid_id = frame[["tyc1", "tyc2", "tyc3"]].notna().all(axis=1)
        frame = frame.loc[valid_id].copy()
        frame["id_tycho"] = _tycho_numeric_id(frame)
        parts.append(_normalize_tycho2(Table.from_pandas(frame), vmax=vmax))
        print(
            f"[parse] Tycho-2 {index:02d}/{len(paths)} "
            f"raw={raw_rows:,} selected={sum(len(part) for part in parts):,}",
            flush=True,
        )
    if not parts:
        return _empty_catalog(), raw_rows
    return vstack(parts, join_type="exact", metadata_conflicts="silent"), raw_rows


def _atomic_save_table(tab: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    _save_table(tab, temp)
    os.replace(temp, path)


def write_tiles(
    catalog: Table,
    *,
    cache_dir: Path,
    nside: int,
    order: str,
    vmax: float,
    prefer_parquet: bool,
) -> dict[str, int]:
    hp = HEALPix(nside=nside, order=order, frame=ICRS())
    coords = SkyCoord(
        ra=np.asarray(catalog["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(catalog["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    pixels = np.asarray(hp.skycoord_to_healpix(coords), dtype=np.int64)
    row_order = np.argsort(pixels, kind="stable")
    sorted_pixels = pixels[row_order]
    starts = np.searchsorted(sorted_pixels, np.arange(hp.npix), side="left")
    stops = np.searchsorted(sorted_pixels, np.arange(hp.npix), side="right")

    total_bytes = 0
    for pix in range(hp.npix):
        key = _bright_tile_cache_key(
            nside=nside,
            order=order,
            pix=pix,
            vmax=vmax,
        )
        path = _path_for_in(cache_dir, key, prefer_parquet)
        indices = row_order[starts[pix]:stops[pix]]
        tile = catalog[indices] if len(indices) else _empty_catalog()
        _atomic_save_table(tile, path)
        total_bytes += path.stat().st_size
        if (pix + 1) % 256 == 0 or pix + 1 == hp.npix:
            print(f"[tiles] {pix + 1:,}/{hp.npix:,}", flush=True)
    return {
        "tiles": int(hp.npix),
        "rows": int(len(catalog)),
        "bytes": int(total_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    cfg = AppConfig().platesolving
    cache_dir = Path(args.cache_dir or cfg.cache_dir).expanduser().resolve()
    raw_dir = Path(
        args.raw_dir or cache_dir / "bulk_bright_catalogs"
    ).expanduser().resolve()

    started = time.monotonic()
    hip_path, tycho_paths = download_catalogs(raw_dir, workers=args.workers)

    hip, hip_raw_rows = parse_hipparcos(hip_path, vmax=cfg.gmax)
    print(
        f"[parse] Hipparcos raw={hip_raw_rows:,} selected={len(hip):,}",
        flush=True,
    )
    if hip_raw_rows != HIPPARCOS_EXPECTED_ROWS:
        raise RuntimeError(
            f"Hipparcos row count mismatch: {hip_raw_rows} != {HIPPARCOS_EXPECTED_ROWS}"
        )

    tycho, tycho_raw_rows = parse_tycho2(tycho_paths, vmax=cfg.gmax)
    if tycho_raw_rows != TYCHO2_EXPECTED_ROWS:
        raise RuntimeError(
            f"Tycho-2 row count mismatch: {tycho_raw_rows} != {TYCHO2_EXPECTED_ROWS}"
        )

    combined = merge_bright_catalog(hip, tycho)
    stats = write_tiles(
        combined,
        cache_dir=cache_dir,
        nside=cfg.nside,
        order=cfg.order,
        vmax=cfg.gmax,
        prefer_parquet=cfg.prefer_parquet,
    )

    if not args.keep_raw:
        for path in [hip_path, *tycho_paths]:
            path.unlink(missing_ok=True)
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    elapsed = time.monotonic() - started
    print(
        "[done] "
        f"rows={stats['rows']:,} tiles={stats['tiles']:,} "
        f"size={stats['bytes'] / (1024 ** 2):.1f} MiB "
        f"elapsed={elapsed / 60.0:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
