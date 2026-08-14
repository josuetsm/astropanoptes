#!/usr/bin/env python3
"""Download the full Gaia HEALPix cache used by the simulator."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
import astropy.units as u
from astropy_healpix import HEALPix
from astropy_healpix.core import nested_to_ring, ring_to_nested

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaia_cache import (  # noqa: E402
    DEFAULT_COLUMNS,
    DEFAULT_TABLE,
    _cache_key,
    _launch_catalog_query,
    _path_for_in,
    _query_healpix_tile_async,
    _save_table,
    _tap_login_only,
    _tap_logout_only,
    load_gaia_auth,
    set_cache_dir,
)


@dataclass
class Progress:
    total_tiles: int
    missing_at_start: int
    cached_at_start: int
    completed_this_run: int = 0
    failed_this_run: int = 0
    skipped_existing: int = 0
    rows_downloaded_this_run: int = 0
    bytes_written_this_run: int = 0
    current_pix: int | None = None
    last_pix: int | None = None
    last_rows: int | None = None
    last_path: str | None = None
    last_error: str | None = None
    relogin_count: int = 0
    last_relogin_error: str | None = None
    total_chunks: int | None = None
    missing_chunks_at_start: int | None = None
    completed_chunks_this_run: int = 0
    failed_chunks_this_run: int = 0
    current_chunk_pix: int | None = None
    last_chunk_pix: int | None = None
    last_chunk_rows: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gaia_tile_cache_path(
    *,
    cache_dir: Path,
    table_name: str,
    columns: Sequence[str],
    gmax: float,
    nside: int,
    order: str,
    pix: int,
    prefer_parquet: bool,
) -> Path:
    key = _cache_key(
        kind="healpix_tile",
        payload={
            "table": table_name,
            "nside": int(nside),
            "order": str(order),
            "pix": int(pix),
            "gmax": float(gmax),
            "columns": list(columns),
        },
    )
    return _path_for_in(cache_dir, key, prefer_parquet)


def _write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _save_table_atomic(tab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp{path.suffix}")
    try:
        _save_table(tab, tmp)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


_AUTH_SESSION_ERROR_MARKERS = (
    "401",
    "403",
    "auth",
    "credential",
    "forbidden",
    "invalid session",
    "login",
    "logged in",
    "not authorized",
    "session",
    "token",
    "unauthorized",
)


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def _looks_like_auth_session_error(exc: BaseException) -> bool:
    text = _exception_chain_text(exc)
    return any(marker in text for marker in _AUTH_SESSION_ERROR_MARKERS)


def _logout_best_effort() -> None:
    try:
        _tap_logout_only()
    except Exception:
        pass


def _download_tile_with_optional_relogin(
    *,
    args: argparse.Namespace,
    poly_sky,
    auth: tuple[str, str] | None,
    progress: Progress,
):
    try:
        return _query_healpix_tile_async(
            table_name=args.table_name,
            columns=args.columns,
            gmax=args.gmax,
            poly_sky=poly_sky,
            row_limit=args.row_limit,
            retries=args.retries,
            backoff_s=args.backoff_s,
            verbose=False,
        )
    except Exception as exc:
        if auth is None or not _looks_like_auth_session_error(exc):
            raise
        progress.relogin_count += 1
        progress.last_relogin_error = f"{type(exc).__name__}: {exc}"
        print(
            "[gaia-full] Gaia TAP session/auth error; relogin and retry current tile",
            flush=True,
        )
        _logout_best_effort()
        _tap_login_only(auth[0], auth[1])
        return _query_healpix_tile_async(
            table_name=args.table_name,
            columns=args.columns,
            gmax=args.gmax,
            poly_sky=poly_sky,
            row_limit=args.row_limit,
            retries=args.retries,
            backoff_s=args.backoff_s,
            verbose=False,
        )


def _source_id_bounds_for_nested_chunk(chunk_pix: int, chunk_nside: int) -> tuple[int, int]:
    level = int(math.log2(int(chunk_nside)))
    shift = 35 + 2 * (12 - level)
    return int(chunk_pix) << shift, (int(chunk_pix) + 1) << shift


def _query_source_id_chunk_async(
    *,
    args: argparse.Namespace,
    source_id_min: int,
    source_id_max: int,
) -> Table:
    cols_sql = ", ".join(args.columns)
    query = f"""
    SELECT {cols_sql}
    FROM {args.table_name}
    WHERE phot_g_mean_mag <= {float(args.gmax)}
      AND source_id >= {int(source_id_min)}
      AND source_id < {int(source_id_max)}
    """
    return _launch_catalog_query(
        query,
        row_limit=args.row_limit,
        retries=args.retries,
        backoff_s=args.backoff_s,
        verbose=False,
        label="gaia_sourceid_chunk",
    )


def _download_source_id_chunk_with_optional_relogin(
    *,
    args: argparse.Namespace,
    source_id_min: int,
    source_id_max: int,
    auth: tuple[str, str] | None,
    progress: Progress,
) -> Table:
    try:
        return _query_source_id_chunk_async(
            args=args,
            source_id_min=source_id_min,
            source_id_max=source_id_max,
        )
    except Exception as exc:
        if auth is None or not _looks_like_auth_session_error(exc):
            raise
        progress.relogin_count += 1
        progress.last_relogin_error = f"{type(exc).__name__}: {exc}"
        print(
            "[gaia-full] Gaia TAP session/auth error; relogin and retry current chunk",
            flush=True,
        )
        _logout_best_effort()
        _tap_login_only(auth[0], auth[1])
        return _query_source_id_chunk_async(
            args=args,
            source_id_min=source_id_min,
            source_id_max=source_id_max,
        )


def _progress_payload(
    *,
    status: str,
    started_at: str,
    args: argparse.Namespace,
    progress: Progress,
) -> dict:
    now = time.time()
    elapsed_s = max(0.0, now - args._started_monotonic)
    done = progress.completed_this_run + progress.failed_this_run
    remaining = max(0, progress.missing_at_start - done)
    avg_s = elapsed_s / done if done else None
    eta_s = avg_s * remaining if avg_s is not None else None
    return {
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": None if eta_s is None else round(float(eta_s), 1),
        "cache_dir": str(args.cache_dir),
        "table_name": args.table_name,
        "columns": list(args.columns),
        "gmax": float(args.gmax),
        "nside": int(args.nside),
        "order": args.order,
        "strategy": args.strategy,
        "chunk_nside": args.chunk_nside,
        "tile_order": "random" if args.shuffle_tiles else "sequential",
        "shuffle_seed": args.shuffle_seed if args.shuffle_tiles else None,
        "total_tiles": progress.total_tiles,
        "cached_at_start": progress.cached_at_start,
        "missing_at_start": progress.missing_at_start,
        "completed_this_run": progress.completed_this_run,
        "failed_this_run": progress.failed_this_run,
        "skipped_existing": progress.skipped_existing,
        "remaining_estimate": remaining,
        "rows_downloaded_this_run": progress.rows_downloaded_this_run,
        "bytes_written_this_run": progress.bytes_written_this_run,
        "current_pix": progress.current_pix,
        "last_pix": progress.last_pix,
        "last_rows": progress.last_rows,
        "last_path": progress.last_path,
        "last_error": progress.last_error,
        "relogin_count": progress.relogin_count,
        "last_relogin_error": progress.last_relogin_error,
        "total_chunks": progress.total_chunks,
        "missing_chunks_at_start": progress.missing_chunks_at_start,
        "completed_chunks_this_run": progress.completed_chunks_this_run,
        "failed_chunks_this_run": progress.failed_chunks_this_run,
        "current_chunk_pix": progress.current_chunk_pix,
        "last_chunk_pix": progress.last_chunk_pix,
        "last_chunk_rows": progress.last_chunk_rows,
    }


def _iter_tile_ids(total_tiles: int, *, start_pix: int, stop_pix: int | None, limit: int | None) -> Iterable[int]:
    stop = total_tiles if stop_pix is None else min(total_tiles, int(stop_pix))
    count = 0
    for pix in range(max(0, int(start_pix)), stop):
        if limit is not None and count >= limit:
            break
        count += 1
        yield pix


def _existing_tiles(args: argparse.Namespace, total_tiles: int) -> tuple[list[int], list[int], dict[int, Path]]:
    paths: dict[int, Path] = {}
    existing: list[int] = []
    missing: list[int] = []
    for pix in _iter_tile_ids(
        total_tiles,
        start_pix=args.start_pix,
        stop_pix=args.stop_pix,
        limit=args.limit,
    ):
        path = _gaia_tile_cache_path(
            cache_dir=args.cache_dir,
            table_name=args.table_name,
            columns=args.columns,
            gmax=args.gmax,
            nside=args.nside,
            order=args.order,
            pix=pix,
            prefer_parquet=args.prefer_parquet,
        )
        paths[pix] = path
        if path.exists():
            existing.append(pix)
        else:
            missing.append(pix)
    return existing, missing, paths


def _prepare_tile_order(missing: list[int], args: argparse.Namespace) -> None:
    if not args.shuffle_tiles or len(missing) <= 1:
        return
    if args.shuffle_seed is None:
        args.shuffle_seed = random.SystemRandom().randrange(0, 2**63)
    random.Random(int(args.shuffle_seed)).shuffle(missing)


def _validate_chunk_nside(args: argparse.Namespace) -> None:
    chunk_nside = int(args.chunk_nside)
    output_nside = int(args.nside)
    if chunk_nside <= 0 or chunk_nside & (chunk_nside - 1):
        raise ValueError("--chunk-nside must be a positive power of two")
    if chunk_nside > output_nside:
        raise ValueError("--chunk-nside must be <= --nside")
    if output_nside % chunk_nside:
        raise ValueError("--nside must be an integer multiple of --chunk-nside")


def _output_tile_parent_chunk(
    *,
    output_pix: int,
    output_nside: int,
    output_order: str,
    chunk_nside: int,
) -> int:
    nested_pix = (
        int(output_pix)
        if output_order == "nested"
        else int(ring_to_nested(int(output_pix), output_nside))
    )
    factor = output_nside // chunk_nside
    return int(nested_pix // (factor * factor))


def _chunk_missing_tiles(
    *,
    missing_tiles: Sequence[int],
    output_nside: int,
    output_order: str,
    chunk_nside: int,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for output_pix in missing_tiles:
        chunk_pix = _output_tile_parent_chunk(
            output_pix=int(output_pix),
            output_nside=output_nside,
            output_order=output_order,
            chunk_nside=chunk_nside,
        )
        out.setdefault(chunk_pix, []).append(int(output_pix))
    return out


def _prepare_chunk_order(chunks: list[int], args: argparse.Namespace) -> None:
    if not args.shuffle_tiles or len(chunks) <= 1:
        return
    if args.shuffle_seed is None:
        args.shuffle_seed = random.SystemRandom().randrange(0, 2**63)
    random.Random(int(args.shuffle_seed)).shuffle(chunks)


def _empty_table(columns: Sequence[str]) -> Table:
    dtype = []
    for column in columns:
        dtype.append("int64" if column == "source_id" else "float64")
    return Table(names=list(columns), dtype=dtype)


def _row_output_pixels(tab: Table, *, nside: int, order: str) -> np.ndarray:
    if len(tab) == 0:
        return np.asarray([], dtype=np.int64)
    hp = HEALPix(nside=int(nside), order=order, frame=ICRS())
    coords = SkyCoord(
        ra=np.asarray(tab["ra"], dtype=np.float64) * u.deg,
        dec=np.asarray(tab["dec"], dtype=np.float64) * u.deg,
        frame="icrs",
    )
    return np.asarray(hp.skycoord_to_healpix(coords), dtype=np.int64)


def _write_chunk_to_tiles(
    *,
    tab: Table,
    output_tiles: Sequence[int],
    paths: dict[int, Path],
    args: argparse.Namespace,
    progress: Progress,
) -> None:
    row_pix = _row_output_pixels(tab, nside=args.nside, order=args.order)
    empty = _empty_table(args.columns)
    for output_pix in output_tiles:
        path = paths[int(output_pix)]
        if path.exists():
            progress.skipped_existing += 1
            continue
        if len(tab) == 0:
            tile_tab = empty
        else:
            tile_tab = tab[row_pix == int(output_pix)]
        _save_table_atomic(tile_tab, path)
        stat = path.stat()
        progress.completed_this_run += 1
        progress.bytes_written_this_run += int(stat.st_size)
        progress.last_pix = int(output_pix)
        progress.last_rows = len(tile_tab)
        progress.last_path = str(path)
        progress.last_error = None


def download_full_cache_source_id_chunks(args: argparse.Namespace) -> int:
    args.cache_dir = Path(args.cache_dir).expanduser().resolve()
    args.progress_json = (
        Path(args.progress_json).expanduser().resolve()
        if args.progress_json
        else args.cache_dir / "gaia_full_download_progress.json"
    )
    set_cache_dir(args.cache_dir)
    _validate_chunk_nside(args)

    total_tiles = int(12 * int(args.nside) * int(args.nside))
    total_chunks = int(12 * int(args.chunk_nside) * int(args.chunk_nside))
    existing, missing, paths = _existing_tiles(args, total_tiles)
    chunk_to_tiles = _chunk_missing_tiles(
        missing_tiles=missing,
        output_nside=int(args.nside),
        output_order=args.order,
        chunk_nside=int(args.chunk_nside),
    )
    chunks = sorted(chunk_to_tiles)
    _prepare_chunk_order(chunks, args)
    progress = Progress(
        total_tiles=total_tiles,
        missing_at_start=len(missing),
        cached_at_start=len(existing),
        skipped_existing=len(existing),
        total_chunks=total_chunks,
        missing_chunks_at_start=len(chunks),
    )
    started_at = _utc_now()
    args._started_monotonic = time.time()

    print(
        "[gaia-full] "
        f"strategy=source-id-chunks nside={args.nside} chunk_nside={args.chunk_nside} "
        f"gmax={args.gmax} tiles={total_tiles} cached={len(existing)} "
        f"missing_tiles={len(missing)} chunks={len(chunks)}/{total_chunks} cache={args.cache_dir}",
        flush=True,
    )
    if args.shuffle_tiles:
        print(f"[gaia-full] chunk order=random seed={args.shuffle_seed}", flush=True)
    else:
        print("[gaia-full] chunk order=sequential", flush=True)
    _write_json(
        args.progress_json,
        _progress_payload(status="starting", started_at=started_at, args=args, progress=progress),
    )

    if args.dry_run or not chunks:
        status = "dry_run" if args.dry_run else "complete"
        _write_json(
            args.progress_json,
            _progress_payload(status=status, started_at=started_at, args=args, progress=progress),
        )
        print(f"[gaia-full] {status}: nothing to download", flush=True)
        return 0

    auth = load_gaia_auth(args.auth_file)
    did_login = False
    if auth:
        print("[gaia-full] Gaia TAP login", flush=True)
        _tap_login_only(auth[0], auth[1])
        did_login = True
    else:
        print("[gaia-full] Gaia TAP anonymous mode", flush=True)

    try:
        for index, chunk_pix in enumerate(chunks, 1):
            output_tiles = [pix for pix in chunk_to_tiles[chunk_pix] if not paths[pix].exists()]
            if not output_tiles:
                continue
            progress.current_chunk_pix = int(chunk_pix)
            progress.current_pix = int(output_tiles[0])
            _write_json(
                args.progress_json,
                _progress_payload(status="running", started_at=started_at, args=args, progress=progress),
            )
            source_id_min, source_id_max = _source_id_bounds_for_nested_chunk(
                int(chunk_pix),
                int(args.chunk_nside),
            )
            print(
                "[gaia-full] "
                f"chunk {index}/{len(chunks)} pix={chunk_pix} "
                f"tiles={len(output_tiles)} source_id=[{source_id_min},{source_id_max})",
                flush=True,
            )
            try:
                tab = _download_source_id_chunk_with_optional_relogin(
                    args=args,
                    source_id_min=source_id_min,
                    source_id_max=source_id_max,
                    auth=auth,
                    progress=progress,
                )
                progress.rows_downloaded_this_run += len(tab)
                progress.completed_chunks_this_run += 1
                progress.last_chunk_pix = int(chunk_pix)
                progress.last_chunk_rows = len(tab)
                _write_chunk_to_tiles(
                    tab=tab,
                    output_tiles=output_tiles,
                    paths=paths,
                    args=args,
                    progress=progress,
                )
                print(
                    "[gaia-full] "
                    f"saved chunk={chunk_pix} rows={len(tab)} "
                    f"tiles={len(output_tiles)} completed_tiles={progress.completed_this_run}",
                    flush=True,
                )
            except Exception as exc:
                progress.failed_chunks_this_run += 1
                progress.failed_this_run += len(output_tiles)
                progress.last_chunk_pix = int(chunk_pix)
                progress.last_chunk_rows = None
                progress.last_pix = int(output_tiles[0])
                progress.last_rows = None
                progress.last_path = None
                progress.last_error = f"{type(exc).__name__}: {exc}"
                _write_json(
                    args.progress_json,
                    _progress_payload(status="error", started_at=started_at, args=args, progress=progress),
                )
                print(f"[gaia-full] ERROR chunk={chunk_pix}: {progress.last_error}", flush=True)
                if not args.continue_on_error:
                    raise

            _write_json(
                args.progress_json,
                _progress_payload(status="running", started_at=started_at, args=args, progress=progress),
            )
            if args.sleep_s > 0 and index < len(chunks):
                time.sleep(args.sleep_s)
    finally:
        progress.current_chunk_pix = None
        progress.current_pix = None
        if did_login:
            print("[gaia-full] Gaia TAP logout", flush=True)
            _tap_logout_only()

    status = (
        "complete_with_errors"
        if progress.failed_this_run or progress.failed_chunks_this_run
        else "complete"
    )
    _write_json(
        args.progress_json,
        _progress_payload(status=status, started_at=started_at, args=args, progress=progress),
    )
    print(
        "[gaia-full] "
        f"{status}: chunks={progress.completed_chunks_this_run}/{len(chunks)} "
        f"tiles={progress.completed_this_run} failed_chunks={progress.failed_chunks_this_run} "
        f"rows={progress.rows_downloaded_this_run}",
        flush=True,
    )
    return 1 if progress.failed_this_run and not args.continue_on_error else 0


def download_full_cache(args: argparse.Namespace) -> int:
    args.cache_dir = Path(args.cache_dir).expanduser().resolve()
    args.progress_json = (
        Path(args.progress_json).expanduser().resolve()
        if args.progress_json
        else args.cache_dir / "gaia_full_download_progress.json"
    )
    set_cache_dir(args.cache_dir)

    hp = HEALPix(nside=int(args.nside), order=args.order, frame=ICRS())
    total_tiles = int(hp.npix)
    existing, missing, paths = _existing_tiles(args, total_tiles)
    _prepare_tile_order(missing, args)
    progress = Progress(
        total_tiles=total_tiles,
        missing_at_start=len(missing),
        cached_at_start=len(existing),
        skipped_existing=len(existing),
    )
    started_at = _utc_now()
    args._started_monotonic = time.time()

    print(
        "[gaia-full] "
        f"nside={args.nside} gmax={args.gmax} tiles={total_tiles} "
        f"cached={len(existing)} missing={len(missing)} cache={args.cache_dir}",
        flush=True,
    )
    if args.shuffle_tiles:
        print(f"[gaia-full] tile order=random seed={args.shuffle_seed}", flush=True)
    else:
        print("[gaia-full] tile order=sequential", flush=True)
    _write_json(
        args.progress_json,
        _progress_payload(status="starting", started_at=started_at, args=args, progress=progress),
    )

    if args.dry_run or not missing:
        status = "dry_run" if args.dry_run else "complete"
        _write_json(
            args.progress_json,
            _progress_payload(status=status, started_at=started_at, args=args, progress=progress),
        )
        print(f"[gaia-full] {status}: nothing to download", flush=True)
        return 0

    auth = load_gaia_auth(args.auth_file)
    did_login = False
    if auth:
        print("[gaia-full] Gaia TAP login", flush=True)
        _tap_login_only(auth[0], auth[1])
        did_login = True
    else:
        print("[gaia-full] Gaia TAP anonymous mode", flush=True)

    try:
        for index, pix in enumerate(missing, 1):
            path = paths[pix]
            if path.exists():
                progress.skipped_existing += 1
                continue

            progress.current_pix = int(pix)
            _write_json(
                args.progress_json,
                _progress_payload(status="running", started_at=started_at, args=args, progress=progress),
            )
            print(
                f"[gaia-full] {index}/{len(missing)} pix={pix} -> {path.name}",
                flush=True,
            )
            try:
                poly = hp.boundaries_skycoord(int(pix), step=1)
                tab = _download_tile_with_optional_relogin(
                    args=args,
                    poly_sky=poly,
                    auth=auth,
                    progress=progress,
                )
                _save_table_atomic(tab, path)
                stat = path.stat()
                progress.completed_this_run += 1
                progress.rows_downloaded_this_run += len(tab)
                progress.bytes_written_this_run += int(stat.st_size)
                progress.last_pix = int(pix)
                progress.last_rows = len(tab)
                progress.last_path = str(path)
                progress.last_error = None
                print(
                    f"[gaia-full] saved pix={pix} rows={len(tab)} bytes={stat.st_size}",
                    flush=True,
                )
            except Exception as exc:
                progress.failed_this_run += 1
                progress.last_pix = int(pix)
                progress.last_rows = None
                progress.last_path = str(path)
                progress.last_error = f"{type(exc).__name__}: {exc}"
                _write_json(
                    args.progress_json,
                    _progress_payload(status="error", started_at=started_at, args=args, progress=progress),
                )
                print(f"[gaia-full] ERROR pix={pix}: {progress.last_error}", flush=True)
                if not args.continue_on_error:
                    raise

            _write_json(
                args.progress_json,
                _progress_payload(status="running", started_at=started_at, args=args, progress=progress),
            )
            if args.sleep_s > 0 and index < len(missing):
                time.sleep(args.sleep_s)
    finally:
        progress.current_pix = None
        if did_login:
            print("[gaia-full] Gaia TAP logout", flush=True)
            _tap_logout_only()

    status = "complete_with_errors" if progress.failed_this_run else "complete"
    _write_json(
        args.progress_json,
        _progress_payload(status=status, started_at=started_at, args=args, progress=progress),
    )
    print(
        "[gaia-full] "
        f"{status}: downloaded={progress.completed_this_run} failed={progress.failed_this_run} "
        f"rows={progress.rows_downloaded_this_run}",
        flush=True,
    )
    return 1 if progress.failed_this_run and not args.continue_on_error else 0


def _columns(value: str) -> tuple[str, ...]:
    out = tuple(part.strip() for part in value.split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("columns cannot be empty")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download all Gaia DR3 HEALPix tiles for the local cache. "
            "Defaults match the simulator cache: Gaia DR3, G<=15, NSIDE 16."
        )
    )
    parser.add_argument("--cache-dir", default=os.environ.get("GAIA_CONE_CACHE_DIR", "~/.cache/gaia_cones"))
    parser.add_argument("--progress-json", default=None)
    parser.add_argument("--auth-file", default=None)
    parser.add_argument("--table-name", default=DEFAULT_TABLE)
    parser.add_argument("--columns", type=_columns, default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--gmax", type=float, default=15.0)
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument(
        "--strategy",
        choices=("tile-polygons", "source-id-chunks"),
        default="tile-polygons",
    )
    parser.add_argument("--chunk-nside", type=int, default=4)
    parser.add_argument("--order", choices=("ring", "nested"), default="ring")
    parser.add_argument("--row-limit", type=int, default=-1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-s", type=float, default=10.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--start-pix", type=int, default=0)
    parser.add_argument("--stop-pix", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--no-shuffle", dest="shuffle_tiles", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ecsv", dest="prefer_parquet", action="store_false")
    parser.set_defaults(prefer_parquet=True, shuffle_tiles=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nside <= 0 or args.nside & (args.nside - 1):
        raise SystemExit("--nside must be a positive power of two")
    if not math.isfinite(args.gmax):
        raise SystemExit("--gmax must be finite")
    if args.strategy == "source-id-chunks":
        return download_full_cache_source_id_chunks(args)
    return download_full_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
