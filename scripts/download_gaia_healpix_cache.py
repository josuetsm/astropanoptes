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
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from astropy.coordinates import ICRS
from astropy_healpix import HEALPix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaia_cache import (  # noqa: E402
    DEFAULT_COLUMNS,
    DEFAULT_TABLE,
    _cache_key,
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
                tab = _query_healpix_tile_async(
                    table_name=args.table_name,
                    columns=args.columns,
                    gmax=args.gmax,
                    poly_sky=poly,
                    row_limit=args.row_limit,
                    retries=args.retries,
                    backoff_s=args.backoff_s,
                    verbose=False,
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
    parser.add_argument("--order", choices=("ring", "nested"), default="ring")
    parser.add_argument("--row-limit", type=int, default=-1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-s", type=float, default=10.0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--start-pix", type=int, default=0)
    parser.add_argument("--stop-pix", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ecsv", dest="prefer_parquet", action="store_false")
    parser.set_defaults(prefer_parquet=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nside <= 0 or args.nside & (args.nside - 1):
        raise SystemExit("--nside must be a positive power of two")
    if not math.isfinite(args.gmax):
        raise SystemExit("--gmax must be finite")
    return download_full_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
