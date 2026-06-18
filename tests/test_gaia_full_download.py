import json
import importlib.util
from pathlib import Path
import random
import sys

from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = REPO_ROOT / "scripts" / "download_gaia_healpix_cache.py"
_SPEC = importlib.util.spec_from_file_location("download_gaia_healpix_cache", DOWNLOADER_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
downloader = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = downloader
_SPEC.loader.exec_module(downloader)


def test_prepare_tile_order_shuffles_with_fixed_seed() -> None:
    args = downloader.build_parser().parse_args(["--shuffle-seed", "123"])
    missing = list(range(12))
    expected = list(range(12))
    random.Random(123).shuffle(expected)

    downloader._prepare_tile_order(missing, args)

    assert missing == expected
    assert missing != list(range(12))


def test_downloader_relogs_once_on_auth_session_error(monkeypatch, tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    args = downloader.build_parser().parse_args(
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "--progress-json",
            str(progress_path),
            "--nside",
            "1",
            "--limit",
            "1",
        ]
    )

    calls: list[str] = []
    monkeypatch.setattr(downloader, "load_gaia_auth", lambda _auth_file=None: ("user", "pass"))
    monkeypatch.setattr(downloader, "_tap_login_only", lambda _user, _password: calls.append("login"))
    monkeypatch.setattr(downloader, "_tap_logout_only", lambda: calls.append("logout"))

    query_calls = 0

    def fake_query(**_kwargs):
        nonlocal query_calls
        query_calls += 1
        if query_calls == 1:
            raise RuntimeError("401 Unauthorized: session expired")
        return Table(
            {
                "source_id": [1],
                "ra": [10.0],
                "dec": [-20.0],
                "phot_g_mean_mag": [12.3],
            }
        )

    def fake_save(tab: Table, path: Path) -> None:
        assert len(tab) == 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr(downloader, "_query_healpix_tile_async", fake_query)
    monkeypatch.setattr(downloader, "_save_table_atomic", fake_save)

    assert downloader.download_full_cache(args) == 0

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert calls == ["login", "logout", "login", "logout"]
    assert query_calls == 2
    assert progress["completed_this_run"] == 1
    assert progress["failed_this_run"] == 0
    assert progress["relogin_count"] == 1
    assert "401 Unauthorized" in progress["last_relogin_error"]


def test_source_id_chunk_strategy_writes_missing_tiles(monkeypatch, tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    args = downloader.build_parser().parse_args(
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "--progress-json",
            str(progress_path),
            "--strategy",
            "source-id-chunks",
            "--nside",
            "1",
            "--chunk-nside",
            "1",
            "--limit",
            "1",
            "--shuffle-seed",
            "1",
        ]
    )

    queries: list[tuple[int, int]] = []

    def fake_query(*, args, source_id_min: int, source_id_max: int):
        queries.append((source_id_min, source_id_max))
        return Table(
            names=list(args.columns),
            dtype=("int64", "float64", "float64", "float64"),
        )

    def fake_save(tab: Table, path: Path) -> None:
        assert len(tab) == 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr(downloader, "load_gaia_auth", lambda _auth_file=None: None)
    monkeypatch.setattr(downloader, "_query_source_id_chunk_async", fake_query)
    monkeypatch.setattr(downloader, "_save_table_atomic", fake_save)

    assert downloader.download_full_cache_source_id_chunks(args) == 0

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["strategy"] == "source-id-chunks"
    assert progress["chunk_nside"] == 1
    assert progress["completed_chunks_this_run"] == 1
    assert progress["completed_this_run"] == 1
    assert progress["rows_downloaded_this_run"] == 0
    assert queries == [(0, 1 << 59)]
