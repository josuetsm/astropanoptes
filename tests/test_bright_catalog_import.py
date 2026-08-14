import gzip
from pathlib import Path

import numpy as np
from astropy.table import vstack
import pyarrow.parquet as pq

from gaia_cache import _bright_tile_cache_key, _path_for_in
from scripts.import_bright_catalogs import (
    parse_hipparcos,
    parse_tycho2,
    write_tiles,
)


def _fixed_width_line(length: int, fields: list[tuple[int, int, str]]) -> str:
    chars = [" "] * length
    for start, stop, value in fields:
        text = str(value).rjust(stop - start)
        assert len(text) == stop - start
        chars[start:stop] = text
    return "".join(chars) + "\n"


def test_cds_parsers_and_local_tiling(tmp_path: Path) -> None:
    hip_path = tmp_path / "hip_main.dat"
    hip_path.write_text(
        _fixed_width_line(
            449,
            [
                (8, 14, "30438"),
                (41, 46, "-0.62"),
                (51, 63, "95.98787790"),
                (64, 76, "-52.6957179"),
                (87, 95, "19.93"),
                (96, 104, "23.24"),
            ],
        ),
        encoding="ascii",
    )

    tycho_path = tmp_path / "tyc2.dat.00.gz"
    with gzip.open(tycho_path, "wt", encoding="ascii") as handle:
        handle.write(
            _fixed_width_line(
                206,
                [
                    (0, 4, "1"),
                    (5, 10, "13"),
                    (11, 12, "1"),
                    (15, 27, "1.12558209"),
                    (28, 40, "2.26739400"),
                    (41, 48, "10.0"),
                    (49, 56, "-5.0"),
                    (110, 116, "10.488"),
                    (123, 129, "8.670"),
                ],
            )
        )

    hip, hip_rows = parse_hipparcos(hip_path, vmax=15.0)
    tycho, tycho_rows = parse_tycho2([tycho_path], vmax=15.0)

    assert hip_rows == 1
    assert tycho_rows == 1
    assert len(hip) == 1
    assert len(tycho) == 1
    assert int(tycho["source_id"][0]) == -(2_000_000_000_000 + 1_000_131)

    catalog = vstack([hip, tycho], join_type="exact")
    stats = write_tiles(
        catalog,
        cache_dir=tmp_path,
        nside=1,
        order="ring",
        vmax=15.0,
        prefer_parquet=True,
    )

    rows = 0
    for pix in range(12):
        key = _bright_tile_cache_key(nside=1, order="ring", pix=pix, vmax=15.0)
        path = _path_for_in(tmp_path, key, True)
        assert path.exists()
        rows += pq.ParquetFile(path).metadata.num_rows

    assert stats["tiles"] == 12
    assert stats["rows"] == 2
    assert rows == 2
    assert np.isfinite(np.asarray(catalog["ra"], dtype=np.float64)).all()
