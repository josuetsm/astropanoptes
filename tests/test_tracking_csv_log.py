"""El tracking no dejaba rastro en disco: un drift o un giro de anclaje solo se
podia reconstruir mirando lo ultimo que quedo en pantalla. ``tracking_samples.csv``
guarda cada muestra (error medido, tasas comandadas, fuente de calibracion) para
poder debuggear o analizar una sesion despues.
"""
from __future__ import annotations

import csv
import os

import pytest

import app_runner
from app_runner import AppRunner
from config import TrackingConfig
from tracking import TrackingOutput


def _sample_output(**overrides) -> TrackingOutput:
    kwargs = dict(
        ok=True,
        mode="TRACK",
        resp=0.9,
        dx=1.5,
        dy=-0.5,
        vx=0.1,
        vy=0.0,
        abs_resp=0.8,
        x_hat=2.0,
        y_hat=-1.0,
        rate_az=120.0,
        rate_alt=-30.0,
        calib_src="auto",
        detA=0.5,
        n_det=42,
    )
    kwargs.update(overrides)
    return TrackingOutput(**kwargs)


def _runner(log_enabled: bool = True, log_hz: float = 1.0) -> AppRunner:
    r = AppRunner.__new__(AppRunner)
    r.cfg = type("Cfg", (), {"tracking": TrackingConfig(log_enabled=log_enabled, log_hz=log_hz)})()
    r._tracking_log_t_last = 0.0
    r._tracking_log_pending_steps_az = 0
    r._tracking_log_pending_steps_alt = 0
    r._current_pointing_az_alt = lambda: (123.4, 56.7)
    r._goto = type(
        "Goto",
        (),
        {"model": type("Model", (), {"steps_est": [1000.0, -250.0]})()},
    )()
    return r


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROPANOPTES_TRACKING_LOG_DIR", str(tmp_path))
    yield tmp_path


def _log_path(tmp_path) -> str:
    return os.path.join(str(tmp_path), "tracking_samples.csv")


def _read_rows(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestTrackingCsvLog:
    def test_writes_one_row_with_the_expected_fields(self, _isolated_log_dir) -> None:
        r = _runner()
        r._maybe_log_tracking_sample(
            _sample_output(),
            frame_t=10.0,
            ff_ready=True,
            rate_cmd_az=90.0,
            rate_cmd_alt=-25.0,
            rate_fb_az=120.0,
            rate_fb_alt=-30.0,
            rate_ff_az=-30.0,
            rate_ff_alt=5.0,
            move_steps_az=3,
            move_steps_alt=-1,
        )
        path = _log_path(_isolated_log_dir)
        assert os.path.exists(path)
        rows = _read_rows(path)
        assert len(rows) == 1
        row = rows[0]
        assert row["mode"] == "TRACK"
        assert row["calib_src"] == "auto"
        assert float(row["rate_cmd_az"]) == 90.0
        assert float(row["error_px"]) == pytest.approx((2.0**2 + 1.0**2) ** 0.5)
        assert float(row["pointing_az_deg"]) == 123.4
        # move_steps_* son los pasos que de verdad salieron hacia la montura;
        # steps_est_* es la posicion acumulada del modelo tras aplicarlos.
        assert int(row["move_steps_az"]) == 3
        assert int(row["move_steps_alt"]) == -1
        assert float(row["steps_est_az"]) == 1000.0
        assert float(row["steps_est_alt"]) == -250.0

    def test_disabled_writes_nothing(self, _isolated_log_dir) -> None:
        r = _runner(log_enabled=False)
        r._maybe_log_tracking_sample(
            _sample_output(),
            frame_t=10.0,
            ff_ready=True,
            rate_cmd_az=0.0,
            rate_cmd_alt=0.0,
            rate_fb_az=0.0,
            rate_fb_alt=0.0,
            rate_ff_az=0.0,
            rate_ff_alt=0.0,
        )
        assert not os.path.exists(_log_path(_isolated_log_dir))

    def test_throttled_below_log_hz(self, _isolated_log_dir, monkeypatch) -> None:
        """Dos muestras separadas por menos de 1/log_hz solo dejan una fila."""
        r = _runner(log_hz=1.0)
        t = [1000.0]
        monkeypatch.setattr(app_runner, "_now_s", lambda: t[0])

        kwargs = dict(
            frame_t=10.0,
            ff_ready=True,
            rate_cmd_az=0.0,
            rate_cmd_alt=0.0,
            rate_fb_az=0.0,
            rate_fb_alt=0.0,
            rate_ff_az=0.0,
            rate_ff_alt=0.0,
        )
        r._maybe_log_tracking_sample(_sample_output(), **kwargs)
        t[0] += 0.1
        r._maybe_log_tracking_sample(_sample_output(), **kwargs)
        assert len(_read_rows(_log_path(_isolated_log_dir))) == 1

        t[0] += 1.0
        r._maybe_log_tracking_sample(_sample_output(), **kwargs)
        assert len(_read_rows(_log_path(_isolated_log_dir))) == 2

    def test_move_steps_accumulate_across_throttled_cycles(
        self, _isolated_log_dir, monkeypatch
    ) -> None:
        """Un MOVE emitido entre dos filas muestreadas no puede desaparecer:
        el lazo de control corre mas rapido que log_hz, asi que dos ciclos
        con movimiento real pueden caer entre la misma fila si no se suman.
        """
        r = _runner(log_hz=1.0)
        t = [2000.0]
        monkeypatch.setattr(app_runner, "_now_s", lambda: t[0])

        kwargs = dict(
            frame_t=10.0,
            ff_ready=True,
            rate_cmd_az=0.0,
            rate_cmd_alt=0.0,
            rate_fb_az=0.0,
            rate_fb_alt=0.0,
            rate_ff_az=0.0,
            rate_ff_alt=0.0,
        )
        # Primera llamada: sin fila previa, escribe de inmediato.
        r._maybe_log_tracking_sample(_sample_output(), move_steps_az=4, move_steps_alt=1, **kwargs)
        rows = _read_rows(_log_path(_isolated_log_dir))
        assert len(rows) == 1
        assert int(rows[0]["move_steps_az"]) == 4

        # Dos ciclos con movimiento real, ambos dentro de la ventana de
        # cadencia: ninguno escribe fila propia, pero sus pasos no se pierden.
        t[0] += 0.1
        r._maybe_log_tracking_sample(_sample_output(), move_steps_az=6, move_steps_alt=-2, **kwargs)
        t[0] += 0.1
        r._maybe_log_tracking_sample(_sample_output(), move_steps_az=3, move_steps_alt=0, **kwargs)
        assert len(_read_rows(_log_path(_isolated_log_dir))) == 1

        # La siguiente fila que si se escribe carga la suma acumulada.
        t[0] += 1.0
        r._maybe_log_tracking_sample(_sample_output(), move_steps_az=0, move_steps_alt=0, **kwargs)
        rows = _read_rows(_log_path(_isolated_log_dir))
        assert len(rows) == 2
        assert int(rows[1]["move_steps_az"]) == 9
        assert int(rows[1]["move_steps_alt"]) == -2

    def test_schema_change_archives_the_old_file_instead_of_mixing_columns(
        self, _isolated_log_dir
    ) -> None:
        path = _log_path(_isolated_log_dir)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["some", "old", "header"])
            writer.writerow(["1", "2", "3"])

        r = _runner()
        r._maybe_log_tracking_sample(
            _sample_output(),
            frame_t=10.0,
            ff_ready=True,
            rate_cmd_az=0.0,
            rate_cmd_alt=0.0,
            rate_fb_az=0.0,
            rate_fb_alt=0.0,
            rate_ff_az=0.0,
            rate_ff_alt=0.0,
        )

        archived = [
            name
            for name in os.listdir(str(_isolated_log_dir))
            if name.startswith("tracking_samples.csv.") and name.endswith(".old")
        ]
        assert len(archived) == 1
        rows = _read_rows(path)
        assert len(rows) == 1
        assert rows[0]["mode"] == "TRACK"
