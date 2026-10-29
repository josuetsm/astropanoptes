"""El contrato de lo que escribe la consola.

La consola la leen dos: una persona en el telescopio y un agente que la maneja
por script. Estos tests fijan la gramatica que comparten -- una linea `ok` o
`error` por comando, eventos en columnas fijas, todo en minusculas -- para que no
se vuelva a desordenar con el tiempo.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import pytest

from console_format import (
    event_line,
    explain,
    reason,
    render_tree,
    render_value,
    result_lines,
    split_subsystem,
    state_word,
)

WHEN = dt.datetime(2026, 10, 18, 21, 54, 6, 931000)
EVENT = re.compile(r"^\d\d:\d\d:\d\d\.\d{3}  (debug|info |aviso|error)  (\S+)\s+  ?\S")


# ------------------------------------------------------------------- eventos


def test_an_event_has_clock_time_level_and_subsystem_in_fixed_columns() -> None:
    line = event_line("info", "GoTo: diagnóstico guardado", when=WHEN)

    assert line.startswith("21:54:06.931  info   goto      diagnóstico guardado")
    assert EVENT.match(line), line


def test_the_subsystem_column_names_the_part_of_the_app_not_the_thread() -> None:
    """Antes la columna era el hilo: AppRunner, GoToWorker, MainThread."""
    assert split_subsystem("Camera: conectada") == ("camara", "conectada")
    assert split_subsystem("Platesolving: resuelto") == ("solve", "resuelto")
    assert split_subsystem("Mount: conectada") == ("montura", "conectada")
    assert split_subsystem("Gaia healpix: consultando") == ("gaia", "consultando")


def test_a_colon_that_is_not_a_subsystem_is_left_alone() -> None:
    assert split_subsystem("error: algo raro") == ("app", "error: algo raro")
    assert split_subsystem("ratio: 3") == ("app", "ratio: 3")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("estado=ERR_NOT_SYNCED", "estado=err_not_synced"),
        ("ok=True", "ok=true"),
        ("ok=False", "ok=false"),
        ("best=None", "best=null"),
        ("estado=OK", "estado=ok"),
    ],
)
def test_stored_codes_and_python_literals_are_shown_in_lowercase(raw, expected) -> None:
    """Los datos se guardan como estan; solo se presentan en minusculas."""
    assert event_line("info", f"GoTo: x {raw}", when=WHEN).endswith(f"x {expected}")


def test_an_environment_variable_named_in_a_message_keeps_its_case() -> None:
    line = event_line("info", "App: activa con ASTROPANOPTES_DEBUG=1", when=WHEN)
    assert "ASTROPANOPTES_DEBUG=1" in line


def test_levels_are_lowercase_words() -> None:
    assert " aviso " in event_line("warning", "App: x", when=WHEN)
    assert " error " in event_line("error", "App: x", when=WHEN)
    assert " debug " in event_line("debug", "App: x", when=WHEN)


# ---------------------------------------------------------------- resultados


def test_every_response_starts_with_ok_or_error() -> None:
    assert result_lines(True) == ["ok"]
    assert result_lines(True, "en cola: conectar cámara 0") == ["ok  en cola: conectar cámara 0"]
    assert result_lines(False, "valor inválido")[0] == "error  valor inválido"


def test_detail_is_indented_so_it_never_looks_like_a_result_line() -> None:
    lines = result_lines(True, "solve terminó", {"inliers": 9, "rms_px": 0.896})

    assert lines[0] == "ok  solve terminó"
    assert all(line.startswith("  ") for line in lines[1:])


def test_nested_detail_aligns_each_level_without_json_punctuation() -> None:
    lines = render_tree({"coverage": {"az": 0.0, "alt": 0.25}, "fitted": False, "samples": 0})

    text = "\n".join(lines)
    assert "{" not in text and '"' not in text
    assert lines == [
        "  coverage",
        "    az   0",
        "    alt  0.25",
        "  fitted   false",
        "  samples  0",
    ]


@pytest.mark.parametrize(
    "value, shown",
    [
        (None, "null"),
        (True, "true"),
        (3.098208999290364, "3.09821"),
        (1789522125.968, "1789522125.968"),
        (float("nan"), "nan"),
        ([1, 2.5, None], "[1, 2.5, null]"),
    ],
)
def test_values_are_data_without_noise(value, shown) -> None:
    assert render_value(value) == shown


# ----------------------------------------------------------------- palabras


def test_states_are_words_that_agree_with_their_row() -> None:
    assert state_word("camara", "OK") == "conectada"
    assert state_word("montura", "DISCONNECTED") == "desconectada"
    assert state_word("tracking", "OFF") == "apagado"
    assert state_word("solve", "FAIL") == "falló"
    assert state_word("goto", "CANCELLED") == "cancelado"


def test_known_failure_codes_explain_what_to_do() -> None:
    assert "mount sync" in explain("ERR_NOT_SYNCED")
    assert "ciclo" not in explain("ERR_NOT_SYNCED")
    assert explain("UN_CODIGO_QUE_NO_EXISTE") == ""


def test_a_failure_message_keeps_its_paths_and_names() -> None:
    """La fila de la montura mostraba /dev/cu.astropanoptes-esp32, un puerto que no existe."""
    message = "sin respuesta a PING en /dev/cu.AstroPanoptes-ESP32"
    assert reason(message) == message
    assert reason("ERR_NOT_SYNCED") == "err_not_synced"
    assert reason("TIMEOUT") == "timeout"


# ------------------------------------------------------------ consola entera


class _Runner:
    """Lo minimo de AppRunner que la consola toca para estos casos."""

    def __init__(self) -> None:
        from config import AppConfig

        self.cfg = AppConfig()
        self.calls = []

    def request_camera_connect(self, index):
        self.calls.append(("connect", index))

    def get_state(self):
        from ap_types import AppState

        return AppState()

    def get_latest_preview_jpeg(self):
        return None


def _terminal(echo: bool):
    from terminal_app import TerminalApp

    out = io.StringIO()
    return TerminalApp(_Runner(), output=out, echo_commands=echo), out


def test_scripts_put_each_command_above_its_answer() -> None:
    """Sin el eco, `-c a -c b -c c` daba tres `OK` seguidos sin saber cual era cual."""
    terminal, out = _terminal(echo=True)

    assert terminal.run_script(["camera connect", "camera connect 2"])

    assert out.getvalue().splitlines() == [
        "› camera connect",
        "ok  en cola: conectar cámara 0",
        "› camera connect 2",
        "ok  en cola: conectar cámara 2",
    ]


def test_a_stopped_script_says_where_and_how_much_was_left() -> None:
    terminal, out = _terminal(echo=True)

    assert not terminal.run_script(["camera connect", "camera connect abc", "camera connect 1"])

    last = out.getvalue().splitlines()[-1]
    assert last == (
        "error  script detenido en el comando 2 de 3: camera connect abc; quedaron 1 sin ejecutar"
    )


def test_a_bad_argument_names_itself_instead_of_leaking_python() -> None:
    """Antes: "invalid literal for int() with base 10: 'abc'"."""
    terminal, out = _terminal(echo=False)

    terminal.execute_line("camera connect abc")

    assert out.getvalue().strip() == "error  se esperaba un número entero para el índice; recibí 'abc'"
    assert "invalid literal" not in out.getvalue()


def test_an_unknown_state_path_lists_what_does_exist() -> None:
    """`get stacking.frames` obligaba a pedir el estado entero para descubrir frames_used."""
    terminal, out = _terminal(echo=False)

    terminal.execute_line("get stacking.frames")

    text = out.getvalue()
    assert text.startswith("error  no existe stacking.frames; en stacking hay:")
    assert "frames_used" in text


def test_json_mode_answers_with_the_document_alone() -> None:
    import json

    terminal, out = _terminal(echo=True)

    terminal.execute_line("get camera --json")

    payload = json.loads(out.getvalue())
    assert payload["path"] == "camera"
    assert "connected" in payload["value"]


def test_comments_and_blank_lines_are_not_commands() -> None:
    terminal, out = _terminal(echo=True)

    assert terminal.run_script(["", "# conectar primero", "camera connect"])
    assert out.getvalue().splitlines()[0] == "› camera connect"
