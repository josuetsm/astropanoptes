from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ap_types import Frame
from config import FocuserConfig
from focuser import (
    FocuserWorker,
    cfa_plane,
    focus_metric,
    parabolic_peak,
)
from mount_arduino import ArduinoConfig, ArduinoController


FIRMWARE = (
    Path(__file__).resolve().parents[1] / "mount_firmware" / "mount_firmware.ino"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------- metrica ---

def _sky(h: int = 480, w: int = 640, level: float = 700.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Cielo con gradiente, como el de Santiago: es la condicion que rompe una
    # estimacion global del fondo.
    yy = np.linspace(-1.0, 1.0, h)[:, None]
    xx = np.linspace(-1.0, 1.0, w)[None, :]
    sky = level * (1.0 + 0.12 * (0.7 * yy + 0.3 * xx))
    return rng.poisson(sky).astype(np.float64)


def _field(sigma_px: float, *, gain: float = 1.0, seed: int = 3) -> np.ndarray:
    """Campo estelar con PSF de ancho ``sigma_px``, a flujo total constante."""
    img = _sky(seed=seed)
    h, w = img.shape
    stars = ((160, 120, 9.0e5), (400, 300, 4.0e5), (520, 210, 2.0e5))
    for x, y, flux in stars:
        radius = int(max(6, np.ceil(4.0 * sigma_px)))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        # Normalizada: desenfocar reparte el mismo flujo, no anade luz.
        amp = flux / (2.0 * np.pi * sigma_px * sigma_px)
        img[y0:y1, x0:x1] += amp * np.exp(-0.5 * r2 / (sigma_px * sigma_px))
    return np.clip(img * gain, 0, 65535).astype(np.uint16)


def test_focus_metric_peaks_at_best_focus() -> None:
    sigmas = [1.2, 1.8, 2.6, 3.6, 5.0, 7.0, 10.0]
    values = [focus_metric(_field(s)) for s in sigmas]
    assert values[0] == max(values), f"el maximo no esta en foco: {values}"
    # y decae de forma monotona al desenfocar
    assert all(b < a for a, b in zip(values, values[1:])), values


def _field_n(n_stars: int, sigma_px: float, seed: int = 3) -> np.ndarray:
    """``n_stars`` estrellas identicas, a flujo total constante cada una."""
    img = _sky(seed=seed)
    h, w = img.shape
    rng = np.random.default_rng(7)
    spots = rng.uniform([40, 40], [w - 40, h - 40], size=(24, 2)).astype(int)
    for x, y in spots[:n_stars]:
        radius = int(max(6, np.ceil(4.0 * sigma_px)))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        amp = 6.0e5 / (2.0 * np.pi * sigma_px * sigma_px)
        img[y0:y1, x0:x1] += amp * np.exp(
            -0.5 * (((xx - x) ** 2 + (yy - y) ** 2) / (sigma_px * sigma_px))
        )
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_focus_metric_is_invariant_to_the_number_of_stars() -> None:
    """Una estrella que sale del campo no puede parecer mejor foco.

    Normalizando por el cuadrado de la suma de flujos, el numerador crecia como
    N y el denominador como N^2: la metrica valia 1/N, y bastaba perder una
    fuente a mitad del barrido para que la nitidez "subiera" sola. Con la suma
    de cuadrados, N se cancela.
    """
    values = [focus_metric(_field_n(n, 2.0)) for n in (3, 6, 12, 24)]
    assert min(values) > 0.0
    spread = (max(values) - min(values)) / max(values)
    assert spread < 0.05, f"la metrica depende del numero de estrellas: {values}"


def test_the_focus_peak_does_not_move_when_stars_leave_the_field() -> None:
    """El maximo tiene que caer en el mismo sitio con 12 estrellas y con 6."""
    sigmas = [1.6, 2.0, 2.6, 3.4, 4.5, 6.0]
    peaks = []
    for n_stars in (12, 6):
        values = [focus_metric(_field_n(n_stars, s)) for s in sigmas]
        peaks.append(sigmas[int(np.argmax(values))])
    assert peaks[0] == peaks[1], f"el pico se movio: {peaks}"


def test_focus_metric_is_invariant_to_gain() -> None:
    """Cambiar ganancia a mitad de barrido no debe mover la curva."""
    base = focus_metric(_field(2.0, gain=1.0))
    doubled = focus_metric(_field(2.0, gain=2.0))
    assert base > 0.0
    assert abs(doubled - base) / base < 0.15, (base, doubled)


def test_focus_metric_survives_a_sky_gradient() -> None:
    """Con gradiente de cielo la metrica sigue viendo las estrellas.

    Una mediana global confunde el gradiente con ruido, infla el umbral al
    doble y en un campo pobre deja la mascara vacia; el fondo por cajas no.
    """
    assert focus_metric(_field(1.5)) > 0.0


def test_cfa_plane_ignores_the_bayer_checkerboard() -> None:
    """Medir sobre el RAW completo mediria el mosaico, no el foco."""
    mosaic = np.full((200, 200), 1000, dtype=np.uint16)
    mosaic[0::2, 0::2] = 4000          # plano R muy distinto del resto
    plane = cfa_plane(mosaic)
    assert float(np.ptp(plane)) == 0.0
    assert focus_metric(mosaic) == 0.0


def test_parabolic_peak_refuses_edge_maxima() -> None:
    assert parabolic_peak([0, 100, 200], [1.0, 5.0, 9.0]) is None
    assert parabolic_peak([0, 100, 200], [9.0, 5.0, 1.0]) is None
    vertex = parabolic_peak([0, 100, 200], [1.0, 9.0, 1.0])
    assert vertex is not None and abs(vertex - 100.0) < 1e-6


def test_parabolic_peak_interpolates_between_samples() -> None:
    vertex = parabolic_peak([0, 100, 200], [4.0, 9.0, 8.0])
    assert vertex is not None
    assert 100.0 < vertex < 200.0


# ---------------------------------------------------------------- firmware ---

def test_firmware_exposes_a_third_axis_for_the_focuser() -> None:
    assert "AX_C = 2, AX_COUNT = 3" in FIRMWARE
    assert "FOCUS=1" in FIRMWARE
    assert "AXES=3" in FIRMWARE
    # Los ejes son arrays: con tres motores, triplicar variables sueltas por eje
    # es como se cuela un STEP con el DIR del eje de al lado.
    assert "STEP_PIN[AX_COUNT]" in FIRMWARE
    assert "DIR_PIN [AX_COUNT]" in FIRMWARE
    assert "for (int ax = 0; ax < AX_COUNT; ++ax)" in FIRMWARE


def test_firmware_test_mode_lifts_only_the_speed_clamp() -> None:
    """TEST corre el mismo perfil que MOVE, sin el tope de la montura."""
    assert "TESTMODE=1" in FIRMWARE
    assert "TEST_MAX_RATE_STEPS_S" in FIRMWARE
    assert "handleMoveCommand(true)" in FIRMWARE
    assert "handleMoveCommand(false)" in FIRMWARE
    # el clamp sigue existiendo para MOVE
    assert "MOVE_MAX_RATE_STEPS_S" in FIRMWARE
    # y la aceleracion se aplica igual en ambos
    assert "unlimited ? TEST_MAX_RATE_STEPS_S : MOVE_MAX_RATE_STEPS_S" in FIRMWARE


def test_firmware_can_stop_one_axis_without_stopping_the_others() -> None:
    assert "STOP A|B|C" in FIRMWARE
    assert "clearMovePlan(axis)" in FIRMWARE


# ------------------------------------------------------------- protocolo ----

def test_focus_move_requires_firmware_with_the_third_axis() -> None:
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with patch.object(ctrl, "status", return_value="EN=1 MS=64 MOVEPROFILES=1"):
        with pytest.raises(RuntimeError, match="FOCUS=1"):
            ctrl.move("C", "FWD", 100, 900, profile="smooth")


def test_focus_move_targets_axis_c() -> None:
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with (
        patch.object(
            ctrl,
            "status",
            return_value="EN=1 MS=64 MOVEPROFILES=1 FOCUS=1 TESTMODE=1",
        ),
        patch.object(ctrl, "send", return_value="OK") as mocked,
    ):
        ctrl.move("C", "REV", 250, 900, profile="smooth")
    assert mocked.call_args.args[0] == "MOVE C REV 250 900 SMOOTH"


def test_bench_test_uses_the_test_command() -> None:
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with (
        patch.object(
            ctrl,
            "status",
            return_value="EN=1 MS=64 MOVEPROFILES=1 FOCUS=1 TESTMODE=1",
        ),
        patch.object(ctrl, "send", return_value="OK") as mocked,
    ):
        ctrl.move("A", "FWD", 500, 2, profile="smooth", unlimited=True)
    assert mocked.call_args.args[0] == "TEST A FWD 500 2 SMOOTH"


def test_bench_test_refuses_legacy_firmware() -> None:
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with patch.object(ctrl, "status", return_value="EN=1 MS=64 MOVEPROFILES=1"):
        with pytest.raises(RuntimeError, match="TESTMODE=1"):
            ctrl.move("A", "FWD", 500, 2, unlimited=True)


def test_stop_can_target_a_single_axis() -> None:
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with patch.object(ctrl, "send", return_value="OK") as mocked:
        ctrl.stop("C")
    assert mocked.call_args.args[0] == "STOP C"

    with patch.object(ctrl, "send", return_value="OK") as mocked:
        ctrl.stop()
    assert mocked.call_args.args[0] == "STOP"

    with pytest.raises(ValueError):
        ctrl.stop("Z")


def test_unknown_axis_is_rejected_instead_of_silently_becoming_az() -> None:
    """Un eje mal escrito movia el azimut: peor que fallar."""
    ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
    with pytest.raises(ValueError):
        ctrl.move("Z", "FWD", 10, 900)


# ---------------------------------------------------------------- worker ----

class _FakeFocusMount:
    """Enfocador de banco: la nitidez depende de la distancia al foco real."""

    def __init__(self, best: int = 640, stop_at: int | None = None) -> None:
        self.best = int(best)
        self.position = 0
        # Tope mecanico retraido. Con dientes rotos, los pasos que sobran
        # patinan en vez de mover: es lo que hace repetible el homing.
        self.stop_at = stop_at
        self.commands: list[tuple[int, int]] = []
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def supports_focuser(self) -> bool:
        return True

    def focus_steps(self, direction, steps, delay_us, *, profile="smooth", blocking=True):
        self.commands.append((int(direction), int(steps)))
        self.position += int(direction) * int(steps)
        if self.stop_at is not None and self.position < self.stop_at:
            self.position = self.stop_at        # patina contra el tope
        return "OK"

    def stop_axis(self, axis_fw: str) -> str:
        return "OK"

    def sharpness(self) -> float:
        # Lorentziana: unimodal, con maximo en el foco real.
        offset = float(self.position - self.best)
        return 1000.0 / (1.0 + (offset / 300.0) ** 2)


def _worker(mount: _FakeFocusMount, cfg: FocuserConfig) -> tuple[FocuserWorker, list]:
    published: list = []
    counter = {"n": 0}

    def get_frame():
        counter["n"] += 1
        # Ruido gaussiano fijo mas un termino proporcional a la nitidez: la
        # metrica real se prueba aparte, aqui interesa la busqueda.
        value = mount.sharpness()
        img = np.full((64, 64), 100.0, dtype=np.float64)
        img[32, 32] += value
        return Frame(
            raw=img.astype(np.uint16),
            t_capture=time.perf_counter(),
            meta={"exp_ms": 0.0, "seq": counter["n"]},
        )

    worker = FocuserWorker(
        get_mount=lambda: mount,
        get_cfg=lambda: cfg,
        get_frame=get_frame,
        publish_state=lambda patch: published.append(patch),
        out_log=None,
    )
    return worker, published


def _run(worker: FocuserWorker, **request) -> None:
    """Ejecuta una peticion en el hilo del test, sin arrancar el worker."""
    worker._handle_request(dict(request))


def test_autofocus_lands_on_the_sharpest_position(monkeypatch) -> None:
    mount = _FakeFocusMount(best=640)
    cfg = FocuserConfig(delay_us=100, autofocus_settle_s=0.0, autofocus_frames=1)
    worker, _published = _worker(mount, cfg)
    monkeypatch.setattr("focuser.focus_metric", lambda img: mount.sharpness())

    _run(
        worker,
        kind="autofocus",
        params={
            "coarse_step": 300,
            "coarse_points": 7,
            "fine_step": 60,
            "fine_points": 5,
            "settle_s": 0.0,
            "frames": 1,
        },
    )
    assert abs(worker.position - 640) <= 60, worker.position


def test_autofocus_extends_the_sweep_when_the_peak_is_outside_it(monkeypatch) -> None:
    """Un maximo en el extremo del barrido no es un maximo.

    Aceptarlo daria por bueno el punto medido mas lejano, que es exactamente lo
    que pasa cuando el enfocador arranca lejos del foco.
    """
    mount = _FakeFocusMount(best=2000)
    cfg = FocuserConfig(delay_us=100, max_travel_steps=0)
    worker, _published = _worker(mount, cfg)
    monkeypatch.setattr("focuser.focus_metric", lambda img: mount.sharpness())

    _run(
        worker,
        kind="autofocus",
        params={
            "coarse_step": 300,
            "coarse_points": 5,      # cubre solo +-600, el foco esta en +2000
            "fine_step": 60,
            "fine_points": 5,
            "settle_s": 0.0,
            "frames": 1,
        },
    )
    assert worker.position > 1500, worker.position


def test_manual_move_tracks_relative_position_and_applies_backlash() -> None:
    mount = _FakeFocusMount()
    cfg = FocuserConfig(step_size=200, delay_us=100, backlash_steps=25)
    worker, _published = _worker(mount, cfg)

    _run(worker, kind="move", direction=+1, steps=200)
    assert worker.position == 200
    assert mount.commands == [(1, 200)]

    # Al invertir se envia primero el juego, que no cuenta como recorrido util.
    _run(worker, kind="move", direction=-1, steps=200)
    assert mount.commands[-2:] == [(-1, 25), (-1, 200)]
    assert worker.position == 0


def test_travel_limit_blocks_moves_that_would_hit_the_stop() -> None:
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100, max_travel_steps=500)
    worker, published = _worker(mount, cfg)

    _run(worker, kind="move", direction=+1, steps=900)
    assert worker.position == 0
    assert mount.commands == []
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "recorrido" in err for err in errors)


def test_invert_flips_the_commanded_direction_only() -> None:
    """La posicion contada sigue siendo «acercar = +», la mande donde la mande."""
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100, invert=True)
    worker, _published = _worker(mount, cfg)

    _run(worker, kind="move", direction=+1, steps=100)
    assert worker.position == 100
    assert mount.commands == [(-1, 100)]


def test_measure_ignores_frames_captured_before_the_move() -> None:
    """El frame que ya estaba en el buffer mide el foco de la posicion anterior.

    Usarlo corre la curva un paso entero y el maximo cae donde no es.
    """
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100)
    stale = Frame(
        raw=np.zeros((64, 64), dtype=np.uint16),
        t_capture=time.perf_counter() - 5.0,
        meta={"exp_ms": 0.0},
    )
    fresh_holder: dict = {"frame": stale}

    worker = FocuserWorker(
        get_mount=lambda: mount,
        get_cfg=lambda: cfg,
        get_frame=lambda: fresh_holder["frame"],
        publish_state=lambda patch: None,
        out_log=None,
    )

    def _publish_fresh() -> None:
        time.sleep(0.15)
        fresh_holder["frame"] = Frame(
            raw=np.ones((64, 64), dtype=np.uint16),
            t_capture=time.perf_counter(),
            meta={"exp_ms": 0.0},
        )

    thread = threading.Thread(target=_publish_fresh, daemon=True)
    thread.start()
    with patch("focuser.focus_metric", side_effect=lambda img: float(img.max())):
        value = worker._measure(settle_s=0.0, frames=1)
    thread.join(timeout=2.0)

    assert value == 1.0, "se midio el frame anterior al movimiento"


def test_autofocus_can_be_cancelled_midway(monkeypatch) -> None:
    mount = _FakeFocusMount(best=640)
    cfg = FocuserConfig(delay_us=100)
    worker, published = _worker(mount, cfg)

    calls = {"n": 0}

    def _metric(_img):
        calls["n"] += 1
        if calls["n"] >= 3:
            worker.cancel()
        return mount.sharpness()

    monkeypatch.setattr("focuser.focus_metric", _metric)
    _run(
        worker,
        kind="autofocus",
        params={
            "coarse_step": 300,
            "coarse_points": 9,
            "fine_step": 60,
            "fine_points": 5,
            "settle_s": 0.0,
            "frames": 1,
        },
    )
    states = [p["focuser"].get("autofocus") for p in published if "focuser" in p]
    assert "cancelled" in states
    assert calls["n"] < 9, "el barrido siguio despues de cancelar"


def test_autofocus_reports_failure_without_measurable_signal(monkeypatch) -> None:
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100)
    worker, published = _worker(mount, cfg)
    monkeypatch.setattr("focuser.focus_metric", lambda img: 0.0)

    _run(
        worker,
        kind="autofocus",
        params={
            "coarse_step": 300,
            "coarse_points": 3,
            "fine_step": 60,
            "fine_points": 3,
            "settle_s": 0.0,
            "frames": 1,
        },
    )
    focuser_states = [p["focuser"] for p in published if "focuser" in p]
    assert focuser_states[-1]["autofocus"] == "failed"
    assert "senal" in str(focuser_states[-1]["last_error"])


def test_worker_refuses_a_firmware_without_the_third_axis() -> None:
    mount = _FakeFocusMount()
    mount.supports_focuser = lambda: False  # type: ignore[method-assign]
    cfg = FocuserConfig(delay_us=100)
    worker, published = _worker(mount, cfg)

    _run(worker, kind="move", direction=+1, steps=100)
    assert mount.commands == []
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "tercer eje" in err for err in errors)


# ------------------------------------------------------------ integracion ---

def test_demo_autofocus_finds_the_simulated_focus() -> None:
    """Extremo a extremo con la metrica real sobre el renderizador del demo.

    Los tests de arriba sustituyen la metrica por una funcion conocida para
    probar la busqueda; este no sustituye nada: mide sobre frames renderizados,
    con su gradiente de cielo, su ruido de Poisson y su seeing variable.
    """
    import time

    from app_runner import AppRunner
    from config import AppConfig
    from terminal_app import TerminalApp

    cfg = AppConfig()
    cfg.simulation.enabled = True
    cfg.simulation.seed = 7
    cfg.simulation.focus_best_offset_steps = 600
    # El campo del demo cambia con la hora real: el apuntado az/alt es fijo pero
    # el cielo gira, asi que a veces toca una zona pobre. Se enfoca sobre una
    # estrella brillante -- que es como se enfoca de verdad -- para que la
    # prueba mida la convergencia y no la suerte del campo.
    cfg.simulation.star_flux_adu = 400_000.0
    cfg.focuser.delay_us = 200
    cfg.focuser.autofocus_coarse_step = 400
    cfg.focuser.autofocus_coarse_points = 5
    cfg.focuser.autofocus_fine_step = 80
    cfg.focuser.autofocus_fine_points = 5
    cfg.focuser.autofocus_settle_s = 0.0
    # Con un solo frame por punto, el seeing del demo (+-12% en sigma) mueve la
    # metrica un +-50%: mas que la diferencia entre posiciones vecinas. La
    # mediana de varios frames es justo para lo que existe este parametro.
    cfg.focuser.autofocus_frames = 3

    runner = AppRunner(cfg)
    terminal = TerminalApp(runner)
    runner.start()
    try:
        terminal.execute_line("camera connect")
        terminal.execute_line("mount connect")
        terminal.execute_line("wait camera.connected true 15")
        truth = float(runner._ensure_simulation_state().focus_best_steps)

        # El apuntado del demo es fijo en az/alt pero el cielo gira con la hora
        # real, asi que a veces toca una zona sin nada medible. Esta prueba es
        # sobre la convergencia, no sobre la suerte del campo: si no hay senal
        # ni en el foco exacto, no hay nada que converger.
        state = runner._ensure_simulation_state()
        state._focus_steps = float(truth)
        deadline = time.monotonic() + 10.0
        best_metric = 0.0
        while time.monotonic() < deadline and best_metric <= 0.0:
            frame = runner._get_latest_frame()
            if frame is not None:
                best_metric = focus_metric(frame.raw)
            time.sleep(0.2)
        state._focus_steps = 0.0
        if best_metric <= 0.0:
            pytest.skip("el campo del demo no tiene senal medible a esta hora")
        terminal.execute_line("focus auto")
        terminal.execute_line("await focus 240")
        state = runner.get_state().focuser
    finally:
        terminal.close()
        runner.stop()

    assert state.autofocus == "done", state.last_error
    # Dos pasos finos son 160 microsteps, o sigma 1.6 px de desenfoque: muy por
    # debajo del seeing del demo (sigma ~3 px), es decir dentro de foco. Pedir
    # menos seria pedir una precision que el propio seeing no deja medir.
    tolerance = 2 * cfg.focuser.autofocus_fine_step
    assert abs(state.position - truth) <= tolerance, (
        f"foco encontrado {state.position:+d}, real {truth:+.0f}"
    )


def test_firmware_focuser_pins_avoid_unusable_and_strapping_gpios() -> None:
    """Los pines del enfocador tienen que poder ser salidas y dejar arrancar.

    GPIO 34/35/36/39 son sólo entrada y no pueden mover STEP ni DIR. GPIO 12
    elige la tensión de flash en el reset: un driver que lo mantenga alto impide
    que la placa arranque, y eso se ve como un ESP32 muerto, no como un error de
    cableado.
    """
    import re

    forbidden = {0, 2, 4, 5, 12, 15, 34, 35, 36, 39}
    step = re.search(r"STEP_PIN\[AX_COUNT\] = \{([^}]*)\}", FIRMWARE)
    dirs = re.search(r"DIR_PIN \[AX_COUNT\] = \{([^}]*)\}", FIRMWARE)
    assert step and dirs
    pins = [
        int(value.strip())
        for group in (step.group(1), dirs.group(1))
        for value in group.split(",")
    ]
    assert len(pins) == 6
    assert not (set(pins) & forbidden), sorted(set(pins) & forbidden)


# ------------------------------------------------- homing y presets ---------

def test_homing_overshoots_the_travel_and_defines_a_repeatable_zero(tmp_path) -> None:
    """El homing tiene que pasarse del recorrido: el tope es lo que busca.

    El pinon patina en ese extremo, asi que los pasos de mas no fuerzan nada y
    dejan el enfocador siempre en el mismo sitio fisico.
    """
    mount = _FakeFocusMount()
    cfg = FocuserConfig(
        delay_us=100,
        home_travel_steps=32_000,
        home_overshoot_steps=3_000,
        max_travel_steps=500,          # deliberadamente mucho menor
        presets_path=str(tmp_path / "presets.json"),
    )
    worker, _published = _worker(mount, cfg)

    _run(worker, kind="home")

    assert worker.homed is True
    assert worker.position == 0
    # un solo movimiento de retraccion, mas largo que el recorrido util
    assert mount.commands == [(-1, 35_000)]


def test_the_travel_guard_becomes_asymmetric_once_homed(tmp_path) -> None:
    """Con homing, el cero es el tope: el rango real es 0..recorrido.

    Sin homing no se sabe donde esta el enfocador dentro de su carrera, y lo
    unico honesto es limitar simetricamente alrededor del punto de partida.
    """
    mount = _FakeFocusMount()
    cfg = FocuserConfig(
        delay_us=100,
        home_travel_steps=1_000,
        home_overshoot_steps=100,
        max_travel_steps=5_000,
        presets_path=str(tmp_path / "presets.json"),
    )
    worker, published = _worker(mount, cfg)

    _run(worker, kind="home")
    mount.commands.clear()

    _run(worker, kind="move", direction=+1, steps=900)   # dentro
    assert worker.position == 900

    _run(worker, kind="move", direction=+1, steps=200)   # se pasaria del tope
    assert worker.position == 900, "se permitio salir del recorrido conocido"

    _run(worker, kind="move", direction=-1, steps=2_000)  # por debajo de cero
    assert worker.position == 900
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "recorrido" in err for err in errors)


def test_presets_round_trip_through_disk(tmp_path) -> None:
    from focuser import FocusPresets

    store = FocusPresets(tmp_path / "presets.json")
    assert store.load() == {}

    store.save("x2", 15_200, homed=True)
    store.save("x5", 18_400, homed=True)
    assert set(store.load()) == {"x2", "x5"}
    assert store.get("x2")["position"] == 15_200

    # sobrescribir, no duplicar
    store.save("x2", 15_400, homed=True)
    assert store.get("x2")["position"] == 15_400

    assert store.delete("x5") is True
    assert store.delete("x5") is False
    assert set(store.load()) == {"x2"}


def test_a_corrupt_preset_file_does_not_take_the_focuser_down(tmp_path) -> None:
    from focuser import FocusPresets

    path = tmp_path / "presets.json"
    path.write_text("{esto no es json", encoding="utf-8")
    store = FocusPresets(path)
    assert store.load() == {}
    # y se puede volver a escribir encima
    store.save("x1", 100, homed=False)
    assert store.get("x1")["position"] == 100


def test_a_preset_saved_without_homing_is_refused_in_a_homed_session(tmp_path) -> None:
    """Un origen arbitrario aplicado a otra sesion mueve el foco a cualquier sitio."""
    mount = _FakeFocusMount()
    cfg = FocuserConfig(
        delay_us=100,
        home_travel_steps=32_000,
        home_overshoot_steps=1_000,
        presets_path=str(tmp_path / "presets.json"),
    )
    worker, published = _worker(mount, cfg)
    worker.presets.save("x2", 15_000, homed=False)

    _run(worker, kind="home")
    mount.commands.clear()
    _run(worker, kind="preset", name="x2")

    assert mount.commands == [], "aplico un preset de origen desconocido"
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "sin homing" in err for err in errors)


def test_a_homed_preset_moves_to_its_position(tmp_path) -> None:
    mount = _FakeFocusMount()
    cfg = FocuserConfig(
        delay_us=100,
        home_travel_steps=32_000,
        home_overshoot_steps=1_000,
        presets_path=str(tmp_path / "presets.json"),
    )
    worker, _published = _worker(mount, cfg)
    worker.presets.save("x5", 18_400, homed=True)

    _run(worker, kind="home")
    mount.commands.clear()
    _run(worker, kind="preset", name="x5")

    assert worker.position == 18_400
    assert mount.commands == [(+1, 18_400)]


def test_an_unknown_preset_is_reported_not_ignored(tmp_path) -> None:
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100, presets_path=str(tmp_path / "presets.json"))
    worker, published = _worker(mount, cfg)

    _run(worker, kind="preset", name="x3")
    assert mount.commands == []
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "x3" in err for err in errors)


def test_a_preset_from_another_unhomed_session_is_refused(tmp_path) -> None:
    """Sin homing el cero era el punto donde se abrio la app aquel dia.

    Ese cero ya no existe hoy, asi que aplicar la posicion moveria el enfocador
    a un sitio arbitrario -- y sin homing tampoco hay tope que lo detenga.
    """
    mount = _FakeFocusMount()
    cfg = FocuserConfig(delay_us=100, presets_path=str(tmp_path / "presets.json"))
    worker, published = _worker(mount, cfg)

    # guardado por "otra sesion"
    worker.presets.save("x1", 9_000, homed=False, session="sesion-de-ayer")
    _run(worker, kind="preset", name="x1")
    assert mount.commands == []
    errors = [p["focuser"].get("last_error") for p in published if "focuser" in p]
    assert any(err and "otra sesion" in err for err in errors)

    # pero dentro de la misma sesion si vale
    worker.presets.save("x1", 400, homed=False, session=worker.session_id)
    _run(worker, kind="preset", name="x1")
    assert worker.position == 400


# ------------------------------------------- busqueda guiada por presets ----

def _homed_worker(tmp_path, best: int, *, stop_at: int | None = 0, **overrides):
    mount = _FakeFocusMount(best=best, stop_at=stop_at)
    cfg = FocuserConfig(
        delay_us=100,
        home_travel_steps=40_000,
        home_overshoot_steps=1_000,
        autofocus_settle_s=0.0,
        autofocus_frames=1,
        autofocus_coarse_step=400,
        autofocus_coarse_points=9,
        autofocus_fine_step=80,
        autofocus_fine_points=5,
        presets_path=str(tmp_path / "presets.json"),
        **overrides,
    )
    worker, published = _worker(mount, cfg)
    return mount, cfg, worker, published


def _autofocus(worker, monkeypatch, mount, **params):
    monkeypatch.setattr("focuser.focus_metric", lambda img: mount.sharpness())
    base = {"settle_s": 0.0, "frames": 1}
    base.update(params)
    _run(worker, kind="autofocus", params=base)


def test_a_preset_prior_narrows_the_search(tmp_path, monkeypatch) -> None:
    """Con una posicion conocida no hay que barrer todo el recorrido."""
    mount, _cfg, worker, _pub = _homed_worker(tmp_path, best=15_300)
    _run(worker, kind="home")
    worker.presets.save("x2", 15_200, homed=True)

    _run(worker, kind="goto", position=15_200)
    mount.commands.clear()
    _autofocus(worker, monkeypatch, mount)

    assert abs(worker.position - 15_300) <= 120, worker.position
    travelled = sum(steps for _d, steps in mount.commands)
    # el barrido general por si solo son 9x400 = 3200 pasos de ida
    assert travelled < 4_000, f"la busqueda dirigida recorrio {travelled} pasos"


def test_the_prior_is_inferred_from_the_nearest_preset(tmp_path, monkeypatch) -> None:
    """Tras 'focus goto x5' la app ya sabe que barlow esta puesto."""
    mount, _cfg, worker, _pub = _homed_worker(tmp_path, best=18_500)
    _run(worker, kind="home")
    worker.presets.save("x1", 12_800, homed=True)
    worker.presets.save("x5", 18_400, homed=True)

    _run(worker, kind="goto", position=18_400)
    assert worker._nearest_preset() == "x5"

    _autofocus(worker, monkeypatch, mount)
    assert abs(worker.position - 18_500) <= 120
    # y el resultado queda anotado en el historial de ese preset
    assert len(worker.presets.get("x5")["history"]) == 1
    assert worker.presets.get("x1")["history"] == []


def test_a_stale_prior_falls_back_to_the_general_search(tmp_path, monkeypatch) -> None:
    """Si el foco se movio mas de lo previsto, la ventana no vale.

    Devolver el mejor punto de una ventana que no contiene el maximo seria peor
    que no usar prior: daria un foco malo con aire de exito.
    """
    # el foco real esta lejisimos del preset
    mount, _cfg, worker, _pub = _homed_worker(
        tmp_path, best=17_000, autofocus_prior_window_steps=300
    )
    _run(worker, kind="home")
    worker.presets.save("x2", 15_200, homed=True)
    _run(worker, kind="goto", position=15_200)

    _autofocus(worker, monkeypatch, mount, coarse_step=600, coarse_points=15)
    assert abs(worker.position - 17_000) <= 200, worker.position


def test_the_window_widens_with_the_measured_spread(tmp_path) -> None:
    """El ancho lo dice la dispersion de este equipo, no una constante.

    El foco cambia de noche a noche con la temperatura; cuanto, solo lo sabe el
    historial.
    """
    _mount, _cfg, worker, _pub = _homed_worker(
        tmp_path, best=0, autofocus_prior_window_steps=200, autofocus_prior_window_sigma=4.0
    )
    _run(worker, kind="home")
    worker.presets.save("x2", 15_000, homed=True)

    _name, _center, tight = worker._search_prior({"preset": "x2"})
    assert tight == 200, "sin historial deberia usar la ventana minima"

    for position in (14_800, 15_100, 15_400, 14_900, 15_300):
        worker.presets.record_result("x2", position)
    name, center, wide = worker._search_prior({"preset": "x2"})

    assert name == "x2"
    assert wide > tight, "la ventana no crecio con la dispersion observada"
    # y el centro sigue a la deriva del historial, sin tocar el nominal
    assert center == pytest.approx(15_100, abs=250)
    assert worker.presets.get("x2")["position"] == 15_000


def test_history_survives_re_saving_the_nominal(tmp_path) -> None:
    from focuser import FocusPresets

    store = FocusPresets(tmp_path / "presets.json")
    store.save("x2", 15_000, homed=True)
    for position in (15_100, 15_200):
        store.record_result("x2", position)

    store.save("x2", 15_150, homed=True)          # el usuario corrige el nominal
    assert store.get("x2")["history"] == [15_100, 15_200]


def test_history_is_capped(tmp_path) -> None:
    from focuser import FocusPresets

    store = FocusPresets(tmp_path / "presets.json")
    store.save("x1", 100, homed=True)
    for i in range(30):
        store.record_result("x1", 100 + i, history_max=5)
    assert len(store.get("x1")["history"]) == 5


def test_without_presets_the_search_is_still_general(tmp_path, monkeypatch) -> None:
    """Sin informacion previa tiene que seguir funcionando igual que antes."""
    # sin homing y sin tope: el barrido general puede ir a ambos lados del cero
    mount, _cfg, worker, _pub = _homed_worker(tmp_path, best=640, stop_at=None)
    _autofocus(worker, monkeypatch, mount, coarse_step=300, coarse_points=7, fine_step=60, fine_points=5)
    assert abs(worker.position - 640) <= 60, worker.position


# ------------------------------------------------ deteccion de eje muerto ---

def test_axis_verdicts_separate_dead_from_healthy() -> None:
    """Un TMC2209 en STEP/DIR no da realimentacion: el firmware responde OK
    aunque no haya motor al otro lado. Lo unico que distingue un eje vivo de uno
    muerto es si el campo se movio."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_axes", Path(__file__).resolve().parents[1] / "scripts" / "check_axes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resuelve su modulo via sys.modules; sin registrarlo, falla.
    sys.modules["check_axes"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("check_axes", None)

    def verdict(measured, expected=120.0, ok=True):
        return mod.AxisResult("az", 500, expected, measured, ok, "ok").verdict

    assert verdict(118.0) == "OK"
    assert verdict(0.0) == "NO RESPONDE"        # cable suelto
    assert verdict(3.0) == "NO RESPONDE"        # solo ruido de medida
    assert verdict(30.0) == "SOSPECHOSO"        # escala optica mal puesta
    assert verdict(0.0, ok=False) == "SIN MEDIDA"
    # un eje sano medido con la escala de otro barlow no puede dar "OK"
    assert verdict(120.0 / 5.0) == "NO RESPONDE"


def test_barlow_list_matches_the_hardware() -> None:
    """Ofrecer factores que no se tienen invita a dejar la escala mal puesta."""
    from ui.tabs_mixin import BARLOW_FACTORS, FOCUS_PRESET_NAMES

    assert BARLOW_FACTORS == (1, 2, 5)
    assert set(FOCUS_PRESET_NAMES) == {f"x{b}" for b in BARLOW_FACTORS}


def test_axis_check_flags_a_wrong_optical_scale() -> None:
    """Un barlow distinto al configurado es tan silencioso como un cable suelto.

    El campo se mueve mucho mas de lo previsto porque la escala supuesta es mas
    gruesa que la real; el plate solving entonces busca un campo que no es y
    falla sin decir por que.
    """
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_axes", Path(__file__).resolve().parents[1] / "scripts" / "check_axes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_axes"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("check_axes", None)

    def verdict(measured, expected=120.0):
        return mod.AxisResult("az", 500, expected, measured, True, "ok").verdict

    # x5 montado, x1 configurado: cinco veces mas desplazamiento del esperado
    assert verdict(600.0) == "ESCALA MAL"
    assert verdict(120.0) == "OK"
    # el acoplamiento de ejes infla algo la medida, y eso no es un error de escala
    assert verdict(180.0) == "OK"


def test_axis_check_sizes_the_test_move_for_the_optics() -> None:
    """El movimiento de prueba tiene que dejar solape entre las dos imagenes.

    Con 400 pasos a x1 el campo se corre 1354 px, mas que el alto del sensor: sin
    solape no hay nada que correlacionar y la prueba no mide nada.
    """
    import importlib.util
    import math
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_axes", Path(__file__).resolve().parents[1] / "scripts" / "check_axes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_axes"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("check_axes", None)

    deg_per_step = 0.000625
    for barlow in (1, 2, 5):
        arcsec_px = math.degrees(2.9e-6 / (0.9 * barlow)) * 3600.0
        steps = mod.test_steps_for_scale(arcsec_px, deg_per_step)
        shift = steps * deg_per_step * 3600.0 / arcsec_px
        assert shift < 1096, f"x{barlow}: {shift:.0f} px no deja solape"
        assert steps >= mod.MIN_TEST_STEPS, f"x{barlow}: {steps} pasos se los come el juego"
