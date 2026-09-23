"""El contador de pasos no puede apuntarse pasos que la montura no dio.

El firmware ASIGNA el plan pendiente (``g_moveRem[ax] = steps``) en vez de
sumarlo: un MOVE nuevo descarta en silencio lo que el anterior aun no habia
ejecutado. Como la emulacion de velocidad contabiliza lo *comandado*, esos
pasos descartados corrompian ``steps_est``, y con el el modelo GoTo: el
telescopio decia estar donde no estaba.

El lote se dimensiona con el dt del ciclo anterior, asi que un ciclo lento
producia un lote largo que el ciclo rapido siguiente machacaba a media
ejecucion. Ahora solo se emite lo que cabe entero en la ventana estimada.
"""
from __future__ import annotations

import types

import pytest

import app_runner
from app_runner import (
    _RATE_EMUL_PULSE_US,
    _RATE_EMUL_WINDOW_FRAC,
    AppRunner,
)


def _runner() -> AppRunner:
    """Un AppRunner sin __init__: solo el estado que toca la emulacion."""
    r = AppRunner.__new__(AppRunner)
    r._rate_emul_period_s = None
    return r


def _batch_duration_s(steps: int, delay_us: float) -> float:
    return abs(steps) * (delay_us + _RATE_EMUL_PULSE_US) / 1.0e6


class TestEmissionWindow:
    def test_without_a_known_period_nothing_is_trimmed(self) -> None:
        """En el primer ciclo no hay ningun MOVE en vuelo al que adelantarse."""
        r = _runner()
        assert r._fit_steps_in_emission_window(400, 1000) == 400

    def test_a_batch_never_outlives_its_window(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        delay_us = 1000
        emitted = r._fit_steps_in_emission_window(400, delay_us)
        assert 0 < emitted < 400, "un lote de 400 no cabe en 300 ms a 1 ms/paso"
        assert _batch_duration_s(emitted, delay_us) <= 0.30

    def test_the_window_keeps_a_margin_for_jitter(self) -> None:
        """El ciclo lo marca la camara, no un reloj: llega antes o despues."""
        r = _runner()
        r._note_rate_emul_period(0.30)
        delay_us = 1000
        emitted = r._fit_steps_in_emission_window(10_000, delay_us)
        assert _batch_duration_s(emitted, delay_us) <= 0.30 * _RATE_EMUL_WINDOW_FRAC + 1e-9

    def test_sign_survives_the_trim(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        assert r._fit_steps_in_emission_window(-400, 1000) < 0

    def test_a_batch_that_already_fits_is_left_alone(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        assert r._fit_steps_in_emission_window(3, 1000) == 3

    def test_a_single_step_longer_than_the_window_still_goes_out(self) -> None:
        """Esperar a que quepa era esperar para siempre.

        A ritmo sideral esta montura pide ~100 ms entre pasos y la ventana
        del lazo son ~16 ms: ningun paso cabe jamas. Recortando a cero el
        acumulador crecia sin emitir nunca, el estado publicaba velocidades
        sanas y la montura no recibia un solo MOVE -- el seguimiento parecia
        andar y la imagen derivaba a ritmo sideral entero. Sale un paso, y de
        que no lo pise el siguiente se encarga la reserva de linea por eje.
        """
        r = _runner()
        r._note_rate_emul_period(0.010)
        assert r._fit_steps_in_emission_window(5, 50_000) == 1

    def test_a_single_step_keeps_its_sign_when_it_does_not_fit(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.010)
        assert r._fit_steps_in_emission_window(-5, 50_000) == -1

    def test_zero_stays_zero(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        assert r._fit_steps_in_emission_window(0, 1000) == 0


class TestPeriodEstimate:
    def test_a_pause_is_not_a_control_period(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        before = r._rate_emul_period_s
        r._note_rate_emul_period(120.0)      # la app estuvo parada
        assert r._rate_emul_period_s == before

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_nonsense_intervals_are_ignored(self, bad: float) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        before = r._rate_emul_period_s
        r._note_rate_emul_period(bad)
        assert r._rate_emul_period_s == before

    def test_the_estimate_follows_a_changing_cadence(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.10)
        for _ in range(40):
            r._note_rate_emul_period(0.50)
        assert r._rate_emul_period_s == pytest.approx(0.50, abs=0.01)


class TestNoStepsAreLost:
    """Lo que no se emite se queda en el acumulador, no se descarta."""

    def test_the_remainder_survives_to_the_next_cycle(self) -> None:
        r = _runner()
        r._note_rate_emul_period(0.30)
        delay_us = 1000
        wanted = 400
        emitted = r._fit_steps_in_emission_window(wanted, delay_us)
        remainder = wanted - emitted
        assert remainder > 0
        # El ciclo siguiente arranca con ese resto en el acumulador y lo saca
        # entero si cabe; en ningun caso se pierde.
        assert r._fit_steps_in_emission_window(remainder, delay_us) > 0


class _FakeMount:
    """Registra los MOVE que recibe, sin hardware detras."""

    def __init__(self) -> None:
        self.moves: list[tuple[str, int, int, int]] = []

    def move_steps(self, *, axis, direction, steps, delay_us, **kwargs) -> None:
        self.moves.append((axis.value, int(direction), int(steps), int(delay_us)))

    def stop(self) -> None:
        pass


def _emitting_runner(monkeypatch, clock: list[float]) -> tuple[AppRunner, _FakeMount]:
    """AppRunner minimo capaz de ejecutar la emulacion de velocidad."""
    import threading

    from config import AppConfig

    r = AppRunner.__new__(AppRunner)
    r.cfg = AppConfig()
    r._mount = _FakeMount()
    r._rate_emul_lock = threading.Lock()
    r._rate_emul_last_t = None
    r._rate_emul_acc_az = 0.0
    r._rate_emul_acc_alt = 0.0
    r._rate_emul_active = False
    r._rate_emul_period_s = None
    r._rate_emul_busy_until_az = 0.0
    r._rate_emul_busy_until_alt = 0.0
    r._is_manual_move_active = lambda: False
    monkeypatch.setattr(app_runner, "_perf", lambda: clock[0])
    return r, r._mount


class TestSiderealRateActuallyReachesTheMount:
    """La regresion que dejaba la montura quieta con el tracking en verde."""

    def test_a_sidereal_rate_emits_moves(self, monkeypatch) -> None:
        clock = [0.0]
        r, mount = _emitting_runner(monkeypatch, clock)
        rate = 10.0  # pasos/s: el orden del sideral en esta montura
        for _ in range(600):  # 10 s de lazo a 60 Hz
            clock[0] += 1.0 / 60.0
            r._mount_rate_safe(-rate, 0.0)
        assert mount.moves, "el lazo no mando ni un MOVE a ritmo sideral"
        emitted = sum(steps for _, _, steps, _ in mount.moves)
        # 10 s a 10 pasos/s = ~100 pasos. La reserva de linea puede diferir
        # alguno al ciclo siguiente, no perderlo.
        assert 80 <= emitted <= 100

    def test_a_move_is_never_sent_over_one_still_in_flight(self, monkeypatch) -> None:
        clock = [0.0]
        r, mount = _emitting_runner(monkeypatch, clock)
        # Rate alto: el acumulador llena un paso mucho antes de que termine
        # el MOVE anterior, que es cuando el firmware descartaria el resto.
        rate = 300.0
        for _ in range(600):
            clock[0] += 1.0 / 60.0
            r._mount_rate_safe(rate, 0.0)
        t = 0.0
        for _, _, steps, delay_us in mount.moves:
            assert t <= clock[0]
            t += _batch_duration_s(steps, delay_us)
        assert t <= clock[0] + 1e-6, "los MOVE emitidos no caben en el tiempo transcurrido"

    def test_the_axes_reserve_their_line_independently(self, monkeypatch) -> None:
        clock = [0.0]
        r, mount = _emitting_runner(monkeypatch, clock)
        for _ in range(600):
            clock[0] += 1.0 / 60.0
            r._mount_rate_safe(-10.0, 6.0)
        axes = {axis for axis, _, _, _ in mount.moves}
        assert axes == {"az", "alt"}, f"algun eje se quedo sin emitir: {axes}"
