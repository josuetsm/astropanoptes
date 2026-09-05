"""El ancla: la referencia original no se pierde al perder el enganche.

Refrescar el keyframe mantiene viva la correlacion cuando el cielo rota y el
seeing cambia. El problema era que ese refresco *redefinia el objetivo*: cada
vez, el blanco se corria unos pixeles, y tras un apagon el tracker adoptaba como
correcta la posicion en la que hubiera quedado. La montura seguia un punto que
se alejaba solo, mostrando error cero en pantalla.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracking import (
    make_tracking_state,
    rechain_keyframe,
    reset_keyframe,
    reset_tracker,
    tracking_step,
)


H, W = 256, 320
_YY, _XX = np.mgrid[0:H, 0:W]
_STARS = ((80, 60, 9000), (200, 100, 6000), (140, 180, 7000), (250, 200, 5000))


def _frame(dx: float = 0.0, dy: float = 0.0, *, blank: bool = False) -> np.ndarray:
    """Campo estelar desplazado (dx, dy), o cielo sin senal."""
    rng = np.random.default_rng(1)
    img = rng.normal(600, 8, (H, W))
    if not blank:
        for sx, sy, amp in _STARS:
            img += amp * np.exp(
                -(((_XX - (sx + dx)) ** 2 + (_YY - (sy + dy)) ** 2) / (2 * 3.0**2))
            )
    return np.clip(img, 0, 65535).astype(np.uint16)


class _Run:
    """Secuencia de frames con reloj propio."""

    def __init__(self) -> None:
        self.state = make_tracking_state()
        self.t = 0.0

    def feed(self, n: int, *, step: float = 0.1, **frame_kw):
        out = None
        for _ in range(n):
            out = tracking_step(
                self.state, _frame(**frame_kw), now_t=self.t, tracking_enabled=False
            )
            self.t += step
        return out


def _err(out) -> float:
    return float(np.hypot(out.x_hat, out.y_hat))


def test_dropout_does_not_erase_the_target() -> None:
    """Tras un apagon con el campo desplazado, el error medido es el real.

    Antes, perder el enganche descartaba el keyframe: el frame siguiente pasaba
    a ser el objetivo y el error volvia a cero, aunque el telescopio estuviera
    apuntando a otro sitio.
    """
    run = _Run()
    settled = run.feed(10)
    assert _err(settled) < 1.0, "el campo quieto deberia dar error nulo"

    run.feed(20, blank=True)                     # apagon: se pierde el enganche
    moved = run.feed(40, step=0.2, dx=30, dy=20)  # vuelve, pero desplazado

    truth = float(np.hypot(30, 20))
    assert moved.anchor_chained, "se solto el ancla en vez de encadenar"
    assert _err(moved) == pytest.approx(truth, abs=2.0), (
        f"error medido {_err(moved):.2f} px, real {truth:.2f} px"
    )


def test_returning_to_the_anchor_collapses_the_chain() -> None:
    """El objetivo es volver a la primera referencia, y volver de verdad."""
    run = _Run()
    run.feed(10)
    run.feed(20, blank=True)
    run.feed(40, step=0.2, dx=30, dy=20)
    back = run.feed(30, step=0.2)                 # el campo vuelve al origen

    assert _err(back) < 2.0, f"no se cerro el lazo: {_err(back):.2f} px"
    assert not back.anchor_chained, "la cadena no se colapso al volver al ancla"
    assert back.anchor_px == pytest.approx(0.0, abs=1e-6)


def test_rechain_preserves_the_distance_to_the_anchor() -> None:
    """Cambiar de keyframe de trabajo no puede alterar el error total."""
    state = make_tracking_state()
    # Firmas centinela: basta con que sean objetos distintos entre si.
    anchor, work1, work2 = object(), object(), object()
    reset_keyframe(state, signature=anchor, now_t=0.0)
    state.x_hat, state.y_hat = 12.0, -5.0

    rechain_keyframe(state, signature=work1, now_t=1.0)
    assert state.x_hat == 0.0 and state.y_hat == 0.0
    # el error se mudo al offset acumulado, no desaparecio
    assert state.anchor_dx == pytest.approx(12.0)
    assert state.anchor_dy == pytest.approx(-5.0)

    state.x_hat, state.y_hat = 3.0, 1.0
    rechain_keyframe(state, signature=work2, now_t=2.0)
    assert state.anchor_dx == pytest.approx(15.0)
    assert state.anchor_dy == pytest.approx(-4.0)


def test_an_explicit_new_target_does_drop_the_anchor() -> None:
    """Un GoTo o un movimiento manual si cambian el objetivo.

    Conservar el ancla ahi mandaria la montura de vuelta al campo anterior.
    """
    state = make_tracking_state()
    anchor, work = object(), object()
    reset_keyframe(state, signature=anchor, now_t=0.0)
    state.x_hat, state.y_hat = 40.0, 40.0
    rechain_keyframe(state, signature=work, now_t=1.0)
    assert state.anchor_dx != 0.0

    reset_tracker(state, now_t=2.0)               # sin keep_anchor
    assert state.anchor_signature is None
    assert state.anchor_dx == 0.0 and state.anchor_dy == 0.0
    assert not state.anchor_chained


def test_losing_the_enganche_keeps_the_anchor() -> None:
    state = make_tracking_state()
    reset_keyframe(state, signature=object(), now_t=0.0)
    state.x_hat, state.y_hat = 7.0, 9.0

    reset_tracker(state, now_t=1.0, keep_anchor=True)
    assert state.anchor_signature is not None
    # la distancia recorrida hasta la perdida es la que hay que deshacer
    assert state.anchor_dx == pytest.approx(7.0)
    assert state.anchor_dy == pytest.approx(9.0)
    assert state.anchor_chained


def test_anchor_is_abandoned_once_it_leaves_the_field() -> None:
    """Perseguir un ancla fuera de campo seria comandar un salto a ciegas.

    El offset encadenado es una suma de estimaciones que ya nadie puede
    verificar; pasado cierto punto lo honesto es decir que se perdio.
    """
    run = _Run()
    run.feed(6)
    state = run.state
    state.anchor_chained = True
    state.anchor_next_t = None
    state.anchor_dx = float(state.cfg.keyframe.anchor_max_offset_px) * 2.0

    out = run.feed(1)
    assert out.anchor_lost
    # y con el ancla perdida, el error reportado vuelve a ser el local
    assert _err(out) < float(state.cfg.keyframe.anchor_max_offset_px)


def test_the_anchor_search_backs_off_when_it_cannot_be_found() -> None:
    """Insistir cada 3 s contra una referencia irrecuperable solo quema CPU."""
    run = _Run()
    run.feed(6)
    state = run.state
    state.anchor_chained = True
    state.anchor_next_t = None
    # ancla imposible de casar con el campo actual
    state.anchor_signature = run.state.key_signature
    state.anchor_dx, state.anchor_dy = 900.0, 900.0

    run.feed(1)
    first = float(state.anchor_backoff_s)
    for _ in range(4):
        state.anchor_next_t = 0.0
        run.feed(1)
    assert float(state.anchor_backoff_s) > first
    assert float(state.anchor_backoff_s) <= float(
        state.cfg.keyframe.anchor_recover_max_s
    )
