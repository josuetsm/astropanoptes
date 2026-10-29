"""El apilado sin tracking: un campo que deriva bajo un sensor con defectos.

Lo que fallo en el cielo el 2026-10-28, en un campo con pocas estrellas
debiles: un pixel caliente a 31856 ADU, fijo en el sensor, gano la correlacion
y el apilador midio ~0 de movimiento mientras el cielo derivaba 1.4 px por
cuadro; despues acepto un salto falso de -43 px y, centrando cada busqueda en
el, rechazo todo lo que vino (574 de 766 cuadros).
"""
from __future__ import annotations

import numpy as np

from sep_utils import repair_isolated_pixels
from stacking import LiveMosaicStackerGray, _bayer_to_gray_code, _bayer_to_rgb_code

H, W = 240, 320
HOT = (121, 161)
DRIFT = (1.4, 1.6)  # px por cuadro, como la deriva sideral medida esa noche


def _sky_field(seed: int = 3, n_stars: int = 9):
    rng = np.random.default_rng(seed)
    # Estrellas en un campo mas grande que el cuadro: el cielo entra por un
    # borde a medida que deriva.
    stars = [
        (float(rng.uniform(-60, W + 60)), float(rng.uniform(-60, H + 60)), float(rng.uniform(900, 2200)))
        for _ in range(n_stars)
    ]
    return stars


def _frame(stars, k: int, *, hot: bool = True, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed + k)
    yy, xx = np.mgrid[:H, :W]
    img = np.full((H, W), 120.0)
    for x, y, peak in stars:
        cx, cy = x + DRIFT[0] * k, y + DRIFT[1] * k
        img += peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6**2))
    img += rng.normal(0.0, 25.0, img.shape)
    if hot:
        img[HOT] = 31856.0
    return np.clip(img, 0, 65535).astype(np.uint16)


def _stacker(**kw) -> LiveMosaicStackerGray:
    return LiveMosaicStackerGray(
        color_mode="mono",
        resp_min=0.25,
        align_median_k=3,
        smooth_k=30,
        max_shift_px=50,
        use_subpixel=True,
        preview_log_vmin=0.0,
        drizzle_scale=1.0,
        bayer_to_gray_code=_bayer_to_gray_code("RGGB"),
        bayer_to_rgb_code=_bayer_to_rgb_code("RGGB"),
        **kw,
    )


def _measured_drift(stacker, frames) -> tuple[float, float]:
    track = []
    for k, raw in enumerate(frames):
        if stacker.add_frame(raw, t_unix=0.1 * k):
            # pos_x esta en coordenadas del lienzo, que se corren cuando el
            # mosaico crece por la izquierda; el origen del cuadro de
            # referencia se corre con ellas.
            track.append((k, stacker.pos_x - stacker.ref_origin_x, stacker.pos_y - stacker.ref_origin_y))
    track = np.array(track)
    vx = np.polyfit(track[:, 0], track[:, 1], 1)[0]
    vy = np.polyfit(track[:, 0], track[:, 2], 1)[0]
    return -vx, -vy


# ----------------------------------------------------------- pixeles calientes


def test_an_isolated_hot_pixel_is_repaired_and_a_star_is_not() -> None:
    stars = [(100.0, 80.0, 2000.0)]
    raw = _frame(stars, 0)

    out = repair_isolated_pixels(raw)

    assert out[HOT] < 400
    assert abs(int(out[80, 100]) - int(raw[80, 100])) == 0


def test_an_undersampled_star_survives_the_repair() -> None:
    """Una estrella de 1.5" en pixeles de 0.66" sigue iluminando a sus vecinos."""
    yy, xx = np.mgrid[:H, :W]
    img = 120.0 + 20000.0 * np.exp(-((xx - 50.3) ** 2 + (yy - 40.6) ** 2) / (2 * 0.95**2))
    raw = np.clip(img + np.random.default_rng(1).normal(0, 25, img.shape), 0, 65535).astype(np.uint16)

    out = repair_isolated_pixels(raw)

    assert out[38:44, 47:54].sum() == raw[38:44, 47:54].sum()


def test_the_stack_follows_the_sky_not_the_hot_pixel() -> None:
    stars = _sky_field()
    frames = [repair_isolated_pixels(_frame(stars, k)) for k in range(40)]

    vx, vy = _measured_drift(_stacker(), frames)

    assert abs(vx - DRIFT[0]) < 0.15 and abs(vy - DRIFT[1]) < 0.15


# ------------------------------------------------------------ modelo de movimiento


def test_a_false_jump_is_rejected_instead_of_bending_the_mosaic() -> None:
    stars = _sky_field()
    stacker = _stacker()
    for k in range(20):
        assert stacker.add_frame(_frame(stars, k, hot=False), t_unix=0.1 * k) or k > 0
    # Un cuadro de otro campo (o un pico falso): no puede mover el mosaico.
    alien = _frame(_sky_field(seed=99), 0, hot=False)
    pos = (stacker.pos_x, stacker.pos_y)

    accepted = stacker.add_frame(alien, t_unix=2.0)

    assert not accepted
    assert (stacker.pos_x, stacker.pos_y) == pos
    # Y el siguiente cuadro bueno vuelve a entrar.
    assert stacker.add_frame(_frame(stars, 21, hot=False), t_unix=2.1)


def test_the_search_follows_time_after_a_run_of_rejections() -> None:
    """Antes la busqueda se centraba en el ultimo desplazamiento por cuadro y
    no sabia cuanto tiempo habia pasado."""
    stars = _sky_field()
    stacker = _stacker()
    for k in range(15):
        stacker.add_frame(_frame(stars, k, hot=False), t_unix=0.1 * k)

    # 1.2 s sin cuadros (nube, cuadros descartados): el cielo avanzo 12 cuadros.
    assert stacker.add_frame(_frame(stars, 27, hot=False), t_unix=2.7)
    assert abs(stacker.last_dx - 13 * DRIFT[0]) < 1.0


def test_after_a_cloud_the_stack_picks_up_where_the_sky_went() -> None:
    stars = _sky_field()
    stacker = _stacker(lost_after_s=1.0)
    for k in range(15):
        stacker.add_frame(_frame(stars, k, hot=False), t_unix=0.1 * k)
    rng = np.random.default_rng(5)
    for k in range(15, 40):  # 2.5 s de nube: el cielo avanza 35 px
        cloud = rng.normal(120, 25, (H, W)).clip(0, 65535).astype(np.uint16)
        assert not stacker.add_frame(cloud, t_unix=0.1 * k)

    assert stacker.add_frame(_frame(stars, 40, hot=False), t_unix=4.0)
    assert stacker.restarts == 0
    assert abs(stacker.last_dx - 26 * DRIFT[0]) < 1.5


def test_a_moved_mount_restarts_the_stack_instead_of_rejecting_forever() -> None:
    """La noche del 2026-10-28 el apilado rechazo 574 de 766 cuadros seguidos."""
    stars = _sky_field()
    stacker = _stacker(lost_after_s=1.0)
    for k in range(15):
        stacker.add_frame(_frame(stars, k, hot=False), t_unix=0.1 * k)
    elsewhere = _sky_field(seed=42)

    results = [stacker.add_frame(_frame(elsewhere, k, hot=False), t_unix=0.1 * k) for k in range(15, 50)]

    assert stacker.restarts == 1
    assert all(results[-8:])
