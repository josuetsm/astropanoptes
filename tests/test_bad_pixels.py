from __future__ import annotations

import numpy as np
import pytest

from sep_utils import (
    load_bad_pixel_map,
    repair_bad_pixels,
    sep_detect_from_raw16,
    set_bad_pixel_map,
)


def _sky(h=200, w=260, bg=1500, seed=4):
    rng = np.random.default_rng(seed)
    return (bg + rng.normal(0, 12, (h, w))).clip(0, 65535).astype(np.uint16)


def _add_star(img, y, x, peak=22000):
    yy, xx = np.mgrid[-4:5, -4:5]
    g = np.exp(-(yy**2 + xx**2) / 4.0) * peak
    img[y - 4:y + 5, x - 4:x + 5] = np.clip(
        img[y - 4:y + 5, x - 4:x + 5].astype(np.float64) + g, 0, 65535
    ).astype(np.uint16)
    return img


@pytest.fixture(autouse=True)
def _isolate_map():
    """Never let a test pick up the repository's real map, or leak its own."""
    set_bad_pixel_map(None)
    yield
    import sep_utils

    sep_utils._BAD_PIXEL_MAP = None
    sep_utils._BAD_PIXEL_MAP_TRIED = False


def test_repair_replaces_defect_with_local_median() -> None:
    img = _sky()
    img[100, 130] = 60000                     # pixel caliente aislado
    mask = np.zeros(img.shape, dtype=bool)
    mask[100, 130] = True

    fixed = repair_bad_pixels(img, mask)
    assert int(fixed[100, 130]) < 3000, "el pixel caliente sigue ahi"
    # el resto del frame no se toca
    untouched = ~mask
    np.testing.assert_array_equal(fixed[untouched], img[untouched])


def test_repair_preserves_real_stars() -> None:
    """La reparacion no debe comerse una estrella que no esta en el mapa."""
    img = _add_star(_sky(), 60, 80)
    before = int(img[60, 80])
    mask = np.zeros(img.shape, dtype=bool)
    mask[150, 200] = True                     # defecto en otro sitio
    fixed = repair_bad_pixels(img, mask)
    assert int(fixed[60, 80]) == before


def test_isolated_hot_pixel_is_already_handled_by_the_median_prefilter() -> None:
    """Un defecto de un solo pixel no llega al detector.

    ``sep_detect_from_raw16`` aplica una mediana 3x3 antes de extraer, y esa
    mediana borra por completo una muestra aislada. Documentarlo evita atribuir
    a la reparacion un merito que ya tenia el prefiltro.
    """
    img = _add_star(_sky(), 60, 80)
    img[140, 190] = 65000
    _d, _b, _o, xy = sep_detect_from_raw16(
        img, sep_bw=32, sep_bh=32, sep_thresh_sigma=3.0,
        sep_minarea=3, max_sources=50, repair_defects=False,
    )
    hit = len(xy) and bool(np.any(np.hypot(xy[:, 0] - 190, xy[:, 1] - 140) <= 4.0))
    assert not hit, "un pixel aislado no deberia sobrevivir a la mediana 3x3"


def test_hot_pixel_cluster_is_removed_by_the_repair() -> None:
    """Un racimo SI sobrevive a la mediana, y es el que enga\u00f1a al solver."""
    img = _add_star(_sky(), 60, 80)
    img[139:142, 189:192] = 65000             # defecto 3x3: la mediana no lo borra

    _d, _bkg, _obj, xy_raw = sep_detect_from_raw16(
        img, sep_bw=32, sep_bh=32, sep_thresh_sigma=3.0,
        sep_minarea=3, max_sources=50, repair_defects=False,
    )
    mask = np.zeros(img.shape, dtype=bool)
    mask[139:142, 189:192] = True
    set_bad_pixel_map(mask)
    _d, _bkg2, _obj2, xy_fix = sep_detect_from_raw16(
        img, sep_bw=32, sep_bh=32, sep_thresh_sigma=3.0,
        sep_minarea=3, max_sources=50, repair_defects=True,
    )

    def near(pts, x, y, r=4.0):
        if len(pts) == 0:
            return False
        return bool(np.any(np.hypot(pts[:, 0] - x, pts[:, 1] - y) <= r))

    assert near(xy_raw, 190, 140), "el defecto deberia detectarse sin reparar"
    assert not near(xy_fix, 190, 140), "el defecto sigue detectandose tras reparar"
    # y la estrella real sobrevive
    assert near(xy_fix, 80, 60), "se perdio la estrella real"


def test_mismatched_mask_is_ignored() -> None:
    """Un cambio de ROI o binning no debe corromper el frame."""
    img = _sky(h=100, w=120)
    mask = np.zeros((200, 260), dtype=bool)
    mask[10, 10] = True
    out = repair_bad_pixels(img, mask)
    np.testing.assert_array_equal(out, img)


def test_missing_map_disables_repair() -> None:
    img = _sky()
    assert load_bad_pixel_map("/no/existe.npy") is None
    np.testing.assert_array_equal(repair_bad_pixels(img, None), img)
