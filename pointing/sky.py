"""Conversiones entre el cielo y el horizonte, sin estado ni modelo.

Se separa del modelo de apuntado porque no sabe nada de el: convierte
coordenadas y calcula a que velocidad se mueve una estrella por el cielo en un
instante y un lugar. El modelo lo necesita para el feed-forward sideral, pero
tambien lo usaria cualquier otra cosa.

La altura es *verdadera*, sin refraccion. El modelo de apuntado trabaja en AltAz
verdadero y la refraccion se aplica -- si se aplica -- en el plate solving, que
es donde entra la atmosfera.
"""
from __future__ import annotations

from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

__all__ = [
    "wrap_deg_180",
    "wrap_deg_360",
    "now_time",
    "altaz_from_icrs_deg",
    "sidereal_world_rate_deg_s",
]


def wrap_deg_180(x: float) -> float:
    """A (-180, 180]."""
    y = (float(x) + 180.0) % 360.0 - 180.0
    if y <= -180.0:
        y += 360.0
    return float(y)


def wrap_deg_360(x: float) -> float:
    """A [0, 360)."""
    y = float(x) % 360.0
    return float(y + 360.0 if y < 0.0 else y)


def now_time() -> Time:
    return Time.now()


def altaz_from_icrs_deg(
    coord_icrs: SkyCoord,
    *,
    location,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    """AltAz verdadero (sin refraccion) de una posicion ICRS."""
    if obstime is None:
        obstime = now_time()
    altaz = coord_icrs.transform_to(AltAz(obstime=obstime, location=location))
    return np.array([wrap_deg_360(float(altaz.az.deg)), float(altaz.alt.deg)], dtype=np.float64)


def sidereal_world_rate_deg_s(
    *,
    az_deg: float,
    alt_deg: float,
    location,
    obstime: Optional[Time] = None,
    dt_s: float = 1.0,
) -> Optional[np.ndarray]:
    """A que velocidad se mueve por el horizonte lo que ahora esta en (az, alt).

    Se calcula por diferencia finita: se congela la posicion del cielo, se
    adelanta el reloj y se mira donde cae. Es lo bastante exacto para el
    feed-forward y evita derivar a mano la transformacion, que en alt-az tiene
    una singularidad en el cenit y es facil equivocarla.
    """
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 1e-6:
        return None

    az0 = wrap_deg_360(float(az_deg))
    alt0 = float(np.clip(float(alt_deg), -89.5, 89.5))
    if not np.isfinite(az0) or not np.isfinite(alt0):
        return None

    t0 = obstime if obstime is not None else now_time()
    frame = AltAz(obstime=t0, location=location)
    fixed = SkyCoord(az=az0 * u.deg, alt=alt0 * u.deg, frame=frame).icrs
    later = altaz_from_icrs_deg(fixed, location=location, obstime=t0 + dt * u.s)

    rate = np.array(
        [wrap_deg_180(float(later[0]) - az0) / dt, (float(later[1]) - alt0) / dt],
        dtype=np.float64,
    )
    return rate if np.all(np.isfinite(rate)) else None
