# goto_math.py
from __future__ import annotations

import math
from typing import Sequence, Optional

import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from platesolve import ObserverConfig


def wrap_deg_180(x: float) -> float:
    """Wrap degrees to (-180, 180]."""
    y = (float(x) + 180.0) % 360.0 - 180.0
    if y <= -180.0:
        y += 360.0
    return float(y)


def wrap_deg_360(x: float) -> float:
    """Wrap degrees to [0, 360)."""
    y = float(x) % 360.0
    if y < 0.0:
        y += 360.0
    return float(y)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def norm2(a: np.ndarray) -> float:
    return float(np.sqrt(float(np.sum(a * a))))


def as_array2(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError("expected a 2-vector")
    return a


def now_time() -> Time:
    # astropy Time uses UTC by default
    return Time.now()


def icrs_to_altaz_deg(
    coord_icrs: SkyCoord,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    if obstime is None:
        obstime = now_time()
    loc = observer.location()
    altaz = coord_icrs.transform_to(AltAz(obstime=obstime, location=loc))
    az = wrap_deg_360(float(altaz.az.deg))
    alt = float(altaz.alt.deg)
    return np.array([az, alt], dtype=np.float64)


def platesolve_center_to_altaz_deg(
    ra_deg: float,
    dec_deg: float,
    *,
    observer: ObserverConfig,
    obstime: Optional[Time] = None,
) -> np.ndarray:
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    return icrs_to_altaz_deg(c, observer=observer, obstime=obstime)
