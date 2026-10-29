"""Como se comporta astropy dentro de la app: sin red y sin avisos de mas.

La primera conversion de coordenadas de la noche tardaba 43 s. Astropy necesita
la tabla IERS de rotacion terrestre (UT1-UTC y movimiento del polo) y, si la que
tiene en cache pasa de 30 dias, la descarga antes de contestar. En el telescopio
eso es una espera; sin internet, un timeout entero, y lo pagaba el primer plate
solve de cada sesion. En demo no se veia porque la simulacion ya lo desactivaba.

Sin descarga, astropy usa la ultima tabla que tenga en cache -- por vieja que sea
-- y extrapola. El error que eso introduce es de arcosegundos: UT1-UTC predicho
con meses de antiguedad yerra bastante menos de un segundo de tiempo (15
arcsec), y el movimiento del polo medio, unas decimas. Un plate solve que busca
en grados y un goto que acierta al arcominuto no lo notan.

Los avisos de precision que astropy emite en ese modo, y los de ERFA sobre "anos
dudosos" mas alla de la tabla de segundos intercalares, salian por stderr en
crudo en medio de los eventos de la consola. Describen exactamente el compromiso
de arriba, asi que se silencian aqui una vez en vez de repetirse en cada solve.
"""
from __future__ import annotations

import threading
import warnings

from logging_utils import log_info

__all__ = ["configure_astropy"]

_LOCK = threading.Lock()
_DONE = False


def configure_astropy(out_log=None) -> None:
    """Fija la politica de astropy para todo el proceso. Idempotente."""
    global _DONE
    with _LOCK:
        if _DONE:
            return
        from astropy.utils import iers

        iers.conf.auto_download = False
        iers.conf.auto_max_age = None

        # Avisos que solo repiten el compromiso documentado arriba.
        warnings.filterwarnings("ignore", message=r".*polar motion.*")
        warnings.filterwarnings("ignore", message=r".*IERS.*")
        warnings.filterwarnings("ignore", message=r".*dubious year.*")
        try:
            from erfa import ErfaWarning

            warnings.filterwarnings("ignore", category=ErfaWarning)
        except ImportError:  # pragma: no cover - erfa viene con astropy
            pass
        _DONE = True
    log_info(
        out_log,
        "App: astropy sin descargas de IERS (precisión de arcosegundos, sin esperas de red)",
    )
