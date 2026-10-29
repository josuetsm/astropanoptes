"""Como se ve todo lo que la consola escribe.

La consola tiene dos lectores: una persona en el telescopio y un agente que la
maneja por script. Los dos necesitan lo mismo -- saber que comando produjo cada
respuesta, si salio bien, y por que no cuando no -- asi que hay una sola
gramatica para los dos.

Por stdout va la conversacion:

    › mount connect                     eco del comando (solo en scripts)
    ok  montura conectada               una linea de resultado: `ok` o `error`
      puerto    AUTO                    y, si hace falta, detalle indentado
      baudios   115200

Por stderr van los eventos, que llegan cuando llegan:

    21:54:06.925  info   goto      diagnostico guardado en stack_output/...
    21:54:06.931  error  goto      la montura no esta sincronizada  codigo=not_synced

Todo en minusculas. Los codigos de estado se guardan en mayusculas en el estado
de la app -- los usan la interfaz, los diagnosticos y `--json` -- y solo aqui se
presentan en minusculas; los datos no cambian.

La convencion de idioma es la que la consola ya seguia: comandos y claves en
ingles, texto en castellano.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "SUBSYSTEMS",
    "code",
    "explain",
    "reason",
    "state_word",
    "render_value",
    "render_tree",
    "result_lines",
    "event_line",
    "split_subsystem",
]

# ----------------------------------------------------------------- subsistemas

# El prefijo "Algo: " de los mensajes de log se traduce a una etiqueta corta y
# fija, la misma que usa la tabla de `status`. Sin esto la columna mostraba el
# nombre del *hilo* (AppRunner, GoToWorker, MainThread), que no dice nada.
SUBSYSTEMS = {
    "camera": "camara",
    "camerastream": "camara",
    "camara": "camara",
    "raw record": "camara",
    "mount": "montura",
    "montura": "montura",
    "tracking": "tracking",
    "transmission": "tracking",
    "stacking": "apilado",
    "apilado": "apilado",
    "savework": "apilado",
    "saveworker": "apilado",
    "live sep": "apilado",
    "preview": "apilado",
    "platesolving": "solve",
    "solve": "solve",
    "expected stars overlay": "solve",
    "gaia download": "gaia",
    "gaia cache": "gaia",
    "gaia coverage": "gaia",
    "gaia healpix": "gaia",
    "gaia": "gaia",
    "goto": "goto",
    "goto fit": "goto",
    "focuser": "foco",
    "focus": "foco",
    "foco": "foco",
    "simulation": "sim",
    "simulation camera": "sim",
    "sim": "sim",
    "diagnostics": "diag",
    "diag": "diag",
    "runner": "app",
    "manual": "montura",
    "settings": "app",
    "observer": "app",
    "top": "app",
    "ui": "app",
    "action failed": "app",
    "app": "app",
    "control": "consola",
    "consola": "consola",
}

_PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z ]{0,30}?)\s*:\s+(.*)$", re.DOTALL)
_SUBSYSTEM_WIDTH = 8


def split_subsystem(message: str) -> tuple[str, str]:
    """Separa "Camera: conectada" en ("camara", "conectada").

    Solo se reconoce un prefijo si es un subsistema conocido: "error: x" o
    "ratio: 3" no son subsistemas y el mensaje se deja entero.
    """
    match = _PREFIX.match(message)
    if match:
        key = match.group(1).strip().lower()
        if key in SUBSYSTEMS:
            return SUBSYSTEMS[key], match.group(2).strip()
    return "app", message.strip()


# ------------------------------------------------------------------- codigos

# Valores de un par clave=valor que se presentan en minusculas: codigos guardados
# en mayusculas (ERR_NOT_SYNCED, OK_MANUAL_SAMPLE), los literales de Python
# (True/False/None) y la palabra suelta de un estado (estado=OK). Solo el lado del
# valor: asi no se toca un nombre de variable de entorno citado en un mensaje.
_KV_CODE = re.compile(
    r"(\b[a-z_][a-zñáéíóú0-9_]*=)"
    r"([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|True|False|None|OK|FAIL|ERROR|IDLE|RUNNING|OFF|ON)\b"
)
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


def code(value: Any) -> str:
    """Un codigo de estado tal como se presenta: en minusculas."""
    if value is None:
        return "null"
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().lower()


_CODE_SHAPE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")


def reason(value: Any) -> str:
    """Un motivo de fallo: si es un codigo va en minusculas, si es una frase va tal cual.

    Los motivos mezclan codigos (ERR_NOT_SYNCED) y mensajes con rutas y nombres
    propios ("sin respuesta a PING en /dev/cu.AstroPanoptes-ESP32"); pasar estos
    ultimos a minusculas dejaba un puerto que no existe.
    """
    if value is None:
        return ""
    if isinstance(value, Enum):
        value = value.value
    text = str(value).strip()
    return text.lower() if _CODE_SHAPE.fullmatch(text) else text


# Que significa cada codigo y que hacer. Son los que aparecieron de verdad en 255
# sesiones de diagnostico; un codigo sin entrada se muestra igual, solo que sin
# explicacion.
_EXPLANATIONS = {
    "err_not_synced": "la montura no está sincronizada: resuelve un campo y haz `mount sync`",
    "not_synced": "la montura no está sincronizada: resuelve un campo y haz `mount sync`",
    "err_model_not_fitted_for_distance": (
        "el modelo no está ajustado y el destino queda demasiado lejos para ir a ciegas"
    ),
    "err_goto_distance_limit": "el destino supera la distancia máxima de un goto",
    "autocal_err_sample_continuity": (
        "el campo resuelto no cuadra con los pasos movidos: o el solve es falso o el "
        "contador de pasos se desvió"
    ),
    "err_sample_continuity": (
        "el campo resuelto no cuadra con los pasos movidos: o el solve es falso o el "
        "contador de pasos se desvió"
    ),
    "motion_continuity_mismatch": (
        "el campo resuelto no cuadra con los pasos movidos: o el solve es falso o el "
        "contador de pasos se desvió"
    ),
    "autocal_err_platesolving": "el plate solve no encontró solución",
    "err_platesolving": "el plate solve no encontró solución",
    "low_inliers": "el solve casa muy pocas estrellas para creérselo",
    "gaia_cache_miss": "faltan teselas de Gaia en la caché local para esa zona",
    "gaia_too_small": "el catálogo cargado es demasiado pequeño para el campo",
    "no_triplet_candidates": "ningún triángulo de la imagen casa con el catálogo",
    "not_enough_seeds": "la imagen tiene muy pocas estrellas para intentar el solve",
    "need_gaia_auth": "hace falta autenticarse en el archivo de Gaia para descargar",
    "fit_model_err_model_outside_mechanical_limits": (
        "el ajuste cae fuera de lo que la mecánica permite: muestras malas o movimientos "
        "de calibración demasiado cortos"
    ),
    "fit_model_err_fit_rms_too_high": "el ajuste no explica las muestras: rms demasiado alto",
    "fit_model_err_degenerate_model": "las muestras no separan los términos del modelo",
    "initial_consensus_mismatch": "dos solves seguidos no coinciden entre sí",
    "initial_consensus_no_new_frame": "no llegó un cuadro nuevo para confirmar el solve",
    "roll_err_drift_delta_small": "la deriva medida es demasiado corta para estimar el giro",
    "timeout": "se agotó el tiempo de espera",
    "cancelled": "cancelado",
}


def explain(value: Any) -> str:
    """La explicacion de un codigo, o cadena vacia si no se conoce."""
    return _EXPLANATIONS.get(code(value), "")


# Palabras de estado, por subsistema. El genero concuerda con el sustantivo de la
# fila: "camara conectada", "tracking apagado".
_STATE_WORDS = {
    "camara": {
        "disconnected": "desconectada", "ok": "conectada",
        "error": "error", "connecting": "conectando",
    },
    "montura": {
        "disconnected": "desconectada", "ok": "conectada",
        "error": "error", "connecting": "conectando",
    },
    "tracking": {"off": "apagado", "running": "siguiendo", "paused": "en pausa", "error": "error"},
    "apilado": {"off": "apagado", "running": "apilando", "saving": "guardando", "error": "error"},
    "solve": {"idle": "inactivo", "running": "resolviendo", "ok": "resuelto", "fail": "falló"},
    "goto": {
        "idle": "inactivo", "running": "en curso", "ok": "ok",
        "fail": "falló", "cancelled": "cancelado",
    },
}


def state_word(subsystem: str, value: Any) -> str:
    """El estado de un subsistema en palabras: OK de la camara es "conectada"."""
    raw = code(value)
    return _STATE_WORDS.get(subsystem, {}).get(raw, raw)


# -------------------------------------------------------------------- valores


def render_value(value: Any) -> str:
    """Un valor escalar en texto, sin ruido.

    Los datos siguen siendo datos: true/false/null, numeros tal cual. Solo los
    flotantes se recortan a seis cifras significativas -- una latencia de
    3.098208999290364 ms es 3.09821 ms, y las cifras de mas eran ruido.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return code(value)
    if isinstance(value, float):
        return _render_float(value)
    if isinstance(value, Path):
        return str(value)
    # numpy escalares sin importar numpy
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return render_value(value.item())
    if isinstance(value, (list, tuple)) and all(_is_scalar(v) for v in value):
        return "[" + ", ".join(render_value(v) for v in value) + "]"
    return str(value)


def _render_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    # Marcas de tiempo y otros numeros grandes: sin notacion cientifica.
    if abs(value) >= 1e6:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text or "0"
    text = f"{value:.6g}"
    return "0" if text in {"-0", "0"} else text


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, Enum, Path))


def render_tree(data: Any, *, indent: int = 2, max_key: int = 28) -> List[str]:
    """Un diccionario anidado como bloque clave/valor alineado, sin llaves ni comillas.

    Cada nivel alinea sus propias claves. Es lo mismo que el JSON indentado de
    antes, en menos de la mitad de lineas y sin la puntuacion que no aporta.
    """
    lines: List[str] = []
    pad = " " * indent
    if isinstance(data, Mapping):
        if not data:
            return [pad + "(vacío)"]
        # Las cabeceras de seccion van en su propia linea, asi que no cuentan
        # para el ancho: solo se alinean las claves que llevan valor al lado.
        leaves = [
            str(k) for k, v in data.items()
            if not (isinstance(v, Mapping) and v)
            and not (isinstance(v, (list, tuple)) and v and not all(_is_scalar(x) for x in v))
        ]
        width = min(max_key, max((len(k) for k in leaves), default=0))
        for key, value in data.items():
            name = str(key)
            if isinstance(value, Mapping) and value:
                lines.append(f"{pad}{name}")
                lines.extend(render_tree(value, indent=indent + 2, max_key=max_key))
            elif isinstance(value, (list, tuple)) and value and not all(_is_scalar(v) for v in value):
                lines.append(f"{pad}{name}")
                for item in value:
                    sub = render_tree(item, indent=indent + 4, max_key=max_key)
                    if sub:
                        sub[0] = " " * (indent + 2) + "- " + sub[0].lstrip()
                    lines.extend(sub)
            else:
                lines.append(f"{pad}{name.ljust(width)}  {render_value(value)}".rstrip())
        return lines
    if isinstance(data, (list, tuple)):
        for item in data:
            if _is_scalar(item):
                lines.append(f"{pad}- {render_value(item)}")
            else:
                sub = render_tree(item, indent=indent + 2, max_key=max_key)
                if sub:
                    sub[0] = pad + "- " + sub[0].lstrip()
                lines.extend(sub)
        return lines
    return [pad + render_value(data)]


def result_lines(
    ok: bool,
    summary: str = "",
    detail: Optional[Any] = None,
) -> List[str]:
    """La respuesta a un comando: una linea `ok`/`error` y detalle indentado."""
    head = "ok" if ok else "error"
    first = f"{head}  {summary}" if summary else head
    lines = [first.rstrip()]
    if detail is not None:
        if isinstance(detail, str):
            lines.extend("  " + line if line else "" for line in detail.splitlines())
        elif isinstance(detail, (Mapping, list, tuple)):
            lines.extend(render_tree(detail))
        else:
            lines.append("  " + render_value(detail))
    return lines


# -------------------------------------------------------------------- eventos

_LEVELS = {"debug": "debug", "info": "info", "warning": "aviso", "warn": "aviso", "error": "error"}


def event_line(level: str, message: str, *, when: Optional[_dt.datetime] = None) -> str:
    """Una linea de evento: hora local, nivel, subsistema y mensaje.

    La hora es de reloj, no tiempo desde el arranque: las carpetas de
    diagnostico y los ficheros de stack llevan hora local, y asi se cruzan.
    """
    now = when or _dt.datetime.now()
    stamp = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    subsystem, text = split_subsystem(message)
    text = _KV_CODE.sub(lambda m: m.group(1) + _PY_LITERALS.get(m.group(2), m.group(2).lower()), text)
    lvl = _LEVELS.get(level.lower(), level.lower())
    return f"{stamp}  {lvl:<5}  {subsystem:<{_SUBSYSTEM_WIDTH}}  {text}"


def indent_block(lines: Iterable[str], spaces: int = 2) -> List[str]:
    pad = " " * spaces
    return [pad + line if line else line for line in lines]
