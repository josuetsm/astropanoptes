"""Ajustes de la GUI que sobreviven entre sesiones.

Hasta ahora cada arranque volvía a los valores de ``config.py``, así que había
que reponer sitio, óptica y cámara todas las noches. Esto guarda un puñado de
ajustes al cerrar y los repone al abrir.

Es deliberadamente pequeño y tolerante: un archivo corrupto, de otra versión o
con claves desconocidas nunca debe impedir que la app abra, así que cualquier
problema de lectura devuelve ``{}`` y la sesión sigue con los defaults.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path.home() / ".astropanoptes" / "settings.json"
SETTINGS_VERSION = 1


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Lee los ajustes guardados. Devuelve ``{}`` si no hay o no se pueden usar."""
    try:
        raw = Path(path).expanduser().read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return {}

    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}

    if not isinstance(data, dict):
        return {}
    if int(data.get("version", 0)) != SETTINGS_VERSION:
        # Formato de otra versión: se ignora en vez de intentar migrarlo.
        return {}
    return data


def save_settings(data: dict[str, Any], path: Path | str = DEFAULT_SETTINGS_PATH) -> bool:
    """Guarda los ajustes. Devuelve False si no se pudo, sin propagar el error."""
    target = Path(path).expanduser()
    payload = dict(data)
    payload["version"] = SETTINGS_VERSION
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atómica: un corte a mitad no deja un JSON truncado.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(target)
        return True
    except (OSError, TypeError, ValueError):
        return False


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Devuelve una sección del archivo, o ``{}`` si falta o no es un objeto."""
    value = data.get(name)
    return value if isinstance(value, dict) else {}
