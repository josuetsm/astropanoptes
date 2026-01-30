from __future__ import annotations

from typing import Any, Dict, Optional
import zlib

from ipywidgets import Widget

from ui.utils.guard import RenderGuard


def safe_set(widget: Widget, attr: str, value: Any, guard: Optional[RenderGuard]) -> None:
    current = getattr(widget, attr)
    if current == value:
        return
    if guard is None:
        setattr(widget, attr, value)
        return
    with guard.hold():
        setattr(widget, attr, value)


def set_html(widget: Widget, value: str, guard: Optional[RenderGuard]) -> None:
    safe_set(widget, "value", value, guard)


def set_image_bytes(
    widget: Widget,
    value: Optional[bytes],
    guard: Optional[RenderGuard],
    cache: Dict[str, int],
    cache_key: str,
) -> None:
    if not value:
        return
    checksum = zlib.adler32(value)
    if cache.get(cache_key) == checksum:
        return
    cache[cache_key] = checksum
    safe_set(widget, "value", value, guard)
