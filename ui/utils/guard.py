from __future__ import annotations

from contextlib import contextmanager


class RenderGuard:
    def __init__(self) -> None:
        self._depth = 0

    @property
    def active(self) -> bool:
        return self._depth > 0

    @contextmanager
    def hold(self):
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
