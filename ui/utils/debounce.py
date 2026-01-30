from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional


@dataclass
class DebouncedCall:
    delay_s: float
    _pending: Optional[Callable[[], None]] = None
    _last_t: Optional[float] = None

    def trigger(self, action: Callable[[], None]) -> None:
        self._pending = action
        self._last_t = time.monotonic()

    def maybe_fire(self, now: float) -> None:
        if self._pending is None or self._last_t is None:
            return
        if now - self._last_t < self.delay_s:
            return
        action = self._pending
        self._pending = None
        self._last_t = None
        action()
