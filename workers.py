from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class BaseWorker:
    """
    Base worker with coalescing request queue ("last request wins").

    Subclasses should implement _handle_request(request: Dict[str, Any]).
    """

    def __init__(self, *, name: str, idle_sleep_s: float = 0.05) -> None:
        self._name = str(name)
        self._idle_sleep_s = float(idle_sleep_s)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._pending: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._busy = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread.start()

    def request(self, **payload: Any) -> None:
        with self._lock:
            self._pending = dict(payload)
        self.start()

    def stop(self) -> None:
        self._cancel.set()

    def join(self, timeout: Optional[float] = 1.0) -> None:
        with self._lock:
            thr = self._thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=timeout)

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._busy)

    def _set_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = bool(busy)

    def _run(self) -> None:
        while not self._cancel.is_set():
            with self._lock:
                req = self._pending
                self._pending = None

            if req is None:
                time.sleep(self._idle_sleep_s)
                continue

            self._set_busy(True)
            try:
                self._handle_request(req)
            finally:
                self._set_busy(False)

    def _handle_request(self, request: Dict[str, Any]) -> None:
        raise NotImplementedError
