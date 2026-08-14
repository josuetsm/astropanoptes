from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

from logging_utils import log_error


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
            except Exception as exc:
                log_error(
                    None,
                    f"{self._name}: unhandled worker error",
                    exc,
                    throttle_s=2.0,
                    throttle_key=f"worker_unhandled_{self._name}",
                )
            finally:
                self._set_busy(False)

    def _handle_request(self, request: Dict[str, Any]) -> None:
        raise NotImplementedError


class SaveWorker:
    """FIFO worker for disk saves; unlike BaseWorker, requests are not coalesced."""

    def __init__(self, callback) -> None:
        self._callback = callback
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._busy = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="SaveWorker",
                daemon=True,
            )
            self._thread.start()

    def request(self, **payload: Any) -> None:
        self._queue.put(dict(payload))
        self.start()

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._busy or not self._queue.empty())

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        self._queue.put(None)
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                request = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if request is None:
                continue
            with self._lock:
                self._busy = True
            try:
                self._callback(request)
            except Exception as exc:
                log_error(
                    None,
                    "SaveWorker: unhandled save error",
                    exc,
                    throttle_s=2.0,
                    throttle_key="save_worker_unhandled",
                )
            finally:
                with self._lock:
                    self._busy = False
