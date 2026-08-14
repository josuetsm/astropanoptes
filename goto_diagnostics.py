from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from logging_utils import log_error, log_info


def _json_safe(value: Any) -> Any:
    """Convert runtime/scientific values into lossless-enough JSON metadata."""
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "isot"):
        try:
            return str(value.isot)
        except Exception:
            pass
    return repr(value)


def _array_sha256(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def raw16_statistics(raw16: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(raw16)
    out: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "c_contiguous": bool(arr.flags.c_contiguous),
        "sha256": _array_sha256(arr),
    }
    if arr.size <= 0:
        return out
    flat = arr.reshape(-1)
    percentiles = np.percentile(flat, [0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0])
    out.update(
        {
            "min": float(percentiles[0]),
            "p01": float(percentiles[1]),
            "p05": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p95": float(percentiles[4]),
            "p99": float(percentiles[5]),
            "max": float(percentiles[6]),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        }
    )
    if np.issubdtype(arr.dtype, np.integer):
        max_value = int(np.iinfo(arr.dtype).max)
        out["saturated_fraction"] = float(np.mean(flat == max_value))
        out["zero_fraction"] = float(np.mean(flat == 0))
    return out


class DiagnosticSession:
    """Best-effort, self-contained artifact bundle for one optical operation.

    Diagnostic writes must never change the result of plate solving or a GoTo.
    The timeline is append-only so it remains useful even after a process crash.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        operation: str,
        enabled: bool,
        context: Optional[Dict[str, Any]] = None,
        out_log: Any = None,
    ) -> None:
        env = str(os.environ.get("ASTROPANOPTES_DIAGNOSTICS", "")).strip().lower()
        if env in {"0", "false", "no", "off"}:
            enabled = False
        elif env in {"1", "true", "yes", "on"}:
            enabled = True

        self.enabled = bool(enabled)
        self.operation = str(operation).strip().lower() or "operation"
        self.out_log = out_log
        self._lock = threading.Lock()
        self._event_index = 0
        self._artifact_index = 0
        self._events: list[Dict[str, Any]] = []
        self._artifacts: list[Dict[str, Any]] = []
        self._started_unix = float(time.time())
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(self._started_unix))
        self.session_id = f"{stamp}_{self.operation}_{uuid.uuid4().hex[:8]}"
        self.path = Path(root_dir).expanduser() / self.session_id
        self._timeline_path = self.path / "timeline.jsonl"

        if not self.enabled:
            return
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            self._write_json(
                self.path / "session.json",
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "operation": self.operation,
                    "started_unix": self._started_unix,
                    "started_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._started_unix)
                    ),
                    "context": dict(context or {}),
                },
            )
            self.record("session_started", context=dict(context or {}))
            log_info(self.out_log, f"Diagnostics: session={self.path}")
        except Exception as exc:
            self.enabled = False
            log_error(self.out_log, "Diagnostics: failed to create session", exc)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @property
    def path_str(self) -> Optional[str]:
        return str(self.path.resolve()) if self.enabled else None

    def record(self, stage: str, **payload: Any) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self._event_index += 1
                event = {
                    "index": int(self._event_index),
                    "ts_unix": float(time.time()),
                    "elapsed_s": float(time.time() - self._started_unix),
                    "stage": str(stage),
                    **payload,
                }
                safe = _json_safe(event)
                with self._timeline_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
                self._events.append(safe)
        except Exception as exc:
            log_error(
                self.out_log,
                f"Diagnostics: failed to record stage={stage}",
                exc,
                throttle_s=2.0,
                throttle_key="diagnostics_record",
            )

    def save_raw(
        self,
        stage: str,
        raw16: np.ndarray,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            arr = np.ascontiguousarray(raw16)
            with self._lock:
                self._artifact_index += 1
                stem = f"{self._artifact_index:04d}_{str(stage)}"
                path = self.path / f"{stem}_raw16.npy"
                np.save(path, arr, allow_pickle=False)
                artifact = {
                    "kind": "raw16",
                    "stage": str(stage),
                    "path": path.name,
                    "bytes": int(path.stat().st_size),
                    "frame": raw16_statistics(arr),
                    "metadata": dict(metadata or {}),
                }
                self._artifacts.append(_json_safe(artifact))
                self._write_json(self.path / f"{stem}_metadata.json", artifact)
            self.record("raw_saved", artifact=artifact)
            return str(path.resolve())
        except Exception as exc:
            log_error(self.out_log, f"Diagnostics: failed to save raw stage={stage}", exc)
            return None

    def save_raw_stack(
        self,
        stage: str,
        frames: Sequence[np.ndarray],
        *,
        frame_metadata: Optional[Sequence[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.enabled or not frames:
            return None
        try:
            stack = np.stack([np.asarray(frame) for frame in frames], axis=0)
            stack = np.ascontiguousarray(stack)
            with self._lock:
                self._artifact_index += 1
                stem = f"{self._artifact_index:04d}_{str(stage)}"
                path = self.path / f"{stem}_raw16_stack.npz"
                # npz compression is exact and materially reduces large dark-sky stacks.
                np.savez_compressed(path, raw16=stack)
                per_frame = list(frame_metadata or [{} for _ in frames])
                artifact = {
                    "kind": "raw16_stack",
                    "stage": str(stage),
                    "path": path.name,
                    "bytes": int(path.stat().st_size),
                    "stack_shape": list(stack.shape),
                    "dtype": str(stack.dtype),
                    "frame_sha256": [_array_sha256(stack[i]) for i in range(stack.shape[0])],
                    "frame_metadata": per_frame,
                    "metadata": dict(metadata or {}),
                }
                self._artifacts.append(_json_safe(artifact))
                self._write_json(self.path / f"{stem}_metadata.json", artifact)
            self.record("raw_stack_saved", artifact=artifact)
            return str(path.resolve())
        except Exception as exc:
            log_error(self.out_log, f"Diagnostics: failed to save raw stack stage={stage}", exc)
            return None

    def close(self, status: str, **summary: Any) -> None:
        if not self.enabled:
            return
        try:
            self.record("session_finished", status=str(status), summary=summary)
            finished = float(time.time())
            manifest = {
                "schema_version": 1,
                "session_id": self.session_id,
                "operation": self.operation,
                "status": str(status),
                "started_unix": self._started_unix,
                "finished_unix": finished,
                "duration_s": float(finished - self._started_unix),
                "summary": summary,
                "artifacts": self._artifacts,
                "events": self._events,
            }
            self._write_json(self.path / "manifest.json", manifest)
            log_info(self.out_log, f"Diagnostics: saved session={self.path}")
        except Exception as exc:
            log_error(self.out_log, "Diagnostics: failed to finalize session", exc)


__all__ = ["DiagnosticSession", "raw16_statistics"]
