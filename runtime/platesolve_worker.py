from __future__ import annotations

import datetime as _dt
import json
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from imaging import ensure_raw16_bayer
from logging_utils import log_info, log_error
from platesolve import PlatesolveConfig, ObserverConfig, platesolve_sweep


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:80] if s else "target"


def _summarize_target(target: Any, max_len: int = 80) -> str:
    text = str(target)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


class PlatesolveWorker:
    def __init__(self, runner: Any, out_log: Any) -> None:
        self._runner = runner
        self._out_log = out_log
        self._lock = threading.Lock()
        self._cfg_lock = threading.Lock()
        self._thr: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._pending: Optional[Dict[str, Any]] = None
        self._last_auto_t = 0.0
        self._auto_target: str = ""
        self._last_result: Optional[Any] = None

    @property
    def cfg_lock(self) -> threading.Lock:
        return self._cfg_lock

    def start_if_needed(self) -> None:
        started = False
        with self._lock:
            if self._thr is not None and self._thr.is_alive():
                return
            self._cancel.clear()
            self._thr = threading.Thread(
                target=self._worker,
                name="PlatesolveWorker",
                daemon=True,
            )
            self._thr.start()
            started = True
        if started:
            log_info(self._out_log, "Platesolve: thread started")

    def stop(self) -> None:
        self._cancel.set()
        log_info(self._out_log, "Platesolve: cancel requested")
        thr = None
        with self._lock:
            thr = self._thr
        if thr is not None:
            thr.join(timeout=2.0)
        with self._lock:
            self._thr = None
            self._pending = None

    def request(self, target: Any) -> None:
        with self._lock:
            self._pending = {"target": target}
        log_info(self._out_log, f"Platesolve: request queued target={_summarize_target(target)}")
        self.start_if_needed()

    def maybe_autosolve(self) -> None:
        cfg = self.get_cfg_snapshot()
        if not bool(cfg.auto_solve):
            return
        target = str(self._auto_target or "").strip()
        if not target:
            return
        st = self._runner.get_state()
        if bool(getattr(st, "platesolve_busy", False)):
            return
        now = time.perf_counter()
        if (now - float(self._last_auto_t)) < max(2.0, float(cfg.solve_every_s)):
            return
        self.request(target=target)
        self._last_auto_t = float(now)

    def get_last_result(self) -> Optional[Any]:
        return self._last_result

    def set_auto_target(self, target: str) -> None:
        self._auto_target = str(target or "")

    def cache_result(self, result: Any) -> None:
        self._last_result = result

    def get_cfg_snapshot(self) -> PlatesolveConfig:
        with self._cfg_lock:
            return replace(self._runner.cfg.platesolve)

    def render_debug_jpeg(
        self,
        frame: Optional[np.ndarray],
        overlay: Optional[List[Any]],
    ) -> Optional[bytes]:
        if frame is None:
            return None
        gray = frame
        if getattr(gray, "ndim", 0) == 3:
            if gray.shape[2] == 1:
                gray = gray[:, :, 0]
            else:
                gray = gray[:, :, :3].astype(np.float32).mean(axis=2)
        gray = np.asarray(gray, dtype=np.float32)
        if gray.ndim != 2:
            return None

        p1, p99 = np.percentile(gray, [1.0, 99.0])
        if p99 <= p1:
            p1 = float(gray.min()) if gray.size else 0.0
            p99 = float(gray.max()) if gray.size else 1.0
        scale = 255.0 / max(1e-6, float(p99 - p1))
        u8 = np.clip((gray - p1) * scale, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

        if overlay:
            h, w = img.shape[:2]
            colors = {
                "det": (255, 0, 0),
                "match": (0, 255, 0),
                "guide": (0, 0, 255),
            }
            for item in overlay:
                x = int(round(float(getattr(item, "x", 0.0))))
                y = int(round(float(getattr(item, "y", 0.0))))
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                kind = str(getattr(item, "kind", "det"))
                color = colors.get(kind, (255, 255, 0))
                radius = 10 if kind == "guide" else 8 if kind == "match" else 7
                cv2.circle(img, (x, y), radius, color, 1, lineType=cv2.LINE_AA)
                label = getattr(item, "label", None)
                if kind == "guide" and label:
                    cv2.putText(
                        img,
                        str(label),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return bytes(buf.tobytes())

    def build_debug_info(self, result: Any) -> Dict[str, Any]:
        metrics = dict(getattr(result, "metrics", {}) or {})
        info = {
            "status": str(getattr(result, "status", "")),
            "response": float(getattr(result, "response", 0.0)),
            "n_det": metrics.get("n_det"),
            "gaia_rows": metrics.get("gaia_rows"),
            "n_inliers": int(getattr(result, "n_inliers", 0)),
            "rms_px": float(getattr(result, "rms_px", 0.0)),
            "theta_deg": float(getattr(result, "theta_deg", 0.0)),
            "dx_px": float(getattr(result, "dx_px", 0.0)),
            "dy_px": float(getattr(result, "dy_px", 0.0)),
            "radius_deg": metrics.get("radius_deg"),
            "scale_arcsec_per_px": float(
                getattr(result, "scale_arcsec_per_px", metrics.get("scale_arcsec_per_px", 0.0))
            ),
        }
        return info

    def _worker(self) -> None:
        """
        Worker que ejecuta plate solving sin bloquear el loop principal.
        Toma requests desde self._pending (la última gana).

        Además, en cada solve guarda un "snapshot" reproducible en disco:
          - raw (exacto desde la cámara) + meta + config + target/source
          - u8_view (si existe) para inspección rápida
          - debug_jpeg + debug_info/result para reproducir el diagnóstico
        """

        def _dump_dir() -> Path:
            d = Path("platesolve_dumps")
            d.mkdir(parents=True, exist_ok=True)
            return d

        def _dump_snapshot(
            *,
            source: str,
            target: Any,
            raw: np.ndarray,
            fmt: str,
            meta: Dict[str, Any],
            u8_view: Optional[np.ndarray],
            cfg: PlatesolveConfig,
            observer: ObserverConfig,
            extra: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            """
            Guarda:
              - *_raw.npy (exacto)
              - *_u8.npy (opcional)
              - *_meta.json (metadatos + cfg/observer + request)
            Devuelve base path (sin extensión) o None.
            """
            try:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base = _dump_dir() / f"{ts}_{str(source)}_{_safe_slug(str(target))}"

                raw_c = np.ascontiguousarray(raw)
                np.save(str(base) + "_raw.npy", raw_c)

                if u8_view is not None:
                    np.save(str(base) + "_u8.npy", np.ascontiguousarray(u8_view))

                info = {
                    "ts": ts,
                    "source": str(source),
                    "target": target,
                    "fmt": str(fmt),
                    "shape": list(raw_c.shape),
                    "dtype": str(raw_c.dtype),
                    "meta": meta or {},
                    "platesolve_cfg": dict(getattr(cfg, "__dict__", {}) or {}),
                    "observer": dict(getattr(observer, "__dict__", {}) or {}),
                }
                if extra:
                    info["extra"] = extra

                with open(str(base) + "_meta.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

                return str(base)
            except (OSError, ValueError, TypeError) as exc:
                log_error(self._out_log, "Platesolve: failed to dump snapshot", exc)
                return None

        log_info(self._out_log, "Platesolve: worker loop started")
        while not self._runner._stop.is_set() and not self._cancel.is_set():
            req: Optional[Dict[str, Any]] = None
            with self._lock:
                req = self._pending
                self._pending = None

            if req is None:
                time.sleep(0.05)
                continue

            t_solve0 = time.perf_counter()
            self._runner._set_state_safe(
                platesolve_busy=True,
                platesolve_status="RUNNING",
                platesolve_debug_jpeg=None,
                platesolve_debug_info=None,
            )

            dump_base: Optional[str] = None

            try:
                target = req.get("target", None)

                if target is None:
                    self._runner._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_TARGET",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_TARGET"},
                    )
                    log_info(self._out_log, "Platesolve: ERR_NO_TARGET")
                    continue

                if self._runner._cam_stream is None:
                    self._runner._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_CAMERA",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_CAMERA"},
                    )
                    log_info(self._out_log, "Platesolve: ERR_NO_CAMERA")
                    continue

                fr = self._runner._cam_stream.latest()
                if fr is None:
                    self._runner._set_state_safe(
                        platesolve_busy=False,
                        platesolve_status="ERR_NO_FRAME",
                        platesolve_last_ok=False,
                        platesolve_debug_jpeg=None,
                        platesolve_debug_info={"status": "ERR_NO_FRAME"},
                    )
                    log_info(self._out_log, "Platesolve: ERR_NO_FRAME")
                    continue

                fmt = str(getattr(fr, "fmt", "") or "RAW16")
                meta = dict(getattr(fr, "meta", {}) or {})
                raw_in = np.ascontiguousarray(fr.raw)
                u8_in = np.ascontiguousarray(fr.u8_view) if getattr(fr, "u8_view", None) is not None else None

                platesolve_cfg = self.get_cfg_snapshot()

                dump_base = _dump_snapshot(
                    source="live",
                    target=target,
                    raw=raw_in,
                    fmt=fmt,
                    meta=meta,
                    u8_view=u8_in,
                    cfg=platesolve_cfg,
                    observer=self._runner._platesolve_observer,
                )

                frame = ensure_raw16_bayer(raw_in)

                debug_stats = bool(getattr(platesolve_cfg, "debug_input_stats", False))

                def _stats(a: np.ndarray, name: str) -> None:
                    if not debug_stats:
                        return
                    a = np.asarray(a)
                    log_info(self._out_log, f"[{name}] shape={a.shape} dtype={a.dtype} C={a.flags['C_CONTIGUOUS']}")
                    if a.size == 0:
                        log_info(self._out_log, "  EMPTY")
                        return
                    if a.ndim == 1:
                        log_info(self._out_log, f"  1D buffer: min={a.min()} max={a.max()} mean={a.mean():.3g}")
                        return
                    flat = a.reshape(-1)
                    p = np.percentile(flat, [0, 1, 5, 50, 95, 99, 100])
                    log_info(self._out_log, f"  min/p1/p5/p50/p95/p99/max = {p}")
                    log_info(self._out_log, f"  mean={flat.mean():.3g} std={flat.std():.3g}")
                    if a.dtype == np.uint16:
                        log_info(self._out_log, f"  sat65535={np.mean(flat == 65535):.4f}")
                    if a.dtype == np.uint8:
                        log_info(self._out_log, f"  sat255={np.mean(flat == 255):.4f}")

                _stats(raw_in, "fr.raw")
                if hasattr(fr, "u8_view") and fr.u8_view is not None:
                    _stats(fr.u8_view, "fr.u8_view")
                _stats(frame, "frame(raw16)")

                result = platesolve_sweep(
                    frame,
                    target=target,
                    cfg=platesolve_cfg,
                    sep_cfg=self._runner.cfg.sep,
                    observer=self._runner._platesolve_observer,
                    progress_cb=None,
                )

                debug_jpeg = self.render_debug_jpeg(
                    frame,
                    list(getattr(result, "overlay", []) or []),
                )
                debug_info = self.build_debug_info(result)

                if dump_base:
                    try:
                        if debug_jpeg:
                            with open(dump_base + "_debug.jpg", "wb") as f:
                                f.write(debug_jpeg)
                        with open(dump_base + "_result.json", "w", encoding="utf-8") as f:
                            json.dump(debug_info, f, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        log_error(self._out_log, "Platesolve: failed to dump debug outputs", exc)

                self._runner._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status=getattr(result, "status", "UNKNOWN"),
                    platesolve_last_ok=bool(getattr(result, "success", False)),
                    platesolve_theta_deg=float(getattr(result, "theta_deg", 0.0)),
                    platesolve_dx_px=float(getattr(result, "dx_px", 0.0)),
                    platesolve_dy_px=float(getattr(result, "dy_px", 0.0)),
                    platesolve_resp=float(getattr(result, "response", 0.0)),
                    platesolve_n_inliers=int(getattr(result, "n_inliers", 0)),
                    platesolve_rms_px=float(getattr(result, "rms_px", 0.0)),
                    platesolve_overlay=list(getattr(result, "overlay", []) or []),
                    platesolve_guides=list(getattr(result, "guides", []) or []),
                    platesolve_debug_jpeg=debug_jpeg,
                    platesolve_debug_info=debug_info,
                    platesolve_center_ra_deg=float(getattr(result, "center_ra_deg", 0.0)),
                    platesolve_center_dec_deg=float(getattr(result, "center_dec_deg", 0.0)),
                )

                status = str(getattr(result, "status", "UNKNOWN"))
                success = bool(getattr(result, "success", False))
                resp = float(getattr(result, "response", 0.0))
                n_inliers = int(getattr(result, "n_inliers", 0))
                rms_px = float(getattr(result, "rms_px", 0.0))
                elapsed_s = time.perf_counter() - t_solve0
                if success:
                    log_info(
                        self._out_log,
                        (
                            f"Platesolve: OK status={status} resp={resp:.3g} "
                            f"inliers={n_inliers} rms_px={rms_px:.3g} t={elapsed_s:.2f}s"
                        ),
                    )
                else:
                    log_info(
                        self._out_log,
                        (
                            f"Platesolve: ERR status={status} resp={resp:.3g} "
                            f"inliers={n_inliers} rms_px={rms_px:.3g} t={elapsed_s:.2f}s"
                        ),
                    )

                if bool(getattr(result, "success", False)):
                    self._last_result = result

            except Exception as exc:
                self._runner._set_state_safe(
                    platesolve_busy=False,
                    platesolve_status="ERR_EXCEPTION",
                    platesolve_last_ok=False,
                )
                log_error(self._out_log, "Platesolve: failed", exc)
        log_info(self._out_log, "Platesolve: worker loop stopped")
