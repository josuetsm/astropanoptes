# stacking.py
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np

from config import AppConfig
from imaging import ensure_raw16_bayer
from logging_utils import log_error
from preview import stretch_to_u8
from workers import BaseWorker

# ============================================================
# Types
# ============================================================

ColorMode = Literal["mono", "rgb"]

_BAYER_TO_GRAY_CODE: Dict[str, int] = {
    "RGGB": cv2.COLOR_BAYER_RG2GRAY,
    "BGGR": cv2.COLOR_BAYER_BG2GRAY,
    "GRBG": cv2.COLOR_BAYER_GR2GRAY,
    "GBRG": cv2.COLOR_BAYER_GB2GRAY,
}

_BAYER_TO_RGB_CODE: Dict[str, int] = {
    # OpenCV Bayer codes are typically consumed as BGR images. We keep an internal
    # RGB stack, so we use the *2BGR conversion constants to land channels in RGB order.
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}


def _odd_ksize(v: int, *, minimum: int = 1) -> int:
    k = max(int(v), int(minimum))
    if (k % 2) == 0:
        k += 1
    return k


def _smooth_kernel(k: int) -> np.ndarray:
    x = np.ones(max(1, int(k)), dtype=np.float64)
    return x / x.sum()


def _bayer_to_gray_code(pattern: str) -> int:
    return _BAYER_TO_GRAY_CODE.get(str(pattern).upper(), cv2.COLOR_BAYER_RG2GRAY)


def _bayer_to_rgb_code(pattern: str) -> int:
    return _BAYER_TO_RGB_CODE.get(str(pattern).upper(), cv2.COLOR_BayerRG2BGR)


# ============================================================
# Live Mosaic Stacker (alignment in gray, stack in gray or RGB)
# ============================================================

class LiveMosaicStackerGray:
    def __init__(
        self,
        *,
        color_mode: ColorMode,
        align_median_k: int,
        smooth_k: int,
        max_shift_px: int,
        use_subpixel: bool,
        preview_log_vmin: float,
        bayer_to_gray_code: int,
        bayer_to_rgb_code: int,
    ):
        self.color_mode: ColorMode = "rgb" if str(color_mode).lower() == "rgb" else "mono"
        self.align_median_k = _odd_ksize(align_median_k, minimum=1)
        self.smooth_k = max(1, int(smooth_k))
        self.max_shift_px = max(1, int(max_shift_px))
        self.use_subpixel = bool(use_subpixel)
        self.preview_log_vmin = float(preview_log_vmin)
        self.bayer_to_gray_code = int(bayer_to_gray_code)
        self.bayer_to_rgb_code = int(bayer_to_rgb_code)
        self.kernel = _smooth_kernel(self.smooth_k)

        self.sum: Optional[np.ndarray] = None
        self.wgt: Optional[np.ndarray] = None
        self.canvas_h = 0
        self.canvas_w = 0
        self.frame_h = 0
        self.frame_w = 0

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.n = 0

    def reset(self) -> None:
        self.sum = None
        self.wgt = None
        self.canvas_h = 0
        self.canvas_w = 0
        self.frame_h = 0
        self.frame_w = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.n = 0

    def has_data(self) -> bool:
        return self.sum is not None and self.wgt is not None and self.n > 0

    def _profile_1d(self, img_u16: np.ndarray, which: str) -> np.ndarray:
        img = cv2.medianBlur(img_u16, self.align_median_k)
        if which == "dx":
            p = img.max(axis=0).astype(np.float64, copy=False)
        elif which == "dy":
            p = img.max(axis=1).astype(np.float64, copy=False)
        else:
            raise ValueError("which must be dx or dy")
        return np.convolve(p, self.kernel, mode="same")

    @staticmethod
    def _shift_1d_centered(
        p_ref: np.ndarray,
        p_cur: np.ndarray,
        *,
        center: float,
        max_shift: int,
        subpixel: bool,
    ) -> float:
        if not np.isfinite(center):
            center = 0.0

        a = p_cur - p_cur.mean()
        b = p_ref - p_ref.mean()

        corr = np.correlate(a, b, mode="full")
        if not np.any(np.isfinite(corr)):
            return 0.0

        L = len(p_cur)
        shifts = np.arange(-(L - 1), L, dtype=np.float64)

        lo = float(center) - float(max_shift)
        hi = float(center) + float(max_shift)
        mask = (shifts >= lo) & (shifts <= hi)
        if np.any(mask):
            corr[~mask] = -np.inf

        i0 = int(np.argmax(corr))
        if not np.isfinite(corr[i0]):
            return 0.0
        shift_int = i0 - (L - 1)

        if not subpixel:
            return float(shift_int)

        delta = 0.0
        if 1 <= i0 < (len(corr) - 1):
            y0, y1, y2 = float(corr[i0 - 1]), float(corr[i0]), float(corr[i0 + 1])
            if np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2):
                denom = y0 - 2.0 * y1 + y2
                if np.isfinite(denom) and abs(denom) > 1e-12:
                    cand = 0.5 * (y0 - y2) / denom
                    if np.isfinite(cand):
                        delta = cand

        out = float(shift_int + delta)
        if not np.isfinite(out):
            return float(shift_int)
        return out

    def _raw_to_gray_align(self, raw_u16: np.ndarray) -> np.ndarray:
        raw_f = cv2.medianBlur(raw_u16, self.align_median_k)
        return cv2.cvtColor(raw_f, self.bayer_to_gray_code)

    def _raw_to_stack(self, raw_u16: np.ndarray) -> np.ndarray:
        if self.color_mode == "rgb":
            return cv2.cvtColor(raw_u16, self.bayer_to_rgb_code)
        return cv2.cvtColor(raw_u16, self.bayer_to_gray_code)

    @staticmethod
    def _warp_img(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _ensure_canvas(self, x0: int, y0: int, w: int, h: int) -> tuple[int, int]:
        if self.sum is None or self.wgt is None:
            return x0, y0

        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x0 + w - self.canvas_w)
        pad_b = max(0, y0 + h - self.canvas_h)

        if pad_l or pad_t or pad_r or pad_b:
            new_h = self.canvas_h + pad_t + pad_b
            new_w = self.canvas_w + pad_l + pad_r

            if self.sum.ndim == 2:
                new_sum = np.zeros((new_h, new_w), np.float64)
                new_sum[pad_t : pad_t + self.canvas_h, pad_l : pad_l + self.canvas_w] = self.sum
            else:
                channels = int(self.sum.shape[2])
                new_sum = np.zeros((new_h, new_w, channels), np.float64)
                new_sum[pad_t : pad_t + self.canvas_h, pad_l : pad_l + self.canvas_w, :] = self.sum
            new_wgt = np.zeros((new_h, new_w), np.float64)

            new_wgt[pad_t : pad_t + self.canvas_h, pad_l : pad_l + self.canvas_w] = self.wgt

            self.sum = new_sum
            self.wgt = new_wgt
            self.canvas_h = new_h
            self.canvas_w = new_w

            self.pos_x += float(pad_l)
            self.pos_y += float(pad_t)

            x0 += pad_l
            y0 += pad_t

        return x0, y0

    def _build_ref_gray_u16(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        if self.sum is None or self.wgt is None:
            return np.zeros((h, w), np.uint16)

        W = self.wgt[y0 : y0 + h, x0 : x0 + w]
        S = self.sum[y0 : y0 + h, x0 : x0 + w]
        m = W > 0

        if self.color_mode == "rgb" and S.ndim == 3:
            ref_rgb = np.zeros((h, w, 3), np.uint16)
            if np.any(m):
                ref_vals = np.clip(S[m] / W[m, None], 0.0, 65535.0)
                ref_rgb[m] = ref_vals.astype(np.uint16, copy=False)
            return cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)

        ref = np.zeros((h, w), np.uint16)
        if np.any(m):
            ref_vals = np.clip(S[m] / W[m], 0.0, 65535.0)
            ref[m] = ref_vals.astype(np.uint16, copy=False)
        return ref

    def add_frame(self, raw_u16: np.ndarray) -> None:
        gray_align = self._raw_to_gray_align(raw_u16)
        stack_img = self._raw_to_stack(raw_u16)
        h, w = stack_img.shape[:2]

        if self.sum is None or self.wgt is None:
            self.sum = stack_img.astype(np.float64, copy=False)
            self.wgt = np.ones((h, w), np.float64)
            self.canvas_h = h
            self.canvas_w = w
            self.frame_h = h
            self.frame_w = w
            self.n = 1
            self.last_dx = 0.0
            self.last_dy = 0.0
            return

        if (h != self.frame_h) or (w != self.frame_w):
            # ROI/binning changed while stacking: reinitialize to keep state consistent.
            self.reset()
            self.sum = stack_img.astype(np.float64, copy=False)
            self.wgt = np.ones((h, w), np.float64)
            self.canvas_h = h
            self.canvas_w = w
            self.frame_h = h
            self.frame_w = w
            self.n = 1
            return

        x_ref = int(np.floor(self.pos_x))
        y_ref = int(np.floor(self.pos_y))
        x_ref, y_ref = self._ensure_canvas(x_ref, y_ref, w, h)
        ref = self._build_ref_gray_u16(y_ref, x_ref, h, w)

        dx = self._shift_1d_centered(
            self._profile_1d(ref, "dx"),
            self._profile_1d(gray_align, "dx"),
            center=self.last_dx,
            max_shift=self.max_shift_px,
            subpixel=self.use_subpixel,
        )
        dy = self._shift_1d_centered(
            self._profile_1d(ref, "dy"),
            self._profile_1d(gray_align, "dy"),
            center=self.last_dy,
            max_shift=self.max_shift_px,
            subpixel=self.use_subpixel,
        )
        if not np.isfinite(dx):
            dx = 0.0
        if not np.isfinite(dy):
            dy = 0.0

        self.pos_x -= dx
        self.pos_y -= dy
        self.last_dx = float(dx)
        self.last_dy = float(dy)

        x0 = int(np.floor(self.pos_x))
        y0 = int(np.floor(self.pos_y))
        fx = float(self.pos_x - x0)
        fy = float(self.pos_y - y0)

        x0, y0 = self._ensure_canvas(x0, y0, w, h)

        if self.use_subpixel:
            warped = self._warp_img(stack_img, fx, fy)
        else:
            warped = stack_img

        if warped.ndim == 2:
            self.sum[y0 : y0 + h, x0 : x0 + w] += warped.astype(np.float64, copy=False)
        else:
            self.sum[y0 : y0 + h, x0 : x0 + w, :] += warped.astype(np.float64, copy=False)
        self.wgt[y0 : y0 + h, x0 : x0 + w] += 1.0
        self.n += 1

    def get_mean_u16(self) -> Optional[np.ndarray]:
        if self.sum is None or self.wgt is None:
            return None

        h = min(self.sum.shape[0], self.wgt.shape[0])
        w = min(self.sum.shape[1], self.wgt.shape[1])
        src_wgt = self.wgt[:h, :w]
        if self.sum.ndim == 2:
            src_sum = self.sum[:h, :w]
            mean = np.zeros_like(src_sum, dtype=np.float64)
            np.divide(src_sum, src_wgt, out=mean, where=src_wgt > 0)
        else:
            src_sum = self.sum[:h, :w, :]
            mean = np.zeros_like(src_sum, dtype=np.float64)
            np.divide(src_sum, src_wgt[..., None], out=mean, where=src_wgt[..., None] > 0)
        mean = np.nan_to_num(mean, nan=0.0, posinf=65535.0, neginf=0.0)
        return np.clip(mean, 0.0, 65535.0).astype(np.uint16)

    def get_weight_f32(self) -> Optional[np.ndarray]:
        if self.wgt is None:
            return None
        return np.nan_to_num(self.wgt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def get_preview_u8(self) -> Optional[np.ndarray]:
        mean = self.get_mean_u16()
        if mean is None:
            return None

        if mean.ndim == 3:
            x = np.log1p(mean.astype(np.float32, copy=False))
            y = np.empty_like(x, dtype=np.float32)
            for c in range(3):
                xc = x[..., c]
                vmax_c = float(xc.max()) if xc.size > 0 else self.preview_log_vmin + 1.0
                denom_c = max(vmax_c - self.preview_log_vmin, 1e-6)
                y[..., c] = np.clip((xc - self.preview_log_vmin) / denom_c, 0.0, 1.0)
            return (y * 255.0 + 0.5).astype(np.uint8)

        x = np.log1p(mean.astype(np.float32, copy=False))
        vmax = float(x.max()) if x.size > 0 else self.preview_log_vmin + 1.0
        denom = max(vmax - self.preview_log_vmin, 1e-6)
        y = (x - self.preview_log_vmin) / denom
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# Metrics + Engine
# ============================================================

@dataclass
class StackingMetrics:
    enabled: bool = False

    # legacy fields preserved for UI/state compatibility
    scale: float = 2.0
    pixfrac: float = 0.8
    tile_size_out: int = 512
    max_tiles: int = 64

    frames_in: int = 0
    frames_used: int = 0
    frames_dropped: int = 0
    frames_rejected: int = 0

    tiles_used: int = 0
    tiles_evicted: int = 0

    last_resp: float = 0.0
    last_dx: float = 0.0
    last_dy: float = 0.0
    last_theta_deg: float = 0.0

    stacking_fps: float = 0.0
    last_preview_t: float = 0.0


@dataclass
class StackEngine:
    cfg: AppConfig
    metrics: StackingMetrics = field(default_factory=StackingMetrics)

    enabled: bool = False
    color_mode: ColorMode = "mono"
    canvas: Optional[Any] = None

    _preview_jpeg: Optional[bytes] = None
    _preview_lock: threading.Lock = field(default_factory=threading.Lock)
    _live_gray: Optional[LiveMosaicStackerGray] = None
    _stack_lock: threading.RLock = field(default_factory=threading.RLock)

    def configure_from_cfg(self) -> None:
        with self._stack_lock:
            scfg = self.cfg.stacking
            color_mode = str(getattr(scfg, "color_mode", "mono")).lower()
            self.color_mode = "rgb" if color_mode == "rgb" else "mono"
            self.canvas = None
            self._live_gray = LiveMosaicStackerGray(
                color_mode=self.color_mode,
                align_median_k=int(getattr(scfg, "align_median_k", 3)),
                smooth_k=int(getattr(scfg, "smooth_k", 30)),
                max_shift_px=int(getattr(scfg, "max_shift_px", 50)),
                use_subpixel=bool(getattr(scfg, "use_subpixel", True)),
                preview_log_vmin=float(getattr(scfg, "preview_log_vmin", 5.0)),
                bayer_to_gray_code=_bayer_to_gray_code(str(scfg.bayer_pattern)),
                bayer_to_rgb_code=_bayer_to_rgb_code(str(scfg.bayer_pattern)),
            )
            self.metrics.scale = float(scfg.drizzle_scale)
            self.metrics.pixfrac = float(scfg.pixfrac)
            self.metrics.tile_size_out = int(scfg.tile_size_out)
            self.metrics.max_tiles = int(scfg.max_tiles)

    def start(self) -> None:
        with self._stack_lock:
            if self._live_gray is None:
                self.configure_from_cfg()
            self.enabled = True
            self.metrics.enabled = True

    def stop(self) -> None:
        with self._stack_lock:
            self.enabled = False
            self.metrics.enabled = False

    def reset(self) -> None:
        with self._stack_lock:
            if self._live_gray is not None:
                self._live_gray.reset()

            with self._preview_lock:
                self._preview_jpeg = None

            self.metrics.frames_in = 0
            self.metrics.frames_used = 0
            self.metrics.frames_dropped = 0
            self.metrics.frames_rejected = 0
            self.metrics.tiles_used = 0
            self.metrics.tiles_evicted = 0
            self.metrics.last_resp = 0.0
            self.metrics.last_dx = 0.0
            self.metrics.last_dy = 0.0
            self.metrics.last_theta_deg = 0.0
            self.metrics.stacking_fps = 0.0
            self.metrics.last_preview_t = 0.0

    def set_params(self, **kwargs: Any) -> None:
        with self._stack_lock:
            scfg = self.cfg.stacking
            for k, v in kwargs.items():
                if hasattr(scfg, k):
                    setattr(scfg, k, v)
            was_enabled = bool(self.enabled)
            self.configure_from_cfg()
            self.enabled = was_enabled
            self.metrics.enabled = was_enabled
            with self._preview_lock:
                self._preview_jpeg = None

    def get_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._preview_jpeg

    def get_stack_mean(self, *, out_dtype: Optional[np.dtype] = np.float32) -> Optional[np.ndarray]:
        with self._stack_lock:
            if self._live_gray is None:
                return None
            mean_u16 = self._live_gray.get_mean_u16()
            if mean_u16 is None:
                return None
            if out_dtype is None:
                return mean_u16
            if out_dtype == np.uint16:
                return mean_u16
            return mean_u16.astype(out_dtype, copy=False)

    def get_stack_weight(self, *, out_dtype: Optional[np.dtype] = np.float32) -> Optional[np.ndarray]:
        with self._stack_lock:
            if self._live_gray is None:
                return None
            w = self._live_gray.get_weight_f32()
            if w is None:
                return None
            if out_dtype is None:
                return w
            return w.astype(out_dtype, copy=False)

    def get_stack_snapshot(
        self,
        *,
        mean_dtype: Optional[np.dtype] = np.float32,
        wgt_dtype: Optional[np.dtype] = np.float32,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._stack_lock:
            if self._live_gray is None:
                return None, None

            mean_u16 = self._live_gray.get_mean_u16()
            wgt_f32 = self._live_gray.get_weight_f32()
            if mean_u16 is None:
                return None, None

            if mean_dtype is None:
                mean = mean_u16
            elif mean_dtype == np.uint16:
                mean = mean_u16
            else:
                mean = mean_u16.astype(mean_dtype, copy=False)

            if wgt_f32 is None:
                wgt = None
            elif wgt_dtype is None:
                wgt = wgt_f32
            else:
                wgt = wgt_f32.astype(wgt_dtype, copy=False)

            return mean, wgt

    def get_latest_stack_frame(
        self,
        *,
        kind: str = "mono",
        strategy: str = "median_tile",
        out_dtype: Optional[np.dtype] = np.uint8,
    ) -> Optional[np.ndarray]:
        with self._stack_lock:
            _ = kind
            if str(strategy) not in ("median_tile", "full", "full_mosaic"):
                return None

            if self._live_gray is None:
                return None
            mean_u16 = self._live_gray.get_mean_u16()
            if mean_u16 is None:
                return None

            img_f32 = mean_u16.astype(np.float32, copy=False)
            if out_dtype is None:
                return img_f32
            if out_dtype == np.uint8:
                return stretch_to_u8(img_f32)
            if out_dtype == np.float32:
                return img_f32
            return img_f32.astype(out_dtype, copy=False)

    def step_batch(self, batch: List[Dict[str, Any]]) -> None:
        with self._stack_lock:
            if not self.enabled or self._live_gray is None or not batch:
                return

            t0 = time.perf_counter()
            scfg = self.cfg.stacking
            self.metrics.frames_in += len(batch)

            used = 0
            rejected = 0

            for item in batch:
                try:
                    raw16_work = ensure_raw16_bayer(item["raw16"])
                    self._live_gray.add_frame(raw16_work)
                    used += 1
                except Exception as exc:
                    rejected += 1
                    log_error(
                        None,
                        "Stacking: live-stacking frame failed",
                        exc,
                        throttle_s=2.0,
                        throttle_key="stacking_live_frame_failed",
                    )

            self.metrics.frames_used += used
            self.metrics.frames_rejected += rejected
            self.metrics.tiles_used = 1 if self._live_gray.has_data() else 0
            self.metrics.tiles_evicted = 0
            self.metrics.last_resp = 1.0 if used > 0 else 0.0
            self.metrics.last_dx = float(self._live_gray.last_dx)
            self.metrics.last_dy = float(self._live_gray.last_dy)
            self.metrics.last_theta_deg = 0.0

            dt = time.perf_counter() - t0
            if used > 0 and dt > 1e-6:
                fps_now = float(used) / dt
                self.metrics.stacking_fps = 0.9 * self.metrics.stacking_fps + 0.1 * fps_now

            now_t = time.time()
            preview_hz = float(getattr(scfg, "preview_hz", 1.0))
            if (now_t - self.metrics.last_preview_t) >= (1.0 / max(1e-6, preview_hz)):
                self.metrics.last_preview_t = now_t
                self._update_preview_jpeg()

    def _update_preview_jpeg(self) -> None:
        with self._stack_lock:
            if self._live_gray is None:
                with self._preview_lock:
                    self._preview_jpeg = None
                return

            u8 = self._live_gray.get_preview_u8()
            if u8 is None:
                with self._preview_lock:
                    self._preview_jpeg = None
                return

            u8s = cv2.resize(
                u8,
                (max(1, u8.shape[1] // 2), max(1, u8.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            u8_for_jpeg = u8s if u8s.ndim == 2 else cv2.cvtColor(u8s, cv2.COLOR_RGB2BGR)
            ok, jpg = cv2.imencode(".jpg", u8_for_jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            with self._preview_lock:
                self._preview_jpeg = jpg.tobytes() if ok else None


# ============================================================
# Worker thread wrapper
# ============================================================

class StackingWorker(BaseWorker):
    """
    Owns a StackEngine and a queue.
    Call enqueue_frame(...) from AppRunner loop (non-blocking),
    and let the worker consume and process in batches.
    """

    def __init__(self, cfg: AppConfig):
        super().__init__(name="StackingWorker")
        self.cfg = cfg
        self.engine = StackEngine(cfg)
        self.engine.configure_from_cfg()

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.stacking.max_queue)

        if bool(cfg.stacking.enabled_init):
            self.start()

    def start(self) -> None:
        self.engine.start()
        super().start()

    def stop(self) -> None:
        self.engine.stop()
        super().stop()
        self.join(timeout=1.0)

    def reset(self) -> None:
        self.engine.reset()

    def set_params(self, **kwargs: Any) -> None:
        self.engine.set_params(**kwargs)

    def enqueue_frame(self, raw16: np.ndarray, t: Optional[float] = None) -> None:
        if not self.engine.enabled:
            return
        item = {"raw16": raw16, "t": float(time.time() if t is None else t)}
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.engine.metrics.frames_dropped += 1
            return
        self.request(op="process")

    def _handle_request(self, request: Dict[str, Any]) -> None:
        _ = request
        batch_size = int(self.cfg.stacking.batch_size)
        while not self._cancel.is_set():
            batch: List[Dict[str, Any]] = []
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break

            for _ in range(batch_size - 1):
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            self.engine.step_batch(batch)
