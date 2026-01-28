# stacking.py
from __future__ import annotations

import math
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List, Literal

import numpy as np
import cv2
from config import AppConfig
from imaging import (
    ensure_raw16_bayer,
    debayer_cv2,
    warp_rgb16,
)
from preview import stretch_to_u8
from logging_utils import log_error
from sep_utils import sep_detect_from_raw16, estimate_shift_from_objects

# ============================================================
# Types
# ============================================================

ColorMode = Literal["mono", "rgb"]

_EPS = 1e-9


@dataclass
class StackingMetrics:
    enabled: bool = False

    # drizzle / mosaic
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
class Tile:
    sum: np.ndarray          # float32 (H,W) or (H,W,3)
    w: np.ndarray            # float32 (H,W)
    hits: int = 0
    last_used_t: float = 0.0


@dataclass
class TileCanvas:
    tile_size: int
    max_tiles: int
    color_mode: ColorMode

    tiles: Dict[Tuple[int, int], Tile] = field(default_factory=dict)
    tiles_evicted: int = 0

    def _alloc_tile(self) -> Tile:
        ts = self.tile_size
        if self.color_mode == "mono":
            s = np.zeros((ts, ts), dtype=np.float32)
        else:
            s = np.zeros((ts, ts, 3), dtype=np.float32)
        w = np.zeros((ts, ts), dtype=np.float32)
        return Tile(sum=s, w=w)

    def get_or_create(self, key: Tuple[int, int], now: float) -> Tile:
        t = self.tiles.get(key)
        if t is not None:
            t.last_used_t = now
            return t

        if len(self.tiles) >= self.max_tiles:
            lru_key = min(self.tiles.items(), key=lambda kv: kv[1].last_used_t)[0]
            del self.tiles[lru_key]
            self.tiles_evicted += 1

        t = self._alloc_tile()
        t.last_used_t = now
        self.tiles[key] = t
        return t

    def num_tiles(self) -> int:
        return len(self.tiles)


# ============================================================
# Alignment utilities (robust, CPU-only)
# ============================================================

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + _EPS


# ============================================================
# Affine / tiling helpers
# ============================================================

def _build_dst_to_src_affine(
    *,
    tile_u0: float,
    tile_v0: float,
    scale: float,
    dx: float,
    dy: float,
    theta_rad: float,
) -> np.ndarray:
    """
    2x3 affine matrix mapping DEST(tile-local output coords) -> SRC(frame coords),
    for cv2.warpAffine with WARP_INVERSE_MAP.

    Forward model (src->global_out):
        [u; v] = scale * ( R*[x; y] + [dx; dy] )

    Inverse (global_out->src):
        [x; y] = R^T * ([u; v]/scale - [dx; dy])

    tile-local dest:
        u = tile_u0 + x_out, v = tile_v0 + y_out
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    inv_scale = 1.0 / scale

    a11 = c * inv_scale
    a12 = s * inv_scale
    a21 = -s * inv_scale
    a22 = c * inv_scale

    b1 = c * (tile_u0 * inv_scale - dx) + s * (tile_v0 * inv_scale - dy)
    b2 = -s * (tile_u0 * inv_scale - dx) + c * (tile_v0 * inv_scale - dy)

    return np.array([[a11, a12, b1],
                     [a21, a22, b2]], dtype=np.float32)


def _warp_affine_tile(src: np.ndarray, M: np.ndarray, dsize: Tuple[int, int], is_color: bool) -> np.ndarray:
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    border_value = (0, 0, 0) if is_color else 0
    warped = cv2.warpAffine(
        src, M, dsize, flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )
    return warped.astype(np.float32, copy=False)


def _warp_mask_tile(src_shape: Tuple[int, int], M: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    """
    Warp a ones-mask to compute valid pixels; returns float32 mask in {0,1}.
    """
    h, w = src_shape
    ones = np.ones((h, w), dtype=np.uint8)
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    m = cv2.warpAffine(
        ones, M, dsize, flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return (m > 0).astype(np.float32, copy=False)


def _dst_tile_fully_inside_src(M_dst_to_src: np.ndarray, dsize: Tuple[int, int], src_w: int, src_h: int) -> bool:
    """
    If the entire destination tile maps inside the source bounds, we can skip warping the mask.
    """
    tw, th = dsize
    corners = np.array(
        [[0.0, 0.0, 1.0],
         [tw - 1.0, 0.0, 1.0],
         [0.0, th - 1.0, 1.0],
         [tw - 1.0, th - 1.0, 1.0]],
        dtype=np.float32
    )
    src_xy = corners @ M_dst_to_src.T  # (4,2)
    x = src_xy[:, 0]
    y = src_xy[:, 1]
    return bool(
        (x.min() >= 0.0) and (y.min() >= 0.0) and (x.max() <= (src_w - 1.0)) and (y.max() <= (src_h - 1.0))
    )


def _corners_bbox_out(
    w: int,
    h: int,
    *,
    scale: float,
    dx: float,
    dy: float,
    theta_rad: float,
) -> Tuple[float, float, float, float]:
    """
    Bounding box in output coords for transformed source frame.
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)

    corners = np.array(
        [[0.0, 0.0],
         [float(w - 1), 0.0],
         [0.0, float(h - 1)],
         [float(w - 1), float(h - 1)]],
        dtype=np.float32
    )
    x = corners[:, 0]
    y = corners[:, 1]
    xr = c * x - s * y
    yr = s * x + c * y
    u = scale * (xr + dx)
    v = scale * (yr + dy)
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


# ============================================================
# Engine
# ============================================================

@dataclass
class StackEngine:
    cfg: AppConfig
    metrics: StackingMetrics = field(default_factory=StackingMetrics)

    enabled: bool = False
    color_mode: ColorMode = "mono"
    canvas: Optional[TileCanvas] = None

    # preview snapshot
    _preview_jpeg: Optional[bytes] = None
    _preview_lock: threading.Lock = field(default_factory=threading.Lock)

    # persistent alignment reference (SEP objects, full-res)
    _ref_align_xy: Optional[np.ndarray] = None

    def configure_from_cfg(self) -> None:
        scfg = self.cfg.stacking
        self.color_mode = scfg.color_mode
        self.canvas = TileCanvas(
            tile_size=scfg.tile_size_out,
            max_tiles=scfg.max_tiles,
            color_mode=scfg.color_mode,
        )
        self.metrics.scale = float(scfg.drizzle_scale)
        self.metrics.pixfrac = float(scfg.pixfrac)
        self.metrics.tile_size_out = int(scfg.tile_size_out)
        self.metrics.max_tiles = int(scfg.max_tiles)

    def start(self) -> None:
        if self.canvas is None:
            self.configure_from_cfg()
        self.enabled = True
        self.metrics.enabled = True

    def stop(self) -> None:
        self.enabled = False
        self.metrics.enabled = False

    def reset(self) -> None:
        if self.canvas is not None:
            self.canvas.tiles.clear()
            self.canvas.tiles_evicted = 0

        self._ref_align_xy = None

        with self._preview_lock:
            self._preview_jpeg = None

        self.metrics.frames_in = 0
        self.metrics.frames_used = 0
        self.metrics.frames_dropped = 0
        self.metrics.frames_rejected = 0
        self.metrics.tiles_evicted = 0
        self.metrics.last_resp = 0.0
        self.metrics.last_dx = 0.0
        self.metrics.last_dy = 0.0
        self.metrics.last_theta_deg = 0.0
        self.metrics.stacking_fps = 0.0

    def set_params(self, **kwargs: Any) -> None:
        scfg = self.cfg.stacking
        for k, v in kwargs.items():
            if hasattr(scfg, k):
                setattr(scfg, k, v)
        self.configure_from_cfg()

    def get_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._preview_jpeg

    def get_latest_stack_frame(
        self,
        *,
        kind: str = "mono",
        strategy: str = "median_tile",
        out_dtype: Optional[np.dtype] = np.uint8,
    ) -> Optional[np.ndarray]:
        if self.canvas is None or self.canvas.num_tiles() == 0:
            return None
        if str(strategy) != "median_tile":
            return None

        keys = list(self.canvas.tiles.keys())
        txs = np.array([k[0] for k in keys], dtype=np.int32)
        tys = np.array([k[1] for k in keys], dtype=np.int32)
        k_med = (int(np.median(txs)), int(np.median(tys)))
        if k_med not in self.canvas.tiles:
            k_med = keys[0]

        tile = self.canvas.tiles[k_med]
        w = np.maximum(tile.w, 1e-6)

        if self.color_mode == "mono":
            img = (tile.sum / w).astype(np.float32, copy=False)
        else:
            img = (tile.sum / w[..., None]).astype(np.float32, copy=False)
            if str(kind) == "mono":
                img = img.mean(axis=2)

        if out_dtype is None:
            return img
        if out_dtype == np.uint8:
            return stretch_to_u8(img)
        if out_dtype == np.float32:
            return img.astype(np.float32, copy=False)
        return img.astype(out_dtype, copy=False)

    # --------------------------------------------------------
    # Core batch step
    # --------------------------------------------------------

    def step_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        batch items: dict with keys:
          - raw16: np.ndarray (H,W) uint16 Bayer
          - t: float timestamp
        """
        if not self.enabled or self.canvas is None or not batch:
            return

        scfg = self.cfg.stacking
        t0 = time.perf_counter()
        self.metrics.frames_in += len(batch)

        # 1) Alignment objects from RAW16 via SEP (full-res).
        align_xy: List[np.ndarray] = []
        raw16_work_list: List[np.ndarray] = []
        for item in batch:
            raw16_work = ensure_raw16_bayer(item["raw16"])
            raw16_work_list.append(raw16_work)
            _, _, _, obj_xy = sep_detect_from_raw16(
                raw16_work,
                sep_bw=int(self.cfg.sep.bw),
                sep_bh=int(self.cfg.sep.bh),
                sep_thresh_sigma=float(self.cfg.sep.thresh_sigma),
                sep_minarea=int(self.cfg.sep.minarea),
                max_sources=None,
            )
            align_xy.append(obj_xy)

        # 2) Persistent reference (stabilizes across batches).
        if self._ref_align_xy is None:
            self._ref_align_xy = align_xy[0].copy()
        ref_xy = self._ref_align_xy

        # 3) Shifts relative to reference (full-res).
        dxs_full = np.zeros(len(batch), dtype=np.float32)
        dys_full = np.zeros(len(batch), dtype=np.float32)
        resps = np.zeros(len(batch), dtype=np.float32)

        ref_shape = raw16_work_list[0].shape if raw16_work_list else (0, 0)
        max_shift = float(max(ref_shape) / 2.0) if ref_shape[0] > 0 else 0.0
        for i, obj_xy in enumerate(align_xy):
            dx, dy, resp, _ = estimate_shift_from_objects(ref_xy, obj_xy, max_shift_px=max_shift)
            dxs_full[i] = float(dx)
            dys_full[i] = float(dy)
            resps[i] = float(resp)

        # 4) Robust gating: response + shift outliers (radial MAD).
        resp_min = float(scfg.resp_min)
        k_mad = float(scfg.outlier_k_mad)
        mad_floor_px = 0.35  # prevents collapse when motion is tiny

        med_dx = float(np.median(dxs_full))
        med_dy = float(np.median(dys_full))
        dist = np.sqrt((dxs_full - med_dx) ** 2 + (dys_full - med_dy) ** 2).astype(np.float32, copy=False)
        mad_dist = max(_mad(dist), mad_floor_px)

        used: List[int] = []
        for i in range(len(batch)):
            if float(resps[i]) < resp_min:
                continue
            if float(dist[i]) > (k_mad * mad_dist):
                continue
            used.append(i)

        if not used:
            self.metrics.frames_rejected += len(batch)
            return

        self.metrics.frames_rejected += (len(batch) - len(used))

        # 5) Reference update (EMA in reference coordinates) using best-response frame.
        i_best = int(max(used, key=lambda j: float(resps[j])))
        dx_best = float(dxs_full[i_best])
        dy_best = float(dys_full[i_best])

        best_xy = align_xy[i_best]
        if best_xy.size > 0:
            shifted_best = best_xy + np.array([[dx_best, dy_best]], dtype=np.float64)
            self._ref_align_xy = shifted_best.astype(np.float64, copy=False)

        # 6) Accumulate accepted frames into tiles (drizzle grid). Rotation hook kept at zero.
        scale = float(scfg.drizzle_scale)
        tile_size = int(scfg.tile_size_out)
        pixfrac = float(scfg.pixfrac)
        theta_rad = 0.0
        theta_deg = 0.0

        now = time.time()
        dsize = (tile_size, tile_size)

        for i in used:
            raw16_work = raw16_work_list[i]
            dx = float(dxs_full[i])
            dy = float(dys_full[i])
            resp = float(resps[i])

            w_frame = float(np.clip((resp - resp_min) / max(1e-6, (1.0 - resp_min)), 0.0, 1.0))
            if w_frame <= 0.0:
                continue

            rgb16 = debayer_cv2(raw16_work, pattern=scfg.bayer_pattern, edge_aware=False)
            M_shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
            rgb16_aligned = warp_rgb16(rgb16, M_shift)
            if self.color_mode == "mono":
                src = rgb16_aligned.astype(np.float32, copy=False).mean(axis=2)
                src_is_color = False
            else:
                src = rgb16_aligned.astype(np.float32, copy=False)
                src_is_color = True
            if pixfrac > 0.7:
                src = cv2.GaussianBlur(src, (0, 0), sigmaX=0.6, sigmaY=0.6)
            h, w = src.shape[:2]

            dx_stack = 0.0
            dy_stack = 0.0

            u_min, v_min, u_max, v_max = _corners_bbox_out(
                w, h, scale=scale, dx=dx_stack, dy=dy_stack, theta_rad=theta_rad
            )

            tx0 = int(math.floor(u_min / tile_size))
            ty0 = int(math.floor(v_min / tile_size))
            tx1 = int(math.floor(u_max / tile_size))
            ty1 = int(math.floor(v_max / tile_size))

            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tile = self.canvas.get_or_create((tx, ty), now)
                    tile.hits += 1

                    tile_u0 = float(tx * tile_size)
                    tile_v0 = float(ty * tile_size)

                    M = _build_dst_to_src_affine(
                        tile_u0=tile_u0,
                        tile_v0=tile_v0,
                        scale=scale,
                        dx=dx_stack,
                        dy=dy_stack,
                        theta_rad=theta_rad,
                    )

                    warped = _warp_affine_tile(src, M, dsize, is_color=src_is_color)

                    if _dst_tile_fully_inside_src(M, dsize, src_w=w, src_h=h):
                        mask = np.full((tile_size, tile_size), w_frame, dtype=np.float32)
                    else:
                        mask = _warp_mask_tile((h, w), M, dsize) * w_frame

                    if self.color_mode == "mono":
                        tile.sum += warped * mask
                    else:
                        tile.sum += warped * mask[..., None]
                    tile.w += mask

        # 7) Metrics + preview.
        self.metrics.frames_used += len(used)
        self.metrics.tiles_used = self.canvas.num_tiles()
        self.metrics.tiles_evicted = self.canvas.tiles_evicted
        self.metrics.last_resp = float(np.median(resps[used]))
        self.metrics.last_dx = float(np.median(dxs_full[used]))
        self.metrics.last_dy = float(np.median(dys_full[used]))
        self.metrics.last_theta_deg = float(theta_deg)

        dt = time.perf_counter() - t0
        if dt > 1e-6:
            fps_now = float(len(used) / dt)
            self.metrics.stacking_fps = 0.9 * self.metrics.stacking_fps + 0.1 * fps_now

        now_t = time.time()
        if (now_t - self.metrics.last_preview_t) >= (1.0 / max(1e-6, float(scfg.preview_hz))):
            self.metrics.last_preview_t = now_t
            self._update_preview_jpeg()

    def _update_preview_jpeg(self) -> None:
        """
        Build a lightweight preview from a representative tile (median of keys).
        """
        if self.canvas is None or self.canvas.num_tiles() == 0:
            with self._preview_lock:
                self._preview_jpeg = None
            return

        keys = list(self.canvas.tiles.keys())
        txs = np.array([k[0] for k in keys], dtype=np.int32)
        tys = np.array([k[1] for k in keys], dtype=np.int32)
        k_med = (int(np.median(txs)), int(np.median(tys)))
        if k_med not in self.canvas.tiles:
            k_med = keys[0]

        tile = self.canvas.tiles[k_med]
        w = np.maximum(tile.w, 1e-6)

        if self.color_mode == "mono":
            img = (tile.sum / w).astype(np.float32, copy=False)
            u8 = stretch_to_u8(img)
            u8s = cv2.resize(
                u8,
                (max(1, u8.shape[1] // 2), max(1, u8.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            ok, jpg = cv2.imencode(".jpg", u8s, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        else:
            img = (tile.sum / w[..., None]).astype(np.float32, copy=False)
            u8 = stretch_to_u8(img)  # supports (H,W,3)
            u8s = cv2.resize(
                u8,
                (max(1, u8.shape[1] // 2), max(1, u8.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            ok, jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(u8s, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )

        with self._preview_lock:
            self._preview_jpeg = jpg.tobytes() if ok else None


# ============================================================
# Worker thread wrapper
# ============================================================

class StackingWorker:
    """
    Owns a StackEngine and a queue.
    Call enqueue_frame(...) from AppRunner loop (non-blocking),
    and let the worker consume and process in batches.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.engine = StackEngine(cfg)
        self.engine.configure_from_cfg()

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.stacking.max_queue)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        if bool(cfg.stacking.enabled_init):
            self.start()

    def start(self) -> None:
        self.engine.start()
        self._stop.clear()
        if self._thr is None or not self._thr.is_alive():
            self._thr = threading.Thread(target=self._run, name="stacking-worker", daemon=True)
            self._thr.start()

    def stop(self) -> None:
        self.engine.stop()
        self._stop.set()

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

    def _run(self) -> None:
        batch_size = int(self.cfg.stacking.batch_size)
        while not self._stop.is_set():
            batch: List[Dict[str, Any]] = []
            try:
                batch.append(self._q.get(timeout=0.1))
            except queue.Empty:
                continue

            for _ in range(batch_size - 1):
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            self.engine.step_batch(batch)
