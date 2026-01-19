# stacking.py
from __future__ import annotations

import math
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List, Literal

import importlib.util
import numpy as np
import cv2

from config import AppConfig
from imaging import (
    downsample_u16,
    extract_align_mono_u16,
    debayer_cv2,
    stretch_to_u8,
)

# ============================================================
# Config / Types
# ============================================================

ColorMode = Literal["mono", "rgb"]
BackendMode = Literal["auto", "cpu", "mps"]


@dataclass
class StackingMetrics:
    enabled: bool = False
    backend: str = "cpu"
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
        if key in self.tiles:
            t = self.tiles[key]
            t.last_used_t = now
            return t

        # If exceeding max_tiles, evict LRU tile.
        if len(self.tiles) >= self.max_tiles:
            # Evict least-recently-used
            lru_key = min(self.tiles.items(), key=lambda kv: kv[1].last_used_t)[0]
            del self.tiles[lru_key]
            self.tiles_evicted += 1

        t = self._alloc_tile()
        t.last_used_t = now
        self.tiles[key] = t
        return t

    def tiles_used(self) -> int:
        return len(self.tiles)


# ============================================================
# Backend selection (CPU now; MPS hook)
# ============================================================

class WarpBackend:
    def __init__(self, mode: BackendMode = "auto"):
        self.mode = mode
        self._torch = None
        self._device = None

        if mode in ("auto", "mps"):
            if importlib.util.find_spec("torch") is not None:
                import torch  # type: ignore
                if torch.backends.mps.is_available():
                    self._torch = torch
                    self._device = torch.device("mps")

    @property
    def name(self) -> str:
        if self._torch is not None and self._device is not None:
            return "mps"
        return "cpu"

    def warp_affine_tile(
        self,
        src: np.ndarray,
        M_dst_to_src: np.ndarray,
        dsize: Tuple[int, int],
        is_color: bool,
    ) -> np.ndarray:
        """
        Return warped tile image (float32).
        Uses CPU OpenCV by default.
        M_dst_to_src must be 2x3 mapping from dst coords -> src coords (OpenCV WARP_INVERSE_MAP).
        """
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        if not is_color:
            warped = cv2.warpAffine(
                src, M_dst_to_src, dsize, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        else:
            warped = cv2.warpAffine(
                src, M_dst_to_src, dsize, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
        return warped.astype(np.float32, copy=False)

    def warp_mask_tile(
        self,
        src_shape: Tuple[int, int],
        M_dst_to_src: np.ndarray,
        dsize: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp a ones-mask to compute valid pixels; returns float32 mask in [0,1].
        """
        h, w = src_shape
        ones = np.ones((h, w), dtype=np.uint8)
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        m = cv2.warpAffine(ones, M_dst_to_src, dsize, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (m > 0).astype(np.float32, copy=False)


# ============================================================
# Alignment (batch) utilities
# ============================================================

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-9


def _phasecorr_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate shift (dx, dy) such that b(x,y) aligns to a(x,y).
    Returns (dx, dy, resp). Uses cv2.phaseCorrelate on float32.
    Note: OpenCV returns shift (dx, dy) where (x+dx, y+dy).
    """
    a32 = a.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)

    # Hanning window improves robustness
    win = cv2.createHanningWindow((a32.shape[1], a32.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a32, b32, win)
    return float(dx), float(dy), float(resp)


def _estimate_theta_deg_logpolar(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Optional: estimate small rotation via log-polar phase correlation.
    Returns (theta_deg, resp). Keep as a hook: can be enabled later.
    """
    # For now, return 0 with resp=0 (disabled by default in step_batch)
    return 0.0, 0.0


# ============================================================
# Transform helpers
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
    Build a 2x3 affine matrix that maps DEST (tile-local output coords) -> SRC (frame coords),
    for cv2.warpAffine with WARP_INVERSE_MAP.
    Output coords are in drizzle grid (already scaled).
    Global output coord: u = tile_u0 + x_out, v = tile_v0 + y_out.
    Forward model (src->global_out): [u;v] = scale * (R*[x;y] + [dx;dy])
    So inverse (global_out->src): [x;y] = R^-1 * ([u;v]/scale - [dx;dy])
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)

    # R^-1 = R^T for rotation
    # [x;y] = R^T * ([u;v]/scale - [dx;dy])
    # Expand for tile-local (x_out, y_out):
    # u = tile_u0 + x_out, v = tile_v0 + y_out

    inv_scale = 1.0 / scale

    # Let U = (tile_u0 + x_out)*inv_scale - dx
    # Let V = (tile_v0 + y_out)*inv_scale - dy
    # x =  c*U + s*V
    # y = -s*U + c*V

    # x = c*(inv_scale*x_out + tile_u0*inv_scale - dx) + s*(inv_scale*y_out + tile_v0*inv_scale - dy)
    #   = (c*inv_scale)*x_out + (s*inv_scale)*y_out + c*(tile_u0*inv_scale - dx) + s*(tile_v0*inv_scale - dy)

    a11 = c * inv_scale
    a12 = s * inv_scale
    a21 = -s * inv_scale
    a22 = c * inv_scale

    b1 = c * (tile_u0 * inv_scale - dx) + s * (tile_v0 * inv_scale - dy)
    b2 = -s * (tile_u0 * inv_scale - dx) + c * (tile_v0 * inv_scale - dy)

    M = np.array([[a11, a12, b1],
                  [a21, a22, b2]], dtype=np.float32)
    return M


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
    backend: WarpBackend = field(default_factory=lambda: WarpBackend("auto"))

    # preview snapshot
    _preview_jpeg: Optional[bytes] = None
    _preview_lock: threading.Lock = field(default_factory=threading.Lock)

    # alignment reference for batches
    _key_align: Optional[np.ndarray] = None

    def configure_from_cfg(self) -> None:
        scfg = self.cfg.stacking
        self.color_mode = scfg.color_mode
        self.backend = WarpBackend(scfg.backend)
        self.canvas = TileCanvas(
            tile_size=scfg.tile_size_out,
            max_tiles=scfg.max_tiles,
            color_mode=scfg.color_mode,
        )
        self.metrics.scale = scfg.drizzle_scale
        self.metrics.pixfrac = scfg.pixfrac
        self.metrics.tile_size_out = scfg.tile_size_out
        self.metrics.max_tiles = scfg.max_tiles
        self.metrics.backend = self.backend.name

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
        self._key_align = None
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
        """
        Update selected parameters dynamically.
        """
        scfg = self.cfg.stacking
        for k, v in kwargs.items():
            if hasattr(scfg, k):
                setattr(scfg, k, v)
        # reconfigure canvas/backend if relevant
        self.configure_from_cfg()

    def get_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._preview_jpeg

    # ---------- Core batch step ----------

    def step_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        batch items: dict with keys:
          - raw: np.ndarray (H,W) uint8/uint16 (RAW Bayer or mono) OR rgb already
          - fmt: str ("RAW8","RAW16","MONO8","MONO16","RGB24","RGB48", etc.)
          - t: float timestamp
        """
        if not self.enabled or self.canvas is None:
            return

        scfg = self.cfg.stacking
        t0 = time.perf_counter()

        self.metrics.frames_in += len(batch)

        # 1) Build alignment images (mono u16), downsample.
        align_imgs: List[np.ndarray] = []
        for item in batch:
            raw = item["raw"]
            fmt = item.get("fmt", "MONO16")
            mono_u16 = extract_align_mono_u16(raw, fmt, bayer_pattern=scfg.bayer_pattern)
            small = downsample_u16(mono_u16, factor=scfg.align_downsample)
            align_imgs.append(small)

        # 2) Choose keyframe within batch: use first, or best by self-consistency.
        key_idx = 0
        key = align_imgs[key_idx]
        self._key_align = key

        # 3) Estimate per-frame shift (dx,dy) and resp vs key.
        dxs = np.zeros(len(batch), dtype=np.float32)
        dys = np.zeros(len(batch), dtype=np.float32)
        resps = np.zeros(len(batch), dtype=np.float32)

        for i, img in enumerate(align_imgs):
            if i == key_idx:
                dxs[i] = 0.0
                dys[i] = 0.0
                resps[i] = 1.0
                continue
            dx, dy, resp = _phasecorr_shift(key, img)
            # shift estimated on downsampled grid -> scale up to full-res pixels
            dxs[i] = dx * scfg.align_downsample
            dys[i] = dy * scfg.align_downsample
            resps[i] = resp

        # 4) Robust filtering (median/MAD) + resp_min.
        med_dx = float(np.median(dxs))
        med_dy = float(np.median(dys))
        mad_dx = _mad(dxs)
        mad_dy = _mad(dys)

        used_indices: List[int] = []
        for i in range(len(batch)):
            if resps[i] < scfg.resp_min:
                continue
            if abs(float(dxs[i]) - med_dx) > scfg.outlier_k_mad * mad_dx:
                continue
            if abs(float(dys[i]) - med_dy) > scfg.outlier_k_mad * mad_dy:
                continue
            used_indices.append(i)

        if len(used_indices) == 0:
            self.metrics.frames_rejected += len(batch)
            return

        self.metrics.frames_rejected += (len(batch) - len(used_indices))

        # 5) Rotation estimation (optional hook): keep 0 for now
        theta_deg = 0.0
        theta_rad = 0.0

        # 6) Accumulate each accepted frame into tiles (drizzle scale)
        scale = scfg.drizzle_scale
        tile_size = scfg.tile_size_out

        # origin: choose (0,0) at first batch keyframe, i.e. key frame centered at positive coords.
        # For simplicity, we use origin = (0,0) in output; tiles can be negative indices.
        # In practice you may want to set origin so initial frame lies in tile (0,0) region.
        # We'll anchor origin by shifting so that frame top-left maps near (0,0) when dx,dy ~0.
        # Here: origin is implicit in tile_u0/tile_v0; we let negative tiles exist.
        # (UI can render a viewport later.)

        for i in used_indices:
            raw = batch[i]["raw"]
            fmt = batch[i].get("fmt", "MONO16")

            dx = float(dxs[i])
            dy = float(dys[i])
            resp = float(resps[i])

            # frame weight: simple mapping (can refine later)
            w_frame = float(np.clip((resp - scfg.resp_min) / max(1e-6, (1.0 - scfg.resp_min)), 0.0, 1.0))
            if w_frame <= 0.0:
                continue

            # Build source image for stacking
            if self.color_mode == "mono":
                src_u16 = extract_align_mono_u16(raw, fmt, bayer_pattern=scfg.bayer_pattern)
                # drizzle "pixfrac" approximation: prefilter a bit when pixfrac is high
                src = src_u16.astype(np.float32, copy=False)
                if scfg.pixfrac > 0.7:
                    src = cv2.GaussianBlur(src, (0, 0), sigmaX=0.6, sigmaY=0.6)
                src_is_color = False
                src_shape = src_u16.shape
            else:
                # Debayer if raw Bayer; if already RGB, accept it
                if fmt.startswith("RAW"):
                    rgb = debayer_cv2(raw, pattern=scfg.bayer_pattern, edge_aware=False)
                elif fmt.startswith("RGB"):
                    rgb = raw
                else:
                    # mono -> fake RGB (replicate)
                    rgb = np.repeat(raw[..., None], 3, axis=2)
                src = rgb.astype(np.float32, copy=False)
                if scfg.pixfrac > 0.7:
                    src = cv2.GaussianBlur(src, (0, 0), sigmaX=0.6, sigmaY=0.6)
                src_is_color = True
                src_shape = (src.shape[0], src.shape[1])

            h, w = src_shape

            # Determine approximate bbox in output (scaled) to know tiles touched.
            # Forward mapping of src corners, assuming small theta:
            # u = scale*(x + dx), v = scale*(y + dy) when theta=0
            # We'll compute bbox conservatively.
            u_min = scale * (0.0 + dx)
            v_min = scale * (0.0 + dy)
            u_max = scale * ((w - 1) + dx)
            v_max = scale * ((h - 1) + dy)

            # Convert to tile indices
            tx0 = int(math.floor(u_min / tile_size))
            ty0 = int(math.floor(v_min / tile_size))
            tx1 = int(math.floor(u_max / tile_size))
            ty1 = int(math.floor(v_max / tile_size))

            now = time.time()
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tile = self.canvas.get_or_create((tx, ty), now)
                    tile.hits += 1

                    tile_u0 = tx * tile_size
                    tile_v0 = ty * tile_size

                    M = _build_dst_to_src_affine(
                        tile_u0=tile_u0,
                        tile_v0=tile_v0,
                        scale=scale,
                        dx=dx,
                        dy=dy,
                        theta_rad=theta_rad,
                    )

                    dsize = (tile_size, tile_size)
                    warped = self.backend.warp_affine_tile(src, M, dsize, is_color=src_is_color)
                    mask = self.backend.warp_mask_tile((h, w), M, dsize) * w_frame

                    if self.color_mode == "mono":
                        tile.sum += warped * mask
                    else:
                        tile.sum += warped * mask[..., None]
                    tile.w += mask

        # Update metrics
        self.metrics.frames_used += len(used_indices)
        self.metrics.tiles_used = self.canvas.tiles_used()
        self.metrics.tiles_evicted = self.canvas.tiles_evicted
        self.metrics.last_resp = float(np.median(resps[used_indices]))
        self.metrics.last_dx = float(np.median(dxs[used_indices]))
        self.metrics.last_dy = float(np.median(dys[used_indices]))
        self.metrics.last_theta_deg = float(theta_deg)

        dt = time.perf_counter() - t0
        if dt > 1e-6:
            # approximate "fps" in terms of frames processed
            self.metrics.stacking_fps = 0.9 * self.metrics.stacking_fps + 0.1 * (len(used_indices) / dt)

        # Preview update (slow)
        now_t = time.time()
        if (now_t - self.metrics.last_preview_t) >= (1.0 / max(1e-6, scfg.preview_hz)):
            self.metrics.last_preview_t = now_t
            self._update_preview_jpeg()

    def _update_preview_jpeg(self) -> None:
        """
        Build a small preview from currently most-used tiles.
        Strategy: pick a tile near median of keys and render its normalized image.
        This keeps it simple; UI can later do a viewport compositor.
        """
        if self.canvas is None or len(self.canvas.tiles) == 0:
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
            # downsample for preview
            u8s = cv2.resize(u8, (u8.shape[1] // 2, u8.shape[0] // 2), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", u8s, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        else:
            img = (tile.sum / w[..., None]).astype(np.float32, copy=False)
            # stretch per-channel (simple) and convert to u8
            u8 = stretch_to_u8(img)  # supports (H,W,3)
            u8s = cv2.resize(u8, (u8.shape[1] // 2, u8.shape[0] // 2), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(u8s, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        with self._preview_lock:
            self._preview_jpeg = jpg.tobytes() if ok else None


# ============================================================
# Worker thread wrapper
# ============================================================

class StackingWorker:
    """
    Owns a StackEngine and a queue.
    Call .enqueue_frame(...) from AppRunner loop (non-blocking),
    and let .thread() consume and process in batches.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.engine = StackEngine(cfg)
        self.engine.configure_from_cfg()

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.stacking.max_queue)
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

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

    def enqueue_frame(self, raw: np.ndarray, fmt: str, t: Optional[float] = None) -> None:
        if not self.engine.enabled:
            return
        item = {"raw": raw, "fmt": fmt, "t": float(t if t is not None else time.time())}
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.engine.metrics.frames_dropped += 1

    def _run(self) -> None:
        batch_size = self.cfg.stacking.batch_size
        while not self._stop.is_set():
            batch: List[Dict[str, Any]] = []
            try:
                # Block briefly for first item
                item = self._q.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                continue

            # Drain up to batch_size quickly
            for _ in range(batch_size - 1):
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            self.engine.step_batch(batch)
