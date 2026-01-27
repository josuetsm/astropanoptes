# camera_poa.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from logging_utils import log_error

try:
    import pyPOACamera  # user-provided SDK wrapper
except Exception as exc:
    log_error(None, "Camera: failed to import pyPOACamera", exc)
    pyPOACamera = None
    POA_UNAVAILABLE_REASON = str(exc)
else:
    POA_UNAVAILABLE_REASON = ""

from ap_types import Frame
from config import CameraConfig, PreviewConfig


# -------------------------
# Utilities
# -------------------------
def _perf() -> float:
    return time.perf_counter()


def _sleep_s(s: float) -> None:
    # sleep mínimo para no quemar CPU en polling
    if s <= 0:
        return
    time.sleep(s)


def _require_poa() -> None:
    if pyPOACamera is None:
        raise RuntimeError(f"pyPOACamera no disponible: {POA_UNAVAILABLE_REASON}")


def _imgfmt_from_str(name: str) -> pyPOACamera.POAImgFormat:
    _require_poa()
    n = (name or "").strip().upper()
    if n in ("RAW16", "POA_RAW16"):
        return pyPOACamera.POAImgFormat.POA_RAW16
    # fallback seguro (RAW16 único permitido)
    return pyPOACamera.POAImgFormat.POA_RAW16


def _bayerpattern_to_str(bp: Any) -> str:
    # bp puede ser Enum POABayerPattern o int
    try:
        if hasattr(bp, "name"):
            return str(bp.name)
        return str(bp)
    except Exception as exc:
        log_error(None, "Camera: failed to decode bayer pattern", exc, throttle_s=10.0, throttle_key="camera_bayerpattern")
        return "UNKNOWN"


def _bytes_per_px(img_fmt: pyPOACamera.POAImgFormat) -> int:
    if img_fmt in (pyPOACamera.POAImgFormat.POA_RAW8, pyPOACamera.POAImgFormat.POA_MONO8):
        return 1
    if img_fmt == pyPOACamera.POAImgFormat.POA_RAW16:
        return 2
    if img_fmt == pyPOACamera.POAImgFormat.POA_RGB24:
        return 3
    # fallback
    return 2


# -------------------------
# Device wrapper
# -------------------------
@dataclass
class CameraInfo:
    camera_id: int
    model: str
    sensor: str
    is_color: bool
    is_usb3: bool
    max_w: int
    max_h: int
    bit_depth: int
    pixel_um: float
    bayer_pattern: str
    bins: Tuple[int, ...]
    img_formats: Tuple[str, ...]


class POACameraDevice:
    """
    Wrapper de bajo nivel sobre pyPOACamera.
    - No hace OpenCV ni preview.
    - Solo I/O + configuración + lectura a buffer prealocado.
    """

    def __init__(self) -> None:
        self._cam_id: Optional[int] = None
        self._opened = False
        self._started = False
        self._info: Optional[CameraInfo] = None

        self._w: int = 0
        self._h: int = 0
        self._fmt: pyPOACamera.POAImgFormat = pyPOACamera.POAImgFormat.POA_RAW16
        self._bytes_per_px: int = 2

    @property
    def cam_id(self) -> int:
        if self._cam_id is None:
            raise RuntimeError("Camera not opened")
        return self._cam_id

    @property
    def info(self) -> CameraInfo:
        if self._info is None:
            raise RuntimeError("CameraInfo not available (open first)")
        return self._info

    def open(self, index: int = 0) -> CameraInfo:
        _require_poa()
        n = pyPOACamera.GetCameraCount()
        if n <= 0:
            raise RuntimeError("No se detectan cámaras (GetCameraCount=0).")
        if index < 0 or index >= n:
            raise ValueError(f"camera_index fuera de rango: {index} (count={n})")

        err, props = pyPOACamera.GetCameraProperties(int(index))
        if err != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(f"GetCameraProperties falló (err={err}).")

        cam_id = int(props.cameraID)

        if pyPOACamera.OpenCamera(cam_id) != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError("OpenCamera falló.")
        if pyPOACamera.InitCamera(cam_id) != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError("InitCamera falló.")

        self._cam_id = cam_id
        self._opened = True

        # Construir info (usando properties del wrapper)
        model = props.cameraModelName.decode(errors="ignore").strip("\x00").strip()
        sensor = props.sensorModelName.decode(errors="ignore").strip("\x00").strip()
        is_color = bool(int(props.isColorCamera))
        is_usb3 = bool(int(props.isUSB3Speed))
        max_w = int(props.maxWidth)
        max_h = int(props.maxHeight)
        bit_depth = int(props.bitDepth)
        pixel_um = float(props.pixelSize)
        bayer_pattern = _bayerpattern_to_str(props.bayerPattern)  # property

        bins = tuple(int(x) for x in props.bins)
        img_formats = tuple(fmt.name for fmt in props.imgFormats)

        self._info = CameraInfo(
            camera_id=cam_id,
            model=model,
            sensor=sensor,
            is_color=is_color,
            is_usb3=is_usb3,
            max_w=max_w,
            max_h=max_h,
            bit_depth=bit_depth,
            pixel_um=pixel_um,
            bayer_pattern=bayer_pattern,
            bins=bins,
            img_formats=img_formats,
        )
        return self._info

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.stop()
        except Exception as exc:
            log_error(None, "Camera: stop failed during close", exc, throttle_s=5.0, throttle_key="camera_stop_close")
        try:
            err = pyPOACamera.CloseCamera(self.cam_id)
            if err != pyPOACamera.POAErrors.POA_OK:
                log_error(None, f"Camera: CloseCamera failed (err={err})", throttle_s=5.0, throttle_key="camera_close")
        finally:
            self._opened = False
            self._cam_id = None
            self._info = None

    def configure(self, cfg: CameraConfig, *, force_no_binning: bool = True) -> None:
        """
        Configura ROI/bin/format/exp/gain.

        Política solicitada:
        - Para stacking, NO queremos binning en la captura raw.
          Por eso force_no_binning=True fuerza binning=1.
        """
        if not self._opened:
            raise RuntimeError("Camera not opened")

        _require_poa()
        pyPOACamera.StopExposure(self.cam_id)

        # ROI
        if cfg.use_roi:
            x, y, w, h = int(cfg.roi_x), int(cfg.roi_y), int(cfg.roi_w), int(cfg.roi_h)
        else:
            x, y = 0, 0
            w, h = self.info.max_w, self.info.max_h

        # binning
        bin_hw = 1 if force_no_binning else int(cfg.binning)
        if bin_hw < 1:
            bin_hw = 1

        img_fmt = pyPOACamera.POAImgFormat.POA_RAW16

        # aplicar
        pyPOACamera.SetImageStartPos(self.cam_id, x, y)
        pyPOACamera.SetImageSize(self.cam_id, w, h)
        pyPOACamera.SetImageBin(self.cam_id, bin_hw)
        pyPOACamera.SetImageFormat(self.cam_id, img_fmt)

        # leer tamaño final
        _, ww, hh = pyPOACamera.GetImageSize(self.cam_id)
        self._w, self._h = int(ww), int(hh)

        self._fmt = img_fmt
        self._bytes_per_px = _bytes_per_px(img_fmt)

        # exposure/gain
        # pyPOACamera.SetExp usa microsegundos según tu snippet: SetExp(cam_id, int(EXP_MS * 1000), False)
        # eso sugiere unidad = microsegundos.
        exp_us = int(float(cfg.exp_ms) * 1000.0)
        pyPOACamera.SetExp(self.cam_id, exp_us, False)

        pyPOACamera.SetGain(self.cam_id, int(cfg.gain), bool(cfg.auto_gain))

        # Gamma / Debayer: depende de SDK; se controlará desde otro módulo si procede.
        # Por ahora se deja en metadata; no forzamos aquí.

    def start(self) -> None:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        if self._started:
            return
        e = pyPOACamera.StartExposure(self.cam_id, False)
        if e != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(f"StartExposure falló (err={e}).")
        self._started = True

    def stop(self) -> None:
        if not self._opened or not self._started:
            return
        try:
            pyPOACamera.StopExposure(self.cam_id)
        finally:
            self._started = False

    def get_size(self) -> Tuple[int, int]:
        return self._w, self._h

    def get_format(self) -> pyPOACamera.POAImgFormat:
        return self._fmt

    def bytes_per_px(self) -> int:
        return self._bytes_per_px

    def wait_ready(self, ready_sleep_s: float = 0.0005) -> None:
        """
        Poll de ImageReady. Mantener sleep mínimo para no saturar CPU.
        """
        while True:
            _, ready = pyPOACamera.ImageReady(self.cam_id)
            if ready:
                return
            _sleep_s(ready_sleep_s)

    def read_into(self, buf_u8: np.ndarray, timeout_ms: int = 1000) -> None:
        """
        Lee un frame al buffer prealocado (uint8 plano).
        """
        e = pyPOACamera.GetImageData(self.cam_id, buf_u8, int(timeout_ms))
        if e != pyPOACamera.POAErrors.POA_OK:
            raise RuntimeError(f"GetImageData falló (err={e}).")


# -------------------------
# High-rate capture thread
# -------------------------
class CameraStream:
    """
    Stream de captura a máxima FPS, con ring-buffer de N buffers.
    - No hace JPEG, no hace UI.
    - Publica el último frame disponible vía latest().

    Nota: el pipeline solo usa RAW16; u8_view queda opcional para compatibilidad.
    """

    def __init__(self, ring: int = 3) -> None:
        if ring < 2:
            ring = 2
        self._ring = ring

        self._dev: Optional[POACameraDevice] = None
        self._cfg: Optional[CameraConfig] = None
        self._preview_cfg: Optional[PreviewConfig] = None

        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None

        # stats
        self._fps_capture: float = 0.0
        self._dropped: int = 0
        self._seq: int = 0

        # buffers
        self._bufs: Optional[np.ndarray] = None  # shape (ring, bytes)
        self._buf_bytes: int = 0

    def start(self, dev: POACameraDevice, cfg: CameraConfig, preview_cfg: PreviewConfig) -> None:
        if self._thr is not None:
            return

        self._dev = dev
        self._cfg = cfg
        self._preview_cfg = preview_cfg

        # IMPORTANTE: política pedida: sin binning en captura raw
        dev.configure(cfg, force_no_binning=True)
        dev.start()

        w, h = dev.get_size()
        bpp = dev.bytes_per_px()
        self._buf_bytes = int(w * h * bpp)

        # ring buffer: uint8 plano
        self._bufs = np.zeros((self._ring, self._buf_bytes), dtype=np.uint8)

        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="CameraStream", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        thr = self._thr
        if thr is not None:
            thr.join(timeout=2.0)
        self._thr = None

        if self._dev is not None:
            try:
                self._dev.stop()
            except Exception as exc:
                log_error(None, "CameraStream: device stop failed", exc, throttle_s=5.0, throttle_key="camera_stream_stop")

        self._dev = None
        self._cfg = None
        self._preview_cfg = None
        self._bufs = None
        self._latest = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def stats(self) -> Dict[str, Any]:
        return {
            "fps_capture": float(self._fps_capture),
            "dropped": int(self._dropped),
            "seq": int(self._seq),
        }

    def camera_info(self) -> Optional[CameraInfo]:
        if self._dev is None:
            return None
        try:
            return self._dev.info
        except Exception as exc:
            log_error(None, "CameraStream: failed to read camera info", exc, throttle_s=10.0, throttle_key="camera_info")
            return None

    def _run(self) -> None:
        assert self._dev is not None
        assert self._preview_cfg is not None
        assert self._bufs is not None

        dev = self._dev
        ready_sleep_s = float(self._preview_cfg.ready_sleep_s)
        w, h = dev.get_size()
        fmt = dev.get_format()
        bpp = dev.bytes_per_px()

        # metadata común
        info = dev.info
        meta_common = {
            "camera_model": info.model,
            "sensor_model": info.sensor,
            "is_color": info.is_color,
            "is_usb3": info.is_usb3,
            "bit_depth": info.bit_depth,
            "pixel_um": info.pixel_um,
            "bayer_pattern": info.bayer_pattern,
            "roi": (0, 0, w, h) if not self._cfg.use_roi else (self._cfg.roi_x, self._cfg.roi_y, w, h),
            "binning_hw": 1,  # forzado
            "img_format": fmt.name,
        }

        # FPS capture
        t0 = _perf()
        n = 0

        ring_i = 0

        while not self._stop.is_set():
            # esperar ready y leer
            dev.wait_ready(ready_sleep_s=ready_sleep_s)

            buf = self._bufs[ring_i]
            t_cap = _perf()
            try:
                dev.read_into(buf, timeout_ms=1000)
            except Exception as exc:
                # si falla, contamos como drop y seguimos
                log_error(None, "CameraStream: read failed", exc, throttle_s=5.0, throttle_key="camera_read")
                self._dropped += 1
                continue

            self._seq += 1
            seq = self._seq

            # construir vistas sin copias grandes cuando se puede
            if fmt != pyPOACamera.POAImgFormat.POA_RAW16:
                raise RuntimeError(f"CameraStream: unexpected format {fmt} (RAW16 required)")

            # reinterpretación little-endian
            u16 = buf[: w * h * 2].view("<u2").reshape(h, w)
            raw = u16
            u8_view = None

            # Importante: raw/u8_view apuntan a memoria del ring buffer.
            # El consumidor debe copiar si necesita persistir.
            fr = Frame(
                t_capture=t_cap,
                seq=seq,
                w=w,
                h=h,
                fmt=fmt.name,
                raw=raw,
                u8_view=u8_view,
                meta=dict(meta_common),
            )

            with self._lock:
                self._latest = fr

            # stats fps
            n += 1
            now = _perf()
            if (now - t0) >= 1.0:
                self._fps_capture = float(n / (now - t0))
                t0 = now
                n = 0

            # avanzar ring
            ring_i = (ring_i + 1) % self._ring
