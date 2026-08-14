from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

import camera_poa
from camera_poa import POACameraDevice
from config import CameraConfig


CalibrationKind = Literal["dark", "blank"]
CombineMethod = Literal["median", "mean"]


@dataclass(frozen=True)
class CalibrationRequest:
    kind: CalibrationKind
    exposure_ms: float = 100.0
    gain: int = 360
    offset: int = 350
    frames: int = 32
    warmup_frames: int = 2
    camera_index: int = 0
    combine: CombineMethod = "median"

    def validate(self) -> None:
        if self.kind not in ("dark", "blank"):
            raise ValueError(f"Tipo de calibración inválido: {self.kind!r}")
        if not np.isfinite(self.exposure_ms) or self.exposure_ms <= 0:
            raise ValueError("La exposición debe ser mayor que cero.")
        if self.gain < 0:
            raise ValueError("La ganancia no puede ser negativa.")
        if self.offset < 0:
            raise ValueError("El offset no puede ser negativo.")
        if self.frames < 1:
            raise ValueError("Se necesita al menos un frame.")
        if self.warmup_frames < 0:
            raise ValueError("Los frames de calentamiento no pueden ser negativos.")
        if self.camera_index < 0:
            raise ValueError("El índice de cámara no puede ser negativo.")
        if self.combine not in ("median", "mean"):
            raise ValueError(f"Método de combinación inválido: {self.combine!r}")


def infer_raw_shift(raw: np.ndarray, bit_depth: int) -> tuple[int, float]:
    """Detecta si los bits nativos están alineados a la izquierda en RAW16.

    Player One entrega RAW16 en un contenedor uint16. En la Mars-C de 12 bits,
    cada ADU suele estar desplazado cuatro bits a la izquierda. No se asume ese
    detalle a ciegas: se comprueba que los bits bajos estén vacíos.
    """

    arr = np.asarray(raw)
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError("Se esperaba un frame RAW16 uint16 de dos dimensiones.")

    extra_bits = max(0, 16 - int(bit_depth))
    if extra_bits == 0:
        return 0, 1.0

    sample = arr[::8, ::8]
    low_mask = np.uint16((1 << extra_bits) - 1)
    aligned_fraction = float(np.mean((sample & low_mask) == 0))
    shift = extra_bits if aligned_fraction >= 0.999 else 0
    return int(shift), aligned_fraction


def frame_statistics(
    raw: np.ndarray,
    *,
    bit_depth: int,
    raw_shift: int,
) -> dict[str, float | int]:
    arr = np.asarray(raw)
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError("Se esperaba un frame RAW16 uint16 de dos dimensiones.")

    scale = float(1 << max(0, int(raw_shift)))
    native_max = float((1 << int(bit_depth)) - 1)
    # The Mars-C RAW16 path tops out one code below the nominal 12-bit value:
    # 4094 ADU is encoded as 65504 after the four-bit left shift. Treat both
    # 4094 and 4095 as clipped instead of silently accepting an all-white frame.
    saturation_level = max(0.0, native_max - 1.0)
    sample = arr[::8, ::8].astype(np.float64) / scale
    percentiles = np.percentile(sample, [1.0, 50.0, 99.0])

    return {
        "raw_min": int(arr.min()),
        "raw_max": int(arr.max()),
        "native_mean_adu": float(np.mean(arr, dtype=np.float64) / scale),
        "native_p01_adu": float(percentiles[0]),
        "native_median_adu": float(percentiles[1]),
        "native_p99_adu": float(percentiles[2]),
        "floor_fraction": float(np.mean(arr == 0)),
        "saturation_level_adu": float(saturation_level),
        "saturation_fraction": float(
            np.mean((arr.astype(np.float64) / scale) >= saturation_level)
        ),
    }


def combine_frames(
    frame_paths: Sequence[Path],
    *,
    method: CombineMethod = "median",
    chunk_rows: int = 64,
) -> np.ndarray:
    """Combina frames por franjas para mantener acotado el uso de memoria."""

    if not frame_paths:
        raise ValueError("No hay frames para combinar.")
    if method not in ("median", "mean"):
        raise ValueError(f"Método de combinación inválido: {method!r}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows debe ser positivo.")

    arrays = [np.load(Path(path), mmap_mode="r") for path in frame_paths]
    shape = arrays[0].shape
    if len(shape) != 2 or arrays[0].dtype != np.uint16:
        raise ValueError("Los frames deben ser matrices uint16 de dos dimensiones.")
    if any(arr.shape != shape or arr.dtype != np.uint16 for arr in arrays[1:]):
        raise ValueError("Todos los frames deben tener el mismo tamaño y dtype uint16.")

    master = np.empty(shape, dtype=np.uint16)
    height = int(shape[0])
    for y0 in range(0, height, int(chunk_rows)):
        y1 = min(height, y0 + int(chunk_rows))
        block = np.stack([arr[y0:y1] for arr in arrays], axis=0)
        if method == "median":
            combined = np.median(block, axis=0)
        else:
            combined = np.mean(block, axis=0, dtype=np.float64)
        master[y0:y1] = np.floor(combined + 0.5).astype(np.uint16)
    return master


def _number_slug(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def create_session_directory(
    output_root: Path,
    request: CalibrationRequest,
    *,
    now: datetime | None = None,
) -> Path:
    request.validate()
    current = (now or datetime.now().astimezone()).astimezone()
    stamp = current.strftime("%Y%m%d_%H%M%S")
    name = (
        f"{stamp}_{request.kind}_{_number_slug(request.exposure_ms)}ms_"
        f"gain{request.gain}_offset{request.offset}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Nombre de sesión inseguro: {name!r}")
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    session = root / name
    session.mkdir(exist_ok=False)
    (session / "frames").mkdir()
    return session


def _save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.uint16), allow_pickle=False)
    os.replace(temporary, path)


def _camera_temperature_c(device: POACameraDevice) -> float | None:
    try:
        status, temperature = camera_poa.pyPOACamera.GetCameraTEMP(device.cam_id)
        if status == camera_poa.pyPOACamera.POAErrors.POA_OK:
            value = float(temperature)
            if np.isfinite(value):
                return value
    except Exception:
        pass
    return None


def _dropped_frames(device: POACameraDevice) -> int | None:
    try:
        status, dropped = camera_poa.pyPOACamera.GetDroppedImagesCount(device.cam_id)
        if status == camera_poa.pyPOACamera.POAErrors.POA_OK:
            return int(dropped)
    except Exception:
        pass
    return None


def _write_fits(
    path: Path,
    master: np.ndarray,
    *,
    metadata: dict[str, object],
) -> None:
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.asarray(master, dtype=np.uint16))
    header = hdu.header
    header["FRAMETYP"] = (str(metadata["kind"]).upper(), "Calibration frame type")
    header["EXPTIME"] = (float(metadata["exposure_ms"]) / 1000.0, "Exposure time [s]")
    header["GAIN"] = (int(metadata["gain"]), "Camera gain")
    header["OFFSET"] = (int(metadata["offset"]), "Camera offset")
    header["NFRAMES"] = (int(metadata["frames_captured"]), "Combined frame count")
    header["COMBINE"] = (str(metadata["combine"]).upper(), "Combination method")
    header["CAMMODEL"] = (str(metadata["camera_model"]), "Camera model")
    header["SENSOR"] = (str(metadata["sensor_model"]), "Sensor model")
    header["BITDEPTH"] = (int(metadata["bit_depth"]), "Native ADC bit depth")
    header["RAWSHIFT"] = (int(metadata["raw_shift"]), "RAW16 left alignment bits")
    header["DATE-OBS"] = (str(metadata["started_at"]), "Series start")
    if metadata.get("temperature_start_c") is not None:
        header["TEMPSTRT"] = (float(metadata["temperature_start_c"]), "Start temperature [C]")
    if metadata.get("temperature_end_c") is not None:
        header["TEMPEND"] = (float(metadata["temperature_end_c"]), "End temperature [C]")
    hdu.writeto(path, overwrite=False, checksum=True)


def _write_quicklook(path: Path, master: np.ndarray, raw_shift: int) -> None:
    import cv2

    image = master.astype(np.float32) / float(1 << max(0, int(raw_shift)))
    sample = image[::8, ::8]
    low, high = (float(x) for x in np.percentile(sample, [1.0, 99.5]))
    if high <= low:
        preview = np.zeros(image.shape, dtype=np.uint8)
    else:
        preview = np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"No se pudo guardar el quicklook: {path}")


def capture_calibration_series(
    request: CalibrationRequest,
    *,
    output_root: Path = Path("calibration_frames"),
    device_factory: Callable[[], POACameraDevice] = POACameraDevice,
    progress: Callable[[str], None] | None = print,
) -> Path:
    request.validate()
    session = create_session_directory(Path(output_root), request)
    frame_dir = session / "frames"
    started_at = datetime.now().astimezone()
    device = device_factory()
    frame_paths: list[Path] = []
    per_frame: list[dict[str, float | int]] = []
    info = None
    cfg = CameraConfig(
        camera_index=int(request.camera_index),
        use_roi=False,
        binning=1,
        img_format="RAW16",
        exp_ms=float(request.exposure_ms),
        gain=int(request.gain),
        offset=int(request.offset),
        auto_gain=False,
    )
    raw_shift: int | None = None
    alignment_fraction: float | None = None
    temperature_start: float | None = None
    temperature_end: float | None = None
    dropped: int | None = None

    try:
        info = device.open(int(request.camera_index))
        device.configure(cfg, force_no_binning=True)
        temperature_start = _camera_temperature_c(device)
        device.start()
        width, height = device.get_size()
        buffer = np.empty(width * height * device.bytes_per_px(), dtype=np.uint8)
        timeout_s = max(5.0, float(cfg.exp_ms) / 1000.0 + 5.0)
        timeout_ms = int(np.ceil(timeout_s * 1000.0))

        total = int(request.warmup_frames) + int(request.frames)
        for sequence in range(total):
            if not device.wait_ready(timeout_s=timeout_s):
                raise TimeoutError(f"Timeout esperando el frame {sequence + 1}/{total}.")
            device.read_into(buffer, timeout_ms=timeout_ms)
            if sequence < int(request.warmup_frames):
                if progress is not None:
                    progress(f"Calentamiento {sequence + 1}/{request.warmup_frames}")
                continue

            frame = buffer.view("<u2").reshape(height, width).copy()
            if raw_shift is None:
                raw_shift, alignment_fraction = infer_raw_shift(frame, int(info.bit_depth))

            frame_number = sequence - int(request.warmup_frames) + 1
            path = frame_dir / f"{request.kind}_{frame_number:04d}.npy"
            _save_npy(path, frame)
            frame_paths.append(path)
            stats = frame_statistics(
                frame,
                bit_depth=int(info.bit_depth),
                raw_shift=int(raw_shift),
            )
            stats["frame"] = int(frame_number)
            per_frame.append(stats)
            if progress is not None:
                progress(
                    f"{request.kind} {frame_number:04d}/{request.frames:04d}  "
                    f"mean={stats['native_mean_adu']:.3f} ADU  "
                    f"median={stats['native_median_adu']:.3f} ADU"
                )

            saturation_window = per_frame[-3:]
            if (
                request.kind == "blank"
                and len(saturation_window) == 3
                and all(float(item["saturation_fraction"]) >= 0.95 for item in saturation_window)
            ):
                raise RuntimeError(
                    "Tres blanks consecutivos tienen al menos 95% de píxeles saturados; "
                    "se aborta la serie para corregir iluminación o exposición."
                )

        temperature_end = _camera_temperature_c(device)
        dropped = _dropped_frames(device)
    except Exception as exc:
        failure = {
            "status": "failed",
            "kind": request.kind,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "frames_captured": len(frame_paths),
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now().astimezone().isoformat(),
        }
        (session / "capture_error.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        device.close()

    if info is None or raw_shift is None or alignment_fraction is None:
        raise RuntimeError("La captura terminó sin frames válidos.")

    master = combine_frames(frame_paths, method=request.combine)
    master_stats = frame_statistics(
        master,
        bit_depth=int(info.bit_depth),
        raw_shift=int(raw_shift),
    )
    master_npy = session / f"master_{request.kind}.npy"
    master_fits = session / f"master_{request.kind}.fits"
    quicklook = session / f"master_{request.kind}_quicklook.png"
    _save_npy(master_npy, master)

    finished_at = datetime.now().astimezone()
    metadata: dict[str, object] = {
        "status": "complete",
        "kind": request.kind,
        "camera_model": info.model,
        "sensor_model": info.sensor,
        "width": int(master.shape[1]),
        "height": int(master.shape[0]),
        "bit_depth": int(info.bit_depth),
        "bayer_pattern": info.bayer_pattern,
        "raw_format": "RAW16",
        "raw_shift": int(raw_shift),
        "raw_alignment_fraction": float(alignment_fraction),
        "exposure_ms": float(cfg.exp_ms),
        "gain": int(cfg.gain),
        "offset": int(cfg.offset),
        "frames_requested": int(request.frames),
        "frames_captured": len(frame_paths),
        "warmup_frames": int(request.warmup_frames),
        "combine": request.combine,
        "temperature_start_c": temperature_start,
        "temperature_end_c": temperature_end,
        "dropped_frames": dropped,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "master_statistics": master_stats,
        "frame_statistics": per_frame,
        "request": asdict(request),
        "files": {
            "frames": "frames/*.npy",
            "master_npy": master_npy.name,
            "master_fits": master_fits.name,
            "quicklook_png": quicklook.name,
        },
    }
    _write_fits(master_fits, master, metadata=metadata)
    _write_quicklook(quicklook, master, int(raw_shift))
    (session / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return session


__all__ = [
    "CalibrationRequest",
    "capture_calibration_series",
    "combine_frames",
    "create_session_directory",
    "frame_statistics",
    "infer_raw_shift",
]
