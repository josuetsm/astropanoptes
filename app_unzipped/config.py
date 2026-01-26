from __future__ import annotations

from dataclasses import dataclass, field

from config import PlatesolveConfig as _PlatesolveConfig


@dataclass
class HotPixelsConfig:
    enabled: bool = False
    base_ksize: int = 3
    mask_enabled_for_stacking: bool = False
    mask_path_base: str = "./hotpixel_mask"
    mask_ksize: int = 3
    thr_k: float = 8.0
    min_hits_frac: float = 0.7
    max_component_area: int = 4
    calib_frames: int = 30


@dataclass
class PlatesolveConfig(_PlatesolveConfig):
    N_seed: int = 3
    search_radius_deg: float | None = 1.0


@dataclass
class AppConfig:
    hotpixels: HotPixelsConfig = field(default_factory=HotPixelsConfig)
    platesolve: PlatesolveConfig = field(default_factory=PlatesolveConfig)
