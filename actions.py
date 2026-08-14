# actions.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
import time

from ap_types import Axis


class ActionType(str, Enum):
    # camera
    CAMERA_CONNECT = "CAMERA_CONNECT"
    CAMERA_DISCONNECT = "CAMERA_DISCONNECT"
    CAMERA_SET_PARAM = "CAMERA_SET_PARAM"
    CAMERA_RECORD_RAW = "CAMERA_RECORD_RAW"
    RESET_CAMERA_DEFAULTS = "RESET_CAMERA_DEFAULTS"
    RESET_PREVIEW_DEFAULTS = "RESET_PREVIEW_DEFAULTS"

    # mount
    MOUNT_CONNECT = "MOUNT_CONNECT"
    MOUNT_DISCONNECT = "MOUNT_DISCONNECT"
    MOUNT_SET_MICROSTEPS = "MOUNT_SET_MICROSTEPS"
    MOUNT_MOVE_STEPS = "MOUNT_MOVE_STEPS"
    MOUNT_STOP = "MOUNT_STOP"
    RESET_MOUNT_DEFAULTS = "RESET_MOUNT_DEFAULTS"

    # tracking
    TRACKING_START = "TRACKING_START"
    TRACKING_STOP = "TRACKING_STOP"
    TRACKING_SET_PARAMS = "TRACKING_SET_PARAMS"
    TRACKING_KEYFRAME_RESET = "TRACKING_KEYFRAME_RESET"
    RESET_TRACKING_DEFAULTS = "RESET_TRACKING_DEFAULTS"

    TRACKING_CALIB_AZ = "TRACKING_CALIB_AZ"
    TRACKING_CALIB_ALT = "TRACKING_CALIB_ALT"
    TRACKING_CALIB_RESET = "TRACKING_CALIB_RESET"

    TRACKING_AUTO_RESET = "TRACKING_AUTO_RESET"
    TRACKING_BOOTSTRAP = "TRACKING_BOOTSTRAP"

    # stacking
    STACKING_START = "STACKING_START"
    STACKING_STOP = "STACKING_STOP"
    STACKING_RESET = "STACKING_RESET"
    STACKING_SAVE = "STACKING_SAVE"
    STACKING_SET_PARAMS = "STACKING_SET_PARAMS"
    RESET_STACKING_DEFAULTS = "RESET_STACKING_DEFAULTS"

    # platesolving (OBLIGATORIO)
    PLATESOLVING_RUN = "PLATESOLVING_RUN"
    PLATESOLVING_SET_PARAMS = "PLATESOLVING_SET_PARAMS"
    PLATESOLVING_DOWNLOAD_CURRENT_FIELD = "PLATESOLVING_DOWNLOAD_CURRENT_FIELD"
    RESET_PLATESOLVING_DEFAULTS = "RESET_PLATESOLVING_DEFAULTS"
    MOUNT_SYNC = "MOUNT_SYNC"
    MOUNT_GOTO = "MOUNT_GOTO"

    # goto
    GOTO_CALIBRATE = "GOTO_CALIBRATE"
    GOTO_AUTOCALIBRATE = "GOTO_AUTOCALIBRATE"
    GOTO_FIT_MODEL = "GOTO_FIT_MODEL"
    GOTO_ESTIMATE_ROLL = "GOTO_ESTIMATE_ROLL"
    GOTO_VALIDATE_SAMPLE = "GOTO_VALIDATE_SAMPLE"
    GOTO_RESET = "GOTO_RESET"
    GOTO_CANCEL = "GOTO_CANCEL"
    GOTO_LIST_SAMPLES = "GOTO_LIST_SAMPLES"
    GOTO_PRUNE_OUTLIERS = "GOTO_PRUNE_OUTLIERS"
    GOTO_RESTORE_LAST_LOG = "GOTO_RESTORE_LAST_LOG"

    # live overlay
    LIVE_SEP_SET_PARAMS = "LIVE_SEP_SET_PARAMS"
    EXPECTED_STARS_SET_PARAMS = "EXPECTED_STARS_SET_PARAMS"


@dataclass(frozen=True)
class Action:
    type: ActionType
    payload: Dict[str, Any]
    t: float


def _now() -> float:
    return time.perf_counter()


# -------------------------
# Factories: Camera
# -------------------------
def camera_connect(camera_index: int) -> Action:
    return Action(ActionType.CAMERA_CONNECT, {"camera_index": int(camera_index)}, _now())


def camera_disconnect() -> Action:
    return Action(ActionType.CAMERA_DISCONNECT, {}, _now())


def camera_set_param(name: str, value: Any) -> Action:
    return Action(ActionType.CAMERA_SET_PARAM, {"name": str(name), "value": value}, _now())


def camera_record_raw(duration_s: float = 20.0, out_dir: str = "raw_output", basename: str | None = None) -> Action:
    payload: Dict[str, Any] = {
        "duration_s": float(duration_s),
        "out_dir": str(out_dir),
    }
    if basename:
        payload["basename"] = str(basename)
    return Action(ActionType.CAMERA_RECORD_RAW, payload, _now())


def camera_reset_defaults() -> Action:
    return Action(ActionType.RESET_CAMERA_DEFAULTS, {}, _now())


def preview_reset_defaults() -> Action:
    return Action(ActionType.RESET_PREVIEW_DEFAULTS, {}, _now())


# -------------------------
# Factories: Mount
# -------------------------
def mount_connect(port: str, baudrate: int) -> Action:
    return Action(ActionType.MOUNT_CONNECT, {"port": str(port), "baudrate": int(baudrate)}, _now())


def mount_disconnect() -> Action:
    return Action(ActionType.MOUNT_DISCONNECT, {}, _now())


def mount_set_microsteps(az_div: int, alt_div: int) -> Action:
    return Action(
        ActionType.MOUNT_SET_MICROSTEPS,
        {"az_div": int(az_div), "alt_div": int(alt_div)},
        _now(),
    )


def mount_move_steps(
    axis: Axis,
    direction: int,
    steps: int,
    delay_us: int,
    profile: str = "smooth",
) -> Action:
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1")
    profile_value = str(profile or "smooth").strip().lower()
    if profile_value not in {"smooth", "direct"}:
        raise ValueError("profile must be smooth or direct")
    return Action(
        ActionType.MOUNT_MOVE_STEPS,
        {
            "axis": axis.value,
            "direction": int(direction),
            "steps": int(steps),
            "delay_us": int(delay_us),
            "profile": profile_value,
        },
        _now(),
    )


def mount_stop() -> Action:
    return Action(ActionType.MOUNT_STOP, {}, _now())


def mount_reset_defaults() -> Action:
    return Action(ActionType.RESET_MOUNT_DEFAULTS, {}, _now())


# -------------------------
# Factories: Tracking
# -------------------------
def tracking_start() -> Action:
    return Action(ActionType.TRACKING_START, {}, _now())


def tracking_stop() -> Action:
    return Action(ActionType.TRACKING_STOP, {}, _now())


def tracking_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.TRACKING_SET_PARAMS, dict(kwargs), _now())


def tracking_keyframe_reset() -> Action:
    return Action(ActionType.TRACKING_KEYFRAME_RESET, {}, _now())


def tracking_reset_defaults() -> Action:
    return Action(ActionType.RESET_TRACKING_DEFAULTS, {}, _now())


def tracking_calib_az() -> Action:
    return Action(ActionType.TRACKING_CALIB_AZ, {}, _now())


def tracking_calib_alt() -> Action:
    return Action(ActionType.TRACKING_CALIB_ALT, {}, _now())


def tracking_calib_reset() -> Action:
    return Action(ActionType.TRACKING_CALIB_RESET, {}, _now())


def tracking_auto_reset() -> Action:
    return Action(ActionType.TRACKING_AUTO_RESET, {}, _now())


def tracking_bootstrap() -> Action:
    return Action(ActionType.TRACKING_BOOTSTRAP, {}, _now())


def goto_autocalibrate(params: Dict[str, Any] | None = None) -> Action:
    return Action(ActionType.GOTO_AUTOCALIBRATE, {"params": params or {}}, _now())


def goto_estimate_roll(params: Dict[str, Any] | None = None) -> Action:
    return Action(ActionType.GOTO_ESTIMATE_ROLL, {"params": params or {}}, _now())


def goto_validate_sample(result: Any) -> Action:
    return Action(ActionType.GOTO_VALIDATE_SAMPLE, {"result": result}, _now())


# -------------------------
# Factories: Stacking
# -------------------------
def stacking_start() -> Action:
    return Action(ActionType.STACKING_START, {}, _now())


def stacking_stop() -> Action:
    return Action(ActionType.STACKING_STOP, {}, _now())


def stacking_reset() -> Action:
    return Action(ActionType.STACKING_RESET, {}, _now())


def stacking_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.STACKING_SET_PARAMS, dict(kwargs), _now())


def stacking_reset_defaults() -> Action:
    return Action(ActionType.RESET_STACKING_DEFAULTS, {}, _now())


def stacking_save(out_dir: str, basename: str, fmt: str) -> Action:
    return Action(
        ActionType.STACKING_SAVE,
        {"out_dir": str(out_dir), "basename": str(basename), "fmt": str(fmt)},
        _now(),
    )


# -------------------------
# Factories: Platesolving
# -------------------------
def platesolving_run(target: Any, **kwargs: Any) -> Action:
    payload = {"target": target}
    payload.update(dict(kwargs))
    return Action(ActionType.PLATESOLVING_RUN, payload, _now())

def platesolving_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.PLATESOLVING_SET_PARAMS, dict(kwargs), _now())


def platesolving_download_current_field(radius_deg: float | None = None) -> Action:
    payload: Dict[str, Any] = {}
    if radius_deg is not None:
        payload["radius_deg"] = float(radius_deg)
    return Action(ActionType.PLATESOLVING_DOWNLOAD_CURRENT_FIELD, payload, _now())


def platesolving_reset_defaults() -> Action:
    return Action(ActionType.RESET_PLATESOLVING_DEFAULTS, {}, _now())


# -------------------------
# Factories: GoTo
# -------------------------
def mount_sync() -> Action:
    return Action(ActionType.MOUNT_SYNC, {}, _now())


def mount_goto(target: Any, **kwargs: Any) -> Action:
    payload = {"target": target}
    payload.update(dict(kwargs))
    return Action(ActionType.MOUNT_GOTO, payload, _now())

# -------------------------
# Factories: GoTo (extras)
# -------------------------

def goto_calibrate(params: Dict[str, Any]) -> Action:
    return Action(ActionType.GOTO_CALIBRATE, {"params": dict(params)}, _now())

def goto_fit_model(params: Dict[str, Any] | None = None) -> Action:
    return Action(ActionType.GOTO_FIT_MODEL, {"params": params or {}}, _now())


def goto_reset() -> Action:
    return Action(ActionType.GOTO_RESET, {}, _now())


def goto_cancel() -> Action:
    return Action(ActionType.GOTO_CANCEL, {}, _now())


def goto_list_samples(params: Dict[str, Any] | None = None) -> Action:
    return Action(ActionType.GOTO_LIST_SAMPLES, {"params": params or {}}, _now())


def goto_prune_outliers(params: Dict[str, Any] | None = None) -> Action:
    return Action(ActionType.GOTO_PRUNE_OUTLIERS, {"params": params or {}}, _now())


def goto_restore_last_log() -> Action:
    return Action(ActionType.GOTO_RESTORE_LAST_LOG, {}, _now())


# -------------------------
# Factories: Live overlay (SEP)
# -------------------------
def live_sep_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.LIVE_SEP_SET_PARAMS, dict(kwargs), _now())


def expected_stars_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.EXPECTED_STARS_SET_PARAMS, dict(kwargs), _now())
