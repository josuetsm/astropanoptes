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

    # mount
    MOUNT_CONNECT = "MOUNT_CONNECT"
    MOUNT_DISCONNECT = "MOUNT_DISCONNECT"
    MOUNT_SET_MICROSTEPS = "MOUNT_SET_MICROSTEPS"
    MOUNT_MOVE_STEPS = "MOUNT_MOVE_STEPS"
    MOUNT_STOP = "MOUNT_STOP"

    # tracking
    TRACKING_START = "TRACKING_START"
    TRACKING_STOP = "TRACKING_STOP"
    TRACKING_SET_PARAMS = "TRACKING_SET_PARAMS"
    TRACKING_KEYFRAME_RESET = "TRACKING_KEYFRAME_RESET"

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

    # platesolve (OBLIGATORIO)
    PLATESOLVE_RUN = "PLATESOLVE_RUN"
    PLATESOLVE_SET_PARAMS = "PLATESOLVE_SET_PARAMS"
    MOUNT_SYNC = "MOUNT_SYNC"
    MOUNT_GOTO = "MOUNT_GOTO"

    # goto
    GOTO_CALIBRATE = "GOTO_CALIBRATE"
    GOTO_CANCEL = "GOTO_CANCEL"


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


def mount_move_steps(axis: Axis, direction: int, steps: int, delay_us: int) -> Action:
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1")
    return Action(
        ActionType.MOUNT_MOVE_STEPS,
        {
            "axis": axis.value,
            "direction": int(direction),
            "steps": int(steps),
            "delay_us": int(delay_us),
        },
        _now(),
    )


def mount_stop() -> Action:
    return Action(ActionType.MOUNT_STOP, {}, _now())


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


def stacking_save(out_dir: str, basename: str, fmt: str) -> Action:
    return Action(
        ActionType.STACKING_SAVE,
        {"out_dir": str(out_dir), "basename": str(basename), "fmt": str(fmt)},
        _now(),
    )


# -------------------------
# Factories: PlateSolve
# -------------------------
def platesolve_run(source: str, target: Any, **kwargs: Any) -> Action:
    payload = {"source": str(source), "target": target}
    payload.update(dict(kwargs))
    return Action(ActionType.PLATESOLVE_RUN, payload, _now())

def platesolve_set_params(**kwargs: Any) -> Action:
    return Action(ActionType.PLATESOLVE_SET_PARAMS, dict(kwargs), _now())


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


def goto_cancel() -> Action:
    return Action(ActionType.GOTO_CANCEL, {}, _now())
