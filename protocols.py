from __future__ import annotations

from typing import Any, Optional, Protocol

from ap_types import Axis, Frame


class CameraStreamProtocol(Protocol):
    def latest(self) -> Optional[Frame]:
        ...


class MountProtocol(Protocol):
    def is_connected(self) -> bool:
        ...

    def move_steps(
        self,
        axis: Axis,
        direction: int,
        steps: int,
        delay_us: int,
        *,
        profile: str = "smooth",
    ) -> None:
        ...

    def stop(self) -> None:
        ...


class StatePublisherProtocol(Protocol):
    def __call__(self, patch: dict[str, dict[str, Any]]) -> None:
        ...
