from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from actions import Action
from ap_types import AppState
from config import AppConfig
from control_client import AttachError, connect, run_commands
from control_server import ControlServer, ControlServerError


class _FakeRunner:
    def __init__(self) -> None:
        self.cfg = AppConfig()
        self.state = AppState()
        self.preview = b"\xff\xd8test-jpeg\xff\xd9"
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.actions: list[Action] = []
        self.operation_counters = {
            "platesolving": {"started": 0, "finished": 0},
            "goto": {"started": 0, "finished": 0},
            "mount_move": {"started": 0, "finished": 0},
            "camera_record": {"started": 0, "finished": 0},
        }

    def get_state(self) -> AppState:
        return self.state.snapshot()

    def get_latest_preview_jpeg(self) -> bytes:
        return self.preview

    def get_operation_counters(self):
        return {name: dict(values) for name, values in self.operation_counters.items()}

    def set_simulation_enabled(self, enabled: bool) -> None:
        self.cfg.simulation.enabled = enabled
        self.calls.append(("set_simulation_enabled", (enabled,), {}))

    def cancel_platesolving(self) -> None:
        self.calls.append(("cancel_platesolving", (), {}))

    def enqueue(self, action: Action) -> None:
        self.actions.append(action)

    def __getattr__(self, name: str):
        if name.startswith("request_"):
            def request(*args: Any, **kwargs: Any) -> None:
                self.calls.append((name, args, kwargs))

            return request
        raise AttributeError(name)


class ControlServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.socket_path = Path(self.tmp.name) / "control.sock"
        self.runner = _FakeRunner()
        self.server = ControlServer(self.runner, socket_path=self.socket_path)
        self.addCleanup(self.server.stop)

    def test_start_creates_owner_only_socket(self) -> None:
        self.server.start()
        self.assertTrue(self.socket_path.exists())
        self.assertEqual(self.socket_path.stat().st_mode & 0o777, 0o600)

    def test_client_can_run_command_against_running_gui_session(self) -> None:
        self.server.start()
        sock = connect(self.socket_path)
        try:
            ok = run_commands(sock, ["camera connect 1"])
        finally:
            sock.close()
        self.assertTrue(ok)
        self.assertEqual(self.runner.calls, [("request_camera_connect", (1,), {})])

    def test_client_sees_command_error(self) -> None:
        self.server.start()
        sock = connect(self.socket_path)
        try:
            ok = run_commands(sock, ["not-a-real-command"])
        finally:
            sock.close()
        self.assertFalse(ok)

    def test_connect_without_running_server_raises_clear_error(self) -> None:
        with self.assertRaises(AttachError):
            connect(self.socket_path)

    def test_stop_removes_socket_and_allows_restart(self) -> None:
        self.server.start()
        self.server.stop()
        self.assertFalse(self.socket_path.exists())
        self.server.start()
        self.assertTrue(self.socket_path.exists())

    def test_second_server_on_same_socket_refuses_to_start(self) -> None:
        self.server.start()
        other = ControlServer(_FakeRunner(), socket_path=self.socket_path)
        with self.assertRaises(ControlServerError):
            other.start()

    def test_stale_socket_file_is_replaced(self) -> None:
        self.server.start()
        self.server.stop()
        # Simulate a leftover socket file from a crashed process: nothing is
        # listening on it anymore, so a new server must be able to bind.
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.touch()
        self.server.start()
        self.assertTrue(self.socket_path.exists())


if __name__ == "__main__":
    unittest.main()
