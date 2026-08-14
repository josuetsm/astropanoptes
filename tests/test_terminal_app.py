from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any

from actions import Action
from ap_types import AppState, Axis, CameraStatus, GotoStatus, PlatesolvingStatus
from config import AppConfig
from terminal_app import TerminalApp, _jsonable, _parse_assignments, _parse_value, build_parser


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
        return {
            name: dict(values)
            for name, values in self.operation_counters.items()
        }

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
                operation = None
                if name == "request_platesolving_run":
                    operation = "platesolving"
                elif name.startswith("request_goto_") or name == "request_mount_goto":
                    operation = "goto"
                elif name == "request_mount_move_steps":
                    operation = "mount_move"
                elif name == "request_camera_record_raw":
                    operation = "camera_record"
                if operation is not None:
                    self.operation_counters[operation]["started"] += 1
                    self.operation_counters[operation]["finished"] = self.operation_counters[operation]["started"]

            return request
        raise AttributeError(name)


class TerminalAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _FakeRunner()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.tmp = tempfile.TemporaryDirectory()
        self.opened_urls: list[str] = []
        self.terminal = TerminalApp(
            self.runner,  # type: ignore[arg-type]
            images_dir=Path(self.tmp.name) / "images",
            output=self.stdout,
            error_output=self.stderr,
            browser_open=lambda url: self.opened_urls.append(url) or True,
        )

    def tearDown(self) -> None:
        self.terminal.close()
        self.tmp.cleanup()

    def test_value_and_assignment_parsing_is_script_friendly(self) -> None:
        self.assertIs(_parse_value("on"), True)
        self.assertIs(_parse_value("false"), False)
        self.assertEqual(_parse_value("12"), 12)
        self.assertAlmostEqual(_parse_value("1.25"), 1.25)
        self.assertEqual(_parse_value('[1, 2]'), [1, 2])
        self.assertEqual(
            _parse_assignments(["enabled=true", "gain=2.5", 'labels=["a","b"]']),
            {"enabled": True, "gain": 2.5, "labels": ["a", "b"]},
        )

    def test_routes_camera_and_altaz_commands(self) -> None:
        self.assertTrue(self.terminal.execute_line("camera connect 2"))
        self.assertTrue(self.terminal.execute_line("camera set exp_ms 250.5"))
        self.assertTrue(self.terminal.execute_line("goto altaz 180 45"))

        self.assertEqual(self.runner.calls[0], ("request_camera_connect", (2,), {}))
        self.assertEqual(self.runner.calls[1], ("request_camera_param", ("exp_ms", 250.5), {}))
        self.assertEqual(
            self.runner.calls[2],
            ("request_mount_goto", ({"az_deg": 180.0, "alt_deg": 45.0},), {}),
        )
        self.assertFalse(self.terminal.had_error)

    def test_status_json_is_machine_readable_and_does_not_embed_images(self) -> None:
        self.runner.state.camera.status = CameraStatus.OK
        self.runner.state.camera.connected = True
        self.runner.state.platesolving.status = PlatesolvingStatus.OK
        self.runner.state.platesolving.debug_jpeg = b"large-image"

        self.terminal.execute_line("status --json")
        payload = json.loads(self.stdout.getvalue())

        self.assertTrue(payload["camera"]["connected"])
        self.assertEqual(payload["camera"]["status"], "OK")
        self.assertEqual(
            payload["platesolving"]["debug_jpeg"],
            {"available": True, "bytes": 11},
        )

    def test_wait_uses_nested_state_paths(self) -> None:
        self.runner.state.camera.connected = True
        self.terminal.execute_line("wait camera.connected true 0.1")
        self.assertIn("OK camera.connected=True", self.stdout.getvalue())
        self.assertFalse(self.terminal.had_error)

    def test_get_reads_one_state_field_for_compact_diagnostics(self) -> None:
        self.runner.state.tracking.error_px = 3.25
        self.terminal.execute_line("get tracking.error_px")
        self.assertEqual(self.stdout.getvalue().strip(), "3.25")
        self.assertFalse(self.terminal.had_error)

    def test_await_waits_for_the_operation_queued_by_this_console(self) -> None:
        self.terminal.execute_line("goto altaz 180 45")
        self.terminal.execute_line("await goto 0.1")

        output = self.stdout.getvalue()
        self.assertIn('"operation": "goto"', output)
        self.assertFalse(self.terminal.had_error)

    def test_await_supports_manual_moves_and_raw_recordings(self) -> None:
        self.terminal.execute_line("mount move az 1 25 1800")
        self.terminal.execute_line("await mount 0.1")
        self.terminal.execute_line("camera record 0.1 raw_output test")
        self.terminal.execute_line("await record 0.1")

        output = self.stdout.getvalue()
        self.assertIn('"operation": "mount_move"', output)
        self.assertIn('"operation": "camera_record"', output)
        self.assertFalse(self.terminal.had_error)

    def test_manual_move_accepts_smooth_and_direct_profiles(self) -> None:
        self.terminal.execute_line("mount move alt 1 30000 10 smooth")
        self.terminal.execute_line("mount move alt -1 30000 10 direct")

        move_calls = [
            call for call in self.runner.calls
            if call[0] == "request_mount_move_steps"
        ]
        self.assertEqual(move_calls[0][1], (Axis.ALT, 1, 30000, 10, "smooth"))
        self.assertEqual(move_calls[1][1], (Axis.ALT, -1, 30000, 10, "direct"))

    def test_stop_requests_full_emergency_stop_sequence(self) -> None:
        self.terminal.execute_line("stop")
        names = [name for name, _args, _kwargs in self.runner.calls]
        self.assertEqual(
            names,
            ["request_mount_stop"],
        )

    def test_health_does_not_report_success_reason_as_error(self) -> None:
        self.runner.state.goto.status = GotoStatus.OK
        self.runner.state.goto.reason = "AUTOCAL"

        self.terminal.execute_line("health")

        payload = json.loads(self.stdout.getvalue())
        self.assertIsNone(payload["errors"]["goto"])

    def test_demo_seed_is_available_for_reproducible_debugging(self) -> None:
        args = build_parser().parse_args(["--demo", "--seed", "42"])
        self.assertTrue(args.demo)
        self.assertEqual(args.seed, 42)

    def test_image_is_saved_inside_images_folder_with_state_metadata(self) -> None:
        self.terminal.execute_line("image live ../outside 0")

        image_path = Path(self.stdout.getvalue().strip())
        self.assertEqual(image_path.parent, (Path(self.tmp.name) / "images").resolve())
        self.assertEqual(image_path.name, "outside.jpg")
        self.assertEqual(image_path.read_bytes(), self.runner.preview)

        metadata = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source"], "live")
        self.assertEqual(metadata["image"], str(image_path))
        self.assertIn("camera", metadata["state"])

    def test_live_view_serves_preview_and_mirrors_latest_image(self) -> None:
        self.terminal.execute_line("view start 0 20 no-open")
        payload = json.loads(self.stdout.getvalue().strip())
        latest_path = Path(payload["latest_image"])
        deadline = time.monotonic() + 1.0
        while not latest_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["browser_opened"])
        self.assertEqual(latest_path.read_bytes(), self.runner.preview)
        with urllib.request.urlopen(payload["url"] + "latest.jpg", timeout=1.0) as response:
            self.assertEqual(response.read(), self.runner.preview)

        self.terminal.execute_line("camera connect 2")
        with urllib.request.urlopen(payload["url"] + "console.json", timeout=1.0) as response:
            console = json.loads(response.read())
        self.assertEqual(console["entries"][-2]["kind"], "command")
        self.assertEqual(console["entries"][-2]["text"], "camera connect 2")
        self.assertEqual(console["entries"][-1]["kind"], "output")
        self.assertEqual(console["entries"][-1]["text"], "OK")

        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.terminal.execute_line("view stop")
        self.assertEqual(self.stdout.getvalue().strip(), "OK")

    def test_live_view_opens_browser_by_default(self) -> None:
        self.terminal.execute_line("view start 0 2")
        payload = json.loads(self.stdout.getvalue().strip())
        self.assertTrue(payload["browser_opened"])
        self.assertEqual(self.opened_urls, [payload["url"]])

    def test_advanced_action_accepts_json_payload(self) -> None:
        self.terminal.execute_line('action CAMERA_SET_PARAM {"name":"gain","value":380}')
        self.assertEqual(len(self.runner.actions), 1)
        self.assertEqual(self.runner.actions[0].type.value, "CAMERA_SET_PARAM")
        self.assertEqual(self.runner.actions[0].payload, {"name": "gain", "value": 380})

    def test_script_stops_after_first_error(self) -> None:
        ok = self.terminal.run_script(["camera connect", "unknown", "camera disconnect"])
        self.assertFalse(ok)
        self.assertEqual(len(self.runner.calls), 1)
        self.assertIn("comando desconocido", self.stderr.getvalue())

    def test_jsonable_describes_arrays_instead_of_expanding_them(self) -> None:
        import numpy as np

        value = _jsonable(np.zeros((100, 200), dtype=np.uint16))
        self.assertEqual(value, {"type": "ndarray", "shape": [100, 200], "dtype": "uint16"})


if __name__ == "__main__":
    unittest.main()
