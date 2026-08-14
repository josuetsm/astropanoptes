from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import call, patch
from io import BytesIO
import plistlib
from pathlib import Path

from mount_arduino import (
    ArduinoConfig,
    ArduinoController,
    ArduinoMount,
    estimate_firmware_move_duration_s,
    firmware_move_period_us,
    resolve_common_microsteps,
    resolve_mount_port,
)


class MountArduinoTests(TestCase):
    def test_serial_response_filter_ignores_stale_ready_and_ok_lines(self) -> None:
        matches = ArduinoController._response_matches_command

        self.assertFalse(matches("ENABLE 1", "READY"))
        self.assertTrue(matches("ENABLE 1", "OK"))
        self.assertFalse(matches("STATUS", "OK"))
        self.assertTrue(
            matches(
                "STATUS",
                "EN=1 MS=64 MOVEPROFILES=1 MOVE=0,0 PROFILE=SMOOTH,SMOOTH",
            )
        )
        self.assertFalse(matches("PING", "OK"))
        self.assertTrue(matches("PING", "READY"))

    def test_firmware_keeps_ms_hardware_fixed_and_uses_symmetric_ramp(self) -> None:
        firmware = (
            Path(__file__).resolve().parents[1]
            / "mount_firmware"
            / "mount_firmware.ino"
        ).read_text(encoding="utf-8")

        self.assertIn("FIXED_MICROSTEPS = 64", firmware)
        self.assertIn("MSFIXED=1", firmware)
        self.assertIn("MOVEPROFILES=1", firmware)
        self.assertNotIn("pinMode(MS", firmware)
        self.assertNotIn("digitalWrite(MS", firmware)
        self.assertIn("MOVE_SMOOTH_MAX_ACCEL_STEPS_S2", firmware)
        self.assertIn("MOVE_MAX_RATE_STEPS_S", firmware)
        self.assertIn("min(completed, stoppingEdge)", firmware)
        self.assertIn("x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f)", firmware)
        self.assertIn("SMOOTH|DIRECT", firmware)

    def test_firmware_slew_uses_delay_as_maximum_speed_limit(self) -> None:
        periods = [
            firmware_move_period_us(400, 6000, remaining)
            for remaining in range(6000, 0, -1)
        ]

        self.assertTrue(all(period_us >= 403 for period_us in periods))
        self.assertEqual(min(periods), 403)
        self.assertEqual(periods[0], periods[-1])

    def test_delay_ten_is_speed_limited_and_profiles_remain_distinct(self) -> None:
        total = 30_000
        smooth = [
            firmware_move_period_us(10, total, remaining, profile="smooth")
            for remaining in range(total, 0, -1)
        ]
        direct = [
            firmware_move_period_us(10, total, remaining, profile="direct")
            for remaining in range(total, 0, -1)
        ]

        self.assertTrue(all(period >= 84 for period in smooth))
        self.assertEqual(set(direct), {84})
        self.assertEqual(smooth[0], 2500)
        self.assertEqual(smooth[-1], 2500)
        self.assertGreater(min(smooth), min(direct))
        self.assertGreater(
            estimate_firmware_move_duration_s(total, 10, profile="smooth"),
            estimate_firmware_move_duration_s(total, 10, profile="direct"),
        )

    def test_short_firmware_slew_does_not_reach_max_speed(self) -> None:
        periods = [
            firmware_move_period_us(400, 200, remaining)
            for remaining in range(200, 0, -1)
        ]

        self.assertGreater(min(periods), 403)
        self.assertEqual(periods[0], periods[-1])
        self.assertGreater(
            estimate_firmware_move_duration_s(200, 400),
            200 * 403 / 1.0e6,
        )

    def test_resolve_common_microsteps_is_hardware_fixed(self) -> None:
        self.assertEqual(resolve_common_microsteps(32, 64), 64)
        self.assertEqual(resolve_common_microsteps(16, 16), 64)

    def test_resolve_common_microsteps_falls_back_to_alt_or_default(self) -> None:
        self.assertEqual(resolve_common_microsteps(7, 64), 64)
        self.assertEqual(resolve_common_microsteps(7, 11), 64)

    def test_resolve_mount_port_keeps_explicit_value(self) -> None:
        explicit = "/dev/cu.AstroPanoptes-ESP32"
        self.assertEqual(resolve_mount_port(explicit), explicit)

    def test_resolve_mount_port_auto_prefers_astropanoptes_esp32(self) -> None:
        ports = [
            SimpleNamespace(
                device="/dev/cu.random-usb",
                description="USB serial adapter",
                manufacturer="FTDI",
                product="FT232",
                interface="",
                hwid="USB VID:PID=0403:6001",
            ),
            SimpleNamespace(
                device="/dev/cu.AstroPanoptes-ESP32",
                description="AstroPanoptes-ESP32",
                manufacturer="Espressif",
                product="Bluetooth SPP",
                interface="",
                hwid="SPP",
            ),
        ]

        with patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(resolve_mount_port("AUTO"), "/dev/cu.AstroPanoptes-ESP32")

    def test_resolve_mount_port_auto_never_guesses_generic_bluetooth_port(self) -> None:
        ports = [
            SimpleNamespace(
                device="/dev/cu.Bluetooth-Incoming-Port",
                description="n/a",
                manufacturer="",
                product="",
                interface="",
                hwid="n/a",
            ),
            SimpleNamespace(
                device="/dev/cu.WI-C310",
                description="n/a",
                manufacturer="",
                product="",
                interface="",
                hwid="n/a",
            ),
        ]

        with patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(resolve_mount_port("AUTO"), "")

    def test_set_microsteps_accepts_only_fixed_hardware_value(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
        with patch.object(ctrl, "send", return_value="OK MS_FIXED 64") as mocked_send:
            resp = ctrl.set_microsteps(64, 64)

        self.assertEqual(resp, "OK MS_FIXED 64")
        mocked_send.assert_called_once_with("MS 64", timeout_s=0.60)

        with self.assertRaisesRegex(ValueError, "hardware-fixed"):
            ctrl.set_microsteps(32, 64)

    def test_move_refuses_legacy_firmware_without_profile_capability(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
        with patch.object(ctrl, "status", return_value="EN=1 MS=64"):
            with self.assertRaisesRegex(RuntimeError, "MOVEPROFILES=1"):
                ctrl.move("B", "FWD", 100, 400, profile="smooth")

    def test_direct_move_uses_legacy_command_without_profile_capability(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
        with (
            patch.object(ctrl, "status", return_value="EN=1 MS=64"),
            patch.object(ctrl, "send", return_value="OK") as mocked_send,
        ):
            response = ctrl.move("B", "FWD", 100, 400, profile="direct")

        self.assertEqual(response, "OK")
        mocked_send.assert_called_once_with(
            "MOVE B FWD 100 400",
            timeout_s=3.5,
        )

    def test_move_sends_explicit_profile_when_firmware_supports_it(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
        ctrl._move_profiles_supported = True
        with patch.object(ctrl, "send", return_value="OK") as mocked_send:
            response = ctrl.move("B", "FWD", 100, 400, profile="direct")

        self.assertEqual(response, "OK")
        mocked_send.assert_called_once_with(
            "MOVE B FWD 100 400 DIRECT",
            timeout_s=3.5,
        )

    def test_bluetooth_lookup_uses_inquiry_when_device_is_forgotten(self) -> None:
        ctrl = ArduinoController(
            ArduinoConfig(bt_device_name="AstroPanoptes-ESP32", bt_inquiry_s=3.0)
        )
        inquiry = '[{"address":"AA-BB-CC-DD-EE-FF","name":"AstroPanoptes-ESP32"}]'
        with patch.object(
            ctrl,
            "_blueutil_run",
            side_effect=[(0, "[]", ""), (0, inquiry, "")],
        ) as mocked_run, patch.object(ctrl, "_macos_persistent_device_id", return_value=""):
            self.assertEqual(ctrl._blueutil_get_device_id(), "AA-BB-CC-DD-EE-FF")

        self.assertEqual(mocked_run.call_args_list[0], call("--paired", "--format", "json", timeout_s=15.0))
        self.assertEqual(
            mocked_run.call_args_list[1],
            call("--inquiry", "3", "--format", "json", timeout_s=23.0),
        )

    def test_bluetooth_lookup_recovers_persistent_macos_rfcomm_address(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(bt_device_name="AstroPanoptes-ESP32"))
        payload = plistlib.dumps(
            {
                "PersistentPorts": {
                    "70:4B:CA:22:0E:BA": {
                        "BSDName": "AstroPanoptes-ESP32",
                        "RFCOMMChannel": 1,
                    }
                }
            }
        )
        with patch("builtins.open", return_value=BytesIO(payload)):
            self.assertEqual(ctrl._macos_persistent_device_id(), "70:4B:CA:22:0E:BA")

    def test_refresh_bluetooth_forgets_pairs_and_connects(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(bt_forget_retry_s=0.0))
        with (
            patch.object(ctrl, "_blueutil_path", return_value="/opt/homebrew/bin/blueutil"),
            patch.object(ctrl, "_blueutil_get_device_id", return_value="AA-BB-CC-DD-EE-FF"),
            patch.object(ctrl, "_blueutil_run", return_value=(0, "", "")) as mocked_run,
        ):
            ok, error = ctrl._refresh_bt_pairing()

        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(
            [args.args[:2] for args in mocked_run.call_args_list],
            [
                ("--disconnect", "AA-BB-CC-DD-EE-FF"),
                ("--wait-disconnect", "AA-BB-CC-DD-EE-FF"),
                ("--unpair", "AA-BB-CC-DD-EE-FF"),
                ("--pair", "AA-BB-CC-DD-EE-FF"),
                ("--connect", "AA-BB-CC-DD-EE-FF"),
                ("--wait-connect", "AA-BB-CC-DD-EE-FF"),
            ],
        )

    def test_connect_refreshes_bluetooth_before_resolving_auto_port(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO", open_attempts=1))
        order: list[str] = []

        def _refresh() -> tuple[bool, str]:
            order.append("bluetooth")
            return (True, "")

        def _port() -> str:
            order.append("port")
            return ""

        with (
            patch.object(ctrl, "_refresh_bt_pairing", side_effect=_refresh),
            patch.object(ctrl, "_wait_for_mount_port", side_effect=_port),
            patch("mount_arduino.list_serial_ports", return_value=[]),
        ):
            message = ctrl.connect()

        self.assertEqual(order, ["bluetooth", "port"])
        self.assertIn("no serial port found", message)

    def test_disconnect_closes_even_if_stop_fails(self) -> None:
        def _stop_fail() -> str:
            raise RuntimeError("boom")

        mount = ArduinoMount()
        mount.ctrl = SimpleNamespace(
            stop=_stop_fail,
            close=lambda: setattr(self, "_closed", True),
            is_connected=False,
        )
        self._closed = False

        mount.disconnect()
        self.assertTrue(self._closed)

    def test_disconnect_does_not_reconnect_when_serial_is_closed(self) -> None:
        mount = ArduinoMount()
        stop_calls: list[bool] = []
        mount.ctrl = SimpleNamespace(
            stop=lambda: stop_calls.append(True),
            close=lambda: None,
            is_connected=False,
        )

        mount.disconnect()

        self.assertEqual(stop_calls, [])
