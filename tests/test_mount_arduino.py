from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from mount_arduino import (
    ArduinoConfig,
    ArduinoController,
    ArduinoMount,
    resolve_common_microsteps,
    resolve_mount_port,
)


class MountArduinoTests(TestCase):
    def test_resolve_common_microsteps_prefers_valid_requested_value(self) -> None:
        self.assertEqual(resolve_common_microsteps(32, 64), 32)
        self.assertEqual(resolve_common_microsteps(16, 16), 16)

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

    def test_set_microsteps_uses_single_common_command(self) -> None:
        ctrl = ArduinoController(ArduinoConfig(port="AUTO"))
        with patch.object(ctrl, "send", return_value="OK MS 32") as mocked_send:
            resp = ctrl.set_microsteps(32, 64)

        self.assertEqual(resp, "OK MS 32")
        mocked_send.assert_called_once_with("MS 32", timeout_s=0.60)

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
