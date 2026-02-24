#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import serial
import serial.tools.list_ports


@dataclass(frozen=True)
class ProbeConfig:
    port: str
    baud: int
    serial_timeout_s: float
    response_timeout_s: float
    ready_wait_s: float
    connect_attempts: int
    retry_delay_s: float
    bt_forget_before_connect: bool
    bt_pair_after_forget: bool
    bt_refresh_sleep_s: float


def _port_hint(info: object) -> str:
    fields = (
        getattr(info, "device", ""),
        getattr(info, "description", ""),
        getattr(info, "manufacturer", ""),
        getattr(info, "product", ""),
        getattr(info, "interface", ""),
        getattr(info, "hwid", ""),
    )
    return " ".join(str(v or "") for v in fields).strip().lower()


def list_ports() -> list[object]:
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda p: str(getattr(p, "device", "") or ""))
    return ports


def print_ports() -> None:
    ports = list_ports()
    if not ports:
        print("No serial ports found.")
        return
    print("Serial ports:")
    for p in ports:
        dev = str(getattr(p, "device", "") or "").strip()
        desc = str(getattr(p, "description", "") or "").strip()
        mfg = str(getattr(p, "manufacturer", "") or "").strip()
        hwid = str(getattr(p, "hwid", "") or "").strip()
        print(f"  - {dev} | {desc} | {mfg} | {hwid}")


def resolve_port(requested: str) -> str:
    req = str(requested or "").strip()
    if req and req.upper() != "AUTO":
        return req

    candidates: list[tuple[int, str]] = []
    for p in list_ports():
        dev = str(getattr(p, "device", "") or "").strip()
        if not dev:
            continue
        hint = _port_hint(p)
        score = 0
        if "astropanoptes-esp32" in hint:
            score += 100
        if "astropanoptes" in hint:
            score += 70
        if "esp32" in hint:
            score += 40
        if "bluetooth" in hint or "spp" in hint:
            score += 20
        if dev.startswith("/dev/cu."):
            score += 5
        candidates.append((score, dev))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _decode_line(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip()


def _blueutil_path() -> str:
    return str(shutil.which("blueutil") or "")


def _blueutil_run(*args: str, timeout_s: float = 10.0) -> tuple[int, str, str]:
    bin_path = _blueutil_path()
    if not bin_path:
        return (127, "", "blueutil not found")
    try:
        p = subprocess.run(
            [bin_path, *args],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        return (int(p.returncode), str(p.stdout or "").strip(), str(p.stderr or "").strip())
    except Exception as exc:
        return (1, "", str(exc))


def _find_bt_device_id_by_name(name: str) -> str:
    wanted = str(name or "").strip()
    if not wanted:
        return ""
    rc, out, _ = _blueutil_run("--paired", "--format", "json", timeout_s=6.0)
    if rc == 0 and out:
        try:
            rows = json.loads(out)
            if isinstance(rows, list):
                for row in rows:
                    nm = str((row or {}).get("name", "")).strip()
                    if nm == wanted:
                        addr = str((row or {}).get("address", "")).strip()
                        return addr or wanted
        except Exception:
            pass
    return wanted


def refresh_bt_pairing(name: str, *, pair_after_forget: bool, sleep_s: float) -> None:
    if not _blueutil_path():
        print("BT hint: blueutil no encontrado, se omite forget/pair/connect automatico.")
        return
    dev_id = _find_bt_device_id_by_name(name)
    if not dev_id:
        return
    rc_disc, _, e_disc = _blueutil_run("--disconnect", dev_id, timeout_s=6.0)
    if rc_disc != 0 and e_disc:
        print(f"BT hint (disconnect): {e_disc}")
    rc_unpair, _, e_unpair = _blueutil_run("--unpair", dev_id, timeout_s=8.0)
    if rc_unpair != 0 and e_unpair:
        print(f"BT hint (unpair): {e_unpair}")
    if sleep_s > 0.0:
        time.sleep(float(sleep_s))
    if pair_after_forget:
        rc_pair, _, e_pair = _blueutil_run("--pair", dev_id, timeout_s=12.0)
        if rc_pair != 0:
            print(f"BT hint (pair): {e_pair or 'pair failed'}")
    rc_conn, _, e_conn = _blueutil_run("--connect", dev_id, timeout_s=10.0)
    if rc_conn != 0 and e_conn:
        print(f"BT hint (connect): {e_conn}")
    if sleep_s > 0.0:
        time.sleep(float(sleep_s))


def read_lines(ser: serial.Serial, wait_s: float) -> list[str]:
    out: list[str] = []
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = _decode_line(raw)
        if line:
            out.append(line)
    return out


def wait_for_ready(ser: serial.Serial, wait_s: float) -> bool:
    if wait_s <= 0:
        return False
    deadline = time.monotonic() + float(wait_s)
    print(f"Waiting for READY up to {wait_s:.1f}s...")
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = _decode_line(raw)
        if not line:
            continue
        print(f"RX(init): {line}")
        if line == "READY":
            return True
    return False


def send_cmd(ser: serial.Serial, cmd: str, response_timeout_s: float) -> str:
    wire = (cmd.strip() + "\n").encode("ascii", errors="ignore")
    ser.write(wire)
    ser.flush()

    deadline = time.monotonic() + max(0.0, float(response_timeout_s))
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = _decode_line(raw)
        if line:
            return line
    return ""


def run_test_sequence(ser: serial.Serial, cfg: ProbeConfig) -> int:
    seq = (
        "PING",
        "ENABLE 1",
        "STATUS",
        "STOP",
        "ENABLE 0",
    )
    print("Running test sequence...")
    rc = 0
    for cmd in seq:
        resp = send_cmd(ser, cmd, cfg.response_timeout_s)
        if not resp:
            rc = 1
            print(f"TX: {cmd}")
            print("RX: <timeout>")
            continue
        print(f"TX: {cmd}")
        print(f"RX: {resp}")
    return rc


def run_commands(ser: serial.Serial, commands: Iterable[str], timeout_s: float) -> int:
    rc = 0
    for cmd in commands:
        cmd = str(cmd or "").strip()
        if not cmd:
            continue
        resp = send_cmd(ser, cmd, timeout_s)
        print(f"TX: {cmd}")
        print(f"RX: {resp or '<timeout>'}")
        if not resp:
            rc = 1
    return rc


def interactive_loop(ser: serial.Serial, timeout_s: float) -> None:
    print("Interactive mode. Type commands, or 'quit' to exit.")
    while True:
        try:
            line = input("bt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("")
            break
        if not line:
            continue
        low = line.lower()
        if low in {"q", "quit", "exit"}:
            break
        if low in {"ports", ":ports"}:
            print_ports()
            continue
        resp = send_cmd(ser, line, timeout_s)
        print(f"RX: {resp or '<timeout>'}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BT SPP probe for AstroPanoptes ESP32 firmware")
    p.add_argument("--port", default="AUTO", help="Serial port path or AUTO")
    p.add_argument("--baud", type=int, default=115200, help="Baudrate")
    p.add_argument("--serial-timeout", type=float, default=0.30, help="PySerial read/write timeout")
    p.add_argument("--response-timeout", type=float, default=1.20, help="Per-command response timeout")
    p.add_argument("--ready-wait", type=float, default=6.0, help="Seconds to wait for initial READY")
    p.add_argument("--connect-attempts", type=int, default=3, help="Open/handshake retries before giving up")
    p.add_argument("--retry-delay", type=float, default=1.0, help="Seconds between attempts")
    p.add_argument("--forget-first", action="store_true", default=True, help="Run BT unpair/pair/connect before opening serial")
    p.add_argument("--no-forget-first", dest="forget_first", action="store_false", help="Disable BT unpair/pair/connect pre-step")
    p.add_argument("--pair-after-forget", action="store_true", default=True, help="Pair again after unpair")
    p.add_argument("--no-pair-after-forget", dest="pair_after_forget", action="store_false", help="Skip pair after unpair")
    p.add_argument("--bt-name", default="AstroPanoptes-ESP32", help="Bluetooth device name")
    p.add_argument("--bt-refresh-sleep", type=float, default=0.8, help="Sleep between BT reset steps")
    p.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    p.add_argument("--cmd", action="append", default=[], help="Command to send (repeatable)")
    p.add_argument("--test", action="store_true", help="Run default test sequence")
    p.add_argument("--interactive", action="store_true", help="Interactive terminal")
    p.add_argument("--skip-ready", action="store_true", help="Do not wait for initial READY")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.list_ports:
        print_ports()
        return 0

    port = resolve_port(str(args.port))
    if not port:
        print("ERROR: no candidate serial port found.")
        print_ports()
        return 2

    cfg = ProbeConfig(
        port=port,
        baud=int(args.baud),
        serial_timeout_s=float(args.serial_timeout),
        response_timeout_s=float(args.response_timeout),
        ready_wait_s=float(args.ready_wait),
        connect_attempts=max(1, int(args.connect_attempts)),
        retry_delay_s=max(0.0, float(args.retry_delay)),
        bt_forget_before_connect=bool(args.forget_first),
        bt_pair_after_forget=bool(args.pair_after_forget),
        bt_refresh_sleep_s=max(0.0, float(args.bt_refresh_sleep)),
    )

    for attempt_idx in range(cfg.connect_attempts):
        print(f"Opening {cfg.port} @ {cfg.baud} (attempt {attempt_idx + 1}/{cfg.connect_attempts})...")
        try:
            if cfg.bt_forget_before_connect:
                refresh_bt_pairing(
                    str(args.bt_name),
                    pair_after_forget=cfg.bt_pair_after_forget,
                    sleep_s=cfg.bt_refresh_sleep_s,
                )
            with serial.Serial(
                cfg.port,
                cfg.baud,
                timeout=cfg.serial_timeout_s,
                write_timeout=cfg.serial_timeout_s,
            ) as ser:
                ready = False
                if not args.skip_ready:
                    ready = wait_for_ready(ser, cfg.ready_wait_s)
                    if not ready:
                        print("WARN: READY not received in time.")
                else:
                    stale = read_lines(ser, 0.20)
                    for line in stale:
                        print(f"RX(init): {line}")

                rc = 0
                if args.cmd:
                    rc = max(rc, run_commands(ser, args.cmd, cfg.response_timeout_s))
                elif args.test or (not args.interactive):
                    rc = max(rc, run_test_sequence(ser, cfg))
                elif not ready and not args.skip_ready:
                    # Interactive-only mode should still validate link quickly.
                    ping = send_cmd(ser, "PING", cfg.response_timeout_s)
                    if not ping:
                        rc = 1
                        print("TX: PING")
                        print("RX: <timeout>")

                if rc == 0:
                    if args.interactive:
                        interactive_loop(ser, cfg.response_timeout_s)
                    return 0
        except serial.SerialException as exc:
            print(f"ERROR: serial exception: {exc}")

        if attempt_idx < (cfg.connect_attempts - 1):
            if cfg.retry_delay_s > 0.0:
                print(f"Retrying in {cfg.retry_delay_s:.1f}s...")
                time.sleep(cfg.retry_delay_s)

    return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
