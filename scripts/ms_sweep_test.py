#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mount_arduino import ArduinoConfig, ArduinoController  # noqa: E402


MS_CHOICES: tuple[int, ...] = (64,)


def axis_to_fw(axis: str) -> str:
    a = str(axis or "").strip().lower()
    if a in {"a", "az"}:
        return "A"
    if a in {"b", "alt"}:
        return "B"
    raise ValueError(f"invalid axis: {axis}")


def dir_to_fw(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d in {"fwd", "+", "pos", "forward"}:
        return "FWD"
    if d in {"rev", "-", "neg", "reverse"}:
        return "REV"
    raise ValueError(f"invalid direction: {direction}")


def mechanical_reduction(
    *,
    pulley_teeth: int,
    belt_pitch_m: float,
    ring_radius_m: float,
    gear_reduction: float | None,
) -> float:
    if gear_reduction is not None:
        ratio = float(gear_reduction)
        if math.isfinite(ratio) and ratio > 0.0:
            return float(ratio)
    ring_teeth = (2.0 * math.pi * float(ring_radius_m)) / float(belt_pitch_m)
    return float(ring_teeth / float(pulley_teeth))


def expected_deg_for_steps(
    *,
    steps: int,
    microsteps: int,
    full_steps: int,
    pulley_teeth: int,
    belt_pitch_m: float,
    ring_radius_m: float,
    gear_reduction: float | None,
) -> float:
    ratio = mechanical_reduction(
        pulley_teeth=pulley_teeth,
        belt_pitch_m=belt_pitch_m,
        ring_radius_m=ring_radius_m,
        gear_reduction=gear_reduction,
    )
    steps_per_axis_rev = float(full_steps) * float(microsteps) * ratio
    return float(steps) * (360.0 / steps_per_axis_rev)


def inferred_microsteps(
    *,
    steps: int,
    measured_deg: float,
    full_steps: int,
    pulley_teeth: int,
    belt_pitch_m: float,
    ring_radius_m: float,
    gear_reduction: float | None,
) -> float:
    if not math.isfinite(measured_deg) or abs(measured_deg) <= 1e-9:
        return float("nan")
    ratio = mechanical_reduction(
        pulley_teeth=pulley_teeth,
        belt_pitch_m=belt_pitch_m,
        ring_radius_m=ring_radius_m,
        gear_reduction=gear_reduction,
    )
    return float(steps) * 360.0 / (abs(float(measured_deg)) * float(full_steps) * ratio)


def nearest_ms(ms_value: float) -> int:
    return int(min(MS_CHOICES, key=lambda v: abs(float(v) - float(ms_value))))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the hardware-fixed MS64 scale against a measured angle."
    )
    p.add_argument("--port", default="AUTO", help="Serial port path or AUTO")
    p.add_argument("--baud", type=int, default=115200, help="Baudrate")
    p.add_argument("--axis", default="az", choices=["az", "alt", "a", "b"], help="Axis to move")
    p.add_argument(
        "--direction",
        default="fwd",
        choices=["fwd", "rev", "+", "-"],
        help="Move direction",
    )
    p.add_argument("--steps", type=int, default=20000, help="MOVE steps")
    p.add_argument("--delay-us", type=int, default=1000, help="MOVE delay_us")
    p.add_argument("--settle-s", type=float, default=0.25, help="Extra wait after move finishes")
    p.add_argument(
        "--microsteps",
        nargs="*",
        type=int,
        default=list(MS_CHOICES),
        help="legacy option; only the hardware-fixed value 64 is accepted",
    )

    # Mechanics (same defaults as goto.py)
    p.add_argument("--full-steps", type=int, default=200, help="Motor full-steps per rev")
    p.add_argument("--gear-reduction-az", type=float, default=45.0, help="AZ motor:axis reduction")
    p.add_argument("--gear-reduction-alt", type=float, default=45.0, help="ALT motor:axis reduction")
    p.add_argument("--pulley-teeth", type=int, default=20, help="Motor pulley teeth")
    p.add_argument("--belt-pitch-m", type=float, default=0.002, help="Belt pitch in meters")
    p.add_argument("--ring-radius-az", type=float, default=0.24, help="AZ ring radius fallback in meters")
    p.add_argument("--ring-radius-alt", type=float, default=0.235, help="ALT ring radius fallback in meters")

    p.add_argument("--no-prompt", action="store_true", help="Do not ask for measured degrees")
    return p.parse_args(argv)


def _drain(ctrl: ArduinoController, *, max_lines: int = 60, max_time_s: float = 0.20) -> None:
    try:
        ctrl._drain_lines(max_lines=max_lines, max_time_s=max_time_s)  # type: ignore[attr-defined]
    except Exception as exc:
        print(f"warning: failed to drain serial input: {exc}", file=sys.stderr)


def tx(ctrl: ArduinoController, cmd: str, *, timeout_s: float = 1.0) -> str:
    _drain(ctrl)
    return str(ctrl.send(cmd, timeout_s=float(timeout_s), reset_input=False) or "").strip()


def is_ok(resp: str) -> bool:
    return str(resp or "").strip().upper().startswith("OK")


def is_ok_ms(resp: str) -> bool:
    return str(resp or "").strip().upper().startswith("OK MS")


def is_status(resp: str) -> bool:
    r = str(resp or "").strip().upper()
    return ("EN=" in r) and ("MS=" in r)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    axis_fw = axis_to_fw(args.axis)
    dir_fw = dir_to_fw(args.direction)
    ring_radius_m = float(args.ring_radius_az if axis_fw == "A" else args.ring_radius_alt)
    gear_reduction = float(args.gear_reduction_az if axis_fw == "A" else args.gear_reduction_alt)
    gear_reduction_arg = gear_reduction if gear_reduction > 0.0 else None
    steps = max(1, int(args.steps))
    delay_us = max(1, int(args.delay_us))
    settle_s = max(0.0, float(args.settle_s))

    sweep: list[int] = []
    for ms in args.microsteps:
        m = int(ms)
        if m in MS_CHOICES:
            sweep.append(m)
    if not sweep:
        raise ValueError("microstepping is hardware-fixed at 1/64")

    cfg = ArduinoConfig(port=str(args.port), baud=int(args.baud))
    ctrl = ArduinoController(cfg)

    print(f"Connecting to mount on port={cfg.port} baud={cfg.baud} ...")
    msg = ctrl.connect()
    print(f"CONNECT: {msg}")
    if "error" in str(msg).lower():
        return 2

    try:
        resp = tx(ctrl, "ENABLE 1", timeout_s=1.0)
        print(f"ENABLE 1 -> {resp or 'NO-RESP'}")
        if not is_ok(resp):
            print("ERROR: enable failed")
            return 3

        print("")
        ratio = mechanical_reduction(
            pulley_teeth=int(args.pulley_teeth),
            belt_pitch_m=float(args.belt_pitch_m),
            ring_radius_m=ring_radius_m,
            gear_reduction=gear_reduction_arg,
        )
        print(f"Fixed-MS64 verification axis={axis_fw} direction={dir_fw} steps={steps} delay_us={delay_us} reduction={ratio:.3f}:1")
        print("Enter the measured angle in degrees after the move (blank=skip).")
        print("")

        rows: list[dict[str, float | int | str]] = []
        for ms in sweep:
            set_resp = tx(ctrl, f"MS {ms}", timeout_s=1.2)
            status_before = tx(ctrl, "STATUS", timeout_s=1.0)
            print(f"MS {ms} -> {set_resp or 'NO-RESP'} | STATUS: {status_before or 'NO-RESP'}")

            if (not is_ok_ms(set_resp)) or (not is_status(status_before)):
                rows.append({"ms_cmd": ms, "status": "MS_FAIL"})
                continue

            move_resp = tx(ctrl, f"MOVE {axis_fw} {dir_fw} {steps} {delay_us}", timeout_s=2.0)
            move_time_s = float(steps) * ((float(delay_us) + 3.0) / 1.0e6)
            time.sleep(move_time_s + settle_s)
            stop_resp = tx(ctrl, "STOP", timeout_s=1.0)
            status_after = tx(ctrl, "STATUS", timeout_s=1.0)

            exp_deg = expected_deg_for_steps(
                steps=steps,
                microsteps=ms,
                full_steps=int(args.full_steps),
                pulley_teeth=int(args.pulley_teeth),
                belt_pitch_m=float(args.belt_pitch_m),
                ring_radius_m=ring_radius_m,
                gear_reduction=gear_reduction_arg,
            )
            print(
                f"  MOVE -> {move_resp or 'NO-RESP'} | STOP -> {stop_resp or 'NO-RESP'} "
                f"| expected~{exp_deg:.2f} deg | STATUS: {status_after or 'NO-RESP'}"
            )
            if (not is_ok(move_resp)) or (not is_ok(stop_resp)) or (not is_status(status_after)):
                rows.append({"ms_cmd": ms, "status": "MOVE_OR_STATUS_FAIL"})
                continue

            measured_txt = ""
            measured = float("nan")
            if not bool(args.no_prompt):
                try:
                    measured_txt = input(f"  measured deg for MS {ms}: ").strip()
                except EOFError:
                    measured_txt = ""
            if measured_txt:
                try:
                    measured = float(measured_txt.replace(",", "."))
                except ValueError:
                    measured = float("nan")

            inferred_ms = inferred_microsteps(
                steps=steps,
                measured_deg=measured,
                full_steps=int(args.full_steps),
                pulley_teeth=int(args.pulley_teeth),
                belt_pitch_m=float(args.belt_pitch_m),
                ring_radius_m=ring_radius_m,
                gear_reduction=gear_reduction_arg,
            )
            near = nearest_ms(inferred_ms) if math.isfinite(inferred_ms) else -1
            rows.append(
                {
                    "ms_cmd": ms,
                    "expected_deg": exp_deg,
                    "measured_deg": measured,
                    "inferred_ms": inferred_ms,
                    "nearest_ms": near,
                    "status": "OK",
                }
            )

        print("")
        print("Summary:")
        for row in rows:
            ms_cmd = int(row.get("ms_cmd", -1))
            status = str(row.get("status", ""))
            if status != "OK":
                print(f"  MS {ms_cmd:>2}: {status}")
                continue
            exp_deg = float(row.get("expected_deg", float("nan")))
            measured = float(row.get("measured_deg", float("nan")))
            inferred_ms_val = float(row.get("inferred_ms", float("nan")))
            near = int(row.get("nearest_ms", -1))
            measured_txt = f"{measured:.2f}" if math.isfinite(measured) else "N/A"
            inferred_txt = f"{inferred_ms_val:.1f}" if math.isfinite(inferred_ms_val) else "N/A"
            near_txt = str(near) if near > 0 else "N/A"
            print(
                f"  MS {ms_cmd:>2}: expected={exp_deg:>7.2f} deg | measured={measured_txt:>7} "
                f"| inferred_ms={inferred_txt:>6} | nearest={near_txt}"
            )

        return 0
    finally:
        try:
            tx(ctrl, "ENABLE 0", timeout_s=1.0)
        except Exception as exc:
            print(f"warning: failed to disable mount before close: {exc}", file=sys.stderr)
        ctrl.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
