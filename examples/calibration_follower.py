#!/usr/bin/env python3
"""Zero a DK1 follower's joint encoders at the pose the arm is held in right now.

    python examples/calibration_follower.py PORT            # zero all 7 motors
    python examples/calibration_follower.py PORT --check    # ping motors + print positions, zero nothing

Motors are never enabled here, so the arm stays limp: hold it in the mechanical
zero pose, then run. The zero is written into each Damiao motor's own flash
(CAN command 0xFE) -- nothing is written on the host, and there is no LeRobot
calibration JSON for the DK1. The gripper zero is overwritten again at every
connect() (torque-homing against the open stop), so its pose does not matter.

PORT: the follower's serial device -- LEFT_/RIGHT_FOLLOWER_PORT in make.local,
or `ls -l /dev/serial/by-path/`. Both DK1 boards share one USB serial number,
so use by-path (or ttyACMx), never by-id.
"""

import argparse
import time

import serial

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import DM_variable, MotorControl

ZERO_TOL_RAD = 0.02


def fmt(pos: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:+.3f}" for k, v in pos.items())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("port", help="follower serial port, e.g. /dev/serial/by-path/...:1.0")
    ap.add_argument(
        "--check", action="store_true",
        help="only verify every motor answers and print positions; write nothing",
    )
    ap.add_argument("--yes", action="store_true", help="skip the zero-pose confirmation prompt")
    args = ap.parse_args()

    follower = DK1Follower(DK1FollowerConfig(port=args.port))
    ser = serial.Serial(args.port, 921600, timeout=0.5)
    time.sleep(0.5)
    control = MotorControl(ser)

    def positions() -> dict[str, float]:
        out = {}
        for key, motor in follower.motors.items():
            control.refresh_motor_status(motor)
            time.sleep(0.01)
            out[key] = motor.getPosition()
        return out

    try:
        for key, motor in follower.motors.items():
            control.addMotor(motor)
            for _ in range(3):
                control.refresh_motor_status(motor)
                time.sleep(0.01)
            if control.read_motor_param(motor, DM_variable.CTRL_MODE) is None:
                print(
                    f"FAIL: no answer from {key} ({motor.MotorType.name}) on {args.port}"
                    " -- motor power / E-STOP / wrong port?"
                )
                return 1
            print(f"{key} ({motor.MotorType.name}) connected.")

        print("positions before (rad):", fmt(positions()))
        if args.check:
            return 0

        if not args.yes:
            ans = input("Arm held in ZERO pose? This overwrites the motors' stored zero. [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("aborted, nothing written.")
                return 2

        for key, motor in follower.motors.items():
            control.set_zero_position(motor)
            print(f"{key} zeroed.")

        after = positions()
        print("positions after  (rad):", fmt(after))
        off = {k: v for k, v in after.items() if abs(v) > ZERO_TOL_RAD}
        if off:
            print(f"WARN: still not at zero: {fmt(off)}")
            return 1
        print("OK: all motors read ~0 rad.")
        return 0
    finally:
        ser.close()


if __name__ == "__main__":
    raise SystemExit(main())
