"""Stiff arm, asymmetric: one tuning profile for pushing tasks (both followers).

Behind the session dev toggle ``stiff-arm-asymmetric`` (dreamhub/dev-toggles.json),
read from ``DM_DEV_TOGGLES`` like strong grip. Written for inserting zip ties with
VR teleop: pushing takes about 15 N, and with the normal gains the VR target has
to run about 50 mm ahead of the arm, so the arm drops onto the table once the zip
tie clicks. A stiffer arm gives the same force from a smaller gap.

With the toggle on, one switch turns on all of this:

1. Joint gains per arm (``arm_kp`` / ``arm_kd`` of the RT loop). Default: joints 2
   and 3 kp 100 -> 200; kd stays at 5.0 on joint 2 and goes 4 -> 5.0 on joint 3.
   5.0 is the most the motor's MIT frame can carry (csrc/dm_protocol.h packs kd
   into 0..5 and clamps anything above), so a higher kd would never reach the
   motor. All other joints unchanged.
2. A working over-torque trip on joints 1 to 3 (``joint_torque_limits`` of the RT
   loop, csrc/safety.h): 27 Nm instead of 28. The DM4340 reports torque in -28 to
   +28 Nm, so at the normal 28 Nm limit a reading can never be above it and the
   trip never fires. The same limit clips the gravity feed-forward, which stays
   far below it (worst case in the follower model: 12.8 Nm on joint 2).
3. Strong grip (strong_grip.py): VR gripper action x the close target (default
   1.10), followers accept commands up to 1.10 and report their gripper position
   unclamped above 1.0. ``strong_grip.settings()`` answers for this toggle too,
   so everything that checks "strong grip active" (VR teleop, the gripper-boost
   cap) follows it.
4. Gripper torque limit per arm: right 1.5 Nm, left 0.5 Nm.

Every value can be changed without code changes through environment variables
(the ledream Makefile exports them, so a rig sets them in
/etc/dreammachines/workstation.mk or on the make command line). An invalid value
leaves the whole profile off with a warning, the same as strong grip: this never
raises, so a bad setting can't stop a run.

Which arm is which: the bimanual follower labels its arms "left" and "right". A
single-arm run is matched by its port against LEFT_FOLLOWER_PORT /
RIGHT_FOLLOWER_PORT (exported by the Makefile). When neither tells the side, the
arm gets the shared joint settings if left and right agree (they do by default)
and the usual 1.0 Nm gripper limit, and the log says so.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from trlc_dk1_control.config import DM4310_T_MAX, DM4340_T_MAX, KD_MAX, KP_MAX

from lerobot_robot_trlc_dk1 import strong_grip

logger = logging.getLogger(__name__)

TOGGLE = "stiff-arm-asymmetric"
SIDES = ("left", "right")
KP_ENV = {s: f"STIFF_ARM_ASYM_KP_{s.upper()}" for s in SIDES}
KD_ENV = {s: f"STIFF_ARM_ASYM_KD_{s.upper()}" for s in SIDES}
TORQUE_LIMIT_ENV = {s: f"STIFF_ARM_ASYM_TORQUE_LIMIT_{s.upper()}" for s in SIDES}
GRIPPER_TORQUE_ENV = {s: f"STIFF_ARM_ASYM_GRIPPER_TORQUE_NM_{s.upper()}" for s in SIDES}
CLOSE_TARGET_ENV = "STIFF_ARM_ASYM_CLOSE_TARGET"
PORT_ENV = {s: f"{s.upper()}_FOLLOWER_PORT" for s in SIDES}

# Joints [j1..j6]. Normal values are trlc_dk1_control/config.py arm_kp, arm_kd and
# joint_torque_limits.
DEFAULT_KP = (100.0, 200.0, 200.0, 20.0, 20.0, 10.0)
DEFAULT_KD = (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)
DEFAULT_TORQUE_LIMIT = (27.0, 27.0, 27.0, 10.0, 10.0, 10.0)
DEFAULT_GRIPPER_TORQUE_NM = {"left": 0.5, "right": 1.5}
DEFAULT_CLOSE_TARGET = strong_grip.DEFAULT_CLOSE_TARGET
# Hard caps. kp and kd are the MIT frame's ranges (above them the value is clamped
# on the way to the motor). The torque limit is at most the motor's reporting
# range (DM4340 on joints 1-3, DM4310 on 4-6) and at least well above the gravity
# load (follower model, worst case: joint 2 12.8 Nm, joint 3 6.1, joints 4-6 1.3),
# so a push never trips it by gravity alone: a trip drops the arm into damping
# mode, where it sags. The gripper minimum keeps a grip that still closes on a
# part; its maximum and the close target's are strong grip's.
MIN_TORQUE_LIMIT = (20.0, 20.0, 20.0, 3.0, 3.0, 3.0)
MAX_TORQUE_LIMIT = (DM4340_T_MAX,) * 3 + (DM4310_T_MAX,) * 3
MIN_GRIPPER_TORQUE_NM = 0.3
MAX_GRIPPER_TORQUE_NM = strong_grip.MAX_TORQUE_NM
MAX_CLOSE_TARGET = strong_grip.MAX_CLOSE_TARGET
# Gripper limit for an arm whose side is unknown: the usual limit, as without the toggle.
UNKNOWN_SIDE_GRIPPER_TORQUE_NM = 1.0


@dataclass(frozen=True)
class ArmSettings:
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    torque_limit: tuple[float, ...]


@dataclass(frozen=True)
class Profile:
    arms: dict[str, ArmSettings]
    gripper_torque_nm: dict[str, float]
    close_target: float

    def arm(self, side: str | None) -> ArmSettings | None:
        """Joint settings for this arm. Side unknown: the shared settings if left and
        right agree, else None (the arm keeps its normal values)."""
        if side in SIDES:
            return self.arms[side]
        if self.arms["left"] == self.arms["right"]:
            return self.arms["left"]
        return None

    def gripper_torque(self, side: str | None) -> float:
        if side in SIDES:
            return self.gripper_torque_nm[side]
        return UNKNOWN_SIDE_GRIPPER_TORQUE_NM


def toggle_on(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return TOGGLE in {n.strip() for n in env.get("DM_DEV_TOGGLES", "").split(",") if n.strip()}


def arm_side(label: str, port: str, environ: dict[str, str] | None = None) -> str | None:
    """"left" / "right" from the follower's label, else from its port matching
    LEFT_FOLLOWER_PORT / RIGHT_FOLLOWER_PORT; None when neither tells."""
    if label in SIDES:
        return label
    env = os.environ if environ is None else environ
    if not port:
        return None
    matches = [s for s in SIDES
               if env.get(PORT_ENV[s], "").strip()
               and os.path.realpath(env[PORT_ENV[s]].strip()) == os.path.realpath(port)]
    return matches[0] if len(matches) == 1 else None


def _joint_list(env: dict[str, str], name: str, default: tuple[float, ...],
                lo: tuple[float, ...], hi: tuple[float, ...]) -> tuple[float, ...]:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        values = tuple(float(v) for v in raw.replace(",", " ").split())
    except ValueError:
        raise ValueError(f"{name} must be 6 numbers, got {raw!r}") from None
    if len(values) != 6:
        raise ValueError(f"{name} must be 6 numbers (joints 1 to 6), got {len(values)}")
    for i, v in enumerate(values):
        if not (math.isfinite(v) and lo[i] <= v <= hi[i]):
            raise ValueError(f"{name} joint {i + 1} must be in [{lo[i]:g}, {hi[i]:g}], got {raw!r}")
    return values


def _float(env: dict[str, str], name: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo:g}, {hi:g}], got {value:g}")
    return value


def settings(environ: dict[str, str] | None = None) -> Profile | None:
    """The profile for this run, or None (toggle off, or a setting out of range)."""
    if not toggle_on(environ):
        return None
    env = os.environ if environ is None else environ
    zeros, kp_max, kd_max = (0.0,) * 6, (KP_MAX,) * 6, (KD_MAX,) * 6
    try:
        return Profile(
            arms={
                s: ArmSettings(
                    kp=_joint_list(env, KP_ENV[s], DEFAULT_KP, zeros, kp_max),
                    kd=_joint_list(env, KD_ENV[s], DEFAULT_KD, zeros, kd_max),
                    torque_limit=_joint_list(env, TORQUE_LIMIT_ENV[s], DEFAULT_TORQUE_LIMIT,
                                             MIN_TORQUE_LIMIT, MAX_TORQUE_LIMIT),
                )
                for s in SIDES
            },
            gripper_torque_nm={
                s: _float(env, GRIPPER_TORQUE_ENV[s], DEFAULT_GRIPPER_TORQUE_NM[s],
                          MIN_GRIPPER_TORQUE_NM, MAX_GRIPPER_TORQUE_NM)
                for s in SIDES
            },
            close_target=_float(env, CLOSE_TARGET_ENV, DEFAULT_CLOSE_TARGET, 1.0, MAX_CLOSE_TARGET),
        )
    except ValueError as e:
        logger.warning("dev toggle %s is on, but %s. Stiff arm (and its strong grip) stays OFF", TOGGLE, e)
        return None


def _fmt(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def log_on(who: str, side: str | None, arm: ArmSettings | None, gripper_torque_nm: float) -> None:
    """The start-of-run line for one follower (follower.py _connect_rt)."""
    which = f"{side} arm" if side else "arm side unknown"
    if side is None:
        logger.warning(
            "%s: dev toggle %s: cannot tell left from right (no left/right label, port is not "
            "LEFT_FOLLOWER_PORT or RIGHT_FOLLOWER_PORT)%s",
            who, TOGGLE,
            "" if arm is None else "; using the joint settings left and right share",
        )
    if arm is None:
        logger.warning("%s: Stiff arm ON (%s): left and right joint settings differ, so this arm keeps "
                       "its normal gains and torque limits; gripper limit %.2f Nm", who, which, gripper_torque_nm)
        return
    logger.warning("%s: Stiff arm ON (%s): kp=%s kd=%s, torque trip %s Nm, gripper limit %.2f Nm",
                   who, which, _fmt(arm.kp), _fmt(arm.kd), _fmt(arm.torque_limit), gripper_torque_nm)
