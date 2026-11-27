"""Strong grip for the follower grippers (both arms).

Behind the session dev toggle ``strong-grip`` (dreamhub/dev-toggles.json): the
hub puts the starting member's session toggles in ``DM_DEV_TOGGLES`` when it
spawns a run. A full close sends 1.0, where the fingertips only just touch, so
there is almost no squeeze.

With the toggle on:

- VR teleop (scripts/vr_teleop/teleop/bi_quest_teleop.py) stretches each
  gripper action uniformly by ``STRONG_GRIP_CLOSE_TARGET`` (default 1.10: 0 stays
  0, 0.5 becomes 0.55, 1.0 becomes 1.10). The stretch happens where the action
  is made, so the recorded action equals what the follower is sent, and a policy
  trained on such data outputs 0..1.10 itself. The DK1 leader is not stretched.
- Each follower's RT loop accepts gripper commands up to 1.10 instead of 1.0,
  uses ``STRONG_GRIP_TORQUE_NM`` (default 1.0 Nm, the same as without the
  toggle; cap 3) as its gripper limit, and reports the measured position without
  its upper clamp at 1.0 (csrc/strong_grip.h, control_loop.cpp).

The dev toggle ``stiff-arm-asymmetric`` (stiff_arm.py) turns strong grip on too,
with its own close target and a gripper limit per arm: ``settings()`` answers
for it, and its values win when both toggles are on.

Both values come from the environment (the ledream Makefile exports them, so a
rig can override them in /etc/dreammachines/workstation.mk). Invalid values
leave the feature off with a warning; this never raises, so a bad setting can't
stop a run.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TOGGLE = "strong-grip"
TORQUE_ENV = "STRONG_GRIP_TORQUE_NM"
CLOSE_TARGET_ENV = "STRONG_GRIP_CLOSE_TARGET"
DEFAULT_TORQUE_NM = 1.0  # same as today's limit: by default only the stretch changes
DEFAULT_CLOSE_TARGET = 1.10
MAX_TORQUE_NM = 3.0  # same hard caps as csrc/strong_grip.h
MAX_CLOSE_TARGET = 1.10


def toggle_on(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return TOGGLE in {n.strip() for n in env.get("DM_DEV_TOGGLES", "").split(",") if n.strip()}


def _float_env(env: dict[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    return float(raw) if raw else default


def settings(environ: dict[str, str] | None = None, side: str | None = None) -> tuple[float, float] | None:
    """(torque limit Nm, close-target scale) for this run, or None (feature off).

    ``side`` ("left" / "right", or None when unknown) picks the arm's limit under
    stiff-arm-asymmetric; strong grip alone has one limit for both arms.
    """
    from lerobot_robot_trlc_dk1 import stiff_arm  # here, not at the top: stiff_arm imports this module

    profile = stiff_arm.settings(environ)
    if profile is not None:
        torque = profile.gripper_torque(side)
        if toggle_on(environ):
            logger.warning("dev toggles %s and %s are both on: %s's values win (gripper limit %.2f Nm, "
                           "VR action x %.3f)", TOGGLE, stiff_arm.TOGGLE, stiff_arm.TOGGLE,
                           torque, profile.close_target)
        return torque, profile.close_target
    if not toggle_on(environ):
        return None
    env = os.environ if environ is None else environ
    try:
        torque = _float_env(env, TORQUE_ENV, DEFAULT_TORQUE_NM)
        close_target = _float_env(env, CLOSE_TARGET_ENV, DEFAULT_CLOSE_TARGET)
        if not (0.0 < torque <= MAX_TORQUE_NM):
            raise ValueError(f"{TORQUE_ENV} must be in (0, {MAX_TORQUE_NM}], got {torque}")
        if not (1.0 <= close_target <= MAX_CLOSE_TARGET):
            raise ValueError(f"{CLOSE_TARGET_ENV} must be in [1, {MAX_CLOSE_TARGET}], got {close_target}")
    except ValueError as e:
        logger.warning("dev toggle %s is on, but %s. Strong grip stays OFF", TOGGLE, e)
        return None
    logger.warning("dev toggle %s: gripper VR action x %.3f, follower limit %.2f Nm",
                   TOGGLE, close_target, torque)
    return torque, close_target
