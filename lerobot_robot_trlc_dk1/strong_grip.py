"""Strong grip for the right follower gripper.

Behind the session dev toggle ``strong-grip-right`` (dreamhub/dev-toggles.json):
the hub puts the starting member's session toggles in ``DM_DEV_TOGGLES`` when
it spawns a run. The leader's full close sends 1.0, where the fingertips only
just touch, so there is almost no squeeze. With the toggle on, the right
gripper's motor target is the commanded value stretched uniformly by
``STRONG_GRIP_RIGHT_CLOSE_TARGET`` (default 1.10: 0 stays 0, 0.5 becomes 0.55,
1.0 becomes 1.10), and its current limit is ``STRONG_GRIP_RIGHT_TORQUE_NM``
(default 1.0 Nm, the same as without the toggle; cap 3) for the whole run.

Commands and recorded actions stay in 0..1. The measured gripper position is
reported as it is: with the toggle on its upper clamp at 1.0 is lifted, so
observations and recordings show readings up to about 1.1. The C++ loop does
the scaling and the reporting (csrc/strong_grip.h, control_loop.cpp).

Both values come from the environment (the ledream Makefile exports them, so a
rig can override them in /etc/dreammachines/workstation.mk). Invalid values
leave the feature off with a warning; this never raises, so a bad setting can't
stop a run.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TOGGLE = "strong-grip-right"
TORQUE_ENV = "STRONG_GRIP_RIGHT_TORQUE_NM"
CLOSE_TARGET_ENV = "STRONG_GRIP_RIGHT_CLOSE_TARGET"
DEFAULT_TORQUE_NM = 1.0  # same as today's limit: by default only the stretch changes
DEFAULT_CLOSE_TARGET = 1.10
MAX_TORQUE_NM = 3.0  # same hard caps as csrc/strong_grip.h
MAX_CLOSE_TARGET = 1.15


def toggle_on(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return TOGGLE in {n.strip() for n in env.get("DM_DEV_TOGGLES", "").split(",") if n.strip()}


def is_right_arm(label: str, port: str, environ: dict[str, str] | None = None) -> bool:
    """The bimanual follower labels its arms; a single-arm run is recognised by
    its port matching RIGHT_FOLLOWER_PORT (exported by the Makefile)."""
    if label == "right":
        return True
    if label == "left":
        return False
    env = os.environ if environ is None else environ
    right = env.get("RIGHT_FOLLOWER_PORT", "").strip()
    return bool(right and port) and os.path.realpath(port) == os.path.realpath(right)


def _float_env(env: dict[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    return float(raw) if raw else default


def right_arm_settings(environ: dict[str, str] | None = None) -> tuple[float, float] | None:
    """(torque limit Nm, close-target scale) for this run's right arm, or None
    (feature off)."""
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
    logger.warning("dev toggle %s: right gripper target = command x %.3f, limit %.2f Nm",
                   TOGGLE, close_target, torque)
    return torque, close_target
