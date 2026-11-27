"""Dev toggle stiff-arm-asymmetric: the profile, its caps, and what the follower
hands the RT loop (no hardware: DK1RobotRT is replaced by a recorder)."""

import logging

import numpy as np
import pytest

from lerobot_robot_trlc_dk1 import stiff_arm as sa
from lerobot_robot_trlc_dk1 import strong_grip as sg
from trlc_dk1_control.config import DK1RobotConfig

ON = {"DM_DEV_TOGGLES": f"rtc-live,{sa.TOGGLE}"}


# --- the profile -------------------------------------------------------------


def test_toggle_off_means_off():
    assert sa.settings({}) is None
    assert sa.settings({"DM_DEV_TOGGLES": "strong-grip,stiff-arm"}) is None  # other names: off
    assert sa.settings({"DM_DEV_TOGGLES": "rtc-live"}) is None


def test_toggle_on_uses_defaults():
    p = sa.settings(ON)
    for side in sa.SIDES:
        kp, kd = p.gains(side)
        assert kp == (100.0, 250.0, 250.0, 20.0, 20.0, 10.0)
        assert kd == (5.0, 5.0, 5.0, 1.0, 1.0, 1.0)
    assert p.gripper_torque("right") == 1.5
    assert p.gripper_torque("left") == 0.5
    assert p.close_target == 1.10


def test_defaults_change_only_joints_2_and_3():
    base = DK1RobotConfig()
    kp, kd = sa.settings(ON).gains("left")
    changed = [i for i in range(6) if kp[i] != base.arm_kp[i] or kd[i] != base.arm_kd[i]]
    assert changed == [1, 2]


def test_overrides_per_arm():
    env = {
        **ON,
        sa.KP_ENV["left"]: "100,300,200,20,20,10",
        sa.KD_ENV["right"]: "5 4.5 4 1 1 1",  # spaces work too
        sa.GRIPPER_TORQUE_ENV["left"]: "0.7",
        sa.GRIPPER_TORQUE_ENV["right"]: "3",  # the cap itself is allowed
        sa.CLOSE_TARGET_ENV: "1.05",
    }
    p = sa.settings(env)
    assert p.gains("left") == ((100.0, 300.0, 200.0, 20.0, 20.0, 10.0), sa.DEFAULT_KD)
    assert p.gains("right") == (sa.DEFAULT_KP, (5.0, 4.5, 4.0, 1.0, 1.0, 1.0))
    assert (p.gripper_torque("left"), p.gripper_torque("right")) == (0.7, 3.0)
    assert p.close_target == 1.05
    # The caps themselves are allowed.
    assert sa.settings({**ON, sa.KP_ENV["right"]: "500,500,500,500,500,500"}) is not None
    assert sa.settings({**ON, sa.KD_ENV["right"]: "5,5,5,5,5,5"}) is not None
    assert sa.settings({**ON, sa.GRIPPER_TORQUE_ENV["left"]: "0.3"}) is not None


@pytest.mark.parametrize("name,value", [
    (sa.KP_ENV["left"], "100,501,250,20,20,10"),  # kp above 500
    (sa.KP_ENV["right"], "100,-1,250,20,20,10"),  # negative
    (sa.KP_ENV["left"], "100,250,250,20,20"),  # 5 joints
    (sa.KP_ENV["left"], "100,250,250,20,20,10,1"),  # 7 joints
    (sa.KP_ENV["left"], "100,nan,250,20,20,10"),
    (sa.KD_ENV["left"], "5,8,6.3,1,1,1"),  # kd above the MIT frame's 5
    (sa.KD_ENV["right"], "5,5,x,1,1,1"),
    (sa.GRIPPER_TORQUE_ENV["left"], "0.2"),  # below the 0.3 minimum
    (sa.GRIPPER_TORQUE_ENV["right"], "3.1"),  # above the 3 Nm cap
    (sa.GRIPPER_TORQUE_ENV["right"], "abc"),
    (sa.CLOSE_TARGET_ENV, "1.15"),  # above strong grip's 1.10
    (sa.CLOSE_TARGET_ENV, "0.9"),
])
def test_invalid_values_leave_the_whole_profile_off(name, value, caplog):
    with caplog.at_level(logging.WARNING):
        assert sa.settings({**ON, name: value}) is None
    assert "stays OFF" in caplog.text
    # ... and strong grip with it (unless strong-grip itself is on).
    assert sg.settings({**ON, name: value}) is None


def test_caps_match_the_mit_frame_and_strong_grip():
    from trlc_dk1_control.config import KD_MAX, KP_MAX

    assert (KP_MAX, KD_MAX) == (500.0, 5.0)  # csrc/dm_protocol.h build_mit_frame ranges
    assert sa.MAX_GRIPPER_TORQUE_NM == sg.MAX_TORQUE_NM == 3.0
    assert sa.MAX_CLOSE_TARGET == sg.MAX_CLOSE_TARGET == 1.10


# --- arm side ----------------------------------------------------------------


def test_arm_side_from_label_then_port(tmp_path):
    left, right = tmp_path / "ttyL", tmp_path / "ttyR"
    left.touch()
    right.touch()
    link = tmp_path / "by-id-right"
    link.symlink_to(right)
    env = {"LEFT_FOLLOWER_PORT": str(left), "RIGHT_FOLLOWER_PORT": str(link)}
    assert sa.arm_side("left", str(right), env) == "left"  # the label wins
    assert sa.arm_side("right", "", {}) == "right"
    assert sa.arm_side("", str(right), env) == "right"  # through the symlink
    assert sa.arm_side("F4", str(left), env) == "left"
    assert sa.arm_side("", str(tmp_path / "other"), env) is None
    assert sa.arm_side("", str(left), {}) is None
    assert sa.arm_side("", str(left), {"LEFT_FOLLOWER_PORT": str(left), "RIGHT_FOLLOWER_PORT": str(left)}) is None


def test_unknown_side_shares_equal_gains_and_keeps_the_usual_gripper_limit():
    p = sa.settings(ON)
    assert p.gains(None) == (sa.DEFAULT_KP, sa.DEFAULT_KD)
    assert p.gripper_torque(None) == 1.0
    differ = sa.settings({**ON, sa.KP_ENV["left"]: "100,200,200,20,20,10"})
    assert differ.gains(None) is None


# --- strong grip under the profile ---------------------------------------------


def test_profile_turns_strong_grip_on_with_its_values():
    assert sg.settings(ON, side="right") == (1.5, 1.10)
    assert sg.settings(ON, side="left") == (0.5, 1.10)
    assert sg.settings(ON) == (1.0, 1.10)  # side unknown (VR teleop, gripper boost)
    assert sg.settings({**ON, sa.CLOSE_TARGET_ENV: "1.05"}, side="left") == (0.5, 1.05)


def test_strong_grip_alone_is_unchanged():
    env = {"DM_DEV_TOGGLES": sg.TOGGLE}
    assert sg.settings(env) == (1.0, 1.10)
    assert sg.settings(env, side="left") == (1.0, 1.10)
    # The profile's settings do nothing without its toggle.
    assert sg.settings({**env, sa.GRIPPER_TORQUE_ENV["right"]: "2", sa.CLOSE_TARGET_ENV: "1.0"},
                       side="right") == (1.0, 1.10)


def test_both_toggles_on_the_profile_wins_and_says_so(caplog):
    env = {"DM_DEV_TOGGLES": f"{sg.TOGGLE},{sa.TOGGLE}", sg.TORQUE_ENV: "2.0", sg.CLOSE_TARGET_ENV: "1.02"}
    with caplog.at_level(logging.WARNING):
        assert sg.settings(env, side="left") == (0.5, 1.10)
    assert "both on" in caplog.text
    # An invalid profile falls back to strong grip's own values.
    assert sg.settings({**env, sa.KD_ENV["left"]: "9,9,9,9,9,9"}, side="left") == (2.0, 1.02)


# --- what the follower hands the RT loop ---------------------------------------


class _RecordingRT:
    configs: list = []

    def __init__(self, config):
        _RecordingRT.configs.append(config)

    def connect(self):
        pass


@pytest.fixture
def connect(monkeypatch):
    """Connect a follower with a recorder in place of the RT loop; returns the config."""
    follower = pytest.importorskip("lerobot_robot_trlc_dk1.follower")
    from trlc_dk1_control import rt_robot

    monkeypatch.setattr(rt_robot, "DK1RobotRT", _RecordingRT)
    for name in ["DM_DEV_TOGGLES", "LEFT_FOLLOWER_PORT", "RIGHT_FOLLOWER_PORT",
                 *sa.KP_ENV.values(), *sa.KD_ENV.values(), *sa.GRIPPER_TORQUE_ENV.values(),
                 sa.CLOSE_TARGET_ENV, sg.TORQUE_ENV, sg.CLOSE_TARGET_ENV, "DM_TELEOP_LOG_DIR"]:
        monkeypatch.delenv(name, raising=False)

    def _connect(label="", port="/dev/ttyFAKE0", **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        _RecordingRT.configs.clear()
        arm = follower.DK1Follower(follower.DK1FollowerConfig(port=port, label=label))
        arm._connect_rt()
        (cfg,) = _RecordingRT.configs
        return cfg

    return _connect


def test_follower_toggle_off_is_byte_identical(connect):
    base = DK1RobotConfig()
    for env in ({}, {"DM_DEV_TOGGLES": "rtc-live"},
                {sa.KP_ENV["right"]: "500,500,500,500,500,500"}):  # settings alone do nothing
        cfg = connect(label="right", **env)
        assert cfg.arm_kp.tobytes() == base.arm_kp.tobytes()
        assert cfg.arm_kd.tobytes() == base.arm_kd.tobytes()
        assert cfg.arm_kp.dtype == base.arm_kp.dtype and cfg.arm_kd.dtype == base.arm_kd.dtype
        assert cfg.max_gripper_torque_nm == 1.0
        assert cfg.strong_gripper_torque_nm == 0.0


def test_follower_toggle_on_gets_this_arms_values(connect, caplog):
    with caplog.at_level(logging.WARNING):
        right = connect(label="right", DM_DEV_TOGGLES=sa.TOGGLE)
        left = connect(label="left", DM_DEV_TOGGLES=sa.TOGGLE,
                       **{sa.KP_ENV["left"]: "100,200,200,20,20,10"})
    np.testing.assert_array_equal(right.arm_kp, [100, 250, 250, 20, 20, 10])
    np.testing.assert_array_equal(right.arm_kd, [5, 5, 5, 1, 1, 1])
    assert right.strong_gripper_torque_nm == 1.5
    np.testing.assert_array_equal(left.arm_kp, [100, 200, 200, 20, 20, 10])
    assert left.strong_gripper_torque_nm == 0.5
    # max_gripper_torque_nm (the base limit) is untouched: strong grip carries the limit.
    assert right.max_gripper_torque_nm == left.max_gripper_torque_nm == 1.0
    assert "Stiff arm ON (right arm): kp=[100, 250, 250, 20, 20, 10] kd=[5, 5, 5, 1, 1, 1], " \
           "gripper limit 1.50 Nm" in caplog.text
    assert "Stiff arm ON (left arm): kp=[100, 200, 200, 20, 20, 10]" in caplog.text


def test_follower_single_arm_by_port(connect, tmp_path):
    port = tmp_path / "ttyACM1"
    port.touch()
    cfg = connect(port=str(port), DM_DEV_TOGGLES=sa.TOGGLE, RIGHT_FOLLOWER_PORT=str(port),
                  LEFT_FOLLOWER_PORT=str(tmp_path / "ttyACM0"))
    assert cfg.strong_gripper_torque_nm == 1.5


def test_follower_unknown_side_logs_and_uses_the_usual_gripper_limit(connect, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = connect(DM_DEV_TOGGLES=sa.TOGGLE)
    np.testing.assert_array_equal(cfg.arm_kp, sa.DEFAULT_KP)
    assert cfg.strong_gripper_torque_nm == 1.0
    assert "cannot tell left from right" in caplog.text


def test_follower_invalid_profile_runs_as_today(connect):
    base = DK1RobotConfig()
    cfg = connect(label="left", DM_DEV_TOGGLES=sa.TOGGLE, **{sa.KD_ENV["left"]: "5,8,6.3,1,1,1"})
    assert cfg.arm_kp.tobytes() == base.arm_kp.tobytes()
    assert cfg.arm_kd.tobytes() == base.arm_kd.tobytes()
    assert cfg.strong_gripper_torque_nm == 0.0
