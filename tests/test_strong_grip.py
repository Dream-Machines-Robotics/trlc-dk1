from lerobot_robot_trlc_dk1 import strong_grip as sg


def test_toggle_off_means_off():
    assert sg.right_arm_settings({"DM_DEV_TOGGLES": "rtc-live"}) is None
    assert sg.right_arm_settings({}) is None


def test_toggle_on_uses_defaults():
    assert sg.right_arm_settings({"DM_DEV_TOGGLES": f"rtc-live,{sg.TOGGLE}"}) == (
        sg.DEFAULT_TORQUE_NM,
        sg.DEFAULT_MIN_CLOSED,
        sg.DEFAULT_CLOSE_TARGET,
    )


def test_default_torque_equals_todays_limit():
    torque, _, _ = sg.right_arm_settings({"DM_DEV_TOGGLES": sg.TOGGLE})
    assert torque == 1.0
    assert sg.right_arm_settings({"DM_DEV_TOGGLES": sg.TOGGLE, sg.TORQUE_ENV: "3.0"})[0] == 3.0


def test_overrides_from_env():
    env = {"DM_DEV_TOGGLES": sg.TOGGLE, sg.TORQUE_ENV: "2.0", sg.MIN_CLOSED_ENV: "0.8",
           sg.CLOSE_TARGET_ENV: "1.05"}
    assert sg.right_arm_settings(env) == (2.0, 0.8, 1.05)


def test_invalid_values_leave_it_off():
    on = {"DM_DEV_TOGGLES": sg.TOGGLE}
    assert sg.right_arm_settings({**on, sg.TORQUE_ENV: "3.5"}) is None  # above the hard cap
    assert sg.right_arm_settings({**on, sg.TORQUE_ENV: "0"}) is None
    assert sg.right_arm_settings({**on, sg.MIN_CLOSED_ENV: "1.2"}) is None
    assert sg.right_arm_settings({**on, sg.MIN_CLOSED_ENV: "abc"}) is None
    assert sg.right_arm_settings({**on, sg.MIN_CLOSED_ENV: "1.0"}) is None  # remap needs thr < 1
    assert sg.right_arm_settings({**on, sg.CLOSE_TARGET_ENV: "1.2"}) is None  # above the cap
    assert sg.right_arm_settings({**on, sg.CLOSE_TARGET_ENV: "0.9"}) is None


def test_is_right_arm():
    assert sg.is_right_arm("right", "/dev/x", {})
    assert not sg.is_right_arm("left", "/dev/x", {"RIGHT_FOLLOWER_PORT": "/dev/x"})
    assert sg.is_right_arm("", "/dev/x", {"RIGHT_FOLLOWER_PORT": "/dev/x"})
    assert not sg.is_right_arm("", "/dev/y", {"RIGHT_FOLLOWER_PORT": "/dev/x"})
    assert not sg.is_right_arm("", "/dev/x", {})
