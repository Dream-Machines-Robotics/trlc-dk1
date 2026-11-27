from lerobot_robot_trlc_dk1 import strong_grip as sg


def test_toggle_off_means_off():
    assert sg.settings({"DM_DEV_TOGGLES": "rtc-live"}) is None
    assert sg.settings({"DM_DEV_TOGGLES": "strong-grip-right"}) is None  # old name: off
    assert sg.settings({}) is None


def test_toggle_on_uses_defaults():
    assert sg.settings({"DM_DEV_TOGGLES": f"rtc-live,{sg.TOGGLE}"}) == (1.0, 1.10)


def test_overrides_from_env():
    env = {"DM_DEV_TOGGLES": sg.TOGGLE, sg.TORQUE_ENV: "3.0", sg.CLOSE_TARGET_ENV: "1.05"}
    assert sg.settings(env) == (3.0, 1.05)
    assert sg.settings({**env, sg.CLOSE_TARGET_ENV: "1.10"})[1] == 1.10  # the cap itself is allowed


def test_invalid_values_leave_it_off():
    on = {"DM_DEV_TOGGLES": sg.TOGGLE}
    assert sg.settings({**on, sg.TORQUE_ENV: "3.5"}) is None  # above the hard cap
    assert sg.settings({**on, sg.TORQUE_ENV: "0"}) is None
    assert sg.settings({**on, sg.TORQUE_ENV: "abc"}) is None
    assert sg.settings({**on, sg.CLOSE_TARGET_ENV: "1.15"}) is None  # above the 1.10 cap
    assert sg.settings({**on, sg.CLOSE_TARGET_ENV: "0.9"}) is None
