#pragma once

namespace trlc {

// Strong grip (control_loop.cpp, gripper EMIT frame), dev toggle
// strong-grip-right on the Python side.
//
// The gripper runs position + current limit (EMIT). The leader's full close
// sends 1.0, where the fingertips only just touch, so there is almost no
// squeeze. With strong grip on, the commanded value (0..1) is stretched
// uniformly onto the motor target: target = command * scale (e.g. 1.10, so 0
// stays 0, 0.5 becomes 0.55 and 1.0 becomes 1.10). The current limit is
// strong_gripper_torque_nm for the whole run. Commands and recorded actions
// stay in 0..1; control_loop.cpp reports the measured position without its
// upper clamp at 1.0 (gripper_obs_max_) so observations show the real closure.
//
// Off (strong_gripper_torque_nm <= 0): the target is the command, the limit
// max_gripper_torque_nm, exactly the old behaviour.

constexpr double kStrongGripMaxNm = 3.0;           // hard cap, checked at start()
constexpr double kStrongGripMaxCloseTarget = 1.15;  // hard cap on the scale

inline double strong_grip_target(double cmd, bool on, double scale) {
    return on ? cmd * scale : cmd;
}

} // namespace trlc
