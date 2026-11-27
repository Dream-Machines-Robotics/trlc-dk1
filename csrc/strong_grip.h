#pragma once

namespace trlc {

// Strong grip (control_loop.cpp), dev toggle strong-grip on the Python
// side. The leader's full close sends 1.0, where the fingertips only just
// touch. With strong grip on, the teleop (VR) sends gripper actions
// stretched up to about 1.10, and this loop:
//   - accepts gripper commands up to kStrongGripMaxCloseTarget instead of 1.0
//     (command_gripper), so the recorded action is exactly what is sent;
//   - uses strong_gripper_torque_nm as the gripper current limit;
//   - reports the measured position without its upper clamp at 1.0
//     (gripper_obs_max_), so observations show the real closure.
// Off (strong_gripper_torque_nm <= 0): command clamp, limit and reported
// position exactly as before.

constexpr double kStrongGripMaxNm = 3.0;            // hard cap, checked at start()
constexpr double kStrongGripMaxCloseTarget = 1.10;  // command clamp while on

inline double strong_grip_cmd_max(bool on) { return on ? kStrongGripMaxCloseTarget : 1.0; }

} // namespace trlc
