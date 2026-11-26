#pragma once

#include <algorithm>
#include <cmath>

namespace trlc {

// Strong grip near closed (control_loop.cpp, gripper EMIT frame).
//
// The gripper runs position + current limit (EMIT): once the fingers touch
// something, the current limit sets the squeeze. Normally that limit is
// max_gripper_torque_nm everywhere. With strong grip on, the limit rises to
// strong_nm while the MEASURED gripper is closed past `min_closed` (normalized
// 0 = open, 1 = closed), and falls back to base_nm above it. A finger caught
// in the gripper stops it far wider than the threshold, so it never sees the
// stronger limit. The change ramps linearly over `ramp_cycles` cycles both
// ways, and a small hysteresis keeps it from chattering at the boundary.
//
// Overshoot: the leader's full close sends 1.0, where the fingertips only just
// touch, so there is almost no squeeze. With strong grip on, a commanded value
// at or past `min_closed` is remapped linearly from [min_closed, 1] onto
// [min_closed, close_target] (e.g. 1.10), so the position target sits past
// contact and the current limit sets the squeeze. Continuous at min_closed, no
// jump. Only the motor target changes; commands and recorded actions stay in
// 0..1. While strong grip is on, control_loop.cpp also reports the measured
// position without its upper clamp at 1.0 (gripper_obs_max_), so observations
// show how far the gripper really closed.
//
// strong_nm <= 0 means off: limit() returns base_nm and target() returns the
// command unchanged, exactly the old behaviour.

constexpr double kStrongGripMaxNm = 3.0;       // hard cap, checked at start()
constexpr double kStrongGripMaxCloseTarget = 1.15;  // hard cap on the overshoot target
constexpr double kStrongGripHysteresis = 0.02; // normalized; release below min_closed - this

class StrongGrip {
public:
    StrongGrip(double base_nm, double strong_nm, double min_closed, double ramp_cycles,
               double close_target = 1.0)
        : base_nm_(base_nm),
          strong_nm_(strong_nm),
          min_closed_(min_closed),
          close_target_(close_target),
          cur_nm_(base_nm) {
        const double span = std::abs(strong_nm - base_nm);
        step_nm_ = ramp_cycles >= 1.0 ? span / ramp_cycles : span;
    }

    bool active() const { return strong_nm_ > 0.0; }
    bool engaged() const { return engaged_; }

    // One control cycle. `closed` = measured gripper position, normalized
    // (0 open, 1 closed; values outside are clamped). Returns the torque
    // limit in Nm for this cycle's EMIT frame.
    double limit(double closed) {
        if (!active()) return base_nm_;
        closed = std::clamp(closed, 0.0, 1.0);
        if (engaged_) {
            if (closed < min_closed_ - kStrongGripHysteresis) engaged_ = false;
        } else if (closed >= min_closed_) {
            engaged_ = true;
        }
        const double target = engaged_ ? strong_nm_ : base_nm_;
        if (cur_nm_ < target) cur_nm_ = std::min(cur_nm_ + step_nm_, target);
        else if (cur_nm_ > target) cur_nm_ = std::max(cur_nm_ - step_nm_, target);
        return cur_nm_;
    }

    // Commanded gripper value (0..1) -> normalized motor target (may exceed 1).
    double target(double cmd) const {
        if (!active() || cmd < min_closed_ || min_closed_ >= 1.0) return cmd;
        return min_closed_ + (cmd - min_closed_) * (close_target_ - min_closed_) / (1.0 - min_closed_);
    }

private:
    double base_nm_;
    double strong_nm_;
    double min_closed_;
    double close_target_;
    double step_nm_ = 0.0;
    double cur_nm_;
    bool engaged_ = false;
};

} // namespace trlc
