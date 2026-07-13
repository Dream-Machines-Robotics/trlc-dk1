#pragma once

namespace trlc {

// One cycle of the impedance loop's position-ramp limiter (control_loop.cpp
// step 7). `diff` is the remaining distance from the current slew target to
// the commanded position; the return value is the step to take this cycle.
//
// Velocity: the step is clamped to ±max_delta (rad/cycle) — the slew-rate cap
// that joint_velocity_scaling scales.
//
// Acceleration guard (optional, max_accel > 0, rad/cycle²): the step may grow
// at most max_accel per cycle beyond `prev_step` (last cycle's return value),
// so a discontinuous position command ramps commanded velocity up gradually
// instead of stepping 0 → max_delta in a single 4 ms cycle. That velocity
// step-change is what spikes torque (and supply current) on every joint at
// once when a policy jumps from rest to an extended pose.
//
// The guard is asymmetric on purpose: a step that shrinks toward the target
// is never held back, so the ramp settles on the commanded position exactly
// with no overshoot (a symmetric cap would carry v²/2a of overshoot past the
// target). On a direction reversal the step restarts from zero at max_accel
// per cycle — gentler than the unguarded full reversal within ±max_delta.
inline double slew_step(double diff, double prev_step, double max_delta, double max_accel) {
    if (diff > max_delta) diff = max_delta;
    else if (diff < -max_delta) diff = -max_delta;
    if (max_accel > 0.0) {
        if (diff > 0.0) {
            const double cap = (prev_step > 0.0 ? prev_step : 0.0) + max_accel;
            if (diff > cap) diff = cap;
        } else if (diff < 0.0) {
            const double cap = (prev_step < 0.0 ? prev_step : 0.0) - max_accel;
            if (diff < cap) diff = cap;
        }
    }
    return diff;
}

} // namespace trlc
