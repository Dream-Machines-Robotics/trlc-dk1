#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>

namespace trlc {

struct SafetyState {
    bool damping_mode = false;
    int overcurrent_count = 0;
    int overspeed_count = 0;
};

// Apply safety checks, modifying q_des, kp, and tau_ff in-place.
// ndof: number of joints (typically 6)
inline void apply_safety(
    const double* pos, const double* vel, const double* torque,
    double* q_des, double* kp, double* tau_ff,
    const double* pos_limits_lo, const double* pos_limits_hi,
    const double* torque_limits, const double* velocity_limits,
    double limit_buffer,
    int overcurrent_threshold, int overspeed_threshold,
    int ndof,
    SafetyState& state)
{
    // Position clamping with buffer
    for (int i = 0; i < ndof; ++i) {
        double lo = pos_limits_lo[i] + limit_buffer;
        double hi = pos_limits_hi[i] - limit_buffer;
        q_des[i] = std::clamp(q_des[i], lo, hi);
    }

    // Torque limit clipping
    for (int i = 0; i < ndof; ++i) {
        tau_ff[i] = std::clamp(tau_ff[i], -torque_limits[i], torque_limits[i]);
    }

    const bool was_damping = state.damping_mode;

    // Over-speed detection (track worst offender for the diagnostic log)
    bool any_overspeed = false;
    int overspeed_joint = -1;
    double overspeed_val = 0.0, overspeed_lim = 0.0;
    for (int i = 0; i < ndof; ++i) {
        if (std::abs(vel[i]) > velocity_limits[i]) {
            any_overspeed = true;
            if (std::abs(vel[i]) >= std::abs(overspeed_val)) {
                overspeed_joint = i; overspeed_val = vel[i]; overspeed_lim = velocity_limits[i];
            }
        }
    }

    if (any_overspeed) {
        ++state.overspeed_count;
        if (state.overspeed_count >= overspeed_threshold) {
            state.damping_mode = true;
        }
    } else {
        state.overspeed_count = std::max(0, state.overspeed_count - 1);
    }

    // Over-current detection (track worst offender for the diagnostic log)
    bool any_over = false;
    int overcurrent_joint = -1;
    double overcurrent_val = 0.0, overcurrent_lim = 0.0;
    for (int i = 0; i < ndof; ++i) {
        if (std::abs(torque[i]) > torque_limits[i]) {
            any_over = true;
            if (std::abs(torque[i]) >= std::abs(overcurrent_val)) {
                overcurrent_joint = i; overcurrent_val = torque[i]; overcurrent_lim = torque_limits[i];
            }
        }
    }

    if (any_over) {
        ++state.overcurrent_count;
        if (state.overcurrent_count >= overcurrent_threshold) {
            state.damping_mode = true;
        }
    } else {
        state.overcurrent_count = std::max(0, state.overcurrent_count - 1);
    }

    // Edge-triggered diagnostic: damping mode is otherwise SILENT (it just
    // zeros kp + gravity-comp, so the arm sags with no log line). Print WHY,
    // once, on the false->true transition. Joints are 0-based (index 0 ==
    // joint_1). Damping latches until reset_errors().
    if (!was_damping && state.damping_mode) {
        if (overspeed_joint >= 0 && state.overspeed_count >= overspeed_threshold) {
            std::fprintf(stderr,
                "[SAFETY] damping_mode ENGAGED: OVER-SPEED joint[%d] |vel|=%.3f > limit %.3f rad/s "
                "(%d consecutive >= thr %d). kp->0 + gravity-comp OFF -> arm will sag. "
                "Recover with reset_errors().\n",
                overspeed_joint, std::abs(overspeed_val), overspeed_lim,
                state.overspeed_count, overspeed_threshold);
        } else if (overcurrent_joint >= 0 && state.overcurrent_count >= overcurrent_threshold) {
            std::fprintf(stderr,
                "[SAFETY] damping_mode ENGAGED: OVER-CURRENT joint[%d] |tau|=%.3f > limit %.3f Nm "
                "(%d consecutive >= thr %d). kp->0 + gravity-comp OFF -> arm will sag. "
                "Recover with reset_errors().\n",
                overcurrent_joint, std::abs(overcurrent_val), overcurrent_lim,
                state.overcurrent_count, overcurrent_threshold);
        } else {
            std::fprintf(stderr, "[SAFETY] damping_mode ENGAGED (trigger ambiguous).\n");
        }
    }

    // In damping mode: zero stiffness and feedforward
    if (state.damping_mode) {
        for (int i = 0; i < ndof; ++i) {
            kp[i] = 0.0;
            tau_ff[i] = 0.0;
            q_des[i] = pos[i];  // hold current position for when we exit damping
        }
    }
}

} // namespace trlc
