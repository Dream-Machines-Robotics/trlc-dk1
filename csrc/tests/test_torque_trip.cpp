// The over-torque trip (csrc/safety.h) on joints 1-3, as set by the dev toggle
// stiff-arm-asymmetric (lerobot_robot_trlc_dk1/stiff_arm.py).
//
// The DM4340 reports torque on a 12-bit scale over -28..+28 Nm, so its highest
// reading is exactly 28.0. With the normal limit of 28 Nm a reading can never be
// above it and the trip never fires; the toggle passes 27 Nm, which a held
// full-scale reading exceeds. No C++ logic changes: only the limit differs.

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "control_loop.h"
#include "dm_protocol.h"
#include "safety.h"

#include "fake_bus.h"

using namespace trlc;
using namespace trlc_test;

static constexpr std::array<double, 6> kNormal = {28, 28, 28, 10, 10, 10};
static constexpr std::array<double, 6> kStiffArm = {27, 27, 27, 10, 10, 10};

// Runs apply_safety on a held reading; returns the first cycle (1-based) that
// latched damping mode, or 0 if none did.
static int cycles_to_trip(const std::array<double, 6>& limits, double reading, int cycles) {
    SafetyState st;
    const double pos[6] = {0, 1, 1, 0, 0, 0}, vel[6] = {};
    const double torque[6] = {0, reading, 0, 0, 0, 0};
    const double lo[6] = {-3, -3, -3, -3, -3, -3}, hi[6] = {3, 3, 3, 3, 3, 3};
    const double vlim[6] = {5, 5, 5, 15, 15, 15};
    for (int c = 1; c <= cycles; ++c) {
        double q_des[6] = {0, 1, 1, 0, 0, 0}, kp[6] = {100, 200, 200, 20, 20, 10};
        double tau_ff[6] = {0, 12.8, 6.1, 1.3, 0.5, 0};  // worst-case gravity load (follower model)
        apply_safety(pos, vel, torque, q_des, kp, tau_ff, lo, hi, limits.data(), vlim, 0.05,
                     20, 5, 6, st);
        if (c == 1) {
            // The same limit clips the gravity feed-forward: the worst case passes through.
            assert(tau_ff[1] == 12.8 && tau_ff[2] == 6.1);
        }
        if (st.damping_mode) {
            assert(kp[1] == 0.0 && tau_ff[1] == 0.0);  // damping: kp and feed-forward zeroed
            return c;
        }
    }
    return 0;
}

static void test_full_scale_reading_is_exactly_28() {
    const auto& lim = get_limits(MotorType::DM4340);
    assert(lim.tau_max == 28.0f);
    assert(uint_to_float(4095, -lim.tau_max, lim.tau_max, 12) == 28.0f);
    assert(uint_to_float(float_to_uint(40.0f, -lim.tau_max, lim.tau_max, 12),
                         -lim.tau_max, lim.tau_max, 12) == 28.0f);  // anything higher reads 28
    std::printf("  DM4340 highest torque reading is 28.0 Nm: PASS\n");
}

static void test_safety_trips_after_20_cycles_at_27_not_at_28() {
    const double full = uint_to_float(4095, -28.0f, 28.0f, 12);
    assert(cycles_to_trip(kStiffArm, full, 1000) == 20);  // overcurrent_threshold
    assert(cycles_to_trip(kNormal, full, 1000) == 0);     // today: never
    assert(cycles_to_trip(kStiffArm, 26.9, 1000) == 0);   // just below: never
    std::printf("  held 28.0 Nm trips at cycle 20 with 27 Nm, never with 28 Nm: PASS\n");
}

// The RT loop end to end against the fake bus reporting full-scale torque on joints 1-3.
static bool loop_trips(const std::array<double, 6>& limits) {
    Pty pty;
    FakeBus bus(pty.master);
    RtLoopConfig cfg = loop_config(pty.slave);
    cfg.joint_torque_limits = limits;
    RtControlLoop loop(cfg);
    loop.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    assert(!loop.get_health().damping_mode);  // healthy at 0 Nm
    bus.set_arm_torque(28.0f);
    bool tripped = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(800)) {
        if (loop.get_health().damping_mode) { tripped = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    HealthState h = loop.get_health();
    assert(!h.comm_loss);
    if (!tripped) assert(h.overcurrent_count == 0);  // not even counting
    bus.set_arm_torque(0.0f);
    loop.stop();
    return tripped;
}

static void test_loop_trips_with_27_and_not_with_28() {
    assert(loop_trips(kStiffArm));
    std::printf("  RT loop on the fake bus: 28.0 Nm reading trips damping mode with 27 Nm: PASS\n");
    assert(!loop_trips(kNormal));
    std::printf("  RT loop on the fake bus: no trip with the normal 28 Nm: PASS\n");
}

int main() {
    std::printf("test_torque_trip\n");
    test_full_scale_reading_is_exactly_28();
    test_safety_trips_after_20_cycles_at_27_not_at_28();
    test_loop_trips_with_27_and_not_with_28();
    std::printf("all passed\n");
    return 0;
}
