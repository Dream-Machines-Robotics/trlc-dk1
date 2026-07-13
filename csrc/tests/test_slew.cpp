#include <cassert>
#include <cmath>
#include <cstdio>

#include "slew.h"

using namespace trlc;

static constexpr double MAX_DELTA = 0.02;   // rad/cycle (5 rad/s at 250 Hz)
static constexpr double ACCEL = 0.001;      // rad/cycle²

static void test_guard_off_matches_legacy_clamp() {
    // max_accel = 0 must reproduce the original slew limiter exactly,
    // regardless of the previous step.
    assert(slew_step(0.005, 0.0, MAX_DELTA, 0.0) == 0.005);
    assert(slew_step(1.0, 0.0, MAX_DELTA, 0.0) == MAX_DELTA);
    assert(slew_step(-1.0, 0.02, MAX_DELTA, 0.0) == -MAX_DELTA);
    assert(slew_step(-0.003, -0.02, MAX_DELTA, 0.0) == -0.003);
    std::printf("  guard off = legacy clamp: PASS\n");
}

static void test_ramp_from_rest() {
    // A large jump from rest ramps at ACCEL per cycle until the velocity cap.
    double prev = 0.0;
    for (int cycle = 1; cycle <= 40; ++cycle) {
        double step = slew_step(10.0, prev, MAX_DELTA, ACCEL);
        double expected = std::min(cycle * ACCEL, MAX_DELTA);
        assert(std::abs(step - expected) < 1e-12);
        prev = step;
    }
    assert(prev == MAX_DELTA);  // saturated at the velocity cap
    std::printf("  ramp from rest: PASS\n");
}

static void test_converges_without_overshoot() {
    // Integrate a full approach to a step target: the ramp must land exactly
    // on the target and never pass it (shrinking steps are unrestricted).
    const double target = 0.5;
    double pos = 0.0, prev = 0.0;
    int cycles = 0;
    while (std::abs(target - pos) > 1e-12) {
        double step = slew_step(target - pos, prev, MAX_DELTA, ACCEL);
        pos += step;
        prev = step;
        assert(pos <= target + 1e-12);
        assert(++cycles < 10000);
    }
    // And it must stay parked once there.
    assert(std::abs(slew_step(target - pos, prev, MAX_DELTA, ACCEL)) <= 1e-12);
    std::printf("  converges without overshoot (%d cycles): PASS\n", cycles);
}

static void test_shrinking_step_is_free() {
    // Decelerating toward the target is never held back by the guard.
    assert(slew_step(0.002, 0.02, MAX_DELTA, ACCEL) == 0.002);
    assert(slew_step(-0.002, -0.02, MAX_DELTA, ACCEL) == -0.002);
    std::printf("  shrinking step free: PASS\n");
}

static void test_reversal_restarts_from_zero() {
    // Direction reversal: the step restarts at ±ACCEL, not at the full cap.
    assert(slew_step(-1.0, 0.02, MAX_DELTA, ACCEL) == -ACCEL);
    assert(slew_step(1.0, -0.02, MAX_DELTA, ACCEL) == ACCEL);
    std::printf("  reversal restarts from zero: PASS\n");
}

static void test_never_faster_than_unguarded() {
    // The guarded step magnitude never exceeds the legacy (guard-off) step.
    const double diffs[] = {-1.0, -0.02, -0.001, 0.0, 0.001, 0.02, 1.0};
    const double prevs[] = {-0.02, -0.005, 0.0, 0.005, 0.02};
    for (double diff : diffs) {
        for (double prev : prevs) {
            double guarded = slew_step(diff, prev, MAX_DELTA, ACCEL);
            double legacy = slew_step(diff, prev, MAX_DELTA, 0.0);
            assert(std::abs(guarded) <= std::abs(legacy) + 1e-15);
        }
    }
    std::printf("  never faster than unguarded: PASS\n");
}

int main() {
    std::printf("test_slew:\n");
    test_guard_off_matches_legacy_clamp();
    test_ramp_from_rest();
    test_converges_without_overshoot();
    test_shrinking_step_is_free();
    test_reversal_restarts_from_zero();
    test_never_faster_than_unguarded();
    std::printf("ALL PASS\n");
    return 0;
}
