#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "strong_grip.h"

using namespace trlc;

static bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

static void test_off_is_base_everywhere() {
    StrongGrip g(1.0, 0.0, 0.9, 12.5);
    for (double c : {-0.5, 0.0, 0.5, 0.95, 1.0, 2.0}) assert(near(g.limit(c), 1.0));
    assert(!g.engaged());
    std::printf("  off = base limit: PASS\n");
}

static void test_open_stays_base() {
    StrongGrip g(1.0, 2.5, 0.9, 12.5);
    for (int i = 0; i < 100; ++i) assert(near(g.limit(0.5), 1.0));
    std::printf("  wider than threshold stays base: PASS\n");
}

static void test_ramp_up_and_down() {
    StrongGrip g(1.0, 2.5, 0.9, 10.0);  // 0.15 Nm per cycle
    double v = 0.0;
    for (int i = 1; i <= 10; ++i) {
        v = g.limit(0.95);
        assert(near(v, std::min(1.0 + 0.15 * i, 2.5)));
    }
    assert(near(g.limit(0.95), 2.5));
    // Inside the hysteresis band: stays strong.
    assert(near(g.limit(0.89), 2.5));
    // Opened clearly: ramps back down to base.
    for (int i = 1; i <= 10; ++i) v = g.limit(0.5);
    assert(near(v, 1.0));
    assert(!g.engaged());
    std::printf("  ramp up, hysteresis, ramp down: PASS\n");
}

static void test_cap_constant() {
    assert(kStrongGripMaxNm == 3.0);
    assert(kStrongGripMaxCloseTarget == 1.15);
    std::printf("  hard caps 3 Nm / 1.15: PASS\n");
}

static void test_target_off_is_identity() {
    StrongGrip off(1.0, 0.0, 0.85, 12.5, 1.10);  // strong_nm 0 = off
    for (double c : {0.0, 0.5, 0.85, 0.95, 1.0}) assert(off.target(c) == c);
    StrongGrip no_overshoot(1.0, 2.5, 0.85, 12.5);  // default close_target 1.0
    for (double c : {0.0, 0.85, 0.9, 1.0}) assert(near(no_overshoot.target(c), c));
    std::printf("  target identity when off / no overshoot: PASS\n");
}

static void test_same_limit_still_remaps() {
    // Default strong torque = base (1.0 Nm): active, limit unchanged, overshoot on.
    StrongGrip g(1.0, 1.0, 0.85, 12.5, 1.10);
    assert(g.active());
    for (double c : {0.0, 0.9, 1.0, 0.5}) assert(near(g.limit(c), 1.0));
    assert(near(g.target(1.0), 1.10));
    std::printf("  strong limit = base: limit unchanged, overshoot on: PASS\n");
}

static void test_target_remap() {
    StrongGrip g(1.0, 2.5, 0.85, 12.5, 1.10);
    assert(g.target(0.0) == 0.0);
    assert(g.target(0.84) == 0.84);              // below threshold: unchanged
    assert(near(g.target(0.85), 0.85));          // continuous at the threshold
    assert(near(g.target(1.0), 1.10));           // full close -> overshoot target
    assert(near(g.target(0.925), 0.975));        // halfway maps halfway
    double prev = 0.0;                            // monotonic, no jump
    for (int i = 0; i <= 1000; ++i) {
        double c = i / 1000.0, t = g.target(c);
        assert(t >= prev - 1e-12 && t - prev < 0.002);
        prev = t;
    }
    std::printf("  target remap [0.85,1] -> [0.85,1.10]: PASS\n");
}

int main() {
    std::printf("test_strong_grip\n");
    test_off_is_base_everywhere();
    test_open_stays_base();
    test_ramp_up_and_down();
    test_cap_constant();
    test_target_off_is_identity();
    test_target_remap();
    test_same_limit_still_remaps();
    return 0;
}
