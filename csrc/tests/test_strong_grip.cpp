#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "strong_grip.h"

using namespace trlc;

static bool near(double a, double b) { return std::abs(a - b) < 1e-12; }

static void test_off_is_identity() {
    for (double c : {0.0, 0.25, 0.5, 0.885, 1.0}) assert(strong_grip_target(c, false, 1.10) == c);
    std::printf("  off = command unchanged: PASS\n");
}

static void test_uniform_stretch() {
    assert(strong_grip_target(0.0, true, 1.10) == 0.0);
    assert(near(strong_grip_target(0.5, true, 1.10), 0.55));
    assert(near(strong_grip_target(1.0, true, 1.10), 1.10));
    assert(near(strong_grip_target(1.0, true, 1.0), 1.0));  // scale 1 = no stretch
    std::printf("  uniform stretch 0..1 -> 0..1.10: PASS\n");
}

static void test_caps() {
    assert(kStrongGripMaxNm == 3.0);
    assert(kStrongGripMaxCloseTarget == 1.15);
    std::printf("  hard caps 3 Nm / 1.15: PASS\n");
}

int main() {
    std::printf("test_strong_grip\n");
    test_off_is_identity();
    test_uniform_stretch();
    test_caps();
    return 0;
}
