#include <cassert>
#include <cstdio>

#include "strong_grip.h"

using namespace trlc;

static void test_cmd_clamp() {
    assert(strong_grip_cmd_max(false) == 1.0);  // off: today's clamp
    assert(strong_grip_cmd_max(true) == 1.15);  // on: stretched teleop commands pass
    std::printf("  command clamp 1.0 off / 1.15 on: PASS\n");
}

static void test_caps() {
    assert(kStrongGripMaxNm == 3.0);
    assert(kStrongGripMaxCloseTarget == 1.15);
    std::printf("  hard caps 3 Nm / 1.15: PASS\n");
}

int main() {
    std::printf("test_strong_grip\n");
    test_cmd_clamp();
    test_caps();
    return 0;
}
