// Per-cycle telemetry (telemetry.h): the status-byte decode, the ring's index and
// loss accounting, and the RT loop end to end against the fake motor bus.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "fake_bus.h"
#include "telemetry.h"

using namespace trlc;
using namespace trlc_test;

static void test_decode_motor_status() {
    uint8_t pkt[16] = {};
    pkt[0] = RX_HEADER;
    pkt[1] = 0x11;
    uint8_t* d = pkt + 7;
    d[0] = static_cast<uint8_t>(0x05 | (0xA << 4));  // motor 5, over-current
    d[6] = 41;                                       // driver 41 C
    d[7] = 37;                                       // rotor 37 C
    MotorStatusBytes s = decode_motor_status(pkt);
    assert(s.err == 0xA && s.t_mos == 41 && s.t_rotor == 37);
    assert(decode_response_id(pkt) == 5);  // the id nibble is unaffected
    d[0] = 0x13;                           // motor 3, enabled
    assert(decode_motor_status(pkt).err == 1);
    std::printf("  decode_motor_status: PASS\n");
}

static void write_n(TelemetryRing& ring, uint64_t& counter, int n) {
    for (int i = 0; i < n; ++i) {
        TelemetrySlot& s = ring.next_slot();
        s.loop_count = counter++;
        ring.publish();
    }
}

static void test_ring_reads_in_order_and_counts_losses() {
    TelemetryRing ring(64);
    std::vector<TelemetrySlot> out(256);
    uint64_t next = 0, dropped = 0, counter = 0;

    assert(ring.read(0, out.data(), out.size(), next, dropped) == 0);
    assert(next == 0 && dropped == 0);

    write_n(ring, counter, 10);
    size_t n = ring.read(0, out.data(), out.size(), next, dropped);
    assert(n == 10 && next == 10 && dropped == 0);
    for (size_t i = 0; i < n; ++i) assert(out[i].loop_count == i);

    // A small max: the rest comes next time.
    write_n(ring, counter, 20);
    n = ring.read(next, out.data(), 5, next, dropped);
    assert(n == 5 && next == 15 && out[0].loop_count == 10);
    n = ring.read(next, out.data(), out.size(), next, dropped);
    assert(n == 15 && next == 30 && dropped == 0 && out[14].loop_count == 29);

    // The writer laps the reader: the oldest slots are lost and counted, and what
    // comes back is contiguous and ends at the newest slot.
    write_n(ring, counter, 200);  // index 230 now
    n = ring.read(next, out.data(), out.size(), next, dropped);
    assert(next == 230);
    assert(dropped > 0 && dropped + n == 200);
    assert(n <= 64 && n >= 64 - 64 / 8 - 1);
    for (size_t i = 1; i < n; ++i) assert(out[i].loop_count == out[i - 1].loop_count + 1);
    assert(out[n - 1].loop_count == 229);
    std::printf("  ring order and losses (%llu lost): PASS\n", static_cast<unsigned long long>(dropped));
}

static void test_slot_layout_table() {
    const auto fields = telemetry_fields();
    size_t covered = 0;
    for (const auto& f : fields) {
        assert(f.offset + f.itemsize * f.count <= sizeof(TelemetrySlot));
        covered += f.itemsize * f.count;
    }
    assert(covered <= sizeof(TelemetrySlot) && sizeof(TelemetrySlot) - covered < 8);  // padding only
    assert(sizeof(TelemetrySlot) == 288);
    std::printf("  slot layout (%zu fields, %zu bytes): PASS\n", fields.size(), sizeof(TelemetrySlot));
}

static void test_loop_without_telemetry_has_no_ring() {
    Pty pty;
    FakeBus bus(pty.master);
    RtControlLoop loop(loop_config(pty.slave));
    loop.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    assert(!loop.telemetry_enabled() && loop.telemetry_capacity() == 0);
    TelemetrySlot slot;
    uint64_t next = 7, dropped = 9;
    assert(loop.read_telemetry(7, &slot, 1, next, dropped) == 0);
    assert(next == 7 && dropped == 0);
    assert(loop.stop());
    std::printf("  telemetry off: no ring: PASS\n");
}

static void test_loop_fills_one_slot_per_cycle() {
    Pty pty;
    FakeBus bus(pty.master);
    RtLoopConfig cfg = loop_config(pty.slave);
    cfg.telemetry = true;
    cfg.telemetry_ring_size = 2048;
    RtControlLoop loop(cfg);
    loop.start();
    assert(loop.telemetry_enabled() && loop.telemetry_capacity() == 2048);
    bus.set_status(1, 30, 50);  // enabled; driver 30+id C, rotor 50+id C

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const double q6[6] = {0.10, 0.11, 0.12, 0.13, 0.14, 0.15};
    loop.command_joint_pos(q6);
    loop.command_gripper(0.5);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::vector<TelemetrySlot> out(4096);
    uint64_t next = 0, dropped = 0;
    size_t n = loop.read_telemetry(0, out.data(), out.size(), next, dropped);
    assert(loop.stop());
    assert(dropped == 0);
    assert(n > 60);  // about 125 at 250 Hz in 0.5 s, less the warmup and a slow CI machine
    assert(next == n);
    for (size_t i = 1; i < n; ++i) {
        assert(out[i].loop_count == out[i - 1].loop_count + 1);  // every cycle, none skipped
        assert(out[i].t_ns > out[i - 1].t_ns);
    }
    const TelemetrySlot& last = out[n - 1];
    for (int j = 0; j < 6; ++j) {
        assert(std::fabs(last.q_cmd[j] - q6[j]) < 1e-6);
        assert(std::isfinite(last.q_slew[j]) && std::isfinite(last.q_sent[j]));
        assert(last.kp_sent[j] == static_cast<float>(cfg.default_kp[static_cast<size_t>(j)]));
    }
    assert(std::fabs(last.gripper_cmd - 0.5f) < 1e-6);
    assert(std::isfinite(last.gripper_q_sent));
    assert(last.cmd_ts_ns > 0 && last.cmd_ts_ns <= last.t_ns + 10'000'000ULL);
    assert(last.responded == 0x7F);
    assert(last.stale == 0);
    assert((last.flags & (TF_DAMPING | TF_COMM_LOSS | TF_TX_STALLED)) == 0);
    assert(last.flags & TF_ACCEL_GUARD);
    for (int i = 0; i < 7; ++i) {
        assert(last.err[i] == 1);
        assert(last.t_mos[i] == 30 + i + 1);
        assert(last.t_rotor[i] == 50 + i + 1);
        assert(std::fabs(last.pos[i] - 0.1f) < 1e-3);
    }
    assert(last.cycle_us > 0.0f);
    std::printf("  RT loop fills a slot per cycle (%zu slots): PASS\n", n);
}

int main() {
    std::printf("test_telemetry\n");
    test_decode_motor_status();
    test_ring_reads_in_order_and_counts_losses();
    test_slot_layout_table();
    test_loop_without_telemetry_has_no_ring();
    test_loop_fills_one_slot_per_cycle();
    std::printf("all passed\n");
    return 0;
}
