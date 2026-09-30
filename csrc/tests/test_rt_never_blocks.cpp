// The RT control loop must never block on the motor bus.
//
// Regression for bumblebee 2026-09-30: a USB-CAN adapter stopped taking frames,
// the RT thread sat forever in a blocking ::write(), the arm's feedback froze,
// the comm-loss watchdog (same thread) never fired, and stop() hung in join().
//
// A pty stands in for the adapter: a fake motor bus on the master side answers
// every frame like the DK1's motors, then "wedges" by no longer reading while it
// keeps sending state packets (RX stays alive, TX backs up - the case the RX-only
// watchdog could not see).

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <termios.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "control_loop.h"
#include "dm_protocol.h"
#include "serial_port.h"

using namespace trlc;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Pty {
    int master = -1;
    std::string slave;
    Pty() {
        master = posix_openpt(O_RDWR | O_NOCTTY);
        assert(master >= 0);
        assert(grantpt(master) == 0 && unlockpt(master) == 0);
        slave = ptsname(master);
        fcntl(master, F_SETFL, fcntl(master, F_GETFL) | O_NONBLOCK);
    }
    ~Pty() { if (master >= 0) close(master); }
};

// --- 1. SerialPort::write honours its deadline -------------------------------

static void test_write_returns_within_deadline_when_device_stops_draining() {
    Pty pty;
    SerialPort sp;
    assert(sp.open(pty.slave));
    uint8_t frame[30];
    build_refresh_frame(frame, 1);
    // Nobody reads the master: the pty buffer fills, then every write must fail
    // within its 500 us budget instead of blocking.
    int ok = 0;
    double worst_ms = 0.0;
    bool failed = false;
    for (int i = 0; i < 100000 && !failed; ++i) {
        auto t0 = Clock::now();
        const bool r = sp.write(frame, sizeof(frame), 500);
        worst_ms = std::max(worst_ms, ms_since(t0));
        if (r) ++ok; else failed = true;
    }
    assert(failed);          // the buffer did fill
    assert(ok > 10);         // ...after accepting frames normally first
    assert(worst_ms < 20.0); // bounded (500 us budget + scheduling slack)
    auto t0 = Clock::now();
    assert(!sp.write(frame, sizeof(frame), 500));
    assert(ms_since(t0) < 20.0);
    std::printf("  write deadline (%d frames buffered, worst %.2f ms): PASS\n", ok, worst_ms);
    assert(sp.close_bounded(500) || true);  // must return either way
}

// --- 2. The RT loop latches tx_stalled and stop() returns --------------------

// Fake DK1 motor bus: answers refresh / MIT / EMIT frames with a state packet.
class FakeBus {
public:
    explicit FakeBus(int master) : fd_(master) {
        th_ = std::thread([this] { run(); });
    }
    ~FakeBus() { stop_ = true; th_.join(); }
    void wedge() { wedged_ = true; }

private:
    static void state_packet(uint8_t* p16, uint8_t slave, float q, float tau, float tau_max) {
        std::memset(p16, 0, 16);
        p16[0] = RX_HEADER;
        p16[1] = 0x11;
        const uint32_t can_id = 0x10u + slave;  // == the descriptor's master_id
        p16[3] = can_id & 0xFF;
        uint8_t* d = p16 + 7;
        d[0] = slave;
        const uint16_t qu = float_to_uint(q, -12.5f, 12.5f, 16);
        const uint16_t dqu = float_to_uint(0.0f, -30.0f, 30.0f, 12);
        const uint16_t tu = float_to_uint(tau, -tau_max, tau_max, 12);
        d[1] = qu >> 8;
        d[2] = qu & 0xFF;
        d[3] = dqu >> 4;
        d[4] = static_cast<uint8_t>(((dqu & 0xF) << 4) | ((tu >> 8) & 0xF));
        d[5] = tu & 0xFF;
        p16[15] = RX_TAIL;
    }

    void reply(uint8_t slave) {
        uint8_t pkt[16];
        // The gripper (slave 7) reports a torque spike so calibration ends at once.
        state_packet(pkt, slave, 0.1f, slave == 7 ? 2.0f : 0.0f, 10.0f);
        (void)!::write(fd_, pkt, sizeof(pkt));
    }

    void run() {
        std::vector<uint8_t> buf;
        uint8_t tmp[4096];
        while (!stop_) {
            if (wedged_) {
                // Stop draining the host's frames but keep talking: RX stays alive,
                // so only the TX watchdog can notice.
                for (uint8_t s = 1; s <= 7; ++s) reply(s);
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
                continue;
            }
            ssize_t n = ::read(fd_, tmp, sizeof(tmp));
            if (n > 0) buf.insert(buf.end(), tmp, tmp + n);
            size_t i = 0;
            while (i + 30 <= buf.size()) {
                if (buf[i] != 0x55 || buf[i + 1] != 0xAA) { ++i; continue; }
                const uint16_t id = buf[i + 13] | (buf[i + 14] << 8);
                const uint8_t* data = &buf[i + 21];
                if (id == 0x7FF) {
                    if (data[2] == 0xCC) reply(data[0]);  // refresh
                    // parameter writes (mode switch) get no state reply
                } else if (id >= 0x300) {
                    reply(static_cast<uint8_t>(id - 0x300));  // EMIT (gripper)
                } else if (id >= 0x200) {
                    reply(static_cast<uint8_t>(id - 0x200));  // VEL (gripper cal)
                } else if (id >= 1 && id <= 7) {
                    reply(static_cast<uint8_t>(id));          // MIT / enable / disable
                }
                i += 30;
            }
            buf.erase(buf.begin(), buf.begin() + static_cast<long>(i));
            if (n <= 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    int fd_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> wedged_{false};
    std::thread th_;
};

static RtLoopConfig loop_config(const std::string& port) {
    RtLoopConfig cfg;
    cfg.serial_port = port;
    cfg.label = "left";
    cfg.rt_use_mlockall = false;
    cfg.rt_priority = 0;
    cfg.gripper_cal_timeout_s = 2.0;
    const MotorType types[7] = {MotorType::DM4340, MotorType::DM4340, MotorType::DM4340,
                                MotorType::DM4310, MotorType::DM4310, MotorType::DM4310,
                                MotorType::DM4310};
    for (int i = 0; i < 7; ++i) {
        MotorDescriptor m;
        m.name = i < 6 ? "joint_" + std::to_string(i + 1) : "gripper";
        m.type = types[i];
        m.slave_id = static_cast<uint16_t>(i + 1);
        m.master_id = static_cast<uint16_t>(0x11 + i);
        cfg.motors.push_back(m);
    }
    return cfg;
}

static void test_loop_latches_tx_stall_and_stop_returns() {
    Pty pty;
    FakeBus bus(pty.master);
    RtControlLoop loop(loop_config(pty.slave));
    loop.start();

    // Healthy while the bus drains: past warmup, no comm loss, feedback flowing.
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    HealthState h = loop.get_health();
    assert(h.loop_count > 60);
    assert(!h.comm_loss && !h.tx_stalled);
    const uint64_t before = h.loop_count;

    bus.wedge();
    // The loop must keep cycling (never block) and latch comm loss via TX.
    auto t0 = Clock::now();
    while (!loop.get_health().tx_stalled && ms_since(t0) < 5000.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    h = loop.get_health();
    assert(h.tx_stalled);
    assert(h.comm_loss);
    assert(h.loop_count > before + 50);  // still cycling
    // And it keeps cycling afterwards (DISABLE mode sends nothing, reads go on).
    const uint64_t at_latch = h.loop_count;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(loop.get_health().loop_count > at_latch + 10);
    std::printf("  tx stall latched after %.0f ms of wedge: PASS\n", ms_since(t0));

    // Teardown never hangs, even with the release frames going to a full buffer.
    loop.set_disable_torque_on_disconnect(true);
    t0 = Clock::now();
    const bool clean = loop.stop();
    const double stop_ms = ms_since(t0);
    assert(clean);
    assert(stop_ms < 1500.0);
    std::printf("  stop() on a wedged bus returned in %.0f ms: PASS\n", stop_ms);
}

static void test_reset_clears_tx_stall_once_bus_drains() {
    // reset_errors() must clear the TX latch like the RX one (the operator's
    // recovery path after reseating a connector).
    Pty pty;
    auto bus = std::make_unique<FakeBus>(pty.master);
    RtControlLoop loop(loop_config(pty.slave));
    loop.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    bus->wedge();
    auto t0 = Clock::now();
    while (!loop.get_health().tx_stalled && ms_since(t0) < 5000.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(loop.get_health().tx_stalled);
    // "Reseat": a fresh bus that drains again (drop what the old one left queued).
    bus.reset();
    tcflush(pty.master, TCIOFLUSH);
    bus = std::make_unique<FakeBus>(pty.master);
    loop.reset_errors(500);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    HealthState h = loop.get_health();
    assert(!h.tx_stalled && !h.comm_loss);
    assert(loop.stop());
    std::printf("  reset_errors clears tx stall: PASS\n");
}

int main() {
    std::printf("test_rt_never_blocks\n");
    test_write_returns_within_deadline_when_device_stops_draining();
    test_loop_latches_tx_stall_and_stop_returns();
    test_reset_clears_tx_stall_once_bus_drains();
    std::printf("all passed\n");
    return 0;
}
