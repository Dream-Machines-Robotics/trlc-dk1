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

#include "fake_bus.h"

using namespace trlc_test;

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
