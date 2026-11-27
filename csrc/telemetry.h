#pragma once

// Per-cycle telemetry of the RT loop: one slot per control cycle (250 Hz) with what
// the arm measured, what it was commanded and what was really sent, for detailed
// session logs (RtLoopConfig::telemetry, off by default). The RT thread is the
// single writer; one reader drains it in bulk (read()). Nothing here touches the
// motor bus: every value is one the loop already has.
//
// Off (the default), no ring exists and the loop pays one null check per cycle.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace trlc {

// TelemetrySlot::flags bits.
enum TelemetryFlag : uint8_t {
    TF_DAMPING = 1u << 0,      // safety damping mode (kp 0, no gravity comp)
    TF_COMM_LOSS = 1u << 1,    // comm loss latched (RX or TX watchdog)
    TF_TX_STALLED = 1u << 2,   // the TX watchdog latched it
    TF_TX_FAILED = 1u << 3,    // a frame write failed in this cycle
    TF_CMD_TORN = 1u << 4,     // the command read stayed torn; the last good one was used
    TF_CMD_STALE = 1u << 5,    // command older than command_timeout_s (holding)
    TF_ACCEL_GUARD = 1u << 6,  // the acceleration guard was on
};

// One control cycle. Joints are 0..5, the gripper is index 6 where 7 values exist.
// Units: rad, rad/s, Nm, us, ns (CLOCK_MONOTONIC), degrees C.
struct TelemetrySlot {
    uint64_t t_ns;          // cycle start
    uint64_t loop_count;
    uint64_t cmd_ts_ns;     // when the command this cycle used was written
    float cycle_us;         // work time of this cycle
    float wake_late_us;     // cycle start minus its scheduled start
    float pos[7];           // measured position (gripper: motor rad)
    float vel[7];           // measured velocity
    float torque[7];        // measured torque (the motor's estimate from current)
    float q_cmd[6];         // the command as written by Python
    float q_slew[6];        // after the slew limit (and acceleration guard)
    float q_sent[6];        // after the safety clamp: what the MIT frame carried
    float kp_sent[6];       // kp actually sent (0 in damping mode)
    float tau_ff[6];        // gravity feed-forward actually sent
    float gripper_cmd;      // normalized command (0 open, 1 closed, up to 1.10)
    float gripper_q_sent;   // motor position sent to the gripper (NaN: none sent)
    float gripper_pos_norm; // measured gripper position, normalized like get_gripper_state()
    uint32_t missed_periods;   // cumulative: cycles that overran and reset the schedule
    uint32_t rx_bytes;         // bytes read from the bus in this cycle
    uint16_t overcurrent_count;
    uint16_t overspeed_count;
    uint16_t empty_cycles;     // consecutive cycles with nothing received
    uint8_t t_mos[7];          // driver (MOS) temperature, from the reply frame's byte 6
    uint8_t t_rotor[7];        // rotor temperature, from the reply frame's byte 7
    uint8_t err[7];            // Damiao state code, the high 4 bits of the reply's byte 0
    uint8_t responded;         // bit i: motor i replied in this cycle
    uint8_t stale;             // bit i: motor i silent for per_motor_stale_threshold cycles
    uint8_t flags;             // TelemetryFlag bits
};

// Layout description for readers in other languages (Python builds a numpy dtype
// from it, so the layout is never written down twice).
struct TelemetryField {
    const char* name;
    size_t offset;
    char kind;       // 'u' unsigned int, 'f' float
    size_t itemsize; // bytes per element
    size_t count;    // elements (1 = scalar)
};

#define TRLC_TF(field, kind, n) \
    TelemetryField{#field, offsetof(TelemetrySlot, field), kind, sizeof(TelemetrySlot::field) / (n), n}

inline std::vector<TelemetryField> telemetry_fields() {
    return {
        TRLC_TF(t_ns, 'u', 1), TRLC_TF(loop_count, 'u', 1), TRLC_TF(cmd_ts_ns, 'u', 1),
        TRLC_TF(cycle_us, 'f', 1), TRLC_TF(wake_late_us, 'f', 1),
        TRLC_TF(pos, 'f', 7), TRLC_TF(vel, 'f', 7), TRLC_TF(torque, 'f', 7),
        TRLC_TF(q_cmd, 'f', 6), TRLC_TF(q_slew, 'f', 6), TRLC_TF(q_sent, 'f', 6),
        TRLC_TF(kp_sent, 'f', 6), TRLC_TF(tau_ff, 'f', 6),
        TRLC_TF(gripper_cmd, 'f', 1), TRLC_TF(gripper_q_sent, 'f', 1), TRLC_TF(gripper_pos_norm, 'f', 1),
        TRLC_TF(missed_periods, 'u', 1), TRLC_TF(rx_bytes, 'u', 1),
        TRLC_TF(overcurrent_count, 'u', 1), TRLC_TF(overspeed_count, 'u', 1), TRLC_TF(empty_cycles, 'u', 1),
        TRLC_TF(t_mos, 'u', 7), TRLC_TF(t_rotor, 'u', 7), TRLC_TF(err, 'u', 7),
        TRLC_TF(responded, 'u', 1), TRLC_TF(stale, 'u', 1), TRLC_TF(flags, 'u', 1),
    };
}

#undef TRLC_TF

// Single-producer ring of TelemetrySlot. The writer fills next_slot() and then
// publish()es it; it never waits. A reader that falls behind loses the oldest
// slots and is told how many (read()'s `dropped`).
class TelemetryRing {
public:
    explicit TelemetryRing(size_t capacity)
        : n_(std::max<size_t>(capacity, 16)), slots_(new TelemetrySlot[n_]) {
        std::memset(slots_.get(), 0, n_ * sizeof(TelemetrySlot));
    }

    size_t capacity() const { return n_; }

    // Writer (RT thread) only.
    TelemetrySlot& next_slot() { return slots_[w_ % n_]; }
    void publish() { idx_.store(++w_, std::memory_order_release); }

    // Slots published so far (the index the next one will get).
    uint64_t write_index() const { return idx_.load(std::memory_order_acquire); }

    // Copy up to `max` slots starting at index `from` into `out`. Returns the number
    // copied; `next` is the index to ask for next time, `dropped` how many slots
    // from `from` on were already overwritten (lost) before they could be copied.
    size_t read(uint64_t from, TelemetrySlot* out, size_t max, uint64_t& next, uint64_t& dropped) const {
        const uint64_t w = idx_.load(std::memory_order_acquire);
        if (from > w) from = w;
        // Keep clear of the slots the writer may reach while this copy runs.
        const uint64_t keep = n_ - n_ / 8;
        uint64_t start = std::max<uint64_t>(from, w > keep ? w - keep : 0);
        size_t count = static_cast<size_t>(std::min<uint64_t>(w - start, max));
        for (size_t i = 0; i < count; ++i) out[i] = slots_[(start + i) % n_];
        std::atomic_thread_fence(std::memory_order_acquire);
        // Any index below w2 + 1 - n may have been (partly) overwritten meanwhile.
        const uint64_t w2 = idx_.load(std::memory_order_acquire);
        const uint64_t first_ok = (w2 + 1 > n_) ? w2 + 1 - n_ : 0;
        if (first_ok > start) {
            const size_t skip = static_cast<size_t>(std::min<uint64_t>(first_ok - start, count));
            std::memmove(out, out + skip, (count - skip) * sizeof(TelemetrySlot));
            count -= skip;
            start += skip;
        }
        dropped = start - from;
        next = start + count;
        return count;
    }

private:
    size_t n_;
    std::unique_ptr<TelemetrySlot[]> slots_;
    uint64_t w_ = 0;  // writer-local count
    alignas(64) std::atomic<uint64_t> idx_{0};
};

} // namespace trlc
