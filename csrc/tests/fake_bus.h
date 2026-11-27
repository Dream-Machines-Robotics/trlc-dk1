// Test helpers shared by the RT loop tests: a pty standing in for the USB-CAN
// adapter, a fake DK1 motor bus on its master side, and a loop config for it.
#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "control_loop.h"
#include "dm_protocol.h"

namespace trlc_test {

using namespace trlc;

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


// Fake DK1 motor bus: answers refresh / MIT / EMIT frames with a state packet.
class FakeBus {
public:
    explicit FakeBus(int master) : fd_(master) {
        th_ = std::thread([this] { run(); });
    }
    ~FakeBus() { stop_ = true; th_.join(); }
    void wedge() { wedged_ = true; }
    // What every later reply carries in its status bytes: the state code (high 4
    // bits of byte 0) and the temperatures (bytes 6 and 7), each plus the slave id
    // so tests can tell the motors apart. Default 0: like before.
    void set_status(uint8_t err, uint8_t t_mos_base, uint8_t t_rotor_base) {
        err_ = err;
        t_mos_ = t_mos_base;
        t_rotor_ = t_rotor_base;
    }

private:
    void state_packet(uint8_t* p16, uint8_t slave, float q, float tau, float tau_max) const {
        std::memset(p16, 0, 16);
        p16[0] = RX_HEADER;
        p16[1] = 0x11;
        const uint32_t can_id = 0x10u + slave;  // == the descriptor's master_id
        p16[3] = can_id & 0xFF;
        uint8_t* d = p16 + 7;
        d[0] = static_cast<uint8_t>(slave | (err_.load() << 4));
        const uint16_t qu = float_to_uint(q, -12.5f, 12.5f, 16);
        const uint16_t dqu = float_to_uint(0.0f, -30.0f, 30.0f, 12);
        const uint16_t tu = float_to_uint(tau, -tau_max, tau_max, 12);
        d[1] = qu >> 8;
        d[2] = qu & 0xFF;
        d[3] = dqu >> 4;
        d[4] = static_cast<uint8_t>(((dqu & 0xF) << 4) | ((tu >> 8) & 0xF));
        d[5] = tu & 0xFF;
        if (t_mos_.load() || t_rotor_.load()) {
            d[6] = static_cast<uint8_t>(t_mos_.load() + slave);
            d[7] = static_cast<uint8_t>(t_rotor_.load() + slave);
        }
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
    std::atomic<uint8_t> err_{0};
    std::atomic<uint8_t> t_mos_{0};
    std::atomic<uint8_t> t_rotor_{0};
    std::thread th_;
};

inline RtLoopConfig loop_config(const std::string& port) {
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


} // namespace trlc_test
