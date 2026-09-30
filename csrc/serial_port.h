#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace trlc {

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort();

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    bool open(const std::string& device, int baudrate = 921600);
    void close();

    // close() with a deadline. Closing a tty with un-drained output waits for it
    // (cdc-acm: up to the port's 30 s closing_wait) - exactly what a wedged
    // adapter produces. Discards pending output first, then closes on a helper
    // thread; past timeout_ms the helper is detached and finishes on its own.
    // Returns true when the close completed in time.
    bool close_bounded(int timeout_ms);

    // Write exactly n bytes within timeout_us. Returns true on success, false on
    // error or when the device stops accepting data before the deadline (a
    // USB-CAN adapter that no longer drains: CAN bus-off, no ACK from a loose
    // connector). NEVER blocks longer than timeout_us: the fd is non-blocking and
    // the wait is a poll(). A timed-out write may leave a partial frame queued;
    // the adapter resyncs on the next 0x55 0xAA header.
    bool write(const uint8_t* buf, size_t n, int timeout_us = kDefaultWriteTimeoutUs);
    static constexpr int kDefaultWriteTimeoutUs = 100000;  // setup/teardown path

    // Non-blocking read up to max bytes. Returns number of bytes read.
    size_t read_all(uint8_t* buf, size_t max);

    // Blocking read with timeout (uses select). Returns number of bytes read.
    // Waits up to timeout_us microseconds for data to become available,
    // then reads all available bytes.
    size_t read_with_timeout(uint8_t* buf, size_t max, int timeout_us);

    int fd() const { return fd_; }
    bool is_open() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

} // namespace trlc
