#include "serial_port.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sys/select.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

namespace trlc {

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baudrate) {
    if (fd_ >= 0) close();

    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::fprintf(stderr, "SerialPort: cannot open %s: %s\n",
                     device.c_str(), std::strerror(errno));
        return false;
    }

    // O_NONBLOCK stays set: reads are non-blocking anyway (VMIN=VTIME=0) and
    // writes must never block the RT thread - write() waits with poll() and a
    // deadline instead. A blocking ::write() here once froze an arm's control
    // thread for good when its adapter stopped draining (bumblebee 2026-09-30).

    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        std::fprintf(stderr, "SerialPort: tcgetattr failed: %s\n", std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Map baudrate
    speed_t baud;
    switch (baudrate) {
        case 9600:    baud = B9600;    break;
        case 19200:   baud = B19200;   break;
        case 38400:   baud = B38400;   break;
        case 57600:   baud = B57600;   break;
        case 115200:  baud = B115200;  break;
        case 230400:  baud = B230400;  break;
#ifdef __APPLE__
        // macOS uses the numeric value directly for non-POSIX baud rates
        default:      baud = static_cast<speed_t>(baudrate); break;
#else
        case 460800:  baud = B460800;  break;
        case 500000:  baud = B500000;  break;
        case 576000:  baud = B576000;  break;
        case 921600:  baud = B921600;  break;
        case 1000000: baud = B1000000; break;
        default:
            std::fprintf(stderr, "SerialPort: unsupported baudrate %d\n", baudrate);
            ::close(fd_);
            fd_ = -1;
            return false;
#endif
    }

    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    // Raw mode (8N1, no flow control)
    cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // Non-blocking read: VMIN=0, VTIME=0 -> return immediately with available data
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::fprintf(stderr, "SerialPort: tcsetattr failed: %s\n", std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Flush any existing data
    tcflush(fd_, TCIOFLUSH);

    return true;
}

void SerialPort::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

static int64_t mono_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

bool SerialPort::write(const uint8_t* buf, size_t n, int timeout_us) {
    if (fd_ < 0) return false;
    const int64_t deadline = mono_us() + std::max(timeout_us, 0);
    size_t written = 0;
    while (written < n) {
        ssize_t r = ::write(fd_, buf + written, n - written);
        if (r > 0) {
            written += static_cast<size_t>(r);
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return false;
        // Output buffer full: wait for room, but only until the deadline.
        const int64_t left_us = deadline - mono_us();
        if (left_us <= 0) return false;
        struct pollfd pfd = {fd_, POLLOUT, 0};
#ifdef __linux__
        // ppoll: microsecond deadline (poll()'s ms granularity would turn the RT
        // loop's 500 us budget into 1 ms).
        struct timespec ts = {static_cast<time_t>(left_us / 1000000),
                              static_cast<long>((left_us % 1000000) * 1000)};
        const int ret = ::ppoll(&pfd, 1, &ts, nullptr);
#else
        const int ret = ::poll(&pfd, 1, static_cast<int>((left_us + 999) / 1000));
#endif
        if (ret < 0 && errno != EINTR) return false;
        if (ret == 0 && mono_us() >= deadline) return false;
        if (ret > 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
    }
    return true;
}

size_t SerialPort::read_all(uint8_t* buf, size_t max) {
    if (fd_ < 0) return 0;
    ssize_t r = ::read(fd_, buf, max);
    if (r < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        return 0;
    }
    return static_cast<size_t>(r);
}

size_t SerialPort::read_with_timeout(uint8_t* buf, size_t max, int timeout_us) {
    if (fd_ < 0) return 0;

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    struct timeval tv;
    tv.tv_sec = timeout_us / 1000000;
    tv.tv_usec = timeout_us % 1000000;

    int ret = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
    if (ret <= 0) return 0;  // timeout or error

    // Data available — read all that's there
    size_t total = 0;
    while (total < max) {
        ssize_t r = ::read(fd_, buf + total, max - total);
        if (r <= 0) break;
        total += static_cast<size_t>(r);
        // Check if more data is available without blocking
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        tv.tv_sec = 0;
        tv.tv_usec = 0;
        if (select(fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) break;
    }
    return total;
}

bool SerialPort::close_bounded(int timeout_ms) {
    if (fd_ < 0) return true;
    const int fd = fd_;
    fd_ = -1;
    // Drop whatever the device never took, so close() has nothing to wait for
    // (drivers that honour TCOFLUSH); the helper thread covers those that don't.
    tcflush(fd, TCOFLUSH);
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread closer([fd, done] {
        ::close(fd);
        done->store(true, std::memory_order_release);
    });
    const int64_t deadline = mono_us() + static_cast<int64_t>(std::max(timeout_ms, 0)) * 1000;
    while (!done->load(std::memory_order_acquire) && mono_us() < deadline) {
        struct timespec ts = {0, 1000000};  // 1 ms
        nanosleep(&ts, nullptr);
    }
    if (done->load(std::memory_order_acquire)) {
        closer.join();
        return true;
    }
    closer.detach();  // owns only the fd number + the shared flag: safe to outlive us
    return false;
}

} // namespace trlc
