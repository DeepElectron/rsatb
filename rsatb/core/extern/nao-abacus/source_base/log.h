#ifndef MODULE_BASE_LOG_H
#define MODULE_BASE_LOG_H

#include <iostream>
#include <string>

namespace ModuleBase {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    OFF = 4
};

class Logger {
public:
    class LogProxy {
    public:
        LogProxy(std::ostream& os, bool enabled) : os_(os), enabled_(enabled) {}
        
        template <typename T>
        LogProxy& operator<<(const T& value) {
            if (enabled_) {
                os_ << value;
            }
            return *this;
        }

        // Handle manipulators like std::endl
        LogProxy& operator<<(std::ostream& (*manip)(std::ostream&)) {
            if (enabled_) {
                manip(os_);
            }
            return *this;
        }

    private:
        std::ostream& os_;
        bool enabled_;
    };

    Logger(std::ostream& os = std::cout, LogLevel level = LogLevel::INFO)
        : os_(os), level_(level) {}

    LogProxy debug() const { return LogProxy(os_, level_ <= LogLevel::DEBUG); }
    LogProxy info() const { return LogProxy(os_, level_ <= LogLevel::INFO); }
    LogProxy warn() const { return LogProxy(os_, level_ <= LogLevel::WARN); }
    LogProxy error() const { return LogProxy(os_, level_ <= LogLevel::ERROR); }

    void setLevel(LogLevel level) { level_ = level; }
    LogLevel getLevel() const { return level_; }

private:
    std::ostream& os_;
    LogLevel level_;
};

} // namespace ModuleBase

#endif // MODULE_BASE_LOG_H
