#include "stlqpp/runtime/telemetry/logger.h"

#include <cassert>
#include <iostream>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace stlqpp {
namespace {

const char* LogTag(LogLevel level) noexcept {
  if (level == LogLevel::kDebug) {
    return "DEBUG";
  }
  if (level == LogLevel::kWarn) {
    return "WARN";
  }
  if (level == LogLevel::kError) {
    return "ERROR";
  }
  return "INFO";
}

std::string TaggedLine(LogLevel level, const std::string& message) {
  return std::string("[") + LogTag(level) + "] " + message;
}

}  // namespace

Logger& Logger::Instance() {
  static Logger logger;
  return logger;
}

void Logger::Log(LogLevel level, const std::string& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (progress_active_) {
    std::cerr << '\n';
    progress_active_ = false;
    progress_length_ = 0U;
  }
  const std::string line = TaggedLine(level, message);
  std::cerr << line << '\n';
  if (log_file_ != nullptr) {
    assert(log_file_->is_open());
    (*log_file_) << line << '\n';
    log_file_->flush();
  } else {
    history_.push_back(line);
  }
}

void Logger::LogProgress(
    LogLevel level, const std::string& message, bool done) {
  std::lock_guard<std::mutex> guard(mutex_);
  std::cerr << '\r' << '[' << LogTag(level) << "] " << message;
  const std::size_t new_length = message.size();
  if (progress_active_ && new_length < progress_length_) {
    std::cerr << std::string(progress_length_ - new_length, ' ');
  }
  progress_active_ = !done;
  progress_length_ = done ? 0U : new_length;
  if (done) {
    std::cerr << '\n';
  }
  std::cerr.flush();
  if (!done) {
    return;
  }
  const std::string line = TaggedLine(level, message);
  if (log_file_ != nullptr) {
    assert(log_file_->is_open());
    (*log_file_) << line << '\n';
    log_file_->flush();
  } else {
    history_.push_back(line);
  }
}

void Logger::LogRawProgress(const std::string& message, bool done) {
  std::lock_guard<std::mutex> guard(mutex_);
  std::cerr << '\r' << message;
  const std::size_t new_length = message.size();
  if (progress_active_ && new_length < progress_length_) {
    std::cerr << std::string(progress_length_ - new_length, ' ');
  }
  progress_active_ = !done;
  progress_length_ = done ? 0U : new_length;
  if (done) {
    std::cerr << '\n';
  }
  std::cerr.flush();
}

bool Logger::SetLogFile(
    const std::string& path, bool append, std::string* error) {
  std::lock_guard<std::mutex> guard(mutex_);
  log_file_.reset();
  if (path.empty()) {
    return true;
  }
  auto output = std::make_unique<std::ofstream>(
      path, std::ios::out | std::ios::binary |
          (append ? std::ios::app : std::ios::trunc));
  if (!output->is_open()) {
    if (error != nullptr) {
      *error = "Failed to open log file: " + path;
    }
    return false;
  }
  for (const std::string& line : history_) {
    (*output) << line << '\n';
  }
  output->flush();
  history_.clear();
  log_file_ = std::move(output);
  return true;
}

void Logger::ClearLogFile() {
  std::lock_guard<std::mutex> guard(mutex_);
  log_file_.reset();
}

void LogDebug(const std::string& message) {
#ifndef NDEBUG
  Logger::Instance().Log(LogLevel::kDebug, message);
#else
  (void)message;
#endif
}

void LogInfo(const std::string& message) {
  Logger::Instance().Log(LogLevel::kInfo, message);
}

void LogWarn(const std::string& message) {
  Logger::Instance().Log(LogLevel::kWarn, message);
}

void LogError(const std::string& message) {
  Logger::Instance().Log(LogLevel::kError, message);
}

void LogInfoProgress(const std::string& message) {
  Logger::Instance().LogProgress(LogLevel::kInfo, message, false);
}

void LogInfoProgressDone(const std::string& message) {
  Logger::Instance().LogProgress(LogLevel::kInfo, message, true);
}

void LogRawProgress(const std::string& message) {
  Logger::Instance().LogRawProgress(message, false);
}

void LogRawProgressDone(const std::string& message) {
  Logger::Instance().LogRawProgress(message, true);
}

bool ProgressStreamIsInteractive() noexcept {
#if defined(_WIN32)
  return _isatty(_fileno(stderr)) != 0;
#else
  return isatty(fileno(stderr)) != 0;
#endif
}

bool SetLogFile(
    const std::string& path, bool append, std::string* error) {
  return Logger::Instance().SetLogFile(path, append, error);
}

void ClearLogFile() {
  Logger::Instance().ClearLogFile();
}

}  // namespace stlqpp
