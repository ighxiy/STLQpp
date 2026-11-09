#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace stlqpp {

enum class LogLevel {
  kDebug,
  kInfo,
  kWarn,
  kError,
};

// Production logger contract inherited unchanged from Master. Progress is
// rendered only on stderr; completed lines are retained until a durable log
// file is attached and are then flushed in original order.
class Logger final {
 public:
  [[nodiscard]] static Logger& Instance();

  void Log(LogLevel level, const std::string& message);
  void LogProgress(LogLevel level, const std::string& message, bool done);
  void LogRawProgress(const std::string& message, bool done);
  [[nodiscard]] bool SetLogFile(
      const std::string& path, bool append, std::string* error);
  void ClearLogFile();

 private:
  Logger() = default;

  std::mutex mutex_;
  bool progress_active_ = false;
  std::size_t progress_length_ = 0U;
  std::unique_ptr<std::ofstream> log_file_;
  std::vector<std::string> history_;
};

void LogDebug(const std::string& message);
void LogInfo(const std::string& message);
void LogWarn(const std::string& message);
void LogError(const std::string& message);
void LogInfoProgress(const std::string& message);
void LogInfoProgressDone(const std::string& message);
void LogRawProgress(const std::string& message);
void LogRawProgressDone(const std::string& message);
[[nodiscard]] bool ProgressStreamIsInteractive() noexcept;
[[nodiscard]] bool SetLogFile(
    const std::string& path, bool append, std::string* error);
void ClearLogFile();

}  // namespace stlqpp
