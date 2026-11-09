#pragma once

#include "stlqpp/core/status_or.h"

#include <filesystem>
#include <string>

namespace stlqpp::app {

struct ApplicationRunSessionOptions final {
  std::filesystem::path durable_root;
  bool archive_log = true;
  bool require_existing_root = false;
};

// Shared owner for durable run identity and the process-global logger
// attachment. Build and eval keep their own state/snapshot formats, but may
// not independently reimplement numbering, root resolution or logger cleanup.
class ApplicationRunSession final {
 public:
  [[nodiscard]] static StatusOr<ApplicationRunSession> Start(
      ApplicationRunSessionOptions options);

  ~ApplicationRunSession();
  ApplicationRunSession(ApplicationRunSession&& other) noexcept;
  ApplicationRunSession& operator=(ApplicationRunSession&& other) noexcept;
  ApplicationRunSession(const ApplicationRunSession&) = delete;
  ApplicationRunSession& operator=(const ApplicationRunSession&) = delete;

  [[nodiscard]] const std::filesystem::path& root() const noexcept {
    return root_;
  }
  [[nodiscard]] const std::filesystem::path& log_path() const noexcept {
    return log_path_;
  }
  [[nodiscard]] const std::string& run_id() const noexcept { return run_id_; }
  [[nodiscard]] bool log_attached() const noexcept { return log_attached_; }
  [[nodiscard]] Status MoveTo(const std::filesystem::path& durable_root);
  void Close() noexcept;

 private:
  ApplicationRunSession(std::filesystem::path root,
                        std::filesystem::path log_path,
                        std::string run_id,
                        bool log_attached) noexcept;

  std::filesystem::path root_;
  std::filesystem::path log_path_;
  std::string run_id_ = "latest";
  bool log_attached_ = false;
};

}  // namespace stlqpp::app
