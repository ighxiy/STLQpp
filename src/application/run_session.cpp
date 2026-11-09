#include "stlqpp/application/run_session.h"

#include "stlqpp/application/run_sequence.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <system_error>
#include <utility>

namespace stlqpp::app {

ApplicationRunSession::ApplicationRunSession(
    std::filesystem::path root,
    std::filesystem::path log_path,
    std::string run_id,
    bool log_attached) noexcept
    : root_(std::move(root)), log_path_(std::move(log_path)),
      run_id_(std::move(run_id)), log_attached_(log_attached) {}

ApplicationRunSession::~ApplicationRunSession() {
  Close();
}

void ApplicationRunSession::Close() noexcept {
  if (log_attached_) {
    ClearLogFile();
    log_attached_ = false;
  }
}

ApplicationRunSession::ApplicationRunSession(
    ApplicationRunSession&& other) noexcept
    : root_(std::move(other.root_)), log_path_(std::move(other.log_path_)),
      run_id_(std::move(other.run_id_)),
      log_attached_(other.log_attached_) {
  other.log_attached_ = false;
}

ApplicationRunSession& ApplicationRunSession::operator=(
    ApplicationRunSession&& other) noexcept {
  if (this != &other) {
    Close();
    root_ = std::move(other.root_);
    log_path_ = std::move(other.log_path_);
    run_id_ = std::move(other.run_id_);
    log_attached_ = other.log_attached_;
    other.log_attached_ = false;
  }
  return *this;
}

Status ApplicationRunSession::MoveTo(
    const std::filesystem::path& durable_root) {
  if (durable_root.empty()) {
    return Status::InvalidArgument("durable run root is empty");
  }
  std::error_code error;
  std::filesystem::create_directories(durable_root, error);
  if (error) return Status::Internal("cannot create run root: " + error.message());
  if (!log_attached_) {
    root_ = durable_root;
    return Status::Ok();
  }
  const std::filesystem::path directory = durable_root / "log";
  std::filesystem::create_directories(directory, error);
  if (error) return Status::Internal("cannot create run log dir: " + error.message());
  auto next = ReserveRunLogPath(directory);
  if (!next.ok()) return next.status();
  ClearLogFile();
  log_attached_ = false;
  std::filesystem::rename(log_path_, next.value().path, error);
  if (error == std::errc::cross_device_link) {
    error.clear();
    std::filesystem::copy_file(log_path_, next.value().path,
                                 std::filesystem::copy_options::overwrite_existing, error);
    if (!error) {
      std::error_code remove_error;
      std::filesystem::remove(log_path_, remove_error);
    }
  }
  if (error) return Status::Internal("cannot transfer run log: " + error.message());
  std::string logger_error;
  if (!SetLogFile(next.value().path.string(), true, &logger_error)) {
    return Status::Internal(logger_error);
  }
  log_attached_ = true;
  root_ = durable_root;
  log_path_ = next.value().path;
  run_id_ = next.value().id;
  return Status::Ok();
}

StatusOr<ApplicationRunSession> ApplicationRunSession::Start(
    ApplicationRunSessionOptions options) {
  if (options.durable_root.empty()) {
    return Status::InvalidArgument("durable run root is empty");
  }
  std::error_code error;
  if (options.require_existing_root) {
    if (!std::filesystem::is_directory(options.durable_root, error) || error) {
      return Status::FailedPrecondition(
          "eval requires an existing durable run root: " +
          options.durable_root.string());
    }
  } else {
    std::filesystem::create_directories(options.durable_root, error);
    if (error) {
      return Status::Internal(
          "cannot create durable run root: " + error.message());
    }
  }

  std::string run_id = "latest";
  std::filesystem::path log_path;
  bool log_attached = false;
  if (options.archive_log) {
    const std::filesystem::path log_directory =
        options.durable_root / "log";
    std::filesystem::create_directories(log_directory, error);
    if (error) {
      LogWarn(
          "Failed to create log dir for runtime archive: " +
          log_directory.string());
    } else {
      auto sequence = ReserveRunLogPath(log_directory);
      if (!sequence.ok()) {
        LogWarn(sequence.status().message());
      } else {
        run_id = sequence.value().id;
        log_path = sequence.value().path;
        std::string logger_error;
        if (!SetLogFile(log_path.string(), false, &logger_error)) {
          LogWarn(logger_error);
          log_path.clear();
        } else {
          log_attached = true;
        }
      }
    }
  }
  return ApplicationRunSession(
      std::move(options.durable_root), std::move(log_path),
      std::move(run_id), log_attached);
}

}  // namespace stlqpp::app
