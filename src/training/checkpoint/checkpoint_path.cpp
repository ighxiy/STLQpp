#include "stlqpp/training/checkpoint/checkpoint_path.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace stlqpp {
namespace {

StatusOr<std::string> TodayYyyyMmDd() {
  const std::time_t now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm local{};
#if defined(_WIN32)
  if (localtime_s(&local, &now) != 0) {
#else
  if (localtime_r(&now, &local) == nullptr) {
#endif
    return Status::Internal("failed to resolve the local checkpoint date");
  }
  std::ostringstream text;
  text << std::put_time(&local, "%Y%m%d");
  return text.str();
}

}  // namespace

StatusOr<std::filesystem::path> NextDatedCheckpointPath(
    const std::filesystem::path& base_path) {
  if (base_path.empty() || base_path.filename().empty()) {
    return Status::InvalidArgument(
        "periodic checkpoint base path must name an artifact");
  }
  auto date = TodayYyyyMmDd();
  if (!date.ok()) return date.status();
  const std::filesystem::path parent = base_path.parent_path();
  const std::string stem = base_path.stem().string();
  const std::string extension = base_path.extension().string();
  for (std::uint32_t sequence = 0U; sequence <= 999U; ++sequence) {
    std::ostringstream name;
    name << stem << '_' << date.value();
    if (sequence != 0U) {
      name << '_' << std::setfill('0') << std::setw(3) << sequence;
    }
    name << extension;
    const std::filesystem::path candidate = parent / name.str();
    std::error_code error;
    const bool exists = std::filesystem::exists(candidate, error);
    if (error) {
      return Status::Internal(
          "failed to inspect periodic checkpoint destination " +
          candidate.string());
    }
    if (!exists) return candidate;
  }
  return Status::ResourceExhausted(
      "periodic checkpoint date namespace exhausted for " +
      base_path.string());
}

}  // namespace stlqpp
