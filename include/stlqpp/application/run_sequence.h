#pragma once

#include "stlqpp/core/status_or.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>

namespace stlqpp::app {

struct RunSequencePath final {
  std::uint64_t sequence = 0U;
  std::string id;
  std::filesystem::path path;
};

// DEV treats setw(4) as a minimum width. Every decimal run_<N>.log name,
// including five-digit and non-zero-padded historical names, participates in
// the sequence. Build and eval share this owner so they cannot drift again.
inline StatusOr<RunSequencePath> NextRunLogPath(
    const std::filesystem::path& directory) {
  std::uint64_t maximum = 0U;
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) {
    return Status::Internal(
        "cannot inspect runtime log directory: " + error.message());
  }
  if (exists) {
    std::filesystem::directory_iterator iterator(directory, error);
    const std::filesystem::directory_iterator end;
    if (error) {
      return Status::Internal(
          "cannot scan runtime log directory: " + error.message());
    }
    while (iterator != end) {
      std::error_code entry_error;
      const bool regular = iterator->is_regular_file(entry_error);
      if (entry_error) {
        return Status::Internal(
            "cannot inspect runtime log entry: " + entry_error.message());
      }
      if (regular) {
        const std::string name = iterator->path().filename().string();
        constexpr std::size_t kPrefix = 4U;
        constexpr std::size_t kSuffix = 4U;
        if (name.size() > kPrefix + kSuffix &&
            name.rfind("run_", 0U) == 0U &&
            name.compare(name.size() - kSuffix, kSuffix, ".log") == 0) {
          const std::string digits = name.substr(
              kPrefix, name.size() - kPrefix - kSuffix);
          std::uint64_t value = 0U;
          const auto parsed = std::from_chars(
              digits.data(), digits.data() + digits.size(), value);
          if (parsed.ec == std::errc{} &&
              parsed.ptr == digits.data() + digits.size()) {
            maximum = std::max(maximum, value);
          }
        }
      }
      iterator.increment(error);
      if (error) {
        return Status::Internal(
            "cannot continue runtime log scan: " + error.message());
      }
    }
  }
  if (maximum == std::numeric_limits<std::uint64_t>::max()) {
    return Status::ResourceExhausted("runtime log sequence is exhausted");
  }
  RunSequencePath output;
  output.sequence = maximum + 1U;
  std::ostringstream id;
  id << std::setw(4) << std::setfill('0') << output.sequence;
  output.id = id.str();
  output.path = directory / ("run_" + output.id + ".log");
  return output;
}

// Reserve one empty log with exclusive creation. Only this session may then
// replace/truncate that placeholder; other processes must take the next number.
inline StatusOr<RunSequencePath> ReserveRunLogPath(
    const std::filesystem::path& directory) {
  auto candidate = NextRunLogPath(directory);
  if (!candidate.ok()) return candidate.status();
  auto output = std::move(candidate).value();
  for (;;) {
    if (auto* file = std::fopen(output.path.string().c_str(), "wbx")) {
      if (std::fclose(file) != 0)
        return Status::Internal("cannot close reserved runtime log");
      return output;
    }
    if (errno != EEXIST)
      return Status::Internal("cannot reserve runtime log name");
    if (output.sequence == std::numeric_limits<std::uint64_t>::max())
      return Status::ResourceExhausted("runtime log sequence is exhausted");
    ++output.sequence;
    std::ostringstream id;
    id << std::setw(4) << std::setfill('0') << output.sequence;
    output.id = id.str();
    output.path = directory / ("run_" + output.id + ".log");
  }
}

}  // namespace stlqpp::app
