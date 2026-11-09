#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <string_view>

namespace stlqpp {

enum class OpenMpPlaces : std::uint8_t {
  kUnspecified,
  kCores,
  kThreads,
  kSockets,
};

enum class OpenMpProcBind : std::uint8_t {
  kUnspecified,
  kClose,
  kSpread,
  kPrimary,
};

struct OpenMpAffinityPolicy final {
  OpenMpPlaces places = OpenMpPlaces::kUnspecified;
  OpenMpProcBind proc_bind = OpenMpProcBind::kUnspecified;
  bool override_environment = false;
};

// Empty and "none" both preserve the environment-selected policy.
[[nodiscard]] StatusOr<OpenMpPlaces> ParseOpenMpPlaces(
    std::string_view value);
[[nodiscard]] StatusOr<OpenMpProcBind> ParseOpenMpProcBind(
    std::string_view value);

// Applies process environment policy before the first OpenMP parallel region.
// Algorithms still choose their typed worker count explicitly. This boundary
// controls only placement and binding, and has POSIX/Windows implementations.
[[nodiscard]] Status ConfigureOpenMpAffinityEnvironment(
    OpenMpAffinityPolicy policy);

// Ensures a policy changed by this process is visible when the OpenMP runtime
// initializes.  If applying the policy changes the environment, this function
// replaces the current process once with the same argument vector.  It returns
// only when no restart is needed or process replacement fails.
[[nodiscard]] Status PrepareOpenMpAffinityProcess(
    OpenMpAffinityPolicy policy,
    char* const argument_vector[]);

// Native pipeline threads must not inherit the current OpenMP worker's
// narrowed binding. Configure/Prepare captures the process-allowed CPU set
// before OpenMP initializes; each std::thread restores that baseline on
// entry. Calls are harmless when no baseline has been captured.
[[nodiscard]] Status RestoreNativeThreadAffinity();

[[nodiscard]] const char* OpenMpPlacesName(OpenMpPlaces places) noexcept;
[[nodiscard]] const char* OpenMpProcBindName(
    OpenMpProcBind proc_bind) noexcept;

}  // namespace stlqpp
