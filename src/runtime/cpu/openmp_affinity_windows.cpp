#include "stlqpp/runtime/cpu/openmp_affinity.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <process.h>
#include <string>

#include <windows.h>

namespace stlqpp {
namespace {

constexpr const char* kAffinityReadyEnvironment =
    "STLQPP_OMP_AFFINITY_READY";
constexpr const char* kNativeAffinityEnvironment =
    "STLQPP_NATIVE_AFFINITY_BASELINE";

std::mutex baseline_mutex;
DWORD_PTR baseline_affinity = 0U;

Status CaptureNativeThreadAffinity() {
  std::lock_guard<std::mutex> lock(baseline_mutex);
  if (baseline_affinity != 0U) return Status::Ok();
  char* encoded = nullptr;
  std::size_t encoded_size = 0U;
  const errno_t environment_error = _dupenv_s(
      &encoded, &encoded_size, kNativeAffinityEnvironment);
  std::unique_ptr<char, decltype(&std::free)> owner(encoded, &std::free);
  if (environment_error == 0 && encoded != nullptr) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(encoded, &end, 16);
    if (end == encoded || *end != '\0' || parsed == 0U) {
      return Status::InvalidArgument(
          "invalid native thread affinity baseline");
    }
    baseline_affinity = static_cast<DWORD_PTR>(parsed);
  } else {
    DWORD_PTR process_affinity = 0U;
    DWORD_PTR system_affinity = 0U;
    if (!GetProcessAffinityMask(
            GetCurrentProcess(), &process_affinity, &system_affinity) ||
        process_affinity == 0U) {
      return Status::Internal("failed to capture native thread affinity");
    }
    baseline_affinity = process_affinity;
  }
  char persisted[2U * sizeof(DWORD_PTR) + 1U]{};
  if (std::snprintf(
          persisted, sizeof(persisted), "%llx",
          static_cast<unsigned long long>(baseline_affinity)) < 0 ||
      _putenv_s(kNativeAffinityEnvironment, persisted) != 0) {
    return Status::Internal(
        "failed to preserve native thread affinity baseline");
  }
  return Status::Ok();
}

bool EnvironmentExists(const char* name) {
  char* value = nullptr;
  std::size_t size = 0U;
  const errno_t error = _dupenv_s(&value, &size, name);
  std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return error == 0 && size != 0U;
}

bool EnvironmentEquals(const char* name, const char* expected) {
  char* value = nullptr;
  std::size_t size = 0U;
  const errno_t error = _dupenv_s(&value, &size, name);
  std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return error == 0 && value != nullptr && std::strcmp(value, expected) == 0;
}

Status SetIfRequested(const char* name,
                      const char* value,
                      bool override_environment) {
  if (value[0] == '\0') {
    return Status::Ok();
  }
  if (!override_environment && EnvironmentExists(name)) {
    return Status::Ok();
  }
  if (_putenv_s(name, value) != 0) {
    return Status::Internal(
        "failed to configure OpenMP affinity environment");
  }
  return Status::Ok();
}

}  // namespace

Status ConfigureOpenMpAffinityEnvironment(OpenMpAffinityPolicy policy) {
  if (const Status status = CaptureNativeThreadAffinity(); !status.ok()) {
    return status;
  }
  if (const Status status = SetIfRequested(
          "OMP_PLACES", OpenMpPlacesName(policy.places),
          policy.override_environment);
      !status.ok()) {
    return status;
  }
  return SetIfRequested(
      "OMP_PROC_BIND", OpenMpProcBindName(policy.proc_bind),
      policy.override_environment);
}

Status RestoreNativeThreadAffinity() {
  DWORD_PTR affinity = 0U;
  {
    std::lock_guard<std::mutex> lock(baseline_mutex);
    affinity = baseline_affinity;
  }
  if (affinity == 0U) return Status::Ok();
  return SetThreadAffinityMask(GetCurrentThread(), affinity) != 0U
      ? Status::Ok()
      : Status::Internal("failed to restore native thread affinity");
}

Status PrepareOpenMpAffinityProcess(OpenMpAffinityPolicy policy,
                                    char* const argument_vector[]) {
  const char* places = OpenMpPlacesName(policy.places);
  const char* proc_bind = OpenMpProcBindName(policy.proc_bind);
  const bool places_change =
      places[0] != '\0' &&
      (policy.override_environment || !EnvironmentExists("OMP_PLACES")) &&
      !EnvironmentEquals("OMP_PLACES", places);
  const bool proc_bind_change =
      proc_bind[0] != '\0' &&
      (policy.override_environment ||
       !EnvironmentExists("OMP_PROC_BIND")) &&
      !EnvironmentEquals("OMP_PROC_BIND", proc_bind);

  if (const Status status = ConfigureOpenMpAffinityEnvironment(policy);
      !status.ok()) {
    return status;
  }
  if ((!places_change && !proc_bind_change) ||
      EnvironmentExists(kAffinityReadyEnvironment)) {
    return Status::Ok();
  }
  if (_putenv_s(kAffinityReadyEnvironment, "1") != 0) {
    return Status::Internal(
        "failed to mark OpenMP affinity process preparation");
  }
  _execvp(argument_vector[0], argument_vector);
  _putenv_s(kAffinityReadyEnvironment, "");
  return Status::Internal(
      "failed to restart process for OpenMP affinity");
}

}  // namespace stlqpp
