#include "stlqpp/runtime/cpu/openmp_affinity.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <sched.h>
#include <unistd.h>

namespace stlqpp {
namespace {

constexpr const char* kAffinityReadyEnvironment =
    "STLQPP_OMP_AFFINITY_READY";
constexpr const char* kNativeAffinityEnvironment =
    "STLQPP_NATIVE_AFFINITY_BASELINE";

std::mutex baseline_mutex;
cpu_set_t baseline_affinity;
bool baseline_affinity_valid = false;

char HexDigit(unsigned int value) noexcept {
  return static_cast<char>(value < 10U ? '0' + value : 'a' + value - 10U);
}

int HexValue(char value) noexcept {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

Status DecodeNativeAffinity(const char* encoded, cpu_set_t* output) {
  const std::size_t bytes = sizeof(*output);
  if (encoded == nullptr || std::strlen(encoded) != bytes * 2U) {
    return Status::InvalidArgument(
        "invalid native thread affinity baseline");
  }
  auto* destination = reinterpret_cast<unsigned char*>(output);
  for (std::size_t index = 0U; index < bytes; ++index) {
    const int high = HexValue(encoded[index * 2U]);
    const int low = HexValue(encoded[index * 2U + 1U]);
    if (high < 0 || low < 0) {
      return Status::InvalidArgument(
          "invalid native thread affinity baseline");
    }
    destination[index] = static_cast<unsigned char>((high << 4U) | low);
  }
  return Status::Ok();
}

Status PersistNativeAffinity(const cpu_set_t& affinity) {
  const auto* source = reinterpret_cast<const unsigned char*>(&affinity);
  std::string encoded(sizeof(affinity) * 2U, '0');
  for (std::size_t index = 0U; index < sizeof(affinity); ++index) {
    encoded[index * 2U] = HexDigit(source[index] >> 4U);
    encoded[index * 2U + 1U] = HexDigit(source[index] & 0x0fU);
  }
  return setenv(kNativeAffinityEnvironment, encoded.c_str(), 1) == 0
      ? Status::Ok()
      : Status::Internal(
            "failed to preserve native thread affinity baseline");
}

Status CaptureNativeThreadAffinity() {
  std::lock_guard<std::mutex> lock(baseline_mutex);
  if (baseline_affinity_valid) return Status::Ok();
  CPU_ZERO(&baseline_affinity);
  if (const char* encoded = std::getenv(kNativeAffinityEnvironment);
      encoded != nullptr) {
    if (const Status status = DecodeNativeAffinity(
            encoded, &baseline_affinity);
        !status.ok()) {
      return status;
    }
  } else if (sched_getaffinity(
                 0, sizeof(baseline_affinity), &baseline_affinity) != 0) {
    const int error = errno;
    return Status::Internal(
        std::string("failed to capture native thread affinity: ") +
        std::strerror(error));
  }
  if (const Status status = PersistNativeAffinity(baseline_affinity);
      !status.ok()) {
    return status;
  }
  baseline_affinity_valid = true;
  return Status::Ok();
}

bool EnvironmentEquals(const char* name, const char* value) {
  const char* current = std::getenv(name);
  return current != nullptr && std::strcmp(current, value) == 0;
}

Status SetIfRequested(const char* name,
                      const char* value,
                      bool override_environment) {
  if (value[0] == '\0') {
    return Status::Ok();
  }
  if (!override_environment && std::getenv(name) != nullptr) {
    return Status::Ok();
  }
  if (setenv(name, value, 1) != 0) {
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
  cpu_set_t affinity;
  {
    std::lock_guard<std::mutex> lock(baseline_mutex);
    if (!baseline_affinity_valid) return Status::Ok();
    affinity = baseline_affinity;
  }
  const int result = sched_setaffinity(0, sizeof(affinity), &affinity);
  const int error = errno;
  return result == 0
      ? Status::Ok()
      : Status::Internal(
            std::string("failed to restore native thread affinity: ") +
            std::strerror(error));
}

Status PrepareOpenMpAffinityProcess(OpenMpAffinityPolicy policy,
                                    char* const argument_vector[]) {
  const char* places = OpenMpPlacesName(policy.places);
  const char* proc_bind = OpenMpProcBindName(policy.proc_bind);
  const bool places_change =
      places[0] != '\0' &&
      (policy.override_environment || std::getenv("OMP_PLACES") == nullptr) &&
      !EnvironmentEquals("OMP_PLACES", places);
  const bool proc_bind_change =
      proc_bind[0] != '\0' &&
      (policy.override_environment ||
       std::getenv("OMP_PROC_BIND") == nullptr) &&
      !EnvironmentEquals("OMP_PROC_BIND", proc_bind);

  if (const Status status = ConfigureOpenMpAffinityEnvironment(policy);
      !status.ok()) {
    return status;
  }
  if ((!places_change && !proc_bind_change) ||
      std::getenv(kAffinityReadyEnvironment) != nullptr) {
    return Status::Ok();
  }
  if (setenv(kAffinityReadyEnvironment, "1", 1) != 0) {
    return Status::Internal(
        "failed to mark OpenMP affinity process preparation");
  }
  execvp(argument_vector[0], argument_vector);
  const int error = errno;
  unsetenv(kAffinityReadyEnvironment);
  return Status::Internal(
      std::string("failed to restart process for OpenMP affinity: ") +
      std::strerror(error));
}

}  // namespace stlqpp
