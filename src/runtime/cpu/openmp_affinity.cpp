#include "stlqpp/runtime/cpu/openmp_affinity.h"

namespace stlqpp {

StatusOr<OpenMpPlaces> ParseOpenMpPlaces(std::string_view value) {
  if (value.empty() || value == "none") return OpenMpPlaces::kUnspecified;
  if (value == "cores") return OpenMpPlaces::kCores;
  if (value == "threads") return OpenMpPlaces::kThreads;
  if (value == "sockets") return OpenMpPlaces::kSockets;
  return Status::InvalidArgument("invalid OpenMP places");
}

StatusOr<OpenMpProcBind> ParseOpenMpProcBind(std::string_view value) {
  if (value.empty() || value == "none") {
    return OpenMpProcBind::kUnspecified;
  }
  if (value == "close") return OpenMpProcBind::kClose;
  if (value == "spread") return OpenMpProcBind::kSpread;
  if (value == "primary") return OpenMpProcBind::kPrimary;
  return Status::InvalidArgument("invalid OpenMP proc bind");
}

const char* OpenMpPlacesName(OpenMpPlaces places) noexcept {
  switch (places) {
    case OpenMpPlaces::kUnspecified:
      return "";
    case OpenMpPlaces::kCores:
      return "cores";
    case OpenMpPlaces::kThreads:
      return "threads";
    case OpenMpPlaces::kSockets:
      return "sockets";
  }
  return "";
}

const char* OpenMpProcBindName(OpenMpProcBind proc_bind) noexcept {
  switch (proc_bind) {
    case OpenMpProcBind::kUnspecified:
      return "";
    case OpenMpProcBind::kClose:
      return "close";
    case OpenMpProcBind::kSpread:
      return "spread";
    case OpenMpProcBind::kPrimary:
      return "primary";
  }
  return "";
}

}  // namespace stlqpp
