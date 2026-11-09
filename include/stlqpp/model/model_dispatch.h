#pragma once

#include "stlqpp/model/model_shape.h"
#include <cstdint>
#include <utility>

#include "stlqpp/core/status.h"

namespace stlqpp {

// Fixed-M specializations. Callers must select a separate runtime-M entry
// point for all other supported lengths; this dispatcher never impersonates it.
template <std::uint32_t M = 5U, typename Function>
auto DispatchModelM(std::uint32_t m, Function&& function)
    -> decltype(function.template operator()<M>()) {
  using Result = decltype(function.template operator()<M>());
  if (m == M) return function.template operator()<M>();
  if constexpr (M == 5U) {
    return DispatchModelM<6U>(m, std::forward<Function>(function));
  } else if constexpr (M == 6U) {
    return DispatchModelM<10U>(m, std::forward<Function>(function));
  } else if constexpr (M == 10U) {
    return DispatchModelM<12U>(m, std::forward<Function>(function));
  } else {
    return Result(Status::Unimplemented(
        "this operation has no runtime-M implementation; fixed-M paths "
        "are available for M=5,6,10,12"));
  }
}

template <std::uint32_t M = 5U, typename Function, typename Result>
Result DispatchModelMOr(std::uint32_t m, Function&& function,
                        Result unsupported) {
  if (m == M) return function.template operator()<M>();
  if constexpr (M == 5U) {
    return DispatchModelMOr<6U>(m, std::forward<Function>(function),
                                unsupported);
  } else if constexpr (M == 6U) {
    return DispatchModelMOr<10U>(m, std::forward<Function>(function),
                                 unsupported);
  } else if constexpr (M == 10U) {
    return DispatchModelMOr<12U>(m, std::forward<Function>(function),
                                 unsupported);
  } else {
    return unsupported;
  }
}

// One runtime-width fallback beside the existing fixed-M fast paths.
template <typename Function>
auto DispatchModelMWithRuntime(std::uint32_t m, Function&& function)
    -> decltype(function.template operator()<0U>()) {
  using Result = decltype(function.template operator()<0U>());
  if (m < 2U || m > stlqpp::kMaximumModelLayers)
    return Result(Status::InvalidArgument("model M must be in [2,24]"));
  if (m == 5U || m == 6U || m == 10U || m == 12U)
    return DispatchModelM(m, std::forward<Function>(function));
  return function.template operator()<0U>();
}

}  // namespace stlqpp
