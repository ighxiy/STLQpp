#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/platform/cpu_features.h"

namespace stlqpp {

class DenseCpuPlan final {
 public:
  [[nodiscard]] static StatusOr<DenseCpuPlan> Create(
      const CpuFeatures& features);
  [[nodiscard]] static StatusOr<DenseCpuPlan> Detect();

 private:
  DenseCpuPlan() = default;
};

}  // namespace stlqpp
