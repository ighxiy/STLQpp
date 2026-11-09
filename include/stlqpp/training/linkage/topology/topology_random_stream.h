#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace stlqpp {

// One mutable random stream owned by one causal cluster construction.  It is
// intentionally distinct from the fresh task-local stream used by CUDA
// RSD evaluation.
class TopologyRandomStream final {
 public:
  explicit TopologyRandomStream(std::uint32_t seed) noexcept
      : random_(seed) {}

  [[nodiscard]] std::vector<std::uint32_t> Permutation(
      std::uint32_t count);

  void FillIcmLayerOrder(std::uint32_t layer_count,
                         std::uint8_t* output);

 private:
  std::mt19937 random_;
};

}  // namespace stlqpp
