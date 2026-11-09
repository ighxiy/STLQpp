#include "stlqpp/training/linkage/topology/topology_random_stream.h"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace stlqpp {

std::vector<std::uint32_t> TopologyRandomStream::Permutation(
    std::uint32_t count) {
  std::vector<std::uint32_t> output(count);
  std::iota(output.begin(), output.end(), 0U);
  std::shuffle(output.begin(), output.end(), random_);
  return output;
}

void TopologyRandomStream::FillIcmLayerOrder(
    std::uint32_t layer_count,
    std::uint8_t* output) {
  assert(layer_count <= 256U);
  for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
    output[layer] = static_cast<std::uint8_t>(layer);
  }
  std::shuffle(output, output + layer_count, random_);
}

}  // namespace stlqpp
