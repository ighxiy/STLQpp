#pragma once

#include <cstdint>

namespace stlqpp {

// Producer task boundaries are independent from whether evaluation is
// implemented by direct CUDA lanes or scheduler workers. Stable scheduling
// preserves those boundaries; throughput scheduling lets the evaluator choose
// a wider numerical batch.
enum class LinkageProducerTaskPolicy : std::uint8_t {
  kEvaluatorSelected = 0,
  kStable = 1,
};

}  // namespace stlqpp
