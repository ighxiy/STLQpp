#include "stlqpp/pipeline/final_replay_spool_sink.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
Status FinalReplaySpoolSink<M>::Commit(
    std::uint64_t sequence,
    ArrayView<const FixedTopologyReplayRequest> requests,
    ArrayView<const FixedTopologyReplayResult<M>> results) {
  if (sequence != next_sequence_ || requests.empty() ||
      requests.size() != results.size()) {
    return Status::FailedPrecondition(
        "final replay spool received an out-of-order or invalid window");
  }
  std::vector<FinalForestSpoolClusterView<M>> clusters;
  std::vector<std::vector<float>> source_norms(requests.size());
  clusters.reserve(requests.size());
  for (std::size_t local = 0U; local < requests.size(); ++local) {
    if (results[local].topology.cid != requests[local].cid ||
        requests[local].real_sample_ids.size() !=
            results[local].topology.roles.size()) {
      return Status::InvalidArgument(
          "final replay result changed its source CID or row count");
    }
    if (forest_sink_->RequiresSourceNorm2()) {
      const ColumnMajorMatrixF32View exact =
          requests[local].unrotated_exact_real;
      source_norms[local].assign(exact.columns, 0.0F);
      for (std::uint32_t row = 0U; row < exact.columns; ++row) {
        const float* vector =
            exact.values + static_cast<std::size_t>(row) * exact.rows;
        float norm2 = 0.0F;
#pragma omp simd reduction(+ : norm2)
        for (std::uint32_t axis = 0U; axis < exact.rows; ++axis) {
          norm2 += vector[axis] * vector[axis];
        }
        if (!std::isfinite(norm2) || norm2 < 0.0F) {
          return Status::DataLoss(
              "final replay source contains an invalid original-vector norm");
        }
        source_norms[local][row] = norm2;
      }
    }
    clusters.push_back({
        .real_sample_ids = requests[local].real_sample_ids,
        .source_real_norm2 = source_norms[local],
        .topology = &results[local].topology,
        .virtual_codes_small = results[local].virtual_small_codes,
        .virtual_coefficients_node_major = results[local].virtual_coefficients,
    });
  }
  if (const Status status = forest_sink_->AppendWindow(clusters);
      !status.ok()) {
    return status;
  }
  ++next_sequence_;
  return Status::Ok();
}

template class FinalReplaySpoolSink<0>;
template class FinalReplaySpoolSink<5>;
template class FinalReplaySpoolSink<12>;
template class FinalReplaySpoolSink<6>;
template class FinalReplaySpoolSink<10>;

}  // namespace stlqpp
