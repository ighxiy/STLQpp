#include "stlqpp/training/analysis/initialization_post_edge_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate(
    const BaseListReader<M>& lists,
    const RootOnlyInitializationTranscript<M>& transcript,
    const TrainingModel<M>& model,
    const RootToEdgeCodeMap& root_to_edge,
    const ParallelForExecutor& parallel) {
  const ProductCodebooks<M>& codebooks = model.codebooks();
  const ModelShape& shape = codebooks.shape();
  const BaseBuildShape& list_shape = lists.store().shape();
  if (transcript.m() != shape.m || lists.store().m() != shape.m ||
      transcript.dimension() != shape.dimension ||
      transcript.nlist() != shape.nlist ||
      list_shape.dimension != shape.dimension ||
      list_shape.nlist != shape.nlist ||
      root_to_edge.root_count() != shape.nlist) {
    return Status::InvalidArgument(
        "post-C_one initialization analysis inputs are incompatible");
  }

  InitializationPostEdgeAnalysis output;
  struct Scratch {
    std::vector<float> reconstructions;
    std::vector<std::uint32_t> causal_order;
  };
  struct Result {
    Status status;
    InitializationPostEdgeAnalysis summary;
    std::vector<double> row_errors;
  };
  std::vector<Scratch> scratch(parallel.worker_count());
  auto analyze = [&](std::uint32_t cid, Scratch& work, Result& result) -> Status {
    auto& reconstructions = work.reconstructions;
    auto& causal_order = work.causal_order;
    auto& summary = result.summary;
    auto loaded = lists.LoadCid(cid);
    if (!loaded.ok()) return loaded.status();
    auto staged = transcript.LoadCluster(cid);
    if (!staged.ok()) return staged.status();
    const std::uint32_t rows = loaded.value().row_count();
    if (rows != staged.value().real_count()) {
      return Status::FailedPrecondition(
          "post-C_one initialization analysis row order changed");
    }

    reconstructions.assign(
        static_cast<std::size_t>(shape.dimension) * rows, 0.0F);
    causal_order.resize(rows);
    std::iota(causal_order.begin(), causal_order.end(), 0U);
    std::stable_sort(
        causal_order.begin(), causal_order.end(),
        [&](std::uint32_t left, std::uint32_t right) {
          return staged.value().depths[left] < staged.value().depths[right];
        });

    result.row_errors.reserve(rows);
    const ColumnMajorMatrixF32View exact = loaded.value().exact_vectors();
    for (const std::uint32_t row : causal_order) {
      float* reconstruction = reconstructions.data() +
          static_cast<std::size_t>(row) * shape.dimension;
      const LinkageNodeRole role = staged.value().roles[row];
      const LinkageCodeTuple<M>& support = staged.value().supports[row];
      const float* coefficients = staged.value().coefficients.data() +
          static_cast<std::size_t>(row) * shape.m;
      if (role == LinkageNodeRole::kRoot) {
        if (support.first >= shape.nlist ||
            staged.value().parents[row].has_value() ||
            staged.value().depths[row] != 0U) {
          return Status::DataLoss(
              "post-C_one initialization root is invalid");
        }
        for (std::uint32_t layer = 0U;
             layer < shape.m; ++layer) {
          const float* atom = layer == 0U
              ? codebooks.root0().column(support.first)
              : codebooks.root_small(layer).column(
                    support.trailing[layer - 1U]);
          const float coefficient = coefficients[layer];
          for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
            reconstruction[axis] += coefficient * atom[axis];
          }
        }
      } else {
        const auto& parent = staged.value().parents[row];
        if (role != LinkageNodeRole::kLinked || !parent.has_value() ||
            parent->kind != LinkageParentKind::kReal ||
            parent->local_id >= rows || support.first >= shape.nlist ||
            staged.value().depths[row] == 0U ||
            staged.value().depths[parent->local_id] + 1U !=
                staged.value().depths[row]) {
          return Status::DataLoss(
              "post-C_one initialization link is invalid");
        }
        const float* parent_reconstruction = reconstructions.data() +
            static_cast<std::size_t>(parent->local_id) * shape.dimension;
        std::copy_n(parent_reconstruction, shape.dimension, reconstruction);
        for (std::uint32_t layer = 0U;
             layer < shape.m; ++layer) {
          const std::uint32_t code = layer == 0U
              ? root_to_edge.edge_code(support.first)
              : support.trailing[layer - 1U];
          const float* atom = codebooks.edge(layer).column(code);
          const float coefficient = coefficients[layer];
          for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
            reconstruction[axis] += coefficient * atom[axis];
          }
        }
        ++summary.linked_real_rows;
      }

      double squared_error = 0.0;
      const float* expected = exact.column(row);
      for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
        const double difference =
            static_cast<double>(expected[axis]) - reconstruction[axis];
        squared_error += difference * difference;
      }
      if (!std::isfinite(squared_error)) {
        return Status::DataLoss(
            "post-C_one initialization analysis produced an invalid error");
      }
      result.row_errors.push_back(squared_error);
      summary.depth_sum += staged.value().depths[row];
      summary.maximum_depth = std::max(
          summary.maximum_depth, staged.value().depths[row]);
      ++summary.real_rows;
    }
    ++summary.clusters;
    return Status::Ok();
  };
  // Keep only a bounded window of per-row errors. Reconstruction is parallel,
  // but the public statistics retain the original CID/causal-row sum order.
  constexpr std::uint64_t kWindowRows = 262144U;
  const auto offsets = lists.store().offsets();
  for (std::uint32_t begin = 0; begin < shape.nlist;) {
    std::uint32_t end = begin;
    std::uint64_t rows = 0;
    while (end < shape.nlist) {
      const auto count = offsets[end + 1U] - offsets[end];
      if (end != begin &&
          (end - begin >= parallel.worker_count() * 4U ||
           rows + count > kWindowRows)) break;
      rows += count;
      ++end;
    }
    std::vector<Result> results(end - begin);
    struct Context {
      decltype(analyze)* function;
      Scratch* scratch;
      Result* results;
      std::uint32_t begin;
    } context{&analyze, scratch.data(), results.data(), begin};
    parallel.Run(end - begin, &context,
                 [](void* opaque, std::uint32_t worker,
                    std::uint32_t item) noexcept {
      auto& state = *static_cast<Context*>(opaque);
      state.results[item].status = (*state.function)(
          state.begin + item, state.scratch[worker], state.results[item]);
    });
    for (const auto& result : results) {
      if (!result.status.ok()) return result.status;
      const auto& summary = result.summary;
      output.clusters += summary.clusters;
      output.real_rows += summary.real_rows;
      output.linked_real_rows += summary.linked_real_rows;
      output.depth_sum += summary.depth_sum;
      output.maximum_depth = std::max(output.maximum_depth, summary.maximum_depth);
      for (const double error : result.row_errors) {
        output.squared_error_sum += error;
        if (!output.has_squared_error) {
          output.minimum_squared_error = error;
          output.maximum_squared_error = error;
          output.has_squared_error = true;
        } else {
          output.minimum_squared_error = std::min(output.minimum_squared_error, error);
          output.maximum_squared_error = std::max(output.maximum_squared_error, error);
        }
      }
    }
    begin = end;
  }
  if (output.clusters != shape.nlist ||
      output.real_rows != lists.store().total_vectors() ||
      output.real_rows == 0U || !output.has_squared_error) {
    return Status::DataLoss(
        "post-C_one initialization analysis did not cover the artifact");
  }
  return output;
}

template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<0>(
    const BaseListReader<0>&,
    const RootOnlyInitializationTranscript<0>&,
    const TrainingModel<0>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<5>(
    const BaseListReader<5>&,
    const RootOnlyInitializationTranscript<5>&,
    const TrainingModel<5>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<2>(
    const BaseListReader<2>&,
    const RootOnlyInitializationTranscript<2>&,
    const TrainingModel<2>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<3>(
    const BaseListReader<3>&,
    const RootOnlyInitializationTranscript<3>&,
    const TrainingModel<3>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<4>(
    const BaseListReader<4>&,
    const RootOnlyInitializationTranscript<4>&,
    const TrainingModel<4>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<7>(
    const BaseListReader<7>&,
    const RootOnlyInitializationTranscript<7>&,
    const TrainingModel<7>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<8>(
    const BaseListReader<8>&,
    const RootOnlyInitializationTranscript<8>&,
    const TrainingModel<8>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<9>(
    const BaseListReader<9>&,
    const RootOnlyInitializationTranscript<9>&,
    const TrainingModel<9>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<11>(
    const BaseListReader<11>&,
    const RootOnlyInitializationTranscript<11>&,
    const TrainingModel<11>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<12>(
    const BaseListReader<12>&,
    const RootOnlyInitializationTranscript<12>&,
    const TrainingModel<12>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<13>(
    const BaseListReader<13>&,
    const RootOnlyInitializationTranscript<13>&,
    const TrainingModel<13>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<14>(
    const BaseListReader<14>&,
    const RootOnlyInitializationTranscript<14>&,
    const TrainingModel<14>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<15>(
    const BaseListReader<15>&,
    const RootOnlyInitializationTranscript<15>&,
    const TrainingModel<15>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<16>(
    const BaseListReader<16>&,
    const RootOnlyInitializationTranscript<16>&,
    const TrainingModel<16>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<17>(
    const BaseListReader<17>&,
    const RootOnlyInitializationTranscript<17>&,
    const TrainingModel<17>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<18>(
    const BaseListReader<18>&,
    const RootOnlyInitializationTranscript<18>&,
    const TrainingModel<18>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<19>(
    const BaseListReader<19>&,
    const RootOnlyInitializationTranscript<19>&,
    const TrainingModel<19>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<20>(
    const BaseListReader<20>&,
    const RootOnlyInitializationTranscript<20>&,
    const TrainingModel<20>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<6>(
    const BaseListReader<6>&,
    const RootOnlyInitializationTranscript<6>&,
    const TrainingModel<6>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeInitializationAfterEdgeUpdate<10>(
    const BaseListReader<10>&,
    const RootOnlyInitializationTranscript<10>&,
    const TrainingModel<10>&,
    const RootToEdgeCodeMap&, const ParallelForExecutor&);

}  // namespace stlqpp
