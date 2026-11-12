#include "stlqpp/training/analysis/round_linkage_analysis.h"
#include "stlqpp/training/analysis/causal_covariance_factors.h"

#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <limits>
#include <vector>

namespace stlqpp {

// Only AnalyzeCid may use the metric-validated reconstruction path.
struct RoundLinkageAnalysisAccess final {
  static Status AccumulateValidatedReconstruction(
      RotationCrossCovariance& covariance,
      ColumnMajorMatrixF32View unrotated_exact,
      ColumnMajorMatrixF32View model_reconstructions,
      const F32CrossCovarianceBackend& backend) {
    return covariance.AccumulateValidatedReconstruction(
        unrotated_exact, model_reconstructions, backend);
  }
  static Status ValidatePair(RotationCrossCovariance& covariance,
      ColumnMajorMatrixF32View source, ColumnMajorMatrixF32View reconstructed,
      bool validated_reconstruction, float* maximum_source_abs) {
    return covariance.ValidateReconstructionPair(source, reconstructed,
        validated_reconstruction, maximum_source_abs);
  }
  static Status AccumulateFactors(RotationCrossCovariance& covariance,
      ColumnMajorMatrixF32View root0, ColumnMajorMatrixF32View small,
      ColumnMajorMatrixF32View edge, const float* factors,
      std::uint32_t rows, const F32CrossCovarianceBackend& backend) {
    const auto d = covariance.dimension_;
    for (auto atoms : {root0, small, edge}) {
      if (atoms.columns == 0U) continue;
      if (const auto status = backend.Accumulate({atoms,
              {factors, d, atoms.columns}, covariance.values_.data(), d});
          !status.ok()) return status;
      factors += static_cast<std::size_t>(d) * atoms.columns;
    }
    covariance.sample_count_ += rows;
    return Status::Ok();
  }

};

namespace {

struct CidAnalysis final {
  Status status = Status::Ok();
  InitializationPostEdgeAnalysis summary;
  // Kept only for one bounded CID window so the final FP64 addition order
  // remains exactly the original CID/causal-row order.
  std::vector<double> row_errors;
};

template <int M>
struct CidScratch final {
  std::vector<float> source;
  std::vector<float> exact;
  RoundUpdateTranscriptCluster<M> staged;
  std::vector<float> virtual_reconstructions;
  std::vector<float> real_reconstructions;
  std::vector<std::uint32_t> causal_order;
};

template <int M>
Status AnalyzeCid(
    const BaseListReader<M>& lists,
    const RoundUpdateTranscript<M>& transcript,
    const TrainingModel<M>& model,
    ColumnMajorMatrixF32View rotation,
    const F32GemmBackend& gemm,
    const F32CrossCovarianceBackend* covariance_backend,
    RotationCrossCovariance* covariance,
    bool compute_metrics,
    std::uint32_t cid,
    CidScratch<M>* scratch,
    CidAnalysis* result) {
  const ProductCodebooks<M>& codebooks = model.codebooks();
  const ModelShape& shape = codebooks.shape();
  InitializationPostEdgeAnalysis output;
  auto& source = scratch->source;
  auto& exact = scratch->exact;
  auto& staged = scratch->staged;
  auto& virtual_reconstructions = scratch->virtual_reconstructions;
  auto& real_reconstructions = scratch->real_reconstructions;
  auto& causal_order = scratch->causal_order;
  result->row_errors.clear();
  if (const Status status = transcript.LoadCluster(cid, &staged);
      !status.ok()) return status;
  const std::uint32_t rows = staged.real_count();
  if (compute_metrics) result->row_errors.reserve(rows);
  if (const Status status = lists.GatherCidByStableIds(
          cid, staged.real_sample_ids, &source, nullptr, nullptr);
      !status.ok()) {
    return status;
  }
  if (compute_metrics) {
    exact.resize(static_cast<std::size_t>(shape.dimension) * rows);
  }
  if (compute_metrics && rows != 0U) {
    if (const Status status = gemm.MultiplyLeft({
            .left = rotation,
            .right = source.data(),
            .query_count = rows,
            .right_leading_dimension = shape.dimension,
            .output = exact.data(),
            .output_leading_dimension = shape.dimension,
        }); !status.ok()) {
      return status;
    }
  }

  const std::uint32_t virtual_count = staged.virtual_count();
  virtual_reconstructions.assign(
      static_cast<std::size_t>(shape.dimension) * virtual_count, 0.0F);
  for (std::uint32_t virtual_id = 0U;
       virtual_id < virtual_count; ++virtual_id) {
    float* reconstruction = virtual_reconstructions.data() +
        static_cast<std::size_t>(virtual_id) * shape.dimension;
    const float* coefficients =
        staged.virtual_coefficients.data() +
        static_cast<std::size_t>(virtual_id) * shape.m;
    for (std::uint32_t layer = 0U;
         layer < shape.m; ++layer) {
      const float* atom = layer == 0U
          ? codebooks.root0().column(cid)
          : codebooks.root_small(layer).column(
                staged.virtual_small_codes[
                    static_cast<std::size_t>(virtual_id) * (shape.m - 1U) +
                    layer - 1U]);
      for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
        reconstruction[axis] += coefficients[layer] * atom[axis];
      }
    }
  }

  real_reconstructions.resize(
      static_cast<std::size_t>(shape.dimension) * rows);
  causal_order.resize(rows);
  std::iota(causal_order.begin(), causal_order.end(), 0U);
  std::stable_sort(
      causal_order.begin(), causal_order.end(),
      [&](std::uint32_t left, std::uint32_t right) {
        return staged.depths[left] < staged.depths[right];
      });
  for (const std::uint32_t row : causal_order) {
    float* reconstruction = real_reconstructions.data() +
        static_cast<std::size_t>(row) * shape.dimension;
    const LinkageNodeRole role = staged.roles[row];
    const LinkageCodeTuple<M>& support = staged.supports[row];
    const float* coefficients = staged.coefficients.data() +
        static_cast<std::size_t>(row) * shape.m;
    if (role == LinkageNodeRole::kRoot) {
      assert(support.first == cid && !staged.parents[row].has_value() &&
             staged.depths[row] == 0U);
      std::fill_n(reconstruction, shape.dimension, 0.0F);
      for (std::uint32_t layer = 0U;
           layer < shape.m; ++layer) {
        const float* atom = layer == 0U
            ? codebooks.root0().column(support.first)
            : codebooks.root_small(layer).column(
                  support.trailing[layer - 1U]);
        for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
          reconstruction[axis] += coefficients[layer] * atom[axis];
        }
      }
    } else {
      const auto& parent = staged.parents[row];
      assert(role == LinkageNodeRole::kLinked && parent.has_value() &&
             support.first < 256U && staged.depths[row] != 0U);
      const float* parent_reconstruction = nullptr;
      if (parent->kind == LinkageParentKind::kVirtual) {
        assert(parent->local_id < virtual_count && staged.depths[row] == 1U);
        parent_reconstruction = virtual_reconstructions.data() +
            static_cast<std::size_t>(parent->local_id) * shape.dimension;
      } else {
        assert(parent->local_id < rows &&
               staged.depths[parent->local_id] + 1U == staged.depths[row]);
        parent_reconstruction = real_reconstructions.data() +
            static_cast<std::size_t>(parent->local_id) * shape.dimension;
      }
      std::copy_n(parent_reconstruction, shape.dimension, reconstruction);
      for (std::uint32_t layer = 0U;
           layer < shape.m; ++layer) {
        const std::uint32_t code = layer == 0U
            ? support.first : support.trailing[layer - 1U];
        const float* atom = codebooks.edge(layer).column(code);
        for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
          reconstruction[axis] += coefficients[layer] * atom[axis];
        }
      }
      ++output.linked_real_rows;
    }

    if (compute_metrics) {
      double squared_error = 0.0;
      const float* expected = exact.data() +
          static_cast<std::size_t>(row) * shape.dimension;
      for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
        const double difference =
            static_cast<double>(expected[axis]) - reconstruction[axis];
        squared_error += difference * difference;
      }
      if (!std::isfinite(squared_error)) {
        return Status::DataLoss(
            "round linkage analysis produced an invalid error");
      }
      result->row_errors.push_back(squared_error);
    }
    output.depth_sum += staged.depths[row];
    output.maximum_depth = std::max(
        output.maximum_depth, staged.depths[row]);
    ++output.real_rows;
  }
  ++output.clusters;
  if (covariance_backend != nullptr && rows != 0U) {
    // The source columns were gathered in transcript order, matching the
    // reconstruction columns without another native-list read or ID sort.
    const ColumnMajorMatrixF32View unrotated = {
        source.data(), shape.dimension, rows};
    const ColumnMajorMatrixF32View reconstructed = {
        real_reconstructions.data(), shape.dimension, rows};
    const bool has_linkage = std::find(staged.roles.begin(), staged.roles.end(),
        LinkageNodeRole::kLinked) != staged.roles.end();
    if (detail::UseCausalCovarianceFactors(shape.dimension, shape.m, rows,
                                          virtual_count, has_linkage)) {
      float maximum_source_abs = 0.0F;
      if (const auto status = RoundLinkageAnalysisAccess::ValidatePair(
              *covariance, unrotated, reconstructed, compute_metrics,
              &maximum_source_abs); !status.ok()) return status;
      float maximum_coefficient = 0.0F;
      for (const auto value : staged.coefficients)
        maximum_coefficient = std::max(maximum_coefficient, std::fabs(value));
      for (const auto value : staged.virtual_coefficients)
        maximum_coefficient = std::max(maximum_coefficient, std::fabs(value));
      const double nodes = static_cast<double>(rows) + virtual_count;
      const double limit = std::numeric_limits<float>::max() / 4.0;
      // Preserve the dense path for finite but extreme input that could
      // overflow an intermediate subtree/factor before the final product.
      if (maximum_source_abs * static_cast<double>(rows) < limit &&
          maximum_source_abs * static_cast<double>(maximum_coefficient) * nodes * nodes < limit) {
        const auto small = codebooks.root_small_flat();
        const auto edge = has_linkage ? codebooks.edge_flat() : ColumnMajorMatrixF32View{};
        const auto factor_columns = 1U + small.columns + edge.columns;
        // The workload rule guarantees factors fit in the already allocated
        // real reconstruction buffer. Source and virtual buffers are reused.
        assert(factor_columns <= rows);
        real_reconstructions.resize(static_cast<std::size_t>(shape.dimension) * factor_columns);
        detail::BuildCausalCovarianceFactors<M>(staged, causal_order, shape.dimension,
            source, virtual_reconstructions, real_reconstructions);
        if (const auto status = RoundLinkageAnalysisAccess::AccumulateFactors(
                *covariance, {codebooks.root0().column(cid), shape.dimension, 1U},
                small, edge, real_reconstructions.data(), rows, *covariance_backend);
            !status.ok()) return status;
        result->summary = output;
        return Status::Ok();
      }
    }
    const Status status = compute_metrics
        ? RoundLinkageAnalysisAccess::AccumulateValidatedReconstruction(
              *covariance, unrotated, reconstructed, *covariance_backend)
        : covariance->Accumulate(
              unrotated, reconstructed, *covariance_backend);
    if (!status.ok()) return status;
  }
  result->summary = output;
  return Status::Ok();
}

}  // namespace

template <int M>
StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage(
    const std::filesystem::path& list_directory,
    const RoundUpdateTranscript<M>& transcript,
    const TrainingModel<M>& model,
    const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel,
    const F32CrossCovarianceBackend* covariance_backend,
    std::optional<RotationCrossCovariance>* covariance_output,
    bool compute_metrics,
    std::uint32_t covariance_partition_count) {
  if ((covariance_backend == nullptr) != (covariance_output == nullptr)) {
    return Status::InvalidArgument(
        "round linkage analysis covariance inputs are incomplete");
  }
  if (covariance_backend != nullptr && covariance_partition_count == 0U) {
    return Status::InvalidArgument(
        "round linkage covariance needs numerical partitions");
  }
  auto lists_result = BaseListReader<M>::Open(list_directory);
  if (!lists_result.ok()) return lists_result.status();
  const BaseListReader<M>& lists = lists_result.value();
  const ModelShape& shape = model.codebooks().shape();
  const BaseBuildShape& list_shape = lists.store().shape();
  if (transcript.dimension() != shape.dimension ||
      transcript.nlist() != shape.nlist ||
      list_shape.dimension != shape.dimension ||
      list_shape.nlist != shape.nlist) {
    return Status::InvalidArgument(
        "round linkage analysis inputs are incompatible");
  }
  const std::uint32_t worker_count = std::max(
      1U, std::min(parallel.worker_count(), shape.nlist));
  const std::uint32_t shard_count = covariance_backend == nullptr ? 0U
      : std::min(covariance_partition_count, shape.nlist);
  std::vector<CidScratch<M>> scratch(worker_count);
  std::vector<RotationCrossCovariance> covariances;
  if (covariance_backend != nullptr) {
    covariance_output->reset();
    covariances.reserve(shard_count);
    for (std::uint32_t shard = 0U; shard < shard_count; ++shard) {
      auto created = RotationCrossCovariance::Create(shape.dimension);
      if (!created.ok()) return created.status();
      covariances.push_back(std::move(created).value());
    }
  }
  InitializationPostEdgeAnalysis output;
  constexpr std::uint64_t kWindowRows = 65536U;
  const std::uint32_t max_window_cids = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          shape.nlist,
          static_cast<std::uint64_t>(
              covariance_backend != nullptr ? shard_count : worker_count) *
              4U));
  const auto offsets = lists.store().offsets();
  for (std::uint32_t begin = 0U; begin < shape.nlist;) {
    std::uint32_t end = begin;
    std::uint64_t window_rows = 0U;
    while (end < shape.nlist) {
      const std::uint64_t cid_rows = offsets[end + 1U] - offsets[end];
      if (end != begin &&
          (end - begin >= max_window_cids ||
           window_rows + cid_rows > kWindowRows)) {
        break;
      }
      window_rows += cid_rows;
      ++end;
    }
    std::vector<CidAnalysis> results(end - begin);
    struct Context final {
      const BaseListReader<M>* lists;
      const RoundUpdateTranscript<M>* transcript;
      const TrainingModel<M>* model;
      ColumnMajorMatrixF32View rotation;
      const F32GemmBackend* gemm;
      const F32CrossCovarianceBackend* covariance_backend;
      RotationCrossCovariance* covariances;
      CidScratch<M>* scratch;
      CidAnalysis* results;
      std::uint32_t begin;
      std::uint32_t count;
      std::uint32_t shard_count;
      bool static_shards;
      bool compute_metrics;
    } context = {
        &lists, &transcript, &model, model.rotation(), &gemm,
        covariance_backend, covariances.data(), scratch.data(),
        results.data(), begin, end - begin, shard_count,
        covariance_backend != nullptr, compute_metrics};
    parallel.Run(
        context.static_shards ? shard_count : context.count, &context,
        [](void* opaque, std::uint32_t worker,
           std::uint32_t item) noexcept {
          auto& state = *static_cast<Context*>(opaque);
          const std::uint32_t count = state.static_shards
              ? (state.count / state.shard_count) +
                    (item < state.count % state.shard_count ? 1U : 0U)
              : 1U;
          const std::uint32_t first = state.static_shards
              ? item * (state.count / state.shard_count) +
                    std::min(item, state.count % state.shard_count)
              : item;
          for (std::uint32_t index = first; index < first + count; ++index) {
            state.results[index].status = AnalyzeCid<M>(
                *state.lists, *state.transcript, *state.model,
                state.rotation, *state.gemm, state.covariance_backend,
                state.static_shards ? &state.covariances[item] : nullptr,
                state.compute_metrics, state.begin + index,
                &state.scratch[worker],
                &state.results[index]);
            if (!state.results[index].status.ok()) return;
          }
        });
    for (const CidAnalysis& result : results) {
      if (!result.status.ok()) return result.status;
      const auto& summary = result.summary;
      output.clusters += summary.clusters;
      output.real_rows += summary.real_rows;
      output.linked_real_rows += summary.linked_real_rows;
      output.depth_sum += summary.depth_sum;
      output.maximum_depth = std::max(
          output.maximum_depth, summary.maximum_depth);
      for (const double error : result.row_errors) {
        output.squared_error_sum += error;
        if (!output.has_squared_error) {
          output.minimum_squared_error = error;
          output.maximum_squared_error = error;
          output.has_squared_error = true;
        } else {
          output.minimum_squared_error = std::min(
              output.minimum_squared_error, error);
          output.maximum_squared_error = std::max(
              output.maximum_squared_error, error);
        }
      }
    }
    begin = end;
  }
  if (covariance_backend != nullptr) {
    for (std::uint32_t shard = 1U; shard < shard_count; ++shard) {
      if (const Status status = covariances[0].MergeFrom(covariances[shard]);
          !status.ok()) return status;
    }
    assert(covariances[0].sample_count() == output.real_rows);
    covariance_output->emplace(std::move(covariances[0]));
  }
  assert(output.clusters == shape.nlist &&
         output.real_rows == lists.store().total_vectors() &&
         (!compute_metrics || output.has_squared_error ||
          output.real_rows == 0U));
  if (output.real_rows == 0U) {
    return Status::DataLoss(
        "round linkage analysis did not cover the artifact");
  }
  return output;
}

template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<0>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<0>&,
    const TrainingModel<0>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<5>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<5>&,
    const TrainingModel<5>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<12>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<12>&,
    const TrainingModel<12>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<6>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<6>&,
    const TrainingModel<6>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);
template StatusOr<InitializationPostEdgeAnalysis>
AnalyzeRoundLinkage<10>(
    const std::filesystem::path&,
    const RoundUpdateTranscript<10>&,
    const TrainingModel<10>&,
    const F32GemmBackend&,
    const ParallelForExecutor&,
    const F32CrossCovarianceBackend*,
    std::optional<RotationCrossCovariance>*,
    bool,
    std::uint32_t);

}  // namespace stlqpp
