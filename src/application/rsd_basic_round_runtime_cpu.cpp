#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "stlqpp/application/rsd_basic_round_runtime.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/basic/basic_projection.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/detail/basic_rsd_support.h"
#include "stlqpp/training/basic/icm_ils_cpu.h"
#include "stlqpp/training/basic/runtime_basic_executor.h"
#include "stlqpp/training/basic/support_least_squares.h"
#include "stlqpp/training/linkage/rsd/detail/linkage_rsd_kernel.h"
#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"
#include "stlqpp/training/linkage/rsd/linkage_codebook_algebra.h"
#include "stlqpp/training/linkage/rsd/linkage_projection.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

namespace stlqpp {
namespace {

template <int M>
StatusOr<ProductCodebooks<M>> CreateRootProductCodebooks(
    const RootCodebooks<M>& roots) {
  if (roots.shape().nlist != 256U) {
    return Status::InvalidArgument(
        "basic RSD requires a 256-way layer-zero codebook");
  }
  ProductCodebookView<M> view;
  view.shape = roots.shape();
  view.root0 = roots.root0();
  view.edge[0] = roots.root0();
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(M);
       ++layer) {
    view.root_small[layer - 1U] = roots.root_small(layer);
    view.edge[layer] = roots.root_small(layer);
  }
  return ProductCodebooks<M>::Create(view);
}

template <int M>
void CopyEncodingTile(const BasicEncoding<M>& source,
                      std::uint32_t destination_offset,
                      BasicEncoding<M>* destination) {
  const std::uint32_t count = source.sample_count();
  std::copy_n(
      source.root_codes().data(), count,
      detail::BasicEncodingAccess::Roots(destination) + destination_offset);
  std::copy_n(source.small_codes_row_major().data(),
              static_cast<std::size_t>(count) * (M - 1U),
              detail::BasicEncodingAccess::Codes(destination) +
                  static_cast<std::size_t>(destination_offset) * (M - 1U));
  std::copy_n(source.coefficients_row_major().data(),
              static_cast<std::size_t>(count) * M,
              detail::BasicEncodingAccess::Coefficients(destination) +
                  static_cast<std::size_t>(destination_offset) * M);
  std::copy_n(source.initial_squared_errors().data(), count,
              detail::BasicEncodingAccess::InitialErrors(destination) +
                  destination_offset);
  std::copy_n(
      source.squared_errors().data(), count,
      detail::BasicEncodingAccess::Errors(destination) + destination_offset);
}

template <typename Scalar>
struct ConvertContext final {
  ColumnMajorMatrixView<Scalar> input;
  float* output = nullptr;
};

template <typename Scalar>
void ConvertVector(void* opaque, std::uint32_t, std::uint32_t vector) noexcept {
  auto& context = *static_cast<ConvertContext<Scalar>*>(opaque);
  const Scalar* input = context.input.column(vector);
  float* output =
      context.output + static_cast<std::size_t>(vector) * context.input.rows;
  for (std::uint32_t row = 0U; row < context.input.rows; ++row) {
    output[row] = static_cast<float>(input[row]);
  }
}

template <int M>
struct RsdFitContext final {
  const LinkageCodebookAlgebra<M>* algebra = nullptr;
  const LinkageProjection<M>* projection = nullptr;
  ArrayView<const std::uint32_t> fixed_roots;
  std::uint32_t maximum_rank = 2U;
  LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation;
  BasicEncoding<M>* output = nullptr;
  std::atomic<std::uint32_t>* failures = nullptr;
};

template <int M>
struct LargeRootRsdFitContext final {
  const BasicCodebookAlgebra<M>* algebra = nullptr;
  const BasicProjection<M>* projection = nullptr;
  ArrayView<const std::uint32_t> fixed_roots;
  ArrayView<const std::uint32_t> selected_roots;
  ArrayView<const float> selected_root_dots;
  std::uint32_t maximum_rank = 2U;
  LeastSquaresCostMode cost_mode = LeastSquaresCostMode::kNormalEquation;
  BasicEncoding<M>* output = nullptr;
  std::atomic<std::uint32_t>* failures = nullptr;
};

struct RootSelectionBlock final {
  const float* scores;
  const float* root_norms;
  const RuntimeBasicProjection* projection;
  std::uint32_t begin;
  std::uint32_t count;
  float* errors;
  std::uint32_t* roots;
  float* dots;
};

void SelectRootSample(void* opaque, std::uint32_t, std::uint32_t sample) noexcept {
  const auto& block = *static_cast<RootSelectionBlock*>(opaque);
  const float* scores = block.scores + static_cast<std::size_t>(sample) * block.count;
  const float sample_norm = block.projection->squared_norm(sample);
  for (std::uint32_t local = 0; local < block.count; ++local) {
    const std::uint32_t root = block.begin + local;
    const float norm2 = block.root_norms[root];
    if (!(norm2 > 0.0F)) continue;
    const float dot = scores[local];
    const float error = std::fmax(0.0F, sample_norm - dot * (dot / norm2));
    if (error < block.errors[sample]) {
      block.errors[sample] = error;
      block.roots[sample] = root;
      block.dots[sample] = dot;
    }
  }
}

template <int M>
void FitLargeRootRsdSample(void* opaque, std::uint32_t,
                           std::uint32_t sample) noexcept {
  auto& context = *static_cast<LargeRootRsdFitContext<M>*>(opaque);
  const auto& algebra = *context.algebra;
  const float* vector = context.projection->samples().column(sample);
  const float* small_dots = context.projection->small_dots(sample);
  const float sample_norm2 = context.projection->squared_norm(sample);
  const auto root_dot_for = [&](std::uint32_t code) noexcept {
    const float* atom = algebra.root0().column(code);
    float dot = 0.0F;
    for (std::uint32_t axis = 0U; axis < algebra.shape().dimension; ++axis) {
      dot += vector[axis] * atom[axis];
    }
    return dot;
  };
  std::uint32_t root = 0U;
  float root_dot = 0.0F;
  if (!context.fixed_roots.empty()) {
    root = context.fixed_roots[sample];
    root_dot = root_dot_for(root);
  } else {
    root = context.selected_roots[sample];
    root_dot = context.selected_root_dots[sample];
  }
  std::array<std::uint8_t, M - 1> codes{};
  std::array<float, M> coefficients{};
  float error;
  if (!detail::FitBasicRsdSupport<M>(algebra, root, root_dot, small_dots,
                                     sample_norm2, context.maximum_rank,
                                     context.cost_mode, codes.data(),
                                     coefficients.data(), &error)) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  detail::BasicEncodingAccess::Store<M>(context.output, sample, codes,
                                        coefficients, root, error);
}

template <int M>
std::uint8_t SelectTopOneRoot(const LinkageCodebookAlgebra<M>& algebra,
                              const float* atom_dots,
                              float squared_norm) noexcept {
  float best_error = std::numeric_limits<float>::infinity();
  std::uint32_t best_code = 0U;
  for (std::uint32_t code = 0U; code < 256U; ++code) {
    const float norm2 = algebra.squared_norm(code);
    if (!(norm2 > 0.0F)) continue;
    const float dot = atom_dots[code];
    const float coefficient = dot / norm2;
    const float error = std::fmax(0.0F, squared_norm - dot * coefficient);
    if (error < best_error) {
      best_error = error;
      best_code = code;
    }
  }
  return static_cast<std::uint8_t>(best_code);
}

template <int M>
void FitRsdSample(void* opaque, std::uint32_t, std::uint32_t sample) noexcept {
  auto& context = *static_cast<RsdFitContext<M>*>(opaque);
  LinkageSolution<M> solution;
  const float* atom_dots = context.projection->atom_dots(sample);
  const float squared_norm = context.projection->squared_norm(sample);
  bool solved = false;
  if (!context.fixed_roots.empty()) {
    solved = detail::FitRsdOneFixedLayerZero(
        *context.algebra, atom_dots, squared_norm,
        static_cast<std::uint8_t>(context.fixed_roots[sample]), &solution,
        context.maximum_rank, context.cost_mode);
  } else {
    solved = detail::FitRsdOneFixedLayerZero(
        *context.algebra, atom_dots, squared_norm,
        SelectTopOneRoot(*context.algebra, atom_dots, squared_norm), &solution,
        context.maximum_rank, context.cost_mode);
  }
  if (!solved) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  std::array<std::uint8_t, M - 1> small_codes{};
  for (std::uint32_t layer = 1U; layer < M; ++layer) {
    small_codes[layer - 1U] = solution.support.codes[layer];
  }
  detail::BasicEncodingAccess::Store<M>(
      context.output, sample, small_codes, solution.coefficients,
      solution.support.codes[0], solution.squared_error);
}

template <int M>
class CpuRsdBasicEncoder final {
 public:
  CpuRsdBasicEncoder(const BasicCodebookAlgebra<M>& basic_algebra,
                     const LinkageCodebookAlgebra<M>* rsd_algebra,
                     const F32GemmBackend& gemm,
                     const ParallelForExecutor& parallel,
                     std::uint32_t maximum_rank) noexcept
      : basic_algebra_(basic_algebra),
        rsd_algebra_(rsd_algebra),
        gemm_(gemm),
        parallel_(parallel),
        maximum_rank_(maximum_rank) {}

  Status EncodeInto(ColumnMajorMatrixF32View samples,
                    ArrayView<const std::uint32_t> fixed_roots,
                    ArrayView<const std::uint64_t> sample_ids,
                    BasicSchedule schedule, std::uint32_t seed,
                    std::uint64_t global_sample_offset,
                    LinkageProjection<M>& rsd_projection,
                    BasicProjection<M>& basic_projection,
                    std::vector<float>& root_block_scores,
                    std::vector<float>& best_root_errors,
                    std::vector<std::uint32_t>& selected_roots,
                    std::vector<float>& selected_root_dots,
                    BasicEncoding<M>& output) const {
    assert(fixed_roots.empty() || (fixed_roots.size() == samples.columns &&
                                   sample_ids.size() == samples.columns));
    for (const std::uint32_t root : fixed_roots) {
      if (root >= basic_algebra_.shape().nlist) {
        return Status::InvalidArgument(
            "CPU basic RSD fixed root exceeds the root codebook");
      }
    }
    if (rsd_algebra_ == nullptr) {
      if (const Status status = basic_projection.Recompute(
              samples, basic_algebra_, gemm_, parallel_);
          !status.ok()) {
        return status;
      }
      if (fixed_roots.empty()) {
        constexpr std::uint32_t kRootBlock = 1024U;
        root_block_scores.resize(static_cast<std::size_t>(samples.columns) *
                                 kRootBlock);
        selected_roots.resize(samples.columns);
        selected_root_dots.resize(samples.columns);
        best_root_errors.assign(samples.columns,
                                std::numeric_limits<float>::infinity());
        for (std::uint32_t begin = 0U; begin < basic_algebra_.shape().nlist;
             begin += kRootBlock) {
          const std::uint32_t count =
              std::min(kRootBlock, basic_algebra_.shape().nlist - begin);
          const float* root_block = basic_algebra_.root0().column(begin);
          if (const Status status = gemm_.MultiplyTransposedLeft({
                  .left = {root_block, samples.rows, count},
                  .right = samples.values,
                  .query_count = samples.columns,
                  .right_leading_dimension = samples.rows,
                  .output = root_block_scores.data(),
                  .output_leading_dimension = count,
              });
              !status.ok())
            return status;
          RootSelectionBlock selection{
              root_block_scores.data(), basic_algebra_.root_norms2().data(),
              &basic_projection, begin, count,
              best_root_errors.data(), selected_roots.data(),
              selected_root_dots.data()};
          parallel_.Run(samples.columns, &selection, SelectRootSample);
        }
      }
      detail::BasicEncodingAccess::Resize<M>(&output, samples.columns);
      std::atomic<std::uint32_t> failures{0U};
      LargeRootRsdFitContext<M> context{
          .algebra = &basic_algebra_,
          .projection = &basic_projection,
          .fixed_roots = fixed_roots,
          .selected_roots = selected_roots,
          .selected_root_dots = selected_root_dots,
          .maximum_rank = maximum_rank_,
          .cost_mode = schedule.least_squares_cost_mode(),
          .output = &output,
          .failures = &failures,
      };
      parallel_.Run(samples.columns, &context, FitLargeRootRsdSample<M>);
      if (failures.load(std::memory_order_relaxed) != 0U) {
        return Status::FailedPrecondition(
            "CPU basic RSD contains a singular candidate set");
      }
      CpuIcmIlsOptimizer<M> optimizer(basic_algebra_, parallel_);
      return fixed_roots.empty()
                 ? optimizer.Optimize(basic_projection, schedule, seed,
                                      global_sample_offset, &output)
                 : optimizer.OptimizeWithSampleIds(basic_projection, schedule,
                                                   seed, sample_ids, &output);
    }
    if (const Status status =
            rsd_projection.Recompute(samples, *rsd_algebra_, gemm_, parallel_);
        !status.ok()) {
      return status;
    }
    detail::BasicEncodingAccess::Resize<M>(&output, samples.columns);
    std::atomic<std::uint32_t> failures{0U};
    RsdFitContext<M> context{
        .algebra = rsd_algebra_,
        .projection = &rsd_projection,
        .fixed_roots = fixed_roots,
        .maximum_rank = maximum_rank_,
        .cost_mode = schedule.least_squares_cost_mode(),
        .output = &output,
        .failures = &failures,
    };
    parallel_.Run(samples.columns, &context, FitRsdSample<M>);
    if (failures.load(std::memory_order_relaxed) != 0U) {
      return Status::FailedPrecondition(
          "CPU basic RSD contains a singular candidate set");
    }
    if (const Status status = basic_projection.BindFullAtomProjection(
            samples, rsd_projection.squared_norms(),
            rsd_projection.all_atom_dots(), M * 256U, 256U);
        !status.ok()) {
      return status;
    }
    CpuIcmIlsOptimizer<M> optimizer(basic_algebra_, parallel_);
    return fixed_roots.empty()
               ? optimizer.Optimize(basic_projection, schedule, seed,
                                    global_sample_offset, &output)
               : optimizer.OptimizeWithSampleIds(basic_projection, schedule,
                                                 seed, sample_ids, &output);
  }

 private:
  const BasicCodebookAlgebra<M>& basic_algebra_;
  const LinkageCodebookAlgebra<M>* rsd_algebra_;
  const F32GemmBackend& gemm_;
  const ParallelForExecutor& parallel_;
  std::uint32_t maximum_rank_ = 2U;
};

template <int M, typename Scalar>
class CpuRsdBasicExecutor final : public BasicBlockExecutor<Scalar> {
 public:
  static constexpr std::uint32_t kTileRows = 8192U;

  static StatusOr<CpuRsdBasicExecutor> Create(
      ColumnMajorMatrixF32View rotation,
      const BasicCodebookAlgebra<M>& basic_algebra,
      const LinkageCodebookAlgebra<M>* rsd_algebra, const F32GemmBackend& gemm,
      const ParallelForExecutor& parallel, std::uint32_t maximum_rank) {
    static_assert(std::is_same_v<Scalar, std::uint8_t> ||
                  std::is_same_v<Scalar, float>);
    if (!IsSupportedRsdRank(maximum_rank) || rotation.values == nullptr ||
        rotation.rows != basic_algebra.shape().dimension ||
        rotation.columns != rotation.rows) {
      return Status::InvalidArgument(
          "CPU basic RSD requires rank 1, 2 or 3 and a matching rotation");
    }
    std::vector<float> transposed(static_cast<std::size_t>(rotation.rows) *
                                  rotation.columns);
    for (std::uint32_t column = 0U; column < rotation.columns; ++column) {
      for (std::uint32_t row = 0U; row < rotation.rows; ++row) {
        const float value = rotation.column(column)[row];
        if (!std::isfinite(value)) {
          return Status::InvalidArgument(
              "CPU basic RSD rotation contains a non-finite value");
        }
        transposed[static_cast<std::size_t>(row) * rotation.rows + column] =
            value;
      }
    }
    return CpuRsdBasicExecutor(std::move(transposed), basic_algebra,
                               rsd_algebra, gemm, parallel, maximum_rank);
  }

  StatusOr<BasicEncodingBatch> Encode(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override {
    if (!Valid(samples)) return InvalidInput();
    BasicEncoding<M> output =
        detail::BasicEncodingAccess::Create<M>(samples.columns);
    if (const Status status = EncodeTiles(samples, {}, schedule, seed,
                                         global_sample_offset, output, nullptr);
        !status.ok()) return status;
    return detail::BasicEncodingAccess::Release(std::move(output));
  }

  StatusOr<RetainedModelSpaceBasicEncoding> EncodeRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override {
    if (!Valid(samples)) return InvalidInput();
    return EncodeRetained(samples, {}, schedule, seed, global_sample_offset);
  }

  StatusOr<RetainedModelSpaceBasicEncoding> EncodeFixedRootsRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override {
    if (!Valid(samples) || fixed_roots.size() != samples.columns) {
      return InvalidInput();
    }
    return EncodeRetained(samples, fixed_roots, schedule, seed,
                          global_sample_offset);
  }

 private:
  StatusOr<RetainedModelSpaceBasicEncoding> EncodeRetained(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) {
    std::vector<float> exact_model_space(
        static_cast<std::size_t>(samples.rows) * samples.columns);
    BasicEncoding<M> encoding =
        detail::BasicEncodingAccess::Create<M>(samples.columns);
    if (const Status status = EncodeTiles(
            samples, fixed_roots, schedule, seed, global_sample_offset,
            encoding, &exact_model_space);
        !status.ok()) return status;
    return RetainedModelSpaceBasicEncoding{
        .dimension = samples.rows,
        .exact_model_space = std::move(exact_model_space),
        .encoding = detail::BasicEncodingAccess::Release(std::move(encoding)),
    };
  }

  Status EncodeTiles(ColumnMajorMatrixView<Scalar> samples,
                     ArrayView<const std::uint32_t> fixed_roots,
                     BasicSchedule schedule, std::uint32_t seed,
                     std::uint64_t global_sample_offset,
                     BasicEncoding<M>& output, std::vector<float>* exact) {
    for (std::uint32_t offset = 0U; offset < samples.columns;) {
      const std::uint32_t count = std::min(kTileRows, samples.columns - offset);
      const ColumnMajorMatrixView<Scalar> tile{
          samples.column(offset), samples.rows, count};
      if (const Status status = Rotate(tile); !status.ok()) return status;
      ArrayView<const std::uint32_t> tile_roots;
      ArrayView<const std::uint64_t> tile_ids;
      if (!fixed_roots.empty()) {
        tile_roots = {fixed_roots.data() + offset, count};
        sample_ids_.resize(count);
        for (std::uint32_t row = 0U; row < count; ++row) {
          sample_ids_[row] = global_sample_offset + offset + row;
        }
        tile_ids = sample_ids_;
      }
      EnsureEncoding(count);
      if (const Status status = Encoder().EncodeInto(
              {rotated_.data(), samples.rows, count}, tile_roots, tile_ids,
              schedule, seed, global_sample_offset + offset, rsd_projection_,
              basic_projection_, root_block_scores_, best_root_errors_,
              selected_roots_, selected_root_dots_, *encoding_workspace_);
          !status.ok()) return status;
      if (exact) {
        std::copy_n(rotated_.data(),
                    static_cast<std::size_t>(samples.rows) * count,
                    exact->data() +
                        static_cast<std::size_t>(offset) * samples.rows);
      }
      CopyEncodingTile(*encoding_workspace_, offset, &output);
      offset += count;
    }
    return Status::Ok();
  }

  CpuRsdBasicExecutor(std::vector<float> rotation_transposed,
                      const BasicCodebookAlgebra<M>& basic_algebra,
                      const LinkageCodebookAlgebra<M>* rsd_algebra,
                      const F32GemmBackend& gemm,
                      const ParallelForExecutor& parallel,
                      std::uint32_t maximum_rank) noexcept
      : rotation_transposed_(std::move(rotation_transposed)),
        basic_algebra_(&basic_algebra),
        rsd_algebra_(rsd_algebra),
        gemm_(&gemm),
        parallel_(&parallel),
        maximum_rank_(maximum_rank) {}

  bool Valid(ColumnMajorMatrixView<Scalar> samples) const noexcept {
    return samples.values != nullptr && samples.columns != 0U &&
           samples.rows == basic_algebra_->shape().dimension;
  }

  static Status InvalidInput() {
    return Status::InvalidArgument(
        "CPU basic RSD input has inconsistent model-space extents");
  }

  void EnsureEncoding(std::uint32_t rows) {
    if (!encoding_workspace_.has_value()) {
      encoding_workspace_.emplace(detail::BasicEncodingAccess::Create<M>(rows));
    }
  }

  CpuRsdBasicEncoder<M> Encoder() const noexcept {
    return CpuRsdBasicEncoder<M>(*basic_algebra_, rsd_algebra_, *gemm_,
                                 *parallel_, maximum_rank_);
  }

  Status Rotate(ColumnMajorMatrixView<Scalar> samples) {
    const std::size_t value_count =
        static_cast<std::size_t>(samples.rows) * samples.columns;
    const float* input = nullptr;
    if constexpr (std::is_same_v<Scalar, float>) {
      input = samples.values;
    } else {
      converted_.resize(value_count);
      ConvertContext<Scalar> context{
          .input = samples,
          .output = converted_.data(),
      };
      parallel_->Run(samples.columns, &context, ConvertVector<Scalar>);
      input = converted_.data();
    }
    rotated_.resize(value_count);
    return gemm_->MultiplyTransposedLeft({
        .left = {rotation_transposed_.data(), samples.rows, samples.rows},
        .right = input,
        .query_count = samples.columns,
        .right_leading_dimension = samples.rows,
        .output = rotated_.data(),
        .output_leading_dimension = samples.rows,
    });
  }

  std::vector<float> rotation_transposed_;
  const BasicCodebookAlgebra<M>* basic_algebra_ = nullptr;
  const LinkageCodebookAlgebra<M>* rsd_algebra_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
  std::uint32_t maximum_rank_ = 2U;
  std::vector<float> converted_;
  std::vector<float> rotated_;
  std::vector<std::uint64_t> sample_ids_;
  LinkageProjection<M> rsd_projection_;
  BasicProjection<M> basic_projection_;
  std::vector<float> root_block_scores_;
  std::vector<float> best_root_errors_;
  std::vector<std::uint32_t> selected_roots_;
  std::vector<float> selected_root_dots_;
  std::optional<BasicEncoding<M>> encoding_workspace_;
};

template <int M, typename Scalar>
class CpuRsdBasicRoundRuntime final : public BasicRoundRuntime<Scalar> {
 public:
  CpuRsdBasicRoundRuntime(
      std::shared_ptr<BasicCodebookAlgebra<M>> basic_algebra,
      std::shared_ptr<LinkageCodebookAlgebra<M>> rsd_algebra,
      std::unique_ptr<CpuRsdBasicExecutor<M, Scalar>> primary,
      std::uint32_t maximum_rank) noexcept
      : basic_algebra_(std::move(basic_algebra)),
        rsd_algebra_(std::move(rsd_algebra)),
        primary_(std::move(primary)),
        maximum_rank_(maximum_rank) {}

  BasicBlockExecutor<Scalar>& primary() noexcept override { return *primary_; }
  BasicRuntimeExecutionInfo execution_info() const noexcept override {
    return {
        .rsd_rank = maximum_rank_,
        .primary_backend = BasicPrimaryBackend::kCpu,
        .variable_root_tile_capacity_rows =
            CpuRsdBasicExecutor<M, Scalar>::kTileRows,
        .model_projections_on_device = false,
        .input_on_device = false,
    };
  }

 private:
  std::shared_ptr<BasicCodebookAlgebra<M>> basic_algebra_;
  std::shared_ptr<LinkageCodebookAlgebra<M>> rsd_algebra_;
  std::unique_ptr<CpuRsdBasicExecutor<M, Scalar>> primary_;
  std::uint32_t maximum_rank_ = 2U;
};

}  // namespace

template <int M, typename Scalar>
StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
CpuRsdBasicRoundRuntimeFactory<M, Scalar>::Create(
    const RootCodebooks<M>& codebooks, ColumnMajorMatrixF32View rotation) {
  assert(gemm_ != nullptr && parallel_ != nullptr);
  if (!IsSupportedRsdRank(maximum_rank_)) {
    return Status::InvalidArgument(
        "CPU basic RSD factory requires rank 1, 2 or 3");
  }
  if constexpr (M == 0) {
    return CreateRuntimeCpuRsdRound<Scalar>(codebooks.runtime_view(), rotation,
                                            *gemm_, *parallel_, maximum_rank_);
  } else {
    auto basic = BasicCodebookAlgebra<M>::Create(codebooks, *gemm_, parallel_);
    if (!basic.ok()) return basic.status();
    auto basic_owner =
        std::make_shared<BasicCodebookAlgebra<M>>(std::move(basic).value());
    std::shared_ptr<LinkageCodebookAlgebra<M>> rsd_owner;
    if (codebooks.shape().nlist == 256U) {
      auto product = CreateRootProductCodebooks(codebooks);
      if (!product.ok()) return product.status();
      auto rsd = LinkageCodebookAlgebra<M>::Create(product.value(), *gemm_);
      if (!rsd.ok()) return rsd.status();
      rsd_owner =
          std::make_shared<LinkageCodebookAlgebra<M>>(std::move(rsd).value());
    }
    auto executor = CpuRsdBasicExecutor<M, Scalar>::Create(
        rotation, *basic_owner, rsd_owner.get(), *gemm_, *parallel_,
        maximum_rank_);
    if (!executor.ok()) return executor.status();
    std::unique_ptr<BasicRoundRuntime<Scalar>> output =
        std::make_unique<CpuRsdBasicRoundRuntime<M, Scalar>>(
            std::move(basic_owner), std::move(rsd_owner),
            std::make_unique<CpuRsdBasicExecutor<M, Scalar>>(
                std::move(executor).value()),
            maximum_rank_);
    return output;
  }
}

template class CpuRsdBasicRoundRuntimeFactory<0, std::uint8_t>;
template class CpuRsdBasicRoundRuntimeFactory<0, float>;
template class CpuRsdBasicRoundRuntimeFactory<5, std::uint8_t>;
template class CpuRsdBasicRoundRuntimeFactory<12, std::uint8_t>;
template class CpuRsdBasicRoundRuntimeFactory<6, std::uint8_t>;
template class CpuRsdBasicRoundRuntimeFactory<10, std::uint8_t>;
template class CpuRsdBasicRoundRuntimeFactory<5, float>;
template class CpuRsdBasicRoundRuntimeFactory<12, float>;
template class CpuRsdBasicRoundRuntimeFactory<6, float>;
template class CpuRsdBasicRoundRuntimeFactory<10, float>;

}  // namespace stlqpp
