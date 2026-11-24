#include "stlqpp/training/basic/runtime_basic_executor.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace stlqpp {
namespace {
template <typename Scalar>
struct ConvertContext {
  ColumnMajorMatrixView<Scalar> input;
  float* output;
};
template <typename Scalar>
void Convert(void* opaque, std::uint32_t, std::uint32_t row) noexcept {
  auto& c = *static_cast<ConvertContext<Scalar>*>(opaque);
  for (std::uint32_t axis = 0; axis < c.input.rows; ++axis)
    c.output[static_cast<std::size_t>(row) * c.input.rows + axis] =
        static_cast<float>(c.input.column(row)[axis]);
}
}  // namespace

template <typename Scalar>
StatusOr<RuntimeCpuRsdBasicExecutor<Scalar>>
RuntimeCpuRsdBasicExecutor<Scalar>::Create(RuntimeRootCodebookView roots,
                                           ColumnMajorMatrixF32View rotation,
                                           const F32GemmBackend& gemm,
                                           const ParallelForExecutor& parallel,
                                           std::uint32_t rank) {
  if (rank < 1U || rank > 3U || rotation.columns != rotation.rows ||
      (rotation.rows && (!rotation.values || rotation.rows != roots.shape.dimension)))
    return Status::InvalidArgument(
        "runtime CPU RSD requires rank 1..3 and matching rotation");
  std::vector<float> transposed(static_cast<std::size_t>(rotation.rows) *
                                rotation.rows);
  for (std::uint32_t column = 0; column < rotation.columns; ++column)
    for (std::uint32_t row = 0; row < rotation.rows; ++row) {
      const float value = rotation.column(column)[row];
      if (!std::isfinite(value))
        return Status::InvalidArgument("non-finite basic rotation");
      transposed[static_cast<std::size_t>(row) * rotation.rows + column] =
          value;
    }
  auto encoder = RuntimeCpuRsdBasicEncoder::Create(roots, gemm, parallel);
  if (!encoder.ok()) return encoder.status();
  return RuntimeCpuRsdBasicExecutor(roots.shape, std::move(transposed),
                                    std::move(encoder).value(), gemm, parallel,
                                    rank);
}

template <typename Scalar>
Status RuntimeCpuRsdBasicExecutor<Scalar>::Prepare(
    ColumnMajorMatrixView<Scalar> samples, const float*& prepared) {
  const std::size_t count =
      static_cast<std::size_t>(samples.rows) * samples.columns;
  const float* input;
  if constexpr (std::is_same_v<Scalar, float>)
    input = samples.values;
  else {
    converted_.resize(count);
    ConvertContext<Scalar> context{samples, converted_.data()};
    parallel_->Run(samples.columns, &context, Convert<Scalar>);
    input = converted_.data();
  }
  if (rotation_.empty()) {
    prepared = input;
    return Status::Ok();
  }
  rotated_.resize(count);
  prepared = rotated_.data();
  return gemm_->MultiplyTransposedLeft(
      {.left = {rotation_.data(), samples.rows, samples.rows},
       .right = input,
       .query_count = samples.columns,
       .right_leading_dimension = samples.rows,
       .output = rotated_.data(),
       .output_leading_dimension = samples.rows});
}

template <typename Scalar>
Status RuntimeCpuRsdBasicExecutor<Scalar>::EncodeTiles(
    ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t offset, RuntimeBasicEncoding* output,
    std::vector<float>* exact, ArrayView<const std::uint32_t> roots) {
  if (!samples.values || samples.rows != shape_.dimension ||
      samples.columns == 0U)
    return Status::InvalidArgument(
        "runtime basic input shape disagrees with model");
  if (const auto s = output->Resize(shape_.m, samples.columns); !s.ok())
    return s;
  if (exact)
    exact->resize(static_cast<std::size_t>(samples.rows) * samples.columns);
  for (std::uint32_t first = 0; first < samples.columns;) {
    const auto count = std::min(kTileRows, samples.columns - first);
    const float* prepared = nullptr;
    if (const auto s = Prepare({samples.column(first), samples.rows, count}, prepared);
        !s.ok())
      return s;
    ArrayView<const std::uint32_t> tile_roots;
    ArrayView<const std::uint64_t> tile_ids;
    if (!roots.empty()) {
      tile_roots = {roots.data() + first, count};
      ids_.resize(count);
      for (std::uint32_t row = 0U; row < count; ++row)
        ids_[row] = offset + first + row;
      tile_ids = ids_;
    }
    if (const auto s =
            encoder_.Encode({prepared, samples.rows, count}, rank_,
                            schedule, seed, offset + first, tile_ids, tile_roots,
                            &tile_);
        !s.ok())
      return s;
    std::copy(tile_.roots.begin(), tile_.roots.end(),
              output->roots.begin() + first);
    std::copy(tile_.codes.begin(), tile_.codes.end(),
              output->codes.begin() +
                  static_cast<std::size_t>(first) * (shape_.m - 1U));
    std::copy(tile_.coefficients.begin(), tile_.coefficients.end(),
              output->coefficients.begin() +
                  static_cast<std::size_t>(first) * shape_.m);
    std::copy(tile_.raw_errors.begin(), tile_.raw_errors.end(),
              output->raw_errors.begin() + first);
    std::copy(tile_.errors.begin(), tile_.errors.end(),
              output->errors.begin() + first);
    if (exact)
      std::copy_n(
          prepared, static_cast<std::size_t>(count) * samples.rows,
          exact->begin() + static_cast<std::size_t>(first) * samples.rows);
    first += count;
  }
  return Status::Ok();
}

template <typename Scalar>
StatusOr<BasicEncodingBatch> RuntimeCpuRsdBasicExecutor<Scalar>::Encode(
    ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t offset) {
  RuntimeBasicEncoding output;
  if (const auto s =
          EncodeTiles(samples, schedule, seed, offset, &output, nullptr);
      !s.ok())
    return s;
  return BasicEncodingBatch(std::move(output));
}

template <typename Scalar>
StatusOr<RetainedModelSpaceBasicEncoding>
RuntimeCpuRsdBasicExecutor<Scalar>::EncodeRetainingModelSpace(
    ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t offset) {
  RuntimeBasicEncoding output;
  std::vector<float> exact;
  if (const auto s =
          EncodeTiles(samples, schedule, seed, offset, &output, &exact);
      !s.ok())
    return s;
  return RetainedModelSpaceBasicEncoding{samples.rows, std::move(exact),
                                         BasicEncodingBatch(std::move(output))};
}

template <typename Scalar>
StatusOr<RetainedModelSpaceBasicEncoding>
RuntimeCpuRsdBasicExecutor<Scalar>::EncodeFixedRootsRetainingModelSpace(
    ColumnMajorMatrixView<Scalar> samples, ArrayView<const std::uint32_t> roots,
    BasicSchedule schedule, std::uint32_t seed, std::uint64_t offset) {
  if (!samples.values || samples.rows != shape_.dimension ||
      samples.columns == 0U || roots.size() != samples.columns)
    return Status::InvalidArgument(
        "runtime fixed-root input shape disagrees with model");
  RuntimeBasicEncoding output;
  std::vector<float> exact;
  if (const auto s = EncodeTiles(samples, schedule, seed, offset, &output,
                                 &exact, roots);
      !s.ok())
    return s;
  return RetainedModelSpaceBasicEncoding{samples.rows, std::move(exact),
                                         BasicEncodingBatch(std::move(output))};
}

namespace {
template <typename Scalar>
class RuntimeRsdRound final : public BasicRoundRuntime<Scalar> {
 public:
  RuntimeRsdRound(RuntimeCpuRsdBasicExecutor<Scalar> executor,
                  std::uint32_t rank)
      : executor_(std::move(executor)), rank_(rank) {}
  BasicBlockExecutor<Scalar>& primary() noexcept override { return executor_; }
  BasicRuntimeExecutionInfo execution_info() const noexcept override {
    return {.rsd_rank = rank_,
            .primary_backend = BasicPrimaryBackend::kCpu,
            .variable_root_tile_capacity_rows =
                RuntimeCpuRsdBasicExecutor<Scalar>::kTileRows,
            .model_projections_on_device = false,
            .input_on_device = false};
  }

 private:
  RuntimeCpuRsdBasicExecutor<Scalar> executor_;
  std::uint32_t rank_;
};

}  // namespace

template <typename Scalar>
StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
CreateRuntimeCpuRsdRound(RuntimeRootCodebookView roots,
                         ColumnMajorMatrixF32View rotation,
                         const F32GemmBackend& gemm,
                         const ParallelForExecutor& parallel,
                         std::uint32_t rank) {
  auto executor = RuntimeCpuRsdBasicExecutor<Scalar>::Create(
      roots, rotation, gemm, parallel, rank);
  if (!executor.ok()) return executor.status();
  std::unique_ptr<BasicRoundRuntime<Scalar>> runtime =
      std::make_unique<RuntimeRsdRound<Scalar>>(std::move(executor).value(),
                                                rank);
  return runtime;
}

template StatusOr<std::unique_ptr<BasicRoundRuntime<std::uint8_t>>>
CreateRuntimeCpuRsdRound<std::uint8_t>(RuntimeRootCodebookView,
                                       ColumnMajorMatrixF32View,
                                       const F32GemmBackend&,
                                       const ParallelForExecutor&,
                                       std::uint32_t);
template StatusOr<std::unique_ptr<BasicRoundRuntime<float>>>
CreateRuntimeCpuRsdRound<float>(RuntimeRootCodebookView,
                                ColumnMajorMatrixF32View, const F32GemmBackend&,
                                const ParallelForExecutor&, std::uint32_t);

template class RuntimeCpuRsdBasicExecutor<std::uint8_t>;
template class RuntimeCpuRsdBasicExecutor<float>;
}  // namespace stlqpp
