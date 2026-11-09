#pragma once

#include "stlqpp/training/basic/basic_executor.h"
#include "stlqpp/training/basic/basic_round_runtime.h"
#include "stlqpp/training/basic/runtime_rsd_basic.h"

namespace stlqpp {

// Runtime-M counterpart of the fixed CPU RSD executor. Ordinary encoding uses
// the same 8192-row numerical tiles, including fixed-root replay.
template <typename Scalar>
class RuntimeCpuRsdBasicExecutor final : public BasicBlockExecutor<Scalar> {
 public:
  static constexpr std::uint32_t kTileRows = 8192U;
  [[nodiscard]] static StatusOr<RuntimeCpuRsdBasicExecutor> Create(
      RuntimeRootCodebookView roots, ColumnMajorMatrixF32View rotation,
      const F32GemmBackend& gemm, const ParallelForExecutor& parallel,
      std::uint32_t rank);
  [[nodiscard]] StatusOr<BasicEncodingBatch> Encode(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t offset) override;
  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeRetainingModelSpace(ColumnMajorMatrixView<Scalar> samples,
                            BasicSchedule schedule, std::uint32_t seed,
                            std::uint64_t offset) override;
  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeFixedRootsRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t offset) override;

 private:
  RuntimeCpuRsdBasicExecutor(ModelShape shape, std::vector<float> rotation,
                             RuntimeCpuRsdBasicEncoder encoder,
                             const F32GemmBackend& gemm,
                             const ParallelForExecutor& parallel,
                             std::uint32_t rank)
      : shape_(shape),
        rotation_(std::move(rotation)),
        encoder_(std::move(encoder)),
        gemm_(&gemm),
        parallel_(&parallel),
        rank_(rank) {}
  [[nodiscard]] Status Rotate(ColumnMajorMatrixView<Scalar> samples);
  [[nodiscard]] Status EncodeTiles(ColumnMajorMatrixView<Scalar> samples,
                                   BasicSchedule schedule, std::uint32_t seed,
                                   std::uint64_t offset,
                                   RuntimeBasicEncoding* output,
                                   std::vector<float>* exact,
                                   ArrayView<const std::uint32_t> roots = {});
  ModelShape shape_;
  std::vector<float> rotation_;
  RuntimeCpuRsdBasicEncoder encoder_;
  const F32GemmBackend* gemm_;
  const ParallelForExecutor* parallel_;
  std::uint32_t rank_;
  std::vector<float> converted_, rotated_;
  std::vector<std::uint64_t> ids_;
  RuntimeBasicEncoding tile_;
};

// Single-lane runtime used by both training and final-base composition.
template <typename Scalar>
[[nodiscard]] StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
CreateRuntimeCpuRsdRound(RuntimeRootCodebookView roots,
                         ColumnMajorMatrixF32View rotation,
                         const F32GemmBackend& gemm,
                         const ParallelForExecutor& parallel,
                         std::uint32_t rank);

extern template class RuntimeCpuRsdBasicExecutor<std::uint8_t>;
extern template class RuntimeCpuRsdBasicExecutor<float>;
}  // namespace stlqpp
