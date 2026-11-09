#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "stlqpp/index/norm/reconstruction_norms.h"
#include "stlqpp/index/norm/runtime_reconstruction_norms.h"
#include "stlqpp/runtime/exclusive_index_pool.h"

namespace stlqpp {

class RuntimeCudaReconstructionNormComputer final {
 public:
  [[nodiscard]] static StatusOr<
      std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
  Create(RuntimeProductCodebookView codebooks, int device = 0,
         ArrayView<const float> reference = {});
  [[nodiscard]] static StatusOr<
      std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
  CreateWithGrams(ReconstructionGramModelView grams, int device = 0);
  [[nodiscard]] StatusOr<std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
  Clone() const;
  ~RuntimeCudaReconstructionNormComputer();
  RuntimeCudaReconstructionNormComputer(
      const RuntimeCudaReconstructionNormComputer&) = delete;
  RuntimeCudaReconstructionNormComputer& operator=(
      const RuntimeCudaReconstructionNormComputer&) = delete;
  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const RuntimeReconstructionNormInput& input,
      ReconstructionRootCrossView cross = {});
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloat(
      const RuntimeFloatReconstructionNormInput& input,
      ReconstructionRootCrossView cross = {});
  [[nodiscard]] std::size_t workspace_bytes() const noexcept;
  [[nodiscard]] const void* shared_model_identity() const noexcept;

 private:
  RuntimeCudaReconstructionNormComputer() = default;
  template <typename Input>
  StatusOr<std::vector<float>> ComputeImpl(const Input& input,
                                           ReconstructionRootCrossView cross);
  struct Model;
  struct State;
  std::unique_ptr<State> state_;
  std::mutex mutex_;
};

// Bounded independent streams/workspaces sharing one immutable Gram model.
class CudaReconstructionNormPool final {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<CudaReconstructionNormPool>>
  Create(ReconstructionGramModelView grams, int device,
         std::uint32_t lanes = 8U);
  [[nodiscard]] StatusOr<std::vector<float>> Compute(
      const RuntimeReconstructionNormInput&, ReconstructionRootCrossView);
  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloat(
      const RuntimeFloatReconstructionNormInput&, ReconstructionRootCrossView);

 private:
  std::shared_ptr<ExclusiveIndexPool> leases_;
  std::vector<std::unique_ptr<RuntimeCudaReconstructionNormComputer>> workers_;
};

// One immutable model upload, with bounded per-cluster workspace reuse.
// Concurrent calls lease independent streams sharing the immutable tables.
template <int M>
class CudaReconstructionNormComputer final {
 public:
  static StatusOr<std::unique_ptr<CudaReconstructionNormComputer>> Create(
      const ProductCodebooks<M>& codebooks, int device = 0);
  static StatusOr<std::unique_ptr<CudaReconstructionNormComputer>> Create(
      const SelectedAtomGramTables<M>& grams, int device);
  ~CudaReconstructionNormComputer();
  CudaReconstructionNormComputer(const CudaReconstructionNormComputer&) =
      delete;
  CudaReconstructionNormComputer& operator=(
      const CudaReconstructionNormComputer&) = delete;

  StatusOr<std::vector<float>> Compute(const ReconstructionNormInput<M>& input,
                                       ReconstructionRootCrossView cross = {});

  StatusOr<std::vector<float>> ComputeFloat(
      const FloatReconstructionNormInput<M>&,
      ReconstructionRootCrossView cross = {});

 private:
  CudaReconstructionNormComputer() = default;
  std::unique_ptr<RuntimeCudaReconstructionNormComputer> runtime_;
  std::unique_ptr<CudaReconstructionNormPool> pool_;
};

}  // namespace stlqpp
