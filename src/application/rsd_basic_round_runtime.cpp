#include "stlqpp/application/rsd_basic_round_runtime.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/training/basic/basic_encoder_cuda.h"
#include "stlqpp/training/basic/basic_executor.h"
#include "stlqpp/training/basic/basic_rotator_cuda.h"
#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_cuda.h"
#include "stlqpp/training/linkage/rsd/linkage_rsd_device_plan.h"

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
  if constexpr (M == 0) {
    view.root_small.resize(roots.shape().m - 1U);
    view.edge.resize(roots.shape().m);
  }
  view.edge[0] = roots.root0();
  for (std::uint32_t layer = 1U; layer < roots.shape().m; ++layer) {
    view.root_small[layer - 1U] = roots.root_small(layer);
    view.edge[layer] = roots.root_small(layer);
  }
  return ProductCodebooks<M>::Create(view);
}

template <int M, typename Scalar>
class CudaRsdBasicExecutor final : public BasicBlockExecutor<Scalar> {
 public:
  CudaRsdBasicExecutor(CudaBasicRotator<Scalar> rotator,
                       CudaBasicEncoder<M> basic,
                       std::unique_ptr<CudaLinkageRsdEncoder<M>> rsd,
                       std::uint32_t maximum_rank,
                       LeastSquaresCostMode rsd_cost_mode) noexcept
      : rotator_(std::move(rotator)),
        basic_(std::move(basic)),
        rsd_(std::move(rsd)),
        maximum_rank_(maximum_rank),
        rsd_cost_mode_(rsd_cost_mode) {}

  [[nodiscard]] StatusOr<BasicEncodingBatch> Encode(
      ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override {
    auto rotated = rotator_.Rotate(samples);
    if (!rotated.ok()) return rotated.status();
    return detail::BasicEncodingAccess::Release(EncodeRotated(
        rotated.value(), {}, schedule, seed, global_sample_offset));
  }

  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeRetainingModelSpace(ColumnMajorMatrixView<Scalar> samples,
                            BasicSchedule schedule, std::uint32_t seed,
                            std::uint64_t global_sample_offset) override {
    auto rotated = rotator_.Rotate(samples);
    if (!rotated.ok()) return rotated.status();
    auto encoding = EncodeRotated(rotated.value(), {}, schedule, seed,
                                  global_sample_offset);
    if (!encoding.ok()) return encoding.status();
    auto model_space = rotator_.CopyToHost(rotated.value());
    if (!model_space.ok()) return model_space.status();
    return RetainedModelSpaceBasicEncoding{
        .dimension = samples.rows,
        .exact_model_space = std::move(model_space).value(),
        .encoding =
            detail::BasicEncodingAccess::Release(std::move(encoding).value()),
    };
  }

  [[nodiscard]] StatusOr<RetainedModelSpaceBasicEncoding>
  EncodeFixedRootsRetainingModelSpace(
      ColumnMajorMatrixView<Scalar> samples,
      ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
      std::uint32_t seed, std::uint64_t global_sample_offset) override {
    auto rotated = rotator_.Rotate(samples);
    if (!rotated.ok()) return rotated.status();
    auto encoding = EncodeRotated(rotated.value(), fixed_roots, schedule, seed,
                                  global_sample_offset);
    if (!encoding.ok()) return encoding.status();
    auto model_space = rotator_.CopyToHost(rotated.value());
    if (!model_space.ok()) return model_space.status();
    return RetainedModelSpaceBasicEncoding{
        .dimension = samples.rows,
        .exact_model_space = std::move(model_space).value(),
        .encoding =
            detail::BasicEncodingAccess::Release(std::move(encoding).value()),
    };
  }

 private:
  [[nodiscard]] StatusOr<BasicEncoding<M>> EncodeRotated(
      CudaMatrixF32View samples, ArrayView<const std::uint32_t> fixed_roots,
      BasicSchedule schedule, std::uint32_t seed,
      std::uint64_t global_sample_offset) {
    if (rsd_ == nullptr) {
      return basic_.EncodeRsdDevice(samples, fixed_roots, maximum_rank_,
                                    rsd_cost_mode_, schedule, seed,
                                    global_sample_offset);
    }
    if (!fixed_roots.empty()) {
      auto support = rsd_->EncodeRawRsdFixedLayerZeroSupportDeviceResident(
          samples, fixed_roots);
      if (!support.ok()) return support.status();
      return basic_.RefineDeviceResidentSupports(
          samples, support.value().codes, schedule, seed, global_sample_offset,
          support.value().squared_norms);
    }
    auto support = rsd_->EncodeRawRsdTopOneRootSupportDeviceResident(samples);
    if (!support.ok()) return support.status();
    return basic_.RefineDeviceResidentSupports(
        samples, support.value().codes, schedule, seed, global_sample_offset,
        support.value().squared_norms);
  }

  CudaBasicRotator<Scalar> rotator_;
  CudaBasicEncoder<M> basic_;
  std::unique_ptr<CudaLinkageRsdEncoder<M>> rsd_;
  std::uint32_t maximum_rank_ = 2U;
  LeastSquaresCostMode rsd_cost_mode_ = LeastSquaresCostMode::kNormalEquation;
};

template <int M, typename Scalar>
class CudaRsdBasicRoundRuntime final : public BasicRoundRuntime<Scalar> {
 public:
  CudaRsdBasicRoundRuntime(
      std::unique_ptr<CudaRsdBasicExecutor<M, Scalar>> gpu,
      std::uint32_t maximum_rank) noexcept
      : gpu_(std::move(gpu)),
        maximum_rank_(maximum_rank) {}

  [[nodiscard]] BasicBlockExecutor<Scalar>& primary() noexcept override {
    return *gpu_;
  }

  [[nodiscard]] BasicRuntimeExecutionInfo execution_info()
      const noexcept override {
    // The RSD executor evaluates each source block as one numerical unit. Its
    // fixed-root refinement also deliberately preserves that shape.
    return {
        .rsd_rank = maximum_rank_,
        .primary_backend = BasicPrimaryBackend::kCuda,
        .variable_root_tile_capacity_rows = 0U,
        .model_projections_on_device = true,
        .input_on_device = true,
    };
  }

 private:
  std::unique_ptr<CudaRsdBasicExecutor<M, Scalar>> gpu_;
  std::uint32_t maximum_rank_ = 2U;
};

}  // namespace

template <int M, typename Scalar>
StatusOr<std::unique_ptr<BasicRoundRuntime<Scalar>>>
CudaRsdBasicRoundRuntimeFactory<M, Scalar>::Create(
    const RootCodebooks<M>& codebooks, ColumnMajorMatrixF32View rotation) {
  assert(cuda_ != nullptr);
  if (maximum_rank_ < 1U || maximum_rank_ > 3U) {
    return Status::InvalidArgument(
        "CUDA basic RSD factory requires rank 1, 2 or 3");
  }
  auto rotator = rotation.values
      ? CudaBasicRotator<Scalar>::Create(rotation, *cuda_)
      : CudaBasicRotator<Scalar>::SourceSpace(codebooks.shape().dimension, *cuda_);
  if (!rotator.ok()) return rotator.status();
  std::shared_ptr<const CudaLinkageRsdDeviceModel<M>> rsd_model;
  std::unique_ptr<CudaLinkageRsdEncoder<M>> rsd_owner;
  if (codebooks.shape().nlist == 256U) {
    auto product = CreateRootProductCodebooks(codebooks);
    if (!product.ok()) return product.status();
    auto model = CudaLinkageRsdDeviceModel<M>::Create(product.value(), *cuda_);
    if (!model.ok()) return model.status();
    rsd_model = std::move(model).value();
    auto rsd = CudaLinkageRsdEncoder<M>::Create(rsd_model, *cuda_,
                                                maximum_rank_, cost_mode_);
    if (!rsd.ok()) return rsd.status();
    rsd_owner =
        std::make_unique<CudaLinkageRsdEncoder<M>>(std::move(rsd).value());
  }
  auto basic = rsd_model
      ? CudaBasicEncoder<M>::Create(codebooks, std::move(rsd_model), *cuda_)
      : CudaBasicEncoder<M>::Create(codebooks, *cuda_);
  if (!basic.ok()) return basic.status();
  std::unique_ptr<BasicRoundRuntime<Scalar>> runtime =
      std::make_unique<CudaRsdBasicRoundRuntime<M, Scalar>>(
          std::make_unique<CudaRsdBasicExecutor<M, Scalar>>(
              std::move(rotator).value(), std::move(basic).value(),
              std::move(rsd_owner), maximum_rank_, cost_mode_),
          maximum_rank_);
  return runtime;
}

template class CudaRsdBasicRoundRuntimeFactory<0, std::uint8_t>;
template class CudaRsdBasicRoundRuntimeFactory<0, float>;
template class CudaRsdBasicRoundRuntimeFactory<5, std::uint8_t>;
template class CudaRsdBasicRoundRuntimeFactory<12, std::uint8_t>;
template class CudaRsdBasicRoundRuntimeFactory<6, std::uint8_t>;
template class CudaRsdBasicRoundRuntimeFactory<10, std::uint8_t>;
template class CudaRsdBasicRoundRuntimeFactory<5, float>;
template class CudaRsdBasicRoundRuntimeFactory<12, float>;
template class CudaRsdBasicRoundRuntimeFactory<6, float>;
template class CudaRsdBasicRoundRuntimeFactory<10, float>;

}  // namespace stlqpp
