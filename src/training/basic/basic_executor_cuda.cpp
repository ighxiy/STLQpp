#include "stlqpp/training/basic/basic_executor_cuda.h"

#include <memory>
#include <utility>

#include "stlqpp/training/basic/detail/basic_encoding_access.h"

namespace stlqpp {

template <int M, typename Scalar>
struct CudaBasicBlockExecutor<M, Scalar>::State final {
  State(CudaBasicRotator<Scalar>&& input_rotator,
        CudaBasicEncoder<M>&& input_encoder) noexcept
      : rotator(std::move(input_rotator)), encoder(std::move(input_encoder)) {}

  CudaBasicRotator<Scalar> rotator;
  CudaBasicEncoder<M> encoder;
};

template <int M, typename Scalar>
CudaBasicBlockExecutor<M, Scalar>::CudaBasicBlockExecutor(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M, typename Scalar>
CudaBasicBlockExecutor<M, Scalar>::~CudaBasicBlockExecutor() = default;

template <int M, typename Scalar>
CudaBasicBlockExecutor<M, Scalar>::CudaBasicBlockExecutor(
    CudaBasicBlockExecutor&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M, typename Scalar>
CudaBasicBlockExecutor<M, Scalar>& CudaBasicBlockExecutor<M, Scalar>::operator=(
    CudaBasicBlockExecutor&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M, typename Scalar>
StatusOr<CudaBasicBlockExecutor<M, Scalar>>
CudaBasicBlockExecutor<M, Scalar>::Create(
    ColumnMajorMatrixF32View rotation, const BasicCodebookAlgebra<M>& algebra,
    CudaContext& context) {
  auto rotator = rotation.values
      ? CudaBasicRotator<Scalar>::Create(rotation, context)
      : CudaBasicRotator<Scalar>::SourceSpace(algebra.shape().dimension, context);
  if (!rotator.ok()) {
    return rotator.status();
  }
  auto encoder = CudaBasicEncoder<M>::Create(algebra, context);
  if (!encoder.ok()) {
    return encoder.status();
  }
  auto state = std::make_unique<State>(std::move(rotator).value(),
                                       std::move(encoder).value());
  return CudaBasicBlockExecutor<M, Scalar>(std::move(state));
}

template <int M, typename Scalar>
StatusOr<BasicEncodingBatch> CudaBasicBlockExecutor<M, Scalar>::Encode(
    ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("CUDA basic executor is not initialized");
  }
  auto rotated = state_->rotator.Rotate(samples);
  if (!rotated.ok()) {
    return rotated.status();
  }
  return detail::BasicEncodingAccess::Release(state_->encoder.EncodeDevice(
      rotated.value(), schedule, seed, global_sample_offset));
}

template <int M, typename Scalar>
StatusOr<RetainedModelSpaceBasicEncoding>
CudaBasicBlockExecutor<M, Scalar>::EncodeRetainingModelSpace(
    ColumnMajorMatrixView<Scalar> samples, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("CUDA basic executor is not initialized");
  }
  auto rotated = state_->rotator.Rotate(samples);
  if (!rotated.ok()) {
    return rotated.status();
  }
  auto encoding = state_->encoder.EncodeDevice(rotated.value(), schedule, seed,
                                               global_sample_offset);
  if (!encoding.ok()) {
    return encoding.status();
  }
  auto host = state_->rotator.CopyToHost(rotated.value());
  if (!host.ok()) {
    return host.status();
  }
  return RetainedModelSpaceBasicEncoding{
      .dimension = samples.rows,
      .exact_model_space = std::move(host).value(),
      .encoding =
          detail::BasicEncodingAccess::Release(std::move(encoding).value()),
  };
}

template <int M, typename Scalar>
StatusOr<RetainedModelSpaceBasicEncoding>
CudaBasicBlockExecutor<M, Scalar>::EncodeFixedRootsRetainingModelSpace(
    ColumnMajorMatrixView<Scalar> samples,
    ArrayView<const std::uint32_t> fixed_roots, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("CUDA basic executor is not initialized");
  }
  auto rotated = state_->rotator.Rotate(samples);
  if (!rotated.ok()) {
    return rotated.status();
  }
  auto encoding = state_->encoder.EncodeFixedRootsDevice(
      rotated.value(), fixed_roots, global_sample_offset, schedule, seed);
  if (!encoding.ok()) {
    return encoding.status();
  }
  auto host = state_->rotator.CopyToHost(rotated.value());
  if (!host.ok()) {
    return host.status();
  }
  return RetainedModelSpaceBasicEncoding{
      .dimension = samples.rows,
      .exact_model_space = std::move(host).value(),
      .encoding =
          detail::BasicEncodingAccess::Release(std::move(encoding).value()),
  };
}

template class CudaBasicBlockExecutor<0, std::uint8_t>;
template class CudaBasicBlockExecutor<0, float>;
template class CudaBasicBlockExecutor<5, std::uint8_t>;
template class CudaBasicBlockExecutor<12, std::uint8_t>;
template class CudaBasicBlockExecutor<6, std::uint8_t>;
template class CudaBasicBlockExecutor<10, std::uint8_t>;
template class CudaBasicBlockExecutor<5, float>;
template class CudaBasicBlockExecutor<12, float>;
template class CudaBasicBlockExecutor<6, float>;
template class CudaBasicBlockExecutor<10, float>;

}  // namespace stlqpp
