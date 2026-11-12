#include "stlqpp/model/model_shape.h"
#include "stlqpp/pipeline/training_list_root_initialization_source.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kRootOnlyIcmSeedSalt = 0x0A17C9E5U;

}  // namespace

template <int M>
StatusOr<TrainingListRootInitializationSource<M>>
TrainingListRootInitializationSource<M>::Create(
    const BaseListReader<M>& lists,
    const RootTrainingModel<M>& model,
    std::uint32_t topology_seed) {
  const BaseBuildShape& list_shape = lists.store().shape();
  const ModelShape& model_shape = model.codebooks().shape();
  if (lists.store().m() != model_shape.m ||
      list_shape.dimension != model_shape.dimension ||
      list_shape.nlist != model_shape.nlist) {
    return Status::InvalidArgument(
        "root-initialization list source has incompatible shapes");
  }
  return TrainingListRootInitializationSource<M>(
      lists, model, topology_seed);
}

template <int M>
Status TrainingListRootInitializationSource<M>::ReconstructBasic(
    const LoadedBaseListCid<M>& loaded,
    RootOnlyInitializationClusterWork<M>* work) const {
  const RootCodebooks<M>& books = model_->codebooks();
  const std::uint32_t dimension = work->dimension;
  const std::uint32_t count = work->real_count();
  work->basic_supports.resize(count);
  work->basic_coefficients.assign(
      loaded.coefficients_row_major().begin(),
      loaded.coefficients_row_major().end());
  work->basic_squared_errors.assign(count, 0.0F);
  work->basic_real_reconstructions.assign(
      static_cast<std::size_t>(dimension) * count, 0.0F);
  const auto codes = loaded.small_codes_row_major();
  const float* root = books.root0().column(work->cid);
  for (std::uint32_t row = 0U; row < count; ++row) {
    LinkageCodeTuple<M>& support = work->basic_supports[row];
    support.first = work->cid;
    const float* coefficients = work->basic_coefficients.data() +
        static_cast<std::size_t>(row) * books.m();
    float* reconstruction = work->basic_real_reconstructions.data() +
        static_cast<std::size_t>(row) * dimension;
    std::fill_n(reconstruction, dimension, 0.0F);
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      reconstruction[axis] += coefficients[0U] * root[axis];
    }
    for (std::uint32_t layer = 1U;
         layer < books.m(); ++layer) {
      const std::uint8_t code = codes[
          static_cast<std::size_t>(row) * (books.m() - 1U) + layer - 1U];
      support.trailing[layer - 1U] = code;
      const float* atom = books.root_small(layer).column(code);
#pragma omp simd
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        reconstruction[axis] += coefficients[layer] * atom[axis];
      }
    }
    float error = 0.0F;
    const float* exact = work->exact_real.data() +
        static_cast<std::size_t>(row) * dimension;
#pragma omp simd reduction(+ : error)
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      const float difference = exact[axis] - reconstruction[axis];
      error += difference * difference;
    }
    if (!std::isfinite(error)) {
      return Status::DataLoss(
          "root-initialization reconstruction produced a non-finite error");
    }
    work->basic_squared_errors[row] = error;
  }
  return Status::Ok();
}

template <int M>
StatusOr<std::optional<RootOnlyInitializationClusterWork<M>>>
TrainingListRootInitializationSource<M>::Next() {
  const ModelShape& shape = model_->codebooks().shape();
  if (next_cid_ == shape.nlist) {
    return std::optional<RootOnlyInitializationClusterWork<M>>{};
  }
  auto loaded_result = lists_->LoadCid(next_cid_);
  if (!loaded_result.ok()) {
    return loaded_result.status();
  }
  auto loaded = std::move(loaded_result).value();
  RootOnlyInitializationClusterWork<M> work;
  work.m = shape.m;
  work.cid = next_cid_;
  work.dimension = shape.dimension;
  work.sample_ids.assign(loaded.ids().begin(), loaded.ids().end());
  const ColumnMajorMatrixF32View exact = loaded.exact_vectors();
  if (exact.columns != 0U) {
    work.exact_real.assign(
        exact.values,
        exact.values + static_cast<std::size_t>(exact.rows) * exact.columns);
  }
  auto& order = work.icm_layer_order;
  order.resize(shape.m);
  std::iota(order.begin(), order.end(), static_cast<std::uint8_t>(0U));
  std::mt19937 rng(static_cast<std::mt19937::result_type>(
      (topology_seed_ + 997U * next_cid_) ^ kRootOnlyIcmSeedSalt));
  std::shuffle(order.begin(), order.end(), rng);
  if (const Status status = ReconstructBasic(loaded, &work); !status.ok()) {
    return status;
  }
  work.validated_basic = ValidatedBaseListBasicState{};
  ++next_cid_;
  return std::optional<RootOnlyInitializationClusterWork<M>>(
      std::move(work));
}

template class TrainingListRootInitializationSource<0>;
template class TrainingListRootInitializationSource<5>;
template class TrainingListRootInitializationSource<12>;
template class TrainingListRootInitializationSource<6>;
template class TrainingListRootInitializationSource<10>;

}  // namespace stlqpp
