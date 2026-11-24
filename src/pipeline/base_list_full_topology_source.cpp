#include "stlqpp/pipeline/base_list_full_topology_source.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

bool SameShape(const BaseBuildShape& lists,
               const ModelShape& model,
               std::uint32_t m) noexcept {
  return lists.dimension == model.dimension &&
      lists.nlist == model.nlist && model.m == m;
}

}  // namespace

template <int M>
StatusOr<BaseListFullTopologySource<M>>
BaseListFullTopologySource<M>::Create(
    const BaseListReader<M>& lists,
    const TrainingModel<M>& topology_model,
    const F32GemmBackend& gemm,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    bool collect_source_norm2) {
  const ModelShape& model_shape = topology_model.codebooks().shape();
  if (!SameShape(lists.store().shape(), model_shape, lists.store().m()) ||
      bad_cluster_mask.size() != model_shape.nlist) {
    return Status::InvalidArgument(
        "base full-topology source inputs have incompatible shapes");
  }
  std::vector<std::uint8_t> mask(
      bad_cluster_mask.begin(), bad_cluster_mask.end());
  for (const std::uint8_t value : mask) {
    if (value > 1U) {
      return Status::InvalidArgument(
          "base full-topology bad-cluster mask is not binary");
    }
  }
  auto transform = topology_model.source_space()
      ? StatusOr<ModelSpaceTransform>(ModelSpaceTransform::SourceSpace(model_shape.dimension))
      : ModelSpaceTransform::Create(topology_model.rotation());
  if (!transform.ok()) {
    return transform.status();
  }
  return BaseListFullTopologySource<M>(
      lists, topology_model, gemm,
      std::move(transform).value(), std::move(mask), collect_source_norm2);
}

template <int M>
Status BaseListFullTopologySource<M>::Rotate(
    ColumnMajorMatrixF32View source,
    std::vector<float>* destination) const {
  destination->resize(
      static_cast<std::size_t>(source.rows) * source.columns);
  if (source.columns == 0U) {
    return Status::Ok();
  }
  return gemm_->MultiplyLeft({
      .left = transform_.matrix(),
      .right = source.values,
      .query_count = source.columns,
      .right_leading_dimension = source.rows,
      .output = destination->data(),
      .output_leading_dimension = source.rows,
  });
}

template <int M>
Status BaseListFullTopologySource<M>::ReconstructBasic(
    const LoadedBaseListCid<M>& loaded,
    FullTopologyClusterWork<M>* work) const {
  const ProductCodebooks<M>& books = topology_model_->codebooks();
  const std::uint32_t dimension = work->dimension;
  const std::uint32_t count = work->real_count();
  work->basic_supports.resize(count);
  if (count != 0U) {
    work->basic_coefficients.assign(
        loaded.coefficients_row_major().begin(),
        loaded.coefficients_row_major().end());
  }
  work->basic_squared_errors.assign(count, 0.0F);
  work->basic_real_reconstructions.assign(
      static_cast<std::size_t>(dimension) * count, 0.0F);
  const auto codes = loaded.small_codes_row_major();
  const float* native = books.root0().column(work->cid);
  for (std::uint32_t row = 0U; row < count; ++row) {
    LinkageCodeTuple<M>& support = work->basic_supports[row];
    support.first = work->cid;
    const float* coefficients = work->basic_coefficients.data() +
        static_cast<std::size_t>(row) * books.shape().m;
    float* reconstruction = work->basic_real_reconstructions.data() +
        static_cast<std::size_t>(row) * dimension;
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      reconstruction[axis] = coefficients[0U] * native[axis];
    }
    for (std::uint32_t layer = 1U;
         layer < books.shape().m; ++layer) {
      const std::uint8_t code = codes[
          static_cast<std::size_t>(row) * (books.shape().m - 1U) + layer - 1U];
      support.trailing[layer - 1U] = code;
      const float* atom = books.root_small(layer).column(code);
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        reconstruction[axis] += coefficients[layer] * atom[axis];
      }
    }
    float squared_error = 0.0F;
    const float* exact = work->rotated_exact_real.data() +
        static_cast<std::size_t>(row) * dimension;
#pragma omp simd reduction(+ : squared_error)
    for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
      const float difference = exact[axis] - reconstruction[axis];
      squared_error += difference * difference;
    }
    if (!std::isfinite(squared_error)) {
      return Status::DataLoss(
          "base basic reconstruction produced a non-finite error");
    }
    work->basic_squared_errors[row] = squared_error;
  }
  return Status::Ok();
}

template <int M>
StatusOr<std::optional<FullTopologyClusterWork<M>>>
BaseListFullTopologySource<M>::Next() {
  const ModelShape& shape = topology_model_->codebooks().shape();
  if (next_cid_ == shape.nlist) {
    return std::optional<FullTopologyClusterWork<M>>{};
  }
  auto loaded_result = lists_->LoadCid(next_cid_);
  if (!loaded_result.ok()) {
    return loaded_result.status();
  }
  auto loaded = std::move(loaded_result).value();
  FullTopologyClusterWork<M> work;
  work.m = topology_model_->codebooks().shape().m;
  work.cid = next_cid_;
  work.dimension = shape.dimension;
  work.real_sample_ids.reserve(loaded.ids().size());
  for (const std::uint32_t id : loaded.ids()) {
    work.real_sample_ids.push_back(id);
  }
  const ColumnMajorMatrixF32View unrotated = loaded.exact_vectors();
  if (collect_source_norm2_) {
    work.source_real_norm2.assign(unrotated.columns, 0.0F);
    for (std::uint32_t row = 0U; row < unrotated.columns; ++row) {
      const float* vector = unrotated.values +
          static_cast<std::size_t>(row) * unrotated.rows;
      float norm2 = 0.0F;
#pragma omp simd reduction(+ : norm2)
      for (std::uint32_t axis = 0U; axis < unrotated.rows; ++axis) {
        norm2 += vector[axis] * vector[axis];
      }
      if (!std::isfinite(norm2) || norm2 < 0.0F) {
        return Status::DataLoss(
            "base source produced an invalid original-vector norm");
      }
      work.source_real_norm2[row] = norm2;
    }
  }
  work.bad_cluster =
      !work.real_sample_ids.empty() && bad_cluster_mask_[next_cid_] != 0U;
  if (transform_.source_space()) {
    work.exact_in_source_space = true;
    work.rotated_exact_real = std::move(loaded).TakeExactVectors();
  } else if (const Status status = Rotate(unrotated, &work.rotated_exact_real);
      !status.ok()) {
    return status;
  }
  if (const Status status = ReconstructBasic(loaded, &work); !status.ok()) {
    return status;
  }
  if (work.bad_cluster && !work.exact_in_source_space)
    work.unrotated_exact_real = std::move(loaded).TakeExactVectors();
  work.validated_basic = ValidatedBaseListBasicState{};
  ++next_cid_;
  return std::optional<FullTopologyClusterWork<M>>(std::move(work));
}

template <int M>
Status BaseListFullTopologySource<M>::SkipTo(std::uint32_t next_cid) {
  const std::uint32_t nlist = topology_model_->codebooks().shape().nlist;
  if (next_cid < next_cid_ || next_cid > nlist) {
    return Status::InvalidArgument(
        "base full-topology checkpoint cursor is invalid");
  }
  next_cid_ = next_cid;
  return Status::Ok();
}

template class BaseListFullTopologySource<0>;
template class BaseListFullTopologySource<5>;
template class BaseListFullTopologySource<12>;
template class BaseListFullTopologySource<6>;
template class BaseListFullTopologySource<10>;

}  // namespace stlqpp
