#include "stlqpp/model/root_codebooks.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

Status ValidateBook(ColumnMajorMatrixF32View book,
                    std::uint32_t dimension,
                    std::uint32_t entries) {
  if (book.values == nullptr || book.rows != dimension ||
      book.columns != entries) {
    return Status::InvalidArgument(
        "root-codebook dimensions do not match the registered model shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(dimension) * entries;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(book.values[index])) {
      return Status::InvalidArgument(
          "root codebook contains a non-finite value");
    }
  }
  return Status::Ok();
}

void Append(ColumnMajorMatrixF32View source,
            std::vector<float>* destination) {
  const std::size_t count =
      static_cast<std::size_t>(source.rows) * source.columns;
  destination->insert(destination->end(), source.values,
                      source.values + count);
}

}  // namespace

template <int M>
Status RootCodebooks<M>::ValidateView(
    const RootCodebookView<M>& source) {
  if (const Status status = source.shape.Validate(); !status.ok()) {
    return status;
  }
  if ((M != 0 && source.shape.m != static_cast<std::uint32_t>(M)) ||
      source.root_small.size() != source.shape.m - 1U) {
    return Status::InvalidArgument(
        "root-codebook M disagrees with its registered model shape");
  }
  if (const Status status = ValidateBook(
          source.root0, source.shape.dimension, source.shape.nlist);
      !status.ok()) {
    return status;
  }
  for (const auto& book : source.root_small) {
    if (const Status status = ValidateBook(
            book, source.shape.dimension, 256U);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

template <int M>
StatusOr<RootCodebooks<M>> RootCodebooks<M>::Create(
    const RootCodebookView<M>& source) {
  if (const Status status = ValidateView(source); !status.ok()) {
    return status;
  }
  RootCodebooks<M> output;
  output.shape_ = source.shape;
  Append(source.root0, &output.root0_);
  output.root_small_.reserve(static_cast<std::size_t>(source.shape.m - 1U) * 256U *
                             source.shape.dimension);
  for (const auto& book : source.root_small) {
    Append(book, &output.root_small_);
  }

  output.root_inverse_norms_.resize(source.shape.nlist);
  for (std::uint32_t cid = 0U; cid < source.shape.nlist; ++cid) {
    const float* codeword = source.root0.column(cid);
    double squared_norm = 0.0;
    for (std::uint32_t axis = 0U; axis < source.shape.dimension; ++axis) {
      const double value = static_cast<double>(codeword[axis]);
      squared_norm += value * value;
    }
    output.root_inverse_norms_[cid] = 1.0F / std::sqrt(
        std::max(static_cast<float>(squared_norm), 1.0e-20F));
  }
  return output;
}

template class RootCodebooks<0>;
template class RootCodebooks<5>;
template class RootCodebooks<12>;
template class RootCodebooks<6>;
template class RootCodebooks<10>;

}  // namespace stlqpp
