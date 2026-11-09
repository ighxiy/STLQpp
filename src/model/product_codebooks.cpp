#include "stlqpp/model/product_codebooks.h"

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
        "codebook dimensions do not match the registered model shape");
  }
  const std::size_t count =
      static_cast<std::size_t>(dimension) * entries;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(book.values[index])) {
      return Status::InvalidArgument(
          "codebook contains a non-finite value");
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

template <int M>
Status ValidateEdgeBooks(const ProductCodebookView<M>& source) {
  for (const auto& book : source.edge) {
    if (const Status status = ValidateBook(
            book, source.shape.dimension, 256U);
        !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

}  // namespace

template <int M>
Status ProductCodebooks<M>::ValidateView(
    const ProductCodebookView<M>& source) {
  if (const Status status = source.shape.Validate(); !status.ok()) {
    return status;
  }
  if ((M != 0 && source.shape.m != static_cast<std::uint32_t>(M)) ||
      source.edge.size() != source.shape.m) {
    return Status::InvalidArgument(
        "codebook M disagrees with its registered model shape");
  }
  RootCodebookView<M> root_view;
  root_view.shape = source.shape;
  root_view.root0 = source.root0;
  root_view.root_small = source.root_small;
  if (const Status status = RootCodebooks<M>::ValidateView(root_view);
      !status.ok()) {
    return status;
  }
  return ValidateEdgeBooks(source);
}

template <int M>
StatusOr<ProductCodebooks<M>> ProductCodebooks<M>::Create(
    const ProductCodebookView<M>& source) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  if (const Status shape_status = source.shape.Validate();
      !shape_status.ok()) {
    return shape_status;
  }
  if ((M != 0 && source.shape.m != static_cast<std::uint32_t>(M)) ||
      source.edge.size() != source.shape.m) {
    return Status::InvalidArgument(
        "codebook M disagrees with its registered model shape");
  }
  RootCodebookView<M> root_view;
  root_view.shape = source.shape;
  root_view.root0 = source.root0;
  root_view.root_small = source.root_small;
  auto root = RootCodebooks<M>::Create(root_view);
  if (!root.ok()) {
    return root.status();
  }
  if (const Status status = ValidateEdgeBooks(source); !status.ok()) {
    return status;
  }

  std::vector<float> edge;
  edge.reserve(static_cast<std::size_t>(source.shape.m) * 256U *
               source.shape.dimension);
  for (const auto& book : source.edge) {
    Append(book, &edge);
  }
  return ProductCodebooks<M>(
      std::move(root).value(), std::move(edge));
}

template class ProductCodebooks<0>;
template class ProductCodebooks<5>;
template class ProductCodebooks<12>;
template class ProductCodebooks<6>;
template class ProductCodebooks<10>;

}  // namespace stlqpp
