#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/model/query_transform.h"
#include "stlqpp/model/runtime_codebooks.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
struct CanonicalModelSource final {
  ColumnMajorMatrixF32View rotation;
  ProductCodebookView<M> codebooks;
  ArtifactHash64 source_model_identity{};
};

struct RuntimeCanonicalModelSource final {
  ModelShape shape;
  ColumnMajorMatrixF32View rotation;
  ColumnMajorMatrixF32View root0;
  ArrayView<const ColumnMajorMatrixF32View> root_small;
  ArrayView<const ColumnMajorMatrixF32View> edge;
  ArtifactHash64 source_model_identity{};
};

// Runtime-M view of a decoded canonical model. This owns the exact on-disk
// float sequence; accessors borrow contiguous matrices without repacking.
class RuntimeCanonicalModel final {
 public:
  RuntimeCanonicalModel(ModelShape shape, std::vector<float> values) noexcept
      : shape_(shape), values_(std::move(values)) {}

  static RuntimeCanonicalModel SourceSpace(ModelShape shape,
      std::vector<float> books, ArtifactHash64 base_identity) {
    RuntimeCanonicalModel model(shape, std::move(books));
    model.source_space_ = true;
    model.base_identity_ = base_identity;
    return model;
  }
  [[nodiscard]] bool source_space() const noexcept { return source_space_; }
  [[nodiscard]] ArtifactHash64 base_identity() const noexcept { return base_identity_; }
  [[nodiscard]] Status Validate() const;

  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] ColumnMajorMatrixF32View rotation() const noexcept;
  [[nodiscard]] ColumnMajorMatrixF32View root0() const noexcept;
  [[nodiscard]] ColumnMajorMatrixF32View root_small(
      std::uint32_t layer) const noexcept;
  [[nodiscard]] ColumnMajorMatrixF32View edge(
      std::uint32_t layer) const noexcept;
  [[nodiscard]] RuntimeProductCodebookView codebooks_view() const noexcept {
    return {shape_, root0(),
            {root_small(1U).values, shape_.dimension, (shape_.m - 1U) * 256U},
            {edge(0U).values, shape_.dimension, shape_.m * 256U}};
  }

 private:
  bool source_space_ = false;
  ArtifactHash64 base_identity_;
  ModelShape shape_;
  std::vector<float> values_;
};

template <int M>
class CanonicalModel final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  CanonicalModel(std::vector<float> canonical_rotation,
                 QueryTransform query_transform,
                 ProductCodebooks<M> codebooks) noexcept
      : canonical_rotation_(std::move(canonical_rotation)),
        query_transform_(std::move(query_transform)),
        codebooks_(std::move(codebooks)) {}

  CanonicalModel(CanonicalModel&&) noexcept = default;
  CanonicalModel& operator=(CanonicalModel&&) noexcept = default;
  CanonicalModel(const CanonicalModel&) = delete;
  CanonicalModel& operator=(const CanonicalModel&) = delete;

  [[nodiscard]] QueryTransform TakeQueryTransform() && noexcept {
    return std::move(query_transform_);
  }
  [[nodiscard]] ProductCodebooks<M> TakeCodebooks() && noexcept {
    return std::move(codebooks_);
  }
  [[nodiscard]] const QueryTransform& query_transform() const noexcept {
    return query_transform_;
  }
  [[nodiscard]] ColumnMajorMatrixF32View canonical_rotation() const noexcept {
    if (canonical_rotation_.empty()) return {};
    const std::uint32_t dimension = codebooks_.shape().dimension;
    return {canonical_rotation_.data(), dimension, dimension};
  }
  [[nodiscard]] const ProductCodebooks<M>& codebooks() const noexcept {
    return codebooks_;
  }

 private:
  std::vector<float> canonical_rotation_;
  QueryTransform query_transform_;
  ProductCodebooks<M> codebooks_;
};

class CanonicalModelCodec final {
 public:
  static constexpr std::uint32_t kEncodingVersion = 1U;
  static constexpr std::uint32_t kHeaderBytes = 48U;

  static constexpr std::uint32_t kSourceSpaceVersion = 2U;
  [[nodiscard]] static Status EncodeSearchModel(
      const RuntimeCanonicalModelSource& source,
      std::vector<std::uint8_t>& output);

  [[nodiscard]] static Status EncodeRuntime(
      const RuntimeCanonicalModelSource& source,
      std::vector<std::uint8_t>& output);

  [[nodiscard]] static StatusOr<RuntimeCanonicalModel> DecodeRuntime(
      const std::uint8_t* bytes, std::size_t byte_count);

  template <int M>
  [[nodiscard]] static Status Encode(
      const CanonicalModelSource<M>& source,
      std::vector<std::uint8_t>& output);

  template <int M>
  [[nodiscard]] static StatusOr<CanonicalModel<M>> Decode(
      const std::uint8_t* bytes,
      std::size_t byte_count);
};

extern template Status CanonicalModelCodec::Encode<5>(
    const CanonicalModelSource<5>&, std::vector<std::uint8_t>&);
extern template Status CanonicalModelCodec::Encode<10>(
    const CanonicalModelSource<10>&, std::vector<std::uint8_t>&);
extern template StatusOr<CanonicalModel<5>> CanonicalModelCodec::Decode<5>(
    const std::uint8_t*, std::size_t);
extern template StatusOr<CanonicalModel<10>> CanonicalModelCodec::Decode<10>(
    const std::uint8_t*, std::size_t);

}  // namespace stlqpp
