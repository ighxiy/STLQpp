#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/platform/mapped_file.h"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace stlqpp {

namespace detail {

struct BaseListSupportStorage final {
  std::uint32_t cid = 0U;
  std::vector<std::uint32_t> ids;
  std::vector<std::uint8_t> small_codes;
  std::vector<float> coefficients;
};

struct BaseListCidStorage final {
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint32_t> ids;
  std::vector<std::uint8_t> small_codes;
  std::vector<float> coefficients;
  std::vector<float> exact;
};

// Mapped ownership uses the actual metadata width. Bulk support decoding and
// gather retain their fixed/runtime strides over these common mappings.
struct BaseListReaderStorage final {
  [[nodiscard]] static StatusOr<BaseListReaderStorage> Open(
      const BaseListMetadata& metadata);
  template <int M>
  [[nodiscard]] Status LoadCidSupportPayload(
      const BaseListMetadata& metadata, std::uint32_t cid,
      std::vector<std::uint32_t>* ids,
      std::vector<std::uint8_t>* small_codes,
      std::vector<float>* coefficients) const;
  [[nodiscard]] Status LoadRawCid(
      const BaseListMetadata& metadata, std::uint32_t cid,
      BaseListCidStorage* output) const;

  ReadOnlyMappedFile ids_file;
  ReadOnlyMappedFile codes_file;
  ReadOnlyMappedFile coefficients_file;
  ReadOnlyMappedFile raw_file;
};

}  // namespace detail

// Support-only list view for consumers that do not need the list-major raw
// payload.
template <int M>
class LoadedBaseListSupportCid final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t cid() const noexcept { return storage_.cid; }
  [[nodiscard]] ArrayView<const std::uint32_t> ids() const noexcept {
    return storage_.ids;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes_row_major() const
      noexcept {
    return storage_.small_codes;
  }
  [[nodiscard]] ArrayView<const float> coefficients_row_major() const
      noexcept {
    return storage_.coefficients;
  }

 private:
  template <int N>
  friend class BaseListReader;

  detail::BaseListSupportStorage storage_;
};

// One native-IVF base list decoded into the bounded float domain consumed by
// topology/replay. Vectors remain contiguous by sample, which is exactly a
// column-major d-by-n matrix.
template <int M>
class LoadedBaseListCid final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t cid() const noexcept { return storage_.cid; }
  [[nodiscard]] std::uint32_t row_count() const noexcept {
    return static_cast<std::uint32_t>(storage_.ids.size());
  }
  [[nodiscard]] ArrayView<const std::uint32_t> ids() const noexcept {
    return storage_.ids;
  }
  [[nodiscard]] ArrayView<const std::uint8_t> small_codes_row_major() const
      noexcept {
    return storage_.small_codes;
  }
  [[nodiscard]] ArrayView<const float> coefficients_row_major() const
      noexcept {
    return storage_.coefficients;
  }
  [[nodiscard]] ColumnMajorMatrixF32View exact_vectors() const noexcept {
    return {storage_.exact.data(), storage_.dimension, row_count()};
  }
  [[nodiscard]] std::vector<float> TakeExactVectors() && noexcept {
    return std::move(storage_.exact);
  }

 private:
  template <int N>
  friend class BaseListReader;

  detail::BaseListCidStorage storage_;
};

// Read-only mapped owner for the four list-major base shards. BaseListStore
// validates every complete extent before mapping; LoadCid decodes only one
// bounded CID and is safe to call concurrently for distinct outputs.
template <int M>
class BaseListReader final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<BaseListReader<M>> Open(
      const std::filesystem::path& directory);

  [[nodiscard]] const BaseListStore<M>& store() const noexcept {
    return store_;
  }
  [[nodiscard]] StatusOr<LoadedBaseListCid<M>> LoadCid(
      std::uint32_t cid) const;
  [[nodiscard]] Status LoadCid(
      std::uint32_t cid,
      LoadedBaseListCid<M>* output) const;
  [[nodiscard]] StatusOr<LoadedBaseListSupportCid<M>> LoadCidSupport(
      std::uint32_t cid) const;
  [[nodiscard]] Status LoadCidSupport(
      std::uint32_t cid,
      LoadedBaseListSupportCid<M>* output) const;
  // Passing both support outputs as nullptr gathers only exact vectors;
  // stable-ID membership is still validated.
  [[nodiscard]] Status GatherCidByStableIds(
      std::uint32_t cid,
      ArrayView<const std::uint64_t> stable_ids,
      std::vector<float>* exact_vectors,
      std::vector<std::uint8_t>* small_codes,
      std::vector<float>* coefficients) const;

 private:
  BaseListStore<M> store_;
  detail::BaseListReaderStorage storage_;
};

using RuntimeLoadedBaseListCid = LoadedBaseListCid<0>;
using RuntimeLoadedBaseListSupportCid = LoadedBaseListSupportCid<0>;
using RuntimeBaseListReader = BaseListReader<0>;

extern template class BaseListReader<0>;
extern template class LoadedBaseListCid<0>;
extern template class LoadedBaseListSupportCid<0>;
extern template class LoadedBaseListCid<5>;
extern template class LoadedBaseListCid<10>;
extern template class LoadedBaseListSupportCid<5>;
extern template class LoadedBaseListSupportCid<10>;
extern template class BaseListReader<5>;
extern template class BaseListReader<10>;

}  // namespace stlqpp
