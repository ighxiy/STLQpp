#include "stlqpp/index/emit/base_list_reader.h"

#include "stlqpp/core/numeric/little_endian.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {

StatusOr<detail::BaseListReaderStorage> detail::BaseListReaderStorage::Open(
    const BaseListMetadata& metadata) {
  BaseListReaderStorage output;
  auto ids_result = ReadOnlyMappedFile::Open(metadata.ids_path());
  if (!ids_result.ok()) {
    return ids_result.status();
  }
  auto codes_result =
      ReadOnlyMappedFile::Open(metadata.codes_path());
  if (!codes_result.ok()) {
    return codes_result.status();
  }
  auto coefficients_result =
      ReadOnlyMappedFile::Open(metadata.coefficients_path());
  if (!coefficients_result.ok()) {
    return coefficients_result.status();
  }
  if (metadata.has_raw_payload) {
    auto raw_result = ReadOnlyMappedFile::Open(metadata.raw_path());
    if (!raw_result.ok()) {
      return raw_result.status();
    }
    output.raw_file = std::move(raw_result).value();
  }
  output.ids_file = std::move(ids_result).value();
  output.codes_file = std::move(codes_result).value();
  output.coefficients_file = std::move(coefficients_result).value();
  return output;
}

template <int M>
StatusOr<BaseListReader<M>> BaseListReader<M>::Open(
    const std::filesystem::path& directory) {
  auto store = BaseListStore<M>::Open(directory);
  if (!store.ok()) return store.status();
  auto storage = detail::BaseListReaderStorage::Open(store.value().metadata_);
  if (!storage.ok()) return storage.status();
  BaseListReader<M> output;
  output.store_ = std::move(store).value();
  output.storage_ = std::move(storage).value();
  return output;
}

template <int M>
StatusOr<LoadedBaseListSupportCid<M>> BaseListReader<M>::LoadCidSupport(
    std::uint32_t cid) const {
  LoadedBaseListSupportCid<M> output;
  if (const Status status = LoadCidSupport(cid, &output); !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
Status BaseListReader<M>::LoadCidSupport(
    std::uint32_t cid,
    LoadedBaseListSupportCid<M>* output) const {
  if (output == nullptr) {
    return Status::InvalidArgument(
        "base list support load requires output storage");
  }
  if (const Status status = storage_.LoadCidSupportPayload<M>(
          store_.metadata_, cid, &output->storage_.ids,
          &output->storage_.small_codes, &output->storage_.coefficients);
      !status.ok()) {
    return status;
  }
  output->storage_.cid = cid;
  return Status::Ok();
}

template <int M>
Status detail::BaseListReaderStorage::LoadCidSupportPayload(
    const BaseListMetadata& metadata, std::uint32_t cid,
    std::vector<std::uint32_t>* ids,
    std::vector<std::uint8_t>* small_codes,
    std::vector<float>* coefficients) const {
  const std::uint32_t m = M == 0 ? metadata.m : M;
  if (cid >= metadata.shape.nlist) {
    return Status::InvalidArgument("base list CID is outside native IVF");
  }
  const std::uint64_t first = metadata.offsets[cid];
  const std::uint64_t end = metadata.offsets[cid + 1U];
  const std::uint64_t rows64 = end - first;
  if (rows64 > std::numeric_limits<std::uint32_t>::max() ||
      rows64 > std::numeric_limits<std::size_t>::max() / m) {
    return Status::ResourceExhausted(
        "one base list exceeds the bounded host row domain");
  }
  const std::size_t rows = static_cast<std::size_t>(rows64);
  ids->resize(rows);
  small_codes->resize(rows * (m - 1U));
  coefficients->resize(rows * m);

  const std::uint8_t* encoded_ids = rows == 0U
      ? nullptr
      : ids_file.data() + first * sizeof(std::uint32_t);
  for (std::size_t row = 0U; row < rows; ++row) {
    (*ids)[row] = LoadLittleEndianU32(
        encoded_ids + row * sizeof(std::uint32_t));
  }
  if (!small_codes->empty()) {
    std::memcpy(small_codes->data(),
                codes_file.data() + first * (m - 1U),
                small_codes->size());
  }
  const std::uint8_t* encoded_coefficients = coefficients->empty()
      ? nullptr
      : coefficients_file.data() + first * m * sizeof(float);
  for (std::size_t index = 0U; index < coefficients->size(); ++index) {
    const float value = LoadLittleEndianF32(
        encoded_coefficients + index * sizeof(float));
    if (!std::isfinite(value)) {
      return Status::DataLoss(
          "base list contains a non-finite basic coefficient");
    }
    (*coefficients)[index] = value;
  }

  return Status::Ok();
}

template <int M>
StatusOr<LoadedBaseListCid<M>> BaseListReader<M>::LoadCid(
    std::uint32_t cid) const {
  LoadedBaseListCid<M> output;
  if (const Status status = LoadCid(cid, &output); !status.ok()) {
    return status;
  }
  return output;
}

Status detail::BaseListReaderStorage::LoadRawCid(
    const BaseListMetadata& metadata, std::uint32_t cid,
    BaseListCidStorage* output) const {
  const std::uint32_t dimension = metadata.shape.dimension;
  const std::size_t rows = output->ids.size();
  if (dimension != 0U &&
      rows > std::numeric_limits<std::size_t>::max() / dimension) {
    return Status::ResourceExhausted(
        "one base list raw payload exceeds the bounded host row domain");
  }
  output->cid = cid;
  output->dimension = dimension;
  output->exact.resize(rows * dimension);

  const std::uint64_t first = metadata.offsets[cid];
  const std::uint64_t raw_row_bytes =
      metadata.shape.raw_bytes_per_vector();
  const std::uint8_t* raw = output->exact.empty()
      ? nullptr
      : raw_file.data() + first * raw_row_bytes;
  if (metadata.shape.vector_scalar == BaseVectorScalar::kU8) {
    for (std::size_t index = 0U; index < output->exact.size(); ++index) {
      output->exact[index] = static_cast<float>(raw[index]);
    }
  } else {
    if (!output->exact.empty()) {
      std::memcpy(output->exact.data(), raw,
                  output->exact.size() * sizeof(float));
    }
  }
  return Status::Ok();
}

template <int M>
Status BaseListReader<M>::LoadCid(
    std::uint32_t cid, LoadedBaseListCid<M>* output) const {
  if (output == nullptr) {
    return Status::InvalidArgument(
        "base list load requires output storage");
  }
  if (!store_.has_raw_payload()) {
    return Status::FailedPrecondition(
        "base list raw payload was intentionally removed");
  }
  auto& payload = output->storage_;
  if (const Status status = storage_.LoadCidSupportPayload<M>(
          store_.metadata_, cid, &payload.ids, &payload.small_codes,
          &payload.coefficients); !status.ok()) {
    return status;
  }
  return storage_.LoadRawCid(store_.metadata_, cid, &payload);
}

namespace {

struct PreparedBaseListGather final {
  std::uint64_t first;
  std::uint32_t rows;
  std::uint32_t dimension;
  const std::uint8_t* encoded_ids;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> destinations;
};

StatusOr<PreparedBaseListGather> PrepareBaseListGather(
    const detail::BaseListMetadata& metadata,
    const detail::BaseListReaderStorage& storage,
    std::uint32_t cid, ArrayView<const std::uint64_t> stable_ids,
    std::vector<float>* exact_vectors,
    std::vector<std::uint8_t>* small_codes,
    std::vector<float>* coefficients) {
  const std::uint32_t m = metadata.m;
  if (exact_vectors == nullptr ||
      (small_codes == nullptr) != (coefficients == nullptr)) {
    return Status::InvalidArgument(
        "base-list stable-ID gather requires output storage");
  }
  const bool materialize_support = small_codes != nullptr;
  if (!metadata.has_raw_payload) {
    return Status::FailedPrecondition(
        "base list raw payload was intentionally removed");
  }
  if (cid >= metadata.shape.nlist) {
    return Status::InvalidArgument("base list CID is outside native IVF");
  }
  const std::uint64_t first = metadata.offsets[cid];
  const std::uint64_t end = metadata.offsets[cid + 1U];
  const std::uint64_t rows64 = end - first;
  const std::uint32_t dimension = metadata.shape.dimension;
  if (rows64 != stable_ids.size()) {
    return Status::FailedPrecondition(
        "base-list stable-ID gather extent changed");
  }
  if (rows64 > std::numeric_limits<std::uint32_t>::max() ||
      (dimension != 0U &&
       rows64 > std::numeric_limits<std::size_t>::max() / dimension)) {
    return Status::ResourceExhausted(
        "base-list stable-ID gather exceeds the host row domain");
  }
  const std::uint32_t rows = static_cast<std::uint32_t>(rows64);
  const std::uint8_t* encoded_ids = rows == 0U
      ? nullptr
      : storage.ids_file.data() + first * sizeof(std::uint32_t);
  const auto source_id = [&](std::uint32_t row) noexcept {
    return LoadLittleEndianU32(
        encoded_ids + static_cast<std::size_t>(row) *
                          sizeof(std::uint32_t));
  };
  for (std::uint32_t row = 1U; row < rows; ++row) {
    if (source_id(row - 1U) >= source_id(row)) {
      return Status::DataLoss(
          "base list CID stable IDs are not strictly ordered");
    }
  }

  exact_vectors->resize(static_cast<std::size_t>(rows) * dimension);
  if (materialize_support) {
    small_codes->resize(static_cast<std::size_t>(rows) * (m - 1U));
    coefficients->resize(static_cast<std::size_t>(rows) * m);
  }
  // Forest rows are BFS ordered, while the base list is stable-ID ordered.
  // Sort only the bounded CID permutation: source payloads are then read
  // sequentially and scattered to their BFS destinations.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> destinations;
  destinations.reserve(rows);
  for (std::uint32_t destination = 0U; destination < rows; ++destination) {
    if (stable_ids[destination] >
        std::numeric_limits<std::uint32_t>::max()) {
      return Status::FailedPrecondition(
          "base-list stable-ID gather membership changed");
    }
    destinations.emplace_back(
        static_cast<std::uint32_t>(stable_ids[destination]), destination);
  }
  std::sort(destinations.begin(), destinations.end());
  return PreparedBaseListGather{
      first, rows, dimension, encoded_ids, std::move(destinations)};
}

}  // namespace

template <int M>
Status BaseListReader<M>::GatherCidByStableIds(
    std::uint32_t cid,
    ArrayView<const std::uint64_t> stable_ids,
    std::vector<float>* exact_vectors,
    std::vector<std::uint8_t>* small_codes,
    std::vector<float>* coefficients) const {
  auto prepared = PrepareBaseListGather(
      store_.metadata_, storage_, cid, stable_ids, exact_vectors,
      small_codes, coefficients);
  if (!prepared.ok()) return prepared.status();
  const std::uint32_t m = store_.m();
  const bool materialize_support = small_codes != nullptr;
  const auto& gather = prepared.value();
  const std::uint64_t first = gather.first;
  const std::uint32_t rows = gather.rows;
  const std::uint32_t dimension = gather.dimension;
  const auto& destinations = gather.destinations;
  const auto source_id = [&](std::uint32_t row) noexcept {
    return LoadLittleEndianU32(
        gather.encoded_ids + static_cast<std::size_t>(row) * sizeof(std::uint32_t));
  };
  const std::uint64_t raw_row_bytes =
      store_.shape().raw_bytes_per_vector();
  for (std::uint32_t source = 0U; source < rows; ++source) {
    const auto [wanted, destination] = destinations[source];
    if (source_id(source) != wanted) {
      return Status::FailedPrecondition(
          "base-list stable-ID gather membership changed");
    }

    float* exact = exact_vectors->data() +
        static_cast<std::size_t>(destination) * dimension;
    const std::uint8_t* raw = storage_.raw_file.data() +
        (first + source) * raw_row_bytes;
    if (store_.shape().vector_scalar == BaseVectorScalar::kU8) {
      for (std::uint32_t axis = 0U; axis < dimension; ++axis) {
        exact[axis] = static_cast<float>(raw[axis]);
      }
    } else {
      std::memcpy(exact, raw,
                  static_cast<std::size_t>(dimension) * sizeof(float));
    }

    if (materialize_support) {
      std::memcpy(
          small_codes->data() +
              static_cast<std::size_t>(destination) * (m - 1U),
          storage_.codes_file.data() + (first + source) * (m - 1U), m - 1U);
      const std::uint8_t* encoded_coefficients = storage_.coefficients_file.data() +
          (first + source) * m * sizeof(float);
      float* output_coefficients = coefficients->data() +
          static_cast<std::size_t>(destination) * m;
      for (std::uint32_t layer = 0U;
           layer < static_cast<std::uint32_t>(m); ++layer) {
        const float value = LoadLittleEndianF32(
            encoded_coefficients +
            static_cast<std::size_t>(layer) * sizeof(float));
        if (!std::isfinite(value)) {
          return Status::DataLoss(
              "base list contains a non-finite basic coefficient");
        }
        output_coefficients[layer] = value;
      }
    }
  }
  return Status::Ok();
}

template class BaseListReader<0>;
template class LoadedBaseListCid<0>;
template class LoadedBaseListSupportCid<0>;
template class LoadedBaseListCid<5>;
template class LoadedBaseListCid<12>;
template class LoadedBaseListCid<6>;
template class LoadedBaseListCid<10>;
template class LoadedBaseListSupportCid<5>;
template class LoadedBaseListSupportCid<12>;
template class LoadedBaseListSupportCid<6>;
template class LoadedBaseListSupportCid<10>;
template class BaseListReader<5>;
template class BaseListReader<12>;
template class BaseListReader<6>;
template class BaseListReader<10>;

}  // namespace stlqpp
