#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/update/round_update_transcript.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/training/update/detail/transcript_columns.h"

namespace stlqpp {
namespace {

using detail::TranscriptDecodeColumns;
using detail::TranscriptSupportView;
using detail::TranscriptTopologyView;

constexpr std::array<std::uint8_t, 8U> kMagic = {'S', 'T', 'L', 'Q',
                                                 'R', 'U', 'T', '2'};
constexpr std::uint32_t kVersion = 2U;
constexpr std::uint32_t kHeaderBytes = 32U;
constexpr std::uint32_t kRealParentBit = 0x80000000U;

StatusOr<std::uint64_t> RecordBytes(std::uint32_t real_count,
                                    std::uint32_t virtual_count,
                                    std::uint32_t m) {
  const std::uint64_t kRealBytes = sizeof(std::uint64_t) +
                                   3U * sizeof(std::uint32_t) + (m - 1U) +
                                   m * sizeof(float);
  const std::uint64_t kVirtualBytes = (m - 1U) + m * sizeof(float);
  const std::uint64_t real_bytes =
      static_cast<std::uint64_t>(real_count) * kRealBytes;
  const std::uint64_t virtual_bytes =
      static_cast<std::uint64_t>(virtual_count) * kVirtualBytes;
  if (real_bytes > std::numeric_limits<std::uint64_t>::max() - virtual_bytes) {
    return Status::ResourceExhausted(
        "round update transcript record overflows u64");
  }
  return real_bytes + virtual_bytes;
}

std::vector<std::uint8_t> EncodeHeader(std::uint32_t m, std::uint32_t dimension,
                                       std::uint32_t nlist) {
  std::vector<std::uint8_t> bytes(kHeaderBytes, 0U);
  std::memcpy(bytes.data(), kMagic.data(), kMagic.size());
  StoreLittleEndianU32(kVersion, bytes.data() + 8U);
  StoreLittleEndianU32(kHeaderBytes, bytes.data() + 12U);
  StoreLittleEndianU32(m, bytes.data() + 16U);
  StoreLittleEndianU32(dimension, bytes.data() + 20U);
  StoreLittleEndianU32(nlist, bytes.data() + 24U);
  return bytes;
}

Status ValidateHeader(const ReadOnlyMappedFile& file, std::uint32_t m,
                      std::uint32_t dimension, std::uint32_t nlist) {
  if (file.size() < kHeaderBytes ||
      std::memcmp(file.data(), kMagic.data(), kMagic.size()) != 0 ||
      LoadLittleEndianU32(file.data() + 8U) != kVersion ||
      LoadLittleEndianU32(file.data() + 12U) != kHeaderBytes ||
      LoadLittleEndianU32(file.data() + 16U) != m ||
      LoadLittleEndianU32(file.data() + 20U) != dimension ||
      LoadLittleEndianU32(file.data() + 24U) != nlist ||
      LoadLittleEndianU32(file.data() + 28U) != 0U) {
    return Status::DataLoss("round update transcript header is invalid");
  }
  return Status::Ok();
}

Status ValidateCluster(
    std::uint32_t cid, std::uint32_t topology_dimension,
    std::uint32_t virtual_count, ArrayView<const LinkageNodeRole> roles,
    ArrayView<const std::optional<LinkageParentReference>> parents,
    ArrayView<const std::uint32_t> depths, TranscriptSupportView supports,
    ArrayView<const float> coefficients,
    ArrayView<const std::uint64_t> real_sample_ids,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients, std::uint32_t expected_cid,
    std::uint32_t dimension, std::uint32_t nlist, std::uint32_t m) {
  const std::size_t rows = roles.size();
  if (cid != expected_cid || topology_dimension != dimension ||
      real_sample_ids.size() != rows || parents.size() != rows ||
      depths.size() != rows || supports.size() != rows ||
      coefficients.size() != rows * m ||
      virtual_small_codes.size() != virtual_count * (m - 1U) ||
      virtual_coefficients.size() != virtual_count * m ||
      rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "round update transcript cluster has an invalid shape");
  }
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const LinkageNodeRole role = roles[row];
    const auto& parent = parents[row];
    if (role == LinkageNodeRole::kRoot) {
      if (parent.has_value() || depths[row] != 0U ||
          supports.first(row) != expected_cid) {
        return Status::InvalidArgument(
            "round update transcript root is inconsistent");
      }
    } else if (role != LinkageNodeRole::kLinked || !parent.has_value() ||
               depths[row] == 0U || supports.first(row) >= 256U) {
      return Status::InvalidArgument(
          "round update transcript linked row is inconsistent");
    } else if (parent->kind == LinkageParentKind::kVirtual) {
      if (parent->local_id >= virtual_count || depths[row] != 1U) {
        return Status::InvalidArgument(
            "round update transcript virtual parent is inconsistent");
      }
    } else if (parent->local_id >= rows || parent->local_id == row ||
               depths[parent->local_id] + 1U != depths[row]) {
      return Status::InvalidArgument(
          "round update transcript real parent is inconsistent");
    }
  }
  for (const float coefficient : coefficients) {
    if (!std::isfinite(coefficient)) {
      return Status::InvalidArgument(
          "round update transcript real coefficient is non-finite");
    }
  }
  for (const float coefficient : virtual_coefficients) {
    if (!std::isfinite(coefficient)) {
      return Status::InvalidArgument(
          "round update transcript virtual coefficient is non-finite");
    }
  }
  static_cast<void>(nlist);
  return Status::Ok();
}

Status ValidateRealSupportAndCoefficients(
    const TranscriptTopologyView& topology, std::uint32_t cid) {
  const std::uint32_t m = topology.m;
  const std::size_t rows = topology.roles.size();
  if (topology.supports.size() != rows ||
      topology.coefficients.size() != rows * m) {
    return Status::InvalidArgument(
        "round update transcript real support streams have invalid shape");
  }
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::uint32_t first = topology.supports.first(row);
    if ((topology.roles[row] == LinkageNodeRole::kRoot && first != cid) ||
        (topology.roles[row] == LinkageNodeRole::kLinked && first >= 256U)) {
      return Status::InvalidArgument(
          "round update transcript real support is inconsistent");
    }
  }
  for (const float coefficient : topology.coefficients) {
    if (!std::isfinite(coefficient)) {
      return Status::InvalidArgument(
          "round update transcript real coefficient is non-finite");
    }
  }
  return Status::Ok();
}

}  // namespace

struct detail::RoundUpdateTranscriptStorage final {
  StagingDirectory staging;
  ReadOnlyMappedFile records;
  std::uint32_t m = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::vector<RoundUpdateTranscriptEntry> entries;
  StatusOr<RoundUpdateTranscriptEntry> Find(std::uint32_t cid) const;
  Status Decode(const RoundUpdateTranscriptEntry& entry,
                TranscriptDecodeColumns output,
                TranscriptSupportView support_view,
                ArrayView<std::uint64_t> real_sample_ids,
                ArrayView<std::uint8_t> virtual_small_codes,
                ArrayView<float> virtual_coefficients) const;
};

struct detail::RoundUpdateTranscriptWriterStorage final {
  StagingDirectory staging;
  SequentialFileWriter records;
  std::filesystem::path record_path;
  std::uint32_t m = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::vector<RoundUpdateTranscriptEntry> entries;
  static StatusOr<std::unique_ptr<detail::RoundUpdateTranscriptWriterStorage>>
  Create(const std::filesystem::path&, std::uint32_t, std::uint32_t,
         std::uint32_t);
  Status Append(const TranscriptTopologyView& topology,
                ArrayView<const std::uint64_t> real_sample_ids,
                ArrayView<const std::uint8_t> virtual_small_codes,
                ArrayView<const float> virtual_coefficients,
                bool forest_validated);
  StatusOr<std::unique_ptr<detail::RoundUpdateTranscriptStorage>> Finish();
};

template <int M>
RoundUpdateTranscript<M>::RoundUpdateTranscript(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
RoundUpdateTranscript<M>::~RoundUpdateTranscript() = default;

template <int M>
RoundUpdateTranscript<M>::RoundUpdateTranscript(
    RoundUpdateTranscript&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
RoundUpdateTranscript<M>& RoundUpdateTranscript<M>::operator=(
    RoundUpdateTranscript&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
std::uint32_t RoundUpdateTranscript<M>::m() const noexcept {
  return state_->m;
}

template <int M>
std::uint32_t RoundUpdateTranscript<M>::dimension() const noexcept {
  return state_->dimension;
}

template <int M>
std::uint32_t RoundUpdateTranscript<M>::nlist() const noexcept {
  return state_->nlist;
}

template <int M>
StatusOr<RoundUpdateTranscriptCluster<M>> RoundUpdateTranscript<M>::LoadCluster(
    std::uint32_t cid) const {
  RoundUpdateTranscriptCluster<M> output;
  if (const Status status = LoadCluster(cid, &output); !status.ok()) {
    return status;
  }
  return output;
}

StatusOr<RoundUpdateTranscriptEntry> detail::RoundUpdateTranscriptStorage::Find(
    std::uint32_t cid) const {
  if (cid >= entries.size()) {
    return Status::InvalidArgument(
        "round update transcript CID is out of range");
  }
  const RoundUpdateTranscriptEntry& entry = entries[cid];
  auto expected = RecordBytes(entry.real_count, entry.virtual_count, m);
  if (!expected.ok() || expected.value() != entry.bytes ||
      entry.offset > records.size() ||
      entry.bytes > records.size() - entry.offset) {
    return Status::DataLoss("round update transcript entry is invalid");
  }

  return entry;
}

Status detail::RoundUpdateTranscriptStorage::Decode(
    const RoundUpdateTranscriptEntry& entry, TranscriptDecodeColumns output,
    TranscriptSupportView support_view,
    ArrayView<std::uint64_t> real_sample_ids,
    ArrayView<std::uint8_t> virtual_small_codes,
    ArrayView<float> virtual_coefficients) const {
  const std::uint8_t* bytes = records.data() + entry.offset;
  std::size_t cursor = 0U;
  for (std::uint64_t& stable_id : real_sample_ids) {
    stable_id = LoadLittleEndianU64(bytes + cursor);
    cursor += sizeof(std::uint64_t);
  }
  for (std::uint32_t row = 0U; row < entry.real_count; ++row) {
    const std::uint32_t encoded = LoadLittleEndianU32(bytes + cursor);
    cursor += sizeof(std::uint32_t);
    if (encoded == 0U) {
      output.roles[row] = LinkageNodeRole::kRoot;
      output.parents[row] = std::nullopt;
    } else if ((encoded & kRealParentBit) != 0U) {
      output.roles[row] = LinkageNodeRole::kLinked;
      output.parents[row] = LinkageParentReference{
          LinkageParentKind::kReal, (encoded & ~kRealParentBit) - 1U};
    } else {
      output.roles[row] = LinkageNodeRole::kLinked;
      output.parents[row] =
          LinkageParentReference{LinkageParentKind::kVirtual, encoded - 1U};
    }
  }
  for (std::uint32_t& depth : output.depths) {
    depth = LoadLittleEndianU32(bytes + cursor);
    cursor += sizeof(std::uint32_t);
  }
  for (std::uint32_t row = 0; row < entry.real_count; ++row) {
    output.supports.set_first(row, LoadLittleEndianU32(bytes + cursor));
    cursor += sizeof(std::uint32_t);
  }
  for (std::uint32_t row = 0; row < entry.real_count; ++row) {
    std::memcpy(output.supports.trailing(row), bytes + cursor, m - 1U);
    cursor += m - 1U;
  }
  for (float& coefficient : output.coefficients) {
    coefficient = LoadLittleEndianF32(bytes + cursor);
    cursor += sizeof(float);
  }
  if (!virtual_small_codes.empty()) {
    std::memcpy(virtual_small_codes.data(), bytes + cursor,
                virtual_small_codes.size());
  }
  cursor += virtual_small_codes.size();
  for (float& coefficient : virtual_coefficients) {
    coefficient = LoadLittleEndianF32(bytes + cursor);
    cursor += sizeof(float);
  }

  if (const Status status = ValidateCluster(
          entry.cid, dimension, entry.virtual_count,
          {output.roles.data(), output.roles.size()},
          {output.parents.data(), output.parents.size()},
          {output.depths.data(), output.depths.size()}, support_view,
          {output.coefficients.data(), output.coefficients.size()},
          {real_sample_ids.data(), real_sample_ids.size()},
          {virtual_small_codes.data(), virtual_small_codes.size()},
          {virtual_coefficients.data(), virtual_coefficients.size()}, entry.cid,
          dimension, nlist, m);
      !status.ok()) {
    return Status::DataLoss(
        std::string("round update transcript record is invalid: ") +
        status.message());
  }
  return Status::Ok();
}

template <int M>
Status RoundUpdateTranscript<M>::LoadCluster(
    std::uint32_t cid, RoundUpdateTranscriptCluster<M>* output) const {
  if (output == nullptr)
    return Status::InvalidArgument(
        "round update transcript load requires output storage");
  auto found = state_->Find(cid);
  if (!found.ok()) return found.status();
  const auto& entry = found.value();
  const std::uint32_t m = state_->m;
  output->m = m;
  output->cid = cid;
  output->real_sample_ids.resize(entry.real_count);
  output->roles.resize(entry.real_count);
  output->parents.resize(entry.real_count);
  output->depths.resize(entry.real_count);
  output->supports.resize(entry.real_count);
  output->coefficients.resize(static_cast<std::size_t>(entry.real_count) * m);
  output->virtual_small_codes.resize(
      static_cast<std::size_t>(entry.virtual_count) * (m - 1U));
  output->virtual_coefficients.resize(
      static_cast<std::size_t>(entry.virtual_count) * m);

  return state_->Decode(
      entry,
      {output->roles, output->parents, output->depths,
       detail::TranscriptSupportOutput(output->supports), output->coefficients},
      TranscriptSupportView(output->supports), output->real_sample_ids,
      output->virtual_small_codes, output->virtual_coefficients);
}

template <int M>
RoundUpdateTranscriptWriter<M>::RoundUpdateTranscriptWriter(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
RoundUpdateTranscriptWriter<M>::~RoundUpdateTranscriptWriter() = default;

template <int M>
RoundUpdateTranscriptWriter<M>::RoundUpdateTranscriptWriter(
    RoundUpdateTranscriptWriter&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
RoundUpdateTranscriptWriter<M>& RoundUpdateTranscriptWriter<M>::operator=(
    RoundUpdateTranscriptWriter&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

StatusOr<std::unique_ptr<detail::RoundUpdateTranscriptWriterStorage>>
detail::RoundUpdateTranscriptWriterStorage::Create(
    const std::filesystem::path& transient_destination, std::uint32_t dimension,
    std::uint32_t nlist, std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers)
    return Status::InvalidArgument("transcript M is invalid");
  if (transient_destination.empty() || dimension == 0U || nlist == 0U) {
    return Status::InvalidArgument(
        "round update transcript request is incomplete");
  }
  auto staging = StagingDirectory::CreateTransient(transient_destination);
  if (!staging.ok()) {
    return staging.status();
  }
  auto state = std::make_unique<detail::RoundUpdateTranscriptWriterStorage>();
  state->staging = std::move(staging).value();
  state->record_path = state->staging.path() / "forest_support.bin";
  state->m = m;
  state->dimension = dimension;
  state->nlist = nlist;
  state->entries.reserve(nlist);
  auto records = SequentialFileWriter::Create(state->record_path);
  if (!records.ok()) {
    return records.status();
  }
  state->records = std::move(records).value();
  const std::vector<std::uint8_t> header = EncodeHeader(m, dimension, nlist);
  if (const Status status = state->records.Append(header); !status.ok()) {
    return status;
  }
  return state;
}

template <int M>
StatusOr<RoundUpdateTranscriptWriter<M>> RoundUpdateTranscriptWriter<M>::Create(
    const std::filesystem::path& transient_destination, std::uint32_t dimension,
    std::uint32_t nlist, std::uint32_t m) {
  if constexpr (M != 0) {
    if (m != M)
      return Status::InvalidArgument("transcript M is invalid");
  }
  auto state = State::Create(transient_destination, dimension, nlist, m);
  if (!state.ok()) return state.status();
  return RoundUpdateTranscriptWriter<M>(std::move(state).value());
}

template <int M>
Status RoundUpdateTranscriptWriter<M>::Append(
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint64_t> real_sample_ids,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients) {
  return AppendImpl(topology, real_sample_ids, virtual_small_codes,
                    virtual_coefficients, false);
}

template <int M>
Status RoundUpdateTranscriptWriter<M>::AppendAfterForestValidation(
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint64_t> real_sample_ids,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients) {
  return AppendImpl(topology, real_sample_ids, virtual_small_codes,
                    virtual_coefficients, true);
}

Status detail::RoundUpdateTranscriptWriterStorage::Append(
    const TranscriptTopologyView& topology,
    ArrayView<const std::uint64_t> real_sample_ids,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients, bool forest_validated) {
  const std::uint32_t cid = static_cast<std::uint32_t>(entries.size());
  if (cid >= nlist) {
    return Status::FailedPrecondition(
        "round update transcript received too many CIDs");
  }

  if (topology.m != m)
    return Status::InvalidArgument("transcript topology M mismatch");
  if (forest_validated) {
    if (const Status status = ValidateRealSupportAndCoefficients(topology, cid);
        !status.ok()) {
      return status;
    }
#ifndef NDEBUG
    assert(ValidateCluster(topology.cid, topology.dimension,
                           topology.virtual_count, topology.roles,
                           topology.parents, topology.depths, topology.supports,
                           topology.coefficients, real_sample_ids,
                           virtual_small_codes, virtual_coefficients, cid,
                           dimension, nlist, m)
               .ok());
#endif
  } else {
    if (const Status status = ValidateCluster(
            topology.cid, topology.dimension, topology.virtual_count,
            topology.roles, topology.parents, topology.depths,
            topology.supports, topology.coefficients, real_sample_ids,
            virtual_small_codes, virtual_coefficients, cid, dimension, nlist,
            m);
        !status.ok()) {
      return status;
    }
  }
  const std::uint32_t real_count =
      static_cast<std::uint32_t>(topology.roles.size());
  auto byte_count = RecordBytes(real_count, topology.virtual_count, m);
  if (!byte_count.ok() ||
      byte_count.value() > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "round update transcript record is not addressable");
  }
  std::vector<std::uint8_t> record(
      static_cast<std::size_t>(byte_count.value()));
  std::size_t cursor = 0U;
  for (const std::uint64_t stable_id : real_sample_ids) {
    StoreLittleEndianU64(stable_id, record.data() + cursor);
    cursor += sizeof(std::uint64_t);
  }
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    std::uint32_t encoded = 0U;
    if (topology.roles[row] == LinkageNodeRole::kLinked) {
      const LinkageParentReference parent = *topology.parents[row];
      encoded = parent.local_id + 1U;
      if (parent.kind == LinkageParentKind::kReal) {
        encoded |= kRealParentBit;
      }
    }
    StoreLittleEndianU32(encoded, record.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  for (const std::uint32_t depth : topology.depths) {
    StoreLittleEndianU32(depth, record.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  for (std::size_t row = 0; row < topology.supports.size(); ++row) {
    StoreLittleEndianU32(topology.supports.first(row), record.data() + cursor);
    cursor += sizeof(std::uint32_t);
  }
  for (std::size_t row = 0; row < topology.supports.size(); ++row) {
    std::memcpy(record.data() + cursor, topology.supports.trailing(row),
                m - 1U);
    cursor += m - 1U;
  }
  for (const float coefficient : topology.coefficients) {
    StoreLittleEndianF32(coefficient, record.data() + cursor);
    cursor += sizeof(float);
  }
  if (!virtual_small_codes.empty()) {
    std::memcpy(record.data() + cursor, virtual_small_codes.data(),
                virtual_small_codes.size());
  }
  cursor += virtual_small_codes.size();
  for (const float coefficient : virtual_coefficients) {
    StoreLittleEndianF32(coefficient, record.data() + cursor);
    cursor += sizeof(float);
  }
  const std::uint64_t offset = records.size();
  if (const Status status = records.Append(record); !status.ok()) {
    return status;
  }
  entries.push_back({
      .cid = cid,
      .real_count = real_count,
      .virtual_count = topology.virtual_count,
      .offset = offset,
      .bytes = byte_count.value(),
  });
  return Status::Ok();
}

template <int M>
Status RoundUpdateTranscriptWriter<M>::AppendImpl(
    const ClusterTopologyResult<M>& topology,
    ArrayView<const std::uint64_t> real_sample_ids,
    ArrayView<const std::uint8_t> virtual_small_codes,
    ArrayView<const float> virtual_coefficients, bool forest_validated) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "round update transcript is already finished");
  }
  return state_->Append(detail::ViewTranscriptTopology(topology),
                        real_sample_ids, virtual_small_codes,
                        virtual_coefficients, forest_validated);
}

StatusOr<std::unique_ptr<detail::RoundUpdateTranscriptStorage>>
detail::RoundUpdateTranscriptWriterStorage::Finish() {
  if (entries.size() != nlist) {
    return Status::FailedPrecondition(
        "round update transcript has not consumed every CID");
  }
  if (const Status status = records.Close(); !status.ok()) {
    return status;
  }
  auto mapped = ReadOnlyMappedFile::Open(record_path);
  if (!mapped.ok()) {
    return mapped.status();
  }
  if (const Status status = ValidateHeader(mapped.value(), m, dimension, nlist);
      !status.ok()) {
    return status;
  }
  auto transcript = std::make_unique<detail::RoundUpdateTranscriptStorage>();
  transcript->staging = std::move(staging);
  transcript->records = std::move(mapped).value();
  transcript->m = m;
  transcript->dimension = dimension;
  transcript->nlist = nlist;
  transcript->entries = std::move(entries);
  return transcript;
}

template <int M>
StatusOr<RoundUpdateTranscript<M>> RoundUpdateTranscriptWriter<M>::Finish() {
  if (state_ == nullptr)
    return Status::FailedPrecondition(
        "round update transcript has not consumed every CID");
  auto transcript = state_->Finish();
  if (!transcript.ok()) return transcript.status();
  state_.reset();
  return RoundUpdateTranscript<M>(std::move(transcript).value());
}

template class RoundUpdateTranscript<0>;
template class RoundUpdateTranscript<5>;
template class RoundUpdateTranscript<12>;
template class RoundUpdateTranscript<6>;
template class RoundUpdateTranscript<10>;
template class RoundUpdateTranscriptWriter<0>;
template class RoundUpdateTranscriptWriter<5>;
template class RoundUpdateTranscriptWriter<12>;
template class RoundUpdateTranscriptWriter<6>;
template class RoundUpdateTranscriptWriter<10>;

}  // namespace stlqpp
