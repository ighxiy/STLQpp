#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/update/root_only_initialization_transcript.h"

#include <array>
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

constexpr std::array<std::uint8_t, 8U> kTranscriptMagic = {'S', 'T', 'L', 'Q',
                                                           'R', 'O', 'I', '1'};
constexpr std::uint32_t kTranscriptVersion = 1U;
constexpr std::uint32_t kTranscriptHeaderBytes = 32U;

StatusOr<std::uint64_t> RecordBytes(std::uint32_t real_count, std::uint32_t m) {
  const std::uint64_t kBytesPerRow =
      3U * sizeof(std::uint32_t) + (m - 1U) + m * sizeof(float);
  if (real_count > std::numeric_limits<std::uint64_t>::max() / kBytesPerRow) {
    return Status::ResourceExhausted(
        "root-only initialization transcript record overflows u64");
  }
  return kBytesPerRow * real_count;
}

std::vector<std::uint8_t> EncodeHeader(std::uint32_t m, std::uint32_t dimension,
                                       std::uint32_t nlist) {
  std::vector<std::uint8_t> bytes(kTranscriptHeaderBytes, 0U);
  std::memcpy(bytes.data(), kTranscriptMagic.data(), kTranscriptMagic.size());
  StoreLittleEndianU32(kTranscriptVersion, bytes.data() + 8U);
  StoreLittleEndianU32(kTranscriptHeaderBytes, bytes.data() + 12U);
  StoreLittleEndianU32(m, bytes.data() + 16U);
  StoreLittleEndianU32(dimension, bytes.data() + 20U);
  StoreLittleEndianU32(nlist, bytes.data() + 24U);
  return bytes;
}

Status ValidateHeader(const ReadOnlyMappedFile& file, std::uint32_t m,
                      std::uint32_t dimension, std::uint32_t nlist) {
  if (file.size() < kTranscriptHeaderBytes ||
      std::memcmp(file.data(), kTranscriptMagic.data(),
                  kTranscriptMagic.size()) != 0 ||
      LoadLittleEndianU32(file.data() + 8U) != kTranscriptVersion ||
      LoadLittleEndianU32(file.data() + 12U) != kTranscriptHeaderBytes ||
      LoadLittleEndianU32(file.data() + 16U) != m ||
      LoadLittleEndianU32(file.data() + 20U) != dimension ||
      LoadLittleEndianU32(file.data() + 24U) != nlist ||
      LoadLittleEndianU32(file.data() + 28U) != 0U) {
    return Status::DataLoss(
        "root-only initialization transcript header is invalid");
  }
  return Status::Ok();
}

Status ValidateTopology(const TranscriptTopologyView& topology,
                        std::uint32_t actual_dimension,
                        std::uint32_t virtual_count, std::uint32_t expected_cid,
                        std::uint32_t dimension, std::uint32_t nlist,
                        std::uint32_t m) {
  const std::size_t rows = topology.roles.size();
  if (topology.cid != expected_cid || actual_dimension != dimension ||
      virtual_count != 0U || topology.parents.size() != rows ||
      topology.depths.size() != rows || topology.supports.size() != rows ||
      topology.coefficients.size() != rows * m ||
      rows > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument(
        "root-only initialization transcript cluster has an invalid shape");
  }
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const LinkageNodeRole role = topology.roles[row];
    const auto& parent = topology.parents[row];
    if (topology.supports.first(row) >= nlist) {
      return Status::InvalidArgument(
          "root-only initialization support exceeds C_root");
    }
    if (role == LinkageNodeRole::kRoot) {
      if (parent.has_value() || topology.depths[row] != 0U) {
        return Status::InvalidArgument(
            "root-only initialization root has a parent or depth");
      }
    } else if (role != LinkageNodeRole::kLinked || !parent.has_value() ||
               parent->kind != LinkageParentKind::kReal ||
               parent->local_id >= rows || topology.depths[row] == 0U ||
               topology.depths[parent->local_id] + 1U != topology.depths[row]) {
      return Status::InvalidArgument(
          "root-only initialization link violates causal depth");
    }
    const float* coefficients =
        topology.coefficients.data() + static_cast<std::size_t>(row) * m;
    for (std::uint32_t layer = 0U; layer < m; ++layer) {
      if (!std::isfinite(coefficients[layer])) {
        return Status::InvalidArgument(
            "root-only initialization coefficient is non-finite");
      }
    }
  }
  return Status::Ok();
}

}  // namespace

struct detail::RootOnlyInitializationTranscriptStorage final {
  StagingDirectory staging;
  ReadOnlyMappedFile records;
  std::uint32_t m = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::vector<RootOnlyInitializationTranscriptEntry> entries;
  StatusOr<RootOnlyInitializationTranscriptEntry> Find(std::uint32_t cid) const;
  Status Decode(const RootOnlyInitializationTranscriptEntry& entry,
                TranscriptDecodeColumns output,
                TranscriptSupportView support_view) const;
};

struct detail::RootOnlyInitializationTranscriptWriterStorage final {
  StagingDirectory staging;
  SequentialFileWriter records;
  std::filesystem::path record_path;
  std::uint32_t m = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::vector<RootOnlyInitializationTranscriptEntry> entries;
  static StatusOr<
      std::unique_ptr<detail::RootOnlyInitializationTranscriptWriterStorage>>
  Create(const std::filesystem::path&, std::uint32_t, std::uint32_t,
         std::uint32_t);
  Status Append(const TranscriptTopologyView& topology);
  StatusOr<std::unique_ptr<detail::RootOnlyInitializationTranscriptStorage>>
  Finish();
};

template <int M>
RootOnlyInitializationTranscript<M>::RootOnlyInitializationTranscript(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
RootOnlyInitializationTranscript<M>::~RootOnlyInitializationTranscript() =
    default;

template <int M>
RootOnlyInitializationTranscript<M>::RootOnlyInitializationTranscript(
    RootOnlyInitializationTranscript&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
RootOnlyInitializationTranscript<M>&
RootOnlyInitializationTranscript<M>::operator=(
    RootOnlyInitializationTranscript&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
std::uint32_t RootOnlyInitializationTranscript<M>::m() const noexcept {
  return state_->m;
}

template <int M>
std::uint32_t RootOnlyInitializationTranscript<M>::dimension() const noexcept {
  return state_->dimension;
}

template <int M>
std::uint32_t RootOnlyInitializationTranscript<M>::nlist() const noexcept {
  return state_->nlist;
}

StatusOr<RootOnlyInitializationTranscriptEntry>
detail::RootOnlyInitializationTranscriptStorage::Find(std::uint32_t cid) const {
  if (cid >= entries.size()) {
    return Status::InvalidArgument(
        "root-only initialization transcript CID is out of range");
  }
  const RootOnlyInitializationTranscriptEntry& entry = entries[cid];
  auto expected = RecordBytes(entry.real_count, m);
  if (!expected.ok() || expected.value() != entry.bytes ||
      entry.offset > records.size() ||
      entry.bytes > records.size() - entry.offset) {
    return Status::DataLoss(
        "root-only initialization transcript entry is invalid");
  }

  return entry;
}

Status detail::RootOnlyInitializationTranscriptStorage::Decode(
    const RootOnlyInitializationTranscriptEntry& entry,
    TranscriptDecodeColumns output, TranscriptSupportView support_view) const {
  const std::uint8_t* bytes = records.data() + entry.offset;
  std::size_t cursor = 0U;
  for (std::uint32_t row = 0U; row < entry.real_count; ++row) {
    const std::uint32_t encoded = LoadLittleEndianU32(bytes + cursor);
    cursor += sizeof(std::uint32_t);
    if (encoded == 0U) {
      output.roles[row] = LinkageNodeRole::kRoot;
      output.parents[row] = std::nullopt;
    } else {
      output.roles[row] = LinkageNodeRole::kLinked;
      output.parents[row] =
          LinkageParentReference{LinkageParentKind::kReal, encoded - 1U};
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

  if (const Status status = ValidateTopology(
          {m,
           entry.cid,
           dimension,
           0U,
           {output.roles.data(), output.roles.size()},
           {output.parents.data(), output.parents.size()},
           {output.depths.data(), output.depths.size()},
           support_view,
           {output.coefficients.data(), output.coefficients.size()}},
          dimension, 0U, entry.cid, dimension, nlist, m);
      !status.ok()) {
    return Status::DataLoss(
        std::string("root-only initialization transcript record is invalid: ") +
        status.message());
  }
  return Status::Ok();
}

template <int M>
StatusOr<RootOnlyInitializationTranscriptCluster<M>>
RootOnlyInitializationTranscript<M>::LoadCluster(std::uint32_t cid) const {
  auto found = state_->Find(cid);
  if (!found.ok()) return found.status();
  const auto& entry = found.value();
  RootOnlyInitializationTranscriptCluster<M> output;
  const std::uint32_t m = state_->m;
  output.m = m;
  output.cid = cid;
  output.roles.resize(entry.real_count);
  output.parents.resize(entry.real_count);
  output.depths.resize(entry.real_count);
  output.supports.resize(entry.real_count);
  output.coefficients.resize(static_cast<std::size_t>(entry.real_count) * m);

  if (const Status status =
          state_->Decode(entry,
                         {output.roles, output.parents, output.depths,
                          detail::TranscriptSupportOutput(output.supports),
                          output.coefficients},
                         TranscriptSupportView(output.supports));
      !status.ok())
    return status;
  return output;
}

template <int M>
RootOnlyInitializationTranscriptWriter<
    M>::RootOnlyInitializationTranscriptWriter(std::unique_ptr<State>
                                                   state) noexcept
    : state_(std::move(state)) {}

template <int M>
RootOnlyInitializationTranscriptWriter<
    M>::~RootOnlyInitializationTranscriptWriter() = default;

template <int M>
RootOnlyInitializationTranscriptWriter<M>::
    RootOnlyInitializationTranscriptWriter(
        RootOnlyInitializationTranscriptWriter&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
RootOnlyInitializationTranscriptWriter<M>&
RootOnlyInitializationTranscriptWriter<M>::operator=(
    RootOnlyInitializationTranscriptWriter&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

StatusOr<std::unique_ptr<detail::RootOnlyInitializationTranscriptWriterStorage>>
detail::RootOnlyInitializationTranscriptWriterStorage::Create(
    const std::filesystem::path& transient_destination, std::uint32_t dimension,
    std::uint32_t nlist, std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers)
    return Status::InvalidArgument("transcript M is invalid");
  if (dimension == 0U || nlist == 0U) {
    return Status::InvalidArgument(
        "root-only initialization transcript shape is invalid");
  }
  auto staging = StagingDirectory::CreateTransient(transient_destination);
  if (!staging.ok()) {
    return staging.status();
  }
  auto state =
      std::make_unique<detail::RootOnlyInitializationTranscriptWriterStorage>();
  state->staging = std::move(staging).value();
  state->record_path = state->staging.path() / "topology.bin";
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
StatusOr<RootOnlyInitializationTranscriptWriter<M>>
RootOnlyInitializationTranscriptWriter<M>::Create(
    const std::filesystem::path& transient_destination, std::uint32_t dimension,
    std::uint32_t nlist, std::uint32_t m) {
  if constexpr (M != 0) {
    if (m != M)
      return Status::InvalidArgument("transcript M is invalid");
  }
  auto state = State::Create(transient_destination, dimension, nlist, m);
  if (!state.ok()) return state.status();
  return RootOnlyInitializationTranscriptWriter<M>(std::move(state).value());
}

Status detail::RootOnlyInitializationTranscriptWriterStorage::Append(
    const TranscriptTopologyView& topology) {
  const std::uint32_t cid = static_cast<std::uint32_t>(entries.size());
  if (cid >= nlist) {
    return Status::FailedPrecondition(
        "root-only initialization transcript received too many CIDs");
  }

  if (topology.m != m)
    return Status::InvalidArgument("transcript topology M mismatch");
  if (const Status status =
          ValidateTopology(topology, topology.dimension, topology.virtual_count,
                           cid, dimension, nlist, m);
      !status.ok()) {
    return status;
  }
  const std::uint32_t rows = static_cast<std::uint32_t>(topology.roles.size());
  auto byte_count = RecordBytes(rows, m);
  if (!byte_count.ok()) {
    return byte_count.status();
  }
  if (byte_count.value() > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "root-only initialization transcript record is not addressable");
  }
  std::vector<std::uint8_t> record(
      static_cast<std::size_t>(byte_count.value()));
  std::size_t cursor = 0U;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const std::uint32_t encoded = topology.roles[row] == LinkageNodeRole::kRoot
                                      ? 0U
                                      : topology.parents[row]->local_id + 1U;
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
  const std::uint64_t offset = records.size();
  if (const Status status = records.Append(record); !status.ok()) {
    return status;
  }
  entries.push_back({
      .cid = cid,
      .real_count = rows,
      .offset = offset,
      .bytes = byte_count.value(),
  });
  return Status::Ok();
}

template <int M>
Status RootOnlyInitializationTranscriptWriter<M>::Append(
    const ClusterTopologyResult<M>& topology) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "root-only initialization transcript is already finished");
  }
  return state_->Append(detail::ViewTranscriptTopology(topology));
}

StatusOr<std::unique_ptr<detail::RootOnlyInitializationTranscriptStorage>>
detail::RootOnlyInitializationTranscriptWriterStorage::Finish() {
  if (entries.size() != nlist) {
    return Status::FailedPrecondition(
        "root-only initialization transcript has not consumed every CID");
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
  auto transcript =
      std::make_unique<detail::RootOnlyInitializationTranscriptStorage>();
  transcript->staging = std::move(staging);
  transcript->records = std::move(mapped).value();
  transcript->m = m;
  transcript->dimension = dimension;
  transcript->nlist = nlist;
  transcript->entries = std::move(entries);
  return transcript;
}

template <int M>
StatusOr<RootOnlyInitializationTranscript<M>>
RootOnlyInitializationTranscriptWriter<M>::Finish() {
  if (state_ == nullptr)
    return Status::FailedPrecondition(
        "root-only initialization transcript has not consumed every CID");
  auto transcript = state_->Finish();
  if (!transcript.ok()) return transcript.status();
  state_.reset();
  return RootOnlyInitializationTranscript<M>(std::move(transcript).value());
}

template class RootOnlyInitializationTranscript<0>;
template class RootOnlyInitializationTranscript<5>;
template class RootOnlyInitializationTranscript<12>;
template class RootOnlyInitializationTranscript<6>;
template class RootOnlyInitializationTranscript<10>;
template class RootOnlyInitializationTranscriptWriter<0>;
template class RootOnlyInitializationTranscriptWriter<5>;
template class RootOnlyInitializationTranscriptWriter<12>;
template class RootOnlyInitializationTranscriptWriter<6>;
template class RootOnlyInitializationTranscriptWriter<10>;

}  // namespace stlqpp
