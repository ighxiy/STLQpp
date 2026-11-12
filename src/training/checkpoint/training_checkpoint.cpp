#include "stlqpp/training/checkpoint/training_checkpoint.h"
#include "stlqpp/training/checkpoint/runtime_model_checkpoint.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/trained_model_file.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kCheckpointVersion = 4U;
constexpr std::uint32_t kCheckpointHeaderBytes = 104U;
constexpr std::array<std::uint8_t, 8> kCheckpointMagic = {
    'S', 'T', 'L', 'Q', 'T', 'R', 'N', '4'};
constexpr std::array<std::uint8_t, 8> kTrainedModelMagic = {
    'S', 'T', 'L', 'Q', 'M', 'O', 'D', '1'};
constexpr std::array<std::uint8_t, 8> kForestMagic = {
    'S', 'T', 'L', 'Q', 'T', 'F', 'S', '4'};

struct CheckpointHeader final {
  std::uint32_t completed_round = 0U;
  ModelShape shape;
  std::uint64_t real_count = 0U;
  std::uint64_t virtual_count = 0U;
  TrainingCheckpointIdentity identity;
  std::uint64_t model_bytes = 0U;
  std::uint64_t directory_bytes = 0U;
  std::uint64_t forest_bytes = 0U;
  std::uint64_t bad_cluster_mask_bytes = 0U;
};

Status WriteWholeFile(const std::filesystem::path& path,
                      ArrayView<const std::uint8_t> bytes) {
  auto writer = SequentialFileWriter::Create(path);
  if (!writer.ok()) {
    return writer.status();
  }
  if (const Status status = writer.value().Append(bytes); !status.ok()) {
    return status;
  }
  return writer.value().Close();
}

std::vector<std::uint8_t> EncodeForestHeader(const ModelShape& shape) {
  std::vector<std::uint8_t> bytes(kTrainingForestHeaderBytes, 0U);
  std::memcpy(bytes.data(), kForestMagic.data(), kForestMagic.size());
  StoreLittleEndianU32(kTrainingForestVersion, bytes.data() + 8U);
  StoreLittleEndianU32(kTrainingForestHeaderBytes, bytes.data() + 12U);
  StoreLittleEndianU32(shape.dimension, bytes.data() + 16U);
  StoreLittleEndianU32(shape.m, bytes.data() + 20U);
  StoreLittleEndianU32(shape.nlist, bytes.data() + 24U);
  return bytes;
}

Status ValidateForestHeader(const ReadOnlyMappedFile& forest,
                            const ModelShape& shape) {
  if (forest.size() < kTrainingForestHeaderBytes ||
      std::memcmp(forest.data(), kForestMagic.data(), kForestMagic.size()) !=
          0 ||
      LoadLittleEndianU32(forest.data() + 8U) != kTrainingForestVersion ||
      LoadLittleEndianU32(forest.data() + 12U) !=
          kTrainingForestHeaderBytes ||
      LoadLittleEndianU32(forest.data() + 16U) != shape.dimension ||
      LoadLittleEndianU32(forest.data() + 20U) != shape.m ||
      LoadLittleEndianU32(forest.data() + 24U) != shape.nlist ||
      LoadLittleEndianU32(forest.data() + 28U) != 0U) {
    return Status::DataLoss("training forest header is invalid");
  }
  return Status::Ok();
}

std::vector<std::uint8_t> EncodeCheckpointHeader(
    const CheckpointHeader& header) {
  std::vector<std::uint8_t> bytes(kCheckpointHeaderBytes, 0U);
  std::memcpy(bytes.data(), kCheckpointMagic.data(), kCheckpointMagic.size());
  StoreLittleEndianU32(kCheckpointVersion, bytes.data() + 8U);
  StoreLittleEndianU32(kCheckpointHeaderBytes, bytes.data() + 12U);
  StoreLittleEndianU32(header.completed_round, bytes.data() + 16U);
  StoreLittleEndianU32(header.shape.dimension, bytes.data() + 24U);
  StoreLittleEndianU32(header.shape.m, bytes.data() + 28U);
  StoreLittleEndianU32(header.shape.nlist, bytes.data() + 32U);
  StoreLittleEndianU32(
      header.shape.residual_codebook_size, bytes.data() + 36U);
  StoreLittleEndianU64(header.real_count, bytes.data() + 40U);
  StoreLittleEndianU64(header.virtual_count, bytes.data() + 48U);
  StoreLittleEndianU64(
      header.identity.training_source.value(), bytes.data() + 56U);
  StoreLittleEndianU64(
      header.identity.training_recipe.value(), bytes.data() + 64U);
  StoreLittleEndianU64(header.model_bytes, bytes.data() + 72U);
  StoreLittleEndianU64(header.directory_bytes, bytes.data() + 80U);
  StoreLittleEndianU64(header.forest_bytes, bytes.data() + 88U);
  StoreLittleEndianU64(
      header.bad_cluster_mask_bytes, bytes.data() + 96U);
  return bytes;
}

StatusOr<CheckpointHeader> DecodeCheckpointHeader(
    const ReadOnlyMappedFile& manifest,
    const TrainingCheckpointIdentity& expected_identity,
    bool model_only) {
  if (manifest.size() != kCheckpointHeaderBytes ||
      std::memcmp(
          manifest.data(),
          model_only ? kTrainedModelMagic.data() : kCheckpointMagic.data(),
          kCheckpointMagic.size()) !=
          0 ||
      LoadLittleEndianU32(manifest.data() + 8U) != kCheckpointVersion ||
      LoadLittleEndianU32(manifest.data() + 12U) !=
          kCheckpointHeaderBytes ||
      LoadLittleEndianU32(manifest.data() + 20U) != 0U) {
    return Status::DataLoss("training checkpoint header is invalid");
  }
  CheckpointHeader header;
  header.completed_round = LoadLittleEndianU32(manifest.data() + 16U);
  header.shape = {
      .dimension = LoadLittleEndianU32(manifest.data() + 24U),
      .m = LoadLittleEndianU32(manifest.data() + 28U),
      .nlist = LoadLittleEndianU32(manifest.data() + 32U),
      .residual_codebook_size = LoadLittleEndianU32(manifest.data() + 36U),
  };
  header.real_count = LoadLittleEndianU64(manifest.data() + 40U);
  header.virtual_count = LoadLittleEndianU64(manifest.data() + 48U);
  header.identity.training_source =
      ArtifactHash64::FromValue(LoadLittleEndianU64(manifest.data() + 56U));
  header.identity.training_recipe =
      ArtifactHash64::FromValue(LoadLittleEndianU64(manifest.data() + 64U));
  header.model_bytes = LoadLittleEndianU64(manifest.data() + 72U);
  header.directory_bytes = LoadLittleEndianU64(manifest.data() + 80U);
  header.forest_bytes = LoadLittleEndianU64(manifest.data() + 88U);
  header.bad_cluster_mask_bytes =
      LoadLittleEndianU64(manifest.data() + 96U);
  if (model_only != (header.directory_bytes == 0U &&
                     header.forest_bytes == 0U &&
                     header.real_count == 0U &&
                     header.virtual_count == 0U)) {
    return Status::DataLoss("trained model and checkpoint extents disagree");
  }
  if (const Status status = header.shape.Validate(); !status.ok()) {
    return Status::DataLoss("training checkpoint model shape is invalid");
  }
  if (header.identity.training_source != expected_identity.training_source ||
      header.identity.training_recipe != expected_identity.training_recipe) {
    return Status::FailedPrecondition(
        "training checkpoint source or recipe identity changed");
  }
  return header;
}

Status CheckMappedSize(const ReadOnlyMappedFile& file,
                       std::uint64_t expected,
                       const char* label) {
  if (expected > std::numeric_limits<std::size_t>::max() ||
      file.size() != expected) {
    return Status::DataLoss(std::string("training checkpoint ") + label +
                            " size disagrees with its header");
  }
  return Status::Ok();
}

bool SameShape(const ModelShape& left, const ModelShape& right) noexcept {
  return left.dimension == right.dimension && left.m == right.m &&
      left.nlist == right.nlist &&
      left.residual_codebook_size == right.residual_codebook_size;
}

Status ValidateForestPadding(
    const ReadOnlyMappedFile& forest,
    ArrayView<const TrainingForestEntry> entries) {
  std::uint64_t cursor = kTrainingForestHeaderBytes;
  for (const TrainingForestEntry& entry : entries) {
    for (; cursor < entry.offset; ++cursor) {
      if (forest.data()[static_cast<std::size_t>(cursor)] != 0U) {
        return Status::DataLoss(
            "training forest alignment padding is not canonical");
      }
    }
    cursor = entry.offset + entry.bytes;
  }
  return Status::Ok();
}

// Forest storage is independent of the numerical model width. Keep the
// per-M checkpoint wrappers for model ownership, but compile record IO once.
struct CheckpointWriterStorage {
  std::uint64_t expected_real_rows = 0U;
  std::vector<std::uint8_t> memory_forest;
  ModelShape shape;
  std::uint32_t completed_round = 0U;
  TrainingCheckpointIdentity identity;
  std::vector<TrainingForestEntry> entries;
  bool reused_forest = false;
  std::uint32_t appended_count = 0U;
  std::uint64_t real_count = 0U;
  std::uint64_t virtual_count = 0U;
  std::vector<std::uint8_t> bad_cluster_mask;
};

Status AppendForestCluster(CheckpointWriterStorage& state,
                           const TrainingForestClusterSource& cluster,
                           CanonicalBfsForestOrder* order) {
  if (cluster.cid != state.appended_count || cluster.cid >= state.shape.nlist ||
      cluster.dimension != state.shape.dimension) {
    return Status::InvalidArgument(
        "training checkpoint CIDs must be complete and source ordered");
  }
  if (state.reused_forest) {
    if (order != nullptr) {
      return Status::InvalidArgument(
          "replayed forest has no newly encoded BFS order");
    }
    const TrainingForestEntry& entry = state.entries[cluster.cid];
    if (entry.real_count != cluster.real_roles.size() ||
        entry.virtual_count != cluster.virtual_sample_ids.size()) {
      return Status::FailedPrecondition(
          "replay changed the frozen forest row counts");
    }
    ++state.appended_count;
    return Status::Ok();
  }
  const std::uint64_t offset =
      AlignTrainingForestRecord(state.memory_forest.size());
  auto bytes = TrainingForestCodec::EncodeRecordAt(
      cluster, state.shape.m, state.memory_forest, offset, order);
  if (!bytes.ok()) return bytes.status();
  const TrainingForestEntry entry = {
      .cid = cluster.cid,
      .real_count = static_cast<std::uint32_t>(cluster.real_roles.size()),
      .virtual_count =
          static_cast<std::uint32_t>(cluster.virtual_sample_ids.size()),
      .offset = offset,
      .bytes = bytes.value(),
  };
  state.entries.push_back(entry);
  state.real_count += entry.real_count;
  state.virtual_count += entry.virtual_count;
  ++state.appended_count;
  return Status::Ok();
}

struct CheckpointForestStorage {
  std::uint32_t completed_round = 0U;
  std::filesystem::path directory;
  TrainingCheckpointIdentity identity;
  std::uint64_t directory_bytes = 0U;
  ModelShape shape;
  ReadOnlyMappedFile forest;
  std::shared_ptr<const std::vector<std::uint8_t>> memory_forest;
  std::vector<TrainingForestEntry> entries;
  std::vector<std::uint8_t> bad_cluster_mask;

  CheckpointForestStorage(
      std::uint32_t round, std::filesystem::path directory_in,
      TrainingCheckpointIdentity identity_in, std::uint64_t directory_bytes_in,
      ModelShape shape_in, ReadOnlyMappedFile forest_in,
      std::vector<TrainingForestEntry> entries_in,
      std::vector<std::uint8_t> bad_cluster_mask_in,
      std::shared_ptr<const std::vector<std::uint8_t>> memory_forest_in = {})
      : completed_round(round),
        directory(std::move(directory_in)),
        identity(identity_in),
        directory_bytes(directory_bytes_in),
        shape(shape_in),
        forest(std::move(forest_in)),
        memory_forest(std::move(memory_forest_in)),
        entries(std::move(entries_in)),
        bad_cluster_mask(std::move(bad_cluster_mask_in)) {}

  [[nodiscard]] std::uint64_t forest_bytes() const noexcept {
    return memory_forest ? memory_forest->size() : forest.size();
  }
  [[nodiscard]] bool has_forest() const noexcept {
    return entries.size() == shape.nlist;
  }
  [[nodiscard]] StatusOr<TrainingForestCluster> LoadCluster(
      std::uint32_t cid) const {
    if (cid >= entries.size()) {
      return Status::InvalidArgument("training checkpoint CID is out of range");
    }
    const TrainingForestEntry& entry = entries[cid];
    const std::uint8_t* data = memory_forest ? memory_forest->data()
                                              : forest.data();
    const ArrayView<const std::uint8_t> bytes = {
        data + static_cast<std::size_t>(entry.offset),
        static_cast<std::size_t>(entry.bytes),
    };
    TrainingForestCluster cluster;
    if (const Status status = TrainingForestCodec::DecodeRecord(
            shape.dimension, shape.m, entry, bytes, cluster);
        !status.ok()) {
      return status;
    }
    return cluster;
  }
};

Status InitializeCheckpointWriter(
    CheckpointWriterStorage& state, const ModelShape& shape,
    TrainingCheckpointIdentity identity, std::uint32_t completed_round,
    std::uint64_t expected_real_rows,
    ArrayView<const std::uint8_t> bad_cluster_mask,
    bool reuse_forest,
    ArrayView<const TrainingForestEntry> reused_entries) {
  state.expected_real_rows = expected_real_rows;
  state.shape = shape;
  state.completed_round = completed_round;
  state.identity = identity;
  state.bad_cluster_mask.assign(bad_cluster_mask.begin(),
                                bad_cluster_mask.end());
  state.entries.reserve(shape.nlist);
  if (reuse_forest) {
    state.entries.assign(reused_entries.begin(), reused_entries.end());
    for (const TrainingForestEntry& entry : state.entries) {
      state.real_count += entry.real_count;
      state.virtual_count += entry.virtual_count;
    }
    state.reused_forest = true;
    return Status::Ok();
  }
  state.memory_forest = EncodeForestHeader(shape);
  if (expected_real_rows != 0U) {
    const std::uint64_t overhead = kTrainingForestHeaderBytes +
        static_cast<std::uint64_t>(shape.nlist) *
            (kTrainingForestRecordAlignment - 1U);
    if (overhead > std::numeric_limits<std::size_t>::max() ||
        expected_real_rows >
            (std::numeric_limits<std::size_t>::max() - overhead) /
                (2U * sizeof(std::uint32_t))) {
      return Status::ResourceExhausted(
          "in-memory training forest exceeds addressable memory");
    }
    const std::uint64_t minimum_bytes = overhead +
        expected_real_rows * 2U * sizeof(std::uint32_t);
    state.memory_forest.reserve(static_cast<std::size_t>(minimum_bytes));
  }
  return Status::Ok();
}

Status ValidateCheckpointWriterCompletion(
    const CheckpointWriterStorage& state) {
  if (state.appended_count != state.shape.nlist) {
    return Status::FailedPrecondition(
        "in-memory training forest is incomplete");
  }
  if (state.expected_real_rows != 0U &&
      state.real_count != state.expected_real_rows) {
    return Status::FailedPrecondition(
        "in-memory training forest row total changed from basic source");
  }
  return Status::Ok();
}

CheckpointForestStorage FinishCheckpointWriter(
    CheckpointWriterStorage& state, const CheckpointForestStorage* previous) {
  std::shared_ptr<const std::vector<std::uint8_t>> forest;
  if (previous != nullptr) {
    forest = previous->memory_forest;
    if (!forest) {
      forest = std::make_shared<const std::vector<std::uint8_t>>(
          previous->forest.data(),
          previous->forest.data() + previous->forest.size());
    }
  } else {
    forest = std::make_shared<const std::vector<std::uint8_t>>(
        std::move(state.memory_forest));
  }
  const std::uint64_t directory_bytes =
      kTrainingForestDirectoryHeaderBytes +
      static_cast<std::uint64_t>(state.shape.nlist) *
          kTrainingForestDirectoryEntryBytes;
  return CheckpointForestStorage(
      state.completed_round, {}, state.identity, directory_bytes,
      state.shape, ReadOnlyMappedFile{}, std::move(state.entries),
      std::move(state.bad_cluster_mask), std::move(forest));
}

struct OpenedForest final {
  ReadOnlyMappedFile file;
  std::vector<TrainingForestEntry> entries;
};

struct OpenedCheckpointDirectory final {
  CheckpointHeader header;
  bool model_only = false;
  ReadOnlyMappedFile model_file;
  std::vector<std::uint8_t> bad_cluster_mask;
};

StatusOr<OpenedCheckpointDirectory> OpenCheckpointDirectory(
    const TrainingCheckpointOpenRequest& request, std::uint32_t expected_m) {
  std::error_code error;
  std::filesystem::path model_manifest =
      request.directory / "train_manifest.stlqpp";
  bool model_only = std::filesystem::exists(model_manifest, error);
  if (error) {
    return Status::Internal("cannot inspect trained model artifact: " +
                            error.message());
  }
  if (!model_only) {
    model_manifest = request.directory / "trained_model.stlqpp";
    model_only = std::filesystem::exists(model_manifest, error);
    if (error) {
      return Status::Internal("cannot inspect trained model artifact: " +
                              error.message());
    }
  }
  auto manifest = ReadOnlyMappedFile::Open(
      model_only ? model_manifest : request.directory / "checkpoint.stlqpp");
  if (!manifest.ok()) return manifest.status();
  auto header = DecodeCheckpointHeader(
      manifest.value(), request.expected_identity, model_only);
  if (!header.ok()) return header.status();
  if (expected_m != 0U && header.value().shape.m != expected_m) {
    return Status::FailedPrecondition(
        "training checkpoint M differs from the selected product");
  }
  auto model_file = ReadOnlyMappedFile::Open(request.directory / "model.bin");
  auto bad_cluster_file =
      ReadOnlyMappedFile::Open(request.directory / "bad_clusters.bin");
  if (!model_file.ok()) return model_file.status();
  if (!bad_cluster_file.ok()) return bad_cluster_file.status();
  if (const Status status = CheckMappedSize(
          model_file.value(), header.value().model_bytes, "model");
      !status.ok()) return status;
  if (!model_only && (header.value().directory_bytes == 0U ||
                      header.value().forest_bytes == 0U)) {
    return Status::DataLoss("training checkpoint forest extents are invalid");
  }
  if (header.value().bad_cluster_mask_bytes != header.value().shape.nlist ||
      bad_cluster_file.value().size() !=
          header.value().bad_cluster_mask_bytes) {
    return Status::DataLoss(
        "training checkpoint bad-cluster mask extent is invalid");
  }
  std::vector<std::uint8_t> mask(
      bad_cluster_file.value().data(),
      bad_cluster_file.value().data() + bad_cluster_file.value().size());
  for (const std::uint8_t value : mask) {
    if (value > 1U) {
      return Status::DataLoss(
          "training checkpoint bad-cluster mask is not binary");
    }
  }
  return OpenedCheckpointDirectory{
      std::move(header).value(), model_only, std::move(model_file).value(),
      std::move(mask)};
}

StatusOr<OpenedForest> OpenCheckpointForest(
    const std::filesystem::path& directory, const CheckpointHeader& header) {
  auto directory_file = ReadOnlyMappedFile::Open(directory / "directory.bin");
  if (!directory_file.ok()) return directory_file.status();
  auto forest_file = ReadOnlyMappedFile::Open(directory / "forest.bin");
  if (!forest_file.ok()) return forest_file.status();
  if (const Status status = CheckMappedSize(
          directory_file.value(), header.directory_bytes, "directory");
      !status.ok()) return status;
  if (const Status status = CheckMappedSize(
          forest_file.value(), header.forest_bytes, "forest");
      !status.ok()) return status;
  if (const Status status = ValidateForestHeader(
          forest_file.value(), header.shape);
      !status.ok()) return status;
  OpenedForest result;
  if (const Status status = TrainingForestCodec::DecodeDirectory(
          directory_file.value().data(), directory_file.value().size(),
          header.shape.dimension, header.shape.m, header.shape.nlist,
          header.forest_bytes, result.entries);
      !status.ok()) return status;
  if (const Status status = ValidateForestPadding(
          forest_file.value(), result.entries);
      !status.ok()) return status;
  std::uint64_t real_count = 0U;
  std::uint64_t virtual_count = 0U;
  for (const TrainingForestEntry& entry : result.entries) {
    real_count += entry.real_count;
    virtual_count += entry.virtual_count;
  }
  if (real_count != header.real_count ||
      virtual_count != header.virtual_count) {
    return Status::DataLoss(
        "training checkpoint forest totals disagree with its header");
  }
  result.file = std::move(forest_file).value();
  return result;
}

Status PublishCheckpointForest(
    const CheckpointForestStorage& storage,
    ArrayView<const std::uint8_t> model_bytes,
    const std::filesystem::path& destination) {
  auto publication = StagingDirectory::Create(destination);
  if (!publication.ok()) return publication.status();
  std::vector<std::uint8_t> directory_bytes;
  const ModelShape& shape = storage.shape;
  if (const Status status = TrainingForestCodec::EncodeDirectory(
          shape.dimension, shape.m, shape.nlist, storage.entries,
          directory_bytes); !status.ok()) return status;
  const std::uint8_t* forest = storage.memory_forest
      ? storage.memory_forest->data() : storage.forest.data();
  for (const auto& file : {
           std::pair{"model.bin", model_bytes},
           std::pair{"directory.bin", ArrayView<const std::uint8_t>(directory_bytes)},
           std::pair{"forest.bin", ArrayView<const std::uint8_t>(forest,
               static_cast<std::size_t>(storage.forest_bytes()))},
           std::pair{"bad_clusters.bin", ArrayView<const std::uint8_t>(
               storage.bad_cluster_mask)},
       }) {
    if (const Status status = WriteWholeFile(
            publication.value().path() / file.first, file.second);
        !status.ok()) return status;
  }
  std::uint64_t real_count = 0U;
  std::uint64_t virtual_count = 0U;
  for (const TrainingForestEntry& entry : storage.entries) {
    real_count += entry.real_count;
    virtual_count += entry.virtual_count;
  }
  const CheckpointHeader header = {
      .completed_round = storage.completed_round,
      .shape = shape,
      .real_count = real_count,
      .virtual_count = virtual_count,
      .identity = storage.identity,
      .model_bytes = model_bytes.size(),
      .directory_bytes = directory_bytes.size(),
      .forest_bytes = storage.forest_bytes(),
      .bad_cluster_mask_bytes = storage.bad_cluster_mask.size(),
  };
  const std::vector<std::uint8_t> manifest = EncodeCheckpointHeader(header);
  if (const Status status = WriteWholeFile(
          publication.value().path() / "checkpoint.stlqpp", manifest);
      !status.ok()) return status;
  return publication.value().Publish();
}

}  // namespace

template <int M>
struct TrainingCheckpointWriter<M>::State final : CheckpointWriterStorage {
  const TrainingCheckpoint<M>* reused_checkpoint = nullptr;
};

template <int M>
TrainingCheckpointWriter<M>::TrainingCheckpointWriter(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
TrainingCheckpointWriter<M>::~TrainingCheckpointWriter() = default;

template <int M>
TrainingCheckpointWriter<M>::TrainingCheckpointWriter(
    TrainingCheckpointWriter&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
TrainingCheckpointWriter<M>& TrainingCheckpointWriter<M>::operator=(
    TrainingCheckpointWriter&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<TrainingCheckpointWriter<M>> TrainingCheckpointWriter<M>::Create(
    const TrainingCheckpointWriteRequest<M>& request) {
  if (request.identity.training_source.empty() ||
      request.identity.training_recipe.empty()) {
    return Status::InvalidArgument(
        "training checkpoint request is incomplete");
  }
  if (const Status status = request.shape.Validate(); !status.ok()) {
    return status;
  }
  if constexpr (M != 0) {
    if (request.shape.m != static_cast<std::uint32_t>(M)) {
      return Status::InvalidArgument(
          "training checkpoint shape disagrees with fixed M");
    }
  }
  if (request.bad_cluster_mask.size() != request.shape.nlist) {
    return Status::InvalidArgument(
        "training checkpoint bad-cluster mask has the wrong extent");
  }
  for (const std::uint8_t value : request.bad_cluster_mask) {
    if (value > 1U) {
      return Status::InvalidArgument(
          "training checkpoint bad-cluster mask is not binary");
    }
  }
  if (request.reuse_forest_from != nullptr) {
    if (!request.reuse_forest_from->has_forest() ||
        request.reuse_forest_from->completed_round() + 1U !=
            request.completed_round ||
        request.reuse_forest_from->identity().training_source !=
            request.identity.training_source ||
        request.reuse_forest_from->identity().training_recipe !=
            request.identity.training_recipe ||
        !SameShape(request.reuse_forest_from->model().codebooks().shape(),
                   request.shape)) {
      return Status::FailedPrecondition(
          "replayed forest source is not the preceding compatible round");
    }
  }
  std::unique_ptr<State> state(new State);
  const auto entries = request.reuse_forest_from == nullptr
      ? ArrayView<const TrainingForestEntry>{}
      : request.reuse_forest_from->forest_entries();
  if (const Status status = InitializeCheckpointWriter(
          *state, request.shape, request.identity, request.completed_round,
          request.expected_real_rows, request.bad_cluster_mask,
          request.reuse_forest_from != nullptr, entries);
      !status.ok()) return status;
  state->reused_checkpoint = request.reuse_forest_from;
  return TrainingCheckpointWriter<M>(std::move(state));
}

template <int M>
Status TrainingCheckpointWriter<M>::Append(
    const TrainingForestClusterSource& cluster,
    CanonicalBfsForestOrder* order) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "training checkpoint writer is not accepting clusters");
  }
  return AppendForestCluster(*state_, cluster, order);
}

template <int M>
struct TrainingCheckpoint<M>::State final : CheckpointForestStorage {
  TrainingModel<M> model;

  State(CheckpointForestStorage storage, TrainingModel<M> model_in) noexcept
      : CheckpointForestStorage(std::move(storage)),
        model(std::move(model_in)) {}

  State(std::uint32_t round,
        std::filesystem::path directory_in,
        TrainingCheckpointIdentity identity_in,
        std::uint64_t directory_bytes_in,
        TrainingModel<M> model_in,
        ReadOnlyMappedFile forest_in,
        std::vector<TrainingForestEntry> entries_in,
        std::vector<std::uint8_t> bad_cluster_mask_in,
        std::shared_ptr<const std::vector<std::uint8_t>> memory_forest_in = {}) noexcept
      : CheckpointForestStorage(round, std::move(directory_in), identity_in,
                                directory_bytes_in,
                                model_in.codebooks().shape(),
                                std::move(forest_in), std::move(entries_in),
                                std::move(bad_cluster_mask_in),
                                std::move(memory_forest_in)),
        model(std::move(model_in)) {}
};

template <int M>
TrainingCheckpoint<M>::TrainingCheckpoint(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

template <int M>
TrainingCheckpoint<M>::~TrainingCheckpoint() = default;

template <int M>
TrainingCheckpoint<M>::TrainingCheckpoint(
    TrainingCheckpoint&& other) noexcept
    : state_(std::move(other.state_)) {}

template <int M>
TrainingCheckpoint<M>& TrainingCheckpoint<M>::operator=(
    TrainingCheckpoint&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

template <int M>
StatusOr<TrainingCheckpoint<M>> TrainingCheckpointWriter<M>::FinishMemory(
    TrainingModel<M> model) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "in-memory training forest is incomplete");
  }
  if (const Status status = ValidateCheckpointWriterCompletion(*state_);
      !status.ok()) return status;
  if (!SameShape(model.codebooks().shape(), state_->shape)) {
    return Status::InvalidArgument(
        "in-memory training model changed the forest shape");
  }
  const CheckpointForestStorage* previous =
      state_->reused_checkpoint == nullptr
          ? nullptr
          : state_->reused_checkpoint->state_.get();
  CheckpointForestStorage storage = FinishCheckpointWriter(*state_, previous);
  std::unique_ptr<typename TrainingCheckpoint<M>::State> next(
      new typename TrainingCheckpoint<M>::State(
          std::move(storage), std::move(model)));
  state_.reset();
  return TrainingCheckpoint<M>(std::move(next));
}

StatusOr<RuntimeCheckpointModel> OpenRuntimeCheckpointModel(
    const TrainingCheckpointOpenRequest& request) {
  if (request.directory.empty() || request.expected_identity.training_source.empty() ||
      request.expected_identity.training_recipe.empty())
    return Status::InvalidArgument("runtime checkpoint model request is incomplete");
  std::error_code error;
  if (std::filesystem::is_regular_file(request.directory, error)) {
    auto artifact = TrainedModelFile::Open(request.directory);
    if (!artifact.ok()) return artifact.status();
    const auto& header = artifact.value().header();
    if (header.training_source != request.expected_identity.training_source.value() ||
        header.training_recipe != request.expected_identity.training_recipe.value())
      return Status::FailedPrecondition("trained-model identity differs from the request");
    const auto bytes = artifact.value().model_bytes();
    auto model = CanonicalModelCodec::DecodeRuntime(bytes.data(), bytes.size());
    if (!model.ok()) return model.status();
    if (!SameShape(model.value().shape(), header.shape))
      return Status::DataLoss("trained model shape disagrees with its header");
    const auto mask = artifact.value().bad_cluster_mask();
    return RuntimeCheckpointModel{std::move(model).value(), {mask.begin(), mask.end()}};
  }
  if (error) return Status::Internal("cannot inspect runtime model path: " + error.message());
  if (std::filesystem::exists(request.directory / "initialized.stlqpp", error))
    return OpenRuntimeInitializedModel(request);
  if (error) return Status::Internal("cannot inspect initialized model: " + error.message());
  auto manifest_path = request.directory / "train_manifest.stlqpp";
  bool model_only = std::filesystem::exists(manifest_path, error);
  if (error) return Status::Internal("cannot inspect model manifest: " + error.message());
  if (!model_only) {
    manifest_path = request.directory / "trained_model.stlqpp";
    model_only = std::filesystem::exists(manifest_path, error);
    if (error) return Status::Internal("cannot inspect model manifest: " + error.message());
  }
  auto manifest = ReadOnlyMappedFile::Open(
      model_only ? manifest_path : request.directory / "checkpoint.stlqpp");
  if (!manifest.ok()) return manifest.status();
  auto header = DecodeCheckpointHeader(manifest.value(), request.expected_identity, model_only);
  if (!header.ok()) return header.status();
  auto model_file = ReadOnlyMappedFile::Open(request.directory / "model.bin");
  if (!model_file.ok()) return model_file.status();
  if (const auto s = CheckMappedSize(model_file.value(), header.value().model_bytes, "model"); !s.ok()) return s;
  auto mask_file = ReadOnlyMappedFile::Open(request.directory / "bad_clusters.bin");
  if (!mask_file.ok()) return mask_file.status();
  if (header.value().bad_cluster_mask_bytes != header.value().shape.nlist ||
      mask_file.value().size() != header.value().bad_cluster_mask_bytes)
    return Status::DataLoss("runtime model bad-cluster mask extent is invalid");
  std::vector<std::uint8_t> mask(mask_file.value().data(),
      mask_file.value().data() + mask_file.value().size());
  for (auto value : mask)
    if (value > 1U) return Status::DataLoss("runtime model bad-cluster mask is not binary");
  auto model = CanonicalModelCodec::DecodeRuntime(model_file.value().data(), model_file.value().size());
  if (!model.ok()) return model.status();
  if (!SameShape(model.value().shape(), header.value().shape))
    return Status::DataLoss("runtime checkpoint model shape disagrees with header");
  return RuntimeCheckpointModel{std::move(model).value(), std::move(mask)};
}

template <int M>
StatusOr<TrainingCheckpoint<M>> TrainingCheckpoint<M>::Open(
    const TrainingCheckpointOpenRequest& request) {
  if (request.directory.empty() ||
      request.expected_identity.training_source.empty() ||
      request.expected_identity.training_recipe.empty()) {
    return Status::InvalidArgument(
        "training checkpoint open request is incomplete");
  }
  std::error_code file_error;
  if (std::filesystem::is_regular_file(request.directory, file_error)) {
    auto artifact = TrainedModelFile::Open(request.directory);
    if (!artifact.ok()) return artifact.status();
    const auto& header = artifact.value().header();
    if (header.kind != TrainedModelKind::kTrained ||
        (M != 0 && header.shape.m != static_cast<std::uint32_t>(M)) ||
        header.training_source != request.expected_identity.training_source.value() ||
        header.training_recipe != request.expected_identity.training_recipe.value()) {
      return Status::FailedPrecondition(
          "trained-model kind, M or identity differs from the request");
    }
    const auto bytes = artifact.value().model_bytes();
    auto canonical = CanonicalModelCodec::Decode<M>(bytes.data(), bytes.size());
    if (!canonical.ok()) return canonical.status();
    auto model = TrainingModel<M>::Create(
        canonical.value().canonical_rotation(),
        canonical.value().codebooks().view());
    if (!model.ok()) return model.status();
    if (!SameShape(model.value().codebooks().shape(), header.shape)) {
      return Status::DataLoss("trained-model shape disagrees with payload");
    }
    const auto mask = artifact.value().bad_cluster_mask();
    std::unique_ptr<State> state(new State(
        header.completed_round, request.directory, request.expected_identity,
        0U, std::move(model).value(), ReadOnlyMappedFile{}, {},
        std::vector<std::uint8_t>(mask.begin(), mask.end())));
    return TrainingCheckpoint<M>(std::move(state));
  }
  if (file_error) {
    return Status::Internal("cannot inspect trained-model path: " +
                            file_error.message());
  }
  auto opened_directory = OpenCheckpointDirectory(request, M);
  if (!opened_directory.ok()) return opened_directory.status();
  OpenedCheckpointDirectory payload = std::move(opened_directory).value();
  const CheckpointHeader& header = payload.header;
  auto canonical = CanonicalModelCodec::Decode<M>(
      payload.model_file.data(), payload.model_file.size());
  if (!canonical.ok()) {
    return canonical.status();
  }
  auto model = TrainingModel<M>::Create(
      canonical.value().canonical_rotation(),
      canonical.value().codebooks().view());
  if (!model.ok()) {
    return model.status();
  }
  if (!SameShape(model.value().codebooks().shape(), header.shape)) {
    return Status::DataLoss(
        "training checkpoint model shape disagrees with its header");
  }
  std::vector<TrainingForestEntry> entries;
  ReadOnlyMappedFile forest_file;
  if (!payload.model_only) {
    auto opened = OpenCheckpointForest(request.directory, header);
    if (!opened.ok()) return opened.status();
    OpenedForest result = std::move(opened).value();
    forest_file = std::move(result.file);
    entries = std::move(result.entries);
  }
  std::unique_ptr<State> state(new State(
      header.completed_round, request.directory,
      request.expected_identity, header.directory_bytes,
      std::move(model).value(),
      std::move(forest_file), std::move(entries),
      std::move(payload.bad_cluster_mask)));
  return TrainingCheckpoint<M>(std::move(state));
}

template <int M>
std::uint32_t TrainingCheckpoint<M>::completed_round() const noexcept {
  return state_->completed_round;
}

template <int M>
const std::filesystem::path& TrainingCheckpoint<M>::directory() const noexcept {
  return state_->directory;
}

template <int M>
const TrainingCheckpointIdentity& TrainingCheckpoint<M>::identity()
    const noexcept {
  return state_->identity;
}

template <int M>
ArrayView<const TrainingForestEntry> TrainingCheckpoint<M>::forest_entries()
    const noexcept {
  return state_->entries;
}

template <int M>
std::uint64_t TrainingCheckpoint<M>::forest_bytes() const noexcept {
  return state_->CheckpointForestStorage::forest_bytes();
}

template <int M>
std::uint64_t TrainingCheckpoint<M>::directory_bytes() const noexcept {
  return state_->directory_bytes;
}

template <int M>
bool TrainingCheckpoint<M>::has_forest() const noexcept {
  return state_->CheckpointForestStorage::has_forest();
}

template <int M>
Status TrainingCheckpoint<M>::Publish(
    const std::filesystem::path& destination) const {
  if (destination.empty()) {
    return Status::InvalidArgument("training checkpoint path is empty");
  }
  if (!has_forest()) {
    return Status::FailedPrecondition(
        "full training checkpoint publication requires a forest");
  }
  std::vector<std::uint8_t> model_bytes;
  if (const Status status = CanonicalModelCodec::Encode<M>({
          .rotation = state_->model.rotation(),
          .codebooks = state_->model.codebook_view(),
      }, model_bytes); !status.ok()) return status;
  return PublishCheckpointForest(*state_, model_bytes, destination);
}

template <int M>
Status TrainingCheckpoint<M>::PublishModelOnly(
    const std::filesystem::path& destination) const {
  if (destination.empty()) {
    return Status::InvalidArgument("trained model path is empty");
  }
  std::vector<std::uint8_t> model_bytes;
  if (const Status status = CanonicalModelCodec::Encode<M>({
          .rotation = state_->model.rotation(),
          .codebooks = state_->model.codebook_view(),
      }, model_bytes); !status.ok()) return status;
  const TrainedModelFileHeader header = {
      .kind = TrainedModelKind::kTrained,
      .completed_round = state_->completed_round,
      .shape = state_->model.codebooks().shape(),
      .training_source = state_->identity.training_source.value(),
      .training_recipe = state_->identity.training_recipe.value(),
  };
  return TrainedModelFile::Publish(
      destination, header, model_bytes, state_->bad_cluster_mask);
}

template <int M>
const TrainingModel<M>& TrainingCheckpoint<M>::model() const noexcept {
  return state_->model;
}

template <int M>
ArrayView<const std::uint8_t>
TrainingCheckpoint<M>::bad_cluster_mask() const noexcept {
  return state_->bad_cluster_mask;
}

template <int M>
StatusOr<TrainingForestCluster> TrainingCheckpoint<M>::LoadCluster(
    std::uint32_t cid) const {
  return state_->CheckpointForestStorage::LoadCluster(cid);
}

template class TrainingCheckpointWriter<5>;
template class TrainingCheckpointWriter<12>;
template class TrainingCheckpointWriter<6>;
template class TrainingCheckpointWriter<10>;
template class TrainingCheckpoint<0>;
template class TrainingCheckpointWriter<0>;
template class TrainingCheckpoint<5>;
template class TrainingCheckpoint<12>;
template class TrainingCheckpoint<6>;
template class TrainingCheckpoint<10>;

}  // namespace stlqpp
