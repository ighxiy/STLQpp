#include "stlqpp/training/checkpoint/initialized_training_checkpoint.h"
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
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8U> kMagic = {
    'S', 'T', 'L', 'Q', 'I', 'N', 'I', '1'};
constexpr std::uint32_t kVersion = 1U;
constexpr std::uint32_t kHeaderBytes = 80U;

Status WriteFile(const std::filesystem::path& path,
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

}  // namespace

template <int M>
Status InitializedTrainingCheckpoint<M>::Write(
    const std::filesystem::path& destination,
    const TrainingCheckpointIdentity& identity,
    const TrainingModel<M>& model,
    ArrayView<const std::uint8_t> bad_cluster_mask) {
  const ModelShape& shape = model.codebooks().shape();
  if (destination.empty() || identity.training_source.empty() ||
      identity.training_recipe.empty() ||
      bad_cluster_mask.size() != shape.nlist) {
    return Status::InvalidArgument(
        "initialized checkpoint request is incomplete");
  }
  for (const std::uint8_t value : bad_cluster_mask) {
    if (value > 1U) {
      return Status::InvalidArgument(
          "initialized checkpoint bad-cluster mask is not binary");
    }
  }

  std::vector<std::uint8_t> model_bytes;
  if (const Status status = CanonicalModelCodec::Encode<M>(
          {
              .rotation = model.rotation(),
              .codebooks = model.codebook_view(),
          },
          model_bytes);
      !status.ok()) {
    return status;
  }
  if (destination.extension() == ".stlqmodel") {
    return TrainedModelFile::Publish(destination, {
        .kind = TrainedModelKind::kInitialized,
        .shape = shape,
        .training_source = identity.training_source.value(),
        .training_recipe = identity.training_recipe.value(),
    }, model_bytes, bad_cluster_mask);
  }
  auto staging = StagingDirectory::Create(destination);
  if (!staging.ok()) {
    return staging.status();
  }
  if (const Status status = WriteFile(
          staging.value().path() / "model.bin", model_bytes);
      !status.ok()) {
    return status;
  }
  if (const Status status = WriteFile(
          staging.value().path() / "bad_clusters.bin", bad_cluster_mask);
      !status.ok()) {
    return status;
  }

  std::vector<std::uint8_t> header(kHeaderBytes, 0U);
  std::memcpy(header.data(), kMagic.data(), kMagic.size());
  StoreLittleEndianU32(kVersion, header.data() + 8U);
  StoreLittleEndianU32(kHeaderBytes, header.data() + 12U);
  StoreLittleEndianU32(shape.dimension, header.data() + 16U);
  StoreLittleEndianU32(shape.m, header.data() + 20U);
  StoreLittleEndianU32(shape.nlist, header.data() + 24U);
  StoreLittleEndianU32(shape.residual_codebook_size, header.data() + 28U);
  StoreLittleEndianU64(identity.training_source.value(), header.data() + 32U);
  StoreLittleEndianU64(identity.training_recipe.value(), header.data() + 40U);
  StoreLittleEndianU64(model_bytes.size(), header.data() + 48U);
  StoreLittleEndianU64(bad_cluster_mask.size(), header.data() + 56U);
  if (const Status status = WriteFile(
          staging.value().path() / "initialized.stlqpp", header);
      !status.ok()) {
    return status;
  }
  return staging.value().Publish();
}

template <int M>
StatusOr<InitializedTrainingCheckpoint<M>>
InitializedTrainingCheckpoint<M>::Open(
    const std::filesystem::path& directory,
    const TrainingCheckpointIdentity& expected_identity) {
  std::error_code file_error;
  if (std::filesystem::is_regular_file(directory, file_error)) {
    auto artifact = TrainedModelFile::Open(directory);
    if (!artifact.ok()) return artifact.status();
    const auto& header = artifact.value().header();
    if (header.kind != TrainedModelKind::kInitialized ||
        (M != 0 && header.shape.m != static_cast<std::uint32_t>(M)) ||
        header.training_source != expected_identity.training_source.value() ||
        header.training_recipe != expected_identity.training_recipe.value()) {
      return Status::FailedPrecondition(
          "initialized model kind, M or identity changed");
    }
    const auto bytes = artifact.value().model_bytes();
    auto canonical = CanonicalModelCodec::Decode<M>(bytes.data(), bytes.size());
    if (!canonical.ok()) return canonical.status();
    auto model = TrainingModel<M>::Create(
        canonical.value().canonical_rotation(),
        canonical.value().codebooks().view());
    if (!model.ok()) return model.status();
    if (model.value().codebooks().shape().dimension != header.shape.dimension ||
        model.value().codebooks().shape().nlist != header.shape.nlist ||
        model.value().codebooks().shape().residual_codebook_size !=
            header.shape.residual_codebook_size) {
      return Status::DataLoss("initialized model shape disagrees with payload");
    }
    const auto mask = artifact.value().bad_cluster_mask();
    return InitializedTrainingCheckpoint<M>(
        std::move(model).value(),
        std::vector<std::uint8_t>(mask.begin(), mask.end()));
  }
  if (file_error) {
    return Status::Internal("cannot inspect initialized model path: " +
                            file_error.message());
  }
  auto runtime = OpenRuntimeInitializedModel({directory, expected_identity});
  if (!runtime.ok()) return runtime.status();
  const auto& source = runtime.value().model;
  if (M != 0 && source.shape().m != M)
    return Status::FailedPrecondition("initialized checkpoint M changed");
  ProductCodebookView<M> view;
  view.shape = source.shape();
  if constexpr (M == 0) {
    view.root_small.resize(view.shape.m - 1U);
    view.edge.resize(view.shape.m);
  }
  view.root0 = source.root0();
  for (std::uint32_t layer = 1; layer < source.shape().m; ++layer)
    view.root_small[layer - 1U] = source.root_small(layer);
  for (std::uint32_t layer = 0; layer < source.shape().m; ++layer) view.edge[layer] = source.edge(layer);
  auto model = TrainingModel<M>::Create(source.rotation(), view);
  if (!model.ok()) return model.status();
  return InitializedTrainingCheckpoint<M>(std::move(model).value(),
      std::move(runtime).value().bad_cluster_mask);
}

StatusOr<RuntimeCheckpointModel> OpenRuntimeInitializedModel(
    const TrainingCheckpointOpenRequest& request) {
  const auto& directory = request.directory;
  const auto& expected_identity = request.expected_identity;
  auto header = ReadOnlyMappedFile::Open(directory / "initialized.stlqpp");
  if (!header.ok()) {
    return header.status();
  }
  if (header.value().size() != kHeaderBytes ||
      std::memcmp(header.value().data(), kMagic.data(), kMagic.size()) != 0 ||
      LoadLittleEndianU32(header.value().data() + 8U) != kVersion ||
      LoadLittleEndianU32(header.value().data() + 12U) != kHeaderBytes) {
    return Status::DataLoss("initialized checkpoint header is invalid");
  }
  const ModelShape shape = {
      .dimension = LoadLittleEndianU32(header.value().data() + 16U),
      .m = LoadLittleEndianU32(header.value().data() + 20U),
      .nlist = LoadLittleEndianU32(header.value().data() + 24U),
      .residual_codebook_size =
          LoadLittleEndianU32(header.value().data() + 28U),
  };
  if (const Status status = shape.Validate(); !status.ok()) {
    return Status::DataLoss("initialized checkpoint shape is invalid");
  }
  if (ArtifactHash64::FromValue(
          LoadLittleEndianU64(header.value().data() + 32U)) !=
          expected_identity.training_source ||
      ArtifactHash64::FromValue(
          LoadLittleEndianU64(header.value().data() + 40U)) !=
          expected_identity.training_recipe) {
    return Status::FailedPrecondition(
        "initialized checkpoint identity or M changed");
  }
  const std::uint64_t expected_model_bytes =
      LoadLittleEndianU64(header.value().data() + 48U);
  const std::uint64_t expected_mask_bytes =
      LoadLittleEndianU64(header.value().data() + 56U);
  if (expected_model_bytes > std::numeric_limits<std::size_t>::max() ||
      expected_mask_bytes != shape.nlist) {
    return Status::DataLoss(
        "initialized checkpoint extents are invalid");
  }

  auto model_file = ReadOnlyMappedFile::Open(directory / "model.bin");
  if (!model_file.ok()) {
    return model_file.status();
  }
  auto mask_file = ReadOnlyMappedFile::Open(directory / "bad_clusters.bin");
  if (!mask_file.ok()) {
    return mask_file.status();
  }
  if (model_file.value().size() != expected_model_bytes ||
      mask_file.value().size() != expected_mask_bytes) {
    return Status::DataLoss(
        "initialized checkpoint file sizes disagree with its header");
  }
  auto canonical = CanonicalModelCodec::DecodeRuntime(
      model_file.value().data(), model_file.value().size());
  if (!canonical.ok()) {
    return canonical.status();
  }
  const auto& decoded = canonical.value().shape();
  if (decoded.m != shape.m || decoded.dimension != shape.dimension ||
      decoded.nlist != shape.nlist || decoded.residual_codebook_size != shape.residual_codebook_size)
    return Status::DataLoss("initialized model shape disagrees with its header");
  std::vector<std::uint8_t> mask(
      mask_file.value().data(),
      mask_file.value().data() + mask_file.value().size());
  for (const std::uint8_t value : mask) {
    if (value > 1U) {
      return Status::DataLoss(
          "initialized checkpoint bad-cluster mask is not binary");
    }
  }
  return RuntimeCheckpointModel{std::move(canonical).value(), std::move(mask)};
}

template class InitializedTrainingCheckpoint<0>;
template class InitializedTrainingCheckpoint<5>;
template class InitializedTrainingCheckpoint<12>;
template class InitializedTrainingCheckpoint<6>;
template class InitializedTrainingCheckpoint<10>;

}  // namespace stlqpp
