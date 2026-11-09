#include "stlqpp/storage/canonical_index_writer.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_directory_codec.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_shard_codec.h"
#include "stlqpp/storage/index_manifest_codec.h"
#include "stlqpp/storage/norm_cache_layout.h"
#include "stlqpp/storage/trained_model_file.h"

namespace stlqpp {
namespace {

Status WriteBytes(std::ofstream& stream, const std::uint8_t* bytes,
                  std::size_t byte_count, const char* label) {
  if (byte_count >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    return Status::ResourceExhausted(std::string("canonical ") + label +
                                     " exceeds streamsize");
  }
  if (byte_count != 0U) {
    stream.write(reinterpret_cast<const char*>(bytes),
                 static_cast<std::streamsize>(byte_count));
  }
  if (!stream.good()) {
    return Status::Internal(std::string("failed to write canonical ") + label);
  }
  return Status::Ok();
}

Status WritePayloadHeader(std::ofstream& stream, ShardKind kind,
                          const ModelShape& shape,
                          std::vector<std::uint8_t>& header,
                          const char* label) {
  if (const Status status =
          CanonicalPayloadShardCodec::EncodeHeader(kind, shape, header);
      !status.ok()) {
    return status;
  }
  return WriteBytes(stream, header.data(), header.size(), label);
}

Status WriteFile(const std::filesystem::path& path,
                 ArrayView<const std::uint8_t> bytes, const char* label) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    return Status::Internal(std::string("failed to create canonical ") + label);
  }
  if (const Status status =
          WriteBytes(stream, bytes.data(), bytes.size(), label);
      !status.ok()) {
    return status;
  }
  stream.close();
  if (stream.fail()) {
    return Status::Internal(std::string("failed to close canonical ") + label);
  }
  return Status::Ok();
}

Status OpenOutput(const std::filesystem::path& path, std::ofstream& stream,
                  const char* label) {
  stream.open(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    return Status::Internal(std::string("failed to create canonical ") + label);
  }
  return Status::Ok();
}

Status CloseOutput(std::ofstream& stream, const char* label) {
  stream.close();
  if (stream.fail()) {
    return Status::Internal(std::string("failed to close canonical ") + label);
  }
  return Status::Ok();
}

Status AppendAligned(std::ofstream& stream, std::uint64_t& cursor,
                     ArrayView<const std::uint8_t> record,
                     std::uint64_t& offset, const char* label) {
  offset = AlignCanonicalRecord(cursor);
  const std::uint64_t padding = offset - cursor;
  constexpr std::array<std::uint8_t, kCanonicalRecordAlignment> zeros{};
  if (padding != 0U) {
    if (const Status status = WriteBytes(
            stream, zeros.data(), static_cast<std::size_t>(padding), label);
        !status.ok()) {
      return status;
    }
  }
  if (const Status status =
          WriteBytes(stream, record.data(), record.size(), label);
      !status.ok()) {
    return status;
  }
  if (record.size() > std::numeric_limits<std::uint64_t>::max() - offset) {
    return Status::ResourceExhausted("canonical payload cursor overflows u64");
  }
  cursor = offset + record.size();
  return Status::Ok();
}

std::vector<ShardDescriptor> MakeShards(
    std::uint64_t model_bytes, std::uint64_t directory_bytes,
    std::uint64_t codes_bytes, std::uint64_t parents_bytes,
    std::uint64_t coefficients_bytes, std::uint64_t norm8_bytes,
    std::uint64_t float_coefficients_bytes, std::uint64_t float_norm2_bytes,
    std::uint64_t scaled_score8_bytes, std::uint64_t float_source_norm2_bytes,
    const IndexArtifactHashes& hashes, const ModelShape& shape,
    const IndexCounts& counts, bool retain_float_forest,
    bool publish_quantized_forest, bool has_scaled_score8,
    bool has_float_source_norm2) {
  const std::uint64_t nodes = counts.real + counts.virtual_roots;
  std::vector<ShardDescriptor> output = {
      {ShardKind::kCodebooks, 0U, "model.bin", model_bytes, 1U},
      {ShardKind::kIvfDirectory, 0U, "directory.bin", directory_bytes,
       shape.nlist},
      {ShardKind::kCodes, 0U, "codes.bin", codes_bytes, nodes},
      {ShardKind::kParents, 0U, "parents.bin", parents_bytes, nodes},
  };
  if (publish_quantized_forest) {
    output.push_back({ShardKind::kCoefficients, 0U, "coefficients.bin",
                      coefficients_bytes, nodes});
    output.push_back(
        {ShardKind::kNorm8, 0U,
         (PublishedNorm2Directory(hashes) / "norm2_labels.u8").generic_string(),
         norm8_bytes, counts.real});
    output.push_back(
        {ShardKind::kNorm2Lut, 0U,
         (PublishedNorm2Directory(hashes) / "norm2_lut.f32").generic_string(),
         1024U, 256U});
  }
  if (retain_float_forest) {
    output.push_back({ShardKind::kFloatCoefficients, 0U,
                      "float_coefficients.f32", float_coefficients_bytes,
                      nodes * shape.m});
    if (float_norm2_bytes != 0U)
      output.push_back({ShardKind::kFloatNorm2, 0U,
                        "norm-cache/constant/exact_float_norm2.f32",
                        float_norm2_bytes, counts.real});
  }
  if (has_scaled_score8) {
    output.push_back({ShardKind::kScaledScore8, 0U,
                      (PublishedNorm2Directory(hashes) / "norm2_scale_bias.f32")
                          .generic_string(),
                      scaled_score8_bytes, 512U});
  }
  if (has_float_source_norm2) {
    output.push_back({ShardKind::kFloatSourceNorm2, 0U,
                      "norm-cache/constant/legacy_original_vector_norm2.f32",
                      float_source_norm2_bytes, counts.real});
  }
  return output;
}

Status ValidateEncodedCluster(std::uint32_t m,
                              const CanonicalEncodedClusterView& source,
                              bool retain_float_forest,
                              bool retain_float_norm2,
                              bool retain_float_source_norm2,
                              bool publish_quantized_forest,
                              bool validated_float) {
  if (source.real_root_count > source.real_count ||
      static_cast<std::uint64_t>(source.real_count) + source.virtual_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::DataLoss("encoded canonical CID counts are invalid");
  }
  const std::uint64_t expected_codes =
      static_cast<std::uint64_t>(source.real_count) * (m + 4U) +
      static_cast<std::uint64_t>(source.virtual_count) * (m - 1U);
  const std::uint64_t expected_parents =
      ParentLoudsWordCount(source.real_count + source.virtual_count) * 8U;
  if (source.records.codes.size() != expected_codes ||
      source.records.parents.size() != expected_parents ||
      (publish_quantized_forest &&
       source.records.norm8.size() != source.real_count) ||
      (!publish_quantized_forest && (!source.records.coefficients.empty() ||
                                     !source.records.norm8.empty()))) {
    return Status::DataLoss(
        "encoded canonical CID record extents disagree with its counts");
  }
  if (publish_quantized_forest &&
      source.records.coefficients.size() < 8U + 8U * m) {
    return Status::DataLoss(
        "encoded canonical coefficient record is truncated");
  }
  if (publish_quantized_forest) {
    const std::uint32_t root_bytes =
        LoadLittleEndianU32(source.records.coefficients.data());
    const std::uint32_t linkage_bytes =
        LoadLittleEndianU32(source.records.coefficients.data() + 4U);
    if (CanonicalCoefficientAlphabetFromRecord(
            m, source.records.coefficients.size(), root_bytes, linkage_bytes) ==
        0U) {
      return Status::DataLoss(
          "encoded canonical coefficient payload extents are invalid");
    }
  }
  const std::uint64_t nodes =
      static_cast<std::uint64_t>(source.real_count) + source.virtual_count;
  if (retain_float_forest) {
    if (source.float_coefficients_layer_major.size() != nodes * m ||
        ((retain_float_norm2 || !source.float_real_norm2.empty()) &&
         source.float_real_norm2.size() != source.real_count)) {
      return Status::DataLoss(
          "encoded canonical CID is missing its native-float companion");
    }
    if (!validated_float) {
      for (const float value : source.float_coefficients_layer_major) {
        if (!std::isfinite(value)) {
          return Status::DataLoss(
              "native-float coefficient shard contains a non-finite value");
        }
      }
      for (const float value : source.float_real_norm2) {
        if (!std::isfinite(value) || value < 0.0F) {
          return Status::DataLoss(
              "native-float norm shard contains an invalid value");
        }
      }
    }
  } else if (!source.float_coefficients_layer_major.empty() ||
             !source.float_real_norm2.empty()) {
    return Status::InvalidArgument(
        "native-float records were supplied to a compressed-only writer");
  }
  if (retain_float_source_norm2) {
    if (source.float_source_norm2.size() != source.real_count) {
      return Status::DataLoss(
          "float_scaled CID is missing original-vector norms");
    }
    if (!validated_float) {
      for (const float value : source.float_source_norm2) {
        if (!std::isfinite(value) || value < 0.0F) {
          return Status::DataLoss(
              "float source norm shard contains an invalid value");
        }
      }
    }
  } else if (!source.float_source_norm2.empty()) {
    return Status::InvalidArgument(
        "float source norms were supplied without float_scaled publication");
  }
#ifndef NDEBUG
  if (validated_float) {
    for (const float value : source.float_coefficients_layer_major) {
      assert(std::isfinite(value));
    }
    for (const float value : source.float_real_norm2) {
      assert(std::isfinite(value) && value >= 0.0F);
    }
    for (const float value : source.float_source_norm2) {
      assert(std::isfinite(value) && value >= 0.0F);
    }
  }
#endif
  return Status::Ok();
}

Status AppendF32Aligned(std::ofstream& stream, std::uint64_t& cursor,
                        ArrayView<const float> values, const char* label) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
  for (std::size_t index = 0U; index < values.size(); ++index) {
    StoreLittleEndianF32(values[index], bytes.data() + index * sizeof(float));
  }
  std::uint64_t ignored_offset = 0U;
  return AppendAligned(stream, cursor, bytes, ignored_offset, label);
}

}  // namespace

struct RuntimeCanonicalIndexWriter::State final {
  StagingDirectory publication;
  std::string norm_backend;
  ModelShape shape;
  IndexCounts counts;
  IndexArtifactHashes hashes;
  std::optional<IndexProducerRecipe> producer_recipe;
  std::vector<CanonicalCidEntry> entries;
  std::ofstream codes;
  std::ofstream parents;
  std::ofstream coefficients;
  std::ofstream norm8;
  std::ofstream float_coefficients;
  std::ofstream float_norm2;
  std::ofstream float_source_norm2;
  std::uint64_t model_bytes = 0U;
  std::optional<ShardDescriptor> model_reference;
  std::uint64_t codes_cursor = 0U;
  std::uint64_t parents_cursor = 0U;
  std::uint64_t coefficients_cursor = 0U;
  std::uint64_t norm8_cursor = 0U;
  std::uint64_t float_coefficients_cursor = 0U;
  std::uint64_t float_norm2_cursor = 0U;
  std::uint64_t scaled_score8_bytes = 0U;
  std::uint64_t float_source_norm2_cursor = 0U;
  std::uint64_t code_record_bytes = 0U;
  std::uint64_t parent_record_bytes = 0U;
  std::uint64_t coefficient_record_bytes = 0U;
  std::uint64_t coefficient_header_bytes = 0U;
  std::uint64_t root_coefficient_payload_bytes = 0U;
  std::uint64_t linkage_coefficient_payload_bytes = 0U;
  std::uint64_t norm8_label_bytes = 0U;
  std::uint64_t parent_raw_bits = 0U;
  std::uint64_t parent_word_padding_bits = 0U;
  std::filesystem::path adopted_directory;
  std::vector<CanonicalCidEntry> adopted_entries;
  std::uint32_t norm8_center_count = 0U;
  std::uint32_t coefficient_alphabet_size =
      kCanonicalBaselineCoefficientAlphabetSize;
  ArtifactHash64 scaled_score8_hash;
  bool retain_float_forest = false;
  bool retain_float_norm2 = true;
  bool retain_float_source_norm2 = false;
  bool publish_quantized_forest = true;
};

RuntimeCanonicalIndexWriter::RuntimeCanonicalIndexWriter(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

RuntimeCanonicalIndexWriter::~RuntimeCanonicalIndexWriter() = default;

RuntimeCanonicalIndexWriter::RuntimeCanonicalIndexWriter(
    RuntimeCanonicalIndexWriter&& other) noexcept
    : state_(std::move(other.state_)) {}

RuntimeCanonicalIndexWriter& RuntimeCanonicalIndexWriter::operator=(
    RuntimeCanonicalIndexWriter&& other) noexcept {
  state_ = std::move(other.state_);
  return *this;
}

StatusOr<RuntimeCanonicalIndexWriter> RuntimeCanonicalIndexWriter::Create(
    const RuntimeCanonicalIndexWriteRequest& request) {
  return CreateImpl(request, {}, {});
}
StatusOr<RuntimeCanonicalIndexWriter>
RuntimeCanonicalIndexWriter::CreateAdopting(
    const RuntimeCanonicalIndexWriteRequest& request,
    const std::filesystem::path& directory,
    std::vector<CanonicalCidEntry> entries) {
  return CreateImpl(request, directory, std::move(entries));
}
StatusOr<RuntimeCanonicalIndexWriter> RuntimeCanonicalIndexWriter::CreateImpl(
    const RuntimeCanonicalIndexWriteRequest& request,
    const std::filesystem::path& adopted_directory,
    std::vector<CanonicalCidEntry> adopted_entries) {
  const ModelShape shape = request.model.shape;
  if (request.destination.empty()) {
    return Status::InvalidArgument(
        "canonical index destination must not be empty");
  }
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (request.counts.real == 0U ||
      request.counts.real_roots + request.counts.linked_real !=
          request.counts.real ||
      request.hashes.linkage_list_directory.empty() ||
      (request.publish_quantized_forest &&
       (request.hashes.coefficient_codec.empty() ||
        request.hashes.norm8_cache.empty())) ||
      (request.retain_float_source_norm2 && !request.retain_float_forest) ||
      (!request.publish_quantized_forest &&
       (!request.hashes.coefficient_codec.empty() ||
        !request.hashes.norm8_cache.empty() || !request.retain_float_forest ||
        request.producer_recipe.has_value()))) {
    return Status::InvalidArgument(
        "canonical write request shape, counts or hashes are invalid");
  }
  if ((request.publish_quantized_forest &&
       request.norm8_centers.size() != 256U) ||
      (!request.publish_quantized_forest && !request.norm8_centers.empty())) {
    return Status::InvalidArgument(
        "canonical writer requires the registered 256-center classic norm8 "
        "table");
  }
  const bool adopted = !adopted_directory.empty();
  if (adopted && adopted_entries.size() != shape.nlist)
    return Status::InvalidArgument("adoption requires every CID");
  auto publication = adopted ? StagingDirectory::OpenExisting(
                                   adopted_directory, request.destination)
                             : StagingDirectory::Create(request.destination);
  if (!publication.ok()) {
    return publication.status();
  }
  std::unique_ptr<State> state(new State);
  state->publication = std::move(publication).value();
  state->adopted_directory = adopted_directory;
  state->adopted_entries = std::move(adopted_entries);
  state->shape = shape;
  state->counts = request.counts;
  state->hashes = request.hashes;
  state->norm_backend=request.norm_backend;
  std::error_code layout_error;
  std::filesystem::create_directories(
      state->publication.path() / PublishedNorm2Directory(request.hashes),
      layout_error);
  if (!layout_error)
    std::filesystem::create_directories(
        state->publication.path() / "norm-cache/constant", layout_error);
  if (layout_error)
    return Status::Internal("cannot create norm2 directories: " +
                            layout_error.message());
  state->producer_recipe = request.producer_recipe;
  if (state->producer_recipe.has_value() &&
      !state->producer_recipe->coefficient_bits_per_layer.empty()) {
    const auto& base = state->producer_recipe->coefficient_bits_per_layer;
    const auto& root =
        state->producer_recipe->coefficient_bits_per_layer_root.empty()
            ? base
            : state->producer_recipe->coefficient_bits_per_layer_root;
    const auto& linkage =
        state->producer_recipe->coefficient_bits_per_layer_linkage.empty()
            ? base
            : state->producer_recipe->coefficient_bits_per_layer_linkage;
    const auto valid_bits = [&shape](const std::vector<std::uint32_t>& bits) {
      return bits.size() == shape.m &&
             std::all_of(bits.begin(), bits.end(), [](std::uint32_t value) {
               return value >= kMinimumCoefficientBits &&
                      value <= kMaximumCoefficientBits;
             });
    };
    if (!valid_bits(base) || !valid_bits(root) || !valid_bits(linkage)) {
      return Status::InvalidArgument(
          "canonical coefficient width recipe is invalid");
    }
    state->coefficient_alphabet_size = CoefficientAlphabetFromBits(
        MaximumCoefficientBits(base, root, linkage));
  }
  state->norm8_center_count =
      static_cast<std::uint32_t>(request.norm8_centers.size());
  state->retain_float_forest = request.retain_float_forest;
  state->retain_float_norm2 = request.retain_float_norm2;
  state->retain_float_source_norm2 = request.retain_float_source_norm2;
  state->publish_quantized_forest = request.publish_quantized_forest;
  state->entries.reserve(shape.nlist);

  std::vector<std::uint8_t> model;
  if (const Status status =
          CanonicalModelCodec::EncodeRuntime(request.model, model);
      !status.ok()) {
    return status;
  }
  if (request.model_reference.empty()) {
    if (const Status status = WriteFile(state->publication.path() / "model.bin",
                                        model, "model shard");
        !status.ok())
      return status;
  } else {
    ShardDescriptor descriptor{ShardKind::kCodebooks, 0U, "model.bin",
                               model.size(), 1U};
    if (request.model_reference.extension() == ".stlqmodel") {
      auto trained = TrainedModelFile::Open(request.model_reference);
      if (!trained.ok()) return trained.status();
      const auto bytes = trained.value().model_bytes();
      if (bytes.size() != model.size() ||
          std::memcmp(bytes.data(), model.data(), model.size()) != 0)
        return Status::DataLoss(
            "independent trained model disagrees with index model");
      descriptor.payload_offset = 112U;  // TrainedModelFile v1 wrapper.
    } else {
      auto mapped = ReadOnlyMappedFile::Open(request.model_reference);
      if (!mapped.ok()) return mapped.status();
      if (mapped.value().size() != model.size() ||
          std::memcmp(mapped.value().data(), model.data(), model.size()) != 0)
        return Status::DataLoss(
            "independent checkpoint model disagrees with index model");
    }
    descriptor.relative_path =
        request.model_reference.filename().generic_string();
    auto reference = ReferenceCanonicalShard(
        request.model_reference.parent_path(), request.destination, descriptor);
    if (!reference.ok()) return reference.status();
    reference.value().relative_path = "model.bin";
    state->model_reference = std::move(reference).value();
  }
  state->model_bytes = model.size();

  if (!adopted) {
    if (const Status status =
            OpenOutput(state->publication.path() / "codes.bin", state->codes,
                       "code shard");
        !status.ok()) {
      return status;
    }
    if (const Status status =
            OpenOutput(state->publication.path() / "parents.bin",
                       state->parents, "parent shard");
        !status.ok()) {
      return status;
    }
  }
  if (state->publish_quantized_forest) {
    if (!adopted) {
      if (const Status status =
              OpenOutput(state->publication.path() / "coefficients.bin",
                         state->coefficients, "coefficient shard");
          !status.ok()) {
        return status;
      }
    }
    if (const Status status = OpenOutput(
            state->publication.path() / PublishedNorm2Directory(state->hashes) /
                "norm2_labels.u8",
            state->norm8, "norm8 shard");
        !status.ok()) {
      return status;
    }
  }
  if (state->retain_float_forest) {
    if (!adopted) {
      if (const Status status = OpenOutput(
              state->publication.path() / "float_coefficients.f32",
              state->float_coefficients, "native-float coefficient shard");
          !status.ok()) {
        return status;
      }
    }
    if (state->retain_float_norm2) {
      if (const Status status =
              OpenOutput(state->publication.path() /
                             "norm-cache/constant/exact_float_norm2.f32",
                         state->float_norm2, "native-float norm shard");
          !status.ok()) {
        return status;
      }
    }
  }
  if (state->retain_float_source_norm2) {
    if (const Status status = OpenOutput(
            state->publication.path() /
                "norm-cache/constant/legacy_original_vector_norm2.f32",
            state->float_source_norm2, "float source norm shard");
        !status.ok()) {
      return status;
    }
  }

  std::vector<std::uint8_t> header;
  if (!adopted) {
    if (const Status status = WritePayloadHeader(
            state->codes, ShardKind::kCodes, shape, header, "code header");
        !status.ok()) {
      return status;
    }
    state->codes_cursor = header.size();
    if (const Status status =
            WritePayloadHeader(state->parents, ShardKind::kParents, shape,
                               header, "parent header");
        !status.ok()) {
      return status;
    }
    state->parents_cursor = header.size();
  }
  if (state->publish_quantized_forest) {
    if (!adopted) {
      if (const Status status =
              WritePayloadHeader(state->coefficients, ShardKind::kCoefficients,
                                 shape, header, "coefficient header");
          !status.ok()) {
        return status;
      }
      state->coefficients_cursor = header.size();
    }
    header.resize(1024U);
    for (unsigned label = 0; label < 256; ++label)
      StoreLittleEndianF32(request.norm8_centers[label],
                           header.data() + label * 4U);
    if (auto status = WriteFile(state->publication.path() /
                                    PublishedNorm2Directory(state->hashes) /
                                    "norm2_lut.f32",
                                header, "norm2 LUT");
        !status.ok())
      return status;
    state->norm8_cursor = 0U;
  }
  if (state->retain_float_forest) {
    if (!adopted) {
      if (const Status status = WritePayloadHeader(
              state->float_coefficients, ShardKind::kFloatCoefficients, shape,
              header, "native-float coefficient header");
          !status.ok()) {
        return status;
      }
      state->float_coefficients_cursor = header.size();
    }
    if (state->retain_float_norm2) {
      if (const Status status =
              WritePayloadHeader(state->float_norm2, ShardKind::kFloatNorm2,
                                 shape, header, "native-float norm header");
          !status.ok()) {
        return status;
      }
      state->float_norm2_cursor = header.size();
    }
  }
  if (state->retain_float_source_norm2) {
    if (const Status status = WritePayloadHeader(
            state->float_source_norm2, ShardKind::kFloatSourceNorm2, shape,
            header, "float source norm header");
        !status.ok()) {
      return status;
    }
    state->float_source_norm2_cursor = header.size();
  }
  if (adopted) {
    for (auto item :
         {std::pair{"codes.bin", &state->codes_cursor},
          std::pair{"parents.bin", &state->parents_cursor},
          std::pair{"coefficients.bin", &state->coefficients_cursor},
          std::pair{"float_coefficients.f32",
                    &state->float_coefficients_cursor}}) {
      if (std::string_view(item.first) == "coefficients.bin" &&
          !state->publish_quantized_forest)
        continue;
      if (std::string_view(item.first) == "float_coefficients.f32" &&
          !state->retain_float_forest)
        continue;
      std::error_code error;
      *item.second =
          std::filesystem::file_size(adopted_directory / item.first, error);
      if (error)
        return Status::DataLoss("cannot inspect adopted canonical stream: " +
                                error.message());
    }
  }
  return RuntimeCanonicalIndexWriter(std::move(state));
}

Status RuntimeCanonicalIndexWriter::SetScaledScore8(
    ScaledScore8CodebookView codebook, ArtifactHash64 hash) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "canonical writer is not accepting score tables");
  }
  if (!state_->publish_quantized_forest) {
    return Status::FailedPrecondition(
        "scaled score8 cannot be attached to a float-only index");
  }
  if (state_->scaled_score8_bytes != 0U ||
      !state_->scaled_score8_hash.empty()) {
    return Status::FailedPrecondition(
        "canonical writer already has a scaled score8 table");
  }
  if (!codebook.enabled() || codebook.center_count != 256U || hash.empty()) {
    return Status::InvalidArgument(
        "scaled score8 requires 256 finite scale/bias pairs and a hash");
  }
  std::vector<std::uint8_t> bytes;
  if (const Status status = CanonicalPayloadShardCodec::EncodeHeader(
          ShardKind::kScaledScore8, state_->shape, bytes);
      !status.ok()) {
    return status;
  }
  const std::size_t header = bytes.size();
  bytes.resize(header + 2U * codebook.center_count * sizeof(float));
  for (std::uint32_t label = 0U; label < codebook.center_count; ++label) {
    const float scale = codebook.scales[label];
    const float bias = codebook.biases[label];
    if (!std::isfinite(scale) || scale < 0.0F || !std::isfinite(bias) ||
        bias < 0.0F) {
      return Status::InvalidArgument(
          "scaled score8 table contains a non-finite or negative value");
    }
    StoreLittleEndianF32(scale,
                         bytes.data() + header + (2U * label) * sizeof(float));
    StoreLittleEndianF32(
        bias, bytes.data() + header + (2U * label + 1U) * sizeof(float));
  }
  if (const Status status = WriteFile(
          state_->publication.path() / PublishedNorm2Directory(state_->hashes) /
              "norm2_scale_bias.f32",
          bytes, "scaled score8 shard");
      !status.ok()) {
    return status;
  }
  state_->scaled_score8_bytes = bytes.size();
  state_->scaled_score8_hash = hash;
  return Status::Ok();
}

Status RuntimeCanonicalIndexWriter::AppendEncoded(
    const CanonicalEncodedClusterView& source) {
  return AppendEncodedImpl(source, false);
}

Status RuntimeCanonicalIndexWriter::AppendEncodedValidatedFloat(
    const CanonicalEncodedClusterView& source) {
  return AppendEncodedImpl(source, true);
}

Status RuntimeCanonicalIndexWriter::AppendEncodedImpl(
    const CanonicalEncodedClusterView& source, bool validated_float) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("canonical writer is not accepting CIDs");
  }
  if (source.cid != state_->entries.size() ||
      source.cid >= state_->shape.nlist) {
    return Status::InvalidArgument(
        "canonical CIDs must be appended exactly once in ascending order");
  }
  assert(!state_->publish_quantized_forest ||
         state_->norm8_center_count == 256U);
  const bool adopted = !state_->adopted_directory.empty();
  auto checked = source;
  if (adopted) {
    checked.float_coefficients_layer_major = {};
    checked.float_real_norm2 = {};
  }
  if (const Status status = ValidateEncodedCluster(
          state_->shape.m, checked, state_->retain_float_forest && !adopted,
          state_->retain_float_norm2 && !adopted,
          state_->retain_float_source_norm2, state_->publish_quantized_forest,
          validated_float);
      !status.ok()) {
    return status;
  }
  CanonicalCidEntry entry =
      adopted ? state_->adopted_entries[source.cid] : CanonicalCidEntry{};
  entry.cid = source.cid;
  entry.real_count = source.real_count;
  entry.virtual_count = source.virtual_count;
  entry.real_root_count = source.real_root_count;
  entry.codes_bytes = source.records.codes.size();
  entry.parents_bytes = source.records.parents.size();
  entry.coefficients_bytes = source.records.coefficients.size();
  entry.norm8_bytes = source.records.norm8.size();
  const std::uint64_t nodes =
      static_cast<std::uint64_t>(source.real_count) + source.virtual_count;
  const std::uint64_t raw_parent_bits = 2U * nodes + 1U;
  std::uint64_t root_payload_bytes = 0U;
  std::uint64_t linkage_payload_bytes = 0U;
  if (state_->publish_quantized_forest) {
    root_payload_bytes =
        LoadLittleEndianU32(source.records.coefficients.data());
    linkage_payload_bytes =
        LoadLittleEndianU32(source.records.coefficients.data() + 4U);
    if (CanonicalCoefficientAlphabetFromRecord(
            state_->shape.m, source.records.coefficients.size(),
            root_payload_bytes,
            linkage_payload_bytes) != state_->coefficient_alphabet_size) {
      return Status::DataLoss(
          "coefficient record alphabet disagrees with its producer recipe");
    }
  }
  state_->code_record_bytes += entry.codes_bytes;
  state_->parent_record_bytes += entry.parents_bytes;
  state_->coefficient_record_bytes += entry.coefficients_bytes;
  state_->coefficient_header_bytes +=
      entry.coefficients_bytes - root_payload_bytes - linkage_payload_bytes;
  state_->root_coefficient_payload_bytes += root_payload_bytes;
  state_->linkage_coefficient_payload_bytes += linkage_payload_bytes;
  state_->norm8_label_bytes += entry.norm8_bytes;
  state_->parent_raw_bits += raw_parent_bits;
  state_->parent_word_padding_bits +=
      entry.parents_bytes * 8U - raw_parent_bits;
  if (!adopted) {
    if (const Status status = AppendAligned(state_->codes, state_->codes_cursor,
                                            source.records.codes,
                                            entry.codes_offset, "code record");
        !status.ok()) {
      return status;
    }
  }
  if (state_->retain_float_forest) {
    if (!adopted) {
      entry.float_coefficients_offset =
          AlignCanonicalRecord(state_->float_coefficients_cursor);
      if (const Status status = AppendF32Aligned(
              state_->float_coefficients, state_->float_coefficients_cursor,
              source.float_coefficients_layer_major,
              "native-float coefficient record");
          !status.ok()) {
        return status;
      }
    }
    if (state_->retain_float_norm2) {
      if (const Status status = AppendF32Aligned(
              state_->float_norm2, state_->float_norm2_cursor,
              source.float_real_norm2, "native-float norm record");
          !status.ok()) {
        return status;
      }
    }
  }
  if (state_->retain_float_source_norm2) {
    if (const Status status = AppendF32Aligned(
            state_->float_source_norm2, state_->float_source_norm2_cursor,
            source.float_source_norm2, "float source norm record");
        !status.ok()) {
      return status;
    }
  }
  if (!adopted) {
    if (const Status status = AppendAligned(
            state_->parents, state_->parents_cursor, source.records.parents,
            entry.parents_offset, "parent record");
        !status.ok()) {
      return status;
    }
  }
  if (state_->publish_quantized_forest) {
    if (!adopted) {
      if (const Status status =
              AppendAligned(state_->coefficients, state_->coefficients_cursor,
                            source.records.coefficients,
                            entry.coefficients_offset, "coefficient record");
          !status.ok()) {
        return status;
      }
    }
    if (const Status status = AppendAligned(state_->norm8, state_->norm8_cursor,
                                            source.records.norm8,
                                            entry.norm8_offset, "norm8 record");
        !status.ok()) {
      return status;
    }
  }
  state_->entries.push_back(entry);
  return Status::Ok();
}

Status RuntimeCanonicalIndexWriter::Commit(
    CanonicalIndexWriteStatistics* statistics) {
  if (state_ == nullptr) {
    return Status::FailedPrecondition(
        "canonical writer cannot be committed twice");
  }
  if (state_->entries.size() != state_->shape.nlist) {
    return Status::FailedPrecondition(
        "canonical writer has not received every native-IVF CID");
  }
  std::vector<std::uint8_t> directory;
  if (const Status status = CanonicalDirectoryCodec::Encode(
          state_->shape, state_->counts, state_->entries, directory,
          state_->publish_quantized_forest, true,
          !state_->adopted_directory.empty());
      !status.ok()) {
    return status;
  }
  const bool adopted = !state_->adopted_directory.empty();
  if (!adopted) {
    if (const Status status = CloseOutput(state_->codes, "code shard");
        !status.ok()) {
      return status;
    }
    if (const Status status = CloseOutput(state_->parents, "parent shard");
        !status.ok()) {
      return status;
    }
  }
  if (state_->publish_quantized_forest) {
    if (!adopted) {
      if (const Status status =
              CloseOutput(state_->coefficients, "coefficient shard");
          !status.ok()) {
        return status;
      }
    }
    if (const Status status = CloseOutput(state_->norm8, "norm8 shard");
        !status.ok()) {
      return status;
    }
  }
  if (state_->retain_float_forest) {
    if (!adopted) {
      if (const Status status = CloseOutput(state_->float_coefficients,
                                            "native-float coefficient shard");
          !status.ok()) {
        return status;
      }
    }
    if (state_->retain_float_norm2) {
      if (const Status status =
              CloseOutput(state_->float_norm2, "native-float norm shard");
          !status.ok()) {
        return status;
      }
    }
  }
  if (state_->retain_float_source_norm2) {
    if (const Status status =
            CloseOutput(state_->float_source_norm2, "float source norm shard");
        !status.ok()) {
      return status;
    }
  }
  if (const Status status =
          WriteFile(state_->publication.path() / "directory.bin", directory,
                    "CID directory");
      !status.ok()) {
    return status;
  }

  IndexManifest manifest;
  if (!state_->publish_quantized_forest) {
    manifest.format_version = IndexManifest::kFloatOnlyFormatVersion;
  } else if (!state_->producer_recipe.has_value()) {
    manifest.format_version = IndexManifest::kLegacyFormatVersion;
  } else if (state_->producer_recipe->norm8_mode ==
                 ReconstructionNorm8Mode::kIdentityLloyd &&
             state_->producer_recipe->coefficient_first_magnitude_quantiles
                     .size() == 1U &&
             state_->producer_recipe->coefficient_rest_magnitude_quantiles
                     .size() == 1U &&
             state_->producer_recipe->coefficient_first_magnitude_quantiles
                     .front() ==
                 state_->producer_recipe->coefficient_rest_magnitude_quantiles
                     .front()) {
    // Keep accepted one-candidate manifests byte-compatible with format v2.
    manifest.format_version = IndexManifest::kScalarRecipeFormatVersion;
  } else if (state_->producer_recipe->norm8_mode ==
             ReconstructionNorm8Mode::kIdentityLloyd) {
    // Keep accepted multi-candidate identity-LUT manifests byte-compatible
    // with format v3.
    manifest.format_version = IndexManifest::kMultiCandidateRecipeFormatVersion;
  }
  manifest.shape = state_->shape;
  manifest.counts = state_->counts;
  manifest.hashes = state_->hashes;
  manifest.norm_codec = state_->scaled_score8_bytes == 0U
                            ? NormCodec::kReconstructionNorm8
                            : NormCodec::kReconstructionNorm8WithScaledScore8;
  manifest.scaled_score8_hash = state_->scaled_score8_hash;
  manifest.producer_recipe = state_->producer_recipe;
  const bool configurable_coefficients =
      state_->producer_recipe.has_value() &&
      !state_->producer_recipe->coefficient_bits_per_layer.empty();
  const bool configurable_codec =
      state_->producer_recipe.has_value() &&
      (!state_->producer_recipe->coefficient_weighted_quantile ||
       !state_->producer_recipe->coefficient_allow_clipping ||
       !state_->producer_recipe->coefficient_fit_scale ||
       state_->producer_recipe->coefficient_refinement_sweeps != 3U ||
       state_->producer_recipe->coefficient_refinement_max_layer != 0U ||
       state_->producer_recipe->coefficient_refinement_step_limit != 1U);
  if (state_->publish_quantized_forest) {
    if (configurable_codec) {
      manifest.format_version = IndexManifest::kConfigurableCodecFormatVersion;
    } else if (configurable_coefficients) {
      manifest.format_version =
          IndexManifest::kConfigurableCoefficientFormatVersion;
    } else if (state_->scaled_score8_bytes != 0U) {
      manifest.format_version = IndexManifest::kScaledScoreFormatVersion;
    } else if (state_->retain_float_forest) {
      manifest.format_version = IndexManifest::kCurrentFormatVersion;
    } else if (manifest.format_version ==
               IndexManifest::kCurrentFormatVersion) {
      manifest.format_version = IndexManifest::kNormModeRecipeFormatVersion;
    }
  }
  if (state_->retain_float_source_norm2) {
    manifest.format_version = IndexManifest::kFloatScaledFormatVersion;
  }
  if (state_->retain_float_forest && !state_->retain_float_norm2) {
    manifest.format_version = IndexManifest::kIndependentNormFormatVersion;
  }
  manifest.shards = MakeShards(
      state_->model_bytes, directory.size(), state_->codes_cursor,
      state_->parents_cursor, state_->coefficients_cursor, state_->norm8_cursor,
      state_->float_coefficients_cursor, state_->float_norm2_cursor,
      state_->scaled_score8_bytes, state_->float_source_norm2_cursor,
      state_->hashes, state_->shape, state_->counts,
      state_->retain_float_forest, state_->publish_quantized_forest,
      state_->scaled_score8_bytes != 0U, state_->retain_float_source_norm2);
  if (state_->model_reference) {
    manifest.format_version = IndexManifest::kReferencedShardFormatVersion;
    manifest.shards.front() = *state_->model_reference;
  }
  manifest.format_version = adopted
                                ? IndexManifest::kCompletionOrderFormatVersion
                                : IndexManifest::kNormCacheLayoutFormatVersion;
  if (auto status =
          WritePublishedNorm2Metadata(state_->publication.path(), manifest,state_->norm_backend.c_str());
      !status.ok())
    return status;
  std::vector<std::uint8_t> manifest_bytes;
  if (const Status status =
          IndexManifestCodec::Encode(manifest, manifest_bytes);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          WriteFile(state_->publication.path() / "manifest.stlqpp",
                    manifest_bytes, "manifest");
      !status.ok()) {
    return status;
  }
  if (auto status = state_->publication.Publish(); !status.ok()) return status;

  if (statistics != nullptr) {
    *statistics = {
        .shape = state_->shape,
        .counts = state_->counts,
        .model_shard_bytes = state_->model_bytes,
        .directory_shard_bytes = directory.size(),
        .codes_shard_bytes = state_->codes_cursor,
        .parents_shard_bytes = state_->parents_cursor,
        .coefficients_shard_bytes = state_->coefficients_cursor,
        .norm8_shard_bytes = state_->norm8_cursor +
                             (state_->publish_quantized_forest ? 1024U : 0U),
        .scaled_score8_shard_bytes = state_->scaled_score8_bytes,
        .float_source_norm2_shard_bytes = state_->float_source_norm2_cursor,
        .float_coefficients_shard_bytes = state_->float_coefficients_cursor,
        .float_norm2_shard_bytes = state_->float_norm2_cursor,
        .manifest_bytes = manifest_bytes.size(),
        .code_record_bytes = state_->code_record_bytes,
        .parent_record_bytes = state_->parent_record_bytes,
        .coefficient_record_bytes = state_->coefficient_record_bytes,
        .coefficient_header_bytes = state_->coefficient_header_bytes,
        .root_coefficient_payload_bytes =
            state_->root_coefficient_payload_bytes,
        .linkage_coefficient_payload_bytes =
            state_->linkage_coefficient_payload_bytes,
        .norm8_label_bytes = state_->norm8_label_bytes,
        .parent_raw_bits = state_->parent_raw_bits,
        .parent_word_padding_bits = state_->parent_word_padding_bits,
    };
  }
  state_.reset();
  return Status::Ok();
}

template <int M>
CanonicalIndexWriter<M>::CanonicalIndexWriter(
    RuntimeCanonicalIndexWriter writer) noexcept
    : writer_(std::move(writer)) {}
template <int M>
CanonicalIndexWriter<M>::~CanonicalIndexWriter() = default;
template <int M>
CanonicalIndexWriter<M>::CanonicalIndexWriter(CanonicalIndexWriter&&) noexcept =
    default;
template <int M>
CanonicalIndexWriter<M>& CanonicalIndexWriter<M>::operator=(
    CanonicalIndexWriter&&) noexcept = default;

template <int M>
StatusOr<CanonicalIndexWriter<M>> CanonicalIndexWriter<M>::Create(
    const CanonicalIndexWriteRequest<M>& request) {
  if (M != 0 && request.model.codebooks.shape.m != M)
    return Status::InvalidArgument(
        "canonical writer M disagrees with model shape");
  const auto& books = request.model.codebooks;
  auto writer = RuntimeCanonicalIndexWriter::Create(
      {.destination = request.destination,
       .model = {.shape = books.shape,
                 .rotation = request.model.rotation,
                 .root0 = books.root0,
                 .root_small = books.root_small,
                 .edge = books.edge},
       .counts = request.counts,
       .hashes = request.hashes,
       .producer_recipe = request.producer_recipe,
       .norm8_centers = request.norm8_centers,
       .retain_float_forest = request.retain_float_forest,
       .retain_float_norm2 = request.retain_float_norm2,
       .retain_float_source_norm2 = request.retain_float_source_norm2,
       .publish_quantized_forest = request.publish_quantized_forest,
       .model_reference = request.model_reference,
       .norm_backend=request.norm_backend});
  if (!writer.ok()) return writer.status();
  return CanonicalIndexWriter(std::move(writer).value());
}

template <int M>
Status CanonicalIndexWriter<M>::Append(
    const CanonicalClusterSource<M>& source) {
  CanonicalClusterRecords records;
  if (const Status status = CanonicalClusterCodec<M>::Encode(source, records);
      !status.ok()) {
    return status;
  }
  return AppendEncoded({
      .cid = source.cid,
      .real_count = static_cast<std::uint32_t>(source.real_ids.size()),
      .virtual_count = source.virtual_count,
      .real_root_count = source.real_root_count,
      .records =
          {
              .codes = records.codes,
              .parents = records.parents,
              .coefficients = records.coefficients,
              .norm8 = records.norm8,
          },
      .float_coefficients_layer_major = {},
      .float_real_norm2 = {},
      .float_source_norm2 = {},
  });
}

template <int M>
Status CanonicalIndexWriter<M>::AppendEncoded(
    const CanonicalEncodedClusterView& source) {
  return writer_.AppendEncoded(source);
}
template <int M>
Status CanonicalIndexWriter<M>::AppendEncodedValidatedFloat(
    const CanonicalEncodedClusterView& source) {
  return writer_.AppendEncodedValidatedFloat(source);
}
template <int M>
Status CanonicalIndexWriter<M>::SetScaledScore8(ScaledScore8CodebookView table,
                                                ArtifactHash64 hash) {
  return writer_.SetScaledScore8(table, hash);
}
template <int M>
Status CanonicalIndexWriter<M>::Commit(
    CanonicalIndexWriteStatistics* statistics) {
  return writer_.Commit(statistics);
}

template class CanonicalIndexWriter<0>;
template class CanonicalIndexWriter<5>;
template class CanonicalIndexWriter<12>;
template class CanonicalIndexWriter<6>;
template class CanonicalIndexWriter<10>;

}  // namespace stlqpp
