#include "stlqpp/application/scaled_score8_derivation.h"

#include "stlqpp/index/norm/reconstruction_norm8.h"
#include "stlqpp/index/norm/runtime_reconstruction_norms.h"
#include "stlqpp/index/norm/reconstruction_norms.h"
#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/storage/norm_cache_layout.h"
#if STLQPP_HAS_CUDA_NORM
#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"
#include "stlqpp/index/norm/reconstruction_norms_cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/file_replication.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/canonical_shard_codec.h"
#include "stlqpp/storage/index_manifest_codec.h"

namespace stlqpp {

struct NormOnlyDerivationAccess final {
  static StatusOr<ReconstructionNorm8Codebook> Fit(
      std::vector<float> values, ReconstructionNorm8Mode mode,
      std::uint32_t iterations) {
    const InlineParallelFor serial;
    return ReconstructionNorm8Codebook::FitFromValidatedValues(
        std::move(values), iterations, serial, mode);
  }
  static Status Encode(const ReconstructionNorm8Codebook& codebook,
                       ArrayView<const float> norms,
                       ArrayView<std::uint8_t> labels) {
    return codebook.EncodeValidated(norms, labels);
  }
};

namespace {

constexpr std::uint32_t kClustersPerBatch = 256U;

StatusOr<std::vector<std::uint8_t>> EncodeScaleTable(
    ModelShape shape, ScaledScore8CodebookView table) {
  std::vector<std::uint8_t> bytes;
  if (auto status = CanonicalPayloadShardCodec::EncodeHeader(
          ShardKind::kScaledScore8, shape, bytes); !status.ok()) return status;
  const auto header = bytes.size();
  bytes.resize(header + 2048U);
  for (unsigned l = 0; l < 256U; ++l) {
    StoreLittleEndianF32(table.scales[l], bytes.data() + header + l * 8U);
    StoreLittleEndianF32(table.biases[l], bytes.data() + header + l * 8U + 4U);
  }
  return bytes;
}

RuntimeReconstructionNormInput MakeNormInput(
    const RuntimeDecodedCanonicalCluster& cluster,
    std::vector<std::uint32_t>* parents) {
  const auto& t = cluster.topology;
  const auto& c = cluster.coefficients;
  const auto reals = static_cast<std::uint32_t>(t.real_ids.size());
  parents->assign(reals, 0U);
  ParentLoudsSequentialDecoder decoder(t.parent_louds_words);
  decoder.InitializeAfterValidatedRootPrefix(t.virtual_count +
                                             t.real_root_count);
  for (std::uint32_t real = t.real_root_count; real < reals; ++real)
    (*parents)[real] = decoder.NextParent1Based();
  return {{t.cid, t.virtual_count, t.real_root_count, reals - t.real_root_count,
           t.virtual_codes_small, t.real_codes_small, t.edge_code0},
          *parents,
          c.coefficients_layer_major,
          c.root_scales,
          c.linkage_scales};
}

struct FitContext final {
  const CanonicalIndexFiles* files = nullptr;
  const RuntimeGramReconstructionNormComputer* norm_computer = nullptr;
#if STLQPP_HAS_CUDA_NORM
  std::vector<std::unique_ptr<RuntimeCudaReconstructionNormComputer>>* gpu_norm_computers = nullptr;
#endif
  const SourceNormProvider* source = nullptr;
  std::uint32_t first_cid = 0U;
  std::vector<ScaledScore8Accumulator>* accumulators = nullptr;
  std::vector<Status>* statuses = nullptr;
};

void FitCluster(void* opaque, std::uint32_t worker, std::uint32_t local) noexcept {
  auto& context = *static_cast<FitContext*>(opaque);
  (void)worker;
  const std::uint32_t cid = context.first_cid + local;
  const CanonicalCidEntry& entry = context.files->entry(cid);
  auto decoded = DecodeCanonicalNativeBfsClusterRuntime(
      context.files->manifest().shape.m, cid, entry.real_count,
      entry.virtual_count, entry.real_root_count,
      context.files->cluster_records(cid), context.files->norm8());
  if (!decoded.ok()) {
    (*context.statuses)[local] = decoded.status();
    return;
  }
  const auto& values = decoded.value();
  std::vector<std::uint32_t> parents;
  const auto norm_input = MakeNormInput(values, &parents);
#if STLQPP_HAS_CUDA_NORM
  auto reconstruction = context.gpu_norm_computers != nullptr
                            ? (*context.gpu_norm_computers)[worker]->Compute(norm_input)
                            : context.norm_computer->Compute(norm_input);
#else
  auto reconstruction = context.norm_computer->Compute(norm_input);
#endif
  if (!reconstruction.ok()) {
    (*context.statuses)[local] = reconstruction.status();
    return;
  }
  std::vector<float> source_norm2;
  if (context.source != nullptr) {
    if (const Status status =
            context.source->Read(values.topology.real_ids, &source_norm2);
        !status.ok()) {
      (*context.statuses)[local] = status;
      return;
    }
  } else {
    source_norm2.resize(values.topology.real_ids.size());
    if (const Status status =
            context.files->ReadFloatSourceNorm2(cid, source_norm2);
        !status.ok()) {
      (*context.statuses)[local] = status;
      return;
    }
  }
  (*context.statuses)[local] = (*context.accumulators)[local].Accumulate(
      values.real_norm8, reconstruction.value(), source_norm2);
}

}  // namespace

static StatusOr<ScaledScore8Codebook> FitScaledScore8Impl(
    const std::filesystem::path& index, const SourceNormProvider* source,
    const ParallelForExecutor& parallel, bool use_cuda_norm, int cuda_device,
    std::uint32_t expected_m) {
  auto opened = CanonicalIndexFiles::OpenQuantized(index);
  if (!opened.ok()) return opened.status();
  const CanonicalIndexFiles& files = opened.value();
  std::optional<SourceNormProvider> shared_source;
  if (source == nullptr) {
    auto original = SourceNormProvider::OpenForIndex(
        index, files.manifest(), {}, "auto", 0U, 0U, parallel);
    if (!original.ok()) return original.status();
    shared_source.emplace(std::move(original).value());
    source = &*shared_source;
  }
  if ((expected_m != 0U && files.manifest().shape.m != expected_m) ||
      !files.manifest().has_classic_norm8()) {
    return Status::FailedPrecondition(
        "scaled score fitting requires a matching classic norm8 index");
  }
  auto model = CanonicalModelCodec::DecodeRuntime(files.model_bytes().data(),
                                                  files.model_bytes().size());
  if (!model.ok()) return model.status();
  std::optional<RuntimeGramReconstructionNormComputer> norm_computer;
  if (!use_cuda_norm) {
    auto created = RuntimeGramReconstructionNormComputer::Create(model.value().codebooks_view());
    if (!created.ok()) return created.status();
    norm_computer.emplace(std::move(created).value());
  }
#if STLQPP_HAS_CUDA_NORM
  std::vector<std::unique_ptr<RuntimeCudaReconstructionNormComputer>> gpu_norm_computers;
  if (use_cuda_norm) {
    auto created = RuntimeCudaReconstructionNormComputer::Create(
        model.value().codebooks_view(), cuda_device);
    if (!created.ok()) return created.status();
    gpu_norm_computers.push_back(std::move(created).value());
    for (std::uint32_t worker = 1; worker < parallel.worker_count(); ++worker) {
      auto clone = gpu_norm_computers.front()->Clone();
      if (!clone.ok()) return clone.status();
      gpu_norm_computers.push_back(std::move(clone).value());
    }
  }
#else
  (void)cuda_device;
  if (use_cuda_norm) {
    return Status::Unimplemented("GPU norm reconstruction requires CUDA build");
  }
#endif
  ScaledScore8Accumulator complete;
  for (std::uint32_t first = 0U; first < files.manifest().shape.nlist;
       first += kClustersPerBatch) {
    const std::uint32_t count =
        std::min(kClustersPerBatch, files.manifest().shape.nlist - first);
    std::vector<ScaledScore8Accumulator> accumulators(count);
    std::vector<Status> statuses(count);
    FitContext context = {
        .files = &files,
        .norm_computer = norm_computer ? &*norm_computer : nullptr,
#if STLQPP_HAS_CUDA_NORM
        .gpu_norm_computers = use_cuda_norm ? &gpu_norm_computers : nullptr,
#endif
        .source = source,
        .first_cid = first,
        .accumulators = &accumulators,
        .statuses = &statuses,
    };
    parallel.Run(count, &context, &FitCluster);
    for (std::uint32_t local = 0U; local < count; ++local) {
      if (!statuses[local].ok()) return statuses[local];
      complete.Merge(accumulators[local]);
    }
  }
  return complete.Finish(files.norm8());
}

StatusOr<ScaledScore8Codebook> FitScaledScore8Runtime(
    const std::filesystem::path& index, const SourceNormProvider* source,
    const ParallelForExecutor& parallel, bool use_cuda_norm, int cuda_device) {
  return FitScaledScore8Impl(index, source, parallel, use_cuda_norm,
                             cuda_device, 0U);
}

template <int M>
StatusOr<ScaledScore8Codebook> FitScaledScore8(
    const std::filesystem::path& index, const SourceNormProvider* source,
    const ParallelForExecutor& parallel, bool use_cuda_norm, int cuda_device) {
  return FitScaledScore8Impl(index, source, parallel, use_cuda_norm,
                             cuda_device, M);
}

template StatusOr<ScaledScore8Codebook> FitScaledScore8<5>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);
template StatusOr<ScaledScore8Codebook> FitScaledScore8<12>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);
template StatusOr<ScaledScore8Codebook> FitScaledScore8<6>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);
template StatusOr<ScaledScore8Codebook> FitScaledScore8<10>(
    const std::filesystem::path&, const SourceNormProvider*,
    const ParallelForExecutor&, bool, int);

Status PublishScaledScore8(const std::filesystem::path& source,
                           const std::filesystem::path& destination,
                           ScaledScore8CodebookView table) {
  if (source.empty() || destination.empty() || !table.enabled() ||
      table.center_count != 256U) {
    return Status::InvalidArgument(
        "scaled score publication request is incomplete");
  }
  if (const Status status = ValidateIndexDerivationPaths(source, destination);
      !status.ok()) {
    return status;
  }
  for (std::uint32_t center = 0U; center < 256U; ++center) {
    if (!std::isfinite(table.scales[center]) || table.scales[center] < 0.0F ||
        !std::isfinite(table.biases[center]) || table.biases[center] < 0.0F) {
      return Status::InvalidArgument(
          "scaled score table values must be finite and nonnegative");
    }
  }
  auto opened = CanonicalIndexFiles::Open(source);
  if (!opened.ok()) return opened.status();
  const IndexManifest& original = opened.value().manifest();
  if (!original.has_classic_norm8() || original.has_scaled_score8() ||
      opened.value().norm8().center_count != 256U) {
    return Status::FailedPrecondition(
        "source must have classic norm8 and no scaled score table");
  }
  auto staged = StagingDirectory::Create(destination);
  if (!staged.ok()) return staged.status();
  StagingDirectory publication = std::move(staged).value();
  auto encoded = EncodeScaleTable(original.shape, table);
  if (!encoded.ok()) return encoded.status();
  auto score_bytes = std::move(encoded).value();
  const auto write_file = [](const std::filesystem::path& path,
                             const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return output ? Status::Ok()
                  : Status::Internal("failed to write " + path.string());
  };
  std::error_code layout_error;
  const auto norm_directory = PublishedNorm2Directory(original.hashes);
  std::filesystem::create_directories(publication.path() / norm_directory,
                                      layout_error);
  if (layout_error)
    return Status::Internal("cannot create scale directory: " +
                            layout_error.message());
  if (const Status status = write_file(
          publication.path() / norm_directory / "norm2_scale_bias.f32",
          score_bytes);
      !status.ok())
    return status;
  IndexManifest derived = original;
  if (original.format_version !=
          IndexManifest::kConfigurableCoefficientFormatVersion &&
      original.format_version !=
          IndexManifest::kConfigurableCodecFormatVersion &&
      original.format_version != IndexManifest::kFloatScaledFormatVersion &&
      original.format_version != IndexManifest::kIndependentNormFormatVersion) {
    derived.format_version = IndexManifest::kScaledScoreFormatVersion;
  }
  derived.norm_codec = NormCodec::kReconstructionNorm8WithScaledScore8;
  derived.scaled_score8_hash = ArtifactHashFactory::ScaledScore8({
      .coefficient_codec = original.hashes.coefficient_codec,
      .source_norm8_cache = original.hashes.norm8_cache,
      .scales = {table.scales, 256U},
      .biases = {table.biases, 256U},
  });
  derived.shards.push_back({
      .kind = ShardKind::kScaledScore8,
      .ordinal = 0U,
      .relative_path =
          (norm_directory / "norm2_scale_bias.f32").generic_string(),
      .bytes = score_bytes.size(),
      .elements = 512U,
  });
  derived.format_version = original.format_version;
  for (auto& shard : derived.shards) {
    if (shard.kind == ShardKind::kScaledScore8) {
      shard.reference_path.clear();
      shard.reference_sha256.clear();
      shard.payload_offset = 0U;
      continue;
    }
    auto reference = ReferenceCanonicalShard(source, destination, shard);
    if (!reference.ok()) return reference.status();
    shard = std::move(reference).value();
  }
  if (auto status = WritePublishedNorm2Metadata(publication.path(), derived);
      !status.ok())
    return status;
  if (auto status = SharePublishedNorm2Cache(
          publication.path(), CanonicalSourceIndexDirectory(source, original),
          destination, derived);
      !status.ok())
    return status;
  std::vector<std::uint8_t> manifest_bytes;
  if (const Status status = IndexManifestCodec::Encode(derived, manifest_bytes);
      !status.ok())
    return status;
  if (const Status status =
          write_file(publication.path() / "manifest.stlqpp", manifest_bytes);
      !status.ok())
    return status;
  return publication.Publish();
}

static Status PublishReconstructionNorm8Impl(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, bool use_cuda, int cuda_device,
    std::uint32_t expected_m, bool publish_scaled = false,
    const SourceNormProvider* original_norms = nullptr) {
  if (source.empty() || destination.empty() || lloyd_iterations == 0U ||
      (mode != ReconstructionNorm8Mode::kIdentityLloyd &&
       mode != ReconstructionNorm8Mode::kTailWeightedLloyd)) {
    return Status::InvalidArgument("invalid norm-only derivation request");
  }
  if (const Status status = ValidateIndexDerivationPaths(source, destination);
      !status.ok())
    return status;
  auto opened = CanonicalIndexFiles::Open(source);
  if (!opened.ok()) return opened.status();
  const CanonicalIndexFiles& files = opened.value();
  const IndexManifest& original = files.manifest();
  if ((expected_m != 0U && original.shape.m != expected_m) ||
      !original.has_classic_norm8() || !original.producer_recipe.has_value()) {
    return Status::FailedPrecondition(
        "norm-only derivation requires a versioned classic norm8 index");
  }
  std::optional<SourceNormProvider> shared_source;
  if (publish_scaled && original_norms == nullptr) {
    const InlineParallelFor serial;
    auto provider = SourceNormProvider::OpenForIndex(source, original, {}, "auto", 0U, 0U, serial);
    if (!provider.ok()) return provider.status();
    shared_source.emplace(std::move(provider).value());
    original_norms = &*shared_source;
  }
  auto model = CanonicalModelCodec::DecodeRuntime(files.model_bytes().data(),
                                                  files.model_bytes().size());
  if (!model.ok()) return model.status();
  std::optional<RuntimeGramReconstructionNormComputer> computer;
  if (!use_cuda) {
    auto created = RuntimeGramReconstructionNormComputer::Create(model.value().codebooks_view());
    if (!created.ok()) return created.status();
    computer.emplace(std::move(created).value());
  }
#if STLQPP_HAS_CUDA_NORM
  std::unique_ptr<RuntimeCudaReconstructionNormComputer> gpu_computer;
  if (use_cuda) {
    if (mode != ReconstructionNorm8Mode::kIdentityLloyd) {
      return Status::Unimplemented(
          "GPU norm8 tail-weighted mode is not supported");
    }
    auto created = RuntimeCudaReconstructionNormComputer::Create(
        model.value().codebooks_view(), cuda_device);
    if (!created.ok()) return created.status();
    gpu_computer = std::move(created).value();
  }
#else
  (void)cuda_device;
  if (use_cuda) {
    return Status::Unimplemented(
        "GPU norm reconstruction is unavailable in this build");
  }
#endif
  const auto reconstruct =
      [&](std::uint32_t cid) -> StatusOr<std::vector<float>> {
    const CanonicalCidEntry& entry = files.entry(cid);
    auto decoded = DecodeCanonicalNativeBfsClusterRuntime(
        original.shape.m, cid, entry.real_count, entry.virtual_count,
        entry.real_root_count, files.cluster_records(cid), files.norm8());
    if (!decoded.ok()) return decoded.status();
    const auto& values = decoded.value();
    std::vector<std::uint32_t> parents;
    const auto input = MakeNormInput(values, &parents);
#if STLQPP_HAS_CUDA_NORM
    if (use_cuda) return gpu_computer->Compute(input);
#endif
    return computer->Compute(input);
  };
  if (original.counts.real > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "norm-only population exceeds host address domain");
  }
  std::vector<float> norms;
  try {
    norms.reserve(static_cast<std::size_t>(original.counts.real));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "norm-only population does not fit host memory");
  }
  for (std::uint32_t cid = 0U; cid < original.shape.nlist; ++cid) {
    auto cluster = reconstruct(cid);
    if (!cluster.ok()) return cluster.status();
    norms.insert(norms.end(), cluster.value().begin(), cluster.value().end());
  }
  if (norms.size() != original.counts.real) {
    return Status::DataLoss("norm-only population row count disagrees");
  }
  const auto fit_cpu = [&]() -> StatusOr<ReconstructionNorm8Codebook> {
    return NormOnlyDerivationAccess::Fit(std::move(norms), mode, lloyd_iterations);
  };
#if STLQPP_HAS_CUDA_NORM
  auto initialized =
      use_cuda ? FitIdentityNorm8Cuda(norms, lloyd_iterations, cuda_device)
               : fit_cpu();
#else
  auto initialized = fit_cpu();
#endif
  if (!initialized.ok()) return initialized.status();
  ReconstructionNorm8Codebook codebook = std::move(initialized).value();
  // CUDA fitting leaves the population in logical CID order. CPU sorted
  // Lloyd consumes it; that path retains its bounded second reconstruction.
  if (!use_cuda) std::vector<float>().swap(norms);
#if STLQPP_HAS_CUDA_NORM
  std::unique_ptr<CudaNorm8LabelEncoder> gpu_labels;
  if (use_cuda) {
    auto created = CudaNorm8LabelEncoder::Create(codebook.view(), cuda_device);
    if (!created.ok()) return created.status();
    gpu_labels = std::move(created).value();
  }
#endif

  auto staged = StagingDirectory::Create(destination);
  if (!staged.ok()) return staged.status();
  auto publication = std::move(staged).value();
  IndexManifest derived = original;
  derived.format_version =
      original.format_version == IndexManifest::kCompletionOrderFormatVersion
          ? IndexManifest::kCompletionOrderFormatVersion
          : IndexManifest::kNormCacheLayoutFormatVersion;
  derived.norm_codec = NormCodec::kReconstructionNorm8;
  derived.scaled_score8_hash = {};
  derived.producer_recipe->norm8_mode = mode;
  derived.producer_recipe->norm8_lloyd_iterations = lloyd_iterations;
  derived.hashes.norm8_cache = ArtifactHashFactory::Norm8Cache(
      {.linkage_list_directory = original.hashes.linkage_list_directory,
       .coefficient_codec = original.hashes.coefficient_codec,
       .center_count = 256U,
       .mode = mode,
       .kmeans_iterations = lloyd_iterations});
  const auto norm_directory = PublishedNorm2Directory(derived.hashes);
  std::error_code error;
  std::filesystem::create_directories(publication.path() / norm_directory,
                                      error);
  if (error)
    return Status::Internal("cannot create norm2 derivation directory: " +
                            error.message());
  std::vector<std::uint8_t> center_bytes(1024U);
  for (unsigned l = 0; l < 256; ++l)
    StoreLittleEndianF32(codebook.centers()[l], center_bytes.data() + l * 4U);
  {
    std::ofstream file(publication.path() / norm_directory / "norm2_lut.f32",
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(center_bytes.data()),
               center_bytes.size());
    file.close();
    if (!file) return Status::Internal("cannot write norm2 LUT");
  }
  std::ofstream output(publication.path() / norm_directory / "norm2_labels.u8",
                       std::ios::binary);
  std::uint64_t cursor = 0U;
  std::size_t row_offset = 0U;
  ScaledScore8Accumulator scale_accumulator;
  for (unsigned cid = 0; cid < original.shape.nlist; ++cid) {
    const auto aligned = AlignCanonicalRecord(cursor);
    for (; cursor < aligned; ++cursor) output.put('\0');
    std::vector<float> cpu_norms;
    ArrayView<const float> cluster_norms;
    const auto count = files.entry(cid).real_count;
    if (use_cuda) {
      cluster_norms = {count ? norms.data() + row_offset : nullptr, count};
    } else {
      auto cluster = reconstruct(cid);
      if (!cluster.ok()) return cluster.status();
      cpu_norms = std::move(cluster).value();
      cluster_norms = cpu_norms;
    }
    row_offset += count;
    std::vector<std::uint8_t> labels(count);
    if (const Status status =
#if STLQPP_HAS_CUDA_NORM
            gpu_labels ? gpu_labels->Encode(cluster_norms, labels) :
#endif
                       NormOnlyDerivationAccess::Encode(codebook, cluster_norms, labels);
        !status.ok()) return status;
    if (publish_scaled) {
      std::vector<float> original_values(count);
      if (original_norms) {
        const auto records = files.topology_records(cid);
        std::vector<std::uint32_t> ids(count);
        for (unsigned i = 0; i < count; ++i)
          ids[i] = LoadLittleEndianU32(records.codes.data() + i * 4U);
        if (auto status = original_norms->Read(ids, &original_values); !status.ok()) return status;
      } else if (auto status = files.ReadFloatSourceNorm2(cid, original_values);
                 !status.ok()) return status;
      if (auto status = scale_accumulator.Accumulate(labels, cluster_norms, original_values);
          !status.ok()) return status;
    }
    output.write(reinterpret_cast<const char*>(labels.data()), labels.size());
    cursor += labels.size();
  }
  output.close();
  if (!output) return Status::Internal("cannot write norm2 labels");
  std::erase_if(derived.shards, [](const auto& shard) {
    return shard.kind == ShardKind::kNorm8 ||
           shard.kind == ShardKind::kNorm2Lut ||
           shard.kind == ShardKind::kScaledScore8;
  });
  for (auto& shard : derived.shards) {
    auto ref = ReferenceCanonicalShard(source, destination, shard);
    if (!ref.ok()) return ref.status();
    shard = std::move(ref).value();
  }
  derived.shards.push_back(
      {ShardKind::kNorm8, 0U,
       (norm_directory / "norm2_labels.u8").generic_string(), cursor,
       derived.counts.real});
  derived.shards.push_back({ShardKind::kNorm2Lut, 0U,
                            (norm_directory / "norm2_lut.f32").generic_string(),
                            1024U, 256U});
  if (publish_scaled) {
    auto fitted = scale_accumulator.Finish(codebook.view());
    if (!fitted.ok()) return fitted.status();
    const auto table = fitted.value().view();
    derived.norm_codec = NormCodec::kReconstructionNorm8WithScaledScore8;
    derived.scaled_score8_hash = ArtifactHashFactory::ScaledScore8({
        .coefficient_codec = derived.hashes.coefficient_codec,
        .source_norm8_cache = derived.hashes.norm8_cache,
        .scales = {table.scales, 256U}, .biases = {table.biases, 256U}});
    auto encoded = EncodeScaleTable(derived.shape, table);
    if (!encoded.ok()) return encoded.status();
    auto bytes = std::move(encoded).value();
    const auto name = (norm_directory / "norm2_scale_bias.f32").generic_string();
    std::ofstream out(publication.path() / name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    out.close();
    if (!out) return Status::Internal("cannot write derived norm2 scale table");
    derived.shards.push_back({ShardKind::kScaledScore8, 0U, name, bytes.size(), 512U});
  }
  if (auto status = WritePublishedNorm2Metadata(publication.path(), derived,
                                                use_cuda ? "CUDA" : "CPU");
      !status.ok())
    return status;
  if (auto status = SharePublishedNorm2Cache(
          publication.path(), CanonicalSourceIndexDirectory(source, original),
          destination, derived);
      !status.ok())
    return status;
  std::vector<std::uint8_t> bytes;
  if (auto status = IndexManifestCodec::Encode(derived, bytes); !status.ok())
    return status;
  {
    std::ofstream file(publication.path() / "manifest.stlqpp",
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    if (!file) return Status::Internal("cannot write norm2 manifest");
  }
  return publication.Publish();
}

Status PublishReconstructionNorm8Runtime(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, bool use_cuda, int cuda_device,
    bool publish_scaled, const SourceNormProvider* original_norms) {
  return PublishReconstructionNorm8Impl(
      source, destination, mode, lloyd_iterations, use_cuda, cuda_device, 0U,
      publish_scaled, original_norms);
}

template <int M>
Status PublishReconstructionNorm8(const std::filesystem::path& source,
                                  const std::filesystem::path& destination,
                                  ReconstructionNorm8Mode mode,
                                  std::uint32_t lloyd_iterations, bool use_cuda,
                                  int cuda_device) {
  return PublishReconstructionNorm8Impl(
      source, destination, mode, lloyd_iterations, use_cuda, cuda_device, M);
}

template Status PublishReconstructionNorm8<5>(const std::filesystem::path&,
                                              const std::filesystem::path&,
                                              ReconstructionNorm8Mode,
                                              std::uint32_t, bool, int);
template Status PublishReconstructionNorm8<12>(const std::filesystem::path&,
                                               const std::filesystem::path&,
                                               ReconstructionNorm8Mode,
                                               std::uint32_t, bool, int);
template Status PublishReconstructionNorm8<6>(const std::filesystem::path&,
                                              const std::filesystem::path&,
                                              ReconstructionNorm8Mode,
                                              std::uint32_t, bool, int);
template Status PublishReconstructionNorm8<10>(const std::filesystem::path&,
                                               const std::filesystem::path&,
                                               ReconstructionNorm8Mode,
                                               std::uint32_t, bool, int);

}  // namespace stlqpp
