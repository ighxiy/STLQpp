#include "stlqpp/application/index_republisher.h"

#include "stlqpp/application/registered_artifact_recipes.h"
#include "stlqpp/application/source_norm_provider.h"
#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/index/build/prepared_cluster_spool.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/norm/reconstruction_norm8.h"
#include "stlqpp/storage/norm_cache_layout.h"
#if STLQPP_HAS_CUDA_NORM
#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <chrono>

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_directory_codec.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/canonical_shard_codec.h"
#include "stlqpp/storage/index_manifest_codec.h"

namespace stlqpp {
namespace {

Status ValidatePaths(const CanonicalIndexRepublishRequest& request) {
  const auto valid_candidates = [](const std::vector<double>& values) {
    return !values.empty() &&
           std::all_of(values.begin(), values.end(), [](double value) {
             return std::isfinite(value) && value > 0.0 && value <= 1.0;
           });
  };
  const bool needs_spool =
      !request.publish_quantized_forest || !request.retain_float_forest;
  if (request.source.empty() || request.destination.empty() ||
      (needs_spool && request.prepared_spool.empty())) {
    return Status::InvalidArgument("codec republish paths must not be empty");
  }
  if (const Status status =
          ValidateIndexDerivationPaths(request.source, request.destination);
      !status.ok()) {
    return status;
  }
  if (needs_spool) {
    if (const Status status = ValidateDisjointFilesystemPaths(
            request.source, "codec republish source", request.prepared_spool,
            "codec republish prepared spool");
        !status.ok())
      return status;
    if (const Status status = ValidateDisjointFilesystemPaths(
            request.prepared_spool, "codec republish prepared spool",
            request.destination, "codec republish destination");
        !status.ok())
      return status;
  }
  if (request.maximum_window_clusters == 0U ||
      request.maximum_window_working_bytes == 0U ||
      request.target_window_real_rows == 0U ||
      (!request.publish_quantized_forest &&
       (!request.retain_float_forest || request.publish_scaled_score8)) ||
      (request.publish_quantized_forest &&
       (request.norm8_lloyd_iterations == 0U ||
        !valid_candidates(request.coefficient_first_magnitude_quantiles) ||
        !valid_candidates(request.coefficient_rest_magnitude_quantiles) ||
        (request.norm8_mode != ReconstructionNorm8Mode::kIdentityLloyd &&
         request.norm8_mode != ReconstructionNorm8Mode::kTailWeightedLloyd)))) {
    return Status::InvalidArgument(
        "codec republish window or norm8 recipe is invalid");
  }
  std::error_code error;
  if (std::filesystem::exists(request.destination, error) || error) {
    return Status::FailedPrecondition(
        "codec republish destination already exists or cannot be inspected");
  }
  if (needs_spool) {
    error.clear();
    if (std::filesystem::exists(request.prepared_spool, error) || error) {
      return Status::FailedPrecondition(
          "codec republish prepared spool already exists or cannot be "
          "inspected");
    }
  }
  return Status::Ok();
}

template <int M>
struct RepublishWindowContext final {
  const CanonicalIndexFiles* files = nullptr;
  const SelectedAtomGramTables<M>* grams = nullptr;
  const FinalClusterPostprocessor<M>* postprocessor = nullptr;
  const SourceNormProvider* external_source_norm2 = nullptr;
  bool load_source_norm2 = false;
  std::uint32_t first_cid = 0U;
  std::vector<PreparedCanonicalCluster<M>>* prepared = nullptr;
  std::vector<Status>* statuses = nullptr;
};

std::uint32_t RepublishWindowCount(const CanonicalIndexRepublishRequest& request,
                                  const CanonicalIndexFiles& files,
                                  const ParallelForExecutor& parallel,
                                  std::uint32_t first, std::uint64_t* estimated = nullptr) {
  std::uint32_t count = 0U;
  std::uint64_t rows = 0U, bytes = 0U;
  const auto minimum = std::min(request.maximum_window_clusters,
                                std::max(1U, parallel.worker_count()));
  while (first + count < files.manifest().shape.nlist && count < request.maximum_window_clusters) {
    const auto& e = files.entry(first + count);
    const std::uint64_t next = (std::uint64_t(e.real_count) + e.virtual_count) *
        (32U * files.manifest().shape.m + 64U) + 4096U;
    if (count != 0U && (next > request.maximum_window_working_bytes -
        std::min(bytes, request.maximum_window_working_bytes) ||
        (count >= minimum && rows >= request.target_window_real_rows))) break;
    bytes += next; rows += e.real_count; ++count;
  }
  if (estimated) *estimated = bytes;
  return count;
}

template <int M>
void PrepareRepublishedCluster(void* opaque, std::uint32_t,
                               std::uint32_t local) noexcept {
  auto& context = *static_cast<RepublishWindowContext<M>*>(opaque);
  const std::uint32_t cid = context.first_cid + local;
  const CanonicalCidEntry& entry = context.files->entry(cid);
  const CanonicalClusterRecordViews records =
      context.files->cluster_records(cid);
  const auto m = context.files->manifest().shape.m;
  auto topology = CanonicalClusterCodec<M>::DecodeTopology(
      cid, entry.real_count, entry.virtual_count, entry.real_root_count,
      records.codes, records.parents, m);
  if (!topology.ok()) {
    (*context.statuses)[local] = topology.status();
    return;
  }
  const std::uint64_t node_count =
      static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
  if (node_count > std::numeric_limits<std::size_t>::max() / m) {
    (*context.statuses)[local] = Status::ResourceExhausted(
        "native-float CID exceeds the host coefficient domain");
    return;
  }
  std::vector<float> coefficients(static_cast<std::size_t>(node_count) * m);
  if (const Status status =
          context.files->ReadFloatCoefficients(cid, coefficients);
      !status.ok()) {
    (*context.statuses)[local] = status;
    return;
  }
  // Codec replacement consumes float coefficients, not float reconstruction
  // norms. Existing float shards remain references; eval owns missing caches.
  std::vector<float> source_norm2;
  if (context.load_source_norm2) {
    source_norm2.resize(entry.real_count);
    if (const Status status =
            context.files->ReadFloatSourceNorm2(cid, source_norm2);
        !status.ok()) {
      (*context.statuses)[local] = status;
      return;
    }
  } else if (context.external_source_norm2 != nullptr) {
    if (const Status status = context.external_source_norm2->Read(
            topology.value().real_ids(), &source_norm2);
        !status.ok()) {
      (*context.statuses)[local] = status;
      return;
    }
  }
  const auto& decoded = topology.value();
  const CanonicalFloatClusterView source = {
      .m = m,
      .cid = cid,
      .virtual_count = entry.virtual_count,
      .real_root_count = entry.real_root_count,
      .real_ids = decoded.real_ids(),
      .parent_local_1based = decoded.parents(),
      .real_codes_small = decoded.real_codes_small(),
      .edge_code0 = decoded.edge_code0(),
      .virtual_codes_small = decoded.virtual_codes_small(),
      .coefficients_layer_major = coefficients,
      .source_real_norm2 = source_norm2,
  };
  auto prepared = [&]() -> StatusOr<PreparedCanonicalCluster<M>> {
    if (context.postprocessor == nullptr) {
      return FinalClusterPostprocessor<M>::PrepareRetainedFloatOnly(
          source, {}, context.files->manifest().shape.nlist);
    }
    auto root_cross = context.grams->BuildRootCrossBlock(cid, 1U);
    if (!root_cross.ok()) return root_cross.status();
    return context.postprocessor->PrepareFromRetainedFloatForest(
        source, {}, root_cross.value());
  }();
  if (!prepared.ok()) {
    (*context.statuses)[local] = prepared.status();
    return;
  }
  (*context.prepared)[local] = std::move(prepared).value();
}

template <int M>
struct RepublishLanes {
  RepublishWindowContext<M> context;
  std::uint32_t count;
  std::atomic<std::uint32_t> next{0U};
  std::atomic<bool> failed{false};
};

template <int M>
void PrepareRepublishedLane(void* opaque, std::uint32_t, std::uint32_t) noexcept {
  auto& lanes = *static_cast<RepublishLanes<M>*>(opaque);
  while (!lanes.failed.load(std::memory_order_relaxed)) {
    const auto local = lanes.next.fetch_add(1U, std::memory_order_relaxed);
    if (local >= lanes.count || lanes.failed.load(std::memory_order_relaxed)) return;
    PrepareRepublishedCluster<M>(&lanes.context, 0U, local);
    if (!(*lanes.context.statuses)[local].ok()) {
      lanes.failed.store(true, std::memory_order_relaxed);
      return;
    }
  }
}

template <int M>
Status PopulateRepublishSpoolWithModel(
    const CanonicalIndexRepublishRequest& request,
    const CanonicalIndexFiles& files, const ProductCodebooks<M>& codebooks,
    const F32GemmBackend& gemm, const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel, const IndexArtifactHashes& hashes,
    const SourceNormProvider* external_source_norm2) {
  auto spool_result = PreparedClusterSpoolWriter<M>::Create({
      .destination = request.prepared_spool,
      .shape = files.manifest().shape,
      .hashes = hashes,
      .retain_float_forest = request.retain_float_forest,
      .retain_source_norm2 =
          request.publish_scaled_score8 || request.publish_float_scaled,
      .publish_quantized_forest = request.publish_quantized_forest,
      .retain_float_norm2 = false,
  });
  if (!spool_result.ok()) return spool_result.status();
  auto spool = std::move(spool_result).value();
  std::optional<SelectedAtomGramTables<M>> grams;
  std::optional<FinalClusterPostprocessor<M>> postprocessor;
  if (request.publish_quantized_forest) {
    auto result = SelectedAtomGramTables<M>::Create(
        codebooks, gemm, SelectedAtomGramDemand::kReconstructionNorm);
    if (!result.ok()) return result.status();
    grams.emplace(std::move(result).value());
    postprocessor.emplace(codebooks, *grams, coefficient_solver,
                          request.coefficient_bits_per_layer,
                          request.coefficient_bits_per_layer_root.empty()
                              ? request.coefficient_bits_per_layer
                              : request.coefficient_bits_per_layer_root,
                          request.coefficient_bits_per_layer_linkage.empty()
                              ? request.coefficient_bits_per_layer
                              : request.coefficient_bits_per_layer_linkage,
                          request.coefficient_first_magnitude_quantiles,
                          request.coefficient_rest_magnitude_quantiles,
                          request.retain_float_forest,
                          request.coefficient_codec_options,
                          request.publish_quantized_forest,
                          request.use_cuda_norm8_fit, request.cuda_norm_device,
                          false);
  }

  std::uint32_t first = 0U;
  while (first < files.manifest().shape.nlist) {
    const auto count = RepublishWindowCount(request, files, parallel, first);
    std::vector<PreparedCanonicalCluster<M>> prepared(count);
    std::vector<Status> statuses(count);
    RepublishWindowContext<M> context = {
        .files = &files,
        .grams = grams ? &*grams : nullptr,
        .postprocessor = postprocessor ? &*postprocessor : nullptr,
        .external_source_norm2 = external_source_norm2,
        .load_source_norm2 =
            (request.publish_scaled_score8 || request.publish_float_scaled) &&
            external_source_norm2 == nullptr,
        .first_cid = first,
        .prepared = &prepared,
        .statuses = &statuses,
    };
    RepublishLanes<M> lanes{context, count};
    parallel.Run(std::min(count, parallel.worker_count()), &lanes, &PrepareRepublishedLane<M>);
    for (const Status& status : statuses) {
      if (!status.ok()) return status;
    }
    for (const PreparedCanonicalCluster<M>& cluster : prepared) {
      if (const Status status = spool.Append(cluster); !status.ok()) {
        return status;
      }
    }
    first += count;
  }
  return spool.Commit();
}

Status WriteRepublishBytes(std::ofstream& output,
                           ArrayView<const std::uint8_t> bytes,
                           const char* label) {
  if (bytes.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    return Status::ResourceExhausted(std::string("derived ") + label +
                                     " exceeds streamsize");
  }
  if (!bytes.empty()) {
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  return output ? Status::Ok()
                : Status::Internal(std::string("failed to write derived ") +
                                   label);
}

Status WriteRepublishFile(const std::filesystem::path& path,
                          ArrayView<const std::uint8_t> bytes,
                          const char* label) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return Status::Internal(std::string("failed to create derived ") + label);
  if (const Status status = WriteRepublishBytes(output, bytes, label);
      !status.ok())
    return status;
  output.close();
  return output ? Status::Ok()
                : Status::Internal(std::string("failed to close derived ") +
                                   label);
}

Status AppendRepublishRecord(std::ofstream& output, std::uint64_t& cursor,
                             ArrayView<const std::uint8_t> record,
                             std::uint64_t& offset, const char* label) {
  offset = AlignCanonicalRecord(cursor);
  const std::uint8_t padding[kCanonicalRecordAlignment] = {};
  if (const Status status = WriteRepublishBytes(
          output, {padding, static_cast<std::size_t>(offset - cursor)}, label);
      !status.ok())
    return status;
  if (const Status status = WriteRepublishBytes(output, record, label);
      !status.ok())
    return status;
  if (record.size() > std::numeric_limits<std::uint64_t>::max() - offset) {
    return Status::ResourceExhausted("derived shard offset overflows u64");
  }
  cursor = offset + record.size();
  return Status::Ok();
}

StatusOr<ReconstructionNorm8Codebook> FitRepublishNorm8(
    const ReadOnlyMappedFile& stream, std::uint64_t rows,
    ReconstructionNorm8Mode mode, std::uint32_t iterations,
    const ParallelForExecutor& parallel, bool use_cuda_norm8_fit,
    int cuda_norm_device) {
  if (use_cuda_norm8_fit && mode != ReconstructionNorm8Mode::kIdentityLloyd) {
    return Status::Unimplemented(
        "GPU norm8 tail-weighted mode is not supported");
  }
#if !STLQPP_HAS_CUDA_NORM
  (void)cuda_norm_device;
  if (use_cuda_norm8_fit) {
    return Status::Unimplemented("GPU norm8 fit is unavailable in this build");
  }
#endif
  if (rows == 0U ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      stream.size() != rows * sizeof(float)) {
    return Status::DataLoss("derived reconstruction norm stream is incomplete");
  }
  std::vector<float> values(static_cast<std::size_t>(rows));
  for (std::size_t row = 0U; row < values.size(); ++row) {
    values[row] = LoadLittleEndianF32(stream.data() + row * sizeof(float));
  }
#if STLQPP_HAS_CUDA_NORM
  if (use_cuda_norm8_fit) {
    auto accelerated =
        FitIdentityNorm8Cuda(values, iterations, cuda_norm_device);
    if (!accelerated.ok()) return accelerated.status();
    return accelerated;
  }
#endif
  return ReconstructionNorm8Codebook::FitFromValues(std::move(values), iterations, parallel, mode);
}

class RepublishWriteTimer final {
 public:
  explicit RepublishWriteTimer(double& seconds) : seconds_(seconds) {}
  ~RepublishWriteTimer() {
    seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_).count();
  }
 private:
  double& seconds_;
  const std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
};

// The numerical worker hands this sink already encoded coefficients and norm
// spans. Transactions, offsets, byte serialization and accounting have no M.
struct RepublishDeltaSink {
  const CanonicalIndexFiles& files;
  std::ofstream& coefficient_stream;
  std::uint64_t& coefficient_cursor;
  std::vector<CanonicalCidEntry>& entries;
  std::vector<std::uint64_t>& norm_begins;
  std::uint64_t& norm_rows;
  std::ofstream& norm_work_stream;
  bool collect_source_norm;
  std::vector<float>& source_norm_population;
  CanonicalIndexWriteStatistics& accounting;
  double& write_seconds;

  Status Append(std::uint32_t cid, const std::vector<std::uint8_t>& record,
                ArrayView<const float> norms,
                ArrayView<const float> source_norms) {
    const RepublishWriteTimer write_timer(write_seconds);
    CanonicalCidEntry entry = files.entry(cid);
    entry.coefficients_bytes = record.size();
    if (const Status status = AppendRepublishRecord(
            coefficient_stream, coefficient_cursor, record,
            entry.coefficients_offset, "coefficient record");
        !status.ok())
      return status;
    entries.push_back(entry);
    norm_begins.push_back(norm_rows);
    if (norms.size() != entry.real_count) {
      return Status::DataLoss(
          "derived reconstruction norm count disagrees with CID");
    }
    const std::uint32_t endian = 1U;
    if (*reinterpret_cast<const std::uint8_t*>(&endian) == 1U) {
      norm_work_stream.write(reinterpret_cast<const char*>(norms.data()),
                             static_cast<std::streamsize>(norms.size() * sizeof(float)));
      if (!norm_work_stream) return Status::Internal("failed to write norm work stream");
    } else {
      std::vector<std::uint8_t> bytes(norms.size() * sizeof(float));
      for (std::size_t row = 0U; row < norms.size(); ++row)
        StoreLittleEndianF32(norms[row], bytes.data() + row * sizeof(float));
      if (const auto status = WriteRepublishBytes(norm_work_stream, bytes, "norm work stream"); !status.ok())
        return status;
    }
    norm_rows += entry.real_count;
    if (collect_source_norm) {
      if (source_norms.size() != entry.real_count)
        return Status::DataLoss("derived source norm count disagrees with CID");
      source_norm_population.insert(source_norm_population.end(),
                                    source_norms.begin(), source_norms.end());
    }
    const std::uint64_t root_bytes = LoadLittleEndianU32(record.data());
    const std::uint64_t linkage_bytes = LoadLittleEndianU32(record.data() + 4U);
    accounting.coefficient_record_bytes += record.size();
    accounting.coefficient_header_bytes +=
        record.size() - root_bytes - linkage_bytes;
    accounting.root_coefficient_payload_bytes += root_bytes;
    accounting.linkage_coefficient_payload_bytes += linkage_bytes;
    accounting.code_record_bytes += entry.codes_bytes;
    accounting.parent_record_bytes += entry.parents_bytes;
    const std::uint64_t nodes =
        static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
    const std::uint64_t parent_bits = 2U * nodes + 1U;
    accounting.parent_raw_bits += parent_bits;
    accounting.parent_word_padding_bits +=
        entry.parents_bytes * 8U - parent_bits;
    return Status::Ok();
  }
};

using PrepareDeltaWindow = Status (*)(void*, std::uint32_t, std::uint32_t,
                                      RepublishDeltaSink&);

Status PublishRepublishDeltaImpl(const CanonicalIndexRepublishRequest& request,
                                 const CanonicalIndexFiles& files,
                                 const ParallelForExecutor& parallel,
                                 const IndexArtifactHashes& hashes,
                                 void* prepare_context,
                                 PrepareDeltaWindow prepare_window,
                                 FinalIndexPublicationStatistics* statistics) {
  const IndexManifest& source_manifest = files.manifest();
  auto stage_result = StagingDirectory::Create(request.destination);
  if (!stage_result.ok()) return stage_result.status();
  auto stage = std::move(stage_result).value();

  // Unchanged payloads remain at their original paths; only reference metadata
  // is published, with no copy or filesystem link.
  IndexManifest derived = source_manifest;
  derived.shards.clear();
  for (const ShardDescriptor& shard : source_manifest.shards) {
    if ((!request.retain_float_forest &&
         (shard.kind == ShardKind::kFloatCoefficients ||
          shard.kind == ShardKind::kFloatNorm2 ||
          shard.kind == ShardKind::kFloatSourceNorm2)) ||
        shard.kind == ShardKind::kIvfDirectory ||
        shard.kind == ShardKind::kCoefficients ||
        shard.kind == ShardKind::kNorm8 || shard.kind == ShardKind::kNorm2Lut ||
        shard.kind == ShardKind::kScaledScore8) {
      continue;
    }
    auto reference =
        ReferenceCanonicalShard(request.source, request.destination, shard);
    if (!reference.ok()) return reference.status();
    derived.shards.push_back(std::move(reference).value());
  }
  std::ofstream coefficient_stream(stage.path() / "coefficients.bin",
                                   std::ios::binary);
  if (!coefficient_stream) {
    return Status::Internal("failed to create derived coefficient shard");
  }
  std::vector<std::uint8_t> header;
  if (const Status status = CanonicalPayloadShardCodec::EncodeHeader(
          ShardKind::kCoefficients, source_manifest.shape, header);
      !status.ok())
    return status;
  if (const Status status =
          WriteRepublishBytes(coefficient_stream, header, "coefficient header");
      !status.ok())
    return status;
  std::uint64_t coefficient_cursor = header.size();
  std::vector<float> source_norm_population;
  if (request.publish_scaled_score8)
    source_norm_population.reserve(source_manifest.counts.real);
  if (source_manifest.counts.real >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return Status::ResourceExhausted(
        "derived norm population exceeds host memory domain");
  }
  const auto norm_work_path = stage.path() / "reconstruction_norms.tmp";
  std::ofstream norm_work_stream(norm_work_path, std::ios::binary);
  if (!norm_work_stream) {
    return Status::Internal("failed to create derived norm work stream");
  }
  std::uint64_t norm_rows = 0U;
  std::vector<std::uint64_t> norm_begins;
  norm_begins.reserve(source_manifest.shape.nlist);
  std::vector<CanonicalCidEntry> entries;
  entries.reserve(source_manifest.shape.nlist);
  CanonicalIndexWriteStatistics accounting;
  accounting.shape = source_manifest.shape;
  accounting.counts = source_manifest.counts;

  double write_seconds = 0.0;
  RepublishDeltaSink sink{files,
                          coefficient_stream,
                          coefficient_cursor,
                          entries,
                          norm_begins,
                          norm_rows,
                          norm_work_stream,
                          request.publish_scaled_score8,
                          source_norm_population,
                          accounting, write_seconds};
  const auto preparation_begin = std::chrono::steady_clock::now();
  for (std::uint32_t first = 0U; first < source_manifest.shape.nlist;) {
    std::uint64_t working_bytes = 0U;
    const auto count = RepublishWindowCount(request, files, parallel, first, &working_bytes);
    if (statistics) statistics->coefficient_peak_window_bytes =
        std::max(statistics->coefficient_peak_window_bytes, working_bytes);
    if (const auto status = prepare_window(prepare_context, first, count, sink);
        !status.ok())
      return status;
    first += count;
  }
  if (statistics) statistics->coefficient_prepare_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - preparation_begin).count();
  coefficient_stream.close();
  if (!coefficient_stream) {
    return Status::Internal("failed to close derived coefficient shard");
  }
  norm_work_stream.close();
  if (!norm_work_stream) {
    return Status::Internal("failed to close derived norm work stream");
  }
  if (norm_rows != source_manifest.counts.real) {
    return Status::DataLoss("derived norm population is incomplete");
  }
  auto mapped_norms_result = ReadOnlyMappedFile::Open(norm_work_path);
  if (!mapped_norms_result.ok()) return mapped_norms_result.status();
  auto mapped_norms = std::move(mapped_norms_result).value();
  const auto fit_begin = std::chrono::steady_clock::now();
  auto norm_result =
      FitRepublishNorm8(mapped_norms, norm_rows, request.norm8_mode,
                        request.norm8_lloyd_iterations, parallel,
                        request.use_cuda_norm8_fit, request.cuda_norm_device);
  if (!norm_result.ok()) return norm_result.status();
  if (statistics) statistics->norm8_fit_wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - fit_begin).count();
  const auto& norm8 = norm_result.value();
#if STLQPP_HAS_CUDA_NORM
  std::unique_ptr<CudaNorm8LabelEncoder> gpu_labels;
  if (request.use_cuda_norm8_fit) {
    auto created =
        CudaNorm8LabelEncoder::Create(norm8.view(), request.cuda_norm_device);
    if (!created.ok()) return created.status();
    gpu_labels = std::move(created).value();
  }
#endif
  const auto norm_directory = PublishedNorm2Directory(hashes);
  std::error_code layout_error;
  std::filesystem::create_directories(stage.path() / norm_directory,
                                      layout_error);
  if (layout_error)
    return Status::Internal("cannot create derived norm2 directory: " +
                            layout_error.message());
  header.resize(1024U);
  for (unsigned l = 0; l < 256; ++l)
    StoreLittleEndianF32(norm8.centers()[l], header.data() + l * 4U);
  if (auto status = WriteRepublishFile(
          stage.path() / norm_directory / "norm2_lut.f32", header, "norm2 LUT");
      !status.ok())
    return status;
  std::ofstream norm_stream(stage.path() / norm_directory / "norm2_labels.u8",
                            std::ios::binary);
  if (!norm_stream)
    return Status::Internal("cannot create derived norm2 labels");
  std::uint64_t norm_cursor = 0U;
  ScaledScore8Accumulator scaled_accumulator;
  const double writes_before_labels = write_seconds;
  const auto labels_begin = std::chrono::steady_clock::now();
  for (std::uint32_t first = 0U; first < source_manifest.shape.nlist;) {
    const std::size_t target = request.publish_scaled_score8 ? 262144U : 65536U;
    std::uint32_t end = first;
    std::size_t rows = 0;
    do { rows += entries[end++].real_count; }
    while (end < source_manifest.shape.nlist && rows < target);
    const auto begin = norm_begins[first];
    std::vector<float> values(rows);
    for (std::size_t row = 0; row < rows; ++row)
      values[row] = LoadLittleEndianF32(mapped_norms.data() + (begin + row) * sizeof(float));
    std::vector<std::uint8_t> labels(rows);
    if (const Status status =
#if STLQPP_HAS_CUDA_NORM
            gpu_labels ? gpu_labels->Encode(values, labels) :
#endif
                         norm8.Encode(values, labels); !status.ok()) return status;
    if (request.publish_scaled_score8) {
      const ArrayView<const float> source_norms{source_norm_population.data() + begin, rows};
      if (const auto status = scaled_accumulator.Accumulate(labels, values, source_norms, parallel);
          !status.ok()) return status;
    }
    std::size_t offset = 0;
    for (auto cid = first; cid < end; ++cid) {
      auto& entry = entries[cid];
      entry.norm8_bytes = entry.real_count;
      const ArrayView<const std::uint8_t> cid_labels{labels.data() + offset, entry.real_count};
      const RepublishWriteTimer write_timer(write_seconds);
      if (const auto status = AppendRepublishRecord(norm_stream, norm_cursor, cid_labels,
              entry.norm8_offset, "norm8 record"); !status.ok()) return status;
      accounting.norm8_label_bytes += entry.real_count;
      offset += entry.real_count;
    }
    first = end;
  }
  if (statistics) statistics->norm8_labels_scale_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - labels_begin).count() -
      (write_seconds - writes_before_labels);
  norm_stream.close();
  if (!norm_stream)
    return Status::Internal("failed to close derived norm8 shard");
  mapped_norms = ReadOnlyMappedFile{};
  std::error_code work_error;
  std::filesystem::remove(norm_work_path, work_error);
  if (work_error) {
    return Status::Internal("failed to remove derived norm work stream: " +
                            work_error.message());
  }
  const auto commit_begin = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> directory;
  if (const Status status = CanonicalDirectoryCodec::Encode(
          source_manifest.shape, source_manifest.counts, entries, directory,
          true, true,
          source_manifest.format_version ==
              IndexManifest::kCompletionOrderFormatVersion);
      !status.ok())
    return status;
  if (const Status status = WriteRepublishFile(stage.path() / "directory.bin",
                                               directory, "directory");
      !status.ok())
    return status;
  derived.hashes = hashes;
  derived.producer_recipe = IndexProducerRecipe{
      .coefficient_bits_per_layer = request.coefficient_bits_per_layer,
      .coefficient_bits_per_layer_root =
          request.coefficient_bits_per_layer_root,
      .coefficient_bits_per_layer_linkage =
          request.coefficient_bits_per_layer_linkage,
      .coefficient_first_magnitude_quantiles =
          request.coefficient_first_magnitude_quantiles,
      .coefficient_rest_magnitude_quantiles =
          request.coefficient_rest_magnitude_quantiles,
      .coefficient_weighted_quantile =
          request.coefficient_codec_options.weighted_quantile,
      .coefficient_allow_clipping =
          request.coefficient_codec_options.allow_clipping,
      .coefficient_fit_scale =
          request.coefficient_codec_options.fit_reconstruction_scales,
      .coefficient_refinement_sweeps =
          request.coefficient_codec_options.refinement_sweeps,
      .coefficient_refinement_max_layer =
          request.coefficient_codec_options.refinement_max_layer,
      .coefficient_refinement_step_limit =
          request.coefficient_codec_options.refinement_step_limit,
      .norm8_mode = request.norm8_mode,
      .norm8_lloyd_iterations = request.norm8_lloyd_iterations,
  };
  derived.format_version = files.manifest().format_version ==
                                   IndexManifest::kCompletionOrderFormatVersion
                               ? IndexManifest::kCompletionOrderFormatVersion
                               : IndexManifest::kNormCacheLayoutFormatVersion;
  derived.norm_codec = NormCodec::kReconstructionNorm8;
  derived.scaled_score8_hash = ArtifactHash64{};
  derived.shards.push_back({ShardKind::kIvfDirectory, 0U, "directory.bin",
                            directory.size(), source_manifest.shape.nlist});
  derived.shards.push_back(
      {ShardKind::kCoefficients, 0U, "coefficients.bin", coefficient_cursor,
       source_manifest.counts.real + source_manifest.counts.virtual_roots});
  derived.shards.push_back({ShardKind::kNorm2Lut, 0U,
                            (norm_directory / "norm2_lut.f32").generic_string(),
                            1024U, 256U});
  derived.shards.push_back(
      {ShardKind::kNorm8, 0U,
       (norm_directory / "norm2_labels.u8").generic_string(), norm_cursor,
       source_manifest.counts.real});
  if (request.publish_scaled_score8) {
    auto table = scaled_accumulator.Finish(norm8.view());
    if (!table.ok()) return table.status();
    const auto view = table.value().view();
    if (const Status status = CanonicalPayloadShardCodec::EncodeHeader(
            ShardKind::kScaledScore8, source_manifest.shape, header);
        !status.ok())
      return status;
    const std::size_t header_size = header.size();
    header.resize(header_size + 512U * sizeof(float));
    for (std::uint32_t label = 0U; label < 256U; ++label) {
      StoreLittleEndianF32(
          view.scales[label],
          header.data() + header_size +
              static_cast<std::size_t>(2U * label) * sizeof(float));
      StoreLittleEndianF32(
          view.biases[label],
          header.data() + header_size +
              static_cast<std::size_t>(2U * label + 1U) * sizeof(float));
    }
    if (const Status status = WriteRepublishFile(
            stage.path() / norm_directory / "norm2_scale_bias.f32", header,
            "scaled score8");
        !status.ok())
      return status;
    derived.norm_codec = NormCodec::kReconstructionNorm8WithScaledScore8;
    derived.scaled_score8_hash = ArtifactHashFactory::ScaledScore8({
        .coefficient_codec = hashes.coefficient_codec,
        .source_norm8_cache = hashes.norm8_cache,
        .scales = table.value().scales(),
        .biases = table.value().biases(),
    });
    derived.shards.push_back(
        {ShardKind::kScaledScore8, 0U,
         (norm_directory / "norm2_scale_bias.f32").generic_string(),
         header.size(), 512U});
    accounting.scaled_score8_shard_bytes = header.size();
  }
  std::sort(derived.shards.begin(), derived.shards.end(),
            [](const ShardDescriptor& left, const ShardDescriptor& right) {
              return left.kind < right.kind;
            });
  if (auto status = WritePublishedNorm2Metadata(
          stage.path(), derived, request.use_cuda_norm8_fit ? "CUDA" : "CPU");
      !status.ok())
    return status;
  if (auto status = SharePublishedNorm2Cache(
          stage.path(),
          CanonicalSourceIndexDirectory(request.source, source_manifest),
          request.destination, derived);
      !status.ok())
    return status;
  std::vector<std::uint8_t> manifest_bytes;
  if (const Status status = IndexManifestCodec::Encode(derived, manifest_bytes);
      !status.ok())
    return status;
  if (const Status status = WriteRepublishFile(stage.path() / "manifest.stlqpp",
                                               manifest_bytes, "manifest");
      !status.ok())
    return status;
  if (const Status status = stage.Publish(); !status.ok()) return status;
  accounting.directory_shard_bytes = directory.size();
  accounting.coefficients_shard_bytes = coefficient_cursor;
  accounting.norm8_shard_bytes = norm_cursor + 1024U;
  accounting.float_source_norm2_shard_bytes = 0U;
  accounting.manifest_bytes = manifest_bytes.size();
  for (const auto& shard : source_manifest.shards) {
    if (shard.kind == ShardKind::kCodebooks)
      accounting.model_shard_bytes = shard.bytes;
    if (shard.kind == ShardKind::kCodes)
      accounting.codes_shard_bytes = shard.bytes;
    if (shard.kind == ShardKind::kParents)
      accounting.parents_shard_bytes = shard.bytes;
    if (shard.kind == ShardKind::kFloatCoefficients) {
      accounting.float_coefficients_shard_bytes = shard.bytes;
    }
    if (shard.kind == ShardKind::kFloatNorm2) {
      accounting.float_norm2_shard_bytes = shard.bytes;
    }
    if (shard.kind == ShardKind::kFloatSourceNorm2) {
      accounting.float_source_norm2_shard_bytes = shard.bytes;
    }
  }
  if (statistics != nullptr) {
    statistics->delta_write_seconds = write_seconds;
    statistics->commit_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - commit_begin).count();
    statistics->canonical = accounting;
    statistics->norm8_mode = request.norm8_mode;
    statistics->norm8_lloyd_iterations = request.norm8_lloyd_iterations;
    statistics->norm8_center_count = 256U;
  }
  return Status::Ok();
}

template <int M>
struct RepublishDeltaPreparation {
  RepublishWindowContext<M> window;
  const ParallelForExecutor& parallel;
};

template <int M>
Status PrepareRepublishDeltaWindow(void* opaque, std::uint32_t first,
                                   std::uint32_t count,
                                   RepublishDeltaSink& sink) {
  const auto& preparation = *static_cast<RepublishDeltaPreparation<M>*>(opaque);
  std::vector<PreparedCanonicalCluster<M>> prepared(count);
  std::vector<Status> statuses(count);
  auto context = preparation.window;
  context.first_cid = first;
  context.prepared = &prepared;
  context.statuses = &statuses;
  RepublishLanes<M> lanes{context, count};
  preparation.parallel.Run(std::min(count, preparation.parallel.worker_count()), &lanes, &PrepareRepublishedLane<M>);
  for (const auto& status : statuses) {
    if (!status.ok()) return status;
  }
  for (std::uint32_t local = 0U; local < count; ++local) {
    const auto& cluster = prepared[local];
    std::vector<std::uint8_t> record;
    if (const auto status = CanonicalClusterCodec<M>::EncodeCoefficientRecord(
            cluster.payload_source().coefficients, record);
        !status.ok())
      return status;
    if (const auto status =
            sink.Append(first + local, record, cluster.exact_real_norm2(),
                        cluster.source_real_norm2());
        !status.ok())
      return status;
  }
  return Status::Ok();
}

template <int M>
Status PublishRepublishDelta(const CanonicalIndexRepublishRequest& request,
                             const CanonicalIndexFiles& files,
                             const ProductCodebooks<M>& codebooks,
                             const F32GemmBackend& gemm,
                             const SpdSolverF64& coefficient_solver,
                             const ParallelForExecutor& parallel,
                             const IndexArtifactHashes& hashes,
                             const SourceNormProvider* external_source_norm2,
                             FinalIndexPublicationStatistics* statistics) {
  auto gram_result = SelectedAtomGramTables<M>::Create(
        codebooks, gemm, SelectedAtomGramDemand::kReconstructionNorm);
  if (!gram_result.ok()) return gram_result.status();
  auto grams = std::move(gram_result).value();
  auto codec_options = request.coefficient_codec_options;
  codec_options.collect_sse_statistics = false;
  FinalClusterPostprocessor<M> postprocessor(
      codebooks, grams, coefficient_solver, request.coefficient_bits_per_layer,
      request.coefficient_bits_per_layer_root.empty()
          ? request.coefficient_bits_per_layer
          : request.coefficient_bits_per_layer_root,
      request.coefficient_bits_per_layer_linkage.empty()
          ? request.coefficient_bits_per_layer
          : request.coefficient_bits_per_layer_linkage,
      request.coefficient_first_magnitude_quantiles,
      request.coefficient_rest_magnitude_quantiles, false,
      codec_options, true, request.use_cuda_norm8_fit,
      request.cuda_norm_device, false);

  RepublishDeltaPreparation<M> preparation{
      {.files = &files,
       .grams = &grams,
       .postprocessor = &postprocessor,
       .external_source_norm2 = external_source_norm2,
       .load_source_norm2 =
           (request.publish_scaled_score8 || request.publish_float_scaled) &&
           files.manifest().has_float_source_norm2() &&
           external_source_norm2 == nullptr},
      parallel};
  const auto status = PublishRepublishDeltaImpl(request, files, parallel, hashes,
                                   &preparation, &PrepareRepublishDeltaWindow<M>, statistics);
  if (statistics) {
    statistics->reconstruction_norm_service = postprocessor.norm_timing();
    statistics->coefficient_service_seconds = postprocessor.coefficient_service_seconds();
    statistics->coefficient_peak_workers = postprocessor.coefficient_peak_workers();
  }
  return status;
}

}  // namespace

namespace {
template <int M>
StatusOr<CanonicalIndexRepublishResult> RepublishTypedModel(
    const CanonicalIndexRepublishRequest& request,
    const CanonicalIndexFiles& files, const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver, const ParallelForExecutor& parallel,
    const IndexArtifactHashes& hashes,
    const SourceNormProvider* external_source_norm2) {
  auto model_result = CanonicalModelCodec::Decode<M>(
      files.model_bytes().data(), files.model_bytes().size());
  if (!model_result.ok()) return model_result.status();
  auto model = std::move(model_result).value();
  CanonicalIndexRepublishResult output;
  output.source_hashes = files.manifest().hashes;
  output.published_hashes = hashes;
  if (request.publish_quantized_forest) {
    const Status publish = PublishRepublishDelta<M>(
        request, files, model.codebooks(), gemm, coefficient_solver, parallel,
        hashes, external_source_norm2, &output.publication);
    if (!publish.ok()) return publish;
    return output;
  }
  auto stage = StagingDirectory::Create(request.destination);
  if (!stage.ok()) return stage.status();
  IndexManifest derived = files.manifest();
  derived.format_version = files.manifest().format_version ==
                                   IndexManifest::kCompletionOrderFormatVersion
                               ? IndexManifest::kCompletionOrderFormatVersion
                               : IndexManifest::kNormCacheLayoutFormatVersion;
  derived.hashes = hashes;
  derived.producer_recipe.reset();
  derived.scaled_score8_hash = {};
  derived.norm_codec = NormCodec::kReconstructionNorm8;
  std::erase_if(derived.shards, [&](const auto& shard) {
    return shard.kind == ShardKind::kCoefficients ||
           shard.kind == ShardKind::kNorm8 ||
           shard.kind == ShardKind::kNorm2Lut ||
           shard.kind == ShardKind::kScaledScore8 ||
           (shard.kind == ShardKind::kFloatSourceNorm2 &&
            !request.publish_float_scaled);
  });
  const bool replace_directory = files.manifest().has_quantized_forest();
  std::vector<std::uint8_t> directory;
  if (replace_directory) {
    std::vector<CanonicalCidEntry> entries;
    entries.reserve(derived.shape.nlist);
    for (std::uint32_t cid = 0; cid < derived.shape.nlist; ++cid) {
      auto entry = files.entry(cid);
      entry.coefficients_offset = entry.coefficients_bytes = 0U;
      entry.norm8_offset = entry.norm8_bytes = 0U;
      entries.push_back(entry);
    }
    if (const auto status = CanonicalDirectoryCodec::Encode(
            derived.shape, derived.counts, entries, directory, false, true,
            derived.format_version ==
                IndexManifest::kCompletionOrderFormatVersion);
        !status.ok())
      return status;
    if (const auto status = WriteRepublishFile(
            stage.value().path() / "directory.bin", directory, "directory");
        !status.ok())
      return status;
  }
  for (auto& shard : derived.shards) {
    if (shard.kind == ShardKind::kIvfDirectory && replace_directory) {
      shard.relative_path = "directory.bin";
      shard.bytes = directory.size();
      shard.reference_path.clear();
      shard.reference_sha256.clear();
      shard.payload_offset = 0U;
      continue;
    }
    auto reference =
        ReferenceCanonicalShard(request.source, request.destination, shard);
    if (!reference.ok()) return reference.status();
    shard = std::move(reference).value();
  }
  std::vector<std::uint8_t> bytes;
  if (const auto status = IndexManifestCodec::Encode(derived, bytes);
      !status.ok())
    return status;
  if (const auto status = WriteRepublishFile(
          stage.value().path() / "manifest.stlqpp", bytes, "manifest");
      !status.ok())
    return status;
  if (const auto status = stage.value().Publish(); !status.ok()) return status;

  return output;
}

using RepublishModel = StatusOr<CanonicalIndexRepublishResult> (*)(
    const CanonicalIndexRepublishRequest&, const CanonicalIndexFiles&,
    const F32GemmBackend&, const SpdSolverF64&, const ParallelForExecutor&,
    const IndexArtifactHashes&, const SourceNormProvider*);

StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndexImpl(
    const CanonicalIndexRepublishRequest& input, const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver, const ParallelForExecutor& parallel,
    int expected_m, RepublishModel publish_model) {
  if (const Status status = ValidatePaths(input); !status.ok()) {
    return status;
  }
  auto files_result = CanonicalIndexFiles::OpenCoefficients(input.source, true);
  if (!files_result.ok()) return files_result.status();
  auto files = std::move(files_result).value();
  for (const auto& shard : files.manifest().shards)
    if (shard.kind == ShardKind::kFloatNorm2) {
      std::error_code error;
      if (std::filesystem::exists(CanonicalShardPath(input.source, shard),
                                  error)) {
        auto complete = CanonicalIndexFiles::OpenNativeFloat(input.source);
        if (!complete.ok()) return complete.status();
        files = std::move(complete).value();
      } else if (error)
        return Status::Internal("cannot inspect native norm file: " +
                                error.message());
      break;
    }

  const auto m = files.manifest().shape.m;
  CanonicalIndexRepublishRequest effective = input;
  auto base_bits =
      NormalizeCoefficientBits(std::move(effective.coefficient_bits_per_layer),
                               m, "large.linkage_coeff_codec.bits_per_layer");
  if (!base_bits.ok()) return base_bits.status();
  effective.coefficient_bits_per_layer = std::move(base_bits).value();
  if (!effective.coefficient_bits_per_layer_root.empty()) {
    auto root_bits = NormalizeCoefficientBits(
        std::move(effective.coefficient_bits_per_layer_root), m,
        "large.linkage_coeff_codec.bits_per_layer_root");
    if (!root_bits.ok()) return root_bits.status();
    effective.coefficient_bits_per_layer_root = std::move(root_bits).value();
  }
  if (!effective.coefficient_bits_per_layer_linkage.empty()) {
    auto linkage_bits = NormalizeCoefficientBits(
        std::move(effective.coefficient_bits_per_layer_linkage), m,
        "large.linkage_coeff_codec.bits_per_layer_linkage");
    if (!linkage_bits.ok()) return linkage_bits.status();
    effective.coefficient_bits_per_layer_linkage =
        std::move(linkage_bits).value();
  }
  const CanonicalIndexRepublishRequest& request = effective;
  const bool needs_source_norm =
      request.publish_scaled_score8 || request.publish_float_scaled;
  std::optional<SourceNormProvider> external_source_norm2;
  const SourceNormProvider* source_norm2 = request.source_norm2;
  if (needs_source_norm && source_norm2 == nullptr) {
    auto source = SourceNormProvider::OpenForIndex(
        request.source, files.manifest(), request.base_source,
        request.base_source_format, request.base_source_first_row,
        request.base_source_row_count, parallel);
    if (!source.ok()) return source.status();
    external_source_norm2.emplace(std::move(source).value());
    source_norm2 = &*external_source_norm2;
  }
  if (expected_m != 0 && m != static_cast<std::uint32_t>(expected_m)) {
    return Status::InvalidArgument(
        "codec republish M disagrees with the source index");
  }
  CoefficientCodecRecipe codec = RegisteredCoefficientCodecRecipe(
      request.coefficient_first_magnitude_quantiles,
      request.coefficient_rest_magnitude_quantiles);
  codec.bits_per_layer = request.coefficient_bits_per_layer;
  codec.bits_per_layer_root = request.coefficient_bits_per_layer_root;
  codec.bits_per_layer_linkage = request.coefficient_bits_per_layer_linkage;
  codec.weighted_quantile = request.coefficient_codec_options.weighted_quantile;
  codec.allow_clipping = request.coefficient_codec_options.allow_clipping;
  codec.fit_reconstruction_scales =
      request.coefficient_codec_options.fit_reconstruction_scales;
  codec.refinement_sweeps = request.coefficient_codec_options.refinement_sweeps;
  codec.refine_all_layers =
      request.coefficient_codec_options.refinement_max_layer == 0U;
  codec.refinement_max_layer =
      request.coefficient_codec_options.refinement_max_layer;
  codec.refinement_step_limit =
      request.coefficient_codec_options.refinement_step_limit;
  const ArtifactHash64 linkage = files.manifest().hashes.linkage_list_directory;
  const ArtifactHash64 coefficient = ArtifactHashFactory::CoefficientCodec({
      .linkage_list_directory = linkage,
      .codec = codec,
  });
  const IndexArtifactHashes hashes = {
      .linkage_list_directory = linkage,
      .coefficient_codec =
          request.publish_quantized_forest ? coefficient : ArtifactHash64{},
      .norm8_cache =
          request.publish_quantized_forest
              ? ArtifactHashFactory::Norm8Cache({
                    .linkage_list_directory = linkage,
                    .coefficient_codec = coefficient,
                    .center_count = 256U,
                    .mode = request.norm8_mode,
                    .kmeans_iterations = request.norm8_lloyd_iterations,
                })
              : ArtifactHash64{},
  };
  return publish_model(request, files, gemm, coefficient_solver, parallel,
                       hashes, needs_source_norm ? source_norm2 : nullptr);
}
}  // namespace

template <int M>
StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex(
    const CanonicalIndexRepublishRequest& input, const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel) {
  return RepublishCanonicalIndexImpl(input, gemm, coefficient_solver, parallel,
                                     M, &RepublishTypedModel<M>);
}

template StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex<5>(
    const CanonicalIndexRepublishRequest&, const F32GemmBackend&,
    const SpdSolverF64&, const ParallelForExecutor&);

template StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex<12>(
    const CanonicalIndexRepublishRequest&, const F32GemmBackend&,
    const SpdSolverF64&, const ParallelForExecutor&);

template StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex<6>(
    const CanonicalIndexRepublishRequest&, const F32GemmBackend&,
    const SpdSolverF64&, const ParallelForExecutor&);
template StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex<10>(
    const CanonicalIndexRepublishRequest&, const F32GemmBackend&,
    const SpdSolverF64&, const ParallelForExecutor&);

template StatusOr<CanonicalIndexRepublishResult> RepublishCanonicalIndex<0>(
    const CanonicalIndexRepublishRequest&, const F32GemmBackend&,
    const SpdSolverF64&, const ParallelForExecutor&);

}  // namespace stlqpp
