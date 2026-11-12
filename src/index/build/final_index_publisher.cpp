#include "stlqpp/index/build/final_index_publisher.h"

#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/file_replication.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#if STLQPP_HAS_CUDA_NORM
#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/canonical_index_writer.h"

namespace stlqpp {

// This bridge is defined only in the spool publisher. Each input below has
// just returned successfully from a Read* method that validates its values.
struct PreparedSpoolPublicationAccess final {
  static StatusOr<ReconstructionNorm8Codebook> FitNorm8(
      std::vector<float> values, ReconstructionNorm8Mode mode,
      std::uint32_t iterations, const ParallelForExecutor& parallel) {
    return ReconstructionNorm8Codebook::FitFromValidatedValues(
        std::move(values), iterations, parallel, mode);
  }
  static Status EncodeNorm8(const ReconstructionNorm8Codebook& codebook,
                            ArrayView<const float> norm2,
                            ArrayView<std::uint8_t> labels) {
    return codebook.EncodeValidated(norm2, labels);
  }

  static Status AccumulateScaled(ScaledScore8Accumulator& accumulator,
                                 ArrayView<const std::uint8_t> labels,
                                 ArrayView<const float> reconstruction_norm2,
                                 ArrayView<const float> source_norm2,
                                 const ParallelForExecutor& parallel) {
    return accumulator.AccumulateValidated(labels, reconstruction_norm2,
                                           source_norm2, parallel);
  }

#if STLQPP_HAS_CUDA_NORM
  static StatusOr<ReconstructionNorm8Codebook> FitCudaNorm8(
      ArrayView<const float> values, std::uint32_t iterations, int device) {
    auto gpu = CudaNorm8Fitter::Create(values, device);
    if (!gpu.ok()) return gpu.status();
    auto centers = gpu.value()->Fit(iterations);
    if (!centers.ok()) return centers.status();
    return ReconstructionNorm8Codebook::FromIdentityCenters(centers.value());
  }
#endif

  static Status AppendEncoded(RuntimeCanonicalIndexWriter& writer,
                              const CanonicalEncodedClusterView& source) {
    return writer.AppendEncodedValidatedFloat(source);
  }
};

namespace {

// Borrow recipe metadata for the duration of publication; payloads and model
// matrices are not copied by the fixed/runtime adapter.
struct FinalIndexPublishRequestView {
  const std::filesystem::path& destination;
  RuntimeCanonicalModelSource model;
  const IndexArtifactHashes& hashes;
  const std::vector<std::uint32_t>& coefficient_bits_per_layer;
  const std::vector<std::uint32_t>& coefficient_bits_per_layer_root;
  const std::vector<std::uint32_t>& coefficient_bits_per_layer_linkage;
  const std::vector<double>& coefficient_first_magnitude_quantiles;
  const std::vector<double>& coefficient_rest_magnitude_quantiles;
  const ScalarCoefficientCodecOptions& coefficient_codec_options;
  ReconstructionNorm8Mode norm8_mode;
  std::uint32_t norm8_lloyd_iterations;
  bool use_cuda_norm8_fit;
  int cuda_norm_device;
  bool retain_float_forest;
  bool retain_float_norm2;
  bool publish_quantized_forest;
  bool publish_scaled_score8;
  bool publish_float_scaled;
  const std::filesystem::path& model_reference;
};

constexpr std::uint64_t kNormRowsPerChunk = 65536U;
constexpr std::size_t kNormLabelBatchRows = 65536U;
constexpr std::size_t kScaledNormBatchRows = 262144U;

bool SameHashes(const IndexArtifactHashes& left,
                const IndexArtifactHashes& right) noexcept {
  return left.linkage_list_directory == right.linkage_list_directory &&
         left.coefficient_codec == right.coefficient_codec &&
         left.norm8_cache == right.norm8_cache;
}

bool SameShape(const ModelShape& left, const ModelShape& right) noexcept {
  return left.dimension == right.dimension && left.m == right.m &&
         left.nlist == right.nlist;
}

StatusOr<std::vector<float>> ReadCompleteNormPopulation(
    const PreparedClusterSpoolStorage& spool) {
  const std::uint64_t rows = spool.counts().real;
  if (rows == 0U || rows > std::numeric_limits<std::size_t>::max()) {
    return Status::ResourceExhausted(
        "prepared norm stream does not fit the host row domain");
  }
  std::vector<float> values(static_cast<std::size_t>(rows));
  for (std::uint64_t begin = 0U; begin < rows; begin += kNormRowsPerChunk) {
    const std::uint64_t count = std::min(kNormRowsPerChunk, rows - begin);
    if (const Status status = spool.ReadNorm2(
            begin, {values.data() + static_cast<std::size_t>(begin),
                    static_cast<std::size_t>(count)});
        !status.ok()) {
      return status;
    }
  }
  return values;
}


Status ValidatePublishRequest(const PreparedClusterSpoolStorage& spool,
                              const FinalIndexPublishRequestView& request) {
  const auto valid_candidates = [](const std::vector<double>& candidates) {
    return !candidates.empty() &&
           std::all_of(candidates.begin(), candidates.end(), [](double value) {
             return std::isfinite(value) && value > 0.0 && value <= 1.0;
           });
  };
  if (request.destination.empty() || request.destination == spool.directory() ||
      !SameShape(request.model.shape, spool.shape()) ||
      !SameHashes(request.hashes, spool.hashes()) ||
      request.retain_float_forest != spool.has_float_forest() ||
      (request.retain_float_forest && request.retain_float_norm2 &&
       !spool.has_float_norm2()) ||
      request.publish_quantized_forest != spool.has_quantized_forest() ||
      (request.publish_scaled_score8 || request.publish_float_scaled) !=
          spool.has_source_norm2() ||
      (!request.publish_quantized_forest &&
       (!request.retain_float_forest || request.publish_scaled_score8)) ||
      (request.publish_float_scaled && !request.retain_float_forest) ||
      (request.publish_quantized_forest &&
       (!valid_candidates(request.coefficient_first_magnitude_quantiles) ||
        !valid_candidates(request.coefficient_rest_magnitude_quantiles) ||
        (request.norm8_mode != ReconstructionNorm8Mode::kIdentityLloyd &&
         request.norm8_mode != ReconstructionNorm8Mode::kTailWeightedLloyd) ||
        request.norm8_lloyd_iterations == 0U))) {
    return Status::InvalidArgument(
        "final publication request does not match its prepared spool");
  }
  return Status::Ok();
}

}  // namespace

StatusOr<ReconstructionNorm8Codebook> FitPreparedSpoolNorm8Impl(
    const PreparedClusterSpoolStorage& spool, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, const ParallelForExecutor& parallel,
    bool use_cuda_norm8_fit, int cuda_norm_device) {
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
  const auto fit_begin = std::chrono::steady_clock::now();
  LogInfo(std::string("Norm population/fit: begin (backend=") +
          (use_cuda_norm8_fit ? "CUDA)" : "CPU)"));
  auto values = ReadCompleteNormPopulation(spool);
  const double population_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - fit_begin).count();
  const auto log_fit_done = [&] {
    const double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fit_begin).count();
    LogInfo("Norm population/fit wall time(s): " + std::to_string(total) +
            " (population: " + std::to_string(population_seconds) +
            "; fit: " + std::to_string(total - population_seconds) + ")");
  };
  if (!values.ok()) {
    return values.status();
  }
#if STLQPP_HAS_CUDA_NORM
  if (use_cuda_norm8_fit) {
    auto accelerated = PreparedSpoolPublicationAccess::FitCudaNorm8(values.value(), lloyd_iterations,
                                            cuda_norm_device);
    if (!accelerated.ok()) return accelerated.status();
    log_fit_done();
    return accelerated;
  }
#endif
  auto codebook_result = PreparedSpoolPublicationAccess::FitNorm8(
      std::move(values).value(), mode, lloyd_iterations, parallel);
  if (!codebook_result.ok()) return codebook_result.status();
  log_fit_done();
  return codebook_result;
}

StatusOr<ReconstructionNorm8Codebook> FitPreparedSpoolNorm8(
    const PreparedClusterSpoolStorage& spool, ReconstructionNorm8Mode mode,
    std::uint32_t lloyd_iterations, const ParallelForExecutor& parallel) {
  return FitPreparedSpoolNorm8Impl(spool, mode, lloyd_iterations, parallel,
                                   false, 0);
}

static Status PublishPreparedClusterSpoolImpl(
    const std::filesystem::path& spool_directory,
    const FinalIndexPublishRequestView& request,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* statistics) {
  if (const Status status = ValidateDisjointFilesystemPaths(
          spool_directory, "prepared cluster spool", request.destination,
          "final index destination");
      !status.ok()) {
    return status;
  }
  FinalIndexPublicationStatistics completed;
  if (request.publish_quantized_forest || request.publish_scaled_score8 ||
      request.publish_float_scaled)
    completed.norm8_labels_scale_seconds = 0.0;
  bool direct_payloads = false;
  {
    auto spool_result = PreparedClusterSpoolStorage::Open(
        spool_directory, request.model.shape.m);
    if (!spool_result.ok()) {
      return spool_result.status();
    }
    auto spool = std::move(spool_result).value();
    direct_payloads = spool.direct_payloads();
    if (const Status status = ValidatePublishRequest(spool, request);
        !status.ok()) {
      return status;
    }
    std::optional<ReconstructionNorm8Codebook> norm8;
    if (request.publish_quantized_forest) {
      const auto begin = std::chrono::steady_clock::now();
      auto norm_result = FitPreparedSpoolNorm8Impl(
          spool, request.norm8_mode, request.norm8_lloyd_iterations, parallel,
          request.use_cuda_norm8_fit, request.cuda_norm_device);
      if (!norm_result.ok()) {
        return norm_result.status();
      }
      norm8.emplace(std::move(norm_result).value());
      completed.norm8_fit_wall_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - begin).count();
    }
#if STLQPP_HAS_CUDA_NORM
    std::unique_ptr<CudaNorm8LabelEncoder> gpu_labels;
    if (request.publish_quantized_forest && request.use_cuda_norm8_fit) {
      const auto begin = std::chrono::steady_clock::now();
      auto created = CudaNorm8LabelEncoder::Create(norm8->view(),
                                                   request.cuda_norm_device);
      if (!created.ok()) return created.status();
      gpu_labels = std::move(created).value();
      *completed.norm8_labels_scale_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - begin).count();
    }
#endif
    std::optional<IndexProducerRecipe> producer_recipe;
    if (request.publish_quantized_forest) {
      producer_recipe = IndexProducerRecipe{
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
    }
    RuntimeCanonicalIndexWriteRequest writer_request{
        .destination = request.destination,
        .model = request.model,
        .counts = spool.counts(),
        .hashes = request.hashes,
        .producer_recipe = std::move(producer_recipe),
        .norm8_centers = request.publish_quantized_forest
                             ? norm8->centers()
                             : ArrayView<const float>{},
        .retain_float_forest = request.retain_float_forest,
        .retain_float_norm2 = request.retain_float_norm2,
        .retain_float_source_norm2 = false,
        .publish_quantized_forest = request.publish_quantized_forest,
        .model_reference = request.model_reference,
        .norm_backend = request.use_cuda_norm8_fit ? "CUDA" : "CPU",
    };
    std::vector<CanonicalCidEntry> adopted_entries;
    if (spool.direct_payloads()) {
      adopted_entries.reserve(spool.shape().nlist);
      for (std::uint32_t cid = 0; cid < spool.shape().nlist; ++cid) {
        const auto& e = spool.entry(cid);
        adopted_entries.push_back(
            {.cid = cid,
             .real_count = e.real_count,
             .virtual_count = e.virtual_count,
             .real_root_count = e.real_root_count,
             .codes_offset = e.codes_offset,
             .codes_bytes = e.codes_bytes,
             .parents_offset = e.parents_offset,
             .parents_bytes = e.parents_bytes,
             .coefficients_offset = e.coefficients_offset,
             .coefficients_bytes = e.coefficients_bytes,
             .float_coefficients_offset = e.float_coefficient_byte_offset});
      }
    }
    auto writer_result =
        spool.direct_payloads()
            ? RuntimeCanonicalIndexWriter::CreateAdopting(
                  writer_request, spool_directory, std::move(adopted_entries))
            : RuntimeCanonicalIndexWriter::Create(writer_request);
    if (!writer_result.ok()) {
      return writer_result.status();
    }
    auto writer = std::move(writer_result).value();
    std::optional<SequentialFileWriter> source_norm_stream;
    if (request.publish_scaled_score8 || request.publish_float_scaled) {
      auto stream = SequentialFileWriter::Create(
          spool_directory / "source_norm2.ordered");
      if (!stream.ok()) return stream.status();
      source_norm_stream.emplace(std::move(stream).value());
    }
    const auto payload_begin = std::chrono::steady_clock::now();
    LogInfo("Norm labels/scale and payload publication: begin");
    ScaledScore8Accumulator scaled_accumulator;
    std::vector<float> norms;
    std::vector<float> source_norms;
    std::vector<std::uint8_t> labels;
    std::vector<float> float_coefficients;
    std::vector<float> float_norm2;
    std::uint32_t norm_batch_end = 0U;
    std::size_t norm_batch_cursor = 0U;
    for (std::uint32_t cid = 0U; cid < spool.shape().nlist; ++cid) {
      const PreparedClusterSpoolEntry& entry = spool.entry(cid);
      ArrayView<const std::uint8_t> cid_labels;
      ArrayView<const float> cid_source_norms;
      if (request.publish_quantized_forest) {
        if (cid >= norm_batch_end) {
          const auto begin = std::chrono::steady_clock::now();
          norm_batch_end = cid + 1U;
          std::size_t batch_rows = entry.real_count;
          bool batch_labels = request.publish_scaled_score8;
#if STLQPP_HAS_CUDA_NORM
          batch_labels = batch_labels || gpu_labels != nullptr;
#endif
          if (batch_labels) {
            // Read in logical CID order through the spool's physical-offset
            // mapping. Reuse the existing buffers; one oversized CID stays
            // whole so canonical append and source-norm alignment are unchanged.
            const auto budget = request.publish_scaled_score8
                ? kScaledNormBatchRows : kNormLabelBatchRows;
            while (norm_batch_end < spool.shape().nlist &&
                   batch_rows <= budget) {
              const auto next_rows = spool.entry(norm_batch_end).real_count;
              if (next_rows > budget - batch_rows) break;
              batch_rows += next_rows;
              ++norm_batch_end;
            }
          }
          norm_batch_cursor = 0U;
          norms.resize(batch_rows);
          labels.resize(batch_rows);
          if (const Status status =
                  spool.ReadNorm2(entry.norm2_row_offset, norms);
              !status.ok()) {
            return status;
          }
          if (const Status status =
#if STLQPP_HAS_CUDA_NORM
                  gpu_labels ? gpu_labels->Encode(norms, labels) :
#endif
                             PreparedSpoolPublicationAccess::EncodeNorm8(
                                 *norm8, norms, labels);
              !status.ok()) {
            return status;
          }
          if (request.publish_scaled_score8 || request.publish_float_scaled) {
            source_norms.resize(batch_rows);
            if (const auto status = spool.ReadSourceNorm2(entry.norm2_row_offset, source_norms);
                !status.ok()) return status;
            if (request.publish_scaled_score8) {
              if (const auto status = PreparedSpoolPublicationAccess::AccumulateScaled(
                      scaled_accumulator, labels, norms, source_norms, parallel);
                  !status.ok()) return status;
            }
          }
          *completed.norm8_labels_scale_seconds += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - begin).count();
        }
        if (entry.real_count != 0U) {
          cid_labels = {labels.data() + norm_batch_cursor, entry.real_count};
          if (request.publish_scaled_score8 || request.publish_float_scaled)
            cid_source_norms = {source_norms.data() + norm_batch_cursor, entry.real_count};
        }
        norm_batch_cursor += entry.real_count;
      }
      if (request.publish_scaled_score8 || request.publish_float_scaled) {
        if (!request.publish_quantized_forest) {
          const auto begin = std::chrono::steady_clock::now();
          source_norms.resize(entry.real_count);
          if (const auto status = spool.ReadSourceNorm2(entry.norm2_row_offset, source_norms);
              !status.ok()) return status;
          cid_source_norms = source_norms;
          *completed.norm8_labels_scale_seconds += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - begin).count();
        }
        if constexpr (std::endian::native == std::endian::little) {
          if (auto status = source_norm_stream->Append({
                  reinterpret_cast<const std::uint8_t*>(cid_source_norms.data()),
                  cid_source_norms.size() * sizeof(float)}); !status.ok()) return status;
        } else {
          std::vector<std::uint8_t> bytes(cid_source_norms.size() * sizeof(float));
          for (std::size_t row = 0; row < cid_source_norms.size(); ++row)
            StoreLittleEndianF32(cid_source_norms[row], bytes.data() + row * 4U);
          if (auto status = source_norm_stream->Append(bytes); !status.ok()) return status;
        }
      }
      CanonicalClusterRecordViews records = spool.payload_records(cid);
      records.norm8 = cid_labels;
      if (request.retain_float_forest) {
        const std::uint64_t nodes =
            static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
        if (nodes > std::numeric_limits<std::size_t>::max() / spool.shape().m) {
          return Status::ResourceExhausted(
              "native-float CID exceeds the host coefficient domain");
        }
        if (!spool.direct_payloads()) {
          float_coefficients.resize(static_cast<std::size_t>(nodes) *
                                    spool.shape().m);
          if (const Status status =
                  spool.ReadFloatCoefficients(cid, float_coefficients);
              !status.ok()) {
            return status;
          }
        }
        if (request.retain_float_norm2) {
          float_norm2.resize(entry.real_count);
          if (const Status status =
                  spool.ReadFloatNorm2(entry.norm2_row_offset, float_norm2);
              !status.ok()) {
            return status;
          }
        }
      } else {
        float_coefficients.clear();
        float_norm2.clear();
      }
      if (const Status status = PreparedSpoolPublicationAccess::AppendEncoded(
              writer,
              {
                  .cid = cid,
                  .real_count = entry.real_count,
                  .virtual_count = entry.virtual_count,
                  .real_root_count = entry.real_root_count,
                  .records = records,
                  .float_coefficients_layer_major = float_coefficients,
                  .float_real_norm2 = float_norm2,
                  .float_source_norm2 = {},
              });
          !status.ok()) {
        return status;
      }
    }
    if (request.publish_scaled_score8) {
      const auto begin = std::chrono::steady_clock::now();
      auto scaled_result = scaled_accumulator.Finish(norm8->view());
      if (!scaled_result.ok()) return scaled_result.status();
      ScaledScore8Codebook scaled = std::move(scaled_result).value();
      const ArtifactHash64 hash = ArtifactHashFactory::ScaledScore8({
          .coefficient_codec = request.hashes.coefficient_codec,
          .source_norm8_cache = request.hashes.norm8_cache,
          .scales = scaled.scales(),
          .biases = scaled.biases(),
      });
      if (const Status status = writer.SetScaledScore8(scaled.view(), hash);
          !status.ok()) {
        return status;
      }
      *completed.norm8_labels_scale_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - begin).count();
    }
    if (source_norm_stream.has_value()) {
      if (auto status = source_norm_stream->Close(); !status.ok()) return status;
    }
    const double payload_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - payload_begin).count();
    // All payloads have been consumed. Release mappings before adopting and
    // renaming the spool directory; Windows cannot move mapped descendants.
    spool = PreparedClusterSpoolStorage{};
    const auto commit_begin = std::chrono::steady_clock::now();
    if (const Status status = writer.Commit(&completed.canonical);
        !status.ok()) {
      return status;
    }
    const double commit_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - commit_begin).count();
    LogInfo("Norm labels/scale and payload publication wall time(s): " +
            std::to_string(payload_seconds + commit_seconds) +
            " (payload loop: " + std::to_string(payload_seconds) +
            " [labels/scale service: " +
            std::to_string(completed.norm8_labels_scale_seconds.value_or(0.0)) +
            "]; canonical commit: " + std::to_string(commit_seconds) + ")");
    if (request.publish_quantized_forest) {
      completed.norm8_mode = request.norm8_mode;
      completed.norm8_lloyd_iterations = request.norm8_lloyd_iterations;
      completed.norm8_center_count =
          static_cast<std::uint32_t>(norm8->centers().size());
    }
  }
  std::error_code error;
  if (request.publish_scaled_score8 || request.publish_float_scaled) {
    // Preserve the already computed source norms until the application binds
    // them to the source-slice identity and writes the stable-ID norm cache.
    // Direct publication adopts the spool directory; copied publication does
    // not. Neither path needs another traversal of the original vectors.
    const auto pending = request.destination /
        "norm-cache/constant/source_norm2.pending";
    std::filesystem::create_directories(pending.parent_path(), error);
    if (error) return Status::Internal("cannot create source norm cache directory: " +
                                       error.message());
    const auto source = (direct_payloads ? request.destination
                                        : spool_directory) /
                        "source_norm2.ordered";
    std::filesystem::rename(source, pending, error);
    if (error == std::errc::cross_device_link) {
      auto copied = ReplicateFile(source, pending);
      if (!copied.ok()) return copied.status();
      std::filesystem::remove(source, error);
    }
    if (error) return Status::Internal("cannot retain source norm stream: " +
                                       error.message());
  }
  for (const auto* name :
       {"records.bin", "norm2.f32", "float_norm2.f32", "source_norm2.f32",
        "spool.manifest", "checkpoint.0.stlqpp", "checkpoint.1.stlqpp"}) {
    std::filesystem::remove(request.destination / name, error);
    if (error)
      return Status::Internal("published index temporary cleanup failed: " +
                              error.message());
  }
  std::filesystem::remove_all(spool_directory, error);
  if (error) {
    return Status::Internal(
        "canonical index published but prepared spool cleanup failed: " +
        error.message());
  }
  if (statistics != nullptr) {
    *statistics = completed;
  }
  return Status::Ok();
}

template <int M>
Status PublishPreparedClusterSpool(
    const std::filesystem::path& directory,
    const FinalIndexPublishRequest<M>& request,
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* statistics) {
  const auto& books = request.model.codebooks;
  const RuntimeCanonicalModelSource model = {
      books.shape, request.model.rotation, books.root0, books.root_small,
      books.edge};
  return PublishPreparedClusterSpoolImpl(
      directory,
      {request.destination, model, request.hashes,
       request.coefficient_bits_per_layer,
       request.coefficient_bits_per_layer_root,
       request.coefficient_bits_per_layer_linkage,
       request.coefficient_first_magnitude_quantiles,
       request.coefficient_rest_magnitude_quantiles,
       request.coefficient_codec_options, request.norm8_mode,
       request.norm8_lloyd_iterations, request.use_cuda_norm8_fit,
       request.cuda_norm_device, request.retain_float_forest, request.retain_float_norm2,
       request.publish_quantized_forest, request.publish_scaled_score8,
       request.publish_float_scaled, request.model_reference},
      parallel, statistics);
}

template Status PublishPreparedClusterSpool<0>(
    const std::filesystem::path&, const FinalIndexPublishRequest<0>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);
template Status PublishPreparedClusterSpool<5>(
    const std::filesystem::path&, const FinalIndexPublishRequest<5>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);
template Status PublishPreparedClusterSpool<12>(
    const std::filesystem::path&, const FinalIndexPublishRequest<12>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);
template Status PublishPreparedClusterSpool<6>(
    const std::filesystem::path&, const FinalIndexPublishRequest<6>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);
template Status PublishPreparedClusterSpool<10>(
    const std::filesystem::path&, const FinalIndexPublishRequest<10>&,
    const ParallelForExecutor&, FinalIndexPublicationStatistics*);

}  // namespace stlqpp
