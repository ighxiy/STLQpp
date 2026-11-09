#include "eval_archive.h"
#include "stlqpp/application/query_disk_footprint.h"
#include "eval_command.h"
#include "evaluate_base_basic.h"
#include "run_exact_dense_eval.h"
#include "texmex_reader.h"

#include "stlqpp/application/final_index_log.h"
#include "stlqpp/application/index_republisher.h"
#include "stlqpp/application/registered_artifact_recipes.h"
#include "stlqpp/application/scaled_score8_derivation.h"
#include "stlqpp/application/eval_norm_cache.h"
#include "stlqpp/application/sha256.h"
#include "stlqpp/model/model_dispatch.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#include "stlqpp/runtime/cpu/lapack_spd_solver_f64.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/cpu/openmp_parallel_for.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/search/plans/dense_cpu_plan.h"
#include "stlqpp/search/plans/open_exact_dense_searcher.h"
#include "stlqpp/search/plans/runtime_exact_dense_searcher.h"
#include "stlqpp/search/rerank/exact_l2_reranker.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/index_manifest_codec.h"
#include "stlqpp/storage/artifact_hash_factory.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class ScopedCodecWorkspace final {
 public:
  explicit ScopedCodecWorkspace(std::filesystem::path path) noexcept
      : path_(std::move(path)) {}
  ScopedCodecWorkspace(const ScopedCodecWorkspace&) = delete;
  ScopedCodecWorkspace& operator=(const ScopedCodecWorkspace&) = delete;
  ScopedCodecWorkspace(ScopedCodecWorkspace&& other) noexcept
      : path_(std::exchange(other.path_, {})) {}
  ScopedCodecWorkspace& operator=(ScopedCodecWorkspace&&) = delete;
  ~ScopedCodecWorkspace() {
    if (!path_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

stlqpp::StatusOr<ScopedCodecWorkspace> CreateCodecWorkspace(
    const std::filesystem::path& work_root) {
  std::random_device random;
  for (std::uint32_t attempt = 0U; attempt < 32U; ++attempt) {
    const std::filesystem::path candidate = work_root /
        (".codec-work-" + std::to_string(random()) + "-" +
         std::to_string(random()));
    std::error_code error;
    if (std::filesystem::create_directory(candidate, error)) {
      return ScopedCodecWorkspace(candidate);
    }
    if (error && error != std::errc::file_exists) {
      return stlqpp::Status::Internal(
          "cannot create codec workspace: " + error.message());
    }
  }
  return stlqpp::Status::ResourceExhausted(
      "cannot reserve a unique codec workspace");
}

stlqpp::StatusOr<stlqpp::IndexManifest> ReadIndexManifest(
    const std::filesystem::path& directory) {
  auto file = stlqpp::ReadOnlyMappedFile::Open(directory / "manifest.stlqpp");
  if (!file.ok()) return file.status();
  stlqpp::IndexManifest manifest;
  if (const stlqpp::Status status = stlqpp::IndexManifestCodec::Decode(
          file.value().data(), file.value().size(), manifest);
      !status.ok()) {
    return status;
  }
  return manifest;
}

bool ShardExists(const std::filesystem::path& index, const stlqpp::IndexManifest& manifest,
                 stlqpp::ShardKind kind, std::error_code& error) {
  for (const auto& shard : manifest.shards) if (shard.kind == kind)
    return std::filesystem::exists(stlqpp::CanonicalShardPath(index, shard), error);
  error.clear();
  return false;
}

stlqpp::Status PrepareOriginalNorms(stlqpp::app::EvalCommand& command,
                                  const stlqpp::IndexManifest& manifest,
                                  const stlqpp::ParallelForExecutor& parallel) {
  if (command.prepared_original_norms) return stlqpp::Status::Ok();
  auto source = stlqpp::SourceNormProvider::OpenForIndex(
      command.index, manifest, command.effective_base_source, command.base_source_format,
      command.base_source_first_row, command.base_source_row_count, parallel);
  if (!source.ok()) return source.status();
  command.prepared_original_norms = std::make_shared<stlqpp::SourceNormProvider>(std::move(source).value());
  return stlqpp::Status::Ok();
}

stlqpp::Status BindFloatScaledContentIdentity(
    stlqpp::app::EvalCommand& command,
    const stlqpp::IndexManifest& manifest) {
  if (std::find(command.representation_runs.begin(),
                command.representation_runs.end(),
                stlqpp::EvalRepresentation::kFloatScaled) ==
      command.representation_runs.end()) {
    return stlqpp::Status::Ok();
  }
  const auto shard = std::find_if(
      manifest.shards.begin(), manifest.shards.end(),
      [](const stlqpp::ShardDescriptor& value) {
        return value.kind == stlqpp::ShardKind::kFloatSourceNorm2;
      });
  if (shard == manifest.shards.end()) {
    return stlqpp::Status::DataLoss(
        "float_scaled source-norm shard is absent");
  }
  const auto content = stlqpp::CanonicalShardIdentity(command.index, manifest, *shard);
  const std::string identity = stlqpp::app::Sha256Bytes(
      "float-scaled-score-v1\n" +
      std::to_string(manifest.format_version) + "\n" +
      manifest.hashes.linkage_list_directory.ToHex() + "\n" +
      manifest.hashes.coefficient_codec.ToHex() + "\n" +
      content);
  std::uint64_t value = 0U;
  const auto parsed = std::from_chars(
      identity.data(), identity.data() + 16U, value, 16);
  if (parsed.ec != std::errc{}) {
    return stlqpp::Status::Internal(
        "cannot derive float_scaled score identity");
  }
  command.float_scaled_score_codec_hash =
      stlqpp::ArtifactHash64::FromValue(value);
  return stlqpp::Status::Ok();
}

stlqpp::StatusOr<std::filesystem::path> DerivedCacheDestination(
    const stlqpp::app::EvalCommand& command,
    std::string_view kind, bool uses_external_source) {
  auto source_manifest = stlqpp::ReadOnlyMappedFile::Open(
      command.index / "manifest.stlqpp");
  if (!source_manifest.ok()) return source_manifest.status();
  std::error_code error;
  const auto source_path = std::filesystem::canonical(command.index, error);
  if (error) {
    return stlqpp::Status::InvalidArgument(
        "cannot resolve scaled derivation source index");
  }
  std::string identity = "scaled-derivation-v2\n";
  identity += kind;
  identity += '\n';
  identity += source_path.generic_string();
  identity += '\n';
  identity.append(
      reinterpret_cast<const char*>(source_manifest.value().data()),
      source_manifest.value().size());
  if (uses_external_source && command.prepared_original_norms) {
    identity += "\n" + command.prepared_original_norms->identity();
  } else if (uses_external_source) {
    const auto base_path = std::filesystem::canonical(
        command.effective_base_source, error);
    if (error) {
      return stlqpp::Status::InvalidArgument(
          "cannot resolve dataset.base_path for scaled derivation");
    }
    const auto bytes = std::filesystem::file_size(base_path, error);
    if (error) {
      return stlqpp::Status::InvalidArgument(
          "cannot inspect dataset.base_path size");
    }
    const auto modified = std::filesystem::last_write_time(base_path, error);
    if (error) {
      return stlqpp::Status::InvalidArgument(
          "cannot inspect dataset.base_path timestamp");
    }
    identity += "\n" + base_path.generic_string() + "\n" +
        std::to_string(bytes) + "\n" +
        std::to_string(modified.time_since_epoch().count()) + "\n" +
        command.base_source_format + "\n" +
        std::to_string(command.base_source_first_row) + "\n" +
        std::to_string(command.base_source_row_count);
  }
  // Keep all representations beside the original index shards, including
  // derivations made from an earlier derived representation.
  std::filesystem::path base_index;
  for (auto path = source_path; !path.empty(); path = path.parent_path()) {
    if (path.filename() == "linkage_list") {
      base_index = path;
      break;
    }
    if (path == path.root_path()) break;
  }
  std::filesystem::path cache_root;
  if (!base_index.empty()) {
    cache_root = base_index / "derived-index-cache";
  } else {
    // Non-product indexes have no linkage_list owner. Keep their existing
    // per-run cache behavior; only product indexes use the single nested
    // cache namespace.
    cache_root = command.run_root / "derived-index-cache";
    const auto resolved_cache = std::filesystem::weakly_canonical(
        cache_root, error);
    if (error) {
      return stlqpp::Status::InvalidArgument(
          "cannot resolve scaled derivation cache directory");
    }
    const auto overlap = std::mismatch(
        source_path.begin(), source_path.end(),
        resolved_cache.begin(), resolved_cache.end());
    if (overlap.first == source_path.end()) {
      cache_root = source_path.parent_path() /
          (source_path.filename().string() + "-derived-index-cache");
    }
  }
  std::filesystem::create_directories(cache_root, error);
  if (error) {
    return stlqpp::Status::Internal(
        "cannot create scaled derivation cache directory: " +
        error.message());
  }
  return cache_root / stlqpp::app::Sha256Bytes(identity);
}

stlqpp::StatusOr<bool> UseCachedDerivedIndex(
    stlqpp::app::EvalCommand& command,
    stlqpp::IndexManifest* manifest,
    const std::filesystem::path& destination,
    stlqpp::EvalRepresentation required) {
  std::error_code error;
  if (!std::filesystem::exists(destination, error)) {
    if (error) {
      return stlqpp::Status::Internal(
          "cannot inspect scaled derivation cache: " + error.message());
    }
    return false;
  }
  auto cached = ReadIndexManifest(destination);
  if (!cached.ok()) return cached.status();
  const bool has_representation =
      required == stlqpp::EvalRepresentation::kQuantizedScaled
          ? cached.value().has_scaled_score8()
          : cached.value().has_float_source_norm2();
  if (!has_representation ||
      cached.value().hashes.linkage_list_directory !=
          manifest->hashes.linkage_list_directory ||
      cached.value().hashes.coefficient_codec !=
          manifest->hashes.coefficient_codec ||
      cached.value().hashes.norm8_cache !=
          manifest->hashes.norm8_cache) {
    return stlqpp::Status::DataLoss(
        "scaled derivation cache disagrees with its source index");
  }
  command.index = destination;
  *manifest = std::move(cached).value();
  if (command.norm_preparation_backend == "existing_index") {
    command.norm_preparation_backend = "cache_hit";
  }
  stlqpp::LogInfo("Eval derived representation cache hit: " +
                  destination.string() + "; no norm computation");
  return true;
}

stlqpp::Status EnsureRequestedCoefficientCodec(
    stlqpp::app::EvalCommand& command,
    stlqpp::IndexManifest* manifest,
    bool quantized, bool quantized_scaled, bool float_scaled) {
  if (!quantized) return stlqpp::Status::Ok();
  const bool codec_requested =
      command.expected_coefficient_bits_per_layer.has_value() ||
      command.expected_coefficient_bits_per_layer_root.has_value() ||
      command.expected_coefficient_bits_per_layer_linkage.has_value() ||
      command.expected_coefficient_weighted_quantile.has_value() ||
      command.expected_coefficient_allow_clipping.has_value() ||
      command.expected_coefficient_fit_scale.has_value() ||
      command.expected_coefficient_refinement_sweeps.has_value() ||
      command.expected_coefficient_refinement_max_layer.has_value() ||
      command.expected_coefficient_refinement_step_limit.has_value() ||
      !command.expected_coefficient_first_quantile_percents.empty() ||
      !command.expected_coefficient_rest_quantile_percents.empty();
  std::error_code coefficient_error;
  const bool coefficients_present = manifest->has_quantized_forest() &&
      ShardExists(command.index, *manifest, stlqpp::ShardKind::kCoefficients, coefficient_error);
  if (coefficient_error) return stlqpp::Status::Internal("cannot inspect coefficient shard: " + coefficient_error.message());
  if (!codec_requested && manifest->has_classic_norm8() && coefficients_present) {
    return stlqpp::Status::Ok();
  }
  // A native float-only publication has no quantized producer. Its retained
  // forest is sufficient to create the first codec from this eval recipe.
  stlqpp::IndexProducerRecipe initial_producer;
  if (!manifest->producer_recipe.has_value()) {
    const auto defaults = stlqpp::RegisteredCoefficientCodecRecipe(1.0);
    initial_producer.coefficient_bits_per_layer.assign(
        command.m, stlqpp::kDefaultCoefficientBits);
    initial_producer.coefficient_first_magnitude_quantiles =
        defaults.first_magnitude_quantiles;
    initial_producer.coefficient_rest_magnitude_quantiles =
        defaults.rest_magnitude_quantiles;
    initial_producer.norm8_lloyd_iterations = 100U;
    stlqpp::LogInfo(
        "Eval native-float source has no coefficient producer; deriving "
        "the first quantized representation from the effective eval recipe "
        "(unspecified defaults: 6 bits/layer, 100% quantiles, 100 norm8 "
        "Lloyd iterations)");
  }
  const auto& producer = manifest->producer_recipe.has_value()
      ? *manifest->producer_recipe : initial_producer;
  stlqpp::CoefficientCodecRecipe producer_codec;
  producer_codec.bits_per_layer = producer.coefficient_bits_per_layer;
  producer_codec.bits_per_layer_root =
      producer.coefficient_bits_per_layer_root;
  producer_codec.bits_per_layer_linkage =
      producer.coefficient_bits_per_layer_linkage;
  producer_codec.first_magnitude_quantiles =
      producer.coefficient_first_magnitude_quantiles;
  producer_codec.rest_magnitude_quantiles =
      producer.coefficient_rest_magnitude_quantiles;
  producer_codec.weighted_quantile = producer.coefficient_weighted_quantile;
  producer_codec.allow_clipping = producer.coefficient_allow_clipping;
  producer_codec.fit_reconstruction_scales = producer.coefficient_fit_scale;
  producer_codec.refinement_sweeps = producer.coefficient_refinement_sweeps;
  producer_codec.refinement_max_layer =
      producer.coefficient_refinement_max_layer;
  producer_codec.refine_all_layers =
      producer_codec.refinement_max_layer == 0U;
  producer_codec.refinement_step_limit =
      producer.coefficient_refinement_step_limit;
  stlqpp::CoefficientCodecRecipe codec = producer_codec;
  if (command.expected_coefficient_bits_per_layer.has_value()) {
    codec.bits_per_layer = *command.expected_coefficient_bits_per_layer;
  }
  if (command.expected_coefficient_bits_per_layer_root.has_value()) {
    codec.bits_per_layer_root =
        *command.expected_coefficient_bits_per_layer_root;
  }
  if (command.expected_coefficient_bits_per_layer_linkage.has_value()) {
    codec.bits_per_layer_linkage =
        *command.expected_coefficient_bits_per_layer_linkage;
  }
  if (codec.bits_per_layer.empty() &&
      (!codec.bits_per_layer_root.empty() ||
       !codec.bits_per_layer_linkage.empty())) {
    codec.bits_per_layer.assign(command.m, stlqpp::kDefaultCoefficientBits);
  }
  if (!command.expected_coefficient_first_quantile_percents.empty()) {
    codec.first_magnitude_quantiles.clear();
    codec.rest_magnitude_quantiles.clear();
    for (double value : command.expected_coefficient_first_quantile_percents)
      codec.first_magnitude_quantiles.push_back(value / 100.0);
    for (double value : command.expected_coefficient_rest_quantile_percents)
      codec.rest_magnitude_quantiles.push_back(value / 100.0);
  }
  codec.weighted_quantile =
      command.expected_coefficient_weighted_quantile.value_or(
          codec.weighted_quantile);
  codec.allow_clipping =
      command.expected_coefficient_allow_clipping.value_or(
          codec.allow_clipping);
  codec.fit_reconstruction_scales =
      command.expected_coefficient_fit_scale.value_or(
          codec.fit_reconstruction_scales);
  codec.refinement_sweeps =
      command.expected_coefficient_refinement_sweeps.value_or(
          codec.refinement_sweeps);
  codec.refinement_max_layer =
      command.expected_coefficient_refinement_max_layer.value_or(
          codec.refinement_max_layer);
  codec.refine_all_layers = codec.refinement_max_layer == 0U;
  codec.refinement_step_limit =
      command.expected_coefficient_refinement_step_limit.value_or(
          codec.refinement_step_limit);
  const stlqpp::ArtifactHash64 requested =
      stlqpp::ArtifactHashFactory::CoefficientCodec({
          .linkage_list_directory =
              manifest->hashes.linkage_list_directory,
          .codec = codec,
      });
  auto producer_effective_codec = producer_codec;
  if (producer_effective_codec.bits_per_layer.empty()) {
    producer_effective_codec.bits_per_layer.assign(
        command.m, stlqpp::kDefaultCoefficientBits);
  }
  const stlqpp::ArtifactHash64 producer_effective =
      stlqpp::ArtifactHashFactory::CoefficientCodec({
          .linkage_list_directory =
              manifest->hashes.linkage_list_directory,
          .codec = producer_effective_codec,
      });
  if (coefficients_present && requested == manifest->hashes.coefficient_codec) {
    return stlqpp::Status::Ok();
  }
  if (coefficients_present && manifest->producer_recipe.has_value() &&
      requested == producer_effective) {
    stlqpp::LogInfo(
        "Eval effective coefficient codec matches the stored producer "
        "recipe; reusing existing coefficients despite a legacy hash "
        "difference");
    return stlqpp::Status::Ok();
  }
  stlqpp::LogInfo(
      "Eval coefficient-codec hash changed; regenerate int coefficients "
      "and norm8 from the retained float forest");
  const auto target_norm_mode = stlqpp::PublishedNorm8Mode(
      command.expected_norm8_mode.value_or(producer.norm8_mode));
  const std::uint32_t target_norm_iterations =
      command.expected_norm8_lloyd_iterations.value_or(
          producer.norm8_lloyd_iterations);
  if (quantized_scaled || float_scaled) {
    auto parallel = stlqpp::OpenMpParallelFor::Create(command.effective_threads);
    if (!parallel.ok()) return parallel.status();
    if (const auto status = PrepareOriginalNorms(command, *manifest, parallel.value()); !status.ok())
      return status;
  }
  auto destination = DerivedCacheDestination(
      command, "coefficient-codec-v2-" + requested.ToHex() + "-norm8-" +
          std::to_string(static_cast<std::uint32_t>(target_norm_mode)) +
          "-" + std::to_string(target_norm_iterations),
      (quantized_scaled || float_scaled) &&
          !manifest->has_float_source_norm2());
  if (!destination.ok()) return destination.status();
  std::error_code error;
  const auto cache_base = destination.value();
  for (std::uint32_t generation = 0U;; ++generation) {
    const bool exists = std::filesystem::exists(destination.value(), error);
    if (error) return stlqpp::Status::Internal("cannot inspect coefficient-codec cache: " + error.message());
    if (!exists) break;
    const bool metadata = std::filesystem::exists(destination.value() / "manifest.stlqpp", error);
    if (error) return stlqpp::Status::Internal("cannot inspect coefficient cache metadata: " + error.message());
    const bool payload = std::filesystem::exists(destination.value() / "coefficients.bin", error);
    if (error) return stlqpp::Status::Internal("cannot inspect coefficient cache payload: " + error.message());
    if (metadata && payload) break;
    destination.value() = cache_base.parent_path() /
        (cache_base.filename().string() + "-rebuild-" + std::to_string(generation + 1U));
  }
  if (std::filesystem::exists(destination.value(), error)) {
    if (error) return stlqpp::Status::Internal(
        "cannot inspect coefficient-codec cache: " + error.message());
    auto cached = ReadIndexManifest(destination.value());
    if (!cached.ok()) return cached.status();
    if (cached.value().hashes.linkage_list_directory !=
            manifest->hashes.linkage_list_directory ||
        cached.value().hashes.coefficient_codec != requested ||
        !cached.value().producer_recipe.has_value() ||
        cached.value().producer_recipe->norm8_mode != target_norm_mode ||
        cached.value().producer_recipe->norm8_lloyd_iterations !=
            target_norm_iterations ||
        (manifest->has_float_source_norm2() &&
         !cached.value().has_float_source_norm2())) {
      return stlqpp::Status::DataLoss(
          "coefficient-codec cache hash disagrees with its source");
    }
    command.index = destination.value();
    *manifest = std::move(cached).value();
    if (command.norm_preparation_backend == "existing_index") {
      command.norm_preparation_backend = "cache_hit";
    }
    stlqpp::LogInfo("Eval coefficient-codec cache hit: " +
                    command.index.string() + "; norm cache hit, no recomputation");
    return stlqpp::Status::Ok();
  }
  if (error) return stlqpp::Status::Internal(
      "cannot inspect coefficient-codec cache: " + error.message());
  if (!manifest->has_float_forest()) {
    return stlqpp::Status::FailedPrecondition(
        "coefficient-codec hash changed, but the linkage index has no "
        "retained float forest from which to regenerate int coefficients");
  }
  if (command.gpu_norm_enable && !command.use_cuda) {
    return stlqpp::Status::FailedPrecondition(
        "eval.linkage.gpu_norm_enable=true requires runtime.use_cuda=true "
        "for coefficient-codec norm republish");
  }
  if (command.gpu_norm_enable &&
      target_norm_mode != stlqpp::ReconstructionNorm8Mode::kIdentityLloyd) {
    return stlqpp::Status::Unimplemented(
        "GPU norm8 tail-weighted mode is not supported; set "
        "eval.linkage.gpu_norm_enable=false or use lut");
  }
  auto parallel_result = stlqpp::OpenMpParallelFor::Create(
      command.effective_threads);
  if (!parallel_result.ok()) return parallel_result.status();
  stlqpp::OpenMpParallelFor parallel = std::move(parallel_result).value();
  const stlqpp::CblasF32Gemm gemm;
  const stlqpp::LapackSpdSolverF64 solver;
  auto workspace_result = CreateCodecWorkspace(command.run_root);
  if (!workspace_result.ok()) return workspace_result.status();
  ScopedCodecWorkspace workspace = std::move(workspace_result).value();
  stlqpp::CanonicalIndexRepublishRequest request;
  request.source = command.index;
  request.prepared_spool = workspace.path() / "prepared";
  request.destination = destination.value();
  request.base_source = command.effective_base_source;
  request.base_source_format = command.base_source_format;
  request.base_source_first_row = command.base_source_first_row;
  request.base_source_row_count = command.base_source_row_count;
  request.coefficient_bits_per_layer = codec.bits_per_layer;
  request.coefficient_bits_per_layer_root = codec.bits_per_layer_root;
  request.coefficient_bits_per_layer_linkage = codec.bits_per_layer_linkage;
  request.coefficient_first_magnitude_quantiles =
      codec.first_magnitude_quantiles;
  request.coefficient_rest_magnitude_quantiles =
      codec.rest_magnitude_quantiles;
  request.coefficient_codec_options = {
      .weighted_quantile = codec.weighted_quantile,
      .allow_clipping = codec.allow_clipping,
      .fit_reconstruction_scales = codec.fit_reconstruction_scales,
      .refinement_sweeps = codec.refinement_sweeps,
      .refinement_max_layer = codec.refinement_max_layer,
      .refinement_step_limit = codec.refinement_step_limit,
  };
  request.norm8_mode = target_norm_mode;
  request.norm8_lloyd_iterations = target_norm_iterations;
  request.retain_float_forest = true;
  request.publish_quantized_forest = true;
  request.use_cuda_norm8_fit = command.gpu_norm_enable;
  request.cuda_norm_device = command.cuda_device;
  request.publish_scaled_score8 = quantized_scaled;
  request.publish_float_scaled = false;
  request.source_norm2 = command.prepared_original_norms.get();
  const auto start = std::chrono::steady_clock::now();
  const stlqpp::Status status = stlqpp::DispatchModelMWithRuntime(
      command.m, [&]<std::uint32_t M>() -> stlqpp::Status {
        auto republished = stlqpp::RepublishCanonicalIndex<M>(
            request, gemm, solver, parallel);
        return republished.ok() ? stlqpp::Status::Ok()
                                : republished.status();
      });
  if (!status.ok()) return status;
  auto updated = ReadIndexManifest(destination.value());
  if (!updated.ok()) return updated.status();
  if (updated.value().hashes.coefficient_codec != requested) {
    return stlqpp::Status::DataLoss(
        "republished coefficient codec has unexpected hash");
  }
  command.index = destination.value();
  *manifest = std::move(updated).value();
  command.norm_preparation_backend = command.gpu_norm_enable ? "CUDA" : "CPU";
  stlqpp::LogInfo("Eval coefficient codec rebuilt from float forest: " +
                  command.index.string() + " seconds=" +
                  std::to_string(std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start).count()));
  return stlqpp::Status::Ok();
}

stlqpp::Status EnsureRequestedRepresentations(
    stlqpp::app::EvalCommand& command,
    stlqpp::IndexManifest* manifest) {
  if (!command.linkage_enabled) {
    return stlqpp::Status::Ok();
  }
  if (command.requested_score_mode !=
          stlqpp::SearchScoreMode::kClassic &&
      std::any_of(command.requested_representations.begin(),
                  command.requested_representations.end(),
                  [](stlqpp::EvalRepresentation value) {
                    return value == stlqpp::EvalRepresentation::kFloatScaled ||
                           value == stlqpp::EvalRepresentation::kQuantizedScaled;
                  })) {
    return stlqpp::Status::InvalidArgument(
        "eval.linkage.score_mode=" + std::string(
            command.requested_score_mode == stlqpp::SearchScoreMode::kMipsRecon
                ? "mips_recon" : "cosine_recon") + " requires "
        "eval.linkage.modes=[float], [quantized], or [auto]");
  }
  const auto requested = [&](stlqpp::EvalRepresentation value) {
    if (command.requested_representations_auto) {
      switch (value) {
        case stlqpp::EvalRepresentation::kFloat:
          return manifest->has_float_forest();
        case stlqpp::EvalRepresentation::kFloatScaled:
          return command.requested_score_mode ==
                     stlqpp::SearchScoreMode::kClassic &&
                 manifest->has_float_source_norm2();
        case stlqpp::EvalRepresentation::kQuantized:
          return manifest->has_classic_norm8();
        case stlqpp::EvalRepresentation::kQuantizedScaled:
          return command.requested_score_mode ==
                     stlqpp::SearchScoreMode::kClassic &&
                 manifest->has_scaled_score8();
      }
    }
    return std::find(command.requested_representations.begin(),
                     command.requested_representations.end(), value) !=
           command.requested_representations.end();
  };
  const bool quantized = requested(
      stlqpp::EvalRepresentation::kQuantized) || requested(
      stlqpp::EvalRepresentation::kQuantizedScaled);
  const bool quantized_scaled = command.requested_norm_mode != stlqpp::app::EvalNormMode::kFloat && requested(
      stlqpp::EvalRepresentation::kQuantizedScaled);
  const bool float_scaled = command.requested_norm_mode == stlqpp::app::EvalNormMode::kFloat && requested(
      stlqpp::EvalRepresentation::kFloatScaled);
  if (const stlqpp::Status status = EnsureRequestedCoefficientCodec(
          command, manifest, quantized, quantized_scaled, float_scaled);
      !status.ok()) return status;
  std::error_code norm_error;
  const bool norm8_present = ShardExists(command.index, *manifest, stlqpp::ShardKind::kNorm8, norm_error);
  if (norm_error) return stlqpp::Status::Internal("cannot inspect norm8 shard: " + norm_error.message());
  const bool lut_present =
      manifest->format_version <
          stlqpp::IndexManifest::kNormCacheLayoutFormatVersion ||
      ShardExists(command.index, *manifest, stlqpp::ShardKind::kNorm2Lut,
                  norm_error);
  if(norm_error)return stlqpp::Status::Internal("cannot inspect norm2 LUT: "+norm_error.message());
  const bool scaled_present = ShardExists(command.index, *manifest, stlqpp::ShardKind::kScaledScore8, norm_error);
  if (norm_error) return stlqpp::Status::Internal("cannot inspect scaled score shard: " + norm_error.message());
  command.independent_quantized_norms = quantized &&
      (command.requested_norm_mode == stlqpp::app::EvalNormMode::kFloat ||
       command.expected_norm8_mode == stlqpp::ReconstructionNorm8Mode::kReferenceLloyd ||
       !norm8_present || !lut_present ||
       (quantized_scaled && manifest->has_scaled_score8() && !scaled_present));
  const bool norm_requested = quantized && !command.independent_quantized_norms &&
      command.requested_score_mode != stlqpp::SearchScoreMode::kMipsRecon &&
      (command.expected_norm8_mode.has_value() || command.expected_norm8_lloyd_iterations.has_value() ||
       (manifest->producer_recipe.has_value() && manifest->producer_recipe->norm8_mode !=
            stlqpp::ReconstructionNorm8Mode::kIdentityLloyd));
  if (norm_requested) {
    if (!manifest->producer_recipe.has_value()) {
      return stlqpp::Status::FailedPrecondition(
          "norm8 derivation requires producer recipe provenance");
    }
    const auto& recipe = *manifest->producer_recipe;
    const auto target_mode = command.expected_norm8_mode.value_or(
        stlqpp::ReconstructionNorm8Mode::kIdentityLloyd);
    const std::uint32_t target_iterations =
        command.expected_norm8_lloyd_iterations.value_or(
            recipe.norm8_lloyd_iterations);
    if (target_mode != recipe.norm8_mode ||
        target_iterations != recipe.norm8_lloyd_iterations) {
      const auto preparation_begin = std::chrono::steady_clock::now();
      if (quantized_scaled) {
        auto parallel = stlqpp::OpenMpParallelFor::Create(command.effective_threads);
        if (!parallel.ok()) return parallel.status();
        if (auto status = PrepareOriginalNorms(command, *manifest, parallel.value());
            !status.ok()) return status;
      }
      auto derived = DerivedCacheDestination(
          command,
          std::string(quantized_scaled ? "norm8-scaled-" : "norm8-") + std::to_string(static_cast<std::uint32_t>(target_mode)) +
              "-" + std::to_string(target_iterations), quantized_scaled);
      if (!derived.ok()) return derived.status();
      std::error_code error;
      const bool cache_exists = std::filesystem::exists(derived.value(), error);
      if (error) return stlqpp::Status::Internal(
          "cannot inspect norm8 derivation cache: " + error.message());
      if (cache_exists) {
        auto cached = ReadIndexManifest(derived.value());
        if (!cached.ok()) return cached.status();
        if (!cached.value().producer_recipe.has_value() ||
            cached.value().producer_recipe->norm8_mode != target_mode ||
            cached.value().producer_recipe->norm8_lloyd_iterations !=
                target_iterations ||
            cached.value().hashes.linkage_list_directory !=
                manifest->hashes.linkage_list_directory ||
            cached.value().hashes.coefficient_codec !=
                manifest->hashes.coefficient_codec) {
          return stlqpp::Status::DataLoss(
              "norm8 derivation cache disagrees with its source index");
        }
        command.index = derived.value();
        *manifest = std::move(cached).value();
        if (command.norm_preparation_backend == "existing_index") {
          command.norm_preparation_backend = "cache_hit";
        }
        stlqpp::LogInfo("Eval norm8 derivation cache hit: " +
                        command.index.string() + "; no norm recomputation");
      } else {
        if (command.gpu_norm_enable && !command.use_cuda) {
          return stlqpp::Status::FailedPrecondition(
              "eval.linkage.gpu_norm_enable=true requires runtime.use_cuda=true "
              "for norm-only derivation");
        }
        const stlqpp::Status status = stlqpp::PublishReconstructionNorm8Runtime(
            command.index, derived.value(), target_mode,
            target_iterations, command.gpu_norm_enable, command.cuda_device,
            quantized_scaled, command.prepared_original_norms.get());
        if (!status.ok()) return status;
        command.norm_preparation_backend =
            command.gpu_norm_enable ? "CUDA" : "CPU";
        command.index = derived.value();
        auto updated = ReadIndexManifest(command.index);
        if (!updated.ok()) return updated.status();
        *manifest = std::move(updated).value();
      }
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - preparation_begin).count();
      stlqpp::LogInfo("Eval norm8 derived outside QPS: " +
                      command.index.string() + " seconds=" +
                      std::to_string(seconds));
    }
  }
  if ((!quantized_scaled || command.independent_quantized_norms) || manifest->has_scaled_score8()) {
    return stlqpp::Status::Ok();
  }
  auto parallel_result = stlqpp::OpenMpParallelFor::Create(
      command.effective_threads);
  if (!parallel_result.ok()) return parallel_result.status();
  stlqpp::OpenMpParallelFor parallel = std::move(parallel_result).value();
  if (const auto status = PrepareOriginalNorms(command, *manifest, parallel); !status.ok()) return status;
  const auto preparation_begin = std::chrono::steady_clock::now();
  if (quantized_scaled && !command.independent_quantized_norms && !manifest->has_scaled_score8()) {
    auto derived = DerivedCacheDestination(
        command, "quantized_scaled", true);
    if (!derived.ok()) return derived.status();
    auto hit = UseCachedDerivedIndex(
        command, manifest, derived.value(),
        stlqpp::EvalRepresentation::kQuantizedScaled);
    if (!hit.ok()) return hit.status();
    if (!hit.value()) {
      if (command.gpu_norm_enable && !command.use_cuda) {
        return stlqpp::Status::FailedPrecondition(
            "eval.linkage.gpu_norm_enable=true requires runtime.use_cuda=true "
            "for scaled-score derivation");
      }
      auto table = stlqpp::FitScaledScore8Runtime(
          command.index, command.prepared_original_norms.get(), parallel,
          command.gpu_norm_enable, command.cuda_device);
      if (!table.ok()) return table.status();
      command.norm_preparation_backend =
          command.gpu_norm_enable ? "CUDA" : "CPU";
      if (const stlqpp::Status status = stlqpp::PublishScaledScore8(
              command.index, derived.value(), table.value().view());
          !status.ok()) return status;
      command.index = derived.value();
      auto updated = ReadIndexManifest(command.index);
      if (!updated.ok()) return updated.status();
      *manifest = std::move(updated).value();
    }
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - preparation_begin).count();
  stlqpp::LogInfo("Eval derived representation prepared outside QPS: " +
                  command.index.string() + " seconds=" +
                  std::to_string(seconds));
  return stlqpp::Status::Ok();
}

stlqpp::Status PrepareActiveNorms(stlqpp::app::EvalCommand& command,
                                const stlqpp::IndexManifest& manifest) {
  using namespace stlqpp;
  using namespace stlqpp::app;
  const bool native = command.active_coefficient_mode == SearchCoefficientMode::kFloat;
  const bool exact = command.active_norm_mode == EvalNormMode::kFloat;
  const bool scaled = command.active_score_mode == SearchScoreMode::kScaled;
  std::error_code error;
  const bool stored_float = manifest.has_float_norm2() &&
      ShardExists(command.index, manifest, stlqpp::ShardKind::kFloatNorm2, error);
  if (error) return Status::Internal("cannot inspect float norm shard: " + error.message());
  if (command.active_score_mode == SearchScoreMode::kMipsRecon) {
    if (!native && !command.independent_quantized_norms) return Status::Ok();
    auto files = CanonicalIndexFiles::OpenCoefficients(command.index, native);
    if (!files.ok()) return files.status();
    IndexRowScores scores;
    scores.clusters.resize(manifest.shape.nlist);
    for (std::uint32_t cid = 0; cid < manifest.shape.nlist; ++cid)
      scores.clusters[cid].norm2.resize(files.value().entry(cid).real_count, 0.0F);
    command.prepared_row_scores = std::make_shared<IndexRowScores>(std::move(scores));
    command.norm_preparation_backend = "not_required_mips";
    return Status::Ok();
  }
  if (scaled) {
    auto threads = OpenMpParallelFor::Create(command.effective_threads);
    if (!threads.ok()) return threads.status();
    if (const auto status = PrepareOriginalNorms(command, manifest, threads.value()); !status.ok()) return status;
  }
  if ((native && exact && stored_float && !scaled) ||
      (!native && !exact && !command.independent_quantized_norms)) return Status::Ok();
  auto parallel_result = OpenMpParallelFor::Create(command.effective_threads);
  if (!parallel_result.ok()) return parallel_result.status();
  auto parallel = std::move(parallel_result).value();
  if (scaled) {
    if (const auto status = PrepareOriginalNorms(command, manifest, parallel); !status.ok()) return status;
  }
  const std::string source_identity = scaled ? command.prepared_original_norms->identity() : "";
  std::filesystem::path cache_root = command.run_root / "norm-cache";
  for (auto path = command.index; !path.empty(); path = path.parent_path()) {
    if (path.filename() == "linkage_list") { cache_root = path / "norm-cache"; break; }
    if (path == path.root_path()) break;
  }
  const auto mode = command.active_norm_mode == EvalNormMode::kLutReference
      ? ReconstructionNorm8Mode::kReferenceLloyd
      : command.active_norm_mode == EvalNormMode::kLutTailWeighted
      ? ReconstructionNorm8Mode::kTailWeightedLloyd : ReconstructionNorm8Mode::kIdentityLloyd;
  const auto iterations = command.expected_norm8_lloyd_iterations.value_or(
      manifest.producer_recipe && manifest.producer_recipe->norm8_lloyd_iterations != 0U
          ? manifest.producer_recipe->norm8_lloyd_iterations : 25U);
  auto prepared = PrepareEvalNormCache({
      .index = command.index, .cache_root = cache_root,
      .external_source_identity = source_identity,
      .native_float = native, .float_norm = exact, .scaled = scaled,
      .use_cuda = command.gpu_norm_enable, .cuda_compute_allowed = command.use_cuda,
      .cuda_device = command.cuda_device, .mode = mode, .iterations = iterations,
      .source = scaled ? command.prepared_original_norms.get() : nullptr, .parallel = &parallel});
  if (!prepared.ok()) return prepared.status();
  command.prepared_norm_payloads = prepared.value().query_payloads;
  command.active_score_codec_hash = prepared.value().hash;
  if (!exact) command.norm8_lloyd_iterations = iterations;
  command.norm8_mode = mode;
  command.norm_preparation_backend = prepared.value().cache_hit ? "cache_hit"
      : command.gpu_norm_enable
          ? (mode == ReconstructionNorm8Mode::kReferenceLloyd ? "CPU+CUDA" : "CUDA")
          : "CPU";
  LogInfo("Eval independent norm cache " + std::string(prepared.value().cache_hit ? "hit: " : "built: ") +
      prepared.value().directory.string() + " coeff=" + (native ? "float" : "quantized") +
      " norm=" + EvalNormModeName(command.active_norm_mode) +
      " hash=" + command.active_score_codec_hash.ToHex());
  command.prepared_row_scores = std::make_shared<IndexRowScores>(std::move(prepared.value().scores));
  return Status::Ok();
}

std::uint64_t IdFingerprint(const stlqpp::SearchResult& result) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint64_t id : result.ids) {
    for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
      hash ^= static_cast<std::uint8_t>(id >> shift);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

stlqpp::StatusOr<stlqpp::app::TexmexVectorShape> InspectQueries(
    const stlqpp::app::EvalCommand& command) {
  if (command.effective_query_format == "fvecs") {
    return stlqpp::app::InspectFvecs(command.effective_queries);
  }
  if (command.effective_query_format == "fbin") {
    return stlqpp::app::InspectFbin(command.effective_queries);
  }
  if (command.effective_query_format == "bvecs") {
    return stlqpp::app::InspectBvecs(command.effective_queries);
  }
  if (command.effective_query_format == "npy") {
    return stlqpp::app::InspectNpyFloat16(command.effective_queries);
  }
  return stlqpp::Status::InvalidArgument(
      "query reader route was not resolved");
}

stlqpp::StatusOr<stlqpp::app::TexmexVectorShape> InspectGroundtruth(
    const stlqpp::app::EvalCommand& command) {
  if (command.effective_groundtruth_format == "ivecs") {
    return stlqpp::app::InspectIvecs(command.effective_groundtruth);
  }
  if (command.effective_groundtruth_format == "ibin") {
    return stlqpp::app::InspectIbin(command.effective_groundtruth);
  }
  if (command.effective_groundtruth_format == "npy") {
    return stlqpp::app::InspectNpyInt64(command.effective_groundtruth);
  }
  return stlqpp::Status::InvalidArgument(
      "groundtruth reader route was not resolved");
}

stlqpp::StatusOr<stlqpp::app::TexmexFloatVectors> ReadQueries(
    const stlqpp::app::EvalCommand& command,
    std::uint32_t count) {
  if (command.effective_query_format == "fvecs") {
    return stlqpp::app::ReadFvecsPrefix(
        command.effective_queries, count);
  }
  if (command.effective_query_format == "fbin") {
    return stlqpp::app::ReadFbinPrefix(
        command.effective_queries, count);
  }
  if (command.effective_query_format == "bvecs") {
    auto source = stlqpp::app::ReadBvecsPrefix(
        command.effective_queries, count);
    if (!source.ok()) return source.status();
    stlqpp::app::TexmexFloatVectors output;
    output.count = source.value().count;
    output.dimension = source.value().dimension;
    output.values.resize(source.value().values.size());
    for (std::size_t index = 0U;
         index < source.value().values.size(); ++index) {
      output.values[index] =
          static_cast<float>(source.value().values[index]);
    }
    return output;
  }
  if (command.effective_query_format == "npy") {
    return stlqpp::app::ReadNpyFloat16Prefix(
        command.effective_queries, count);
  }
  return stlqpp::Status::InvalidArgument(
      "query reader route was not resolved");
}

stlqpp::StatusOr<stlqpp::app::TexmexU32Vectors> ReadGroundtruth(
    const stlqpp::app::EvalCommand& command,
    std::uint32_t count) {
  if (command.effective_groundtruth_format == "ivecs") {
    return stlqpp::app::ReadIvecsPrefix(
        command.effective_groundtruth, count);
  }
  if (command.effective_groundtruth_format == "ibin") {
    return stlqpp::app::ReadIbinPrefix(
        command.effective_groundtruth, count);
  }
  if (command.effective_groundtruth_format == "npy") {
    return stlqpp::app::ReadNpyInt64Prefix(
        command.effective_groundtruth, count);
  }
  return stlqpp::Status::InvalidArgument(
      "groundtruth reader route was not resolved");
}

stlqpp::StatusOr<std::string> QueryDiskReport(
    const stlqpp::app::EvalCommand& command,
    const std::filesystem::path& routing_index) {
  using namespace stlqpp;
  auto decoded = ReadIndexManifest(command.index);
  if (!decoded.ok()) return decoded.status();
  const auto& manifest = decoded.value();
  auto collected = CollectQueryDiskFootprint(command.index, manifest,
      command.active_coefficient_mode == SearchCoefficientMode::kFloat,
      command.active_score_mode == SearchScoreMode::kScaled,
      command.active_score_mode == SearchScoreMode::kMipsRecon,
      command.prepared_norm_payloads, routing_index);
  if (!collected.ok()) return collected.status();
  return collected.value().Summary(command.label,
      stlqpp::app::EvalNormModeName(command.active_norm_mode), command.ivf_probe_mode,
      manifest.counts.real, manifest.counts.virtual_roots);
}

template <typename OpenSearcher>
stlqpp::StatusOr<std::filesystem::path> Run(
    const stlqpp::app::EvalCommand& command,
    OpenSearcher&& open_searcher) {
  auto query_shape = InspectQueries(command);
  if (!query_shape.ok()) return query_shape.status();
  auto groundtruth_shape = InspectGroundtruth(command);
  if (!groundtruth_shape.ok()) return groundtruth_shape.status();
  std::uint32_t selected_query_count = query_shape.value().count;
  if (command.nquery != 0U) {
    selected_query_count = std::min(
        selected_query_count, command.nquery);
  }
  if (groundtruth_shape.value().count < selected_query_count) {
    stlqpp::LogWarn(
        "Ground truth count (" +
        std::to_string(groundtruth_shape.value().count) +
        ") is smaller than selected query vector count (" +
        std::to_string(selected_query_count) +
        "); shrinking queries to match.");
  }
  const std::uint32_t query_count = std::min(
      selected_query_count, groundtruth_shape.value().count);
  if (query_count == 0U || groundtruth_shape.value().dimension == 0U) {
    return stlqpp::Status::InvalidArgument(
        "query and groundtruth files have no comparable rows");
  }
  auto queries = ReadQueries(command, query_count);
  if (!queries.ok()) return queries.status();
  auto groundtruth = ReadGroundtruth(command, query_count);
  if (!groundtruth.ok()) return groundtruth.status();
  if (queries.value().count != query_count ||
      groundtruth.value().count != query_count) {
    return stlqpp::Status::DataLoss(
        "query or groundtruth prefix changed after metadata inspection");
  }
  if (command.groundtruth_add1) {
    for (std::uint32_t& id : groundtruth.value().values) ++id;
  }
  auto plan = stlqpp::DenseCpuPlan::Detect();
  if (!plan.ok()) return plan.status();
  auto parallel_result = stlqpp::OpenMpParallelFor::Create(
      command.effective_threads);
  if (!parallel_result.ok()) return parallel_result.status();
  auto gemm = std::make_shared<stlqpp::CblasF32Gemm>(
      command.effective_threads);
  auto parallel = std::make_shared<stlqpp::OpenMpParallelFor>(
      std::move(parallel_result).value());
  auto searcher = open_searcher(gemm, parallel, plan.value());
  if (!searcher.ok()) return searcher.status();
  if (command.minimal_query_disk_report && command.minimal_query_disk_report->empty()) {
    auto report = QueryDiskReport(command, searcher.value()->routing_index_path());
    if (!report.ok()) return report.status();
    *command.minimal_query_disk_report = std::move(report).value();
    stlqpp::LogInfo(*command.minimal_query_disk_report);
  }
  const stlqpp::QueryBatchView query_view = {
      .data = queries.value().values.data(),
      .count = queries.value().count,
      .dimension = queries.value().dimension,
      .stride_floats = queries.value().dimension,
  };
  auto trusted_queries = searcher.value()->ValidateQueries(query_view);
  if (!trusted_queries.ok()) return trusted_queries.status();
  std::unique_ptr<stlqpp::ExactL2Reranker> reranker;
  if (command.rerank_candidates != 0U) {
    auto manifest = ReadIndexManifest(command.index);
    if (!manifest.ok()) return manifest.status();
    if (command.base_source_row_count != 0U &&
        command.base_source_row_count != manifest.value().counts.real)
      return stlqpp::Status::InvalidArgument("rerank source row count differs from index real count");
    auto opened = stlqpp::ExactL2Reranker::Open({
        .path = command.effective_base_source,
        .format = command.base_source_format,
        .storage = command.rerank_storage,
        .dimension = query_view.dimension,
        .first_row = command.base_source_first_row,
        .row_count = manifest.value().counts.real}, parallel->worker_count());
    if (!opened.ok()) return opened.status();
    reranker = std::move(opened).value();
  }
  const auto execute = [&](const stlqpp::SearchOptions& options,
                           stlqpp::SearchResult* result,
                           stlqpp::ExactDenseSearchStatistics* statistics) {
    if (reranker) {
      const auto begin = std::chrono::steady_clock::now();
      const auto status = reranker->SearchTrusted(*searcher.value(), trusted_queries.value(),
          {.nprobe = options.nprobe, .top_k = command.rerank_candidates}, options.top_k, result);
      if (statistics) {
        *statistics = {};
        statistics->wall_seconds = statistics->core_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      }
      return status;
    }
    return statistics ? searcher.value()->SearchProfiledTrusted(
        trusted_queries.value(), options, result, statistics)
        : searcher.value()->SearchTrusted(trusted_queries.value(), options, result);
  };
  // Once per active representation, outside all query/QPS timing. Inspect CID
  // coefficient extents and headers without decoding any vector or symbols.
  std::optional<stlqpp::LinkageBitBudget> bit_budget;
  {
    const bool native_float = command.active_coefficient_mode ==
        stlqpp::SearchCoefficientMode::kFloat;
    auto files = stlqpp::CanonicalIndexFiles::OpenCoefficients(
        command.index, native_float);
    if (!files.ok()) return files.status();
    auto budget = stlqpp::ReadEvalLinkageBitBudget(files.value(), native_float);
    if (!budget.ok()) return budget.status();
    bit_budget.emplace(budget.value());
    stlqpp::LogInfo(stlqpp::app::FormatEvalBitBudget(command, *bit_budget));
  }
  std::filesystem::path last_archive = command.run_root;
  std::vector<stlqpp::app::EvalProbeSummary> probe_summaries;
  probe_summaries.reserve(command.nprobes.size());
  for (const std::uint32_t nprobe : command.nprobes) {
    if (command.nprobes.size() > 1U) {
      stlqpp::LogInfo(
          "===== Linkage disk eval nprobe=" + std::to_string(nprobe) +
          " =====");
    }
    stlqpp::LogInfo(
        "========= Evaluating linkage recall (disk IVF, " +
        command.display_label + (reranker ? ", exact L2 rerank" : "") + ") =========");
    stlqpp::LogInfo(
        "Linkage recall mode: disk-ivf (nprobe=" +
        std::to_string(nprobe) + ") norm2_mode=" +
        stlqpp::app::EvalNormModeName(command.active_norm_mode) +
        " score_codec=" +
        stlqpp::app::EvalScoreCodecName(command.active_score_codec) +
        " score_mode=" +
        (command.active_score_mode == stlqpp::SearchScoreMode::kCosineRecon
             ? "cosine_recon"
             : command.active_score_mode == stlqpp::SearchScoreMode::kMipsRecon
                   ? "mips_recon" : "l2") +
        " probe_mode=" + command.ivf_probe_mode +
        (command.ivf_probe_mode == "hnsw" ? " ef_search=" + std::to_string(command.active_ivf_hnsw_ef_search) : "") +
        " query_tables=" +
        (command.pretransform_query_codebooks
             ? "source-pretransformed" : "model-query-rotation"));
    const stlqpp::SearchOptions options = {
        .nprobe = nprobe, .top_k = command.top_k};
    for (std::uint32_t warmup = 0U; warmup < command.warmup; ++warmup) {
      stlqpp::SearchResult discarded;
      if (const stlqpp::Status status =
              execute(options, &discarded, nullptr);
          !status.ok()) return status;
    }
    std::vector<double> core_seconds;
    core_seconds.reserve(command.repeat);
    stlqpp::SearchResult accepted;
    stlqpp::ExactDenseSearchStatistics accepted_statistics;
    std::uint64_t fingerprint = 0U;
    for (std::uint32_t repeat = 0U; repeat < command.repeat; ++repeat) {
      stlqpp::LogInfo("--- " + std::to_string(repeat + 1U) + "/" +
                      std::to_string(command.repeat) + " ---");
      stlqpp::SearchResult result;
      stlqpp::ExactDenseSearchStatistics statistics;
      if (const stlqpp::Status status =
              execute(options, &result, &statistics);
          !status.ok()) return status;
      const std::uint64_t current = IdFingerprint(result);
      if (repeat != 0U && current != fingerprint) {
        return stlqpp::Status::FailedPrecondition(
            "exact-dense output changed between repeats");
      }
      fingerprint = current;
      accepted = std::move(result);
      accepted_statistics = statistics;
      core_seconds.push_back(statistics.core_seconds);
      const stlqpp::app::EvalArchiveTiming repeat_timing = {
          .detailed_profile = statistics.detailed_profile,
          .core_seconds = statistics.core_seconds,
          .qt_rotate_seconds = statistics.qt_rotate_seconds,
          .table_build_seconds = statistics.table_build_seconds,
          .probe_selection_seconds = statistics.probe_selection_seconds,
          .scan_topk_seconds = statistics.scan_topk_seconds,
          .route_scan_topk_seconds = statistics.probe_selection_seconds +
              statistics.scan_topk_seconds,
          .route_scan_phase_seconds =
              statistics.route_scan_phase_seconds,
          .finalize_phase_seconds = statistics.finalize_phase_seconds,
          .route_worker_seconds = statistics.route_worker_seconds,
          .scan_worker_seconds = statistics.scan_worker_seconds,
          .online_topk_worker_seconds =
              statistics.online_topk_worker_seconds,
          .finalize_worker_seconds = statistics.finalize_worker_seconds,
          .logical_route_seconds = statistics.logical_route_seconds,
          .logical_scan_seconds = statistics.logical_scan_seconds,
          .logical_topk_seconds = statistics.logical_topk_seconds,
      };
      for (const std::string& repeat_line :
           stlqpp::app::FormatEvalRepeatLines(
               command, nprobe, repeat, repeat_timing, query_view.count)) {
        stlqpp::LogInfo(repeat_line);
      }
    }
    stlqpp::EvalMetricBundle metrics;
    if (!stlqpp::ComputeEvalMetricBundle(
            accepted, groundtruth.value().values,
            groundtruth.value().dimension, command.metric_mode,
            command.metric_gt_topks, &metrics)) {
      return stlqpp::Status::InvalidArgument(
          "groundtruth does not support requested eval metrics");
    }
    for (const std::string& line :
         stlqpp::BuildEvalMetricReportLines(metrics)) {
      stlqpp::LogInfo(line);
    }
    const stlqpp::app::EvalTimingSummary timing =
        stlqpp::app::ComputeEvalConsoleTimingSummary(
            core_seconds, query_view.count);
    if (command.archive) {
      const stlqpp::app::EvalArchiveTiming archive_timing = {
          .detailed_profile = accepted_statistics.detailed_profile,
          .core_seconds = accepted_statistics.core_seconds,
          .qt_rotate_seconds = accepted_statistics.qt_rotate_seconds,
          .table_build_seconds = accepted_statistics.table_build_seconds,
          .probe_selection_seconds = accepted_statistics.probe_selection_seconds,
          .scan_topk_seconds = accepted_statistics.scan_topk_seconds,
          .route_scan_topk_seconds = accepted_statistics.probe_selection_seconds +
              accepted_statistics.scan_topk_seconds,
          .route_scan_phase_seconds =
              accepted_statistics.route_scan_phase_seconds,
          .finalize_phase_seconds =
              accepted_statistics.finalize_phase_seconds,
          .route_worker_seconds =
              accepted_statistics.route_worker_seconds,
          .scan_worker_seconds = accepted_statistics.scan_worker_seconds,
          .online_topk_worker_seconds =
              accepted_statistics.online_topk_worker_seconds,
          .finalize_worker_seconds =
              accepted_statistics.finalize_worker_seconds,
          .logical_route_seconds =
              accepted_statistics.logical_route_seconds,
          .logical_scan_seconds =
              accepted_statistics.logical_scan_seconds,
          .logical_topk_seconds =
              accepted_statistics.logical_topk_seconds,
      };
      auto archive = stlqpp::app::ArchiveEvalResult(
          command, nprobe, metrics, core_seconds, &archive_timing,
          &*bit_budget);
      if (!archive.ok()) {
        stlqpp::LogWarn(archive.status().message());
      } else {
        last_archive = std::move(archive).value();
      }
    }
    stlqpp::LogInfo(stlqpp::app::FormatEvalSummaryLine(
        command, nprobe, query_view.count, timing));
    probe_summaries.push_back({
        .nprobe = nprobe,
        .metrics = metrics,
        .median_core_seconds = timing.median_core_seconds,
        .median_qps_core = timing.median_qps_core,
        .average_core_seconds = timing.average_core_seconds,
        .average_qps_core = timing.average_qps_core,
    });
  }
  if (probe_summaries.size() > 1U) {
    stlqpp::LogInfo(
        "===== Linkage disk recall batch summary (" + command.label +
        ") =====");
    for (const auto& summary : probe_summaries) {
      stlqpp::LogInfo(
          stlqpp::app::FormatEvalProbeBatchSummaryLine(summary));
    }
    const std::vector<std::size_t> pareto =
        stlqpp::app::SelectEvalParetoByRecallAt1(probe_summaries);
    if (!pareto.empty()) {
      stlqpp::LogInfo(
          "===== Linkage disk recall Pareto summary (" + command.label +
          ", by r@1) =====");
      for (const std::size_t index : pareto) {
        stlqpp::LogInfo(stlqpp::app::FormatEvalProbeBatchSummaryLine(
            probe_summaries[index]));
      }
    }
  }
  return last_archive;
}

template <int M>
stlqpp::StatusOr<std::filesystem::path> Run(
    const stlqpp::app::EvalCommand& command) {
  return Run(command, [&](const auto& gemm, const auto& parallel,
                          const auto& plan) {
    return stlqpp::OpenExactDenseSearcher<M>(
        command.index, gemm, parallel, plan,
        {.query_block_size = command.query_block,
         .pretransform_query_codebooks =
             command.pretransform_query_codebooks,
         .profile_timing = command.profile_timing,
         .ivf_hnsw = {.enabled = command.ivf_probe_mode == "hnsw",
                      .m = command.ivf_hnsw_m,
                      .ef_construction = command.ivf_hnsw_ef_construction,
                      .ef_search = command.active_ivf_hnsw_ef_search,
                      .cache_root = command.index.parent_path() / "ivf_hnsw"}},
        command.active_coefficient_mode,
        command.parent_louds_native_eval
            ? stlqpp::ParentAccessMode::kNativeLouds
            : stlqpp::ParentAccessMode::kMaterialized,
        command.active_score_mode, command.prepared_row_scores.get());
  });
}

stlqpp::StatusOr<std::filesystem::path> RunRuntime(
    const stlqpp::app::EvalCommand& command) {
  return Run(command, [&](const auto& gemm, const auto& parallel, const auto&) {
    return stlqpp::RuntimeExactDenseSearcher::Open(
        command.index, command.parent_louds_native_eval
            ? stlqpp::ParentAccessMode::kNativeLouds
            : stlqpp::ParentAccessMode::kMaterialized,
        command.active_coefficient_mode, command.active_score_mode,
        gemm, parallel,
        {.query_block_size = command.query_block,
         .pretransform_query_codebooks = command.pretransform_query_codebooks,
         .profile_timing = command.profile_timing,
         .ivf_hnsw = {.enabled = command.ivf_probe_mode == "hnsw",
                      .m = command.ivf_hnsw_m,
                      .ef_construction = command.ivf_hnsw_ef_construction,
                      .ef_search = command.active_ivf_hnsw_ef_search,
                      .cache_root = command.index.parent_path() / "ivf_hnsw"}}, command.prepared_row_scores.get());
  });
}

}  // namespace

namespace stlqpp::app {

Status RunExactDenseEval(
    EvalCommand command, char** argv, bool use_build_session) {
  bool report_quantized_budget = false;
  // Empty settings select the production scan policy, including old cfgs.
  if (command.omp_places.empty()) command.omp_places = "cores";
  if (command.omp_proc_bind.empty()) command.omp_proc_bind = "spread";
  auto places = ParseOpenMpPlaces(command.omp_places);
  auto proc_bind = ParseOpenMpProcBind(command.omp_proc_bind);
  if (!places.ok()) return places.status();
  if (!proc_bind.ok()) return proc_bind.status();
  Status affinity = PrepareOpenMpAffinityProcess({
      .places = places.value(),
      .proc_bind = proc_bind.value(),
      .override_environment = true,
  }, argv);
  if (!affinity.ok()) return affinity;
  command.effective_threads = command.threads > 0
      ? static_cast<std::uint32_t>(command.threads)
      : OpenMpDefaultWorkerCount();
  std::optional<EvalRunSession> session;
  if (!use_build_session) {
    auto started = EvalRunSession::Start(command);
    if (!started.ok()) return started.status();
    session.emplace(std::move(started).value());
  }
  const auto fail_preparation = [&](const Status& failure) {
    LogWarn("Eval preparation failed: " + failure.message());
    if (session.has_value()) {
      if (const Status state = session->Fail(failure); !state.ok()) {
        LogWarn(state.message());
      }
    }
    return failure;
  };
  std::optional<IndexManifest> manifest;
  if (command.linkage_enabled || command.train_checkpoint_file.empty()) {
    auto opened = ReadIndexManifest(command.index);
    if (!opened.ok()) return fail_preparation(opened.status());
    manifest.emplace(std::move(opened).value());
  }
  if (manifest.has_value()) {
    if (const Status status = EnsureRequestedRepresentations(
            command, &*manifest);
        !status.ok()) return fail_preparation(status);
    if (const Status status = BindEvalCommandToIndex(
            command, *manifest);
        !status.ok()) return fail_preparation(status);
    std::error_code source_norm_error;
    if (command.requested_norm_mode == EvalNormMode::kFloat &&
        manifest->has_float_source_norm2() &&
        ShardExists(command.index, *manifest, ShardKind::kFloatSourceNorm2, source_norm_error)) {
      if (const Status status = BindFloatScaledContentIdentity(command, *manifest);
          !status.ok()) return fail_preparation(status);
    }
    const bool native_scaled_requested = std::find(command.representation_runs.begin(), command.representation_runs.end(),
        EvalRepresentation::kFloatScaled) != command.representation_runs.end();
    const bool any_scaled_requested = native_scaled_requested || std::find(command.representation_runs.begin(), command.representation_runs.end(),
        EvalRepresentation::kQuantizedScaled) != command.representation_runs.end();
    if (any_scaled_requested) {
      auto parallel = OpenMpParallelFor::Create(command.effective_threads);
      if (!parallel.ok()) return fail_preparation(parallel.status());
      if (const auto status = PrepareOriginalNorms(command, *manifest, parallel.value()); !status.ok()) return fail_preparation(status);
    }
    report_quantized_budget = std::any_of(
        command.representation_runs.begin(),
        command.representation_runs.end(),
        [](EvalRepresentation representation) {
          return representation == EvalRepresentation::kQuantized ||
                 representation == EvalRepresentation::kQuantizedScaled;
        });
  }
  std::filesystem::path last_archive = command.run_root;
  if (command.base_enabled) {
    LogInfo("scope=minimal_query_data representation=basic status=unsupported");
    if (const Status status = RunBaseBasicEval(command); !status.ok()) {
      if (session.has_value()) {
        if (const Status state = session->Fail(status); !state.ok()) {
          LogWarn(state.message());
        }
      }
      return status;
    }
  }
  if (!command.linkage_enabled) {
    return session.has_value()
        ? session->Complete(last_archive) : Status::Ok();
  }
  if (report_quantized_budget) {
    auto files = CanonicalIndexFiles::OpenCoefficients(command.index, false);
    if (!files.ok()) {
      if (session.has_value()) {
        if (const Status state = session->Fail(files.status());
            !state.ok()) LogWarn(state.message());
      }
      return files.status();
    }
    if (const Status status = LogEvalLinkageBitBudget(files.value());
        !status.ok()) {
      if (session.has_value()) {
        if (const Status state = session->Fail(status); !state.ok()) {
          LogWarn(state.message());
        }
      }
      return status;
    }
  }
  for (const EvalRepresentation representation :
       command.representation_runs) {
    EvalCommand active = command;
    active.minimal_query_disk_report = std::make_shared<std::string>();
    if (const Status status = SelectEvalRepresentation(
            active, representation);
        !status.ok()) {
      if (session.has_value()) {
        if (const Status state = session->Fail(status); !state.ok()) {
          LogWarn(state.message());
        }
      }
      return status;
    }
    if (representation == EvalRepresentation::kFloat) {
      const bool cosine = active.active_score_mode ==
          stlqpp::SearchScoreMode::kCosineRecon;
      const bool mips = active.active_score_mode ==
          stlqpp::SearchScoreMode::kMipsRecon;
      active.label = cosine ? "float_cosine" : mips ? "float_mips" : "float";
      active.display_label = cosine ? "float coeffs cosine_recon"
                             : mips ? "float coeffs mips_recon"
                                    : "float coeffs";
    } else if (representation == EvalRepresentation::kFloatScaled) {
      active.label = "float_scaled";
      active.display_label = active.active_norm_mode == EvalNormMode::kFloat
          ? "float coeffs exact radial scale" : "float coeffs scaled score8";
    } else if (representation == EvalRepresentation::kQuantized) {
      const bool cosine = active.active_score_mode ==
          stlqpp::SearchScoreMode::kCosineRecon;
      const bool mips = active.active_score_mode ==
          stlqpp::SearchScoreMode::kMipsRecon;
      active.label = cosine ? "quantized_cosine"
                           : mips ? "quantized_mips" : "quantized";
      active.display_label = cosine ? "quantized cosine_recon"
                             : mips ? "quantized mips_recon"
                                    : active.active_norm_mode == EvalNormMode::kFloat
                                        ? "quantized coeffs float norm" : "quantized classic norm8";
    } else {
      active.label = "quantized_scaled";
      active.display_label = active.active_norm_mode == EvalNormMode::kFloat
          ? "quantized coeffs exact radial scale" : "quantized scaled score8";
    }
    if (const Status status = PrepareActiveNorms(active, *manifest); !status.ok())
      return fail_preparation(status);
    const auto ef_values = active.ivf_probe_mode == "hnsw"
        ? active.ivf_hnsw_ef_search : std::vector<std::uint32_t>{32U};
    for (const auto ef : ef_values) {
    active.active_ivf_hnsw_ef_search = ef;
    auto result = (active.m == 5U || active.m == 6U ||
                   active.m == 10U || active.m == 12U)
        ? stlqpp::DispatchModelM(
              active.m, [&]<std::uint32_t M>() { return Run<M>(active); })
        : RunRuntime(active);
    if (!result.ok()) {
      if (session.has_value()) {
        if (const Status state = session->Fail(result.status());
            !state.ok()) LogWarn(state.message());
      }
      return result.status();
    }
    last_archive = std::move(result).value();
    }
  }
  if (session.has_value()) {
    if (const Status state = session->Complete(last_archive);
        !state.ok()) return state;
  }
  return Status::Ok();
}

}  // namespace stlqpp::app
