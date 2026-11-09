#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/eval/eval_metrics.h"
#include "stlqpp/search/coefficient_mode.h"
#include "stlqpp/search/eval_representation.h"
#include "stlqpp/search/score_mode.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <memory>
#include "stlqpp/search/row_score_data.h"
#include "stlqpp/application/source_norm_provider.h"
#include <string>
#include <vector>

namespace stlqpp::app {

// The public DEV norm selector is an eval-time representation choice.  Keep
// it separate from ReconstructionNorm8Mode, which describes how the
// compressed norm shard stored in the index was produced.
enum class EvalNormMode : std::uint8_t {
  kFloat = 0U,
  kLut,
  kLutTailWeighted,
  kLutReference,
};

enum class EvalScoreCodec : std::uint8_t {
  kNativeFloat = 0U,
  kFloatScaled,
  kReconstructionNorm8,
  kScaledScore8,
  kReconstructionNormF32,
  kRadialScoreF32,
};

struct FrozenEvalConfigInput final {
  std::filesystem::path path;
  std::string bytes;
};

struct EvalCommand final {
  std::uint32_t m = 0U;
  std::filesystem::path index;
  std::filesystem::path train_checkpoint_file;
  // Basic eval consumes the durable list-major product in the run's output
  // directory, or the explicitly selected io.base_file.
  std::filesystem::path work_directory;
  std::filesystem::path requested_base_list_directory;
  std::filesystem::path base_list_directory;
  std::filesystem::path dataset_data_root = "../data";
  std::filesystem::path base_source;
  std::filesystem::path effective_base_source;
  std::string base_source_format = "auto";
  std::uint64_t base_source_first_row = 0U;
  std::uint64_t base_source_row_count = 0U;
  // Requested paths are archived verbatim; readers consume the resolved
  // paths so relative overrides retain DEV's data-root semantics.
  std::filesystem::path queries;
  std::filesystem::path effective_queries;
  std::filesystem::path groundtruth;
  std::filesystem::path effective_groundtruth;
  // Zero evaluates every query in the input, matching DEV.
  std::uint32_t nquery = 0U;
  std::uint32_t top_k = 100U;
  bool groundtruth_add1 = false;
  std::string query_format = "auto";
  std::string groundtruth_format = "auto";
  // Resolved reader routes. Requested public values above remain unchanged
  // for round-trip snapshots.
  std::string effective_query_format;
  std::string effective_groundtruth_format;
  std::vector<std::uint32_t> nprobes;
  std::string ivf_probe_mode = "exact";
  std::uint32_t ivf_hnsw_m = 32U;
  std::uint32_t ivf_hnsw_ef_construction = 200U;
  std::vector<std::uint32_t> ivf_hnsw_ef_search = {32U};
  std::uint32_t active_ivf_hnsw_ef_search = 32U;
  bool base_enabled = false;
  bool linkage_enabled = true;
  bool base_use_ivf = true;
  std::uint32_t base_nprobe = 8U;
  std::uint32_t base_warmup = 0U;
  std::uint32_t base_repeat = 1U;
  std::int32_t threads = 0;
  bool use_cuda = true;
  bool gpu_norm_enable = true;
  // Runtime outcome, not part of the artifact recipe or requested cfg.
  std::string norm_preparation_backend = "existing_index";
  int cuda_device = 0;
  // Zero keeps the DEV/OpenMP environment selection. The actual positive
  // worker count is resolved after affinity setup.
  std::uint32_t effective_threads = 0U;
  std::uint32_t warmup = 0U;
  std::uint32_t repeat = 1U;
  std::uint32_t query_block = 1000U;
  // false is the DEV baseline: materialize an adaptive u16/u32 parent cache
  // while opening the canonical ParentLOUDS index. true decodes LOUDS in the
  // query kernel.
  bool parent_louds_native_eval = false;
  // Build R^T C once when the searcher opens, then build query tables from
  // source-space queries without a per-request query rotation.
  bool pretransform_query_codebooks = true;
  // Detailed logical Scan/TopK timing uses a separately compiled int-only
  // diagnostic path. Ordinary eval remains the production fused path.
  bool profile_timing = false;
  // Zero keeps ordinary ADC. Otherwise retrieve this many ADC candidates,
  // then immediately return exact L2 top_k from the registered original base.
  std::uint32_t rerank_candidates = 0U;
  std::string rerank_storage = "memory";
  bool requested_representations_auto = false;
  std::vector<EvalRepresentation> requested_representations = {
      EvalRepresentation::kQuantizedScaled};
  std::vector<EvalRepresentation> representation_runs;
  EvalRepresentation active_representation =
      EvalRepresentation::kQuantizedScaled;
  SearchCoefficientMode active_coefficient_mode =
      SearchCoefficientMode::kInt8;
  SearchScoreMode active_score_mode = SearchScoreMode::kScaled;
  SearchScoreMode requested_score_mode = SearchScoreMode::kClassic;
  std::optional<EvalNormMode> requested_norm_mode;
  bool independent_quantized_norms = false;
  std::shared_ptr<const IndexRowScores> prepared_row_scores;
  std::vector<std::filesystem::path> prepared_norm_payloads;
  // Shared across ef/nprobe repetitions of one prepared representation.
  std::shared_ptr<std::string> minimal_query_disk_report;
  std::shared_ptr<const SourceNormProvider> prepared_original_norms;
  EvalNormMode active_norm_mode = EvalNormMode::kLut;
  EvalScoreCodec active_score_codec = EvalScoreCodec::kScaledScore8;
  ArtifactHash64 active_score_codec_hash;
  ArtifactHash64 classic_score_codec_hash;
  ArtifactHash64 scaled_score_codec_hash;
  ArtifactHash64 float_scaled_score_codec_hash;
  // Effective producer provenance. Parse-time values are assertions only;
  // these fields are populated from the identity-checked index before eval.
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  std::vector<std::uint32_t> coefficient_bits_per_layer;
  std::vector<std::uint32_t> coefficient_bits_per_layer_root;
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage;
  bool coefficient_weighted_quantile = true;
  bool coefficient_allow_clipping = true;
  bool coefficient_fit_scale = true;
  std::uint32_t coefficient_refinement_sweeps = 3U;
  std::uint32_t coefficient_refinement_max_layer = 0U;
  std::uint32_t coefficient_refinement_step_limit = 1U;
  std::optional<std::vector<std::uint32_t>>
      expected_coefficient_bits_per_layer;
  std::optional<std::vector<std::uint32_t>>
      expected_coefficient_bits_per_layer_root;
  std::optional<std::vector<std::uint32_t>>
      expected_coefficient_bits_per_layer_linkage;
  std::optional<bool> expected_coefficient_weighted_quantile;
  std::optional<bool> expected_coefficient_allow_clipping;
  std::optional<bool> expected_coefficient_fit_scale;
  std::optional<std::uint32_t> expected_coefficient_refinement_sweeps;
  std::optional<std::uint32_t> expected_coefficient_refinement_max_layer;
  std::optional<std::uint32_t> expected_coefficient_refinement_step_limit;
  std::vector<double> coefficient_first_quantile_percents;
  std::vector<double> coefficient_rest_quantile_percents;
  std::optional<std::uint32_t> expected_norm8_lloyd_iterations;
  std::optional<ReconstructionNorm8Mode> expected_norm8_mode;
  std::vector<double> expected_coefficient_first_quantile_percents;
  std::vector<double> expected_coefficient_rest_quantile_percents;
  bool producer_recipe_verified = false;
  std::string omp_places;
  std::string omp_proc_bind;
  bool quiet = false;
  EvalMetricMode metric_mode = EvalMetricMode::kTop1Recall;
  std::vector<std::uint32_t> metric_gt_topks = {1U, 10U, 50U, 100U};
  std::filesystem::path run_root;
  std::string label = "quantized_scaled";
  std::string display_label = "quantized scaled score8";
  bool archive = true;
  bool archive_log = true;
  std::vector<std::filesystem::path> config_files;
  std::vector<FrozenEvalConfigInput> frozen_config_inputs;
};

[[nodiscard]] StatusOr<EvalCommand> ParseEvalCommand(
    int argc, char** argv, bool allow_deferred_build_sources = false);
[[nodiscard]] Status BindEvalCommandToIndex(
    EvalCommand& command, const IndexManifest& manifest);
[[nodiscard]] const char* EvalNormModeName(EvalNormMode mode) noexcept;
[[nodiscard]] const char* EvalScoreCodecName(EvalScoreCodec codec) noexcept;
[[nodiscard]] Status SelectEvalRepresentation(
    EvalCommand& command, EvalRepresentation representation);
void WriteRoundTripEvalConfig(const EvalCommand& command,
                              std::ostream& output,
                              bool include_eval_only = true);

}  // namespace stlqpp::app
