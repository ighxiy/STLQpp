#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/search/eval_representation.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace stlqpp::app {

struct TopologyCommand final {
  std::uint32_t radial_layer_count = 0U;
  double root_fraction = 0.0;
  std::uint32_t depth_k = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
  std::uint32_t seed = 0U;
  std::uint32_t rsd_rank = 2U;
  std::uint32_t icm_rounds = 1U;
  std::string bad_cluster_knn_backend = "cpu_exact";
  std::uint32_t bad_cluster_cpu_exact_max_points = 150000U;
};

struct VirtualRootCommand final {
  bool enabled = true;
  double ratio = 0.0;
  std::uint32_t minimum = 0U;
  std::uint32_t maximum = 0U;
  std::uint32_t neighbor_count = 0U;
  std::uint32_t local_connectivity = 0U;
  float overlap_threshold = 0.0F;
  bool prefer_density_peaks = true;
  std::uint32_t anchor_neighbor_k = 0U;
  bool use_fixed_per_cluster = false;
  std::uint32_t fixed_per_cluster = 256U;
};

struct RootInitializationCommand final {
  std::string seed_method = "kmeansll";
  std::int32_t maximum_iterations = 0;
  double tolerance = 0.0;
  std::int32_t kmeans_parallel_rounds = 0;
  double kmeans_parallel_oversample = 0.0;
  std::uint64_t kmeans_parallel_seed = 0U;
  std::int32_t candidate_cap = 0;
  double initial_weight = 0.0;
  double minimum_weight = 0.0;
  double outlier_quantile = 0.0;
  double cost_threshold = 0.0;
  double annealing_factor = 0.0;
  std::int32_t warmup_iterations = 0;
  std::int32_t quantile_bins = 0;
  std::int32_t large_layer_threshold = 0;
  std::int32_t large_layer_top_coarse = 0;
  bool large_layer_hierarchy_enabled = true;
  std::int32_t large_layer_fixed_coarse_count = 0;
  std::int32_t large_layer_minimum_coarse_count = 128;
  std::int32_t large_layer_maximum_coarse_count = 1'024;
  bool large_layer_round_coarse_to_power_of_two = true;
};

struct BuildConfigAssignment final {
  std::string key;
  std::string value;
  std::string source;
};

struct FrozenBuildConfigInput final {
  std::filesystem::path path;
  std::string bytes;
};

enum class SourceFormat : std::uint8_t {
  kAuto = 0,
  kBvecsU8 = 1,
  kFvecsF32 = 2,
  kFbinF32 = 3,
  kNpyFloat16 = 4,
};

enum class CudaMode : std::uint8_t {
  kStrict = 0,
  kFast = 1,
};

enum class CleanupPreset : std::uint8_t {
  kNone = 0U,
  kDevAll,
  kEvalBoth,
  kEvalFloatMin,
  kEvalInt8Min,
  kCustom,
};

struct CleanupCommand final {
  bool enabled = false;
  CleanupPreset preset = CleanupPreset::kEvalBoth;
  std::int32_t keep_base_basic = -1;
  std::int32_t keep_base_basic_cluster_id = -1;
  std::int32_t keep_base_basic_ivf_lists = -1;
  std::int32_t keep_base_list = -1;
  std::int32_t keep_base_list_raw = -1;
  std::int32_t keep_linkage_list_f32 = -1;
  std::int32_t keep_linkage_parent_u32 = -1;
};

struct ResolvedCleanupPolicy final {
  bool keep_base_basic = true;
  bool keep_base_basic_cluster_id = true;
  bool keep_base_basic_ivf_lists = true;
  bool keep_base_list = true;
  bool keep_base_list_raw = true;
  bool keep_linkage_list_f32 = true;
  bool keep_linkage_parent_u32 = true;
};

struct BuildCommand final {
  // Stage controls retain the DEV distinction between producing a stage and
  // reopening its durable input. advanced.eval_only is resolved after all
  // argv-order assignments and disables all four build stages.
  bool train_enabled = true;
  bool train_initialization_enabled = true;
  bool base_encode_enabled = true;
  bool base_linkage_enabled = true;
  bool eval_only = false;
  // A standalone evaluation config need not contain inactive build inputs.
  bool eval_only_sparse_config = false;
  bool train_checkpoint_enabled = false;
  std::uint32_t train_checkpoint_every_r = 0U;
  // io.train_file is an exact input only. Fresh training does not read it;
  // resumed training and base-only builds read it but publish new models
  // under this run's output directory and prefix.
  std::filesystem::path train_checkpoint_file;
  std::string output_prefix = "test";
  // Optional exact input for linkage-only construction; never inferred from
  // a public, non-namespaced work directory.
  std::filesystem::path reused_base_list_directory;
  bool base_basic_checkpoint_enabled = false;
  bool linkage_checkpoint_enabled = false;
  std::uint32_t m = 0U;
  SourceFormat training_format = SourceFormat::kAuto;
  SourceFormat base_format = SourceFormat::kAuto;
  // Resolved once from the requested value and path. Metadata inspection,
  // scalar dispatch and source construction must consume these same routes.
  SourceFormat effective_training_format = SourceFormat::kAuto;
  SourceFormat effective_base_format = SourceFormat::kAuto;
  std::string dataset_name;
  std::filesystem::path dataset_data_root = "../data";
  // Public path values are retained for faithful cfg snapshots. Readers,
  // metadata checks and experiment guards consume the resolved paths below.
  std::filesystem::path training_file;
  std::filesystem::path effective_training_file;
  std::uint64_t training_first_row = 0U;
  std::uint64_t training_rows = 0U;
  std::filesystem::path base_file;
  std::filesystem::path effective_base_file;
  std::uint64_t base_first_row = 0U;
  std::uint64_t base_rows = 0U;
  std::uint32_t dimension = 0U;
  std::uint32_t nlist = 0U;
  std::uint64_t initialization_samples = 0U;
  RootInitializationCommand root_initialization;
  std::uint32_t train_seed = 0U;
  std::uint32_t training_rounds = 0U;
  bool train_use_opq_rotation = true;
  bool train_log_metrics = true;
  bool train_log_linkage_pre_c1 = false;
  std::string init_linkage_mode = "legacy_root_only";
  // -1 inherits base.linkage.icm_round, as in Master initialization.
  std::int32_t init_linkage_icm_rounds = 1;
  std::uint32_t train_basic_ils_iterations = 8U;
  std::uint32_t train_basic_icm_iterations = 4U;
  std::uint32_t train_basic_perturbations = 3U;
  std::uint32_t training_basic_rsd_rank = 2U;
  std::uint32_t base_seed = 0U;
  std::uint32_t base_basic_ils_iterations = 32U;
  std::uint32_t base_basic_icm_iterations = 4U;
  std::uint32_t base_basic_perturbations = 3U;
  std::uint32_t base_basic_rsd_rank = 2U;
  // One product-wide numerical mode for ordinary basic, replay, virtual-root
  // and linkage least-squares proposal costs. Root-model initialization keeps
  // its separate frozen algorithm.
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  double good_cluster_fraction = 0.0;
  double bad_cluster_training_weight = 0.0;
  TopologyCommand training_topology;
  TopologyCommand base_topology;
  VirtualRootCommand virtual_roots;
  std::uint32_t hnsw_m = 24U;
  std::uint32_t hnsw_ef_construction_cap = 0U;
  std::uint32_t hnsw_ef_search = 0U;
  std::filesystem::path work_directory;
  std::filesystem::path output_directory;
  bool archive_log = true;
  bool protect_existing_outputs = false;
  // Optional create-only preflight evidence produced by the control-variable
  // guard. The run session freezes its bytes into the durable output root.
  std::filesystem::path control_evidence;
  std::filesystem::path resume_initialized_checkpoint;
  std::filesystem::path resume_round_checkpoint;
  std::int32_t cpu_threads = 0;
  std::string omp_places;
  std::string omp_proc_bind;
  std::int32_t base_list_threads = 0;
  // Zero selects the size-dependent production policy. A positive value is
  // an explicit reduction-tree/worker-count override.
  std::uint32_t root_moment_partitions = 0U;
  std::uint32_t rotation_moment_partitions = 0U;
  std::int32_t initialization_edge_moment_shards = 0;
  std::int32_t training_edge_moment_shards = 0;
  // Runtime-resolved values. Public/requested zero values above remain intact
  // for DEV-compatible snapshots; execution and manifests consume these after
  // ResolveBuildRuntimeThreadCounts().
  std::uint32_t effective_cpu_threads = 0U;
  std::uint32_t effective_base_list_threads = 0U;
  std::uint32_t effective_root_moment_partitions = 1U;
  std::uint32_t effective_rotation_moment_partitions = 0U;
  std::uint32_t effective_initialization_edge_moment_shards = 0U;
  std::uint32_t effective_training_edge_moment_shards = 0U;
  int cuda_device = 0;
  bool use_cuda = true;
  bool gpu_norm_enable = true;
  std::int32_t training_block_rows = 0;
  std::int32_t base_block_rows = 0;
  std::int32_t cids_per_bucket = 0;
  std::int32_t bucket_flush_mebibytes = 256;
  CleanupCommand cleanup;
  bool base_linkage_store_coefficients_f32 = true;
  std::int32_t basic_inflight_blocks = 0;
  bool basic_async_io_enabled = true;
  std::int32_t basic_async_io_depth = 6;
  std::int32_t basic_async_io_mebibytes = 1024;
  bool basic_async_write_enabled = true;
  std::int32_t basic_async_write_depth = 8;
  std::int32_t basic_async_write_mebibytes = 0;
  // DEV owns two independent requested fields. Fast mode forces TF32 on;
  // strict mode defers to the separately requested allow flag.
  CudaMode cuda_mode = CudaMode::kStrict;
  bool requested_cuda_allow_tf32 = true;
  bool cuda_allow_tf32 = true;
  std::uint32_t root_score_buffer_mebibytes = 0U;
  std::uint32_t root_normalized_cache_mebibytes = 0U;
  std::uint32_t linkage_large_root_xc0_chunk_mebibytes = 256U;
  std::uint32_t linkage_large_root_workspace_mebibytes = 512U;
  std::int32_t init_linkage_large_root_xc0_chunk_mebibytes = 0;
  std::int32_t init_linkage_large_root_workspace_mebibytes = 0;
  std::uint32_t linkage_chunk_max_pairs = 0U;
  // Zero inherits the ordinary evaluator limit.
  std::uint32_t init_linkage_chunk_max_pairs = 0U;
  // Fractions in (0, 1]. DEV evaluates these in listed order independently
  // for layer zero and the remaining layers, retaining the first minimum.
  std::vector<std::uint32_t> coefficient_bits_per_layer = {
      kDefaultCoefficientBits};
  std::vector<std::uint32_t> coefficient_bits_per_layer_root;
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage;
  bool coefficient_codec_enabled = true;
  std::vector<double> coefficient_first_magnitude_quantiles;
  std::vector<double> coefficient_rest_magnitude_quantiles;
  bool coefficient_weighted_quantile = true;
  bool coefficient_allow_clipping = true;
  bool coefficient_fit_scale = true;
  std::uint32_t coefficient_refinement_sweeps = 3U;
  std::uint32_t coefficient_refinement_max_layer = 0U;
  std::uint32_t coefficient_refinement_step_limit = 1U;
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  // The public DEV key selects an eval representation. "float" has no
  // norm8 producer recipe, so preserve that request separately while any
  // quantized publisher retains a concrete internal recipe.
  std::string eval_disk_norm2_mode = "lut";
  // DEV's large-scale basic-only recall runs after base-list construction and
  // before its temporary storage may be cleaned.  The ordinary build entry
  // consumes these fields; EvalCommand owns the query/GT execution details.
  bool eval_base_enabled = false;
  bool eval_base_use_ivf = true;
  std::uint32_t eval_base_nprobe = 8U;
  std::uint32_t eval_base_warmup = 0U;
  std::uint32_t eval_base_repeat = 1U;
  // Ordinary-build linkage evaluation is opt-in. If requested, it consumes
  // the index published by that same build.
  bool eval_linkage_enabled = false;
  std::string eval_score_mode = "l2";
  bool eval_representations_auto = false;
  std::vector<EvalRepresentation> eval_representations = {
      EvalRepresentation::kQuantizedScaled};
  std::uint32_t topology_inflight_clusters = 0U;
  bool linkage_async_io_enabled = true;
  std::int32_t linkage_async_io_depth = 6;
  bool linkage_scheduler_enabled = true;
  bool linkage_scheduler_transport_batching_enabled = true;
  bool linkage_scheduler_stable = false;
  std::uint32_t linkage_direct_pool_size = 32U;
  // A non-positive value inherits the ordinary phase value, matching DEV.
  std::int32_t initialization_linkage_direct_pool_size = 24;
  std::int32_t linkage_eval_async_pinned_mebibytes = 128;
  // A non-positive value inherits the ordinary pinned budget, matching DEV.
  std::int32_t initialization_linkage_eval_async_pinned_mebibytes = 256;
  std::int32_t linkage_scheduler_workers = 0;
  std::uint32_t linkage_scheduler_queue_capacity = 0U;
  std::uint32_t linkage_rfull_cache_buffers = 0U;
  std::uint32_t linkage_rfull_cache_mebibytes = 0U;
  std::uint32_t linkage_safety_reserve_mebibytes = 0U;
  std::int32_t linkage_inner_batch_rows = 0;
  std::uint32_t linkage_max_batch_pairs = 0U;
  // A non-positive value inherits linkage_max_batch_pairs, matching DEV.
  std::int32_t linkage_init_max_batch_pairs = 0;
  std::int32_t linkage_many_nodes_preflush_target_pairs = 0;
  std::int32_t linkage_same_layer_block_rows = 0;
  // Non-positive disables same-layer window batching.
  std::int32_t linkage_same_layer_minimum_pairs = 0;
  std::int32_t linkage_same_layer_window_maximum_pairs = 0;
  std::int32_t linkage_same_layer_maximum_pending_nodes = 0;
  std::int32_t linkage_same_layer_preflush_target_pairs = 0;
  std::int32_t linkage_same_layer_preflush_target_nodes = 0;
  std::int32_t linkage_tiny_forced_pair_limit = 0;
  bool linkage_tiny_forced_cpu_enabled = false;
  bool initialization_linkage_tiny_forced_cpu_enabled = false;
  std::int32_t linkage_tiny_forced_node_limit = 0;
  std::uint32_t linkage_single_gpu_min_candidates = 16U;
  std::uint32_t linkage_single_gpu_max_candidates = 0U;
  std::uint32_t linkage_same_layer_single_gpu_min_candidates = 6U;
  std::int32_t linkage_bad_window_rows = 0;
  std::int32_t linkage_bad_window_maximum_pairs = 0;
  std::uint32_t final_spool_window_clusters = 0U;
  std::uint64_t final_spool_window_real_rows = 0U;
  // Parser-owned provenance. Config files and --set/direct CLI assignments
  // are resolved from left to right; this contains only the final source of
  // each effective field.
  std::vector<std::filesystem::path> config_files;
  // Immutable bytes captured by ParseBuildCommand. Runtime provenance must
  // write these bytes and must never reopen the source path.
  std::vector<FrozenBuildConfigInput> frozen_config_inputs;
  std::vector<BuildConfigAssignment> effective_assignments;
};

struct EffectiveBuildExecutionConfig final {
  std::uint32_t initialization_direct_pool_size = 0U;
  std::uint32_t initialization_maximum_batch_pairs = 0U;
  std::uint32_t initialization_async_pinned_mebibytes = 0U;
  std::uint32_t initialization_large_root_xc0_chunk_mebibytes = 0U;
  std::uint32_t initialization_large_root_workspace_mebibytes = 0U;
};

// Only an ordinary training round or enabled base encoding consumes these
// encoders. Initialization has its own basic path and capability contract.

[[nodiscard]] std::uint32_t EffectiveTrainingBlockRows(
    const BuildCommand& command) noexcept;
[[nodiscard]] std::uint32_t EffectiveBaseBlockRows(
    const BuildCommand& command) noexcept;
[[nodiscard]] std::uint32_t EffectiveCidsPerBucket(
    const BuildCommand& command) noexcept;
[[nodiscard]] std::size_t EffectiveBucketFlushBytes(
    const BuildCommand& command) noexcept;
[[nodiscard]] std::uint32_t EffectiveLinkageSchedulerWorkers(
    const BuildCommand& command,
    std::uint32_t phase_direct_pool_size) noexcept;
[[nodiscard]] std::uint32_t EffectiveLinkageAsyncPinnedMebibytes(
    const BuildCommand& command) noexcept;
[[nodiscard]] ResolvedCleanupPolicy ResolveCleanupPolicy(
    const CleanupCommand& command) noexcept;
[[nodiscard]] const char* CleanupPresetName(CleanupPreset preset) noexcept;

// Maximum number of admitted basic blocks in the bounded I/O pipeline.
[[nodiscard]] std::uint32_t ResolveBasicInflightBlockCount(
    const BuildCommand& command, std::uint32_t source_scalar_bytes,
    std::uint32_t block_rows) noexcept;

// DEV's basic prefetch queue and ordered asynchronous writer are independent
// from the total inflight window. PP keeps those three capacities distinct
// even though one pipeline owns their lifetimes.
[[nodiscard]] std::uint32_t ResolveBasicPrefetchBlockCount(
    const BuildCommand& command, std::uint32_t source_scalar_bytes,
    std::uint32_t block_rows) noexcept;
[[nodiscard]] std::uint32_t ResolveBasicPendingWriteCount(
    const BuildCommand& command, std::uint32_t source_scalar_bytes,
    std::uint32_t block_rows) noexcept;
[[nodiscard]] std::uint32_t ResolveLinkagePrefetchClusterCount(
    const BuildCommand& command) noexcept;

// Resolves DEV's zero/inherit and shard-clamping rules against the OpenMP
// runtime selected after affinity/environment setup.
void ResolveBuildRuntimeThreadCounts(
    BuildCommand* command, std::uint32_t openmp_default_threads) noexcept;

[[nodiscard]] EffectiveBuildExecutionConfig ResolveBuildExecutionConfig(
    const BuildCommand& command) noexcept;

// Resolved ownership of execution resources by production phase. CPU and
// CUDA capabilities remain separate fields so a no-CUDA run cannot be
// represented by fake CUDA contexts. The runner, logs, manifest and
// completion validator all consume this one plan.
struct BuildResourcePlan final {
  std::uint32_t basic_cuda_instances = 1U;
  std::uint32_t basic_cpu_instances = 0U;
  std::uint32_t initialization_scheduler_backends = 0U;
  std::uint32_t initialization_direct_backends = 0U;
  std::uint32_t initialization_cpu_direct_backends = 0U;
  std::uint32_t ordinary_scheduler_backends = 0U;
  std::uint32_t ordinary_direct_backends = 0U;
  std::uint32_t ordinary_cpu_direct_backends = 0U;
  std::uint32_t replay_cuda_instances = 1U;
  std::uint32_t replay_cpu_instances = 0U;
};

[[nodiscard]] BuildResourcePlan ResolveBuildResourcePlan(
    const BuildCommand& command,
    std::optional<bool> scheduler_enabled_override = std::nullopt) noexcept;

// Emits every parsed production field in a stable, typed representation.
// Tuple values and booleans are reconstructed from BuildCommand rather than
// echoed from input text, so aliases and lexical spellings compare equally.
void WriteCanonicalBuildConfig(
    const BuildCommand& command, std::ostream& output);

// Emits a complete cfg accepted by ParseBuildCommand. String values are
// escaped and quoted; lexical input spellings are normalized. Enabled
// STLQpp-only extensions are emitted after the exact DEV field projection.
void WriteRoundTripBuildConfig(
    const BuildCommand& command, std::ostream& output);

[[nodiscard]] StatusOr<BuildCommand> ParseBuildCommand(
    int argc, char** argv);

// Resolves symlinks for known build paths. The final index path is supplied
// only after the trained-model hash has resolved its identity namespace.
[[nodiscard]] Status ValidateBuildPathIsolation(
    const BuildCommand& command,
    const std::filesystem::path& final_index = {});

// The publication stem is independent of io.train_file, which may identify
// an existing model belonging to another run.
[[nodiscard]] std::filesystem::path TrainingModelPublicationBase(
    const BuildCommand& command, ArtifactHash64 training_source,
    ArtifactHash64 training_recipe);

// Resolves the model dimension from the actual vector source or reopened
// checkpoint. DEV does not expose a dataset.dimension config field.
[[nodiscard]] Status ResolveBuildInputMetadata(BuildCommand* command);

void PrintBuildUsage(std::ostream& output);

}  // namespace stlqpp::app
