#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/artifact_hash.h"
#include "stlqpp/core/numeric/column_major_matrix.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/least_squares_cost_mode.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace stlqpp {

enum class SourceScalarKind : std::uint8_t {
  kU8 = 1U,
  kF32 = 2U,
};

struct TrainingSourceSliceHashInput final {
  ArtifactHash64 registered_source;
  SourceScalarKind scalar = SourceScalarKind::kU8;
  std::uint32_t dimension = 0U;
  std::uint64_t first_vector = 0U;
  std::uint64_t vector_count = 0U;
};

struct TrainedModelHashView final {
  ColumnMajorMatrixF32View rotation;
  std::vector<ColumnMajorMatrixF32View> root_codebooks;
  std::vector<ColumnMajorMatrixF32View> edge_codebooks;
};

struct BaseStoreLayout final {
  std::uint64_t source_block_rows = 0U;
  std::uint32_t cids_per_bucket = 0U;
};

struct BasicEncoderRecipe final {
  std::uint32_t rsd_rank = 2U;
  std::uint32_t ils_iterations = 0U;
  std::uint32_t icm_iterations = 0U;
  std::uint32_t perturb_layers = 0U;
  std::uint32_t norm_bins = 0U;
  std::uint64_t seed = 0U;
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
};

enum class BasicExecutorKind : std::uint8_t {
  kCpuOnly = 0U,
  kCudaOnly = 1U,
  // Historical hash recipe only; no mixed executor is constructed.
  kLegacyCpuCudaHybrid = 2U,
};

// Resource controls stay outside artifact identity. This distinction is
// persisted because merged GEMM widths can change FP32 winner selection.
enum class LinkageNumericalPolicy : std::uint8_t {
  kSourceTask = 1U,
  kMergedThroughput = 2U,
};

// CPU and CUDA implement the same candidate and tie rules, but their FP32
// reductions are not byte-identical on every dense input. The fixed block
// assignment is therefore part of a persisted basic artifact's identity.
// Queue depth, thread count and calibration timings remain execution-only.
struct BasicExecutionRecipe final {
  BasicExecutorKind executor = BasicExecutorKind::kCpuOnly;
  // Retained in the hash layout for existing artifacts. New recipes use zero.
  std::uint32_t cpu_stride = 0U;
  bool cuda_allow_tf32 = false;
};

struct BaseBasicHashInput final {
  TrainedModelHashView trained_model;
  std::string dataset_name;
  bool source_is_u8 = false;
  std::uint64_t source_first_vector = 0U;
  std::uint64_t effective_base_count = 0U;
  ModelShape shape;
  std::vector<std::uint32_t> codebook_sizes;
  BaseStoreLayout store_layout;
  BasicEncoderRecipe encoder;
  BasicExecutionRecipe execution;
};

struct BaseListHashInput final {
  ArtifactHash64 base_basic_directory;
  ModelShape shape;
  std::uint64_t real_count = 0U;
};

struct LinkageTopologyRecipe final {
  double root_percentile = 0.0;
  std::uint32_t topology_layers = 0U;
  std::uint32_t max_depth = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t depth_k = 0U;
  // RSD always yields the native global-greedy support plus one vertical
  // deviation at each of M steps. M+1 is derived, not configured. rsd_rank
  // selects the horizontal choice at that deviation; rank 1 is native.
  std::uint32_t rsd_rank = 0U;
  std::uint32_t rsd_icm_rounds = 0U;
  std::uint32_t inner_batch_rows = 0U;
  std::uint32_t maximum_batch_pairs = 0U;
  std::uint32_t many_nodes_preflush_target_pairs = 0U;
  std::uint32_t same_layer_block_rows = 0U;
  std::uint32_t same_layer_minimum_batch_pairs = 0U;
  std::uint32_t same_layer_window_maximum_pairs = 0U;
  std::uint32_t same_layer_maximum_pending_nodes = 0U;
  std::int32_t same_layer_preflush_target_pairs = 0;
  std::uint32_t same_layer_preflush_target_nodes = 0U;
  bool tiny_forced_cpu_enabled = true;
  std::uint32_t tiny_forced_pair_limit = 0U;
  std::uint32_t tiny_forced_node_limit = 0U;
  bool single_gpu_route_enabled = false;
  bool single_gpu_scheduler_backed = false;
  std::uint32_t single_gpu_min_candidates = 16U;
  std::uint32_t single_gpu_max_candidates = 0U;
  std::uint32_t same_layer_single_gpu_min_candidates = 6U;
  std::uint32_t bad_frozen_window_rows = 0U;
  std::uint32_t bad_frozen_maximum_pairs = 0U;
  bool cuda_allow_tf32 = false;
  std::uint64_t seed = 0U;
  LeastSquaresCostMode least_squares_cost_mode =
      LeastSquaresCostMode::kNormalEquation;
  LinkageNumericalPolicy numerical_policy =
      LinkageNumericalPolicy::kSourceTask;
};

struct VirtualRootRecipe final {
  bool enabled = true;
  double target_ratio = 0.0;
  std::uint32_t minimum_per_cluster = 0U;
  std::uint32_t maximum_per_cluster = 0U;
  bool use_fixed_per_cluster = false;
  std::uint32_t fixed_per_cluster = 0U;
  std::uint32_t umap_knn_k = 0U;
  std::uint32_t local_connectivity = 0U;
  double overlap_threshold = 0.0;
  bool prefer_density_peaks = false;
  std::uint32_t anchor_neighbor_k = 0U;
};

enum class GoodParentCandidateRecipe : std::uint8_t {
  kCpuInnerExact = 1U,
};

// Preserve the exact recipe value for existing train/base identities.
// Value 2 belonged to the retired virtual HNSW recipe; never reuse it.
enum class BadVirtualCandidateRecipe : std::uint8_t {
  kCpuExact = 1U,
};

enum class BadClusterGraphRecipe : std::uint8_t {
  // Value 1 belonged to the retired exact/HNSW recipe. Never reuse it.
  kExactOnly = 2U,
  kAutoHnsw = 3U,
  kHnsw = 4U,
};

enum class ParentCandidateQuotaRecipe : std::uint8_t {
  kEligibleLayerSlotsHalfK = 1U,
};

struct CandidateSearchRecipe final {
  GoodParentCandidateRecipe good_parent =
      GoodParentCandidateRecipe::kCpuInnerExact;
  BadVirtualCandidateRecipe bad_virtual =
      BadVirtualCandidateRecipe::kCpuExact;
  BadClusterGraphRecipe bad_graph =
      BadClusterGraphRecipe::kExactOnly;
  ParentCandidateQuotaRecipe parent_quota =
      ParentCandidateQuotaRecipe::kEligibleLayerSlotsHalfK;
  std::uint32_t cpu_exact_max_points = 150000U;
  std::uint32_t hnsw_m = 24U;
  std::uint32_t hnsw_ef_construction_cap = 0U;
  std::uint32_t hnsw_ef_search = 0U;
};

struct SphericalRvqHashRecipe final {
  std::uint8_t seed_method = 0U;
  std::uint64_t sample_vectors = 0U;
  std::uint32_t maximum_iterations = 0U;
  float tolerance = 0.0F;
  std::uint32_t kmeans_parallel_rounds = 0U;
  float kmeans_parallel_oversample = 0.0F;
  std::uint64_t kmeans_parallel_seed = 0U;
  std::uint32_t candidate_cap = 0U;
  float initial_weight = 0.0F;
  float minimum_weight = 0.0F;
  float outlier_quantile = 0.0F;
  float cost_threshold = 0.0F;
  float annealing_factor = 0.0F;
  std::uint32_t warmup_iterations = 0U;
  std::uint32_t quantile_bins = 0U;
  std::uint32_t large_layer_threshold = 0U;
  std::uint32_t large_layer_top_coarse = 0U;
  bool large_layer_hierarchy_enabled = true;
  std::uint32_t large_layer_fixed_coarse_count = 0U;
  std::uint32_t large_layer_minimum_coarse_count = 128U;
  std::uint32_t large_layer_maximum_coarse_count = 1'024U;
  bool large_layer_round_coarse_to_power_of_two = true;
};

struct RootOnlyTopologyHashRecipe final {
  std::uint32_t radial_layer_count = 0U;
  double root_fraction = 0.0;
  std::uint32_t depth_k = 0U;
  std::uint32_t knn_k = 0U;
  std::uint32_t maximum_depth = 0U;
  std::uint32_t seed = 0U;
  std::uint32_t icm_rounds = 1U;
  bool shared_cluster_seed = false;
  std::uint32_t inner_batch_rows = 0U;
  std::uint32_t maximum_batch_pairs = 0U;
  std::uint32_t many_nodes_preflush_target_pairs = 0U;
  std::uint32_t same_layer_block_rows = 0U;
  std::uint32_t same_layer_minimum_batch_pairs = 0U;
  std::uint32_t same_layer_window_maximum_pairs = 0U;
  std::uint32_t same_layer_maximum_pending_nodes = 0U;
  std::int32_t same_layer_preflush_target_pairs = 0;
  std::uint32_t same_layer_preflush_target_nodes = 0U;
  bool tiny_forced_cpu_enabled = true;
  std::uint32_t tiny_forced_pair_limit = 0U;
  std::uint32_t tiny_forced_node_limit = 0U;
  std::uint32_t large_root_xc0_chunk_mebibytes = 0U;
  std::uint32_t large_root_workspace_mebibytes = 0U;
};

// Exact semantic authority for initialized and ordinary-round checkpoints.
// Product resource controls are absent. Fixed baseline algorithms are bound
// by the scope version; configurable numerical lane assignment is explicit.
struct TrainResumeHashInput final {
  ModelShape shape;
  // One authoritative Master train.seed drives RVQ, initialization/basic and
  // every ordinary training-basic round.
  std::uint32_t train_seed = 0U;
  bool fast_initialization = false;
  SphericalRvqHashRecipe root_initialization;
  double good_cluster_fraction = 0.0;
  RootOnlyTopologyHashRecipe initialization_topology;
  // Root-update accumulator count controls both the FP64 reduction tree and
  // its accumulation/Schur worker count, so it is checkpoint identity.
  std::uint32_t root_moment_partitions = 1U;
  // Frozen DEV also binds the post-C_one OPQ cross-covariance reduction tree
  // to the OpenMP team size. STLQpp names that numerical partition explicitly
  // so changing a general CPU capacity cannot alter checkpoint bytes.
  std::uint32_t rotation_moment_partitions = 0U;
  std::uint32_t initialization_edge_moment_shards = 0U;
  std::uint32_t training_edge_moment_shards = 0U;
  // Phase-local MKL policy can change FP32 reduction and training trajectory.
  std::uint32_t root_spd_threads = 1U;
  std::uint32_t edge_spd_threads = 1U;
  std::uint32_t source_block_rows = 0U;
  BasicEncoderRecipe basic_encoder;
  BasicExecutionRecipe basic_execution;
  // Deprecated compatibility slot. DEV train.max_R_iters is deliberately
  // excluded from the resume identity because it controls future work.
  std::uint32_t training_round_count = 0U;
  bool update_opq_rotation = true;
  LinkageTopologyRecipe topology;
  VirtualRootRecipe virtual_roots;
  float bad_cluster_training_weight = 0.0F;
  CandidateSearchRecipe candidate_search;
};

struct LinkageListHashInput final {
  TrainedModelHashView trained_model;
  ArtifactHash64 base_list_directory;
  ModelShape shape;
  std::vector<std::uint32_t> codebook_sizes;
  std::uint32_t edge_root_codebook_size = 0U;
  ArrayView<const std::uint8_t> bad_cluster_mask;
  LinkageTopologyRecipe topology;
  VirtualRootRecipe virtual_roots;
  CandidateSearchRecipe candidate_search;
};

struct CoefficientCodecRecipe final {
  std::vector<std::uint32_t> bits_per_layer = {};
  std::vector<std::uint32_t> bits_per_layer_root = {};
  std::vector<std::uint32_t> bits_per_layer_linkage = {};
  bool weighted_quantile = false;
  std::vector<double> first_magnitude_quantiles;
  std::vector<double> rest_magnitude_quantiles;
  bool allow_clipping = false;
  bool fit_reconstruction_scales = false;
  std::uint32_t refinement_sweeps = 0U;
  // Compatibility bit retained in the established default hash stream.
  bool refine_all_layers = false;
  // Zero means all layers. Positive values refine that layer prefix.
  std::uint32_t refinement_max_layer = 0U;
  std::uint32_t refinement_step_limit = 0U;
};

struct CoefficientCodecHashInput final {
  ArtifactHash64 linkage_list_directory;
  CoefficientCodecRecipe codec;
};

struct Norm8HashInput final {
  ArtifactHash64 linkage_list_directory;
  ArtifactHash64 coefficient_codec;
  std::uint32_t center_count = 256U;
  ReconstructionNorm8Mode mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t kmeans_iterations = 0U;
};

struct ScaledScore8HashInput final {
  ArtifactHash64 coefficient_codec;
  ArtifactHash64 source_norm8_cache;
  ArrayView<const float> scales;
  ArrayView<const float> biases;
};

class ArtifactHashFactory final {
 public:
  // Content identity of the rotation and root codebooks consumed by basic
  // encoding. Persist this separately when a product must later prove that
  // it is being decoded with the same model: composite artifact hashes are
  // deliberately not reversible.
  [[nodiscard]] static ArtifactHash64 BaseModel(
      const TrainedModelHashView& model) noexcept;
  // DEV binds training resume compatibility to dataset.name while excluding
  // dataset.data_root, so relocating the same registered dataset does not
  // invalidate a checkpoint.  Keep that registration boundary explicit
  // instead of relying on a process-wide hidden numeric source ID.
  [[nodiscard]] static ArtifactHash64 RegisteredTrainingSource(
      std::string_view dataset_name) noexcept;
  [[nodiscard]] static ArtifactHash64 TrainingSourceSlice(
      const TrainingSourceSliceHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 TrainResume(
      const TrainResumeHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 BaseBasicDirectory(
      const BaseBasicHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 BaseListDirectory(
      const BaseListHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 LinkageListDirectory(
      const LinkageListHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 CoefficientCodec(
      const CoefficientCodecHashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 Norm8Cache(
      const Norm8HashInput& input) noexcept;
  [[nodiscard]] static ArtifactHash64 ScaledScore8(
      const ScaledScore8HashInput& input) noexcept;
};

}  // namespace stlqpp
