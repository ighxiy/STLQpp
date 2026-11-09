#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/legacy_rsd_training_identity.h"

#include "stlqpp/model/codec/coefficient_bit_width.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>

namespace stlqpp {
namespace {

class CanonicalHash64 final {
 public:
  void AddBool(bool value) noexcept { AddByte(value ? 1U : 0U); }

  void AddU32(std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
      AddByte(static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
    }
  }

  void AddI32(std::int32_t value) noexcept {
    AddU32(static_cast<std::uint32_t>(value));
  }

  void AddU64(std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
      AddByte(static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
    }
  }

  void AddF32(float value) noexcept {
    static_assert(std::numeric_limits<float>::is_iec559);
    std::uint32_t bits = 0U;
    if (std::isnan(value)) {
      bits = 0x7fc00000U;
    } else if (value != 0.0F) {
      std::memcpy(&bits, &value, sizeof(bits));
    }
    AddU32(bits);
  }

  void AddF64(double value) noexcept {
    static_assert(std::numeric_limits<double>::is_iec559);
    std::uint64_t bits = 0U;
    if (std::isnan(value)) {
      bits = 0x7ff8000000000000ULL;
    } else if (value != 0.0) {
      std::memcpy(&bits, &value, sizeof(bits));
    }
    AddU64(bits);
  }

  void AddString(std::string_view value) noexcept {
    AddU64(static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
      AddByte(static_cast<std::uint8_t>(character));
    }
  }

  [[nodiscard]] ArtifactHash64 Finish() const noexcept {
    return ArtifactHash64::FromValue(value_);
  }

 private:
  void AddByte(std::uint8_t value) noexcept {
    value_ ^= static_cast<std::uint64_t>(value);
    value_ *= 1099511628211ULL;
  }

  std::uint64_t value_ = 14695981039346656037ULL;
};

void AddShape(CanonicalHash64& hash, const ModelShape& shape) noexcept {
  hash.AddU32(shape.dimension);
  hash.AddU32(shape.m);
  hash.AddU32(shape.nlist);
  hash.AddU32(shape.residual_codebook_size);
}

void AddU32Vector(CanonicalHash64& hash,
                  const std::vector<std::uint32_t>& values) noexcept {
  hash.AddU64(static_cast<std::uint64_t>(values.size()));
  for (const std::uint32_t value : values) {
    hash.AddU32(value);
  }
}

void AddMatrix(CanonicalHash64& hash,
               const ColumnMajorMatrixF32View& matrix) noexcept {
  hash.AddU32(matrix.rows);
  hash.AddU32(matrix.columns);
  const auto count = static_cast<std::size_t>(matrix.rows) *
                     static_cast<std::size_t>(matrix.columns);
  assert(count == 0U || matrix.values != nullptr);
  for (std::size_t index = 0U; index < count; ++index) {
    hash.AddF32(matrix.values[index]);
  }
}

void AddMatrices(CanonicalHash64& hash,
                 const std::vector<ColumnMajorMatrixF32View>& matrices) noexcept {
  hash.AddU64(static_cast<std::uint64_t>(matrices.size()));
  for (const ColumnMajorMatrixF32View& matrix : matrices) {
    AddMatrix(hash, matrix);
  }
}

ArtifactHash64 HashTrainedModel(const TrainedModelHashView& model,
                               bool include_edge_codebooks) noexcept {
  CanonicalHash64 hash;
  hash.AddString(include_edge_codebooks
                     ? "stlqpp/trained-linkage-model/v1"
                     : "stlqpp/trained-base-model/v1");
  AddMatrix(hash, model.rotation);
  AddMatrices(hash, model.root_codebooks);
  if (include_edge_codebooks) {
    AddMatrices(hash, model.edge_codebooks);
  }
  return hash.Finish();
}

}  // namespace

ArtifactHash64 ArtifactHashFactory::BaseModel(
    const TrainedModelHashView& model) noexcept {
  return HashTrainedModel(model, false);
}

ArtifactHash64 ArtifactHashFactory::RegisteredTrainingSource(
    std::string_view dataset_name) noexcept {
  CanonicalHash64 hash;
  hash.AddString("stlqpp/registered-training-source/v1");
  hash.AddString(dataset_name);
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::TrainingSourceSlice(
    const TrainingSourceSliceHashInput& input) noexcept {
  CanonicalHash64 hash;
  hash.AddString("stlqpp/training-source-slice/v1");
  hash.AddU64(input.registered_source.value());
  hash.AddU32(static_cast<std::uint32_t>(input.scalar));
  hash.AddU32(input.dimension);
  hash.AddU64(input.first_vector);
  hash.AddU64(input.vector_count);
  return hash.Finish();
}

namespace {
ArtifactHash64 HashRsdTrainingRecipe(
    const TrainResumeHashInput& input, const char* domain,
    bool legacy_model = false) noexcept {
  CanonicalHash64 hash;
  const bool normal_equation =
      input.basic_encoder.least_squares_cost_mode ==
          LeastSquaresCostMode::kNormalEquation ||
      input.topology.least_squares_cost_mode ==
          LeastSquaresCostMode::kNormalEquation;
  hash.AddString(domain);
  AddShape(hash, input.shape);
  hash.AddU32(input.train_seed);
  if (input.root_initialization.seed_method != 0U) {
    hash.AddString("spherical-seed-method/v1");
    hash.AddU32(input.root_initialization.seed_method);
  }
  hash.AddU64(input.root_initialization.sample_vectors);
  hash.AddU32(input.root_initialization.maximum_iterations);
  hash.AddF32(input.root_initialization.tolerance);
  hash.AddU32(input.root_initialization.kmeans_parallel_rounds);
  hash.AddF32(input.root_initialization.kmeans_parallel_oversample);
  hash.AddU64(input.root_initialization.kmeans_parallel_seed);
  hash.AddU32(input.root_initialization.candidate_cap);
  hash.AddF32(input.root_initialization.initial_weight);
  hash.AddF32(input.root_initialization.minimum_weight);
  hash.AddF32(input.root_initialization.outlier_quantile);
  hash.AddF32(input.root_initialization.cost_threshold);
  hash.AddF32(input.root_initialization.annealing_factor);
  hash.AddU32(input.root_initialization.warmup_iterations);
  hash.AddU32(input.root_initialization.quantile_bins);
  hash.AddU32(input.root_initialization.large_layer_threshold);
  hash.AddU32(input.root_initialization.large_layer_top_coarse);
  const auto& large = input.root_initialization;
  if (!large.large_layer_hierarchy_enabled ||
      large.large_layer_fixed_coarse_count != 0U ||
      large.large_layer_minimum_coarse_count != 128U ||
      large.large_layer_maximum_coarse_count != 1'024U ||
      !large.large_layer_round_coarse_to_power_of_two) {
    hash.AddString("root-init-large-k-split");
    hash.AddBool(large.large_layer_hierarchy_enabled);
    hash.AddU32(large.large_layer_fixed_coarse_count);
    hash.AddU32(large.large_layer_minimum_coarse_count);
    hash.AddU32(large.large_layer_maximum_coarse_count);
    hash.AddBool(large.large_layer_round_coarse_to_power_of_two);
  }
  hash.AddF64(input.good_cluster_fraction);
  hash.AddU32(input.initialization_topology.radial_layer_count);
  hash.AddF64(input.initialization_topology.root_fraction);
  hash.AddU32(input.initialization_topology.depth_k);
  hash.AddU32(input.initialization_topology.knn_k);
  hash.AddU32(input.initialization_topology.maximum_depth);
  hash.AddU32(input.initialization_topology.seed);
  if (input.initialization_topology.icm_rounds != 1U) {
    hash.AddString("init-linkage-icm-rounds");
    hash.AddU32(input.initialization_topology.icm_rounds);
  }
  hash.AddBool(input.initialization_topology.shared_cluster_seed);
  hash.AddU32(input.initialization_topology.inner_batch_rows);
  hash.AddU32(input.initialization_topology.maximum_batch_pairs);
  hash.AddU32(
      input.initialization_topology.many_nodes_preflush_target_pairs);
  hash.AddU32(input.initialization_topology.same_layer_block_rows);
  hash.AddU32(
      input.initialization_topology.same_layer_minimum_batch_pairs);
  hash.AddU32(
      input.initialization_topology.same_layer_window_maximum_pairs);
  hash.AddU32(
      input.initialization_topology.same_layer_maximum_pending_nodes);
  hash.AddI32(
      input.initialization_topology.same_layer_preflush_target_pairs);
  hash.AddU32(
      input.initialization_topology.same_layer_preflush_target_nodes);
  hash.AddBool(input.initialization_topology.tiny_forced_cpu_enabled);
  hash.AddU32(input.initialization_topology.tiny_forced_pair_limit);
  hash.AddU32(input.initialization_topology.tiny_forced_node_limit);
  hash.AddU32(
      input.initialization_topology.large_root_xc0_chunk_mebibytes);
  hash.AddU32(
      input.initialization_topology.large_root_workspace_mebibytes);
  if (input.fast_initialization) {
    hash.AddString("fast-two-codebook-initialization/v1");
  }
  hash.AddU32(input.root_moment_partitions);
  hash.AddU32(input.rotation_moment_partitions);
  hash.AddU32(input.initialization_edge_moment_shards);
  hash.AddU32(input.training_edge_moment_shards);
  if (input.root_spd_threads > 1U || input.edge_spd_threads > 1U) {
    // Preserve existing serial identities while separating new numerical
    // paths from checkpoints produced by the old global-single-thread mode.
    hash.AddString("stlqpp/large-spd-threads/v1");
    hash.AddU32(input.root_spd_threads);
    hash.AddU32(input.edge_spd_threads);
  }
  hash.AddU32(input.source_block_rows);
  hash.AddU32(input.basic_encoder.rsd_rank);
  hash.AddU32(1U);  // Fixed-top1 root policy.
  hash.AddBool(false);
  if (legacy_model) hash.AddU32(2U);  // Reserved v24 serialization slot.
  hash.AddU32(input.basic_encoder.ils_iterations);
  hash.AddU32(input.basic_encoder.icm_iterations);
  hash.AddU32(input.basic_encoder.perturb_layers);
  hash.AddU32(input.basic_encoder.norm_bins);
  hash.AddU64(input.basic_encoder.seed);
  hash.AddU32(static_cast<std::uint32_t>(input.basic_execution.executor));
  hash.AddU32(input.basic_execution.cpu_stride);
  hash.AddBool(input.basic_execution.cuda_allow_tf32);
  // DEV train.max_R_iters is an invocation-local append count. Including it
  // here would make the checkpoint identity change precisely when a caller
  // asks to append more rounds to that checkpoint.
  hash.AddBool(input.update_opq_rotation);

  hash.AddF64(input.topology.root_percentile);
  hash.AddU32(input.topology.topology_layers);
  hash.AddU32(input.topology.max_depth);
  hash.AddU32(input.topology.knn_k);
  hash.AddU32(input.topology.depth_k);
  // Keep the established hash byte stream: the old route field was always
  // the structural M+1 candidate count, and the old level was RSD rank.
  hash.AddU32(input.shape.m + 1U);
  hash.AddU32(input.topology.rsd_rank);
  hash.AddU32(input.topology.rsd_icm_rounds);
  hash.AddU32(input.topology.inner_batch_rows);
  hash.AddU32(input.topology.maximum_batch_pairs);
  hash.AddU32(input.topology.many_nodes_preflush_target_pairs);
  hash.AddU32(input.topology.same_layer_block_rows);
  hash.AddU32(input.topology.same_layer_minimum_batch_pairs);
  hash.AddU32(input.topology.same_layer_window_maximum_pairs);
  hash.AddU32(input.topology.same_layer_maximum_pending_nodes);
  hash.AddI32(input.topology.same_layer_preflush_target_pairs);
  hash.AddU32(input.topology.same_layer_preflush_target_nodes);
  hash.AddBool(input.topology.tiny_forced_cpu_enabled);
  hash.AddU32(input.topology.tiny_forced_pair_limit);
  hash.AddU32(input.topology.tiny_forced_node_limit);
  if (input.topology.single_gpu_route_enabled) {
    hash.AddString("single-node-route/v2");
    hash.AddBool(input.topology.single_gpu_scheduler_backed);
    hash.AddU32(input.topology.single_gpu_min_candidates);
    hash.AddU32(input.topology.single_gpu_max_candidates);
    hash.AddU32(input.topology.same_layer_single_gpu_min_candidates);
  }
  hash.AddU32(input.topology.bad_frozen_window_rows);
  hash.AddU32(input.topology.bad_frozen_maximum_pairs);
  hash.AddBool(input.topology.cuda_allow_tf32);
  hash.AddU64(input.topology.seed);
  hash.AddU32(static_cast<std::uint32_t>(
      input.topology.numerical_policy));

  hash.AddF64(input.virtual_roots.target_ratio);
  hash.AddU32(input.virtual_roots.minimum_per_cluster);
  hash.AddU32(input.virtual_roots.maximum_per_cluster);
  hash.AddBool(input.virtual_roots.use_fixed_per_cluster);
  hash.AddU32(input.virtual_roots.fixed_per_cluster);
  hash.AddU32(input.virtual_roots.umap_knn_k);
  hash.AddU32(input.virtual_roots.local_connectivity);
  hash.AddF64(input.virtual_roots.overlap_threshold);
  hash.AddBool(input.virtual_roots.prefer_density_peaks);
  hash.AddU32(input.virtual_roots.anchor_neighbor_k);
  if (!input.virtual_roots.enabled) hash.AddString("virtual-disabled");
  hash.AddF32(input.bad_cluster_training_weight);

  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.good_parent));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.bad_virtual));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.bad_graph));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.parent_quota));
  if (input.candidate_search.bad_graph != BadClusterGraphRecipe::kExactOnly ||
      input.candidate_search.bad_virtual != BadVirtualCandidateRecipe::kCpuExact) {
    hash.AddString("hnsw-candidates/dev-v1");
    if (input.candidate_search.bad_graph == BadClusterGraphRecipe::kAutoHnsw)
      hash.AddU32(input.candidate_search.cpu_exact_max_points);
    hash.AddU32(input.candidate_search.hnsw_m);
    hash.AddU32(input.candidate_search.hnsw_ef_construction_cap);
    hash.AddU32(input.candidate_search.hnsw_ef_search);
  }

  if (normal_equation) {
    hash.AddU32(static_cast<std::uint32_t>(
        input.basic_encoder.least_squares_cost_mode));
    hash.AddU32(static_cast<std::uint32_t>(
        input.topology.least_squares_cost_mode));
  }
  return hash.Finish();
}

}  // namespace

ArtifactHash64 ArtifactHashFactory::TrainResume(
    const TrainResumeHashInput& input) noexcept {
  // Keep configured RSD/refit while restoring tiny-tail per-pair ICM draws.
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v30");
}

ArtifactHash64 SharedCpuTailIcmTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept {
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v29");
}

ArtifactHash64 PrePositivePivotTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept {
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v27");
}

ArtifactHash64 PreCpuTailRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept {
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v26");
}

ArtifactHash64 PreviousRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept {
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v25");
}

ArtifactHash64 LegacyRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept {
  return HashRsdTrainingRecipe(input, "stlqpp/train-resume/v24", true);
}

ArtifactHash64 ArtifactHashFactory::BaseBasicDirectory(
    const BaseBasicHashInput& input) noexcept {
  CanonicalHash64 hash;
  const bool normal_equation =
      input.encoder.least_squares_cost_mode ==
      LeastSquaresCostMode::kNormalEquation;
  hash.AddString("stlqpp/base-basic-directory/v7");
  hash.AddU64(BaseModel(input.trained_model).value());
  hash.AddString(input.dataset_name);
  hash.AddBool(input.source_is_u8);
  hash.AddU64(input.source_first_vector);
  hash.AddU64(input.effective_base_count);
  AddShape(hash, input.shape);
  AddU32Vector(hash, input.codebook_sizes);
  hash.AddU64(input.store_layout.source_block_rows);
  hash.AddU32(input.store_layout.cids_per_bucket);
  hash.AddU32(input.encoder.rsd_rank);
  hash.AddU32(1U);
  hash.AddBool(false);
  // Reserved v7 serialization slot: preserve existing RSD base identities.
  hash.AddU32(2U);
  hash.AddU32(input.encoder.ils_iterations);
  hash.AddU32(input.encoder.icm_iterations);
  hash.AddU32(input.encoder.perturb_layers);
  hash.AddU32(input.encoder.norm_bins);
  hash.AddU64(input.encoder.seed);
  hash.AddU32(static_cast<std::uint32_t>(input.execution.executor));
  hash.AddU32(input.execution.cpu_stride);
  hash.AddBool(input.execution.cuda_allow_tf32);
  if (normal_equation) {
    hash.AddU32(static_cast<std::uint32_t>(
        input.encoder.least_squares_cost_mode));
  }
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::BaseListDirectory(
    const BaseListHashInput& input) noexcept {
  CanonicalHash64 hash;
  hash.AddString("stlqpp/base-list-directory/v1");
  hash.AddU64(input.base_basic_directory.value());
  AddShape(hash, input.shape);
  hash.AddU64(input.real_count);
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::LinkageListDirectory(
    const LinkageListHashInput& input) noexcept {
  CanonicalHash64 hash;
  const bool normal_equation =
      input.topology.least_squares_cost_mode ==
      LeastSquaresCostMode::kNormalEquation;
  hash.AddString("stlqpp/linkage-list-directory/v15");
  hash.AddU64(HashTrainedModel(input.trained_model, true).value());
  hash.AddU64(input.base_list_directory.value());
  AddShape(hash, input.shape);
  AddU32Vector(hash, input.codebook_sizes);
  hash.AddU32(input.edge_root_codebook_size);
  hash.AddU64(static_cast<std::uint64_t>(input.bad_cluster_mask.size()));
  for (const std::uint8_t value : input.bad_cluster_mask) {
    hash.AddU32(value);
  }

  hash.AddF64(input.topology.root_percentile);
  hash.AddU32(input.topology.topology_layers);
  hash.AddU32(input.topology.max_depth);
  hash.AddU32(input.topology.knn_k);
  hash.AddU32(input.topology.depth_k);
  // Keep the established hash byte stream while deriving M+1 from shape.
  hash.AddU32(input.shape.m + 1U);
  hash.AddU32(input.topology.rsd_rank);
  hash.AddU32(input.topology.rsd_icm_rounds);
  hash.AddU32(input.topology.inner_batch_rows);
  hash.AddU32(input.topology.maximum_batch_pairs);
  hash.AddU32(input.topology.many_nodes_preflush_target_pairs);
  hash.AddU32(input.topology.same_layer_block_rows);
  hash.AddU32(input.topology.same_layer_minimum_batch_pairs);
  hash.AddU32(input.topology.same_layer_window_maximum_pairs);
  hash.AddU32(input.topology.same_layer_maximum_pending_nodes);
  hash.AddI32(input.topology.same_layer_preflush_target_pairs);
  hash.AddU32(input.topology.same_layer_preflush_target_nodes);
  hash.AddBool(input.topology.tiny_forced_cpu_enabled);
  hash.AddU32(input.topology.tiny_forced_pair_limit);
  hash.AddU32(input.topology.tiny_forced_node_limit);
  if (input.topology.single_gpu_route_enabled) {
    hash.AddString("single-node-route/v2");
    hash.AddBool(input.topology.single_gpu_scheduler_backed);
    hash.AddU32(input.topology.single_gpu_min_candidates);
    hash.AddU32(input.topology.single_gpu_max_candidates);
    hash.AddU32(input.topology.same_layer_single_gpu_min_candidates);
  }
  hash.AddU32(input.topology.bad_frozen_window_rows);
  hash.AddU32(input.topology.bad_frozen_maximum_pairs);
  hash.AddBool(input.topology.cuda_allow_tf32);
  hash.AddU64(input.topology.seed);
  hash.AddU32(static_cast<std::uint32_t>(
      input.topology.numerical_policy));

  hash.AddF64(input.virtual_roots.target_ratio);
  hash.AddU32(input.virtual_roots.minimum_per_cluster);
  hash.AddU32(input.virtual_roots.maximum_per_cluster);
  hash.AddBool(input.virtual_roots.use_fixed_per_cluster);
  hash.AddU32(input.virtual_roots.fixed_per_cluster);
  hash.AddU32(input.virtual_roots.umap_knn_k);
  hash.AddU32(input.virtual_roots.local_connectivity);
  hash.AddF64(input.virtual_roots.overlap_threshold);
  hash.AddBool(input.virtual_roots.prefer_density_peaks);
  hash.AddU32(input.virtual_roots.anchor_neighbor_k);
  if (!input.virtual_roots.enabled) hash.AddString("virtual-disabled");

  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.good_parent));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.bad_virtual));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.bad_graph));
  hash.AddU32(static_cast<std::uint32_t>(input.candidate_search.parent_quota));
  if (input.candidate_search.bad_graph != BadClusterGraphRecipe::kExactOnly ||
      input.candidate_search.bad_virtual != BadVirtualCandidateRecipe::kCpuExact) {
    hash.AddString("hnsw-candidates/dev-v1");
    if (input.candidate_search.bad_graph == BadClusterGraphRecipe::kAutoHnsw)
      hash.AddU32(input.candidate_search.cpu_exact_max_points);
    hash.AddU32(input.candidate_search.hnsw_m);
    hash.AddU32(input.candidate_search.hnsw_ef_construction_cap);
    hash.AddU32(input.candidate_search.hnsw_ef_search);
  }

  if (normal_equation) {
    hash.AddU32(static_cast<std::uint32_t>(
        input.topology.least_squares_cost_mode));
  }
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::CoefficientCodec(
    const CoefficientCodecHashInput& input) noexcept {
  CanonicalHash64 hash;
  const bool legacy_width = input.codec.bits_per_layer.empty() &&
      input.codec.bits_per_layer_root.empty() &&
      input.codec.bits_per_layer_linkage.empty();
  const bool scalar_compatible =
      input.codec.first_magnitude_quantiles.size() == 1U &&
      input.codec.rest_magnitude_quantiles.size() == 1U &&
      input.codec.first_magnitude_quantiles.front() ==
          input.codec.rest_magnitude_quantiles.front();
  if (legacy_width) {
    hash.AddString(scalar_compatible
                       ? "stlqpp/per-cid-scalar-int6-huffman/v2"
                       : "stlqpp/per-cid-scalar-int6-huffman/v3");
  } else {
    hash.AddString("stlqpp/per-cid-scalar-huffman/v5");
  }
  hash.AddU64(input.linkage_list_directory.value());
  if (legacy_width) {
    hash.AddU32(kDefaultCoefficientBits);
  } else {
    // Explicit role widths override the common fallback. Hash the widths
    // actually consumed by each role, not the spelling of that fallback.
    AddU32Vector(hash, input.codec.bits_per_layer_root.empty()
        ? input.codec.bits_per_layer : input.codec.bits_per_layer_root);
    AddU32Vector(hash, input.codec.bits_per_layer_linkage.empty()
        ? input.codec.bits_per_layer : input.codec.bits_per_layer_linkage);
  }
  hash.AddBool(input.codec.weighted_quantile);
  if (scalar_compatible) {
    // Preserve every accepted single-candidate artifact identity.
    hash.AddF64(input.codec.first_magnitude_quantiles.front());
  } else {
    hash.AddU64(input.codec.first_magnitude_quantiles.size());
    for (const double candidate :
         input.codec.first_magnitude_quantiles) {
      hash.AddF64(candidate);
    }
    hash.AddU64(input.codec.rest_magnitude_quantiles.size());
    for (const double candidate : input.codec.rest_magnitude_quantiles) {
      hash.AddF64(candidate);
    }
  }
  hash.AddBool(input.codec.allow_clipping);
  hash.AddBool(input.codec.fit_reconstruction_scales);
  hash.AddU32(input.codec.refinement_sweeps);
  hash.AddBool(input.codec.refine_all_layers &&
               input.codec.refinement_max_layer == 0U);
  if (input.codec.refinement_max_layer != 0U) {
    hash.AddU32(input.codec.refinement_max_layer);
  }
  hash.AddU32(input.codec.refinement_step_limit);
  if (!input.codec.fit_reconstruction_scales &&
      (input.codec.first_magnitude_quantiles.size() > 1U ||
       input.codec.rest_magnitude_quantiles.size() > 1U)) {
    // Only multi-candidate fixed-scale selection changed. Keep fitted and
    // single-candidate identities, but do not reuse the old selection output.
    hash.AddString("fixed-scale-candidate-sse/v1");
  }
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::Norm8Cache(
    const Norm8HashInput& input) noexcept {
  CanonicalHash64 hash;
  const bool identity =
      input.mode == ReconstructionNorm8Mode::kIdentityLloyd;
  hash.AddString(identity ? "stlqpp/reconstruction-norm8/v1"
                          : "stlqpp/reconstruction-norm8/v2");
  hash.AddU64(input.linkage_list_directory.value());
  hash.AddU64(input.coefficient_codec.value());
  hash.AddU32(input.center_count);
  if (!identity) {
    hash.AddU32(static_cast<std::uint32_t>(input.mode));
  }
  hash.AddU32(input.kmeans_iterations);
  return hash.Finish();
}

ArtifactHash64 ArtifactHashFactory::ScaledScore8(
    const ScaledScore8HashInput& input) noexcept {
  CanonicalHash64 hash;
  hash.AddString("stlqpp/fixed-label-scaled-score8/v1");
  hash.AddU64(input.coefficient_codec.value());
  hash.AddU64(input.source_norm8_cache.value());
  hash.AddU64(input.scales.size());
  for (const float scale : input.scales) hash.AddF32(scale);
  hash.AddU64(input.biases.size());
  for (const float bias : input.biases) hash.AddF32(bias);
  return hash.Finish();
}

}  // namespace stlqpp
