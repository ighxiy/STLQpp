#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>

namespace stlqpp::app {

enum class ProductConfigScope : std::uint8_t {
  kBuild = 1,
  kEval = 2,
};

// Public names form the canonical STLQpp product snapshot. Most were imported
// from the frozen source recipe; renamed concepts such as RSD use their current
// product spelling rather than preserving obsolete aliases.
// private_option only bridges to the typed implementation parser; it is not
// admitted as a cfg/--set key and is never emitted in a public snapshot.
struct ProductConfigField final {
  enum class Status : std::uint8_t {
    kRuntimeConsumer = 1,
    kFixedProductContract = 2,
  };

  constexpr ProductConfigField(
      std::string_view public_name,
      std::string_view implementation_option) noexcept
      : public_key(public_name), private_option(implementation_option),
        status(implementation_option.empty()
                   ? Status::kFixedProductContract
                   : Status::kRuntimeConsumer) {}

  std::string_view public_key;
  std::string_view private_option;
  Status status;
};

inline ArrayView<const ProductConfigField> ProductConfigFields(
    ProductConfigScope scope) noexcept {
  static constexpr ProductConfigField kBuild[] = {
      {"model.m", "m"},
      {"dataset.name", "dataset"},
      {"dataset.data_root", "data-root"},
      {"dataset.train_path", "train"},
      {"dataset.base_path", "base"},
      {"dataset.base_first_row", "base-first-row"},
      {"dataset.ntrain", "train-rows"},
      {"dataset.nbase", "base-rows"},
      {"model.h0", "nlist"},
      {"large.train_format", "train-format"},
      {"large.base_format", "base-format"},

      {"train.enabled", "train-enabled"},
      {"train.init_enabled", "train-init-enabled"},
      {"base.encode.enabled", "base-encode-enabled"},
      {"base.linkage.enabled", "base-linkage-enabled"},
      {"advanced.eval_only", "eval-only"},
      {"train.ckpt.enabled", "train-checkpoint-enabled"},
      {"train.ckpt.every_R", "train-checkpoint-every-r"},
      {"io.train_file", "train-checkpoint-file"},
      {"io.pre_fix", "output-prefix"},
      {"io.base_file", "reused-base-list-directory"},
      {"large.base_basic_checkpoint", "base-basic-checkpoint-enabled"},
      {"large.linkage_checkpoint", "linkage-checkpoint-enabled"},

      {"train.init_samples", "init-samples"},
      {"train.seed", "train-seed"},
      {"train.max_R_iters", "rounds"},
      {"train.use_opq_rotation", "train-use-opq-rotation"},
      {"train.log_metrics", "train-log-metrics"},
      {"train.log_linkage_pre_c1", "train-log-linkage-pre-c1"},
      {"train.init_linkage_mode", "train-init-linkage-mode"},
      {"train.init_linkage_icm_round", "init-linkage-icm-rounds"},
      {"train.encode.ils_iters", "train-basic-ils-iterations"},
      {"train.encode.icm_iters", "train-basic-icm-iterations"},
      {"train.encode.perturb_k", "train-basic-perturbations"},
      {"train.encode.rsd_rank", "training-basic-rsd-rank"},
      {"least_squares.cost_mode", "least-squares-cost-mode"},
      {"train.kmeans_init", "kmeans-init"},
      {"train.kmeans_iters", "kmeans-iters"},
      {"train.kmeans_tol", "kmeans-tol"},
      {"train.kmeansll_rounds", "kmeansll-rounds"},
      {"train.kmeansll_oversample", "kmeansll-oversample"},
      {"train.kmeansll_candidate_cap", "kmeansll-candidate-cap"},
      {"train.kmeansll_seed", "kmeansll-seed"},
      {"train.kmeans_initial_weight", "kmeans-initial-weight"},
      {"train.kmeans_min_weight", "kmeans-min-weight"},
      {"train.kmeans_outlier_quantile", "kmeans-outlier-quantile"},
      {"train.kmeans_cost_threshold", "kmeans-cost-threshold"},
      {"train.kmeans_annealing_factor", "kmeans-annealing-factor"},
      {"train.kmeans_warmup_iters", "kmeans-warmup-iters"},
      {"train.kmeans_quantile_bins", "kmeans-quantile-bins"},
      {"train.kmeans_large_k_threshold", "kmeans-large-k-threshold"},
      {"train.kmeans_large_k_top_coarse", "kmeans-large-k-top-coarse"},
      {"train.kmeans_device_cache_mb", "root-normalized-cache-mib"},
      {"train.kmeans_large_k_hier2_enable", "kmeans-large-k-hier2-enable"},
      {"train.kmeans_large_k_fixed_k1", "kmeans-large-k-fixed-k1"},
      {"train.kmeans_large_k_k1_min", "kmeans-large-k-k1-min"},
      {"train.kmeans_large_k_k1_max", "kmeans-large-k-k1-max"},
      {"train.kmeans_large_k_k1_pow2", "kmeans-large-k-k1-pow2"},

      {"train.linkage.num_layers", "training-topology-layers"},
      {"train.linkage.root_percentile", "training-topology-root-fraction"},
      {"train.linkage.depth_k", "training-topology-depth-k"},
      {"train.linkage.knn_k", "training-topology-knn-k"},
      {"train.linkage.max_depth", "training-topology-maximum-depth"},
      {"train.linkage.seed", "training-topology-seed"},
      {"train.linkage.icm_round", "training-linkage-icm-rounds"},
      {"train.linkage.rsd_rank", "training-linkage-rsd-rank"},
      {"train.linkage.bad_cluster_knn_backend", "train-linkage-bad-cluster-knn-backend"},
      {"train.linkage.bad_cluster_cpu_exact_max_points", "train-linkage-bad-cluster-cpu-exact-max-points"},

      {"base.linkage.num_layers", "base-topology-layers"},
      {"base.linkage.root_percentile", "base-topology-root-fraction"},
      {"base.linkage.depth_k", "base-topology-depth-k"},
      {"base.linkage.knn_k", "base-topology-knn-k"},
      {"base.linkage.max_depth", "base-topology-maximum-depth"},
      {"base.linkage.seed", "base-topology-seed"},
      {"base.linkage.icm_round", "base-linkage-icm-rounds"},
      {"base.linkage.rsd_rank", "base-linkage-rsd-rank"},
      {"base.linkage.bad_cluster_knn_backend", "base-linkage-bad-cluster-knn-backend"},
      {"base.linkage.bad_cluster_cpu_exact_max_points", "base-linkage-bad-cluster-cpu-exact-max-points"},

      {"base.encode.seed", "base-seed"},
      {"base.encode.ils_iters", "base-basic-ils-iterations"},
      {"base.encode.icm_iters", "base-basic-icm-iterations"},
      {"base.encode.perturb_k", "base-basic-perturbations"},
      {"base.encode.rsd_rank", "base-basic-rsd-rank"},

      {"hnsw.M", "hnsw-m"},
      {"hnsw.ef_construction_cap", "hnsw-ef-construction-cap"},
      {"hnsw.ef_search", "hnsw-ef-search"},
      {"virtual.good_fraction", "good-cluster-fraction"},
      {"virtual.enabled", "virtual-enabled"},
      {"virtual.alpha_bad", "bad-cluster-weight"},
      {"virtual.virtual_ratio", "virtual-ratio"},
      {"virtual.min_virtual", "virtual-minimum"},
      {"virtual.max_virtual", "virtual-maximum"},
      {"virtual.umap_knn_k", "virtual-neighbor-count"},
      {"virtual.local_connectivity", "virtual-local-connectivity"},
      {"virtual.overlap_thr", "virtual-overlap-threshold"},
      {"virtual.prefer_peaks", "virtual-prefer-peaks"},
      {"virtual.anchor_neighbor_k", "virtual-anchor-neighbor-k"},
      {"virtual.use_fixed_virtual_per_cluster",
       "virtual-use-fixed-per-cluster"},
      {"virtual.fixed_virtual_per_cluster", "virtual-fixed-per-cluster"},

      {"large.tmp_dir", "work"},
      {"large.output_dir", "output"},
      {"large.profile_timing", ""},
      {"large.protect_existing_outputs", "protect-existing-outputs"},
      {"large.archive_log", "archive-log"},
      {"large.train_block", "train-block-rows"},
      {"large.base_block", "base-block-rows"},
      {"large.cluster_bucket_size", "cids-per-bucket"},
      {"large.bucket_flush_mb", "bucket-flush-mib"},
      {"large.base_list_build_threads", "base-list-threads"},
      {"large.cleanup.enabled", "cleanup-enabled"},
      {"large.cleanup.preset", "cleanup-preset"},
      {"large.cleanup.keep_base_basic", "cleanup-keep-base-basic"},
      {"large.cleanup.keep_base_basic_cluster_id",
       "cleanup-keep-base-basic-cluster-id"},
      {"large.cleanup.keep_base_basic_ivf_lists",
       "cleanup-keep-base-basic-ivf-lists"},
      {"large.cleanup.keep_base_list", "cleanup-keep-base-list"},
      {"large.cleanup.keep_base_list_raw", "cleanup-keep-base-list-raw"},
      {"large.cleanup.keep_linkage_list_f32",
       "cleanup-keep-linkage-list-f32"},
      {"large.cleanup.keep_linkage_parent_u32",
       "cleanup-keep-linkage-parent-u32"},
      {"large.base_linkage_store_coeffs_f32",
       "base-linkage-store-coefficients-f32"},
      {"large.linkage_coeff_codec.enabled", "coefficient-codec-enabled"},
      {"large.linkage_coeff_codec.bits_per_layer",
       "coefficient-bits-per-layer"},
      {"large.linkage_coeff_codec.bits_per_layer_root",
       "coefficient-bits-per-layer-root"},
      {"large.linkage_coeff_codec.bits_per_layer_linkage",
       "coefficient-bits-per-layer-linkage"},
      {"large.linkage_coeff_codec.use_weighted_quantile",
       "coefficient-weighted-quantile"},
      {"large.linkage_coeff_codec.allow_clip",
       "coefficient-allow-clipping"},
      {"large.linkage_coeff_codec.fit_scale", "coefficient-fit-scale"},
      {"large.linkage_coeff_codec.q_refine_sweeps",
       "coefficient-refinement-sweeps"},
      {"large.linkage_coeff_codec.q_refine_max_layer",
       "coefficient-refinement-max-layer"},
      {"large.linkage_coeff_codec.q_refine_step_limit",
       "coefficient-refinement-step-limit"},
      {"large.linkage_coeff_codec.p_first_candidates",
       "coefficient-p-first-candidates"},
      {"large.linkage_coeff_codec.p_rest_candidates",
       "coefficient-p-rest-candidates"},
      {"eval.disk_norm2_mode", "disk-norm2-mode"},
      {"eval.disk_norm2_lut_mode", "norm8-mode"},
      {"eval.disk_norm2_lut_kmeans_niter", "norm8-lloyd-iterations"},
      {"eval.base.enabled", "base-eval-enabled"},
      {"eval.base.use_ivf", "base-eval-use-ivf"},
      {"eval.base.nprobe", "base-eval-nprobe"},
      {"eval.base.warmup", "base-eval-warmup"},
      {"eval.base.repeat", "base-eval-repeat"},
      {"eval.linkage.enabled", "linkage-eval-enabled"},
      {"eval.linkage.gpu_norm_enable", "gpu-norm-enable"},
      {"eval.linkage.modes", "eval-modes"},
      {"eval.linkage.score_mode", "eval-score-mode"},

      {"runtime.omp_threads", "threads"},
      {"runtime.omp_places", "omp-places"},
      {"runtime.omp_proc_bind", "omp-proc-bind"},
      {"runtime.cuda_device", "cuda-device"},
      {"runtime.use_cuda", "use-cuda"},
      {"runtime.cuda_mode", "cuda-mode"},
      {"runtime.cuda_allow_tf32", "cuda-allow-tf32"},
      {"runtime.cuda_pool_size", "linkage-direct-pool-size"},
      {"runtime.cuda_pool_size_init_linkage",
       "init-linkage-direct-pool-size"},
      {"runtime.cuda_linkage_cross_cluster_batch",
       "linkage-scheduler-enabled"},
      {"runtime.cuda_linkage_cross_cluster_workers",
       "linkage-scheduler-workers"},
      {"runtime.cuda_linkage_eval_async_pinned_mb",
       "linkage-eval-async-pinned-mib"},
      {"runtime.cuda_linkage_eval_async_pinned_mb_init_linkage",
       "init-linkage-eval-async-pinned-mib"},
      {"runtime.cuda_linkage_large_root_xc0_chunk_mb",
       "linkage-large-root-xc0-chunk-mib"},
      {"runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage",
       "init-linkage-large-root-xc0-chunk-mib"},
      {"runtime.cuda_linkage_mem_budget_mb",
       "linkage-large-root-workspace-mib"},
      {"runtime.cuda_linkage_mem_budget_mb_init_linkage",
       "init-linkage-large-root-workspace-mib"},
      {"runtime.c_one_update_shards", "train-edge-moment-shards"},
      {"runtime.c_one_update_shards_init_linkage",
       "init-edge-moment-shards"},
      {"runtime.basic_inflight_blocks", "basic-inflight-blocks"},
      {"runtime.basic_async_io", "basic-async-io-enabled"},
      {"runtime.basic_async_io_depth", "basic-async-io-depth"},
      {"runtime.basic_async_io_mb", "basic-async-io-mib"},
      {"runtime.basic_async_write", "basic-async-write-enabled"},
      {"runtime.basic_async_write_depth", "basic-async-write-depth"},
      {"runtime.basic_async_write_mb", "basic-async-write-mib"},
      {"runtime.linkage_async_io", "linkage-async-io-enabled"},
      {"runtime.linkage_async_io_depth", "linkage-async-io-depth"},
      {"runtime.cuda_linkage_single_gpu_min_candidates",
       "linkage-single-gpu-min-candidates"},
      {"runtime.cuda_linkage_single_gpu_max_candidates",
       "linkage-single-gpu-max-candidates"},
      {"runtime.cuda_linkage_inner_many_nodes_target_nodes",
       "linkage-inner-batch-rows"},
      {"runtime.cuda_linkage_many_nodes_max_pairs", "linkage-max-batch-pairs"},
      {"runtime.cuda_linkage_many_nodes_max_pairs_init_linkage",
       "linkage-init-max-batch-pairs"},
      {"runtime.cuda_linkage_many_nodes_preflush_target_pairs",
       "linkage-many-nodes-preflush-target-pairs"},
      {"runtime.cuda_linkage_same_layer_block_nodes",
       "linkage-same-layer-block-rows"},
      {"runtime.cuda_linkage_same_layer_single_gpu_min_candidates",
       "linkage-same-layer-single-gpu-min-candidates"},
      {"runtime.cuda_linkage_same_layer_window_min_pairs",
       "linkage-same-layer-minimum-pairs"},
      {"runtime.cuda_linkage_same_layer_window_max_pairs",
       "linkage-same-layer-window-maximum-pairs"},
      {"runtime.cuda_linkage_same_layer_window_max_pending_nodes",
       "linkage-same-layer-maximum-pending-nodes"},
      {"runtime.cuda_linkage_same_layer_preflush_target_pairs",
       "linkage-same-layer-preflush-target-pairs"},
      {"runtime.cuda_linkage_same_layer_preflush_target_nodes",
       "linkage-same-layer-preflush-target-nodes"},
      {"runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable",
       "linkage-tiny-forced-cpu-enabled"},
      {"runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable",
       "init-linkage-tiny-forced-cpu-enabled"},
      {"runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs",
       "linkage-tiny-forced-pair-limit"},
      {"runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes",
       "linkage-tiny-forced-node-limit"},
      {"runtime.cuda_linkage_bad_frozen_window_nodes",
       "linkage-bad-window-rows"},
      {"runtime.cuda_linkage_bad_frozen_window_max_pairs",
       "linkage-bad-window-maximum-pairs"},
      {"runtime.cuda_linkage_chunk_max_pairs", "linkage-chunk-max-pairs"},
      {"runtime.cuda_linkage_chunk_max_pairs_init_linkage",
       "init-linkage-chunk-max-pairs"},
  };
  static constexpr ProductConfigField kEval[] = {
      {"model.m", "m"},
      {"dataset.name", "dataset"},
      {"io.train_file", "train-checkpoint-file"},
      {"io.linkage_file", "index"},
      {"io.base_file", "reused-base-list-directory"},
      {"large.tmp_dir", "work"},
      {"large.output_dir", "output"},
      {"dataset.data_root", "data-root"},
      {"dataset.base_path", "base"},
      {"dataset.base_first_row", "base-first-row"},
      {"dataset.nbase", "base-rows"},
      {"large.base_format", "base-format"},
      {"dataset.query_path", "queries"},
      {"dataset.groundtruth_path", "groundtruth"},
      {"dataset.nquery", "nquery"},
      {"dataset.k", "k"},
      {"dataset.groundtruth_add1", "groundtruth-add1"},
      {"large.query_format", "query-format"},
      {"large.gt_format", "gt-format"},
      {"eval.base.enabled", "base-enabled"},
      {"eval.base.use_ivf", "base-use-ivf"},
      {"eval.base.nprobe", "base-nprobe"},
      {"eval.base.warmup", "base-warmup"},
      {"eval.base.repeat", "base-repeat"},
      {"eval.linkage.enabled", "linkage-enabled"},
      {"eval.linkage.gpu_norm_enable", "gpu-norm-enable"},
      {"eval.linkage.nprobe", "nprobe"},
      {"eval.linkage.score_mode", "score-mode"},
      {"runtime.omp_threads", "threads"},
      {"runtime.use_cuda", "use-cuda"},
      {"runtime.cuda_device", "cuda-device"},
      {"runtime.omp_places", "omp-places"},
      {"runtime.omp_proc_bind", "omp-proc-bind"},
      {"eval.linkage.warmup", "warmup"},
      {"eval.linkage.repeat", "repeat"},
      {"eval.linkage.query_block", "query-block"},
      {"eval.linkage.rerank_candidates", "rerank-candidates"},
      {"eval.linkage.rerank_storage", "rerank-storage"},
      {"eval.linkage.ivf_probe_mode", "ivf-probe-mode"},
      {"eval.linkage.ivf_hnsw_M", "ivf-hnsw-m"},
      {"eval.linkage.ivf_hnsw_ef_construction", "ivf-hnsw-ef-construction"},
      {"eval.linkage.ivf_hnsw_ef_search", "ivf-hnsw-ef-search"},
      {"eval.linkage.parent_louds_native_eval",
       "parent-louds-native-eval"},
      {"eval.linkage.pretransform_query_codebooks",
       "pretransform-query-codebooks"},
      {"large.profile_timing", "profile-timing"},
      {"eval.linkage.modes", "eval-modes"},
      {"eval.disk_norm2_mode", "disk-norm2-mode"},
      {"eval.disk_norm2_lut_mode", "norm8-mode"},
      {"eval.disk_norm2_lut_kmeans_niter", "norm8-lloyd-iterations"},
      {"large.linkage_coeff_codec.bits_per_layer",
       "coefficient-bits-per-layer"},
      {"large.linkage_coeff_codec.bits_per_layer_root",
       "coefficient-bits-per-layer-root"},
      {"large.linkage_coeff_codec.bits_per_layer_linkage",
       "coefficient-bits-per-layer-linkage"},
      {"large.linkage_coeff_codec.use_weighted_quantile",
       "coefficient-weighted-quantile"},
      {"large.linkage_coeff_codec.allow_clip",
       "coefficient-allow-clipping"},
      {"large.linkage_coeff_codec.fit_scale", "coefficient-fit-scale"},
      {"large.linkage_coeff_codec.q_refine_sweeps",
       "coefficient-refinement-sweeps"},
      {"large.linkage_coeff_codec.q_refine_max_layer",
       "coefficient-refinement-max-layer"},
      {"large.linkage_coeff_codec.q_refine_step_limit",
       "coefficient-refinement-step-limit"},
      {"large.linkage_coeff_codec.p_first_candidates",
       "coefficient-p-first-candidates"},
      {"large.linkage_coeff_codec.p_rest_candidates",
       "coefficient-p-rest-candidates"},
      {"eval.bench.quiet", "quiet"},
      {"eval.metric_mode", "metric-mode"},
      {"eval.metric_gt_topks", "metric-gt-topks"},
      {"eval.linkage.archive_eval_result", "archive"},
      {"large.archive_log", "archive-log"},
  };
  return scope == ProductConfigScope::kBuild
      ? ArrayView<const ProductConfigField>(kBuild, std::size(kBuild))
      : ArrayView<const ProductConfigField>(kEval, std::size(kEval));
}

// STLQpp-only operational extensions are deliberately kept out of the product
// recipe table above. Their namespace makes ownership explicit, while the
// shared field representation gives them the same typed parsing, provenance
// and round-trip guarantees.
inline ArrayView<const ProductConfigField> ProductExtensionConfigFields(
    ProductConfigScope scope) noexcept {
  static constexpr ProductConfigField kBuild[] = {
      {"stlqpp.linkage_scheduler.stable", "linkage-scheduler-stable"},
      {"stlqpp.root_update.accumulator_partitions",
       "root-moment-partitions"},
  };
  return scope == ProductConfigScope::kBuild
      ? ArrayView<const ProductConfigField>(kBuild, std::size(kBuild))
      : ArrayView<const ProductConfigField>();
}

inline bool IsProductConfigField(
    ProductConfigScope scope, std::string_view public_key) noexcept {
  for (const ProductConfigField& field : ProductConfigFields(scope)) {
    if (field.public_key == public_key) return true;
  }
  return false;
}

inline bool IsProductExtensionConfigField(
    ProductConfigScope scope, std::string_view public_key) noexcept {
  for (const ProductConfigField& field :
       ProductExtensionConfigFields(scope)) {
    if (field.public_key == public_key) return true;
  }
  return false;
}

inline StatusOr<std::string> ResolveProductConfigKey(
    ProductConfigScope scope, std::string_view public_key) {
  for (const ProductConfigField& field : ProductConfigFields(scope)) {
    if (field.public_key == public_key) {
      return std::string(field.private_option);
    }
  }
  return Status::InvalidArgument(
      "unsupported product config key " + std::string(public_key));
}

}  // namespace stlqpp::app
