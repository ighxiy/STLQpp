# Configuration reference

Edit a dataset cfg, or pass `--key value` after `--config` (for example,
`--runtime.omp_threads 16`). `--set key=value` is equivalent; later assignments
take precedence. Boolean values
accept `true`/`false` or `1`/`0`; arrays use brackets. The executable can write its complete
resolved configuration with `--dump-config effective.cfg` and then exits.

This reference covers the public build/evaluation configuration fields and the
two `stlqpp.*` operational extensions. Dataset recipes provide explicit values;
they take precedence over parser defaults. Use `--dump-config` to inspect the
effective values for a particular recipe.

## Assignments and examples

Start from a supplied dataset cfg. Strings are quoted, numeric values are not,
and `#` introduces a comment. For example, these assignments select ordinary
L2 evaluation in two representations:

```ini
io.pre_fix = "sift1m_m6"
eval.base.enabled = false
eval.linkage.enabled = true
eval.linkage.score_mode = "l2"
eval.linkage.modes = [quantized, quantized_scaled]
eval.disk_norm2_mode = "lut"
eval.disk_norm2_lut_mode = "identity"
eval.linkage.nprobe = [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

The same settings can be overridden at launch. Quote arrays in the shell:

```bash
./build-release-lto/stlqpp_build_index --config configs/sift1m_m6.cfg \
  --eval.linkage.modes '[quantized,quantized_scaled]' \
  --eval.linkage.nprobe '[1,4,16,64,256]'

# Resolve configuration without training or evaluation.
./build-release-lto/stlqpp_build_index --config configs/sift1m_m6.cfg \
  --runtime.cuda_pool_size 12 --dump-config effective.cfg
```

Paths and run prefixes are free-form strings. The tables below list the choices
for string-valued selectors; ordinary numeric budgets are described by their
purpose and units.

## Data and model

| Field | Purpose |
|---|---|
| `model.m` | Number of selected atoms in a basic/edge code; independent of IVF count. |
| `model.h0` | Root dictionary size and number of IVF lists. |
| `dataset.name` | Dataset identity used in output organization. |
| `dataset.data_root` | Root for relative input-data paths. |
| `dataset.train_path`, `dataset.base_path` | Original training and base vectors. |
| `dataset.query_path`, `dataset.groundtruth_path` | Query vectors and matching nearest-neighbor IDs. |
| `dataset.ntrain`, `dataset.nbase`, `dataset.nquery` | Requested training, base and query counts. |
| `dataset.base_first_row` | Starting row of the original base input; supplied recipes use zero. |
| `dataset.k` | Number of returned neighbors, normally 100. |
| `dataset.groundtruth_add1` | Add one to loaded GT IDs; supplied zero-based recipes use false. |
| `large.train_format`, `large.base_format`, `large.query_format` | `auto`, `bvecs`, `fvecs`, `fbin`, or `npy`. `auto` infers the reader from the input path. Accepted format spellings also include `siftbin`/`u8` for `bvecs` and `float`/`f32` for `fvecs`. See [file formats](DATASETS.md#file-formats-and-gt). |
| `large.gt_format` | `auto`, `ivecs`, `ibin`, or `npy`. |

## Stages, paths and retention

| Field | Purpose |
|---|---|
| `advanced.eval_only` | Load existing artifacts and run evaluation. |
| `train.init_enabled`, `train.enabled` | Model initialization and subsequent training rounds. |
| `base.encode.enabled`, `base.linkage.enabled` | Independent basic-encoding and linkage-construction stages. |
| `eval.base.enabled`, `eval.linkage.enabled` | Independent basic and final-linkage evaluation stages. |
| `io.pre_fix` | Human-readable run prefix, followed by the artifact identity. |
| `io.train_file` | Published model input, or round directory for training resume. |
| `io.base_file` | Retained encoded base-list input. |
| `io.linkage_file` | Published linkage index for evaluation. |
| `large.output_dir`, `large.tmp_dir` | Final output and intermediate-work roots. |
| `large.protect_existing_outputs` | Preserve existing completed output namespaces. |
| `train.ckpt.enabled`, `train.ckpt.every_R` | Export periodic training checkpoints and their interval. |
| `large.base_basic_checkpoint`, `large.linkage_checkpoint` | Enable progress reuse for basic and linkage construction. |
| `large.cleanup.enabled`, `large.cleanup.preset` | Enable post-build cleanup and choose `none`, `dev_all`, `eval_both`, `eval_float_min`, `eval_int8_min`, or `custom`; see [retention policies](#retention-policies). |
| `large.base_linkage_store_coeffs_f32` | Retain float coefficients for float evaluation and later codec derivation. |
| `large.archive_log` | Archive native logs alongside the run. |
| `train.log_metrics` | Print training metrics. |
| `train.log_linkage_pre_c1` | Also report linkage error before the edge-codebook update. |
| `large.profile_timing` | Additional eval-only diagnostic timing; use false for builds and ordinary QPS measurement. |

### Retention policies

Cleanup runs after successful construction when `large.cleanup.enabled = true`.
With cleanup disabled, artifacts are retained. Presets apply to artifacts owned
by the current build; a reused `io.base_file` remains an input.

| `large.cleanup.preset` | Retention policy |
|---|---|
| `"none"` | Keep all cleanup-managed artifacts. |
| `"dev_all"` | Keep the reusable basic list store, including raw vectors, and final representations; discard transient basic buckets. |
| `"eval_both"` | Keep final float/quantized representations; remove the basic list store and transient buckets after linkage construction. |
| `"eval_float_min"` | Retain final float coefficients and remove basic/list intermediates. It does not delete an already published quantized representation. |
| `"eval_int8_min"` | Remove basic/list intermediates and omit retained final float coefficients. |
| `"custom"` | Start with all retention flags enabled, then apply the overrides below. |

The following integer overrides accept `-1` (inherit the preset), `0` (discard),
or `1` (retain). Their default is `-1`, and overrides apply after any preset.

| Field | Retained content |
|---|---|
| `large.cleanup.keep_base_basic` | Basic buckets and the basic list-store role. |
| `large.cleanup.keep_base_basic_cluster_id` | Stable-ID/CID data in the shared basic list store. |
| `large.cleanup.keep_base_basic_ivf_lists` | IVF-order data in the shared basic list store. |
| `large.cleanup.keep_base_list` | Reusable basic list store. |
| `large.cleanup.keep_base_list_raw` | Original-vector payload within a retained list store. |
| `large.cleanup.keep_linkage_list_f32` | Final float coefficients, also controlled by `large.base_linkage_store_coeffs_f32`. |
| `large.cleanup.keep_linkage_parent_u32` | Compatibility field: the current format stores parents as LOUDS and has no separate persisted uint32 parent array to remove. |

The basic list store combines several roles, so it is retained while any of
those roles is requested. To reuse basic encoding for a second linkage recipe:

```ini
large.cleanup.enabled = true
large.cleanup.preset = "dev_all"
large.base_linkage_store_coeffs_f32 = true
```

Then select the published model and retained list directory in the next cfg:

```ini
train.init_enabled = false
train.enabled = false
base.encode.enabled = false
base.linkage.enabled = true
io.train_file = "/data/models/final.stlqmodel"
io.base_file = "/data/run/base-lists"
base.linkage.rsd_rank = 3
```

## Training and basic encoding

| Field | Purpose |
|---|---|
| `train.max_R_iters` | Number of training rounds after initialization. |
| `train.seed` | Initialization/training random seed. |
| `train.use_opq_rotation` | Learn the shared orthogonal rotation. |
| `train.init_samples` | Initialization sample budget, bounded by available training data. |
| `train.init_linkage_mode` | `legacy_root_only` uses the root-only initialization path; `fast_init` uses the full-topology initialization path. Default and supplied recipes: `legacy_root_only`. |
| `train.init_linkage_icm_round` | Initialization-linkage ICM sweeps. |
| `train.encode.rsd_rank`, `base.encode.rsd_rank` | RSD rank for independent basic codes: 1, 2 or 3. |
| `train.encode.icm_iters`, `base.encode.icm_iters` | ICM refinement sweeps for basic encoding. |
| `train.encode.ils_iters`, `base.encode.ils_iters` | ILS iterations for basic encoding. |
| `train.encode.perturb_k`, `base.encode.perturb_k` | Number of layers perturbed by basic ILS. |
| `base.encode.seed` | Base basic-encoding seed. |
| `least_squares.cost_mode` | Candidate-cost evaluation: `normal_equation` or `explicit_quadratic`. |

The supplied recipes use RSD basic encoding. Linkage RSD rank is configured
separately below.

## Topology and virtual roots

| Field | Purpose |
|---|---|
| `train.linkage.num_layers`, `base.linkage.num_layers` | Radial grouping of nodes within a GOOD cluster. |
| `train.linkage.depth_k`, `base.linkage.depth_k` | Layer window used to form eligible parent candidates. |
| `train.linkage.knn_k`, `base.linkage.knn_k` | Nearby-parent candidate budget. |
| `train.linkage.root_percentile`, `base.linkage.root_percentile` | Inner fraction reserved for independent root representations. |
| `train.linkage.max_depth`, `base.linkage.max_depth` | Maximum linkage-chain depth. |
| `train.linkage.rsd_rank`, `base.linkage.rsd_rank` | Edge residual-search rank, 1/2/3; the training setting also applies to Replay. |
| `train.linkage.icm_round`, `base.linkage.icm_round` | ICM sweeps for linkage refinement. |
| `train.linkage.seed`, `base.linkage.seed` | Linkage random seeds. |
| `virtual.enabled` | Enable virtual roots in the dispersed-cluster path. |
| `virtual.good_fraction` | Fraction of clusters assigned to GOOD; the rest use BAD/dispersed topology. |
| `virtual.alpha_bad` | BAD-cluster weight in the edge-codebook training update. |
| `virtual.virtual_ratio` | Virtual-root count relative to a BAD cluster's population. |
| `virtual.min_virtual`, `virtual.max_virtual` | Per-cluster lower/upper limits on virtual-root count. |
| `virtual.use_fixed_virtual_per_cluster`, `virtual.fixed_virtual_per_cluster` | Select a fixed per-cluster virtual-root budget instead of the ratio. |
| `virtual.umap_knn_k` | Neighborhood size used to estimate local density. |
| `virtual.local_connectivity` | Local-connectivity parameter of that neighborhood weighting. |
| `virtual.overlap_thr` | Overlap threshold used when selecting virtual anchors. |
| `virtual.prefer_peaks` | Prefer density peaks in anchor selection. |
| `virtual.anchor_neighbor_k` | Neighbor budget used around anchors. |
| `train.linkage.bad_cluster_knn_backend`, `base.linkage.bad_cluster_knn_backend` | BAD-cluster candidate construction: `cpu_exact`, `auto` or `hnsw`. |
| `train.linkage.bad_cluster_cpu_exact_max_points`, `base.linkage.bad_cluster_cpu_exact_max_points` | Cluster-size threshold used by `auto` to switch candidate construction. |
| `hnsw.M`, `hnsw.ef_construction_cap`, `hnsw.ef_search` | Connectivity and search budgets for the construction-side HNSW graph; zero budget values select the automatic policy. |

Train and base may use different topology budgets. Changing M or IVF changes
the trained model; changing only base linkage rank/topology can reuse it.

## Coefficients and scoring

| Field | Purpose |
|---|---|
| `large.linkage_coeff_codec.enabled` | Publish compressed coefficients. |
| `large.linkage_coeff_codec.bits_per_layer` | Shared coefficient widths; `[b]` broadcasts to all layers. |
| `large.linkage_coeff_codec.bits_per_layer_root` | Root-specific widths, overriding shared widths when nonempty. |
| `large.linkage_coeff_codec.bits_per_layer_linkage` | Linked-row widths, overriding shared widths when nonempty. |
| `large.linkage_coeff_codec.p_first_candidates` | Candidate clipping percentiles for the first coefficient layer. |
| `large.linkage_coeff_codec.p_rest_candidates` | Candidate clipping percentiles for the remaining layers. |
| `large.linkage_coeff_codec.use_weighted_quantile` | Use weighted coefficient statistics during fitting. |
| `large.linkage_coeff_codec.allow_clip` | Allow coefficient clipping to the fitted range. |
| `large.linkage_coeff_codec.fit_scale` | Fit coefficient-quantizer scales; separate from ADC radial score correction. |
| `large.linkage_coeff_codec.q_refine_sweeps` | Integer coefficient-refinement sweeps. |
| `large.linkage_coeff_codec.q_refine_max_layer` | Layer limit for coefficient refinement; zero selects the implementation's full-layer policy. |
| `large.linkage_coeff_codec.q_refine_step_limit` | Maximum integer-step displacement considered during refinement. |
| `eval.linkage.modes` | Explicit list drawn from `quantized`, `quantized_scaled`, `float`, `float_scaled`, or `[auto]` to select the available compatible representations. Parser default: `[quantized_scaled]`; dataset cfgs can request multiple modes. |
| `eval.disk_norm2_mode` | `lut` or `exact` reconstruction norms, independent of coefficient representation. |
| `eval.disk_norm2_lut_mode` | `identity`, `tail_weighted`, or `reference` norm representation. |
| `eval.disk_norm2_lut_kmeans_niter` | Norm-LUT fitting iterations. |
| `eval.linkage.gpu_norm_enable` | Use CUDA for norm preparation; independent of CPU query scanning. |
| `eval.linkage.score_mode` | `l2` (default), `cosine_recon`, or `mips_recon`. The latter two are development evaluation modes; see [scoring examples](#scoring-examples-and-development-modes). |

Keep one width per layer when using heterogeneous widths. Increasing M allocates
more code bits; increasing coefficient widths improves coefficient precision.
Virtual roots add shared representation overhead. Use the reported rate and
minimal query disk footprint to compare these allocations.

### Scoring examples and development modes

`modes` selects the coefficient/score representation. `score_mode` selects the
ranking rule. Let `h` be the reconstructed vector in the selected coefficient
representation and `q` the query. The scan minimizes the following scores:

| Setting | Ranking score, omitting query-only constants |
|---|---|
| `score_mode = "l2"`, `modes = [quantized]` or `[float]` | `norm2(h) - 2 q·h` |
| `score_mode = "l2"`, a scaled mode with `disk_norm2_mode = "lut"` | `bias[label] - 2 scale[label] q·h` |
| `score_mode = "l2"`, a scaled mode with `disk_norm2_mode = "exact"` | `norm2(x) - 2 (norm(x) / norm(h)) q·h`, using the original vector `x` |
| `score_mode = "cosine_recon"` | `-q·h / norm(h)`; query normalization is constant for ranking within one query. |
| `score_mode = "mips_recon"` | `-q·h`; no reconstruction-length normalization. |

The formulas describe origin-based scoring; reference-LUT mode applies the
corresponding reference correction. Both scaled modes use label tables with
`"lut"`, and per-row radial correction with `"exact"`; `float_scaled` refers
to float coefficients, not automatically to exact norms.

`eval.disk_norm2_mode = "lut"` obtains reconstruction norms from the selected
norm representation; `"exact"` uses per-row reconstruction norms.
`eval.disk_norm2_lut_mode` chooses `"identity"` for ordinary norm-LUT fitting,
`"tail_weighted"` for weighted fitting, or `"reference"` for reference-LUT L2
scoring. CUDA norm publication supports the identity/reference paths;
tail-weighted publication uses `eval.linkage.gpu_norm_enable = false`.

For a cosine-ranking experiment, edit the evaluation fields of a dataset cfg:

```ini
advanced.eval_only = true
io.linkage_file = "/data/index/linkage_list"
eval.base.enabled = false
eval.linkage.enabled = true
eval.linkage.modes = [quantized]
eval.linkage.score_mode = "cosine_recon"
eval.disk_norm2_mode = "lut"
eval.disk_norm2_lut_mode = "identity"
```

Use `"mips_recon"` in the same example for maximum-inner-product ranking.
The exact accepted names are `cosine_recon` and `mips_recon`, not `angle` or
`mips`. Both work with `[float]`, `[quantized]`, or `[auto]`; scaled
representations and the reference norm LUT use the L2 score definition.

These are development scoring options, separate from the paper's L2 recipes.
They change final candidate scoring, not the L2 reconstruction training
objective or the IVF routing rule. Evaluate them against ground truth for the
chosen metric. For cosine, the norm is the reconstruction's length, not the
original vector's length; the scaled representations implement a different
radial correction.

For an M6 coefficient-width experiment, a single entry broadcasts to all six
layers; a six-entry array sets each layer separately:

```ini
large.linkage_coeff_codec.bits_per_layer = [6]
large.linkage_coeff_codec.bits_per_layer_root = [5, 5, 5, 5, 5, 5]
large.linkage_coeff_codec.bits_per_layer_linkage = [4, 5, 5, 5, 5, 5]
```

An empty root/linkage override (`[]`) restores the shared widths. Changing
coefficient widths during eval can derive a new codec from retained float
coefficients, without retraining the model.

## Evaluation and CPU search

| Field | Purpose |
|---|---|
| `eval.linkage.nprobe` | IVF lists visited; accepts a scalar or array for a curve. |
| `eval.linkage.ivf_probe_mode` | `exact` center search or `hnsw` routing. |
| `eval.linkage.ivf_hnsw_M`, `eval.linkage.ivf_hnsw_ef_construction` | IVF routing-graph connectivity and construction budget. |
| `eval.linkage.ivf_hnsw_ef_search` | HNSW routing search budget; accepts an array. |
| `eval.linkage.warmup`, `eval.linkage.repeat` | Untimed warmup count and measured repetition count. |
| `eval.linkage.query_block` | Number of queries processed per block. |
| `eval.linkage.parent_louds_native_eval` | Decode parents during scanning; false materializes the parent array during loading. |
| `eval.linkage.pretransform_query_codebooks` | Precompute source-space codebooks to remove per-query rotation; default true. |
| `eval.linkage.rerank_candidates` | Number of candidates reranked with original vectors; default zero. |
| `eval.linkage.rerank_storage` | Original-vector access for reranking: `memory` or `disk`. |
| `eval.base.nprobe`, `eval.base.warmup`, `eval.base.repeat` | Corresponding controls for basic-only evaluation. |
| `eval.base.use_ivf` | Default true: use basic IVF routing and `eval.base.nprobe`. False scans all basic lists. `eval.base.enabled` independently turns basic evaluation on/off. |
| `eval.metric_mode` | Select recall reporting; mode 1 includes top-k set overlap. |
| `eval.metric_gt_topks` | Ground-truth cutoffs for reporting, e.g. `[1,10,50,100]`. |
| `eval.bench.quiet` | Reduce per-point console detail while retaining recall/QPS output; default false. |
| `eval.linkage.archive_eval_result` | Write evaluation-result archives; default true. Log archiving is controlled separately by `large.archive_log`. |

For HNSW probe routing, set:

```ini
eval.linkage.ivf_probe_mode = "hnsw"
eval.linkage.ivf_hnsw_M = 32
eval.linkage.ivf_hnsw_ef_construction = 200
eval.linkage.ivf_hnsw_ef_search = [64, 128]
eval.linkage.nprobe = [1, 4, 16, 64]
```

For exact L2 reranking, `rerank_candidates` is the ADC candidate count before
returning `dataset.k` neighbors. Set it to at least `dataset.k`; zero disables
reranking. Original vectors come from `dataset.base_path`. Choose
`rerank_storage = "memory"` for resident vectors or `"disk"` for disk access,
with `score_mode = "l2"` and `large.profile_timing = false`.

## Resources

The ordinary CUDA pool and initialization pool each default to **12 in the
supplied configs**. They are separate from the four cross-cluster workers.
Use fewer lanes and smaller buffers for a smaller GPU. Query search is CPU-based;
its thread count and affinity are also cfg-controlled.

| Field | Purpose |
|---|---|
| `runtime.use_cuda`, `runtime.cuda_device` | Enable construction acceleration and select the device. |
| `runtime.cuda_pool_size`, `runtime.cuda_pool_size_init_linkage` | Ordinary and initialization CUDA lane counts. |
| `runtime.cuda_mode`, `runtime.cuda_allow_tf32` | `fast` enables TF32; `strict` follows the explicit `cuda_allow_tf32` flag. To disable TF32, use `strict` with false. Supplied configs use `fast` with true. |
| `runtime.omp_threads` | CPU worker count; select according to available physical cores. |
| `runtime.omp_places` | `""`, `none`, `cores`, `threads`, or `sockets`; supplied configs use `cores`. |
| `runtime.omp_proc_bind` | `""`, `none`, `close`, `spread`, or `primary`; supplied configs use `spread`. |
| `large.train_block`, `large.base_block` | Training/base input-block sizes in vectors. |
| `runtime.cuda_linkage_cross_cluster_batch` | Aggregate independent ready tasks from different clusters. |
| `runtime.cuda_linkage_cross_cluster_workers` | Number of cross-cluster scheduler workers. |
| `stlqpp.linkage_scheduler.stable` | Use stable scheduling. |
| `runtime.c_one_update_shards`, `runtime.c_one_update_shards_init_linkage` | Edge-moment accumulation shards during ordinary/initialization updates. |
| `stlqpp.root_update.accumulator_partitions` | Root-update accumulation partitions; zero selects automatically. |

### CUDA workspace and dispatch

Each initialization-specific key below applies to the initial linkage stage;
its unsuffixed counterpart controls ordinary linkage.

| Field | Purpose |
|---|---|
| `runtime.cuda_linkage_eval_async_pinned_mb`, `runtime.cuda_linkage_eval_async_pinned_mb_init_linkage` | Pinned staging-buffer budget in MiB. |
| `runtime.cuda_linkage_large_root_xc0_chunk_mb`, `runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage` | Large-root projection chunk budget in MiB. |
| `runtime.cuda_linkage_mem_budget_mb`, `runtime.cuda_linkage_mem_budget_mb_init_linkage` | Linkage workspace sizing budget in MiB. |
| `runtime.cuda_linkage_many_nodes_max_pairs`, `runtime.cuda_linkage_many_nodes_max_pairs_init_linkage` | Candidate-pair limit for multi-node batches. |
| `runtime.cuda_linkage_chunk_max_pairs`, `runtime.cuda_linkage_chunk_max_pairs_init_linkage` | Additional candidate-pair chunk limit; zero selects automatic sizing. |
| `runtime.cuda_linkage_single_gpu_min_candidates`, `runtime.cuda_linkage_single_gpu_max_candidates` | Candidate-count bounds for the single-node GPU dispatch path. |
| `runtime.cuda_linkage_inner_many_nodes_target_nodes` | Target nodes per inner multi-node batch. |
| `runtime.cuda_linkage_many_nodes_preflush_target_pairs` | Multi-node preflush target in candidate pairs. |
| `runtime.cuda_linkage_same_layer_block_nodes` | Same-layer node-block size. |
| `runtime.cuda_linkage_same_layer_single_gpu_min_candidates` | Same-layer single-node GPU threshold. |
| `runtime.cuda_linkage_same_layer_window_min_pairs`, `runtime.cuda_linkage_same_layer_window_max_pairs` | Lower/upper candidate-pair bounds for a same-layer window. |
| `runtime.cuda_linkage_same_layer_window_max_pending_nodes` | Pending-node limit for a same-layer window. |
| `runtime.cuda_linkage_same_layer_preflush_target_pairs`, `runtime.cuda_linkage_same_layer_preflush_target_nodes` | Same-layer early-flush targets. |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable`, `runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable` | Enable tiny-tail CPU evaluation in ordinary/initialization linkage. |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs`, `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes` | Tiny-tail dispatch limits. |
| `runtime.cuda_linkage_bad_frozen_window_nodes`, `runtime.cuda_linkage_bad_frozen_window_max_pairs` | Node/pair bounds for BAD frozen windows. |

### Streaming and list preparation

| Field | Purpose |
|---|---|
| `runtime.basic_inflight_blocks` | Maximum basic blocks in flight. |
| `runtime.basic_async_io`, `runtime.basic_async_io_depth`, `runtime.basic_async_io_mb` | Basic input prefetch, queue depth and buffer budget. |
| `runtime.basic_async_write`, `runtime.basic_async_write_depth`, `runtime.basic_async_write_mb` | Basic output pipeline, queue depth and buffer budget. |
| `runtime.linkage_async_io`, `runtime.linkage_async_io_depth` | Linkage input prefetch and queue depth. |
| `large.cluster_bucket_size` | Number of CIDs grouped into an intermediate bucket. |
| `large.bucket_flush_mb` | Bucket buffering/flush budget in MiB. |
| `large.base_list_build_threads` | Workers preparing base-list order. |

## Initialization details

| Field | Purpose |
|---|---|
| `train.kmeans_init` | Initial-center method: `kmeansll`, `kmeans++`, or `random`. |
| `train.kmeans_iters`, `train.kmeans_tol` | Maximum k-means iterations and stopping tolerance. |
| `train.kmeansll_rounds`, `train.kmeansll_oversample` | k-means\|\| sampling rounds and oversampling. |
| `train.kmeansll_candidate_cap` | Candidate cap; zero uses automatic policy. |
| `train.kmeansll_seed` | k-means\|\| seed override. |
| `train.kmeans_initial_weight`, `train.kmeans_min_weight` | Initial/minimum adaptive sample weights. |
| `train.kmeans_outlier_quantile` | Quantile used to identify high-error samples. |
| `train.kmeans_cost_threshold` | Cost-change threshold for the adaptive-weight schedule. |
| `train.kmeans_annealing_factor` | Weight annealing factor. |
| `train.kmeans_warmup_iters` | Warmup iterations before adaptive weighting. |
| `train.kmeans_quantile_bins` | Histogram resolution for quantile estimation. |
| `train.kmeans_device_cache_mb` | Normalized training-data device-cache budget; 20,480 MiB in supplied configs. Allocates up to the data size and is released after RVQ initialization. |
| `train.kmeans_large_k_threshold` | Dictionary size at which large-K initialization is selected. |
| `train.kmeans_large_k_top_coarse` | Number of coarse candidates for large-K assignment. |
| `train.kmeans_large_k_hier2_enable` | Enable two-level large-K k-means. |
| `train.kmeans_large_k_fixed_k1` | Fixed coarse partition count; zero uses automatic selection. |
| `train.kmeans_large_k_k1_min`, `train.kmeans_large_k_k1_max` | Bounds on the automatically selected coarse count. |
| `train.kmeans_large_k_k1_pow2` | Restrict that count to powers of two. |
