# Troubleshooting

## Build and libraries

| Symptom | Action |
|---|---|
| CMake cannot find oneMKL | Set `MKLROOT` or `STLQpp_MKL_ROOT` to the directory containing `include/mkl.h` and the runtime library. |
| CMake cannot find OpenBLAS/LAPACKE | Install both development packages; select `OPENBLAS` for both backends. A custom prefix uses `STLQpp_OPENBLAS_ROOT`. |
| The build has no `stlqpp_build_index` target | Check CMake's BLAS, LAPACKE and OpenMP detection messages. These dependencies are required for the executable. |
| nvcc is not detected | Pass `CMAKE_CUDA_COMPILER=/path/to/nvcc`, or configure a separate CPU-only build. |
| `no kernel image` / invalid device function | Reconfigure `CMAKE_CUDA_ARCHITECTURES` for the GPU and rebuild. The default is `89`. |
| Compiler does not support the selected CUDA architecture | Use a CUDA Toolkit that supports the GPU, and a host compiler supported by that toolkit. |
| LTO link exhausts RAM | Reduce build parallelism, or configure `STLQpp_ENABLE_LTO=OFF`. |
| A new compiler turns a warning into an error | Inspect the warning; `STLQpp_WARNINGS_AS_ERRORS=OFF` allows building without treating warnings as errors. |
| A shared library is not found at startup | Activate the BLAS/CUDA installation environment or add its library directory to the platform loader path. |

Use a fresh build directory when changing compiler or numerical backend.
The release builds are x86-64 with AVX2/FMA kernels.

## GPU or host memory pressure

The supplied configs use 12 ordinary and 12 initialization CUDA lanes. Lower both
on smaller GPUs:

```ini
runtime.cuda_pool_size = 4
runtime.cuda_pool_size_init_linkage = 4
train.kmeans_device_cache_mb = 512
large.train_block = 50000
large.base_block = 50000
```

The lane pools have separate workspaces. Cross-cluster scheduler workers are
controlled by `runtime.cuda_linkage_cross_cluster_workers`; lower this from 4
to 2 if its workspaces are a significant part of GPU usage. The k-means cache is
an initialization resource, not a limit on the whole process's device memory.

For host memory, lower `runtime.omp_threads`, `large.base_list_build_threads`,
`runtime.basic_inflight_blocks`, and the asynchronous I/O buffer budgets. For
large indexes, `eval.linkage.parent_louds_native_eval=true` avoids the resident
materialized parent array. The search index itself is still resident in memory.

Resource changes should be saved with the effective cfg when collecting results.

## CUDA is disabled but norm preparation still requests a GPU

Training/encoding and norm preparation have separate switches. For a CPU run use:

```ini
runtime.use_cuda = false
eval.linkage.gpu_norm_enable = false
```

When prepared norm/scale caches already exist, evaluation reuses them and may
perform no new GPU work even when GPU norm preparation is selected. Query scanning
runs on the CPU.

## File not found, wrong split, or unexpected output location

- Dataset-relative paths resolve under `dataset.data_root`.
- `large.output_dir` and `large.tmp_dir` resolve from the current working directory.
- `io.train_file` selects a published model or a resumable round checkpoint.
- `io.base_file` selects an encoded base-list directory, not the original vectors.
- `io.linkage_file` selects the published `linkage_list` directory.

Use absolute artifact paths for reuse. The final run log and `run_state.txt` record
the selected files. A completed checkpoint's exported model is selected by that
record, rather than by guessing the largest numbered filename.

For eval-only, point directly to the index and retain its cfg. Automatic index
location from a model requires the matching base/topology recipe and output root.

## Existing outputs or interrupted work

A changed training/base/codec recipe receives the corresponding identity path.
For a separately named repeat of the same recipe, set a new `io.pre_fix` or
`large.output_dir`. When `large.protect_existing_outputs=true`, choose a new
namespace instead of replacing a completed run.

To resume, retain the cfg and output/tmp directories. Training resumes from the
round directory supplied through `io.train_file`; base stages reuse their
checkpoints when enabled. To rebuild only linkage, supply the retained base-list
through `io.base_file` and set `base.encode.enabled=false`.

## Disk is full during construction

Construction uses staging in addition to final query files. Check both
`large.tmp_dir` and `large.output_dir`. Use `large.cleanup.preset="eval_both"`
for final runs, or `"dev_all"` when you need retained basic data for another
linkage recipe. Move workspace to a larger volume through `large.tmp_dir` before
starting a new build. Preserve an interrupted run's needed inputs/checkpoints
before removing any staging data.

## Missing float coefficients or scaled tables

Float evaluation and coefficient-width changes use retained float coefficients;
keep `large.base_linkage_store_coeffs_f32=true` during construction. Scaled-table
preparation also needs original-vector norms, obtained from retained data or the
original base source. Eval-only derives missing representations when these
inputs are available. Point `dataset.base_path` to the original base if it has
moved. An index retaining only its compact query files can search those published
representations; rebuilding other representations requires their source data.

## Recall differs from an expected result

Check the dataset split, row order and GT first, especially for GloVe, MSong and
OpenAI-ArXiv. Then compare M, IVF count, the four seeds, training rounds, basic
refinement, linkage rank/topology, coefficient widths, and norm/scale mode.
Use full probe on a small dataset to separate routing coverage from quantization.

`R@10` (one true nearest neighbor found in ten results) is different from
`Top-10@10` (overlap of two sets of ten). `eval.metric_mode=1` with
`eval.metric_gt_topks=[1,10,50,100]` reports both types.

## QPS is unexpectedly low

Use Release, the same CPU allocation, and one benchmark at a time. Set
`runtime.omp_places="cores"` and `runtime.omp_proc_bind="spread"` for explicit
binding; use `runtime.omp_threads=1` for single-thread measurements. Enable
`eval.linkage.pretransform_query_codebooks` to move codebook transformation to
preparation. Sweep probes in one invocation and compare the same representations,
parent-access mode, and rerank setting. For large IVF counts, the BIGANN configs
select HNSW probe routing.

Compare the measured query section separately from index loading, coefficient
republishing, and norm/scale preparation. Full probe evaluates every IVF list
and can take substantially longer than a low-probe curve point.
