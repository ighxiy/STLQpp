# Quick start

Run the following commands from the repository root. See the
[README](README.md) for CPU-only builds, model reuse, checkpoints, and measurement
interpretation.

For Windows, follow the [MSVC build and run guide](docs/WINDOWS.md).
It covers CPU and CUDA builds with either oneMKL or OpenBLAS.

## 1. Build

With GCC, OpenMP, oneMKL, and the CUDA Toolkit installed:

```bash
cmake -S . -B build-release-lto \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQpp_ENABLE_LTO=ON \
  -DSTLQpp_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTLQpp_CBLAS_BACKEND=MKL \
  -DSTLQpp_LAPACK_BACKEND=MKL
cmake --build build-release-lto --target stlqpp_build_index -j 8
```

Replace `89` with your GPU's compute capability without the decimal point.
Set `MKLROOT` if oneMKL is installed outside the standard location.

## 2. Prepare SIFT1M

Extract [SIFT1M](http://corpus-texmex.irisa.fr/) into `data/sift/` with these files:

```text
sift_learn.fvecs     sift_base.fvecs
sift_query.fvecs     sift_groundtruth.ivecs
```

For other datasets, follow [DATASETS.md](docs/DATASETS.md).

## 3. Train, build, and preview recall

```bash
./build-release-lto/stlqpp_build_index --config configs/sift1m_m6.cfg
```

To use another data directory or fewer CPU cores, append:

```text
--set 'dataset.data_root="/path/to/sift"' --set runtime.omp_threads=16
```

The cfg uses 12 ordinary CUDA lanes and 12 initialization lanes. GPU construction
and CPU query evaluation run in sequence. Logs print to the terminal; the program
also archives them. The final log and `run_state.txt` identify the model and
`linkage_list` directory under `outputs/SIFT1M/`.

## 4. Measure a recall/QPS curve

Set `INDEX` to the `linkage_list` path printed by the completed build:

```bash
INDEX=/absolute/path/to/linkage_list
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set advanced.eval_only=true \
  --set "io.linkage_file=\"$INDEX\"" \
  --set 'eval.linkage.nprobe=[1,2,4,8,16,32,64,128,256]' \
  --set 'eval.linkage.modes=[quantized,quantized_scaled]' \
  --set eval.linkage.warmup=1 --set eval.linkage.repeat=5 \
  --set eval.metric_mode=1 --set 'eval.metric_gt_topks=[1,10,50,100]' \
  --set runtime.omp_threads=16 \
  --set 'runtime.omp_places="cores"' --set 'runtime.omp_proc_bind="spread"'
```

Use the same data-root override here if it was changed during construction.
Each probe is measured in this invocation. Native result files are written to
`eval_result/`; logs stay in `log/`.

## 5. Common changes

| Goal | Setting |
|---|---|
| Larger code length | Select the dataset's second cfg. |
| RSD3 linkage | `base.linkage.rsd_rank=3` |
| Native LOUDS scanning | `eval.linkage.parent_louds_native_eval=true` |
| Basic recall | `eval.base.enabled=true` during a run retaining/building basic codes |
| Keep basic codes for another linkage recipe | `large.cleanup.preset="dev_all"` |
| Lower GPU memory use | Reduce both CUDA pool sizes, then block/cache sizes. |
| Run entirely on CPU | Use a CPU build, or set `runtime.use_cuda=false` in a CUDA build. |

See [CONFIGURATION.md](docs/CONFIGURATION.md) for the exact resource keys and
[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common setup problems.
