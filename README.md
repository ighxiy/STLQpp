# STLQ++: Efficient Index Construction and Search with Topological Linkage Vector Quantization

STLQ++ builds compact vector indexes using topological linkage: a vector is
represented by a root reconstruction or by a previously encoded parent plus a
coded residual. RSD searches the residual support, ICM/ILS refine basic codes,
and alternating codebook and rotation updates train the representation. Search
combines IVF routing with CPU ADC scanning, compressed coefficients, and optional
radial score correction.

The executable `stlqpp_build_index` provides training, base encoding, index
publication, and recall/QPS evaluation. Construction supports CPU and CUDA;
query scanning runs on the CPU. CPU and CUDA builds use the same artifact formats.

- [Quick start](GUIDELINE.md): build, run SIFT1M, and evaluate an index.
- [Datasets](docs/DATASETS.md): download sources, file layouts, and preprocessing.
- [Configuration](docs/CONFIGURATION.md): parameters and resource settings.
- [MIPS results](docs/MIPS.md): Although STLQ++ is not specially designed for MIPS, we compares MIPS results with ScaNN and LSQ++.
- [Rate allocation guide](docs/RATE_ALLOCATION.md): General performance of rate allocation between codes and coefficients.
- [IVF sensitivity](docs/IVF_SENSITIVITY.md) compares recall and throughput across five IVF list counts at two code rates.
- [Troubleshooting](docs/TROUBLESHOOTING.md): build, memory, paths, and evaluation.

## Build

Use an x86-64 CPU with AVX2/FMA, a C++20-capable GCC or MSVC compiler, CMake
3.18 or newer, OpenMP, and either Intel oneMKL or OpenBLAS with LAPACKE. CUDA
construction also uses the NVIDIA CUDA Toolkit and a compatible driver.

### CUDA build

From the repository root:

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

Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of your GPU, without the
decimal point. The default is `89`. Set `MKLROOT` or pass
`-DSTLQpp_MKL_ROOT=/path/to/mkl` for a non-system oneMKL installation. To select
nvcc explicitly, add `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc`.
OpenBLAS/LAPACKE can also be used with CUDA: set both backend options to `OPENBLAS`.

### CPU-only build

For example, on Debian/Ubuntu the CPU dependencies can be installed with:

```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapacke-dev
```

Then build:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQpp_ENABLE_LTO=ON \
  -DSTLQpp_ENABLE_CUDA=OFF \
  -DSTLQpp_CBLAS_BACKEND=OPENBLAS \
  -DSTLQpp_LAPACK_BACKEND=OPENBLAS
cmake --build build-cpu --target stlqpp_build_index -j 8
```

For a custom OpenBLAS installation, add
`-DSTLQpp_OPENBLAS_ROOT=/path/to/openblas`. CPU builds automatically select CPU
execution and CPU norm preparation. In a CUDA build, `runtime.use_cuda=false`
or no visible CUDA device selects the same CPU path.

For Windows, see the [Windows build and run guide](docs/WINDOWS.md) for MSVC CPU and
CUDA builds with either oneMKL or OpenBLAS. With a multi-configuration
generator, append `--config Release` to the build command; the executable is in
the build directory's `Release/` subdirectory.

## First run: SIFT1M

Download and extract SIFT1M from the [TexMex collection](http://corpus-texmex.irisa.fr/).
Place these files in `data/sift/`:

```text
data/sift/
  sift_learn.fvecs
  sift_base.fvecs
  sift_query.fvecs
  sift_groundtruth.ivecs
```

Run from the repository root:

```bash
./build-release-lto/stlqpp_build_index --config configs/sift1m_m6.cfg
```

This initializes and trains the model, encodes the base, builds the linkage
index, and evaluates `quantized` and `quantized_scaled` at `nprobe=256`.
The cfg selects 25 training rounds, 100,000 training vectors, and 1,000,000 base
vectors. Basic evaluation is disabled; final linkage evaluation is enabled.

For data stored elsewhere:

```bash
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set 'dataset.data_root="/path/to/sift"'
```

For CPU-only execution:

```bash
./build-cpu/stlqpp_build_index \
  --config configs/sift1m_m6.cfg
```

Logs remain visible in the terminal and are archived by the program.

## Choose a configuration

The paired cfg files select two code lengths for each dataset. Each cfg contains
its input filenames, four random seeds, topology settings, coefficient widths,
and evaluation mode. All small-dataset recipes use IVF256; BIGANN uses IVF65536.

`model.m` sets the number of codebooks (2–24). Training, base encoding, and
the AVX2 search scan kernels have compile-time specializations for **M = 5, 6,
10, and 12**. Other supported values use runtime-width kernels, including a
scalar search scan, so training and encoding can take longer and search QPS
can be lower than at nearby specialized values. For a frequently used M,
developers can specialize the
performance-critical kernels for that width and rebuild.

| Dataset | Dimension | M | Training rows | Base rows | Rounds | Config files |
|---|---:|---|---:|---:|---:|---|
| SIFT1M | 128 | 6 / 10 | 100,000 | 1,000,000 | 25 | [M6](configs/sift1m_m6.cfg), [M10](configs/sift1m_m10.cfg) |
| DEEP1M | 96 | 6 / 10 | 500,000 | 1,000,000 | 25 | [M6](configs/deep1m_m6.cfg), [M10](configs/deep1m_m10.cfg) |
| GIST1M | 960 | 5 / 10 | 500,000 | 1,000,000 | 15 | [M5](configs/gist1m_m5.cfg), [M10](configs/gist1m_m10.cfg) |
| GloVe100 | 100 | 6 / 12 | 300,000 | 883,514 | 25 | [M6](configs/glove100_train300k_m6.cfg), [M12](configs/glove100_train300k_m12.cfg) |
| MSong | 420 | 5 / 10 | 300,000 | 692,565 | 25 | [M5](configs/msong_noself_train300k_m5.cfg), [M10](configs/msong_noself_train300k_m10.cfg) |
| Contriever1M | 768 | 6 / 12 | 500,000 | 1,000,000 | 15 | [M6](configs/contriever1m_m6.cfg), [M12](configs/contriever1m_m12.cfg) |
| OpenAI-ArXiv | 1536 | 6 / 12 | 500,000 | 1,821,096 | 15 | [M6](configs/openai_arxiv_train500k_m6.cfg), [M12](configs/openai_arxiv_train500k_m12.cfg) |
| BIGANN / SIFT1B | 128 | 6 / 10 | 30,000,000 | 1,000,000,000 | 25 | [M6](configs/sift1b_m6_65k_30m_10615_161520.cfg), [M10](configs/sift1b_m10_65k_30m_10615_161520.cfg) |

The supplied configs use RSD2 for both basic encoding and base linkage.
To compare RSD2 and RSD3 with one trained model, change only
`base.linkage.rsd_rank` during base construction.

Resource defaults are **12 ordinary CUDA lanes, 12 initialization lanes, four
cross-cluster workers, and a 20,480 MiB k-means device-cache budget**. The cache
allocates only the required data up to this limit and is released after RVQ
initialization. Adjust CPU threads
and memory settings for your machine using the [resource guide](docs/CONFIGURATION.md#resources).

## Choosing a rate allocation

At a fixed storage budget, a longer code leaves fewer bits for its coefficients.
Compare the measured curves for five datasets and use a small coefficient-width
sweep to choose a configuration. [Rate allocation guide](docs/RATE_ALLOCATION.md)

## Configure a run

Cfg files use `key = value`, quoted strings, arrays such as `[1,2,4]`, and `#`
comments. They are organized into dataset settings, common controls, and detailed
subsystem settings. Override cfg values directly with `--key value` after
`--config`:

```bash
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --io.pre_fix sift_rsd3 \
  --base.linkage.rsd_rank 3 \
  --runtime.omp_threads 16
```

`--set key=value` is also supported: `--runtime.omp_threads 16` is equivalent
to `--set runtime.omp_threads=16`. Later assignments take precedence.

Run `stlqpp_build_index --help` for command-line usage.
To save the resolved configuration without starting training or evaluation:

```bash
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg --dump-config effective.cfg
```

| Task | Settings |
|---|---|
| Train, build, evaluate | Use the supplied cfg. |
| Train only | Turn off both base stages and both evaluation stages. |
| Encode a base using an existing model | Set `io.train_file`; disable initialization and training. |
| Rebuild linkage from retained basic lists | Also set `io.base_file` and disable `base.encode.enabled`. |
| Evaluate an existing index | Set `advanced.eval_only=true` and `io.linkage_file`. |

`base.encode.enabled` controls basic encoding; `base.linkage.enabled` controls
linkage construction. `eval.base.enabled` and `eval.linkage.enabled` control
evaluation independently.

### Train only

```bash
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set base.encode.enabled=false \
  --set base.linkage.enabled=false \
  --set eval.base.enabled=false \
  --set eval.linkage.enabled=false
```

### Build from a trained model

Use the final `.stlqmodel` reported in the completed run's log or `run_state.txt`:

```bash
MODEL=/absolute/path/to/final.stlqmodel
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set train.init_enabled=false --set train.enabled=false \
  --set "io.train_file=\"$MODEL\"" \
  --set base.linkage.rsd_rank=3
```

To reuse basic encoding for several linkage recipes, use
`--set 'large.cleanup.preset="dev_all"'` on the first build. For the next build,
set `io.base_file` to that run's retained **base-list directory**, keep
`io.train_file` unchanged, and set `base.encode.enabled=false`:

```bash
BASE_LIST=/absolute/path/to/base-list
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set train.init_enabled=false --set train.enabled=false \
  --set "io.train_file=\"$MODEL\"" \
  --set "io.base_file=\"$BASE_LIST\"" \
  --set base.encode.enabled=false --set base.linkage.enabled=true \
  --set base.linkage.rsd_rank=3
```

`io.base_file` identifies encoded basic lists; `dataset.base_path` identifies
original vectors. Retained input lists can be reused while the model and basic
recipe remain the same.

## Optimize the paper’s disk storage

The paper reports the minimum on-disk search index, including the FP32 rotation
matrix. Commit `3315f05` reproduces that storage. Latest version has been optimized.
After training, or when loading a trained checkpoint, STLQ++ applies the inverse
rotation once to the IVF centroids and all root and linkage codewords before
base encoding. Basic encoding, linkage construction, and search then operate
in the original vector coordinates. The index stores these codebooks directly,
saving **4d² bytes** per search model, e.g., **9 MiB** on OpenAI-ArXiv.
Training checkpoints retain the learned rotation.

## Evaluate recall and query throughput

Set `INDEX` to a published `linkage_list` directory. Eval-only loads the model
associated with that index; it also accepts an explicit `io.train_file`.
Keep the dataset paths and coefficient widths in the cfg for that index.

```bash
INDEX=/absolute/path/to/linkage_list
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set advanced.eval_only=true \
  --set "io.linkage_file=\"$INDEX\"" \
  --set 'eval.linkage.modes=[quantized,quantized_scaled]' \
  --set 'eval.linkage.nprobe=[1,2,4,8,16,32,64,128,256]' \
  --set eval.linkage.warmup=1 --set eval.linkage.repeat=5 \
  --set eval.metric_mode=1 --set 'eval.metric_gt_topks=[1,10,50,100]' \
  --set runtime.omp_threads=16 \
  --set 'runtime.omp_places="cores"' --set 'runtime.omp_proc_bind="spread"'
```

The probe array is processed in one invocation. Index loading and norm/scale
preparation happen before measured queries. Warmup is untimed; repeated query
runs report individual and aggregate timings, including median QPS. Use one
active benchmark process and the same physical CPU allocation when comparing
methods. For a quick recall preview, use `warmup=0`, `repeat=1`, and one probe.

| Representation | Coefficients | Distance scoring |
|---|---|---|
| `quantized` | Compressed integer coefficients | ADC with the selected reconstruction-norm representation |
| `quantized_scaled` | Same compressed coefficients | ADC with fitted radial scale/bias correction |
| `float` | Retained FP32 coefficients | ADC with the selected reconstruction-norm representation |
| `float_scaled` | Retained FP32 coefficients | Radially corrected float-coefficient scoring |

`eval.disk_norm2_mode="lut"` uses norm labels and center tables;
`"exact"` uses exact reconstruction norms for the selected coefficients.
For LUT fitting, `identity`, `tail_weighted`, and `reference` select different
norm representations. MSong configs use `reference`. The score correction
uses base-vector geometry and does not require calibration queries.

Additional evaluation options:

- **Native LOUDS:** `--set eval.linkage.parent_louds_native_eval=true` decodes
  parent pointers during scanning. The default materializes a parent array at
  index load time. Both use the same on-disk topology.
- **Basic recall:** enable `eval.base.enabled` while building/retaining the basic
  artifact, and choose `eval.base.nprobe`. It measures the independent basic
  representation before linkage.
- **Exact reranking:** set `eval.linkage.rerank_candidates` to a positive
  candidate count and keep the original base vectors available. The default is
  zero, which evaluates the compressed index without reranking.
- **New coefficient widths:** eval-only can derive a new compressed
  representation from retained float coefficients. It prepares the associated
  norms and scale tables in a separate cache generation.

`R@k` measures whether the true nearest neighbor occurs among the first k
returned IDs. `Top-k@k` measures overlap with the true top-k set. `dataset.k`
is the number of returned neighbors; it is separate from `nprobe`, the number
of visited IVF lists.

### BIGANN routing

The BIGANN configs use HNSW to select IVF lists. Run a curve with, for example:

```bash
./build-release-lto/stlqpp_build_index \
  --config configs/sift1b_m10_65k_30m_10615_161520.cfg \
  --set advanced.eval_only=true \
  --set "io.linkage_file=\"$INDEX\"" \
  --set 'eval.linkage.nprobe=[16,32,64,128,256,384,512,1024]' \
  --set eval.linkage.ivf_probe_mode=hnsw \
  --set eval.linkage.ivf_hnsw_ef_search=2048
```

The IVF routing graph is separate from the optional BAD-cluster candidate graph
used during construction. Their respective options are `eval.linkage.ivf_hnsw_*`
and `base.linkage.bad_cluster_knn_backend` / `hnsw.*`.

## Outputs and checkpoints

Paths below are relative to the working directory unless made absolute in the cfg:

```text
outputs/<dataset>/
  models/                         # published .stlqmodel files
  <prefix>__0x<identity>/          # run namespace
    log/                          # native execution logs
    run_state.txt                 # selected model, progress and artifact paths
    linkage_list/                 # published search index
    eval_result/                  # recall, timing and storage reports

tmp/<dataset>/.../                # basic/list staging and construction workspace
```

The program prints the actual model, basic, linkage, and codec identities.
Different training, base, or codec recipes use the corresponding identity paths.
`io.pre_fix` supplies the readable prefix. A separate prefix or output root is
also useful for independently named runs of the same recipe.

`large.cleanup.preset="eval_both"` retains final quantized and float-coefficient
representations and removes consumed construction intermediates. `"dev_all"`
retains the basic/list intermediates for reuse. An explicitly supplied
`io.base_file` is an input, and is not removed by a later linkage run's cleanup.
Keep the original base vectors available for deriving missing scaled tables and
for exact reranking.

Enable periodic training checkpoints with `train.ckpt.enabled=true` and
`train.ckpt.every_R=5`. Resume from a completed round directory as follows:

```bash
ROUND=/absolute/path/to/round_00000009
./build-release-lto/stlqpp_build_index \
  --config configs/sift1m_m6.cfg \
  --set train.init_enabled=false --set train.enabled=true \
  --set "io.train_file=\"$ROUND\""
```

`train.max_R_iters` remains the total requested round count. To encode a base
without continuing training, use the exported `.stlqmodel` and disable training
as in the model-reuse example. BIGANN configs additionally enable
`large.base_basic_checkpoint` and `large.linkage_checkpoint`; restart with the
same recipe and output/tmp locations to reuse completed work.

## Read the measurements

- **Construction:** native stage logs report initialization, training, basic
  encoding, list-order preparation, linkage construction, and norm/codec work.
  Use the stage boundaries when comparing construction costs.
- **Recall/QPS:** result files contain the selected representation, probe count,
  recall cutoffs, and query timing. QPS covers query execution, including routing
  and query-table construction; loading and derived-cache preparation are separate.
- **Rate:** `payload_bit` and `total_bit` describe the encoded representation.
  `payload_bit` keeps the payload-only budget. `total_bit` includes IVF,
  amortized virtual roots, coefficient records, raw ParentLOUDS, and the active
  scoring data listed below. These are additional bits per vector within
  `total_bit`, not the total rate; they are excluded from `payload_bit`.
  `eval.disk_norm2_mode` selects the column independently of the `float` or
  `quantized` coefficient representation. Shared lookup tables are excluded
  from this logical per-vector budget. `ivf_bit` is already included in `total_bit`.

  | Scoring mode | `lut` | `exact` |
  |---|---|---|
  | Plain L2 | 8-bit label | 32-bit FP32 squared reconstruction norm |
  | Scaled L2 | 8-bit label into a gain/bias table | 64 bits: FP32 gain + FP32 original squared-norm bias, 32 bits each |
  | Cosine | 8-bit label | 32-bit FP32 squared reconstruction norm |
  | Plain MIPS | 0 bits | 0 bits |
  | Scaled MIPS | 8-bit label into a gain table | 32-bit FP32 gain |

  Typical payload targets, with suitable coefficient widths, are:

  | M | Approximate payload | Bytes |
  |---|---:|---:|
  | 5 / 6 | 64 bits | 8 |
  | 10 / 12 | 128 bits | 16 |
  | 15 / 18 | 192 bits | 24 |
  | 20 / 24 | 256 bits | 32 |

  These are rate families, not a fixed M-to-byte mapping. Coefficient widths,
  linkage and virtual-root counts affect the actual rate; use the reported
  `payload_bit` and `total_bit`. 
- **Disk:** `minimal_query_disk_bytes` reports the files required by the active
  query representation, including model, norm/scale, and routing artifacts.
  Temporary files, logs, and retained alternative representations are separate.

For multi-seed experiments, record the four seeds `train.seed`,
`train.linkage.seed`, `base.encode.seed`, and `base.linkage.seed` together with the
resolved cfg, executable build settings, and native results. Keep the data split,
query set, norm mode, and evaluation resources fixed across comparisons.

## Source layout

| Directory | Contents |
|---|---|
| `applications/` | Unified CLI, build orchestration, evaluation and reporting |
| `include/stlqpp/`, `src/` | Training, RSD, topology, codecs, storage and search |
| `configs/` | Dataset recipes and resource settings |
| `cmake/` | Compiler, BLAS/LAPACK, OpenMP and CUDA integration |
| `third_party/hnswlib/` | HNSW routing and candidate construction |

## License

STLQ++ is released under the [BSD 3-Clause License](LICENSE).
Third-party licenses and provenance are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
