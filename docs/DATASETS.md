# Datasets and input preparation

All supplied recipes evaluate squared L2 distance against zero-based ground-truth
IDs. Dataset preprocessing defines the metric: GloVe is row-normalized; MSong,
Contriever, and OpenAI-ArXiv preserve the supplied vector values. Apply the same
preprocessing and base membership when comparing methods.

`dataset.data_root` is the directory containing the prepared files. Relative
`dataset.train_path`, `base_path`, `query_path`, and `groundtruth_path` are resolved
under it. Absolute paths can select files stored elsewhere. Output/tmp paths
are relative to the process working directory, independently of the data root.

## Sources

| Dataset | Source |
|---|---|
| SIFT1M, GIST1M, BIGANN | [TexMex](http://corpus-texmex.irisa.fr/) |
| DEEP1M, 96 dimensions | [QINCo DEEP download script](https://github.com/facebookresearch/Qinco/blob/main/data/deep1b/download_data.sh), small split |
| GloVe100 | [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets), `glove-100-angular.hdf5` |
| MSong, 420 dimensions | [GQR datasets](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html) |
| Contriever1M | [QINCo Contriever download script](https://github.com/facebookresearch/Qinco/blob/main/data/contriever/download_data.sh), `ContrieverEmb` |
| OpenAI-ArXiv | [Big ANN dataset definitions](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py), `arxiv-openaiv2-2M` |

The configs describe the following layouts. Filenames can be changed in the cfg;
conversion is only needed when the file format or preprocessing changes.

## SIFT1M and GIST1M

```text
data/sift/                        data/gist/
  sift_learn.fvecs                  gist_learn.fvecs
  sift_base.fvecs                   gist_base.fvecs
  sift_query.fvecs                  gist_query.fvecs
  sift_groundtruth.ivecs            gist_groundtruth.ivecs
```

Use the published learn/base/query/GT files. SIFT uses 100,000 training rows;
GIST uses 500,000. Both use a one-million-row base. Query counts are 10,000 and
1,000 respectively. Preserve the original values.

## DEEP1M

```text
data/deep1m/
  learn.fvecs
  base.fvecs
  deep1B_queries.fvecs
  deep1M_groundtruth.ivecs
```

Use the 96-dimensional DEEP1B small split: one million base vectors and its
matching 10,000-query GT. (not 256-dimensional version)

Run the linked QINCo download script with **`-small`** from the QINCo repository
root:

```bash
bash data/deep1b/download_data.sh -small
```

Without `-small`, that script downloads the 350M-row training source and the
one-billion-row base. The small release provides `learn10M10k.fvecs` as
`learn.fvecs`, containing 10,010,000 training rows; the supplied configs read only
its first 500,000 rows. Point the train/base fields at the downloaded filenames,
or create links with the names above. The formats are fvecs/ivecs.

## Contriever1M

```text
data/contriever/
  training_set_1M.npy
  database1M.npy
  queries.npy
  ground_truth1M.npy
```

Use the QINCo `ContrieverEmb` release and its L2 ground truth. The cfg reads
500,000 training vectors; `dataset.train_path` can point directly to the released
`training_set.npy` instead of a local `training_set_1M.npy`. No extra normalization
is applied. Native NPY loading accepts C-contiguous float16/float32 vectors and
int32/int64 GT; vectors are converted to FP32 while reading.

The released `training_set.npy` is much larger than the training prefix needed
here. Our local `training_set_1M.npy` was prepared with **HTTP Range requests**,
without downloading the complete training array. The helper first reads the NPY
header, then downloads only the payload for the first 1,000,000 rows. It writes a
new, valid NPY header with shape `(1000000, 768)`, preserving the source FP16 dtype,
row order and vector bytes. This file is approximately 1.43 GiB; the supplied cfg
uses its first 500,000 rows. There is no resampling or extra normalization, and
the base/query/GT files are unchanged.

```bash
python3 -m pip install numpy requests
python3 scripts/download_contriever_train.py \
  --rows 1000000 --output data/contriever/training_set_1M.npy
```

This is the same range-download method as `curl --range`, followed by rebuilding
the local NPY header. **Do not simply truncate the original file**: its header
would still declare the full remote array shape. The helper requires HTTP 206
responses and stops if the server ignores Range, checks the downloaded payload
length, and refuses to overwrite an existing output. The remaining release files
can be downloaded with the linked QINCo script's individual commands; running
that entire script also downloads the full training array.

## GloVe100: normalized, disjoint train/base

From `glove-100-angular.hdf5`:

1. Read the HDF5 `train` array (1,183,514 vectors) and `test` array (10,000 queries).
2. Convert to FP32 and divide each row by its L2 norm.
3. Use the first 300,000 normalized database rows for training.
4. Use the remaining 883,514 rows as the search base, preserving their order.
5. Recompute exact L2 top-100 GT on this reduced base. IDs start at zero in the
   reduced base, rather than retaining the original HDF5 row IDs.

```text
data/glove100/
  glove100_train_300k.fvecs
  glove100_base_after_train_300k.fvecs
  glove100_query.fvecs
  glove100_groundtruth_after_train_300k.ivecs
```

## MSong: remove query self-matches, then split

Start with the GQR MSong base, query, and ground-truth files. Preserve the original
420-dimensional vector values; this recipe does not normalize them.

1. Remove all database rows exactly matching a query, and near-identical query
   matches among the original GT candidates using `numpy.allclose` with
   `rtol=1e-5`, `atol=1e-6`.
2. Preserve the remaining row order. The prepared source has 992,565 rows after
   removing 1,620 rows from the 994,185-row source.
3. Use its first 300,000 rows for training and the remaining 692,565 for the base.
4. Recompute L2 top-100 GT for the original 1,000 queries against that base.

```text
data/msong/
  msong_train_300k.fvecs
  msong_base_noself_after_train_300k.fvecs
  msong_query.fvecs
  msong_groundtruth_noself_after_train_300k.ivecs
```

## OpenAI-ArXiv: 500K disjoint training split

Use the `arxiv-openaiv2-2M` source, with 2,321,096 database vectors, 20,000 queries,
and dimension 1536. This Big ANN benchmark encodes arXiv abstracts with OpenAI's
`text-embedding-ada-002`; it is distinct from the Qdrant DBpedia datasets encoded
with `text-embedding-3-large`. The original vectors are approximately unit length;
preserve their FP32 values. The source download names are `openai_base.bin` and
`openai_query.bin`, both in the flat binary format below. The `openai1536` prefix
in the prepared filenames is a legacy name for this same OpenAI-ArXiv dataset.

Use database rows `[0:500000]` for training and `[500000:]` for the base.
Recompute exact L2 top-100 GT against the resulting 1,821,096-row base.

```text
data/openai-arxiv/
  openai1536_train_500k.fbin
  openai1536_base_after_train_500k.fbin
  openai_query.fbin
  openai1536_groundtruth_after_train_500k.ibin
```

## Prepare the three custom splits

The helper uses NumPy and CPU Faiss; GloVe also uses h5py. These Python packages
are only needed for data preparation, not for building or running STLQ++.

```bash
python3 -m pip install numpy faiss-cpu h5py
python3 scripts/prepare_dataset.py glove \
  --source /data/raw/glove-100-angular.hdf5 --output data/glove100
python3 scripts/prepare_dataset.py msong \
  --source /data/raw/msong_base.fvecs \
  --query /data/raw/msong_query.fvecs \
  --gt /data/raw/msong_groundtruth.ivecs --output data/msong
python3 scripts/prepare_dataset.py openai \
  --source /data/raw/openai_base.bin \
  --query /data/raw/openai_query.bin --output data/openai-arxiv
```

The helper writes the cfg filenames, a preparation manifest, and exact top-100
L2 GT. Exact search holds a FP32 copy of the prepared base in RAM; OpenAI needs
about 11 GiB for that copy alone. GT construction is a one-time preparation step.
`--threads` selects the Faiss CPU thread count. Existing output files are kept;
use a new output directory for a different preparation.

## BIGANN / SIFT1B

```text
data/bigann/
  learn.bvecs
  bigann_base.bvecs
  queries.bvecs
  bigann_gnd/idx_1000M.ivecs
```

Use the full one-billion-row base, 30 million training rows, and 10,000 queries.
The published files may be named `bigann_learn.bvecs` and `bigann_query.bvecs`;
change the two cfg paths or create links for `learn.bvecs` and `queries.bvecs`.
Both configs use 65,536 IVF centers and HNSW probe routing. Place `large.tmp_dir`
on a disk with room for basic codes, list-order data, and linkage publication
workspace in addition to the final index.

## File formats and GT

| Format | Layout |
|---|---|
| `fvecs` | Per row: little-endian int32 dimension, then FP32 components |
| `bvecs` | Per row: little-endian int32 dimension, then uint8 components |
| `ivecs` | Per row: little-endian int32 count, then int32 IDs |
| `fbin` / float32 `xbin` | uint32 row count, uint32 dimension, then row-major FP32 components |
| `ibin` | uint32 row count, uint32 neighbor count, then row-major 32-bit IDs |
| `npy` | C-order NumPy array; choose the native `npy` input format |

Select formats with `large.train_format`, `large.base_format`,
`large.query_format`, and `large.gt_format`. GT must contain at least the requested
metric cutoffs, sorted nearest first. The supplied files use zero-based IDs and
`dataset.groundtruth_add1=false`. If you change base membership, normalization,
or order, regenerate GT against the resulting base.
