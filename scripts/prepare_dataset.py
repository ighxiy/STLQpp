#!/usr/bin/env python3
"""Prepare the GloVe, MSong and OpenAI-ArXiv splits used by configs/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def open_vecs(path: Path, integer: bool = False) -> np.ndarray:
    with path.open('rb') as stream:
        dim = struct.unpack('<i', stream.read(4))[0]
    dtype = '<i4' if integer else '<f4'
    return np.memmap(path, dtype=dtype, mode='r').reshape(-1, dim + 1)[:, 1:]


def open_fbin(path: Path) -> np.ndarray:
    with path.open('rb') as stream:
        rows, dim = struct.unpack('<II', stream.read(8))
    return np.memmap(path, dtype='<f4', mode='r', offset=8, shape=(rows, dim))


def write_vectors(path: Path, values, flat: bool = False,
                  normalize: bool = False, block_rows: int = 8192) -> None:
    rows, dim = values.shape
    with path.open('xb') as stream:
        if flat:
            stream.write(struct.pack('<II', rows, dim))
        for first in range(0, rows, block_rows):
            block = np.array(values[first:first + block_rows], dtype='<f4', copy=True)
            if normalize:
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError('cannot normalize a zero vector')
                block /= norms
            if flat:
                stream.write(block.tobytes())
            else:
                framed = np.empty((len(block), dim + 1), dtype='<f4')
                framed[:, 0].view('<i4')[:] = dim
                framed[:, 1:] = block
                stream.write(framed.tobytes())
    print(f'[DATA] {path}: {rows} x {dim}', flush=True)


def exact_gt(base: np.ndarray, query: np.ndarray, output: Path,
             flat: bool, threads: int) -> None:
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(base.shape[1])
    for first in range(0, len(base), 65536):
        index.add(np.ascontiguousarray(base[first:first + 65536], dtype='float32'))
    topk = min(100, len(base))
    with output.open('xb') as stream:
        if flat:
            stream.write(struct.pack('<II', len(query), topk))
        for first in range(0, len(query), 128):
            _, ids = index.search(np.ascontiguousarray(query[first:first + 128],
                                                      dtype='float32'), topk)
            ids = ids.astype('<i4')
            if flat:
                stream.write(ids.tobytes())
            else:
                framed = np.empty((len(ids), topk + 1), dtype='<i4')
                framed[:, 0] = topk
                framed[:, 1:] = ids
                stream.write(framed.tobytes())
            print(f'[GT] {first + len(ids)}/{len(query)} queries', flush=True)


def msong_keep(base: np.ndarray, query: np.ndarray, gt: np.ndarray) -> np.ndarray:
    keep = np.ones(len(base), dtype=bool)
    row_type = np.dtype((np.void, base.shape[1] * 4))
    query_keys = np.unique(np.ascontiguousarray(query).view(row_type).reshape(-1))
    for first in range(0, len(base), 8192):
        block = np.ascontiguousarray(base[first:first + 8192])
        matches = np.isin(block.view(row_type).reshape(-1), query_keys)
        keep[first:first + len(block)] &= ~matches
    for qi, candidates in enumerate(gt):
        for candidate in candidates:
            if 0 <= candidate < len(base) and np.allclose(
                    base[candidate], query[qi], rtol=1e-5, atol=1e-6):
                keep[candidate] = False
    return keep


def prepare(args: argparse.Namespace) -> None:
    # Preprocessing outputs are created in a fresh directory; sources stay read-only.
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'choose an empty output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    flat = args.dataset == 'openai'
    removed = 0
    if args.dataset == 'glove':
        import h5py
        with h5py.File(args.source, 'r') as source:
            base, query = source['train'], source['test']
            train_count = 300000
            names = ('glove100_train_300k.fvecs', 'glove100_base_after_train_300k.fvecs',
                     'glove100_query.fvecs', 'glove100_groundtruth_after_train_300k.ivecs')
            # Read HDF5 ranges in blocks without loading the complete array.
            class Rows:
                def __init__(self, dataset, begin, end):
                    self.dataset, self.begin = dataset, begin
                    self.shape = (end - begin, dataset.shape[1])
                def __getitem__(self, item):
                    return self.dataset[self.begin + item.start:
                                        self.begin + min(item.stop, self.shape[0])]
            write_vectors(args.output / names[0], Rows(base, 0, train_count), normalize=True)
            write_vectors(args.output / names[1], Rows(base, train_count, len(base)), normalize=True)
            write_vectors(args.output / names[2], query, normalize=True)
    else:
        if args.dataset == 'openai':
            base, query = open_fbin(args.source), open_fbin(args.query)
            train_count = 500000
            names = ('openai1536_train_500k.fbin', 'openai1536_base_after_train_500k.fbin',
                     'openai_query.fbin', 'openai1536_groundtruth_after_train_500k.ibin')
        else:
            base, query = open_vecs(args.source), open_vecs(args.query)
            keep = msong_keep(base, query, open_vecs(args.gt, integer=True))
            removed = int((~keep).sum())
            base = base[keep]
            train_count = 300000
            names = ('msong_train_300k.fvecs', 'msong_base_noself_after_train_300k.fvecs',
                     'msong_query.fvecs', 'msong_groundtruth_noself_after_train_300k.ivecs')
            print(f'[NOSELF] removed {removed} rows; retained {len(base)}', flush=True)
        if len(base) <= train_count:
            raise ValueError(f'expected more than {train_count} source database rows')
        write_vectors(args.output / names[0], base[:train_count], flat=flat)
        write_vectors(args.output / names[1], base[train_count:], flat=flat)
        write_vectors(args.output / names[2], query, flat=flat)
    reader = open_fbin if flat else open_vecs
    base, query = reader(args.output / names[1]), reader(args.output / names[2])
    exact_gt(base, query, args.output / names[3], flat, args.threads)
    manifest = dict(dataset=args.dataset, source=str(args.source.resolve()),
                    query_source=str(args.query.resolve()) if args.query else 'HDF5 test',
                    original_gt=str(args.gt.resolve()) if args.gt else None,
                    train_count=train_count, base_count=len(base), query_count=len(query),
                    dimension=base.shape[1], removed_self_rows=removed,
                    normalization='row_l2_float32' if args.dataset == 'glove' else 'none',
                    metric='squared_l2', gt='exact IndexFlatL2; zero-based split-base IDs',
                    outputs=dict(zip(('train', 'base', 'query', 'gt'), names)))
    (args.output / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'[DONE] {args.output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('glove', 'msong', 'openai'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--query', type=Path)
    parser.add_argument('--gt', type=Path, help='Original MSong ground truth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.dataset != 'glove' and args.query is None:
        parser.error('--query is required for msong/openai')
    if args.dataset == 'msong' and args.gt is None:
        parser.error('--gt is required for msong')
    if args.threads < 1:
        parser.error('--threads must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
