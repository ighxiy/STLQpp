#!/usr/bin/env python3
"""Download a row prefix of QINCo ContrieverEmb with a new NPY header.

Adapted from the original dataset preparation helper; vector payload is unchanged.
"""
import argparse
from pathlib import Path
import io
import requests
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="https://dl.fbaipublicfiles.com/QINCo/datasets/ContrieverEmb/training_set.npy")
parser.add_argument("--rows", type=int, default=1_000_000)
parser.add_argument("--output", type=Path, default=Path("data/contriever/training_set_1M.npy"))
args = parser.parse_args()
if args.rows <= 0:
    parser.error("--rows must be positive")
if args.output.exists():
    parser.error(f"output already exists: {args.output}")
args.output.parent.mkdir(parents=True, exist_ok=True)
URL = args.url
N = args.rows
OUT = str(args.output)

# ------------------------------------------------
# 1. 只下载文件最前面的 4 KB，用来解析 .npy header
# ------------------------------------------------
r = requests.get(
    URL,
    headers={"Range": "bytes=0-4095"},
    stream=True,
    timeout=60,
)

if r.status_code != 206:
    r.close()
    raise RuntimeError(
        f"服务器没有按 HTTP Range 返回数据，status={r.status_code}。"
        "为避免下载完整 28GB，程序已终止。"
    )

prefix = r.content

bio = io.BytesIO(prefix)

version = np.lib.format.read_magic(bio)

if version == (1, 0):
    shape, fortran_order, dtype = \
        np.lib.format.read_array_header_1_0(bio)
elif version in [(2, 0), (3, 0)]:
    shape, fortran_order, dtype = \
        np.lib.format.read_array_header_2_0(bio)
else:
    raise RuntimeError(f"不支持的 npy version: {version}")

data_offset = bio.tell()

print("Remote NPY information:")
print("  version       =", version)
print("  shape         =", shape)
print("  dtype         =", dtype)
print("  fortran_order =", fortran_order)
print("  data offset   =", data_offset)

if fortran_order:
    raise RuntimeError("该文件为 Fortran-order，不能直接按行进行 HTTP Range 截取。")

if dtype.hasobject:
    raise RuntimeError("object dtype 不支持。")

if N > shape[0]:
    raise ValueError(f"N={N} 超过训练集大小 {shape[0]}")

# ------------------------------------------------
# 2. 计算前 N 行对应的数据字节区间
# ------------------------------------------------
row_elements = int(np.prod(shape[1:]))
row_bytes = row_elements * dtype.itemsize

data_bytes = N * row_bytes

start = data_offset
end = start + data_bytes - 1

new_shape = (N,) + shape[1:]

print()
print("Subset:")
print("  shape =", new_shape)
print("  download bytes =", data_bytes)
print("  HTTP range =", f"{start}-{end}")

# ------------------------------------------------
# 3. 创建一个具有正确 header 的新 .npy
# ------------------------------------------------
mm = np.lib.format.open_memmap(
    OUT,
    mode="w+",
    dtype=dtype,
    shape=new_shape,
    fortran_order=False,
)

new_data_offset = mm.offset
mm.flush()
del mm

# ------------------------------------------------
# 4. 只下载需要的数据区域
# ------------------------------------------------
r = requests.get(
    URL,
    headers={"Range": f"bytes={start}-{end}"},
    stream=True,
    timeout=60,
)

if r.status_code != 206:
    r.close()
    raise RuntimeError(
        f"数据下载没有返回 HTTP 206，status={r.status_code}。"
        "程序终止，以防误下载完整文件。"
    )

downloaded = 0

with open(OUT, "r+b") as f:
    f.seek(new_data_offset)

    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
        if not chunk:
            continue

        f.write(chunk)
        downloaded += len(chunk)

        print(
            f"\rDownloaded: {downloaded / 1024**2:.1f} MiB "
            f"/ {data_bytes / 1024**2:.1f} MiB",
            end="",
            flush=True,
        )

print()

if downloaded != data_bytes:
    raise RuntimeError(
        f"下载字节数不匹配：expected={data_bytes}, got={downloaded}"
    )

# ------------------------------------------------
# 5. 验证结果
# ------------------------------------------------
x = np.load(OUT, mmap_mode="r")

print()
print("Finished:")
print("  file  =", OUT)
print("  shape =", x.shape)
print("  dtype =", x.dtype)
print("  size  =", f"{x.nbytes / 1024**3:.3f} GiB")
