# Windows build and run

Build with MSVC, CMake, and OpenMP. Select oneMKL or OpenBLAS for host-side
BLAS/LAPACK; enable CUDA for GPU construction, which uses cuBLAS.
The commands below use MSVC 19.44, CMake 3.31.6, CUDA 13.1, and OpenBLAS 0.3.30.
CUDA sources use C++20.

Recommended dependencies:

- Visual Studio 2022 Build Tools compatible with your CUDA Toolkit. If you need a specific/current VS 2022 release,
  use Microsoft's release-history installer page:
  <https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#updating-your-installation-to-a-specific-release>.
  In the installer, select:
  - MSVC v143 x64/x86 build tools
  - Windows 10/11 SDK
  - C++ CMake tools for Windows
- NVIDIA CUDA Toolkit.
- Intel oneAPI MKL or OpenBLAS.

### Known Windows CUDA Limitation

Windows builds are supported, including CUDA builds. However, large linkage build can issue many small GPU tasks
when cluster-local candidate batches are fragmented. Based on the observed behavior on a `consumer` 3090 GPU setup
and the Windows display-driver execution model, these small-task launches may have much higher latency than the same
path on Linux. We have not validated this with a `professional` NVIDIA card or a compute-driver/TCC-style setup, so the
limitation should be treated as an inferred environment issue rather than a resolved STLQpp kernel issue. For serious
large-linkage benchmarking, Linux remains the recommended environment.

## Environment

Run these commands in **cmd.exe** from the repository root:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
chcp 65001
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
```

Use the same UTF-8 console encoding for configuring and building so Ninja can
recognize MSVC's header-dependency output. Adjust installation paths below to
match your machine.

Enable Windows **Win32 long paths** for the hashed cache directories:
`LongPathsEnabled=1` under
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`.

## CUDA + oneMKL

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;%PATH%"
"%CMAKE%" -S . -B build-win-cuda-mkl -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DSTLQpp_ENABLE_CUDA=ON ^
  -DSTLQpp_CBLAS_BACKEND=MKL -DSTLQpp_LAPACK_BACKEND=MKL ^
  -DSTLQpp_MKL_ROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/latest" ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe"
"%CMAKE%" --build build-win-cuda-mkl -j 8
```

`86` targets the RTX 3090. Set `CMAKE_CUDA_ARCHITECTURES` for your GPU.

## CPU + oneMKL

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
"%CMAKE%" -S . -B build-win-cpu-mkl -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DSTLQpp_ENABLE_CUDA=OFF ^
  -DSTLQpp_CBLAS_BACKEND=MKL -DSTLQpp_LAPACK_BACKEND=MKL
"%CMAKE%" --build build-win-cpu-mkl -j 8
```

## CPU + OpenBLAS

Use an x64 OpenBLAS distribution containing CBLAS and LAPACKE headers, an MSVC
import library, and runtime DLLs. Set `STLQpp_OPENBLAS_ROOT` to its root and add
its `bin` directory to `PATH`:

```bat
set "PATH=D:\code\OpenBLAS-0.3.30-x64\bin;%PATH%"
"%CMAKE%" -S . -B build-win-cpu-openblas -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DSTLQpp_ENABLE_CUDA=OFF ^
  -DSTLQpp_CBLAS_BACKEND=OPENBLAS -DSTLQpp_LAPACK_BACKEND=OPENBLAS ^
  -DSTLQpp_OPENBLAS_ROOT="D:/code/OpenBLAS-0.3.30-x64"
"%CMAKE%" --build build-win-cpu-openblas -j 8
```

## CUDA + OpenBLAS

```bat
set "PATH=D:\code\OpenBLAS-0.3.30-x64\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;%PATH%"
"%CMAKE%" -S . -B build-win-cuda-openblas -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DSTLQpp_ENABLE_CUDA=ON ^
  -DSTLQpp_CBLAS_BACKEND=OPENBLAS -DSTLQpp_LAPACK_BACKEND=OPENBLAS ^
  -DSTLQpp_OPENBLAS_ROOT="D:/code/OpenBLAS-0.3.30-x64" ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe"
"%CMAKE%" --build build-win-cuda-openblas -j 8
```

## Run SIFT1M

For example, place the complete SIFT1M dataset in `E:\sift`:

```text
E:\sift\
  sift_learn.fvecs        100,000 training vectors
  sift_base.fvecs         1,000,000 base vectors
  sift_query.fvecs        10,000 queries
  sift_groundtruth.ivecs  ground truth for all queries
```

The supplied `configs/sift1m_m6.cfg` uses M=6, IVF256, 25 training rounds, and
RSD2. It trains the model, builds the index, and evaluates `quantized` and
`quantized_scaled` at `nprobe=256`.

Run the selected executable in the environment initialized above:

```bat
build-win-cuda-mkl\stlqpp_build_index.exe --config configs\sift1m_m6.cfg ^
  --dataset.data_root E:/sift ^
  --large.output_dir outputs/windows/cuda-mkl --large.tmp_dir tmp/windows/cuda-mkl

build-win-cpu-mkl\stlqpp_build_index.exe --config configs\sift1m_m6.cfg ^
  --dataset.data_root E:/sift ^
  --large.output_dir outputs/windows/cpu-mkl --large.tmp_dir tmp/windows/cpu-mkl

build-win-cpu-openblas\stlqpp_build_index.exe --config configs\sift1m_m6.cfg ^
  --dataset.data_root E:/sift ^
  --large.output_dir outputs/windows/cpu-openblas --large.tmp_dir tmp/windows/cpu-openblas

build-win-cuda-openblas\stlqpp_build_index.exe --config configs\sift1m_m6.cfg ^
  --dataset.data_root E:/sift ^
  --large.output_dir outputs/windows/cuda-openblas --large.tmp_dir tmp/windows/cuda-openblas
```

CPU builds automatically use CPU execution and CPU norm preparation. Set
`runtime.use_cuda=false` to select CPU execution in a CUDA build. CUDA builds
also select CPU execution when no CUDA device is visible.

Executables are in their respective build directories. Models, indexes,
evaluation results, and logs are under `outputs/windows/<backend>/SIFT1M/`;
working files are under `tmp/windows/<backend>/SIFT1M/`.

