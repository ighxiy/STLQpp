#include "stlqpp/training/initialization/cuda_kmeans_parallel_seeder.h"

#include "stlqpp/training/initialization/spherical_rvq_log.h"

#include "stlqpp/runtime/cuda/device_buffer.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"
#include "stlqpp/runtime/cuda/status.h"
#include "stlqpp/training/initialization/cuda_spherical_assignment.h"
#include "stlqpp/training/initialization/spherical_kmeans_plus_plus.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint64_t kRoundIncrement = 0x9e3779b97f4a7c15ULL;
__global__ void FillBestKernel(float* best, std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    best[index] = -FLT_MAX;
  }
}

__global__ void IotaKernel(std::uint32_t* indices, std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    indices[index] = index;
  }
}

__global__ void GatherColumnsKernel(const float* source,
                                    std::uint32_t dimension,
                                    const std::uint32_t* indices,
                                    std::uint32_t columns,
                                    float* output) {
  const std::size_t count = static_cast<std::size_t>(dimension) * columns;
  for (std::size_t value =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       value < count;
       value += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint32_t column =
        static_cast<std::uint32_t>(value / dimension);
    const std::uint32_t axis = static_cast<std::uint32_t>(value % dimension);
    output[value] = source[
        static_cast<std::size_t>(indices[column]) * dimension + axis];
  }
}

__global__ void MergeScoreKernel(const float* scores,
                                 std::uint32_t center_count,
                                 std::uint32_t columns,
                                 std::uint32_t sample_offset,
                                 float* best) {
  const std::uint32_t column =
      static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  const float* score = scores +
      static_cast<std::size_t>(column) * center_count;
  float value = best[sample_offset + column];
  for (std::uint32_t center = 0U; center < center_count; ++center) {
    value = fmaxf(value, score[center]);
  }
  best[sample_offset + column] = value;
}

__global__ void ComputeDistanceKernel(const float* best,
                                      float* distances,
                                      std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    float dot = best[index];
    if (!(dot > -1.0e20F)) {
      dot = -1.0F;
    }
    dot = fminf(1.0F, fmaxf(-1.0F, dot));
    distances[index] = fmaxf(0.0F, 2.0F - 2.0F * dot);
  }
}

__device__ std::uint32_t Xorshift32(std::uint32_t value) {
  value ^= value << 13U;
  value ^= value >> 17U;
  value ^= value << 5U;
  return value;
}

__global__ void SampleFlagsKernel(const float* distances,
                                  std::uint8_t* flags,
                                  std::uint32_t count,
                                  float inverse_phi,
                                  float oversample,
                                  std::uint64_t seed) {
  constexpr float kInverse2To32 = 1.0F / 4294967296.0F;
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    const float probability = fminf(
        1.0F, fmaxf(0.0F, oversample * distances[index] * inverse_phi));
    if (probability <= 0.0F) {
      flags[index] = 0U;
      continue;
    }
    std::uint32_t value = static_cast<std::uint32_t>(
        seed ^ (static_cast<std::uint64_t>(index) * kRoundIncrement));
    value = Xorshift32(value);
    const float uniform =
        (static_cast<float>(value) + 1.0F) * kInverse2To32;
    flags[index] = uniform < probability ? 1U : 0U;
  }
}

std::uint32_t BlocksFor(std::uint64_t count) noexcept {
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      65'535U, std::max<std::uint64_t>(1U, (count + kThreads - 1U) /
                                               kThreads)));
}

template <typename Scalar>
StatusOr<std::uint64_t> FillSample(
    CudaContext& context,
    InitializationBasicBlockSourceFactory<Scalar>& source_factory,
    CudaSphericalAssignment<Scalar>& residual_builder,
    std::uint64_t source_rows,
    std::uint32_t dimension,
    std::uint32_t sample_count,
    std::mt19937& random,
    CudaDeviceBuffer<float>* sample) {
  std::uint64_t sample_start = 0U;
  if (source_rows > sample_count) {
    std::uniform_int_distribution<std::int64_t> start_distribution(
        0, static_cast<std::int64_t>(source_rows - sample_count));
    sample_start = static_cast<std::uint64_t>(start_distribution(random));
  }
  auto source = source_factory.OpenRange(sample_start, sample_count);
  if (!source.ok() && source.status().code() == StatusCode::kUnimplemented) {
    sample_start = 0U;
    source = source_factory.OpenRange(0U, sample_count);
  }
  if (!source.ok()) {
    return source.status();
  }
  if (const Status status = sample->Ensure(
          static_cast<std::size_t>(dimension) * sample_count); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  std::uint64_t expected = sample_start;
  std::uint32_t filled = 0U;
  while (filled < sample_count) {
    auto next = source.value()->Next();
    if (!next.ok()) {
      return next.status();
    }
    if (!next.value().has_value()) {
      return Status::DataLoss(
          "kmeans|| initialization source ended before its sample window");
    }
    BasicInputBlock<Scalar>& block = *next.value();
    if (block.first_id != expected || block.dimension != dimension ||
        block.count == 0U) {
      return Status::DataLoss(
          "kmeans|| initialization source order or shape changed");
    }
    const std::uint32_t take =
        std::min(block.count, sample_count - filled);
    block.count = take;
    block.values.resize(static_cast<std::size_t>(take) * dimension);
    expected += take;
    auto residual = residual_builder.PrepareResidualsDevice(block);
    if (!residual.ok()) {
      return residual.status();
    }
    const std::size_t values =
        static_cast<std::size_t>(dimension) * block.count;
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync kmeans|| sample D2D",
            cudaMemcpyAsync(
                sample->data() + static_cast<std::size_t>(filled) * dimension,
                residual.value().values, values * sizeof(float),
                cudaMemcpyDeviceToDevice, stream)); !status.ok()) {
      return status;
    }
    filled += take;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return sample_start;
}

Status UpdateBestDots(CudaContext& context,
                      const CudaDeviceBuffer<float>& sample,
                      std::uint32_t dimension,
                      std::uint32_t sample_count,
                      const std::vector<std::uint32_t>& new_indices,
                      CudaKmeansParallelSeedOptions options,
                      CudaDeviceBuffer<std::uint32_t>* indices,
                      CudaDeviceBuffer<float>* gathered,
                      CudaDeviceBuffer<float>* scores,
                      CudaDeviceBuffer<float>* best) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  for (std::size_t first = 0U; first < new_indices.size();
       first += options.candidate_batch) {
    const std::uint32_t batch = static_cast<std::uint32_t>(std::min(
        static_cast<std::size_t>(options.candidate_batch),
        new_indices.size() - first));
    if (const Status status = indices->Ensure(batch); !status.ok()) {
      return status;
    }
    if (const Status status = gathered->Ensure(
            static_cast<std::size_t>(dimension) * batch); !status.ok()) {
      return status;
    }
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync kmeans|| candidate indices H2D",
            cudaMemcpyAsync(indices->data(), new_indices.data() + first,
                            batch * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream)); !status.ok()) {
      return status;
    }
    GatherColumnsKernel<<<BlocksFor(
                              static_cast<std::uint64_t>(dimension) * batch),
                          kThreads, 0U, stream>>>(
        sample.data(), dimension, indices->data(), batch, gathered->data());
    if (const Status status = CudaRuntimeStatus(
            "GatherColumnsKernel kmeans|| candidates", cudaPeekAtLastError());
        !status.ok()) {
      return status;
    }

    const std::uint32_t maximum_tile =
        std::min(sample_count, options.score_tile_columns);
    if (const Status status = scores->Ensure(
            static_cast<std::size_t>(batch) * maximum_tile); !status.ok()) {
      return status;
    }
    const float one = 1.0F;
    const float zero = 0.0F;
    for (std::uint32_t sample_first = 0U; sample_first < sample_count;
         sample_first += maximum_tile) {
      const std::uint32_t tile =
          std::min(maximum_tile, sample_count - sample_first);
      if (const Status status = CudaGemmF32(
              context, CudaGemmOperation::kTranspose,
              CudaGemmOperation::kNone, static_cast<int>(batch),
              static_cast<int>(tile), static_cast<int>(dimension), one,
              gathered->data(), static_cast<int>(dimension),
              sample.data() +
                  static_cast<std::size_t>(sample_first) * dimension,
              static_cast<int>(dimension), zero, scores->data(),
              static_cast<int>(batch)); !status.ok()) {
        return status;
      }
      MergeScoreKernel<<<BlocksFor(tile), kThreads, 0U, stream>>>(
          scores->data(), batch, tile, sample_first, best->data());
      if (const Status status = CudaRuntimeStatus(
              "MergeScoreKernel kmeans||", cudaPeekAtLastError());
          !status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

StatusOr<float> ComputePhi(CudaContext& context,
                           std::uint32_t count,
                           CudaDeviceBuffer<float>* best,
                           CudaDeviceBuffer<float>* distances,
                           CudaDeviceBuffer<float>* phi,
                           CudaDeviceBuffer<std::uint8_t>* cub_scratch) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  ComputeDistanceKernel<<<BlocksFor(count), kThreads, 0U, stream>>>(
      best->data(), distances->data(), count);
  if (const Status status = CudaRuntimeStatus(
          "ComputeDistanceKernel kmeans||", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  std::size_t scratch_bytes = 0U;
  if (const Status status = CudaRuntimeStatus(
          "cub reduce query kmeans||",
          cub::DeviceReduce::Sum(nullptr, scratch_bytes, distances->data(),
                                 phi->data(), count, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = cub_scratch->Ensure(scratch_bytes); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cub reduce kmeans||",
          cub::DeviceReduce::Sum(cub_scratch->data(), scratch_bytes,
                                 distances->data(), phi->data(), count,
                                 stream)); !status.ok()) {
    return status;
  }
  float host_phi = 0.0F;
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync kmeans|| phi D2H",
          cudaMemcpyAsync(&host_phi, phi->data(), sizeof(float),
                          cudaMemcpyDeviceToHost, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return host_phi;
}

__global__ void SelectPrefixIndexKernel(const float* prefix,
                                        std::uint32_t count,
                                        float threshold,
                                        std::uint32_t* selected) {
  std::uint32_t first = 0U;
  std::uint32_t last = count;
  while (first < last) {
    const std::uint32_t middle = first + (last - first) / 2U;
    if (prefix[middle] < threshold) {
      first = middle + 1U;
    } else {
      last = middle;
    }
  }
  selected[0] = min(first, count - 1U);
}

StatusOr<std::uint32_t> SelectKmeansPlusPlusIndex(
    CudaContext& context,
    const CudaDeviceBuffer<float>& distances,
    std::uint32_t count,
    float threshold,
    CudaDeviceBuffer<float>* prefix,
    CudaDeviceBuffer<std::uint32_t>* selected,
    CudaDeviceBuffer<std::uint8_t>* scratch) {
  if (const Status status = prefix->Ensure(count); !status.ok()) {
    return status;
  }
  if (const Status status = selected->Ensure(1U); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  std::size_t scratch_bytes = 0U;
  if (const Status status = CudaRuntimeStatus(
          "cub kmeans++ scan query",
          cub::DeviceScan::InclusiveSum(
              nullptr, scratch_bytes, distances.data(), prefix->data(),
              count, stream)); !status.ok()) {
    return status;
  }
  if (const Status status = scratch->Ensure(scratch_bytes); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cub kmeans++ scan",
          cub::DeviceScan::InclusiveSum(
              scratch->data(), scratch_bytes, distances.data(),
              prefix->data(), count, stream)); !status.ok()) {
    return status;
  }
  SelectPrefixIndexKernel<<<1U, 1U, 0U, stream>>>(
      prefix->data(), count, threshold, selected->data());
  if (const Status status = CudaRuntimeStatus(
          "SelectPrefixIndexKernel", cudaPeekAtLastError()); !status.ok()) {
    return status;
  }
  std::uint32_t host_index = 0U;
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync kmeans++ index D2H",
          cudaMemcpyAsync(&host_index, selected->data(),
                          sizeof(host_index), cudaMemcpyDeviceToHost,
                          stream)); !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return host_index;
}

StatusOr<std::vector<std::uint32_t>> SelectCandidates(
    CudaContext& context,
    const CudaDeviceBuffer<float>& distances,
    const CudaDeviceBuffer<std::uint32_t>& all_indices,
    std::uint32_t sample_count,
    float phi,
    float oversample,
    std::uint64_t seed,
    std::uint32_t maximum,
    CudaDeviceBuffer<std::uint8_t>* flags,
    CudaDeviceBuffer<std::uint32_t>* selected,
    CudaDeviceBuffer<std::uint32_t>* selected_count,
    CudaDeviceBuffer<std::uint8_t>* cub_scratch) {
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  SampleFlagsKernel<<<BlocksFor(sample_count), kThreads, 0U, stream>>>(
      distances.data(), flags->data(), sample_count, 1.0F / phi,
      oversample, seed);
  if (const Status status = CudaRuntimeStatus(
          "SampleFlagsKernel kmeans||", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  std::size_t scratch_bytes = 0U;
  if (const Status status = CudaRuntimeStatus(
          "cub select query kmeans||",
          cub::DeviceSelect::Flagged(
              nullptr, scratch_bytes, all_indices.data(), flags->data(),
              selected->data(), selected_count->data(), sample_count,
              stream)); !status.ok()) {
    return status;
  }
  if (const Status status = cub_scratch->Ensure(scratch_bytes); !status.ok()) {
    return status;
  }
  if (const Status status = CudaRuntimeStatus(
          "cub select kmeans||",
          cub::DeviceSelect::Flagged(
              cub_scratch->data(), scratch_bytes, all_indices.data(),
              flags->data(), selected->data(), selected_count->data(),
              sample_count, stream)); !status.ok()) {
    return status;
  }
  std::uint32_t host_count = 0U;
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync kmeans|| selected count D2H",
          cudaMemcpyAsync(&host_count, selected_count->data(),
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                          stream)); !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  host_count = std::min(host_count, maximum);
  std::vector<std::uint32_t> output(host_count);
  if (host_count != 0U) {
    if (const Status status = CudaRuntimeStatus(
            "cudaMemcpyAsync kmeans|| selected indices D2H",
            cudaMemcpyAsync(output.data(), selected->data(),
                            host_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream)); !status.ok()) {
      return status;
    }
    if (const Status status = context.Synchronize(); !status.ok()) {
      return status;
    }
  }
  return output;
}

StatusOr<std::vector<float>> GatherToHost(
    CudaContext& context,
    const CudaDeviceBuffer<float>& sample,
    std::uint32_t dimension,
    const std::vector<std::uint32_t>& host_indices,
    CudaDeviceBuffer<std::uint32_t>* indices,
    CudaDeviceBuffer<float>* gathered) {
  if (const Status status = indices->Ensure(host_indices.size());
      !status.ok()) {
    return status;
  }
  const std::size_t values =
      static_cast<std::size_t>(dimension) * host_indices.size();
  if (const Status status = gathered->Ensure(values); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync kmeans|| gather indices H2D",
          cudaMemcpyAsync(indices->data(), host_indices.data(),
                          host_indices.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream)); !status.ok()) {
    return status;
  }
  GatherColumnsKernel<<<BlocksFor(values), kThreads, 0U, stream>>>(
      sample.data(), dimension, indices->data(),
      static_cast<std::uint32_t>(host_indices.size()), gathered->data());
  if (const Status status = CudaRuntimeStatus(
          "GatherColumnsKernel kmeans|| output", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  std::vector<float> output(values);
  if (const Status status = CudaRuntimeStatus(
          "cudaMemcpyAsync kmeans|| gathered D2H",
          cudaMemcpyAsync(output.data(), gathered->data(),
                          values * sizeof(float), cudaMemcpyDeviceToHost,
                          stream)); !status.ok()) {
    return status;
  }
  if (const Status status = context.Synchronize(); !status.ok()) {
    return status;
  }
  return output;
}

}  // namespace

template <typename Scalar>
StatusOr<CudaKmeansParallelSeedResult>
CudaKmeansParallelSeeder<Scalar>::Seed(
    CudaContext& context,
    InitializationBasicBlockSourceFactory<Scalar>& source,
    const SphericalRvqInitializationRequest& request,
    ArrayView<const ColumnMajorMatrixF32View> prior_codebooks,
    std::uint32_t layer,
    const RvqAssignmentStore& assignments,
    std::mt19937& random,
    const ParallelForExecutor& parallel,
    CudaKmeansParallelSeedOptions options) {
  if (const Status status = request.recipe.Validate(); !status.ok()) {
    return status;
  }
  const std::uint64_t sample64 =
      request.recipe.EffectiveSampleVectors(request.source_rows);
  const std::uint32_t center_count = layer == 0U ? request.shape.nlist : 256U;
  if (layer >= request.shape.m ||
      request.source_rows != assignments.row_count() || sample64 == 0U ||
      sample64 > static_cast<std::uint64_t>(
          std::numeric_limits<int>::max()) ||
      options.maximum_score_bytes < sizeof(float) ||
      options.candidate_batch == 0U || options.score_tile_columns == 0U) {
    return Status::InvalidArgument("CUDA kmeans|| seed request is invalid");
  }
  const std::uint32_t sample_count = static_cast<std::uint32_t>(sample64);
  const std::uint32_t candidate_cap =
      request.recipe.EffectiveCandidateCap(center_count, sample_count);
  if (request.recipe.seed_method == SphericalSeedMethod::kKmeansParallel) {
    LogSphericalRvqSeedStart({
      .gpu = true,
      .sample_count = sample_count,
      .center_count = center_count,
      .requested_rounds = request.recipe.kmeans_parallel_rounds,
      .oversample_factor = request.recipe.kmeans_parallel_oversample,
      .candidate_cap = candidate_cap,
    });
  }

  auto residual_builder = CudaSphericalAssignment<Scalar>::Create(
      context, request.shape, prior_codebooks, layer,
      layer == 0U ? nullptr : &assignments,
      {.maximum_score_bytes = options.maximum_score_bytes});
  if (!residual_builder.ok()) {
    return residual_builder.status();
  }

  const std::uint64_t base_seed =
      request.recipe.seed_method != SphericalSeedMethod::kKmeansParallel
      ? 0U : request.recipe.kmeans_parallel_seed == 0U
      ? (static_cast<std::uint64_t>(random()) << 32U) ^ random()
      : request.recipe.kmeans_parallel_seed;
  const auto sample_started = std::chrono::steady_clock::now();
  CudaDeviceBuffer<float> sample;
  auto sample_start = FillSample<Scalar>(
      context, source, residual_builder.value(), request.source_rows,
      request.shape.dimension, sample_count, random, &sample);
  if (!sample_start.ok()) {
    return sample_start.status();
  }
  const double sample_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sample_started).count();

  if (request.recipe.seed_method != SphericalSeedMethod::kKmeansParallel) {
    const auto seed_started = std::chrono::steady_clock::now();
    std::uniform_int_distribution<int> sample_index(
        0, static_cast<int>(sample_count - 1U));
    std::vector<std::uint32_t> indices;
    indices.reserve(center_count);
    if (request.recipe.seed_method == SphericalSeedMethod::kRandom) {
      std::vector<std::uint32_t> permutation(sample_count);
      std::iota(permutation.begin(), permutation.end(), 0U);
      std::shuffle(permutation.begin(), permutation.end(), random);
      for (std::uint32_t center = 0U; center < center_count; ++center) {
        indices.push_back(permutation[center % sample_count]);
      }
    } else {
      CudaDeviceBuffer<float> best;
      CudaDeviceBuffer<float> distances;
      CudaDeviceBuffer<float> phi;
      CudaDeviceBuffer<float> prefix;
      CudaDeviceBuffer<std::uint32_t> selected;
      CudaDeviceBuffer<std::uint32_t> index_scratch;
      CudaDeviceBuffer<float> gathered;
      CudaDeviceBuffer<float> scores;
      CudaDeviceBuffer<std::uint8_t> scratch;
      for (const Status& status : {
               best.Ensure(sample_count), distances.Ensure(sample_count),
               phi.Ensure(1U)}) {
        if (!status.ok()) return status;
      }
      const cudaStream_t stream =
          static_cast<cudaStream_t>(context.native_stream());
      FillBestKernel<<<BlocksFor(sample_count), kThreads, 0U, stream>>>(
          best.data(), sample_count);
      if (const Status status = CudaRuntimeStatus(
              "initialize kmeans++ distances", cudaPeekAtLastError());
          !status.ok()) return status;
      indices.push_back(static_cast<std::uint32_t>(sample_index(random)));
      for (std::uint32_t center = 1U; center < center_count; ++center) {
        if (const Status status = UpdateBestDots(
                context, sample, request.shape.dimension, sample_count,
                {indices.back()}, options, &index_scratch, &gathered,
                &scores, &best); !status.ok()) return status;
        auto total = ComputePhi(
            context, sample_count, &best, &distances, &phi, &scratch);
        if (!total.ok()) return total.status();
        std::uint32_t chosen =
            static_cast<std::uint32_t>(sample_index(random));
        if (total.value() > 1.0e-6F) {
          std::uniform_real_distribution<float> threshold(
              0.0F, total.value());
          auto sampled = SelectKmeansPlusPlusIndex(
              context, distances, sample_count, threshold(random),
              &prefix, &selected, &scratch);
          if (!sampled.ok()) return sampled.status();
          chosen = sampled.value();
        }
        indices.push_back(chosen);
      }
    }
    CudaDeviceBuffer<std::uint32_t> index_scratch;
    CudaDeviceBuffer<float> gathered;
    auto centers = GatherToHost(
        context, sample, request.shape.dimension, indices,
        &index_scratch, &gathered);
    if (!centers.ok()) return centers.status();
    std::vector<std::uint32_t> fallback_indices(center_count);
    for (std::uint32_t& index : fallback_indices) {
      index = static_cast<std::uint32_t>(sample_index(random));
    }
    auto fallback = GatherToHost(
        context, sample, request.shape.dimension, fallback_indices,
        &index_scratch, &gathered);
    if (!fallback.ok()) return fallback.status();
    const double seed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seed_started).count();
    return CudaKmeansParallelSeedResult{
        .centers = std::move(centers).value(),
        .empty_cluster_fallback = std::move(fallback).value(),
        .statistics = {
            .sample_start = sample_start.value(),
            .sample_count = sample_count,
            .reduced_candidates = request.recipe.seed_method ==
                    SphericalSeedMethod::kRandom
                ? std::min(center_count, sample_count) : sample_count,
            .sample_wall_seconds = sample_wall_seconds,
            .candidate_wall_seconds = seed_seconds,
        },
    };
  }

  const auto candidate_started = std::chrono::steady_clock::now();
  std::uniform_int_distribution<int> sample_index(
      0, static_cast<int>(sample_count - 1U));
  std::vector<std::uint32_t> candidates{
      static_cast<std::uint32_t>(sample_index(random))};
  std::vector<std::uint8_t> seen(sample_count, 0U);
  seen[candidates.front()] = 1U;
  std::uint32_t picked_raw = 1U;
  std::uint32_t picked_new = 1U;

  CudaDeviceBuffer<float> best;
  CudaDeviceBuffer<float> distances;
  CudaDeviceBuffer<float> phi;
  CudaDeviceBuffer<std::uint32_t> all_indices;
  CudaDeviceBuffer<std::uint8_t> flags;
  CudaDeviceBuffer<std::uint32_t> selected;
  CudaDeviceBuffer<std::uint32_t> selected_count;
  CudaDeviceBuffer<std::uint32_t> index_scratch;
  CudaDeviceBuffer<float> gathered;
  CudaDeviceBuffer<float> scores;
  CudaDeviceBuffer<std::uint8_t> cub_scratch;
  if (const Status status = best.Ensure(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = distances.Ensure(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = phi.Ensure(1U); !status.ok()) {
    return status;
  }
  if (const Status status = all_indices.Ensure(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = flags.Ensure(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = selected.Ensure(sample_count); !status.ok()) {
    return status;
  }
  if (const Status status = selected_count.Ensure(1U); !status.ok()) {
    return status;
  }
  const cudaStream_t stream =
      static_cast<cudaStream_t>(context.native_stream());
  FillBestKernel<<<BlocksFor(sample_count), kThreads, 0U, stream>>>(
      best.data(), sample_count);
  IotaKernel<<<BlocksFor(sample_count), kThreads, 0U, stream>>>(
      all_indices.data(), sample_count);
  if (const Status status = CudaRuntimeStatus(
          "initialize CUDA kmeans|| state", cudaPeekAtLastError());
      !status.ok()) {
    return status;
  }
  if (const Status status = UpdateBestDots(
          context, sample, request.shape.dimension, sample_count, candidates,
          options, &index_scratch, &gathered, &scores, &best); !status.ok()) {
    return status;
  }

  std::uint32_t rounds_ran = 0U;
  const float oversample = request.recipe.kmeans_parallel_oversample *
      static_cast<float>(center_count);
  for (std::uint32_t round = 0U;
       round < request.recipe.kmeans_parallel_rounds &&
       candidates.size() < candidate_cap;
       ++round) {
    auto phi_value = ComputePhi(
        context, sample_count, &best, &distances, &phi, &cub_scratch);
    if (!phi_value.ok()) {
      return phi_value.status();
    }
    if (!(phi_value.value() > 0.0F)) {
      break;
    }
    rounds_ran = round + 1U;
    const std::uint32_t remaining =
        candidate_cap - static_cast<std::uint32_t>(candidates.size());
    auto picked = SelectCandidates(
        context, distances, all_indices, sample_count, phi_value.value(),
        oversample, base_seed + kRoundIncrement * round, remaining, &flags,
        &selected, &selected_count, &cub_scratch);
    if (!picked.ok()) {
      return picked.status();
    }
    picked_raw += static_cast<std::uint32_t>(picked.value().size());
    std::vector<std::uint32_t> added;
    added.reserve(picked.value().size());
    for (const std::uint32_t index : picked.value()) {
      if (seen[index] != 0U) {
        continue;
      }
      seen[index] = 1U;
      candidates.push_back(index);
      added.push_back(index);
      ++picked_new;
      if (candidates.size() == candidate_cap) {
        break;
      }
    }
    if (added.empty()) {
      break;
    }
    // best/distances/phi are used only by the next sampling round. The final
    // reducer consumes gathered candidate vectors, not these device scores.
    if (candidates.size() == candidate_cap ||
        round + 1U == request.recipe.kmeans_parallel_rounds) {
      break;
    }
    if (const Status status = UpdateBestDots(
            context, sample, request.shape.dimension, sample_count, added,
            options, &index_scratch, &gathered, &scores, &best);
        !status.ok()) {
      return status;
    }
  }

  const std::uint32_t unique_candidates =
      static_cast<std::uint32_t>(candidates.size());
  while (candidates.size() < center_count) {
    candidates.push_back(static_cast<std::uint32_t>(sample_index(random)));
  }
  const std::uint32_t padded_candidates =
      static_cast<std::uint32_t>(candidates.size()) - unique_candidates;
  const std::size_t candidates_before_cap = candidates.size();
  if (candidates.size() > candidate_cap) {
    candidates.resize(candidate_cap);
  }
  const std::uint32_t capped_candidates = static_cast<std::uint32_t>(
      candidates_before_cap - candidates.size());
  LogSphericalRvqSeedFinish({
      .center_count = center_count,
      .rounds_ran = rounds_ran,
      .picked_raw = picked_raw,
      .picked_new = picked_new,
      .unique_candidates = unique_candidates,
      .padded_candidates = padded_candidates,
      .capped_candidates = capped_candidates,
      .reducer_input_candidates =
          static_cast<std::uint32_t>(candidates.size()),
  });
  const double candidate_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - candidate_started).count();
  const auto reduction_started = std::chrono::steady_clock::now();
  auto host_candidates = GatherToHost(
      context, sample, request.shape.dimension, candidates, &index_scratch,
      &gathered);
  if (!host_candidates.ok()) {
    return host_candidates.status();
  }
  auto centers = ReduceSphericalKmeansPlusPlus(
      {host_candidates.value().data(), request.shape.dimension,
       static_cast<std::uint32_t>(candidates.size())},
      center_count, random, parallel);
  if (!centers.ok()) {
    return centers.status();
  }
  const double reduction_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - reduction_started).count();

  const auto fallback_started = std::chrono::steady_clock::now();
  std::vector<std::uint32_t> fallback_indices(center_count);
  for (std::uint32_t& index : fallback_indices) {
    index = static_cast<std::uint32_t>(sample_index(random));
  }
  auto fallback = GatherToHost(
      context, sample, request.shape.dimension, fallback_indices,
      &index_scratch, &gathered);
  if (!fallback.ok()) {
    return fallback.status();
  }
  const double fallback_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - fallback_started).count();
  return CudaKmeansParallelSeedResult{
      .centers = std::move(centers).value(),
      .empty_cluster_fallback = std::move(fallback).value(),
      .statistics = {
          .sample_start = sample_start.value(),
          .sample_count = sample_count,
          .rounds = rounds_ran,
          .picked_raw = picked_raw,
          .picked_new = picked_new,
          .unique_candidates = unique_candidates,
          .padded_candidates = padded_candidates,
          .capped_candidates = capped_candidates,
          .reduced_candidates = static_cast<std::uint32_t>(candidates.size()),
          .sample_wall_seconds = sample_wall_seconds,
          .candidate_wall_seconds = candidate_wall_seconds,
          .reduction_wall_seconds = reduction_wall_seconds,
          .fallback_wall_seconds = fallback_wall_seconds,
      },
  };
}

template class CudaKmeansParallelSeeder<std::uint8_t>;
template class CudaKmeansParallelSeeder<float>;

}  // namespace stlqpp
