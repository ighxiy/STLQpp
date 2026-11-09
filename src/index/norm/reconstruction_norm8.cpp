#include "stlqpp/index/norm/reconstruction_norm8.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

std::uint8_t NearestCenter(const float* centers, float value) noexcept {
  const float* begin = centers;
  const float* end = centers + 256U;
  const float* upper = std::lower_bound(begin, end, value);
  if (upper == begin) {
    return 0U;
  }
  if (upper == end) {
    return 255U;
  }
  const std::uint32_t high = static_cast<std::uint32_t>(upper - begin);
  const std::uint32_t low = high - 1U;
  return value - centers[low] <= centers[high] - value
      ? static_cast<std::uint8_t>(low)
      : static_cast<std::uint8_t>(high);
}

Status ValidateNorms(ArrayView<const float> norm2) {
  for (const float value : norm2) {
    if (!std::isfinite(value) || value < 0.0F) {
      return Status::InvalidArgument(
          "reconstruction norm8 input must be finite and nonnegative");
    }
  }
  return Status::Ok();
}

float SortedQuantileValue(ArrayView<const float> sorted,
                          double quantile) noexcept {
  quantile = std::min(1.0, std::max(0.0, quantile));
  const double position = quantile * static_cast<double>(sorted.size() - 1U);
  const std::size_t low = static_cast<std::size_t>(std::floor(position));
  const std::size_t high = std::min(sorted.size() - 1U, low + 1U);
  const double fraction = position - static_cast<double>(low);
  return static_cast<float>(
      (1.0 - fraction) * static_cast<double>(sorted[low]) +
      fraction * static_cast<double>(sorted[high]));
}

std::uint64_t TailWeight(float value, float p99, float p999) noexcept {
  if (value > p999) return 16U;
  if (value > p99) return 4U;
  return 1U;
}

struct SortedPass {
  ArrayView<const float> values;
  const std::array<std::size_t, 257>* boundaries;
  std::size_t tail_begin, far_tail_begin;
  std::uint32_t first_chunk;
  std::vector<Norm8LloydAccumulator>* chunks;
};
constexpr std::size_t kSortedChunkRows = 1U << 20U;

void SumSortedChunk(void* opaque, std::uint32_t, std::uint32_t item) noexcept {
  const auto& pass = *static_cast<SortedPass*>(opaque);
  const auto first = static_cast<std::size_t>(pass.first_chunk + item) * kSortedChunkRows;
  const auto last = std::min(first + kSortedChunkRows, pass.values.size());
  std::array<std::uint64_t, 256> counts{};
  std::array<double, 256> sums{};
  const auto& bounds = *pass.boundaries;
  for (std::size_t bin = 0U; bin < 256U; ++bin) {
    auto begin = std::max(first, bounds[bin]);
    const auto end = std::min(last, bounds[bin + 1U]);
    while (begin < end) {
      const auto stop = std::min(end, begin < pass.tail_begin ? pass.tail_begin :
          begin < pass.far_tail_begin ? pass.far_tail_begin : end);
      const std::uint64_t weight = begin < pass.tail_begin ? 1U :
          begin < pass.far_tail_begin ? 4U : 16U;
      double sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (auto row = begin; row < stop; ++row)
        sum += static_cast<double>(pass.values[row]);
      sums[bin] += static_cast<double>(weight) * sum;
      counts[bin] += weight * (stop - begin);
      begin = stop;
    }
  }
  (*pass.chunks)[item].MergeBins(counts, sums);
}

}  // namespace

void Norm8LloydAccumulator::Clear() noexcept {
  counts_.fill(0U);
  sums_.fill(0.0);
}

Status Norm8LloydAccumulator::Accumulate(
    Norm8CodebookView codebook,
    ArrayView<const float> norm2,
    ReconstructionNorm8Mode mode,
    float tail_p99,
    float tail_p999) {
  return AccumulateImpl(
      codebook, norm2, mode, tail_p99, tail_p999, false);
}

Status Norm8LloydAccumulator::AccumulateValidated(
    Norm8CodebookView codebook,
    ArrayView<const float> norm2,
    ReconstructionNorm8Mode mode,
    float tail_p99,
    float tail_p999) {
  return AccumulateImpl(
      codebook, norm2, mode, tail_p99, tail_p999, true);
}

Status Norm8LloydAccumulator::AccumulateImpl(
    Norm8CodebookView codebook,
    ArrayView<const float> norm2,
    ReconstructionNorm8Mode mode,
    float tail_p99,
    float tail_p999,
    bool validated_norms) {
  if (codebook.centers == nullptr || codebook.center_count != 256U) {
    return Status::InvalidArgument(
        "norm8 Lloyd accumulation requires exactly 256 centers");
  }
  if (mode != ReconstructionNorm8Mode::kIdentityLloyd &&
      mode != ReconstructionNorm8Mode::kTailWeightedLloyd) {
    return Status::InvalidArgument("unsupported reconstruction norm8 mode");
  }
  if (!validated_norms) {
    if (const Status status = ValidateNorms(norm2); !status.ok()) {
      return status;
    }
  }
#ifndef NDEBUG
  if (validated_norms) assert(ValidateNorms(norm2).ok());
#endif
  for (const float value : norm2) {
    const std::uint8_t label = NearestCenter(codebook.centers, value);
    const std::uint64_t weight = mode ==
            ReconstructionNorm8Mode::kTailWeightedLloyd
        ? TailWeight(value, tail_p99, tail_p999)
        : 1U;
    counts_[label] += weight;
    sums_[label] += static_cast<double>(weight) * value;
  }
  return Status::Ok();
}

void Norm8LloydAccumulator::Merge(
    const Norm8LloydAccumulator& other) noexcept {
  for (std::uint32_t center = 0U; center < 256U; ++center) {
    counts_[center] += other.counts_[center];
    sums_[center] += other.sums_[center];
  }
}

void Norm8LloydAccumulator::MergeBins(
    const std::array<std::uint64_t, 256>& counts,
    const std::array<double, 256>& sums) noexcept {
  for (std::uint32_t center = 0U; center < 256U; ++center) {
    counts_[center] += counts[center];
    sums_[center] += sums[center];
  }
}

StatusOr<ReconstructionNorm8Codebook>
ReconstructionNorm8Codebook::InitializeFromValues(
    std::vector<float> values,
    ReconstructionNorm8Mode mode) {
  if (values.empty()) return Status::InvalidArgument("reconstruction norm8 initializer requires a nonempty sample");
  if (const auto status = ValidateNorms(values); !status.ok()) return status;
  std::sort(values.begin(), values.end());
  return InitializeSorted(values, mode);
}

StatusOr<ReconstructionNorm8Codebook> ReconstructionNorm8Codebook::FitFromValues(
    std::vector<float> values, std::uint32_t iterations,
    const ParallelForExecutor& parallel, ReconstructionNorm8Mode mode) {
  if (const auto status = ValidateNorms(values); !status.ok()) return status;
  return FitFromValidatedValues(std::move(values), iterations, parallel, mode);
}

StatusOr<ReconstructionNorm8Codebook>
ReconstructionNorm8Codebook::FitFromValidatedValues(
    std::vector<float> values, std::uint32_t iterations,
    const ParallelForExecutor& parallel, ReconstructionNorm8Mode mode) {
  if (values.empty() || iterations == 0U)
    return Status::InvalidArgument("norm8 fit requires values and passes");
  std::sort(values.begin(), values.end());
  auto initialized = InitializeSorted(values, mode);
  if (!initialized.ok()) return initialized.status();
  auto result = std::move(initialized).value();
  const auto tail_begin = mode == ReconstructionNorm8Mode::kTailWeightedLloyd
      ? static_cast<std::size_t>(std::upper_bound(values.begin(), values.end(), result.tail_p99_) - values.begin())
      : values.size();
  const auto far_tail_begin = mode == ReconstructionNorm8Mode::kTailWeightedLloyd
      ? static_cast<std::size_t>(std::upper_bound(values.begin(), values.end(), result.tail_p999_) - values.begin())
      : values.size();
  const auto chunk_count = (values.size() + kSortedChunkRows - 1U) / kSortedChunkRows;
  for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
    std::array<std::size_t, 257> bounds{};
    bounds[256] = values.size();
    for (std::uint32_t bin = 1U; bin < 256U; ++bin) {
      auto low = bounds[bin - 1U], high = values.size();
      while (low < high) {
        const auto middle = low + (high - low) / 2U;
        if (NearestCenter(result.centers_.data(), values[middle]) < bin) low = middle + 1U;
        else high = middle;
      }
      bounds[bin] = low;
    }
    Norm8LloydAccumulator total;
    for (std::size_t first = 0U; first < chunk_count; first += 64U) {
      const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(64U, chunk_count - first));
      std::vector<Norm8LloydAccumulator> chunks(count);
      SortedPass pass{values, &bounds, tail_begin, far_tail_begin,
                      static_cast<std::uint32_t>(first), &chunks};
      parallel.Run(count, &pass, &SumSortedChunk);
      for (const auto& chunk : chunks) total.Merge(chunk);
    }
    if (result.FinishPass(total)) break;
  }
  return result;
}

StatusOr<ReconstructionNorm8Codebook>
ReconstructionNorm8Codebook::FromIdentityCenters(
    const std::array<float, 256>& centers) {
  if (const Status status = ValidateNorms(centers); !status.ok()) {
    return status;
  }
  if (!std::is_sorted(centers.begin(), centers.end())) {
    return Status::InvalidArgument(
        "norm8 identity centers must be sorted");
  }
  ReconstructionNorm8Codebook output;
  output.mode_ = ReconstructionNorm8Mode::kIdentityLloyd;
  output.centers_ = centers;
  return output;
}

StatusOr<ReconstructionNorm8Codebook>
ReconstructionNorm8Codebook::InitializeSorted(
    ArrayView<const float> values, ReconstructionNorm8Mode mode) {
  ReconstructionNorm8Codebook output;
  output.mode_ = mode;
  if (mode == ReconstructionNorm8Mode::kIdentityLloyd) {
    for (std::uint32_t center = 0U; center < 256U; ++center) {
      const std::size_t position =
          static_cast<std::size_t>(center) * values.size() / 256U;
      output.centers_[center] = values[position];
    }
    return output;
  }
  if (mode != ReconstructionNorm8Mode::kTailWeightedLloyd) {
    return Status::InvalidArgument("unsupported reconstruction norm8 mode");
  }
  output.tail_p99_ = SortedQuantileValue(values, 0.99);
  output.tail_p999_ = SortedQuantileValue(values, 0.999);
  double total_weight = 0.0;
  for (const float value : values) {
    total_weight += static_cast<double>(
        TailWeight(value, output.tail_p99_, output.tail_p999_));
  }
  double prefix_weight = 0.0;
  std::size_t sorted_index = 0U;
  for (std::uint32_t center = 0U; center < 256U; ++center) {
    const double target =
        static_cast<double>(center) * total_weight / 256.0;
    while (sorted_index + 1U < values.size() && prefix_weight < target) {
      prefix_weight += static_cast<double>(TailWeight(
          values[sorted_index], output.tail_p99_, output.tail_p999_));
      ++sorted_index;
    }
    output.centers_[center] = values[sorted_index];
  }
  return output;
}

bool ReconstructionNorm8Codebook::FinishPass(
    const Norm8LloydAccumulator& pass) noexcept {
  bool converged = true;
  for (std::uint32_t center = 0U; center < 256U; ++center) {
    if (pass.counts_[center] == 0U) {
      continue;
    }
    const float updated = static_cast<float>(
        pass.sums_[center] / static_cast<double>(pass.counts_[center]));
    if (std::abs(updated - centers_[center]) > 1.0e-6F) {
      converged = false;
    }
    centers_[center] = updated;
  }
  return converged;
}

Status ReconstructionNorm8Codebook::Encode(
    ArrayView<const float> norm2,
    ArrayView<std::uint8_t> labels) const {
  return EncodeImpl(norm2, labels, false);
}

Status ReconstructionNorm8Codebook::EncodeValidated(
    ArrayView<const float> norm2,
    ArrayView<std::uint8_t> labels) const {
  return EncodeImpl(norm2, labels, true);
}

Status ReconstructionNorm8Codebook::EncodeImpl(
    ArrayView<const float> norm2,
    ArrayView<std::uint8_t> labels,
    bool validated_norms) const {
  if (norm2.size() != labels.size()) {
    return Status::InvalidArgument(
        "reconstruction norm8 label output has the wrong row count");
  }
  if (!validated_norms) {
    if (const Status status = ValidateNorms(norm2); !status.ok()) {
      return status;
    }
  }
#ifndef NDEBUG
  if (validated_norms) assert(ValidateNorms(norm2).ok());
#endif
  for (std::size_t row = 0U; row < norm2.size(); ++row) {
    labels[row] = NearestCenter(centers_.data(), norm2[row]);
  }
  return Status::Ok();
}

}  // namespace stlqpp
