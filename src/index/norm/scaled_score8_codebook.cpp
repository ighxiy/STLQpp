#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace stlqpp {
namespace {

StatusOr<float> ComputeRadialScoreScaleImpl(
    float source_norm2, float reconstruction_norm2,
    bool validated_norms) {
  if (!validated_norms &&
      (!std::isfinite(source_norm2) || source_norm2 < 0.0F ||
       !std::isfinite(reconstruction_norm2) ||
       reconstruction_norm2 < 0.0F)) {
    return Status::InvalidArgument(
        "radial scoring requires finite nonnegative norms");
  }
#ifndef NDEBUG
  if (validated_norms) {
    assert(std::isfinite(source_norm2) && source_norm2 >= 0.0F);
    assert(std::isfinite(reconstruction_norm2) &&
           reconstruction_norm2 >= 0.0F);
  }
#endif
  float scale = 1.0F;
  if (source_norm2 == 0.0F && reconstruction_norm2 != 0.0F) {
    scale = 0.0F;
  } else if (source_norm2 != 0.0F && reconstruction_norm2 != 0.0F) {
    const float ratio = source_norm2 / reconstruction_norm2;
    if (std::isfinite(ratio) && ratio != 0.0F) {
      // Keep the accepted ordinary-data FP32 operation order unchanged.
      scale = std::sqrt(ratio);
    } else {
      const double stable_ratio =
          static_cast<double>(source_norm2) /
          static_cast<double>(reconstruction_norm2);
      scale = static_cast<float>(std::sqrt(stable_ratio));
    }
  }
  const float dot_multiplier = -2.0F * scale;
  if (!std::isfinite(scale) || scale < 0.0F ||
      !std::isfinite(dot_multiplier)) {
    return Status::DataLoss(
        "radial scoring scale is outside the finite FP32 score range");
  }
  return scale;
}

}  // namespace

StatusOr<float> ComputeRadialScoreScale(
    float source_norm2, float reconstruction_norm2) {
  return ComputeRadialScoreScaleImpl(
      source_norm2, reconstruction_norm2, false);
}

StatusOr<ScaledScore8Codebook> ScaledScore8Codebook::Create(
    ArrayView<const float> scales, ArrayView<const float> biases) {
  if (scales.size() != 256U || biases.size() != 256U) {
    return Status::InvalidArgument(
        "scaled score8 requires exactly 256 scale/bias pairs");
  }
  ScaledScore8Codebook output;
  for (std::size_t label = 0U; label < 256U; ++label) {
    if (!std::isfinite(scales[label]) || scales[label] < 0.0F ||
        !std::isfinite(-2.0F * scales[label]) ||
        !std::isfinite(biases[label]) || biases[label] < 0.0F) {
      return Status::InvalidArgument(
          "scaled score8 contains a non-finite or negative value");
    }
    output.scales_[label] = scales[label];
    output.biases_[label] = biases[label];
  }
  output.enabled_ = true;
  return output;
}

Status ScaledScore8Accumulator::Accumulate(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2) {
  return AccumulateImpl(labels, reconstruction_norm2, source_norm2, false);
}

Status ScaledScore8Accumulator::AccumulateValidated(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2) {
  return AccumulateImpl(labels, reconstruction_norm2, source_norm2, true);
}

Status ScaledScore8Accumulator::Accumulate(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2,
    const ParallelForExecutor& parallel) {
  return AccumulateParallel(labels, reconstruction_norm2, source_norm2, false,
                            parallel);
}

Status ScaledScore8Accumulator::AccumulateValidated(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2,
    const ParallelForExecutor& parallel) {
  return AccumulateParallel(labels, reconstruction_norm2, source_norm2, true,
                            parallel);
}

Status ScaledScore8Accumulator::AccumulateParallel(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2,
    bool validated_norms, const ParallelForExecutor& parallel) {
  if (labels.size() != reconstruction_norm2.size() || labels.size() != source_norm2.size())
    return Status::InvalidArgument("scaled score8 rows and fixed labels disagree");
  constexpr std::size_t kMinimumRows = 262144U;
  constexpr std::uint32_t kMaximumTasks = 32U;
  if (labels.size() < kMinimumRows)
    return AccumulateImpl(labels, reconstruction_norm2, source_norm2, validated_norms);
  std::array<ScaledScore8Accumulator, kMaximumTasks> partials;
  std::array<Status, kMaximumTasks> statuses;
  // Fixed numerical shards keep the reduction independent of thread count.
  const auto tasks = kMaximumTasks;
  struct Work {
    ArrayView<const std::uint8_t> labels;
    ArrayView<const float> reconstruction, source;
    bool validated;
    std::uint32_t tasks;
    ScaledScore8Accumulator* partials;
    Status* statuses;
  } work{labels, reconstruction_norm2, source_norm2, validated_norms, tasks,
         partials.data(), statuses.data()};
  parallel.Run(tasks, &work, [](void* opaque, std::uint32_t, std::uint32_t item) noexcept {
    auto& w = *static_cast<Work*>(opaque);
    const auto begin = w.labels.size() * item / w.tasks;
    const auto end = w.labels.size() * (item + 1U) / w.tasks;
    w.statuses[item] = w.partials[item].AccumulateImpl(
        {w.labels.data() + begin, end - begin},
        {w.reconstruction.data() + begin, end - begin},
        {w.source.data() + begin, end - begin}, w.validated);
  });
  // Check every task before merging so a failed call leaves this accumulator
  // unchanged. The executor joins all tasks before their stack buffers die.
  for (std::uint32_t item = 0U; item < tasks; ++item)
    if (!statuses[item].ok()) return statuses[item];
  for (std::uint32_t item = 0U; item < tasks; ++item) Merge(partials[item]);
  return Status::Ok();
}

Status ScaledScore8Accumulator::AccumulateImpl(
    ArrayView<const std::uint8_t> labels,
    ArrayView<const float> reconstruction_norm2,
    ArrayView<const float> source_norm2,
    bool validated_norms) {
  if (labels.size() != reconstruction_norm2.size() ||
      labels.size() != source_norm2.size()) {
    return Status::InvalidArgument(
        "scaled score8 rows and fixed labels disagree");
  }
  for (std::size_t row = 0U; row < labels.size(); ++row) {
    const float rho = reconstruction_norm2[row];
    const float source = source_norm2[row];
    if (!validated_norms &&
        (!std::isfinite(rho) || rho < 0.0F ||
         !std::isfinite(source) || source < 0.0F)) {
      return Status::InvalidArgument(
          "scaled score8 requires finite nonnegative norms");
    }
    auto scale_result = ComputeRadialScoreScaleImpl(source, rho, true);
    if (!scale_result.ok()) return scale_result.status();
    const float scale = scale_result.value();
    const std::uint8_t label = labels[row];
    scale_numerators_[label] +=
        static_cast<double>(rho) * static_cast<double>(scale);
    scale_denominators_[label] += static_cast<double>(rho);
    bias_sums_[label] += static_cast<double>(source);
    ++counts_[label];
  }
  return Status::Ok();
}

void ScaledScore8Accumulator::Merge(
    const ScaledScore8Accumulator& other) noexcept {
  for (std::size_t label = 0U; label < 256U; ++label) {
    scale_numerators_[label] += other.scale_numerators_[label];
    scale_denominators_[label] += other.scale_denominators_[label];
    bias_sums_[label] += other.bias_sums_[label];
    counts_[label] += other.counts_[label];
  }
}

StatusOr<ScaledScore8Codebook> ScaledScore8Accumulator::Finish(
    Norm8CodebookView classic) const {
  if (classic.centers == nullptr || classic.center_count != 256U) {
    return Status::InvalidArgument(
        "scaled score8 requires the complete classic norm8 codebook");
  }
  ScaledScore8Codebook output;
  for (std::uint32_t label = 0U; label < 256U; ++label) {
    float scale = 1.0F;
    float bias = classic.centers[label];
    if (scale_denominators_[label] > 0.0) {
      scale = static_cast<float>(
          scale_numerators_[label] / scale_denominators_[label]);
    }
    if (counts_[label] != 0U) {
      bias = static_cast<float>(
          bias_sums_[label] / static_cast<double>(counts_[label]));
    }
    if (!std::isfinite(scale) || scale < 0.0F ||
        !std::isfinite(-2.0F * scale) ||
        !std::isfinite(bias) || bias < 0.0F) {
      return Status::DataLoss(
          "scaled score8 fit produced an invalid center");
    }
    output.scales_[label] = scale;
    output.biases_[label] = bias;
  }
  output.enabled_ = true;
  return output;
}

}  // namespace stlqpp
