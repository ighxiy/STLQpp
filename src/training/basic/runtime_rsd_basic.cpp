#include "stlqpp/training/basic/runtime_rsd_basic.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

#include "stlqpp/training/basic/detail/basic_rsd_support.h"
#include "stlqpp/training/linkage/rsd/detail/rsd_candidate_selector.h"

namespace stlqpp {
namespace {
struct LargeContext {
  const RuntimeBasicCodebookAlgebra* algebra;
  const RuntimeBasicProjection* projection;
  ArrayView<const std::uint32_t> fixed_roots;
  ArrayView<const std::uint32_t> selected_roots;
  ArrayView<const float> selected_root_dots;
  std::uint32_t maximum_rank;
  LeastSquaresCostMode cost_mode;
  RuntimeBasicEncoding* output;
  std::atomic<std::uint32_t>* failures;
};
template <std::uint32_t Capacity>
void FitLargeSample(void* opaque, std::uint32_t,
                    std::uint32_t sample) noexcept {
  auto& context = *static_cast<LargeContext*>(opaque);
  const auto& algebra = *context.algebra;
  const auto M = algebra.shape().m;
  const float* vector = context.projection->samples().column(sample);
  const float* small_dots = context.projection->small_dots(sample);
  const float sample_norm2 = context.projection->squared_norm(sample);
  const auto root_dot_for = [&](std::uint32_t code) noexcept {
    const float* atom = algebra.root0().column(code);
    float dot = 0.0F;
    for (std::uint32_t axis = 0U; axis < algebra.shape().dimension; ++axis) {
      dot += vector[axis] * atom[axis];
    }
    return dot;
  };
  std::uint32_t root = 0U;
  float root_dot = 0.0F;
  if (!context.fixed_roots.empty()) {
    root = context.fixed_roots[sample];
    root_dot = root_dot_for(root);
  } else {
    root = context.selected_roots[sample];
    root_dot = context.selected_root_dots[sample];
  }
  auto& out = *context.output;
  float error;
  if (!detail::FitBasicRsdSupport<0, Capacity>(
          algebra, root, root_dot, small_dots, sample_norm2,
          context.maximum_rank, context.cost_mode,
          out.codes.data() + static_cast<std::size_t>(sample) * (M - 1U),
          out.coefficients.data() + static_cast<std::size_t>(sample) * M,
          &error)) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  out.roots[sample] = root;
  out.raw_errors[sample] = out.errors[sample] = error;
}

struct SmallContext {
  const LinkageCodebookAlgebra<0>* algebra;
  const float* dots;
  const float* norms;
  ArrayView<const std::uint32_t> fixed_roots;
  std::uint32_t rank;
  LeastSquaresCostMode mode;
  RuntimeBasicEncoding* output;
  std::atomic<std::uint32_t>* failures;
};

template <std::uint32_t Capacity>
void FitSmallSample(void* opaque, std::uint32_t,
                    std::uint32_t sample) noexcept {
  auto& context = *static_cast<SmallContext*>(opaque);
  const auto& algebra = *context.algebra;
  const auto m = algebra.shape().m;
  const auto* dots =
      context.dots + static_cast<std::size_t>(sample) * algebra.atom_count();
  std::uint32_t root = 0U;
  if (context.fixed_roots.empty()) {
    float best = std::numeric_limits<float>::infinity();
    for (std::uint32_t code = 0; code < 256U; ++code) {
      const float norm = algebra.squared_norm(code);
      if (!(norm > 0.0F)) continue;
      const float error = std::fmax(
          0.0F, context.norms[sample] - dots[code] * (dots[code] / norm));
      if (error < best) {
        best = error;
        root = code;
      }
    }
  } else {
    root = context.fixed_roots[sample];
  }
  LinkageSolution<0, Capacity> solution;
  if (!detail::FitRsdOneFixedLayerZero<0, Capacity>(
          algebra, dots, context.norms[sample], static_cast<std::uint8_t>(root),
          &solution, context.rank, context.mode)) {
    context.failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  auto& out = *context.output;
  out.roots[sample] = root;
  std::copy_n(solution.support.codes.data() + 1U, m - 1U,
              out.codes.data() + static_cast<std::size_t>(sample) * (m - 1U));
  std::copy_n(solution.coefficients.data(), m,
              out.coefficients.data() + static_cast<std::size_t>(sample) * m);
  out.raw_errors[sample] = out.errors[sample] = solution.squared_error;
}
}  // namespace

StatusOr<RuntimeCpuRsdBasicEncoder> RuntimeCpuRsdBasicEncoder::Create(
    RuntimeRootCodebookView codebooks, const F32GemmBackend& gemm,
    const ParallelForExecutor& parallel) {
  auto algebra = RuntimeBasicCodebookAlgebra::Create(codebooks, gemm, &parallel);
  if (!algebra.ok()) return algebra.status();
  std::optional<LinkageCodebookAlgebra<0>> rsd;
  if (codebooks.shape.nlist == 256U) {
    // Preserve the same full M*256 Gram and full projection numerical shape
    // used by the ordinary fixed-M basic RSD path.
    auto created = LinkageCodebookAlgebra<0>::Create(
        codebooks.shape, algebra.value().full_flat(), gemm);
    if (!created.ok()) return created.status();
    rsd.emplace(std::move(created).value());
  }
  return RuntimeCpuRsdBasicEncoder(std::move(algebra).value(), std::move(rsd),
                                   gemm, parallel);
}

Status RuntimeCpuRsdBasicEncoder::Encode(
    ColumnMajorMatrixF32View samples, std::uint32_t rank,
    BasicSchedule schedule, std::uint32_t seed,
    std::uint64_t global_sample_offset,
    ArrayView<const std::uint64_t> sample_ids,
    ArrayView<const std::uint32_t> fixed_roots, RuntimeBasicEncoding* output) {
  const auto shape = algebra_.shape();
  if (output == nullptr || rank < 1U || rank > 3U ||
      samples.values == nullptr || samples.rows != shape.dimension ||
      samples.columns == 0U ||
      (!sample_ids.empty() && sample_ids.size() != samples.columns) ||
      (!fixed_roots.empty() && fixed_roots.size() != samples.columns))
    return Status::InvalidArgument(
        "runtime RSD basic input shape or rank is invalid");
  for (const auto root : fixed_roots)
    if (root >= shape.nlist)
      return Status::InvalidArgument("runtime RSD fixed root exceeds codebook");
  if (const auto s = output->Resize(shape.m, samples.columns); !s.ok())
    return s;
  std::atomic<std::uint32_t> failures{0U};
  if (rsd_) {
    norms_.resize(samples.columns);
    for (std::uint32_t sample = 0; sample < samples.columns; ++sample) {
      float norm = 0.0F;
      const float* row = samples.column(sample);
      for (std::uint32_t axis = 0; axis < samples.rows; ++axis)
        norm += row[axis] * row[axis];
      norms_[sample] = norm;
    }
    dots_.resize(static_cast<std::size_t>(samples.columns) *
                 rsd_->atom_count());
    if (const auto s = gemm_->MultiplyTransposedLeft(
            {.left = rsd_->atoms(),
             .right = samples.values,
             .query_count = samples.columns,
             .right_leading_dimension = samples.rows,
             .output = dots_.data(),
             .output_leading_dimension = rsd_->atom_count()});
        !s.ok())
      return s;
    if (const auto s = projection_.BindFullAtomProjection(
            samples, norms_, dots_, rsd_->atom_count(), 256U);
        !s.ok())
      return s;
    SmallContext context{&*rsd_,        dots_.data(),
                         norms_.data(), fixed_roots,
                         rank,          schedule.least_squares_cost_mode(),
                         output,        &failures};
    parallel_->Run(samples.columns, &context, shape.m > kLegacyRuntimeScratchLayers
        ? FitSmallSample<kWideRuntimeScratchLayers> : FitSmallSample<kLegacyRuntimeScratchLayers>);
  } else {
    if (const auto s =
            projection_.Recompute(samples, algebra_, *gemm_, *parallel_);
        !s.ok())
      return s;
    if (fixed_roots.empty()) {
      constexpr std::uint32_t kRootBlock = 1024U;
      roots_.assign(samples.columns, 0U);
      root_dots_.resize(samples.columns);
      root_errors_.assign(samples.columns,
                          std::numeric_limits<float>::infinity());
      root_scores_.resize(static_cast<std::size_t>(samples.columns) *
                          kRootBlock);
      for (std::uint32_t begin = 0; begin < shape.nlist; begin += kRootBlock) {
        const auto count = std::min(kRootBlock, shape.nlist - begin);
        if (const auto s = gemm_->MultiplyTransposedLeft(
                {.left = {algebra_.root0().column(begin), samples.rows, count},
                 .right = samples.values,
                 .query_count = samples.columns,
                 .right_leading_dimension = samples.rows,
                 .output = root_scores_.data(),
                 .output_leading_dimension = count});
            !s.ok())
          return s;
        for (std::uint32_t sample = 0; sample < samples.columns; ++sample) {
          const float* scores =
              root_scores_.data() + static_cast<std::size_t>(sample) * count;
          for (std::uint32_t local = 0; local < count; ++local) {
            const auto root = begin + local;
            const float norm = algebra_.root_norm2(root);
            if (!(norm > 0.0F)) continue;
            const float dot = scores[local];
            const float error = std::fmax(
                0.0F, projection_.squared_norm(sample) - dot * (dot / norm));
            if (error < root_errors_[sample]) {
              root_errors_[sample] = error;
              roots_[sample] = root;
              root_dots_[sample] = dot;
            }
          }
        }
      }
    }
    LargeContext context{&algebra_,
                         &projection_,
                         fixed_roots,
                         roots_,
                         root_dots_,
                         rank,
                         schedule.least_squares_cost_mode(),
                         output,
                         &failures};
    parallel_->Run(samples.columns, &context, shape.m > kLegacyRuntimeScratchLayers
        ? FitLargeSample<kWideRuntimeScratchLayers> : FitLargeSample<kLegacyRuntimeScratchLayers>);
  }
  if (failures.load(std::memory_order_relaxed) != 0U)
    return Status::FailedPrecondition(
        "runtime basic RSD contains a singular candidate set");
  return detail::OptimizeRuntimeBasicTrusted(
      algebra_, projection_, *parallel_, schedule, seed, global_sample_offset,
      sample_ids, {}, output);
}
}  // namespace stlqpp
