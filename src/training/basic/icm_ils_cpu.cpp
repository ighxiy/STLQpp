#include "stlqpp/training/basic/icm_ils_cpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "stlqpp/training/basic/detail/basic_encoding_access.h"
#include "stlqpp/training/basic/detail/icm_candidate_selector.h"
#include "stlqpp/training/basic/support_least_squares.h"

namespace stlqpp {
namespace {

constexpr float kAcceptanceEpsilon = 1.0e-6F;

class SplitMix64 final {
 public:
  explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  [[nodiscard]] std::uint64_t Next() noexcept {
    std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

 private:
  std::uint64_t state_ = 0U;
};

[[nodiscard]] std::uint32_t UniformBelow(SplitMix64* random,
                                         std::uint32_t bound) noexcept {
  const std::uint64_t value = random->Next();
  const std::uint64_t high = value >> 32U;
  const std::uint64_t low = value & 0xffffffffULL;
  const std::uint64_t middle = high * bound + ((low * bound) >> 32U);
  return static_cast<std::uint32_t>(middle >> 32U);
}

template <int M>
struct MutableState final {
  std::uint32_t count = 0U;
  const std::uint32_t* roots = nullptr;
  std::uint8_t* codes = nullptr;
  float* coefficients = nullptr;
  float* errors = nullptr;
};

template <int FixedM>
using KernelAlgebra =
    std::conditional_t<FixedM == 0, RuntimeBasicCodebookAlgebra,
                       BasicCodebookAlgebra<FixedM>>;
template <int FixedM>
using KernelProjection = std::conditional_t<FixedM == 0, RuntimeBasicProjection,
                                            BasicProjection<FixedM>>;
template <int FixedM>
using KernelLeastSquares =
    std::conditional_t<FixedM == 0, RuntimeBasicSupportLeastSquares,
                       BasicSupportLeastSquares<FixedM>>;
std::uint32_t RuntimeIlsRounds(BasicSchedule schedule) noexcept {
  return schedule.perturbations() == 0U ? 0U : schedule.ils_rounds();
}

template <int FixedM, LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
[[nodiscard]] bool RefitLocal(const KernelAlgebra<FixedM>& algebra,
                              const KernelProjection<FixedM>& projection,
                              std::uint32_t sample, std::uint32_t root,
                              float root_dot, std::uint8_t* codes,
                              float* coefficients, float* error) noexcept {
  const float* small_dots = projection.small_dots(sample);
  float applied_diagonal_bump = 0.0F;
  if (!detail::SolveBasicSupport<FixedM, Capacity>(algebra, root, codes, root_dot,
                                         small_dots, coefficients,
                                         &applied_diagonal_bump)) {
    *error = std::numeric_limits<float>::infinity();
    return false;
  }
  *error = KernelLeastSquares<FixedM>::template Cost<CostMode>(
      algebra, root, codes, root_dot, small_dots, coefficients,
      projection.squared_norm(sample), applied_diagonal_bump);
  return true;
}

template <int FixedM, bool UseAbsoluteScore, LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
[[nodiscard]] bool IcmLocal(
    const KernelAlgebra<FixedM>& algebra,
    const KernelProjection<FixedM>& projection, std::uint32_t sample,
    std::uint32_t root, float root_dot,
    const std::array<std::uint8_t, Capacity>& layer_order,
    std::uint8_t* codes, float* coefficients, float* error) noexcept {
  const float* small_dots = projection.small_dots(sample);
  const std::uint32_t M = FixedM == 0 ? algebra.shape().m : FixedM;
  bool changed = false;
  for (std::uint32_t order_index = 0U;
       order_index < static_cast<std::uint32_t>(M); ++order_index) {
    const std::uint32_t layer = layer_order[order_index];
    if (layer == 0U) continue;
    const std::uint32_t offset = algebra.small_offset(layer);
    const std::uint8_t old_code = codes[layer - 1U];
    std::uint32_t best_code = old_code;
    std::array<float, 256> residual_dots{};
    for (std::uint32_t code = 0U; code < 256U; ++code) {
      const std::uint32_t flat = offset + code;
      residual_dots[code] =
          small_dots[flat] -
          coefficients[0] * algebra.root_small_dot(root, flat);
    }
    const float* small_gram = algebra.small_gram().data();
    const std::uint32_t small_count = algebra.small_count();
    for (std::uint32_t other_layer = 1U;
         other_layer < static_cast<std::uint32_t>(M); ++other_layer) {
      if (other_layer == layer) continue;
      const std::uint32_t other_flat =
          algebra.small_offset(other_layer) + codes[other_layer - 1U];
      const float* gram_slice =
          small_gram + static_cast<std::size_t>(other_flat) * small_count +
          offset;
      const float coefficient = coefficients[other_layer];
      for (std::uint32_t code = 0U; code < 256U; ++code) {
        residual_dots[code] -= coefficient * gram_slice[code];
      }
    }
    const float* inverse_norm = algebra.small_inverse_norms().data() + offset;
    const detail::IcmCandidateSelection selection = detail::SelectIcmCandidate(
        residual_dots.data(), inverse_norm, old_code, UseAbsoluteScore);
    best_code = selection.code;
    if (best_code == old_code) continue;

    std::array<float, Capacity> backup{};
    std::copy(coefficients, coefficients + M, backup.begin());
    codes[layer - 1U] = static_cast<std::uint8_t>(best_code);
    float applied_diagonal_bump = 0.0F;
    if (!detail::SolveBasicSupport<FixedM, Capacity>(algebra, root, codes, root_dot,
                                           small_dots, coefficients,
                                           &applied_diagonal_bump)) {
      codes[layer - 1U] = old_code;
      std::copy_n(backup.data(), M, coefficients);
      continue;
    }
    const float candidate_error =
        KernelLeastSquares<FixedM>::template Cost<CostMode>(
            algebra, root, codes, root_dot, small_dots, coefficients,
            projection.squared_norm(sample), applied_diagonal_bump);
    if (candidate_error + kAcceptanceEpsilon < *error) {
      *error = candidate_error;
      changed = true;
    } else {
      codes[layer - 1U] = old_code;
      std::copy_n(backup.data(), M, coefficients);
    }
  }
  return changed;
}

template <int FixedM>
void PerturbLocal(std::uint32_t active_m, std::uint8_t* codes,
                  std::uint32_t round, std::uint32_t seed,
                  std::uint32_t perturbations,
                  std::uint64_t global_sample) noexcept {
  const std::uint32_t M = FixedM == 0 ? active_m : FixedM;
  SplitMix64 random(
      static_cast<std::uint64_t>(seed) ^
      (static_cast<std::uint64_t>(round + 1U) * 0x9e3779b97f4a7c15ULL) ^
      ((global_sample + 1U) * 0xbf58476d1ce4e5b9ULL));
  for (std::uint32_t index = 0U; index < perturbations; ++index) {
    const std::uint32_t layer =
        1U + UniformBelow(&random, static_cast<std::uint32_t>(M - 1));
    const std::uint8_t old_code = codes[layer - 1U];
    const std::uint32_t raw = UniformBelow(&random, 255U);
    const std::uint32_t replacement = raw >= old_code ? raw + 1U : raw;
    codes[layer - 1U] = static_cast<std::uint8_t>(replacement);
  }
}

template <int FixedM, bool UseAbsoluteScore, bool UseExplicitIds,
          LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
struct SampleLocalContext final {
  const KernelAlgebra<FixedM>* algebra = nullptr;
  const KernelProjection<FixedM>* projection = nullptr;
  BasicSchedule schedule = BasicSchedule::Training();
  std::uint32_t seed = 0U;
  std::uint64_t global_sample_offset = 0U;
  const std::uint64_t* global_sample_ids = nullptr;
  const float* projected_root_dots = nullptr;
  const std::array<std::uint8_t, Capacity>* layer_orders = nullptr;
  MutableState<FixedM> current;
  std::atomic<std::uint32_t>* initial_failures = nullptr;
};

template <int FixedM, bool UseAbsoluteScore, bool UseExplicitIds,
          LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
void RefineSampleLocal(void* opaque, std::uint32_t,
                       std::uint32_t sample) noexcept {
  auto& context = *static_cast<
      SampleLocalContext<FixedM, UseAbsoluteScore, UseExplicitIds, CostMode, Capacity>*>(
      opaque);
  const std::uint32_t M = FixedM == 0 ? context.algebra->shape().m : FixedM;
  const std::uint32_t root = context.current.roots[sample];
  std::uint8_t* current_codes =
      context.current.codes + static_cast<std::size_t>(sample) * (M - 1U);
  float* current_coefficients =
      context.current.coefficients + static_cast<std::size_t>(sample) * M;
  float* current_error = context.current.errors + sample;
  const float root_dot = [&]() noexcept {
    if (context.projected_root_dots != nullptr) {
      return context.projected_root_dots[sample];
    }
    const float* root_vector = context.algebra->root0().column(root);
    const float* input = context.projection->samples().column(sample);
    float dot = 0.0F;
    for (std::uint32_t row = 0U; row < context.projection->samples().rows;
         ++row) {
      dot += root_vector[row] * input[row];
    }
    return dot;
  }();

  if (!RefitLocal<FixedM, CostMode, Capacity>(*context.algebra, *context.projection,
                                    sample, root, root_dot, current_codes,
                                    current_coefficients, current_error)) {
    context.initial_failures->fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  std::size_t order_index = 0U;
  bool current_active = true;
  for (std::uint32_t sweep = 0U; sweep < context.schedule.icm_sweeps();
       ++sweep) {
    if (current_active) {
      current_active = IcmLocal<FixedM, UseAbsoluteScore, CostMode, Capacity>(
          *context.algebra, *context.projection, sample, root, root_dot,
          context.layer_orders[order_index], current_codes,
          current_coefficients, current_error);
    }
    ++order_index;
  }

  std::array<std::uint8_t, Capacity - 1> candidate_codes{};
  std::array<float, Capacity> candidate_coefficients{};
  for (std::uint32_t round = 0U; round < RuntimeIlsRounds(context.schedule);
       ++round) {
    std::copy(current_codes, current_codes + (M - 1U), candidate_codes.begin());
    const std::uint64_t global_sample = [&]() noexcept {
      if constexpr (UseExplicitIds) {
        return context.global_sample_ids[sample];
      } else {
        return context.global_sample_offset + sample;
      }
    }();
    PerturbLocal<FixedM>(M, candidate_codes.data(), round, context.seed,
                         std::min(M, context.schedule.perturbations()),
                         global_sample);
    float candidate_error = std::numeric_limits<float>::infinity();
    bool candidate_active = RefitLocal<FixedM, CostMode, Capacity>(
        *context.algebra, *context.projection, sample, root, root_dot,
        candidate_codes.data(), candidate_coefficients.data(),
        &candidate_error);
    for (std::uint32_t sweep = 0U; sweep < context.schedule.icm_sweeps();
         ++sweep) {
      if (candidate_active) {
        candidate_active = IcmLocal<FixedM, UseAbsoluteScore, CostMode, Capacity>(
            *context.algebra, *context.projection, sample, root, root_dot,
            context.layer_orders[order_index], candidate_codes.data(),
            candidate_coefficients.data(), &candidate_error);
      }
      ++order_index;
    }
    if (candidate_error + kAcceptanceEpsilon < *current_error) {
      std::copy_n(candidate_codes.data(), M - 1U, current_codes);
      std::copy_n(candidate_coefficients.data(), M, current_coefficients);
      *current_error = candidate_error;
    }
  }
}

template <int FixedM, bool UseAbsoluteScore, bool UseExplicitIds,
          LeastSquaresCostMode CostMode, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
[[nodiscard]] Status OptimizeFixedRule(
    const KernelAlgebra<FixedM>& algebra,
    const KernelProjection<FixedM>& projection,
    const ParallelForExecutor& parallel, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset,
    const std::uint64_t* global_sample_ids,
    ArrayView<const float> projected_root_dots, MutableState<FixedM> current) {
  const std::uint32_t M = FixedM == 0 ? algebra.shape().m : FixedM;
  std::vector<std::array<std::uint8_t, Capacity>> layer_orders;
  const std::uint32_t ils_rounds = RuntimeIlsRounds(schedule);
  layer_orders.reserve(static_cast<std::size_t>(ils_rounds + 1U) *
                       schedule.icm_sweeps());
  std::mt19937 layer_random(seed);
  for (std::uint32_t phase = 0U; phase <= ils_rounds; ++phase) {
    std::array<std::uint8_t, Capacity> order{};
    for (std::uint32_t layer = 0U; layer < static_cast<std::uint32_t>(M);
         ++layer) {
      order[layer] = static_cast<std::uint8_t>(layer);
    }
    for (std::uint32_t sweep = 0U; sweep < schedule.icm_sweeps(); ++sweep) {
      std::shuffle(order.begin(), order.begin() + M, layer_random);
      layer_orders.push_back(order);
    }
  }
  std::atomic<std::uint32_t> initial_failures{0U};
  SampleLocalContext<FixedM, UseAbsoluteScore, UseExplicitIds, CostMode, Capacity>
      context{
          .algebra = &algebra,
          .projection = &projection,
          .schedule = schedule,
          .seed = seed,
          .global_sample_offset = global_sample_offset,
          .global_sample_ids = global_sample_ids,
          .projected_root_dots = projected_root_dots.empty()
                                     ? nullptr
                                     : projected_root_dots.data(),
          .layer_orders = layer_orders.data(),
          .current = current,
          .initial_failures = &initial_failures,
      };
  parallel.Run(
      current.count, &context,
      RefineSampleLocal<FixedM, UseAbsoluteScore, UseExplicitIds, CostMode, Capacity>);
  if (initial_failures.load(std::memory_order_relaxed) != 0U) {
    return Status::FailedPrecondition(
        "initial basic support is singular after registered Cholesky retries");
  }
  return Status::Ok();
}

template <int FixedM, bool UseAbsoluteScore, bool UseExplicitIds, std::uint32_t Capacity = kModelScratchLayers<FixedM>>
[[nodiscard]] Status OptimizeSelectedCostMode(
    const KernelAlgebra<FixedM>& algebra,
    const KernelProjection<FixedM>& projection,
    const ParallelForExecutor& parallel, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset,
    const std::uint64_t* global_sample_ids,
    ArrayView<const float> projected_root_dots, MutableState<FixedM> current) {
  if constexpr (FixedM == 0 && Capacity == kLegacyRuntimeScratchLayers) {
    if (algebra.shape().m > kLegacyRuntimeScratchLayers)
      return OptimizeSelectedCostMode<0, UseAbsoluteScore, UseExplicitIds, kWideRuntimeScratchLayers>(
          algebra, projection, parallel, schedule, seed, global_sample_offset,
          global_sample_ids, projected_root_dots, current);
  }

  if (schedule.least_squares_cost_mode() ==
      LeastSquaresCostMode::kNormalEquation) {
    return OptimizeFixedRule<FixedM, UseAbsoluteScore, UseExplicitIds,
                             LeastSquaresCostMode::kNormalEquation, Capacity>(
        algebra, projection, parallel, schedule, seed, global_sample_offset,
        global_sample_ids, projected_root_dots, current);
  }
  return OptimizeFixedRule<FixedM, UseAbsoluteScore, UseExplicitIds,
                           LeastSquaresCostMode::kExplicitQuadratic, Capacity>(
      algebra, projection, parallel, schedule, seed, global_sample_offset,
      global_sample_ids, projected_root_dots, current);
}

}  // namespace

template <int M>
Status CpuIcmIlsOptimizer<M>::Optimize(const BasicProjection<M>& projection,
                                       BasicSchedule schedule,
                                       std::uint32_t seed,
                                       std::uint64_t global_sample_offset,
                                       BasicEncoding<M>* encoding) const {
  if (encoding == nullptr ||
      encoding->sample_count() != projection.sample_count()) {
    return Status::InvalidArgument(
        "basic optimization input and output counts must match");
  }
  MutableState<M> state{
      .count = encoding->sample_count(),
      .roots = detail::BasicEncodingAccess::Roots(encoding),
      .codes = detail::BasicEncodingAccess::Codes(encoding),
      .coefficients = detail::BasicEncodingAccess::Coefficients(encoding),
      .errors = detail::BasicEncodingAccess::Errors(encoding),
  };
  if (schedule.score_rule() == BasicScoreRule::kAbsolute) {
    return OptimizeSelectedCostMode<M, true, false>(
        algebra_, projection, parallel_, schedule, seed, global_sample_offset,
        nullptr, {}, state);
  }
  return OptimizeSelectedCostMode<M, false, false>(
      algebra_, projection, parallel_, schedule, seed, global_sample_offset,
      nullptr, {}, state);
}

template <int M>
Status CpuIcmIlsOptimizer<M>::OptimizeWithSampleIds(
    const BasicProjection<M>& projection, BasicSchedule schedule,
    std::uint32_t seed, ArrayView<const std::uint64_t> global_sample_ids,
    BasicEncoding<M>* encoding) const {
  if (encoding == nullptr ||
      encoding->sample_count() != projection.sample_count() ||
      global_sample_ids.size() != projection.sample_count()) {
    return Status::InvalidArgument(
        "basic optimization explicit IDs must match the sample count");
  }
  MutableState<M> state{
      .count = encoding->sample_count(),
      .roots = detail::BasicEncodingAccess::Roots(encoding),
      .codes = detail::BasicEncodingAccess::Codes(encoding),
      .coefficients = detail::BasicEncodingAccess::Coefficients(encoding),
      .errors = detail::BasicEncodingAccess::Errors(encoding),
  };
  if (schedule.score_rule() == BasicScoreRule::kAbsolute) {
    return OptimizeSelectedCostMode<M, true, true>(
        algebra_, projection, parallel_, schedule, seed, 0U,
        global_sample_ids.data(), {}, state);
  }
  return OptimizeSelectedCostMode<M, false, true>(
      algebra_, projection, parallel_, schedule, seed, 0U,
      global_sample_ids.data(), {}, state);
}

template <int M>
Status CpuIcmIlsOptimizer<M>::OptimizeWithRootDots(
    const BasicProjection<M>& projection, ArrayView<const float> root_dots,
    BasicSchedule schedule, std::uint32_t seed,
    BasicEncoding<M>* encoding) const {
  if (encoding == nullptr ||
      encoding->sample_count() != projection.sample_count() ||
      root_dots.size() != projection.sample_count()) {
    return Status::InvalidArgument(
        "projected root dots must match the basic sample count");
  }
  MutableState<M> state{
      .count = encoding->sample_count(),
      .roots = detail::BasicEncodingAccess::Roots(encoding),
      .codes = detail::BasicEncodingAccess::Codes(encoding),
      .coefficients = detail::BasicEncodingAccess::Coefficients(encoding),
      .errors = detail::BasicEncodingAccess::Errors(encoding),
  };
  if (schedule.score_rule() == BasicScoreRule::kAbsolute) {
    return OptimizeSelectedCostMode<M, true, false>(
        algebra_, projection, parallel_, schedule, seed, 0U, nullptr, root_dots,
        state);
  }
  return OptimizeSelectedCostMode<M, false, false>(
      algebra_, projection, parallel_, schedule, seed, 0U, nullptr, root_dots,
      state);
}

template <int M>
Status CpuIcmIlsOptimizer<M>::OptimizeWithRootDotsAndSampleIds(
    const BasicProjection<M>& projection, ArrayView<const float> root_dots,
    ArrayView<const std::uint64_t> global_sample_ids, BasicSchedule schedule,
    std::uint32_t seed, BasicEncoding<M>* encoding) const {
  if (encoding == nullptr ||
      encoding->sample_count() != projection.sample_count() ||
      root_dots.size() != projection.sample_count() ||
      global_sample_ids.size() != projection.sample_count()) {
    return Status::InvalidArgument(
        "projected root dots and explicit IDs must match the sample count");
  }
  MutableState<M> state{
      .count = encoding->sample_count(),
      .roots = detail::BasicEncodingAccess::Roots(encoding),
      .codes = detail::BasicEncodingAccess::Codes(encoding),
      .coefficients = detail::BasicEncodingAccess::Coefficients(encoding),
      .errors = detail::BasicEncodingAccess::Errors(encoding),
  };
  if (schedule.score_rule() == BasicScoreRule::kAbsolute) {
    return OptimizeSelectedCostMode<M, true, true>(
        algebra_, projection, parallel_, schedule, seed, 0U,
        global_sample_ids.data(), root_dots, state);
  }
  return OptimizeSelectedCostMode<M, false, true>(
      algebra_, projection, parallel_, schedule, seed, 0U,
      global_sample_ids.data(), root_dots, state);
}

Status OptimizeRuntimeBasic(const RuntimeBasicCodebookAlgebra& algebra,
                            const RuntimeBasicProjection& projection,
                            const ParallelForExecutor& parallel,
                            BasicSchedule schedule, std::uint32_t seed,
                            std::uint64_t global_sample_offset,
                            ArrayView<const std::uint64_t> sample_ids,
                            ArrayView<const float> root_dots,
                            RuntimeBasicEncoding* encoding) {
  if (encoding == nullptr ||
      projection.samples().rows != algebra.shape().dimension ||
      projection.small_count() != algebra.small_count() ||
      (!sample_ids.empty() && sample_ids.size() != projection.sample_count()) ||
      (!root_dots.empty() && root_dots.size() != projection.sample_count()))
    return Status::InvalidArgument("runtime refinement input shape mismatch");
  if (const auto status = encoding->Validate(
          algebra.shape().m, projection.sample_count(), algebra.shape().nlist);
      !status.ok())
    return status;
  return detail::OptimizeRuntimeBasicTrusted(
      algebra, projection, parallel, schedule, seed, global_sample_offset,
      sample_ids, root_dots, encoding);
}

Status detail::OptimizeRuntimeBasicTrusted(
    const RuntimeBasicCodebookAlgebra& algebra,
    const RuntimeBasicProjection& projection,
    const ParallelForExecutor& parallel, BasicSchedule schedule,
    std::uint32_t seed, std::uint64_t global_sample_offset,
    ArrayView<const std::uint64_t> sample_ids, ArrayView<const float> root_dots,
    RuntimeBasicEncoding* encoding) {
  assert(encoding != nullptr);
  assert(encoding
             ->Validate(algebra.shape().m, projection.sample_count(),
                        algebra.shape().nlist)
             .ok());
  MutableState<0> state{projection.sample_count(), encoding->roots.data(),
                        encoding->codes.data(), encoding->coefficients.data(),
                        encoding->errors.data()};
  const auto run = [&]<std::uint32_t Capacity>() -> Status {
  if (sample_ids.empty()) {
    if (schedule.score_rule() == BasicScoreRule::kAbsolute)
      return OptimizeSelectedCostMode<0, true, false, Capacity>(
          algebra, projection, parallel, schedule, seed, global_sample_offset,
          nullptr, root_dots, state);
    return OptimizeSelectedCostMode<0, false, false, Capacity>(
        algebra, projection, parallel, schedule, seed, global_sample_offset,
        nullptr, root_dots, state);
  }
  if (schedule.score_rule() == BasicScoreRule::kAbsolute)
    return OptimizeSelectedCostMode<0, true, true, Capacity>(
        algebra, projection, parallel, schedule, seed, 0U, sample_ids.data(),
        root_dots, state);
  return OptimizeSelectedCostMode<0, false, true, Capacity>(
      algebra, projection, parallel, schedule, seed, 0U, sample_ids.data(),
      root_dots, state);
  };
  return algebra.shape().m > kLegacyRuntimeScratchLayers
      ? run.template operator()<kWideRuntimeScratchLayers>()
      : run.template operator()<kLegacyRuntimeScratchLayers>();
}

template class CpuIcmIlsOptimizer<0>;
template class CpuIcmIlsOptimizer<5>;
template class CpuIcmIlsOptimizer<12>;
template class CpuIcmIlsOptimizer<6>;
template class CpuIcmIlsOptimizer<10>;

}  // namespace stlqpp
