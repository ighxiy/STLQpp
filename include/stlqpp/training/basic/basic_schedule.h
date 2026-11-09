#pragma once

#include <cstdint>

#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/least_squares_cost_mode.h"

namespace stlqpp {

enum class BasicScoreRule : std::uint8_t {
  kSigned = 0U,
  kAbsolute = 1U,
};

class BasicSchedule final {
 public:
  [[nodiscard]] static constexpr BasicSchedule Training() noexcept {
    return BasicSchedule(BasicScoreRule::kSigned, 4U, 8U, 3U);
  }

  [[nodiscard]] static constexpr BasicSchedule BaseBuild() noexcept {
    return BasicSchedule(BasicScoreRule::kAbsolute, 4U, 32U, 3U);
  }

  [[nodiscard]] static constexpr BasicSchedule TrainingVirtualRoots() noexcept {
    return BasicSchedule(BasicScoreRule::kAbsolute, 4U, 8U, 3U);
  }

  [[nodiscard]] static constexpr BasicSchedule
  FinalBaseVirtualRoots() noexcept {
    return BasicSchedule(BasicScoreRule::kAbsolute, 4U, 32U, 3U);
  }

  [[nodiscard]] static constexpr BasicSchedule Configured(
      BasicScoreRule score_rule, std::uint32_t icm_sweeps,
      std::uint32_t ils_rounds, std::uint32_t perturbations,
      LeastSquaresCostMode least_squares_cost_mode =
          LeastSquaresCostMode::kNormalEquation) noexcept {
    return BasicSchedule(score_rule, icm_sweeps, ils_rounds, perturbations,
                         least_squares_cost_mode);
  }

  // Research convenience for existing sensitivity callers.
  [[nodiscard]] static constexpr BasicSchedule Sensitivity(
      BasicScoreRule score_rule, std::uint32_t icm_sweeps,
      std::uint32_t ils_rounds) noexcept {
    return BasicSchedule(score_rule, icm_sweeps, ils_rounds, 3U);
  }

  [[nodiscard]] constexpr std::uint32_t icm_sweeps() const noexcept {
    return icm_sweeps_;
  }
  [[nodiscard]] constexpr std::uint32_t ils_rounds() const noexcept {
    return ils_rounds_;
  }
  [[nodiscard]] constexpr std::uint32_t perturbations() const noexcept {
    return perturbations_;
  }
  [[nodiscard]] constexpr BasicScoreRule score_rule() const noexcept {
    return score_rule_;
  }
  [[nodiscard]] constexpr LeastSquaresCostMode least_squares_cost_mode()
      const noexcept {
    return least_squares_cost_mode_;
  }

 private:
  constexpr BasicSchedule(BasicScoreRule score_rule, std::uint32_t icm_sweeps,
                          std::uint32_t ils_rounds, std::uint32_t perturbations,
                          LeastSquaresCostMode least_squares_cost_mode =
                              LeastSquaresCostMode::kNormalEquation) noexcept
      : score_rule_(score_rule),
        icm_sweeps_(icm_sweeps),
        ils_rounds_(ils_rounds),
        perturbations_(perturbations),
        least_squares_cost_mode_(least_squares_cost_mode) {}

  BasicScoreRule score_rule_ = BasicScoreRule::kSigned;
  std::uint32_t icm_sweeps_ = 0U;
  std::uint32_t ils_rounds_ = 0U;
  std::uint32_t perturbations_ = 0U;
  LeastSquaresCostMode least_squares_cost_mode_ =
      LeastSquaresCostMode::kNormalEquation;
};

template <int M>
[[nodiscard]] constexpr std::uint32_t EffectiveBasicIlsRounds(
    BasicSchedule schedule,
    std::uint32_t actual_m = FixedModelLayers<M>()) noexcept {
  const std::uint32_t m = M == 0 ? actual_m : M;
  return m <= 1 || schedule.perturbations() == 0U ? 0U : schedule.ils_rounds();
}

template <int M>
[[nodiscard]] constexpr std::uint32_t EffectiveBasicPerturbations(
    BasicSchedule schedule,
    std::uint32_t actual_m = FixedModelLayers<M>()) noexcept {
  const std::uint32_t m = M == 0 ? actual_m : M;
  return schedule.perturbations() < m ? schedule.perturbations() : m;
}

}  // namespace stlqpp
