#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_cluster_builder.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace stlqpp {

template <int M>
struct RootOnlyInitializationClusterWork final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::uint32_t dimension = 0U;
  std::vector<std::uint32_t> sample_ids;
  std::vector<float> exact_real;
  std::vector<float> basic_real_reconstructions;
  std::vector<LinkageCodeTuple<M>> basic_supports;
  std::vector<float> basic_coefficients;
  std::vector<float> basic_squared_errors;
  std::optional<ValidatedBaseListBasicState> validated_basic = std::nullopt;
  std::vector<std::uint8_t> icm_layer_order;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(sample_ids.size());
  }
  [[nodiscard]] RootOnlyInitializationClusterRequest<M> request()
      const noexcept {
    return {
        .cid = cid,
        .exact_real = {exact_real.data(), dimension, real_count()},
        .basic_real_reconstructions = {
            basic_real_reconstructions.data(), dimension, real_count()},
        .basic_supports = basic_supports,
        .basic_coefficients = basic_coefficients,
        .basic_squared_errors = basic_squared_errors,
        .validated_basic = validated_basic,
        .icm_layer_order = icm_layer_order,
    };
  }
};

template <int M>
class RootOnlyInitializationClusterSource {
 public:
  virtual ~RootOnlyInitializationClusterSource() = default;
  RootOnlyInitializationClusterSource(
      RootOnlyInitializationClusterSource&&) noexcept = default;
  RootOnlyInitializationClusterSource& operator=(
      RootOnlyInitializationClusterSource&&) noexcept = default;
  RootOnlyInitializationClusterSource(
      const RootOnlyInitializationClusterSource&) = delete;
  RootOnlyInitializationClusterSource& operator=(
      const RootOnlyInitializationClusterSource&) = delete;

  [[nodiscard]] virtual StatusOr<std::optional<
      RootOnlyInitializationClusterWork<M>>> Next() = 0;

 protected:
  RootOnlyInitializationClusterSource() = default;
};

template <int M>
class RootOnlyInitializationClusterExecutor {
 public:
  virtual ~RootOnlyInitializationClusterExecutor() = default;
  RootOnlyInitializationClusterExecutor(
      const RootOnlyInitializationClusterExecutor&) = delete;
  RootOnlyInitializationClusterExecutor& operator=(
      const RootOnlyInitializationClusterExecutor&) = delete;

  [[nodiscard]] virtual StatusOr<ClusterTopologyResult<M>> Build(
      const RootOnlyInitializationClusterRequest<M>& request,
      const RootOnlyInitializationTopologyPolicy& policy) = 0;

 protected:
  RootOnlyInitializationClusterExecutor() = default;
};

template <int M>
class RootOnlyInitializationBuilderExecutor final
    : public RootOnlyInitializationClusterExecutor<M> {
 public:
  explicit RootOnlyInitializationBuilderExecutor(
      const RootOnlyInitializationClusterBuilder<M>& builder) noexcept
      : builder_(&builder) {}

  [[nodiscard]] StatusOr<ClusterTopologyResult<M>> Build(
      const RootOnlyInitializationClusterRequest<M>& request,
      const RootOnlyInitializationTopologyPolicy& policy) override {
    return builder_->Build(request, policy);
  }

 private:
  const RootOnlyInitializationClusterBuilder<M>* builder_ = nullptr;
};

template <int M>
class RootOnlyInitializationClusterSink {
 public:
  virtual ~RootOnlyInitializationClusterSink() = default;
  RootOnlyInitializationClusterSink(
      RootOnlyInitializationClusterSink&&) noexcept = default;
  RootOnlyInitializationClusterSink& operator=(
      RootOnlyInitializationClusterSink&&) noexcept = default;
  RootOnlyInitializationClusterSink(
      const RootOnlyInitializationClusterSink&) = delete;
  RootOnlyInitializationClusterSink& operator=(
      const RootOnlyInitializationClusterSink&) = delete;

  [[nodiscard]] virtual Status Commit(
      std::uint64_t sequence,
      RootOnlyInitializationClusterWork<M> work,
      ClusterTopologyResult<M> topology) = 0;

 protected:
  RootOnlyInitializationClusterSink() = default;
};

struct RootOnlyInitializationPipelineOptions final {
  std::uint32_t maximum_inflight_clusters = 0U;
  std::uint32_t maximum_prefetched_clusters = 0U;
  std::uint32_t progress_total_clusters = 0U;
};

struct RootOnlyInitializationWorkerStatistics final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  double build_seconds = 0.0;
};

struct RootOnlyInitializationPipelineStatistics final {
  std::uint64_t clusters = 0U;
  std::uint64_t real_rows = 0U;
  double wall_seconds = 0.0;
  std::vector<RootOnlyInitializationWorkerStatistics> workers;
};

template <int M>
class RootOnlyInitializationPipeline final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<
      RootOnlyInitializationPipelineStatistics> Run(
      RootOnlyInitializationClusterSource<M>& source,
      RootOnlyInitializationClusterSink<M>& sink,
      ArrayView<RootOnlyInitializationClusterExecutor<M>*> executors,
      const RootOnlyInitializationTopologyPolicy& policy,
      RootOnlyInitializationPipelineOptions options);
};

extern template class RootOnlyInitializationPipeline<5>;
extern template class RootOnlyInitializationPipeline<10>;

}  // namespace stlqpp
