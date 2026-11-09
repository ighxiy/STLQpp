#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/model/model_space_transform.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/update/training_model.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

// Bounded list-major source for a final base full-topology pass. It rotates
// one CID and reconstructs the already encoded basic roots against the
// immutable topology model. Candidate-graph construction belongs to parallel
// topology executors, never this source-ordered producer.
template <int M>
class BaseListFullTopologySource final
    : public FullTopologyClusterSource<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<BaseListFullTopologySource<M>> Create(
      const BaseListReader<M>& lists,
      const TrainingModel<M>& topology_model,
      const F32GemmBackend& gemm,
      ArrayView<const std::uint8_t> bad_cluster_mask,
      bool collect_source_norm2 = false);

  BaseListFullTopologySource(BaseListFullTopologySource&&) noexcept = default;
  BaseListFullTopologySource& operator=(
      BaseListFullTopologySource&&) noexcept = default;
  BaseListFullTopologySource(const BaseListFullTopologySource&) = delete;
  BaseListFullTopologySource& operator=(
      const BaseListFullTopologySource&) = delete;

  [[nodiscard]] StatusOr<std::optional<FullTopologyClusterWork<M>>>
  Next() override;
  [[nodiscard]] Status SkipTo(std::uint32_t next_cid) override;
  [[nodiscard]] Status SkipCompletedCid(std::uint32_t cid) override {
    return SkipTo(cid + 1U);
  }

 private:
  BaseListFullTopologySource(
      const BaseListReader<M>& lists,
      const TrainingModel<M>& topology_model,
      const F32GemmBackend& gemm,
      ModelSpaceTransform transform,
      std::vector<std::uint8_t> bad_cluster_mask,
      bool collect_source_norm2) noexcept
      : lists_(&lists),
        topology_model_(&topology_model),
        gemm_(&gemm),
        transform_(std::move(transform)),
        bad_cluster_mask_(std::move(bad_cluster_mask)),
        collect_source_norm2_(collect_source_norm2) {}

  [[nodiscard]] Status Rotate(
      ColumnMajorMatrixF32View source,
      std::vector<float>* destination) const;
  [[nodiscard]] Status ReconstructBasic(
      const LoadedBaseListCid<M>& loaded,
      FullTopologyClusterWork<M>* work) const;

  const BaseListReader<M>* lists_ = nullptr;
  const TrainingModel<M>* topology_model_ = nullptr;
  const F32GemmBackend* gemm_ = nullptr;
  ModelSpaceTransform transform_;
  std::vector<std::uint8_t> bad_cluster_mask_;
  bool collect_source_norm2_ = false;
  std::uint32_t next_cid_ = 0U;
};

extern template class BaseListFullTopologySource<5>;
extern template class BaseListFullTopologySource<10>;

}  // namespace stlqpp
