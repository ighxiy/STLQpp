#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_pipeline.h"
#include "stlqpp/training/update/root_training_model.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace stlqpp {

// Reads the first basic encoding in native-list order and reconstructs it
// against the just-updated C_root. The stored F32 rows are already in model
// space, because initialization does not update R.
template <int M>
class TrainingListRootInitializationSource final
    : public RootOnlyInitializationClusterSource<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<TrainingListRootInitializationSource<M>>
  Create(const BaseListReader<M>& lists,
         const RootTrainingModel<M>& model,
         std::uint32_t topology_seed);

  TrainingListRootInitializationSource(
      TrainingListRootInitializationSource&&) noexcept = default;
  TrainingListRootInitializationSource& operator=(
      TrainingListRootInitializationSource&&) noexcept = default;
  TrainingListRootInitializationSource(
      const TrainingListRootInitializationSource&) = delete;
  TrainingListRootInitializationSource& operator=(
      const TrainingListRootInitializationSource&) = delete;

  [[nodiscard]] StatusOr<std::optional<
      RootOnlyInitializationClusterWork<M>>> Next() override;

 private:
  TrainingListRootInitializationSource(
      const BaseListReader<M>& lists,
      const RootTrainingModel<M>& model,
      std::uint32_t topology_seed) noexcept
      : lists_(&lists),
        model_(&model),
        topology_seed_(topology_seed) {}

  [[nodiscard]] Status ReconstructBasic(
      const LoadedBaseListCid<M>& loaded,
      RootOnlyInitializationClusterWork<M>* work) const;

  const BaseListReader<M>* lists_ = nullptr;
  const RootTrainingModel<M>* model_ = nullptr;
  std::uint32_t topology_seed_ = 0U;
  std::uint32_t next_cid_ = 0U;
};

extern template class TrainingListRootInitializationSource<5>;
extern template class TrainingListRootInitializationSource<10>;

}  // namespace stlqpp
