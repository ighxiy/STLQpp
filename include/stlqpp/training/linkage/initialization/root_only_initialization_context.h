#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/root_codebooks.h"
#include "stlqpp/training/linkage/initialization/root_only_initialization_pipeline.h"
#include "stlqpp/training/linkage/scheduler/linkage_resource_statistics.h"

#include <memory>
#include <optional>

namespace stlqpp {

template <int M>
class RootOnlyInitializationTopologyContext {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  virtual ~RootOnlyInitializationTopologyContext() = default;
  RootOnlyInitializationTopologyContext(
      const RootOnlyInitializationTopologyContext&) = delete;
  RootOnlyInitializationTopologyContext& operator=(
      const RootOnlyInitializationTopologyContext&) = delete;

  [[nodiscard]] virtual ArrayView<
      RootOnlyInitializationClusterExecutor<M>*> executors() noexcept = 0;
  [[nodiscard]] virtual std::optional<LinkageResourceStatistics>
  resource_statistics() const noexcept = 0;

 protected:
  RootOnlyInitializationTopologyContext() = default;
};

template <int M>
class RootOnlyInitializationTopologyContextFactory {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  virtual ~RootOnlyInitializationTopologyContextFactory() = default;
  RootOnlyInitializationTopologyContextFactory(
      const RootOnlyInitializationTopologyContextFactory&) = delete;
  RootOnlyInitializationTopologyContextFactory& operator=(
      const RootOnlyInitializationTopologyContextFactory&) = delete;

  [[nodiscard]] virtual StatusOr<std::unique_ptr<
      RootOnlyInitializationTopologyContext<M>>> Create(
      const RootCodebooks<M>& codebooks) = 0;

 protected:
  RootOnlyInitializationTopologyContextFactory() = default;
};

}  // namespace stlqpp
