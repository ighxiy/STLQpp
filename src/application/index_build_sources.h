#pragma once

#include <memory>
#include <variant>

#include "stlqpp/application/index_source_traits.h"
#include "stlqpp/pipeline/training_basic_pass.h"
#include "stlqpp/training/initialization/root_model_initializer.h"

namespace stlqpp::detail {

// Source-specific ownership is independent for training and final base. These
// bundles do not multiply the lifetime manager by train/base scalar pairs.
template <int M, typename T>
struct IndexTrainingSources final {
  using Scalar = T;
  std::unique_ptr<typename IndexSourceTraits<T>::InitializationFactory>
      initialization_source;
  std::unique_ptr<typename IndexSourceTraits<T>::TrainingFactory>
      training_source;
  std::unique_ptr<SphericalRvqInitializationBackend<T>> root_backend;
  std::unique_ptr<BasicRoundRuntimeFactory<M, T>>
      initialization_basic_runtime;
  std::unique_ptr<BasicRoundRuntimeFactory<M, T>> training_basic_runtime;
  std::unique_ptr<ScalarTrainingBasicPass<M, T>> training_basic;
};

template <int M, typename T>
struct IndexBaseSources final {
  using Scalar = T;
  std::unique_ptr<typename IndexSourceTraits<T>::BaseFactory> base_source;
  std::unique_ptr<BasicRoundRuntimeFactory<M, T>> base_basic_runtime;
};

template <int M>
using IndexTrainingSourceStorage =
    std::variant<IndexTrainingSources<M, std::uint8_t>,
                 IndexTrainingSources<M, float>>;
template <int M>
using IndexBaseSourceStorage =
    std::variant<IndexBaseSources<M, std::uint8_t>, IndexBaseSources<M, float>>;

}  // namespace stlqpp::detail
