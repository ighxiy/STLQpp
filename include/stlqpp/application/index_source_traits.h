#pragma once

#include <cstdint>
#include <variant>

#include "stlqpp/training/io/bvecs_u8_source.h"
#include "stlqpp/training/io/fvecs_f32_source.h"

namespace stlqpp {

using IndexSourceSpec = std::variant<BvecsU8SourceSpec, FvecsF32SourceSpec>;

// Source scalar is selected once at product construction. Runtime execution
// receives concrete U8 or F32 factories without an inner-loop format branch.
template <typename Scalar>
struct IndexSourceTraits;

template <>
struct IndexSourceTraits<std::uint8_t> final {
  using Spec = BvecsU8SourceSpec;
  using InitializationFactory = InitializationBvecsU8SourceFactory;
  using TrainingFactory = TrainingBvecsU8SourceFactory;
  using BaseFactory = BaseBvecsU8SourceFactory;
};

template <>
struct IndexSourceTraits<float> final {
  using Spec = FvecsF32SourceSpec;
  using InitializationFactory = InitializationFvecsF32SourceFactory;
  using TrainingFactory = TrainingFvecsF32SourceFactory;
  using BaseFactory = BaseFvecsF32SourceFactory;
};

}  // namespace stlqpp
