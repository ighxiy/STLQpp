#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/update/edge_codebook_solver.h"
#include "stlqpp/training/update/root_codebook_solver.h"
#include "stlqpp/training/update/rotation_update.h"
#include "stlqpp/training/update/training_model.h"

#include <array>

namespace stlqpp {

template <int M>
struct SolvedRoundModelView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  ModelShape shape;
  ColumnMajorMatrixF32View rotation;
  ColumnMajorMatrixF32View root0;
  ModelLayers<ColumnMajorMatrixF32View, M, 1> root_small{};
  ModelLayers<ColumnMajorMatrixF32View, M> edge{};
};

// Joins the three independently solved portions of a round into one immutable
// model. No model-visible state is changed until the returned object exists.
template <int M>
class RoundModelAssembler final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<TrainingModel<M>> Assemble(
      const ModelShape& shape,
      const RootCodebookSolveResult<M>& root,
      const EdgeCodebookSolveResult<M>& edge,
      const RotationUpdateResult& rotation);

  [[nodiscard]] static StatusOr<TrainingModel<M>> Assemble(
      const SolvedRoundModelView<M>& solved);
};

using RuntimeRoundModelAssembler = RoundModelAssembler<0>;
extern template class RoundModelAssembler<0>;
extern template class RoundModelAssembler<5>;
extern template class RoundModelAssembler<10>;

}  // namespace stlqpp
