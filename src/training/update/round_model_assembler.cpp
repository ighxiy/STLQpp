#include "stlqpp/training/update/round_model_assembler.h"

#include "stlqpp/model/product_codebooks.h"

#include <cstdint>

namespace stlqpp {

template <int M>
StatusOr<TrainingModel<M>> RoundModelAssembler<M>::Assemble(
    const ModelShape& shape,
    const RootCodebookSolveResult<M>& root,
    const EdgeCodebookSolveResult<M>& edge,
    const RotationUpdateResult& rotation) {
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if ((M != 0 && shape.m != static_cast<std::uint32_t>(M)) ||
      root.m() != shape.m || edge.m() != shape.m) {
    return Status::InvalidArgument(
        "round model shape disagrees with the fixed-M assembler");
  }

  SolvedRoundModelView<M> solved;
  solved.shape = shape;
  solved.rotation = rotation.rotation();
  solved.root0 = root.root0();
  if constexpr (M == 0) {
    solved.root_small.resize(shape.m - 1U);
    solved.edge.resize(shape.m);
  }
  for (std::uint32_t layer = 1U;
       layer < shape.m; ++layer) {
    solved.root_small[layer - 1U] = root.small_layer(layer);
  }
  for (std::uint32_t layer = 0U;
       layer < shape.m; ++layer) {
    solved.edge[layer] = edge.layer(layer);
  }
  return Assemble(solved);
}

template <int M>
StatusOr<TrainingModel<M>> RoundModelAssembler<M>::Assemble(
    const SolvedRoundModelView<M>& solved) {
  ProductCodebookView<M> codebooks;
  codebooks.shape = solved.shape;
  codebooks.root0 = solved.root0;
  codebooks.root_small = solved.root_small;
  codebooks.edge = solved.edge;
  return TrainingModel<M>::Create(solved.rotation, codebooks);
}

template class RoundModelAssembler<0>;
template class RoundModelAssembler<5>;
template class RoundModelAssembler<12>;
template class RoundModelAssembler<6>;
template class RoundModelAssembler<10>;

}  // namespace stlqpp
