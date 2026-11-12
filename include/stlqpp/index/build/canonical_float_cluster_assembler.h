#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/model/topology/typed_forest.h"
#include "stlqpp/model/linkage_code_tuple.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

// Neutral handoff accepted from either a full topology build or the final
// fixed-topology replay. Training uses node-major coefficient rows; virtual
// root code zero and real-root code zero are the native CID and are implicit.
// Assemble is the final-forest liveness boundary: virtual roots not referenced
// by any real row are removed in stable source order before BFS, codec and norm
// publication. Virtual roots cannot have virtual children in TypedForestView,
// so direct real-parent references are the complete liveness relation.
template <int M>
struct FinalForestClusterView final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  ArrayView<const std::uint32_t> real_ids;
  TypedForestView forest;
  ArrayView<const LinkageCodeTuple<M>> real_supports;
  ArrayView<const float> real_coefficients_node_major;
  // Original unrotated vector norms in the same source-row order as
  // real_ids. Empty for classic-only publication.
  ArrayView<const float> source_real_norm2;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const float> virtual_coefficients_node_major;
};

template <int M>
class OwnedCanonicalFloatCluster final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] CanonicalFloatClusterView view() const noexcept;

 private:
  template <int N>
  friend class CanonicalFloatClusterAssembler;

  std::uint32_t m_ = M;
  std::uint32_t cid_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::vector<std::uint32_t> real_ids_;
  std::vector<std::uint32_t> parent_local_1based_;
  std::vector<std::uint8_t> real_codes_small_;
  std::vector<std::uint8_t> edge_code0_;
  std::vector<std::uint8_t> virtual_codes_small_;
  std::vector<float> coefficients_layer_major_;
  std::vector<float> source_real_norm2_;
};

template <int M>
class CanonicalFloatClusterAssembler final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<OwnedCanonicalFloatCluster<M>> Assemble(
      const FinalForestClusterView<M>& source);
};

extern template class OwnedCanonicalFloatCluster<5>;
extern template class OwnedCanonicalFloatCluster<10>;
extern template class CanonicalFloatClusterAssembler<5>;
extern template class CanonicalFloatClusterAssembler<10>;

}  // namespace stlqpp
