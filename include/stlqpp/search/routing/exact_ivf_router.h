#pragma once

#include "stlqpp/core/array_view.h"

#include <cstdint>
#include <vector>

namespace stlqpp {

template <int M>
class ResidentIndex;

// Native STLQ routing is cosine routing over C_root[0]. Query table entries
// hold q dot C_root[0][cid]; the index owns one inverse centroid norm per CID.
class TrustedNativeIvfView final {
 public:
  [[nodiscard]] std::uint32_t nlist() const noexcept { return nlist_; }
  [[nodiscard]] float root_inverse_norm(std::uint32_t cid) const noexcept {
    return root_inverse_norms_[cid];
  }

 private:
  template <int M>
  friend class ResidentIndex;

  TrustedNativeIvfView(const float* root_inverse_norms,
                       std::uint32_t nlist) noexcept
      : root_inverse_norms_(root_inverse_norms), nlist_(nlist) {}

  const float* root_inverse_norms_ = nullptr;
  std::uint32_t nlist_ = 0U;
};

struct IvfRoute final {
  std::uint32_t cid = 0U;
  float score = 0.0F;
};

// Query-owner scratch. Storage is allocated before the query loop and reused
// for every query handled by that owner.
class ExactIvfRouteWorkspace final {
 public:
  explicit ExactIvfRouteWorkspace(std::uint32_t nlist)
      : selected_cids_(nlist), heap_scores_(nlist) {}

  [[nodiscard]] std::uint32_t capacity() const noexcept {
    return static_cast<std::uint32_t>(selected_cids_.size());
  }
  [[nodiscard]] ArrayView<const std::uint32_t> selected() const noexcept {
    return {selected_cids_.data(), selected_count_};
  }

 private:
  friend void SelectExactIvfRoutes(
      const TrustedNativeIvfView&,
      ArrayView<const float>,
      std::uint32_t,
      ExactIvfRouteWorkspace*) noexcept;

  // Matches the registered DEV selector: IDs and min-heap scores are separate
  // hot arrays. Only the selected CID prefix survives route selection.
  std::vector<std::uint32_t> selected_cids_;
  std::vector<float> heap_scores_;
  std::uint32_t selected_count_ = 0U;
};

// Inputs are trusted by ResidentIndex, TrustedQueryTables and SearchOptions.
// The selected set is exact and the returned prefix is ordered by descending
// cosine score, then ascending CID. No allocation occurs here.
void SelectExactIvfRoutes(const TrustedNativeIvfView& index,
                          ArrayView<const float> root0_dot_products,
                          std::uint32_t nprobe,
                          ExactIvfRouteWorkspace* workspace) noexcept;

}  // namespace stlqpp
