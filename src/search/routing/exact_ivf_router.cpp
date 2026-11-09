#include "stlqpp/search/routing/exact_ivf_router.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace stlqpp {
namespace {

bool WorseRoute(float lhs_score, std::uint32_t lhs_id,
                float rhs_score, std::uint32_t rhs_id) noexcept {
  return lhs_score < rhs_score ||
         (lhs_score == rhs_score && lhs_id > rhs_id);
}

void SiftUpMinHeap(float* scores,
                   std::uint32_t* ids,
                   std::uint32_t child) noexcept {
  const float value = scores[child];
  const std::uint32_t id = ids[child];
  while (child != 0U) {
    const std::uint32_t parent = (child - 1U) / 2U;
    if (!WorseRoute(value, id, scores[parent], ids[parent])) {
      break;
    }
    scores[child] = scores[parent];
    ids[child] = ids[parent];
    child = parent;
  }
  scores[child] = value;
  ids[child] = id;
}

void SiftDownMinHeap(float* scores,
                     std::uint32_t* ids,
                     std::uint32_t size) noexcept {
  const float value = scores[0];
  const std::uint32_t id = ids[0];
  std::uint32_t parent = 0U;
  while (true) {
    const std::uint32_t left = 2U * parent + 1U;
    if (left >= size) {
      break;
    }
    const std::uint32_t right = left + 1U;
    std::uint32_t smaller_child = left;
    if (right < size &&
        WorseRoute(scores[right], ids[right], scores[left], ids[left])) {
      smaller_child = right;
    }
    if (!WorseRoute(scores[smaller_child], ids[smaller_child], value, id)) {
      break;
    }
    scores[parent] = scores[smaller_child];
    ids[parent] = ids[smaller_child];
    parent = smaller_child;
  }
  scores[parent] = value;
  ids[parent] = id;
}

}  // namespace

void SelectExactIvfRoutes(const TrustedNativeIvfView& index,
                          ArrayView<const float> root0_dot_products,
                          std::uint32_t nprobe,
                          ExactIvfRouteWorkspace* workspace) noexcept {
  assert(root0_dot_products.size() == index.nlist());
  assert(workspace != nullptr && workspace->capacity() == index.nlist());
  assert(nprobe != 0U && nprobe <= index.nlist());

  float* heap_scores = workspace->heap_scores_.data();
  std::uint32_t* heap_ids = workspace->selected_cids_.data();
  for (std::uint32_t cid = 0U; cid < index.nlist(); ++cid) {
    const float score =
        root0_dot_products[cid] * index.root_inverse_norm(cid);
    if (cid < nprobe) {
      heap_scores[cid] = score;
      heap_ids[cid] = cid;
      SiftUpMinHeap(heap_scores, heap_ids, cid);
    } else if (WorseRoute(heap_scores[0], heap_ids[0], score, cid)) {
      heap_scores[0] = score;
      heap_ids[0] = cid;
      SiftDownMinHeap(heap_scores, heap_ids, nprobe);
    }
  }
  std::sort(heap_ids, heap_ids + nprobe,
            [&](std::uint32_t lhs, std::uint32_t rhs) {
              const float lhs_score =
                  root0_dot_products[lhs] * index.root_inverse_norm(lhs);
              const float rhs_score =
                  root0_dot_products[rhs] * index.root_inverse_norm(rhs);
              return lhs_score > rhs_score ||
                     (lhs_score == rhs_score && lhs < rhs);
            });
  workspace->selected_count_ = nprobe;
}

}  // namespace stlqpp
