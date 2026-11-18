#pragma once

#include "stlqpp/runtime/cpu/parallel_for.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stlqpp {

// Cold norm populations only: one bounded merge buffer, no per-row tasks.
// Sorting independent ranges and merging preserves the same sorted values.
template <class T>
void ParallelSort(std::vector<T>& values, const ParallelForExecutor& parallel) {
  const auto parts = static_cast<std::uint32_t>(std::min<std::size_t>(
      parallel.worker_count(), values.size() / 65536U));
  if (parts <= 1U) {
    std::sort(values.begin(), values.end());
    return;
  }
  struct Work {
    T* input;
    T* output;
    std::size_t count;
    std::uint32_t parts;
    std::uint32_t width;
  } work{values.data(), nullptr, values.size(), parts, 1U};
  parallel.Run(parts, &work, [](void* opaque, std::uint32_t, std::uint32_t item) noexcept {
    auto& w = *static_cast<Work*>(opaque);
    std::sort(w.input + w.count * item / w.parts,
              w.input + w.count * (item + 1U) / w.parts);
  });
  std::vector<T> scratch(values.size());
  work.output = scratch.data();
  for (; work.width < parts; work.width *= 2U) {
    const auto jobs = (parts + 2U * work.width - 1U) / (2U * work.width);
    parallel.Run(jobs, &work, [](void* opaque, std::uint32_t, std::uint32_t item) noexcept {
      auto& w = *static_cast<Work*>(opaque);
      const auto first = item * 2U * w.width;
      const auto middle = std::min(first + w.width, w.parts);
      const auto last = std::min(first + 2U * w.width, w.parts);
      const auto begin = w.count * first / w.parts;
      std::merge(w.input + begin, w.input + w.count * middle / w.parts,
                 w.input + w.count * middle / w.parts, w.input + w.count * last / w.parts,
                 w.output + begin);
    });
    std::swap(work.input, work.output);
  }
  if (work.input != values.data()) values.swap(scratch);
}

}  // namespace stlqpp
