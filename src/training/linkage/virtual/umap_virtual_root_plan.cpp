#include "stlqpp/training/linkage/virtual/umap_virtual_root_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr float kUmapEpsilon = 1.0e-6F;

float DirectedProbability(float distance,
                          float rho,
                          float sigma) noexcept {
  const float residual = distance - rho;
  return residual <= 0.0F ? 1.0F : std::exp(-residual / sigma);
}

float SymmetricWeight(float forward, float reverse) noexcept {
  return forward + reverse - forward * reverse;
}

float FindDirectedProbability(const PreparedKnnRows& graph,
                              ArrayView<const float> directed,
                              std::uint32_t width,
                              std::uint32_t source,
                              std::uint32_t destination) noexcept {
  const PreparedKnnRow row = graph.row(source);
  const std::size_t begin =
      static_cast<std::size_t>(source) * width;
  for (std::uint32_t slot = 0U; slot < width; ++slot) {
    if (row.local_ids[slot] == destination) {
      return directed[begin + slot];
    }
  }
  return 0.0F;
}

bool BetterDensity(std::uint32_t left,
                   std::uint32_t right,
                   ArrayView<const float> degrees) noexcept {
  return degrees[left] > degrees[right];
}

}  // namespace

StatusOr<UmapVirtualRootPlan> UmapVirtualRootPlan::Build(
    const PreparedKnnRows& real_graph,
    std::uint32_t requested_roots,
    UmapVirtualRootPolicy policy) {
  const std::uint32_t real_count = real_graph.row_count();
  const std::uint32_t prepared_width = real_graph.neighbors_per_row();
  const std::uint32_t width = policy.neighbor_count == 0U
      ? prepared_width
      : std::min(policy.neighbor_count, prepared_width);
  if (real_count == 0U || requested_roots == 0U || width == 0U ||
      width >= real_count || policy.local_connectivity == 0U ||
      !std::isfinite(policy.overlap_threshold) ||
      policy.overlap_threshold < 0.0F ||
      policy.overlap_threshold > 1.0F) {
    return Status::InvalidArgument(
        "UMAP virtual-root plan has an invalid graph or policy");
  }
  const std::uint32_t root_count = std::min(requested_roots, real_count);
  const std::uint32_t local_connectivity =
      std::min(policy.local_connectivity, width);
  const float target = std::log2(
      static_cast<float>(std::max(width, 2U)));

  std::vector<float> rho(real_count);
  std::vector<float> sigma(real_count, 1.0F);
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    const PreparedKnnRow neighbors = real_graph.row(row);
    const float row_rho = std::max(
        neighbors.squared_distances[local_connectivity - 1U], 0.0F);
    rho[row] = row_rho;
    std::uint32_t unit_count = 0U;
    for (std::uint32_t slot = 0U; slot < width; ++slot) {
      unit_count += neighbors.squared_distances[slot] <=
              row_rho + kUmapEpsilon
          ? 1U
          : 0U;
    }
    if (target <= static_cast<float>(unit_count) + kUmapEpsilon) {
      sigma[row] = kUmapEpsilon;
      continue;
    }
    float low = kUmapEpsilon;
    float high = std::max(
        neighbors.squared_distances[width - 1U], kUmapEpsilon);
    for (std::uint32_t iteration = 0U; iteration < 64U; ++iteration) {
      const float middle = 0.5F * (low + high);
      float probability_sum = 0.0F;
      for (std::uint32_t slot = 0U; slot < width; ++slot) {
        probability_sum += DirectedProbability(
            neighbors.squared_distances[slot], row_rho, middle);
      }
      if (probability_sum > target) {
        high = middle;
      } else {
        low = middle;
      }
    }
    sigma[row] = high;
  }

  const std::size_t edge_count =
      static_cast<std::size_t>(real_count) * width;
  std::vector<float> directed(edge_count);
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    const PreparedKnnRow neighbors = real_graph.row(row);
    const std::size_t begin = static_cast<std::size_t>(row) * width;
    for (std::uint32_t slot = 0U; slot < width; ++slot) {
      directed[begin + slot] = DirectedProbability(
          neighbors.squared_distances[slot], rho[row], sigma[row]);
    }
  }
  std::vector<float> symmetric(edge_count);
  std::vector<float> degree(real_count, 0.0F);
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    const PreparedKnnRow neighbors = real_graph.row(row);
    const std::size_t begin = static_cast<std::size_t>(row) * width;
    for (std::uint32_t slot = 0U; slot < width; ++slot) {
      const std::uint32_t neighbor = neighbors.local_ids[slot];
      const float weight = SymmetricWeight(
          directed[begin + slot],
          FindDirectedProbability(
              real_graph, directed, width, neighbor, row));
      symmetric[begin + slot] = weight;
      degree[row] += weight;
    }
  }

  std::vector<std::uint32_t> by_density(real_count);
  for (std::uint32_t row = 0U; row < real_count; ++row) {
    by_density[row] = row;
  }
  std::sort(by_density.begin(), by_density.end(),
            [&](std::uint32_t left, std::uint32_t right) {
              return BetterDensity(left, right, degree);
            });
  std::vector<std::uint32_t> candidates;
  candidates.reserve(static_cast<std::size_t>(real_count) * 2U);
  if (policy.prefer_density_peaks) {
    for (std::uint32_t row = 0U; row < real_count; ++row) {
      bool peak = true;
      const PreparedKnnRow neighbors = real_graph.row(row);
      for (std::uint32_t slot = 0U; slot < width; ++slot) {
        const std::uint32_t neighbor = neighbors.local_ids[slot];
        if (degree[neighbor] > degree[row]) {
          peak = false;
          break;
        }
      }
      if (peak) {
        candidates.push_back(row);
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](std::uint32_t left, std::uint32_t right) {
                return BetterDensity(left, right, degree);
              });
  }
  candidates.insert(
      candidates.end(), by_density.begin(), by_density.end());

  UmapVirtualRootPlan output;
  output.real_count_ = real_count;
  std::vector<std::uint8_t> selected(real_count, 0U);
  for (const std::uint32_t candidate : candidates) {
    if (output.seed_local_ids_.size() == root_count) {
      break;
    }
    if (selected[candidate] != 0U) {
      continue;
    }
    float maximum_overlap = 0.0F;
    for (const std::uint32_t seed : output.seed_local_ids_) {
      const float forward = FindDirectedProbability(
          real_graph, directed, width, candidate, seed);
      const float reverse = FindDirectedProbability(
          real_graph, directed, width, seed, candidate);
      maximum_overlap = std::max(
          maximum_overlap, SymmetricWeight(forward, reverse));
    }
    if (output.seed_local_ids_.empty() ||
        maximum_overlap <= policy.overlap_threshold) {
      output.seed_local_ids_.push_back(candidate);
      selected[candidate] = 1U;
    }
  }
  for (const std::uint32_t candidate : by_density) {
    if (output.seed_local_ids_.size() == root_count) {
      break;
    }
    if (selected[candidate] == 0U) {
      output.seed_local_ids_.push_back(candidate);
      selected[candidate] = 1U;
    }
  }

  output.offsets_.reserve(root_count + 1U);
  output.offsets_.push_back(0U);
  std::vector<std::pair<float, std::uint32_t>> ranked_neighbors;
  ranked_neighbors.reserve(width);
  const std::uint32_t used_neighbors =
      std::min(policy.anchor_neighbor_k, width);
  for (const std::uint32_t seed : output.seed_local_ids_) {
    output.source_local_ids_.push_back(seed);
    output.raw_weights_.push_back(1.0F);
    ranked_neighbors.clear();
    const PreparedKnnRow neighbors = real_graph.row(seed);
    const std::size_t begin = static_cast<std::size_t>(seed) * width;
    for (std::uint32_t slot = 0U; slot < width; ++slot) {
      ranked_neighbors.push_back(
          {symmetric[begin + slot], neighbors.local_ids[slot]});
    }
    std::partial_sort(
        ranked_neighbors.begin(),
        ranked_neighbors.begin() + used_neighbors,
        ranked_neighbors.end(),
        [](const auto& left, const auto& right) {
          return left.first > right.first;
        });
    float weight_sum = 1.0F;
    for (std::uint32_t slot = 0U; slot < used_neighbors; ++slot) {
      if (!(ranked_neighbors[slot].first > 0.0F)) {
        continue;
      }
      output.source_local_ids_.push_back(ranked_neighbors[slot].second);
      output.raw_weights_.push_back(ranked_neighbors[slot].first);
      weight_sum += ranked_neighbors[slot].first;
    }
    const std::uint32_t end_offset = static_cast<std::uint32_t>(
        output.raw_weights_.size());
    output.inverse_weight_sums_.push_back(1.0F / weight_sum);
    output.offsets_.push_back(end_offset);
  }
  return output;
}

StatusOr<std::vector<float>> UmapVirtualRootPlan::Materialize(
    ColumnMajorMatrixF32View real_vectors) const {
  if (real_vectors.rows == 0U || real_vectors.columns != real_count_ ||
      real_vectors.values == nullptr) {
    return Status::InvalidArgument(
        "virtual-root materialization matrix does not match its plan");
  }
  std::vector<float> output(
      static_cast<std::size_t>(real_vectors.rows) * root_count(), 0.0F);
  for (std::uint32_t root = 0U; root < root_count(); ++root) {
    float* destination = output.data() +
        static_cast<std::size_t>(root) * real_vectors.rows;
    const std::uint32_t begin = offsets_[root];
    const float* seed = real_vectors.column(source_local_ids_[begin]);
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < real_vectors.rows; ++axis) {
      destination[axis] = seed[axis];
    }
    for (std::uint32_t offset = begin + 1U;
         offset < offsets_[root + 1U]; ++offset) {
      const float* source = real_vectors.column(source_local_ids_[offset]);
      const float weight = raw_weights_[offset];
#pragma omp simd
      for (std::uint32_t axis = 0U; axis < real_vectors.rows; ++axis) {
        destination[axis] += weight * source[axis];
      }
    }
    const float inverse_weight_sum = inverse_weight_sums_[root];
#pragma omp simd
    for (std::uint32_t axis = 0U; axis < real_vectors.rows; ++axis) {
      destination[axis] *= inverse_weight_sum;
    }
  }
  return output;
}

}  // namespace stlqpp
