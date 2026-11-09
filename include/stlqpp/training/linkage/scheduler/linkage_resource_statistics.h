#pragma once

#include <cstdint>

namespace stlqpp {

// End-of-stage observation of the bounded CUDA linkage owner. Values are
// gathered from the construction resource plan and scheduler counters; no
// device query or branch is added to a linkage kernel.
struct LinkageResourceStatistics final {
  std::uint64_t scheduler_backend_count = 0U;
  std::uint64_t direct_backend_count = 0U;
  std::uint64_t device_visible_bytes = 0U;
  std::uint64_t device_free_before_construction_bytes = 0U;
  std::uint64_t fixed_workspace_resident_bytes = 0U;
  std::uint64_t device_free_after_workspace_bytes = 0U;
  std::uint64_t idle_rfull_reserve_bytes = 0U;
  std::uint64_t deferred_workspace_reserve_bytes = 0U;
  std::uint64_t runtime_safety_reserve_bytes = 0U;
  std::uint64_t live_registration_capacity_bytes = 0U;
  std::uint64_t live_registration_peak_bytes = 0U;
  std::uint64_t worker_idle_cache_peak_sum_bytes = 0U;
  std::uint64_t idle_cache_evictions = 0U;
  std::uint64_t allocation_retries = 0U;
  std::uint64_t evaluation_source_tasks = 0U;
  std::uint64_t evaluation_transport_batches = 0U;
  std::uint64_t evaluation_cross_cluster_transport_batches = 0U;
  std::uint64_t evaluation_numerical_batches = 0U;
  std::uint64_t evaluation_cross_cluster_numerical_batches = 0U;
  std::uint64_t merged_numerical_source_tasks = 0U;
  std::uint64_t maximum_source_tasks_per_transport_batch = 0U;
  std::uint64_t maximum_source_tasks_per_numerical_batch = 0U;
  std::uint64_t evaluation_nodes = 0U;
  std::uint64_t evaluation_pairs = 0U;
  std::uint64_t update_requests = 0U;
  std::uint64_t update_batches = 0U;
  std::uint64_t update_columns = 0U;
  std::uint64_t ordinary_direct_lane_count = 0U;
  std::uint64_t virtual_root_model_resident_bytes = 0U;
  std::uint64_t virtual_root_workspace_resident_bytes = 0U;
};

}  // namespace stlqpp
