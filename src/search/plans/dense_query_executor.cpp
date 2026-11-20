#include "stlqpp/search/plans/dense_query_executor.h"

#include "stlqpp/search/kernels/dense_avx2.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <new>

namespace stlqpp {

template <int M>
StatusOr<DenseQueryWorkspace<M>> DenseQueryWorkspace<M>::Create(
    const ResidentIndex<M>& index,
    std::uint32_t top_k) {
  if (top_k == 0U) {
    return Status::InvalidArgument(
        "dense query workspace top_k must be positive");
  }
  return DenseQueryWorkspace<M>(&index, top_k);
}

template <int M>
Status DenseQueryWorkspace<M>::PrepareIntProfiling() {
  try {
    profile_distance_scratch_.resize(owner_->max_node_count());
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "dense int profile distance scratch allocation failed");
  }
}

template <int M>
StatusOr<DenseQueryExecutor<M>> DenseQueryExecutor<M>::Create(
    const ResidentIndex<M>& index,
    const DenseCpuPlan& plan) {
  (void)plan;
  return DenseQueryExecutor<M>(&index);
}

template <int M>
Status DenseQueryExecutor<M>::Search(
    const TrustedQueryTables<M>& query,
    const SearchOptions& options,
    DenseQueryWorkspace<M>* workspace) const {
  if (const Status option_status = options.Validate(index_->shape());
      !option_status.ok()) {
    return option_status;
  }
  if (query.root0_entries() != index_->nlist()) {
    return Status::InvalidArgument(
        "query root0 table does not span native IVF nlist");
  }
  if (!query.selected_routes().empty() &&
      query.selected_routes().size() != options.nprobe) {
    return Status::InvalidArgument("selected IVF route count differs from nprobe");
  }
  if (workspace == nullptr || workspace->owner_ != index_ ||
      workspace->top_k_.capacity() != options.top_k) {
    return Status::InvalidArgument(
        "dense query workspace does not match this index/request");
  }

  SearchTrusted(query, options.nprobe, workspace);
  return Status::Ok();
}

template <int M>
void DenseQueryExecutor<M>::SearchTrusted(
    const TrustedQueryTables<M>& query,
    std::uint32_t nprobe,
    DenseQueryWorkspace<M>* workspace) const noexcept {
  SearchTrustedImpl<false>(query, nprobe, workspace, nullptr, nullptr);
}

template <int M>
void DenseQueryExecutor<M>::SearchTrustedTo(
    const TrustedQueryTables<M>& query,
    std::uint32_t nprobe,
    DenseQueryWorkspace<M>* workspace,
    std::uint64_t* ids,
    float* squared_distances) const noexcept {
  SearchTrustedImpl<true>(query, nprobe, workspace, ids,
                          squared_distances);
}

template <int M>
void DenseQueryExecutor<M>::ScanTrustedProfiled(
    const TrustedQueryTables<M>& query,
    std::uint32_t nprobe,
    DenseQueryWorkspace<M>* workspace,
    QueryTopK* top_k,
    DenseIntQueryProfile* profile) const noexcept {
  using Clock = std::chrono::steady_clock;
  assert(query.root0_entries() == index_->nlist());
  assert(workspace != nullptr && workspace->owner_ == index_);
  assert(top_k != nullptr && top_k->capacity() != 0U);
  assert(profile != nullptr);
  assert(index_->coefficient_mode() == SearchCoefficientMode::kInt8);
  assert(index_->score_mode() == SearchScoreMode::kScaled);
  assert(index_->parent_access_mode() == ParentAccessMode::kMaterialized);
  assert(workspace->profile_distance_scratch_.size() >=
         index_->max_node_count());
  assert(nprobe != 0U && nprobe <= index_->nlist());

  top_k->Reset();
  auto begin = Clock::now();
  auto routes = query.selected_routes();
  if (routes.empty()) {
    SelectExactIvfRoutes(index_->native_ivf(), query.root0_values(),
                         nprobe, &workspace->routes_);
    routes = workspace->routes_.selected();
  }
  profile->route_worker_seconds += std::chrono::duration<double>(
      Clock::now() - begin).count();

  DenseIntScanTiming timing;
  const Norm8CodebookView norm8 = index_->norm8();
  const ScaledScoreTableView scaled = index_->scaled_score();
  for (const std::uint32_t cid : routes) {
    const ResidentCluster<M>& cluster = index_->cluster(cid);
    if (cluster.empty()) continue;
    const auto view = cluster.view(norm8, scaled);
    if (index_->materialized_parent_width() == MaterializedParentWidth::kU16) {
      PrepareAndScanDenseMaterializedU16ProfiledAvx2(
          query, view, &workspace->scaled_tables_, top_k,
          workspace->dot_scratch_.data(),
          workspace->profile_distance_scratch_.data(), &timing);
    } else if (index_->materialized_parent_width() ==
               MaterializedParentWidth::kU32) {
      PrepareAndScanDenseMaterializedU32ProfiledAvx2(
          query, view, &workspace->scaled_tables_, top_k,
          workspace->dot_scratch_.data(),
          workspace->profile_distance_scratch_.data(), &timing);
    } else {
      PrepareAndScanDenseMaterializedProfiledAvx2(
          query, view, &workspace->scaled_tables_, top_k,
          workspace->dot_scratch_.data(),
          workspace->profile_distance_scratch_.data(), &timing);
    }
  }
  profile->scan_worker_seconds += timing.scan_seconds;
  profile->online_topk_worker_seconds += timing.online_topk_seconds;
}

template <int M>
void DenseQueryExecutor<M>::FinalizeProfiled(
    QueryTopK* top_k,
    std::uint64_t* ids,
    float* squared_distances,
    DenseIntQueryProfile* profile) const noexcept {
  assert(top_k != nullptr && ids != nullptr && squared_distances != nullptr);
  assert(profile != nullptr);
  const auto begin = std::chrono::steady_clock::now();
  top_k->FinalizeTo(ids, squared_distances);
  profile->finalize_worker_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
}

template <int M>
template <bool DirectOutput>
void DenseQueryExecutor<M>::SearchTrustedImpl(
    const TrustedQueryTables<M>& query,
    std::uint32_t nprobe,
    DenseQueryWorkspace<M>* workspace,
    std::uint64_t* ids,
    float* squared_distances) const noexcept {
  assert(query.root0_entries() == index_->nlist());
  assert(workspace != nullptr && workspace->owner_ == index_);
  assert(nprobe != 0U && nprobe <= index_->nlist());
  if constexpr (DirectOutput) {
    assert(ids != nullptr && squared_distances != nullptr);
  } else {
    (void)ids;
    (void)squared_distances;
  }

  workspace->top_k_.Reset();
  auto routes = query.selected_routes();
  if (routes.empty()) {
    SelectExactIvfRoutes(index_->native_ivf(), query.root0_values(),
                         nprobe, &workspace->routes_);
    routes = workspace->routes_.selected();
  }
  if (index_->score_mode() == SearchScoreMode::kCosineRecon ||
      IsMipsScoreMode(index_->score_mode())) {
    const bool normalize =
        index_->score_mode() == SearchScoreMode::kCosineRecon;
    const bool mips_scaled = index_->score_mode() == SearchScoreMode::kMipsScaled;
    if (index_->coefficient_mode() == SearchCoefficientMode::kFloat) {
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        ScanFloatAngularDenseAvx2(
            query, cluster.float_view(), &workspace->top_k_,
            workspace->dot_scratch_.data(), normalize, mips_scaled);
      }
    } else {
      const Norm8CodebookView norm8 = index_->norm8();
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        PrepareAndScanAngularDenseAvx2(
            query, cluster.view(norm8, index_->scaled_score()), &workspace->scaled_tables_,
            &workspace->top_k_, workspace->dot_scratch_.data(), normalize,
            index_->cosine_inverse_norms(), mips_scaled);
      }
    }
    if constexpr (DirectOutput) {
      workspace->top_k_.FinalizeTo(ids, squared_distances);
    } else {
      (void)workspace->top_k_.Finalize();
    }
    return;
  }
  if (index_->coefficient_mode() == SearchCoefficientMode::kFloat) {
    if (index_->score_mode() == SearchScoreMode::kScaled) {
      if (index_->parent_access_mode() == ParentAccessMode::kMaterialized) {
        if (index_->materialized_parent_width() ==
            MaterializedParentWidth::kU16) {
          for (const std::uint32_t cid : routes) {
            const ResidentCluster<M>& cluster = index_->cluster(cid);
            if (cluster.empty()) continue;
            ScanFloatScaledDenseMaterializedU16Avx2(
                query, cluster.float_view(), &workspace->top_k_,
                workspace->dot_scratch_.data());
          }
        } else if (index_->materialized_parent_width() ==
                   MaterializedParentWidth::kU32) {
          for (const std::uint32_t cid : routes) {
            const ResidentCluster<M>& cluster = index_->cluster(cid);
            if (cluster.empty()) continue;
            ScanFloatScaledDenseMaterializedU32Avx2(
                query, cluster.float_view(), &workspace->top_k_,
                workspace->dot_scratch_.data());
          }
        } else {
          for (const std::uint32_t cid : routes) {
            const ResidentCluster<M>& cluster = index_->cluster(cid);
            if (cluster.empty()) continue;
            ScanFloatScaledDenseMaterializedAvx2(
                query, cluster.float_view(), &workspace->top_k_,
                workspace->dot_scratch_.data());
          }
        }
      } else {
        for (const std::uint32_t cid : routes) {
          const ResidentCluster<M>& cluster = index_->cluster(cid);
          if (cluster.empty()) continue;
          ScanFloatScaledDenseNativeLoudsAvx2(
              query, cluster.float_view(), &workspace->top_k_,
              workspace->dot_scratch_.data());
        }
      }
      if constexpr (DirectOutput) {
        workspace->top_k_.FinalizeTo(ids, squared_distances);
      } else {
        (void)workspace->top_k_.Finalize();
      }
      return;
    }
    if (index_->parent_access_mode() == ParentAccessMode::kMaterialized) {
      if (index_->materialized_parent_width() ==
          MaterializedParentWidth::kU16) {
        for (const std::uint32_t cid : routes) {
          const ResidentCluster<M>& cluster = index_->cluster(cid);
          if (cluster.empty()) continue;
          ScanFloatDenseMaterializedU16Avx2(
              query, cluster.float_view(), &workspace->top_k_,
              workspace->dot_scratch_.data());
        }
      } else if (index_->materialized_parent_width() ==
                 MaterializedParentWidth::kU32) {
        for (const std::uint32_t cid : routes) {
          const ResidentCluster<M>& cluster = index_->cluster(cid);
          if (cluster.empty()) continue;
          ScanFloatDenseMaterializedU32Avx2(
              query, cluster.float_view(), &workspace->top_k_,
              workspace->dot_scratch_.data());
        }
      } else {
        for (const std::uint32_t cid : routes) {
          const ResidentCluster<M>& cluster = index_->cluster(cid);
          if (cluster.empty()) continue;
          ScanFloatDenseMaterializedAvx2(
              query, cluster.float_view(), &workspace->top_k_,
              workspace->dot_scratch_.data());
        }
      }
    } else {
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        ScanFloatDenseNativeLoudsAvx2(
            query, cluster.float_view(), &workspace->top_k_,
            workspace->dot_scratch_.data());
      }
    }
    if constexpr (DirectOutput) {
      workspace->top_k_.FinalizeTo(ids, squared_distances);
    } else {
      (void)workspace->top_k_.Finalize();
    }
    return;
  }
  const Norm8CodebookView norm8 = index_->norm8();
  const ScaledScoreTableView scaled = index_->scaled_score();
  if (index_->parent_access_mode() == ParentAccessMode::kMaterialized) {
    if (index_->materialized_parent_width() ==
        MaterializedParentWidth::kU16) {
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        const auto view = cluster.view(norm8, scaled);
        PrepareAndScanDenseMaterializedU16Avx2(
            query, view, &workspace->scaled_tables_, &workspace->top_k_,
            workspace->dot_scratch_.data());
      }
    } else if (index_->materialized_parent_width() ==
               MaterializedParentWidth::kU32) {
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        const auto view = cluster.view(norm8, scaled);
        PrepareAndScanDenseMaterializedU32Avx2(
            query, view, &workspace->scaled_tables_, &workspace->top_k_,
            workspace->dot_scratch_.data());
      }
    } else {
      for (const std::uint32_t cid : routes) {
        const ResidentCluster<M>& cluster = index_->cluster(cid);
        if (cluster.empty()) continue;
        const auto view = cluster.view(norm8, scaled);
        PrepareAndScanDenseMaterializedAvx2(
            query, view, &workspace->scaled_tables_, &workspace->top_k_,
            workspace->dot_scratch_.data());
      }
    }
  } else {
    for (const std::uint32_t cid : routes) {
      const ResidentCluster<M>& cluster = index_->cluster(cid);
      if (cluster.empty()) continue;
      const auto view = cluster.view(norm8, scaled);
      PrepareAndScanDenseNativeLoudsAvx2(
          query, view, &workspace->scaled_tables_, &workspace->top_k_,
          workspace->dot_scratch_.data());
    }
  }
  if constexpr (DirectOutput) {
    workspace->top_k_.FinalizeTo(ids, squared_distances);
  } else {
    (void)workspace->top_k_.Finalize();
  }
}

template class DenseQueryWorkspace<5>;
template class DenseQueryWorkspace<12>;
template class DenseQueryWorkspace<6>;
template class DenseQueryWorkspace<10>;
template class DenseQueryExecutor<5>;
template class DenseQueryExecutor<12>;
template class DenseQueryExecutor<6>;
template class DenseQueryExecutor<10>;

}  // namespace stlqpp
