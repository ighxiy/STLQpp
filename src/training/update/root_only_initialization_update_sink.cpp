#include "stlqpp/training/update/root_only_initialization_update_sink.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace stlqpp {

template <int M>
StatusOr<RootOnlyInitializationUpdateSink<M>>
RootOnlyInitializationUpdateSink<M>::Create(
    RootOnlyInitializationTranscriptWriter<M>& transcript,
    std::uint32_t nlist,
    double good_cluster_fraction) {
  if (nlist == 0U || !std::isfinite(good_cluster_fraction) ||
      good_cluster_fraction < 0.0 || good_cluster_fraction > 1.0) {
    return Status::InvalidArgument(
        "root-only initialization sink policy is invalid");
  }
  return RootOnlyInitializationUpdateSink<M>(
      transcript, nlist, good_cluster_fraction);
}

template <int M>
Status RootOnlyInitializationUpdateSink<M>::Commit(
    std::uint64_t sequence,
    RootOnlyInitializationClusterWork<M> work,
    ClusterTopologyResult<M> topology) {
  if (finished_ || sequence != next_cid_ || work.cid != next_cid_ ||
      topology.cid != next_cid_ ||
      topology.dimension != work.dimension ||
      topology.roles.size() != work.real_count() ||
      topology.squared_errors.size() != work.real_count()) {
    return Status::FailedPrecondition(
        "root-only initialization sink received an unordered cluster");
  }
  if (const Status status = transcript_->Append(topology); !status.ok()) {
    return status;
  }
  double sum = 0.0;
  for (const float error : topology.squared_errors) {
    if (!std::isfinite(error) || error < 0.0F) {
      return Status::DataLoss(
          "root-only initialization topology contains an invalid error");
    }
    sum += error;
    if (!has_squared_error_) {
      minimum_squared_error_ = error;
      maximum_squared_error_ = error;
      has_squared_error_ = true;
    } else {
      minimum_squared_error_ = std::min(
          minimum_squared_error_, static_cast<double>(error));
      maximum_squared_error_ = std::max(
          maximum_squared_error_, static_cast<double>(error));
    }
  }
  sums_[next_cid_] = sum;
  counts_[next_cid_] = work.real_count();
  ++next_cid_;
  return Status::Ok();
}

template <int M>
StatusOr<InitializationBadClusterResult>
RootOnlyInitializationUpdateSink<M>::Finish() {
  if (finished_ || next_cid_ != sums_.size()) {
    return Status::FailedPrecondition(
        "root-only initialization sink has not consumed every CID");
  }
  std::vector<double> averages;
  averages.reserve(sums_.size());
  double total_sum = 0.0;
  std::uint64_t total_rows = 0U;
  for (std::size_t cid = 0U; cid < sums_.size(); ++cid) {
    if (counts_[cid] == 0U) {
      continue;
    }
    averages.push_back(sums_[cid] / static_cast<double>(counts_[cid]));
    total_sum += sums_[cid];
    total_rows += counts_[cid];
  }

  double baseline = 0.0;
  if (!averages.empty()) {
    std::sort(averages.begin(), averages.end());
    const std::size_t used = averages.size();
    const long long rounded = std::llround(
        good_cluster_fraction_ * static_cast<double>(used));
    const std::size_t position = static_cast<std::size_t>(
        std::max<long long>(1LL,
            std::min<long long>(rounded, static_cast<long long>(used))));
    baseline = averages[position - 1U];
  }

  InitializationBadClusterResult result;
  result.mask.assign(sums_.size(), 0U);
  result.quantile_baseline = baseline;
  result.real_rows = total_rows;
  result.minimum_squared_error = has_squared_error_
      ? minimum_squared_error_ : 0.0;
  result.maximum_squared_error = has_squared_error_
      ? maximum_squared_error_ : 0.0;
  result.mean_squared_error = total_rows == 0U
      ? 0.0
      : total_sum / static_cast<double>(total_rows);
  for (std::size_t cid = 0U; cid < sums_.size(); ++cid) {
    if (counts_[cid] != 0U &&
        sums_[cid] / static_cast<double>(counts_[cid]) > baseline) {
      result.mask[cid] = 1U;
    }
  }
  finished_ = true;
  return result;
}

template class RootOnlyInitializationUpdateSink<0>;
template class RootOnlyInitializationUpdateSink<5>;
template class RootOnlyInitializationUpdateSink<12>;
template class RootOnlyInitializationUpdateSink<6>;
template class RootOnlyInitializationUpdateSink<10>;

}  // namespace stlqpp
