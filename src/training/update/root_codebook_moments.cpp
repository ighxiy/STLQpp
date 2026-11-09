#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/update/root_codebook_moments.h"

#include "stlqpp/model/model_layers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace stlqpp {
namespace {

struct OrderedPartitionReduction final {
  const double* partitions = nullptr;
  float* reduced = nullptr;
  std::size_t values_per_partition = 0U;
  std::uint32_t partition_count = 0U;
};

void ReducePartitionValue(void* opaque,
                          std::uint32_t,
                          std::uint32_t item) noexcept {
  auto& reduction = *static_cast<OrderedPartitionReduction*>(opaque);
  double sum = 0.0;
  for (std::uint32_t partition = 0U;
       partition < reduction.partition_count; ++partition) {
    sum += reduction.partitions[
        static_cast<std::size_t>(partition) *
            reduction.values_per_partition + item];
  }
  reduction.reduced[item] = static_cast<float>(sum);
}

}  // namespace

template <int M>
RootCodebookMoments<M>::RootCodebookMoments(
    std::uint32_t dimension,
    std::uint32_t nlist,
    std::uint32_t partition_count,
    const ParallelForExecutor* partition_parallel,
    const ParallelForExecutor* conversion_parallel,
    std::uint32_t m)
    : m_(m), dimension_(dimension),
      nlist_(nlist),
      partition_count_(partition_count),
      cids_per_partition_((nlist + partition_count - 1U) / partition_count),
      source_squared_norm_partitions_(partition_count, 0.0),
      root_diagonal_(nlist, 0.0),
      root_atom_major_rhs_(
          static_cast<std::size_t>(nlist) * dimension, 0.0),
      root_small_cross_(
          static_cast<std::size_t>(nlist) * small_atoms(), 0.0F),
      cross_masks_(
          static_cast<std::size_t>(nlist) * (this->m() - 1U) *
              kMaskWordsPerLayer,
          0U),
      packed_small_gram_(PackedSmallGramCount() * partition_count, 0.0),
      small_atom_major_rhs_(
          static_cast<std::size_t>(small_atoms()) * dimension * partition_count,
          0.0),
      partition_parallel_(partition_parallel),
      conversion_parallel_(conversion_parallel) {}

template <int M>
Status RootCodebookMoments<M>::EnablePostRootMetric() {
  if (sample_count_ != 0U || prepared_for_solve_) {
    return Status::FailedPrecondition(
        "post-root metric must be enabled before basic accumulation");
  }
  collect_post_root_metric_ = true;
  return Status::Ok();
}

template <int M>
double RootCodebookMoments<M>::source_squared_norm_sum() const noexcept {
  double total = 0.0;
  for (const double partition : source_squared_norm_partitions_) {
    total += partition;
  }
  return total;
}

template <int M>
StatusOr<RootCodebookMoments<M>> RootCodebookMoments<M>::Create(
    std::uint32_t dimension,
    std::uint32_t nlist,
    std::uint32_t partition_count,
    std::uint32_t m) {
  if (m < 2U || m > stlqpp::kMaximumModelLayers || (M != 0 && m != M)) {
    return Status::InvalidArgument("root-codebook moment M is invalid");
  }
  const std::uint32_t atoms = (m - 1U) * 256U;
  const std::size_t packed = static_cast<std::size_t>(atoms) * (atoms + 1U) / 2U;
  if (dimension == 0U || nlist == 0U || partition_count == 0U ||
      static_cast<std::size_t>(dimension) >
          std::numeric_limits<std::size_t>::max() / nlist ||
      static_cast<std::size_t>(nlist) >
          std::numeric_limits<std::size_t>::max() / atoms ||
      packed >
          std::numeric_limits<std::size_t>::max() / partition_count ||
      static_cast<std::size_t>(atoms) * dimension >
          std::numeric_limits<std::size_t>::max() / partition_count) {
    return Status::InvalidArgument(
        "root-codebook moment shape is invalid");
  }
  return RootCodebookMoments<M>(
      dimension, nlist, partition_count, nullptr, nullptr, m);
}

template <int M>
StatusOr<RootCodebookMoments<M>> RootCodebookMoments<M>::Create(
    std::uint32_t dimension,
    std::uint32_t nlist,
    const ParallelForExecutor& partition_parallel,
    std::uint32_t m) {
  return Create(
      dimension, nlist, partition_parallel, partition_parallel, m);
}

template <int M>
StatusOr<RootCodebookMoments<M>> RootCodebookMoments<M>::Create(
    std::uint32_t dimension,
    std::uint32_t nlist,
    const ParallelForExecutor& partition_parallel,
    const ParallelForExecutor& conversion_parallel,
    std::uint32_t m) {
  auto moments = Create(
      dimension, nlist, partition_parallel.worker_count(), m);
  if (!moments.ok()) {
    return moments.status();
  }
  moments.value().partition_parallel_ = &partition_parallel;
  moments.value().conversion_parallel_ = &conversion_parallel;
  return std::move(moments).value();
}

template <int M>
void RootCodebookMoments<M>::RunPartitions(
    void* context, ParallelForTask task) const noexcept {
  if (partition_parallel_ != nullptr && partition_count_ > 1U) {
    partition_parallel_->Run(partition_count_, context, task);
    return;
  }
  for (std::uint32_t partition = 0U;
       partition < partition_count_; ++partition) {
    task(context, 0U, partition);
  }
}

template <int M>
std::uint32_t RootCodebookMoments<M>::output_worker_count() const noexcept {
  return conversion_parallel_ == nullptr
      ? 1U : std::max(1U, conversion_parallel_->worker_count());
}

template <int M>
void RootCodebookMoments<M>::RunOutputTasks(
    std::uint32_t task_count, void* context, ParallelForTask task)
    const noexcept {
  if (conversion_parallel_ != nullptr && task_count > 1U) {
    conversion_parallel_->Run(task_count, context, task);
    return;
  }
  for (std::uint32_t output = 0U; output < task_count; ++output) {
    task(context, 0U, output);
  }
}

template <int M>
struct RootCodebookMoments<M>::AccumulationWork final {
  RootCodebookMoments<M>* moments = nullptr;
  ColumnMajorMatrixF32View exact;
  const BasicEncoding<M>* encoding = nullptr;
  const std::uint32_t* offsets = nullptr;
  const std::uint32_t* samples = nullptr;
};

template <int M>
template <std::uint32_t Capacity>
void RootCodebookMoments<M>::AccumulatePartition(
    void* opaque,
    std::uint32_t,
    std::uint32_t partition) noexcept {
  auto& work = *static_cast<AccumulationWork*>(opaque);
  double source_squared_norm = 0.0;
  double* metric = work.moments->collect_post_root_metric_
      ? &source_squared_norm : nullptr;
  for (std::uint32_t position = work.offsets[partition];
       position < work.offsets[partition + 1U]; ++position) {
    work.moments->template AccumulateSample<Capacity>(
        work.samples[position], work.exact, *work.encoding, partition,
        metric);
  }
  if (metric != nullptr) {
    work.moments->source_squared_norm_partitions_[partition] +=
        source_squared_norm;
  }
}

template <int M>
template <std::uint32_t Capacity>
void RootCodebookMoments<M>::AccumulateSample(
    std::uint32_t sample,
    ColumnMajorMatrixF32View exact,
    const BasicEncoding<M>& encoding,
    std::uint32_t partition,
    double* source_squared_norm) noexcept {
  std::array<std::uint32_t, Capacity - 1U> atoms{};
  std::array<float, Capacity - 1U> small_coefficients{};
  const std::uint32_t cid = encoding.root_code(sample);
  const float root_coefficient = encoding.coefficient(sample, 0U);
  const float* vector = exact.column(sample);
  if (source_squared_norm != nullptr) {
    double squared_norm = 0.0;
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      const double value = static_cast<double>(vector[axis]);
      squared_norm += value * value;
    }
    *source_squared_norm += squared_norm;
  }
  const std::size_t gram_partition_offset =
      static_cast<std::size_t>(partition) * PackedSmallGramCount();
  const std::size_t rhs_partition_offset =
      static_cast<std::size_t>(partition) * SmallRhsCount();
  root_diagonal_[cid] +=
      static_cast<double>(root_coefficient) * root_coefficient;
  double* root_rhs = root_atom_major_rhs_.data() +
      static_cast<std::size_t>(cid) * dimension_;
  for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
    root_rhs[axis] += static_cast<double>(root_coefficient) *
        static_cast<double>(vector[axis]);
  }
  for (std::uint32_t layer = 1U;
       layer < m(); ++layer) {
    const std::uint32_t code = encoding.small_code(sample, layer);
    const std::uint32_t small_layer = layer - 1U;
    const std::uint32_t atom = small_layer * 256U + code;
    const float coefficient = encoding.coefficient(sample, layer);
    atoms[small_layer] = atom;
    small_coefficients[small_layer] = coefficient;
    root_small_cross_[
        static_cast<std::size_t>(cid) * small_atoms() + atom] +=
        root_coefficient * coefficient;
    const std::size_t mask =
        (static_cast<std::size_t>(cid) * (m() - 1U) + small_layer) *
            kMaskWordsPerLayer + code / 64U;
    cross_masks_[mask] |= std::uint64_t{1U} << (code % 64U);
    double* small_rhs = small_atom_major_rhs_.data() +
        rhs_partition_offset + static_cast<std::size_t>(atom) * dimension_;
    for (std::uint32_t axis = 0U; axis < dimension_; ++axis) {
      small_rhs[axis] += static_cast<double>(coefficient) *
          static_cast<double>(vector[axis]);
    }
  }
  for (std::uint32_t left = 0U;
       left < static_cast<std::uint32_t>(m() - 1U); ++left) {
    for (std::uint32_t right = left;
         right < static_cast<std::uint32_t>(m() - 1U); ++right) {
      const std::uint32_t row = std::min(atoms[left], atoms[right]);
      const std::uint32_t column = std::max(atoms[left], atoms[right]);
      packed_small_gram_[gram_partition_offset +
                         PackedUpperIndex(row, column)] +=
          static_cast<double>(small_coefficients[left]) *
          small_coefficients[right];
    }
  }
}

template <int M>
Status RootCodebookMoments<M>::Accumulate(
    ColumnMajorMatrixF32View exact,
    const BasicEncoding<M>& encoding) {
  if constexpr (M == 0) {
    if (m() > kLegacyRuntimeScratchLayers)
      return AccumulateWithCapacity<kWideRuntimeScratchLayers>(exact, encoding);
  }
  return AccumulateWithCapacity<kModelScratchLayers<M>>(exact, encoding);
}

template <int M>
template <std::uint32_t Capacity>
Status RootCodebookMoments<M>::AccumulateWithCapacity(
    ColumnMajorMatrixF32View exact,
    const BasicEncoding<M>& encoding) {
  if constexpr (M == 0) {
    if (encoding.m() != m()) {
      return Status::InvalidArgument("root moment encoding M differs from model");
    }
  }
  if (prepared_for_solve_) {
    return Status::FailedPrecondition(
        "root-codebook moments were already prepared for solve");
  }
  if (exact.rows != dimension_ ||
      exact.columns != encoding.sample_count() ||
      (exact.columns != 0U && exact.values == nullptr)) {
    return Status::InvalidArgument(
        "root-codebook training block has an invalid shape");
  }
  const std::size_t value_count =
      static_cast<std::size_t>(exact.rows) * exact.columns;
  for (std::size_t index = 0U; index < value_count; ++index) {
    if (!std::isfinite(exact.values[index])) {
      return Status::InvalidArgument(
          "root-codebook training block contains non-finite data");
    }
  }

  if (partition_count_ == 1U) {
    double source_squared_norm = 0.0;
    double* metric = collect_post_root_metric_
        ? &source_squared_norm : nullptr;
    for (std::uint32_t sample = 0U; sample < exact.columns; ++sample) {
      const std::uint32_t cid = encoding.root_code(sample);
      if (cid >= nlist_) {
        return Status::InvalidArgument(
            "basic encoding contains an out-of-range native CID");
      }
      const float root_coefficient = encoding.coefficient(sample, 0U);
      if (!std::isfinite(root_coefficient)) {
        return Status::InvalidArgument(
            "basic root coefficient is non-finite");
      }
      for (std::uint32_t layer = 1U;
           layer < m(); ++layer) {
        if (!std::isfinite(encoding.coefficient(sample, layer))) {
          return Status::InvalidArgument(
              "basic small-layer coefficient is non-finite");
        }
      }
      AccumulateSample<Capacity>(sample, exact, encoding, 0U, metric);
    }
    if (metric != nullptr) {
      source_squared_norm_partitions_[0U] += source_squared_norm;
    }
    sample_count_ += exact.columns;
    return Status::Ok();
  }

  std::vector<std::uint32_t> counts(partition_count_, 0U);
  for (std::uint32_t sample = 0U; sample < exact.columns; ++sample) {
    const std::uint32_t cid = encoding.root_code(sample);
    if (cid >= nlist_) {
      return Status::InvalidArgument(
          "basic encoding contains an out-of-range native CID");
    }
    const float root_coefficient = encoding.coefficient(sample, 0U);
    if (!std::isfinite(root_coefficient)) {
      return Status::InvalidArgument(
          "basic root coefficient is non-finite");
    }
    for (std::uint32_t layer = 1U;
         layer < m(); ++layer) {
      const float coefficient = encoding.coefficient(sample, layer);
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "basic small-layer coefficient is non-finite");
      }
    }
    const std::uint32_t partition = std::min(
        partition_count_ - 1U, cid / cids_per_partition_);
    ++counts[partition];
  }

  std::vector<std::uint32_t> offsets(partition_count_ + 1U, 0U);
  for (std::uint32_t partition = 0U;
       partition < partition_count_; ++partition) {
    offsets[partition + 1U] = offsets[partition] + counts[partition];
  }
  std::vector<std::uint32_t> cursors(offsets.begin(), offsets.end() - 1U);
  std::vector<std::uint32_t> samples(exact.columns);
  for (std::uint32_t sample = 0U; sample < exact.columns; ++sample) {
    const std::uint32_t cid = encoding.root_code(sample);
    const std::uint32_t partition = std::min(
        partition_count_ - 1U, cid / cids_per_partition_);
    samples[cursors[partition]++] = sample;
  }
  AccumulationWork work{
      .moments = this,
      .exact = exact,
      .encoding = &encoding,
      .offsets = offsets.data(),
      .samples = samples.data(),
  };
  RunPartitions(&work, &AccumulatePartition<Capacity>);
  sample_count_ += exact.columns;
  return Status::Ok();
}

template <int M>
Status RootCodebookMoments<M>::PrepareForSolve() {
  if (prepared_for_solve_) {
    return Status::Ok();
  }
  if (sample_count_ == 0U) {
    return Status::FailedPrecondition(
        "root-codebook solve requires accumulated samples");
  }
  const std::size_t packed_count = PackedSmallGramCount();
  const std::size_t rhs_count = SmallRhsCount();
  if (packed_count > std::numeric_limits<std::uint32_t>::max() ||
      rhs_count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "root-codebook reduced moment extent exceeds the parallel domain");
  }
  reduced_small_gram_.resize(packed_count);
  reduced_small_atom_major_rhs_.resize(rhs_count);
  OrderedPartitionReduction gram{
      .partitions = packed_small_gram_.data(),
      .reduced = reduced_small_gram_.data(),
      .values_per_partition = packed_count,
      .partition_count = partition_count_,
  };
  OrderedPartitionReduction rhs{
      .partitions = small_atom_major_rhs_.data(),
      .reduced = reduced_small_atom_major_rhs_.data(),
      .values_per_partition = rhs_count,
      .partition_count = partition_count_,
  };
  if (conversion_parallel_ != nullptr) {
    conversion_parallel_->Run(
        static_cast<std::uint32_t>(packed_count),
        &gram, &ReducePartitionValue);
    conversion_parallel_->Run(
        static_cast<std::uint32_t>(rhs_count),
        &rhs, &ReducePartitionValue);
  } else {
    for (std::uint32_t index = 0U;
         index < static_cast<std::uint32_t>(packed_count); ++index) {
      ReducePartitionValue(&gram, 0U, index);
    }
    for (std::uint32_t index = 0U;
         index < static_cast<std::uint32_t>(rhs_count); ++index) {
      ReducePartitionValue(&rhs, 0U, index);
    }
  }
  std::vector<double>().swap(packed_small_gram_);
  std::vector<double>().swap(small_atom_major_rhs_);
  prepared_for_solve_ = true;
  return Status::Ok();
}

template class RootCodebookMoments<0>;
template class RootCodebookMoments<5>;
template class RootCodebookMoments<12>;
template class RootCodebookMoments<6>;
template class RootCodebookMoments<10>;

}  // namespace stlqpp
