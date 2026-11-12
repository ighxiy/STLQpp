#include "stlqpp/model/model_shape.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/index/build/canonical_float_cluster_assembler.h"
#if STLQPP_HAS_CUDA_NORM
#include "stlqpp/index/norm/reconstruction_norms_cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

// One update per cluster/representation, never per row or candidate.
class NormServiceTimer final {
 public:
  explicit NormServiceTimer(std::atomic<std::int64_t>& total) noexcept
      : total_(total), begin_(std::chrono::steady_clock::now()) {}
  ~NormServiceTimer() {
    total_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin_).count(),
        std::memory_order_relaxed);
  }
 private:
  std::atomic<std::int64_t>& total_;
  std::chrono::steady_clock::time_point begin_;
};

class CodecServiceTimer final {
 public:
  CodecServiceTimer(std::atomic<std::int64_t>& total,
                    std::atomic<std::uint32_t>& active,
                    std::atomic<std::uint32_t>& peak) : timer_(total), active_(active) {
    const auto count = active_.fetch_add(1U, std::memory_order_relaxed) + 1U;
    auto observed = peak.load(std::memory_order_relaxed);
    while (observed < count && !peak.compare_exchange_weak(
               observed, count, std::memory_order_relaxed)) {}
  }
  ~CodecServiceTimer() { active_.fetch_sub(1U, std::memory_order_relaxed); }
 private:
  NormServiceTimer timer_;
  std::atomic<std::uint32_t>& active_;
};

Status ValidateSource(const CanonicalFloatClusterView& source,
                      std::uint32_t expected_m,
                      std::uint32_t nlist,
                      std::uint32_t& real_count,
                      std::uint32_t& linked_count,
                      std::uint32_t& node_count,
                      bool assembled) {
  const auto m = expected_m;
  if (m < 2U || m > stlqpp::kMaximumModelLayers || source.m != m || source.cid >= nlist ||
      source.real_root_count > source.real_ids.size()) {
    return Status::InvalidArgument(
        "final cluster CID or real-root count is invalid");
  }
  if (source.real_ids.size() > std::numeric_limits<std::uint32_t>::max() ||
      static_cast<std::uint64_t>(source.virtual_count) +
              source.real_ids.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "final cluster exceeds canonical u32 row limits");
  }
  real_count = static_cast<std::uint32_t>(source.real_ids.size());
  linked_count = real_count - source.real_root_count;
  node_count = source.virtual_count + real_count;
  if (source.parent_local_1based.size() != real_count ||
      source.real_codes_small.size() !=
          static_cast<std::size_t>(real_count) * (m - 1) ||
      source.edge_code0.size() != real_count ||
      source.virtual_codes_small.size() !=
          static_cast<std::size_t>(source.virtual_count) * (m - 1) ||
      source.coefficients_layer_major.size() !=
          static_cast<std::size_t>(node_count) * m) {
    return Status::InvalidArgument(
        "final cluster array shape disagrees with role counts");
  }
  if (!assembled) {
    for (std::uint32_t real = 0U; real < real_count; ++real) {
      const std::uint32_t parent = source.parent_local_1based[real];
      const std::uint32_t node = source.virtual_count + real;
      if ((real < source.real_root_count && parent != 0U) ||
          (real >= source.real_root_count &&
           (parent == 0U || parent > node))) {
        return Status::InvalidArgument(
            "final cluster parent stream is not canonical and causal");
      }
    }
    for (const float coefficient : source.coefficients_layer_major) {
      if (!std::isfinite(coefficient)) {
        return Status::InvalidArgument(
            "final cluster contains a non-finite float coefficient");
      }
    }
  }
#ifndef NDEBUG
  if (assembled) {
    for (std::uint32_t real = 0U; real < real_count; ++real) {
      const std::uint32_t parent = source.parent_local_1based[real];
      const std::uint32_t node = source.virtual_count + real;
      assert(real < source.real_root_count ? parent == 0U
                                          : parent != 0U && parent <= node);
    }
    for (const float coefficient : source.coefficients_layer_major) {
      assert(std::isfinite(coefficient));
    }
  }
#endif
  if (!source.source_real_norm2.empty() &&
      source.source_real_norm2.size() != real_count) {
    return Status::InvalidArgument(
        "final cluster source norm stream has the wrong row count");
  }
  if (!assembled) {
    for (const float norm2 : source.source_real_norm2) {
      if (!std::isfinite(norm2) || norm2 < 0.0F) {
        return Status::InvalidArgument(
            "final cluster source norm stream contains an invalid value");
      }
    }
  }
#ifndef NDEBUG
  if (assembled) {
    for (const float norm2 : source.source_real_norm2) {
      assert(std::isfinite(norm2) && norm2 >= 0.0F);
    }
  }
#endif
  return Status::Ok();
}

}  // namespace

template <int M>
CanonicalClusterPayloadSource<M>
PreparedCanonicalCluster<M>::payload_source() const noexcept {
  return {
      .m = m(),
      .cid = cid_,
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_,
      .parent_local_1based = parent_local_1based_,
      .real_codes_small = real_codes_small_,
      .edge_code0 = edge_code0_,
      .virtual_codes_small = virtual_codes_small_,
      .coefficients = coefficients_.view(),
  };
}

template <int M>
CanonicalClusterTopologySource<M>
PreparedCanonicalCluster<M>::topology_source() const noexcept {
  return {
      .m = m(),
      .cid = cid_,
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_,
      .parent_local_1based = parent_local_1based_,
      .real_codes_small = real_codes_small_,
      .edge_code0 = edge_code0_,
      .virtual_codes_small = virtual_codes_small_,
  };
}

template <int M>
StatusOr<CanonicalClusterSource<M>>
PreparedCanonicalCluster<M>::MakeWriteSource(
    ArrayView<const std::uint8_t> real_norm8,
    Norm8CodebookView norm8) const {
  if (real_norm8.size() != real_ids_.size() || norm8.centers == nullptr ||
      norm8.center_count != 256U) {
    return Status::InvalidArgument(
        "final cluster norm8 labels or global codebook are invalid");
  }
  return CanonicalClusterSource<M>{
      .m = m(),
      .cid = cid_,
      .virtual_count = virtual_count_,
      .real_root_count = real_root_count_,
      .real_ids = real_ids_,
      .parent_local_1based = parent_local_1based_,
      .real_codes_small = real_codes_small_,
      .edge_code0 = edge_code0_,
      .virtual_codes_small = virtual_codes_small_,
      .coefficients = coefficients_.view(),
      .real_norm8 = real_norm8,
      .norm8 = norm8,
  };
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>> FinalClusterPostprocessor<M>::Prepare(
    const CanonicalFloatClusterView& source,
    const RootSmallCrossBlock<M>& root_cross) const {
  return PrepareImpl(source, root_cross, false);
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalClusterPostprocessor<M>::PrepareAssembled(
    const OwnedCanonicalFloatCluster<M>& source,
    const RootSmallCrossBlock<M>& root_cross) const {
  return PrepareImpl(source.view(), root_cross, true);
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalClusterPostprocessor<M>::PrepareRetainedFloatOnly(
    const CanonicalFloatClusterView& source,
    ArrayView<const float> float_real_norm2,
    std::uint32_t nlist) {
  std::uint32_t real_count = 0U;
  std::uint32_t linked_count = 0U;
  std::uint32_t node_count = 0U;
  if (const Status status = ValidateSource(
          source, M == 0 ? source.m : M, nlist, real_count, linked_count,
          node_count, false);
      !status.ok()) {
    return status;
  }
  if (!float_real_norm2.empty() && float_real_norm2.size() != real_count) {
    return Status::InvalidArgument(
        "retained native-float norm stream has the wrong row count");
  }
  for (const float norm2 : float_real_norm2) {
    if (!std::isfinite(norm2) || norm2 < 0.0F) {
      return Status::DataLoss(
          "retained native-float norm stream contains an invalid value");
    }
  }

  PreparedCanonicalCluster<M> output;
  output.m_ = source.m;
  output.cid_ = source.cid;
  output.virtual_count_ = source.virtual_count;
  output.real_root_count_ = source.real_root_count;
  output.real_ids_.assign(source.real_ids.begin(), source.real_ids.end());
  output.parent_local_1based_.assign(
      source.parent_local_1based.begin(), source.parent_local_1based.end());
  output.real_codes_small_.assign(
      source.real_codes_small.begin(), source.real_codes_small.end());
  output.edge_code0_.assign(source.edge_code0.begin(), source.edge_code0.end());
  output.virtual_codes_small_.assign(
      source.virtual_codes_small.begin(), source.virtual_codes_small.end());
  output.float_coefficients_layer_major_.assign(
      source.coefficients_layer_major.begin(),
      source.coefficients_layer_major.end());
  output.float_real_norm2_.assign(
      float_real_norm2.begin(), float_real_norm2.end());
  output.source_real_norm2_.assign(
      source.source_real_norm2.begin(), source.source_real_norm2.end());
  output.has_quantized_forest_ = false;
  return output;
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>> FinalClusterPostprocessor<M>::PrepareImpl(
    const CanonicalFloatClusterView& source,
    const RootSmallCrossBlock<M>& root_cross,
    bool assembled) const {
  if (!publish_quantized_forest_) {
    std::uint32_t real_count = 0U;
    std::uint32_t linked_count = 0U;
    std::uint32_t node_count = 0U;
    if (!retain_float_forest_) {
      return Status::InvalidArgument(
          "float-only publication must retain the native-float forest");
    }
    if (const Status status = ValidateSource(
            source, codebooks_->shape().m, codebooks_->shape().nlist,
            real_count, linked_count,
            node_count, assembled);
        !status.ok()) {
      return status;
    }
    const CanonicalClusterCodeView<M> codes = {
        .m = source.m,
      .cid = source.cid,
        .virtual_count = source.virtual_count,
        .real_root_count = source.real_root_count,
        .linked_real_count = linked_count,
        .virtual_codes_small = source.virtual_codes_small,
        .real_codes_small = source.real_codes_small,
        .edge_code0 = source.edge_code0,
    };
    auto float_norms = [&]() -> StatusOr<std::vector<float>> {
      if (!prepare_float_norm2_) return std::vector<float>{};
      const NormServiceTimer timer(float_norm_nanoseconds_);
      const FloatReconstructionNormInput<M> norm_input = {
          .codes = codes,
          .parent_local_1based = source.parent_local_1based,
          .coefficients_layer_major = source.coefficients_layer_major,
      };
      return ComputeFloatNorms(norm_input, root_cross, assembled);
    }();
    if (!float_norms.ok()) return float_norms.status();

    PreparedCanonicalCluster<M> output;
    output.m_ = source.m;
  output.cid_ = source.cid;
    output.virtual_count_ = source.virtual_count;
    output.real_root_count_ = source.real_root_count;
    output.real_ids_.assign(source.real_ids.begin(), source.real_ids.end());
    output.parent_local_1based_.assign(
        source.parent_local_1based.begin(), source.parent_local_1based.end());
    output.real_codes_small_.assign(
        source.real_codes_small.begin(), source.real_codes_small.end());
    output.edge_code0_.assign(source.edge_code0.begin(), source.edge_code0.end());
    output.virtual_codes_small_.assign(
        source.virtual_codes_small.begin(), source.virtual_codes_small.end());
    output.float_coefficients_layer_major_.assign(
        source.coefficients_layer_major.begin(),
        source.coefficients_layer_major.end());
    output.float_real_norm2_ = std::move(float_norms).value();
    output.source_real_norm2_.assign(
        source.source_real_norm2.begin(), source.source_real_norm2.end());
    output.has_quantized_forest_ = false;
    return output;
  }
  auto encoded_result = EncodeCoefficientsImpl(
      source, root_cross, assembled);
  if (!encoded_result.ok()) {
    return encoded_result.status();
  }
  auto output = std::move(encoded_result).value();
  if (retain_float_forest_ && prepare_float_norm2_) {
    const NormServiceTimer timer(float_norm_nanoseconds_);
    const CanonicalClusterCodeView<M> float_codes = {
        .m = source.m,
      .cid = source.cid,
        .virtual_count = source.virtual_count,
        .real_root_count = source.real_root_count,
        .linked_real_count = static_cast<std::uint32_t>(
            source.real_ids.size()) - source.real_root_count,
        .virtual_codes_small = source.virtual_codes_small,
        .real_codes_small = source.real_codes_small,
        .edge_code0 = source.edge_code0,
    };
    const FloatReconstructionNormInput<M> norm_input = {
        .codes = float_codes,
        .parent_local_1based = source.parent_local_1based,
        .coefficients_layer_major = source.coefficients_layer_major,
    };
    auto float_norms = ComputeFloatNorms(norm_input, root_cross, assembled);
    if (!float_norms.ok()) {
      return float_norms.status();
    }
    output.float_real_norm2_ = std::move(float_norms).value();
  }
  if (const Status status = ComputeExactNorms(output, root_cross);
      !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalClusterPostprocessor<M>::PrepareFromRetainedFloatForest(
    const CanonicalFloatClusterView& source,
    ArrayView<const float> source_float_real_norm2,
    const RootSmallCrossBlock<M>& root_cross) const {
  if (!publish_quantized_forest_) {
    if (!retain_float_forest_) {
      return Status::InvalidArgument(
          "float-only publication must retain the native-float forest");
    }
    return PrepareRetainedFloatOnly(
        source, source_float_real_norm2, codebooks_->shape().nlist);
  }
  if (retain_float_forest_ && prepare_float_norm2_ &&
      source_float_real_norm2.size() != source.real_ids.size()) {
    return Status::InvalidArgument(
        "retained native-float norm stream has the wrong row count");
  }
  for (const float norm2 : source_float_real_norm2) {
    if (!std::isfinite(norm2) || norm2 < 0.0F) {
      return Status::DataLoss(
          "retained native-float norm stream contains an invalid value");
    }
  }
  auto encoded_result = EncodeCoefficients(source, root_cross);
  if (!encoded_result.ok()) return encoded_result.status();
  auto output = std::move(encoded_result).value();
  if (retain_float_forest_ && prepare_float_norm2_) {
    output.float_real_norm2_.assign(
        source_float_real_norm2.begin(), source_float_real_norm2.end());
  }
  if (const Status status = ComputeExactNorms(output, root_cross);
      !status.ok()) {
    return status;
  }
  return output;
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalClusterPostprocessor<M>::EncodeCoefficients(
    const CanonicalFloatClusterView& source,
    const RootSmallCrossBlock<M>& root_cross) const {
  return EncodeCoefficientsImpl(source, root_cross, false);
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalClusterPostprocessor<M>::EncodeCoefficientsImpl(
    const CanonicalFloatClusterView& source,
    const RootSmallCrossBlock<M>& root_cross,
    bool assembled) const {
  const CodecServiceTimer timer(coefficient_nanoseconds_, coefficient_active_, coefficient_peak_);
  std::uint32_t real_count = 0U;
  std::uint32_t linked_count = 0U;
  std::uint32_t node_count = 0U;
  if (const Status status = ValidateSource(
          source, codebooks_->shape().m, codebooks_->shape().nlist,
          real_count, linked_count,
          node_count, assembled);
      !status.ok()) {
    return status;
  }
  const CanonicalClusterCodeView<M> codes = {
      .m = source.m,
      .cid = source.cid,
      .virtual_count = source.virtual_count,
      .real_root_count = source.real_root_count,
      .linked_real_count = linked_count,
      .virtual_codes_small = source.virtual_codes_small,
      .real_codes_small = source.real_codes_small,
      .edge_code0 = source.edge_code0,
  };
  auto selected_result = grams_->SelectCluster(codes, root_cross);
  if (!selected_result.ok()) {
    return selected_result.status();
  }
  const ScalarInt6Encoder<M> encoder(
      *solver_, coefficient_base_bits_, coefficient_root_bits_,
      coefficient_linkage_bits_,
      coefficient_first_magnitude_quantiles_,
      coefficient_rest_magnitude_quantiles_, coefficient_codec_options_);
  const SelectedScalarInt6EncodingInput<M> coefficient_input = {
      .virtual_root_count = source.virtual_count,
      .real_root_count = source.real_root_count,
      .linked_real_count = linked_count,
      .coefficients_layer_major = source.coefficients_layer_major,
      .selected_atom_grams = &selected_result.value(),
  };
  auto coefficient_result = assembled
      ? encoder.EncodeSelectedValidated(coefficient_input)
      : encoder.EncodeSelected(coefficient_input);
  if (!coefficient_result.ok()) {
    return coefficient_result.status();
  }

  PreparedCanonicalCluster<M> output;
  output.m_ = source.m;
  output.cid_ = source.cid;
  output.virtual_count_ = source.virtual_count;
  output.real_root_count_ = source.real_root_count;
  output.real_ids_.assign(source.real_ids.begin(), source.real_ids.end());
  output.parent_local_1based_.assign(
      source.parent_local_1based.begin(), source.parent_local_1based.end());
  output.real_codes_small_.assign(
      source.real_codes_small.begin(), source.real_codes_small.end());
  output.edge_code0_.assign(source.edge_code0.begin(),
                            source.edge_code0.end());
  output.virtual_codes_small_.assign(
      source.virtual_codes_small.begin(), source.virtual_codes_small.end());
  output.coefficients_ = std::move(coefficient_result).value();
  output.source_real_norm2_.assign(
      source.source_real_norm2.begin(), source.source_real_norm2.end());
  output.has_quantized_forest_ = true;
  output.completion_.stage =
      PreparedCanonicalCluster<M>::Completion::Stage::kEncoded;
  if (retain_float_forest_) {
    output.float_coefficients_layer_major_.assign(
        source.coefficients_layer_major.begin(),
        source.coefficients_layer_major.end());
  }
  return output;
}

template <int M>
Status FinalClusterPostprocessor<M>::ComputeExactNorms(
    PreparedCanonicalCluster<M>& cluster,
    const RootSmallCrossBlock<M>& root_cross) const {
  const NormServiceTimer timer(quantized_norm_nanoseconds_);
  if (cluster.completion_.stage !=
      PreparedCanonicalCluster<M>::Completion::Stage::kEncoded) {
    return Status::FailedPrecondition(
        "exact norms require a completed coefficient encoding");
  }
  // Both backends use shared Gram cross tables and causal ancestor recurrence.
  // CUDA workers share immutable tables and own independent scalar workspaces.
  const std::uint32_t real_count = cluster.real_count();
  const std::uint32_t linked_count =
      real_count - cluster.real_root_count_;
  const CanonicalClusterCodeView<M> codes = {
      .m = cluster.m(),
      .cid = cluster.cid_,
      .virtual_count = cluster.virtual_count_,
      .real_root_count = cluster.real_root_count_,
      .linked_real_count = linked_count,
      .virtual_codes_small = cluster.virtual_codes_small_,
      .real_codes_small = cluster.real_codes_small_,
      .edge_code0 = cluster.edge_code0_,
  };
  const auto coefficient_view = cluster.coefficients_.view();
  const ReconstructionNormInput<M> norm_input = {
      .codes = codes,
      .parent_local_1based = cluster.parent_local_1based_,
      .coefficients_layer_major =
          cluster.coefficients_.quantized_coefficients(),
      .root_scales = coefficient_view.root_scales,
      .linkage_scales = coefficient_view.linkage_scales,
  };
#if STLQPP_HAS_CUDA_NORM
  if (use_cuda_reconstruction_norm_) {
    std::shared_ptr<CudaReconstructionNormComputer<M>> worker_pool;
    {
      std::lock_guard<std::mutex> lock(cuda_norm_mutex_);
      if (!cuda_norm_) {
        auto created = CudaReconstructionNormComputer<M>::Create(*grams_, cuda_norm_device_);
        if (!created.ok()) return created.status();
        cuda_norm_ = std::move(created).value();
      }
      worker_pool = cuda_norm_;
    }
    auto accelerated = worker_pool->Compute(norm_input, root_cross.norm_view(codes.cid));
    if (!accelerated.ok()) return accelerated.status();
    cluster.exact_real_norm2_ = std::move(accelerated).value();
    cluster.completion_.stage = PreparedCanonicalCluster<M>::Completion::Stage::kReady;
    return Status::Ok();
  }
#else
  if (use_cuda_reconstruction_norm_) {
    return Status::Unimplemented("GPU norm reconstruction is unavailable in this build");
  }
#endif
  auto selected = grams_->SelectCluster(codes, root_cross);
  if (!selected.ok()) return selected.status();
  const SelectedAtomReconstructionNormComputer<M> norm_computer;
  auto norm_result = norm_computer.Compute(selected.value(), norm_input);
  if (!norm_result.ok()) {
    return norm_result.status();
  }
  cluster.exact_real_norm2_ = std::move(norm_result).value();
  cluster.completion_.stage =
      PreparedCanonicalCluster<M>::Completion::Stage::kReady;
  return Status::Ok();
}

template <int M>
StatusOr<std::vector<float>> FinalClusterPostprocessor<M>::ComputeFloatNorms(
    const FloatReconstructionNormInput<M>& input,
    const RootSmallCrossBlock<M>& cross, bool assembled) const {
#if STLQPP_HAS_CUDA_NORM
  if (use_cuda_reconstruction_norm_) {
    std::shared_ptr<CudaReconstructionNormComputer<M>> pool;
    {
      std::lock_guard<std::mutex> lock(cuda_norm_mutex_);
      if (!cuda_norm_) {
        auto created = CudaReconstructionNormComputer<M>::Create(*grams_, cuda_norm_device_);
        if (!created.ok()) return created.status();
        cuda_norm_ = std::move(created).value();
      }
      pool = cuda_norm_;
    }
    return pool->ComputeFloat(input, cross.norm_view(input.codes.cid));
  }
#else
  if (use_cuda_reconstruction_norm_)
    return Status::Unimplemented("GPU norm reconstruction is unavailable in this build");
#endif
  auto selected = grams_->SelectCluster(input.codes, cross);
  if (!selected.ok()) return selected.status();
  const SelectedAtomReconstructionNormComputer<M> computer;
  return assembled ? computer.ComputeFloatValidated(selected.value(), input)
                   : computer.ComputeFloat(selected.value(), input);
}

template class PreparedCanonicalCluster<0>;
template class PreparedCanonicalCluster<5>;
template class PreparedCanonicalCluster<12>;
template class PreparedCanonicalCluster<6>;
template class PreparedCanonicalCluster<10>;
template class FinalClusterPostprocessor<0>;
template class FinalClusterPostprocessor<5>;
template class FinalClusterPostprocessor<12>;
template class FinalClusterPostprocessor<6>;
template class FinalClusterPostprocessor<10>;

}  // namespace stlqpp
