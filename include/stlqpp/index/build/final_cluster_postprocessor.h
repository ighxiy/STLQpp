#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/codec/scalar_int6_encoder.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/index/norm/reconstruction_norms.h"
#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/model/product_codebooks.h"
#include "stlqpp/storage/canonical_cluster_codec.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace stlqpp {

// Concurrent per-cluster service time, including backend waits. Not GPU time
// and not subtractable from pipeline wall time.
struct ReconstructionNormPreparationTiming final {
  double float_seconds = 0.0;
  double quantized_seconds = 0.0;
};

template <int M>
class OwnedCanonicalFloatCluster;
template <int M>
class FinalForestSpoolWriter;
template <int M>
class PreparedClusterSpoolWriter;
template <int M>
class CudaReconstructionNormComputer;

// Final float cluster after canonical BFS ordering. This is the only accepted
// handoff from base/linkage construction into persistent postprocessing.
struct CanonicalFloatClusterView final {
  std::uint32_t m = 0U;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ArrayView<const float> coefficients_layer_major;
  // Original unrotated vector norm. Optional for classic-only publication;
  // required for the geometry-only scaled score8 fit.
  ArrayView<const float> source_real_norm2;
};

template <int M>
class PreparedCanonicalCluster final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t m() const noexcept { return M == 0 ? m_ : M; }
  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_ids_.size());
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] ArrayView<const float> exact_real_norm2() const noexcept {
    return exact_real_norm2_;
  }
  [[nodiscard]] ArrayView<const float> float_coefficients_layer_major()
      const noexcept {
    return float_coefficients_layer_major_;
  }
  [[nodiscard]] ArrayView<const float> float_real_norm2() const noexcept {
    return float_real_norm2_;
  }
  [[nodiscard]] ArrayView<const float> source_real_norm2() const noexcept {
    return source_real_norm2_;
  }
  [[nodiscard]] bool has_quantized_forest() const noexcept {
    return has_quantized_forest_;
  }
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
    return sizeof(*this) +
        static_cast<std::uint64_t>(real_ids_.capacity()) * sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(parent_local_1based_.capacity()) * sizeof(std::uint32_t) +
        real_codes_small_.capacity() + edge_code0_.capacity() +
        virtual_codes_small_.capacity() + coefficients_.resident_bytes() +
        static_cast<std::uint64_t>(exact_real_norm2_.capacity() +
                                   float_coefficients_layer_major_.capacity() +
                                   float_real_norm2_.capacity() +
                                   source_real_norm2_.capacity()) * sizeof(float);
  }
  [[nodiscard]] const ScalarInt6EncodingStatistics& codec_statistics() const
      noexcept {
    return coefficients_.statistics();
  }

  [[nodiscard]] CanonicalClusterPayloadSource<M> payload_source() const
      noexcept;
  [[nodiscard]] CanonicalClusterTopologySource<M> topology_source() const
      noexcept;

  [[nodiscard]] StatusOr<CanonicalClusterSource<M>> MakeWriteSource(
      ArrayView<const std::uint8_t> real_norm8,
      Norm8CodebookView norm8) const;

 private:
  template <int N>
  friend class FinalClusterPostprocessor;
  template <int N>
  friend class PreparedClusterSpoolWriter;

  // Moving a completed product transfers its trusted provenance. A default,
  // Stage-8, or moved-from object cannot enter the prepared spool path.
  struct Completion final {
    enum class Stage : std::uint8_t { kEmpty, kEncoded, kReady };
    Stage stage = Stage::kEmpty;
    Completion() = default;
    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;
    Completion(Completion&& other) noexcept
        : stage(std::exchange(other.stage, Stage::kEmpty)) {}
    Completion& operator=(Completion&& other) noexcept {
      if (this != &other) {
        stage = std::exchange(other.stage, Stage::kEmpty);
      }
      return *this;
    }
  };

  std::uint32_t m_ = M;
  std::uint32_t cid_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::vector<std::uint32_t> real_ids_;
  std::vector<std::uint32_t> parent_local_1based_;
  std::vector<std::uint8_t> real_codes_small_;
  std::vector<std::uint8_t> edge_code0_;
  std::vector<std::uint8_t> virtual_codes_small_;
  EncodedScalarInt6Cluster<M> coefficients_;
  std::vector<float> exact_real_norm2_;
  std::vector<float> float_coefficients_layer_major_;
  std::vector<float> float_real_norm2_;
  std::vector<float> source_real_norm2_;
  bool has_quantized_forest_ = true;
  Completion completion_;
};

template <int M>
class FinalClusterPostprocessor final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  FinalClusterPostprocessor(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      std::vector<std::uint32_t> coefficient_base_bits,
      std::vector<std::uint32_t> coefficient_root_bits,
      std::vector<std::uint32_t> coefficient_linkage_bits,
      std::vector<double> coefficient_first_magnitude_quantiles,
      std::vector<double> coefficient_rest_magnitude_quantiles,
      bool retain_float_forest = true,
      ScalarCoefficientCodecOptions coefficient_codec_options = {},
      bool publish_quantized_forest = true,
      bool use_cuda_reconstruction_norm = false,
      int cuda_norm_device = 0,
      bool prepare_float_norm2 = true) noexcept
      : codebooks_(&codebooks),
        grams_(&grams),
        solver_(&solver),
        coefficient_base_bits_(std::move(coefficient_base_bits)),
        coefficient_root_bits_(std::move(coefficient_root_bits)),
        coefficient_linkage_bits_(std::move(coefficient_linkage_bits)),
        coefficient_first_magnitude_quantiles_(
            std::move(coefficient_first_magnitude_quantiles)),
        coefficient_rest_magnitude_quantiles_(
            std::move(coefficient_rest_magnitude_quantiles)),
        coefficient_codec_options_(coefficient_codec_options),
        retain_float_forest_(retain_float_forest),
        publish_quantized_forest_(publish_quantized_forest),
        use_cuda_reconstruction_norm_(use_cuda_reconstruction_norm),
        cuda_norm_device_(cuda_norm_device),
        prepare_float_norm2_(prepare_float_norm2) {}

  FinalClusterPostprocessor(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      std::vector<double> coefficient_first_magnitude_quantiles,
      std::vector<double> coefficient_rest_magnitude_quantiles,
      bool retain_float_forest = true) noexcept
      : FinalClusterPostprocessor(
            codebooks, grams, solver,
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::move(coefficient_first_magnitude_quantiles),
            std::move(coefficient_rest_magnitude_quantiles),
            retain_float_forest) {}

  FinalClusterPostprocessor(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      double coefficient_magnitude_quantile,
      bool retain_float_forest = true) noexcept
      : FinalClusterPostprocessor(
            codebooks, grams, solver, {coefficient_magnitude_quantile},
            {coefficient_magnitude_quantile}, retain_float_forest) {}

  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> Prepare(
      const CanonicalFloatClusterView& source,
      const RootSmallCrossBlock<M>& root_cross) const;

  // Retained native-float publication does not need a Gram or a norm rebuild.
  // The optional norm view may be empty when eval prepares norms on demand.
  [[nodiscard]] static StatusOr<PreparedCanonicalCluster<M>>
  PrepareRetainedFloatOnly(
      const CanonicalFloatClusterView& source,
      ArrayView<const float> float_real_norm2,
      std::uint32_t nlist);

  // Codec-only rebuild boundary. Float coefficients are the immutable source;
  // optional stored float norms are consumed only when explicitly requested.
  // Compressed coefficients and their matching reconstruction norms change.
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>>
  PrepareFromRetainedFloatForest(
      const CanonicalFloatClusterView& source,
      ArrayView<const float> source_float_real_norm2,
      const RootSmallCrossBlock<M>& root_cross) const;

  // Stage-8 boundary. Produces the final signed-int6/Huffman coefficient
  // state but intentionally leaves exact_real_norm2 empty.
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> EncodeCoefficients(
      const CanonicalFloatClusterView& source,
      const RootSmallCrossBlock<M>& root_cross) const;

  // Stage-9 boundary. Consumes the already final coefficient state and adds
  // exact reconstruction norms without changing the codec payload.
  [[nodiscard]] Status ComputeExactNorms(
      PreparedCanonicalCluster<M>& cluster,
      const RootSmallCrossBlock<M>& root_cross) const;

  [[nodiscard]] ReconstructionNormPreparationTiming norm_timing() const noexcept {
    return {static_cast<double>(float_norm_nanoseconds_.load(
                std::memory_order_relaxed)) * 1e-9,
            static_cast<double>(quantized_norm_nanoseconds_.load(
                std::memory_order_relaxed)) * 1e-9};
  }

  [[nodiscard]] double coefficient_service_seconds() const noexcept {
    return static_cast<double>(coefficient_nanoseconds_.load(std::memory_order_relaxed)) * 1.0e-9;
  }
  [[nodiscard]] std::uint32_t coefficient_peak_workers() const noexcept {
    return coefficient_peak_.load(std::memory_order_relaxed);
  }

 private:
  friend class FinalForestSpoolWriter<M>;

  // Only the spool writer can pass the immutable result of Assemble().
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> PrepareAssembled(
      const OwnedCanonicalFloatCluster<M>& source,
      const RootSmallCrossBlock<M>& root_cross) const;
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> PrepareImpl(
      const CanonicalFloatClusterView& source,
      const RootSmallCrossBlock<M>& root_cross,
      bool assembled) const;
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>>
  EncodeCoefficientsImpl(
      const CanonicalFloatClusterView& source,
      const RootSmallCrossBlock<M>& root_cross,
      bool assembled) const;

  [[nodiscard]] StatusOr<std::vector<float>> ComputeFloatNorms(
      const FloatReconstructionNormInput<M>& input,
      const RootSmallCrossBlock<M>& root_cross, bool assembled) const;

  const ProductCodebooks<M>* codebooks_ = nullptr;
  const SelectedAtomGramTables<M>* grams_ = nullptr;
  const SpdSolverF64* solver_ = nullptr;
  std::vector<std::uint32_t> coefficient_base_bits_;
  std::vector<std::uint32_t> coefficient_root_bits_;
  std::vector<std::uint32_t> coefficient_linkage_bits_;
  std::vector<double> coefficient_first_magnitude_quantiles_;
  std::vector<double> coefficient_rest_magnitude_quantiles_;
  ScalarCoefficientCodecOptions coefficient_codec_options_;
  bool retain_float_forest_ = true;
  bool publish_quantized_forest_ = true;
  bool use_cuda_reconstruction_norm_ = false;
  int cuda_norm_device_ = 0;
  bool prepare_float_norm2_ = true;
  mutable std::atomic<std::int64_t> float_norm_nanoseconds_{0};
  mutable std::atomic<std::int64_t> quantized_norm_nanoseconds_{0};
  mutable std::atomic<std::int64_t> coefficient_nanoseconds_{0};
  mutable std::atomic<std::uint32_t> coefficient_active_{0}, coefficient_peak_{0};
  mutable std::mutex cuda_norm_mutex_;
  mutable std::shared_ptr<CudaReconstructionNormComputer<M>> cuda_norm_;
};

extern template class PreparedCanonicalCluster<5>;
extern template class PreparedCanonicalCluster<10>;
extern template class FinalClusterPostprocessor<5>;
extern template class FinalClusterPostprocessor<10>;

}  // namespace stlqpp
