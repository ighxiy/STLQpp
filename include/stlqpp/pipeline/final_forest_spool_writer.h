#pragma once

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/index/build/canonical_float_cluster_assembler.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/index/build/prepared_cluster_spool.h"
#include "stlqpp/index/codec/selected_atom_gram.h"
#include "stlqpp/pipeline/final_forest_window_sink.h"
#include "stlqpp/runtime/cpu/parallel_for.h"

#include <cstdint>

namespace stlqpp {

// The only float-forest ingress into the prepared canonical spool. Windows
// are contiguous in CID order so one bounded root-cross block serves every
// CID in the window. Both full construction and replay delegate here.
template <int M>
class FinalForestSpoolWriter final : public FinalForestWindowSink<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  FinalForestSpoolWriter(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      const std::vector<std::uint32_t>& coefficient_base_bits,
      const std::vector<std::uint32_t>& coefficient_root_bits,
      const std::vector<std::uint32_t>& coefficient_linkage_bits,
      const std::vector<double>& coefficient_first_magnitude_quantiles,
      const std::vector<double>& coefficient_rest_magnitude_quantiles,
      const ParallelForExecutor& parallel,
      PreparedClusterSpoolWriter<M>& spool,
      bool retain_float_forest = true,
      ScalarCoefficientCodecOptions coefficient_codec_options = {},
      bool publish_quantized_forest = true,
      bool use_cuda_reconstruction_norm = false,
      int cuda_norm_device = 0,
      bool prepare_float_norm2 = true) noexcept
      : shape_(codebooks.shape()),
        grams_(&grams),
        parallel_(&parallel),
        spool_(&spool),
        postprocessor_(codebooks, grams, solver,
                       coefficient_base_bits,
                       coefficient_root_bits,
                       coefficient_linkage_bits,
                       coefficient_first_magnitude_quantiles,
                       coefficient_rest_magnitude_quantiles,
                       retain_float_forest, coefficient_codec_options,
                       publish_quantized_forest,
                       use_cuda_reconstruction_norm, cuda_norm_device,
                       prepare_float_norm2),
        next_cid_(spool.next_cid()) {}

  FinalForestSpoolWriter(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      const std::vector<double>& coefficient_first_magnitude_quantiles,
      const std::vector<double>& coefficient_rest_magnitude_quantiles,
      const ParallelForExecutor& parallel,
      PreparedClusterSpoolWriter<M>& spool,
      bool retain_float_forest = true) noexcept
      : FinalForestSpoolWriter(
            codebooks, grams, solver,
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            coefficient_first_magnitude_quantiles,
            coefficient_rest_magnitude_quantiles, parallel, spool,
            retain_float_forest) {}

  FinalForestSpoolWriter(
      const ProductCodebooks<M>& codebooks,
      const SelectedAtomGramTables<M>& grams,
      const SpdSolverF64& solver,
      double coefficient_magnitude_quantile,
      const ParallelForExecutor& parallel,
      PreparedClusterSpoolWriter<M>& spool,
      bool retain_float_forest = true) noexcept
      : FinalForestSpoolWriter(
            codebooks, grams, solver,
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            std::vector<std::uint32_t>(codebooks.shape().m, kDefaultCoefficientBits),
            {coefficient_magnitude_quantile},
            {coefficient_magnitude_quantile}, parallel, spool,
            retain_float_forest) {}

  [[nodiscard]] Status AppendWindow(
      ArrayView<const FinalForestSpoolClusterView<M>> clusters) override;
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> PrepareCluster(
      const FinalForestSpoolClusterView<M>& cluster) const override;
  [[nodiscard]] Status AppendPreparedWindow(
      ArrayView<const PreparedCanonicalCluster<M>> clusters) override;

  [[nodiscard]] std::uint32_t next_cid() const noexcept {
    return next_cid_;
  }

  [[nodiscard]] ReconstructionNormPreparationTiming norm_timing() const noexcept {
    return postprocessor_.norm_timing();
  }

 private:
  ModelShape shape_;
  const SelectedAtomGramTables<M>* grams_ = nullptr;
  const ParallelForExecutor* parallel_ = nullptr;
  PreparedClusterSpoolWriter<M>* spool_ = nullptr;
  FinalClusterPostprocessor<M> postprocessor_;
  std::uint32_t next_cid_ = 0U;
  bool failed_ = false;
};

extern template class FinalForestSpoolWriter<5>;
extern template class FinalForestSpoolWriter<10>;

}  // namespace stlqpp
