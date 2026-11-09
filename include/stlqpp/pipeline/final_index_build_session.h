#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/final_index_publisher.h"
#include "stlqpp/pipeline/final_forest_spool_writer.h"
#include "stlqpp/training/update/training_model.h"

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace stlqpp {

struct FinalIndexBuildRequest final {
  std::filesystem::path prepared_spool;
  std::filesystem::path destination;
  bool protect_existing_outputs = false;
  bool checkpoint_enabled = false;
  IndexArtifactHashes hashes;
  std::vector<std::uint32_t> coefficient_bits_per_layer = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_root = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage = {};
  std::vector<double> coefficient_first_magnitude_quantiles;
  std::vector<double> coefficient_rest_magnitude_quantiles;
  ScalarCoefficientCodecOptions coefficient_codec_options;
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  bool use_cuda_norm8_fit = false;
  int cuda_norm_device = 0;
  bool retain_float_forest = true;
  bool retain_float_norm2 = true;
  bool publish_quantized_forest = true;
  bool publish_scaled_score8 = false;
  bool publish_float_scaled = false;
  std::filesystem::path model_reference = {};
};

// Top-level final-index publication transaction. The immutable trained model,
// Gram tables outlive the session. Full-build or replay
// products enter through the same ordered final-forest boundary; Publish first
// commits the restartable spool and then performs global norm8/canonical
// publication. A publication failure leaves the committed spool reusable.
template <int M>
class FinalIndexBuildSession final : public FinalForestWindowSink<M> {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<std::unique_ptr<FinalIndexBuildSession<M>>>
  Begin(const FinalIndexBuildRequest& request,
        const TrainingModel<M>& model,
        const SelectedAtomGramTables<M>& grams,
        const SpdSolverF64& coefficient_solver,
        const ParallelForExecutor& parallel);

  ~FinalIndexBuildSession();
  FinalIndexBuildSession(const FinalIndexBuildSession&) = delete;
  FinalIndexBuildSession& operator=(const FinalIndexBuildSession&) = delete;

  [[nodiscard]] bool RequiresOrderedCommit() const noexcept override;
  [[nodiscard]] const std::vector<std::uint8_t>& completed_cids()
      const noexcept;
  [[nodiscard]] bool RequiresSourceNorm2() const noexcept override;
  [[nodiscard]] Status AppendWindow(
      ArrayView<const FinalForestSpoolClusterView<M>> clusters) override;
  [[nodiscard]] StatusOr<PreparedCanonicalCluster<M>> PrepareCluster(
      const FinalForestSpoolClusterView<M>& cluster) const override;
  [[nodiscard]] Status AppendPreparedWindow(
      ArrayView<const PreparedCanonicalCluster<M>> clusters) override;
  [[nodiscard]] Status Publish(
      const ParallelForExecutor& parallel,
      FinalIndexPublicationStatistics* statistics = nullptr);
  [[nodiscard]] std::uint32_t next_cid() const noexcept;
  [[nodiscard]] std::uint64_t committed_real_rows() const noexcept;

 private:
  struct State;
  explicit FinalIndexBuildSession(std::unique_ptr<State> state) noexcept
      : state_(std::move(state)) {}

  std::unique_ptr<State> state_;
};

extern template class FinalIndexBuildSession<5>;
extern template class FinalIndexBuildSession<10>;

}  // namespace stlqpp
