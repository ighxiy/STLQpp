#include "stlqpp/pipeline/final_index_build_session.h"

#include <array>
#include <memory>
#include <optional>
#include <utility>

#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/platform/staging_directory.h"

namespace stlqpp {

template <int M>
struct FinalIndexBuildSession<M>::State final {
  State(FinalIndexBuildRequest input,
        const TrainingModel<M>& trained_model,
        const SelectedAtomGramTables<M>& grams,
        const SpdSolverF64& coefficient_solver,
        const ParallelForExecutor& parallel,
        PreparedClusterSpoolWriter<M> prepared) noexcept
      : request(std::move(input)),
        model(&trained_model),
        spool(std::move(prepared)),
        forest_writer(trained_model.codebooks(), grams, coefficient_solver,
                      request.coefficient_bits_per_layer,
                      request.coefficient_bits_per_layer_root.empty()
                          ? request.coefficient_bits_per_layer
                          : request.coefficient_bits_per_layer_root,
                      request.coefficient_bits_per_layer_linkage.empty()
                          ? request.coefficient_bits_per_layer
                          : request.coefficient_bits_per_layer_linkage,
                      request.coefficient_first_magnitude_quantiles,
                      request.coefficient_rest_magnitude_quantiles,
                      parallel, spool, request.retain_float_forest,
                      request.coefficient_codec_options,
                      request.publish_quantized_forest,
                      request.use_cuda_norm8_fit,
                      request.cuda_norm_device,
                      request.retain_float_norm2) {}

  FinalIndexBuildRequest request;
  const TrainingModel<M>* model = nullptr;
  PreparedClusterSpoolWriter<M> spool;
  FinalForestSpoolWriter<M> forest_writer;
  bool spool_committed = false;
  bool published = false;
};

template <int M>
FinalIndexBuildSession<M>::~FinalIndexBuildSession() = default;

template <int M>
StatusOr<std::unique_ptr<FinalIndexBuildSession<M>>>
FinalIndexBuildSession<M>::Begin(
    const FinalIndexBuildRequest& request,
    const TrainingModel<M>& model,
    const SelectedAtomGramTables<M>& grams,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel) {
  FinalIndexBuildRequest effective_request = request;
  auto base_bits = NormalizeCoefficientBits(
      std::move(effective_request.coefficient_bits_per_layer), model.codebooks().shape().m,
      "large.linkage_coeff_codec.bits_per_layer");
  if (!base_bits.ok()) return base_bits.status();
  effective_request.coefficient_bits_per_layer =
      std::move(base_bits).value();
  if (!effective_request.coefficient_bits_per_layer_root.empty()) {
    auto root_bits = NormalizeCoefficientBits(
        std::move(effective_request.coefficient_bits_per_layer_root), model.codebooks().shape().m,
        "large.linkage_coeff_codec.bits_per_layer_root");
    if (!root_bits.ok()) return root_bits.status();
    effective_request.coefficient_bits_per_layer_root =
        std::move(root_bits).value();
  }
  if (!effective_request.coefficient_bits_per_layer_linkage.empty()) {
    auto linkage_bits = NormalizeCoefficientBits(
        std::move(effective_request.coefficient_bits_per_layer_linkage), model.codebooks().shape().m,
        "large.linkage_coeff_codec.bits_per_layer_linkage");
    if (!linkage_bits.ok()) return linkage_bits.status();
    effective_request.coefficient_bits_per_layer_linkage =
        std::move(linkage_bits).value();
  }
  if (request.prepared_spool.empty() || request.destination.empty() ||
      request.prepared_spool == request.destination ||
      (!request.publish_quantized_forest &&
       (!request.retain_float_forest || request.publish_scaled_score8)) ||
      (request.publish_float_scaled &&
       !request.retain_float_forest)) {
    return Status::InvalidArgument(
        "final index build paths are empty or alias each other");
  }
  if (const Status status = ValidateDisjointFilesystemPaths(
          request.prepared_spool, "prepared spool",
          request.destination, "final index destination");
      !status.ok()) {
    return status;
  }
  // Direct payloads become the final shards by rename. On another filesystem,
  // use a deterministic sibling of the output; other scratch data stays in
  // tmp_dir.
  std::error_code prepared_error;
  const bool requested_checkpoint =
      request.checkpoint_enabled &&
      std::filesystem::exists(request.prepared_spool, prepared_error);
  if (prepared_error)
    return Status::Internal("cannot inspect prepared checkpoint");
  if (!requested_checkpoint) {
    auto same_filesystem =
        StagingDirectory::SharesFilesystem(request.prepared_spool.parent_path(),
                                           request.destination.parent_path());
    if (!same_filesystem.ok()) return same_filesystem.status();
    if (!same_filesystem.value()) {
      effective_request.prepared_spool = request.destination;
      effective_request.prepared_spool += ".prepared-spool";
    }
  }
  if (request.protect_existing_outputs) {
    std::error_code error;
    const bool exists = std::filesystem::exists(request.destination, error);
    if (error) {
      return Status::Internal(
          "failed to inspect the final-index destination: " +
          request.destination.string());
    }
    if (exists) {
      return Status::FailedPrecondition(
          "final-index destination exists and output protection is enabled: " +
          request.destination.string());
    }
  }
  const PreparedClusterSpoolRequest spool_request = {
      .destination = effective_request.prepared_spool,
      .shape = model.codebooks().shape(),
      .hashes = request.hashes,
      .retain_float_forest = request.retain_float_forest,
      .retain_source_norm2 =
          request.publish_scaled_score8 || request.publish_float_scaled,
      .publish_quantized_forest = request.publish_quantized_forest,
      .direct_payloads = true,
      .retain_float_norm2 = request.retain_float_norm2,
  };
  std::error_code spool_error;
  const bool resume_spool =
      request.checkpoint_enabled &&
      std::filesystem::exists(effective_request.prepared_spool, spool_error);
  if (spool_error) {
    return Status::Internal(
        "failed to inspect prepared spool checkpoint destination");
  }
  std::optional<PreparedClusterSpoolWriter<M>> prepared;
  if (resume_spool) {
    auto opened =
        PreparedClusterSpoolWriter<M>::OpenCheckpointed(spool_request);
    if (!opened.ok()) {
      return opened.status();
    }
    prepared.emplace(std::move(opened).value());
  }
  // Keep DEV replacement policy at this application-stage boundary: generic
  // canonical writers remain create-only and atomically publish one new tree.
  std::array<std::filesystem::path, 2U> replace_paths = {
      request.destination, effective_request.prepared_spool};
  const std::size_t replace_count = request.checkpoint_enabled ? 1U : 2U;
  for (std::size_t index = 0U; index < replace_count; ++index) {
    const std::filesystem::path& path = replace_paths[index];
    std::error_code error;
    std::filesystem::remove_all(path, error);
    if (error) {
      return Status::Internal(
          "failed to clear an existing final-index stage destination: " +
          path.string());
    }
  }
  if (!prepared.has_value()) {
    auto created = request.checkpoint_enabled
        ? PreparedClusterSpoolWriter<M>::CreateCheckpointed(spool_request)
        : PreparedClusterSpoolWriter<M>::Create(spool_request);
    if (!created.ok()) {
      return created.status();
    }
    prepared.emplace(std::move(created).value());
  }
  auto state = std::make_unique<State>(
      std::move(effective_request), model, grams, coefficient_solver, parallel,
      std::move(*prepared));
  return std::unique_ptr<FinalIndexBuildSession<M>>(
      new FinalIndexBuildSession<M>(std::move(state)));
}

template <int M>
bool FinalIndexBuildSession<M>::RequiresOrderedCommit() const noexcept {
  return !state_->spool.direct_payloads();
}
template <int M>
const std::vector<std::uint8_t>& FinalIndexBuildSession<M>::completed_cids()
    const noexcept {
  return state_->spool.completed_cids();
}

template <int M>
bool FinalIndexBuildSession<M>::RequiresSourceNorm2() const noexcept {
  return state_ != nullptr &&
      (state_->request.publish_scaled_score8 ||
       state_->request.publish_float_scaled);
}

template <int M>
Status FinalIndexBuildSession<M>::AppendWindow(
    ArrayView<const FinalForestSpoolClusterView<M>> clusters) {
  if (state_->spool_committed) {
    return Status::FailedPrecondition(
        "final forest cannot be appended after spool commit");
  }
  return state_->forest_writer.AppendWindow(clusters);
}

template <int M>
StatusOr<PreparedCanonicalCluster<M>>
FinalIndexBuildSession<M>::PrepareCluster(
    const FinalForestSpoolClusterView<M>& cluster) const {
  if ((state_->request.publish_scaled_score8 ||
       state_->request.publish_float_scaled) &&
      cluster.source_real_norm2.size() != cluster.real_sample_ids.size()) {
    return Status::FailedPrecondition(
        "scaled-score publication requires original-vector norms from the final topology source");
  }
  return state_->forest_writer.PrepareCluster(cluster);
}

template <int M>
Status FinalIndexBuildSession<M>::AppendPreparedWindow(
    ArrayView<const PreparedCanonicalCluster<M>> clusters) {
  if (state_->spool_committed) {
    return Status::FailedPrecondition(
        "prepared final forest cannot be appended after spool commit");
  }
  return state_->forest_writer.AppendPreparedWindow(clusters);
}

template <int M>
Status FinalIndexBuildSession<M>::Publish(
    const ParallelForExecutor& parallel,
    FinalIndexPublicationStatistics* statistics) {
  if (state_->published) {
    return Status::FailedPrecondition(
        "final canonical index was already published");
  }
  if (!state_->spool_committed) {
    if (const Status status = state_->spool.Commit(); !status.ok()) {
      return status;
    }
    state_->spool_committed = true;
  }
  const CanonicalModelSource<M> model_source = {
      .rotation = state_->model->rotation(),
      .codebooks = state_->model->codebook_view(),
  };
  const Status status = PublishPreparedClusterSpool<M>(
      state_->request.prepared_spool,
      {
          .destination = state_->request.destination,
          .model = model_source,
          .hashes = state_->request.hashes,
          .coefficient_bits_per_layer =
              state_->request.coefficient_bits_per_layer,
          .coefficient_bits_per_layer_root =
              state_->request.coefficient_bits_per_layer_root,
          .coefficient_bits_per_layer_linkage =
              state_->request.coefficient_bits_per_layer_linkage,
          .coefficient_first_magnitude_quantiles =
              state_->request.coefficient_first_magnitude_quantiles,
          .coefficient_rest_magnitude_quantiles =
              state_->request.coefficient_rest_magnitude_quantiles,
          .coefficient_codec_options =
              state_->request.coefficient_codec_options,
          .norm8_mode = state_->request.norm8_mode,
          .norm8_lloyd_iterations =
              state_->request.norm8_lloyd_iterations,
          .use_cuda_norm8_fit = state_->request.use_cuda_norm8_fit,
          .cuda_norm_device = state_->request.cuda_norm_device,
          .retain_float_forest = state_->request.retain_float_forest,
          .retain_float_norm2 = state_->request.retain_float_norm2,
          .publish_quantized_forest =
              state_->request.publish_quantized_forest,
          .publish_scaled_score8 = state_->request.publish_scaled_score8,
          .publish_float_scaled = state_->request.publish_float_scaled,
          .model_reference = state_->request.model_reference,
      },
      parallel, statistics);
  if (status.ok()) {
    if (statistics != nullptr) {
      statistics->reconstruction_norm_service =
          state_->forest_writer.norm_timing();
    }
    state_->published = true;
  }
  return status;
}

template <int M>
std::uint32_t FinalIndexBuildSession<M>::next_cid() const noexcept {
  return state_ == nullptr ? 0U : state_->spool.next_cid();
}

template <int M>
std::uint64_t FinalIndexBuildSession<M>::committed_real_rows() const noexcept {
  return state_ == nullptr ? 0U : state_->spool.committed_counts().real;
}

template class FinalIndexBuildSession<0>;
template class FinalIndexBuildSession<5>;
template class FinalIndexBuildSession<12>;
template class FinalIndexBuildSession<6>;
template class FinalIndexBuildSession<10>;

}  // namespace stlqpp
