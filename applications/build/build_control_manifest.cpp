#include "build_control_manifest.h"

#include "stlqpp/application/config_schema.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>

namespace stlqpp::app {
namespace {

std::string_view ConfigFieldStatusName(
    ProductConfigField::Status status) noexcept {
  switch (status) {
    case ProductConfigField::Status::kRuntimeConsumer:
      return "runtime-consumer";
    case ProductConfigField::Status::kFixedProductContract:
      return "fixed-product-contract";
  }
  return "invalid";
}

std::string OrdinaryRsdAuthority(const BuildCommand& command) {
  const std::uint32_t training = command.training_topology.rsd_rank;
  const std::uint32_t base = command.base_topology.rsd_rank;
  if (training == base) {
    return "rsd-through-rank" + std::to_string(training) +
        "-icm1-ils0";
  }
  return "rsd-train-through-rank" + std::to_string(training) +
      "-base-through-rank" + std::to_string(base) + "-icm1-ils0";
}

}  // namespace

void WriteBuildControlManifest(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe,
    std::ostream& output) {
  WriteCanonicalBuildConfig(command, output);
  for (const ProductConfigField& field :
       ProductConfigFields(ProductConfigScope::kBuild)) {
    output << "contract.field." << field.public_key << '\t'
           << ConfigFieldStatusName(field.status) << '\n';
  }
  for (const ProductConfigField& field :
       ProductExtensionConfigFields(ProductConfigScope::kBuild)) {
    output << "extension.field." << field.public_key << '\t'
           << ConfigFieldStatusName(field.status) << '\n';
  }
  for (std::size_t index = 0U; index < command.config_files.size(); ++index) {
    output << "provenance.config-file."
           << std::setw(4) << std::setfill('0') << index + 1U << '\t'
           << command.config_files[index].generic_string() << '\n';
  }
  output << std::setfill(' ');
  for (const BuildConfigAssignment& assignment :
       command.effective_assignments) {
    output << "provenance.field." << assignment.key << '\t'
           << assignment.source << '\n';
  }
  const BuildResourcePlan resources = ResolveBuildResourcePlan(command);
  const EffectiveBuildExecutionConfig execution =
      ResolveBuildExecutionConfig(command);
  const bool initialization_direct_async =
      !command.linkage_scheduler_enabled &&
      execution.initialization_async_pinned_mebibytes != 0U;
  const bool ordinary_direct_async =
      !command.linkage_scheduler_enabled &&
      EffectiveLinkageAsyncPinnedMebibytes(command) != 0U;
  const bool stable_scheduler = command.linkage_scheduler_enabled &&
      command.linkage_scheduler_stable;
  const bool initialization_source_async =
      command.linkage_scheduler_enabled || initialization_direct_async;
  const bool ordinary_source_async = stable_scheduler || ordinary_direct_async;
  const CausalTopologyBatching& initialization_batching =
      recipe.initialization_topology.batching;
  const CausalTopologyBatching& ordinary_batching =
      recipe.base_topology.batching;
  const std::uint32_t training_scalar_bytes =
      command.effective_training_format == SourceFormat::kBvecsU8 ? 1U : 4U;
  const std::uint32_t base_scalar_bytes =
      command.effective_base_format == SourceFormat::kBvecsU8 ? 1U : 4U;
  const std::uint32_t initialization_preflush_pairs =
      initialization_batching.same_layer_minimum_batch_pairs == 0U
      ? 0U
      : initialization_batching.same_layer_preflush_target_pairs < 0
          ? std::max(
                initialization_batching.same_layer_minimum_batch_pairs,
                initialization_batching.many_nodes_preflush_target_pairs)
          : initialization_batching.same_layer_preflush_target_pairs == 0
              ? (initialization_source_async
                    ? 0U
                    : std::max(
                          initialization_batching
                              .same_layer_minimum_batch_pairs,
                          initialization_batching.inner_batch_rows))
              : static_cast<std::uint32_t>(
                    initialization_batching
                        .same_layer_preflush_target_pairs);
  output << std::setprecision(std::numeric_limits<float>::max_digits10)
      << "manifest.record-kind\tplanned\n"
      << "authority.schedule\tperiod-two-first-last-full\n"
      << "authority.schedule.closed-loop-controls\tfixed:enabled-inner0-period2\n"
      << "authority.basic.training\trsd2-ils8-icm4-signed\n"
      << "authority.basic.final-base\trsd2-ils32-icm4-absolute\n"
      << "authority.basic.virtual-training\tglobal-greedy-ils8-icm4-absolute\n"
      << "authority.basic.virtual-final-base\tglobal-greedy-ils32-icm4-absolute\n"
      << "authority.basic.replay\tfixed-root-rsd2-ils8-icm4-signed\n"
      << "authority.linkage.initialization\tlegacy-root-only-icm1-ils0\n"
      << "authority.linkage.ordinary\t"
      << OrdinaryRsdAuthority(command) << '\n'
      << "planned.output.protect-existing\t"
      << static_cast<int>(command.protect_existing_outputs) << '\n'
      << "planned.checkpoint.train.periodic-enabled\t"
      << static_cast<int>(command.train_checkpoint_enabled &&
                          command.train_checkpoint_every_r != 0U) << '\n'
      << "planned.checkpoint.train.every-rounds\t"
      << command.train_checkpoint_every_r << '\n'
      << "planned.checkpoint.train.base-path\t"
      << command.train_checkpoint_file.generic_string() << '\n'
      << "planned.checkpoint.base-basic\t"
      << static_cast<int>(command.base_basic_checkpoint_enabled) << '\n'
      << "planned.checkpoint.linkage\t"
      << static_cast<int>(command.linkage_checkpoint_enabled) << '\n'
      << "planned.storage.bucket-flush-bytes\t"
      << EffectiveBucketFlushBytes(command) << '\n'
      << "planned.storage.base-shard-size\teliminated:2000000\n"
      << "planned.search.parent-representation\tconfigurable:materialized-or-native-louds\n"
      << "planned.storage.cleanup\t"
      << (command.cleanup.enabled ? CleanupPresetName(command.cleanup.preset)
                                  : "disabled")
      << '\n'
      << "planned.storage.float-forest\t"
      << static_cast<int>(
             command.base_linkage_store_coefficients_f32 &&
             ResolveCleanupPolicy(command.cleanup).keep_linkage_list_f32)
      << '\n'
      << "planned.cuda.cublas-workspace-bytes\tfixed:0\n"
      << "planned.cuda.linkage-reconstruction\tfixed:device-rfull\n"
      << "planned.cuda.linkage-lane-admission\tfixed:blocking\n"
      << "planned.precompute.root-small-layout\tfixed:root-major-transpose\n"
      << "planned.precompute.root-small-limit-bytes\tfixed:unlimited\n"
      << "planned.root-init.kmeansll-backend\t"
      << (command.use_cuda ? "cuda-kmeans-parallel" : "cpu-kmeans-parallel")
      << '\n'
      << "planned.root-init.annealing\t"
      << (command.use_cuda ? "onepass-device-no-spill"
                           : "bounded-cpu-streaming")
      << '\n'
      << "planned.root-init.u8-upload\tfixed:source-scalar-aware\n"
      << "planned.root-init.spill-directory\teliminated:empty\n"
      << "planned.root-init.large-k-layout\t"
         "configured:sqrt-or-fixed-clamp-optional-pow2\n"
      << "planned.root-init.large-k-train\t"
      << (recipe.root_initialization.large_layer_hierarchy_enabled
              ? "hier2-enabled\n" : "flat\n")
      << "planned.root-init.large-k-encode\t"
      << (recipe.root_initialization.large_layer_hierarchy_enabled
              ? "hier2-enabled\n" : "flat\n")
      << "planned.topology.bad-cluster\tfixed:multicenter\n"
      << "planned.topology.real-density-anchor\tfixed:disabled\n"
      << "planned.virtual.ivf-prefix\tfixed:disabled\n"
      << "planned.effective.root-init.sample-vectors\t"
      << recipe.root_initialization.sample_vectors << '\n'
      << "planned.effective.root-init.kmeans-parallel-rounds\t"
      << recipe.root_initialization.kmeans_parallel_rounds << '\n'
      << "planned.effective.root-init.kmeans-parallel-oversample\t"
      << recipe.root_initialization.kmeans_parallel_oversample << '\n'
      << "planned.effective.root-init.candidate-cap\t"
      << recipe.root_initialization.candidate_cap << '\n'
      << "planned.effective.root-init.large-layer-top-coarse\t"
      << recipe.root_initialization.large_layer_top_coarse << '\n'
      << "planned.consumer.basic.cuda-instances\t"
      << resources.basic_cuda_instances << '\n'
      << "planned.consumer.basic.cpu-instances\t"
      << resources.basic_cpu_instances << '\n'
      << "planned.effective.large.train-block\t"
      << EffectiveTrainingBlockRows(command) << '\n'
      << "planned.effective.large.base-block\t"
      << EffectiveBaseBlockRows(command) << '\n'
      << "planned.effective.large.cluster-bucket-size\t"
      << EffectiveCidsPerBucket(command) << '\n'
      << "planned.consumer.basic.training-inflight-blocks\t"
      << ResolveBasicInflightBlockCount(
             command, training_scalar_bytes,
             EffectiveTrainingBlockRows(command))
      << '\n'
      << "planned.consumer.basic.training-prefetch-blocks\t"
      << ResolveBasicPrefetchBlockCount(
             command, training_scalar_bytes,
             EffectiveTrainingBlockRows(command))
      << '\n'
      << "planned.consumer.basic.training-pending-writes\t"
      << ResolveBasicPendingWriteCount(
             command, training_scalar_bytes,
             EffectiveTrainingBlockRows(command))
      << '\n'
      << "planned.consumer.basic.base-inflight-blocks\t"
      << ResolveBasicInflightBlockCount(
             command, base_scalar_bytes, EffectiveBaseBlockRows(command))
      << '\n'
      << "planned.consumer.basic.base-prefetch-blocks\t"
      << ResolveBasicPrefetchBlockCount(
             command, base_scalar_bytes, EffectiveBaseBlockRows(command))
      << '\n'
      << "planned.consumer.basic.base-pending-writes\t"
      << ResolveBasicPendingWriteCount(
             command, base_scalar_bytes, EffectiveBaseBlockRows(command))
      << '\n'
      << "planned.consumer.linkage.prefetch-clusters\t"
      << ResolveLinkagePrefetchClusterCount(command) << '\n'
      << "planned.consumer.init.scheduler-backends\t"
      << resources.initialization_scheduler_backends << '\n'
      << "planned.consumer.init.direct-backends\t"
      << resources.initialization_direct_backends << '\n'
      << "planned.consumer.init.cpu-direct-backends\t"
      << resources.initialization_cpu_direct_backends << '\n'
      << "planned.consumer.ordinary.scheduler-backends\t"
      << resources.ordinary_scheduler_backends << '\n'
      << "planned.consumer.ordinary.direct-backends\t"
      << resources.ordinary_direct_backends << '\n'
      << "planned.consumer.ordinary.cpu-direct-backends\t"
      << resources.ordinary_cpu_direct_backends << '\n'
      << "planned.consumer.replay.cuda-instances\t"
      << resources.replay_cuda_instances << '\n'
      << "planned.consumer.replay.cpu-instances\t"
      << resources.replay_cpu_instances << '\n'
      << "planned.consumer.cpu.openmp-workers\t"
      << command.effective_cpu_threads << '\n'
      << "planned.consumer.cpu.basic-workers\t"
      << command.effective_cpu_threads
      << '\n'
      << "planned.consumer.cpu.base-list-workers\t"
      << command.effective_base_list_threads
      << '\n'
      << "planned.consumer.cpu.training-edge-shards\t"
      << command.effective_training_edge_moment_shards
      << '\n'
      << "planned.consumer.cpu.initialization-edge-shards\t"
      << command.effective_initialization_edge_moment_shards
      << '\n'
      << "planned.consumer.scheduler.transport-batching-enabled\t"
      << static_cast<int>(command.linkage_scheduler_enabled &&
                          command.linkage_scheduler_transport_batching_enabled)
      << '\n'
      << "planned.consumer.scheduler.stable\t"
      << static_cast<int>(command.linkage_scheduler_stable)
      << '\n'
      << "planned.consumer.scheduler.numerical-merge-enabled\t"
      << static_cast<int>(
          command.linkage_scheduler_enabled &&
          command.linkage_scheduler_transport_batching_enabled &&
          !stable_scheduler)
      << '\n'
      << "planned.consumer.init.preflush-pair-target\t"
      << initialization_preflush_pairs << '\n'
      << "planned.consumer.init.direct-async-evaluation\t"
      << static_cast<int>(initialization_direct_async) << '\n'
      << "planned.consumer.init.source-preflush-enabled\t"
      << static_cast<int>(initialization_source_async) << '\n'
      << "planned.consumer.ordinary.preflush-enabled\t"
      << static_cast<int>(
          ordinary_source_async &&
          ordinary_batching.same_layer_minimum_batch_pairs != 0U &&
          (ordinary_batching.same_layer_preflush_target_pairs != 0 ||
           ordinary_batching.same_layer_preflush_target_nodes != 0U))
      << '\n'
      << "planned.consumer.init.same-layer-window-enabled\t"
      << static_cast<int>(
          initialization_batching.same_layer_minimum_batch_pairs != 0U)
      << '\n'
      << "planned.consumer.init.same-layer-window-max-pairs\t"
      << initialization_batching.same_layer_window_maximum_pairs << '\n'
      << "planned.consumer.init.same-layer-max-pending-nodes\t"
      << initialization_batching.same_layer_maximum_pending_nodes << '\n'
      << "planned.consumer.ordinary.same-layer-window-enabled\t"
      << static_cast<int>(
          ordinary_batching.same_layer_minimum_batch_pairs != 0U) << '\n'
      << "planned.consumer.ordinary.same-layer-window-max-pairs\t"
      << ordinary_batching.same_layer_window_maximum_pairs << '\n'
      << "planned.consumer.ordinary.same-layer-max-pending-nodes\t"
      << ordinary_batching.same_layer_maximum_pending_nodes << '\n'
      << "planned.consumer.init.single-task-pair-limit\t"
      << MaximumProducedEvaluationPairs(recipe.initialization_topology) << '\n'
      << "planned.consumer.init.initial-workspace-pairs\t"
      << execution.initialization_maximum_batch_pairs << '\n'
      << "planned.consumer.ordinary.single-task-pair-limit\t"
      << MaximumProducedEvaluationPairs(recipe.base_topology) << '\n'
      << "planned.consumer.ordinary.initial-workspace-pairs\t"
      << command.linkage_max_batch_pairs << '\n';
}

}  // namespace stlqpp::app
