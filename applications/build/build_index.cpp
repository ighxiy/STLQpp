#include "build_command.h"
#include "build_completion_log.h"
#include "effective_build_recipe.h"
#include "run_index_build.h"
#include "eval_command.h"
#include "evaluate_base_basic.h"
#include "run_exact_dense_eval.h"

#include "stlqpp/application/config_schema.h"
#include "stlqpp/application/config_text.h"
#include "stlqpp/application/index_builder.h"
#include "stlqpp/index/emit/base_list_emitter.h"
#include "stlqpp/model/model_dispatch.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/trained_model_file.h"
#include "stlqpp/runtime/cpu/openmp_affinity.h"
#include "stlqpp/runtime/cpu/openmp_parallel_for.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/training/checkpoint/checkpoint_path.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {


void WriteCompleteConfig(
    const stlqpp::app::BuildCommand& build,
    const stlqpp::app::EvalCommand& eval,
    std::ostream& output) {
  std::ostringstream build_text;
  stlqpp::app::WriteRoundTripBuildConfig(build, build_text);
  std::ostringstream eval_text;
  stlqpp::app::WriteRoundTripEvalConfig(eval, eval_text, false);
  std::vector<std::string> eval_lines;
  std::vector<std::string> eval_keys;
  std::istringstream lines(eval_text.str());
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) continue;
    eval_keys.push_back(stlqpp::app::TrimConfigText(
        std::string_view(line).substr(0U, equals)));
    eval_lines.push_back(line);
  }
  std::istringstream build_lines(build_text.str());
  while (std::getline(build_lines, line)) {
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) continue;
    const std::string key = stlqpp::app::TrimConfigText(
        std::string_view(line).substr(0U, equals));
    if (std::find(eval_keys.begin(), eval_keys.end(), key) ==
        eval_keys.end()) {
      output << line << '\n';
    }
  }
  for (const std::string& eval_line : eval_lines) {
    output << eval_line << '\n';
  }
}

bool RetainsBaseListStore(
    const stlqpp::app::BuildCommand& command) noexcept {
  if (!command.cleanup.enabled) return true;
  const stlqpp::app::ResolvedCleanupPolicy policy =
      stlqpp::app::ResolveCleanupPolicy(command.cleanup);
  return policy.keep_base_basic || policy.keep_base_list ||
      policy.keep_base_basic_cluster_id ||
      policy.keep_base_basic_ivf_lists;
}

void RemoveDeferredBaseListBestEffort(
    const std::filesystem::path& path, bool deferred) {
  if (!deferred) return;
  std::error_code error;
  if (!std::filesystem::exists(path, error)) {
    if (error) {
      stlqpp::LogWarn(
          "AutoCleanup: failed to inspect deferred base_list " +
          path.string() + " : " + error.message());
    }
    return;
  }
  std::filesystem::remove_all(path, error);
  if (error) {
    stlqpp::LogWarn(
        "AutoCleanup: failed to remove deferred base_list " +
        path.string() + " : " + error.message());
    return;
  }
  stlqpp::LogInfo(
      "Cleanup: deleted base_list directory after base recall " +
      path.string());
}

void RemoveDeferredBaseListRawBestEffort(
    const std::filesystem::path& path, bool deferred) {
  if (!deferred) return;
  const stlqpp::Status status = stlqpp::RuntimeBaseListStore::DiscardRawPayload(path);
  if (!status.ok()) {
    stlqpp::LogWarn(
        "AutoCleanup: failed to remove deferred base_list raw payload in " +
        path.string() + " : " + status.message());
  }
}

void RemoveTemporaryTrainingDirectoryBestEffort(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  if (error) {
    stlqpp::LogWarn("AutoCleanup: failed to remove training temporary " +
                   path.string() + " : " + error.message());
  }
}

void RemoveEmptyDirectoryBestEffort(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::is_directory(path, error) || error) return;
  if (!std::filesystem::is_empty(path, error) || error) return;
  std::filesystem::remove(path, error);
  if (error) {
    stlqpp::LogWarn("AutoCleanup: failed to remove empty directory " +
                   path.string() + " : " + error.message());
  }
}

void CleanupCompletedBuildWorkspace(
    const stlqpp::app::BuildCommand& command,
    const stlqpp::app::BuildExecutionReport& report,
    const std::filesystem::path& training_workspace) {
  // The trained model is durable outside the provisional workspace. Keep
  // failed runs intact; this hook runs only after the log and config archive
  // has been moved and the completed state has been written.
  if (command.cleanup.enabled && command.train_enabled &&
      !report.trained_model_artifact.empty()) {
    RemoveTemporaryTrainingDirectoryBestEffort(
        training_workspace / "initialized");
    RemoveTemporaryTrainingDirectoryBestEffort(
        training_workspace / "rounds");
  }
  RemoveEmptyDirectoryBestEffort(training_workspace / "log");
  RemoveEmptyDirectoryBestEffort(training_workspace);
  RemoveEmptyDirectoryBestEffort(report.work_root);
  const std::filesystem::path workspace_parent =
      command.work_directory / command.dataset_name;
  RemoveEmptyDirectoryBestEffort(workspace_parent);
}

}  // namespace

#ifndef STLQPP_BUILD_CONFIGURATION
#define STLQPP_BUILD_CONFIGURATION "unknown"
#endif
#ifndef STLQPP_BUILD_LTO_ENABLED
#define STLQPP_BUILD_LTO_ENABLED 0
#endif
#ifndef STLQPP_BUILD_SOURCE_IDENTITY
#define STLQPP_BUILD_SOURCE_IDENTITY "unregistered"
#endif
#ifndef STLQPP_BUILD_SOURCE_STATE
#define STLQPP_BUILD_SOURCE_STATE "unregistered"
#endif

int main(int argc, char** argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                    std::string_view(argv[1]) == "-h")) {
    stlqpp::app::PrintBuildUsage(std::cout);
    return EXIT_SUCCESS;
  }
  if (argc == 2 &&
      std::string_view(argv[1]) == "--print-build-identity") {
    std::cout << "build.configuration\t" STLQPP_BUILD_CONFIGURATION "\n"
              << "build.lto\t" << STLQPP_BUILD_LTO_ENABLED << '\n'
              << "build.source-identity\t"
              << STLQPP_BUILD_SOURCE_IDENTITY << '\n'
              << "build.source-state\t" << STLQPP_BUILD_SOURCE_STATE
              << '\n';
    return EXIT_SUCCESS;
  }
  std::optional<std::filesystem::path> dump_config;
  std::vector<char*> parsed_argv;
  parsed_argv.reserve(static_cast<std::size_t>(argc));
  if (argc > 0) parsed_argv.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--dump-config") {
      if (dump_config.has_value() || index + 1 >= argc) {
        stlqpp::LogError("--dump-config may be specified only once");
        return EXIT_FAILURE;
      }
      dump_config = argv[++index];
    } else {
      parsed_argv.push_back(argv[index]);
    }
  }
  auto command = stlqpp::app::ParseBuildCommand(
      static_cast<int>(parsed_argv.size()), parsed_argv.data());
  if (!command.ok()) {
    stlqpp::LogError(command.status().message());
    stlqpp::app::PrintBuildUsage(std::cerr);
    return EXIT_FAILURE;
  }
  std::optional<stlqpp::app::EvalCommand> eval_command;
  const bool post_build_base_eval =
      !command.value().eval_only && command.value().eval_base_enabled;
  const bool post_build_linkage_eval =
      !command.value().eval_only && command.value().eval_linkage_enabled;
  const bool post_build_eval =
      post_build_base_eval || post_build_linkage_eval;
  if (command.value().eval_only || post_build_eval) {
    auto parsed_eval = stlqpp::app::ParseEvalCommand(
        static_cast<int>(parsed_argv.size()), parsed_argv.data(),
        post_build_eval ||
            (command.value().eval_only &&
             !command.value().eval_only_sparse_config));
    if (!parsed_eval.ok()) {
      stlqpp::LogError(parsed_eval.status().message());
      return EXIT_FAILURE;
    }
    eval_command.emplace(std::move(parsed_eval).value());
  }
  if (dump_config.has_value()) {
    std::ofstream output(*dump_config, std::ios::binary | std::ios::trunc);
    if (!output) {
      stlqpp::LogError(
          "cannot create config snapshot " + dump_config->string());
      return EXIT_FAILURE;
    }
    if (command.value().eval_only_sparse_config) {
      stlqpp::app::WriteRoundTripEvalConfig(*eval_command, output);
    } else if (eval_command.has_value()) {
      WriteCompleteConfig(
          command.value(), *eval_command, output);
    } else {
      stlqpp::app::WriteRoundTripBuildConfig(command.value(), output);
    }
    output.close();
    return output.fail() ? EXIT_FAILURE : EXIT_SUCCESS;
  }
  if (command.value().eval_only) {
    std::optional<stlqpp::app::CompletedEvalRun> completed_run;
    if (eval_command->train_checkpoint_file.empty() &&
        eval_command->index.empty()) {
      if (eval_command->linkage_enabled &&
          !command.value().eval_only_sparse_config) {
        auto found = stlqpp::app::FindCompletedEvalRun(
            command.value(), stlqpp::OpenMpDefaultWorkerCount());
        if (!found.ok()) {
          stlqpp::LogError(found.status().message());
          return EXIT_FAILURE;
        }
        completed_run = std::move(found).value();
      }
      auto selected = completed_run.has_value()
          ? stlqpp::StatusOr<std::filesystem::path>(completed_run->trained_model)
          : stlqpp::app::ResolveImplicitTrainModel(
                command.value(), stlqpp::OpenMpDefaultWorkerCount());
      if (!selected.ok()) {
        stlqpp::LogError(selected.status().message());
        return EXIT_FAILURE;
      }
      command.value().train_checkpoint_file = selected.value();
      eval_command->train_checkpoint_file = selected.value();
      stlqpp::LogInfo("Auto-selected train model: " +
                      selected.value().string());
    }
    if (!eval_command->train_checkpoint_file.empty() &&
        (eval_command->train_checkpoint_file.filename() ==
             "checkpoint.stlqpp" ||
         eval_command->train_checkpoint_file.filename() ==
             "train_manifest.stlqpp" ||
         eval_command->train_checkpoint_file.filename() ==
             "trained_model.stlqpp" ||
         eval_command->train_checkpoint_file.filename() == "model.bin")) {
      eval_command->train_checkpoint_file =
          eval_command->train_checkpoint_file.parent_path();
    }
    const bool derive_linkage =
        eval_command->linkage_enabled && eval_command->index.empty();
    const bool derive_base_list =
        eval_command->base_enabled &&
        eval_command->requested_base_list_directory.empty() &&
        eval_command->index.empty();
    if (derive_linkage || derive_base_list) {
      if (command.value().eval_only_sparse_config) {
        stlqpp::LogError(
            "eval-only cannot derive the artifact paths from a sparse "
            "config; supply the complete build recipe and io.train_file "
            "or explicitly name the required index/base-list");
        return EXIT_FAILURE;
      }
      auto hashes = completed_run.has_value()
          ? stlqpp::StatusOr<stlqpp::IndexBuildArtifactHashes>(completed_run->hashes)
          : stlqpp::app::ComputeEvalArtifactHashes(
                command.value(), stlqpp::OpenMpDefaultWorkerCount());
      if (!hashes.ok()) {
        stlqpp::LogError("cannot derive eval linkage identity: " +
                         hashes.status().message());
        return EXIT_FAILURE;
      }
      const stlqpp::ArtifactHash64 linkage =
          hashes.value().index.linkage_list_directory;
      std::ostringstream tag;
      tag << command.value().output_prefix << "__0x" << std::hex
          << linkage.value();
      eval_command->run_root = completed_run.has_value()
          ? completed_run->root
          : command.value().output_directory /
                command.value().dataset_name / tag.str();
      if (derive_linkage) {
        eval_command->index = eval_command->run_root / "linkage_list";
      }
      if (eval_command->requested_base_list_directory.empty()) {
        eval_command->base_list_directory =
            eval_command->run_root / "base-lists";
      }
      stlqpp::LogInfo("Eval base-basic hash: " +
                      hashes.value().base_basic.ToHex());
      stlqpp::LogInfo("Eval base-list hash: " +
                      hashes.value().base_list.ToHex());
      stlqpp::LogInfo("Eval requested coefficient-codec hash: " +
                      hashes.value().index.coefficient_codec.ToHex());
      if (derive_base_list) {
        stlqpp::LogInfo("Eval derived base-list: " +
                        eval_command->base_list_directory.string());
        std::error_code error;
        if (!std::filesystem::is_directory(
                eval_command->base_list_directory, error) || error) {
          stlqpp::LogError(
              "eval-only base-list is absent for this train model and "
              "base recipe: " +
              eval_command->base_list_directory.string() +
              " (base-list hash " + hashes.value().base_list.ToHex() +
              "); set io.base_file to an existing base-list or rebuild "
              "base.encode");
          return EXIT_FAILURE;
        }
      }
      if (derive_linkage) {
        stlqpp::LogInfo("Eval derived linkage index: " +
                        eval_command->index.string());
        std::error_code error;
        if (!std::filesystem::is_directory(eval_command->index, error) ||
            error) {
          stlqpp::LogError(
              "eval-only linkage index is absent for this train model and "
              "base/topology recipe: " + eval_command->index.string() +
              " (hash " + linkage.ToHex() +
              "); set io.linkage_file to an existing index or rebuild "
              "base.linkage");
          return EXIT_FAILURE;
        }
      }
    }
    const stlqpp::Status status = stlqpp::app::RunExactDenseEval(
        std::move(*eval_command), argv);
    if (!status.ok()) {
      stlqpp::LogError(status.message());
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  if (command.value().train_checkpoint_file.empty() &&
      (!command.value().train_enabled ||
       (!command.value().train_initialization_enabled &&
        command.value().training_rounds == 0U))) {
    auto selected = stlqpp::app::ResolveImplicitTrainModel(
        command.value(), stlqpp::OpenMpDefaultWorkerCount());
    if (!selected.ok()) {
      stlqpp::LogError(selected.status().message());
      return EXIT_FAILURE;
    }
    command.value().train_checkpoint_file = selected.value();
    stlqpp::LogInfo("Auto-selected train model: " +
                    selected.value().string());
  }
  if (const stlqpp::Status status =
          stlqpp::app::ResolveBuildInputMetadata(&command.value());
      !status.ok()) {
    stlqpp::LogError(status.message());
    return EXIT_FAILURE;
  }
  if (const stlqpp::Status status =
          stlqpp::app::ValidateBuildPathIsolation(command.value());
      !status.ok()) {
    stlqpp::LogError(status.message());
    return EXIT_FAILURE;
  }
  // Prepare post-build eval affinity before any training/OpenMP work. A late
  // re-exec at the eval boundary would restart the complete build pipeline.
  auto places = stlqpp::ParseOpenMpPlaces(
      post_build_eval ? eval_command->omp_places : command.value().omp_places);
  auto proc_bind =
      stlqpp::ParseOpenMpProcBind(
          post_build_eval ? eval_command->omp_proc_bind
                          : command.value().omp_proc_bind);
  if (!places.ok() || !proc_bind.ok()) {
    stlqpp::LogError(
        !places.ok() ? places.status().message()
                     : proc_bind.status().message());
    return EXIT_FAILURE;
  }
  if (const stlqpp::Status status = stlqpp::PrepareOpenMpAffinityProcess({
          .places = places.value(),
          .proc_bind = proc_bind.value(),
          .override_environment = true,
      }, argv); !status.ok()) {
    stlqpp::LogError(status.message());
    return EXIT_FAILURE;
  }
  stlqpp::app::ResolveBuildRuntimeThreadCounts(
      &command.value(), stlqpp::OpenMpDefaultWorkerCount());
  auto recipe = stlqpp::app::ComposeEffectiveBuildRecipe(command.value());
  if (!recipe.ok()) {
    stlqpp::LogError(recipe.status().message());
    return EXIT_FAILURE;
  }
  const std::filesystem::path workspace_parent =
      command.value().work_directory / command.value().dataset_name;
  std::error_code workspace_error;
  std::filesystem::create_directories(workspace_parent, workspace_error);
  if (workspace_error) {
    stlqpp::LogError("cannot create training workspace parent: " +
                     workspace_error.message());
    return EXIT_FAILURE;
  }
  std::filesystem::path training_workspace;
  for (unsigned attempt = 0; attempt < 1000U; ++attempt) {
    auto candidate = stlqpp::NextDatedCheckpointPath(
        workspace_parent / (command.value().output_prefix + "_work"));
    if (!candidate.ok()) {
      stlqpp::LogError(candidate.status().message());
      return EXIT_FAILURE;
    }
    workspace_error.clear();
    if (std::filesystem::create_directory(
            candidate.value(), workspace_error)) {
      training_workspace = candidate.value();
      break;
    }
    if (workspace_error) {
      stlqpp::LogError("cannot reserve training workspace: " +
                       workspace_error.message());
      return EXIT_FAILURE;
    }
  }
  if (training_workspace.empty()) {
    stlqpp::LogError("training workspace date namespace exhausted");
    return EXIT_FAILURE;
  }
  auto session = stlqpp::app::BuildRunSession::StartProvisional(
      command.value(), recipe.value(), training_workspace);
  if (!session.ok()) {
    stlqpp::LogError(session.status().message());
    return EXIT_FAILURE;
  }
  stlqpp::app::BuildCommand execution_command = command.value();
  execution_command.work_directory = training_workspace;
  if (command.value().train_enabled) {
    const bool scheduler = command.value().linkage_scheduler_enabled;
    const bool merged = command.value().use_cuda && scheduler &&
        command.value().linkage_scheduler_transport_batching_enabled &&
        !command.value().linkage_scheduler_stable;
    auto identity = stlqpp::app::ComputeTrainingIdentity(
        command.value(), recipe.value(), scheduler,
        merged ? stlqpp::LinkageNumericalPolicy::kMergedThroughput
               : stlqpp::LinkageNumericalPolicy::kSourceTask);
    if (!identity.ok()) {
      stlqpp::LogError(identity.status().message());
      (void)session.value().Fail(identity.status());
      return EXIT_FAILURE;
    }
    stlqpp::LogInfo("Train model output stem: " +
                    stlqpp::app::TrainingModelPublicationBase(
                        command.value(), identity.value().training_source,
                        identity.value().training_recipe).string());
    if (command.value().train_initialization_enabled &&
        !command.value().train_checkpoint_file.empty()) {
      stlqpp::LogWarn(
          "fresh training does not read or publish to io.train_file=" +
          command.value().train_checkpoint_file.string());
    }
  }
  const bool deferred_base_list_cleanup = post_build_base_eval &&
      command.value().base_encode_enabled &&
      !RetainsBaseListStore(command.value());
  const bool deferred_base_list_raw_cleanup = post_build_base_eval &&
      command.value().base_encode_enabled &&
      command.value().cleanup.enabled &&
      !stlqpp::app::ResolveCleanupPolicy(
          command.value().cleanup).keep_base_list_raw;
  if (deferred_base_list_cleanup) {
    execution_command.cleanup.keep_base_list = 1;
  }
  if (deferred_base_list_raw_cleanup) {
    execution_command.cleanup.keep_base_list_raw = 1;
  }
  auto result = stlqpp::app::RunIndexBuild(
      execution_command, recipe.value(), {}, command.value().work_directory);
  if (!result.ok()) {
    stlqpp::LogError(result.status().message());
    if (const stlqpp::Status status =
            session.value().Fail(result.status());
        !status.ok()) {
      stlqpp::LogWarn(status.message());
    }
    return EXIT_FAILURE;
  }
  if (const stlqpp::Status status = session.value().AttachArtifactRoot(
          command.value(), recipe.value(), result.value()); !status.ok()) {
    stlqpp::LogError(status.message());
    (void)session.value().Fail(status);
    return EXIT_FAILURE;
  }
  if (post_build_base_eval) {
    stlqpp::app::EvalCommand base_eval = *eval_command;
    base_eval.base_enabled = true;
    base_eval.linkage_enabled = false;
    base_eval.base_list_directory =
        result.value().base_list_directory;
    if (!result.value().trained_model_artifact.empty()) {
      base_eval.train_checkpoint_file =
          result.value().trained_model_artifact;
    }
    base_eval.effective_threads = base_eval.threads > 0
        ? static_cast<std::uint32_t>(base_eval.threads)
        : stlqpp::OpenMpDefaultWorkerCount();
    const stlqpp::Status eval_status =
        stlqpp::app::RunBaseBasicEval(base_eval);
    RemoveDeferredBaseListBestEffort(
        result.value().base_list_directory,
        deferred_base_list_cleanup && result.value().owns_base_list);
    RemoveDeferredBaseListRawBestEffort(
        result.value().base_list_directory,
        deferred_base_list_raw_cleanup && !deferred_base_list_cleanup &&
            result.value().owns_base_list);
    if (!eval_status.ok()) {
      stlqpp::LogError(eval_status.message());
      if (const stlqpp::Status status =
              session.value().Fail(eval_status);
          !status.ok()) {
        stlqpp::LogWarn(status.message());
      }
      return EXIT_FAILURE;
    }
  }
  stlqpp::app::LogCompletedIndexBuild(result.value());
  if (post_build_linkage_eval) {
    stlqpp::app::EvalCommand linkage_eval = *eval_command;
    linkage_eval.base_enabled = false;
    linkage_eval.linkage_enabled = true;
    linkage_eval.index = result.value().index_directory;
    linkage_eval.run_root = result.value().run_root;
    if (!result.value().trained_model_artifact.empty()) {
      linkage_eval.train_checkpoint_file =
          result.value().trained_model_artifact;
    }
    if (command.value().base_linkage_enabled &&
        command.value().coefficient_codec_enabled) {
      linkage_eval.norm_preparation_backend =
          command.value().gpu_norm_enable ? "CUDA" : "CPU";
    }
    if (linkage_eval.index.empty()) {
      const stlqpp::Status status = stlqpp::Status::FailedPrecondition(
          "post-build linkage eval requested but this build did not "
          "publish a linkage index");
      stlqpp::LogError(status.message());
      (void)session.value().Fail(status);
      return EXIT_FAILURE;
    }
    const stlqpp::Status eval_status = stlqpp::app::RunExactDenseEval(
        std::move(linkage_eval), argv, true);
    if (!eval_status.ok()) {
      stlqpp::LogError(eval_status.message());
      (void)session.value().Fail(eval_status);
      return EXIT_FAILURE;
    }
  }
  if (const stlqpp::Status status =
          session.value().Complete(result.value());
      !status.ok()) {
    stlqpp::LogError(status.message());
    return EXIT_FAILURE;
  }
  CleanupCompletedBuildWorkspace(
      command.value(), result.value(), training_workspace);
  return EXIT_SUCCESS;
}
