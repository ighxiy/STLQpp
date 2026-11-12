#include "build_output_preparation.h"

#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stlqpp::app {
namespace {

struct Destination final {
  std::filesystem::path path;
  const char* role;
};

}  // namespace

Status PrepareBuildOutputDestinations(
    const BuildCommand& command, const EffectiveBuildRecipe& recipe) {
  std::vector<Destination> destinations;
  const bool fresh_initialization = command.train_enabled &&
      command.train_initialization_enabled &&
      command.resume_round_checkpoint.empty() &&
      command.resume_initialized_checkpoint.empty();
  if (fresh_initialization) {
    destinations.push_back({command.work_directory / "initial-buckets",
                            "initial training buckets"});
    destinations.push_back({command.work_directory / "initial-lists",
                            "initial training lists"});
    if (!command.train_checkpoint_enabled ||
        command.train_checkpoint_every_r == 0U) {
      destinations.push_back({command.work_directory / "initialized",
                              "initialized training checkpoint"});
    }
  }
  if (command.train_enabled &&
      recipe.additional_training_round_count != 0U &&
      command.resume_round_checkpoint.empty()) {
    destinations.push_back({command.work_directory / "rounds",
                            "training round checkpoints"});
  }
  if (command.base_encode_enabled) {
    std::error_code error;
    const bool resume_buckets = command.base_basic_checkpoint_enabled &&
        std::filesystem::exists(
            command.work_directory / "base-buckets", error);
    if (error) {
      return Status::Internal(
          "failed to inspect base basic checkpoint: " + error.message());
    }
    const bool resume_lists = command.base_basic_checkpoint_enabled &&
        std::filesystem::exists(command.output_directory / "base-lists", error);
    if (error)
      return Status::Internal(
          "failed to inspect completed base-list checkpoint: " + error.message());
    // The base stage validates identity and completeness before reusing lists
    // or releasing buckets. Output preparation must preserve either recovery
    // boundary, even when replacement of other outputs is allowed.
    const bool resume_base = resume_buckets || resume_lists;
    if (!resume_base) {
      destinations.push_back({command.output_directory / "base-lists",
                              "base encoding lists"});
      destinations.push_back({command.work_directory / "base-buckets",
                              "base encoding buckets"});
    }
  }

  // Finish all read-only checks before replacing the first existing artifact.
  for (const Destination& destination : destinations) {
    for (const auto& source : {
             command.effective_training_file.empty()
                 ? command.training_file : command.effective_training_file,
             command.effective_base_file.empty()
                 ? command.base_file : command.effective_base_file,
             command.train_checkpoint_file,
             command.resume_initialized_checkpoint,
             command.resume_round_checkpoint,
             command.reused_base_list_directory,
             command.control_evidence}) {
      if (source.empty()) continue;
      if (const Status status = ValidateDisjointFilesystemPaths(
              destination.path, destination.role, source, "build input");
          !status.ok()) return status;
    }
    for (const std::filesystem::path& config : command.config_files) {
      if (const Status status = ValidateDisjointFilesystemPaths(
              destination.path, destination.role, config, "build config");
          !status.ok()) return status;
    }
    std::error_code error;
    const bool exists = std::filesystem::exists(destination.path, error);
    if (error) {
      return Status::Internal(
          "failed to inspect " + std::string(destination.role) + ": " +
          destination.path.string() + ": " + error.message());
    }
    if (exists && command.protect_existing_outputs) {
      return Status::FailedPrecondition(
          std::string(destination.role) +
          " exists and large.protect_existing_outputs=true: " +
          destination.path.string());
    }
  }
  if (command.protect_existing_outputs) return Status::Ok();
  for (const Destination& destination : destinations) {
    std::error_code error;
    const bool exists = std::filesystem::exists(destination.path, error);
    if (error) {
      return Status::Internal(
          "failed to inspect " + std::string(destination.role) + ": " +
          destination.path.string() + ": " + error.message());
    }
    if (!exists) continue;
    std::filesystem::remove_all(destination.path, error);
    if (error) {
      return Status::Internal(
          "failed to replace " + std::string(destination.role) + ": " +
          destination.path.string() + ": " + error.message());
    }
    LogInfo("Replacing " + std::string(destination.role) + ": " +
            destination.path.string());
  }
  return Status::Ok();
}

}  // namespace stlqpp::app
