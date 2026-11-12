#pragma once

#include "build_command.h"
#include "build_execution_report.h"

#include "stlqpp/application/run_session.h"
#include "stlqpp/core/status_or.h"

#include <filesystem>
#include <string>

namespace stlqpp::app {

struct EffectiveBuildRecipe;

class BuildRunSession final {
 public:
  [[nodiscard]] static StatusOr<BuildRunSession> Start(
      const BuildCommand& command);
  [[nodiscard]] static StatusOr<BuildRunSession> Start(
      const BuildCommand& command,
      const EffectiveBuildRecipe& recipe);
  [[nodiscard]] static StatusOr<BuildRunSession> StartProvisional(
      const BuildCommand& command,
      const EffectiveBuildRecipe& recipe,
      const std::filesystem::path& provisional_root);

  ~BuildRunSession() = default;
  BuildRunSession(BuildRunSession&& other) noexcept = default;
  BuildRunSession& operator=(BuildRunSession&& other) noexcept = default;
  BuildRunSession(const BuildRunSession&) = delete;
  BuildRunSession& operator=(const BuildRunSession&) = delete;

  [[nodiscard]] Status Complete(const BuildExecutionReport& report);
  [[nodiscard]] Status AttachArtifactRoot(
      const BuildCommand& command,
      const EffectiveBuildRecipe& recipe,
      const BuildExecutionReport& report);
  [[nodiscard]] Status Fail(const Status& failure);
  [[nodiscard]] const std::filesystem::path& log_path() const noexcept {
    return owner_.log_path();
  }

 private:
  [[nodiscard]] static StatusOr<BuildRunSession> StartImpl(
      const BuildCommand& command,
      const EffectiveBuildRecipe* recipe,
      const std::filesystem::path& provisional_root = {});
  BuildRunSession(ApplicationRunSession owner,
                  BuildResourcePlan planned_resources,
                  bool active) noexcept;
  [[nodiscard]] Status WriteState(
      const char* state, const std::string& detail);

  ApplicationRunSession owner_;
  BuildResourcePlan planned_resources_;
  bool active_ = false;
  bool deferred_archive_ = false;
};

// The production completion vocabulary is inherited from Master. Stage-local
// messages are added only when their ordered migration gate is accepted.
void LogCompletedIndexBuild(const BuildExecutionReport& report);

}  // namespace stlqpp::app
