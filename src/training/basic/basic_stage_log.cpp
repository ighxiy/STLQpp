#include "stlqpp/training/basic/basic_stage_log.h"

#include "stlqpp/runtime/telemetry/logger.h"

#include <iomanip>
#include <sstream>
#include <string>

namespace stlqpp {
namespace {

std::string FormatFixed(double value, int precision) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(precision) << value;
  return output.str();
}

std::string Prefix(const BasicStageLogContext& context) {
  return context.source_is_f32
      ? "Base streaming(F32)" : "Base streaming";
}

}  // namespace

void LogBasicExecution(const BasicRuntimeExecutionInfo& execution,
                       BasicSchedule schedule) {
  std::ostringstream message;
  message << "Streaming basic encode: RSD-" << execution.rsd_rank
          << " icm_iters=" << schedule.icm_sweeps()
          << " ils_iters=" << schedule.ils_rounds()
          << " primary=" << (execution.primary_backend == BasicPrimaryBackend::kCuda
                                 ? "cuda" : "cpu");
  LogInfo(message.str());
}

void LogLargeScaleBaseEncodingCompletion(double wall_seconds) {
  std::ostringstream output;
  output << "Large-scale base encoding finished in: " << wall_seconds << "s";
  LogInfo(output.str());
}

void LogBaseListStoreCompletion(double wall_seconds) {
  std::ostringstream output;
  output << "Base list-order store built in: " << wall_seconds << "s";
  LogInfo(output.str());
}

void LogTrainingRoundPreparation(double precompute_root_seconds,
                                 double basic_open_seconds,
                                 double total_seconds) {
  LogInfo(
      "Train round prep wall time(s): precomp_croot=" +
      FormatFixed(precompute_root_seconds, 3) +
      " basic_open=" + FormatFixed(basic_open_seconds, 3) +
      " total=" + FormatFixed(total_seconds, 3));
}

void LogBasicStageCompletion(
    const BasicStageLogContext& context,
    const BasicPipelineStatistics& statistics) {
  const std::string prefix = Prefix(context);
  if (context.pipeline.log_metrics && statistics.metric_vectors != 0U) {
    const double rows = static_cast<double>(statistics.metric_vectors);
    const std::string initializer =
        " RSD-" + std::to_string(context.execution.rsd_rank) +
              " MSE: ";
    LogInfo(prefix + initializer + FormatFixed(
        statistics.initial_squared_error_sum / rows, 6));
    LogInfo(prefix + " final MSE: " + FormatFixed(
        statistics.final_squared_error_sum / rows, 6));
  }
  LogInfo(prefix + " wall time(s): " +
          FormatFixed(statistics.wall_seconds, 3));
}

}  // namespace stlqpp
