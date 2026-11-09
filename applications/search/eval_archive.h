#pragma once

#include "eval_command.h"

#include "stlqpp/application/run_session.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/eval/eval_metrics.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stlqpp {
struct LinkageBitBudget;
}

namespace stlqpp::app {

// Per-vector norm storage of the active scoring representation; shared
// center/gain/reference tables are accounted for by the disk-footprint report.
[[nodiscard]] std::uint32_t EvalNormBitsPerVector(
    const EvalCommand& command) noexcept;
[[nodiscard]] std::string FormatEvalBitBudget(
    const EvalCommand& command, const LinkageBitBudget& budget);

struct EvalTimingSummary final {
  double median_core_seconds = 0.0;
  double average_core_seconds = 0.0;
  double median_qps_core = 0.0;
  double average_qps_core = 0.0;
};

struct EvalArchiveTiming final {
  bool detailed_profile = false;
  double core_seconds = 0.0;
  double qt_rotate_seconds = 0.0;
  double table_build_seconds = 0.0;
  double probe_selection_seconds = 0.0;
  double scan_topk_seconds = 0.0;
  // Aggregate retained in archives for consumers of the existing timing key.
  double route_scan_topk_seconds = 0.0;
  double route_scan_phase_seconds = 0.0;
  double finalize_phase_seconds = 0.0;
  double route_worker_seconds = 0.0;
  double scan_worker_seconds = 0.0;
  double online_topk_worker_seconds = 0.0;
  double finalize_worker_seconds = 0.0;
  double logical_route_seconds = 0.0;
  double logical_scan_seconds = 0.0;
  double logical_topk_seconds = 0.0;
};

// The console and recall archive share the same even-repeat median.
[[nodiscard]] EvalTimingSummary ComputeEvalConsoleTimingSummary(
    const std::vector<double>& core_seconds,
    std::uint64_t query_count);

[[nodiscard]] std::vector<std::string> FormatEvalRepeatLines(
    const EvalCommand& command,
    std::uint32_t nprobe,
    std::uint32_t run,
    const EvalArchiveTiming& timing,
    std::uint64_t query_count);

[[nodiscard]] std::string FormatEvalSummaryLine(
    const EvalCommand& command,
    std::uint32_t nprobe,
    std::uint64_t query_count,
    const EvalTimingSummary& timing);

struct EvalProbeSummary final {
  std::uint32_t nprobe = 0U;
  EvalMetricBundle metrics;
  double median_core_seconds = 0.0;
  double median_qps_core = 0.0;
  double average_core_seconds = 0.0;
  double average_qps_core = 0.0;
};

[[nodiscard]] std::string FormatEvalProbeBatchSummaryLine(
    const EvalProbeSummary& summary);
[[nodiscard]] std::vector<std::size_t> SelectEvalParetoByRecallAt1(
    const std::vector<EvalProbeSummary>& summaries);

class EvalRunSession final {
 public:
  [[nodiscard]] static StatusOr<EvalRunSession> Start(
      const EvalCommand& command);
  ~EvalRunSession() = default;
  EvalRunSession(EvalRunSession&& other) noexcept = default;
  EvalRunSession& operator=(EvalRunSession&& other) noexcept = default;
  EvalRunSession(const EvalRunSession&) = delete;
  EvalRunSession& operator=(const EvalRunSession&) = delete;

  [[nodiscard]] Status Complete(const std::filesystem::path& archive);
  [[nodiscard]] Status Fail(const Status& failure);
  [[nodiscard]] const std::filesystem::path& log_path() const noexcept {
    return owner_.log_path();
  }

 private:
  explicit EvalRunSession(ApplicationRunSession owner) noexcept;
  ApplicationRunSession owner_;
};

[[nodiscard]] std::uint64_t ComputeEvalArchiveHash(
    const EvalCommand& command, std::uint32_t nprobe) noexcept;

[[nodiscard]] StatusOr<std::filesystem::path> ArchiveEvalResult(
    const EvalCommand& command,
    std::uint32_t nprobe,
    const EvalMetricBundle& metrics,
    const std::vector<double>& repeat_core_seconds,
    const EvalArchiveTiming* timing = nullptr,
    const LinkageBitBudget* bit_budget = nullptr);

}  // namespace stlqpp::app
