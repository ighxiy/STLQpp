#pragma once

#include "stlqpp/search/searcher.h"

#include <cstdint>
#include <string>
#include <vector>

namespace stlqpp {

enum class EvalMetricMode : std::uint8_t {
  kTop1Recall = 0,
  kTopKRecall = 1,
  kTopKNdcg = 2,
};

struct EvalGtMetricCurve final {
  std::uint32_t gt_k = 0U;
  std::vector<float> recall_curve;
  std::vector<float> ndcg_curve;
};

struct EvalMetricBundle final {
  EvalMetricMode mode = EvalMetricMode::kTop1Recall;
  std::uint32_t query_count = 0U;
  std::uint32_t result_k = 0U;
  std::vector<float> recall_curve;
  std::vector<EvalGtMetricCurve> gt_curves;
};

[[nodiscard]] bool ComputeEvalMetricBundle(
    const SearchResult& result,
    const std::vector<std::uint32_t>& groundtruth,
    std::uint32_t groundtruth_k,
    EvalMetricMode mode,
    const std::vector<std::uint32_t>& requested_gt_ks,
    EvalMetricBundle* output);

[[nodiscard]] std::vector<std::string> BuildEvalMetricReportLines(
    const EvalMetricBundle& metrics);
[[nodiscard]] std::string FormatEvalMetricSummary(
    const EvalMetricBundle& metrics);

}  // namespace stlqpp
