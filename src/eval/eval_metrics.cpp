#include "stlqpp/eval/eval_metrics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace stlqpp {
namespace {

constexpr std::uint32_t kReportPoints[] = {
    1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 20U, 35U, 50U, 100U,
    150U, 200U, 500U, 1000U, 2000U, 5000U, 10000U};

bool Contains(const std::vector<std::uint64_t>& values,
              std::uint64_t target) {
  return std::find(values.begin(), values.end(), target) != values.end();
}

float At(const std::vector<float>& values, std::uint32_t k) {
  return k == 0U || k > values.size() ? -1.0F : values[k - 1U];
}

}  // namespace

bool ComputeEvalMetricBundle(
    const SearchResult& result,
    const std::vector<std::uint32_t>& groundtruth,
    std::uint32_t groundtruth_k,
    EvalMetricMode mode,
    const std::vector<std::uint32_t>& requested_gt_ks,
    EvalMetricBundle* output) {
  if (output == nullptr || result.query_count == 0U || result.top_k == 0U ||
      result.ids.size() != static_cast<std::size_t>(result.query_count) *
                               result.top_k ||
      groundtruth_k == 0U ||
      groundtruth.size() != static_cast<std::size_t>(result.query_count) *
                                groundtruth_k) {
    return false;
  }
  *output = {};
  output->mode = mode;
  output->query_count = result.query_count;
  output->result_k = result.top_k;
  output->recall_curve.assign(result.top_k, 0.0F);
  std::vector<std::uint32_t> gt_ks;
  if (mode != EvalMetricMode::kTop1Recall) {
    for (const std::uint32_t value : requested_gt_ks) {
      if (value != 0U && value <= groundtruth_k) {
        gt_ks.push_back(value);
      }
    }
    std::sort(gt_ks.begin(), gt_ks.end());
    gt_ks.erase(std::unique(gt_ks.begin(), gt_ks.end()), gt_ks.end());
    if (gt_ks.empty()) {
      if (groundtruth_k >= 10U) gt_ks.push_back(10U);
      if (groundtruth_k >= 50U) gt_ks.push_back(50U);
      if (gt_ks.empty() && groundtruth_k > 1U) gt_ks.push_back(groundtruth_k);
    }
    if (gt_ks.empty()) return false;
    for (const std::uint32_t gt_k : gt_ks) {
      EvalGtMetricCurve curve;
      curve.gt_k = gt_k;
      curve.recall_curve.assign(result.top_k, 0.0F);
      if (mode == EvalMetricMode::kTopKNdcg) {
        curve.ndcg_curve.assign(result.top_k, 0.0F);
      }
      output->gt_curves.push_back(std::move(curve));
    }
  }

  std::vector<double> ideal_dcg;
  if (mode == EvalMetricMode::kTopKNdcg) {
    const std::uint32_t max_ideal = std::min(result.top_k, groundtruth_k);
    ideal_dcg.resize(static_cast<std::size_t>(max_ideal) + 1U, 0.0);
    for (std::uint32_t rank = 0U; rank < max_ideal; ++rank) {
      ideal_dcg[rank + 1U] = ideal_dcg[rank] +
          1.0 / std::log2(static_cast<double>(rank + 2U));
    }
  }

  for (std::uint32_t query = 0U; query < result.query_count; ++query) {
    const std::size_t pred_first = static_cast<std::size_t>(query) * result.top_k;
    const std::size_t gt_first = static_cast<std::size_t>(query) * groundtruth_k;
    bool top1_found = false;
    for (std::uint32_t rank = 0U; rank < result.top_k; ++rank) {
      top1_found = top1_found || result.ids[pred_first + rank] ==
                                     groundtruth[gt_first];
      output->recall_curve[rank] += top1_found ? 1.0F : 0.0F;
    }
    for (EvalGtMetricCurve& curve : output->gt_curves) {
      std::vector<std::uint64_t> relevant;
      relevant.reserve(curve.gt_k);
      for (std::uint32_t rank = 0U; rank < curve.gt_k; ++rank) {
        const std::uint64_t id = groundtruth[gt_first + rank];
        if (!Contains(relevant, id)) relevant.push_back(id);
      }
      std::vector<std::uint64_t> seen;
      seen.reserve(result.top_k);
      std::uint32_t intersection = 0U;
      double dcg = 0.0;
      for (std::uint32_t rank = 0U; rank < result.top_k; ++rank) {
        const std::uint64_t id = result.ids[pred_first + rank];
        const bool first = !Contains(seen, id);
        if (first) seen.push_back(id);
        if (first && Contains(relevant, id)) {
          ++intersection;
          if (mode == EvalMetricMode::kTopKNdcg) {
            dcg += 1.0 / std::log2(static_cast<double>(rank + 2U));
          }
        }
        curve.recall_curve[rank] += static_cast<float>(intersection) /
            static_cast<float>(std::max<std::size_t>(1U, relevant.size()));
        if (mode == EvalMetricMode::kTopKNdcg) {
          const double ideal = ideal_dcg[
              std::min<std::size_t>(relevant.size(), rank + 1U)];
          curve.ndcg_curve[rank] +=
              ideal == 0.0 ? 0.0F : static_cast<float>(dcg / ideal);
        }
      }
    }
  }
  const float denominator = static_cast<float>(result.query_count);
  for (float& value : output->recall_curve) value /= denominator;
  for (EvalGtMetricCurve& curve : output->gt_curves) {
    for (float& value : curve.recall_curve) value /= denominator;
    for (float& value : curve.ndcg_curve) value /= denominator;
  }
  return true;
}

std::vector<std::string> BuildEvalMetricReportLines(
    const EvalMetricBundle& metrics) {
  std::vector<std::string> output;
  auto percent = [&](const std::string& name, std::uint32_t k, float value) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(2) << name << k << " = "
         << 100.0F * value;
    output.push_back(line.str());
  };
  for (const std::uint32_t k : kReportPoints) {
    const float value = At(metrics.recall_curve, k);
    if (value >= 0.0F) percent("r@", k, value);
  }
  for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
    for (const std::uint32_t k : kReportPoints) {
      if (k < curve.gt_k) continue;
      const float value = At(curve.recall_curve, k);
      if (value >= 0.0F) {
        percent("top" + std::to_string(curve.gt_k) + "_r@", k, value);
      }
    }
    if (metrics.mode == EvalMetricMode::kTopKNdcg) {
      for (const std::uint32_t k : kReportPoints) {
        if (k < curve.gt_k) continue;
        const float value = At(curve.ndcg_curve, k);
        if (value >= 0.0F) {
          std::ostringstream line;
          line << std::fixed << std::setprecision(4) << "ndcg" << curve.gt_k
               << '@' << k << " = " << value;
          output.push_back(line.str());
        }
      }
    }
  }
  return output;
}

std::string FormatEvalMetricSummary(const EvalMetricBundle& metrics) {
  std::ostringstream output;
  bool first = true;
  const std::uint32_t points[] = {1U, 10U, 50U, 100U};
  auto append = [&](const std::string& name, float value) {
    if (!first) output << ' ';
    first = false;
    output << std::fixed << std::setprecision(4) << name << '=' << value;
  };
  for (const std::uint32_t k : points) {
    const float value = At(metrics.recall_curve, k);
    if (value >= 0.0F) append("r@" + std::to_string(k), value);
  }
  for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
    for (const std::uint32_t k : points) {
      if (k < curve.gt_k) continue;
      const float value = At(curve.recall_curve, k);
      if (value >= 0.0F) append(
          "top" + std::to_string(curve.gt_k) + "_r@" + std::to_string(k),
          value);
    }
  }
  return output.str();
}

}  // namespace stlqpp
