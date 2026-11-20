#include "eval_archive.h"

#include "stlqpp/application/final_index_log.h"
#include "stlqpp/runtime/telemetry/logger.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace stlqpp::app {
namespace {

std::uint64_t MixU64(std::uint64_t hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash ^= hash >> 30U;
  hash *= 0xbf58476d1ce4e5b9ULL;
  hash ^= hash >> 27U;
  hash *= 0x94d049bb133111ebULL;
  hash ^= hash >> 31U;
  return hash;
}

std::uint64_t MixString(
    std::uint64_t hash, std::string_view value) noexcept {
  for (const unsigned char byte : value) {
    hash = MixU64(hash, byte);
  }
  return MixU64(hash, value.size());
}

std::uint64_t DoubleBits(double value) noexcept {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool UsesCoefficientCodec(const EvalCommand& command) noexcept {
  return command.active_coefficient_mode == SearchCoefficientMode::kInt8;
}

const char* ActiveCoefficientModeName(const EvalCommand& command) noexcept {
  return UsesCoefficientCodec(command) ? "quantized" : "float";
}

void AppendCurrentCodecConfig(
    std::ostream& output, const EvalCommand& command) {
  const auto append_nonempty = [&output](const char* name, const auto& values) {
    if (values.empty()) return;
    output << "# " << name << "=[";
    for (std::size_t index = 0U; index < values.size(); ++index) {
      if (index != 0U) output << ',';
      output << values[index];
    }
    output << "]\n";
  };
  append_nonempty("coeff_codec_bits_per_layer",
                  command.coefficient_bits_per_layer);
  append_nonempty("coeff_codec_bits_per_layer_root",
                  command.coefficient_bits_per_layer_root);
  append_nonempty("coeff_codec_bits_per_layer_linkage",
                  command.coefficient_bits_per_layer_linkage);
  append_nonempty("coeff_codec_p_first_candidates",
                  command.coefficient_first_quantile_percents);
  append_nonempty("coeff_codec_p_rest_candidates",
                  command.coefficient_rest_quantile_percents);
  output << "# coeff_codec_use_weighted_quantile="
         << static_cast<int>(command.coefficient_weighted_quantile) << '\n'
         << "# coeff_codec_allow_clip="
         << static_cast<int>(command.coefficient_allow_clipping) << '\n'
         << "# coeff_codec_fit_scale="
         << static_cast<int>(command.coefficient_fit_scale) << '\n'
         << "# coeff_codec_q_refine_sweeps="
         << command.coefficient_refinement_sweeps << '\n'
         << "# coeff_codec_q_refine_max_layer="
         << command.coefficient_refinement_max_layer << '\n'
         << "# coeff_codec_q_refine_step_limit="
         << command.coefficient_refinement_step_limit << '\n';
}

}  // namespace

std::uint32_t EvalNormBitsPerVector(const EvalCommand& command) noexcept {
  if (command.active_score_mode == SearchScoreMode::kMipsRecon) return 0U;
  if (command.active_norm_mode != EvalNormMode::kFloat) return 8U;
  if (command.active_score_mode == SearchScoreMode::kMipsScaled) return 32U;
  return command.active_score_mode == SearchScoreMode::kScaled ? 64U : 32U;
}

std::string FormatEvalBitBudget(
    const EvalCommand& command, const LinkageBitBudget& budget) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "Eval bit budget (" << command.display_label << "): payload_bit="
         << budget.payload_bit << " total_bit="
         << budget.total_bit(EvalNormBitsPerVector(command))
         << " ivf_bit=" << budget.ivf_bit;
  return output.str();
}

EvalRunSession::EvalRunSession(ApplicationRunSession owner) noexcept
    : owner_(std::move(owner)) {}

StatusOr<EvalRunSession> EvalRunSession::Start(const EvalCommand& command) {
  auto owner = ApplicationRunSession::Start({
      .durable_root = command.run_root,
      .archive_log = command.archive_log,
      .require_existing_root = true,
  });
  if (!owner.ok()) return owner.status();
  EvalRunSession session(std::move(owner).value());
  if (session.owner_.log_attached()) {
    LogInfo("Archiving runtime log to: " +
            session.owner_.log_path().string());
  }
  return session;
}

EvalTimingSummary ComputeEvalConsoleTimingSummary(
    const std::vector<double>& core_seconds,
    std::uint64_t query_count) {
  EvalTimingSummary output;
  if (core_seconds.empty()) return output;
  std::vector<double> sorted = core_seconds;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t count = sorted.size();
  output.median_core_seconds = (count % 2U) == 0U
      ? 0.5 * (sorted[count / 2U - 1U] + sorted[count / 2U])
      : sorted[count / 2U];
  double sum = 0.0;
  for (const double value : core_seconds) sum += value;
  output.average_core_seconds = sum / static_cast<double>(count);
  output.median_qps_core = output.median_core_seconds == 0.0
      ? 0.0 : static_cast<double>(query_count) / output.median_core_seconds;
  output.average_qps_core = output.average_core_seconds == 0.0
      ? 0.0 : static_cast<double>(query_count) / output.average_core_seconds;
  return output;
}

std::vector<std::string> FormatEvalRepeatLines(
    const EvalCommand& command,
    std::uint32_t nprobe,
    std::uint32_t run,
    const EvalArchiveTiming& timing,
    std::uint64_t query_count) {
  const double qps = timing.core_seconds == 0.0
      ? 0.0 : static_cast<double>(query_count) / timing.core_seconds;
  if (command.rerank_candidates != 0U) {
    std::ostringstream line;
    line << "Linkage exact L2 rerank: nprobe=" << nprobe
         << " candidates=" << command.rerank_candidates
         << " storage=" << command.rerank_storage
         << " total_seconds=" << timing.core_seconds << " qps=" << qps;
    return {line.str()};
  }
  if (command.quiet && !timing.detailed_profile) {
    std::ostringstream output;
    output << "Linkage disk recall QPS(" << command.label
           << "): nprobe=" << nprobe << " run=" << run
           << " qps_core=" << qps;
    return {output.str()};
  }
  std::vector<std::string> lines;
  if (command.quiet) {
    std::ostringstream output;
    output << "Linkage disk recall QPS(" << command.label
           << "): nprobe=" << nprobe << " run=" << run
           << " qps_core=" << qps;
    lines.push_back(output.str());
  } else {
    if (timing.qt_rotate_seconds > 0.0) {
      lines.push_back(
          "Disk IVF linkage qt_rotate time: " +
          std::to_string(timing.qt_rotate_seconds) + "s");
    }
    lines.push_back(
        "Disk IVF linkage qt_build time: " +
        std::to_string(timing.table_build_seconds) + "s");
    lines.push_back(
        "Disk IVF linkage probe_sel time: " +
        std::to_string(timing.probe_selection_seconds) + "s");
    lines.push_back(
        "Disk IVF linkage scan+topk time: " +
        std::to_string(timing.scan_topk_seconds) + "s");
  }
  if (timing.detailed_profile) {
    lines.push_back(
        "Disk IVF linkage profile phase wall: scan+online_heap=" +
        std::to_string(timing.route_scan_phase_seconds) +
        "s finalize=" + std::to_string(timing.finalize_phase_seconds) +
        "s");
    lines.push_back(
        "Disk IVF linkage profile worker breakdown (sum over workers): "
        "probe_sel=" + std::to_string(timing.route_worker_seconds) +
        "s scan=" + std::to_string(timing.scan_worker_seconds) +
        "s online_heap=" +
        std::to_string(timing.online_topk_worker_seconds) +
        "s finalize=" + std::to_string(timing.finalize_worker_seconds) +
        "s");
    lines.push_back(
        "Disk IVF linkage logical breakdown (normalized wall; "
        "topk=online_heap+finalize): probe_sel=" +
        std::to_string(timing.logical_route_seconds) +
        "s scan=" + std::to_string(timing.logical_scan_seconds) +
        "s topk=" + std::to_string(timing.logical_topk_seconds) +
        "s total=" +
        std::to_string(timing.logical_route_seconds +
                       timing.logical_scan_seconds +
                       timing.logical_topk_seconds) + "s");
  }
  if (command.quiet) return lines;
  lines.push_back(
      "Disk IVF linkage core time "
      "(qt_rotate+qt_build+probe_sel+scan+topk): " +
      std::to_string(timing.core_seconds) + "s");
  lines.push_back(
      "Disk IVF linkage QPS (core): " + std::to_string(qps));
  return lines;
}

std::string FormatEvalSummaryLine(
    const EvalCommand& command,
    std::uint32_t nprobe,
    std::uint64_t query_count,
    const EvalTimingSummary& timing) {
  std::ostringstream output;
  if (command.rerank_candidates != 0U) {
    output << "Linkage exact L2 rerank(" << command.label << ") summary: nprobe=" << nprobe
           << " nq=" << query_count << " repeat=" << command.repeat
           << " median_total_seconds=" << timing.median_core_seconds
           << " median_qps=" << timing.median_qps_core
           << " average_total_seconds=" << timing.average_core_seconds
           << " average_qps=" << timing.average_qps_core;
    return output.str();
  }
  if (command.quiet) {
    output << "Linkage disk recall QPS(" << command.label
           << ") summary: nprobe=" << nprobe
           << " nq=" << query_count << " repeat=" << command.repeat
           << " median_qps_core=" << timing.median_qps_core
           << " avg_qps_core=" << timing.average_qps_core;
  } else {
    output << "Linkage disk recall bench(" << command.label
           << ") summary: nprobe=" << nprobe
           << " nq=" << query_count << " repeat=" << command.repeat
           << " | median_core_sec=" << timing.median_core_seconds
           << " median_qps_core=" << timing.median_qps_core
           << " | avg_core_sec=" << timing.average_core_seconds
           << " avg_qps_core=" << timing.average_qps_core;
  }
  return output.str();
}

std::string FormatEvalProbeBatchSummaryLine(
    const EvalProbeSummary& summary) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(4)
         << "nprobe=" << summary.nprobe;
  const std::uint32_t report_k = summary.metrics.result_k;
  if (report_k != 0U && report_k <= summary.metrics.recall_curve.size() &&
      report_k != 1U && report_k != 10U && report_k != 50U &&
      report_k != 100U) {
    output << " recall@" << report_k << '='
           << summary.metrics.recall_curve[report_k - 1U];
  }
  const std::string metrics = FormatEvalMetricSummary(summary.metrics);
  if (!metrics.empty()) output << ' ' << metrics;
  output << std::setprecision(1)
         << " median_qps_core=" << summary.median_qps_core
         << " avg_qps_core=" << summary.average_qps_core;
  return output.str();
}

std::vector<std::size_t> SelectEvalParetoByRecallAt1(
    const std::vector<EvalProbeSummary>& summaries) {
  constexpr double kQpsEpsilon = 1e-9;
  constexpr float kRecallEpsilon = 1e-7F;
  auto recall_at_one = [](const EvalProbeSummary& summary) {
    return summary.metrics.recall_curve.empty()
        ? -1.0F : summary.metrics.recall_curve.front();
  };
  std::vector<std::size_t> output;
  for (std::size_t candidate = 0U; candidate < summaries.size();
       ++candidate) {
    bool dominated = false;
    for (std::size_t other = 0U; other < summaries.size(); ++other) {
      if (candidate == other) continue;
      const bool qps_not_worse =
          summaries[other].average_qps_core + kQpsEpsilon >=
          summaries[candidate].average_qps_core;
      const bool recall_not_worse =
          recall_at_one(summaries[other]) + kRecallEpsilon >=
          recall_at_one(summaries[candidate]);
      const bool strictly_better =
          summaries[other].average_qps_core >
              summaries[candidate].average_qps_core + kQpsEpsilon ||
          recall_at_one(summaries[other]) >
              recall_at_one(summaries[candidate]) + kRecallEpsilon;
      if (qps_not_worse && recall_not_worse && strictly_better) {
        dominated = true;
        break;
      }
    }
    if (!dominated) output.push_back(candidate);
  }
  std::sort(output.begin(), output.end(),
            [&](std::size_t left, std::size_t right) {
              if (summaries[left].average_qps_core !=
                  summaries[right].average_qps_core) {
                return summaries[left].average_qps_core >
                    summaries[right].average_qps_core;
              }
              if (recall_at_one(summaries[left]) !=
                  recall_at_one(summaries[right])) {
                return recall_at_one(summaries[left]) >
                    recall_at_one(summaries[right]);
              }
              return summaries[left].nprobe < summaries[right].nprobe;
            });
  return output;
}

Status EvalRunSession::Complete(const std::filesystem::path&) {
  owner_.Close();
  return Status::Ok();
}

Status EvalRunSession::Fail(const Status& failure) {
  LogError("Eval failed: " + failure.message());
  owner_.Close();
  return Status::Ok();
}

std::uint64_t ComputeEvalArchiveHash(
    const EvalCommand& command, std::uint32_t nprobe) noexcept {
  // Exact frozen-DEV mixer and field order, projected onto the admitted
  // canonical native-IVF product. Paths and quiet presentation remain
  // excluded. Scaled representations and non-default evaluation contracts
  // extend the historical stream so distinct results cannot collide.
  std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
  hash = MixU64(hash, nprobe);
  if (command.rerank_candidates != 0U) {
    hash = MixString(hash, "exact-l2-rerank-v1");
    hash = MixU64(hash, command.rerank_candidates);
    hash = MixString(hash, command.rerank_storage);
    hash = MixString(hash, command.effective_base_source.generic_string());
    hash = MixString(hash, command.base_source_format);
    hash = MixU64(hash, command.base_source_first_row);
    hash = MixU64(hash, command.base_source_row_count);
  }
  hash = MixString(hash, EvalNormModeName(command.active_norm_mode));
  if (command.active_score_mode == SearchScoreMode::kCosineRecon) {
    hash = MixString(hash, "cosine_recon-v1");
  } else if (command.active_score_mode == SearchScoreMode::kMipsRecon) {
    hash = MixString(hash, "mips_recon-v1");
  } else if (command.active_score_mode == SearchScoreMode::kMipsScaled) {
    hash = MixString(hash, "mips_scaled-v1");
  }
  if (command.active_representation ==
          EvalRepresentation::kQuantizedScaled ||
      command.active_representation == EvalRepresentation::kFloatScaled) {
    // Preserve every legacy DEV-compatible archive name while preventing a
    // scaled-score result from overwriting its classic norm8 control.
    hash = MixString(hash, EvalScoreCodecName(command.active_score_codec));
    hash = MixU64(hash, command.active_score_codec_hash.value());
  }
  if (command.prepared_row_scores && command.active_score_mode != SearchScoreMode::kMipsRecon) {
    hash = MixString(hash, "independent-norm-cache-v1");
    hash = MixU64(hash, command.active_score_codec_hash.value());
  }
  if (command.active_norm_mode != EvalNormMode::kFloat) {
    hash = MixU64(hash, command.norm8_lloyd_iterations);
  }
  hash = MixString(hash, command.ivf_probe_mode);
  if (command.ivf_probe_mode == "hnsw") {
    hash = MixU64(hash, command.ivf_hnsw_m);
    hash = MixU64(hash, command.ivf_hnsw_ef_construction);
    hash = MixU64(hash, command.active_ivf_hnsw_ef_search);
    hash = MixString(hash, "unit-ip-cid-sequential-v1");
  }
  hash = MixU64(hash, 1U);  // inactive hier2 default
  hash = MixU64(hash, 0U);  // GPU norm preparation is materialized in index
  hash = MixU64(hash, 0U);  // CPU production dense scan
  hash = MixU64(hash, 0U);  // no truncated integer-qdot path
  hash = MixU64(hash, DoubleBits(1024.0));
  hash = MixU64(hash, DoubleBits(262144.0));
  hash = MixU64(hash, 0U);  // no DEV experimental specialized switch
  hash = MixString(hash, "generic");
  hash = MixString(hash, "");  // embedded norm8 labels
  hash = MixString(hash, "");  // embedded norm8 centers
  hash = MixU64(hash, 0U);      // no parent-free screen
  hash = MixU64(
      hash, static_cast<std::uint64_t>(
                static_cast<std::int64_t>(command.threads)));
  hash = MixString(hash, command.omp_places);
  hash = MixString(hash, command.omp_proc_bind);
  hash = MixU64(hash, 1U);  // configured affinity always overrides environment
  hash = MixU64(hash, command.query_block);
  // The alternate FP32 association receives an explicit identity tag.
  if (command.pretransform_query_codebooks) {
    hash = MixString(hash, "pretransform-query-codebooks-v1");
  }
  hash = MixU64(hash, 1U);  // query owner performs final top-k
  hash = MixU64(hash, 0U);  // base IVF eval is not this frontend
  hash = MixU64(hash, 1U);  // native linkage IVF is active
  hash = MixU64(hash, command.warmup);
  hash = MixU64(hash, command.repeat);
  hash = MixU64(hash, 0U);    // resident/preloaded canonical index
  hash = MixU64(hash, 1U);    // one decode bench-equivalent identity
  hash = MixU64(hash, 1U);    // DEV adaptive-u16 materialized representation
  hash = MixU64(hash, 1U);    // canonical ParentLOUDS representation
  hash = MixU64(hash, command.parent_louds_native_eval ? 1U : 0U);
  hash = MixU64(hash, 0U);    // no LOUDS index build
  hash = MixU64(hash, 128U);  // inactive DEV default
  hash = MixU64(hash, 4U);    // inactive DEV default
  const bool use_codec = UsesCoefficientCodec(command);
  hash = MixU64(hash, use_codec ? 1U : 0U);
  if (use_codec) {
    hash = MixU64(hash, 1U);  // codec enabled
    hash = MixString(hash, "cluster");
    for (const std::uint32_t bits : command.coefficient_bits_per_layer) {
      hash = MixU64(hash, bits);
    }
    for (const std::uint32_t bits :
         command.coefficient_bits_per_layer_root) {
      hash = MixU64(hash, bits);
    }
    for (const std::uint32_t bits :
         command.coefficient_bits_per_layer_linkage) {
      hash = MixU64(hash, bits);
    }
    const bool scalar_compatible =
        command.coefficient_first_quantile_percents.size() == 1U &&
        command.coefficient_rest_quantile_percents.size() == 1U &&
        command.coefficient_first_quantile_percents.front() ==
            command.coefficient_rest_quantile_percents.front();
    if (scalar_compatible) {
      hash = MixU64(hash, DoubleBits(
          command.coefficient_first_quantile_percents.front()));
      hash = MixU64(hash, DoubleBits(
          command.coefficient_rest_quantile_percents.front()));
    } else {
      hash = MixU64(
          hash, command.coefficient_first_quantile_percents.size());
      for (const double candidate :
           command.coefficient_first_quantile_percents) {
        hash = MixU64(hash, DoubleBits(candidate));
      }
      hash = MixU64(
          hash, command.coefficient_rest_quantile_percents.size());
      for (const double candidate :
           command.coefficient_rest_quantile_percents) {
        hash = MixU64(hash, DoubleBits(candidate));
      }
    }
    hash = MixU64(
        hash, command.coefficient_weighted_quantile ? 1U : 0U);
    hash = MixU64(
        hash, command.coefficient_allow_clipping ? 1U : 0U);
    hash = MixU64(hash, command.coefficient_fit_scale ? 1U : 0U);
    hash = MixU64(hash, command.coefficient_refinement_sweeps);
    hash = MixU64(hash, command.coefficient_refinement_max_layer);
    hash = MixU64(hash, command.coefficient_refinement_step_limit);
  }
  if (command.metric_mode != EvalMetricMode::kTop1Recall) {
    hash = MixU64(hash, static_cast<std::uint64_t>(command.metric_mode));
    for (const std::uint32_t gt_k : command.metric_gt_topks) {
      hash = MixU64(hash, gt_k);
    }
  }
  // Keep the historic default identity, but isolate short runs and other
  // evaluation contracts from full-result archives.
  if (command.nquery != 0U || command.top_k != 100U ||
      command.groundtruth_add1) {
    hash = MixString(hash, "eval-input-contract-v1");
    hash = MixU64(hash, command.nquery);
    hash = MixU64(hash, command.top_k);
    hash = MixU64(hash, command.groundtruth_add1 ? 1U : 0U);
  }
  return hash;
}

StatusOr<std::filesystem::path> ArchiveEvalResult(
    const EvalCommand& command,
    std::uint32_t nprobe,
    const EvalMetricBundle& metrics,
    const std::vector<double>& repeat_core_seconds,
    const EvalArchiveTiming* timing,
    const LinkageBitBudget* bit_budget) {
  if (metrics.recall_curve.empty() || repeat_core_seconds.empty()) {
    return Status::InvalidArgument("cannot archive empty eval result");
  }
  if (UsesCoefficientCodec(command) &&
      !command.producer_recipe_verified) {
    return Status::FailedPrecondition(
        "cannot archive quantized eval without verified index producer provenance");
  }
  const std::filesystem::path directory = command.run_root / "eval_result";
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) return Status::Internal("cannot create eval result directory");
  std::ostringstream filename;
  EvalCommand archive_identity = command;
  archive_identity.nquery = static_cast<std::uint32_t>(metrics.query_count);
  filename << (metrics.mode == EvalMetricMode::kTop1Recall
                   ? "recall_" : "metrics_")
           << command.label << "_0x" << std::hex << std::setw(16)
           << std::setfill('0')
           << ComputeEvalArchiveHash(archive_identity, nprobe)
           << ".txt";
  const std::filesystem::path path = directory / filename.str();
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return Status::Internal("cannot open eval archive");
  auto write_percent_curve = [&](const char* section,
                                 const std::vector<float>& curve,
                                 std::uint32_t start,
                                 bool archive_start) {
    if (section != nullptr) {
      output << "# section=" << section << '\n';
      if (archive_start) output << "# start_k=" << start << '\n';
    }
    for (std::uint32_t k = start; k <= curve.size(); ++k) {
      output << std::fixed << std::setprecision(2)
             << 100.0F * curve[k - 1U] << '\n';
    }
    if (section != nullptr) output << '\n';
  };
  if (metrics.mode == EvalMetricMode::kTop1Recall) {
    write_percent_curve(nullptr, metrics.recall_curve, 1U, false);
    output << '\n';
  } else {
    write_percent_curve(
        "top1_recall_percent", metrics.recall_curve, 1U, false);
    for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
      const std::string section = "top" + std::to_string(curve.gt_k) +
          "_recall_percent";
      write_percent_curve(
          section.c_str(), curve.recall_curve, curve.gt_k, true);
      if (metrics.mode == EvalMetricMode::kTopKNdcg) {
        output << "# section=ndcg" << curve.gt_k << "_raw\n"
               << "# start_k=" << curve.gt_k << '\n';
        for (std::uint32_t k = curve.gt_k; k <= curve.ndcg_curve.size(); ++k) {
          output << std::fixed << std::setprecision(6)
                 << curve.ndcg_curve[k - 1U] << '\n';
        }
        output << '\n';
      }
    }
    output << "# metric_mode=" << static_cast<int>(metrics.mode) << '\n'
           << "# metric_gt_topks=[";
    for (std::size_t index = 0U; index < metrics.gt_curves.size(); ++index) {
      if (index != 0U) output << ',';
      output << metrics.gt_curves[index].gt_k;
    }
    output << "]\n"
           << "# metric_summary=" << FormatEvalMetricSummary(metrics) << '\n';
  }
  output << "# nq=" << metrics.query_count << '\n'
         << "# linkage_gpu_norm_enable="
         << static_cast<int>(command.gpu_norm_enable) << '\n'
         << "# linkage_gpu_norm_execution="
         << command.norm_preparation_backend << '\n'
         << "# nprobe=" << nprobe << '\n'
         << "# coeff_mode=" << ActiveCoefficientModeName(command) << '\n'
         << "# linkage_score_mode="
         << (command.active_score_mode == SearchScoreMode::kCosineRecon
                 ? "cosine_recon"
                 : IsMipsScoreMode(command.active_score_mode)
                       ? "mips_recon" : "l2") << '\n'
         << "# disk_norm2_mode="
         << EvalNormModeName(command.active_norm_mode) << '\n'
         << "# score_codec="
         << EvalScoreCodecName(command.active_score_codec) << '\n'
         << "# score_codec_hash="
         << command.active_score_codec_hash.ToHex() << '\n';
  if (command.rerank_candidates != 0U) {
    std::error_code source_error;
    const auto bytes = std::filesystem::file_size(command.effective_base_source, source_error);
    if (source_error) return Status::Internal("cannot inspect rerank source size: " + source_error.message());
    output << "# rerank_candidates=" << command.rerank_candidates << '\n'
           << "# rerank_storage=" << command.rerank_storage << '\n'
           << "# rerank_source=" << command.effective_base_source.string() << '\n'
           << "# rerank_source_file_bytes=" << bytes << '\n'
           << "# result_distance=exact_squared_l2\n"
           << "# qps_scope=query_tables+route+scan+exact_rerank; excludes source_open\n";
  }
  if (command.minimal_query_disk_report && !command.minimal_query_disk_report->empty())
    output << "# " << *command.minimal_query_disk_report << '\n';
  if (bit_budget != nullptr) {
    const auto flags = output.flags();
    const auto precision = output.precision();
    output << std::fixed << std::setprecision(6)
           << "# payload_bit=" << bit_budget->payload_bit << '\n'
           << "# total_bit="
           << bit_budget->total_bit(EvalNormBitsPerVector(command)) << '\n'
           << "# ivf_bit=" << bit_budget->ivf_bit << '\n';
    output.flags(flags);
    output.precision(precision);
  }
  if (command.active_norm_mode != EvalNormMode::kFloat) {
    output << "# disk_norm2_lut_mode="
           << ReconstructionNorm8ModeName(command.norm8_mode) << '\n'
           << "# disk_norm2_lut_kmeans_niter="
           << command.norm8_lloyd_iterations << '\n';
  }
  if (command.ivf_probe_mode == "hnsw") {
    output << "# ivf_probe_mode=hnsw\n"
           << "# ivf_hnsw_M=" << command.ivf_hnsw_m << '\n'
           << "# ivf_hnsw_ef_construction=" << command.ivf_hnsw_ef_construction << '\n'
           << "# ivf_hnsw_ef_search=" << command.active_ivf_hnsw_ef_search << '\n';
  }
  if (metrics.mode != EvalMetricMode::kTop1Recall) {
    if (UsesCoefficientCodec(command)) {
      AppendCurrentCodecConfig(output, command);
    }
    output.close();
    if (output.fail()) return Status::Internal("cannot close eval archive");
    LogInfo("Archived eval metrics: " + path.string());
    return path;
  }
  output << "# omp_threads=" << command.threads << '\n';
  if (!command.omp_places.empty()) {
    output << "# omp_places=" << command.omp_places << '\n';
  }
  if (!command.omp_proc_bind.empty()) {
    output << "# omp_proc_bind=" << command.omp_proc_bind << '\n';
  }
  output << "# query_block=" << command.query_block << '\n'
         << "# pretransform_query_codebooks="
         << static_cast<int>(command.pretransform_query_codebooks) << '\n'
         << "# profile_timing="
         << static_cast<int>(command.profile_timing) << '\n'
         << "# warmup=" << command.warmup << '\n'
         << "# parent_louds_native_eval="
         << static_cast<int>(command.parent_louds_native_eval) << '\n';
  if (UsesCoefficientCodec(command)) {
    AppendCurrentCodecConfig(output, command);
  }
  if (timing != nullptr) {
    output << "# core_sec=" << std::fixed << std::setprecision(6)
           << timing->core_seconds << '\n'
           << "# table_build_sec=" << timing->table_build_seconds << '\n'
           << "# qt_rotate_sec=" << timing->qt_rotate_seconds << '\n'
           << "# probe_sel_sec=" << timing->probe_selection_seconds << '\n'
           << "# scan_topk_sec=" << timing->scan_topk_seconds << '\n'
           << "# route_scan_topk_sec="
           << timing->route_scan_topk_seconds << '\n'
           << "# profile_timing_active="
           << static_cast<int>(timing->detailed_profile) << '\n';
    if (timing->detailed_profile) {
      output << "# profile_route_scan_phase_sec="
             << timing->route_scan_phase_seconds << '\n'
             << "# profile_finalize_phase_sec="
             << timing->finalize_phase_seconds << '\n'
             << "# profile_route_worker_sec="
             << timing->route_worker_seconds << '\n'
             << "# profile_scan_worker_sec="
             << timing->scan_worker_seconds << '\n'
             << "# profile_online_topk_worker_sec="
             << timing->online_topk_worker_seconds << '\n'
             << "# profile_finalize_worker_sec="
             << timing->finalize_worker_seconds << '\n'
             << "# profile_logical_probe_sel_sec="
             << timing->logical_route_seconds << '\n'
             << "# profile_logical_scan_sec="
             << timing->logical_scan_seconds << '\n'
             << "# profile_logical_topk_sec="
             << timing->logical_topk_seconds << '\n';
    }
  }
  std::vector<double> qps;
  qps.reserve(repeat_core_seconds.size());
  for (std::size_t run = 0U; run < repeat_core_seconds.size(); ++run) {
    const double value = repeat_core_seconds[run];
    const double run_qps = value == 0.0 ? 0.0 : metrics.query_count / value;
    qps.push_back(run_qps);
    output << "# run=" << run << " core_sec=" << std::fixed
           << std::setprecision(6) << value << " qps_core="
           << std::setprecision(1) << run_qps << '\n';
  }
  const EvalTimingSummary timing_summary = ComputeEvalConsoleTimingSummary(
      repeat_core_seconds, metrics.query_count);
  output << "# repeat=" << repeat_core_seconds.size() << '\n'
         << "# median_core_sec=" << std::fixed << std::setprecision(6)
         << timing_summary.median_core_seconds << '\n'
         << "# median_qps_core=" << std::setprecision(1)
         << timing_summary.median_qps_core << '\n'
         << "# avg_core_sec=" << std::setprecision(6)
         << timing_summary.average_core_seconds << '\n'
         << "# avg_qps_core=" << std::setprecision(1)
         << timing_summary.average_qps_core << '\n';
  output.close();
  if (output.fail()) return Status::Internal("cannot close eval archive");
  LogInfo("Archived eval result: " + path.string());
  return path;
}

}  // namespace stlqpp::app
