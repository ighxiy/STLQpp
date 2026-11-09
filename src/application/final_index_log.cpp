#include "stlqpp/application/final_index_log.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_index_files.h"

#include <cmath>
#include <cstdint>
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

std::string FormatTimerSeconds(double value) {
  std::ostringstream output;
  output << value;
  return output.str();
}

std::uint64_t ShardBytes(const IndexManifest& manifest, ShardKind kind) {
  for (const ShardDescriptor& shard : manifest.shards) {
    if (shard.kind == kind) return shard.bytes;
  }
  return 0U;
}

LinkageBitBudget ComputeLinkageBitBudget(
    const CanonicalIndexWriteStatistics& value) {
  const double real = static_cast<double>(value.counts.real);
  const double total = real + static_cast<double>(value.counts.virtual_roots);
  const double m = static_cast<double>(value.shape.m);
  const double linkage_ratio =
      static_cast<double>(value.counts.linked_real) / real;
  const double virtual_ratio =
      static_cast<double>(value.counts.virtual_roots) / real;
  const double coefficient_bits =
      8.0 * static_cast<double>(value.coefficient_record_bytes);
  const double ivf_bits = value.shape.nlist > 1U
      ? std::log2(static_cast<double>(value.shape.nlist)) : 0.0;
  return {
      .payload_bit = (1.0 - linkage_ratio) * 8.0 * (m - 1.0) +
          linkage_ratio * 8.0 * m + coefficient_bits / total,
      .information_bit = ivf_bits +
          (1.0 - linkage_ratio + virtual_ratio) * 8.0 * (m - 1.0) +
          linkage_ratio * 8.0 * m + coefficient_bits / real +
          static_cast<double>(value.parent_raw_bits) / real,
      .ivf_bit = ivf_bits,
  };
}

Status LogLinkageBitBudget(const CanonicalIndexWriteStatistics& value) {
  if (!value.shape.Validate().ok() || value.counts.real == 0U ||
      value.counts.real_roots + value.counts.linked_real !=
          value.counts.real ||
      value.coefficient_header_bytes +
              value.root_coefficient_payload_bytes +
              value.linkage_coefficient_payload_bytes !=
          value.coefficient_record_bytes ||
      value.parent_record_bytes * 8U < value.parent_raw_bits ||
      value.parents_shard_bytes <
          kCanonicalPayloadHeaderBytes + value.parent_record_bytes) {
    return Status::DataLoss("linkage bit-budget inputs are incomplete");
  }
  const std::uint64_t coeff_payload_bytes =
      value.root_coefficient_payload_bytes +
      value.linkage_coefficient_payload_bytes;
  const std::uint64_t nreal = value.counts.real;
  const std::uint64_t nvirt = value.counts.virtual_roots;
  const std::uint64_t ntotal = nreal + nvirt;
  const double real = static_cast<double>(nreal);
  const double total = static_cast<double>(ntotal);
  const double m = static_cast<double>(value.shape.m);
  const double linkage_ratio =
      static_cast<double>(value.counts.linked_real) / real;
  const double virtual_ratio = static_cast<double>(nvirt) / real;
  const double root_ratio =
      static_cast<double>(value.counts.real_roots) / real;
  const double coeff_data_bits_per_coeff_nreal =
      8.0 * static_cast<double>(coeff_payload_bytes) / (real * m);
  const double coeff_all_bits_per_coeff_nreal =
      8.0 * static_cast<double>(value.coefficient_record_bytes) /
      (real * m);
  const double coeff_data_bits_per_coeff_ntotal =
      8.0 * static_cast<double>(coeff_payload_bytes) / (total * m);
  const double coeff_all_bits_per_coeff_ntotal =
      8.0 * static_cast<double>(value.coefficient_record_bytes) /
      (total * m);
  const double louds_raw_bits_per_real =
      static_cast<double>(value.parent_raw_bits) / real;
  const double louds_disk_bits_per_real =
      8.0 * static_cast<double>(value.parents_shard_bytes) / real;
  const std::uint64_t parent_shard_header_bits =
      static_cast<std::uint64_t>(kCanonicalPayloadHeaderBytes) * 8U;
  const std::uint64_t parent_alignment_bits =
      (value.parents_shard_bytes - kCanonicalPayloadHeaderBytes -
       value.parent_record_bytes) * 8U;

  LogInfo("========== Linkage Bit Budget ==========");
  LogInfo("Linkage store stats: nreal=" + std::to_string(nreal) +
          " nvirt=" + std::to_string(nvirt) +
          " virt_ratio(nvirt/nreal)=" + FormatFixed(virtual_ratio, 6) +
          " nroot_real=" + std::to_string(value.counts.real_roots) +
          " root_ratio=" + FormatFixed(root_ratio, 6) +
          " linkage_ratio=" + FormatFixed(linkage_ratio, 6));
  LogInfo(
      "Linkage parent LOUDS bit usage (canonical per-CID records): "
      "raw_bits=Σ(2n+1)=" + std::to_string(value.parent_raw_bits) +
      " extra(shard_header_bits=" +
      std::to_string(parent_shard_header_bits) +
      " padded_word_bits=" +
      std::to_string(value.parent_word_padding_bits) +
      " record_alignment_bits=" +
      std::to_string(parent_alignment_bits) + ")");
  LogInfo("Linkage parent LOUDS bits/real: louds_raw=" +
          FormatFixed(louds_raw_bits_per_real, 6) +
          " louds_disk(reference)=" +
          FormatFixed(louds_disk_bits_per_real, 6));
  LogInfo("Linkage coeff codec bits/real: coeff_data_nreal=" +
          FormatFixed(coeff_data_bits_per_coeff_nreal, 6) +
          " coeff_all_nreal(data+header)=" +
          FormatFixed(coeff_all_bits_per_coeff_nreal, 6));
  LogInfo("Linkage coeff codec bits/ntotal: coeff_data_ntotal=" +
          FormatFixed(coeff_data_bits_per_coeff_ntotal, 6) +
          " coeff_all_ntotal(data+header)=" +
          FormatFixed(coeff_all_bits_per_coeff_ntotal, 6));

  const LinkageBitBudget budget = ComputeLinkageBitBudget(value);
  LogInfo(
      "Summary of payload bits budget(root_code_real+linkage_code_real+"
      "coeff_all_ntotal): [(1-linkage_ratio) * 8 * (m-1)] + "
      "(linkage_ratio * 8 * m) + (coeff_all_ntotal * m)= " +
      FormatFixed(budget.payload_bit, 6) + " bit/per vector");
  LogInfo(
      "Summary of information bits budget(ivf+root_code_virt+"
      "root_code_real+linkage_code_real+coeff_all_nreal+louds_raw): "
      "log2(nlist) + [(1-linkage_ratio+virt_ratio) * 8 * (m-1)] + "
      "(linkage_ratio * 8 * m) + (coeff_all_nreal * m) + louds_raw= " +
      FormatFixed(budget.information_bit, 6) +
      " bit/per vector");
  return Status::Ok();
}

}  // namespace

Status LogFinalTopologySummary(
    const FullTopologyPipelineStatistics& statistics,
    const IndexCounts& published_counts,
    const std::filesystem::path& index_directory) {
  if (index_directory.empty() || statistics.real_rows == 0U ||
      statistics.resumed_real_rows > statistics.real_rows ||
      published_counts.real != statistics.real_rows ||
      published_counts.linked_real > published_counts.real) {
    return Status::DataLoss(
        "final topology log counts disagree with publication");
  }
  const double rows = static_cast<double>(statistics.real_rows);
  const auto measured_rows = statistics.real_rows - statistics.resumed_real_rows;
  if (measured_rows != 0U && !statistics.has_squared_error)
    return Status::DataLoss("final topology log is missing current-run quality statistics");
  LogInfo("Linkage topology/codec/norm pipeline wall time(s): " +
          FormatTimerSeconds(statistics.wall_seconds) + "s");
  LogInfo("Linkage store dir: " + index_directory.string());
  LogInfo("BaseSet linkage summary:");
  LogInfo("  Total virtual nodes: " +
          std::to_string(published_counts.virtual_roots));
  LogInfo("  Linkage Ratio: " +
          FormatFixed(100.0 *
                          static_cast<double>(published_counts.linked_real) /
                          rows,
                      2) +
          "%");
  if (statistics.resumed_real_rows != 0U) {
    LogInfo("  Checkpoint rows reused: " +
            std::to_string(statistics.resumed_real_rows) +
            "; newly encoded rows: " + std::to_string(measured_rows));
    LogInfo("  Full-build depth/error/removal statistics unavailable: "
            "checkpoint does not retain historical producer metrics.");
    if (measured_rows == 0U) return Status::Ok();
    LogInfo("  Following depth/error/removal statistics cover newly encoded rows only:");
  }
  LogInfo("  Unused virtual nodes removed before publication: " +
          std::to_string(statistics.unused_virtual_roots_removed) +
          " (generated=" +
          std::to_string(statistics.virtual_roots +
                         statistics.unused_virtual_roots_removed) +
          ", retained=" + std::to_string(statistics.virtual_roots) + ")");
  LogInfo("  Max depth: " +
          std::to_string(statistics.maximum_depth) +
          " Mean depth: " +
          FormatFixed(static_cast<double>(statistics.depth_sum) /
                          static_cast<double>(measured_rows), 2));
  LogInfo("  Linkage Error: Max=" +
          FormatFixed(statistics.maximum_squared_error, 4) +
          ", Min=" + FormatFixed(statistics.minimum_squared_error, 4) +
          ", Mean=" +
          FormatFixed(statistics.squared_error_sum /
                          static_cast<double>(measured_rows), 4));
  return Status::Ok();
}

Status LogFinalPublicationSummary(
    const FinalIndexPublicationStatistics& statistics) {
  if (statistics.norm8_fit_wall_seconds)
    LogInfo("Norm8 LUT fit wall time(s): " +
          FormatFixed(*statistics.norm8_fit_wall_seconds, 6));
  const CanonicalIndexWriteStatistics& value = statistics.canonical;
  if (value.shape.Validate().ok() == false || value.counts.real == 0U ||
      value.counts.real_roots + value.counts.linked_real !=
          value.counts.real ||
      value.coefficient_header_bytes +
              value.root_coefficient_payload_bytes +
              value.linkage_coefficient_payload_bytes !=
          value.coefficient_record_bytes ||
      value.parent_record_bytes * 8U < value.parent_raw_bits ||
      value.parents_shard_bytes <
          kCanonicalPayloadHeaderBytes + value.parent_record_bytes ||
      value.norm8_shard_bytes < value.norm8_label_bytes) {
    return Status::DataLoss(
        "final publication log input is missing complete writer statistics");
  }

  return Status::Ok();
}

namespace {

StatusOr<CanonicalIndexWriteStatistics> ReadLinkageBitStatistics(
    const CanonicalIndexFiles& files, bool native_float) {
  const IndexManifest& manifest = files.manifest();
  CanonicalIndexWriteStatistics value;
  value.shape = manifest.shape;
  value.counts = manifest.counts;
  value.parents_shard_bytes = ShardBytes(manifest, ShardKind::kParents);
  if (value.parents_shard_bytes == 0U) {
    return Status::DataLoss("linkage bit budget has no parent shard");
  }
  for (std::uint32_t cid = 0U; cid < manifest.shape.nlist; ++cid) {
    const CanonicalCidEntry& entry = files.entry(cid);
    const std::uint64_t nodes =
        static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
    const std::uint64_t raw_parent_bits = 2U * nodes + 1U;
    if (entry.parents_bytes * 8U < raw_parent_bits) {
      return Status::DataLoss("linkage bit budget has an invalid CID record");
    }
    if (native_float) {
      const auto coefficients = files.float_cluster_records(cid).coefficients;
      if (coefficients.size() != nodes * manifest.shape.m * sizeof(float)) {
        return Status::DataLoss("linkage bit budget has no float coefficients");
      }
      value.coefficient_record_bytes += coefficients.size();
    } else {
      const auto records = files.cluster_records(cid);
      if (records.coefficients.size() < 8U) {
        return Status::DataLoss("linkage bit budget has no coefficient record");
      }
      const std::uint64_t root_payload_bytes =
          LoadLittleEndianU32(records.coefficients.data());
      const std::uint64_t linkage_payload_bytes =
          LoadLittleEndianU32(records.coefficients.data() + 4U);
      if (root_payload_bytes + linkage_payload_bytes >
          entry.coefficients_bytes) {
        return Status::DataLoss(
            "linkage bit budget coefficient payload exceeds its record");
      }
      value.coefficient_record_bytes += entry.coefficients_bytes;
      value.coefficient_header_bytes += entry.coefficients_bytes -
          root_payload_bytes - linkage_payload_bytes;
      value.root_coefficient_payload_bytes += root_payload_bytes;
      value.linkage_coefficient_payload_bytes += linkage_payload_bytes;
    }
    value.parent_record_bytes += entry.parents_bytes;
    value.parent_raw_bits += raw_parent_bits;
    value.parent_word_padding_bits +=
        entry.parents_bytes * 8U - raw_parent_bits;
  }
  return value;
}

}  // namespace

StatusOr<LinkageBitBudget> ReadEvalLinkageBitBudget(
    const CanonicalIndexFiles& files, bool native_float) {
  auto statistics = ReadLinkageBitStatistics(files, native_float);
  if (!statistics.ok()) return statistics.status();
  if (statistics.value().counts.real == 0U) {
    return Status::InvalidArgument("cannot report bit budget for an empty index");
  }
  return ComputeLinkageBitBudget(statistics.value());
}

Status LogEvalLinkageBitBudget(const CanonicalIndexFiles& files) {
  if (!files.manifest().has_quantized_forest()) return Status::Ok();
  auto statistics = ReadLinkageBitStatistics(files, false);
  if (!statistics.ok()) return statistics.status();
  return LogLinkageBitBudget(statistics.value());
}

}  // namespace stlqpp
