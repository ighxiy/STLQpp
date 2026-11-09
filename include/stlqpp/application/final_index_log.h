#pragma once

#include "stlqpp/core/status.h"
#include "stlqpp/index/build/final_index_publisher.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"

#include <filesystem>

namespace stlqpp {

class CanonicalIndexFiles;

// Bits per real vector. The payload/information components exclude norms,
// stable IDs, shared tables/models, file headers and alignment. Information
// includes IVF, virtual-root amortization and raw ParentLOUDS; payload does not.
struct LinkageBitBudget final {
  double payload_bit = 0.0;
  double information_bit = 0.0;
  double ivf_bit = 0.0;

  // IVF is already included in information_bit. Norm labels or exact scoring
  // fields are counted once for the active representation.
  [[nodiscard]] double total_bit(std::uint32_t norm_bits) const noexcept {
    return information_bit + static_cast<double>(norm_bits);
  }
};

[[nodiscard]] StatusOr<LinkageBitBudget> ReadEvalLinkageBitBudget(
    const CanonicalIndexFiles& files, bool native_float);

// DEV-compatible renderer for the final ordinary-linkage result.  It accepts
// producer quality statistics and writer counts, without reopening/scanning
// an index. Resumed quality statistics cover only newly encoded rows.
[[nodiscard]] Status LogFinalTopologySummary(
    const FullTopologyPipelineStatistics& statistics,
    const IndexCounts& published_counts,
    const std::filesystem::path& index_directory);

// Validate publication statistics and report the norm LUT fit wall time.
[[nodiscard]] Status LogFinalPublicationSummary(
    const FinalIndexPublicationStatistics& statistics);

// Report the published linkage bit budget when eval opens the canonical
// index, matching the DEV eval-stage log order.
[[nodiscard]] Status LogEvalLinkageBitBudget(
    const CanonicalIndexFiles& files);

}  // namespace stlqpp
