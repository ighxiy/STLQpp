#pragma once

#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/final_index_publisher.h"
#include "stlqpp/index/codec/scalar_int6_encoder.h"
#include "stlqpp/index/norm/reconstruction_norm8_recipe.h"
#include "stlqpp/runtime/cpu/f32_gemm.h"
#include "stlqpp/runtime/cpu/parallel_for.h"
#include "stlqpp/runtime/cpu/spd_solver_f64.h"
#include "stlqpp/storage/index_manifest.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace stlqpp {

class SourceNormProvider;

struct CanonicalIndexRepublishRequest final {
  std::filesystem::path source;
  std::filesystem::path prepared_spool;
  std::filesystem::path destination;
  // Used only when a scaled product is requested and the source index has no
  // retained source norm shard. Stable IDs refer to this base slice.
  std::filesystem::path base_source = {};
  std::string base_source_format = "auto";
  std::uint64_t base_source_first_row = 0U;
  // Zero uses the indexed real count; a larger source slice may be needed
  // when the retained stable IDs are sparse.
  std::uint64_t base_source_row_count = 0U;
  std::vector<std::uint32_t> coefficient_bits_per_layer = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_root = {};
  std::vector<std::uint32_t> coefficient_bits_per_layer_linkage = {};
  std::vector<double> coefficient_first_magnitude_quantiles;
  std::vector<double> coefficient_rest_magnitude_quantiles;
  ScalarCoefficientCodecOptions coefficient_codec_options;
  ReconstructionNorm8Mode norm8_mode =
      ReconstructionNorm8Mode::kIdentityLloyd;
  std::uint32_t norm8_lloyd_iterations = 0U;
  bool use_cuda_norm8_fit = false;
  int cuda_norm_device = 0;
  std::uint32_t maximum_window_clusters = 64U;
  std::uint64_t target_window_real_rows = 4096U;
  // Conservative live preparation estimate; one oversized CID is admitted
  // alone. Normal windows first supply the configured workers, not one CID.
  std::uint64_t maximum_window_working_bytes = 256ULL * 1024ULL * 1024ULL;
  bool retain_float_forest = true;
  bool publish_quantized_forest = true;
  bool publish_scaled_score8 = false;
  bool publish_float_scaled = false;
  // Optional prepared original norms, borrowed for this synchronous call.
  const SourceNormProvider* source_norm2 = nullptr;
};

struct CanonicalIndexRepublishResult final {
  IndexArtifactHashes source_hashes;
  IndexArtifactHashes published_hashes;
  FinalIndexPublicationStatistics publication;
};

// Rebuilds only the registered coefficient codec and its matching norm8
// product. Topology, codes, model and original float coefficients are immutable
// inputs; native-float norms are optional and are not rebuilt here. The source
// is never modified and destination is create-only.
template <int M>
[[nodiscard]] StatusOr<CanonicalIndexRepublishResult>
RepublishCanonicalIndex(
    const CanonicalIndexRepublishRequest& request,
    const F32GemmBackend& gemm,
    const SpdSolverF64& coefficient_solver,
    const ParallelForExecutor& parallel);

extern template StatusOr<CanonicalIndexRepublishResult>
RepublishCanonicalIndex<5>(
    const CanonicalIndexRepublishRequest&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&);
extern template StatusOr<CanonicalIndexRepublishResult>
RepublishCanonicalIndex<10>(
    const CanonicalIndexRepublishRequest&,
    const F32GemmBackend&,
    const SpdSolverF64&,
    const ParallelForExecutor&);

}  // namespace stlqpp
