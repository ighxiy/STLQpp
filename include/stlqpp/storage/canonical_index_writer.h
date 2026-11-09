#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_forest_format.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/index_manifest.h"

namespace stlqpp {

struct PreparedSpoolPublicationAccess;

struct CanonicalEncodedClusterView final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  CanonicalClusterRecordViews records;
  ArrayView<const float> float_coefficients_layer_major;
  ArrayView<const float> float_real_norm2;
  ArrayView<const float> float_source_norm2;
};

// Publication accounting captured by the writer that owns the canonical
// byte streams. Logical record bytes exclude shard headers and inter-record
// alignment; shard bytes are the exact persisted extents recorded in the
// manifest. This avoids reopening a production index solely for logging.
struct CanonicalIndexWriteStatistics final {
  ModelShape shape;
  IndexCounts counts;
  std::uint64_t model_shard_bytes = 0U;
  std::uint64_t directory_shard_bytes = 0U;
  std::uint64_t codes_shard_bytes = 0U;
  std::uint64_t parents_shard_bytes = 0U;
  std::uint64_t coefficients_shard_bytes = 0U;
  std::uint64_t norm8_shard_bytes = 0U;
  std::uint64_t scaled_score8_shard_bytes = 0U;
  std::uint64_t float_source_norm2_shard_bytes = 0U;
  std::uint64_t float_coefficients_shard_bytes = 0U;
  std::uint64_t float_norm2_shard_bytes = 0U;
  std::uint64_t manifest_bytes = 0U;
  std::uint64_t code_record_bytes = 0U;
  std::uint64_t parent_record_bytes = 0U;
  std::uint64_t coefficient_record_bytes = 0U;
  std::uint64_t coefficient_header_bytes = 0U;
  std::uint64_t root_coefficient_payload_bytes = 0U;
  std::uint64_t linkage_coefficient_payload_bytes = 0U;
  std::uint64_t norm8_label_bytes = 0U;
  std::uint64_t parent_raw_bits = 0U;
  std::uint64_t parent_word_padding_bits = 0U;
};

struct RuntimeCanonicalIndexWriteRequest final {
  std::filesystem::path destination;
  RuntimeCanonicalModelSource model;
  IndexCounts counts;
  IndexArtifactHashes hashes;
  // Missing only for explicitly imported legacy fixtures. Production builds
  // must provide the recipe and therefore emit canonical format v2.
  std::optional<IndexProducerRecipe> producer_recipe;
  ArrayView<const float> norm8_centers;
  bool retain_float_forest = false;
  bool retain_float_norm2 = true;
  bool retain_float_source_norm2 = false;
  bool publish_quantized_forest = true;
  std::filesystem::path model_reference = {};
  std::string norm_backend = "provided";
};

// One implementation owns publication for every runtime M. Fixed-M callers
// adapt their model views; publication itself needs no template specialization.
class RuntimeCanonicalIndexWriter final {
 public:
  [[nodiscard]] static StatusOr<RuntimeCanonicalIndexWriter> Create(
      const RuntimeCanonicalIndexWriteRequest& request);
  ~RuntimeCanonicalIndexWriter();
  RuntimeCanonicalIndexWriter(RuntimeCanonicalIndexWriter&&) noexcept;
  RuntimeCanonicalIndexWriter& operator=(
      RuntimeCanonicalIndexWriter&&) noexcept;
  RuntimeCanonicalIndexWriter(const RuntimeCanonicalIndexWriter&) = delete;
  RuntimeCanonicalIndexWriter& operator=(const RuntimeCanonicalIndexWriter&) =
      delete;
  [[nodiscard]] Status AppendEncoded(const CanonicalEncodedClusterView& source);
  [[nodiscard]] static StatusOr<RuntimeCanonicalIndexWriter> CreateAdopting(
      const RuntimeCanonicalIndexWriteRequest&, const std::filesystem::path&,
      std::vector<CanonicalCidEntry>);
  [[nodiscard]] Status SetScaledScore8(ScaledScore8CodebookView,
                                       ArtifactHash64);
  [[nodiscard]] Status Commit(
      CanonicalIndexWriteStatistics* statistics = nullptr);

 private:
  template <int M>
  friend class CanonicalIndexWriter;
  friend struct PreparedSpoolPublicationAccess;
  [[nodiscard]] Status AppendEncodedValidatedFloat(
      const CanonicalEncodedClusterView& source);
  [[nodiscard]] Status AppendEncodedImpl(
      const CanonicalEncodedClusterView& source, bool validated_float);
  [[nodiscard]] static StatusOr<RuntimeCanonicalIndexWriter> CreateImpl(
      const RuntimeCanonicalIndexWriteRequest&, const std::filesystem::path&,
      std::vector<CanonicalCidEntry>);
  struct State;
  explicit RuntimeCanonicalIndexWriter(std::unique_ptr<State> state) noexcept;
  std::unique_ptr<State> state_;
};

template <int M>
struct CanonicalIndexWriteRequest final {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  std::filesystem::path destination;
  CanonicalModelSource<M> model;
  IndexCounts counts;
  IndexArtifactHashes hashes;
  std::optional<IndexProducerRecipe> producer_recipe;
  ArrayView<const float> norm8_centers;
  bool retain_float_forest = false;
  bool retain_float_norm2 = true;
  bool retain_float_source_norm2 = false;
  bool publish_quantized_forest = true;
  std::filesystem::path model_reference = {};
  std::string norm_backend = "provided";
};

// Sequential, bounded-memory canonical publisher. A writer accepts CIDs in
// ascending order, writes the manifest last, closes every file and finally
// renames its sibling staging directory into place.
template <int M>
class CanonicalIndexWriter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<CanonicalIndexWriter<M>> Create(
      const CanonicalIndexWriteRequest<M>& request);

  ~CanonicalIndexWriter();
  CanonicalIndexWriter(CanonicalIndexWriter&& other) noexcept;
  CanonicalIndexWriter& operator=(CanonicalIndexWriter&& other) noexcept;
  CanonicalIndexWriter(const CanonicalIndexWriter&) = delete;
  CanonicalIndexWriter& operator=(const CanonicalIndexWriter&) = delete;

  [[nodiscard]] Status Append(const CanonicalClusterSource<M>& source);
  // Construction-only fast path for records emitted by the registered
  // payload codec and replayed from the atomic prepared-CID spool. It checks
  // the canonical extents but does not decode/re-encode Huffman payloads.
  [[nodiscard]] Status AppendEncoded(const CanonicalEncodedClusterView& source);
  // Adds the optional v7 score table without replacing classic norm8. It may
  // be called exactly once after the fixed-label table has been fitted.
  [[nodiscard]] Status SetScaledScore8(ScaledScore8CodebookView codebook,
                                       ArtifactHash64 hash);
  [[nodiscard]] Status Commit(
      CanonicalIndexWriteStatistics* statistics = nullptr);

 private:
  friend struct PreparedSpoolPublicationAccess;

  [[nodiscard]] Status AppendEncodedValidatedFloat(
      const CanonicalEncodedClusterView& source);
  explicit CanonicalIndexWriter(RuntimeCanonicalIndexWriter writer) noexcept;
  RuntimeCanonicalIndexWriter writer_;
};

extern template class CanonicalIndexWriter<5>;
extern template class CanonicalIndexWriter<10>;

}  // namespace stlqpp
