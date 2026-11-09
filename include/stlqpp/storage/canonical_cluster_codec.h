#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "stlqpp/core/array_view.h"
#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/codec/scalar_int6_codec.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/model/norm8_codebook.h"
#include "stlqpp/storage/canonical_cluster.h"

namespace stlqpp {

template <int M>
class PreparedClusterSpoolWriter;

// Provenance follows the owning decoded buffers and is cleared on move-from.
// Only the codec may mark a successful external ParentLOUDS validation.
class ValidatedLoudsProvenance final {
 public:
  ValidatedLoudsProvenance() = default;
  ValidatedLoudsProvenance(const ValidatedLoudsProvenance&) = delete;
  ValidatedLoudsProvenance& operator=(const ValidatedLoudsProvenance&) = delete;
  ValidatedLoudsProvenance(ValidatedLoudsProvenance&& other) noexcept
      : valid_(std::exchange(other.valid_, false)) {}
  ValidatedLoudsProvenance& operator=(
      ValidatedLoudsProvenance&& other) noexcept {
    if (this != &other) valid_ = std::exchange(other.valid_, false);
    return *this;
  }
  [[nodiscard]] bool valid() const noexcept { return valid_; }

 private:
  template <int M>
  friend class CanonicalClusterCodec;
  void Mark() noexcept { valid_ = true; }
  bool valid_ = false;
};

// Shared topology payload used by both compressed and native-float-only
// products. It deliberately has no coefficient representation.
template <int M>
struct CanonicalClusterTopologySource final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
};

// Canonical CID payload before the index-global norm8 codebook is fitted.
// Codes, ParentLOUDS and scalar-int6 coefficients are final at this boundary;
// only the one-byte real-row norm labels remain to be supplied.
template <int M>
struct CanonicalClusterPayloadSource final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ScalarInt6ClusterView<M> coefficients;
};

template <int M>
struct CanonicalClusterSource final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
  ScalarInt6ClusterView<M> coefficients;
  ArrayView<const std::uint8_t> real_norm8;
  Norm8CodebookView norm8;
};

struct CanonicalClusterRecords final {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> parents;
  std::vector<std::uint8_t> coefficients;
  std::vector<std::uint8_t> norm8;
};

struct CanonicalClusterRecordViews final {
  ArrayView<const std::uint8_t> codes;
  ArrayView<const std::uint8_t> parents;
  ArrayView<const std::uint8_t> coefficients;
  ArrayView<const std::uint8_t> norm8;
};

struct RuntimeCanonicalClusterTopologySource final {
  std::uint32_t m = 0U;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  ArrayView<const std::uint32_t> real_ids;
  // Real rows only, exactly as in CanonicalClusterTopologySource<M>.
  ArrayView<const std::uint32_t> parent_local_1based;
  ArrayView<const std::uint8_t> real_codes_small;
  ArrayView<const std::uint8_t> edge_code0;
  ArrayView<const std::uint8_t> virtual_codes_small;
};

[[nodiscard]] Status EncodeCanonicalTopologyRuntime(
    const RuntimeCanonicalClusterTopologySource& source,
    CanonicalClusterRecords& output);

[[nodiscard]] Status EncodeCanonicalClusterRuntime(
    const RuntimeCanonicalClusterTopologySource& source,
    const RuntimeScalarInt6ClusterView& coefficients,
    ArrayView<const std::uint8_t> real_norm8, Norm8CodebookView norm8,
    CanonicalClusterRecords& output);

// Owning, runtime-M form of the immutable BFS topology. The code streams
// have the same layout as DecodedCanonicalTopology<M>; parent IDs remain in
// the validated LOUDS record and are not materialized.
struct RuntimeDecodedCanonicalTopology final {
  std::uint32_t m = 0U;
  std::uint32_t cid = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::vector<std::uint32_t> real_ids;
  std::vector<std::uint64_t> parent_louds_words;
  std::vector<std::uint8_t> real_codes_small;
  std::vector<std::uint8_t> edge_code0;
  std::vector<std::uint8_t> virtual_codes_small;
};

[[nodiscard]] StatusOr<RuntimeDecodedCanonicalTopology>
DecodeCanonicalNativeBfsTopologyRuntime(std::uint32_t m, std::uint32_t cid,
                                        std::uint32_t real_count,
                                        std::uint32_t virtual_count,
                                        std::uint32_t real_root_count,
                                        ArrayView<const std::uint8_t> codes,
                                        ArrayView<const std::uint8_t> parents);

struct RuntimeDecodedCanonicalCluster final {
  RuntimeDecodedCanonicalTopology topology;
  RuntimeDecodedScalarInt6Cluster coefficients;
  std::vector<std::uint8_t> real_norm8;
};

[[nodiscard]] StatusOr<RuntimeDecodedCanonicalCluster>
DecodeCanonicalNativeBfsClusterRuntime(std::uint32_t m, std::uint32_t cid,
                                       std::uint32_t real_count,
                                       std::uint32_t virtual_count,
                                       std::uint32_t real_root_count,
                                       CanonicalClusterRecordViews records,
                                       Norm8CodebookView norm8,
                                       bool decode_norms = true);

// Owning decode of only the immutable forest structure.  Codec republishing
// deliberately uses this boundary so the old scalar-int6 payload and norm8
// labels cannot become inputs to a new coefficient/norm product.
template <int M>
class DecodedCanonicalTopology final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_ids_.size());
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& real_ids() const noexcept {
    return real_ids_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& parents() const noexcept {
    return parent_local_1based_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& parent_louds_words()
      const noexcept {
    return parent_louds_words_;
  }
  [[nodiscard]] bool has_validated_parent_louds() const noexcept {
    return louds_provenance_.valid();
  }
  [[nodiscard]] const std::vector<std::uint8_t>& real_codes_small()
      const noexcept {
    return real_codes_small_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& edge_code0() const noexcept {
    return edge_code0_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& virtual_codes_small()
      const noexcept {
    return virtual_codes_small_;
  }

 private:
  template <int N>
  friend class CanonicalClusterCodec;

  std::uint32_t cid_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::vector<std::uint32_t> real_ids_;
  std::vector<std::uint32_t> parent_local_1based_;
  std::vector<std::uint64_t> parent_louds_words_;
  ValidatedLoudsProvenance louds_provenance_;
  std::vector<std::uint8_t> real_codes_small_;
  std::vector<std::uint8_t> edge_code0_;
  std::vector<std::uint8_t> virtual_codes_small_;
};

template <int M>
class DecodedCanonicalCluster final {
 public:
  static_assert(M >= 2 && M <= 20);

  [[nodiscard]] std::uint32_t cid() const noexcept { return cid_; }
  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(real_ids_.size());
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return virtual_count_;
  }
  [[nodiscard]] std::uint32_t real_root_count() const noexcept {
    return real_root_count_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& real_ids() const noexcept {
    return real_ids_;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& parents() const noexcept {
    return parent_local_1based_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>& parent_louds_words()
      const noexcept {
    return parent_louds_words_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& real_codes_small()
      const noexcept {
    return real_codes_small_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& edge_code0() const noexcept {
    return edge_code0_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& virtual_codes_small()
      const noexcept {
    return virtual_codes_small_;
  }
  [[nodiscard]] const DecodedScalarInt6Cluster<M>& coefficients()
      const noexcept {
    return coefficients_;
  }
  [[nodiscard]] const std::vector<std::uint8_t>& real_norm8() const noexcept {
    return real_norm8_;
  }

  [[nodiscard]] StatusOr<TrustedClusterView<M>> trusted_view(
      Norm8CodebookView norm8) const;
  [[nodiscard]] StatusOr<TrustedBfsClusterView<M>> trusted_bfs_view(
      Norm8CodebookView norm8) const;

 private:
  template <int N>
  friend class CanonicalClusterCodec;

  std::uint32_t cid_ = 0U;
  std::uint32_t virtual_count_ = 0U;
  std::uint32_t real_root_count_ = 0U;
  std::vector<std::uint32_t> real_ids_;
  std::vector<std::uint32_t> parent_local_1based_;
  std::vector<std::uint64_t> parent_louds_words_;
  ValidatedLoudsProvenance louds_provenance_;
  std::vector<std::uint8_t> real_codes_small_;
  std::vector<std::uint8_t> edge_code0_;
  std::vector<std::uint8_t> virtual_codes_small_;
  DecodedScalarInt6Cluster<M> coefficients_;
  std::vector<std::uint8_t> real_norm8_;
};

template <int M>
class CanonicalClusterCodec final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static Status EncodeTopology(
      const CanonicalClusterTopologySource<M>& source,
      CanonicalClusterRecords& output);

  [[nodiscard]] static Status EncodePayload(
      const CanonicalClusterPayloadSource<M>& source,
      CanonicalClusterRecords& output);

  // Recode an already prepared coefficient product without serializing the
  // unchanged topology. Used by codec-only index publication.
  [[nodiscard]] static Status EncodeCoefficientRecord(
      const ScalarInt6ClusterView<M>& source,
      std::vector<std::uint8_t>& output);

  [[nodiscard]] static Status Encode(const CanonicalClusterSource<M>& source,
                                     CanonicalClusterRecords& output);

  [[nodiscard]] static StatusOr<DecodedCanonicalCluster<M>> Decode(
      std::uint32_t cid, std::uint32_t real_count, std::uint32_t virtual_count,
      std::uint32_t real_root_count, CanonicalClusterRecordViews records,
      Norm8CodebookView norm8);

  // Production retrieval decode: validates and retains ParentLOUDS words
  // without expanding a parent integer for every node.
  [[nodiscard]] static StatusOr<DecodedCanonicalCluster<M>> DecodeNativeBfs(
      std::uint32_t cid, std::uint32_t real_count, std::uint32_t virtual_count,
      std::uint32_t real_root_count, CanonicalClusterRecordViews records,
      Norm8CodebookView norm8, bool decode_norms = true);

  [[nodiscard]] static StatusOr<DecodedCanonicalTopology<M>> DecodeTopology(
      std::uint32_t cid, std::uint32_t real_count, std::uint32_t virtual_count,
      std::uint32_t real_root_count, ArrayView<const std::uint8_t> codes,
      ArrayView<const std::uint8_t> parents,
      std::uint32_t m = FixedModelLayers<M>());

  // Float retrieval boundary: decodes only shared codes/IDs and retains the
  // canonical ParentLOUDS stream.  It never reads scalar-int6 or norm8 data.
  [[nodiscard]] static StatusOr<DecodedCanonicalTopology<M>>
  DecodeNativeBfsTopology(std::uint32_t cid, std::uint32_t real_count,
                          std::uint32_t virtual_count,
                          std::uint32_t real_root_count,
                          ArrayView<const std::uint8_t> codes,
                          ArrayView<const std::uint8_t> parents);

 private:
  template <int N>
  friend class PreparedClusterSpoolWriter;

  // Only the spool writer can serialize the completed in-process product
  // without decoding its freshly generated Huffman stream again.
  [[nodiscard]] static Status EncodePreparedPayload(
      const CanonicalClusterPayloadSource<M>& source,
      CanonicalClusterRecords& output);
};

extern template class DecodedCanonicalTopology<5>;
extern template class DecodedCanonicalTopology<10>;
extern template class DecodedCanonicalCluster<5>;
extern template class DecodedCanonicalCluster<10>;
extern template class CanonicalClusterCodec<5>;
extern template class CanonicalClusterCodec<10>;

}  // namespace stlqpp
