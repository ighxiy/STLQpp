#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/index/build/final_cluster_postprocessor.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/storage/index_manifest.h"

namespace stlqpp {

struct PreparedClusterSpoolRequest final {
  std::filesystem::path destination;
  ModelShape shape;
  IndexArtifactHashes hashes;
  bool retain_float_forest = true;
  bool retain_source_norm2 = false;
  bool publish_quantized_forest = true;
  // Completion-order streams adopted without copying. A non-checkpoint spool
  // uses transient publication; the canonical writer supplies durability.
  bool direct_payloads = false;
  // Internal publication demand, independent of retaining float coefficients.
  bool retain_float_norm2 = true;
};

struct PreparedClusterSpoolEntry final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint32_t real_root_count = 0U;
  std::uint64_t codes_offset = 0U;
  std::uint64_t codes_bytes = 0U;
  std::uint64_t parents_offset = 0U;
  std::uint64_t parents_bytes = 0U;
  std::uint64_t coefficients_offset = 0U;
  std::uint64_t coefficients_bytes = 0U;
  std::uint64_t norm2_row_offset = 0U;
  std::uint64_t float_coefficient_row_offset = 0U;
  std::uint64_t float_coefficient_byte_offset = 0U;
  std::uint64_t physical_norm2_row_offset = 0U;
};

template <int M>
class PreparedClusterSpoolWriter;

// Shared persistence state. M belongs to the record/model metadata, not to
// file ownership, transactions, restart or byte-stream publication.
class PreparedClusterSpoolWriterStorage final {
 public:
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriterStorage> Create(
      const PreparedClusterSpoolRequest&, int expected_m = 0);
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriterStorage>
  CreateCheckpointed(const PreparedClusterSpoolRequest&, int expected_m = 0);
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriterStorage>
  OpenCheckpointed(const PreparedClusterSpoolRequest&, int expected_m = 0);
  ~PreparedClusterSpoolWriterStorage();
  PreparedClusterSpoolWriterStorage(
      PreparedClusterSpoolWriterStorage&&) noexcept;
  PreparedClusterSpoolWriterStorage& operator=(
      PreparedClusterSpoolWriterStorage&&) noexcept;
  [[nodiscard]] Status Commit();
  [[nodiscard]] std::uint32_t next_cid() const noexcept;
  [[nodiscard]] const IndexCounts& committed_counts() const noexcept;
  [[nodiscard]] bool direct_payloads() const noexcept;
  [[nodiscard]] const std::vector<std::uint8_t>& completed_cids()
      const noexcept;

 private:
  template <int M>
  friend class PreparedClusterSpoolWriter;
  struct Payload {
    std::uint32_t cid, real_count, virtual_count, real_root_count;
    const CanonicalClusterRecords& records;
    ArrayView<const float> exact_real_norm2;
    ArrayView<const float> float_coefficients_layer_major;
    ArrayView<const float> float_real_norm2;
    ArrayView<const float> source_real_norm2;
  };
  [[nodiscard]] Status CheckAppend(std::uint32_t m, std::uint32_t cid,
                                   bool quantized, bool complete) const;
  [[nodiscard]] Status AppendSerialized(const Payload&);
  struct State;
  explicit PreparedClusterSpoolWriterStorage(std::unique_ptr<State>) noexcept;
  std::unique_ptr<State> state_;
};

// The typed adapter only invokes the matching payload codec. All persistence
// and checkpoint work belongs to the non-templated storage implementation.
template <int M>
class PreparedClusterSpoolWriter final {
 public:
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriter<M>> Create(
      const PreparedClusterSpoolRequest& request) {
    auto storage = PreparedClusterSpoolWriterStorage::Create(request, M);
    if (!storage.ok()) return storage.status();
    return PreparedClusterSpoolWriter(std::move(storage).value());
  }
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriter<M>>
  CreateCheckpointed(const PreparedClusterSpoolRequest& request) {
    auto storage =
        PreparedClusterSpoolWriterStorage::CreateCheckpointed(request, M);
    if (!storage.ok()) return storage.status();
    return PreparedClusterSpoolWriter(std::move(storage).value());
  }
  [[nodiscard]] static StatusOr<PreparedClusterSpoolWriter<M>> OpenCheckpointed(
      const PreparedClusterSpoolRequest& request) {
    auto storage =
        PreparedClusterSpoolWriterStorage::OpenCheckpointed(request, M);
    if (!storage.ok()) return storage.status();
    return PreparedClusterSpoolWriter(std::move(storage).value());
  }
  [[nodiscard]] Status Append(const PreparedCanonicalCluster<M>& cluster);
  [[nodiscard]] Status Commit() { return storage_.Commit(); }
  [[nodiscard]] bool direct_payloads() const noexcept {
    return storage_.direct_payloads();
  }
  [[nodiscard]] const std::vector<std::uint8_t>& completed_cids()
      const noexcept {
    return storage_.completed_cids();
  }
  [[nodiscard]] std::uint32_t next_cid() const noexcept {
    return storage_.next_cid();
  }
  [[nodiscard]] const IndexCounts& committed_counts() const noexcept {
    return storage_.committed_counts();
  }

 private:
  explicit PreparedClusterSpoolWriter(PreparedClusterSpoolWriterStorage storage)
      : storage_(std::move(storage)) {}
  PreparedClusterSpoolWriterStorage storage_;
};

// Read-only owner for an atomically completed spool. Large files are mapped
// into virtual address space; callers decode only bounded row ranges.
class PreparedClusterSpoolStorage {
 public:
  [[nodiscard]] static StatusOr<PreparedClusterSpoolStorage> Open(
      const std::filesystem::path& directory, int expected_m = 0);

  [[nodiscard]] bool direct_payloads() const noexcept {
    return direct_payloads_;
  }
  [[nodiscard]] const std::filesystem::path& directory() const noexcept {
    return directory_;
  }
  [[nodiscard]] const ModelShape& shape() const noexcept { return shape_; }
  [[nodiscard]] const IndexCounts& counts() const noexcept { return counts_; }
  [[nodiscard]] const IndexArtifactHashes& hashes() const noexcept {
    return hashes_;
  }
  [[nodiscard]] const PreparedClusterSpoolEntry& entry(
      std::uint32_t cid) const noexcept {
    return entries_[cid];
  }
  [[nodiscard]] CanonicalClusterRecordViews payload_records(
      std::uint32_t cid) const noexcept;
  [[nodiscard]] Status ReadNorm2(std::uint64_t row_begin,
                                 ArrayView<float> output) const;
  [[nodiscard]] bool has_float_forest() const noexcept {
    return has_float_forest_;
  }
  [[nodiscard]] bool has_float_norm2() const noexcept {
    return has_float_norm2_;
  }
  [[nodiscard]] Status ReadFloatCoefficients(std::uint32_t cid,
                                             ArrayView<float> output) const;
  [[nodiscard]] Status ReadFloatNorm2(std::uint64_t row_begin,
                                      ArrayView<float> output) const;
  [[nodiscard]] bool has_source_norm2() const noexcept {
    return has_source_norm2_;
  }
  [[nodiscard]] bool has_quantized_forest() const noexcept {
    return has_quantized_forest_;
  }
  [[nodiscard]] Status ReadSourceNorm2(std::uint64_t row_begin,
                                       ArrayView<float> output) const;

 private:
  std::filesystem::path directory_;
  ModelShape shape_;
  IndexCounts counts_;
  IndexArtifactHashes hashes_;
  std::vector<PreparedClusterSpoolEntry> entries_;
  ReadOnlyMappedFile records_;
  ReadOnlyMappedFile codes_, parents_, coefficients_;
  bool direct_payloads_ = false;
  std::vector<std::uint64_t> logical_norm_starts_;
  [[nodiscard]] Status ReadNormStream(const ReadOnlyMappedFile&, std::uint64_t,
                                      ArrayView<float>) const;
  ReadOnlyMappedFile norm2_;
  ReadOnlyMappedFile float_coefficients_;
  ReadOnlyMappedFile float_norm2_;
  ReadOnlyMappedFile source_norm2_;
  bool has_float_forest_ = false;
  bool has_float_norm2_ = false;
  bool has_source_norm2_ = false;
  bool has_quantized_forest_ = true;
};

// Fixed-width callers retain a checked boundary, not another reader
// implementation.
template <int M>
class PreparedClusterSpool final : public PreparedClusterSpoolStorage {
 public:
  PreparedClusterSpool() = default;
  static_assert(M == 0 ||
                (M >= kMinimumModelLayers && M <= kMaximumModelLayers));
  [[nodiscard]] static StatusOr<PreparedClusterSpool<M>> Open(
      const std::filesystem::path& directory) {
    auto opened = PreparedClusterSpoolStorage::Open(directory, M);
    if (!opened.ok()) return opened.status();
    return PreparedClusterSpool(std::move(opened).value());
  }

 private:
  explicit PreparedClusterSpool(PreparedClusterSpoolStorage storage)
      : PreparedClusterSpoolStorage(std::move(storage)) {}
};

extern template class PreparedClusterSpoolWriter<5>;
extern template class PreparedClusterSpoolWriter<10>;

}  // namespace stlqpp
