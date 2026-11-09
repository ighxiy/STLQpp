#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "stlqpp/core/status.h"
#include "stlqpp/core/status_or.h"
#include "stlqpp/model/model_shape.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/training/linkage/topology/cluster_topology_state.h"

namespace stlqpp {

namespace detail {
struct RoundUpdateTranscriptStorage;
struct RoundUpdateTranscriptWriterStorage;
}  // namespace detail

template <int M>
class FullTopologyCheckpointSink;

struct RoundUpdateTranscriptEntry final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint32_t virtual_count = 0U;
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;
};

template <int M>
struct RoundUpdateTranscriptCluster final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::vector<std::uint64_t> real_sample_ids;
  std::vector<LinkageNodeRole> roles;
  std::vector<std::optional<LinkageParentReference>> parents;
  std::vector<std::uint32_t> depths;
  std::vector<LinkageCodeTuple<M>> supports;
  std::vector<float> coefficients;
  std::vector<std::uint8_t> virtual_small_codes;
  std::vector<float> virtual_coefficients;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(roles.size());
  }
  [[nodiscard]] std::uint32_t virtual_count() const noexcept {
    return static_cast<std::uint32_t>(virtual_coefficients.size() /
                                      static_cast<std::size_t>(m));
  }
};

// Round-local support transcript. Exact vectors remain in the native-list
// artifact; only the frozen forest and the final support/coefficient streams
// cross the C_one solve barrier.
template <int M>
class RoundUpdateTranscript final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ~RoundUpdateTranscript();
  RoundUpdateTranscript(RoundUpdateTranscript&&) noexcept;
  RoundUpdateTranscript& operator=(RoundUpdateTranscript&&) noexcept;
  RoundUpdateTranscript(const RoundUpdateTranscript&) = delete;
  RoundUpdateTranscript& operator=(const RoundUpdateTranscript&) = delete;

  [[nodiscard]] std::uint32_t m() const noexcept;
  [[nodiscard]] std::uint32_t dimension() const noexcept;
  [[nodiscard]] std::uint32_t nlist() const noexcept;
  [[nodiscard]] StatusOr<RoundUpdateTranscriptCluster<M>> LoadCluster(
      std::uint32_t cid) const;
  [[nodiscard]] Status LoadCluster(
      std::uint32_t cid, RoundUpdateTranscriptCluster<M>* output) const;

 private:
  template <int N>
  friend class RoundUpdateTranscriptWriter;
  using State = detail::RoundUpdateTranscriptStorage;
  explicit RoundUpdateTranscript(std::unique_ptr<State> state) noexcept;
  std::unique_ptr<State> state_;
};

template <int M>
class RoundUpdateTranscriptWriter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<RoundUpdateTranscriptWriter<M>> Create(
      const std::filesystem::path& transient_destination,
      std::uint32_t dimension, std::uint32_t nlist,
      std::uint32_t m = FixedModelLayers<M>());

  ~RoundUpdateTranscriptWriter();
  RoundUpdateTranscriptWriter(RoundUpdateTranscriptWriter&&) noexcept;
  RoundUpdateTranscriptWriter& operator=(
      RoundUpdateTranscriptWriter&&) noexcept;
  RoundUpdateTranscriptWriter(const RoundUpdateTranscriptWriter&) = delete;
  RoundUpdateTranscriptWriter& operator=(const RoundUpdateTranscriptWriter&) =
      delete;

  [[nodiscard]] Status Append(const ClusterTopologyResult<M>& topology,
                              ArrayView<const std::uint64_t> real_sample_ids,
                              ArrayView<const std::uint8_t> virtual_small_codes,
                              ArrayView<const float> virtual_coefficients);
  [[nodiscard]] StatusOr<RoundUpdateTranscript<M>> Finish();

 private:
  friend class FullTopologyCheckpointSink<M>;
  // The full sink has just encoded and validated this same immutable forest.
  // It still needs the transcript's first check of real support/coefficient.
  [[nodiscard]] Status AppendAfterForestValidation(
      const ClusterTopologyResult<M>& topology,
      ArrayView<const std::uint64_t> real_sample_ids,
      ArrayView<const std::uint8_t> virtual_small_codes,
      ArrayView<const float> virtual_coefficients);
  [[nodiscard]] Status AppendImpl(
      const ClusterTopologyResult<M>& topology,
      ArrayView<const std::uint64_t> real_sample_ids,
      ArrayView<const std::uint8_t> virtual_small_codes,
      ArrayView<const float> virtual_coefficients, bool forest_validated);
  using State = detail::RoundUpdateTranscriptWriterStorage;
  explicit RoundUpdateTranscriptWriter(std::unique_ptr<State> state) noexcept;
  std::unique_ptr<State> state_;
};

extern template class RoundUpdateTranscript<5>;
extern template class RoundUpdateTranscript<10>;
extern template class RoundUpdateTranscriptWriter<5>;
extern template class RoundUpdateTranscriptWriter<10>;

}  // namespace stlqpp
