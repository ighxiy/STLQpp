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
struct RootOnlyInitializationTranscriptStorage;
struct RootOnlyInitializationTranscriptWriterStorage;
}  // namespace detail

struct RootOnlyInitializationTranscriptEntry final {
  std::uint32_t cid = 0U;
  std::uint32_t real_count = 0U;
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;
};

// One decoded root-only initialization forest. The row domain is the native
// list order used by the topology builder; no exact vectors are duplicated.
template <int M>
struct RootOnlyInitializationTranscriptCluster final {
  static_assert(M == 0 || (M >= 2 && M <= 20));

  std::uint32_t m = M;
  std::uint32_t cid = 0U;
  std::vector<LinkageNodeRole> roles;
  std::vector<std::optional<LinkageParentReference>> parents;
  std::vector<std::uint32_t> depths;
  std::vector<LinkageCodeTuple<M>> supports;
  std::vector<float> coefficients;

  [[nodiscard]] std::uint32_t real_count() const noexcept {
    return static_cast<std::uint32_t>(roles.size());
  }
};

// Process-local, disk-backed transcript of the pre-C_one forest. The first
// pass needs the complete CID-error distribution before DEV's bad-cluster
// weights are known. Keeping only topology/support/coefficients makes the
// second exact-LS pass scale to the 30M training recipe without retaining
// vectors or d-dimensional reconstructions.
template <int M>
class RootOnlyInitializationTranscript final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  ~RootOnlyInitializationTranscript();
  RootOnlyInitializationTranscript(
      RootOnlyInitializationTranscript&& other) noexcept;
  RootOnlyInitializationTranscript& operator=(
      RootOnlyInitializationTranscript&& other) noexcept;
  RootOnlyInitializationTranscript(const RootOnlyInitializationTranscript&) =
      delete;
  RootOnlyInitializationTranscript& operator=(
      const RootOnlyInitializationTranscript&) = delete;

  [[nodiscard]] std::uint32_t m() const noexcept;
  [[nodiscard]] std::uint32_t dimension() const noexcept;
  [[nodiscard]] std::uint32_t nlist() const noexcept;
  [[nodiscard]] StatusOr<RootOnlyInitializationTranscriptCluster<M>>
  LoadCluster(std::uint32_t cid) const;

 private:
  template <int N>
  friend class RootOnlyInitializationTranscriptWriter;

  using State = detail::RootOnlyInitializationTranscriptStorage;
  explicit RootOnlyInitializationTranscript(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

template <int M>
class RootOnlyInitializationTranscriptWriter final {
 public:
  static_assert(M == 0 || (M >= 2 && M <= 20));

  [[nodiscard]] static StatusOr<RootOnlyInitializationTranscriptWriter<M>>
  Create(const std::filesystem::path& transient_destination,
         std::uint32_t dimension, std::uint32_t nlist,
         std::uint32_t m = FixedModelLayers<M>());

  ~RootOnlyInitializationTranscriptWriter();
  RootOnlyInitializationTranscriptWriter(
      RootOnlyInitializationTranscriptWriter&& other) noexcept;
  RootOnlyInitializationTranscriptWriter& operator=(
      RootOnlyInitializationTranscriptWriter&& other) noexcept;
  RootOnlyInitializationTranscriptWriter(
      const RootOnlyInitializationTranscriptWriter&) = delete;
  RootOnlyInitializationTranscriptWriter& operator=(
      const RootOnlyInitializationTranscriptWriter&) = delete;

  [[nodiscard]] Status Append(const ClusterTopologyResult<M>& topology);
  [[nodiscard]] StatusOr<RootOnlyInitializationTranscript<M>> Finish();

 private:
  using State = detail::RootOnlyInitializationTranscriptWriterStorage;
  explicit RootOnlyInitializationTranscriptWriter(
      std::unique_ptr<State> state) noexcept;

  std::unique_ptr<State> state_;
};

extern template class RootOnlyInitializationTranscript<5>;
extern template class RootOnlyInitializationTranscript<10>;
extern template class RootOnlyInitializationTranscriptWriter<5>;
extern template class RootOnlyInitializationTranscriptWriter<10>;

}  // namespace stlqpp
