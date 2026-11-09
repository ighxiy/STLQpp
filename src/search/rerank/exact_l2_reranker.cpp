#include "stlqpp/search/rerank/exact_l2_reranker.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/npy_array.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace stlqpp {
namespace {
constexpr std::uint64_t kPage = 4096U;
constexpr auto kMissing = std::numeric_limits<std::uint64_t>::max();
}

struct ExactL2Reranker::State final {
  struct Worker final {
    std::vector<std::uint64_t> ordered;
    std::vector<std::pair<float, std::uint64_t>> scored;
    std::vector<std::uint8_t> read_buffer;
  };
  ExactL2RerankSource source;
  std::uint32_t scalar_bytes = 4U;
  std::uint32_t prefix = 0U;
  std::uint64_t stride = 0U;
  std::uint64_t offset = 0U;
  std::uint64_t file_size = 0U;
  std::vector<std::uint8_t> resident;
  std::vector<Worker> workers;
  QueryBatchView queries;
  SearchResult output;
  int fd = -1;
  ~State() {
#if defined(__linux__)
    if (fd >= 0) ::close(fd);
#endif
  }

  StatusOr<const std::uint8_t*> Record(std::uint64_t id, Worker& worker) const {
    if (id >= source.row_count)
      return Status::DataLoss("rerank candidate ID exceeds the original-vector slice");
    if (fd < 0) return resident.data() + static_cast<std::size_t>(id * stride);
#if defined(__linux__)
    const auto position = offset + id * stride;
    const auto aligned_position = position / kPage * kPage;
    const auto inside = position - aligned_position;
    const auto read_size = (inside + stride + kPage - 1U) / kPage * kPage;
    auto* buffer = worker.read_buffer.data();
    buffer += (kPage - reinterpret_cast<std::uintptr_t>(buffer) % kPage) % kPage;
    ssize_t bytes;
    do {
      bytes = ::pread(fd, buffer, static_cast<std::size_t>(read_size),
                      static_cast<off_t>(aligned_position));
    } while (bytes < 0 && errno == EINTR);
    const auto required = std::min(read_size, file_size - aligned_position);
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) != required)
      return Status::DataLoss("rerank direct read failed or source was truncated");
    return buffer + inside;
#else
    (void)worker;
    return Status::FailedPrecondition("direct reranking requires Linux O_DIRECT");
#endif
  }
};

ExactL2Reranker::ExactL2Reranker(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}
ExactL2Reranker::~ExactL2Reranker() = default;

StatusOr<std::unique_ptr<ExactL2Reranker>> ExactL2Reranker::Open(
    const ExactL2RerankSource& requested, std::uint32_t worker_count) {
  if (requested.dimension == 0U || worker_count == 0U ||
      (requested.storage != "memory" && requested.storage != "disk"))
    return Status::InvalidArgument("rerank requires a dimension, workers and memory/disk storage");
  try {
    auto state = std::make_unique<State>();
    state->source = requested;
    auto& source = state->source;
    auto file = ReadOnlyMappedFile::Open(source.path);
    if (!file.ok()) return file.status();
    state->file_size = file.value().size();
    if (source.format == "auto") {
      source.format = source.path.extension().string();
      if (!source.format.empty()) source.format.erase(0U, 1U);
    }
    std::uint64_t rows = 0U;
    std::uint32_t dimension = 0U;
    if (source.format == "npy") {
      auto layout = InspectNpyArray(file.value());
      if (!layout.ok()) return layout.status();
      if (layout.value().scalar != NpyScalar::kFloat16)
        return Status::InvalidArgument("rerank npy source must be float16");
      state->scalar_bytes = 2U;
      state->offset = layout.value().data_offset;
      rows = layout.value().rows;
      dimension = layout.value().columns;
    } else if (source.format == "fbin") {
      if (state->file_size < 8U) return Status::DataLoss("truncated rerank fbin header");
      state->offset = 8U;
      rows = LoadLittleEndianU32(file.value().data());
      dimension = LoadLittleEndianU32(file.value().data() + 4U);
    } else if (source.format == "fvecs" || source.format == "bvecs") {
      if (state->file_size < 4U) return Status::DataLoss("truncated rerank vector header");
      state->prefix = 4U;
      state->scalar_bytes = source.format == "bvecs" ? 1U : 4U;
      dimension = LoadLittleEndianU32(file.value().data());
    } else {
      return Status::InvalidArgument("rerank source format must be auto, bvecs, fvecs, fbin or npy");
    }
    if (dimension != source.dimension)
      return Status::DataLoss("rerank original-vector dimension differs from index");
    state->stride = state->prefix + static_cast<std::uint64_t>(dimension) * state->scalar_bytes;
    if (state->prefix) rows = state->file_size / state->stride;
    if (state->offset > state->file_size ||
        rows != (state->file_size - state->offset) / state->stride ||
        (state->file_size - state->offset) % state->stride != 0U)
      return Status::DataLoss("rerank source extent disagrees with its header");
    if (source.first_row > rows)
      return Status::InvalidArgument("rerank source slice starts beyond the file");
    if (source.row_count == 0U) source.row_count = rows - source.first_row;
    if (source.row_count > rows - source.first_row)
      return Status::InvalidArgument("rerank source slice exceeds the file");
    state->offset += source.first_row * state->stride;
    state->workers.resize(worker_count);
    if (source.storage == "memory") {
      const auto bytes = source.row_count * state->stride;
      if (bytes > std::numeric_limits<std::size_t>::max())
        return Status::ResourceExhausted("rerank source exceeds addressable memory");
      const auto* begin = file.value().data() + state->offset;
      state->resident.assign(begin, begin + static_cast<std::size_t>(bytes));
    } else {
#if defined(__linux__)
      state->fd = ::open(source.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
      if (state->fd < 0)
        return Status::FailedPrecondition("cannot open rerank source with O_DIRECT: " + std::string(std::strerror(errno)));
      struct stat info {};
      if (::fstat(state->fd, &info) != 0 || static_cast<std::uint64_t>(info.st_size) != state->file_size)
        return Status::DataLoss("rerank source changed while opening direct reader");
      // Worst unaligned record plus aligned IO extent and buffer alignment.
      const auto bytes = ((state->stride + 2U * kPage - 2U) / kPage + 1U) * kPage;
      for (auto& worker : state->workers) worker.read_buffer.resize(static_cast<std::size_t>(bytes));
#else
      return Status::FailedPrecondition("disk reranking requires Linux O_DIRECT");
#endif
    }
    return std::unique_ptr<ExactL2Reranker>(new ExactL2Reranker(std::move(state)));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("rerank source allocation failed");
  }
}

Status ExactL2Reranker::Prepare(
    const TrustedQueryBatchView& queries, std::uint32_t candidates,
    std::uint32_t top_k, SearchResult* result) {
  if (result == nullptr || queries.dimension() != state_->source.dimension ||
      top_k == 0U || candidates < top_k)
    return Status::InvalidArgument("rerank requires matching queries and candidates >= top_k > 0");
  try {
    state_->queries = {queries.data(), queries.count(), queries.dimension(), queries.stride_floats()};
    auto& output = state_->output;
    output.query_count = queries.count();
    output.top_k = top_k;
    const auto count = static_cast<std::uint64_t>(queries.count()) * top_k;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
      return Status::ResourceExhausted("rerank output extent overflows");
    output.ids.assign(static_cast<std::size_t>(count), kMissing);
    output.adc_distances.assign(static_cast<std::size_t>(count), std::numeric_limits<float>::infinity());
    for (auto& worker : state_->workers) {
      worker.ordered.reserve(candidates);
      worker.scored.reserve(candidates);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("rerank query workspace allocation failed");
  }
}

Status ExactL2Reranker::Complete(
    void* opaque, std::uint32_t worker_index, std::uint32_t query_index,
    ArrayView<const std::uint64_t> candidates) noexcept {
  auto& state = *static_cast<ExactL2Reranker*>(opaque)->state_;
  auto& worker = state.workers[worker_index];
  const float* query = state.queries.data + static_cast<std::size_t>(query_index) * state.queries.stride_floats;
  worker.ordered.assign(candidates.begin(), candidates.end());
  // Read in ID/file order only after this query's candidate set is finalized.
  std::sort(worker.ordered.begin(), worker.ordered.end());
  worker.scored.clear();
  for (const auto id : worker.ordered) {
    if (id == kMissing) break;
    auto record = state.Record(id, worker);
    if (!record.ok()) return record.status();
    if (state.prefix && LoadLittleEndianU32(record.value()) != state.source.dimension)
      return Status::DataLoss("rerank source record dimension differs");
    const auto* payload = record.value() + state.prefix;
    float distance = 0.0F;
    if (state.scalar_bytes == 4U) {
      const auto* values = reinterpret_cast<const float*>(payload);
      for (std::uint32_t d = 0U; d < state.source.dimension; ++d) {
        const float delta = query[d] - values[d]; distance += delta * delta;
      }
    } else if (state.scalar_bytes == 1U) {
      for (std::uint32_t d = 0U; d < state.source.dimension; ++d) {
        const float delta = query[d] - payload[d]; distance += delta * delta;
      }
    } else {
      for (std::uint32_t d = 0U; d < state.source.dimension; ++d) {
        const float delta = query[d] - LoadNpyFloat16(payload + d * 2U); distance += delta * delta;
      }
    }
    if (!std::isfinite(distance)) return Status::DataLoss("nonfinite exact rerank distance");
    worker.scored.emplace_back(distance, id);
  }
  const auto count = std::min<std::size_t>(state.output.top_k, worker.scored.size());
  std::partial_sort(worker.scored.begin(), worker.scored.begin() + count, worker.scored.end());
  const auto offset = static_cast<std::size_t>(query_index) * state.output.top_k;
  for (std::size_t i = 0U; i < count; ++i) {
    state.output.ids[offset + i] = worker.scored[i].second;
    state.output.adc_distances[offset + i] = worker.scored[i].first;
  }
  return Status::Ok();
}
void ExactL2Reranker::Publish(SearchResult* result) {
  *result = std::move(state_->output);
}
}  // namespace stlqpp
