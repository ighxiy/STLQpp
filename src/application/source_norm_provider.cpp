#include "stlqpp/application/source_norm_provider.h"

#include <array>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

#include "stlqpp/application/sha256.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/index_manifest_codec.h"
#include "stlqpp/training/io/bvecs_u8_source.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "stlqpp/training/io/fvecs_f32_source.h"

namespace stlqpp {
namespace {

constexpr std::uint32_t kRowsPerBatch = 65536U;
constexpr std::size_t kNormHeaderBytes = 176U;

std::uint32_t BatchRows(std::uint32_t dimension) {
  return std::min(kRowsPerBatch, std::max(256U, (4U * 1024U * 1024U) / dimension));
}

struct SourceNormRows {
  const float* values;
  float* output;
  std::uint32_t dimension;
  std::size_t rows;
  bool u8;
  std::atomic<bool> invalid{false};
};

void ComputeSourceNormRows(void* opaque, std::uint32_t, std::uint32_t task) noexcept {
  auto& work = *static_cast<SourceNormRows*>(opaque);
  const auto end = std::min(work.rows, (std::size_t(task) + 1U) * 256U);
  for (std::size_t row = std::size_t(task) * 256U; row < end; ++row) {
    const float* values = work.values + row * work.dimension;
    float norm = 0.0F;
    if (work.u8) {
      std::uint64_t sum = 0U;
      for (std::uint32_t axis = 0; axis < work.dimension; ++axis) {
        const auto value = static_cast<std::uint32_t>(values[axis]);
        sum += value * value;
      }
      norm = static_cast<float>(sum);
    } else {
      for (std::uint32_t axis = 0; axis < work.dimension; ++axis)
        norm += values[axis] * values[axis];
    }
    if (!std::isfinite(norm)) work.invalid.store(true, std::memory_order_relaxed);
    work.output[row] = norm;
  }
}

std::string ResolveFormat(const std::filesystem::path& path,
                          std::string_view requested) {
  if (requested != "auto") return std::string(requested);
  const std::string suffix = path.extension().string();
  if (suffix == ".bvecs" || suffix == ".siftbin") return "bvecs";
  if (suffix == ".fvecs") return "fvecs";
  if (suffix == ".fbin") return "fbin";
  if (suffix == ".npy") return "npy";
  return "auto";
}

constexpr std::uint64_t kScatterRows = 65536U;
struct PendingNormScatter {
  const CanonicalIndexFiles* files;
  const std::uint8_t* norms;
  const std::vector<std::uint64_t>* starts;
  std::vector<float>* output;
  std::atomic<unsigned> failure{0U};
};

void ScatterPendingNorms(void* opaque, std::uint32_t, std::uint32_t chunk) noexcept {
  auto& work = *static_cast<PendingNormScatter*>(opaque);
  if (work.failure.load(std::memory_order_relaxed) != 0U) return;
  const auto& starts = *work.starts;
  auto begin = static_cast<std::uint64_t>(chunk) * kScatterRows;
  const auto end = std::min(begin + kScatterRows, starts.back());
  auto cid = static_cast<std::uint32_t>(
      std::upper_bound(starts.begin(), starts.end(), begin) - starts.begin() - 1);
  while (begin < end) {
    const auto limit = std::min(end, starts[cid + 1U]);
    const auto codes = work.files->topology_records(cid).codes;
    for (; begin < limit; ++begin) {
      const auto local = begin - starts[cid];
      const auto id = LoadLittleEndianU32(codes.data() + local * 4U);
      const float norm = LoadLittleEndianF32(work.norms + begin * 4U);
      if (id >= work.output->size() || !std::isfinite(norm) || norm < 0.0F) {
        unsigned expected_failure = 0U;
        (void)work.failure.compare_exchange_strong(expected_failure,
            id >= work.output->size() ? 1U : 2U, std::memory_order_relaxed);
        return;
      }
      // The existing duplicate-ID check must remain race-free for corrupt
      // external input. A relaxed compare/exchange combines validation and
      // scatter; no per-row side table or additional population is allocated.
      float expected = std::numeric_limits<float>::quiet_NaN();
#if defined(_MSC_VER)
      const bool inserted = std::atomic_ref<float>((*work.output)[id]).compare_exchange_strong(
          expected, norm, std::memory_order_relaxed);
#else
      float desired = norm;
      const bool inserted = __atomic_compare_exchange(
          &(*work.output)[id], &expected, &desired, false,
          __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif
      if (!inserted) {
        unsigned expected_failure = 0U;
        (void)work.failure.compare_exchange_strong(expected_failure, 2U, std::memory_order_relaxed);
        return;
      }
    }
    ++cid;
  }
}


}  // namespace

StatusOr<SourceNormProvider> SourceNormProvider::Open(
    const std::filesystem::path& path, std::string_view requested_format,
    std::uint64_t first_row, std::uint64_t row_count, std::uint32_t dimension,
    const ParallelForExecutor& parallel) {
  if (dimension == 0U) return Status::InvalidArgument("source dimension must be positive");
  const std::string format = ResolveFormat(path, requested_format);
  if (format == "bvecs") {
    BaseBvecsU8SourceFactory factory(
        {
            .path = path,
            .dimension = dimension,
            .first_vector = first_row,
            .vector_count = row_count,
            .block_vectors = BatchRows(dimension),
        },
        parallel);
    auto source = factory.OpenExactRows();
    if (!source.ok()) return source.status();
    return SourceNormProvider(std::move(source).value(), dimension, true);
  }
  F32SourceLayout layout;
  if (format == "fvecs")
    layout = F32SourceLayout::kFvecs;
  else if (format == "fbin")
    layout = F32SourceLayout::kFbin;
  else if (format == "npy")
    layout = F32SourceLayout::kNpyFloat16;
  else
    return Status::InvalidArgument(
        "base format must be auto, bvecs, fvecs, fbin or npy");
  BaseFvecsF32SourceFactory factory(
      {
          .path = path,
          .dimension = dimension,
          .first_vector = first_row,
          .vector_count = row_count,
          .block_vectors = BatchRows(dimension),
          .layout = layout,
      },
      parallel);
  auto source = factory.OpenExactRows();
  if (!source.ok()) return source.status();
  return SourceNormProvider(std::move(source).value(), dimension, false);
}

Status SourceNormProvider::Read(ArrayView<const std::uint32_t> stable_ids,
                                std::vector<float>* output) const {
  const InlineParallelFor serial;
  return Read(stable_ids, output, serial);
}

Status SourceNormProvider::Read(ArrayView<const std::uint32_t> stable_ids,
                                std::vector<float>* output,
                                const ParallelForExecutor& parallel) const {
  if (output == nullptr) {
    return Status::InvalidArgument("source norm output is null");
  }
  output->resize(stable_ids.size());
  const SourceNormRead read{stable_ids, *output};
  return ReadBatch({&read, 1U}, parallel);
}

Status SourceNormProvider::ReadBatch(ArrayView<const SourceNormRead> reads,
                                     const ParallelForExecutor& parallel) const {
  for (const auto& read : reads)
    if (read.stable_ids.size() != read.output.size())
      return Status::InvalidArgument("source norm batch output extent mismatch");
  if (!cache_mapping_.empty() || !norms_by_id_.empty()) {
    std::vector<SourceNormRead> chunks;
    std::size_t chunk_count = 0U;
    for (const auto& read : reads) chunk_count += (read.stable_ids.size() + 4095U) / 4096U;
    chunks.reserve(chunk_count);
    for (const auto& read : reads) {
      for (std::size_t first = 0U; first < read.stable_ids.size(); first += 4096U) {
        const auto count = std::min<std::size_t>(4096U, read.stable_ids.size() - first);
        chunks.push_back({{read.stable_ids.data() + first, count}, {read.output.data() + first, count}});
      }
    }
    if (chunks.empty()) return Status::Ok();
    struct CachedRows {
      const std::uint8_t* mapped;
      const float* values;
      std::size_t rows;
      const SourceNormRead* chunks;
      std::atomic<bool> invalid{false};
    } work{cache_mapping_.empty() ? nullptr : cache_mapping_.data() + kNormHeaderBytes,
           norms_by_id_.data(), cache_mapping_.empty() ? norms_by_id_.size()
               : (cache_mapping_.size() - kNormHeaderBytes) / 4U,
           chunks.data()};
    parallel.Run(static_cast<std::uint32_t>(chunks.size()), &work,
        [](void* opaque, std::uint32_t, std::uint32_t chunk) noexcept {
      auto& w = *static_cast<CachedRows*>(opaque);
      const auto& read = w.chunks[chunk];
      for (std::size_t row = 0U; row < read.stable_ids.size(); ++row) {
        const auto id = read.stable_ids[row];
        if (id >= w.rows) { w.invalid.store(true, std::memory_order_relaxed); continue; }
        const auto norm = w.mapped ? LoadLittleEndianF32(w.mapped + id * 4ULL) : w.values[id];
        if (!std::isfinite(norm) || norm < 0.0F)
          w.invalid.store(true, std::memory_order_relaxed);
        read.output[row] = norm;
      }
    });
    return work.invalid.load(std::memory_order_relaxed)
        ? Status::DataLoss("original-vector norm cache does not cover stable ID or has invalid norm")
        : Status::Ok();
  }
  for (const auto& read : reads)
    if (const auto status = ReadRaw(read, parallel); !status.ok()) return status;
  return Status::Ok();
}

Status SourceNormProvider::ReadRaw(const SourceNormRead& read,
                                   const ParallelForExecutor& parallel) const {
  const auto stable_ids = read.stable_ids;
  std::vector<std::uint64_t> ids;
  std::vector<float> exact;
  const auto batch_rows = BatchRows(dimension_);
  for (std::size_t first = 0U; first < stable_ids.size();
       first += batch_rows) {
    const std::size_t count =
        std::min<std::size_t>(batch_rows, stable_ids.size() - first);
    ids.resize(count);
    for (std::size_t row = 0U; row < count; ++row) {
      ids[row] = stable_ids[first + row];
    }
    if (const Status status = source_->Gather(ids, &exact); !status.ok()) {
      return status;
    }
    SourceNormRows work{exact.data(), read.output.data() + first, dimension_, count, source_is_u8_};
    parallel.Run(static_cast<std::uint32_t>((count + 255U) / 256U), &work, &ComputeSourceNormRows);
    if (work.invalid.load(std::memory_order_relaxed))
      return Status::DataLoss("source base contains a non-finite norm");
  }
  return Status::Ok();
}

StatusOr<SourceNormProvider> SourceNormProvider::OpenForIndex(
    const std::filesystem::path& index, const IndexManifest& manifest,
    const std::filesystem::path& base, std::string_view format,
    std::uint64_t first_row, std::uint64_t requested_rows,
    const ParallelForExecutor& parallel) {
  constexpr std::size_t header_bytes = kNormHeaderBytes;
  const auto owner = CanonicalSourceIndexDirectory(index, manifest);
  auto cache = owner / "norm-cache/constant/original_vector_norm2.f32";
  const auto new_cache = cache;
  const auto pending_norms = owner / "norm-cache/constant/source_norm2.pending";
  std::error_code error;
  bool cached = std::filesystem::exists(cache, error);
  if (!cached && !error &&
      std::filesystem::exists(owner / "original_vector_norm2.f32", error)) {
    cache = owner / "original_vector_norm2.f32";
    cached = true;
  }
  if (error)
    return Status::Internal("cannot inspect original-vector norm cache: " +
                            error.message());
  std::uint64_t rows =
      requested_rows == 0U ? manifest.counts.real : requested_rows;
  std::string identity;
  const bool base_available =
      !base.empty() && std::filesystem::exists(base, error);
  if (error)
    return Status::Internal("cannot inspect original base: " + error.message());
  const auto make_identity = [&](std::uint64_t count) -> StatusOr<std::string> {
    const auto path = std::filesystem::canonical(base, error);
    if (error)
      return Status::Internal("cannot resolve original base: " +
                              error.message());
    const auto size = std::filesystem::file_size(path, error);
    if (error)
      return Status::Internal("cannot inspect original base size: " +
                              error.message());
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error)
      return Status::Internal("cannot inspect original base timestamp: " +
                              error.message());
    std::ostringstream id;
    id << "original-vector-norm-v1\n"
       << path.generic_string() << '\n'
       << size << '\n'
       << modified.time_since_epoch().count() << '\n'
       << ResolveFormat(base, format) << '\n'
       << first_row << '\n'
       << count << '\n'
       << manifest.shape.dimension;
    return app::Sha256Bytes(id.str());
  };
  SourceNormProvider out(nullptr, manifest.shape.dimension, false);
  if (cached) {
    auto mapped = ReadOnlyMappedFile::Open(cache);
    if (!mapped.ok()) return mapped.status();
    const auto& file = mapped.value();
    if (file.size() < header_bytes ||
        std::memcmp(file.data(), "STLQORN1", 8U) != 0 ||
        (LoadLittleEndianU32(file.data() + 8U) != 1U &&
         LoadLittleEndianU32(file.data() + 8U) != 2U) ||
        LoadLittleEndianU32(file.data() + 12U) != header_bytes ||
        LoadLittleEndianU32(file.data() + 16U) != manifest.shape.dimension ||
        LoadLittleEndianU32(file.data() + 20U) > 1U ||
        LoadLittleEndianU64(file.data() + 40U) !=
            manifest.hashes.linkage_list_directory.value())
      return Status::DataLoss(
          "original-vector norm cache identity disagrees with index");
    rows = LoadLittleEndianU64(file.data() + 24U);
    if (rows < manifest.counts.real ||
        (requested_rows != 0U && requested_rows != rows) ||
        (base_available &&
         first_row != LoadLittleEndianU64(file.data() + 32U)) ||
        rows > (std::numeric_limits<std::size_t>::max() - header_bytes) / 4U ||
        file.size() != header_bytes + rows * 4U)
      return Status::DataLoss(
          "original-vector norm cache extent or base slice mismatch");
    out.identity_.assign(reinterpret_cast<const char*>(file.data() + 48U), 64U);
    // A legacy cache created without the base is bound to its source shard
    // and forest, rather than to unavailable base file metadata.
    if (base_available && LoadLittleEndianU32(file.data() + 20U) == 0U) {
      auto expected = make_identity(rows);
      if (!expected.ok()) return expected.status();
      if (out.identity_ != expected.value())
        return Status::DataLoss(
            "original-vector norm cache base identity mismatch");
    }
    const std::string checksum(
        reinterpret_cast<const char*>(file.data() + 112U), 64U);
    // Old headers carry a content checksum in this slot. It remains an
    // opaque generation token for old scale-cache keys; never recompute it.
    // New headers carry a lightweight recipe/source token in the same slot.
    out.identity_ = checksum;
    out.cache_path_ = cache;
    out.cache_mapping_ = std::move(mapped).value();
    std::filesystem::remove(pending_norms, error);
    if (error) return Status::Internal("cannot remove consumed source norm stream: " +
                                       error.message());
    return out;
  }
  if (rows > std::numeric_limits<std::uint32_t>::max())
    return Status::ResourceExhausted(
        "original-vector norm cache exceeds stable-ID domain");
  const auto prepare_begin = std::chrono::steady_clock::now();
  LogInfo("Original-vector norm cache: begin preparation");
  auto legacy_index = index;
  IndexManifest legacy_manifest = manifest;
  if (!manifest.has_float_source_norm2() && !base_available && owner != index) {
    auto metadata = ReadOnlyMappedFile::Open(owner / "manifest.stlqpp");
    if (!metadata.ok()) return metadata.status();
    if (const auto status = IndexManifestCodec::Decode(
            metadata.value().data(), metadata.value().size(), legacy_manifest);
        !status.ok())
      return status;
    if (legacy_manifest.hashes.linkage_list_directory !=
        manifest.hashes.linkage_list_directory)
      return Status::DataLoss(
          "original norm owner disagrees with derived index");
    legacy_index = owner;
  }
  const bool pending_available = std::filesystem::exists(pending_norms, error);
  if (error) return Status::Internal("cannot inspect source norm stream: " + error.message());
  if (pending_available) {
    auto stream = ReadOnlyMappedFile::Open(pending_norms);
    if (!stream.ok()) return stream.status();
    if (stream.value().size() != manifest.counts.real * 4ULL)
      return Status::DataLoss("source norm stream extent disagrees with index");
    auto files = CanonicalIndexFiles::OpenCoefficients(
        index, manifest.has_float_forest());
    if (!files.ok()) return files.status();
    out.norms_by_id_.assign(rows, std::numeric_limits<float>::quiet_NaN());
    bool serial_scatter = true;
    if (manifest.counts.real >= kScatterRows && parallel.worker_count() > 1U) {
      std::vector<std::uint64_t> starts(manifest.shape.nlist + 1U, 0U);
      for (std::uint32_t cid = 0U; cid < manifest.shape.nlist; ++cid)
        starts[cid + 1U] = starts[cid] + files.value().entry(cid).real_count;
      PendingNormScatter work{&files.value(), stream.value().data(), &starts, &out.norms_by_id_};
      parallel.Run(static_cast<std::uint32_t>((starts.back() + kScatterRows - 1U) / kScatterRows),
                   &work, &ScatterPendingNorms);
      const auto failure = work.failure.load(std::memory_order_relaxed);
      if (failure == 1U && requested_rows == 0U) {
        // Sparse imported IDs may exceed the inferred dense prefix. Retry the
        // existing growable path; ordinary full-base publication is one pass.
        std::fill(out.norms_by_id_.begin(), out.norms_by_id_.end(), std::numeric_limits<float>::quiet_NaN());
      } else if (failure != 0U) {
        return Status::DataLoss(failure == 1U
            ? "source norm stable ID exceeds base slice"
            : "source norm stream contains invalid or duplicate rows");
      } else {
        serial_scatter = false;
      }
    }
    if (serial_scatter) {
      std::size_t offset = 0;
      for (std::uint32_t cid = 0; cid < manifest.shape.nlist; ++cid) {
        const auto codes = files.value().topology_records(cid).codes;
        for (std::uint32_t row = 0; row < files.value().entry(cid).real_count; ++row) {
          const auto id = LoadLittleEndianU32(codes.data() + row * 4ULL);
          if (id >= out.norms_by_id_.size()) {
            if (requested_rows != 0U)
              return Status::DataLoss("source norm stable ID exceeds base slice");
            out.norms_by_id_.resize(static_cast<std::size_t>(id) + 1U,
                                    std::numeric_limits<float>::quiet_NaN());
          }
          const float norm = LoadLittleEndianF32(stream.value().data() + offset);
          offset += 4U;
          if (!std::isfinite(norm) || norm < 0 || !std::isnan(out.norms_by_id_[id]))
            return Status::DataLoss("source norm stream contains invalid or duplicate rows");
          out.norms_by_id_[id] = norm;
        }
      }
    }
    rows = out.norms_by_id_.size();
    for (auto& norm : out.norms_by_id_)
      if (std::isnan(norm)) norm = 0.0F;  // Unindexed IDs are never scored.
  } else if (legacy_manifest.has_float_source_norm2() && !base_available) {
    // Read legacy source norms without creating another legacy shard.
    auto files = CanonicalIndexFiles::OpenCoefficients(
        legacy_index, legacy_manifest.has_float_forest(), true);
    if (!files.ok()) return files.status();
    out.norms_by_id_.assign(rows, std::numeric_limits<float>::quiet_NaN());
    for (std::uint32_t cid = 0; cid < manifest.shape.nlist; ++cid) {
      const auto& entry = files.value().entry(cid);
      const auto topology = files.value().topology_records(cid);
      std::vector<float> norms(entry.real_count);
      if (const auto status = files.value().ReadFloatSourceNorm2(cid, norms);
          !status.ok())
        return status;
      for (std::size_t row = 0; row < norms.size(); ++row) {
        const auto id = LoadLittleEndianU32(topology.codes.data() + row * 4U);
        if (id >= out.norms_by_id_.size()) {
          if (requested_rows != 0U)
            return Status::DataLoss(
                "legacy original norm stable ID exceeds base slice");
          out.norms_by_id_.resize(static_cast<std::size_t>(id) + 1U,
                                  std::numeric_limits<float>::quiet_NaN());
        }
        out.norms_by_id_[id] = norms[row];
      }
    }
    rows = out.norms_by_id_.size();
    // A sparse legacy forest need only cover indexed IDs; unused entries are
    // never read. Persist zero placeholders for those entries.
    for (auto& norm : out.norms_by_id_)
      if (std::isnan(norm)) norm = 0.0F;
    for (const auto& shard : legacy_manifest.shards)
      if (shard.kind == ShardKind::kFloatSourceNorm2) {
        identity = CanonicalShardIdentity(legacy_index, legacy_manifest, shard);
      }
  } else {
    if (!base_available)
      return Status::FailedPrecondition(
          "scaled evaluation requires original-vector norms or "
          "dataset.base_path");
    auto opened =
        Open(base, format, first_row, rows, manifest.shape.dimension, parallel);
    if (!opened.ok()) return opened.status();
    out.norms_by_id_.resize(rows);
    std::vector<std::uint32_t> ids;
    std::vector<float> norms;
    const auto batch_rows = BatchRows(manifest.shape.dimension);
    for (std::uint64_t first = 0U; first < rows; first += batch_rows) {
      const auto count = std::min<std::uint64_t>(batch_rows, rows - first);
      ids.resize(count);
      std::iota(ids.begin(), ids.end(), static_cast<std::uint32_t>(first));
      if (const auto status = opened.value().Read(ids, &norms, parallel); !status.ok())
        return status;
      std::copy(norms.begin(), norms.end(), out.norms_by_id_.begin() + first);
    }
    auto resolved = make_identity(rows);
    if (!resolved.ok()) return resolved.status();
    identity = std::move(resolved).value();
  }
  // When a base is available, its identity also binds legacy norm reuse.
  if (base_available) {
    auto resolved = make_identity(rows);
    if (!resolved.ok()) return resolved.status();
    identity = std::move(resolved).value();
  }
  if (identity.empty())
    identity = app::Sha256Bytes("original-vector-owner-v2\n" +
        manifest.hashes.linkage_list_directory.ToHex() + "\n" +
        std::to_string(first_row) + "\n" + std::to_string(rows) + "\n" +
        std::to_string(manifest.shape.dimension));
  out.identity_ = app::Sha256Bytes("original-vector-norm-generation-v2\n" +
      identity + "\n" + manifest.hashes.linkage_list_directory.ToHex());
  const double prepare_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - prepare_begin).count();
  const auto write_begin = std::chrono::steady_clock::now();
  {
    cache = new_cache;
    std::filesystem::create_directories(cache.parent_path(), error);
    if (error)
      return Status::Internal("cannot create constant norm2 cache: " +
                              error.message());
    std::array<std::uint8_t, header_bytes> header{};
    std::memcpy(header.data(), "STLQORN1", 8U);
    StoreLittleEndianU32(2U, header.data() + 8U);
    StoreLittleEndianU32(header_bytes, header.data() + 12U);
    StoreLittleEndianU32(manifest.shape.dimension, header.data() + 16U);
    StoreLittleEndianU32(
        !base_available ? 1U : 0U,
        header.data() + 20U);
    StoreLittleEndianU64(rows, header.data() + 24U);
    StoreLittleEndianU64(first_row, header.data() + 32U);
    StoreLittleEndianU64(manifest.hashes.linkage_list_directory.value(),
                         header.data() + 40U);
    std::memcpy(header.data() + 48U, identity.data(), 64U);
    std::memcpy(header.data() + 112U, out.identity_.data(), 64U);
    auto stage = StagingDirectory::Create(cache);
    if (!stage.ok()) return stage.status();
    const auto pending = stage.value().path() / cache.filename();
    auto writer = SequentialFileWriter::Create(pending);
    if (!writer.ok()) return writer.status();
    if (auto status = writer.value().Append(header); !status.ok()) return status;
    if constexpr (std::endian::native == std::endian::little) {
      if (auto status = writer.value().Append({
              reinterpret_cast<const std::uint8_t*>(out.norms_by_id_.data()),
              out.norms_by_id_.size() * sizeof(float)}); !status.ok())
        return status;
    } else {
    std::array<std::uint8_t, kRowsPerBatch * 4U> bytes;
    for (std::size_t first = 0; first < rows; first += kRowsPerBatch) {
      const auto count = std::min<std::size_t>(kRowsPerBatch, rows - first);
      for (std::size_t i = 0; i < count; ++i)
        StoreLittleEndianF32(out.norms_by_id_[first + i], bytes.data() + i * 4U);
      if (auto status = writer.value().Append({bytes.data(), count * 4U});
          !status.ok()) return status;
    }
    }
    if (const auto status = writer.value().Flush(); !status.ok()) return status;
    if (const auto status = writer.value().Close(); !status.ok()) return status;
#if defined(_WIN32)
    if (!MoveFileExW(pending.c_str(), cache.c_str(), MOVEFILE_WRITE_THROUGH))
      return Status::FailedPrecondition(
          "cannot publish original-vector norm cache without replacing a file");
#elif defined(__linux__)
    if (syscall(SYS_renameat2, AT_FDCWD, pending.c_str(), AT_FDCWD,
                cache.c_str(), RENAME_NOREPLACE) != 0)
      return Status::FailedPrecondition(
          "cannot publish original-vector norm cache without replacing a file");
#else
    return Status::Unimplemented(
        "create-only original-vector norm cache publication requires Linux");
#endif
  }
  LogInfo("Original-vector norm cache wall time(s): " +
          std::to_string(std::chrono::duration<double>(
              std::chrono::steady_clock::now() - prepare_begin).count()) +
          " (reorder/compute: " + std::to_string(prepare_seconds) +
          "; write/flush: " + std::to_string(std::chrono::duration<double>(
              std::chrono::steady_clock::now() - write_begin).count()) + ")");
  out.cache_path_ = cache;
  std::filesystem::remove(pending_norms, error);
  if (error) return Status::Internal("cannot remove consumed source norm stream: " +
                                     error.message());
  return out;
}

}  // namespace stlqpp
