#include "stlqpp/application/eval_norm_cache.h"
#include "stlqpp/application/compute_backend.h"

#include "stlqpp/application/sha256.h"
#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/index/norm/reconstruction_norm8.h"
#include "stlqpp/index/norm/runtime_reconstruction_norms.h"
#include "stlqpp/index/norm/reconstruction_norms.h"
#include "stlqpp/runtime/cpu/parallel_sort.h"
#if STLQPP_HAS_CUDA_NORM
#include "stlqpp/index/norm/reconstruction_norm8_cuda.h"
#include "stlqpp/index/norm/reconstruction_norms_cuda.h"
#endif
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>

#include "stlqpp/model/scaled_score8_codebook.h"
#include "stlqpp/model/topology/parent_louds.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_model_codec.h"
#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace stlqpp {
namespace {

Status Write(const std::filesystem::path& path, const void* data,
             std::size_t size) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(static_cast<const char*>(data),
             static_cast<std::streamsize>(size));
  file.close();
  return file ? Status::Ok()
              : Status::Internal("cannot write norm cache: " + path.string());
}

void AppendF32(std::vector<std::uint8_t>& bytes, float value) {
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(float));
  StoreLittleEndianF32(value, bytes.data() + offset);
}

struct DecodedNormCluster {
  RuntimeDecodedCanonicalCluster cluster;
  std::vector<std::uint32_t> parents;
  std::vector<float> floating;
  RuntimeCanonicalClusterCodeView codes(std::uint32_t cid) const {
    const auto& t = cluster.topology;
    return {cid, t.virtual_count, t.real_root_count,
            static_cast<std::uint32_t>(t.real_ids.size()) - t.real_root_count,
            t.virtual_codes_small, t.real_codes_small, t.edge_code0};
  }
  RuntimeReconstructionNormInput quantized(std::uint32_t cid) const {
    const auto& c = cluster.coefficients;
    return {codes(cid), parents, c.coefficients_layer_major,
            c.root_scales, c.linkage_scales};
  }
  RuntimeFloatReconstructionNormInput native(std::uint32_t cid) const {
    return {codes(cid), parents, floating};
  }

};

// Decoded work is bounded by both worker count and a row budget. Results are
// consumed in CID order, preserving the reference-mean accumulation order.
template <class Function>
Status ForNormClusterWindows(const CanonicalIndexFiles& files,
                             const ParallelForExecutor& parallel,
                             Function&& function) {
  const auto nlist = files.manifest().shape.nlist;
  for (std::uint32_t first = 0U; first < nlist;) {
    auto end = first;
    std::uint64_t rows = 0U;
    while (end < nlist && end - first < parallel.worker_count()) {
      const auto& entry = files.entry(end);
      const auto count = std::uint64_t(entry.real_count) + entry.virtual_count;
      if (end != first && rows + count > 1048576U) break;
      rows += count;
      ++end;
    }
    if (auto status = function(first, end); !status.ok()) return status;
    first = end;
  }
  return Status::Ok();
}
StatusOr<DecodedNormCluster> DecodeNormCluster(
    const CanonicalIndexFiles& files, bool native, std::uint32_t cid) {
  const auto shape = files.manifest().shape;
  const auto& e = files.entry(cid);
  DecodedNormCluster out;
  if (native) {
    const auto records = files.topology_records(cid);
    auto topology = DecodeCanonicalNativeBfsTopologyRuntime(
        shape.m, cid, e.real_count, e.virtual_count, e.real_root_count,
        records.codes, records.parents);
    if (!topology.ok()) return topology.status();
    out.cluster.topology = std::move(topology).value();
    out.floating.resize((std::size_t(e.real_count) + e.virtual_count) * shape.m);
    if (const auto status = files.ReadFloatCoefficients(cid, out.floating);
        !status.ok()) return status;
  } else {
    auto decoded = DecodeCanonicalNativeBfsClusterRuntime(
        shape.m, cid, e.real_count, e.virtual_count, e.real_root_count,
        files.cluster_records(cid), {}, false);
    if (!decoded.ok()) return decoded.status();
    out.cluster = std::move(decoded).value();
  }
  out.parents.resize(e.real_count, 0U);
  ParentLoudsSequentialDecoder decoder(out.cluster.topology.parent_louds_words);
  decoder.InitializeAfterValidatedRootPrefix(e.virtual_count + e.real_root_count);
  for (std::uint32_t row = e.real_root_count; row < e.real_count; ++row)
    out.parents[row] = decoder.NextParent1Based();
  return out;
}

struct Population {
  std::vector<float> norm2, original;
  // Reference-only preparation state; none of these per-row arrays persist.
  std::vector<float> origin_norm2, reference;
  std::vector<double> projection;
};
StatusOr<Population> Reconstruct(const EvalNormCacheRequest& r,
                                 const CanonicalIndexFiles& files,
                                 bool prepare_reference_mean = false) {
  InlineParallelFor serial;
  const auto& parallel = r.parallel ? *r.parallel : serial;
  const auto shape = files.manifest().shape;
  auto model = CanonicalModelCodec::DecodeRuntime(files.model_bytes().data(),
                                                  files.model_bytes().size());
  if (!model.ok()) return model.status();
  const auto reconstructed_shape = model.value().shape();
  if (reconstructed_shape.dimension != shape.dimension ||
      reconstructed_shape.m != shape.m ||
      reconstructed_shape.nlist != shape.nlist ||
      reconstructed_shape.residual_codebook_size !=
          shape.residual_codebook_size)
    return Status::DataLoss(
        "norm reconstruction model shape disagrees with manifest");
  std::optional<RuntimeGramReconstructionNormComputer> cpu;
  if (!r.use_cuda) {
    auto created = RuntimeGramReconstructionNormComputer::Create(model.value().codebooks_view());
    if (!created.ok()) return created.status();
    cpu.emplace(std::move(created).value());
  }
  std::optional<RuntimeReconstructionMeanAccumulator> mean;
  if (prepare_reference_mean) mean.emplace(model.value().codebooks_view());
#if STLQPP_HAS_CUDA_NORM
  std::unique_ptr<RuntimeCudaReconstructionNormComputer> gpu;
  if (r.use_cuda && files.manifest().counts.real != 0U) {
    auto created = RuntimeCudaReconstructionNormComputer::Create(
        model.value().codebooks_view(), r.cuda_device);
    if (!created.ok()) return created.status();
    gpu = std::move(created).value();
  }
#else
  if (r.use_cuda)
    return Status::Unimplemented("GPU norm preparation requires CUDA build");
#endif
  if (files.manifest().counts.real > std::numeric_limits<std::size_t>::max())
    return Status::ResourceExhausted("norm population exceeds host domain");
  std::vector<float> all, source_norms;
  all.reserve(static_cast<std::size_t>(files.manifest().counts.real));
  if (r.scaled) source_norms.resize(static_cast<std::size_t>(files.manifest().counts.real));
  std::vector<std::vector<std::uint32_t>> source_ids;
  if (r.scaled && r.source) source_ids.reserve(parallel.worker_count());
  std::size_t source_offset = 0U, queued_rows = 0U;
  const auto flush_source = [&]() -> Status {
    if (source_ids.empty()) return Status::Ok();
    std::vector<SourceNormRead> reads;
    reads.reserve(source_ids.size());
    auto offset = source_offset;
    for (const auto& ids : source_ids) {
      reads.push_back({ids, {source_norms.data() + offset, ids.size()}});
      offset += ids.size();
    }
    if (auto status = r.source->ReadBatch(reads, parallel); !status.ok()) return status;
    source_offset = offset;
    queued_rows = 0U;
    source_ids.clear();
    return Status::Ok();
  };
  const auto queue_source = [&](std::uint32_t cid, std::vector<std::uint32_t>& ids) -> Status {
    if (!r.scaled) return Status::Ok();
    if (!r.source) {
      auto status = files.ReadFloatSourceNorm2(cid, {source_norms.data() + source_offset, ids.size()});
      source_offset += ids.size();
      return status;
    }
    if (source_ids.size() == parallel.worker_count() ||
        (!source_ids.empty() && queued_rows + ids.size() > 1048576U))
      if (auto status = flush_source(); !status.ok()) return status;
    queued_rows += ids.size();
    source_ids.push_back(std::move(ids));
    return Status::Ok();
  };

  if (!r.use_cuda && r.parallel != nullptr && r.parallel->worker_count() > 1U) {
    const auto status = ForNormClusterWindows(files, *r.parallel,
        [&](std::uint32_t first, std::uint32_t end) -> Status {
      struct Slot {
        Status status;
        std::optional<DecodedNormCluster> cluster;
        std::vector<float> norms;
      };
      std::vector<Slot> slots(end - first);
      struct Work {
        const CanonicalIndexFiles* files;
        const RuntimeGramReconstructionNormComputer* cpu;
        bool native;
        std::uint32_t first;
        std::vector<Slot>* slots;
      } work{&files, &*cpu, r.native_float, first, &slots};
      r.parallel->Run(end - first, &work,
          [](void* opaque, std::uint32_t, std::uint32_t item) noexcept {
        auto& w = *static_cast<Work*>(opaque);
        const auto cid = w.first + item;
        if (w.files->entry(cid).real_count == 0U) return;
        auto& slot = (*w.slots)[item];
        auto decoded = DecodeNormCluster(*w.files, w.native, cid);
        if (!decoded.ok()) { slot.status = decoded.status(); return; }
        slot.cluster = std::move(decoded).value();
        auto norms = w.native ? w.cpu->ComputeFloat(slot.cluster->native(cid))
                              : w.cpu->Compute(slot.cluster->quantized(cid));
        if (!norms.ok()) { slot.status = norms.status(); return; }
        slot.norms = std::move(norms).value();
      });
      for (std::uint32_t item = 0U; item < end - first; ++item) {
        auto& slot = slots[item];
        if (!slot.status.ok()) return slot.status;
        if (!slot.cluster) continue;
        const auto cid = first + item;
        if (mean) {
          const auto accumulated = r.native_float
              ? mean->AccumulateFloat(slot.cluster->native(cid))
              : mean->Accumulate(slot.cluster->quantized(cid));
          if (!accumulated.ok()) return accumulated;
        }
        all.insert(all.end(), slot.norms.begin(), slot.norms.end());
        if (auto queued = queue_source(cid, slot.cluster->cluster.topology.real_ids);
            !queued.ok()) return queued;
      }
      return Status::Ok();
    });
    if (!status.ok()) return status;
    if (auto flushed = flush_source(); !flushed.ok()) return flushed;
    Population out{std::move(all), std::move(source_norms), {}, {}, {}};
    if (mean) {
      auto reference = mean->Finish();
      if (!reference.ok()) return reference.status();
      out.reference = std::move(reference).value();
    }
    return out;
  }
  for (std::uint32_t cid = 0; cid < shape.nlist; ++cid) {
    const auto& e = files.entry(cid);
    if (e.real_count == 0U) continue;
    auto decoded = DecodeNormCluster(files, r.native_float, cid);
    if (!decoded.ok()) return decoded.status();
    auto& cluster = decoded.value();
    if (mean) {
      const auto status = r.native_float ? mean->AccumulateFloat(cluster.native(cid))
                                        : mean->Accumulate(cluster.quantized(cid));
      if (!status.ok()) return status;
    }
    auto norms = [&]() -> StatusOr<std::vector<float>> {
      if (r.native_float) {
#if STLQPP_HAS_CUDA_NORM
        if (gpu) return gpu->ComputeFloat(cluster.native(cid));
#endif
        return cpu->ComputeFloat(cluster.native(cid));
      }
#if STLQPP_HAS_CUDA_NORM
      if (gpu) return gpu->Compute(cluster.quantized(cid));
#endif
      return cpu->Compute(cluster.quantized(cid));
    }();
    if (!norms.ok()) return norms.status();
    all.insert(all.end(), norms.value().begin(), norms.value().end());
    if (auto status = queue_source(cid, cluster.cluster.topology.real_ids);
        !status.ok()) return status;
  }

  if (auto flushed = flush_source(); !flushed.ok()) return flushed;
  Population out{std::move(all), std::move(source_norms), {}, {}, {}};
  if (mean) {
    auto reference = mean->Finish();
    if (!reference.ok()) return reference.status();
    out.reference = std::move(reference).value();
  }
  return out;
}

StatusOr<std::vector<double>> ReferenceProjections(
    const CanonicalIndexFiles& files, bool native,
    ArrayView<const float> reference, const ParallelForExecutor* executor) {
  auto model = CanonicalModelCodec::DecodeRuntime(files.model_bytes().data(),
                                                  files.model_bytes().size());
  if (!model.ok()) return model.status();
  auto projector = RuntimeReconstructionProjectionComputer::Create(
      model.value().codebooks_view(), reference);
  if (!projector.ok()) return projector.status();
  std::vector<double> all;
  all.reserve(files.manifest().counts.real);
  InlineParallelFor serial;
  const auto& parallel = executor ? *executor : serial;
  const auto status = ForNormClusterWindows(files, parallel,
      [&](std::uint32_t first, std::uint32_t end) -> Status {
    struct Slot { Status status; std::vector<double> values; };
    std::vector<Slot> slots(end - first);
    struct Work {
      const CanonicalIndexFiles* files;
      const RuntimeReconstructionProjectionComputer* projector;
      bool native;
      std::uint32_t first;
      std::vector<Slot>* slots;
    } work{&files, &projector.value(), native, first, &slots};
    parallel.Run(end - first, &work,
        [](void* opaque, std::uint32_t, std::uint32_t item) noexcept {
      auto& w = *static_cast<Work*>(opaque);
      const auto cid = w.first + item;
      auto& slot = (*w.slots)[item];
      auto cluster = DecodeNormCluster(*w.files, w.native, cid);
      if (!cluster.ok()) { slot.status = cluster.status(); return; }
      auto values = w.native ? w.projector->ComputeFloat(cluster.value().native(cid))
                             : w.projector->Compute(cluster.value().quantized(cid));
      if (!values.ok()) { slot.status = values.status(); return; }
      slot.values = std::move(values).value();
    });
    for (auto& slot : slots) {
      if (!slot.status.ok()) return slot.status;
      all.insert(all.end(), slot.values.begin(), slot.values.end());
    }
    return Status::Ok();
  });
  if (!status.ok()) return status;
  return all;
}

StatusOr<Population> ReconstructReference(
    const EvalNormCacheRequest& r, const CanonicalIndexFiles& files,
    ArrayView<const float> cached_reference) {
  auto origin = Reconstruct(r, files, cached_reference.empty());
  if (!origin.ok()) return origin.status();
  std::vector<float> reference(cached_reference.begin(), cached_reference.end());
  if (reference.empty()) reference = origin.value().reference;
  auto projection = ReferenceProjections(files, r.native_float, reference, r.parallel);
  if (!projection.ok()) return projection.status();
  auto out = std::move(origin).value();
  out.origin_norm2 = std::move(out.norm2);
  out.reference = std::move(reference);
  out.projection = std::move(projection).value();
  double reference_norm2 = 0.0;
  for (const auto value : out.reference) reference_norm2 += double(value)*value;
  out.norm2.resize(out.origin_norm2.size());
  for (std::size_t row = 0; row < out.norm2.size(); ++row) {
    const double value = double(out.origin_norm2[row]) -
        2.0*out.projection[row] + reference_norm2;
    if (!std::isfinite(value))
      return Status::DataLoss("reference residual norm overflow");
    // FP32 origin norms can cancel just below zero in the expanded formula.
    out.norm2[row] = static_cast<float>(std::max(0.0, value));
  }
  return out;
}

// Log-domain Lloyd determines labels; decoded norm2 centers are arithmetic
// means in the original residual-norm domain. Ties choose the lower label.
StatusOr<std::pair<std::array<float, 256>, std::vector<std::uint8_t>>>
FitReferenceNorms(const std::vector<float>& norms, std::uint32_t iterations,
                  const ParallelForExecutor& parallel) {
  std::array<float, 256> centers{};
  std::vector<std::uint8_t> labels(norms.size(), 0U);
  if (norms.empty()) return std::make_pair(centers, std::move(labels));
  std::vector<double> values(norms.begin(), norms.end());
  ParallelSort(values, parallel);
  const auto n = values.size();
  const double median = n % 2 ? values[n/2] : (values[n/2-1]+values[n/2])*0.5;
  const double epsilon = std::max(median * 1.0e-6, 1.0e-12);
  std::vector<double> prefix(n+1, 0.0);
  struct Logs { std::vector<double>* values; double epsilon; } logs{&values, epsilon};
  parallel.Run(static_cast<std::uint32_t>((n + 4095U) / 4096U), &logs,
      [](void* opaque, std::uint32_t, std::uint32_t chunk) noexcept {
    auto& w = *static_cast<Logs*>(opaque);
    const auto end = std::min(w.values->size(), (std::size_t(chunk) + 1U) * 4096U);
    for (std::size_t i = std::size_t(chunk) * 4096U; i < end; ++i)
      (*w.values)[i] = std::log((*w.values)[i] + w.epsilon);
  });
  // Keep this inexpensive prefix scan in its original arithmetic order.
  for (std::size_t i = 0; i < n; ++i) prefix[i+1] = prefix[i] + values[i];
  std::array<double, 256> log_centers{};
  for (unsigned l = 0; l < 256; ++l)
    log_centers[l] = static_cast<float>(values[std::size_t(l)*n/256]);
  for (unsigned it = 0; it < iterations; ++it) {
    std::size_t begin = 0;
    double change = 0.0;
    const auto old = log_centers;
    for (unsigned l = 0; l < 256; ++l) {
      const auto end = l == 255 ? n : std::size_t(
          std::upper_bound(values.begin(), values.end(),
              (old[l] + old[l+1])*0.5) - values.begin());
      if (end != begin) log_centers[l] = static_cast<float>(
          (prefix[end] - prefix[begin]) / static_cast<double>(end-begin));
      change = std::max(change, std::abs(log_centers[l] - old[l]));
      begin = end;
    }
    if (change <= 1.0e-6) break;
  }
  std::array<double, 255> boundaries{};
  for (unsigned l = 0; l < 255; ++l)
    boundaries[l] = (log_centers[l]+log_centers[l+1])*0.5;
  std::array<double, 256> sums{};
  std::array<std::uint64_t, 256> counts{};
  struct Labels {
    const std::vector<float>* norms;
    const std::array<double, 255>* boundaries;
    std::vector<std::uint8_t>* labels;
    double epsilon;
  } label_work{&norms, &boundaries, &labels, epsilon};
  parallel.Run(static_cast<std::uint32_t>((n + 4095U) / 4096U), &label_work,
      [](void* opaque, std::uint32_t, std::uint32_t chunk) noexcept {
    auto& w = *static_cast<Labels*>(opaque);
    const auto end = std::min(w.norms->size(), (std::size_t(chunk) + 1U) * 4096U);
    for (std::size_t i = std::size_t(chunk) * 4096U; i < end; ++i)
      (*w.labels)[i] = static_cast<std::uint8_t>(std::lower_bound(
          w.boundaries->begin(), w.boundaries->end(),
          std::log(double((*w.norms)[i]) + w.epsilon)) - w.boundaries->begin());
  });
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const auto l = labels[i];
    sums[l] += norms[i];
    ++counts[l];
  }
  for (unsigned l = 0; l < 256; ++l)
    centers[l] = counts[l] ? static_cast<float>(sums[l]/static_cast<double>(counts[l])) : 0.0F;
  return std::make_pair(centers, std::move(labels));
}

StatusOr<std::pair<std::string, std::string>> Identity(
    const EvalNormCacheRequest& r, const CanonicalIndexFiles& files) {
  const auto& manifest = files.manifest();
  std::ostringstream text;
  text << "reconstruction-norm2-v4\ncoefficient_mode="
       << (r.native_float ? "float" : "quantized")
       << "\nnorm_mode=" << (r.float_norm ? "exact" : "lut")
       << "\nm=" << manifest.shape.m
       << "\ndimension=" << manifest.shape.dimension
       << "\nnlist=" << manifest.shape.nlist
       << "\nreal_rows=" << manifest.counts.real
       << "\nforest_hash=" << manifest.hashes.linkage_list_directory.ToHex()
       << "\nbackend=" << (r.use_cuda ? "CUDA" : "CPU")
       << "\ncuda_device=" << (r.use_cuda ? r.cuda_device : 0) << '\n';
  if (!r.float_norm)
    text << "lut_mode="
         << ReconstructionNorm8ModeName(r.mode)
         << "\nlut_kmeans_niter=" << r.iterations << '\n';
  if (r.mode == ReconstructionNorm8Mode::kReferenceLloyd) {
    text << "reference=global_reconstruction_mean\n"
            "reference_domain=model_coordinates\n"
            "reference_mean=float64_descendant_coefficient_reduction\n"
            "residual_norm=origin_norm2_minus_2_projection_plus_reference_norm2\n"
            "binning=log\nlog_epsilon=max(median_residual_norm2*1e-6,1e-12)\n"
            "decode_centers=arithmetic_residual_norm2_mean\n"
            "reference_mean_backend=CPU\n"
            "log_fit_backend=CPU\n"
            "projection=float64_scalar_forest_dp\n";

  }
  text << "origin_norm_method=" << (r.use_cuda ? "CUDA_gram_ancestor" : "CPU_gram_ancestor") << '\n';
  if (!r.native_float)
    text << "coefficient_codec_hash="
         << manifest.hashes.coefficient_codec.ToHex() << '\n';
  if (manifest.producer_recipe && !r.native_float) {
    const auto widths = [&](const char* name, const auto& values) {
      text << name << "=[";
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) text << ',';
        text << values[i];
      }
      text << "]\n";
    };
    widths("bits_per_layer",
           manifest.producer_recipe->coefficient_bits_per_layer);
    widths("bits_per_layer_root",
           manifest.producer_recipe->coefficient_bits_per_layer_root);
    widths("bits_per_layer_linkage",
           manifest.producer_recipe->coefficient_bits_per_layer_linkage);
  }
  for (const auto& shard : manifest.shards) {
    const bool coefficient =
        shard.kind == (r.native_float ? ShardKind::kFloatCoefficients
                                      : ShardKind::kCoefficients);
    if (coefficient || shard.kind == ShardKind::kCodebooks ||
        shard.kind == ShardKind::kCodes || shard.kind == ShardKind::kParents) {
      text << "source_shard_" << static_cast<unsigned>(shard.kind) << "="
           << CanonicalShardIdentity(r.index, manifest, shard) << '\n';
    }
  }
  // The readable archive stops at the shared recipe/source metadata. Keep
  // CID counts in the identity stream to preserve existing cache identities
  // and layout validation without publishing a per-cluster listing.
  const auto config = text.str();
  for (unsigned cid = 0; cid < manifest.shape.nlist; ++cid) {
    const auto& e = files.entry(cid);
    text << "cid_" << cid << "=" << e.real_count << ',' << e.virtual_count
         << ',' << e.real_root_count << '\n';
  }
  return std::make_pair(app::Sha256Bytes(text.str()), config);
}

StatusOr<std::vector<std::uint8_t>> ReadPayload(
    const std::filesystem::path& directory, const std::string& identity,
    const std::vector<std::string>& names,
    const char* metadata_name = "identity.txt") {
  std::ifstream metadata(directory / metadata_name);
  std::string stored, name, checksum, extra;
  if (!(metadata >> stored) || stored != identity)
    return Status::DataLoss(
        "norm2 cache identity mismatch; cache was not overwritten");
  std::vector<std::uint8_t> result;
  for (const auto& expected : names) {
    if (!(metadata >> name >> checksum) || name != expected)
      return Status::DataLoss("invalid norm2 cache metadata");
    // Legacy 64-character checksums are opaque metadata. New publications
    // register extents, and no cache hit reads a payload twice to authenticate it.
    if (checksum.rfind("bytes=", 0U) != 0U && checksum.size() != 64U)
      return Status::DataLoss("invalid legacy norm2 cache token");
    auto file = ReadOnlyMappedFile::Open(directory / name);
    if (!file.ok()) return file.status();
    if (checksum.rfind("bytes=", 0U) == 0U) {
      std::uint64_t bytes = 0U;
      const auto parsed = std::from_chars(checksum.data() + 6U,
                                          checksum.data() + checksum.size(), bytes);
      if (parsed.ec != std::errc{} || parsed.ptr != checksum.data() + checksum.size() ||
          bytes != file.value().size())
        return Status::DataLoss("norm2 cache payload extent mismatch");
    }
    result.insert(result.end(), file.value().data(),
                  file.value().data() + file.value().size());
  }
  if (metadata >> extra)
    return Status::DataLoss("unexpected norm2 cache metadata");
  return result;
}

// Reuse complete historical generations, preserving their original identity.
// v3 used opaque payload SHA metadata; v4 binds lightweight source extents.
std::string LegacyNormIdentity(const std::filesystem::path& directory,
                               const char* config_name, const char* identity_name,
                               const std::string& config,
                               const CanonicalIndexFiles& files) {
  std::ifstream input(directory / config_name);
  if (!input) return {};
  const std::string old((std::istreambuf_iterator<char>(input)), {});
  const bool v3 = old.starts_with("reconstruction-norm2-v3\n");
  if (!v3 && !old.starts_with("reconstruction-norm2-v4\n")) return {};
  const auto recipe = [v3](const std::string& text) {
    std::istringstream lines(text);
    std::string line, result;
    while (std::getline(lines, line)) {
      if (line.starts_with("reconstruction-norm2-v") ||
          (v3 && line.starts_with("source_shard_"))) continue;
      // Only the known mathematically equivalent implementations may share
      // an existing population. All other recipe/source fields stay strict.
      if (line == "origin_norm_method=CPU_gram_ancestor" ||
          line == "origin_norm_method=CUDA_gram_ancestor" ||
          line == "origin_norm_method=CPU_selected_atom_gram_fp32_gemm1" ||
          line == "origin_norm_method=CPU_dense_reconstruction" ||
          line == "origin_norm_method=CUDA_dense_reconstruction") continue;
      result += line + "\n";
    }
    return result;
  };
  if (recipe(old) != recipe(config)) return {};
  std::ostringstream identity;
  identity << old;
  for (unsigned cid = 0; cid < files.manifest().shape.nlist; ++cid) {
    const auto& entry = files.entry(cid);
    identity << "cid_" << cid << "=" << entry.real_count << ','
             << entry.virtual_count << ',' << entry.real_root_count << '\n';
  }
  std::ifstream metadata(directory / identity_name);
  std::string stored;
  if (!(metadata >> stored) || stored != app::Sha256Bytes(identity.str())) return {};
  return stored;
}

// Incomplete generations are never overwritten; a replacement is a sibling.
StatusOr<std::pair<std::filesystem::path, bool>> Destination(
    const std::filesystem::path& primary,
    const std::vector<std::string>& names) {
  std::error_code error;
  auto directory = primary;
  for (unsigned generation = 0;; ++generation) {
    if (!std::filesystem::exists(directory, error)) {
      if (error)
        return Status::Internal("cannot inspect norm2 cache: " +
                                error.message());
      return std::make_pair(directory, false);
    }
    bool complete = std::filesystem::exists(directory / "identity.txt", error);
    for (const auto& name : names)
      complete &= std::filesystem::exists(directory / name, error);
    if (error)
      return Status::Internal("cannot inspect norm2 cache payload: " +
                              error.message());
    if (complete) return std::make_pair(directory, true);
    directory =
        primary.parent_path() / (primary.filename().string() + "-rebuild-" +
                                 std::to_string(generation + 1));
  }
}

Status Publish(
    const std::filesystem::path& directory, const std::string& identity,
    const std::string& config,
    const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>&
        files) {
  std::error_code error;
  std::filesystem::create_directories(directory.parent_path(), error);
  if (error)
    return Status::Internal("cannot create norm2 cache: " + error.message());
  auto stage = StagingDirectory::Create(directory);
  if (!stage.ok()) return stage.status();
  std::string metadata = identity + "\n";
  for (const auto& file : files) {
    if (auto status = Write(stage.value().path() / file.first,
                            file.second.data(), file.second.size());
        !status.ok())
      return status;
    metadata += file.first + " bytes=" + std::to_string(file.second.size()) + "\n";
  }
  if (auto status = Write(stage.value().path() / "identity.txt",
                          metadata.data(), metadata.size());
      !status.ok())
    return status;
  if (auto status = Write(stage.value().path() / "config.txt", config.data(),
                          config.size());
      !status.ok())
    return status;
  return stage.value().Publish();
}

// The primary float exact norm is a single shared file alongside original
// norms. Metadata is committed last. Incomplete/other-backend versions use an
// immutable profile directory instead of replacing any already published file.
Status PublishFloatExact(const std::filesystem::path& directory,
                         const std::string& identity, const std::string& config,
                         const std::vector<std::uint8_t>& norm) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return Status::Internal("cannot create constant norm2 cache: " +
                            error.message());
  auto staged = StagingDirectory::Create(directory / "float-exact-publication");
  if (!staged.ok()) return staged.status();
  const std::string metadata = identity + "\nexact_float_norm2.f32 bytes=" +
      std::to_string(norm.size()) + "\n";
  for (const auto& file :
       std::vector<std::pair<std::string, std::vector<std::uint8_t>>>{
           {"exact_float_norm2.f32", norm},
           {"exact_float_config.txt", {config.begin(), config.end()}},
           {"exact_float_identity.txt", {metadata.begin(), metadata.end()}}}) {
    auto status = Write(staged.value().path() / file.first, file.second.data(),
                        file.second.size());
    if (!status.ok()) return status;
#if defined(_WIN32)
    if (!MoveFileExW((staged.value().path() / file.first).c_str(),
                     (directory / file.first).c_str(), MOVEFILE_WRITE_THROUGH))
      return Status::FailedPrecondition(
          "cannot publish constant float norm2 without replacing a file: " +
          std::to_string(GetLastError()));
#elif defined(__linux__)
    if (syscall(SYS_renameat2, AT_FDCWD,
                (staged.value().path() / file.first).c_str(), AT_FDCWD,
                (directory / file.first).c_str(), RENAME_NOREPLACE) != 0)
      return Status::FailedPrecondition(
          "cannot publish constant float norm2 without replacing a file");
#else
    return Status::Unimplemented(
        "constant norm2 publication requires create-only file rename");
#endif
  }
  return Status::Ok();
}

}  // namespace

StatusOr<PreparedEvalNormCache> PrepareEvalNormCache(
    const EvalNormCacheRequest& request) {
  try {
    auto r = request;
    app::ResolveCudaExecutionPolicy(STLQPP_HAS_CUDA_NORM != 0,
                                    r.cuda_compute_allowed, r.use_cuda);
    if (r.use_cuda) {
      if (const Status status = app::ResolveRuntimeCudaExecutionPolicy(
              r.cuda_compute_allowed, r.use_cuda); !status.ok()) {
        return status;
      }
    }
    if (r.float_norm) {
      r.mode = ReconstructionNorm8Mode::kIdentityLloyd;
      r.iterations = 0U;
    }
    auto opened = CanonicalIndexFiles::OpenCoefficients(r.index, r.native_float);
    if (!opened.ok()) return opened.status();
    const auto& files = opened.value();
    InlineParallelFor serial;
    const auto& parallel = r.parallel == nullptr ? static_cast<const ParallelForExecutor&>(serial) : *r.parallel;
    std::optional<SourceNormProvider> owned_source;
    if (r.scaled && r.source == nullptr) {
      auto source = SourceNormProvider::OpenForIndex(
          r.index, files.manifest(), {}, "auto", 0U, 0U, parallel);
      if (!source.ok()) return source.status();
      owned_source.emplace(std::move(source).value());
      r.source = &*owned_source;
      r.external_source_identity = r.source->identity();
    }
    const bool reference_mode = r.mode == ReconstructionNorm8Mode::kReferenceLloyd;
    std::vector<float> reference;
    const auto rows = files.manifest().counts.real;
    auto id = Identity(r, files);
    if (!id.ok()) return id.status();
    std::string identity = id.value().first;
    std::vector<std::string> norm_names =
        r.float_norm
            ? std::vector<std::string>{r.native_float ? "exact_float_norm2.f32"
                                                      : "norm2.f32"}
            : std::vector<std::string>{"norm2_lut.f32", "norm2_labels.u8"};
    if (reference_mode) norm_names.push_back("reference_vector.f32");
    const auto primary =
        r.cache_root /
        (r.float_norm ? (r.native_float ? "constant" : "exact") : "lut") /
        (std::string(r.native_float ? "float-" : "quantized-") +
         (reference_mode ? "reference-" : "") + identity);
    bool shared_float = false, norm_hit = false;
    std::filesystem::path directory;
    if (r.native_float && r.float_norm) {
      const auto constant = r.cache_root / "constant";
      std::error_code error;
      const bool payload =
          std::filesystem::exists(constant / "exact_float_norm2.f32", error);
      const bool metadata =
          std::filesystem::exists(constant / "exact_float_identity.txt", error);
      if (error)
        return Status::Internal("cannot inspect constant norm2 cache: " +
                                error.message());
      std::string previous;
      if (metadata) {
        std::ifstream file(constant / "exact_float_identity.txt");
        if (!(file >> previous))
          return Status::DataLoss("invalid constant float norm2 identity");
      }
      if (payload && metadata && previous != identity) {
        const auto legacy = LegacyNormIdentity(constant, "exact_float_config.txt",
            "exact_float_identity.txt", id.value().second, files);
        if (!legacy.empty()) identity = legacy;
      }
      if ((!payload && !metadata) ||
          (payload && metadata && previous == identity)) {
        directory = constant;
        shared_float = true;
        norm_hit = payload && metadata;
      }
    }
    if (!shared_float) {
      auto destination = Destination(primary, norm_names);
      if (!destination.ok()) return destination.status();
      directory = destination.value().first;
      norm_hit = destination.value().second;
      if (!norm_hit && std::filesystem::exists(primary.parent_path())) {
        for (const auto& entry : std::filesystem::directory_iterator(primary.parent_path())) {
          if (!entry.is_directory()) continue;
          const auto legacy = LegacyNormIdentity(entry.path(), "config.txt",
              "identity.txt", id.value().second, files);
          if (legacy.empty()) continue;
          bool complete = true;
          for (const auto& name : norm_names)
            complete &= std::filesystem::exists(entry.path() / name);
          if (!complete) continue;
          directory = entry.path();
          identity = legacy;
          norm_hit = true;
          break;
        }
      }
    }
    std::optional<Population> population;
    const auto reconstruct = [&]() -> Status {
      if (population) return Status::Ok();
      auto value = reference_mode ? ReconstructReference(r, files, reference)
                                  : Reconstruct(r, files);
      if (!value.ok()) return value.status();
      population.emplace(std::move(value).value());
      if (reference_mode) reference = population->reference;
      return Status::Ok();
    };
    if (!norm_hit) {
      if (auto status = reconstruct(); !status.ok()) return status;
      std::vector<std::pair<std::string, std::vector<std::uint8_t>>> payload;
      auto& all = population->norm2;
      if (r.float_norm) {
        std::vector<std::uint8_t> norms;
        norms.reserve(all.size() * 4U);
        for (auto value : all) AppendF32(norms, value);
        payload.push_back({norm_names[0], std::move(norms)});
      } else if (reference_mode) {
        auto fitted = FitReferenceNorms(all, r.iterations, parallel);
        if (!fitted.ok()) return fitted.status();
        std::vector<std::uint8_t> centers;
        for (auto value : fitted.value().first) AppendF32(centers, value);
        payload.push_back({norm_names[0], std::move(centers)});
        payload.push_back({norm_names[1], std::move(fitted.value().second)});
        std::vector<std::uint8_t> vector;
        for (auto value : reference) AppendF32(vector, value);
        payload.push_back({norm_names[2], std::move(vector)});
      } else {
        auto fitted = [&]() -> StatusOr<ReconstructionNorm8Codebook> {
          if (all.empty())
            return ReconstructionNorm8Codebook::FromIdentityCenters({});
#if STLQPP_HAS_CUDA_NORM
          if (r.use_cuda) {
            if (r.mode != ReconstructionNorm8Mode::kIdentityLloyd)
              return Status::Unimplemented(
                  "GPU norm8 tail-weighted mode is not supported; select CPU "
                  "or lut");
            return FitIdentityNorm8Cuda(all, r.iterations, r.cuda_device);
          }
#endif
          return ReconstructionNorm8Codebook::FitFromValues(all, r.iterations, parallel, r.mode);
        }();
        if (!fitted.ok()) return fitted.status();
        std::vector<std::uint8_t> labels(all.size());
#if STLQPP_HAS_CUDA_NORM
        if (r.use_cuda && !all.empty()) {
          auto encoder = CudaNorm8LabelEncoder::Create(fitted.value().view(),
                                                       r.cuda_device);
          if (!encoder.ok()) return encoder.status();
          if (const Status status = encoder.value()->Encode(all, labels);
              !status.ok())
            return status;
        } else
#endif
            if (const Status status = fitted.value().Encode(all, labels);
                !status.ok())
          return status;

        std::vector<std::uint8_t> centers;
        for (auto value : fitted.value().centers()) AppendF32(centers, value);
        payload.push_back({norm_names[0], std::move(centers)});
        payload.push_back({norm_names[1], std::move(labels)});
      }
      const auto status =
          shared_float
              ? PublishFloatExact(directory, identity, id.value().second,
                                  payload[0].second)
              : Publish(directory, identity, id.value().second, payload);
      if (!status.ok()) return status;
    }
    auto norm =
        ReadPayload(directory, identity, norm_names,
                    shared_float ? "exact_float_identity.txt" : "identity.txt");
    if (!norm.ok()) return norm.status();
    if (rows > (std::numeric_limits<std::size_t>::max() - 1024U) / 4U ||
        norm.value().size() != (r.float_norm ? rows * 4U : rows + 1024U) +
            (reference_mode ? std::size_t(files.manifest().shape.dimension)*4U : 0U))
      return Status::DataLoss("norm2 cache extent disagrees with index");
    std::vector<double> projection;
    double reference_norm2 = 0.0;
    if (reference_mode) {
      reference.resize(files.manifest().shape.dimension);
      for (std::size_t axis = 0; axis < reference.size(); ++axis) {
        reference[axis] = LoadLittleEndianF32(
            norm.value().data() + 1024U + rows + axis*4U);
        if (!std::isfinite(reference[axis]))
          return Status::DataLoss("invalid norm2 reference vector");
        reference_norm2 += double(reference[axis])*reference[axis];
      }
    }
    const auto prepare_projection = [&]() -> Status {
      if (!projection.empty() || rows == 0U) return Status::Ok();
      if (population) projection = population->projection;
      else {
        auto dots = ReferenceProjections(files, r.native_float, reference, r.parallel);
        if (!dots.ok()) return dots.status();
        projection = std::move(dots).value();
      }
      return Status::Ok();
    };
    if (reference_mode && !r.gain_only) {
      if (auto status = prepare_projection(); !status.ok()) return status;
    }
    std::vector<std::uint8_t> scale;
    bool scale_hit = true;
    std::filesystem::path scale_path;
    std::string active_identity = identity;
    if (r.scaled) {
      if (r.source && r.external_source_identity.empty())
        return Status::InvalidArgument(
            "scale requires original-vector norm identity");
      std::string source_identity = r.external_source_identity;
      const std::string scale_config =
          "norm2-scale-v3\nnorm_identity=" + identity +
          "\noriginal_vector_norm_identity=" + source_identity + "\n";
      active_identity = app::Sha256Bytes(scale_config);
      const std::vector<std::string> scale_names{
          r.float_norm ? "norm2_scale.f32" : "norm2_scale_bias.f32"};
      const auto scale_root = directory / (shared_float ? "float-scale" : "scale");
      std::ifstream old_scale(scale_root / "identity.txt");
      std::string old_identity;
      const bool legacy_scale_hit = (old_scale >> old_identity) && old_identity == active_identity;
      auto sd = Destination(legacy_scale_hit ? scale_root : scale_root / active_identity,
                            scale_names);
      if (!sd.ok()) return sd.status();
      scale_hit = sd.value().second;
      scale_path = sd.value().first / scale_names.front();
      if (!scale_hit) {
        if (r.float_norm) {
          // Exact cached rho already is the scale denominator. Load only
          // original norms by stable ID, without decoding coefficients/parents.
          std::size_t offset = 0U;
          for (unsigned cid = 0; cid < files.manifest().shape.nlist; ++cid) {
            const auto count = files.entry(cid).real_count;
            std::vector<float> original(count);
            if (population) {
              if (count != 0U)
                std::copy_n(population->original.data() + offset, count, original.data());
            } else if (r.source) {
              const auto records = files.topology_records(cid);
              std::vector<std::uint32_t> ids(count);
              for (unsigned i = 0; i < count; ++i)
                ids[i] = LoadLittleEndianU32(records.codes.data() + i * 4U);
              if (auto status = r.source->Read(ids, &original, parallel); !status.ok()) return status;
            } else if (auto status = files.ReadFloatSourceNorm2(cid, original);
                       !status.ok()) return status;
            for (unsigned i = 0; i < count; ++i, ++offset) {
              auto gain = ComputeRadialScoreScale(original[i],
                  LoadLittleEndianF32(norm.value().data() + offset * 4U));
              if (!gain.ok()) return gain.status();
              AppendF32(scale, gain.value());
            }
          }
        } else {
          if (auto status = reconstruct(); !status.ok()) return status;
          if (reference_mode) {
            if (auto status = prepare_projection(); !status.ok()) return status;
          }
          std::array<float, 256> centers{};
          for (unsigned l = 0; l < 256; ++l)
            centers[l] = LoadLittleEndianF32(norm.value().data() + l * 4U);
          auto cb = reference_mode
              ? ReconstructionNorm8Codebook::FromIdentityCenters({})
              : ReconstructionNorm8Codebook::FromIdentityCenters(centers);
          if (!cb.ok()) return cb.status();
          const ArrayView<const std::uint8_t> labels{
              norm.value().data() + 1024U, static_cast<std::size_t>(rows)};
          ScaledScore8Accumulator accumulator;
          if (auto status = accumulator.Accumulate(labels,
                  reference_mode ? population->origin_norm2 : population->norm2,
                  population->original, parallel);
              !status.ok())
            return status;
          auto table = accumulator.Finish(
              reference_mode ? Norm8CodebookView{centers.data(), 256U} : cb.value().view());
          if (!table.ok()) return table.status();
          std::array<double, 256> projection_sum{};
          std::array<std::uint64_t, 256> counts{};
          if (reference_mode)
            for (std::size_t i = 0; i < rows; ++i) {
              projection_sum[labels[i]] += projection[i];
              ++counts[labels[i]];
            }
          for (unsigned l = 0; l < 256; ++l) {
            const auto gain = table.value().scales()[l];
            double bias = table.value().biases()[l];
            if (reference_mode && counts[l])
              bias += reference_norm2 - 2.0*gain*projection_sum[l]/static_cast<double>(counts[l]);
            AppendF32(scale, gain);
            AppendF32(scale, static_cast<float>(bias));
          }
        }
        if (auto status = Publish(sd.value().first, active_identity,
                                  scale_config, {{scale_names[0], scale}});
            !status.ok())
          return status;
      }
      auto decoded =
          ReadPayload(sd.value().first, active_identity, scale_names);
      if (!decoded.ok()) return decoded.status();
      scale = std::move(decoded).value();
      if (scale.size() != (r.float_norm ? rows * 4U : 2048U))
        return Status::DataLoss("norm2 scale extent mismatch");
    }
    PreparedEvalNormCache out;
    out.directory = directory;
    if (!r.scaled) {
      for (const auto& name : norm_names) out.query_payloads.push_back(directory / name);
    } else {
      out.query_payloads.push_back(scale_path);
      if (!r.float_norm) out.query_payloads.push_back(directory / "norm2_labels.u8");
      if (reference_mode && !r.gain_only) out.query_payloads.push_back(directory / "reference_vector.f32");
      // Exact scaled scoring needs the original per-row bias, while LUT
      // scaled scoring carries its bias in the gain/bias table.
      if (r.float_norm && !r.gain_only) out.query_payloads.push_back(r.source->cache_path());
    }
    out.cache_hit = norm_hit && scale_hit;
    std::uint64_t hash = 0;
    const auto parsed = std::from_chars(active_identity.data(),
                                        active_identity.data() + 16, hash, 16);
    if (parsed.ec != std::errc{})
      return Status::Internal("cannot parse norm2 cache hash");
    out.hash = ArtifactHash64::FromValue(hash);
    out.scores.clusters.resize(files.manifest().shape.nlist);
    std::size_t row_offset = 0;
    for (unsigned cid = 0; cid < files.manifest().shape.nlist; ++cid) {
      auto& scores = out.scores.clusters[cid];
      const auto count = files.entry(cid).real_count;
      scores.norm2.resize(count);
      scores.has_reference_correction = reference_mode && !r.gain_only;
      std::vector<float> original;
      if (r.scaled) {
        scores.biases.resize(count);
        scores.dot_multipliers.resize(count);
      }
      if (r.scaled && r.float_norm && !r.gain_only) {
        if (r.source) {
          auto records = files.topology_records(cid);
          std::vector<std::uint32_t> ids(count);
          for (unsigned i = 0; i < count; ++i)
            ids[i] = LoadLittleEndianU32(records.codes.data() + i * 4U);
          if (auto status = r.source->Read(ids, &original, parallel); !status.ok())
            return status;
        } else {
          original.resize(count);
          if (auto status = files.ReadFloatSourceNorm2(cid, original);
              !status.ok())
            return status;
        }
      }
      for (unsigned i = 0; i < count; ++i, ++row_offset) {
        const auto label = r.float_norm ? 0U : norm.value()[1024U + row_offset];
        const auto n = LoadLittleEndianF32(
            norm.value().data() + (r.float_norm ? row_offset : label) * 4U);
        if (!std::isfinite(n) || n < 0)
          return Status::DataLoss("invalid norm2 cache value");
        scores.norm2[i] = r.gain_only ? 0.0F : reference_mode
            ? static_cast<float>(n + 2.0*projection[row_offset] - reference_norm2) : n;
        if (r.scaled) {
          const auto gain = LoadLittleEndianF32(
              scale.data() + (r.float_norm ? row_offset : label * 2U) * 4U);
          const auto bias = r.gain_only ? 0.0F : r.float_norm
                  ? original[i]
                  : LoadLittleEndianF32(scale.data() + (label * 2U + 1U) * 4U);
          if (!std::isfinite(-2.0F * gain) || gain < 0 ||
              !std::isfinite(bias) || (!reference_mode && bias < 0))
            return Status::DataLoss("invalid norm2 scale/bias");
          scores.dot_multipliers[i] = -2.0F * gain;
          scores.biases[i] = !r.gain_only && reference_mode
              ? static_cast<float>(bias + 2.0*gain*projection[row_offset] - reference_norm2) : bias;
        }
      }
    }
    return out;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("norm2 preparation exhausted memory");
  }
}
}  // namespace stlqpp
