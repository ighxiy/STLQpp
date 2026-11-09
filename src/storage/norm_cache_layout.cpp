#include "stlqpp/storage/norm_cache_layout.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "stlqpp/model/codec/coefficient_bit_width.h"
#include "stlqpp/platform/staging_directory.h"
#include "stlqpp/storage/canonical_index_files.h"
#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "stlqpp/application/sha256.h"
namespace stlqpp {
Status WritePublishedNorm2Metadata(const std::filesystem::path& root,
                                   IndexManifest& manifest,
                                   const char* backend) {
  std::ostringstream config, checksums;
  config << "coefficient_mode="
         << (manifest.has_quantized_forest() ? "quantized" : "float")
         << "\nnorm_mode="
         << (manifest.has_quantized_forest() ? "lut" : "exact")
         << "\ncoefficient_codec_hash="
         << manifest.hashes.coefficient_codec.ToHex()
         << "\nnorm_hash=" << manifest.hashes.norm8_cache.ToHex() << '\n';
  if (manifest.producer_recipe) {
    const auto& r = *manifest.producer_recipe;
    config << "lut_mode="
           << (r.norm8_mode == ReconstructionNorm8Mode::kIdentityLloyd
                   ? "identity"
                   : "tail_weighted")
           << "\nlut_kmeans_niter=" << r.norm8_lloyd_iterations << '\n';
    const auto list = [&](const char* name, const auto& values) {
      config << name << "=[";
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) config << ',';
        config << values[i];
      }
      config << "]\n";
    };
    const auto base = NormalizeCoefficientBits(
        r.coefficient_bits_per_layer, manifest.shape.m, "bits_per_layer");
    if (!base.ok()) return base.status();
    const auto root_bits =
        NormalizeCoefficientBits(r.coefficient_bits_per_layer_root.empty()
                                     ? base.value()
                                     : r.coefficient_bits_per_layer_root,
                                 manifest.shape.m, "bits_per_layer_root");
    if (!root_bits.ok()) return root_bits.status();
    const auto linkage =
        NormalizeCoefficientBits(r.coefficient_bits_per_layer_linkage.empty()
                                     ? base.value()
                                     : r.coefficient_bits_per_layer_linkage,
                                 manifest.shape.m, "bits_per_layer_linkage");
    if (!linkage.ok()) return linkage.status();
    list("bits_per_layer", base.value());
    list("bits_per_layer_root", root_bits.value());
    list("bits_per_layer_linkage", linkage.value());
    list("first_magnitude_quantiles", r.coefficient_first_magnitude_quantiles);
    list("rest_magnitude_quantiles", r.coefficient_rest_magnitude_quantiles);
    config << "weighted_quantile=" << r.coefficient_weighted_quantile
           << "\nallow_clipping=" << r.coefficient_allow_clipping
           << "\nfit_reconstruction_scales=" << r.coefficient_fit_scale
           << "\nrefinement_sweeps=" << r.coefficient_refinement_sweeps
           << "\nrefinement_max_layer=" << r.coefficient_refinement_max_layer
           << "\nrefinement_step_limit=" << r.coefficient_refinement_step_limit
           << '\n';
  }
  config << "backend=" << backend << "\nforest_hash="
         << manifest.hashes.linkage_list_directory.ToHex() << '\n';
  bool local = false;
  for (auto& shard : manifest.shards) {
    if (shard.kind != ShardKind::kNorm8 && shard.kind != ShardKind::kNorm2Lut &&
        shard.kind != ShardKind::kScaledScore8 &&
        shard.kind != ShardKind::kFloatNorm2)
      continue;
    if (!shard.reference_path.empty()) continue;
    checksums << shard.relative_path << " bytes=" << shard.bytes << '\n';
    shard.reference_sha256.clear();
    local = true;
  }
  if (!local) return Status::Ok();
  auto directory = root / (manifest.has_quantized_forest()
                               ? PublishedNorm2Directory(manifest.hashes)
                               : std::filesystem::path("norm-cache/constant"));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return Status::Internal(
        "cannot create published norm2 metadata directory: " + error.message());
  for (auto item :
       {std::pair<const char*, std::string>{"config.txt", config.str()},
        {"identity.txt", checksums.str()}}) {
    std::ofstream f(directory / item.first, std::ios::binary);
    f << item.second;
    f.close();
    if (!f) return Status::Internal("cannot write published norm2 metadata");
  }
  return Status::Ok();
}

Status SharePublishedNorm2Cache(const std::filesystem::path& staged,
                                const std::filesystem::path& source_owner,
                                const std::filesystem::path& destination,
                                IndexManifest& manifest) {
  const auto relative = PublishedNorm2Directory(manifest.hashes);
  std::vector<ShardDescriptor*> owned;
  for (auto& shard : manifest.shards) {
    if (shard.kind != ShardKind::kNorm8 && shard.kind != ShardKind::kNorm2Lut &&
        shard.kind != ShardKind::kScaledScore8)
      continue;
    if (shard.reference_path != shard.relative_path &&
        !shard.reference_path.empty())
      continue;
    shard.reference_sha256.clear();
    owned.push_back(&shard);
  }
  if (owned.empty()) return Status::Ok();
  std::ifstream metadata(staged / relative / "config.txt");
  const std::string config((std::istreambuf_iterator<char>(metadata)), {});
  if (!metadata) return Status::DataLoss("shared norm2 recipe metadata is missing");
  const std::string content = "norm-family-v2\n" + config;

  auto family = source_owner / relative.parent_path() /
                (relative.filename().string() + "-" +
                 app::Sha256Bytes(content).substr(0, 16));
  std::error_code error;
  const auto primary = family;
  const bool new_population = std::any_of(owned.begin(), owned.end(),
      [](const auto* shard) { return shard->kind != ShardKind::kScaledScore8; });
  for (unsigned generation = 0;; ++generation) {
    const bool exists = std::filesystem::exists(family, error);
    if (error)
      return Status::Internal("cannot inspect shared norm2 generation: " +
                              error.message());
    if (!exists) break;
    bool complete = std::filesystem::exists(family / "identity.txt", error);
    for (const auto* shard : owned)
      if (shard->kind != ShardKind::kScaledScore8)
        complete &= std::filesystem::exists(
            family / std::filesystem::path(shard->relative_path).filename(),
            error);
    if (error)
      return Status::Internal("cannot inspect shared norm2 payload: " +
                              error.message());
    // A newly computed population and its fitted scale are one immutable
    // generation. Equal recipe/extent does not prove equal label assignments.
    // Cache hits reuse a whole manifest upstream; never replace new labels
    // here with a previous generation's payloads.
    if (complete && !new_population) break;
    family =
        primary.parent_path() / (primary.filename().string() + "-rebuild-" +
                                 std::to_string(generation + 1));
  }
  const auto target_name = [&](const ShardDescriptor& shard) {
    const auto name = std::filesystem::path(shard.relative_path).filename();
    return shard.kind == ShardKind::kScaledScore8
        ? std::filesystem::path("scale-" + manifest.scaled_score8_hash.ToHex()) / name
        : name;
  };
  const bool family_exists = std::filesystem::exists(family, error);
  if (error)
    return Status::Internal("cannot inspect shared norm2 cache: " +
                            error.message());
  if (!family_exists) {
    std::filesystem::create_directories(family.parent_path(), error);
    if (error)
      return Status::Internal("cannot create shared norm2 cache: " +
                              error.message());
    auto pending = StagingDirectory::Create(family);
    if (!pending.ok()) return pending.status();
    for (auto* shard : owned) {
      const auto target = pending.value().path() / target_name(*shard);
      std::filesystem::create_directories(target.parent_path(), error);
      if (error) return Status::Internal("cannot create norm2 scale directory: " + error.message());
      std::filesystem::rename(staged / shard->relative_path, target, error);
      if (error)
        return Status::Internal("cannot move prepared norm2 payload: " +
                                error.message());
    }
    for (const char* name : {"config.txt", "identity.txt"}) {
      if (std::filesystem::exists(staged / relative / name, error))
        std::filesystem::rename(staged / relative / name,
                                pending.value().path() / name, error);
      if (error)
        return Status::Internal("cannot move norm2 metadata: " +
                                error.message());
    }
    if (auto status = pending.value().Publish(); !status.ok()) return status;
  }
  for (auto* shard : owned) {
    const auto name = target_name(*shard);
    const auto target = family / name;
    if (std::filesystem::exists(target, error)) {
      if (std::filesystem::file_size(target, error) != shard->bytes || error)
        return Status::DataLoss("shared norm2 cache extent mismatch; cache was not overwritten");
    } else {
      if (error)
        return Status::Internal("cannot inspect shared norm2 cache: " +
                                error.message());
      // Scale generations have their own small-table identity. Publish each
      // without replacing norm files or other source-norm generations.
      if (shard->kind != ShardKind::kScaledScore8)
        return Status::DataLoss(
            "shared norm2 cache is incomplete; cache was not overwritten");
      std::filesystem::create_directories(target.parent_path(), error);
      if (error) return Status::Internal("cannot create norm2 scale directory: " + error.message());
#if defined(__linux__)
      if (syscall(SYS_renameat2, AT_FDCWD,
                  (staged / shard->relative_path).c_str(), AT_FDCWD,
                  target.c_str(), RENAME_NOREPLACE) != 0)
        return Status::FailedPrecondition(
            "cannot publish norm2 scale without replacing a file");
#else
      return Status::Unimplemented(
          "shared norm2 scale publication requires create-only rename");
#endif
    }
    ShardDescriptor file = *shard;
    file.relative_path = name.generic_string();
    file.reference_path.clear();
    file.reference_sha256.clear();
    auto ref = ReferenceCanonicalShard(family, destination, file);
    if (!ref.ok()) return ref.status();
    ref.value().relative_path = shard->relative_path;
    *shard = std::move(ref).value();
  }
  std::filesystem::remove_all(staged / "norm-cache", error);
  if (error)
    return Status::Internal("cannot remove empty norm2 staging directory: " +
                            error.message());
  return Status::Ok();
}
}  // namespace stlqpp
