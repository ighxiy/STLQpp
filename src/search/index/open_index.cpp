#include "stlqpp/search/index/open_index.h"

#include "stlqpp/search/layout/resident_cluster.h"
#include "stlqpp/search/tables/query_table_builder.h"
#include "stlqpp/storage/canonical_cluster_codec.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/core/numeric/little_endian.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

bool SameShape(const ModelShape& left, const ModelShape& right) noexcept {
  return left.dimension == right.dimension && left.m == right.m &&
         left.nlist == right.nlist &&
         left.residual_codebook_size == right.residual_codebook_size;
}

StatusOr<CanonicalIndexFiles> OpenCheckedIndexFiles(
    const std::filesystem::path& directory, std::uint32_t m,
    SearchCoefficientMode coefficient_mode, SearchScoreMode score_mode,
    bool external_norms) {
  auto files_result = external_norms
      ? CanonicalIndexFiles::OpenCoefficients(directory, coefficient_mode == SearchCoefficientMode::kFloat)
      : coefficient_mode == SearchCoefficientMode::kFloat
      ? (IsScaledScoreMode(score_mode)
             ? CanonicalIndexFiles::OpenNativeFloatScaled(directory)
             : CanonicalIndexFiles::OpenNativeFloat(directory))
      : CanonicalIndexFiles::OpenQuantized(directory);
  if (!files_result.ok()) return files_result.status();
  CanonicalIndexFiles files = std::move(files_result).value();
  if (files.manifest().shape.m != m) {
    return Status::FailedPrecondition(
        "canonical index M does not match the selected product decoder");
  }
  if (!external_norms && coefficient_mode == SearchCoefficientMode::kInt8 &&
      !IsScaledScoreMode(score_mode) &&
      !files.manifest().has_classic_norm8()) {
    return Status::FailedPrecondition(
        "classic quantized search requested but the index has no classic norm8 representation");
  }
  if (!external_norms && coefficient_mode == SearchCoefficientMode::kInt8 &&
      IsScaledScoreMode(score_mode) &&
      !files.manifest().has_scaled_score8()) {
    return Status::FailedPrecondition(
        "scaled quantized search requested but the index has no scaled score8 table");
  }
  return files;
}

struct FloatClusterValues final {
  std::vector<float> coefficients;
  std::vector<float> norms;
  std::vector<float> source_norms;
};

StatusOr<FloatClusterValues> LoadFloatClusterValues(
    const CanonicalIndexFiles& files, std::uint32_t cid,
    SearchScoreMode score_mode, const ClusterRowScores* scores) {
  const CanonicalCidEntry& entry = files.entry(cid);
  const CanonicalFloatRecordViews records = files.float_cluster_records(cid);
  const std::uint64_t node_count =
      static_cast<std::uint64_t>(entry.real_count) + entry.virtual_count;
  const std::uint32_t m = files.manifest().shape.m;
  if (node_count > std::numeric_limits<std::size_t>::max() / m ||
      records.coefficients.size() != node_count * m * sizeof(float) ||
      (scores == nullptr && records.norm2.size() !=
          static_cast<std::uint64_t>(entry.real_count) * sizeof(float))) {
    return Status::DataLoss("native-float CID record extent is invalid");
  }
  FloatClusterValues values;
  values.coefficients.resize(static_cast<std::size_t>(node_count) * m);
  values.norms.resize(entry.real_count);
  for (std::size_t offset = 0U; offset < values.coefficients.size(); ++offset) {
    values.coefficients[offset] = LoadLittleEndianF32(
        records.coefficients.data() + offset * sizeof(float));
  }
  for (std::size_t row = 0U; row < values.norms.size(); ++row) {
    values.norms[row] = scores != nullptr
        ? (scores->has_reference_correction ? 0.0F : scores->norm2[row])
        : LoadLittleEndianF32(records.norm2.data() + row * sizeof(float));
  }
  if (scores == nullptr && IsScaledScoreMode(score_mode)) {
    values.source_norms.resize(entry.real_count);
    if (const Status status = files.ReadFloatSourceNorm2(
            cid, values.source_norms); !status.ok()) return status;
  }
  return values;
}

}  // namespace

template <int M>
StatusOr<std::shared_ptr<const ResidentIndex<M>>> OpenIndex(
    const std::filesystem::path& directory,
    SearchCoefficientMode coefficient_mode,
    ParentAccessMode parent_access_mode,
    SearchScoreMode score_mode,
    const F32GemmBackend* query_codebook_pretransform_gemm,
    const IndexRowScores* row_scores) {
  static_assert(M >= 2 && M <= 20);
  try {
    auto files_result = OpenCheckedIndexFiles(
        directory, M, coefficient_mode, score_mode, row_scores != nullptr);
    if (!files_result.ok()) {
      return files_result.status();
    }
    CanonicalIndexFiles files = std::move(files_result).value();
    if (row_scores != nullptr && row_scores->clusters.size() != files.manifest().shape.nlist)
      return Status::DataLoss("evaluation norm CID count disagrees with index");
    std::vector<float> placeholder_centers(256U, 0.0F);
    const Norm8CodebookView norms = row_scores != nullptr
        ? Norm8CodebookView{placeholder_centers.data(), 256U} : files.norm8();
    const ArrayView<const std::uint8_t> model_bytes = files.model_bytes();
    auto model_result = CanonicalModelCodec::Decode<M>(
        model_bytes.data(), model_bytes.size());
    if (!model_result.ok()) {
      return model_result.status();
    }
    CanonicalModel<M> model = std::move(model_result).value();
    if (!SameShape(model.codebooks().shape(), files.manifest().shape)) {
      return Status::DataLoss(
          "canonical model shape disagrees with the index manifest");
    }
    QueryTransform transform = std::move(model).TakeQueryTransform();
    ProductCodebooks<M> codebooks = std::move(model).TakeCodebooks();
    const bool query_codebooks_pretransformed =
        query_codebook_pretransform_gemm != nullptr;
    if (query_codebooks_pretransformed) {
      auto transformed = PretransformQueryCodebooks<M>(
          transform, codebooks, *query_codebook_pretransform_gemm);
      if (!transformed.ok()) {
        return transformed.status();
      }
      // Move assignment releases the original model-space C immediately;
      // the resident index owns only R^T C in this mode.
      codebooks = std::move(transformed).value();
    }

    std::vector<ResidentCluster<M>> clusters;
    clusters.resize(files.manifest().shape.nlist);
    for (auto cid : files.physical_cid_order()) {
      const CanonicalCidEntry& entry = files.entry(cid);
      if (coefficient_mode == SearchCoefficientMode::kFloat) {
        const CanonicalTopologyRecordViews topology_records =
            files.topology_records(cid);
        auto topology =
            CanonicalClusterCodec<M>::DecodeNativeBfsTopology(
                cid, entry.real_count, entry.virtual_count,
                entry.real_root_count, topology_records.codes,
                topology_records.parents);
        if (!topology.ok()) return topology.status();
        auto values = LoadFloatClusterValues(files, cid, score_mode,
            row_scores ? &row_scores->clusters[cid] : nullptr);
        if (!values.ok()) return values.status();
        const auto& decoded = topology.value();
        auto resident = BuildFloatBfsResidentClusterFromDecodedTopology<M>(
            decoded, values.value().coefficients, values.value().norms,
            values.value().source_norms);
        if (!resident.ok()) return resident.status();
        ResidentCluster<M> cluster = std::move(resident).value();
        if (row_scores != nullptr) {
          if (const Status status = cluster.ConfigureRowScores(row_scores->clusters[cid]); !status.ok()) return status;
        }
        if (const Status status = cluster.ConfigureParentAccess(
                parent_access_mode);
            !status.ok()) {
          return status;
        }
        clusters[cid] = std::move(cluster);
      } else {
        auto decoded_result = CanonicalClusterCodec<M>::DecodeNativeBfs(
            cid, entry.real_count, entry.virtual_count,
            entry.real_root_count, files.cluster_records(cid), norms, row_scores == nullptr);
        if (!decoded_result.ok()) return decoded_result.status();
        DecodedCanonicalCluster<M> decoded =
            std::move(decoded_result).value();
        auto trusted = decoded.trusted_bfs_view(norms);
        if (!trusted.ok()) return trusted.status();
        ResidentCluster<M> cluster =
            BuildBfsResidentCluster(trusted.value());
        if (row_scores != nullptr) {
          if (const Status status = cluster.ConfigureRowScores(row_scores->clusters[cid]); !status.ok()) return status;
        }
        if (const Status status = cluster.ConfigureParentAccess(
                parent_access_mode);
            !status.ok()) {
          return status;
        }
        clusters[cid] = std::move(cluster);
      }
    }

    std::vector<float> norm8_centers;
    ScaledScore8CodebookView scaled_score8;
    if (coefficient_mode == SearchCoefficientMode::kInt8) {
      norm8_centers.assign(
          norms.centers, norms.centers + norms.center_count);
      if (row_scores == nullptr && IsScaledScoreMode(score_mode)) {
        scaled_score8 = files.scaled_score8();
      }
    }
    auto index_result = ResidentIndex<M>::Create(
        std::move(transform), std::move(codebooks),
        std::move(norm8_centers), std::move(clusters), coefficient_mode,
        parent_access_mode, score_mode, scaled_score8,
        query_codebooks_pretransformed);
    if (!index_result.ok()) {
      return index_result.status();
    }
    if (index_result.value().real_count() != files.manifest().counts.real) {
      return Status::DataLoss(
          "resident real-row count disagrees with the canonical manifest");
    }
    return std::shared_ptr<const ResidentIndex<M>>(
        std::make_shared<ResidentIndex<M>>(
            std::move(index_result).value()));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "canonical index resident materialization exhausted memory");
  }
}

template StatusOr<std::shared_ptr<const ResidentIndex<5>>> OpenIndex<5>(
    const std::filesystem::path& directory, SearchCoefficientMode,
    ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);
template StatusOr<std::shared_ptr<const ResidentIndex<12>>> OpenIndex<12>(
    const std::filesystem::path& directory, SearchCoefficientMode,
    ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);
template StatusOr<std::shared_ptr<const ResidentIndex<6>>> OpenIndex<6>(
    const std::filesystem::path& directory, SearchCoefficientMode,
    ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);
template StatusOr<std::shared_ptr<const ResidentIndex<10>>> OpenIndex<10>(
    const std::filesystem::path& directory, SearchCoefficientMode,
    ParentAccessMode, SearchScoreMode, const F32GemmBackend*, const IndexRowScores*);

}  // namespace stlqpp
