#include "stlqpp/model/model_shape.h"
#include "evaluate_base_basic.h"

#include "texmex_reader.h"

#include "stlqpp/eval/eval_metrics.h"
#include "stlqpp/index/emit/base_list_reader.h"
#include "stlqpp/index/norm/reconstruction_norm8.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#include "stlqpp/runtime/cpu/openmp_parallel_for.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/search/searcher.h"
#include "stlqpp/search/topk/query_topk.h"
#include "stlqpp/storage/artifact_hash_factory.h"
#include "stlqpp/storage/canonical_index_files.h"
#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/storage/trained_model_file.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace stlqpp::app {
namespace {

using Clock = std::chrono::steady_clock;

StatusOr<TexmexVectorShape> InspectQueries(const EvalCommand& command) {
  if (command.effective_query_format == "fvecs") {
    return InspectFvecs(command.effective_queries);
  }
  if (command.effective_query_format == "fbin") {
    return InspectFbin(command.effective_queries);
  }
  if (command.effective_query_format == "bvecs") {
    return InspectBvecs(command.effective_queries);
  }
  if (command.effective_query_format == "npy") {
    return InspectNpyFloat16(command.effective_queries);
  }
  return Status::InvalidArgument("base query reader route was not resolved");
}

StatusOr<TexmexVectorShape> InspectGroundtruth(const EvalCommand& command) {
  if (command.effective_groundtruth_format == "ivecs") {
    return InspectIvecs(command.effective_groundtruth);
  }
  if (command.effective_groundtruth_format == "ibin") {
    return InspectIbin(command.effective_groundtruth);
  }
  if (command.effective_groundtruth_format == "npy") {
    return InspectNpyInt64(command.effective_groundtruth);
  }
  return Status::InvalidArgument(
      "base groundtruth reader route was not resolved");
}

StatusOr<TexmexFloatVectors> ReadQueries(const EvalCommand& command,
                                         std::uint32_t count) {
  if (command.effective_query_format == "fvecs") {
    return ReadFvecsPrefix(command.effective_queries, count);
  }
  if (command.effective_query_format == "fbin") {
    return ReadFbinPrefix(command.effective_queries, count);
  }
  if (command.effective_query_format == "bvecs") {
    auto bytes = ReadBvecsPrefix(command.effective_queries, count);
    if (!bytes.ok()) return bytes.status();
    TexmexFloatVectors output;
    output.count = bytes.value().count;
    output.dimension = bytes.value().dimension;
    output.values.resize(bytes.value().values.size());
    for (std::size_t index = 0U; index < output.values.size(); ++index) {
      output.values[index] = static_cast<float>(bytes.value().values[index]);
    }
    return output;
  }
  if (command.effective_query_format == "npy") {
    return ReadNpyFloat16Prefix(command.effective_queries, count);
  }
  return Status::InvalidArgument("base query reader route was not resolved");
}

StatusOr<TexmexU32Vectors> ReadGroundtruth(const EvalCommand& command,
                                           std::uint32_t count) {
  if (command.effective_groundtruth_format == "ivecs") {
    return ReadIvecsPrefix(command.effective_groundtruth, count);
  }
  if (command.effective_groundtruth_format == "ibin") {
    return ReadIbinPrefix(command.effective_groundtruth, count);
  }
  if (command.effective_groundtruth_format == "npy") {
    return ReadNpyInt64Prefix(command.effective_groundtruth, count);
  }
  return Status::InvalidArgument(
      "base groundtruth reader route was not resolved");
}

std::uint64_t IdFingerprint(const SearchResult& result) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint64_t id : result.ids) {
    for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
      hash ^= static_cast<std::uint8_t>(id >> shift);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

struct BasicModelTables final {
  RuntimeCanonicalModel model;
  ArtifactHash64 source_model_identity;
  std::vector<float> root_norm2;
  std::vector<float> root_inverse_norm;
  // Column-major C0^T Csmall: nlist x ((m-1)*256).
  std::vector<float> root_small_gram;
  // Column-major Csmall^T Csmall: Hsmall x Hsmall.
  std::vector<float> small_gram;
};

StatusOr<BasicModelTables> BuildModelTables(
    const EvalCommand& command,
    const CblasF32Gemm& gemm) {
  StatusOr<RuntimeCanonicalModel> decoded = [&]()
      -> StatusOr<RuntimeCanonicalModel> {
    if (!command.train_checkpoint_file.empty()) {
      std::filesystem::path model_file = command.train_checkpoint_file;
      if (model_file.extension() == ".stlqmodel") {
        auto artifact = TrainedModelFile::Open(model_file);
        if (!artifact.ok()) return artifact.status();
        const auto bytes = artifact.value().model_bytes();
        return CanonicalModelCodec::DecodeRuntime(bytes.data(), bytes.size());
      }
      std::error_code ec;
      if (std::filesystem::is_directory(model_file, ec)) {
        model_file /= "model.bin";
      } else if (ec) {
        return Status::Internal(
            "cannot inspect the base-eval training checkpoint path");
      }
      auto mapped = ReadOnlyMappedFile::Open(model_file);
      if (!mapped.ok()) return mapped.status();
      return CanonicalModelCodec::DecodeRuntime(
          mapped.value().data(), mapped.value().size());
    }
    auto files = CanonicalIndexFiles::Open(command.index);
    if (!files.ok()) return files.status();
    return CanonicalModelCodec::DecodeRuntime(
        files.value().model_bytes().data(),
        files.value().model_bytes().size());
  }();
  if (!decoded.ok()) return decoded.status();
  RuntimeCanonicalModel model = std::move(decoded).value();
  const RuntimeProductCodebookView books = model.codebooks_view();
  const ModelShape& shape = books.shape;
  const std::uint32_t m = shape.m;
  if (m != command.m) return Status::FailedPrecondition("base-eval model M differs from request");
  const std::uint32_t hsmall = (m - 1U) * 256U;

  BasicModelTables output{
      .model = std::move(model),
      .source_model_identity = {},
      .root_norm2 = std::vector<float>(shape.nlist, 0.0F),
      .root_inverse_norm = std::vector<float>(shape.nlist, 1.0F),
      .root_small_gram = std::vector<float>(
          static_cast<std::size_t>(shape.nlist) * hsmall),
      .small_gram = std::vector<float>(
          static_cast<std::size_t>(hsmall) * hsmall),
  };
  const RuntimeProductCodebookView owned = output.model.codebooks_view();
  if (output.model.source_space()) {
    output.source_model_identity = output.model.base_identity();
  } else {
    TrainedModelHashView hash_view;
    hash_view.rotation = output.model.rotation();
    hash_view.root_codebooks.reserve(m);
    hash_view.root_codebooks.push_back(owned.root0);
    for (std::uint32_t layer = 1U;
         layer < static_cast<std::uint32_t>(m); ++layer) {
      hash_view.root_codebooks.push_back(owned.root_small(layer));
  }
  output.source_model_identity = ArtifactHashFactory::BaseModel(hash_view);
  }
  if (const Status status = gemm.MultiplyTransposedLeft({
          .left = owned.root0,
          .right = owned.root_small_flat.values,
          .query_count = hsmall,
          .right_leading_dimension = shape.dimension,
          .output = output.root_small_gram.data(),
          .output_leading_dimension = shape.nlist,
      }); !status.ok()) {
    return status;
  }
  if (const Status status = gemm.MultiplyTransposedLeft({
          .left = owned.root_small_flat,
          .right = owned.root_small_flat.values,
          .query_count = hsmall,
          .right_leading_dimension = shape.dimension,
          .output = output.small_gram.data(),
          .output_leading_dimension = hsmall,
      }); !status.ok()) {
    return status;
  }
  for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid) {
    const float* root = owned.root0.column(cid);
    float norm2 = 0.0F;
    for (std::uint32_t axis = 0U; axis < shape.dimension; ++axis) {
      norm2 += root[axis] * root[axis];
    }
    output.root_norm2[cid] = norm2;
    output.root_inverse_norm[cid] =
        1.0F / std::sqrt(std::max(norm2, 1.0e-20F));
  }
  return output;
}

float ReconstructionNorm2(const BasicModelTables& model,
                          std::uint32_t cid,
                          const std::uint8_t* codes,
                          const float* coefficients) noexcept {
  const std::uint32_t nlist = model.model.codebooks_view().shape.nlist;
  const std::uint32_t m = model.model.shape().m;
  const std::uint32_t kHsmall = (m - 1U) * 256U;
  std::uint32_t flat[stlqpp::kMaximumModelLayers];
  flat[0] = 0U;
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    flat[layer] = (layer - 1U) * 256U + codes[layer - 1U];
  }
  const float a0 = coefficients[0U];
  float cross0 = 0.0F;
  for (std::uint32_t layer = 1U; layer < static_cast<std::uint32_t>(m);
       ++layer) {
    cross0 += coefficients[layer] * model.root_small_gram[
        cid + static_cast<std::size_t>(flat[layer]) * nlist];
  }
  float norm2 = (a0 * a0) * model.root_norm2[cid] +
      2.0F * a0 * cross0;
  float small = 0.0F;
  for (std::uint32_t left = 1U; left < static_cast<std::uint32_t>(m);
       ++left) {
    const float aleft = coefficients[left];
    const std::uint32_t fleft = flat[left];
    small += (aleft * aleft) * model.small_gram[
        fleft + static_cast<std::size_t>(fleft) * kHsmall];
    for (std::uint32_t right = left + 1U;
         right < static_cast<std::uint32_t>(m); ++right) {
      const float product = 2.0F * aleft * coefficients[right];
      small += product * model.small_gram[
          fleft + static_cast<std::size_t>(flat[right]) * kHsmall];
    }
  }
  return norm2 + small;
}

struct PreparedBasicCluster final {
  std::uint32_t cid = 0U;
  RuntimeLoadedBaseListSupportCid payload;
  std::vector<float> norm2;
};

Status PrepareNorms(const EvalCommand& command,
                    const BasicModelTables& model,
                    PreparedBasicCluster* cluster) {
  const std::uint32_t m = model.model.shape().m;
  const auto codes = cluster->payload.small_codes_row_major();
  const auto coefficients = cluster->payload.coefficients_row_major();
  const std::size_t rows = cluster->payload.ids().size();
  cluster->norm2.resize(rows);
  for (std::size_t row = 0U; row < rows; ++row) {
    const float value = ReconstructionNorm2(
        model, cluster->cid, codes.data() + row * (m - 1U),
        coefficients.data() + row * m);
    if (!std::isfinite(value) || value < 0.0F) {
      return Status::DataLoss(
          "base-basic reconstruction produced an invalid squared norm");
    }
    cluster->norm2[row] = value;
  }
  const EvalNormMode mode = command.requested_norm_mode.value_or(
      EvalNormMode::kLut);
  if (mode == EvalNormMode::kFloat || rows == 0U) return Status::Ok();
  const ReconstructionNorm8Mode recipe =
      mode == EvalNormMode::kLutTailWeighted
      ? ReconstructionNorm8Mode::kTailWeightedLloyd
      : ReconstructionNorm8Mode::kIdentityLloyd;
  auto codebook = ReconstructionNorm8Codebook::InitializeFromValues(
      cluster->norm2, recipe);
  if (!codebook.ok()) return codebook.status();
  const std::uint32_t iterations =
      command.expected_norm8_lloyd_iterations.value_or(25U);
  for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
    Norm8LloydAccumulator pass;
    if (const Status status = pass.Accumulate(
            codebook.value().view(), cluster->norm2, recipe,
            codebook.value().tail_p99(), codebook.value().tail_p999());
        !status.ok()) {
      return status;
    }
    if (codebook.value().FinishPass(pass)) break;
  }
  std::vector<std::uint8_t> labels(rows);
  if (const Status status = codebook.value().Encode(
          cluster->norm2, labels); !status.ok()) {
    return status;
  }
  const auto centers = codebook.value().centers();
  for (std::size_t row = 0U; row < rows; ++row) {
    cluster->norm2[row] = centers[labels[row]];
  }
  return Status::Ok();
}

struct RouteScratch final {
  std::vector<std::uint32_t> ids;
};

void SelectRoutes(const float* root_dot,
                  const std::vector<float>& inverse_norm,
                  std::uint32_t nprobe,
                  RouteScratch* scratch,
                  std::uint32_t* output) {
  const std::uint32_t nlist =
      static_cast<std::uint32_t>(inverse_norm.size());
  scratch->ids.resize(nlist);
  std::iota(scratch->ids.begin(), scratch->ids.end(), 0U);
  const auto better = [&](std::uint32_t left, std::uint32_t right) {
    const float left_score = root_dot[left] * inverse_norm[left];
    const float right_score = root_dot[right] * inverse_norm[right];
    return left_score > right_score ||
        (left_score == right_score && left < right);
  };
  std::partial_sort(scratch->ids.begin(),
                    scratch->ids.begin() + nprobe,
                    scratch->ids.end(), better);
  std::copy_n(scratch->ids.begin(), nprobe, output);
}

struct QueryScanContext final {
  std::uint32_t m = 0U;
  const std::vector<std::uint32_t>* routes = nullptr;
  std::uint32_t nprobe = 0U;
  const std::vector<std::int32_t>* cid_to_active = nullptr;
  const std::vector<PreparedBasicCluster>* clusters = nullptr;
  const float* root_dot = nullptr;
  const float* small_dot = nullptr;
  std::uint32_t nlist = 0U;
  std::uint32_t query_count = 0U;
  std::uint32_t top_k = 0U;
  std::uint64_t* result_ids = nullptr;
  float* result_distances = nullptr;
  std::vector<std::unique_ptr<QueryTopK>>* heaps = nullptr;
};

void ScanOneQuery(void* opaque,
                  std::uint32_t worker,
                  std::uint32_t query) noexcept {
  auto& context = *static_cast<QueryScanContext*>(opaque);
  const std::uint32_t m = context.m;
  QueryTopK& heap = *(*context.heaps)[worker];
  heap.Reset();
  const float* qroot = context.root_dot +
      static_cast<std::size_t>(query) * context.nlist;
  const float* qsmall = context.small_dot +
      static_cast<std::size_t>(query) * (m - 1U) * 256U;
  const std::uint32_t* routes = context.routes->data() +
      static_cast<std::size_t>(query) * context.nprobe;
  for (std::uint32_t route = 0U; route < context.nprobe; ++route) {
    const std::uint32_t cid = routes[route];
    const std::int32_t active = (*context.cid_to_active)[cid];
    assert(active >= 0);
    const PreparedBasicCluster& cluster =
        (*context.clusters)[static_cast<std::size_t>(active)];
    const auto ids = cluster.payload.ids();
    const auto codes = cluster.payload.small_codes_row_major();
    const auto coefficients = cluster.payload.coefficients_row_major();
    for (std::size_t row = 0U; row < ids.size(); ++row) {
      const float* a = coefficients.data() + row * m;
      const std::uint8_t* code = codes.data() + row * (m - 1U);
      float dot = a[0U] * qroot[cid];
      for (std::uint32_t layer = 1U;
           layer < static_cast<std::uint32_t>(m); ++layer) {
        dot += a[layer] * qsmall[
            static_cast<std::size_t>(layer - 1U) * 256U +
            code[layer - 1U]];
      }
      heap.Consider(ids[row], cluster.norm2[row] - 2.0F * dot);
    }
  }
  const std::vector<SearchCandidate>& ordered = heap.Finalize();
  std::uint64_t* ids = context.result_ids +
      static_cast<std::size_t>(query) * context.top_k;
  float* distances = context.result_distances +
      static_cast<std::size_t>(query) * context.top_k;
  std::uint32_t rank = 0U;
  for (; rank < ordered.size(); ++rank) {
    ids[rank] = ordered[rank].id;
    distances[rank] = ordered[rank].squared_distance;
  }
  for (; rank < context.top_k; ++rank) {
    ids[rank] = std::numeric_limits<std::uint64_t>::max();
    distances[rank] = std::numeric_limits<float>::infinity();
  }
}

Status SearchBaseBasic(const EvalCommand& command,
                       const TexmexFloatVectors& queries,
                       const BasicModelTables& model,
                       const RuntimeBaseListReader& lists,
                       const CblasF32Gemm& gemm,
                       const OpenMpParallelFor& parallel,
                       SearchResult* result) {
  const ModelShape& shape = model.model.codebooks_view().shape;
  const std::uint32_t m = shape.m;
  if (lists.store().m() != m || queries.dimension != shape.dimension ||
      lists.store().shape().dimension != shape.dimension ||
      lists.store().shape().nlist != shape.nlist) {
    return Status::FailedPrecondition(
        "base-basic list, query and model shapes do not match");
  }
  const std::uint32_t nprobe = command.base_use_ivf
      ? std::min(command.base_nprobe, shape.nlist) : shape.nlist;
  const std::uint32_t top_k = static_cast<std::uint32_t>(std::min<
      std::uint64_t>(command.top_k, lists.store().total_vectors()));
  if (nprobe == 0U || top_k == 0U) {
    return Status::InvalidArgument("base-basic eval has no searchable rows");
  }
  result->query_count = queries.count;
  result->top_k = top_k;
  result->ids.assign(static_cast<std::size_t>(queries.count) * top_k, 0U);
  result->adc_distances.assign(
      static_cast<std::size_t>(queries.count) * top_k,
      std::numeric_limits<float>::infinity());

  const std::uint32_t block_capacity = std::min<std::uint32_t>(
      std::max<std::uint32_t>(1U, command.query_block), 256U);
  std::vector<RouteScratch> route_scratch(parallel.worker_count());
  std::vector<std::unique_ptr<QueryTopK>> heaps;
  heaps.reserve(parallel.worker_count());
  for (std::uint32_t worker = 0U; worker < parallel.worker_count(); ++worker) {
    heaps.push_back(std::make_unique<QueryTopK>(top_k));
  }
  const RuntimeProductCodebookView books = model.model.codebooks_view();
  for (std::uint32_t first = 0U; first < queries.count;
       first += block_capacity) {
    const std::uint32_t count =
        std::min(block_capacity, queries.count - first);
    const float* source = queries.values.data() +
        static_cast<std::size_t>(first) * shape.dimension;
    std::vector<float> transformed;
    std::vector<float> root_dot(
        static_cast<std::size_t>(shape.nlist) * count);
    std::vector<float> small_dot(
        static_cast<std::size_t>((m - 1U) * 256U) * count);
    if (!model.model.source_space()) {
      transformed.resize(static_cast<std::size_t>(shape.dimension) * count);
      if (const Status status = gemm.MultiplyLeft({
              .left = model.model.rotation(),
              .right = source,
              .query_count = count,
              .right_leading_dimension = shape.dimension,
              .output = transformed.data(),
              .output_leading_dimension = shape.dimension,
          }); !status.ok()) {
        return status;
      }
      source = transformed.data();
    }
    if (const Status status = gemm.MultiplyTransposedLeft({
            .left = books.root0,
            .right = source,
            .query_count = count,
            .right_leading_dimension = shape.dimension,
            .output = root_dot.data(),
            .output_leading_dimension = shape.nlist,
        }); !status.ok()) {
      return status;
    }
    if (const Status status = gemm.MultiplyTransposedLeft({
            .left = books.root_small_flat,
            .right = source,
            .query_count = count,
            .right_leading_dimension = shape.dimension,
            .output = small_dot.data(),
            .output_leading_dimension = (m - 1U) * 256U,
        }); !status.ok()) {
      return status;
    }

    std::vector<std::uint32_t> routes(
        static_cast<std::size_t>(count) * nprobe);
    struct RouteContext final {
      const std::vector<float>* root_dot;
      const std::vector<float>* inverse_norm;
      std::uint32_t nlist;
      std::uint32_t nprobe;
      std::vector<RouteScratch>* scratch;
      std::vector<std::uint32_t>* routes;
    } route_context{&root_dot, &model.root_inverse_norm, shape.nlist,
                    nprobe, &route_scratch, &routes};
    parallel.Run(count, &route_context,
        [](void* opaque, std::uint32_t worker,
           std::uint32_t query) noexcept {
          auto& context = *static_cast<RouteContext*>(opaque);
          SelectRoutes(
              context.root_dot->data() +
                  static_cast<std::size_t>(query) * context.nlist,
              *context.inverse_norm, context.nprobe,
              &(*context.scratch)[worker],
              context.routes->data() +
                  static_cast<std::size_t>(query) * context.nprobe);
        });

    std::vector<std::uint8_t> active_flags(shape.nlist, 0U);
    for (const std::uint32_t cid : routes) active_flags[cid] = 1U;
    std::vector<std::uint32_t> active_cids;
    for (std::uint32_t cid = 0U; cid < shape.nlist; ++cid) {
      if (active_flags[cid] != 0U) active_cids.push_back(cid);
    }
    std::vector<std::int32_t> cid_to_active(shape.nlist, -1);
    std::vector<PreparedBasicCluster> clusters;
    clusters.reserve(active_cids.size());
    for (const std::uint32_t cid : active_cids) {
      auto loaded = lists.LoadCidSupport(cid);
      if (!loaded.ok()) return loaded.status();
      PreparedBasicCluster cluster{
          .cid = cid,
          .payload = std::move(loaded).value(),
          .norm2 = {},
      };
      if (const Status status = PrepareNorms(command, model, &cluster);
          !status.ok()) {
        return status;
      }
      cid_to_active[cid] = static_cast<std::int32_t>(clusters.size());
      clusters.push_back(std::move(cluster));
    }

    QueryScanContext scan_context{
        .m = m,
        .routes = &routes,
        .nprobe = nprobe,
        .cid_to_active = &cid_to_active,
        .clusters = &clusters,
        .root_dot = root_dot.data(),
        .small_dot = small_dot.data(),
        .nlist = shape.nlist,
        .query_count = count,
        .top_k = top_k,
        .result_ids = result->ids.data() +
            static_cast<std::size_t>(first) * top_k,
        .result_distances = result->adc_distances.data() +
            static_cast<std::size_t>(first) * top_k,
        .heaps = &heaps,
    };
    parallel.Run(count, &scan_context, &ScanOneQuery);
  }
  return Status::Ok();
}

Status RunTyped(const EvalCommand& command) {
  auto query_shape = InspectQueries(command);
  if (!query_shape.ok()) return query_shape.status();
  auto groundtruth_shape = InspectGroundtruth(command);
  if (!groundtruth_shape.ok()) return groundtruth_shape.status();
  std::uint32_t selected = query_shape.value().count;
  if (command.nquery != 0U) selected = std::min(selected, command.nquery);
  if (groundtruth_shape.value().count < selected) {
    LogWarn("Ground truth is shorter than the selected base-eval query "
            "prefix; shrinking queries to match.");
  }
  const std::uint32_t query_count =
      std::min(selected, groundtruth_shape.value().count);
  if (query_count == 0U || groundtruth_shape.value().dimension == 0U) {
    return Status::InvalidArgument(
        "base query and groundtruth files have no comparable rows");
  }
  auto queries = ReadQueries(command, query_count);
  if (!queries.ok()) return queries.status();
  auto groundtruth = ReadGroundtruth(command, query_count);
  if (!groundtruth.ok()) return groundtruth.status();
  if (command.groundtruth_add1) {
    for (std::uint32_t& id : groundtruth.value().values) ++id;
  }
  auto lists = RuntimeBaseListReader::Open(command.base_list_directory);
  if (!lists.ok()) return lists.status();
  auto parallel = OpenMpParallelFor::Create(command.effective_threads);
  if (!parallel.ok()) return parallel.status();
  CblasF32Gemm gemm(command.effective_threads);
  auto model = BuildModelTables(command, gemm);
  if (!model.ok()) return model.status();
  if (lists.value().store().identity().value() == 0U) {
    return Status::FailedPrecondition(
        "base-list product has no source identity");
  }
  if (lists.value().store().source_model_identity().empty()) {
    return Status::FailedPrecondition(
        "base-list product predates model-identity binding; re-encode the "
        "base set before basic recall evaluation");
  }
  if (lists.value().store().source_model_identity() !=
      model.value().source_model_identity) {
    return Status::FailedPrecondition(
        "base-list product was encoded by a different training model");
  }

  const std::uint32_t effective_nprobe = command.base_use_ivf
      ? std::min(command.base_nprobe,
                 model.value().model.codebooks_view().shape.nlist)
      : model.value().model.codebooks_view().shape.nlist;
  LogInfo("========= Evaluating base basic recall (disk IVF, float "
          "coefficients) =========");
  LogInfo("Base basic model source: " +
          (command.train_checkpoint_file.empty()
               ? (command.index / "model.bin").string()
               : command.train_checkpoint_file.string()));
  LogInfo("Base basic recall mode: " + std::string(command.base_use_ivf
              ? "disk-ivf" : "full-scan") + " (nprobe=" +
          std::to_string(effective_nprobe) + ") norm2_mode=" +
          EvalNormModeName(command.requested_norm_mode.value_or(
              EvalNormMode::kLut)));
  for (std::uint32_t warmup = 0U; warmup < command.base_warmup; ++warmup) {
    SearchResult discarded;
    if (const Status status = SearchBaseBasic(
            command, queries.value(), model.value(), lists.value(), gemm,
            parallel.value(), &discarded); !status.ok()) {
      return status;
    }
  }
  SearchResult accepted;
  std::uint64_t fingerprint = 0U;
  std::vector<double> seconds;
  for (std::uint32_t repeat = 0U; repeat < command.base_repeat; ++repeat) {
    SearchResult current;
    const Clock::time_point start = Clock::now();
    if (const Status status = SearchBaseBasic(
            command, queries.value(), model.value(), lists.value(), gemm,
            parallel.value(), &current); !status.ok()) {
      return status;
    }
    const double elapsed = std::chrono::duration<double>(
        Clock::now() - start).count();
    const std::uint64_t current_fingerprint = IdFingerprint(current);
    if (repeat != 0U && current_fingerprint != fingerprint) {
      return Status::FailedPrecondition(
          "base-basic output changed between repeats");
    }
    fingerprint = current_fingerprint;
    accepted = std::move(current);
    seconds.push_back(elapsed);
    LogInfo("Base basic recall QPS: nprobe=" +
            std::to_string(effective_nprobe) + " run=" +
            std::to_string(repeat) + " qps_core=" +
            std::to_string(static_cast<double>(query_count) / elapsed));
  }
  EvalMetricBundle metrics;
  if (!ComputeEvalMetricBundle(
          accepted, groundtruth.value().values,
          groundtruth.value().dimension, command.metric_mode,
          command.metric_gt_topks, &metrics)) {
    return Status::InvalidArgument(
        "base groundtruth does not support requested eval metrics");
  }
  for (const std::string& line : BuildEvalMetricReportLines(metrics)) {
    LogInfo(line);
  }
  std::sort(seconds.begin(), seconds.end());
  const double median = seconds.size() % 2U == 0U
      ? 0.5 * (seconds[seconds.size() / 2U - 1U] +
               seconds[seconds.size() / 2U])
      : seconds[seconds.size() / 2U];
  double sum = 0.0;
  for (const double value : seconds) sum += value;
  const double average = sum / static_cast<double>(seconds.size());
  LogInfo("Base basic recall summary: nprobe=" +
          std::to_string(effective_nprobe) + " nq=" +
          std::to_string(query_count) + " repeat=" +
          std::to_string(command.base_repeat) + " " +
          FormatEvalMetricSummary(metrics) + " median_core_sec=" +
          std::to_string(median) + " median_qps_core=" +
          std::to_string(static_cast<double>(query_count) / median) +
          " avg_core_sec=" + std::to_string(average) +
          " avg_qps_core=" +
          std::to_string(static_cast<double>(query_count) / average));
  return Status::Ok();
}

}  // namespace

Status RunBaseBasicEval(const EvalCommand& command) {
  if (command.expected_norm8_mode == ReconstructionNorm8Mode::kReferenceLloyd &&
      command.requested_norm_mode != EvalNormMode::kFloat)
    return Status::InvalidArgument(
        "reference norm LUT is a linkage option; disable eval.base.enabled");
  try {
    return RunTyped(command);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted(
        "base-basic evaluation allocation failed");
  }
}

}  // namespace stlqpp::app
