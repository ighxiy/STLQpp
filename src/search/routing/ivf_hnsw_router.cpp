#include "stlqpp/search/routing/ivf_hnsw_router.h"
#include "stlqpp/application/sha256.h"
#include "stlqpp/platform/staging_directory.h"
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace stlqpp {
struct IvfHnswRouter::Impl final {
  ColumnMajorMatrixF32View roots;
  std::filesystem::path index_path;
  hnswlib::InnerProductSpace space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
  explicit Impl(ColumnMajorMatrixF32View source) : roots(source), space(source.rows) {}
  struct Block {
    const Impl* self;
    const TrustedQueryBatchView* queries;
    std::uint32_t nprobe;
    std::uint32_t* routes;
    float* dots;
    std::vector<Status>* failures;
  };
  Status RouteQuery(const float* query, std::uint32_t nprobe,
                    std::uint32_t* ids, float* dots) const noexcept {
    try {
      auto heap = graph->searchKnn(query, nprobe);
      if (heap.size() != nprobe)
        return Status::DataLoss("IVF HNSW returned an incomplete probe set");
      // DEV's descending IP order; pair ties follow hnswlib's label ordering.
      for (std::uint32_t i = nprobe; i != 0U; --i) {
        ids[i - 1U] = static_cast<std::uint32_t>(heap.top().second);
        heap.pop();
      }
      for (std::uint32_t i = 0; i < nprobe; ++i) {
        const auto cid = ids[i];
        const float* root = roots.column(cid);
        float dot = 0.0F;
        for (std::uint32_t axis = 0; axis < roots.rows; ++axis)
          dot += query[axis] * root[axis];
        dots[cid] = dot;  // ADC needs the unnormalized root, not 1-IP.
      }
      return Status::Ok();
    } catch (const std::bad_alloc&) {
      return Status::ResourceExhausted("IVF HNSW query allocation failed");
    } catch (const std::exception& e) {
      return Status::Internal(std::string("IVF HNSW query failed: ") + e.what());
    }
  }
  static void Query(void* context, std::uint32_t worker, std::uint32_t q) noexcept {
    auto& b = *static_cast<Block*>(context);
    if (!(*b.failures)[worker].ok()) return;
    (*b.failures)[worker] = b.self->RouteQuery(
        b.queries->data() + static_cast<std::size_t>(q) * b.queries->stride_floats(),
        b.nprobe, b.routes + static_cast<std::size_t>(q) * b.nprobe,
        b.dots + static_cast<std::size_t>(q) * b.self->roots.columns);
  }
};
IvfHnswRouter::IvfHnswRouter(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
IvfHnswRouter::~IvfHnswRouter() = default;
const std::filesystem::path& IvfHnswRouter::index_path() const noexcept { return impl_->index_path; }

StatusOr<std::unique_ptr<IvfHnswRouter>> IvfHnswRouter::Open(
    ColumnMajorMatrixF32View roots, const IvfHnswOptions& o) {
  if (!roots.values || roots.rows == 0U || roots.columns == 0U ||
      o.m < 2U || o.m > 10000U || o.ef_construction == 0U || o.ef_search == 0U)
    return Status::InvalidArgument("invalid IVF HNSW shape or parameters");
  try {
    auto impl = std::make_unique<Impl>(roots);
    std::vector<float> units(static_cast<std::size_t>(roots.rows) * roots.columns);
    for (std::uint32_t cid = 0; cid < roots.columns; ++cid) {
      const float* source = roots.column(cid);
      double ss = 0.0;
      for (std::uint32_t axis = 0; axis < roots.rows; ++axis) {
        if (!std::isfinite(source[axis])) return Status::DataLoss("nonfinite IVF centroid");
        ss += static_cast<double>(source[axis]) * source[axis];
      }
      const float inv = 1.0F / std::sqrt(std::max(static_cast<float>(ss), 1.0e-20F));
      for (std::uint32_t axis = 0; axis < roots.rows; ++axis)
        units[static_cast<std::size_t>(cid) * roots.rows + axis] = source[axis] * inv;
    }
    const auto source_hash = app::Sha256Bytes({reinterpret_cast<const char*>(units.data()), units.size() * sizeof(float)});
    std::ostringstream recipe;
    recipe << "version=1\nspace=ip_unit\nd=" << roots.rows << "\nnlist=" << roots.columns
           << "\nM=" << o.m << "\nef_construction=" << o.ef_construction
           << "\nseed=100\ninsertion=cid_sequential\ncentroids_unit_sha256=" << source_hash << '\n';
    const std::string config = recipe.str();
    std::filesystem::path destination;
    bool exists = false;
    if (!o.cache_root.empty()) {
      destination = o.cache_root / ("unit-ip-M" + std::to_string(o.m) + "-efc" +
          std::to_string(o.ef_construction) + "-" + app::Sha256Bytes(config));
      exists = std::filesystem::exists(destination);
    }
    if (exists) {
      std::ifstream in(destination / "meta.txt");
      const std::string meta((std::istreambuf_iterator<char>(in)), {});
      if (!in || meta.rfind(config, 0U) != 0U)
        return Status::DataLoss("IVF HNSW cache recipe mismatch: " + destination.string());
      const auto extent = std::filesystem::file_size(destination / "index.bin");
      const auto token = meta.substr(config.size());
      if (token.rfind("index_bytes=", 0U) == 0U) {
        if (token != "index_bytes=" + std::to_string(extent) + "\n")
          return Status::DataLoss("IVF HNSW cache extent mismatch");
      } else if (token.rfind("index_sha256=", 0U) != 0U || token.size() != 78U) {
        return Status::DataLoss("invalid legacy IVF HNSW cache metadata");
      }
      impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          &impl->space, (destination / "index.bin").string(), false, roots.columns);
      if (impl->graph->cur_element_count != roots.columns)
        return Status::DataLoss("IVF HNSW cache centroid count mismatch");
    } else {
      impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
          &impl->space, roots.columns, o.m, std::max(8U, o.ef_construction), 100U);
      // Stable local IDs and seed. Parallel DEV insertion is schedule dependent;
      // sequential insertion makes a cached generation reproducible.
      for (std::uint32_t cid = 0; cid < roots.columns; ++cid)
        impl->graph->addPoint(units.data() + static_cast<std::size_t>(cid) * roots.rows, cid);
      if (!destination.empty()) {
        std::filesystem::create_directories(o.cache_root);
        auto stage = StagingDirectory::Create(destination);
        if (!stage.ok()) {
          if (std::filesystem::exists(destination)) return Open(roots, o);
          return stage.status();
        }
        impl->graph->saveIndex((stage.value().path() / "index.bin").string());
        const auto bytes = std::filesystem::file_size(stage.value().path() / "index.bin");
        std::ofstream out(stage.value().path() / "meta.txt");
        out << config << "index_bytes=" << bytes << '\n';
        out.close();
        if (!out) return Status::Internal("IVF HNSW metadata write failed");
        const auto status = stage.value().Publish();
        if (!status.ok()) {
          // A concurrent builder may have published the same immutable recipe.
          if (std::filesystem::exists(destination)) return Open(roots, o);
          return status;
        }
      }
    }
    if (!destination.empty()) impl->index_path = destination / "index.bin";
    impl->graph->setEf(std::max(8U, o.ef_search));
    return std::unique_ptr<IvfHnswRouter>(new IvfHnswRouter(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("IVF HNSW graph allocation failed");
  } catch (const std::exception& e) {
    return Status::Internal(std::string("IVF HNSW open/build failed: ") + e.what());
  }
}
Status IvfHnswRouter::RouteQueryTrusted(
    const float* query, std::uint32_t nprobe,
    std::uint32_t* routes, float* root_dots) const noexcept {
  return impl_->RouteQuery(query, nprobe, routes, root_dots);
}
Status IvfHnswRouter::RouteBlock(
    const TrustedQueryBatchView& queries, std::uint32_t nprobe,
    const ParallelForExecutor& parallel, std::vector<std::uint32_t>& routes,
    float* dots, std::vector<Status>& failures) const {
  try {
    if (nprobe == 0U || nprobe > impl_->roots.columns || queries.dimension() != impl_->roots.rows)
      return Status::InvalidArgument("IVF HNSW query shape or nprobe is invalid");
    routes.resize(static_cast<std::size_t>(queries.count()) * nprobe);
    failures.assign(parallel.worker_count(), Status::Ok());
    Impl::Block block{impl_.get(), &queries, nprobe, routes.data(), dots, &failures};
    parallel.Run(queries.count(), &block, Impl::Query);
    for (const auto& failure : failures) if (!failure.ok()) return failure;
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("IVF HNSW route block allocation failed");
  }
}
}  // namespace stlqpp
