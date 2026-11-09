#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "stlqpp/index/norm/reconstruction_norms_cuda.h"
#include "stlqpp/runtime/cuda/context.h"
#include "stlqpp/runtime/cuda/gemm_f32.h"

namespace stlqpp {
namespace {
Status CudaStatus(cudaError_t e, const char* op) {
  return e == cudaSuccess
             ? Status::Ok()
             : Status::Internal(std::string(op) + ": " + cudaGetErrorString(e));
}
template <class T>
class DeviceBuffer final {
 public:
  ~DeviceBuffer() { cudaFree(data_); }
  Status Reserve(std::size_t n) {
    if (n <= capacity_) return Status::Ok();
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      return Status::ResourceExhausted("CUDA norm extent overflow");
    T* replacement = nullptr;
    auto s = CudaStatus(
        cudaMalloc(reinterpret_cast<void**>(&replacement), n * sizeof(T)),
        "allocate CUDA norm workspace");
    if (!s.ok()) return s;
    cudaFree(data_);
    data_ = replacement;
    capacity_ = n;
    return Status::Ok();
  }
  Status Upload(const T* src, std::size_t n, cudaStream_t stream) {
    if (!n) return Status::Ok();
    auto s = Reserve(n);
    if (!s.ok()) return s;
    return CudaStatus(cudaMemcpyAsync(data_, src, n * sizeof(T),
                                      cudaMemcpyHostToDevice, stream),
                      "upload CUDA norm data");
  }
  T* data() const { return data_; }
  std::size_t bytes() const { return capacity_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0;
};
struct StreamDrain final {
  CudaContext& context;
  ~StreamDrain() { (void)context.Synchronize(); }
};
struct DeviceGrams {
  std::uint32_t m;
  const float* small;
  const float* edge;
  const float* small_edge;
  const float* cross;  // c0 norm, c0-small, c0-edge, one CID only.
};
__global__ void RootCross(const float* root0, const float* small,
                          const float* edge, std::uint32_t cid, std::uint32_t d,
                          std::uint32_t small_n, float* cross) {
  const auto atom = blockIdx.x;
  const float* root = root0 + std::size_t(cid) * d;
  const float* other = atom == 0 ? root
                       : atom <= small_n
                           ? small + std::size_t(atom - 1) * d
                           : edge + std::size_t(atom - small_n - 1) * d;
  float sum = 0;
  for (auto axis = threadIdx.x; axis < d; axis += blockDim.x)
    sum += root[axis] * other[axis];
  __shared__ float values[128];
  values[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned step = 64; step; step /= 2) {
    if (threadIdx.x < step) values[threadIdx.x] += values[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0) cross[atom] = values[0];
}
__device__ unsigned SmallCode(unsigned node, unsigned layer, unsigned m,
                              unsigned virt, const std::uint8_t* v,
                              const std::uint8_t* r) {
  return node < virt ? v[std::size_t(node) * (m - 1) + layer - 1]
                     : r[std::size_t(node - virt) * (m - 1) + layer - 1];
}
__device__ unsigned Atom(unsigned node, unsigned layer, bool root, unsigned m,
                         unsigned virt, const std::uint8_t* v,
                         const std::uint8_t* r, const std::uint8_t* e0) {
  if (root)
    return layer == 0
               ? 0
               : (layer - 1) * 256 + SmallCode(node, layer, m, virt, v, r);
  return layer * 256 +
         (layer == 0 ? e0[node - virt] : SmallCode(node, layer, m, virt, v, r));
}
template <class C>
__device__ float Weight(const C* q, const float* scales, unsigned node,
                        unsigned layer, unsigned nodes) {
  if constexpr (std::is_same_v<C, float>)
    return q[std::size_t(layer) * nodes + node];
  else
    return float(q[std::size_t(layer) * nodes + node]) * scales[layer];
}
__device__ float RootGram(DeviceGrams g, unsigned l, unsigned a, unsigned k,
                          unsigned b) {
  if (l == 0 && k == 0) return g.cross[0];
  if (l == 0 || k == 0) return g.cross[1 + (l == 0 ? b : a)];
  return g.small[std::size_t(b) * (g.m - 1) * 256 + a];
}
template <int FixedM, std::uint32_t Capacity, class C>
__global__ void NormRoots(DeviceGrams g, const C* q, const float* scales,
                          const std::uint8_t* vc, const std::uint8_t* rc,
                          unsigned virt, unsigned roots, unsigned nodes,
                          float* norms, const double* proj0,
                          const double* proj_small, unsigned cid,
                          double* projections) {
  const unsigned node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= roots) return;
  const unsigned m = FixedM ? FixedM : g.m;
  float weights[Capacity];
  unsigned atoms[Capacity];
  for (unsigned l = 0; l < m; ++l) {
    weights[l] = Weight(q, scales, node, l, nodes);
    atoms[l] = Atom(node, l, true, m, virt, vc, rc, nullptr);
  }
  float value = 0;
  for (unsigned l = 0; l < m; ++l) {
    value += weights[l] * weights[l] * RootGram(g, l, atoms[l], l, atoms[l]);
    for (unsigned k = l + 1; k < m; ++k)
      value += 2.0F * weights[l] * weights[k] *
               RootGram(g, l, atoms[l], k, atoms[k]);
  }
  norms[node] = value < 0 ? 0 : value;
  if (projections) {
    double dot = double(weights[0]) * proj0[cid];
    for (unsigned l = 1; l < m; ++l)
      dot += double(weights[l]) * proj_small[atoms[l]];
    projections[node] = dot;
  }
}
template <int FixedM, std::uint32_t Capacity, class C>
__global__ void NormLinkedDepth(
    DeviceGrams g, const C* q, const float* root_scales,
    const float* link_scales, const std::uint8_t* vc, const std::uint8_t* rc,
    const std::uint8_t* e0, const std::uint32_t* parents, unsigned virt,
    unsigned roots, unsigned nodes, unsigned begin, unsigned end, float* norms,
    const double* proj_edge, double* projections) {
  const unsigned node = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= end) return;
  const unsigned m = FixedM ? FixedM : g.m, edge_n = m * 256,
                 small_n = (m - 1) * 256;
  float weights[Capacity];
  unsigned atoms[Capacity];
  for (unsigned l = 0; l < m; ++l) {
    weights[l] = Weight(q, link_scales, node, l, nodes);
    atoms[l] = Atom(node, l, false, m, virt, vc, rc, e0);
  }
  float local = 0;
  for (unsigned l = 0; l < m; ++l) {
    local += weights[l] * weights[l] *
             g.edge[std::size_t(atoms[l]) * edge_n + atoms[l]];
    for (unsigned k = l + 1; k < m; ++k)
      local += 2.0F * weights[l] * weights[k] *
               g.edge[std::size_t(atoms[k]) * edge_n + atoms[l]];
  }
  const unsigned parent = parents[node - virt] - 1;
  float cross = 0;
  unsigned ancestor = parent;
  for (;;) {
    const bool root = ancestor < roots;
    for (unsigned l = 0; l < m; ++l) {
      const unsigned atom = Atom(ancestor, l, root, m, virt, vc, rc, e0);
      const float a =
          Weight(q, root ? root_scales : link_scales, ancestor, l, nodes);
      if (root) {
        float dot = 0;
        for (unsigned k = 0; k < m; ++k)
          dot +=
              weights[k] *
              (l == 0 ? g.cross[1 + small_n + atoms[k]]
                      : g.small_edge[std::size_t(atoms[k]) * small_n + atom]);
        cross += a * dot;
      } else {
        for (unsigned k = 0; k < m; ++k)
          cross +=
              a * weights[k] * g.edge[std::size_t(atoms[k]) * edge_n + atom];
      }
    }
    if (root) break;
    ancestor = parents[ancestor - virt] - 1;
  }
  const float value = norms[parent] + local + 2.0F * cross;
  norms[node] = value < 0 ? 0 : value;
  if (projections) {
    double dot = projections[parent];
    for (unsigned l = 0; l < m; ++l)
      dot += double(weights[l]) * proj_edge[atoms[l]];
    projections[node] = dot;
  }
}
__global__ void ApplyReference(const float* norms, const double* projections,
                               unsigned virt, unsigned n, double ref_norm,
                               float* out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double v = double(norms[virt + i]) - 2.0 * projections[virt + i] + ref_norm;
    out[i] = v < 0 ? 0 : float(v);
  }
}
}  // namespace

struct RuntimeCudaReconstructionNormComputer::Model final {
  ModelShape shape;
  int device = 0;
  bool has_atoms = false, has_reference = false;
  DeviceBuffer<float> root0, small_atoms, edge_atoms, small, edge, small_edge;
  DeviceBuffer<double> projection0, projection_small, projection_edge;
  double reference_norm = 0;
  ~Model() { cudaSetDevice(device); }
};
struct RuntimeCudaReconstructionNormComputer::State final {
  std::shared_ptr<const Model> model;
  CudaContext context;
  DeviceBuffer<float> cross, root_scales, linkage_scales, float_coefficients,
      norms, reference_norms;
  DeviceBuffer<std::int8_t> coefficients;
  DeviceBuffer<std::uint8_t> virtual_codes, real_codes, edge_code0;
  DeviceBuffer<std::uint32_t> parents;
  DeviceBuffer<double> projections;
};
RuntimeCudaReconstructionNormComputer::
    ~RuntimeCudaReconstructionNormComputer() {
  if (state_) {
    cudaSetDevice(state_->model->device);
    (void)state_->context.Synchronize();
  }
}
StatusOr<std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
RuntimeCudaReconstructionNormComputer::Create(RuntimeProductCodebookView books,
                                              int device,
                                              ArrayView<const float> ref) {
  if (auto s = books.Validate(); !s.ok()) return s;
  if (!ref.empty() && ref.size() != books.shape.dimension)
    return Status::InvalidArgument("CUDA norm reference shape");
  for (float x : ref)
    if (!std::isfinite(x))
      return Status::InvalidArgument("nonfinite CUDA norm reference");
  auto context = CudaContext::Create(device, false);
  if (!context.ok()) return context.status();
  auto out = std::unique_ptr<RuntimeCudaReconstructionNormComputer>(
      new RuntimeCudaReconstructionNormComputer);
  out->state_ = std::make_unique<State>();
  out->state_->context = std::move(context).value();
  auto model = std::make_shared<Model>();
  model->device = device;
  model->shape = books.shape;
  model->has_atoms = true;
  out->state_->model = model;
  auto stream = static_cast<cudaStream_t>(out->state_->context.native_stream());
  const StreamDrain drain{out->state_->context};
  const auto small_n = (books.shape.m - 1) * 256, edge_n = books.shape.m * 256,
             d = books.shape.dimension;
  for (auto s :
       {model->root0.Upload(books.root0.values,
                            std::size_t(books.shape.nlist) * d, stream),
        model->small_atoms.Upload(books.root_small_flat.values,
                                  std::size_t(small_n) * d, stream),
        model->edge_atoms.Upload(books.edge_flat.values,
                                 std::size_t(edge_n) * d, stream),
        model->small.Reserve(std::size_t(small_n) * small_n),
        model->edge.Reserve(std::size_t(edge_n) * edge_n),
        model->small_edge.Reserve(std::size_t(small_n) * edge_n)})
    if (!s.ok()) return s;
  const auto multiply = [&](const float* a, const float* b, unsigned rows,
                            unsigned cols, float* dst) {
    return CudaGemmF32(out->state_->context, CudaGemmOperation::kTranspose,
                       CudaGemmOperation::kNone, int(rows), int(cols), int(d),
                       1.0F, a, int(d), b, int(d), 0.0F, dst, int(rows));
  };
  for (auto s : {multiply(model->small_atoms.data(), model->small_atoms.data(),
                          small_n, small_n, model->small.data()),
                 multiply(model->edge_atoms.data(), model->edge_atoms.data(),
                          edge_n, edge_n, model->edge.data()),
                 multiply(model->small_atoms.data(), model->edge_atoms.data(),
                          small_n, edge_n, model->small_edge.data())})
    if (!s.ok()) return s;
  if (!ref.empty()) {
    model->has_reference = true;
    for (float x : ref) model->reference_norm += double(x) * x;
    const auto upload_projection = [&](ColumnMajorMatrixF32View a,
                                       DeviceBuffer<double>& dst) {
      std::vector<double> dots(a.columns, 0.0);
      for (unsigned col = 0; col < a.columns; ++col)
        for (unsigned axis = 0; axis < d; ++axis)
          dots[col] += double(ref[axis]) * a.column(col)[axis];
      auto s = dst.Upload(dots.data(), dots.size(), stream);
      if (!s.ok()) return s;
      return out->state_->context.Synchronize();
    };
    for (auto s :
         {upload_projection(books.root0, model->projection0),
          upload_projection(books.root_small_flat, model->projection_small),
          upload_projection(books.edge_flat, model->projection_edge)})
      if (!s.ok()) return s;
  }
  if (auto s = out->state_->context.Synchronize(); !s.ok()) return s;
  return std::move(out);
}
StatusOr<std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
RuntimeCudaReconstructionNormComputer::CreateWithGrams(
    ReconstructionGramModelView g, int device) {
  const auto small_n = (g.shape.m - 1) * 256, edge_n = g.shape.m * 256;
  if (auto s = g.shape.Validate(); !s.ok()) return s;
  if (g.root_small.size() != std::size_t(small_n) * small_n ||
      g.edge.size() != std::size_t(edge_n) * edge_n ||
      g.root_small_edge.size() != std::size_t(small_n) * edge_n)
    return Status::InvalidArgument("CUDA norm Gram extents");
  auto context = CudaContext::Create(device, false);
  if (!context.ok()) return context.status();
  auto out = std::unique_ptr<RuntimeCudaReconstructionNormComputer>(
      new RuntimeCudaReconstructionNormComputer);
  out->state_ = std::make_unique<State>();
  out->state_->context = std::move(context).value();
  auto model = std::make_shared<Model>();
  model->shape = g.shape;
  model->device = device;
  out->state_->model = model;
  auto stream = static_cast<cudaStream_t>(out->state_->context.native_stream());
  const StreamDrain drain{out->state_->context};
  for (auto s :
       {model->small.Upload(g.root_small.data(), g.root_small.size(), stream),
        model->edge.Upload(g.edge.data(), g.edge.size(), stream),
        model->small_edge.Upload(g.root_small_edge.data(),
                                 g.root_small_edge.size(), stream)})
    if (!s.ok()) return s;
  if (auto s = out->state_->context.Synchronize(); !s.ok()) return s;
  return std::move(out);
}
StatusOr<std::unique_ptr<RuntimeCudaReconstructionNormComputer>>
RuntimeCudaReconstructionNormComputer::Clone() const {
  auto context = CudaContext::Create(state_->model->device, false);
  if (!context.ok()) return context.status();
  auto out = std::unique_ptr<RuntimeCudaReconstructionNormComputer>(
      new RuntimeCudaReconstructionNormComputer);
  out->state_ = std::make_unique<State>();
  out->state_->model = state_->model;
  out->state_->context = std::move(context).value();
  return std::move(out);
}
StatusOr<std::vector<float>> RuntimeCudaReconstructionNormComputer::Compute(
    const RuntimeReconstructionNormInput& x, ReconstructionRootCrossView c) {
  return ComputeImpl(x, c);
}
StatusOr<std::vector<float>>
RuntimeCudaReconstructionNormComputer::ComputeFloat(
    const RuntimeFloatReconstructionNormInput& x,
    ReconstructionRootCrossView c) {
  return ComputeImpl(x, c);
}
const void* RuntimeCudaReconstructionNormComputer::shared_model_identity()
    const noexcept {
  return state_->model.get();
}
std::size_t RuntimeCudaReconstructionNormComputer::workspace_bytes()
    const noexcept {
  const auto& s = *state_;
  return s.cross.bytes() + s.root_scales.bytes() + s.linkage_scales.bytes() +
         s.float_coefficients.bytes() + s.norms.bytes() +
         s.reference_norms.bytes() + s.coefficients.bytes() +
         s.virtual_codes.bytes() + s.real_codes.bytes() + s.edge_code0.bytes() +
         s.parents.bytes() + s.projections.bytes();
}
template <class Input>
StatusOr<std::vector<float>> RuntimeCudaReconstructionNormComputer::ComputeImpl(
    const Input& input, ReconstructionRootCrossView cross) {
  using C = std::remove_cv_t<
      std::remove_reference_t<decltype(input.coefficients_layer_major[0])>>;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& s = *state_;
  const auto& model = *s.model;
  if (auto x = CudaStatus(cudaSetDevice(model.device),
                          "select CUDA Gram norm device");
      !x.ok())
    return x;
  if (auto x = ValidateRuntimeNormInput(model.shape, input); !x.ok()) return x;
  const auto m = model.shape.m, small_n = (m - 1) * 256;
  const auto& c = input.codes;
  const auto reals = c.real_root_count + c.linked_real_count,
             roots = c.virtual_count + c.real_root_count,
             nodes = c.virtual_count + reals;
  if (!reals) return std::vector<float>{};
  std::vector<std::pair<unsigned, unsigned>> levels;
  std::vector<unsigned> depths(nodes, 0);
  for (unsigned real = c.real_root_count; real < reals; ++real) {
    unsigned node = c.virtual_count + real,
             depth = depths[input.parent_local_1based[real] - 1] + 1;
    if (depth < levels.size())
      return Status::InvalidArgument("CUDA norm input is not BFS ordered");
    if (depth > levels.size()) levels.emplace_back(node, node);
    levels.back().second = node + 1;
    depths[node] = depth;
  }
  std::vector<float> out(reals);
  const StreamDrain drain{s.context};
  auto stream = static_cast<cudaStream_t>(s.context.native_stream());
  const auto upload = [&](auto& dst, const auto& source) {
    return dst.Upload(source.data(), source.size(), stream);
  };
  if (auto x = s.cross.Reserve(1 + (2 * m - 1) * 256); !x.ok()) return x;
  if (!cross.small.empty() || !cross.edge.empty()) {
    if (cross.small.size() != small_n || cross.edge.size() != m * 256)
      return Status::InvalidArgument("CUDA CID cross extents");
    if (auto x = CudaStatus(
            cudaMemcpyAsync(s.cross.data(), &cross.norm2, sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "upload root energy");
        !x.ok())
      return x;
    if (auto x =
            CudaStatus(cudaMemcpyAsync(s.cross.data() + 1, cross.small.data(),
                                       cross.small.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "upload root-small cross");
        !x.ok())
      return x;
    if (auto x = CudaStatus(
            cudaMemcpyAsync(s.cross.data() + 1 + small_n, cross.edge.data(),
                            cross.edge.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "upload root-edge cross");
        !x.ok())
      return x;
  } else {
    if (!model.has_atoms)
      return Status::InvalidArgument(
          "shared Gram norm requires prepared CID cross tables");
    RootCross<<<1 + (2 * m - 1) * 256, 128, 0, stream>>>(
        model.root0.data(), model.small_atoms.data(), model.edge_atoms.data(),
        c.cid, model.shape.dimension, small_n, s.cross.data());
  }
  for (auto x :
       {upload(s.virtual_codes, c.virtual_codes_small),
        upload(s.real_codes, c.real_codes_small),
        upload(s.edge_code0, c.edge_code0),
        upload(s.parents, input.parent_local_1based), s.norms.Reserve(nodes)})
    if (!x.ok()) return x;
  auto& q = [&]() -> auto& {
    if constexpr (std::is_same_v<C, float>)
      return s.float_coefficients;
    else
      return s.coefficients;
  }();
  if (auto x = upload(q, input.coefficients_layer_major); !x.ok()) return x;
  if constexpr (!std::is_same_v<C, float>)
    for (auto x : {upload(s.root_scales, input.root_scales),
                   upload(s.linkage_scales, input.linkage_scales)})
      if (!x.ok()) return x;
  if (model.has_reference) {
    if (auto x = s.projections.Reserve(nodes); !x.ok()) return x;
    if (auto x = s.reference_norms.Reserve(reals); !x.ok()) return x;
  }
  DeviceGrams g{m, model.small.data(), model.edge.data(),
                model.small_edge.data(), s.cross.data()};
  const auto launch = [&](auto tag, auto capacity_tag) {
    constexpr int FixedM = decltype(tag)::value;
    constexpr auto Capacity = decltype(capacity_tag)::value;
    if (roots)
      NormRoots<FixedM, Capacity, C><<<(roots + 127) / 128, 128, 0, stream>>>(
          g, q.data(), s.root_scales.data(), s.virtual_codes.data(),
          s.real_codes.data(), c.virtual_count, roots, nodes, s.norms.data(),
          model.projection0.data(), model.projection_small.data(), c.cid,
          model.has_reference ? s.projections.data() : nullptr);
    for (const auto& level : levels)
      NormLinkedDepth<FixedM, Capacity, C>
          <<<(level.second - level.first + 127) / 128, 128, 0, stream>>>(
              g, q.data(), s.root_scales.data(), s.linkage_scales.data(),
              s.virtual_codes.data(), s.real_codes.data(), s.edge_code0.data(),
              s.parents.data(), c.virtual_count, roots, nodes, level.first,
              level.second, s.norms.data(), model.projection_edge.data(),
              model.has_reference ? s.projections.data() : nullptr);
  };
  switch (m) {
    case 5:
      launch(std::integral_constant<int, 5>{}, std::integral_constant<unsigned, 5>{});
      break;
    case 6:
      launch(std::integral_constant<int, 6>{}, std::integral_constant<unsigned, 6>{});
      break;
    case 10:
      launch(std::integral_constant<int, 10>{}, std::integral_constant<unsigned, 10>{});
      break;
    case 12:
      launch(std::integral_constant<int, 12>{}, std::integral_constant<unsigned, 12>{});
      break;
    default:
      if (m > kLegacyRuntimeScratchLayers)
        launch(std::integral_constant<int, 0>{}, std::integral_constant<unsigned, kWideRuntimeScratchLayers>{});
      else
        launch(std::integral_constant<int, 0>{}, std::integral_constant<unsigned, kLegacyRuntimeScratchLayers>{});
  }
  if (model.has_reference)
    ApplyReference<<<(reals + 127) / 128, 128, 0, stream>>>(
        s.norms.data(), s.projections.data(), c.virtual_count, reals,
        model.reference_norm, s.reference_norms.data());
  if (auto x = CudaStatus(cudaGetLastError(), "launch Gram causal norm");
      !x.ok())
    return x;
  if (auto x = CudaStatus(
          cudaMemcpyAsync(
              out.data(),
              model.has_reference ? s.reference_norms.data()
                                  : s.norms.data() + c.virtual_count,
              out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
          "download Gram norms");
      !x.ok())
    return x;
  if (auto x = s.context.Synchronize(); !x.ok()) return x;
  for (float x : out)
    if (!std::isfinite(x) || x < 0)
      return Status::DataLoss("nonfinite CUDA Gram norm");
  return std::move(out);
}
StatusOr<std::unique_ptr<CudaReconstructionNormPool>>
CudaReconstructionNormPool::Create(ReconstructionGramModelView g, int device,
                                   std::uint32_t lanes) {
  auto leases = ExclusiveIndexPool::Create(lanes);
  if (!leases.ok()) return leases.status();
  auto first =
      RuntimeCudaReconstructionNormComputer::CreateWithGrams(g, device);
  if (!first.ok()) return first.status();
  auto out = std::make_unique<CudaReconstructionNormPool>();
  out->leases_ = std::move(leases).value();
  out->workers_.push_back(std::move(first).value());
  for (unsigned i = 1; i < lanes; ++i) {
    auto clone = out->workers_.front()->Clone();
    if (!clone.ok()) return clone.status();
    out->workers_.push_back(std::move(clone).value());
  }
  return std::move(out);
}
StatusOr<std::vector<float>> CudaReconstructionNormPool::Compute(
    const RuntimeReconstructionNormInput& x,
    ReconstructionRootCrossView cross) {
  auto index = leases_->Acquire();
  if (!index.ok()) return index.status();
  const auto i = index.value();
  auto result = workers_[i]->Compute(x, cross);
  leases_->Release(i);
  return result;
}
StatusOr<std::vector<float>> CudaReconstructionNormPool::ComputeFloat(
    const RuntimeFloatReconstructionNormInput& x,
    ReconstructionRootCrossView cross) {
  auto index = leases_->Acquire();
  if (!index.ok()) return index.status();
  const auto i = index.value();
  auto result = workers_[i]->ComputeFloat(x, cross);
  leases_->Release(i);
  return result;
}
template <int M>
CudaReconstructionNormComputer<M>::~CudaReconstructionNormComputer() = default;
template <int M>
StatusOr<std::unique_ptr<CudaReconstructionNormComputer<M>>>
CudaReconstructionNormComputer<M>::Create(const ProductCodebooks<M>& books,
                                          int device) {
  auto r = RuntimeCudaReconstructionNormComputer::Create(books.runtime_view(),
                                                         device);
  if (!r.ok()) return r.status();
  auto out = std::unique_ptr<CudaReconstructionNormComputer>(
      new CudaReconstructionNormComputer);
  out->runtime_ = std::move(r).value();
  return std::move(out);
}
template <int M>
StatusOr<std::unique_ptr<CudaReconstructionNormComputer<M>>>
CudaReconstructionNormComputer<M>::Create(
    const SelectedAtomGramTables<M>& grams, int device) {
  auto r = CudaReconstructionNormPool::Create(grams.norm_model_view(), device);
  if (!r.ok()) return r.status();
  auto out = std::unique_ptr<CudaReconstructionNormComputer>(
      new CudaReconstructionNormComputer);
  out->pool_ = std::move(r).value();
  return std::move(out);
}
template <int M>
StatusOr<std::vector<float>> CudaReconstructionNormComputer<M>::Compute(
    const ReconstructionNormInput<M>& input,
    ReconstructionRootCrossView cross) {
  const auto& c = input.codes;
  RuntimeReconstructionNormInput x{
      {c.cid, c.virtual_count, c.real_root_count, c.linked_real_count,
       c.virtual_codes_small, c.real_codes_small, c.edge_code0},
      input.parent_local_1based,
      input.coefficients_layer_major,
      input.root_scales,
      input.linkage_scales};
  return pool_ ? pool_->Compute(x, cross) : runtime_->Compute(x, cross);
}
template <int M>
StatusOr<std::vector<float>> CudaReconstructionNormComputer<M>::ComputeFloat(
    const FloatReconstructionNormInput<M>& input,
    ReconstructionRootCrossView cross) {
  const auto& c = input.codes;
  RuntimeFloatReconstructionNormInput x{
      {c.cid, c.virtual_count, c.real_root_count, c.linked_real_count,
       c.virtual_codes_small, c.real_codes_small, c.edge_code0},
      input.parent_local_1based,
      input.coefficients_layer_major};
  return pool_ ? pool_->ComputeFloat(x, cross)
               : runtime_->ComputeFloat(x, cross);
}
#define INST(M) template class CudaReconstructionNormComputer<M>;
INST(0) INST(5) INST(6) INST(10) INST(12)
#undef INST
}  // namespace stlqpp
