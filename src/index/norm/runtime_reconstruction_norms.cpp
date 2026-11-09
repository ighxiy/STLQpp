#include "stlqpp/model/model_shape.h"
#include "stlqpp/index/norm/runtime_reconstruction_norms.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace stlqpp {
namespace {

template <typename Input>
Status Validate(const ModelShape& shape, const Input& input) {
  const auto& codes = input.codes;
  const std::uint64_t reals =
      std::uint64_t{codes.real_root_count} + codes.linked_real_count;
  const std::uint64_t nodes = reals + codes.virtual_count;
  if (codes.cid >= shape.nlist || shape.m < 2U || shape.m > stlqpp::kMaximumModelLayers ||
      nodes > std::numeric_limits<std::uint32_t>::max() ||
      input.parent_local_1based.size() != reals ||
      input.coefficients_layer_major.size() != nodes * shape.m ||
      codes.virtual_codes_small.size() !=
          std::uint64_t{codes.virtual_count} * (shape.m - 1U) ||
      codes.real_codes_small.size() != reals * (shape.m - 1U) ||
      codes.edge_code0.size() != reals) {
    return Status::InvalidArgument(
        "runtime reconstruction norm shape mismatch");
  }
  for (std::uint32_t real = 0; real < reals; ++real) {
    const auto parent = input.parent_local_1based[real];
    if ((real < codes.real_root_count && parent != 0U) ||
        (real >= codes.real_root_count &&
         (parent == 0U || parent > codes.virtual_count + real))) {
      return Status::InvalidArgument(
          "runtime reconstruction parent is not causal");
    }
  }
  return Status::Ok();
}

template <typename Input>
StatusOr<std::vector<float>> Reconstruct(RuntimeProductCodebookView model,
                                         const Input& input, ArrayView<const float> reference,
                                         ArrayView<double> vector_sum) {
  if ((!reference.empty() && reference.size() != model.shape.dimension) ||
      (!vector_sum.empty() && vector_sum.size() != model.shape.dimension))
    return Status::InvalidArgument("runtime norm reference dimension mismatch");
  if (const auto s = ValidateRuntimeNormInput(model.shape, input); !s.ok())
    return s;
  const auto& codes = input.codes;
  const auto m = model.shape.m;
  const auto d = model.shape.dimension;
  const auto reals = codes.real_root_count + codes.linked_real_count;
  const auto nodes = codes.virtual_count + reals;
  const auto roots = codes.virtual_count + codes.real_root_count;
  std::vector<float> recon(static_cast<std::size_t>(nodes) * d, 0.0F);
  for (std::uint32_t node = 0; node < nodes; ++node) {
    const bool root = node < roots;
    const bool is_virtual = node < codes.virtual_count;
    const auto real = node - codes.virtual_count;
    float* destination = recon.data() + static_cast<std::size_t>(node) * d;
    if (!root) {
      const auto* parent =
          recon.data() +
          static_cast<std::size_t>(input.parent_local_1based[real] - 1U) * d;
      std::copy_n(parent, d, destination);
    }
    for (std::uint32_t layer = 0; layer < m; ++layer) {
      float weight = static_cast<float>(
          input.coefficients_layer_major[static_cast<std::size_t>(layer) *
                                             nodes +
                                         node]);
      if constexpr (std::is_same_v<Input, RuntimeReconstructionNormInput>) {
        weight *= root ? input.root_scales[layer] : input.linkage_scales[layer];
      }
      const std::uint8_t code =
          layer == 0U
              ? (root ? 0U : codes.edge_code0[real])
              : (is_virtual
                     ? codes
                           .virtual_codes_small[static_cast<std::size_t>(node) *
                                                    (m - 1U) +
                                                layer - 1U]
                     : codes.real_codes_small[static_cast<std::size_t>(real) *
                                                  (m - 1U) +
                                              layer - 1U]);
      const float* atom =
          root ? (layer == 0U ? model.root0.column(codes.cid)
                              : model.root_small(layer).column(code))
               : model.edge(layer).column(code);
      for (std::uint32_t axis = 0; axis < d; ++axis)
        destination[axis] += weight * atom[axis];
    }
  }
  std::vector<float> result(reals);
  for (std::uint32_t real = 0; real < reals; ++real) {
    const auto* row =
        recon.data() + static_cast<std::size_t>(codes.virtual_count + real) * d;
    float norm = 0.0F;
    for (std::uint32_t axis = 0; axis < d; ++axis) {
      const float value = reference.empty() ? row[axis] : row[axis] - reference[axis];
      norm += value * value;
    }
    if (!vector_sum.empty())
      for (std::uint32_t axis = 0; axis < d; ++axis) vector_sum[axis] += row[axis];
    if (!std::isfinite(norm))
      return Status::DataLoss("runtime reconstruction norm overflow");
    result[real] = norm;
  }
  return result;
}

}  // namespace

Status ValidateRuntimeNormInput(const ModelShape& shape,
                                const RuntimeReconstructionNormInput& input) {
  if (const auto s = Validate(shape, input); !s.ok()) return s;
  if (input.root_scales.size() != shape.m ||
      input.linkage_scales.size() != shape.m)
    return Status::InvalidArgument(
        "runtime reconstruction scale count mismatch");
  for (const auto scales : {input.root_scales, input.linkage_scales}) {
    for (const auto value : scales) {
      if (!std::isfinite(value))
        return Status::InvalidArgument(
            "runtime reconstruction scale is not finite");
    }
  }
  return Status::Ok();
}

Status ValidateRuntimeNormInput(
    const ModelShape& shape, const RuntimeFloatReconstructionNormInput& input) {
  if (const auto s = Validate(shape, input); !s.ok()) return s;
  for (const auto value : input.coefficients_layer_major) {
    if (!std::isfinite(value))
      return Status::InvalidArgument("runtime float coefficient is not finite");
  }
  return Status::Ok();
}

StatusOr<std::vector<float>> RuntimeReconstructionNormComputer::Compute(
    const RuntimeReconstructionNormInput& input,
    ArrayView<const float> reference, ArrayView<double> vector_sum) const {
  return Reconstruct(model_, input, reference, vector_sum);
}

StatusOr<std::vector<float>> RuntimeReconstructionNormComputer::ComputeFloat(
    const RuntimeFloatReconstructionNormInput& input,
    ArrayView<const float> reference, ArrayView<double> vector_sum) const {
  return Reconstruct(model_, input, reference, vector_sum);
}

namespace {
template <class Input>
StatusOr<std::vector<double>> Project(
    const ModelShape& shape, const std::vector<double>& root0,
    const std::vector<double>& small, const std::vector<double>& edge,
    const Input& input) {
  if (const auto status = ValidateRuntimeNormInput(shape, input); !status.ok())
    return status;
  const auto& codes = input.codes;
  const auto reals = codes.real_root_count + codes.linked_real_count;
  const auto nodes = codes.virtual_count + reals;
  const auto roots = codes.virtual_count + codes.real_root_count;
  const auto m = shape.m;
  std::vector<double> dot(nodes, 0.0);
  for (std::uint32_t node = 0; node < nodes; ++node) {
    const bool root = node < roots, virt = node < codes.virtual_count;
    const auto real = node - codes.virtual_count;
    double value = root ? 0.0 : dot[input.parent_local_1based[real] - 1U];
    for (std::uint32_t layer = 0; layer < m; ++layer) {
      float weight = static_cast<float>(
          input.coefficients_layer_major[std::size_t(layer) * nodes + node]);
      if constexpr (std::is_same_v<Input, RuntimeReconstructionNormInput>)
        weight *= root ? input.root_scales[layer] : input.linkage_scales[layer];
      const auto code = layer == 0U ? (root ? 0U : codes.edge_code0[real])
          : virt ? codes.virtual_codes_small[std::size_t(node)*(m-1U)+layer-1U]
                 : codes.real_codes_small[std::size_t(real)*(m-1U)+layer-1U];
      const auto atom = root ? (layer == 0U ? root0[codes.cid]
          : small[std::size_t(layer-1U)*256U+code])
          : edge[std::size_t(layer)*256U+code];
      value += weight * atom;
    }
    if (!std::isfinite(value)) return Status::DataLoss("reference projection overflow");
    dot[node] = value;
  }
  return std::vector<double>(dot.begin() + codes.virtual_count, dot.end());
}

template <class Input>
Status AccumulateMean(const ModelShape& shape, const Input& input,
                      std::vector<double>& root0,
                      std::vector<double>& small, std::vector<double>& edge,
                      std::uint64_t& real_count) {
  if (const auto status = ValidateRuntimeNormInput(shape, input); !status.ok())
    return status;
  const auto& codes = input.codes;
  const auto reals = codes.real_root_count + codes.linked_real_count;
  const auto nodes = codes.virtual_count + reals;
  const auto roots = codes.virtual_count + codes.real_root_count;
  const auto m = shape.m;
  std::vector<std::uint64_t> descendants(nodes, 1U);
  std::fill_n(descendants.begin(), codes.virtual_count, 0U);
  for (std::uint32_t real = reals; real > codes.real_root_count;) {
    --real;
    descendants[input.parent_local_1based[real] - 1U] +=
        descendants[codes.virtual_count + real];
  }
  for (std::uint32_t node = 0; node < nodes; ++node) {
    const bool root = node < roots, virt = node < codes.virtual_count;
    const auto real = node - codes.virtual_count;
    for (std::uint32_t layer = 0; layer < m; ++layer) {
      float weight = static_cast<float>(
          input.coefficients_layer_major[std::size_t(layer) * nodes + node]);
      if constexpr (std::is_same_v<Input, RuntimeReconstructionNormInput>)
        weight *= root ? input.root_scales[layer] : input.linkage_scales[layer];
      const double sum = double(weight) * double(descendants[node]);
      if (root && layer == 0U) {
        root0[codes.cid] += sum;
      } else {
        const auto code = layer == 0U ? codes.edge_code0[real]
            : virt ? codes.virtual_codes_small[std::size_t(node)*(m-1U)+layer-1U]
                   : codes.real_codes_small[std::size_t(real)*(m-1U)+layer-1U];
        auto& weights = root ? small : edge;
        weights[std::size_t(root ? layer-1U : layer)*256U+code] += sum;
      }
    }
  }
  real_count += reals;
  return Status::Ok();
}
}  // namespace

StatusOr<RuntimeReconstructionProjectionComputer>
RuntimeReconstructionProjectionComputer::Create(
    RuntimeProductCodebookView model, ArrayView<const float> reference) {
  if (reference.size() != model.shape.dimension)
    return Status::InvalidArgument("reference projection dimension mismatch");
  for (auto value : reference)
    if (!std::isfinite(value)) return Status::DataLoss("nonfinite reference vector");
  RuntimeReconstructionProjectionComputer out;
  out.shape_ = model.shape;
  const auto project = [&](ColumnMajorMatrixF32View atoms, auto& output) {
    output.resize(atoms.columns);
    for (std::uint32_t col = 0; col < atoms.columns; ++col) {
      double value = 0.0;
      for (std::uint32_t axis = 0; axis < atoms.rows; ++axis)
        value += double(reference[axis]) * atoms.column(col)[axis];
      output[col] = value;
    }
  };
  project(model.root0, out.root0_);
  project(model.root_small_flat, out.root_small_);
  project(model.edge_flat, out.edge_);
  return out;
}
StatusOr<std::vector<double>> RuntimeReconstructionProjectionComputer::Compute(
    const RuntimeReconstructionNormInput& input) const {
  return Project(shape_, root0_, root_small_, edge_, input);
}
StatusOr<std::vector<double>> RuntimeReconstructionProjectionComputer::ComputeFloat(
    const RuntimeFloatReconstructionNormInput& input) const {
  return Project(shape_, root0_, root_small_, edge_, input);
}

RuntimeReconstructionMeanAccumulator::RuntimeReconstructionMeanAccumulator(
    RuntimeProductCodebookView model)
    : model_(model), root0_(model.root0.columns, 0.0),
      root_small_(model.root_small_flat.columns, 0.0),
      edge_(model.edge_flat.columns, 0.0) {}

Status RuntimeReconstructionMeanAccumulator::Accumulate(
    const RuntimeReconstructionNormInput& input) {
  return AccumulateMean(model_.shape, input, root0_, root_small_, edge_, real_count_);
}

Status RuntimeReconstructionMeanAccumulator::AccumulateFloat(
    const RuntimeFloatReconstructionNormInput& input) {
  return AccumulateMean(model_.shape, input, root0_, root_small_, edge_, real_count_);
}

StatusOr<std::vector<float>> RuntimeReconstructionMeanAccumulator::Finish() const {
  std::vector<double> sum(model_.shape.dimension, 0.0);
  const auto reduce = [&](ColumnMajorMatrixF32View atoms, const auto& weights) {
    for (std::uint32_t col = 0; col < atoms.columns; ++col)
      for (std::uint32_t axis = 0; axis < atoms.rows; ++axis)
        sum[axis] += weights[col] * double(atoms.column(col)[axis]);
  };
  if (real_count_ != 0U) {
    reduce(model_.root0, root0_);
    reduce(model_.root_small_flat, root_small_);
    reduce(model_.edge_flat, edge_);
  }
  std::vector<float> result(sum.size(), 0.0F);
  for (std::size_t axis = 0; axis < sum.size(); ++axis) {
    if (real_count_ != 0U)
      result[axis] = static_cast<float>(sum[axis] / double(real_count_));
    if (!std::isfinite(result[axis]))
      return Status::DataLoss("reference mean overflow");
  }
  return result;
}

}  // namespace stlqpp
