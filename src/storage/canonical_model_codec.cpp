#include "stlqpp/storage/canonical_model_codec.h"

#include "stlqpp/core/numeric/little_endian.h"

#include "stlqpp/storage/artifact_hash_factory.h"
#ifdef STLQPP_MODEL_CBLAS
#include "stlqpp/runtime/cpu/cblas_f32_gemm.h"
#endif

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'S', 'T', 'L', 'Q', 'M', 'O', 'D', 'L',
};

StatusOr<std::uint64_t> ExpectedFloatCount(const ModelShape& shape, bool source_space = false) {
  const std::uint64_t dimension = shape.dimension;
  const std::uint64_t residual_columns =
      static_cast<std::uint64_t>(2U * shape.m - 1U) * 256U;
  const std::uint64_t columns =
      (source_space ? 0 : dimension) + shape.nlist + residual_columns;
  if (dimension != 0U &&
      columns > std::numeric_limits<std::uint64_t>::max() / dimension) {
    return Status::ResourceExhausted(
        "canonical model float extent overflows u64");
  }
  return dimension * columns;
}

void AppendMatrix(ColumnMajorMatrixF32View matrix,
                  std::vector<std::uint8_t>& output) {
  const std::size_t count =
      static_cast<std::size_t>(matrix.rows) * matrix.columns;
  const std::size_t begin = output.size();
  output.resize(begin + count * sizeof(float));
  for (std::size_t index = 0U; index < count; ++index) {
    StoreLittleEndianF32(matrix.values[index],
                         output.data() + begin + index * sizeof(float));
  }
}

Status ValidateRuntimeBook(ColumnMajorMatrixF32View book,
                           std::uint32_t dimension,
                           std::uint32_t columns) {
  if (book.values == nullptr || book.rows != dimension ||
      book.columns != columns) {
    return Status::InvalidArgument("runtime model codebook shape mismatch");
  }
  const std::size_t count = static_cast<std::size_t>(dimension) * columns;
  for (std::size_t index = 0U; index < count; ++index) {
    if (!std::isfinite(book.values[index])) {
      return Status::InvalidArgument("runtime model has non-finite codebook");
    }
  }
  return Status::Ok();
}

void WriteHeader(const ModelShape& shape, std::uint32_t version,
                 std::uint64_t float_count, ArtifactHash64 base_identity,
                 std::vector<std::uint8_t>& output) {
  output.assign(CanonicalModelCodec::kHeaderBytes, 0U);
  std::memcpy(output.data(), kMagic.data(), kMagic.size());
  StoreLittleEndianU32(version,
                       output.data() + 8U);
  StoreLittleEndianU32(CanonicalModelCodec::kHeaderBytes,
                       output.data() + 12U);
  StoreLittleEndianU32(shape.dimension, output.data() + 16U);
  StoreLittleEndianU32(shape.m, output.data() + 20U);
  StoreLittleEndianU32(shape.nlist, output.data() + 24U);
  StoreLittleEndianU32(shape.residual_codebook_size, output.data() + 28U);
  StoreLittleEndianU64(float_count, output.data() + 32U);
  StoreLittleEndianU64(base_identity.value(), output.data() + 40U);
}

void WritePayload(const ModelShape& shape,
                  ColumnMajorMatrixF32View rotation,
                  ColumnMajorMatrixF32View root0,
                  ArrayView<const ColumnMajorMatrixF32View> root_small,
                  ArrayView<const ColumnMajorMatrixF32View> edge,
                  std::uint64_t float_count,
                  std::vector<std::uint8_t>& output) {
  WriteHeader(shape, CanonicalModelCodec::kEncodingVersion, float_count, {}, output);
  AppendMatrix(rotation, output);
  AppendMatrix(root0, output);
  for (const auto& book : root_small) AppendMatrix(book, output);
  for (const auto& book : edge) AppendMatrix(book, output);
}

StatusOr<RuntimeCanonicalModel> DecodePayload(
    const std::uint8_t* bytes, std::size_t byte_count,
    std::uint32_t expected_m, bool validate_values) {
  if (bytes == nullptr || byte_count < CanonicalModelCodec::kHeaderBytes ||
      std::memcmp(bytes, kMagic.data(), kMagic.size()) != 0 ||
      LoadLittleEndianU32(bytes + 12U) != CanonicalModelCodec::kHeaderBytes) {
    return Status::DataLoss(
        "canonical model header is truncated or unsupported");
  }
  const auto version = LoadLittleEndianU32(bytes + 8U);
  const bool source_space = version == CanonicalModelCodec::kSourceSpaceVersion;
  const auto identity = ArtifactHash64::FromValue(LoadLittleEndianU64(bytes + 40U));
  if (!source_space && (version != CanonicalModelCodec::kEncodingVersion || !identity.empty()))
    return Status::DataLoss("unsupported canonical model version");
  ModelShape shape = {
      .dimension = LoadLittleEndianU32(bytes + 16U),
      .m = LoadLittleEndianU32(bytes + 20U),
      .nlist = LoadLittleEndianU32(bytes + 24U),
      .residual_codebook_size = LoadLittleEndianU32(bytes + 28U),
  };
  if (const Status status = shape.Validate(); !status.ok()) {
    return status;
  }
  if (expected_m != 0U && shape.m != expected_m) {
    return Status::DataLoss(
        "canonical model M does not match the selected fixed-M decoder");
  }
  auto expected_count = ExpectedFloatCount(shape, source_space);
  if (!expected_count.ok()) {
    return expected_count.status();
  }
  const std::uint64_t stored_count = LoadLittleEndianU64(bytes + 32U);
  if (stored_count != expected_count.value() ||
      stored_count >
          (std::numeric_limits<std::size_t>::max() -
           CanonicalModelCodec::kHeaderBytes) /
              sizeof(float) ||
      byte_count != CanonicalModelCodec::kHeaderBytes +
                        static_cast<std::size_t>(stored_count) *
                            sizeof(float)) {
    return Status::DataLoss(
        "canonical model payload does not match its registered shape");
  }
  std::vector<float> values(static_cast<std::size_t>(stored_count));
  for (std::size_t index = 0U; index < values.size(); ++index) {
    values[index] = LoadLittleEndianF32(
        bytes + CanonicalModelCodec::kHeaderBytes + index * sizeof(float));
    if (validate_values && !std::isfinite(values[index])) {
      return Status::DataLoss("canonical model contains a non-finite value");
    }
  }
  if (source_space) return RuntimeCanonicalModel::SourceSpace(shape, std::move(values), identity);
  return RuntimeCanonicalModel(shape, std::move(values));
}

}  // namespace

Status RuntimeCanonicalModel::Validate() const {
  if (const Status status = shape_.Validate(); !status.ok()) return status;
  auto expected = ExpectedFloatCount(shape_, source_space_);
  if (!expected.ok()) return expected.status();
  if (values_.size() != expected.value()) {
    return Status::InvalidArgument("runtime model payload does not match shape");
  }
  for (const float value : values_) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument("runtime model contains non-finite values");
    }
  }
  return Status::Ok();
}

ColumnMajorMatrixF32View RuntimeCanonicalModel::rotation() const noexcept {
  if (source_space_) return {};
  return {values_.data(), shape_.dimension, shape_.dimension};
}

ColumnMajorMatrixF32View RuntimeCanonicalModel::root0() const noexcept {
  const std::size_t offset =
      static_cast<std::size_t>(rotation().columns) * shape_.dimension;
  return {values_.data() + offset, shape_.dimension, shape_.nlist};
}

ColumnMajorMatrixF32View RuntimeCanonicalModel::root_small(
    std::uint32_t layer) const noexcept {
  assert(layer > 0U && layer < shape_.m);
  const std::size_t offset = static_cast<std::size_t>(shape_.dimension) *
      (rotation().columns + shape_.nlist +
       static_cast<std::size_t>(layer - 1U) * 256U);
  return {values_.data() + offset, shape_.dimension, 256U};
}

ColumnMajorMatrixF32View RuntimeCanonicalModel::edge(
    std::uint32_t layer) const noexcept {
  assert(layer < shape_.m);
  const std::size_t offset = static_cast<std::size_t>(shape_.dimension) *
      (rotation().columns + shape_.nlist +
       static_cast<std::size_t>(shape_.m - 1U + layer) * 256U);
  return {values_.data() + offset, shape_.dimension, 256U};
}

StatusOr<RuntimeCanonicalModel> CanonicalModelCodec::DecodeRuntime(
    const std::uint8_t* bytes, std::size_t byte_count) {
  return DecodePayload(bytes, byte_count, 0U, true);
}

static Status ValidateSource(const RuntimeCanonicalModelSource& source, bool folded = false) {
  if (const Status status = source.shape.Validate(); !status.ok()) {
    return status;
  }
  if (!folded) {
    if (auto s = QueryTransform::ValidateView(source.rotation); !s.ok()) return s;
    if (source.rotation.rows != source.shape.dimension)
      return Status::InvalidArgument("runtime model rotation dimension mismatch");
  }
  if (source.root_small.size() != source.shape.m - 1U ||
      source.edge.size() != source.shape.m ||
      source.root_small.data() == nullptr || source.edge.data() == nullptr) {
    return Status::InvalidArgument("runtime model layer layout mismatch");
  }
  if (const Status status = ValidateRuntimeBook(
          source.root0, source.shape.dimension, source.shape.nlist);
      !status.ok()) {
    return status;
  }
  for (const auto& book : source.root_small) {
    if (const Status status = ValidateRuntimeBook(
            book, source.shape.dimension, 256U); !status.ok()) {
      return status;
    }
  }
  for (const auto& book : source.edge) {
    if (const Status status = ValidateRuntimeBook(
            book, source.shape.dimension, 256U); !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status CanonicalModelCodec::EncodeSearchModel(
    const RuntimeCanonicalModelSource& source, std::vector<std::uint8_t>& output) {
  const bool folded = !source.source_model_identity.empty() && !source.rotation.values;
  if (auto s = ValidateSource(source, folded); !s.ok()) return s;
  const auto d = source.shape.dimension;
  auto identity = source.source_model_identity;
  if (!folded) {
    TrainedModelHashView original;
    original.rotation = source.rotation;
    original.root_codebooks.push_back(source.root0);
    for (const auto& book : source.root_small) original.root_codebooks.push_back(book);
    identity = ArtifactHashFactory::BaseModel(original);
  }
  auto count = ExpectedFloatCount(source.shape, true);
  if (!count.ok()) return count.status();
  if (count.value() > (std::numeric_limits<std::size_t>::max() - kHeaderBytes) / sizeof(float))
    return Status::ResourceExhausted("published model exceeds size_t");
  WriteHeader(source.shape, kSourceSpaceVersion, count.value(),
              identity, output);
  output.reserve(kHeaderBytes + count.value() * sizeof(float));
  std::vector<float> scratch;
  const auto append = [&](ColumnMajorMatrixF32View book) -> Status {
    if (folded) {
      AppendMatrix(book, output);
      return Status::Ok();
    }
    scratch.resize(static_cast<std::size_t>(d) * book.columns);
#ifdef STLQPP_MODEL_CBLAS
    const auto status = CblasF32Gemm().MultiplyTransposedLeft({
        .left=source.rotation, .right=book.values, .query_count=book.columns,
        .right_leading_dimension=d, .output=scratch.data(), .output_leading_dimension=d});
    if (!status.ok()) return status;
#else
    for (std::uint32_t j=0; j<book.columns; ++j)
      for (std::uint32_t i=0; i<d; ++i) {
        float v=0;
        for (std::uint32_t k=0; k<d; ++k) v+=source.rotation.column(i)[k]*book.column(j)[k];
        scratch[static_cast<std::size_t>(j)*d+i]=v;
      }
#endif
    AppendMatrix({scratch.data(), d, book.columns}, output);
    return Status::Ok();
  };
  if (auto s = append(source.root0); !s.ok()) return s;
  for (auto book : source.root_small) if (auto s = append(book); !s.ok()) return s;
  for (auto book : source.edge) if (auto s = append(book); !s.ok()) return s;
  return Status::Ok();
}

Status CanonicalModelCodec::EncodeRuntime(
    const RuntimeCanonicalModelSource& source, std::vector<std::uint8_t>& output) {
  if (auto s = ValidateSource(source); !s.ok()) return s;
  auto count = ExpectedFloatCount(source.shape);
  if (!count.ok()) return count.status();
  if (count.value() >
      (std::numeric_limits<std::size_t>::max() - kHeaderBytes) /
          sizeof(float)) {
    return Status::ResourceExhausted(
        "canonical model byte extent exceeds size_t");
  }
  WritePayload(source.shape, source.rotation, source.root0,
               source.root_small, source.edge, count.value(), output);
  return Status::Ok();
}

template <int M>
Status CanonicalModelCodec::Encode(
    const CanonicalModelSource<M>& source,
    std::vector<std::uint8_t>& output) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  if (const Status status = QueryTransform::ValidateView(source.rotation);
      !status.ok()) {
    return status;
  }
  if (const Status status =
          ProductCodebooks<M>::ValidateView(source.codebooks);
      !status.ok()) {
    return status;
  }
  if (source.rotation.rows != source.codebooks.shape.dimension) {
    return Status::InvalidArgument(
        "canonical rotation and codebooks have different dimensions");
  }
  auto float_count = ExpectedFloatCount(source.codebooks.shape);
  if (!float_count.ok()) {
    return float_count.status();
  }
  if (float_count.value() >
      (std::numeric_limits<std::size_t>::max() - kHeaderBytes) /
          sizeof(float)) {
    return Status::ResourceExhausted(
        "canonical model byte extent exceeds size_t");
  }

  WritePayload(source.codebooks.shape, source.rotation,
               source.codebooks.root0, source.codebooks.root_small,
               source.codebooks.edge, float_count.value(), output);
  return Status::Ok();
}

template <int M>
StatusOr<CanonicalModel<M>> CanonicalModelCodec::Decode(
    const std::uint8_t* bytes,
    std::size_t byte_count) {
  static_assert(M == 0 || (M >= 2 && M <= 20));
  auto payload = DecodePayload(bytes, byte_count, M, false);
  if (!payload.ok()) {
    return payload.status();
  }
  const RuntimeCanonicalModel& runtime = payload.value();
  const ModelShape& shape = runtime.shape();
  const ColumnMajorMatrixF32View rotation = runtime.rotation();
  ProductCodebookView<M> codebook_view;
  codebook_view.shape = shape;
  if constexpr (M == 0) {
    codebook_view.root_small.resize(shape.m - 1U);
    codebook_view.edge.resize(shape.m);
  }
  codebook_view.root0 = runtime.root0();
  for (std::uint32_t layer = 1U; layer < shape.m; ++layer) {
    codebook_view.root_small[layer - 1U] = runtime.root_small(layer);
  }
  for (std::uint32_t layer = 0U; layer < shape.m; ++layer) {
    codebook_view.edge[layer] = runtime.edge(layer);
  }

  StatusOr<QueryTransform> query_transform = runtime.source_space()
      ? StatusOr<QueryTransform>(QueryTransform::SourceSpace(shape.dimension))
      : QueryTransform::Create(rotation);
  if (!query_transform.ok()) {
    return query_transform.status();
  }
  auto codebooks = ProductCodebooks<M>::Create(codebook_view);
  if (!codebooks.ok()) {
    return codebooks.status();
  }
  std::vector<float> canonical_rotation;
  if (!runtime.source_space())
    canonical_rotation.assign(rotation.values, rotation.values +
        static_cast<std::size_t>(shape.dimension) * shape.dimension);
  return CanonicalModel<M>(std::move(canonical_rotation),
                           std::move(query_transform).value(),
                           std::move(codebooks).value());
}

template Status CanonicalModelCodec::Encode<0>(
    const CanonicalModelSource<0>&, std::vector<std::uint8_t>&);
template Status CanonicalModelCodec::Encode<5>(
    const CanonicalModelSource<5>&, std::vector<std::uint8_t>&);
template Status CanonicalModelCodec::Encode<12>(
    const CanonicalModelSource<12>&, std::vector<std::uint8_t>&);
template Status CanonicalModelCodec::Encode<6>(
    const CanonicalModelSource<6>&, std::vector<std::uint8_t>&);
template Status CanonicalModelCodec::Encode<10>(
    const CanonicalModelSource<10>&, std::vector<std::uint8_t>&);
template StatusOr<CanonicalModel<0>> CanonicalModelCodec::Decode<0>(
    const std::uint8_t*, std::size_t);
template StatusOr<CanonicalModel<5>> CanonicalModelCodec::Decode<5>(
    const std::uint8_t*, std::size_t);
template StatusOr<CanonicalModel<12>> CanonicalModelCodec::Decode<12>(
    const std::uint8_t*, std::size_t);
template StatusOr<CanonicalModel<6>> CanonicalModelCodec::Decode<6>(
    const std::uint8_t*, std::size_t);
template StatusOr<CanonicalModel<10>> CanonicalModelCodec::Decode<10>(
    const std::uint8_t*, std::size_t);

}  // namespace stlqpp
