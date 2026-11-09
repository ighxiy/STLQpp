#include "stlqpp/training/io/fvecs_f32_source.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/npy_array.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

namespace stlqpp {
namespace {

struct MappedFvecsSlice final {
  ReadOnlyMappedFile mapped;
  FvecsF32SourceSpec spec;
  std::uint64_t record_bytes = 0U;
  std::uint64_t data_offset_bytes = 0U;
};

StatusOr<MappedFvecsSlice> OpenMappedSlice(
    const FvecsF32SourceSpec& spec) {
  if (spec.path.empty() || spec.dimension == 0U ||
      spec.vector_count == 0U || spec.block_vectors == 0U) {
    return Status::InvalidArgument(
        "fvecs source description is incomplete");
  }
  auto mapped = ReadOnlyMappedFile::Open(spec.path);
  if (!mapped.ok()) {
    return mapped.status();
  }
  const bool npy = spec.layout == F32SourceLayout::kNpyFloat16;
  const bool flat_binary = spec.layout == F32SourceLayout::kFbin;
  std::uint64_t data_offset_bytes = flat_binary ? 8U : 0U;
  std::uint64_t record_bytes =
      static_cast<std::uint64_t>(spec.dimension) * sizeof(float) +
      (flat_binary ? 0U : sizeof(std::uint32_t));
  if (npy) {
    auto layout = InspectNpyArray(mapped.value());
    if (!layout.ok()) return layout.status();
    if (layout.value().scalar != NpyScalar::kFloat16 ||
        layout.value().columns != spec.dimension) {
      return Status::DataLoss("npy vector dtype or dimension disagrees");
    }
    data_offset_bytes = layout.value().data_offset;
    record_bytes = static_cast<std::uint64_t>(spec.dimension) * 2U;
  }
  if (mapped.value().size() < data_offset_bytes ||
      (mapped.value().size() - data_offset_bytes) % record_bytes != 0U) {
    return Status::DataLoss(
        "f32 vector file extent is not an integral record sequence");
  }
  const std::uint64_t file_vectors =
      (mapped.value().size() - data_offset_bytes) / record_bytes;
  if (flat_binary &&
      (LoadLittleEndianU32(mapped.value().data()) != file_vectors ||
       LoadLittleEndianU32(mapped.value().data() + 4U) != spec.dimension)) {
    return Status::DataLoss("fbin header disagrees with the source shape");
  }
  if (spec.first_vector > file_vectors ||
      spec.vector_count > file_vectors - spec.first_vector) {
    return Status::DataLoss(
        "fvecs source slice exceeds the physical record sequence");
  }
  const std::uint8_t* first =
      mapped.value().data() + data_offset_bytes;
  const std::uint8_t* last = first +
      static_cast<std::size_t>(file_vectors - 1U) * record_bytes;
  const std::uint8_t* slice_first = first +
      static_cast<std::size_t>(spec.first_vector) * record_bytes;
  const std::uint8_t* slice_last = slice_first +
      static_cast<std::size_t>(spec.vector_count - 1U) * record_bytes;
  if (!flat_binary && !npy &&
      (LoadLittleEndianU32(first) != spec.dimension ||
       LoadLittleEndianU32(last) != spec.dimension ||
       LoadLittleEndianU32(slice_first) != spec.dimension ||
       LoadLittleEndianU32(slice_last) != spec.dimension)) {
    return Status::DataLoss(
        "fvecs boundary record dimension disagrees with the source shape");
  }
  return MappedFvecsSlice{
      .mapped = std::move(mapped).value(),
      .spec = spec,
      .record_bytes = record_bytes,
      .data_offset_bytes = data_offset_bytes,
  };
}

class MappedFvecsF32BlockSource final : public F32BasicBlockSource {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<F32BasicBlockSource>> Open(
      const FvecsF32SourceSpec& spec,
      std::uint64_t stable_first) {
    auto slice = OpenMappedSlice(spec);
    if (!slice.ok()) {
      return slice.status();
    }
    std::unique_ptr<F32BasicBlockSource> source(
        new MappedFvecsF32BlockSource(
            std::move(slice).value(), stable_first));
    return source;
  }

  [[nodiscard]] StatusOr<std::optional<F32BasicInputBlock>> Next() override {
    if (next_row_ == spec_.vector_count) {
      return std::optional<F32BasicInputBlock>{};
    }
    const std::uint64_t remaining = spec_.vector_count - next_row_;
    const std::uint32_t count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(remaining, spec_.block_vectors));
    F32BasicInputBlock block;
    block.first_id = stable_first_ + next_row_;
    block.dimension = spec_.dimension;
    block.count = count;
    block.values.resize(static_cast<std::size_t>(count) * spec_.dimension);
    for (std::uint32_t local = 0U; local < count; ++local) {
      const std::uint64_t row = spec_.first_vector + next_row_ + local;
      const std::uint8_t* record = mapped_.data() +
          static_cast<std::size_t>(data_offset_bytes_ + row * record_bytes_);
      if (spec_.layout == F32SourceLayout::kFvecs &&
          LoadLittleEndianU32(record) != spec_.dimension) {
        return Status::DataLoss(
            "fvecs record dimension changed inside the registered source");
      }
      const std::uint8_t* payload = record +
          (spec_.layout == F32SourceLayout::kFvecs
               ? sizeof(std::uint32_t) : 0U);
      float* destination = block.values.data() +
          static_cast<std::size_t>(local) * spec_.dimension;
      if (spec_.layout == F32SourceLayout::kNpyFloat16) {
        for (std::uint32_t axis = 0U; axis < spec_.dimension; ++axis) {
          destination[axis] = LoadNpyFloat16(payload + axis * 2U);
        }
      } else {
        std::memcpy(destination, payload,
                    static_cast<std::size_t>(spec_.dimension) * sizeof(float));
      }
    }
    next_row_ += count;
    return std::optional<F32BasicInputBlock>(std::move(block));
  }

 private:
  explicit MappedFvecsF32BlockSource(
      MappedFvecsSlice slice, std::uint64_t stable_first) noexcept
      : mapped_(std::move(slice.mapped)),
        spec_(std::move(slice.spec)),
        record_bytes_(slice.record_bytes),
        data_offset_bytes_(slice.data_offset_bytes),
        stable_first_(stable_first) {}

  ReadOnlyMappedFile mapped_;
  FvecsF32SourceSpec spec_;
  std::uint64_t record_bytes_ = 0U;
  std::uint64_t data_offset_bytes_ = 0U;
  std::uint64_t stable_first_ = 0U;
  std::uint64_t next_row_ = 0U;
};

class MappedFvecsExactRowSource final : public TrainingExactRowSource {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<TrainingExactRowSource>> Open(
      const FvecsF32SourceSpec& spec) {
    auto slice = OpenMappedSlice(spec);
    if (!slice.ok()) {
      return slice.status();
    }
    std::unique_ptr<TrainingExactRowSource> source(
        new MappedFvecsExactRowSource(std::move(slice).value()));
    return source;
  }

  [[nodiscard]] std::uint32_t dimension() const noexcept override {
    return spec_.dimension;
  }

  [[nodiscard]] std::uint64_t row_count() const noexcept override {
    return spec_.vector_count;
  }

  [[nodiscard]] Status Gather(
      ArrayView<const std::uint64_t> stable_ids,
      std::vector<float>* exact_columns) const override {
    if (exact_columns == nullptr) {
      return Status::InvalidArgument("fvecs gather output is null");
    }
    for (const std::uint64_t id : stable_ids) {
      if (id >= spec_.vector_count) {
        return Status::InvalidArgument(
            "fvecs gather stable ID exceeds the registered slice");
      }
    }
    exact_columns->resize(
        static_cast<std::size_t>(spec_.dimension) * stable_ids.size());
    for (std::size_t column = 0U; column < stable_ids.size(); ++column) {
      const std::uint64_t physical_row =
          spec_.first_vector + stable_ids[column];
      const std::uint8_t* record = mapped_.data() +
          static_cast<std::size_t>(data_offset_bytes_ +
                                   physical_row * record_bytes_);
      if (spec_.layout == F32SourceLayout::kFvecs &&
          LoadLittleEndianU32(record) != spec_.dimension) {
        return Status::DataLoss(
            "fvecs gathered record dimension changed");
      }
      const std::uint8_t* payload = record +
          (spec_.layout == F32SourceLayout::kFvecs
               ? sizeof(std::uint32_t) : 0U);
      float* destination = exact_columns->data() +
          column * spec_.dimension;
      if (spec_.layout == F32SourceLayout::kNpyFloat16) {
        for (std::uint32_t axis = 0U; axis < spec_.dimension; ++axis) {
          destination[axis] = LoadNpyFloat16(payload + axis * 2U);
        }
      } else {
        std::memcpy(destination, payload,
                    static_cast<std::size_t>(spec_.dimension) * sizeof(float));
      }
    }
    return Status::Ok();
  }

 private:
  explicit MappedFvecsExactRowSource(MappedFvecsSlice slice) noexcept
      : mapped_(std::move(slice.mapped)),
        spec_(std::move(slice.spec)),
        record_bytes_(slice.record_bytes),
        data_offset_bytes_(slice.data_offset_bytes) {}

  ReadOnlyMappedFile mapped_;
  FvecsF32SourceSpec spec_;
  std::uint64_t record_bytes_ = 0U;
  std::uint64_t data_offset_bytes_ = 0U;
};

StatusOr<std::unique_ptr<F32BasicBlockSource>> OpenSource(
    const FvecsF32SourceSpec& spec) {
  return MappedFvecsF32BlockSource::Open(spec, 0U);
}

}  // namespace

InitializationFvecsF32SourceFactory::InitializationFvecsF32SourceFactory(
    FvecsF32SourceSpec spec,
    const ParallelForExecutor&) noexcept
    : spec_(std::move(spec)),
      cache_(spec_.vector_count, spec_.dimension, spec_.block_vectors) {}

StatusOr<std::unique_ptr<F32BasicBlockSource>>
InitializationFvecsF32SourceFactory::Open() {
  return cache_.Open(0U, spec_.vector_count,
                     [this]() { return OpenSource(spec_); });
}

StatusOr<std::unique_ptr<F32BasicBlockSource>>
InitializationFvecsF32SourceFactory::OpenRange(
    std::uint64_t first_row, std::uint64_t row_count) {
  if (row_count == 0U || first_row > spec_.vector_count ||
      row_count > spec_.vector_count - first_row) {
    return Status::InvalidArgument(
        "fvecs initialization range exceeds the registered source");
  }
  return cache_.Open(first_row, row_count, [this, first_row, row_count]() {
    FvecsF32SourceSpec range = spec_;
    range.first_vector += first_row;
    range.vector_count = row_count;
    return MappedFvecsF32BlockSource::Open(range, first_row);
  });
}

Status InitializationFvecsF32SourceFactory::ReleaseRepeatedScanCache() {
  return cache_.Release();
}

TrainingFvecsF32SourceFactory::TrainingFvecsF32SourceFactory(
    FvecsF32SourceSpec spec,
    const ParallelForExecutor&) noexcept : spec_(std::move(spec)) {}

StatusOr<std::unique_ptr<F32BasicBlockSource>>
TrainingFvecsF32SourceFactory::Open(std::uint32_t) {
  return OpenSource(spec_);
}

StatusOr<std::unique_ptr<TrainingExactRowSource>>
TrainingFvecsF32SourceFactory::OpenExactRows() {
  return MappedFvecsExactRowSource::Open(spec_);
}

BaseFvecsF32SourceFactory::BaseFvecsF32SourceFactory(
    FvecsF32SourceSpec spec,
    const ParallelForExecutor&) noexcept : spec_(std::move(spec)) {}

StatusOr<std::unique_ptr<F32BasicBlockSource>>
BaseFvecsF32SourceFactory::Open() {
  return OpenSource(spec_);
}

StatusOr<std::unique_ptr<F32BasicBlockSource>>
BaseFvecsF32SourceFactory::OpenRange(
    std::uint64_t first_row, std::uint64_t row_count) {
  if (row_count == 0U || first_row > spec_.vector_count ||
      row_count > spec_.vector_count - first_row) {
    return Status::InvalidArgument(
        "f32 base range exceeds the registered source");
  }
  FvecsF32SourceSpec range = spec_;
  range.first_vector += first_row;
  range.vector_count = row_count;
  return MappedFvecsF32BlockSource::Open(range, first_row);
}

StatusOr<std::unique_ptr<TrainingExactRowSource>>
BaseFvecsF32SourceFactory::OpenExactRows() {
  return MappedFvecsExactRowSource::Open(spec_);
}

}  // namespace stlqpp
