#include "stlqpp/training/io/bvecs_u8_source.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/mapped_file.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

namespace stlqpp {
namespace {

struct MappedBvecsSlice final {
  ReadOnlyMappedFile mapped;
  BvecsU8SourceSpec spec;
  std::uint64_t record_bytes = 0U;
};

StatusOr<MappedBvecsSlice> OpenMappedSlice(
    const BvecsU8SourceSpec& spec) {
  if (spec.path.empty() || spec.dimension == 0U ||
      spec.vector_count == 0U || spec.block_vectors == 0U) {
    return Status::InvalidArgument(
        "bvecs source description is incomplete");
  }
  const std::uint64_t record_bytes =
      sizeof(std::uint32_t) + static_cast<std::uint64_t>(spec.dimension);
  auto mapped = ReadOnlyMappedFile::Open(spec.path);
  if (!mapped.ok()) {
    return mapped.status();
  }
  if (mapped.value().size() % record_bytes != 0U) {
    return Status::DataLoss(
        "bvecs file extent is not an integral record sequence");
  }
  const std::uint64_t file_vectors = mapped.value().size() / record_bytes;
  if (spec.first_vector > file_vectors ||
      spec.vector_count > file_vectors - spec.first_vector) {
    return Status::DataLoss(
        "bvecs source slice exceeds the physical record sequence");
  }
  const std::uint8_t* first = mapped.value().data();
  const std::uint8_t* last = first +
      static_cast<std::size_t>(file_vectors - 1U) * record_bytes;
  const std::uint8_t* slice_first = first +
      static_cast<std::size_t>(spec.first_vector) * record_bytes;
  const std::uint8_t* slice_last = slice_first +
      static_cast<std::size_t>(spec.vector_count - 1U) * record_bytes;
  if (LoadLittleEndianU32(first) != spec.dimension ||
      LoadLittleEndianU32(last) != spec.dimension ||
      LoadLittleEndianU32(slice_first) != spec.dimension ||
      LoadLittleEndianU32(slice_last) != spec.dimension) {
    return Status::DataLoss(
        "bvecs boundary record dimension disagrees with the source shape");
  }
  return MappedBvecsSlice{
      .mapped = std::move(mapped).value(),
      .spec = spec,
      .record_bytes = record_bytes,
  };
}

class MappedBvecsU8BlockSource final : public U8BasicBlockSource {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<U8BasicBlockSource>> Open(
      const BvecsU8SourceSpec& spec,
      std::uint64_t stable_first,
      const ParallelForExecutor& parallel) {
    auto slice = OpenMappedSlice(spec);
    if (!slice.ok()) {
      return slice.status();
    }
    std::unique_ptr<U8BasicBlockSource> source(
        new MappedBvecsU8BlockSource(
            std::move(slice).value(), stable_first, parallel));
    return source;
  }

  [[nodiscard]] StatusOr<std::optional<U8BasicInputBlock>> Next() override {
    if (next_row_ == spec_.vector_count) {
      return std::optional<U8BasicInputBlock>{};
    }
    const std::uint64_t remaining = spec_.vector_count - next_row_;
    const std::uint32_t count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(remaining, spec_.block_vectors));
    U8BasicInputBlock block;
    block.first_id = stable_first_ + next_row_;
    block.dimension = spec_.dimension;
    block.count = count;
    block.values.resize(static_cast<std::size_t>(count) * spec_.dimension);
#ifndef NDEBUG
    const std::uint64_t first_physical = spec_.first_vector + next_row_;
    const std::uint8_t* first_record = mapped_.data() +
        static_cast<std::size_t>(first_physical * record_bytes_);
    if (LoadLittleEndianU32(first_record) != spec_.dimension) {
      return Status::DataLoss(
          "bvecs block dimension changed inside the registered source");
    }
#endif
    CopyContext context{
        .mapped = mapped_.data(),
        .first_physical = spec_.first_vector + next_row_,
        .record_bytes = record_bytes_,
        .dimension = spec_.dimension,
        .destination = block.values.data(),
    };
    if (count >= 4'096U) {
      parallel_->Run(count, &context, CopyPayload);
    } else {
      for (std::uint32_t local = 0U; local < count; ++local) {
        CopyPayload(&context, 0U, local);
      }
    }
    next_row_ += count;
    return std::optional<U8BasicInputBlock>(std::move(block));
  }

 private:
  explicit MappedBvecsU8BlockSource(
      MappedBvecsSlice slice,
      std::uint64_t stable_first,
      const ParallelForExecutor& parallel) noexcept
      : mapped_(std::move(slice.mapped)),
        spec_(std::move(slice.spec)),
        record_bytes_(slice.record_bytes),
        stable_first_(stable_first),
        parallel_(&parallel) {}

  struct CopyContext final {
    const std::uint8_t* mapped = nullptr;
    std::uint64_t first_physical = 0U;
    std::uint64_t record_bytes = 0U;
    std::uint32_t dimension = 0U;
    std::uint8_t* destination = nullptr;
  };

  static void CopyPayload(void* opaque,
                          std::uint32_t,
                          std::uint32_t local) noexcept {
    auto& context = *static_cast<CopyContext*>(opaque);
    const std::uint64_t row = context.first_physical + local;
    const std::uint8_t* record = context.mapped +
        static_cast<std::size_t>(row * context.record_bytes);
    std::memcpy(context.destination +
                    static_cast<std::size_t>(local) * context.dimension,
                record + sizeof(std::uint32_t), context.dimension);
  }

  ReadOnlyMappedFile mapped_;
  BvecsU8SourceSpec spec_;
  std::uint64_t record_bytes_ = 0U;
  std::uint64_t stable_first_ = 0U;
  std::uint64_t next_row_ = 0U;
  const ParallelForExecutor* parallel_ = nullptr;
};

class MappedBvecsExactRowSource final : public TrainingExactRowSource {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<TrainingExactRowSource>> Open(
      const BvecsU8SourceSpec& spec) {
    auto slice = OpenMappedSlice(spec);
    if (!slice.ok()) {
      return slice.status();
    }
    std::unique_ptr<TrainingExactRowSource> source(
        new MappedBvecsExactRowSource(std::move(slice).value()));
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
      return Status::InvalidArgument("bvecs gather output is null");
    }
    for (const std::uint64_t id : stable_ids) {
      if (id >= spec_.vector_count) {
        return Status::InvalidArgument(
            "bvecs gather stable ID exceeds the registered slice");
      }
    }
    exact_columns->resize(
        static_cast<std::size_t>(spec_.dimension) * stable_ids.size());
    for (std::size_t column = 0U; column < stable_ids.size(); ++column) {
      const std::uint64_t physical_row =
          spec_.first_vector + stable_ids[column];
      const std::uint8_t* record = mapped_.data() +
          static_cast<std::size_t>(physical_row * record_bytes_);
      if (LoadLittleEndianU32(record) != spec_.dimension) {
        return Status::DataLoss(
            "bvecs gathered record dimension changed");
      }
      float* destination = exact_columns->data() +
          column * spec_.dimension;
      const std::uint8_t* payload = record + sizeof(std::uint32_t);
      for (std::uint32_t axis = 0U; axis < spec_.dimension; ++axis) {
        destination[axis] = static_cast<float>(payload[axis]);
      }
    }
    return Status::Ok();
  }

 private:
  explicit MappedBvecsExactRowSource(MappedBvecsSlice slice) noexcept
      : mapped_(std::move(slice.mapped)),
        spec_(std::move(slice.spec)),
        record_bytes_(slice.record_bytes) {}

  ReadOnlyMappedFile mapped_;
  BvecsU8SourceSpec spec_;
  std::uint64_t record_bytes_ = 0U;
};

StatusOr<std::unique_ptr<U8BasicBlockSource>> OpenSource(
    const BvecsU8SourceSpec& spec,
    const ParallelForExecutor& parallel) {
  return MappedBvecsU8BlockSource::Open(spec, 0U, parallel);
}

}  // namespace

InitializationBvecsU8SourceFactory::InitializationBvecsU8SourceFactory(
    BvecsU8SourceSpec spec,
    const ParallelForExecutor& parallel) noexcept
    : spec_(std::move(spec)),
      parallel_(&parallel),
      cache_(spec_.vector_count, spec_.dimension, spec_.block_vectors) {}

StatusOr<std::unique_ptr<U8BasicBlockSource>>
InitializationBvecsU8SourceFactory::Open() {
  return cache_.Open(0U, spec_.vector_count,
                     [this]() { return OpenSource(spec_, *parallel_); });
}

StatusOr<std::unique_ptr<U8BasicBlockSource>>
InitializationBvecsU8SourceFactory::OpenRange(
    std::uint64_t first_row, std::uint64_t row_count) {
  if (row_count == 0U || first_row > spec_.vector_count ||
      row_count > spec_.vector_count - first_row) {
    return Status::InvalidArgument(
        "bvecs initialization range exceeds the registered source");
  }
  return cache_.Open(first_row, row_count, [this, first_row, row_count]() {
    BvecsU8SourceSpec range = spec_;
    range.first_vector += first_row;
    range.vector_count = row_count;
    return MappedBvecsU8BlockSource::Open(
        range, first_row, *parallel_);
  });
}

Status InitializationBvecsU8SourceFactory::ReleaseRepeatedScanCache() {
  return cache_.Release();
}

TrainingBvecsU8SourceFactory::TrainingBvecsU8SourceFactory(
    BvecsU8SourceSpec spec,
    const ParallelForExecutor& parallel) noexcept
    : spec_(std::move(spec)), parallel_(&parallel) {}

StatusOr<std::unique_ptr<U8BasicBlockSource>>
TrainingBvecsU8SourceFactory::Open(std::uint32_t) {
  return OpenSource(spec_, *parallel_);
}

StatusOr<std::unique_ptr<TrainingExactRowSource>>
TrainingBvecsU8SourceFactory::OpenExactRows() {
  return MappedBvecsExactRowSource::Open(spec_);
}

BaseBvecsU8SourceFactory::BaseBvecsU8SourceFactory(
    BvecsU8SourceSpec spec,
    const ParallelForExecutor& parallel) noexcept
    : spec_(std::move(spec)), parallel_(&parallel) {}

StatusOr<std::unique_ptr<U8BasicBlockSource>>
BaseBvecsU8SourceFactory::Open() {
  return OpenSource(spec_, *parallel_);
}

StatusOr<std::unique_ptr<U8BasicBlockSource>>
BaseBvecsU8SourceFactory::OpenRange(
    std::uint64_t first_row, std::uint64_t row_count) {
  if (row_count == 0U || first_row > spec_.vector_count ||
      row_count > spec_.vector_count - first_row) {
    return Status::InvalidArgument(
        "bvecs base range exceeds the registered source");
  }
  BvecsU8SourceSpec range = spec_;
  range.first_vector += first_row;
  range.vector_count = row_count;
  return MappedBvecsU8BlockSource::Open(
      range, first_row, *parallel_);
}

StatusOr<std::unique_ptr<TrainingExactRowSource>>
BaseBvecsU8SourceFactory::OpenExactRows() {
  return MappedBvecsExactRowSource::Open(spec_);
}

}  // namespace stlqpp
