#include "stlqpp/storage/trained_model_file.h"

#include "stlqpp/core/numeric/little_endian.h"
#include "stlqpp/platform/sequential_file_writer.h"
#include "stlqpp/platform/staging_directory.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace stlqpp {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'S', 'T', 'L', 'Q', 'M', 'D', 'L', '1'};
constexpr std::uint32_t kVersion = 1U;
constexpr std::size_t kHeaderBytes = 112U;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t HashBytes(std::uint64_t hash,
                        ArrayView<const std::uint8_t> bytes) {
  for (const std::uint8_t byte : bytes) {
    hash = (hash ^ byte) * kFnvPrime;
  }
  return hash;
}

std::filesystem::path NumberedPath(const std::filesystem::path& base,
                                   std::uint32_t sequence) {
  if (sequence == 0U) return base;
  std::ostringstream name;
  name << base.stem().string() << '_' << std::setfill('0')
       << std::setw(3) << sequence << base.extension().string();
  return base.parent_path() / name.str();
}

bool ModelSequence(const std::filesystem::path& base,
                   const std::filesystem::path& candidate,
                   std::uint32_t* sequence) {
  if (candidate.extension() != base.extension()) return false;
  const std::string stem = candidate.stem().string();
  const std::string expected = base.stem().string();
  if (stem == expected) {
    *sequence = 0U;
    return true;
  }
  if (stem.size() < expected.size() + 4U ||
      stem.compare(0U, expected.size(), expected) != 0 ||
      stem[expected.size()] != '_') {
    return false;
  }
  const auto suffix = std::string_view(stem).substr(expected.size() + 1U);
  std::uint32_t parsed = 0U;
  const auto result = std::from_chars(
      suffix.data(), suffix.data() + suffix.size(), parsed);
  if (result.ec != std::errc{} ||
      result.ptr != suffix.data() + suffix.size() || parsed == 0U) {
    return false;
  }
  *sequence = parsed;
  return true;
}

bool IsLegacyDatedModel(const std::filesystem::path& base,
                        const std::filesystem::path& candidate) {
  if (candidate.extension() != base.extension()) return false;
  const std::string base_stem = base.stem().string();
  const std::size_t marker = base_stem.rfind("_train_h");
  if (marker == std::string::npos) return false;
  const std::string prefix = base_stem.substr(0U, marker) + "_train_";
  const std::string stem = candidate.stem().string();
  if (!stem.starts_with(prefix)) return false;
  const std::string_view suffix(stem.data() + prefix.size(),
                                stem.size() - prefix.size());
  if (suffix.size() < 8U ||
      !std::all_of(suffix.begin(), suffix.begin() + 8U,
                   [](char value) { return value >= '0' && value <= '9'; })) {
    return false;
  }
  if (suffix.size() == 8U) return true;
  if (suffix.size() < 10U || suffix[8U] != '_') return false;
  return std::all_of(suffix.begin() + 9U, suffix.end(),
                     [](char value) { return value >= '0' && value <= '9'; });
}

Status SyncParent(const std::filesystem::path& path) {
#if defined(_WIN32)
  (void)path;
  return Status::Ok();
#else
  const auto parent = path.parent_path().empty()
      ? std::filesystem::path(".") : path.parent_path();
  const int descriptor = open(parent.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0) {
    return Status::Internal("cannot open trained-model parent directory");
  }
  const int result = fsync(descriptor);
  const int closed = close(descriptor);
  if (result != 0 || closed != 0) {
    return Status::Internal("cannot sync trained-model parent directory");
  }
  return Status::Ok();
#endif
}

}  // namespace

Status TrainedModelFile::Publish(
    const std::filesystem::path& destination,
    const TrainedModelFileHeader& header,
    ArrayView<const std::uint8_t> model,
    ArrayView<const std::uint8_t> bad_cluster_mask) {
  if (destination.empty() ||
      destination.extension() != ".stlqmodel" ||
      (header.kind != TrainedModelKind::kInitialized &&
       header.kind != TrainedModelKind::kTrained) ||
      header.training_source == 0U || header.training_recipe == 0U ||
      bad_cluster_mask.size() != header.shape.nlist || model.empty()) {
    return Status::InvalidArgument("trained-model publication is incomplete");
  }
  if (const Status status = header.shape.Validate(); !status.ok()) {
    return status;
  }
  for (const auto value : bad_cluster_mask) {
    if (value > 1U) {
      return Status::InvalidArgument("bad-cluster mask is not binary");
    }
  }
  if (model.size() > std::numeric_limits<std::uint64_t>::max() -
                         kHeaderBytes - bad_cluster_mask.size()) {
    return Status::ResourceExhausted("trained-model size overflows u64");
  }
  std::vector<std::uint8_t> bytes(kHeaderBytes, 0U);
  std::memcpy(bytes.data(), kMagic.data(), kMagic.size());
  StoreLittleEndianU32(kVersion, bytes.data() + 8U);
  StoreLittleEndianU32(kHeaderBytes, bytes.data() + 12U);
  StoreLittleEndianU32(header.completed_round, bytes.data() + 16U);
  StoreLittleEndianU32(static_cast<std::uint32_t>(header.kind),
                       bytes.data() + 20U);
  StoreLittleEndianU32(header.shape.dimension, bytes.data() + 24U);
  StoreLittleEndianU32(header.shape.m, bytes.data() + 28U);
  StoreLittleEndianU32(header.shape.nlist, bytes.data() + 32U);
  StoreLittleEndianU32(header.shape.residual_codebook_size,
                       bytes.data() + 36U);
  StoreLittleEndianU64(header.training_source, bytes.data() + 56U);
  StoreLittleEndianU64(header.training_recipe, bytes.data() + 64U);
  StoreLittleEndianU64(model.size(), bytes.data() + 72U);
  StoreLittleEndianU64(bad_cluster_mask.size(), bytes.data() + 96U);
  std::uint64_t hash = HashBytes(kFnvOffset, {bytes.data(), 104U});
  hash = HashBytes(hash, model);
  hash = HashBytes(hash, bad_cluster_mask);
  StoreLittleEndianU64(hash, bytes.data() + 104U);

  auto staging = StagingDirectory::Create(destination);
  if (!staging.ok()) return staging.status();
  const auto pending = staging.value().path() / destination.filename();
  auto writer = SequentialFileWriter::Create(pending);
  if (!writer.ok()) return writer.status();
  for (const ArrayView<const std::uint8_t> chunk : {
           ArrayView<const std::uint8_t>(bytes), model,
           bad_cluster_mask}) {
    if (const Status status = writer.value().Append(chunk); !status.ok()) {
      return status;
    }
  }
  if (const Status status = writer.value().Flush(); !status.ok()) return status;
  if (const Status status = writer.value().Close(); !status.ok()) return status;
  std::error_code error;
  std::filesystem::create_hard_link(pending, destination, error);
  if (error) {
    return Status::FailedPrecondition(
        "cannot publish trained model without replacing an existing file: " +
        destination.string() + ": " + error.message());
  }
  return SyncParent(destination);
}

StatusOr<TrainedModelFile> TrainedModelFile::Open(
    const std::filesystem::path& path) {
  auto mapped = ReadOnlyMappedFile::Open(path);
  if (!mapped.ok()) return mapped.status();
  const auto* data = mapped.value().data();
  const auto size = mapped.value().size();
  if (size < kHeaderBytes ||
      std::memcmp(data, kMagic.data(), kMagic.size()) != 0 ||
      LoadLittleEndianU32(data + 8U) != kVersion ||
      LoadLittleEndianU32(data + 12U) != kHeaderBytes) {
    return Status::DataLoss("trained-model file header is invalid");
  }
  TrainedModelFile result;
  result.header_.completed_round = LoadLittleEndianU32(data + 16U);
  result.header_.kind = static_cast<TrainedModelKind>(
      LoadLittleEndianU32(data + 20U));
  result.header_.shape = {
      .dimension = LoadLittleEndianU32(data + 24U),
      .m = LoadLittleEndianU32(data + 28U),
      .nlist = LoadLittleEndianU32(data + 32U),
      .residual_codebook_size = LoadLittleEndianU32(data + 36U),
  };
  result.header_.training_source = LoadLittleEndianU64(data + 56U);
  result.header_.training_recipe = LoadLittleEndianU64(data + 64U);
  const std::uint64_t model_size = LoadLittleEndianU64(data + 72U);
  const std::uint64_t mask_size = LoadLittleEndianU64(data + 96U);
  if ((result.header_.kind != TrainedModelKind::kInitialized &&
       result.header_.kind != TrainedModelKind::kTrained) ||
      !result.header_.training_source || !result.header_.training_recipe ||
      !result.header_.shape.Validate().ok() ||
      mask_size != result.header_.shape.nlist ||
      model_size > size - kHeaderBytes ||
      size - kHeaderBytes - model_size != mask_size) {
    return Status::DataLoss("trained-model file extents are invalid");
  }
  std::uint64_t hash = HashBytes(kFnvOffset, {data, 104U});
  hash = HashBytes(hash, {data + kHeaderBytes,
                          static_cast<std::size_t>(model_size)});
  hash = HashBytes(hash, {data + kHeaderBytes + model_size,
                          static_cast<std::size_t>(mask_size)});
  if (hash != LoadLittleEndianU64(data + 104U)) {
    return Status::DataLoss("trained-model file checksum disagrees");
  }
  for (std::size_t i = 0; i < mask_size; ++i) {
    if (data[kHeaderBytes + model_size + i] > 1U) {
      return Status::DataLoss("trained-model bad-cluster mask is not binary");
    }
  }
  result.model_size_ = static_cast<std::size_t>(model_size);
  result.file_ = std::move(mapped).value();
  return result;
}

ArrayView<const std::uint8_t> TrainedModelFile::model_bytes() const noexcept {
  return {file_.data() + kHeaderBytes, model_size_};
}

ArrayView<const std::uint8_t> TrainedModelFile::bad_cluster_mask()
    const noexcept {
  return {file_.data() + kHeaderBytes + model_size_, header_.shape.nlist};
}

std::string ShortTrainedModelNamespace(
    std::uint64_t training_source, std::uint64_t training_recipe) {
  constexpr std::string_view domain = "stlqpp/trained-model-namespace/v1";
  std::uint64_t hash = HashBytes(kFnvOffset, {
      reinterpret_cast<const std::uint8_t*>(domain.data()), domain.size()});
  std::array<std::uint8_t, 16U> identity{};
  StoreLittleEndianU64(training_source, identity.data());
  StoreLittleEndianU64(training_recipe, identity.data() + 8U);
  hash = HashBytes(hash, identity);
  std::ostringstream result;
  result << std::hex << std::setfill('0') << std::setw(12)
         << (hash & 0x0000ffffffffffffULL);
  return result.str();
}

StatusOr<std::filesystem::path> NextNumberedTrainedModelPath(
    const std::filesystem::path& base_path) {
  if (base_path.empty() || base_path.extension() != ".stlqmodel") {
    return Status::InvalidArgument("trained-model base path is invalid");
  }
  const auto parent = base_path.parent_path();
  std::error_code error;
  if (!std::filesystem::is_directory(parent, error)) {
    if (error == std::errc::no_such_file_or_directory) return base_path;
    if (error) return Status::Internal(
        "cannot inspect trained-model directory: " + error.message());
    return Status::FailedPrecondition(
        "trained-model parent is not a directory: " + parent.string());
  }
  bool found = false;
  std::uint32_t maximum = 0U;
  for (std::filesystem::directory_iterator it(parent, error), end;
       !error && it != end; it.increment(error)) {
    std::uint32_t sequence = 0U;
    if (ModelSequence(base_path, it->path(), &sequence)) {
      found = true;
      maximum = std::max(maximum, sequence);
    }
  }
  if (error) return Status::Internal(
      "cannot enumerate trained-model directory: " + error.message());
  if (!found) return base_path;
  if (maximum == std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("trained-model sequence is exhausted");
  }
  return NumberedPath(base_path, maximum + 1U);
}

StatusOr<std::filesystem::path> FindFirstTrainedModelPath(
    const std::filesystem::path& base_path,
    std::uint64_t training_source, std::uint64_t training_recipe) {
  if (base_path.empty() || base_path.extension() != ".stlqmodel" ||
      training_source == 0U || training_recipe == 0U) {
    return Status::InvalidArgument("trained-model lookup is incomplete");
  }
  std::error_code error;
  const auto parent = base_path.parent_path();
  if (!std::filesystem::is_directory(parent, error)) {
    if (error == std::errc::no_such_file_or_directory) {
      return Status::FailedPrecondition(
          "trained-model directory does not exist: " + parent.string() +
          " (expected " + base_path.filename().string() +
          "; set io.train_file to use another model)");
    }
    if (error) return Status::Internal(
        "cannot inspect trained-model directory: " + error.message());
    return Status::FailedPrecondition(
        "trained-model directory does not exist: " + parent.string());
  }
  std::vector<std::pair<std::uint32_t, std::filesystem::path>> candidates;
  std::vector<std::filesystem::path> dated_candidates;
  for (std::filesystem::directory_iterator it(parent, error), end;
       !error && it != end; it.increment(error)) {
    std::uint32_t sequence = 0U;
    if (ModelSequence(base_path, it->path(), &sequence)) {
      candidates.emplace_back(sequence, it->path());
    } else if (IsLegacyDatedModel(base_path, it->path())) {
      dated_candidates.push_back(it->path());
    }
  }
  if (error) return Status::Internal(
      "cannot enumerate trained-model directory: " + error.message());
  std::sort(candidates.begin(), candidates.end());
  for (const auto& [sequence, path] : candidates) {
    (void)sequence;
    auto file = TrainedModelFile::Open(path);
    if (!file.ok()) return file.status();
    if (file.value().header().training_source == training_source &&
        file.value().header().training_recipe == training_recipe) {
      return path;
    }
  }
  std::sort(dated_candidates.begin(), dated_candidates.end());
  for (const auto& path : dated_candidates) {
    auto file = TrainedModelFile::Open(path);
    if (!file.ok()) return file.status();
    if (file.value().header().training_source == training_source &&
        file.value().header().training_recipe == training_recipe) {
      return path;
    }
  }
  return Status::FailedPrecondition(
      "no model matches this training identity under " + parent.string() +
      " (stem " + base_path.filename().string() +
      "; set io.train_file to use another model)");
}

}  // namespace stlqpp
