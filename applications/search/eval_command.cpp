#include "stlqpp/model/model_shape.h"
#include "eval_command.h"
#include "stlqpp/application/compute_backend.h"

#include "stlqpp/model/codec/coefficient_bit_width.h"

#include "stlqpp/application/config_schema.h"
#include "stlqpp/application/config_text.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/runtime/telemetry/logger.h"
#include "stlqpp/storage/index_manifest_codec.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace stlqpp::app {
namespace {

using Values = std::unordered_map<std::string, std::string>;

std::filesystem::path NormalizeDirectoryPath(
    const std::filesystem::path& input) {
  std::filesystem::path normalized = input.lexically_normal();
  while (!normalized.empty() && normalized != normalized.root_path() &&
         normalized.filename().empty()) {
    normalized = normalized.parent_path();
  }
  return normalized;
}

std::string Lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

std::filesystem::path FirstDatasetFile(
    const std::filesystem::path& root,
    std::initializer_list<std::filesystem::path> names) {
  for (const auto& name : names) {
    const auto candidate = root / name;
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error)) return candidate;
  }
  return root / *names.begin();
}

StatusOr<std::uint64_t> IndexRealRowsIfPresent(
    const std::filesystem::path& index) {
  if (index.empty()) return 0U;
  const auto manifest_path = index / "manifest.stlqpp";
  std::error_code error;
  if (!std::filesystem::is_regular_file(manifest_path, error)) {
    if (error == std::errc::no_such_file_or_directory) return 0U;
    if (error) return Status::Internal(
        "cannot inspect eval index manifest: " + error.message());
    return 0U;
  }
  auto bytes = ReadOnlyMappedFile::Open(manifest_path);
  if (!bytes.ok()) return bytes.status();
  IndexManifest manifest;
  if (const Status status = IndexManifestCodec::Decode(
          bytes.value().data(), bytes.value().size(), manifest);
      !status.ok()) return status;
  return manifest.counts.real;
}

Status InferEvalDatasetPaths(const std::string& name,
                             const std::filesystem::path& root,
                             std::uint64_t base_rows,
                             std::uint64_t first_row,
                             std::filesystem::path* query,
                             std::filesystem::path* groundtruth) {
  if (!query->empty() && !groundtruth->empty()) return Status::Ok();
  std::filesystem::path inferred_query;
  std::filesystem::path inferred_groundtruth;
  const std::string dataset = Lowercase(name);
  if (groundtruth->empty() && first_row != 0U)
    return Status::InvalidArgument(
        "automatic groundtruth does not cover nonzero dataset.base_first_row; "
        "set dataset.groundtruth_path for the selected slice");
  if (groundtruth->empty() && base_rows != 0U &&
      ((dataset == "sift1m" && base_rows != 1'000'000U) ||
       (dataset == "gist1m" && base_rows != 1'000'000U) ||
       (dataset == "siftsmall" && base_rows != 10'000U)))
    return Status::InvalidArgument(
        "automatic groundtruth requires the complete dataset; "
        "set dataset.groundtruth_path for the selected subset");
  if (dataset == "gist1m") {
    inferred_query = FirstDatasetFile(root,
        {"gist_query.fvecs", "gist/gist_query.fvecs"});
    inferred_groundtruth = FirstDatasetFile(root,
        {"gist_groundtruth.ivecs", "gist/gist_groundtruth.ivecs"});
  } else if (dataset == "sift1m") {
    inferred_query = FirstDatasetFile(root,
        {"sift_query.fvecs", "sift/sift_query.fvecs"});
    inferred_groundtruth = FirstDatasetFile(root,
        {"sift_groundtruth.ivecs", "sift/sift_groundtruth.ivecs"});
  } else if (dataset == "siftsmall") {
    inferred_query = FirstDatasetFile(root,
        {"siftsmall_query.fvecs", "siftsmall/siftsmall_query.fvecs"});
    inferred_groundtruth = FirstDatasetFile(root,
        {"siftsmall_groundtruth.ivecs",
         "siftsmall/siftsmall_groundtruth.ivecs"});
  } else if (dataset == "msong") {
    // The no-self split GT is reindexed after removing the training rows.
    // Never infer the original-base GT for one of these shortened bases.
    if (base_rows == 892'565U || base_rows == 692'565U) {
      const bool train_100k = base_rows == 892'565U;
      const auto split = train_100k ? "train_100k" : "train_300k";
      const auto gt = train_100k
          ? "msong_groundtruth_noself_after_train_100k.ivecs"
          : "msong_groundtruth_noself_after_train_300k.ivecs";
      inferred_query = FirstDatasetFile(root,
          {std::filesystem::path("noself/splits") / split /
               "msong_query.fvecs",
           std::filesystem::path("msong/noself/splits") / split /
               "msong_query.fvecs",
           "msong_query.fvecs"});
      inferred_groundtruth = FirstDatasetFile(root,
          {std::filesystem::path("noself/splits") / split / gt,
           std::filesystem::path("msong/noself/splits") / split / gt,
           gt});
    } else if (base_rows == 0U || base_rows == 994'185U) {
      inferred_query = FirstDatasetFile(root,
          {"msong_query.fvecs", "msong/msong_query.fvecs"});
      inferred_groundtruth = FirstDatasetFile(root,
          {"msong_groundtruth.ivecs",
           "msong/msong_groundtruth.ivecs"});
    } else {
      inferred_query = FirstDatasetFile(root,
          {"msong_query.fvecs", "msong/msong_query.fvecs"});
      if (groundtruth->empty()) {
        return Status::InvalidArgument(
            "MSONG has no built-in groundtruth for dataset.nbase=" +
            std::to_string(base_rows) +
            "; set dataset.groundtruth_path explicitly");
      }
    }
  } else if (dataset == "sift1b") {
    inferred_query = FirstDatasetFile(root,
        {"queries.bvecs", "bigann/queries.bvecs",
         "bigann_query.bvecs", "bigann/bigann_query.bvecs"});
    constexpr std::uint64_t subsets[] = {
        1U, 2U, 5U, 10U, 20U, 50U, 100U, 200U, 500U, 1000U};
    std::uint64_t selected = base_rows == 0U ? 1000U : 0U;
    for (const auto subset : subsets) {
      if (base_rows >= subset * 1'000'000U) selected = subset;
    }
    if (selected == 0U && groundtruth->empty()) {
      return Status::InvalidArgument(
          "SIFT1B automatic groundtruth requires dataset.nbase >= 1M; "
          "set dataset.groundtruth_path explicitly");
    }
    if (base_rows != 0U && base_rows != selected * 1'000'000U &&
        groundtruth->empty()) {
      return Status::InvalidArgument(
          "SIFT1B has no exact groundtruth for dataset.nbase=" +
          std::to_string(base_rows) +
          "; set dataset.groundtruth_path explicitly");
    }
    const auto file = "idx_" + std::to_string(selected) + "M.ivecs";
    inferred_groundtruth = FirstDatasetFile(root,
        {std::filesystem::path("bigann_gnd") / file,
         std::filesystem::path("bigann") / "bigann_gnd" / file,
         std::filesystem::path("gnd") / file,
         std::filesystem::path("bigann") / "gnd" / file});
  } else if (dataset == "deep10m") {
    inferred_query = FirstDatasetFile(root,
        {"query.public.10K.fbin", "deep10m/query.public.10K.fbin"});
    if (base_rows == 0U || base_rows == 1'000'000U ||
        base_rows == 10'000'000U) {
      const auto file = base_rows == 1'000'000U
          ? "groundtruth.1M.ibin" : "groundtruth.10M.ibin";
      inferred_groundtruth = FirstDatasetFile(root,
          {file, std::filesystem::path("deep10m") / file});
    } else if (groundtruth->empty()) {
      return Status::InvalidArgument(
          "DEEP10M automatic groundtruth requires dataset.nbase=1M or "
          "10M; set dataset.groundtruth_path explicitly");
    }
  }
  if (query->empty()) *query = inferred_query;
  if (groundtruth->empty()) *groundtruth = inferred_groundtruth;
  return Status::Ok();
}

bool HasSuffix(const std::filesystem::path& path, std::string_view suffix) {
  const std::string value = Lowercase(path.generic_string());
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

StatusOr<std::string> ResolveQueryFormat(
    std::string requested, const std::filesystem::path& path) {
  requested = Lowercase(std::move(requested));
  if (requested.empty()) requested = "auto";
  if (requested == "siftbin" || requested == "u8") requested = "bvecs";
  if (requested == "float" || requested == "f32") requested = "fvecs";
  if (requested != "auto") {
    if (requested == "bvecs" || requested == "fvecs" ||
        requested == "fbin" || requested == "npy") {
      return requested;
    }
    return Status::InvalidArgument("unsupported query vector format");
  }
  if (HasSuffix(path, ".bvecs") || HasSuffix(path, ".siftbin")) {
    return std::string("bvecs");
  }
  if (HasSuffix(path, ".fbin")) return std::string("fbin");
  if (HasSuffix(path, ".fvecs")) return std::string("fvecs");
  if (HasSuffix(path, ".npy")) return std::string("npy");
  return Status::InvalidArgument(
      "cannot infer large.query_format from dataset.query_path=" +
      path.string() + "; set large.query_format explicitly");
}

StatusOr<std::string> ResolveGroundtruthFormat(
    std::string requested, const std::filesystem::path& path) {
  requested = Lowercase(std::move(requested));
  if (requested.empty()) requested = "auto";
  if (requested != "auto") {
    if (requested == "ivecs" || requested == "ibin" ||
        requested == "npy") return requested;
    return Status::InvalidArgument("unsupported groundtruth format");
  }
  if (HasSuffix(path, ".ibin")) return std::string("ibin");
  if (HasSuffix(path, ".ivecs")) return std::string("ivecs");
  if (HasSuffix(path, ".npy")) return std::string("npy");
  return Status::InvalidArgument(
      "cannot infer large.gt_format from dataset.groundtruth_path=" +
      path.string() + "; set large.gt_format explicitly");
}

std::string Trim(std::string_view value) {
  return TrimConfigText(value);
}

StatusOr<std::string> CanonicalKey(std::string key) {
  if (key.rfind("--", 0U) == 0U) key.erase(0U, 2U);
  // Read old frozen cfgs without modifying their output directories. The
  // removed override switch no longer controls execution or new snapshots.
  if (key == "runtime.omp_affinity_override_env") return std::string{};
  auto canonical = ResolveProductConfigKey(ProductConfigScope::kEval, key);
  if (canonical.ok()) return canonical;
  // A complete DEV cfg also carries producer fields. They are consumed by
  // the build parser and are intentionally irrelevant to an eval-only run.
  if (IsProductConfigField(ProductConfigScope::kBuild, key) ||
      IsProductExtensionConfigField(ProductConfigScope::kBuild, key)) {
    return std::string{};
  }
  return Status::InvalidArgument("unsupported eval config key " + key);
}

bool ParseEvalNormMode(std::string_view text,
                       EvalNormMode* output) noexcept {
  if (text == "exact") {
    *output = EvalNormMode::kFloat;
    return true;
  }
  if (text == "lut") {
    *output = EvalNormMode::kLut;
    return true;
  }
  return false;
}


Status Assign(std::string_view text, Values* values) {
  auto stripped = StripConfigComment(text);
  if (!stripped.ok()) return stripped.status();
  const std::string line = std::move(stripped).value();
  if (line.empty()) return Status::Ok();
  const std::size_t equal = line.find('=');
  if (equal == std::string::npos) {
    return Status::InvalidArgument("invalid eval config assignment");
  }
  auto key = CanonicalKey(Trim(std::string_view(line).substr(0U, equal)));
  if (!key.ok()) return key.status();
  auto decoded = DecodeConfigValue(
      std::string_view(line).substr(equal + 1U));
  if (!decoded.ok()) return decoded.status();
  std::string value = std::move(decoded).value();
  if (key.value().empty()) return Status::Ok();
  (*values)[std::move(key).value()] = std::move(value);
  return Status::Ok();
}

Status Load(const std::filesystem::path& path,
            std::string_view bytes,
            Values* values) {
  std::istringstream input{std::string(bytes)};
  std::string line;
  while (std::getline(input, line)) {
    if (const Status status = Assign(line, values); !status.ok()) return status;
  }
  return input.eof() ? Status::Ok()
                     : Status::InvalidArgument(
                           "failed reading eval config " + path.string());
}

StatusOr<FrozenEvalConfigInput> ReadFrozenConfig(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status::InvalidArgument(
        "cannot open eval config " + path.string());
  }
  std::string bytes{
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()};
  if (input.bad()) {
    return Status::InvalidArgument(
        "failed reading eval config " + path.string());
  }
  return FrozenEvalConfigInput{.path = path, .bytes = std::move(bytes)};
}

template <typename Integer>
bool IntegerValue(const Values& values, const char* key, Integer* output) {
  const auto found = values.find(key);
  if (found == values.end()) return false;
  const std::string& value = found->second;
  const auto parsed = std::from_chars(
      value.data(), value.data() + value.size(), *output);
  return parsed.ec == std::errc{} &&
         parsed.ptr == value.data() + value.size();
}

bool BoolValue(const Values& values, const char* key, bool* output) {
  const auto found = values.find(key);
  if (found == values.end()) return false;
  std::string normalized = found->second;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (normalized == "1" || normalized == "true") {
    *output = true; return true;
  }
  if (normalized == "0" || normalized == "false") {
    *output = false; return true;
  }
  return false;
}

bool DoubleValue(const Values& values, const char* key, double* output) {
  const auto found = values.find(key);
  if (found == values.end()) return false;
  std::istringstream input(found->second);
  input >> std::noskipws >> *output;
  return input.eof() && !input.fail() && std::isfinite(*output);
}

bool ListValue(const std::string& raw, std::vector<std::uint32_t>* output) {
  output->clear();
  std::string text = Trim(raw);
  if (text.size() >= 2U && text.front() == '[' && text.back() == ']') {
    text = text.substr(1U, text.size() - 2U);
  }
  std::string_view remaining = text;
  while (!remaining.empty()) {
    const std::size_t comma = remaining.find(',');
    const std::string token = Trim(remaining.substr(0U, comma));
    std::uint32_t value = 0U;
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size() || value == 0U) return false;
    output->push_back(value);
    if (comma == std::string_view::npos) break;
    remaining.remove_prefix(comma + 1U);
  }
  return !output->empty();
}

bool CoefficientBitsValue(const Values& values, const char* key,
                          std::vector<std::uint32_t>* output) {
  const auto found = values.find(key);
  if (found == values.end()) return false;
  output->clear();
  std::string text = Trim(found->second);
  if (text.size() >= 2U && text.front() == '[' && text.back() == ']') {
    text = Trim(std::string_view(text).substr(1U, text.size() - 2U));
  }
  if (text.empty()) return true;
  std::string_view remaining = text;
  while (!remaining.empty()) {
    const std::size_t comma = remaining.find(',');
    const std::string token = Trim(remaining.substr(0U, comma));
    std::uint32_t value = 0U;
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size()) {
      output->clear();
      return false;
    }
    output->push_back(value);
    if (comma == std::string_view::npos) break;
    remaining.remove_prefix(comma + 1U);
  }
  return true;
}

bool PercentCandidateListValue(
    const Values& values, const char* key, std::vector<double>* output) {
  const auto found = values.find(key);
  if (found == values.end()) return false;
  std::string text = Trim(found->second);
  if (text.size() >= 2U && text.front() == '[' && text.back() == ']') {
    text = Trim(std::string_view(text).substr(1U, text.size() - 2U));
  }
  output->clear();
  std::string_view remaining = text;
  while (!remaining.empty()) {
    const std::size_t comma = remaining.find(',');
    const std::string token = Trim(remaining.substr(0U, comma));
    Values scalar{{"candidate", token}};
    double candidate = 0.0;
    if (token.empty() || !DoubleValue(scalar, "candidate", &candidate) ||
        candidate <= 0.0 || candidate > 100.0) {
      output->clear();
      return false;
    }
    output->push_back(candidate);
    if (comma == std::string_view::npos) break;
    remaining.remove_prefix(comma + 1U);
  }
  return !output->empty();
}

}  // namespace

StatusOr<EvalCommand> ParseEvalCommand(
    int argc, char** argv, bool allow_deferred_build_sources) {
  Values values = {{"warmup", "0"}, {"repeat", "1"},
                   {"query-block", "1000"}, {"nprobe", "[8]"},
                   {"ivf-probe-mode", "exact"}, {"ivf-hnsw-m", "32"},
                   {"ivf-hnsw-ef-construction", "200"}, {"ivf-hnsw-ef-search", "32"},
                   {"train-checkpoint-file", ""},
                   {"reused-base-list-directory", ""},
                   {"work", ""}, {"output", ""},
                   {"omp-places", ""},
                   {"base-enabled", "false"},
                   {"linkage-enabled", "true"},
                   {"base-use-ivf", "true"},
                   {"base-nprobe", "8"},
                   {"base-warmup", "0"},
                   {"base-repeat", "1"},
                   {"parent-louds-native-eval", "false"},
                   {"pretransform-query-codebooks", "true"},
                   {"profile-timing", "false"},
                   {"rerank-candidates", "0"}, {"rerank-storage", "memory"},
                   {"omp-proc-bind", ""}, {"quiet", "false"},
                   {"nquery", "0"}, {"k", "100"},
                   {"groundtruth-add1", "false"},
                   {"query-format", "auto"}, {"gt-format", "auto"},
                   {"metric-mode", "0"},
                   {"metric-gt-topks", "[1,10,50,100]"},
                   {"archive", "true"},
                   {"archive-log", "true"},
                   {"eval-modes", "quantized_scaled"},
                   {"score-mode", "l2"},
                   {"dataset", ""},
                   {"data-root", "../data"},
                   {"base", ""}, {"base-format", "auto"},
                   {"base-first-row", "0"}, {"base-rows", "0"}};
  values["use-cuda"] = "true";
  values["gpu-norm-enable"] = "true";
  values["cuda-device"] = "0";
  std::vector<std::filesystem::path> config_files;
  std::vector<FrozenEvalConfigInput> frozen_config_inputs;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      return Status::InvalidArgument("missing eval option value");
    }
    const std::string option = argv[index];
    const std::string value = argv[++index];
    if (option == "--config") {
      auto decoded = DecodeConfigValue(value);
      if (!decoded.ok()) return decoded.status();
      const std::filesystem::path path = std::move(decoded).value();
      auto frozen = ReadFrozenConfig(path);
      if (!frozen.ok()) return frozen.status();
      if (const Status status = Load(path, frozen.value().bytes, &values);
          !status.ok()) return status;
      config_files.emplace_back(path);
      frozen_config_inputs.push_back(std::move(frozen).value());
    } else if (option == "--set") {
      if (const Status status = Assign(value, &values); !status.ok()) return status;
    } else {
      auto key = CanonicalKey(option);
      if (!key.ok()) return key.status();
      if (key.value().empty()) continue;
      values[std::move(key).value()] = value;
    }
  }
  EvalCommand output;
  if (values["score-mode"] == "cosine_recon") {
    output.requested_score_mode = SearchScoreMode::kCosineRecon;
  } else if (values["score-mode"] == "mips_recon") {
    output.requested_score_mode = SearchScoreMode::kMipsRecon;
  } else if (values["score-mode"] != "l2") {
    return Status::InvalidArgument(
        "eval.linkage.score_mode must be l2, cosine_recon, or mips_recon");
  }
  std::uint32_t mode = 0U;
  EvalNormMode requested_norm_mode = EvalNormMode::kLut;
  std::uint32_t expected_norm8_iterations = 0U;
  std::vector<double> expected_coefficient_first_percents;
  std::vector<double> expected_coefficient_rest_percents;
  EvalRepresentationSelection representation_selection;
  std::vector<std::uint32_t> expected_bits;
  std::vector<std::uint32_t> expected_root_bits;
  std::vector<std::uint32_t> expected_linkage_bits;
  bool expected_weighted_quantile = true;
  bool expected_allow_clipping = true;
  bool expected_fit_scale = true;
  std::uint32_t expected_refinement_sweeps = 0U;
  std::uint32_t expected_refinement_max_layer = 0U;
  std::uint32_t expected_refinement_step_limit = 0U;
  const bool have_expected_norm8 =
      values.contains("norm8-lloyd-iterations");
  const bool have_expected_norm8_mode = values.contains("norm8-mode");
  const bool have_expected_coefficient =
      values.contains("coefficient-p-first-candidates") ||
      values.contains("coefficient-p-rest-candidates");
  const bool have_expected_bits = values.contains("coefficient-bits-per-layer");
  const bool have_expected_root_bits =
      values.contains("coefficient-bits-per-layer-root");
  const bool have_expected_linkage_bits =
      values.contains("coefficient-bits-per-layer-linkage");
  const bool have_expected_weighted_quantile =
      values.contains("coefficient-weighted-quantile");
  const bool have_expected_allow_clipping =
      values.contains("coefficient-allow-clipping");
  const bool have_expected_fit_scale =
      values.contains("coefficient-fit-scale");
  const bool have_expected_refinement_sweeps =
      values.contains("coefficient-refinement-sweeps");
  const bool have_expected_refinement_max_layer =
      values.contains("coefficient-refinement-max-layer");
  const bool have_expected_refinement_step_limit =
      values.contains("coefficient-refinement-step-limit");
  if (!IntegerValue(values, "m", &output.m) ||
      !IntegerValue(values, "base-first-row",
                    &output.base_source_first_row) ||
      !IntegerValue(values, "base-rows",
                    &output.base_source_row_count) ||
      !IntegerValue(values, "threads", &output.threads) ||
      !BoolValue(values, "use-cuda", &output.use_cuda) ||
      !BoolValue(values, "gpu-norm-enable", &output.gpu_norm_enable) ||
      !IntegerValue(values, "cuda-device", &output.cuda_device) ||
      output.cuda_device < 0 ||
      !IntegerValue(values, "nquery", &output.nquery) ||
      !IntegerValue(values, "k", &output.top_k) ||
      !IntegerValue(values, "warmup", &output.warmup) ||
      !IntegerValue(values, "repeat", &output.repeat) ||
      !IntegerValue(values, "query-block", &output.query_block) ||
      !IntegerValue(values, "rerank-candidates", &output.rerank_candidates) ||
      !IntegerValue(values, "ivf-hnsw-m", &output.ivf_hnsw_m) ||
      !IntegerValue(values, "ivf-hnsw-ef-construction", &output.ivf_hnsw_ef_construction) ||
      !ListValue(values["ivf-hnsw-ef-search"], &output.ivf_hnsw_ef_search) ||
      output.ivf_hnsw_m < 2U || output.ivf_hnsw_m > 10000U ||
      output.ivf_hnsw_ef_construction == 0U ||
      (values["ivf-probe-mode"] != "exact" && values["ivf-probe-mode"] != "hnsw") ||
      !BoolValue(values, "base-enabled", &output.base_enabled) ||
      !BoolValue(values, "linkage-enabled", &output.linkage_enabled) ||
      !BoolValue(values, "base-use-ivf", &output.base_use_ivf) ||
      !IntegerValue(values, "base-nprobe", &output.base_nprobe) ||
      !IntegerValue(values, "base-warmup", &output.base_warmup) ||
      !IntegerValue(values, "base-repeat", &output.base_repeat) ||
      !BoolValue(values, "parent-louds-native-eval",
                 &output.parent_louds_native_eval) ||
      !BoolValue(values, "pretransform-query-codebooks",
                 &output.pretransform_query_codebooks) ||
      !BoolValue(values, "profile-timing", &output.profile_timing) ||
      !ParseEvalRepresentationSelection(
          values["eval-modes"], &representation_selection) ||
      (have_expected_bits && !CoefficientBitsValue(
          values, "coefficient-bits-per-layer", &expected_bits)) ||
      (have_expected_root_bits && !CoefficientBitsValue(
          values, "coefficient-bits-per-layer-root", &expected_root_bits)) ||
      (have_expected_linkage_bits && !CoefficientBitsValue(
          values, "coefficient-bits-per-layer-linkage",
          &expected_linkage_bits)) ||
      (have_expected_weighted_quantile && !BoolValue(
          values, "coefficient-weighted-quantile",
          &expected_weighted_quantile)) ||
      (have_expected_allow_clipping && !BoolValue(
          values, "coefficient-allow-clipping",
          &expected_allow_clipping)) ||
      (have_expected_fit_scale && !BoolValue(
          values, "coefficient-fit-scale", &expected_fit_scale)) ||
      (have_expected_refinement_sweeps && !IntegerValue(
          values, "coefficient-refinement-sweeps",
          &expected_refinement_sweeps)) ||
      (have_expected_refinement_max_layer && !IntegerValue(
          values, "coefficient-refinement-max-layer",
          &expected_refinement_max_layer)) ||
      (have_expected_refinement_step_limit && !IntegerValue(
          values, "coefficient-refinement-step-limit",
          &expected_refinement_step_limit)) ||
      (have_expected_norm8 &&
       !IntegerValue(values, "norm8-lloyd-iterations",
                     &expected_norm8_iterations)) ||
      (values.contains("disk-norm2-mode") &&
       !ParseEvalNormMode(values["disk-norm2-mode"], &requested_norm_mode)) ||
      (have_expected_norm8_mode && values["norm8-mode"] != "identity" && values["norm8-mode"] != "tail_weighted" && values["norm8-mode"] != "reference") ||
      (have_expected_coefficient &&
       (!PercentCandidateListValue(
            values, "coefficient-p-first-candidates",
            &expected_coefficient_first_percents) ||
        !PercentCandidateListValue(
            values, "coefficient-p-rest-candidates",
            &expected_coefficient_rest_percents))) ||
      !IntegerValue(values, "metric-mode", &mode) ||
      !BoolValue(values, "quiet", &output.quiet) ||
      !BoolValue(values, "archive", &output.archive) ||
      !BoolValue(values, "archive-log", &output.archive_log) ||
      !BoolValue(values, "groundtruth-add1", &output.groundtruth_add1) ||
      !ListValue(values["nprobe"], &output.nprobes) ||
      !ListValue(values["metric-gt-topks"], &output.metric_gt_topks) ||
      (output.m < 2U || output.m > stlqpp::kMaximumModelLayers) ||
      output.repeat == 0U || output.base_repeat == 0U ||
      output.base_nprobe == 0U || output.query_block == 0U ||
      output.top_k == 0U ||
      (have_expected_norm8 && expected_norm8_iterations == 0U) ||
      mode > 2U) {
    return Status::InvalidArgument("invalid or incomplete eval config");
  }
  output.rerank_storage = values["rerank-storage"];
  if (output.rerank_storage != "memory" && output.rerank_storage != "disk")
    return Status::InvalidArgument("eval.linkage.rerank_storage must be memory or disk");
  if (output.rerank_candidates != 0U &&
      (output.rerank_candidates < output.top_k || !output.linkage_enabled ||
       output.requested_score_mode != SearchScoreMode::kClassic || output.profile_timing))
    return Status::InvalidArgument(
        "L2 reranking requires linkage enabled, candidates >= dataset.k, score_mode=l2 and profile_timing=false");
  output.ivf_probe_mode = values["ivf-probe-mode"];
  output.active_ivf_hnsw_ef_search = output.ivf_hnsw_ef_search.front();
  output.index = NormalizeDirectoryPath(values["index"]);
  output.train_checkpoint_file =
      NormalizeDirectoryPath(values["train-checkpoint-file"]);
  output.work_directory = NormalizeDirectoryPath(values["work"]);
  output.requested_base_list_directory =
      NormalizeDirectoryPath(values["reused-base-list-directory"]);
  output.requested_representations_auto = representation_selection.automatic;
  output.requested_representations =
      std::move(representation_selection.explicit_values);
  output.dataset_data_root = values["data-root"];
  output.base_source = values["base"];
  output.base_source_format = values["base-format"];
  output.queries = values["queries"];
  output.groundtruth = values["groundtruth"];
  std::uint64_t eval_base_rows = output.base_source_row_count;
  const std::string eval_dataset = Lowercase(values["dataset"]);
  if (eval_base_rows == 0U && output.groundtruth.empty() &&
      (eval_dataset == "sift1b" || eval_dataset == "deep10m" ||
       eval_dataset == "msong" || eval_dataset == "sift1m" ||
       eval_dataset == "gist1m" || eval_dataset == "siftsmall")) {
    auto indexed_rows = IndexRealRowsIfPresent(output.index);
    if (!indexed_rows.ok()) return indexed_rows.status();
    eval_base_rows = indexed_rows.value();
  }
  if (const Status status = InferEvalDatasetPaths(
          values["dataset"], output.dataset_data_root,
          eval_base_rows, output.base_source_first_row,
          &output.queries, &output.groundtruth);
      !status.ok()) return status;
  const auto resolve_dataset_path = [&](const std::filesystem::path& path) {
    return path.empty() || path.is_absolute()
        ? path : output.dataset_data_root / path;
  };
  output.effective_queries = resolve_dataset_path(output.queries);
  output.effective_groundtruth = resolve_dataset_path(output.groundtruth);
  output.effective_base_source = resolve_dataset_path(output.base_source);
  if (output.effective_queries.empty()) {
    return Status::InvalidArgument(
        "eval requires dataset.query_path (query file path is empty)");
  }
  if (output.effective_groundtruth.empty()) {
    return Status::InvalidArgument(
        "eval requires dataset.groundtruth_path "
        "(groundtruth file path is empty)");
  }
  output.query_format = values["query-format"];
  output.groundtruth_format = values["gt-format"];
  auto effective_query_format =
      ResolveQueryFormat(output.query_format, output.effective_queries);
  if (!effective_query_format.ok()) return effective_query_format.status();
  output.effective_query_format = std::move(effective_query_format).value();
  auto effective_groundtruth_format =
      ResolveGroundtruthFormat(output.groundtruth_format,
                               output.effective_groundtruth);
  if (!effective_groundtruth_format.ok()) {
    return effective_groundtruth_format.status();
  }
  output.effective_groundtruth_format =
      std::move(effective_groundtruth_format).value();
  output.run_root = !output.index.empty()
      ? output.index.parent_path()
      : (!allow_deferred_build_sources &&
         !output.requested_base_list_directory.empty()
             ? output.requested_base_list_directory.parent_path()
             : NormalizeDirectoryPath(values["output"]));
  // Derived product indexes still belong to the original run. Keep logs,
  // eval archives and the basic-list lookup outside the derived cache.
  for (auto path = output.index; !path.empty(); path = path.parent_path()) {
    if (path.filename() == "linkage_list") {
      output.run_root = path.parent_path();
      break;
    }
    if (path == path.root_path()) break;
  }
  if (output.run_root.empty()) {
    output.run_root = output.index.parent_path();
  }
  if (output.run_root.empty()) output.run_root = ".";
  output.base_list_directory =
      !output.requested_base_list_directory.empty()
      ? output.requested_base_list_directory
      : (!output.index.empty()
             ? output.run_root / "base-lists"
             : std::filesystem::path{});
  output.omp_places = values["omp-places"];
  output.omp_proc_bind = values["omp-proc-bind"];
  if (output.omp_places.empty()) output.omp_places = "cores";
  if (output.omp_proc_bind.empty()) output.omp_proc_bind = "spread";
  output.metric_mode = static_cast<EvalMetricMode>(mode);
  if (have_expected_norm8) {
    output.expected_norm8_lloyd_iterations = expected_norm8_iterations;
  }
  if (values.contains("disk-norm2-mode")) output.requested_norm_mode = requested_norm_mode;
  if (have_expected_norm8_mode) {
    output.expected_norm8_mode = values["norm8-mode"] == "reference"
        ? ReconstructionNorm8Mode::kReferenceLloyd : values["norm8-mode"] == "tail_weighted"
        ? ReconstructionNorm8Mode::kTailWeightedLloyd : ReconstructionNorm8Mode::kIdentityLloyd;
  }
  if(output.expected_norm8_mode==ReconstructionNorm8Mode::kTailWeightedLloyd &&
      output.requested_norm_mode.value_or(EvalNormMode::kLut)!=EvalNormMode::kFloat)
    output.requested_norm_mode=EvalNormMode::kLutTailWeighted;
  if (have_expected_coefficient) {
    output.expected_coefficient_first_quantile_percents =
        std::move(expected_coefficient_first_percents);
    output.expected_coefficient_rest_quantile_percents =
        std::move(expected_coefficient_rest_percents);
  }
  const auto normalize_expected = [&](std::vector<std::uint32_t> widths,
                                      const char* field)
      -> StatusOr<std::vector<std::uint32_t>> {
    if (widths.empty()) return widths;
    return NormalizeCoefficientBits(std::move(widths), output.m, field);
  };
  if (have_expected_bits) {
    auto normalized = normalize_expected(
        std::move(expected_bits),
        "large.linkage_coeff_codec.bits_per_layer");
    if (!normalized.ok()) return normalized.status();
    if (!normalized.value().empty()) {
      output.expected_coefficient_bits_per_layer =
          std::move(normalized).value();
    }
  }
  if (have_expected_root_bits) {
    auto normalized = normalize_expected(
        std::move(expected_root_bits),
        "large.linkage_coeff_codec.bits_per_layer_root");
    if (!normalized.ok()) return normalized.status();
    output.expected_coefficient_bits_per_layer_root =
        std::move(normalized).value();
  }
  if (have_expected_linkage_bits) {
    auto normalized = normalize_expected(
        std::move(expected_linkage_bits),
        "large.linkage_coeff_codec.bits_per_layer_linkage");
    if (!normalized.ok()) return normalized.status();
    output.expected_coefficient_bits_per_layer_linkage =
        std::move(normalized).value();
  }
  if (have_expected_weighted_quantile) {
    output.expected_coefficient_weighted_quantile =
        expected_weighted_quantile;
  }
  if (have_expected_allow_clipping) {
    output.expected_coefficient_allow_clipping = expected_allow_clipping;
  }
  if (have_expected_fit_scale) {
    output.expected_coefficient_fit_scale = expected_fit_scale;
  }
  if (have_expected_refinement_sweeps) {
    output.expected_coefficient_refinement_sweeps =
        expected_refinement_sweeps;
  }
  if (have_expected_refinement_max_layer) {
    output.expected_coefficient_refinement_max_layer =
        expected_refinement_max_layer;
  }
  if (have_expected_refinement_step_limit) {
    output.expected_coefficient_refinement_step_limit =
        expected_refinement_step_limit;
  }
  output.config_files = std::move(config_files);
  if (const Status status = ResolveRuntimeCudaExecutionPolicy(
          output.use_cuda, output.gpu_norm_enable); !status.ok()) {
    return status;
  }
  output.frozen_config_inputs = std::move(frozen_config_inputs);
  if (!output.base_enabled && !output.linkage_enabled) {
    return Status::InvalidArgument(
        "eval.base.enabled and eval.linkage.enabled cannot both be false");
  }
  if (output.linkage_enabled && output.index.empty() &&
      output.train_checkpoint_file.empty() &&
      !allow_deferred_build_sources) {
    return Status::InvalidArgument(
        "eval-only linkage requires io.train_file for hash-based path "
        "derivation or explicit io.linkage_file");
  }
  if (output.base_enabled && output.index.empty() &&
      output.train_checkpoint_file.empty() &&
      !allow_deferred_build_sources) {
    return Status::InvalidArgument(
        "basic eval requires io.train_file or io.linkage_file");
  }
  if (output.base_enabled && output.base_list_directory.empty() &&
      output.train_checkpoint_file.empty() &&
      !allow_deferred_build_sources) {
    return Status::InvalidArgument(
        "basic eval requires a resolvable base-list directory");
  }
  if (!std::all_of(output.label.begin(), output.label.end(),
                   [](unsigned char ch) {
                     return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
                   }) ||
      (output.omp_places != "" && output.omp_places != "none" &&
       output.omp_places != "cores" &&
       output.omp_places != "threads" && output.omp_places != "sockets") ||
      (output.omp_proc_bind != "" && output.omp_proc_bind != "none" &&
       output.omp_proc_bind != "close" &&
       output.omp_proc_bind != "spread" &&
       output.omp_proc_bind != "primary")) {
    return Status::InvalidArgument(
        "eval label or OpenMP affinity setting is invalid");
  }
  return output;
}

Status BindEvalCommandToIndex(
    EvalCommand& command, const IndexManifest& manifest) {
  if (manifest.shape.m != command.m) {
    return Status::FailedPrecondition(
        "eval M does not match the loaded index manifest");
  }
  command.classic_score_codec_hash = manifest.hashes.norm8_cache;
  command.scaled_score_codec_hash = manifest.scaled_score8_hash;
  command.float_scaled_score_codec_hash =
      manifest.hashes.linkage_list_directory;
  command.representation_runs.clear();
  if (!command.linkage_enabled) {
    command.producer_recipe_verified = false;
    command.active_norm_mode = command.requested_norm_mode.value_or(
        EvalNormMode::kLut);
    return Status::Ok();
  }
  if (command.requested_representations_auto) {
    if (manifest.has_float_forest()) {
      command.representation_runs.push_back(EvalRepresentation::kFloat);
    }
    if (command.requested_score_mode == SearchScoreMode::kClassic &&
        manifest.has_float_source_norm2()) {
      command.representation_runs.push_back(
          EvalRepresentation::kFloatScaled);
    }
    if (manifest.has_classic_norm8()) {
      command.representation_runs.push_back(EvalRepresentation::kQuantized);
    }
    if (command.requested_score_mode == SearchScoreMode::kClassic &&
        manifest.has_scaled_score8()) {
      command.representation_runs.push_back(
          EvalRepresentation::kQuantizedScaled);
    }
  }
  for (const EvalRepresentation representation :
       command.requested_representations) {
    if (command.requested_score_mode != SearchScoreMode::kClassic &&
        (representation == EvalRepresentation::kFloatScaled ||
         representation == EvalRepresentation::kQuantizedScaled)) {
      return Status::InvalidArgument(
          "angular eval.linkage.score_mode requires "
          "eval.linkage.modes=[float], [quantized], or [auto]; "
          "scaled representations use a different score definition");
    }
    if (representation == EvalRepresentation::kFloat &&
        !manifest.has_float_forest()) {
      return Status::FailedPrecondition(
          "eval.linkage.modes requests float but the index has no native-float forest");
    }
    if (representation == EvalRepresentation::kFloatScaled &&
        !manifest.has_float_forest()) {
      return Status::FailedPrecondition(
          "eval.linkage.modes requests float_scaled but the index has no native-float forest");
    }
    if (representation == EvalRepresentation::kQuantized &&
        !manifest.has_classic_norm8()) {
      return Status::FailedPrecondition(
          "eval.linkage.modes requests quantized but the index has no classic norm8 representation");
    }
    if (representation == EvalRepresentation::kQuantizedScaled &&
        !manifest.has_scaled_score8() && !command.independent_quantized_norms) {
      return Status::FailedPrecondition(
          "eval.linkage.modes requests quantized_scaled but the norm2 scale/bias table is absent; regenerate it from retained source norms or select quantized");
    }
    command.representation_runs.push_back(representation);
  }
  if (command.representation_runs.empty()) {
    return Status::FailedPrecondition(
        "eval.linkage.modes=auto found no searchable representation in the index manifest");
  }
  const bool runs_quantized = std::any_of(
      command.representation_runs.begin(), command.representation_runs.end(),
      [](EvalRepresentation value) {
        return value == EvalRepresentation::kQuantized ||
            value == EvalRepresentation::kQuantizedScaled;
      });
  // Codec width, fitting and norm8 assertions belong only to a quantized
  // leg. A shared build cfg may contain them while auto resolves a
  // float-only product; they must not make that inactive producer mandatory.
  if (!runs_quantized) {
    command.producer_recipe_verified = false;
    return SelectEvalRepresentation(
        command, command.representation_runs.front());
  }
  if (!manifest.producer_recipe.has_value()) {
    if (command.expected_coefficient_bits_per_layer.has_value() ||
        command.expected_coefficient_bits_per_layer_root.has_value() ||
        command.expected_coefficient_bits_per_layer_linkage.has_value() ||
        command.expected_coefficient_weighted_quantile.has_value() ||
        command.expected_coefficient_allow_clipping.has_value() ||
        command.expected_coefficient_fit_scale.has_value() ||
        command.expected_coefficient_refinement_sweeps.has_value() ||
        command.expected_coefficient_refinement_max_layer.has_value() ||
        command.expected_coefficient_refinement_step_limit.has_value()) {
      return Status::FailedPrecondition(
          "eval coefficient-width assertions require producer recipe provenance");
    }
    if (command.archive && runs_quantized) {
      return Status::FailedPrecondition(
          "eval archive requires index producer recipe provenance; "
          "legacy format may be searched only with archive disabled");
    }
    command.producer_recipe_verified = false;
    if (!command.representation_runs.empty()) {
      return SelectEvalRepresentation(
          command, command.representation_runs.front());
    }
    return Status::Ok();
  }
  const IndexProducerRecipe& recipe = *manifest.producer_recipe;
  command.coefficient_bits_per_layer =
      recipe.coefficient_bits_per_layer.empty()
      ? std::vector<std::uint32_t>(command.m, kDefaultCoefficientBits)
      : recipe.coefficient_bits_per_layer;
  command.coefficient_bits_per_layer_root =
      recipe.coefficient_bits_per_layer_root;
  command.coefficient_bits_per_layer_linkage =
      recipe.coefficient_bits_per_layer_linkage;
  const std::vector<std::uint32_t> effective_root_bits =
      command.coefficient_bits_per_layer_root.empty()
      ? command.coefficient_bits_per_layer
      : command.coefficient_bits_per_layer_root;
  const std::vector<std::uint32_t> effective_linkage_bits =
      command.coefficient_bits_per_layer_linkage.empty()
      ? command.coefficient_bits_per_layer
      : command.coefficient_bits_per_layer_linkage;
  const auto& requested_common =
      command.expected_coefficient_bits_per_layer.value_or(
          command.coefficient_bits_per_layer);
  const auto requested_role = [&](const std::vector<std::uint32_t>& stored,
                                  const std::optional<
                                      std::vector<std::uint32_t>>& requested)
      -> const std::vector<std::uint32_t>& {
    const auto& role = requested.has_value() ? *requested : stored;
    return role.empty() ? requested_common : role;
  };
  if (requested_role(command.coefficient_bits_per_layer_root,
                     command.expected_coefficient_bits_per_layer_root) !=
      effective_root_bits) {
    return Status::FailedPrecondition(
        "eval root coefficient-width assertion disagrees with index producer");
  }
  if (requested_role(command.coefficient_bits_per_layer_linkage,
                     command.expected_coefficient_bits_per_layer_linkage) !=
      effective_linkage_bits) {
    return Status::FailedPrecondition(
        "eval linkage coefficient-width assertion disagrees with index producer");
  }
  const auto require_equal = [](const auto& expected, const auto actual,
                                const char* field) -> Status {
    if (expected.has_value() && *expected != actual) {
      return Status::FailedPrecondition(
          std::string("eval ") + field +
          " assertion disagrees with index producer");
    }
    return Status::Ok();
  };
  if (const Status status = require_equal(
          command.expected_coefficient_weighted_quantile,
          recipe.coefficient_weighted_quantile,
          "weighted-quantile"); !status.ok()) return status;
  if (const Status status = require_equal(
          command.expected_coefficient_allow_clipping,
          recipe.coefficient_allow_clipping,
          "allow-clipping"); !status.ok()) return status;
  if (const Status status = require_equal(
          command.expected_coefficient_fit_scale,
          recipe.coefficient_fit_scale,
          "fit-scale"); !status.ok()) return status;
  if (const Status status = require_equal(
          command.expected_coefficient_refinement_sweeps,
          recipe.coefficient_refinement_sweeps,
          "refinement-sweeps"); !status.ok()) return status;
  if (const Status status = require_equal(
          command.expected_coefficient_refinement_max_layer,
          recipe.coefficient_refinement_max_layer,
          "refinement-max-layer"); !status.ok()) return status;
  if (const Status status = require_equal(
          command.expected_coefficient_refinement_step_limit,
          recipe.coefficient_refinement_step_limit,
          "refinement-step-limit"); !status.ok()) return status;
  // DEV exposes percent while the producer stores fractions. Normalize each
  // decimal at twelve places so 0.9993 maps to the registered 99.93 identity.
  const auto as_percents = [](const std::vector<double>& fractions) {
    std::vector<double> output;
    output.reserve(fractions.size());
    for (const double fraction : fractions) {
      output.push_back(
          std::round(fraction * 100.0 * 1.0e12) / 1.0e12);
    }
    return output;
  };
  const std::vector<double> first_percents = as_percents(
      recipe.coefficient_first_magnitude_quantiles);
  const std::vector<double> rest_percents = as_percents(
      recipe.coefficient_rest_magnitude_quantiles);
  const auto same_candidates = [](const std::vector<double>& left,
                                  const std::vector<double>& right) {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin(),
                   [](double a, double b) {
                     return std::abs(a - b) <= 1.0e-12;
                   });
  };
  if (!command.independent_quantized_norms && command.expected_norm8_lloyd_iterations.has_value() &&
      *command.expected_norm8_lloyd_iterations !=
          recipe.norm8_lloyd_iterations) {
    return Status::FailedPrecondition(
        "eval norm8 iteration assertion disagrees with index producer");
  }
  if (!command.independent_quantized_norms && command.expected_norm8_mode.has_value() &&
      *command.expected_norm8_mode != recipe.norm8_mode) {
    return Status::FailedPrecondition(
        "eval norm8 mode assertion disagrees with index producer");
  }
  if ((!command.expected_coefficient_first_quantile_percents.empty() ||
       !command.expected_coefficient_rest_quantile_percents.empty()) &&
      (!same_candidates(
           command.expected_coefficient_first_quantile_percents,
           first_percents) ||
       !same_candidates(
           command.expected_coefficient_rest_quantile_percents,
           rest_percents))) {
    return Status::FailedPrecondition(
        "eval coefficient candidate assertion disagrees with index producer");
  }
  command.norm8_lloyd_iterations = recipe.norm8_lloyd_iterations;
  command.norm8_mode = recipe.norm8_mode;
  command.coefficient_first_quantile_percents = first_percents;
  command.coefficient_rest_quantile_percents = rest_percents;
  command.coefficient_weighted_quantile =
      recipe.coefficient_weighted_quantile;
  command.coefficient_allow_clipping = recipe.coefficient_allow_clipping;
  command.coefficient_fit_scale = recipe.coefficient_fit_scale;
  command.coefficient_refinement_sweeps =
      recipe.coefficient_refinement_sweeps;
  command.coefficient_refinement_max_layer =
      recipe.coefficient_refinement_max_layer;
  command.coefficient_refinement_step_limit =
      recipe.coefficient_refinement_step_limit;
  command.producer_recipe_verified = true;
  return SelectEvalRepresentation(command, command.representation_runs.front());
}

const char* EvalNormModeName(EvalNormMode mode) noexcept {
  switch (mode) {
    case EvalNormMode::kFloat: return "exact";
    case EvalNormMode::kLut: return "lut";
    case EvalNormMode::kLutTailWeighted: return "lut";
    case EvalNormMode::kLutReference: return "lut";
  }
  return "lut";
}

const char* EvalScoreCodecName(EvalScoreCodec codec) noexcept {
  switch (codec) {
    case EvalScoreCodec::kNativeFloat:
      return "native_float";
    case EvalScoreCodec::kFloatScaled:
      return "exact_float_radial_scale";
    case EvalScoreCodec::kReconstructionNorm8:
      return "reconstruction_norm8";
    case EvalScoreCodec::kScaledScore8:
      return "norm2_scale8";
    case EvalScoreCodec::kReconstructionNormF32:
      return "reconstruction_norm_f32";
    case EvalScoreCodec::kRadialScoreF32:
      return "exact_radial_scale_f32";
  }
  return "unknown";
}

Status SelectEvalRepresentation(
    EvalCommand& command, EvalRepresentation representation) {
  command.active_representation = representation;
  const bool is_float = representation == EvalRepresentation::kFloat ||
      representation == EvalRepresentation::kFloatScaled;
  const bool is_scaled =
      representation == EvalRepresentation::kFloatScaled ||
      representation == EvalRepresentation::kQuantizedScaled;
  if (is_scaled &&
      command.requested_score_mode != SearchScoreMode::kClassic) {
    return Status::InvalidArgument(
        "cosine_recon/mips_recon cannot use a scaled representation");
  }
  command.active_coefficient_mode = is_float
      ? SearchCoefficientMode::kFloat : SearchCoefficientMode::kInt8;
  command.active_score_mode = is_scaled
      ? SearchScoreMode::kScaled : command.requested_score_mode;
  command.active_norm_mode = command.requested_norm_mode.value_or(EvalNormMode::kLut);
  if (command.active_norm_mode == EvalNormMode::kLut &&
      command.expected_norm8_mode.value_or(ReconstructionNorm8Mode::kIdentityLloyd) == ReconstructionNorm8Mode::kTailWeightedLloyd)
    command.active_norm_mode = EvalNormMode::kLutTailWeighted;
  if (command.active_norm_mode == EvalNormMode::kLut &&
      command.expected_norm8_mode == ReconstructionNorm8Mode::kReferenceLloyd)
    command.active_norm_mode = EvalNormMode::kLutReference;
  if (command.active_norm_mode == EvalNormMode::kLutReference &&
      command.active_score_mode != SearchScoreMode::kClassic &&
      command.active_score_mode != SearchScoreMode::kScaled)
    return Status::InvalidArgument("reference norm LUT requires L2 scoring");
  const bool float_norm = command.active_norm_mode == EvalNormMode::kFloat;
  command.active_score_codec = float_norm
      ? (is_float ? (is_scaled ? EvalScoreCodec::kFloatScaled : EvalScoreCodec::kNativeFloat)
                  : (is_scaled ? EvalScoreCodec::kRadialScoreF32 : EvalScoreCodec::kReconstructionNormF32))
      : (is_scaled ? EvalScoreCodec::kScaledScore8 : EvalScoreCodec::kReconstructionNorm8);
  command.active_score_codec_hash = is_scaled
      ? (is_float ? command.float_scaled_score_codec_hash : command.scaled_score_codec_hash)
      : (is_float ? ArtifactHash64{} : command.classic_score_codec_hash);
  return Status::Ok();
}

void WriteRoundTripEvalConfig(
    const EvalCommand& command, std::ostream& output,
    bool include_eval_only) {
  if (include_eval_only) output << "advanced.eval_only = true\n";
  auto write_string = [&](const char* key, const auto& value) {
    output << key << " = " << QuoteConfigValue(value) << '\n';
  };
  output << "model.m = " << command.m << '\n';
  write_string(
      "io.train_file", command.train_checkpoint_file.generic_string());
  write_string("io.linkage_file", command.index.generic_string());
  if (!command.requested_base_list_directory.empty()) {
    write_string("io.base_file",
                 command.requested_base_list_directory.generic_string());
  }
  write_string("large.tmp_dir", command.work_directory.generic_string());
  write_string("large.output_dir", command.run_root.generic_string());
  write_string(
      "dataset.data_root", command.dataset_data_root.generic_string());
  write_string("dataset.base_path", command.base_source.generic_string());
  output << "dataset.base_first_row = " << command.base_source_first_row
         << "\ndataset.nbase = " << command.base_source_row_count << '\n';
  write_string("large.base_format", command.base_source_format);
  write_string("dataset.query_path", command.queries.generic_string());
  write_string(
      "dataset.groundtruth_path", command.groundtruth.generic_string());
  output << "dataset.nquery = " << command.nquery
         << "\ndataset.k = " << command.top_k
         << "\ndataset.groundtruth_add1 = "
         << static_cast<int>(command.groundtruth_add1) << '\n';
  write_string("large.query_format", command.query_format);
  write_string("large.gt_format", command.groundtruth_format);
  output << "eval.base.enabled = " << static_cast<int>(command.base_enabled)
         << "\neval.base.use_ivf = "
         << static_cast<int>(command.base_use_ivf)
         << "\neval.base.nprobe = " << command.base_nprobe
         << "\neval.base.warmup = " << command.base_warmup
         << "\neval.base.repeat = " << command.base_repeat
         << "\neval.linkage.enabled = "
         << static_cast<int>(command.linkage_enabled)
         << "\neval.linkage.nprobe = [";
  for (std::size_t index = 0U; index < command.nprobes.size(); ++index) {
    if (index != 0U) output << ',';
    output << command.nprobes[index];
  }
  output << "]\nruntime.omp_threads = " << command.threads
         << "\nruntime.use_cuda = " << static_cast<int>(command.use_cuda)
         << "\neval.linkage.gpu_norm_enable = "
         << static_cast<int>(command.gpu_norm_enable)
         << "\neval.linkage.score_mode = "
         << (command.requested_score_mode == SearchScoreMode::kCosineRecon
                 ? "cosine_recon"
                 : command.requested_score_mode == SearchScoreMode::kMipsRecon
                       ? "mips_recon" : "l2")
         << "\nruntime.cuda_device = " << command.cuda_device
         << "\neval.linkage.warmup = " << command.warmup
         << "\neval.linkage.repeat = " << command.repeat
         << "\neval.linkage.ivf_probe_mode = \"" << command.ivf_probe_mode << "\""
         << "\neval.linkage.ivf_hnsw_M = " << command.ivf_hnsw_m
         << "\neval.linkage.ivf_hnsw_ef_construction = " << command.ivf_hnsw_ef_construction
         << "\neval.linkage.query_block = " << command.query_block
         << "\neval.linkage.rerank_candidates = " << command.rerank_candidates
         << "\neval.linkage.rerank_storage = \"" << command.rerank_storage << "\""
         << "\neval.linkage.parent_louds_native_eval = "
         << static_cast<int>(command.parent_louds_native_eval)
         << "\neval.linkage.pretransform_query_codebooks = "
         << static_cast<int>(command.pretransform_query_codebooks)
         << "\nlarge.profile_timing = "
         << static_cast<int>(command.profile_timing) << '\n';
  output << "eval.linkage.ivf_hnsw_ef_search = [";
  for (std::size_t i = 0; i < command.ivf_hnsw_ef_search.size(); ++i) {
    if (i) output << ',';
    output << command.ivf_hnsw_ef_search[i];
  }
  output << "]\n";
  output << "eval.linkage.modes = ";
  if (command.requested_representations_auto) {
    output << "auto";
  } else {
    output << '[';
    for (std::size_t index = 0U;
         index < command.requested_representations.size(); ++index) {
      if (index != 0U) output << ',';
      output << EvalRepresentationName(
          command.requested_representations[index]);
    }
    output << ']';
  }
  output << '\n';
  const std::optional<std::uint32_t> norm8 = command.producer_recipe_verified
      ? std::optional<std::uint32_t>{command.norm8_lloyd_iterations}
      : command.expected_norm8_lloyd_iterations;
  const std::vector<double>& coefficient_first =
      command.producer_recipe_verified
      ? command.coefficient_first_quantile_percents
      : command.expected_coefficient_first_quantile_percents;
  const std::vector<double>& coefficient_rest =
      command.producer_recipe_verified
      ? command.coefficient_rest_quantile_percents
      : command.expected_coefficient_rest_quantile_percents;
  if (norm8.has_value() && *norm8 != 0U) {
    output << "eval.disk_norm2_lut_kmeans_niter = " << *norm8 << '\n';
  }
  const EvalNormMode norm_mode = command.requested_norm_mode.value_or(EvalNormMode::kLut);
  output << "eval.disk_norm2_mode = "
         << QuoteConfigValue(EvalNormModeName(norm_mode)) << '\n';
  output << "eval.disk_norm2_lut_mode = " << QuoteConfigValue(
      ReconstructionNorm8ModeName(command.expected_norm8_mode.value_or(command.norm8_mode))) << '\n';
  const auto write_candidates = [&](const std::vector<double>& candidates) {
    output << '[';
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
      if (index != 0U) output << ',';
      output << std::setprecision(std::numeric_limits<double>::max_digits10)
             << candidates[index];
    }
    output << ']';
  };
  if (!coefficient_first.empty() && !coefficient_rest.empty()) {
    output << "large.linkage_coeff_codec.p_first_candidates = ";
    write_candidates(coefficient_first);
    output << "\nlarge.linkage_coeff_codec.p_rest_candidates = ";
    write_candidates(coefficient_rest);
    output << '\n';
  }
  const auto write_bits = [&](std::string_view key,
                              const std::vector<std::uint32_t>& bits) {
    output << key << " = [";
    for (std::size_t index = 0U; index < bits.size(); ++index) {
      if (index != 0U) output << ',';
      output << bits[index];
    }
    output << "]\n";
  };
  if (command.producer_recipe_verified) {
    write_bits("large.linkage_coeff_codec.bits_per_layer",
               command.coefficient_bits_per_layer);
    write_bits("large.linkage_coeff_codec.bits_per_layer_root",
               command.coefficient_bits_per_layer_root);
    write_bits("large.linkage_coeff_codec.bits_per_layer_linkage",
               command.coefficient_bits_per_layer_linkage);
  } else {
    if (command.expected_coefficient_bits_per_layer.has_value()) {
      write_bits("large.linkage_coeff_codec.bits_per_layer",
                 *command.expected_coefficient_bits_per_layer);
    }
    if (command.expected_coefficient_bits_per_layer_root.has_value()) {
      write_bits("large.linkage_coeff_codec.bits_per_layer_root",
                 *command.expected_coefficient_bits_per_layer_root);
    }
    if (command.expected_coefficient_bits_per_layer_linkage.has_value()) {
      write_bits("large.linkage_coeff_codec.bits_per_layer_linkage",
                 *command.expected_coefficient_bits_per_layer_linkage);
    }
  }
  const auto write_codec_scalar = [&](std::string_view key,
                                      const auto& actual,
                                      const auto& expected) {
    if (command.producer_recipe_verified) {
      output << key << " = " << actual << '\n';
    } else if (expected.has_value()) {
      output << key << " = " << *expected << '\n';
    }
  };
  write_codec_scalar(
      "large.linkage_coeff_codec.use_weighted_quantile",
      static_cast<int>(command.coefficient_weighted_quantile),
      command.expected_coefficient_weighted_quantile);
  write_codec_scalar(
      "large.linkage_coeff_codec.allow_clip",
      static_cast<int>(command.coefficient_allow_clipping),
      command.expected_coefficient_allow_clipping);
  write_codec_scalar(
      "large.linkage_coeff_codec.fit_scale",
      static_cast<int>(command.coefficient_fit_scale),
      command.expected_coefficient_fit_scale);
  write_codec_scalar(
      "large.linkage_coeff_codec.q_refine_sweeps",
      command.coefficient_refinement_sweeps,
      command.expected_coefficient_refinement_sweeps);
  write_codec_scalar(
      "large.linkage_coeff_codec.q_refine_max_layer",
      command.coefficient_refinement_max_layer,
      command.expected_coefficient_refinement_max_layer);
  write_codec_scalar(
      "large.linkage_coeff_codec.q_refine_step_limit",
      command.coefficient_refinement_step_limit,
      command.expected_coefficient_refinement_step_limit);
  write_string("runtime.omp_places", command.omp_places);
  write_string("runtime.omp_proc_bind", command.omp_proc_bind);
  output << "eval.bench.quiet = " << static_cast<int>(command.quiet)
         << "\neval.metric_mode = " << static_cast<int>(command.metric_mode)
         << "\neval.metric_gt_topks = [";
  for (std::size_t index = 0U;
       index < command.metric_gt_topks.size(); ++index) {
    if (index != 0U) output << ',';
    output << command.metric_gt_topks[index];
  }
  output << "]\n";
  output << "eval.linkage.archive_eval_result = "
         << static_cast<int>(command.archive)
         << "\nlarge.archive_log = " << static_cast<int>(command.archive_log)
         << '\n';
}

}  // namespace stlqpp::app
