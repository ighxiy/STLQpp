#include "stlqpp/model/model_shape.h"
#include "build_command.h"

#include "stlqpp/application/config_schema.h"
#include "stlqpp/application/config_text.h"
#include "stlqpp/application/compute_backend.h"
#include "stlqpp/core/filesystem_path.h"
#include "stlqpp/core/execution/workload_thread_budget.h"
#include "stlqpp/platform/mapped_file.h"
#include "stlqpp/platform/npy_array.h"
#include "stlqpp/storage/trained_model_file.h"
#include "stlqpp/training/linkage/rsd/ranked_single_deviation.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace stlqpp::app {
namespace {

template <typename Integer>
bool ParseInteger(std::string_view text, Integer* output) {
  int base = 10;
  if (text.size() > 2U && text[0] == '0' &&
      (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2U);
    base = 16;
  }
  const auto parsed = std::from_chars(
      text.data(), text.data() + text.size(), *output, base);
  return parsed.ec == std::errc{} &&
      parsed.ptr == text.data() + text.size();
}

bool ParseDouble(std::string_view text, double* output) {
  std::istringstream input{std::string(text)};
  input.imbue(std::locale::classic());
  input >> std::noskipws >> *output;
  return input.eof() && !input.fail() && std::isfinite(*output);
}

bool ParsePercentCandidates(
    std::string_view text, std::vector<double>* fractions) {
  fractions->clear();
  std::string normalized = TrimConfigText(text);
  text = normalized;
  if (text.size() >= 2U && text.front() == '[' && text.back() == ']') {
    text.remove_prefix(1U);
    text.remove_suffix(1U);
  }
  while (!text.empty()) {
    const std::size_t comma = text.find(',');
    const std::string scalar = TrimConfigText(text.substr(0U, comma));
    double percent = 0.0;
    if (scalar.empty() || !ParseDouble(scalar, &percent) ||
        !(percent > 0.0) || percent > 100.0) {
      fractions->clear();
      return false;
    }
    fractions->push_back(percent / 100.0);
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1U);
  }
  return !fractions->empty();
}

bool ParseBool(std::string_view text, bool* output) {
  std::string normalized(text);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  text = normalized;
  if (text == "1" || text == "true") {
    *output = true;
    return true;
  }
  if (text == "0" || text == "false") {
    *output = false;
    return true;
  }
  return false;
}

bool ParseCleanupPreset(std::string_view text, CleanupPreset* output) {
  if (text == "none") *output = CleanupPreset::kNone;
  else if (text == "dev_all") *output = CleanupPreset::kDevAll;
  else if (text == "eval_both") *output = CleanupPreset::kEvalBoth;
  else if (text == "eval_float_min") {
    *output = CleanupPreset::kEvalFloatMin;
  } else if (text == "eval_int8_min") {
    *output = CleanupPreset::kEvalInt8Min;
  } else if (text == "custom") *output = CleanupPreset::kCustom;
  else return false;
  return true;
}

bool ParseSourceFormat(std::string_view text, SourceFormat* output) {
  std::string normalized(text);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  text = normalized;
  if (text.empty() || text == "auto") {
    *output = SourceFormat::kAuto;
    return true;
  }
  if (text == "bvecs" || text == "siftbin" || text == "u8") {
    *output = SourceFormat::kBvecsU8;
    return true;
  }
  if (text == "fvecs" || text == "float" || text == "f32") {
    *output = SourceFormat::kFvecsF32;
    return true;
  }
  if (text == "fbin") {
    *output = SourceFormat::kFbinF32;
    return true;
  }
  if (text == "npy") {
    *output = SourceFormat::kNpyFloat16;
    return true;
  }
  return false;
}

bool ParseLeastSquaresCostMode(
    std::string_view text, LeastSquaresCostMode* output) noexcept {
  if (text == "explicit_quadratic") {
    *output = LeastSquaresCostMode::kExplicitQuadratic;
    return true;
  }
  if (text == "normal_equation") {
    *output = LeastSquaresCostMode::kNormalEquation;
    return true;
  }
  return false;
}

const char* SourceFormatName(SourceFormat format) noexcept {
  switch (format) {
    case SourceFormat::kAuto:
      return "auto";
    case SourceFormat::kBvecsU8:
      return "bvecs";
    case SourceFormat::kFvecsF32:
      return "fvecs";
    case SourceFormat::kFbinF32:
      return "fbin";
    case SourceFormat::kNpyFloat16:
      return "npy";
  }
  return "auto";
}

bool ParseNorm8Mode(std::string_view text,
                    ReconstructionNorm8Mode* output) noexcept {
  if (text == "identity") {
    *output = ReconstructionNorm8Mode::kIdentityLloyd;
    return true;
  }
  if (text == "reference") {
    *output = ReconstructionNorm8Mode::kReferenceLloyd;
    return true;
  }
  if (text == "tail_weighted") {
    *output = ReconstructionNorm8Mode::kTailWeightedLloyd;
    return true;
  }
  return false;
}

bool ParseEvalRepresentations(
    std::string_view text, bool* automatic,
    std::vector<EvalRepresentation>* output) {
  EvalRepresentationSelection selection;
  if (!ParseEvalRepresentationSelection(text, &selection)) return false;
  *automatic = selection.automatic;
  *output = std::move(selection.explicit_values);
  return true;
}

const char* Norm8ModeName(ReconstructionNorm8Mode mode) noexcept {
  return ReconstructionNorm8ModeName(mode);
}

template <typename Consumer>
bool ParseCsv(std::string_view text, Consumer consume) {
  std::uint32_t position = 0U;
  while (!text.empty()) {
    const std::size_t separator = text.find(',');
    const std::string_view token = text.substr(0U, separator);
    if (token.empty() || !consume(position++, token)) {
      return false;
    }
    if (separator == std::string_view::npos) {
      return true;
    }
    text.remove_prefix(separator + 1U);
  }
  return false;
}

bool ParseU32List(std::string_view text,
                  std::vector<std::uint32_t>* output) {
  std::string normalized = TrimConfigText(text);
  text = normalized;
  if (text.size() < 2U || text.front() != '[' || text.back() != ']') {
    return false;
  }
  text.remove_prefix(1U);
  text.remove_suffix(1U);
  output->clear();
  if (TrimConfigText(text).empty()) return true;
  return ParseCsv(text, [&](std::uint32_t, std::string_view token) {
    std::uint32_t value = 0U;
    if (!ParseInteger(TrimConfigText(token), &value)) return false;
    output->push_back(value);
    return true;
  });
}

Status ValidateFixedProductField(
    std::string_view key, std::string_view value) {
  auto invalid = [&]() {
    return Status::InvalidArgument(
        "build command: DEV field " + std::string(key) +
        " disagrees with the admitted product semantics");
  };
  auto expect_bool = [&](bool expected) {
    bool parsed = false;
    return ParseBool(value, &parsed) && parsed == expected;
  };
  if (key == "large.linkage_coeff_codec.bits_per_layer" ||
      key == "large.linkage_coeff_codec.bits_per_layer_root" ||
      key == "large.linkage_coeff_codec.bits_per_layer_linkage") {
    return Status::Ok();
  }
  const bool valid =
      (key == "large.profile_timing" && expect_bool(false));
  return valid ? Status::Ok() : invalid();
}

bool ParseTopology(std::string_view text, TopologyCommand* output) {
  std::uint32_t count = 0U;
  const bool parsed = ParseCsv(
      text, [output, &count](std::uint32_t position, std::string_view token) {
        ++count;
        switch (position) {
          case 0U:
            return ParseInteger(token, &output->radial_layer_count);
          case 1U:
            return ParseDouble(token, &output->root_fraction);
          case 2U:
            return ParseInteger(token, &output->depth_k);
          case 3U:
            return ParseInteger(token, &output->knn_k);
          case 4U:
            return ParseInteger(token, &output->maximum_depth);
          case 5U:
            return ParseInteger(token, &output->seed);
          default:
            return false;
        }
      });
  return parsed && count == 6U && output->radial_layer_count != 0U &&
      output->root_fraction > 0.0 && output->root_fraction <= 1.0 &&
      output->depth_k != 0U && output->knn_k != 0U &&
      output->maximum_depth != 0U;
}

bool ParseVirtualRoots(std::string_view text,
                       VirtualRootCommand* output) {
  double overlap = 0.0;
  std::uint32_t count = 0U;
  const bool parsed = ParseCsv(
      text, [output, &overlap, &count](std::uint32_t position,
                              std::string_view token) {
        ++count;
        switch (position) {
          case 0U:
            return ParseDouble(token, &output->ratio);
          case 1U:
            return ParseInteger(token, &output->minimum);
          case 2U:
            return ParseInteger(token, &output->maximum);
          case 3U:
            return ParseInteger(token, &output->neighbor_count);
          case 4U:
            return ParseInteger(token, &output->local_connectivity);
          case 5U:
            return ParseDouble(token, &overlap);
          case 6U:
            return ParseInteger(token, &output->anchor_neighbor_k);
          default:
            return false;
        }
      });
  output->overlap_threshold = static_cast<float>(overlap);
  return parsed && count == 7U && output->ratio > 0.0 &&
      output->ratio <= 1.0 &&
      output->minimum != 0U && output->maximum >= output->minimum &&
      output->neighbor_count != 0U &&
      output->local_connectivity != 0U &&
      output->overlap_threshold >= 0.0F &&
      output->overlap_threshold <= 1.0F &&
      output->anchor_neighbor_k != 0U;
}

bool ParseRootInitialization(
    std::string_view text,
    RootInitializationCommand* output) {
  std::uint32_t count = 0U;
  const bool parsed = ParseCsv(
      text, [output, &count](std::uint32_t position,
                             std::string_view token) {
        ++count;
        switch (position) {
          case 0U:
            return ParseInteger(token, &output->maximum_iterations);
          case 1U:
            return ParseDouble(token, &output->tolerance);
          case 2U:
            return ParseInteger(token, &output->kmeans_parallel_rounds);
          case 3U:
            return ParseDouble(token, &output->kmeans_parallel_oversample);
          case 4U:
            return ParseInteger(token, &output->candidate_cap);
          case 5U:
            return ParseDouble(token, &output->initial_weight);
          case 6U:
            return ParseDouble(token, &output->minimum_weight);
          case 7U:
            return ParseDouble(token, &output->outlier_quantile);
          case 8U:
            return ParseDouble(token, &output->cost_threshold);
          case 9U:
            return ParseDouble(token, &output->annealing_factor);
          case 10U:
            return ParseInteger(token, &output->warmup_iterations);
          case 11U:
            return ParseInteger(token, &output->quantile_bins);
          case 12U:
            return ParseInteger(token, &output->large_layer_threshold);
          case 13U:
            return ParseInteger(token, &output->large_layer_top_coarse);
          default:
            return false;
        }
      });
  return parsed && count == 14U && output->maximum_iterations != 0U &&
      output->tolerance >= 0.0 &&
      output->kmeans_parallel_rounds != 0U &&
      output->kmeans_parallel_oversample >= 1.0 &&
      output->initial_weight > 0.0 && output->minimum_weight > 0.0 &&
      output->minimum_weight <= output->initial_weight &&
      output->outlier_quantile > 0.0 && output->outlier_quantile < 1.0 &&
      output->cost_threshold > 0.0 &&
      output->annealing_factor > 0.0 &&
      output->annealing_factor <= 1.0 &&
      output->quantile_bins != 0U &&
      output->large_layer_threshold != 0U &&
      output->large_layer_top_coarse != 0U;
}

Status InvalidCommand(std::string message) {
  return Status::InvalidArgument("build command: " + std::move(message));
}

template <typename Value>
Status InvalidField(std::string_view field, const Value& value,
                    std::string_view requirement) {
  std::ostringstream message;
  message.imbue(std::locale::classic());
  message << std::setprecision(17) << field << "=[" << value
          << "] (expected " << requirement << ')';
  return InvalidCommand(message.str());
}

}  // namespace

StatusOr<BuildCommand> ParseFlatBuildCommand(int argc, char** argv) {
  BuildCommand output;
  std::vector<std::string_view> seen_options;
  bool have_m = false;
  bool have_training_format = false;
  bool have_base_format = false;
  bool have_dataset = false;
  bool have_train = false;
  bool have_train_rows = false;
  bool have_base = false;
  bool have_base_rows = false;
  bool have_nlist = false;
  bool have_init_samples = false;
  bool have_root_initialization = false;
  bool have_train_seed = false;
  bool have_training_rounds = false;
  bool have_base_seed = false;
  bool have_good_fraction = false;
  bool have_bad_cluster_weight = false;
  bool have_training_topology = false;
  bool have_base_topology = false;
  bool have_virtual_roots = false;
  bool have_work = false;
  bool have_output = false;
  bool have_archive_log = false;
  bool have_threads = false;
  bool have_base_list_threads = false;
  bool have_root_moment_partitions = false;
  bool have_rotation_moment_partitions = false;
  bool have_initialization_edge_moment_shards = false;
  bool have_training_edge_moment_shards = false;
  bool have_device = false;
  bool have_use_cuda = false;
  bool have_training_block_rows = false;
  bool have_base_block_rows = false;
  bool have_cids_per_bucket = false;
  bool have_basic_inflight_blocks = false;
  bool have_basic_async_io_enabled = false;
  bool have_basic_async_io_depth = false;
  bool have_basic_async_io_mebibytes = false;
  bool have_basic_async_write_enabled = false;
  bool have_basic_async_write_depth = false;
  bool have_basic_async_write_mebibytes = false;
  bool have_cuda_mode = false;
  bool have_cuda_allow_tf32 = false;
  bool have_root_score_buffer = false;
  bool have_root_normalized_cache = false;
  bool have_linkage_xc0_chunk = false;
  bool have_linkage_workspace = false;
  bool have_init_linkage_xc0_chunk = false;
  bool have_init_linkage_workspace = false;
  bool have_coefficient_first_quantiles = false;
  bool have_coefficient_rest_quantiles = false;
  bool have_norm8_iterations = false;
  bool have_topology_inflight = false;
  bool have_linkage_async_io_enabled = false;
  bool have_linkage_async_io_depth = false;
  bool have_linkage_scheduler_workers = false;
  bool have_linkage_scheduler_queue = false;
  bool have_linkage_cache_buffers = false;
  bool have_linkage_cache_mebibytes = false;
  bool have_linkage_safety_reserve = false;
  bool have_linkage_inner_rows = false;
  bool have_linkage_max_pairs = false;
  bool have_linkage_init_max_pairs = false;
  bool have_linkage_many_nodes_preflush_pairs = false;
  bool have_linkage_same_block_rows = false;
  bool have_linkage_same_minimum_pairs = false;
  bool have_linkage_same_window_maximum_pairs = false;
  bool have_linkage_same_maximum_pending_nodes = false;
  bool have_linkage_same_preflush_pairs = false;
  bool have_linkage_same_preflush_nodes = false;
  bool have_linkage_tiny_limit = false;
  bool have_linkage_bad_window_rows = false;
  bool have_linkage_bad_window_max_pairs = false;
  bool have_final_spool_clusters = false;
  bool have_final_spool_rows = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (std::find(seen_options.begin(), seen_options.end(), option) !=
        seen_options.end()) {
      return InvalidCommand("duplicate option " + std::string(option));
    }
    seen_options.push_back(option);
    if (index + 1 >= argc) {
      return InvalidCommand("missing value for " + std::string(option));
    }
    const std::string_view value = argv[++index];
    // Early returns keep this dispatch below MSVC's block-nesting limit.
    const auto parsed_option = [&]() -> std::optional<bool> {
      bool parsed = true;
      if (option == "--train-enabled") {
        parsed = ParseBool(value, &output.train_enabled);
        return parsed;
      }
      if (option == "--train-init-enabled") {
        parsed = ParseBool(value, &output.train_initialization_enabled);
        return parsed;
      }
      if (option == "--train-init-linkage-mode") {
        parsed = value == "legacy_root_only" || value == "fast_init";
        if (parsed) output.init_linkage_mode = value;
        return parsed;
      }
      if (option == "--base-encode-enabled") {
        parsed = ParseBool(value, &output.base_encode_enabled);
        return parsed;
      }
      if (option == "--base-linkage-enabled") {
        parsed = ParseBool(value, &output.base_linkage_enabled);
        return parsed;
      }
      if (option == "--eval-only") {
        parsed = ParseBool(value, &output.eval_only);
        return parsed;
      }
      if (option == "--train-checkpoint-enabled") {
        parsed = ParseBool(value, &output.train_checkpoint_enabled);
        return parsed;
      }
      if (option == "--train-checkpoint-every-r") {
        parsed = ParseInteger(value, &output.train_checkpoint_every_r);
        return parsed;
      }
      if (option == "--train-checkpoint-file") {
        output.train_checkpoint_file = value;
        return parsed;
      }
      if (option == "--output-prefix") {
        output.output_prefix = value;
        return parsed;
      }
      if (option == "--reused-base-list-directory") {
        output.reused_base_list_directory = value;
        return parsed;
      }
      if (option == "--base-basic-checkpoint-enabled") {
        parsed = ParseBool(value, &output.base_basic_checkpoint_enabled);
        return parsed;
      }
      if (option == "--linkage-checkpoint-enabled") {
        parsed = ParseBool(value, &output.linkage_checkpoint_enabled);
        return parsed;
      }
      if (option == "--m") {
        parsed = ParseInteger(value, &output.m);
        have_m = true;
        return parsed;
      }
      if (option == "--train-format") {
        parsed = ParseSourceFormat(value, &output.training_format);
        have_training_format = true;
        return parsed;
      }
      if (option == "--base-format") {
        parsed = ParseSourceFormat(value, &output.base_format);
        have_base_format = true;
        return parsed;
      }
      if (option == "--dataset") {
        output.dataset_name = value;
        have_dataset = true;
        return parsed;
      }
      if (option == "--data-root") {
        output.dataset_data_root = value;
        return parsed;
      }
      if (option == "--train") {
        output.training_file = value;
        have_train = true;
        return parsed;
      }
      if (option == "--train-first-row") {
        parsed = ParseInteger(value, &output.training_first_row);
        return parsed;
      }
      if (option == "--train-rows") {
        parsed = ParseInteger(value, &output.training_rows);
        have_train_rows = true;
        return parsed;
      }
      if (option == "--base") {
        output.base_file = value;
        have_base = true;
        return parsed;
      }
      if (option == "--base-first-row") {
        parsed = ParseInteger(value, &output.base_first_row);
        return parsed;
      }
      if (option == "--base-rows") {
        parsed = ParseInteger(value, &output.base_rows);
        have_base_rows = true;
        return parsed;
      }
      if (option == "--dimension") {
        parsed = ParseInteger(value, &output.dimension);
        return parsed;
      }
      if (option == "--nlist") {
        parsed = ParseInteger(value, &output.nlist);
        have_nlist = true;
        return parsed;
      }
      if (option == "--init-samples") {
        parsed = ParseInteger(value, &output.initialization_samples);
        have_init_samples = true;
        return parsed;
      }
      if (option == "--root-initialization") {
        parsed = ParseRootInitialization(value, &output.root_initialization);
        have_root_initialization = true;
        return parsed;
      }
      if (option == "--kmeans-iters") {
        parsed = ParseInteger(value, &output.root_initialization.maximum_iterations);
        return parsed;
      }
      if (option == "--kmeans-tol") {
        parsed = ParseDouble(value, &output.root_initialization.tolerance);
        return parsed;
      }
      if (option == "--kmeans-init") {
        parsed = value == "kmeansll" || value == "random" ||
            value == "kmeans++";
        if (parsed) output.root_initialization.seed_method = value;
        return parsed;
      }
      if (option == "--kmeansll-rounds") {
        parsed = ParseInteger(value, &output.root_initialization.kmeans_parallel_rounds);
        return parsed;
      }
      if (option == "--kmeansll-oversample") {
        parsed = ParseDouble(value, &output.root_initialization.kmeans_parallel_oversample);
        return parsed;
      }
      if (option == "--kmeansll-candidate-cap") {
        parsed = ParseInteger(value, &output.root_initialization.candidate_cap);
        return parsed;
      }
      if (option == "--kmeans-initial-weight") {
        parsed = ParseDouble(value, &output.root_initialization.initial_weight);
        return parsed;
      }
      if (option == "--kmeans-min-weight") {
        parsed = ParseDouble(value, &output.root_initialization.minimum_weight);
        return parsed;
      }
      if (option == "--kmeans-outlier-quantile") {
        parsed = ParseDouble(value, &output.root_initialization.outlier_quantile);
        return parsed;
      }
      if (option == "--kmeans-cost-threshold") {
        parsed = ParseDouble(value, &output.root_initialization.cost_threshold);
        return parsed;
      }
      if (option == "--kmeans-annealing-factor") {
        parsed = ParseDouble(value, &output.root_initialization.annealing_factor);
        return parsed;
      }
      if (option == "--kmeans-warmup-iters") {
        parsed = ParseInteger(value, &output.root_initialization.warmup_iterations);
        return parsed;
      }
      if (option == "--kmeans-quantile-bins") {
        parsed = ParseInteger(value, &output.root_initialization.quantile_bins);
        return parsed;
      }
      if (option == "--kmeans-large-k-threshold") {
        parsed = ParseInteger(value, &output.root_initialization.large_layer_threshold);
        return parsed;
      }
      if (option == "--kmeans-large-k-top-coarse") {
        parsed = ParseInteger(value, &output.root_initialization.large_layer_top_coarse);
        return parsed;
      }
      if (option == "--kmeans-large-k-hier2-enable") {
        parsed = ParseBool(value, &output.root_initialization.large_layer_hierarchy_enabled);
        return parsed;
      }
      if (option == "--kmeans-large-k-fixed-k1") {
        parsed = ParseInteger(value, &output.root_initialization.large_layer_fixed_coarse_count);
        return parsed;
      }
      if (option == "--kmeans-large-k-k1-min") {
        parsed = ParseInteger(value, &output.root_initialization.large_layer_minimum_coarse_count);
        return parsed;
      }
      if (option == "--kmeans-large-k-k1-max") {
        parsed = ParseInteger(value, &output.root_initialization.large_layer_maximum_coarse_count);
        return parsed;
      }
      if (option == "--kmeans-large-k-k1-pow2") {
        parsed = ParseBool(value, &output.root_initialization.large_layer_round_coarse_to_power_of_two);
        return parsed;
      }
      if (option == "--train-seed") {
        parsed = ParseInteger(value, &output.train_seed);
        have_train_seed = true;
        return parsed;
      }
      if (option == "--rounds") {
        parsed = ParseInteger(value, &output.training_rounds);
        have_training_rounds = true;
        return parsed;
      }
      if (option == "--train-use-opq-rotation") {
        parsed = ParseBool(value, &output.train_use_opq_rotation);
        return parsed;
      }
      if (option == "--train-log-metrics") {
        parsed = ParseBool(value, &output.train_log_metrics);
        return parsed;
      }
      if (option == "--train-log-linkage-pre-c1") {
        parsed = ParseBool(value, &output.train_log_linkage_pre_c1);
        return parsed;
      }
      if (option == "--train-basic-ils-iterations") {
        parsed = ParseInteger(value, &output.train_basic_ils_iterations);
        return parsed;
      }
      if (option == "--train-basic-icm-iterations") {
        parsed = ParseInteger(value, &output.train_basic_icm_iterations);
        return parsed;
      }
      if (option == "--train-basic-perturbations") {
        parsed = ParseInteger(value, &output.train_basic_perturbations);
        return parsed;
      }
      if (option == "--training-basic-rsd-rank") {
        parsed = ParseInteger(value, &output.training_basic_rsd_rank);
        return parsed;
      }
      if (option == "--least-squares-cost-mode") {
        parsed = ParseLeastSquaresCostMode(
            value, &output.least_squares_cost_mode);
        return parsed;
      }
      if (option == "--kmeansll-seed") {
        parsed = ParseInteger(
            value, &output.root_initialization.kmeans_parallel_seed);
        return parsed;
      }
      if (option == "--base-seed") {
        parsed = ParseInteger(value, &output.base_seed);
        have_base_seed = true;
        return parsed;
      }
      if (option == "--base-basic-ils-iterations") {
        parsed = ParseInteger(value, &output.base_basic_ils_iterations);
        return parsed;
      }
      if (option == "--base-basic-icm-iterations") {
        parsed = ParseInteger(value, &output.base_basic_icm_iterations);
        return parsed;
      }
      if (option == "--base-basic-perturbations") {
        parsed = ParseInteger(value, &output.base_basic_perturbations);
        return parsed;
      }
      if (option == "--base-basic-rsd-rank") {
        parsed = ParseInteger(value, &output.base_basic_rsd_rank);
        return parsed;
      }
      if (option == "--good-cluster-fraction") {
        parsed = ParseDouble(value, &output.good_cluster_fraction);
        have_good_fraction = true;
        return parsed;
      }
      if (option == "--bad-cluster-weight") {
        parsed = ParseDouble(value, &output.bad_cluster_training_weight);
        have_bad_cluster_weight = true;
        return parsed;
      }
      if (option == "--training-topology") {
        parsed = ParseTopology(value, &output.training_topology);
        have_training_topology = true;
        return parsed;
      }
      if (option == "--training-topology-layers") {
        parsed = ParseInteger(value, &output.training_topology.radial_layer_count);
        return parsed;
      }
      if (option == "--training-topology-root-fraction") {
        parsed = ParseDouble(value, &output.training_topology.root_fraction);
        return parsed;
      }
      if (option == "--training-topology-depth-k") {
        parsed = ParseInteger(value, &output.training_topology.depth_k);
        return parsed;
      }
      if (option == "--training-topology-knn-k") {
        parsed = ParseInteger(value, &output.training_topology.knn_k);
        return parsed;
      }
      if (option == "--training-topology-maximum-depth") {
        parsed = ParseInteger(value, &output.training_topology.maximum_depth);
        return parsed;
      }
      if (option == "--training-topology-seed") {
        parsed = ParseInteger(value, &output.training_topology.seed);
        return parsed;
      }
      if (option == "--training-linkage-rsd-rank") {
        parsed = ParseInteger(value, &output.training_topology.rsd_rank);
        return parsed;
      }
      if (option == "--init-linkage-icm-rounds") {
        parsed = ParseInteger(value, &output.init_linkage_icm_rounds);
        return parsed;
      }
      if (option == "--training-linkage-icm-rounds") {
        parsed = ParseInteger(value, &output.training_topology.icm_rounds);
        return parsed;
      }
      if (option == "--base-topology") {
        parsed = ParseTopology(value, &output.base_topology);
        have_base_topology = true;
        return parsed;
      }
      if (option == "--base-topology-layers") {
        parsed = ParseInteger(value, &output.base_topology.radial_layer_count);
        return parsed;
      }
      if (option == "--base-topology-root-fraction") {
        parsed = ParseDouble(value, &output.base_topology.root_fraction);
        return parsed;
      }
      if (option == "--base-topology-depth-k") {
        parsed = ParseInteger(value, &output.base_topology.depth_k);
        return parsed;
      }
      if (option == "--base-topology-knn-k") {
        parsed = ParseInteger(value, &output.base_topology.knn_k);
        return parsed;
      }
      if (option == "--base-topology-maximum-depth") {
        parsed = ParseInteger(value, &output.base_topology.maximum_depth);
        return parsed;
      }
      if (option == "--base-topology-seed") {
        parsed = ParseInteger(value, &output.base_topology.seed);
        return parsed;
      }
      if (option == "--base-linkage-rsd-rank") {
        parsed = ParseInteger(value, &output.base_topology.rsd_rank);
        return parsed;
      }
      if (option == "--base-linkage-icm-rounds") {
        parsed = ParseInteger(value, &output.base_topology.icm_rounds);
        return parsed;
      }
      if (option == "--train-linkage-bad-cluster-knn-backend") {
        output.training_topology.bad_cluster_knn_backend = value;
        return parsed;
      }
      if (option == "--train-linkage-bad-cluster-cpu-exact-max-points") {
        parsed = ParseInteger(value, &output.training_topology.bad_cluster_cpu_exact_max_points);

        return parsed;
      }
      if (option == "--base-linkage-bad-cluster-knn-backend") {
        output.base_topology.bad_cluster_knn_backend = value;
        return parsed;
      }
      if (option == "--base-linkage-bad-cluster-cpu-exact-max-points") {
        parsed = ParseInteger(value, &output.base_topology.bad_cluster_cpu_exact_max_points);

        return parsed;
      }
      if (option == "--hnsw-m") {
        parsed = ParseInteger(value, &output.hnsw_m);
        return parsed;
      }
      if (option == "--hnsw-ef-construction-cap") {
        parsed = ParseInteger(value, &output.hnsw_ef_construction_cap);
        return parsed;
      }
      if (option == "--hnsw-ef-search") {
        parsed = ParseInteger(value, &output.hnsw_ef_search);
        return parsed;
      }
      if (option == "--virtual-roots") {
        parsed = ParseVirtualRoots(value, &output.virtual_roots);
        have_virtual_roots = true;
        return parsed;
      }
      if (option == "--virtual-ratio") {
        parsed = ParseDouble(value, &output.virtual_roots.ratio);
        return parsed;
      }
      if (option == "--virtual-minimum") {
        parsed = ParseInteger(value, &output.virtual_roots.minimum);
        return parsed;
      }
      if (option == "--virtual-maximum") {
        parsed = ParseInteger(value, &output.virtual_roots.maximum);
        return parsed;
      }
      if (option == "--virtual-neighbor-count") {
        parsed = ParseInteger(value, &output.virtual_roots.neighbor_count);
        return parsed;
      }
      if (option == "--virtual-local-connectivity") {
        parsed = ParseInteger(value, &output.virtual_roots.local_connectivity);
        return parsed;
      }
      if (option == "--virtual-overlap-threshold") {
        double overlap = 0.0;
        parsed = ParseDouble(value, &overlap);
        output.virtual_roots.overlap_threshold = static_cast<float>(overlap);
        return parsed;
      }
      if (option == "--virtual-anchor-neighbor-k") {
        parsed = ParseInteger(value, &output.virtual_roots.anchor_neighbor_k);
        return parsed;
      }
      if (option == "--virtual-prefer-peaks") {
        parsed = ParseBool(value, &output.virtual_roots.prefer_density_peaks);
        return parsed;
      }
      if (option == "--virtual-enabled") {
        parsed = ParseBool(value, &output.virtual_roots.enabled);
        return parsed;
      }
      if (option == "--virtual-use-fixed-per-cluster") {
        parsed = ParseBool(value, &output.virtual_roots.use_fixed_per_cluster);
        return parsed;
      }
      if (option == "--virtual-fixed-per-cluster") {
        parsed = ParseInteger(value, &output.virtual_roots.fixed_per_cluster);
        return parsed;
      }
      if (option == "--work") {
        output.work_directory = value;
        have_work = true;
        return parsed;
      }
      if (option == "--output") {
        output.output_directory = value;
        have_output = true;
        return parsed;
      }
      if (option == "--archive-log") {
        parsed = ParseBool(value, &output.archive_log);
        have_archive_log = true;
        return parsed;
      }
      if (option == "--protect-existing-outputs") {
        parsed = ParseBool(value, &output.protect_existing_outputs);
        return parsed;
      }
      if (option == "--control-evidence") {
        output.control_evidence = value;
        return parsed;
      }
      if (option == "--resume-initialized") {
        output.resume_initialized_checkpoint = value;
        return parsed;
      }
      if (option == "--threads") {
        parsed = ParseInteger(value, &output.cpu_threads);
        have_threads = true;
        return parsed;
      }
      if (option == "--omp-places") {
        output.omp_places = value;
        return parsed;
      }
      if (option == "--omp-proc-bind") {
        output.omp_proc_bind = value;
        return parsed;
      }
      if (option == "--base-list-threads") {
        parsed = ParseInteger(value, &output.base_list_threads);
        have_base_list_threads = true;
        return parsed;
      }
      if (option == "--root-moment-partitions") {
        parsed = ParseInteger(value, &output.root_moment_partitions);
        have_root_moment_partitions = true;
        return parsed;
      }
      if (option == "--rotation-moment-partitions") {
        parsed = ParseInteger(value, &output.rotation_moment_partitions);
        have_rotation_moment_partitions = true;
        return parsed;
      }
      if (option == "--init-edge-moment-shards") {
        parsed = ParseInteger(
            value, &output.initialization_edge_moment_shards);
        have_initialization_edge_moment_shards = true;
        return parsed;
      }
      if (option == "--train-edge-moment-shards") {
        parsed = ParseInteger(value, &output.training_edge_moment_shards);
        have_training_edge_moment_shards = true;
        return parsed;
      }
      if (option == "--cuda-device") {
        parsed = ParseInteger(value, &output.cuda_device);
        have_device = true;
        return parsed;
      }
      if (option == "--use-cuda") {
        parsed = ParseBool(value, &output.use_cuda);
        have_use_cuda = true;
        return parsed;
      }
      if (option == "--gpu-norm-enable") {
        parsed = ParseBool(value, &output.gpu_norm_enable);
        return parsed;
      }
      if (option == "--train-block-rows") {
        parsed = ParseInteger(value, &output.training_block_rows);
        have_training_block_rows = true;
        return parsed;
      }
      if (option == "--base-block-rows") {
        parsed = ParseInteger(value, &output.base_block_rows);
        have_base_block_rows = true;
        return parsed;
      }
      if (option == "--cids-per-bucket") {
        parsed = ParseInteger(value, &output.cids_per_bucket);
        have_cids_per_bucket = true;
        return parsed;
      }
      if (option == "--bucket-flush-mib") {
        parsed = ParseInteger(value, &output.bucket_flush_mebibytes);
        return parsed;
      }
      if (option == "--cleanup-enabled") {
        parsed = ParseBool(value, &output.cleanup.enabled);
        return parsed;
      }
      if (option == "--cleanup-preset") {
        parsed = ParseCleanupPreset(value, &output.cleanup.preset);
        return parsed;
      }
      if (option == "--cleanup-keep-base-basic") {
        parsed = ParseInteger(value, &output.cleanup.keep_base_basic);
        return parsed;
      }
      if (option == "--cleanup-keep-base-basic-cluster-id") {
        parsed = ParseInteger(
            value, &output.cleanup.keep_base_basic_cluster_id);
        return parsed;
      }
      if (option == "--cleanup-keep-base-basic-ivf-lists") {
        parsed = ParseInteger(
            value, &output.cleanup.keep_base_basic_ivf_lists);
        return parsed;
      }
      if (option == "--cleanup-keep-base-list") {
        parsed = ParseInteger(value, &output.cleanup.keep_base_list);
        return parsed;
      }
      if (option == "--cleanup-keep-base-list-raw") {
        parsed = ParseInteger(value, &output.cleanup.keep_base_list_raw);
        return parsed;
      }
      if (option == "--cleanup-keep-linkage-list-f32") {
        parsed = ParseInteger(
            value, &output.cleanup.keep_linkage_list_f32);
        return parsed;
      }
      if (option == "--cleanup-keep-linkage-parent-u32") {
        parsed = ParseInteger(
            value, &output.cleanup.keep_linkage_parent_u32);
        return parsed;
      }
      if (option == "--base-linkage-store-coefficients-f32") {
        parsed = ParseBool(
            value, &output.base_linkage_store_coefficients_f32);
        return parsed;
      }
      if (option == "--basic-inflight-blocks") {
        parsed = ParseInteger(value, &output.basic_inflight_blocks);
        have_basic_inflight_blocks = true;
        return parsed;
      }
      if (option == "--basic-async-io-enabled") {
        parsed = ParseBool(value, &output.basic_async_io_enabled);
        have_basic_async_io_enabled = true;
        return parsed;
      }
      if (option == "--basic-async-io-depth") {
        parsed = ParseInteger(value, &output.basic_async_io_depth);
        have_basic_async_io_depth = true;
        return parsed;
      }
      if (option == "--basic-async-io-mib") {
        parsed = ParseInteger(value, &output.basic_async_io_mebibytes);
        have_basic_async_io_mebibytes = true;
        return parsed;
      }
      if (option == "--basic-async-write-enabled") {
        parsed = ParseBool(value, &output.basic_async_write_enabled);
        have_basic_async_write_enabled = true;
        return parsed;
      }
      if (option == "--basic-async-write-depth") {
        parsed = ParseInteger(value, &output.basic_async_write_depth);
        have_basic_async_write_depth = true;
        return parsed;
      }
      if (option == "--basic-async-write-mib") {
        parsed = ParseInteger(value, &output.basic_async_write_mebibytes);
        have_basic_async_write_mebibytes = true;
        return parsed;
      }
      if (option == "--cuda-mode") {
        if (value == "fast") {
          output.cuda_mode = CudaMode::kFast;
        } else if (value == "strict") {
          output.cuda_mode = CudaMode::kStrict;
        } else {
          parsed = false;
        }
        have_cuda_mode = true;
        return parsed;
      }
      if (option == "--cuda-allow-tf32") {
        parsed = ParseBool(value, &output.requested_cuda_allow_tf32);
        have_cuda_allow_tf32 = true;
        return parsed;
      }
      if (option == "--root-score-buffer-mib") {
        parsed = ParseInteger(value, &output.root_score_buffer_mebibytes);
        have_root_score_buffer = true;
        return parsed;
      }
      if (option == "--root-normalized-cache-mib") {
        parsed = ParseInteger(
            value, &output.root_normalized_cache_mebibytes);
        have_root_normalized_cache = true;
        return parsed;
      }
      if (option == "--linkage-large-root-xc0-chunk-mib") {
        parsed = ParseInteger(
            value, &output.linkage_large_root_xc0_chunk_mebibytes);
        have_linkage_xc0_chunk = true;
        return parsed;
      }
      if (option == "--linkage-large-root-workspace-mib") {
        parsed = ParseInteger(
            value, &output.linkage_large_root_workspace_mebibytes);
        have_linkage_workspace = true;
        return parsed;
      }
      if (option == "--init-linkage-large-root-xc0-chunk-mib") {
        parsed = ParseInteger(
            value, &output.init_linkage_large_root_xc0_chunk_mebibytes);
        have_init_linkage_xc0_chunk = true;
        return parsed;
      }
      if (option == "--init-linkage-large-root-workspace-mib") {
        parsed = ParseInteger(
            value, &output.init_linkage_large_root_workspace_mebibytes);
        have_init_linkage_workspace = true;
        return parsed;
      }
      if (option == "--linkage-chunk-max-pairs") {
        parsed = ParseInteger(value, &output.linkage_chunk_max_pairs);
        return parsed;
      }
      if (option == "--init-linkage-chunk-max-pairs") {
        parsed = ParseInteger(value, &output.init_linkage_chunk_max_pairs);
        return parsed;
      }
      if (option == "--coefficient-quantile") {
        double candidate = 0.0;
        parsed = ParseDouble(value, &candidate) && candidate > 0.0 &&
            candidate <= 1.0;
        if (parsed) {
          output.coefficient_first_magnitude_quantiles = {candidate};
          output.coefficient_rest_magnitude_quantiles = {candidate};
        }
        have_coefficient_first_quantiles = true;
        have_coefficient_rest_quantiles = true;
        return parsed;
      }
      if (option == "--coefficient-p-first-candidates") {
        parsed = ParsePercentCandidates(
            value, &output.coefficient_first_magnitude_quantiles);
        have_coefficient_first_quantiles = true;
        return parsed;
      }
      if (option == "--coefficient-p-rest-candidates") {
        parsed = ParsePercentCandidates(
            value, &output.coefficient_rest_magnitude_quantiles);
        have_coefficient_rest_quantiles = true;
        return parsed;
      }
      if (option == "--coefficient-codec-enabled") {
        parsed = ParseBool(value, &output.coefficient_codec_enabled);
        return parsed;
      }
      if (option == "--coefficient-weighted-quantile") {
        parsed = ParseBool(value, &output.coefficient_weighted_quantile);
        return parsed;
      }
      if (option == "--coefficient-allow-clipping") {
        parsed = ParseBool(value, &output.coefficient_allow_clipping);
        return parsed;
      }
      if (option == "--coefficient-fit-scale") {
        parsed = ParseBool(value, &output.coefficient_fit_scale);
        return parsed;
      }
      if (option == "--coefficient-refinement-sweeps") {
        parsed = ParseInteger(value, &output.coefficient_refinement_sweeps);
        return parsed;
      }
      if (option == "--coefficient-refinement-max-layer") {
        parsed = ParseInteger(value, &output.coefficient_refinement_max_layer);
        return parsed;
      }
      if (option == "--coefficient-refinement-step-limit") {
        parsed = ParseInteger(value, &output.coefficient_refinement_step_limit);
        return parsed;
      }
      if (option == "--coefficient-bits-per-layer") {
        parsed = ParseU32List(value, &output.coefficient_bits_per_layer);
        return parsed;
      }
      if (option == "--coefficient-bits-per-layer-root") {
        parsed = ParseU32List(value, &output.coefficient_bits_per_layer_root);
        return parsed;
      }
      if (option == "--coefficient-bits-per-layer-linkage") {
        parsed = ParseU32List(value, &output.coefficient_bits_per_layer_linkage);
        return parsed;
      }
      if (option == "--disk-norm2-mode") {
        output.eval_disk_norm2_mode = value;
        parsed = value == "lut" || value == "exact";
        return parsed;
      }
      if (option == "--norm8-mode") {
        parsed = ParseNorm8Mode(value, &output.norm8_mode);
        return parsed;
      }
      if (option == "--norm8-lloyd-iterations") {
        parsed = ParseInteger(value, &output.norm8_lloyd_iterations);
        have_norm8_iterations = true;
        return parsed;
      }
      if (option == "--base-eval-enabled") {
        parsed = ParseBool(value, &output.eval_base_enabled);
        return parsed;
      }
      if (option == "--base-eval-use-ivf") {
        parsed = ParseBool(value, &output.eval_base_use_ivf);
        return parsed;
      }
      if (option == "--base-eval-nprobe") {
        parsed = ParseInteger(value, &output.eval_base_nprobe);
        return parsed;
      }
      if (option == "--base-eval-warmup") {
        parsed = ParseInteger(value, &output.eval_base_warmup);
        return parsed;
      }
      if (option == "--base-eval-repeat") {
        parsed = ParseInteger(value, &output.eval_base_repeat);
        return parsed;
      }
      if (option == "--linkage-eval-enabled") {
        parsed = ParseBool(value, &output.eval_linkage_enabled);
        return parsed;
      }
      if (option == "--eval-modes") {
        parsed = ParseEvalRepresentations(
            value, &output.eval_representations_auto,
            &output.eval_representations);
        return parsed;
      }
      if (option == "--eval-score-mode") {
        parsed = value == "l2" || value == "cosine_recon" ||
                 value == "mips_recon";
        if (parsed) output.eval_score_mode = value;
        return parsed;
      }
      if (option == "--topology-inflight-clusters") {
        parsed = ParseInteger(value, &output.topology_inflight_clusters);
        have_topology_inflight = true;
        return parsed;
      }
      if (option == "--linkage-async-io-enabled") {
        parsed = ParseBool(value, &output.linkage_async_io_enabled);
        have_linkage_async_io_enabled = true;
        return parsed;
      }
      if (option == "--linkage-async-io-depth") {
        parsed = ParseInteger(value, &output.linkage_async_io_depth);
        have_linkage_async_io_depth = true;
        return parsed;
      }
      if (option == "--linkage-scheduler-enabled") {
        parsed = ParseBool(value, &output.linkage_scheduler_enabled);
        return parsed;
      }
      if (option ==
                 "--linkage-scheduler-transport-batching-enabled") {
        parsed = ParseBool(
            value, &output.linkage_scheduler_transport_batching_enabled);
        return parsed;
      }
      if (option == "--linkage-scheduler-stable") {
        parsed = ParseBool(value, &output.linkage_scheduler_stable);
        return parsed;
      }
      if (option == "--linkage-direct-pool-size") {
        parsed = ParseInteger(value, &output.linkage_direct_pool_size);
        return parsed;
      }
      if (option == "--init-linkage-direct-pool-size") {
        parsed = ParseInteger(
            value, &output.initialization_linkage_direct_pool_size);
        return parsed;
      }
      if (option == "--linkage-eval-async-pinned-mib") {
        parsed = ParseInteger(
            value, &output.linkage_eval_async_pinned_mebibytes);
        return parsed;
      }
      if (option == "--init-linkage-eval-async-pinned-mib") {
        parsed = ParseInteger(
            value,
            &output.initialization_linkage_eval_async_pinned_mebibytes);
        return parsed;
      }
      if (option == "--linkage-scheduler-workers") {
        parsed = ParseInteger(value, &output.linkage_scheduler_workers);
        have_linkage_scheduler_workers = true;
        return parsed;
      }
      if (option == "--linkage-scheduler-queue-capacity") {
        parsed = ParseInteger(
            value, &output.linkage_scheduler_queue_capacity);
        have_linkage_scheduler_queue = true;
        return parsed;
      }
      if (option == "--linkage-rfull-cache-buffers") {
        parsed = ParseInteger(value, &output.linkage_rfull_cache_buffers);
        have_linkage_cache_buffers = true;
        return parsed;
      }
      if (option == "--linkage-rfull-cache-mib") {
        parsed = ParseInteger(value, &output.linkage_rfull_cache_mebibytes);
        have_linkage_cache_mebibytes = true;
        return parsed;
      }
      if (option == "--linkage-safety-reserve-mib") {
        parsed = ParseInteger(
            value, &output.linkage_safety_reserve_mebibytes);
        have_linkage_safety_reserve = true;
        return parsed;
      }
      if (option == "--linkage-inner-batch-rows") {
        parsed = ParseInteger(value, &output.linkage_inner_batch_rows);
        have_linkage_inner_rows = true;
        return parsed;
      }
      if (option == "--linkage-max-batch-pairs") {
        parsed = ParseInteger(value, &output.linkage_max_batch_pairs);
        have_linkage_max_pairs = true;
        return parsed;
      }
      if (option == "--linkage-init-max-batch-pairs") {
        parsed = ParseInteger(value, &output.linkage_init_max_batch_pairs);
        have_linkage_init_max_pairs = true;
        return parsed;
      }
      if (option == "--linkage-many-nodes-preflush-target-pairs") {
        parsed = ParseInteger(
            value, &output.linkage_many_nodes_preflush_target_pairs);
        have_linkage_many_nodes_preflush_pairs = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-block-rows") {
        parsed = ParseInteger(value, &output.linkage_same_layer_block_rows);
        have_linkage_same_block_rows = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-minimum-pairs") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_minimum_pairs);
        have_linkage_same_minimum_pairs = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-window-maximum-pairs") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_window_maximum_pairs);
        have_linkage_same_window_maximum_pairs = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-maximum-pending-nodes") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_maximum_pending_nodes);
        have_linkage_same_maximum_pending_nodes = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-preflush-target-pairs") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_preflush_target_pairs);
        have_linkage_same_preflush_pairs = true;
        return parsed;
      }
      if (option == "--linkage-same-layer-preflush-target-nodes") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_preflush_target_nodes);
        have_linkage_same_preflush_nodes = true;
        return parsed;
      }
      if (option == "--linkage-tiny-forced-pair-limit") {
        parsed = ParseInteger(value, &output.linkage_tiny_forced_pair_limit);
        have_linkage_tiny_limit = true;
        return parsed;
      }
      if (option == "--linkage-tiny-forced-cpu-enabled") {
        parsed = ParseBool(value, &output.linkage_tiny_forced_cpu_enabled);
        return parsed;
      }
      if (option == "--init-linkage-tiny-forced-cpu-enabled") {
        parsed = ParseBool(
            value, &output.initialization_linkage_tiny_forced_cpu_enabled);
        return parsed;
      }
      if (option == "--linkage-tiny-forced-node-limit") {
        parsed = ParseInteger(value, &output.linkage_tiny_forced_node_limit);
        return parsed;
      }
      if (option == "--linkage-single-gpu-min-candidates") {
        parsed = ParseInteger(
            value, &output.linkage_single_gpu_min_candidates);
        return parsed;
      }
      if (option == "--linkage-single-gpu-max-candidates") {
        parsed = ParseInteger(
            value, &output.linkage_single_gpu_max_candidates);
        return parsed;
      }
      if (option ==
                 "--linkage-same-layer-single-gpu-min-candidates") {
        parsed = ParseInteger(
            value, &output.linkage_same_layer_single_gpu_min_candidates);
        return parsed;
      }
      if (option == "--linkage-bad-window-rows") {
        parsed = ParseInteger(value, &output.linkage_bad_window_rows);
        have_linkage_bad_window_rows = true;
        return parsed;
      }
      if (option == "--linkage-bad-window-maximum-pairs") {
        parsed = ParseInteger(value, &output.linkage_bad_window_maximum_pairs);
        have_linkage_bad_window_max_pairs = true;
        return parsed;
      }
      if (option == "--final-spool-window-clusters") {
        parsed = ParseInteger(value, &output.final_spool_window_clusters);
        have_final_spool_clusters = true;
        return parsed;
      }
      if (option == "--final-spool-window-real-rows") {
        parsed = ParseInteger(value, &output.final_spool_window_real_rows);
        have_final_spool_rows = true;
        return parsed;
      }
      return std::nullopt;
    }();
    if (!parsed_option.has_value()) {
      return InvalidCommand("unknown option " + std::string(option));
    }
    if (!*parsed_option) {
      return InvalidCommand("invalid value for " + std::string(option));
    }
  }

  if (const Status status = ResolveRuntimeCudaExecutionPolicy(
          output.use_cuda, output.gpu_norm_enable); !status.ok()) {
    return status;
  }
  if (output.eval_only) {
    output.train_enabled = false;
    output.base_encode_enabled = false;
    output.base_linkage_enabled = false;
  }
  if (output.eval_base_nprobe == 0U || output.eval_base_repeat == 0U) {
    return InvalidCommand(
        "eval.base.nprobe and eval.base.repeat must be positive");
  }
  if (!output.eval_only && output.eval_linkage_enabled &&
      !output.base_linkage_enabled) {
    return InvalidCommand(
        "eval.linkage.enabled=true requires base.linkage.enabled=true "
        "during an ordinary build");
  }
  const bool complete = have_m && have_training_format && have_base_format &&
      have_dataset &&
      (!output.train_enabled || have_train) &&
      have_train_rows && (!output.base_encode_enabled || have_base) &&
      have_base_rows &&
      have_nlist && (!(output.train_enabled &&
                       output.train_initialization_enabled) ||
                     have_init_samples) &&
      have_root_initialization && have_train_seed && have_training_rounds &&
      have_base_seed && have_good_fraction && have_bad_cluster_weight &&
      have_training_topology && have_base_topology && have_virtual_roots &&
      have_work && have_output && have_archive_log &&
      have_threads && have_base_list_threads &&
      have_root_moment_partitions &&
      have_rotation_moment_partitions &&
      have_initialization_edge_moment_shards &&
      have_training_edge_moment_shards &&
      have_device && have_use_cuda &&
      have_training_block_rows && have_base_block_rows &&
      have_cids_per_bucket && have_basic_inflight_blocks &&
      have_basic_async_io_enabled && have_basic_async_io_depth &&
      have_basic_async_io_mebibytes && have_basic_async_write_enabled &&
      have_basic_async_write_depth && have_basic_async_write_mebibytes &&
      have_cuda_mode &&
      have_cuda_allow_tf32 &&
      have_root_score_buffer && have_root_normalized_cache &&
      have_linkage_xc0_chunk && have_linkage_workspace &&
      have_init_linkage_xc0_chunk && have_init_linkage_workspace &&
      have_coefficient_first_quantiles &&
      have_coefficient_rest_quantiles && have_norm8_iterations &&
      have_topology_inflight &&
      have_linkage_async_io_enabled && have_linkage_async_io_depth &&
      have_linkage_scheduler_workers && have_linkage_scheduler_queue &&
      have_linkage_cache_buffers && have_linkage_cache_mebibytes &&
      have_linkage_safety_reserve &&
      have_linkage_inner_rows &&
      have_linkage_max_pairs && have_linkage_init_max_pairs &&
      have_linkage_many_nodes_preflush_pairs &&
      have_linkage_same_block_rows &&
      have_linkage_same_minimum_pairs &&
      have_linkage_same_window_maximum_pairs &&
      have_linkage_same_maximum_pending_nodes &&
      have_linkage_same_preflush_pairs &&
      have_linkage_same_preflush_nodes &&
      have_linkage_tiny_limit &&
      have_linkage_bad_window_rows &&
      have_linkage_bad_window_max_pairs && have_final_spool_clusters &&
      have_final_spool_rows;
  if (!complete) {
    if (output.eval_only) {
      output.eval_only_sparse_config = true;
      return output;
    }
    return InvalidCommand("required options are incomplete");
  }
  // Full and replay topology alternate; a periodic archive needs an even
  // spacing so that every occurrence can stay on a full round.
  if ((output.train_checkpoint_every_r & 1U) != 0U) {
    if (output.train_checkpoint_every_r ==
        std::numeric_limits<std::uint32_t>::max()) {
      return InvalidCommand(
          "train.ckpt.every_R cannot be rounded to an even interval");
    }
    ++output.train_checkpoint_every_r;
  }
  const auto safe_component = [](const std::string& value) {
    return !value.empty() && value != "." && value != ".." &&
        value.find('/') == std::string::npos &&
        value.find('\\') == std::string::npos;
  };
  if (!safe_component(output.output_prefix)) {
    return InvalidField("io.pre_fix", output.output_prefix,
                        "non-empty path component");
  }
  if (!safe_component(output.dataset_name)) {
    return InvalidField("dataset.name", output.dataset_name,
                        "non-empty path component");
  }
#define CHECK_FIELD(condition, field, value, requirement) \
  if (condition) return InvalidField(field, value, requirement)
  CHECK_FIELD(output.m < 2U || output.m > stlqpp::kMaximumModelLayers,
              "model.m", output.m, "integer in [2,24]");
  CHECK_FIELD(output.dataset_name.empty(), "dataset.name",
              output.dataset_name, "non-empty dataset name");
  CHECK_FIELD(output.train_enabled && output.training_file.empty(),
              "dataset.train_path", output.training_file,
              "non-empty path when train.enabled=true");
  CHECK_FIELD(output.base_encode_enabled && output.base_file.empty(),
              "dataset.base_path", output.base_file,
              "non-empty path when base.encode.enabled=true");
  CHECK_FIELD(output.training_rows == 0U, "dataset.ntrain",
              output.training_rows, "positive row count");
  CHECK_FIELD(output.base_rows == 0U, "dataset.nbase",
              output.base_rows, "positive row count");
  CHECK_FIELD(output.nlist == 0U, "model.h0", output.nlist,
              "positive cluster count");
  CHECK_FIELD(output.root_initialization.large_layer_fixed_coarse_count < 0,
              "train.kmeans_large_k_fixed_k1",
              output.root_initialization.large_layer_fixed_coarse_count,
              "nonnegative count");
  CHECK_FIELD(output.root_initialization.large_layer_minimum_coarse_count <= 0,
              "train.kmeans_large_k_k1_min",
              output.root_initialization.large_layer_minimum_coarse_count,
              "positive count");
  CHECK_FIELD(output.root_initialization.large_layer_maximum_coarse_count <
                  output.root_initialization.large_layer_minimum_coarse_count,
              "train.kmeans_large_k_k1_max",
              output.root_initialization.large_layer_maximum_coarse_count,
              "at least train.kmeans_large_k_k1_min");
  CHECK_FIELD(output.train_enabled && output.train_initialization_enabled &&
                  output.initialization_samples == 0U,
              "train.init_samples", output.initialization_samples,
              "positive sample count when initialization is enabled");
  CHECK_FIELD(!std::isfinite(output.good_cluster_fraction) ||
                  output.good_cluster_fraction <= 0.0 ||
                  output.good_cluster_fraction > 1.0,
              "virtual.good_fraction", output.good_cluster_fraction,
              "finite fraction in (0,1]");
  CHECK_FIELD(!std::isfinite(output.bad_cluster_training_weight) ||
                  output.bad_cluster_training_weight <= 0.0,
              "virtual.alpha_bad", output.bad_cluster_training_weight,
              "finite positive weight");
  CHECK_FIELD(output.work_directory.empty(), "large.tmp_dir",
              output.work_directory, "non-empty path");
  CHECK_FIELD(output.output_directory.empty(), "large.output_dir",
              output.output_directory, "non-empty path");
  CHECK_FIELD(output.work_directory == output.output_directory,
              "large.tmp_dir", output.work_directory,
              "different from large.output_dir");
  CHECK_FIELD(output.train_enabled && !output.train_initialization_enabled &&
                  output.training_rounds != 0U && !output.eval_only &&
                  output.train_checkpoint_file.empty(),
              "io.train_file", output.train_checkpoint_file,
              "explicit trained model or checkpoint when appending training rounds");
  CHECK_FIELD(!output.resume_initialized_checkpoint.empty() &&
                  !output.resume_round_checkpoint.empty(),
              "io.train_file",
              output.resume_round_checkpoint,
              "not a round checkpoint when train.resume_initialized_checkpoint is set");
  CHECK_FIELD(output.cuda_device < 0, "runtime.cuda_device",
              output.cuda_device, "nonnegative device index");
  CHECK_FIELD(output.training_first_row >
                  std::numeric_limits<std::uint64_t>::max() -
                      output.training_rows,
              "dataset.train_first_row", output.training_first_row,
              "first row plus dataset.ntrain fits uint64");
  CHECK_FIELD(output.base_first_row >
                  std::numeric_limits<std::uint64_t>::max() - output.base_rows,
              "dataset.base_first_row", output.base_first_row,
              "first row plus dataset.nbase fits uint64");
  CHECK_FIELD(output.omp_places != "" && output.omp_places != "none" &&
                  output.omp_places != "cores" &&
                  output.omp_places != "threads" &&
                  output.omp_places != "sockets",
              "runtime.omp_places", output.omp_places,
              "empty, none, cores, threads, or sockets");
  CHECK_FIELD(output.omp_proc_bind != "" && output.omp_proc_bind != "none" &&
                  output.omp_proc_bind != "close" &&
                  output.omp_proc_bind != "spread" &&
                  output.omp_proc_bind != "primary",
              "runtime.omp_proc_bind", output.omp_proc_bind,
              "empty, none, close, spread, or primary");
  CHECK_FIELD(output.rotation_moment_partitions == 0U,
              "internal --rotation-moment-partitions",
              output.rotation_moment_partitions, "positive partition count");
  CHECK_FIELD(output.root_score_buffer_mebibytes == 0U,
              "internal --root-score-buffer-mib",
              output.root_score_buffer_mebibytes, "positive MiB budget");
  CHECK_FIELD(output.linkage_large_root_xc0_chunk_mebibytes == 0U,
              "runtime.cuda_linkage_large_root_xc0_chunk_mb",
              output.linkage_large_root_xc0_chunk_mebibytes,
              "positive MiB budget");
  CHECK_FIELD(output.linkage_large_root_workspace_mebibytes == 0U,
              "runtime.cuda_linkage_mem_budget_mb",
              output.linkage_large_root_workspace_mebibytes,
              "positive MiB budget");
  CHECK_FIELD(output.coefficient_first_magnitude_quantiles.empty(),
              "large.linkage_coeff_codec.p_first_candidates", "[]",
              "non-empty percentage list in (0,100]");
  CHECK_FIELD(output.coefficient_rest_magnitude_quantiles.empty(),
              "large.linkage_coeff_codec.p_rest_candidates", "[]",
              "non-empty percentage list in (0,100]");
  for (std::size_t i = 0U;
       i < output.coefficient_first_magnitude_quantiles.size(); ++i) {
    const double value = output.coefficient_first_magnitude_quantiles[i];
    CHECK_FIELD(!std::isfinite(value) || value <= 0.0 || value > 1.0,
                "large.linkage_coeff_codec.p_first_candidates",
                value * 100.0, "each percentage in (0,100]");
  }
  for (std::size_t i = 0U;
       i < output.coefficient_rest_magnitude_quantiles.size(); ++i) {
    const double value = output.coefficient_rest_magnitude_quantiles[i];
    CHECK_FIELD(!std::isfinite(value) || value <= 0.0 || value > 1.0,
                "large.linkage_coeff_codec.p_rest_candidates",
                value * 100.0, "each percentage in (0,100]");
  }
  CHECK_FIELD(output.norm8_lloyd_iterations == 0U,
              "eval.disk_norm2_lut_kmeans_niter",
              output.norm8_lloyd_iterations, "positive iteration count");
  CHECK_FIELD(output.training_topology.radial_layer_count == 0U,
              "train.linkage.num_layers",
              output.training_topology.radial_layer_count,
              "positive layer count");
  CHECK_FIELD(output.training_topology.root_fraction <= 0.0 ||
                  output.training_topology.root_fraction > 1.0,
              "train.linkage.root_percentile",
              output.training_topology.root_fraction,
              "fraction in (0,1]");
  CHECK_FIELD(output.training_topology.depth_k == 0U,
              "train.linkage.depth_k", output.training_topology.depth_k,
              "positive k");
  CHECK_FIELD(output.training_topology.knn_k == 0U,
              "train.linkage.knn_k", output.training_topology.knn_k,
              "positive k");
  CHECK_FIELD(output.training_topology.maximum_depth == 0U,
              "train.linkage.max_depth",
              output.training_topology.maximum_depth, "positive depth");
  for (const auto* topology : {&output.training_topology, &output.base_topology}) {
    if (topology->bad_cluster_knn_backend != "cpu_exact" &&
        topology->bad_cluster_knn_backend != "auto" &&
        topology->bad_cluster_knn_backend != "hnsw")
      return Status::InvalidArgument("bad_cluster_knn_backend must be cpu_exact, auto or hnsw");
    if (topology->bad_cluster_knn_backend == "auto" && topology->bad_cluster_cpu_exact_max_points == 0)
      return Status::InvalidArgument("auto KNN needs a positive bad_cluster_cpu_exact_max_points");
  }
  if (output.hnsw_m < 2U || output.hnsw_m > 10000U)
    return Status::InvalidArgument("hnsw.M must be in [2,10000]");
  CHECK_FIELD(output.init_linkage_icm_rounds < -1,
              "train.init_linkage_icm_round",
              output.init_linkage_icm_rounds, "-1 or nonnegative rounds");
  CHECK_FIELD(!IsSupportedRsdRank(output.training_topology.rsd_rank),
              "train.linkage.rsd_rank", output.training_topology.rsd_rank,
              "supported RSD rank");
  CHECK_FIELD(!IsSupportedRsdRank(output.training_basic_rsd_rank),
              "train.encode.rsd_rank", output.training_basic_rsd_rank,
              "supported RSD rank");
  CHECK_FIELD(output.base_topology.radial_layer_count == 0U,
              "base.linkage.num_layers",
              output.base_topology.radial_layer_count,
              "positive layer count");
  CHECK_FIELD(output.base_topology.root_fraction <= 0.0 ||
                  output.base_topology.root_fraction > 1.0,
              "base.linkage.root_percentile",
              output.base_topology.root_fraction,
              "fraction in (0,1]");
  CHECK_FIELD(output.base_topology.depth_k == 0U,
              "base.linkage.depth_k", output.base_topology.depth_k,
              "positive k");
  CHECK_FIELD(output.base_topology.knn_k == 0U,
              "base.linkage.knn_k", output.base_topology.knn_k,
              "positive k");
  CHECK_FIELD(output.base_topology.maximum_depth == 0U,
              "base.linkage.max_depth", output.base_topology.maximum_depth,
              "positive depth");
  CHECK_FIELD(!IsSupportedRsdRank(output.base_topology.rsd_rank),
              "base.linkage.rsd_rank", output.base_topology.rsd_rank,
              "supported RSD rank");
  CHECK_FIELD(!IsSupportedRsdRank(output.base_basic_rsd_rank),
              "base.encode.rsd_rank", output.base_basic_rsd_rank,
              "supported RSD rank");
  CHECK_FIELD(!std::isfinite(output.virtual_roots.ratio) ||
                  output.virtual_roots.ratio <= 0.0 ||
                  output.virtual_roots.ratio > 1.0,
              "virtual.virtual_ratio", output.virtual_roots.ratio,
              "finite fraction in (0,1]");
  CHECK_FIELD(output.virtual_roots.minimum == 0U,
              "virtual.min_virtual", output.virtual_roots.minimum,
              "positive count");
  CHECK_FIELD(output.virtual_roots.maximum < output.virtual_roots.minimum,
              "virtual.max_virtual", output.virtual_roots.maximum,
              "at least virtual.min_virtual");
  CHECK_FIELD(output.virtual_roots.neighbor_count == 0U,
              "virtual.umap_knn_k", output.virtual_roots.neighbor_count,
              "positive k");
  CHECK_FIELD(output.virtual_roots.local_connectivity == 0U,
              "virtual.local_connectivity",
              output.virtual_roots.local_connectivity, "positive count");
  CHECK_FIELD(!std::isfinite(output.virtual_roots.overlap_threshold) ||
                  output.virtual_roots.overlap_threshold < 0.0F ||
                  output.virtual_roots.overlap_threshold > 1.0F,
              "virtual.overlap_thr", output.virtual_roots.overlap_threshold,
              "finite fraction in [0,1]");
  CHECK_FIELD(output.virtual_roots.anchor_neighbor_k == 0U,
              "virtual.anchor_neighbor_k",
              output.virtual_roots.anchor_neighbor_k, "positive k");
  CHECK_FIELD(output.virtual_roots.use_fixed_per_cluster &&
                  output.virtual_roots.fixed_per_cluster == 0U,
              "virtual.fixed_virtual_per_cluster",
              output.virtual_roots.fixed_per_cluster,
              "positive count when fixed virtual roots are enabled");
  CHECK_FIELD(output.topology_inflight_clusters == 0U,
              "internal --topology-inflight-clusters",
              output.topology_inflight_clusters, "positive count");
  CHECK_FIELD(output.linkage_async_io_enabled &&
                  output.linkage_async_io_depth <= 0,
              "runtime.linkage_async_io_depth",
              output.linkage_async_io_depth,
              "positive depth when runtime.linkage_async_io=true");
  CHECK_FIELD(output.basic_async_io_enabled && output.basic_async_io_depth <= 0,
              "runtime.basic_async_io_depth", output.basic_async_io_depth,
              "positive depth when runtime.basic_async_io=true");
  CHECK_FIELD(output.basic_async_write_enabled &&
                  output.basic_async_write_depth <= 0,
              "runtime.basic_async_write_depth",
              output.basic_async_write_depth,
              "positive depth when runtime.basic_async_write=true");
  CHECK_FIELD(output.linkage_direct_pool_size == 0U,
              "runtime.cuda_pool_size", output.linkage_direct_pool_size,
              "positive pool size");
  CHECK_FIELD(output.linkage_scheduler_queue_capacity == 0U,
              "internal --linkage-scheduler-queue-capacity",
              output.linkage_scheduler_queue_capacity, "positive capacity");
  CHECK_FIELD(output.linkage_rfull_cache_buffers == 0U,
              "internal --linkage-rfull-cache-buffers",
              output.linkage_rfull_cache_buffers, "positive buffer count");
  CHECK_FIELD(output.linkage_rfull_cache_mebibytes == 0U,
              "internal --linkage-rfull-cache-mib",
              output.linkage_rfull_cache_mebibytes, "positive MiB budget");
  CHECK_FIELD(output.linkage_safety_reserve_mebibytes == 0U,
              "internal --linkage-safety-reserve-mib",
              output.linkage_safety_reserve_mebibytes, "positive MiB budget");
  CHECK_FIELD(output.linkage_max_batch_pairs == 0U ||
                  output.linkage_max_batch_pairs > 32768U,
              "runtime.cuda_linkage_many_nodes_max_pairs",
              output.linkage_max_batch_pairs, "integer in [1,32768]");
  CHECK_FIELD(output.linkage_chunk_max_pairs > 32768U,
              "runtime.cuda_linkage_chunk_max_pairs",
              output.linkage_chunk_max_pairs, "integer in [0,32768]");
  CHECK_FIELD(output.init_linkage_chunk_max_pairs > 32768U,
              "runtime.cuda_linkage_chunk_max_pairs_init_linkage",
              output.init_linkage_chunk_max_pairs,
              "integer in [0,32768]");
  CHECK_FIELD(output.linkage_init_max_batch_pairs > 32768,
              "runtime.cuda_linkage_many_nodes_max_pairs_init_linkage",
              output.linkage_init_max_batch_pairs,
              "integer at most 32768");
  CHECK_FIELD(output.linkage_same_layer_window_maximum_pairs > 32768,
              "runtime.cuda_linkage_same_layer_window_max_pairs",
              output.linkage_same_layer_window_maximum_pairs,
              "integer at most 32768");
  CHECK_FIELD(output.linkage_bad_window_maximum_pairs > 32768,
              "runtime.cuda_linkage_bad_frozen_window_max_pairs",
              output.linkage_bad_window_maximum_pairs,
              "integer at most 32768");
  CHECK_FIELD(output.final_spool_window_clusters == 0U,
              "internal --final-spool-window-clusters",
              output.final_spool_window_clusters, "positive cluster count");
  CHECK_FIELD(output.final_spool_window_real_rows == 0U,
              "internal --final-spool-window-real-rows",
              output.final_spool_window_real_rows, "positive row count");
  CHECK_FIELD(output.cleanup.keep_base_basic < -1 ||
                  output.cleanup.keep_base_basic > 1,
              "large.cleanup.keep_base_basic",
              output.cleanup.keep_base_basic, "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_base_basic_cluster_id < -1 ||
                  output.cleanup.keep_base_basic_cluster_id > 1,
              "large.cleanup.keep_base_basic_cluster_id",
              output.cleanup.keep_base_basic_cluster_id,
              "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_base_basic_ivf_lists < -1 ||
                  output.cleanup.keep_base_basic_ivf_lists > 1,
              "large.cleanup.keep_base_basic_ivf_lists",
              output.cleanup.keep_base_basic_ivf_lists,
              "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_base_list < -1 ||
                  output.cleanup.keep_base_list > 1,
              "large.cleanup.keep_base_list", output.cleanup.keep_base_list,
              "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_base_list_raw < -1 ||
                  output.cleanup.keep_base_list_raw > 1,
              "large.cleanup.keep_base_list_raw",
              output.cleanup.keep_base_list_raw, "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_linkage_list_f32 < -1 ||
                  output.cleanup.keep_linkage_list_f32 > 1,
              "large.cleanup.keep_linkage_list_f32",
              output.cleanup.keep_linkage_list_f32, "one of -1, 0, 1");
  CHECK_FIELD(output.cleanup.keep_linkage_parent_u32 < -1 ||
                  output.cleanup.keep_linkage_parent_u32 > 1,
              "large.cleanup.keep_linkage_parent_u32",
              output.cleanup.keep_linkage_parent_u32,
              "one of -1, 0, 1");
#undef CHECK_FIELD
  output.cuda_allow_tf32 = output.cuda_mode == CudaMode::kFast ||
      output.requested_cuda_allow_tf32;
  if (output.cpu_threads > 0) {
    ResolveBuildRuntimeThreadCounts(
        &output, static_cast<std::uint32_t>(output.cpu_threads));
  }
  return output;
}

std::filesystem::path TrainingModelPublicationBase(
    const BuildCommand& command, ArtifactHash64 training_source,
    ArtifactHash64 training_recipe) {
  return command.output_directory / command.dataset_name / "models" /
      (command.output_prefix + "_m" + std::to_string(command.m) +
       "_train_h" + ShortTrainedModelNamespace(
           training_source.value(), training_recipe.value()) +
       ".stlqmodel");
}

Status ValidateBuildPathIsolation(
    const BuildCommand& command,
    const std::filesystem::path& final_index) {
  if (command.eval_only) return Status::Ok();
  if (command.work_directory.empty() || command.output_directory.empty()) {
    return Status::InvalidArgument(
        "build work and output paths must be non-empty");
  }
  if (const Status status = ValidateDisjointFilesystemPaths(
          command.work_directory, "build work directory",
          command.output_directory, "build output directory");
      !status.ok()) {
    return status;
  }
  if (!command.base_linkage_enabled || final_index.empty()) {
    return Status::Ok();
  }
  const auto reject_live_input = [&](const std::filesystem::path& input,
                                     std::string_view role) -> Status {
    if (input.empty()) return Status::Ok();
    return ValidateDisjointFilesystemPaths(
        final_index, "final index replacement directory", input, role);
  };
  const std::filesystem::path& training_source =
      command.effective_training_file.empty()
      ? command.training_file : command.effective_training_file;
  const std::filesystem::path& base_source =
      command.effective_base_file.empty()
      ? command.base_file : command.effective_base_file;
  for (const auto& [path, role] : {
           std::pair<std::filesystem::path, std::string_view>{
               training_source, "training source"},
           {base_source, "base source"},
           {command.train_checkpoint_file, "training checkpoint"},
           {command.resume_initialized_checkpoint,
            "initialized checkpoint"},
           {command.resume_round_checkpoint, "round checkpoint"},
           {command.reused_base_list_directory, "reused base list"},
           {command.control_evidence, "control evidence"}}) {
    if (const Status status = reject_live_input(path, role); !status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

namespace {

struct ResolvedAssignment final {
  std::string option;
  std::string value;
  std::string source;
  // Empty only for implementation-only fixed/default inputs. Public
  // provenance is keyed by the exact DEV field, never by option.
  std::string public_key{};
};

struct ResolvedConfigKey final {
  std::string public_key;
  std::string option;
};

std::string Trim(std::string_view input) {
  return TrimConfigText(input);
}

bool IsApplicationFieldOption(std::string_view option) {
  return option.rfind("--", 0U) == 0U &&
      (IsProductConfigField(
           ProductConfigScope::kBuild, option.substr(2U)) ||
       IsProductExtensionConfigField(
           ProductConfigScope::kBuild, option.substr(2U)) ||
       IsProductConfigField(
           ProductConfigScope::kEval, option.substr(2U)));
}

StatusOr<ResolvedConfigKey> ResolveConfigKey(std::string_view raw_key) {
  const std::string key = Trim(raw_key);
  const std::string_view unprefixed = key.rfind("--", 0U) == 0U
      ? std::string_view(key).substr(2U) : std::string_view(key);
  for (const ProductConfigField& field :
       ProductConfigFields(ProductConfigScope::kBuild)) {
    if (field.public_key == unprefixed) {
      return ResolvedConfigKey{
          .public_key = std::string(field.public_key),
          .option = field.private_option.empty()
              ? std::string{}
              : "--" + std::string(field.private_option),
      };
    }
  }
  for (const ProductConfigField& field :
       ProductExtensionConfigFields(ProductConfigScope::kBuild)) {
    if (field.public_key == unprefixed) {
      return ResolvedConfigKey{
          .public_key = std::string(field.public_key),
          .option = "--" + std::string(field.private_option),
      };
    }
  }
  // A complete DEV cfg contains both build and eval fields. Keep eval-only
  // provenance for reproducible run snapshots while leaving typed execution
  // to the eval parser.
  if (IsProductConfigField(ProductConfigScope::kEval, unprefixed)) {
    return ResolvedConfigKey{
        .public_key = std::string(unprefixed),
        .option = {},
    };
  }
  return InvalidCommand("unsupported config key " + key);
}

void AssignResolved(std::vector<ResolvedAssignment>* assignments,
                    ResolvedAssignment assignment) {
  const auto found = std::find_if(
      assignments->begin(), assignments->end(),
      [&](const ResolvedAssignment& old) {
        return assignment.public_key.empty()
            ? old.public_key.empty() && old.option == assignment.option
            : old.public_key == assignment.public_key;
      });
  if (found == assignments->end()) {
    assignments->push_back(std::move(assignment));
  } else {
    *found = std::move(assignment);
  }
}

Status ApplyConfigLine(std::string_view raw_line,
                       std::string source,
                       std::vector<ResolvedAssignment>* assignments) {
  auto stripped = stlqpp::app::StripConfigComment(raw_line);
  if (!stripped.ok()) {
    return InvalidCommand(stripped.status().message() + " in " + source);
  }
  const std::string line = std::move(stripped).value();
  if (line.empty()) {
    return Status::Ok();
  }
  const std::size_t equals = line.find('=');
  if (equals == std::string::npos) {
    return InvalidCommand("invalid config assignment in " + source);
  }
  const std::string key = Trim(std::string_view(line).substr(0U, equals));
  auto decoded = DecodeConfigValue(
      std::string_view(line).substr(equals + 1U));
  if (!decoded.ok()) {
    return InvalidCommand(decoded.status().message() + " in " + source);
  }
  std::string value = std::move(decoded).value();
  if (key.empty()) {
    return InvalidCommand("empty config key in " + source);
  }
  // Compatibility for frozen configs; never emit or execute this old switch.
  if (key == "runtime.omp_affinity_override_env") return Status::Ok();
  auto resolved = ResolveConfigKey(key);
  if (!resolved.ok()) {
    return resolved.status();
  }
  ResolvedConfigKey field = std::move(resolved).value();
  AssignResolved(assignments, {
      .option = std::move(field.option),
      .value = std::move(value),
      .source = std::move(source),
      .public_key = std::move(field.public_key),
  });
  return Status::Ok();
}

Status LoadConfigAssignments(
    const std::filesystem::path& path,
    std::string_view bytes,
    std::vector<ResolvedAssignment>* assignments) {
  std::istringstream input{std::string(bytes)};
  std::string line;
  std::uint64_t line_number = 0U;
  while (std::getline(input, line)) {
    ++line_number;
    const Status status = ApplyConfigLine(
        line, path.string() + ":" + std::to_string(line_number),
        assignments);
    if (!status.ok()) {
      return status;
    }
  }
  if (!input.eof()) {
    return InvalidCommand("failed while reading config file " + path.string());
  }
  return Status::Ok();
}

StatusOr<FrozenBuildConfigInput> ReadFrozenConfig(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return InvalidCommand("cannot open config file " + path.string());
  }
  std::string bytes{
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()};
  if (input.bad()) {
    return InvalidCommand("failed while reading config file " + path.string());
  }
  return FrozenBuildConfigInput{.path = path, .bytes = std::move(bytes)};
}

std::uint32_t DecodeLittleEndianU32(const unsigned char* bytes) noexcept {
  return static_cast<std::uint32_t>(bytes[0]) |
      (static_cast<std::uint32_t>(bytes[1]) << 8U) |
      (static_cast<std::uint32_t>(bytes[2]) << 16U) |
      (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

struct ObservedBuildShape final {
  std::uint32_t dimension = 0U;
  std::uint32_t m = 0U;
  std::uint32_t nlist = 0U;
  std::string source;
};

StatusOr<ObservedBuildShape> ReadVectorShape(
    const std::filesystem::path& path, SourceFormat format,
    std::string role) {
  if (format == SourceFormat::kNpyFloat16) {
    auto file = ReadOnlyMappedFile::Open(path);
    if (!file.ok()) return file.status();
    auto layout = InspectNpyArray(file.value());
    if (!layout.ok()) return layout.status();
    if (layout.value().scalar != NpyScalar::kFloat16) {
      return InvalidCommand(role + " npy vectors must be float16");
    }
    return ObservedBuildShape{
        .dimension = layout.value().columns,
        .source = std::move(role) + " npy header",
    };
  }
  std::ifstream input(path, std::ios::binary);
  unsigned char bytes[8]{};
  const std::size_t header_bytes =
      format == SourceFormat::kFbinF32 ? 8U : 4U;
  if (!input ||
      !input.read(reinterpret_cast<char*>(bytes),
                  static_cast<std::streamsize>(header_bytes))) {
    return InvalidCommand(
        "cannot read " + role + " vector header " + path.string());
  }
  const std::uint32_t dimension = DecodeLittleEndianU32(
      bytes + (format == SourceFormat::kFbinF32 ? 4U : 0U));
  if (dimension == 0U) {
    return InvalidCommand(role + " vector dimension is zero");
  }
  return ObservedBuildShape{
      .dimension = dimension,
      .source = std::move(role) + " vector header",
  };
}

StatusOr<SourceFormat> ResolveSourceFormat(
    SourceFormat requested, const std::filesystem::path& path,
    std::string_view role) {
  if (requested != SourceFormat::kAuto) return requested;
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (extension == ".bvecs" || extension == ".siftbin") {
    return SourceFormat::kBvecsU8;
  }
  if (extension == ".fvecs") return SourceFormat::kFvecsF32;
  if (extension == ".fbin") return SourceFormat::kFbinF32;
  if (extension == ".npy") return SourceFormat::kNpyFloat16;
  return InvalidCommand(
      "cannot resolve auto " + std::string(role) +
      " vector format from path " + path.string());
}

StatusOr<ObservedBuildShape> ReadCheckpointShape(
    const std::filesystem::path& directory, bool initialized) {
  if (directory.extension() == ".stlqmodel") {
    auto artifact = TrainedModelFile::Open(directory);
    if (!artifact.ok()) return artifact.status();
    const auto& shape = artifact.value().header().shape;
    return ObservedBuildShape{
        .dimension = shape.dimension,
        .m = shape.m,
        .nlist = shape.nlist,
        .source = directory.string(),
    };
  }
  std::filesystem::path path = directory /
      (initialized ? "initialized.stlqpp" : "checkpoint.stlqpp");
  bool model_only = false;
  if (!initialized) {
    std::error_code error;
    const bool has_round_manifest = std::filesystem::exists(path, error);
    if (error) {
      return InvalidCommand("cannot inspect checkpoint header " +
                            path.string());
    }
    if (!has_round_manifest) {
      path = directory / "train_manifest.stlqpp";
      model_only = std::filesystem::exists(path, error);
      if (error) {
        return InvalidCommand("cannot inspect trained model header " +
                              path.string());
      }
      if (!model_only) {
        path = directory / "trained_model.stlqpp";
        model_only = std::filesystem::exists(path, error);
        if (error) {
          return InvalidCommand("cannot inspect trained model header " +
                                path.string());
        }
      }
      if (!model_only) {
        initialized = true;
        path = directory / "initialized.stlqpp";
      }
    }
  }
  std::ifstream input(path, std::ios::binary);
  unsigned char bytes[40]{};
  if (!input || !input.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) {
    return InvalidCommand("cannot read checkpoint header " + path.string());
  }
  const char* expected = initialized ? "STLQINI1" :
      (model_only ? "STLQMOD1" : "STLQTRN4");
  if (!std::equal(bytes, bytes + 8U,
                  reinterpret_cast<const unsigned char*>(expected))) {
    return InvalidCommand("checkpoint header magic is invalid: " +
                          path.string());
  }
  return ObservedBuildShape{
      .dimension = DecodeLittleEndianU32(bytes + (initialized ? 16U : 24U)),
      .m = DecodeLittleEndianU32(bytes + (initialized ? 20U : 28U)),
      .nlist = DecodeLittleEndianU32(bytes + (initialized ? 24U : 32U)),
      .source = path.string(),
  };
}

}  // namespace

Status ResolveBuildInputMetadata(BuildCommand* command) {
  if (command == nullptr) {
    return InvalidCommand("metadata destination is null");
  }
  if (command->eval_only) return Status::Ok();
  if ((!command->train_enabled || !command->train_initialization_enabled) &&
      (command->train_checkpoint_file.filename() == "checkpoint.stlqpp" ||
       command->train_checkpoint_file.filename() == "train_manifest.stlqpp" ||
       command->train_checkpoint_file.filename() == "trained_model.stlqpp" ||
       command->train_checkpoint_file.filename() == "model.bin")) {
    command->train_checkpoint_file =
        command->train_checkpoint_file.parent_path();
  }
  const auto resolve_dataset_path = [&](const std::filesystem::path& path) {
    return path.empty() || path.is_absolute()
        ? path : command->dataset_data_root / path;
  };
  command->effective_training_file =
      resolve_dataset_path(command->training_file);
  command->effective_base_file = resolve_dataset_path(command->base_file);
  command->resume_round_checkpoint.clear();
  if (!command->train_enabled || !command->train_initialization_enabled) {
    command->resume_round_checkpoint = command->train_checkpoint_file;
  }
  const bool training_source_needed =
      !command->effective_training_file.empty() &&
      ((command->train_enabled && command->train_initialization_enabled) ||
       (command->train_enabled && command->training_rounds != 0U));
  const bool base_source_needed =
      !command->effective_base_file.empty() && command->base_encode_enabled;
  auto training_format = ResolveSourceFormat(
      command->training_format, command->effective_training_file,
      "training");
  if (!training_format.ok()) return training_format.status();
  command->effective_training_format = training_format.value();
  auto base_format = ResolveSourceFormat(
      command->base_format, command->effective_base_file, "base");
  if (!base_format.ok()) return base_format.status();
  command->effective_base_format = base_format.value();
  std::vector<ObservedBuildShape> observed;
  if (training_source_needed) {
    auto shape = ReadVectorShape(
        command->effective_training_file,
        command->effective_training_format,
        "training");
    if (!shape.ok()) return shape.status();
    observed.push_back(std::move(shape).value());
  }
  if (base_source_needed) {
    auto shape = ReadVectorShape(
        command->effective_base_file, command->effective_base_format,
        "base");
    if (!shape.ok()) return shape.status();
    observed.push_back(std::move(shape).value());
  }
  if (!command->resume_round_checkpoint.empty()) {
    auto shape = ReadCheckpointShape(command->resume_round_checkpoint, false);
    if (!shape.ok()) return shape.status();
    observed.push_back(std::move(shape).value());
  } else if (!command->resume_initialized_checkpoint.empty()) {
    auto shape = ReadCheckpointShape(
        command->resume_initialized_checkpoint, true);
    if (!shape.ok()) return shape.status();
    observed.push_back(std::move(shape).value());
  }
  if (observed.empty()) {
    return InvalidCommand(
        "cannot derive model shape without an active vector source or "
        "training checkpoint");
  }
  const std::uint32_t dimension = observed.front().dimension;
  for (const ObservedBuildShape& shape : observed) {
    if (shape.dimension != dimension ||
        (shape.m != 0U && shape.m != command->m) ||
        (shape.nlist != 0U && shape.nlist != command->nlist)) {
      return InvalidCommand(
          "source/checkpoint shape disagreement at " + shape.source);
    }
  }
  command->dimension = dimension;
  return Status::Ok();
}

std::uint32_t EffectiveTrainingBlockRows(
    const BuildCommand& command) noexcept {
  return static_cast<std::uint32_t>(
      std::max(1, command.training_block_rows));
}

std::uint32_t EffectiveBaseBlockRows(
    const BuildCommand& command) noexcept {
  return static_cast<std::uint32_t>(std::max(1, command.base_block_rows));
}

std::uint32_t EffectiveCidsPerBucket(
    const BuildCommand& command) noexcept {
  return static_cast<std::uint32_t>(std::max(1, command.cids_per_bucket));
}

std::size_t EffectiveBucketFlushBytes(
    const BuildCommand& command) noexcept {
  constexpr std::size_t kMebibyte = 1024U * 1024U;
  const std::size_t mebibytes = static_cast<std::size_t>(
      std::max<std::int32_t>(1, command.bucket_flush_mebibytes));
  return mebibytes > std::numeric_limits<std::size_t>::max() / kMebibyte
      ? std::numeric_limits<std::size_t>::max()
      : mebibytes * kMebibyte;
}

std::uint32_t EffectiveLinkageSchedulerWorkers(
    const BuildCommand& command,
    std::uint32_t phase_direct_pool_size) noexcept {
  return std::max(
      1U, std::min(
              phase_direct_pool_size,
              static_cast<std::uint32_t>(
                  std::max(1, command.linkage_scheduler_workers))));
}

std::uint32_t EffectiveLinkageAsyncPinnedMebibytes(
    const BuildCommand& command) noexcept {
  return static_cast<std::uint32_t>(
      std::max(0, command.linkage_eval_async_pinned_mebibytes));
}

const char* CleanupPresetName(CleanupPreset preset) noexcept {
  switch (preset) {
    case CleanupPreset::kNone: return "none";
    case CleanupPreset::kDevAll: return "dev_all";
    case CleanupPreset::kEvalBoth: return "eval_both";
    case CleanupPreset::kEvalFloatMin: return "eval_float_min";
    case CleanupPreset::kEvalInt8Min: return "eval_int8_min";
    case CleanupPreset::kCustom: return "custom";
  }
  return "none";
}

ResolvedCleanupPolicy ResolveCleanupPolicy(
    const CleanupCommand& command) noexcept {
  ResolvedCleanupPolicy output;
  if (!command.enabled) return output;
  switch (command.preset) {
    case CleanupPreset::kNone:
    case CleanupPreset::kCustom:
      break;
    case CleanupPreset::kDevAll:
      output.keep_base_basic = false;
      output.keep_base_basic_cluster_id = false;
      break;
    case CleanupPreset::kEvalBoth:
      output.keep_base_basic = false;
      output.keep_base_basic_cluster_id = false;
      output.keep_base_basic_ivf_lists = false;
      output.keep_base_list = false;
      output.keep_base_list_raw = false;
      break;
    case CleanupPreset::kEvalFloatMin:
      output.keep_base_basic = false;
      output.keep_base_basic_cluster_id = false;
      output.keep_base_basic_ivf_lists = false;
      output.keep_base_list = false;
      output.keep_base_list_raw = false;
      output.keep_linkage_parent_u32 = false;
      break;
    case CleanupPreset::kEvalInt8Min:
      output.keep_base_basic = false;
      output.keep_base_basic_cluster_id = false;
      output.keep_base_basic_ivf_lists = false;
      output.keep_base_list = false;
      output.keep_base_list_raw = false;
      output.keep_linkage_list_f32 = false;
      output.keep_linkage_parent_u32 = false;
      break;
  }
  const auto override_value = [](std::int32_t value, bool* destination) {
    if (value >= 0) *destination = value != 0;
  };
  override_value(command.keep_base_basic, &output.keep_base_basic);
  override_value(
      command.keep_base_basic_cluster_id,
      &output.keep_base_basic_cluster_id);
  override_value(
      command.keep_base_basic_ivf_lists,
      &output.keep_base_basic_ivf_lists);
  override_value(command.keep_base_list, &output.keep_base_list);
  override_value(command.keep_base_list_raw, &output.keep_base_list_raw);
  override_value(
      command.keep_linkage_list_f32, &output.keep_linkage_list_f32);
  override_value(
      command.keep_linkage_parent_u32,
      &output.keep_linkage_parent_u32);
  return output;
}

std::uint32_t ResolveBasicInflightBlockCount(
    const BuildCommand& command, std::uint32_t, std::uint32_t) noexcept {
  return std::max(1, command.basic_inflight_blocks);
}

std::uint32_t ResolveBasicPrefetchBlockCount(
    const BuildCommand& command, std::uint32_t source_scalar_bytes,
    std::uint32_t block_rows) noexcept {
  if (!command.basic_async_io_enabled || command.basic_async_io_depth <= 0) {
    return 1U;
  }
  std::uint32_t depth = static_cast<std::uint32_t>(
      command.basic_async_io_depth);
  if (command.basic_async_io_mebibytes <= 0 ||
      source_scalar_bytes == 0U || block_rows == 0U ||
      command.dimension == 0U) {
    return depth;
  }
  const std::uint64_t bytes_per_row =
      static_cast<std::uint64_t>(command.dimension) * source_scalar_bytes;
  if (bytes_per_row == 0U ||
      bytes_per_row >
          std::numeric_limits<std::uint64_t>::max() / block_rows) {
    return 1U;
  }
  const std::uint64_t bytes_per_block = bytes_per_row * block_rows;
  const std::uint64_t budget_bytes = static_cast<std::uint64_t>(
      command.basic_async_io_mebibytes) * 1024U * 1024U;
  const std::uint64_t budget_depth = std::max<std::uint64_t>(
      1U, budget_bytes / bytes_per_block);
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      depth, std::min<std::uint64_t>(
                 budget_depth,
                 std::numeric_limits<std::uint32_t>::max())));
}

std::uint32_t ResolveBasicPendingWriteCount(
    const BuildCommand& command, std::uint32_t source_scalar_bytes,
    std::uint32_t block_rows) noexcept {
  if (!command.basic_async_write_enabled ||
      command.basic_async_write_depth <= 0) {
    return 1U;
  }
  std::uint32_t depth = static_cast<std::uint32_t>(
      command.basic_async_write_depth);
  if (command.basic_async_write_mebibytes <= 0 ||
      source_scalar_bytes == 0U || block_rows == 0U ||
      command.dimension == 0U || command.m == 0U) {
    return depth;
  }
  const std::uint64_t bytes_per_row =
      sizeof(std::uint32_t) +
      static_cast<std::uint64_t>(command.m - 1U) * sizeof(std::uint8_t) +
      static_cast<std::uint64_t>(command.m) * sizeof(float) +
      static_cast<std::uint64_t>(command.dimension) * source_scalar_bytes;
  if (bytes_per_row == 0U ||
      bytes_per_row >
          std::numeric_limits<std::uint64_t>::max() / block_rows) {
    return 1U;
  }
  const std::uint64_t bytes_per_block = bytes_per_row * block_rows;
  const std::uint64_t budget_bytes = static_cast<std::uint64_t>(
      command.basic_async_write_mebibytes) * 1024U * 1024U;
  const std::uint64_t budget_depth = std::max<std::uint64_t>(
      1U, budget_bytes / bytes_per_block);
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      depth, std::min<std::uint64_t>(
                 budget_depth,
                 std::numeric_limits<std::uint32_t>::max())));
}

std::uint32_t ResolveLinkagePrefetchClusterCount(
    const BuildCommand& command) noexcept {
  return command.linkage_async_io_enabled &&
          command.linkage_async_io_depth > 0
      ? static_cast<std::uint32_t>(command.linkage_async_io_depth)
      : 1U;
}

void ResolveBuildRuntimeThreadCounts(
    BuildCommand* command, std::uint32_t openmp_default_threads) noexcept {
  if (command == nullptr) return;
  const std::uint32_t runtime_threads = std::max(1U, openmp_default_threads);
  command->effective_cpu_threads = command->cpu_threads > 0
      ? static_cast<std::uint32_t>(command->cpu_threads) : runtime_threads;
  command->effective_base_list_threads =
      static_cast<std::uint32_t>(std::max(1, command->base_list_threads));
  std::uint32_t root_moment_partitions = command->root_moment_partitions;
  if (root_moment_partitions == 0U) {
    // The 30M sweep places p16 on the best plateau. Scale one power-of-two
    // partition per roughly 2M training rows, then cap the private-state and
    // worker count. This retains p1 for the small 100K/1M recipes and selects
    // p16/p32 for the registered 30M/50M large-data recipes.
    root_moment_partitions = SelectWorkloadThreadCount({
        .work_units = command->training_rows,
        .units_per_worker = 2'000'000U,
        .available_workers = command->effective_cpu_threads,
        .concurrent_jobs = 1U,
        .maximum_workers = 32U,
    });
  }
  command->effective_root_moment_partitions = std::min(
      {root_moment_partitions, command->effective_cpu_threads,
       command->nlist});
  command->effective_rotation_moment_partitions = std::min(
      command->rotation_moment_partitions, command->effective_cpu_threads);
  command->effective_training_edge_moment_shards =
      command->training_edge_moment_shards > 0
      ? std::min(static_cast<std::uint32_t>(
                     command->training_edge_moment_shards),
                 command->effective_cpu_threads)
      : std::min(command->effective_cpu_threads, 16U);
  command->effective_initialization_edge_moment_shards =
      command->initialization_edge_moment_shards > 0
      ? std::min(static_cast<std::uint32_t>(
                     command->initialization_edge_moment_shards),
                 command->effective_cpu_threads)
      : command->effective_training_edge_moment_shards;
}

BuildResourcePlan ResolveBuildResourcePlan(
    const BuildCommand& command,
    std::optional<bool> scheduler_enabled_override) noexcept {
  const bool scheduler_enabled = scheduler_enabled_override.value_or(
      command.linkage_scheduler_enabled);
  const bool ordinary_linkage_needed =
      (command.train_enabled && command.training_rounds != 0U) ||
      command.base_linkage_enabled;
  const bool initialization_needed =
      command.train_enabled && command.train_initialization_enabled;
  const bool basic_needed = initialization_needed ||
      (command.train_enabled && command.training_rounds != 0U) ||
      command.base_encode_enabled;
  if (!command.use_cuda) {
    return {
        .basic_cuda_instances = 0U,
        .basic_cpu_instances = basic_needed ? 1U : 0U,
        .initialization_scheduler_backends = 0U,
        .initialization_direct_backends = 0U,
        .initialization_cpu_direct_backends = initialization_needed
            ? command.effective_cpu_threads : 0U,
        .ordinary_scheduler_backends = 0U,
        .ordinary_direct_backends = 0U,
        .ordinary_cpu_direct_backends = ordinary_linkage_needed
            ? command.effective_cpu_threads : 0U,
        .replay_cuda_instances = 0U,
        .replay_cpu_instances =
            command.train_enabled && command.training_rounds != 0U
            ? 1U : 0U,
    };
  }
  const EffectiveBuildExecutionConfig effective =
      ResolveBuildExecutionConfig(command);
  return {
      .basic_cuda_instances = basic_needed ? 1U : 0U,
      .basic_cpu_instances = 0U,
      .initialization_scheduler_backends =
          initialization_needed && scheduler_enabled
          ? EffectiveLinkageSchedulerWorkers(
                command, effective.initialization_direct_pool_size) : 0U,
      .initialization_direct_backends =
          initialization_needed &&
                  (!scheduler_enabled || command.init_linkage_mode == "fast_init")
          ? effective.initialization_direct_pool_size : 0U,
      .ordinary_scheduler_backends =
          ordinary_linkage_needed && scheduler_enabled
          ? EffectiveLinkageSchedulerWorkers(
                command, command.linkage_direct_pool_size)
          : 0U,
      .ordinary_direct_backends = ordinary_linkage_needed
          ? command.linkage_direct_pool_size : 0U,
      .replay_cuda_instances =
          command.train_enabled && command.training_rounds != 0U ? 1U : 0U,
      .replay_cpu_instances = 0U,
  };
}

EffectiveBuildExecutionConfig ResolveBuildExecutionConfig(
    const BuildCommand& command) noexcept {
  return {
      .initialization_direct_pool_size =
          command.initialization_linkage_direct_pool_size <= 0
          ? command.linkage_direct_pool_size
          : static_cast<std::uint32_t>(
                command.initialization_linkage_direct_pool_size),
      .initialization_maximum_batch_pairs =
          command.linkage_init_max_batch_pairs <= 0
          ? command.linkage_max_batch_pairs
          : static_cast<std::uint32_t>(
                command.linkage_init_max_batch_pairs),
      .initialization_async_pinned_mebibytes =
          static_cast<std::uint32_t>(std::max(
              0, command.initialization_linkage_eval_async_pinned_mebibytes <=
                         0
                     ? command.linkage_eval_async_pinned_mebibytes
                     : command
                           .initialization_linkage_eval_async_pinned_mebibytes)),
      .initialization_large_root_xc0_chunk_mebibytes =
          command.init_linkage_large_root_xc0_chunk_mebibytes <= 0
          ? command.linkage_large_root_xc0_chunk_mebibytes
          : static_cast<std::uint32_t>(
                command.init_linkage_large_root_xc0_chunk_mebibytes),
      .initialization_large_root_workspace_mebibytes =
          command.init_linkage_large_root_workspace_mebibytes <= 0
          ? command.linkage_large_root_workspace_mebibytes
          : static_cast<std::uint32_t>(
                command.init_linkage_large_root_workspace_mebibytes),
  };
}

void WriteCanonicalBuildConfig(
    const BuildCommand& command, std::ostream& output) {
  const auto write_percent_candidates =
      [&](const std::vector<double>& candidates) {
        output << '[';
        for (std::size_t index = 0U; index < candidates.size(); ++index) {
          if (index != 0U) output << ',';
          output << candidates[index] * 100.0;
        }
        output << ']';
      };
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "train.linkage.rsd_rank\t" << command.training_topology.rsd_rank
      << '\n'
      << "base.linkage.rsd_rank\t" << command.base_topology.rsd_rank << '\n'
      << "train.encode.ils_iters\t" << command.train_basic_ils_iterations
      << "\ntrain.encode.icm_iters\t" << command.train_basic_icm_iterations
      << "\ntrain.encode.perturb_k\t" << command.train_basic_perturbations
      << "\ntrain.encode.rsd_rank\t" << command.training_basic_rsd_rank
      << "\nleast_squares.cost_mode\t"
      << LeastSquaresCostModeName(command.least_squares_cost_mode)
      << '\n'
      << "train.kmeans_init\t"
      << command.root_initialization.seed_method << '\n'
      << "train.linkage.icm_round\t"
      << command.training_topology.icm_rounds << '\n'
      << "base.encode.ils_iters\t" << command.base_basic_ils_iterations
      << "\nbase.encode.icm_iters\t" << command.base_basic_icm_iterations
      << "\nbase.encode.perturb_k\t" << command.base_basic_perturbations
      << "\nbase.encode.rsd_rank\t" << command.base_basic_rsd_rank
      << '\n'
      << "base.linkage.icm_round\t"
      << command.base_topology.icm_rounds << '\n'
      << "virtual.enabled\t"
      << static_cast<int>(command.virtual_roots.enabled) << '\n'
      << "large.linkage_coeff_codec.bits_per_layer\t[";
  for (std::size_t layer = 0U;
       layer < command.coefficient_bits_per_layer.size(); ++layer) {
    if (layer != 0U) output << ',';
    output << command.coefficient_bits_per_layer[layer];
  }
  output << "]\nlarge.linkage_coeff_codec.bits_per_layer_root\t[";
  for (std::size_t layer = 0U;
       layer < command.coefficient_bits_per_layer_root.size(); ++layer) {
    if (layer != 0U) output << ',';
    output << command.coefficient_bits_per_layer_root[layer];
  }
  output << "]\nlarge.linkage_coeff_codec.bits_per_layer_linkage\t[";
  for (std::size_t layer = 0U;
       layer < command.coefficient_bits_per_layer_linkage.size(); ++layer) {
    if (layer != 0U) output << ',';
    output << command.coefficient_bits_per_layer_linkage[layer];
  }
  output << "]\n"
      << "large.linkage_coeff_codec.use_weighted_quantile\t"
      << static_cast<int>(command.coefficient_weighted_quantile) << '\n'
      << "large.linkage_coeff_codec.allow_clip\t"
      << static_cast<int>(command.coefficient_allow_clipping) << '\n'
      << "large.linkage_coeff_codec.fit_scale\t"
      << static_cast<int>(command.coefficient_fit_scale) << '\n'
      << "large.linkage_coeff_codec.q_refine_sweeps\t"
      << command.coefficient_refinement_sweeps << '\n'
      << "large.linkage_coeff_codec.q_refine_max_layer\t"
      << command.coefficient_refinement_max_layer << '\n'
      << "large.linkage_coeff_codec.q_refine_step_limit\t"
      << command.coefficient_refinement_step_limit << '\n'
      << "train.enabled\t" << static_cast<int>(command.train_enabled) << '\n'
      << "train.init_enabled\t"
      << static_cast<int>(command.train_initialization_enabled) << '\n'
      << "base.encode.enabled\t"
      << static_cast<int>(command.base_encode_enabled) << '\n'
      << "base.linkage.enabled\t"
      << static_cast<int>(command.base_linkage_enabled) << '\n'
      << "advanced.eval_only\t" << static_cast<int>(command.eval_only) << '\n'
      << "train.ckpt.enabled\t"
      << static_cast<int>(command.train_checkpoint_enabled) << '\n'
      << "train.ckpt.every_R\t" << command.train_checkpoint_every_r << '\n'
      << "io.train_file\t"
      << command.train_checkpoint_file.generic_string() << '\n'
      << "io.pre_fix\t" << command.output_prefix << '\n'
      << "io.base_file\t"
      << command.reused_base_list_directory.generic_string() << '\n'
      << "large.base_basic_checkpoint\t"
      << static_cast<int>(command.base_basic_checkpoint_enabled) << '\n'
      << "large.linkage_checkpoint\t"
      << static_cast<int>(command.linkage_checkpoint_enabled) << '\n'
      << "model.m\t" << command.m << '\n'
      << "dataset.name\t" << command.dataset_name << '\n'
      << "dataset.data_root\t"
      << command.dataset_data_root.generic_string() << '\n'
      << "dataset.train_path\t" << command.training_file.generic_string()
      << '\n'
      << "dataset.ntrain\t" << command.training_rows << '\n'
      << "dataset.base_path\t" << command.base_file.generic_string() << '\n'
      << "dataset.base_first_row\t" << command.base_first_row << '\n'
      << "dataset.nbase\t" << command.base_rows << '\n'
      << "model.h0\t" << command.nlist << '\n'
      << "large.train_format\t"
      << SourceFormatName(command.training_format) << '\n'
      << "large.base_format\t" << SourceFormatName(command.base_format)
      << '\n'
      << "train.init_samples\t" << command.initialization_samples << '\n'
      << "train.seed\t" << command.train_seed << '\n'
      << "train.max_R_iters\t" << command.training_rounds << '\n'
      << "train.use_opq_rotation\t"
      << static_cast<int>(command.train_use_opq_rotation) << '\n'
      << "train.log_metrics\t" << static_cast<int>(command.train_log_metrics)
      << '\n'
      << "train.log_linkage_pre_c1\t"
      << static_cast<int>(command.train_log_linkage_pre_c1) << '\n'
      << "train.init_linkage_mode\t" << command.init_linkage_mode << '\n'
      << "train.init_linkage_icm_round\t"
      << command.init_linkage_icm_rounds << '\n'
      << "train.kmeans_iters\t"
      << command.root_initialization.maximum_iterations << '\n'
      << "train.kmeans_tol\t" << command.root_initialization.tolerance << '\n'
      << "train.kmeansll_rounds\t"
      << command.root_initialization.kmeans_parallel_rounds << '\n'
      << "train.kmeansll_oversample\t"
      << command.root_initialization.kmeans_parallel_oversample << '\n'
      << "train.kmeansll_candidate_cap\t"
      << command.root_initialization.candidate_cap << '\n'
      << "train.kmeansll_seed\t"
      << command.root_initialization.kmeans_parallel_seed << '\n'
      << "train.kmeans_initial_weight\t"
      << command.root_initialization.initial_weight << '\n'
      << "train.kmeans_min_weight\t"
      << command.root_initialization.minimum_weight << '\n'
      << "train.kmeans_outlier_quantile\t"
      << command.root_initialization.outlier_quantile << '\n'
      << "train.kmeans_cost_threshold\t"
      << command.root_initialization.cost_threshold << '\n'
      << "train.kmeans_annealing_factor\t"
      << command.root_initialization.annealing_factor << '\n'
      << "train.kmeans_warmup_iters\t"
      << command.root_initialization.warmup_iterations << '\n'
      << "train.kmeans_quantile_bins\t"
      << command.root_initialization.quantile_bins << '\n'
      << "train.kmeans_large_k_threshold\t"
      << command.root_initialization.large_layer_threshold << '\n'
      << "train.kmeans_large_k_top_coarse\t"
      << command.root_initialization.large_layer_top_coarse << '\n'
      << "train.kmeans_large_k_hier2_enable\t"
      << static_cast<int>(command.root_initialization.large_layer_hierarchy_enabled) << '\n'
      << "train.kmeans_large_k_fixed_k1\t"
      << command.root_initialization.large_layer_fixed_coarse_count << '\n'
      << "train.kmeans_large_k_k1_min\t"
      << command.root_initialization.large_layer_minimum_coarse_count << '\n'
      << "train.kmeans_large_k_k1_max\t"
      << command.root_initialization.large_layer_maximum_coarse_count << '\n'
      << "train.kmeans_large_k_k1_pow2\t"
      << static_cast<int>(command.root_initialization.large_layer_round_coarse_to_power_of_two) << '\n'
      << "train.kmeans_device_cache_mb\t"
      << command.root_normalized_cache_mebibytes << '\n'
      << "base.encode.seed\t" << command.base_seed << '\n'
      << "virtual.good_fraction\t" << command.good_cluster_fraction << '\n'
      << "virtual.alpha_bad\t" << command.bad_cluster_training_weight << '\n'
      << "train.linkage.num_layers\t"
      << command.training_topology.radial_layer_count << '\n'
      << "train.linkage.root_percentile\t"
      << command.training_topology.root_fraction << '\n'
      << "train.linkage.depth_k\t" << command.training_topology.depth_k << '\n'
      << "train.linkage.knn_k\t" << command.training_topology.knn_k << '\n'
      << "train.linkage.bad_cluster_knn_backend\t" << command.training_topology.bad_cluster_knn_backend << '\n'
      << "train.linkage.bad_cluster_cpu_exact_max_points\t" << command.training_topology.bad_cluster_cpu_exact_max_points << '\n'
      << "base.linkage.bad_cluster_knn_backend\t" << command.base_topology.bad_cluster_knn_backend << '\n'
      << "base.linkage.bad_cluster_cpu_exact_max_points\t" << command.base_topology.bad_cluster_cpu_exact_max_points << '\n'
      << "hnsw.M\t" << command.hnsw_m << '\n'
      << "hnsw.ef_construction_cap\t" << command.hnsw_ef_construction_cap << '\n'
      << "hnsw.ef_search\t" << command.hnsw_ef_search << '\n'
      << "train.linkage.max_depth\t"
      << command.training_topology.maximum_depth << '\n'
      << "train.linkage.seed\t" << command.training_topology.seed << '\n'
      << "base.linkage.num_layers\t"
      << command.base_topology.radial_layer_count << '\n'
      << "base.linkage.root_percentile\t"
      << command.base_topology.root_fraction << '\n'
      << "base.linkage.depth_k\t" << command.base_topology.depth_k << '\n'
      << "base.linkage.knn_k\t" << command.base_topology.knn_k << '\n'
      << "base.linkage.max_depth\t"
      << command.base_topology.maximum_depth << '\n'
      << "base.linkage.seed\t" << command.base_topology.seed << '\n'
      << "virtual.virtual_ratio\t" << command.virtual_roots.ratio << '\n'
      << "virtual.min_virtual\t" << command.virtual_roots.minimum << '\n'
      << "virtual.max_virtual\t" << command.virtual_roots.maximum << '\n'
      << "virtual.umap_knn_k\t" << command.virtual_roots.neighbor_count << '\n'
      << "virtual.local_connectivity\t"
      << command.virtual_roots.local_connectivity << '\n'
      << "virtual.overlap_thr\t" << command.virtual_roots.overlap_threshold
      << '\n'
      << "virtual.prefer_peaks\t"
      << static_cast<int>(command.virtual_roots.prefer_density_peaks) << '\n'
      << "virtual.anchor_neighbor_k\t"
      << command.virtual_roots.anchor_neighbor_k << '\n'
      << "virtual.use_fixed_virtual_per_cluster\t"
      << static_cast<int>(command.virtual_roots.use_fixed_per_cluster) << '\n'
      << "virtual.fixed_virtual_per_cluster\t"
      << command.virtual_roots.fixed_per_cluster << '\n'
      << "large.tmp_dir\t" << command.work_directory.generic_string() << '\n'
      << "large.output_dir\t" << command.output_directory.generic_string()
      << '\n'
      << "large.profile_timing\t0\n"
      << "large.protect_existing_outputs\t"
      << static_cast<int>(command.protect_existing_outputs) << '\n'
      << "large.archive_log\t" << static_cast<int>(command.archive_log)
      << '\n'
      << "large.train_block\t" << command.training_block_rows << '\n'
      << "large.base_block\t" << command.base_block_rows << '\n'
      << "large.cluster_bucket_size\t" << command.cids_per_bucket << '\n'
      << "large.bucket_flush_mb\t" << command.bucket_flush_mebibytes << '\n'
      << "large.base_list_build_threads\t" << command.base_list_threads
      << '\n'
      << "large.cleanup.enabled\t"
      << static_cast<int>(command.cleanup.enabled) << '\n'
      << "large.cleanup.preset\t"
      << CleanupPresetName(command.cleanup.preset) << '\n'
      << "large.cleanup.keep_base_basic\t"
      << command.cleanup.keep_base_basic << '\n'
      << "large.cleanup.keep_base_basic_cluster_id\t"
      << command.cleanup.keep_base_basic_cluster_id << '\n'
      << "large.cleanup.keep_base_basic_ivf_lists\t"
      << command.cleanup.keep_base_basic_ivf_lists << '\n'
      << "large.cleanup.keep_base_list\t"
      << command.cleanup.keep_base_list << '\n'
      << "large.cleanup.keep_base_list_raw\t"
      << command.cleanup.keep_base_list_raw << '\n'
      << "large.cleanup.keep_linkage_list_f32\t"
      << command.cleanup.keep_linkage_list_f32 << '\n'
      << "large.cleanup.keep_linkage_parent_u32\t"
      << command.cleanup.keep_linkage_parent_u32 << '\n'
      << "large.base_linkage_store_coeffs_f32\t"
      << static_cast<int>(command.base_linkage_store_coefficients_f32)
      << '\n'
      << "large.linkage_coeff_codec.enabled\t"
      << static_cast<int>(command.coefficient_codec_enabled) << '\n'
      << "large.linkage_coeff_codec.p_first_candidates\t";
  write_percent_candidates(command.coefficient_first_magnitude_quantiles);
  output << '\n'
      << "large.linkage_coeff_codec.p_rest_candidates\t";
  write_percent_candidates(command.coefficient_rest_magnitude_quantiles);
  output << '\n'
      << "eval.disk_norm2_lut_kmeans_niter\t"
      << command.norm8_lloyd_iterations << '\n'
      << "eval.disk_norm2_mode\t" << command.eval_disk_norm2_mode
      << '\n' << "eval.disk_norm2_lut_mode\t" << Norm8ModeName(command.norm8_mode) << '\n'
      << "eval.base.enabled\t"
      << static_cast<int>(command.eval_base_enabled) << '\n'
      << "eval.base.use_ivf\t"
      << static_cast<int>(command.eval_base_use_ivf) << '\n'
      << "eval.base.nprobe\t" << command.eval_base_nprobe << '\n'
      << "eval.base.warmup\t" << command.eval_base_warmup << '\n'
      << "eval.base.repeat\t" << command.eval_base_repeat << '\n'
      << "eval.linkage.enabled\t"
      << static_cast<int>(command.eval_linkage_enabled) << '\n'
      << "eval.linkage.score_mode\t" << command.eval_score_mode << '\n'
      << "eval.linkage.modes\t";
  if (command.eval_representations_auto) {
    output << "auto";
  } else {
    output << '[';
    for (std::size_t index = 0U;
         index < command.eval_representations.size(); ++index) {
      if (index != 0U) output << ',';
      output << EvalRepresentationName(command.eval_representations[index]);
    }
    output << ']';
  }
  output << "\n"
      << "runtime.omp_threads\t" << command.cpu_threads << '\n'
      << "runtime.omp_places\t" << command.omp_places << '\n'
      << "runtime.omp_proc_bind\t" << command.omp_proc_bind << '\n'
      << "runtime.c_one_update_shards_init_linkage\t"
      << command.initialization_edge_moment_shards << '\n'
      << "runtime.c_one_update_shards\t"
      << command.training_edge_moment_shards << '\n'
      << "runtime.cuda_device\t" << command.cuda_device << '\n'
      << "runtime.use_cuda\t" << static_cast<int>(command.use_cuda) << '\n'
      << "eval.linkage.gpu_norm_enable\t"
      << static_cast<int>(command.gpu_norm_enable) << '\n'
      << "runtime.basic_inflight_blocks\t"
      << command.basic_inflight_blocks << '\n'
      << "runtime.basic_async_io\t"
      << static_cast<int>(command.basic_async_io_enabled) << '\n'
      << "runtime.basic_async_io_depth\t"
      << command.basic_async_io_depth << '\n'
      << "runtime.basic_async_io_mb\t"
      << command.basic_async_io_mebibytes << '\n'
      << "runtime.basic_async_write\t"
      << static_cast<int>(command.basic_async_write_enabled) << '\n'
      << "runtime.basic_async_write_depth\t"
      << command.basic_async_write_depth << '\n'
      << "runtime.basic_async_write_mb\t"
      << command.basic_async_write_mebibytes << '\n'
      << "runtime.cuda_mode\t"
      << (command.cuda_mode == CudaMode::kFast ? "fast" : "strict") << '\n'
      << "runtime.cuda_allow_tf32\t"
      << static_cast<int>(command.requested_cuda_allow_tf32) << '\n'
      << "runtime.linkage_async_io\t"
      << static_cast<int>(command.linkage_async_io_enabled) << '\n'
      << "runtime.linkage_async_io_depth\t"
      << command.linkage_async_io_depth << '\n'
      << "runtime.cuda_pool_size\t" << command.linkage_direct_pool_size
      << '\n'
      << "runtime.cuda_pool_size_init_linkage\t"
      << command.initialization_linkage_direct_pool_size << '\n'
      << "runtime.cuda_linkage_cross_cluster_batch\t"
      << static_cast<int>(command.linkage_scheduler_enabled) << '\n'
      << "runtime.cuda_linkage_cross_cluster_workers\t"
      << command.linkage_scheduler_workers << '\n'
      << "runtime.cuda_linkage_eval_async_pinned_mb\t"
      << command.linkage_eval_async_pinned_mebibytes << '\n'
      << "runtime.cuda_linkage_eval_async_pinned_mb_init_linkage\t"
      << command.initialization_linkage_eval_async_pinned_mebibytes << '\n'
      << "runtime.cuda_linkage_large_root_xc0_chunk_mb\t"
      << command.linkage_large_root_xc0_chunk_mebibytes << '\n'
      << "runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage\t"
      << command.init_linkage_large_root_xc0_chunk_mebibytes << '\n'
      << "runtime.cuda_linkage_mem_budget_mb\t"
      << command.linkage_large_root_workspace_mebibytes << '\n'
      << "runtime.cuda_linkage_mem_budget_mb_init_linkage\t"
      << command.init_linkage_large_root_workspace_mebibytes << '\n'
      << "runtime.cuda_linkage_single_gpu_min_candidates\t"
      << command.linkage_single_gpu_min_candidates << '\n'
      << "runtime.cuda_linkage_single_gpu_max_candidates\t"
      << command.linkage_single_gpu_max_candidates << '\n'
      << "runtime.cuda_linkage_inner_many_nodes_target_nodes\t"
      << command.linkage_inner_batch_rows << '\n'
      << "runtime.cuda_linkage_many_nodes_max_pairs\t"
      << command.linkage_max_batch_pairs << '\n'
      << "runtime.cuda_linkage_many_nodes_max_pairs_init_linkage\t"
      << command.linkage_init_max_batch_pairs << '\n'
      << "runtime.cuda_linkage_many_nodes_preflush_target_pairs\t"
      << command.linkage_many_nodes_preflush_target_pairs << '\n'
      << "runtime.cuda_linkage_same_layer_block_nodes\t"
      << command.linkage_same_layer_block_rows << '\n'
      << "runtime.cuda_linkage_same_layer_single_gpu_min_candidates\t"
      << command.linkage_same_layer_single_gpu_min_candidates << '\n'
      << "runtime.cuda_linkage_same_layer_window_min_pairs\t"
      << command.linkage_same_layer_minimum_pairs << '\n'
      << "runtime.cuda_linkage_same_layer_window_max_pairs\t"
      << command.linkage_same_layer_window_maximum_pairs << '\n'
      << "runtime.cuda_linkage_same_layer_window_max_pending_nodes\t"
      << command.linkage_same_layer_maximum_pending_nodes << '\n'
      << "runtime.cuda_linkage_same_layer_preflush_target_pairs\t"
      << command.linkage_same_layer_preflush_target_pairs << '\n'
      << "runtime.cuda_linkage_same_layer_preflush_target_nodes\t"
      << command.linkage_same_layer_preflush_target_nodes << '\n'
      << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable\t"
      << static_cast<int>(command.linkage_tiny_forced_cpu_enabled) << '\n'
      << "runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable\t"
      << static_cast<int>(
             command.initialization_linkage_tiny_forced_cpu_enabled) << '\n'
      << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs\t"
      << command.linkage_tiny_forced_pair_limit << '\n'
      << "runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes\t"
      << command.linkage_tiny_forced_node_limit << '\n'
      << "runtime.cuda_linkage_bad_frozen_window_nodes\t"
      << command.linkage_bad_window_rows << '\n'
      << "runtime.cuda_linkage_bad_frozen_window_max_pairs\t"
      << command.linkage_bad_window_maximum_pairs << '\n'
      << "runtime.cuda_linkage_chunk_max_pairs\t"
      << command.linkage_chunk_max_pairs << '\n'
      << "runtime.cuda_linkage_chunk_max_pairs_init_linkage\t"
      << command.init_linkage_chunk_max_pairs << '\n'
      << "stlqpp.root_update.accumulator_partitions\t"
      << command.root_moment_partitions << '\n';
}

void WriteRoundTripBuildConfig(
    const BuildCommand& command, std::ostream& output) {
  std::ostringstream canonical;
  WriteCanonicalBuildConfig(command, canonical);
  std::string line;
  std::istringstream input(canonical.str());
  while (std::getline(input, line)) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0U, separator);
    const std::string value = line.substr(separator + 1U);
    const bool string_value =
        key == "large.train_format" || key == "large.base_format" ||
        key == "dataset.name" || key == "dataset.data_root" ||
        key == "dataset.train_path" ||
        key == "dataset.base_path" || key == "large.tmp_dir" ||
        key == "large.output_dir" || key == "io.train_file" ||
        key == "io.pre_fix" || key == "io.base_file" ||
        key == "runtime.omp_places" || key == "runtime.omp_proc_bind" ||
        key == "runtime.cuda_mode" || key == "train.kmeans_init" ||
        key == "train.init_linkage_mode" ||
        key.ends_with(".bad_cluster_knn_backend") ||
        key == "large.cleanup.preset";
    output << key << " = "
           << (string_value ? QuoteConfigValue(value) : value) << '\n';
  }
  if (command.linkage_scheduler_stable) {
    output << "stlqpp.linkage_scheduler.stable = true\n";
  }
  for (const BuildConfigAssignment& assignment :
       command.effective_assignments) {
    if (IsProductConfigField(
            ProductConfigScope::kEval, assignment.key) &&
        !IsProductConfigField(
            ProductConfigScope::kBuild, assignment.key)) {
      output << assignment.key << " = "
             << QuoteConfigValue(assignment.value) << '\n';
    }
  }
}

StatusOr<BuildCommand> ParseBuildCommand(int argc, char** argv) {
  std::vector<ResolvedAssignment> assignments = {
      {"--train-enabled", "true", "default"},
      {"--train-init-enabled", "true", "default"},
      {"--base-encode-enabled", "true", "default"},
      {"--base-linkage-enabled", "true", "default"},
      {"--eval-only", "false", "default"},
      {"--train-checkpoint-enabled", "false", "default"},
      {"--train-checkpoint-every-r", "0", "default"},
      {"--archive-log", "true", "default"},
      {"--protect-existing-outputs", "false", "DEV default"},
      {"--work", "tmp", "Master default"},
      {"--output", "outputs", "Master default"},
      {"--train-first-row", "0", "default"},
      {"--base-first-row", "0", "default"},
      {"--data-root", "../data", "DEV default"},
      {"--root-initialization",
       "40,0.000001,4,16,0,1,0.1,0.85,0.115,0.9,2,65536,8192,1",
       "DEV defaults"},
      {"--training-topology", "10,0.01,6,10,40,49979687",
       "registered internal default"},
      {"--training-linkage-rsd-rank", "2", "registered default"},
      {"--base-topology", "14,0.01,12,30,40,86028121",
       "registered internal default"},
      {"--base-linkage-rsd-rank", "2", "registered default"},
      {"--virtual-roots", "0.1,1,2000,50,1,0.9,16", "DEV defaults"},
      {"--root-moment-partitions", "0", "STLQpp size-dependent default"},
      {"--rotation-moment-partitions", "32",
       "registered internal default"},
      {"--init-edge-moment-shards", "0", "DEV auto default"},
      {"--train-edge-moment-shards", "0", "DEV auto default"},
      {"--base-list-threads", "1", "DEV default"},
      {"--cuda-device", "0", "DEV default"},
      {"--train-block-rows", "200000", "registered internal default"},
      {"--base-block-rows", "200000", "registered internal default"},
      {"--cids-per-bucket", "256", "registered internal default"},
      {"--bucket-flush-mib", "256", "DEV default"},
      {"--basic-inflight-blocks", "6", "DEV default"},
      {"--basic-async-io-enabled", "true", "registered DEV recipe"},
      {"--basic-async-io-depth", "6", "registered DEV recipe"},
      {"--basic-async-io-mib", "1024", "registered DEV recipe"},
      {"--basic-async-write-enabled", "true", "DEV default"},
      {"--basic-async-write-depth", "8", "registered DEV recipe"},
      {"--basic-async-write-mib", "0", "DEV default"},
      {"--cuda-mode", "strict", "default"},
      {"--cuda-allow-tf32", "true", "default"},
      {"--train-basic-ils-iterations", "8", "DEV default"},
      {"--train-basic-icm-iterations", "4", "DEV default"},
      {"--train-basic-perturbations", "3", "DEV default"},
      {"--base-basic-ils-iterations", "32", "DEV default"},
      {"--base-basic-icm-iterations", "4", "DEV default"},
      {"--base-basic-perturbations", "3", "DEV default"},
      {"--train-use-opq-rotation", "true", "default"},
      {"--train-log-metrics", "true", "default"},
      {"--kmeansll-seed", "0", "default"},
      {"--virtual-prefer-peaks", "true", "default"},
      {"--virtual-enabled", "true", "default"},
      {"--virtual-use-fixed-per-cluster", "false", "default"},
      {"--virtual-fixed-per-cluster", "256", "default"},
      {"--linkage-tiny-forced-cpu-enabled", "false", "default"},
      {"--init-linkage-tiny-forced-cpu-enabled", "false", "default"},
      {"--linkage-tiny-forced-node-limit", "0", "default"},
      {"--linkage-scheduler-enabled", "true", "default"},
      {"--linkage-scheduler-transport-batching-enabled", "true", "default"},
      {"--linkage-scheduler-stable", "false", "default"},
      {"--linkage-direct-pool-size", "32", "default"},
      {"--init-linkage-direct-pool-size", "24", "default"},
      {"--linkage-eval-async-pinned-mib", "128", "default"},
      {"--init-linkage-eval-async-pinned-mib", "256", "default"},
      {"--linkage-large-root-xc0-chunk-mib", "256", "default"},
      {"--linkage-large-root-workspace-mib", "512", "default"},
      {"--init-linkage-large-root-xc0-chunk-mib", "0", "DEV fallback"},
      {"--init-linkage-large-root-workspace-mib", "0", "DEV fallback"},
      {"--linkage-chunk-max-pairs", "0", "DEV default"},
      {"--init-linkage-chunk-max-pairs", "0", "DEV fallback"},
      {"--root-score-buffer-mib", "256", "registered internal default"},
      {"--root-normalized-cache-mib", "20480",
       "registered internal default"},
      {"--topology-inflight-clusters", "48",
       "registered internal default"},
      {"--linkage-async-io-enabled", "true", "registered DEV recipe"},
      {"--linkage-async-io-depth", "6", "registered DEV recipe"},
      {"--linkage-scheduler-queue-capacity", "256",
       "registered internal default"},
      {"--linkage-rfull-cache-buffers", "8",
       "registered internal default"},
      {"--linkage-rfull-cache-mib", "256",
       "registered internal default"},
      {"--linkage-safety-reserve-mib", "512",
       "registered internal default"},
      {"--linkage-inner-batch-rows", "64", "DEV default"},
      {"--linkage-max-batch-pairs", "8192",
       "registered scheduler contract"},
      {"--linkage-init-max-batch-pairs", "0", "DEV fallback"},
      {"--linkage-many-nodes-preflush-target-pairs", "2048",
       "DEV default"},
      {"--linkage-same-layer-block-rows", "64", "DEV default"},
      {"--linkage-same-layer-minimum-pairs", "1", "DEV default"},
      {"--linkage-same-layer-window-maximum-pairs", "8192",
       "registered scheduler contract"},
      {"--linkage-same-layer-maximum-pending-nodes", "64",
       "DEV default"},
      {"--linkage-same-layer-preflush-target-pairs", "0", "DEV default"},
      {"--linkage-same-layer-preflush-target-nodes", "0", "DEV default"},
      {"--linkage-tiny-forced-pair-limit", "16", "DEV default"},
      {"--linkage-bad-window-rows", "64", "registered internal default"},
      {"--linkage-bad-window-maximum-pairs", "20000",
       "registered internal default"},
      {"--final-spool-window-clusters", "64",
       "registered internal default"},
      {"--final-spool-window-real-rows", "4096",
       "registered internal default"},
      {"--coefficient-weighted-quantile", "true", "DEV default"},
      {"--coefficient-codec-enabled", "true", "DEV default"},
      {"--coefficient-allow-clipping", "true", "DEV default"},
      {"--coefficient-fit-scale", "true", "DEV default"},
      {"--coefficient-refinement-sweeps", "3", "DEV default"},
      {"--coefficient-refinement-max-layer", "0", "DEV default"},
      {"--coefficient-refinement-step-limit", "1", "DEV default"},
      {"--eval-modes", "quantized_scaled", "production default"},
      {"--eval-score-mode", "l2", "production default"},
  };
  std::vector<std::filesystem::path> config_files;
  std::vector<FrozenBuildConfigInput> frozen_config_inputs;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc) {
      return InvalidCommand("missing value for " + std::string(option));
    }
    const std::string value = argv[++index];
    if (option == "--config") {
      auto decoded_path = DecodeConfigValue(value);
      if (!decoded_path.ok()) {
        return InvalidCommand(decoded_path.status().message());
      }
      const std::filesystem::path path = std::move(decoded_path).value();
      auto frozen = ReadFrozenConfig(path);
      if (!frozen.ok()) {
        return frozen.status();
      }
      if (const Status status = LoadConfigAssignments(
              path, frozen.value().bytes, &assignments);
          !status.ok()) {
        return status;
      }
      config_files.push_back(path);
      frozen_config_inputs.push_back(std::move(frozen).value());
      continue;
    }
    if (option == "--set") {
      if (const Status status = ApplyConfigLine(
              value, "argv[" + std::to_string(index) + "] --set",
              &assignments);
          !status.ok()) {
        return status;
      }
      continue;
    }
    if (!IsApplicationFieldOption(option)) {
      return InvalidCommand("unknown option " + std::string(option));
    }
    auto resolved = ResolveConfigKey(option);
    if (!resolved.ok()) return resolved.status();
    ResolvedConfigKey field = std::move(resolved).value();
    AssignResolved(&assignments, {
        .option = std::move(field.option),
        .value = value,
        .source = "argv[" + std::to_string(index - 1) + "]",
        .public_key = std::move(field.public_key),
    });
  }

  // Master supplies "tmp" and "outputs" when the fields are omitted. Treat
  // an explicitly empty path as omitted too: it is not a useful destination,
  // and complete copied recipes commonly leave these operational fields blank.
  for (ResolvedAssignment& assignment : assignments) {
    if (assignment.value.empty() &&
        (assignment.option == "--work" || assignment.option == "--output")) {
      assignment.value = assignment.option == "--work" ? "tmp" : "outputs";
      assignment.source += " (empty path: Master default)";
    }
  }

  const auto find_public_assignment =
      [&](std::string_view key) -> const ResolvedAssignment* {
    const auto found = std::find_if(
        assignments.begin(), assignments.end(),
        [&](const ResolvedAssignment& assignment) {
          return assignment.public_key == key;
        });
    return found == assignments.end() ? nullptr : &*found;
  };
  bool resolved_eval_only = false;
  if (const ResolvedAssignment* assignment = find_public_assignment(
          "advanced.eval_only");
      assignment != nullptr &&
      !ParseBool(assignment->value, &resolved_eval_only)) {
    return InvalidCommand("advanced.eval_only must be boolean");
  }
  const ResolvedAssignment* coefficient_first = find_public_assignment(
      "large.linkage_coeff_codec.p_first_candidates");
  const ResolvedAssignment* coefficient_rest = find_public_assignment(
      "large.linkage_coeff_codec.p_rest_candidates");
  if (!resolved_eval_only &&
      (coefficient_first == nullptr || coefficient_rest == nullptr)) {
    return InvalidCommand(
        "the admitted codec requires both first/rest candidate arrays");
  }
  const ResolvedAssignment* requested_disk_norm = find_public_assignment(
      "eval.disk_norm2_mode");
  const bool requested_float_norm = requested_disk_norm != nullptr &&
      requested_disk_norm->value == "exact";

  std::vector<std::string> storage;
  storage.reserve(1U + assignments.size() * 2U);
  storage.emplace_back(argc > 0 ? argv[0] : "stlqpp_build_index");
  for (const ResolvedAssignment& assignment : assignments) {
    if (assignment.option.empty()) continue;
    // Eval-only selects its LUT recipe after inspecting the source producer;
    // the build recipe must not override that source during this projection.
    if (resolved_eval_only &&
        assignment.public_key == "eval.disk_norm2_lut_mode") {
      continue;
    }
    if (assignment.public_key.empty() &&
        std::any_of(assignments.begin(), assignments.end(),
                    [&](const ResolvedAssignment& product) {
                      return !product.public_key.empty() &&
                          product.option == assignment.option;
                    })) {
      continue;
    }
    storage.push_back(assignment.option);
    storage.push_back(assignment.value);
  }
  std::vector<char*> flat_argv;
  flat_argv.reserve(storage.size());
  for (std::string& value : storage) {
    flat_argv.push_back(value.data());
  }
  auto parsed = ParseFlatBuildCommand(
      static_cast<int>(flat_argv.size()), flat_argv.data());
  if (!parsed.ok()) {
    return parsed.status();
  }
  if (parsed.value().eval_only_sparse_config) {
    return parsed;
  }
  parsed.value().eval_disk_norm2_mode = requested_float_norm ? "exact" : "lut";
  auto normalized_bits = NormalizeCoefficientBits(
      std::move(parsed.value().coefficient_bits_per_layer),
      parsed.value().m,
      "large.linkage_coeff_codec.bits_per_layer");
  if (!normalized_bits.ok()) return normalized_bits.status();
  parsed.value().coefficient_bits_per_layer =
      std::move(normalized_bits).value();
  if (!parsed.value().coefficient_bits_per_layer_root.empty()) {
    auto normalized_root = NormalizeCoefficientBits(
        std::move(parsed.value().coefficient_bits_per_layer_root),
        parsed.value().m,
        "large.linkage_coeff_codec.bits_per_layer_root");
    if (!normalized_root.ok()) return normalized_root.status();
    parsed.value().coefficient_bits_per_layer_root =
        std::move(normalized_root).value();
  }
  if (!parsed.value().coefficient_bits_per_layer_linkage.empty()) {
    auto normalized_linkage = NormalizeCoefficientBits(
        std::move(parsed.value().coefficient_bits_per_layer_linkage),
        parsed.value().m,
        "large.linkage_coeff_codec.bits_per_layer_linkage");
    if (!normalized_linkage.ok()) return normalized_linkage.status();
    parsed.value().coefficient_bits_per_layer_linkage =
        std::move(normalized_linkage).value();
  }
  for (const ResolvedAssignment& assignment : assignments) {
    if (!assignment.public_key.empty() && assignment.option.empty()) {
      if (IsProductConfigField(
              ProductConfigScope::kEval, assignment.public_key) &&
          !IsProductConfigField(
              ProductConfigScope::kBuild, assignment.public_key)) {
        continue;
      }
      // This DEV key is a fixed false contract during index construction, but
      // the unified advanced.eval_only entry routes it to the eval parser.
      // Keep build-time rejection while allowing the same complete cfg to
      // enable the int-only timing diagnostic during eval-only execution.
      if (parsed.value().eval_only &&
          assignment.public_key == "large.profile_timing") {
        bool profile_timing = false;
        if (!ParseBool(assignment.value, &profile_timing)) {
          return InvalidCommand("large.profile_timing must be boolean");
        }
        continue;
      }
      if (const Status status = ValidateFixedProductField(
              assignment.public_key, assignment.value);
          !status.ok()) {
        return status;
      }
    }
  }
  if (!parsed.value().eval_only && parsed.value().base_linkage_enabled &&
      !parsed.value().coefficient_codec_enabled &&
      !parsed.value().eval_representations_auto &&
      std::any_of(parsed.value().eval_representations.begin(),
                  parsed.value().eval_representations.end(),
                  [](EvalRepresentation representation) {
                    return representation != EvalRepresentation::kFloat &&
                        representation != EvalRepresentation::kFloatScaled;
                  })) {
    return InvalidCommand(
        "large.linkage_coeff_codec.enabled=false requires "
        "eval.linkage.modes=float, float_scaled or auto for the produced float-only index");
  }
  parsed.value().config_files = std::move(config_files);
  parsed.value().frozen_config_inputs = std::move(frozen_config_inputs);
  parsed.value().effective_assignments.reserve(assignments.size());
  for (ResolvedAssignment& assignment : assignments) {
    if (assignment.public_key.empty()) continue;
    parsed.value().effective_assignments.push_back({
        .key = std::move(assignment.public_key),
        .value = std::move(assignment.value),
        .source = std::move(assignment.source),
    });
  }
  if (const Status status = ValidateBuildPathIsolation(parsed.value());
      !status.ok()) {
    return status;
  }
  return parsed;
}

void PrintBuildUsage(std::ostream& output) {
  output
      << "usage: stlqpp_build_index --config <file.cfg> "
         "[--config <file.cfg>]... [--set <key=value>]... "
         "[--dump-config <path>]\n"
      << "Use --set to override configuration fields, for example "
         "--set runtime.omp_threads=16. --dump-config writes the resolved "
         "configuration and exits. See README.md for training, model reuse, "
         "and evaluation examples.\n";
}

}  // namespace stlqpp::app
