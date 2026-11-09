#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace stlqpp {

// One complete, user-visible search representation.  Keeping coefficient
// and score choices together prevents invalid cross-products such as quantized
// coefficients with native-float norms.
enum class EvalRepresentation : std::uint8_t {
  kFloat = 0U,
  kFloatScaled,
  kQuantized,
  kQuantizedScaled,
};

// The requested selection policy must survive parsing until the index
// manifest is available.  In particular, auto is not an alias for a fixed
// representation list.
struct EvalRepresentationSelection final {
  bool automatic = false;
  std::vector<EvalRepresentation> explicit_values;
};

[[nodiscard]] constexpr const char* EvalRepresentationName(
    EvalRepresentation value) noexcept {
  switch (value) {
    case EvalRepresentation::kFloat: return "float";
    case EvalRepresentation::kFloatScaled: return "float_scaled";
    case EvalRepresentation::kQuantized: return "quantized";
    case EvalRepresentation::kQuantizedScaled: return "quantized_scaled";
  }
  return "unknown";
}

[[nodiscard]] inline bool ParseEvalRepresentationSelection(
    std::string_view raw, EvalRepresentationSelection* output) {
  if (output == nullptr) return false;
  const auto trim = [](std::string_view value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front())) != 0) {
      value.remove_prefix(1U);
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back())) != 0) {
      value.remove_suffix(1U);
    }
    return value;
  };
  raw = trim(raw);
  if (raw.size() >= 2U && raw.front() == '[' && raw.back() == ']') {
    raw = trim(raw.substr(1U, raw.size() - 2U));
  }
  if (raw == "auto") {
    output->automatic = true;
    output->explicit_values.clear();
    return true;
  }
  // The old two-way "both" spelling became ambiguous after scaled score8
  // was admitted.  Public configuration uses auto or an explicit list.
  if (raw == "both") return false;

  EvalRepresentationSelection parsed;
  while (!raw.empty()) {
    const std::size_t separator = raw.find_first_of(", ");
    std::string token(trim(raw.substr(0U, separator)));
    if (token.size() >= 2U &&
        ((token.front() == '\'' && token.back() == '\'') ||
         (token.front() == '"' && token.back() == '"'))) {
      token = token.substr(1U, token.size() - 2U);
    }
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char ch) {
                     return static_cast<char>(std::tolower(ch));
                   });
    EvalRepresentation value;
    if (token == "float") {
      value = EvalRepresentation::kFloat;
    } else if (token == "float_scaled") {
      value = EvalRepresentation::kFloatScaled;
    } else if (token == "quantized") {
      value = EvalRepresentation::kQuantized;
    } else if (token == "quantized_scaled") {
      value = EvalRepresentation::kQuantizedScaled;
    } else if (!token.empty()) {
      return false;
    } else if (separator == std::string_view::npos) {
      break;
    } else {
      raw.remove_prefix(separator + 1U);
      raw = trim(raw);
      continue;
    }
    if (std::find(parsed.explicit_values.begin(),
                  parsed.explicit_values.end(), value) !=
        parsed.explicit_values.end()) {
      return false;
    }
    parsed.explicit_values.push_back(value);
    if (separator == std::string_view::npos) break;
    raw.remove_prefix(separator + 1U);
    raw = trim(raw);
  }
  if (parsed.explicit_values.empty()) return false;
  *output = std::move(parsed);
  return true;
}

}  // namespace stlqpp
