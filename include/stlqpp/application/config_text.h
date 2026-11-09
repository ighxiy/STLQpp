#pragma once

#include "stlqpp/core/status_or.h"

#include <string>
#include <string_view>

namespace stlqpp::app {

inline std::string TrimConfigText(std::string_view input) {
  const std::size_t first = input.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const std::size_t last = input.find_last_not_of(" \t\r\n");
  return std::string(input.substr(first, last - first + 1U));
}

// Config comments are recognized only outside quoted values. Escaping is
// stateful so an even run of backslashes does not accidentally protect the
// following quote.
inline StatusOr<std::string> StripConfigComment(std::string_view input) {
  bool quoted = false;
  bool escaped = false;
  char quote = '\0';
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const char ch = input[index];
    if (quoted) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == quote) {
        quoted = false;
      }
      continue;
    }
    if (ch == '\'' || ch == '"') {
      quoted = true;
      quote = ch;
      continue;
    }
    if (ch == '#' ||
        (ch == '/' && index + 1U < input.size() &&
         input[index + 1U] == '/')) {
      return TrimConfigText(input.substr(0U, index));
    }
  }
  if (quoted || escaped) {
    return Status::InvalidArgument("unterminated quoted config value");
  }
  return TrimConfigText(input);
}

inline StatusOr<std::string> DecodeConfigValue(std::string_view input) {
  const std::string value = TrimConfigText(input);
  if (value.empty() || (value.front() != '\'' && value.front() != '"')) {
    return value;
  }
  const char quote = value.front();
  if (value.size() < 2U || value.back() != quote) {
    return Status::InvalidArgument("unterminated quoted config value");
  }
  std::string output;
  output.reserve(value.size() - 2U);
  bool escaped = false;
  for (std::size_t index = 1U; index + 1U < value.size(); ++index) {
    const char ch = value[index];
    if (!escaped) {
      if (ch == '\\') {
        escaped = true;
      } else {
        output.push_back(ch);
      }
      continue;
    }
    escaped = false;
    switch (ch) {
      case '\\': output.push_back('\\'); break;
      case '\'': output.push_back('\''); break;
      case '"': output.push_back('"'); break;
      case 'n': output.push_back('\n'); break;
      case 'r': output.push_back('\r'); break;
      case 't': output.push_back('\t'); break;
      default:
        return Status::InvalidArgument("unsupported config escape");
    }
  }
  if (escaped) {
    return Status::InvalidArgument("unterminated config escape");
  }
  return output;
}

// Snapshots always quote strings. This makes spaces, comment markers and
// backslashes round-trip through the same production parser on every platform.
inline std::string QuoteConfigValue(std::string_view input) {
  std::string output;
  output.reserve(input.size() + 2U);
  output.push_back('"');
  for (const char ch : input) {
    switch (ch) {
      case '\\': output.append("\\\\"); break;
      case '"': output.append("\\\""); break;
      case '\n': output.append("\\n"); break;
      case '\r': output.append("\\r"); break;
      case '\t': output.append("\\t"); break;
      default: output.push_back(ch); break;
    }
  }
  output.push_back('"');
  return output;
}

}  // namespace stlqpp::app
