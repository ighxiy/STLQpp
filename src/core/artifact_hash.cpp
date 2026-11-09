#include "stlqpp/core/artifact_hash.h"

#include <array>

namespace stlqpp {

std::string ArtifactHash64::ToHex() const {
  static constexpr std::array<char, 16> kDigits = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
  };
  std::string text(18U, '0');
  text[0] = '0';
  text[1] = 'x';
  for (std::uint32_t digit = 0U; digit < 16U; ++digit) {
    const std::uint32_t shift = (15U - digit) * 4U;
    const auto nibble = static_cast<std::size_t>((value_ >> shift) & 0x0fU);
    text[2U + digit] = kDigits[nibble];
  }
  return text;
}

}  // namespace stlqpp
