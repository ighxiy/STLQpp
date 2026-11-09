#include "stlqpp/application/sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace stlqpp::app {
namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

class Sha256 final {
 public:
  void Add(const std::uint8_t* bytes, std::size_t count) noexcept {
    bit_count_ += static_cast<std::uint64_t>(count) * 8U;
    while (count != 0U) {
      const std::size_t copied = std::min(count, block_.size() - used_);
      std::copy_n(bytes, copied, block_.data() + used_);
      bytes += copied;
      count -= copied;
      used_ += copied;
      if (used_ == block_.size()) {
        Compress(block_.data());
        used_ = 0U;
      }
    }
  }

  std::string Finish() noexcept {
    const std::uint64_t original_bits = bit_count_;
    const std::uint8_t marker = 0x80U;
    Add(&marker, 1U);
    const std::uint8_t zero = 0U;
    while (used_ != 56U) Add(&zero, 1U);
    std::array<std::uint8_t, 8> length{};
    for (std::size_t index = 0U; index < length.size(); ++index) {
      length[7U - index] = static_cast<std::uint8_t>(
          original_bits >> (index * 8U));
    }
    Add(length.data(), length.size());
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint32_t value : state_) {
      output << std::setw(8) << value;
    }
    return output.str();
  }

 private:
  void Compress(const std::uint8_t* block) noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0U; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = static_cast<std::uint32_t>(block[offset]) << 24U |
          static_cast<std::uint32_t>(block[offset + 1U]) << 16U |
          static_cast<std::uint32_t>(block[offset + 2U]) << 8U |
          static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t x = words[index - 15U];
      const std::uint32_t y = words[index - 2U];
      const std::uint32_t s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
      const std::uint32_t s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0U; index < words.size(); ++index) {
      const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^
          std::rotr(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + sum1 + choose + kRound[index] +
          words[index];
      const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^
          std::rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = sum0 + majority;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  std::array<std::uint8_t, 64> block_{};
  std::size_t used_ = 0U;
  std::uint64_t bit_count_ = 0U;
};

}  // namespace

std::string Sha256Bytes(std::string_view bytes) {
  Sha256 hash;
  hash.Add(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  return hash.Finish();
}

StatusOr<std::string> Sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status::FailedPrecondition(
        "cannot open identity input " + path.string());
  }
  Sha256 hash;
  std::array<char, 1U << 16U> bytes{};
  while (input) {
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash.Add(reinterpret_cast<const std::uint8_t*>(bytes.data()),
               static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) {
    return Status::Internal("failed reading identity input " + path.string());
  }
  return hash.Finish();
}

StatusOr<std::string> Sha256CurrentExecutable() {
#if defined(_WIN32)
  std::vector<wchar_t> path(1024U);
  while (true) {
    const DWORD copied = GetModuleFileNameW(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (copied == 0U) {
      return Status::FailedPrecondition(
          "cannot identify the running executable");
    }
    if (static_cast<std::size_t>(copied) < path.size() - 1U) {
      return Sha256File(std::filesystem::path(
          path.data(), path.data() + copied));
    }
    path.resize(path.size() * 2U);
  }
#elif defined(__linux__)
  // Opening this procfs identity hashes the running executable inode, even
  // when argv[0] was a bare PATH name or a symlink spelling.
  return Sha256File("/proc/self/exe");
#else
  return Status::FailedPrecondition(
      "current executable identity is unsupported on this platform");
#endif
}

}  // namespace stlqpp::app
