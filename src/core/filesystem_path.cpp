#include "stlqpp/core/filesystem_path.h"

#include <string>

#ifdef _WIN32
#include <cwctype>
#endif

namespace stlqpp {
namespace {

bool ComponentEquals(const std::filesystem::path& left,
                     const std::filesystem::path& right) noexcept {
#ifdef _WIN32
  const auto& a = left.native();
  const auto& b = right.native();
  if (a.size() != b.size()) return false;
  for (std::size_t index = 0U; index < a.size(); ++index) {
    if (std::towlower(a[index]) != std::towlower(b[index])) return false;
  }
  return true;
#else
  return left == right;
#endif
}

bool IsEqualOrAncestor(const std::filesystem::path& ancestor,
                       const std::filesystem::path& descendant) noexcept {
  auto ancestor_part = ancestor.begin();
  auto descendant_part = descendant.begin();
  while (ancestor_part != ancestor.end() &&
         descendant_part != descendant.end()) {
    if (!ComponentEquals(*ancestor_part, *descendant_part)) return false;
    ++ancestor_part;
    ++descendant_part;
  }
  return ancestor_part == ancestor.end();
}

}  // namespace

StatusOr<std::filesystem::path> ResolveFilesystemPath(
    const std::filesystem::path& path) {
  if (path.empty()) {
    return Status::InvalidArgument(
        "cannot resolve an empty filesystem path");
  }
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    return Status::InvalidArgument(
        "cannot make filesystem path absolute: " + path.string());
  }
  std::filesystem::path resolved =
      std::filesystem::weakly_canonical(absolute, error);
  if (error) {
    return Status::InvalidArgument(
        "cannot resolve filesystem path: " + path.string());
  }
  return resolved.lexically_normal();
}

Status ValidateDisjointFilesystemPaths(
    const std::filesystem::path& first,
    std::string_view first_role,
    const std::filesystem::path& second,
    std::string_view second_role) {
  auto resolved_first = ResolveFilesystemPath(first);
  if (!resolved_first.ok()) return resolved_first.status();
  auto resolved_second = ResolveFilesystemPath(second);
  if (!resolved_second.ok()) return resolved_second.status();
  if (IsEqualOrAncestor(resolved_first.value(), resolved_second.value()) ||
      IsEqualOrAncestor(resolved_second.value(), resolved_first.value())) {
    return Status::InvalidArgument(
        std::string(first_role) + " overlaps " + std::string(second_role) +
        " after path and symlink resolution: " +
        resolved_first.value().string() + " vs " +
        resolved_second.value().string());
  }
  return Status::Ok();
}

Status ValidateIndexDerivationPaths(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  auto resolved_source = ResolveFilesystemPath(source);
  if (!resolved_source.ok()) return resolved_source.status();
  auto resolved_destination = ResolveFilesystemPath(destination);
  if (!resolved_destination.ok()) return resolved_destination.status();
  if (resolved_destination.value().parent_path() ==
          resolved_source.value() / "derived-index-cache" &&
      resolved_destination.value().filename() != "." &&
      resolved_destination.value().filename() != "..") {
    return Status::Ok();
  }
  return ValidateDisjointFilesystemPaths(
      source, "derived index source", destination,
      "derived index destination");
}

}  // namespace stlqpp
