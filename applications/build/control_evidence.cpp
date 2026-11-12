#include "control_evidence.h"

#include "build_control_manifest.h"

#include "stlqpp/application/sha256.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace stlqpp::app {
namespace {

using Fields = std::map<std::string, std::string>;

struct Evidence final {
  Fields top;
  Fields left;
  Fields right;
  Fields sections;
  Fields allowed;
  Fields required;
  Fields common;
};

Status InvalidEvidence(std::string detail) {
  return Status::FailedPrecondition(
      "guarded control evidence: " + std::move(detail));
}

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = line.find('\t', begin);
    fields.push_back(line.substr(begin, end - begin));
    if (end == std::string::npos) return fields;
    begin = end + 1U;
  }
}

Status InsertUnique(Fields& fields,
                    std::string key,
                    std::string value) {
  if (key.empty() || !fields.emplace(std::move(key), std::move(value)).second) {
    return InvalidEvidence("contains an empty or duplicate field");
  }
  return Status::Ok();
}

StatusOr<Evidence> ReadEvidence(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return InvalidEvidence("cannot be opened: " + path.string());
  Evidence output;
  std::string line;
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = SplitTabs(line);
    Status status;
    if (fields.size() == 2U && fields[0] == "section") {
      status = InsertUnique(output.sections, fields[1], "present");
    } else if (fields.size() == 2U && fields[0] == "allowed") {
      status = InsertUnique(output.allowed, fields[1], "present");
    } else if (fields.size() == 2U && fields[0] == "required") {
      status = InsertUnique(output.required, fields[1], "present");
    } else if (fields.size() == 3U && fields[0] == "common") {
      status = InsertUnique(output.common, fields[1], fields[2]);
    } else if (fields.size() == 2U) {
      status = InsertUnique(output.top, fields[0], fields[1]);
    } else if (fields.size() == 3U && fields[0] == "left") {
      status = InsertUnique(output.left, fields[1], fields[2]);
    } else if (fields.size() == 3U && fields[0] == "right") {
      status = InsertUnique(output.right, fields[1], fields[2]);
    } else {
      return InvalidEvidence("contains a malformed TSV row");
    }
    if (!status.ok()) return status;
  }
  if (!input.eof()) return InvalidEvidence("could not be read completely");
  const auto version = output.top.find("evidence.version");
  if (version == output.top.end() || version->second != "2") {
    return InvalidEvidence("requires evidence.version 2");
  }
  return output;
}

StatusOr<Fields> ParseManifest(std::string_view text) {
  Fields output;
  std::istringstream input{std::string(text)};
  std::string line;
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = SplitTabs(line);
    if (fields.size() != 2U) {
      return InvalidEvidence("production manifest contains malformed rows");
    }
    if (const Status status = InsertUnique(
            output, fields[0], fields[1]); !status.ok()) {
      return status;
    }
  }
  return output;
}

std::filesystem::path NormalizedAbsolute(
    const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  return (error ? path : absolute).lexically_normal();
}

bool SamePath(const std::filesystem::path& left,
              const std::filesystem::path& right) {
  return !left.empty() && !right.empty() &&
      NormalizedAbsolute(left) == NormalizedAbsolute(right);
}

StatusOr<Fields> ReadIdentitySidecar(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return InvalidEvidence("cannot open identity sidecar");
  Fields output;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t equal = line.find('=');
    if (equal == std::string::npos || equal == 0U) {
      return InvalidEvidence("identity sidecar contains a malformed row");
    }
    if (const Status status = InsertUnique(
            output, line.substr(0U, equal), line.substr(equal + 1U));
        !status.ok()) {
      return status;
    }
  }
  if (!input.eof()) return InvalidEvidence("identity sidecar read failed");
  for (const char* key : {"identity.version", "content.id", "shape", "slice"}) {
    const auto found = output.find(key);
    if (found == output.end() || found->second.empty()) {
      return InvalidEvidence(
          std::string("identity sidecar is missing ") + key);
    }
  }
  if (output.at("identity.version") != "1") {
    return InvalidEvidence("unsupported content identity sidecar version");
  }
  return output;
}

Status ValidateDirectoryIdentity(
    const std::filesystem::path& directory,
    const std::filesystem::path& sidecar,
    std::string_view expected_sidecar_sha) {
  auto sidecar_sha = Sha256File(sidecar);
  if (!sidecar_sha.ok()) return sidecar_sha.status();
  if (sidecar_sha.value() != expected_sidecar_sha) {
    return InvalidEvidence("content identity sidecar changed after guard");
  }
  auto identity = ReadIdentitySidecar(sidecar);
  if (!identity.ok()) return identity.status();
  const Fields& fields = identity.value();
  const auto fingerprint = fields.contains("manifest.fingerprint")
      ? fields.find("manifest.fingerprint") : fields.find("manifest.sha256");
  if (fingerprint == fields.end() || fingerprint->second.empty()) {
    return InvalidEvidence("identity sidecar lacks a manifest fingerprint");
  }
  std::filesystem::path manifest;
  if (const auto named = fields.find("manifest.relative_path");
      named != fields.end()) {
    const std::filesystem::path relative = named->second;
    if (relative.empty() || relative.is_absolute() ||
        relative.lexically_normal().string().starts_with("..")) {
      return InvalidEvidence("identity manifest path is not portable relative");
    }
    manifest = directory / relative;
  } else if (std::filesystem::is_regular_file(
                 directory / "checkpoint.stlqpp")) {
    manifest = directory / "checkpoint.stlqpp";
  } else if (std::filesystem::is_regular_file(
                 directory / "train_manifest.stlqpp")) {
    manifest = directory / "train_manifest.stlqpp";
  } else if (std::filesystem::is_regular_file(
                 directory / "trained_model.stlqpp")) {
    manifest = directory / "trained_model.stlqpp";
  } else if (std::filesystem::is_regular_file(
                 directory / "manifest.stlqpp")) {
    manifest = directory / "manifest.stlqpp";
  } else {
    return InvalidEvidence("identity directory has no registered manifest");
  }
  auto manifest_sha = Sha256File(manifest);
  if (!manifest_sha.ok()) return manifest_sha.status();
  if (manifest_sha.value() != fingerprint->second) {
    return InvalidEvidence("identity sidecar describes different content");
  }
  static constexpr std::array<std::pair<std::string_view, std::string_view>, 4>
      kKnownFiles = {{{"model.sha256", "model.bin"},
                      {"forest.sha256", "forest.bin"},
                      {"directory.sha256", "directory.bin"},
                      {"bad-clusters.sha256", "bad_clusters.bin"}}};
  for (const auto& [key, filename] : kKnownFiles) {
    const auto expected = fields.find(std::string(key));
    if (expected == fields.end()) continue;
    auto actual = Sha256File(directory / filename);
    if (!actual.ok()) return actual.status();
    if (actual.value() != expected->second) {
      return InvalidEvidence(
          "identity sidecar file hash disagrees with directory content");
    }
  }
  return Status::Ok();
}

Status ValidateInputIdentity(
    const Evidence& evidence,
    std::string_view side,
    std::string_view role,
    const std::filesystem::path& declared_path,
    std::string_view expected_sha) {
  const std::string prefix(side);
  const std::string path_key = prefix + "." + std::string(role);
  const std::string sidecar_key = path_key + "-identity-sidecar";
  const auto path = evidence.top.find(path_key);
  const auto sidecar = evidence.top.find(sidecar_key);
  if (path == evidence.top.end() || sidecar == evidence.top.end() ||
      !SamePath(path->second, declared_path)) {
    return InvalidEvidence(
        std::string(role) + " path is not bound to this build command");
  }
  if (std::filesystem::is_regular_file(declared_path)) {
    if (!sidecar->second.empty()) {
      return InvalidEvidence("file input cannot use a directory sidecar");
    }
    auto actual = Sha256File(declared_path);
    if (!actual.ok()) return actual.status();
    return actual.value() == expected_sha
        ? Status::Ok()
        : InvalidEvidence(std::string(role) + " file changed after guard");
  }
  if (!std::filesystem::is_directory(declared_path) ||
      sidecar->second.empty()) {
    return InvalidEvidence(
        std::string(role) + " directory lacks a bound sidecar");
  }
  return ValidateDirectoryIdentity(
      declared_path, sidecar->second, expected_sha);
}

bool ManifestMatches(const Fields& evidence, const Fields& actual) {
  for (const auto& [key, value] : actual) {
    const auto found = evidence.find(key);
    if (found == evidence.end() || found->second != value) return false;
  }
  return true;
}

const std::string* Find(const Fields& fields, const std::string& key) {
  const auto found = fields.find(key);
  return found == fields.end() ? nullptr : &found->second;
}

Status ValidateSide(
    const Evidence& evidence,
    std::string_view side,
    const Fields& side_fields,
    const BuildCommand& command) {
  const std::string prefix(side);
  const std::string* composed_config =
      Find(evidence.top, prefix + ".composed-config");
  const std::string* composed_sha =
      Find(evidence.top, prefix + ".composed-config-sha256");
  if (composed_config == nullptr || composed_sha == nullptr ||
      command.config_files.size() != 1U ||
      command.frozen_config_inputs.size() != 1U ||
      !SamePath(*composed_config, command.config_files.front()) ||
      command.frozen_config_inputs.front().path != command.config_files.front() ||
      Sha256Bytes(command.frozen_config_inputs.front().bytes) != *composed_sha) {
    return InvalidEvidence(
        "side is not bound to the exact guarded composed config");
  }
  const std::string* checkpoint_sha =
      Find(side_fields, "experiment.checkpoint-sha256");
  const std::string* workset_sha =
      Find(side_fields, "experiment.workset-sha256");
  if (checkpoint_sha == nullptr || workset_sha == nullptr) {
    return InvalidEvidence("side lacks input identities");
  }
  const std::filesystem::path checkpoint =
      !command.resume_round_checkpoint.empty()
          ? command.resume_round_checkpoint
          : !command.resume_initialized_checkpoint.empty()
              ? command.resume_initialized_checkpoint
              : command.effective_training_file.empty()
                  ? command.training_file
                  : command.effective_training_file;
  if (const Status status = ValidateInputIdentity(
          evidence, side, "checkpoint", checkpoint, *checkpoint_sha);
      !status.ok()) {
    return status;
  }
  const std::filesystem::path workset = command.effective_base_file.empty()
      ? command.base_file : command.effective_base_file;
  return ValidateInputIdentity(
      evidence, side, "workset", workset, *workset_sha);
}

}  // namespace

Status ValidateControlEvidence(
    const BuildCommand& command,
    const EffectiveBuildRecipe& recipe) {
  if (command.control_evidence.empty()) return Status::Ok();
  auto binary_sha = Sha256CurrentExecutable();
  if (!binary_sha.ok()) return binary_sha.status();
  auto evidence = ReadEvidence(command.control_evidence);
  if (!evidence.ok()) return evidence.status();
  std::ostringstream manifest_text;
  WriteBuildControlManifest(command, recipe, manifest_text);
  auto actual_manifest = ParseManifest(manifest_text.str());
  if (!actual_manifest.ok()) return actual_manifest.status();

  std::vector<std::string_view> matching;
  for (const auto& [name, fields] :
       std::array<std::pair<std::string_view, const Fields*>, 2>{{
           {"left", &evidence.value().left},
           {"right", &evidence.value().right}}}) {
    const std::string* recorded =
        Find(*fields, "experiment.binary-sha256");
    if (recorded != nullptr && *recorded == binary_sha.value() &&
        ManifestMatches(*fields, actual_manifest.value())) {
      matching.push_back(name);
    }
  }
  if (matching.size() != 1U) {
    return InvalidEvidence(
        "must identify exactly one matching executable/config leg");
  }
  const Fields& side_fields = matching.front() == "left"
      ? evidence.value().left : evidence.value().right;
  return ValidateSide(
      evidence.value(), matching.front(), side_fields, command);
}

}  // namespace stlqpp::app
