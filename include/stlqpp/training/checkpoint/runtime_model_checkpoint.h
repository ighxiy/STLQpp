#pragma once

#include "stlqpp/storage/canonical_model_codec.h"
#include "stlqpp/training/checkpoint/training_checkpoint.h"

namespace stlqpp {
struct RuntimeCheckpointModel final {
  RuntimeCanonicalModel model;
  std::vector<std::uint8_t> bad_cluster_mask;
};

// Reads the model boundary of a trained/initialized artifact. It does not
// expose a replay forest. Further training must first rebuild Full topology
// before any Replay round can consume a forest.
[[nodiscard]] StatusOr<RuntimeCheckpointModel> OpenRuntimeCheckpointModel(
    const TrainingCheckpointOpenRequest& request);
[[nodiscard]] StatusOr<RuntimeCheckpointModel> OpenRuntimeInitializedModel(
    const TrainingCheckpointOpenRequest& request);
}  // namespace stlqpp
