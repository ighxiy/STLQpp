#pragma once

#include "stlqpp/storage/artifact_hash_factory.h"

namespace stlqpp {

// Read-only lookup of v29 models before restoring per-pair CPU tiny ICM.
// Never use this identity to publish or resume training.
[[nodiscard]] ArtifactHash64 SharedCpuTailIcmTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept;

// Read-only lookup of v27 models before positive-pivot CPU refit unification.
// Existing trained codebooks remain usable; never use this for resume.
[[nodiscard]] ArtifactHash64 PrePositivePivotTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept;

// Read-only lookup of v26 models before CPU tiny-tail RSD unification.
// Existing trained codebooks remain usable; never use this for resume.
[[nodiscard]] ArtifactHash64 PreCpuTailRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept;

// Read-only lookup of v25 published models, before the GOOD-parent fix.
// Never use this identity to publish or resume training.
[[nodiscard]] ArtifactHash64 PreviousRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept;

// Read compatibility for v24 RSD models whose initialization used the archived
// basic initializer. Never use this identity to publish or resume training.
[[nodiscard]] ArtifactHash64 LegacyRsdTrainingModelIdentity(
    const TrainResumeHashInput& input) noexcept;

}  // namespace stlqpp
