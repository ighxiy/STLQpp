#pragma once

#include "stlqpp/training/initialization/root_model_initializer.h"
#include "stlqpp/training/linkage/scheduler/linkage_resource_statistics.h"
#include "stlqpp/training/linkage/topology/full_topology_pipeline.h"
#include "stlqpp/training/trainer.h"
#include "stlqpp/core/artifact_hash.h"

#include <filesystem>
#include <optional>
#include <vector>

namespace stlqpp::app {

// Non-template observation of a completed product build.  It contains no
// model payload and cannot affect construction. Production logging and
// benchmark CSV consume this value through separate applications.
struct BuildExecutionReport final {
  struct WallTimes final {
    double root_model = 0.0;
    double initialization = 0.0;
    double training = 0.0;
    double base_basic = 0.0;
    double final_linkage_publication = 0.0;
    double total = 0.0;
  } wall_times;

  std::vector<SphericalRvqLayerStatistics> root_layers;
  bool fast_initialization = false;
  std::optional<LinkageResourceStatistics> initialization_resources;
  std::vector<TrainingRoundStatistics> training_rounds;
  FullTopologyPipelineStatistics final_topology;
  std::optional<LinkageResourceStatistics> final_resources;
  std::filesystem::path trained_model_artifact;
  std::filesystem::path index_directory;
  std::filesystem::path base_list_directory;
  bool owns_base_list = false;
  std::filesystem::path run_root;
  std::filesystem::path work_root;
  ArtifactHash64 base_basic_hash;
  ArtifactHash64 base_list_hash;
  ArtifactHash64 linkage_hash;
};

}  // namespace stlqpp::app
