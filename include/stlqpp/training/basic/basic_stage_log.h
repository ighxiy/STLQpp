#pragma once

#include "stlqpp/model/model_shape.h"
#include "stlqpp/training/basic/basic_schedule.h"
#include "stlqpp/training/basic/basic_pipeline.h"
#include "stlqpp/training/basic/basic_round_runtime.h"

#include <cstdint>

namespace stlqpp {

struct BasicStageLogContext final {
  ModelShape shape;
  bool source_is_f32 = false;
  std::uint64_t source_rows = 0U;
  std::uint64_t source_block_rows = 0U;
  std::uint32_t cpu_threads = 0U;
  bool fixed_roots = false;
  BasicRuntimeExecutionInfo execution;
  BasicSchedule schedule = BasicSchedule::Training();
  std::uint32_t seed = 0U;
  BasicPipelineOptions pipeline;
};

// DEV-compatible stage renderer. It consumes only effective execution values
// and producer statistics; it never affects scheduling, search or artifacts.
void LogBaseListStoreCompletion(double wall_seconds);
void LogLargeScaleBaseEncodingCompletion(double wall_seconds);
void LogTrainingRoundPreparation(double precompute_root_seconds,
                                 double basic_open_seconds,
                                 double total_seconds);
void LogBasicStageCompletion(
    const BasicStageLogContext& context,
    const BasicPipelineStatistics& statistics);
void LogBasicExecution(const BasicRuntimeExecutionInfo& execution,
                       BasicSchedule schedule);

}  // namespace stlqpp
