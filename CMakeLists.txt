cmake_minimum_required(VERSION 3.18)

project(STLQpp VERSION 0.1.0 LANGUAGES CXX)

include(cmake/StlqppOptions.cmake)
include(cmake/StlqppCblas.cmake)
include(cmake/StlqppOpenMp.cmake)
include(cmake/StlqppCuda.cmake)
include(cmake/StlqppLapack.cmake)

if(STLQpp_ENABLE_CUDA)
  stlqpp_detect_cuda()
else()
  message(STATUS "STLQpp CUDA runtime backend: disabled by build option")
endif()
find_package(Threads REQUIRED)

add_library(stlqpp_core STATIC
    src/core/artifact_hash.cpp
    src/core/filesystem_path.cpp
    src/core/status.cpp)
add_library(STLQpp::core ALIAS stlqpp_core)
stlqpp_configure_target(stlqpp_core)

add_library(stlqpp_platform STATIC
    src/platform/cpu_features_x86.cpp
    src/platform/npy_array.cpp)
if(WIN32)
  target_sources(stlqpp_platform PRIVATE
      src/platform/file_replication_windows.cpp
      src/platform/process_heap_windows.cpp
      src/platform/mapped_file_windows.cpp
      src/platform/positioned_file_windows.cpp
      src/platform/sequential_file_writer_windows.cpp
      src/platform/staging_directory_windows.cpp)
else()
  target_sources(stlqpp_platform PRIVATE
      src/platform/file_replication_posix.cpp
      src/platform/process_heap_posix.cpp
      src/platform/mapped_file_posix.cpp
      src/platform/positioned_file_posix.cpp
      src/platform/sequential_file_writer_posix.cpp
      src/platform/staging_directory_posix.cpp)
endif()
add_library(STLQpp::platform ALIAS stlqpp_platform)
target_link_libraries(stlqpp_platform PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_platform)

add_library(stlqpp_runtime_cpu STATIC
    src/runtime/cpu/openmp_affinity.cpp
    src/runtime/cpu/parallel_for.cpp)
if(WIN32)
  target_sources(stlqpp_runtime_cpu PRIVATE
      src/runtime/cpu/openmp_affinity_windows.cpp)
else()
  target_sources(stlqpp_runtime_cpu PRIVATE
      src/runtime/cpu/openmp_affinity_posix.cpp)
endif()
add_library(STLQpp::runtime_cpu ALIAS stlqpp_runtime_cpu)
target_link_libraries(stlqpp_runtime_cpu PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_runtime_cpu)

add_library(stlqpp_runtime_telemetry STATIC
    src/runtime/telemetry/logger.cpp)
add_library(STLQpp::runtime_telemetry ALIAS stlqpp_runtime_telemetry)
target_link_libraries(stlqpp_runtime_telemetry PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_runtime_telemetry)

add_library(stlqpp_application_support STATIC
    src/application/compute_backend.cpp
    src/application/run_session.cpp
    src/application/sha256.cpp)
add_library(STLQpp::application_support ALIAS stlqpp_application_support)
target_link_libraries(stlqpp_application_support
    PUBLIC STLQpp::runtime_telemetry STLQpp::core)
stlqpp_configure_target(stlqpp_application_support)
target_compile_definitions(stlqpp_application_support PRIVATE
    STLQPP_HAS_CUDA_RUNTIME=$<BOOL:${STLQpp_CUDA_AVAILABLE}>)
if(STLQpp_CUDA_AVAILABLE)
  target_link_libraries(stlqpp_application_support PRIVATE CUDA::cudart)
endif()

if(STLQpp_CUDA_AVAILABLE)
  add_library(stlqpp_runtime_cuda STATIC
      src/runtime/cuda/context.cpp
      src/runtime/cuda/gemm_f32.cpp
      src/runtime/cuda/host_f32_gemm.cpp
      src/runtime/cuda/status.cpp)
  add_library(STLQpp::runtime_cuda ALIAS stlqpp_runtime_cuda)
  target_link_libraries(stlqpp_runtime_cuda
      PUBLIC STLQpp::core CUDA::cudart CUDA::cublas)
  stlqpp_configure_cuda_target(stlqpp_runtime_cuda)
endif()

add_library(stlqpp_model STATIC
    src/model/codec/canonical_huffman.cpp
    src/model/codec/scalar_int6_codec.cpp
    src/model/model_space_transform.cpp
    src/model/model_shape.cpp
    src/model/product_codebooks.cpp
    src/model/runtime_codebooks.cpp
    src/model/root_codebooks.cpp
    src/model/topology/canonical_bfs_forest_order.cpp
    src/model/topology/parent_louds.cpp)
add_library(STLQpp::model ALIAS stlqpp_model)
target_link_libraries(stlqpp_model PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_model)

add_library(stlqpp_storage STATIC
    src/index/norm/scaled_score8_codebook.cpp
    src/storage/artifact_hash_factory.cpp
    src/storage/canonical_cluster.cpp
    src/storage/canonical_cluster_codec.cpp
    src/storage/canonical_directory_codec.cpp
    src/storage/canonical_index_files.cpp
    src/storage/canonical_index_writer.cpp
    src/storage/canonical_model_codec.cpp
    src/storage/canonical_shard_codec.cpp
    src/storage/norm_cache_layout.cpp
    src/storage/index_manifest.cpp
    src/storage/index_manifest_codec.cpp
    src/storage/trained_model_file.cpp)
add_library(STLQpp::storage ALIAS stlqpp_storage)
target_link_libraries(stlqpp_storage
    PUBLIC STLQpp::model STLQpp::platform STLQpp::core
    PRIVATE STLQpp::application_support)
stlqpp_configure_target(stlqpp_storage)

add_library(stlqpp_search_avx2 STATIC
    src/search/kernels/dense_avx2.cpp)
target_link_libraries(stlqpp_search_avx2
    PUBLIC STLQpp::storage STLQpp::model STLQpp::core)
stlqpp_configure_target(stlqpp_search_avx2)
if(MSVC)
  target_compile_definitions(stlqpp_search_avx2 PRIVATE _HAS_EXCEPTIONS=0)
  target_compile_options(stlqpp_search_avx2 PRIVATE /arch:AVX2 /GR- /EHs-c-)
else()
  target_compile_options(stlqpp_search_avx2 PRIVATE
      -mavx2 -mfma -fno-exceptions -fno-rtti)
endif()

add_library(stlqpp_search STATIC
    src/search/index/open_index.cpp
    src/search/index/resident_index.cpp
    src/search/kernels/runtime_dense.cpp
    src/search/layout/resident_cluster.cpp
    src/search/plans/dense_cpu_plan.cpp
    src/search/plans/dense_query_executor.cpp
    src/search/plans/exact_dense_searcher.cpp
    src/search/plans/open_exact_dense_searcher.cpp
    src/search/plans/runtime_exact_dense_searcher.cpp
    src/search/rerank/exact_l2_reranker.cpp
    src/search/query_batch.cpp
    src/search/query_tables.cpp
    src/search/routing/exact_ivf_router.cpp
    src/search/routing/ivf_hnsw_router.cpp
    src/search/search_options.cpp
    src/search/tables/query_table_builder.cpp
    src/search/topk/query_topk.cpp)
add_library(STLQpp::search ALIAS stlqpp_search)
target_link_libraries(stlqpp_search
    PUBLIC STLQpp::storage STLQpp::model STLQpp::platform
           STLQpp::runtime_cpu STLQpp::core
    PRIVATE stlqpp_search_avx2)
stlqpp_configure_target(stlqpp_search)
target_include_directories(stlqpp_search SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(stlqpp_training_io STATIC
    src/training/io/bvecs_u8_source.cpp
    src/training/io/fvecs_f32_source.cpp)
add_library(STLQpp::training_io ALIAS stlqpp_training_io)
target_link_libraries(stlqpp_training_io
    PUBLIC STLQpp::runtime_cpu STLQpp::platform STLQpp::core
    PRIVATE STLQpp::runtime_telemetry)
stlqpp_configure_target(stlqpp_training_io)

add_library(stlqpp_training_basic_avx2 STATIC
    src/training/basic/icm_candidate_selector_avx2.cpp)
target_link_libraries(stlqpp_training_basic_avx2 PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_training_basic_avx2)
if(MSVC)
  target_compile_definitions(stlqpp_training_basic_avx2 PRIVATE _HAS_EXCEPTIONS=0)
  target_compile_options(stlqpp_training_basic_avx2 PRIVATE
      /arch:AVX2 /GR- /EHs-c-)
else()
  target_compile_options(stlqpp_training_basic_avx2 PRIVATE
      -mavx2 -mno-fma -ffp-contract=off -fno-exceptions -fno-rtti)
endif()

add_library(stlqpp_training_basic STATIC
    src/training/basic/basic_stage_log.cpp
    src/training/basic/basic_projection.cpp
    src/training/basic/cpu_virtual_root_support_encoder.cpp
    src/training/basic/fixed_root_replay_encoder_pool.cpp
    src/training/basic/forced_root_global_greedy.cpp
    src/training/basic/basic_pipeline.cpp
    src/training/basic/icm_candidate_selector.cpp
    src/training/basic/icm_ils_cpu.cpp
    src/training/basic/large_root_algebra.cpp
    src/training/basic/support_least_squares.cpp
    src/training/basic/virtual_root_support_encoder_pool.cpp)
add_library(STLQpp::training_basic ALIAS stlqpp_training_basic)
target_link_libraries(stlqpp_training_basic
    PUBLIC STLQpp::training_io STLQpp::model STLQpp::runtime_cpu
           STLQpp::runtime_telemetry STLQpp::platform STLQpp::core
    PRIVATE stlqpp_training_basic_avx2 Threads::Threads)
stlqpp_configure_target(stlqpp_training_basic)

if(STLQpp_CUDA_AVAILABLE)
  add_library(stlqpp_training_basic_cuda STATIC
      src/training/basic/basic_executor_cuda.cpp
      src/training/basic/basic_rotator_cuda.cu
      src/training/basic/basic_encoder_cuda.cu
      src/training/basic/cuda_fixed_root_replay_encoder.cpp
      src/training/basic/cuda_fixed_root_support_refiner.cu
      src/training/basic/cuda_virtual_root_support_encoder.cpp)
  add_library(STLQpp::training_basic_cuda ALIAS stlqpp_training_basic_cuda)
  target_link_libraries(stlqpp_training_basic_cuda
      PUBLIC STLQpp::training_basic STLQpp::runtime_cuda CUDA::cudart CUDA::cublas)
  stlqpp_configure_cuda_target(stlqpp_training_basic_cuda)
endif()

add_library(stlqpp_training_linkage_avx2 STATIC
    src/training/linkage/candidates/exact_parent_prefilter_avx2.cpp
    src/training/linkage/rsd/rsd_candidate_selector_avx2.cpp)
target_link_libraries(stlqpp_training_linkage_avx2 PUBLIC STLQpp::core)
stlqpp_configure_target(stlqpp_training_linkage_avx2)
if(MSVC)
  target_compile_definitions(stlqpp_training_linkage_avx2 PRIVATE _HAS_EXCEPTIONS=0)
  target_compile_options(stlqpp_training_linkage_avx2 PRIVATE
      /arch:AVX2 /GR- /EHs-c-)
else()
  target_compile_options(stlqpp_training_linkage_avx2 PRIVATE
      -mavx2 -mno-fma -ffp-contract=off -fno-exceptions -fno-rtti)
endif()

add_library(stlqpp_training_linkage STATIC
    src/training/linkage/candidates/cpu_exact_knn.cpp
    src/training/linkage/candidates/cluster_knn.cpp
    src/training/linkage/candidates/cpu_exact_parent_candidates.cpp
    src/training/linkage/initialization/cpu_root_only_initialization_context.cpp
    src/training/linkage/initialization/root_only_initialization_cluster_builder.cpp
    src/training/linkage/initialization/root_only_initialization_layer_executor.cpp
    src/training/linkage/initialization/root_only_initialization_pipeline.cpp
    src/training/linkage/initialization/spherical_root_code_partitioner.cpp
    src/training/linkage/rsd/cpu_linkage_rsd_encoder.cpp
    src/training/linkage/rsd/icm_layer_orders.cpp
    src/training/linkage/rsd/linkage_codebook_algebra.cpp
    src/training/linkage/rsd/linkage_rsd_kernel.cpp
    src/training/linkage/rsd/linkage_projection.cpp
    src/training/linkage/rsd/linkage_solution_refinement.cpp
    src/training/linkage/rsd/linkage_support_least_squares.cpp
    src/training/linkage/rsd/ranked_single_deviation.cpp
    src/training/linkage/rsd/rsd_candidate_selector.cpp
    src/training/linkage/root_only/cpu_root_only_linkage_encoder.cpp
    src/training/linkage/root_only/root_only_initialization_cpu_tail.cpp
    src/training/linkage/replay/cpu_fixed_edge_replay_encoder.cpp
    src/training/linkage/replay/fixed_topology_replay_plan.cpp
    src/training/linkage/replay/fixed_topology_replay_executor.cpp
    src/training/linkage/replay/fixed_topology_replay_pipeline.cpp
    src/training/linkage/replay/period_two_topology_schedule.cpp
    src/training/linkage/scheduler/bounded_linkage_scheduler.cpp
    src/training/linkage/scheduler/cpu_linkage_worker_backend.cpp
    src/training/linkage/scheduler/direct_linkage_evaluator_pool.cpp
    src/training/linkage/topology/bad_cluster_topology_builder.cpp
    src/training/linkage/topology/causal_cpu_linkage_evaluator.cpp
    src/training/linkage/topology/causal_topology_window_executor.cpp
    src/training/linkage/topology/cluster_topology_state.cpp
    src/training/linkage/topology/cluster_topology_session.cpp
    src/training/linkage/topology/full_topology_cluster_builder.cpp
    src/training/linkage/topology/full_topology_pipeline.cpp
    src/training/linkage/topology/good_cluster_topology_builder.cpp
    src/training/linkage/topology/radial_layer_plan.cpp
    src/training/linkage/topology/topology_random_stream.cpp)
target_sources(stlqpp_training_linkage PRIVATE
    src/training/linkage/virtual/umap_virtual_root_plan.cpp
    src/training/linkage/virtual/virtual_root_encoder.cpp
)
add_library(STLQpp::training_linkage ALIAS stlqpp_training_linkage)
target_include_directories(stlqpp_training_linkage SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(stlqpp_training_linkage
    PUBLIC STLQpp::training_basic STLQpp::model STLQpp::runtime_cpu
           STLQpp::runtime_telemetry STLQpp::platform STLQpp::core
    PRIVATE stlqpp_training_linkage_avx2 Threads::Threads)
stlqpp_configure_target(stlqpp_training_linkage)

add_library(stlqpp_runtime_basic_encoder STATIC
    src/training/basic/runtime_rsd_basic.cpp
    src/training/basic/runtime_basic_executor.cpp)
target_link_libraries(stlqpp_runtime_basic_encoder PUBLIC
    STLQpp::training_basic STLQpp::training_linkage)
stlqpp_configure_target(stlqpp_runtime_basic_encoder)

if(STLQpp_CUDA_AVAILABLE)
  add_library(stlqpp_training_linkage_cuda STATIC
      src/training/linkage/initialization/cuda_root_only_initialization_context.cpp
      src/training/linkage/rsd/linkage_rsd_cuda.cu
      src/training/linkage/replay/cuda_fixed_edge_replay_encoder.cpp
      src/training/linkage/replay/cuda_fixed_edge_replay_kernels.cu
      src/training/linkage/root_only/root_only_linkage_device_plan.cu
      src/training/linkage/scheduler/cuda_linkage_scheduler_factory.cu
      src/training/linkage/scheduler/cuda_linkage_worker_backend.cu)
  add_library(STLQpp::training_linkage_cuda ALIAS
      stlqpp_training_linkage_cuda)
  target_link_libraries(stlqpp_training_linkage_cuda
      PUBLIC STLQpp::training_linkage STLQpp::training_basic_cuda
             STLQpp::runtime_cuda
             CUDA::cudart CUDA::cublas)
  stlqpp_configure_cuda_target(stlqpp_training_linkage_cuda)
endif()

add_library(stlqpp_training_update STATIC
    src/training/update/edge_codebook_moments.cpp
    src/training/update/frozen_forest_reconstruction.cpp
    src/training/update/root_codebook_moments.cpp
    src/training/update/root_only_initialization_transcript.cpp
    src/training/update/round_update_transcript.cpp
    src/training/update/training_model.cpp
    src/training/update/root_training_model.cpp
    src/training/update/rotation_update.cpp)
add_library(STLQpp::training_update ALIAS stlqpp_training_update)
target_link_libraries(stlqpp_training_update
    PUBLIC STLQpp::training_linkage STLQpp::model STLQpp::core
    PRIVATE STLQpp::platform)
stlqpp_configure_target(stlqpp_training_update)

add_library(stlqpp_training_initialization STATIC
    src/training/initialization/cpu_spherical_rvq_backend.cpp
    src/training/initialization/root_model_initializer.cpp
    src/training/initialization/rvq_assignment_store.cpp
    src/training/initialization/spherical_hierarchical_layout.cpp
    src/training/initialization/spherical_kmeans_plus_plus.cpp
    src/training/initialization/spherical_lloyd_code_emission.cpp
    src/training/initialization/spherical_rvq_log.cpp
    src/training/initialization/spherical_rvq_residual.cpp)
add_library(STLQpp::training_initialization ALIAS
    stlqpp_training_initialization)
target_link_libraries(stlqpp_training_initialization
    PUBLIC STLQpp::training_io STLQpp::training_update
           STLQpp::runtime_cpu STLQpp::core
    PRIVATE STLQpp::runtime_telemetry)
stlqpp_configure_target(stlqpp_training_initialization)

if(STLQpp_CUDA_AVAILABLE)
  add_library(stlqpp_training_initialization_cuda STATIC
      src/training/initialization/cuda_hierarchical_spherical_scorer.cu
      src/training/initialization/cuda_kmeans_parallel_seeder.cu
      src/training/initialization/cuda_normalized_source_cache.cu
      src/training/initialization/cuda_rvq_assignment_publisher.cu
      src/training/initialization/cuda_spherical_rvq_backend.cpp
      src/training/initialization/cuda_streaming_spherical_lloyd.cpp
      src/training/initialization/cuda_spherical_accumulator.cu
      src/training/initialization/cuda_spherical_assignment.cu)
  add_library(STLQpp::training_initialization_cuda ALIAS
      stlqpp_training_initialization_cuda)
  target_link_libraries(stlqpp_training_initialization_cuda
      PUBLIC STLQpp::training_initialization STLQpp::runtime_cuda
             CUDA::cudart CUDA::cublas)
  stlqpp_configure_cuda_target(stlqpp_training_initialization_cuda)
endif()

add_library(stlqpp_training_checkpoint STATIC
    src/training/checkpoint/checkpoint_path.cpp
    src/training/checkpoint/checkpoint_replay_loader.cpp
    src/training/checkpoint/checkpoint_replay_sink.cpp
    src/training/checkpoint/initialized_training_checkpoint.cpp
    src/training/checkpoint/training_checkpoint.cpp
    src/training/checkpoint/training_forest_codec.cpp)
add_library(STLQpp::training_checkpoint ALIAS stlqpp_training_checkpoint)
target_link_libraries(stlqpp_training_checkpoint
    PUBLIC STLQpp::training_update STLQpp::storage STLQpp::platform
           STLQpp::model STLQpp::core)
stlqpp_configure_target(stlqpp_training_checkpoint)

add_library(stlqpp_index_norm_runtime STATIC
    src/index/norm/runtime_reconstruction_norms.cpp)
target_link_libraries(stlqpp_index_norm_runtime PUBLIC STLQpp::model STLQpp::core)
stlqpp_configure_target(stlqpp_index_norm_runtime)

add_library(stlqpp_index_build STATIC
    src/index/build/canonical_float_cluster_assembler.cpp
    src/index/build/final_cluster_postprocessor.cpp
    src/index/build/final_index_publisher.cpp
    src/index/build/prepared_cluster_spool.cpp
    src/index/codec/selected_atom_gram.cpp
    src/index/codec/scalar_int6_encoder.cpp
    src/index/emit/base_list_emitter.cpp
    src/index/emit/base_list_reader.cpp
    src/index/emit/basic_bucket_store.cpp
    src/index/norm/reconstruction_norm8.cpp
    src/index/norm/reconstruction_norms.cpp
    src/training/analysis/initialization_post_edge_analysis.cpp
    src/training/analysis/round_linkage_analysis.cpp
    src/training/analysis/causal_covariance_factors.cpp)
add_library(STLQpp::index_build ALIAS stlqpp_index_build)
target_link_libraries(stlqpp_index_build
    PUBLIC STLQpp::storage STLQpp::model STLQpp::platform
           STLQpp::runtime_cpu STLQpp::runtime_telemetry STLQpp::core
           stlqpp_index_norm_runtime)
stlqpp_configure_target(stlqpp_index_build)
if(STLQpp_CUDA_AVAILABLE)
  target_sources(stlqpp_index_build PRIVATE
      src/index/norm/reconstruction_norm8_cuda_fit.cpp)
  add_library(stlqpp_index_norm_cuda STATIC
      src/index/norm/reconstruction_norm8_cuda.cu
      src/index/norm/reconstruction_norms_cuda.cu)
  target_link_libraries(stlqpp_index_norm_cuda
      PUBLIC stlqpp_index_norm_runtime STLQpp::model STLQpp::core
             STLQpp::runtime_cuda CUDA::cudart)
  stlqpp_configure_cuda_target(stlqpp_index_norm_cuda)
  target_link_libraries(stlqpp_index_build
      PUBLIC stlqpp_index_norm_cuda)
  target_compile_definitions(stlqpp_index_build PRIVATE
      STLQPP_HAS_CUDA_NORM=1)
else()
  target_compile_definitions(stlqpp_index_build PRIVATE
      STLQPP_HAS_CUDA_NORM=0)
endif()

add_library(stlqpp_build_pipeline STATIC
    src/pipeline/base_list_full_topology_source.cpp
    src/pipeline/base_list_replay_source.cpp
    src/pipeline/stable_id_replay_source.cpp
    src/pipeline/final_full_topology_index_builder.cpp
    src/pipeline/final_forest_spool_writer.cpp
    src/pipeline/final_index_build_session.cpp
    src/pipeline/final_replay_index_builder.cpp
    src/pipeline/final_replay_spool_sink.cpp
    src/pipeline/full_topology_spool_sink.cpp
    src/pipeline/base_basic_stage.cpp)
target_sources(stlqpp_build_pipeline PRIVATE
    src/pipeline/root_only_initialization_moment_replay.cpp
    src/pipeline/training_list_root_initialization_source.cpp)
add_library(STLQpp::build_pipeline ALIAS stlqpp_build_pipeline)
target_link_libraries(stlqpp_build_pipeline
    PUBLIC STLQpp::index_build STLQpp::training_checkpoint STLQpp::model
           STLQpp::core stlqpp_runtime_basic_encoder)
stlqpp_configure_target(stlqpp_build_pipeline)

stlqpp_configure_cblas()
stlqpp_configure_openmp()
stlqpp_configure_lapack()

if(STLQpp_CBLAS_AVAILABLE)
  target_link_libraries(stlqpp_index_build PUBLIC STLQpp::runtime_cblas)
  target_compile_definitions(stlqpp_index_build PRIVATE STLQPP_HAS_CBLAS_GRAM_NORM=1)
else()
  target_compile_definitions(stlqpp_index_build PRIVATE STLQPP_HAS_CBLAS_GRAM_NORM=0)
endif()

if(STLQpp_CUDA_AVAILABLE AND STLQpp_CBLAS_AVAILABLE)
  target_link_libraries(stlqpp_training_linkage_cuda
      PUBLIC STLQpp::runtime_cblas)
endif()

if(STLQpp_OPENMP_AVAILABLE)
  target_link_libraries(stlqpp_index_build
      PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(stlqpp_training_basic
      PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(stlqpp_training_linkage
      PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(stlqpp_training_initialization
      PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(stlqpp_build_pipeline
      PRIVATE OpenMP::OpenMP_CXX)
endif()

if(STLQpp_LAPACK_AVAILABLE)
  target_sources(stlqpp_training_update PRIVATE
      src/training/update/edge_codebook_solver.cpp
      src/training/update/initialization_model_updater.cpp
      src/training/update/root_only_initialization_update_sink.cpp
      src/training/update/replay_round_update_sink.cpp
      src/training/update/round_model_assembler.cpp
      src/training/update/round_model_updater.cpp
      src/training/update/root_codebook_solver.cpp)
  target_link_libraries(stlqpp_training_update
      PUBLIC STLQpp::runtime_lapack)
  target_sources(stlqpp_training_checkpoint PRIVATE
      src/training/checkpoint/full_topology_checkpoint_round.cpp
      src/training/checkpoint/full_topology_checkpoint_sink.cpp
      src/training/checkpoint/replay_checkpoint_round.cpp)
  target_sources(stlqpp_build_pipeline PRIVATE
      src/pipeline/training_initializer.cpp
      src/pipeline/root_initialization_basic_stage.cpp
      src/pipeline/training_basic_pass.cpp)

  add_library(stlqpp_training STATIC
      src/training/trainer.cpp
      src/pipeline/round_rotation_replay.cpp)
  add_library(STLQpp::training ALIAS stlqpp_training)
  target_link_libraries(stlqpp_training
      PUBLIC STLQpp::training_checkpoint STLQpp::training_update
             STLQpp::training_linkage STLQpp::training_basic
             STLQpp::runtime_telemetry STLQpp::core
      PRIVATE STLQpp::index_build)
  stlqpp_configure_target(stlqpp_training)

  add_library(stlqpp_final_index_log STATIC
      src/application/final_index_log.cpp)
  target_link_libraries(stlqpp_final_index_log
      PUBLIC STLQpp::storage STLQpp::runtime_telemetry)
  stlqpp_configure_target(stlqpp_final_index_log)

  add_library(stlqpp_application STATIC
      src/application/index_builder.cpp
      src/application/index_republisher.cpp
      src/application/registered_artifact_recipes.cpp)
  add_library(STLQpp::application ALIAS stlqpp_application)
  target_link_libraries(stlqpp_application
      PUBLIC stlqpp_final_index_log STLQpp::build_pipeline STLQpp::training
             STLQpp::training_initialization STLQpp::runtime_telemetry)
  stlqpp_configure_target(stlqpp_application)
  if(STLQpp_CUDA_AVAILABLE)
    target_compile_definitions(stlqpp_application PRIVATE
        STLQPP_HAS_CUDA_NORM=1)
  else()
    target_compile_definitions(stlqpp_application PRIVATE
        STLQPP_HAS_CUDA_NORM=0)
  endif()

  if(STLQpp_CBLAS_AVAILABLE AND STLQpp_OPENMP_AVAILABLE)
    add_library(stlqpp_application_cpu STATIC
        src/application/cpu_index_build_runtime.cpp
        src/application/rsd_basic_round_runtime_cpu.cpp
        src/pipeline/cpu_training_topology_context_factory.cpp)
    add_library(STLQpp::application_cpu ALIAS stlqpp_application_cpu)
    target_link_libraries(stlqpp_application_cpu
        PUBLIC STLQpp::application STLQpp::runtime_cblas
               STLQpp::runtime_lapack STLQpp::runtime_openmp)
    stlqpp_configure_target(stlqpp_application_cpu)
  endif()

  if(STLQpp_CUDA_AVAILABLE)
    add_library(stlqpp_training_runtime_cuda STATIC
        src/pipeline/cuda_training_topology_context_factory.cpp)
    add_library(STLQpp::training_runtime_cuda ALIAS
        stlqpp_training_runtime_cuda)
    target_link_libraries(stlqpp_training_runtime_cuda
        PUBLIC STLQpp::application
               STLQpp::training_linkage_cuda)
    stlqpp_configure_target(stlqpp_training_runtime_cuda)
  endif()

  if(STLQpp_CUDA_AVAILABLE AND STLQpp_CBLAS_AVAILABLE AND
     STLQpp_OPENMP_AVAILABLE)
    add_library(stlqpp_application_cuda STATIC
        src/application/cuda_index_build_runtime.cpp
        src/application/rsd_basic_round_runtime.cpp)
    add_library(STLQpp::application_cuda ALIAS stlqpp_application_cuda)
    target_link_libraries(stlqpp_application_cuda
        PUBLIC STLQpp::application STLQpp::application_cpu STLQpp::training_initialization_cuda
               STLQpp::training_runtime_cuda STLQpp::training_basic_cuda
               STLQpp::runtime_cblas STLQpp::runtime_lapack
               STLQpp::runtime_openmp)
    stlqpp_configure_target(stlqpp_application_cuda)

    add_library(stlqpp_effective_build_recipe STATIC
        applications/build/effective_build_recipe.cpp)
    target_include_directories(stlqpp_effective_build_recipe
        PUBLIC applications/build)
    target_link_libraries(stlqpp_effective_build_recipe
        PUBLIC STLQpp::application stlqpp_build_command)
    stlqpp_configure_target(stlqpp_effective_build_recipe)

    add_library(stlqpp_build_control_manifest STATIC
        applications/build/build_control_manifest.cpp)
    target_include_directories(stlqpp_build_control_manifest
        PUBLIC applications/build)
    target_link_libraries(stlqpp_build_control_manifest
        PUBLIC stlqpp_effective_build_recipe stlqpp_build_command)
    stlqpp_configure_target(stlqpp_build_control_manifest)

    add_library(stlqpp_index_build_runner STATIC
        applications/build/run_index_build.cpp
        applications/build/build_output_preparation.cpp)
    target_include_directories(stlqpp_index_build_runner
        PUBLIC applications/build)
    target_link_libraries(stlqpp_index_build_runner
        PUBLIC STLQpp::application_cuda STLQpp::application_cpu
               stlqpp_build_command
               stlqpp_effective_build_recipe)
    target_compile_definitions(stlqpp_index_build_runner
        PUBLIC STLQPP_HAS_CUDA_RUNTIME=1)
    stlqpp_configure_target(stlqpp_index_build_runner)

    add_library(stlqpp_build_completion_log STATIC
        applications/build/build_completion_log.cpp
        applications/build/control_evidence.cpp)
    target_include_directories(stlqpp_build_completion_log
        PUBLIC applications/build)
    target_link_libraries(stlqpp_build_completion_log
        PUBLIC STLQpp::application STLQpp::application_support
               stlqpp_build_command stlqpp_effective_build_recipe
               stlqpp_build_control_manifest)
    stlqpp_configure_target(stlqpp_build_completion_log)

    add_executable(stlqpp_build_index
        applications/build/build_index.cpp)
    target_link_libraries(stlqpp_build_index
        PRIVATE stlqpp_index_build_runner stlqpp_build_completion_log
                STLQpp::runtime_telemetry)
    stlqpp_configure_target(stlqpp_build_index)
    target_compile_definitions(stlqpp_build_index PRIVATE
        STLQPP_BUILD_CONFIGURATION="$<CONFIG>"
        STLQPP_BUILD_LTO_ENABLED=$<BOOL:${STLQpp_ENABLE_LTO}>
        STLQPP_BUILD_SOURCE_IDENTITY="${STLQpp_SOURCE_IDENTITY}"
        STLQPP_BUILD_SOURCE_STATE="${STLQpp_SOURCE_STATE}")

  elseif(STLQpp_CBLAS_AVAILABLE AND STLQpp_OPENMP_AVAILABLE)
    add_library(stlqpp_effective_build_recipe STATIC
        applications/build/effective_build_recipe.cpp)
    target_include_directories(stlqpp_effective_build_recipe
        PUBLIC applications/build)
    target_link_libraries(stlqpp_effective_build_recipe
        PUBLIC STLQpp::application stlqpp_build_command)
    stlqpp_configure_target(stlqpp_effective_build_recipe)

    add_library(stlqpp_build_control_manifest STATIC
        applications/build/build_control_manifest.cpp)
    target_include_directories(stlqpp_build_control_manifest
        PUBLIC applications/build)
    target_link_libraries(stlqpp_build_control_manifest
        PUBLIC stlqpp_effective_build_recipe stlqpp_build_command)
    stlqpp_configure_target(stlqpp_build_control_manifest)

    add_library(stlqpp_index_build_runner STATIC
        applications/build/run_index_build.cpp
        applications/build/build_output_preparation.cpp)
    target_include_directories(stlqpp_index_build_runner
        PUBLIC applications/build)
    target_link_libraries(stlqpp_index_build_runner
        PUBLIC STLQpp::application_cpu stlqpp_build_command
               stlqpp_effective_build_recipe)
    target_compile_definitions(stlqpp_index_build_runner
        PUBLIC STLQPP_HAS_CUDA_RUNTIME=0)
    stlqpp_configure_target(stlqpp_index_build_runner)

    add_library(stlqpp_build_completion_log STATIC
        applications/build/build_completion_log.cpp
        applications/build/control_evidence.cpp)
    target_include_directories(stlqpp_build_completion_log
        PUBLIC applications/build)
    target_link_libraries(stlqpp_build_completion_log
        PUBLIC STLQpp::application STLQpp::application_support
               stlqpp_build_command stlqpp_effective_build_recipe
               stlqpp_build_control_manifest)
    stlqpp_configure_target(stlqpp_build_completion_log)

    add_executable(stlqpp_build_index
        applications/build/build_index.cpp)
    target_link_libraries(stlqpp_build_index
        PRIVATE stlqpp_index_build_runner stlqpp_build_completion_log
                STLQpp::runtime_telemetry)
    stlqpp_configure_target(stlqpp_build_index)
    target_compile_definitions(stlqpp_build_index PRIVATE
        STLQPP_BUILD_CONFIGURATION="$<CONFIG>"
        STLQPP_BUILD_LTO_ENABLED=$<BOOL:${STLQpp_ENABLE_LTO}>
        STLQPP_BUILD_SOURCE_IDENTITY="${STLQpp_SOURCE_IDENTITY}"
        STLQPP_BUILD_SOURCE_STATE="${STLQpp_SOURCE_STATE}")
  endif()
endif()

add_library(stlqpp_app_io STATIC
    applications/common/texmex_reader.cpp)
target_include_directories(stlqpp_app_io
    PUBLIC applications/common)
target_link_libraries(stlqpp_app_io
    PUBLIC STLQpp::platform STLQpp::core)
stlqpp_configure_target(stlqpp_app_io)

add_library(stlqpp_eval STATIC src/eval/eval_metrics.cpp)
add_library(STLQpp::eval ALIAS stlqpp_eval)
target_link_libraries(stlqpp_eval PUBLIC STLQpp::search STLQpp::core)
stlqpp_configure_target(stlqpp_eval)

add_library(stlqpp_build_command STATIC
    applications/build/build_command.cpp)
target_include_directories(stlqpp_build_command
    PUBLIC applications/build)
target_link_libraries(stlqpp_build_command
    PUBLIC STLQpp::core STLQpp::application_support PRIVATE STLQpp::storage)
stlqpp_configure_target(stlqpp_build_command)

if(STLQpp_CBLAS_AVAILABLE AND STLQpp_OPENMP_AVAILABLE)

  add_library(stlqpp_eval_application STATIC
      applications/search/eval_command.cpp
      applications/search/eval_archive.cpp)
  target_include_directories(stlqpp_eval_application
      PUBLIC applications/search)
  target_link_libraries(stlqpp_eval_application
      PUBLIC STLQpp::eval STLQpp::application_support STLQpp::core
             STLQpp::platform STLQpp::storage)
  stlqpp_configure_target(stlqpp_eval_application)

  add_library(stlqpp_exact_dense_eval_runner STATIC
      applications/search/evaluate_exact_dense.cpp
      applications/search/evaluate_base_basic.cpp)
  target_include_directories(stlqpp_exact_dense_eval_runner
      PUBLIC applications/search applications/common)
  target_link_libraries(stlqpp_exact_dense_eval_runner
      PUBLIC stlqpp_eval_application STLQpp::search
             STLQpp::runtime_cblas STLQpp::runtime_openmp stlqpp_app_io)
  target_link_libraries(stlqpp_exact_dense_eval_runner
      PRIVATE stlqpp_final_index_log STLQpp::index_build
              STLQpp::application STLQpp::runtime_lapack
              stlqpp_scaled_score8_derivation)
  stlqpp_configure_target(stlqpp_exact_dense_eval_runner)

  if(TARGET stlqpp_build_index)
    target_link_libraries(stlqpp_build_index
        PRIVATE stlqpp_exact_dense_eval_runner)
  endif()
endif()

add_library(stlqpp_source_norm_provider STATIC
    src/application/source_norm_provider.cpp)
target_link_libraries(stlqpp_source_norm_provider
    PUBLIC STLQpp::training_io STLQpp::core
    PRIVATE STLQpp::storage STLQpp::application_support)
stlqpp_configure_target(stlqpp_source_norm_provider)
if(TARGET stlqpp_index_build_runner)
  target_link_libraries(stlqpp_index_build_runner
      PRIVATE stlqpp_source_norm_provider)
endif()
if(TARGET stlqpp_application)
  target_link_libraries(stlqpp_application
      PUBLIC stlqpp_source_norm_provider)
endif()

if(WIN32 AND TARGET stlqpp_build_index)
  # Hashed cache generations can exceed the legacy MAX_PATH limit.
  target_sources(stlqpp_build_index PRIVATE applications/stlqpp.manifest)
endif()

add_library(stlqpp_scaled_score8_derivation STATIC
    src/application/scaled_score8_derivation.cpp
    src/application/eval_norm_cache.cpp
    src/application/query_disk_footprint.cpp)
target_link_libraries(stlqpp_scaled_score8_derivation
    PUBLIC STLQpp::index_build STLQpp::storage
           stlqpp_source_norm_provider STLQpp::application_support)
stlqpp_configure_target(stlqpp_scaled_score8_derivation)
if(STLQpp_CUDA_AVAILABLE)
  target_compile_definitions(stlqpp_scaled_score8_derivation PRIVATE
      STLQPP_HAS_CUDA_NORM=1)
else()
  target_compile_definitions(stlqpp_scaled_score8_derivation PRIVATE
      STLQPP_HAS_CUDA_NORM=0)
endif()
