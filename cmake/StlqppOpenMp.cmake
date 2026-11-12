include_guard(GLOBAL)

set(STLQpp_OPENMP_AVAILABLE OFF)

function(stlqpp_configure_openmp)
  # The numerical kernels use omp simd, which plain MSVC /openmp rejects.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT DEFINED OpenMP_CXX_FLAGS)
    set(OpenMP_CXX_FLAGS "/openmp:experimental")
  endif()
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(STATUS "STLQpp OpenMP runtime backend: unavailable")
    set(STLQpp_OPENMP_AVAILABLE OFF PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC accepts but ignores some SIMD clauses (including reduction).
    # The loops retain their scalar semantics; keep all other /W4 checks.
    target_compile_options(OpenMP::OpenMP_CXX INTERFACE /wd4849)
  endif()

  add_library(stlqpp_runtime_openmp STATIC
      src/runtime/cpu/openmp_parallel_for.cpp)
  add_library(STLQpp::runtime_openmp ALIAS stlqpp_runtime_openmp)
  target_link_libraries(stlqpp_runtime_openmp
      PUBLIC STLQpp::runtime_cpu OpenMP::OpenMP_CXX)
  stlqpp_configure_target(stlqpp_runtime_openmp)
  set(STLQpp_OPENMP_AVAILABLE ON PARENT_SCOPE)
  message(STATUS "STLQpp OpenMP runtime backend: enabled")
endfunction()
