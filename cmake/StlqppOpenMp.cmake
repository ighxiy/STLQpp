include_guard(GLOBAL)

set(STLQpp_OPENMP_AVAILABLE OFF)

function(stlqpp_configure_openmp)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(STATUS "STLQpp OpenMP runtime backend: unavailable")
    set(STLQpp_OPENMP_AVAILABLE OFF PARENT_SCOPE)
    return()
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
