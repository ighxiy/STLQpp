include_guard(GLOBAL)

set(STLQpp_CBLAS_BACKEND "AUTO" CACHE STRING
    "CBLAS backend: AUTO, MKL, OPENBLAS, or NONE")
set_property(CACHE STLQpp_CBLAS_BACKEND PROPERTY STRINGS
    AUTO MKL OPENBLAS NONE)
set(STLQpp_MKL_ROOT "" CACHE PATH "Intel oneMKL root")
set(STLQpp_OPENBLAS_ROOT "" CACHE PATH "OpenBLAS root")

function(stlqpp_configure_cblas)
  string(TOUPPER "${STLQpp_CBLAS_BACKEND}" backend)
  if(backend STREQUAL "NONE")
    message(STATUS "STLQpp CBLAS runtime backend: disabled")
    return()
  endif()

  set(try_mkl FALSE)
  set(try_openblas FALSE)
  if(backend STREQUAL "AUTO")
    set(try_mkl TRUE)
    set(try_openblas TRUE)
  elseif(backend STREQUAL "MKL")
    set(try_mkl TRUE)
  elseif(backend STREQUAL "OPENBLAS")
    set(try_openblas TRUE)
  else()
    message(FATAL_ERROR
        "STLQpp_CBLAS_BACKEND must be AUTO, MKL, OPENBLAS, or NONE")
  endif()

  if(try_mkl)
    set(mkl_hints "${STLQpp_MKL_ROOT}")
    if(DEFINED ENV{MKLROOT})
      list(APPEND mkl_hints "$ENV{MKLROOT}")
    endif()
    if(WIN32)
      list(APPEND mkl_hints
          "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
    else()
      list(APPEND mkl_hints "/opt/intel/oneapi/mkl/latest")
    endif()
    find_path(STLQpp_MKL_INCLUDE_DIR mkl.h
        HINTS ${mkl_hints} PATH_SUFFIXES include)
    find_library(STLQpp_MKL_RUNTIME mkl_rt
        HINTS ${mkl_hints} PATH_SUFFIXES lib lib/intel64 latest/lib/intel64)
    if(STLQpp_MKL_INCLUDE_DIR AND STLQpp_MKL_RUNTIME)
      add_library(stlqpp_runtime_cblas STATIC
          src/runtime/cpu/blas_thread_policy.cpp
          src/runtime/cpu/cblas_f32_cross_covariance.cpp
          src/runtime/cpu/cblas_f32_gemm.cpp)
      add_library(STLQpp::runtime_cblas ALIAS stlqpp_runtime_cblas)
      target_compile_definitions(stlqpp_runtime_cblas PRIVATE
          STLQPP_CBLAS_MKL=1)
      target_include_directories(stlqpp_runtime_cblas PRIVATE
          "${STLQpp_MKL_INCLUDE_DIR}")
      target_link_libraries(stlqpp_runtime_cblas
          PUBLIC STLQpp::core
          PRIVATE "${STLQpp_MKL_RUNTIME}" Threads::Threads ${CMAKE_DL_LIBS})
      stlqpp_configure_target(stlqpp_runtime_cblas)
      set(STLQpp_CBLAS_AVAILABLE TRUE PARENT_SCOPE)
      set(STLQpp_CBLAS_PROVIDER MKL PARENT_SCOPE)
      message(STATUS
          "STLQpp CBLAS runtime backend: MKL (${STLQpp_MKL_RUNTIME})")
      return()
    endif()
    if(backend STREQUAL "MKL")
      message(FATAL_ERROR
          "MKL requested but mkl.h or mkl_rt was not found")
    endif()
  endif()

  if(try_openblas)
    find_path(STLQpp_OPENBLAS_INCLUDE_DIR cblas.h
        HINTS "${STLQpp_OPENBLAS_ROOT}" PATH_SUFFIXES include)
    find_library(STLQpp_OPENBLAS_LIBRARY openblas
        HINTS "${STLQpp_OPENBLAS_ROOT}" PATH_SUFFIXES lib)
    if(STLQpp_OPENBLAS_INCLUDE_DIR AND STLQpp_OPENBLAS_LIBRARY)
      add_library(stlqpp_runtime_cblas STATIC
          src/runtime/cpu/blas_thread_policy.cpp
          src/runtime/cpu/cblas_f32_cross_covariance.cpp
          src/runtime/cpu/cblas_f32_gemm.cpp)
      add_library(STLQpp::runtime_cblas ALIAS stlqpp_runtime_cblas)
      target_include_directories(stlqpp_runtime_cblas PRIVATE
          "${STLQpp_OPENBLAS_INCLUDE_DIR}")
      target_compile_definitions(stlqpp_runtime_cblas PRIVATE
          STLQPP_CBLAS_OPENBLAS=1)
      target_link_libraries(stlqpp_runtime_cblas
          PUBLIC STLQpp::core
          PRIVATE "${STLQpp_OPENBLAS_LIBRARY}" Threads::Threads)
      stlqpp_configure_target(stlqpp_runtime_cblas)
      set(STLQpp_CBLAS_AVAILABLE TRUE PARENT_SCOPE)
      set(STLQpp_CBLAS_PROVIDER OPENBLAS PARENT_SCOPE)
      message(STATUS
          "STLQpp CBLAS runtime backend: OpenBLAS (${STLQpp_OPENBLAS_LIBRARY})")
      return()
    endif()
    if(backend STREQUAL "OPENBLAS")
      message(FATAL_ERROR
          "OpenBLAS requested but cblas.h or libopenblas was not found")
    endif()
  endif()

  message(STATUS
      "STLQpp CBLAS runtime backend: unavailable (search contracts still build)")
endfunction()
