include_guard(GLOBAL)

set(STLQpp_LAPACK_BACKEND "AUTO" CACHE STRING
    "LAPACKE backend for training: AUTO, MKL, OPENBLAS, or NONE")
set_property(CACHE STLQpp_LAPACK_BACKEND PROPERTY STRINGS
    AUTO MKL OPENBLAS NONE)

function(stlqpp_configure_lapack)
  string(TOUPPER "${STLQpp_LAPACK_BACKEND}" backend)
  if(backend STREQUAL "NONE")
    message(STATUS "STLQpp LAPACKE training backend: disabled")
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
        "STLQpp_LAPACK_BACKEND must be AUTO, MKL, OPENBLAS, or NONE")
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
    find_path(STLQpp_LAPACK_MKL_INCLUDE_DIR mkl.h
        HINTS ${mkl_hints} PATH_SUFFIXES include)
    find_library(STLQpp_LAPACK_MKL_RUNTIME mkl_rt
        HINTS ${mkl_hints} PATH_SUFFIXES lib lib/intel64 latest/lib/intel64)
    if(STLQpp_LAPACK_MKL_INCLUDE_DIR AND STLQpp_LAPACK_MKL_RUNTIME)
      add_library(stlqpp_runtime_lapack STATIC
          src/runtime/cpu/lapack_square_svd_f32.cpp
          src/runtime/cpu/lapack_spd_solver_f32.cpp
          src/runtime/cpu/lapack_spd_solver_f64.cpp)
      add_library(STLQpp::runtime_lapack ALIAS stlqpp_runtime_lapack)
      target_compile_definitions(stlqpp_runtime_lapack PRIVATE
          STLQPP_LAPACKE_MKL=1)
      target_include_directories(stlqpp_runtime_lapack PRIVATE
          "${STLQpp_LAPACK_MKL_INCLUDE_DIR}")
      target_link_libraries(stlqpp_runtime_lapack
          PUBLIC STLQpp::core
          PRIVATE "${STLQpp_LAPACK_MKL_RUNTIME}" ${CMAKE_DL_LIBS})
      stlqpp_configure_target(stlqpp_runtime_lapack)
      set(STLQpp_LAPACK_AVAILABLE TRUE PARENT_SCOPE)
      message(STATUS
          "STLQpp LAPACKE training backend: MKL (${STLQpp_LAPACK_MKL_RUNTIME})")
      return()
    endif()
    if(backend STREQUAL "MKL")
      message(FATAL_ERROR "MKL LAPACKE requested but unavailable")
    endif()
  endif()

  if(try_openblas)
    find_path(STLQpp_LAPACKE_INCLUDE_DIR lapacke.h
        HINTS "${STLQpp_OPENBLAS_ROOT}" PATH_SUFFIXES include)
    find_library(STLQpp_LAPACKE_LIBRARY lapacke
        HINTS "${STLQpp_OPENBLAS_ROOT}" PATH_SUFFIXES lib)
    find_library(STLQpp_LAPACK_OPENBLAS_LIBRARY openblas
        HINTS "${STLQpp_OPENBLAS_ROOT}" PATH_SUFFIXES lib)
    # Windows OpenBLAS distributions commonly export LAPACKE from openblas
    # itself instead of shipping a separate lapacke import library.
    if(STLQpp_LAPACKE_INCLUDE_DIR AND STLQpp_LAPACK_OPENBLAS_LIBRARY AND
       NOT STLQpp_LAPACKE_LIBRARY)
      include(CheckCXXSourceCompiles)
      set(CMAKE_REQUIRED_INCLUDES "${STLQpp_LAPACKE_INCLUDE_DIR}")
      set(CMAKE_REQUIRED_LIBRARIES "${STLQpp_LAPACK_OPENBLAS_LIBRARY}")
      if(MSVC)
        set(CMAKE_REQUIRED_DEFINITIONS -DHAVE_LAPACK_CONFIG_H -D_CRT_USE_C_COMPLEX_H)
      endif()
      check_cxx_source_compiles("
        #include <lapacke.h>
        int main() {
          float s = 1.0f;
          double d = 1.0;
          return LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', 1, &s, 1) +
              LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', 1, 1, &s, 1, &s, 1) +
              LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', 1, &d, 1) +
              LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', 1, 1, &d, 1, &d, 1) +
              LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', 1, 1,
                  &d, 1, &d, &d, 1, &d, 1, &d);
        }" STLQpp_OPENBLAS_HAS_LAPACKE)
      if(STLQpp_OPENBLAS_HAS_LAPACKE)
        set(STLQpp_LAPACKE_LIBRARY "${STLQpp_LAPACK_OPENBLAS_LIBRARY}")
      endif()
    endif()
    if(STLQpp_LAPACKE_INCLUDE_DIR AND STLQpp_LAPACKE_LIBRARY AND
       STLQpp_LAPACK_OPENBLAS_LIBRARY)
      add_library(stlqpp_runtime_lapack STATIC
          src/runtime/cpu/lapack_square_svd_f32.cpp
          src/runtime/cpu/lapack_spd_solver_f32.cpp
          src/runtime/cpu/lapack_spd_solver_f64.cpp)
      add_library(STLQpp::runtime_lapack ALIAS stlqpp_runtime_lapack)
      target_include_directories(stlqpp_runtime_lapack PRIVATE
          "${STLQpp_LAPACKE_INCLUDE_DIR}")
      if(MSVC)
        # Select LAPACKE's MSVC complex types instead of C99 _Complex.
        target_compile_definitions(stlqpp_runtime_lapack PRIVATE
            HAVE_LAPACK_CONFIG_H _CRT_USE_C_COMPLEX_H)
      endif()
      target_link_libraries(stlqpp_runtime_lapack
          PUBLIC STLQpp::core
          PRIVATE "${STLQpp_LAPACKE_LIBRARY}"
                  "${STLQpp_LAPACK_OPENBLAS_LIBRARY}")
      stlqpp_configure_target(stlqpp_runtime_lapack)
      set(STLQpp_LAPACK_AVAILABLE TRUE PARENT_SCOPE)
      message(STATUS
          "STLQpp LAPACKE training backend: OpenBLAS/LAPACKE")
      return()
    endif()
    if(backend STREQUAL "OPENBLAS")
      message(FATAL_ERROR "OpenBLAS LAPACKE requested but unavailable")
    endif()
  endif()

  message(STATUS "STLQpp LAPACKE training backend: unavailable")
endfunction()
