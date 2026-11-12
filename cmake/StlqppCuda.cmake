include_guard(GLOBAL)

include(CheckLanguage)

set(STLQpp_CUDA_AVAILABLE OFF)
# Default to Ada; callers can select their GPU with -DCMAKE_CUDA_ARCHITECTURES.
if(STLQpp_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA target architectures")
endif()

macro(stlqpp_detect_cuda)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "STLQpp CUDA runtime backend: unavailable")
    set(STLQpp_CUDA_AVAILABLE OFF)
  else()
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      message(FATAL_ERROR
          "STLQpp CUDA builds require NVIDIA nvcc")
    endif()
    find_package(CUDAToolkit REQUIRED)
    # MSVC requires C++20 for the designated initializers in shared headers.
    # Preserve the existing CUDA dialect for the GCC/Linux toolchain.
    if(MSVC)
      set(CMAKE_CUDA_STANDARD 20)
    else()
      set(CMAKE_CUDA_STANDARD 17)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(STLQpp_CUDA_AVAILABLE ON)
    message(STATUS "STLQpp CUDA runtime backend: enabled")
  endif()
endmacro()

function(stlqpp_configure_cuda_target target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  target_include_directories(${target_name}
      PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
  set_target_properties(${target_name} PROPERTIES
      CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION OFF)
  if(MSVC)
    target_compile_definitions(${target_name} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    if(STLQpp_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  else()
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
    if(STLQpp_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  endif()
endfunction()
