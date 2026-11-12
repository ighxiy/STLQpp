include_guard(GLOBAL)

if(NOT MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
      "STLQpp supports the registered GCC and MSVC toolchains only")
endif()

option(STLQpp_ENABLE_CUDA "Build the optional CUDA runtime backend" ON)
option(STLQpp_ENABLE_LTO "Enable interprocedural optimization" OFF)
option(STLQpp_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
set(STLQpp_SOURCE_IDENTITY "unregistered" CACHE STRING
    "Registered source commit or frozen source-inventory SHA-256")
set(STLQpp_SOURCE_STATE "unregistered" CACHE STRING
    "Registered source state: clean or frozen-inventory")

set(STLQpp_IPO_AVAILABLE OFF)
if(STLQpp_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT STLQpp_IPO_AVAILABLE OUTPUT STLQpp_IPO_ERROR)
  if(NOT STLQpp_IPO_AVAILABLE)
    message(FATAL_ERROR "STLQpp LTO requested but unavailable: ${STLQpp_IPO_ERROR}")
  endif()
endif()

function(stlqpp_configure_target target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  target_include_directories(${target_name}
      PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive- /utf-8 /Zc:preprocessor)
    target_compile_definitions(${target_name} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    if(STLQpp_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(STLQpp_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
    # Allow fused operations in production numerical kernels.
    target_compile_options(${target_name} PRIVATE -ffp-contract=fast)
  endif()

  if(STLQpp_ENABLE_LTO AND STLQpp_IPO_AVAILABLE)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endfunction()
