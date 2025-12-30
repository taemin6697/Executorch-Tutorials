cmake_minimum_required(VERSION 3.29)

project(RunningLlama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LLAMA_BUILD_VULKAN "Build ExecuTorch Vulkan backend (for Vulkan-delegated .pte models)" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ExecuTorch source (sibling of example/)
set(EXECUTORCH_SOURCE_DIR "/home/tm0118/Desktop/executorch" CACHE PATH "ExecuTorch source directory")

# Build options (tune as needed)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_LLM ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER ON CACHE BOOL "" FORCE)

set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_LLM ON CACHE BOOL "" FORCE)

set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PTHREADPOOL ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CPUINFO ON CACHE BOOL "" FORCE)

if(LLAMA_BUILD_VULKAN)
  set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "executorch_build")

# Build the ExecuTorch llama C++ example as llama_chat
set(LLAMA_EXAMPLE_DIR "${EXECUTORCH_SOURCE_DIR}/examples/models/llama")
add_executable(llama_chat
  "${LLAMA_EXAMPLE_DIR}/main.cpp"
  "${LLAMA_EXAMPLE_DIR}/runner/runner.cpp"
  "${LLAMA_EXAMPLE_DIR}/tokenizer/llama_tiktoken.cpp"
)

target_include_directories(llama_chat PRIVATE
  "${EXECUTORCH_SOURCE_DIR}"
  "${EXECUTORCH_SOURCE_DIR}/.."
  "${LLAMA_EXAMPLE_DIR}"
  "${EXECUTORCH_SOURCE_DIR}/extension/llm/tokenizers/include"
)

# gflags target selection (varies by config)
set(LLAMA_GFLAGS_TARGET "")
if(TARGET gflags_nothreads_static)
  set(LLAMA_GFLAGS_TARGET gflags_nothreads_static)
elseif(TARGET gflags::gflags)
  set(LLAMA_GFLAGS_TARGET gflags::gflags)
elseif(TARGET gflags)
  set(LLAMA_GFLAGS_TARGET gflags)
endif()
if(NOT LLAMA_GFLAGS_TARGET)
  message(FATAL_ERROR "gflags target not found.")
endif()

# Base libs (always)
set(LLAMA_LINK_LIBS
  executorch
  ${LLAMA_GFLAGS_TARGET}
)

# Optional extensions commonly needed by the llama example
foreach(_t IN ITEMS extension_module extension_data_loader extension_llm_runner tokenizers extension_threadpool)
  if(TARGET ${_t})
    list(APPEND LLAMA_LINK_LIBS ${_t})
  endif()
endforeach()

# Kernels / ops (match ExecuTorch llama example logic)
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND LLAMA_LINK_LIBS
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
elseif(TARGET portable_ops_lib)
  list(APPEND LLAMA_LINK_LIBS portable_ops_lib portable_kernels)
elseif(TARGET portable_kernels)
  list(APPEND LLAMA_LINK_LIBS portable_kernels)
endif()

if(TARGET quantized_kernels)
  list(APPEND LLAMA_LINK_LIBS quantized_kernels)
endif()
if(TARGET quantized_ops_lib)
  list(APPEND LLAMA_LINK_LIBS quantized_ops_lib)
endif()

if(TARGET xnnpack_backend)
  list(APPEND LLAMA_LINK_LIBS xnnpack_backend)
endif()

if(TARGET vulkan_backend)
  list(APPEND LLAMA_LINK_LIBS vulkan_backend)
endif()

# LLM custom ops registration (prevents Missing operator: llama::custom_sdpa.out)
if(TARGET custom_ops)
  if(UNIX AND NOT APPLE)
    list(APPEND LLAMA_LINK_LIBS "-Wl,--whole-archive" custom_ops "-Wl,--no-whole-archive")
  else()
    list(APPEND LLAMA_LINK_LIBS custom_ops)
  endif()
endif()

target_link_libraries(llama_chat PRIVATE ${LLAMA_LINK_LIBS})
