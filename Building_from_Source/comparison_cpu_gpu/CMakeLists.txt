cmake_minimum_required(VERSION 3.10)

# -------------------------------
# Build mode switch
# -------------------------------
# Host(WSL/Linux):   -DHOST_BUILD=ON  (default)
# Android:           -DHOST_BUILD=OFF
option(HOST_BUILD "Build for host (Linux/WSL). If OFF, build for Android." ON)

# Keep HOST_BUILD and ANDROID_BUILD consistent (only one active)
if(ANDROID_BUILD AND HOST_BUILD)
    message(FATAL_ERROR "Both ANDROID_BUILD and HOST_BUILD are ON. Set only one (HOST_BUILD=ON for host, HOST_BUILD=OFF for Android).")
endif()
if(ANDROID_BUILD)
    set(HOST_BUILD OFF CACHE BOOL "" FORCE)
endif()
if(NOT HOST_BUILD)
    set(ANDROID_BUILD ON CACHE BOOL "" FORCE)
endif()

# -------------------------------
# Android toolchain defaults (only when ANDROID_BUILD=ON)
# -------------------------------
if(ANDROID_BUILD)
    # Prefer -DANDROID_NDK=...; otherwise use env or a local default.
    if(NOT DEFINED ANDROID_NDK)
        if(DEFINED ENV{ANDROID_NDK_HOME})
            set(ANDROID_NDK "$ENV{ANDROID_NDK_HOME}" CACHE PATH "")
        elseif(DEFINED ENV{ANDROID_NDK_ROOT})
            set(ANDROID_NDK "$ENV{ANDROID_NDK_ROOT}" CACHE PATH "")
        else()
            set(ANDROID_NDK "$ENV{HOME}/android-ndk-r27d" CACHE PATH "")
        endif()
    endif()

    if(NOT EXISTS "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")
        message(FATAL_ERROR "Android NDK not found or invalid. ANDROID_NDK='${ANDROID_NDK}'")
    endif()

    # If user already passed these via command line, do not override.
    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" CACHE PATH "")
    endif()
    if(NOT ANDROID_ABI)
        set(ANDROID_ABI "arm64-v8a" CACHE STRING "")
    endif()
    if(NOT ANDROID_PLATFORM)
        set(ANDROID_PLATFORM "android-34" CACHE STRING "")
    endif()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "")
    endif()
    if(NOT CMAKE_SYSTEM_NAME)
        set(CMAKE_SYSTEM_NAME Android CACHE STRING "")
    endif()
endif()

project(ExecuTorchApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place the built executable in the project root (so adb push is easy).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# -------------------------------
# ExecuTorch as sibling directory (B-style)
# -------------------------------
# Default assumes:
#   /home/tm0118/Desktop/executorch
#   /home/tm0118/Desktop/example/Building_from_Source/comparison_cpu_gpu  (this project)
set(EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../executorch" CACHE PATH "ExecuTorch source directory")

# ExecuTorch build options (safe defaults for this app)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "" FORCE)

# Optimized kernels can require Torch headers; keep OFF unless you explicitly enable it.
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "" FORCE)

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "executorch_build")

# -------------------------------
# App
# -------------------------------
add_executable(my_app main.cpp)

# Link targets: prefer namespaced aliases; fallback to non-namespaced targets if needed.
set(_ET_BACKENDS_TARGET "")
if(TARGET executorch::backends)
    set(_ET_BACKENDS_TARGET executorch::backends)
elseif(TARGET executorch_backends)
    set(_ET_BACKENDS_TARGET executorch_backends)
endif()

set(_ET_EXTENSIONS_TARGET "")
if(TARGET executorch::extensions)
    set(_ET_EXTENSIONS_TARGET executorch::extensions)
elseif(TARGET executorch_extensions)
    set(_ET_EXTENSIONS_TARGET executorch_extensions)
endif()

set(_ET_KERNELS_TARGET "")
if(TARGET executorch::kernels)
    set(_ET_KERNELS_TARGET executorch::kernels)
elseif(TARGET executorch_kernels)
    set(_ET_KERNELS_TARGET executorch_kernels)
endif()

target_link_libraries(my_app PRIVATE executorch)
if(_ET_BACKENDS_TARGET)
    target_link_libraries(my_app PRIVATE ${_ET_BACKENDS_TARGET})
endif()
if(_ET_EXTENSIONS_TARGET)
    target_link_libraries(my_app PRIVATE ${_ET_EXTENSIONS_TARGET})
endif()
if(_ET_KERNELS_TARGET)
    target_link_libraries(my_app PRIVATE ${_ET_KERNELS_TARGET})
endif()


