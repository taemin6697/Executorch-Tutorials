cmake_minimum_required(VERSION 3.24)
project(ExampleSuite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This top-level CMakeLists is just a convenient entry point.
# Each subproject has its own CMakeLists.txt and can be built independently.

option(EXAMPLE_BUILD_RUNNING_LLAMA "Build Running_Llama (llama_chat)" ON)
option(EXAMPLE_BUILD_COMPARISON_CPU_GPU "Build Building_from_Source/comparison_cpu_gpu" OFF)
option(EXAMPLE_BUILD_RUNNING_ON_DEVICES_CPU "Build Building_from_Source/running_on_devices_cpu" OFF)
option(EXAMPLE_BUILD_RUNNING_ON_DEVICES_GPU "Build Building_from_Source/running_on_devices_gpu" OFF)

if(EXAMPLE_BUILD_RUNNING_LLAMA)
  add_subdirectory(Running_Llama)
endif()

if(EXAMPLE_BUILD_COMPARISON_CPU_GPU)
  add_subdirectory(Building_from_Source/comparison_cpu_gpu)
endif()

if(EXAMPLE_BUILD_RUNNING_ON_DEVICES_CPU)
  add_subdirectory(Building_from_Source/running_on_devices_cpu)
endif()

if(EXAMPLE_BUILD_RUNNING_ON_DEVICES_GPU)
  add_subdirectory(Building_from_Source/running_on_devices_gpu)
endif()


