cmake_minimum_required(VERSION 3.10)

# [중요] 안드로이드 빌드 설정 (반드시 project() 선언보다 위에!)
if(ANDROID_BUILD)
    # NDK 경로는 -DANDROID_NDK=... 로 넘기는 것을 권장
    # 기본값은 환경변수 ANDROID_NDK_HOME/ANDROID_NDK_ROOT 또는 ~/android-ndk-r27d 를 사용
    if(NOT DEFINED ANDROID_NDK)
        if(DEFINED ENV{ANDROID_NDK_HOME})
            set(ANDROID_NDK "$ENV{ANDROID_NDK_HOME}" CACHE PATH "" FORCE)
        elseif(DEFINED ENV{ANDROID_NDK_ROOT})
            set(ANDROID_NDK "$ENV{ANDROID_NDK_ROOT}" CACHE PATH "" FORCE)
        else()
            set(ANDROID_NDK "$ENV{HOME}/android-ndk-r27d" CACHE PATH "" FORCE)
        endif()
    endif()

    if(NOT EXISTS "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")
        message(FATAL_ERROR "Android NDK not found or invalid. ANDROID_NDK='${ANDROID_NDK}'")
    endif()

    set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" CACHE PATH "" FORCE)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "" FORCE)
    set(ANDROID_PLATFORM "android-34" CACHE STRING "" FORCE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    set(CMAKE_SYSTEM_NAME Android CACHE STRING "" FORCE) # 명시적으로 안드로이드임을 선언
endif()

project(ExecuTorchApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# [수정] 실행 파일이 생성되는 위치를 프로젝트 루트 폴더(pipeline)로 고정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# 3. ExecuTorch 경로 설정
set(EXECUTORCH_SOURCE_DIR "/home/tm0118/Desktop/executorch")

# 4. 필수 의존성 옵션 모두 켜기 (에러 방지용)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
# Optimized kernels는 PyTorch(torch) 헤더를 요구할 수 있어서 기본은 OFF로 둠.
# 필요하면 configure 시 -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON 으로 켜기.
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "" FORCE)

# 5. 외부 폴더 추가
add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "executorch_build")

# 6. 실행 파일 및 링크
add_executable(my_app main.cpp)

target_link_libraries(my_app
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
)

