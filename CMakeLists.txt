cmake_minimum_required(VERSION 3.16)
project(yolov8_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 当前构建脚本使用 GCC/Clang 风格选项；不把 MSVC 伪装成已验证的平台。
if(MSVC)
  message(FATAL_ERROR "当前构建支持 GCC/Clang（建议 WSL）；尚未实现 MSVC 的 ISA 与 sanitizer 配置")
endif()
option(YI_SANITIZE "AddressSanitizer + UBSan" OFF)
option(YI_TSAN "ThreadSanitizer（并发正确性检查；与 YI_SANITIZE 互斥）" OFF)
option(YI_ENABLE_X86_SIMD "在 x86-64 GCC/Clang 上构建 SSE2 与 AVX2/FMA 后端" ON)

add_compile_options(-Wall -Wextra)
if(YI_SANITIZE AND YI_TSAN)
  message(FATAL_ERROR "YI_SANITIZE 与 YI_TSAN 不能同时开启")
endif()
if(YI_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(YI_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

add_library(yinfer_core STATIC
  runtime/common.cpp
  runtime/backend.cpp
  runtime/model.cpp
  runtime/memory_plan.cpp
  runtime/thread_pool.cpp
  runtime/executor.cpp
  runtime/ops/ref_ops.cpp
  runtime/ops/dispatch.cpp)
target_include_directories(yinfer_core PUBLIC runtime)
target_link_libraries(yinfer_core PUBLIC Threads::Threads)

if(YI_ENABLE_X86_SIMD AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_definitions(yinfer_core PRIVATE YI_X86_SIMD=1)
  # 默认代码保持基线 ISA；只有两个 SIMD 源文件获得更高 ISA 权限。
  target_compile_options(yinfer_core PRIVATE -march=x86-64 -mtune=generic -ffp-contract=off)
  target_sources(yinfer_core PRIVATE runtime/ops/sse_ops.cpp runtime/ops/avx2_ops.cpp)
  set_source_files_properties(runtime/ops/sse_ops.cpp PROPERTIES COMPILE_OPTIONS "-msse2;-mno-avx;-mno-avx2;-mno-fma")
  set_source_files_properties(runtime/ops/avx2_ops.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

add_executable(yinfer tools/yinfer.cpp tools/reference.cpp)
target_link_libraries(yinfer PRIVATE yinfer_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_memory_plan tests/test_memory_plan.cpp)
  target_link_libraries(test_memory_plan PRIVATE yinfer_core)
  add_test(NAME memory_plan COMMAND test_memory_plan)
  add_executable(test_simd_runtime tests/test_simd_runtime.cpp)
  target_link_libraries(test_simd_runtime PRIVATE yinfer_core)
  add_test(NAME simd_runtime COMMAND test_simd_runtime)
endif()
