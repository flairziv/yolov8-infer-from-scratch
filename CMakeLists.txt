cmake_minimum_required(VERSION 3.16)
project(yolov8_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)                     # 用 -std=c++17 而不是 -std=gnu++17
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(YI_SANITIZE "AddressSanitizer + UBSan：越界读写、未定义行为当场报错（慢 2~3 倍，写算子时用）" OFF)

add_compile_options(-Wall -Wextra)
if(YI_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# 运行时核心：模型加载、内存规划、执行器、算子
add_library(yinfer_core STATIC
  runtime/common.cpp
  runtime/model.cpp
  runtime/memory_plan.cpp
  runtime/executor.cpp
  runtime/ops/ref_ops.cpp)
target_include_directories(yinfer_core PUBLIC runtime)

# 命令行工具：info / run / verify
add_executable(yinfer tools/yinfer.cpp tools/reference.cpp)
target_link_libraries(yinfer PRIVATE yinfer_core)
