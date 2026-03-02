cmake_minimum_required(VERSION 3.20)

project(cuda_demo LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)
find_package(CUDAToolkit REQUIRED)

include("${CMAKE_SOURCE_DIR}/gtest.cmake")
include("${CMAKE_SOURCE_DIR}/cuda.cmake")

add_subdirectory(src/demo)