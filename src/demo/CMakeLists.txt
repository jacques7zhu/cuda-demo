
cuda_library(
    TARGET kernel_lib 
    SRCS src/kernel.cu
    INCLUDES ${CMAKE_SOURCE_DIR}/src inc
)

cuda_gtest(
    TARGET test_kernel
    SRCS test/test_kernel.cpp
    DEPS kernel_lib
)