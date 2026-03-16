cuda_library(
    TARGET fft_lib
    SRCS src/fft.cu src/kernels.cu
    INCLUDES ${CMAKE_SOURCE_DIR}/src inc
)

cuda_gtest(
    TARGET test_fft
    SRCS test/test_fft.cpp
    DEPS fft_lib
)
