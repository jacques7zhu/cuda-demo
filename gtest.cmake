
function(cuda_gtest)
    set(oneValueArgs TARGET)
    set(multiValueArgs SRCS INCLUDES DEPS)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${ARG_TARGET} ${ARG_SRCS})

    # 2. Link the required libraries (User libs + CUDA Runtime + GTest)
    target_link_libraries(${ARG_TARGET} PUBLIC
        ${ARG_DEPS}
        CUDA::cudart          # Required for .cpp files to call cudaMalloc/cudaMemcpy
        GTest::gtest_main     # Provides the main() entry point and GTest framework
    )
    if(ARG_INCLUDES)
        target_include_directories(${ARG_TARGET} PUBLIC ${ARG_INCLUDES})
    endif()
    gtest_discover_tests(${ARG_TARGET})
endfunction()