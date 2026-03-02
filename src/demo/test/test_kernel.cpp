#include "kernel.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>

TEST(test_kernel, Basic)
{
    const int num_blocks = 2, block_size = 4;
    const int n = num_blocks * block_size * 256;
    const size_t bytes = n * sizeof(int);
    std::vector<int> h_A(n, 1);
    std::vector<int> h_B(n, 2);
    std::vector<int> h_C(n, 0);
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    vadd(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();    
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_C[i] != 3) {
            std::cout << "Mismatch at index " << i << ": " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) std::cout << "Success! Vector addition completed using async memcpy." << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}