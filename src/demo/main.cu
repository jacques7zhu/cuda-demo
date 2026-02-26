#include <iostream>

// A minimal CUDA kernel
__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;

    // Launch the kernel with 1 block and 5 threads
    helloFromGPU<<<1, 5>>>();

    // Wait for the GPU to finish before the CPU exits
    cudaDeviceSynchronize();

    return 0;
}