#include "kernel.h"

#include <cuda/barrier>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <stdio.h>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;


static __device__ void vadd_256(const int *local_a, const int *local_b, int *local_c)
{
    for (int i = 0; i < 256; i++)
    {
        local_c[i] = local_a[i] + local_b[i];
    }
}

__global__ void vadd_kernel(const __DDR int *a, const __DDR int *b, __DDR int *c, int n) 
{
    __mem0__ int local_a[256];
    __mem1__ int local_b[256];
    __mem2__ int local_c[256];

    auto block = cooperative_groups::this_thread_block();
    __shared__ barrier_t bar;

    if (block.thread_rank() == 0)
    {
        init(&bar, block.size());
    }
    block.sync();
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = tid * 256;
    int size = sizeof(local_a);
    cuda::memcpy_async((int *)local_a, (__DDR int *)(a + offset), size, bar);
    cuda::memcpy_async((int *)local_b, (__DDR int *)(b + offset), size, bar);
    bar.arrive_and_wait();

    vadd_256(local_a, local_b, local_c);
    
    cuda::memcpy_async(c + offset, (int *)local_c, size, bar);
    bar.arrive_and_wait();
}

void vadd(const __DDR int *a, const __DDR int *b, __DDR int *c, int n)
{
    int threads = 4;
    int blocks = (n + threads - 1) / threads / 256;
    vadd_kernel<<<blocks, threads>>>(a, b, c, n);
}