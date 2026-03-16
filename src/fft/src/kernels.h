#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

/**
 * @brief CUDA kernel for 3D FFT computation. (Stub)
 */
__global__ void fft3d_compute_kernel(
    int nx, int ny, int nz,
    cuDoubleComplex* d_input,
    cuDoubleComplex* d_output);

/**
 * @brief CUDA kernel for reordering spheroid data to grid. (Stub)
 */
__global__ void spheroid_reorder_kernel(
    int nx, int ny, int nz,
    double rx, double ry, double rz,
    cuDoubleComplex* d_raw,
    cuDoubleComplex* d_grid,
    void* d_map,
    bool inverse);
