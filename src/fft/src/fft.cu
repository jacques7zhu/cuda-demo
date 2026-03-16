#include "fft/inc/fft.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>

FftStatus fft3d_spheroid_create_plan(
    SpheroidLayout layout,
    Fft3dSpheroidPlan** plan) {
    
    if (plan == nullptr) return FFT_INVALID_PARAM;
    
    *plan = (Fft3dSpheroidPlan*)malloc(sizeof(Fft3dSpheroidPlan));
    if (*plan == nullptr) return FFT_PLAN_FAILED;
    
    (*plan)->layout = layout;
    (*plan)->stream = nullptr;
    
    // In a real implementation, we would allocate d_reorder_map here
    // based on the layout logic. For now, it's a stub.
    (*plan)->d_reorder_map = nullptr;
    
    return FFT_SUCCESS;
}

FftStatus fft3d_spheroid_destroy_plan(
    Fft3dSpheroidPlan* plan) {
    if (plan == nullptr) return FFT_INVALID_PARAM;
    
    if (plan->d_reorder_map) {
        cudaFree(plan->d_reorder_map);
    }
    
    free(plan);
    return FFT_SUCCESS;
}

FftStatus fft3d_spheroid_reorder(
    Fft3dSpheroidPlan* plan,
    __DDR cuDoubleComplex* d_raw_data,
    __DDR cuDoubleComplex* d_grid_data,
    bool inverse,
    cudaStream_t stream) {
    
    if (plan == nullptr || d_raw_data == nullptr || d_grid_data == nullptr) {
        return FFT_INVALID_PARAM;
    }
    
    int total_elements = plan->layout.nx * plan->layout.ny * plan->layout.nz;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    spheroid_reorder_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        plan->layout.nx, plan->layout.ny, plan->layout.nz,
        plan->layout.radius_x, plan->layout.radius_y, plan->layout.radius_z,
        d_raw_data, d_grid_data,
        plan->d_reorder_map,
        inverse
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? FFT_SUCCESS : FFT_CUDA_ERROR;
}

FftStatus fft3d_spheroid_execute(
    Fft3dSpheroidPlan* plan,
    __DDR cuDoubleComplex* d_input,
    __DDR cuDoubleComplex* d_output,
    cudaStream_t stream) {
    
    if (plan == nullptr || d_input == nullptr || d_output == nullptr) {
        return FFT_INVALID_PARAM;
    }
    
    // Compute-Storage Separation: 
    // This wrapper would typically manage the sequence of:
    // 1. Reorder spheroid -> grid (if not done externally)
    // 2. Perform 3D FFT on grid
    // 3. Reorder grid -> spheroid (if needed)
    
    int total_elements = plan->layout.nx * plan->layout.ny * plan->layout.nz;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    fft3d_compute_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        plan->layout.nx, plan->layout.ny, plan->layout.nz,
        d_input, d_output
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? FFT_SUCCESS : FFT_CUDA_ERROR;
}
