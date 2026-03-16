#pragma once
#include "common/inc/predefine.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

/**
 * @brief Status codes for FFT operations.
 */
enum FftStatus {
    FFT_SUCCESS = 0,
    FFT_INVALID_PARAM = -1,
    FFT_CUDA_ERROR = -2,
    FFT_PLAN_FAILED = -3,
    FFT_EXECUTION_FAILED = -4
};

/**
 * @brief Spheroid data layout definition.
 * For compute-storage separation, this defines how the spheroid
 * data maps to a 3D Cartesian grid.
 */
struct SpheroidLayout {
    int nx, ny, nz;      // Dimensions of the enclosing Cartesian grid
    double radius_x;     // Spheroid semi-major axis X
    double radius_y;     // Spheroid semi-major axis Y
    double radius_z;     // Spheroid semi-major axis Z
    // Add layout specific flags for reordering (e.g. padding, alignment)
    int padding;         
};

/**
 * @brief Plan for 3D FFT on spheroid data.
 * Pre-calculates mapping/reordering patterns to separate storage logic from compute.
 */
struct Fft3dSpheroidPlan {
    SpheroidLayout layout;
    void* d_reorder_map; // Device memory pointer for reordering mapping
    cudaStream_t stream;
};

/**
 * @brief Initialize a plan for spheroid 3D FFT.
 * 
 * @param layout Input spheroid layout.
 * @param plan Output plan handle.
 * @return FftStatus 
 */
FftStatus fft3d_spheroid_create_plan(
    SpheroidLayout layout,
    Fft3dSpheroidPlan** plan);

/**
 * @brief Destroy a plan and free associated device resources.
 * 
 * @param plan Plan handle to destroy.
 * @return FftStatus 
 */
FftStatus fft3d_spheroid_destroy_plan(
    Fft3dSpheroidPlan* plan);

/**
 * @brief Execute 3D FFT on spheroid data with compute-storage separation.
 * 
 * @param plan The pre-initialized plan.
 * @param d_input Device memory pointer (DDR) for spheroid input data.
 * @param d_output Device memory pointer (DDR) for FFT result.
 * @param stream CUDA stream for execution.
 * @return FftStatus 
 */
FftStatus fft3d_spheroid_execute(
    Fft3dSpheroidPlan* plan,
    __DDR cuDoubleComplex* d_input,
    __DDR cuDoubleComplex* d_output,
    cudaStream_t stream);

/**
 * @brief Separate kernel for reordering/transposition if needed externally.
 */
FftStatus fft3d_spheroid_reorder(
    Fft3dSpheroidPlan* plan,
    __DDR cuDoubleComplex* d_raw_data,
    __DDR cuDoubleComplex* d_grid_data,
    bool inverse,
    cudaStream_t stream);
