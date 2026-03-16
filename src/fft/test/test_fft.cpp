#include <gtest/gtest.h>
#include "fft/inc/fft.h"
#include <cuda_runtime.h>

class FftTest : public ::testing::Test {
protected:
    void SetUp() override {
        layout.nx = 16;
        layout.ny = 16;
        layout.nz = 16;
        layout.radius_x = 8.0;
        layout.radius_y = 8.0;
        layout.radius_z = 8.0;
        layout.padding = 0;
    }

    SpheroidLayout layout;
};

TEST_F(FftTest, CreateAndDestroyPlan) {
    Fft3dSpheroidPlan* plan = nullptr;
    FftStatus status = fft3d_spheroid_create_plan(layout, &plan);
    EXPECT_EQ(status, FFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    
    status = fft3d_spheroid_destroy_plan(plan);
    EXPECT_EQ(status, FFT_SUCCESS);
}

TEST_F(FftTest, ExecuteStub) {
    Fft3dSpheroidPlan* plan = nullptr;
    fft3d_spheroid_create_plan(layout, &plan);
    
    size_t size = layout.nx * layout.ny * layout.nz * sizeof(cuDoubleComplex);
    cuDoubleComplex *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    FftStatus status = fft3d_spheroid_execute(plan, d_input, d_output, nullptr);
    EXPECT_EQ(status, FFT_SUCCESS);
    
    cudaFree(d_input);
    cudaFree(d_output);
    fft3d_spheroid_destroy_plan(plan);
}

TEST_F(FftTest, ReorderStub) {
    Fft3dSpheroidPlan* plan = nullptr;
    fft3d_spheroid_create_plan(layout, &plan);
    
    size_t size = layout.nx * layout.ny * layout.nz * sizeof(cuDoubleComplex);
    cuDoubleComplex *d_raw, *d_grid;
    cudaMalloc(&d_raw, size);
    cudaMalloc(&d_grid, size);
    
    FftStatus status = fft3d_spheroid_reorder(plan, d_raw, d_grid, false, nullptr);
    EXPECT_EQ(status, FFT_SUCCESS);
    
    cudaFree(d_raw);
    cudaFree(d_grid);
    fft3d_spheroid_destroy_plan(plan);
}
