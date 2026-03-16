# Spheroid-3DFFT

## What This Is

A high-performance 3D FFT library for custom GPU hardware, implementing a compute-storage separated design for spheroid data. It leverages a proprietary CUDA-like toolchain to bypass SIMT and provide full control over each compute core.

## Core Value

Efficiently compute 3D FFT on spheroid data while minimizing DDR traffic through compute-storage separation.

## Requirements

### Validated

- ✓ **FFT1D-01**: 1D FFT implementation (single-core device function) — *existing*

### Active

- [ ] **FFT3D-01**: Small-scale 3D FFT (all data fits in on-chip APC cache)
- [ ] **FFT3D-02**: Medium-scale 3D FFT (one plane fits on-chip, total fits on DDR)
- [ ] **FFT3D-03**: Large-scale 3D FFT (out-of-core, exceeds DDR capacity)
- [ ] **REORDER-01**: Spheroid-to-grid reordering (forward and inverse)
- [ ] **LAYOUT-01**: Optimal data layout and twiddler factor design for spheroid
- [ ] **SCHED-01**: Binary search + Greedy task allocation for load balancing

### Out of Scope

- **GPU-GENERIC**: Support for standard NVIDIA/AMD GPUs — *Project is focused on custom hardware*
- **FFT-GENERIC**: General Cartesian 3D FFT — *Library is specialized for spheroid data*

## Context

- **Environment**: Custom hardware with 32 StreamProcessors (4 cores/SP), 16GB DDR, 256KB local memory per core.
- **Protocol**: Compute-Storage Separation (CSS) using a subset of CUDA.
- **Design Doc**: Refer to `docs/fft-algo-design.pdf` for detailed algorithm and performance models.

## Constraints

- **Hardware**: No SIMT support, explicit core management required.
- **Memory**: 256KB local memory per core limit.
- **Toolchain**: Proprietary compiler and runtime, no CUDA Graph.
- **Performance**: High DDR latency requires aggressive compute-storage separation.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Compute-Storage Separation | Minimize DDR interaction given bandwidth bottlenecks | — Pending |
| Binary Search + Greedy Allocation | Ensure load balancing given sparsity of spheroid data | — Pending |

---
*Last updated: 2026-03-16 after initialization*
