# Feature Landscape

**Domain:** 3D FFT for Custom Hardware (Spheroid Data)
**Researched:** 2026-03-16

## Table Stakes

Features users expect in any high-performance FFT library. Missing = product feels incomplete or cannot scale.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **1D/2D/3D C2C Transforms** | Core mathematical primitives required for any multidimensional FFT. | Medium | Must support forward and inverse variants, along with mixed-radix (Radix-2, 3, 5). |
| **Small-Scale Support** | Base processing capability when all data fits in the on-chip cache (SRAM / APC). | Medium | Total data ≤ 120MB. Requires 0 DDR interactions for intermediate steps. |
| **Medium-Scale Support** | Handles typical workloads where total data exceeds SRAM but fits in system memory (DDR). | High | 120MB < Total Data < 2GB. Requires slab-wise processing and 1 DDR roundtrip. |
| **Large-Scale Support** | Out-of-core handling for massive datasets exceeding DDR capacity. | High | Total Data > 2GB. Requires disk/multi-die swapping and 2+ DDR roundtrips. |

## Differentiators

Features that set this specific library apart, highly optimized for the custom non-SIMT hardware and the sparse "spheroid" data structure.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Compute-Storage Separation (CSS)** | Masks high DDR latency by explicitly dedicating cores (C-APC) to compute and others (S-APC) to caching and layout management. | High | Transforms a transmission-bound problem into a balanced/compute-bound one. |
| **Spheroid-Aware Sparse Processing** | Skips computation on zero-padded empty regions outside the spheroid, significantly saving MAC operations. | High | A massive efficiency gain over dense 3D FFTs. |
| **Binary Search + Greedy Load Balancing** | Dynamically resolves severe load imbalance caused by varying 1D "pencil" lengths in spheroid cross-sections. | Medium | Uses analytical cost models to explicitly balance tasks across Compute APCs. |
| **In-SRAM Transposition (x2y)** | Eliminates the need for expensive DDR-based matrix transpositions during the 3D FFT pipeline. | High | Leverages high-bandwidth intra-chip networks (SDB) via Storage APCs. |
| **Hybrid Task Decomposition** | Adapts memory patterns optimally based on dimension: uses "Pencil" for X-direction and "Slab" for YZ planes. | Medium | Essential for fitting chunks efficiently into the 256KB/core local memory. |

## Anti-Features

Features to explicitly NOT build, as they degrade performance on this hardware or contradict the spheroid requirement.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **General-Purpose Cartesian FFT** | Computes the full dense 3D grid, wasting vast amounts of compute cycles on the "empty" corners of the bounding box. | Implement specialized sparse spheroid pencil extraction and computation. |
| **SIMT Execution & Auto-Vectorization** | The custom hardware lacks SIMT execution; relying on generic GPU execution models will fail. | Rely on explicit core-level programming, local memory management, and CSS. |
| **Runtime Auto-Tuning Planners (e.g., FFTW)** | Empirical runtime tuning adds unnecessary overhead since the hardware limits and data shape are strictly modeled and known ahead of time. | Use the deterministic Binary Search + Greedy algorithm based on analytical cost functions. |

## Feature Dependencies

- **1D Transform Primitives** → **2D/3D Transforms** (Prerequisite for multi-dimensional operations)
- **Compute-Storage Separation (CSS)** → **In-SRAM Transposition** (Transposes rely on Storage APCs)
- **Spheroid-to-Grid Reordering** → **Binary Search + Greedy Load Balancing** (Pencil lengths must be known to balance)
- **Binary Search + Greedy Load Balancing** → **Small/Medium/Large Scale Support** (Required for scaling execution)

## MVP Recommendation

**Prioritize:**
1. **1D FFT Primitives (Radix-2/3/5)**: The bedrock of all calculations.
2. **Small-Scale System Support**: End-to-end flow validation entirely within on-chip memory (no DDR complexity).
3. **Compute-Storage Separation Logic**: Establish the fundamental interaction between C-APC and S-APC.
4. **Binary Search + Greedy Load Balancing**: Validate efficiency on sparse spheroid geometries.

**Defer:** 
- **Large-Scale Support (Out-of-Core / Multi-die)**: Introduce only after Small and Medium scales are robust, as it adds significant I/O and external storage latency complexities.

## Sources

- `.planning/PROJECT.md` (Project Context)
- `docs/fft-algo-design.pdf` (Existing Algorithm & Hardware Architecture Design)
- Web Research on High-Performance 3D FFTs (SpFFT, P3DFFT, cuFFT) emphasizing sparse frequency handling and domain decomposition strategies.