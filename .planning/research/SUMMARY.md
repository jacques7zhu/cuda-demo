# Research Summary: Spheroid-3DFFT

**Domain:** 3D FFT on Compute-Storage Separated Custom Hardware
**Researched:** 2026-03-16
**Overall confidence:** HIGH

## Executive Summary

The Spheroid-3DFFT project operates under strict hardware constraints: high-latency, limited-bandwidth DDR memory and small 256KB local caches per Compute APC (C-APC), completely lacking standard SIMT support. Implementing a fast 3D Fast Fourier Transform on this hardware transforms the problem from a compute-bound algorithmic challenge into a complex data orchestration and routing problem. Traditional approaches, such as uniform MPI-style data partitioning or all-compute execution paths, will severely degrade performance.

Through the analysis of the project's design document and corresponding architecture/stack decisions, the paramount strategy is **Compute-Storage Separation (CSS)**. By explicitly dividing the processing units into Compute cores (C-APCs) for pure math and Storage cores (S-APCs) for buffering and transposition over the 50GB/s APC2APC mesh, the system can effectively hide the DDR latency and bypass the half-duplex DDR bandwidth constraints. Furthermore, because the target data geometry is a "Spheroid" rather than a dense cube, it exhibits high sparsity at the edges. Thus, dynamic load balancing using binary search and greedy cost modeling is required to prevent center-core synchronization bottlenecks.

## Key Findings

**Stack:** Explicitly managed, non-SIMT proprietary toolchain relying on C++ with Compute-Storage Separation (CSS) and Stockham FFT patterns.
**Architecture:** Data streams from DDR → C-APC (1D FFT) → S-APC (Transpose via SDB) → C-APC (2D FFT) → DDR.
**Critical pitfall:** Uniform task partitioning ($N/P$) for the spheroid dataset and attempting in-core x2y transpositions that cause local SRAM bank conflicts and stall compute.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 1: Dynamic Load Balancer (SCHED-01)** - Must be addressed early. If uniform distribution is assumed in the core FFT logic, rewriting it later to support variable-sparsity pencils will be structurally invasive.
   - Addresses: Spheroid pencil sparsity logic.
   - Avoids: Synchronization stalls and center-core bottlenecks.

2. **Phase 2: Compute-Storage Separation Pipeline (FFT3D-01 / REORDER-01)** - Establish the fundamental S-APC vs C-APC interaction pattern on Small Scale data entirely within the cache.
   - Addresses: Small-scale 3D FFT, Stockham 1D FFT, Mesh transpositions.
   - Avoids: Local memory bank conflicts; DDR thrashing on small data.

3. **Phase 3: DDR Half-Duplex Orchestration (FFT3D-02)** - Expand the CSS pipeline to stream slabs from DDR without stalling.
   - Addresses: Medium-scale 3D FFT (Pencil + Slab with 1 DDR hit).
   - Avoids: Simultaneous read/write channel contention on half-duplex DDR.

4. **Phase 4: Out-of-Core Processing (FFT3D-03)** - External storage streaming for workloads exceeding DDR limits.
   - Addresses: Large-scale 3D FFT (3x Pencil).
   - Avoids: Pipeline blocking on slow disk I/O.

**Phase ordering rationale:**
- The Load Balancer dictates how data bounds are calculated and fed into the pipeline; it must precede the core FFT loops. Small Scale (in-cache) proves the CSS mesh routing, Medium Scale proves the DDR DMA orchestration, and Large Scale extends the DDR logic to external I/O.

**Research flags for phases:**
- Phase 3 (Medium Scale): Likely needs deeper research on hardware-specific DMA scatter-gather lists to effectively overlap asynchronous memory transfers with compute.
- Phase 4 (Large Scale): Requires validation of the disk I/O latency to determine the necessary prefetch buffer depth for S-APCs.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strictly dictated by project constraints and the existing design doc. |
| Features | HIGH | Table stakes defined by scaling limits (Small/Medium/Large). |
| Architecture | HIGH | Derived from the provided `fft-algo-design.pdf`. |
| Pitfalls | HIGH | Validated against hardware specs (256KB local mem, half-duplex DDR). |

## Gaps to Address

- Twiddle factor optimal packing strategies for 256KB local SRAM (needs exact bounds for Small vs Medium sizes).
- Multi-die interconnect latency variance for Large-scale 3D FFTs was not thoroughly characterized in the current scope.