# Roadmap: Spheroid-3DFFT

This roadmap outlines the development of a high-performance 3D FFT library for custom GPU hardware with Compute-Storage Separation (CSS).

## Phases

- [ ] **Phase 1: Spheroid Layout & Mathematical Foundation** - Precompute optimal data layouts and twiddle factors for spheroid geometry.
- [ ] **Phase 2: Dynamic Load Balancing** - Implement the cost-modeled scheduler for spheroid pencil assignment.
- [ ] **Phase 3: Small-Scale CSS Pipeline (On-Chip)** - Complete 3D FFT for data fitting in 120MB APC memory using mesh transpositions.
- [ ] **Phase 4: Medium-Scale DDR Orchestration (Slab-wise)** - Extend the pipeline to stream data from DDR using slab-wise decomposition.
- [ ] **Phase 5: Performance Tuning & Verification** - Final optimizations and throughput validation against the performance model.

## Phase Details

### Phase 1: Spheroid Layout & Mathematical Foundation
**Goal**: Precompute optimal data layouts and twiddle factors for spheroid geometry.
**Depends on**: Project Initialization
**Requirements**: LAYOUT-01, FFT-02
**Success Criteria**:
  1. Mixed-radix (Radix-2, 3, 5) twiddle factors are generated and packed within 256KB local SRAM limits.
  2. Spheroid bounding box and sparsity map are calculated accurately for arbitrary input radii.
  3. Precomputation time is negligible (< 10ms) for target volume dimensions.
**Plans**: TBD

### Phase 2: Dynamic Load Balancing
**Goal**: Implement the cost-modeled scheduler for spheroid pencil assignment.
**Depends on**: Phase 1
**Requirements**: SCHED-01, OPT-01
**Success Criteria**:
  1. Scheduler assigns 100% of non-zero pencils to available SPs (32 SPs, 128 cores total).
  2. Load variance between compute cores is < 5% for highly sparse spheroid data.
  3. Binary Search + Greedy algorithm converges within 1ms for $512^3$ volume.
**Plans**: TBD

### Phase 3: Small-Scale CSS Pipeline (On-Chip)
**Goal**: Complete 3D FFT for data fitting in 120MB APC memory using mesh transpositions.
**Depends on**: Phase 2
**Requirements**: FFT-01, FFT-03, SCALE-01, CSS-01, XPOSE-01, REORDER-01
**Success Criteria**:
  1. Forward and inverse C2C 3D FFT produces results with $< 10^{-6}$ error compared to CPU reference.
  2. Data transposes (x2y) occur over APC2APC mesh without DDR roundtrips.
  3. Compute cores (C-APCs) never stall waiting for Storage cores (S-APCs) for in-cache data.
  4. Spheroid-to-grid reordering is implemented and verified.
**Plans**: TBD

### Phase 4: Medium-Scale DDR Orchestration (Slab-wise)
**Goal**: Extend the pipeline to handle up to 2GB data using DDR streaming.
**Depends on**: Phase 3
**Requirements**: SCALE-02, DECOMP-01
**Success Criteria**:
  1. 3D FFT on $1024^3$ volume completes using exactly 1 DDR pass (forward) and 1 DDR pass (inverse).
  2. DMA transfers are overlapped with compute, achieving $> 80\%$ DDR bandwidth utilization.
  3. Hybrid decomposition (Pencils for X, Slabs for YZ) successfully processes data exceeding 120MB.
**Plans**: TBD

### Phase 5: Performance Tuning & Verification
**Goal**: Final optimizations and throughput validation against the performance model.
**Depends on**: Phase 4
**Requirements**: (Final verification of all)
**Success Criteria**:
  1. Total execution time meets the performance model in `fft-algo-design.pdf` (approx. 3.2ms for $512^3$).
  2. Memory usage stays strictly within 16GB DDR and 256KB local limits across all test cases.
  3. Library passes stress tests with varying spheroid eccentricities and rotation angles.
**Plans**: TBD

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Spheroid Layout & Foundation | 0/1 | Not started | - |
| 2. Dynamic Load Balancing | 0/1 | Not started | - |
| 3. Small-Scale CSS Pipeline | 0/1 | Not started | - |
| 4. Medium-Scale DDR Orchestration | 0/1 | Not started | - |
| 5. Performance Tuning & Verification | 0/1 | Not started | - |
