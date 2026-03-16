# v1 Requirements: Spheroid-3DFFT

## Core Mathematical Transforms
- [ ] **FFT-01**: Support 3D C2C (complex-to-complex) transforms.
- [ ] **FFT-02**: Implement mixed-radix support (Radix-2, 3, and 5) to handle various dimensions.
- [ ] **FFT-03**: Support both forward and inverse 3D FFT variants.

## Scale Support
- [ ] **SCALE-01**: Small-Scale Support: Full computation within 120MB on-chip APC memory (0 DDR hits).
- [ ] **SCALE-02**: Medium-Scale Support: Handles data up to 2GB in DDR, using slab-wise processing (1 DDR pass).

## Performance & Optimization (CSS)
- [ ] **CSS-01**: Implement Compute-Storage Separation (CSS) logic: C-APCs for math, S-APCs for caching/reordering.
- [ ] **OPT-01**: Spheroid-Aware Sparse Processing: Skip computations for zero-padded regions outside the spheroid bounding box.
- [ ] **SCHED-01**: Dynamic Load Balancing: Binary Search + Greedy algorithm to assign pencils based on compute/transfer costs.
- [ ] **XPOSE-01**: In-SRAM Transposition (x2y): Leverage S-APCs and SDB bus for transposing data without DDR roundtrips.
- [ ] **DECOMP-01**: Hybrid Task Decomposition: Use "Pencil" for X-direction transforms and "Slab" for YZ-plane transforms.

## Spheroid Specifics (Synchronized with PROJECT.md)
- [ ] **REORDER-01**: Spheroid-to-grid reordering (forward and inverse).
- [ ] **LAYOUT-01**: Optimal data layout and twiddler factor design for spheroid.

## v2 Requirements (Deferred)
- [ ] **SCALE-03**: Large-Scale Support: Out-of-core handling for datasets > 2GB using multi-die/disk swapping (2+ DDR hits).

## Out of Scope
- **FFT-GENERIC**: General-purpose Cartesian 3D FFT (Library is specialized for spheroid data).
- **GPU-GENERIC**: Support for standard NVIDIA/AMD GPUs (Project is custom hardware only).
- **SIMT-AUTO**: Reliance on SIMT execution or auto-vectorization (Hardware uses explicit core control).

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FFT-01 | Phase 3 | Pending |
| FFT-02 | Phase 1 | Pending |
| FFT-03 | Phase 3 | Pending |
| SCALE-01 | Phase 3 | Pending |
| SCALE-02 | Phase 4 | Pending |
| CSS-01 | Phase 3 | Pending |
| OPT-01 | Phase 2 | Pending |
| SCHED-01 | Phase 2 | Pending |
| XPOSE-01 | Phase 3 | Pending |
| DECOMP-01 | Phase 4 | Pending |
| REORDER-01 | Phase 3 | Pending |
| LAYOUT-01 | Phase 1 | Pending |
