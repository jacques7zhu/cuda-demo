# Project State: Spheroid-3DFFT

## Project Reference
**Core Value**: High-performance 3D FFT on spheroid data using Compute-Storage Separation (CSS) on custom hardware.
**Current Focus**: Phase 1: Spheroid Layout & Mathematical Foundation

## Current Position
- **Phase**: 1
- **Plan**: TBD
- **Status**: Initialization Complete
- **Progress**: [░░░░░░░░░░░░░░░░░░░░] 0%

## Performance Metrics
- **DDR Bandwidth Utilization**: N/A
- **Compute Stall Ratio**: N/A
- **Algorithm Error ($L_2$)**: N/A

## Accumulated Context

### Key Decisions
- **Compute-Storage Separation**: Essential to bypass DDR latency; C-APCs for math, S-APCs for routing.
- **Dynamic Load Balancing**: Required due to spheroid sparsity; avoids center-core synchronization hotspots.
- **Slab-wise Decomposition**: Used for Medium-scale data to minimize DDR passes.

### Important Technical Details
- **Local Memory**: 256KB per core (C-APC).
- **On-chip Cache**: 120MB total (across all APCs).
- **DDR**: 16GB total capacity.
- **Mesh Interconnect**: 50GB/s APC2APC (SDB bus).

## Session Continuity
- **Last Action**: Roadmap and State initialized.
- **Next Step**: Start Planning for Phase 1 (`/gsd:plan-phase 1`).
- **Open Questions**: Finalize exact twiddle factor packing layout for mixed-radix support.
