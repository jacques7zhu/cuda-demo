# Architecture Research

**Domain:** 3D FFT on Compute-Storage Separated Custom Hardware
**Researched:** 2026-03-16
**Confidence:** HIGH

## Standard Architecture

The architecture for the 3D FFT (Spheroid) relies on **Compute-Storage Separation (CSS)** to overcome the memory wall (DDR bandwidth and latency limitations). It implements a high-performance streaming pipeline by decoupling mathematical computation from data transposition/reordering.

### System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                           DDR Memory                        │
│                     (16GB, High Latency)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │   DDR2APC (Load)        │   │   APC2DDR (Store)       │  │
│  └──────────┬──────────────┘   └───────────▲─────────────┘  │
│             │                              │                │
├─────────────▼──────────────────────────────┴────────────────┤
│                    APC Cluster (In-Core)                    │
│                                                             │
│  ┌────────────────┐     ┌───────────────┐   ┌────────────┐  │
│  │ Phase A:       │     │ Reorder:      │   │ Phase B:   │  │
│  │ Compute APCs   ├────►│ Storage APCs  ├──►│ Compute    │  │
│  │ (1D FFT Pencil)│     │ (x2y Transpos)│   │ (2D FFT YZ)│  │
│  └────────────────┘     └───────────────┘   └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Compute APC (C-APC)** | Fast mathematical operations (1D/2D FFT). Strict coalesced memory access requirements. | Specialized arithmetic cores with small cache (1MB). Executes pure math kernels. |
| **Storage APC (S-APC)** | Data staging, reordering, and masking latency. Transforms column/depth-major data to row-major. | Staging buffers with larger cache (5MB). Executes x2y matrix transpose kernels. |
| **Pencil FFT (Phase A)**| Resolves 1D FFT along the X-dimension. | Divides domain into 1D "Pencils". Load balanced via binary search & greedy algos. |
| **Slab FFT (Phase B)**  | Resolves 2D FFT along the YZ-plane. | Divides domain into 2D "Slabs" (cylinders). Typically uniform distribution. |

## Recommended Project Structure

```text
src/
├── fft/                  # Core 3D FFT Implementation
│   ├── src/fft.cu        # Main orchestrator (Small/Medium/Large scale policies)
│   ├── src/pencil.cu     # 1D FFT Pencil logic executed on C-APCs
│   ├── src/slab.cu       # 2D FFT Slab logic executed on C-APCs
│   └── src/reorder.cu    # Data transposition (x2y) executed on S-APCs
├── common/               # Shared utilities
│   ├── src/scheduler.cu  # Dynamic task allocation (Binary search + greedy)
│   └── inc/layout.h      # Optimal data layout and twiddler factors for Spheroid
└── io/                   # Memory transfer layer
    ├── src/ddr2apc.cu    # Asynchronous DMA loading from DDR
    └── src/apc2ddr.cu    # Asynchronous DMA storing to DDR
```

### Structure Rationale

- **fft/:** Separating `pencil.cu`, `slab.cu`, and `reorder.cu` maps cleanly to the Compute-Storage Separation paradigm. Each kernel is targeted to its respective hardware unit (C-APC vs S-APC).
- **common/scheduler.cu:** Spheroid data has non-uniform sparsity. A dedicated scheduler module is needed to decouple load balancing logic from the math kernels.
- **io/:** Isolate all DDR interactions. This allows the system to tune DMA burst sizes and memory alignment independently of the FFT logic.

## Architectural Patterns

### Pattern 1: Compute-Storage Separation (CSS)

**What:** Decoupling mathematical operations (FFT) from data movement and memory transformations (Transpose). S-APCs act as a programmable memory hierarchy between DDR and C-APCs.
**When to use:** When the performance bottleneck is memory bandwidth (DDR limits) rather than raw compute FLOPS.
**Trade-offs:** Maximizes math core utilization and achieves perfect memory coalescing, but reduces the absolute number of cores available for calculation (e.g., dedicating 3 APCs to storage for every 1 APC computing).

### Pattern 2: Pencil-to-Slab Pipelining

**What:** Performing 1D FFTs on thin 1D arrays ("Pencils") along X, transposing in S-APCs, and then feeding thick 2D slices ("Slabs") along YZ into the next compute phase.
**When to use:** Ideal for in-core or specialized multi-die custom architectures where full all-to-all communication (3x Pencil) is too expensive, but simple 1D Slab decomposition lacks enough parallelism.
**Trade-offs:** Highly parallel and efficient memory-wise. However, it requires careful synchronization between the Pencil output and Slab input phases.

### Pattern 3: Dynamic Spheroid Load Balancing (Binary Search + Greedy)

**What:** Because the dataset is a "spheroid", edge pencils contain much less data than center pencils. A static division causes severe idle time. Tasks are dynamically distributed based on actual workload metrics using binary search over cost functions.
**When to use:** Whenever the geometry of the data domain leads to sparse sub-regions (like a spheroid inscribed in a cube).
**Trade-offs:** Adds minor scheduling overhead at the start of the pipeline but prevents massive tail-latency delays from heavily-loaded center processors.

## Data Flow

### 3D FFT Pipeline Flow

```text
[DDR Data]
    ↓ (ddr2apc)
[C-APC: Phase A] → Computes 1D FFT on X-Pencils
    ↓ (apc2apc)
[S-APC: Reorder] → Transposes sparse data (x2y), aligns to dense cylinders
    ↓ (apc2apc)
[C-APC: Phase B] → Computes 2D FFT on YZ-Slabs
    ↓ (apc2ddr)
[DDR Results]
```

## Scaling Considerations

Depending on the problem size relative to the APC cache (120MB boundary) and DDR limit (16GB/2GB usable limit):

| Scale | Definition | Architecture Adjustments |
|-------|------------|--------------------------|
| **Small** | Total Data < 120MB | **Pencil + Independent Slab:** Fits entirely in APC cache. Zero intermediate DDR hits. Maximum speed. |
| **Medium** | Total > 120MB,<br>Plane < 16MB | **Pencil + Slab:** Fits in DDR but not APC. Requires exactly 1 intermediate DDR read/write. Reordering happens in DDR/S-APC combination. |
| **Large** | Total > 2GB (Out-of-core) | **3x Pencil passes:** Total dataset exceeds DDR. Requires external storage/multi-die strategies. Switches to pure out-of-core streaming algorithms. |

### Scaling Priorities

1. **First bottleneck (APC Cache Misses):** The system breaks when the Slab (YZ plane) exceeds APC memory. Addressed by switching from Independent Slab to cooperative multi-pass Slab, or ultimately to 3x Pencils.
2. **Second bottleneck (DDR Bandwidth):** Addressed by dynamically adjusting the ratio of Compute APCs vs Storage APCs depending on whether the workload is currently compute-bound or memory-bound.

## Anti-Patterns

### Anti-Pattern 1: In-Place Transpose on Compute Cores
**What people do:** Using the same processor cores to compute the 1D FFT and then manually shuffling data for the next dimension within the same kernel.
**Why it's wrong:** Forces the math-optimized C-APCs to wait on strided, non-coalesced memory accesses, destroying performance.
**Do this instead:** Stream outputs to the S-APCs and let the S-APCs handle the X-to-Y transposition concurrently with the next FFT.

### Anti-Pattern 2: Uniform Task Partitioning for Spheroids
**What people do:** Dividing the 3D volume into equal $N/P$ chunks (e.g., standard MPI chunking).
**Why it's wrong:** A Spheroid has empty/zero-padded regions at the corners of the grid. Central cores will take 3x longer than edge cores, bottlenecking the entire pipeline.
**Do this instead:** Use the Binary Search + Greedy assignment based on actual non-zero data density to achieve near-perfect workload balancing.

## Sources

- `docs/fft-algo-design.pdf` (Project documentation on Compute-Storage Separation Optimization)
- `P3DFFT` and industry standard 2D-Decomposition (Pencil) best practices
- Best Practices for GPU Data Transposition and Memory Coalescing

---
*Architecture research for: Spheroid 3D FFT on Compute-Storage Separated Custom Hardware*
*Researched: 2026-03-16*
