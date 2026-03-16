# Technology Stack

**Project:** Spheroid-3DFFT
**Researched:** 2026-03-16

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Proprietary CUDA-subset Toolchain | N/A | Core computing platform | Required by hardware constraints (no SIMT support, proprietary Custom Hardware) |
| C++ / Explicit Core Management API | N/A | Fine-grained kernel scheduling | Since there is no CUDA Graph or SIMT, explicit management of each core's local memory and execution flow is necessary. |

### Data Movement & Storage Strategies
| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Compute-Storage Separation (CSS) | To hide high DDR latency and manage limited on-chip memory. | Use constantly across the architecture. APCs are partitioned into Compute (C-APC) and Storage (S-APC) roles. S-APCs act as a user-managed software cache to buffer data from DDR to C-APCs. |
| Double Buffering (Ping-Pong Buffer) | To overlap compute and memory transfers. | Crucial for keeping C-APCs fed. While C-APC computes on Buffer A, S-APC uses DMA to fetch the next data block into Buffer B and writes the previous results from Buffer C back to DDR. |
| Global Transpose via SDB Mesh Network | Avoid redundant DDR roundtrips during 3D FFT transposes (x2y, y2z). | When performing intermediate 1D FFTs. Instead of writing back to DDR, C-APCs push transposed data directly through the high-bandwidth SDB bus (50 GB/s) to designated S-APCs. |
| Slab & Pencil Decompositions | Efficient spatial partitioning of 3D data. | Use Pencil decomposition for 1D FFTs along the X-axis (where the sparsity of the spheroid is highest). Use Slab decomposition for the subsequent 2D FFTs (Y-Z planes) distributed across APC groups. |

### Synchronization Primitives
| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Producer-Consumer Queues | Decoupling DDR fetching from computation. | Between S-APCs and C-APCs. S-APC pushes loaded chunks to C-APC's local memory queue; C-APC signals completion to unblock S-APC's write-back pipeline. |
| Local Memory Barriers | Ensure structural consistency inside the group. | Before transitioning from 1D FFTs (Pencil stage) to 2D FFTs (Slab stage), ensuring all grouped S-APCs have finished gathering the transposed data via the SDB bus. |

### Memory Bank Management & Local Store
| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Stockham Auto-Sorting FFT | Avoid bit-reversal operations and minimize bank conflicts. | For the base 1D FFT kernel. Stockham relies on ordered strided accesses, fitting naturally into wide 512-bit registers/SIMD lanes and reducing random local-memory access overheads associated with Cooley-Tukey. |
| DMA Scatter-Gather (DMA Lists) | Handling the irregular bounds of the "Spheroid" geometry. | When loading non-contiguous "pencils" from DDR. It reduces unused memory fetching overhead for empty regions outside the spheroid. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Compute Model | Compute-Storage Separation (CSS) | All-Compute (All APCs execute FFT) | Evaluated in docs, All-Compute is memory-bound and thrashes DDR. CSS reduced communication time by ~32% for small systems by maximizing continuous DDR bursts and dedicating specific APCs to SDB/DDR routing. |
| Load Balancing | Binary Search + Greedy Allocation | Uniform Block Distribution | Spheroid boundaries create sparse pencils with varying non-zero elements. Uniform distribution leads to massive C-APC idling. Greedy allocation clusters variable-length pencils dynamically based on performance models. |
| FFT Algorithm | Stockham FFT | Cooley-Tukey FFT | Cooley-Tukey demands in-place bit-reversal permutations, which causes high local-memory latency and bank conflicts on non-SIMT SIMD architectures (like the Cell B.E. or this custom HW). |

## Implementation Notes

- **Network Constraints**: DDR is half-duplex (256 GB/s total, 22.4 GB/s effective per group). The SDB bus (APC2APC) operates at 50.11 GB/s. Transposes *must* be handled within the SDB mesh when the slab fits in S-APC local memory limits (which is total ~5MB per group, leaving ~16MB slab plane constraints).
- **Out-of-Core Processing**: For large systems (> DDR capacity), the algorithm must swap out entirely to an external drive. S-APCs manage the disk-to-DDR-to-S-APC pipeline through chunked Slab iterations.
- **S-APC / C-APC Ratio**: Default grouping is 1 C-APC to 3 S-APCs (per Die of 32 APCs = 8 compute groups). The system can dynamically shift to more C-APCs if the kernel becomes compute-bound rather than memory-bound.

## Sources

- `docs/fft-algo-design.pdf` (Compute-Storage Separation architecture, SDB vs DDR bandwidth ratios, Binary Search load balancing, Spheroid pencil sparsity logic)
- Architectural standards derived from analogous high-bandwidth, local-memory constrained architectures (e.g., Cell B.E. / TPU DMA & Double Buffering patterns)
