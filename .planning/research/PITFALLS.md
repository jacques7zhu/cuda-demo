# Domain Pitfalls

**Domain:** 3D FFT on Compute-Storage Separated Custom Hardware
**Researched:** 2026-03-16

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Uniform Task Partitioning for Spheroid Geometry (Load Imbalance)
**What goes wrong:** Distributing 1D FFT "Pencils" evenly ($N/P$) across Compute APCs.
**Why it happens:** Assuming a dense, uniform cubic grid instead of a Spheroid. The bounding box's "edges" have far fewer non-zero elements than the center, creating a massive disparity in required FLOPs per pencil.
**Consequences:** Center APCs become bottlenecks while edge APCs sit idle. The overall pipeline stalls waiting for the slowest center cores to finish, drastically reducing hardware utilization.
**Prevention:** Implement Dynamic Binary Search + Greedy scheduling. Assign pencils based on the actual count of non-zero elements and performance cost models, grouping sparse pencils together to balance the compute load.
**Detection:** High variance in Compute APC (C-APC) active times; overall 1D FFT completion time matches the theoretical time of the most heavily loaded center C-APC.

### Pitfall 2: DDR Bandwidth Thrashing (All-Compute Anti-Pattern)
**What goes wrong:** Assigning all 32 APCs to be Compute APCs (C-APCs) that directly read/write from/to the 16GB DDR for every intermediate 3D FFT phase.
**Why it happens:** Attempting to maximize raw arithmetic throughput (FLOPs) while ignoring the "memory wall" of the high-latency, limited-bandwidth DDR bus.
**Consequences:** The system becomes severely memory-bound. Effective DDR bandwidth drops due to strided accesses during transpositions, and arithmetic units starve for data.
**Prevention:** Strictly enforce the **Compute-Storage Separation (CSS)** pattern. Reserve a portion of APCs as Storage APCs (S-APCs) to buffer data, handle X-to-Y transpositions locally over the high-speed 50GB/s APC2APC mesh, and hide DDR latency from the C-APCs.
**Detection:** Performance counters show C-APCs constantly stalling on memory loads; DDR utilization hits peak limits while ALUs are idle.

### Pitfall 3: In-Core Transpose Bank Conflicts
**What goes wrong:** C-APCs manually perform corner turns (X-to-Y transposition) directly within their small 256KB local SRAM.
**Why it happens:** Standard Cooley-Tukey FFTs require bit-reversal, and standard matrix transpositions require large power-of-two strided accesses.
**Consequences:** Strided accesses predictably map to the same physical SRAM memory banks, collapsing parallel vector/SIMD reads into serial accesses and causing massive local memory latency spikes.
**Prevention:** Use a Stockham Auto-Sorting FFT for the base 1D FFT to avoid bit-reversal memory patterns. Offload the transposition step entirely to S-APCs via the SDB bus, streaming sequential outputs instead of computing transposes in-place.
**Detection:** Unusually low IPC (Instructions Per Clock) during the FFT and transposition phases despite high arithmetic intensity; local memory subsystem metrics indicate severe bank conflicts.

## Moderate Pitfalls

### Pitfall 1: APC Local Memory Overflow during Slab Phase
**What goes wrong:** Loading full 2D "Slabs" (Y-Z planes) that exceed the combined 16MB local memory of the S-APC group.
**Why it happens:** Hardcoding slab sizes or attempting the "Independent Slab" phase regardless of the physical dimensions of the input dataset.
**Prevention:** Dynamically compute memory bounds based on the dataset size. If the 2D slab exceeds the APC cache boundary, gracefully fallback from the Small-scale "Independent Slab" (0 DDR hits) strategy to the Medium-scale "Pencil + Slab" (1 DDR hit) strategy.

### Pitfall 2: Half-Duplex DDR Synchronization Stalls
**What goes wrong:** Attempting to simultaneously read and write to the DDR memory, causing channel contention and severe bandwidth degradation.
**Why it happens:** Uncoordinated, asynchronous DMA engine firing between the DDR2APC (Load) and APC2DDR (Store) processes, ignoring the hardware's half-duplex DDR limitation.
**Prevention:** Carefully orchestrate double-buffering ping-pong states. Ensure that S-APC reads from DDR and writes to DDR are staggered or strictly interleaved with APC2APC local transfers to hide the half-duplex turnaround latency.

## Minor Pitfalls

### Pitfall 1: Redundant Twiddle Factor Computation
**What goes wrong:** Computing twiddle factors on the fly or redundantly fetching them from DDR for every pencil.
**Prevention:** Precompute and intelligently pack twiddle factors specific to the Spheroid domain constraints, caching them at the highest possible memory hierarchy to preserve the 256KB local C-APC memory.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **SCHED-01** (Load Balancing) | Static uniform partitioning causing severe stalls on Spheroid edges. | Use Binary Search + Greedy Allocation based on sparsity profiling. |
| **FFT3D-01** (Small Scale) | In-place transpose stalling C-APC local memory via bank conflicts. | Use Stockham FFT; stream sequential results to S-APCs for handling transposes over the SDB mesh. |
| **FFT3D-02** (Medium Scale) | Half-duplex DDR read/write channel contention. | Synchronize and interleave DDR2APC and APC2DDR DMA calls with internal APC2APC transfers. |
| **FFT3D-03** (Large Scale) | Out-of-core pipeline blocking indefinitely on disk I/O. | Utilize a fully chunked 3x Pencil strategy, entirely offloading chunk/disk management to S-APCs to hide I/O latency. |

## Sources

- `docs/fft-algo-design.pdf` (Compute-Storage Separation Optimization Design)
- Standard 3D FFT literature on FPGA/ASIC implementations (Memory wall, bank conflicts, transposition bottlenecks)