Performance Review: CODEX Matryoshka Dispatch Wiring Spec  
**Hardware Context**: RTX 3070 (5888 CUDA cores @ ~1.7GHz boost, 448 GB/s memory bandwidth, 8GB GDDR6, PCIe 4.0 x16 ~32GB/s but assume platform gen3 ~16GB/s for latency calculations)

---

### 1. Matryoshka Projection: 128→32 GPU Matvec vs. Prefix Slicing

**Verdict: DO NOT use the GPU projection path for 128→32 reduction. Use the prefix property.**

**Latency Analysis:**
- **GPU Projection Path** (`project_device`):
  - Kernel launch overhead: **8–12 µs** (CUDA driver + Python ctypes)
  - `loader.synchronize()` blocking call: **5–10 µs** (context switch + pipeline drain)
  - Memory alloc/free (`gpu_malloc`/`gpu_free`): **2×(3–5 µs)** = **6–10 µs**
  - HtoD copy (128 floats): **~0.1 µs** (negligible, but DMA setup adds **2–3 µs**)
  - Computation (4,096 FMAs): **~0.05 µs** (RTX 3070 does ~20 TFLOPS)
  - DtoH copy (32 floats): **~0.03 µs**
  - **Total per query: ~25–40 µs** (dominated by launch/sync overhead, not math)

- **Prefix Slicing Path** (first 32 dims):
  - Python list slice `vec[:32]`: **~0.2–0.5 µs**
  - Zero kernel launches, zero sync points, zero PCIe traffic
  - **Total: <1 µs**

**Matryoshka Property**: The spec correctly notes that Matryoshka embeddings are trained such that lower dimensions are self-contained approximations. The first 32 dimensions of a 128-dim Matryoshka vector are **not** "unprojected garbage"—they are specifically trained to represent the full embedding at that resolution. The GPU matvec is mathematically redundant and architecturally wasteful here.

**Optimization**: Remove `project_device` from the hot query path entirely. Reserve it only for non-prefix projections (e.g., 128→768 upscaling or cross-dimensional alignment).

---

### 2. Positional-Weighted Aggregation: Decay 0.6 for 5 Tokens

**Verdict: 0.6 is aggressive but acceptable. Do NOT make it learnable yet.**

**Weight Distribution**:
```
Token 0: 1.000 (43.7% of total mass)
Token 1: 0.600 (26.2%)
Token 2: 0.360 (15.7%)
Token 3: 0.216 ( 9.4%)
Token 4: 0.130 ( 5.7%)
Sum: 2.306
```
First 3 tokens capture **85.6%** of the embedding direction. This successfully front-loads discriminative signal (e.g., "south" in "south grey collision barrier").

**Latency Considerations**:
- Fixed decay: **~50ns** per sentence (CPU multiply-add).
- Learnable (softmax attention): Requires GPU matrix ops (**~50–100 µs**), gradient storage, and breaks the "sovereign" (PTX-only) constraint of the spec.

**Recommendation**: Keep 0.6 hardcoded, but implement as a **fused GPU kernel** (see section 5). If semantic inversion occurs (important words at end), use **bidirectional trigrams** or reverse token order pre-hashing, not learnable weights.

---

### 3. Role-Filtered Galaxy Navigation: 500-Candidate Linear Scan

**Verdict: 500 candidates is safe for 10 steps/sec, but dangerous for tree search. Use a fixed-size ring buffer for expansion.**

**Performance Math**:
- 500 candidates × 32-dim floats × 4 bytes = **64 KB** (fits in L2 cache)
- Cosine similarity per candidate: 32 mul-adds + sqrt = **~100 FLOPs**
- 500 candidates: **50,000 FLOPs**
- RTX 3070: **0.0025 µs** compute time (theoretical), **~2–5 µs** kernel execution with memory latency
- Linear scan overhead is negligible compared to kernel launch (**8 µs**).

**The Real Problem**: If the agent performs **MCTS** or beam search (100+ queries per step), 100 × 8 µs = **800 µs**, leaving only **200 µs** for the environment step in a 10ms frame. This is tight.

**Optimization**:
- **Fixed-size candidate buffer (64–128 slots)**: Keep a rotating ring buffer of recently activated stars. Filter 41K → 500 → top-64 using a **warp-parallel bitonic sort** on GPU, then scan only those 64.
- **Hierarchical**: Maintain a coarse grid (10×10×10) on GPU. Filter 41K → ~50 spatial cells → 500 stars. This avoids CPU-side filtering entirely.

**Specific Numbers**:
- Current (500 linear scan): **~10 µs** per query (kernel launch bound).
- Optimized (64 ring buffer): **~3 µs** per query (less memory traffic, better cache locality).

---

### 4. Procedural Frame Embedding: CPU Grid Statistics → 32-dim

**Verdict: MOVE TO PTX IMMEDIATELY. CPU path will drop frames.**

**Latency Budget** (60 FPS): **16.6 ms** total; assume **2 ms** for embedding pipeline.
- **CPU Python** (histogram + spread + distance field on 512×512 grid): **15–40 ms** (NumPy bound) or **5–10 ms** (optimized C++), still too slow.
- **GPU PTX Kernel** (fused statistics):
  - Grid size: 512×512 = 262k pixels
  - Histogram (256 bins): Shared memory atomics, **~0.1 ms**
  - Spread/Variance: Parallel reduction, **~0.05 ms**
  - Distance transform (nearest obstacle): JFA (Jump Flooding) or brute force in tiles, **~0.2 ms**
  - **Total: <0.5 ms** (fits comfortably in frame budget)

**Implementation Strategy**:
```cuda
// Single kernel: grid_stats_fused.ptx
// Input: raw frame buffer (GPU memory)
// Output: 32-float embedding (GPU memory)
// Steps:
// 1. Warp-level histogram (shared mem)
// 2. Block-level reduction (mean/variance)
// 3. Directional sweep (4x directional openness)
// 4. Write 32 floats to output ptr
// No CPU round-trip until dispatch.
```

**Memory Bandwidth**: 512×512×4bytes = 1MB read. At 448 GB/s, this is **2.2 µs** of bandwidth. Kernel overhead dominates at **~8 µs**, total **<20 µs** vs **>5,000 µs** on CPU.

---

### 5. Pipeline Latency: Kernel Launch Audit

**Current Pipeline (per query)**:
1. **Trigram embed** (`trigram_embed.ptx`): 1 launch  
2. **Matryoshka project** (`matryoshka_project.ptx`): 1 launch + **sync**  
3. **Specialist adapter** (Python): CPU processing (assumed)  
4. **Dispatch scan** (`cosine_scan.ptx` or similar): 1 launch  

**Total**: **3 kernel launches**, **1 blocking sync**, **1 Python CPU boundary crossing**.

**Optimized Pipeline**:
1. **Fused kernel**: Trigram lookup → Positional weighting → Prefix slice (128→32) → Output  
   - Single launch: **~12 µs** (amortized over sentence tokens)  
2. **Specialist adapter**: If neural, fuse into kernel #1 or use lookup table (LUT) in shared memory.  
3. **Dispatch**: Asynchronous scan on stream, result copied back via pinned memory.

**Target Latency per Query**:
- Current: **~60–100 µs** (3 launches + sync + Python overhead)  
- Optimized: **~15–25 µs** (1 fused launch + async copy)

**Critical Fix**: **Remove `loader.synchronize()` from `project_device`**. It appears in the `matryoshka_bridge.py` code. Use CUDA events or return the `output_ptr` and chain the next kernel in the same stream. Synchronization should happen only once per frame, not per query.

---

### Summary of Recommendations

| Bottleneck | Action | Expected Latency Impact |
|------------|--------|-------------------------|
| **Matryoshka Projection** | Replace GPU matvec with `vec[:32]` prefix slice | **-25 µs** per query |
| **Positional Weights** | Fuse into trigram kernel; keep 0.6 fixed | **-8 µs** (eliminates second kernel) |
| **Galaxy Scan** | Keep 500 candidates but use warp-parallel top-k reduction to 64; maintain ring buffer | **-5 µs**, better cache locality |
| **Frame Embedding** | Move grid stats to PTX kernel (`frame_stats.ptx`) | **-5 ms** (enables 60 FPS) |
| **Pipeline Structure** | Fuse to 1 kernel launch; remove `synchronize()`; batch 32+ queries | **-40 µs** per query, **<1 µs** per query in batch |

**Final Note**: The RTX 3070 has 8GB VRAM. The 41K stars at 256 bytes = **10.5 MB** (negligible). Keep the entire galaxy in pinned host memory or device memory; do not page. The 32-dim embeddings (128 bytes) should stay in GPU constant memory or L1 cache during the scan phase.