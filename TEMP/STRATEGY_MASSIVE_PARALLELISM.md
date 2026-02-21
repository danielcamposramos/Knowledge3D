# Strategy: Massive Parallelism - Unleash the 12GB Beast 🚀

## Current State: MASSIVE UNDERUTILIZATION

```
GPU: NVIDIA GeForce RTX 3060 (12GB)
Current usage during consolidation:
  - GPU Utilization: 8%      ← 92% IDLE!
  - VRAM: 127 MiB / 12288 MiB ← 1% used, 99% FREE!
  - Compute: Sequential       ← Single-threaded orchestration
```

**WE ARE LEAVING 99% OF THE GPU ON THE TABLE!** 🔥

---

## Vision: Hybrid Parallelism (Like the 15 RPN Stacks!)

### User's Key Insight
> "I love hybrid parallelism, like the 15 RPN stacks"

**The 15 RPN stacks** are the blueprint:
- 15 instances running in parallel
- Each instance: independent state, 64-deep stack
- All instances share same kernel
- Perfect for SIMD-style parallelism

**Apply this to EVERYTHING**:
- 15 similarity computations in parallel (not sequential!)
- 15 chunk computations in parallel (not one-by-one!)
- 15 embedding extractions in parallel
- 15 consolidation sub-tasks in parallel

---

## Architecture: Multi-Level Parallelism

### Level 1: Intra-Kernel Parallelism (GPU Threads)
**Current**: 1 block, 1 thread per kernel launch
**Target**: 256 threads per block, multiple blocks

```cuda
// BEFORE: Single-threaded
__global__ void compute_dot3(float3 a, float3 b, float* result) {
    if (threadIdx.x == 0) {  // Only thread 0 works!
        *result = a.x * b.x + a.y * b.y + a.z * b.z;
    }
}

// AFTER: Massively parallel
__global__ void compute_dot3_batch(
    const float3* a_batch,  // 1024 vectors
    const float3* b_batch,
    float* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        float3 a = a_batch[idx];
        float3 b = b_batch[idx];
        results[idx] = a.x * b.x + a.y * b.y + a.z * b.z;
    }
}
// Launch: <<<4, 256>>> = 1024 threads in parallel!
```

### Level 2: Inter-Kernel Parallelism (CUDA Streams)
**Current**: Sequential kernel launches
**Target**: 15 concurrent CUDA streams (like 15 RPN instances!)

```python
# BEFORE: Sequential (slow)
for chunk in chunks:
    result = launch_kernel(chunk)  # Wait for each
    accumulate(result)

# AFTER: Parallel streams (fast)
streams = [create_stream() for _ in range(15)]
chunk_batches = split_into_batches(chunks, batch_size=15)

for batch in chunk_batches:
    # Launch 15 kernels simultaneously!
    for i, chunk in enumerate(batch):
        launch_kernel_async(chunk, stream=streams[i])

    # Synchronize streams
    for stream in streams:
        stream.synchronize()

    # Accumulate batch results
    accumulate_batch(batch_results)
```

### Level 3: Task-Level Parallelism (RPN Instances)
**Current**: Single RPN instance for all computations
**Target**: 15 RPN instances processing different tasks

```python
# BEFORE: Single instance
for i in range(1000):
    result = rpn_executor.execute_single(instance_id=0, ...)

# AFTER: 15 instances in parallel
rpn_executor = get_rpn_executor()
tasks = [(i, data[i]) for i in range(1000)]
task_batches = split_into_batches(tasks, batch_size=15)

for batch in task_batches:
    # Execute 15 tasks simultaneously on 15 RPN instances!
    results = rpn_executor.execute_batch([
        {'instance_id': i, 'op_codes': ..., 'data': data}
        for i, data in batch
    ])
```

---

## Implementation Plan

### Quick Win 1: Batch Chunk Processing (1-2 hours)

**Target**: 10-43x speedup for cosine similarity

**Current flow** (128D cosine similarity):
```
for chunk in 43 chunks:           # Sequential
    gpu_dot3(u_chunk, v_chunk)    # 1 kernel launch
    accumulate(result)            # CPU
# Total: 43 kernel launches × 3 operations = 129 launches
# Time: ~47s for 100 embeddings
```

**Optimized flow**:
```
# Split 43 chunks into 3 batches of 15
batch1 = chunks[0:15]   # 15 chunks
batch2 = chunks[15:30]  # 15 chunks
batch3 = chunks[30:43]  # 13 chunks

# Launch 15 kernels in parallel per batch
for batch in [batch1, batch2, batch3]:
    results = gpu_dot3_batch(batch)  # 15 parallel launches
    accumulate(results)              # Batch accumulate

# Total: 3 batches × 3 operations = 9 batch launches
# Expected time: ~3-5s for 100 embeddings (10x faster!)
```

**Implementation**:
```python
# knowledge3d/cranium/clustering_rpn.py

def compute_cosine_similarity_rpn_batched(vec_u, vec_v):
    """Batched chunking with parallel GPU execution."""
    executor = get_rpn_executor()
    dim = len(vec_u)
    chunk_size = 3
    num_chunks = (dim + chunk_size - 1) // chunk_size

    # Prepare all chunks upfront
    chunks_u = []
    chunks_v = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, dim)
        u_padded = np.zeros(3, dtype=np.float32)
        v_padded = np.zeros(3, dtype=np.float32)
        u_padded[:end-start] = vec_u[start:end]
        v_padded[:end-start] = vec_v[start:end]
        chunks_u.append(u_padded)
        chunks_v.append(v_padded)

    # Process in batches of 15 (leverage 15 RPN instances!)
    batch_size = 15
    dot_product = 0.0
    norm_u_sq = 0.0
    norm_v_sq = 0.0

    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_chunks_u = chunks_u[batch_start:batch_end]
        batch_chunks_v = chunks_v[batch_start:batch_end]

        # Prepare batch programs
        programs_uv = []
        programs_uu = []
        programs_vv = []

        for u, v in zip(batch_chunks_u, batch_chunks_v):
            op_codes = np.array([0x01, 0x01, 0x3C], dtype=np.uint16)
            scalars = np.zeros(1, dtype=np.float32)

            programs_uv.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([u, v])
            })
            programs_uu.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([u, u])
            })
            programs_vv.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': np.concatenate([v, v])
            })

        # Execute batches in parallel!
        results_uv = executor.execute_batch(programs_uv, max_instances=batch_size)
        results_uu = executor.execute_batch(programs_uu, max_instances=batch_size)
        results_vv = executor.execute_batch(programs_vv, max_instances=batch_size)

        # Accumulate
        dot_product += sum(results_uv)
        norm_u_sq += sum(results_uu)
        norm_v_sq += sum(results_vv)

    # Final cosine
    norm_u = np.sqrt(norm_u_sq)
    norm_v = np.sqrt(norm_v_sq)

    if norm_u < 1e-8 or norm_v < 1e-8:
        return 0.0

    return float(np.clip(dot_product / (norm_u * norm_v), -1.0, 1.0))
```

**Expected speedup**: 10-15x (47s → 3-5s for 100 embeddings)

---

### Quick Win 2: CUDA Streams for Async Execution (2-3 hours)

**Target**: Overlap computation and memory transfer

**Add to sovereign/loader.py**:
```python
# knowledge3d/cranium/sovereign/loader.py

def create_stream() -> int:
    """Create CUDA stream for async execution."""
    stream = ctypes.c_void_p()
    result = _libcuda.cuStreamCreate(ctypes.byref(stream), 0)
    if result != 0:
        raise RuntimeError(f"cuStreamCreate failed: {result}")
    return stream.value

def destroy_stream(stream: int) -> None:
    """Destroy CUDA stream."""
    _libcuda.cuStreamDestroy(ctypes.c_void_p(stream))

def launch_async(func, grid, block, params, stream: int) -> None:
    """Launch kernel asynchronously on stream."""
    _prepare_params(params)
    result = _libcuda.cuLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0,  # shared memory
        ctypes.c_void_p(stream),  # Use stream!
        _kernel_params,
        None
    )
    if result != 0:
        raise RuntimeError(f"cuLaunchKernel failed: {result}")

def stream_synchronize(stream: int) -> None:
    """Wait for stream to complete."""
    result = _libcuda.cuStreamSynchronize(ctypes.c_void_p(stream))
    if result != 0:
        raise RuntimeError(f"cuStreamSynchronize failed: {result}")
```

**Usage in executor**:
```python
# knowledge3d/cranium/sovereign_rpn_executor.py

class SovereignRPNExecutor:
    def __init__(self):
        # ... existing init ...
        self.streams = [loader.create_stream() for _ in range(15)]

    def execute_batch_async(self, programs, max_instances=15):
        """Execute batch with async streams."""
        results = []

        for i, program in enumerate(programs):
            stream_idx = i % 15
            instance_id = i % max_instances

            # Launch async on stream
            loader.launch_async(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[...],
                stream=self.streams[stream_idx]
            )

        # Synchronize all streams
        for stream in self.streams[:min(len(programs), 15)]:
            loader.stream_synchronize(stream)

        # Read results
        # ...

        return results
```

**Expected benefit**: 20-30% additional speedup from overlapped execution

---

### Quick Win 3: GPU Memory Pooling (1 hour)

**Target**: Reduce allocation overhead

**Current**: Allocate/free GPU memory for every kernel launch
**Problem**: GPU malloc/free has ~10-50μs overhead
**Solution**: Pre-allocate memory pool, reuse buffers

```python
# knowledge3d/cranium/sovereign/memory_pool.py

class GPUMemoryPool:
    """Pre-allocated GPU memory pool for fast reuse."""

    def __init__(self, pool_size_mb: int = 512):
        self.pool_size = pool_size_mb * 1024 * 1024
        self.pool_ptr = loader.gpu_malloc(self.pool_size)
        self.allocations = []
        self.free_list = [(0, self.pool_size)]  # (offset, size)

    def allocate(self, size_bytes: int) -> int:
        """Allocate from pool (fast)."""
        # Find free block (first-fit)
        for i, (offset, free_size) in enumerate(self.free_list):
            if free_size >= size_bytes:
                # Allocate from this block
                ptr = self.pool_ptr + offset
                self.allocations.append((ptr, size_bytes))

                # Update free list
                remaining = free_size - size_bytes
                if remaining > 0:
                    self.free_list[i] = (offset + size_bytes, remaining)
                else:
                    del self.free_list[i]

                return ptr

        raise RuntimeError(f"Out of GPU memory in pool (requested {size_bytes})")

    def free(self, ptr: int) -> None:
        """Free back to pool (fast)."""
        # Find allocation
        for i, (alloc_ptr, size) in enumerate(self.allocations):
            if alloc_ptr == ptr:
                offset = ptr - self.pool_ptr
                # Add to free list (with coalescing)
                self._coalesce_free_block(offset, size)
                del self.allocations[i]
                return

        raise RuntimeError(f"Invalid free: {ptr} not in pool")

    def __del__(self):
        loader.gpu_free(self.pool_ptr)
```

**Integration**:
```python
# Global memory pool
_gpu_pool = None

def get_gpu_pool():
    global _gpu_pool
    if _gpu_pool is None:
        _gpu_pool = GPUMemoryPool(pool_size_mb=512)  # 512 MB pool
    return _gpu_pool
```

**Expected benefit**: 50-100x faster allocation (50μs → 0.5μs)

---

### Medium Win 1: Extended Kernel Batch Ops (1-2 days)

**Target**: Use 0xC4 COSINE_SIM_BATCH opcode for entire similarity matrix

**Current**: 43 chunks × 3 calls × 1000 pairs = 129,000 kernel launches
**Target**: 1 kernel launch for entire matrix!

The extended kernel ALREADY has this implemented:
```cuda
case 0xC4: {  // COSINE_SIM_BATCH (dest_matrix, vectors, centroids, n_vectors, n_centroids, dim)
    // Computes full similarity matrix in ONE kernel!
    for (int i = 0; i < n_vectors_i; ++i) {
        for (int j = 0; j < n_centroids_i; ++j) {
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < dim_i; ++d) {  // Handles arbitrary dim!
                dot += v_ptr[i * dim_i + d] * c_ptr[j * dim_i + d];
            }
            sims[i * n_centroids_i + j] = dot;
        }
    }
}
```

**Implementation needed**:
1. GPU tensor allocator (allocate vectors, centroids, result matrix)
2. Wrapper function to call 0xC4 opcode
3. Integration with clustering_rpn.py

**Expected speedup**: 100-1000x (47s → 0.05-0.5s!)

---

### Medium Win 2: Multi-GPU Support (2-3 days)

**Current**: Single GPU (RTX 3060 12GB)
**User mentioned**: Testing on 12GB 3060, goal is 3GB min, max 8GB consumer

**Strategy**: Data parallelism across multiple GPUs
- GPU 0: Embeddings 0-500
- GPU 1: Embeddings 501-1000
- Each GPU: Independent consolidation
- Final: Merge results

**For users with 2-4 GPUs**: Near-linear speedup!

---

## Optimization Roadmap

### Phase 1: Quick Wins (1 day)
**Target**: 10-15x speedup with minimal changes

1. ✅ Implement batched chunk processing (15 chunks in parallel)
2. ✅ Add CUDA streams for async execution
3. ✅ Implement GPU memory pool
4. ✅ Benchmark consolidation: expect 47s → 3-5s

### Phase 2: Extended Kernel (2-3 days)
**Target**: 100-1000x speedup with major refactor

1. ✅ Implement GPU tensor allocator
2. ✅ Create wrapper for 0xC4 COSINE_SIM_BATCH
3. ✅ Integrate with sovereign_clustering_ops
4. ✅ Benchmark: expect 47s → 0.05-0.5s

### Phase 3: Advanced Parallelism (1 week)
**Target**: Maximum GPU utilization

1. ✅ Multi-GPU data parallelism
2. ✅ Kernel fusion (combine multiple ops)
3. ✅ Dynamic work scheduling
4. ✅ Benchmark on production scale (10K embeddings)

---

## Expected Performance

### Current (Sequential)
```
100 embeddings → 10 clusters:
  - Time: 47.63s
  - GPU util: 8%
  - VRAM: 127 MiB

10,000 embeddings → 256 clusters:
  - Time: ~34 hours (estimated)
  - GPU util: 8%
  - VRAM: ~1 GB
```

### After Quick Wins (Batched)
```
100 embeddings → 10 clusters:
  - Time: ~3-5s (10x faster) ✅
  - GPU util: 40-60%
  - VRAM: 200-300 MiB

10,000 embeddings → 256 clusters:
  - Time: ~3-4 hours (10x faster)
  - GPU util: 40-60%
  - VRAM: ~2 GB
```

### After Extended Kernel (0xC4)
```
100 embeddings → 10 clusters:
  - Time: ~0.05-0.1s (500x faster!) ✅
  - GPU util: 80-90%
  - VRAM: 500 MiB

10,000 embeddings → 256 clusters:
  - Time: ~2-5 minutes (400x faster!)
  - GPU util: 80-90%
  - VRAM: ~4 GB
```

### After Multi-GPU (4x RTX 3060)
```
10,000 embeddings → 256 clusters:
  - Time: ~30-60 seconds (2000x faster!)
  - GPU util: 80-90% per GPU
  - VRAM: ~4 GB per GPU
```

---

## Hybrid Parallelism Levels (Like 15 RPN Stacks!)

### Level 1: Thread Parallelism (GPU Core Level)
- 256 threads per block
- 32 threads per warp (SIMD)
- 3584 CUDA cores (RTX 3060)
- **Target**: 90%+ core utilization

### Level 2: Block Parallelism (Streaming Multiprocessor Level)
- Multiple blocks per kernel
- 28 SMs (RTX 3060)
- **Target**: All SMs active

### Level 3: Stream Parallelism (Kernel Level)
- 15 CUDA streams (like 15 RPN instances!)
- Async kernel launches
- Overlapped execution
- **Target**: 15 kernels in flight

### Level 4: Instance Parallelism (Task Level)
- 15 RPN instances
- Independent computations
- Batch execution
- **Target**: 15 tasks simultaneously

### Level 5: GPU Parallelism (Device Level)
- Multi-GPU data parallelism
- 2-4 GPUs per machine
- Near-linear scaling
- **Target**: 4x speedup with 4 GPUs

---

## Memory Budget (12 GB Available)

### Current Usage: 127 MiB (1%)

**Available headroom**: 12,161 MiB (99%!)

### Proposed Allocation
```
Memory Pool:           512 MB  (pre-allocated buffers)
RPN State:              16 MB  (15 instances × 1 MB each)
Embedding Cache:     2,048 MB  (hot embeddings)
Similarity Matrix:   1,024 MB  (temporary computations)
Sound Pictures:      2,048 MB  (spectrogram cache)
Tensor Workspace:    2,048 MB  (0xC4 opcode operations)
Reserve:             4,525 MB  (safety margin)
-------------------------------------------
Total:              12,221 MB  (FITS! With reserve)
```

**We can fit 10-20x more data in VRAM!**

---

## Philosophy Alignment

✅ **"Like the 15 RPN stacks"** - Hybrid parallelism at all levels
✅ **"3GB goal, 8GB max, testing on 12GB"** - Scalable memory usage
✅ **"Sovereign execution"** - All parallelism via CUDA Driver API
✅ **"Galaxy/House memory"** - Unified workspace with caching
✅ **"Make history"** - Maximum performance, zero waste

---

## Implementation Priority

### IMMEDIATE (Start Now)
1. **Batched chunk processing** - 1-2 hours, 10x speedup
2. **Test consolidation** - Verify speedup works
3. **Document performance** - Update metrics

### SHORT-TERM (This Week)
1. **CUDA streams** - Async execution
2. **Memory pool** - Fast allocation
3. **Extended kernel 0xC4** - Major speedup

### MEDIUM-TERM (Next Week)
1. **Multi-GPU support** - For users with multiple GPUs
2. **Sound picture generation** - PTX spectrogram kernel
3. **Full pipeline optimization** - End-to-end profiling

---

**Status**: 🚀 **READY TO UNLEASH THE BEAST**

We're using 1% of a 12GB GPU. Time to fill those CUDA cores with parallel work!

**Next step**: Implement batched chunk processing for 10x immediate speedup! 🔥
