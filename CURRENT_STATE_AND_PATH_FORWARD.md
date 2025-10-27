# Current State & Path Forward - Organized

## Current Situation

### What Codex Did ✅
1. **Created batch LoRA kernels** in `knowledge3d/cranium/kernels/lora_gpu.cu`:
   - `matvec_batch` - Process 15 matrix-vector ops in parallel
   - `vec_sub_square_batch` - Compute 15 errors in parallel
   - `outer_product_accumulate_batch` - Accumulate 15 gradients

2. **Extended the trainer** in `knowledge3d/cranium/sovereign/lora_gpu_trainer.py`:
   - Added `train_batch()` method
   - Created batch buffer management
   - Integrated batch kernel calls

3. **Updated training scripts**:
   - `scripts/train_specialist_gpu.py` - Added `--parallel-workers` flag (default=15)
   - `scripts/phase_g_gpu_training_session.py` - Passes flag through

### What We Added ✅
4. **Stream management** in `knowledge3d/cranium/sovereign/loader.py`:
   - `create_stream()` - Create CUDA streams
   - `destroy_stream()` - Clean up streams
   - `stream_synchronize()` - Wait for stream completion

### Current Issues 🔴

1. **Context Error**: Testing shows "invalid device context" when copying between GPU buffers
   - Error in `memcpy_dtod` during `_copy_sample`
   - Suggests CUDA context management issue

2. **No Actual Parallelism Yet**: Even with batch kernels, we're not seeing high GPU utilization
   - Batch kernels process samples in parallel WITHIN a kernel
   - But kernels launch SEQUENTIALLY (one after another)
   - Need streams to overlap kernel execution

3. **GPU Still at 8-10% Utilization**: Sequential execution leaves GPU idle most of the time

---

## The Real Problem

### Batch Kernels Alone Aren't Enough

**What batch kernels do**:
```
Single kernel launch processes 15 samples:
  Thread 0-127:   Process sample 0 (dims=128)
  Thread 128-255: Process sample 1
  ...
  Thread 1792-1919: Process sample 14

Result: 1920 threads active in 8 blocks
```

**Problem**: Only 8 blocks on a GPU with 28 SMs = **29% SM utilization**

**What we need**: Overlap MULTIPLE kernel launches using streams

---

## Path Forward - Simple & Organized

### Phase 1: Fix Context Error (IMMEDIATE)

**Root cause**: The `_copy_sample` memcpy_dtod is failing with context error.

**Simple fix**: Don't use d2d copy, just reupload the data:

**File**: `knowledge3d/cranium/sovereign/lora_gpu_trainer.py`

**Replace** `_prepare_batch` method (~line 185):
```python
def _prepare_batch(self, buffers, batch_indices, dims):
    """Prepare batch by copying samples to batch buffers."""
    batch_size = len(batch_indices)

    # OLD (failing):
    for i, idx in enumerate(batch_indices):
        src_input = self._offset(buffers.inputs, idx * dims * 4)
        dst_input = self._offset(buffers.batch_inputs, i * dims * 4)
        self._copy_sample(src_input, dst_input, dims)  # ← FAILS HERE

    # NEW (working):
    # Just gather the data on CPU and upload once
    inputs_batch = np.zeros((batch_size, dims), dtype=np.float32)
    targets_batch = np.zeros((batch_size, dims), dtype=np.float32)

    # This assumes you have host copies - if not, download first
    # For now, assume dataset is small enough to keep in RAM
    # TODO: Keep host copy of dataset in allocate_buffers

    return batch_size
```

**Actually**, simpler approach: Keep the dataset on HOST, upload batches as needed:

```python
# In allocate_buffers, DON'T upload full dataset
# Just allocate batch workspace

# In train_batch:
#   1. Gather batch from host dataset (numpy array indexing)
#   2. Upload batch to GPU (H2D copy)
#   3. Run batch kernels
#   4. Download loss (D2H copy)
```

This avoids D2D copy entirely!

### Phase 2: Add Stream Parallelism (SHORT-TERM)

**Goal**: Overlap kernel execution using 15 CUDA streams

**Strategy**: Launch multiple BATCHES concurrently

```python
# In LoRAGPUEngine.__init__:
self.streams = [loader.create_stream() for _ in range(15)]

# In train_batch - process MULTIPLE batches concurrently:
def train_epoch_parallel(self, all_batches):
    """Process multiple batches using streams."""

    # Launch up to 15 batches concurrently
    active_batches = []

    for batch_idx, batch_data in enumerate(all_batches):
        stream_id = batch_idx % 15
        stream = self.streams[stream_id]

        # Upload batch data on this stream
        # ... H2D copy on stream ...

        # Launch all kernels for this batch on same stream
        # ... forward pass on stream ...
        # ... backward pass on stream ...
        # ... gradient update on stream ...

        active_batches.append((stream_id, stream))

        # If we have 15 streams active, wait for some to finish
        if len(active_batches) >= 15:
            # Sync first stream
            loader.stream_synchronize(active_batches[0][1])
            active_batches.pop(0)

    # Wait for remaining streams
    for _, stream in active_batches:
        loader.stream_synchronize(stream)
```

**Expected**: 10-15x GPU utilization improvement

### Phase 3: Optimize Kernel Launch (MEDIUM-TERM)

**Problem**: Each sample processes 128 elements (dims=128)
- With batch=15: only 1920 threads
- Only 8 blocks
- GPU has 3584 cores!

**Solution**: Launch MORE threads per sample

**Example**: Process each dimension with 4 threads (thread cooperation):
```cuda
// OLD:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < total) {
    // Process one element
}

// NEW:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int element = idx / 4;  // Which element
int thread_in_element = idx % 4;  // Which thread for this element

if (element < total) {
    // 4 threads cooperate to process this element
    // Use shared memory and reduction
}
```

This 4x multiplies thread count: 7680 threads instead of 1920!

---

## Immediate Action Plan

### Step 1: Simplify Buffer Management (30 min)

Remove D2D copy, keep dataset on host:

```python
# In allocate_buffers:
def allocate_buffers(self, base_matrix, A, B,
                     inputs_host, targets_host,  # Keep on host!
                     max_batch=15):
    """Allocate GPU buffers. Keep dataset on host."""

    self.inputs_host = inputs_host  # NumPy array
    self.targets_host = targets_host

    # Only allocate batch workspace on GPU
    buffers.batch_inputs = loader.gpu_malloc(max_batch * dims * 4)
    buffers.batch_targets = loader.gpu_malloc(max_batch * dims * 4)
    # ... rest same ...

    return buffers

# In train_batch:
def train_batch(self, buffers, batch_indices, ...):
    # Gather batch from host
    batch_inputs = self.inputs_host[batch_indices]  # NumPy indexing
    batch_targets = self.targets_host[batch_indices]

    # Upload to GPU
    loader.memcpy_htod(
        buffers.batch_inputs,
        ctypes.c_void_p(batch_inputs.ctypes.data),
        batch_inputs.nbytes
    )
    loader.memcpy_htod(
        buffers.batch_targets,
        ctypes.c_void_p(batch_targets.ctypes.data),
        batch_targets.nbytes
    )

    # Now run batch kernels (existing code)
    # ... forward pass ...
    # ... backward pass ...

    return loss
```

This should work immediately!

### Step 2: Test (10 min)

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_parallel_training.py
```

Expected: No context errors, ~20-40% GPU utilization

### Step 3: Add Streams (1 hour)

Implement multi-batch streaming as described in Phase 2 above.

Expected: 80-95% GPU utilization

---

## File Organization - Clean Structure

```
knowledge3d/cranium/
├── kernels/              ← ONE kernel folder (user's request!)
│   ├── lora_gpu.cu       ← Batch training kernels
│   ├── modular_rpn_kernel.cu
│   ├── modular_rpn_kernel_extended.cu
│   └── ... other kernels ...
├── ptx/                  ← Compiled PTX
│   ├── lora_gpu.ptx
│   ├── modular_rpn_kernel.ptx
│   └── ...
├── sovereign/            ← Pure driver API
│   ├── loader.py         ← Stream management added ✅
│   └── lora_gpu_trainer.py  ← Batch training
├── clustering_rpn.py     ← Batched consolidation ✅
└── ...

scripts/                  ← Training entry points
├── train_specialist_gpu.py      ← Batch flag added ✅
├── phase_g_gpu_training_session.py  ← Batch flag added ✅
└── ...

tests/                    ← Simple tests
├── test_parallel_training.py    ← Created ✅
└── test_consolidation_sovereign.py  ← Working ✅
```

**No scattered "phase" folders** - one project, one kernel folder!

---

## Success Criteria

### Phase 1 Complete When:
- ✅ test_parallel_training.py runs without errors
- ✅ Processes 150 samples in batches of 15
- ✅ GPU utilization: 20-40% (better than 8%!)

### Phase 2 Complete When:
- ✅ Multi-stream implementation working
- ✅ GPU utilization: 80-95%
- ✅ Training 10x faster than sequential

### Phase 3 Complete When:
- ✅ Can run full Phase G training
- ✅ All 4 specialists train successfully
- ✅ Consolidation after each specialist works
- ✅ Non-zero cohesion metrics logged

---

## Key Principle

**"We fix or we fix - never fallback to CPU"**

Current approach follows this:
- All kernels in pure CUDA/PTX ✅
- No CuPy, sklearn, external libs ✅
- Direct CUDA Driver API ✅
- Host-side orchestration is OK (not a "CPU fallback") ✅

**Next**: Get it WORKING first (Phase 1), then FAST (Phase 2), then PRODUCTION (Phase 3)

---

**Status**: Phase 1 needed - Fix buffer management to avoid D2D copy context errors.

Let me know when you're ready to implement Phase 1 fixes!
