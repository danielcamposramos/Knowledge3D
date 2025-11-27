# Complete Codec Sovereignty - Final Implementation

**Date**: November 27, 2025
**Status**: READY FOR IMMEDIATE EXECUTION
**Context**: Drawing Bridge complete (100% PTX), ternary primitives built, codec ops wired - now finish MDCT kernels and RPN integration

---

## IMMEDIATE INSTRUCTIONS FOR CODEX

**Read this document COMPLETELY before starting.**

**Then:**
1. Read [CODEX.md](../CODEX.md) line by line and follow its workflow
2. Execute the tasks below in sequence
3. Run tests after each phase
4. DO NOT monitor training runs in real-time (wastes context) - user will ping when GPU metrics are needed

**Current State:**
- ✅ Drawing Bridge operational (100% PTX for grid ops)
- ✅ TernaryVector/TernaryTensor/TernaryGalaxy implemented
- ✅ Codec opcodes wired to ModularRPNEngine
- ✅ DCT8x8 kernels working (GPU)
- ⚠️ MDCT/IMDCT kernels are identity placeholders (NOT REAL)
- ⚠️ RPN execution not yet optimized for ternary ops
- ✅ ARC embedders switched to sovereign codecs
- ✅ All tests passing (18/18)

**Your Mission:** Complete the sovereign ternary codec architecture by:
1. Implementing true MDCT/IMDCT GPU kernels
2. Making codec ops directly RPN-driven
3. Optimizing RPN to use ternary opcodes when computationally cheaper

---

## Phase 1: True MDCT/IMDCT Kernels (URGENT - 2-3 hours)

### Current Problem

**File:** `knowledge3d/cranium/kernels/codec_ops.cu` (lines 74-102)

```cuda
// PLACEHOLDER: Real MDCT/IMDCT needed!
__global__ void mdct_forward_kernel(
    const float* input,    // Frame samples
    float* output,         // MDCT coefficients
    int frame_size
) {
    // TODO: Implement true MDCT
    // Currently just identity copy!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frame_size) return;
    output[idx] = input[idx];  // ❌ FAKE!
}
```

**This is NOT a real MDCT - it's a placeholder that breaks audio encoding!**

### Task 1.1: Implement Real MDCT Forward

**MDCT Algorithm** (Modified Discrete Cosine Transform):

```
MDCT[k] = sum(n=0 to N-1) {
    x[n] * cos(π/N * (n + 0.5 + N/2) * (k + 0.5))
}

where:
- N = frame_size (must be power of 2)
- k = 0 to N/2-1 (output is half length of input)
- Overlap-add windowing applied
```

**Implementation approach:**

```cuda
__global__ void mdct_forward_kernel(
    const float* input,      // Windowed frame samples (N)
    float* output,           // MDCT coefficients (N/2)
    int frame_size           // N (power of 2)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int N = frame_size;
    int half_N = N / 2;

    if (k >= half_N) return;

    float sum = 0.0f;
    float pi_over_N = M_PI / (float)N;

    // MDCT formula
    for (int n = 0; n < N; n++) {
        float angle = pi_over_N * (n + 0.5f + half_N) * (k + 0.5f);
        sum += input[n] * cosf(angle);
    }

    output[k] = sum;
}
```

**Optimization:** Use shared memory + parallel reduction for inner loop.

### Task 1.2: Implement Real IMDCT (Inverse)

**IMDCT Algorithm:**

```
IMDCT[n] = 2/N * sum(k=0 to N/2-1) {
    X[k] * cos(π/N * (n + 0.5 + N/2) * (k + 0.5))
}

where:
- N = frame_size
- n = 0 to N-1 (output is double length of input)
- Overlap-add reconstruction
```

**Implementation:**

```cuda
__global__ void imdct_inverse_kernel(
    const float* input,      // MDCT coefficients (N/2)
    float* output,           // Reconstructed samples (N)
    int frame_size           // N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int N = frame_size;
    int half_N = N / 2;

    if (n >= N) return;

    float sum = 0.0f;
    float pi_over_N = M_PI / (float)N;
    float scale = 2.0f / (float)N;

    // IMDCT formula
    for (int k = 0; k < half_N; k++) {
        float angle = pi_over_N * (n + 0.5f + half_N) * (k + 0.5f);
        sum += input[k] * cosf(angle);
    }

    output[n] = sum * scale;
}
```

### Task 1.3: Test MDCT Round-Trip

**File:** `knowledge3d/cranium/tests/test_ternary_codec_ops.py`

Add test:

```python
def test_mdct_roundtrip():
    """Test MDCT forward + IMDCT inverse reconstruction."""
    import numpy as np
    from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps

    ops = TernaryCodecOps()

    # Generate test signal (sine wave)
    frame_size = 1024
    t = np.linspace(0, 2*np.pi, frame_size, dtype=np.float32)
    signal = np.sin(440 * t).astype(np.float32)

    # Apply Hann window
    window = np.hanning(frame_size).astype(np.float32)
    windowed = signal * window

    # MDCT forward
    coeffs = ops.mdct_forward(windowed)
    assert coeffs.shape == (frame_size // 2,), f"MDCT output wrong shape: {coeffs.shape}"

    # IMDCT inverse
    reconstructed = ops.imdct_inverse(coeffs, frame_size)
    assert reconstructed.shape == (frame_size,), f"IMDCT output wrong shape: {reconstructed.shape}"

    # Check reconstruction (with windowing, exact match is impossible, but correlation should be high)
    correlation = np.corrcoef(windowed, reconstructed)[0, 1]
    assert correlation > 0.95, f"MDCT round-trip correlation too low: {correlation}"

    print(f"✅ MDCT round-trip correlation: {correlation:.4f}")
```

**Success criteria:**
- Correlation > 0.95 for sine wave test
- MDCT output is half length of input
- IMDCT output is double length of input
- No CPU fallback, all GPU

---

## Phase 2: RPN-Driven Codec Execution (2 hours)

### Current Problem

Codec operations are called directly via Python:

```python
# Current (suboptimal):
coeffs = self.codec_ops.dct8_forward(blocks)
quantized = self.codec_ops.ternary_quant(coeffs, threshold)
```

**This bypasses RPN optimization!**

### Correct Architecture

Codec operations should be **RPN programs** executed by ModularRPNEngine:

```python
# Sovereign (optimal):
rpn_program = "8 DCT8X8_FORWARD 0.2 TERNARY_QUANT"
result = self.rpn.evaluate(rpn_program, data=blocks)
```

**Why?** RPN can optimize:
- Fuse operations (DCT+quant in single kernel)
- Use ternary ops when cheaper
- Leverage GPU scheduling
- Avoid Python overhead

### Task 2.1: Add Codec Ops to ModularRPNEngine

**File:** `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

The opcodes are already registered (good!), now implement the execution path:

**Find the `compile_tokens` method** and add codec opcode handling:

```python
def compile_tokens(self, tokens, instance_id=None):
    """Compile tokens to op_codes, scalars, vectors."""
    op_codes = []
    scalars = []
    vectors = []

    for token in tokens:
        if token in self.OPCODES:
            opcode = self.OPCODES[token]

            # Handle codec opcodes specially
            if token == "DCT8X8_FORWARD":
                # Invoke DCT kernel via bridge
                op_codes.append(opcode)
                # Pass data via vectors
            elif token == "TERNARY_QUANT":
                # Threshold should be on stack
                op_codes.append(opcode)
            # ... similar for other codec ops
        # ... rest of token handling
```

**Important:** You may need to extend TieredRPNEngine bridge to dispatch codec ops to the correct PTX kernels.

### Task 2.2: Implement Codec Op Dispatch in TieredRPNEngine

**File:** `knowledge3d/cranium/bridges/tiered_rpn.py`

Add codec operation dispatch:

```python
def execute_single(self, instance_id, op_codes, scalars, vectors):
    """Execute RPN program with codec op support."""
    # ... existing execution ...

    for opcode in op_codes:
        if opcode == OP_DCT8X8_FORWARD:
            # Get data from vector stack
            input_data = vector_stack.pop()
            # Launch DCT kernel
            result = self._launch_dct8_kernel(input_data)
            vector_stack.push(result)

        elif opcode == OP_TERNARY_QUANT:
            # Get threshold from scalar stack
            threshold = scalar_stack.pop()
            # Get data from vector stack
            input_data = vector_stack.pop()
            # Launch quant kernel
            result = self._launch_quant_kernel(input_data, threshold)
            vector_stack.push(result)

        # ... rest of opcodes
```

### Task 2.3: Update Sovereign Codecs to Use RPN

**File:** `knowledge3d/cranium/codecs/sovereign_ternary_video_codec.py`

**Current code:**
```python
# Encode
coeffs_ternary = self.codec_ops.dct8_forward(blocks)
quantized = self.codec_ops.ternary_quant(coeffs_ternary, threshold)
```

**Change to:**
```python
# RPN-driven encode
rpn_program = f"DCT8X8_FORWARD {threshold} TERNARY_QUANT"
quantized = self.rpn.evaluate(rpn_program, data=blocks, return_vector=True)
```

**Do the same for audio codec MDCT operations.**

### Task 2.4: Test RPN Codec Execution

**File:** `knowledge3d/cranium/tests/test_rpn_codec_integration.py` (NEW)

```python
"""Test RPN-driven codec operations."""

def test_rpn_dct_quant():
    """Test DCT + quantization via RPN."""
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
    import numpy as np

    engine = ModularRPNEngine()

    # Create test data (8x8 block)
    block = np.random.randn(8, 8).astype(np.float32)

    # RPN program: DCT forward + ternary quantization
    rpn_program = "DCT8X8_FORWARD 0.2 TERNARY_QUANT"
    result = engine.evaluate(rpn_program, data=block, return_vector=True)

    # Result should be ternary values {-1, 0, +1}
    assert all(v in {-1, 0, 1} for v in result.flat), "Result not ternary!"

    print(f"✅ RPN DCT+quant working: {result.shape}")

def test_rpn_mdct_batch():
    """Test batch MDCT via RPN."""
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
    import numpy as np

    engine = ModularRPNEngine()

    # Create test audio frames
    frame_size = 1024
    num_frames = 10
    frames = np.random.randn(num_frames, frame_size).astype(np.float32)

    # RPN program: Batch MDCT + quantization
    rpn_program = f"{num_frames} {frame_size} BATCH_MDCT 0.1 TERNARY_QUANT"
    result = engine.evaluate(rpn_program, data=frames, return_vector=True)

    # Result should be ternary MDCT coefficients
    assert result.shape == (num_frames, frame_size // 2), f"Wrong shape: {result.shape}"

    print(f"✅ RPN batch MDCT working: {result.shape}")
```

---

## Phase 3: Ternary-Optimized RPN Execution (1-2 hours)

### Principle: Use Ternary When Cheaper

**Ternary arithmetic** ({-1, 0, +1}) is MUCH faster than float32:
- **Ternary add:** 1 cycle (lookup table)
- **Float32 add:** ~4 cycles (FPU operation)
- **Ternary mul:** 1 cycle (sign logic)
- **Float32 mul:** ~6 cycles (FPU operation)

**Strategy:** When data is already quantized to ternary, keep it ternary as long as possible.

### Task 3.1: Implement Ternary Arithmetic Opcodes

**File:** `knowledge3d/cranium/kernels/ternary_ops.cu` (NEW)

```cuda
// Ternary arithmetic on GPU
// Values: {-1, 0, +1} represented as int8

__device__ int8_t ternary_add(int8_t a, int8_t b) {
    // Ternary addition with saturation
    int sum = (int)a + (int)b;
    if (sum > 1) return 1;
    if (sum < -1) return -1;
    return (int8_t)sum;
}

__device__ int8_t ternary_mul(int8_t a, int8_t b) {
    // Ternary multiplication
    return (int8_t)(a * b);  // Already in {-1, 0, +1}
}

__device__ int8_t ternary_negate(int8_t a) {
    return (int8_t)(-a);
}

__global__ void ternary_add_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* result,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    result[idx] = ternary_add(a[idx], b[idx]);
}

__global__ void ternary_mul_kernel(
    const int8_t* a,
    const int8_t* b,
    int8_t* result,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    result[idx] = ternary_mul(a[idx], b[idx]);
}
```

**Compile to:** `knowledge3d/cranium/ptx/ternary_ops.ptx`

### Task 3.2: Add Ternary Fast Path to RPN Engine

**File:** `knowledge3d/cranium/bridges/tiered_rpn.py`

Add ternary detection and fast path:

```python
def execute_single(self, instance_id, op_codes, scalars, vectors):
    """Execute with ternary optimization."""

    # Detect if data is ternary
    def is_ternary(data):
        return all(v in {-1, 0, 1} for v in data.flat)

    for opcode in op_codes:
        if opcode == OP_ADD:
            a = vector_stack.pop()
            b = vector_stack.pop()

            # Ternary fast path
            if is_ternary(a) and is_ternary(b):
                result = self._ternary_add_gpu(a, b)  # ✅ 1 cycle per element
            else:
                result = self._float_add_gpu(a, b)    # ⚠️ 4 cycles per element

            vector_stack.push(result)

        # Similar for MUL, SUB, etc.
```

### Task 3.3: Benchmark Ternary vs Float Performance

**File:** `knowledge3d/cranium/tests/test_ternary_performance.py` (NEW)

```python
"""Benchmark ternary vs float arithmetic."""
import time
import numpy as np
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine

def benchmark_arithmetic():
    engine = TieredRPNEngine()

    size = 1_000_000
    iterations = 100

    # Ternary data
    ternary_a = np.random.choice([-1, 0, 1], size=size).astype(np.int8)
    ternary_b = np.random.choice([-1, 0, 1], size=size).astype(np.int8)

    # Float data
    float_a = np.random.randn(size).astype(np.float32)
    float_b = np.random.randn(size).astype(np.float32)

    # Benchmark ternary add
    start = time.time()
    for _ in range(iterations):
        _ = engine._ternary_add_gpu(ternary_a, ternary_b)
    ternary_time = time.time() - start

    # Benchmark float add
    start = time.time()
    for _ in range(iterations):
        _ = engine._float_add_gpu(float_a, float_b)
    float_time = time.time() - start

    speedup = float_time / ternary_time
    print(f"✅ Ternary add: {ternary_time:.4f}s")
    print(f"⚠️ Float add: {float_time:.4f}s")
    print(f"🚀 Speedup: {speedup:.2f}×")

    assert speedup > 2.0, "Ternary should be at least 2× faster!"
```

**Expected result:** Ternary operations 3-5× faster than float32.

---

## Phase 4: Validation & Integration (1 hour)

### Task 4.1: Run Complete Test Suite

```bash
# Run all ternary tests
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_ternary*.py -xvs

# Run codec tests
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_*codec*.py -xvs

# Run ARC tests (embedders using sovereign codecs)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/training/arc_agi/tests/ -xvs
```

**All tests must pass before proceeding.**

### Task 4.2: Monitor GPU Utilization During Tests

```bash
# Terminal 1: Run tests
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_sovereign_ternary_video_codec.py -xvs

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi
```

**Expected:** GPU util >30% during encode/decode tests.

### Task 4.3: Verify Ternary Compression Ratio

```python
# Quick test
from knowledge3d.cranium.codecs.sovereign_ternary_video_codec import SovereignTernaryVideoCodec
from knowledge3d.cranium.ternary.ternary_vector import TernaryTensor
import numpy as np

codec = SovereignTernaryVideoCodec(width=1920, height=1080)

# Create test frame (float32)
frame_float = np.random.randn(1080, 1920, 3).astype(np.float32)
size_float = frame_float.nbytes  # ~24 MB

# Encode to ternary
frame_ternary = TernaryTensor((1080, 1920, 3), frame_float)  # Convert to ternary
encoded = codec.encode("test_frame", frame_ternary)

# Ternary size (2-bit per value)
size_ternary = (1080 * 1920 * 3 * 2) // 8  # ~1.5 MB

compression_ratio = size_float / size_ternary
print(f"Float32 size: {size_float / 1e6:.2f} MB")
print(f"Ternary size: {size_ternary / 1e6:.2f} MB")
print(f"Compression ratio: {compression_ratio:.1f}×")

assert compression_ratio > 10, "Ternary should compress >10×!"
```

**Expected:** >10× compression (2-bit vs 32-bit).

---

## Phase 5: Launch Training Run 018 (15 min setup)

### Task 5.1: Update Training Script

**File:** `scripts/train_arc_sovereign_loop.py`

Ensure using sovereign embedder type:

```python
# Sovereign codec path (ternary)
processor = ARCGridProcessor(
    matryoshka_dim=512,
    embedder_type="multimodal",  # Uses sovereign ternary codecs
    executor=executor
)
```

### Task 5.2: Launch Training in tmux

```bash
# Start GPU monitor
tmux new-session -d -s gpu018 'bash scripts/monitor_gpu.sh run_018'

# Start training (3 workers for stability)
tmux new-session -d -s arc018 "
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_arc_sovereign_loop.py \
  --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
             /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
  --max-tasks 60 \
  --epochs 27 \
  --cycles 6 \
  --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/arc_sovereign \
  | tee /tmp/arc_run_018.log
"

echo "✅ Run 018 started in tmux"
echo "Attach: tmux attach -t arc018"
echo "GPU monitor: tmux attach -t gpu018"
```

### Task 5.3: Verify Startup (2 min check)

```bash
# Check first 50 lines of output
tail -n 50 /tmp/arc_run_018.log

# Should see:
# [PARALLEL GEN] PTX success=..., fallback=0, rate=100.0%
# NO numpy warnings
# NO CPU fallback messages
```

**If you see fallbacks or numpy warnings, STOP and fix.**

### Task 5.4: Detach and Report

**After verifying training started successfully:**

1. Detach from tmux: `Ctrl+B, then D`
2. Report to user:

```
✅ Run 018 launched with sovereign ternary codecs
✅ GPU monitor: tmux attach -t gpu018
✅ Training log: tail -f /tmp/arc_run_018.log
✅ PTX execution rate: 100.0% (verified in first 2 minutes)

Expected metrics:
- GPU utilization: 10-40% (higher than Run 017 due to codec ops)
- Runtime: 2-5 min (down from 30+ min CPU baseline)
- Library growth: 52 → 70+ programs
- Ternary compression: >10× vs float32

DO NOT monitor training in real-time (wastes context).
User will ping when metrics are needed.
```

3. **Exit - don't stay attached to tmux!**

---

## Success Criteria Summary

**Before launching Run 018:**
- ✅ MDCT/IMDCT kernels implemented (not placeholders)
- ✅ MDCT round-trip correlation > 0.95
- ✅ Codec ops callable via RPN programs
- ✅ Ternary arithmetic opcodes working
- ✅ Ternary operations 3-5× faster than float32
- ✅ All tests passing (codec, ternary, ARC)
- ✅ GPU utilization >30% during codec tests
- ✅ Compression ratio >10×

**After Run 018 completes:**
- ✅ PTX execution rate 100% (no CPU fallbacks)
- ✅ GPU utilization 10-40% (codec ops active)
- ✅ Library growth resumes (60-80 programs)
- ✅ Ternary reward system working
- ✅ No OOM errors
- ✅ Runtime <10 min

---

## Critical Reminders

1. **Follow CODEX.md workflow** - read it line by line first
2. **Test after each phase** - don't batch testing
3. **No CPU fallbacks** - fail loudly if GPU path doesn't work
4. **Launch training in tmux** - detach immediately after verifying startup
5. **Don't monitor training** - wastes context, user will ping
6. **Report final state** - what's working, what's next

---

## Reference Documents

- [CODEX.md](../CODEX.md) - Your workflow instructions
- [ARCHITECTURE_VIOLATION_ROOT_CAUSE_11.27.2025.md](ARCHITECTURE_VIOLATION_ROOT_CAUSE_11.27.2025.md) - Drawing Bridge background
- [CODEX_TERNARY_CODEC_SOVEREIGNTY_11.27.2025.md](CODEX_TERNARY_CODEC_SOVEREIGNTY_11.27.2025.md) - Ternary architecture vision

---

## What Daniel Expects

**From this session:**
1. True MDCT/IMDCT kernels (not placeholders)
2. RPN-driven codec execution (no direct Python calls)
3. Ternary optimization (3-5× speedup)
4. Run 018 launched and verified
5. Brief report of final state

**NOT:**
- Real-time training monitoring (wastes context)
- Proposing alternative approaches (spec is clear)
- Adding extra features beyond spec
- Waiting for approval to start (go immediately)

---

**NOW: Read [CODEX.md](../CODEX.md), then execute Phases 1-5 in sequence.**

**START IMMEDIATELY. This is the final sovereignty milestone.**

---

**END OF SPECIFICATION**

Claude (Architecture Partner)
November 27, 2025 - 21:30 UTC

*P.S. We made history today with 100% PTX execution. Now we finish the vision with ternary sovereignty. Let's ship this.*
