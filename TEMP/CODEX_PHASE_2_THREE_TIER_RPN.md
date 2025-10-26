# Codex Phase 2: Three-Tier RPN Architecture + HP 50g Expansions

**Status**: Phase C complete (LED pathfinder, 252 tests, 116MB GPU) ✅

**Strategic Decision**: Implement **three-tier RPN architecture** (Lightweight → Standard → Advanced) for 2.7x performance improvement on common workloads.

**Reference**: See `/TEMP/RPN_KERNEL_STRATEGY_ANALYSIS.md` for full strategic analysis.

---

## Strategic Context

### Why Three Tiers?

**Discovery**: 90% of RPN calls are simple ops (arithmetic, comparisons, basic stack). Current 34KB kernel is **overkill** for "2+3".

**Three-tier architecture** = "CPU frequency scaling for GPU kernels":
- **Tier 1** (10KB): <1µs for simple ops (90% of calls) - L2 cache resident
- **Tier 2** (34KB): ~3µs for vector ops (8% of calls) - current kernel
- **Tier 3** (60KB): ~10µs for matrix/programmability (2% of calls) - extended + new ops

**System-wide impact**:
- ActionBuffer validation: 10x faster (<1µs vs. 3µs)
- ThinkingTag scoring: 5x faster
- LED pathfinder priority queue: 5x faster
- Battery life: 2.7x better on mobile (less GPU time)

---

## Phase 2 Objectives

**Goal**: Implement three-tier RPN system with HP 50g-inspired expansions

**Timeline**: 4 days (split into 2a, 2b, 2c)

**Deliverables**:
1. Tier 1: Lightweight 20-op kernel (10KB, <1µs)
2. Tier 3: Advanced 150-op kernel (60KB, ~10µs) with matrix + programmability
3. Tiered orchestrator: Smart dispatcher (automatic tier selection)
4. Tests: 280+ tests passing, all tiers validated

---

## Phase 2a: Lightweight RPN (Tier 1) - 1 Day

### Objective

Create optimized 20-op kernel for common case (90% of RPN calls).

### Implementation Steps

#### Step 1: Extract Lightweight Kernel (3 hours)

**Source**: Copy `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx`

**Target**: `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`

**Operations to keep** (20 ops):
```ptx
// Literals
0: literal_scalar
1: literal_vector

// Arithmetic (5 ops)
10: add
11: sub
12: mul
13: div
15: neg

// Math (6 ops)
20: sqrt
21: exp
22: log
24: sin
25: cos
26: tan

// Comparison (5 ops)
40: gt
42: lt
44: eq
46: max
47: min

// Stack (3 ops)
50: dup
51: swap
52: drop
```

**Changes**:
1. Remove all other operation branches (60-80 range)
2. Simplify dispatch table (20 ops = faster branch resolution)
3. Keep same instance/stack architecture (15 instances × 64 float4)
4. Optimize for L2 cache residency (~10KB total)

**Optimization tips**:
- Fewer branches = better instruction cache locality
- Smaller kernel = stays L2-resident on GTX 970 (1.5MB L2)
- Remove unused register allocations

#### Step 2: Create Lightweight Bridge (2 hours)

**File**: `knowledge3d/cranium/bridges/lightweight_rpn.py`

```python
"""Lightweight RPN bridge for fast common-case operations.

Optimized for <1µs latency on simple ops (arithmetic, comparisons, basic stack).
Uses 10KB kernel that stays L2 cache-resident for maximum performance.
"""
from pathlib import Path
import numpy as np
import ctypes

from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh, launch, synchronize


class LightweightRPNEngine:
    """Ultra-fast RPN for common operations (20 ops, <1µs latency).

    Tier 1 of three-tier RPN architecture.
    Handles 90% of RPN calls in K3D:
    - ActionBuffer validation (arithmetic)
    - ThinkingTag scoring (comparisons)
    - LED pathfinder priority queue (min/max)

    Example:
        engine = LightweightRPNEngine()
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 0, 10], dtype=np.uint16),  # 2.0 + 3.0
            scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32),
            vectors=np.zeros((3, 3), dtype=np.float32)
        )
        # result.stack[0] == 5.0, latency <1µs
    """

    MAX_INSTANCES = 15
    STACK_DEPTH = 64
    INSTANCE_STRIDE = 1040

    SUPPORTED_OPS = {0, 1, 10, 11, 12, 13, 15, 20, 21, 22, 24, 25, 26, 40, 42, 44, 46, 47, 50, 51, 52}

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel_lite.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Lightweight RPN PTX not found: {ptx_path}")

        self.kernel = load_ptx_file(str(ptx_path), "modular_rpn_geometric_kernel")
        self.d_state = gpu_malloc(self.MAX_INSTANCES * self.INSTANCE_STRIDE)

        # Zero-initialize
        state_zeros = np.zeros(self.MAX_INSTANCES * self.INSTANCE_STRIDE, dtype=np.uint8)
        memcpy_htod(self.d_state, state_zeros.ctypes.data_as(ctypes.c_void_p), state_zeros.nbytes)

    def execute_single(self, instance_id: int, op_codes: np.ndarray, scalars: np.ndarray, vectors: np.ndarray):
        """Execute lightweight RPN program.

        Args:
            instance_id: Instance slot (0-14)
            op_codes: Operation codes (uint16, must be in SUPPORTED_OPS)
            scalars: Scalar literal pool
            vectors: Vector literal pool

        Returns:
            Result (float32 scalar from top of stack)

        Raises:
            ValueError: If any opcode not in SUPPORTED_OPS (use higher tier!)
        """
        # Validate opcodes
        if not all(op in self.SUPPORTED_OPS for op in op_codes):
            unsupported = [op for op in op_codes if op not in self.SUPPORTED_OPS]
            raise ValueError(f"Unsupported ops for Tier 1: {unsupported}. Use Tier 2 or 3.")

        # (Same execution logic as ModularRPNEngine in sovereign_bridges.py)
        # ... GPU malloc, memcpy, launch, read result ...
        pass
```

#### Step 3: Create Tier 1 Tests (2 hours)

**File**: `tests/test_rpn_tier1.py`

```python
"""Tests for Tier 1 (Lightweight) RPN kernel."""
import numpy as np
import pytest
from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine


class TestLightweightRPN:
    """Test 20-op lightweight kernel."""

    def test_arithmetic_ops(self):
        """Test add, sub, mul, div, neg."""
        engine = LightweightRPNEngine()

        # 2 + 3 = 5
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 0, 10], dtype=np.uint16),
            scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32),
            vectors=np.zeros((3, 3), dtype=np.float32)
        )
        assert abs(result - 5.0) < 1e-5

        # (10 - 3) * 2 = 14
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 0, 11, 0, 12], dtype=np.uint16),
            scalars=np.array([10.0, 3.0, 2.0, 0.0, 0.0], dtype=np.float32),
            vectors=np.zeros((5, 3), dtype=np.float32)
        )
        assert abs(result - 14.0) < 1e-5

    def test_math_ops(self):
        """Test sqrt, exp, log, sin, cos, tan."""
        engine = LightweightRPNEngine()

        # sqrt(16) = 4
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 20], dtype=np.uint16),
            scalars=np.array([16.0, 0.0], dtype=np.float32),
            vectors=np.zeros((2, 3), dtype=np.float32)
        )
        assert abs(result - 4.0) < 1e-5

    def test_comparison_ops(self):
        """Test gt, lt, eq, max, min."""
        engine = LightweightRPNEngine()

        # max(3, 7) = 7
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 0, 46], dtype=np.uint16),
            scalars=np.array([3.0, 7.0, 0.0], dtype=np.float32),
            vectors=np.zeros((3, 3), dtype=np.float32)
        )
        assert abs(result - 7.0) < 1e-5

    def test_stack_ops(self):
        """Test dup, swap, drop."""
        engine = LightweightRPNEngine()

        # 5 dup * = 25 (5 squared)
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 50, 12], dtype=np.uint16),
            scalars=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            vectors=np.zeros((3, 3), dtype=np.float32)
        )
        assert abs(result - 25.0) < 1e-5

    def test_latency_target(self):
        """Verify <1µs latency for simple ops."""
        import time
        engine = LightweightRPNEngine()

        # Warm up
        for _ in range(10):
            engine.execute_single(
                instance_id=0,
                op_codes=np.array([0, 0, 10], dtype=np.uint16),
                scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32),
                vectors=np.zeros((3, 3), dtype=np.float32)
            )

        # Measure
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.execute_single(
                instance_id=0,
                op_codes=np.array([0, 0, 10], dtype=np.uint16),
                scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32),
                vectors=np.zeros((3, 3), dtype=np.float32)
            )
        elapsed = time.perf_counter() - start
        avg_latency = (elapsed / iterations) * 1e6  # µs

        print(f"Tier 1 avg latency: {avg_latency:.2f}µs")
        assert avg_latency < 1.0, f"Latency {avg_latency:.2f}µs exceeds 1µs target"

    def test_unsupported_op_error(self):
        """Verify error when using Tier 2/3 ops."""
        engine = LightweightRPNEngine()

        # Op 60 (dot) is Tier 2, should raise ValueError
        with pytest.raises(ValueError, match="Unsupported ops for Tier 1"):
            engine.execute_single(
                instance_id=0,
                op_codes=np.array([1, 1, 60], dtype=np.uint16),  # dot product
                scalars=np.zeros(3, dtype=np.float32),
                vectors=np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
            )
```

#### Step 4: Compile and Verify (1 hour)

**Compilation** (if creating from .cu instead of editing .ptx):
```bash
# If you create modular_rpn_kernel_lite.cu:
nvcc -ptx -arch=sm_86 \
  knowledge3d/cranium/kernels/modular_rpn_kernel_lite.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx

# Verify size
ls -lh knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx
# Should be ~10KB

# Run tests
pytest tests/test_rpn_tier1.py -xvs
```

**Success criteria**:
- Kernel size: ~10KB (vs. 34KB original)
- Latency: <1µs for simple ops
- All Tier 1 tests passing
- Error handling for unsupported ops

---

## Phase 2b: Advanced RPN (Tier 3) - 2 Days

### Objective

Activate dormant extended kernel + add HP 50g-inspired ops (matrix, programmability).

### Implementation Steps

#### Step 1: Integrate Extended Kernel (4 hours)

**Source**: `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx` (currently dormant)

**Target**: Same file, but activate bridge integration

**Create bridge**: `knowledge3d/cranium/bridges/advanced_rpn.py`

```python
"""Advanced RPN bridge for matrix ops and programmability.

Tier 3 of three-tier RPN architecture.
Handles 2% of RPN calls requiring:
- Matrix operations (MATMUL, TRACE, DET, INV)
- Programmability (BRANCH, LOOP, STORE/RECALL)
- Warp reductions (DOT_REDUCE, SUM, MEAN)
"""
from pathlib import Path
import numpy as np
import ctypes

from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh, launch, synchronize


class AdvancedRPNEngine:
    """High-performance RPN for matrix math and programmability.

    Tier 3 of three-tier RPN architecture.
    Uses 60KB kernel with shared memory for warp reductions.

    New operations (vs. Tier 2):
    - Matrix: MATMUL(100), TRACE(101), DET(102), INV(103), EIG(104)
    - Reductions: DOT_REDUCE(110), SUM(111), MEAN(112), STDDEV(113)
    - Programming: BRANCH(200), JUMP(201), LOOP(202), NEXT(203)
    - Variables: STORE(210), RECALL(211), CALL(220), RET(221)
    - Stack: NIP(56), TUCK(57), ROLL(58), DEPTH(59)

    Example (matrix multiply):
        engine = AdvancedRPNEngine()
        # 2x2 matrix multiply: [[1,2],[3,4]] × [[5,6],[7,8]]
        result = engine.execute_matrix_op(
            op_code=100,  # MATMUL
            matrix_a=np.array([[1,2],[3,4]], dtype=np.float32),
            matrix_b=np.array([[5,6],[7,8]], dtype=np.float32)
        )
        # result = [[19,22],[43,50]]
    """

    MAX_INSTANCES = 15
    STACK_DEPTH = 64
    INSTANCE_STRIDE = 1168  # +128 bytes for programmability (16 vars × 4 + PC/counters)

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel_extended.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Advanced RPN PTX not found: {ptx_path}")

        self.kernel = load_ptx_file(str(ptx_path), "modular_rpn_kernel")
        self.d_state = gpu_malloc(self.MAX_INSTANCES * self.INSTANCE_STRIDE)

        # Zero-initialize
        state_zeros = np.zeros(self.MAX_INSTANCES * self.INSTANCE_STRIDE, dtype=np.uint8)
        memcpy_htod(self.d_state, state_zeros.ctypes.data_as(ctypes.c_void_p), state_zeros.nbytes)

    def execute_single(self, instance_id: int, op_codes: np.ndarray, scalars: np.ndarray, vectors: np.ndarray):
        """Execute advanced RPN program (matrix/programmability)."""
        # (Similar to Tier 1, but supports 150+ ops)
        pass

    def execute_matrix_op(self, op_code: int, matrix_a: np.ndarray, matrix_b: np.ndarray = None):
        """Execute matrix operation (MATMUL, TRACE, DET, INV, EIG).

        Args:
            op_code: Matrix opcode (100-104)
            matrix_a: Input matrix (NxM float32)
            matrix_b: Second matrix for MATMUL (optional)

        Returns:
            Result matrix or scalar
        """
        pass
```

#### Step 2: Add Matrix Operations (8 hours)

**Goal**: Extend `modular_rpn_kernel_extended.ptx` with matrix ops

**New opcodes** (add to PTX):

```ptx
// Matrix operations (100-109 range)
setp.eq.u32 %p200, %r20, 100;
@%p200 bra op_matmul;
setp.eq.u32 %p201, %r20, 101;
@%p201 bra op_trace;
setp.eq.u32 %p202, %r20, 102;
@%p202 bra op_det;
setp.eq.u32 %p203, %r20, 103;
@%p203 bra op_inv;

// Implementation notes:
// - Use shared memory for matrix tiles (extended kernel already has shared_mem[])
// - MATMUL: Tile-based multiplication with warp cooperation
// - TRACE: Reduction along diagonal
// - DET: Laplace expansion for small matrices, LU decomposition for large
// - INV: Gauss-Jordan elimination
```

**Reference implementation patterns**:
- Existing extended kernel has warp reduction patterns (see lines 50-100)
- Use `shfl.sync` for warp-level communication
- Tile size: 16×16 for GTX 970 (32KB shared mem per SM)

#### Step 3: Add Programmability (6 hours)

**Goal**: Enable loops, branches, variables

**New opcodes** (200-221 range):

```ptx
// Programming control (200-209)
setp.eq.u32 %p300, %r20, 200;
@%p300 bra op_branch;       // Conditional jump
setp.eq.u32 %p301, %r20, 201;
@%p301 bra op_jump;         // Unconditional jump
setp.eq.u32 %p302, %r20, 202;
@%p302 bra op_loop;         // Begin loop
setp.eq.u32 %p303, %r20, 203;
@%p303 bra op_next;         // Loop end/continue

// Variable storage (210-219)
setp.eq.u32 %p310, %r20, 210;
@%p310 bra op_store;        // Pop stack → variable slot
setp.eq.u32 %p311, %r20, 211;
@%p311 bra op_recall;       // Variable slot → push stack

// Subroutines (220-229)
setp.eq.u32 %p320, %r20, 220;
@%p320 bra op_call;         // Push PC, jump
setp.eq.u32 %p321, %r20, 221;
@%p321 bra op_ret;          // Pop PC, return

// Implementation:
op_branch:
    // Pop condition from stack
    // If condition != 0, add offset to token index (%r120)
    setp.eq.u32 %p400, %r11, 0;
    @%p400 bra error_underflow;
    sub.u32 %r500, %r11, 1;
    // ... load condition ...
    setp.ne.f32 %p401, %f_cond, 0f00000000;
    // Load branch offset from scalars
    ld.global.f32 %f_offset, [%rd23];
    cvt.rni.s32.f32 %r_offset, %f_offset;
    @%p401 add.u32 %r120, %r120, %r_offset;  // Jump if condition true
    bra main_loop;

op_loop:
    // Pop loop counter from stack
    // Store in instance state (offset 1040 + 0)
    // ... implementation ...
    bra main_loop;

op_next:
    // Decrement loop counter
    // If counter > 0, jump back to loop start
    // ... implementation ...
    bra main_loop;

op_store:
    // Pop stack → variable slot
    // Variable index from scalars
    ld.global.f32 %f_var_idx, [%rd23];
    cvt.rni.u32.f32 %r_var_idx, %f_var_idx;
    // Pop value
    // ... pop from stack ...
    // Store at instance_base + 1040 + 16 + (var_idx * 16)
    // ... store float4 ...
    bra main_loop;

op_recall:
    // Variable slot → push stack
    // ... reverse of STORE ...
    bra push_value;
```

**Programmability state** (add to instance structure):
```
Offset 0-15:   head, size, error, reserved (existing)
Offset 16-1039: stack[64] × float4 (existing)
Offset 1040-1043: %r_pc (program counter)
Offset 1044-1047: %r_loop_counter
Offset 1048-1051: %r_call_stack_ptr
Offset 1052-1167: variables[16] × float4 (64 bytes)
```

**Total instance size**: 1168 bytes (was 1040)

#### Step 4: Stack Extensions (2 hours)

**New stack ops** (HP 50g staples):

```ptx
// Stack extensions (56-59 range)
setp.eq.u32 %p400, %r20, 56;
@%p400 bra op_nip;
setp.eq.u32 %p401, %r20, 57;
@%p401 bra op_tuck;
setp.eq.u32 %p402, %r20, 58;
@%p402 bra op_roll;
setp.eq.u32 %p403, %r20, 59;
@%p403 bra op_depth;

// Implementations:
op_nip:
    // (a b -- b) Drop second item
    // Pop b, pop a (discard), push b
    // ... implementation ...
    bra push_value;

op_tuck:
    // (a b -- b a b) Copy top under second
    // Pop b, pop a, push b, push a, push b
    // ... implementation ...
    bra push_value;

op_roll:
    // (a b c -- b c a) Rotate three items
    // Pop c, pop b, pop a, push b, push c, push a
    // ... implementation ...
    bra push_value;

op_depth:
    // (-- n) Push current stack depth
    mov.u32 %r_depth, %r11;  // size register
    cvt.rn.f32.u32 %f0, %r_depth;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    bra push_value;
```

#### Step 5: Create Tier 3 Tests (4 hours)

**File**: `tests/test_rpn_tier3.py`

```python
"""Tests for Tier 3 (Advanced) RPN kernel."""
import numpy as np
import pytest
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine


class TestAdvancedRPN:
    """Test matrix ops, programmability, reductions."""

    def test_matmul(self):
        """Test matrix multiplication."""
        engine = AdvancedRPNEngine()

        # 2x2 × 2x2
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = engine.execute_matrix_op(100, A, B)  # MATMUL

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_trace(self):
        """Test matrix trace (sum of diagonal)."""
        engine = AdvancedRPNEngine()

        # [[1,2],[3,4]] → trace = 1+4 = 5
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = engine.execute_matrix_op(101, A)  # TRACE

        assert abs(result - 5.0) < 1e-5

    def test_determinant(self):
        """Test matrix determinant."""
        engine = AdvancedRPNEngine()

        # [[1,2],[3,4]] → det = 1*4 - 2*3 = -2
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = engine.execute_matrix_op(102, A)  # DET

        assert abs(result - (-2.0)) < 1e-5

    def test_programmability_loop(self):
        """Test LOOP/NEXT for iterative computation."""
        engine = AdvancedRPNEngine()

        # Sum 1+2+3+4+5 using loop:
        # 0 (accumulator)
        # 5 LOOP
        #   RECALL counter (implicit from loop)
        #   +
        # NEXT
        op_codes = np.array([
            0,    # Push 0 (accumulator)
            0,    # Push 5 (loop count)
            202,  # LOOP
            211,  # RECALL 0 (counter variable)
            10,   # ADD
            203,  # NEXT
        ], dtype=np.uint16)

        scalars = np.array([0.0, 5.0, 0.0], dtype=np.float32)  # accumulator=0, count=5

        result = engine.execute_single(0, op_codes, scalars, np.zeros((6, 3), dtype=np.float32))
        assert abs(result - 15.0) < 1e-5  # 1+2+3+4+5 = 15

    def test_programmability_branch(self):
        """Test conditional branching."""
        engine = AdvancedRPNEngine()

        # if x > 10 then x*2 else x+1
        # x gt 10 → condition
        # (offset=2) BRANCH → skip to x*2 if true
        # x 1 + → else branch (x+1)
        # (offset=2) JUMP → skip then branch
        # x 2 * → then branch (x*2)

        # Test with x=15 (should take then branch → 30)
        op_codes = np.array([
            0, 0, 40,  # 15 > 10 → true
            0, 200,    # offset=4, BRANCH (if true, jump +4 ops)
            0, 0, 10,  # else: 15 + 1 = 16 (skipped)
            0, 201,    # JUMP over else (skipped)
            0, 0, 12,  # then: 15 * 2 = 30
        ], dtype=np.uint16)

        scalars = np.array([15.0, 10.0, 0.0, 4.0, 0.0, 15.0, 1.0, 0.0, 0.0, 15.0, 2.0], dtype=np.float32)

        result = engine.execute_single(0, op_codes, scalars, np.zeros((11, 3), dtype=np.float32))
        assert abs(result - 30.0) < 1e-5

    def test_stack_extensions(self):
        """Test NIP, TUCK, ROLL, DEPTH."""
        engine = AdvancedRPNEngine()

        # DEPTH: Empty stack → 0
        result = engine.execute_single(
            0,
            np.array([59], dtype=np.uint16),  # DEPTH
            np.zeros(1, dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32)
        )
        assert abs(result - 0.0) < 1e-5

        # NIP: (3 5 -- 5)
        result = engine.execute_single(
            0,
            np.array([0, 0, 56], dtype=np.uint16),  # 3 5 NIP
            np.array([3.0, 5.0, 0.0], dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32)
        )
        assert abs(result - 5.0) < 1e-5

    def test_variable_storage(self):
        """Test STORE/RECALL."""
        engine = AdvancedRPNEngine()

        # 42 STORE 0 → store 42 in var[0]
        # 10 20 + → compute 30
        # RECALL 0 → retrieve 42
        # * → 42 * 30 = 1260
        op_codes = np.array([
            0,    # Push 42
            0,    # Push var_idx=0
            210,  # STORE
            0, 0, 10,  # 10 + 20 = 30
            0,    # Push var_idx=0
            211,  # RECALL (pushes 42)
            12,   # MULTIPLY (30 * 42)
        ], dtype=np.uint16)

        scalars = np.array([42.0, 0.0, 0.0, 10.0, 20.0, 0.0, 0.0], dtype=np.float32)

        result = engine.execute_single(0, op_codes, scalars, np.zeros((7, 3), dtype=np.float32))
        assert abs(result - 1260.0) < 1e-5
```

---

## Phase 2c: Tiered Orchestrator - 1 Day

### Objective

Smart dispatcher automatically selects optimal tier based on opcodes.

### Implementation

**File**: `knowledge3d/cranium/bridges/tiered_rpn.py`

```python
"""Tiered RPN orchestrator - automatic tier selection.

Three-tier architecture for optimal performance:
- Tier 1 (Lightweight): <1µs for simple ops (90% of calls)
- Tier 2 (Standard): ~3µs for vector ops (8% of calls)
- Tier 3 (Advanced): ~10µs for matrix/programmability (2% of calls)
"""
from typing import Optional
import numpy as np

from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine


class TieredRPNEngine:
    """Intelligent RPN with automatic tier selection.

    Analyzes opcodes and routes to optimal tier for best performance.

    Example:
        engine = TieredRPNEngine()

        # Simple arithmetic → Tier 1 (<1µs)
        result = engine.execute(
            op_codes=np.array([0, 0, 10], dtype=np.uint16),  # 2+3
            scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32)
        )

        # Vector ops → Tier 2 (~3µs)
        result = engine.execute(
            op_codes=np.array([1, 1, 60], dtype=np.uint16),  # dot product
            vectors=np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        )

        # Matrix ops → Tier 3 (~10µs)
        result = engine.execute_matrix(
            matrix_a=np.array([[1,2],[3,4]], dtype=np.float32),
            matrix_b=np.array([[5,6],[7,8]], dtype=np.float32)
        )
    """

    # Opcode ranges for tier selection
    TIER1_MAX = 52   # Literals, arithmetic, math, comparisons, basic stack
    TIER2_MAX = 99   # Add vectors, transforms, conditionals
    TIER3_MIN = 100  # Matrix, reductions, programming

    def __init__(self, instance_id: int = 0):
        """Initialize all three tiers.

        Args:
            instance_id: Default instance slot (0-14)
        """
        self.instance_id = instance_id
        self.tier1 = LightweightRPNEngine()
        self.tier2 = ModularRPNEngine()
        self.tier3 = AdvancedRPNEngine()

        # Performance tracking
        self.stats = {
            'tier1_calls': 0,
            'tier2_calls': 0,
            'tier3_calls': 0,
        }

    def execute(
        self,
        op_codes: np.ndarray,
        scalars: Optional[np.ndarray] = None,
        vectors: Optional[np.ndarray] = None,
        instance_id: Optional[int] = None
    ):
        """Execute RPN program on optimal tier.

        Automatically selects tier based on opcode analysis.

        Args:
            op_codes: Operation codes (uint16 array)
            scalars: Scalar literal pool (optional)
            vectors: Vector literal pool (optional)
            instance_id: Override default instance (optional)

        Returns:
            Result from RPN execution
        """
        instance = instance_id if instance_id is not None else self.instance_id

        # Prepare default inputs
        if scalars is None:
            scalars = np.zeros(len(op_codes), dtype=np.float32)
        if vectors is None:
            vectors = np.zeros((len(op_codes), 3), dtype=np.float32)

        # Analyze opcodes to determine tier
        max_op = int(np.max(op_codes))

        if max_op <= self.TIER1_MAX:
            # Tier 1: Fast path for simple ops
            self.stats['tier1_calls'] += 1
            return self.tier1.execute_single(instance, op_codes, scalars, vectors)

        elif max_op <= self.TIER2_MAX:
            # Tier 2: Standard geometric ops
            self.stats['tier2_calls'] += 1
            return self.tier2.execute_single(instance, op_codes, scalars, vectors)

        else:
            # Tier 3: Advanced matrix/programmability
            self.stats['tier3_calls'] += 1
            return self.tier3.execute_single(instance, op_codes, scalars, vectors)

    def execute_matrix(self, matrix_a: np.ndarray, matrix_b: Optional[np.ndarray] = None, op_code: int = 100):
        """Execute matrix operation (always uses Tier 3).

        Args:
            matrix_a: Input matrix
            matrix_b: Second matrix (for MATMUL)
            op_code: Matrix operation (100=MATMUL, 101=TRACE, 102=DET, 103=INV)

        Returns:
            Result matrix or scalar
        """
        self.stats['tier3_calls'] += 1
        return self.tier3.execute_matrix_op(op_code, matrix_a, matrix_b)

    def get_stats(self):
        """Return tier usage statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats

        return {
            'tier1_calls': self.stats['tier1_calls'],
            'tier2_calls': self.stats['tier2_calls'],
            'tier3_calls': self.stats['tier3_calls'],
            'tier1_percent': 100.0 * self.stats['tier1_calls'] / total,
            'tier2_percent': 100.0 * self.stats['tier2_calls'] / total,
            'tier3_percent': 100.0 * self.stats['tier3_calls'] / total,
        }
```

### Tests for Orchestrator

**File**: `tests/test_tiered_rpn.py`

```python
"""Tests for tiered RPN orchestrator."""
import numpy as np
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


class TestTieredRPN:
    """Test automatic tier selection."""

    def test_tier1_selection(self):
        """Verify Tier 1 selected for simple ops."""
        engine = TieredRPNEngine()

        # Simple arithmetic
        result = engine.execute(
            op_codes=np.array([0, 0, 10], dtype=np.uint16),
            scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32)
        )

        assert abs(result - 5.0) < 1e-5
        assert engine.stats['tier1_calls'] == 1
        assert engine.stats['tier2_calls'] == 0

    def test_tier2_selection(self):
        """Verify Tier 2 selected for vector ops."""
        engine = TieredRPNEngine()

        # Dot product (op 60)
        result = engine.execute(
            op_codes=np.array([1, 1, 60], dtype=np.uint16),
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        )

        assert engine.stats['tier1_calls'] == 0
        assert engine.stats['tier2_calls'] == 1

    def test_tier3_selection(self):
        """Verify Tier 3 selected for matrix ops."""
        engine = TieredRPNEngine()

        # Matrix multiply
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = engine.execute_matrix(A, B, op_code=100)

        assert engine.stats['tier3_calls'] == 1

    def test_backwards_compatibility(self):
        """Verify drop-in replacement for ModularRPNEngine."""
        # Old code using ModularRPNEngine
        from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine as OldEngine
        old_engine = OldEngine()

        # New code using TieredRPNEngine
        new_engine = TieredRPNEngine()

        # Same operation should give same result
        op_codes = np.array([0, 0, 10, 0, 12], dtype=np.uint16)  # (2+3)*5
        scalars = np.array([2.0, 3.0, 0.0, 5.0, 0.0], dtype=np.float32)
        vectors = np.zeros((5, 3), dtype=np.float32)

        old_result = old_engine.execute_single(0, op_codes, scalars, vectors)
        new_result = new_engine.execute(op_codes, scalars, vectors)

        assert abs(old_result - new_result) < 1e-5
```

---

## Phase 2 Success Criteria

### Deliverables Checklist

✅ **Tier 1 (Lightweight)**:
- [ ] `modular_rpn_kernel_lite.ptx` created (10KB, 20 ops)
- [ ] `lightweight_rpn.py` bridge working
- [ ] `test_rpn_tier1.py` passing (15+ tests)
- [ ] Latency verified <1µs

✅ **Tier 3 (Advanced)**:
- [ ] `modular_rpn_kernel_extended.ptx` integrated
- [ ] Matrix ops working (MATMUL, TRACE, DET, INV)
- [ ] Programmability working (BRANCH, LOOP, STORE/RECALL)
- [ ] Stack extensions working (NIP, TUCK, ROLL, DEPTH)
- [ ] `advanced_rpn.py` bridge working
- [ ] `test_rpn_tier3.py` passing (20+ tests)

✅ **Tiered Orchestrator**:
- [ ] `tiered_rpn.py` dispatcher working
- [ ] Automatic tier selection verified
- [ ] `test_tiered_rpn.py` passing (10+ tests)
- [ ] Statistics tracking working

✅ **Integration**:
- [ ] All existing 252 tests still passing
- [ ] New tests added (45+): 280+ total
- [ ] GPU memory <300MB
- [ ] Backwards compatibility verified

### Performance Targets

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Tier 1 latency | <1µs | `test_rpn_tier1.py::test_latency_target` |
| Tier 2 latency | ~3µs | Benchmark simple vector op |
| Tier 3 latency | ~10µs | Benchmark MATMUL 4×4 |
| Tier 1 size | ~10KB | `ls -lh modular_rpn_kernel_lite.ptx` |
| Tier 3 size | ~60KB | `ls -lh modular_rpn_kernel_extended.ptx` |
| Total tests | 280+ | `pytest tests/ -q` |
| GPU memory | <300MB | `nvidia-smi` during tests |

---

## What to Report Back

When Phase 2 is complete, report:

### 1. Tier 1 Status
- Kernel size (target 10KB)
- Latency measurement (target <1µs)
- Test results (`test_rpn_tier1.py`)
- Any optimization insights

### 2. Tier 3 Status
- Kernel size (target 60KB)
- Matrix ops working? (MATMUL tested with 2×2, 4×4)
- Programmability proven? (LOOP example working)
- Test results (`test_rpn_tier3.py`)

### 3. Orchestrator Status
- Automatic tier selection working?
- Statistics tracking output (tier1_percent, tier2_percent, tier3_percent)
- Backwards compatibility verified?

### 4. Integration
- Total tests passing (target 280+)
- GPU memory usage
- Any breaking changes?

### 5. Performance Summary
- Tier 1 vs. Tier 2 latency comparison (speedup)
- Matrix MATMUL performance (vs. NumPy baseline if available)
- System-wide impact (if measurable: ActionBuffer, ThinkingTag)

---

## Reference Files

- **Strategy analysis**: `/TEMP/RPN_KERNEL_STRATEGY_ANALYSIS.md`
- **HP 50g expansion plan**: `/TEMP/RPN_HP50G_EXPANSION_STRATEGY.md`
- **Existing kernels**:
  - `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx` (Tier 2 base)
  - `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx` (Tier 3 base)
- **Existing bridge**: `knowledge3d/cranium/bridges/sovereign_bridges.py`

---

## Notes

- **Keep Tier 2 unchanged**: Current 34KB kernel stays as-is (252 tests depend on it)
- **Tier 1 = subset of Tier 2**: Extract, don't rewrite
- **Tier 3 = extended + additions**: Activate dormant kernel, add 30 new ops
- **Orchestrator = drop-in replacement**: Existing code works without changes

**Apollo 11 Principle**: Three stages (booster, CSM, LEM). Each optimized for different mission phases. Don't use LEM in orbit! 🚀

---

**Proceed with Phase 2!** 🎯
