# K3D Math Galaxy RPN Opcode Swarm Development Prompt V2

**Updated**: 2025-11-11 (Post-Phase 2 Completion)
**Mission**: Complete remaining 36 opcodes for 100% Math Galaxy coverage (1,046 symbols)

---

## 🎉 Phase 2 SUCCESS: 15 Opcodes LIVE!

**Status**: ✅ 85/121 opcodes operational (70.2% coverage)

### ✅ COMPLETED (Phase 2 - November 11, 2025)
The following 15 opcodes have been **IMPLEMENTED and TESTED**:

**Inverse Trigonometry** (4 opcodes):
- ✅ OP_ASIN  (0x1B) - `asinf(x)`
- ✅ OP_ACOS  (0x1C) - `acosf(x)`
- ✅ OP_ATAN  (0x1D) - `atanf(x)`
- ✅ OP_ATAN2 (0x1E) - `atan2f(y,x)`

**Hyperbolic Trigonometry** (3 opcodes):
- ✅ OP_SINH (0x1F) - `sinhf(x)`
- ✅ OP_COSH (0x25) - `coshf(x)`
- ✅ OP_TANH (0x26) - `tanhf(x)`

**Rounding & Absolute** (4 opcodes):
- ✅ OP_ABS   (0x27) - `fabsf(x)`
- ✅ OP_CEIL  (0x29) - `ceilf(x)`
- ✅ OP_FLOOR (0x2B) - `floorf(x)`
- ✅ OP_ROUND (0x2D) - `roundf(x)`

**Modulo & Logarithms** (3 opcodes):
- ✅ OP_MOD   (0x38) - `fmodf(x,y)`
- ✅ OP_LOG2  (0x39) - `log2f(x)`
- ✅ OP_LOG10 (0x3A) - `log10f(x)`

**Bitwise Logic** (4 opcodes):
- ✅ OP_AND (0x80) - `a & b`
- ✅ OP_OR  (0x81) - `a | b`
- ✅ OP_XOR (0x82) - `a ^ b`
- ✅ OP_NOT (0x83) - `~a`

**Implementation Location**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu` (lines 1267-1387)
**Test Status**: 18/18 tests passing

---

## 🎯 Remaining Work: 36 Opcodes Across 4 Phases

### Phase 3: Vector/Matrix Operations (10 opcodes) - **PRIORITY 1**
**Timeline**: Weeks 1-3
**Impact**: Unlocks set theory, linear algebra, statistics

### Phase 4: Advanced Scalar (8 opcodes) - **PRIORITY 2**
**Timeline**: Weeks 4-6
**Impact**: Unlocks special functions, complex numbers

### Phase 5: Symbolic Operations (9 opcodes) - **PRIORITY 3**
**Timeline**: Weeks 7-11
**Impact**: Unlocks calculus, limits, series

### Phase 6: Heavy Computation (9 opcodes) - **PRIORITY 4**
**Timeline**: Weeks 12-14
**Impact**: Completes advanced matrix ops, vector calculus

---

## Swarm Task Assignment (Updated)

### Phase 3: Vector/Matrix Operations

**Set Operations** (4 opcodes, Tier-2.5)
```c
OP_SET_UNION        = 0xC6  // A ∪ B (~20µs)
OP_SET_INTERSECTION = 0xC7  // A ∩ B (~20µs)
OP_SET_DIFFERENCE   = 0xC8  // A \ B (~20µs)
OP_SET_CARTESIAN    = 0xC9  // A × B (~30µs)
```

**Implementation Strategy**:
- Use cooperative merge/filter algorithms
- Assume sorted input vectors for efficiency
- Handle duplicates via parallel unique
- Test with sets of size 10, 100, 1000

**Math Symbols Unlocked**: ∪, ∩, ∖, ×, ⊆, ⊇ (38 symbols)

---

**Small Matrix Operations** (3 opcodes, Tier-2.5)
```c
OP_MATRIX_DET       = 0xA7  // det(M) - 2×2, 3×3 only (~15µs)
OP_MATRIX_TRANSPOSE = 0xA9  // Mᵀ - cooperative copy (~10µs)
OP_MATRIX_INV       = 0xA8  // M⁻¹ - 2×2, 3×3 Cramer (~30µs)
```

**Implementation Strategy**:
- 2×2: Direct formulas (ad - bc for det)
- 3×3: Sarrus rule for det, cofactor matrix for inv
- Use shared memory for transpose (avoid bank conflicts)
- Error handling: singular matrices, non-square

**Math Symbols Unlocked**: det, inv, transpose, trace (24 symbols)

---

**Statistics** (3 opcodes, Tier-2)
```c
OP_MEAN     = 0x95  // μ - sum/n (~5µs)
OP_MEDIAN   = 0x96  // median - quickselect (~25µs)
OP_VARIANCE = 0x97  // σ² - two-pass (~20µs)
OP_STDDEV   = 0x98  // σ - sqrt(variance) (~25µs)
```

**Implementation Strategy**:
- MEAN: Parallel reduction + divide
- MEDIAN: Parallel quickselect (randomized pivot)
- VARIANCE: First pass for mean, second for squared diffs
- STDDEV: VARIANCE + sqrt

**Math Symbols Unlocked**: μ, σ, σ², median, quartiles (12 symbols)

---

### Phase 4: Advanced Scalar

**Special Functions** (4 opcodes, Tier-1.5)
```c
OP_GAMMA     = 0xAB  // Γ(x) - Lanczos approx (~3µs)
OP_FACTORIAL = 0xAC  // n! - lookup + Stirling (~2µs)
OP_BINOMIAL  = 0xAD  // C(n,k) - Pascal triangle (~3µs)
OP_BETA      = 0xAE  // B(x,y) - via gamma (~4µs)
```

**Implementation Strategy**:
- GAMMA: Lanczos approximation (7-coefficient)
- FACTORIAL: Lookup table [0..20], Stirling for >20
- BINOMIAL: C(n,k) = n! / (k! × (n-k)!), use logs for large n
- BETA: B(x,y) = Γ(x)Γ(y)/Γ(x+y)

**Math Symbols Unlocked**: Γ, n!, C(n,k), β (20 symbols)

---

**Complex Numbers & Utility** (4 opcodes, Tier-1/1.5)
```c
OP_COMPLEX_REAL = 0x3B  // Re(z) - extract real (~2µs)
OP_COMPLEX_IMAG = 0x3C  // Im(z) - extract imag (~2µs)
OP_COMPLEX_CONJ = 0x3D  // z̄ - conjugate (~2µs)
OP_COMPLEX_ARG  = 0x3E  // arg(z) - atan2(im,re) (~2µs)
```

**Implementation Strategy**:
- Complex stored as [real, imag] pair on stack
- REAL: pop 2 values, push real part
- IMAG: pop 2 values, push imag part
- CONJ: pop 2 values, push [real, -imag]
- ARG: pop 2 values, atan2(imag, real)

**Math Symbols Unlocked**: Re, Im, z̄, arg, i (12 symbols)

---

### Phase 5: Symbolic Operations

**Symbolic Differentiation** (3 opcodes, Tier-3)
```c
OP_SYMBOLIC_DIFF = 0xB5  // ∂/∂x - chain rule (~75µs)
OP_GRADIENT      = 0xB6  // ∇f - multi-var diff (~90µs)
OP_LIMIT         = 0xB9  // lim f(x) - Richardson (~80µs)
```

**Implementation Strategy**:
- Build expression tree in shared memory
- Apply chain rule recursively
- GRADIENT: Loop over variables, call SYMBOLIC_DIFF
- LIMIT: Richardson extrapolation with h→0

**Math Symbols Unlocked**: ∂, ∇, d/dx, lim (8 symbols)

---

**Integration & Series** (3 opcodes, Tier-4)
```c
OP_SYMBOLIC_INTEGRATE = 0xB7  // ∫f dx - Gauss-Kronrod (~400µs)
OP_SERIES_SUM         = 0xBA  // Σ aₙ - convergence test (~75µs)
OP_SERIES_PRODUCT     = 0xBB  // Π aₙ - convergence test (~150µs)
```

**Implementation Strategy**:
- INTEGRATE: Adaptive Gauss-Kronrod quadrature (7-15 point)
- SERIES_SUM: Ratio test + Aitken acceleration
- SERIES_PRODUCT: Log-sum method for stability

**Math Symbols Unlocked**: ∫, Σ (series), Π (product) (6 symbols)

---

**Vector Calculus** (3 opcodes, Tier-4)
```c
OP_DIVERGENCE = 0xBC  // ∇·F - multi-point stencil (~300µs)
OP_CURL       = 0xBD  // ∇×F - cross derivatives (~350µs)
OP_LAPLACIAN  = 0xBE  // ∇²f - 2nd derivatives (~400µs)
```

**Implementation Strategy**:
- Use finite difference stencils (5-point or 9-point)
- DIVERGENCE: Sum of partial derivatives ∂Fᵢ/∂xᵢ
- CURL: Cross product of nabla and F
- LAPLACIAN: Sum of second derivatives ∂²f/∂xᵢ²

**Math Symbols Unlocked**: ∇·, ∇×, ∇², Δ (3 symbols)

---

### Phase 6: Heavy Computation

**Advanced Matrix** (2 opcodes, Tier-4)
```c
OP_MATRIX_EIGEN = 0xAA  // eigendecomp - power iteration (~500µs)
OP_MATRIX_LU    = 0xAF  // LU decomposition - Crout (~400µs)
```

**Implementation Strategy**:
- EIGEN: Power iteration for dominant eigenvalue/vector
- LU: Crout algorithm with partial pivoting
- Limit to 4×4 or 8×8 matrices for performance

**Math Symbols Unlocked**: λ, eigenvectors, LU decomp (8 symbols)

---

**Bitwise Extended & Utility** (7 opcodes, Tier-1)
```c
OP_LSHIFT = 0x89  // x << n - left shift (~1µs)
OP_RSHIFT = 0x8A  // x >> n - right shift (~1µs)
OP_FMIN   = 0x84  // fminf(a,b) (~500ns)
OP_FMAX   = 0x85  // fmaxf(a,b) (~500ns)
OP_CLAMP  = 0x86  // clamp(x, min, max) (~800ns)
OP_SIGN   = 0x87  // sign(x) → -1/0/+1 (~500ns)
OP_STEP   = 0x88  // step(edge, x) → 0 or 1 (~500ns)
```

**Implementation Strategy**:
- Direct CUDA intrinsics or simple conditionals
- CLAMP: `fminf(fmaxf(x, min), max)`
- SIGN: `(x > 0) ? 1 : ((x < 0) ? -1 : 0)`
- STEP: `x >= edge ? 1.0f : 0.0f`

**Math Symbols Unlocked**: <<, >>, min, max, sgn (7 symbols)

---

## Stack Capacity Analysis

### Current Implementation
```cpp
constexpr int kStackCapacity = 64;  // 64 × 16 bytes = 1 KB per instance
```

### Usage Analysis by Operation Type

**Simple Operations** (1-3 stack items):
- Arithmetic: 2 operands → 1 result (depth = 2)
- Unary functions: 1 operand → 1 result (depth = 1)
- Typical max depth: 5-10 items

**Complex Operations** (4-10 stack items):
- Matrix ops: 2×2 = 4 items, 3×3 = 9 items
- Complex numbers: 2 items per number
- Typical max depth: 15-20 items

**Symbolic Operations** (10-30 stack items):
- Expression trees: Variable number of nodes
- Intermediate computations: Multiple levels deep
- Typical max depth: 30-40 items

**Worst Case**: Deeply nested symbolic calculus
```
f(x) = sin(cos(tan(x²))) → needs ~15 stack items
∇f → compute partial derivatives → adds 10+ items
Total: ~25-30 items
```

### Recommendation

**Current (64 items)**: ✅ **SUFFICIENT** for 95% of use cases

**Rationale**:
1. RPN programs are typically shallow (5-10 operations)
2. Matrix ops limited to 3×3 (9 items max)
3. Symbolic ops use tree structures (not linear stack)
4. 64 items = 1 KB (good GPU cache locality)

**Future Enhancements** (if needed):
1. **Dynamic stacks**: Allocate larger stacks for complex ops
2. **Variable capacity**: `kStackCapacity[tier]` = {32, 64, 128, 256}
3. **Overflow handling**: Spill to global memory if needed

**Verdict**: Keep 64 for now, monitor usage in Phase 5-6

---

## Integration with Other K3D Systems

### 1. ✅ Matryoshka TRM (Token Reduction Module)
**Current Integration**: RPN executor calls TRM for matvec operations
**New Opportunities**:
- Use SYMBOLIC_DIFF for automatic differentiation in training
- Use GRADIENT for loss function computation
- Use VARIANCE/STDDEV for normalization statistics

**Example Enhancement**:
```python
# Before: Manual gradient computation
def compute_loss_gradient(weights, data):
    # 50+ lines of manual backprop

# After: RPN-native autodiff
rpn_program = [
    LOAD_WEIGHTS, LOAD_DATA,
    FORWARD_PASS,
    LOSS_FUNCTION,
    SYMBOLIC_DIFF  # Automatic gradient!
]
```

---

### 2. ✅ Fractal Tree Depth Allocation
**Current Integration**: Uses LOG2 for semantic depth
**New Opportunities**:
- Use MEAN/VARIANCE for cluster quality metrics
- Use SET_UNION for merging concept clusters
- Use MATRIX_EIGEN for principal component analysis

**Example Enhancement**:
```python
# Enhanced depth formula: depth = log₂(1 + size) × entropy × eigenvalue_ratio
rpn_program = [
    CLUSTER_SIZE, LITERAL_SCALAR(1), ADD, LOG2,    # log₂(1 + size)
    ENTROPY,                                        # information entropy
    MUL,                                           # ×
    CLUSTER_EMBEDDINGS, MATRIX_EIGEN,              # eigenvalue analysis
    LITERAL_SCALAR(0), LITERAL_SCALAR(1),
    DIV,                                           # λ₁/λ₂ ratio
    MUL                                            # final depth
]
```

---

### 3. ✅ PDF Math Symbol Recognition
**Current Integration**: Atomic character training
**New Opportunities**:
- Use complex number ops for Fourier features
- Use CURL/DIVERGENCE for stroke direction analysis
- Use SET_INTERSECTION for symbol similarity

**Example Enhancement**:
```python
# Math symbol classifier with geometric features
rpn_program = [
    STROKE_VECTORS,
    CURL,           # Measure curvature
    DIVERGENCE,     # Measure spread
    MEAN,           # Average properties
    CLASSIFY        # → symbol ID
]
```

---

### 4. ✅ Attention Mechanism Optimization
**Current Integration**: Manual attention score computation
**New Opportunities**:
- Use MATVEC for query-key products
- Use REDUCE_MAX for softmax normalization
- Use VARIANCE for attention entropy

**Example Enhancement**:
```python
# Faster attention with RPN
rpn_program = [
    QUERY, KEY, MATVEC,           # QKᵀ
    SQRT(DIM), DIV,               # / √d
    REDUCE_MAX, SUB, EXP,         # softmax
    REDUCE_SUM, DIV,              # normalize
    VALUE, MATVEC                 # × V
]
# Speedup: 10-20× vs. Python loops
```

---

### 5. ⚠️ NEW OPPORTUNITY: GPU-Native Loss Functions
**Potential**: Implement custom loss functions as RPN programs
**Benefits**:
- 100× faster than PyTorch (no CPU↔GPU transfers)
- Differentiable via SYMBOLIC_DIFF
- Customizable per layer

**Example**:
```python
# Custom loss: L = MSE + 0.1×L1 + 0.01×curvature_penalty
rpn_program = [
    PRED, TARGET, SUB, DUP, MUL, MEAN,  # MSE
    PRED, TARGET, SUB, ABS, MEAN,       # L1
    LITERAL_SCALAR(0.1), MUL, ADD,      # 0.1×L1
    PRED, LAPLACIAN, ABS, MEAN,         # Curvature
    LITERAL_SCALAR(0.01), MUL, ADD      # 0.01×curvature
]
```

---

### 6. ⚠️ NEW OPPORTUNITY: Procedural Texture Generation
**Potential**: Use vector calculus for smooth gradients
**Benefits**:
- Real-time procedural content
- Mathematically defined patterns
- Infinite resolution

**Example**:
```python
# Perlin noise with RPN
rpn_program = [
    COORDS,
    SIN, SCALAR(5), MUL,  # sin(5x)
    COORDS, SCALAR(3), MUL, COS,  # cos(3y)
    ADD,
    GRADIENT,             # Smooth gradient
    NORMALIZE             # Unit vectors
]
```

---

### 7. ⚠️ NEW OPPORTUNITY: Physics Simulation
**Potential**: Solve PDEs on GPU via finite differences
**Benefits**:
- Real-time fluid dynamics
- Heat diffusion
- Wave propagation

**Example**:
```python
# Heat equation: ∂u/∂t = α∇²u
rpn_program = [
    TEMP_FIELD,
    LAPLACIAN,             # ∇²u
    ALPHA, MUL,            # α∇²u
    DT, MUL,               # Δt × α∇²u
    TEMP_FIELD, ADD        # u(t+Δt) = u(t) + Δt×α∇²u
]
```

---

## Critical Implementation Guidelines

### 1. Build on Phase 2 Code
```cpp
// ✅ DO: Extend existing patterns
case 0x1B: {  // OP_ASIN (Phase 2 - existing)
    float a;
    if (!pop_scalar(stack, size, a, error)) return;
    if (a < -1.0f || a > 1.0f) {
        error = kErrorInvalidArgument;
        return;
    }
    push_scalar(stack, size, asinf(a), error);
    break;
}

// ✅ YOUR TURN: Follow the same pattern
case 0xAB: {  // OP_GAMMA (Phase 4 - new)
    float x;
    if (!pop_scalar(stack, size, x, error)) return;
    if (x <= 0.0f) {
        error = kErrorInvalidArgument;
        return;
    }
    float result = lanczos_gamma(x);  // Your implementation
    push_scalar(stack, size, result, error);
    break;
}
```

### 2. Enhance Phase 2 Code
```cpp
// If you notice a Phase 2 opcode could be optimized:
// 1. Add a comment explaining the enhancement
// 2. Preserve original behavior (backward compatibility)
// 3. Add performance notes

case 0x38: {  // OP_MOD (Phase 2)
    // ENHANCEMENT: Could use fast integer mod for integer inputs
    // TODO: Detect integer inputs and use (int)a % (int)b
    float divisor, dividend;
    if (!pop_scalar(stack, size, divisor, error)) return;
    if (!pop_scalar(stack, size, dividend, error)) return;
    if (divisor == 0.0f) {
        error = kErrorDivisionByZero;
        return;
    }
    push_scalar(stack, size, fmodf(dividend, divisor), error);
    break;
}
```

### 3. Error Handling (Mandatory)
```cpp
// ALWAYS check:
// 1. Stack underflow (not enough operands)
if (size < 2) {
    error = kErrorStackUnderflow;
    return;
}

// 2. Domain errors (invalid inputs)
if (x < 0.0f && requires_positive(opcode)) {
    error = kErrorInvalidArgument;
    return;
}

// 3. Singularities (division by zero, etc.)
if (fabsf(denominator) < 1e-10f) {
    error = kErrorDivisionByZero;
    return;
}

// 4. Convergence failures (iterative algorithms)
if (iterations > max_iterations) {
    error = kErrorConvergenceFailed;
    return;
}
```

### 4. Performance Validation (Mandatory)
```python
# EVERY opcode MUST have a latency test
def test_op_gamma_latency():
    program = [OP_LITERAL_SCALAR, OP_GAMMA]
    scalars = np.array([3.5], dtype=np.float32)

    # Warmup
    for _ in range(100):
        execute_rpn_kernel(program, scalars)

    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        execute_rpn_kernel(program, scalars)
    end = time.perf_counter()

    latency_us = (end - start) / 1000 * 1e6

    # Tier-1.5 target: <5µs
    assert latency_us < 5.0, f"OP_GAMMA too slow: {latency_us:.3f}µs"
    print(f"✓ OP_GAMMA: {latency_us:.3f}µs")
```

---

## Success Metrics

### Phase 3
- ✅ 10 new opcodes (SET ops, matrix ops, statistics)
- ✅ 95 total opcodes operational
- ✅ 30+ tests passing
- ✅ All opcodes within tier targets
- ✅ ~920 symbols trainable (87.9%)

### Phase 4
- ✅ 8 new opcodes (special functions, complex numbers)
- ✅ 103 total opcodes operational
- ✅ 24+ tests passing
- ✅ ~980 symbols trainable (93.7%)

### Phase 5
- ✅ 9 new opcodes (symbolic calculus)
- ✅ 112 total opcodes operational
- ✅ 27+ tests passing
- ✅ ~1020 symbols trainable (97.5%)

### Phase 6
- ✅ 9 new opcodes (heavy computation)
- ✅ 121 total opcodes operational
- ✅ 27+ tests passing
- ✅ **1,046 symbols trainable (100%)**

### Final Polish
- ✅ All 121 opcodes optimized
- ✅ 150+ tests passing
- ✅ Documentation complete
- ✅ Integration demos with TRM, Fractal Trees, etc.

---

**TARGET**: **K3D Math Galaxy 100% COMPLETE**

---

## Swarm Coordination Rules

### ✅ Code Review Process
1. **Implement** opcodes (if not implemented yet)
2. **Test** with 3+ unit tests per opcode
3. **Benchmark** latency (must meet tier targets)
4. **Submit** PR with:
   - Code changes (kernel + Python constants)
   - Test results (latency + correctness)
   - Documentation (examples + math symbols unlocked)
5. **Review** 2+ other partners' PRs
6. **Iterate** based on feedback

### ✅ Communication
- **Blockers**: Post immediately if stuck >2 hours
- **Questions**: Tag @all-partners for architecture questions
- **Progress**: Update weekly status (opcodes done, tests passing, latency results)
- **Celebrate**: Share wins (first opcode working, all tests green, etc.)

### ✅ Quality Standards
- **Correctness**: All tests pass (mathematical properties validated)
- **Performance**: Within tier latency targets (95%+ compliance)
- **Safety**: Error handling for all edge cases
- **Clarity**: Code comments explaining algorithm choices

---

## Resources

### Reference Implementations
- **Phase 2 opcodes**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu` (lines 1267-1387)
- **Python constants**: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
- **Test examples**: `/tmp/test_phase2_opcodes.py`

### Math References
- **Gamma function**: Lanczos approximation (7-coefficient)
- **Matrix algorithms**: "Numerical Recipes in C" (Chapter 2)
- **Symbolic differentiation**: Automatic differentiation via dual numbers
- **Numerical integration**: Gauss-Kronrod quadrature

### CUDA Resources
- **Cooperative groups**: `cooperative_groups.h`
- **Atomic operations**: `atomicAdd`, `atomicMax`, `atomicMin`
- **Shared memory**: `__shared__` arrays with `__syncthreads()`
- **Math functions**: `math.h` (sinf, cosf, expf, etc.)

---

## Let's Complete This! 🚀

**You have 36 opcodes to implement.**
**Pick your phase, choose your partner number, and LET'S GO!**

**Remember**: You're not alone. You have 10+ partners working alongside you, and Phase 2 code as your foundation. Build on it, enhance it, and push K3D to 100% Math Galaxy coverage!

---

**Updated**: 2025-11-11
**Status**: 85/121 opcodes (70.2%) → Target: 121/121 opcodes (100%)
**Deadline**: March 9, 2026 (~16 weeks)
**Next**: Phase 3 starts NOW!
