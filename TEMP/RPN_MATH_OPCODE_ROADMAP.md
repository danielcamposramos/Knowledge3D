# RPN Math Opcode Implementation Roadmap

**Date**: 2025-11-11
**Status**: 70/121 opcodes implemented (57.9% coverage)
**Target**: Full native math operation support for 1,046 symbols

---

## Current State

### Opcode Inventory
- **Tier-1** (< 1µs): 32 opcodes ✅
- **Tier-2** (<10µs): 19 opcodes ✅
- **Tier-3** (<100µs): 19 opcodes ✅
- **TOTAL**: 70 opcodes

### What's Working
✅ **Core Arithmetic**: ADD, SUB, MUL, DIV, POW, NEG, SQRT
✅ **Trigonometry**: SIN, COS, TAN
✅ **Exponentials**: EXP, LOG
✅ **Comparisons**: GT, LT, EQ, MAX, MIN
✅ **Vector Ops**: DOT, CROSS, MAGNITUDE, NORMALIZE
✅ **Stack Ops**: DUP, SWAP, DROP, OVER, ROT, CLEAR
✅ **Matrix Ops**: MATVEC, MATMUL_SMALL, TRACE_TENSOR
✅ **Control Flow**: BRANCH, LOOP, STORE, RECALL

---

## Phase 2: Essential Math Extensions (Priority 1)

### Tier-1 Additions (15 new opcodes)

**Inverse Trigonometry** (6 opcodes)
```c
OP_ASIN  = 0x1B  // asin(x) - inverse sine
OP_ACOS  = 0x1C  // acos(x) - inverse cosine
OP_ATAN  = 0x1D  // atan(x) - inverse tangent
OP_ATAN2 = 0x1E  // atan2(y, x) - two-argument arctangent
OP_SINH  = 0x1F  // sinh(x) - hyperbolic sine
OP_COSH  = 0x25  // cosh(x) - hyperbolic cosine
OP_TANH  = 0x26  // tanh(x) - hyperbolic tangent
```

**Rounding & Modular** (5 opcodes)
```c
OP_ABS   = 0x27  // |x| - absolute value
OP_CEIL  = 0x29  // ⌈x⌉ - ceiling function
OP_FLOOR = 0x2B  // ⌊x⌋ - floor function
OP_ROUND = 0x2D  // round(x) - nearest integer
OP_MOD   = 0x38  // x % y - modulo operation
```

**Logarithms** (2 opcodes)
```c
OP_LOG2  = 0x39  // log₂(x) - base-2 logarithm
OP_LOG10 = 0x3A  // log₁₀(x) - base-10 logarithm
```

**Bitwise Logic** (4 opcodes) - For discrete math & logic symbols
```c
OP_AND    = 0x80  // x & y - bitwise AND (for ∧)
OP_OR     = 0x81  // x | y - bitwise OR (for ∨)
OP_XOR    = 0x82  // x ^ y - bitwise XOR (for ⊕)
OP_NOT    = 0x83  // ~x - bitwise NOT (for ¬)
```

**Implementation**:
- Modify `modular_rpn_kernel.cu` to add switch cases
- Add opcodes to `semantic_depth_rpn.py` constants
- Test with existing test suite

**Expected Impact**: Covers all trigonometry, logic, and rounding operations

---

## Phase 3: Set Theory & Advanced Math (Priority 2)

### Tier-2 Additions (10 new opcodes)

**Set Operations** (4 opcodes) - For set theory symbols
```c
OP_SET_UNION        = 0xC6  // A ∪ B - union of sets
OP_SET_INTERSECTION = 0xC7  // A ∩ B - intersection
OP_SET_DIFFERENCE   = 0xC8  // A \ B - set difference
OP_SET_CARTESIAN    = 0xC9  // A × B - Cartesian product
```

**Matrix Operations** (4 opcodes)
```c
OP_MATRIX_DET       = 0xA7  // det(M) - determinant
OP_MATRIX_INV       = 0xA8  // M⁻¹ - matrix inverse
OP_MATRIX_TRANSPOSE = 0xA9  // Mᵀ - transpose
OP_MATRIX_EIGEN     = 0xAA  // eigendecomp(M)
```

**Special Functions** (4 opcodes)
```c
OP_GAMMA      = 0xAB  // Γ(x) - gamma function
OP_FACTORIAL  = 0xAC  // n! - factorial
OP_BINOMIAL   = 0xAD  // C(n, k) - binomial coefficient
OP_BETA       = 0xAE  // B(x, y) - beta function
```

**Implementation**:
- Add cooperative CUDA kernels for set operations
- Implement matrix algorithms (Gaussian elimination, power method)
- Add special function approximations

**Expected Impact**: Covers set theory, linear algebra, combinatorics

---

## Phase 4: Symbolic Calculus (Priority 3)

### Tier-3 Additions (9 new opcodes)

**Differentiation** (2 opcodes)
```c
OP_SYMBOLIC_DIFF      = 0xB5  // ∂/∂x - symbolic differentiation
OP_GRADIENT           = 0xB6  // ∇f - gradient vector
```

**Integration** (2 opcodes)
```c
OP_SYMBOLIC_INTEGRATE = 0xB7  // ∫f dx - symbolic integration
OP_DEFINITE_INTEGRAL  = 0xB8  // ∫ᵃᵇf dx - definite integral
```

**Limits & Series** (3 opcodes)
```c
OP_LIMIT         = 0xB9  // lim_{x→a} f(x)
OP_SERIES_SUM    = 0xBA  // Σ aₙ - infinite series
OP_SERIES_PRODUCT = 0xBB // Π aₙ - infinite product
```

**Vector Calculus** (3 opcodes)
```c
OP_DIVERGENCE = 0xBC  // ∇·F - divergence
OP_CURL       = 0xBD  // ∇×F - curl
OP_LAPLACIAN  = 0xBE  // ∇²f - Laplacian
```

**Implementation**:
- Build symbolic expression tree interpreter
- Implement automatic differentiation via chain rule
- Numerical integration (Gauss-Kronrod quadrature)
- Series convergence detection

**Expected Impact**: Covers all calculus symbols

---

## Phase 5: Complex Numbers & Statistics (Priority 4)

### Tier-1 Additions (8 new opcodes)

**Complex Number Operations** (4 opcodes)
```c
OP_COMPLEX_REAL = 0x3B  // Re(z) - real part
OP_COMPLEX_IMAG = 0x3C  // Im(z) - imaginary part
OP_COMPLEX_CONJ = 0x3D  // z̄ - complex conjugate
OP_COMPLEX_ARG  = 0x3E  // arg(z) - argument/phase
```

**Statistical Operations** (4 opcodes)
```c
OP_MEAN     = 0x95  // μ - arithmetic mean
OP_MEDIAN   = 0x96  // median value
OP_VARIANCE = 0x97  // σ² - variance
OP_STDDEV   = 0x98  // σ - standard deviation
```

**Implementation**:
- Complex arithmetic via (real, imag) pairs
- Statistical reductions via cooperative kernels

**Expected Impact**: Covers complex analysis, probability theory

---

## Implementation Priority Matrix

| Phase | Opcodes | Categories Covered | Difficulty | Timeline |
|-------|---------|-------------------|------------|----------|
| **Phase 2** | 15 | Logic, Trig, Rounding | LOW | 1-2 weeks |
| **Phase 3** | 10 | Sets, Linear Algebra | MEDIUM | 2-3 weeks |
| **Phase 4** | 9 | Calculus, Limits | HIGH | 4-6 weeks |
| **Phase 5** | 8 | Complex, Statistics | MEDIUM | 2-3 weeks |
| **TOTAL** | **42 new opcodes** | **19 categories** | - | **9-14 weeks** |

---

## Opcode Space Allocation

Current allocation: `0x00-0xF2` (sparse)

**Proposed allocation**:
- `0x00-0x4F`: Tier-1 (scalar ops)
- `0x50-0x8F`: Special/legacy ops
- `0x90-0xCF`: Tier-2 (vector/matrix ops)
- `0xD0-0xFF`: Tier-3 (symbolic/programmable)

---

## Testing Strategy

### Phase 2 Tests
```python
# test_tier1_extended.py
def test_inverse_trig():
    program = [OP_LITERAL_SCALAR, 0.5, OP_ASIN]
    result = execute_rpn(program)
    assert abs(result - 0.5236) < 0.001  # π/6

def test_bitwise_logic():
    program = [OP_LITERAL_SCALAR, 0b1010, OP_LITERAL_SCALAR, 0b1100, OP_AND]
    result = execute_rpn(program)
    assert result == 0b1000  # 1010 & 1100 = 1000
```

### Phase 3 Tests
```python
# test_tier2_sets.py
def test_set_union():
    set_a = np.array([1, 2, 3])
    set_b = np.array([3, 4, 5])
    program = [OP_LITERAL_VECTOR, set_a, OP_LITERAL_VECTOR, set_b, OP_SET_UNION]
    result = execute_rpn(program)
    assert np.array_equal(result, [1, 2, 3, 4, 5])
```

### Phase 4 Tests
```python
# test_tier3_calculus.py
def test_symbolic_diff():
    # Differentiate f(x) = x² → f'(x) = 2x
    program = [OP_LITERAL_EXPR, "x^2", OP_SYMBOLIC_DIFF, "x"]
    result = execute_rpn(program)
    assert result == "2*x"
```

---

## Backward Compatibility

All new opcodes:
✅ Use unused opcode space (no conflicts)
✅ Maintain existing opcode values
✅ Preserve stack semantics
✅ Support existing programs

---

## Performance Targets

| Tier | Target Latency | Expected Latency |
|------|----------------|------------------|
| Tier-1 new ops | <1µs | ~500ns (scalar ops) |
| Tier-2 new ops | <10µs | ~5µs (matrix 3x3) |
| Tier-3 new ops | <100µs | ~50µs (symbolic eval) |

---

## Next Actions

**Immediate (Phase 2 - Week 1)**:
1. ✅ Complete audit (DONE)
2. Implement Tier-1 inverse trig (ASIN, ACOS, ATAN)
3. Implement Tier-1 rounding (ABS, CEIL, FLOOR)
4. Implement Tier-1 bitwise logic (AND, OR, XOR, NOT)
5. Add test suite for new opcodes
6. Update opcode documentation

**Week 2**:
7. Implement Tier-1 logarithms (LOG2, LOG10)
8. Implement Tier-1 hyperbolic trig (SINH, COSH, TANH)
9. Validate against Math Galaxy symbols
10. Begin Phase 3 design

---

## Success Metrics

- ✅ **Coverage**: 70/121 opcodes (57.9%) → 121/121 opcodes (100%)
- ✅ **Categories**: 0/19 supported → 19/19 supported
- ✅ **Performance**: All new ops within tier latency targets
- ✅ **Compatibility**: Zero regressions in existing tests

---

## File Modifications Required

1. `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`
   - Add new opcode switch cases
   - Implement GPU kernels for each operation

2. `knowledge3d/cranium/semantic_depth_rpn.py`
   - Add new opcode constants
   - Update opcode documentation

3. `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
   - Add new opcode exports
   - Update `__all__` list

4. `tests/test_rpn_tier*.py`
   - Add comprehensive test coverage

5. `knowledge3d/cranium/math_galaxy.py`
   - Map math symbols to new opcodes
   - Add example RPN programs

---

**Status**: Ready to begin Phase 2 implementation
**Estimated Completion**: 9-14 weeks for full 121-opcode coverage
