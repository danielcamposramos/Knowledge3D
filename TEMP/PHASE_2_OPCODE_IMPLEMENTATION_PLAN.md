# Phase 2 RPN Opcode Implementation Plan

**Date**: 2025-11-11
**Timeline**: 1-2 weeks
**Goal**: Add 15 essential opcodes to unlock 200+ additional math symbols
**Current**: 70 opcodes (57.9% coverage) → **Target**: 85 opcodes (70% coverage)

---

## Executive Summary

Phase 2 adds **15 Tier-1 opcodes** for essential mathematical operations:
- **6 inverse/hyperbolic trig opcodes** → Unlock 30 symbols
- **5 rounding/modular opcodes** → Unlock 15 symbols
- **4 bitwise logic opcodes** → Unlock 12 symbols

**Implementation Complexity**: LOW (all are standard CUDA math functions)
**Testing**: Straightforward (known mathematical properties)
**Impact**: Enables 80%+ of undergraduate mathematics

---

## Opcode Allocation Plan

### Current Tier-1 Space Usage
```
0x00-0x02: Literals
0x0A-0x0F: Arithmetic (ADD, SUB, MUL, DIV, POW, NEG)
0x14-0x1A: Math functions (SQRT, EXP, LOG, SIN, COS, TAN)
0x28-0x2F: Comparisons (GT, LT, EQ, MAX, MIN)
0x32-0x3F: Stack & vector ops
0x40-0x48: Special ops
```

### Phase 2 Allocations
```
0x1B-0x1E: Inverse trig (ASIN, ACOS, ATAN, ATAN2)
0x1F-0x26: Hyperbolic trig (SINH, COSH, TANH)
0x27-0x2D: Rounding (ABS, CEIL, FLOOR, ROUND)
0x38-0x3A: Logarithms (MOD, LOG2, LOG10)
0x80-0x83: Bitwise logic (AND, OR, XOR, NOT)
```

---

## Implementation Tasks

### Task 1: Inverse Trigonometry (6 opcodes)

**Opcodes**:
```c
OP_ASIN  = 0x1B  // asin(x) → [-π/2, π/2]
OP_ACOS  = 0x1C  // acos(x) → [0, π]
OP_ATAN  = 0x1D  // atan(x) → (-π/2, π/2)
OP_ATAN2 = 0x1E  // atan2(y, x) → [-π, π]
OP_SINH  = 0x1F  // sinh(x) = (eˣ - e⁻ˣ)/2
OP_COSH  = 0x25  // cosh(x) = (eˣ + e⁻ˣ)/2
OP_TANH  = 0x26  // tanh(x) = sinh(x)/cosh(x)
```

**CUDA Implementation** (modular_rpn_kernel.cu):
```cuda
case 0x1B: { // OP_ASIN
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    if (x < -1.0f || x > 1.0f) { errno_flag = 2; break; } // Domain error
    stack[sp++] = asinf(x);
    break;
}

case 0x1C: { // OP_ACOS
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    if (x < -1.0f || x > 1.0f) { errno_flag = 2; break; }
    stack[sp++] = acosf(x);
    break;
}

case 0x1D: { // OP_ATAN
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = atanf(x);
    break;
}

case 0x1E: { // OP_ATAN2
    if (sp < 2) { errno_flag = 1; break; }
    float x = stack[--sp];  // x (second arg)
    float y = stack[--sp];  // y (first arg)
    stack[sp++] = atan2f(y, x);
    break;
}

case 0x1F: { // OP_SINH
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = sinhf(x);
    break;
}

case 0x25: { // OP_COSH
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = coshf(x);
    break;
}

case 0x26: { // OP_TANH
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = tanhf(x);
    break;
}
```

**Python Constants** (semantic_depth_rpn.py):
```python
OP_ASIN = 0x1B
OP_ACOS = 0x1C
OP_ATAN = 0x1D
OP_ATAN2 = 0x1E
OP_SINH = 0x1F
OP_COSH = 0x25
OP_TANH = 0x26
```

**Test Cases** (test_tier1_extended.py):
```python
def test_inverse_trig():
    """Test inverse trigonometric functions"""
    # arcsin(0.5) = π/6 ≈ 0.5236
    program = [OP_LITERAL_SCALAR, 0.5, OP_ASIN]
    result = execute_rpn(program)
    assert abs(result - 0.5236) < 0.001

    # arccos(0.5) = π/3 ≈ 1.0472
    program = [OP_LITERAL_SCALAR, 0.5, OP_ACOS]
    result = execute_rpn(program)
    assert abs(result - 1.0472) < 0.001

    # arctan(1) = π/4 ≈ 0.7854
    program = [OP_LITERAL_SCALAR, 1.0, OP_ATAN]
    result = execute_rpn(program)
    assert abs(result - 0.7854) < 0.001

    # atan2(1, 1) = π/4
    program = [OP_LITERAL_SCALAR, 1.0, OP_LITERAL_SCALAR, 1.0, OP_ATAN2]
    result = execute_rpn(program)
    assert abs(result - 0.7854) < 0.001

def test_hyperbolic_trig():
    """Test hyperbolic functions"""
    # sinh(0) = 0
    program = [OP_LITERAL_SCALAR, 0.0, OP_SINH]
    result = execute_rpn(program)
    assert abs(result) < 0.0001

    # cosh(0) = 1
    program = [OP_LITERAL_SCALAR, 0.0, OP_COSH]
    result = execute_rpn(program)
    assert abs(result - 1.0) < 0.0001

    # tanh(0) = 0
    program = [OP_LITERAL_SCALAR, 0.0, OP_TANH]
    result = execute_rpn(program)
    assert abs(result) < 0.0001
```

**Unlocked Symbols**: arcsin, arccos, arctan, arcsec, arccsc, arccot, sinh, cosh, tanh, sech, csch, coth

---

### Task 2: Rounding & Absolute Value (5 opcodes)

**Opcodes**:
```c
OP_ABS   = 0x27  // |x|
OP_CEIL  = 0x29  // ⌈x⌉
OP_FLOOR = 0x2B  // ⌊x⌋
OP_ROUND = 0x2D  // round(x)
OP_MOD   = 0x38  // x mod y
```

**CUDA Implementation**:
```cuda
case 0x27: { // OP_ABS
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = fabsf(x);
    break;
}

case 0x29: { // OP_CEIL
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = ceilf(x);
    break;
}

case 0x2B: { // OP_FLOOR
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = floorf(x);
    break;
}

case 0x2D: { // OP_ROUND
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    stack[sp++] = roundf(x);
    break;
}

case 0x38: { // OP_MOD
    if (sp < 2) { errno_flag = 1; break; }
    float y = stack[--sp];  // Divisor
    float x = stack[--sp];  // Dividend
    if (y == 0.0f) { errno_flag = 3; break; }  // Division by zero
    stack[sp++] = fmodf(x, y);
    break;
}
```

**Test Cases**:
```python
def test_rounding_functions():
    """Test rounding and absolute value"""
    # |−5| = 5
    program = [OP_LITERAL_SCALAR, -5.0, OP_ABS]
    result = execute_rpn(program)
    assert result == 5.0

    # ⌈3.2⌉ = 4
    program = [OP_LITERAL_SCALAR, 3.2, OP_CEIL]
    result = execute_rpn(program)
    assert result == 4.0

    # ⌊3.8⌋ = 3
    program = [OP_LITERAL_SCALAR, 3.8, OP_FLOOR]
    result = execute_rpn(program)
    assert result == 3.0

    # round(3.5) = 4
    program = [OP_LITERAL_SCALAR, 3.5, OP_ROUND]
    result = execute_rpn(program)
    assert result == 4.0

    # 7 mod 3 = 1
    program = [OP_LITERAL_SCALAR, 7.0, OP_LITERAL_SCALAR, 3.0, OP_MOD]
    result = execute_rpn(program)
    assert result == 1.0
```

**Unlocked Symbols**: |x|, ⌈x⌉, ⌊x⌋, mod, %

---

### Task 3: Logarithms (2 opcodes)

**Opcodes**:
```c
OP_LOG2  = 0x39  // log₂(x)
OP_LOG10 = 0x3A  // log₁₀(x)
```

**CUDA Implementation**:
```cuda
case 0x39: { // OP_LOG2
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    if (x <= 0.0f) { errno_flag = 2; break; }  // Domain error
    stack[sp++] = log2f(x);
    break;
}

case 0x3A: { // OP_LOG10
    if (sp < 1) { errno_flag = 1; break; }
    float x = stack[--sp];
    if (x <= 0.0f) { errno_flag = 2; break; }
    stack[sp++] = log10f(x);
    break;
}
```

**Test Cases**:
```python
def test_logarithms():
    """Test logarithm functions"""
    # log₂(8) = 3
    program = [OP_LITERAL_SCALAR, 8.0, OP_LOG2]
    result = execute_rpn(program)
    assert abs(result - 3.0) < 0.0001

    # log₁₀(1000) = 3
    program = [OP_LITERAL_SCALAR, 1000.0, OP_LOG10]
    result = execute_rpn(program)
    assert abs(result - 3.0) < 0.0001
```

---

### Task 4: Bitwise Logic (4 opcodes)

**Opcodes**:
```c
OP_AND = 0x80  // x ∧ y (bitwise AND)
OP_OR  = 0x81  // x ∨ y (bitwise OR)
OP_XOR = 0x82  // x ⊕ y (bitwise XOR)
OP_NOT = 0x83  // ¬x (bitwise NOT)
```

**CUDA Implementation**:
```cuda
case 0x80: { // OP_AND
    if (sp < 2) { errno_flag = 1; break; }
    int b = (int)stack[--sp];
    int a = (int)stack[--sp];
    stack[sp++] = (float)(a & b);
    break;
}

case 0x81: { // OP_OR
    if (sp < 2) { errno_flag = 1; break; }
    int b = (int)stack[--sp];
    int a = (int)stack[--sp];
    stack[sp++] = (float)(a | b);
    break;
}

case 0x82: { // OP_XOR
    if (sp < 2) { errno_flag = 1; break; }
    int b = (int)stack[--sp];
    int a = (int)stack[--sp];
    stack[sp++] = (float)(a ^ b);
    break;
}

case 0x83: { // OP_NOT
    if (sp < 1) { errno_flag = 1; break; }
    int a = (int)stack[--sp];
    stack[sp++] = (float)(~a);
    break;
}
```

**Test Cases**:
```python
def test_bitwise_logic():
    """Test bitwise logic operations"""
    # 1010 ∧ 1100 = 1000 (10 AND 12 = 8)
    program = [OP_LITERAL_SCALAR, 10.0, OP_LITERAL_SCALAR, 12.0, OP_AND]
    result = execute_rpn(program)
    assert result == 8.0

    # 1010 ∨ 1100 = 1110 (10 OR 12 = 14)
    program = [OP_LITERAL_SCALAR, 10.0, OP_LITERAL_SCALAR, 12.0, OP_OR]
    result = execute_rpn(program)
    assert result == 14.0

    # 1010 ⊕ 1100 = 0110 (10 XOR 12 = 6)
    program = [OP_LITERAL_SCALAR, 10.0, OP_LITERAL_SCALAR, 12.0, OP_XOR]
    result = execute_rpn(program)
    assert result == 6.0

    # ¬1 = -2 (bitwise NOT of 1)
    program = [OP_LITERAL_SCALAR, 1.0, OP_NOT]
    result = execute_rpn(program)
    assert result == -2.0
```

**Unlocked Symbols**: ∧, ∨, ¬, ⊕, ⇒, ⇔

---

## File Modification Checklist

### 1. Update CUDA Kernel
**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`
- [ ] Add 15 new case statements in the main switch
- [ ] Add domain checking for inverse trig (x ∈ [-1, 1])
- [ ] Add error handling for MOD (division by zero)
- [ ] Compile and verify no syntax errors

### 2. Update Python Constants
**File**: `knowledge3d/cranium/semantic_depth_rpn.py`
- [ ] Add 15 new opcode constants
- [ ] Update opcode documentation strings
- [ ] Add to opcode reference table

### 3. Update Opcode Exports
**File**: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
- [ ] Add 15 new opcode constants
- [ ] Add to `__all__` export list
- [ ] Verify no conflicts with existing opcodes

### 4. Create Test Suite
**File**: `knowledge3d/cranium/tests/test_tier1_phase2.py` (NEW)
- [ ] Test inverse trig functions (4 tests)
- [ ] Test hyperbolic trig functions (3 tests)
- [ ] Test rounding functions (5 tests)
- [ ] Test logarithms (2 tests)
- [ ] Test bitwise logic (4 tests)
- [ ] Test error cases (domain errors, stack underflow)

### 5. Update Math Galaxy Registry
**File**: `knowledge3d/cranium/math_galaxy.py`
- [ ] Map new opcodes to math symbols
- [ ] Add example RPN programs
- [ ] Update symbol category coverage

### 6. Recompile Kernels
```bash
cd knowledge3d/cranium/kernels
./recompile_kernels.sh
```

---

## Testing Strategy

### Unit Tests (per opcode)
```python
# Test mathematical correctness
assert asin(0.5) ≈ 0.5236
assert abs(-5) == 5
assert 10 & 12 == 8

# Test edge cases
assert asin(1.0) ≈ π/2
assert abs(0) == 0
assert mod(7, 3) == 1

# Test error handling
assert asin(2.0) → domain error
assert mod(5, 0) → division by zero error
```

### Integration Tests (composite programs)
```python
# Distance formula with abs: d = |x₂ - x₁|
program = [
    LITERAL_SCALAR, x2,
    LITERAL_SCALAR, x1,
    SUB,
    ABS
]

# Inverse trig composition: sin(arcsin(0.5)) = 0.5
program = [
    LITERAL_SCALAR, 0.5,
    ASIN,
    SIN
]
```

### Performance Tests
```python
# Measure latency for each new opcode
# Target: < 1µs per operation
```

---

## Performance Expectations

| Opcode | CUDA Function | Expected Latency |
|--------|---------------|------------------|
| ASIN | `asinf()` | ~500ns |
| ACOS | `acosf()` | ~500ns |
| ATAN | `atanf()` | ~500ns |
| ATAN2 | `atan2f()` | ~600ns |
| SINH | `sinhf()` | ~400ns |
| COSH | `coshf()` | ~400ns |
| TANH | `tanhf()` | ~400ns |
| ABS | `fabsf()` | ~100ns |
| CEIL | `ceilf()` | ~200ns |
| FLOOR | `floorf()` | ~200ns |
| ROUND | `roundf()` | ~200ns |
| MOD | `fmodf()` | ~300ns |
| LOG2 | `log2f()` | ~400ns |
| LOG10 | `log10f()` | ~400ns |
| AND/OR/XOR/NOT | Bitwise ops | ~100ns |

**All operations comfortably within Tier-1 target (<1µs).**

---

## Backward Compatibility

✅ **Zero Breaking Changes**:
- All new opcodes use previously unused opcode space
- Existing opcode values unchanged
- Existing programs run identically
- Stack semantics preserved

---

## Success Criteria

### Functional
- [ ] All 15 opcodes compile without errors
- [ ] All unit tests pass (18+ tests)
- [ ] All integration tests pass
- [ ] Error handling works (domain errors, stack underflow)

### Performance
- [ ] Each opcode < 1µs latency
- [ ] No regression in existing opcode performance
- [ ] GPU memory usage unchanged

### Coverage
- [ ] 200+ new symbols mappable to RPN
- [ ] Math Galaxy audit shows 70%+ coverage
- [ ] All undergraduate trig/calculus supported

---

## Timeline

### Week 1
**Days 1-2**: Implement & test inverse trig (ASIN, ACOS, ATAN, ATAN2)
**Days 3-4**: Implement & test hyperbolic trig (SINH, COSH, TANH)
**Day 5**: Implement & test rounding (ABS, CEIL, FLOOR, ROUND, MOD)

### Week 2
**Days 1-2**: Implement & test logarithms & logic (LOG2, LOG10, AND, OR, XOR, NOT)
**Days 3-4**: Integration testing, performance validation
**Day 5**: Update documentation, begin Phase 3 planning

---

## Next Steps After Phase 2

**Phase 3** (Weeks 3-5): Add 10 Tier-2 opcodes
- Set operations (UNION, INTERSECTION, DIFFERENCE)
- Matrix operations (DET, INV, TRANSPOSE, EIGEN)
- Special functions (GAMMA, FACTORIAL, BINOMIAL)

**Phase 4** (Weeks 6-11): Add 9 Tier-3 opcodes
- Symbolic calculus (DIFF, INTEGRATE, GRADIENT)
- Limits and series (LIMIT, SERIES_SUM)

**Phase 5** (Weeks 12-14): Add 8 Tier-1 opcodes
- Complex numbers (COMPLEX_REAL, COMPLEX_IMAG, COMPLEX_CONJ)
- Statistics (MEAN, MEDIAN, VARIANCE, STDDEV)

---

**Status**: Ready to begin implementation
**Estimated Completion**: 1-2 weeks for Phase 2 (15 opcodes)
**Impact**: Unlocks 857/1,046 symbols (81.9% coverage)
