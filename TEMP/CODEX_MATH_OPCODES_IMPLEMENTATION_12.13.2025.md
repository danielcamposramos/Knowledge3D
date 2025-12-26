# CODEX BRIEFING: Math Opcode Implementation for Benchmark Improvement

**Date:** December 13, 2025
**Priority:** HIGH - Blocking math benchmark accuracy improvements
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Executive Summary

GSM8K accuracy reached **90.92%** with word problem grammar rules. However, MATH dataset (1.93%), Omni-MATH (10.07%), and AMC-AIME (2.99%) remain low because they require **factorial, binomial, and gamma** operations which return `error 9001` (unknown opcode).

The opcodes are **defined in Python** (`rpn_opcodes.py`) but **NOT implemented in the CUDA kernel**.

---

## Task: Implement 4 Missing Opcodes in modular_rpn_kernel_extended.cu

### File Location
```
knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu
```

### Insert Location
Before line 2514 (`default:`), add cases for these opcodes.

---

## Opcode Implementations Required

### 1. OP_GAMMA (0xAB) - Gamma Function Γ(x)

**Stack:** `[x] → [Γ(x)]`

**Implementation:** Use Stirling's approximation for x > 0, or the identity Γ(n) = (n-1)! for integers.

```cuda
case 0xAB: {  // gamma - Gamma function Γ(x)
    float x = 0.0f;
    if (!pop_scalar(stack, stack_size, x, error_code)) break;
    if (x <= 0.0f) {
        error_code = kErrorInvalidArgument;
        break;
    }
    // Lanczos approximation (g=7, n=9)
    const float g = 7.0f;
    const float c[9] = {
        0.99999999999980993f,
        676.5203681218851f,
        -1259.1392167224028f,
        771.32342877765313f,
        -176.61502916214059f,
        12.507343278686905f,
        -0.13857109526572012f,
        9.9843695780195716e-6f,
        1.5056327351493116e-7f
    };
    float z = x - 1.0f;
    float sum = c[0];
    for (int i = 1; i < 9; ++i) {
        sum += c[i] / (z + (float)i);
    }
    float t = z + g + 0.5f;
    float result = sqrtf(2.0f * 3.14159265358979323846f) * powf(t, z + 0.5f) * expf(-t) * sum;
    push_scalar(stack, stack_size, result, error_code);
    break;
}
```

---

### 2. OP_FACTORIAL (0xAC) - n!

**Stack:** `[n] → [n!]`

**Implementation:** Iterative multiplication for n ≤ 12 (float precision), use gamma for larger values.

```cuda
case 0xAC: {  // factorial - n!
    float n_f = 0.0f;
    if (!pop_scalar(stack, stack_size, n_f, error_code)) break;
    int n = (int)n_f;
    if (n < 0) {
        error_code = kErrorInvalidArgument;
        break;
    }
    float result = 1.0f;
    // Direct computation for small values (accurate in float32)
    if (n <= 12) {
        for (int i = 2; i <= n; ++i) {
            result *= (float)i;
        }
    } else {
        // Use Stirling approximation for larger values
        // n! ≈ sqrt(2πn) * (n/e)^n
        float nf = (float)n;
        result = sqrtf(2.0f * 3.14159265358979323846f * nf) *
                 powf(nf / 2.718281828459045f, nf);
    }
    push_scalar(stack, stack_size, result, error_code);
    break;
}
```

---

### 3. OP_BINOMIAL (0xAD) - (n choose k)

**Stack:** `[n, k] → [C(n,k)]`

**Implementation:** Use multiplicative formula to avoid overflow.

```cuda
case 0xAD: {  // binomial - (n choose k)
    float k_f = 0.0f;
    float n_f = 0.0f;
    if (!pop_scalar(stack, stack_size, k_f, error_code)) break;
    if (!pop_scalar(stack, stack_size, n_f, error_code)) break;
    int n = (int)n_f;
    int k = (int)k_f;
    if (n < 0 || k < 0 || k > n) {
        // C(n,k) = 0 when k > n
        push_scalar(stack, stack_size, 0.0f, error_code);
        break;
    }
    // Optimize: C(n,k) = C(n, n-k)
    if (k > n - k) {
        k = n - k;
    }
    // Multiplicative formula: C(n,k) = prod_{i=0}^{k-1} (n-i)/(i+1)
    float result = 1.0f;
    for (int i = 0; i < k; ++i) {
        result = result * (float)(n - i) / (float)(i + 1);
    }
    push_scalar(stack, stack_size, result, error_code);
    break;
}
```

---

### 4. OP_BETA (0xAE) - Beta Function B(a,b)

**Stack:** `[a, b] → [B(a,b)]`

**Implementation:** B(a,b) = Γ(a)Γ(b)/Γ(a+b) using the gamma implementation.

```cuda
case 0xAE: {  // beta - B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    float b = 0.0f;
    float a = 0.0f;
    if (!pop_scalar(stack, stack_size, b, error_code)) break;
    if (!pop_scalar(stack, stack_size, a, error_code)) break;
    if (a <= 0.0f || b <= 0.0f) {
        error_code = kErrorInvalidArgument;
        break;
    }
    // Use log-gamma for numerical stability: B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    float result = expf(lgammaf(a) + lgammaf(b) - lgammaf(a + b));
    push_scalar(stack, stack_size, result, error_code);
    break;
}
```

Note: CUDA's `lgammaf()` is available in math.h for log-gamma.

---

## Testing Requirements

After implementation, run:

```bash
# Compile the kernel
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx

# Test factorial
PYTHONPATH=. python3 -c "
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
engine = ModularRPNEngine()
# Test: 5! = 120
result = engine.evaluate([5.0, 'factorial'])
print(f'5! = {result}')  # Expected: 120.0
# Test: C(10,3) = 120
result = engine.evaluate([10.0, 3.0, 'binomial'])
print(f'C(10,3) = {result}')  # Expected: 120.0
"

# Re-run math benchmarks
PYTHONPATH=. python3 scripts/train_math_benchmarks.py --dataset math --samples 100
```

---

## Success Criteria

| Test | Expected |
|------|----------|
| `[5, factorial]` | 120.0 |
| `[10, factorial]` | 3628800.0 |
| `[10, 3, binomial]` | 120.0 |
| `[20, 10, binomial]` | 184756.0 |
| `[5, gamma]` | 24.0 (Γ(5) = 4!) |
| `[0.5, gamma]` | ~1.7724 (√π) |
| `[2, 3, beta]` | ~0.0833 |

---

## Architecture Notes

- **NO numpy, NO cupy** - Pure CUDA implementation
- Stack order: operands pushed first, operator last
- Error codes: Use `kErrorInvalidArgument` (9005) for invalid inputs
- `lgammaf()` is available in CUDA's math.h - use it for numerical stability

---

## Expected Impact

With these opcodes implemented, MATH benchmark accuracy should increase from **1.93% → 15-30%** as many competition math problems require factorial and binomial operations.

---

**Codex:** Implement these 4 cases in modular_rpn_kernel_extended.cu, compile to PTX, and verify with the test script. Report back with test results.
