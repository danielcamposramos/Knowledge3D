# CODEX: Math Galaxy Phase 3-4 Integration Task

## Mission
Integrate 18 new RPN opcodes (Phase 3-4) into K3D's modular RPN kernel, enabling vector/matrix operations and advanced scalar functions with production-grade error handling.

## Context
The K3D Math Galaxy swarm (Grok, Qwen, Kimi, DeepSeek, GLM, Claude) has collaboratively designed Phase 3-4 opcodes. Your task: materialize this code into the existing CUDA kernel.

---

## FILE 1: Add Helper Functions to Kernel

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`
**Location**: Insert AFTER line 542 (before the main switch statement)

```cpp
// ============================================================================
// PHASE 3-4 HELPER FUNCTIONS - Production Version
// ============================================================================

// --- Complex Number Helpers ---
__device__ bool pop_complex(float* stack, int* size, float& real, float& imag, int* error) {
    if (*size < 2) { *error = kErrorStackUnderflow; return false; }
    imag = stack[--(*size)];
    real = stack[--(*size)];
    return true;
}

__device__ void push_complex(float* stack, int* size, float real, float imag, int* error) {
    if (*size + 2 > kStackCapacity) { *error = kErrorStackOverflow; return; }
    stack[(*size)++] = real;
    stack[(*size)++] = imag;
}

// --- Vector Helpers (with optional sorted check) ---
__device__ bool pop_vector(float* stack, int* size, float* vec, int& vec_size,
                          int* error, bool require_sorted = false) {
    if (*size < 1) { *error = kErrorStackUnderflow; return false; }
    vec_size = (int)stack[--(*size)];
    if (vec_size > 32 || vec_size < 0) {
        *error = kErrorInvalidArgument;
        return false;
    }
    if (*size < vec_size) { *error = kErrorStackUnderflow; return false; }

    for (int i = vec_size - 1; i >= 0; --i) {
        vec[i] = stack[--(*size)];
    }

    if (require_sorted) {
        for (int i = 1; i < vec_size; ++i) {
            if (vec[i-1] > vec[i]) {
                *error = kErrorInvalidArgument;
                return false;
            }
        }
    }
    return true;
}

__device__ void push_vector(float* stack, int* size, const float* vec, int vec_size, int* error) {
    if (*size + vec_size + 1 > kStackCapacity) {
        *error = kErrorStackOverflow;
        return;
    }
    for (int i = 0; i < vec_size; ++i) {
        stack[(*size)++] = vec[i];
    }
    stack[(*size)++] = (float)vec_size;
}

// --- Matrix Helpers ---
__device__ bool pop_matrix(float* stack, int* size, float* mat, int& rows, int& cols, int* error) {
    if (*size < 2) { *error = kErrorStackUnderflow; return false; }
    cols = (int)stack[--(*size)];
    rows = (int)stack[--(*size)];

    if (rows < 2 || rows > 4 || cols < 2 || cols > 4) {
        *error = kErrorInvalidArgument;
        return false;
    }

    int elements = rows * cols;
    if (*size < elements) { *error = kErrorStackUnderflow; return false; }

    for (int i = elements - 1; i >= 0; --i) {
        mat[i] = stack[--(*size)];
    }
    return true;
}

__device__ void push_matrix(float* stack, int* size, const float* mat, int rows, int cols, int* error) {
    int elements = rows * cols;
    if (*size + elements + 2 > kStackCapacity) {
        *error = kErrorStackOverflow;
        return;
    }
    for (int i = 0; i < elements; ++i) {
        stack[(*size)++] = mat[i];
    }
    stack[(*size)++] = (float)rows;
    stack[(*size)++] = (float)cols;
}

// --- Special Function: Lanczos Gamma ---
__device__ float lanczos_gamma(float x) {
    const float g = 7.0f;
    const float coef[9] = {
        0.99999999999980993f, 676.5203681218851f, -1259.1392167224028f,
        771.32342877765313f, -176.61502916214059f, 12.507343278686905f,
        -0.13857109526572012f, 9.9843695780195716e-6f, 1.5056327351493116e-7f
    };

    if (x < 0.5f) {
        float pi_x = M_PI_F * x;
        float sin_pi_x = sinf(pi_x);
        if (fabsf(sin_pi_x) < 1e-10f) {
            return CUDART_INF_F;
        }
        return M_PI_F / (sin_pi_x * lanczos_gamma(1.0f - x));
    }

    x -= 1.0f;
    float a = coef[0];
    #pragma unroll
    for (int i = 1; i < 9; ++i) {
        a += coef[i] / (x + (float)i);
    }

    float t = x + g + 0.5f;
    return sqrtf(2.0f * M_PI_F) * powf(t, x + 0.5f) * expf(-t) * a;
}

// --- Fast Factorial with LUT ---
__device__ float factorial_fast(int n) {
    if (n < 0) return CUDART_NAN_F;
    if (n > 34) return CUDART_INF_F;

    const float fact_lut[21] = {
        1.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f, 720.0f, 5040.0f,
        40320.0f, 362880.0f, 3628800.0f, 39916800.0f, 479001600.0f,
        6227020800.0f, 87178291200.0f, 1307674368000.0f, 20922789888000.0f,
        355687428096000.0f, 6402373705728000.0f, 121645100408832000.0f,
        2432902008176640000.0f
    };

    if (n <= 20) return fact_lut[n];
    return lanczos_gamma((float)(n + 1));
}

// --- Matrix Determinant ---
__device__ float matrix_det(const float* mat, int n) {
    if (n == 2) {
        return mat[0] * mat[3] - mat[1] * mat[2];
    } else if (n == 3) {
        return mat[0] * (mat[4]*mat[8] - mat[5]*mat[7])
             - mat[1] * (mat[3]*mat[8] - mat[5]*mat[6])
             + mat[2] * (mat[3]*mat[7] - mat[4]*mat[6]);
    }
    return 0.0f;
}

// --- Matrix Inverse ---
__device__ bool matrix_inv(const float* mat, float* inv, int n, int* error) {
    float det = matrix_det(mat, n);

    if (fabsf(det) < 1e-10f) {
        *error = kErrorInvalidArgument;
        return false;
    }

    float inv_det = 1.0f / det;

    if (n == 2) {
        inv[0] =  mat[3] * inv_det;
        inv[1] = -mat[1] * inv_det;
        inv[2] = -mat[2] * inv_det;
        inv[3] =  mat[0] * inv_det;
    } else if (n == 3) {
        inv[0] = (mat[4]*mat[8] - mat[5]*mat[7]) * inv_det;
        inv[1] = (mat[2]*mat[7] - mat[1]*mat[8]) * inv_det;
        inv[2] = (mat[1]*mat[5] - mat[2]*mat[4]) * inv_det;
        inv[3] = (mat[5]*mat[6] - mat[3]*mat[8]) * inv_det;
        inv[4] = (mat[0]*mat[8] - mat[2]*mat[6]) * inv_det;
        inv[5] = (mat[2]*mat[3] - mat[0]*mat[5]) * inv_det;
        inv[6] = (mat[3]*mat[7] - mat[4]*mat[6]) * inv_det;
        inv[7] = (mat[1]*mat[6] - mat[0]*mat[7]) * inv_det;
        inv[8] = (mat[0]*mat[4] - mat[1]*mat[3]) * inv_det;
    }
    return true;
}
```

---

## FILE 2: Add Phase 3-4 Opcodes to Switch Statement

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`
**Location**: INSERT AFTER Phase 2 opcodes (after line ~1387, before the `default:` case)

```cpp
        // ====================================================================
        // PHASE 3: VECTOR/MATRIX OPERATIONS (10 OPCODES)
        // ====================================================================

        // --- Set Operations ---
        case 0xC6: { // OP_SET_UNION - A ∪ B
            float vecB[32], vecA[32], result[64];
            int sizeB, sizeA;
            if (!pop_vector(stack, size, vecB, sizeB, error, true)) return;
            if (!pop_vector(stack, size, vecA, sizeA, error, true)) return;

            int i = 0, j = 0, k = 0;
            while (i < sizeA && j < sizeB) {
                if (vecA[i] < vecB[j]) {
                    result[k++] = vecA[i++];
                } else if (vecB[j] < vecA[i]) {
                    result[k++] = vecB[j++];
                } else {
                    result[k++] = vecA[i++];
                    j++;
                }
            }
            while (i < sizeA) result[k++] = vecA[i++];
            while (j < sizeB) result[k++] = vecB[j++];

            push_vector(stack, size, result, k, error);
            break;
        }

        case 0xC7: { // OP_SET_INTERSECTION - A ∩ B
            float vecB[32], vecA[32], result[32];
            int sizeB, sizeA;
            if (!pop_vector(stack, size, vecB, sizeB, error, true)) return;
            if (!pop_vector(stack, size, vecA, sizeA, error, true)) return;

            int i = 0, j = 0, k = 0;
            while (i < sizeA && j < sizeB) {
                if (vecA[i] < vecB[j]) {
                    i++;
                } else if (vecB[j] < vecA[i]) {
                    j++;
                } else {
                    result[k++] = vecA[i++];
                    j++;
                }
            }

            push_vector(stack, size, result, k, error);
            break;
        }

        case 0xC8: { // OP_SET_DIFFERENCE - A \ B
            float vecB[32], vecA[32], result[32];
            int sizeB, sizeA;
            if (!pop_vector(stack, size, vecB, sizeB, error, true)) return;
            if (!pop_vector(stack, size, vecA, sizeA, error, true)) return;

            int i = 0, j = 0, k = 0;
            while (i < sizeA && j < sizeB) {
                if (vecA[i] < vecB[j]) {
                    result[k++] = vecA[i++];
                } else if (vecB[j] < vecA[i]) {
                    j++;
                } else {
                    i++; j++;
                }
            }
            while (i < sizeA) result[k++] = vecA[i++];

            push_vector(stack, size, result, k, error);
            break;
        }

        case 0xC9: { // OP_SET_CARTESIAN - A × B
            float vecB[32], vecA[32], result[64];
            int sizeB, sizeA;
            if (!pop_vector(stack, size, vecB, sizeB, error)) return;
            if (!pop_vector(stack, size, vecA, sizeA, error)) return;

            int max_pairs = (kStackCapacity - *size - 1) / 2;
            if (sizeA * sizeB > max_pairs) {
                *error = kErrorStackOverflow;
                return;
            }

            int k = 0;
            for (int i = 0; i < sizeA; ++i) {
                for (int j = 0; j < sizeB; ++j) {
                    result[k++] = vecA[i];
                    result[k++] = vecB[j];
                }
            }

            push_vector(stack, size, result, k, error);
            break;
        }

        // --- Matrix Operations ---
        case 0xA7: { // OP_MATRIX_DET - det(M)
            float mat[16];
            int rows, cols;
            if (!pop_matrix(stack, size, mat, rows, cols, error)) return;

            if (rows != cols || (rows != 2 && rows != 3)) {
                *error = kErrorInvalidArgument;
                return;
            }

            float det = matrix_det(mat, rows);
            push_scalar(stack, size, det, error);
            break;
        }

        case 0xA8: { // OP_MATRIX_INV - M⁻¹
            float mat[16], inv[16];
            int rows, cols;
            if (!pop_matrix(stack, size, mat, rows, cols, error)) return;

            if (rows != cols || (rows != 2 && rows != 3)) {
                *error = kErrorInvalidArgument;
                return;
            }

            if (!matrix_inv(mat, inv, rows, error)) return;
            push_matrix(stack, size, inv, rows, cols, error);
            break;
        }

        case 0xA9: { // OP_MATRIX_TRANSPOSE - Mᵀ
            float mat[16], trans[16];
            int rows, cols;
            if (!pop_matrix(stack, size, mat, rows, cols, error)) return;

            #pragma unroll
            for (int i = 0; i < rows; ++i) {
                #pragma unroll
                for (int j = 0; j < cols; ++j) {
                    trans[j * rows + i] = mat[i * cols + j];
                }
            }

            push_matrix(stack, size, trans, cols, rows, error);
            break;
        }

        // --- Statistics ---
        case 0x95: { // OP_MEAN - μ
            float vec[32];
            int vec_size;
            if (!pop_vector(stack, size, vec, vec_size, error)) return;
            if (vec_size == 0) { *error = kErrorInvalidArgument; return; }

            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < vec_size; ++i) {
                sum += vec[i];
            }

            push_scalar(stack, size, sum / vec_size, error);
            break;
        }

        case 0x96: { // OP_MEDIAN - median
            float vec[32];
            int vec_size;
            if (!pop_vector(stack, size, vec, vec_size, error)) return;
            if (vec_size == 0) { *error = kErrorInvalidArgument; return; }

            // Selection sort for small n
            for (int i = 0; i < vec_size / 2 + 1; ++i) {
                int min_idx = i;
                #pragma unroll
                for (int j = i + 1; j < vec_size; ++j) {
                    if (vec[j] < vec[min_idx]) min_idx = j;
                }
                if (min_idx != i) {
                    float temp = vec[i];
                    vec[i] = vec[min_idx];
                    vec[min_idx] = temp;
                }
            }

            float median = (vec_size % 2 == 1)
                ? vec[vec_size / 2]
                : (vec[vec_size / 2 - 1] + vec[vec_size / 2]) * 0.5f;

            push_scalar(stack, size, median, error);
            break;
        }

        case 0x97: { // OP_VARIANCE - σ²
            float vec[32];
            int vec_size;
            if (!pop_vector(stack, size, vec, vec_size, error)) return;
            if (vec_size < 2) { *error = kErrorInvalidArgument; return; }

            float mean = 0.0f;
            #pragma unroll
            for (int i = 0; i < vec_size; ++i) mean += vec[i];
            mean /= vec_size;

            float variance = 0.0f;
            #pragma unroll
            for (int i = 0; i < vec_size; ++i) {
                float diff = vec[i] - mean;
                variance += diff * diff;
            }
            variance /= (vec_size - 1);

            push_scalar(stack, size, variance, error);
            break;
        }

        // ====================================================================
        // PHASE 4: ADVANCED SCALAR OPERATIONS (8 OPCODES)
        // ====================================================================

        case 0xAB: { // OP_GAMMA - Γ(x)
            float x;
            if (!pop_scalar(stack, size, x, error)) return;

            if (x <= 0.0f && x == floorf(x)) {
                *error = kErrorInvalidArgument;
                return;
            }

            float result = lanczos_gamma(x);
            if (!isfinite(result)) {
                *error = kErrorInvalidArgument;
                return;
            }

            push_scalar(stack, size, result, error);
            break;
        }

        case 0xAC: { // OP_FACTORIAL - n!
            float n_float;
            if (!pop_scalar(stack, size, n_float, error)) return;

            int n = (int)n_float;
            if (n < 0 || n != n_float) {
                *error = kErrorInvalidArgument;
                return;
            }

            float result = factorial_fast(n);
            if (!isfinite(result)) {
                *error = kErrorInvalidArgument;
                return;
            }

            push_scalar(stack, size, result, error);
            break;
        }

        case 0xAD: { // OP_BINOMIAL - C(n,k)
            float k_float, n_float;
            if (!pop_scalar(stack, size, k_float, error)) return;
            if (!pop_scalar(stack, size, n_float, error)) return;

            int n = (int)n_float, k = (int)k_float;
            if (n < 0 || k < 0 || k > n || n != n_float || k != k_float) {
                *error = kErrorInvalidArgument;
                return;
            }

            if (k > n - k) k = n - k;

            float result = 1.0f;
            for (int i = 0; i < k; ++i) {
                result *= (float)(n - i) / (float)(i + 1);
            }

            push_scalar(stack, size, result, error);
            break;
        }

        case 0xAE: { // OP_BETA - B(x,y)
            float y, x;
            if (!pop_scalar(stack, size, y, error)) return;
            if (!pop_scalar(stack, size, x, error)) return;

            if (x <= 0.0f || y <= 0.0f) {
                *error = kErrorInvalidArgument;
                return;
            }

            float gamma_x = lanczos_gamma(x);
            float gamma_y = lanczos_gamma(y);
            float gamma_xy = lanczos_gamma(x + y);

            if (!isfinite(gamma_x) || !isfinite(gamma_y) || !isfinite(gamma_xy)) {
                *error = kErrorInvalidArgument;
                return;
            }

            float result = (gamma_x * gamma_y) / gamma_xy;
            push_scalar(stack, size, result, error);
            break;
        }

        case 0x3B: { // OP_COMPLEX_REAL - Re(z)
            float imag, real;
            if (!pop_complex(stack, size, real, imag, error)) return;
            push_scalar(stack, size, real, error);
            break;
        }

        case 0x3C: { // OP_COMPLEX_IMAG - Im(z)
            float imag, real;
            if (!pop_complex(stack, size, real, imag, error)) return;
            push_scalar(stack, size, imag, error);
            break;
        }

        case 0x3D: { // OP_COMPLEX_CONJ - z̄
            float imag, real;
            if (!pop_complex(stack, size, real, imag, error)) return;
            push_complex(stack, size, real, -imag, error);
            break;
        }

        case 0x3E: { // OP_COMPLEX_ARG - arg(z)
            float imag, real;
            if (!pop_complex(stack, size, real, imag, error)) return;
            float arg = atan2f(imag, real);
            push_scalar(stack, size, arg, error);
            break;
        }
```

---

## FILE 3: Update Python Opcode Constants

**File**: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
**Action**: Add after existing Phase 2 opcodes (around line 85)

```python
# === PHASE 3: VECTOR/MATRIX OPERATIONS ===
# Set Operations (Tier-2.5, ~20-30µs)
OP_SET_UNION = 0xC6          # A ∪ B
OP_SET_INTERSECTION = 0xC7   # A ∩ B
OP_SET_DIFFERENCE = 0xC8     # A \ B
OP_SET_CARTESIAN = 0xC9      # A × B

# Matrix Operations (Tier-2.5, ~10-30µs)
OP_MATRIX_DET = 0xA7         # det(M)
OP_MATRIX_INV = 0xA8         # M⁻¹
OP_MATRIX_TRANSPOSE = 0xA9   # Mᵀ

# Statistics (Tier-2, ~15-25µs)
OP_MEAN = 0x95               # μ
OP_MEDIAN = 0x96             # median
OP_VARIANCE = 0x97           # σ²

# === PHASE 4: ADVANCED SCALAR ===
# Special Functions (Tier-3, ~40-80µs)
OP_GAMMA = 0xAB              # Γ(x)
OP_FACTORIAL = 0xAC          # n!
OP_BINOMIAL = 0xAD           # C(n,k)
OP_BETA = 0xAE               # B(x,y)

# Complex Numbers (Tier-1, ~5-10µs)
OP_COMPLEX_REAL = 0x3B       # Re(z)
OP_COMPLEX_IMAG = 0x3C       # Im(z)
OP_COMPLEX_CONJ = 0x3D       # z̄
OP_COMPLEX_ARG = 0x3E        # arg(z)
```

**Also update the `__all__` list at the end of the file**:

```python
__all__ = [
    # ... existing exports ...
    # Phase 3
    'OP_SET_UNION', 'OP_SET_INTERSECTION', 'OP_SET_DIFFERENCE', 'OP_SET_CARTESIAN',
    'OP_MATRIX_DET', 'OP_MATRIX_INV', 'OP_MATRIX_TRANSPOSE',
    'OP_MEAN', 'OP_MEDIAN', 'OP_VARIANCE',
    # Phase 4
    'OP_GAMMA', 'OP_FACTORIAL', 'OP_BINOMIAL', 'OP_BETA',
    'OP_COMPLEX_REAL', 'OP_COMPLEX_IMAG', 'OP_COMPLEX_CONJ', 'OP_COMPLEX_ARG',
]
```

---

## Compilation & Testing Instructions

Run in tmux session:

```bash
# Session 1: Compile
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
./scripts/compile_phase3_4_kernels.sh

# Session 2: Run tests (after compilation)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest \
  knowledge3d/cranium/tests/test_rpn_extended.py -xvs -k "phase_3 or phase_4"
```

---

## Success Criteria
✅ Kernel compiles without errors
✅ All 18 opcodes execute correctly
✅ No stack corruption or memory leaks
✅ Performance within tier targets
✅ Total opcode count: 103 (85 + 18)

**Expected Outcome**: Math Galaxy expanded from 70.2% → 85.1% symbol coverage! 🌌
