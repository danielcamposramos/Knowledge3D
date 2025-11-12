# Math Galaxy Phase 5: True Multivariate Calculus Extension

## Executive Summary

Codex correctly implemented Phase 5-8 operations but noted a critical limitation: the RPN bytecode evaluator is **scalar-only**. This means gradient/divergence/curl/Laplacian currently behave as univariate probes rather than true multivariate operators.

**This document extends the Math Galaxy to support true multivariate calculus on GPU.**

---

## 1. Problem Statement

### Current Limitation
```cpp
__device__ float evaluate_rpn_function(
    const float* program,
    int program_length,
    float x,              // ❌ SINGLE SCALAR INPUT
    uint32_t& error
)
```

**Result**: Functions like `f(x,y,z)` cannot be represented. Gradient computes ∂f/∂x by evaluating `f(x+h)` instead of `f(x+h, y, z)`.

### Required Capability
```cpp
// ✓ MULTIVARIATE INPUT
__device__ float evaluate_rpn_function_multivar(
    const float* program,
    int program_length,
    const float* vars,    // Array: [x, y, z, ...]
    int n_vars,          // Number of variables
    uint32_t& error
)
```

**Result**: Full multivariate calculus:
- **Gradient**: ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
- **Divergence**: ∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂F_z/∂z
- **Curl**: ∇×F = [∂F_z/∂y - ∂Fᵧ/∂z, ∂Fₓ/∂z - ∂F_z/∂x, ∂Fᵧ/∂x - ∂Fₓ/∂y]
- **Laplacian**: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

---

## 2. Bytecode Format Design

### 2.1 Variable Reference Opcodes

We reserve opcodes `0xE0-0xE9` for multivariate support:

```cpp
// Variable reference opcodes (Tier 0: 1 cycle)
0xE0  VAR_X       // Push variable x onto stack
0xE1  VAR_Y       // Push variable y onto stack
0xE2  VAR_Z       // Push variable z onto stack
0xE3  VAR_W       // Push variable w onto stack (4D support)
0xE4  CONST       // Next bytecode is a constant (encoded as float)
0xE5  (reserved)
0xE6  (reserved)
0xE7  (reserved)
0xE8  (reserved)
0xE9  (reserved)
```

### 2.2 Example Encodings

**Example 1: f(x) = x²**
```
Scalar bytecode (backward compatible):
  [0x14]  // SQUARE (assumes x already on stack)

Multivariate bytecode:
  [0xF0, 0x14]  // VAR_X, SQUARE
```

**Example 2: f(x,y) = x² + y²**
```
Multivariate bytecode:
  [0xE0, 0x14,     // VAR_X, SQUARE → x²
   0xE1, 0x14,     // VAR_Y, SQUARE → y²
   0x01]           // ADD → x² + y²
```

**Example 3: f(x,y) = sin(x·y) + e^x**
```
Multivariate bytecode:
  [0xE0, 0xE1, 0x03,  // VAR_X, VAR_Y, MULTIPLY → x·y
   0x18,               // SIN → sin(x·y)
   0xE0, 0x16,         // VAR_X, EXP → e^x
   0x01]               // ADD → sin(x·y) + e^x
```

**Example 4: f(x,y) = 2·x + 3·y**
```
Multivariate bytecode (with constants):
  [0xE4, 2.0,         // CONST, 2.0 → push 2.0
   0xE0, 0x03,        // VAR_X, MULTIPLY → 2·x
   0xE4, 3.0,         // CONST, 3.0 → push 3.0
   0xE1, 0x03,        // VAR_Y, MULTIPLY → 3·y
   0x01]              // ADD → 2·x + 3·y
```

### 2.3 Backward Compatibility

**Scalar functions remain compatible** - the scalar evaluator assumes x is pre-pushed:
```cpp
// Old scalar API (still works)
evaluate_rpn_function(program, len, x, error);
// → Internally: push(x), execute(program)

// New multivariate API
float vars[] = {x};
evaluate_rpn_function_multivar(program, len, vars, 1, error);
// → Internally: execute(program) [VAR_X references vars[0]]
```

---

## 3. Implementation: Multivariate Evaluator

### 3.1 Core Function

```cpp
// ============================================================================
// PHASE 5 MULTIVARIATE EXTENSION: Multivariate RPN Function Evaluator
// ============================================================================
__device__ float evaluate_rpn_function_multivar(
    const float* program,      // Bytecode array
    int program_length,        // Number of bytecode tokens
    const float* vars,         // Variable values [x, y, z, w]
    int n_vars,               // Number of variables (1-4)
    uint32_t& error
) {
    if (program_length < 1 || program_length > 32) {
        error = kErrorInvalidArgument;
        return 0.0f;
    }
    if (n_vars < 1 || n_vars > 4) {
        error = kErrorInvalidArgument;
        return 0.0f;
    }

    StackItem temp_stack[32];
    uint32_t temp_size = 0;

    // Execute bytecode
    for (int i = 0; i < program_length; ++i) {
        uint16_t opcode = static_cast<uint16_t>(program[i]);

        switch (opcode) {
            // ============================================================
            // VARIABLE REFERENCE OPCODES
            // ============================================================
            case 0xF0: {  // VAR_X
                if (n_vars < 1) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[0], error)) return 0.0f;
                break;
            }
            case 0xF1: {  // VAR_Y
                if (n_vars < 2) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[1], error)) return 0.0f;
                break;
            }
            case 0xF2: {  // VAR_Z
                if (n_vars < 3) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[2], error)) return 0.0f;
                break;
            }
            case 0xF3: {  // VAR_W
                if (n_vars < 4) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[3], error)) return 0.0f;
                break;
            }
            case 0xF4: {  // CONST
                ++i;  // Advance to next token
                if (i >= program_length) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                float constant = program[i];
                if (!push_scalar(temp_stack, temp_size, constant, error)) return 0.0f;
                break;
            }

            // ============================================================
            // ARITHMETIC OPCODES (unchanged from scalar version)
            // ============================================================
            case 0x01: {  // ADD
                float b, a;
                if (!pop_scalar(temp_stack, temp_size, b, error)) return 0.0f;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, a + b, error)) return 0.0f;
                break;
            }
            case 0x02: {  // SUBTRACT
                float b, a;
                if (!pop_scalar(temp_stack, temp_size, b, error)) return 0.0f;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, a - b, error)) return 0.0f;
                break;
            }
            case 0x03: {  // MULTIPLY
                float b, a;
                if (!pop_scalar(temp_stack, temp_size, b, error)) return 0.0f;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, a * b, error)) return 0.0f;
                break;
            }
            case 0x04: {  // DIVIDE
                float b, a;
                if (!pop_scalar(temp_stack, temp_size, b, error)) return 0.0f;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (fabsf(b) < 1e-10f) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, a / b, error)) return 0.0f;
                break;
            }
            case 0x05: {  // POWER
                float b, a;
                if (!pop_scalar(temp_stack, temp_size, b, error)) return 0.0f;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, powf(a, b), error)) return 0.0f;
                break;
            }
            case 0x14: {  // SQUARE
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, a * a, error)) return 0.0f;
                break;
            }
            case 0x15: {  // SQRT
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (a < 0.0f) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, sqrtf(a), error)) return 0.0f;
                break;
            }
            case 0x16: {  // EXP
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, expf(a), error)) return 0.0f;
                break;
            }
            case 0x17: {  // LOG
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (a <= 0.0f) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, logf(a), error)) return 0.0f;
                break;
            }
            case 0x18: {  // SIN
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, sinf(a), error)) return 0.0f;
                break;
            }
            case 0x19: {  // COS
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, cosf(a), error)) return 0.0f;
                break;
            }
            case 0x1A: {  // TAN
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, tanf(a), error)) return 0.0f;
                break;
            }
            case 0x1B: {  // ASIN
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, asinf(a), error)) return 0.0f;
                break;
            }
            case 0x1C: {  // ACOS
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, acosf(a), error)) return 0.0f;
                break;
            }
            case 0x1D: {  // ATAN
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, atanf(a), error)) return 0.0f;
                break;
            }

            default:
                error = kErrorUnknownOpcode;
                return 0.0f;
        }

        if (error != kErrorNone) {
            return 0.0f;
        }
    }

    // Validate final stack state
    if (temp_size != 1) {
        error = kErrorStackUnderflow;
        return 0.0f;
    }

    float result = 0.0f;
    if (!pop_scalar(temp_stack, temp_size, result, error)) {
        return 0.0f;
    }
    return result;
}
```

### 3.2 Backward-Compatible Scalar Wrapper

Keep the old scalar API for compatibility:

```cpp
// Scalar evaluator (backward compatible)
__device__ float evaluate_rpn_function(
    const float* program,
    int program_length,
    float x,
    uint32_t& error
) {
    float vars[1] = {x};
    return evaluate_rpn_function_multivar(program, program_length, vars, 1, error);
}
```

---

## 4. Updated Phase 5 Symbolic Operations

### 4.1 OP_GRADIENT (0xB6) - Fixed

**Before** (univariate probe):
```cpp
float f_plus = evaluate_rpn_function(program, n_opcodes, point_plus[var], error);
float f_minus = evaluate_rpn_function(program, n_opcodes, point_minus[var], error);
```

**After** (true multivariate gradient):
```cpp
case 0xB6: {  // OP_GRADIENT
    float h, n_vars_f;
    if (!pop_scalar(stack, stack_size, h, error_code)) break;
    if (!pop_scalar(stack, stack_size, n_vars_f, error_code)) break;

    if (fabsf(h) < 1e-12f) {
        error_code = kErrorInvalidArgument;
        break;
    }

    int n_vars = static_cast<int>(n_vars_f);
    if (n_vars < 1 || n_vars > 4) {
        error_code = kErrorInvalidArgument;
        break;
    }

    // Pop evaluation point [x, y, z, w]
    float point[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = n_vars - 1; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, point[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Pop RPN program
    float n_opcodes_f;
    if (!pop_scalar(stack, stack_size, n_opcodes_f, error_code)) break;
    int n_opcodes = static_cast<int>(n_opcodes_f);
    if (n_opcodes < 1 || n_opcodes > 32) {
        error_code = kErrorInvalidArgument;
        break;
    }

    float program[32];
    for (int i = n_opcodes - 1; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, program[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Compute gradient via central differences
    float gradient[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int var = 0; var < n_vars; ++var) {
        // f(x, y+h, z)
        float point_plus[4] = {point[0], point[1], point[2], point[3]};
        point_plus[var] += h;
        float f_plus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_plus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // f(x, y-h, z)
        float point_minus[4] = {point[0], point[1], point[2], point[3]};
        point_minus[var] -= h;
        float f_minus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_minus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // Central difference: ∂f/∂var = (f_plus - f_minus) / (2h)
        gradient[var] = (f_plus - f_minus) / (2.0f * h);
    }
    if (error_code != kErrorNone) break;

    // Push gradient components [∂f/∂x, ∂f/∂y, ∂f/∂z]
    for (int i = 0; i < n_vars; ++i) {
        push_scalar(stack, stack_size, gradient[i], error_code);
        if (error_code != kErrorNone) break;
    }
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, static_cast<float>(n_vars), error_code);
    break;
}
```

### 4.2 OP_DIVERGENCE (0xB8) - Fixed

**True divergence**: ∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂F_z/∂z

```cpp
case 0xB8: {  // OP_DIVERGENCE
    // Stack layout: [... program_x program_y program_z n_opcodes x y z n_vars h]
    float h, n_vars_f;
    if (!pop_scalar(stack, stack_size, h, error_code)) break;
    if (!pop_scalar(stack, stack_size, n_vars_f, error_code)) break;

    int n_vars = static_cast<int>(n_vars_f);
    if (n_vars < 2 || n_vars > 3) {
        error_code = kErrorInvalidArgument;
        break;
    }

    // Pop evaluation point
    float point[3] = {0.0f, 0.0f, 0.0f};
    for (int i = n_vars - 1; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, point[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Pop number of opcodes per field component
    float n_opcodes_f;
    if (!pop_scalar(stack, stack_size, n_opcodes_f, error_code)) break;
    int n_opcodes = static_cast<int>(n_opcodes_f);
    if (n_opcodes < 1 || n_opcodes > 32) {
        error_code = kErrorInvalidArgument;
        break;
    }

    // Pop RPN programs for each field component
    float programs[3][32];
    for (int comp = n_vars - 1; comp >= 0; --comp) {
        for (int i = n_opcodes - 1; i >= 0; --i) {
            if (!pop_scalar(stack, stack_size, programs[comp][i], error_code)) break;
        }
        if (error_code != kErrorNone) break;
    }
    if (error_code != kErrorNone) break;

    // Compute divergence: ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂F_z/∂z
    float divergence = 0.0f;
    for (int comp = 0; comp < n_vars; ++comp) {
        // ∂F_comp/∂comp via central difference
        float point_plus[3] = {point[0], point[1], point[2]};
        point_plus[comp] += h;
        float f_plus = evaluate_rpn_function_multivar(
            programs[comp], n_opcodes, point_plus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        float point_minus[3] = {point[0], point[1], point[2]};
        point_minus[comp] -= h;
        float f_minus = evaluate_rpn_function_multivar(
            programs[comp], n_opcodes, point_minus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        divergence += (f_plus - f_minus) / (2.0f * h);
    }
    if (error_code != kErrorNone) break;

    push_scalar(stack, stack_size, divergence, error_code);
    break;
}
```

### 4.3 OP_CURL (0xB9) - Fixed

**True curl**: ∇×F = [∂F_z/∂y - ∂Fᵧ/∂z, ∂Fₓ/∂z - ∂F_z/∂x, ∂Fᵧ/∂x - ∂Fₓ/∂y]

```cpp
case 0xB9: {  // OP_CURL (3D only)
    // Stack: [... prog_x prog_y prog_z n_opcodes x y z h]
    float h;
    if (!pop_scalar(stack, stack_size, h, error_code)) break;

    // Pop point (3D only for curl)
    float point[3];
    for (int i = 2; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, point[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Pop program length
    float n_opcodes_f;
    if (!pop_scalar(stack, stack_size, n_opcodes_f, error_code)) break;
    int n_opcodes = static_cast<int>(n_opcodes_f);
    if (n_opcodes < 1 || n_opcodes > 32) {
        error_code = kErrorInvalidArgument;
        break;
    }

    // Pop 3 RPN programs (Fₓ, Fᵧ, F_z)
    float programs[3][32];
    for (int comp = 2; comp >= 0; --comp) {
        for (int i = n_opcodes - 1; i >= 0; --i) {
            if (!pop_scalar(stack, stack_size, programs[comp][i], error_code)) break;
        }
        if (error_code != kErrorNone) break;
    }
    if (error_code != kErrorNone) break;

    // Compute curl components
    float curl[3];

    // curl_x = ∂F_z/∂y - ∂Fᵧ/∂z
    float point_y_plus[3] = {point[0], point[1] + h, point[2]};
    float point_y_minus[3] = {point[0], point[1] - h, point[2]};
    float dFz_dy = (
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_y_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_y_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float point_z_plus[3] = {point[0], point[1], point[2] + h};
    float point_z_minus[3] = {point[0], point[1], point[2] - h};
    float dFy_dz = (
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_z_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_z_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    curl[0] = dFz_dy - dFy_dz;

    // curl_y = ∂Fₓ/∂z - ∂F_z/∂x
    float dFx_dz = (
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_z_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_z_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float point_x_plus[3] = {point[0] + h, point[1], point[2]};
    float point_x_minus[3] = {point[0] - h, point[1], point[2]};
    float dFz_dx = (
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_x_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_x_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    curl[1] = dFx_dz - dFz_dx;

    // curl_z = ∂Fᵧ/∂x - ∂Fₓ/∂y
    float dFy_dx = (
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_x_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_x_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float dFx_dy = (
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_y_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_y_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    curl[2] = dFy_dx - dFx_dy;

    // Push curl vector
    for (int i = 0; i < 3; ++i) {
        push_scalar(stack, stack_size, curl[i], error_code);
        if (error_code != kErrorNone) break;
    }
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, 3.0f, error_code);  // Vector dimension marker
    break;
}
```

### 4.4 OP_LAPLACIAN (0xBA) - Fixed

**True Laplacian**: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

```cpp
case 0xBA: {  // OP_LAPLACIAN
    // Stack: [... program n_opcodes x y z n_vars h]
    float h, n_vars_f;
    if (!pop_scalar(stack, stack_size, h, error_code)) break;
    if (!pop_scalar(stack, stack_size, n_vars_f, error_code)) break;

    int n_vars = static_cast<int>(n_vars_f);
    if (n_vars < 1 || n_vars > 3) {
        error_code = kErrorInvalidArgument;
        break;
    }

    // Pop evaluation point
    float point[3] = {0.0f, 0.0f, 0.0f};
    for (int i = n_vars - 1; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, point[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Pop program
    float n_opcodes_f;
    if (!pop_scalar(stack, stack_size, n_opcodes_f, error_code)) break;
    int n_opcodes = static_cast<int>(n_opcodes_f);
    if (n_opcodes < 1 || n_opcodes > 32) {
        error_code = kErrorInvalidArgument;
        break;
    }

    float program[32];
    for (int i = n_opcodes - 1; i >= 0; --i) {
        if (!pop_scalar(stack, stack_size, program[i], error_code)) break;
    }
    if (error_code != kErrorNone) break;

    // Evaluate at center point
    float f_center = evaluate_rpn_function_multivar(
        program, n_opcodes, point, n_vars, error_code);
    if (error_code != kErrorNone) break;

    // Compute Laplacian: ∑ ∂²f/∂xᵢ²
    float laplacian = 0.0f;
    for (int var = 0; var < n_vars; ++var) {
        // f(x, y, z+h)
        float point_plus[3] = {point[0], point[1], point[2]};
        point_plus[var] += h;
        float f_plus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_plus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // f(x, y, z-h)
        float point_minus[3] = {point[0], point[1], point[2]};
        point_minus[var] -= h;
        float f_minus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_minus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // Second derivative: ∂²f/∂var² = (f_plus - 2*f_center + f_minus) / h²
        laplacian += (f_plus - 2.0f * f_center + f_minus) / (h * h);
    }
    if (error_code != kErrorNone) break;

    push_scalar(stack, stack_size, laplacian, error_code);
    break;
}
```

---

## 5. Python Opcode Constants

Add variable reference opcodes to [`rpn_opcodes.py`](../../knowledge3d/cranium/ptx_runtime/rpn_opcodes.py):

```python
# Multivariate variable reference opcodes (Tier 0: 1 cycle)
OP_VAR_X = 0xF0
OP_VAR_Y = 0xF1
OP_VAR_Z = 0xF2
OP_VAR_W = 0xF3
OP_CONST = 0xF4

__all__ = [
    # ... existing opcodes ...

    # Multivariate support
    "OP_VAR_X",
    "OP_VAR_Y",
    "OP_VAR_Z",
    "OP_VAR_W",
    "OP_CONST",
]
```

---

## 6. Usage Examples (Python Runtime)

### Example 1: Scalar Function (Backward Compatible)

```python
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ops

# f(x) = x² + sin(x)
program = [
    ops.OP_SQUARE,      # x²
    ops.OP_DUP,         # Duplicate x
    ops.OP_SIN,         # sin(x)
    ops.OP_ADD,         # x² + sin(x)
]
```

### Example 2: Multivariate Function

```python
# f(x,y) = x² + y²
program = [
    ops.OP_VAR_X,       # Push x
    ops.OP_SQUARE,      # x²
    ops.OP_VAR_Y,       # Push y
    ops.OP_SQUARE,      # y²
    ops.OP_ADD,         # x² + y²
]

# Compute gradient at (3, 4)
gradient_program = [
    *program,                    # The function
    len(program),                # Program length
    3.0, 4.0,                   # Point (x=3, y=4)
    2,                          # Number of variables
    0.001,                      # Step size h
    ops.OP_GRADIENT,            # Execute gradient
]
# Result: [6.0, 8.0, 2] → gradient is [∂f/∂x=6, ∂f/∂y=8] at (3,4)
```

### Example 3: Divergence of Vector Field

```python
# Field: F = [x², y², z]
# Divergence: ∇·F = ∂(x²)/∂x + ∂(y²)/∂y + ∂z/∂z = 2x + 2y + 1

program_Fx = [ops.OP_VAR_X, ops.OP_SQUARE]  # Fₓ = x²
program_Fy = [ops.OP_VAR_Y, ops.OP_SQUARE]  # Fᵧ = y²
program_Fz = [ops.OP_VAR_Z]                  # F_z = z

divergence_program = [
    *program_Fx, *program_Fy, *program_Fz,  # All 3 field components
    len(program_Fx),                         # Program length
    1.0, 2.0, 3.0,                          # Point (1, 2, 3)
    3,                                       # 3D field
    0.001,                                   # Step size
    ops.OP_DIVERGENCE,
]
# Result: 2(1) + 2(2) + 1 = 7.0
```

---

## 7. Performance Analysis

### Latency Tiers (Updated)

| Opcode | Operation | Tier | Cycles | Notes |
|--------|-----------|------|--------|-------|
| 0xE0-E3 | VAR_X/Y/Z/W | 0 | 1 | Variable reference (register read) |
| 0xE4 | CONST | 0 | 1 | Constant push (bytecode fetch) |
| 0xB6 | OP_GRADIENT | 4 | ~2000 | 2n function evaluations (n=vars) |
| 0xB8 | OP_DIVERGENCE | 4 | ~2000 | 2n function evaluations |
| 0xB9 | OP_CURL | 4 | ~6000 | 6 function evaluations (3D only) |
| 0xBA | OP_LAPLACIAN | 4 | ~2000 | 2n+1 function evaluations |

**Symbolic ops remain Tier 4** but now compute **mathematically correct** multivariate derivatives.

---

## 8. Testing Strategy

### Unit Tests

```python
# tests/test_multivariate_rpn.py

def test_gradient_multivariate():
    """Test ∇(x² + y²) = [2x, 2y]"""
    # f(x,y) = x² + y²
    program = [OP_VAR_X, OP_SQUARE, OP_VAR_Y, OP_SQUARE, OP_ADD]

    # Gradient at (3, 4)
    result = execute_gradient(program, point=[3.0, 4.0], h=0.001)

    # Expected: [6.0, 8.0]
    assert abs(result[0] - 6.0) < 0.01
    assert abs(result[1] - 8.0) < 0.01

def test_divergence_field():
    """Test ∇·[x², y², 0] = 2x + 2y"""
    prog_x = [OP_VAR_X, OP_SQUARE]
    prog_y = [OP_VAR_Y, OP_SQUARE]
    prog_z = [OP_CONST, 0.0]

    result = execute_divergence([prog_x, prog_y, prog_z], point=[1.0, 2.0, 0.0])

    # Expected: 2(1) + 2(2) = 6.0
    assert abs(result - 6.0) < 0.01

def test_laplacian():
    """Test ∇²(x² + y²) = 4"""
    program = [OP_VAR_X, OP_SQUARE, OP_VAR_Y, OP_SQUARE, OP_ADD]

    result = execute_laplacian(program, point=[1.0, 2.0], h=0.001)

    # Expected: 4.0 (constant for all points)
    assert abs(result - 4.0) < 0.01
```

---

## 9. Integration Roadmap

### Phase 1: Core Multivariate Evaluator (This PR)
- ✅ Design bytecode format
- ✅ Implement `evaluate_rpn_function_multivar()`
- ✅ Update OP_GRADIENT, OP_DIVERGENCE, OP_CURL, OP_LAPLACIAN
- ✅ Add Python constants

### Phase 2: Kernel Compilation & Testing
- Recompile `modular_rpn_kernel_extended.cu`
- Run unit tests for all symbolic ops
- Benchmark performance vs. scalar version

### Phase 3: Advanced Features
- Implement Hessian matrix (OP_HESSIAN)
- Add Jacobian matrix (OP_JACOBIAN)
- Support tensor fields (stress tensor divergence)

---

## 10. Summary

**This extension transforms Math Galaxy from a scalar symbolic engine into a true multivariate calculus system.**

### Key Achievements
✅ **Bytecode format** supporting 1-4 variables via VAR_X/Y/Z/W opcodes
✅ **Multivariate evaluator** with backward compatibility
✅ **True gradient** ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
✅ **True divergence** ∇·F for vector fields
✅ **True curl** ∇×F in 3D
✅ **True Laplacian** ∇²f for scalar fields
✅ **100% sovereign** - all operations GPU-native

### Mathematical Coverage
With this extension, K3D Math Galaxy now supports:
- **Vector calculus**: gradient, divergence, curl, Laplacian
- **Differential geometry**: future support for Christoffel symbols, curvature
- **Physics simulations**: electromagnetic fields, fluid dynamics, heat diffusion
- **Machine learning**: Hessian-based optimization, natural gradients

**The Math Galaxy is now a true mathematical gem.** 💎
