// ============================================================================
// MATH GALAXY PHASE 5 MULTIVARIATE EXTENSION
// True multivariate calculus for gradient, divergence, curl, Laplacian
// ============================================================================
//
// This file contains the complete implementation of multivariate RPN
// evaluation and updated symbolic operators for the Math Galaxy.
//
// INSERT LOCATION: After xorshift128+ PRNG (line ~560 in modular_rpn_kernel_extended.cu)
// REPLACE: Lines 376-527 (scalar evaluator) with multivariate version
// UPDATE: Lines 1823-2295 (Phase 5 symbolic ops) with multivariate versions
// ============================================================================

namespace {

// ============================================================================
// MULTIVARIATE RPN FUNCTION EVALUATOR
// ============================================================================
// Evaluates RPN bytecode programs with support for 1-4 variables
//
// Bytecode format:
//   0xE0       VAR_X    - Push variable x onto stack
//   0xE1       VAR_Y    - Push variable y onto stack
//   0xE2       VAR_Z    - Push variable z onto stack
//   0xE3       VAR_W    - Push variable w onto stack
//   0xE4       CONST    - Next token is a constant to push
//   0x01-0x1D  ...      - Standard arithmetic/trig opcodes
//
// Example: f(x,y) = x² + y²
//   Bytecode: [0xE0, 0x14, 0xE1, 0x14, 0x01]
//   Meaning:  [VAR_X, SQUARE, VAR_Y, SQUARE, ADD]
//
__device__ float evaluate_rpn_function_multivar(
    const float* program,      // Bytecode array
    int program_length,        // Number of tokens
    const float* vars,         // Variable array [x, y, z, w]
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

    // Execute bytecode program
    for (int i = 0; i < program_length; ++i) {
        uint16_t opcode = static_cast<uint16_t>(program[i]);

        switch (opcode) {
            // ================================================================
            // VARIABLE REFERENCE OPCODES (Tier 0: 1 cycle)
            // ================================================================
            case 0xE0: {  // VAR_X
                if (n_vars < 1) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[0], error)) return 0.0f;
                break;
            }
            case 0xE1: {  // VAR_Y
                if (n_vars < 2) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[1], error)) return 0.0f;
                break;
            }
            case 0xE2: {  // VAR_Z
                if (n_vars < 3) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[2], error)) return 0.0f;
                break;
            }
            case 0xE3: {  // VAR_W
                if (n_vars < 4) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                if (!push_scalar(temp_stack, temp_size, vars[3], error)) return 0.0f;
                break;
            }
            case 0xE4: {  // CONST
                ++i;  // Advance to next token
                if (i >= program_length) {
                    error = kErrorInvalidArgument;
                    return 0.0f;
                }
                float constant = program[i];
                if (!push_scalar(temp_stack, temp_size, constant, error)) return 0.0f;
                break;
            }

            // ================================================================
            // ARITHMETIC OPCODES (Tier 0-1: 1-4 cycles)
            // ================================================================
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
            case 0x06: {  // NEGATE
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, -a, error)) return 0.0f;
                break;
            }
            case 0x07: {  // ABS
                float a;
                if (!pop_scalar(temp_stack, temp_size, a, error)) return 0.0f;
                if (!push_scalar(temp_stack, temp_size, fabsf(a), error)) return 0.0f;
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

// ============================================================================
// BACKWARD-COMPATIBLE SCALAR WRAPPER
// ============================================================================
// Maintains compatibility with existing scalar-only code
//
__device__ float evaluate_rpn_function(
    const float* program,
    int program_length,
    float x,
    uint32_t& error
) {
    float vars[1] = {x};
    return evaluate_rpn_function_multivar(program, program_length, vars, 1, error);
}

}  // namespace


// ============================================================================
// UPDATED PHASE 5 SYMBOLIC OPERATIONS (MULTIVARIATE)
// ============================================================================
// These replace the existing implementations in the main switch statement
// (lines ~1823-2295 in modular_rpn_kernel_extended.cu)
// ============================================================================

// ----------------------------------------------------------------------------
// OP_SYMBOLIC_DIFF (0xB5) - Unchanged (already uses scalar evaluator correctly)
// ----------------------------------------------------------------------------
// Uses central differences: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
// This one is fine as-is since it's truly univariate

// ----------------------------------------------------------------------------
// OP_GRADIENT (0xB6) - TRUE MULTIVARIATE VERSION
// ----------------------------------------------------------------------------
// Computes ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z, ...]
//
// Stack input:  [... program n_opcodes x y z n_vars h]
// Stack output: [... ∂f/∂x ∂f/∂y ∂f/∂z n_vars]
//
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
    // ∂f/∂xᵢ ≈ (f(..., xᵢ+h, ...) - f(..., xᵢ-h, ...)) / (2h)
    float gradient[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int var = 0; var < n_vars; ++var) {
        // Perturb variable in positive direction
        float point_plus[4] = {point[0], point[1], point[2], point[3]};
        point_plus[var] += h;
        float f_plus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_plus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // Perturb variable in negative direction
        float point_minus[4] = {point[0], point[1], point[2], point[3]};
        point_minus[var] -= h;
        float f_minus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_minus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        // Central difference
        gradient[var] = (f_plus - f_minus) / (2.0f * h);
    }
    if (error_code != kErrorNone) break;

    // Push gradient components
    for (int i = 0; i < n_vars; ++i) {
        push_scalar(stack, stack_size, gradient[i], error_code);
        if (error_code != kErrorNone) break;
    }
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, static_cast<float>(n_vars), error_code);
    break;
}

// ----------------------------------------------------------------------------
// OP_SYMBOLIC_INTEGRATE (0xB7) - Unchanged (Gauss-Kronrod is 1D)
// ----------------------------------------------------------------------------
// This one is fine as-is since integration over 1D interval

// ----------------------------------------------------------------------------
// OP_DIVERGENCE (0xB8) - TRUE MULTIVARIATE VERSION
// ----------------------------------------------------------------------------
// Computes ∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y + ∂F_z/∂z
//
// Stack input:  [... prog_x prog_y prog_z n_opcodes x y z n_vars h]
// Stack output: [... divergence]
//
case 0xB8: {  // OP_DIVERGENCE
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

    // Pop program length
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

    // Compute divergence: ∑ ∂F_comp/∂comp
    float divergence = 0.0f;
    for (int comp = 0; comp < n_vars; ++comp) {
        // Perturb component variable
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

// ----------------------------------------------------------------------------
// OP_CURL (0xB9) - TRUE MULTIVARIATE VERSION (3D only)
// ----------------------------------------------------------------------------
// Computes ∇×F = [∂F_z/∂y - ∂Fᵧ/∂z, ∂Fₓ/∂z - ∂F_z/∂x, ∂Fᵧ/∂x - ∂Fₓ/∂y]
//
// Stack input:  [... prog_x prog_y prog_z n_opcodes x y z h]
// Stack output: [... curl_x curl_y curl_z 3]
//
case 0xB9: {  // OP_CURL
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

    // Compute partial derivatives needed for curl
    float point_x_plus[3] = {point[0] + h, point[1], point[2]};
    float point_x_minus[3] = {point[0] - h, point[1], point[2]};
    float point_y_plus[3] = {point[0], point[1] + h, point[2]};
    float point_y_minus[3] = {point[0], point[1] - h, point[2]};
    float point_z_plus[3] = {point[0], point[1], point[2] + h};
    float point_z_minus[3] = {point[0], point[1], point[2] - h};

    // curl_x = ∂F_z/∂y - ∂Fᵧ/∂z
    float dFz_dy = (
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_y_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_y_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float dFy_dz = (
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_z_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[1], n_opcodes, point_z_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float curl_x = dFz_dy - dFy_dz;

    // curl_y = ∂Fₓ/∂z - ∂F_z/∂x
    float dFx_dz = (
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_z_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[0], n_opcodes, point_z_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float dFz_dx = (
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_x_plus, 3, error_code) -
        evaluate_rpn_function_multivar(programs[2], n_opcodes, point_x_minus, 3, error_code)
    ) / (2.0f * h);
    if (error_code != kErrorNone) break;

    float curl_y = dFx_dz - dFz_dx;

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

    float curl_z = dFy_dx - dFx_dy;

    // Push curl vector
    push_scalar(stack, stack_size, curl_x, error_code);
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, curl_y, error_code);
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, curl_z, error_code);
    if (error_code != kErrorNone) break;
    push_scalar(stack, stack_size, 3.0f, error_code);  // Dimension marker
    break;
}

// ----------------------------------------------------------------------------
// OP_LIMIT (0xBA) - Unchanged (Richardson extrapolation is 1D)
// ----------------------------------------------------------------------------
// This is fine as-is

// ----------------------------------------------------------------------------
// OP_LAPLACIAN (0xBB) - TRUE MULTIVARIATE VERSION
// ----------------------------------------------------------------------------
// Computes ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
//
// Stack input:  [... program n_opcodes x y z n_vars h]
// Stack output: [... laplacian]
//
case 0xBB: {  // OP_LAPLACIAN
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
    // Using finite difference: ∂²f/∂x² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    float laplacian = 0.0f;
    for (int var = 0; var < n_vars; ++var) {
        float point_plus[3] = {point[0], point[1], point[2]};
        point_plus[var] += h;
        float f_plus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_plus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        float point_minus[3] = {point[0], point[1], point[2]};
        point_minus[var] -= h;
        float f_minus = evaluate_rpn_function_multivar(
            program, n_opcodes, point_minus, n_vars, error_code);
        if (error_code != kErrorNone) break;

        laplacian += (f_plus - 2.0f * f_center + f_minus) / (h * h);
    }
    if (error_code != kErrorNone) break;

    push_scalar(stack, stack_size, laplacian, error_code);
    break;
}

// ============================================================================
// END OF MULTIVARIATE IMPLEMENTATION
// ============================================================================
