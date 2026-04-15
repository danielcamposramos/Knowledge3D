#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t k3d_popcount_u32(uint32_t value) {
    uint32_t count = 0u;
    while (value != 0u) {
        count += value & 1u;
        value >>= 1u;
    }
    return count;
}

__device__ __forceinline__ uint32_t k3d_reasoning_precedence_rank(uint32_t symbol) {
    switch (symbol & 0xFFu) {
        case 1u: return 240u;  // f
        case 2u: return 224u;  // g
        case 3u: return 208u;  // h
        case 4u: return 176u;  // x
        case 5u: return 160u;  // y
        case 6u: return 144u;  // z
        case 7u: return 96u;   // a
        case 8u: return 80u;   // b
        case 9u: return 64u;   // c
        case 10u: return 48u;  // d
        default:
            break;
    }
    return 1u + (mix32(symbol) & 0x3Fu);
}

__device__ __forceinline__ uint32_t k3d_kbo_precedence_rank(uint32_t term) {
    return k3d_reasoning_precedence_rank(term & ~K3D_TERM_VARIABLE_BIT);
}

__device__ __forceinline__ uint32_t k3d_kbo_weight(uint32_t term) {
    const uint32_t symbol = term & ~K3D_TERM_VARIABLE_BIT;
    const uint32_t precedence = k3d_kbo_precedence_rank(symbol);
    const uint32_t structural = k3d_popcount_u32(symbol);
    const uint32_t variable_penalty = k3d_term_is_variable(term) ? 1u : 4u;
    return precedence + structural + variable_penalty;
}

__device__ __forceinline__ bool op_torder(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float rhs_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, rhs_scalar, error)) {
        return false;
    }
    const int32_t rhs_signed = k3d_round_i32(rhs_scalar);
    const uint32_t rhs_term = k3d_abs_term_u32(rhs_signed);
    if (stack_size > 0u) {
        float lhs_scalar = 0.0f;
        if (!pop_scalar(stack, stack_size, lhs_scalar, error)) {
            return false;
        }
        const int32_t lhs_signed = k3d_round_i32(lhs_scalar);
        const uint32_t lhs_term = k3d_abs_term_u32(lhs_signed);
        const uint32_t lhs_weight = k3d_kbo_weight(lhs_term);
        const uint32_t rhs_weight = k3d_kbo_weight(rhs_term);
        int32_t order = 0;
        if (lhs_weight > rhs_weight) {
            order = 1;
        } else if (lhs_weight < rhs_weight) {
            order = -1;
        } else {
            const uint32_t lhs_prec = k3d_kbo_precedence_rank(lhs_term);
            const uint32_t rhs_prec = k3d_kbo_precedence_rank(rhs_term);
            if (lhs_prec > rhs_prec) {
                order = 1;
            } else if (lhs_prec < rhs_prec) {
                order = -1;
            }
        }
        push(stack, stack_size, make_scalar(static_cast<float>(order)), error);
        return error == kErrorNone;
    }
    push(stack, stack_size, make_scalar(static_cast<float>(k3d_kbo_weight(rhs_term))), error);
    return error == kErrorNone;
}
