#pragma once

#include <stdint.h>

__device__ __forceinline__ int32_t k3d_round_i32(float value) {
    return value >= 0.0f
        ? static_cast<int32_t>(floorf(value + 0.5f))
        : static_cast<int32_t>(ceilf(value - 0.5f));
}

__device__ __forceinline__ uint32_t k3d_abs_term_u32(int32_t literal) {
    return static_cast<uint32_t>(literal < 0 ? -literal : literal);
}

__device__ __forceinline__ bool k3d_literals_are_complements(int32_t a_literal, int32_t b_literal) {
    if (a_literal == 0 || b_literal == 0) {
        return false;
    }
    return (a_literal < 0) != (b_literal < 0);
}

__device__ __forceinline__ uint32_t k3d_resolvent_handle(uint32_t subst, uint32_t a_term, uint32_t b_term) {
    return mix32(subst ^ (a_term * 0x9E3779B9u) ^ (b_term * 0x85EBCA6Bu) ^ 0xC1C1C1C1u);
}

__device__ __forceinline__ bool op_tresolve(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float b_scalar = 0.0f;
    float a_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, b_scalar, error)) {
        return false;
    }
    if (!pop_scalar(stack, stack_size, a_scalar, error)) {
        return false;
    }
    const int32_t a_literal = k3d_round_i32(a_scalar);
    const int32_t b_literal = k3d_round_i32(b_scalar);
    const uint32_t a_term = k3d_abs_term_u32(a_literal);
    const uint32_t b_term = k3d_abs_term_u32(b_literal);
    const uint32_t subst = robinson_unify_scalar_terms(a_term, b_term);
    const uint32_t resolvent = (k3d_literals_are_complements(a_literal, b_literal) && subst != K3D_SUBST_FAIL)
        ? k3d_resolvent_handle(subst, a_term, b_term)
        : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(resolvent)), error);
    return error == kErrorNone;
}
