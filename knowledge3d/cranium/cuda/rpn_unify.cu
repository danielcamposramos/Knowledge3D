#pragma once

#include <stdint.h>

#define K3D_TERM_VARIABLE_BIT 0x80000000u
#define K3D_SUBST_FAIL 0u

__device__ __forceinline__ bool k3d_term_is_variable(uint32_t term) {
    return (term & K3D_TERM_VARIABLE_BIT) != 0u;
}

__device__ __forceinline__ uint32_t k3d_subst_handle(uint32_t a, uint32_t b) {
    uint32_t x = a ^ (b * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    return x == 0u ? 1u : x;
}

__device__ __forceinline__ uint32_t robinson_unify_scalar_terms(uint32_t a, uint32_t b) {
    if (a == b) {
        return k3d_subst_handle(a, b);
    }
    const bool a_var = k3d_term_is_variable(a);
    const bool b_var = k3d_term_is_variable(b);
    if (a_var && b_var) {
        return k3d_subst_handle(a, b);
    }
    if (a_var) {
        const uint32_t variable_id = a & ~K3D_TERM_VARIABLE_BIT;
        return variable_id == b ? K3D_SUBST_FAIL : k3d_subst_handle(a, b);
    }
    if (b_var) {
        const uint32_t variable_id = b & ~K3D_TERM_VARIABLE_BIT;
        return variable_id == a ? K3D_SUBST_FAIL : k3d_subst_handle(a, b);
    }
    return K3D_SUBST_FAIL;
}

__device__ __forceinline__ bool op_tunify(
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
    const uint32_t a = static_cast<uint32_t>(max(0.0f, floorf(a_scalar + 0.5f)));
    const uint32_t b = static_cast<uint32_t>(max(0.0f, floorf(b_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(robinson_unify_scalar_terms(a, b))), error);
    return error == kErrorNone;
}
