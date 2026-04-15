#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t k3d_superposition_pair_handle(
    uint32_t rule_handle,
    uint32_t target_term,
    uint32_t replacement_term
) {
    return mix32(
        rule_handle ^
        (target_term * 0x9E3779B9u) ^
        (replacement_term * 0x85EBCA6Bu) ^
        0xC4C4C4C4u
    );
}

__device__ __forceinline__ bool op_tsuperpos(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float target_scalar = 0.0f;
    float rule_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, target_scalar, error)) {
        return false;
    }
    if (!pop_scalar(stack, stack_size, rule_scalar, error)) {
        return false;
    }
    const uint32_t rule_handle = static_cast<uint32_t>(max(0.0f, floorf(rule_scalar + 0.5f)));
    const uint32_t target_term = static_cast<uint32_t>(max(0.0f, floorf(target_scalar + 0.5f)));
    const uint32_t lhs_term = k3d_rewrite_rule_lhs(rule_handle);
    const uint32_t rhs_term = k3d_rewrite_rule_rhs(rule_handle);

    uint32_t critical_pair = 0u;
    if (lhs_term != 0u && rhs_term != 0u && k3d_rewrite_is_oriented(lhs_term, rhs_term)) {
        if (target_term == lhs_term || target_term == rhs_term) {
            const uint32_t replacement_term = (target_term == lhs_term) ? rhs_term : lhs_term;
            critical_pair = k3d_superposition_pair_handle(rule_handle, target_term, replacement_term);
        }
    }
    push(stack, stack_size, make_scalar(static_cast<float>(critical_pair)), error);
    return error == kErrorNone;
}
