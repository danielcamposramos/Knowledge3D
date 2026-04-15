#pragma once

#include <stdint.h>

constexpr uint32_t K3D_REWRITE_TERM_BITS = 12u;
constexpr uint32_t K3D_REWRITE_TERM_MASK = (1u << K3D_REWRITE_TERM_BITS) - 1u;

__device__ __forceinline__ uint32_t k3d_pack_rewrite_rule(uint32_t lhs_term, uint32_t rhs_term) {
    return (lhs_term & K3D_REWRITE_TERM_MASK) |
        ((rhs_term & K3D_REWRITE_TERM_MASK) << K3D_REWRITE_TERM_BITS);
}

__device__ __forceinline__ uint32_t k3d_rewrite_rule_lhs(uint32_t rule_handle) {
    return rule_handle & K3D_REWRITE_TERM_MASK;
}

__device__ __forceinline__ uint32_t k3d_rewrite_rule_rhs(uint32_t rule_handle) {
    return (rule_handle >> K3D_REWRITE_TERM_BITS) & K3D_REWRITE_TERM_MASK;
}

__device__ __forceinline__ bool k3d_rewrite_is_oriented(uint32_t lhs_term, uint32_t rhs_term) {
    const uint32_t lhs_weight = k3d_kbo_weight(lhs_term);
    const uint32_t rhs_weight = k3d_kbo_weight(rhs_term);
    if (lhs_weight != rhs_weight) {
        return lhs_weight > rhs_weight;
    }
    return k3d_kbo_precedence_rank(lhs_term) > k3d_kbo_precedence_rank(rhs_term);
}

__device__ __forceinline__ bool op_trewrite(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float rule_scalar = 0.0f;
    float term_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, rule_scalar, error)) {
        return false;
    }
    if (!pop_scalar(stack, stack_size, term_scalar, error)) {
        return false;
    }
    const uint32_t rule_handle = static_cast<uint32_t>(max(0.0f, floorf(rule_scalar + 0.5f)));
    const uint32_t term_handle = static_cast<uint32_t>(max(0.0f, floorf(term_scalar + 0.5f)));
    const uint32_t lhs_term = k3d_rewrite_rule_lhs(rule_handle);
    const uint32_t rhs_term = k3d_rewrite_rule_rhs(rule_handle);
    uint32_t rewritten_term = 0u;
    if (lhs_term != 0u && rhs_term != 0u && term_handle == lhs_term && k3d_rewrite_is_oriented(lhs_term, rhs_term)) {
        rewritten_term = rhs_term;
    }
    push(stack, stack_size, make_scalar(static_cast<float>(rewritten_term)), error);
    return error == kErrorNone;
}
