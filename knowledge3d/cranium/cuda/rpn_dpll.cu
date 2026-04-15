#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t k3d_clause_positive_mask(uint32_t clause) {
    return clause & 0xFFFFu;
}

__device__ __forceinline__ uint32_t k3d_clause_negative_mask(uint32_t clause) {
    return (clause >> 16) & 0xFFFFu;
}

__device__ __forceinline__ uint32_t k3d_trail_true_mask(uint32_t trail) {
    return trail & 0xFFFFu;
}

__device__ __forceinline__ uint32_t k3d_trail_false_mask(uint32_t trail) {
    return (trail >> 16) & 0xFFFFu;
}

__device__ __forceinline__ bool k3d_clause_satisfied(uint32_t clause, uint32_t trail) {
    const uint32_t positive = k3d_clause_positive_mask(clause);
    const uint32_t negative = k3d_clause_negative_mask(clause);
    return ((positive & k3d_trail_true_mask(trail)) != 0u) ||
        ((negative & k3d_trail_false_mask(trail)) != 0u);
}

__device__ __forceinline__ bool k3d_clause_conflict(uint32_t clause, uint32_t trail) {
    const uint32_t positive = k3d_clause_positive_mask(clause);
    const uint32_t negative = k3d_clause_negative_mask(clause);
    if (k3d_clause_satisfied(clause, trail)) {
        return false;
    }
    const bool all_positive_falsified = positive == 0u || (positive & ~k3d_trail_false_mask(trail)) == 0u;
    const bool all_negative_falsified = negative == 0u || (negative & ~k3d_trail_true_mask(trail)) == 0u;
    return all_positive_falsified && all_negative_falsified;
}

__device__ __forceinline__ bool op_tbcp(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float trail_scalar = 0.0f;
    float clause_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, trail_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, clause_scalar, error)) return false;
    const uint32_t trail = static_cast<uint32_t>(max(0.0f, floorf(trail_scalar + 0.5f)));
    const uint32_t clause = static_cast<uint32_t>(max(0.0f, floorf(clause_scalar + 0.5f)));
    const uint32_t conflict = k3d_clause_conflict(clause, trail) ? clause : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(conflict)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_tlearnt(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float trail_scalar = 0.0f;
    float conflict_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, trail_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, conflict_scalar, error)) return false;
    const uint32_t trail = static_cast<uint32_t>(max(0.0f, floorf(trail_scalar + 0.5f)));
    const uint32_t conflict = static_cast<uint32_t>(max(0.0f, floorf(conflict_scalar + 0.5f)));
    const uint32_t learnt = conflict == 0u ? 0u : (k3d_trail_true_mask(trail) << 16);
    push(stack, stack_size, make_scalar(static_cast<float>(learnt)), error);
    return error == kErrorNone;
}
