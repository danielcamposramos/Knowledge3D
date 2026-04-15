#pragma once

#include <stdint.h>

constexpr uint32_t K3D_ALP_HEAD_BITS = 8u;
constexpr uint32_t K3D_ALP_MASK_BITS = 8u;
constexpr uint32_t K3D_ALP_HEAD_MASK = (1u << K3D_ALP_HEAD_BITS) - 1u;
constexpr uint32_t K3D_ALP_BODY_MASK_SHIFT = K3D_ALP_HEAD_BITS;
constexpr uint32_t K3D_ALP_IC_MASK_SHIFT = K3D_ALP_HEAD_BITS + K3D_ALP_MASK_BITS;
constexpr uint32_t K3D_ALP_BODY_MASK = ((1u << K3D_ALP_MASK_BITS) - 1u) << K3D_ALP_BODY_MASK_SHIFT;
constexpr uint32_t K3D_ALP_IC_MASK = ((1u << K3D_ALP_MASK_BITS) - 1u) << K3D_ALP_IC_MASK_SHIFT;

__device__ __forceinline__ uint32_t k3d_alp_head_symbol(uint32_t horn_rule_handle) {
    return horn_rule_handle & K3D_ALP_HEAD_MASK;
}

__device__ __forceinline__ uint32_t k3d_alp_body_mask(uint32_t horn_rule_handle) {
    return (horn_rule_handle & K3D_ALP_BODY_MASK) >> K3D_ALP_BODY_MASK_SHIFT;
}

__device__ __forceinline__ uint32_t k3d_alp_ic_mask(uint32_t horn_rule_handle) {
    return (horn_rule_handle & K3D_ALP_IC_MASK) >> K3D_ALP_IC_MASK_SHIFT;
}

__device__ __forceinline__ bool op_alpchain(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float horn_rule_scalar = 0.0f;
    float goal_mask_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, horn_rule_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, goal_mask_scalar, error)) return false;

    const uint32_t horn_rule = static_cast<uint32_t>(max(0.0f, floorf(horn_rule_scalar + 0.5f)));
    const uint32_t goal_mask = static_cast<uint32_t>(max(0.0f, floorf(goal_mask_scalar + 0.5f))) & 0xFFu;
    const uint32_t head_symbol = k3d_alp_head_symbol(horn_rule) & 0x1Fu;
    const uint32_t body_mask = k3d_alp_body_mask(horn_rule) & 0xFFu;
    const uint32_t head_bit = (head_symbol == 0u || head_symbol >= 32u) ? 0u : (1u << head_symbol);

    uint32_t residual_goal = 0u;
    if ((goal_mask & head_bit) != 0u) {
        residual_goal = (goal_mask & ~head_bit) | body_mask;
    }
    push(stack, stack_size, make_scalar(static_cast<float>(residual_goal)), error);
    return error == kErrorNone;
}
