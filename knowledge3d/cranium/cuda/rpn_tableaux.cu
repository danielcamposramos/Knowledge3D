#pragma once

#include <stdint.h>

__device__ __constant__ uint32_t k3d_tableaux_skolem_counter = 1u;

constexpr uint32_t K3D_BRANCH_NODE_BITS = 8u;
constexpr uint32_t K3D_BRANCH_MASK_BITS = 16u;
constexpr uint32_t K3D_BRANCH_NODE_MASK = (1u << K3D_BRANCH_NODE_BITS) - 1u;
constexpr uint32_t K3D_BRANCH_CONCEPT_MASK = (1u << K3D_BRANCH_MASK_BITS) - 1u;
constexpr uint32_t K3D_EDGE_NODE_BITS = 12u;
constexpr uint32_t K3D_EDGE_NODE_MASK = (1u << K3D_EDGE_NODE_BITS) - 1u;

__device__ __forceinline__ uint32_t k3d_pack_branch_handle(uint32_t node_id, uint32_t concept_mask) {
    return (node_id & K3D_BRANCH_NODE_MASK) |
        ((concept_mask & K3D_BRANCH_CONCEPT_MASK) << K3D_BRANCH_NODE_BITS);
}

__device__ __forceinline__ uint32_t k3d_branch_node_id(uint32_t branch_handle) {
    return branch_handle & K3D_BRANCH_NODE_MASK;
}

__device__ __forceinline__ uint32_t k3d_branch_concept_mask(uint32_t branch_handle) {
    return (branch_handle >> K3D_BRANCH_NODE_BITS) & K3D_BRANCH_CONCEPT_MASK;
}

__device__ __forceinline__ uint32_t k3d_pack_edge_handle(uint32_t src_node, uint32_t dst_node) {
    return (src_node & K3D_EDGE_NODE_MASK) |
        ((dst_node & K3D_EDGE_NODE_MASK) << K3D_EDGE_NODE_BITS);
}

__device__ __forceinline__ uint32_t k3d_edge_src(uint32_t edge_handle) {
    return edge_handle & K3D_EDGE_NODE_MASK;
}

__device__ __forceinline__ uint32_t k3d_edge_dst(uint32_t edge_handle) {
    return (edge_handle >> K3D_EDGE_NODE_BITS) & K3D_EDGE_NODE_MASK;
}

__device__ __forceinline__ uint32_t tableaux_branch_id(uint32_t formula, uint32_t side) {
    uint32_t x = formula ^ (side * 0x85EBCA6Bu);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    return x == 0u ? side + 1u : x;
}

__device__ __forceinline__ bool op_tsplit(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float beta_scalar = 0.0f;
    float alpha_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, beta_scalar, error)) {
        return false;
    }
    if (!pop_scalar(stack, stack_size, alpha_scalar, error)) {
        return false;
    }
    const uint32_t alpha = static_cast<uint32_t>(max(0.0f, floorf(alpha_scalar + 0.5f)));
    const uint32_t beta = static_cast<uint32_t>(max(0.0f, floorf(beta_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(tableaux_branch_id(alpha, 1u))), error);
    push(stack, stack_size, make_scalar(static_cast<float>(tableaux_branch_id(beta, 2u))), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_tclose(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float lit_b_scalar = 0.0f;
    float lit_a_scalar = 0.0f;
    float branch_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, lit_b_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, lit_a_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, branch_scalar, error)) return false;
    const int32_t lit_a = static_cast<int32_t>(floorf(lit_a_scalar + (lit_a_scalar >= 0.0f ? 0.5f : -0.5f)));
    const int32_t lit_b = static_cast<int32_t>(floorf(lit_b_scalar + (lit_b_scalar >= 0.0f ? 0.5f : -0.5f)));
    const uint32_t closed = (branch_scalar > 0.0f && lit_a != 0 && lit_a == -lit_b) ? 1u : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(closed)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_texpand(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float formula_scalar = 0.0f;
    float branch_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, formula_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, branch_scalar, error)) return false;
    const uint32_t formula = static_cast<uint32_t>(max(0.0f, floorf(formula_scalar + 0.5f)));
    const uint32_t branch = static_cast<uint32_t>(max(0.0f, floorf(branch_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(tableaux_branch_id(formula ^ branch, 3u))), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_euler_complete(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float edge_bc_scalar = 0.0f;
    float edge_ab_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, edge_bc_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, edge_ab_scalar, error)) return false;
    const uint32_t edge_ab = static_cast<uint32_t>(max(0.0f, floorf(edge_ab_scalar + 0.5f)));
    const uint32_t edge_bc = static_cast<uint32_t>(max(0.0f, floorf(edge_bc_scalar + 0.5f)));
    const uint32_t a = k3d_edge_src(edge_ab);
    const uint32_t b_left = k3d_edge_dst(edge_ab);
    const uint32_t b_right = k3d_edge_src(edge_bc);
    const uint32_t c = k3d_edge_dst(edge_bc);
    const uint32_t closure = (a != 0u && c != 0u && b_left != 0u && b_left == b_right)
        ? k3d_pack_edge_handle(a, c)
        : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(closure)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_dl_saturate(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float concept_mask_scalar = 0.0f;
    float branch_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, concept_mask_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, branch_scalar, error)) return false;
    const uint32_t branch_handle = static_cast<uint32_t>(max(0.0f, floorf(branch_scalar + 0.5f)));
    const uint32_t concept_mask = static_cast<uint32_t>(max(0.0f, floorf(concept_mask_scalar + 0.5f))) & K3D_BRANCH_CONCEPT_MASK;
    const uint32_t node_id = k3d_branch_node_id(branch_handle);
    const uint32_t existing_mask = k3d_branch_concept_mask(branch_handle);
    const uint32_t saturated = (node_id != 0u)
        ? k3d_pack_branch_handle(node_id, existing_mask | concept_mask)
        : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(saturated)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_blocking_check(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float predecessor_scalar = 0.0f;
    float node_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, predecessor_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, node_scalar, error)) return false;
    const uint32_t node_handle = static_cast<uint32_t>(max(0.0f, floorf(node_scalar + 0.5f)));
    const uint32_t predecessor_handle = static_cast<uint32_t>(max(0.0f, floorf(predecessor_scalar + 0.5f)));
    const uint32_t node_mask = k3d_branch_concept_mask(node_handle);
    const uint32_t predecessor_mask = k3d_branch_concept_mask(predecessor_handle);
    const uint32_t blocked = (
        k3d_branch_node_id(node_handle) != 0u &&
        k3d_branch_node_id(predecessor_handle) != 0u &&
        predecessor_mask != 0u &&
        (node_mask & ~predecessor_mask) == 0u
    ) ? 1u : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(blocked)), error);
    return error == kErrorNone;
}
