#pragma once

#include <stdint.h>

constexpr uint32_t K3D_CASE_ID_BITS = 6u;
constexpr uint32_t K3D_CASE_ANCHOR_BITS = 8u;
constexpr uint32_t K3D_CASE_CONTEXT_BITS = 6u;
constexpr uint32_t K3D_CASE_ETHICAL_BITS = 2u;
constexpr uint32_t K3D_CASE_FLAGS_BITS = 2u;

constexpr uint32_t K3D_CASE_ID_MASK = (1u << K3D_CASE_ID_BITS) - 1u;
constexpr uint32_t K3D_CASE_ANCHOR_MASK = (1u << K3D_CASE_ANCHOR_BITS) - 1u;
constexpr uint32_t K3D_CASE_CONTEXT_MASK = (1u << K3D_CASE_CONTEXT_BITS) - 1u;
constexpr uint32_t K3D_CASE_ETHICAL_MASK = (1u << K3D_CASE_ETHICAL_BITS) - 1u;
constexpr uint32_t K3D_CASE_FLAGS_MASK = (1u << K3D_CASE_FLAGS_BITS) - 1u;

enum K3DCaseEthicalCode : uint32_t {
    K3D_CASE_ETHICAL_FORBIDDEN = 0u,
    K3D_CASE_ETHICAL_OK = 1u,
    K3D_CASE_ETHICAL_DEFEASIBLE = 2u,
};

__device__ __forceinline__ uint32_t k3d_case_pack(
    uint32_t case_id,
    uint32_t anchor,
    uint32_t context_id,
    uint32_t ethical_code,
    uint32_t flags
) {
    return (case_id & K3D_CASE_ID_MASK) |
        ((anchor & K3D_CASE_ANCHOR_MASK) << K3D_CASE_ID_BITS) |
        ((context_id & K3D_CASE_CONTEXT_MASK) << (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS)) |
        ((ethical_code & K3D_CASE_ETHICAL_MASK) << (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS + K3D_CASE_CONTEXT_BITS)) |
        ((flags & K3D_CASE_FLAGS_MASK) << (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS + K3D_CASE_CONTEXT_BITS + K3D_CASE_ETHICAL_BITS));
}

__device__ __forceinline__ uint32_t k3d_case_id(uint32_t handle) {
    return handle & K3D_CASE_ID_MASK;
}

__device__ __forceinline__ uint32_t k3d_case_anchor(uint32_t handle) {
    return (handle >> K3D_CASE_ID_BITS) & K3D_CASE_ANCHOR_MASK;
}

__device__ __forceinline__ uint32_t k3d_case_context(uint32_t handle) {
    return (handle >> (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS)) & K3D_CASE_CONTEXT_MASK;
}

__device__ __forceinline__ uint32_t k3d_case_ethical(uint32_t handle) {
    return (handle >> (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS + K3D_CASE_CONTEXT_BITS)) & K3D_CASE_ETHICAL_MASK;
}

__device__ __forceinline__ uint32_t k3d_case_flags(uint32_t handle) {
    return (handle >> (K3D_CASE_ID_BITS + K3D_CASE_ANCHOR_BITS + K3D_CASE_CONTEXT_BITS + K3D_CASE_ETHICAL_BITS)) & K3D_CASE_FLAGS_MASK;
}

__device__ __forceinline__ uint32_t k3d_case_pack_rebind(
    uint32_t symbol_mask,
    uint32_t anchor_bias,
    uint32_t context_override,
    uint32_t flags
) {
    return (symbol_mask & 0xFFu) |
        ((anchor_bias & 0xFFu) << 8) |
        ((context_override & 0x3Fu) << 16) |
        ((flags & 0x3u) << 22);
}

__device__ __forceinline__ uint32_t k3d_case_rebind_symbol_mask(uint32_t handle) {
    return handle & 0xFFu;
}

__device__ __forceinline__ uint32_t k3d_case_rebind_anchor_bias(uint32_t handle) {
    return (handle >> 8) & 0xFFu;
}

__device__ __forceinline__ uint32_t k3d_case_rebind_context(uint32_t handle) {
    return (handle >> 16) & 0x3Fu;
}

__device__ __forceinline__ uint32_t k3d_case_rebind_flags(uint32_t handle) {
    return (handle >> 22) & 0x3u;
}

__device__ __forceinline__ uint32_t k3d_case_pack_constraint(
    uint32_t anchor_floor,
    uint32_t revise_delta,
    uint32_t required_context,
    uint32_t ethical_policy,
    uint32_t conflict_code
) {
    return (anchor_floor & 0xFFu) |
        ((revise_delta & 0x3Fu) << 8) |
        ((required_context & 0x3Fu) << 14) |
        ((ethical_policy & 0x3u) << 20) |
        ((conflict_code & 0x3u) << 22);
}

__device__ __forceinline__ uint32_t k3d_case_constraint_anchor_floor(uint32_t handle) {
    return handle & 0xFFu;
}

__device__ __forceinline__ uint32_t k3d_case_constraint_revise_delta(uint32_t handle) {
    return (handle >> 8) & 0x3Fu;
}

__device__ __forceinline__ uint32_t k3d_case_constraint_context(uint32_t handle) {
    return (handle >> 14) & 0x3Fu;
}

__device__ __forceinline__ uint32_t k3d_case_constraint_policy(uint32_t handle) {
    return (handle >> 20) & 0x3u;
}

__device__ __forceinline__ uint32_t k3d_case_constraint_conflict(uint32_t handle) {
    return (handle >> 22) & 0x3u;
}

__device__ __forceinline__ bool k3d_case_context_ok(uint32_t active_context, uint32_t case_context, uint32_t required_context) {
    const uint32_t effective_context = required_context != 0u ? required_context : active_context;
    return rpn_context_allows_star(effective_context, case_context);
}

__device__ __forceinline__ bool k3d_case_stage3_gate(uint32_t ethical_code, uint32_t conflict_code) {
    if (ethical_code == K3D_CASE_ETHICAL_FORBIDDEN) {
        return false;
    }
    if (ethical_code == K3D_CASE_ETHICAL_OK) {
        return true;
    }
    return conflict_code == 2u;
}

__device__ __forceinline__ uint32_t k3d_case_similarity(uint32_t query_anchor, uint32_t candidate_anchor) {
    const uint32_t delta = query_anchor > candidate_anchor
        ? (query_anchor - candidate_anchor)
        : (candidate_anchor - query_anchor);
    return 255u - min(delta, 255u);
}

__device__ __forceinline__ uint32_t k3d_case_select_best(
    uint32_t query,
    uint32_t candidate_a,
    uint32_t candidate_b,
    uint32_t active_context
) {
    const uint32_t query_anchor = k3d_case_anchor(query);
    const uint32_t query_context = k3d_case_context(query);

    uint32_t best = 0u;
    uint32_t best_score = 0u;
    const uint32_t candidates[2] = {candidate_a, candidate_b};
    #pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t candidate = candidates[i];
        if (k3d_case_id(candidate) == 0u) {
            continue;
        }
        const uint32_t ethical = k3d_case_ethical(candidate);
        if (ethical == K3D_CASE_ETHICAL_FORBIDDEN) {
            continue;
        }
        if (!k3d_case_context_ok(active_context, k3d_case_context(candidate), query_context)) {
            continue;
        }
        uint32_t score = k3d_case_similarity(query_anchor, k3d_case_anchor(candidate));
        if (k3d_case_context(candidate) == query_context && query_context != 0u) {
            score += 24u;
        }
        if (ethical == K3D_CASE_ETHICAL_OK) {
            score += 12u;
        } else if (ethical == K3D_CASE_ETHICAL_DEFEASIBLE) {
            score += 6u;
        }
        if (score > best_score || (score == best_score && k3d_case_id(candidate) < k3d_case_id(best))) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

__device__ __forceinline__ bool op_case_fetch(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t active_context
) {
    float candidate_b_scalar = 0.0f;
    float candidate_a_scalar = 0.0f;
    float query_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, candidate_b_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, candidate_a_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, query_scalar, error)) return false;

    const uint32_t query = static_cast<uint32_t>(max(0.0f, floorf(query_scalar + 0.5f)));
    const uint32_t candidate_a = static_cast<uint32_t>(max(0.0f, floorf(candidate_a_scalar + 0.5f)));
    const uint32_t candidate_b = static_cast<uint32_t>(max(0.0f, floorf(candidate_b_scalar + 0.5f)));
    const uint32_t best = k3d_case_select_best(query, candidate_a, candidate_b, active_context);
    push(stack, stack_size, make_scalar(static_cast<float>(best)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_case_rebind(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float rebind_scalar = 0.0f;
    float case_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, rebind_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, case_scalar, error)) return false;

    const uint32_t case_handle = static_cast<uint32_t>(max(0.0f, floorf(case_scalar + 0.5f)));
    const uint32_t rebind = static_cast<uint32_t>(max(0.0f, floorf(rebind_scalar + 0.5f)));
    uint32_t rebound = 0u;
    if (case_handle != 0u && k3d_case_id(case_handle) != 0u) {
        const uint32_t anchor = (k3d_case_anchor(case_handle) ^ k3d_case_rebind_symbol_mask(rebind) ^ k3d_case_rebind_anchor_bias(rebind)) & K3D_CASE_ANCHOR_MASK;
        const uint32_t context_id = k3d_case_rebind_context(rebind) != 0u
            ? k3d_case_rebind_context(rebind)
            : k3d_case_context(case_handle);
        const uint32_t flags = (k3d_case_flags(case_handle) ^ k3d_case_rebind_flags(rebind)) & K3D_CASE_FLAGS_MASK;
        rebound = k3d_case_pack(
            k3d_case_id(case_handle),
            anchor,
            context_id,
            k3d_case_ethical(case_handle),
            flags
        );
    }
    push(stack, stack_size, make_scalar(static_cast<float>(rebound)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_case_revise(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t active_context
) {
    float constraint_scalar = 0.0f;
    float case_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, constraint_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, case_scalar, error)) return false;

    const uint32_t case_handle = static_cast<uint32_t>(max(0.0f, floorf(case_scalar + 0.5f)));
    const uint32_t constraint = static_cast<uint32_t>(max(0.0f, floorf(constraint_scalar + 0.5f)));
    uint32_t revised = 0u;
    if (case_handle != 0u && k3d_case_id(case_handle) != 0u) {
        const uint32_t ethical_code = k3d_case_ethical(case_handle);
        const uint32_t required_context = k3d_case_constraint_context(constraint);
        const uint32_t policy = k3d_case_constraint_policy(constraint);
        const bool context_ok = k3d_case_context_ok(active_context, k3d_case_context(case_handle), required_context);
        const bool policy_ok = k3d_rete_policy_allows(ethical_code, policy);
        const bool anchor_ok = k3d_case_anchor(case_handle) >= k3d_case_constraint_anchor_floor(constraint);
        const bool ethical_gate_ok = k3d_case_stage3_gate(ethical_code, k3d_case_constraint_conflict(constraint));
        if (context_ok && policy_ok && anchor_ok && ethical_gate_ok) {
            const uint32_t revised_anchor = min(
                static_cast<uint32_t>(255u),
                k3d_case_anchor(case_handle) + k3d_case_constraint_revise_delta(constraint)
            );
            revised = k3d_case_pack(
                k3d_case_id(case_handle),
                revised_anchor,
                k3d_case_context(case_handle),
                ethical_code,
                min(k3d_case_flags(case_handle) + 1u, K3D_CASE_FLAGS_MASK)
            );
        }
    }
    push(stack, stack_size, make_scalar(static_cast<float>(revised)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_case_retain_hint(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float case_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, case_scalar, error)) return false;
    const uint32_t case_handle = static_cast<uint32_t>(max(0.0f, floorf(case_scalar + 0.5f)));

    uint32_t hint = 0u;
    if (case_handle != 0u && k3d_case_id(case_handle) != 0u) {
        hint = k3d_case_pack(
            k3d_case_id(case_handle),
            k3d_case_anchor(case_handle),
            k3d_case_context(case_handle),
            k3d_case_ethical(case_handle),
            min(k3d_case_flags(case_handle) + 1u, K3D_CASE_FLAGS_MASK)
        );
    }
    push(stack, stack_size, make_scalar(static_cast<float>(hint)), error);
    return error == kErrorNone;
}
