#pragma once

#include <stdint.h>

constexpr uint32_t K3D_RETE_PREDICATE_BITS = 8u;
constexpr uint32_t K3D_RETE_CONTEXT_BITS = 8u;
constexpr uint32_t K3D_RETE_CLUSTER_BITS = 4u;
constexpr uint32_t K3D_RETE_ETHICAL_BITS = 2u;
constexpr uint32_t K3D_RETE_HEURISTIC_BITS = 2u;
constexpr uint32_t K3D_RETE_TOKEN_BINDING_BITS = 8u;
constexpr uint32_t K3D_RETE_TOKEN_JOIN_BITS = 8u;
constexpr uint32_t K3D_RETE_TOKEN_CONTEXT_BITS = 4u;
constexpr uint32_t K3D_RETE_TOKEN_CLUSTER_BITS = 4u;
constexpr uint32_t K3D_RETE_AGENDA_PRIORITY_BITS = 8u;
constexpr uint32_t K3D_RETE_AGENDA_PAYLOAD_BITS = 8u;
constexpr uint32_t K3D_RETE_AGENDA_DEPTH_BITS = 8u;

constexpr uint32_t K3D_RETE_PREDICATE_MASK = (1u << K3D_RETE_PREDICATE_BITS) - 1u;
constexpr uint32_t K3D_RETE_CONTEXT_MASK = (1u << K3D_RETE_CONTEXT_BITS) - 1u;
constexpr uint32_t K3D_RETE_CLUSTER_MASK = (1u << K3D_RETE_CLUSTER_BITS) - 1u;
constexpr uint32_t K3D_RETE_ETHICAL_MASK = (1u << K3D_RETE_ETHICAL_BITS) - 1u;
constexpr uint32_t K3D_RETE_HEURISTIC_MASK = (1u << K3D_RETE_HEURISTIC_BITS) - 1u;

enum K3DReteEthicalPolicy : uint32_t {
    K3D_RETE_POLICY_ANY_NON_FORBIDDEN = 0u,
    K3D_RETE_POLICY_OK_ONLY = 1u,
    K3D_RETE_POLICY_OK_OR_DEFEASIBLE = 2u,
};

__device__ __forceinline__ uint32_t k3d_rete_pack_fact(
    uint32_t predicate_mask,
    uint32_t context_id,
    uint32_t cluster_id,
    uint32_t ethical_code
) {
    return (predicate_mask & K3D_RETE_PREDICATE_MASK) |
        ((context_id & K3D_RETE_CONTEXT_MASK) << K3D_RETE_PREDICATE_BITS) |
        ((cluster_id & K3D_RETE_CLUSTER_MASK) << (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS)) |
        ((ethical_code & K3D_RETE_ETHICAL_MASK) << (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS + K3D_RETE_CLUSTER_BITS));
}

__device__ __forceinline__ uint32_t k3d_rete_fact_predicate_mask(uint32_t fact_handle) {
    return fact_handle & K3D_RETE_PREDICATE_MASK;
}

__device__ __forceinline__ uint32_t k3d_rete_fact_context(uint32_t fact_handle) {
    return (fact_handle >> K3D_RETE_PREDICATE_BITS) & K3D_RETE_CONTEXT_MASK;
}

__device__ __forceinline__ uint32_t k3d_rete_fact_cluster(uint32_t fact_handle) {
    return (fact_handle >> (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS)) & K3D_RETE_CLUSTER_MASK;
}

__device__ __forceinline__ uint32_t k3d_rete_fact_ethical(uint32_t fact_handle) {
    return (fact_handle >> (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS + K3D_RETE_CLUSTER_BITS)) & K3D_RETE_ETHICAL_MASK;
}

__device__ __forceinline__ uint32_t k3d_rete_pack_alpha(
    uint32_t required_predicate_mask,
    uint32_t required_context,
    uint32_t required_cluster,
    uint32_t ethical_policy
) {
    return k3d_rete_pack_fact(required_predicate_mask, required_context, required_cluster, ethical_policy);
}

__device__ __forceinline__ uint32_t k3d_rete_pack_alpha_heuristic(
    uint32_t required_predicate_mask,
    uint32_t required_context,
    uint32_t required_cluster,
    uint32_t ethical_policy,
    uint32_t heuristic_floor
) {
    return k3d_rete_pack_fact(required_predicate_mask, required_context, required_cluster, ethical_policy) |
        ((heuristic_floor & K3D_RETE_HEURISTIC_MASK)
         << (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS + K3D_RETE_CLUSTER_BITS + K3D_RETE_ETHICAL_BITS));
}

__device__ __forceinline__ uint32_t k3d_rete_alpha_heuristic(uint32_t alpha_handle) {
    return (alpha_handle >> (K3D_RETE_PREDICATE_BITS + K3D_RETE_CONTEXT_BITS + K3D_RETE_CLUSTER_BITS + K3D_RETE_ETHICAL_BITS)) & K3D_RETE_HEURISTIC_MASK;
}

__device__ __forceinline__ bool k3d_rete_policy_allows(uint32_t fact_ethical, uint32_t policy) {
    switch (policy) {
        case K3D_RETE_POLICY_OK_ONLY:
            return fact_ethical == 1u;
        case K3D_RETE_POLICY_OK_OR_DEFEASIBLE:
            return fact_ethical == 1u || fact_ethical == 2u;
        case K3D_RETE_POLICY_ANY_NON_FORBIDDEN:
        default:
            return fact_ethical != 0u;
    }
}

__device__ __forceinline__ uint32_t k3d_rete_pack_token(
    uint32_t binding_mask,
    uint32_t join_key,
    uint32_t context_id,
    uint32_t cluster_id
) {
    return (binding_mask & ((1u << K3D_RETE_TOKEN_BINDING_BITS) - 1u)) |
        ((join_key & ((1u << K3D_RETE_TOKEN_JOIN_BITS) - 1u)) << K3D_RETE_TOKEN_BINDING_BITS) |
        ((context_id & ((1u << K3D_RETE_TOKEN_CONTEXT_BITS) - 1u)) << (K3D_RETE_TOKEN_BINDING_BITS + K3D_RETE_TOKEN_JOIN_BITS)) |
        ((cluster_id & ((1u << K3D_RETE_TOKEN_CLUSTER_BITS) - 1u)) << (K3D_RETE_TOKEN_BINDING_BITS + K3D_RETE_TOKEN_JOIN_BITS + K3D_RETE_TOKEN_CONTEXT_BITS));
}

__device__ __forceinline__ uint32_t k3d_rete_token_binding_mask(uint32_t token_handle) {
    return token_handle & ((1u << K3D_RETE_TOKEN_BINDING_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_token_join_key(uint32_t token_handle) {
    return (token_handle >> K3D_RETE_TOKEN_BINDING_BITS) & ((1u << K3D_RETE_TOKEN_JOIN_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_token_context(uint32_t token_handle) {
    return (token_handle >> (K3D_RETE_TOKEN_BINDING_BITS + K3D_RETE_TOKEN_JOIN_BITS)) & ((1u << K3D_RETE_TOKEN_CONTEXT_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_token_cluster(uint32_t token_handle) {
    return (token_handle >> (K3D_RETE_TOKEN_BINDING_BITS + K3D_RETE_TOKEN_JOIN_BITS + K3D_RETE_TOKEN_CONTEXT_BITS)) & ((1u << K3D_RETE_TOKEN_CLUSTER_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_pack_agenda(
    uint32_t priority,
    uint32_t payload,
    uint32_t depth
) {
    return (priority & ((1u << K3D_RETE_AGENDA_PRIORITY_BITS) - 1u)) |
        ((payload & ((1u << K3D_RETE_AGENDA_PAYLOAD_BITS) - 1u)) << K3D_RETE_AGENDA_PRIORITY_BITS) |
        ((depth & ((1u << K3D_RETE_AGENDA_DEPTH_BITS) - 1u)) << (K3D_RETE_AGENDA_PRIORITY_BITS + K3D_RETE_AGENDA_PAYLOAD_BITS));
}

__device__ __forceinline__ uint32_t k3d_rete_agenda_priority(uint32_t agenda_handle) {
    return agenda_handle & ((1u << K3D_RETE_AGENDA_PRIORITY_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_agenda_payload(uint32_t agenda_handle) {
    return (agenda_handle >> K3D_RETE_AGENDA_PRIORITY_BITS) & ((1u << K3D_RETE_AGENDA_PAYLOAD_BITS) - 1u);
}

__device__ __forceinline__ uint32_t k3d_rete_agenda_depth(uint32_t agenda_handle) {
    return (agenda_handle >> (K3D_RETE_AGENDA_PRIORITY_BITS + K3D_RETE_AGENDA_PAYLOAD_BITS)) & ((1u << K3D_RETE_AGENDA_DEPTH_BITS) - 1u);
}

__device__ __forceinline__ bool k3d_rete_agenda_precedes(uint32_t lhs, uint32_t rhs) {
    const uint32_t lhs_priority = k3d_rete_agenda_priority(lhs);
    const uint32_t rhs_priority = k3d_rete_agenda_priority(rhs);
    if (lhs_priority != rhs_priority) {
        return lhs_priority > rhs_priority;
    }
    const uint32_t lhs_payload = k3d_rete_agenda_payload(lhs);
    const uint32_t rhs_payload = k3d_rete_agenda_payload(rhs);
    if (lhs_payload != rhs_payload) {
        return lhs_payload > rhs_payload;
    }
    return k3d_rete_agenda_depth(lhs) <= k3d_rete_agenda_depth(rhs);
}

__device__ __forceinline__ void k3d_rete_bitonic_sort32_desc(uint32_t* agenda) {
    for (uint32_t k = 2u; k <= 32u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = 0u; i < 32u; ++i) {
                const uint32_t ixj = i ^ j;
                if (ixj <= i || ixj >= 32u) {
                    continue;
                }
                const bool ascending = ((i & k) == 0u);
                const bool should_swap = ascending
                    ? !k3d_rete_agenda_precedes(agenda[i], agenda[ixj])
                    : k3d_rete_agenda_precedes(agenda[i], agenda[ixj]);
                if (should_swap) {
                    const uint32_t tmp = agenda[i];
                    agenda[i] = agenda[ixj];
                    agenda[ixj] = tmp;
                }
            }
        }
    }
}

__device__ __forceinline__ bool k3d_rete_agenda_insert_top32(
    uint32_t* agenda,
    uint32_t& agenda_count,
    uint32_t activation
) {
    if (k3d_rete_agenda_payload(activation) == 0u) {
        return false;
    }

    if (agenda_count < 32u) {
        agenda[agenda_count++] = activation;
        for (uint32_t i = agenda_count; i < 32u; ++i) {
            agenda[i] = 0u;
        }
        k3d_rete_bitonic_sort32_desc(agenda);
        return true;
    }

    k3d_rete_bitonic_sort32_desc(agenda);
    if (!k3d_rete_agenda_precedes(activation, agenda[31])) {
        return false;
    }
    agenda[31] = activation;
    k3d_rete_bitonic_sort32_desc(agenda);
    return true;
}

__device__ __forceinline__ bool op_rete_alpha_test(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t active_context
) {
    float alpha_scalar = 0.0f;
    float fact_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, alpha_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, fact_scalar, error)) return false;

    const uint32_t alpha = static_cast<uint32_t>(max(0.0f, floorf(alpha_scalar + 0.5f)));
    const uint32_t fact = static_cast<uint32_t>(max(0.0f, floorf(fact_scalar + 0.5f)));

    const uint32_t required_mask = k3d_rete_fact_predicate_mask(alpha);
    const uint32_t required_context = k3d_rete_fact_context(alpha);
    const uint32_t required_cluster = k3d_rete_fact_cluster(alpha);
    const uint32_t policy = k3d_rete_fact_ethical(alpha);
    const uint32_t heuristic_bucket = k3d_rete_alpha_heuristic(alpha);

    const uint32_t fact_mask = k3d_rete_fact_predicate_mask(fact);
    const uint32_t fact_context = k3d_rete_fact_context(fact);
    const uint32_t fact_cluster = k3d_rete_fact_cluster(fact);
    const uint32_t fact_ethical = k3d_rete_fact_ethical(fact);
    const uint32_t effective_context = required_context != 0u ? required_context : active_context;
    const bool context_ok = rpn_context_allows_star(effective_context, fact_context);
    const bool cluster_ok = (required_cluster == 0u || fact_cluster == required_cluster);
    const bool ethical_ok = k3d_rete_policy_allows(fact_ethical, policy);
    const uint32_t matched_bits = __popc(fact_mask & required_mask);
    uint32_t heuristic_score = matched_bits * 24u;
    heuristic_score += context_ok ? 24u : 0u;
    heuristic_score += cluster_ok ? 16u : 0u;
    heuristic_score += (fact_ethical == 1u) ? 12u : ((fact_ethical == 2u) ? 6u : 0u);
    const uint32_t heuristic_floor = heuristic_bucket * 24u;
    const bool heuristic_ok = (heuristic_bucket == 0u) || (heuristic_score >= heuristic_floor);

    const bool matched = (
        required_mask != 0u &&
        (fact_mask & required_mask) == required_mask &&
        context_ok &&
        cluster_ok &&
        ethical_ok &&
        heuristic_ok
    );

    uint32_t token = 0u;
    if (matched) {
        token = k3d_rete_pack_token(
            fact_mask,
            fact_context,
            fact_context & ((1u << K3D_RETE_TOKEN_CONTEXT_BITS) - 1u),
            fact_cluster
        );
    }
    push(stack, stack_size, make_scalar(static_cast<float>(token)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_rete_beta_join(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float right_scalar = 0.0f;
    float left_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, right_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, left_scalar, error)) return false;

    const uint32_t right = static_cast<uint32_t>(max(0.0f, floorf(right_scalar + 0.5f)));
    const uint32_t left = static_cast<uint32_t>(max(0.0f, floorf(left_scalar + 0.5f)));

    const bool compatible = (
        k3d_rete_token_join_key(left) != 0u &&
        k3d_rete_token_join_key(left) == k3d_rete_token_join_key(right) &&
        k3d_rete_token_context(left) == k3d_rete_token_context(right) &&
        k3d_rete_token_cluster(left) == k3d_rete_token_cluster(right)
    );

    uint32_t joined = 0u;
    if (compatible) {
        joined = k3d_rete_pack_token(
            k3d_rete_token_binding_mask(left) | k3d_rete_token_binding_mask(right),
            k3d_rete_token_join_key(left),
            k3d_rete_token_context(left),
            k3d_rete_token_cluster(left)
        );
    }
    push(stack, stack_size, make_scalar(static_cast<float>(joined)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_agenda_insert(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float activation_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, activation_scalar, error)) return false;
    const uint32_t activation = static_cast<uint32_t>(max(0.0f, floorf(activation_scalar + 0.5f)));

    const uint32_t priority = k3d_rete_agenda_priority(activation);
    const uint32_t payload = k3d_rete_agenda_payload(activation);
    const uint32_t depth = k3d_rete_agenda_depth(activation);

    uint32_t agenda = 0u;
    if (payload != 0u && depth < 32u) {
        agenda = k3d_rete_pack_agenda(priority, payload, depth + 1u);
    }
    push(stack, stack_size, make_scalar(static_cast<float>(agenda)), error);
    return error == kErrorNone;
}
