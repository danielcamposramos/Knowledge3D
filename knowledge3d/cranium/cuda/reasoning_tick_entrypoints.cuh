#pragma once

#include <math.h>
#include <stdint.h>

#include "reasoning_tick_io.cuh"

static __device__ __forceinline__ uint32_t k3d_reasoning_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_atlas_u32(
    const uint8_t* __restrict__ galaxy_atlas,
    uint32_t index
) {
    if (galaxy_atlas == nullptr) {
        return 0u;
    }
    return reinterpret_cast<const uint32_t*>(galaxy_atlas)[index];
}

static __device__ __forceinline__ int32_t k3d_reasoning_atlas_i32(
    const uint8_t* __restrict__ galaxy_atlas,
    uint32_t index
) {
    return static_cast<int32_t>(k3d_reasoning_atlas_u32(galaxy_atlas, index));
}

static __device__ __forceinline__ uint32_t k3d_reasoning_halt_after(const ReasoningTickIO& io) {
    const uint32_t halt_after = k3d_reasoning_atlas_u32(io.galaxy_atlas, 15u);
    return halt_after == 0u ? 1u : halt_after;
}

static __device__ __forceinline__ bool k3d_reasoning_context_allows(
    uint32_t active_context,
    uint32_t star_context
) {
    return active_context == 0u || star_context == 0u || star_context == active_context;
}

static __device__ __forceinline__ void k3d_reasoning_zero_lane_output(ReasoningLaneOutput* out) {
    out->halt_flag = 1u;
    out->result_handle = 0u;
    out->belief_q15 = 0u;
    out->_pad0 = 0u;
    #pragma unroll
    for (uint32_t i = 0u; i < K3D_REASONING_LANE_OUTPUT_BYTES - 16u; ++i) {
        out->payload[i] = 0u;
    }
}

static __device__ __forceinline__ void k3d_reasoning_store_payload_u32(
    ReasoningLaneOutput* out,
    uint32_t slot,
    uint32_t value
) {
    if (slot >= (K3D_REASONING_LANE_OUTPUT_BYTES - 16u) / sizeof(uint32_t)) {
        return;
    }
    reinterpret_cast<uint32_t*>(out->payload)[slot] = value;
}

static __device__ __forceinline__ void k3d_reasoning_set_halt(
    const ReasoningTickIO& io,
    ReasoningLaneOutput* out
) {
    const uint32_t halt_after = k3d_reasoning_halt_after(io);
    out->halt_flag = ((io.tick_seed + 1u) >= halt_after) ? 1u : 0u;
    k3d_reasoning_store_payload_u32(out, 2u, io.tick_seed + 1u);
    k3d_reasoning_store_payload_u32(out, 3u, halt_after);
}

static __device__ __forceinline__ void k3d_reasoning_set_pending(
    const ReasoningTickIO& io,
    ReasoningLaneOutput* out
) {
    out->halt_flag = 0u;
    k3d_reasoning_store_payload_u32(out, 2u, io.tick_seed + 1u);
    k3d_reasoning_store_payload_u32(out, 3u, k3d_reasoning_halt_after(io));
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_id(uint32_t handle) {
    return handle & 0x3fu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_anchor(uint32_t handle) {
    return (handle >> 6) & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_context(uint32_t handle) {
    return (handle >> 14) & 0x3fu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_ethical(uint32_t handle) {
    return (handle >> 20) & 0x3u;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_flags(uint32_t handle) {
    return (handle >> 22) & 0x3u;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_case(
    uint32_t case_id,
    uint32_t anchor,
    uint32_t context_id,
    uint32_t ethical_code,
    uint32_t flags
) {
    return (case_id & 0x3fu) |
        ((anchor & 0xffu) << 6) |
        ((context_id & 0x3fu) << 14) |
        ((ethical_code & 0x3u) << 20) |
        ((flags & 0x3u) << 22);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_case_similarity(uint32_t query_anchor, uint32_t candidate_anchor) {
    const uint32_t delta = query_anchor > candidate_anchor
        ? (query_anchor - candidate_anchor)
        : (candidate_anchor - query_anchor);
    return 255u - k3d_reasoning_min_u32(delta, 255u);
}

static __device__ __forceinline__ bool k3d_reasoning_case_policy_allows(uint32_t ethical_code, uint32_t policy) {
    switch (policy & 0x3u) {
        case 1u:
            return ethical_code == 1u;
        case 2u:
            return ethical_code == 1u || ethical_code == 2u;
        default:
            return ethical_code != 0u;
    }
}

static __device__ __forceinline__ bool k3d_reasoning_case_stage3_gate(uint32_t ethical_code, uint32_t conflict_code) {
    if (ethical_code == 0u) {
        return false;
    }
    if (ethical_code == 1u) {
        return true;
    }
    return (conflict_code & 0x3u) == 2u;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_select_best_case(
    const ReasoningTickIO& io,
    uint32_t query,
    uint32_t candidate_a,
    uint32_t candidate_b
) {
    const uint32_t query_anchor = k3d_reasoning_case_anchor(query);
    const uint32_t query_context = k3d_reasoning_case_context(query);
    const uint32_t candidates[2] = {candidate_a, candidate_b};

    uint32_t best = 0u;
    uint32_t best_score = 0u;
    #pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t candidate = candidates[i];
        if (k3d_reasoning_case_id(candidate) == 0u) {
            continue;
        }
        const uint32_t ethical = k3d_reasoning_case_ethical(candidate);
        if (ethical == 0u || io.ethical_trit < 0) {
            continue;
        }
        if (!k3d_reasoning_context_allows(io.context_id != 0u ? io.context_id : query_context, k3d_reasoning_case_context(candidate))) {
            continue;
        }
        uint32_t score = k3d_reasoning_case_similarity(query_anchor, k3d_reasoning_case_anchor(candidate));
        if (k3d_reasoning_case_context(candidate) == query_context && query_context != 0u) {
            score += 24u;
        }
        if (ethical == 1u) {
            score += 12u;
        } else if (ethical == 2u) {
            score += 6u;
        }
        if (score > best_score || (score == best_score && k3d_reasoning_case_id(candidate) < k3d_reasoning_case_id(best))) {
            best = candidate;
            best_score = score;
        }
    }
    return best;
}

static __device__ __forceinline__ void rpn_case_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t query = k3d_reasoning_atlas_u32(io.galaxy_atlas, 0u);
    const uint32_t candidate_a = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    const uint32_t candidate_b = k3d_reasoning_atlas_u32(io.galaxy_atlas, 2u);
    const uint32_t rebind = k3d_reasoning_atlas_u32(io.galaxy_atlas, 3u);
    const uint32_t constraint = k3d_reasoning_atlas_u32(io.galaxy_atlas, 4u);

    const uint32_t selected = k3d_reasoning_select_best_case(io, query, candidate_a, candidate_b);
    uint32_t rebound = selected;
    if (selected != 0u && rebind != 0u) {
        const uint32_t anchor = (k3d_reasoning_case_anchor(selected) ^ (rebind & 0xffu) ^ ((rebind >> 8) & 0xffu)) & 0xffu;
        const uint32_t context_id = ((rebind >> 16) & 0x3fu) != 0u
            ? ((rebind >> 16) & 0x3fu)
            : k3d_reasoning_case_context(selected);
        const uint32_t flags = (k3d_reasoning_case_flags(selected) ^ ((rebind >> 22) & 0x3u)) & 0x3u;
        rebound = k3d_reasoning_pack_case(
            k3d_reasoning_case_id(selected),
            anchor,
            context_id,
            k3d_reasoning_case_ethical(selected),
            flags
        );
    }

    uint32_t revised = rebound;
    if (rebound != 0u && constraint != 0u) {
        const uint32_t anchor_floor = constraint & 0xffu;
        const uint32_t revise_delta = (constraint >> 8) & 0x3fu;
        const uint32_t required_context = (constraint >> 14) & 0x3fu;
        const uint32_t policy = (constraint >> 20) & 0x3u;
        const uint32_t conflict_code = (constraint >> 22) & 0x3u;
        const uint32_t ethical = k3d_reasoning_case_ethical(rebound);
        const bool context_ok = k3d_reasoning_context_allows(
            required_context != 0u ? required_context : io.context_id,
            k3d_reasoning_case_context(rebound)
        );
        const bool policy_ok = k3d_reasoning_case_policy_allows(ethical, policy);
        const bool anchor_ok = k3d_reasoning_case_anchor(rebound) >= anchor_floor;
        const bool ethical_ok = k3d_reasoning_case_stage3_gate(ethical, conflict_code);
        if (context_ok && policy_ok && anchor_ok && ethical_ok) {
            revised = k3d_reasoning_pack_case(
                k3d_reasoning_case_id(rebound),
                k3d_reasoning_min_u32(255u, k3d_reasoning_case_anchor(rebound) + revise_delta),
                k3d_reasoning_case_context(rebound),
                ethical,
                k3d_reasoning_min_u32(3u, k3d_reasoning_case_flags(rebound) + 1u)
            );
        } else {
            revised = 0u;
        }
    }

    out->result_handle = revised;
    out->belief_q15 = revised != 0u ? 24576u : 0u;
    k3d_reasoning_store_payload_u32(out, 0u, selected);
    k3d_reasoning_store_payload_u32(out, 1u, rebound);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rule_lhs(uint32_t rule_handle) {
    return rule_handle & 0xfffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rule_rhs(uint32_t rule_handle) {
    return (rule_handle >> 12) & 0xfffu;
}

static __device__ __forceinline__ void rpn_superpos_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t rule = k3d_reasoning_atlas_u32(io.galaxy_atlas, 0u);
    const uint32_t target = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    const uint32_t lhs = k3d_reasoning_rule_lhs(rule);
    const uint32_t rhs = k3d_reasoning_rule_rhs(rule);
    uint32_t result = 0u;
    if (lhs != 0u && rhs != 0u && lhs != rhs && (target == lhs || target == rhs)) {
        const uint32_t replacement = target == lhs ? rhs : lhs;
        result = k3d_reasoning_mix32(rule ^ (target * 0x9e3779b9u) ^ (replacement * 0x85ebca6bu) ^ 0xc4c4c4c4u);
    }
    out->result_handle = result;
    out->belief_q15 = result != 0u ? 22938u : 0u;
    k3d_reasoning_store_payload_u32(out, 0u, lhs);
    k3d_reasoning_store_payload_u32(out, 1u, rhs);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_frame(
    uint32_t preserved_mask,
    uint32_t missing_mask,
    uint32_t status
) {
    return (preserved_mask & 0xffu) | ((missing_mask & 0xffu) << 8) | ((status & 0x3u) << 16);
}

static __device__ __forceinline__ void rpn_frame_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t pre_mask = k3d_reasoning_atlas_u32(io.galaxy_atlas, 0u) & 0xffu;
    const uint32_t post_mask = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u) & 0xffu;
    const uint32_t preserved_mask = pre_mask & post_mask;
    const uint32_t missing_mask = post_mask & ~pre_mask;
    const uint32_t incompatible_mask = pre_mask & ~post_mask;
    if (preserved_mask != 0u && incompatible_mask == 0u) {
        out->result_handle = k3d_reasoning_pack_frame(preserved_mask, missing_mask, 1u);
        out->belief_q15 = 21845u;
    }
    k3d_reasoning_store_payload_u32(out, 0u, preserved_mask);
    k3d_reasoning_store_payload_u32(out, 1u, missing_mask);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_opinion(
    uint32_t belief,
    uint32_t disbelief,
    uint32_t uncertainty,
    uint32_t status
) {
    return (belief & 0x7fu) |
        ((disbelief & 0x7fu) << 7) |
        ((uncertainty & 0x7fu) << 14) |
        ((status & 0x3u) << 21);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_clamp_opinion_field(int32_t value) {
    if (value < 0) {
        return 0u;
    }
    if (value > 127) {
        return 127u;
    }
    return static_cast<uint32_t>(value);
}

static __device__ __forceinline__ void rpn_ebelief_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t opinion = k3d_reasoning_atlas_u32(io.galaxy_atlas, 0u);
    const uint32_t supportive = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u) & 0x7fu;
    const uint32_t contradictory = k3d_reasoning_atlas_u32(io.galaxy_atlas, 2u) & 0x7fu;
    const uint32_t prior_belief = opinion & 0x7fu;
    const uint32_t prior_disbelief = (opinion >> 7) & 0x7fu;
    const uint32_t prior_uncertainty = (opinion >> 14) & 0x7fu;
    const uint32_t overlap = supportive < contradictory ? supportive : contradictory;
    const uint32_t consistent_support = supportive - overlap;
    const uint32_t consistent_contra = contradictory - overlap;

    const uint32_t belief = k3d_reasoning_clamp_opinion_field(
        static_cast<int32_t>(prior_belief) +
        static_cast<int32_t>(consistent_support) -
        static_cast<int32_t>(contradictory / 2u)
    );
    const uint32_t disbelief = k3d_reasoning_clamp_opinion_field(
        static_cast<int32_t>(prior_disbelief) +
        static_cast<int32_t>(contradictory) -
        static_cast<int32_t>(consistent_support / 4u)
    );
    int32_t uncertainty_value = static_cast<int32_t>(prior_uncertainty);
    uncertainty_value -= static_cast<int32_t>((consistent_support + consistent_contra) / 2u);
    uncertainty_value += static_cast<int32_t>(overlap / 2u);
    const uint32_t uncertainty = k3d_reasoning_clamp_opinion_field(uncertainty_value);

    uint32_t status = 0u;
    if (belief >= 96u && uncertainty <= 16u && contradictory == 0u) {
        status = 1u;
    } else if (contradictory != 0u || uncertainty >= 24u) {
        status = 2u;
    }
    out->result_handle = k3d_reasoning_pack_opinion(belief, disbelief, uncertainty, status);
    out->belief_q15 = k3d_reasoning_min_u32(32768u, belief * 256u);
    k3d_reasoning_store_payload_u32(out, 0u, supportive);
    k3d_reasoning_store_payload_u32(out, 1u, contradictory);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rete_fact_predicate(uint32_t fact) {
    return fact & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rete_fact_context(uint32_t fact) {
    return (fact >> 8) & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rete_fact_cluster(uint32_t fact) {
    return (fact >> 16) & 0x0fu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rete_fact_ethical(uint32_t fact) {
    return (fact >> 20) & 0x3u;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_rete_alpha_heuristic(uint32_t alpha) {
    return (alpha >> 22) & 0x3u;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_token(
    uint32_t binding_mask,
    uint32_t join_key,
    uint32_t context_id,
    uint32_t cluster_id
) {
    return (binding_mask & 0xffu) |
        ((join_key & 0xffu) << 8) |
        ((context_id & 0x0fu) << 16) |
        ((cluster_id & 0x0fu) << 20);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_agenda(
    uint32_t priority,
    uint32_t payload,
    uint32_t depth
) {
    return (priority & 0xffu) | ((payload & 0xffu) << 8) | ((depth & 0xffu) << 16);
}

static __device__ __forceinline__ bool k3d_reasoning_rete_policy_allows(uint32_t fact_ethical, uint32_t policy) {
    switch (policy & 0x3u) {
        case 1u:
            return fact_ethical == 1u;
        case 2u:
            return fact_ethical == 1u || fact_ethical == 2u;
        default:
            return fact_ethical != 0u;
    }
}

static __device__ __forceinline__ void rpn_rete_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t fact = k3d_reasoning_atlas_u32(io.galaxy_atlas, 0u);
    const uint32_t alpha = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    const uint32_t required_mask = k3d_reasoning_rete_fact_predicate(alpha);
    const uint32_t required_context = k3d_reasoning_rete_fact_context(alpha);
    const uint32_t required_cluster = k3d_reasoning_rete_fact_cluster(alpha);
    const uint32_t policy = k3d_reasoning_rete_fact_ethical(alpha);
    const uint32_t heuristic_bucket = k3d_reasoning_rete_alpha_heuristic(alpha);
    const uint32_t fact_mask = k3d_reasoning_rete_fact_predicate(fact);
    const uint32_t fact_context = k3d_reasoning_rete_fact_context(fact);
    const uint32_t fact_cluster = k3d_reasoning_rete_fact_cluster(fact);
    const uint32_t fact_ethical = k3d_reasoning_rete_fact_ethical(fact);
    const uint32_t effective_context = required_context != 0u ? required_context : io.context_id;
    const bool context_ok = k3d_reasoning_context_allows(effective_context, fact_context);
    const bool cluster_ok = required_cluster == 0u || fact_cluster == required_cluster;
    const bool ethical_ok = k3d_reasoning_rete_policy_allows(fact_ethical, policy);
    const uint32_t matched_bits = __popc(fact_mask & required_mask);
    uint32_t heuristic_score = matched_bits * 24u;
    heuristic_score += context_ok ? 24u : 0u;
    heuristic_score += cluster_ok ? 16u : 0u;
    heuristic_score += fact_ethical == 1u ? 12u : (fact_ethical == 2u ? 6u : 0u);
    const uint32_t heuristic_floor = heuristic_bucket * 24u;
    const bool heuristic_ok = heuristic_bucket == 0u || heuristic_score >= heuristic_floor;
    if (
        required_mask != 0u &&
        (fact_mask & required_mask) == required_mask &&
        context_ok &&
        cluster_ok &&
        ethical_ok &&
        heuristic_ok &&
        io.ethical_trit >= 0
    ) {
        const uint32_t token = k3d_reasoning_pack_token(fact_mask, fact_context, fact_context & 0x0fu, fact_cluster);
        out->result_handle = k3d_reasoning_pack_agenda(
            heuristic_score & 0xffu,
            (token >> 8) & 0xffu,
            1u
        );
        out->belief_q15 = k3d_reasoning_min_u32(32768u, heuristic_score * 256u);
        k3d_reasoning_store_payload_u32(out, 0u, token);
    }
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_branch(
    uint32_t node_id,
    uint32_t concept_mask
) {
    return (node_id & 0xffu) | ((concept_mask & 0xffffu) << 8);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_branch_node(uint32_t branch_handle) {
    return branch_handle & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_branch_mask(uint32_t branch_handle) {
    return (branch_handle >> 8) & 0xffffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_pack_ctx_view(
    uint32_t context_id,
    uint32_t include_global,
    int8_t ethical_trit
) {
    return (context_id & 0xffffu) |
        ((include_global & 0x1u) << 16) |
        ((static_cast<uint32_t>(static_cast<uint8_t>(ethical_trit))) << 24);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_alp_head(uint32_t horn_rule) {
    return horn_rule & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_alp_body(uint32_t horn_rule) {
    return (horn_rule >> 8) & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_alp_ic(uint32_t horn_rule) {
    return (horn_rule >> 16) & 0xffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_clause_positive_mask(uint32_t clause) {
    return clause & 0xffffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_clause_negative_mask(uint32_t clause) {
    return (clause >> 16) & 0xffffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_trail_true_mask(uint32_t trail) {
    return trail & 0xffffu;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_trail_false_mask(uint32_t trail) {
    return (trail >> 16) & 0xffffu;
}

static __device__ __forceinline__ bool k3d_reasoning_clause_satisfied(uint32_t clause, uint32_t trail) {
    return ((k3d_reasoning_clause_positive_mask(clause) & k3d_reasoning_trail_true_mask(trail)) != 0u) ||
        ((k3d_reasoning_clause_negative_mask(clause) & k3d_reasoning_trail_false_mask(trail)) != 0u);
}

static __device__ __forceinline__ bool k3d_reasoning_clause_conflict(uint32_t clause, uint32_t trail) {
    if (k3d_reasoning_clause_satisfied(clause, trail)) {
        return false;
    }
    const uint32_t positive = k3d_reasoning_clause_positive_mask(clause);
    const uint32_t negative = k3d_reasoning_clause_negative_mask(clause);
    const bool all_positive_falsified = positive == 0u || (positive & ~k3d_reasoning_trail_false_mask(trail)) == 0u;
    const bool all_negative_falsified = negative == 0u || (negative & ~k3d_reasoning_trail_true_mask(trail)) == 0u;
    return all_positive_falsified && all_negative_falsified;
}

static __device__ __forceinline__ uint32_t k3d_reasoning_abs_i32(int32_t value) {
    return static_cast<uint32_t>(value < 0 ? -value : value);
}

static __device__ __forceinline__ uint32_t k3d_reasoning_unify_terms(uint32_t a, uint32_t b) {
    if (a == b) {
        return k3d_reasoning_mix32(a ^ (b * 0x9e3779b9u));
    }
    const bool a_var = (a & 0x80000000u) != 0u;
    const bool b_var = (b & 0x80000000u) != 0u;
    if (a_var || b_var) {
        return k3d_reasoning_mix32(a ^ (b * 0x85ebca6bu) ^ 0xc0ffeeu);
    }
    return 0u;
}

static __device__ __forceinline__ void rpn_tableaux_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    if (io.ethical_trit < 0) {
        out->halt_flag = 1u;
        return;
    }
    const uint32_t branch = io.query_handle;
    const uint32_t expand_mask = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u) & 0xffffu;
    const int32_t lit_a = k3d_reasoning_atlas_i32(io.galaxy_atlas, 2u);
    const int32_t lit_b = k3d_reasoning_atlas_i32(io.galaxy_atlas, 3u);
    const uint32_t branch_context = k3d_reasoning_atlas_u32(io.galaxy_atlas, 5u);
    if (!k3d_reasoning_context_allows(io.context_id, branch_context)) {
        out->halt_flag = 1u;
        return;
    }

    const uint32_t node_id = k3d_reasoning_branch_node(branch);
    const uint32_t concept_mask = k3d_reasoning_branch_mask(branch);
    const uint32_t split_a = k3d_reasoning_mix32(branch ^ static_cast<uint32_t>(lit_a) ^ 0xd0d0d0d0u);
    const uint32_t split_b = k3d_reasoning_mix32(branch ^ static_cast<uint32_t>(lit_b) ^ 0xd1d1d1d1u);
    const bool clash = node_id != 0u && lit_a != 0 && lit_a == -lit_b;
    const uint32_t saturated = node_id != 0u ? k3d_reasoning_pack_branch(node_id, concept_mask | expand_mask) : 0u;

    out->result_handle = clash ? k3d_reasoning_mix32(branch ^ 0xd2d2d2d2u) : saturated;
    out->belief_q15 = clash ? 32768u : (saturated != 0u ? 16384u : 0u);
    k3d_reasoning_store_payload_u32(out, 0u, split_a);
    k3d_reasoning_store_payload_u32(out, 1u, split_b);
    if (clash || saturated != 0u) {
        k3d_reasoning_set_halt(io, out);
    } else {
        k3d_reasoning_set_pending(io, out);
    }
}

static __device__ __forceinline__ void rpn_unify_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t a = io.query_handle;
    const uint32_t b = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    out->result_handle = k3d_reasoning_unify_terms(a, b);
    out->belief_q15 = out->result_handle != 0u ? 32768u : 0u;
    k3d_reasoning_store_payload_u32(out, 0u, a);
    k3d_reasoning_store_payload_u32(out, 1u, b);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ void rpn_resolve_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const int32_t left = static_cast<int32_t>(io.query_handle);
    const int32_t right = k3d_reasoning_atlas_i32(io.galaxy_atlas, 1u);
    const uint32_t subst = k3d_reasoning_unify_terms(k3d_reasoning_abs_i32(left), k3d_reasoning_abs_i32(right));
    const bool complementary = left != 0 && right != 0 && left == -right && subst != 0u;
    out->result_handle = complementary ? 0xffffffffu : 0u;
    out->belief_q15 = complementary ? 32768u : 0u;
    k3d_reasoning_store_payload_u32(out, 0u, subst);
    k3d_reasoning_store_payload_u32(out, 1u, static_cast<uint32_t>(right));
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ void rpn_subsume_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const int32_t candidate = static_cast<int32_t>(io.query_handle);
    const int32_t existing = k3d_reasoning_atlas_i32(io.galaxy_atlas, 1u);
    const bool same_polarity = candidate != 0 && existing != 0 && ((candidate < 0) == (existing < 0));
    const uint32_t subst = k3d_reasoning_unify_terms(k3d_reasoning_abs_i32(candidate), k3d_reasoning_abs_i32(existing));
    out->result_handle = (same_polarity && subst != 0u) ? 1u : 0u;
    out->belief_q15 = out->result_handle != 0u ? 32768u : 0u;
    k3d_reasoning_store_payload_u32(out, 0u, subst);
    k3d_reasoning_set_halt(io, out);
}

static __device__ __forceinline__ void rpn_alp_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t goal_mask = io.query_handle & 0xffu;
    const uint32_t horn_rule = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    const uint32_t kb_support = k3d_reasoning_atlas_u32(io.galaxy_atlas, 2u) & 0xffu;
    const uint32_t head_symbol = k3d_reasoning_alp_head(horn_rule) & 0x1fu;
    const uint32_t head_bit = head_symbol == 0u ? 0u : (1u << head_symbol);
    const uint32_t residual = (goal_mask & head_bit) != 0u
        ? ((goal_mask & ~head_bit) | k3d_reasoning_alp_body(horn_rule))
        : goal_mask;
    const uint32_t assumptions = residual & ~kb_support;
    const bool integrity_fail = (assumptions & k3d_reasoning_alp_ic(horn_rule)) != 0u;

    if (integrity_fail) {
        out->result_handle = 0u;
        out->belief_q15 = 0u;
        k3d_reasoning_store_payload_u32(out, 0u, residual);
        k3d_reasoning_store_payload_u32(out, 1u, assumptions);
        out->halt_flag = 1u;
        k3d_reasoning_store_payload_u32(out, 2u, io.tick_seed + 1u);
        return;
    }

    out->result_handle = assumptions != 0u ? assumptions : residual;
    out->belief_q15 = out->result_handle != 0u ? 32768u : 16384u;
    k3d_reasoning_store_payload_u32(out, 0u, residual);
    k3d_reasoning_store_payload_u32(out, 1u, assumptions);
    if (out->result_handle != 0u || residual == 0u) {
        k3d_reasoning_set_halt(io, out);
    } else {
        k3d_reasoning_set_pending(io, out);
    }
}

static __device__ __forceinline__ void rpn_dpll_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t clause = io.query_handle;
    const uint32_t trail = k3d_reasoning_atlas_u32(io.galaxy_atlas, 1u);
    const uint32_t aux_clause = k3d_reasoning_atlas_u32(io.galaxy_atlas, 2u);
    const uint32_t model_hint = k3d_reasoning_atlas_u32(io.galaxy_atlas, 3u);
    const bool conflict = k3d_reasoning_clause_conflict(clause, trail);
    const bool aux_conflict = aux_clause != 0u && k3d_reasoning_clause_conflict(aux_clause, trail);
    const bool satisfied = k3d_reasoning_clause_satisfied(clause, trail) &&
        (aux_clause == 0u || k3d_reasoning_clause_satisfied(aux_clause, trail) || aux_conflict);
    const uint32_t learnt = aux_conflict ? (k3d_reasoning_trail_true_mask(trail) << 16) : 0u;

    k3d_reasoning_store_payload_u32(out, 0u, learnt);
    k3d_reasoning_store_payload_u32(out, 1u, trail);

    if (conflict) {
        out->result_handle = 0u;
        out->belief_q15 = 0u;
        out->halt_flag = 1u;
        k3d_reasoning_store_payload_u32(out, 2u, io.tick_seed + 1u);
        return;
    }
    if (satisfied || learnt != 0u) {
        out->result_handle = model_hint != 0u
            ? model_hint
            : k3d_reasoning_mix32(clause ^ trail ^ aux_clause ^ 0xd4d4d4d4u);
        out->belief_q15 = 32768u;
        k3d_reasoning_set_halt(io, out);
        return;
    }
    k3d_reasoning_set_pending(io, out);
}

static __device__ __forceinline__ void rpn_ctx_switch_tick(const ReasoningTickIO& io, ReasoningLaneOutput* out) {
    k3d_reasoning_zero_lane_output(out);
    const uint32_t include_global = 1u;
    out->result_handle = k3d_reasoning_pack_ctx_view(io.context_id, include_global, io.ethical_trit);
    out->belief_q15 = io.context_id == 0u ? 16384u : 32768u;
    k3d_reasoning_store_payload_u32(out, 0u, io.context_id);
    k3d_reasoning_store_payload_u32(out, 1u, include_global);
    out->halt_flag = 1u;
    k3d_reasoning_store_payload_u32(out, 2u, io.tick_seed + 1u);
    k3d_reasoning_store_payload_u32(out, 3u, 1u);
}
