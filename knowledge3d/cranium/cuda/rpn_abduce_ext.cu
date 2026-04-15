#pragma once

#include <stdint.h>

static constexpr uint32_t K3D_ABDUCE_HALT_THRESHOLD = 16u;

__device__ __forceinline__ uint32_t abduce_assumption_handle(uint32_t goal_clause, uint32_t kb_clause) {
    return mix32(goal_clause ^ (kb_clause * 0x9E3779B9u) ^ 0xA6A6A6A6u);
}

__device__ __forceinline__ bool op_abduce_halt(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t& halt_requested
) {
    float simplicity_scalar = 0.0f;
    float hyp_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, simplicity_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, hyp_scalar, error)) return false;
    const uint32_t simplicity = static_cast<uint32_t>(max(0.0f, floorf(simplicity_scalar + 0.5f)));
    const uint32_t hypothesis = static_cast<uint32_t>(max(0.0f, floorf(hyp_scalar + 0.5f)));
    const uint32_t halted = (hypothesis != 0u && simplicity >= K3D_ABDUCE_HALT_THRESHOLD) ? 1u : 0u;
    if (halted != 0u) {
        halt_requested = 1u;
    }
    push(stack, stack_size, make_scalar(static_cast<float>(halted)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_scunion(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float mask_b_scalar = 0.0f;
    float mask_a_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, mask_b_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, mask_a_scalar, error)) return false;
    const uint32_t mask_a = static_cast<uint32_t>(max(0.0f, floorf(mask_a_scalar + 0.5f)));
    const uint32_t mask_b = static_cast<uint32_t>(max(0.0f, floorf(mask_b_scalar + 0.5f)));
    const uint32_t union_mask = mask_a | mask_b;
    (void)__popc(union_mask);
    push(stack, stack_size, make_scalar(static_cast<float>(union_mask)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_icheck(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float constraint_scalar = 0.0f;
    float hyp_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, constraint_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, hyp_scalar, error)) return false;
    const uint32_t constraint = static_cast<uint32_t>(max(0.0f, floorf(constraint_scalar + 0.5f)));
    const uint32_t hypothesis = static_cast<uint32_t>(max(0.0f, floorf(hyp_scalar + 0.5f)));
    const uint32_t horn_ic = k3d_alp_ic_mask(constraint);
    uint32_t ok = 0u;
    if (horn_ic != 0u) {
        ok = ((hypothesis & horn_ic) == 0u) ? 1u : 0u;
    } else {
        ok = (constraint != 0u && (hypothesis & constraint) == constraint) ? 1u : 0u;
    }
    push(stack, stack_size, make_scalar(static_cast<float>(ok)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_abdres(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float kb_scalar = 0.0f;
    float goal_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, kb_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, goal_scalar, error)) return false;
    const int32_t kb_literal = k3d_round_i32(kb_scalar);
    const int32_t goal_literal = k3d_round_i32(goal_scalar);
    const uint32_t kb_term = k3d_abs_term_u32(kb_literal);
    const uint32_t goal_term = k3d_abs_term_u32(goal_literal);

    const uint32_t goal_mask = goal_term & 0xFFu;
    const uint32_t kb_mask = kb_term & 0xFFu;
    if ((goal_mask | kb_mask) != 0u) {
        const uint32_t resolved = goal_mask & kb_mask;
        const uint32_t assumptions_mask = goal_mask & ~kb_mask;
        uint32_t assumptions = 0u;
        if (assumptions_mask != 0u) {
            assumptions = assumptions_mask;
        } else if (resolved != 0u) {
            assumptions = resolved;
        } else if (goal_mask != 0u) {
            assumptions = abduce_assumption_handle(goal_mask, kb_mask);
        }
        push(stack, stack_size, make_scalar(static_cast<float>(assumptions)), error);
        return error == kErrorNone;
    }

    const uint32_t subst = robinson_unify_scalar_terms(goal_term, kb_term);
    uint32_t assumptions = 0u;
    if (k3d_literals_are_complements(goal_literal, kb_literal) && subst != K3D_SUBST_FAIL) {
        assumptions = k3d_resolvent_handle(subst, goal_term, kb_term);
    } else if (goal_term != 0u) {
        assumptions = abduce_assumption_handle(goal_term, kb_term);
    }
    push(stack, stack_size, make_scalar(static_cast<float>(assumptions)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_abdneg(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float budget_scalar = 0.0f;
    float goal_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, budget_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, goal_scalar, error)) return false;
    const uint32_t budget = static_cast<uint32_t>(max(0.0f, floorf(budget_scalar + 0.5f)));
    const uint32_t goal = k3d_abs_term_u32(k3d_round_i32(goal_scalar));
    const uint32_t complexity = k3d_popcount_u32(goal);
    const uint32_t finitely_failed = (goal != 0u && budget >= complexity) ? 1u : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(finitely_failed)), error);
    return error == kErrorNone;
}
