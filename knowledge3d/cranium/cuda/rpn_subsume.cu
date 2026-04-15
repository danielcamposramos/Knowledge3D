#pragma once

#include <stdint.h>

__device__ __forceinline__ bool op_tsubsume(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float existing_scalar = 0.0f;
    float candidate_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, existing_scalar, error)) {
        return false;
    }
    if (!pop_scalar(stack, stack_size, candidate_scalar, error)) {
        return false;
    }
    const int32_t existing_literal = k3d_round_i32(existing_scalar);
    const int32_t candidate_literal = k3d_round_i32(candidate_scalar);
    const uint32_t existing_term = k3d_abs_term_u32(existing_literal);
    const uint32_t candidate_term = k3d_abs_term_u32(candidate_literal);
    const bool same_polarity = ((existing_literal < 0) == (candidate_literal < 0)) &&
        existing_literal != 0 &&
        candidate_literal != 0;
    const uint32_t subsumes = (same_polarity &&
        robinson_unify_scalar_terms(existing_term, candidate_term) != K3D_SUBST_FAIL)
        ? 1u
        : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(subsumes)), error);
    return error == kErrorNone;
}
