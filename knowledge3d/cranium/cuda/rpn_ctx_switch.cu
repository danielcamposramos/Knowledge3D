#pragma once

#include <stdint.h>

__device__ __forceinline__ bool op_ctx_switch(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t& active_context
) {
    float ctx_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, ctx_scalar, error)) {
        return false;
    }
    active_context = static_cast<uint32_t>(ctx_scalar < 0.0f ? 0.0f : floorf(ctx_scalar + 0.5f));
    return true;
}

__device__ __forceinline__ bool rpn_context_allows_star(
    uint32_t active_context,
    uint32_t star_context
) {
    return active_context == 0u || star_context == 0u || star_context == active_context;
}
