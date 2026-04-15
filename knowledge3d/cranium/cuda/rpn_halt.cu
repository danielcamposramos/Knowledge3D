#pragma once

#include <stdint.h>

__device__ __forceinline__ bool op_halt_set(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t& halt_requested
) {
    float halt_signal = 0.0f;
    if (!pop_scalar(stack, stack_size, halt_signal, error)) {
        return false;
    }
    if (halt_signal > 0.0f) {
        halt_requested = 1u;
    }
    return true;
}

__device__ __forceinline__ bool op_halt_sync(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error,
    uint32_t halt_requested,
    uint32_t& halt_sync_seen,
    uint32_t& halt_should_break
) {
    if (halt_sync_seen == 0u) {
        halt_sync_seen = 1u;
        if (halt_requested != 0u) {
            halt_should_break = 1u;
        }
    }
    return push(stack, stack_size, make_scalar(halt_requested != 0u ? 1.0f : 0.0f), error);
}
