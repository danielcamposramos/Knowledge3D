#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t abduce_empty_hypothesis_query(uint32_t observation) {
    (void)observation;
    return 0u;
}

__device__ __forceinline__ uint32_t explain_empty_hypothesis(uint32_t hypothesis, uint32_t observation) {
    (void)hypothesis;
    (void)observation;
    return 0u;
}

__device__ __forceinline__ uint32_t suspect_simplicity_score(uint32_t hypothesis) {
    if (hypothesis == 0u) {
        return 0u;
    }
    uint32_t x = hypothesis;
    uint32_t bits = 0u;
    while (x != 0u) {
        bits += x & 1u;
        x >>= 1u;
    }
    return max(1u, 32u - bits);
}

__device__ __forceinline__ bool op_abduce(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float obs_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, obs_scalar, error)) return false;
    const uint32_t obs = static_cast<uint32_t>(max(0.0f, floorf(obs_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(abduce_empty_hypothesis_query(obs))), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_explain(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float obs_scalar = 0.0f;
    float hyp_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, obs_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, hyp_scalar, error)) return false;
    const uint32_t obs = static_cast<uint32_t>(max(0.0f, floorf(obs_scalar + 0.5f)));
    const uint32_t hyp = static_cast<uint32_t>(max(0.0f, floorf(hyp_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(explain_empty_hypothesis(hyp, obs))), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_suspect(StackValue* stack, uint32_t& stack_size, uint32_t& error) {
    float hyp_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, hyp_scalar, error)) return false;
    const uint32_t hyp = static_cast<uint32_t>(max(0.0f, floorf(hyp_scalar + 0.5f)));
    push(stack, stack_size, make_scalar(static_cast<float>(suspect_simplicity_score(hyp))), error);
    return error == kErrorNone;
}
