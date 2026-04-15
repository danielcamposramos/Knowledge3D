#pragma once

#include <stdint.h>

constexpr uint32_t K3D_FRAME_MASK_BITS = 8u;
constexpr uint32_t K3D_FRAME_MASK_FIELD = (1u << K3D_FRAME_MASK_BITS) - 1u;
constexpr uint32_t K3D_FRAME_PRESERVED_SHIFT = 0u;
constexpr uint32_t K3D_FRAME_MISSING_SHIFT = K3D_FRAME_MASK_BITS;
constexpr uint32_t K3D_FRAME_STATUS_SHIFT = K3D_FRAME_MASK_BITS * 2u;

enum K3DFrameStatus : uint32_t {
    K3D_FRAME_STATUS_EMPTY = 0u,
    K3D_FRAME_STATUS_OK = 1u,
    K3D_FRAME_STATUS_INCOMPATIBLE = 2u,
};

__device__ __forceinline__ uint32_t k3d_pack_frame_handle(
    uint32_t preserved_mask,
    uint32_t missing_mask,
    uint32_t status
) {
    return ((preserved_mask & K3D_FRAME_MASK_FIELD) << K3D_FRAME_PRESERVED_SHIFT) |
        ((missing_mask & K3D_FRAME_MASK_FIELD) << K3D_FRAME_MISSING_SHIFT) |
        ((status & 0x3u) << K3D_FRAME_STATUS_SHIFT);
}

__device__ __forceinline__ uint32_t k3d_frame_preserved_mask(uint32_t frame_handle) {
    return (frame_handle >> K3D_FRAME_PRESERVED_SHIFT) & K3D_FRAME_MASK_FIELD;
}

__device__ __forceinline__ uint32_t k3d_frame_missing_mask(uint32_t frame_handle) {
    return (frame_handle >> K3D_FRAME_MISSING_SHIFT) & K3D_FRAME_MASK_FIELD;
}

__device__ __forceinline__ bool op_frame(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float post_scalar = 0.0f;
    float pre_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, post_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, pre_scalar, error)) return false;

    const uint32_t post_mask = static_cast<uint32_t>(max(0.0f, floorf(post_scalar + 0.5f))) & K3D_FRAME_MASK_FIELD;
    const uint32_t pre_mask = static_cast<uint32_t>(max(0.0f, floorf(pre_scalar + 0.5f))) & K3D_FRAME_MASK_FIELD;
    const uint32_t preserved_mask = pre_mask & post_mask;
    const uint32_t missing_mask = 0u;
    const uint32_t status = (preserved_mask != 0u) ? K3D_FRAME_STATUS_OK : K3D_FRAME_STATUS_INCOMPATIBLE;
    const uint32_t frame_handle = (status == K3D_FRAME_STATUS_OK)
        ? k3d_pack_frame_handle(preserved_mask, missing_mask, status)
        : 0u;
    push(stack, stack_size, make_scalar(static_cast<float>(frame_handle)), error);
    return error == kErrorNone;
}

__device__ __forceinline__ bool op_biduce(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float post_scalar = 0.0f;
    float pre_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, post_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, pre_scalar, error)) return false;

    const uint32_t post_mask = static_cast<uint32_t>(max(0.0f, floorf(post_scalar + 0.5f))) & K3D_FRAME_MASK_FIELD;
    const uint32_t pre_mask = static_cast<uint32_t>(max(0.0f, floorf(pre_scalar + 0.5f))) & K3D_FRAME_MASK_FIELD;
    const uint32_t preserved_mask = pre_mask & post_mask;
    const uint32_t missing_mask = post_mask & ~pre_mask;
    const uint32_t incompatible_mask = pre_mask & ~post_mask;

    uint32_t frame_handle = 0u;
    if (preserved_mask != 0u && incompatible_mask == 0u) {
        frame_handle = k3d_pack_frame_handle(preserved_mask, missing_mask, K3D_FRAME_STATUS_OK);
    }
    push(stack, stack_size, make_scalar(static_cast<float>(frame_handle)), error);
    return error == kErrorNone;
}
