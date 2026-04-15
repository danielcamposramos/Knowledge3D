#pragma once

#include <stdint.h>

constexpr uint32_t K3D_OPINION_FIELD_BITS = 7u;
constexpr uint32_t K3D_OPINION_FIELD_MASK = (1u << K3D_OPINION_FIELD_BITS) - 1u;
constexpr uint32_t K3D_OPINION_DISBELIEF_SHIFT = K3D_OPINION_FIELD_BITS;
constexpr uint32_t K3D_OPINION_UNCERTAINTY_SHIFT = K3D_OPINION_FIELD_BITS * 2u;
constexpr uint32_t K3D_OPINION_STATUS_SHIFT = K3D_OPINION_FIELD_BITS * 3u;

enum K3DOpinionStatus : uint32_t {
    K3D_OPINION_UNRESOLVED = 0u,
    K3D_OPINION_OK = 1u,
    K3D_OPINION_DEFEASIBLE = 2u,
};

__device__ __forceinline__ uint32_t k3d_pack_opinion_handle(
    uint32_t belief,
    uint32_t disbelief,
    uint32_t uncertainty,
    uint32_t status
) {
    return (belief & K3D_OPINION_FIELD_MASK) |
        ((disbelief & K3D_OPINION_FIELD_MASK) << K3D_OPINION_DISBELIEF_SHIFT) |
        ((uncertainty & K3D_OPINION_FIELD_MASK) << K3D_OPINION_UNCERTAINTY_SHIFT) |
        ((status & 0x3u) << K3D_OPINION_STATUS_SHIFT);
}

__device__ __forceinline__ uint32_t k3d_unpack_opinion_belief(uint32_t opinion_handle) {
    return opinion_handle & K3D_OPINION_FIELD_MASK;
}

__device__ __forceinline__ uint32_t k3d_unpack_opinion_disbelief(uint32_t opinion_handle) {
    return (opinion_handle >> K3D_OPINION_DISBELIEF_SHIFT) & K3D_OPINION_FIELD_MASK;
}

__device__ __forceinline__ uint32_t k3d_unpack_opinion_uncertainty(uint32_t opinion_handle) {
    return (opinion_handle >> K3D_OPINION_UNCERTAINTY_SHIFT) & K3D_OPINION_FIELD_MASK;
}

__device__ __forceinline__ uint32_t k3d_unpack_opinion_status(uint32_t opinion_handle) {
    return (opinion_handle >> K3D_OPINION_STATUS_SHIFT) & 0x3u;
}

__device__ __forceinline__ uint32_t k3d_clamp_opinion_field(int32_t value) {
    if (value < 0) return 0u;
    if (value > static_cast<int32_t>(K3D_OPINION_FIELD_MASK)) return K3D_OPINION_FIELD_MASK;
    return static_cast<uint32_t>(value);
}

__device__ __forceinline__ bool op_ebelief(
    StackValue* stack,
    uint32_t& stack_size,
    uint32_t& error
) {
    float contradictory_scalar = 0.0f;
    float supportive_scalar = 0.0f;
    float opinion_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, contradictory_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, supportive_scalar, error)) return false;
    if (!pop_scalar(stack, stack_size, opinion_scalar, error)) return false;

    const uint32_t opinion = static_cast<uint32_t>(max(0.0f, floorf(opinion_scalar + 0.5f)));
    const uint32_t prior_belief = k3d_unpack_opinion_belief(opinion);
    const uint32_t prior_disbelief = k3d_unpack_opinion_disbelief(opinion);
    const uint32_t prior_uncertainty = k3d_unpack_opinion_uncertainty(opinion);
    const uint32_t supportive = static_cast<uint32_t>(max(0.0f, floorf(supportive_scalar + 0.5f))) & K3D_OPINION_FIELD_MASK;
    const uint32_t contradictory = static_cast<uint32_t>(max(0.0f, floorf(contradictory_scalar + 0.5f))) & K3D_OPINION_FIELD_MASK;

    const uint32_t contradiction_overlap = supportive < contradictory ? supportive : contradictory;
    const uint32_t consistent_support = supportive - contradiction_overlap;
    const uint32_t consistent_contra = contradictory - contradiction_overlap;

    const uint32_t belief = k3d_clamp_opinion_field(
        static_cast<int32_t>(prior_belief) +
        static_cast<int32_t>(consistent_support) -
        static_cast<int32_t>(contradictory / 2u)
    );
    const uint32_t disbelief = k3d_clamp_opinion_field(
        static_cast<int32_t>(prior_disbelief) +
        static_cast<int32_t>(contradictory) -
        static_cast<int32_t>(consistent_support / 4u)
    );

    int32_t uncertainty_value = static_cast<int32_t>(prior_uncertainty);
    uncertainty_value -= static_cast<int32_t>((consistent_support + consistent_contra) / 2u);
    uncertainty_value += static_cast<int32_t>(contradiction_overlap / 2u);
    const uint32_t uncertainty = k3d_clamp_opinion_field(uncertainty_value);

    uint32_t status = K3D_OPINION_UNRESOLVED;
    if (belief >= 96u && uncertainty <= 16u && contradictory == 0u) {
        status = K3D_OPINION_OK;
    } else if (contradictory != 0u || uncertainty >= 24u) {
        status = K3D_OPINION_DEFEASIBLE;
    }

    push(
        stack,
        stack_size,
        make_scalar(static_cast<float>(k3d_pack_opinion_handle(belief, disbelief, uncertainty, status))),
        error
    );
    return error == kErrorNone;
}
