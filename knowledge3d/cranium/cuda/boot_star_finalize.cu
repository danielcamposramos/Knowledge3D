#include <stdint.h>

#include "device_functions.cuh"

#define RAW_CATALOG_INPUT_ENTRY_BYTES        176
#define FINAL_CATALOG_INPUT_ENTRY_BYTES      152

#define FINAL_STAR_EMBEDDING16_OFFSET          0
#define FINAL_STAR_GALAXY_ID_OFFSET           64
#define FINAL_STAR_TYPE_OFFSET                68
#define FINAL_STAR_SELECTION_ROLE_OFFSET      72
#define FINAL_STAR_LAYER_ID_OFFSET            76
#define FINAL_STAR_FLAGS_OFFSET               80
#define FINAL_STAR_ANSWER_ELIGIBLE_OFFSET     84
#define FINAL_STAR_SEMANTIC_POLARITY_OFFSET   88
#define FINAL_STAR_SEMANTIC_FOCUS_OFFSET      92
#define FINAL_STAR_SEMANTIC_MASS_OFFSET       96
#define FINAL_STAR_ATTRACTIVE_PRIOR_OFFSET   100
#define FINAL_STAR_REPULSIVE_PRIOR_OFFSET    104
#define FINAL_STAR_ROUTE_POLICY_OFFSET       108
#define FINAL_STAR_HASH_OFFSET               112
#define FINAL_STAR_POSITION_OFFSET           120

#define RAW_EMBEDDING16_OFFSET                 0
#define RAW_GALAXY_ID_OFFSET                  64
#define RAW_TYPE_OFFSET                       68
#define RAW_SELECTION_ROLE_OFFSET             72
#define RAW_LAYER_ID_OFFSET                   76
#define RAW_FLAGS_OFFSET                      80
#define RAW_ANSWER_ELIGIBLE_OFFSET            84
#define RAW_SEMANTIC_POLARITY_OFFSET          88
#define RAW_SEMANTIC_FOCUS_OFFSET             92
#define RAW_SEMANTIC_MASS_OFFSET              96
#define RAW_ATTRACTIVE_PRIOR_OFFSET          100
#define RAW_REPULSIVE_PRIOR_OFFSET           104
#define RAW_CONFIDENCE_OFFSET                108
#define RAW_ROUTE_POLICY_FLAGS_OFFSET        112
#define RAW_BRANCH_TOPK_OFFSET               116
#define RAW_EXPLICIT_MASK_OFFSET             120
#define RAW_ROUTER_REF_HINT_COUNT_OFFSET     124
#define RAW_EXECUTOR_REF_HINT_COUNT_OFFSET   128
#define RAW_VALIDATOR_REF_HINT_COUNT_OFFSET  132
#define RAW_ANTI_REF_HINT_COUNT_OFFSET       136
#define RAW_STAR_HASH_OFFSET                 140
#define RAW_DOMAIN_HASH_OFFSET               148
#define RAW_SUBJECT_HASH_OFFSET              152

#define RAW_EXPLICIT_POLARITY_BIT   0x01u
#define RAW_EXPLICIT_FOCUS_BIT      0x02u
#define RAW_EXPLICIT_MASS_BIT       0x04u
#define RAW_EXPLICIT_ATTRACTIVE_BIT 0x08u
#define RAW_EXPLICIT_REPULSIVE_BIT  0x10u

static __device__ __forceinline__ float raw_f32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const float*>(ptr + offset);
}

static __device__ __forceinline__ unsigned int raw_u32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

static __device__ __forceinline__ int raw_i32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const int*>(ptr + offset);
}

static __device__ __forceinline__ unsigned long long raw_u64(const unsigned char* ptr, const unsigned int offset) {
    const unsigned long long low = static_cast<unsigned long long>(raw_u32(ptr, offset));
    const unsigned long long high = static_cast<unsigned long long>(raw_u32(ptr, offset + 4u));
    return low | (high << 32u);
}

static __device__ __forceinline__ void write_f32(unsigned char* ptr, const unsigned int offset, const float value) {
    *reinterpret_cast<float*>(ptr + offset) = value;
}

static __device__ __forceinline__ void write_u32(unsigned char* ptr, const unsigned int offset, const unsigned int value) {
    *reinterpret_cast<unsigned int*>(ptr + offset) = value;
}

static __device__ __forceinline__ void write_i32(unsigned char* ptr, const unsigned int offset, const int value) {
    *reinterpret_cast<int*>(ptr + offset) = value;
}

static __device__ __forceinline__ void write_u64(unsigned char* ptr, const unsigned int offset, const unsigned long long value) {
    *reinterpret_cast<unsigned long long*>(ptr + offset) = value;
}

extern "C" __global__ void boot_star_finalize(
    const unsigned char* __restrict__ raw_input,
    unsigned char* __restrict__ finalized_input,
    unsigned int entry_count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entry_count) {
        return;
    }

    const unsigned char* src = raw_input + (i * RAW_CATALOG_INPUT_ENTRY_BYTES);
    unsigned char* dst = finalized_input + (i * FINAL_CATALOG_INPUT_ENTRY_BYTES);

    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        write_f32(dst, FINAL_STAR_EMBEDDING16_OFFSET + (d * 4), raw_f32(src, RAW_EMBEDDING16_OFFSET + (d * 4)));
    }

    const unsigned int role_id = raw_u32(src, RAW_SELECTION_ROLE_OFFSET);
    const unsigned int layer_id = raw_u32(src, RAW_LAYER_ID_OFFSET);
    const unsigned int answer_eligible = raw_u32(src, RAW_ANSWER_ELIGIBLE_OFFSET);
    const unsigned int explicit_mask = raw_u32(src, RAW_EXPLICIT_MASK_OFFSET);
    const unsigned int executor_ref_hints = raw_u32(src, RAW_EXECUTOR_REF_HINT_COUNT_OFFSET);
    const unsigned int validator_ref_hints = raw_u32(src, RAW_VALIDATOR_REF_HINT_COUNT_OFFSET);
    const unsigned int anti_ref_hints = raw_u32(src, RAW_ANTI_REF_HINT_COUNT_OFFSET);
    const unsigned int route_policy_flags = raw_u32(src, RAW_ROUTE_POLICY_FLAGS_OFFSET);
    const unsigned int branch_topk = raw_u32(src, RAW_BRANCH_TOPK_OFFSET);

    const bool polarity_explicit = (explicit_mask & RAW_EXPLICIT_POLARITY_BIT) != 0u;
    const bool focus_explicit = (explicit_mask & RAW_EXPLICIT_FOCUS_BIT) != 0u;
    const bool mass_explicit = (explicit_mask & RAW_EXPLICIT_MASS_BIT) != 0u;
    const bool attractive_explicit = (explicit_mask & RAW_EXPLICIT_ATTRACTIVE_BIT) != 0u;
    const bool repulsive_explicit = (explicit_mask & RAW_EXPLICIT_REPULSIVE_BIT) != 0u;

    const float confidence = device_clamp_range(device_finite_or_default(raw_f32(src, RAW_CONFIDENCE_OFFSET), 0.0f), 0.0f, 1.0f);
    const int semantic_polarity = polarity_explicit
        ? static_cast<int>(clamp_trit_int_device(raw_i32(src, RAW_SEMANTIC_POLARITY_OFFSET)))
        : (
            role_id == GALAXY_ROLE_ANTI_PATTERN
                ? -1
                : (answer_eligible != 0u ? 1 : 0)
        );
    const float semantic_focus = focus_explicit
        ? device_clamp_min(device_finite_or_default(raw_f32(src, RAW_SEMANTIC_FOCUS_OFFSET), 0.0f), 0.0f)
        : (answer_eligible != 0u ? 1.0f : 0.5f);
    const float semantic_mass = mass_explicit
        ? device_clamp_min(device_finite_or_default(raw_f32(src, RAW_SEMANTIC_MASS_OFFSET), 0.0f), 0.05f)
        : device_maxf(0.05f, confidence);
    const float attractive_prior = attractive_explicit
        ? device_clamp_min(device_finite_or_default(raw_f32(src, RAW_ATTRACTIVE_PRIOR_OFFSET), 0.0f), 0.0f)
        : ((role_id == GALAXY_ROLE_ROUTER && executor_ref_hints > 0u) ? 0.25f : 0.0f);
    const float repulsive_prior = repulsive_explicit
        ? device_clamp_min(device_finite_or_default(raw_f32(src, RAW_REPULSIVE_PRIOR_OFFSET), 0.0f), 0.0f)
        : (((role_id == GALAXY_ROLE_ANTI_PATTERN) || anti_ref_hints > 0u) ? 0.30f : 0.0f);
    const float position_z = device_clamp_range(static_cast<float>(layer_id) / 4.0f, 0.0f, 1.0f);

    write_u32(dst, FINAL_STAR_GALAXY_ID_OFFSET, raw_u32(src, RAW_GALAXY_ID_OFFSET));
    write_u32(dst, FINAL_STAR_TYPE_OFFSET, raw_u32(src, RAW_TYPE_OFFSET));
    write_u32(dst, FINAL_STAR_SELECTION_ROLE_OFFSET, role_id);
    write_u32(dst, FINAL_STAR_LAYER_ID_OFFSET, layer_id);
    write_u32(dst, FINAL_STAR_FLAGS_OFFSET, raw_u32(src, RAW_FLAGS_OFFSET));
    write_u32(dst, FINAL_STAR_ANSWER_ELIGIBLE_OFFSET, answer_eligible);
    write_i32(dst, FINAL_STAR_SEMANTIC_POLARITY_OFFSET, semantic_polarity);
    write_f32(dst, FINAL_STAR_SEMANTIC_FOCUS_OFFSET, semantic_focus);
    write_f32(dst, FINAL_STAR_SEMANTIC_MASS_OFFSET, semantic_mass);
    write_f32(dst, FINAL_STAR_ATTRACTIVE_PRIOR_OFFSET, attractive_prior);
    write_f32(dst, FINAL_STAR_REPULSIVE_PRIOR_OFFSET, repulsive_prior);
    write_u32(
        dst,
        FINAL_STAR_ROUTE_POLICY_OFFSET,
        pack_route_policy_device(
            (route_policy_flags & ROUTE_POLICY_DECOMPOSE_ON_FAIL) != 0u,
            (route_policy_flags & ROUTE_POLICY_REQUIRES_EXECUTOR) != 0u,
            (route_policy_flags & ROUTE_POLICY_REQUIRES_VALIDATOR) != 0u,
            (route_policy_flags & ROUTE_POLICY_ANSWER_GATE) != 0u,
            branch_topk
        )
    );
    write_u64(dst, FINAL_STAR_HASH_OFFSET, raw_u64(src, RAW_STAR_HASH_OFFSET));
    write_f32(dst, FINAL_STAR_POSITION_OFFSET + 0u, device_finite_or_default(raw_f32(src, RAW_DOMAIN_HASH_OFFSET), 0.0f));
    write_f32(dst, FINAL_STAR_POSITION_OFFSET + 4u, device_finite_or_default(raw_f32(src, RAW_SUBJECT_HASH_OFFSET), 0.0f));
    write_f32(dst, FINAL_STAR_POSITION_OFFSET + 8u, position_z);
}
