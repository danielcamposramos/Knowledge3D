#include <stdint.h>

#include "device_functions.cuh"

#define RAW_ROW_BYTES 176u
#define RAW_SELECTION_ROLE_ID_OFFSET 72u
#define RAW_LAYER_ID_OFFSET 76u
#define RAW_ANSWER_ELIGIBLE_OFFSET 84u
#define RAW_ROUTE_POLICY_FLAGS_OFFSET 112u
#define RAW_ROUTER_REF_COUNT_OFFSET 124u
#define RAW_EXECUTOR_REF_COUNT_OFFSET 128u
#define RAW_VALIDATOR_REF_COUNT_OFFSET 132u

static __device__ __forceinline__ unsigned int read_u32_route(
    const unsigned char* ptr,
    const unsigned int offset
) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

extern "C" __global__ void route_capability_trit(
    const unsigned char* __restrict__ raw_rows,
    unsigned int star_count,
    int* __restrict__ route_trits,
    unsigned int* __restrict__ audit_counts
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= star_count) {
        return;
    }

    const unsigned char* row = raw_rows + (i * RAW_ROW_BYTES);
    const unsigned int role_id = read_u32_route(row, RAW_SELECTION_ROLE_ID_OFFSET);
    const unsigned int layer_id = read_u32_route(row, RAW_LAYER_ID_OFFSET);
    const unsigned int answer_eligible = read_u32_route(row, RAW_ANSWER_ELIGIBLE_OFFSET);
    const unsigned int route_policy_flags = read_u32_route(row, RAW_ROUTE_POLICY_FLAGS_OFFSET);
    const unsigned int executor_refs = read_u32_route(row, RAW_EXECUTOR_REF_COUNT_OFFSET);
    const unsigned int validator_refs = read_u32_route(row, RAW_VALIDATOR_REF_COUNT_OFFSET);

    int route_state = 0;
    switch (role_id) {
        case GALAXY_ROLE_ROUTER:
            route_state = (layer_id > 0u
                && answer_eligible == 0u
                && (route_policy_flags & ROUTE_POLICY_REQUIRES_EXECUTOR) != 0u
                && (route_policy_flags & ROUTE_POLICY_REQUIRES_VALIDATOR) != 0u
                && (route_policy_flags & ROUTE_POLICY_ANSWER_GATE) != 0u
                && executor_refs > 0u
                && validator_refs > 0u) ? 1 : -1;
            break;
        case GALAXY_ROLE_EXECUTOR:
            route_state = (layer_id > 0u
                && (((route_policy_flags & ROUTE_POLICY_REQUIRES_VALIDATOR) == 0u) || validator_refs > 0u)) ? 1 : -1;
            break;
        case GALAXY_ROLE_VALIDATOR:
        case GALAXY_ROLE_ANSWER:
        case GALAXY_ROLE_ANTI_PATTERN:
            route_state = layer_id > 0u ? 1 : -1;
            break;
        default:
            route_state = 0;
            break;
    }

    const int trit = static_cast<int>(clamp_trit_int_device(route_state));
    route_trits[i] = trit;
    if (trit > 0) {
        atomicAdd(audit_counts + 0, 1u);
    } else if (trit < 0) {
        atomicAdd(audit_counts + 2, 1u);
    } else {
        atomicAdd(audit_counts + 1, 1u);
    }
}
