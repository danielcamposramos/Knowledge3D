#include <stdint.h>

#define K3D_MODEL_CHECK_MAX_STATES 64u
#define K3D_MODEL_CHECK_STATUS_UNKNOWN 0u
#define K3D_MODEL_CHECK_STATUS_PASS 1u
#define K3D_MODEL_CHECK_STATUS_FAIL 2u

struct __align__(16) ModelCheckResult {
    uint32_t status;
    uint32_t visited_count;
    uint32_t frontier_peak;
    uint32_t witness_state;
};

extern "C" __global__ void k3d_model_check_reuse(
    const uint32_t* __restrict__ state_props,
    const uint32_t* __restrict__ adjacency,
    uint32_t num_states,
    uint32_t max_degree,
    uint32_t root_state,
    uint32_t target_mask,
    uint32_t forbidden_mask,
    ModelCheckResult* __restrict__ result
) {
    if (threadIdx.x != 0u || blockIdx.x != 0u || result == nullptr) {
        return;
    }

    result->status = K3D_MODEL_CHECK_STATUS_UNKNOWN;
    result->visited_count = 0u;
    result->frontier_peak = 0u;
    result->witness_state = 0xFFFFFFFFu;

    if (state_props == nullptr || adjacency == nullptr || num_states == 0u || max_degree == 0u || root_state >= num_states) {
        return;
    }
    if (num_states > K3D_MODEL_CHECK_MAX_STATES) {
        return;
    }

    uint32_t queue[K3D_MODEL_CHECK_MAX_STATES];
    uint64_t visited_mask = 0u;
    uint32_t head = 0u;
    uint32_t tail = 0u;
    queue[tail++] = root_state;
    visited_mask |= (1ull << root_state);
    result->frontier_peak = 1u;

    while (head < tail) {
        const uint32_t state = queue[head++];
        result->visited_count += 1u;
        const uint32_t props = state_props[state];

        if ((props & forbidden_mask) != 0u) {
            result->status = K3D_MODEL_CHECK_STATUS_FAIL;
            result->witness_state = state;
            return;
        }
        if ((props & target_mask) == target_mask) {
            result->status = K3D_MODEL_CHECK_STATUS_PASS;
            result->witness_state = state;
            return;
        }

        const uint32_t edge_base = state * max_degree;
        for (uint32_t edge = 0u; edge < max_degree; ++edge) {
            const uint32_t next = adjacency[edge_base + edge];
            if (next == 0xFFFFFFFFu) {
                continue;
            }
            if (next >= num_states) {
                result->status = K3D_MODEL_CHECK_STATUS_UNKNOWN;
                result->witness_state = state;
                return;
            }
            const uint64_t bit = 1ull << next;
            if ((visited_mask & bit) != 0u) {
                continue;
            }
            if (tail >= K3D_MODEL_CHECK_MAX_STATES) {
                result->status = K3D_MODEL_CHECK_STATUS_UNKNOWN;
                result->witness_state = state;
                return;
            }
            visited_mask |= bit;
            queue[tail++] = next;
            if ((tail - head) > result->frontier_peak) {
                result->frontier_peak = tail - head;
            }
        }
    }
}
