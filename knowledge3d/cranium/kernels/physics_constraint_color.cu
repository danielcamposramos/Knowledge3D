#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void physics_constraint_color(
    ContactManifoldSOA manifold,
    uint32_t constraint_count,
    uint32_t* __restrict__ color_count_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t max_color = 0u;
    for (uint32_t idx = 0; idx < constraint_count && idx < manifold.capacity; ++idx) {
        uint8_t chosen = 0u;
        const uint32_t a = manifold.body_a_id[idx];
        const uint32_t b = manifold.body_b_id[idx];
        bool assigned = false;

        while (!assigned) {
            bool conflicts = false;
            for (uint32_t prev = 0; prev < idx; ++prev) {
                if (manifold.color_id[prev] != chosen) continue;
                const uint32_t pa = manifold.body_a_id[prev];
                const uint32_t pb = manifold.body_b_id[prev];
                if (a == pa || a == pb || b == pa || b == pb) {
                    conflicts = true;
                    break;
                }
            }
            if (!conflicts) {
                manifold.color_id[idx] = chosen;
                if (chosen + 1u > max_color) max_color = chosen + 1u;
                assigned = true;
            } else {
                chosen = static_cast<uint8_t>(chosen + 1u);
            }
        }
    }

    if (color_count_out != nullptr) {
        *color_count_out = max_color;
    }
}
