#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float physics_absf(float v) {
    return v < 0.0f ? -v : v;
}

__device__ __forceinline__ float physics_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ uint32_t physics_expand_bits(uint32_t n) {
    n &= 0x000003ffu;
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n << 8)) & 0x0300f00fu;
    n = (n ^ (n << 4)) & 0x030c30c3u;
    n = (n ^ (n << 2)) & 0x09249249u;
    return n;
}

__device__ __forceinline__ uint32_t morton_encode_point(float3 position, float3 world_min, float world_extent) {
    float inv_extent = world_extent > 1e-6f ? (1.0f / world_extent) : 0.0f;
    float nx = physics_clampf((position.x - world_min.x) * inv_extent, 0.0f, 1.0f);
    float ny = physics_clampf((position.y - world_min.y) * inv_extent, 0.0f, 1.0f);
    float nz = physics_clampf((position.z - world_min.z) * inv_extent, 0.0f, 1.0f);
    uint32_t ix = static_cast<uint32_t>(nx * 1023.0f);
    uint32_t iy = static_cast<uint32_t>(ny * 1023.0f);
    uint32_t iz = static_cast<uint32_t>(nz * 1023.0f);
    return (physics_expand_bits(ix) << 2) | (physics_expand_bits(iy) << 1) | physics_expand_bits(iz);
}

extern "C" __global__ void physics_update_morton_codes(
    const PhysicsBodySOA* __restrict__ bodies,
    uint32_t* __restrict__ morton_codes,
    uint32_t* __restrict__ body_ids,
    float3 world_min,
    float world_extent
) {
    const uint32_t body_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!physics_body_valid(bodies, body_id)) return;
    if (physics_body_is_sleeping(bodies, body_id) && !((bodies->island_flags[body_id] & PHYSICS_FLAG_DIRTY) != 0u)) {
        morton_codes[body_id] = 0xffffffffu;
        body_ids[body_id] = body_id;
        return;
    }
    morton_codes[body_id] = morton_encode_point(physics_position(bodies, body_id), world_min, world_extent);
    body_ids[body_id] = body_id;
}

extern "C" __global__ void physics_broad_phase_sap(
    const PhysicsBodySOA* __restrict__ bodies,
    const uint32_t* __restrict__ sorted_body_ids,
    ContactManifoldSOA* __restrict__ manifold,
    uint32_t* __restrict__ pair_count_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!physics_body_valid(bodies, idx)) return;
    if (manifold == nullptr) return;
    const uint32_t body_a = sorted_body_ids ? sorted_body_ids[idx] : idx;
    if (!physics_body_valid(bodies, body_a) || physics_body_is_static(bodies, body_a)) return;

    const float3 pos_a = physics_position(bodies, body_a);
    const float radius_a = physics_bound_radius(bodies, body_a);
    const uint32_t total = bodies->body_count;

    for (uint32_t scan = idx + 1; scan < total; ++scan) {
        const uint32_t body_b = sorted_body_ids ? sorted_body_ids[scan] : scan;
        if (!physics_body_valid(bodies, body_b)) continue;
        if (physics_body_is_sleeping(bodies, body_a) && physics_body_is_sleeping(bodies, body_b)) continue;

        const float3 pos_b = physics_position(bodies, body_b);
        const float radius_b = physics_bound_radius(bodies, body_b);
        const float overlap = radius_a + radius_b;
        if (physics_absf(pos_b.x - pos_a.x) > overlap) break;  // sorted sweep axis cut
        if (physics_absf(pos_b.y - pos_a.y) > overlap || physics_absf(pos_b.z - pos_a.z) > overlap) continue;

        const uint32_t slot = atomicAdd(&manifold->write_head, 1u);
        if (slot >= manifold->capacity) break;
        manifold->body_a_id[slot] = body_a;
        manifold->body_b_id[slot] = body_b;
        if (pair_count_out != nullptr) {
            atomicAdd(pair_count_out, 1u);
        }
    }
}
