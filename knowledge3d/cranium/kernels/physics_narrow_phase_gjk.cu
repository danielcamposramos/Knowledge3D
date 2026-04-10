#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr uint32_t kWarpSize = 32u;

__device__ __forceinline__ float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 vmul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float vlen(float3 a) {
    return sqrtf(vdot(a, a));
}

__device__ __forceinline__ float3 vnormalize(float3 a) {
    float len = vlen(a);
    if (len <= 1e-6f) return make_float3(0.0f, 1.0f, 0.0f);
    return vmul(a, 1.0f / len);
}

__device__ __forceinline__ float3 gjk_support_sphere(float3 center, float radius, float3 direction) {
    return vadd(center, vmul(vnormalize(direction), radius));
}
}  // namespace

extern "C" __global__ void physics_narrow_phase_gjk(
    const PhysicsBodySOA* __restrict__ bodies,
    ContactManifoldSOA manifold,
    uint32_t pair_count,
    uint32_t current_frame,
    uint32_t* __restrict__ contact_count_out
) {
    const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = global_thread / kWarpSize;
    const uint32_t lane = threadIdx.x & (kWarpSize - 1u);
    if (warp_id >= pair_count) return;

    const uint32_t body_a = manifold.body_a_id[warp_id];
    const uint32_t body_b = manifold.body_b_id[warp_id];
    if (!physics_body_valid(bodies, body_a) || !physics_body_valid(bodies, body_b)) return;

    const float3 pos_a = physics_position(bodies, body_a);
    const float3 pos_b = physics_position(bodies, body_b);
    const float radius_a = physics_bound_radius(bodies, body_a);
    const float radius_b = physics_bound_radius(bodies, body_b);

    float3 direction = make_float3(1.0f, 0.0f, 0.0f);
    float3 simplex[4];
    int simplex_size = 0;

    for (int iteration = 0; iteration < 4; ++iteration) {
        float3 support_a = gjk_support_sphere(pos_a, radius_a, direction);
        float3 support_b = gjk_support_sphere(pos_b, radius_b, vmul(direction, -1.0f));
        float3 support = vsub(support_a, support_b);
        if (lane == 0u) {
            simplex[simplex_size] = support;
            simplex_size += 1;
            direction = vmul(support, -1.0f);
        }
        float support_len = __shfl_sync(0xffffffffu, vlen(support), 0);
        int keep_iterating = support_len > 1e-5f && simplex_size < 4;
        if (!__any_sync(0xffffffffu, keep_iterating)) {
            break;
        }
    }

    if (lane != 0u) return;

    const float3 delta = vsub(pos_b, pos_a);
    const float distance = vlen(delta);
    const float combined_radius = radius_a + radius_b;
    if (distance >= combined_radius) {
        return;
    }

    const float3 normal = distance > 1e-6f ? vmul(delta, 1.0f / distance) : make_float3(0.0f, 1.0f, 0.0f);
    const float penetration = combined_radius - distance;
    const float3 contact = vadd(pos_a, vmul(normal, radius_a - 0.5f * penetration));

    manifold.contact_x[warp_id] = contact.x;
    manifold.contact_y[warp_id] = contact.y;
    manifold.contact_z[warp_id] = contact.z;
    manifold.normal_x[warp_id] = normal.x;
    manifold.normal_y[warp_id] = normal.y;
    manifold.normal_z[warp_id] = normal.z;
    manifold.penetration_depth[warp_id] = penetration;
    manifold.frame_stamp[warp_id] = static_cast<uint8_t>(current_frame & 0xffu);

    if (contact_count_out != nullptr) {
        atomicAdd(contact_count_out, 1u);
    }
}
