#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float3 tangent_from_normal(float3 n, int axis) {
    float3 seed = axis == 0 ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 t = make_float3(
        n.y * seed.z - n.z * seed.y,
        n.z * seed.x - n.x * seed.z,
        n.x * seed.y - n.y * seed.x);
    float mag = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    if (mag <= 1e-6f) {
        return axis == 0 ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 0.0f, 1.0f);
    }
    return make_float3(t.x / mag, t.y / mag, t.z / mag);
}
}  // namespace

extern "C" __global__ void physics_constraint_generate(
    const PhysicsBodySOA* __restrict__ bodies,
    ContactManifoldSOA* __restrict__ manifold,
    uint32_t contact_count,
    uint32_t* __restrict__ constraint_count_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (manifold == nullptr || idx >= contact_count || idx >= manifold->capacity) return;

    const uint32_t body_a = manifold->body_a_id[idx];
    const uint32_t body_b = manifold->body_b_id[idx];
    if (!physics_body_valid(bodies, body_a) || !physics_body_valid(bodies, body_b)) return;

    const float nx = manifold->normal_x[idx];
    const float ny = manifold->normal_y[idx];
    const float nz = manifold->normal_z[idx];
    const float3 n = make_float3(nx, ny, nz);

    if (manifold->persistent_id[idx] == 0u) {
        manifold->persistent_id[idx] = atomicAdd(&manifold->persistent_counter, 1u) + 1u;
        manifold->lambda_normal[idx] = 0.0f;
        manifold->lambda_tangent0[idx] = 0.0f;
        manifold->lambda_tangent1[idx] = 0.0f;
    } else {
        manifold->lambda_normal[idx] *= 0.85f;
        manifold->lambda_tangent0[idx] *= 0.85f;
        manifold->lambda_tangent1[idx] *= 0.85f;
    }

    const float friction = 0.5f * (physics_friction(bodies, body_a) + physics_friction(bodies, body_b));
    const float restitution = 0.5f * (physics_restitution(bodies, body_a) + physics_restitution(bodies, body_b));
    const float3 tangent0 = tangent_from_normal(n, 0);
    const float3 tangent1 = tangent_from_normal(n, 1);

    manifold->compliance_normal[idx] = restitution > 0.75f ? 1.0e-5f : 0.0f;
    manifold->lambda_tangent0[idx] = tangent0.x * friction;
    manifold->lambda_tangent1[idx] = tangent1.z * friction;

    if (constraint_count_out != nullptr) {
        atomicAdd(constraint_count_out, 1u);
    }
}
