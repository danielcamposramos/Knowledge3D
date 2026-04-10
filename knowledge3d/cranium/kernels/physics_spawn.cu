#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void physics_spawn_body(
    PhysicsBodySOA* __restrict__ bodies,
    uint32_t material_star_id,
    uint32_t shape_star_id,
    float3 position,
    float3 velocity,
    float inv_mass,
    uint32_t* __restrict__ body_idx_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (bodies == nullptr || bodies->body_count >= bodies->capacity) return;
    const uint32_t body_id = atomicAdd(&bodies->body_count, 1u);
    if (body_id >= bodies->capacity) return;

    bodies->pos_inv[body_id] = make_float4(position.x, position.y, position.z, inv_mass);
    bodies->vel_sleep[body_id] = make_float4(velocity.x, velocity.y, velocity.z, 0.0f);
    bodies->orientation[body_id] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    bodies->ang_vel_damp[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.01f);
    bodies->inv_inertia_rest[body_id] = make_float4(1.0f, 1.0f, 1.0f, 0.25f);
    bodies->galaxy_handles[body_id] = make_uint2(material_star_id, shape_star_id);
    bodies->island_flags[body_id] = PHYSICS_FLAG_DIRTY;
    bodies->bound_friction[body_id] = make_float2(0.5f, 0.5f);
    if (body_idx_out != nullptr) {
        *body_idx_out = body_id;
    }
}

extern "C" __global__ void physics_despawn_body(
    PhysicsBodySOA* __restrict__ bodies,
    uint32_t body_id
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!physics_body_valid(bodies, body_id)) return;
    bodies->pos_inv[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bodies->vel_sleep[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bodies->orientation[body_id] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    bodies->ang_vel_damp[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bodies->inv_inertia_rest[body_id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bodies->galaxy_handles[body_id] = make_uint2(0u, 0u);
    bodies->island_flags[body_id] = 0u;
    bodies->bound_friction[body_id] = make_float2(0.0f, 0.0f);
}
