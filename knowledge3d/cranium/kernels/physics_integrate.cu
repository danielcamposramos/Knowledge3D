#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {
__device__ __forceinline__ float4 quat_mul(float4 a, float4 b) {
    return make_float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

__device__ __forceinline__ float4 quat_normalize(float4 q) {
    float norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm <= 1e-8f) return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float inv = 1.0f / norm;
    return make_float4(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
}
}  // namespace

extern "C" __global__ void physics_integrate(
    PhysicsBodySOA* __restrict__ bodies,
    const PhysicsPredictedSOA predicted,
    float dt,
    float gravity_y
) {
    const uint32_t body_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!physics_body_valid(bodies, body_id)) return;
    if (physics_body_is_static(bodies, body_id)) return;

    float3 position = physics_position(bodies, body_id);
    float3 velocity = physics_velocity(bodies, body_id);
    const float3 angular_velocity = physics_angular_velocity(bodies, body_id);
    const float inv_mass = physics_inv_mass(bodies, body_id);

    if (inv_mass > 0.0f) {
        velocity.y += dt * gravity_y;
    }

    if (body_id < predicted.capacity && predicted.predicted_pos_inv != nullptr) {
        const float4 predicted_pos = predicted.predicted_pos_inv[body_id];
        position = make_float3(predicted_pos.x, predicted_pos.y, predicted_pos.z);
        if (predicted.predicted_orientation != nullptr) {
            physics_store_orientation(bodies, body_id, quat_normalize(predicted.predicted_orientation[body_id]));
        }
    } else {
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;
        const float4 q = physics_orientation(bodies, body_id);
        const float4 dq = make_float4(0.5f * dt * angular_velocity.x, 0.5f * dt * angular_velocity.y, 0.5f * dt * angular_velocity.z, 1.0f);
        physics_store_orientation(bodies, body_id, quat_normalize(quat_mul(dq, q)));
    }

    physics_store_position(bodies, body_id, position);
    physics_store_velocity(bodies, body_id, velocity);
    physics_mark_dirty(bodies, body_id);
}
