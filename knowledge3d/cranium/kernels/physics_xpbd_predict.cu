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

extern "C" __global__ void physics_xpbd_predict(
    PhysicsBodySOA* __restrict__ bodies,
    PhysicsPredictedSOA predicted,
    float dt
) {
    const uint32_t body_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!physics_body_valid(bodies, body_id)) return;

    const float3 pos = physics_position(bodies, body_id);
    const float3 vel = physics_velocity(bodies, body_id);
    const float3 ang_vel = physics_angular_velocity(bodies, body_id);
    const float4 q = physics_orientation(bodies, body_id);
    const float inv_mass = physics_inv_mass(bodies, body_id);

    const float3 predicted_pos = make_float3(
        pos.x + vel.x * dt,
        pos.y + vel.y * dt,
        pos.z + vel.z * dt);
    const float4 dq = make_float4(0.5f * dt * ang_vel.x, 0.5f * dt * ang_vel.y, 0.5f * dt * ang_vel.z, 1.0f);
    const float4 predicted_q = quat_normalize(quat_mul(dq, q));

    physics_store_predicted_pose(&predicted, body_id, predicted_pos, inv_mass, predicted_q);
}
