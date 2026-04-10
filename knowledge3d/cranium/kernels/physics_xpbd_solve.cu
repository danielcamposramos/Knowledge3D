#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {
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
}  // namespace

extern "C" __global__ void physics_xpbd_solve(
    PhysicsBodySOA* __restrict__ bodies,
    PhysicsPredictedSOA predicted,
    ContactManifoldSOA manifold,
    uint32_t constraint_count,
    uint8_t active_color,
    float* __restrict__ error_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_error = 0.0f;

    if (idx < constraint_count && idx < manifold.capacity && manifold.color_id[idx] == active_color) {
        const uint32_t body_a = manifold.body_a_id[idx];
        const uint32_t body_b = manifold.body_b_id[idx];
        const float inv_mass_a = physics_inv_mass(bodies, body_a);
        const float inv_mass_b = physics_inv_mass(bodies, body_b);
        const float compliance = manifold.compliance_normal[idx];
        const float c = manifold.penetration_depth[idx];
        const float denom = inv_mass_a + inv_mass_b + compliance;
        if (denom > 1e-8f && c > 0.0f) {
            const float delta_lambda = (-c - compliance * manifold.lambda_normal[idx]) / denom;
            manifold.lambda_normal[idx] += delta_lambda;

            const float3 normal = make_float3(
                manifold.normal_x[idx],
                manifold.normal_y[idx],
                manifold.normal_z[idx]);
            const float3 correction = vmul(normal, delta_lambda);

            if (body_a < predicted.capacity && inv_mass_a > 0.0f) {
                float4 pa = predicted.predicted_pos_inv[body_a];
                pa.x -= correction.x * inv_mass_a;
                pa.y -= correction.y * inv_mass_a;
                pa.z -= correction.z * inv_mass_a;
                predicted.predicted_pos_inv[body_a] = pa;
            }
            if (body_b < predicted.capacity && inv_mass_b > 0.0f) {
                float4 pb = predicted.predicted_pos_inv[body_b];
                pb.x += correction.x * inv_mass_b;
                pb.y += correction.y * inv_mass_b;
                pb.z += correction.z * inv_mass_b;
                predicted.predicted_pos_inv[body_b] = pb;
            }
            local_error = fabsf(c);
        }
    }

    for (int delta = 16; delta > 0; delta >>= 1) {
        local_error += __shfl_xor_sync(0xffffffffu, local_error, delta);
    }
    if ((threadIdx.x & 31) == 0 && error_out != nullptr) {
        atomicAdd(error_out, local_error);
    }
}
