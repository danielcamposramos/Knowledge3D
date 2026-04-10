#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
}  // namespace

extern "C" __global__ void physics_raycast(
    const PhysicsBodySOA* __restrict__ bodies,
    float3 ray_origin,
    float3 ray_dir,
    uint32_t* __restrict__ body_idx_out,
    float* __restrict__ t_out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (body_idx_out != nullptr) *body_idx_out = 0xffffffffu;
    if (t_out != nullptr) *t_out = -1.0f;
    if (bodies == nullptr) return;

    float best_t = 1.0e30f;
    uint32_t best_body = 0xffffffffu;
    for (uint32_t body_id = 0; body_id < bodies->body_count; ++body_id) {
        const float3 center = physics_position(bodies, body_id);
        const float radius = physics_bound_radius(bodies, body_id);
        const float3 oc = vsub(ray_origin, center);
        const float a = vdot(ray_dir, ray_dir);
        const float b = 2.0f * vdot(oc, ray_dir);
        const float c = vdot(oc, oc) - radius * radius;
        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) continue;
        const float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        if (t >= 0.0f && t < best_t) {
            best_t = t;
            best_body = body_id;
        }
    }
    if (body_idx_out != nullptr) *body_idx_out = best_body;
    if (t_out != nullptr) *t_out = best_t == 1.0e30f ? -1.0f : best_t;
}
