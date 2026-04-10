#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void physics_emit_collision_events(
    const PhysicsBodySOA* __restrict__ bodies,
    const ContactManifoldSOA manifold,
    uint32_t contact_count,
    CollisionEventQueue* __restrict__ event_queue
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_queue == nullptr || idx >= contact_count || idx >= manifold.capacity) return;
    if (manifold.penetration_depth[idx] <= 0.0f) return;

    const uint32_t slot = atomicAdd(&event_queue->write_head, 1u);
    if (slot >= event_queue->capacity) return;

    const uint32_t body_a = manifold.body_a_id[idx];
    const uint32_t body_b = manifold.body_b_id[idx];
    event_queue->body_a_id[slot] = body_a;
    event_queue->body_b_id[slot] = body_b;
    event_queue->material_a_star_id[slot] = physics_material_star_id(bodies, body_a);
    event_queue->material_b_star_id[slot] = physics_material_star_id(bodies, body_b);
    event_queue->impulse_magnitude[slot] = fabsf(manifold.lambda_normal[idx]);
    event_queue->normal_x[slot] = manifold.normal_x[idx];
    event_queue->normal_y[slot] = manifold.normal_y[idx];
    event_queue->normal_z[slot] = manifold.normal_z[idx];
}

extern "C" __global__ void physics_collision_event_write(
    const CollisionEventQueue event_queue,
    uint32_t* __restrict__ edge_src_material,
    uint32_t* __restrict__ edge_dst_material,
    float* __restrict__ edge_weight,
    float4* __restrict__ edge_normal_frame,
    uint32_t frame_index,
    uint32_t* __restrict__ edge_count_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_queue.write_head || idx >= event_queue.capacity) return;

    edge_src_material[idx] = event_queue.material_a_star_id[idx];
    edge_dst_material[idx] = event_queue.material_b_star_id[idx];
    edge_weight[idx] = event_queue.impulse_magnitude[idx];
    edge_normal_frame[idx] = make_float4(
        event_queue.normal_x[idx],
        event_queue.normal_y[idx],
        event_queue.normal_z[idx],
        static_cast<float>(frame_index));
    if (edge_count_out != nullptr) {
        atomicAdd(edge_count_out, 1u);
    }
}
