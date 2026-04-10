#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#define CONTACT_CAPACITY 65536u
#define COLLISION_QUEUE_CAPACITY 4096u

enum PhysicsBodyFlags : uint32_t {
    PHYSICS_FLAG_SLEEPING = 1u << 7,
    PHYSICS_FLAG_STATIC = 1u << 6,
    PHYSICS_FLAG_KINEMATIC = 1u << 5,
    PHYSICS_FLAG_DIRTY = 1u << 4,
    PHYSICS_FLAG_TRIGGER = 1u << 3,
};

typedef struct {
    float4* pos_inv;            // xyz=position, w=inv_mass
    float4* vel_sleep;          // xyz=velocity, w=sleep_energy_accum
    float4* orientation;        // xyzw quaternion
    float4* ang_vel_damp;       // xyz=angular_velocity, w=angular_damping
    float4* inv_inertia_rest;   // xyz=local inverse inertia diagonal, w=restitution
    uint2* galaxy_handles;      // x=material_star_id, y=shape_star_id
    uint32_t* island_flags;     // packed island + state bits
    float2* bound_friction;     // x=bounding radius, y=friction coeff
    uint32_t body_count;
    uint32_t capacity;
} PhysicsBodySOA;

typedef struct {
    float4* predicted_pos_inv;        // xyz=predicted position, w=inv_mass
    float4* predicted_orientation;    // predicted orientation quaternion
    uint32_t capacity;
} PhysicsPredictedSOA;

typedef struct {
    uint32_t* body_a_id;
    uint32_t* body_b_id;
    float* contact_x;
    float* contact_y;
    float* contact_z;
    float* normal_x;
    float* normal_y;
    float* normal_z;
    float* penetration_depth;
    float* lambda_normal;
    float* lambda_tangent0;
    float* lambda_tangent1;
    float* compliance_normal;
    uint32_t* persistent_id;
    uint8_t* frame_stamp;
    uint8_t* color_id;
    uint32_t capacity;
    uint32_t write_head;
    uint32_t persistent_counter;
} ContactManifoldSOA;

typedef struct {
    uint32_t* body_a_id;
    uint32_t* body_b_id;
    uint32_t* material_a_star_id;
    uint32_t* material_b_star_id;
    float* impulse_magnitude;
    float* normal_x;
    float* normal_y;
    float* normal_z;
    uint32_t write_head;
    uint32_t capacity;
} CollisionEventQueue;

__device__ __forceinline__ bool physics_body_valid(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa != nullptr && body_id < soa->body_count && body_id < soa->capacity;
}

__device__ __forceinline__ bool physics_body_is_sleeping(const PhysicsBodySOA* soa, uint32_t body_id) {
    return physics_body_valid(soa, body_id) &&
        ((soa->island_flags[body_id] & PHYSICS_FLAG_SLEEPING) != 0u);
}

__device__ __forceinline__ bool physics_body_is_static(const PhysicsBodySOA* soa, uint32_t body_id) {
    return physics_body_valid(soa, body_id) &&
        ((soa->island_flags[body_id] & PHYSICS_FLAG_STATIC) != 0u);
}

__device__ __forceinline__ void physics_set_sleeping(PhysicsBodySOA* soa, uint32_t body_id, bool sleeping) {
    if (!physics_body_valid(soa, body_id)) return;
    uint32_t flags = soa->island_flags[body_id];
    if (sleeping) {
        flags |= PHYSICS_FLAG_SLEEPING;
    } else {
        flags &= ~PHYSICS_FLAG_SLEEPING;
    }
    soa->island_flags[body_id] = flags;
}

__device__ __forceinline__ void physics_mark_dirty(PhysicsBodySOA* soa, uint32_t body_id) {
    if (!physics_body_valid(soa, body_id)) return;
    soa->island_flags[body_id] |= PHYSICS_FLAG_DIRTY;
}

__device__ __forceinline__ float3 physics_position(const PhysicsBodySOA* soa, uint32_t body_id) {
    const float4 v = soa->pos_inv[body_id];
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float physics_inv_mass(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->pos_inv[body_id].w;
}

__device__ __forceinline__ float3 physics_velocity(const PhysicsBodySOA* soa, uint32_t body_id) {
    const float4 v = soa->vel_sleep[body_id];
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float physics_sleep_energy(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->vel_sleep[body_id].w;
}

__device__ __forceinline__ float4 physics_orientation(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->orientation[body_id];
}

__device__ __forceinline__ float3 physics_angular_velocity(const PhysicsBodySOA* soa, uint32_t body_id) {
    const float4 v = soa->ang_vel_damp[body_id];
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float physics_angular_damping(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->ang_vel_damp[body_id].w;
}

__device__ __forceinline__ float3 physics_local_inv_inertia(const PhysicsBodySOA* soa, uint32_t body_id) {
    const float4 v = soa->inv_inertia_rest[body_id];
    return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ float physics_restitution(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->inv_inertia_rest[body_id].w;
}

__device__ __forceinline__ float physics_bound_radius(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->bound_friction[body_id].x;
}

__device__ __forceinline__ float physics_friction(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->bound_friction[body_id].y;
}

__device__ __forceinline__ uint32_t physics_material_star_id(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->galaxy_handles[body_id].x;
}

__device__ __forceinline__ uint32_t physics_shape_star_id(const PhysicsBodySOA* soa, uint32_t body_id) {
    return soa->galaxy_handles[body_id].y;
}

__device__ __forceinline__ uint32_t physics_island_id(const PhysicsBodySOA* soa, uint32_t body_id) {
    return physics_body_valid(soa, body_id) ? (soa->island_flags[body_id] >> 8) : 0u;
}

__device__ __forceinline__ void physics_store_position(PhysicsBodySOA* soa, uint32_t body_id, float3 position) {
    float4 packed = soa->pos_inv[body_id];
    packed.x = position.x;
    packed.y = position.y;
    packed.z = position.z;
    soa->pos_inv[body_id] = packed;
}

__device__ __forceinline__ void physics_store_velocity(PhysicsBodySOA* soa, uint32_t body_id, float3 velocity) {
    float4 packed = soa->vel_sleep[body_id];
    packed.x = velocity.x;
    packed.y = velocity.y;
    packed.z = velocity.z;
    soa->vel_sleep[body_id] = packed;
}

__device__ __forceinline__ void physics_store_sleep_energy(PhysicsBodySOA* soa, uint32_t body_id, float energy) {
    float4 packed = soa->vel_sleep[body_id];
    packed.w = energy;
    soa->vel_sleep[body_id] = packed;
}

__device__ __forceinline__ void physics_store_orientation(PhysicsBodySOA* soa, uint32_t body_id, float4 q) {
    soa->orientation[body_id] = q;
}

__device__ __forceinline__ void physics_store_predicted_pose(
    PhysicsPredictedSOA* predicted,
    uint32_t body_id,
    float3 position,
    float inv_mass,
    float4 orientation
) {
    if (predicted == nullptr || body_id >= predicted->capacity) return;
    predicted->predicted_pos_inv[body_id] = make_float4(position.x, position.y, position.z, inv_mass);
    predicted->predicted_orientation[body_id] = orientation;
}
