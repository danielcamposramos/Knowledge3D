#include "physics_body_soa.h"

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void physics_sleep_island(
    PhysicsBodySOA* __restrict__ bodies,
    float energy_threshold,
    uint32_t wake_star_id,
    uint32_t* __restrict__ island_count_out,
    uint32_t* __restrict__ woken_count_out
) {
    const uint32_t body_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!physics_body_valid(bodies, body_id)) return;

    const float3 v = physics_velocity(bodies, body_id);
    const float3 w = physics_angular_velocity(bodies, body_id);
    const float energy = 0.5f * (v.x * v.x + v.y * v.y + v.z * v.z + w.x * w.x + w.y * w.y + w.z * w.z);
    const unsigned int warp_sleep_mask = __ballot_sync(0xffffffffu, energy < energy_threshold);
    const bool warp_sleeping = (__popc(warp_sleep_mask) == 32);

    physics_store_sleep_energy(bodies, body_id, energy);
    if (warp_sleeping) {
        physics_set_sleeping(bodies, body_id, true);
        if ((threadIdx.x & 31) == 0 && island_count_out != nullptr) {
            atomicAdd(island_count_out, 1u);
        }
    }

    const bool wake_triggered = wake_star_id != 0u && physics_material_star_id(bodies, body_id) == wake_star_id;
    if (wake_triggered) {
        const bool was_sleeping = physics_body_is_sleeping(bodies, body_id);
        physics_set_sleeping(bodies, body_id, false);
        if (was_sleeping && woken_count_out != nullptr) {
            atomicAdd(woken_count_out, 1u);
        }
    }
}
