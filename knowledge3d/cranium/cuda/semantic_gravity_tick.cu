#include "device_functions.cuh"

extern "C" __global__ void semantic_gravity_tick(
    unsigned char* __restrict__ galaxy_table,
    unsigned int n_stars,
    const unsigned int* __restrict__ router_offsets,
    const unsigned int* __restrict__ router_counts,
    const unsigned int* __restrict__ executor_offsets,
    const unsigned int* __restrict__ executor_counts,
    const unsigned int* __restrict__ validator_offsets,
    const unsigned int* __restrict__ validator_counts,
    const unsigned int* __restrict__ anti_pattern_offsets,
    const unsigned int* __restrict__ anti_pattern_counts,
    const unsigned int* __restrict__ ref_indices,
    float dt,
    float damping
) {
    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= n_stars) {
        return;
    }
    const GalaxyRoleAdjacencyDeviceView adjacency = {
        router_offsets,
        router_counts,
        executor_offsets,
        executor_counts,
        validator_offsets,
        validator_counts,
        anti_pattern_offsets,
        anti_pattern_counts,
        ref_indices,
    };

    const float* emb_i = galaxy_read_embedding(galaxy_table, index);
    const float mass_i = device_maxf(galaxy_read_mass(galaxy_table, index), 1.0e-3f);
    const int polarity_i = galaxy_read_polarity(galaxy_table, index);
    const float attr_i = galaxy_read_attractive_prior(galaxy_table, index);
    const float repel_i = galaxy_read_repulsive_prior(galaxy_table, index);

    float* pos_i = galaxy_write_float_ptr(galaxy_table, index, GALAXY_STAR_POSITION_OFFSET);
    float* vel_i = galaxy_write_float_ptr(galaxy_table, index, GALAXY_STAR_VELOCITY_OFFSET);

    float px = pos_i[0];
    float py = pos_i[1];
    float pz = pos_i[2];
    float vx = vel_i[0];
    float vy = vel_i[1];
    float vz = vel_i[2];

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    float norm_i = 0.0f;
    for (unsigned int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
        const float value = emb_i[d];
        norm_i += value * value;
    }
    norm_i = sqrtf(device_maxf(norm_i, 1.0e-12f));

    for (unsigned int other = 0u; other < n_stars; ++other) {
        if (other == index) {
            continue;
        }

        const float* emb_j = galaxy_read_embedding(galaxy_table, other);
        float dot = 0.0f;
        float norm_j = 0.0f;
        for (unsigned int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
            const float left = emb_i[d];
            const float right = emb_j[d];
            dot += left * right;
            norm_j += right * right;
        }
        norm_j = sqrtf(device_maxf(norm_j, 1.0e-12f));
        const float cosine = dot / device_maxf(norm_i * norm_j, 1.0e-12f);

        float link_bias = 0.0f;
        for (unsigned int role_kind = GALAXY_ROLE_ROUTER; role_kind <= GALAXY_ROLE_ANTI_PATTERN; ++role_kind) {
            const unsigned int ref_count = galaxy_read_role_ref_count_csr(adjacency, index, role_kind);
            const unsigned int bounded_refs = ref_count > GPU_ROUTE_FRONTIER_WIDTH ? GPU_ROUTE_FRONTIER_WIDTH : ref_count;
            for (unsigned int ref_slot = 0u; ref_slot < bounded_refs; ++ref_slot) {
                const unsigned int ref_index = galaxy_read_role_ref_csr(adjacency, index, role_kind, ref_slot);
                if (ref_index != other) {
                    continue;
                }
                if (role_kind == GALAXY_ROLE_ANTI_PATTERN) {
                    link_bias -= 0.45f;
                } else {
                    link_bias += 0.35f;
                }
            }
        }

        float semantic_force = cosine + link_bias + attr_i - repel_i;
        const int polarity_j = galaxy_read_polarity(galaxy_table, other);
        if (polarity_i != 0 && polarity_j != 0 && polarity_i == -polarity_j) {
            semantic_force -= 0.35f;
        }
        if (semantic_force == 0.0f) {
            continue;
        }

        float* pos_j = galaxy_write_float_ptr(galaxy_table, other, GALAXY_STAR_POSITION_OFFSET);
        float dx = pos_j[0] - px;
        float dy = pos_j[1] - py;
        float dz = pos_j[2] - pz;
        float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);
        if (dist_sq < 1.0e-6f) {
            dx = float(int(other) - int(index));
            dy = float(((index + other) % 7u)) - 3.0f;
            dz = float(((other * 3u + index) % 5u)) - 2.0f;
            dist_sq = (dx * dx) + (dy * dy) + (dz * dz);
        }
        dist_sq = device_maxf(dist_sq, 1.0e-3f);
        const float inv_dist = rsqrtf(dist_sq);
        const float magnitude = semantic_force * mass_i * device_maxf(galaxy_read_mass(galaxy_table, other), 1.0e-3f) / dist_sq;
        fx += magnitude * dx * inv_dist;
        fy += magnitude * dy * inv_dist;
        fz += magnitude * dz * inv_dist;
    }

    vx = (vx + ((fx / mass_i) * dt)) * damping;
    vy = (vy + ((fy / mass_i) * dt)) * damping;
    vz = (vz + ((fz / mass_i) * dt)) * damping;
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    pos_i[0] = px;
    pos_i[1] = py;
    pos_i[2] = pz;
    vel_i[0] = vx;
    vel_i[1] = vy;
    vel_i[2] = vz;
}
