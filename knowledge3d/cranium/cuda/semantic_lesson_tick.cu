#include "device_functions.cuh"

extern "C" __global__ void semantic_lesson_tick(
    unsigned char* __restrict__ galaxy_table,
    unsigned int galaxy_star_count,
    const unsigned char* __restrict__ lesson_buffer,
    unsigned int lesson_count,
    unsigned char* __restrict__ stats_buffer
) {
    const unsigned int lesson_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (lesson_index >= lesson_count) {
        return;
    }

    const unsigned int lesson_base = lesson_index * LESSON_RECORD_BYTES;
    const unsigned int family_id =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_FAMILY_ID_OFFSET);
    const unsigned int router_index =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_ROUTER_INDEX_OFFSET);
    const unsigned int executor_index =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_EXECUTOR_INDEX_OFFSET);
    const unsigned int validator_index =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_VALIDATOR_INDEX_OFFSET);
    const unsigned int winner_index =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_WINNER_INDEX_OFFSET);
    const unsigned int winner_role =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_WINNER_ROLE_OFFSET);
    const unsigned long long expected_hash =
        *reinterpret_cast<const unsigned long long*>(lesson_buffer + lesson_base + LESSON_EXPECTED_HASH_OFFSET);
    const unsigned long long predicted_hash =
        *reinterpret_cast<const unsigned long long*>(lesson_buffer + lesson_base + LESSON_PREDICTED_HASH_OFFSET);
    const int anti_pattern_signal =
        *reinterpret_cast<const int*>(lesson_buffer + lesson_base + LESSON_ANTI_PATTERN_OFFSET);
    const unsigned int route_depth =
        *reinterpret_cast<const unsigned int*>(lesson_buffer + lesson_base + LESSON_ROUTE_DEPTH_OFFSET);

    float reward = *reinterpret_cast<const float*>(lesson_buffer + lesson_base + LESSON_REWARD_OFFSET);
    if (reward == 0.0f && expected_hash != 0ull && predicted_hash != 0ull) {
        reward = (expected_hash == predicted_hash) ? 1.0f : -1.0f;
    }
    if (reward == 0.0f) {
        return;
    }

    unsigned int* positive_steps =
        reinterpret_cast<unsigned int*>(stats_buffer + LESSON_STATS_POSITIVE_STEPS_OFFSET);
    unsigned int* negative_steps =
        reinterpret_cast<unsigned int*>(stats_buffer + LESSON_STATS_NEGATIVE_STEPS_OFFSET);
    unsigned int* anti_pattern_hits =
        reinterpret_cast<unsigned int*>(stats_buffer + LESSON_STATS_ANTI_PATTERN_HITS_OFFSET);
    float* last_positive_loss =
        reinterpret_cast<float*>(stats_buffer + LESSON_STATS_LAST_POSITIVE_LOSS_OFFSET);
    float* last_negative_loss =
        reinterpret_cast<float*>(stats_buffer + LESSON_STATS_LAST_NEGATIVE_LOSS_OFFSET);

    const float family_scale =
        family_is_math_like(family_id) ? 1.20f : (family_is_question_like(family_id) ? 1.05f : 1.00f);
    const float route_scale = route_depth > 0u ? device_minf(1.60f, 1.0f + (0.08f * static_cast<float>(route_depth - 1u))) : 1.0f;
    const float positive_delta = 0.035f * family_scale * route_scale;
    const float negative_delta = 0.050f * family_scale * route_scale;

    if (reward > 0.0f) {
        atomicAdd(positive_steps, 1u);
        *last_positive_loss = 1.0f - reward;
        const unsigned int positive_targets[4] = {router_index, executor_index, validator_index, winner_index};
        for (int slot = 0; slot < 4; ++slot) {
            const unsigned int star_index = positive_targets[slot];
            if (star_index >= galaxy_star_count) {
                continue;
            }
            atomicAdd(galaxy_write_float_ptr(galaxy_table, star_index, GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET), positive_delta);
            atomicAdd(galaxy_write_float_ptr(galaxy_table, star_index, GALAXY_STAR_SEMANTIC_MASS_OFFSET), 0.005f);
        }
        return;
    }

    atomicAdd(negative_steps, 1u);
    *last_negative_loss = device_absf(reward);
    if (anti_pattern_signal > 0 || winner_role == GALAXY_ROLE_ANTI_PATTERN) {
        atomicAdd(anti_pattern_hits, 1u);
    }
    if (winner_index < galaxy_star_count) {
        atomicAdd(galaxy_write_float_ptr(galaxy_table, winner_index, GALAXY_STAR_REPULSIVE_PRIOR_OFFSET), negative_delta);
        atomicAdd(galaxy_write_float_ptr(galaxy_table, winner_index, GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET), -0.020f);
    }
    if (router_index < galaxy_star_count && router_index != winner_index) {
        atomicAdd(galaxy_write_float_ptr(galaxy_table, router_index, GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET), 0.010f);
    } else if (router_index < galaxy_star_count && route_depth <= 1u) {
        atomicAdd(galaxy_write_float_ptr(galaxy_table, router_index, GALAXY_STAR_REPULSIVE_PRIOR_OFFSET), negative_delta * 0.85f);
    }
}
