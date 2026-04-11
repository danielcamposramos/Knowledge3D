#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#pragma pack(push, 1)
// GENERATED at boot from MeaningCentricStar entries with meaning_class == "entity".
// This is a compact hot-path projection for the BEHAVIOR_PHASE tick, not the
// canonical entity representation.
struct EntityHotPath {
    uint32_t star_table_idx;
    uint32_t physics_body_id;
    uint64_t behavior_rpn_addr;

    float house_x;
    float house_y;
    float house_z;

    uint8_t sleep_state;
    uint8_t faction;
    uint8_t ai_tier;
    uint8_t perception_flags;

    float perception_radius;
    float last_player_dist;
    float awareness;

    uint32_t blackboard_star_id;
    uint32_t meta_rule_addr;

    float cranial_origin[3];
    float gaze_yaw;
    float gaze_pitch;
    float gaze_fov;
    uint32_t attention_entity_id;
    float motor_output[3];
    uint32_t current_goal_star;
};
#pragma pack(pop)

static_assert(sizeof(EntityHotPath) == 96, "EntityHotPath layout mismatch");

extern "C" {
extern __device__ __constant__ unsigned long long g_entity_hot_path_ptr;
extern __device__ __constant__ unsigned int g_entity_count;
}
