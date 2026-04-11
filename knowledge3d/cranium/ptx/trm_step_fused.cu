#define K3D_TRM_DEFINE_TRANSITION_TABLE 1

#include <cuda_runtime.h>

#include "../kernels/physics_body_soa.h"
#include "../cuda/device_functions.cuh"
#include "../cuda/rpn_execute_device.cuh"
#include "../cuda/trm_game_loop.cuh"
#include "../cuda/trm_recursive_core.cuh"

#define TRM_WORKSPACE_FLOATS_PER_ENTITY 4096
#define TRM_INVALID_ENTITY_ID 0xFFFFFFFFu
#define K3D_ACTION_WORDS 72u
#define K3D_ACTION_BYTES 288u
#define K3D_ACTION_NAV_MOVE 0x00u
#define K3D_ACTION_NAV_LOOK 0x01u
#define K3D_ACTION_DIALOGUE 0x02u
#define K3D_ACTION_WRITE_MEM 0x03u
#define K3D_ACTION_UPDATE_TABLET 0x04u
#define K3D_ACTION_NO_ACTION 0xFFu
#define K3D_GALAXY_EMBEDDING_DIMS 64u
#define K3D_GALAXY_STAR_RECORD_BYTES 400u
#define K3D_GALAXY_STAR_FLAGS_OFFSET 272u
#define K3D_GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET 276u
#define K3D_GALAXY_STAR_SELECTION_ROLE_OFFSET 264u
#define K3D_GALAXY_STAR_STAR_HASH_OFFSET 304u
#define K3D_GALAXY_STAR_META_RULE_ADDR_OFFSET 384u
#define K3D_GALAXY_STAR_PROGRAM_LENGTH_OFFSET 392u
#define K3D_GALAXY_STAR_POSITION_OFFSET 360u
#define K3D_GALAXY_STAR_FLAG_ACTIVE 0x01u
#define K3D_INVALID_STAR_INDEX 0xFFFFFFFFu
#define K3D_ACTION_WORD_OFFSET_MUTATION_TYPE 60u
#define K3D_ACTION_WORD_OFFSET_TABLET_DATA 61u
#define K3D_ACTION_WORD_OFFSET_TABLET_RESERVED 67u
#define K3D_TABLET_MUTATION_MATH_RESULT 2u

__device__ __forceinline__ uint32_t trm_read_u32(const unsigned char* base) {
    return *reinterpret_cast<const uint32_t*>(base);
}

__device__ __forceinline__ uint64_t trm_read_u64(const unsigned char* base) {
    return *reinterpret_cast<const uint64_t*>(base);
}

__device__ __forceinline__ unsigned char trm_operator_hint_to_opcode(uint32_t operator_hint) {
    switch (operator_hint) {
        case static_cast<uint32_t>('+'):
            return RPN_OP_ADD;
        case static_cast<uint32_t>('-'):
            return RPN_OP_SUB;
        case static_cast<uint32_t>('*'):
            return RPN_OP_MUL;
        case static_cast<uint32_t>('/'):
            return RPN_OP_DIV;
        case static_cast<uint32_t>('^'):
            return RPN_OP_POW;
        default:
            return 0u;
    }
}

__device__ __forceinline__ float trm_clampf(float value, float lo, float hi) {
    return fminf(hi, fmaxf(lo, value));
}

__device__ __forceinline__ uint32_t trm_float_bits(float value) {
    return __float_as_uint(value);
}

__device__ __forceinline__ uint64_t trm_mix_action_hash(uint32_t goal_star, uint32_t meta_rule_addr) {
    uint64_t value = (static_cast<uint64_t>(goal_star) << 32) | static_cast<uint64_t>(meta_rule_addr);
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

__device__ __forceinline__ float3 trm_vec_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 trm_vec_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 trm_vec_scale(float3 a, float scale) {
    return make_float3(a.x * scale, a.y * scale, a.z * scale);
}

__device__ __forceinline__ float trm_vec_dot(float3 a, float3 b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__device__ __forceinline__ float trm_vec_len(float3 value) {
    return sqrtf(trm_vec_dot(value, value));
}

__device__ __forceinline__ float3 trm_vec_normalize(float3 value) {
    const float length = trm_vec_len(value);
    if (length <= 1.0e-6f) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    return trm_vec_scale(value, 1.0f / length);
}

__device__ __forceinline__ uint32_t trm_part1by2(uint32_t value) {
    value &= 0x000003ffu;
    value = (value ^ (value << 16)) & 0xff0000ffu;
    value = (value ^ (value << 8)) & 0x0300f00fu;
    value = (value ^ (value << 4)) & 0x030c30c3u;
    value = (value ^ (value << 2)) & 0x09249249u;
    return value;
}

__device__ __forceinline__ uint32_t trm_morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t mx = trm_part1by2(x);
    const uint32_t my = trm_part1by2(y);
    const uint32_t mz = trm_part1by2(z);
    return (mx << 2) | (my << 1) | mz;
}

__device__ __forceinline__ uint32_t trm_quantize_axis(float value) {
    const float normalized = trm_clampf((value + 512.0f) / 1024.0f, 0.0f, 1.0f);
    return static_cast<uint32_t>(normalized * 1023.0f);
}

__device__ __forceinline__ uint32_t trm_encode_position_morton(float3 position) {
    return trm_morton_encode_3d(
        trm_quantize_axis(position.x),
        trm_quantize_axis(position.y),
        trm_quantize_axis(position.z)
    );
}

__device__ __forceinline__ float3 trm_entity_position(
    const EntityHotPath* entities,
    const PhysicsBodySOA* bodies,
    uint32_t entity_count,
    uint32_t entity_idx
) {
    if (entities == nullptr || entity_idx >= entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const uint32_t body_id = entities[entity_idx].physics_body_id;
    if (physics_body_valid(bodies, body_id)) {
        return physics_position(bodies, body_id);
    }
    return make_float3(
        entities[entity_idx].house_x,
        entities[entity_idx].house_y,
        entities[entity_idx].house_z
    );
}

__device__ __forceinline__ float3 trm_entity_velocity(
    const EntityHotPath* entities,
    const PhysicsBodySOA* bodies,
    uint32_t entity_count,
    uint32_t entity_idx
) {
    if (entities == nullptr || entity_idx >= entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const uint32_t body_id = entities[entity_idx].physics_body_id;
    if (physics_body_valid(bodies, body_id)) {
        return physics_velocity(bodies, body_id);
    }
    return make_float3(
        entities[entity_idx].motor_output[0],
        entities[entity_idx].motor_output[1],
        entities[entity_idx].motor_output[2]
    );
}

__device__ __forceinline__ void trm_store_motor_output(EntityHotPath* entity, float3 value) {
    if (entity == nullptr) {
        return;
    }
    entity->motor_output[0] = value.x;
    entity->motor_output[1] = value.y;
    entity->motor_output[2] = value.z;
}

__device__ __forceinline__ float3 trm_load_motor_output(const EntityHotPath* entity) {
    if (entity == nullptr) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(entity->motor_output[0], entity->motor_output[1], entity->motor_output[2]);
}

__device__ __forceinline__ float3 trm_gaze_forward(const EntityHotPath* entity) {
    if (entity == nullptr) {
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    const float yaw = entity->gaze_yaw;
    const float pitch = entity->gaze_pitch;
    const float cos_pitch = cosf(pitch);
    return trm_vec_normalize(make_float3(
        sinf(yaw) * cos_pitch,
        sinf(pitch),
        cosf(yaw) * cos_pitch
    ));
}

__device__ __forceinline__ void trm_face_direction(EntityHotPath* entity, float3 direction) {
    if (entity == nullptr) {
        return;
    }
    const float3 norm = trm_vec_normalize(direction);
    entity->gaze_yaw = atan2f(norm.x, norm.z);
    entity->gaze_pitch = asinf(trm_clampf(norm.y, -1.0f, 1.0f));
}

__device__ __forceinline__ void trm_emit_event(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t entity_id,
    uint8_t event_type,
    uint8_t priority,
    uint64_t payload
) {
    GPUEvent event = {};
    event.entity_id = entity_id;
    event.event_type = event_type;
    event.priority = priority;
    event.payload = payload;
    (void)trm_event_queue_push(ring_buffer, head_ptr, tail_ptr, event);
}

__device__ __forceinline__ uint32_t trm_find_goal_entity(
    const EntityHotPath* entities,
    uint32_t entity_count,
    const EntityHotPath* my_entity
) {
    if (entities == nullptr || my_entity == nullptr) {
        return TRM_INVALID_ENTITY_ID;
    }
    if (my_entity->attention_entity_id < entity_count) {
        return my_entity->attention_entity_id;
    }
    if (my_entity->current_goal_star == 0u) {
        return TRM_INVALID_ENTITY_ID;
    }
    for (uint32_t idx = 0u; idx < entity_count; ++idx) {
        if (entities[idx].star_table_idx == my_entity->current_goal_star) {
            return idx;
        }
    }
    return TRM_INVALID_ENTITY_ID;
}

__device__ __forceinline__ float3 trm_galaxy_star_position(
    const unsigned char* galaxy_table,
    uint32_t star_index
) {
    if (galaxy_table == nullptr || star_index == K3D_INVALID_STAR_INDEX) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const uint32_t base = star_index * K3D_GALAXY_STAR_RECORD_BYTES;
    const float* position = reinterpret_cast<const float*>(galaxy_table + base + K3D_GALAXY_STAR_POSITION_OFFSET);
    return make_float3(position[0], position[1], position[2]);
}

__device__ __forceinline__ int trm_galaxy_star_active(
    const unsigned char* galaxy_table,
    uint32_t star_index
) {
    if (galaxy_table == nullptr || star_index == K3D_INVALID_STAR_INDEX) {
        return 0;
    }
    const uint32_t base = star_index * K3D_GALAXY_STAR_RECORD_BYTES;
    const uint32_t flags = *reinterpret_cast<const uint32_t*>(galaxy_table + base + K3D_GALAXY_STAR_FLAGS_OFFSET);
    return (flags & K3D_GALAXY_STAR_FLAG_ACTIVE) != 0u;
}

__device__ __forceinline__ float3 trm_separation_force(
    const EntityHotPath* entities,
    uint32_t entity_count,
    uint32_t self_idx,
    float radius
) {
    if (entities == nullptr || entity_count == 0u || self_idx >= entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 self_pos = make_float3(
        entities[self_idx].house_x,
        entities[self_idx].house_y,
        entities[self_idx].house_z
    );
    const float radius_sq = radius * radius;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t idx = 0u; idx < entity_count; ++idx) {
        if (idx == self_idx) {
            continue;
        }
        const float3 other_pos = make_float3(entities[idx].house_x, entities[idx].house_y, entities[idx].house_z);
        const float3 delta = trm_vec_sub(self_pos, other_pos);
        const float dist_sq = trm_vec_dot(delta, delta);
        if (dist_sq <= 1.0e-6f || dist_sq > radius_sq) {
            continue;
        }
        force = trm_vec_add(force, trm_vec_scale(delta, 1.0f / dist_sq));
    }
    return force;
}

__device__ __forceinline__ float3 trm_seek_force(
    const EntityHotPath* entities,
    const PhysicsBodySOA* bodies,
    uint32_t entity_count,
    uint32_t self_idx,
    uint32_t target_idx,
    float max_speed,
    float slowing_radius
) {
    if (entities == nullptr || self_idx >= entity_count || target_idx >= entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 self_pos = trm_entity_position(entities, bodies, entity_count, self_idx);
    const float3 target_pos = trm_entity_position(entities, bodies, entity_count, target_idx);
    const float3 delta = trm_vec_sub(target_pos, self_pos);
    const float dist = trm_vec_len(delta);
    if (dist <= 1.0e-5f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 desired_dir = trm_vec_scale(delta, 1.0f / dist);
    float desired_speed = max_speed;
    if (slowing_radius > 0.0f && dist < slowing_radius) {
        desired_speed *= dist / slowing_radius;
    }
    const float3 desired = trm_vec_scale(desired_dir, desired_speed);
    const float3 current = trm_entity_velocity(entities, bodies, entity_count, self_idx);
    return trm_vec_sub(desired, current);
}

__device__ __forceinline__ void trm_perceiving_phase(
    EntityHotPath* entity_hot_paths,
    EntityHotPath* my_entity,
    const PhysicsBodySOA* bodies,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t owner_entity_id,
    uint32_t entity_count,
    const unsigned char* galaxy_table,
    uint32_t galaxy_star_count,
    int tid
) {
    if (tid != 0 || my_entity == nullptr || entity_hot_paths == nullptr || owner_entity_id >= entity_count) {
        return;
    }

    const float3 self_pos = trm_entity_position(entity_hot_paths, bodies, entity_count, owner_entity_id);
    const float3 forward = trm_gaze_forward(my_entity);
    const float cos_fov = cosf(trm_clampf(my_entity->gaze_fov, 0.05f, 1.55f));
    const uint32_t self_morton = trm_encode_position_morton(self_pos);

    uint32_t perceived_count = 0u;
    uint32_t best_idx = TRM_INVALID_ENTITY_ID;
    uint32_t best_star_idx = K3D_INVALID_STAR_INDEX;
    float best_score = -1.0f;
    float best_dist = 999.0f;

    for (uint32_t idx = 0u; idx < entity_count; ++idx) {
        if (idx == owner_entity_id) {
            continue;
        }
        const float3 other_pos = trm_entity_position(entity_hot_paths, bodies, entity_count, idx);
        const float3 delta = trm_vec_sub(other_pos, self_pos);
        const float dist = trm_vec_len(delta);
        if (dist <= 1.0e-5f || dist > my_entity->perception_radius) {
            continue;
        }
        const float3 direction = trm_vec_scale(delta, 1.0f / dist);
        const float align = trm_vec_dot(direction, forward);
        if (align < cos_fov) {
            continue;
        }

        perceived_count += 1u;
        const uint32_t other_morton = trm_encode_position_morton(other_pos);
        const float morton_novelty = 1.0f + (static_cast<float>(__popc(self_morton ^ other_morton)) * (1.0f / 64.0f));
        const float goal_bias = (
            my_entity->current_goal_star != 0u
            && entity_hot_paths[idx].star_table_idx == my_entity->current_goal_star
        ) ? 1.25f : 1.0f;
        const float score = goal_bias * morton_novelty * align / (1.0f + dist);
        if (score > best_score) {
            best_score = score;
            best_dist = dist;
            best_idx = idx;
        }
    }

    for (uint32_t star_idx = 0u; star_idx < galaxy_star_count; ++star_idx) {
        if (!trm_galaxy_star_active(galaxy_table, star_idx)) {
            continue;
        }
        const float3 star_pos = trm_galaxy_star_position(galaxy_table, star_idx);
        const float3 delta = trm_vec_sub(star_pos, self_pos);
        const float dist = trm_vec_len(delta);
        if (dist <= 1.0e-5f || dist > my_entity->perception_radius) {
            continue;
        }
        const float3 direction = trm_vec_scale(delta, 1.0f / dist);
        const float align = trm_vec_dot(direction, forward);
        if (align < cos_fov) {
            continue;
        }
        perceived_count += 1u;
        const uint32_t star_morton = trm_encode_position_morton(star_pos);
        const float morton_novelty = 1.0f + (static_cast<float>(__popc(self_morton ^ star_morton)) * (1.0f / 64.0f));
        const float score = 1.10f * morton_novelty * align / (1.0f + dist);
        if (score > best_score) {
            best_score = score;
            best_dist = dist;
            best_idx = TRM_INVALID_ENTITY_ID;
            best_star_idx = star_idx;
        }
    }

    my_entity->last_player_dist = perceived_count > 0u ? best_dist : 999.0f;
    my_entity->awareness = perceived_count > 0u
        ? fminf(1.0f, static_cast<float>(perceived_count) / 8.0f)
        : fmaxf(0.0f, my_entity->awareness - 0.02f);

    if (best_idx != TRM_INVALID_ENTITY_ID) {
        my_entity->attention_entity_id = best_idx;
        my_entity->current_goal_star = entity_hot_paths[best_idx].star_table_idx;
        my_entity->blackboard_star_id = entity_hot_paths[best_idx].star_table_idx;
        trm_emit_event(
            ring_buffer,
            head_ptr,
            tail_ptr,
            owner_entity_id,
            TRM_EVENT_INTERNAL,
            32u,
            static_cast<uint64_t>(my_entity->current_goal_star)
        );
    } else if (best_star_idx != K3D_INVALID_STAR_INDEX) {
        my_entity->attention_entity_id = TRM_INVALID_ENTITY_ID;
        my_entity->current_goal_star = best_star_idx;
        my_entity->blackboard_star_id = best_star_idx;
        trm_emit_event(
            ring_buffer,
            head_ptr,
            tail_ptr,
            owner_entity_id,
            TRM_EVENT_INTERNAL,
            32u,
            static_cast<uint64_t>(best_star_idx)
        );
    } else {
        my_entity->attention_entity_id = TRM_INVALID_ENTITY_ID;
        my_entity->current_goal_star = 0u;
        trm_store_motor_output(my_entity, make_float3(0.0f, 0.0f, 0.0f));
    }
}

__device__ __forceinline__ void trm_navigating_phase(
    EntityHotPath* entity_hot_paths,
    EntityHotPath* my_entity,
    const PhysicsBodySOA* bodies,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t owner_entity_id,
    uint32_t entity_count,
    int tid
) {
    if (tid != 0 || my_entity == nullptr || entity_hot_paths == nullptr || owner_entity_id >= entity_count) {
        return;
    }

    const uint32_t target_idx = trm_find_goal_entity(entity_hot_paths, entity_count, my_entity);
    if (target_idx == TRM_INVALID_ENTITY_ID || target_idx == owner_entity_id) {
        trm_store_motor_output(my_entity, make_float3(0.0f, 0.0f, 0.0f));
        return;
    }

    const float3 seek = trm_seek_force(
        entity_hot_paths,
        bodies,
        entity_count,
        owner_entity_id,
        target_idx,
        3.0f,
        6.0f
    );
    const float3 separation = trm_separation_force(entity_hot_paths, entity_count, owner_entity_id, 3.0f);
    float3 steering = trm_vec_add(seek, trm_vec_scale(separation, 0.35f));
    const float magnitude = trm_vec_len(steering);
    if (magnitude > 3.0f) {
        steering = trm_vec_scale(steering, 3.0f / magnitude);
    }
    trm_store_motor_output(my_entity, steering);

    const float3 self_pos = trm_entity_position(entity_hot_paths, bodies, entity_count, owner_entity_id);
    const float3 target_pos = trm_entity_position(entity_hot_paths, bodies, entity_count, target_idx);
    const float3 delta = trm_vec_sub(target_pos, self_pos);
    trm_face_direction(my_entity, delta);

    if (trm_vec_len(delta) <= 1.0f) {
        trm_emit_event(
            ring_buffer,
            head_ptr,
            tail_ptr,
            owner_entity_id,
            TRM_EVENT_TIMER,
            48u,
            static_cast<uint64_t>(my_entity->current_goal_star)
        );
    }
}

__device__ __forceinline__ void trm_acting_phase(
    EntityHotPath* entity_hot_paths,
    EntityHotPath* my_entity,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t owner_entity_id,
    uint32_t entity_count,
    int tid
) {
    if (tid != 0 || my_entity == nullptr || entity_hot_paths == nullptr || owner_entity_id >= entity_count) {
        return;
    }

    const float gain = 1.0f + (0.05f * static_cast<float>(my_entity->meta_rule_addr & 0x7u));
    float3 motor = trm_load_motor_output(my_entity);
    motor = trm_vec_scale(motor, gain);
    const float magnitude = trm_vec_len(motor);
    if (magnitude > 4.0f) {
        motor = trm_vec_scale(motor, 4.0f / magnitude);
    }
    trm_store_motor_output(my_entity, motor);
    my_entity->awareness = fminf(1.0f, my_entity->awareness + (0.05f * gain));
    if (my_entity->current_goal_star != 0u) {
        my_entity->blackboard_star_id = my_entity->current_goal_star;
    }

    trm_emit_event(
        ring_buffer,
        head_ptr,
        tail_ptr,
        owner_entity_id,
        TRM_EVENT_INTERNAL,
        16u,
        static_cast<uint64_t>(my_entity->meta_rule_addr)
    );
}

__device__ __forceinline__ void trm_phase2_physics_step(
    EntityHotPath* my_entity,
    const void* physics_soa_ptr,
    const void* contact_soa_ptr,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t owner_entity_id,
    float delta_time
) {
    if (my_entity == nullptr || delta_time <= 0.0f) {
        return;
    }

    const PhysicsBodySOA* bodies_ro = static_cast<const PhysicsBodySOA*>(physics_soa_ptr);
    PhysicsBodySOA* bodies = const_cast<PhysicsBodySOA*>(bodies_ro);
    const ContactManifoldSOA* manifold = static_cast<const ContactManifoldSOA*>(contact_soa_ptr);
    const float3 desired_velocity = trm_load_motor_output(my_entity);

    if (physics_body_valid(bodies, my_entity->physics_body_id) && !physics_body_is_static(bodies, my_entity->physics_body_id)) {
        const uint32_t body_id = my_entity->physics_body_id;
        float3 position = physics_position(bodies, body_id);
        float3 velocity = desired_velocity;
        position = trm_vec_add(position, trm_vec_scale(velocity, delta_time));
        physics_store_position(bodies, body_id, position);
        physics_store_velocity(bodies, body_id, velocity);
        physics_store_sleep_energy(bodies, body_id, 0.5f * trm_vec_dot(velocity, velocity));
        physics_mark_dirty(bodies, body_id);
        my_entity->house_x = position.x;
        my_entity->house_y = position.y;
        my_entity->house_z = position.z;
    } else {
        my_entity->house_x += desired_velocity.x * delta_time;
        my_entity->house_y += desired_velocity.y * delta_time;
        my_entity->house_z += desired_velocity.z * delta_time;
    }

    if (manifold != nullptr) {
        const uint32_t contact_count = manifold->write_head < manifold->capacity
            ? manifold->write_head
            : manifold->capacity;
        for (uint32_t idx = 0u; idx < contact_count; ++idx) {
            if (manifold->penetration_depth[idx] <= 0.0f) {
                continue;
            }
            const uint32_t body_a = manifold->body_a_id[idx];
            const uint32_t body_b = manifold->body_b_id[idx];
            if (body_a != my_entity->physics_body_id && body_b != my_entity->physics_body_id) {
                continue;
            }
            const uint32_t other_body = body_a == my_entity->physics_body_id ? body_b : body_a;
            trm_emit_event(
                ring_buffer,
                head_ptr,
                tail_ptr,
                owner_entity_id,
                TRM_EVENT_COLLISION,
                64u,
                static_cast<uint64_t>(other_body)
            );
            break;
        }
    }
}

__device__ __forceinline__ uint32_t trm_action_type_from_state(
    const EntityHotPath* entity,
    uint8_t current_state
) {
    switch (current_state) {
        case TRM_STATE_NAVIGATING:
            return K3D_ACTION_NAV_MOVE;
        case TRM_STATE_PERCEIVING:
            return K3D_ACTION_NAV_LOOK;
        case TRM_STATE_ACTING:
            if (entity == nullptr) {
                return K3D_ACTION_NO_ACTION;
            }
            switch (entity->meta_rule_addr & 0x3u) {
                case 0u:
                    return K3D_ACTION_DIALOGUE;
                case 1u:
                    return K3D_ACTION_WRITE_MEM;
                case 2u:
                    return K3D_ACTION_UPDATE_TABLET;
                default:
                    return K3D_ACTION_NAV_MOVE;
            }
        default:
            return K3D_ACTION_NO_ACTION;
    }
}

__device__ __forceinline__ void trm_write_action_nav_block(
    uint32_t* slot,
    const EntityHotPath* entity,
    uint32_t action_type,
    float awareness,
    float speed
) {
    const float3 position = entity != nullptr
        ? make_float3(entity->house_x, entity->house_y, entity->house_z)
        : make_float3(0.0f, 0.0f, 0.0f);
    const float3 motor = trm_load_motor_output(entity);
    const float3 gaze = trm_gaze_forward(entity);
    const float3 direction = action_type == K3D_ACTION_NAV_LOOK
        ? gaze
        : (speed > 1.0e-6f ? trm_vec_scale(motor, 1.0f / speed) : gaze);

    slot[4] = trm_float_bits(position.x);
    slot[5] = trm_float_bits(position.y);
    slot[6] = trm_float_bits(position.z);
    slot[7] = trm_float_bits(direction.x);
    slot[8] = trm_float_bits(direction.y);
    slot[9] = trm_float_bits(direction.z);
    slot[10] = trm_float_bits(speed);
    slot[11] = entity != nullptr ? entity->current_goal_star : 0u;
    slot[12] = trm_float_bits(awareness);
}

__device__ __forceinline__ void trm_emit_action_buffer(
    void* __restrict__ action_buffer_out,
    const EntityHotPath* entity,
    uint32_t entity_idx,
    uint8_t current_state,
    int tid,
    int stride
) {
    if (action_buffer_out == nullptr) {
        return;
    }

    uint32_t* slot = static_cast<uint32_t*>(action_buffer_out) + (entity_idx * K3D_ACTION_WORDS);
    for (uint32_t word = static_cast<uint32_t>(tid); word < K3D_ACTION_WORDS; word += static_cast<uint32_t>(stride)) {
        slot[word] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        const float awareness = entity != nullptr ? trm_clampf(entity->awareness, 0.0f, 1.0f) : 0.0f;
        const float3 motor = trm_load_motor_output(entity);
        const float speed = trm_vec_len(motor);
        const float curiosity = trm_clampf(speed + awareness, 0.0f, 1.0f);
        const uint32_t action_type = trm_action_type_from_state(entity, current_state);
        const uint32_t goal_star = entity != nullptr ? entity->current_goal_star : 0u;
        const uint32_t meta_rule_addr = entity != nullptr ? entity->meta_rule_addr : 0u;

        slot[0] = action_type;
        slot[1] = trm_float_bits(awareness);
        slot[2] = trm_float_bits(curiosity);
        slot[3] = static_cast<uint32_t>(current_state) | ((meta_rule_addr & 0xFFu) << 8);
        trm_write_action_nav_block(slot, entity, action_type, awareness, speed);

        if (action_type == K3D_ACTION_DIALOGUE) {
            // dialogue_token_ids starts at byte 76 (word 19) as uint16[32].
            slot[19] = goal_star & 0xFFFFu;
            slot[35] = 1u; // dialogue_length
            slot[37] = trm_float_bits(awareness); // dialogue_thinking_score
        } else if (action_type == K3D_ACTION_WRITE_MEM) {
            const uint64_t hash = trm_mix_action_hash(goal_star, meta_rule_addr);
            slot[44] = static_cast<uint32_t>(hash & 0xFFFFFFFFULL);
            slot[45] = static_cast<uint32_t>(hash >> 32);
            slot[46] = goal_star;
            slot[47] = trm_float_bits(awareness);
            slot[48] = entity != nullptr ? trm_float_bits(entity->house_x) : 0u;
            slot[49] = entity != nullptr ? trm_float_bits(entity->house_y) : 0u;
            slot[50] = entity != nullptr ? trm_float_bits(entity->house_z) : 0u;
            slot[51] = trm_float_bits(awareness);
        } else if (action_type == K3D_ACTION_UPDATE_TABLET) {
            slot[60] = meta_rule_addr >> 2;
            slot[61] = goal_star;
            slot[62] = static_cast<uint32_t>(current_state);
            slot[63] = trm_float_bits(awareness);
        }

        __threadfence_system();
    }
    __syncthreads();
}

__device__ __forceinline__ void trm_write_query_tablet_result(
    void* __restrict__ action_buffer_out,
    uint32_t entity_idx,
    int result_value,
    uint32_t top_star_idx,
    uint64_t top_star_hash,
    uint32_t program_length
) {
    if (action_buffer_out == nullptr) {
        return;
    }
    uint32_t* slot = static_cast<uint32_t*>(action_buffer_out) + (entity_idx * K3D_ACTION_WORDS);
    slot[0] = K3D_ACTION_UPDATE_TABLET;
    slot[K3D_ACTION_WORD_OFFSET_MUTATION_TYPE] = K3D_TABLET_MUTATION_MATH_RESULT;
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 0u] = static_cast<uint32_t>(result_value);
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 1u] = top_star_idx;
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 2u] = static_cast<uint32_t>(top_star_hash & 0xFFFFFFFFULL);
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 3u] = static_cast<uint32_t>(top_star_hash >> 32);
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 4u] = program_length;
    slot[K3D_ACTION_WORD_OFFSET_TABLET_DATA + 5u] = 1u;
    __threadfence_system();
}

__device__ __forceinline__ void trm_sleep_phase(
    EntityHotPath* entity_hot_path,
    int tid,
    int stride,
    float* y_new,
    float* z_new,
    float delta_time
) {
    if (y_new != nullptr && z_new != nullptr) {
        for (int index = tid; index < GPU_TASK_TRM_DIMS; index += stride) {
            y_new[index] *= 0.98f;
            z_new[index] *= 0.98f;
        }
    }
    __syncthreads();
    if (tid == 0 && entity_hot_path != nullptr) {
        entity_hot_path->awareness = fmaxf(0.0f, entity_hot_path->awareness - (0.15f * delta_time));
    }
}

__device__ __forceinline__ void trm_idle_phase(
    EntityHotPath* entity_hot_path,
    int tid,
    float delta_time
) {
    if (tid == 0 && entity_hot_path != nullptr) {
        entity_hot_path->awareness = fmaxf(0.0f, entity_hot_path->awareness - (0.05f * delta_time));
    }
}

__device__ __forceinline__ void trm_reasoning_gate(
    const float* __restrict__ q,
    float* __restrict__ workspace,
    int tid,
    int* __restrict__ converged
) {
    float* chain_states = workspace;
    float* swarm_output = chain_states + (GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS);
    float* resonance_scores = swarm_output + GPU_TASK_EMBED_DIMS;

    if (tid == 0) {
        *converged = 0;
    }
    __syncthreads();

    nine_chain_swarm_device(
        q,
        chain_states,
        swarm_output,
        resonance_scores,
        2
    );
    __syncthreads();

    if (tid == 0) {
        *converged = halting_gate_device(
            resonance_scores,
            GPU_TASK_NUM_CHAINS,
            0.04f,
            0.04f,
            0.7f
        );
    }
    __syncthreads();
}

__device__ __forceinline__ void trm_reasoning_phase(
    const float* __restrict__ q,
    float* __restrict__ y_new,
    float* __restrict__ z_new,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    EntityHotPath* entity_hot_path,
    int tid,
    int stride,
    int max_steps,
    float epsilon,
    int* __restrict__ steps_taken,
    float* __restrict__ drift_value,
    int* __restrict__ converged
) {
    trm_reasoning_gate(q, workspace, tid, converged);
    trm_recursive_core_device(
        q,
        y_new,
        z_new,
        W1,
        W2,
        W3,
        W4,
        workspace,
        tid,
        stride,
        max_steps,
        epsilon,
        steps_taken,
        drift_value,
        steps_taken,
        drift_value
    );
    __syncthreads();
    if (tid == 0 && entity_hot_path != nullptr) {
        entity_hot_path->awareness = (*converged != 0)
            ? fminf(1.0f, entity_hot_path->awareness + 0.08f)
            : fminf(1.0f, entity_hot_path->awareness + 0.03f);
    }
}

__device__ __forceinline__ void trm_query_fast_lane_phase(
    const float* __restrict__ q,
    float* __restrict__ y_new,
    float* __restrict__ z_new,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    int tid,
    int stride,
    int max_steps,
    float epsilon,
    int* __restrict__ steps_taken,
    float* __restrict__ drift_value
) {
    trm_recursive_core_device(
        q,
        y_new,
        z_new,
        W1,
        W2,
        W3,
        W4,
        workspace,
        tid,
        stride,
        max_steps,
        epsilon,
        steps_taken,
        drift_value,
        steps_taken,
        drift_value
    );
    __syncthreads();
}

extern "C" __global__ void trm_step_fused(
    const float* __restrict__ q,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ z_new,
    float* __restrict__ y_new,
    float* __restrict__ workspace,
    const void* __restrict__ physics_soa_ptr,
    const void* __restrict__ contact_soa_ptr,
    unsigned int body_count,
    unsigned int solver_iterations,
    void* __restrict__ ring_buffer_ptr,
    uint32_t* __restrict__ head_ptr,
    uint32_t* __restrict__ tail_ptr,
    TRMStateMachine* __restrict__ state_machine_ptr,
    void* __restrict__ entity_hot_path_ptr,
    unsigned int entity_count,
    float delta_time,
    unsigned long long tick,
    int max_steps,
    float epsilon,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out,
    const void* __restrict__ galaxy_table_ptr,
    unsigned int galaxy_star_count,
    void* __restrict__ action_buffer_out,
    const void* __restrict__ program_table_ptr,
    const void* __restrict__ action_buffer_in_ptr
) {
    const uint32_t entity_idx = static_cast<uint32_t>(blockIdx.x);
    if (entity_idx >= entity_count) {
        return;
    }

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    __shared__ int steps_taken;
    __shared__ float drift_value;
    __shared__ int converged;
    __shared__ uint8_t current_state;
    __shared__ uint8_t emission_state;
    __shared__ uint32_t query_top_star_idx;
    __shared__ uint64_t query_top_star_hash;
    __shared__ uint32_t query_program_length;
    __shared__ int query_result_value;
    __shared__ uint32_t query_result_valid;
    const bool has_latent_lane = (entity_idx == 0u);

    if (tid == 0) {
        steps_taken = 0;
        drift_value = 0.0f;
        converged = 0;
        current_state = TRM_STATE_IDLE;
        emission_state = TRM_STATE_IDLE;
        query_top_star_idx = K3D_INVALID_STAR_INDEX;
        query_top_star_hash = 0ull;
        query_program_length = 0u;
        query_result_value = 0;
        query_result_valid = 0u;
    }
    __syncthreads();

    if (has_latent_lane && y != nullptr && z != nullptr && y_new != nullptr && z_new != nullptr) {
        for (int index = tid; index < GPU_TASK_TRM_DIMS; index += stride) {
            y_new[index] = y[index];
            z_new[index] = z[index];
        }
    }
    __syncthreads();

    EntityHotPath* entity_hot_paths = static_cast<EntityHotPath*>(entity_hot_path_ptr);
    GPUEvent* ring_buffer = static_cast<GPUEvent*>(ring_buffer_ptr);
    const unsigned char* galaxy_table = static_cast<const unsigned char*>(galaxy_table_ptr);
    const unsigned char* program_table = static_cast<const unsigned char*>(program_table_ptr);
    const uint32_t* action_buffer_in = static_cast<const uint32_t*>(action_buffer_in_ptr);
    EntityHotPath* my_entity = entity_hot_paths != nullptr ? &entity_hot_paths[entity_idx] : nullptr;
    TRMStateMachine* my_state_machine = state_machine_ptr != nullptr ? &state_machine_ptr[entity_idx] : nullptr;
    float* my_workspace = workspace != nullptr ? workspace + (entity_idx * TRM_WORKSPACE_FLOATS_PER_ENTITY) : nullptr;
    int* my_steps_out = steps_out != nullptr ? &steps_out[entity_idx] : nullptr;
    float* my_drift_out = drift_out != nullptr ? &drift_out[entity_idx] : nullptr;
    const PhysicsBodySOA* physics_bodies = static_cast<const PhysicsBodySOA*>(physics_soa_ptr);

    if (tid == 0 && my_state_machine != nullptr) {
        trm_state_machine_step_device(
            my_entity,
            my_state_machine,
            ring_buffer,
            head_ptr,
            tail_ptr,
            delta_time,
            static_cast<uint64_t>(tick)
        );
        current_state = my_state_machine->current_state;
        emission_state = current_state;
    }
    __syncthreads();

    // BEHAVIOR_PHASE: lifecycle state gates the per-tick reasoning/action slices.
    switch (current_state) {
        case TRM_STATE_SLEEP:
            trm_sleep_phase(my_entity, tid, stride, has_latent_lane ? y_new : nullptr, has_latent_lane ? z_new : nullptr, delta_time);
            break;
        case TRM_STATE_IDLE:
            trm_idle_phase(my_entity, tid, delta_time);
            break;
        case TRM_STATE_REASONING:
            if (has_latent_lane) {
                trm_reasoning_phase(
                    q,
                    y_new,
                    z_new,
                    W1,
                    W2,
                    W3,
                    W4,
                    my_workspace,
                    my_entity,
                    tid,
                    stride,
                    max_steps,
                    epsilon,
                    &steps_taken,
                    &drift_value,
                    &converged
                );
            }
            break;
        case TRM_STATE_HANDLING_QUERY:
            if (has_latent_lane) {
                trm_query_fast_lane_phase(
                    q,
                    y_new,
                    z_new,
                    W1,
                    W2,
                    W3,
                    W4,
                    my_workspace,
                    tid,
                    stride,
                    max_steps,
                    epsilon,
                    &steps_taken,
                    &drift_value
                );
            }
            __syncthreads();
            if (
                tid == 0
                && has_latent_lane
                && galaxy_table != nullptr
                && galaxy_star_count > 0u
                && program_table != nullptr
                && action_buffer_in != nullptr
            ) {
                const int operand_0 = static_cast<int>(action_buffer_in[K3D_ACTION_WORD_OFFSET_TABLET_RESERVED + 0u]);
                const int operand_1 = static_cast<int>(action_buffer_in[K3D_ACTION_WORD_OFFSET_TABLET_RESERVED + 1u]);
                const uint32_t operand_count = action_buffer_in[K3D_ACTION_WORD_OFFSET_TABLET_RESERVED + 2u];
                const uint32_t operator_hint = action_buffer_in[K3D_ACTION_WORD_OFFSET_TABLET_RESERVED + 3u];
                const unsigned char required_opcode = trm_operator_hint_to_opcode(operator_hint);
                float y_norm_sq = 0.0f;
                for (uint32_t dim = 0u; dim < K3D_GALAXY_EMBEDDING_DIMS; ++dim) {
                    const float value = y_new[dim];
                    y_norm_sq += value * value;
                }
                const float y_norm = sqrtf(y_norm_sq);
                if (operand_count >= 2u && y_norm > 1.0e-8f) {
                    float best_score = -1.0e30f;
                    uint32_t best_index = K3D_INVALID_STAR_INDEX;
                    uint64_t best_hash = 0ull;
                    uint32_t best_program_length = 0u;
                    uint32_t best_meta_rule_addr = 0u;
                    for (uint32_t star_index = 0u; star_index < galaxy_star_count; ++star_index) {
                        const uint32_t base = star_index * K3D_GALAXY_STAR_RECORD_BYTES;
                        const unsigned char* record = galaxy_table + base;
                        const uint32_t flags = trm_read_u32(record + K3D_GALAXY_STAR_FLAGS_OFFSET);
                        if ((flags & K3D_GALAXY_STAR_FLAG_ACTIVE) == 0u) {
                            continue;
                        }
                        if (trm_read_u32(record + K3D_GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET) == 0u) {
                            continue;
                        }
                        if (trm_read_u32(record + K3D_GALAXY_STAR_SELECTION_ROLE_OFFSET) != GALAXY_ROLE_EXECUTOR) {
                            continue;
                        }
                        const uint32_t meta_rule_addr = trm_read_u32(record + K3D_GALAXY_STAR_META_RULE_ADDR_OFFSET);
                        const uint32_t program_length = trm_read_u32(record + K3D_GALAXY_STAR_PROGRAM_LENGTH_OFFSET);
                        if (meta_rule_addr == 0u || program_length == 0u) {
                            continue;
                        }
                        if (required_opcode != 0u) {
                            const unsigned char opcode = program_table[meta_rule_addr + 4u + 2u];
                            if (opcode != required_opcode) {
                                continue;
                            }
                        }
                        const float* embedding = reinterpret_cast<const float*>(record);
                        float dot = 0.0f;
                        float star_norm_sq = 0.0f;
                        for (uint32_t dim = 0u; dim < K3D_GALAXY_EMBEDDING_DIMS; ++dim) {
                            const float value = embedding[dim];
                            dot += y_new[dim] * value;
                            star_norm_sq += value * value;
                        }
                        const float star_norm = sqrtf(star_norm_sq);
                        if (star_norm <= 1.0e-8f) {
                            continue;
                        }
                        const float score = dot / (y_norm * star_norm);
                        if (score > best_score || (score == best_score && star_index < best_index)) {
                            best_score = score;
                            best_index = star_index;
                            best_hash = trm_read_u64(record + K3D_GALAXY_STAR_STAR_HASH_OFFSET);
                            best_program_length = program_length;
                            best_meta_rule_addr = meta_rule_addr;
                        }
                    }
                    if (best_index != K3D_INVALID_STAR_INDEX) {
                        int result_value = 0;
                        const int ok = rpn_execute_device(
                            program_table,
                            best_meta_rule_addr + 4u,
                            best_program_length,
                            operand_0,
                            operand_1,
                            &result_value
                        );
                        if (ok != 0) {
                            query_top_star_idx = best_index;
                            query_top_star_hash = best_hash;
                            query_program_length = best_program_length;
                            query_result_value = result_value;
                            query_result_valid = 1u;
                        }
                    }
                }
            }
            __syncthreads();
            if (tid == 0 && my_state_machine != nullptr) {
                my_state_machine->deferred_event_mask &= ~TRM_DEFERRED_QUERY_POP;
                my_state_machine->interrupt_priority_level = 0u;
                trm_state_pop(*my_state_machine, static_cast<uint64_t>(tick));
                current_state = my_state_machine->current_state;
                if (my_entity != nullptr) {
                    my_entity->sleep_state = current_state;
                }
                __threadfence();
            }
            break;
        case TRM_STATE_PERCEIVING:
            trm_perceiving_phase(
                entity_hot_paths,
                my_entity,
                physics_bodies,
                ring_buffer,
                head_ptr,
                tail_ptr,
                entity_idx,
                entity_count,
                galaxy_table,
                galaxy_star_count,
                tid
            );
            break;
        case TRM_STATE_NAVIGATING:
            trm_navigating_phase(
                entity_hot_paths,
                my_entity,
                physics_bodies,
                ring_buffer,
                head_ptr,
                tail_ptr,
                entity_idx,
                entity_count,
                tid
            );
            break;
        case TRM_STATE_ACTING:
            trm_acting_phase(
                entity_hot_paths,
                my_entity,
                ring_buffer,
                head_ptr,
                tail_ptr,
                entity_idx,
                entity_count,
                tid
            );
            break;
        default:
            break;
    }

    __syncthreads();
    if (tid == 0) {
        // PHYSICS_PHASE: consume motor output and surface collisions on the GPU tick path.
        if (current_state != TRM_STATE_SLEEP) {
            trm_phase2_physics_step(
                my_entity,
                physics_soa_ptr,
                contact_soa_ptr,
                ring_buffer,
                head_ptr,
                tail_ptr,
                entity_idx,
                delta_time
            );
        }
    }
    __syncthreads();

    trm_emit_action_buffer(action_buffer_out, my_entity, entity_idx, emission_state, tid, stride);

    __syncthreads();
    if (tid == 0 && query_result_valid != 0u) {
        trm_write_query_tablet_result(
            action_buffer_out,
            entity_idx,
            query_result_value,
            query_top_star_idx,
            query_top_star_hash,
            query_program_length
        );
    }
    __syncthreads();
    if (tid == 0) {
        if (steps_out != nullptr) {
            if (my_steps_out != nullptr) {
                *my_steps_out = steps_taken;
            }
        }
        if (drift_out != nullptr) {
            if (my_drift_out != nullptr) {
                *my_drift_out = drift_value;
            }
        }
        if (my_entity != nullptr) {
            my_entity->sleep_state = current_state;
        }
        if (my_state_machine != nullptr) {
            my_state_machine->last_tick = static_cast<uint32_t>(tick & 0xFFFFFFFFu);
        }
    }
}
