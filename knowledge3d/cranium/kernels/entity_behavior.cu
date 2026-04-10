__device__ inline uint32_t bh_self_index(uint32_t instance_id, uint32_t entity_count) {
    if (entity_count == 0u) {
        return 0u;
    }
    return min(instance_id, entity_count - 1u);
}

__device__ inline float3 bh_entity_position(const EntityHotPath* entities, const PhysicsBodySOA* bodies, uint32_t idx) {
    if (entities == nullptr || idx >= g_entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const uint32_t body_id = entities[idx].physics_body_id;
    if (physics_body_valid(bodies, body_id)) {
        return physics_position(bodies, body_id);
    }
    return make_float3(entities[idx].house_x, entities[idx].house_y, entities[idx].house_z);
}

__device__ inline float3 bh_entity_velocity(const EntityHotPath* entities, const PhysicsBodySOA* bodies, uint32_t idx) {
    if (entities == nullptr || idx >= g_entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const uint32_t body_id = entities[idx].physics_body_id;
    if (physics_body_valid(bodies, body_id)) {
        return physics_velocity(bodies, body_id);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline uint32_t bh_perceive_count(
    float radius,
    EntityHotPath* entities,
    uint32_t entity_count,
    uint32_t self_idx,
    uint32_t* scratch,
    uint32_t scratch_capacity,
    float* nearest_dist
) {
    if (entities == nullptr || scratch == nullptr || scratch_capacity == 0u || entity_count == 0u) {
        if (nearest_dist != nullptr) {
            *nearest_dist = 999.0f;
        }
        return 0u;
    }
    const float3 self_pos = make_float3(
        entities[self_idx].house_x,
        entities[self_idx].house_y,
        entities[self_idx].house_z);
    const float radius_sq = radius * radius;
    uint32_t count = 0u;
    float nearest = 1.0e30f;
    for (uint32_t i = 0u; i < entity_count; ++i) {
        if (i == self_idx) {
            continue;
        }
        const float dx = entities[i].house_x - self_pos.x;
        const float dy = entities[i].house_y - self_pos.y;
        const float dz = entities[i].house_z - self_pos.z;
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq > radius_sq) {
            continue;
        }
        if (count + 1u < scratch_capacity) {
            scratch[count + 1u] = i;
        }
        count += 1u;
        const float dist = sqrtf(dist_sq);
        if (dist < nearest) {
            nearest = dist;
        }
    }
    scratch[0] = count;
    entities[self_idx].last_player_dist = count > 0u ? nearest : 999.0f;
    entities[self_idx].awareness = count > 0u ? fminf(1.0f, count / 8.0f) : 0.0f;
    if (nearest_dist != nullptr) {
        *nearest_dist = count > 0u ? nearest : 999.0f;
    }
    return count;
}

__device__ inline float3 bh_seek_force(
    const EntityHotPath* entities,
    const PhysicsBodySOA* bodies,
    uint32_t self_idx,
    uint32_t target_idx,
    float max_speed,
    float slowing_radius
) {
    if (entities == nullptr || self_idx >= g_entity_count || target_idx >= g_entity_count) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 self_pos = bh_entity_position(entities, bodies, self_idx);
    const float3 target_pos = bh_entity_position(entities, bodies, target_idx);
    const float3 delta = physics_vec_sub(target_pos, self_pos);
    const float dist = physics_vec_len(delta);
    if (dist <= 1.0e-5f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 desired_dir = physics_vec_scale(delta, 1.0f / dist);
    float desired_speed = max_speed;
    if (slowing_radius > 0.0f && dist < slowing_radius) {
        desired_speed *= dist / slowing_radius;
    }
    const float3 desired = physics_vec_scale(desired_dir, desired_speed);
    const float3 current = bh_entity_velocity(entities, bodies, self_idx);
    return physics_vec_sub(desired, current);
}

__device__ inline float3 bh_separation_force(
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
        entities[self_idx].house_z);
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    const float radius_sq = radius * radius;
    for (uint32_t idx = 0u; idx < entity_count; ++idx) {
        if (idx == self_idx) {
            continue;
        }
        const float3 other_pos = make_float3(entities[idx].house_x, entities[idx].house_y, entities[idx].house_z);
        const float3 delta = physics_vec_sub(self_pos, other_pos);
        const float dist_sq = physics_vec_dot(delta, delta);
        if (dist_sq <= 1.0e-6f || dist_sq > radius_sq) {
            continue;
        }
        force = physics_vec_add(force, physics_vec_scale(delta, 1.0f / dist_sq));
    }
    return force;
}
