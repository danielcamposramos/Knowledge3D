## 1. EntityHotPath Field Usage Analysis

Based on a static analysis of `entity_hot_path.h`, `trm_step_fused.cu`, and `entity_behavior.cu`, here is the exact read/write status of the `EntityHotPath` fields.

**Critical Finding:** The `trm_step_fused.cu` kernel (the main tick) **does not access any fields**. It passes the pointer to `trm_behavior_phase_stub`, which void-casts all arguments. However, `entity_behavior.cu` contains implemented device functions that *would* access these fields if linked into the tick.

| Field | Type | Status | Access Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `star_table_idx` | `uint32_t` | **UNUSED** | None | Canonical ID, not used in hot-path math. |
| `physics_body_id` | `uint32_t` | **READ** | `bh_entity_position`, `bh_entity_velocity` | Used to dereference Physics SOA for true transform. |
| `behavior_rpn_addr` | `uint64_t` | **UNUSED** | None | Pointer to RPN bytecode. Critical for AI, currently ignored. |
| `house_x/y/z` | `float[3]` | **READ** | `bh_perceive_count`, `bh_seek_force`, `bh_separation_force` | Fallback position if physics body invalid. Primary spatial data. |
| `sleep_state` | `uint8_t` | **UNUSED** | None | LOD/Culling flag. Not checked in behavior logic. |
| `faction` | `uint8_t` | **UNUSED** | None | Team ID. Required for friendly fire checks, currently missing. |
| `ai_tier` | `uint8_t` | **UNUSED** | None | Complexity budget. Not used to gate behavior cost. |
| `perception_flags` | `uint8_t` | **UNUSED** | None | Bitmask for sight/hearing. Ignored in `bh_perceive_count`. |
| `perception_radius` | `float` | **READ** | *Implicit* | `bh_perceive_count` takes radius as arg, should read this field. |
| `last_player_dist` | `float` | **WRITE** | `bh_perceive_count` | **Only written field.** Updated with nearest neighbor dist. |
| `awareness` | `float` | **WRITE** | `bh_perceive_count` | **Only written field.** Calculated as `count / 8.0f`. |
| `blackboard_star_id` | `uint32_t` | **UNUSED** | None | Shared memory key for group AI. Not accessed. |
| `meta_rule_addr` | `uint32_t` | **UNUSED** | None | Pointer to higher-order logic. Ignored. |
| `cranial_origin[3]` | `float[3]` | **UNUSED** | None | Sensor offset. Ignored; logic uses `house_x/y/z`. |
| `_pad` | `float` | **UNUSED** | None | Alignment. |

**Summary:** The system is **90% Stubbed**. Only spatial position (`house_*`) and two perception outputs (`last_player_dist`, `awareness`) are touched by the available behavior primitives. The decision-making fields (`behavior_rpn_addr`, `blackboard`, `faction`) are completely disconnected from the tick loop.

---

## 2. Real Behavior Phase Design (5-Phase Tick)

The current `trm_behavior_phase_stub` is a no-op. A real implementation must replace the void-casts with a structured pipeline that consumes `EntityHotPath` and produces Physics Forces or Events.

**Proposed `trm_behavior_phase_real` Implementation:**

```cuda
__global__ void trm_behavior_phase_real(
    EntityHotPath* __restrict__ entities,
    PhysicsBodySOA* __restrict__ bodies,
    EventQueue* __restrict__ events,
    unsigned int entity_count,
    unsigned int frame_counter
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entity_count) return;
    
    EntityHotPath& self = entities[idx];
    if (self.sleep_state > 0) return; // Culling

    // 1. SENSE (Perception)
    // Replace O(N) loop with Spatial Hash lookup (missing infra)
    float nearest_dist = 999.0f;
    uint32_t perceived_count = bh_perceive_count(
        self.perception_radius, entities, entity_count, idx, 
        scratch_buffer, SCRATCH_CAP, &nearest_dist
    );
    self.last_player_dist = nearest_dist;
    self.awareness = fminf(1.0f, perceived_count / 8.0f);

    // 2. THINK (Reasoning)
    // Evaluate RPN Behavior Tree pointed to by behavior_rpn_addr
    // Inputs: awareness, faction, last_player_dist
    // Output: Intent (Flee, Chase, Idle)
    uint32_t intent = rpn_evaluate(self.behavior_rpn_addr, &self); 

    // 3. DECIDE (Target Selection)
    uint32_t target_idx = NO_TARGET;
    if (intent == INTENT_CHASE && perceived_count > 0) {
        target_idx = find_hostile_target(entities, idx, self.faction);
    }

    // 4. ACT (Steering & Physics)
    float3 force = make_float3(0,0,0);
    if (target_idx != NO_TARGET) {
        force = bh_seek_force(entities, bodies, idx, target_idx, 5.0f, 2.0f);
    } else {
        force = bh_separation_force(entities, entity_count, idx, 1.5f);
    }
    // Apply to Physics SOA (Direct write, not via stub)
    physics_apply_force(bodies, self.physics_body_id, force);

    // 5. LEARN (Blackboard Update)
    // Update long-term memory based on frame success
    if (self.awareness > 0.8f) {
        self.blackboard_star_id = update_threat_map(self.blackboard_star_id, idx);
    }
}
```

**Phase Breakdown:**
1.  **Sense:** Reads `house_x/y/z`, `perception_radius`. Writes `awareness`, `last_player_dist`.
2.  **Think:** Reads `behavior_rpn_addr`, `meta_rule_addr`. Executes bytecode.
3.  **Decide:** Reads `faction`, `ai_tier`. Selects `target_idx`.
4.  **Act:** Reads `physics_body_id`. Writes to `PhysicsBodySOA` (velocity/force).
5.  **Learn:** Writes `blackboard_star_id`. Updates `ai_tier` (difficulty scaling).

---

## 3. Event Queue Design

The `event_queue_ptr` in `trm_step_fused.cu` is currently passed to a stub and ignored. To enable decoupled subsystems (e.g., Physics notifying Behavior of collisions, or Behavior notifying Audio of footsteps), a compact GPU event struct is required.

**Constraint:** ≤32 Bytes.
**Alignment:** 4-byte aligned for coalesced access.

```cuda
#pragma pack(push, 1)
struct GpuEvent {
    uint8_t  event_type;      // 1 byte: 0x01=Damage, 0x02=Perception, 0x03=NavBlock
    uint8_t  priority;        // 1 byte: 0-255 (culling threshold)
    uint16_t source_entity;   // 2 bytes: Entity ID (max 65k entities)
    uint32_t target_entity;   // 4 bytes: Target ID or 0xFFFFFFFF for AO
    float3   position;        // 12 bytes: World space trigger point
    uint32_t timestamp;       // 4 bytes: Frame counter
    float    payload;         // 4 bytes: Damage amt, Sound vol, etc.
    // Total: 28 Bytes. 4 Bytes padding implicit or used for flags.
};
#pragma pack(pop)
static_assert(sizeof(GpuEvent) <= 32, "Event struct exceeds budget");
```

**Event Types Flow:**
1.  **Physics → Behavior:** `CollisionEvent`. Source=Wall, Target=Entity. Payload=ImpactForce. Triggers `bh_separation_force`.
2.  **Behavior → Audio:** `FootstepEvent`. Source=Entity. Position=`house_x/y/z`. Payload=SurfaceType.
3.  **Behavior → Combat:** `DamageEvent`. Source=Attacker, Target=Victim. Payload=HPLoss.
4.  **Perception → AI:** `SightEvent`. Source=SeenEntity, Target=Observer. Position=LastSeenPos.

**Infrastructure Gap:** K3D currently lacks an **Atomic Event Ring Buffer**. The `event_queue_ptr` needs a head/tail index in constant memory to allow lock-free push/pop from kernels.

---

## 4. Kernel Category Mapping (88 PTX vs. 5 Active)

The repository claims 88 PTX kernels, but `trm_step_fused.cu` only actively utilizes the **TRM Recursive Core** (Swarm/Neural) and stubs the rest. Here is the mapping of existing infrastructure to Game Engine Subsystems, highlighting the "Active" vs. "Dead" code.

| Game Subsystem | K3D Infrastructure | Status | Mapping Analysis |
| :--- | :--- | :--- | :--- |
| **Perception** | `bh_perceive_count` (in `entity_behavior.cu`) | **Dead** | Implemented as O(N²) device function. Needs **Spatial Hash Kernel** (likely one of the 83 missing PTX kernels) to be performant. Currently no kernel populates the perception data. |
| **Navigation** | `bh_seek_force`, `bh_separation_force` | **Dead** | Steering behaviors exist but are not called. Missing **Pathfinding Kernel** (A* or Flow Field) for `meta_rule_addr` navigation. |
| **Reasoning** | `behavior_rpn_addr`, `meta_rule_addr` | **Dead** | Pointers exist in struct, but no **RPN Evaluator Kernel** is invoked in `trm_step_fused`. The AI is effectively lobotomized. |
| **Physics** | `physics_body_id`, `trm_physics_phase_stub` | **Stubbed** | `trm_physics_phase_stub` void-casts the SOA pointers. The **Integrator Kernel** (Verlet/Euler) is missing from the active path. |
| **Consolidation** | `trm_recursive_core_device` | **ACTIVE** | This is the only real work being done. It updates `y_new`, `z_new` (TRM state). This handles **Swarm Intelligence** but not individual entity logic. |
| **Rendering** | (Not shown) | **N/A** | No frustum/LOD kernels active in this compute path. |

**The "5 Active" Kernels:**
1.  `trm_step_fused` (Master Dispatch)
2.  `trm_recursive_core_device` (Neural/Swarm Update)
3.  `bh_perceive_count` (If inlined/compiled)
4.  `bh_entity_position` (Helper)
5.  `bh_seek_force` (Helper)

**The "83 Missing" Kernels (Inferred):**
*   **Spatial Partitioning:** `kernel_build_hash_grid`, `kernel_query_neighbors`.
*   **RPN Execution:** `kernel_execute_behavior_tree`.
*   **Physics Integration:** `kernel_integrate_velocities`, `kernel_solve_constraints`.
*   **Event Processing:** `kernel_drain_event_queue`.
*   **Culling:** `kernel_update_sleep_state`.

**Conclusion:** K3D's current `trm_step_fused` path is a **Neural Swarm Simulator**, not a full Game Engine. It updates global TRM state but fails to drive individual entity physics or logic because the Behavior and Physics phases are explicitly stubbed out, despite the logic primitives existing in `entity_behavior.cu`.