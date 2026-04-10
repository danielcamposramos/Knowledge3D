# K3D RPN Reality Engine: Always-On Living AI Entities

## Executive Summary

The K3D RPN Reality Engine introduces a paradigm shift from traditional game NPC AI systems. Instead of transient Python objects spawned near player FOV, entities are **persistent Galaxy stars** with behavior defined as **RPN programs** that tick via GPU kernels. This architecture enables truly living worlds where entities exist continuously, plan during sleep cycles, and coordinate through faction blackboards—all without CPU intervention.

## 1. Classical AI Algorithms to Absorb (RPN-Compatible)

### Behavior Tree RPN Transformation
Traditional behavior trees become compiled RPN bytecode:
```
SELECTOR: [child1] [child2] ... [childN] BT_SELECTOR_N
SEQUENCE: [child1] [child2] ... [childN] BT_SEQUENCE_N  
LEAF_ACTION: [param1] [param2] ... BH_<ACTION>
```

### Utility AI RPN Transformation
Each utility curve becomes a GPU-evaluated function:
```
[world_state_1] [weight_1] CURVE_EVAL
[world_state_2] [weight_2] CURVE_EVAL
...
N BH_UTILITY_EVAL  // Pushes best_action_idx
```

### Steering Behaviors as Vector Operations
All steering calculations operate on PhysicsBodySOA data:
```
[entity_id] [target_id] BH_SEEK  // → [force_x] [force_y] [force_z]
[entity_id] [threat_id] BH_FLEE  // → [force_x] [force_y] [force_z]
```

## 2. K3D Entity Star Schema

### Galaxy Star Layout (64-byte packed)
```
Offset | Field               | Type  | Description
-------|---------------------|-------|--------------------------------
0x00   | entity_id           | u32   | Unique entity handle
0x04   | physics_body_id     | u32   | SOA slot in PhysicsBodySOA
0x08   | behavior_rpn        | u64   | Address in RPN program Galaxy
0x10   | faction             | u8    | Faction ID (0-255)
0x11   | sleep_state         | u8    | 0=awake(60Hz), 1=dozing(10Hz), 2=asleep(1Hz)
0x12   | perception_flags    | u8    | Bitmask: SEE_VISUAL, HEAR_AUDIO, DETECT_SCENT
0x13   | ai_tier             | u8    | 0=reactive, 1=planning, 2=strategic
0x14   | perception_radius   | f32   | Sphere radius for queries
0x18   | goal_stack_ptr      | u32   | Pointer to goal stack in Galaxy
0x1C   | blackboard_star_id  | u32   | Faction shared blackboard
0x20   | last_player_dist    | f32   | Distance to TRM avatar
0x24   | awareness           | f32   | 0.0=unaware to 1.0=fully aware
0x28   | emotion_vector[4]   | f32x4 | (anger, fear, happiness, curiosity)
0x38   | reserved            | u8x8  | Future expansion
```

### PhysicsBodySOA Reference Layout
Each entity references 8 arrays in PhysicsBodySOA:
```cuda
float4* pos;      // world position (w=timestamp)
float4* vel;      // velocity (w=drag)
float4* orient;   // quaternion orientation (w=1.0)
float4* ang_vel;  // angular velocity
float4* scale;    // scale (w=uniform)
float4* forces;   // accumulated forces (w=mass)
float4* bounds;   // bounding sphere (xyz=center, w=radius)
uint4*  phys_flags;// physics flags and collision layers
```

## 3. Behavior RPN Opcodes (0x180-0x1BF)

### Perception & Sensing (0x180-0x18F)
```
0x180: BH_PERCEIVE
  Input: [radius: f32]
  Output: [entity_list_addr: u64]
  Description: Queries Morton Octree for entities within radius.
               Uses same GPU kernel as TRM frustum culling.
               Entity list stored in shared Galaxy memory.

0x181: BH_FILTER_BY_FACTION
  Input: [entity_list_addr: u64] [faction_mask: u32]
  Output: [filtered_list_addr: u64]
  Description: Filters perceived entities by faction mask.

0x182: BH_SENSE_PLAYER
  Input: [none]
  Output: [player_distance: f32] [player_visible: u32]
  Description: Gets distance and visibility to TRM avatar.
```

### Steering & Movement (0x190-0x19F)
```
0x190: BH_SEEK
  Input: [target_id: u32]
  Output: [force_x: f32] [force_y: f32] [force_z: f32]
  Description: Calculates seek steering force toward target entity.

0x191: BH_FLEE  
  Input: [threat_id: u32]
  Output: [force_x: f32] [force_y: f32] [force_z: f32]
  Description: Calculates flee steering force away from threat.

0x192: BH_ARRIVE
  Input: [target_id: u32] [slow_radius: f32] [arrival_radius: f32]
  Output: [force_x: f32] [force_y: f32] [force_z: f32]
  Description: Decelerates upon approaching target.

0x193: BH_SEPARATE
  Input: [neighbor_list_addr: u64]
  Output: [force_x: f32] [force_y: f32] [force_z: f32]
  Description: Calculates separation force from nearby entities.

0x194: BH_APPLY_FORCE
  Input: [force_x: f32] [force_y: f32] [force_z: f32]
  Output: [none]
  Description: Writes force to PhysicsBodySOA forces array.
               Force accumulates until physics phase clears.

0x195: BH_PATHFIND
  Input: [target_x: f32] [target_y: f32] [target_z: f32]
  Output: [waypoint_addr: u64] [path_length: u32]
  Description: Calls LED-A* pathfinding, returns first waypoint.
```

### Behavior Systems (0x1A0-0x1AF)
```
0x1A0: BH_BT_TICK
  Input: [bt_program_addr: u64]
  Output: [status: u32] (0=FAILURE, 1=SUCCESS, 2=RUNNING)
  Description: Executes behavior tree RPN program.

0x1A1: BH_UTILITY_EVAL
  Input: [N: u32] ([weight: f32] [curve_idx: u8] [state_value: f32] ...)
  Output: [best_action_idx: u32]
  Description: Evaluates N weighted utility curves, returns highest.

0x1A2: BH_GOAP_PLAN
  Input: [goal_state_star_id: u64]
  Output: [action_sequence_addr: u64] [plan_valid: u32]
  Description: Goal-Oriented Action Planning using A* over action space.
               Returns sequence of action RPN addresses.

0x1A3: BH_SLEEP_CHECK
  Input: [player_distance: f32]
  Output: [new_sleep_state: u8]
  Description: Determines sleep state based on player proximity:
               0=awake (<50m), 1=dozing (50-200m), 2=asleep (>200m)
```

### Communication & Memory (0x1B0-0x1BF)
```
0x1B0: BH_BLACKBOARD_READ
  Input: [key_hash: u32]
  Output: [value: f32] [timestamp: f32]
  Description: Reads from faction blackboard star.

0x1B1: BH_BLACKBOARD_WRITE
  Input: [key_hash: u32] [value: f32]
  Output: [none]
  Description: Writes to faction blackboard star with atomic update.

0x1B2: BH_EMIT_EVENT
  Input: [event_type: u32] [target_id: u32] [intensity: f32]
  Output: [none]
  Description: Writes to CollisionEventQueue for other entities to perceive.

0x1B3: BH_LEARN_REINFORCE
  Input: [behavior_trace_addr: u64] [reward: f32]
  Output: [none]
  Description: Strengthens successful behavior traces during sleep.

0x1B4: BH_PRUNE_BRANCHES
  Input: [behavior_tree_addr: u64]
  Output: [pruned_count: u32]
  Description: Removes unused behavior branches during sleep cycles.
```

## 4. Always-On Game Loop Integration

### BEHAVIOR_PHASE Kernel (behavior_fused.ptx)
```cuda
__global__ void behavior_phase_kernel(
    EntityStar* entities,
    PhysicsBodySOA* physics,
    GalaxyMemory* galaxy,
    uint32_t entity_count,
    float4 player_pos)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entity_count) return;
    
    EntityStar* entity = &entities[idx];
    
    // Skip based on sleep state
    if (entity->sleep_state == 2) {
        // Asleep: only update at 1Hz (once per 60 frames)
        if (global_frame_counter % 60 != 0) return;
    } else if (entity->sleep_state == 1) {
        // Dozing: update at 10Hz (once per 6 frames)
        if (global_frame_counter % 6 != 0) return;
    }
    // Awake entities update every frame (60Hz)
    
    // Calculate distance to player for sleep transitions
    float4 entity_pos = physics->pos[entity->physics_body_id];
    float dist_to_player = length3f(entity_pos - player_pos);
    entity->last_player_dist = dist_to_player;
    
    // Execute behavior RPN program
    execute_rpn_program(
        entity->behavior_rpn,
        entity,           // Self pointer
        physics,          // Physics SOA
        galaxy,           // Galaxy memory
        dist_to_player    // Parameter
    );
    
    // Update sleep state based on new perception
    if (dist_to_player < 50.0f) entity->sleep_state = 0;
    else if (dist_to_player < 200.0f) entity->sleep_state = 1;
    else entity->sleep_state = 2;
}
```

### TRM Game Loop Integration
```
// trm_step_fused.ptx main loop
while (simulation_running) {
    // Phase 1: Swarm update (Nine-Chain Swarm)
    swarm_phase_kernel<<<grid, block>>>(swarm_data);
    
    // Phase 2: Physics integration
    physics_phase_kernel<<<grid, block>>>(physics_bodies);
    
    // Phase 3: Behavior update
    behavior_phase_kernel<<<grid, block>>>(
        entity_stars, physics_bodies, galaxy, 
        entity_count, player_position);
    
    // Phase 4: Rendering prep
    draw_phase_kernel<<<grid, block>>>(render_data);
    
    cudaDeviceSynchronize();
    frame_counter++;
}
```

### Sleep-Time Behavior Processing
A separate kernel runs during idle GPU cycles:
```cuda
__global__ void sleep_consolidation_kernel(
    EntityStar* entities,
    GalaxyMemory* galaxy,
    uint32_t entity_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entity_count) return;
    
    EntityStar* entity = &entities[idx];
    
    // Only process sleeping entities
    if (entity->sleep_state != 2) return;
    
    // 1. Consolidate successful behavior traces
    BH_LEARN_REINFORCE(entity->behavior_rpn, 0.1f);
    
    // 2. Prune unused behavior branches  
    BH_PRUNE_BRANCHES(entity->behavior_rpn);
    
    // 3. Plan future actions using GOAP
    if (entity->ai_tier >= 1) {
        uint64_t goal_addr = galaxy->alloc_temp();
        // Set goal based on faction blackboard
        BH_GOAP_PLAN(goal_addr);
        galaxy->free_temp(goal_addr);
    }
}
```

## 5. Integration Points

### Morton Octree Shared Query System
```
BH_PERCEIVE → morton_query_kernel() ← TRM_FRUSTUM_CULL
      ↓                              ↓
   Entity list                   Visible objects
```

The same Morton code index and GPU kernels serve both behavior perception and rendering culling.

### LED-A* Pathfinding Shared Pool
```
Entity A: BH_PATHFIND → LED-A* pool slot 0
Entity B: BH_PATHFIND → LED-A* pool slot 1
TRM Avatar: NAV_REQUEST → LED-A* pool slot 2
```

Pathfinding requests are batched and processed in parallel each frame.

### Physics Force Integration
```
Behavior Phase:
  BH_APPLY_FORCE → writes to PhysicsBodySOA.forces[body_id]

Physics Phase:
  PH_INTEGRATE (0x151) reads forces, applies to velocity
  PH_CLEAR_FORCES (0x152) zeros force accumulators
```

### Collision Event Communication
```
Physics Phase:
  Collision detected → writes to CollisionEventQueue

Behavior Phase:
  BH_PERCEIVE reads CollisionEventQueue as "hearing"
  Entities react to nearby collision events
```

### Faction Blackboard Coordination
Each faction has a shared Galaxy star containing:
```
struct FactionBlackboard {
    float4 rally_point;          // Where to gather
    float4 enemy_last_seen;      // Last enemy sighting
    uint32_t alarm_level;        // 0-100 alertness
    uint32_t resource_count;     // Available resources
    uint32_t member_bitmask;     // Which entities are present
    float timestamp;             // Last update time
};
```

All faction members read/write atomically to this shared memory.

## 6. Example Behavior RPN Programs

### Simple Guard Patrol (Reactive Tier)
```
// Load patrol points
[patrol_point_1_addr] [patrol_point_2_addr] [patrol_point_3_addr]
  
// Perception phase
[30.0] BH_PERCEIVE                // Get nearby entities
[0x01] BH_FILTER_BY_FACTION      // Filter by enemy faction
  
// If enemies detected
BH_IF_THEN_ELSE
  THEN:
    [nearest_enemy_id] BH_SEEK    // Move toward enemy
    [10.0] BH_ARRIVE              // Stop at attack range
    [ATTACK_EVENT] BH_EMIT_EVENT  // Trigger attack
  ELSE:
    [current_patrol_point] BH_SEEK  // Continue patrol
    [5.0] BH_ARRIVE                // Arrive at point
    [NEXT_PATROL_POINT]            // Cycle to next point
```

### Strategic Commander (Planning Tier)
```
// Read faction blackboard for objectives
[KEY_RALLY_POINT] BH_BLACKBOARD_READ
  
// If under attack, plan defense
[defense_goal_addr] BH_GOAP_PLAN
[plan_valid] BH_IF
  THEN:
    // Execute defense plan
    [action_sequence_addr] BH_EXECUTE_SEQUENCE
    
    // Update blackboard with status
    [KEY_DEFENSE_MODE] [1.0] BH_BLACKBOARD_WRITE
```

## 7. Performance Characteristics

### Memory Footprint
```
1000 entities: ~64KB (EntityStar) + ~256KB (RPN programs)
PhysicsBodySOA: 1000 * 8 arrays * 16 bytes = 128KB
Faction blackboards: 256 factions * 64 bytes = 16KB
Total: ~464KB for 1000 living entities
```

### GPU Throughput
- Awake entities: 60Hz update (≤50m from player)
- Dozing entities: 10Hz update (50-200m from player)  
- Asleep entities: 1Hz update (>200m from player)

Estimated GPU load: 10,000 entities at 60Hz = ~3ms on RTX 4090

## 8. Evolution Path

### Phase 1: Reactive Behaviors (Current)
- Steering behaviors
- Basic perception
- Simple behavior trees

### Phase 2: Planning Entities
- GOAP integration
- Faction coordination
- Resource management

### Phase 3: Learning Entities
- Reinforcement learning of behavior traces
- Genetic evolution of RPN programs
- Culture propagation through factions

## Conclusion

The K3D RPN Reality Engine represents a fundamental rearchitecture of game AI: from transient Python scripts to persistent GPU-executed RPN programs. By treating entities as Galaxy stars with continuous existence, we enable emergent complexity through simple components—steering behaviors become forces, perception becomes octree queries, and coordination becomes blackboard writes. This system scales to millions of entities through sleep-state management and GPU parallelism, creating truly living worlds within the TRM framework.

---

*Document generated for:*
**Knowledge 3D Standard** · **GitHub Repository**  
**Path:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/docs/research/rpn_reality_engine_ai_spec.md`  
**Version:** 1.0 · **Last Updated:** 2024-12-01 · **Status:** Active Specification