# Embodied Game Loop Gap Analysis: What K3D Is Missing

**Date:** 2026-04-10
**Author:** Claude (Architecture Partner)
**Reviewed by:** Kimi K2.5 Swarm (game engine + cognitive), Nemotron (GPU event dispatcher), DeepSeek (lifecycle state machine), Qwen 3.5 (entity behavior audit), Kimi (kernel phase mapping), Kimi (state machine CUDA), GLM (perception pipeline)
**Grounding:** THREE_BRAIN_SYSTEM_SPECIFICATION.md, AVATAR_EMBODIMENT_SPECIFICATION.md, KNOWLEDGEVERSE_SPECIFICATION.md, SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md, SLEEPTIME_PROTOCOL_SPECIFICATION.md

---

## Executive Summary

K3D has the **data structures** of an embodied entity but not the **runtime systems** to make it live. The specs describe a TRM avatar that lives in a House (Memory Palace), thinks in a Galaxy (VRAM brain), and runs as a continuous game loop. The implementation has:

- `EntityHotPath` struct with 16 fields (position, physics, perception, behavior, sleep) -- **90% unused**
- `trm_step_fused.cu` with physics and behavior phase slots -- **both are void-cast stubs**
- `entity_behavior.cu` with 5 implemented device functions -- **never called**
- `event_queue_ptr` kernel parameter -- **allocated but never populated**
- `TRMGameLoop` Python class -- **synchronous request/response queue, NOT a game loop**
- 88 PTX kernels compiled -- **only ~5 active in the query path**

The TRM is a "brain in a jar." It can think (nine-chain swarm) but cannot see (no perception), move (no physics), interact (no hands), sleep on its own (no idle detection), or react to events (no event system). This document identifies the 7 missing subsystems and maps them to existing K3D infrastructure.

---

## 1. The Seven Missing Subsystems

### 1.1 GPU Event Dispatcher (The Avatar's Nervous System)

**Status:** `event_queue_ptr` is passed to `trm_step_fused.cu` line 48 but void-cast at line 19. No event types, no ring buffer, no dispatch mechanism.

**What it does:** Multiple kernels (physics, perception, timer, I/O) produce events. The behavior phase consumes them each tick. Without it, the TRM can only process explicit queries -- it cannot react to world changes.

**Design (from Nemotron + Kimi):**
- **GPUEvent struct:** 16 bytes (entity_id: u32, event_type: u8, priority: u8, pad: u16, payload: u64)
- **Ring buffer:** 4096 entries in VRAM (64 KB), atomic head/tail via `atomicCAS` for lock-free MPSC
- **Dispatch table:** `__constant__` memory function pointer array (7 event types, broadcast-efficient)
- **Event types:** PERCEPTION_STIMULUS, COLLISION, INTERACTION, TIMER, IO, INTERNAL, WAKEUP
- **Integration:** 3 new kernel params to `trm_step_fused`: ring_buffer_ptr, head_ptr, tail_ptr

**Spec grounding:** AVATAR_EMBODIMENT_SPECIFICATION.md S8 (interaction model) requires event-driven behavior. SLEEPTIME_PROTOCOL_SPECIFICATION.md S0.2 requires idle-triggered consolidation via timer events.

### 1.2 Entity Lifecycle State Machine (The Avatar's Clockwork)

**Status:** `EntityHotPath.sleep_state` (uint8_t) exists but is never read or written by any kernel. No state transitions, no idle detection, no wake/sleep cycle.

**What it does:** Governs what the TRM does each tick: perceive, navigate, reason, act, sleep. Without it, the TRM has no autonomous behavior -- it only responds to external stimulation.

**Design (from DeepSeek + Kimi):**
- **7 states:** SLEEP, IDLE, PERCEIVING, NAVIGATING, REASONING, ACTING, HANDLING_QUERY
- **VRAM-resident struct:** `TRMStateMachine` (32 bytes) parallel to `EntityHotPath`
  - state_stack[4] for interrupt recovery (push current state, handle query, pop)
  - idle_accumulator for 30-second sleep transition
  - state_entry_tick for timeout tracking
- **Transition table:** `__constant__` memory, table-driven (from_state, event_type) -> to_state
- **Priority interrupt:** HANDLING_QUERY can preempt any state except atomic sections
- **Idle -> Sleep:** After 30 seconds of no events, awareness decays below threshold, consolidation triggers automatically

**Spec grounding:** THREE_BRAIN_SYSTEM_SPECIFICATION.md S2.1 ("runs as a continuous game loop"). SLEEPTIME_PROTOCOL_SPECIFICATION.md S0.2 ("idle timeout: no input for N seconds AND pending briefs > 0"). AVATAR_EMBODIMENT_SPECIFICATION.md S10 (avatar lifecycle).

### 1.3 Perception Pipeline (The Avatar's Senses)

**Status:** `EntityHotPath` has `perception_radius`, `perception_flags`, `cranial_origin[3]`, `awareness`. `entity_behavior.cu` has `bh_perceive_count()` doing O(N^2) brute-force scan. Morton Octree, Frustum Cull, and Dynamic LOD kernels exist but are NOT connected to entity perception.

**What it does:** Translates House (external 3D reality) into Galaxy (internal VRAM brain) entries. Without it, the avatar is blind -- it can only process explicit queries, not discover knowledge by looking around.

**Design (from GLM):**
6-stage GPU pipeline, all existing kernels + 2 new:

```
Stage 0: QUERY BUILD    - EntityHotPath -> search params (cranial_origin + gaze -> AABB + frustum planes)
Stage 1: SPATIAL QUERY  - Morton Octree range query [EXISTING kernel, needs range query mode]
Stage 2: FRUSTUM FILTER - Frustum Cull from avatar's gaze direction [EXISTING kernel, needs per-entity mode]
Stage 3: LOD GATING     - Dynamic LOD based on distance/perception_radius [EXISTING kernel]
Stage 4: SALIENCY       - Score by motion + novelty + threat + goal relevance [NEW kernel]
Stage 5: GALAXY BINDING - Perceived objects become Galaxy working memory stars [NEW kernel]
```

**Missing EntityHotPath fields:** `gaze_yaw`, `gaze_pitch`, `gaze_fov`, `attention_entity_id`, `attention_weight`. Without gaze direction, the frustum cull has no origin.

**Spec grounding:** AVATAR_EMBODIMENT_SPECIFICATION.md S4 ("the Cranial Galaxy: AI Brain Space") -- perception populates the Galaxy from House stimuli. SGI_SPECIFICATION.md S2.1 ("Spatial Grounding: intelligence MUST operate within navigable 3D environment").

### 1.4 Physics Integration (The Avatar's Body)

**Status:** `EntityHotPath.physics_body_id` exists. `trm_physics_phase_stub()` void-casts all params: `physics_soa_ptr`, `contact_soa_ptr`, `event_queue_ptr`, `body_count`, `physics_dt`, `solver_iterations`.

**What it does:** Makes the avatar a physical body in the House -- collisions with walls, gravity, standing on floors. Without it, `house_x/y/z` is a ghost coordinate that can pass through anything.

**What's needed:**
- Physics SOA (Structure of Arrays) for rigid bodies -- position, velocity, force, collision shape
- Collision detection kernel (broad-phase via Morton Octree, narrow-phase via GJK/SAT)
- Collision events fed to event queue (COLLISION events)
- Avatar as kinematic body (driven by behavior output, not dynamic forces)

**Spec grounding:** AVATAR_EMBODIMENT_SPECIFICATION.md S2 ("unified body architecture: same skeletal structure"). The avatar must physically navigate rooms, be blocked by walls, stand on floors.

### 1.5 Behavior Execution (The Avatar's Agency)

**Status:** `EntityHotPath.behavior_rpn_addr` (uint64_t pointer to RPN bytecode) exists but is never dereferenced. `entity_behavior.cu` has `bh_seek_force()`, `bh_separation_force()` -- never called. `blackboard_star_id` exists but no blackboard system.

**What it does:** Turns decisions (from Reasoning phase) into actions: move toward target, pick up object, create knowledge star. Without it, the TRM can reason but cannot act on its conclusions.

**What's needed:**
- RPN behavior interpreter kernel (read bytecode at `behavior_rpn_addr`, execute opcodes)
- Latent action support (actions that span multiple ticks: "walk to door" = pathfind + move over N frames)
- Blackboard system for `blackboard_star_id` (entity working memory: target, goal, plan)
- Motor output pipeline (behavior decisions -> forces applied to physics body)

**Spec grounding:** THREE_BRAIN_SYSTEM_SPECIFICATION.md game engine analogy: "NPC decision = Nine-Chain Swarm + Halting Gate." The decision is computed; the execution is missing.

### 1.6 Fixed-Timestep Game Loop (The Avatar's Heartbeat)

**Status:** `TRMGameLoop` at `knowledgeverse/trm_game_loop.py` is a Python request/response queue. `tick()` is called manually. No fixed timestep, no delta time, no accumulator.

**What it does:** Ensures deterministic, regular entity updates at a fixed frequency (e.g., 50Hz). Without it, the avatar updates at arbitrary intervals -- physics tunneling, behavior oscillation, non-reproducible results.

**What's needed:**
- Replace manual Python `tick()` with a continuous loop (C++ daemon or Python asyncio with GPU stream management)
- Fixed delta_time parameter passed to `trm_step_fused.cu` (currently not passed)
- Time accumulator for sub-stepping physics when frame time > fixed step
- Phase budget management (20ms total at 50Hz, partitioned across phases)

**Spec grounding:** THREE_BRAIN_SYSTEM_SPECIFICATION.md S2.1 ("runs as a continuous game loop via trm_step_fused.ptx"). KNOWLEDGEVERSE_SPECIFICATION.md S2.1 ("K3D is NOT a program you run -- it is a living, always-on, embodied AI").

### 1.7 Interaction System (The Avatar's Hands)

**Status:** No interaction component defined anywhere. Avatar cannot pick up objects, open books, touch surfaces, or hold the Memory Tablet.

**What it does:** Enables the avatar to manipulate objects in the House -- open books (load Galaxy), pick up stars, activate doors (network interfaces), hold the Memory Tablet (primary interface).

**What's needed:**
- Interaction volume (raycast from cranial_origin + gaze direction for focus detection)
- Action binding (map "use" behavior to specific House objects: "open door", "pick up star")
- Memory Tablet as held object (MEMORY_TABLET_SPECIFICATION.md)
- Object permanence in Galaxy (when an object leaves the frustum, its star persists with decreasing saliency)

**Spec grounding:** MEMORY_TABLET_SPECIFICATION.md (primary interface object, held in avatar's hand). AVATAR_EMBODIMENT_SPECIFICATION.md S9 ("Memory Tablet as Held Object").

---

## 2. Kernel Phase Mapping (What Goes Where in the Tick)

The 88 PTX kernels map to 6 game engine phases. The existing composed head pipeline (Morton -> LED-A* -> Frustum -> LOD -> Swarm -> Halting Gate) maps as:

| Phase | Duration | Domain | Existing Kernels | Status |
|-------|----------|--------|-----------------|--------|
| **INPUT** | 0.5ms | I/O | Event queue drain | Missing |
| **PHYSICS** | 3.0ms | House | `trm_physics_phase_stub` -> need real physics | Stubbed |
| **PERCEPTION** | 4.0ms | House->Galaxy | `morton_octree.cu`, `frustum_cull.cu`, `dynamic_lod.cu`, `bh_perceive_count()` | Exists, disconnected |
| **NAVIGATION** | 3.0ms | Galaxy | `led_astar.cu` | Exists, disconnected |
| **REASONING** | 6.0ms | Galaxy | `nine_chain_swarm.cu`, `halting_gate.cu`, `semantic_gravity_tick.ptx` | Active (only these work) |
| **ACTION** | 2.0ms | Galaxy->House | `bh_seek_force()`, `bh_separation_force()`, `star_materializer.cu` | Exists, never called |
| **CONSOLIDATION** | 1.0ms | Galaxy | `galaxy_memory_updater.cu`, `semantic_lesson_tick.ptx` | Exists, mis-wired (Python) |
| **MICRO-SLEEP** | 0.5ms | Interstitial | `sleep_time_micro.cu`, `sleep_cluster_refiner.ptx` | Exists, mis-wired (Python) |

**Total: 20ms @ 50Hz**

**Key insight:** The composed head pipeline currently runs as a flat sequence for query answering. In the game loop, these kernels must be phase-gated by the lifecycle state machine -- you don't run Reasoning when the entity is asleep, you don't run Perception when it's deep in thought.

---

## 3. The `trm_step_fused.cu` Refactor

### Current (3 phases, 2 stubbed):
```c
trm_step_fused() {
    trm_recursive_core(...)     // Reasoning only -- occupies entire tick
    trm_physics_phase_stub()    // void-casts everything
    trm_behavior_phase_stub()   // void-casts everything
}
```

### Target (7 phases, state-machine-gated):
```c
trm_step_fused(entities, state_machines, event_queue, delta_time, tick) {
    // 1. STATE MACHINE: Read lifecycle state, process events, transition
    trm_state_machine_step(entity, event_queue, delta_time, tick)

    // 2. PHASE DISPATCH (gated by current state):
    switch (entity.current_state) {
        case SLEEP:
            sleep_time_micro(...)             // Consolidation kernel
            break
        case IDLE:
            awareness_decay(entity)           // Simple exponential decay
            break
        case PERCEIVING:
            morton_query(...)                  // Spatial query
            frustum_cull(...)                  // Visibility filter
            dynamic_lod(...)                   // Detail selection
            saliency_score(...)               // Rank perceived objects
            galaxy_bind(...)                  // Write to Galaxy
            break
        case NAVIGATING:
            led_astar(...)                    // Pathfinding
            bh_seek_force(...)                // Movement toward target
            break
        case REASONING:
            nine_chain_swarm(...)             // Parallel reasoning
            halting_gate(...)                 // Convergence check
            break
        case ACTING:
            star_materializer(...)            // Create Galaxy entries
            bh_separation_force(...)          // Spatial adjustments
            break
        case HANDLING_QUERY:
            // Fast path: direct swarm dispatch for query
            nine_chain_swarm(...)
            halting_gate(...)
            emit_answer(...)
            sm_pop_state(...)                 // Return to previous state
            break
    }

    // 3. PHYSICS (runs every tick regardless of state, except SLEEP):
    if (entity.current_state != SLEEP)
        physics_step(entity, physics_soa, contacts)

    // 4. WRITEBACK: Commit EntityHotPath changes
    commit_position(entity)
}
```

---

## 4. EntityHotPath Field Usage Audit

From Qwen's analysis -- which fields are actually read/written today:

| Field | Size | Read | Written | By What |
|-------|------|------|---------|---------|
| `star_table_idx` | u32 | No | No | Nothing |
| `physics_body_id` | u32 | Yes | No | `bh_entity_position`, `bh_entity_velocity` |
| `behavior_rpn_addr` | u64 | No | No | Nothing (pointer to bytecode, never dereferenced) |
| `house_x/y/z` | 3xf32 | Yes | No | `bh_perceive_count`, `bh_seek_force`, `bh_separation_force` |
| `sleep_state` | u8 | No | No | Nothing (should be lifecycle state) |
| `faction` | u8 | No | No | Nothing |
| `ai_tier` | u8 | No | No | Nothing |
| `perception_flags` | u8 | No | No | Nothing |
| `perception_radius` | f32 | Implicit | No | Passed as arg to `bh_perceive_count`, not read from struct |
| `last_player_dist` | f32 | No | Yes | `bh_perceive_count` (only written field) |
| `awareness` | f32 | No | Yes | `bh_perceive_count` (only written field) |
| `blackboard_star_id` | u32 | No | No | Nothing |
| `meta_rule_addr` | u32 | No | No | Nothing |
| `cranial_origin[3]` | 3xf32 | No | No | Nothing (should be perception sensor offset) |
| `_pad` | f32 | No | No | Alignment |

**Summary:** 2 of 16 fields are written. 4 of 16 are read. The struct is 90% dead weight.

---

## 5. New Fields Needed in EntityHotPath

To support the full embodied lifecycle:

| Field | Type | Purpose | Phase |
|-------|------|---------|-------|
| `gaze_yaw` | f32 | Gaze direction (horizontal) | PERCEPTION |
| `gaze_pitch` | f32 | Gaze direction (vertical) | PERCEPTION |
| `gaze_fov` | f32 | Field of view angle | PERCEPTION |
| `attention_entity_id` | u32 | What entity is being attended to | PERCEPTION/REASONING |
| `motor_output[3]` | 3xf32 | Force/velocity to apply to physics body | ACTION |
| `current_goal_star` | u32 | Galaxy star being pursued | NAVIGATION/REASONING |

Combined with `TRMStateMachine` (32-byte parallel struct for lifecycle state), this adds ~44 bytes. Current EntityHotPath is 68 bytes -> new would be ~112 bytes (fits in 2 cache lines).

---

## 6. Priority Order for Implementation

Based on dependency chains and the specs' emphasis on sovereignty:

### Phase 1: The Clockwork (Enables Everything Else)
1. **GPU Event Ring Buffer** - VRAM-resident, lock-free, 64KB
2. **TRMStateMachine struct** - 32-byte parallel array in VRAM
3. **Lifecycle state machine kernel** - Transition table in `__constant__` memory
4. **Fixed-timestep delta_time** - Pass to `trm_step_fused.cu`

### Phase 2: The Senses (Perception + Embodiment)
5. **Wire entity_behavior.cu** - Unstub `behavior_phase`, call `bh_perceive_count()`, `bh_seek_force()`, etc.
6. **Perception query builder** - EntityHotPath -> Morton range + frustum planes
7. **Connect Morton Octree** - Range query mode for neighbor finding (replaces O(N^2))
8. **Connect Frustum Cull** - Per-entity gaze-based filtering
9. **Saliency kernel** - New: rank perceived entities by relevance

### Phase 3: The Mind (Agency + Consolidation)
10. **RPN behavior interpreter** - Execute bytecode at `behavior_rpn_addr`
11. **Blackboard system** - Working memory at `blackboard_star_id`
12. **Galaxy binding** - Perceived House objects -> Galaxy working memory stars
13. **Idle -> Sleep transition** - 30-second timer -> auto-consolidation
14. **Physics stub -> real physics** - Kinematic avatar body in House

### Phase 4: The Hands (Interaction + Memory Tablet)
15. **Interaction raycast** - cranial_origin + gaze -> focus detection
16. **Object manipulation** - Pick up, open, touch, hold
17. **Memory Tablet integration** - Held object, external query interface

---

## 7. What This Means for the Architecture

The specs are right. The vision is right. The infrastructure exists. What's missing is the **wiring** -- the game engine glue that connects 88 kernels into a coherent entity lifecycle. The TRM was designed as an avatar, but it was built as a query processor.

The transition from "query processor" to "living entity" requires:
- Event-driven behavior (not request/response)
- Autonomous state transitions (not Python orchestration)
- Continuous perception (not query-triggered embedding)
- Physical presence (not ghost coordinates)
- Self-directed exploration (not externally-prompted reasoning)

The composed head pipeline (Morton -> LED-A* -> Frustum -> LOD -> Swarm -> Halting Gate) is the avatar's cognitive process. It needs to be state-gated: the avatar perceives when awake, reasons when stimulated, consolidates when idle, and sleeps when nothing happens. That gating is the state machine. The state machine is driven by events. The events come from perception and I/O. That's the full loop.

---

## Specialist Outputs (Full Reports in TEMP/)

| Specialist | File | Focus |
|-----------|------|-------|
| Kimi K2.5 Swarm (thinking) | `KIMI_SWARM_GAME_ENGINE_GAPS_2026-04-10.md` | Game engine vs cognitive architecture dual analysis |
| Nemotron | `NEMOTRON_GPU_EVENT_DISPATCHER_2026-04-10.md` | GPU event struct, ring buffer, dispatch table CUDA code |
| DeepSeek v3.2 | `DEEPSEEK_ENTITY_LIFECYCLE_2026-04-10.md` | Entity lifecycle state machine, tick budget |
| Qwen 3.5 397B | `QWEN_ENTITY_BEHAVIOR_ANALYSIS_2026-04-10.md` | EntityHotPath field audit, behavior phase design |
| Kimi K2.5 Swarm (thinking) | `KIMI_SWARM_KERNEL_PHASE_MAP_2026-04-10.md` | 88 kernels mapped to game phases, tick refactor |
| Kimi K2.5 | `KIMI_ENTITY_STATE_MACHINE_2026-04-10.md` | Complete CUDA state machine implementation |
| GLM 5.1 | `GLM_PERCEPTION_PIPELINE_2026-04-10.md` | 6-stage perception pipeline, saliency, Galaxy binding |
