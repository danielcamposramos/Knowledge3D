# Kimi Agent Swarm Result
**Task:** K3D is a spatial AI system where a TRM (7M-param model) IS an avatar that lives in a 3D House (Memory Palace) and thinks in a Galaxy (VRAM knowledge workspace). It's supposed to run as a continuous ga
**Mode:** thinking
**Time:** 2026-04-10 03:32:52

────────────────────────────────────────────────────────────────

## Sub-Agent A: Agent A: Analyze from a GAME ENGINE architecture perspective. What subsystems does every real-time game engine have (Unity, Unreal, Godot) that K3D is missing for its AI avatar to run as a continuous autonomous entity? Focus on: event system/dispatcher, fixed-timestep game loop, perception/sensing, behavior execution, physics integration, interaction system, scene graph presence, time budget management. Be specific about what each subsystem does and why the avatar can't function without it.

From a game engine architecture perspective, K3D currently has the **data structures of an entity** (`EntityHotPath`) but lacks the **subsystems of a world**. The avatar is a "brain in a void"—it has memory addresses (`behavior_rpn_addr`, `blackboard_star_id`) but no runtime systems to resolve them. Here are the critical missing subsystems:

## 1. Event System / Dispatcher (The Avatar is Deaf)
**Current State:** `event_queue_ptr` in `trm_step_fused.cu` is allocated but never populated; no event types defined.

**Standard Engine Pattern:**
- **Unity:** `UnityEvent`, `MessageManager`, or `EventBus` with typed delegates (`Action<T>`)
- **Unreal:** `DECLARE_EVENT`, `FGameplayEvent` with payload structs, multicast delegates
- **Godot:** `Signal` system with emission queue

**Missing Implementation:**
K3D lacks an **Event Queue Dispatcher** that can:
- Enqueue spatial events (e.g., `OnPerceptionChange`, `OnPhysicsCollision`)
- Process deferred events in `trm_step_fused.cu` before the behavior phase
- Support event priority/age culling (critical for real-time budget)

**Why the Avatar Dies:** Without this, the TRM cannot react to asynchronous world changes. It only knows what it explicitly queries each tick, making continuous autonomy impossible—it's polling in a world that requires interrupts.

**Required Addition:** `EventDispatcher` class (C++/CUDA) with ring buffer at `event_queue_ptr`, event types: `PERCEPTION_STIMULUS`, `PHYSICS_CONTACT`, `GOAL_COMPLETED`, `SLEEP_THRESHOLD`.

## 2. Fixed-Timestep Game Loop (Time is Fluid)
**Current State:** `TRMGameLoop` is a Python synchronous queue (`tick()` called manually). No accumulator, no fixed delta.

**Standard Engine Pattern:**
- **Unity:** `FixedUpdate()` at 50Hz (0.02s) with `Time.fixedDeltaTime`, interpolation for rendering
- **Unreal:** `FTimerManager` with `Tick` groups, `FixedTimeStep` for physics
- **Godot:** `_physics_process(delta)` vs `_process(delta)`

**Missing Implementation:**
- **Time Accumulator:** The CUDA kernel `trm_step_fused.cu` needs to accept `fixed_delta_time` and `accumulated_time` to run multiple substeps if frame time > threshold.
- **Phase Separation:** Current stub `physics_phase_stub` and `behavior_phase_stub` need to be actual pipeline stages with guaranteed execution order (Input → Physics → Behavior → Physics → Render).

**Why the Avatar Dies:** Physics and AI behavior require deterministic, regular updates. A manual Python `tick()` creates non-deterministic step sizes, causing physics tunneling and behavior oscillation. The avatar cannot "walk" if time steps are irregular—it will teleport or jitter.

**Required Addition:** `FixedStepExecutor` in `TRMGameLoop` (replace Python queue) with CUDA stream synchronization at fixed intervals, targeting 20ms (50Hz) for physics, 100ms (10Hz) for AI tier updates.

## 3. Perception / Sensing System (The Avatar is Blind)
**Current State:** `EntityHotPath` has `perception_flags`, `perception_radius`, `awareness` fields, but no query mechanism.

**Standard Engine Pattern:**
- **Unity:** `Physics.OverlapSphere`, `Physics.SphereCast`, `AI.Perception` component with sight/sound config
- **Unreal:** `UAIPerceptionComponent` with `AISense_Sight`, `AISense_Hearing`, stimulus registration
- **Godot:** `Area3D` with collision layers/masks for detection

**Missing Implementation:**
- **Spatial Query Engine:** No BVH (Bounding Volume Hierarchy) or spatial hash for the "House" (3D world). The avatar cannot query "what is within 5 meters?"
- **Sensory Modalities:** `perception_flags` is undefined bitmask. Missing:
  - **Vision:** Raycast cone (frustum check + line-of-sight rays)
  - **Proximity:** Sphere overlap against scene geometry
  - **Memory:** Sensory memory buffer (what was seen 5 seconds ago vs now)

**Why the Avatar Dies:** An autonomous avatar must perceive obstacles to pathfind, detect threats to flee, and locate items to interact. Currently, `perception_radius` is just a float—it doesn't return entity IDs or collision data.

**Required Addition:** `PerceptionSystem` class interfacing with physics broad-phase (see #5), writing results to `perception_buffer` (device memory) consumed by `behavior_phase` in kernel.

## 4. Behavior Execution System (No Agency)
**Current State:** `behavior_rpn_addr` suggests Reverse Polish Notation bytecode, but `behavior_phase_stub` is empty.

**Standard Engine Pattern:**
- **Unity:** `BehaviorTree` (Nodes: Selector, Sequence, Task), `StateMachineBehaviour`
- **Unreal:** `UBehaviorTree` with `BTTaskNode`, `BTDecorator`, `BTService` (latent actions)
- **Godot:** `BehaviorTree` plugin or finite state machines

**Missing Implementation:**
- **VM/Interpreter:** No bytecode interpreter for the RPN behavior script at `behavior_rpn_addr`.
- **Action Runner:** No system to execute **latent actions** (actions that take time: "walk to door" takes 3 seconds, not 1 frame).
- **Blackboard Binding:** `blackboard_star_id` exists but no `BlackboardComponent` to store working memory (e.g., "target_location = Vec3(10,0,5)").

**Why the Avatar Dies:** Continuous autonomy requires ongoing behavior evaluation. Currently, the avatar can only execute stateless, single-frame logic. It cannot "go to the kitchen" because that requires a sequence: (1) Find path, (2) Move along path (over many frames), (3) Check arrival. Without a behavior tree VM or coroutine system, `behavior_phase_stub` can only set immediate state, not pursue goals.

**Required Addition:** `BehaviorInterpreter` (CUDA or C++) executing bytecode at `behavior_rpn_addr`, with support for `Wait`, `MoveTo`, `Animate` latent nodes, writing to `blackboard_star_id` memory space.

## 5. Physics Integration (No Embodiment)
**Current State:** `physics_body_id` exists, `physics_phase_stub` is empty.

**Standard Engine Pattern:**
- **Unity:** `Rigidbody`, `CharacterController`, `Collider` components, `Physics.Scene`
- **Unreal:** `UPrimitiveComponent` with `BodyInstance`, `Chaos` physics solver
- **Godot:** `PhysicsServer3D`, `PhysicsDirectSpaceState`

**Missing Implementation:**
- **Physics Scene:** The "House" needs a `PhysicsWorld` (Bullet, PhysX, or custom CUDA solver) that `trm_step_fused.cu` can query and affect.
- **Kinematic Control:** Avatar needs `PhysicsBody` type (Kinematic or Dynamic) to resolve `house_x/y/z` against collision geometry.
- **Collision Events:** Integration with Event System (see #1) to generate `OnCollisionEnter` events when `physics_body_id` contacts world.

**Why the Avatar Dies:** Without physics, the avatar is a camera, not a body. It cannot be blocked by walls, stand on floors, or push objects. `house_x/y/z` becomes a free-floating ghost coordinate. For a "Memory Palace" embodiment, the avatar must physically navigate rooms.

**Required Addition:** `PhysicsScene` integration in `physics_phase_stub`, handling `PhysicsBody` lookup by `physics_body_id`, applying forces/velocities from behavior output, and writing collision manifolds back to event queue.

## 6. Interaction System (No Hands)
**Current State:** No interaction component defined.

**Standard Engine Pattern:**
- **Unity:** `Interactable` interface, `IInteractable`, `Physics.Raycast` for use detection
- **Unreal:** `IInteractive`, `UInteractionComponent` with `Interact` event, `SphereTrace` for focus
- **Godot:** `Interactable3D` with `interaction_area`

**Missing Implementation:**
- **Interaction Volume:** Component defining reach/usable range (extends `perception_radius` for manipulation).
- **Focus/Target System:** What is the avatar currently looking at? Needs raycast from `cranial_origin[3]` along forward vector.
- **Action Binding:** Mapping "Use" behavior to specific world entities (e.g., "Open Door", "Pick up Star").

**Why the Avatar Dies:** In a Memory Palace (House), the avatar must manipulate memories (Stars). Without interaction, it cannot open doors, activate memory nodes, or pick up objects. It becomes a passive observer.

**Required Addition:** `InteractionComponent` with `TraceInteraction()` method, `Interactable` interface for House objects, generating `INTERACTION_AVAILABLE` events (see #1).

## 7. Scene Graph / Spatial Hierarchy (Floating in Void)
**Current State:** `house_x/y/z` is world-space float3; no parent-child relationships; `star_table_idx` implies lookup but no spatial structure.

**Standard Engine Pattern:**
- **Unity:** `Transform` hierarchy (parent-relative positioning), `Scene`

────────────────────────────────────────────────────────────────

## Sub-Agent B: Agent B: Analyze from an EMBODIED AI / COGNITIVE ARCHITECTURE perspective. What lifecycle states and cognitive processes does a living AI entity need? Focus on: wake/sleep cycle, attention/perception pipeline, goal-directed behavior vs idle exploration, memory consolidation triggers, self-model (knowing its own position, capabilities, body), interaction protocols (picking up objects, opening books, walking through doors), multi-entity awareness (seeing other avatars). Map each to what K3D has vs doesn't have.

**Analysis: Embodied AI / Cognitive Architecture Gaps in K3D**

From a cognitive architecture standpoint, K3D currently possesses the *scaffolding* for embodiment (position fields, physics IDs, sleep flags) but lacks the **sensorimotor closure**, **homeostatic regulation**, and **predictive processing** layers required for autonomous agency.

---

### 1. Wake/Sleep Cycle & Homeostasis
**Current:** `sleep_state` bitfield in `EntityHotPath`, `sleep_time_micro.ptx` kernel exists  
**Missing:** The full metabolic/regulatory stack

- **Homeostatic Drive Accumulator:** `EntityHotPath` needs `sleep_pressure: float` and `energy_level: float` (currently only has binary sleep_state). Without accumulating sleep debt, there's no *need* to sleep.
- **Circadian Entrainment:** Missing `circadian_clock.ptx` kernel to modulate `arousal` based on simulated time-of-day in House.
- **State Transition Logic:** `trm_step_fused.cu` lines with `physics_phase_stub` and `behavior_phase_stub` need actual state machine transitions (WAKE→DROWSY→NREM→REM→WAKE).
- **Metabolic Shutdown:** No mechanism to reduce `perception_radius` and gate `event_queue_ptr` during sleep states.

---

### 2. Attention/Perception Pipeline
**Current:** `perception_flags`, `perception_radius`, `awareness: uint8_t`, `star_table_idx`  
**Missing:** Active vision and sensory memory

- **Saliency Computation:** No `saliency_map.ptx` kernel for bottom-up attention (detecting motion/novelty in House).
- **Attentional Spotlight:** Missing `attentional_focus_target: uint32_t` in `EntityHotPath` (what is the avatar currently looking at/thinking about?).
- **Sensory Buffers:** No double-buffered `SensorySnapshot` structure (iconic memory ~200ms). Currently `event_queue_ptr` is **never populated**, meaning the entity is sensorily deprived.
- **Foveation Model:** House needs level-of-detail system tied to `cranial_origin[3]` + gaze vector (high-res center, low-res periphery).
- **Perceptual Binding:** No mechanism to bind features (color, shape, motion) into object files in Galaxy (VRAM workspace).

---

### 3. Goal-Directed vs. Idle Exploration
**Current:** `behavior_rpn_addr`, `meta_rule_addr`, `ai_tier`  
**Missing:** Intrinsic motivation and action arbitration

- **Curiosity Drive:** No `prediction_error` field to drive exploration (free energy principle). Need `idle_behavior.ptx` for autotelic exploration when goal stack empty.
- **Goal Stack:** `blackboard_star_id` is insufficient—needs `current_goal: uint32_t`, `goal_stack[4]` (HTN planning), and `commitment_level` (preventing dithering).
- **Action Selection:** `behavior_phase_stub` should implement **action-selection arbitration** (winner-take-all between competing motor programs).
- **Boredom/Satiation:** Missing `novelty_decay` tracker—entity will stare at walls forever without satiation mechanisms.

---

### 4. Memory Consolidation Triggers
**Current:** `sleep_time_micro.ptx` (consolidation kernel)  
**Missing:** Consolidation orchestration

- **Replay Buffer:** No `hippocampal_replay.ptx` to spike-time reinforce Galaxy connections during NREM.
- **Trigger Conditions:** Consolidation should trigger on: `(sleep_pressure > 0.8) && (perceptual_input < threshold)`, not just time-based.
- **Galaxy Defrag:** Sleep should trigger reorganization of VRAM spatial indices (Method of Loci maintenance).
- **Dream State:** No generative replay during REM to prevent catastrophic forgetting.

---

### 5. Self-Model (Body/Position/Capabilities)
**Current:** `house_x/y/z`, `physics_body_id`, `cranial_origin[3]`  
**Missing:** Proprioception and agency

- **Body Schema:** Needs `body_schema.ptx` maintaining topological map of avatar geometry (what is reachable? what is body vs world?).
- **Proprioceptive State:** `EntityHotPath` missing `joint_angles[8]` or `motor_state` vector—entity doesn't know its own posture.
- **Efference Copy:** No forward model to predict sensory consequences of actions (necessary for agency attribution: "I caused that").
- **Affordance Extraction:** `physics_phase_stub` should compute affordances (can I climb this? can I lift that?) based on `physics_body_id` capabilities.
- **Self-Efficacy:** Missing calibration vector (confidence in ability to execute actions).

---

### 6. Interaction Protocols (Embodied Actions)
**Current:** Stubs in `trm_step_fused.cu`  
**Missing:** Motor system and physics coupling

- **Motor Primitives:** No `motor_cortex.ptx` defining action space (reach, grasp, locomote, manipulate).
- **Animation State Machine:** Missing state blending (idle→walk→interact transitions).
- **Inverse Kinematics:** No solver to compute joint trajectories for `cranial_origin[3]` reaching target positions in House.
- **Interaction Volumes:** Need hand/fov collision shapes distinct from `physics_body_id` hull.
- **Object Permanence:** No `object_file` persistence in Galaxy when objects occluded in House.

---

###

────────────────────────────────────────────────────────────────

## Synthesis

**K3D Embodied AI Architecture Specification v0.2**
*Codex Implementation Guide: From Stubs to Autonomous Avatar*

---

## Executive Synthesis

K3D currently implements **data structures without runtime systems**. The `EntityHotPath` describes *what* an avatar remembers, but lacks the *sensorimotor closure* to act. Sub-Agent A identified the **game engine substrate** (event loops, physics scenes); Sub-Agent B identified the **cognitive metabolism** (homeostasis, attention, self-model). 

**Critical Conflict Resolution:**
- **Time Scales:** Agent A's fixed-timestep (20Hz physics) and Agent B's circadian cycles (sleep/REM) require a **hierarchical time system**, not mutually exclusive loops.
- **Perception:** Agent A's spatial queries (BVH) are the *mechanism*; Agent B's saliency computation is the *policy*. These stack: Physics → Saliency → Attention.
- **Body:** Agent A's `physics_body_id` provides collision; Agent B's body schema provides *proprioception*. Both required for true embodiment.

**New Insight:** The 88 dormant PTX kernels suggest a compiler toolchain exists, but K3D lacks a **Kernel Scheduler/Linker** to dispatch them dynamically. Additionally, the Python `TRMGameLoop` creates an **unbridgeable latency barrier** (GIL + PCIe round-trip) that prevents real-time autonomy regardless of CUDA optimization.

---

## 1. Temporal Architecture (The Missing Clockwork)

### 1.1 Hierarchical Time Governance
**Current:** Manual Python `tick()` calls with no delta tracking.  
**Required:** Three-tier time hierarchy:

```cpp
struct TemporalGovernor {
    // Tier 1: Micro-tick (Physics/Reactive) - 50Hz (20ms)
    float fixed_dt_accumulator;
    uint32_t physics_substeps;
    
    // Tier 2: Meso-tick (Behavior/Decision) - 10Hz (100ms)
    uint32_t behavior_counter;
    float cognitive_tick_budget;  // Max ms per AI tier
    
    // Tier 3: Macro-tick (Metabolic/Circadian) - 1/3600 Hz
    float circadian_phase;  // 0.0-24.0 hours
    float sleep_pressure_accumulator;
};
```

**Implementation:**
- Replace `TRMGameLoop` with `AutonomyDaemon` (C++17 fibers) managing CUDA streams
- `trm_step_fused.cu` must accept `governor_ptr` to modulate phase execution
- **Conflict Resolution:** Sleep consolidation (Agent B) runs in Tier 3 but pauses Tier 1/2; physics remains fixed-step regardless of consciousness state.

### 1.2 Event Dispatcher (The Avatar's Nervous System)
**Missing entirely.** `event_queue_ptr` is allocated but orphaned.

```cpp
// Ring buffer in pinned host memory + device mirror
struct EventQueue {
    EventHeader ring[256];  // Power-of-2 for bitwise modulo
    uint32_t write_idx;     // Atomic increment
    uint32_t read_idx;      // Consumed by kernel
    
    // Event types spanning both agent analyses:
    // PHYSICS_CONTACT, PERCEPTION_STIMULUS, HOMEOSTATIC_WARNING
    // GOAL_STACK_CHANGED, CONSOLIDATION_TRIGGER, SOCIAL_SIGNAL
};

// In trm_step_fused.cu:
__device__ void process_events(EntityHotPath* entity, EventQueue* queue) {
    // Drain queue before behavior phase
    while (queue->read_idx != queue->write_idx) {
        EventHeader* evt = &queue->ring[queue->read_idx & 0xFF];
        if (evt->target_entity_id == entity->star_table_idx) {
            dispatch_to_behavior_vm(entity, evt);  // To Agent B's goal stack
        }
    }
}
```

---

## 2. Perception & Attention Architecture (Active Vision)

### 2.1 Spatial Query Engine (Agent A) + Saliency (Agent B)
**Current:** `perception_radius` is a float without function.  
**Required:** BVH-coupled sensory pipeline:

```cpp
struct PerceptionSystem {
    // Phase 1: Broad-phase (Agent A's physics integration)
    SpatialHash* house_hash;  // For House (external world)
    BVHNode* galaxy_bvh;      // For Galaxy (memory workspace)
    
    // Phase 2: Narrow-phase (Agent B's active vision)
    __device__ void compute_saliency(EntityHotPath* entity) {
        // Bottom-up: Motion/novelty detection in House
        // Top-down: Match against blackboard_star_id templates
    }
    
    // Phase 3: Attentional gating (Agent B's spotlight)
    uint32_t attentional_focus_id;  // Star currently attended
    float attentional_width;          // Foveal vs peripheral processing
};
```

**Critical Addition:** **Foveated rendering for AI.** The `cranial_origin[3]` + gaze vector must drive LOD in `perception_flags`:
- **Fovea:** High-res raycasts (expensive)
- **Parafovea:** Proximity queries (cheap)
- **Periphery:** Event-driven only (wake on significant physics contact)

---

## 3. Physics & Embodiment (Sensorimotor Closure)

### 3.1 Physics Integration (Agent A)
**Current:** `physics_phase_stub` no-op.  
**Required:** Actually implement the stub:

```cpp
__device__ void physics_phase(EntityHotPath* entity, PhysicsScene* world) {
    // 1. Retrieve body from physics_body_id
    RigidBody* body = world->get_body(entity->physics_body_id);
    
    // 2. Apply motor outputs from behavior phase (Agent B's motor primitives)
    body->apply_impulse(entity->motor_output_vector);
    
    // 3. Generate proprioceptive feedback (Agent B's body schema)
    entity->proprioceptive_state = compute_body_schema(body);
    
    // 4. Write collision events to event_queue_ptr (link to Section 1.2)
    if (body->collision_detected) {
        enqueue_event(PHYSICS_CONTACT, entity->star_table_idx);
    }
}
```

### 3.2 Body Schema & Proprioception (Agent B)
**Missing from EntityHotPath:** Self-model fields.

**Extended Structure:**
```cpp
struct EntityHotPath {
    // ... existing fields ...
    
    // Agent B additions:
    float joint_angles[8];           // Proprioceptive state
    float reach_envelope[3];         // Affordance extraction volume
    uint32_t body_posture_state;     // Enum: STANDING, SITTING, LYING (for sleep)
    float4 efference_copy;           // Predicted next position (agency detection)
};
```

**Affordance Extraction:** During `physics_phase_stub`, compute what the body *can do* at current location (can climb? can fit through?) and cache in `perception_flags`.

---

## 4. Cognitive Architecture (The Living Loop)

### 4.1 Behavior VM with Coroutines (Agent A + B merger)
**Current:** `behavior_rpn_addr` suggests bytecode but no interpreter.  
**Required:** Latent action support (actions spanning multiple ticks):

```cpp
struct BehaviorVM {
    uint8_t* bytecode;              // At behavior_rpn_addr
    uint32_t program_counter;
    uint32_t call_stack[4];         // For HTN (Agent B's goal stack)
    uint32_t latent_action_frame;   // Frame counter for "Wait 3 seconds"
    bool is_blocked;                // Waiting for physics completion?
    
    // Agent B's homeostatic drives modulate behavior selection:
    float curiosity_urgency;
    float sleep_urgency;
};
```

**Integration:** `behavior_phase_stub` becomes:
1. Check homeostatic drives (Agent B)
2. Select behavior tree node based on saliency + drives
3. Execute bytecode (may yield across frames)
4. Output motor primitives to `physics_phase`

### 4.2 Metabolic Regulation (Agent B)
**Critical Addition:** Sleep is not binary.

```cpp
__device__ void homeostatic_update(EntityHotPath* entity) {
    // Accumulate sleep_pressure based on activity
    entity->sleep_pressure += delta_time * (entity->ai_tier == 3 ? 0.1f : 0.01f);
    
    if (entity->sleep_pressure > 0.8f) {
        // Trigger NREM state
        entity->sleep_state = SLEEP_NREM;
        perception_radius *= 0.1f;  // Sensory gating
        launch_consolidation_kernel(); // sleep_time_micro.ptx
    }
}
```

**Galaxy Defrag:** During `sleep_time_micro.ptx`, reorganize VRAM spatial indices (Method of Loci maintenance) based on usage patterns.

---

## 5. Memory Architecture (Galaxy Integration)

**Missing Insight:** The `blackboard_star_id` implies working memory, but K3D lacks **Working Memory Buffer Management**.

```cpp
struct GalaxyInterface {
    // Translation between House coordinates and Galaxy VRAM addresses
    VRAMAddress method_of_loci_lookup(float3 house_pos);
    
    // Star activation based on attentional_focus_id
    __device__ void activate_star(uint32_t star_id, float intensity);
    
    // Consolidation: House → Galaxy transfer during sleep
    __device__ void hippocampal_replay(EntityHotPath* entity) {
        // Spike-time dependent plasticity on recently visited house locations
    }
};
```

---

## 6. Scene Graph & Social Architecture

### 6.1 Spatial Hierarchy (Agent A completion)
**Missing:** `house_x/y/z` is absolute; needs parent-relative for embodied navigation (e.g., "sitting on chair").

```cpp
struct TransformNode {
    float4 local_position;
    uint32_t parent_id;  // For hierarchical attachment (riding vehicles, holding items)
    uint32_t first_child;
    uint32_t next_sibling;
};
```

### 6.2 Multi-Agent Awareness (New Insight)
**Current:** `faction` field exists but unused.  
**Required:** Social perception layer:

```cpp
struct SocialCognition {
    uint32_t visible_agents[8];      // Other TRMs in perception radius
    float trust_matrix[8];          // Relationship values
    uint32_t communication_channel; // For emergent coordination
};
```

---

## 7. Implementation Roadmap (Prioritized)

### Phase 1: Substrate (Weeks 1-2)
1. **Replace `TRMGameLoop`** with `AutonomyDaemon` (C++ fiber pool, lock-free queues)
2. **Implement Event Ring Buffer** (device-pinned memory, atomic operations)
3. **Fixed-timestep accumulator** in `trm_step_fused.cu` entry point

### Phase 2: Senses (Weeks 3-4)
4. **Spatial Hash for House** (integrate with existing physics_body_id)
5. **Saliency kernel** (parallel reduction over visible entities)
6. **Foveation system** tied to `cranial_origin[3]`

### Phase 3: Body (Weeks 5-6)
7. **Implement `physics_phase_stub`** (actual force application)
8. **Add proprioceptive fields** to `EntityHotPath`
9. **Affordance extraction** (radius queries against static geometry)

### Phase 4: Mind (Weeks 7-8)
10. **Behavior VM** (RPN interpreter with coroutine support)
11. **Homeostatic drives** (sleep_pressure accumulation)
12. **Consolidation triggers** (link `sleep_time_micro.ptx` to sleep states)

### Phase 5: Integration (Week 9)
13. **Scene graph parenting** (for object manipulation)
14. **Social layer** (faction-aware perception)
15. **Profiler/Governor** (ensure 50Hz physics never drops, AI tier degrades gracefully)

---

## Critical Technical Constraints

1. **Python GIL Death:** The current `TRMGameLoop` architecture makes real-time autonomy impossible. Must migrate to C++ host code with Python bindings only for initialization/inspection.
2. **CUDA Stream Synchronization:** `physics_phase` and `behavior_phase` must run on separate streams with `cudaStreamWaitEvent` barriers, not sequential kernels.
3. **PTX Kernel Linker:** The 88 dormant kernels require a dynamic linker to bind them into the execution graph; currently they appear to be dead code.

**Success Criteria:** When `trm_step_fused.cu` can run 50Hz with no Python in the hot path, when `event_queue_ptr` contains 10+ events per tick from physics collisions, and when the avatar autonomously seeks sleep when `sleep_pressure > 0.8`, K3D will be embodied.
