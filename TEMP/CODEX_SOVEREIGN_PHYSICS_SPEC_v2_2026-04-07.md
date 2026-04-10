# CODEX Sovereign Physics Engine Spec v2
**Date:** April 7, 2026  
**Author:** Claude Code (Architecture Partner)  
**Source:** MVCIC chain `mvcic_nvidia_physics_k3d_2026-04-07_2329.md` (5 partners: Kimi → Qwen → GLM → DeepSeek → Nemotron) + manual post-chain grounding  
**Status:** READY FOR CODEX IMPLEMENTATION

---

## 1. Architecture Overview

K3D's physics engine is NOT a port of PhysX or Warp — it is a **sovereign GPU re-expression** of the same physical models, where:

- All computation runs in PTX kernels (hot path — zero Python, zero external libs)
- Physical constants live as **Reality Galaxy Layer-2 stars**
- Force laws live as **Grammar Galaxy Layer-3 RPN programs** (`physics_rpn_addr` field)
- Sleep/wake policies live as **Layer-4 Meta-Rules**
- Shape visuals cross-link to **Drawing Galaxy Layer-1** `visual_rpn`
- Collision events feed back as **Layer-2 Galaxy edge updates**

### 1.1 PHYSICS_PHASE Slot in trm_step_fused.ptx

Insert between `SWARM_PHASE` and `DRAW_PHASE`:

```
SWARM_PHASE      → nine_chain_swarm_kernel (parallel cognitive workers)
PHYSICS_PHASE    → physics sub-phase sequence (see §2)
DRAW_PHASE       → drawing_primitives + frustum cull + LOD
```

Physics is NOT a new top-level system — it is a **composed phase** inside the existing game loop.

### 1.2 RPN-SOA Dual Dispatch Pattern

Physics opcodes (0x150–0x17F) are **meta-dispatch opcodes**, identical in pattern to `GALAXY_SCAN (0xE2)`:

```
1. Pop parameters from RPN stack  (dt, body_count, iteration_count, ...)
2. Dispatch a SOA-wide CUDA kernel (operates on ALL bodies in PhysicsBodySOA)
3. Push result/status back to RPN stack  (contact_count, active_islands, error, ...)
```

This solves the fundamental RPN-SOA mismatch identified by GLM: per-thread RPN stack vs. body-indexed SOA arrays. The opcode is the dispatch bridge.

Extend `modular_rpn_kernel.cu` — the existing switch statement at case `0x150` and above. Do NOT create a separate dispatcher.

---

## 2. PHYSICS_PHASE Sub-Phases (Ordered)

Each sub-phase is a separate kernel callable via its RPN opcode. The Grammar Galaxy `physics_rpn_addr` program sequences them:

| Step | Sub-Phase | RPN Opcode | Primary Kernel | Reuses |
|---|---|---|---|---|
| 1 | Morton Update | `0x150 PH_BROAD_PHASE` | `physics_update_morton_codes.cu` | `morton_octree.ptx:morton_encode_point` |
| 2 | Broad Phase SAP | (bundled in 0x150) | `physics_broad_phase_sap.cu` | `morton_octree.ptx` sorted codes |
| 3 | Narrow Phase GJK | `0x151 PH_NARROW_PHASE` | `physics_narrow_phase_gjk.cu` | warp-cooperative simplex (see §6) |
| 4 | Constraint Generate | `0x152 PH_CONSTRAINT_GENERATE` | `physics_constraint_generate.cu` | `cosine_similarity.ptx` for friction clustering |
| 5 | Constraint Color | `0x159 PH_CONSTRAINT_COLOR` | `physics_constraint_color.cu` | `led_astar.ptx:astar_expand_node` |
| 6 | XPBD Predict | `0x158 PH_PREDICT_POS` | `physics_xpbd_predict.cu` | — |
| 7 | XPBD Solve (×N) | `0x153 PH_XPBD_SOLVE` | `physics_xpbd_solve.cu` | `gre_defeasible_resolver.cu:resolve_defeasible_constraint` |
| 8 | Integrate | `0x154 PH_INTEGRATE` | `physics_integrate.cu` | quaternion from `trm_extensions.ptx` |
| 9 | Sleep Check | `0x155 PH_SLEEP_CHECK` | `physics_sleep_island.cu` | `sleep_cluster_refiner.cu`, `__ballot_sync` |
| 10 | Galaxy Write | `0x156 PH_GALAXY_WRITE` | `physics_collision_event_write.cu` | `galaxy_memory_updater.cu` |

---

## 3. Complete Physics Opcode Table (0x150–0x17F)

| Opcode | Name | Stack In | Stack Out | Kernel | Notes |
|---|---|---|---|---|---|
| `0x150` | `PH_BROAD_PHASE` | dt, body_count | pair_count | `physics_broad_phase_sap.cu` | Morton update + SAP; reuses `morton_octree.ptx` |
| `0x151` | `PH_NARROW_PHASE` | pair_count | contact_count | `physics_narrow_phase_gjk.cu` | GJK/EPA; warp-cooperative simplex |
| `0x152` | `PH_CONSTRAINT_GENERATE` | contact_count | constraint_count | `physics_constraint_generate.cu` | Contact → XPBD constraints |
| `0x153` | `PH_XPBD_SOLVE` | iter_count | error | `physics_xpbd_solve.cu` | One Jacobi iteration per color; adapts defeasible resolver |
| `0x154` | `PH_INTEGRATE` | dt | 0 | `physics_integrate.cu` | Symplectic Euler; quaternion integrate |
| `0x155` | `PH_SLEEP_CHECK` | energy_threshold | island_count | `physics_sleep_island.cu` | `__ballot_sync` warp vote + multi-pass |
| `0x156` | `PH_GALAXY_WRITE` | — | edge_count | `physics_collision_event_write.cu` | Ring buffer → galaxy_memory_updater |
| `0x157` | `PH_MATERIAL_FETCH` | star_id | friction, restitution, density | inline in solve kernel | Galaxy star fetch; no new kernel |
| `0x158` | `PH_PREDICT_POS` | dt | 0 | `physics_xpbd_predict.cu` | Store predicted pos in separate SOA |
| `0x159` | `PH_CONSTRAINT_COLOR` | constraint_count | color_count | `physics_constraint_color.cu` | Graph coloring via `led_astar.ptx` |
| `0x15A` | `PH_IMPULSE_PROPAGATE` | island_id | 0 | inline via `gre_graph_crystallizer.cu` | Multi-hop impulse wave |
| `0x15B` | `PH_RESTITUTION_APPLY` | contact_count | 0 | bundled in solve | Restitution impulse (post-solve) |
| `0x15C` | `PH_FRICTION_APPLY` | contact_count | 0 | bundled in solve | Friction impulse within friction cone |
| `0x15D` | `PH_ISLAND_WAKE` | trigger_star_id | woken_count | `physics_sleep_island.cu` | Wake island on external event |
| `0x15E` | `PH_BODY_SPAWN` | star_id, pos, vel | body_idx | `physics_spawn.cu` | Add body from Galaxy star |
| `0x15F` | `PH_BODY_DESPAWN` | body_idx | 0 | `physics_spawn.cu` | Remove body, update Galaxy |
| `0x160` | `PH_GRAVITY_APPLY` | G_star_id | 0 | bundled in integrate | Fetch G from Reality Galaxy star |
| `0x161` | `PH_COLLISION_QUERY` | ray_origin, ray_dir | body_idx, t | `physics_raycast.cu` | Ray vs physics bodies |
| `0x162` | `PH_TERNARY_CLASSIFY` | body_idx | +1/0/-1 | bundled in sleep | Ternary sleep state |
| `0x163–0x16F` | Reserved: cloth/rope | — | — | — | XPBD distance + bending (Phase 2) |
| `0x170–0x177` | Reserved: fluid | — | — | — | SPH particle (Phase 3) |
| `0x178–0x17F` | Reserved: soft-body | — | — | — | Volumetric XPBD (Phase 3) |

---

## 4. SOA Memory Layout

### 4.1 PhysicsBodySOA

All arrays are `float4` or `float2` for 128-bit coalesced warp loads. **Nemotron's validated layout:**

```c
// knowledge3d/cranium/kernels/physics_body_soa.h
// 128-bit coalesced loads — each warp reads 32 bodies × 16 bytes = 512 bytes
typedef struct {
    // Pack 1: Position + Inverse Mass (float4, 16B/body)
    float4* pos_inv;          // xyz=position, w=inv_mass (0.0=static)

    // Pack 2: Linear Velocity + Sleep Accumulator (float4, 16B/body)
    float4* vel_sleep;        // xyz=velocity, w=sleep_energy_accum

    // Pack 3: Orientation Quaternion (float4, 16B/body)
    float4* orientation;      // xyzw quaternion (normalized)

    // Pack 4: Angular Velocity + Angular Damping (float4, 16B/body)
    float4* ang_vel_damp;     // xyz=angular_velocity, w=ang_damping

    // Pack 5: Local Inverse Inertia (diagonal) + Restitution (float4, 16B/body)
    // World-space: I_world = R * diag(inv_inertia_local) * R^T (computed on-the-fly)
    float4* inv_inertia_rest; // xyz=inv_inertia_local (principal axes), w=restitution

    // Pack 6: Galaxy Handles (float2 as uint2, 8B/body)
    uint2* galaxy_handles;    // x=material_star_id, y=shape_star_id

    // Pack 7: Island Data + Flags (uint32, 4B/body)
    uint32_t* island_flags;   // [31:8]=island_id, [7]=sleeping, [6]=static,
                              // [5]=kinematic, [4]=dirty, [3]=trigger, [2:0]=reserved

    // Pack 8: Bounding + Friction (float2, 8B/body)
    float2* bound_friction;   // x=bbox_sphere_radius, y=friction_coeff

    // Metadata (scalar — not per-body)
    uint32_t body_count;
    uint32_t capacity;
} PhysicsBodySOA;
// Total: 88 bytes per body across 8 arrays. Each array is 128-bit aligned.
```

**Key correctness note (GLM):** Never store full 3×3 world-space inertia tensor. Store `inv_inertia_local` (3 diagonal values). Compute world-space in constraint kernel:
```c
// In constraint solve kernel (per-body):
float3 invI_local = load_inv_inertia_rest(body_id).xyz;
float4 q = load_orientation(body_id);
float3x3 R = quat_to_rotation_matrix(q);
float3x3 invI_world = R * diag3(invI_local) * transpose(R);
// ~30 FLOPs, saves 24 bytes/body vs. storing full 3x3
```

### 4.2 ContactManifoldSOA

Ring buffer with power-of-2 capacity. Persistent contacts survive frames.

```c
// knowledge3d/cranium/kernels/contact_manifold_soa.h
#define CONTACT_CAPACITY 65536  // Power of 2; tune for scene complexity

typedef struct {
    // Body pair (8B/contact)
    uint32_t* body_a_id;
    uint32_t* body_b_id;

    // Contact point (12B → 3 arrays)
    float* contact_x;
    float* contact_y;
    float* contact_z;

    // Contact normal from A→B (12B → 3 arrays)
    float* normal_x;
    float* normal_y;
    float* normal_z;

    // XPBD state (20B)
    float* penetration_depth;
    float* lambda_normal;       // Lagrange multiplier (warm-started across frames)
    float* lambda_tangent0;     // Friction axis 0
    float* lambda_tangent1;     // Friction axis 1
    float* compliance_normal;   // 0 = rigid contact

    // Persistence
    uint32_t* persistent_id;    // Globally unique; allocated by atomic counter
    uint8_t*  frame_stamp;      // Frame counter; stale = frame_stamp < current - 1
    uint8_t*  color_id;         // Graph-coloring color; set by PH_CONSTRAINT_COLOR

    // Ring buffer metadata
    uint32_t capacity;
    uint32_t write_head;        // Atomic counter (% capacity for slot allocation)
    uint32_t persistent_counter; // Global ID allocator
} ContactManifoldSOA;
```

### 4.3 CollisionEventQueue

Written by physics kernel; consumed by `galaxy_memory_updater.cu` to create Layer-2 Galaxy edges.

```c
// knowledge3d/cranium/kernels/collision_event_queue.h
#define COLLISION_QUEUE_CAPACITY 4096

typedef struct {
    uint32_t* body_a_id;
    uint32_t* body_b_id;
    uint32_t* material_a_star_id;  // Layer-2 star to link in Galaxy edge
    uint32_t* material_b_star_id;
    float*    impulse_magnitude;
    float*    normal_x;
    float*    normal_y;
    float*    normal_z;
    uint32_t  write_head;   // Atomic increment; reset each frame
    uint32_t  capacity;
} CollisionEventQueue;
```

---

## 5. Galaxy Star Schemas

### 5.1 Physical Constants (Reality Galaxy, Layer-2)

One star per physical constant. Fields:

```python
{
    "star_id": "physics_constant_gravitational",
    "facet": "physical_constant",
    "symbol": "G",
    "value_f64_hi": 6.674e-11,   # double precision split (f32 not sufficient)
    "value_f64_lo": 0.0,
    "si_units": "m^3 kg^-1 s^-2",
    "uncertainty": 2.2e-15,
    "surface_forms": {"en": "gravitational constant", "pt": "constante gravitacional"},
}
```

Stars to create: G, c, ħ, k_B, ε₀, μ₀, σ (Stefan-Boltzmann), N_A, e (charge), m_e, m_p.

Fetch from physics kernels via `GALAXY_SIMILARITY (0xE1)` or dedicated `PH_MATERIAL_FETCH (0x157)`.

### 5.2 Physical Material (Reality Galaxy, Layer-2)

One star per material type (steel, wood, rubber, ice, etc.).

```python
{
    "star_id": "physics_material_steel",
    "facet": "physical_material",
    "density": 7850.0,         # kg/m^3
    "restitution": 0.25,
    "friction_static": 0.74,
    "friction_dynamic": 0.57,
    "young_modulus": 200e9,    # Pa (for soft-body extension)
    "poisson_ratio": 0.29,
    "visual_rpn_addr": "...",  # Layer-1 Drawing Galaxy link (material appearance)
    "base_material_star_id": None,  # Parent for inheritance
}
```

### 5.3 Force Law Program (Grammar Galaxy, Layer-3)

Stored in Grammar Galaxy — `physics_rpn_addr` field (NOT `behavior_rpn`).

```
# Example: Gravity force law RPN program
LOAD_STAR physics_constant_gravitational  # push G
STACK_SWAP
PHYSICS_GET_MASS body_a
PHYSICS_GET_MASS body_b
OP_MUL                                    # G * m_a * m_b
PHYSICS_GET_DISTANCE body_a body_b        # r
DUP
OP_MUL                                    # r^2
OP_DIV                                    # F = G*m_a*m_b / r^2
PHYSICS_APPLY_FORCE_PAIR body_a body_b    # apply along connecting vector
```

### 5.4 Rigid Body Star (Reality Galaxy, Layer-2)

Physics objects in the House are Reality Galaxy stars.

```python
{
    "star_id": "house_object_wooden_box_001",
    "facet": "rigid_body",
    "material_star_id": "physics_material_wood",
    "shape_star_id": "drawing_box_0.5x0.5x0.5",   # Layer-1 Drawing Galaxy
    "physics_rpn_addr": "physics_law_gravity_rpn",  # Layer-3 Grammar Galaxy
    "mass": 1.5,                # kg (redundant with material+volume; kept for fast boot)
    "position_x": 0.0,
    "position_y": 2.0,
    "position_z": 0.0,
    "is_sleeping": False,
}
```

### 5.5 Sleep/Wake Meta-Rule (Layer-4)

```python
{
    "star_id": "physics_meta_sleep_policy_default",
    "facet": "meta_rule",
    "layer": 4,
    "energy_sleep_threshold": 0.001,   # kinetic + angular energy below this → sleep
    "frames_before_sleep": 60,
    "wake_impulse_threshold": 0.01,    # impulse above this wakes the island
    "strategy_rpn_addr": "...",        # RPN program returning: sleep/wake decision
}
```

---

## 6. Kernel Specifications

### 6.1 physics_update_morton_codes.cu + physics_broad_phase_sap.cu (PH_BROAD_PHASE)

**Reuse:** `morton_octree.ptx:morton_encode_point` — call directly for body positions.

**Delta optimization (Nemotron):** Only recompute Morton codes for bodies with `position_delta > threshold`. Use `island_flags[4] (dirty bit)` to mark moved bodies. Skip sleeping bodies entirely.

```c
// Per-body thread:
if (body.dirty || !body.sleeping) {
    new_morton = morton_encode_point(body.pos);
    // write sorted array; bitonic sort → SAP sweep → output pairs to ContactManifoldSOA
}
```

**SAP output:** Pairs written to `ContactManifoldSOA` ring buffer via `atomicAdd(&write_head, 1) % CONTACT_CAPACITY`.

### 6.2 physics_narrow_phase_gjk.cu (PH_NARROW_PHASE)

**Warp-cooperative GJK (Nemotron's pattern):**

Each warp processes one potential pair. Four threads hold one simplex vertex each (GJK simplex = up to 4 vertices). Use `__shfl_sync` to share vertex data without shared memory. `__any_sync` for early-out when all threads converge.

```ptx
// PTX warp-cooperative pattern:
.setp.lt.u32  %p1, %iter_count, %max_iter;
.any.sync      %p2, %p1, 0xffffffff;  // any thread needs more iterations?
@%p2  bra     gjk_loop;
// All threads done → EPA for penetration depth
```

### 6.3 physics_constraint_generate.cu (PH_CONSTRAINT_GENERATE)

Converts narrow-phase contact data → XPBD constraints in `ContactManifoldSOA`.

**Persistent contact matching:** Compare new contacts against previous frame via `persistent_id`. Warm-start `lambda_normal` from prior frame (reduces solver iterations by ~40%). Allocate new IDs via `atomicAdd(&persistent_counter, 1)`.

**Friction clustering via cosine_similarity.ptx:** Group contact normals into friction cone axes. Reuse `cosine_similarity_batch` to cluster — reduces friction constraints by grouping nearly-parallel normals.

### 6.4 physics_constraint_color.cu (PH_CONSTRAINT_COLOR)

Graph coloring for parallel XPBD. Reuses `led_astar.ptx:astar_expand_node`.

- Nodes = constraints
- Edges = shared body (two constraints sharing a body cannot be same color)
- Output: `color_id` written to each contact's `ContactManifoldSOA.color_id`
- Typically 2–6 colors for rigid scenes

Solve in `PH_XPBD_SOLVE`: one kernel launch per color (parallel within color, sequential across colors = Gauss-Seidel convergence).

### 6.5 physics_xpbd_solve.cu (PH_XPBD_SOLVE)

**Adapts `gre_defeasible_resolver.cu:resolve_defeasible_constraint`** — replace defeasible logic with XPBD position-correction:

```c
// Per-constraint thread (same color):
float C = contact.penetration_depth;           // Constraint error
float w_a = body_a.inv_mass + ...              // Effective mass
float delta_lambda = (-C - compliance * lambda) / (w_a + compliance);
lambda += delta_lambda;
// Apply positional corrections to body A and B
```

**Warp reduction for global error (Nemotron):** After each color pass, use `__shfl_xor_sync` to sum `|C|` across warp. Broadcast sum to decide early termination.

```ptx
.shfl.xor.sync  %r1, %error, 1, 0xffffffff;   // butterfly reduction
.shfl.xor.sync  %r1, %r1,    2, 0xffffffff;
.shfl.xor.sync  %r1, %r1,    4, 0xffffffff;
// ... 5 more levels → %r1 = warp-sum of error
```

### 6.6 physics_integrate.cu (PH_INTEGRATE)

**Symplectic Euler** (energy-conserving for oscillatory systems):

```
v_new = v_old + dt * F / m
x_new = x_old + dt * v_new   // NOT v_old — symplectic order
```

**Quaternion integration (GLM correctness fix):**

```c
// NOT simple Euler — must use quaternion exponential map:
float3 half_omega = 0.5f * dt * body.angular_velocity;
float4 dq = {half_omega.x, half_omega.y, half_omega.z, 1.0f};
body.orientation = quat_normalize(quat_multiply(dq, body.orientation));
```

### 6.7 physics_sleep_island.cu (PH_SLEEP_CHECK)

**Warp ballot (Nemotron's pattern):**

```ptx
// Thread per body; check if energy below threshold:
.setp.lt.f32  %p1, %energy, 0.001;         // energy < threshold?
.ballot.sync  %r2, %p1, 0xffffffff;        // vote across warp
.popc.b32     %r3, %r2;                    // count sleeping in warp
.setp.eq.u32  %p2, %r3, 32;               // entire warp sleeping?
```

**Multi-warp islands:** Use `sleep_cluster_refiner.cu` for islands spanning multiple warps. Reuse its existing cluster-refinement logic: bodies → island_id → cluster → mark all sleeping.

### 6.8 physics_collision_event_write.cu (PH_GALAXY_WRITE)

Drains `CollisionEventQueue` ring buffer → writes collision edges to Galaxy via `galaxy_memory_updater.cu`.

Each collision event becomes a **Layer-2 Galaxy edge**:
- Source: `body_a.material_star_id`
- Target: `body_b.material_star_id`
- Edge type: `collision_contact`
- Edge metadata: impulse magnitude, contact normal, frame timestamp

Galaxy edge schema (existing `galaxy_memory_updater` format — no schema change needed):
```python
{
    "edge_type": "collision_contact",
    "source_star_id": material_a_star_id,
    "target_star_id": material_b_star_id,
    "weight": impulse_normalized,  # 0..1
    "contact_normal": [nx, ny, nz],
    "frame": current_frame,
}
```

Sleep-time consolidation (SLEEPTIME_PROTOCOL) will cluster these edges: repeated collision patterns between the same material pair → strengthen the edge weight → TRM learns "wood-on-wood collisions are common in this environment."

---

## 7. trm_step_fused.ptx Wiring

Extend `trm_step_fused.ptx` signature and body:

```ptx
.visible .entry trm_step_fused(
    ...existing params...,
    .param .u64 physics_soa_ptr,      // NEW: PhysicsBodySOA*
    .param .u64 contact_soa_ptr,      // NEW: ContactManifoldSOA*
    .param .u64 event_queue_ptr,      // NEW: CollisionEventQueue*
    .param .u32 body_count,           // NEW
    .param .f32 physics_dt,           // NEW: physics timestep
    .param .u32 solver_iterations     // NEW: XPBD iterations per step
)
{
    // ... existing TRM state loading ...

    // ── SWARM_PHASE (existing) ────────────────────────────────────
    call nine_chain_swarm_kernel(...);

    // ── PHYSICS_PHASE (NEW) ───────────────────────────────────────
    // Dispatched via RPN program at physics_rpn_addr in Grammar Galaxy
    // Typical program sequence (dispatched per opcode):
    call modular_rpn_kernel_physics_phase(
        physics_soa_ptr, contact_soa_ptr, event_queue_ptr,
        body_count, physics_dt, solver_iterations
    );
    // → internally calls PH_BROAD_PHASE → PH_NARROW_PHASE → PH_CONSTRAINT_GENERATE
    //   → PH_CONSTRAINT_COLOR → PH_PREDICT_POS → PH_XPBD_SOLVE (×N)
    //   → PH_INTEGRATE → PH_SLEEP_CHECK → PH_GALAXY_WRITE

    // ── DRAW_PHASE (existing) ─────────────────────────────────────
    call drawing_primitives(...);
    call frustum_cull_simd(...);
    call dynamic_lod_tune(...);
}
```

---

## 8. NVIDIA Warp Ingestion (Ingestion Path Only)

NVIDIA Warp uses `wp.sim.ModelBuilder` as its physics scene definition API. K3D can import Warp scene definitions via ingestion (not hot path):

```python
# knowledge3d/ingestion/warp_importer.py  (ingestion-path ONLY)
# Reads NVIDIA Warp ModelBuilder state and converts to Galaxy stars

def import_warp_model(model):
    """Convert warp.sim.Model → Reality Galaxy rigid body stars."""
    for i, body in enumerate(model.bodies):
        star = {
            "star_id": f"warp_body_{i}",
            "facet": "rigid_body",
            "mass": model.body_mass[i],
            "shape_star_id": warp_shape_to_drawing_star(model.shapes[i]),
            "material_star_id": warp_material_to_reality_star(model.shape_materials[i]),
            "physics_rpn_addr": "physics_law_default_gravity",
        }
        yield star
```

NVIDIA Omniverse USD physics schema (`UsdPhysics`) can similarly be parsed at ingestion. All ingestion-path tools (numpy, warp Python SDK, pxr) are allowed — they run once to populate Galaxy, not in the hot path.

---

## 9. New RPN Cross-Domain Programs (for Physics)

### 9.1 Velocity-to-Color (Physics + Drawing)

```rpn
PH_BROAD_PHASE 0x150 dt body_count     # run physics
PH_INTEGRATE   0x154 dt                # update positions
LOAD_GALAXY    E0 body_star_id         # fetch body star
PH_MATERIAL_FETCH 0x157                # push friction, restitution, density
VEC_NORMALIZE  C1                      # normalize velocity vector
COSINE_BATCH   C4 reference_dir        # map speed to 0..1 scalar
LOAD_GALAXY    E0 drawing_star_id      # fetch visual star
DRAW_SET_COLOR                         # set color in Drawing Galaxy star
```

### 9.2 Collision-Triggered Behavior Dispatch (Physics + Grammar)

```rpn
PH_NARROW_PHASE 0x151 pair_count       # detect contacts
PUSH_CONST 0.0                         # threshold
GT                                     # contact_count > 0?
BRANCH_IF_NONZERO collision_handler
  LOAD_GALAXY E0 body_a_star_id
  GET_FIELD behavior_rpn               # fetch collision behavior program
  EXECUTE_RPN                          # run it (Grammar Galaxy Layer-3)
  PH_GALAXY_WRITE 0x156                # write collision edge to Layer-2
END_BRANCH collision_handler
```

### 9.3 Sleep-Time Collision Learning (Physics + Sleep Consolidation)

```rpn
# During sleep_time_micro.ptx phase
PH_GALAXY_WRITE 0x156                  # drain collision event queue
GALAXY_SCAN E2 collision_contact_edges # load recent collision edges
TEMPORAL_AGGREGATE F2 last_1000_frames # cluster by time window
CLUSTER C4 contact_normal              # group by collision angle
UPDATE_GALAXY_EDGE weight              # strengthen high-frequency pairs
SLEEP_CLUSTER_REFINER                  # crystallize into material interaction rules
```

### 9.4 Physics-Constrained Pathfinding (Physics + LED-A*)

```rpn
# Pathfinding that avoids physics-active regions
LED_ASTAR start_pos end_pos           # existing navigation
PH_BROAD_PHASE 0x150 dt body_count    # get current active bodies
PH_COLLISION_QUERY 0x161 ray pos dir  # cast ray along path
BRANCH_IF_NONZERO replan              # if path intersects physics body, replan
  LED_ASTAR_REPLAN                    # reroute
END_BRANCH replan
```

---

## 10. Files to Create (Codex P0 Handoff)

### P0 — Blocking (nothing works without these)

| File | What | Success Criterion |
|---|---|---|
| `knowledge3d/cranium/kernels/physics_body_soa.h` | Header: `PhysicsBodySOA`, `ContactManifoldSOA`, `CollisionEventQueue` struct definitions | All downstream kernels compile with this header |
| `knowledge3d/cranium/kernels/physics_broad_phase_sap.cu` | Morton update (calls `morton_octree.ptx`) + SAP pair output | 1000 bodies, 60 FPS, zero duplicate pairs |
| `knowledge3d/cranium/kernels/physics_narrow_phase_gjk.cu` | Warp-cooperative GJK; outputs `penetration_depth` + contact data | Cube vs. plane: correct penetration within 1e-4 |
| `modular_rpn_kernel.cu` (extend existing) | Add cases `0x150–0x162` as meta-dispatch wrappers | `echo "150\n154\n155" | rpn_exec` returns without crash |
| `knowledge3d/cranium/kernels/physics_integrate.cu` | Symplectic Euler + quaternion integration | Dropped sphere falls 4.9m in 1.0s under G |

### P1 — Score Engine

| File | What | Success Criterion |
|---|---|---|
| `knowledge3d/cranium/kernels/physics_xpbd_solve.cu` | XPBD per-color Jacobi; adapts `gre_defeasible_resolver.cu` | Stack of 10 boxes stable at rest < 1mm drift |
| `knowledge3d/cranium/kernels/physics_sleep_island.cu` | `__ballot_sync` sleep detect + `sleep_cluster_refiner.cu` | Sleeping island: zero CPU wakeup, GPU vote only |
| `knowledge3d/cranium/kernels/physics_collision_event_write.cu` | Drain `CollisionEventQueue` → `galaxy_memory_updater` | Post-collision, Galaxy edge exists for material pair |
| `knowledge3d/cranium/kernels/physics_constraint_color.cu` | Graph coloring via `led_astar.ptx` | 100 constraints colored in ≤ 6 colors |
| `knowledge3d/ingestion/warp_importer.py` | NVIDIA Warp ModelBuilder → Reality Galaxy stars | 10-body Warp scene imports as 10 Galaxy stars |

### P2 — House Integration

| File | What | Success Criterion |
|---|---|---|
| `knowledge3d/cranium/ptx/trm_step_fused.ptx` (extend) | Add PHYSICS_PHASE between SWARM and DRAW | Full game loop runs: SWARM → PHYSICS → DRAW at 60 FPS |
| Reality Galaxy stars: G, c, ħ, k_B, ε₀ | Ingest physical constants as Galaxy stars | `GALAXY_SCAN 0xE2 facet:physical_constant` returns 11 stars |
| Grammar Galaxy stars: gravity law RPN | Gravity force law as `physics_rpn_addr` program | Falling body uses Galaxy-fetched G, not hardcoded |
| Sleep-time consolidation hook | Wire collision edges into `sleep_time_micro.ptx` | After 1000 frames, material pair edge weights updated |

---

## 11. Sovereignty Audit

| Data Flow | Sovereign? | Notes |
|---|---|---|
| Body position → Morton encode → SAP pairs | ✅ | PTX only; `morton_octree.ptx` reused |
| GJK narrow phase → penetration depth | ✅ | `physics_narrow_phase_gjk.cu` PTX |
| Constraint solve → position correction | ✅ | `physics_xpbd_solve.cu` adapts defeasible resolver |
| Material properties → Galaxy star fetch | ✅ | `GALAXY_SIMILARITY (0xE1)` in kernel |
| Collision event → Galaxy edge | ✅ | `galaxy_memory_updater.cu` pipeline |
| Sleep island detection | ✅ | `__ballot_sync` warp vote; no Python |
| NVIDIA Warp ModelBuilder import | ✅ Ingestion | Allowed: ingestion path, runs once |
| Python orchestrating physics sub-phases | ❌ VIOLATION | ALL sub-phases dispatched via RPN program `physics_rpn_addr` from Grammar Galaxy; Python never calls individual kernels |
| Gravity constant hardcoded in kernel | ❌ VIOLATION | Must fetch G from Reality Galaxy `physics_constant_gravitational` star via `PH_MATERIAL_FETCH` |

---

## 12. Open Issues (GLM Prioritized List, Resolved)

| Issue | Resolution |
|---|---|
| RPN-SOA mismatch | Dual-dispatch meta-opcode pattern (§1.2) — mirrors GALAXY_SCAN |
| ContactManifold pointer coalescing | Separate float arrays per component (§4.2) |
| StackValue tag lane convention | Physics opcodes pop/push `float4` with `.w=tag` per existing convention |
| Full 3×3 inertia tensor | Store local diagonal only; compute world-space on-the-fly (§6.1) |
| XPBD parallelization | Graph coloring; per-color Jacobi (§6.4) |
| Quaternion normalization | Quaternion exponential map in `physics_integrate.cu` (§6.6) |
| Contact ring buffer overflow | Ring buffer with power-of-2 capacity; oldest contacts evicted by frame stamp |
| Multi-warp sleep islands | `sleep_cluster_refiner.cu` integration (§6.7) |
| Galaxy edge write from PTX | `CollisionEventQueue` → `galaxy_memory_updater` pipeline (§6.8) |
| Persistent contact ID allocation | `atomicAdd(&persistent_counter, 1)` in `physics_constraint_generate.cu` |

---

## 13. Relationship to Prior Spec (CODEX_SOVEREIGN_PHYSICS_SPEC.md)

| Item | Prior Spec | v2 Update |
|---|---|---|
| Solver | PGS (Projected Gauss-Seidel) | **XPBD** preferred (convergence stability per Daniel's direction) |
| Inertia | Full 3×3 or unspecified | **Local diagonal only** + on-the-fly world-space transform |
| Contact structure | Single `float3*` pointer | **Separate float arrays per axis** (SOA coalescing) |
| RPN-SOA bridge | Unresolved | **Meta-dispatch pattern** mirroring GALAXY_SCAN |
| Kernel reuse | Stated abstractly | **Concrete entry points** named: `morton_encode_point`, `astar_expand_node`, `resolve_defeasible_constraint`, etc. |
| Warp primitives | Mentioned `__ballot_sync` | **Full PTX patterns** for ballot, shfl_xor, any_sync, cooperative GJK |
| NVIDIA adaptation | General intent | **Concrete**: Warp ModelBuilder ingestion path + USD physics schema |

---

*Handoff to Codex: implement P0 files first. Physics is not live until `physics_broad_phase_sap.cu` + `modular_rpn_kernel.cu` case extensions + `physics_integrate.cu` exist and the falling-sphere test passes.*
