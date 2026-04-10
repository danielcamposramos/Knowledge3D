# CODEX IMPLEMENTATION SPEC — Sovereign K3D Physics Engine
**Date:** 2026-04-07  
**Source:** MVCIC Chain (Kimi → Qwen → GLM → DeepSeek), synthesized by Claude Code  
**Assignee:** Codex  
**License:** Apache 2.0  
**Sovereignty:** Zero external physics dependency. All compute in PTX/CUDA. No PhysX, Bullet, or any runtime lib.

---

## Architecture Statement

The Sovereign K3D physics engine treats every Galaxy star as a first-class physical entity governed by a deterministic RPN-driven impulse solver over a 128-byte-aligned Structure-of-Arrays layout (up to 4096 stars in a free-list pool), where broadphase leverages Morton-ordered swept AABBs for built-in CCD, narrowphase computes shape-specific manifolds via GJK/EPA against a constant-memory shape table, and the PGS solver integrates per-star friction and restitution as material properties; collision events feed asynchronously into the House knowledge system which infers latent physical traits and governs adaptive star spawning — closing the loop between simulation and semantics — while the Python bridge remains a sovereign client accessing state only through validated byte-offset RPCs, and the full opcode space 0x150–0x17F covers broadphase, force eval, narrowphase, solve, integrate, draw, spawn, despawn, material update, query, and event-sync operations, all running inside the existing `trm_step_fused.ptx` game loop between SWARM_PHASE and DRAW_PHASE.

---

## 1. New Files to Create

### 1.1 CUDA Kernel Files (`knowledge3d/cranium/kernels/`)

| File | Purpose |
|---|---|
| `physics_rpn_extensions.cu` | Dispatch table for opcodes 0x150-0x17F; extends `modular_rpn_kernel.cu` |
| `k3d_physics_galaxy.cu` | Root physics scheduler; owns the full step pipeline; compiled to PTX |
| `k3d_broad_phase_morton.cu` | Swept-AABB broadphase on Morton-sorted stars; reuses nav octree codes |
| `k3d_narrow_phase_gjk.cu` | GJK distance + EPA penetration depth; shape-aware via shape table |
| `k3d_force_grammar.cu` | Evaluates `behavior_rpn` of every force/constraint star; outputs wrench vec |
| `k3d_constraint_resolver.cu` | PGS velocity solver with restitution + Coulomb friction; calls CAS SOLVE(0x122) |
| `k3d_integrator_chain.cu` | Symplectic Euler per star; quaternion renormalization mandatory post-step |
| `k3d_query_filter.cu` | RAYCAST and SHAPE_OVERLAP queries; separate from simulation loop |
| `k3d_sleep_islands.cu` | Morton-prefix warp-vote sleep detection; island wake on neighbor proximity |
| `k3d_collision_events.cu` | Double-buffered event ring; writes CollisionEvent on impulse > ε |

### 1.2 Bridge File (`knowledge3d/cranium/bridges/`)

| File | Purpose |
|---|---|
| `physics_galaxy_bridge.py` | Sovereign ctypes bridge; byte-offset access only; `load_kernels()`, `step()`, `spawn_star()`, `read_events()` |

### 1.3 PTX Entry (`knowledge3d/cranium/ptx/`)
- `k3d_physics_galaxy.ptx` — compiled output of `k3d_physics_galaxy.cu`; hooked into `trm_step_fused.ptx`

---

## 2. RPN Opcode Table — Physics Range 0x150–0x17F

### 2.1 Dispatch Extension in `modular_rpn_kernel.cu`

Add to the opcode switch BEFORE implementing any physics kernel:

```cpp
case 0x150 ... 0x17F: {
    dispatch_physics_subop(opcode, stack, physics_soa_ptr);
    break;
}
```

`dispatch_physics_subop` lives in `physics_rpn_extensions.cu`.

### 2.2 Complete Opcode Table

| Opcode | Mnemonic | Stack Signature | Phase | Notes |
|---|---|---|---|---|
| 0x150 | `BROAD` | – → contact_count | BROAD | Morton sweptAABB sort + __clz overlap; CCD built-in |
| 0x151 | `FORCE_EVAL` | – → – | FORCE | Run behavior_rpn of all force stars; accumulate wrench into SOA |
| 0x152 | `NARROW` | (pair_buf) → manifold_count | NARROW | GJK + EPA per pair; shape-aware; ternary output |
| 0x153 | `SOLVE` | (manifold_buf, iterations) → – | SOLVE | PGS: normal impulse + Coulomb friction + Baumgarte bias |
| 0x154 | `INTEGRATE` | (dt) → – | INTEGRATE | Symplectic Euler; quaternion renorm mandatory |
| 0x155 | `DRAW_SYNC` | – → – | DRAW | Atomic swap of double-buffered transforms for renderer |
| 0x156 | `SPAWN_STAR` | (mass, pos, shape_handle, friction, restitution) → star_id | SPAWN | Pop from free_list; init all SOA fields |
| 0x157 | `DESPAWN_STAR` | (star_id) → – | DESPAWN | Push to free_list; zero SOA slot |
| 0x158 | `SET_SHAPE` | (star_id, shape_handle) → – | STATE | Update shape ref; re-compute AABB |
| 0x159 | `SET_MATERIAL` | (star_id, friction_u8, restitution_u8) → – | STATE | Per-star material update |
| 0x15A | `CLEAR_EVENTS` | – → – | MAINT | GC event_tags older than N frames (call every 1024 frames) |
| 0x15B | `READ_EVENT_BUFFER` | – → event_ptr | BRIDGE | Swap event double-buffer; return CPU-visible ptr |
| 0x15C | `WAKE_STAR` | (star_id) → – | STATE | Force wake; clear sleeping bit |
| 0x15D | `SLEEP_STAR` | (star_id) → – | STATE | Force sleep; set sleeping bit |
| 0x15E | `KINEMATIC_SET` | (star_id, bool) → – | STATE | Kinematic flag; immune to forces |
| 0x15F | `PHYSICS_MODE` | (chain_id, mode_u8) → – | META | Set swarm chain subtype: 0=normal, 1=physics_solver |
| 0x160 | `RAYCAST` | (origin, dir, max_dist) → hit_id, t, normal | QUERY | Broadphase → shape-specific ray test → closest hit |
| 0x161 | `SHAPE_OVERLAP` | (shape_handle, pos, orient) → count, id_buf | QUERY | All stars overlapping given shape |
| 0x162 | `PH_TERNARY_CLASSIFY` | (gjk_distance, epsilon) → trit | UTIL | Maps GJK result to PHYSICS_TRUE/FALSE/UNKNOWN/DEGENERATE |
| 0x163–0x17F | RESERVED | – | – | Cloth, fluid, soft body, magnetic fields |

---

## 3. PhysicsGalaxySOA — Complete Final Layout

128 bytes per star. 4096 max. Free-list pool allocation (`free_list[4096]` + atomic `free_head`).

```cpp
#define PHYS_MAX_STARS 4096

// SOA: all arrays are length PHYS_MAX_STARS, 128-byte aligned
typedef struct {
    // Transform (28 bytes × 4096 = 112 KB)
    float  pos_x[PHYS_MAX_STARS];
    float  pos_y[PHYS_MAX_STARS];
    float  pos_z[PHYS_MAX_STARS];
    float  orient_x[PHYS_MAX_STARS];   // quaternion
    float  orient_y[PHYS_MAX_STARS];
    float  orient_z[PHYS_MAX_STARS];
    float  orient_w[PHYS_MAX_STARS];

    // Velocity (24 bytes × 4096 = 96 KB)
    float  linVel_x[PHYS_MAX_STARS];
    float  linVel_y[PHYS_MAX_STARS];
    float  linVel_z[PHYS_MAX_STARS];
    float  angVel_x[PHYS_MAX_STARS];
    float  angVel_y[PHYS_MAX_STARS];
    float  angVel_z[PHYS_MAX_STARS];

    // Force accumulator (12 bytes × 4096 = 48 KB) — zeroed each frame after INTEGRATE
    float  force_x[PHYS_MAX_STARS];
    float  force_y[PHYS_MAX_STARS];
    float  force_z[PHYS_MAX_STARS];

    // Inertia & mass (16 bytes × 4096 = 64 KB)
    float  invInertia_x[PHYS_MAX_STARS]; // diagonal inv inertia tensor
    float  invInertia_y[PHYS_MAX_STARS];
    float  invInertia_z[PHYS_MAX_STARS];
    float  invMass[PHYS_MAX_STARS];      // 0 = static/kinematic

    // Shape (4 bytes × 4096 = 16 KB)
    uint16_t shape_handle[PHYS_MAX_STARS]; // index into constant shape_table[64]
    uint8_t  shape_type[PHYS_MAX_STARS];   // 0=sphere, 1=box, 2=capsule
    uint8_t  integrator[PHYS_MAX_STARS];   // 0=symplectic Euler, 1=RK4

    // Material (2 bytes × 4096 = 8 KB)
    uint8_t  friction[PHYS_MAX_STARS];     // [0-255] → [0.0-2.0] (μ = v/127.5)
    uint8_t  restitution[PHYS_MAX_STARS];  // [0-255] → [0.0-1.0] (e = v/255)

    // State flags (1 byte × 4096 = 4 KB)
    // bit0=kinematic, bit1=sleeping, bit2=awake_pending, bit3=ccd_enabled
    uint8_t  flags[PHYS_MAX_STARS];

    // Morton code (4 bytes × 4096 = 16 KB) — recomputed each broad phase
    uint32_t mortonCode[PHYS_MAX_STARS];

    // Collision filter (4 bytes × 4096 = 16 KB)
    // Layout: (myLayer[7:0] | myGroup[15:8] | maskBits[31:16])
    // Collides if: (A.maskBits & (1 << B.myLayer)) != 0
    uint32_t collisionFilter[PHYS_MAX_STARS];

    // Knowledge-link (4 bytes × 4096 = 16 KB)
    uint32_t galaxyStarId[PHYS_MAX_STARS]; // links physics slot → Galaxy star UUID

    // Event tag (4 bytes × 4096 = 16 KB) — frame of last collision (debounce)
    uint32_t event_tag[PHYS_MAX_STARS];

    // Render (8 bytes × 4096 = 32 KB)
    uint32_t color[PHYS_MAX_STARS];   // ARGB; updated by House knowledge feedback
    float    size[PHYS_MAX_STARS];    // render scale / sphere radius

    // Game attributes (8 bytes × 4096 = 32 KB)
    float    health[PHYS_MAX_STARS];  // ≤0 triggers DESPAWN_STAR
    int32_t  ttl[PHYS_MAX_STARS];     // frames to live; -1 = infinite

    // Double-buffered transforms for renderer (no physics/render race condition)
    float    xform_back[PHYS_MAX_STARS][12];  // 4×3 col-major; physics writes here
    float    xform_front[PHYS_MAX_STARS][12]; // renderer reads here
    uint32_t xform_gen[PHYS_MAX_STARS];       // generation counter; swap on DRAW_SYNC

    // Pool management (host-accessible)
    uint32_t free_list[PHYS_MAX_STARS];
    uint32_t free_head;   // atomic; pop = fetch_add, push = store
    uint32_t star_count;  // active count
} PhysicsGalaxySOA;
```

**Shape table (constant memory, 64 entries):**
```cpp
struct ShapeEntry {
    uint8_t  type;        // 0=sphere, 1=box, 2=capsule
    uint8_t  pad[3];
    float3   params;      // sphere: (r,0,0); box: halfextents; capsule: (r, hh, 0)
};
__constant__ ShapeEntry shape_table[64];
```

---

## 4. Collision Event System

```cpp
struct CollisionEvent {
    uint32_t star_a;      // physics slot index
    uint32_t star_b;
    float3   impulse;     // total impulse applied (energy signature)
    uint32_t frame;       // simulation frame
    uint8_t  shape_a;
    uint8_t  shape_b;
    uint16_t pad;         // 20 bytes total — 4-byte aligned
};

#define COLLISION_EVENT_RING_SIZE 2048
// Double-buffered: kernel writes to [write_buf], bridge reads [read_buf]
// 0x15B (READ_EVENT_BUFFER) atomically swaps buffers
```

**Bridge polling** (every 60 frames):
- Compute `energy = 0.5 * dot(impulse, impulse) * (invMass_a + invMass_b)`
- Look up `galaxyStarId` for both slots → map to House knowledge graph nodes
- Update edge weight by energy; accumulate shape-type frequency per node
- Infer: high-energy + low restitution = dense/brittle; high-freq sphere = gas-like
- Feed back as spawn hints: when `SPAWN_STAR` called, House advises (shape, friction, restitution)

---

## 5. trm_step_fused.ptx Integration

Insert physics island between `SWARM_PHASE` and `DRAW_PHASE`:

```
... existing TRM game loop ...

SWARM_PHASE (nine_chain_swarm — existing reasoning)
    ↓
PHYSICS_PHASE:
    0x150  BROAD          — swept-AABB Morton sort; outputs candidate pair buffer
    0x151  FORCE_EVAL     — run behavior_rpn of all force stars; accumulate wrench
    0x152  NARROW         — GJK+EPA on pairs; outputs manifold buffer
    0x153  SOLVE          — PGS iterations (default 8); applies impulses + friction
    0x154  INTEGRATE      — symplectic Euler; renorm quaternions; zero force accumulator
    0x155  DRAW_SYNC      — atomic swap xform double-buffer
    (async) 0x15B READ_EVENT_BUFFER — if frame % 60 == 0, swap event ring for bridge

DRAW_PHASE (existing rendering — reads xform_front only)
```

**Nine-chain swarm allocation during PHYSICS_PHASE:**
- Chains 0-5: existing reasoning semantics (unchanged)
- Chain 6: `PHYSICS_SOLVER_CHAIN` — PGS solver islands
- Chain 7: `PHYSICS_BROAD_CHAIN` — Morton sort + broadphase
- Chain 8: `PHYSICS_NARROW_CHAIN` — GJK/EPA narrowphase

Set via `PHYSICS_MODE(0x15F)` at start of PHYSICS_PHASE; restored to 0 at end.

---

## 6. Key Implementation Notes

### 6.1 GJK Degeneracy Guard (GLM critical fix)
```cpp
if (simplex_dim < 1 || simplex_volume < 1e-7f) {
    // Fall back to AABB overlap test; skip EPA
    result.type = PHYSICS_DEGENERATE;
    result.distance = aabb_overlap_distance(a, b);
    return result;
}
```
Use `PH_TERNARY_CLASSIFY(0x162)` to map distance to ternary before passing to solver.

### 6.2 Quaternion Renormalization (GLM critical fix)
After every `INTEGRATE`, before updating Morton codes:
```cpp
float4 q = {orient_x[i], orient_y[i], orient_z[i], orient_w[i]};
float inv_len = rsqrtf(dot(q, q));
q = q * inv_len;  // mandatory — prevents NaN drift
```

### 6.3 Morton Broadphase Bit Ordering (GLM correction)
Use **lower Morton bits** for fine-grained spatial bins (leaf level). Upper bits are the coarse BVH used by LED-A* — do NOT partition by upper bits.
- Bin key = `mortonCode[i] & 0x00FFFFFF` (lower 24 bits = leaf cell address)
- Two stars in same bin → candidate pair

### 6.4 behavior_rpn Field Separation
Do NOT reuse the existing `behavior_rpn` star field for physics. Add a new field:
- `physics_rpn_addr`: uint32 pointer into a constant-memory physics bytecode buffer
- This avoids clobbering existing AI reasoning behavior_rpn programs
- Force star bytecode (gravity example): `0x151 0x55 0x20 0x30` (gravity field → fetch mass → mul → apply)

### 6.5 PGS Solver Stack Convention
Before calling `CAS SOLVE(0x122)` for each contact:
```
Stack (top to bottom):
  [0] float3  ContactNormal    (world space, unit vector)
  [1] float   EffectiveMass    (invMassA + invMassB + angular terms)
  [2] float   Bias             (Baumgarte: β/dt * penetration_depth)
  [3] float   RelativeVelocity (dot(vRel, normal))

After SOLVE(0x122):
  [0] float   ComputedImpulse  (clamped to [0, ∞] for contact)
```
Friction impulse computed separately: tangential component clamped to `[-μ|J|, +μ|J|]`.

---

## 7. Python Bridge — `physics_galaxy_bridge.py`

```python
import ctypes, struct
from pathlib import Path
from knowledge3d.cranium.sovereign import loader

class PhysicsGalaxyBridge:
    """Sovereign bridge — accesses PhysicsGalaxySOA via byte offsets only.
    No knowledge of internal SOA layout exposed outside kernel boundary."""

    # SOA field byte offsets (must match PhysicsGalaxySOA layout exactly)
    OFFSET_POS_X   = 0
    OFFSET_INV_MASS = PHYS_MAX_STARS * 16 * 4  # after pos/orient/vel/force/inertia
    STRIDE         = 4  # float32

    def __init__(self, ptx_path: Path, max_stars: int = 4096):
        self.max_stars = max_stars
        self._lib = loader.load_ptx(ptx_path)
        self._soa_ptr = None
        self._event_buf = None

    def load_kernels(self):
        self._soa_ptr = self._lib.k3d_phys_alloc_soa(self.max_stars)
        self._lib.k3d_phys_init_shape_table()

    def spawn_star(self, mass: float, pos: tuple, shape_handle: int,
                   friction: int = 127, restitution: int = 50) -> int:
        """Returns physics slot index. Bridge validates bounds."""
        return self._lib.k3d_phys_spawn(
            self._soa_ptr,
            ctypes.c_float(mass),
            ctypes.c_float(pos[0]), ctypes.c_float(pos[1]), ctypes.c_float(pos[2]),
            ctypes.c_uint16(shape_handle),
            ctypes.c_uint8(friction),
            ctypes.c_uint8(restitution),
        )

    def step(self, dt: float, solver_iterations: int = 8):
        self._lib.k3d_phys_step(self._soa_ptr, ctypes.c_float(dt),
                                ctypes.c_int(solver_iterations))

    def read_collision_events(self) -> list[dict]:
        """Swap event double-buffer; return events as dicts for House knowledge update."""
        count = ctypes.c_uint32(0)
        ptr = self._lib.k3d_phys_read_events(self._soa_ptr, ctypes.byref(count))
        events = []
        for i in range(count.value):
            raw = (ctypes.c_uint8 * 20).from_address(ptr + i * 20)
            star_a, star_b, frame = struct.unpack_from('<IIxxxxI', bytes(raw), 0)
            ix, iy, iz = struct.unpack_from('<fff', bytes(raw), 8)
            energy = 0.5 * (ix*ix + iy*iy + iz*iz)
            events.append({'star_a': star_a, 'star_b': star_b,
                           'frame': frame, 'energy': energy})
        return events

    def despawn_star(self, star_id: int):
        self._lib.k3d_phys_despawn(self._soa_ptr, ctypes.c_uint32(star_id))
```

---

## 8. Test Plan (implement in this order)

| Priority | Test | What it validates |
|---|---|---|
| P0 | `test_soa_alignment` | All SOA arrays 128-byte aligned; static_assert passes |
| P0 | `test_free_list_pool` | spawn 4096 → despawn 2048 → re-spawn 2048; no leaks |
| P0 | `test_rpn_dispatch_0x150` | `BROAD` opcode dispatches without CUDA error |
| P0 | `test_quaternion_renorm` | After 1000 integration steps, all quaternions unit length ±1e-5 |
| P1 | `test_gravity_rpn_bytecode` | Single star, gravity force RPN, step once → linVel_y decreases by 9.81*dt ±ε |
| P1 | `test_morton_sort_coherence` | 4096 random positions → sorted mortonCodes → adjacent codes produce valid pairs |
| P1 | `test_pgs_stack_convention` | Known J/mass/bias/relVel → SOLVE(0x122) → impulse matches analytical |
| P1 | `test_gjk_sphere_sphere` | Two spheres, known separation → GJK returns correct distance |
| P1 | `test_gjk_degenerate` | Coplanar triangle → GJK degenerate path → AABB fallback, no NaN |
| P1 | `test_friction_non_zero` | Box on slope, friction > 0 → box stays; friction = 0 → box slides |
| P2 | `test_ccd_no_tunnel` | High-velocity sphere vs thin wall → contact detected, no pass-through |
| P2 | `test_sleep_island` | 100 resting boxes → sleep after N frames → BROAD skips them |
| P2 | `test_double_buffer_race` | Physics + render threads run concurrently → renderer always reads valid xform |
| P2 | `test_collision_event_knowledge` | Collision → event ring → bridge polls → Galaxy star edge weight updated |
| P2 | `test_raycast_hit` | Ray vs sphere at known pos → correct hit_t, normal |

---

## 9. What NOT to Implement (Codex boundary)

- Do NOT implement cloth, fluid, or soft-body (0x163-0x17F reserved)
- Do NOT implement articulations or ragdolls in this phase
- Do NOT add any external physics library as fallback — zero fallbacks, sovereign only
- Do NOT couple physics step to Python GIL — bridge is async polling only
- Do NOT reuse existing `behavior_rpn` field — use new `physics_rpn_addr` field

---

## 10. Deliverables Checklist

- [ ] `physics_rpn_extensions.cu` — dispatch table 0x150-0x17F wired into modular_rpn_kernel.cu
- [ ] `k3d_physics_galaxy.cu` + compiled `.ptx`
- [ ] `k3d_broad_phase_morton.cu`
- [ ] `k3d_narrow_phase_gjk.cu` (with degeneracy guard)
- [ ] `k3d_force_grammar.cu`
- [ ] `k3d_constraint_resolver.cu` (PGS + friction + Baumgarte)
- [ ] `k3d_integrator_chain.cu` (symplectic Euler + quaternion renorm)
- [ ] `k3d_sleep_islands.cu`
- [ ] `k3d_collision_events.cu` (double-buffered ring)
- [ ] `k3d_query_filter.cu` (RAYCAST + SHAPE_OVERLAP)
- [ ] `physics_galaxy_bridge.py`
- [ ] `trm_step_fused.ptx` updated with PHYSICS_PHASE hook
- [ ] All P0+P1 tests passing

---

*MVCIC chain authors: Kimi (architecture), Qwen (code), GLM (analysis), DeepSeek (synthesis)*  
*Spec synthesized by: Claude Code (Architecture Partner)*  
*Chain file: `TEMP/mvcic_sovereign_physics.md`*
