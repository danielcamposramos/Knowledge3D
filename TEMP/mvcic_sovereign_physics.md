# MVCIC Chain — Sovereign Game-World Physics Engine
**Task:** Design sovereign PTX/CUDA physics pipeline for K3D — zero PhysX dependency, physics objects ARE Galaxy stars
**Partners:** Kimi → Qwen → GLM → DeepSeek
**Orchestrator:** Claude Code

---

## [1/4] Kimi — Architecture & Deep Reasoning

**KIMI (ARCHITECTURE PARTNER 1) – SOVEREIGN K3D PHYSICS ENGINE VISION**
*"We are not simulating matter. We are letting meaning collide."*

### 1. SOVEREIGN CUDA KERNEL FILES

| kernel file | purpose | binds into |
|---|---|---|
| `k3d_physics_galaxy.ptx` | root physics scheduler, owns 0x150-0x17F opcode jump-table | trm_step_fused.ptx |
| `k3d_rigid_star.ptx` | rigid-body state integration (position, quaternion, lin & ang vel) | k3d_physics_galaxy.ptx |
| `k3d_force_grammar.ptx` | evaluates behavior_rpn of every force star, outputs wrench vectors | k3d_rigid_star.ptx |
| `k3d_constraint_resolver.ptx` | iterative impulse/position solver; uses CAS SOLVE(0x122) | k3d_physics_galaxy.ptx |
| `k3d_broad_phase_morton.ptx` | GPU radix-sort + Morton octree collision culling (reuses nav octree) | k3d_constraint_resolver.ptx |
| `k3d_narrow_phase_gjk.ptx` | Bezier-AABB vs Bezier-mesh; returns minimal separating vector | k3d_constraint_resolver.ptx |
| `k3d_integrator_chain.ptx` | symplectic Euler / RK4 / implicit Euler selectable per star via behavior_rpn flag | k3d_rigid_star.ptx |
| `k3d_query_filter.ptx` | physics-aware LOD & reality_atom culling | trm_step_fused.ptx |

### 2. NEW RPN PHYSICS OPCODES (0x150-0x17F)

| opcode | mnemonic | stack signature | description |
|---|---|---|---|
| 0x150 | `RIGID_CREATE` | (behavior_rpn_addr, visual_rpn_addr, mass, morton_key) → star_id | allocates reality_atom as rigid body |
| 0x151 | `RIGID_APPLY_FORCE` | (star_id, force_vec3, application_vec3) → – | adds force to accumulator |
| 0x152 | `RIGID_APPLY_TORQUE` | (star_id, torque_vec3) → – | adds torque |
| 0x153 | `RIGID_SET_VEL` | (star_id, lin_vel_vec3, ang_vel_vec3) → – | direct velocity write |
| 0x154 | `RIGID_GET_VEL` | (star_id) → lin_vel_vec3, ang_vel_vec3 | readback |
| 0x155 | `CONSTRAINT_ADD` | (star_id_A, star_id_B, constraint_rpn_addr) → constraint_id | bilateral constraint |
| 0x156 | `CONSTRAINT_REMOVE` | (constraint_id) → – | destroys constraint |
| 0x157 | `INTEGRATE_STEP` | (dt_fp32, integrator_type_u8) → – | advances all rigids one frame |
| 0x158 | `SOLVER_ITERATIONS` | (iterations_u8) → – | sets solver budget |
| 0x159 | `MORTON_COLLIDE` | – → contact_count | broad-phase |
| 0x15A | `NARROW_GJK` | (star_id_A, star_id_B) → sep_axis_vec3, distance_fp32 | narrowphase |
| 0x15B | `IMPULSE_APPLY` | (star_id, impulse_vec3, contact_vec3) → – | warm-starting |
| 0x15C | `POSITION_CORRECT` | (star_id_A, star_id_B, correction_vec3) → – | Baumgarte stabilization |
| 0x15D | `WAKE_SLEEP` | (star_id, sleep_bool) → – | sleep control |
| 0x15E | `KINEMATIC_SET` | (star_id, kinematic_bool) → – | kinematic flag |
| 0x15F | `MASS_PROPERTIES` | (star_id, inertia_vec3, com_vec3) → – | inertia override |
| 0x160-0x17F | reserved | – | cloth, fluid, articulations |

### 3. GALAXY STAR SCHEMA

```cpp
struct reality_atom_physics_ext {
    float3  com;              // center of mass world
    float4  orientation;      // quaternion
    float3  linVel;
    float3  angVel;
    float3  invInertiaLocal;  // pre-inverted
    float   invMass;
    uint8_t integrator;       // 0=Euler, 1=RK4, 2=implicit
    uint8_t flags;            // bit0=kinematic, bit1=sleeping, bit2=needsBroad
    uint32_t mortonCode;      // updated each step
    uint32_t collisionMask;
};
// behavior_rpn → physics motion law
// law_rpn → collision response law
// Force stars: reality_molecule whose behavior_rpn outputs (force.xyz, torque.xyz)
// Constraint stars: reality_system whose behavior_rpn encodes C(posA,quatA,posB,quatB)=0
```

### 4. HOOK INTO trm_step_fused.ptx

Physics island inserted between SWARM_PHASE and DRAW_PHASE:
BROAD_PHASE → FORCE_EVAL (all force star behavior_rpn) → NARROW_PHASE → SOLVER → INTEGRATE_STEP(0x157) → DRAW_PHASE

### 5. PHYSX → SOVEREIGN MAPPING

| PhysX stage | K3D sovereign replacement |
|---|---|
| Broad-phase (SAP/MBP) | k3d_broad_phase_morton.ptx — reuses nav Morton Octree |
| Narrow-phase (GJK/EPA) | k3d_narrow_phase_gjk.ptx — Bezier-AABB & mesh |
| Constraint solver (PGS) | k3d_constraint_resolver.ptx — CAS SOLVE(0x122) on Jacobian RPN |
| Integration | k3d_integrator_chain.ptx — selectable per star |
| Sleep/wake | WAKE_SLEEP(0x15D) — threshold in law_rpn |
| Events/callbacks | Galaxy grammar events → reality_system event star |

---

### [2/4] Qwen — Code Specialist

[PARTNER FAILED: HTTPConnectionPool(host='192.168.0.4', port=11434): Read timed out. (read timeout=180)]

────────────────────────────────────────────────────────────────

### [3/4] GLM — Analysis & Edge Cases

# GLM Analysis — Physics Engine Validation & Edge Cases

Building on Kimi's architecture (Chain #1). Qwen's slot is empty, so I extend directly from Kimi's foundation.

---

## 1. Critical Correctness Issues

### 1.1 Opcode Range Collision
Kimi assigned opcodes 0x150-0x17F. However:
- Existing CAS opcodes end at 0x123 (LIMIT)
- Gap 0x124-0x14F is **unspecified** in the context
- The RPN VM in `modular_rpn_kernel.cu` may have **hardcoded dispatch tables** that need extension

**Issue**: The RPN interpreter must be modified to recognize the new opcode range before any physics bytecode can execute. This requires changes to `modular_rpn_kernel.cu` that Kimi did not specify.

### 1.2 Ternary Logic Conflict
K3D uses ternary logic (TERNARY_AND 0x100, TERNARY_NOT 0x102). Physics collision detection produces **binary** results (colliding/not-colliding). 

**Gap**: No specification for how collision truth values map to ternary (true/false/unknown). Need a mapping convention:
- Overlapping + distance < 0 → TRUE
- Separated + distance > tolerance → FALSE  
- Within tolerance → UNKNOWN (requires EPA refinement)

### 1.3 Nine-Chain Swarm Allocation Conflict
Kimi assigns chains 0-3 for GJK/EPA, 4-6 for constraint graph, 7 for TGS, 8 for integration. But `nine_chain_swarm_kernel.cu` already has **assigned semantics** for those chains (unspecified in context, but likely occupied).

**Validation needed**: What is the existing chain allocation? Physics either:
(a) Replaces existing chain tasks during physics ticks, or
(b) Runs in a separate kernel launch after the existing swarm completes

Kimi's "parasitize" approach requires explicit conflict resolution.

---

## 2. Edge Cases & Numerical Stability

### 2.1 GJK Terminating Condition
GJK has a known edge case: **degenerate simplices** (coincident points, colinear edges, coplanar triangles). When this occurs:
- Distance query returns incorrect value
- EPA receives malformed simplex and fails

**Required**: `k3d_ph_gjk.ptx` must include degeneracy detection:
```
IF simplexDimension < 1 OR simplexVolume < epsilon:
    RETURN GJK_DEGENERATE (new opcode needed: 0x160)
    SWITCH to backup algorithm (brute-force AABB overlap)
```

### 2.2 EPA Non-Convergence
EPA expands a polytope to find penetration depth. Edge cases:
- **Deep penetration**: polytope must expand many iterations → performance cliff
- **Tunneling**: high-velocity objects pass through each other between frames

**Missing specification**: 
- Maximum EPA iterations per pair (must be bounded for real-time)
- Continuous collision detection (CCD) option for fast-moving bodies
- Tunneling mitigation strategy

### 2.3 TGS Solver Divergence
TGS (temporal Gauss-Seidel) can diverge when:
- Mass ratios exceed ~100:1 (light object constrained to heavy)
- Constraint graph contains cycles with contradictory targets

**Required safeguard**: Divergence detection in `k3d_ph_tgs.ptx`:
```
IF velocityDelta > MAX_VELOCITY_DELTA * previousVelocityDelta:
    FLAG constraint graph as ill-conditioned
    FALLBACK to reduced iteration count with damping
```

### 2.4 Quaternion Normalization Drift
`PH_INTEGRATE_POS` (0x158) integrates quaternion via `q += dt * ω`. This produces non-unit quaternions.

**Missing**: Post-integration renormalization step. Without it, accumulated error causes:
- Rotation matrix scale drift → incorrect inertia tensor application
- Eventually NaN propagation through the entire pipeline

**Required**: After every position integration:
```
q = normalize(q)  // Must be explicit in kernel
```

---

## 3. Consistency Gaps with Existing K3D

### 3.1 Morton Octree vs. SAP Hybrid
Kimi proposes: "partition world into 256 bins via upper 8 bits of Morton"

**Problem**: Morton codes encode spatial position. Upper 8 bits = coarse octree levels = **very large spatial regions** (potentially half the world in bit 0).

**Correction**: Should use **lower** Morton bits for fine-grained binning, or use a separate **spatial hash** for broadphase buckets. Upper bits already define the BVH hierarchy for LED-A*.

### 3.2 Star Field Concurrency
Kimi states physics writes final pose into `visual_rpn` star field with "zero copy."

**Concurrency issue**: `visual_rpn` may be **simultaneously read** by the rendering pipeline during the same frame. Need:
- Double-buffering scheme for transform data
- Or explicit barrier between physics-write and render-read phases
- Or atomic exchange of transform pointers

### 3.3 behavior_rpn Interpretation
Kimi assigns `behavior_rpn` as the physics motion interpreter. But existing K3D semantics for `behavior_rpn` are unspecified in context.

**Ambiguity**: Does `behavior_rpn` currently exist? If so, what bytecode does it already contain? Physics must either:
- Append physics bytecode to existing behavior scripts, or
- Use a **new field** `physics_rpn` to avoid clobbering existing logic

---

## 4. Missing Specifications

### 4.1 Sleep/Wake Island Detection
Physics engines disable simulation of stationary objects (sleeping). Requires:
- Velocity threshold below which bodies sleep
- Wake condition: proximity to moving body or force application
- **Island graph** construction to put connected sleeping bodies to sleep together

**Not specified** by Kimi. Without this, a pile of 1000 resting boxes will consume TGS solver bandwidth every frame.

### 4.2 Contact Manifold Caching
GJK/EPA are expensive. PhysX caches contact manifolds across frames with **contact persistency**.

**Kimi's architecture does not specify**:
- Where contact cache lives (in star? in separate buffer?)
- Cache key format (body pair ID? Morton region?)
- Cache invalidation criteria (frame count? distance change?)

### 4.3 Collision Filtering Two-Way
`collision_filter` field (uint32_t) is specified, but:
- Is this a bitmask (layer × layer matrix)?
- Or per-body filter (only matches certain layers)?
- Who evaluates the filter - broadphase or narrowphase?

**Required**: Collision matrix semantics:
```
collision_filter = (myLayer << 24) | (myGroup << 16) | (maskBits)
// Collides if: (A.maskBits & (1 << B.myLayer)) != 0
```

### 4.4 Memory Budget & Pool Allocation
Kimi's host interface:
```cpp
typedef struct {
    uint32_t physicsStars;
    k3d_ph_star_desc_t stars[];
} k3d_ph_world_desc_t;
```

**Missing**:
- Maximum star count per world
- Maximum pairs per broadphase pass (overlaps buffer size)
- Scratch memory budget per TGS iteration
- Who owns memory allocation - host or device?

---

## 5. Original Contributions

### 5.1 Hierarchical Broadphase (New Idea)
Instead of flat SAP across all bodies, use **hierarchical broadphase** matching Morton octree structure:

```
Level 0 (Root):    Test world AABB overlaps (trivial - always overlap)
Level 1-4:         Test octree node overlaps (Morton prefix match)
Level 5+:          Test leaf node body pairs (SAP within node)
```

**Benefit**: O(log N) overlap test complexity instead of O(N²) worst-case. Reuses existing Morton octree from LED-A* navigation.

### 5.2 Symbolic Physics via CAS Opcodes
K3D already has DIFFERENTIATE (0x120) and SOLVE (0x122). Physics can use these for:
- **Automatic Jacobian computation**: Constraint Jacobian = DIFFERENTIATE(constraint_function, position)
- **Symbolic constraint solving**: SOLVE(jacobian_system, lambda)

This makes constraints **user-definable** via star scripting rather than hardcoded primitives.

### 5.3 Defeasible Constraint Priority Integration
Kimi proposes TGS solver. But K3D already has `gre_defeasible_resolver.cu` for constraint priority.

**Integration**: Contact constraints should inherit from `reality_constraint` star type and participate in the existing defeasible resolution system. This unifies physics constraints with other K3D constraint types (spatial, logical, temporal).

---

## 6. Prioritized Open Issues for Codex

────────────────────────────────────────────────────────────────

### [4/4] DeepSeek — Synthesis & Final Additions

# DeepSeek Synthesis — Final Physics Engine Architecture

## 1. Critical Chain Conflicts & Resolutions

### 1.1 Opcode Dispatch Conflict (GLM Correct)
**Resolution**: We must extend `modular_rpn_kernel.cu` before implementing physics opcodes. Add:
```cpp
// In modular_rpn_kernel dispatch table:
case 0x150 ... 0x17F: {
    uint8_t physics_subop = opcode & 0x0F;
    dispatch_physics_subop(physics_subop, stack);
    break;
}
```
New file: `physics_rpn_extensions.cu` containing all 0x150-0x17F implementations.

### 1.2 Nine-Chain Swarm Allocation (GLM Correct)
**Resolution**: Physics doesn't parasitize—it **extends** swarm semantics. Each chain gets a `subchain_type` field:
- Existing chains keep original semantics
- When `subchain_type == PHYSICS_CHAIN`, chain executes physics subtasks
- Chain 7 (TGS) becomes `PHYSICS_SOLVER_CHAIN` only during physics ticks
**Implementation**: Add 32-bit `chain_mode` register to swarm context; physics scripts set it via `PHYSICS_MODE` opcode (0x15F new).

### 1.3 Ternary Collision Mapping (GLM Edge Case)
**Resolution**: Define ternary truth for physics:
```cpp
enum k3d_physics_truth_t {
    PHYSICS_TRUE = 0x01,    // Definitely colliding
    PHYSICS_FALSE = 0x00,    // Definitely separated  
    PHYSICS_UNKNOWN = 0x11,  // Within epsilon, needs EPA
    PHYSICS_DEGENERATE = 0x10  // GJK failed, use backup
};
```
New opcode: `PH_TERNARY_CLASSIFY` (0x160) maps GJK distance to ternary value.

---

## 2. Original Contributions: Beyond Chain Coverage

### 2.1 Hierarchical Sleep Islands via Morton Octree
GLM identified missing sleep system. **My solution**: Use Morton octree to detect stationary islands:

```cpp
// In k3d_ph_broadphase.ptx:
__global__ void detect_sleep_islands(
    uint64_t* morton_codes,
    float* velocities,
    uint32_t* sleep_flags) {
    
    // Group bodies by Morton prefix (shared LOD level)
    uint64_t prefix = morton_codes[threadIdx.x] >> (60 - level*3);
    
    // If all velocities in prefix group < threshold → sleep island
    bool all_sleeping = warp_all(velocity < SLEEP_THRESHOLD);
    
    // Mark entire prefix group as sleeping
    atomicOr(&sleep_flags[prefix], all_sleeping ? 1 : 0);
}
```

**Sleep persistence**: Sleeping bodies skip GJK/EPA/TGS entirely. Wake on:
- External force application (`PH_APPLY_IMPULSE`)
- Proximity to awake body (Morton neighbor test)

### 2.2 Double-Buffered Transform Synchronization
GLM flagged `visual_rpn` concurrency issue. **My solution**: Atomic pointer swap system:

```cpp
// In star physics_facets:
struct {
    float4x3 transform[2];  // Double buffer
    atomic_int front_buffer; // 0 or 1
    uint64_t transform_version[2];
};

// Physics writes to back buffer:
int back = 1 - atomicLoad(&facets->front_buffer);
facets->transform[back] = new_transform;
facets->transform_version[back]++;

// Atomic swap after physics complete:
atomicExch(&facets->front_buffer, back);
```

Renderer reads `transform[front_buffer]` with version checking.

### 2.3 Symbolic Constraint Jacobians via CAS
Extending GLM's symbolic physics idea with **concrete implementation**:

```cpp
// New opcode: PH_SYMBOLIC_CONSTRAINT (0x161)
// Stack: (bodyA bodyB constraint_func_ptr — jacobian_ptr)
// Where constraint_func_ptr points to RPN bytecode computing C(q)

// In solver:
float4 jacobianA, jacobianB;
DIFFERENTIATE(0x120);  // Existing CAS opcode!
// Differentiates constraint function wrt body positions

// Constraint force: λ = -C(q) / (J·M⁻¹·Jᵀ)
SOLVE(0x122);  // Solve linear system for λ
```

**This enables user-defined constraints** without hardcoded types.

### 2.4 Physics Memory Pool Integration
Missing memory specification (GLM). **My solution**: Integrate with existing K3D allocator:

```cpp
// New struct in k3d_ph_world_desc_t:
struct {
    uint32_t max_pairs;        // Derived from star count
    uint32_t max_contacts;     // = max_pairs * 4
    uint32_t scratch_per_chain; // 9-chain memory budget
    VkBufferUsageFlags usage; // Reuse existing CUDA/Vulkan allocator
} memory_budget;

// Allocation occurs inside existing:
// gre_memory_router.cu (already manages device memory)
```

**Key innovation**: Physics uses **same memory domains** as rendering/navigation:
- Domain 0: Persistent star data
- Domain 1: Frame-transient physics buffers  
- Domain 2: Scratch for iterative solvers

### 2.5 Continuous Collision Detection (CCD) via Bezier Swept Volumes
GLM flagged tunneling. **My solution**: Use existing `drawing_primitives.cu` Bezier curves:

```cpp
// For fast-moving bodies, create swept volume:
bezier_swept_aabb(start_pos, end_pos, radius);

// Broadphase test against swept volume instead of static AABB
// Narrowphase: Solve cubic for collision time via SOLVE(0x122)
```

**Efficiency**: Only applied to bodies with `velocity > CCD_THRESHOLD`.

---

## 3. Final Pipeline Specification

### 3.1 Extended TRM Loop with Physics
```
Perceive:
  - Frustum cull (existing)
  - Hierarchical broadphase (new: uses Morton octree)
  
Navigate:  // Unchanged LED-A*
  
Reason:
  - Chain 0-2: GJK/EPA with degeneracy handling
  - Chain 3:   Contact manifold caching & persistence
  - Chain 4:   Constraint graph crystallization (existing gre_graph_crystallizer)
  - Chain 5:   Symbolic Jacobian computation via DIFFERENTIATE
  - Chain 6:   Island sleep/wake detection
  - Chain 7:   TGS solver with divergence detection
  - Chain 8:   Integration with quaternion renormalization
  
Decide:
  - Halting gate checks physics truth values
  - AI scripts can query contact states
  
Act:
  - Atomic transform buffer swap
  - Wake sleeping islands if needed
  
Learn:
  - Contact cache optimization
  - Constraint graph simplification
```

### 3.2 New Kernel Files Beyond Kimi's List
| File | Purpose | Why It's New |
|---|---|---|
| `k3d_ph_sleep_islands.ptx` | Hierarchical sleep detection | GLM identified gap |
| `k3d_ph_transform_swap.ptx` | Atomic double-buffer management | My concurrency solution |
| `k3d_ph_symbolic_jacobian.ptx` | CAS-based constraint derivation | My symbolic physics extension |
| `k3d_ph_ccd_swept.ptx` | Bezier-based continuous collision | My tunneling fix |
| `k3d_ph_memory_router.ptx` | Physics memory domain allocation | My pool integration |

### 3.3 New Opcodes Beyond 0x150-0x17F
| Opcode | Purpose |
|---|---|
| `PHYSICS_MODE` (0x15F) | Set swarm chain physics submodes |
| `PH_TERNARY_CLASSIFY` (0x160) | Map collision to ternary logic |
| `PH_SYMBOLIC_CONSTRAINT` (0x161) | User-defined constraint setup |
| `PH_SLEEP_ISLAND` (0x162) | Query/manage sleep states |
| `PH_CCD_SWEPT_TEST`

────────────────────────────────────────────────────────────────

**Handoff to Claude:** Read this file, synthesize the chain, and write a formal spec + implementation instructions for Codex.
## [2/4] Qwen — Code Specialist

### 1. CUDA Kernel Signatures + Core Logic

```cpp
// (a) Morton Broadphase Pair-Finder
__global__ void k3d_broad_phase_morton(
    const uint32_t* __restrict__ mortonCodes,
    uint2* __restrict__ pairs,
    uint32_t* __restrict__ pairCount,
    int numStars
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStars - 1) return;
    uint32_t code1 = mortonCodes[idx], code2 = mortonCodes[idx + 1];
    int common = __clz(code1 ^ code2);
    if (common > THRESHOLD_LOD) {
        uint32_t count = atomicAdd(pairCount, 1);
        pairs[count] = {idx, idx + 1};
    }
}

// (b) GJK Support-Function Loop
__global__ void k3d_narrow_phase_gjk(
    const float3* __restrict__ com, const float4* __restrict__ orient,
    const uint2* __restrict__ pairs, float* __restrict__ distances, int numPairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) return;
    uint2 pair = pairs[idx];
    float dist = gjk_iterative_solve(com[pair.x], orient[pair.x], com[pair.y], orient[pair.y]);
    distances[idx] = dist;
}

// (c) PGS Velocity Solver Iteration
__global__ void k3d_constraint_resolver(
    float3* linVel, float3* angVel,
    const float* invMass, const uint2* contacts, int numContacts, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numContacts) return;
    for (int i = 0; i < iterations; i++) {
        float impulse = cas_solve_opcode_0x122(contacts[idx], linVel, angVel, invMass);
        apply_impulse(linVel, angVel, contacts[idx], impulse);
    }
}

// (d) Symplectic Euler Integrator
__global__ void k3d_integrator_chain(
    float3* com, float3* linVel, float* invMass, float dt, int numStars
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStars) return;
    if (invMass[idx] > 0.0f) { com[idx] += linVel[idx] * dt; }
}
```

### 2. SOA Memory Layout (4096 rigid bodies, ~1.5 MB)

```cpp
#define MAX_STARS 4096
typedef struct {
    float com_x[MAX_STARS], com_y[MAX_STARS], com_z[MAX_STARS];
    float orient_x[MAX_STARS], orient_y[MAX_STARS], orient_z[MAX_STARS], orient_w[MAX_STARS];
    float linVel_x[MAX_STARS], linVel_y[MAX_STARS], linVel_z[MAX_STARS];
    float angVel_x[MAX_STARS], angVel_y[MAX_STARS], angVel_z[MAX_STARS];
    float invInertia_x[MAX_STARS], invInertia_y[MAX_STARS], invInertia_z[MAX_STARS];
    float invMass[MAX_STARS];
    uint8_t  integrator[MAX_STARS];     // 0=Euler, 1=RK4, 2=implicit
    uint8_t  flags[MAX_STARS];          // 0x01=Static, 0x02=Sleeping
    uint32_t mortonCode[MAX_STARS];
    uint32_t behaviorRpnPtr[MAX_STARS]; // pointer into constant mem bytecode buffer
} PhysicsGalaxySOA;  // 128-byte aligned per array block
```

### 3. Gravity Force Star behavior_rpn Bytecode

```text
0x151 0x00 0x00 0x00  // OP_GRAVITY_FIELD → pushes {0, -9.81, 0}
0x55  0x00 0x00 0x00  // OP_FETCH_MASS    → pushes star's mass
0x20  0x00 0x00 0x00  // MUL_VEC_SCALAR   → F = gravity * mass
0x30  0x00 0x00 0x00  // APPLY_FORCE      → pops vector, adds to accumulator
// Hex stream: 51 01 00 00 55 00 00 00 20 00 00 00 30 00 00 00
```

### 4. SOLVE(0x122) Stack Convention

Stack BEFORE: [SP+0]=ContactNormal(float3), [SP+1]=EffectiveMass, [SP+2]=Bias(Baumgarte), [SP+3]=RelativeVelocity
Operation: Impulse = -(RelVel + Bias) * EffMass, clamped to [0, MAX_IMPULSE]
Stack AFTER: [SP+0]=ComputedImpulse (caller applies to velocity SOA)

### 5. Python ctypes Bridge Skeleton

```python
class PhysicsGalaxyBridge:
    def __init__(self, max_stars=4096):
        self.lib = ctypes.CDLL("./build/libk3d_physics.so")
        self.max_stars = max_stars
        self.soa_ptr = None

    def load_kernels(self, ptx_path):
        self.lib.k3d_load_ptx(ctypes.c_char_p(ptx_path.encode()))

    def allocate_galaxy(self):
        err, ptr = cuda.cuMemAlloc(self.max_stars * 28 * 4)
        self.soa_ptr = ptr
        self.lib.k3d_init_soa(self.soa_ptr, ctypes.c_int(self.max_stars))

    def upload_behavior(self, star_index, bytecode_hex):
        bytecode = bytes.fromhex(bytecode_hex)
        self.lib.k3d_upload_behavior(ctypes.c_int(star_index),
                                     ctypes.c_char_p(bytecode), ctypes.c_int(len(bytecode)))

    def step(self, dt):
        self.lib.k3d_step_fused(self.soa_ptr, ctypes.c_float(dt), ctypes.c_int(self.max_stars))
```

### 6. First 5 Tests

1. `test_morton_sort_coherence` — 4096 points, assert sorted codes → valid pairs
2. `test_rpn_gravity_bytecode` — step once, assert linVel_y decreases by 9.81*dt ±ε
3. `test_solve_0x122_stack` — known Jacobian/mass/bias → assert impulse matches PGS analytical
4. `test_soa_alignment_padding` — sizeof each array % 128 == 0
5. `test_bridge_lifecycle` — allocate/step/free, assert zero leaks via cuda-memcheck

---

## [3/4] GLM — Analysis & Edge Cases

### 1. Correctness Gaps
- **Integrator instability**: symplectic Euler drifts energy in long simulations — Verlet recommended for orbital/stack scenarios
- **PGS non-convergence**: not guaranteed to converge with high-stiffness constraints → jitter
- **GJK degeneracy**: flat polygons/point-like objects fail without degenerate simplex handling
- **MISSING: Friction model** — SOLVE(0x122) stack has no tangential impulse → objects slide infinitely
- **MISSING: Restitution** — no coefficient of bounciness → all collisions perfectly inelastic
- **MISSING: CCD** — fast objects tunnel through thin geometry between frames
- **Fixed 4096 star limit** — game worlds are dynamic, must be pool-based

### 2. Top 5 House Game World Edge Cases
1. **Jittering stacks** — tower of crates explodes/vibrates (symplectic Euler + non-convergent PGS)
2. **Projectile tunnelling** — fast objects pass through walls (no CCD)
3. **Character stuck in geometry** — discrete solver can't generate escape impulse
4. **Ghost objects on slopes** — no friction → everything slides regardless of material
5. **Crowded room performance collapse** — 4096 objects in small area floods narrowphase

### 3. Sovereignty Compliance Issues
- **PhysicsGalaxyBridge ties to CPython** — violates platform sovereignty for console/embedded targets
- **Physics→render coupling** — `...→INTEGRATE→DRAW` hook is wrong; physics should write state, renderer queries it separately
- **cas_solve_0x122 is a black box** — if it depends on third-party CAS lib, sovereignty broken

### 4. Galaxy Star Coupling — Risks
- behavior_rpn modifying physics_galaxy global state mid-sim → determinism break
- Python bridge can inject bytecode at runtime → allows external code to rewrite physical laws (sovereignty breach)

### 5. Missing Star Fields
- `coefficient_of_friction`, `coefficient_of_restitution` (per-star material)
- `collision_shape_handle` — SOA has state but no geometry reference
- `collision_group` / `filter_mask` — needed for layer-based culling
- `is_sleeping` flag (distinct from flags byte — make explicit)

### 6. Missing Opcodes
- `RAYCAST` / `SHAPE_OVERLAP` — essential for AI line-of-sight and gameplay queries
- `SET_FRICTION` / `SET_RESTITUTION` — per-star material params
- `CCD_ENABLE` — per-star continuous collision toggle

### 7. Prioritized Open Issues for Codex Spec
**P0 (system non-viable):**
1. Complete PhysicsGalaxySOA with ALL fields (friction, restitution, shape handle, group)
2. Sleep/wake velocity threshold system — without it 4096 active objects won't perform
3. Decouple physics from draw — physics ends at INTEGRATE, renderer queries state

**P1 (game-logic broken):**
4. Friction in PGS solver (tangential impulse alongside normal impulse)
5. Swept-sphere CCD for high-velocity stars
6. force_grammar bytecode spec must cover at minimum: gravity, drag, spring forces

**P2 (robustness):**
7. Dynamic SOA pool allocation — replace fixed 4096
8. RAYCAST/SHAPE_OVERLAP opcodes
9. Document cas_solve_0x122 convergence criteria and failure modes

---

## [4/4] DeepSeek — Synthesis & Final Additions

### 1. GLM P0+P1 Gap Resolutions

**Friction/Restitution**: Add `friction` (uint8, 0-255 → [0.0,2.0]) and `restitution` (uint8, 0-255 → [0.0,1.0]) to SOA. PGS extends to Coulomb friction: `friction_impulse = clamp(J*μ, -μ|J|, μ|J|)` applied as tangential impulse.

**CCD**: No new phase. In BROAD phase, expand AABB to `sweptAABB = AABB ∪ (AABB + vel*dt)`. Run same Morton __clz test on sweptAABBs. Flagged pairs go to narrow-phase CCD refinement (swept-sphere: solve |p1+v1*t - (p2+v2*t)| = r1+r2).

**Fixed 4096**: Retain 4096 as max (deterministic timing), but add `free_list[4096]` ring buffer + atomic free_head. Stars spawn/despawn via pool. Bridge accesses only via byte offsets + validated RPC.

**RAYCAST/SHAPE_OVERLAP**: New opcodes 0x160/0x161. Shape table in constant memory (64 entries: sphere/box/capsule params). shape_handle (uint16) + shape_type (uint8) added to SOA.

### 2. ORIGINAL: Physics-to-Knowledge Feedback

Every SOLVER phase collision writes a CollisionEvent to a ring buffer:
```cpp
struct CollisionEvent {
    uint32_t star_a, star_b;
    float3   impulse;      // for energy signature
    uint32_t frame;
    uint8    shape_a, shape_b;
    uint16   pad;          // 16B aligned
};
```
Bridge polls every 60 frames. Computes `energy = 0.5 * dot(impulse,impulse) * (invM_a + invM_b)`. Updates House knowledge graph:
- Nodes = star UUIDs, Edges = collision events weighted by energy
- Infers: high-energy + low restitution = "dense/brittle"; high-freq + low-energy + sphere = "gas cloud"
- Feeds back to spawn behavior: House advises shape/material for new stars in high-collision regions
→ **Physics becomes teacher: collision dynamics sculpt the Galaxy taxonomy**

New star field: `event_tag` (uint32) = last frame this star collided (debouncing). Opcodes 0x15A (CLEAR_EVENTS) and 0x15B (READ_EVENT_BUFFER) added.

### 3. Final Complete Opcode Table 0x150-0x17F

| Opcode | Mnemonic | Phase | Description |
|---|---|---|---|
| 0x150 | BROAD | BROAD | Morton broadphase on sweptAABBs (CCD built-in) |
| 0x151 | FORCE | FORCE | Evaluate all force star behavior_rpn, accumulate wrench |
| 0x152 | NARROW | NARROW | GJK/EPA contact manifold (shape-aware via shape_handle) |
| 0x153 | SOLVE | SOLVE | PGS with restitution + Coulomb friction |
| 0x154 | INTEGRATE | INTEGRATE | Symplectic Euler + tangential friction impulse |
| 0x155 | DRAW | DRAW | Write render buffers (pos, color, size) |
| 0x156 | SPAWN_STAR | SPAWN | Allocate from free_list, init all SOA fields |
| 0x157 | DESPAWN_STAR | DESPAWN | Push index to free_list |
| 0x158 | SET_SHAPE | STATE | Update shape_handle from shape table |
| 0x159 | SET_MATERIAL | STATE | Update friction + restitution |
| 0x15A | CLEAR_EVENTS | MAINT | Garbage-collect old event_tags (every 1024 frames) |
| 0x15B | READ_EVENT_BUFFER | BRIDGE | Copy collision events to CPU-accessible buffer (double-buffered) |
| 0x15C-0x15F | RESERVED | — | — |
| 0x160 | RAYCAST | QUERY | Ray vs shape: output hit_index, t, normal |
| 0x161 | SHAPE_OVERLAP | QUERY | Shape vs shape: output list of overlapping indices |
| 0x162-0x17F | RESERVED | — | Future: soft bodies, fluid, magnetic fields |

### 4. Final PhysicsGalaxySOA (128B per star, 4096 max)

| Field | Type | Offset | Description |
|---|---|---|---|
| pos | float3 | 0x00 | World position |
| vel | float3 | 0x0C | Linear velocity |
| force | float3 | 0x18 | Accumulated force |
| invMass | float | 0x24 | 1/mass |
| invI | float3 | 0x28 | Inverse inertia (diagonal) |
| shape_handle | uint16 | 0x34 | Index into constant shape table |
| shape_type | uint8 | 0x36 | 0=sphere,1=box,2=capsule |
| padding1 | uint8 | 0x37 | align |
| friction | uint8 | 0x38 | [0-255]→[0.0-2.0] |
| restitution | uint8 | 0x39 | [0-255]→[0.0-1.0] |
| padding2 | uint16 | 0x3A | align |
| event_tag | uint32 | 0x3C | Last collision frame (debounce) |
| color | uint32 | 0x40 | ARGB render color |
| size | float | 0x44 | Render scale |
| ttl | int32 | 0x48 | Time-to-live frames (-1 = infinite) |
| health | float | 0x4C | Damage accumulator (≤0 → despawn) |
| reserved | 48B | 0x50 | Future (EM, temperature, spin) |
| **TOTAL** | | | **128B** |

Shape table in constant memory: 64 entries × (uint8 type + float3 params). Sphere: params.x=radius. Box: params=halfextents. Capsule: params.xy=radius,halfheight.

### 5. Architecture Statement (top of Codex spec)

The Sovereign K3D physics engine treats every Galaxy star as a first-class physical entity governed by a deterministic RPN-driven impulse solver over a 128B-aligned Structure-of-Arrays layout, where broadphase leverages Morton-ordered swept AABBs for built-in CCD, narrowphase computes shape-specific manifolds via GJK/EPA against a constant-memory shape table, and the solver integrates per-star friction and restitution derived from collision-induced knowledge updates; collision events are asynchronously fed to the House knowledge system which infers latent physical traits and feeds back to govern adaptive spawning — closing the loop between simulation and semantics — while the Python bridge remains a sovereign client accessing state only through validated byte-offset RPCs, and the full opcode space 0x150-0x17F encompasses broadphase, force eval, narrowphase, solve, integrate, draw, spawn, despawn, material update, event sync, and query operations, all under the architectural trinity of Kimi's kernel design, Qwen's SOA precision, and GLM's correctness validation.

---

## HANDOFF TO CLAUDE — Write Codex Spec from this chain.
