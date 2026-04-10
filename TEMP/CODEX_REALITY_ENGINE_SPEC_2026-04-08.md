# Codex Directive: K3D Reality Engine Extension
**Date:** 2026-04-08  
**Author:** Claude (Architecture)  
**Research basis:** `docs/research/kkrieger_procedural_texture_spec.md`, `docs/research/rpn_reality_engine_ai_spec.md`, `docs/research/2d_engine_techniques_spec.md`, `docs/research/kkrieger_source_analysis.md`  
**Grounding:** `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`, `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

---

## Scope

Three interlocking extensions that together form the "K3D Reality Engine":

1. **Always-On Entity AI** — persistent Galaxy stars with behavior RPN programs (BH_* opcodes)
2. **Procedural Texture Synthesis** — kkrieger-inspired GPU-native texture generation (TEX_* opcodes)
3. **2D Physics Sub-range** — AABB sweep / verlet cloth sharing the 3D physics SOA buffers

Inspired by studying `fr_public/ktg/gentexture.cpp` (kkrieger's actual texture generator) and the `werkkzeug3_kkrieger` operator graph. Key insight from kkrieger: **store algorithms, not assets** — a texture is a small DAG of operators parameterized as numbers. K3D already does this for drawing primitives via RPN; textures and behaviors are the same pattern.

---

## 1. Opcode Registry Extension

### 1.1 Conflict Analysis

Reading `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` reveals double-assignments:
- `OP_TRM_SWIGLU_1024 = 0x64` AND `OP_DRAW_MOVE = 0x64`
- `OP_REL_LINE = 0x70` AND `OP_DRAW_PUSH_STATE = 0x70`
- `OP_FIELD_COEF = 0x71` AND `OP_DRAW_POP_STATE = 0x71`

Additionally, 0xC0–0xD7 is fully occupied (clustering, vector, quantum ops). The research specs proposed TEX at 0xC0–0xDF — **this conflicts**. Texture opcodes move to 0x1C0–0x1DF.

**Codex must fix the 0x64/0x70/0x71 double-assignments** before adding new opcodes. Check which constant is actually dispatched in `modular_rpn_kernel.cu`, rename the unused Python alias.

### 1.2 New Opcode Ranges (16-bit, safe from existing 0x00–0xFF and 0x150–0x17F)

```
Range         Count  Domain
0x130–0x14F   32     2D Physics  (PHYS2D_*)
0x150–0x17F   48     3D Physics  (PH_*)           ← EXISTING, DO NOT TOUCH
0x180–0x1BF   64     Entity AI   (BH_*)
0x1C0–0x1DF   32     Texture synthesis (TEX_*)
0x1E0–0x1FF   32     Signed Distance Fields (SDF_*)   ← future
0x200–0x21F   32     Drawing 2D primitives (DR2_*)     ← future
0x280–0x2BF   64     WFC / BSP / L-systems (GEN_*)    ← future
```

### 1.3 Texture Synthesis Opcodes (0x1C0–0x1CF — P0)

All texture ops produce `texture_handle: u32` pointing to a GPU-side intermediate buffer. `TEX_BAKE` finalizes to a CUDA texture array slot.

| Opcode | Mnemonic | Stack In | Stack Out | Notes |
|--------|----------|----------|-----------|-------|
| 0x1C0 | TEX_PERLIN_NOISE | octaves:f32, freq:f32, amp:f32, persist:f32 | handle:u32 | quintic smoothstep (kkrieger `SmoothStep`) |
| 0x1C1 | TEX_VORONOI | cell_count:f32, jitter:f32 | handle:u32 | Worley F1 distance |
| 0x1C2 | TEX_VALUE_NOISE | freq:f32, octaves:f32 | handle:u32 | bilinear lerp on permutation table |
| 0x1C3 | TEX_GRID_NOISE | scale:f32, falloff:f32 | handle:u32 | radial (1-r²)⁴ (kkrieger `GNoise2`) |
| 0x1C4 | TEX_FFT_BLUR | handle:u32, sigma:f32 | handle:u32 | cuFFT R2C forward, Gaussian multiply, C2R |
| 0x1C5 | TEX_WARP | base:u32, warp:u32, intensity:f32 | handle:u32 | bilinear sample with displacement |
| 0x1C6 | TEX_BLEND | tex_a:u32, tex_b:u32, alpha:f32, mode:u8 | handle:u32 | modes: OVER, MUL, ADD, SCREEN |
| 0x1C7 | TEX_NORMAL_MAP | height:u32, strength:f32 | handle:u32 | Sobel filter on height → RGB normals |
| 0x1C8 | TEX_COLOR_RAMP | handle:u32, ramp_addr:u64 | handle:u32 | gradient table lookup |
| 0x1C9 | TEX_TURBULENCE | handle:u32, octaves:f32 | handle:u32 | abs(noise) fractal sum |
| 0x1CA | TEX_MARBLE | vein_scale:f32, turbulence:f32 | handle:u32 | sin(x + turbulence_noise) |
| 0x1CB | TEX_TRANSFORM | handle:u32, sx:f32, sy:f32, rot:f32 | handle:u32 | UV transform before sampling |
| 0x1CF | TEX_BAKE | handle:u32, width:u32, height:u32 | cuda_tex_id:u32 | writes to CUDA texture array, frees handle |

**Physics link:** `PH_MATERIAL_FETCH` (0x157) can take a `cuda_tex_id` and derive friction from RMS of noise gradient. Add `PhysicsMaterialEntry.texture_id: u32` and a GPU lookup that samples the baked texture at contact UV for material variation.

### 1.4 Entity AI Behavior Opcodes (0x180–0x1BF — P0)

| Opcode | Mnemonic | Stack In | Stack Out | Notes |
|--------|----------|----------|-----------|-------|
| 0x180 | BH_PERCEIVE | radius:f32 | list_addr:u64 | Morton Octree spatial query — same kernel as TRM Frustum |
| 0x181 | BH_SEEK | target_id:u32 | fx:f32, fy:f32, fz:f32 | normalize(target-pos)×max_speed - vel |
| 0x182 | BH_FLEE | threat_id:u32 | fx:f32, fy:f32, fz:f32 | inverse seek |
| 0x183 | BH_ARRIVE | target_id:u32, slow_r:f32 | fx:f32, fy:f32, fz:f32 | decelerates inside slow_radius |
| 0x184 | BH_SEPARATE | list_addr:u64 | fx:f32, fy:f32, fz:f32 | sum(normalize(pos - neighbor) / dist) |
| 0x185 | BH_APPLY_FORCE | fx:f32, fy:f32, fz:f32 | — | atomic-add to PhysicsBodySOA.forces[body_id] |
| 0x186 | BH_BT_TICK | bt_addr:u64 | status:u32 | run behavior tree RPN; 0=FAIL 1=OK 2=RUNNING |
| 0x187 | BH_UTILITY_EVAL | N:u32, (weight:f32, state:f32)×N | best_idx:u32 | argmax of weighted scores |
| 0x188 | BH_GOAP_PLAN | goal_star_id:u64 | plan_addr:u64, valid:u32 | A* on action graph via LED-A* |
| 0x189 | BH_SLEEP_CHECK | dist:f32 | new_state:u8 | <50m→0(60Hz), <200m→1(10Hz), else→2(1Hz) |
| 0x18A | BH_BLACKBOARD_READ | key:u32 | val:f32 | atomic read from faction blackboard star |
| 0x18B | BH_BLACKBOARD_WRITE | key:u32, val:f32 | — | atomic write to faction blackboard star |
| 0x18C | BH_PATHFIND | tx:f32, ty:f32, tz:f32 | waypoint_addr:u64 | delegates to LED-A* pool |
| 0x18D | BH_EMIT_EVENT | type:u32, target_id:u32 | — | writes to CollisionEventQueue |
| 0x18E | BH_FILTER_FACTION | list_addr:u64, mask:u32 | filtered_addr:u64 | filter perceived entities by faction bitmask |

### 1.5 2D Physics Opcodes (0x130–0x14F — P1)

| Opcode | Mnemonic | Stack In | Stack Out |
|--------|----------|----------|-----------|
| 0x130 | PHYS2D_AABB_CREATE | x:f32, y:f32, w:f32, h:f32 | body_id:u32 |
| 0x131 | PHYS2D_AABB_SWEEP | body_id:u32, dx:f32, dy:f32 | t:f32, nx:f32, ny:f32 |
| 0x132 | PHYS2D_SLOPE_CORRECT | body_id:u32, max_angle:f32 | — |
| 0x133 | PHYS2D_COYOTE | body_id:u32, grace_s:f32 | grounded:u32 |
| 0x134 | PHYS2D_JUMP_BUFFER | body_id:u32, window_s:f32 | buffered:u32 |
| 0x135 | PHYS2D_ONE_WAY | body_id:u32, dir_y:f32 | — |
| 0x136 | PHYS2D_PUSH_SLIDE | body_id:u32, other:u32 | — |
| 0x137 | PHYS2D_VERLET_CREATE | point_count:u32 | cloth_id:u32 |
| 0x138 | PHYS2D_VERLET_INTEGRATE | cloth_id:u32, dt:f32 | — |
| 0x139 | PHYS2D_VERLET_CONSTRAINT | cloth_id:u32, ia:u32, ib:u32, rest_len:f32 | — |

---

## 2. P0: Entity Star Schema

### 2.1 EntityStar Struct (64 bytes, Grammar Galaxy L3)

```c
// knowledge3d/cranium/kernels/entity_star.h
struct EntityStar {
    // [L2 Meaning anchor]
    uint32_t  entity_id;          // 0x00 unique handle
    uint32_t  physics_body_id;    // 0x04 slot in PhysicsBodySOA

    // [L3 Behavior program]
    uint64_t  behavior_rpn;       // 0x08 addr in RPN program region of Galaxy

    // [Identity]
    uint8_t   faction;            // 0x10 0-255 faction ID
    uint8_t   sleep_state;        // 0x11 0=awake(60Hz) 1=doze(10Hz) 2=asleep(1Hz)
    uint8_t   ai_tier;            // 0x12 0=reactive 1=planning 2=strategic
    uint8_t   perception_flags;   // 0x13 SEE=0x1 HEAR=0x2 SCENT=0x4

    // [Spatial]
    float     perception_radius;  // 0x14
    float     last_player_dist;   // 0x18
    float     awareness;          // 0x1C 0.0=unaware 1.0=fully aware

    // [Memory pointers]
    uint32_t  goal_stack_ptr;     // 0x20 pointer into Galaxy goal storage
    uint32_t  blackboard_star_id; // 0x24 faction shared blackboard star

    // [Emotion]
    float     emotion[4];         // 0x28 anger, fear, happiness, curiosity

    // [L4 meta-rule linkage]
    uint32_t  meta_rule_addr;     // 0x38 sleep-consolidation behavior program
    uint32_t  _pad;               // 0x3C align to 64 bytes
};                                // total: 0x40 = 64 bytes
```

**GalaxyEntry compatibility:** EntityStar fits within the `user_data[64]` field of a GalaxyEntry. Store with `type = GALAXY_TYPE_ENTITY` (define as 0x0E, next after existing types).

### 2.2 BH_PERCEIVE → Morton Octree Contract

`BH_PERCEIVE` pops `radius:f32`, then calls:
```c
morton_spatial_query(
    g_morton_octree_ptr,        // existing constant global
    physics_soa->pos[body_id],  // entity world position
    radius,
    output_list_addr             // temp buffer in Galaxy scratch region
);
```
This is the **same kernel entrypoint** as TRM Frustum culling — no new kernel needed. Output layout: `[count:u32, entity_id_0:u32, entity_id_1:u32, ...]`.

---

## 3. P0: Procedural Texture Pipeline

### 3.1 New CUDA Kernel Files

| File | Kernels | Priority |
|------|---------|----------|
| `knowledge3d/cranium/kernels/tex_noise_kernels.cu` | `tex_perlin_noise_kernel`, `tex_voronoi_kernel`, `tex_value_noise_kernel`, `tex_grid_noise_kernel` | P0 |
| `knowledge3d/cranium/kernels/tex_filter_kernels.cu` | `tex_fft_blur_kernel`, `tex_warp_kernel`, `tex_normal_map_kernel` | P0 |
| `knowledge3d/cranium/kernels/tex_bake_kernel.cu` | `tex_bake_kernel`, `tex_blend_kernel`, `tex_color_ramp_kernel` | P0 |

### 3.2 tex_perlin_noise_kernel Specification

Based on kkrieger's actual `ktg/gentexture.cpp`:
```c
// - 4096-entry permutation table (LFSR seed 0x93638245u)
// - PGradient2(hash,x,y): 8 gradients via (hash&1 ? -u : u) + (hash&2 ? -2v : 2v)
// - Quintic smoothstep: f(x) = x³(10 + x(6x - 15))
// - Tiling via power-of-two bitmask: (X & mask) where mask = period-1
__global__ void tex_perlin_noise_kernel(
    float* output,          // [width × height] float
    int width, int height,
    int octaves,
    float frequency,
    float amplitude,
    float persistence,
    uint32_t tile_mask_x,   // period_x - 1 (must be power of two)
    uint32_t tile_mask_y
);
```

### 3.3 Texture Handle Pool

Add to `modular_rpn_kernel.cu`:
```c
// 256 intermediate slots; TEX_BAKE finalizes to cudaArray (CUDA texture object)
struct TextureHandlePool {
    float*    slot_ptr[256];       // device pointers to intermediate float buffers
    uint32_t  width[256];
    uint32_t  height[256];
    uint8_t   in_use[256];
    cudaArray_t cuda_arrays[64];  // finalized texture slots
};
extern __constant__ TextureHandlePool* g_texture_pool_ptr;
```

### 3.4 Physics Material Texture Link

Extend `PhysicsMaterialEntry` (case 0x157 in `modular_rpn_kernel.cu`):
```c
struct PhysicsMaterialEntry {
    float    friction;
    float    restitution;
    float    density;
    uint32_t texture_id;   // cuda_tex_id from TEX_BAKE; 0xFFFFFFFF = none
};
```
When `texture_id != 0xFFFFFFFF`, `PH_MATERIAL_FETCH` samples the baked texture at contact UV and modulates friction by local gradient RMS (rougher texture → higher friction).

---

## 4. P1: 2D Physics Sub-range

### 4.1 Sharing SOA Buffers with 3D Physics

2D bodies live in `PhysicsBodySOA` as regular entries with `pos.z = 0`, `vel.z = 0`, and `phys_flags |= PHYS_FLAG_2D_CONSTRAINED`. The 3D pipeline skips z-motion for constrained bodies. No separate SOA needed.

### 4.2 PHYS2D_AABB_SWEEP Kernel Spec

```c
// Swept AABB — Celeste-style. Returns earliest collision t ∈ [0,1] + contact normal.
__device__ void phys2d_aabb_sweep(
    float4 aabb,          // {x, y, w, h}
    float2 velocity,      // {dx, dy}
    PhysicsBodySOA* soa,
    uint32_t body_count,
    float* out_t,
    float2* out_normal
);
```
Slope correction: if contact normal angle < `max_slope_angle`, slide along slope. Coyote time: entity-local `frames_since_grounded` counter in EntityStar.goal_stack_ptr scratch region.

---

## 5. Implementation Files — Complete List

### 5.1 New Files

```
knowledge3d/cranium/kernels/entity_star.h               ← EntityStar struct
knowledge3d/cranium/kernels/tex_noise_kernels.cu         ← TEX_PERLIN, VORONOI, VALUE, GRID
knowledge3d/cranium/kernels/tex_filter_kernels.cu        ← TEX_FFT_BLUR, WARP, NORMAL_MAP
knowledge3d/cranium/kernels/tex_bake_kernel.cu           ← TEX_BAKE, BLEND, COLOR_RAMP
knowledge3d/cranium/kernels/entity_behavior.cu           ← BH_* kernel implementations
knowledge3d/cranium/kernels/phys2d_aabb.cu               ← PHYS2D_* implementations
knowledge3d/cranium/ptx_runtime/sovereign_entity.py      ← EntityStar Python side (ingestion only)
tests/test_sovereign_entity_surface.py
tests/test_procedural_texture_surface.py
tests/test_phys2d_aabb_surface.py
```

### 5.2 Files to Modify

```
knowledge3d/cranium/kernels/modular_rpn_kernel.cu
  ← FIX: resolve 0x64/0x70/0x71 double-assignments
  ← add cases 0x130–0x14F (PHYS2D_*)
  ← add cases 0x180–0x1BF (BH_*)
  ← add cases 0x1C0–0x1CF (TEX_*)
  ← add __constant__ TextureHandlePool* g_texture_pool_ptr
  ← extend PhysicsMaterialEntry with texture_id field

knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
  ← FIX: deduplicate 0x64, 0x70, 0x71
  ← add PH2D_*, BH_*, TEX_* Python constants

knowledge3d/cranium/bridges/sovereign_bridges.py
  ← add bind_texture_pool(pool_ptr, slot_count)
  ← add bind_entity_soa(entity_arr_ptr, count)

knowledge3d/cranium/ptx/trm_step_fused.cu
  ← add BEHAVIOR_PHASE slot after PHYSICS_PHASE stub
  ← extend signature: + entity_star_ptr, entity_count

knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py
  ← register new opcode tokens: ph2d_*, bh_*, tex_*
```

### 5.3 Compile Directive (after source edits)

```bash
nvcc --ptx -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

---

## 6. Sovereignty Audit

| System | Hot Path Component | Verdict |
|--------|--------------------|---------|
| Texture synthesis | tex_*_kernel (PTX), TEX_* opcodes in RPN | ✓ Sovereign |
| Texture parameterization | Galaxy star `visual_rpn` (L1) contains TEX_* program | ✓ Sovereign |
| Entity behavior tick | BH_* opcodes dispatched via modular_rpn_kernel | ✓ Sovereign |
| BH_PERCEIVE | morton_spatial_query (existing PTX kernel) | ✓ Sovereign |
| BH_PATHFIND | LED-A* PTX kernel pool | ✓ Sovereign |
| Entity sleep state | GPU-side EntityStar.sleep_state, updated by BH_SLEEP_CHECK | ✓ Sovereign |
| Faction blackboard | Galaxy star atomic float ops (PTX atomicAdd) | ✓ Sovereign |
| PHYS2D sweep | phys2d_aabb_sweep device function | ✓ Sovereign |
| Entity bootstrap | `sovereign_entity.py` ingestion (one-time, not hot path) | ✓ Acceptable |
| Texture ingestion | `ingest_texture_star()` Python (one-time) | ✓ Acceptable |

**Critical:** Entity `behavior_rpn` addr must be passed to `execute_rpn_program()` on-device only. Never execute behavior RPN from Python.

---

## 7. Smoke Tests

### test_sovereign_entity_surface.py

```python
def test_entity_perceive_returns_list():
    # Allocate EntityStar array (3 entries)
    # Place entity 0 at (0,0,0), entity 1 at (5,0,0), entity 2 at (100,0,0)
    # Execute RPN: [10.0] BH_PERCEIVE
    # Assert: returned list contains entity 1, NOT entity 2

def test_bh_seek_produces_force():
    # Two entities, entity 0 seeks entity 1 at (5,0,0)
    # Execute RPN: [entity_1_id] BH_SEEK
    # Assert: force_x > 0.0, |force| ≈ max_speed
```

### test_procedural_texture_surface.py

```python
def test_tex_perlin_bake():
    # Execute RPN: [4] [1.0] [1.0] [0.5] TEX_PERLIN_NOISE [256] [256] TEX_BAKE
    # Assert: cuda_tex_id != 0xFFFFFFFF
    # Assert: sampled value at (0.5, 0.5) ∈ [0.0, 1.0]

def test_tex_voronoi_bake():
    # Execute RPN: [8.0] [0.5] TEX_VORONOI [256] [256] TEX_BAKE
    # Assert: cuda_tex_id valid
```

### test_phys2d_aabb_surface.py

```python
def test_phys2d_aabb_ground_collision():
    # Box at y=0.5 (height=1.0), moving dy=-2.0, floor at y=0
    # PHYS2D_AABB_CREATE 0.0 0.5 1.0 1.0 → body_id
    # PHYS2D_AABB_SWEEP body_id 0.0 -2.0
    # Assert: t ≈ 0.25, ny ≈ 1.0
```

---

## 8. Implementation Order

```
Step 1 — Fix opcode conflicts (30 min):
  rpn_opcodes.py: deduplicate 0x64, 0x70, 0x71
  modular_rpn_kernel.cu: verify which case is live, remove dead alias

Step 2 — Texture pipeline (P0):
  Create tex_noise_kernels.cu + tex_filter_kernels.cu + tex_bake_kernel.cu
  Add cases 0x1C0–0x1CF to modular_rpn_kernel.cu
  Add g_texture_pool_ptr constant global + bind_texture_pool() to bridges
  Recompile PTX → run test_procedural_texture_surface.py → PASS

Step 3 — Entity AI (P0):
  Create entity_star.h + entity_behavior.cu
  Add cases 0x180–0x1BF to modular_rpn_kernel.cu
  Add bind_entity_soa() to sovereign_bridges.py
  Recompile PTX → run test_sovereign_entity_surface.py → PASS

Step 4 — 2D physics (P1):
  Create phys2d_aabb.cu
  Add cases 0x130–0x14F to modular_rpn_kernel.cu
  Recompile PTX → run test_phys2d_aabb_surface.py → PASS

Step 5 — BEHAVIOR_PHASE in game loop (P1):
  Extend trm_step_fused.cu BEHAVIOR_PHASE stub → real dispatch
  Register entity_count and entity_star_ptr in TRMLauncher.refine_step()
```

---

## Source Reference: fr_public (kkrieger)

Cloned at: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/fr_public/`

Key files:
- `ktg/gentexture.cpp` — noise algorithms (Perlin, Value, Grid), LFSR seed `0x93638245u`, gradient table
- `werkkzeug3_kkrieger/` — operator graph DAG evaluation order  
- `v2/` — V2 music synthesizer (future: Audio Galaxy procedural synthesis)

The kkrieger `PGradient2` gradient function and `SmoothStep` quintic are the reference for `tex_perlin_noise_kernel`. Preserve LFSR seed `0x93638245u` for canonical kkrieger noise character.
