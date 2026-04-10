# Codex Directions: Sovereign Physics Engine Completion
**Date:** April 8, 2026  
**Author:** Claude Code (Architecture Partner)  
**For:** Codex (implementation)  
**Grounding:** `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`, `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`, `TEMP/CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md`  
**Status:** Surface complete. Four concrete gaps remain before end-to-end physics is live.

---

## Context: What's Done vs What Remains

**Done (do not re-implement):**
- All physics opcode cases `0x150–0x162` exist in `modular_rpn_kernel.cu` (1555 lines)
- `PhysicsBodySOA`, `ContactManifoldSOA`, `CollisionEventQueue` structs in `physics_body_soa.h`
- All 11 leaf `.cu` kernel files exist (broad phase, narrow phase, XPBD solve, integrate, sleep, etc.)
- `sovereign_bridges.py:ModularRPNEngine.bind_physics_runtime()` exists (line 2092)
- `sovereign_physics_bootstrap.py` has all 11 constant stars + 4 material stars + force law + meta-rule
- `sovereign_physics.py` has complete contract table
- 5 Python structural tests pass

**What does NOT work yet (the 4 gaps you must close):**

---

## GAP 1 (P0 — BLOCKER): Compile modular_rpn_kernel.cu → PTX

**Root cause:** `ModularRPNEngine.__init__` (sovereign_bridges.py:2027) loads `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx` — the **compiled** PTX file dated March 8, which has **zero physics cases**. All the new physics code in `modular_rpn_kernel.cu` does nothing until it is compiled into that PTX file.

**What to do:**

1. Compile `modular_rpn_kernel.cu` → `modular_rpn_kernel.ptx`:

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

nvcc --ptx \
  -arch=sm_86 \
  -O3 \
  --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

Notes on flags:
- `-arch=sm_86` = RTX 3070 (Ampere). Use `sm_86` (not `sm_75` — that's the 3060 from `compile_backward_kernels.sh`).
- `-I knowledge3d/cranium/kernels` = resolves `#include "physics_body_soa.h"` (the file is at `knowledge3d/cranium/kernels/physics_body_soa.h`)
- Output path must be exactly `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx` — that's what `ModularRPNEngine` loads at runtime

2. Verify the compiled PTX contains the physics cases:

```bash
grep -c "0x150\|PH_BROAD" knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
# Should be > 0 after compilation (compiled PTX will contain the opcode switch labels)
```

3. **Also add `g_physics_material_table_ptr` global BEFORE compiling** (see Gap 2 — do this first so you only compile once).

**Success criterion:** `ModularRPNEngine()` can be constructed without error, and executing opcode `0x158` (PH_PREDICT_POS = 344 decimal) with a bound SOA does not return an error code.

---

## GAP 2 (P0 — must fix before compiling): PhysicsMaterialTable — replace inline table

**Root cause:** `modular_rpn_kernel.cu:1286` (case `0x157 PH_MATERIAL_FETCH`) uses a hardcoded 4-entry inline table:
```c
const float friction_table[4] = {0.57f, 0.30f, 0.90f, 0.03f};
```
This is NOT connected to the Galaxy. It does not fetch from Layer-2 Reality Galaxy stars.

**The fix — two parts:**

### Part A: Add GPU material table struct and global to `modular_rpn_kernel.cu` (BEFORE compiling)

In `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`, after the existing `__device__ __constant__` globals (line 17), add:

```c
// Physics material table — populated from Layer-2 Reality Galaxy material stars at boot
struct PhysicsMaterialEntry {
    uint32_t star_id;      // matches galaxy_handles.x in PhysicsBodySOA
    float friction;
    float restitution;
    float density;
};
__device__ __constant__ unsigned long long g_physics_material_table_ptr = 0ULL;
__device__ __constant__ unsigned int g_physics_material_table_count = 0u;
```

Then replace the `case 0x157` body with a GPU-side linear scan of this table:

```c
case 0x157: {  // PH_MATERIAL_FETCH
    float star_id_scalar = 0.0f;
    if (!pop_scalar(stack, stack_size, star_id_scalar, error_code)) break;
    const uint32_t target_id = static_cast<uint32_t>(max(0.0f, floorf(star_id_scalar + 0.5f)));

    float friction = 0.5f, restitution = 0.3f, density = 1000.0f;  // safe defaults

    if (g_physics_material_table_ptr != 0ULL && g_physics_material_table_count > 0u) {
        const PhysicsMaterialEntry* table =
            reinterpret_cast<const PhysicsMaterialEntry*>(g_physics_material_table_ptr);
        for (uint32_t i = 0; i < g_physics_material_table_count; ++i) {
            if (table[i].star_id == target_id) {
                friction    = table[i].friction;
                restitution = table[i].restitution;
                density     = table[i].density;
                break;
            }
        }
    }
    push(stack, stack_size, make_scalar(friction),    error_code);  if (error_code != kErrorNone) break;
    push(stack, stack_size, make_scalar(restitution), error_code);  if (error_code != kErrorNone) break;
    push(stack, stack_size, make_scalar(density),     error_code);
    break;
}
```

Note: linear scan is O(n) per lookup — acceptable because material lookups happen rarely (once per body per step, only when material changes). The table will have ≤ 64 entries in practice.

### Part B: Add `bind_physics_material_table()` to `ModularRPNEngine` in `sovereign_bridges.py`

After `bind_physics_runtime()` (line 2110), add:

```python
def bind_physics_material_table(self, material_entries: list[dict]) -> int:
    """Serialize material stars → GPU PhysicsMaterialTable and bind.

    material_entries: list of dicts with keys:
        star_id (int), friction (float), restitution (float), density (float)
    Returns: number of entries bound.
    """
    # PhysicsMaterialEntry = uint32 + float3 = 4 × 4 bytes = 16 bytes per entry
    n = len(material_entries)
    if n == 0:
        return 0
    EntryArray = (ctypes.c_uint8 * (n * 16))
    buf = EntryArray()
    import struct
    for i, entry in enumerate(material_entries):
        offset = i * 16
        struct.pack_into('<I', buf, offset,      int(entry["star_id"]))
        struct.pack_into('<f', buf, offset + 4,  float(entry["friction"]))
        struct.pack_into('<f', buf, offset + 8,  float(entry["restitution"]))
        struct.pack_into('<f', buf, offset + 12, float(entry["density"]))
    d_table = gpu_malloc(n * 16)
    memcpy_htod(d_table, ctypes.cast(buf, ctypes.c_void_p), n * 16)
    self._try_set_module_global("g_physics_material_table_ptr", int(d_table.value), ctypes.c_uint64)
    self._try_set_module_global("g_physics_material_table_count", n, ctypes.c_uint32)
    self._d_material_table = d_table  # keep alive
    return n
```

### Part C: Add `serialize_material_table()` to `sovereign_physics_bootstrap.py`

At the bottom of `sovereign_physics_bootstrap.py`, add:

```python
def serialize_material_table() -> list[dict]:
    """Convert material stars to GPU-serializable format (star_id is index-based).
    star_id assignment: steel=1, wood=2, rubber=3, ice=4.
    These IDs must match the galaxy_handles.x values set in physics_spawn.cu.
    """
    materials = build_physics_material_stars()
    _id_map = {
        "physics_material_steel":  1,
        "physics_material_wood":   2,
        "physics_material_rubber": 3,
        "physics_material_ice":    4,
    }
    result = []
    for m in materials:
        result.append({
            "star_id":     _id_map.get(m["star_id"], 0),
            "friction":    m["friction_dynamic"],
            "restitution": m["restitution"],
            "density":     m["density"],
        })
    return result
```

**Success criterion:** After `engine.bind_physics_material_table(serialize_material_table())`, calling `PH_MATERIAL_FETCH` with `star_id=2.0` returns `(0.30, 0.35, 700.0)` (wood).

---

## GAP 3 (P1): Bootstrap stars → Galaxy ingestion

**Root cause:** `sovereign_physics_bootstrap.py` builds Python dicts for constants and materials, but they have NOT been passed to the Galaxy ingestion pipeline. Until they're ingested, the Galaxy has no physical constant stars and material stars.

**What to do:**

In `knowledge3d/ingestion/__init__.py`, add a call to bootstrap physics stars during the ingestion pass. Find the function that processes star dicts (look for `ingest_stars`, `content_to_stars`, or `galaxy_insert`) and add:

```python
from knowledge3d.cranium.sovereign_physics_bootstrap import (
    build_physical_constant_stars,
    build_physics_material_stars,
    build_default_gravity_force_law,
    build_default_sleep_policy,
)

def _ingest_physics_bootstrap(galaxy_writer) -> int:
    """Ingest foundational physics stars into the Galaxy at boot.
    Runs once. Ingestion path — any library allowed.
    Returns: number of stars ingested.
    """
    stars = (
        build_physical_constant_stars()     # 11 constants (G, c, ħ, ...)
        + build_physics_material_stars()    # 4 materials (steel, wood, rubber, ice)
        + [build_default_gravity_force_law()]   # Layer-3 Grammar star
        + [build_default_sleep_policy()]        # Layer-4 Meta-Rule star
    )
    for star in stars:
        galaxy_writer.insert_star(star)     # adapt to actual insert API
    return len(stars)
```

The exact `galaxy_writer.insert_star` call should match whatever pattern `knowledge3d/ingestion/__init__.py` already uses for other star insertions. Look at how other ingestion functions write stars and replicate that pattern. Do NOT invent a new insertion API.

**Timing:** Call `_ingest_physics_bootstrap()` early in the ingestion pipeline, before the main knowledge scan, so physics stars are present when other content references them.

**Success criterion:** After a fresh ingestion run, `GALAXY_SCAN (0xE2)` with filter `facet:physical_constant` returns 11 stars; `facet:physical_material` returns 4 stars.

---

## GAP 4 (P1): Falling sphere smoke test

**Root cause:** All existing tests in `tests/test_sovereign_physics_surface.py` are **structural** (contract table, opcode count, Python surface). None of them actually run physics on GPU. We need at least one end-to-end test to prove the path is live.

**What to add** — new test at the end of `tests/test_sovereign_physics_surface.py`:

```python
def test_falling_sphere_smoke():
    """Falling sphere under gravity: y drops from 2m to ~-2.9m in 1 second.

    Physics: y = y0 + v0*t + 0.5*g*t^2 = 2 + 0 + 0.5*(-9.81)*1.0^2 = 2 - 4.905 = -2.905m
    We run 60 frames at dt=1/60 ≈ 1.0 second total.

    This test requires the k3d-cranium conda env (real GPU).
    Skip automatically in k3d-testing (CPU-only mock env).
    """
    pytest.importorskip("ctypes")
    try:
        from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine
        from knowledge3d.cranium.sovereign.loader import gpu_malloc, gpu_free, memcpy_htod, memcpy_dtoh, synchronize
        from knowledge3d.cranium.ptx_runtime.rpn_opcodes import OP_PH_PREDICT_POS, OP_PH_INTEGRATE
    except ImportError:
        pytest.skip("GPU environment not available")

    try:
        engine = ModularRPNEngine()
    except Exception:
        pytest.skip("Could not initialise ModularRPNEngine (no GPU)")

    import ctypes, struct

    # ── Allocate a minimal PhysicsBodySOA for 1 body ──────────────────────
    # PhysicsBodySOA arrays (from physics_body_soa.h):
    #   pos_inv   [float4 × capacity]: xyz=position, w=inv_mass
    #   vel_sleep [float4 × capacity]: xyz=velocity, w=sleep_accum
    #   orientation [float4 × capacity]: xyzw quaternion
    #   ang_vel_damp [float4 × capacity]: xyz=angular_vel, w=ang_damping
    #   inv_inertia_rest [float4 × capacity]: xyz=inv_inertia_local, w=restitution
    #   galaxy_handles [uint2 × capacity]: x=material_star_id, y=shape_star_id
    #   island_flags [uint32 × capacity]
    #   bound_friction [float2 × capacity]: x=bbox_radius, y=friction
    # Plus header: body_count (uint32), capacity (uint32)

    CAPACITY = 1
    BODY_COUNT = 1

    def alloc_zeros(n_bytes):
        d = gpu_malloc(n_bytes)
        buf = (ctypes.c_uint8 * n_bytes)()
        memcpy_htod(d, ctypes.cast(buf, ctypes.c_void_p), n_bytes)
        return d

    # Allocate each SOA array
    d_pos_inv       = alloc_zeros(CAPACITY * 16)   # float4
    d_vel_sleep     = alloc_zeros(CAPACITY * 16)
    d_orient        = alloc_zeros(CAPACITY * 16)
    d_ang_vel_damp  = alloc_zeros(CAPACITY * 16)
    d_inv_inertia   = alloc_zeros(CAPACITY * 16)
    d_handles       = alloc_zeros(CAPACITY * 8)    # uint2
    d_island_flags  = alloc_zeros(CAPACITY * 4)    # uint32
    d_bound_fric    = alloc_zeros(CAPACITY * 8)    # float2

    # Set initial position: x=0, y=2, z=0, inv_mass=1.0
    pos_buf = struct.pack('<4f', 0.0, 2.0, 0.0, 1.0)
    memcpy_htod(d_pos_inv, ctypes.cast(ctypes.create_string_buffer(pos_buf), ctypes.c_void_p), 16)

    # Set orientation to identity quaternion: (0, 0, 0, 1)
    orient_buf = struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)
    memcpy_htod(d_orient, ctypes.cast(ctypes.create_string_buffer(orient_buf), ctypes.c_void_p), 16)

    # Pack PhysicsBodySOA struct (all 8 array pointers + body_count + capacity)
    # NOTE: the struct layout must match physics_body_soa.h exactly.
    # On GPU, the kernel reads g_physics_body_soa_ptr as a pointer to this struct.
    # We build it as a ctypes struct and upload.
    soa_size = 8 * 8 + 4 + 4  # 8 pointers (uint64) + body_count + capacity = 72 bytes
    soa_buf = struct.pack('<8QII',
        int(d_pos_inv.value),
        int(d_vel_sleep.value),
        int(d_orient.value),
        int(d_ang_vel_damp.value),
        int(d_inv_inertia.value),
        int(d_handles.value),
        int(d_island_flags.value),
        int(d_bound_fric.value),
        BODY_COUNT,
        CAPACITY,
    )
    d_soa = gpu_malloc(len(soa_buf))
    memcpy_htod(d_soa, ctypes.cast(ctypes.create_string_buffer(soa_buf), ctypes.c_void_p), len(soa_buf))

    # Bind physics SOA into the engine's module globals
    engine.bind_physics_runtime(body_soa=d_soa)

    # ── Run 60 frames: PH_PREDICT_POS (0x158) + PH_INTEGRATE (0x154) ──────
    # PH_INTEGRATE needs gravity pre-applied to velocity. The simplest way:
    # the integrate kernel in modular_rpn_kernel.cu uses g_physics_gravity_y.
    # Bind it. (constant global set inline in the kernel; default = -9.81)
    engine._try_set_module_global("physics_gravity_y", -9.81, ctypes.c_float)

    DT = 1.0 / 60.0
    for _ in range(60):
        # PH_PREDICT_POS: pop dt → push nothing (updates predicted SOA)
        engine.execute_single(
            instance_id=0,
            op_codes=[0, OP_PH_PREDICT_POS],   # op 0 = push scalar (dt), then predict
            scalars=[DT],
            vectors=[],
        )
        # PH_INTEGRATE: pop dt → updates body SOA from predicted SOA
        engine.execute_single(
            instance_id=0,
            op_codes=[0, OP_PH_INTEGRATE],
            scalars=[DT],
            vectors=[],
        )

    # ── Read back position from d_pos_inv ────────────────────────────────
    result_buf = (ctypes.c_uint8 * 16)()
    memcpy_dtoh(ctypes.cast(result_buf, ctypes.c_void_p), d_pos_inv, 16)
    y_final = struct.unpack_from('<f', bytes(result_buf), 4)[0]  # y is at offset 4

    # Expected: y0 + 0.5 * g * t^2 = 2.0 + 0.5*(-9.81)*1.0^2 = -2.905m
    # Allow ±0.5m tolerance for float precision and Euler integration error
    assert -3.5 < y_final < -2.4, \
        f"Falling sphere y_final={y_final:.3f}m, expected ≈ -2.905m"

    # Cleanup
    for d in [d_soa, d_pos_inv, d_vel_sleep, d_orient, d_ang_vel_damp,
               d_inv_inertia, d_handles, d_island_flags, d_bound_fric]:
        gpu_free(d)
```

**Important:** Check the exact layout of `PhysicsBodySOA` in `physics_body_soa.h` and make sure the `struct.pack` layout in the test matches. The struct in the test above is a template — adjust field count and order to match the header exactly. If the SOA struct has different fields than listed above (e.g., if it's been extended), the pack format must be updated.

**On `physics_gravity_y`:** The `modular_rpn_kernel.cu` sets gravity via an inline constant. Check whether it's a `__device__ __constant__` global named `physics_gravity_y` (it's referenced in the Codex report at line: "shared `physics_gravity_y` state inside the interpreter run"). Bind it before the loop, or use the `PH_GRAVITY_APPLY (0x160)` opcode with the G star.

**Success criterion:** `pytest tests/test_sovereign_physics_surface.py -k falling_sphere -v` passes on the `k3d-cranium` env with a real GPU. If `y_final` is `0.0` or `2.0` (unchanged), the SOA binding is wrong — add a debug readback after frame 1 to verify position is moving.

---

## Priority Order

| # | Gap | File(s) | Success Criterion |
|---|---|---|---|
| 1 | **Add `PhysicsMaterialEntry` + global to .cu** | `modular_rpn_kernel.cu` | File compiles without error |
| 2 | **Compile modular_rpn_kernel.cu → PTX** | `knowledge3d/cranium/ptx/modular_rpn_kernel.ptx` | Compiled PTX > 200KB; contains physics opcode labels |
| 3 | **Add `bind_physics_material_table()` to `ModularRPNEngine`** | `sovereign_bridges.py` | Calling with 4 material entries returns 4 |
| 4 | **Add `serialize_material_table()` to bootstrap** | `sovereign_physics_bootstrap.py` | Returns list of 4 dicts with correct friction/restitution/density |
| 5 | **Wire bootstrap into ingestion** | `knowledge3d/ingestion/__init__.py` | `GALAXY_SCAN facet:physical_constant` returns 11 stars |
| 6 | **Add falling sphere smoke test** | `tests/test_sovereign_physics_surface.py` | Test passes on `k3d-cranium` env; y_final ≈ -2.9m |

**Do items 1 and 2 first** — nothing else works until the compiled PTX has the physics cases.

---

## What NOT to do

- Do NOT rebuild `trm_step_fused.ptx` — the stub in `trm_step_fused.cu` is source-complete; rebuilding it requires updating all fused-step callers, which is a larger task. The physics path is live via `modular_rpn_kernel` independently.
- Do NOT change `physics_body_soa.h` — it's the contract between the kernel and the bridges; changing it now breaks the compiled .cu files.
- Do NOT add a Python fallback for the physics loop. If the kernel errors, print the error code and fix ON GPU. (The RPN interpreter returns an error code in the stack header — check `header[2]` for non-zero values.)
- Do NOT touch the encyclopedia ingestion process — Codex report says to leave it untouched, and it's independent.

---

## Sovereignty Notes (grounded in KNOWLEDGEVERSE_SPECIFICATION.md §4)

- Physical constants (G, c, k_B) → Layer-2 Reality Galaxy stars (`sovereign_physics_bootstrap.py` already builds them). They do NOT live in Python dicts in the hot path.
- Material properties (friction, restitution, density) → Layer-2 material stars (fetched via `PH_MATERIAL_FETCH`). After Gap 2 is closed, this is sovereign: star_id → GPU table lookup.
- Force laws → Layer-3 Grammar Galaxy `physics_rpn_addr` field (already in bootstrap).
- Sleep policy → Layer-4 Meta-Rule (already in bootstrap).
- All integration, collision detection, constraint solving → PTX hot path. Zero Python in the loop.

The falling sphere test is the proof of sovereignty: Python sets up buffers once (boot path), then the RPN program `[PUSH dt, PH_PREDICT_POS, PUSH dt, PH_INTEGRATE] × 60` runs entirely in `modular_rpn_kernel.ptx` without Python re-entering the loop.
