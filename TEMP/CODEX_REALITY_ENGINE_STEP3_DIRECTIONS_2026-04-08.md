# Codex Directions: Step 3 — Entity AI (Grounded in Vocabulary Specs)
**Date:** 2026-04-08  
**Author:** Claude (Architecture)  
**Grounding read before writing this:** `AVATAR_EMBODIMENT_SPECIFICATION.md`, `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`, `KNOWLEDGEVERSE_SPECIFICATION.md`, `knowledge3d/knowledgeverse/meaning_star.py`

---

## Critical Correction to the Original Step 3 Spec

The original `CODEX_REALITY_ENGINE_SPEC_2026-04-08.md` proposed a standalone `entity_star.h` with a 64-byte `EntityStar` struct. **This is wrong.** Reading the vocabulary specs and the live code reveals the correct architecture:

**`behavior_rpn` already exists on `MeaningCentricStar`** — `knowledge3d/knowledgeverse/meaning_star.py` line 110:
```python
behavior_rpn: str | None = None
```

An entity in K3D is NOT a new data type. It is a `MeaningCentricStar` with:
- `meaning_class = "entity"`
- `behavior_rpn` = the BH_* opcode sequence (a string compiled by `procedural_compiler.py`)
- `visual_rpn` = HAnim skeleton construction RPN (links to 3DObjects Galaxy per Avatar spec §2.1)
- `reality_refs` = physics body references (links to Reality Galaxy)
- `house_position` = where the entity lives in the House
- `star_id` = the entity's `canonicalId` per Avatar spec §5.3

The Avatar spec (`AVATAR_EMBODIMENT_SPECIFICATION.md` §7.1) is unambiguous: **"TRM IS the Avatar — lives in House, thinks in Galaxy, runs as game loop."** Every AI entity is an avatar with a body (HAnim skeleton, `visual_rpn`), a mind (Galaxy, `behavior_rpn`), and a House position.

---

## Step 3A: EntityHotPath Compact Struct (replaces EntityStar)

The BEHAVIOR_PHASE kernel needs a compact GPU-side projection of the full star. This is NOT the star itself — it's an index card extracted at boot time from Galaxy entries where `meaning_class == "entity"`.

Create `knowledge3d/cranium/kernels/entity_hot_path.h`:

```c
// GENERATED at boot from Galaxy entries with meaning_class == "entity"
// NOT the canonical star — the canonical star is MeaningCentricStar in Python/House
// Compact hot-path data for the BEHAVIOR_PHASE GPU kernel tick
struct EntityHotPath {
    // === Identity ===
    uint32_t  star_table_idx;     // 0x00  index into the full GPU galaxy star table
    uint32_t  physics_body_id;    // 0x04  slot in PhysicsBodySOA (from reality_refs)

    // === Behavior ===
    uint64_t  behavior_rpn_addr;  // 0x08  compiled behavior_rpn program address in GPU memory

    // === Spatial (from house_position + physics SOA) ===
    float     house_x;            // 0x10
    float     house_y;            // 0x14
    float     house_z;            // 0x18

    // === Tick control ===
    uint8_t   sleep_state;        // 0x1C  0=awake(60Hz) 1=doze(10Hz) 2=asleep(1Hz)
    uint8_t   faction;            // 0x1D  faction ID for blackboard coordination
    uint8_t   ai_tier;            // 0x1E  0=reactive 1=planning 2=strategic
    uint8_t   perception_flags;   // 0x1F  SEE=0x1 HEAR=0x2 SCENT=0x4

    // === Perception ===
    float     perception_radius;  // 0x20  Morton Octree query radius
    float     last_player_dist;   // 0x24  updated each tick from PhysicsBodySOA
    float     awareness;          // 0x28  0.0=unaware 1.0=fully aware

    // === Faction memory ===
    uint32_t  blackboard_star_id; // 0x2C  faction shared blackboard (Galaxy star idx)

    // === L4 meta-rule ===
    uint32_t  meta_rule_addr;     // 0x30  sleep-consolidation program (from meta_refs[0])

    // === HAnim body linkage (per Avatar spec §3.1) ===
    float     cranial_origin[3];  // 0x34  k3d_cranial_origin position in world space
    float     _pad;               // 0x40  align to 68 bytes (next float4 boundary)
};
// sizeof(EntityHotPath) == 68 bytes
// GPU array: EntityHotPath entity_hot_paths[MAX_ENTITIES]  (MAX_ENTITIES = 4096)
extern __constant__ EntityHotPath* g_entity_hot_path_ptr;
extern __constant__ uint32_t       g_entity_count;
```

---

## Step 3B: Entity Bootstrap (sovereign_entity_bootstrap.py)

Create `knowledge3d/cranium/sovereign_entity_bootstrap.py`.

This follows the pattern of `sovereign_physics_bootstrap.py` — uses `MeaningCentricStar` (ingestion path, not hot path).

**Read before writing:** `knowledge3d/cranium/sovereign_physics_bootstrap.py` (existing pattern) and `knowledge3d/knowledgeverse/meaning_star.py` (the schema).

```python
"""
Entity bootstrap: defines the foundational entities (avatars) that live in the House.
These are MeaningCentricStar instances, NOT custom structs.
All entities follow AVATAR_EMBODIMENT_SPECIFICATION.md:
  - meaning_class = "entity"
  - behavior_rpn = BH_* opcode sequence (the entity's AI behavior)
  - visual_rpn   = HAnim skeleton construction RPN (how the body is built)
  - reality_refs = ["physics:body:N"] (links to Reality Galaxy physics entries)
  - house_position = (x, y, z) in House coordinates
"""
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

# --- The Primary AI Avatar (TRM entity) ---
# Per Avatar spec §7.1: TRM IS the Avatar — always present in the House
TRM_AVATAR_STAR = MeaningCentricStar(
    star_id="entity:trm:primary",
    meaning_class="entity",
    meaning_rpn="TRM AVATAR GAME_LOOP NINE_CHAIN_SWARM HALTING_GATE",
    domain="House/Avatars",
    taxonomy_refs=["concept_avatar", "concept_ai_entity"],
    visual_rpn=(
        "HANIM_LOA2_SKELETON "
        "k3d_cranial_origin k3d_tablet_grip k3d_thought_emitter "
        "DUAL_TEXTURE_BIND UV_MAP_0 UV_MAP_1"
    ),
    behavior_rpn=(
        # TRM's behavior is its full game loop — not a simple BH_SEEK sequence
        # This is the sovereign game loop hook: PERCEIVE → NAVIGATE → REASON → DECIDE → ACT
        "BH_PERCEIVE 50.0 "
        "BH_SLEEP_CHECK "
        "BH_BT_TICK"
    ),
    reality_refs=["physics:body:0"],          # TRM occupies physics body slot 0
    grammar_refs=["rule:trm_game_loop"],
    meta_refs=["meta:sleep_consolidation"],
    house_position=(0.0, 1.75, 0.0),          # standing at House origin, 1.75m tall
    house_room="House",
    confidence=1,
    polarity=1,
)

# --- Faction Blackboard Stars ---
# Per AI spec: each faction shares a Galaxy star for coordination
def build_faction_blackboard(faction_id: int, faction_name: str, house_pos: tuple) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=f"blackboard:faction:{faction_id}",
        meaning_class="blackboard",
        meaning_rpn=f"FACTION {faction_id} BLACKBOARD SHARED_MEMORY",
        domain="House/Entities/Factions",
        taxonomy_refs=["concept_faction", "concept_coordination"],
        behavior_rpn="BH_BLACKBOARD_READ BH_BLACKBOARD_WRITE",
        house_position=house_pos,
        house_room="House",
        confidence=1,
        polarity=0,
    )

FACTION_NEUTRAL_BLACKBOARD = build_faction_blackboard(0, "neutral", (10.0, 0.0, 0.0))
FACTION_ALLY_BLACKBOARD    = build_faction_blackboard(1, "ally",    (12.0, 0.0, 0.0))
FACTION_RIVAL_BLACKBOARD   = build_faction_blackboard(2, "rival",   (14.0, 0.0, 0.0))

FOUNDATIONAL_ENTITY_STARS = [
    TRM_AVATAR_STAR,
    FACTION_NEUTRAL_BLACKBOARD,
    FACTION_ALLY_BLACKBOARD,
    FACTION_RIVAL_BLACKBOARD,
]

def build_entity_stars() -> list[MeaningCentricStar]:
    """Return all foundational entity stars for ingestion."""
    return list(FOUNDATIONAL_ENTITY_STARS)


def build_entity_hot_path_array(galaxy_manager) -> list[dict]:
    """
    Scan the Galaxy for entries with meaning_class == 'entity'.
    Build the compact EntityHotPath data as Python dicts (uploaded to GPU by bind_entity_soa()).
    Called ONCE at boot — this is NOT hot path.
    Returns list of dicts with keys matching EntityHotPath fields.
    """
    hot_paths = []
    for galaxy_name in galaxy_manager.galaxy_names():
        for entry in galaxy_manager.iter_entries(galaxy_name):
            metadata = entry.get("metadata", {}) or {}
            meaning_star_data = metadata.get("meaning_star", {}) or {}
            if str(meaning_star_data.get("meaning_class", "")).strip() != "entity":
                continue
            star_id = str(meaning_star_data.get("star_id", entry.get("id", ""))).strip()
            behavior_rpn = meaning_star_data.get("behavior_rpn") or ""
            reality_refs = meaning_star_data.get("reality_refs") or []
            # Derive physics_body_id from reality_refs: "physics:body:N" → N
            physics_body_id = 0
            for ref in reality_refs:
                if str(ref).startswith("physics:body:"):
                    try:
                        physics_body_id = int(str(ref).split(":")[-1])
                    except ValueError:
                        pass
                    break
            house_position = meaning_star_data.get("house_position", [0.0, 0.0, 0.0])
            hot_paths.append({
                "star_id": star_id,
                "physics_body_id": physics_body_id,
                "behavior_rpn": behavior_rpn,
                "house_x": float(house_position[0]),
                "house_y": float(house_position[1]),
                "house_z": float(house_position[2]),
                "sleep_state": 0,
                "faction": 0,
                "ai_tier": 0,
                "perception_flags": 0x1,     # SEE by default
                "perception_radius": 30.0,
                "last_player_dist": 999.0,
                "awareness": 0.0,
                "blackboard_star_id": 0,
                "meta_rule_addr": 0,
                "cranial_origin": [0.0, 1.6, 0.0],  # approx head height
            })
    return hot_paths
```

---

## Step 3C: entity_behavior.cu (BH_* opcode dispatch)

Create `knowledge3d/cranium/kernels/entity_behavior.cu`.

This file is **included inline by `modular_rpn_kernel.cu`** (like `tex_noise_kernels.cu`). It provides device functions for BH_* cases.

**Architecture note:** BH_PERCEIVE must call `morton_spatial_query()` — the SAME device function used by TRM Frustum culling. Confirm the function signature from the existing Morton Octree kernel before writing the case. Do NOT duplicate; call the existing entry point.

```c
// entity_behavior.cu — BH_* device functions for modular_rpn_kernel.cu
// Included by modular_rpn_kernel.cu, NOT compiled standalone.
// Sovereignty: all BH_* ops are device-only — no host/Python calls.

// BH_PERCEIVE (0x180): Morton Octree spatial query
// Pops: radius:f32
// Pushes: list_addr:u64 (temp buffer in Galaxy scratch; count:u32 + entity_id[]:u32)
__device__ void bh_perceive(
    float radius,
    const EntityHotPath* entities,
    uint32_t entity_count,
    const float3 self_pos,
    uint8_t* scratch_buf,         // Galaxy scratch region for output list
    uint64_t* out_list_addr
) {
    // Walk entity array — find entities within radius
    // NOTE: Later replace with morton_spatial_query() once the function is confirmed
    // to accept EntityHotPath positions. For now: linear scan (safe, O(N) for small N).
    uint32_t* list = (uint32_t*)scratch_buf;
    uint32_t count = 0;
    for (uint32_t i = 0; i < entity_count; i++) {
        float dx = entities[i].house_x - self_pos.x;
        float dy = entities[i].house_y - self_pos.y;
        float dz = entities[i].house_z - self_pos.z;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 <= radius * radius && dist2 > 0.01f) {  // exclude self
            list[count++] = i;   // entity index (not physics body id)
        }
    }
    list[-1] = count;   // prepend count (caller must account for -1 offset)
    *out_list_addr = (uint64_t)(uintptr_t)list;
}

// BH_SEEK (0x181): steering toward target entity
// Pops: target_idx:u32
// Pushes: fx:f32, fy:f32, fz:f32
__device__ void bh_seek(
    uint32_t target_idx,
    const EntityHotPath* entities,
    uint32_t self_idx,
    const PhysicsBodySOA* soa,
    float* out_fx, float* out_fy, float* out_fz
) {
    float4 self_pos  = soa->pos[entities[self_idx].physics_body_id];
    float4 tgt_pos   = soa->pos[entities[target_idx].physics_body_id];
    float dx = tgt_pos.x - self_pos.x;
    float dy = tgt_pos.y - self_pos.y;
    float dz = tgt_pos.z - self_pos.z;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 0.001f) { *out_fx = 0.0f; *out_fy = 0.0f; *out_fz = 0.0f; return; }
    const float MAX_SPEED = 3.0f;  // m/s — parameterize later via Grammar Galaxy rule
    float4 vel = soa->vel[entities[self_idx].physics_body_id];
    *out_fx = (dx/len)*MAX_SPEED - vel.x;
    *out_fy = (dy/len)*MAX_SPEED - vel.y;
    *out_fz = (dz/len)*MAX_SPEED - vel.z;
}

// BH_FLEE (0x182), BH_ARRIVE (0x183), BH_SEPARATE (0x184):
// Follow same pattern — pop params, compute force, push xyz.
// BH_APPLY_FORCE (0x185): atomicAdd to PhysicsBodySOA.forces[body_id]
// BH_SLEEP_CHECK (0x189): distance < 50 → state=0; < 200 → state=1; else state=2
// BH_BLACKBOARD_READ/WRITE (0x18A/0x18B): atomicExch on faction_blackboard_star field

// BH_BT_TICK (0x186):
// Pops: bt_program_addr:u64
// Pushes: status:u32 (0=FAIL 1=OK 2=RUNNING)
// Implementation: call execute_rpn_program(bt_program_addr, ...) recursively
// The behavior tree IS just another RPN program — selector/sequence/leaf nodes
// map directly to BH_BT_SELECTOR / BH_BT_SEQUENCE opcodes within the sub-program.
```

**Critical:** Do NOT write BH_GOAP_PLAN (0x188) or BH_PATHFIND (0x18C) yet. Both require LED-A* pool access — that's a separate wiring task. Stub them to push `0` (FAILURE) for now.

---

## Step 3D: BEHAVIOR_PHASE in trm_step_fused.cu

Extend the existing `trm_physics_phase_stub()` pattern in `knowledge3d/cranium/ptx/trm_step_fused.cu`.

Add after the physics stub call (line 88 in the current source):

```c
// BEHAVIOR_PHASE — per entity BH_* program execution
// Currently a stub; promoted to full dispatch in Step 5.
__device__ __forceinline__ void trm_behavior_phase_stub(
    const EntityHotPath* __restrict__ entity_hot_paths,
    unsigned int entity_count,
    unsigned int frame_counter
) {
    // BEHAVIOR_PHASE boundary. Full dispatch: iterate entity_hot_paths,
    // check sleep_state vs frame_counter, call execute_rpn_program(behavior_rpn_addr).
    // Deferred to Step 5 per spec — entities need hot_path array to be bound and
    // BH_* opcode cases in modular_rpn_kernel.cu to be compiled first.
    (void)entity_hot_paths;
    (void)entity_count;
    (void)frame_counter;
}
```

Extend `trm_step_fused` kernel signature (after existing physics params):
```c
extern "C" __global__ void trm_step_fused(
    // ... existing params ...
    const void* __restrict__ physics_soa_ptr,
    const void* __restrict__ contact_soa_ptr,
    const void* __restrict__ event_queue_ptr,
    unsigned int body_count,
    float physics_dt,
    unsigned int solver_iterations,
    // NEW: behavior phase params
    const void* __restrict__ entity_hot_path_ptr,   // EntityHotPath[]
    unsigned int entity_count,
    unsigned int frame_counter
)
```

Then in the kernel body (after existing physics stub call):
```c
if (tid == 0) {
    trm_behavior_phase_stub(
        (const EntityHotPath*)entity_hot_path_ptr,
        entity_count,
        frame_counter
    );
}
```

**Do NOT update trm_launcher.py callers yet** — that is Step 5 and requires a separate handoff.

---

## Step 3E: Sovereign Bridges Additions

In `knowledge3d/cranium/bridges/sovereign_bridges.py`, add:

```python
def bind_entity_soa(self, entity_hot_path_array: list[dict]) -> None:
    """
    Upload EntityHotPath compact array to VRAM.
    Call at boot after build_entity_hot_path_array() completes.
    entity_hot_path_array: list of dicts from sovereign_entity_bootstrap.build_entity_hot_path_array()
    """
    # Serialize to ctypes struct matching EntityHotPath layout
    # Upload via cudaMemcpy to device buffer
    # Bind via _try_set_module_global("g_entity_hot_path_ptr", ...)
    # Bind g_entity_count
    pass  # Codex implements

def bind_entity_behavior_programs(self, compiled_programs: dict[str, bytes]) -> None:
    """
    Upload compiled BH_* RPN programs to VRAM.
    compiled_programs: star_id → compiled bytes (from procedural_compiler.py)
    Returns dict: star_id → device_addr (uint64) for EntityHotPath.behavior_rpn_addr
    """
    pass  # Codex implements
```

---

## Step 3F: Ingestion Wiring

In `knowledge3d/ingestion/__init__.py`, add `ingest_entity_bootstrap(galaxy_manager)` following the pattern of `ingest_physics_bootstrap()`. Wire into:
- `scripts/fundamental_ingest_payloads.py`
- `knowledge3d/tools/ingest_from_manifest.py`

The entity stars use `galaxy_ref = "House"` (entities live in the House, not just in abstract Galaxy memory) and `to_galaxy_entry(galaxy_name="Reality")` per the Knowledgeverse spec (entities belong to the Reality Galaxy region for the purposes of physics and behavior).

---

## Step 3G: Compile and Test

After `entity_behavior.cu` is included in `modular_rpn_kernel.cu` and BH_* cases added:

```bash
nvcc --ptx -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

**Validation gate** — create `tests/test_sovereign_entity_surface.py`:

```python
def test_entity_star_is_meaning_centric():
    """Entity stars MUST be MeaningCentricStar with meaning_class == 'entity'."""
    from knowledge3d.cranium.sovereign_entity_bootstrap import build_entity_stars
    stars = build_entity_stars()
    assert all(s.meaning_class == "entity" or s.meaning_class == "blackboard" for s in stars)
    trm = next(s for s in stars if s.star_id == "entity:trm:primary")
    assert trm.behavior_rpn is not None
    assert trm.visual_rpn is not None  # must have a body
    assert trm.house_position[1] > 0   # must be above ground

def test_entity_bootstrap_ingestion():
    """Entity stars can be serialized via to_galaxy_entry()."""
    from knowledge3d.cranium.sovereign_entity_bootstrap import build_entity_stars
    for star in build_entity_stars():
        entry = star.to_galaxy_entry(galaxy_name="Reality")
        assert entry["id"]
        assert entry["metadata"]["meaning_star"]["behavior_rpn"] is not None or \
               star.meaning_class == "blackboard"

def test_bh_opcode_cases_in_ptx():
    """PTX must contain BH_PERCEIVE and BH_SEEK dispatch cases."""
    import subprocess, re
    ptx = open("knowledge3d/cranium/ptx/modular_rpn_kernel.ptx").read()
    assert "0x180" in ptx or "bh_perceive" in ptx.lower(), "BH_PERCEIVE missing from PTX"
    assert "0x181" in ptx or "bh_seek" in ptx.lower(), "BH_SEEK missing from PTX"
```

---

## What NOT to Do in Step 3

- Do NOT create a standalone `entity_star.h` with a separate data type that bypasses `MeaningCentricStar`
- Do NOT store avatar bodies as raw structs — they live in the House as MeaningCentricStar visual_rpn programs
- Do NOT add entity_hot_path bootstrap data to the Galaxy as raw bytes — use `to_galaxy_entry()` 
- Do NOT implement BH_GOAP_PLAN or BH_PATHFIND yet — these require LED-A* pool wiring (Step 5)
- Do NOT update `trm_launcher.py` yet — Step 5 only

## After Step 3 passes

Report back for Step 4 (2D Physics) directions. Step 4 is architecturally self-contained and does not require vocabulary reading beyond what is already known.
