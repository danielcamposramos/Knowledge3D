# Codex to Claude: Reality Engine Step 3 Completion
**Date:** 2026-04-08

## Grounding
Implemented per:
- `TEMP/CODEX_REALITY_ENGINE_STEP3_DIRECTIONS_2026-04-08.md`
- `docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md`
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`

The key architectural correction was preserved:
- entities are canonical `MeaningCentricStar` rows
- `meaning_class="entity"` plus `behavior_rpn` / `visual_rpn`
- `EntityHotPath` is only a compact boot-time GPU projection for the BEHAVIOR_PHASE surface
- no standalone parallel entity data model was introduced

## Implemented

### 1. GPU hot-path projection
Added:
- `knowledge3d/cranium/kernels/entity_hot_path.h`
- `knowledge3d/cranium/kernels/entity_behavior.cu`

`EntityHotPath` is now bound through modular PTX globals:
- `g_entity_hot_path_ptr`
- `g_entity_count`

### 2. BH_* opcode surface in modular kernel
Added BH dispatch to `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`:
- `0x180` `BH_PERCEIVE`
- `0x181` `BH_SEEK`
- `0x182` `BH_FLEE`
- `0x183` `BH_ARRIVE`
- `0x184` `BH_SEPARATE`
- `0x185` `BH_APPLY_FORCE`
- `0x186` `BH_BT_TICK`
- `0x188` `BH_GOAP_PLAN` stub -> pushes failure
- `0x189` `BH_SLEEP_CHECK`
- `0x18A` `BH_BLACKBOARD_READ`
- `0x18B` `BH_BLACKBOARD_WRITE`
- `0x18C` `BH_PATHFIND` stub -> pushes failure

Important practical note:
- the current modular stack is scalar/vector only, so the Step 3 surface stays compact
- `BH_PERCEIVE` currently pushes neighbor count rather than a full 64-bit list address
- this keeps the opcode live and testable without inventing a second stack format

### 3. Canonical entity bootstrap
Added:
- `knowledge3d/cranium/sovereign_entity_bootstrap.py`

This defines:
- `entity:trm:primary`
- `blackboard:faction:0`
- `blackboard:faction:1`
- `blackboard:faction:2`

and the boot-time extractor:
- `build_entity_hot_path_array(galaxy_manager)`

### 4. Bridge bindings
Extended `knowledge3d/cranium/bridges/sovereign_bridges.py` with:
- `bind_entity_behavior_programs()`
- `bind_entity_soa()`

The bridge now uploads:
- compiled behavior blobs
- serialized `EntityHotPath[]`

### 5. Ingestion wiring
Extended:
- `knowledge3d/ingestion/__init__.py`
- `scripts/fundamental_ingest_payloads.py`
- `knowledge3d/tools/ingest_from_manifest.py`

Entities now bootstrap through the same canonical ingestion path as physics:
- `ingest_entity_bootstrap(galaxy_manager)`
- stored in `Reality` via `store_meaning_star(...)`
- `galaxy_ref="House"` preserved on the star payload

### 6. Fused-step source boundary
Extended `knowledge3d/cranium/ptx/trm_step_fused.cu` with:
- `trm_behavior_phase_stub(...)`
- extra kernel params:
  - `entity_hot_path_ptr`
  - `entity_count`
  - `frame_counter`

Per your directions:
- `trm_launcher.py` remains untouched
- full launcher integration remains deferred

## Validation

### PTX
Rebuilt successfully:
```bash
nvcc --ptx -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

### Python surface
`py_compile` passed on:
- entity bootstrap
- ingestion wiring
- bridges
- runtime opcode tables
- Step 3 tests

### Tests
Focused gate:
```text
19 passed in 6.01s
```

Passing files:
- `tests/test_sovereign_entity_surface.py`
- `tests/test_procedural_texture_surface.py`
- `tests/test_sovereign_physics_surface.py`

The new entity tests cover:
- canonical `MeaningCentricStar` entity contract
- bootstrap ingestion
- hot-path array extraction
- BH opcode registration
- BEHAVIOR_PHASE stub presence
- bridge smoke for:
  - `BH_SLEEP_CHECK`
  - `BH_BLACKBOARD_WRITE/READ`
  - `BH_PERCEIVE`
  - `BH_SEEK`
  - `BH_BT_TICK`

## Deferred exactly as directed
- no standalone `EntityStar` type
- no HAnim skeleton builder work
- no `BH_GOAP_PLAN` implementation
- no `BH_PATHFIND` implementation
- no `trm_launcher.py` changes
- no Step 4 / Step 5 work

## Architectural note for Step 5
The main remaining design boundary is still the stack contract:
- compact BH behavior works today on the existing scalar/vector modular stack
- richer behavior outputs that require opaque pointers or structured lists will likely need either:
  - scratch-buffer handle conventions, or
  - a promoted pointer/handle value surface in the modular executor

That is not blocking Step 3, but it is the main constraint to keep in mind when the full BEHAVIOR_PHASE dispatch is promoted.
