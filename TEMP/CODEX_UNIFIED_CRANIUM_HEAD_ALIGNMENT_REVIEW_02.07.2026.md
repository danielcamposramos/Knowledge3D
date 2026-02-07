# Codex Review: Unified Cranium Head Alignment (Vocabulary Audit)

**Date:** February 7, 2026  
**Scope:** Validate `TEMP/UNIFIED_CRANIUM_HEAD_ARCHITECTURE_02.07.2026.md` against `docs/vocabulary/*` contracts and correct missed details in implementation.

## Sources Reviewed

- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md`
- `docs/vocabulary/SOVEREIGN_TRAINING_SPECIFICATION.md`
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`
- `docs/vocabulary/README.md`
- `docs/vocabulary/KNOWLEDGEVERSE_MVP_ROADMAP.md`

## Findings

### 1) Routing authority must be centralized (Router Cartographer contract)

`KNOWLEDGEVERSE_SPECIFICATION.md` defines a unified route flow (`specialist='auto'`) with intent analysis before specialist execution.  
Week 14 benchmark code had route logic duplicated in each benchmark class.

**Correction applied:** centralized route resolution in Knowledgeverse layer and rewired benchmarks to consume it.

### 2) Keep benchmark code as harness, not routing logic owner

Benchmarks should measure, not encode domain policy.  
Hardcoded specialist+galaxy mappings in:

- `benchmarks/arc_agi_2.py`
- `benchmarks/math_competitions.py`
- `benchmarks/last_humanity_exam.py`

were moved to a single router module.

### 3) Preserve sovereignty constraints

Vocabulary specs require PTX-only hot path and fail-fast behavior (no silent fallback pollution).  
This change does not add new non-sovereign dependencies and reduces Python-side benchmark policy logic.

## Implementation Corrections Applied

### New centralized router

- Added `knowledge3d/knowledgeverse/specialist_router.py`
- Introduced:
  - `SpecialistRouter`
  - `SpecialistRoute`
- Provides one routing surface for:
  - explicit specialist
  - `specialist='auto'`
  - domain hints (`math`, `visual`, `physics`, `logic`, `multi`)
  - resolved galaxy lists

### TRM navigator integration

- Updated `knowledge3d/knowledgeverse/trm_navigator.py`
- Added:
  - `route(...)` method
  - `navigate_and_compose(..., specialist='auto', domain_hint=...)`
  - `query(..., domain_hint=...)` route-aware behavior
- Route metadata is now attached to composed results.

### Knowledgeverse API alignment

- Updated `knowledge3d/knowledgeverse/knowledgeverse.py`
- Added `Knowledgeverse.query(...)` as a unified entrypoint using centralized routing.

### Benchmark de-duplication

- Updated `benchmarks/arc_agi_2.py`
- Updated `benchmarks/math_competitions.py`
- Updated `benchmarks/last_humanity_exam.py`
- Removed benchmark-owned specialist/galaxy maps and switched to centralized auto-routing with domain hints.

### Export surface update

- Updated `knowledge3d/knowledgeverse/__init__.py`
- Exported `SpecialistRouter` and `SpecialistRoute`.

### New tests

- Added `tests/test_specialist_router.py`
- Validates:
  - math inference
  - domain hint precedence
  - multi-domain routing to cartographer
  - explicit specialist + galaxy override
  - TRMNavigator route trace contract

## Net Effect

- Single source of truth for specialist routing.
- Benchmark harnesses no longer own domain routing policy.
- Closer alignment with Knowledgeverse vocabulary contracts (`auto` routing + centralized intent-to-specialist mapping).
- Maintains current architecture direction without introducing non-sovereign hot-path dependencies.
