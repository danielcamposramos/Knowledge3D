# Codex Prompt: Phase H.0 — Meaning-Centric Star Schema

## Priority Shift

We're moving from benchmark parameter tuning to building the actual knowledge system. The composed head pipeline works (ARC 10/10, Math 20/20, GSM8K 10/10). What's missing is CONTENT — meaning-centric knowledge that lives in the House and loads into Galaxy on demand. Phase H (House Construction) is now primary.

## Your Task

Implement the **Meaning-Centric Star Schema** — the atomic unit of knowledge in K3D. A star represents a CONCEPT, not a word. "Cat" is cat in every language; the meaning is the center, language forms are references.

**Read these first:**
- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` — full architecture spec
- `TEMP/CODEX_STAR_SCHEMA_PHASE_H0_03.16.2026.md` — detailed implementation directive with code sketches

## Deliverables (in order)

### 1. `knowledge3d/knowledgeverse/meaning_star.py` (NEW)

- `MeaningCentricStar` dataclass with all fields from spec §2.1:
  - Identity: `star_id` (content-addressed hash), `meaning_class` (concept/relation/action/property/meta)
  - Meaning center: `meaning_rpn` (RPN program — THE essential field), `domain`, `taxonomy_refs`
  - Form: `surface_forms` dict (ISO 639-1 → word_ref + char_refs — symlinks, NOT copies)
  - Visual/Audio/Behavior: `visual_rpn`, `audio_rpn`, `behavior_rpn` + refs to existing Galaxy entries
  - Rules/Meta: `grammar_refs`, `meta_refs`
  - Spatial: `house_position` (Vec3), `house_room`
  - Ternary: `confidence` {+1,0,−1}, `polarity` {+1,0,−1}
  - Embeddings: Matryoshka tiers {64, 128, 512, 2048}
  - Composition: `component_refs`, `composite_of`
- `SurfaceForm` dataclass: `word_ref` + `char_refs`
- `compute_star_id(meaning_rpn, meaning_class, domain) -> str` — SHA-256 content hash, deterministic

### 2. `knowledge3d/knowledgeverse/semantic_gravity.py` (NEW)

**CRITICAL**: Semantic gravity operates in GALAXY working memory during reasoning, NOT in the House. The House has intentional physical organization (TRM places knowledge like a librarian).

- `ternary_semantic_force(star_a, star_b) -> int` — returns +1 (attract), 0 (neutral), −1 (repel). Use tier_128 cosine similarity as initial heuristic.
- `meaning_mass(star) -> float` — count of references/connections across all modalities
- `semantic_gravity_force(star_a, star_b, distance) -> Tuple[float,float,float]` — F = T(a,b) × M(a) × M(b) / d²
- `gravity_tick(stars, dt, damping)` — one tick of Galaxy working memory organization

### 3. `knowledge3d/knowledgeverse/seed_stars.py` (NEW)

Create 10 anchor stars that define domain centers:
- `concept_mathematics`, `concept_physics`, `concept_chemistry`, `concept_biology`
- `concept_language`, `concept_tool`, `concept_growth`
- `concept_visual_art`, `concept_sound`, `concept_self_reflection`

Each seed star MUST have:
- `meaning_rpn` (RPN program defining what the domain IS)
- `surface_forms` in at least 3 languages (en, pt, ja)
- `visual_rpn` (at minimum a geometric symbol)
- `taxonomy_refs` connecting to other seeds
- High meaning_mass (many component_refs linking to existing Galaxy entries)

### 4. Galaxy Integration

- Modify `knowledge3d/knowledgeverse/galaxy_manager.py`: store/load MeaningCentricStar in existing Galaxy table
- Modify `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py`: wrap existing number entries as MeaningCentricStar (e.g., number 5 = concept with surface_forms {"en": "five", "pt": "cinco", "ja": "五"})
- Content-addressed deduplication: same star_id → same entry

### 5. `tests/test_meaning_star.py` (NEW)

7 tests:
1. `test_star_creation_and_id_determinism` — same meaning_rpn → same star_id every time
2. `test_surface_forms_are_references` — surface forms point to Galaxy entries, not copies
3. `test_meaning_mass_computation` — mass increases correctly with refs
4. `test_ternary_force_attract_repel_neutral` — cat↔mammal = +1, alive↔dead = −1, cat↔integer = 0
5. `test_gravity_tick_clusters_related` — after N ticks, related stars closer than unrelated
6. `test_seed_stars_have_required_fields` — all 10 seeds complete
7. `test_existing_galaxy_compatibility` — MeaningCentricStar stores/loads from Galaxy table

## Constraints

- **NO changes to hot-path inference** — star schema is data definition + ingestion-path
- **Backward compatible** — existing Galaxy entries keep working
- **No benchmark regression** — ARC 10/10, Math 20/20, GSM8K 10/10 must hold
- **Ternary fields** — confidence and polarity are int {-1, 0, +1}, not float
- **Content-addressed** — star_id deterministic from meaning_rpn, not UUID
- **Sovereignty** — semantic_gravity.py is ingestion/sleep-time code, not hot-path

## Key Architecture Points

- **House = intentional physical organization** (librarian placement). Stars are PLACED on shelves, not dropped by gravity.
- **Galaxy = fluid working memory** where semantic gravity organizes loaded stars during reasoning.
- **Same star, multiple House locations via symlinks** (tree leaf in Garden AND book entry in Library)
- **"Semantic gravity cohered by meaning"** (Christoph Dorn) — the force in Galaxy working memory

## Order of Work

1. Create `meaning_star.py` with dataclass + star_id
2. Create `semantic_gravity.py` with force functions
3. Create `seed_stars.py` with 10 anchors
4. Write tests
5. Galaxy integration (galaxy_manager.py, foundational_operations_bootstrap.py)
6. Run full benchmark regression

## Files to Read First

- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` (full spec)
- `TEMP/CODEX_STAR_SCHEMA_PHASE_H0_03.16.2026.md` (detailed directive with code sketches)
- `knowledge3d/knowledgeverse/galaxy_manager.py` (existing Galaxy system)
- `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` (existing Galaxy population)
- `knowledge3d/cranium/reality_nodes.py` (existing dual-program star pattern)
