# Codex Directive: Phase H.0 — Meaning-Centric Star Schema Implementation

**Date**: March 16, 2026
**Priority**: PRIMARY (supersedes benchmark expansion)
**Spec**: `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`
**Roadmap**: `TEMP/CLAUDE_HOUSE_FIRST_ROADMAP_03.16.2026.md`

---

## Context

We're pivoting from benchmark-parameter tuning to building the actual knowledge system. The composed head pipeline WORKS. What's missing is CONTENT — meaning-centric knowledge that lives in the House and loads into Galaxy on demand.

This directive implements the **star schema** — the atomic unit of knowledge in K3D.

---

## What to Build

### 1. MeaningCentricStar Dataclass

**File**: `knowledge3d/knowledgeverse/meaning_star.py` (NEW)

Extend the existing Galaxy entry format into a proper meaning-centric star:

```python
@dataclass
class MeaningCentricStar:
    """The atomic unit of knowledge in K3D.

    A star represents a CONCEPT, not a word.
    Meaning is the center; language forms are references.
    """
    # Identity
    star_id: str                          # ContentHash of meaning_rpn
    meaning_class: str                    # concept | relation | action | property | meta

    # Meaning (THE CENTER — Layer 2)
    meaning_rpn: str                      # RPN program defining WHAT this concept IS
    domain: str                           # House location hint: "Library/Biology/Mammalia"
    taxonomy_refs: List[str]              # Parent/child star_ids in ontology

    # Form (Layer 1 — Language References, NOT copies)
    surface_forms: Dict[str, SurfaceForm] # ISO 639-1 → {word_ref, char_refs}

    # Visual (Drawing Galaxy)
    visual_rpn: Optional[str]             # Procedural drawing program
    visual_refs: List[str]                # Drawing Galaxy star_ids

    # Audio (Audio Galaxy)
    audio_rpn: Optional[str]              # Procedural sound program
    pronunciations: Dict[str, str]        # ISO 639-1 → Audio Galaxy star_id

    # Behavior (Reality Enabler)
    behavior_rpn: Optional[str]           # How it BEHAVES (physics, biology)
    reality_refs: List[str]               # Reality Galaxy star_ids

    # Rules (Layer 3) + Meta (Layer 4)
    grammar_refs: List[str]               # Grammar Galaxy star_ids
    meta_refs: List[str]                  # Meta-rule star_ids

    # Spatial (House Location)
    house_position: Tuple[float, float, float]  # 3D coordinates in House

    # Ternary State
    confidence: int                       # +1 established, 0 uncertain, -1 contested
    polarity: int                         # +1 affirming, 0 neutral, -1 negating

    # Embeddings (Matryoshka LOD)
    embedding_64: Optional[np.ndarray]    # Coarse search
    embedding_128: Optional[np.ndarray]   # Structural
    embedding_512: Optional[np.ndarray]   # Fine-grained
    embedding_2048: Optional[np.ndarray]  # Full fidelity

    # Composition
    component_refs: List[str]             # What this star is MADE OF
    composite_of: List[str]               # What includes this star


@dataclass
class SurfaceForm:
    """Language-specific surface form for a concept."""
    word_ref: str                         # Word Galaxy star_id
    char_refs: List[str]                  # Character Galaxy star_ids
```

### 2. Content-Addressed star_id

**In same file**: `meaning_star.py`

```python
import hashlib

def compute_star_id(meaning_rpn: str, meaning_class: str, domain: str) -> str:
    """Content-addressed identity: same concept = same ID everywhere."""
    content = f"{meaning_rpn}|{meaning_class}|{domain}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 3. Semantic Gravity Ternary Force (Galaxy Working Memory, NOT House)

**File**: `knowledge3d/knowledgeverse/semantic_gravity.py` (NEW)

**CRITICAL**: Semantic gravity operates in GALAXY (working memory) during reasoning,
NOT in the House. The House has intentional physical organization (librarian placement).
Galaxy uses semantic gravity to organize loaded stars for efficient reasoning.

```python
def ternary_semantic_force(star_a: MeaningCentricStar, star_b: MeaningCentricStar) -> int:
    """Compute ternary semantic operator T(a, b) for Galaxy organization.

    Returns: +1 (attract/affinity), 0 (neutral/unknown), -1 (repel/contradiction)

    Used in Galaxy working memory to organize loaded stars during reasoning.
    NOT used for House placement (House uses deliberate librarian placement).

    Uses embedding cosine similarity at tier_128 as initial heuristic.
    Future: RPN-based TCOMP execution.
    """
    ...

def meaning_mass(star: MeaningCentricStar) -> float:
    """Compute semantic 'mass' — richness/connectedness of a star."""
    mass = len(star.surface_forms)
    mass += len(star.visual_refs)
    mass += len(star.reality_refs)
    mass += len(star.component_refs)
    mass += len(star.composite_of)
    mass += len(star.grammar_refs)
    if star.audio_rpn:
        mass += 1
    if star.behavior_rpn:
        mass += 1
    return float(mass)

def semantic_gravity_force(star_a, star_b, distance: float) -> Tuple[float, float, float]:
    """Compute force vector between two stars IN GALAXY working memory.

    F = T(a,b) * M(a) * M(b) / d^2
    Direction: from a toward b (if attract) or away (if repel)
    """
    ...

def gravity_tick(stars: List[MeaningCentricStar], dt: float = 0.01, damping: float = 0.95):
    """One tick of semantic gravity — stars drift toward equilibrium.

    Called during sleep-time consolidation, NOT during inference.
    """
    ...
```

### 4. Seed Stars (Anchor Concepts)

**File**: `knowledge3d/knowledgeverse/seed_stars.py` (NEW)

Create 10 heavily-connected anchor stars that form gravitational wells:

```python
SEED_STARS = [
    # Domain nuclei — these are HEAVY (many refs) and define room centers
    "concept_mathematics",      # Library/Mathematics center
    "concept_physics",          # Library/Physics center
    "concept_chemistry",        # Library/Chemistry center
    "concept_biology",          # Library/Biology center
    "concept_language",         # Library/Languages center
    "concept_tool",             # Workshop center
    "concept_growth",           # Garden center
    "concept_visual_art",       # Gallery/Visual center
    "concept_sound",            # Gallery/Audio center
    "concept_self_reflection",  # Garden/Observatory center
]
```

Each seed star should have:
- `meaning_rpn` defining what the domain IS
- `surface_forms` in at least 3 languages (en, pt, ja)
- `visual_rpn` (even a simple geometric symbol)
- `taxonomy_refs` connecting them to each other
- High `meaning_mass` (many component_refs to existing Galaxy entries)

### 5. Integration with Existing Galaxy

**File**: Modify `knowledge3d/knowledgeverse/galaxy_manager.py`

Add ability to:
- Store MeaningCentricStar entries in existing Galaxy table
- Convert between Galaxy entry format and MeaningCentricStar
- Content-addressed deduplication (same star_id → same entry)

**File**: Modify `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py`

- Wrap existing number entries as MeaningCentricStar (number 5 = concept with meaning_rpn "5 STORE_value", surface_forms {"en": "five", "pt": "cinco", "ja": "五"})

---

## Constraints

1. **Sovereignty**: Star schema is a data definition + ingestion-path code. NO changes to hot-path inference.
2. **Backward compatibility**: Existing Galaxy entries continue to work. MeaningCentricStar is an enhancement layer, not a replacement.
3. **No benchmark regression**: ARC 10/10, Math 20/20, GSM8K 10/10 must hold.
4. **Ternary fields**: confidence and polarity are int {-1, 0, +1}, not float.
5. **Content-addressed**: star_id is deterministic from meaning_rpn, not random UUID.

---

## Tests

**File**: `tests/test_meaning_star.py` (NEW)

1. `test_star_creation_and_id_determinism` — same meaning_rpn → same star_id
2. `test_surface_forms_are_references` — surface forms point to existing Galaxy entries, not copies
3. `test_meaning_mass_computation` — mass increases with refs
4. `test_ternary_force_attract_repel_neutral` — cat↔mammal = +1, alive↔dead = -1, cat↔integer = 0
5. `test_gravity_tick_clusters_related` — after N ticks, related stars are closer than unrelated
6. `test_seed_stars_have_required_fields` — all 10 seeds have meaning_rpn, surface_forms, visual_rpn
7. `test_existing_galaxy_compatibility` — MeaningCentricStar can be stored/loaded from Galaxy table

---

## Success Criteria

- [ ] MeaningCentricStar dataclass exists with all fields from spec §2.1
- [ ] star_id is content-addressed (deterministic)
- [ ] 10 seed stars created with 3+ languages each
- [ ] Semantic gravity computes ternary force between any two stars
- [ ] gravity_tick moves related stars closer, contradictory stars apart
- [ ] Existing Galaxy entries unaffected (backward compatible)
- [ ] All benchmark baselines hold
- [ ] 7/7 tests pass

---

## Order of Work

1. Create `meaning_star.py` with dataclass + star_id computation
2. Create `semantic_gravity.py` with force functions
3. Create `seed_stars.py` with 10 anchor concepts
4. Write tests
5. Integration with galaxy_manager.py (store/load)
6. Wrap existing number entries as MeaningCentricStar
7. Run full benchmark regression

---

## References

- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` — full spec
- `TEMP/CLAUDE_HOUSE_FIRST_ROADMAP_03.16.2026.md` — roadmap context
- `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md` §3 — dual-program stars (existing pattern)
- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` — 4-layer architecture
- `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` — existing Galaxy population
- `knowledge3d/cranium/reality_nodes.py` — existing reality node pattern
