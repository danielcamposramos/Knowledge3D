# Codex Directive: Dedup Galaxy — Meaning-Centric Multilingual Stars ONLY

**Date:** 2026-03-18
**Problem:** Adding 9,716 keyword-matched H19 stars REGRESSED MMLU from 12-13/50 → 8/50. The retrieval noise from loosely-matched synsets drowned out the existing foundation knowledge. Duped content that isn't truly meaning-centric and multilingual hurts more than it helps.
**Fix:** Strict quality filter — only ingest stars that are genuinely meaning-centric AND multilingual. Strip everything else.

---

## The Regression Explained

The token-based Galaxy query (`_query_token_implementation` in `galaxy_manager.py`) searches ALL entries by keyword overlap. Adding 9,716 language-focused entries flooded results with generic word definitions (like synset entries for "the", "is", "of") that rank high on token match but add zero reasoning value. The real foundation stars (elements, constants, formulas) got pushed down in the results.

---

## CRITICAL: Enhance Math Galaxy with Multilingual Meaning

**Do NOT exclude math synsets!** The whole point is to ENHANCE existing galaxies with multilingual meaning. Math operations like "addition", "subtraction", "multiply", "solve" exist as synsets with 12-19 languages. These MUST land in the Math galaxy, giving those operations their meaning-centric multilingual identity.

**Route by domain, not into a generic "Language" bucket:**

| Synset meaning domain | Target Galaxy | Example |
|----------------------|---------------|---------|
| Math operation/concept | **Math** | "addition" (12 langs), "multiply" (17 langs), "solve" (19 langs) |
| Physics/science concept | **Reality** | "force", "energy", "gravity" |
| Visual/spatial concept | **Drawing** | "circle", "line", "shape" |
| Grammar/linguistic concept | **Grammar** | "verb", "noun", "syntax" |
| General/other | **Language** | everything else |

There are **146 math-meaning synsets with 5+ languages** in the OMW data. Stars like "solve" (19 langs), "calculate" (17 langs), "multiply" (17 langs), "subtract" (16 langs), "divide" (15 langs), "equation" (7 langs). These are HIGH VALUE for the Math galaxy.

### Domain routing logic

```python
# Math-domain keywords for routing synsets to Math galaxy
_MATH_LEMMAS = {
    "addition", "subtraction", "multiplication", "division", "equation",
    "derivative", "integral", "function", "algebra", "geometry", "calculus",
    "number", "sum", "product", "quotient", "remainder", "fraction", "ratio",
    "exponent", "logarithm", "polynomial", "matrix", "vector", "theorem",
    "proof", "axiom", "hypothesis", "probability", "statistic", "average",
    "mean", "median", "variance", "formula", "computation", "arithmetic",
    "subtract", "multiply", "divide", "calculate", "compute", "solve",
    "equal", "greater", "less", "infinite", "finite", "zero", "prime",
    "square", "cube", "root", "factor", "coefficient", "constant",
    "variable", "graph", "coordinate", "angle", "triangle", "circle",
    "area", "volume", "perimeter", "circumference", "diameter", "radius",
    "symmetry", "proportion", "percentage", "decimal", "integer",
}

_PHYSICS_LEMMAS = {
    "force", "energy", "mass", "velocity", "acceleration", "gravity",
    "momentum", "pressure", "temperature", "wave", "frequency", "light",
    "electric", "magnetic", "atom", "molecule", "element", "compound",
    "reaction", "density", "friction", "inertia", "orbit", "field",
}

_VISUAL_LEMMAS = {
    "line", "circle", "rectangle", "square", "triangle", "shape",
    "color", "pattern", "grid", "pixel", "image", "draw", "sketch",
}

def _route_meaning_star_to_galaxy(star: MeaningCentricStar) -> str:
    """Route synset star to the galaxy where its MEANING belongs."""
    en_surface = star.surface_forms.get("en")
    if en_surface is None:
        return "Language"
    word_ref = str(en_surface.word_ref or "").lower()
    lemma = word_ref.split("_", 1)[1] if "_" in word_ref else word_ref
    if lemma in _MATH_LEMMAS:
        return "Math"
    if lemma in _PHYSICS_LEMMAS:
        return "Reality"
    if lemma in _VISUAL_LEMMAS:
        return "Drawing"
    return "Language"
```

**The existing `target_galaxy_for_star()` that routes everything to "Language" must be updated to use this domain-aware routing for synset stars.**

---

## What to Change in `scripts/ingest_meaning_layer.py`

### 1. Enforce minimum language threshold

Change `select_meaning_layer_stars()` to REQUIRE `min_languages=5` (not 2). Stars with only 2-3 languages aren't proving multilingual semantic gravity — they're just English + Finnish noise.

```python
def select_meaning_layer_stars(
    meaning_stars: list[MeaningCentricStar],
    *,
    benchmark_keywords: set[str],
    full_load: bool = False,
    min_languages: int = 5,  # NEW: enforce multilingual quality
) -> list[MeaningCentricStar]:
    # Step 1: Filter to genuinely multilingual stars
    multilingual = [s for s in meaning_stars if len(s.surface_forms) >= min_languages]

    # Step 2: Filter out stopword-level noise
    multilingual = [s for s in multilingual if not _is_stopword_star(s)]

    # Step 3: Apply keyword filter if not full load
    if not full_load:
        multilingual = [s for s in multilingual if star_matches_keywords(s, benchmark_keywords)]

    return dedup_stars(multilingual)
```

### 2. Add stopword filter

Many synsets are for function words (articles, prepositions, pronouns) that match every benchmark question but add zero domain knowledge. Filter them out:

```python
_STOPWORD_LEMMAS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "that", "this", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "whose",
}

def _is_stopword_star(star: MeaningCentricStar) -> bool:
    """Return True if this star's primary English lemma is a stopword."""
    en_surface = star.surface_forms.get("en")
    if en_surface is None:
        return False
    # Extract the lemma from word_ref (format: "en_word")
    word_ref = str(en_surface.word_ref or "").strip().lower()
    lemma = word_ref.split("_", 1)[1] if "_" in word_ref else word_ref
    return lemma in _STOPWORD_LEMMAS
```

### 3. Deduplicate against existing foundation stars (EXCEPT math operations)

Before ingesting H19 stars, build a set of concepts already covered by foundation stars. Skip synsets that EXACTLY duplicate a foundation entry (e.g., don't add a synset for "carbon" when `element_c` already has richer chemistry data).

**BUT: Do NOT skip math operation synsets even if a foundation entry exists.** The synset for "addition" with 12 language surface forms ADDS VALUE to the Math galaxy by providing the multilingual meaning identity. The existing Math galaxy entries may have the formula but lack the multilingual meaning links.

```python
def _existing_foundation_lemmas(knowledgeverse) -> set[str]:
    """Extract lemmas from foundation stars (elements, constants, units)."""
    lemmas: set[str] = set()
    manager = knowledgeverse.galaxy_manager
    for galaxy_name in list(manager._galaxies.keys()):
        galaxy = manager._galaxies.get(galaxy_name)
        if galaxy is None:
            continue
        for entry in getattr(galaxy, "entries", []):
            entry_id = str(entry.get("id", "")).strip().lower()
            # Only skip for non-math foundation entries
            if any(entry_id.startswith(prefix) for prefix in (
                "element_", "constant_", "unit_", "material_", "script_",
                "numeral_system_", "format_", "standard_size_",
            )):
                name = entry.get("name", entry_id)
                lemmas.update(_tokenize_text(str(name)))
    return lemmas


def _should_skip_as_duplicate(star: MeaningCentricStar, foundation_lemmas: set[str]) -> bool:
    """Skip if star duplicates a foundation entry, UNLESS it's a math operation."""
    en_surface = star.surface_forms.get("en")
    if en_surface is None:
        return False
    word_ref = str(en_surface.word_ref or "").lower()
    lemma = word_ref.split("_", 1)[1] if "_" in word_ref else word_ref
    # NEVER skip math operations — they add multilingual value
    if lemma in _MATH_LEMMAS:
        return False
    # Skip if this lemma is already a foundation star
    return lemma in foundation_lemmas
```

### 4. Cap the total loaded count

Even after quality filtering, cap at a reasonable number. The Galaxy query is O(n) token match — too many entries = slow + noisy. Start with:

- **Max 2,000 meaning stars** (sorted by language count descending — most multilingual first)

This ensures only the highest-quality, most multilingual stars make it in. If scores improve, we can raise the cap later.

```python
# After all filters, sort by multilingual quality and cap
filtered.sort(key=lambda s: len(s.surface_forms), reverse=True)
if max_stars is not None and len(filtered) > max_stars:
    filtered = filtered[:max_stars]
```

---

### 5. Update `ingest_enriched_galaxy()` to route by meaning domain

Replace the current "everything → Language" routing with domain-aware routing:

```python
# In ingest_enriched_galaxy(), change the meaning star loop:
with manager.bulk_disk_sync():
    for star in meaning_selected:
        galaxy_name = _route_meaning_star_to_galaxy(star)  # NOT always "Language"
        status = manager.store_meaning_star(galaxy_name, star)
        bucket = counts.setdefault(galaxy_name, {"inserted": 0, "updated": 0})
        bucket[status] = int(bucket.get(status, 0)) + 1
```

This means the ingestion report should show stars distributed across galaxies:
```
Math: +146 meaning stars (addition, multiply, solve, etc.)
Reality: +XX meaning stars (force, energy, etc.)
Drawing: +XX meaning stars (circle, line, etc.)
Language: +remaining
```

---

## What to Change in `scripts/run_enriched_benchmarks.py`

Update the ingestion call to use the new parameters:

```python
summary = ingest_enriched_galaxy(
    kv,
    meaning_path=...,
    mmlu_path=...,
    gsm8k_path=...,
    full_load=False,
    min_languages=5,    # genuinely multilingual only
    max_stars=2000,      # cap to avoid noise
)
```

---

## Then Re-Run Benchmarks

Same approach as before — ONE session, ALL suites, accumulative log, sleep-time after:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py
```

### Expected improvement

With the noise stripped:
- MMLU should recover to at least 12-13/50 (pre-regression baseline)
- If meaning stars genuinely help, it should go ABOVE 13
- ARC/Math/GSM8K should stay at or near 10/10, 20/20, 10/10

### Key comparison point

Report the loaded star count alongside scores so we can correlate:

```
Loaded meaning stars: {N} (from 117,497 available)
  Min languages: 5
  Stopwords removed: {N}
  Foundation duplicates removed: {N}
  Final after cap: {N}
```

---

## Files to modify

| File | Change |
|------|--------|
| `scripts/ingest_meaning_layer.py` | Add `min_languages`, stopword filter, foundation dedup, cap |
| `scripts/run_enriched_benchmarks.py` | Pass new params to ingestion |

**Do NOT modify** any Galaxy query logic, benchmark health check, or Knowledgeverse internals. The fix is in the INGESTION filter, not the query engine.

---

## Success Criteria

1. Loaded meaning star count drops from 9,716 to ~500-2,000 (quality over quantity)
2. ALL loaded stars have 5+ languages (genuinely multilingual)
3. NO stopword stars (the, is, of, etc.)
4. NO duplication of existing foundation knowledge (except math ops which ADD multilingual value)
5. **Math galaxy ENHANCED** with ~146 multilingual operation meanings (addition, multiply, solve, etc.)
6. Stars routed to correct galaxies by meaning domain (Math, Reality, Drawing, Language)
7. MMLU recovers to at least 12/50 (pre-regression baseline)
8. Other suites stay stable (ARC 10/10, Math 20/20, GSM8K 10/10)
9. Health log appended (accumulative), sleep-time runs after
10. Report includes per-galaxy breakdown of ingested meaning stars
