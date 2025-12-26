# Phase 2 Complete: Re-Ingestion with Sovereign Knowledge Articulator

**Date:** December 18, 2025  
**Implementer:** Codex (GPT‑5.2)

## Output Location

Re-ingested Book Galaxies written to:

- `/K3D/Knowledge3D.local/galaxies/books_v2/`

Baseline (pre-Phase-2) Book Galaxies were under:

- `/K3D/Knowledge3D.local/galaxies/books/`

## Summary Results

Key change in Phase 2: `artifacts.jsonl` is now **semantic-block only** (theorem/definition/lemma/corollary/proposition/example/exercise). Generic equation lines are kept in `templates.jsonl` (to avoid drowning “articulated” artifacts in raw formulas).

| Book (book_id) | Baseline Pages | Baseline Templates | v2 Pages | v2 Templates | v2 Artifacts | v2 Size |
|---|---:|---:|---:|---:|---:|---:|
| Linear Algebra Done Right (`la_done_right`) | 200 | 81 | 353 | 924 | 43 | 27M |
| Advanced Calculus (`advanced_calculus`) | 200 | 1820 | 593 | 5305 | 163 | 64M |
| Discrete Math (`dmoi3`) | 200 | 0 | 413 | 1661 | 60 | 32M |
| Transition v104 (`transition_v104`) | 260 | 0 | 291 | 2120 | 235 | 27M |
| Area & Volume (`areavol`) | 2 | 4 | 2 | 3 | 0 | 240K |
| Number Sets (`numbersets`) | 2 | 9 | 2 | 9 | 0 | 216K |
| Physical Quantities (`physquantities`) | 2 | 32 | 2 | 32 | 0 | 200K |
| Math Gems (`mathgems`) | 2 | 0 | 2 | 0 | 0 | 168K |
| **TOTAL** | — | **1946** | — | **10054** | **501** | **~182M** |

Notes:
- Baseline “Pages” for large PDFs were previously capped/partial (200 pages). Phase 2 ingests the full PDF text stream via `pdftotext`, hence larger page counts.
- `MathGems.pdf` is mostly a poster-style page of statements with few/no `lhs = rhs` lines and no theorem/definition headings, so it yields no templates/artifacts.

## Quality Metrics (v2)

Computed over all `artifacts.jsonl` in `books_v2`:

- Total articulated artifacts: **501**
- Artifacts with `conditions`: **401 (80.04%)**
- Artifacts with `symbol_bindings`: **435 (86.83%)**
- Artifacts with executable `rpn` / `conclusion_rpn`: **438 (87.43%)**

Per-book (artifacts only):
- `advanced_calculus`: 163 artifacts; 86.5% with conditions; 88.3% with bindings; 88.3% with RPN
- `dmoi3`: 60 artifacts; 80.0% with conditions; 100% with bindings; 100% with RPN
- `la_done_right`: 43 artifacts; 88.4% with conditions; 100% with bindings; 100% with RPN
- `transition_v104`: 235 artifacts; 74.0% with conditions; 80.0% with bindings; 81.3% with RPN

## Sample Artifacts (v2)

### Advanced Calculus (definition)

```json
{
  "book_id": "advanced_calculus",
  "artifact_type": "definition",
  "page_number": 35,
  "name": "Let V be a set, and let there be given a mapping -< a, fl >- .a",
  "conditions": [
    "Let V be a set, and let there be given a mapping ...",
    "=   + fl = fl + a for all a, fl, I' E V"
  ],
  "conclusion": "+ (fl + 1') = (a + tJ) + I'",
  "rpn": "a x + i y +"
}
```

### Linear Algebra Done Right (example)

```json
{
  "book_id": "la_done_right",
  "artifact_type": "example",
  "page_number": 20,
  "name": "Show that ˛ˇ D ˇ˛ for all ˛; ˇ; \u0002 2 C.",
  "conditions": [
    "Example Show that ˛ˇ D ˇ˛ for all ˛",
    "Suppose ˛ D a C bi and ˇ D c C d i, where a"
  ],
  "conclusion": "D .ac = bd / C .ad C bc/i",
  "rpn": "x c y c z / i /"
}
```

### DmoI3 (example / theorem usage cue)

```json
{
  "book_id": "dmoi3",
  "artifact_type": "example",
  "page_number": 26,
  "name": "ifa 2 + b 2 = c 2 ,",
  "conditions": ["triangle is right-angled"],
  "conclusion": "example, the Pythagorean theorem has a true converse: ifa 2 + b 2 = c 2 ,",
  "rpn": "x y z a b c a d e f g 2 b 2 +"
}
```

### Transition v104 (exercise / interval constraints)

```json
{
  "book_id": "transition_v104",
  "artifact_type": "exercise",
  "page_number": 22,
  "name": "Let a, b ∈ R with a < b. We define four interval notations, as follows:",
  "conditions": ["Let a, b ∈ R with a < b"],
  "rpn": "x 3 y 7 z y q a x 7 y 9 z b r a x 3 y 8 z c d e -"
}
```

## Issues Encountered / Fixes Applied

1. **`pdftotext` control characters breaking equality detection**
   - Observed `\x03` (ETX) emitted in place of `=` in some PDFs (notably `dmoi3-tablet.pdf`).
   - Fix: normalize `\x03 → =` in ingestion and articulator.

2. **Artifacts polluted by raw formulas**
   - Initially `artifacts.jsonl` was dominated by loose “formula” artifacts (duplicating the role of `templates.jsonl`), which defeats the “conditions-first” goal.
   - Fix: `SovereignKnowledgeArticulator` no longer emits loose `formula` artifacts; only semantic blocks become artifacts.

3. **BasicMath mini-PDFs yield templates but few/no semantic blocks**
   - These PDFs are tiny and mostly formula lists; they still contribute templates, but no (or few) theorem/definition-style blocks for articulation.

## Logs

Ingestion logs were captured to `/tmp/phase2_*_v3.log` (plus earlier iterations).

## Next Steps

Phase 3 (TRM integration) should focus on:
- Using `artifacts.jsonl` for condition-gated application (avoid wrong_computation).
- Keeping `templates.jsonl` as a broad fallback for TTC / numeric instantiation.
- Ranking artifacts by specificity (conditions length/keyword match) before attempting RPN execution.

