# Codex Directive: H19/B3 Execution Pipeline (4 sequential tasks)

**Date:** 2026-03-18
**Prerequisite:** Dependency cleanup DONE. H19 + B3 code DONE. All tests passing.
**Execute in order:** Task 5 → Task 1 → Task 2 → Task 3

---

## Task 5: Semantic Gravity Correction in H19 Docstrings

### What to fix

The semantic gravity description needs to be precise in the codebase. Add/update docstrings in `multilingual_meanings.py` to reflect this critical correction from Daniel:

**CORRECT:** Semantic gravity (`F = T(s₁,s₂) × M(s₁) × M(s₂) / d²`) operates BETWEEN **different** meaning stars that have **close meanings**. Since each star is already multilingual (all surface forms inside one star), the force doesn't attract surface forms to their star — they're already there. It attracts RELATED CONCEPTS to each other.

Example:
- "water" star and "liquid" star = two different stars that ATTRACT (close meanings)
- "water" (en) and "água" (pt) = NOT two attracting stars — they are the SAME star (same meaning, different surface forms)
- Language is irrelevant to the gravitational force — only meaning distance matters

### Where to add

1. **Module docstring** at top of `multilingual_meanings.py` — expand the one-liner to include:
   ```python
   """Build meaning-centric stars from Open Multilingual Wordnet synsets.

   Each star represents ONE meaning (synset) with surface_forms from all available
   languages. In Galaxy working memory, semantic gravity operates BETWEEN different
   meaning stars based on proximity of their meanings — e.g. "water" and "liquid"
   attract each other. Language is irrelevant to the force; only meaning distance
   matters. Surface forms within a star (e.g. "water"(en), "água"(pt)) are NOT
   separate gravitational bodies — they are symlink references inside one star.
   """
   ```

2. **`synset_to_star()` docstring** — expand to:
   ```python
   """Convert one synset into a multilingual meaning-centric star.

   One synset = one meaning = one star. All languages are surface_forms inside
   the star. Semantic gravity clusters different but related meaning stars
   together in Galaxy working memory (e.g. "water" near "liquid" near "ice").
   """
   ```

3. **`build_meaning_layer_stars()` docstring** — expand to:
   ```python
   """Return a list of multilingual meaning stars.

   These stars form the meaning layer where semantic gravity operates between
   concepts by meaning proximity, not by language. Each star is already
   multilingual — gravity acts between stars, not within them.
   """
   ```

**Do NOT change any logic or function signatures.** Only docstrings.

---

## Task 1: Full Test Suite Execution

### Run the complete test suite

You already confirmed collection (1395/1409 collected, 0 errors). Now run the full execution:

```bash
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -40
```

**Expected:** All sovereign tests pass, torch-dependent tests skip, ARC test skips.

If there are failures, categorize them:
- **New failures from H19/B3 or dependency cleanup** → FIX these
- **Pre-existing failures unrelated to our changes** → REPORT but don't fix (note the count)

Also run the H19/B3 non-regression slice to confirm it's still clean:
```bash
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -m pytest tests/test_multilingual_meanings.py tests/test_knowledge_proceduralizer.py tests/test_universal_knowledge.py -q
```

---

## Task 2: Live H19 Run — Parse Full OMW Dataset

### Goal

Run the actual OMW parser against the real dataset and produce meaning stars JSONL output.

### Steps

**2a.** Write a small runner script (or use Python one-liner) that:
1. Calls `build_meaning_layer_stars(min_languages=2)` with the real OMW path
2. Calls `meaning_layer_stats()` on the result
3. Writes the stars to `/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl`
4. Prints stats summary

```python
#!/usr/bin/env python3
"""Run H19: parse full OMW dataset into meaning layer stars."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
    build_meaning_layer_stars,
    meaning_layer_stats,
)
from knowledge3d.tools.content_to_stars import write_stars_jsonl

# Parse ALL synsets with 2+ languages
stars = build_meaning_layer_stars(min_languages=2)
stats = meaning_layer_stats(stars)

# Write output
output_path = Path("/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl")
write_stars_jsonl(stars, output_path)

print(json.dumps(stats, indent=2, default=str))
print(f"\nWrote {len(stars)} stars to {output_path}")
```

Save this as `scripts/run_h19_meaning_layer.py` and run with:
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_h19_meaning_layer.py
```

Use `k3d-cranium` env (it has all deps including sentence-transformers).

### Expected output

- ~30,000-80,000 stars (depends on how many synsets have 2+ languages)
- Average 5-10 languages per star
- English should be the most common language
- JSONL file at `/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl`

### Validation

After the run, verify the output:
```bash
wc -l /K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl
head -3 /K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl | python3 -m json.tool | head -30
```

Check that:
- Stars have `star_id` starting with `synset_`
- `meaning_rpn` is in English (starts with `SYNSET`)
- `surface_forms` has multiple languages
- Each surface form has `word_ref` and `char_refs`

Report: total stars, total surface forms, avg languages, top 10 languages, POS distribution.

---

## Task 3: Live B3 Ollama Proceduralization (Small Batch)

### Prerequisites

- Ollama must be running: `curl -s localhost:11434/api/tags | python3 -m json.tool | head -5`
- Models available: `qwen3:8b` and `qwen2.5:32b`
- H19 meaning layer stars must exist (from Task 2) — the proceduralizer uses them for RAG context

### IMPORTANT: ONE GPU

Only ONE model can be loaded at a time. Run MMLU batch first (qwen3:8b), then GSM8K batch (qwen2.5:32b). **NEVER run both simultaneously.**

### Step 3a: Small MMLU val batch (10 entries, fast model)

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m knowledge3d.tools.knowledge_proceduralizer \
    --source mmlu_val \
    --count 10 \
    --subjects abstract_algebra \
    --model qwen3:8b \
    --timeout 120 \
    --output /K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_val_10.jsonl
```

**Expected:** 10 processed, 10 created (or close), ~1-2 min total.

### Validate MMLU output

```bash
wc -l /K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_val_10.jsonl
head -2 /K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_val_10.jsonl | python3 -m json.tool | head -40
```

Check that stars have:
- `star_id` like `mmlu_val_abstract_algebra_0`
- `meaning_rpn` in English
- `meta_refs` containing `source:mmlu_val` and `subject:abstract_algebra`
- Any `star_refs` referencing existing stars (element_*, constant_*, synset_*)

### Step 3b: Small GSM8K batch (10 entries, reasoning model)

**WAIT** for the MMLU batch to finish completely before starting this. The 32b model needs to load, replacing qwen3:8b in VRAM.

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m knowledge3d.tools.knowledge_proceduralizer \
    --source gsm8k_train \
    --count 10 \
    --model qwen2.5:32b \
    --timeout 300 \
    --output /K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl
```

**Expected:** 10 processed, 10 created, ~10-15 min total (32b model is slower).

### Validate GSM8K output

```bash
wc -l /K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl
head -2 /K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl | python3 -m json.tool | head -40
```

Check that stars have:
- `star_id` like `gsm8k_train_0`
- `meaning_rpn` with math/arithmetic content
- `meta_refs` with `source:gsm8k_train` and `subject:arithmetic`
- Any `star_refs` pointing to existing stars

### Step 3c: Report summary

After both batches, report:
1. MMLU: processed/created count, sample star_id, sample meaning_rpn, sample star_refs
2. GSM8K: processed/created count, sample star_id, sample meaning_rpn, sample star_refs
3. Any errors, timeouts, or fallback results (confidence < 0.3)
4. File sizes of both JSONL outputs
5. Whether the symlink principle worked — did the model actually reference existing star_ids?

---

## Environment Notes

- **k3d-trm** env for pytest (Tasks 1, 5)
- **k3d-cranium** env for live runs (Tasks 2, 3) — has sentence-transformers and full stack
- **Ollama** must be running for Task 3: `systemctl --user status ollama` or `pgrep ollama`
- **ONE GPU** — never run two models simultaneously
- All output goes to `/K3D/Knowledge3D.local/galaxies/`

## Files to create/modify

| File | Action |
|------|--------|
| `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py` | Update 3 docstrings (Task 5) |
| `scripts/run_h19_meaning_layer.py` | Create runner script (Task 2) |

**Do NOT modify** any function logic, test files, or other source files.

## Success Criteria

1. **Task 5:** Three docstrings updated, no logic changes, tests still pass
2. **Task 1:** Full `pytest tests/ -q` run with results categorized
3. **Task 2:** `meaning_layer_stars.jsonl` produced with stats reported
4. **Task 3:** Two JSONL files (MMLU + GSM8K) with symlink references to existing stars
