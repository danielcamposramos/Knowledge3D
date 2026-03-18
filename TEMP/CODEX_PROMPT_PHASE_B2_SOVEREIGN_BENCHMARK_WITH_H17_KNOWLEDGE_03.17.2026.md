# Phase B2 — Sovereign Benchmark Re-Run with H17 Universal Knowledge

**Depends on:** H17 (Universal Knowledge Foundation), Phase B+ (existing benchmark infrastructure)
**Modifies:** `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py`
**Runs:** `scripts/run_all_benchmarks.py` (or equivalent)
**Goal:** Ingest H17 foundation stars into Galaxy VRAM, then re-run ALL benchmarks to measure improvement

---

## Objective

The Galaxy had sparse content when Phase B+ benchmarks ran (ARC 10/10, Math 20/20, GSM8K 10/10, LHE 7/10, MMLU 16/50). H17 just added 350+ foundation stars covering the periodic table, physical constants, measurement units, writing systems, numeral systems, materials science, and standard formats.

**Hypothesis:** MMLU should improve significantly — many MMLU questions are about science, measurements, and general knowledge that was previously absent from the Galaxy.

---

## Step 1: Wire H17 Stars into Galaxy Bootstrap

**File:** `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py`

Add H17 foundation stars to `populate_always_on_foundational_galaxies()`:

```python
from knowledge3d.ingestion.universal_knowledge import build_foundation_stars

def populate_always_on_foundational_galaxies(galaxy_manager: Any) -> dict[str, Any]:
    # ... existing population (Grammar, Math, Reality, etc.) ...

    # H17: Universal Knowledge Foundation
    foundation_stars = build_foundation_stars(include_elements=True, include_units=True)
    h17_inserted = 0
    for star in foundation_stars:
        # Route star to appropriate galaxy based on domain
        galaxy_name = _domain_to_galaxy(star.domain)
        status = galaxy_manager.store_meaning_star(galaxy_name, star)
        if status == "inserted":
            h17_inserted += 1

    # ... return stats including h17_inserted ...
```

### Domain-to-Galaxy Routing

```python
def _domain_to_galaxy(domain: str) -> str:
    """Route H17 foundation stars to the correct Galaxy."""
    domain_lower = domain.lower()
    if "physics" in domain_lower or "constant" in domain_lower:
        return "Reality"
    if "chemistry" in domain_lower or "element" in domain_lower or "material" in domain_lower:
        return "Reality"
    if "math" in domain_lower or "numeral" in domain_lower:
        return "Math"
    if "language" in domain_lower or "writing" in domain_lower or "script" in domain_lower:
        return "Character"
    if "tool" in domain_lower or "format" in domain_lower:
        return "Tool"
    if "standard" in domain_lower or "paper" in domain_lower or "book" in domain_lower:
        return "Reality"  # Standards are reality knowledge
    return "Reality"  # Default for general knowledge
```

### Expected Star Distribution

| Galaxy | H17 Stars | Content |
|--------|-----------|---------|
| Reality | ~200+ | 118 elements, 80+ units, 16 constants, 43 sizes, 5 materials |
| Math | ~20+ | 18 numeral systems, conversion rules |
| Character | ~25+ | 25 writing systems |
| Tool | ~40+ | 40 file formats |
| **Total** | **~350+** | Universal foundational knowledge |

---

## Step 2: Verify Ingestion

After wiring, verify stars are stored:

```python
# Quick verification script
from knowledge3d.knowledgeverse import Knowledgeverse

kv = Knowledgeverse(bootstrap_foundational_galaxies=True)

# Check Reality Galaxy has elements
hydrogen = kv.galaxy_manager.load_meaning_star("Reality", "element_h")
assert hydrogen is not None, "Hydrogen not found in Reality Galaxy"

# Check units
metre = kv.galaxy_manager.query("metre length SI unit", galaxies=["Reality"], top_k=5)
assert len(metre) > 0, "No measurement units found"

# Count total
total = sum(len(kv.galaxy_manager.list_entries(g)) for g in kv.galaxy_manager.list_galaxies())
print(f"Total Galaxy entries: {total}")
```

---

## Step 3: Re-Run All Benchmarks

Use the existing benchmark harness:

```bash
# Activate GPU environment
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

# Run all benchmarks
python scripts/run_all_benchmarks.py \
  --suites arc_agi_2,math_competitions,gsm8k,last_humanity_exam,mmlu \
  --log ../Knowledge3D.local/logs/health_log_b2.jsonl
```

Or via daemon (if running):
```bash
# Start daemon
python -m knowledge3d.daemon.main --port 7777 &

# Run benchmark senders
python benchmarks/mmlu_sender.py --host 127.0.0.1 --port 7777 --count 50
python benchmarks/arc_sender.py --host 127.0.0.1 --port 7777 --count 10
python benchmarks/math_sender.py --host 127.0.0.1 --port 7777 --count 20
python benchmarks/lhe_sender.py --host 127.0.0.1 --port 7777 --count 10
```

### Benchmark Expectations

| Suite | Phase B+ Score | Expected B2 Score | Rationale |
|-------|---------------|-------------------|-----------|
| ARC | 10/10 | 10/10 | Visual reasoning — H17 doesn't directly add visual patterns |
| Math | 20/20 | 20/20 | Already perfect — should hold |
| GSM8K | 10/10 | 10/10 | Word problems — should hold, possibly improve on unit questions |
| LHE | 7/10 | 8-9/10 | Multi-hop — more facts available for cross-domain reasoning |
| MMLU | 16/50 | 25-35/50 | **Primary target** — science, chemistry, physics questions now have Galaxy entries |

---

## Step 4: Run Sleep-Time Consolidation

After benchmarks complete:

```python
kv.sleeptime.execute()
# This will:
# 1. Summarize health_log_b2.jsonl
# 2. Strengthen paths for correct answers
# 3. Weaken paths for incorrect answers
# 4. Materialize frequent patterns into specialist LoRA adapters
```

---

## Step 5: Report Results

Output a comparison report:

```python
# Compare B+ vs B2 scores
import json

b_plus = {"arc": "10/10", "math": "20/20", "gsm8k": "10/10", "lhe": "7/10", "mmlu": "16/50"}
b2 = {}  # Fill from health_log_b2.jsonl results

for suite in b_plus:
    print(f"{suite}: {b_plus[suite]} → {b2.get(suite, '?')}")
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py` | **MODIFY** — Add H17 `build_foundation_stars()` call + domain routing |
| `scripts/run_all_benchmarks.py` | **NO CHANGE** — existing harness, just re-run |

This is a minimal change — one function call addition + domain router. The benchmark infrastructure already exists.

---

## Success Criteria

1. H17 foundation stars successfully stored in Galaxy (verify count)
2. ARC, Math, GSM8K scores hold (non-regression)
3. LHE improves by at least 1 (more facts for multi-hop)
4. **MMLU improves by at least 5 points** (16/50 → 21+/50) — the primary target
5. Health log written correctly for sleep-time consolidation
6. Sleep-time consolidation runs without errors
7. No sovereignty violations (all answers via GPU composed head pipeline)
