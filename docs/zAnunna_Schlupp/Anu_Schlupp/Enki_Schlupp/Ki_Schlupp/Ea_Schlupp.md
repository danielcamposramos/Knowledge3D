# Codex Directive: Post-Full-Benchmark Fixes — Cap Removal + Math Dataset

**Date:** 2026-03-19
**Context:** Full benchmark run completed: 3,324/15,601 (21.31%). Architecture proven. Two confirmed bottlenecks to fix.
**Principle:** Knowledge volume is managed by the sovereign pipeline (LOD + Frustum), NOT by Python caps. The Galaxy must be full for the pipeline to work.

---

## Fix 1: Remove H19 Knowledge Cap (CRITICAL)

### The problem

`scripts/ingest_meaning_layer.py` caps H19 meaning stars at `max_stars=5000`. The full run showed:

```
H19 stars available:        117,497
After quality filter:        42,727   (min_languages=5, stopwords removed, foundation dedup)
After cap:                    5,000   ← 37,727 quality stars THROWN AWAY
```

**37,727 quality-filtered, genuinely multilingual stars were discarded.** This is the #1 bottleneck for MMLU coverage.

### The fix

In `scripts/ingest_meaning_layer.py`, find `select_meaning_layer_stars()` and REMOVE any `max_stars` parameter or cap logic:

```python
# REMOVE this pattern wherever it appears:
if max_stars is not None and len(filtered) > max_stars:
    filtered = filtered[:max_stars]

# REMOVE max_stars from function signatures
# REMOVE max_stars from callers in run_enriched_benchmarks.py
```

**Keep these quality filters** (they're valuable):
- `min_languages >= 5` (genuinely multilingual)
- Stopword removal (the, is, of, etc.)
- Foundation dedup (except math operations which ADD multilingual value)
- Domain-aware routing (math→Math, physics→Reality, visual→Drawing)

**Remove ONLY the quantity cap.** Load ALL stars that pass quality filters.

### What to change

| File | Change |
|------|--------|
| `scripts/ingest_meaning_layer.py` | Remove `max_stars` parameter from `select_meaning_layer_stars()`. Remove the `filtered = filtered[:max_stars]` cap. Remove `max_stars` from `ingest_enriched_galaxy()` signature. |
| `scripts/run_enriched_benchmarks.py` | Remove `max_stars=` from the `ingest_enriched_galaxy()` call. |

### Expected result

- Loaded H19 stars: ~42,727 (was 5,000)
- All quality filters still apply
- MMLU coverage should improve significantly (more Galaxy entries = more relevant results per question)

---

## Fix 2: Math Benchmark — Use Real Dataset

### The problem

`MathCompetitionBenchmark._load_problems()` defaults to `dataset_mode="synthetic"` when `dataset_path` is not explicitly passed with `dataset_mode="present"`. The synthetic mode returns only 20 hardcoded guard problems.

The real MATH dataset has **12,500 problems** at:
```
/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl
```

The full run requested 500 math questions but got 20.

### The fix

**Option A (preferred):** In `benchmarks/math_competitions.py`, change `_load_problems()` to default to `dataset_mode="present"` when the dataset file exists on disk:

```python
def _load_problems(self) -> list[dict]:
    """Load problems, preferring real dataset when available."""
    dataset_path = Path("/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl")
    if self._dataset_mode == "synthetic" and dataset_path.is_file():
        # Real data exists — use it instead of synthetic guard set
        self._dataset_mode = "present"
        self._dataset_path = dataset_path
    # ... rest of existing logic
```

**Option B:** In `knowledge3d/tools/benchmark_health_check.py`, when creating `MathCompetitionBenchmark`, explicitly pass `dataset_mode="present"` and the path:

```python
# In the math suite creation, add:
MathCompetitionBenchmark(
    knowledgeverse=kv,
    dataset_path=Path("/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl"),
    dataset_mode="present",
)
```

**Pick whichever is cleaner.** The goal: when the full run requests 500 math questions, it gets 500 real competition problems from the JSONL file.

### What to change

| File | Change |
|------|--------|
| `benchmarks/math_competitions.py` | Auto-detect real dataset in `_load_problems()` |
| OR `knowledge3d/tools/benchmark_health_check.py` | Pass explicit `dataset_mode="present"` + path |

### Expected result

- Math questions: 500 (or however many requested) from real dataset
- Scoring: will likely drop from 100% (the synthetic 20 are guard problems the system was tuned for)
- This is GOOD — we want honest numbers on real competition problems

---

## After Both Fixes: Re-Run Full Benchmark

Once both fixes land, re-run the full benchmark to get the real picture:

```bash
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py --full
```

### Expected changes

| Suite | Previous | Expected direction | Why |
|-------|----------|--------------------|-----|
| ARC | 10/120 | Same or slightly up | Cap removal adds some visual stars |
| Math | 20/20 | Drop to real score | Now testing on real competition problems |
| GSM8K | 30/1319 | Up | More math operation stars in Galaxy |
| LHE | 9/100 | Slightly up | More knowledge to reason over |
| MMLU | 3255/14042 | **UP significantly** | 42,727 stars vs 5,000 = 8.5x more coverage |

The MMLU improvement is the main signal. If it goes from 23% → 30%+ with just the cap removal, that proves knowledge volume = score improvement, and the path to higher scores is more/better proceduralized knowledge.

---

## Do NOT change

- Galaxy query logic (`_query_token_implementation`)
- Composed head pipeline (Morton → LED-A* → Frustum → LOD → Swarm → Halting)
- Sleep-time consolidation
- Benchmark health check evaluation logic
- Domain-aware routing (already working: Math 73+20, Reality 10, Drawing 9, Grammar 17)

The architecture is proven. Only fix the knowledge loading bottleneck.

---

## Environment

- **k3d-cranium** env (GPU access)
- `export CUDA_VISIBLE_DEVICES=0`
- Health log: accumulative (append, don't reset)
- Sleep-time: runs after all suites complete
- Checkpoint: TRM routing state persists between runs

## Success Criteria

1. `max_stars` cap completely removed from ingestion pipeline
2. Quality filters (min_languages, stopword, dedup) still active
3. Math benchmark loads from real dataset (12,500 available)
4. Re-run completes with honest numbers
5. H19 loaded count: ~42,727 (not 5,000)
6. Math questions: 500 (not 20)
7. Results logged accumulative (not fresh)
8. Sleep-time consolidation runs after
