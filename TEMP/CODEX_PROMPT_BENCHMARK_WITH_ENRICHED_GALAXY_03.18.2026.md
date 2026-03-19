# Codex Directive: Sovereign Benchmark Run on Enriched Galaxy + Sleep-Time Consolidation

**Date:** 2026-03-18
**Phase:** B3+ (benchmark health check on enriched knowledge base)
**Principle:** This is NOT a fresh start. The system ACCUMULATES knowledge. H19 meaning layer (117K stars) and B3 proceduralized stars (20 entries) are NEW knowledge that the Galaxy must absorb. Then benchmarks run as NATURAL QUESTIONS the always-on system answers. Then sleep-time consolidation strengthens the paths. Cycle: Ingest → Query → Log → Sleep → Consolidate → Stronger Galaxy.

---

## Context: What Changed Since Last Benchmark Run

| Before | After |
|--------|-------|
| ~367 foundation stars (elements, constants, units, materials) | +117,497 meaning layer stars (H19 OMW synsets, multilingual) |
| No word-level meaning links between languages | Every synset is a meaning star with avg 6.7 languages |
| No proceduralized benchmark knowledge | +10 MMLU stars + 10 GSM8K stars (B3, with symlink refs) |
| Galaxy token search had sparse vocabulary | 784,044 surface forms across 20+ languages now searchable |

The sovereign composed head pipeline (Morton → LED-A* → Frustum → LOD → Nine-Chain Swarm → Halting Gate) is unchanged. What changed is the KNOWLEDGE the pipeline navigates.

---

## Task 1: Ingest H19 + B3 Stars into Galaxy Manager

### Why

The stars are in JSONL files on disk but NOT yet loaded into the GalaxyManager that the benchmark pipeline queries. We need a bootstrap step that loads them.

### What to build

Create `scripts/ingest_meaning_layer.py` — a script that:

1. Initializes the Knowledgeverse (which bootstraps foundational galaxies)
2. Loads `meaning_layer_stars.jsonl` into a "Language" galaxy via `galaxy_manager.store_meaning_star()`
3. Loads `proceduralized_mmlu_val_10.jsonl` and `proceduralized_gsm8k_train_10.jsonl` into appropriate galaxies (e.g., "Math" for GSM8K, domain-based for MMLU)
4. Runs `populate_always_on_foundational_galaxies()` first (existing bootstrap), THEN layers H19/B3 on top
5. Reports: how many stars ingested per galaxy, total Galaxy entry count after

### Key: APPEND, don't replace

The existing foundational galaxies (Drawing, Grammar, Math, Reality, Tool, 3DObjects, Books) stay. H19/B3 stars ADD to them. The `galaxy_manager.store_meaning_star()` method already handles dedup via `upsert_entry()`.

### Performance consideration

117K stars is a LOT. Loading all of them into the in-memory GalaxyManager for a benchmark run may be too heavy. Two approaches (pick the pragmatic one):

**Option A (full load):** Load all 117K into a "Language" galaxy. The token-based query will search through them. May be slow but correct.

**Option B (selective load):** Load only synsets that match benchmark vocabulary. Build a keyword index from the benchmark questions, then filter `meaning_layer_stars.jsonl` to only those that have matching English surface_forms. Much faster, still gives the TRM relevant knowledge.

**Recommendation:** Start with Option B — load H19 stars whose English lemma appears in the benchmark question set. This keeps the Galaxy lean and the benchmark fast. Report both the filtered count and the total available count.

### JSONL reading

Stars were written by `write_stars_jsonl()` which uses `MeaningCentricStar.to_galaxy_entry()`. Read them back:

```python
import json
from pathlib import Path
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

def load_stars_from_jsonl(path: Path) -> list[MeaningCentricStar]:
    stars = []
    for line in path.open("r", encoding="utf-8"):
        entry = json.loads(line)
        star = MeaningCentricStar.from_galaxy_entry(entry)
        if star is not None:
            stars.append(star)
    return stars
```

If `from_galaxy_entry` doesn't exist or returns None, fall back to reading the raw dict and doing `galaxy_manager.add_entry()` directly.

---

## Task 2: Run Sovereign Benchmarks (ALL suites, accumulative)

### Critical: ONE session, ALL suites

Do NOT run each suite in isolation. The Knowledgeverse must be initialized ONCE with the enriched Galaxy, then ALL benchmark suites run in the SAME session. This is the always-on model — the system doesn't reset between questions.

### Execution order

Run ALL suites sequentially in one Python process:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py
```

Create `scripts/run_enriched_benchmarks.py` that:

1. Initializes Knowledgeverse (full bootstrap + H19/B3 ingestion from Task 1)
2. Runs benchmark_health_check for each suite IN ORDER:
   - ARC (10 questions)
   - Math (20 questions)
   - GSM8K (10 questions)
   - LHE (10 questions)
   - MMLU (50 questions)
3. Appends ALL results to the SAME `health_log.jsonl` (accumulative, not fresh)
4. Prints summary table after each suite
5. Prints final combined summary

### Health log path

```
/K3D/Knowledge3D.local/logs/health_log.jsonl
```

This file may already have entries from previous runs — KEEP THEM. The health check appends (line 348 of benchmark_health_check.py: `path.open("a")`). Sleep-time consolidation reads the full log to find patterns.

### Handling the GPU segfault

The pre-existing segfault in `modal_affinity_matrix.py` → `loader.py` at `memcpy_htod` means the full Knowledgeverse may crash. If this happens:

**Fallback plan:** Use the `--provider ollama` path instead for the benchmarks that crash. This uses Ollama as the query backend rather than the sovereign GPU pipeline. It's not sovereign (hot path uses LLM), but it validates that the enriched Galaxy produces better RAG context.

```bash
# If sovereign path segfaults, use Ollama path:
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m knowledge3d.tools.benchmark_health_check \
    --suite mmlu --count 50 --provider ollama \
    --log /K3D/Knowledge3D.local/logs/health_log.jsonl
```

**Report BOTH paths if needed** — sovereign (if it works) AND Ollama (as fallback). Note which path each score came from.

### Previous scores for comparison

| Suite | Phase B Score | Phase B+ Score | Notes |
|-------|--------------|----------------|-------|
| ARC | 10/10 | 10/10 | Composed head |
| Math | 20/20 | 20/20 | Composed head |
| GSM8K | 10/10 | 10/10 | Halting override + slot binding |
| LHE | 7/10 | 7/10 | Multi-hop still weak |
| MMLU | — | 12-13/50 shared | Galaxy coverage was sparse |

**The big question:** Does MMLU improve now that 117K meaning stars and domain knowledge are in the Galaxy? Even a few points up proves the knowledge pipeline works.

---

## Task 3: Sleep-Time Consolidation After Benchmarks

### What sleep-time does

After queries are logged, sleep-time consolidation reads the health log and:
1. **Stage A (Knowledge):** Summarizes health log — which suites, which questions correct/incorrect
2. **Stage B (Logic):** Reads shadow copy events, calls `trm.consolidate_weights_from_events()` to strengthen successful reasoning paths and weaken failed ones

### How to trigger

```python
from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation

sleeptime = SleepTimeConsolidation(
    knowledgeverse=kv,  # the Knowledgeverse instance from Task 2
    health_log_path="/K3D/Knowledge3D.local/logs/health_log.jsonl",
    consume_health_log=True,  # process the entries
)
result = sleeptime.execute()
print(json.dumps(result, indent=2, default=str))
```

Run this AFTER all benchmarks complete, in the SAME Python process (same Knowledgeverse instance with shadow copy events from the run).

### What to report

- Stage A: health log summary (total, correct, incorrect, per-suite)
- Stage B: which specialists updated, weight paths, updated count
- Journal path: verify entries appended to `sleeptime_journal.jsonl`

### If Stage B fails

Stage B requires the TRM navigator and shadow copy to have recorded events during the benchmark run. If those components aren't active (e.g., Ollama fallback path was used), Stage B will return `updated_specialists: []` — that's OK. Stage A (health log analysis) is still valuable.

---

## Task 4: Report & Compare

After all three tasks, produce a summary:

```
=== ENRICHED GALAXY BENCHMARK RESULTS ===

Galaxy State:
  Foundation stars: {count}
  H19 meaning stars loaded: {count}
  B3 proceduralized stars loaded: {count}
  Total Galaxy entries: {count}

Benchmark Scores (enriched):
  ARC:   {score}  (prev: 10/10)
  Math:  {score}  (prev: 20/20)
  GSM8K: {score}  (prev: 10/10)
  LHE:   {score}  (prev: 7/10)
  MMLU:  {score}  (prev: 12-13/50)

Sleep-Time Consolidation:
  Stage A: {summary}
  Stage B: {specialists_updated}

Path Used: sovereign / ollama / mixed
```

---

## Files to create

| File | Purpose |
|------|---------|
| `scripts/ingest_meaning_layer.py` | Load H19/B3 JSONL into GalaxyManager |
| `scripts/run_enriched_benchmarks.py` | Full benchmark suite + sleep-time in one session |

## Data paths

| Path | Content |
|------|---------|
| `/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl` | 117,497 H19 meaning stars |
| `/K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_val_10.jsonl` | 10 B3 MMLU stars |
| `/K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl` | 10 B3 GSM8K stars |
| `/K3D/Knowledge3D.local/logs/health_log.jsonl` | Accumulative benchmark log |
| `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl` | Sleep-time consolidation journal |

## Environment

- **k3d-cranium** for everything (GPU access needed)
- `export CUDA_VISIBLE_DEVICES=0` before running (RTX 3070)
- ONE GPU — sequential everything
- Ollama must be running if sovereign path segfaults

## Success criteria

1. H19/B3 stars ingested into Galaxy (report counts)
2. All 5 benchmark suites run (sovereign or Ollama fallback)
3. Results logged to `health_log.jsonl` (accumulative, not fresh)
4. Sleep-time consolidation executed after benchmarks
5. Comparison table showing improvement (or not) vs previous scores
6. The system KEPT its previous knowledge — nothing was reset or cleared
