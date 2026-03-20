# Codex Directive: Run Full Benchmark with Math Fix

**Date:** 2026-03-20
**Priority:** HIGH
**Context:** Math 0/500 fix has been implemented and merged. Need to re-run the full benchmark to validate.

---

## What Changed Since Last Run

Three fixes were implemented for the Math 0/500 problem:

### 1. Answer extraction no longer returns Galaxy entry names (`knowledgeverse.py`)
- New `_explicit_math_answer()` method (line ~2429) blocks `match.get("name")` and `match.get("id")` from being returned as math answers
- Unresolved math now returns `""` instead of junk like "Sum All Values" or "en_service_area"
- New `_math_match_allows_direct_eval()` gates RPN evaluation via `metadata.direct_eval`

### 2. Language↔Math Galaxy symlinks (`ingest_meaning_layer.py`)
- Math-related H19 meaning stars now STAY in Language Galaxy (meaning/navigation layer)
- Additionally get a paired Math Galaxy bridge entry with:
  - `symlink_to`: Language star ID
  - `language_star_ref`: pointer back to Language meaning
  - `math_galaxy_ref`: pointer from Language to Math execution entry
- This follows the Save Information Principle — Language = what it means, Math = how to compute it

### 3. New math rules ingestion (`scripts/ingest_math_rules.py`)
- 695 new Math Galaxy entries across all 7 MATH dataset types
- Categories: `formula_fact`, `symbolic_rpn`, `math_rule`, `template_program`, `template_support`
- Covers: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- Each entry has `rpn_program`, `metadata.tags`, `metadata.math_type`

### 4. Benchmark runner wired to ingest math rules (`run_enriched_benchmarks.py`)
- Now imports and calls `ingest_math_rules()` after `ingest_enriched_galaxy()`
- Math rules summary included in `ingest_summary["math_rules"]`

---

## Problem Observed

When running the benchmark via:
```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py \
    --full --full-load \
    --log /K3D/Knowledge3D.local/logs/health_log.jsonl \
    --journal /K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl
```

The process loaded into memory (3.3GB RSS, 100% CPU) but produced **zero output** for ~16 minutes. No log output, no progress. Likely stuck during the ingestion phase — possibly the meaning layer ingestion with `--full-load` (117,497 stars) combined with the new symlink bridge creation is slow or hitting an infinite loop.

---

## Your Task

### Step 1: Diagnose the hang

Run the ingestion steps separately to find which one hangs:

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
export CUDA_VISIBLE_DEVICES=0
PYTHON=/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python
```

**Test math rules ingestion alone:**
```bash
$PYTHON scripts/ingest_math_rules.py
```
This should produce 695 entries quickly. If it hangs, the problem is in the math rules script.

**Test meaning layer ingestion alone:**
```bash
$PYTHON scripts/ingest_meaning_layer.py --full-load
```
This loads 117,497 stars + creates Math Galaxy symlinks. If it hangs, the problem is in the new symlink bridge creation code. Check `build_language_math_bridge_entry()` and the loop that creates bridge entries.

**Test benchmark without ingestion** (if Galaxy is already populated from a previous run):
```bash
$PYTHON -c "
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
kv = Knowledgeverse(storage_root='/K3D/Knowledge3D.local')
print('Galaxy entries:', kv.galaxy_manager.total_entries())
print('Math entries:', kv.galaxy_manager.galaxy_entry_count('Math'))
"
```

### Step 2: Fix the hang

Most likely causes:
1. **Embedding computation during bridge creation** — if `build_language_math_bridge_entry()` computes embeddings for each of the 73+ math-lemma-matched stars, and the embedding model is slow, this could take a long time
2. **Disk sync thrashing** — `bulk_disk_sync()` context manager might be flushing to disk per-entry instead of batching
3. **Infinite loop in galaxy routing** — the new symlink logic might match stars that match their own bridge entries, creating a cycle
4. **Sentence-transformers model loading** — loading the embedding model takes ~30-60 seconds on first use, but 16 minutes of silence suggests something deeper

### Step 3: Run the full benchmark

Once the hang is fixed, run the full benchmark:

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py \
    --full --full-load \
    --log /K3D/Knowledge3D.local/logs/health_log.jsonl \
    --journal /K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl \
    2>&1 | tee /tmp/k3d_math_fix_benchmark_03.20.2026.log
```

**Add progress logging** — the script currently produces no output during ingestion. Add `print()` statements at key milestones:
- After meaning layer load: `print(f"Meaning layer: {count} stars loaded")`
- After math rules ingestion: `print(f"Math rules: {count} entries ingested")`
- Before each benchmark suite: `print(f"Starting {suite} benchmark ({count} questions)...")`
- After each suite: `print(f"{suite}: {correct}/{total} ({pct}%)")`

### Step 4: Save evidence

After the benchmark completes:

1. Copy the new log files to the evidence directory:
```bash
DEST="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/TEMP/benchmark_evidence_03.20.2026"
cp /K3D/Knowledge3D.local/logs/health_log.full.run_state.json "$DEST/health_log.full.run_state.post_math_fix.json"
cp /K3D/Knowledge3D.local/logs/health_log.jsonl "$DEST/health_log.post_math_fix.jsonl"
cp /K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl "$DEST/sleeptime_journal.post_math_fix.jsonl"
```

2. Print the math-specific results:
```bash
grep '"suite".*"math"' /K3D/Knowledge3D.local/logs/health_log.jsonl | grep '"correct": true' | wc -l
```

---

## Success Criteria

1. **Ingestion completes** — no hangs, progress visible in stdout
2. **Math > 0/500** — any improvement proves the fix works
3. **No regression on other suites:**
   - ARC: ~10/120 (baseline)
   - GSM8K: ~29/1319 (baseline)
   - LHE: ~10/100 (baseline)
   - MMLU: ~3211/14042 (baseline)
4. **Sleep-time consolidation runs** after benchmark
5. **Evidence files saved** to `TEMP/benchmark_evidence_03.20.2026/`

---

## Environment

- **Python:** `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python`
- **GPU:** RTX 3060 (CUDA_VISIBLE_DEVICES=0)
- **Working dir:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D`
- **Storage:** `/K3D/Knowledge3D.local`
- **Logs:** `/K3D/Knowledge3D.local/logs/`
- **Checkpoints:** `/K3D/Knowledge3D.local/checkpoints/`

---

## Files Modified (review before running)

| File | Status | What Changed |
|------|--------|-------------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Modified | `_explicit_math_answer()`, `_math_match_allows_direct_eval()` |
| `scripts/ingest_meaning_layer.py` | Modified | `build_language_math_bridge_entry()`, symlink creation |
| `scripts/ingest_math_rules.py` | NEW | 695 math rules across 7 MATH types |
| `scripts/run_enriched_benchmarks.py` | Modified | Imports + calls `ingest_math_rules()` |
| `tests/test_math_zero_fix.py` | NEW | 6 unit tests (all passing) |
| `benchmarks/math_competitions.py` | Modified | Minor (from previous session) |

## Known Issue

The existing GPU-env test `tests/test_gpu_math_query.py` has a pre-existing assertion failure:
- GSM8K test expects match id `benchmark_math_gsm8k_0_direct` but selection returns `math_arithmetic_mul_add_2_10_6`
- This is an exact-match assertion issue, NOT caused by this patch
- Do not block the benchmark run on this test failure
