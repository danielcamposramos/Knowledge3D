# Codex Directive: Full Benchmark Run — All Questions, All Suites

**Date:** 2026-03-18
**Goal:** Run the COMPLETE benchmark datasets through the sovereign enriched pipeline. No more small samples — we need the real picture, no matter what the results are.
**Principle:** Same session, accumulative log, sleep-time after. The system keeps everything.

---

## Dataset Sizes

| Suite | Total Available | Source |
|-------|----------------|--------|
| ARC | 120 eval tasks | `/K3D/K3D_llama_cpp/datasets/ARC-AGI-2-main/data/evaluation/` |
| Math | 12,500 train problems | `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl` |
| GSM8K | 1,319 test problems | `/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl` |
| LHE | 2,500 questions | `/K3D/K3D_llama_cpp/datasets/last_humanity_exam/last_humanity_exam.json` |
| MMLU | 14,042 test (57 subjects) | `/K3D/K3D_llama_cpp/datasets/MMLU/data/test/*.csv` |

**Total: ~30,481 questions.**

This WILL take a long time on the sovereign pipeline. That's fine — we need real numbers.

---

## Execution Plan

### Phase 1: Full sovereign run (all suites, one session)

Update `scripts/run_enriched_benchmarks.py` to accept full counts:

```python
BENCHMARK_COUNTS = {
    "arc": 120,       # ALL eval tasks
    "math": 500,      # first 500 (12,500 is too many for one session — start here)
    "gsm8k": 1319,    # ALL test questions
    "lhe": 100,       # first 100 (2,500 full is aspirational — start with 100)
    "mmlu": 14042,    # ALL 57 subjects, all test questions
}
```

**Rationale for limits:**
- ARC 120: full eval set, each task is a grid transform — fast on sovereign pipeline
- Math 500: the dataset has 12,500 but each is a competition-level problem. Start with 500. If it's fast enough, run more.
- GSM8K 1319: full test set, word problems with arithmetic — sovereign pipeline handles these
- LHE 100: multi-hop reasoning, hardest suite — 100 is a meaningful sample
- MMLU 14,042: full test set, 57 subjects — this is the big one. Each question is multiple-choice, relatively fast per question.

### Phase 2: Run order

Run ALL suites sequentially in ONE Python process with the SAME Knowledgeverse instance:

1. **ARC** (120) — fast, visual/grid reasoning
2. **Math** (500) — competition math
3. **GSM8K** (1319) — word problem arithmetic
4. **LHE** (100) — multi-hop reasoning
5. **MMLU** (14042) — broad knowledge across 57 subjects

**Add `--full` flag** to `run_enriched_benchmarks.py`:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py --full
```

When `--full` is passed, use the full counts above instead of the default small counts.

### Phase 3: Progress reporting

Since this will run for a long time, add periodic progress output:

```python
# After each suite completes:
print(f"\n{'='*60}")
print(f"COMPLETED: {suite} {correct}/{total}")
print(f"Running total: {total_correct}/{total_questions}")
print(f"Elapsed: {elapsed:.1f}s")
print(f"{'='*60}\n")
```

Also add per-suite timing so we know where the bottleneck is.

### Phase 4: Crash resilience

The run may take hours. If it crashes midway (GPU segfault, OOM, etc.), we need to be able to RESUME, not restart:

1. **Check `health_log.jsonl` for already-completed suites** before running each suite
2. If a suite's questions are already logged from THIS session (check timestamps), skip it
3. This way, re-running the script after a crash picks up where it left off

Simple approach:
```python
def _suite_already_done(log_path: Path, suite: str, expected_count: int, session_start: float) -> bool:
    """Check if this suite was already completed in the current session."""
    if not log_path.exists():
        return False
    count = 0
    for line in log_path.open("r"):
        try:
            row = json.loads(line)
            if row.get("suite") == suite and row.get("timestamp", 0) >= session_start:
                count += 1
        except json.JSONDecodeError:
            continue
    return count >= expected_count
```

---

## What to Modify

### `scripts/run_enriched_benchmarks.py`

1. Add `--full` CLI flag
2. Update `BENCHMARK_COUNTS` for full run (use the counts above)
3. Add per-suite progress reporting with timing
4. Add crash-resume logic (skip suites already in health log from this session)
5. Keep single-session Knowledgeverse — NO reset between suites
6. Sleep-time consolidation AFTER all suites complete

### `scripts/ingest_meaning_layer.py`

When `--full` is passed, the benchmark keyword set should expand to cover ALL questions (not just the small sample). Update `collect_benchmark_keywords()` to accept the full counts.

**Note:** With 14,042 MMLU questions, the keyword set will be much larger. The H19 filter may select more than 2,000 stars. That's OK — raise the cap to 5,000 for the full run. The dedup/stopword/quality filters still apply.

```python
if args.full:
    max_stars = 5000
    benchmark_counts = FULL_BENCHMARK_COUNTS
else:
    max_stars = 2000
    benchmark_counts = DEFAULT_COUNTS
```

---

## Environment

- **k3d-cranium** env (GPU access)
- `export CUDA_VISIBLE_DEVICES=0`
- ONE GPU, sequential everything
- Health log: `/K3D/Knowledge3D.local/logs/health_log.jsonl` (accumulative)
- Sleep-time journal: `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`

## Timing Estimate

| Suite | Questions | Est. per question | Est. total |
|-------|-----------|-------------------|------------|
| ARC | 120 | ~1-2s | ~2-4 min |
| Math | 500 | ~1-2s | ~8-16 min |
| GSM8K | 1319 | ~1-2s | ~20-40 min |
| LHE | 100 | ~2-3s | ~3-5 min |
| MMLU | 14042 | ~0.5-1s | ~2-4 hours |

**Total: roughly 3-5 hours.** MMLU dominates. This is expected.

If any single suite takes more than 2x the estimate, report the per-question timing and continue. Don't abort — we want all the data.

---

## Success Criteria

1. ALL suites run to completion (or as far as they get before crash)
2. Real per-suite scores reported with N questions each
3. Per-suite timing reported
4. Health log grows with all results (accumulative)
5. Sleep-time consolidation runs after completion
6. Results are HONEST — no cherry-picking, no small-sample illusions
7. Crash-resume works if needed

## Report Format

```
=== FULL BENCHMARK RESULTS (Enriched Galaxy) ===

Galaxy State:
  Foundation stars: {N}
  H19 meaning stars: {N} (Math: {N}, Reality: {N}, Drawing: {N}, Language: {N})
  B3 proceduralized: {N}

Suite Results:
  ARC:   {correct}/{total}  ({pct}%)  [{elapsed}s, {per_q}s/q]
  Math:  {correct}/{total}  ({pct}%)  [{elapsed}s, {per_q}s/q]
  GSM8K: {correct}/{total}  ({pct}%)  [{elapsed}s, {per_q}s/q]
  LHE:   {correct}/{total}  ({pct}%)  [{elapsed}s, {per_q}s/q]
  MMLU:  {correct}/{total}  ({pct}%)  [{elapsed}s, {per_q}s/q]

Combined: {total_correct}/{total_questions} ({pct}%)
Elapsed: {total_time}

Sleep-Time:
  Stage A: {summary}
  Stage B: {specialists_updated}

MMLU Breakdown (top 10 subjects):
  {subject}: {correct}/{total} ({pct}%)
  ...
```

The MMLU per-subject breakdown is critical — it shows which domains the Galaxy helps vs. doesn't.
