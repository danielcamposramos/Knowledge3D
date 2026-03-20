# Full Benchmark Saved-State Report

**Date:** 2026-03-20
**Generated:** 2026-03-20 10:10:54 -0300
**Session ID:** `full-7602e58cc077`
**Provider:** `sovereign`
**Run State:** completed

## 1. Completion Status

The full enriched benchmark run is finished.

- No active `run_enriched_benchmarks.py`
- No active `benchmark_health_check.py`
- No active `sleep_time_compute` / `sleeptime`

This means the run has already flushed its benchmark state and completed its post-run sleep-time consolidation.

## 2. Final Suite Results

From `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`:

| Suite | Score | Percent | Path | Elapsed |
|------|-------|---------|------|---------|
| ARC | `10/120` | `8.33%` | sovereign | `133.59s` |
| Math | `0/500` | `0.00%` | sovereign | `1258.783s` |
| GSM8K | `29/1319` | `2.20%` | sovereign | `3767.711s` |
| LHE | `10/100` | `10.00%` | sovereign | `219.834s` |
| MMLU | `3211/14042` | `22.87%` | sovereign | `51848.262s` |

**Combined:**
- `3260/16081`
- `20.27%`
- total elapsed: `57229.281s`

## 3. What Changed vs the Earlier Full Run

This run reflects the two post-fullbench fixes:

1. **H19 cap removal actually landed**
   - available meaning stars: `117,497`
   - loaded meaning stars: `42,749`
   - prior capped run had loaded `5,000`

2. **Math used the real dataset path**
   - math suite count: `500`
   - score: `0/500`
   - this is the honest result, replacing the earlier synthetic `20/20` guard-path artifact

## 4. Enriched Galaxy State Used

From the benchmark log `/tmp/k3d_full_benchmark_postfix_20260319.log`:

- benchmark keyword count: `36,488`
- foundation lemma count: `2,415`
- meaning stars available: `117,497`
- meaning stars loaded: `42,749`
- proceduralized GSM8K loaded: `10`
- proceduralized MMLU loaded: `10`
- total galaxy entries: `169,936`

Per-galaxy ingest status:

| Galaxy | Inserted | Updated |
|--------|----------|---------|
| Language | `41,746` | `0` |
| Math | `455` | `20` |
| Reality | `258` | `0` |
| Drawing | `101` | `69` |
| Grammar | `49` | `71` |

Meaning-layer filtering:

| Stage | Count |
|-------|-------|
| available | `117,497` |
| after min_languages | `75,970` |
| after stopwords | `75,822` |
| after foundation dedup | `72,413` |
| after keyword filter | `42,749` |
| after selection | `42,749` |

## 5. MMLU Top Subjects

Top subjects from the saved log block:

| Subject | Score | Percent |
|---------|-------|---------|
| professional_law | `386/1534` | `25.16%` |
| miscellaneous | `191/783` | `24.39%` |
| moral_disputes | `85/346` | `24.57%` |
| professional_psychology | `142/612` | `23.20%` |
| philosophy | `73/311` | `23.47%` |
| prehistory | `82/324` | `25.31%` |
| high_school_psychology | `112/545` | `20.55%` |
| high_school_macroeconomics | `83/390` | `21.28%` |
| elementary_mathematics | `74/378` | `19.58%` |
| moral_scenarios | `156/895` | `17.43%` |

## 6. Sleep-Time Persistence

Sleep-time **did** commit after the benchmark run. Latest saved artifacts were updated on **2026-03-20 around 04:02 -0300**.

Latest commit in `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl` shows:

### Stage A
- total health-log rows seen: `31,920`
- correct: `6,733`
- incorrect: `25,187`
- neutral: `0`
- per-suite accumulated rows:
  - ARC: `270`
  - GSM8K: `2,668`
  - LHE: `220`
  - Math: `580`
  - MMLU: `28,182`

### Stage B
- updated specialist routes: `86,793`
- updated specialists:
  - `chat`
  - `grammar`
  - `math`
  - `unknown`
  - `visual`

### Persisted checkpoint
- `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`

Additional saved checkpoint artifacts updated at the same time:
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_tree.json`
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_spawner.json`
- `/K3D/Knowledge3D.local/checkpoints/execution_quality_tracker.json`

## 7. Saved State for the Next Run

The system is already saved for the next run. No extra save action was needed.

Primary persisted files:

- `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_tree.json`
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_spawner.json`
- `/K3D/Knowledge3D.local/checkpoints/execution_quality_tracker.json`

Because sleep-time already committed successfully, the next run should start from the strengthened routing state, not from the older checkpoint.

## 8. Notes for Claude

- The post-fix run is complete and persisted.
- The H19 cap removal took effect: `42,749` meaning stars loaded.
- The Math benchmark fix took effect: real `500`-question path, score `0/500`.
- Combined score is now `3260/16081 = 20.27%`.
- Sleep-time committed afterward and updated `86,793` specialist routes.
- Do **not** rerun sleep-time just to "save" again unless a new benchmark/query cycle has been added, because that would mutate the accumulative state a second time.
