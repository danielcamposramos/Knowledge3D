# Claude Benchmark Rerun Report

**Date:** 2026-03-21
**Generated:** 2026-03-21 America/Sao_Paulo
**Session ID:** `full-90b2ae46bc08`
**Provider:** `sovereign`
**Run State:** completed

## 1. Completion Status

The benchmark rerun is finished.

- no active `run_enriched_benchmarks.py --full`
- no active `benchmark_health_check.py`
- no active `sleep_time_compute`

The rerun already committed sleep-time. I did **not** rerun sleep-time again, so the saved state remains the actual post-benchmark state.

## 2. Final Suite Results

From `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`:

| Suite | Score | Percent | Path | Elapsed | Fresh vs Resumed |
|---|---|---:|---|---:|---|
| ARC | `10/120` | `8.33%` | sovereign | `146.552s` | resumed |
| Math | `0/500` | `0.00%` | sovereign | `1248.281s` | fresh |
| GSM8K | `30/1319` | `2.27%` | sovereign | `3666.888s` | fresh |
| LHE | `8/100` | `8.00%` | sovereign | `231.559s` | fresh |
| MMLU | `3272/14042` | `23.30%` | sovereign | `53540.938s` | fresh |

**Combined**

- `3320/16081`
- `20.65%`

## 3. Delta vs the Prior Saved-State Full Run

Baseline for comparison: `TEMP/FULL_BENCHMARK_SAVED_STATE_REPORT_03.20.2026.md`

| Suite | Previous | Current | Delta |
|---|---|---|---|
| ARC | `10/120` | `10/120` | `0` |
| Math | `0/500` | `0/500` | `0` |
| GSM8K | `29/1319` | `30/1319` | `+1` |
| LHE | `10/100` | `8/100` | `-2` |
| MMLU | `3211/14042` | `3272/14042` | `+61` |
| Combined | `3260/16081` | `3320/16081` | `+60` |

Net effect: the rerun improved the overall score slightly, almost entirely through MMLU (`+61`) plus a tiny GSM8K gain (`+1`), while Math remained at zero and LHE regressed.

## 4. Enriched Galaxy State Used

From `/tmp/k3d_benchmark_rerun_postfix_03.20.2026.log`:

- meaning stars available: `117,497`
- meaning layer selected / loaded: `71,312`
- min-languages filter: `5`
- after stopwords: `75,822`
- foundation duplicates removed: `4,510`
- keyword filter removed: `0`
- proceduralized GSM8K loaded: `10`
- proceduralized MMLU loaded: `10`
- math rules updated: `695`
- total galaxy entries after ingest: `199,755`

Per-galaxy ingest status:

| Galaxy | Inserted | Updated |
|---|---:|---:|
| Language | `70,799` | `0` |
| Math | `556` | `20` |
| Reality | `243` | `0` |
| Drawing | `0` | `152` |
| Grammar | `0` | `118` |

Math-rule catalog summary:

| Type | Count |
|---|---:|
| `formula_fact` | `145` |
| `math_rule` | `145` |
| `rule` | `290` |
| `template_program` | `23` |
| `template_support` | `92` |

Math-type coverage:

- Algebra: `107`
- Counting & Probability: `71`
- Geometry: `175`
- Intermediate Algebra: `77`
- Number Theory: `79`
- Prealgebra: `10`
- Precalculus: `176`

## 5. MMLU Top Subjects

Top subject block from the final log:

| Subject | Score | Percent |
|---|---|---:|
| professional_law | `369/1534` | `24.05%` |
| miscellaneous | `190/783` | `24.27%` |
| moral_scenarios | `155/895` | `17.32%` |
| professional_psychology | `148/612` | `24.18%` |
| high_school_psychology | `126/545` | `23.12%` |
| moral_disputes | `86/346` | `24.86%` |
| prehistory | `82/324` | `25.31%` |
| philosophy | `82/311` | `26.37%` |
| high_school_macroeconomics | `82/390` | `21.03%` |
| elementary_mathematics | `80/378` | `21.16%` |

## 6. Sleep-Time Persistence

Sleep-time committed successfully after the rerun.

Latest commit in `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`:

### Stage A

- total health-log rows: `48,000`
- correct: `10,053`
- incorrect: `37,947`
- neutral: `0`
- per-suite accumulated rows:
  - ARC: `390`
  - GSM8K: `3,987`
  - LHE: `320`
  - Math: `1,080`
  - MMLU: `42,223`

### Stage B

- updated specialist routes: `85,150`
- updated specialists: `chat`, `grammar`, `math`, `visual`
- weights saved to `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`

## 7. Persisted State for the Next Run

Saved artifacts:

- `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_tree.json`
- `/K3D/Knowledge3D.local/checkpoints/trm_specialist_spawner.json`

This means the next benchmark run will start from the post-rerun sleep-time state, not the earlier checkpoint.

## 8. Honest Read

The rerun did **not** fix the Math benchmark failure. The math-zero issue remains real despite:

- `695` math-rule entries being present
- Math galaxy inserts occurring
- meaning-layer load increasing from the prior run

The improvement signal is currently:

- modest MMLU improvement from broader loaded knowledge and updated routing
- marginal GSM8K improvement
- no Math recovery
- small LHE regression

So the system grew and sleep-time persisted correctly, but the root cause for `0/500` Math is still unresolved.

## 9. Evidence Paths

- final run log: `/tmp/k3d_benchmark_rerun_postfix_03.20.2026.log`
- run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- earlier baseline report: `TEMP/FULL_BENCHMARK_SAVED_STATE_REPORT_03.20.2026.md`
