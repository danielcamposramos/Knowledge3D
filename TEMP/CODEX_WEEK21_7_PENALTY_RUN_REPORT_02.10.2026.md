# Week 21.7 Penalty Run Report (PTX-Enforced, Unified Persistence)

**Run timestamp:** 2026-02-10  
**Output dir:** `../Knowledge3D.local/results/week21_7_penalty_scoring`  
**Summary:** `../Knowledge3D.local/results/week21_7_penalty_scoring/week14_benchmark_summary.json`  
**ARC enriched details:** `../Knowledge3D.local/results/week21_7_penalty_scoring/arc_agi_2_enriched.json`  
**Usage telemetry:** `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl` (last line)  

## 1) Core results

- ARC enriched accuracy: **0.05** (5/100)
- Math enriched accuracy: **0.3333**
- LHE enriched accuracy: **1.0**

Compared to Week 21.6 medium (`../Knowledge3D.local/results/week21_6_family_constrained_full100/arc_agi_2_enriched.json`):

- top_1_accuracy: `0.03 -> 0.05` (**+0.02**)
- fuzzy_oracle_at_all: `0.03 -> 0.05` (**+0.02**)
- oracle_fuzzy_0_90: `0.07 -> 0.12` (**+0.05**)
- oracle_fuzzy_0_95: `0.03 -> 0.05` (**+0.02**)
- oracle_at_all: `0.0 -> 0.0` (no unlock yet)
- generation_failure_rate: `1.0 -> 1.0` (still hard blocked)

## 2) Requested KPI pack

### rejected_was_better

From `oracle_diagnostics` (ARC enriched):

- `rejected_was_better_count = 9`
- `rejected_was_better_rate = 0.09`
- `rejected_fuzzy_delta_mean = 0.10175306398805528`

From task-level breakdown (`results[*]`):

- all 9 "rejected was better" cases had `best_rejected_reason = family`

### Fuzzy oracle improvement

- Week 21.6 medium fuzzy oracle at all: **0.03**
- Week 21.7 penalty fuzzy oracle at all: **0.05**
- **Delta: +0.02**

### Score distribution analysis (top candidate components)

From `oracle_diagnostics`:

- `ranking_family_score_mean = 0.7855`
- `ranking_shape_score_mean = 0.9656359493798853`
- `ranking_palette_score_mean = 0.6213730158730159`
- `ranking_object_score_mean = 0.92`

Correct vs incorrect split (from `results[*].ranking_top_components`):

- Correct tasks (n=5):
  - family: `0.74`
  - shape: `0.9606`
  - palette: `0.7151`
  - object: `0.8857`
- Incorrect tasks (n=95):
  - family: `0.7879`
  - shape: `0.9659`
  - palette: `0.6164`
  - object: `0.9218`

Interpretation:

- **Palette is the strongest discriminator** (higher on correct tasks).
- Family/object are not differentiating winners correctly yet.
- Shape is high on both classes (likely too permissive / low informational value in current scorer).

## 3) Persistence / world continuity status

Unified persistence is confirmed (no clean reset world):

- `shared_instance = true`
- same `instance_ids` for empty/enriched
- storage mode uses same root:
  - `../Knowledge3D.local/galaxies_enriched`

This world is **not clean**; it is persistent and pre-populated:

- enriched start counts include:
  - Grammar: 26797
  - Math: 5507
  - Drawing: 3055
  - Character: 2152
  - Word: 14021
  - Reality: 1584
  - Audio: 351
  - 3DObjects: 434

During this run, only Grammar grew:

- Grammar: `26797 -> 27483` (**+686**)
- others unchanged

## 4) Coverage gate status (multi-curriculum)

`require-min-galaxies-per-block=5` soft gate failed:

- touched galaxies union: `['Grammar']`
- empty block pass: false
- enriched block pass: false

So the system is persistent and loaded, but ARC path still writes/reads mostly Grammar in this benchmark flow.

## 5) Suggested next-step questions for Claude

1. Should `family` switch from hard penalty to soft tie-breaker only when `palette_score < 0.7`? (all rejected_was_better were family-rejected).
2. Should we raise palette weight in final score (e.g., from current implicit blend to explicit higher coefficient), since palette best separates correct/incorrect?
3. Should we demote shape contribution (currently near-saturated ~0.96 for both correct/incorrect)?
4. Should we add a second-stage rescue: if top accepted fuzzy < top rejected fuzzy by >0.08, auto-promote rejected candidate into top-k for oracle check?
5. Should Stage gate include `fuzzy_oracle_0_90` target first (already improved to 0.12) before `oracle_at_all` hard gate?
6. For multi-curriculum compliance, should ARC loop explicitly query Reality/Math/3DObjects per task (even when grammar has top matches) to meet coverage gate and improve composition?

## 6) Immediate technical recommendation

- Keep PTX-enforced path and penalty mode.
- Run a targeted ablation on weighting:
  1. family weight down
  2. palette weight up
  3. shape clipped/flattened
- Re-run 100 ARC tasks and compare:
  - rejected_was_better_count
  - fuzzy_oracle_at_all
  - oracle_fuzzy_0_90 / 0.95
  - top_1_accuracy
