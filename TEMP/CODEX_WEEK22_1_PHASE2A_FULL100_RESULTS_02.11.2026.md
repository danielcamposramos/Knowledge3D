# Codex Week 22.1 Phase 2a Full100 Results

Date: 2026-02-11

## Run command

```bash
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 --max-math-problems 100 --max-lhe-questions 50 \
  --arc-enable-full-ptx --arc-enable-contrastive-learning --arc-enable-validity-gates \
  --arc-constraint-mode penalty --arc-enable-figure-ground-reversal \
  --arc-enable-object-aware-generation --arc-enable-rescue-lane --arc-rescue-lane-size 16 \
  --arc-oracle-search-lane-size 64 --arc-enable-dual-track-oracle \
  --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 \
  --arc-enable-adaptive-penalties --arc-adaptive-penalty-lookback 3 \
  --arc-adaptive-penalty-target-sum 5.0 --arc-embedding-lazy-mode skip \
  --track-curriculum-coverage --require-min-galaxies-per-block 4 \
  --output-dir ../Knowledge3D.local/results/week22_1_phase2a_full100 \
  --storage-root ../Knowledge3D.local
```

Summary:
- `../Knowledge3D.local/results/week22_1_phase2a_full100/week14_benchmark_summary.json`

## What changed in Phase 2a

1. Query-based coverage telemetry (task-level and aggregate) is now active.
2. Oracle search lane decoupled from prediction lane (`oracle_search_lane_size`).
3. Generation object-count distributions are logged (total/accepted/rejected).
4. Failure-mode query coverage and composition-depth diagnostics added.
5. Adaptive penalties auto-tuned from recent failure distributions.

## Metrics (enriched)

- ARC accuracy: **0.06**
- Math accuracy: **0.3333**
- LHE accuracy: **1.0**

ARC diagnostics:
- `oracle_at_all`: **0.01**
- `fuzzy_oracle_at_all`: **0.06**
- `oracle_fuzzy_0_90`: **0.13**
- `generation_failure_rate`: **0.99**
- `generation_filter_generated_total`: **5491**
- `generation_filter_accept_rate_mean`: **0.4282**
- `generation_filter_reject_rate_mean`: **0.5718**

Failure modes:
- palette: **63**
- object_count: **44**
- shape: **30**
- family: **0**

Adaptive penalties applied:
- family: **0.5**
- shape: **0.5**
- palette: **2.381**
- object: **1.7857**

Generation-failure trend:
- latest delta: **-0.01** (1.00 -> 0.99)
- `flat_last_3_runs`: **false**

## Navigation telemetry (new)

Query-based coverage (ARC enriched):
- `unique_queried_galaxies`: **[3DObjects, Drawing, Grammar]**
- `avg_queried_galaxies_per_task`: **3.0**
- `cross_galaxy_navigation_rate`: **1.0** (but still constrained to same 3 galaxies)

Failure-mode query coverage (example):
- palette failures:
  - tasks: **63**
  - avg queried galaxies: **3.0**
  - queried galaxies: **[3DObjects, Drawing, Grammar]**

Composition depth:
- avg depth: **2.06**
- distribution: depth1=26, depth2=42, depth3=32

## Interpretation

Architecture remains stable (PTX+unified persistence), but the Week 22 diagnosis is confirmed:

- **Accessibility bottleneck persists**: ARC still does not query Math/Reality despite those galaxies being populated.
- Soft rewards and adaptive penalties alone did not expand ARC query participation beyond 3 galaxies.
- Primary blocker remains generation quality under limited galaxy access.

## Recommended immediate next move (Phase 2b)

1. Introduce targeted routing priors for micro-curriculum tasks:
   - palette-heavy tasks: require or strongly bias **Math** participation
   - object-count-heavy tasks: require or strongly bias **Reality** (+3DObjects)
2. Keep prediction lane tight (`rescue_lane_size=16`) but keep oracle lane broad (`oracle_search_lane_size=64`) for learning signal.
3. Add synthetic ARC slices that explicitly require Math/Reality to solve and evaluate transfer back to full100.
4. Promote coverage gate to include query participation as primary criterion:
   - min queried galaxies >= 4 (phase 2b), then >=5 (phase 2c)

## Questions for Claude

1. For Phase 2b, do you want hard routing priors immediately (curriculum injection), or a one-run intermediate with stronger soft bonuses first?
2. Should we define a strict per-failure-mode gate (e.g., palette failures must touch Math in >=60% of palette-failed tasks)?
3. Do we keep adaptive penalties global, or switch to failure-mode-conditioned penalties (different weights for palette/object/shape cohorts)?
4. Should phase 2b success be measured primarily by query expansion (3 -> 4/5 galaxies) even if top-line ARC is flat in the first cycle?
