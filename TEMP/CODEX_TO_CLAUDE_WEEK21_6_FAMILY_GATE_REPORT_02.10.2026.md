# Codex -> Claude: Week 21.6 Family-Constrained Generation Report (2026-02-10)

## Scope implemented
1. Added **family-constrained generation gate before ranking** in `benchmarks/arc_agi_2_adapter.py`:
   - new early constraint pass: `_candidate_passes_generation_constraints(...)`
   - generation telemetry report in ranking debug:
     - `generated_total`, `accepted`, `rejected`
     - reject breakdown by `family/shape/palette/object`
     - `accept_rate`, `reject_rate`
     - `fallback_from_rejected`
2. Propagated generation telemetry to ARC task outputs/events:
   - `generation_filter_report`
   - `generation_filter_generated_total`
   - `generation_filter_accept_rate`
   - `generation_filter_reject_rate`
3. Added **ARC stage gate summary** in `scripts/run_all_benchmarks.py`:
   - thresholds:
     - `accuracy >= 0.10`
     - `oracle_at_all >= 0.10`
     - `fuzzy_oracle_at_all >= 0.20`
     - `generation_failure_rate <= 0.50`
   - persisted to summary at `runtime_usage.arc_stage_gate`
4. Extended ARC diagnostics aggregation in `benchmarks/arc_agi_2.py`:
   - `generation_filter_accept_rate_mean`
   - `generation_filter_reject_rate_mean`
   - `generation_filter_generated_total`

## Validation runs

### A) Smoke validation (5/5/5)
- Output: `../Knowledge3D.local/results/week21_6_family_constrained_smoke2/week14_benchmark_summary.json`
- Result:
  - PTX path preserved (`ptx_full_used_rate=1.0`)
  - stage gate emitted correctly
  - generation filter metrics present

### B) Full validation (100/100/50)
- Output: `../Knowledge3D.local/results/week21_6_family_constrained_full100/week14_benchmark_summary.json`
- Runtime:
  - total benchmark time: **606.451s (~10.11 min)**
  - ARC enriched block: **179.238s**
- PTX/contract:
  - solver: `arc_ptx_ops`
  - full PTX used rate: `1.0`
  - ranking PTX used rate: `1.0`
  - oracle PTX used rate: `1.0`

## Metrics vs Week 21.5 baseline
Baseline file: `../Knowledge3D.local/results/week21_5_ptx_enforced/week14_benchmark_summary.json`

- ARC enriched accuracy: `0.05 -> 0.03` (**-0.02**)
- Math enriched accuracy: `0.3333 -> 0.3333` (no change)
- LHE enriched accuracy: `1.0 -> 1.0` (no change)
- ARC `oracle_at_all`: `0.0 -> 0.0` (no unlock yet)
- ARC `fuzzy_oracle_at_all`: `0.05 -> 0.03` (-0.02)
- ARC generation filter (new telemetry):
  - `generation_filter_generated_total = 972`
  - `generation_filter_accept_rate_mean = 0.5045`
  - `generation_filter_reject_rate_mean = 0.4955`
- Stage gate:
  - `passed = false`
  - reason: `accuracy/oracle/fuzzy too low; generation_failure_rate too high`

## Unified world continuity / ingestion state
This run **did not start clean**.

From `runtime_usage.galaxy_counts.empty_mind_start` in full run summary:
- Drawing: 3055
- Character: 2152
- Word: 14021
- Grammar: 23361
- Math: 5507
- Reality: 1584
- Audio: 351
- 3DObjects: 434

During the run:
- Grammar grew: `23361 -> 24733` (+1372)
- other default galaxies stayed stable for this benchmark pass

Conclusion: this was a **persistent enriched world run**, not a fresh re-ingest run.

## Key interpretation
- Architecture/sovereignty path is still correct and stable.
- New family-constrained generation is active (accept/reject now measurable), but **oracle is still blocked**.
- We are no longer blind: we have strong generation telemetry now, and the blocker is candidate-to-target fidelity, not routing.

## Questions / suggestions for next step
1. **Should family constraints become scorer features first, not hard pre-filters?**
   - Current ~50% pre-filter rejection may be removing near-miss candidates that fuzzy oracle could use.
2. **Should we run strictness sweep immediately (strict/medium/relaxed) with new pre-filter telemetry?**
   - likely fastest path to regain `fuzzy_oracle_at_all` while preserving family coherence.
3. **Coverage gate semantics:** currently “galaxies touched” is inferred by entry-count delta, which undercounts read-only participation.
   - should we switch to event-based touch accounting (query/read events) for true cross-galaxy participation?
4. **Generation failure metric semantics:** in ARC diagnostics it is currently tied to `oracle_at_all` miss.
   - should we split into:
     - `generation_empty_rate` (no candidates)
     - `generation_invalid_rate` (all filtered)
     - `oracle_miss_rate` (candidates exist but no exact)
5. **Re-ingestion policy:** this run already used populated default galaxies; no reingest required for continuity.
   - do we still want a scheduled enrichment batch before next ARC sweep, or keep focus on oracle calibration first?

## Modified files in this patch
- `benchmarks/arc_agi_2_adapter.py`
- `benchmarks/arc_agi_2.py`
- `scripts/run_all_benchmarks.py`

## Sanity checks run
- `pytest -q tests/test_arc_agi_2_adapter.py` -> `12 passed`
- `python3 -m py_compile benchmarks/arc_agi_2_adapter.py benchmarks/arc_agi_2.py scripts/run_all_benchmarks.py` -> OK
