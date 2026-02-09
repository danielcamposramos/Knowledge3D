# Codex Week 21.3 Oracle Unlock Results

Date: 2026-02-08

## Run Commands

```bash
env PYTHONPATH=. python3 scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --output-dir ../Knowledge3D.local/results/week21_3_oracle_unlock_full \
  --storage-root ../Knowledge3D.local
```

Summary artifact:
- `../Knowledge3D.local/results/week21_3_oracle_unlock_full/week14_benchmark_summary.json`

## Core Metrics

### ARC-AGI 2 (enriched)
- Accuracy: `0.28` (`28/100`)
- Generated patterns total: `686`
- Tasks with generated patterns: `100/100`
- Generation failure rate: `1.0`
- Oracle@3: `0.0`
- Oracle@10: `0.0`
- Oracle@all: `0.0`
- Fuzzy oracle@all: `0.05`
- Fuzzy best-score mean: `0.6250`
- Validity reject-rate mean: `0.4383`
- Ranking change rate: `0.45`
- Ranking failure rate: `0.0`
- Top-1 accuracy: `0.28`

### Pattern source accuracy (ARC enriched)
- `legacy_pipeline`: `9/20 = 0.45`
- `autonomous_generation`: `13/67 = 0.1940`
- `contrastive_anti`: `6/13 = 0.4615`

### Same-run empty vs enriched
- ARC: `0.32 -> 0.28` (`-0.04`)
- Math: `0.00 -> 0.3333` (`+0.3333`)
- LHE: `0.50 -> 1.00` (`+0.50`)

### Historical comparison (runner output)
- ARC: `0.25 -> 0.28` (`+0.03`, IMPROVEMENT)
- Math: `0.3333 -> 0.3333` (MAINTAINED)
- LHE: `1.0 -> 1.0` (MAINTAINED)

## Interpretation

1. Oracle unlock is only partially unlocked.
- Exact oracle remains `0.0` (no exact-correct candidate generated).
- Fuzzy oracle reached `0.05` and fuzzy mean `0.625`, so candidate proximity is improving.

2. Generation volume is no longer the blocker.
- `686` generated patterns over `100` ARC tasks confirms generation is active.
- Quality is the blocker: autonomous source precision is low (`~19.4%`).

3. Ranking is active but not sufficient yet.
- Ranking changes top selection frequently (`45%`) and does not fail to run.
- But top-1 remains `28%`, so current ranking signals are not selecting truly correct candidates often enough.

4. Negative branch is promising.
- `contrastive_anti` source currently has best precision (`46.15%`) in this run.
- This supports ternary contrastive learning and opposite-sign feedback direction.

## Universe Load/Persistence Verification

`Knowledgeverse.ensure_default_galaxies_loaded()` is called in:
- `knowledge3d/knowledgeverse/knowledgeverse.py`
- `benchmarks/arc_agi_2.py`
- `benchmarks/arc_agi_2_adapter.py`

Observed loaded counts (post-run check):

- Storage root `../Knowledge3D.local/galaxies_empty_mind`
  - Drawing `188`, Character `0`, Word `0`, Grammar `1028`, Math `0`, Reality `0`, Audio `0`, 3DObjects `0`
- Storage root `../Knowledge3D.local/galaxies_enriched`
  - Drawing `3055`, Character `0`, Word `0`, Grammar `11905`, Math `507`, Reality `1584`, Audio `0`, 3DObjects `434`

Note: default galaxies are present and loaded; some remain sparsely populated (Character/Word/Audio in this benchmark path).

Global runner wiring update:
- `scripts/run_all_global_benchmarks.py` now accepts and forwards ARC oracle-unlock flags:
  - `--arc-enable-contrastive-learning`
  - `--arc-enable-validity-gates`
  - `--arc-enable-fuzzy-oracle`
  - `--arc-fuzzy-oracle-threshold`

## What Is Still Missing (Focused)

1. Candidate validity consistency is too weak before ranking.
- Need stricter train-pair consistency filters:
  - shape transform family compatibility
  - palette mapping consistency across all train pairs
  - object-count delta consistency across train pairs

2. Ranking needs source-aware priors beyond static weights.
- Given current precision, ranking should downweight raw `autonomous_generation` and upweight `contrastive_anti` until priors improve.
- Integrate ternary quality priors directly in top-k selection (not only metadata update).

3. Fuzzy oracle should be stratified, not single-threshold.
- Add `fuzzy_oracle@{0.80,0.90,0.95}` and report calibration curves.
- This will expose near-miss density and help decide gating thresholds.

4. ARC-specific teacher augmentation should target transformation families, not generic patterns.
- Use Ollama augmentation to emit family tags (`rotation`, `mirror`, `translation`, `copy-fill`, `object-map`) and enforce family agreement during candidate filtering.

## RLWHF Mapping (Teacher/Student, Ternary)

The current bridge in `knowledge3d/training/rlwhf/teacher_student_bridge.py` already supports ternary pooling and contrastive recommendation. To align with your non-binary RLWHF principle and apply reverse-thinking across chains, I recommend:

1. Use teacher signals as 4-axis ternary state for every chain step.
- Axes: `correctness_t`, `honesty_t`, `transfer_t`, `novelty_t`.
- Keep updating `pool_id` (81 pools) per iteration and per task-family.

2. Apply forward/backward/fusion at RLWHF event level (not only query parsing).
- Forward: successful candidate trajectory.
- Backward: failed trajectory -> anti-pattern synthesis.
- Fusion: deduplicated union with ternary uncertainty weighting.

3. Add hierarchical ternary pooling for richer options without losing sovereignty.
- Level-0: single ternary axis per signal.
- Level-1: 4-axis pool (`3^4=81`).
- Level-2: chain-context pool (`stage x domain x pool_id`) for larger discrete state space in Galaxy metadata.

4. Use negative examples as opposite-sign gradient in ranking priors.
- Success: increment pattern/source priors.
- Failure: decrement priors and increase anti-pattern pressure.
- Uncertain: keep exploration weight elevated for top-k stochastic selection.

5. Store all RLWHF outcomes in Grammar/Shadow Copy with provenance.
- Event tags should include `chain`, `phase`, `source_galaxy`, `target_galaxy`, `pool_id`, and `anti_pattern_pressure`.
- This allows SleepTime to consolidate both positive and negative learning traces.

## Next Patch Set (recommended)

1. Hard consistency gates v2 in `benchmarks/arc_agi_2_adapter.py`
- Add train-pair family inference and reject candidates violating inferred family.
- Add palette mapping stability check over all train pairs.
- Add object-count delta check over all train pairs.

2. Ternary source priors in ranking
- Introduce dynamic source prior from observed precision per source for current run window.
- Blend with per-pattern ternary quality prior from `knowledge3d/knowledgeverse/ternary_quality_memory.py`.

3. Oracle diagnostics expansion
- Emit exact + fuzzy@0.80/0.90/0.95 in summary.
- Emit per-source oracle hits (which source produced best fuzzy/exact candidate).

4. Keep default-galaxy eager load hard-on in benchmark flows
- Already wired; maintain as invariant for all training/testing commands.

## Test Status (targeted)

- `tests/test_arc_agi_2_adapter.py`: `7 passed`
- `tests/test_run_all_benchmarks_history.py`: `1 passed`
- `tests/test_navigator_specialist.py`: `9 passed`
- `tests/test_run_all_global_benchmarks_history.py`: `1 passed`
- `tests/test_global_benchmark_scripts.py`: `3 passed`

Total in this validation pass: `21 passed`.
