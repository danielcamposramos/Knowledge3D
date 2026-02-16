# Codex Week 22 Ingestion + Validation Report

Date: 2026-02-10
Storage world: `../Knowledge3D.local/galaxies_enriched` (single persistent world)

## 1) What was implemented now

Added script:
- `scripts/week22_multicurriculum_ingestion.py`

What it does (single Knowledgeverse instance, ordered pipeline):
1. Character expansion (drawing-symlinked form-to-meaning metadata)
2. Word expansion (requires existing Character refs)
3. Math expansion (tokenized corpus to procedural entries)
4. Reality + 3DObjects default idempotent append
5. Reality + 3DObjects corpus/sweep expansion (parametric procedural entries)
6. Grammar cross-modal bridge expansion
7. Optional SleepTime consolidation
8. Coverage report with touched galaxies

## 2) Full ingestion run executed

Command used:

```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/week22_multicurriculum_ingestion.py \
  --include-global-benchmarks \
  --max-words 5000 \
  --max-math-entries 5000 \
  --max-reality-corpus-entries 800 \
  --max-reality-sweep-entries 1200 \
  --max-3d-corpus-entries 400 \
  --max-3d-sweep-entries 800 \
  --max-grammar-bridge-entries 1500 \
  --run-sleeptime \
  --output ../Knowledge3D.local/results/week22_multicurriculum/week22_ingestion_full.json
```

Result file:
- `../Knowledge3D.local/results/week22_multicurriculum/week22_ingestion_full.json`

### Ingestion deltas (from report)
- Word: `+5000`
- Math: `+5000`
- Reality: `+1200`
- 3DObjects: `+531`
- Grammar: `+800`
- Character: `+0` (already saturated in current range)
- Audio: `+0` (not targeted in this pass)

Coverage (ingestion gate):
- touched galaxies: `3DObjects, Grammar, Math, Reality, Word`
- min required: `5`
- passed: `true`

SleepTime:
- executed successfully
- updated specialists recorded
- weights checkpoint updated

## 3) Full benchmark validation executed after ingestion

Command used:

```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-constraint-mode penalty \
  --arc-enable-negative-forms \
  --arc-enable-object-aware-generation \
  --arc-enable-rescue-lane \
  --arc-rescue-lane-size 16 \
  --arc-enable-dual-track-oracle \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip \
  --track-curriculum-coverage \
  --require-min-galaxies-per-block 5 \
  --output-dir ../Knowledge3D.local/results/week22_multicurriculum/week22_post_ingest_full100 \
  --storage-root ../Knowledge3D.local
```

Result file:
- `../Knowledge3D.local/results/week22_multicurriculum/week22_post_ingest_full100/week14_benchmark_summary.json`

### Key metrics (enriched)
- ARC accuracy: `0.06` (maintained)
- ARC oracle_at_all: `0.01` (maintained)
- ARC fuzzy_oracle_at_all: `0.06` (maintained)
- ARC fuzzy@0.90: `0.13` (maintained)
- ARC generation_failure_rate: `0.99` (still bottleneck)
- ARC palette_score_mean: `0.7566` (high, but still failure-dominant)
- Math accuracy: `0.3333` (maintained)
- LHE accuracy: `1.0` (maintained)

### Architecture checks (passed)
- solver: `arc_ptx_ops`
- `ptx_full_used_rate = 1.0`
- `ptx_ranking_used_rate = 1.0`
- `ptx_oracle_used_rate = 1.0`
- persistence mode: `unified`
- shared instance: `true`
- lazy embeddings mode: `skip`

## 4) Important blocker diagnosis

The architecture is stable and ingestion growth is real, but benchmark metrics did not move because the **current dominant blocker remains generation quality** (`generation_failure_rate=0.99`) under ARC.

Also, current curriculum-coverage telemetry in `run_all_benchmarks.py` is based on **count deltas** during benchmark blocks. In this run it reports only Grammar touched in benchmark execution blocks, which under-represents actual retrieval/navigation participation. This is a telemetry model issue, not necessarily a multi-galaxy architecture failure.

## 5) Questions/suggestions for Claude (next-step alignment)

1. Should we redefine benchmark curriculum coverage from "entry-count changed" to "galaxy read/query participation" (route evidence) to match intended multi-galaxy gating?
2. Should ARC stage gate include a hard requirement on `generation_failure_rate` trend (delta over last N runs) rather than absolute threshold first?
3. For Week 22.1, do we prioritize **object/palette-consistent generator primitives** over additional broad ingestion, since broad ingestion didn’t shift oracle/generation metrics yet?
4. Do we want a dedicated `--arc-oracle-search-lane-size` (e.g., 32) separated from prediction lane size, to strengthen learning signal without changing top-1 metric semantics?
5. Should we add a small ARC curriculum subset that explicitly targets the top observed failure modes (`palette`, `object_count`, `shape`) and track per-mode lift before full100?

## 6) Notes on world continuity

This was **not a clean world run**. The same enriched world was reused and expanded in place.

Evidence:
- storage root remained `../Knowledge3D.local/galaxies_enriched`
- single instance in ingestion
- unified shared instance in benchmark run
- large additive galaxy deltas persisted

