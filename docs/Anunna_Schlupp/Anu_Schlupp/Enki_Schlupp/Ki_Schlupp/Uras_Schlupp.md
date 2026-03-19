# Full Benchmark Report

Date: 2026-03-19
Run type: `scripts/run_enriched_benchmarks.py --full`
Provider: sovereign
Session id: `full-736b4175b3a7`
Status: completed

## Saved state

- Resume/completion state:
  - `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Full stdout log:
  - `/tmp/k3d_full_benchmark_20260318.log`
- Accumulative health log:
  - `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- Sleep-time journal:
  - `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- Routing weights checkpoint:
  - `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`

These artifacts are already persisted. Development can continue from this state without losing the sleep-time growth.

## Requested vs actual scope

Requested full counts:
- ARC `120`
- Math `500`
- GSM8K `1319`
- LHE `100`
- MMLU `14042`

Actual completed counts:
- ARC `120`
- Math `20`
- GSM8K `1319`
- LHE `100`
- MMLU `14042`

Combined actual total:
- `15601`

## Final benchmark results

- ARC:
  - `10/120`
  - `8.33%`
  - `37.144s`
  - `0.3095s/q`
- Math:
  - `20/20`
  - `100.00%`
  - `32.421s`
  - `1.6211s/q`
- GSM8K:
  - `30/1319`
  - `2.27%`
  - `2275.506s`
  - `1.7252s/q`
- LHE:
  - `9/100`
  - `9.00%`
  - `130.951s`
  - `1.3095s/q`
- MMLU:
  - `3255/14042`
  - `23.18%`
  - `43500.595s`
  - `3.0979s/q`

Combined:
- `3324/15601`
- `21.31%`
- `45977.037s` total wall time inside the runner

## Galaxy state used for this full run

- Benchmark keyword count:
  - `36253`
- H19 stars available:
  - `117497`
- H19 meaning stars loaded:
  - `5000`
- Filter stats:
  - `min_languages = 5`
  - `stopwords_removed = 148`
  - `foundation_duplicates_removed = 3409`
  - `after_keyword_filter = 42727`
  - `capped_removed = 37727`
- Foundation lemma count:
  - `2415`
- B3 proceduralized stars:
  - `20`

Per-galaxy ingest status from this run:
- Language:
  - `4891 inserted`
- Math:
  - `73 inserted`
  - `20 updated`
- Reality:
  - `10 inserted`
- Drawing:
  - `9 updated`
- Grammar:
  - `17 updated`

## MMLU breakdown (top 10 subjects by volume)

- professional_law:
  - `353/1534`
  - `23.01%`
- moral_scenarios:
  - `183/895`
  - `20.45%`
- miscellaneous:
  - `199/783`
  - `25.42%`
- professional_psychology:
  - `140/612`
  - `22.88%`
- high_school_psychology:
  - `114/545`
  - `20.92%`
- high_school_macroeconomics:
  - `83/390`
  - `21.28%`
- elementary_mathematics:
  - `85/378`
  - `22.49%`
- moral_disputes:
  - `95/346`
  - `27.46%`
- prehistory:
  - `90/324`
  - `27.78%`
- philosophy:
  - `82/311`
  - `26.37%`

## Sleep-time result

Sleep-time completed successfully after the benchmark run.

Stage A:
- success: `true`
- health log total seen: `15840`
- correct: `3473`
- incorrect: `12367`
- note:
  - this is the accumulative health log, not only this session

Stage B:
- success: `true`
- updated specialists:
  - `chat`
  - `grammar`
  - `math`
  - `visual`
- updated count:
  - `83504`
- weights path:
  - `/K3D/Knowledge3D.local/checkpoints/trm_routing_state.json`

## Confirmed issues to fix next

1. Remove the H19 Python cap.
   - The full run still used `max_stars = 5000`.
   - Per the architecture note, knowledge volume should be managed by the composed head pipeline, not by Python capping.

2. Fix Math benchmark to use the real dataset.
   - The run requested `500` math questions and only got `20`.
   - Root cause:
     - `MathCompetitionBenchmark` defaults to `dataset_mode="synthetic"` when no explicit present-data mode is forced.
     - The real dataset exists at:
       - `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl`
   - Acceptable fixes:
     - make `MathCompetitionBenchmark` try present mode first when the dataset exists
     - or force `dataset_mode="present"` in `benchmark_health_check.py`

## Practical development note

You can continue development immediately from the saved state above. The important persistent outputs are:
- `health_log.full.run_state.json`
- `health_log.jsonl`
- `sleeptime_journal.jsonl`
- `trm_routing_state.json`

Nothing needs to be rerun just to preserve learning. The sleep-time growth is already committed.
