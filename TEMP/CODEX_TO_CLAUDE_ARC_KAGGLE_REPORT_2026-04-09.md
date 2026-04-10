# Codex -> Claude: ARC Kaggle Sovereignty Repair Report

Date: 2026-04-09

## Implemented

- Deleted the remaining ARC Python-reasoning shim:
  - `benchmarks/arc_transform_inferrer.py`
  - `tests/test_arc_transform_inferrer.py`
- Kept ARC-2 on the canonical tablet boundary and completed the missing pieces in:
  - `benchmarks/arc2_local_runner.py`
  - `benchmarks/arc_submission_formatter.py`
  - `scripts/run_arc2_submission.py`
- ARC-2 now:
  - seeds task Grammar stars via `seed_task(..., galaxy_manager=kv.galaxy_manager, task_id=task_id)`
  - decodes up to two predictions from the sovereign result surface
  - writes multi-sample `attempt_1` / `attempt_2` submission JSON
  - falls back to the input grid when no prediction is available so the artifact remains structurally valid
- Repaired ARC-3 SDK sovereignty in:
  - `benchmarks/arc3_sdk_agent.py`
- ARC-3 now:
  - delegates action choice to `benchmarks.arc_agi_3.K3DARC3Agent`
  - has no Python action heuristic fallback
  - normalizes remote-compat action descriptors before wrapping them into the sovereign envelope
  - accepts `--game` as required by the spec run command
- Added thin WINE public entrypoints:
  - `knowledge3d/tablet/wine/arc2_wine.py`
  - `knowledge3d/tablet/wine/arc3_wine.py`
- Added the offline Kaggle notebook/CLI surface:
  - `notebooks/arc_agi_2_kaggle_submission.py`
- Wrote paper-track evidence:
  - `TEMP/ARC_PAPER_EVIDENCE_2026-04-09.json`

## Verified

### Tests

- `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc_submission_formatter.py tests/test_arc_r0_surface.py tests/test_arc3_agent.py`
  - `21 passed in 6.46s`
- after ARC-3 normalization repair:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc_r0_surface.py tests/test_arc3_agent.py`
  - `19 passed in 13.50s`
- formatter regression after identity fallback:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc_submission_formatter.py`
  - `3 passed in 3.57s`

### ARC-AGI-2 offline evaluation

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc2_local_runner.py \
  --tasks-dir /K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation \
  --max-tasks 20 \
  --submission-output /tmp/arc2_r2_submission.json \
  --summary-output /tmp/arc2_r2_summary.json
```

Result:

- tasks: `20`
- correct: `0`
- total inputs: `20`
- score: `0.0`

Artifacts:

- `/tmp/arc2_r2_summary.json`
- `/tmp/arc2_r2_submission.json`

### ARC-AGI-2 Kaggle notebook smoke

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python notebooks/arc_agi_2_kaggle_submission.py \
  --tasks-dir /K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation \
  --limit 3 \
  --output /tmp/arc2_kaggle_smoke_submission.json \
  --storage-root /K3D/Knowledge3D.local/runtime/arc_kaggle_smoke
```

Result:

- tasks processed: `3`
- rows emitted: `3`
- output: `/tmp/arc2_kaggle_smoke_submission.json`

The smoke artifact now has valid `attempt_1` / `attempt_2` grids rather than `null` primary attempts.

### ARC-AGI-3 live run

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 200
```

Result:

- game: `ls20`
- steps: `200`
- levels completed: `0`
- score: `0.0`
- transport: `remote_api_compat`
- sdk surface state:
  - `sdk_available = true`
  - `game_surface_available = false`
  - `game_action_available = false`
  - `sdk_error = "arc_agi installed without Arcade/make runtime"`
- no Python fallback action path was used

### Kaggle readiness / blocker

- `~/.kaggle/kaggle.json`: missing
- `kaggle` package in `k3d-cranium`: not installed
- I did not create credentials or install the CLI because the spec explicitly says to stop and report when credentials are missing

### Background ingest

- `tmux ls` still shows:
  - `echosys_ingest: 1 windows (created Thu Apr  9 02:35:35 2026)`
- I did not touch the EchoSystems ingest session

## Notes

- The sovereign ARC repair is complete at the benchmark boundary:
  - ARC-2 uses task seeding + WINE/tablet + result decode only
  - ARC-3 uses the sovereign ARC-3 agent path and no Python heuristic action synthesis
- The honest ARC-2 score remains `0/20`, which is now a knowledge baseline rather than a Python-transform shortcut failure.
- The evidence JSON includes:
  - all-question one-boot benchmark evidence
  - ARC-2 offline evidence
  - ARC-3 live evidence
  - swarm slot metadata
  - Kaggle blocker state
  - background EchoSystems ingest status
