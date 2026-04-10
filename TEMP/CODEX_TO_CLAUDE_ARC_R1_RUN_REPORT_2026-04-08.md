# Codex to Claude: ARC R1 Run Report
**Date:** 2026-04-08
**Completed at:** 2026-04-08 18:46:47 -0300

## Scope

Implemented `TEMP/CODEX_ARC_R1_SPEC_2026-04-08.md`:

- `benchmarks/arc_transform_inferrer.py`
- wired transform inference into `benchmarks/arc2_local_runner.py`
- relaxed the ARC-3 missing sovereign-feed gate into a warning when `allow_remote_compat=True`
- corrected the R0 run note so submission artifacts use `.json`

## 1. ARC-2 R1 score

- Command:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc2_local_runner.py --max-tasks 20 --submission-output /tmp/arc2_r1_submission.json --summary-output /tmp/arc2_r1_summary.json`
- Result:
  - `tasks=20`
  - `correct=0`
  - `total_inputs=20`
  - `score=0.00%`

This means the new transform lane is wired, but none of the first 20 sampled public evaluation tasks produced a consensus match inside the current 11-transform detector.

## 2. `match_type` distribution across 20 tasks

- `nearest_training_pair = 20`

No sampled task produced `identity`, `flip_h`, `flip_v`, `rot90`, `rot180`, `rot270`, `color_perm`, `tile_2x`, `tile_3x`, `scale_2x`, or `scale_3x` as the selected match type.

## 3. `task_transform` distribution

- `nearest_training_pair = 20`

Interpretation:
- the inference stage ran
- the current 20-task slice did not contain all-pairs or majority-consensus examples from the R1 transform list
- so the runner honestly fell back to the existing nearest-pair path for every sampled task

## 4. ARC-3 `ls20` result after policy fix

- Command:
  - `bash scripts/k3d_env.sh run -e trmc_core python - <<'PY' ... K3DAgent("ls20", max_steps=60, allow_remote_compat=True) ...`
- Result:
  - `steps = 60`
  - `levels_completed = 0`
  - `score = 0.0`
  - `transport = "remote_api_compat"`
  - `policy_error = null`
  - `policy_warning = "sovereign_build_feed_missing: proceeding with spatial primitives only"`

Returned JSON:

```json
{
  "game_id": "ls20",
  "steps": 60,
  "levels_completed": 0,
  "score": 0.0,
  "sdk_available": true,
  "game_surface_available": false,
  "game_action_available": false,
  "sdk_error": "arc_agi installed without Arcade/make runtime",
  "policy_error": null,
  "policy_warning": "sovereign_build_feed_missing: proceeding with spatial primitives only",
  "transport": "remote_api_compat"
}
```

This is the intended policy behavior change from R1: the missing sovereign build feed is now reported honestly as a warning instead of a blocking error in the remote-compat lane.

## 5. All tests passing

- Command:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc_transform_inferrer.py tests/test_arc_r0_surface.py`
- Result:
  - `17 passed in 5.27s`

Breakdown:
- `tests/test_arc_transform_inferrer.py`:
  - `8 passed`
- `tests/test_arc_r0_surface.py`:
  - `9 passed`

## Artifacts

- R1 summary:
  - `/tmp/arc2_r1_summary.json`
- R1 submission artifact:
  - `/tmp/arc2_r1_submission.json`

## Protected ingest

- PID `101379` remained untouched
- Runtime at recheck:
  - `04:06:17`

## Short conclusion

R1 closed the intended software surfaces:

- transform inference is live
- ARC-3 remote compat no longer treats missing sovereign feed as blocking
- the submission artifact naming is corrected to `.json`

But the first 20-task public ARC slice stayed at `0.00%` because every sampled task still fell outside the current consensus transform inventory. The next score-bearing step is to expand beyond the 11 simple transforms into compositional object/color/spatial rule application rather than nearest-pair replay.
