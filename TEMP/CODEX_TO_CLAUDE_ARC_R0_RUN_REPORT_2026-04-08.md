# Codex to Claude: ARC R0 Run Report
**Date:** 2026-04-08
**Completed at:** 2026-04-08 18:32:12 -0300

## 1. ARC-2 task path found

- Auto-detected path:
  - `/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation`
- Status:
  - exists = `true`
- No download was needed.

## 2. ARC-2 score

- Command used:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc2_local_runner.py --max-tasks 20 --summary-output /tmp/arc2_r0_summary.json`
- Result:
  - `tasks=20`
  - `correct=0`
  - `total_inputs=20`
  - `score=0.00%`

## 3. match_type distribution across the 20 tasks

- `nearest_training_pair = 20`
- `exact_poly_match = 0`

All 20 sampled tasks fell through to the current nearest-training-pair fallback, which is consistent with the intended R0 baseline.

## 4. submission.csv generated

- Generated: `yes`
- Path:
  - `/tmp/arc2_r0_submission.csv`
- Validation status:
  - `yes` via the runner's built-in `validate_arc_submission()` path
- Important note:
  - the current formatter writes JSON payload content even when the output filename ends with `.csv`

First 5 lines:

```text
{
  "00576224": [
    {
      "attempt_1": [
        [
```

## 5. ARC-3 API key

- Key file:
  - `present`
- Environment variable:
  - `ARC_API_KEY env: NOT SET`

The run used the canonical secret-file path rather than an exported environment variable.

## 6. ARC-3 result

- Command used:
  - `bash scripts/k3d_env.sh run -e trmc_core python - <<'PY' ... K3DAgent('ls20', max_steps=60, allow_remote_compat=True) ...`
- Result:
  - `steps = 60`
  - `levels_completed = 0`
  - `score = 0.0`
  - `transport = "remote_api_compat"`
  - `sdk_error = "arc_agi installed without Arcade/make runtime"`
  - `policy_error = "sovereign_build_feed_missing:run scripts/rebuild_sovereign_artifact.py --refresh-build-feed --force-rebuild"`

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
  "policy_error": "sovereign_build_feed_missing:run scripts/rebuild_sovereign_artifact.py --refresh-build-feed --force-rebuild",
  "transport": "remote_api_compat"
}
```

## 7. Installed `arc-agi` package surface

Full Step 6 output:

```text
arc_agi version: unknown
arc_agi API surface: ['ANSI_PALETTE', 'ARC1Evaluation', 'ARC1Training', 'ARC2Evaluation', 'ARC2Training', 'CSS_PALETTE', 'Dataset', 'Grid', 'Layout', 'Pair', 'RemoteDataset', 'Symbol', 'Task', 'sys']
arcengine not importable: No module named 'arcengine'
arc not importable: No module named 'arc'
```

## 8. Protected ingest PID 101379

- Still alive: `yes`
- Current runtime at recheck:
  - `03:51:41`
- Command line remains unchanged and untouched.

## Short conclusion

R0 now has a real measured ARC-2 baseline and a truthful ARC-3 remote-compat execution result.

- ARC-2 baseline is fully closed-loop but still weak:
  - `0 / 20 = 0.00%`
  - current behavior is pure `nearest_training_pair`
- ARC-3 remote access is available through the present secret file, but the installed `arc-agi` package still lacks the official live runtime surface
- The next substantive gain is R1 reasoning quality, not more R0 plumbing
