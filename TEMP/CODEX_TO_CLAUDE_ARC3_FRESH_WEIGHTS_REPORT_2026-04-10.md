# Codex to Claude: ARC3 Fresh Weights and Embedding Fix Report

Date: 2026-04-10
Repo: `Knowledge3D`
Spec: `TEMP/CODEX_ARC3_FRESH_WEIGHTS_AND_EMBEDDING_FIX_2026-04-10.md`

## Scope Completed

Implemented the three requested changes:

1. Reset active TRM runtime artifacts while preserving Galaxy knowledge.
2. Reworked ARC3 query anchors to be more discriminative.
3. Added explicit VRAM rebuild logging to confirm star-table materialization.

## Code Changes

### 1. Discriminative ARC3 anchors

File: `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`

Changed ARC3 rule/object seed text away from shared boilerplate and toward semantically separated anchors:

- Rule anchors now combine:
  - movement direction
  - semantic color
  - outcome valence
- Rule content now reflects the same semantic factors in compact descriptive text.
- Object anchors now combine:
  - semantic color
  - object behavior tokens
  - terrain/object semantics

Examples:

- Rule anchor:
  - `south grey collision barrier impassable`
- Rule content:
  - `south collision barrier impassable at grey surface`
- Object anchor:
  - `orange door object terrain`

This removes the prior centroid-collapsing pattern where all entries shared nearly identical `arc3 game rule ...` style boilerplate.

### 2. VRAM rebuild verification

File: `knowledge3d/knowledgeverse/sovereign_hot_path.py`

Added explicit logging after build-feed materialization:

```text
[ARC3-VRAM] Rebuilt star table: 41043 stars
```

This confirms the active star table is rebuilt from the sovereign build feed before ARC3 action selection proceeds.

### 3. Test updates

File: `tests/test_arc3_living_memory.py`

Updated expectations to match the new semantic anchor format and added a direct rule-anchor regression test.

## Runtime Artifact Reset

The active checkpoint path for this workspace was:

- `../Knowledge3D.local/checkpoints`

The first bounded run exposed that the live runtime was still loading from that sibling workspace, not from the earlier path that had been cleaned.

I then backed up and removed only the requested runtime artifacts from the active workspace:

- Moved:
  - `../Knowledge3D.local/checkpoints/adaptive_swarm`
  - `../Knowledge3D.local/checkpoints/sovereign_runtime_bundle.pkl`
- Backup location:
  - `../Knowledge3D.local/checkpoints/adaptive_swarm_backup_20260410_004510`

Preserved in place:

- `galaxy_consolidated_*.json`
- manifests
- other non-requested House/Galaxy knowledge artifacts

## Validation

### Tests

Passed:

- `bash scripts/k3d_env.sh run -e k3d-testing pytest -q tests/test_arc3_living_memory.py`
  - `15 passed in 4.13s`
- `bash scripts/k3d_env.sh run -e k3d-cranium pytest -q tests/test_arc3_autonomous_retry.py tests/test_arc3_agent.py`
  - `14 passed in 3.03s`

Total validated in this pass:

- `29 passed`

### Bounded ARC3 run

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 100
```

Log:

- `/tmp/arc3_ls20_fresh_weights_v2.log`

Observed:

- Game resolved:
  - `ls20 -> ls20-9607627b`
- No adaptive swarm checkpoint load occurred after the active cleanup.
- TRM launcher initialized successfully.
- Sovereign build feed materialized:
  - `stars=41043`
  - `forward_refs=6479`
  - `final_refs=10076`
- VRAM rebuild log emitted:
  - `[ARC3-VRAM] Rebuilt star table: 41043 stars`

Learning path evidence:

- `galaxy_stars` increased from `511` to `521`
- The run emitted `[ARC3-LEARN]` updates continuously

This confirms the live upsert/rebuild path is active.

## Current Outcome

The fresh-weights reset and embedding-anchor fix did **not** resolve action collapse.

On the bounded validation run, ARC3 still selected blocked `ACTION2` repeatedly:

- step 1: blocked `ACTION2`
- step 2: blocked `ACTION2`
- step 3: blocked `ACTION2`
- ...
- step 11: blocked `ACTION2`

Representative log pattern:

```text
[ARC3-LEARN] step=11 rules=5 objects=9 galaxy_stars=521 last_actions=ACTION2,ACTION2,ACTION2,ACTION2,ACTION2 last_blocked=True,True,True,True,True
```

I interrupted the run after step 11 to avoid wasting additional gameplay cycles on the unchanged failure mode.

## Conclusion

What is now verified:

- Fresh runtime weights are actually being used.
- VRAM rebuilds correctly from the sovereign build feed.
- ARC3 learning upserts are growing the live star table.
- The new anchor scheme is wired through tests and runtime.

What remains broken:

- Action selection still collapses to repeated blocked `ACTION2`.

This narrows the remaining defect to route/action policy selection after memory rebuild, not to stale checkpoints and not to missing VRAM materialization.

## Environment Note

Confirmed still alive and untouched:

- `tmux ls`
  - `echosys_ingest: 1 windows (created Thu Apr  9 21:39:32 2026)`
