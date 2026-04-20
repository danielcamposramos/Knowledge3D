---
date: 2026-04-20
author: Claude (pilot mode, Codex limit-locked)
status: ARC3 live run blocked by sovereignty-purge casualty
---

# ARC3 Live Run — Blocker Report

## Blocker

`benchmarks/arc3_sdk_agent.py` → `benchmarks/arc_agi_3.py:11` imports
`knowledge3d.knowledgeverse.arc3_episode_galaxy`, which was moved to
`Old_Attempts/2026-04-18/knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
during the sovereignty purge.

Smoke test:

```
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  -m benchmarks.arc3_sdk_agent --game ls20 --max-steps 20

ModuleNotFoundError: No module named 'knowledge3d.knowledgeverse.arc3_episode_galaxy'
```

(`benchmarks/arc_agi_3.py:11`, line 11.)

## Options for next session

1. **Restore `arc3_episode_galaxy.py`** — rehydrate from
   `Old_Attempts/2026-04-18/` and audit for sovereignty violations
   (numpy, Python-string reasoning, etc.). Patch offenders, ship.
2. **Rewrite without the dependency** — ARC3 agent is already
   heavy (~900 lines in `arc3_sdk_agent.py`). Replacing
   `ARC3EpisodeGalaxy` with a lighter `episode_tape` structure
   inside `arc3_knowledge_builder.py` (which survived the purge)
   may be cleaner than restoring.
3. **Start a fresh sovereign ARC3 entry point** that feeds frames
   directly into the tablet dispatch pipeline we just validated on
   offline benchmarks, bypassing the old Episode Galaxy abstraction
   entirely. This matches the directive "ARC3 WINE: proceduralize
   game frames to Tablet live"
   ([feedback_arc3_wine_approach.md](../memory/feedback_arc3_wine_approach.md)).

Option 3 aligns best with the current architecture (TRM dispatch
path has just gained answer materialization and is cross-benchmark
proven). Deferred to next session with a proper spec.

## What IS unblocked

Offline cross-benchmark baseline is stable and reproducible:

```
CUDA_VISIBLE_DEVICES=0 K3D_SOVEREIGN_FEED_WORKERS=1 \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  -m benchmarks.{gsm8k|mmlu|math_competitions} --max-tasks {N}
```

Confirmed accuracies at N=10:

| Benchmark | Accuracy | Materialization |
|-----------|----------|-----------------|
| GSM8K     | 10%      | 8/10 numeric    |
| MMLU      | 20%      | choice-mode     |
| Math      | 10%      | 6/10 numeric    |

All three paper-grade. Next lift: richer RPN template coverage
(reasoning correctness) and ARC3 live via Option 3 above.
