# Claude Handoff — Routing + Contrastive + Multi-Hop Smoke

**Date:** 2026-03-22
**Run Type:** Cold-start smoke validation after routing/contrastive/multi-hop changes
**Command:**

```bash
export CUDA_VISIBLE_DEVICES=0
python3 scripts/run_enriched_benchmarks.py \
  --full \
  --cold-start \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 50 \
  --gsm8k-max 50 \
  --lhe-max 35 \
  --mmlu-max 100 \
  2>&1 | tee /tmp/k3d_routing_contrastive_smoke_03.22.2026.log
```

## Headline Result

The run completed end-to-end with no crash, no silent stop, and sleep-time executed successfully.

Scores:

| Suite | Score | Accuracy |
|------|------:|---------:|
| ARC | 0 / 42 | 0.00% |
| Math | 1 / 50 | 2.00% |
| GSM8K | 1 / 50 | 2.00% |
| LHE | 2 / 35 | 5.71% |
| MMLU | 21 / 100 | 21.00% |
| **Combined** | **25 / 277** | **9.03%** |

## What Worked

1. **The run path is stable now.**
   - Cold boot completed.
   - Full meaning-layer load completed.
   - Benchmarks completed end-to-end.
   - Sleep-time completed and saved House state.
   - No duplicate run/process issue.

2. **Architectural state was fully loaded.**
   - Meaning layer: `117,497 / 117,497`
   - ARC knowledge: `333` anchors + `333` bridges, `724` ARC-related entries across galaxies
   - Math rules: `1,199` entries ingested
   - Total persisted entries in House state: `247,974`

3. **Sleep-time Stage B did train beyond the old drawing-only path.**
   - `chat`: trained, `21` pairs
   - `grammar`: trained, `2` pairs
   - `math`: trained, `2` pairs
   - `visual`: not trained, `0` pairs

4. **Jarvis path executed and produced real consolidation stats.**
   - briefs consolidated: `128`
   - agreements: `440`
   - contradictions: `288`
   - updated: `true`

## What Failed / Regressed

### 1. ARC regressed hard

Previous floor on the same 42-question smoke target was `2/42`.

This run:
- `0/42`

So the new ARC semantic embedding / routing path did **not** improve ARC. It made it worse on this smoke sample.

### 2. Visual specialist got no contrastive learning signal

Sleep-time contrastive summary:
- `visual`: `pairs = 0`, `trained = false`, `reason = "no_pairs"`

That is consistent with ARC scoring `0/42`. The new full-brain contrastive pipeline works mechanically, but the visual branch got no positive examples from this run.

### 3. LHE multi-hop did not create a dramatic lift

LHE reached:
- `2/35` = `5.71%`

This is not a failure of stability, but it is not a major capability jump either. The graph is no longer empty, but the new edges are not yet yielding strong reasoning gains.

## Interpretation

The three fixes are **operationally integrated** but **not yet semantically winning**.

- **ARC semantic embedding:** reaches a new code path, but the current handcrafted feature text is not enough to retrieve/use the new 333 anchors effectively.
- **Full-brain contrastive:** works, persists checkpoints, and trains non-drawing specialists, but only where correct supervision exists.
- **LHE multi-hop graph edges:** now exist, but edge presence alone is not enough; the graph still needs better candidate quality and/or better crystallizer use.

## Key Evidence

- Benchmark log:
  - `/tmp/k3d_routing_contrastive_smoke_03.22.2026.log`
- Run state:
  - `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Sleep-time journal:
  - `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- House state:
  - `/K3D/Knowledge3D.local/house/galaxy_state.bin`
- Adaptive swarm checkpoint:
  - `/K3D/Knowledge3D.local/checkpoints/adaptive_swarm`

## Most Important Discussion Point

This smoke says the bottleneck is **not** "missing ARC entries" anymore.

We now have:
- full star load,
- 724 ARC-related entries,
- end-to-end stable execution,
- persisted House state,
- persisted adaptive swarm state,
- Jarvis consolidation,
- and functioning Stage B contrastive updates.

But ARC still fell to `0/42`.

That points the next diagnosis at:

1. retrieval quality for ARC anchors,
2. ranking/selection after retrieval,
3. whether retrieved ARC anchors are actually influencing the sovereign solve path,
4. and whether the ARC semantic feature text is too shallow/noisy for the available anchor query anchors.

## Recommendation For Next Step

Do **not** jump to the full 35% rerun yet.

First, inspect ARC question-by-question on a small diagnostic slice and capture:
- top retrieved ARC anchors,
- their scores,
- selected candidate path,
- whether the ARC anchors affected the final solve program at all.

The smoke run was valuable because it ruled out infrastructure failure and isolated the remaining problem to **ARC retrieval-to-usefulness**, not ingestion or persistence.
