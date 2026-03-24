# Warm 35% Validation Report — Contrastive Fix Pass

**Date:** 2026-03-23
**Session:** `full-1564a6228402`
**Boot:** Warm boot
**Log:** `/tmp/k3d_warm_contrastive_fix_35pct_03.23.2026.log`
**Run state:** `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
**Sleep-time journal:** `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`

---

## Executive Summary

The warm 35% benchmark **completed end-to-end**. The House warm boot worked correctly, all five suites finished, and sleep-time committed successfully.

However, the core validation goal for this pass **did not succeed**:

- Contrastive training still failed for **all four specialists**
- The same error persisted: `argument 2: TypeError: Don't know how to convert parameter 2`
- The contrastive checkpoint remained **empty**: `{}`  
- Combined score **regressed** from the prior warm run

There is one real improvement:

- The ARC missed-positive fix clearly increased **visual positives** from `2` to `41`

But the visual specialist still logged only:

- `positives: 41`
- `negatives: 1`
- `trained: false`

So the data collection improved, but the actual GPU contrastive update path is still broken.

---

## Final Results

| Suite | Score | Accuracy |
|------|------:|---------:|
| ARC | `2/42` | `4.76%` |
| Math | `4/500` | `0.80%` |
| GSM8K | `7/462` | `1.52%` |
| LHE | `2/35` | `5.71%` |
| MMLU | `1091/4915` | `22.20%` |
| **Combined** | **`1106/5954`** | **`18.58%`** |

---

## Comparison vs Previous Warm 35% Run

Previous warm run (`full-ee0836e235e2`, 2026-03-22):

- ARC: `2/42`
- Math: `4/500`
- GSM8K: `6/462`
- LHE: `2/35`
- MMLU: `1129/4915`
- Combined: `1143/5954 = 19.20%`

Current run (`full-1564a6228402`, 2026-03-23):

- ARC: `2/42` (`no change`)
- Math: `4/500` (`no change`)
- GSM8K: `7/462` (`+1`)
- LHE: `2/35` (`no change`)
- MMLU: `1091/4915` (`-38`)
- Combined: `1106/5954 = 18.58%` (`-37 overall`)

Conclusion:

- The benchmark **did not meet** the target `>= 19.20%`
- Any benefit from the contrastive-fix pass did **not** translate into net benchmark improvement in this run

---

## Boot / Persistence Status

Confirmed from the run log:

- `Warm boot: House loaded (247889 entries across 19 galaxies)`

The run completed fully and the suite summaries were persisted in run-state:

- ARC completed
- Math completed
- GSM8K completed
- LHE completed
- MMLU completed

Sleep-time also committed successfully after the benchmark.

---

## Contrastive Summary

From the final matching sleep-time commit for `session_id = full-1564a6228402`:

### Checkpoint

- `checkpoint: {}`
- No saved adaptive-swarm contrastive checkpoint payload

### Specialists

| Specialist | Positives | Negatives | Trained | Result |
|-----------|----------:|----------:|:-------:|--------|
| chat | `1091` | `3824` | `false` | failed |
| grammar | `2` | `33` | `false` | failed |
| math | `440` | `522` | `false` | failed |
| visual | `41` | `1` | `false` | failed |

### Error

All four specialists failed with the same error:

`argument 2: TypeError: Don't know how to convert parameter 2`

This exact error also appears in the benchmark log.

---

## What Improved

The ARC `None -> missed positive` correction clearly changed the data entering sleep-time:

- Previous visual counts: `positives: 2`, `negatives: 1`
- Current visual counts: `positives: 41`, `negatives: 1`

So the missed-positive fix is working as intended for **positive signal capture**.

That means:

1. ARC failures with `answer = None` are no longer being silently discarded.
2. Visual specialist training data is reaching sleep-time in much larger quantity.

---

## What Did Not Improve

### 1. Contrastive update still fails at execution time

The data is now better, but the GPU-side contrastive training path still does not complete.

### 2. Visual negatives did not climb

Expected:

- visual negatives should rise substantially from ARC misses / wrong outputs

Observed:

- visual negatives stayed at `1`

So the pass improved missed-positive accounting, but **not** negative-pair generation for visual.

### 3. Adapter-weight update remains unverified in real run

Because contrastive training failed for all specialists and checkpoint output stayed empty, this run does **not** provide evidence that the saved adapter weights actually changed during sleep-time.

---

## Jarvis / Sleep-Time Notes

Jarvis summary from the same sleep-time commit:

- `agreements: 512`
- `contradictions: 344`
- `briefs_consolidated: 128`
- `updated: true`

Updated specialists list:

- `chat`
- `grammar`
- `math`
- `visual`

But this reflects the broader sleep-time logic stage, not successful contrastive adapter training.

---

## Honest Conclusion

This run proves four things:

1. The benchmark infrastructure is healthy enough to complete a full warm 35% run.
2. Warm boot and persistence are working.
3. The ARC missed-positive fix materially increased visual positive training examples.
4. The actual contrastive GPU update path is **still broken** for all specialists.

So the next debugging target is no longer "are we collecting the right examples?"  
It is:

**Why does the GPU contrastive path still throw `TypeError: Don't know how to convert parameter 2` even after the PTX copy fixes?**

That is the remaining blocker before contrastive learning can be evaluated honestly.
