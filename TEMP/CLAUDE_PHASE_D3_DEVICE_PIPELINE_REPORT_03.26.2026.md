# Claude Handoff — Phase D.3 Device Pipeline Warm 35% Report

**Date:** 2026-03-26  
**Session:** `full-6264439cf31b`  
**Run Type:** warm 35% validation  
**Runner:** `scripts/run_enriched_benchmarks.py` with `K3D_DEVICE_PIPELINE=1`

## What Changed

Phase D.3 was implemented and validated before the run:

- device-output Morton query path
- device-fed Frustum culling path
- device-fed LOD path with late readback
- feature-gated device composed-head path in `knowledgeverse.py`
- targeted NumPy cleanup on the D.3 migration slice

Focused validation before the benchmark:

- `13 passed`

Direct internal smoke proved the new chained device path actually fired:

- `Device pipeline: morton -> frustum -> lod chained on GPU`
- `Morton locate/device: 21 raw candidates`
- `Frustum cull/device: 18/21 visible`
- `Dynamic LOD/device: range=3-6 across 18 visible nodes`

## Benchmark Result

Warm boot confirmed in the live log:

- `Warm boot: House loaded (247889 entries across 19 galaxies)`

Final completed totals:

- ARC: `2/42` = `4.76%`
- Math: `1/500` = `0.20%`
- GSM8K: `3/462` = `0.65%`
- LHE: `1/35` = `2.86%`
- MMLU: `850/4915` = `17.29%`
- Combined: `857/5954` = `14.39%`

## Comparison vs D.2

Previous D.2 warm 35% result:

- ARC: `2/42`
- Math: `3/500`
- GSM8K: `4/462`
- LHE: `2/35`
- MMLU: `1062/4915`
- Combined: `1073/5954` = `18.02%`

D.3 delta vs D.2:

- ARC: unchanged
- Math: `-2`
- GSM8K: `-1`
- LHE: `-1`
- MMLU: `-212`
- Combined: `-216 correct`, `-3.63 points`

## Sleep-Time / Learning Outcome

Sleep-time completed successfully for the D.3 session.

- checkpoint saved: `/K3D/Knowledge3D.local/checkpoints/adaptive_swarm`
- updated specialists: `chat`, `grammar`, `math`, `visual`
- Jarvis updated: `true`
- updated count: `18189`

Contrastive summary:

- `chat`: positives `850`, negatives `4065`, steps `3003`, trained `true`
- `grammar`: positives `1`, negatives `34`, steps `6`, trained `true`
- `math`: positives `75`, negatives `887`, steps `906`, trained `true`
- `visual`: positives `42`, negatives `0`, steps `126`, trained `true`

## GPU / CPU Runtime Signal

Use the corrected Python-PID monitor, not the earlier wrapper-PID capture.

Correct D.3 monitor:

- file: `TEMP/WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_PYTHONPID_03.26.2026.md`
- GPU util avg/min/max: `0.17% / 0.00% / 1.00%`
- process GPU mem avg/max: `1380 MB / 1502 MB`
- process CPU avg/max: `106.92% / 108.00%`
- hottest thread avg/max: `98.09% / 98.10%`

D.2 comparison monitor:

- file: `TEMP/WARM_35PCT_PHASE_D2_RECURSIVE_TRM_LIVE_MONITOR_2MIN_03.25.2026.md`
- GPU util avg/min/max: `3.88% / 0.00% / 25.00%`
- process CPU avg/max: `112.26% / 113.00%`

## Honest Interpretation

Phase D.3 is architecturally real:

- the device-buffer chain exists
- the internal smoke proved it executed on GPU
- the full warm benchmark completed end-to-end
- sleep-time and checkpointing succeeded

But benchmark quality regressed sharply, and the corrected live monitor did **not** show better GPU utilization than D.2.

Most likely explanation:

- the current D.3 device-navigation slice bypasses or weakens important accuracy-critical logic still present in the richer CPU-side path
- likely candidates: semantic rescue/scoring details, LED-informed local routing, or composed-head candidate shaping that the simplified device chain does not yet preserve
- the Python outer orchestration loop is still dominant, so D.3 did not yet deliver the intended utilization lift

## Recommendation For Next Plan

Plan the next phase around **semantic parity before more acceleration**:

1. Compare D.2 vs D.3 candidate traces for the same questions and identify where D.3 loses relevant candidates or ranking quality.
2. Preserve the full composed-head semantics while keeping Morton/Frustum/LOD buffers resident on device.
3. Only after parity is restored, continue pushing elimination of Python between kernels.
4. Keep using the corrected Python-PID monitor for utilization claims.

## Evidence

- Log: `/tmp/k3d_phaseD3_device_pipeline_warm_35pct_03.26.log`
- Run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Row log: `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- Sleep journal: `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- D.3 status note: `TEMP/CLAUDE_PHASE_D3_DEVICE_PIPELINE_STATUS_03.26.2026.md`
- Corrected D.3 monitor: `TEMP/WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_PYTHONPID_03.26.2026.md`
- D.2 comparison report: `TEMP/CLAUDE_PHASE_D2_RECURSIVE_TRM_REPORT_03.26.2026.md`
- D.2 comparison monitor: `TEMP/WARM_35PCT_PHASE_D2_RECURSIVE_TRM_LIVE_MONITOR_2MIN_03.25.2026.md`
