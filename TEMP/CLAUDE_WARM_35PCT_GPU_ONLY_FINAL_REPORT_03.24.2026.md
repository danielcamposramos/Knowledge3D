# Claude Warm 35% GPU-Only Final Report — 2026-03-24

## Context

- Run type: warm 35% benchmark, sovereign GPU-only contrastive intent
- Session: `full-062d5a5a3af7`
- State source: `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- Run-state file: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Sleeptime source: `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`

Important caveat:

- This finished session is a resumed session, not a fully fresh benchmark ID.
- `health_log.full.run_state.json` shows `completed: true`, but its `summaries` block is empty, so the scores below were reconstructed directly from persisted rows in `health_log.jsonl`.
- The warm benchmark log path used for this family of runs is `/tmp/k3d_warm_contrastive_gpu_only_35pct_03.23.log`.

## Final Scores

| Suite | Correct | Total | Percent |
|---|---:|---:|---:|
| ARC | 2 | 42 | 4.76% |
| Math | 4 | 500 | 0.80% |
| GSM8K | 6 | 462 | 1.30% |
| LHE | 2 | 35 | 5.71% |
| MMLU | 1130 | 4915 | 22.99% |
| Combined | 1144 | 5954 | 19.21% |

## Warm Boot / Persistence

- Warm boot confirmed in benchmark log:
  - `Warm boot: House loaded (247889 entries across 19 galaxies)`
- The persisted session completed successfully:
  - `session_id: full-062d5a5a3af7`
  - `completed: true`
- No benchmark or pytest process remained active after completion.

## Contrastive / Sleeptime Outcome

Latest `stage_b.contrastive` summary for this completed session:

- `rows: 5954`
- `checkpoint: {}`
- `skipped: false`

Per-specialist outcome:

| Specialist | Trained | Positives | Negatives | Result |
|---|---:|---:|---:|---|
| chat | false | 1130 | 3785 | `argument 2: TypeError: Don't know how to convert parameter 2` |
| grammar | false | 2 | 33 | `argument 2: TypeError: Don't know how to convert parameter 2` |
| math | false | 439 | 523 | `argument 2: TypeError: Don't know how to convert parameter 2` |
| visual | false | 41 | 1 | `argument 2: TypeError: Don't know how to convert parameter 2` |

Interpretation:

- The isolated sovereign GPU smoke had already passed earlier, but the real benchmark session still logged the same contrastive failure during sleeptime.
- So the benchmark path and the isolated smoke path are still not behaviorally aligned.
- Visual data improved materially on the data side (`41` positives instead of the earlier starved visual signal), but training still did not complete.

## Live Runtime Evidence

Two capture files were taken specifically to preserve evidence for the migration from Python orchestration to a GPU-resident live game loop:

- `TEMP/WARM_35PCT_LIVE_MONITOR_03.24.2026.md`
- `TEMP/WARM_35PCT_LIVE_MONITOR_1MIN_03.24.2026.md`

### Capture A: 3-minute window

From `TEMP/WARM_35PCT_LIVE_MONITOR_03.24.2026.md`:

- duration: `180s`
- samples: `36`
- GPU util avg: `1.31%`
- GPU util max: `4.00%`
- GPU mem avg/max: `306 MB`
- process GPU mem avg/max: `292 MB`
- process CPU avg: `148.83%`
- process CPU max: `149.00%`
- process RSS avg: `4.33 GB`
- hottest thread avg: `148.86%`

### Capture B: 1-minute window

From `TEMP/WARM_35PCT_LIVE_MONITOR_1MIN_03.24.2026.md`:

- duration: `60s`
- samples: `12`
- GPU util avg: `1.25%`
- GPU util max: `3.00%`
- GPU util min: `1.00%`
- GPU mem avg: `306 MB`
- process GPU mem avg: `292 MB`
- process CPU avg: `149.25%`
- process CPU max: `150.00%`
- process RSS avg: `4.34 GB`
- hottest thread avg: `149.25%`

## Architectural Conclusion

The two captures strongly support the same conclusion:

- The system is still overwhelmingly Python/CPU driven during benchmark execution.
- GPU memory is allocated, but GPU compute remains near-idle.
- VRAM footprint is tiny and flat instead of behaving like a saturated live game substrate.
- One dominant hot thread is still owning the loop.

In other words:

- The benchmark is not currently acting like a continuously resident GPU game world.
- It is still acting like a Python orchestrator that occasionally touches the GPU.

## Most Important Transfer Insight

These captures should be treated as hard evidence for the ongoing Phase D migration:

1. TRM/main model perception and routing are still too Python-mediated.
2. Retrieval/ranking/graph-building are still not fully resident in the GPU loop.
3. Jarvis coordination still sits downstream of too much host-side orchestration.
4. The next wins must keep moving execution out of Python and into the live GPU game loop, not add more host-layer logic.

## Evidence Paths

- `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- `/tmp/k3d_warm_contrastive_gpu_only_35pct_03.23.log`
- `TEMP/WARM_35PCT_LIVE_MONITOR_03.24.2026.md`
- `TEMP/WARM_35PCT_LIVE_MONITOR_1MIN_03.24.2026.md`
