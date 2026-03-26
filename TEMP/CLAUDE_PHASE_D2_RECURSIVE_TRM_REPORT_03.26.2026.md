# Claude Phase D2 Recursive TRM Report

Session:
- benchmark session: `full-5a5441097c4a`
- benchmark log: `/tmp/k3d_phaseD2_recursive_trm_warm_35pct_03.25.log`
- run-state marker: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- row log: `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- sleeptime journal: `/K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl`
- live monitors:
  - `TEMP/WARM_35PCT_PHASE_D2_RECURSIVE_TRM_LIVE_MONITOR_03.25.2026.md`
  - `TEMP/WARM_35PCT_PHASE_D2_RECURSIVE_TRM_LIVE_MONITOR_2MIN_03.25.2026.md`

What changed in Phase D2:
- `knowledgeverse.py::_run_single_trm_tick()` was rewired to the recursive fused TRM kernel instead of the dead `kernel_fused=None` path.
- `trm_launcher.py` is NumPy-free, fused is now the default backend, and the launcher uses `HostTensorF32 + ctypes` staging.
- `trm_engine.py` is explicitly deprecated for legacy-only use.
- The benchmark really executed on the sovereign fused launcher path; TRM is no longer silently skipped at the live tick.

Benchmark result:
- ARC: `2/42` = `4.76%`
- Math: `3/500` = `0.60%`
- GSM8K: `4/462` = `0.87%`
- LHE: `2/35` = `5.71%`
- MMLU: `1062/4915` = `21.61%`
- Combined: `1073/5954` = `18.02%`

Timing:
- benchmark elapsed: `24205.541 s` = `6.72 h`
- session start: `2026-03-25T12:23:22.957377`
- sleeptime save finished by: `2026-03-25T19:30:05.528384`

Warm-boot / House facts:
- actual boot mode in benchmark: `warm`
- House loaded: `247889 entries across 19 galaxies`
- incremental update: `added 0`, `skipped 1865`
- House state saved again at end

MMLU top-subject block from final log:
- `professional_law`: `109/537` = `20.30%`
- `moral_scenarios`: `55/313` = `17.57%`
- `miscellaneous`: `65/274` = `23.72%`
- `professional_psychology`: `50/214` = `23.36%`
- `high_school_psychology`: `42/191` = `21.99%`
- `high_school_macroeconomics`: `39/136` = `28.68%`
- `elementary_mathematics`: `34/133` = `25.56%`
- `moral_disputes`: `21/121` = `17.36%`
- `prehistory`: `23/114` = `20.18%`
- `philosophy`: `25/109` = `22.94%`

Sleep-time outcome:
- sleep-time completed successfully
- adaptive swarm checkpoint saved
- checkpoint path: `/K3D/Knowledge3D.local/checkpoints/adaptive_swarm`
- trained specialists: `chat`, `grammar`, `math`, `visual`
- checkpoint saved: `true`

Contrastive summary:
- `chat`: `trained true`, `positives 1062`, `negatives 3853`, `steps 2150`, `avg_loss 1.4344`
- `grammar`: `trained true`, `positives 2`, `negatives 33`, `steps 5`, `avg_loss 1.3897`
- `math`: `trained true`, `positives 392`, `negatives 570`, `steps 831`, `avg_loss 1.3657`
- `visual`: `trained true`, `positives 42`, `negatives 0`, `steps 84`, `avg_loss 1.6099`

Jarvis summary:
- `updated true`
- `briefs_consolidated 128`
- `agreements 512`
- `contradictions 354`
- recommended groups: ARC `5`, LHE `4`, MATH `4`, MMLU `4`
- updated_count: `30410`

GPU / CPU live-monitor evidence:
- early capture:
  - GPU util avg/max: `0.02% / 1.00%`
  - process CPU avg/max: `117.73% / 126.00%`
  - process GPU mem avg: `415 MB`
- later 2-minute capture:
  - GPU util avg/min/max: `3.88% / 0.00% / 25.00%`
  - process CPU avg/max: `112.26% / 113.00%`
  - process GPU mem avg/max: `1198 MB / 1198 MB`
  - RSS avg/max: `4.58 GB / 4.60 GB`

Comparison vs previous completed warm 35% run (`full-0571aedcb6e1`):
- previous combined: `1084/5954` = `18.21%`
- Phase D2 combined: `1073/5954` = `18.02%`
- delta: `-11 correct`, `-0.19 pts`
- previous MMLU: `1067/4915`
- Phase D2 MMLU: `1062/4915`
- delta: `-5`

Honest conclusion:
- Phase D2 succeeded architecturally: the recursive TRM kernel is now actually wired into the live benchmark path, the launcher is NumPy-free, fused is the default, and the run completed end-to-end with successful sleep-time training/checkpointing.
- The live monitors show a real change in runtime behavior: the benchmark now exhibits visible GPU spikes up to `25%`, whereas the early window was effectively flat. So the recursive TRM path is being exercised.
- But the benchmark is still predominantly CPU-orchestrated overall. Average GPU utilization remained low, and score quality did not improve yet.
- The next bottleneck remains outside the recursive kernel itself: the Python-owned outer loop / pipeline wiring still dominates runtime.

