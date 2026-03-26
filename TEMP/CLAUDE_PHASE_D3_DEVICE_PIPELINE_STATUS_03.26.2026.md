# Claude Phase D.3 Device Pipeline Status — 2026-03-26

## Scope completed

- Added device-output Morton query path in [morton_octree.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/spatial_sovereign/morton_octree.py)
- Added device-fed Frustum path and flag readback helper in [frustum.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/spatial_sovereign/frustum.py)
- Added device-fed LOD path and late readback helper in [query_head_substrate.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/knowledgeverse/query_head_substrate.py)
- Added `execute_swarm_device(...)` in [nine_chain_specialized_bridge.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py)
- Added feature-gated device pipeline in [knowledgeverse.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/knowledgeverse/knowledgeverse.py)

## Hot-path cleanup done in this pass

- `_reset_trm_state()` no longer uses NumPy zero staging
- `_decode_trm_galaxy_distribution()` no longer uses `np.asarray(...)`
- `_trm_shadow_probe()` top-k / entropy path no longer uses NumPy
- `_dispatch_swarm_weights()` no longer uses `np.asarray(trust_weights)`
- `_compose_head_navigation_candidates()` no longer uses `np.asarray(list(dict.fromkeys(...)))` for candidate dedupe

## Validation

- `python3 -m compileall ...` on touched D.3 files: passed
- `/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -m pytest -q tests/test_query_head_substrate.py tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py`
- result: `13 passed in 41.41s`
- `git diff --check`: clean

## Direct internal device smoke

Direct call to `_compose_head_navigation_candidates_device(...)` on a real `Knowledgeverse` instance produced:

- `candidate_count = 14`
- steps included:
  - `Device pipeline: morton -> frustum -> lod chained on GPU`
  - `Morton locate/device: 21 raw candidates`
  - `Frustum cull/device: 18/21 visible`
  - `Dynamic LOD/device: range=3-6 across 18 visible nodes`

This confirms the D.3 path is firing in the real runtime, not only in tests.

## Live benchmark

Launched real runner with:

```bash
CUDA_VISIBLE_DEVICES=0 \
K3D_DEVICE_PIPELINE=1 \
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -u scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 500 \
  --gsm8k-max 462 \
  --lhe-max 35 \
  --mmlu-max 4915
```

Artifacts:

- log: [/tmp/k3d_phaseD3_device_pipeline_warm_35pct_03.26.log](/tmp/k3d_phaseD3_device_pipeline_warm_35pct_03.26.log)
- run-state: [/K3D/Knowledge3D.local/logs/health_log.full.run_state.json](/K3D/Knowledge3D.local/logs/health_log.full.run_state.json)
- session id: `full-6264439cf31b`

Latest confirmed progress at write time:

- warm boot: `House loaded (247889 entries across 19 galaxies)`
- ARC completed: `2/42`
- benchmark entered unified math and reached at least `20/962`

## D.3 live monitor captures

Wrapper-PID capture:

- [WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_03.26.2026.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/TEMP/WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_03.26.2026.md)

Correct Python-PID capture:

- [WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_PYTHONPID_03.26.2026.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/TEMP/WARM_35PCT_PHASE_D3_DEVICE_PIPELINE_LIVE_MONITOR_PYTHONPID_03.26.2026.md)

Correct Python-PID numbers over ~60s:

- GPU util avg/min/max: `0.17% / 0.00% / 1.00%`
- GPU mem avg/max: `1396 MB / 1518 MB`
- process GPU mem avg/max: `1380 MB / 1502 MB`
- process CPU avg/max: `106.92% / 108.00%`
- hottest thread avg/max: `98.09% / 98.10%`

## Honest state

- D.3 device-buffer chaining is implemented and verified in the runtime path
- the full warm 35% benchmark is still running
- GPU utilization has not yet shown a clear uplift during the early live monitor window
- the next honest step is to wait for the benchmark to finish, then compare final score + utilization against D.2
