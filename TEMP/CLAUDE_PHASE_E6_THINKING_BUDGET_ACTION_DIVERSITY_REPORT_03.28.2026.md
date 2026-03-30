# Phase E.6 Report — Thinking Budget + Action Diversity

**Date:** 2026-03-28  
**Scope:** GPU thinking-budget loop, ARC3 action diversity, live ARC-AGI-3 verification  
**Status:** Implemented and validated

## Code Changes

- [gpu_task_dispatch.cu](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/cuda/gpu_task_dispatch.cu)
  - outer thinking-budget loop is live
  - action-history suppression is GPU-side
  - ARC3 candidate scoring now adds frame-conditioned action priors on device
- [device_functions.cuh](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/cuda/device_functions.cuh)
  - stronger ARC3 frame-to-action mapping
  - added `arc3_action_prior_device(...)`
- [action_embedding_loader.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/knowledgeverse/action_embedding_loader.py)
  - moved action-type differentiation into low semantic dims
  - removed the accidental reversal/parameter markers that were colliding with frame symmetry features
- [gpu_task_dispatch.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/knowledgeverse/gpu_task_dispatch.py)
  - CPU reference now matches the E.6 GPU semantics exactly
- [arc_agi_3.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/arc_agi_3.py)
  - packs `thinking_budget`, `action_history`, and `ternary_signal`

## Validation

- Focused suite:
  - `/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -m pytest -q tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py`
  - result: `23 passed in 3.85s`
- hygiene:
  - `git diff --check` clean

## Root Cause Fixed

The E.5 fixation was not just “insufficient budget.” The real collision was semantic:

- ARC3 frame encoder uses dims `14..31` for symmetry/transition/occupancy features
- action embeddings were also using tail dims like `16..18` for reversal/parameter markers
- this accidentally gave `ACTION7` / undo free cosine alignment on ordinary frames

E.6 removes that collision and adds explicit frame-conditioned movement/interaction priors on GPU.

## Offline Benchmark

Summary:
- [summary.json](/K3D/Knowledge3D.local/logs/phase_e_20260328_174652/summary.json)

Results:
- Synthetic: `10/10`
- MMLU: `15/50` (`30.0%`)
- ARC3 synthetic: `20/20`

ARC3 synthetic action distribution:
- `ACTION1`: `8`
- `ACTION2`: `3`
- `ACTION3`: `7`
- `ACTION4`: `2`

This is the key change versus E.5:
- E.5 ARC3 synthetic collapsed to a single action
- E.6 ARC3 synthetic spreads across 4 movement actions

## Live ARC-AGI-3

Probe run, 10 actions:
- log: [/K3D/Knowledge3D.local/logs/arc3_live_e6_probe_tn36_20260328.jsonl](/K3D/Knowledge3D.local/logs/arc3_live_e6_probe_tn36_20260328.jsonl)
- scorecard: `https://three.arcprize.org/scorecards/f96e5184-a236-4fdd-b188-2c40ebf9e5b1`
- result: `NOT_FINISHED`, `0` levels completed
- action distribution:
  - `ACTION1`: `3`
  - `ACTION2`: `2`
  - `ACTION3`: `2`
  - `ACTION4`: `2`
  - `ACTION5`: `1`

Full run, 80 actions:
- log: [/K3D/Knowledge3D.local/logs/arc3_live_e6_full_tn36_20260328.jsonl](/K3D/Knowledge3D.local/logs/arc3_live_e6_full_tn36_20260328.jsonl)
- scorecard: `https://three.arcprize.org/scorecards/0c3f7279-2ce6-4d12-b02d-2155c069b910`
- result: `NOT_FINISHED`, `0` levels completed
- action distribution:
  - `ACTION1`: `18`
  - `ACTION2`: `17`
  - `ACTION3`: `17`
  - `ACTION4`: `17`
  - `ACTION5`: `11`

## Comparison vs E.5

Previous live baseline on the same family of run:
- one sample game log was `{'ACTION5': 80}`
- live batch summary showed global fixation on a single action

E.6 live behavior:
- no single-action fixation
- 5 unique actions in live play
- sovereign GPU pipeline still end-to-end
- no LLM calls

## Honest Boundary

E.6 fixes the brain-pathology that made ARC3 degenerate into one action. It does **not** solve ARC-AGI-3 yet.

Current status:
- architecture proof: **yes**
- fixation broken: **yes**
- action diversity: **yes**
- level completion: **not yet**

## Best Next Target

The next step should focus on **policy quality**, not architecture proof:

- richer frame encoder semantics for target/object salience
- state-delta learning from previous action outcome
- explicit “perform vs move” readiness tied to local cursor/object overlap
- larger action-space handling after 7-slot movement/interaction baseline is stable
