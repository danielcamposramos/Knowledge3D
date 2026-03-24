# Codex Sovereign NumPy Audit

**Date:** 2026-03-24  
**Prompt:** `TEMP/CODEX_PROMPT_SOVEREIGN_GAME_WORLD_MIGRATION_03.24.2026.md`

## Audit Command

```bash
rg -n "import numpy|from numpy|np\." \
  knowledge3d/cranium/trm_adapters.py \
  knowledge3d/cranium/adaptive_swarm.py \
  knowledge3d/cranium/ptx_runtime/rpn_math_core.py \
  knowledge3d/cranium/bridges/sovereign_bridges.py \
  knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py \
  knowledge3d/cranium/spatial_sovereign/morton_octree.py \
  knowledge3d/cranium/spatial_sovereign/frustum.py \
  knowledge3d/cranium/spatial_sovereign/led_pathfinder.py \
  knowledge3d/cranium/matryoshka_trm.py \
  knowledge3d/cranium/router_specialist.py
```

## Findings

The audit confirmed Daniel’s diagnosis:

- the three immediate sovereign leak files were:
  - `knowledge3d/cranium/trm_adapters.py`
  - `knowledge3d/cranium/adaptive_swarm.py`
  - `knowledge3d/cranium/ptx_runtime/rpn_math_core.py`
- the surrounding bridge/spatial/base files still carry larger audited debt:
  - `knowledge3d/cranium/bridges/sovereign_bridges.py`
  - `knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py`
  - `knowledge3d/cranium/spatial_sovereign/morton_octree.py`
  - `knowledge3d/cranium/spatial_sovereign/frustum.py`
  - `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py`
  - `knowledge3d/cranium/matryoshka_trm.py`
  - `knowledge3d/cranium/router_specialist.py`

That means Phase 1 can honestly claim only this:

- **the live specialist-learning leak path is cut**
- **the wider spatial/bridge/base migration is still pending**

## Classification of the Three Target Files

| File | Old NumPy usage | Classification | Phase 1 action |
|------|------------------|----------------|----------------|
| `rpn_math_core.py` | `np.asarray`, `np.array`, `np.copyto`, `np.float32` | host↔device transfer prep | replaced with `ctypes` staging + generic array-like coercion |
| `trm_adapters.py` | `np.random.randn`, `np.zeros_like`, `np.copyto`, `np.ascontiguousarray`, `A @ B`, transpose helpers, `.npz` save/load | host tensor ownership, gradient staging, checkpoint serialization | replaced with `HostTensorF32`, PTX math via `RPNMathCore`, zip+binary checkpoint format |
| `adaptive_swarm.py` | `np.linalg.norm`, `np.outer`, `np.pad`, placeholder random gradients, A/B fallback math | contrastive loss/gradient math, fake host-side training | replaced contrastive path with PTX-backed helpers; removed fake placeholder training paths via fail-fast |

## Phase 1 Result

Current grep for the three immediate files:

```bash
rg -n "import numpy|from numpy|np\." \
  knowledge3d/cranium/trm_adapters.py \
  knowledge3d/cranium/adaptive_swarm.py \
  knowledge3d/cranium/ptx_runtime/rpn_math_core.py
```

Result:

- **zero matches**

## Important Boundary

This audit does **not** say NumPy is gone from the whole cranium.

It says:

1. the three active specialist-learning files are now NumPy-free,
2. the old fake-random host training stubs were removed instead of preserved,
3. the larger migration still remains in the spatial/bridge/base layers already identified by the audit.
