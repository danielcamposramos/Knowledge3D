# Knowledge3D Sovereign Test Log

## 2025-10-15 — Phase 0 (Step 12 focus)

- **Environment**: `k3d-cranium` (CUDA 12.4, Python 3.10.18)
- **GPU config**: RTX 3060 dedicated (KDE on Ryzen 5 5600G iGPU)
- **Command**:
  ```bash
  . $HOME/miniconda3/etc/profile.d/conda.sh \
    && conda activate /K3D/Knowledge3D.local/envs/k3d-cranium \
    && export PYTHONPATH=. \
    && export K3D_PTX_STRICT=1 \
    && export K3D_FORCE_PTX_FUSE=1 \
    && export CUDA_VISIBLE_DEVICES=0 \
    && pytest tests/test_step12_*.py -v --tb=short
  ```
- **Artifacts**: `reports/phase0_results.txt`

### Summary
- 65 tests collected; **57 failures**, 8 passed.
- Failures concentrated in:
  - `tests/test_step12_action_buffer_integration.py`
  - `tests/test_step12_cognitive_pipeline.py`
  - `tests/test_step12_dynamic_lod.py`
- Primary failure mode: mocked `ThinkingTagBridge` lacks required PTX-backed methods/fields (state trace, action buffer, dynamic LOD) when imported via `knowledge3d.cranium.ptx_runtime.thinking_tag_bridge`.
- Next step: augment bridge fixtures/mocks to satisfy tests or route through sovereign PTX implementations.

