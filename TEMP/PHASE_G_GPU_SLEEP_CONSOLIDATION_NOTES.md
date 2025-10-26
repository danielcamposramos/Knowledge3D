# Phase G — Sovereign Sleep Consolidation Notes

## Summary
- `knowledge3d/cranium/sleep_time_consolidator.py` no longer calls sklearn or NumPy kernels.
- All cluster refinement/pruning runs on the GPU:
  - CuPy drives the vector math (resides on device).
  - Cosine cohesion metrics use the modular RPN PTX executor.
- Retains adapters in-memory while consolidation executes so `AdaptiveSwarmTRM` can continue training without reloads.

## Operational Flow
1. `scripts/phase_g_gpu_training_session.py` loads `AdaptiveSwarmTRM` once, trains each specialist via `run_training(..., swarm=swarm)`.
2. After each training block:
   - Waits for a cool-down window (default 300 s).
   - Calls `SleepTimeConsolidator.consolidate()` on the active RPN embeddings (GPU path).
3. Consolidator stages:
   - GPU K-means (CuPy) with RPN-powered cohesion scores.
   - GPU redundancy pruning (vector similarity + centroid merge).
   - Writes back to `RPNEmbeddingEngine`, updates vocab statistics, optional metrics JSONL.

## Verifying Sovereign Execution
```bash
# Run a single specialist + consolidation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists multimodal --cooldown-seconds 120

# Monitor GPU activity
watch -n 2 nvidia-smi
```
Expected:
- Training + consolidation keeps GPU memory ≈8–10 GB.
- GPU utilisation peaks (70–95%) during both training and consolidation.
- CPU usage stays low (≤20%) apart from orchestration.

## House/Galaxy Persistence Checklist
After the session completes:
1. `tail /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl`
   - Check `cohesion_before`, `cohesion_after`, `merged_pairs`.
2. `ls -lh /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl`
   - Confirm timestamp updated by consolidator.
3. Verify materialised artifacts (books/trees/diary entries) in the House GLB via:
   ```
   python -m knowledge3d.tools.house_memory_builder --status
   ```
4. Optional: load the viewer (tablet) to inspect newly spawned Garden/Mirror objects.

## Notes
- The consolidator samples RPN similarities for validation; adjust sample count via `SLEEP_RPN_SAMPLE_SIZE` env if needed.
- `phase_g_gpu_training_session.py --skip-sleep` remains for debugging but **must not** be used for production runs.

