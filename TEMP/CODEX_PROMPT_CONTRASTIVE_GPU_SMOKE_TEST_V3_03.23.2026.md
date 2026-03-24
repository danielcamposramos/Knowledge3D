# Codex Prompt: CONTRASTIVE GPU SMOKE TEST v3 — Final Step Only

**Date:** 2026-03-23
**Priority:** HIGH
**Context:** v2 PROVED the GPU gradient path works (steps 1-3 passed, weight delta non-zero). Step 4 failed on wrong config constructor. This v3 runs ONLY step 4 with the correct constructor, then launches the warm benchmark.

---

## RUN THIS EXACTLY

Steps 1-3 are PROVEN. The sovereign GPU contrastive gradient path WORKS. Skip straight to the full contrastive flow test:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

python3 -c "
import numpy as np
from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter

print('=== CONTRASTIVE GPU SMOKE TEST v3 — Full Flow ===')
print()

# Create adapter with production dims (math specialist: 128x128, rank 16)
adapter = SelfUpdatingAdapter(shape=(128, 128), rank=16, specialist_name='smoke_math')

# Create swarm with REAL production config (from knowledgeverse.py:484)
print('[1] Creating AdaptiveSwarmTRM with production SwarmConfig...')
config = SwarmConfig(base_dims=128, min_dims=64, base_learning_rate=0.001, specialist_learning_rate=0.002)
swarm = AdaptiveSwarmTRM(config)

# Register specialist the production way
print('[2] Registering test specialist...')
swarm.register_specialist('smoke_math', required_dims=128, rank=16)

# Swap in our adapter (specialists dict lives on the MatryoshkaTRM base)
swarm.base.specialists['smoke_math']['adapter'] = adapter

# Build contrastive pairs
pos_pairs = [(np.random.randn(128).astype(np.float32), np.random.randn(128).astype(np.float32)) for _ in range(5)]
neg_pairs = [(np.random.randn(128).astype(np.float32), np.random.randn(128).astype(np.float32)) for _ in range(3)]

# Snapshot weights
A_before = adapter.A.copy()
B_before = adapter.B.copy()

# Run full contrastive training
print('[3] Running train_specialist_contrastive...')
result = swarm.train_specialist_contrastive('smoke_math', pos_pairs, neg_pairs, learning_rate=0.001)
print(f'    positive_steps: {result[\"positive_steps\"]}')
print(f'    negative_steps: {result[\"negative_steps\"]}')
print(f'    avg_loss: {result[\"avg_loss\"]:.6f}')

# Verify weights changed
A_delta = np.linalg.norm(adapter.A - A_before)
B_delta = np.linalg.norm(adapter.B - B_before)
print(f'    A weight delta: {A_delta:.6f}')
print(f'    B weight delta: {B_delta:.6f}')
assert result['positive_steps'] > 0, 'NO positive steps completed'
assert A_delta > 0 or B_delta > 0, 'WEIGHTS DID NOT CHANGE after contrastive training'

print()
print('=== FULL CONTRASTIVE FLOW PASSED — SOVEREIGN GPU PATH VERIFIED ===')
"
```

## IF IT PASSES

Launch the warm 35% benchmark IMMEDIATELY:

```bash
nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_warm_contrastive_sovereign_35pct_03.23.2026.log 2>&1 &
echo "Warm 35% sovereign contrastive benchmark launched. PID: $!"
echo "Log: /tmp/k3d_warm_contrastive_sovereign_35pct_03.23.2026.log"
```

## IF IT FAILS

Print FULL traceback. Do NOT add fallbacks. Report.
