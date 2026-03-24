# Codex Prompt: CONTRASTIVE GPU SMOKE TEST v2 — Fixed Constructor

**Date:** 2026-03-23
**Priority:** CRITICAL
**Context:** v1 smoke test failed at adapter creation — wrong constructor signature. `SelfUpdatingAdapter` takes `shape=(D,D)`, NOT `input_dim/output_dim`. The CUDA memcpy path was never reached. This v2 fixes the constructor and retries.

---

## RUN THIS EXACTLY

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

python3 -c "
import numpy as np
from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmManager, AdaptiveSwarmConfig
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter

print('=== CONTRASTIVE GPU SMOKE TEST v2 ===')
print()

# Step A: Create adapter with REAL production dimensions
# From knowledgeverse.py ADAPTIVE_SWARM_SPECS: math=(128,16), chat=(64,16)
print('[1] Creating SelfUpdatingAdapter(shape=(128,128), rank=16) — math specialist dims...')
adapter = SelfUpdatingAdapter(shape=(128, 128), rank=16, specialist_name='smoke_test')
print(f'    Adapter type: {type(adapter).__name__}')
print(f'    Has apply_gradient: {hasattr(adapter, \"apply_gradient\")}')
print(f'    Has A matrix: {hasattr(adapter, \"A\")} shape={adapter.A.shape if hasattr(adapter, \"A\") else \"N/A\"}')
print(f'    Has B matrix: {hasattr(adapter, \"B\")} shape={adapter.B.shape if hasattr(adapter, \"B\") else \"N/A\"}')

# Step B: Snapshot weights BEFORE gradient
A_before = adapter.A.copy()
B_before = adapter.B.copy()

# Step C: Apply gradient through the SOVEREIGN path
print('[2] Applying gradient via adapter.apply_gradient()...')
gradient = np.random.randn(128, 128).astype(np.float32) * 0.01
adapter.apply_gradient(gradient, lr=0.001)
print('    GPU gradient application: PASSED')

# Step D: Verify weights CHANGED
A_delta = np.linalg.norm(adapter.A - A_before)
B_delta = np.linalg.norm(adapter.B - B_before)
print(f'    A weight delta: {A_delta:.6f}')
print(f'    B weight delta: {B_delta:.6f}')
assert A_delta > 0 or B_delta > 0, 'WEIGHTS DID NOT CHANGE'
print('    Weight change verification: PASSED')

# Step E: Test _apply_adapter_gradient static method
print('[3] Testing _apply_adapter_gradient static method...')
AdaptiveSwarmManager._apply_adapter_gradient(adapter, gradient, 0.001)
print('    _apply_adapter_gradient: PASSED')

# Step F: Full contrastive training flow
print('[4] Testing full train_specialist_contrastive flow...')
config = AdaptiveSwarmConfig(specialist_dims=128)
swarm = AdaptiveSwarmManager(config)

swarm.base.specialists['test'] = {
    'adapter': adapter,
    'dims': 128,
    'activated': True,
}
swarm.specialist_steps['test'] = 0

pos_pairs = [(np.random.randn(128).astype(np.float32), np.random.randn(128).astype(np.float32)) for _ in range(5)]
neg_pairs = [(np.random.randn(128).astype(np.float32), np.random.randn(128).astype(np.float32)) for _ in range(3)]

result = swarm.train_specialist_contrastive('test', pos_pairs, neg_pairs, learning_rate=0.001)
print(f'    positive_steps: {result[\"positive_steps\"]}')
print(f'    negative_steps: {result[\"negative_steps\"]}')
print(f'    avg_loss: {result[\"avg_loss\"]:.6f}')
assert result['positive_steps'] > 0, 'NO positive steps completed'
print('    Full contrastive flow: PASSED')

print()
print('=== ALL SMOKE TESTS PASSED — GPU CONTRASTIVE PATH IS SOVEREIGN ===')
"
```

## RULES

Same as v1:
1. Do NOT add try/except around `apply_gradient`
2. Do NOT suggest fallbacks
3. Report EXACT output — every line
4. If it PASSES: launch the warm 35% benchmark:

```bash
nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_warm_contrastive_sovereign_35pct_03.23.2026.log 2>&1 &
echo "PID: $!"
```

5. If it FAILS: print FULL traceback, do NOT fix anything, just report
