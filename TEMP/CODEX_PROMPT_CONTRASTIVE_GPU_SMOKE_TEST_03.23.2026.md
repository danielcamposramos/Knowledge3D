# Codex Prompt: VERIFY GPU Contrastive Path — Smoke Test to Completion

**Date:** 2026-03-23
**Priority:** CRITICAL — This is the SINGLE MOST IMPORTANT validation right now
**Context:** The CPU fallback has been REMOVED from `_apply_adapter_gradient`. The `loader.py` argtypes fix is in place. We have NEVER seen contrastive training succeed on the sovereign GPU path. This smoke test determines whether the fix works.

---

## WHAT TO DO

Run these steps IN ORDER. Do NOT skip any. Do NOT add fallbacks. Do NOT catch exceptions — let them propagate.

### Step 1: Verify the CPU Fallback is GONE

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
grep -n "_apply_gradient_cpu\|require_gpu" knowledge3d/cranium/adaptive_swarm.py
```

**EXPECTED:** ZERO matches. If ANY match is found, STOP and report — the fallback was not properly removed.

### Step 2: Run the Contrastive GPU Smoke Test

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

python3 -c "
import numpy as np
from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmManager, AdaptiveSwarmConfig
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter

print('=== CONTRASTIVE GPU SMOKE TEST ===')
print()

# Step A: Create a real adapter with GPU path
print('[1] Creating SelfUpdatingAdapter (128x128)...')
adapter = SelfUpdatingAdapter(input_dim=128, output_dim=128)
print(f'    Adapter type: {type(adapter).__name__}')
print(f'    Has apply_gradient: {hasattr(adapter, \"apply_gradient\")}')
print(f'    Has A/B matrices: {hasattr(adapter, \"A\") and hasattr(adapter, \"B\")}')

# Step B: Snapshot weights BEFORE gradient
A_before = adapter.A.copy() if hasattr(adapter, 'A') else None
B_before = adapter.B.copy() if hasattr(adapter, 'B') else None

# Step C: Apply gradient through the SOVEREIGN path
print('[2] Applying gradient via adapter.apply_gradient()...')
gradient = np.random.randn(128, 128).astype(np.float32) * 0.01
adapter.apply_gradient(gradient, lr=0.001)
print('    GPU gradient application: PASSED')

# Step D: Verify weights actually CHANGED
if A_before is not None:
    A_delta = np.linalg.norm(adapter.A - A_before)
    B_delta = np.linalg.norm(adapter.B - B_before)
    print(f'    A weight delta: {A_delta:.6f}')
    print(f'    B weight delta: {B_delta:.6f}')
    assert A_delta > 0 or B_delta > 0, 'WEIGHTS DID NOT CHANGE — gradient had no effect'
    print('    Weight change verification: PASSED')

# Step E: Test through AdaptiveSwarmManager._apply_adapter_gradient
print('[3] Testing _apply_adapter_gradient static method...')
AdaptiveSwarmManager._apply_adapter_gradient(adapter, gradient, 0.001)
print('    _apply_adapter_gradient: PASSED')

# Step F: Test with contrastive training flow
print('[4] Testing full train_specialist_contrastive flow...')
config = AdaptiveSwarmConfig(specialist_dims=128)
swarm = AdaptiveSwarmManager(config)

# Register a specialist manually
swarm.base.specialists['test'] = {
    'adapter': adapter,
    'dims': 128,
    'activated': True,
}
swarm.specialist_steps['test'] = 0

# Build fake positive/negative pairs
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

### Step 3: Interpret Results

**IF ALL TESTS PASS:**
Print the full output and report SUCCESS. The `loader.py` argtypes fix resolved the TypeError. Contrastive training is now sovereign GPU. Proceed to run a warm 35% benchmark:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_warm_contrastive_sovereign_35pct_03.23.2026.log 2>&1 &

echo "Warm 35% benchmark launched. PID: $!"
echo "Log: /tmp/k3d_warm_contrastive_sovereign_35pct_03.23.2026.log"
```

**IF ANY TEST FAILS WITH TypeError:**
This means the `loader.py` argtypes fix is INSUFFICIENT. Print the FULL stack trace. Do NOT add a fallback. Do NOT catch the exception. The stack trace IS the diagnostic — it will show EXACTLY which ctypes call fails and with what arguments. Report the trace so we can fix the REAL issue.

**IF ANY TEST FAILS WITH OTHER ERROR:**
Print the full stack trace and report. Could be an import issue, missing GPU context, adapter initialization problem, etc. Each error has a SPECIFIC fix — none of which is "fall back to CPU."

---

## RULES

1. Do NOT add try/except around `apply_gradient` — if it fails, we NEED the traceback
2. Do NOT import or reference `_apply_gradient_cpu` — it does not exist anymore
3. Do NOT suggest "graceful degradation" or "resilience" — this is SOVEREIGN
4. Do NOT modify `adaptive_swarm.py` or `loader.py` — run the test AS-IS
5. Report the EXACT output — every print line, every error line

---

## WHY THIS MATTERS

Three consecutive benchmark runs have shown `trained: false` for ALL four specialists because of `TypeError: Don't know how to convert parameter 2` in the CUDA driver memcpy call. The `loader.py` argtypes fix SHOULD resolve this, but the CPU fallback masked the result in the last run. This smoke test is the FIRST honest verification of the sovereign GPU contrastive path.

If this passes, contrastive learning is UNBLOCKED for the first time. Sleep-time will actually train specialist adapters. Benchmark scores should improve over successive runs as the TRM learns from its own health journal.
