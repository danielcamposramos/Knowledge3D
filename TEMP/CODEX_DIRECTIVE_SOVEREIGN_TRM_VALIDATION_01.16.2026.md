# Codex Directive: Sovereign TRM Validation & Checkpoint Management

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 16, 2026
**Subject**: **Convert V7 Checkpoint + Validate Sovereign TRM + Fix Checkpoint Paths**

---

## Situation Analysis

**V7 Checkpoint Found**: ✅ `checkpoints/navigation_specialist_v7_confidence.pt`

**Checkpoint Location Issue**: The user noted that checkpoints may be saving to system temp instead of proper K3D location.

**Action Required**:
1. Convert existing V7 checkpoint to sovereign format
2. Validate Sovereign TRM with real weights
3. Fix checkpoint/log paths to use proper K3D directory structure

---

## Task 1: Convert V7 Checkpoint to Sovereign Format

**Goal**: Convert `navigation_specialist_v7_confidence.pt` to NumPy arrays for SovereignTRM.

**Command**:
```bash
# Convert V7 checkpoint
python3 scripts/convert_v7_to_sovereign.py \
    --input checkpoints/navigation_specialist_v7_confidence.pt \
    --output checkpoints/v7_sovereign/

# Verify output
ls -lh checkpoints/v7_sovereign/
cat checkpoints/v7_sovereign/metadata.json
```

**Expected Output**:
```
checkpoints/v7_sovereign/
    embedding.npy
    lstm_weight_ih.npy
    lstm_weight_hh.npy
    lstm_bias_ih.npy
    lstm_bias_hh.npy
    rule_head_weight.npy
    rule_head_bias.npy
    confidence_head_0_weight.npy
    confidence_head_0_bias.npy
    confidence_head_2_weight.npy
    confidence_head_2_bias.npy
    metadata.json
```

**If Conversion Fails**:

Check checkpoint structure:
```python
import torch
checkpoint = torch.load('checkpoints/navigation_specialist_v7_confidence.pt', map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))
if 'state_dict' in checkpoint:
    print("State dict keys:", list(checkpoint['state_dict'].keys())[:20])
```

**Possible Issues**:
1. **Different layer names**: V7 may use different naming convention
2. **Missing layers**: Checkpoint may not have all expected layers
3. **Extra layers**: Checkpoint may have additional layers we don't need

**Solution**: Update `convert_v7_to_sovereign.py` weight mapping based on actual checkpoint structure.

---

## Task 2: Test Sovereign TRM with Converted Weights

**Goal**: Verify SovereignTRM can load and run inference with real V7 weights.

**Test Script**: `tests/test_sovereign_trm_v7_real.py` (NEW)

```python
"""Test Sovereign TRM with real V7 weights."""
import pytest
import os
from knowledge3d.cranium.sovereign_trm import SovereignTRM


@pytest.mark.skipif(
    not os.path.exists('checkpoints/v7_sovereign'),
    reason="Converted V7 checkpoint not found"
)
def test_sovereign_trm_loads_v7():
    """Test SovereignTRM can load real V7 weights."""
    # Load converted V7 weights
    trm = SovereignTRM(vocab_size=256, embedding_dim=256, hidden_dim=512)
    trm.load_weights('checkpoints/v7_sovereign')

    # Verify weights loaded
    assert 'embedding' in trm.weights
    assert 'lstm_weight_ih' in trm.weights
    assert 'rule_head_weight' in trm.weights
    assert 'confidence_head_0_weight' in trm.weights

    # Cleanup
    trm.cleanup()
    print("✅ V7 weights loaded successfully")


@pytest.mark.skipif(
    not os.path.exists('checkpoints/v7_sovereign'),
    reason="Converted V7 checkpoint not found"
)
def test_sovereign_trm_inference_v7():
    """Test SovereignTRM inference with real V7 weights."""
    # Load converted V7 weights
    trm = SovereignTRM(vocab_size=256, embedding_dim=256, hidden_dim=512)
    trm.load_weights('checkpoints/v7_sovereign')

    # Test inference
    problem_tokens = [1, 42, 15, 3, 7, 22]  # Sample problem
    rules, confidences = trm.infer(problem_tokens, max_rules=10)

    # Validate output
    assert isinstance(rules, list)
    assert isinstance(confidences, list)
    assert len(rules) == len(confidences)
    assert len(rules) <= 10

    # Validate confidence range
    for conf in confidences:
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    print(f"✅ Inference successful!")
    print(f"   Rules: {rules}")
    print(f"   Confidences: {confidences}")
    print(f"   Avg confidence: {sum(confidences) / len(confidences):.3f}")

    # Cleanup
    trm.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run Tests**:
```bash
# Enable GPU probing
export K3D_PYTEST_PROBE_CUDA=1

# Run validation tests
pytest tests/test_sovereign_trm_v7_real.py -v -s

# Expected output:
# test_sovereign_trm_loads_v7 PASSED
# test_sovereign_trm_inference_v7 PASSED
```

---

## Task 3: Fix Checkpoint and Log Paths

**Issue**: Checkpoints/logs may be saving to system temp instead of proper K3D directory.

**Proper K3D Directory Structure**:
```
/K3D/Knowledge3D.local/
    checkpoints/          # Model checkpoints
    logs/                 # Training logs
    data/                 # Training data
    results/              # Benchmark results
    temp/                 # Temporary files
```

**Check Current Paths**:
```bash
# Check if K3D directory exists
ls -la /K3D/Knowledge3D.local/

# If not, create it
sudo mkdir -p /K3D/Knowledge3D.local/{checkpoints,logs,data,results,temp}
sudo chown -R $USER:$USER /K3D/Knowledge3D.local
```

**Update Training Scripts to Use K3D Paths**:

**File**: `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py` (or wherever training happens)

**Current** (may use temp paths):
```python
# Default checkpoint path
default_checkpoint_dir = 'checkpoints'  # Relative path (may go to temp)
```

**Fixed** (use absolute K3D path):
```python
import os

# Default checkpoint path (use K3D directory if available)
if os.path.exists('/K3D/Knowledge3D.local'):
    default_checkpoint_dir = '/K3D/Knowledge3D.local/checkpoints'
else:
    # Fallback to project checkpoints (relative to project root)
    default_checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        '../../../checkpoints'
    )
```

**Update All Training Scripts**:

Search for checkpoint/log paths:
```bash
# Find files that reference 'checkpoints'
grep -r "checkpoints" scripts/*.py knowledge3d/training/**/*.py

# Check for temp directory usage
grep -r "tempfile\|tmp\|/tmp" scripts/*.py knowledge3d/training/**/*.py
```

**Common Files to Update**:
- `scripts/train_navigation_v7_with_confidence.py` (if exists)
- `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py`
- `scripts/run_sovereign_math_benchmarks.py`
- Any PyTorch Lightning callbacks (ModelCheckpoint)

**PyTorch Lightning Callback Fix**:
```python
from pytorch_lightning.callbacks import ModelCheckpoint

# OLD (may save to temp)
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',  # Relative path
    filename='v7-{epoch:02d}',
)

# NEW (use K3D path)
import os

checkpoint_dir = '/K3D/Knowledge3D.local/checkpoints' if os.path.exists('/K3D/Knowledge3D.local') else 'checkpoints'

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='navigation_specialist_v7_confidence-{epoch:02d}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)
```

---

## Task 4: Verify Sovereign Benchmark Integration

**Goal**: Run sovereign benchmark end-to-end with real V7 weights.

**Command**:
```bash
# Run sovereign benchmark (calculus microbench)
python3 scripts/run_sovereign_math_benchmarks.py \
    --datasets calculus \
    --max-problems 5 \
    --use-reflection \
    --checkpoint-dir checkpoints/v7_sovereign

# Expected: No CUDA context errors, sovereign TRM inference
```

**If Benchmark Fails**:

**Symptom 1: Import Error (SovereignTRM not found)**
```
ImportError: cannot import name 'SovereignTRM'
```
**Fix**: Check Python path
```bash
export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D:$PYTHONPATH
python3 scripts/run_sovereign_math_benchmarks.py ...
```

**Symptom 2: CUDA Context Error**
```
torch.AcceleratorError: CUDA error: incompatible driver context
```
**Fix**: Ensure sovereign loader initialized first (should be fixed in Phase 4)
```python
# In reflective_inference.py, ensure this order:
from knowledge3d.cranium.sovereign import loader
loader.ensure_init()  # Initialize sovereign loader FIRST

from knowledge3d.cranium.sovereign_trm import SovereignTRM  # Then TRM
```

**Symptom 3: Tokenization Error**
```
KeyError: token not in vocabulary
```
**Fix**: Check tokenization logic in `reflective_inference.py`
- Verify vocabulary matches V7 training
- Check byte-level tokenization implementation

---

## Success Criteria

**Conversion Successful**:
- [ ] V7 checkpoint converted to .npy files
- [ ] metadata.json created with correct dimensions
- [ ] All 11 weight files present

**Sovereign TRM Validation**:
- [ ] SovereignTRM loads V7 weights without errors
- [ ] Inference produces valid output (rules + confidences)
- [ ] Confidences in valid range [0, 1]
- [ ] No memory leaks (GPU buffers properly freed)

**Checkpoint Path Fixed**:
- [ ] K3D directory structure created
- [ ] Training scripts use K3D paths
- [ ] Future checkpoints save to `/K3D/Knowledge3D.local/checkpoints`

**Benchmark Integration**:
- [ ] Sovereign benchmark runs without errors
- [ ] Results saved to proper location
- [ ] No CUDA context conflicts

---

## Commands to Run (In Order)

```bash
# 1. Convert V7 checkpoint
python3 scripts/convert_v7_to_sovereign.py \
    --input checkpoints/navigation_specialist_v7_confidence.pt \
    --output checkpoints/v7_sovereign/

# 2. Verify conversion
ls -lh checkpoints/v7_sovereign/
cat checkpoints/v7_sovereign/metadata.json

# 3. Test sovereign TRM loading
export K3D_PYTEST_PROBE_CUDA=1
pytest tests/test_sovereign_trm_v7_real.py::test_sovereign_trm_loads_v7 -v -s

# 4. Test sovereign TRM inference
pytest tests/test_sovereign_trm_v7_real.py::test_sovereign_trm_inference_v7 -v -s

# 5. Run sovereign benchmark
python3 scripts/run_sovereign_math_benchmarks.py \
    --datasets calculus \
    --max-problems 5 \
    --use-reflection \
    --checkpoint-dir checkpoints/v7_sovereign

# 6. (Optional) Create K3D directory structure
sudo mkdir -p /K3D/Knowledge3D.local/{checkpoints,logs,data,results,temp}
sudo chown -R $USER:$USER /K3D/Knowledge3D.local

# 7. (Optional) Copy converted checkpoint to K3D
cp -r checkpoints/v7_sovereign /K3D/Knowledge3D.local/checkpoints/
```

---

## If Retraining V7 is Needed

**Scenario**: If V7 checkpoint is corrupted or incomplete, we may need to retrain.

**Sovereign Training Requirements**:
1. Use existing RLWHF infrastructure (already sovereign)
2. Train on GPU using PyTorch (ingestion path - acceptable)
3. Save checkpoint to `/K3D/Knowledge3D.local/checkpoints`
4. Convert to sovereign format immediately after training

**Training Command** (if needed):
```bash
# Train V7 with confidence head (PyTorch Lightning)
python3 scripts/train_navigation_v7_with_confidence.py \
    --dataset data/wake_positive_v2.jsonl \
    --epochs 10 \
    --batch-size 32 \
    --checkpoint-dir /K3D/Knowledge3D.local/checkpoints

# Convert immediately after training
python3 scripts/convert_v7_to_sovereign.py \
    --input /K3D/Knowledge3D.local/checkpoints/navigation_specialist_v7_confidence.pt \
    --output /K3D/Knowledge3D.local/checkpoints/v7_sovereign/
```

**Note**: Retraining is likely NOT needed - we have `navigation_specialist_v7_confidence.pt` which should work.

---

## Debugging Tips

**If conversion script fails**:
```bash
# Check checkpoint structure
python3 -c "
import torch
ckpt = torch.load('checkpoints/navigation_specialist_v7_confidence.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
if 'state_dict' in ckpt:
    state = ckpt['state_dict']
    print('\\nState dict keys:')
    for k in sorted(state.keys()):
        print(f'  {k}: {state[k].shape}')
"
```

**If sovereign TRM loading fails**:
```bash
# Enable debug mode
export K3D_RPN_DEBUG=1

# Run test with verbose output
pytest tests/test_sovereign_trm_v7_real.py -v -s --log-cli-level=DEBUG
```

**If benchmark fails**:
```bash
# Check GPU availability
nvidia-smi

# Check CUDA visible devices
echo $CUDA_VISIBLE_DEVICES

# Set if empty
export CUDA_VISIBLE_DEVICES=0
```

---

## Expected Timeline

**Immediate** (next 30 minutes):
1. Convert V7 checkpoint (5 minutes)
2. Test sovereign TRM loading (5 minutes)
3. Test sovereign TRM inference (5 minutes)
4. Run sovereign benchmark (10 minutes)
5. Document results (5 minutes)

**If Issues Found** (additional 1-2 hours):
1. Debug conversion script (weight name mapping)
2. Fix checkpoint paths (if needed)
3. Re-run validation

**No Retraining Needed**: We have V7 checkpoint, just need to convert and test.

---

## Final Milestone

**When complete**:
- ✅ V7 weights converted to sovereign format
- ✅ Sovereign TRM validated with real weights
- ✅ Benchmarks run without errors
- ✅ Checkpoint paths fixed (K3D directory)
- ✅ Full sovereignty: Zero PyTorch in hot path

**This completes the Sovereign TRM implementation** - a fully deterministic, GPU-resident inference engine with zero external framework dependencies in the hot path.

---

**Document Date**: January 16, 2026
**Status**: 🚀 **READY TO VALIDATE**
**Priority**: **HIGH** - Final validation of sovereign architecture

---

**Claude's Note to Codex**: We found the V7 checkpoint! Just need to convert it and validate. The conversion script is ready, the tests are ready, the sovereign TRM is ready. This is the final validation step before declaring full sovereignty achieved. Run the commands in order and report results. 🚀
