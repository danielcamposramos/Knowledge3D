# CODEX: Fix Bug & Run Embodied Training

**Date:** December 10, 2025
**From:** Claude (Architecture)
**Priority:** Quick fix then run

---

## Bug to Fix

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`
**Line:** 286

**Current (Broken):**
```python
print(f"  [GRAMMAR] Loaded {self.grammar.count()} rules")
```

**Fix:**
```python
print(f"  [GRAMMAR] Loaded {len(self.grammar.rules)} rules")
```

GrammarGalaxy doesn't have a `count()` method — it has `rules` dict and `list_rules()`.

---

## Verification After Fix

Run this to verify all components work:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python -c "
from knowledge3d.cranium.embodied_agent import EmbodiedSovereignAgent
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline

print('=== Testing Embodied Agent ===')
agent = EmbodiedSovereignAgent(working_capacity=1024)
print(f'Drawing: {agent.drawing_galaxy.summary()}')
print(f'Grammar: {len(agent.grammar_galaxy.rules)} rules')
print(f'Math: {len(agent.math_galaxy.symbols)} symbols')
print(f'Thresholds: top_k={agent.thresholds.candidate_top_k}')

print('\\n=== Testing Pipeline with Embodied Agent ===')
pipeline = SovereignAIPipeline(embodied_agent=agent)
print('Pipeline created successfully!')

# Quick task test
grid = [[1,0,0],[0,1,0],[0,0,1]]
result = pipeline.process_task('test_diag', grid)
print(f'Test result: score={result.score:.2f}, type={result.program_type}')
print('=== All OK ===')
"
```

---

## Run Training with Embodied Agent

Once verification passes, start the 162-epoch training:

```bash
tmux new-session -d -s k3d_embodied "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 162 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/embodied_$(date +%Y%m%d_%H%M%S).log
'"

# Monitor
tmux attach -t k3d_embodied
# Or: tail -f /K3D/Knowledge3D.local/logs/embodied_*.log
```

---

## Expected Improvements

With embodied architecture:
1. **Galaxy loaded ONCE** — no more "[GrammarGalaxy] Loaded 747 rules" spam
2. **Working memory accumulates** — discoveries stored per task
3. **Consolidation triggers** — at 85% utilization or batch end
4. **Adaptive thresholds** — top_k=69 from Galaxy, not hardcoded

**Baseline:** 46.19% avg (stateless)
**Target:** 50%+ with embodied persistence

---

## Sovereignty Check

Verify no CPU fallbacks during run:
```bash
grep "CPU fallback\|numpy\|PTX.*fail" /K3D/Knowledge3D.local/logs/embodied_*.log
```

Should return empty (100% PTX).

---

**Fix the bug, verify, then launch. You have the conn.**
