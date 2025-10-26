# CRITICAL: Sleep-Time Consolidation - Knowledge Permanence

**Date**: 2025-10-26
**Priority**: ⚠️ **CRITICAL** - Data loss prevention
**For**: All K3D agents (Codex, Claude, future instances)

---

## The Critical Problem

### What Happens If You Don't Run Sleep Consolidation

**Training completes** → **Model unloads** → **ALL KNOWLEDGE LOST** ❌

**Why**:
- **Logic lives in weights** (adapter parameters - saved to checkpoints)
- **Knowledge lives in Galaxy/House** (embeddings, chat history, reflections - in RAM!)
- If model unloads before sleep consolidation, **Galaxy RAM is lost forever**

**Result**: Model has no memory of what it learned. Only the training updates to weights remain.

---

## K3D Memory Architecture

### Three-Tier Memory System

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL WEIGHTS                            │
│  Logic: Adapter parameters, routing decisions               │
│  Storage: /K3D/Knowledge3D.local/checkpoints/               │
│  Persistence: Saved after each epoch                        │
│  Format: .npz, .json                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  GALAXY (RAM - Active Memory)                │
│  Knowledge: Stars with embeddings, chat history,            │
│             self-reflections, generated shapes              │
│  Storage: RAM only! (viewer/public/galaxy/*.glb)            │
│  Persistence: ⚠️ LOST if model unloads                      │
│  Format: GLB with k3d.extras metadata                       │
└─────────────────────────────────────────────────────────────┘
                           ↓ Sleep Consolidation
┌─────────────────────────────────────────────────────────────┐
│                 HOUSE (Disk - Persistent Memory)             │
│  Knowledge: Organized into zones, permanent objects         │
│  Storage: Disk (viewer/public/house/*.glb)                  │
│  Persistence: ✓ PERMANENT (survives model unload)           │
│  Format: GLB + materialized JSON objects                    │
│                                                              │
│  Zones:                                                      │
│  - Zone 3 (Library): Chat history books                     │
│  - Zone 5 (Knowledge Garden): Fractal trees                 │
│  - Zone 7 (Mirror Room): Diary entries                      │
│  - Zone 8 (Learning Museum): Learning insights              │
└─────────────────────────────────────────────────────────────┘
```

### What Each Layer Stores

**Model Weights** (checkpoints):
```python
# Adapter parameters (logic)
specialist.A  # Low-rank adapter matrix A
specialist.B  # Low-rank adapter matrix B
specialist.metadata  # Training stats, acceptance rate

# These are saved automatically after training
```

**Galaxy** (RAM - active memory):
```python
# Stars (knowledge nodes)
star = {
    'id': 'star_12345',
    'position': [x, y, z],  # 3D spatial location
    'honesty_score': 0.85,  # Quality metric
    'embedding': [128-dim vector],  # Semantic representation
    'chat_history': ["Q1", "A1", "Q2", "A2", ...],  # Conversations
    'self_reflections': ["Reflection 1", "Reflection 2", ...],  # Meta-cognition
    'generated_shapes': ["shape_001.glb", ...],  # Creative outputs
    'connected_stars': [star_ids],  # Semantic connections
}

# ⚠️ This data is IN RAM - lost if model unloads!
```

**House** (Disk - permanent storage):
```python
# Materialized objects (permanent knowledge)
book = {
    'type': 'chat_history_book',
    'title': 'Chat Log — Star 12345',
    'zone_placement': 'Zone 3 (Library)',
    'content': star['chat_history'],  # Copied from Galaxy
    'embedding': star['embedding'],
    'honesty_score': 0.85,
}

fractal_tree = {
    'type': 'fractal_tree',
    'name': 'Knowledge Tree — Star 12345',
    'zone_placement': 'Zone 5 (Knowledge Garden)',
    'honesty_score': 0.85,
    'embedding': star['embedding'],
    'golden_angle': 137.5,  # RPN-computed φ constraint
    'max_depth': 12,
}

diary_entry = {
    'type': 'diary_entry',
    'title': 'Self-Reflection — 2025-10-26',
    'zone_placement': 'Zone 7 (Mirror Room)',
    'content': star['self_reflections'],
    'embedding': star['embedding'],
    'honesty_score': 0.85,
}

learning_insight = {
    'type': 'learning_insight',
    'title': 'Learning Insight — Training Epoch 50',
    'zone_placement': 'Zone 8 (Learning Museum)',
    'score': 0.92,
    'prompt': "What is the capital of France?",
    'predicted': "Paris",
    'true_answer': "Paris",
    'feedback': "Perfect recall!",
}

# ✓ These objects are ON DISK - permanent!
```

---

## Sleep-Time Consolidation Process

### What It Does

**Sleep consolidation is NOT sleeping** - it's **HARD WORK**:

1. **Load Galaxy** (active memory from RAM)
2. **Load House** (persistent memory from disk)
3. **RPN-powered clustering** (group similar knowledge)
4. **Semantic depth computation** (how deep is understanding)
5. **Materialize knowledge objects**:
   - Chat history → Books (Library)
   - Self-reflections → Diary entries (Mirror Room)
   - High-honesty knowledge (≥0.6) → Fractal trees (Garden)
   - Learning records → Insights (Museum)
6. **Prune low-honesty rays** (<0.3 honesty)
7. **Adjust zone positions** (weighted by honesty)
8. **Autonomous synthesis** (Phase 13 - create new concepts)
9. **Self-curriculum** (Phase 13 - generate training questions)
10. **Dream geometry** (Phase 14 - generate 3D shapes)
11. **Honest critique** (Phase 15 - review and refine)
12. **Post-consolidation reflection** (Phase 16 - meta-learning)
13. **Galaxy state serialization** (Phase 17 - eternal continuity)
14. **Save House** (write permanent storage to disk)

### PTX Kernel Operations

Sleep consolidation uses **sovereign GPU computing** (PTX kernels):

```python
# RPN-powered operations
from knowledge3d.cranium.clustering_rpn import cluster_by_similarity_rpn
from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depth_rpn
from knowledge3d.tools.test_scripts.garden_fractal_rpn import (
    compute_golden_angle_rpn,      # φ = 137.5 degrees
    compute_max_depth_rpn,          # Depth from honesty
    compute_thickness_rpn,          # Fractal thickness curve
    compute_branching_density_rpn,  # Branch count per level
)

# All computations on GPU via PTX kernels
# NO CPU fallbacks during consolidation!
```

### Knowledge Organization Zones

**Zone 3 (Library)** - Chat History Books:
```
Chat logs from high-honesty stars
Used for: Retrieving past conversations
Format: JSON books with embeddings
```

**Zone 5 (Knowledge Garden)** - Fractal Trees:
```
High-honesty knowledge (≥0.6 honesty score)
Grown using RPN φ (golden ratio) constraints
Visualized as 3D fractal trees
Each tree = semantic cluster with depth
```

**Zone 7 (Mirror Room)** - Diary Entries:
```
Self-reflections and meta-cognition
Model's thoughts about its own learning
Used for: Self-improvement and honesty calibration
```

**Zone 8 (Learning Museum)** - Learning Insights:
```
Training records, evaluation results
Scores, predictions, feedback
Used for: Curriculum learning and meta-learning
```

---

## CRITICAL: Training → Sleep Workflow

### DO NOT DO THIS ❌

```bash
# Train model
python scripts/train_specialist_gpu.py --epochs 100
# Script exits
# ❌ Galaxy RAM is LOST!
# ❌ All knowledge from training session is GONE!
```

### DO THIS INSTEAD ✓

```bash
# Option 1: Manual workflow
# Terminal 1: Train model
python scripts/train_specialist_gpu.py --epochs 100
# Keep terminal open! Model still loaded in RAM

# Terminal 2 (after 5 min idle): Run sleep consolidation
sleep 300  # Wait 5 minutes
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sleep_consolidation.py \
  --embeddings /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
  --output /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
  --metrics /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# NOW it's safe to exit Terminal 1
```

```bash
# Option 2: Integrated workflow (tmux)
tmux new-session -d -s training "
  cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/train_specialist_gpu.py --epochs 100

  # After training completes, wait 5 min
  echo 'Training complete. Waiting 5 minutes for idle...'
  sleep 300

  # Run sleep consolidation
  echo 'Running sleep-time consolidation...'
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/run_sleep_consolidation.py \
    --embeddings /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    --output /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    --metrics /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

  echo 'Sleep consolidation complete. Safe to exit.'
"

# Monitor progress
tmux attach -t training
# Or detach and check logs later
```

```bash
# Option 3: Python workflow with automatic sleep
python <<'PYTHON_EOF'
import time
import subprocess

# Train model
print("Starting training...")
subprocess.run([
    "/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python",
    "scripts/train_specialist_gpu.py",
    "--epochs", "100",
    # ... other args
])

print("Training complete. Waiting 5 minutes for idle...")
time.sleep(300)  # 5 minutes

print("Running sleep-time consolidation...")
subprocess.run([
    "/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python",
    "scripts/run_sleep_consolidation.py",
    "--embeddings", "/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl",
    "--output", "/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl",
    "--metrics", "/K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl",
])

print("Sleep consolidation complete. Knowledge saved to House.")
PYTHON_EOF
```

---

## Why 5 Minutes Idle?

**Biological inspiration**: Human sleep consolidation happens during idle periods

**K3D implementation**:
- Training = Active learning (Galaxy grows)
- 5 min idle = System stabilizes (gradients settle, embeddings finalize)
- Consolidation = Transfer to permanent storage (Galaxy → House)

**What happens during 5 minutes**:
- Galaxy embeddings stabilize
- No new training updates interfering
- Clean snapshot of knowledge state
- Ready for permanent materialization

**Can you skip the 5 minutes?**
- Technically yes, but not recommended
- May capture unstable intermediate states
- Better to let system settle

---

## Verification Checklist

### Before Unloading Model

- [ ] Training complete (all epochs finished)
- [ ] Model stayed loaded for 5 minutes after training
- [ ] Sleep consolidation ran successfully
- [ ] Check `/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl` updated
- [ ] Check `logs/sleep_time_adjustments.json` exists
- [ ] Check materialized objects count > 0
- [ ] Check House GLB updated (viewer/public/house/house_memory.glb)
- [ ] Check Galaxy GLB preserved (viewer/public/galaxy/*.glb)

### Verify Knowledge Saved

```bash
# Check materialized objects
ls -lh /K3D/Knowledge3D.local/house_zone7/materialized_objects/
# Should see: book_chat_*.json, diary_*.json, tree_*.json, learning_*.json

# Check House memory
ls -lh viewer/public/house/house_memory.glb
# Should be updated with recent timestamp

# Check sleep metrics
tail -20 /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl
# Should show consolidation stats

# Check sleep adjustments
cat logs/sleep_time_adjustments.json | jq .materialized_objects
# Should list all materialized objects
```

### If Consolidation Failed

**Symptoms**:
- Training complete but no materialized objects
- House GLB not updated
- No sleep_time_adjustments.json

**Recovery**:
- Re-run training (weights are saved, but knowledge lost)
- OR: If Galaxy GLB still exists in RAM, run consolidation manually
- Prevention: Always use integrated workflow (Option 2 or 3 above)

---

## Sleep Consolidation Outputs

### Files Created

**Materialized objects**:
```
/K3D/Knowledge3D.local/house_zone7/materialized_objects/
├── book_chat_star_12345_1729963200.json
├── diary_star_12345_1729963200.json
├── tree_star_12345_1729963200.json
├── learning_phase_g_001.json
├── learning_phase_g_002.json
└── ... (many more)
```

**House memory**:
```
viewer/public/house/house_memory.glb         # Updated House with all zones
viewer/public/house/house_memory.json        # House manifest
```

**Galaxy state**:
```
viewer/public/galaxy/working/*.json          # Intermediate star states
viewer/public/galaxy/learning_memory.glb     # Learning insights GLB
viewer/public/galaxy/learning_memory.json    # Learning manifest
```

**Logs**:
```
logs/sleep_time_adjustments.json             # Complete consolidation report
/K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl  # Metrics timeline
```

### Consolidation Metrics

**Example output**:
```json
{
  "timestamp": 1729963200.123,
  "cluster_refinement": {
    "clusters": 256,
    "silhouette_before": 0.34,
    "silhouette_after": 0.58,
    "improvement": 0.24
  },
  "redundancy_pruning": {
    "before": 10523,
    "after": 8941,
    "removed": 1582,
    "compression": 0.150
  },
  "zone_shifts": [...],
  "materialized_objects": [
    {"type": "chat_history_book", "zone": "Zone 3 (Library)", "count": 42},
    {"type": "fractal_tree", "zone": "Zone 5 (Knowledge Garden)", "count": 38},
    {"type": "diary_entry", "zone": "Zone 7 (Mirror Room)", "count": 15},
    {"type": "learning_insight", "zone": "Zone 8 (Learning Museum)", "count": 127}
  ],
  "semantic_clusters": 64,
  "elapsed_seconds": 23.4,
  "vocab_size": 8941
}
```

---

## Integration with GPU Training

### Updated Training Script Pattern

```python
#!/usr/bin/env python3
"""GPU-Accelerated Training with Automatic Sleep Consolidation"""

import time
import subprocess
from pathlib import Path

def train_specialist_gpu(specialist_name, epochs=100):
    """Train specialist with GPU and run sleep consolidation."""

    print(f"[{specialist_name}] Starting GPU training ({epochs} epochs)...")

    # Train using PTX kernels
    # ... (GPU training code here)

    print(f"[{specialist_name}] Training complete!")
    print(f"[{specialist_name}] Weights saved to checkpoint")
    print(f"[{specialist_name}] Galaxy knowledge in RAM (not yet permanent)")

    # CRITICAL: Wait 5 minutes for idle
    print(f"[{specialist_name}] Waiting 5 minutes for system stabilization...")
    time.sleep(300)

    # Run sleep consolidation
    print(f"[{specialist_name}] Running sleep-time consolidation...")
    subprocess.run([
        "/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python",
        "scripts/run_sleep_consolidation.py",
        "--embeddings", "/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl",
        "--output", "/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl",
        "--metrics", f"/K3D/Knowledge3D.local/logs/sleep_metrics_{specialist_name}.jsonl",
    ])

    print(f"[{specialist_name}] Sleep consolidation complete!")
    print(f"[{specialist_name}] Knowledge saved to House (permanent)")
    print(f"[{specialist_name}] Safe to unload model")

if __name__ == "__main__":
    train_specialist_gpu("multimodal", epochs=100)
```

---

## Common Mistakes

### Mistake 1: Exiting Too Early ❌

```bash
python train.py --epochs 100
# Training done!
# Exit immediately
```

**Problem**: Galaxy RAM lost before consolidation
**Solution**: Wait 5 min, run consolidation, then exit

### Mistake 2: Separate Sessions ❌

```bash
# Session 1: Train
python train.py --epochs 100

# Session 2 (later): Consolidate
python run_sleep_consolidation.py
```

**Problem**: Galaxy was in Session 1 RAM (now gone)
**Solution**: Keep same session, or use tmux

### Mistake 3: Skipping Consolidation ❌

```bash
python train.py --epochs 100
# Weights saved, that's enough right?
```

**Problem**: Weights have logic, but knowledge is lost
**Solution**: ALWAYS run consolidation after training

### Mistake 4: Using CPU for Consolidation ❌

```python
# Consolidation with scikit-learn (CPU)
kmeans = MiniBatchKMeans(n_clusters=256)
```

**Problem**: Violates K3D sovereign GPU principle
**Solution**: Use RPN-powered clustering (PTX kernels)

**Current status**: Consolidation uses sklearn (CPU) - needs GPU port!
**Future**: Port to RPN kernels for complete sovereign computing

---

## Summary

### The Golden Rule

**NEVER unload model before sleep consolidation runs!**

### The Workflow

1. **Train** (GPU, PTX kernels)
2. **Wait 5 min** (system stabilizes)
3. **Consolidate** (Galaxy → House)
4. **Verify** (check materialized objects)
5. **Exit** (safe to unload)

### The Architecture

- **Weights**: Logic (saved automatically)
- **Galaxy**: Active knowledge in RAM (lost if model unloads)
- **House**: Permanent knowledge on disk (survives forever)
- **Sleep**: Transfer from Galaxy to House (CRITICAL!)

### The Zones

- **Library**: Chat history books
- **Garden**: Fractal trees (high-honesty knowledge)
- **Mirror Room**: Diary entries (self-reflections)
- **Museum**: Learning insights

### The Principle

> "Sleep-time consolidation is NOT sleeping - it's the model's most important work: Making knowledge permanent."

---

**Remember**: Model weights are worthless without knowledge. Knowledge lives in Galaxy. Galaxy dies without consolidation. Consolidation makes knowledge eternal.

⚠️ **ALWAYS RUN SLEEP CONSOLIDATION** ⚠️

---

*This is not optional. This is not a nice-to-have. This is CRITICAL for K3D to function as designed.*

♾️⚛️🌙
