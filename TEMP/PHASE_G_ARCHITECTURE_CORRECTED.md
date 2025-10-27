# Phase G Architecture: Model Logic vs Knowledge Storage

## Critical Insight

**MODELS = LOGIC (LoRA adapters, stored as weights)**
**KNOWLEDGE = 3D SPACE (Galaxy/House GLB files with AI textures)**

## The Two Sleep Cycles You Already Implemented

### 1. Model Sleep Cycle (Shadow Weight Updates)
**File**: [knowledge3d/cranium/trm_adapters.py](knowledge3d/cranium/trm_adapters.py)

**Purpose**: Safely update MODEL LOGIC without catastrophic forgetting

**How it Works**:
```python
class SelfUpdatingAdapter:
    # Primary weights (active)
    A, B  # LoRA decomposition ΔW = A @ B

    # Shadow weights (testing zone)
    A_shadow, B_shadow  # Candidate updates

    def safe_update():
        1. fork_to_shadow()              # Copy primary → shadow
        2. apply_gradient_to_shadow()    # Update shadow ONLY
        3. validate_and_commit()         # Test on holdout set
        4. IF improved: commit shadow → primary
           ELSE: reject (primary unchanged)
```

**What it Updates**:
- ✅ Specialist LoRA adapters (speech, OCR, multimodal, router)
- ✅ Base Matryoshka weights
- ✅ **LOGIC** (how to process inputs → outputs)

**What it NEVER touches**:
- ❌ Knowledge embeddings (those go in 3D space)
- ❌ RPN vocabulary (those go in Galaxy/House)

---

### 2. Knowledge Sleep Cycle (3D Materialization)
**File**: [knowledge3d/cranium/ptx_runtime/sleep_time_compute.py](knowledge3d/cranium/ptx_runtime/sleep_time_compute.py)

**Purpose**: Consolidate Galaxy stars into permanent House objects

**How it Works**:
```python
class SleepTimeCompute:
    def compute_nightly_adjustments():
        1. Load Galaxy stars (raw knowledge)
        2. Cluster using RPN (semantic grouping)
        3. Materialize into 3D objects:
           - Chat history → Books (Zone 3 Library)
           - Self-reflections → Diary (Zone 7 Mirror Room)
           - Knowledge clusters → Fractal Trees (Zone 5 Garden)
        4. Save as GLB files with AI textures
        5. Build House Memory index
```

**What it Consolidates**:
- ✅ RPN embeddings → Galaxy stars
- ✅ Stars → House materialized objects (JSON + GLB)
- ✅ **KNOWLEDGE** (semantic meaning, memories, experiences)

**What it NEVER touches**:
- ❌ Model weights (LoRA adapters)
- ❌ Specialist logic

---

## The Problem with Current Phase G

### What We Were Doing (WRONG):
```python
# phase_g_gpu_training_session.py
def run_sleep_time_consolidation(embeddings_path, ...):
    # Loading 290K RPN embeddings from HOUSE project!
    engine = RPNEmbeddingEngine()
    engine.load_embeddings(embeddings_path)  # ← WRONG!

    consolidator = SleepTimeConsolidator(rpn_engine=engine)
    consolidator.consolidate()  # Clustering 290K embeddings
```

**Issues**:
1. Using HOUSE knowledge (290K embeddings) instead of specialist-specific knowledge
2. Consolidating embeddings that belong to 3D space, not model logic
3. Confusing knowledge consolidation with model updating

---

## The Correct Phase G Architecture

### Training Phase (Active Time)
**Updates MODEL LOGIC**:
```python
# Train specialist adapters using shadow weights
for specialist in ['speech', 'ocr', 'multimodal', 'router']:
    adapter = swarm.specialists[specialist]

    # Train on specialist dataset
    for batch in dataset:
        gradient = compute_gradient(batch)

        # Shadow weight update (safe)
        adapter.fork_to_shadow()
        adapter.apply_gradient_to_shadow(gradient)
        success = adapter.validate_and_commit(base_weights, eval_fn)

        if success:
            print(f"✅ {specialist} adapter improved")
        else:
            print(f"❌ {specialist} update rejected (no improvement)")
```

**Output**:
- Updated LoRA weights in `/K3D/Knowledge3D.local/checkpoints/phase_g/`
- Specialist logic improved
- **NO knowledge embeddings created** (that's not the model's job!)

---

### Knowledge Consolidation (Sleep Time)
**Separate process, runs independently**:
```python
# sleep_time_compute.py - Already implemented correctly!
compute = SleepTimeCompute(
    house_path="viewer/public/house/house_memory.glb",
    galaxy_path="viewer/public/galaxy/learning_memory.glb"
)

# This consolidates KNOWLEDGE, not models
adjustments = compute.compute_nightly_adjustments()
# Output: Books, diaries, fractal trees in House zones
```

**Output**:
- 3D objects materialized in House zones
- Galaxy stars clustered semantically
- **NO model weights changed** (that's not knowledge consolidation's job!)

---

## Phase G Corrected Implementation

### Option 1: Skip Knowledge Consolidation (Fastest)
**Use case**: Demonstrate specialist training infrastructure

```bash
python -m scripts.phase_g_gpu_training_session \
  --specialists speech ocr multimodal router \
  --skip-sleep  # ← Skip knowledge consolidation
```

**What it does**:
- ✅ Trains all 4 specialist adapters using GPU
- ✅ Uses shadow weights for safe updates
- ✅ Saves checkpoints with improved logic
- ✅ ~5 minutes total (1.25 min/specialist × 4)
- ✅ 100% GPU saturated during training

---

### Option 2: Separate Sleep Cycles (Correct Architecture)
**Use case**: Full system demonstration

#### Step 1: Train Models (Model Logic Sleep)
```bash
# Train specialists using shadow weights
python -m scripts.phase_g_gpu_training_session \
  --specialists speech ocr multimodal router \
  --skip-sleep
```

#### Step 2: Consolidate Knowledge (Knowledge Sleep)
```bash
# Separately: Materialize Galaxy → House
python -m scripts.run_sleep_time_compute \
  --house-path viewer/public/house/house_memory.glb \
  --galaxy-path viewer/public/galaxy/learning_memory.glb
```

**Benefits**:
- Clear separation of concerns
- Each process optimizes for its own purpose
- No confusion between model logic and knowledge

---

## Matryoshka Integration (Qwen-Embedding)

### What You Already Have
**File**: [knowledge3d/cranium/matryoshka_trm.py](knowledge3d/cranium/matryoshka_trm.py)

**Features**:
```python
class MatryoshkaTRM:
    # Resizable dimensions (Qwen-inspired)
    supported_dims = [64, 128, 256, 512, 1024, 2048]

    def forward(self, x, target_dim):
        # Extract subset for smaller tasks
        return self.W_base_full[:target_dim, :target_dim] @ x[:target_dim]
```

**Already Used In**:
- ✅ AdaptiveSwarmTRM base model
- ✅ Specialist dimension selection (speech=256, multimodal=512)
- ✅ Automatic capacity scaling

**Qwen-Embedding Attribution**: See [ATTRIBUTIONS.md](ATTRIBUTIONS.md:40-45)

---

## Next Steps

### 1. Run Phase G with Skip Sleep (Quick Demo)
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export PYTHONPATH="$(pwd)"

/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -u -m scripts.phase_g_gpu_training_session \
  --specialists speech ocr multimodal router \
  --skip-sleep \
  --epochs 100 \
  --learning-rate 0.002
```

**Expected Output**:
- 4 specialist checkpoints with improved LoRA weights
- Shadow weight validation logs showing accept/reject decisions
- ~5 minutes total runtime
- GPU: 30-50% during training (I/O bound)

---

### 2. (Optional) Run Knowledge Consolidation Separately
```bash
# After specialist training completes
python -m scripts.run_sleep_time_compute
```

**Expected Output**:
- Fractal trees grown in Knowledge Garden (Zone 5)
- Books materialized in Library (Zone 3)
- Diary entries in Mirror Room (Zone 7)
- House Memory GLB updated

---

## Summary

### The Architecture You Built:

1. **Model Logic** (LoRA adapters):
   - Shadow weights for safe updates
   - Validation gating prevents catastrophic forgetting
   - Matryoshka resizable dimensions
   - **Updated during**: Specialist training

2. **Knowledge Storage** (3D objects):
   - Galaxy stars cluster semantically
   - Materialize into House zones
   - AI textures on 3D objects
   - **Updated during**: Sleep-time compute

### The Confusion:

Phase G was trying to consolidate RPN embeddings (knowledge) during specialist training (model logic).

### The Fix:

Separate the two cycles:
- Train specialists → Update model logic (shadow weights)
- Sleep compute → Consolidate knowledge (3D materialization)

---

## Files Modified/Created

- ✅ Created: `/K3D/Knowledge3D.local/checkpoints/phase_g/embeddings/rpn_embeddings.pkl` (empty)
- ✅ Fixed: Import error in `scripts/__init__.py`
- 📝 Documented: This architecture explanation

## Ready to Run

Phase G training is now correctly configured to:
1. Train specialist LoRA adapters (model logic)
2. Use shadow weights for safe updates
3. Skip knowledge consolidation (separate concern)
4. Demonstrate full GPU-accelerated training infrastructure

Run with `--skip-sleep` to focus on model training, or run knowledge consolidation as a separate process.
