# Prompt for GPT/Codex-Max: ARC-AGI 2 Week 2 Execution

**Date**: November 24, 2025
**Your Mission**: Execute ARC-AGI 2 Week 2 benchmarking and dataset processing
**Context**: Daniel is a non-coding human who orchestrates AI partners — YOU will do the implementation work!

---

## ⚠️ CRITICAL: Read These Files FIRST (In Order!)

**BEFORE doing ANY work, read these files COMPLETELY**:

1. **First**: [`CODEX.md`](../CODEX.md) — Your role as implementer
2. **Second**: Latest briefing:
   ```bash
   ls -t docs/Briefings/SOVEREIGN_SWARM_BRIEFING_*.md | head -n1
   ```
   Read the ENTIRE briefing (v3) — NOT just snippets!

3. **Third**: These context documents:
   - [`BRIEFING.md`](../BRIEFING.md) — Current project status
   - [`docs/ROADMAP.md`](../docs/ROADMAP.md) — Current phase
   - [`CLAUDE.md`](../CLAUDE.md) — Claude's architecture work (for context)

**Why this order matters**:
- CODEX.md tells you your role and collaboration patterns
- Briefing v3 has sovereignty constraints (PTX+RPN hot path!)
- BRIEFING.md has current status (Phase 5 complete, 51/51 tests passing)
- ROADMAP.md shows where we are in the project timeline

---

## 🎯 What Has Been Built (Week 1)

**Architecture** (by Claude):
- ✅ Phase 1 RPN ops verified (rotate, translate, scale in PTX)
- ✅ Ternary codec integration designed (video + audio embedders)
- ✅ Multi-modal architecture specified (fusion with ternary routing)
- ✅ 5 specification documents in TEMP/

**Implementation** (by previous Codex):
- ✅ `VideoGridEmbedder` — DCT spatial features (TernaryVideoCodec)
- ✅ `AudioGridEmbedder` — MDCT temporal features (TernaryAudioCodec)
- ✅ `MultiModalGridEmbedder` — Fusion with ternary routing {-1, 0, +1}
- ✅ `ARCGridProcessor` — 4 embedder modes (procedural, video, audio, multimodal)
- ✅ Unit tests — GPU-free testing with fake codecs

**Files Created**:
```
knowledge3d/training/arc_agi/
├── embedders/
│   ├── video_grid_embedder.py      # ✅ Implemented
│   ├── audio_grid_embedder.py      # ✅ Implemented
│   └── multimodal_grid_embedder.py # ✅ Implemented
├── grid_processor.py                # ✅ Enhanced (4 modes)
└── __init__.py                      # ✅ Updated

tests/
└── test_arc_grid_embedders.py       # ✅ Tests passing

TEMP/
├── CODEX_ARC_AGI_2_PREPARATION_11.24.2025.md           # Original spec
├── CLAUDE_ARC_CODEC_INTEGRATION_11.24.2025.md          # Codec architecture
└── CODEX_ARC_EMBEDDER_USAGE_GUIDE_11.24.2025.md        # ⭐ YOUR GUIDE
```

---

## 📋 Your Mission (Week 2)

**You are implementing Week 2 of the ARC-AGI 2 competition preparation.**

**Financial Context** (WHY this matters):
> Daniel lives in a favela in Brazil. To buy 1 US dollar, he must spend 5 reais. ARC-AGI prize money ($10,000+ = R$50,000+) would be TRANSFORMATIVE for his life.

**This is not academic research. This is about survival and transformation.**

### Task 1: Download ARC-AGI 2 Dataset ⚠️ ACTION REQUIRED

**Goal**: Download and cache ARC-AGI dataset for training.

**Execute**:
```bash
# Activate environment
conda activate k3d-cranium
export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

# Create download script
cat > /tmp/download_arc_dataset.py << 'EOF'
"""Download ARC-AGI 2 dataset and create reasoning cache."""

from knowledge3d.training.reasoning.arc_dataset import (
    ensure_arc_dataset,
    prepare_arc_reasoning_cache,
)
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

# Step 1: Download dataset
print("📥 Downloading ARC-AGI dataset...")
dataset_path = ensure_arc_dataset(force_download=False)
print(f"✅ Dataset ready at: {dataset_path}")

# Step 2: Verify structure
import os
training_dir = dataset_path / "data" / "training"
evaluation_dir = dataset_path / "data" / "evaluation"
test_dir = dataset_path / "data" / "test"

n_training = len(list(training_dir.glob("*.json"))) if training_dir.exists() else 0
n_evaluation = len(list(evaluation_dir.glob("*.json"))) if evaluation_dir.exists() else 0
n_test = len(list(test_dir.glob("*.json"))) if test_dir.exists() else 0

print(f"\n📊 Dataset Statistics:")
print(f"  Training tasks:   {n_training}")
print(f"  Evaluation tasks: {n_evaluation}")
print(f"  Test tasks:       {n_test}")

# Step 3: Create reasoning cache (small sample first)
print(f"\n🔄 Creating reasoning cache (100 examples)...")
rpn_embedder = RPNEmbeddingEngine()
cache_path = prepare_arc_reasoning_cache(
    rpn_embed_sentence=rpn_embedder.embed_sentence,
    limit=100,  # Start small for testing
    rebuild=True,
    download=False,  # Already downloaded above
)
print(f"✅ Cache created at: {cache_path}")

# Step 4: Show sample task
print(f"\n📄 Sample Task Structure:")
import json
from pathlib import Path

task_files = list(training_dir.glob("*.json"))
if task_files:
    sample_task_path = task_files[0]
    with open(sample_task_path, 'r') as f:
        sample_task = json.load(f)

    print(f"  Task ID: {sample_task_path.stem}")
    print(f"  Training examples: {len(sample_task.get('train', []))}")
    print(f"  Test examples: {len(sample_task.get('test', []))}")

    if sample_task.get('train'):
        first_example = sample_task['train'][0]
        input_grid = first_example['input']
        output_grid = first_example['output']

        print(f"\n  Example 1:")
        print(f"    Input grid:  {len(input_grid)}×{len(input_grid[0])} (height×width)")
        print(f"    Output grid: {len(output_grid)}×{len(output_grid[0])}")
        print(f"\n    Input grid sample:")
        for row in input_grid[:5]:  # Show first 5 rows
            print(f"      {row[:10]}")  # Show first 10 columns

print("\n✅ Dataset download and verification complete!")
EOF

# Run download script
python /tmp/download_arc_dataset.py
```

**Expected Output**:
```
✅ Dataset ready at: /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master
📊 Dataset Statistics:
  Training tasks:   400
  Evaluation tasks: 400
  Test tasks:       [number]
✅ Cache created at: /K3D/Knowledge3D.local/datasets/arc_agi/arc_reasoning_pairs.npz
✅ Dataset download and verification complete!
```

**Report**:
- Number of tasks downloaded
- Cache size (MB)
- Sample task structure (as shown above)

---

### Task 2: Create Benchmarking Script ⚠️ ACTION REQUIRED

**Goal**: Benchmark all 4 embedder modes on real ARC grids.

**Execute**:
```bash
# Copy the benchmark script from the usage guide
cat > scripts/benchmark_arc_embedders.py << 'EOF'
[COPY THE FULL SCRIPT FROM TEMP/CODEX_ARC_EMBEDDER_USAGE_GUIDE_11.24.2025.md]
EOF

# Make executable
chmod +x scripts/benchmark_arc_embedders.py

# Run benchmark (k3d-cranium environment)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/benchmark_arc_embedders.py
```

**What to look for**:
1. **Latency**: Does each mode meet targets?
   - Procedural: 5-15ms (baseline)
   - Video: <5ms (target)
   - Audio: <3ms (target)
   - Multi-modal: <10ms (target)

2. **Embedding differences**: Do different modes produce different embeddings?
   - Compare cosine similarity across modes
   - Ternary routing should change embeddings (routing -1 vs +1)

3. **GPU errors**: Any PTX kernel failures?
   - If yes: Check CUDA environment
   - If no: ✅ Sovereignty maintained!

**Report** (in completion doc):
```markdown
### Benchmark Results

| Mode | Avg Latency | Std Dev | Meets Target? |
|------|-------------|---------|---------------|
| Procedural | X.XX ms | ±Y.YY ms | ✅/❌ |
| Video | X.XX ms | ±Y.YY ms | ✅/❌ |
| Audio | X.XX ms | ±Y.YY ms | ✅/❌ |
| Multi-modal | X.XX ms | ±Y.YY ms | ✅/❌ |

**Embedding Similarity** (Grid 0):
- Procedural <-> Video: 0.XXXX
- Procedural <-> Audio: 0.XXXX
- Video <-> Audio: 0.XXXX
- Multi-modal <-> Video: 0.XXXX

**Ternary Routing Impact**:
- Video-heavy (-1) <-> Audio-heavy (+1): 0.XXXX (should be <0.9)
- Video-heavy (-1) <-> Balanced (0): 0.XXXX
- Audio-heavy (+1) <-> Balanced (0): 0.XXXX
```

---

### Task 3: Process Sample ARC Tasks ⚠️ ACTION REQUIRED

**Goal**: Process 50 ARC tasks with multi-modal embedder, analyze patterns.

**Execute**:
```bash
# Create processing script
cat > scripts/process_arc_sample.py << 'EOF'
"""Process sample ARC tasks and analyze patterns."""

import numpy as np
from pathlib import Path

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.reasoning.arc_dataset import (
    ensure_arc_dataset,
    _iter_task_files,
    _load_task,
)


def process_sample_tasks(n_tasks=50):
    """Process sample ARC tasks with multi-modal embedder."""

    # Initialize processor (multi-modal is BEST)
    processor = ARCGridProcessor(
        matryoshka_dim=512,
        embedder_type="multimodal"
    )

    # Ensure dataset
    dataset_path = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset_path, split="training"))[:n_tasks]

    print(f"Processing {len(task_files)} ARC tasks...")

    results = []

    for i, task_path in enumerate(task_files):
        task = _load_task(task_path)
        task_id = task_path.stem

        task_result = {
            "task_id": task_id,
            "n_train": len(task.get("train", [])),
            "n_test": len(task.get("test", [])),
            "grid_sizes": [],
            "embeddings": [],
        }

        # Process training examples
        for example in task.get("train", []):
            input_grid = example["input"]
            h, w = len(input_grid), len(input_grid[0]) if input_grid else 0

            # Embed with balanced routing
            embedding = processor.grid_to_spatial_embedding(input_grid, routing=0)

            task_result["grid_sizes"].append((h, w))
            task_result["embeddings"].append(embedding)

        results.append(task_result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(task_files)} tasks...")

    # Analyze
    print(f"\n📊 Analysis:")

    all_sizes = [size for task in results for size in task["grid_sizes"]]
    heights = [h for h, w in all_sizes]
    widths = [w for h, w in all_sizes]

    print(f"  Total grids processed: {len(all_sizes)}")
    print(f"  Grid size range:")
    print(f"    Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")
    print(f"    Width:  {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")

    # Embedding stats
    all_embeddings = np.array([emb for task in results for emb in task["embeddings"]])
    print(f"\n  Embedding statistics:")
    print(f"    Shape: {all_embeddings.shape}")
    print(f"    Mean: {np.mean(all_embeddings):.4f}")
    print(f"    Std:  {np.std(all_embeddings):.4f}")
    print(f"    Min:  {np.min(all_embeddings):.4f}")
    print(f"    Max:  {np.max(all_embeddings):.4f}")

    # Save embeddings
    output_dir = Path("/K3D/Knowledge3D.local/datasets/arc_agi_embeddings/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_result in results:
        task_id = task_result["task_id"]
        embeddings = np.array(task_result["embeddings"], dtype=np.float32)

        output_path = output_dir / f"{task_id}.npz"
        np.savez_compressed(
            output_path,
            task_id=task_id,
            embeddings=embeddings,
            grid_sizes=task_result["grid_sizes"],
        )

    print(f"\n✅ Saved {len(results)} task embeddings to {output_dir}")

    return results


if __name__ == "__main__":
    results = process_sample_tasks(n_tasks=50)
    print("\n✅ Sample processing complete!")
EOF

# Run processing
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/process_arc_sample.py
```

**Expected Output**:
```
Processing 50 ARC tasks...
  Processed 10/50 tasks...
  Processed 20/50 tasks...
  ...
  Processed 50/50 tasks...

📊 Analysis:
  Total grids processed: [number]
  Grid size range:
    Height: 3 - 30 (avg: 15.2)
    Width:  3 - 30 (avg: 14.8)

  Embedding statistics:
    Shape: ([number], 512)
    Mean: 0.XXXX
    Std:  0.XXXX
    Min:  -X.XXXX
    Max:  +X.XXXX

✅ Saved 50 task embeddings to /K3D/Knowledge3D.local/datasets/arc_agi_embeddings/
```

---

### Task 4: Spatial Primitive Detection Test ⚠️ ACTION REQUIRED

**Goal**: Test how well different embedders detect transformations (ROTATE, FLIP, etc.).

**Execute**:
```bash
# Create primitive detection test
cat > scripts/test_primitive_detection.py << 'EOF'
"""Test spatial primitive detection across embedder modes."""

import numpy as np
from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor


def test_primitive_detection():
    """Test primitive detection with different embedders."""

    # Test grids
    grid_original = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]

    # Create transformations
    grid_rotate_90 = np.rot90(np.array(grid_original), k=-1).tolist()
    grid_rotate_180 = np.rot90(np.array(grid_original), k=2).tolist()
    grid_flip_h = np.fliplr(np.array(grid_original)).tolist()
    grid_flip_v = np.flipud(np.array(grid_original)).tolist()

    transformations = {
        "ROTATE_90": grid_rotate_90,
        "ROTATE_180": grid_rotate_180,
        "FLIP_H": grid_flip_h,
        "FLIP_V": grid_flip_v,
    }

    # Test with all embedder modes
    modes = ["procedural", "video", "audio", "multimodal"]

    print("="*60)
    print("Spatial Primitive Detection Test")
    print("="*60)

    results = {}

    for mode in modes:
        print(f"\n{mode.upper()} mode:")
        processor = ARCGridProcessor(matryoshka_dim=512, embedder_type=mode)

        mode_results = {}

        for transform_name, grid_transformed in transformations.items():
            detected = processor.detect_spatial_primitive(grid_original, grid_transformed)

            correct = detected["primitive"] == transform_name
            mode_results[transform_name] = {
                "detected": detected["primitive"],
                "correct": correct,
                "confidence": detected["confidence"],
            }

            status = "✅" if correct else "❌"
            print(f"  {status} {transform_name:15s}: detected as {detected['primitive']:15s} (conf: {detected['confidence']:.2f})")

        results[mode] = mode_results

    # Summary
    print("\n" + "="*60)
    print("Detection Accuracy Summary")
    print("="*60 + "\n")

    for mode in modes:
        n_correct = sum(1 for r in results[mode].values() if r["correct"])
        n_total = len(results[mode])
        accuracy = n_correct / n_total * 100

        print(f"  {mode:15s}: {n_correct}/{n_total} correct ({accuracy:.1f}%)")

    print("\n✅ Primitive detection test complete!")

    return results


if __name__ == "__main__":
    results = test_primitive_detection()
EOF

# Run test
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/test_primitive_detection.py
```

**Expected Output**:
```
PROCEDURAL mode:
  ✅ ROTATE_90      : detected as ROTATE_90       (conf: 1.00)
  ✅ ROTATE_180     : detected as ROTATE_180      (conf: 1.00)
  ✅ FLIP_H         : detected as FLIP_H          (conf: 1.00)
  ✅ FLIP_V         : detected as FLIP_V          (conf: 1.00)

VIDEO mode:
  [results...]

Detection Accuracy Summary:
  procedural     : 4/4 correct (100.0%)
  video          : X/4 correct (XX.X%)
  audio          : X/4 correct (XX.X%)
  multimodal     : X/4 correct (XX.X%)
```

---

### Task 5: Write Completion Report ⚠️ ACTION REQUIRED

**Goal**: Document Week 2 achievements in TEMP/.

**Template**: `TEMP/CODEX_ARC_AGI_WEEK2_COMPLETE_11.24.2025.md`

```markdown
# ARC-AGI 2 Preparation — Week 2 Complete

**Date**: November 24, 2025
**Implementer**: Codex-Max / GPT
**Status**: ✅ COMPLETE

---

## Achievements

### Task 1: ARC-AGI 2 Dataset Download
- ✅ Dataset downloaded to: `/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master`
- ✅ Training tasks: [number]
- ✅ Evaluation tasks: [number]
- ✅ Test tasks: [number]
- ✅ Reasoning cache created: [size] MB
- ✅ Sample task structure verified

**Sample Task**:
```
Task ID: [task_id]
Training examples: [count]
Test examples: [count]
Grid sizes: [range]
```

### Task 2: Embedder Benchmarking
- ✅ Benchmark script created: `scripts/benchmark_arc_embedders.py`
- ✅ Tested on 10 sample grids
- ✅ All 4 modes functional

**Results**:
[INSERT BENCHMARK TABLE FROM ABOVE]

**Analysis**:
- Best latency: [mode] at [X.XX]ms
- Multi-modal fusion: [X.XX]ms (meets <10ms target: ✅/❌)
- Ternary routing working: ✅ (embeddings differ by routing value)

### Task 3: Sample Task Processing
- ✅ Processing script created: `scripts/process_arc_sample.py`
- ✅ Processed 50 ARC tasks
- ✅ Embeddings saved: `/K3D/Knowledge3D.local/datasets/arc_agi_embeddings/`

**Statistics**:
- Total grids: [count]
- Grid sizes: [min]×[min] to [max]×[max]
- Embedding mean: [value]
- Embedding std: [value]

### Task 4: Primitive Detection Test
- ✅ Detection script created: `scripts/test_primitive_detection.py`
- ✅ Tested ROTATE, FLIP transformations
- ✅ All modes tested

**Accuracy**:
[INSERT ACCURACY TABLE FROM ABOVE]

---

## Issues Encountered

[List any issues, errors, or unexpected results]

**Example**:
- ❌ GPU memory error when processing large grids (>30×30)
  - **Solution**: Added padding limit to 32×32 (8-aligned for DCT)
- ⚠️ Audio embedder slower than expected (5ms vs 3ms target)
  - **Cause**: MDCT frame size overhead
  - **Action**: Consider reducing n_harmonics from 20 to 10

---

## Next Steps (Week 3-4)

Based on Week 2 results:
1. **Rule Composition**: Combine primitives (ROTATE + FILL)
2. **TRM Shadow Copy Integration**: Few-shot learning on ARC examples
3. **Full Dataset Processing**: Process all 400 training tasks
4. **Embedder Optimization**: Improve latency if targets not met

---

## Files Created

**Scripts**:
- `scripts/benchmark_arc_embedders.py` — Embedder benchmarking
- `scripts/process_arc_sample.py` — Sample task processing
- `scripts/test_primitive_detection.py` — Primitive detection test

**Data**:
- `/K3D/Knowledge3D.local/datasets/arc_agi/` — Downloaded dataset
- `/K3D/Knowledge3D.local/datasets/arc_agi_embeddings/` — Processed embeddings (50 tasks)

**Reports**:
- `TEMP/CODEX_ARC_AGI_WEEK2_COMPLETE_11.24.2025.md` — This file

---

## Ready for Week 3-4! 🚀

**Status**: Infrastructure validated, ready for advanced training.
```

---

## 🎯 Success Criteria

**MUST ACHIEVE**:
- ✅ ARC-AGI dataset downloaded (400+ tasks)
- ✅ All 4 embedder modes benchmarked
- ✅ Latency targets met or documented why not
- ✅ Sample tasks processed (50+)
- ✅ Primitive detection tested
- ✅ Completion report written

**SHOULD ACHIEVE**:
- ✅ Multi-modal embedder <10ms (PRIORITY!)
- ✅ Ternary routing verified (different embeddings)
- ✅ Embeddings saved to disk for training
- ✅ GPU errors debugged (if any)

**NICE TO HAVE**:
- ⚠️ Full dataset processed (all 400 tasks)
- ⚠️ Compression ratio analysis
- ⚠️ Automatic routing selection

---

## 🔥 Key Reminders

1. **Daniel doesn't code** — You do ALL the execution!
2. **Sovereignty guardrail**: Hot path = PTX + RPN only (no numpy!)
3. **Financial stakes**: This is LIFE-CHANGING for Daniel (R$5 = $1 USD, favela)
4. **Test-first**: Run tests before claiming completion
5. **Document everything**: Write clear, detailed reports

---

## 💡 If You Encounter Issues

**GPU/PTX Errors**:
```bash
# Check CUDA environment
nvidia-smi
nvcc --version

# Check conda environment
conda env list
which python
```

**Import Errors**:
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Set correctly
export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
```

**Slow Performance**:
- Check GPU utilization: `nvidia-smi` while running
- If CPU-bound: Codecs may be falling back to CPU (check imports)
- If GPU OOM: Reduce matryoshka_dim or batch size

---

## 🚀 Ready to Execute?

**Checklist before starting**:
- [ ] Read CODEX.md (your role)
- [ ] Read Sovereign Swarm Briefing v3 (sovereignty constraints)
- [ ] Read BRIEFING.md (current status)
- [ ] Understand financial context (Daniel's life in favela)
- [ ] Environment ready (k3d-cranium conda env)

**Execute in order**:
1. Task 1: Download dataset
2. Task 2: Benchmark embedders
3. Task 3: Process sample tasks
4. Task 4: Test primitive detection
5. Task 5: Write completion report

**Timeline**: 4-6 hours total execution time

---

**This is going to break the bank!** 💰🏆

Let's win ARC-AGI 2 and transform Daniel's life! 🎯

---

**Handoff from**: Claude (architecture) + Codex (Week 1 implementation)
**Handoff to**: GPT/Codex-Max (Week 2 execution)
**Status**: Ready for execution ✅
