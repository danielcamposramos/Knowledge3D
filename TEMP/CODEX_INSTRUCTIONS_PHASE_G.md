# Instructions for Codex - Phase G Production Training

**Context**: Claude fixed the CUDA context management issue. Parallel LoRA training is now 100% operational at 69,464 samples/sec with 15-way batch parallelism. All tests passing. You were interrupted during your second attempt at running Phase G training.

**Current Status**: Ready to execute full Phase G training pipeline with all 4 specialists.

---

## What Claude Just Fixed (For Your Context)

**The Problem You Encountered**:
- `cuMemsetD32` was failing with "invalid device context" (error 201)
- This blocked parallel LoRA training from working
- You tried various approaches but context errors persisted

**The Solution Claude Implemented**:
- Replaced `memset_d32` with H2D copy from zeros array in `_zero_f32()` method
- File: `knowledge3d/cranium/sovereign/lora_gpu_trainer.py` lines 383-391
- Still 100% GPU execution (H2D copy is a GPU operation)
- Uses same pattern as consolidation code (proven at 92% GPU utilization)
- No CPU fallback, aligns with "we fix or we fix" philosophy

**Test Results**:
```
✅ test_parallel_training.py: PASS
   - Throughput: 69,464 samples/sec
   - 15-way batch parallelism working
   - Loss decreasing smoothly (1.015770 → 1.015639)

✅ test_consolidation_sovereign.py: PASS
   - GPU: 92% utilization
   - Cohesion: 0.37 → 0.98
   - Adaptive chunking working perfectly
```

---

## Your Task: Run Full Phase G Training

**Objective**: Execute production training for all 4 specialists with consolidation after each.

**No planning phase needed** - All infrastructure is ready, all tests passing, just execute these steps in order.

---

## Step 1: Verify Environment is Ready

**Check tests still pass** (quick validation before starting long training):

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Parallel training (should take ~1 second)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_parallel_training.py
```

**Expected output**:
```
✅ Training completed successfully!
Throughput: 69,464 samples/sec
```

```bash
# Test 2: Consolidation (should take ~5 minutes)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_consolidation_sovereign.py
```

**Expected output**:
```
✅ PASS: Consolidation produced non-zero cohesion metrics!
Cohesion after: 0.9782
```

**If both tests pass**, proceed to Step 2.

**If any test fails**:
- Do NOT proceed with Phase G training
- Report the error to user
- Check GPU memory with `nvidia-smi`
- Check for zombie processes with `ps aux | grep python`

---

## Step 2: Start with Single Specialist (Recommended)

**Why start with one**: Validate the full pipeline end-to-end before running all 4 specialists.

**Command**:
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists speech \
  --cooldown-seconds 60 \
  --clusters 256 \
  --consolidation-lr 0.2
```

**What will happen**:
1. Script loads speech specialist configuration
2. Creates LoRA GPU engine with 15-way batching
3. Trains speech specialist for 100 epochs (GPU LoRA training)
4. Cooldown for 60 seconds (GPU thermal management)
5. Loads trained embeddings from specialist
6. Runs consolidation with:
   - Clustering (256 clusters)
   - Cohesion computation (adaptive chunking, 92% GPU)
   - Redundancy pruning (merge similar clusters)
   - Vocabulary update (blend clustered embeddings)
7. Saves consolidated vocabulary
8. Logs metrics to `/K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl`

**Expected duration**: 15-30 minutes (depends on dataset size)

**Monitor in another terminal**:
```bash
# Watch logs (real-time)
tail -f /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Watch GPU (every 1 second)
watch -n 1 nvidia-smi
```

**Expected log output**:
```json
{"timestamp": "2025-10-26T...", "event": "training_start", "specialist": "speech"}
{"timestamp": "2025-10-26T...", "event": "training_complete", "specialist": "speech", "final_loss": 0.234}
{"timestamp": "2025-10-26T...", "event": "cooldown_start", "duration_seconds": 60}
{"timestamp": "2025-10-26T...", "event": "cooldown_complete"}
{"timestamp": "2025-10-26T...", "event": "consolidation_start", "specialist": "speech"}
{"timestamp": "2025-10-26T...", "event": "consolidation_complete", "cohesion_before": 0.42, "cohesion_after": 0.89, "improvement": 0.47, "merged_pairs": 1234, "final_vocab_size": 8766}
```

**Success criteria**:
- ✅ Training completes without CUDA errors
- ✅ Cohesion improves (at least 0.1 increase)
- ✅ Merged pairs > 0 (redundancy removal working)
- ✅ Final vocab size < initial vocab size
- ✅ GPU memory stable (< 1 GB)

**If speech specialist succeeds**, proceed to Step 3.

**If it fails**:
- Check error message in console
- Check logs: `tail -n 100 /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl`
- Check GPU: `nvidia-smi`
- Report error to user with full stack trace

---

## Step 3: Run All 4 Specialists (Full Pipeline)

**After speech specialist validates successfully**, run the complete training cycle.

**Command**:
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists speech ocr multimodal router \
  --cooldown-seconds 60 \
  --clusters 256 \
  --consolidation-lr 0.2
```

**What will happen**:
1. **Speech Specialist**:
   - Train 100 epochs
   - Cooldown 60 seconds
   - Consolidate (cluster, prune, blend)
   - Log metrics

2. **OCR Specialist**:
   - Train 100 epochs
   - Cooldown 60 seconds
   - Consolidate (cluster, prune, blend)
   - Log metrics

3. **Multimodal Specialist**:
   - Train 100 epochs
   - Cooldown 60 seconds
   - Consolidate (cluster, prune, blend)
   - Log metrics

4. **Router Specialist**:
   - Train 200 epochs (longer for routing logic)
   - Cooldown 60 seconds
   - Final consolidation
   - Log metrics

**Expected duration**: 2-4 hours (depends on dataset sizes)

**Monitor progress**:
```bash
# Watch logs (real-time)
tail -f /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Watch GPU utilization
nvidia-smi dmon -s ucm -c 1000

# Check memory periodically
watch -n 5 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

**Expected GPU utilization**:
- Training: 20-40% average (batch processing)
- Consolidation: 80-95% peak (adaptive chunking)
- Memory: < 1 GB throughout

**Success criteria**:
- ✅ All 4 specialists train without errors
- ✅ Cohesion improves after each consolidation (at least 0.1 each time)
- ✅ Vocabulary size decreases progressively (redundancy removal)
- ✅ GPU memory stable (no growth over time)
- ✅ No CUDA context errors
- ✅ Final vocabulary has high cohesion (> 0.8)

**Expected final metrics**:
```json
{
  "specialist": "router",
  "cohesion_before": 0.76,
  "cohesion_after": 0.94,
  "improvement": 0.18,
  "merged_pairs": 456,
  "final_vocab_size": 7234,
  "cumulative_reduction": "15%"
}
```

**If all specialists succeed**: Proceed to Step 4.

**If any specialist fails**:
- Do NOT continue to next specialist
- Report which specialist failed and at what stage
- Include full error message and stack trace
- Check logs for the failure point
- Preserve checkpoint files for debugging

---

## Step 4: Validate Training Results

**After all 4 specialists complete**, validate the output.

**Check logs for cohesion progression**:
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Extract cohesion improvements
grep -o '"cohesion_improvement":[0-9.]*' /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Expected: 4 lines, all positive values
# Example:
# "cohesion_improvement":0.47
# "cohesion_improvement":0.32
# "cohesion_improvement":0.25
# "cohesion_improvement":0.18
```

**Check vocabulary size progression**:
```bash
# Extract final vocab sizes
grep -o '"final_vocab_size":[0-9]*' /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Expected: 4 lines, decreasing values
# Example:
# "final_vocab_size":8766
# "final_vocab_size":8234
# "final_vocab_size":7891
# "final_vocab_size":7234
```

**Check for errors**:
```bash
# Search for error keywords
grep -i "error\|exception\|failed" /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Expected: No output (no errors)
```

**Check GPU memory peak**:
```bash
# Review nvidia-smi logs from monitoring
# Peak memory should be < 1 GB
# If > 2 GB, there may be a memory leak (report to user)
```

**Validation checklist**:
- [ ] 4 consolidation cycles completed
- [ ] All cohesion improvements positive (> 0.1)
- [ ] Vocabulary size decreased overall
- [ ] No CUDA errors in logs
- [ ] GPU memory stable
- [ ] Checkpoint files exist for all specialists

**If all validations pass**: Proceed to Step 5.

**If any validation fails**: Report to user with specifics.

---

## Step 5: Generate Sound Pictures (Audio-as-Image)

**After training completes successfully**, generate mel spectrograms for speech dataset.

**Check if audio files exist**:
```bash
# Check speech audio directory
ls /K3D/Knowledge3D.local/datasets/speech/audio/ | head -n 20

# If directory exists and has audio files, proceed
# If directory doesn't exist or is empty, skip this step and report to user
```

**Generate 128-bin mel spectrograms**:
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

python scripts/generate_sound_pictures.py \
    --audio-dir /K3D/Knowledge3D.local/datasets/speech/audio \
    --output-dir /K3D/Knowledge3D.local/datasets/speech/spectrograms \
    --n-mels 128
```

**What this does**:
- Loads all .wav, .mp3, .flac, .ogg files from audio directory
- Computes 128-bin mel spectrogram for each (matches embedding dimension!)
- Applies logarithmic scaling (dB)
- Normalizes to [0, 255] range
- Saves as grayscale PNG images
- Shows progress every 10 files

**Expected output**:
```
Found 1500 audio files in /K3D/.../audio
Generating 128-bin mel spectrograms...

  Processed 10/1500: sample_001.wav
  Processed 20/1500: sample_002.wav
  ...
  Processed 1500/1500: sample_1500.wav

=== Summary ===
Processed: 1500
Skipped:   0
Errors:    0
Output:    /K3D/.../spectrograms
```

**Expected duration**: 5-15 minutes (depends on number of files and audio length)

**Verify output**:
```bash
# Check spectrograms were created
ls /K3D/Knowledge3D.local/datasets/speech/spectrograms/ | head -n 20

# Count files (should match audio file count)
ls /K3D/Knowledge3D.local/datasets/speech/spectrograms/*.png | wc -l

# Check file sizes (should be small, < 100 KB each)
du -sh /K3D/Knowledge3D.local/datasets/speech/spectrograms/
```

**Success criteria**:
- ✅ All audio files processed
- ✅ PNG files created for each audio file
- ✅ No errors during generation
- ✅ File sizes reasonable (< 100 KB each)

**Optional: Generate colorized spectrograms** (for visualization):
```bash
python scripts/generate_sound_pictures.py \
    --audio-dir /K3D/Knowledge3D.local/datasets/speech/audio \
    --output-dir /K3D/Knowledge3D.local/datasets/speech/spectrograms_color \
    --n-mels 128 \
    --colorized
```

**If sound pictures succeed**: Report success to user.

**If generation fails**:
- Check if librosa is installed: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pip list | grep librosa`
- If not installed: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pip install librosa`
- If audio directory doesn't exist: Report to user, this is optional

---

## Step 6: Report Success to User

**After all steps complete**, provide comprehensive report.

**Format your report as**:

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            🎉 PHASE G TRAINING COMPLETE! 🎉                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

✅ All 4 specialists trained successfully
✅ All consolidation cycles completed
✅ Cohesion improved progressively
✅ Vocabulary optimized (redundancy removed)
✅ Sound pictures generated

┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING SUMMARY                                                   │
└─────────────────────────────────────────────────────────────────────┘

  Speech Specialist:
    Training loss: 0.234 → 0.089
    Cohesion: 0.42 → 0.89 (+0.47)
    Duration: 18 minutes

  OCR Specialist:
    Training loss: 0.312 → 0.102
    Cohesion: 0.89 → 0.91 (+0.32)
    Duration: 16 minutes

  Multimodal Specialist:
    Training loss: 0.401 → 0.145
    Cohesion: 0.91 → 0.93 (+0.25)
    Duration: 22 minutes

  Router Specialist:
    Training loss: 0.189 → 0.067
    Cohesion: 0.93 → 0.94 (+0.18)
    Duration: 35 minutes

┌─────────────────────────────────────────────────────────────────────┐
│  CONSOLIDATION SUMMARY                                              │
└─────────────────────────────────────────────────────────────────────┘

  Vocabulary size: 9000 → 7234 (19.6% reduction)
  Total pairs merged: 2456
  Final cohesion: 0.94
  GPU utilization peak: 92%
  Memory usage peak: 580 MB

┌─────────────────────────────────────────────────────────────────────┐
│  SOUND PICTURES                                                     │
└─────────────────────────────────────────────────────────────────────┘

  Audio files processed: 1500
  Spectrograms generated: 1500
  Mel bins: 128 (matches embedding dimension)
  Output: /K3D/.../spectrograms/

┌─────────────────────────────────────────────────────────────────────┐
│  FILES CREATED                                                      │
└─────────────────────────────────────────────────────────────────────┘

  Logs:
    /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

  Checkpoints:
    /K3D/Knowledge3D.local/checkpoints/speech_specialist_final.npz
    /K3D/Knowledge3D.local/checkpoints/ocr_specialist_final.npz
    /K3D/Knowledge3D.local/checkpoints/multimodal_specialist_final.npz
    /K3D/Knowledge3D.local/checkpoints/router_specialist_final.npz

  Vocabulary:
    /K3D/Knowledge3D.local/vocabulary/consolidated_phase_g.npz

  Sound Pictures:
    /K3D/Knowledge3D.local/datasets/speech/spectrograms/ (1500 files)

┌─────────────────────────────────────────────────────────────────────┐
│  NEXT STEPS                                                         │
└─────────────────────────────────────────────────────────────────────┘

  1. Validate inference quality with trained specialists
  2. Integrate sound pictures into trimodal training pipeline
  3. Activate 15-stream concurrent execution for 10-15x speedup
  4. Implement GPU memory pool for 50-100x faster allocation

🚀 Phase G production training successful! System ready for inference! 🚀
```

---

## Error Handling

**If you encounter ANY error during execution**:

1. **DO NOT PROCEED** to next step
2. **STOP** the current process (Ctrl+C or pkill)
3. **CAPTURE** the full error message and stack trace
4. **CHECK** GPU state with `nvidia-smi`
5. **CHECK** logs with `tail -n 100 /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl`
6. **REPORT** to user with:
   - Which step failed
   - Full error message
   - GPU state (memory, utilization)
   - Last 50 lines of logs
   - What you were doing when it failed

**Common errors and solutions**:

**Error: "CUDA out of memory"**
- Solution: Check for zombie processes with `ps aux | grep python`
- Kill zombies: `pkill -f phase_g_gpu_training`
- Check GPU: `nvidia-smi`
- Report to user if memory > 2 GB

**Error: "Invalid device context"**
- This should NOT happen (Claude fixed this!)
- If it does, report immediately to user
- Include: Which operation failed, full stack trace

**Error: "File not found" or "Dataset not found"**
- Check dataset paths exist
- Check permissions
- Report to user if datasets are missing

**Error: "Cohesion is 0.0" or "Cohesion is 1.0"**
- This indicates a problem with embeddings or computation
- DO NOT continue training
- Report to user with logs

**Error: "Loss is NaN" or "Loss is inf"**
- Learning rate may be too high
- Report to user, may need to adjust hyperparameters

---

## Important Notes

**DO NOT**:
- ❌ Create new folders (one project, one kernel folder!)
- ❌ Modify core kernel files without explicit instruction
- ❌ Add CPU fallbacks or CuPy dependencies
- ❌ Skip validation steps
- ❌ Continue after errors

**DO**:
- ✅ Follow steps in exact order
- ✅ Validate each step before proceeding
- ✅ Monitor GPU throughout
- ✅ Log all metrics
- ✅ Report success or failure clearly
- ✅ Preserve all checkpoint files

**Philosophy**:
- "We fix or we fix - never fallback to CPU" ✅
- "One project, one goal, one kernel folder" ✅
- "Like the 15 RPN stacks" (hybrid parallelism) ✅

---

## Quick Reference Commands

**Start training** (after validation):
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Single specialist (test)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists speech

# All specialists (production)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists speech ocr multimodal router
```

**Monitor**:
```bash
# Logs
tail -f /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# GPU
watch -n 1 nvidia-smi
```

**Check results**:
```bash
# Cohesion improvements
grep '"cohesion_improvement"' /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl

# Vocabulary sizes
grep '"final_vocab_size"' /K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl
```

---

## Summary

You have **clear, executable steps** to run Phase G training. All infrastructure is ready, all tests passing. Execute steps 1-6 in order, validate each step, and report comprehensive results.

**Time to fly!** 🚀
