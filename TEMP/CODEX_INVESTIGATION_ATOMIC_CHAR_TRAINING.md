# Codex Investigation Task: Atomic Character Training Failures

## Mission
Investigate the root cause of character training failures in Phase G (atomic character recognition) by analyzing all K3D GPU kernels and training pipeline components. 63/100 characters trained successfully; 37 are failing with consistent patterns.

## Success Baseline (What Works)
**Successfully trained characters (63 total)**:
- Digits: 0-9 (ASCII 48-57)
- Uppercase: A-Z (ASCII 65-90)
- Lowercase: a-z (ASCII 97-122)
- Special: ∑ (summation symbol, ASCII 8721)

**Checkpoint location**: `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/*_weights.npz`

**Evidence of success**:
- Training completed to full epochs
- Accuracy metrics stored in checkpoints
- Stable convergence patterns

## Failure Pattern (What's Broken)
**Failed characters (37 remaining)**:
- 33 punctuation marks: `' !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'`
- 4 letters (unspecified which ones failed)

**Failure symptoms**:
1. **Process termination**: Exit codes -10 (SIGTERM), -15 (SIGTERM)
2. **OOM killer suspected**: Processes killed before completion despite low GPU usage
3. **Gradient explosion**: Many gradients capped at 100.0
4. **Accuracy plateau**: Stuck at ~50% (random guessing)
5. **Low GPU utilization**: Only 1536 MB (14% of 11GB) before failures
6. **Training logs missing**: No successful completion logs for failed characters

**Last attempt details**:
- Command: Background parallel training processes
- Environment: CUDA 12.6, PyTorch, K3D GPU kernels
- System: 11GB GPU, sufficient RAM
- Pattern: All 37 failed characters exhibit identical failure mode

## Investigation Areas

### 1. GPU Kernel Analysis
**Location**: `knowledge3d/cranium/ptx_runtime/`

**42 hand-written PTX kernels to review**:
- Embedding extraction kernels
- CNN forward/backward pass kernels
- Gradient computation kernels
- Memory allocation patterns
- Synchronization primitives

**Questions to answer**:
- Do any kernels have different behavior for ASCII ranges?
- Are there memory access patterns that fail for punctuation characters?
- Is there a kernel that handles alphanumeric differently from special characters?
- Are there buffer size assumptions that break for certain character encodings?

### 2. Character Training Pipeline
**Location**: `knowledge3d/cranium/tests/test_atomic_chars_training.py`

**Components to investigate**:
- Character embedding generation (does it handle punctuation differently?)
- Dataset preparation (are punctuation samples properly formatted?)
- CNN architecture (any character-specific preprocessing?)
- Loss function (numerical stability for all character classes?)
- Gradient clipping (is 100.0 cap appropriate for all characters?)

### 3. Memory Management
**Questions to answer**:
- Why is GPU usage so low (1536 MB) when we have 11GB available?
- Is there a memory leak that accumulates during punctuation training?
- Are punctuation characters triggering larger embedding dimensions?
- Is the OOM killer triggered by CPU RAM exhaustion, not GPU?

### 4. Data Pipeline Differences
**Hypothesis**: Punctuation characters may have different:
- Embedding dimensionality
- Font rendering complexity
- OCR preprocessing requirements
- Training sample distribution

**Investigation needed**:
- Compare embedding sizes for successful vs failed characters
- Check if punctuation triggers different code paths in data loaders
- Verify training sample counts are balanced across character types

## Code Locations to Review

### Primary Investigation Targets
1. **PTX Kernels**: `knowledge3d/cranium/ptx_runtime/*.ptx` and `knowledge3d/cranium/ptx_runtime/*.py`
2. **Training Script**: `knowledge3d/cranium/tests/test_atomic_chars_training.py`
3. **CNN Architecture**: Search for atomic character CNN model definition
4. **Embedding Generation**: Search for character → embedding conversion
5. **Gradient Computation**: PTX kernels for backpropagation

### Checkpoints to Analyze
- **Successful checkpoints**: `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/char_65_A_weights.npz` (example)
- **Failed attempts**: Check for partial checkpoints or error logs for punctuation characters

## Investigation Methodology

### Step 1: Differential Analysis
Compare successful (alphanumeric) vs failed (punctuation) character code paths:
```bash
# Example: Check if character type triggers different kernel selection
grep -r "isalnum\|ispunct\|isalpha" knowledge3d/cranium/
```

### Step 2: Kernel Memory Profiling
Identify which kernel is running when failures occur:
```bash
# Find CUDA memory allocation patterns in PTX runtime
grep -r "cudaMalloc\|torch.cuda.memory" knowledge3d/cranium/ptx_runtime/
```

### Step 3: Gradient Flow Analysis
Check gradient computation for numerical stability:
```bash
# Find gradient clipping and normalization
grep -r "clip_grad\|gradient.*norm" knowledge3d/cranium/
```

### Step 4: Character Encoding Check
Verify ASCII encoding consistency:
```bash
# Check character encoding handling
grep -r "ord\|chr\|decode\|encode" knowledge3d/cranium/tests/test_atomic_chars_training.py
```

## Expected Outputs

### 1. Root Cause Identification
**Format**: Clear, technical explanation of why punctuation training fails

**Required evidence**:
- Specific kernel or code path that behaves differently
- Memory/gradient/numerical issue with technical details
- Reproducible test case demonstrating the failure

### 2. Differential Report
**Format**: Side-by-side comparison

| Aspect | Alphanumeric (Success) | Punctuation (Failure) |
|--------|------------------------|----------------------|
| Kernel path | ... | ... |
| Memory usage | ... | ... |
| Gradient flow | ... | ... |
| Embedding dimension | ... | ... |

### 3. Proposed Fix
**Format**: Actionable code changes or configuration adjustments

**Requirements**:
- Specific file and line numbers
- Before/after code snippets
- Rationale for why the fix addresses root cause

## Priority Questions
1. **Character-specific kernel routing**: Does the system route alphanumeric and punctuation through different GPU kernels?
2. **Memory allocation difference**: Why is GPU usage 10× lower than expected before failure?
3. **Gradient explosion source**: Which kernel/layer produces the capped 100.0 gradients?
4. **OOM trigger**: Is the killer responding to GPU memory, CPU memory, or process limits?
5. **Training data quality**: Are punctuation character samples fundamentally different from alphanumeric?

## Success Criteria
Your investigation is complete when you can:
1. **Reproduce the failure** with a minimal test case
2. **Explain the root cause** with technical evidence (kernel traces, memory profiles, gradient logs)
3. **Propose a fix** that addresses the underlying issue without breaking successful character training

## Context: Why This Matters
Atomic character recognition is the foundation for K3D's text understanding in the Cranium layer. The 63 successfully trained characters prove the architecture works; the 37 failures indicate a systematic issue (not random bugs). This is likely a kernel-level or memory-management problem, not a model architecture issue.

Your expertise in code analysis and kernel debugging makes you the ideal investigator. Focus on **what's different about punctuation** at the GPU/kernel level—that's where the answer lies.

---

## Findings (2025‑11‑14)

### What actually breaks
- **Punctuation/math glyphs are absent from the curated font DB** (`/K3D/Knowledge3D.local/font_db.pkl`), so `_has_glyph(...)` in `scripts/train_atomic_character.py:249-292` always returns `False` for them. Every such character therefore falls back to `discover_system_fonts()` (lines 196-214), which enumerates **2,807** system fonts on this workstation.
- When `--fonts 0` (the value used for the universal batch), the fallback path does **not** down-sample. `_build_dataset(...)` then produces **12,568 augmented samples per punctuation character**, which we measured at **~589 MB** of float32 tensors before shuffles or gradient buffers.
- `scripts/train_all_characters_batch.py:305-339` hard-codes **15 concurrent training jobs per batch**. Launching a batch of punctuation/math characters therefore allocates 15 × 0.6–0.8 GB = **9–12 GB of host RAM** just for image tensors (GPU usage stays around 1.5 GB). The box only has ~16 GB available, so the Linux OOM killer terminates the newest processes, yielding exit codes `-10`/`-15` and missing logs.
- Alphanumeric characters never hit the fallback path; their fonts come from the curated DB and are immediately capped to the requested `--fonts` limit (200 in our runs). Each process therefore holds only ~75 MB of image tensors, so the earlier batches complete without pressure.

### Differential snapshot

| Aspect | Alphanumeric (Success) | Punctuation / Math (Failure) |
| --- | --- | --- |
| Font source | `font_db.pkl` → filtered list (≃1,999 latin fonts) capped by `--fonts` | `discover_system_fonts()` (2,807 paths) because `_has_glyph` misses punctuation |
| Dataset size (`_build_dataset`) | 200 fonts → 1,600 samples → ~75 MB tensor | 2,807 fonts → 12,568 samples → ~589 MB tensor when `--fonts 0` |
| Batch orchestrator | Prior runs used sequential/dynamic script (`MAX_PARALLEL=8`) | `train_all_characters_batch.py` always spawns 15 workers |
| Observed behaviour | Full 1,500 epochs, checkpoints saved | Processes receive SIGTERM/SIGKILL after a few dozen epochs; logs truncated; gradients never flush to disk |

### Proposed repairs

1. **Throttle parallelism for mixed-script runs.** Added a `--parallel` CLI flag (default 6) in `scripts/train_all_characters_batch.py` so universal batches can be limited to the amount of host RAM actually available (use `--parallel 4` on the current 16 GB box). This change keeps existing behaviour for users who explicitly pass `--parallel 15`.
2. **Document the memory blow-up.** We should update the runbook to warn that `--fonts 0` + non-latin scripts bypass the DB cap and balloon to >500 MB per process. In practice, keep `--fonts` ≤ 200 until punctuation glyphs are indexed in `font_db.pkl`.
3. **Long-term fix** (not yet implemented): extend the font ingestion pipeline so punctuation/math glyph coverage lives in `font_db.pkl`. Once those glyphs are indexed we can cap `fonts` before allocating arrays, removing the fallback entirely.

### Validation / next steps

1. Re-run a small universal batch with the new `--parallel` limit plus `--fonts 200` to confirm RAM stays <8 GB and all characters finish.
2. When ready, regenerate `font_db.pkl` with punctuation glyph metadata so that `_has_glyph(...)` succeeds for ASCII 32‑47 etc., then relax the fallback reliance.
3. Keep an eye on `nvidia-smi` and `/var/log/kern.log` during the next universal run to ensure SIGTERM events disappear.

---

### Nov‑14 Updates
- A full font harvest (`scripts/harvest_fonts_for_ocr.py --max-fonts 0`) completed successfully, producing `font_db.pkl` with 2,790 usable fonts and ~2.73 M glyph entries. The harvester now includes the entire math-symbol registry plus ASCII punctuation in its character inventory, so `_has_glyph(...)` no longer defaults punctuation runs to the slow system-font scan.
- The dynamic trainer (`scripts/train_atomic_characters_dynamic.py`) was historically run with **sequential 6‑way scheduling**, which finished early characters quickly because new jobs only launched after another finished. The recent regression came from relaunching eight simultaneous workers *without* clearing their corrupt checkpoints, so all of them resumed at ~50 % accuracy and promptly saturated the CUDA context.
- To prevent that failure mode we now ignore checkpoints whose recorded accuracy is below **60 %**. `scripts/train_atomic_character.py` simply refuses to load CNN/FC weights or resume metadata unless `accuracy ≥ 0.60`, forcing a clean initialization whenever a run stalls near random chance. This threshold keeps good runs (typically ≥85 %) intact while ensuring the “old paradigm” embeddings do not poison new trainings under the procedural memory engine.

---
**Repository**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D`

**Codex, begin your investigation. Report findings and proposed fixes when complete.**
