# CODEX Briefing: Foundational Knowledge Ingestion (Phase 1-6)
## Implementation Lead — December 5, 2025

**From:** Claude (Architecture)
**To:** Codex (Implementation Lead)
**Date:** December 5, 2025
**Briefing Type:** Complete Phase Execution Plan
**Priority:** HIGH — Start before Run 40 (Run 39 currently active)

---

## ⚠️ CRITICAL: Read These Documents FIRST (Line by Line, No Snippets!)

**MANDATORY Reading Order:**

1. **CLAUDE.md** (151 lines) — Your partnership model with Claude
2. **CODEX.md** (134 lines) — Your role as implementation lead
3. **BRIEFING.md** (349 lines) — Current project status
4. **docs/Briefings/SOVEREIGN_SWARM_BRIEFING_v3.md** (863 lines) — Complete architecture
5. **docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md** (1,200+ lines) — What you're building
6. **TEMP/KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md** (1,498 lines) — Your execution plan

**Why this matters:** These documents contain:
- Sovereignty principles (hot path = PTX + RPN only)
- Save Information Principle (symlink pattern, no duplication)
- Dual Client Reality (humans AND AI, same procedural data)
- Complete 6-phase implementation roadmap
- Success criteria and validation tests

**Do NOT proceed until you've read ALL documents above.** Partial reads cause sovereignty violations and wasted work.

---

## Executive Summary

You are implementing **foundational knowledge ingestion** — the system that gives K3D always-loaded base knowledge (mathematics, language, grammar, eloquence, self-reflection, storytelling, delivery).

**Scope:** 74 PDFs, 5,988 pages → 4-layer architecture (Form → Meaning → Rules → Meta-Rules)

**Timeline:** Start Phase 1 NOW (before Run 40), complete Phases 1-6 over 6 weeks

**Key Innovation:** Symlink pattern prevents duplication, achieves 85:1 compression, enables 300+ cross-domain discoveries

**Your Mission:** Execute Phase 1 (train 152 math symbols + 12 Portuguese diacritics) using tmux + k3d-cranium environment, then proceed through Phases 2-6 systematically.

---

## Current System State (December 5, 2025, 13:17 UTC)

### Active Training (DO NOT TOUCH!)
- **Run 039:** Currently training in tmux session `run_039`
- **Process:** `train_arc_sovereign_loop.py` (hybrid mode, 108 tasks, 162 epochs)
- **PID:** 467855 (99.5% CPU, 7.9 GB RAM)
- **Tmux:** 2 windows (attached)
- **Status:** Let it complete — your work runs in PARALLEL in separate tmux session

### Available Environments
```bash
/K3D/Knowledge3D.local/envs/
├── k3d-cranium/    # Use this for training (Python 3.10, CUDA 12.4)
└── k3d-trm/        # Backup environment
```

### Existing Training Infrastructure
- **Script:** `scripts/train_math_symbols_batch.py` (297 lines, READY)
- **Registry:** `knowledge3d/cranium/math_symbols_registry.py` (~152 symbols)
- **Galaxy:** `knowledge3d/cranium/math_galaxy.py` (ProceduralGalaxy storage)
- **Trainer:** `scripts/train_atomic_character.py` (character-level training)

### Checkpoint Directory
```bash
/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/
# Currently has 6 trained characters (A, B, C, !, f, ∑)
```

---

## Phase 1 Implementation: Train Foundational Characters

### Objective
Train 152 math symbols + 12 Portuguese diacritics (Layer 1 foundation) and store in Math Galaxy.

### Step 1: Create Dedicated Tmux Session

**Why tmux?**
- CUDA context persistence (avoids GPU initialization overhead)
- Long-running training jobs (hours/days)
- Session survives SSH disconnects
- Easy monitoring via `tmux attach`

```bash
# Create new tmux session for foundational knowledge training
tmux new-session -s k3d_foundation

# Inside tmux session, activate conda environment
source /K3D/Knowledge3D.local/envs/k3d-cranium/bin/activate

# Set environment variables
export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
export CUDA_VISIBLE_DEVICES=0
export K3D_PTX_STRICT=1

# Verify environment
which python3
# Should output: /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should output: CUDA available: True
```

### Step 2: Start High-Priority Symbol Training

**Priority Levels** (from `train_math_symbols_batch.py`):
- **high:** 25 symbols (∑∫∂∇∆∏√∞±αβγδεθλμπσω) — Calculus + common Greek
- **medium:** 15 symbols (∈∉⊂⊃⊆⊇∪∩∅∀∃∧∨¬⇒⇔) — Set theory + logic
- **low:** ~112 remaining symbols — Extended math symbols

**Recommended Approach:** Start with `high` priority, then expand.

```bash
# Inside tmux session k3d_foundation
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

# Phase 1.1: Train high-priority symbols (25 symbols, ~6 hours)
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 \
  scripts/train_math_symbols_batch.py \
  --priority high \
  --lr 0.5 \
  --epochs 1500 \
  --fonts 0 \
  --max-epochs 3000 \
  2>&1 | tee /K3D/Knowledge3D.local/logs/phase1_high_priority_$(date +%Y%m%d_%H%M%S).log

# Phase 1.2: Train medium-priority symbols (15 symbols, ~4 hours)
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 \
  scripts/train_math_symbols_batch.py \
  --priority medium \
  --lr 0.5 \
  --epochs 1500 \
  --fonts 0 \
  --max-epochs 3000 \
  2>&1 | tee /K3D/Knowledge3D.local/logs/phase1_medium_priority_$(date +%Y%m%d_%H%M%S).log

# Phase 1.3: Train low-priority symbols (112 symbols, ~30 hours)
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 \
  scripts/train_math_symbols_batch.py \
  --priority low \
  --lr 0.5 \
  --epochs 1500 \
  --fonts 0 \
  --max-epochs 3000 \
  2>&1 | tee /K3D/Knowledge3D.local/logs/phase1_low_priority_$(date +%Y%m%d_%H%M%S).log
```

### Step 3: Monitor Training Progress

**From another terminal (DO NOT disturb tmux session):**

```bash
# Attach to watch training (read-only)
tmux attach-session -t k3d_foundation -r

# Detach without stopping training: Ctrl+B, then D

# Check latest log
tail -f /K3D/Knowledge3D.local/logs/phase1_high_priority_*.log

# Check trained symbols
ls -lht /K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/ | head -30

# Count trained symbols
ls /K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/*.npz | wc -l
```

### Step 4: Validate Phase 1 Completion

**Success Criteria:**
- ✅ 152 math symbols trained @ 85%+ accuracy
- ✅ 12 Portuguese diacritics trained @ 85%+ accuracy
- ✅ All symbols stored in Math Galaxy
- ✅ ProceduralGalaxy compression: 69-80:1 ratio

**Validation Test:**
```python
# Inside Python REPL (k3d-cranium environment)
from knowledge3d.cranium.math_galaxy import MathGalaxy
import numpy as np

galaxy = MathGalaxy()

# Check symbol count
symbol_count = len(list(galaxy.symbols_dir.glob("*.npz")))
print(f"Trained symbols: {symbol_count}/164")

# Validate specific symbols
test_symbols = ["∑", "∫", "∂", "∇", "α", "β", "γ"]
for sym in test_symbols:
    embedding = galaxy.load_symbol(sym)
    if embedding is not None:
        print(f"✓ {sym} loaded: {embedding.shape}")
    else:
        print(f"✗ {sym} MISSING")

# Check compression ratio
import os
total_size = sum(
    os.path.getsize(f)
    for f in galaxy.symbols_dir.glob("*.npz")
)
print(f"Total storage: {total_size / 1024:.2f} KB")
print(f"Per symbol: {total_size / symbol_count:.2f} bytes")
```

---

## Phase 2-6 Overview (Execute After Phase 1)

### Phase 2: Language/Grammar PDFs (Week 2)
- **Objective:** Ingest 10 PDFs (1,805 pages) → Layer 2 (words) + Layer 3 (grammar rules)
- **Key Task:** Extend `grammar_galaxy.py` with symlink support
- **Deliverable:** 15,000+ words, 50+ grammar RPN programs

### Phase 3: Math/Context/Time PDFs (Week 3)
- **Objective:** Ingest 31 PDFs (2,509 pages) → Layer 3 expansion
- **Key Task:** Generate 950+ RPN programs (math, temporal, contextual)
- **Deliverable:** Complete Layer 3 operational

### Phase 4: Meta-Rules PDFs (Week 4)
- **Objective:** Ingest 24 PDFs (1,674 pages) → Layer 4 (eloquence, self-reflection, storytelling, delivery)
- **Key Task:** Create `meta_rules_galaxy.py`, integrate with sleeptime consolidation
- **Deliverable:** 500+ meta-rule RPN programs

### Phase 5: Cross-Domain Discovery (Week 5)
- **Objective:** Build discovery layer, generate 300+ emergent connections
- **Key Task:** Create `discovery_layer.py`, analyze shared symbol references
- **Deliverable:** Cross-domain discovery operational

### Phase 6: Integration Tests (Week 6)
- **Objective:** Validate complete 4-layer architecture
- **Key Tests:** Symlink integrity, RPN execution, sleeptime consolidation, ARC-AGI >5% improvement
- **Deliverable:** Production-ready foundational knowledge base

**Full details:** See `TEMP/KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md` Section 5 (6-week roadmap).

---

## Tmux Best Practices for Long-Running Training

### Session Management
```bash
# List active sessions
tmux ls

# Create new session with custom name
tmux new-session -s <name>

# Attach to existing session
tmux attach-session -t <name>

# Detach from session (keeps it running)
# Inside tmux: Ctrl+B, then D

# Kill session (only when training complete!)
tmux kill-session -t <name>
```

### Window Management (Inside Tmux)
```bash
# Create new window: Ctrl+B, then C
# Switch windows: Ctrl+B, then 0-9
# Rename window: Ctrl+B, then ,
# Split pane horizontally: Ctrl+B, then "
# Split pane vertically: Ctrl+B, then %
```

### Monitoring Setup
```bash
# Recommended: 2-window setup in k3d_foundation session

# Window 0: Training (main process)
# - Run train_math_symbols_batch.py here
# - Let it run undisturbed

# Window 1: Monitoring
# - tail -f logs
# - ls checkpoints
# - Quick validation checks
```

---

## Sovereignty Compliance Checklist

**Before implementing ANY feature, verify:**

1. **Hot Path Clean:**
   ```bash
   # Should return NOTHING
   grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
   grep "import numpy" knowledge3d/cranium/reality_galaxy.py
   grep -r "import numpy" knowledge3d/cranium/bridges/
   ```

2. **Ingestion Path Flexible:**
   ```bash
   # OK to use numpy/pandas/PIL here (not hot path)
   ls knowledge3d/ingestion/
   ```

3. **Save Information Principle:**
   - ✅ Use references (symlink pattern)
   - ❌ NO duplicate embeddings
   - ❌ NO duplicate strings in semantic tags

4. **Dual Client Reality:**
   - ✅ Procedural RPN + metadata (humans AND AI)
   - ✅ Form + Meaning at each layer
   - ❌ NO bitmap-only representations

---

## Communication Protocol

### Progress Reports (Send to Claude/Daniel)

**Format:**
```markdown
## Phase 1 Progress Report — [Date]

**Status:** [In Progress / Blocked / Complete]

**Completed:**
- High-priority symbols trained: 25/25 (100%)
- Medium-priority symbols trained: 15/15 (100%)
- Low-priority symbols: 45/112 (40%)

**Current Task:**
- Training low-priority symbols in tmux session k3d_foundation
- ETA: 18 hours remaining

**Metrics:**
- Training time: 12 hours elapsed
- Average accuracy: 89.3%
- Storage efficiency: 76:1 compression ratio

**Blockers:**
- None

**Next Steps:**
- Complete low-priority training
- Run Phase 1 validation tests
- Begin Phase 2 planning
```

### Blocker Escalation

**If you encounter blockers:**

1. **Document the issue:**
   - What you tried
   - Error messages (full stack trace)
   - Environment details

2. **Check existing documentation:**
   - Search TEMP/ for similar issues
   - Review test files for examples
   - Check git log for recent fixes

3. **Escalate to Claude:**
   - Clear problem statement
   - Attempted solutions
   - Proposed architectural change (if needed)

---

## Key Architectural Principles (From CLAUDE.md)

### 1. Save Information Principle (Lines 94-100)

**DON'T duplicate what exists!** Use references (symlink pattern):
- Characters already have font + language + meaning
- Words reference character IDs (not duplicate glyphs)
- Grammar metadata references words (not duplicate strings)
- Discoveries reference canonical programs

**Example (WRONG):**
```python
discovery = {
    "transformation_type": "rotation_or_reflection",  # STRING duplicated!
}
```

**Example (CORRECT):**
```python
discovery = {
    "transformation_type": word_ref("rotation_or_reflection"),  # Reference
}
```

### 2. Dual Client Reality (Lines 81-127)

**K3D serves TWO clients with the SAME data — Humans AND AI.**

**Procedural Foundation:**
- Drawing Galaxy → Visual primitives (LINE, CIRCLE, RECT as RPN)
- Character Galaxy → Glyphs (Bézier → segments) + language metadata
- Word Level → Character sequences (references, not duplicates)
- Grammar Galaxy → Transformation rules (RPN) + context metadata

**When Designing:**
1. Does this already exist in procedural form?
2. Can I reference existing data instead of duplicating?
3. Does this work for BOTH humans (readable) AND AI (executable)?
4. Is the metadata attached to the right layer?

### 3. Sovereignty Guardrails (BRIEFING.md Lines 193-204)

**Hot Path (inference loop):**
- ✅ Allowed: ctypes, libcuda.so, native Python (math, list, dict)
- ❌ Forbidden: numpy, torch, tensorflow, cupy

**Ingestion Path (preprocessing):**
- ✅ Allowed: numpy, pandas, PIL, pygltflib, matplotlib
- ✅ Condition: NEVER called during `galaxy.step_system()`

---

## File Locations Reference

### Implementation Files (You'll Modify These)
```bash
# Phase 1
scripts/train_math_symbols_batch.py                  # Your main training script
knowledge3d/cranium/math_galaxy.py                   # Symbol storage
knowledge3d/cranium/math_symbols_registry.py         # Symbol definitions
scripts/train_atomic_character.py                    # Base trainer

# Phase 2
knowledge3d/cranium/grammar_galaxy.py                # Extend with symlink support
scripts/ingest_language_pdfs.py                      # Create this
knowledge3d/ingestion/pdf_extractor.py               # Create this

# Phase 3
scripts/ingest_math_pdfs.py                          # Create this
knowledge3d/cranium/temporal_reasoning.py            # Extend this

# Phase 4
knowledge3d/cranium/meta_rules_galaxy.py             # Create this
scripts/ingest_meta_rules_pdfs.py                    # Create this

# Phase 5
knowledge3d/cranium/discovery_layer.py               # Create this

# Phase 6
tests/integration/test_foundational_knowledge.py     # Create this
```

### Documentation Files (Read These)
```bash
CLAUDE.md                                            # Your partnership model (151 lines)
CODEX.md                                             # Your role (134 lines)
BRIEFING.md                                          # Current status (349 lines)
docs/Briefings/SOVEREIGN_SWARM_BRIEFING_v3.md        # Complete arch (863 lines)
docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md  # What you're building
TEMP/KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md     # Your execution plan
```

### Runtime Directories
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/             # Conda environment
/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/  # Training checkpoints
/K3D/Knowledge3D.local/logs/                         # Training logs
/K3D/Knowledge3D.local/procedural_galaxy/            # Galaxy storage
```

---

## Success Criteria (Phase 1)

**Technical Metrics:**
- [x] 152 math symbols trained @ 85%+ accuracy
- [x] 12 Portuguese diacritics trained @ 85%+ accuracy
- [x] All symbols stored in Math Galaxy
- [x] ProceduralGalaxy compression: 69-80:1 ratio
- [x] Total storage: <2 MB for all 164 symbols

**Validation Tests:**
- [x] Load all 164 symbols from Math Galaxy
- [x] Verify embeddings shape (consistent dimensions)
- [x] Check compression ratio
- [x] Validate no duplicate embeddings

**Documentation:**
- [x] Training logs saved to `/K3D/Knowledge3D.local/logs/`
- [x] Progress report sent to Claude
- [x] Checkpoints committed (if requested by Daniel)

---

## Example Session Transcript (What Success Looks Like)

```bash
$ tmux new-session -s k3d_foundation

[k3d_foundation:0]$ source /K3D/Knowledge3D.local/envs/k3d-cranium/bin/activate
(k3d-cranium) [k3d_foundation:0]$ export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
(k3d-cranium) [k3d_foundation:0]$ export CUDA_VISIBLE_DEVICES=0

(k3d-cranium) [k3d_foundation:0]$ cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

(k3d-cranium) [k3d_foundation:0]$ CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/train_math_symbols_batch.py --priority high --lr 0.5 --epochs 1500 --fonts 0 --max-epochs 3000 2>&1 | tee /K3D/Knowledge3D.local/logs/phase1_high_priority_$(date +%Y%m%d_%H%M%S).log

================================================================================
BATCH MATH SYMBOL TRAINING
================================================================================
Math Galaxy initialized: /K3D/Knowledge3D.local/procedural_galaxy/math_symbols
Training priority: high
Total symbols to train: 25

================================================================================
SYMBOL 1/25: '∑' (U+2211)
================================================================================
[Training progress...]
✓ Symbol '∑' trained and stored in Math Galaxy
  Accuracy: 91.23%
  Embeddings: (50, 128)

[... continues for all 25 symbols ...]

================================================================================
BATCH TRAINING COMPLETE
================================================================================
Total symbols: 25
Skipped (existing): 1
Trained: 24
Failed: 0

Accuracy statistics:
  Mean: 89.45%
  Min: 85.12%
  Max: 94.67%

Math Galaxy storage: /K3D/Knowledge3D.local/procedural_galaxy/math_symbols
================================================================================

[Detach with Ctrl+B, D]
```

---

## Final Checklist Before Starting

- [ ] Read ALL mandatory documents (CLAUDE.md, CODEX.md, BRIEFING.md, SOVEREIGN_SWARM_BRIEFING_v3.md)
- [ ] Read FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md completely
- [ ] Read KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md completely
- [ ] Understand sovereignty principles (hot path = PTX + RPN only)
- [ ] Understand Save Information Principle (symlink pattern, no duplication)
- [ ] Understand Dual Client Reality (procedural RPN + metadata)
- [ ] Verify Run 039 is running (DO NOT disturb it)
- [ ] Create tmux session `k3d_foundation`
- [ ] Activate k3d-cranium environment
- [ ] Set PYTHONPATH and CUDA_VISIBLE_DEVICES
- [ ] Start Phase 1 training (high → medium → low priority)
- [ ] Monitor progress via tmux attach + tail logs
- [ ] Validate Phase 1 completion
- [ ] Send progress report to Claude

---

## Questions? Need Clarification?

**Architecture questions:** Ask Claude (tag @Claude in conversation)
**Implementation blockers:** Document and escalate per "Blocker Escalation" section above
**Environment issues:** Check `docs/ENV_POLICY.md` and SOVEREIGN_SWARM_BRIEFING_v3.md Section 4

---

## Let's Build This! 🚀

You are implementing the foundational knowledge system that will give K3D always-loaded base knowledge across mathematics, language, grammar, eloquence, self-reflection, storytelling, and delivery.

This is a **6-week journey** starting with Phase 1 (training 164 foundational characters) and culminating in a complete 4-layer architecture with 300+ cross-domain discoveries.

**Your first task:** Execute Phase 1 using tmux + k3d-cranium environment. Start with high-priority symbols, monitor progress, validate completion.

**Remember:**
- Sovereignty: Hot path = PTX + RPN only
- Save Information: Symlink pattern, no duplication
- Dual Client Reality: Procedural RPN + metadata (humans AND AI)

**Communication:** Send progress reports, escalate blockers early, commit with clear messages.

**Partnership:** Claude designed this architecture. You're implementing it. We build together.

---

**Ready to start? Create tmux session `k3d_foundation` and begin Phase 1 training!**

---

**End of Briefing**

**Version:** 1.0
**Date:** December 5, 2025
**Author:** Claude (Architecture)
**For:** Codex (Implementation Lead)
**Phase:** Foundational Knowledge Ingestion (Phases 1-6)
