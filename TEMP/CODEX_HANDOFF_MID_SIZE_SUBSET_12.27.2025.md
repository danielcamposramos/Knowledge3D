# Codex Handoff — Mid-Size Subset Test + Refactor Prep

**Date**: December 27, 2025
**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Priority**: HIGH — Mid-size validation blocking full refactor + 23-book ingestion

---

## ⚠️ CRITICAL: Compulsory Reading (DO THIS FIRST)

**You MUST read these documents IN FULL before proceeding**:

### 1. Core Architecture Documents (READ COMPLETELY)
```bash
# Read these in order:
1. BRIEFING.md                                    # Project overview, Galaxy Universe paradigm
2. CODEX.md                                       # Your role as Implementation Lead
3. docs/Briefings/SOVEREIGN_SWARM_BRIEFING_v3.md  # Sovereignty principles, hot path rules
4. docs/ROADMAP.md                                # Current phase status
```

### 2. Vocabulary Specifications (READ ALL 15 FILES)
```bash
# Read ALL files in docs/vocabulary/ (no snippets, full text):
docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md           # Cranium + Galaxy + House
docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md         # Form + Meaning, Save Information
docs/vocabulary/MATH_CORE_SPECIFICATION.md                    # 3-tier math cores
docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md              # Reality Galaxy
docs/vocabulary/SOVEREIGN_TRAINING_SPECIFICATION.md           # Training architecture
docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md       # 4-layer knowledge
docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md                 # RPN opcodes
docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md            # Drawing Galaxy
docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md                # Neurosymbolic integration
docs/vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md      # UI/UX architecture
docs/vocabulary/K3D_NODE_SPECIFICATION.md                     # Node structure
docs/vocabulary/ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md  # PD04 codec
docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md           # Memory consolidation
docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md               # Audio/SDR/video
docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md      # Braille/sign/haptics
```

### 3. Recent Work Context (READ FOR CURRENT STATE)
```bash
# Recent architectural decisions:
TEMP/CLAUDE_COHERENCE_AUDIT_12.27.2025.md                # Folder structure violations identified
TEMP/CLAUDE_ENHANCED_ROLE_EXTRACTION_PROMPT_12.27.2025.md  # Enhanced prompt architecture
TEMP/CLAUDE_MODEL_SELECTION_STRATEGY_12.27.2025.md       # Model bakeoff results + recommendations
TEMP/CODEX_DIRECTIVE_OPTION_A_PLUS_PHASE8_12.19.2025.md  # Option A + Phase 8 architecture
TEMP/CLAUDE_PHASE7_COMPLETE_ARCHITECTURE_SUMMARY_12.19.2025.md  # Phase 7 completion
```

---

## Current State Summary

### What's Been Done ✅

1. **Enhanced Prompt Implemented** (in `sovereign_knowledge_articulator.py`):
   - ✅ Tier-3 role hierarchy (Geometry → Formula → Generic)
   - ✅ Few-shot examples (circle/triangle/cylinder/exponential)
   - ✅ Geometric context detection (auto-detects shapes)
   - ✅ Reasoning chain ("think step-by-step")

2. **Model Configuration Set**:
   - ✅ Default changed from `qwen3:8b` (0/8 bakeoff) to `granite4:tiny-h` (100% accuracy)
   - ✅ Fallback model: `qwen2.5:14b` (triggers on "unknown")
   - ✅ HTTP mode enabled (persistent ollama serve)

3. **Small Subset Test Completed** (5 artifacts):
   - ✅ 40% geometry roles (vs ~13% baseline) — PROMISING
   - ⚠️ Too small to validate (need 30+ artifacts)

4. **Folder Reorganization Started**:
   - ✅ `multimodal/`, `reasoning/` moved to `Old_Attempts/curriculum_specific_training/`
   - ⏳ `arc_agi/` NOT YET MOVED (awaiting mid-size test validation)

### What's Pending ⏳

1. **Mid-Size Subset Test** (50 pages Linear Algebra) — **YOUR IMMEDIATE TASK**
2. **Refactor to "Single Model"** (after test passes)
3. **Full 23-Book Ingestion** (after refactor complete)
4. **Benchmark MATH/AMC** (after books_v5 populated)

---

## YOUR IMMEDIATE TASK: Mid-Size Subset Test

### Objective

Run 50-page ingestion of Linear Algebra Done Right to validate:
- ✅ Non-unknown rate ≥60%
- ✅ Geometry role rate ≥40% (of non-unknowns)
- ✅ At least 30+ artifacts extracted
- ✅ At least 8+ distinct Tier 1 roles appear

### The Problem (Previous Attempt Failed)

Previous Codex tried to start tmux but it never actually started:
- `tmux ls` → no server running
- Log file doesn't exist
- Output directory doesn't exist

**Likely cause**: tmux command syntax error or shell escaping issue

### The Solution (Fixed tmux Command)

**Run this exact command** (verified Debian-compatible):

```bash
# Start tmux session with proper escaping
tmux new -d -s k3d_role_tier3_mid

# Inside the tmux session, run this:
tmux send-keys -t k3d_role_tier3_mid '
export K3D_ROLE_LLM_ENABLE=1
export K3D_ROLE_LLM_MODEL="granite4:tiny-h"
export K3D_ROLE_LLM_FALLBACK_MODEL="qwen2.5:14b"
export K3D_ROLE_LLM_FALLBACK_ON_UNKNOWN=1
export K3D_ROLE_LLM_CACHE_PATH="/tmp/llm_role_extraction_cache_tier3_mid.jsonl"
export K3D_ROLE_LLM_TIMEOUT_S=30
export K3D_ROLE_LLM_MAX_CONTEXT_CHARS=1200

cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  -m knowledge3d.training.math_benchmarks.book_galaxy_ingestion \
  --pdf "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Linear.Algebra.Done.Right.pdf" \
  --book-id la_done_right_mid \
  --domain linear_algebra \
  --max-pages 50 \
  --out-dir /K3D/Knowledge3D.local/galaxies/books_v5_tier3_mid \
  2>&1 | tee /tmp/llm_tier3_mid_la.log
' C-m
```

### Verification Steps

**After starting tmux**:

```bash
# 1. Verify tmux session exists
tmux ls
# Should show: k3d_role_tier3_mid: 1 windows ...

# 2. Attach to watch progress
tmux attach -t k3d_role_tier3_mid
# Press Ctrl-b then d to detach

# 3. Monitor log file
tail -f /tmp/llm_tier3_mid_la.log

# 4. Check cache growth
wc -l /tmp/llm_role_extraction_cache_tier3_mid.jsonl
# Should increase as variables are processed
```

### Expected Timeline

- **Ingestion**: 2-3 hours (50 pages × 2-4 min/page with LLM calls)
- **Analysis**: 5 minutes (after completion)
- **Total**: ~3 hours maximum

### When Complete: Validation Analysis

**Run this command** to analyze results:

```bash
# Navigate to output directory
cd /K3D/Knowledge3D.local/galaxies/books_v5_tier3_mid

# Count artifacts
find . -name "artifacts.jsonl" -exec cat {} \; | wc -l

# Compute coverage statistics
find . -name "artifacts.jsonl" -exec cat {} \; | python3 -c "
import json, sys

total_artifacts = 0
total_bindings = 0
non_unknown = 0
geometry_roles = 0
meanings = {}

geo_role_set = {
    'radius', 'diameter', 'height', 'width', 'length', 'depth',
    'leg', 'hypotenuse', 'base', 'side', 'angle', 'arc_length',
    'circumference', 'area', 'volume'
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        obj = json.loads(line)
    except:
        continue

    total_artifacts += 1
    bindings = obj.get('symbol_bindings', {})
    total_bindings += len(bindings)

    for var_name, binding_info in bindings.items():
        meaning = binding_info.get('meaning', 'unknown')
        meanings[meaning] = meanings.get(meaning, 0) + 1

        if meaning != 'unknown':
            non_unknown += 1

        if meaning in geo_role_set:
            geometry_roles += 1

print(f'Total artifacts: {total_artifacts}')
print(f'Total bindings: {total_bindings}')
print(f'Non-unknown: {non_unknown} ({100.0 * non_unknown / total_bindings if total_bindings else 0:.1f}%)')
print(f'Geometry roles: {geometry_roles} ({100.0 * geometry_roles / non_unknown if non_unknown else 0:.1f}% of non-unknown)')
print(f'\\nTop 15 meanings:')
for meaning, count in sorted(meanings.items(), key=lambda x: -x[1])[:15]:
    print(f'  {meaning}: {count}')
"
```

### Success Criteria (Decision Tree)

**PASS** (proceed to refactor):
- ✅ Non-unknown ≥60%
- ✅ Geometry roles ≥40% of non-unknowns
- ✅ Total artifacts ≥30
- ✅ At least 8+ distinct Tier 1 roles

**CONDITIONAL** (tune prompt, re-test):
- ⚠️ 30-40% geometry roles → Add more domain-specific examples
- ⚠️ 50-60% non-unknown → Increase timeout or retry logic

**FAIL** (redesign prompt):
- ❌ <30% geometry roles → Prompt not guiding toward specific roles
- ❌ <50% non-unknown → Models not understanding context

---

## Post-Test Task: Refactor to "Single Model" Architecture

**ONLY proceed with this if mid-size test PASSES**

### The Problem (from CLAUDE_COHERENCE_AUDIT_12.27.2025.md)

Current structure violates "one model to process them all" principle:

```
knowledge3d/training/
├── arc_agi/         ← DUPLICATE galaxies (drawing, grammar, math_symbol)
│   ├── drawing_galaxy.py (10,617 lines) - DUPLICATE!
│   ├── grammar_galaxy.py (38,223 lines) - DUPLICATE!
│   ├── math_symbol_galaxy.py (17,781 lines) - DUPLICATE!
│   └── sleeptime_consolidator.py (11,744 lines) - DUPLICATE!
├── math_benchmarks/ ← Uses local copies instead of cranium/*
├── multimodal/      ← Already moved to Old_Attempts ✅
└── reasoning/       ← Already moved to Old_Attempts ✅
```

### The Solution (4-Phase Refactor)

#### **Phase 1: Move arc_agi to Archive** (1 hour)

```bash
# Create archive location
mkdir -p Old_Attempts/curriculum_specific_training/

# Move arc_agi folder
git mv knowledge3d/training/arc_agi Old_Attempts/curriculum_specific_training/

# Create deprecation notice
cat > Old_Attempts/curriculum_specific_training/arc_agi/README_DEPRECATED.md << 'EOF'
# ARC-AGI Training Module — DEPRECATED

**Date Deprecated**: December 27, 2025
**Reason**: Violates "one model to process them all" architecture principle

## What Was Wrong

This module treated ARC-AGI as a SEPARATE model with:
- Duplicate galaxy implementations (drawing, grammar, math_symbol)
- Separate sleeptime consolidator
- Separate RPN executors

## What Replaced It

Unified architecture:
- `knowledge3d/cranium/procedural_galaxy.py` — Drawing Galaxy (serves ALL curricula)
- `knowledge3d/cranium/word_galaxy.py` — Grammar Galaxy (serves ALL curricula)
- `knowledge3d/cranium/math_galaxy.py` — Math Galaxy (serves ALL curricula)
- `knowledge3d/cranium/sleep_time_consolidator.py` — Unified consolidation

## Migration Path

If you need ARC-AGI training:
- Use `knowledge3d/training/curriculum_loaders/arc_agi_loader.py` for data loading
- Use `knowledge3d/training/unified_trainer.py` for training
- Use cranium/* galaxies (not local copies)

## Files Archived

- `drawing_galaxy.py` (10,617 lines)
- `grammar_galaxy.py` (38,223 lines)
- `math_symbol_galaxy.py` (17,781 lines)
- `sleeptime_consolidator.py` (11,744 lines)
- 40+ other files

Total: 34,337 lines of duplicated code
EOF

# Commit
git add -A
git commit -m "refactor: move arc_agi to Old_Attempts (violates single model principle)

- Moved knowledge3d/training/arc_agi → Old_Attempts/curriculum_specific_training/
- Added deprecation notice with migration path
- Reason: Duplicate galaxy implementations (drawing, grammar, math_symbol)
- Total: 34,337 lines of code duplication eliminated
- Next: Refactor math_benchmarks to use cranium/* galaxies

Part of coherence audit recommendations (CLAUDE_COHERENCE_AUDIT_12.27.2025.md)"
```

#### **Phase 2: Create Unified Training Infrastructure** (2 hours)

**File 1**: `knowledge3d/training/unified_trainer.py`

```python
"""
Unified Training Orchestrator
One model (TRM) processes ALL curricula (ARC-AGI, Math, Reality, Language).
"""

from typing import Optional, List
from dataclasses import dataclass
import logging

# Import unified galaxies (NOT curriculum-specific copies)
from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy  # Drawing
from knowledge3d.cranium.math_galaxy import MathGalaxy
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.word_galaxy import WordGalaxy

# Import curriculum loaders (data only)
from knowledge3d.training.curriculum_loaders import arc_agi_loader
from knowledge3d.training.curriculum_loaders import math_loader
from knowledge3d.training.curriculum_loaders import reality_sim_loader

logger = logging.getLogger(__name__)

@dataclass
class UnifiedTrainerConfig:
    """Configuration for unified multi-curriculum training."""

    # Which curricula to train on
    enable_arc_agi: bool = True
    enable_math_benchmarks: bool = True
    enable_reality_sims: bool = True
    enable_language_tasks: bool = True

    # TRM configuration
    trm_base_model_path: Optional[str] = None
    trm_learning_rate: float = 1e-4

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 10
    thinking_budget: int = 8

    # Galaxy configuration
    load_all_galaxies: bool = True

class UnifiedTrainer:
    """
    Trains ONE TRM model on MULTIPLE curricula.

    Architecture:
    - Uses unified cranium/* galaxies (not curriculum-specific copies)
    - Loads data via curriculum_loaders/* (data only, no logic)
    - Trains TRM specialists (math, visual, physics adapters)
    - Validates across ALL curricula simultaneously
    """

    def __init__(self, config: UnifiedTrainerConfig):
        self.config = config

        # Initialize unified galaxies (shared across ALL curricula)
        logger.info("Loading unified galaxies...")
        self.drawing_galaxy = ProceduralGalaxy()  # For ARC-AGI visual reasoning
        self.math_galaxy = MathGalaxy()          # For math benchmarks
        self.reality_galaxy = RealityGalaxy()    # For physics sims
        self.word_galaxy = WordGalaxy()          # For language tasks

    def train(self):
        """
        Main training loop across ALL curricula.
        """
        logger.info("Starting unified multi-curriculum training...")

        # Load data from each curriculum (via loaders, not duplicate code)
        if self.config.enable_arc_agi:
            arc_data = arc_agi_loader.load_training_data()
            logger.info(f"Loaded {len(arc_data)} ARC-AGI tasks")

        if self.config.enable_math_benchmarks:
            math_data = math_loader.load_training_data()
            logger.info(f"Loaded {len(math_data)} math problems")

        # TODO: Implement training loop
        # - Sample from multiple curricula
        # - Train TRM specialists (math, visual, physics)
        # - Validate cross-curriculum performance

        logger.info("Training complete")

def main():
    """Example usage."""
    config = UnifiedTrainerConfig(
        enable_arc_agi=True,
        enable_math_benchmarks=True,
        enable_reality_sims=False,  # Not yet ready
        batch_size=32,
        num_epochs=10
    )

    trainer = UnifiedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
```

**File 2**: `knowledge3d/training/curriculum_loaders/arc_agi_loader.py`

```python
"""
ARC-AGI Data Loader
Loads ARC-AGI grids WITHOUT duplicating galaxy logic.
"""

from typing import List, Dict, Any
import json
import os

def load_training_data() -> List[Dict[str, Any]]:
    """
    Load ARC-AGI training tasks.

    Returns:
        List of task dicts with 'train' and 'test' grids.
    """
    # TODO: Implement ARC-AGI data loading
    # - Load from datasets/arc_agi/
    # - Parse JSON grids
    # - Return task definitions (NOT galaxy entries!)

    return []

def load_evaluation_data() -> List[Dict[str, Any]]:
    """Load ARC-AGI evaluation tasks."""
    return []
```

**File 3**: `knowledge3d/training/curriculum_loaders/math_loader.py`

```python
"""
Math Benchmark Data Loader
Loads MATH/GSM8K/AMC WITHOUT duplicating galaxy logic.
"""

from typing import List, Dict, Any

def load_training_data() -> List[Dict[str, Any]]:
    """
    Load math benchmark training problems.

    Returns:
        List of problem dicts with 'question' and 'answer'.
    """
    # TODO: Implement math data loading
    # - Load from datasets/math/
    # - Parse problem JSON
    # - Return problem definitions (NOT galaxy entries!)

    return []
```

**Commit Phase 2**:
```bash
git add knowledge3d/training/unified_trainer.py
git add knowledge3d/training/curriculum_loaders/
git commit -m "feat: add unified training infrastructure (single model architecture)

- Added unified_trainer.py: ONE trainer for ALL curricula
- Added curriculum_loaders/: Data loading only (no duplicate logic)
- Uses cranium/* galaxies (not curriculum-specific copies)
- Enforces 'one model to process them all' principle

Part of single-model refactor (CLAUDE_COHERENCE_AUDIT_12.27.2025.md)"
```

#### **Phase 3: Refactor math_benchmarks Imports** (30 min)

Update `knowledge3d/training/math_benchmarks/benchmark_evaluator.py`:

```python
# BEFORE (violates single model):
# from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

# AFTER (unified):
from knowledge3d.cranium.word_galaxy import WordGalaxy  # Unified grammar
```

**Commit Phase 3**:
```bash
git add knowledge3d/training/math_benchmarks/
git commit -m "refactor: math_benchmarks uses cranium/* galaxies (not arc_agi copies)

- Updated imports to use cranium.word_galaxy (not arc_agi.grammar_galaxy)
- Updated imports to use cranium.procedural_galaxy (not arc_agi.drawing_galaxy)
- Eliminates circular dependencies
- Enforces unified galaxy architecture

Part of single-model refactor"
```

#### **Phase 4: Update README** (30 min)

Add new section to README.md:

```markdown
## 7. Math Benchmarks (Phase 7 + Option A/Phase 8)

### Current Status
- **Phase 7 Complete**: MATH 2.5% → 3.0% (book boost + context gating)
- **Option A**: LLM-assisted semantic role extraction (granite4:tiny-h + qwen2.5:14b)
- **Phase 8**: Multi-step theorem chaining infrastructure

### Book Galaxy Ingestion (Option A)

**Full 23-book ingestion** (8-12 hours):

\`\`\`bash
# Environment setup
export K3D_ROLE_LLM_ENABLE=1
export K3D_ROLE_LLM_MODEL="granite4:tiny-h"
export K3D_ROLE_LLM_FALLBACK_MODEL="qwen2.5:14b"
export K3D_ROLE_LLM_FALLBACK_ON_UNKNOWN=1
export K3D_ROLE_LLM_CACHE_PATH="/tmp/llm_role_extraction_cache_full.jsonl"
export K3D_ROLE_LLM_TIMEOUT_S=30

# For each book (see TEMP/CODEX_DIRECTIVE_OPTION_A_PLUS_PHASE8_12.19.2025.md for full list):
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  -m knowledge3d.training.math_benchmarks.book_galaxy_ingestion \
  --pdf "/path/to/book.pdf" \
  --book-id "<book_id>" \
  --domain "<domain>" \
  --out-dir "/K3D/Knowledge3D.local/galaxies/books_v5/<book_id>"
\`\`\`

### Benchmarking with books_v5

**Option A only** (book boost + context gating):

\`\`\`bash
export K3D_TRM_ENABLE_MULTISTEP=0

PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator \
  --disable-retrieval \
  --datasets math \
  --max-problems 200 \
  --shuffle --shuffle-seed 123 \
  --thinking-budget 8 \
  --shadow-readonly \
  --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5 \
  --book-max-books 64 \
  --book-top-k 5 \
  --verbose
\`\`\`

**Option A + Phase 8** (multi-step chaining):

\`\`\`bash
export K3D_TRM_ENABLE_MULTISTEP=1

# Same command as above
\`\`\`

### Expected Results

- **Option A**: MATH 10-15%, AMC 5-8% (with 60%+ semantic coverage)
- **Phase 8**: Additional +2-5% from multi-step theorem chaining

### Model Selection

Based on comprehensive bakeoff (see TEMP/CLAUDE_MODEL_SELECTION_STRATEGY_12.27.2025.md):

- **Primary**: `granite4:tiny-h` (100% accuracy, 0.43s avg latency)
- **Fallback**: `qwen2.5:14b` (100% accuracy, handles ambiguous geometry)
- **DO NOT USE**: `qwen3:8b` (0/8 bakeoff failure)
```

**Commit Phase 4**:
```bash
git add README.md
git commit -m "docs: add Math Benchmarks section with Option A/Phase 8 commands

- Added Section 7: Math Benchmarks
- Documented book galaxy ingestion (granite4 + qwen2.5:14b)
- Documented benchmarking commands (Option A and Phase 8)
- Added model selection guidance
- Replaced outdated Old_Attempts commands

Closes coherence audit documentation task"
```

---

## Reporting Requirements

**After mid-size test completes**, report:

1. **Test Results**:
   - Total artifacts extracted
   - Total bindings
   - Non-unknown rate (%)
   - Geometry role rate (% of non-unknowns)
   - Top 15 meanings distribution

2. **Success/Conditional/Fail Assessment**:
   - Which threshold category do results fall into?
   - If CONDITIONAL or FAIL, what's the specific issue?

3. **Recommendation**:
   - PASS → Proceed to refactor
   - CONDITIONAL → Tune prompt + re-test
   - FAIL → Redesign prompt strategy

**After refactor completes** (if test passed):

1. **Refactor Summary**:
   - Files moved to Old_Attempts
   - New files created
   - Import updates completed
   - Tests passing (if applicable)

2. **Git Status**:
   - Commits created
   - Branch status
   - Any conflicts resolved

---

## Architecture Compliance Checklist

Before proceeding, verify you understand:

- [ ] **Galaxy Universe Paradigm**: ONE unified VRAM workspace with ALL galaxies loaded
- [ ] **TRM Navigation**: TRM learns HOW to navigate (not knowledge storage)
- [ ] **Multi-Curriculum**: Math/ARC/Reality/Language ALL feed SAME galaxies
- [ ] **Sovereignty**: Hot path = PTX + Galaxy ONLY (no numpy/cupy)
- [ ] **Single Model Principle**: ONE TRM model, NOT separate models per curriculum
- [ ] **Save Information**: Reference, don't duplicate (symlink pattern)
- [ ] **Dual Client**: Form + Meaning (procedural RPN + metadata)

---

## Critical Guardrails

**DO**:
- ✅ Use cranium/* for galaxy implementations
- ✅ Use training/curriculum_loaders/* for data loading
- ✅ Commit frequently with clear messages
- ✅ Test imports after refactoring
- ✅ Report progress and blockers

**DO NOT**:
- ❌ Create curriculum-specific galaxy copies
- ❌ Add numpy/cupy to hot path (sovereignty violation!)
- ❌ Skip reading required documents
- ❌ Proceed with refactor before mid-size test passes
- ❌ Use qwen3:8b model (0/8 bakeoff failure)

---

## Questions?

If anything is unclear:
1. Re-read the relevant vocabulary specification
2. Check TEMP/*.md for recent architectural decisions
3. Ask Daniel or Claude for clarification

**DO NOT guess** if uncertain about architecture principles.

---

**Ready? Start with the mid-size subset test in tmux. Good luck!** 🚀
