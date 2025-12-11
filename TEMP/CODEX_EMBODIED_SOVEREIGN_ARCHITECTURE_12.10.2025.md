# CODEX: Embodied Sovereign Architecture — Major Refactor

**Date:** December 10, 2025
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** Critical
**Sovereignty:** MANDATORY — No CPU fallbacks, No NumPy in hot path

---

## Context: Why This Refactor

We ran 162 epochs with 747 ingested grammar rules and got **zero uplift** (46.19% avg vs 46.04% baseline). The rules are loaded but not used. More fundamentally:

**Current Problem:** Each task is treated as a stateless program execution.

**Target State:** Embodied AI that:
1. Sits and works using Galaxy memory (not reloading 747 rules every inference)
2. Accumulates discoveries incrementally
3. Consolidates when memory pressure OR end of question set
4. Learns its own decision trees (not hardcoded thresholds)

---

## Architecture Changes Required

### 1. Galaxy as Persistent Working Memory (NOT reload every task)

**Current (Wrong):**
```python
# Every task reloads everything
grammar_galaxy = get_grammar_galaxy()  # Logs "Loaded 747 rules" every time
result = process_task(task)
```

**Target (Embodied):**
```python
class EmbodiedSovereignAgent:
    def __init__(self):
        # Load once, persist in GPU memory
        self.drawing_galaxy = DrawingGalaxy()  # L0: visual primitives
        self.char_galaxy = CharacterGalaxy()   # L1: glyphs
        self.word_galaxy = WordGalaxy()        # L2: meanings
        self.grammar_galaxy = GrammarGalaxy()  # L3: rules
        self.eloquence_galaxy = EloquenceGalaxy()  # L4: meta-rules

        # Working memory (GPU tensor, not dict)
        self.working_memory = TernaryWorkingMemory(capacity=4096)

    def work_on_task(self, task_id: str, grid: List[List[int]]):
        # Use existing Galaxy state, don't reload
        # Accumulate discoveries in working_memory
        pass

    def should_consolidate(self) -> bool:
        # Memory pressure OR explicit trigger
        return self.working_memory.utilization > 0.85

    def consolidate(self):
        # Move working_memory discoveries to Galaxy (permanent)
        # Clear working_memory
        pass
```

### 2. Drawing Galaxy Rules (CRITICAL — Currently Missing)

ARC-AGI is fundamentally about **visual transformations**. We have Math Galaxy (176 symbols) but almost no Drawing rules.

**Create these in DrawingGalaxy:**

```python
# Core transformation rules (RPN programs, not hardcoded functions)
DRAWING_RULES = {
    # Rotations
    "ROT90_CW": "GRID_H GRID_W SWAP GRID_NEW 0 ROT90_KERNEL APPLY",
    "ROT90_CCW": "GRID_H GRID_W SWAP GRID_NEW 3 ROT90_KERNEL APPLY",
    "ROT180": "ROT90_CW ROT90_CW",

    # Flips
    "FLIP_H": "GRID_W 1 SUB RANGE REVERSE_COLS APPLY",
    "FLIP_V": "GRID_H 1 SUB RANGE REVERSE_ROWS APPLY",
    "FLIP_DIAG": "TRANSPOSE",

    # Scaling
    "SCALE_2X": "GRID_H 2 MUL GRID_W 2 MUL GRID_NEW UPSAMPLE_NN",
    "SCALE_HALF": "GRID_H 2 DIV GRID_W 2 DIV GRID_NEW DOWNSAMPLE_MAX",

    # Tiling
    "TILE_2X2": "DUP HSTACK DUP VSTACK",
    "TILE_3X3": "DUP DUP HSTACK HSTACK DUP DUP VSTACK VSTACK",

    # Color operations
    "RECOLOR": "COLOR_OLD COLOR_NEW GRID_MAP_COLOR",
    "INVERT": "MAX_COLOR SWAP SUB",
    "MASK_COLOR": "COLOR GRID_EQ GRID_MUL",

    # Pattern detection (return pattern, not execute)
    "FIND_OBJECTS": "CONNECTED_COMPONENTS",
    "FIND_SYMMETRY": "SELF FLIP_H EQ SELF FLIP_V EQ OR",
    "FIND_REPETITION": "FFT_2D PEAK_DETECT",

    # Composition
    "OVERLAY": "GRID_A GRID_B GRID_WHERE_NONZERO",
    "SUBTRACT": "GRID_A GRID_B SUB ABS",
    "INTERSECT": "GRID_A GRID_B MIN",
    "UNION": "GRID_A GRID_B MAX",
}
```

**These must be:**
- Pure RPN (no Python functions)
- Executable on PTX (ternary operations)
- Discoverable by the agent (semantic matching)

### 3. Remove Hardcoded Decision Trees

**Current (Wrong) — Hardcoded thresholds:**
```python
# In sovereign_pipeline.py
if fuzzy_score > 0.85:  # HARDCODED
    return REWARD
elif fuzzy_score > 0.70:  # HARDCODED
    return NEUTRAL
```

**Target — Learned thresholds in Galaxy:**
```python
# Thresholds stored as ternary weights in Galaxy
class AdaptiveThresholds:
    def __init__(self, galaxy: GrammarGalaxy):
        # Load from Galaxy, not hardcoded
        self.reward_threshold = galaxy.get_parameter("reward_threshold", default=0.85)
        self.neutral_threshold = galaxy.get_parameter("neutral_threshold", default=0.70)

    def update_from_experience(self, outcomes: List[Tuple[float, bool]]):
        # Learn optimal thresholds from task outcomes
        # Store back to Galaxy (ternary encoded)
        pass
```

**Find and refactor ALL hardcoded numbers:**
- `top_k = 12` → Galaxy parameter
- `fuzzy_threshold = 0.85` → Galaxy parameter
- `confidence += 0.1` → Galaxy parameter
- `batch_size = 4` → Galaxy parameter
- etc.

### 4. Semantic Bridge: Rules → Task Patterns

**Current:** Grammar rules exist but aren't matched to task patterns.

**Target:** Semantic index mapping task hints to applicable rules.

```python
class SemanticRuleBridge:
    """Maps task patterns to applicable Drawing/Grammar rules."""

    # Precomputed embeddings (GPU, ternary)
    pattern_embeddings: TernaryTensor  # Shape: [n_patterns, embed_dim]
    rule_embeddings: TernaryTensor     # Shape: [n_rules, embed_dim]

    def get_applicable_rules(self, task_hints: List[str]) -> List[str]:
        """
        Given semantic hints like ['rotation', 'sparse_grid', 'color_change'],
        return applicable rule IDs from Drawing/Grammar Galaxy.

        MUST use ternary dot product (PTX), not numpy cosine similarity.
        """
        hint_embed = self.encode_hints(task_hints)  # Ternary
        scores = ternary_matmul(hint_embed, self.rule_embeddings.T)  # PTX
        top_rules = ternary_topk(scores, k=self.top_k)  # PTX
        return [self.rules[i] for i in top_rules]
```

### 5. Consolidation Protocol (Sleeptime Enhancement)

**Current:** `sleeptime_consolidator.py` runs after training.

**Target:** Continuous consolidation during embodied operation.

```python
class EmbodiedConsolidator:
    """
    Consolidates working memory to Galaxy when:
    1. Memory utilization > 85%
    2. End of task batch
    3. Explicit trigger

    All operations MUST be PTX (no CPU).
    """

    def consolidate_discoveries(self):
        # 1. Deduplicate working memory (content-hash, ternary)
        unique = self.working_memory.deduplicate()

        # 2. Score by utility (how often used, success rate)
        scores = self.score_discoveries(unique)

        # 3. Merge high-scoring into Galaxy (permanent)
        for discovery, score in zip(unique, scores):
            if score > self.promotion_threshold:
                self.galaxy.add_permanent(discovery)

        # 4. Clear working memory
        self.working_memory.clear()

    def score_discoveries(self, discoveries: List) -> TernaryTensor:
        """
        Score based on:
        - Usage count (how often referenced)
        - Success rate (led to correct answers)
        - Novelty (not duplicate of existing Galaxy content)

        ALL PTX operations.
        """
        pass
```

---

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `knowledge3d/cranium/embodied_agent.py` | **CREATE** | Main embodied agent class |
| `knowledge3d/cranium/drawing_galaxy.py` | **MODIFY** | Add transformation rules (RPN) |
| `knowledge3d/cranium/ternary_working_memory.py` | **CREATE** | GPU-resident working memory |
| `knowledge3d/cranium/semantic_rule_bridge.py` | **CREATE** | Pattern → Rule mapping |
| `knowledge3d/cranium/adaptive_thresholds.py` | **CREATE** | Learned parameters |
| `knowledge3d/training/arc_agi/sovereign_pipeline.py` | **MODIFY** | Use embodied agent, remove hardcoded values |
| `knowledge3d/training/arc_agi/sleeptime_consolidator.py` | **MODIFY** | Continuous consolidation |
| `scripts/train_arc_sovereign_loop.py` | **MODIFY** | Use embodied agent |

---

## Implementation Order

### Phase 1: Drawing Rules (Day 1)
1. Add 20-30 core transformation rules to DrawingGalaxy as RPN programs
2. Create PTX kernels for: ROT90, FLIP, SCALE, TILE, RECOLOR
3. Test each rule in isolation

### Phase 2: Remove Hardcoded Values (Day 1-2)
1. Grep for all hardcoded numbers in sovereign_pipeline.py
2. Create `adaptive_thresholds.py` with Galaxy-backed parameters
3. Replace hardcoded values with Galaxy lookups

### Phase 3: Semantic Bridge (Day 2)
1. Create semantic_rule_bridge.py
2. Precompute embeddings for all Drawing/Grammar rules
3. Wire into candidate generation (replace current hint expansion)

### Phase 4: Embodied Agent (Day 2-3)
1. Create embodied_agent.py with persistent Galaxy state
2. Create ternary_working_memory.py (GPU tensor)
3. Implement consolidation triggers

### Phase 5: Integration & Test (Day 3)
1. Modify train_arc_sovereign_loop.py to use EmbodiedSovereignAgent
2. Run 162 epochs, compare to baseline
3. Verify: Galaxy loaded ONCE, consolidation triggered appropriately

---

## Sovereignty Requirements (NON-NEGOTIABLE)

```
╔════════════════════════════════════════════════════════════════╗
║  ALL operations in hot path MUST be:                          ║
║  ✓ PTX kernels (ternary arithmetic)                           ║
║  ✓ RPN programs (stack-based execution)                       ║
║  ✓ GPU-resident tensors (no CPU round-trips)                  ║
║                                                                ║
║  FORBIDDEN in hot path:                                        ║
║  ✗ numpy operations                                            ║
║  ✗ CPU-side loops over data                                    ║
║  ✗ Python dict lookups for embeddings                          ║
║  ✗ Reloading Galaxy from disk every task                       ║
╚════════════════════════════════════════════════════════════════╝
```

---

## Success Criteria

1. **Galaxy loaded exactly ONCE** per training run (not 17,496 times like current)
2. **Zero CPU fallbacks** (PTX rate = 100%)
3. **No hardcoded thresholds** in sovereign_pipeline.py
4. **Drawing rules working** (ROT90, FLIP, SCALE produce correct outputs)
5. **Semantic bridge active** (rules matched to task patterns)
6. **Consolidation triggered** when memory > 85% or end of batch
7. **Measurable uplift** over 46.19% baseline (target: 55%+)

---

## Environment (Debian)

```bash
# Activate
export PATH="/home/daniel/miniforge/bin:/home/daniel/miniforge/condabin:$PATH"
conda activate k3d-cranium

# Run with sovereignty
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0

# Test individual components
python -c "from knowledge3d.cranium.drawing_galaxy import DrawingGalaxy; dg = DrawingGalaxy(); print(dg.execute_rule('ROT90_CW', test_grid))"

# Full training
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
```

---

## Summary

This refactor transforms K3D from a **stateless task solver** into an **embodied reasoning agent** that:

1. **Persists** Galaxy in GPU memory (load once)
2. **Accumulates** discoveries in working memory
3. **Consolidates** under memory pressure or at batch end
4. **Learns** its own thresholds (not hardcoded)
5. **Uses** Drawing rules for visual transformations
6. **Bridges** semantic patterns to applicable rules

The goal is not just accuracy uplift, but **architectural correctness** — this is how embodied AI should work.

---

**Codex, this is a significant refactor. Take it phase by phase. Sovereignty is non-negotiable. You have the conn.**
