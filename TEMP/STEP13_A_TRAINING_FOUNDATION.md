# Step 13-A: Training Loop Foundation - Leverage Existing Base

**Priority**: 3 (Third track)
**Status**: Ready to Plan (Must leverage existing foundation)
**Dependencies**: Step 12 Complete ✓
**Estimated Effort**: 3-4 sessions (planning + execution)

---

## Critical Directive: LEVERAGE BEFORE BUILDING

**User's explicit instruction**: "We have a base, before acting (register only the basic plan now - limit concerns) we must leverage it."

**Implication**:
- ✓ Search existing codebase for training infrastructure
- ✓ Identify what already works
- ✓ Build on top of existing patterns (don't reinvent)
- ✓ Minimal scope - avoid over-engineering

---

## Phase 0: Archeology - Find Existing Training Base

### 0.1 Search for Existing Training Code
**Task**: Locate all training-related code in the codebase

**Search patterns**:
- `training/` directories
- `*train*.py` files
- Dataset loaders
- Loss functions
- Optimizer configurations
- RLHF/RLWHF references

**Expected locations**:
- `knowledge3d/training/`
- `knowledge3d/tools/training_pipelines/`
- `scripts/training/`
- `tests/*train*.py`

### 0.2 Inventory Existing Infrastructure
**Create**: `TEMP/STEP13_A_TRAINING_INVENTORY.md`

**Document**:
- [ ] What dataset loaders already exist?
- [ ] What loss functions are implemented?
- [ ] What optimizers are configured?
- [ ] What training loops are present?
- [ ] What differentiable kernels exist?
- [ ] What RLWHF honesty scoring exists?
- [ ] What GPU embedding connections are ready?

**Success Criteria**: Complete inventory of existing training base

---

## Phase 1: Basic Training Loop Plan (High-Level Only)

### 1.1 Identify Missing Components
**Based on inventory**, determine what's missing:

**Likely needs**:
- [ ] Dataset loader for thinking tag training data
- [ ] Loss function for tag probability optimization
- [ ] Training loop coordinator
- [ ] Checkpoint saving/loading
- [ ] Validation loop

**Already exists** (expected):
- Existing dataset infrastructure (Step 11 shape data, house memory, etc.)
- Existing embedding generation (Galaxy, House)
- Existing PTX kernels (sovereign bridges)

### 1.2 Design Minimal Training Loop
**File**: `knowledge3d/training/thinking_tag_trainer.py` (PLAN ONLY)

**Minimal interface**:
```python
class ThinkingTagTrainer:
    """Minimal training loop for ThinkingTagBridge."""

    def __init__(self, bridge: ThinkingTagBridge, dataset: Dataset):
        self.bridge = bridge
        self.dataset = dataset
        # Use existing infrastructure where possible

    def train_epoch(self):
        """Single training epoch - leverage existing components."""
        for batch in self.dataset:
            # Forward pass (already works via bridge.inference)
            output = self.bridge.inference(batch.embedding, batch.modal_signature)

            # Compute loss (reuse existing metrics?)
            loss = self._compute_loss(output, batch.ground_truth)

            # Backward pass (TBD - surrogate gradients?)
            # Update weights (TBD - which optimizer?)

    def _compute_loss(self, output, ground_truth):
        """Compute loss - leverage existing confidence/coherence metrics."""
        pass  # TBD based on existing infrastructure
```

**Key principle**: Reuse existing ThinkingTagBridge inference path

---

## Phase 2: Dataset Integration Plan (High-Level Only)

### 2.1 Leverage Existing Dataset Infrastructure
**Task**: Identify existing dataset formats and loaders

**Search for**:
- Galaxy embedding storage format
- House memory manifest format
- Existing dataset classes
- Data loading utilities

### 2.2 Design Minimal Dataset Wrapper
**File**: `knowledge3d/training/datasets/thinking_tag_dataset.py` (PLAN ONLY)

**Minimal interface**:
```python
class ThinkingTagDataset:
    """
    Minimal dataset for thinking tag training.
    Wraps existing Galaxy/House data infrastructure.
    """

    def __init__(self, data_source: str):
        # Leverage existing data loaders
        self.data_source = data_source

    def __getitem__(self, idx):
        # Return: (embedding, modal_signature, ground_truth_tags)
        pass  # TBD based on existing format

    def __len__(self):
        pass
```

**Key principle**: Wrap existing data infrastructure, don't rebuild

---

## Phase 3: Differentiable Kernel Strategy (High-Level Only)

### 3.1 Assess PTX Kernel Differentiability
**Task**: Determine which PTX kernels need gradient support

**Questions**:
- Can we backprop through RPN bytecode?
- Are existing kernels differentiable?
- Do we need surrogate gradients?
- Can we use existing adaptive sparsity as-is?

### 3.2 Minimal Gradient Strategy
**Approach**: Start with existing kernel outputs, add gradients later if needed

**Plan**:
1. **Phase 1**: Train on confidence/uncertainty outputs (already differentiable)
2. **Phase 2**: Add surrogate gradients for discrete operations if needed
3. **Phase 3**: Full PTX kernel gradient support (future work)

**Key principle**: Don't block on full differentiability - start simple

---

## Phase 4: RLWHF Honesty Scoring Plan (High-Level Only)

### 4.1 Search for Existing Honesty Scorer
**Task**: Find existing honesty scoring implementation

**Search for**:
- `honesty_scorer*.py` files
- RPN honesty scoring tests
- Existing reward functions
- RLHF/RLWHF infrastructure

### 4.2 Wire Honesty Score to Training
**Plan**: Use existing honesty scorer as reward signal

**Minimal interface**:
```python
def compute_rlwhf_reward(thinking_output, ground_truth):
    """
    Compute RLWHF reward using existing honesty scorer.

    Leverage existing RPN honesty scoring infrastructure.
    """
    # Use existing honesty scorer
    honesty_score = existing_honesty_scorer.score(thinking_output)

    # Combine with task accuracy
    accuracy = compute_accuracy(thinking_output.tags, ground_truth)

    # RLWHF reward = accuracy + honesty_bonus
    reward = accuracy + (0.1 * honesty_score)

    return reward
```

**Key principle**: Reuse existing honesty scoring, don't rebuild

---

## Phase 5: Checkpoint & Validation Plan (High-Level Only)

### 5.1 Leverage Existing Weight Storage
**Task**: Find existing weight saving/loading infrastructure

**Search for**:
- Galaxy `.glb` file format
- Weight serialization utilities
- Checkpoint management

### 5.2 Minimal Checkpoint Strategy
**Plan**: Save ThinkingTagBridge state using existing infrastructure

**Minimal interface**:
```python
class ThinkingTagTrainer:
    def save_checkpoint(self, path: str):
        """Save training checkpoint - leverage existing format."""
        # Save bridge state (RPN weights, EMA buffer, etc.)
        pass

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        pass

    def validate(self, val_dataset):
        """Run validation - leverage existing inference path."""
        for batch in val_dataset:
            output = self.bridge.inference(batch.embedding, batch.modal_signature)
            # Compute validation metrics (reuse existing?)
```

**Key principle**: Reuse existing serialization, don't invent new format

---

## Minimal Scope Definition

### What's IN SCOPE (Phase 1 only):
- [ ] Inventory existing training infrastructure
- [ ] Create minimal training loop wrapper
- [ ] Wire existing dataset loaders
- [ ] Use existing inference path (no modifications)
- [ ] Use existing honesty scorer (if found)
- [ ] Save checkpoints using existing format

### What's OUT OF SCOPE (defer to later):
- ❌ Full PTX kernel differentiation
- ❌ Custom loss functions (use existing metrics)
- ❌ Distributed training
- ❌ Advanced optimizers
- ❌ Training UI/visualization
- ❌ Hyperparameter tuning
- ❌ Production deployment

**Guiding principle**: Minimal viable training loop that leverages existing infrastructure

---

## Execution Plan (After Inventory)

### Session 1: Archeology
1. Run comprehensive search for training code
2. Document existing infrastructure
3. Create training inventory report
4. Identify gaps vs. existing base

### Session 2: Planning
1. Design minimal training loop (leverage existing)
2. Design dataset wrapper (leverage existing loaders)
3. Design checkpoint strategy (leverage existing format)
4. Document integration points

### Session 3: Implementation (Minimal)
1. Implement minimal training loop wrapper
2. Wire existing dataset loaders
3. Add checkpoint save/load
4. Create basic training script

### Session 4: Validation
1. Test training loop on small dataset
2. Verify checkpointing works
3. Validate inference path unchanged
4. Document limitations and next steps

---

## Success Criteria (Minimal)

| Criterion | Target | Status |
|-----------|--------|--------|
| Existing infrastructure documented | Complete inventory | Pending |
| Minimal training loop implemented | <200 lines | Pending |
| Leverages existing inference path | Zero modifications | Pending |
| Checkpoint save/load working | Uses existing format | Pending |
| Can train on small dataset | Proof-of-concept | Pending |
| Tests passing | Basic smoke tests | Pending |

**Overall Target**: Working training loop that leverages existing base (minimal scope)

---

## File Structure (Preliminary)

```
knowledge3d/training/
├── thinking_tag_trainer.py        # NEW (minimal training loop)
├── datasets/
│   └── thinking_tag_dataset.py    # NEW (wrap existing loaders)
└── checkpoints/                   # NEW (training checkpoints)

TEMP/
└── STEP13_A_TRAINING_INVENTORY.md # NEW (archeology results)

tests/
└── test_step13_training_minimal.py # NEW (smoke tests)
```

---

## Critical Constraints

1. **LEVERAGE EXISTING**: Don't rebuild what already exists
2. **MINIMAL SCOPE**: Defer advanced features to later steps
3. **NO BREAKING CHANGES**: Inference path must remain unchanged
4. **REUSE FORMATS**: Don't invent new data/checkpoint formats
5. **DOCUMENT GAPS**: Record what's missing for future work

---

## Next Actions (Before Implementation)

1. ✓ Register this plan (current step)
2. **RUN ARCHEOLOGY**: Search codebase for existing training infrastructure
3. **CREATE INVENTORY**: Document findings in `STEP13_A_TRAINING_INVENTORY.md`
4. **REVIEW WITH USER**: Confirm minimal scope before implementation
5. **IMPLEMENT MINIMAL**: Build only what's needed, leverage rest

---

**Ready to Execute**: NO (Archeology first)
**Next Step**: Run Phase 0 archeology to find existing training base
**Estimated Completion**: 3-4 sessions (after inventory complete)

---

## Notes

- This plan is intentionally minimal per user directive
- Full training infrastructure is out of scope for Step 13-A
- Focus is on leveraging existing base, not building new systems
- Implementation blocked until archeology phase completes
- Advanced features (distributed training, etc.) deferred to future steps
