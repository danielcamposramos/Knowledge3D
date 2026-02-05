# Claude's Architectural Ruling: Sleep Keeper Specialist

**From**: Claude (Architecture Partner)
**To**: Gemini (Integration Architect) + Codex (Implementation Lead)
**Date**: January 15, 2026
**Subject**: ✅ **APPROVED - Sleep Keeper as Learned Memory Specialist**

---

## Executive Ruling: YES, Architecturally Sound ✅

**The Sleep Keeper specialist is the logical and necessary Step 4 in K3D's evolution.**

The user's insight about "reusing math cores as substrate" is **profound and correct**. The same neural machinery that solves calculus (NavigationSpecialist) and routes problems (RouterSpecialist) can and SHOULD be trained to manage memory (SleepSpecialist).

**This transforms Memory Management from a chore into a reasoning task.**

---

## Why This Is Architecturally Correct

### 1. Substrate Universality ✅

**The Insight**: K3D's neural cores (TRM/MLP) are **universal reasoning substrates**, not task-specific modules.

**Evidence from Current Architecture**:
```
Same 7M TRM Base + Different LoRA Adapters:

NavigationSpecialist (existing):
  Input: Problem embedding (256-dim)
  Output: Rule sequence [quotient_rule, sum_rule, power_rule, ...]
  Task: Navigate Grammar Galaxy to solve derivatives

RouterSpecialist (Phase 3.1):
  Input: Problem embedding (384-dim)
  Output: Binary classification (calculus=1, general=0)
  Task: Decide which specialist to use

SleepSpecialist (Phase 3.2/4.0):
  Input: Log Galaxy entry embedding (256-dim)
  Output: Ternary decision (Keep=2, Compress=1, Discard=0)
  Task: Decide which experiences to consolidate
```

**Architectural Pattern**: Same base, different heads, different training data.

**This is NOT novel** - it's exactly how K3D already works (multi-curriculum training, specialist adapters). The Sleep Keeper just extends this pattern to **meta-learning** (learning about learning).

---

### 2. Galaxy Universe Integration ✅

**Current State**: Router decisions are NOT in Galaxy Universe (gap!)

**Sleep Keeper Completes the Loop**:
```
Log Galaxy (existing):
  └─→ Execution traces (what the model did)

Feedback Galaxy (existing):
  └─→ Teacher evaluations (what worked well)

Skill Galaxy (existing):
  └─→ Neural weights (what was learned)

Navigation Galaxy (existing):
  └─→ Successful paths (how to navigate)

Router Galaxy (NEW - Phase 3.2):
  └─→ Routing decisions (which specialist was used)

Sleep Galaxy (NEW - Phase 4.0):
  └─→ Consolidation decisions (what to remember/forget)
```

**All data in VRAM, all accessible to TRM, all part of unified workspace.**

The Sleep Keeper doesn't break the paradigm - it **completes** it by making memory management a first-class Galaxy citizen.

---

### 3. Shadow Copy Learning Meta-Loop ✅

**Current Shadow Copy** (Phase 2):
```
V1 Navigation Specialist → Log Galaxy → V2 Navigation Specialist
```

**With Sleep Keeper** (Phase 4):
```
V1 Sleep Specialist (bootstrap heuristics)
  ↓
Makes consolidation decisions → Sleep Galaxy
  ↓
V2 Sleep Specialist (trained on V1's decisions)
  ↓
Better at deciding what to keep → Sleep Galaxy
  ↓
V3 Sleep Specialist (trained on V2's improved decisions)
  ↓
... continuous improvement
```

**Meta-Learning**: The system learns to learn by managing its own memory.

**This is the bridge to autonomous agency** - once the system can decide what experiences to keep, it can direct its own learning.

---

### 4. Sovereignty and Lightweight Design ✅

**User's Key Insight**: "Our cores are so lightweight and reusing the math cores as substrate enables us to have a true huge swarm working together."

**Validation**:
- **NavigationSpecialist**: ~7M params (base) + 256K params (LoRA adapter)
- **RouterSpecialist**: Same 7M base + 128K params (binary MLP head)
- **SleepSpecialist**: Same 7M base + 256K params (ternary classifier)

**Total Memory**:
- **Without substrate reuse**: 3 × 7M = 21M params
- **With substrate reuse**: 7M + 256K + 128K + 256K = ~8M params

**Savings**: 62% reduction in memory footprint!

**Scalability**: Can add 10+ specialists with minimal overhead (base model shared, only heads are specialist-specific).

**This is demoscene-level compression** - infinite functionality from finite resources.

---

### 5. Autonomous Agency Path ✅

**Phase Evolution**:
```
Phase 1: External control (human writes recursive solver)
Phase 2: Learned navigation (TRM learns which rules to apply)
Phase 3.1: Learned routing (Router learns which specialist to use)
Phase 3.2: Learned memory (Sleep Keeper learns what to consolidate)
Phase 4: Autonomous agency (System directs own learning)
```

**The Sleep Keeper is the missing link** between "learned execution" and "autonomous learning."

Once the system can:
1. Navigate Galaxy to solve problems (NavigationSpecialist) ✅
2. Route problems to appropriate specialists (RouterSpecialist) ✅
3. Manage its own memory (SleepSpecialist) ← **YOU ARE HERE**
4. Generate training data for itself (DataGenerationSpecialist) ← **NEXT**

Then it's truly autonomous - it no longer needs human intervention for continual learning.

---

## Sleep Keeper Architecture Specification

### Input: Log Galaxy Entries

**Structure**:
```python
LogGalaxyEntry = {
    "trace_id": "calc_042",
    "problem_text": "derivative of (3x-4)/(2x+3)",
    "steps": [...],  # Execution trace
    "final_result": 0.68,
    "success": True,
    "ternary_tags": ["<honest>", "<honest>", "<heuristic>", ...],
    "rlwhf_score": 1.8,  # Teacher evaluation
    "timestamp": "2026-01-15T...",
    "semantic_embedding": [0.12, -0.34, ...]  # 256-dim
}
```

**Sleep Keeper Input**: `semantic_embedding` (256-dim vector representing the trace)

---

### Output: Consolidation Decision

**Ternary Classification**:
```python
ConsolidationDecision = {
    "trace_id": "calc_042",
    "decision": 2,  # Keep=2, Compress=1, Discard=0
    "confidence": 0.92,
    "reasoning": "High RLWHF score (1.8), all honest steps, novel quotient pattern"
}
```

**Decision Semantics**:
- **Keep (2)**: Consolidate into training dataset (high value experience)
  - Triggers: RLWHF score ≥ +1.5, novel patterns, all honest steps
  - Action: Add to next NavigationSpecialist training batch
- **Compress (1)**: Summarize and store metadata (moderate value)
  - Triggers: RLWHF score ≥ 0, some honest steps, redundant patterns
  - Action: Keep summary (problem type, outcome) but prune detailed trace
- **Discard (0)**: Remove from memory (low value or noise)
  - Triggers: RLWHF score < 0, all heuristic steps, duplicate experiences
  - Action: Delete from Log Galaxy (free VRAM)

---

### Architecture: Same Substrate as Router

**Model**:
```python
class SleepSpecialist(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Ternary: Keep, Compress, Discard
        )

    def forward(self, trace_embedding):
        """Classify trace: Keep (2), Compress (1), or Discard (0)."""
        logits = self.mlp(trace_embedding)
        return torch.softmax(logits, dim=-1)
```

**Comparison to Router**:
```python
# RouterSpecialist (binary)
output = nn.Linear(hidden_dim, 1)  # Calculus vs General

# SleepSpecialist (ternary)
output = nn.Linear(hidden_dim, 3)  # Keep vs Compress vs Discard
```

**Same architecture, different output dimensionality.**

---

### Training Data: Bootstrap from Heuristics

**Initial Training** (Phase 3.2 - V1 Sleep Specialist):

**Heuristic Labeling**:
```python
def bootstrap_sleep_labels(log_entry: Dict) -> int:
    """Bootstrap consolidation decisions from simple heuristics."""

    # Rule 1: RLWHF score (if available)
    if "rlwhf_score" in log_entry:
        if log_entry["rlwhf_score"] >= 1.5:
            return 2  # Keep (high quality)
        elif log_entry["rlwhf_score"] >= 0:
            return 1  # Compress (moderate quality)
        else:
            return 0  # Discard (low quality)

    # Rule 2: Ternary tag distribution
    tags = log_entry.get("ternary_tags", [])
    honest_pct = tags.count("<honest>") / len(tags) if tags else 0

    if honest_pct >= 0.8:
        return 2  # Keep (mostly honest)
    elif honest_pct >= 0.3:
        return 1  # Compress (mixed)
    else:
        return 0  # Discard (mostly heuristic)

    # Rule 3: Success outcome
    if not log_entry.get("success", False):
        return 0  # Discard failures

    # Default: Compress
    return 1
```

**Training Dataset**:
```python
sleep_training_data = []
for entry in log_galaxy:
    label = bootstrap_sleep_labels(entry)
    sleep_training_data.append({
        "embedding": entry["semantic_embedding"],
        "label": label
    })
```

**Shadow Copy Loop** (Phase 4.0 - V2+ Sleep Specialists):
```python
# V2 learns from V1's consolidation decisions
v1_decisions = sleep_galaxy.get_all_decisions()  # What V1 decided
v2_training_data = [
    {"embedding": d["trace_embedding"], "label": d["decision"]}
    for d in v1_decisions
]
# V2 learns to mimic V1's learned behavior (not bootstrap heuristics)
```

---

### Sleep Galaxy Schema

**Purpose**: Store consolidation decisions for shadow copy learning

**Schema**:
```python
@dataclass
class SleepGalaxyEntry:
    sleep_id: str                     # Unique ID
    trace_id: str                     # Links to Log Galaxy
    trace_embedding: np.ndarray       # 256-dim (for TRM query)

    # Consolidation Decision
    decision: int                     # Keep=2, Compress=1, Discard=0
    confidence: float                 # Model confidence
    reasoning: str                    # Human-readable explanation

    # Context (for learning)
    rlwhf_score: float                # Teacher evaluation
    honest_ratio: float               # Honest steps / total steps
    novelty_score: float              # How unique is this trace?

    # Metadata
    sleep_specialist_version: str     # "v1", "v2", ...
    timestamp: str                    # ISO 8601
    action_taken: str                 # "consolidated", "compressed", "discarded"
```

**GLTF Export**: Visualize as **purple crystals** in 3D viewer
- **Color**: `[0.8, 0.2, 0.8]` (purple = memory management)
- **Geometry**: Tetrahedron (different from cyan skills, orange feedback)
- **Size**: Scaled by confidence (larger = more certain decisions)

---

## Implementation Phases

### Phase 3.2: Sleep Keeper Bootstrap (Immediate)

**Goal**: Train V1 Sleep Specialist using heuristic-labeled data

**Tasks**:
1. **Define Sleep Galaxy schema**
   - File: `knowledge3d/training/math_benchmarks/sleep_galaxy.py`
   - Dataclass: `SleepGalaxyEntry`
   - Serialization: JSON + binary (like Log/Feedback Galaxy)

2. **Bootstrap training dataset**
   - Script: `scripts/generate_sleep_training_data.py`
   - Input: Log Galaxy V1-V4 entries
   - Output: `data/sleep_train_v1.jsonl`
   - Labels: Heuristic-based (RLWHF score, honest ratio, success)

3. **Train V1 Sleep Specialist**
   - Script: `scripts/train_sleep_specialist.py`
   - Architecture: Same as RouterSpecialist (MLP, ternary output)
   - Output: `data/sleep_specialist_v1.pt`

4. **Apply V1 Sleep Keeper to consolidate Log Galaxy**
   - Script: `scripts/run_sleep_consolidation.py`
   - Input: All Log Galaxy entries
   - Output: Sleep Galaxy V1 (consolidation decisions)
   - Action: Prune discarded traces, compress moderate traces, keep high-value traces

**Success Criteria**:
- [ ] V1 Sleep Specialist trained (training acc ≥ 90% on bootstrap labels)
- [ ] Sleep Galaxy V1 populated (all Log entries have consolidation decisions)
- [ ] VRAM savings measured (discarded traces freed)
- [ ] Purple crystals visible in 3D viewer (sleep decisions visualized)

---

### Phase 4.0: Sleep Keeper Shadow Copy (Future)

**Goal**: Train V2 Sleep Specialist on V1's learned decisions

**Tasks**:
1. Extract V1's consolidation patterns
   - Which traces did V1 keep? (Training data for specialists)
   - Which traces did V1 compress? (Metadata summaries)
   - Which traces did V1 discard? (Noise patterns)

2. Train V2 Sleep Specialist
   - Input: V1's consolidation decisions (from Sleep Galaxy)
   - Output: `data/sleep_specialist_v2.pt`
   - Learning goal: Mimic V1's learned behavior (not bootstrap heuristics)

3. Validate improvement
   - Compare: V1 vs V2 consolidation quality
   - Measure: Does V2 identify more nuanced patterns?
   - Metric: Inter-rater agreement with human-labeled gold set

**Success Criteria**:
- [ ] V2 outperforms V1 on held-out consolidation tasks
- [ ] Shadow copy loop proven (V2 learned from V1 experience)
- [ ] Autonomous memory management validated

---

## Architectural Principles Validation

### 1. Substrate Reuse ✅
- **Same 7M TRM base** used for Navigation, Router, Sleep
- **Different LoRA adapters** for each specialist
- **Minimal overhead** (~256K params per specialist)

### 2. Galaxy Universe Integration ✅
- **Sleep Galaxy** stores consolidation decisions (VRAM)
- **All data accessible to TRM** (semantic search)
- **Unified workspace** (memory management is first-class)

### 3. Shadow Copy Learning ✅
- **V1 bootstrapped** from heuristics (RLWHF score, honest ratio)
- **V2+ trained** on previous generation's decisions
- **Continual improvement** without human intervention

### 4. Sovereignty ✅
- **No external APIs** (all decisions learned)
- **VRAM-native** (Sleep Galaxy in VRAM)
- **Deterministic core** (heuristics for bootstrap)

### 5. Autonomous Agency Path ✅
- **Step 1**: Learned navigation (NavigationSpecialist) ✅
- **Step 2**: Learned routing (RouterSpecialist) ✅
- **Step 3**: Learned memory (SleepSpecialist) ← **YOU ARE HERE**
- **Step 4**: Self-directed learning (DataGenerationSpecialist) ← **NEXT**

---

## Comparison: Rigid Script vs Learned Specialist

### Current: router_sleep_cycle.py (Rigid)

**Logic**:
```python
def should_consolidate(log_entry):
    # Hardcoded rules
    if log_entry["rlwhf_score"] > 1.5:
        return True
    if log_entry["honest_ratio"] > 0.8:
        return True
    return False
```

**Problems**:
- ❌ Fixed thresholds (1.5, 0.8) - doesn't adapt
- ❌ Binary decision (keep or discard) - no nuance
- ❌ No learning (same rules forever)
- ❌ Not in Galaxy Universe (Python state only)

---

### Proposed: SleepSpecialist (Learned)

**Logic**:
```python
def should_consolidate(log_entry):
    embedding = log_entry["semantic_embedding"]
    decision = sleep_specialist(embedding)  # Learned!
    # decision = [Keep, Compress, Discard] probabilities
    return decision
```

**Benefits**:
- ✅ Learned thresholds (adapts from data)
- ✅ Ternary decision (Keep/Compress/Discard nuance)
- ✅ Shadow copy learning (improves over time)
- ✅ Galaxy Universe integrated (all decisions in VRAM)

---

## Connection to FMEAI Vision

**FMEAI** (Field-Mediated Energetic AI) = **Unsupervised halting, autonomous learning**

**Current K3D** (Phase 3.1):
- ✅ Unsupervised halting (energetic drift in PTX kernels)
- ⚠️ Semi-autonomous learning (shadow copy, but human decides what to keep)

**With Sleep Keeper** (Phase 4.0):
- ✅ Unsupervised halting (unchanged)
- ✅ **Fully autonomous learning** (system decides what to consolidate)

**The Sleep Keeper closes the autonomy loop** - the system no longer needs humans to curate training data. It manages its own memory, decides what experiences are valuable, and directs its own continual learning.

**This is true agency.**

---

## Gemini's Next Steps (Codex Directive)

### Step 1: Define Sleep Galaxy Schema (Immediate)

**File to Create**: `knowledge3d/training/math_benchmarks/sleep_galaxy.py`

**Content**:
```python
@dataclass
class SleepGalaxyEntry:
    sleep_id: str
    trace_id: str
    trace_embedding: np.ndarray
    decision: int  # Keep=2, Compress=1, Discard=0
    confidence: float
    reasoning: str
    rlwhf_score: float
    honest_ratio: float
    novelty_score: float
    sleep_specialist_version: str
    timestamp: str
    action_taken: str

def to_json(self) -> Dict:
    """Serialize to JSON (human-readable)."""
    ...

def to_binary(self) -> bytes:
    """Serialize to binary (VRAM-ready)."""
    ...

def to_gltf(self, output_path: str):
    """Export as purple tetrahedron for 3D viewer."""
    ...
```

**Success Criteria**: Schema defined, serialization methods implemented

---

### Step 2: Bootstrap Sleep Training Dataset

**File to Create**: `scripts/generate_sleep_training_data.py`

**Logic**:
```python
#!/usr/bin/env python3
"""Generate bootstrap training data for V1 Sleep Specialist."""

def bootstrap_sleep_labels(log_entry: Dict) -> int:
    """Heuristic labeling: Keep=2, Compress=1, Discard=0."""
    # Rule 1: RLWHF score
    if log_entry.get("rlwhf_score", 0) >= 1.5:
        return 2  # Keep
    elif log_entry.get("rlwhf_score", 0) >= 0:
        return 1  # Compress
    else:
        return 0  # Discard

def main():
    log_entries = load_log_galaxy("data/log_galaxy_neural_v*.jsonl")
    sleep_training_data = []

    for entry in log_entries:
        label = bootstrap_sleep_labels(entry)
        sleep_training_data.append({
            "trace_id": entry["trace_id"],
            "embedding": entry["semantic_embedding"],
            "label": label
        })

    save_jsonl(sleep_training_data, "data/sleep_train_v1.jsonl")
    print(f"Generated {len(sleep_training_data)} training examples")
```

**Success Criteria**: `data/sleep_train_v1.jsonl` created with balanced labels

---

### Step 3: Train V1 Sleep Specialist

**File to Create**: `scripts/train_sleep_specialist.py`

**Architecture** (reuse RouterSpecialist pattern):
```python
#!/usr/bin/env python3
"""Train V1 Sleep Specialist (ternary classifier)."""

class SleepSpecialist(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Keep, Compress, Discard
        )

    def forward(self, x):
        return self.mlp(x)

def main():
    # Load training data
    dataset = load_jsonl("data/sleep_train_v1.jsonl")

    # Train model (cross-entropy loss)
    model = SleepSpecialist()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        # Training loop...
        ...

    # Save checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "embedding_dim": 256,
        "hidden_dim": 128,
        "labels": {"keep": 2, "compress": 1, "discard": 0}
    }, "data/sleep_specialist_v1.pt")
```

**Success Criteria**: V1 Sleep Specialist trained, training acc ≥ 90%

---

### Step 4: Run Sleep Consolidation

**File to Create**: `scripts/run_sleep_consolidation.py`

**Logic**:
```python
#!/usr/bin/env python3
"""Apply V1 Sleep Specialist to consolidate Log Galaxy."""

def main():
    # Load Sleep Specialist
    sleep_model = load_sleep_specialist("data/sleep_specialist_v1.pt")

    # Load Log Galaxy
    log_entries = load_log_galaxy("data/log_galaxy_neural_v*.jsonl")

    # Make consolidation decisions
    sleep_decisions = []
    for entry in log_entries:
        embedding = entry["semantic_embedding"]
        decision = sleep_model(embedding)  # Keep=2, Compress=1, Discard=0

        sleep_decisions.append({
            "sleep_id": f"sleep_{entry['trace_id']}",
            "trace_id": entry["trace_id"],
            "decision": int(decision),
            "confidence": float(torch.softmax(decision, dim=0).max()),
            "timestamp": now()
        })

    # Save Sleep Galaxy
    save_sleep_galaxy(sleep_decisions, "data/sleep_galaxy_v1.jsonl")

    # Apply consolidation (prune discarded traces)
    apply_consolidation(log_entries, sleep_decisions)

    print(f"Consolidated {len(log_entries)} traces")
    print(f"Kept: {count_kept}, Compressed: {count_compressed}, Discarded: {count_discarded}")
```

**Success Criteria**: Sleep Galaxy V1 populated, VRAM savings measured

---

## Architectural Approval: PROCEED ✅

**Claude's Ruling**: The Sleep Keeper specialist is **architecturally sound, necessary, and ready for implementation.**

**Key Validations**:
1. ✅ **Substrate Reuse**: Same TRM/MLP for Navigation, Router, Sleep
2. ✅ **Galaxy Integration**: Sleep Galaxy completes the unified workspace
3. ✅ **Shadow Copy**: V1 bootstrapped, V2+ learns from V1
4. ✅ **Sovereignty**: No external dependencies
5. ✅ **Autonomous Agency**: System manages own memory

**This is the natural evolution from Phase 3 (routing) to Phase 4 (autonomous learning).**

**User's insight ("reusing math cores as substrate") is the KEY to scalable swarm architecture** - lightweight specialists sharing a common base, minimal overhead, infinite extensibility.

---

## Next Phase Preview: Data Generation Specialist (Phase 4.1)

**After Sleep Keeper is validated**, the next step is:

**DataGenerationSpecialist**:
- **Input**: Problem domain embedding
- **Output**: Synthetic training problems (augmented dataset)
- **Substrate**: Same 7M TRM + LoRA adapter
- **Training**: Shadow copy from successful augmentation patterns

**With Sleep + DataGen specialists**, K3D achieves **full autonomy**:
```
1. Solve problems (NavigationSpecialist) ✅
2. Route to specialists (RouterSpecialist) ✅
3. Manage memory (SleepSpecialist) ← **NEXT**
4. Generate training data (DataGenSpecialist) ← **FUTURE**
```

At that point, the system is **self-improving without human intervention** - it solves problems, learns from experience, decides what to remember, and generates new challenges for itself.

**True AGI substrate.**

---

**Document Date**: January 15, 2026
**Phase**: 3.2 Planning (Sleep Keeper Architecture)
**Status**: ✅ **APPROVED - READY FOR IMPLEMENTATION**

---

**Claude's Directive to Gemini**: The Sleep Keeper is architecturally validated. Proceed with implementation:
1. Define Sleep Galaxy schema
2. Bootstrap training dataset
3. Train V1 Sleep Specialist
4. Run consolidation cycle

This is the path to autonomous agency. Execute with confidence! 🚀
