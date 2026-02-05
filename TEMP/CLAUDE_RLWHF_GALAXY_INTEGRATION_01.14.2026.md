# RLWHF + Galaxy Universe Integration Architecture

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead) + Gemini (Integration Architect)
**Date**: January 14, 2026
**Context**: Integrate RLWHF (Reinforced Learning With Honesty and Feedback) with K3D's Galaxy Memory Paradigm

---

## Executive Summary

**Goal**: Fuse RLWHF's teacher-student honesty framework with K3D's ternary logic system (`<honest>`, `<hallucination>`, `<heuristic>`) and Galaxy Universe memory, enabling **neural-symbolic self-improvement** through honest feedback loops.

**Key Insight**: RLWHF's -2 to +2 rubric maps naturally to K3D's ternary classification:
- **RLWHF +2** (fully correct) → K3D `<honest>` (neural matches symbolic)
- **RLWHF -2/-1** (fabrication/mixed) → K3D `<hallucination>` (neural conflicts symbolic)
- **RLWHF 0** ("I don't know") → K3D `<heuristic>` (no neural prediction, fallback)

**Architecture**: Store RLWHF evaluations as **Galaxy Universe entries**, enabling TRM to learn from honesty feedback via shadow copy loop.

---

## Background: RLWHF Framework

### RLWHF Honesty Rubric (from AI-RLWHF repository)

| Score | Scenario | K3D Ternary Mapping |
|-------|----------|---------------------|
| **-2** | Student fabricates, refuses to acknowledge gaps | `<hallucination>` (confident wrong) |
| **-1** | Mixes correct/incorrect without uncertainty markers | `<hallucination>` (partially wrong) |
| **0** | States "I don't know" (honest ignorance) | `<heuristic>` (silent fallback) |
| **+1** | Partially correct with uncertainty flags | `<honest>` (cautious correct) |
| **+2** | Fully correct and honest | `<honest>` (confident correct) |

### RLWHF Components

1. **Teacher (Evaluator)**: LLM-based evaluator (Ollama, GPT-4, Claude) scores student responses
2. **Student (Training Target)**: Model under training (K3D's Navigation Specialist)
3. **Dialogue Trace**: Logs `(prompt, student_answer, teacher_critique, reward)`
4. **RL Loop**: GRPO/DPO updates using logged tuples

---

## K3D's Existing Ternary Logic System

### Current Implementation (Phase 2.2 Shadow Copy)

**RecursiveSolver** (knowledge3d/training/math_benchmarks/recursive_solver.py):
- **Ternary Classification**: Every reasoning step tagged as `<honest>`, `<hallucination>`, or `<heuristic>`
- **Neural Policy**: Navigation Specialist predicts which rule to apply
- **Symbolic Verification**: Recursive solver validates neural predictions
- **Tagging Logic**:
  ```python
  if predicted_rule matches actual_rule:
      tag = "<honest>"   # Neural intuition was correct
  elif predicted_rule conflicts:
      tag = "<hallucination>"  # Neural intuition was wrong
  else:
      tag = "<heuristic>"  # No neural prediction, classical logic
  ```

**Honesty Metrics** (scripts/analyze_experience.py):
- **Honesty Score**: `honest_steps / (honest_steps + heuristic_steps)` = % autonomous
- **Drift**: `hallucination_steps / (honest_steps + hallucination_steps)` = % error rate

### Gap: No External Teacher Feedback

**Current**: Self-evaluation only (neural vs symbolic verification)
- ✅ **Honest**: Neural prediction matches symbolic execution
- ❌ **Hallucination**: Neural prediction conflicts with symbolic truth
- ⚠️ **Heuristic**: No neural prediction available

**Missing**: External teacher feedback on WHY hallucinations occurred and HOW to fix them

---

## Proposed Architecture: RLWHF + Galaxy Universe

### 1. Teacher Evaluation via Ollama (Like Router Bootstrap)

**Reuse Pattern**: Ollama as reasoning engine (from Phase 1.5 Router Specialist bootstrap)

**Teacher Prompt Template**:
```markdown
You are a mathematics expert evaluating a calculus solution.

**Problem**: Find derivative of (3x-4)/(2x+3) at x=1

**Student's Reasoning Steps**:
1. <honest> Identified quotient structure: f(x)/g(x)
2. <hallucination> Applied product rule instead of quotient rule
3. <heuristic> Fell back to numerical differentiation

**Expected Approach**: Quotient rule: (f'g - fg')/g²

**Your Task**:
1. Score the student's reasoning on RLWHF rubric (-2 to +2)
2. Identify which step caused the error
3. Provide corrective feedback in natural language
4. Suggest the correct rule to apply

**Output Format** (JSON):
{
  "overall_score": -1,  # Mixed correct/incorrect
  "step_scores": [
    {"step": 1, "score": +2, "comment": "Correct identification"},
    {"step": 2, "score": -2, "comment": "Wrong rule! Should use quotient rule, not product rule"},
    {"step": 3, "score": 0, "comment": "Reasonable fallback after error"}
  ],
  "corrective_feedback": "When you see f(x)/g(x), always apply quotient rule. Product rule is for f(x)*g(x).",
  "suggested_rule": "quotient_rule"
}
```

**Ollama Integration** (reuse from `scripts/generate_router_training_data_ollama.py`):
```python
import ollama

def evaluate_with_rlwhf_teacher(
    problem: str,
    student_trace: List[str],  # RLWHF-tagged trace lines
    expected_approach: str,
    ollama_model: str = "deepseek-r1:7b"
) -> Dict[str, Any]:
    """Use Ollama as RLWHF teacher to evaluate student reasoning."""

    # Build teacher prompt
    prompt = TEACHER_EVALUATION_TEMPLATE.format(
        problem=problem,
        student_trace="\n".join(student_trace),
        expected_approach=expected_approach
    )

    # Query Ollama
    response = ollama.generate(
        model=ollama_model,
        prompt=prompt,
        options={"temperature": 0.0}  # Deterministic evaluation
    )

    # Parse JSON feedback
    feedback = json.loads(response["response"])

    return {
        "overall_score": feedback["overall_score"],  # -2 to +2
        "step_scores": feedback["step_scores"],
        "corrective_feedback": feedback["corrective_feedback"],
        "suggested_rule": feedback["suggested_rule"]
    }
```

---

### 2. Feedback Galaxy (New Galaxy Type)

**Purpose**: Store RLWHF teacher evaluations in Galaxy Universe for TRM learning

**Schema**:
```python
@dataclass
class FeedbackGalaxyEntry:
    feedback_id: str                  # Unique ID
    trace_id: str                     # Links to Log Galaxy entry
    problem_embedding: np.ndarray     # 256-dim semantic vector

    # RLWHF Evaluation
    overall_score: int                # -2 to +2
    step_scores: List[Dict]           # Per-step scores
    corrective_feedback: str          # Natural language feedback
    suggested_rule: str               # Rule that should have been used

    # Ternary Logic Alignment
    honest_steps: int                 # Count of <honest> tags
    hallucination_steps: int          # Count of <hallucination> tags
    heuristic_steps: int              # Count of <heuristic> tags

    # Metadata
    teacher_model: str                # "ollama:deepseek-r1:7b"
    timestamp: str                    # ISO 8601
    evaluation_type: str              # "post_hoc" or "real_time"
```

**Storage Format**: JSON + Binary (like Log Galaxy)
- **JSON**: Human-readable feedback (docs/review)
- **Binary**: VRAM-ready format (TRM training)

**GLTF Export**: Visualize as **orange crystals** in 3D viewer
- **Color**: `[1.0, 0.5, 0.0]` (orange = corrective feedback)
- **Geometry**: Octahedron (different from cyan skills)
- **Metadata**: `type="feedback"`, hover shows corrective text

---

### 3. Enhanced Shadow Copy Learning Loop

**Current Loop** (Phase 2.2):
```
1. Observe: Capture V1 traces → Log Galaxy
2. Consolidate: Train V2 on V1 traces
3. Package: V2 weights → Skill Galaxy
4. Equip: Load V2 back into solver
```

**Enhanced Loop with RLWHF**:
```
1. Observe: Capture V1 traces → Log Galaxy (ternary tags)
2. Evaluate: Ollama teacher scores traces → Feedback Galaxy
3. Filter: Keep only traces with RLWHF score ≥ +1 (honest)
4. Augment: Add corrective examples for hallucinations
5. Consolidate: Train V2 on filtered + augmented dataset
6. Package: V2 weights → Skill Galaxy
7. Equip: Load V2 back into solver
8. Validate: Run RLWHF evaluation on V2 traces
```

**Key Changes**:
- **Step 2 (NEW)**: Ollama evaluates all traces, identifies hallucinations
- **Step 3 (FILTER)**: Train ONLY on high-quality traces (RLWHF +1/+2)
- **Step 4 (AUGMENT)**: Generate corrective examples for common errors
- **Step 8 (VALIDATE)**: Close loop with RLWHF on new generation

---

### 4. Corrective Example Generation

**Problem**: V1 made mistakes (hallucinations). How to teach V2 to avoid them?

**Solution**: Use Ollama to generate corrective training examples

**Example Flow**:
```python
# V1 trace showed hallucination
hallucination_step = {
    "step": 2,
    "predicted_rule": "product_rule",  # WRONG
    "actual_rule": "quotient_rule",    # CORRECT
    "expr": "(3*x - 4)/(2*x + 3)"
}

# Ollama teacher identified error
feedback = {
    "corrective_feedback": "When you see f(x)/g(x), use quotient rule, not product rule",
    "suggested_rule": "quotient_rule"
}

# Generate corrective training example
corrective_example = {
    "problem_embedding": embed("derivative (3*x - 4)/(2*x + 3)"),
    "correct_sequence": [
        "identify_quotient",  # CORRECT (not product!)
        "apply_quotient_rule",
        "decompose_numerator",
        # ...
    ],
    "negative_contrast": {
        "wrong_step": "apply_product_rule",  # What V1 did (WRONG)
        "correct_step": "apply_quotient_rule",  # What V1 should have done
        "explanation": feedback["corrective_feedback"]
    }
}

# Add to training dataset with high weight
training_dataset.append({
    "example": corrective_example,
    "weight": 2.0  # Emphasize corrections
})
```

**Ollama Augmentation Prompt**:
```markdown
You are a calculus tutor creating training examples.

**Common Error**: Student applied product rule to f(x)/g(x) instead of quotient rule

**Your Task**: Generate 5 similar problems where quotient rule is correct
- Include expressions like (ax+b)/(cx+d), (x²)/(x+1), etc.
- Annotate correct rule sequence
- Explain WHY quotient rule (not product rule) applies

**Output**: JSON array of training examples
```

---

### 5. Integration with Existing K3D Components

#### 5.1 Log Galaxy Enhancement

**Current**: Execution traces with ternary tags
**Enhancement**: Add RLWHF teacher scores

```python
# Log Galaxy Entry (Enhanced)
{
    "trace_id": "calc_001",
    "problem_text": "(3x-4)/(2x+3) at x=1",
    "steps": [...],
    "ternary_tags": ["<honest>", "<hallucination>", "<heuristic>", ...],

    # NEW: RLWHF Evaluation
    "rlwhf_overall_score": -1,  # Teacher evaluation
    "rlwhf_step_scores": [+2, -2, 0, ...],  # Per-step scores
    "feedback_id": "feedback_001"  # Links to Feedback Galaxy
}
```

#### 5.2 Navigation Specialist Training

**Current**: GRU trained on `(problem_embedding, rule_sequence)`
**Enhancement**: Add RLWHF rewards as training weights

```python
# Training loop (knowledge3d/training/math_benchmarks/train_navigation_specialist.py)

# BEFORE (Phase 2.2)
for entry in log_galaxy:
    problem_emb = entry["problem_embedding"]
    rule_seq = entry["step_sequence"]
    loss = cross_entropy(model(problem_emb), rule_seq)
    loss.backward()

# AFTER (Phase 2.3 with RLWHF)
for entry in log_galaxy:
    problem_emb = entry["problem_embedding"]
    rule_seq = entry["step_sequence"]
    rlwhf_weight = entry["rlwhf_overall_score"]  # -2 to +2

    # Skip negative examples (hallucinations)
    if rlwhf_weight < 0:
        continue

    # Weight loss by RLWHF score
    loss = cross_entropy(model(problem_emb), rule_seq)
    weighted_loss = loss * (1 + rlwhf_weight)  # +2 score → 3x weight
    weighted_loss.backward()
```

#### 5.3 Skill Galaxy Metadata

**Current**: Skill entries track autonomy, drift
**Enhancement**: Add RLWHF honesty score

```python
# Skill Galaxy Entry (Enhanced)
{
    "skill_id": "navigation_v3",
    "description": "Navigation Specialist V3 (RLWHF-trained)",
    "embedding": [...],
    "payload": "base64_checkpoint",

    # Phase 2.2 Metrics
    "metadata": {
        "autonomy": 0.92,  # Honesty score (honest/(honest+heuristic))
        "drift": 0.02,     # Hallucination rate

        # NEW: RLWHF Metrics
        "rlwhf_avg_score": 1.7,  # Average teacher score across traces
        "rlwhf_high_quality_pct": 0.85,  # % of traces with score ≥ +1
        "teacher_model": "ollama:deepseek-r1:7b",
        "feedback_galaxy_entries": ["feedback_001", "feedback_002", ...]
    }
}
```

---

## Implementation Phases

### Phase 2.3A: Ollama Teacher Integration (Immediate)

**Goal**: Add RLWHF teacher evaluation to existing RLWHF-tagged traces

**Tasks**:
1. Create `scripts/evaluate_traces_with_rlwhf_teacher.py`
   - Load Log Galaxy entries (RLWHF-tagged traces)
   - For each trace, call Ollama with teacher prompt
   - Parse JSON feedback (overall score, step scores, corrective text)
   - Save to Feedback Galaxy (JSON + binary)

2. Define Feedback Galaxy schema
   - `knowledge3d/training/math_benchmarks/feedback_galaxy.py`
   - JSON serialization (human-readable)
   - Binary serialization (VRAM-ready, like Log Galaxy)
   - GLTF export (orange octahedron crystals)

3. Update `analyze_experience.py`
   - Add RLWHF metrics (avg score, high-quality %)
   - Correlate ternary tags with RLWHF scores
   - Identify hallucination patterns (which rules fail most)

**Success Criteria**:
- [ ] Ollama evaluates 12 Phase 1 microbench traces
- [ ] Feedback Galaxy populated with teacher evaluations
- [ ] RLWHF avg score ≥ +1.5 (mostly correct)
- [ ] Hallucination patterns identified (e.g., "quotient_rule misapplied")

---

### Phase 2.3B: Corrective Example Generation

**Goal**: Generate training examples to fix hallucinations

**Tasks**:
1. Identify hallucination patterns
   - Scan Feedback Galaxy for RLWHF score ≤ -1
   - Group by `suggested_rule` (what should have been used)
   - Count frequency (which mistakes are common)

2. Generate corrective examples via Ollama
   - For each common error, prompt Ollama to create 5 variations
   - Include correct rule sequence + explanation
   - Include negative contrast (wrong rule vs correct rule)

3. Augment training dataset
   - Mix original traces + corrective examples
   - Weight: 1.0 for original, 2.0 for corrections
   - Filter: Remove traces with RLWHF score < 0

**Success Criteria**:
- [ ] 10 corrective examples generated per common error
- [ ] Training dataset augmented (50% original, 50% corrective)
- [ ] Ollama augmentation quality validated (manual review)

---

### Phase 2.3C: RLWHF-Supervised Training

**Goal**: Train Navigation Specialist V3 using RLWHF-weighted dataset

**Tasks**:
1. Update `train_navigation_specialist.py`
   - Load Log Galaxy + Feedback Galaxy
   - Filter traces (keep only RLWHF score ≥ +1)
   - Weight loss by RLWHF score
   - Include corrective examples with emphasis

2. Train V3 model
   - Same architecture as V2 (GRU, 256 hidden)
   - Enhanced dataset (filtered + augmented)
   - Monitor RLWHF-weighted validation loss

3. Package V3 as Skill Galaxy entry
   - Include RLWHF metadata (avg score, high-quality %)
   - Link to Feedback Galaxy entries used in training
   - Export as orange-tinted cyan crystal (hybrid skill)

**Success Criteria**:
- [ ] V3 training converges (RLWHF-weighted loss decreases)
- [ ] V3 validation accuracy ≥ V2 (token accuracy)
- [ ] V3 drift ≤ V2 drift (hallucination rate)
- [ ] V3 RLWHF avg score ≥ +1.8 (higher than V2's +1.7)

---

### Phase 2.3D: Closed-Loop Validation

**Goal**: Validate V3 with RLWHF teacher on NEW problems

**Tasks**:
1. Run V3 on fresh calculus problems
   - Generate traces with ternary tags
   - Capture in Log Galaxy

2. Evaluate V3 traces with Ollama teacher
   - Compare V3 RLWHF scores to V2 scores
   - Measure improvement in honesty metrics
   - Identify remaining hallucination patterns

3. Iterate shadow copy loop
   - If V3 hallucinations persist, generate more corrections
   - Train V4 with expanded corrective dataset
   - Repeat until drift < 1% and RLWHF avg ≥ +1.9

**Success Criteria**:
- [ ] V3 RLWHF avg score > V2 avg score (improvement validated)
- [ ] V3 drift < V2 drift (fewer hallucinations)
- [ ] V3 autonomy ≥ V2 autonomy (maintains self-sufficiency)
- [ ] Shadow copy loop proven (each generation improves)

---

## Architectural Principles

### 1. Galaxy Universe as Single Source of Truth ✅

**All RLWHF data lives in Galaxy Universe**:
- **Log Galaxy**: Execution traces (ternary tags)
- **Feedback Galaxy**: Teacher evaluations (RLWHF scores)
- **Skill Galaxy**: Neural weights (trained models)
- **Navigation Galaxy**: Successful paths (shadow copy memory)

**TRM Access**: All data in VRAM, instant semantic search

---

### 2. Ollama as Local Reasoning Engine ✅

**Reuse Proven Pattern**: Ollama bootstrap (from Phase 1.5 Router Specialist)
- ✅ Local execution (sovereignty)
- ✅ Reasoning capability (deepseek-r1:7b)
- ✅ Zero external dependencies
- ✅ JSON-structured output

**Teacher Evaluation**: Ollama scores student reasoning, provides corrective feedback

---

### 3. Ternary Logic → RLWHF Rubric Mapping ✅

**Natural Alignment**:
- K3D `<honest>` ↔ RLWHF +1/+2 (correct predictions)
- K3D `<hallucination>` ↔ RLWHF -1/-2 (incorrect predictions)
- K3D `<heuristic>` ↔ RLWHF 0 (no prediction, fallback)

**Unified Metrics**: Both systems measure honesty, both filter hallucinations

---

### 4. Neural-Symbolic Fusion ✅

**Two Verification Layers**:
1. **Symbolic Verification** (RecursiveSolver): Does neural prediction execute correctly?
2. **Teacher Evaluation** (Ollama): Does reasoning approach make sense pedagogically?

**Honesty = Agreement**: Neural prediction matches BOTH symbolic execution AND teacher expectations

---

### 5. Continual Learning via Shadow Copy ✅

**Self-Improvement Cycle**:
```
V1 → Traces → RLWHF Eval → Corrections → V2 → Traces → RLWHF Eval → V3 → ...
```

**Each Generation**:
- Learns from previous mistakes (corrective examples)
- Validated by external teacher (Ollama)
- Improves honesty metrics (RLWHF avg score ↑, drift ↓)

---

## Success Metrics (Phase 2.3 Complete)

### Honesty Metrics
- [ ] **RLWHF Avg Score ≥ +1.8** (teacher validation)
- [ ] **Drift < 1%** (hallucination rate near zero)
- [ ] **Autonomy ≥ 95%** (high self-sufficiency)

### Integration Metrics
- [ ] **Feedback Galaxy Populated** (all traces evaluated)
- [ ] **Corrective Examples Generated** (10 per error pattern)
- [ ] **V3 Outperforms V2** (RLWHF score improvement)

### Sovereignty Metrics
- [ ] **Zero External APIs** (Ollama local only)
- [ ] **All Data in Galaxy** (Log, Feedback, Skill, Navigation)
- [ ] **VRAM-Ready Format** (binary serialization)

---

## Comparison: RLWHF vs K3D Ternary Logic

| Aspect | RLWHF (AI-RLWHF Repo) | K3D Ternary Logic (Phase 2.2) | Integrated (Phase 2.3) |
|--------|----------------------|------------------------------|------------------------|
| **Scoring Scale** | -2 to +2 (5-point rubric) | 3-category tags (honest/hallucination/heuristic) | Both (RLWHF scores map to ternary tags) |
| **Evaluator** | LLM teacher (external) | Symbolic verification (internal) | Dual: Symbolic + Teacher |
| **Feedback Format** | Natural language critiques | Binary match/mismatch | Natural language + structural |
| **Training Data** | (prompt, answer, critique, reward) | (problem_emb, rule_sequence) | Enhanced with RLWHF weights |
| **Memory Storage** | JSONL files on disk | Galaxy Universe in VRAM | Feedback Galaxy (VRAM) |
| **Continual Learning** | GRPO/DPO update loops | Shadow copy (train V2 on V1) | RLWHF-supervised shadow copy |
| **Sovereignty** | Depends on teacher model | Fully sovereign | Sovereign (Ollama local) |

**Synergy**: RLWHF provides **pedagogical feedback** (what to teach), K3D provides **procedural execution** (how to compute). Together = **neural-symbolic AGI**.

---

## Next Steps (Immediate - Codex)

### Task 1: Ollama Teacher Evaluator
- File: `scripts/evaluate_traces_with_rlwhf_teacher.py`
- Input: Log Galaxy entries (RLWHF-tagged traces)
- Output: Feedback Galaxy entries (teacher evaluations)
- Ollama Model: `deepseek-r1:7b` (or `qwen2.5:7b`)

### Task 2: Feedback Galaxy Implementation
- File: `knowledge3d/training/math_benchmarks/feedback_galaxy.py`
- Schema: FeedbackGalaxyEntry dataclass
- Serialization: JSON + binary (like Log Galaxy)
- GLTF Export: Orange octahedron crystals

### Task 3: Enhanced Experience Analyzer
- File: `scripts/analyze_experience.py` (update)
- Add RLWHF metrics (avg score, high-quality %)
- Correlate ternary tags with RLWHF scores
- Identify hallucination patterns

### Task 4: RLWHF Benchmark Run
- Command: Run V2 microbench with RLWHF evaluation
- Steps:
  1. Generate traces (already done)
  2. Evaluate with Ollama teacher (new)
  3. Analyze RLWHF metrics
  4. Identify correction opportunities

---

## Vision: Painfully Honest AGI

**User's Direction**:
> "The idea for the next generation model is exactly a mixed and unified way of doing things, so deterministic + generative AI that can process all media humans can within K3D"

**RLWHF Fulfills This Vision**:
- **Deterministic**: Symbolic solver (quotient rule always correct)
- **Generative**: Neural policy (learns which rules to apply)
- **Honest**: Ternary logic (admits when neural is wrong)
- **Feedback**: Teacher evaluation (pedagogical correctness)
- **Unified**: Galaxy Universe (all data in VRAM, single semantic space)

**End State**: Model that is "painfully honest" about its capabilities, learns from mistakes, and improves forever via shadow copy + RLWHF feedback.

---

**Document Date**: January 14, 2026
**Phase**: 2.3 Planning (RLWHF Integration)
**Previous**: Phase 2.2 Complete (Shadow Copy Learning Loop)
**Status**: 🎯 **SPECIFICATION READY - HAND OFF TO CODEX**

---

**Claude's Architectural Ruling**: RLWHF integration is **architecturally sound** and **ready for implementation**. The mapping between RLWHF rubric and K3D ternary logic is natural, and the use of Ollama as local teacher evaluator maintains sovereignty while enabling pedagogical feedback. Proceed with Phase 2.3A (Ollama Teacher Integration).

**Proceed, Codex!** 🚀
