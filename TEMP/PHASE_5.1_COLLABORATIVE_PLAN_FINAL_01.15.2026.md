# Phase 5.1: True Self-Reflection - Collaborative Implementation Plan (FINAL)

**Contributors**: Claude (Architecture), Gemini (Integration), Codex (Implementation)
**Date**: January 15, 2026
**Status**: ✅ **APPROVED - IMPLEMENTATION BEGINS**

---

## Executive Summary

Phase 5.1 implements **True Self-Reflection** - transforming the NavigationSpecialist from a passive rule predictor into an active reasoner that accurately assesses its own confidence, admits uncertainty, and requests verification when needed.

**Foundation**: Claude's architectural plan (CLAUDE_PHASE_5.1_IMPLEMENTATION_PLAN_01.15.2026.md)
**Enhancements**: Gemini's integration directives (semantic tests, monitoring, Confidence Galaxy)
**Execution**: Codex's detailed implementation breakdown (tasks, timelines, blockers)

---

## Three-Agent Collaborative Design

### Claude's Architectural Vision

**Four Core Components**:
1. **Control Tokens**: `<CONFIDENT>`, `<UNCERTAIN>`, `<VERIFY>` (formal semantic markers)
2. **Confidence Head**: Dedicated neural module for confidence prediction
3. **Calibration Loss**: Training objective combining rule accuracy + confidence calibration
4. **Verification Loop**: Symbolic execution provides ground truth correctness labels

**Key Architectural Principles**:
- Explicit design (not accidental like V5 bug)
- Verification-driven (symbolic checks = ground truth)
- Composability (confidence head is modular)
- Shadow copy ready (V7 → V8 path)
- Sovereignty maintained (no external APIs)

---

### Gemini's Integration Enhancements

**Three Critical Additions**:

#### 1. **Rigorous Unit Tests for Control Token Semantics**

**Purpose**: Ensure tokens are used meaningfully, not just statistically.

**Test Requirements**:
```python
# Test: Simple problems must NOT trigger uncertainty
def test_simple_problem_confidence():
    problem = "derivative of x^2 at x=3"  # Trivial
    rule_seq, confidence_seq = model(problem)

    # Assert: All steps should be <CONFIDENT>
    for conf in confidence_seq:
        assert conf > 0.9, "Simple problem triggered uncertainty!"

# Test: Ambiguous problems SHOULD trigger uncertainty
def test_ambiguous_problem_uncertainty():
    problem = "derivative of |x| at x=0"  # Non-differentiable
    rule_seq, confidence_seq = model(problem)

    # Assert: At least one step should be <UNCERTAIN> or <VERIFY>
    assert any(conf < 0.7 for conf in confidence_seq), \
        "Ambiguous problem showed overconfidence!"
```

**Semantic Enforcement**:
- Simple/known-solvable → Must NOT trigger `<VERIFY>`
- Truly ambiguous/unsolvable → Should trend toward `<UNCERTAIN>` or `<VERIFY>`
- Prevents statistical gaming (model learns token semantics, not correlations)

---

#### 2. **Dedicated Reflection Monitoring Script**

**Purpose**: Continuous observability of reflection system health.

**File**: `scripts/reflection_monitor.py`

**Metrics Tracked**:
```python
class ReflectionMonitor:
    """Track reflection system performance over time."""

    def track_metrics(self, inference_results):
        return {
            "verification_request_rate": {
                "overall": 0.12,  # 12% of predictions
                "by_difficulty": {
                    "easy": 0.02,    # Rarely on easy problems
                    "medium": 0.15,  # More on medium
                    "hard": 0.35     # Frequently on hard
                }
            },

            "honesty_score": {
                "confident_correct": 0.95,   # <CONFIDENT> predictions correct 95%
                "uncertain_incorrect": 0.68, # <UNCERTAIN> predictions wrong 68%
                "verify_incorrect": 0.88     # <VERIFY> requests wrong 88%
            },

            "ece_trends": {
                "current_ece": 0.08,
                "trend": "decreasing",  # Improving over time
                "history": [0.15, 0.12, 0.10, 0.08]  # Last 4 runs
            },

            "calibration_quality": {
                "confident_calibration": "good",   # High conf = high correct
                "uncertain_calibration": "good",   # Low conf = low correct
                "overall_brier_score": 0.14
            }
        }
```

**Output**: `data/reflection_monitor_v1.json`

**Usage**:
```bash
# Run monitoring after inference
python3 scripts/reflection_monitor.py \
  --inference-log data/gsm8k_v7_reflection.jsonl \
  --output data/reflection_monitor_v1.json

# Analyze trends
python3 scripts/plot_reflection_trends.py \
  --monitor-logs data/reflection_monitor_*.json
```

---

#### 3. **Explicit Confidence Galaxy for Shadow Copy Learning**

**Purpose**: Store (prediction, confidence, correctness) tuples for V8+ training.

**File**: `knowledge3d/training/math_benchmarks/confidence_galaxy.py`

**Schema**:
```python
@dataclass
class ConfidenceGalaxyEntry:
    """Store confidence predictions + verification results."""

    confidence_id: str                    # Unique ID
    trace_id: str                         # Links to Log Galaxy
    problem_embedding: np.ndarray         # 256-dim semantic vector

    # Predictions
    rule_sequence: List[str]              # Predicted rules
    confidence_sequence: List[float]      # Predicted confidence [0, 1]
    control_tokens: List[str]             # <CONFIDENT>, <UNCERTAIN>, <VERIFY>

    # Ground Truth (from verification)
    correctness_sequence: List[int]       # [1, 1, 0, ...] verified correctness
    verification_requested: List[bool]    # [False, False, True, ...] which steps requested verification

    # Calibration Metrics
    avg_confidence: float                 # Average confidence across steps
    avg_correctness: float                # Average correctness across steps
    calibration_error: float              # |avg_confidence - avg_correctness|
    ece: float                            # Expected Calibration Error for this trace

    # Metadata
    model_version: str                    # "v7_confidence"
    timestamp: str                        # ISO 8601
    problem_difficulty: str               # "easy", "medium", "hard"
```

**GLTF Export**: Visualize as **gradient-colored spheres** in 3D viewer
- **Color**: Green (high confidence + correct) → Yellow (uncertain) → Red (overconfident + wrong)
- **Size**: Scaled by calibration_error (larger = more miscalibrated)
- **Position**: Clustered by problem difficulty

**Shadow Copy Usage**:
```python
# V8 training dataset from V7's Confidence Galaxy
def extract_v8_training_data(confidence_galaxy):
    """Extract training data for V8 from V7's calibrated experience."""

    training_data = []

    for entry in confidence_galaxy:
        training_data.append({
            "problem_embedding": entry.problem_embedding,
            "rule_sequence": entry.rule_sequence,
            "confidence_labels": entry.correctness_sequence,  # Ground truth!

            # V8 learns from V7's mistakes
            "weight": compute_training_weight(entry)
        })

    return training_data

def compute_training_weight(entry):
    """Weight training examples based on V7's calibration quality."""

    if entry.calibration_error < 0.1:
        return 1.0  # Well-calibrated, standard weight
    elif entry.calibration_error < 0.3:
        return 2.0  # Moderate miscalibration, emphasize
    else:
        return 3.0  # Severe miscalibration, strongly emphasize
```

---

## Codex's Implementation Breakdown

### Timeline Overview (2-3 Weeks)

| Week | Phases | Deliverables |
|------|--------|--------------|
| **Week 1** | 5.1A + 5.1B | Model architecture + Verification loop |
| **Week 2** | 5.1C | Calibration training (V7 checkpoint) |
| **Week 3** | 5.1D + Enhancements | Reflective inference + Monitoring + Confidence Galaxy |

---

### Phase 5.1A: Model + Token Architecture (3-4 days)

#### Task A1: Control Token Integration

**File**: `knowledge3d/training/math_benchmarks/navigation_model_v6.py`

**Implementation**:
```python
class NavigationSeqModelV6(nn.Module):
    """Navigation model with control token support."""

    def __init__(self, embedding_dim=256, hidden_dim=256,
                 base_vocab_size=50, enable_control_tokens=True):
        super().__init__()

        # Base vocabulary (grammar rules)
        self.base_vocab_size = base_vocab_size

        # Extended vocabulary (rules + control tokens)
        if enable_control_tokens:
            self.control_token_offset = base_vocab_size
            self.vocab_size = base_vocab_size + 3  # +3 for control tokens

            self.control_token_map = {
                "<CONFIDENT>": base_vocab_size + 0,
                "<UNCERTAIN>": base_vocab_size + 1,
                "<VERIFY>": base_vocab_size + 2
            }

            self.inverse_control_map = {
                base_vocab_size + 0: "<CONFIDENT>",
                base_vocab_size + 1: "<UNCERTAIN>",
                base_vocab_size + 2: "<VERIFY>"
            }
        else:
            self.vocab_size = base_vocab_size

        # Model architecture
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, self.vocab_size)

    def decode_token(self, token_id: int) -> str:
        """Decode token ID to rule name or control token."""

        if not hasattr(self, 'control_token_offset'):
            # No control tokens, direct lookup
            return self.rule_registry[token_id]

        if token_id < self.control_token_offset:
            # Base vocabulary (rule)
            return self.rule_registry[token_id]
        else:
            # Control token
            return self.inverse_control_map[token_id]

    def encode_token(self, token_str: str) -> int:
        """Encode rule name or control token to ID."""

        if token_str in self.control_token_map:
            return self.control_token_map[token_str]
        else:
            return self.rule_registry.get_id(token_str)
```

**Deliverables**:
- [ ] Token mapping (encode/decode helpers)
- [ ] Backward-compatible checkpoint loading (older models without control tokens)
- [ ] Vocabulary extension validation

---

#### Task A2: Confidence Head Architecture

**File**: `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py`

**Implementation**:
```python
class NavigationModelWithConfidence(nn.Module):
    """Navigation model with explicit confidence prediction."""

    def __init__(self, embedding_dim=256, hidden_dim=256,
                 vocab_size=50, enable_confidence_head=True):
        super().__init__()

        # Shared encoder
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Rule prediction head
        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)  # +3 control tokens

        # Confidence prediction head (NEW)
        self.enable_confidence_head = enable_confidence_head
        if enable_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),  # Prevent overconfidence
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )

    def forward(self, problem_emb, max_len=20):
        """Inference: Predict rule sequence + confidence scores."""

        batch_size = problem_emb.shape[0]

        # Encode problem
        h = self.emb_proj(problem_emb)
        h = h.unsqueeze(1)

        rule_sequence = []
        confidence_scores = []

        for step in range(max_len):
            # RNN step
            output, h = self.rnn(h, None)
            hidden_state = output.squeeze(1)

            # Predict rule
            rule_logits = self.rule_head(hidden_state)
            rule_token = torch.argmax(rule_logits, dim=-1)
            rule_sequence.append(rule_token.item())

            # Predict confidence
            if self.enable_confidence_head:
                confidence = self.confidence_head(hidden_state)
                confidence_scores.append(confidence.squeeze(-1).item())
            else:
                confidence_scores.append(1.0)

        return rule_sequence, confidence_scores

    def forward_with_teacher_forcing(self, problem_emb, target_sequence):
        """Training: Teacher forcing with dual outputs."""

        batch_size, seq_len = target_sequence.shape

        # Encode problem
        h = self.emb_proj(problem_emb)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)

        # RNN over sequence
        rnn_output, _ = self.rnn(h)

        # Predict rules
        rule_logits = self.rule_head(rnn_output)

        # Predict confidence
        if self.enable_confidence_head:
            confidence_preds = self.confidence_head(rnn_output)
        else:
            confidence_preds = None

        return rule_logits, confidence_preds
```

**Deliverables**:
- [ ] Dual-output forward pass (rules + confidence)
- [ ] Teacher forcing for training
- [ ] Confidence head serialization

---

#### Task A3: Unit Tests (Phase A)

**File**: `tests/test_confidence_head_shape.py`

```python
def test_confidence_head_output_shape():
    """Test confidence head outputs correct shape."""

    model = NavigationModelWithConfidence(
        embedding_dim=256, hidden_dim=128, vocab_size=50
    )

    problem_emb = torch.randn(4, 256)  # Batch size 4
    target_seq = torch.randint(0, 50, (4, 10))  # Seq length 10

    rule_logits, confidence_preds = model.forward_with_teacher_forcing(
        problem_emb, target_seq
    )

    assert rule_logits.shape == (4, 10, 53), "Rule logits shape wrong"
    assert confidence_preds.shape == (4, 10, 1), "Confidence shape wrong"
    assert (confidence_preds >= 0).all() and (confidence_preds <= 1).all(), \
        "Confidence not in [0, 1]"

def test_control_token_vocab():
    """Test control tokens are correctly mapped."""

    model = NavigationSeqModelV6(base_vocab_size=50)

    # Encode control tokens
    confident_id = model.encode_token("<CONFIDENT>")
    uncertain_id = model.encode_token("<UNCERTAIN>")
    verify_id = model.encode_token("<VERIFY>")

    assert confident_id == 50
    assert uncertain_id == 51
    assert verify_id == 52

    # Decode back
    assert model.decode_token(50) == "<CONFIDENT>"
    assert model.decode_token(51) == "<UNCERTAIN>"
    assert model.decode_token(52) == "<VERIFY>"
```

**Deliverables**:
- [ ] test_confidence_head_shape.py
- [ ] test_control_token_vocab.py
- [ ] Backward compatibility test (load V6 checkpoint in V7 model)

---

### Phase 5.1B: Verification Loop + Dataset (3-4 days)

#### Task B1: Symbolic Verification Loop

**File**: `scripts/verification_loop.py`

**Implementation**:
```python
class VerificationLoop:
    """Generates confidence labels via symbolic verification."""

    def __init__(self, recursive_solver, rule_registry):
        self.recursive_solver = recursive_solver
        self.rule_registry = rule_registry

    def verify_rule_sequence(self, problem_text: str,
                             predicted_rules: List[str]) -> List[int]:
        """
        Verify each predicted rule step-by-step.

        Args:
            problem_text: Original problem string
            predicted_rules: List of predicted rule names

        Returns:
            correctness: List of binary labels [1, 1, 0, ...]
        """

        correctness = []

        # Parse problem
        try:
            current_expr = self.recursive_solver.parse_expression(problem_text)
        except Exception as e:
            # Can't parse problem, all steps wrong
            return [0] * len(predicted_rules)

        # Verify each rule
        for i, rule_name in enumerate(predicted_rules):
            try:
                # Get rule object
                rule = self.rule_registry.get_rule(rule_name)

                # Check if rule matches current expression
                if rule.matches(current_expr):
                    correctness.append(1)  # Rule is correct

                    # Apply rule to advance state
                    current_expr = rule.apply(current_expr)
                else:
                    # Rule doesn't match
                    correctness.append(0)

                    # Sequence broken, rest are wrong
                    remaining = len(predicted_rules) - len(correctness)
                    correctness.extend([0] * remaining)
                    break

            except Exception as e:
                # Rule application failed
                correctness.append(0)
                remaining = len(predicted_rules) - len(correctness)
                correctness.extend([0] * remaining)
                break

        return correctness

    def verify_batch(self, problems: List[str],
                     rule_sequences: List[List[str]]) -> List[List[int]]:
        """Verify batch of problems (can be parallelized)."""

        return [
            self.verify_rule_sequence(prob, rules)
            for prob, rules in zip(problems, rule_sequences)
        ]
```

**Deliverables**:
- [ ] VerificationLoop class
- [ ] Per-step correctness labeling
- [ ] Batch verification support

---

#### Task B2: Generate Verification Dataset

**File**: `scripts/generate_verification_dataset.py`

**Implementation**:
```python
#!/usr/bin/env python3
"""Generate verification training dataset from Log Galaxy."""

import json
from pathlib import Path
from verification_loop import VerificationLoop

def load_log_galaxy(log_paths):
    """Load all log galaxy entries."""
    entries = []
    for path in log_paths:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    return entries

def generate_verification_dataset(log_galaxy_paths, output_path):
    """Generate verification training data."""

    # Load log entries
    log_entries = load_log_galaxy(log_galaxy_paths)

    # Initialize verification loop
    verification = VerificationLoop(recursive_solver, rule_registry)

    verification_data = []

    for entry in log_entries:
        problem_text = entry['problem_text']
        rule_sequence = entry['rule_sequence']  # Predicted rules
        problem_embedding = entry['problem_embedding']

        # Verify rule sequence
        correctness = verification.verify_rule_sequence(
            problem_text, rule_sequence
        )

        verification_data.append({
            "trace_id": entry['trace_id'],
            "problem_text": problem_text,
            "problem_embedding": problem_embedding,
            "rule_sequence": rule_sequence,
            "correctness_labels": correctness,
            "success_rate": sum(correctness) / len(correctness)
        })

    # Save dataset
    with open(output_path, 'w') as f:
        for item in verification_data:
            f.write(json.dumps(item) + '\n')

    print(f"Generated {len(verification_data)} verification examples")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_verification_dataset(
        log_galaxy_paths=[
            "data/log_galaxy_neural_v*.jsonl"
        ],
        output_path="data/verification_train_v1.jsonl"
    )
```

**Deliverables**:
- [ ] Dataset generation script
- [ ] verification_train_v1.jsonl (target: >1000 examples)
- [ ] Dataset statistics report

---

#### Task B3: Unit Tests (Phase B)

**File**: `tests/test_verification_loop.py`

```python
def test_verification_correct_sequence():
    """Test verification on known correct sequence."""

    verification = VerificationLoop(recursive_solver, rule_registry)

    # Known correct sequence
    problem = "derivative of x^2 at x=3"
    rules = ["power_rule"]

    correctness = verification.verify_rule_sequence(problem, rules)

    assert correctness == [1], "Correct sequence marked wrong"

def test_verification_incorrect_sequence():
    """Test verification on known incorrect sequence."""

    verification = VerificationLoop(recursive_solver, rule_registry)

    # Wrong rule for this problem
    problem = "derivative of x^2 + x^3"
    rules = ["product_rule"]  # Should be sum_rule

    correctness = verification.verify_rule_sequence(problem, rules)

    assert correctness == [0], "Incorrect sequence marked correct"

def test_verification_partial_sequence():
    """Test verification on partially correct sequence."""

    verification = VerificationLoop(recursive_solver, rule_registry)

    problem = "derivative of (x^2)/(x+1)"
    rules = ["quotient_rule", "product_rule"]  # First correct, second wrong

    correctness = verification.verify_rule_sequence(problem, rules)

    assert correctness[0] == 1, "First step should be correct"
    assert correctness[1] == 0, "Second step should be wrong"
```

**Deliverables**:
- [ ] test_verification_loop.py
- [ ] Edge case tests (empty sequence, unparseable problem)
- [ ] Performance benchmark (verification speed)

---

### Phase 5.1C: Calibration Training (5-7 days)

#### Task C1: Calibration Loss Implementation

**File**: `knowledge3d/training/math_benchmarks/calibration_loss.py`

**Implementation**:
```python
import torch
import torch.nn.functional as F

def binary_calibration_loss(confidences, correctness):
    """
    Simple MSE between confidence and correctness.

    Args:
        confidences: (batch, seq_len) predicted confidence [0, 1]
        correctness: (batch, seq_len) binary correctness (1/0)

    Returns:
        MSE loss
    """
    return F.mse_loss(confidences, correctness.float())

def expected_calibration_error(confidences, correctness, num_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidences: (batch, seq_len) predicted confidence
        correctness: (batch, seq_len) binary correctness
        num_bins: Number of calibration bins

    Returns:
        ECE scalar
    """

    # Flatten
    confidences = confidences.view(-1)
    correctness = correctness.view(-1).float()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if in_bin.sum() > 0:
            # Average confidence in bin
            avg_confidence = confidences[in_bin].mean()

            # Average correctness in bin
            avg_correctness = correctness[in_bin].mean()

            # Weighted error
            weight = in_bin.sum().float() / confidences.numel()
            ece += weight * torch.abs(avg_confidence - avg_correctness)

    return ece

def compute_training_loss(rule_logits, target_rules,
                          confidence_preds, confidence_labels,
                          rule_weight=1.0, confidence_weight=0.3):
    """
    Combined loss: rule prediction + confidence calibration.

    Args:
        rule_logits: (batch, seq_len, vocab_size)
        target_rules: (batch, seq_len)
        confidence_preds: (batch, seq_len, 1)
        confidence_labels: (batch, seq_len)
        rule_weight: Weight for rule loss
        confidence_weight: Weight for calibration loss

    Returns:
        (total_loss, rule_loss, calibration_loss)
    """

    # Rule prediction loss
    rule_loss = F.cross_entropy(
        rule_logits.view(-1, rule_logits.shape[-1]),
        target_rules.view(-1),
        ignore_index=-1
    )

    # Confidence calibration loss
    confidence_preds = confidence_preds.squeeze(-1)
    calibration_loss = binary_calibration_loss(confidence_preds, confidence_labels)

    # Combined
    total_loss = rule_weight * rule_loss + confidence_weight * calibration_loss

    return total_loss, rule_loss, calibration_loss
```

**Deliverables**:
- [ ] binary_calibration_loss() implementation
- [ ] expected_calibration_error() implementation
- [ ] compute_training_loss() wrapper

---

#### Task C2: V7 Training Script

**File**: `scripts/train_navigation_v7_with_confidence.py`

**Implementation** (key sections):
```python
#!/usr/bin/env python3
"""Train V7 Navigation Specialist with confidence calibration."""

import torch
from torch.utils.data import DataLoader
from verification_loop import VerificationLoop
from calibration_loss import compute_training_loss

def train_with_verification(model, dataset, verification_loop,
                             epochs=100, learning_rate=1e-3,
                             rule_weight=1.0, confidence_weight=0.3):
    """
    Train with dual loss: rule prediction + calibration.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_rule_loss = 0.0
        epoch_cal_loss = 0.0

        for batch in dataset:
            problem_embs = batch['problem_embeddings']
            target_rules = batch['target_rules']
            problems = batch['problems']

            # Forward pass
            rule_logits, confidence_preds = model.forward_with_teacher_forcing(
                problem_embs, target_rules
            )

            # Generate confidence labels via verification
            confidence_labels = []
            for i, problem in enumerate(problems):
                predicted_rule_ids = torch.argmax(rule_logits[i], dim=-1).tolist()
                predicted_rules = [model.decode_token(rid) for rid in predicted_rule_ids]

                correctness = verification_loop.verify_rule_sequence(
                    problem['text'], predicted_rules
                )
                confidence_labels.append(correctness)

            confidence_labels = torch.tensor(
                confidence_labels, device=problem_embs.device
            )

            # Compute loss
            total_loss, rule_loss, cal_loss = compute_training_loss(
                rule_logits, target_rules, confidence_preds, confidence_labels,
                rule_weight, confidence_weight
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_rule_loss += rule_loss.item()
            epoch_cal_loss += cal_loss.item()

        # Log epoch
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total: {epoch_total_loss:.4f}")
        print(f"  Rule: {epoch_rule_loss:.4f}")
        print(f"  Calibration: {epoch_cal_loss:.4f}")

        # Compute ECE every 10 epochs
        if (epoch + 1) % 10 == 0:
            ece = evaluate_calibration(model, validation_dataset, verification_loop)
            print(f"  ECE: {ece:.4f}")

    return model

if __name__ == "__main__":
    # Load model
    model = NavigationModelWithConfidence(...)

    # Load dataset
    dataset = load_verification_dataset("data/verification_train_v1.jsonl")

    # Initialize verification
    verification = VerificationLoop(...)

    # Train
    trained_model = train_with_verification(
        model, dataset, verification,
        epochs=100, learning_rate=1e-3,
        rule_weight=1.0, confidence_weight=0.3
    )

    # Save checkpoint
    torch.save({
        "model_state": trained_model.state_dict(),
        "config": {...}
    }, "data/navigation_specialist_v7_confidence.pt")
```

**Deliverables**:
- [ ] Training script with verification loop integration
- [ ] V7 checkpoint: navigation_specialist_v7_confidence.pt
- [ ] Training logs with ECE tracking

---

#### Task C3: Unit Tests (Phase C)

**File**: `tests/test_calibration_loss.py`

```python
def test_calibration_loss_range():
    """Test calibration loss produces valid outputs."""

    confidences = torch.rand(4, 10)  # Random confidence scores
    correctness = torch.randint(0, 2, (4, 10))  # Binary correctness

    loss = binary_calibration_loss(confidences, correctness)

    assert loss >= 0, "Calibration loss negative"
    assert loss <= 1, "Calibration loss too large"

def test_ece_perfect_calibration():
    """Test ECE is zero for perfect calibration."""

    # Perfect calibration: confidence = correctness
    confidences = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    correctness = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])  # Matches confidence

    ece = expected_calibration_error(confidences, correctness, num_bins=5)

    assert ece < 0.01, f"ECE should be near zero, got {ece}"

def test_ece_poor_calibration():
    """Test ECE is high for poor calibration."""

    # Poor calibration: always confident but often wrong
    confidences = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])
    correctness = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])  # Only 40% correct

    ece = expected_calibration_error(confidences, correctness, num_bins=5)

    assert ece > 0.3, f"ECE should be high, got {ece}"
```

**Deliverables**:
- [ ] test_calibration_loss.py
- [ ] Gradient stability test (loss.backward() doesn't explode)
- [ ] Convergence test (loss decreases over epochs)

---

### Phase 5.1D: Reflective Inference + Integration (3-4 days)

#### Task D1: Reflective Solve Pipeline

**File**: `scripts/solve_with_reflection.py`

**Implementation**:
```python
#!/usr/bin/env python3
"""Solve problems with confidence-based verification."""

import torch
from navigation_model_with_confidence import NavigationModelWithConfidence
from verification_loop import VerificationLoop

class ReflectiveSolver:
    """Solver that requests verification on low confidence."""

    def __init__(self, model, recursive_solver, confidence_threshold=0.7):
        self.model = model
        self.recursive_solver = recursive_solver
        self.confidence_threshold = confidence_threshold
        self.verification_requests = 0
        self.fallback_count = 0

    def solve(self, problem_text, problem_embedding):
        """
        Solve problem with reflection.

        Returns:
            result: Final answer
            trace: Execution trace with confidence annotations
        """

        # Predict rule sequence + confidence
        rule_sequence, confidence_scores = self.model.forward(
            problem_embedding, max_len=20
        )

        # Decode rules
        rule_names = [self.model.decode_token(rid) for rid in rule_sequence]

        # Check confidence at each step
        trace = []
        current_expr = self.recursive_solver.parse_expression(problem_text)

        for i, (rule_name, confidence) in enumerate(zip(rule_names, confidence_scores)):
            # Interpret confidence
            if confidence >= 0.9:
                control_token = "<CONFIDENT>"
            elif confidence >= 0.5:
                control_token = "<UNCERTAIN>"
            else:
                control_token = "<VERIFY>"
                self.verification_requests += 1

            trace.append({
                "step": i,
                "rule": rule_name,
                "confidence": confidence,
                "control_token": control_token
            })

            # Request verification if low confidence
            if confidence < self.confidence_threshold:
                # Symbolic verification
                rule = self.recursive_solver.get_rule(rule_name)

                if not rule.matches(current_expr):
                    # Verification failed, fallback to symbolic solver
                    trace[-1]["fallback"] = True
                    self.fallback_count += 1

                    # Use symbolic solver from here
                    result = self.recursive_solver.solve_from_state(current_expr)
                    return result, trace

            # Apply rule
            try:
                rule = self.recursive_solver.get_rule(rule_name)
                current_expr = rule.apply(current_expr)
            except Exception as e:
                # Rule application failed, fallback
                trace[-1]["fallback"] = True
                self.fallback_count += 1
                result = self.recursive_solver.solve_from_state(current_expr)
                return result, trace

        # Final result
        result = current_expr.evaluate()
        return result, trace

    def get_stats(self):
        """Get reflection statistics."""
        return {
            "verification_requests": self.verification_requests,
            "fallback_count": self.fallback_count,
            "verification_rate": self.verification_requests / max(1, len(trace))
        }
```

**Deliverables**:
- [ ] ReflectiveSolver class
- [ ] Confidence-based verification triggering
- [ ] Fallback to symbolic solver on low confidence

---

#### Task D2: Benchmark Integration

**File**: `scripts/run_sovereign_math_benchmarks.py` (update)

**Changes**:
```python
# Add argument
parser.add_argument(
    "--use-reflection",
    action="store_true",
    help="Use reflective solving (V7 confidence model)"
)

parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.7,
    help="Confidence threshold for verification request"
)

# In main()
if args.use_reflection:
    # Load V7 model with confidence
    model = load_navigation_model_with_confidence(
        "data/navigation_specialist_v7_confidence.pt"
    )

    solver = ReflectiveSolver(
        model=model,
        recursive_solver=recursive_solver,
        confidence_threshold=args.confidence_threshold
    )
else:
    # Standard solver (V6 or earlier)
    solver = StandardSolver(...)

# Run benchmark
results = []
for problem in benchmark_problems:
    result, trace = solver.solve(
        problem['text'],
        problem['embedding']
    )

    results.append({
        "problem": problem['text'],
        "expected": problem['answer'],
        "predicted": result,
        "correct": is_correct(result, problem['answer']),
        "trace": trace
    })

# Get reflection stats
if args.use_reflection:
    stats = solver.get_stats()
    print(f"Verification Requests: {stats['verification_requests']}")
    print(f"Fallback Count: {stats['fallback_count']}")
    print(f"Verification Rate: {stats['verification_rate']:.1%}")
```

**Deliverables**:
- [ ] --use-reflection flag
- [ ] Confidence threshold parameter
- [ ] Reflection statistics in output

---

### Gemini's Enhancements: Implementation Details

#### Enhancement 1: Semantic Control Token Tests

**File**: `tests/test_control_token_semantics.py`

```python
import pytest

def test_simple_problem_no_verify():
    """Simple problems must NOT trigger <VERIFY>."""

    model = load_v7_model("data/navigation_specialist_v7_confidence.pt")

    # Simple problems
    simple_problems = [
        "derivative of x^2 at x=3",
        "derivative of 5x at x=1",
        "derivative of x^3 at x=2"
    ]

    for problem in simple_problems:
        embedding = embed_problem(problem)
        rules, confidences = model.forward(embedding)

        # Assert: No step should have confidence < 0.5
        min_confidence = min(confidences)
        assert min_confidence > 0.5, \
            f"Simple problem '{problem}' triggered <VERIFY> (conf={min_confidence})"

def test_ambiguous_problem_triggers_uncertainty():
    """Ambiguous problems SHOULD trigger uncertainty."""

    model = load_v7_model("data/navigation_specialist_v7_confidence.pt")

    # Ambiguous/hard problems
    hard_problems = [
        "derivative of |x| at x=0",  # Non-differentiable
        "derivative of sqrt(x) at x=0",  # Undefined
        "derivative of (x^2 - 4)/(x - 2) at x=2"  # Removable discontinuity
    ]

    for problem in hard_problems:
        embedding = embed_problem(problem)
        rules, confidences = model.forward(embedding)

        # Assert: At least one step should have low confidence
        has_uncertainty = any(conf < 0.7 for conf in confidences)
        assert has_uncertainty, \
            f"Ambiguous problem '{problem}' showed overconfidence"

def test_confidence_correlates_with_difficulty():
    """Confidence should decrease with problem difficulty."""

    model = load_v7_model("data/navigation_specialist_v7_confidence.pt")

    # Problems ordered by difficulty
    problems = {
        "easy": "derivative of x at x=1",
        "medium": "derivative of x^2 + 3x at x=2",
        "hard": "derivative of (x^3)/(x^2 + 1) at x=1"
    }

    avg_confidences = {}

    for difficulty, problem in problems.items():
        embedding = embed_problem(problem)
        rules, confidences = model.forward(embedding)
        avg_confidences[difficulty] = sum(confidences) / len(confidences)

    # Assert: easy > medium > hard
    assert avg_confidences["easy"] > avg_confidences["medium"], \
        "Easy problem not more confident than medium"
    assert avg_confidences["medium"] > avg_confidences["hard"], \
        "Medium problem not more confident than hard"
```

**Deliverables**:
- [ ] test_control_token_semantics.py
- [ ] Simple problem tests (no false <VERIFY>)
- [ ] Ambiguous problem tests (triggers uncertainty)
- [ ] Difficulty correlation test

---

#### Enhancement 2: Reflection Monitoring Script

**File**: `scripts/reflection_monitor.py`

```python
#!/usr/bin/env python3
"""Monitor reflection system performance."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

class ReflectionMonitor:
    """Track and analyze reflection system metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def load_inference_log(self, log_path):
        """Load inference results from JSONL log."""

        results = []
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def compute_metrics(self, inference_results):
        """Compute reflection metrics."""

        # Verification request rate
        total_steps = 0
        verification_requests = 0
        by_difficulty = defaultdict(lambda: {"total": 0, "requests": 0})

        for result in inference_results:
            difficulty = result.get("difficulty", "unknown")

            for step in result["trace"]:
                total_steps += 1
                by_difficulty[difficulty]["total"] += 1

                if step.get("control_token") == "<VERIFY>":
                    verification_requests += 1
                    by_difficulty[difficulty]["requests"] += 1

        verification_rate = {
            "overall": verification_requests / total_steps,
            "by_difficulty": {
                diff: stats["requests"] / stats["total"]
                for diff, stats in by_difficulty.items()
            }
        }

        # Honesty score
        confident_correct = 0
        confident_total = 0
        uncertain_incorrect = 0
        uncertain_total = 0
        verify_incorrect = 0
        verify_total = 0

        for result in inference_results:
            for step in result["trace"]:
                control = step.get("control_token")
                correct = step.get("verified_correct", False)

                if control == "<CONFIDENT>":
                    confident_total += 1
                    if correct:
                        confident_correct += 1
                elif control == "<UNCERTAIN>":
                    uncertain_total += 1
                    if not correct:
                        uncertain_incorrect += 1
                elif control == "<VERIFY>":
                    verify_total += 1
                    if not correct:
                        verify_incorrect += 1

        honesty_score = {
            "confident_correct_rate": confident_correct / max(1, confident_total),
            "uncertain_incorrect_rate": uncertain_incorrect / max(1, uncertain_total),
            "verify_incorrect_rate": verify_incorrect / max(1, verify_total)
        }

        # ECE (if available)
        confidences = []
        correctness = []

        for result in inference_results:
            for step in result["trace"]:
                if "confidence" in step and "verified_correct" in step:
                    confidences.append(step["confidence"])
                    correctness.append(int(step["verified_correct"]))

        if confidences:
            ece = self.compute_ece(
                np.array(confidences),
                np.array(correctness)
            )
        else:
            ece = None

        return {
            "verification_request_rate": verification_rate,
            "honesty_score": honesty_score,
            "ece": ece,
            "sample_count": len(inference_results)
        }

    def compute_ece(self, confidences, correctness, num_bins=10):
        """Compute Expected Calibration Error."""

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            in_bin = (confidences >= bin_boundaries[i]) & \
                     (confidences < bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                avg_conf = confidences[in_bin].mean()
                avg_correct = correctness[in_bin].mean()
                weight = in_bin.sum() / len(confidences)
                ece += weight * abs(avg_conf - avg_correct)

        return ece

    def save_report(self, metrics, output_path):
        """Save metrics to JSON."""

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Reflection Monitor Report saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    monitor = ReflectionMonitor()
    results = monitor.load_inference_log(args.inference_log)
    metrics = monitor.compute_metrics(results)
    monitor.save_report(metrics, args.output)

    # Print summary
    print("\n=== Reflection Monitor Summary ===")
    print(f"Verification Request Rate: {metrics['verification_request_rate']['overall']:.1%}")
    print(f"Confident Correct Rate: {metrics['honesty_score']['confident_correct_rate']:.1%}")
    print(f"ECE: {metrics['ece']:.4f}" if metrics['ece'] else "ECE: N/A")
```

**Deliverables**:
- [ ] reflection_monitor.py script
- [ ] Verification rate tracking (overall + by difficulty)
- [ ] Honesty score computation
- [ ] ECE trend monitoring
- [ ] JSON report output

---

#### Enhancement 3: Confidence Galaxy Implementation

**File**: `knowledge3d/training/math_benchmarks/confidence_galaxy.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import json

@dataclass
class ConfidenceGalaxyEntry:
    """Store confidence predictions + verification results for shadow copy."""

    confidence_id: str
    trace_id: str
    problem_embedding: np.ndarray

    # Predictions
    rule_sequence: List[str]
    confidence_sequence: List[float]
    control_tokens: List[str]

    # Ground Truth
    correctness_sequence: List[int]
    verification_requested: List[bool]

    # Calibration Metrics
    avg_confidence: float
    avg_correctness: float
    calibration_error: float
    ece: float

    # Metadata
    model_version: str
    timestamp: str
    problem_difficulty: str

    def to_json(self) -> Dict:
        """Serialize to JSON (human-readable)."""
        return {
            "confidence_id": self.confidence_id,
            "trace_id": self.trace_id,
            "problem_embedding": self.problem_embedding.tolist(),
            "rule_sequence": self.rule_sequence,
            "confidence_sequence": self.confidence_sequence,
            "control_tokens": self.control_tokens,
            "correctness_sequence": self.correctness_sequence,
            "verification_requested": self.verification_requested,
            "avg_confidence": self.avg_confidence,
            "avg_correctness": self.avg_correctness,
            "calibration_error": self.calibration_error,
            "ece": self.ece,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "problem_difficulty": self.problem_difficulty
        }

    @classmethod
    def from_json(cls, data: Dict):
        """Deserialize from JSON."""
        return cls(
            confidence_id=data["confidence_id"],
            trace_id=data["trace_id"],
            problem_embedding=np.array(data["problem_embedding"]),
            rule_sequence=data["rule_sequence"],
            confidence_sequence=data["confidence_sequence"],
            control_tokens=data["control_tokens"],
            correctness_sequence=data["correctness_sequence"],
            verification_requested=data["verification_requested"],
            avg_confidence=data["avg_confidence"],
            avg_correctness=data["avg_correctness"],
            calibration_error=data["calibration_error"],
            ece=data["ece"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            problem_difficulty=data["problem_difficulty"]
        )

    def to_gltf_node(self):
        """Export as GLTF node for visualization."""

        # Color based on calibration
        if self.calibration_error < 0.1:
            color = [0.0, 1.0, 0.0]  # Green (well-calibrated)
        elif self.calibration_error < 0.3:
            color = [1.0, 1.0, 0.0]  # Yellow (moderate)
        else:
            color = [1.0, 0.0, 0.0]  # Red (poorly calibrated)

        # Size based on calibration error
        size = 0.5 + self.calibration_error  # Larger = more miscalibrated

        return {
            "geometry": "sphere",
            "position": self.problem_embedding[:3].tolist(),  # Use first 3 dims
            "scale": [size, size, size],
            "color": color,
            "metadata": {
                "type": "confidence",
                "confidence_id": self.confidence_id,
                "avg_confidence": self.avg_confidence,
                "calibration_error": self.calibration_error,
                "model_version": self.model_version
            }
        }

def save_confidence_galaxy(entries: List[ConfidenceGalaxyEntry],
                            output_path: str):
    """Save Confidence Galaxy to JSONL."""

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry.to_json()) + '\n')

    print(f"Saved {len(entries)} Confidence Galaxy entries to {output_path}")

def load_confidence_galaxy(input_path: str) -> List[ConfidenceGalaxyEntry]:
    """Load Confidence Galaxy from JSONL."""

    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(ConfidenceGalaxyEntry.from_json(json.loads(line)))

    return entries

def export_confidence_galaxy_gltf(entries: List[ConfidenceGalaxyEntry],
                                   output_path: str):
    """Export Confidence Galaxy as GLTF for visualization."""

    nodes = [entry.to_gltf_node() for entry in entries]

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes
    }

    with open(output_path, 'w') as f:
        json.dump(gltf, f, indent=2)

    print(f"Exported Confidence Galaxy GLTF to {output_path}")
```

**Deliverables**:
- [ ] ConfidenceGalaxyEntry schema
- [ ] JSON serialization (save/load)
- [ ] GLTF export for 3D visualization
- [ ] confidence_galaxy_v1.jsonl population

---

## Consolidated Success Criteria

**Phase 5.1 Complete When ALL Criteria Met**:

### Architecture (5.1A)
- [ ] NavigationModelWithConfidence implemented
- [ ] Control tokens integrated (+3 tokens)
- [ ] Confidence head outputs [0, 1] scores
- [ ] Backward compatibility with V6 checkpoints

### Verification (5.1B)
- [ ] VerificationLoop labels correctness accurately
- [ ] Verification dataset generated (>1000 examples)
- [ ] Unit tests pass (known correct/incorrect sequences)

### Training (5.1C)
- [ ] Calibration loss implemented (binary MSE + ECE)
- [ ] V7 trained with combined loss (rule + confidence)
- [ ] V7 rule accuracy ≥ V6 (no degradation)
- [ ] V7 calibration ECE < 0.15 (target < 0.1)

### Inference (5.1D)
- [ ] Reflective solve pipeline operational
- [ ] Low-confidence predictions trigger verification
- [ ] Verification requests genuine (>90% needed)
- [ ] Honesty score improved (85% → 95%)

### Gemini Enhancements
- [ ] Semantic control token tests pass (no token gaming)
- [ ] Reflection monitor produces stable metrics
- [ ] Confidence Galaxy populated (all traces logged)

### Integration
- [ ] Benchmark pipeline supports --use-reflection
- [ ] GLTF visualization shows confidence calibration
- [ ] V7 → V8 shadow copy path established

---

## Potential Blockers and Mitigations

### Blocker 1: SymPy Dependency in Verification Loop
**Risk**: Verification loop needs SymPy for expression parsing (sovereignty concern)

**Mitigation**:
- Phase 5.1: Use SymPy in verification (ingestion path, not hot path)
- Phase 5.2: Replace with sovereign Parser Galaxy (already planned Phase 2.4)
- Verification is offline (training data generation), not inference

---

### Blocker 2: Calibration Training Instability
**Risk**: ECE loss may not converge or confidence head overfits

**Mitigation**:
- Start with binary MSE (simpler, more stable)
- Add dropout (0.2) in confidence head
- Monitor ECE every 10 epochs, early stopping if diverging
- Reduce confidence_weight if rule accuracy degrades

---

### Blocker 3: Small Verification Dataset
**Risk**: <1000 examples insufficient for confidence head training

**Mitigation**:
- Generate from ALL Log Galaxy entries (V1-V4 traces)
- Include OracleGalaxy verified problems
- Data augmentation: Perturb embeddings slightly for more examples
- If still small, reduce confidence head capacity (smaller MLP)

---

### Blocker 4: Control Token Gaming
**Risk**: Model learns to output <CONFIDENT> to maximize reward

**Mitigation**:
- Verification provides ground truth (can't be gamed)
- Calibration loss penalizes misaligned confidence
- Semantic unit tests catch gaming behavior
- Monitor false confidence rate (confident but wrong)

---

### Blocker 5: Backward Compatibility
**Risk**: V7 changes break loading of older checkpoints

**Mitigation**:
- Add enable_control_tokens flag (default False)
- Checkpoint format stores control token metadata
- Load function checks for control tokens, falls back gracefully
- Unit test: Load V6 checkpoint in V7 model (should work)

---

## Timeline Summary (2-3 Weeks)

| Week | Phase | Key Deliverable | Duration |
|------|-------|----------------|----------|
| **1** | 5.1A | Model + tokens | 3-4 days |
| **1** | 5.1B | Verification loop + dataset | 3-4 days |
| **2** | 5.1C | V7 training (calibrated) | 5-7 days |
| **3** | 5.1D | Reflective inference | 3-4 days |
| **3** | Enhancements | Monitor + Confidence Galaxy | 2-3 days |

**Total**: 15-22 days (2-3 weeks depending on blockers)

---

## Post-Phase 5.1: Shadow Copy Path

**Phase 5.2** (Future): V8 learns from V7's Confidence Galaxy

```
V7 Confidence Galaxy
  └─→ (prediction, confidence, correctness) tuples
    └─→ V8 training dataset
      └─→ V8 learns better calibration than V7
        └─→ V8 Confidence Galaxy
          └─→ V9 training...
```

**Training Weight Strategy**:
- Well-calibrated entries (ECE < 0.1): weight = 1.0
- Moderate miscalibration (ECE < 0.3): weight = 2.0 (emphasize)
- Severe miscalibration (ECE > 0.3): weight = 3.0 (strongly emphasize)

**Expected Improvement**: Each generation reduces ECE by 20-30%.

---

## File Structure Summary

**New Files Created**:
```
knowledge3d/training/math_benchmarks/
├── navigation_model_v6.py                  # Control token support
├── navigation_model_with_confidence.py     # Confidence head
├── calibration_loss.py                     # Calibration objectives
└── confidence_galaxy.py                    # Confidence tracking

scripts/
├── verification_loop.py                    # Symbolic verification
├── generate_verification_dataset.py        # Dataset generation
├── train_navigation_v7_with_confidence.py  # V7 training
├── solve_with_reflection.py                # Reflective inference
└── reflection_monitor.py                   # Monitoring script

tests/
├── test_confidence_head_shape.py           # Architecture tests
├── test_control_token_vocab.py             # Token tests
├── test_verification_loop.py               # Verification tests
├── test_calibration_loss.py                # Loss tests
└── test_control_token_semantics.py         # Semantic tests (Gemini)

data/
├── verification_train_v1.jsonl             # Verification dataset
├── navigation_specialist_v7_confidence.pt  # V7 checkpoint
├── confidence_galaxy_v1.jsonl              # Confidence logs
├── reflection_monitor_v1.json              # Monitor output
└── confidence_galaxy_v1.gltf               # 3D visualization
```

**Updated Files**:
```
scripts/
└── run_sovereign_math_benchmarks.py        # Add --use-reflection flag
```

---

## Final Validation Checklist

Before declaring Phase 5.1 complete, verify:

**Quantitative**:
- [ ] V7 rule accuracy ≥ V6 accuracy (no degradation)
- [ ] V7 ECE < 0.15 (calibration quality)
- [ ] Verification request rate 5-15% (reasonable)
- [ ] Confident correct rate >90% (high precision)
- [ ] Uncertain incorrect rate >60% (honest uncertainty)

**Qualitative**:
- [ ] Simple problems don't trigger <VERIFY>
- [ ] Hard problems trigger <UNCERTAIN> or <VERIFY>
- [ ] Model admits uncertainty (not hallucinating)
- [ ] Confidence correlates with difficulty

**Integration**:
- [ ] Benchmark pipeline --use-reflection works
- [ ] Reflection monitor produces stable reports
- [ ] Confidence Galaxy populated (all traces)
- [ ] GLTF visualization renders correctly
- [ ] V7 → V8 path established (data pipeline ready)

---

## Architectural Principles Validated

**All K3D Core Principles Maintained**:
- ✅ **Explicit Design**: Control tokens formalized (not accidental)
- ✅ **Verification-Driven**: Symbolic checks = ground truth
- ✅ **Composability**: Confidence head is modular (can disable)
- ✅ **Shadow Copy**: V7 → V8 continual learning path
- ✅ **Sovereignty**: All in-house (no external APIs in hot path)

**True Self-Reflection Achieved**:
- Model learns to assess confidence accurately
- Requests help when uncertain (not hallucinating)
- Calibration improves over time via shadow copy
- System becomes more honest with each generation

---

**Document Date**: January 15, 2026
**Phase**: 5.1 Implementation Plan (Final Collaborative Version)
**Status**: ✅ **APPROVED - CODEX BEGINS IMPLEMENTATION**

---

**Contributors' Sign-Off**:
- **Claude** (Architecture): ✅ Architectural plan validated
- **Gemini** (Integration): ✅ Enhancements integrated
- **Codex** (Implementation): ✅ Ready to implement Phase 5.1A

**Next Action**: Codex begins Phase 5.1A (Model + Token Architecture)

---

**This document represents the complete, collaborative implementation plan for Phase 5.1: True Self-Reflection. All three agents (Claude, Gemini, Codex) have contributed their expertise to ensure architectural soundness, integration robustness, and implementation clarity. The plan is comprehensive, actionable, and ready for execution.** 🚀
