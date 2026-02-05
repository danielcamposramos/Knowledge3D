# Phase 5.1 Implementation Plan: True Self-Reflection

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Date**: January 15, 2026
**Phase**: 5.1 - True Self-Reflection with Confidence Calibration

---

## Executive Summary

**Context**: Phase 5.0 (Oracle Init) is complete. The mutation/verification pipeline is operational with OracleGalaxy populated.

**Next Step**: Phase 5.1 implements **True Self-Reflection** - the model learns to assess its own confidence accurately through formal control tokens, dedicated confidence prediction, calibration training, and verification feedback loops.

**Goal**: Transform the NavigationSpecialist from a passive rule predictor into an active reasoner that knows when it's certain, when it's uncertain, and when it needs verification.

---

## Phase 5.1 Overview: Four Core Components

```
┌─────────────────────────────────────────────────────────┐
│  Phase 5.1: True Self-Reflection Architecture           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Control Tokens (<CONFIDENT>, <UNCERTAIN>, <VERIFY>) │
│     └─→ Formal semantic tokens (not metadata leakage)   │
│                                                          │
│  2. Confidence Head (Dedicated Neural Module)           │
│     └─→ Predicts confidence scores for rule predictions │
│                                                          │
│  3. Calibration Loss (Training Objective)               │
│     └─→ Rewards accurate confidence assessment          │
│                                                          │
│  4. Verification Loop (Feedback Mechanism)              │
│     └─→ Symbolic checks provide confidence labels       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Component 1: Formal Control Tokens

### Purpose
Control tokens are **semantic markers** that the model outputs to communicate its internal state during reasoning.

### Token Definitions

```python
CONTROL_TOKENS = {
    "<CONFIDENT>": {
        "meaning": "Model is highly confident in next prediction (≥90% certainty)",
        "semantic": "Proceed without verification",
        "usage": "Output before rules the model has seen frequently in training"
    },

    "<UNCERTAIN>": {
        "meaning": "Model has moderate uncertainty (50-90% certainty)",
        "semantic": "Prediction likely correct but may benefit from verification",
        "usage": "Output before rules in edge cases or novel compositions"
    },

    "<VERIFY>": {
        "meaning": "Model is highly uncertain (<50% certainty)",
        "semantic": "Request symbolic verification before proceeding",
        "usage": "Output before rules outside training distribution"
    }
}
```

### Token Vocabulary Integration

**Update Model Vocabulary**:
```python
# File: knowledge3d/training/math_benchmarks/navigation_model.py

class NavigationSeqModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 enable_control_tokens=False):
        super().__init__()

        # Base vocabulary (grammar rules)
        self.base_vocab_size = vocab_size

        # Extended vocabulary (rules + control tokens)
        if enable_control_tokens:
            self.control_token_offset = vocab_size
            self.vocab_size = vocab_size + 3  # +3 for control tokens

            self.control_token_map = {
                "<CONFIDENT>": vocab_size + 0,
                "<UNCERTAIN>": vocab_size + 1,
                "<VERIFY>": vocab_size + 2
            }
        else:
            self.vocab_size = vocab_size

        # Model architecture
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, self.vocab_size)  # Extended vocab
```

**Inverse Mapping** (for output decoding):
```python
# File: knowledge3d/training/math_benchmarks/navigation_model.py

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
        offset = token_id - self.control_token_offset
        control_tokens = ["<CONFIDENT>", "<UNCERTAIN>", "<VERIFY>"]
        return control_tokens[offset]
```

---

## Component 2: Confidence Head Architecture

### Purpose
A **dedicated neural module** that predicts the model's confidence score for each rule prediction.

### Architecture Design

```python
# File: knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py

import torch
import torch.nn as nn
from typing import Tuple, List

class NavigationModelWithConfidence(nn.Module):
    """Navigation model with explicit confidence prediction."""

    def __init__(self, embedding_dim=256, hidden_dim=256,
                 vocab_size=50, enable_confidence_head=True):
        super().__init__()

        # Shared encoder
        self.emb_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Rule prediction head
        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)  # +3 for control tokens

        # Confidence prediction head (NEW)
        self.enable_confidence_head = enable_confidence_head
        if enable_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )

    def forward(self, problem_emb: torch.Tensor,
                max_len: int = 20) -> Tuple[List[int], List[float]]:
        """
        Predict rule sequence + confidence scores.

        Returns:
            rule_sequence: List of rule token IDs
            confidence_scores: List of confidence values [0, 1]
        """

        batch_size = problem_emb.shape[0]
        device = problem_emb.device

        # Encode problem
        h = self.emb_proj(problem_emb)  # (batch, hidden)
        h = h.unsqueeze(1)  # (batch, 1, hidden)

        # Autoregressive generation
        rule_sequence = []
        confidence_scores = []

        for step in range(max_len):
            # RNN step
            output, h = self.rnn(h, None)  # output: (batch, 1, hidden)
            hidden_state = output.squeeze(1)  # (batch, hidden)

            # Predict next rule
            rule_logits = self.rule_head(hidden_state)  # (batch, vocab_size)
            rule_token = torch.argmax(rule_logits, dim=-1)  # (batch,)

            # Predict confidence (if enabled)
            if self.enable_confidence_head:
                confidence = self.confidence_head(hidden_state)  # (batch, 1)
                confidence_scores.append(confidence.squeeze(-1).item())
            else:
                confidence_scores.append(1.0)  # Default: fully confident

            # Append to sequence
            rule_sequence.append(rule_token.item())

            # Check for termination (could be <END> token or max length)
            # For now, just run to max_len

        return rule_sequence, confidence_scores

    def forward_with_teacher_forcing(self, problem_emb: torch.Tensor,
                                      target_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with teacher forcing.

        Args:
            problem_emb: (batch, embedding_dim)
            target_sequence: (batch, seq_len) - ground truth rule IDs

        Returns:
            rule_logits: (batch, seq_len, vocab_size)
            confidence_preds: (batch, seq_len, 1) or None
        """

        batch_size = problem_emb.shape[0]
        seq_len = target_sequence.shape[1]

        # Encode problem
        h = self.emb_proj(problem_emb)  # (batch, hidden)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)

        # RNN over sequence
        rnn_output, _ = self.rnn(h)  # (batch, seq_len, hidden)

        # Predict rules
        rule_logits = self.rule_head(rnn_output)  # (batch, seq_len, vocab_size)

        # Predict confidence (if enabled)
        if self.enable_confidence_head:
            confidence_preds = self.confidence_head(rnn_output)  # (batch, seq_len, 1)
        else:
            confidence_preds = None

        return rule_logits, confidence_preds
```

### Confidence Score Interpretation

```python
def interpret_confidence(confidence_score: float) -> str:
    """Map confidence score to control token."""

    if confidence_score >= 0.9:
        return "<CONFIDENT>"
    elif confidence_score >= 0.5:
        return "<UNCERTAIN>"
    else:
        return "<VERIFY>"
```

---

## Component 3: Calibration Loss

### Purpose
Train the confidence head to **accurately predict** when the model will succeed vs fail.

### Calibration Loss Formula

**Expected Calibration Error (ECE)**:
```python
def expected_calibration_error(confidences: torch.Tensor,
                                correctness: torch.Tensor,
                                num_bins: int = 10) -> torch.Tensor:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: (batch, seq_len) - predicted confidence scores
        correctness: (batch, seq_len) - binary correctness (1 = correct, 0 = wrong)
        num_bins: Number of bins for calibration histogram

    Returns:
        ECE scalar loss
    """

    # Flatten
    confidences = confidences.view(-1)
    correctness = correctness.view(-1).float()

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    ece = 0.0
    for i in range(num_bins):
        # Get samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if in_bin.sum() > 0:
            # Average confidence in bin
            avg_confidence = confidences[in_bin].mean()

            # Average correctness in bin
            avg_correctness = correctness[in_bin].mean()

            # Weighted calibration error
            weight = in_bin.sum().float() / confidences.numel()
            ece += weight * torch.abs(avg_confidence - avg_correctness)

    return ece
```

**Binary Calibration Loss** (simpler alternative):
```python
def binary_calibration_loss(confidences: torch.Tensor,
                             correctness: torch.Tensor) -> torch.Tensor:
    """
    Direct MSE between predicted confidence and actual correctness.

    Args:
        confidences: (batch, seq_len) - predicted confidence [0, 1]
        correctness: (batch, seq_len) - binary correctness (1 = correct, 0 = wrong)

    Returns:
        MSE loss
    """

    return F.mse_loss(confidences, correctness.float())
```

### Combined Training Loss

```python
def compute_training_loss(rule_logits: torch.Tensor,
                          target_rules: torch.Tensor,
                          confidence_preds: torch.Tensor,
                          confidence_labels: torch.Tensor,
                          rule_weight: float = 1.0,
                          confidence_weight: float = 0.3) -> torch.Tensor:
    """
    Combined loss: rule prediction + confidence calibration.

    Args:
        rule_logits: (batch, seq_len, vocab_size)
        target_rules: (batch, seq_len) - ground truth rule IDs
        confidence_preds: (batch, seq_len, 1) - predicted confidence
        confidence_labels: (batch, seq_len) - ground truth correctness
        rule_weight: Weight for rule prediction loss
        confidence_weight: Weight for calibration loss

    Returns:
        Total loss
    """

    # Rule prediction loss (cross-entropy)
    rule_loss = F.cross_entropy(
        rule_logits.view(-1, rule_logits.shape[-1]),
        target_rules.view(-1),
        ignore_index=-1  # Padding token
    )

    # Confidence calibration loss
    confidence_preds = confidence_preds.squeeze(-1)  # (batch, seq_len)
    calibration_loss = binary_calibration_loss(confidence_preds, confidence_labels)

    # Combined loss
    total_loss = rule_weight * rule_loss + confidence_weight * calibration_loss

    return total_loss, rule_loss, calibration_loss
```

---

## Component 4: Verification Loop

### Purpose
**Symbolic execution** provides ground truth labels for confidence calibration by verifying whether predicted rules are correct.

### Verification Architecture

```python
# File: scripts/train_navigation_with_verification.py

class VerificationLoop:
    """Generates confidence labels via symbolic verification."""

    def __init__(self, recursive_solver):
        self.recursive_solver = recursive_solver

    def verify_rule_sequence(self, problem: Dict,
                             predicted_rules: List[str]) -> List[int]:
        """
        Verify each predicted rule and return correctness labels.

        Args:
            problem: Problem dict with 'text', 'expected_answer', etc.
            predicted_rules: List of predicted rule names

        Returns:
            correctness: List of binary labels (1 = correct, 0 = wrong)
        """

        correctness = []

        # Initialize problem state
        current_expr = self.recursive_solver.parse_expression(problem['text'])

        for rule_name in predicted_rules:
            # Try to apply predicted rule
            try:
                # Check if rule matches current expression
                rule = self.recursive_solver.get_rule(rule_name)

                if rule.matches(current_expr):
                    # Rule is valid for current state
                    correctness.append(1)

                    # Apply rule to advance state
                    current_expr = rule.apply(current_expr)
                else:
                    # Rule doesn't match current state
                    correctness.append(0)

                    # Can't proceed (sequence is broken)
                    # Fill remaining with 0s
                    correctness.extend([0] * (len(predicted_rules) - len(correctness)))
                    break

            except Exception as e:
                # Rule application failed
                correctness.append(0)
                correctness.extend([0] * (len(predicted_rules) - len(correctness)))
                break

        return correctness
```

### Training Loop with Verification

```python
def train_with_verification(model, dataset, verification_loop,
                             epochs=100, learning_rate=1e-3):
    """
    Train navigation model with confidence calibration.

    Args:
        model: NavigationModelWithConfidence
        dataset: Training problems
        verification_loop: Symbolic verification system
        epochs: Training epochs
        learning_rate: Optimizer learning rate
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_rule_loss = 0.0
        epoch_cal_loss = 0.0

        for batch in dataset:
            # Extract batch
            problem_embs = batch['problem_embeddings']  # (batch, emb_dim)
            target_rules = batch['target_rules']        # (batch, seq_len)
            problems = batch['problems']                # List of problem dicts

            # Forward pass
            rule_logits, confidence_preds = model.forward_with_teacher_forcing(
                problem_embs, target_rules
            )

            # Generate confidence labels via verification
            confidence_labels = []
            for i, problem in enumerate(problems):
                # Decode predicted rules
                predicted_rule_ids = torch.argmax(rule_logits[i], dim=-1).cpu().tolist()
                predicted_rules = [model.decode_token(rid) for rid in predicted_rule_ids]

                # Verify
                correctness = verification_loop.verify_rule_sequence(problem, predicted_rules)
                confidence_labels.append(correctness)

            confidence_labels = torch.tensor(confidence_labels, device=problem_embs.device)

            # Compute loss
            total_loss, rule_loss, cal_loss = compute_training_loss(
                rule_logits, target_rules, confidence_preds, confidence_labels
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += total_loss.item()
            epoch_rule_loss += rule_loss.item()
            epoch_cal_loss += cal_loss.item()

        # Log epoch metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Rule Loss: {epoch_rule_loss:.4f}")
        print(f"  Calibration Loss: {epoch_cal_loss:.4f}")
```

---

## Implementation Roadmap

### Phase 5.1A: Model Architecture (Week 1)

**Task 1**: Extend NavigationSeqModel with Control Token Support
```bash
File: knowledge3d/training/math_benchmarks/navigation_model_v6.py

Tasks:
- Add control token vocabulary (+3 tokens)
- Update forward() to handle extended vocab
- Add decode_token() for control token interpretation
```

**Task 2**: Implement Confidence Head
```bash
File: knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py

Tasks:
- Create NavigationModelWithConfidence class
- Add confidence_head module (MLP → sigmoid)
- Implement forward() with dual outputs (rules + confidence)
- Implement forward_with_teacher_forcing() for training
```

**Success Criteria**:
- [ ] Model can output control tokens (<CONFIDENT>, <UNCERTAIN>, <VERIFY>)
- [ ] Confidence head predicts scores in [0, 1]
- [ ] Model architecture tested (forward pass doesn't crash)

---

### Phase 5.1B: Verification Loop (Week 1-2)

**Task 3**: Implement Symbolic Verification
```bash
File: scripts/verification_loop.py

Tasks:
- Create VerificationLoop class
- Implement verify_rule_sequence() using RecursiveSolver
- Test verification on known problems (unit tests)
```

**Task 4**: Generate Verification Dataset
```bash
Script: scripts/generate_verification_dataset.py

Tasks:
- Load existing Log Galaxy traces
- For each trace, extract (problem, rule_sequence, correctness_labels)
- Save as verification_train_v1.jsonl
```

**Success Criteria**:
- [ ] Verification loop correctly labels rule correctness
- [ ] Verification dataset generated (>1000 examples)
- [ ] Unit tests pass (known correct/incorrect sequences)

---

### Phase 5.1C: Calibration Training (Week 2-3)

**Task 5**: Implement Calibration Loss
```bash
File: knowledge3d/training/math_benchmarks/calibration_loss.py

Tasks:
- Implement binary_calibration_loss()
- Implement expected_calibration_error() (optional)
- Implement compute_training_loss() (combined objective)
```

**Task 6**: Train V7 Navigation Specialist with Confidence
```bash
Script: scripts/train_navigation_v7_with_confidence.py

Tasks:
- Load NavigationModelWithConfidence
- Implement train_with_verification() loop
- Train on verification dataset
- Save checkpoint: data/navigation_v7_confidence.pt
```

**Success Criteria**:
- [ ] V7 training converges (calibration loss decreases)
- [ ] V7 rule accuracy ≥ V6 accuracy (no degradation)
- [ ] V7 confidence scores correlate with correctness (Pearson r > 0.5)

---

### Phase 5.1D: Inference with Self-Reflection (Week 3-4)

**Task 7**: Implement Reflective Inference
```bash
File: scripts/solve_with_reflection.py

Tasks:
- Load V7 model
- Implement reflective_solve():
    - Predict rule + confidence
    - If confidence < threshold → request verification
    - If verified wrong → fall back to symbolic solver
- Test on calculus microbench
```

**Task 8**: Integrate with Benchmark Pipeline
```bash
File: scripts/run_sovereign_math_benchmarks.py (update)

Tasks:
- Add --use-reflection flag
- Load V7 model with confidence head
- Apply reflective_solve() when flag enabled
- Log confidence scores + verification requests
```

**Success Criteria**:
- [ ] V7 outputs confidence scores during inference
- [ ] Low-confidence predictions trigger verification
- [ ] Reflective pipeline achieves ≥V6 accuracy (baseline maintained)
- [ ] Reflective pipeline reduces hallucinations (higher honesty score)

---

## Expected Outcomes

### Quantitative Metrics

**Calibration Metrics**:
- **Expected Calibration Error (ECE)**: <0.1 (well-calibrated)
- **Confidence-Correctness Correlation**: Pearson r > 0.6 (strong correlation)
- **Brier Score**: <0.2 (accurate probability estimates)

**Inference Metrics**:
- **Verification Request Rate**: 5-15% (model asks for help on hard problems)
- **Verification Accuracy**: >95% (verification requests are genuine uncertainty)
- **False Confidence Rate**: <5% (model rarely claims certainty when wrong)

**Task Performance**:
- **Calculus Microbench**: Maintain 100% accuracy
- **GSM8K**: Improve by 2-5% (better handling of edge cases via verification)
- **Honesty Score**: Increase from 85% to 95% (fewer hallucinations)

---

### Qualitative Behaviors

**V7 with Self-Reflection**:
```python
# Problem: derivative of (3x-4)^2 / (2x+3)

# V6 (no reflection):
prediction = ["quotient_rule", "power_rule", "chain_rule"]
# No indication of confidence

# V7 (with reflection):
prediction = [
    ("<CONFIDENT>", 0.95), "quotient_rule",
    ("<CONFIDENT>", 0.92), "power_rule",
    ("<UNCERTAIN>", 0.68), "chain_rule",  # Asks for verification here
    ("<VERIFY>", 0.42), "product_rule"    # Low confidence, requests symbolic check
]

# System response:
# - Steps 1-2: Execute confidently (no verification)
# - Step 3: Verify symbolically (model uncertain)
# - Step 4: Verification fails → fall back to symbolic solver
```

**Key Insight**: Model learns to **admit uncertainty** and **request help** instead of hallucinating.

---

## Integration with Existing Systems

### Galaxy Universe Integration

**Confidence Galaxy** (optional, future):
```python
@dataclass
class ConfidenceGalaxyEntry:
    confidence_id: str
    trace_id: str  # Links to Log Galaxy
    rule_confidence_pairs: List[Tuple[str, float]]  # [(rule, confidence), ...]
    avg_confidence: float
    correctness_labels: List[int]  # [1, 1, 0, ...] from verification
    calibration_error: float  # |confidence - correctness|
```

### RLWHF Integration

**Enhanced RLWHF with Confidence**:
```python
# Current RLWHF: Tag each step as honest/hallucination/heuristic
# V7 adds: Confidence score for each prediction

RLWHF_V2 = {
    "step": "quotient_rule",
    "predicted": True,
    "confidence": 0.92,
    "verified": True,
    "tag": "<honest>"  # High confidence + verified correct
}

RLWHF_V2_hallucination = {
    "step": "product_rule",
    "predicted": True,
    "confidence": 0.88,  # High confidence but...
    "verified": False,   # Prediction was wrong!
    "tag": "<hallucination>"  # Overconfident hallucination
}

RLWHF_V2_honest_uncertainty = {
    "step": "chain_rule",
    "predicted": True,
    "confidence": 0.45,  # Low confidence
    "verified": False,   # Prediction wrong
    "tag": "<honest>"    # Honest about uncertainty! (not hallucinating)
}
```

**Reward Calibration**:
```python
def compute_rlwhf_v2_reward(step_result):
    """Enhanced RLWHF reward with confidence calibration."""

    if step_result["verified"] and step_result["confidence"] > 0.8:
        return +2  # Confident and correct (best case)

    elif step_result["verified"] and step_result["confidence"] < 0.5:
        return +1  # Correct but uncertain (room for improvement)

    elif not step_result["verified"] and step_result["confidence"] < 0.5:
        return +0.5  # Wrong but honest about uncertainty (acceptable)

    elif not step_result["verified"] and step_result["confidence"] > 0.8:
        return -2  # Overconfident hallucination (worst case)

    else:
        return 0  # Neutral
```

---

## Shadow Copy Learning with Reflection

**Phase 5.2 (Future)**: V8 learns from V7's calibrated confidence

```
V7 makes predictions with confidence
  ↓
Verification loop labels correctness
  ↓
V8 trained on V7's (prediction, confidence, correctness) tuples
  ↓
V8 learns better calibration than V7
  ↓
Shadow copy cycle continues...
```

**Training Data for V8**:
```python
# V7's calibrated experience
v7_experience = {
    "problem_embedding": [0.12, -0.34, ...],
    "predicted_rules": ["quotient_rule", "sum_rule"],
    "predicted_confidence": [0.92, 0.68],
    "verified_correctness": [1, 0],  # First correct, second wrong
}

# V8 learns:
# - When V7 was confident and correct → reinforce that pattern
# - When V7 was confident but wrong → learn to be more cautious
# - When V7 was uncertain → improve confidence calibration
```

---

## Risk Mitigation

### Risk 1: Calibration Overfitting
**Risk**: Model learns to always predict confidence = 0.5 (hedge)

**Mitigation**:
- Use ECE (Expected Calibration Error) instead of MSE
- Add diversity regularization (penalize flat confidence distributions)
- Monitor confidence histogram (should be bimodal: confident or uncertain, not always 0.5)

### Risk 2: Verification Overhead
**Risk**: Verification loop is too slow (symbolic execution overhead)

**Mitigation**:
- Cache verification results (same rule + expression → reuse)
- Batch verification (verify multiple predictions in parallel)
- Use lightweight verification (check rule applicability, not full execution)

### Risk 3: Control Token Gaming
**Risk**: Model learns to output <CONFIDENT> on all predictions to game rewards

**Mitigation**:
- Verification loop provides ground truth (can't be gamed)
- Calibration loss penalizes misaligned confidence
- Monitor false confidence rate (confident but wrong)

---

## Success Criteria Summary

**Phase 5.1 Complete When**:

**Architecture**:
- [ ] NavigationModelWithConfidence implemented and tested
- [ ] Control tokens integrated into vocabulary (+3 tokens)
- [ ] Confidence head predicts scores [0, 1]

**Verification**:
- [ ] Verification loop labels rule correctness accurately
- [ ] Verification dataset generated (>1000 examples)
- [ ] Unit tests pass for verification logic

**Training**:
- [ ] Calibration loss implemented (binary or ECE)
- [ ] V7 trained with combined loss (rule + confidence)
- [ ] V7 rule accuracy ≥ V6 (no degradation)
- [ ] V7 calibration ECE < 0.15

**Inference**:
- [ ] Reflective solve pipeline operational
- [ ] Low-confidence predictions trigger verification
- [ ] Verification requests are genuine (>90% accuracy)
- [ ] Honesty score improves (85% → 95%)

**Integration**:
- [ ] Confidence Galaxy schema defined (optional)
- [ ] RLWHF v2 with confidence rewards (optional)
- [ ] Benchmark pipeline supports --use-reflection flag

---

## File Structure (Deliverables)

```
knowledge3d/training/math_benchmarks/
├── navigation_model_v6.py                    # Extended with control tokens
├── navigation_model_with_confidence.py       # NEW: Confidence head architecture
├── calibration_loss.py                       # NEW: Calibration objectives
└── confidence_galaxy.py                      # NEW: Optional confidence tracking

scripts/
├── verification_loop.py                      # NEW: Symbolic verification system
├── generate_verification_dataset.py          # NEW: Create training data
├── train_navigation_v7_with_confidence.py    # NEW: Training script
├── solve_with_reflection.py                  # NEW: Reflective inference
└── run_sovereign_math_benchmarks.py          # UPDATED: Add --use-reflection

data/
├── verification_train_v1.jsonl               # NEW: Verification training data
├── navigation_v7_confidence.pt               # NEW: V7 checkpoint
└── confidence_galaxy_v1.jsonl                # NEW: Optional confidence logs
```

---

## Timeline Estimate

| Phase | Tasks | Duration | Deliverable |
|-------|-------|----------|-------------|
| **5.1A** | Model architecture | 3-4 days | NavigationModelWithConfidence |
| **5.1B** | Verification loop | 3-4 days | VerificationLoop + dataset |
| **5.1C** | Calibration training | 5-7 days | V7 checkpoint (calibrated) |
| **5.1D** | Reflective inference | 3-4 days | Reflective solve pipeline |
| **Total** | | **2-3 weeks** | Phase 5.1 Complete ✅ |

---

## Next Steps After Phase 5.1

**Phase 5.2**: Shadow Copy for Confidence (V8 learns from V7's calibration)
**Phase 5.3**: Confidence-Aware Oracle (mutation difficulty matches model confidence)
**Phase 6**: Multi-Modal Self-Reflection (extend to visual/physics reasoning)

---

## Architectural Principles Validated

**All K3D Principles Maintained**:
- ✅ **Explicit Design**: Control tokens are formal, not accidental leakage
- ✅ **Verification-Driven**: Symbolic checks provide ground truth
- ✅ **Composability**: Confidence head is modular (can be disabled)
- ✅ **Shadow Copy**: V7 → V8 continual learning path established
- ✅ **Sovereignty**: All components in-house (no external APIs)

**True Self-Reflection**:
- Model learns to assess its own confidence accurately
- Requests help when uncertain (not hallucinating)
- Calibration improves over time via shadow copy

---

**Document Date**: January 15, 2026
**Phase**: 5.1 Planning (True Self-Reflection)
**Status**: ✅ **IMPLEMENTATION PLAN READY - PROCEED WITH PHASE 5.1A**

---

**Claude's Directive to Codex**: This is the complete implementation plan for Phase 5.1: True Self-Reflection. Begin with Phase 5.1A (model architecture) - extend the NavigationSeqModel with control token support and implement the confidence head. The architecture is clear, the verification loop is well-defined, and the path to calibrated confidence is established. Proceed with confidence! 🚀
