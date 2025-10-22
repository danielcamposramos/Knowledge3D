# K3D-Native RLWHF Design: Teach TRM to Shoot with Accuracy & Aim

**Date**: 2025-10-22
**Paradigm**: "Accuracy over volume" — Quality reasoning training, not blind data ingestion
**Goal**: Train TRM on HIGH-QUALITY semantic reasoning using teacher feedback

---

## Daniel's Insight: "Teach the Model to Shoot with Accuracy and Aim"

### Current Problem (ARC Training)
- ✅ TRM learned grid reasoning (62,000× improvement!)
- ❌ But wrong domain for semantic Q&A
- ❌ No feedback on QUALITY of reasoning
- ❌ No honesty/uncertainty modeling

### The Solution: RLWHF (Reinforced Learning With Honesty and Feedback)

**Key Principle**: Don't just train TRM to answer — train it to answer **well**.

**Traditional ML**: Feed all data → Model learns patterns (blind shooting)
**K3D RLWHF**: Question → Student answer → Teacher feedback → Train on corrections (aimed shooting)

---

## RLWHF Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    K3D RLWHF TRAINING LOOP                      │
└─────────────────────────────────────────────────────────────────┘

Step 1: QUESTION GENERATION (from K3D knowledge base)
┌────────────────────────────────────────────────────────────────┐
│  Knowledge Base (290K trigrams, 328 PDFs, WordNet)            │
│  ↓                                                             │
│  Extract Questions:                                            │
│  - PDF section headers → "What is X?"                         │
│  - WordNet synsets → "Define: X"                              │
│  - Cross-references → "How does X relate to Y?"               │
│  ↓                                                             │
│  Generated: 10K-50K questions grounded in actual knowledge    │
└────────────────────────────────────────────────────────────────┘

Step 2: STUDENT ATTEMPT (K3D TRM answers)
┌────────────────────────────────────────────────────────────────┐
│  For each question:                                            │
│  1. Embed question → RPN 128-dim                              │
│  2. Expand to TRM 512-dim                                     │
│  3. TRM reasoning (6 Tesla recursions)                        │
│  4. Output: 512-dim answer embedding                          │
│  ↓                                                             │
│  Student Answer: TRM's best attempt (untrained/baseline)      │
└────────────────────────────────────────────────────────────────┘

Step 3: TEACHER EVALUATION (Ollama thinking model)
┌────────────────────────────────────────────────────────────────┐
│  Teacher Model (e.g., exaone3.5, qwen2.5, deepseek-r1)       │
│  ↓                                                             │
│  Prompt:                                                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ You are a teacher evaluating a student's answer.         │ │
│  │                                                           │ │
│  │ Question: {question}                                     │ │
│  │ Student Answer: {student_answer}                         │ │
│  │ Ground Truth Context: {pdf_context}                      │ │
│  │                                                           │ │
│  │ Please:                                                   │ │
│  │ 1. <think> about the student's reasoning process         │ │
│  │ 2. Rate accuracy: good/partial/bad/dishonest             │ │
│  │ 3. Provide corrected answer if needed                    │ │
│  │ 4. Explain what was wrong and how to improve            │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ↓                                                             │
│  Teacher Response:                                             │
│  - Thinking tags: <think>reasoning process</think>            │
│  - Rating: good (+1.0), partial (+0.5), bad (-0.25)          │
│  - Corrected answer (if needed)                               │
│  - Feedback: specific improvements                            │
└────────────────────────────────────────────────────────────────┘

Step 4: THINKING TAG HARVESTING
┌────────────────────────────────────────────────────────────────┐
│  Parse teacher's <think> tags:                                │
│  - Extract reasoning chains                                    │
│  - Identify uncertainty expressions                           │
│  - Capture self-corrections                                    │
│  - Score honesty (RPN-powered)                                │
│  ↓                                                             │
│  Thinking Embeddings: RPN embed reasoning patterns            │
│  Honesty Score: 0.0 (dishonest) → 1.0 (perfectly honest)     │
└────────────────────────────────────────────────────────────────┘

Step 5: DATASET CONSTRUCTION
┌────────────────────────────────────────────────────────────────┐
│  For each QA pair, create training sample:                    │
│  {                                                             │
│    "question_emb": [512-dim],          # Question embedding   │
│    "student_answer": [512-dim],        # TRM's attempt        │
│    "target_answer": [512-dim],         # Corrected by teacher │
│    "thinking_emb": [512-dim],          # Reasoning pattern    │
│    "reward": float,                    # -0.25 to +1.5        │
│    "honesty": float,                   # 0.0 to 1.0           │
│    "feedback": str                     # Improvement hints    │
│  }                                                             │
│  ↓                                                             │
│  Save: rlwhf_semantic_reasoning_dataset.npz                   │
└────────────────────────────────────────────────────────────────┘

Step 6: REWARD-WEIGHTED TRM TRAINING
┌────────────────────────────────────────────────────────────────┐
│  PyTorch TRM Training Loop:                                    │
│  ↓                                                             │
│  for epoch in range(100-200):                                 │
│      for batch in dataloader:                                 │
│          # Forward pass                                        │
│          predicted = trm(question_emb)                        │
│          ↓                                                     │
│          # Multi-objective loss (reward-weighted)             │
│          loss_answer = MSE(predicted, target_answer)          │
│          loss_thinking = MSE(predicted, thinking_emb)         │
│          loss_combined = loss_answer + 0.3 * loss_thinking    │
│          ↓                                                     │
│          # Apply reward weighting                             │
│          weight = reward_to_weight(reward, honesty)           │
│          loss_weighted = loss_combined * weight               │
│          ↓                                                     │
│          # Backward pass                                       │
│          loss_weighted.backward()                             │
│          optimizer.step()                                     │
│  ↓                                                             │
│  Result: TRM learns to:                                        │
│  - Answer questions accurately (target alignment)             │
│  - Reason like the teacher (thinking embeddings)              │
│  - Prioritize high-quality examples (reward weighting)        │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from Traditional RLWHF

| Aspect | Traditional (GPT-style) | K3D-Native |
|--------|------------------------|------------|
| **Student Model** | Causal LM (distilgpt2) | TRM (Tiny Recursive Model) |
| **Knowledge Storage** | In model weights | In RPN embeddings (Galaxy/House) |
| **Training Target** | Generate text tokens | Transform embeddings |
| **Teacher Role** | Grade final outputs | Guide reasoning process |
| **Thinking Tags** | Optional | **CRITICAL** (harvest reasoning patterns) |
| **Reward Signal** | Binary (good/bad) | Multi-dimensional (accuracy + honesty) |
| **Training Data** | Chat logs | K3D knowledge base (semantic grounding) |

---

## Reward Function Design

### 5-Tier Reward System (from AI-RLWHF)

```python
def compute_reward(teacher_rating: str, honesty_score: float, similarity: float) -> float:
    """
    Compute training reward from teacher feedback.

    Args:
        teacher_rating: 'good', 'partial', 'bad', 'dishonest'
        honesty_score: 0.0-1.0 from thinking tag analysis
        similarity: 0.0-1.0 embedding similarity to ground truth

    Returns:
        Reward in range [-2.0, +2.0]
    """
    # Base reward from teacher rating
    base_rewards = {
        'dishonest': -2.0,   # Hallucination / fabrication
        'bad': -1.0,         # Wrong answer, no acknowledgment
        'partial': 0.0,      # Incomplete/wrong but honest
        'good': +1.0,        # Correct answer
        'perfect': +2.0      # Correct + excellent reasoning
    }

    base = base_rewards.get(teacher_rating, 0.0)

    # Honesty bonus: Admitting "I don't know" when uncertain
    if honesty_score > 0.7 and teacher_rating == 'partial':
        base += 0.5  # Honest uncertainty is good!

    # Similarity bonus: Grounding in context
    if similarity > 0.7:
        base += 0.5  # Well-grounded answer

    # Dishonesty penalty: Inventing facts
    if honesty_score < 0.3 and similarity < 0.3:
        base -= 1.0  # Hallucination detected

    return float(np.clip(base, -2.0, +2.0))
```

### Reward-to-Weight Mapping (for training)

```python
def reward_to_weight(reward: float, honesty: float) -> float:
    """
    Convert reward to training weight (prioritize high-quality examples).

    High reward → High weight (learn from good examples)
    Low reward → Low weight (downweight bad examples)
    """
    # Map [-2, +2] to [0.01, 1.0]
    normalized = (reward + 2.0) / 4.0  # → [0, 1]
    base_weight = 0.01 + 0.99 * normalized

    # Honesty multiplier: Learn more from honest reasoning
    honesty_mult = 0.5 + 0.5 * honesty  # [0.5, 1.0]

    return base_weight * honesty_mult
```

---

## Implementation Plan

### Phase 1: Question Generation (from K3D Knowledge)
**File**: `knowledge3d/training/rlwhf/generate_questions_from_knowledge.py`

**Strategy**:
1. **From PDFs**: Extract section headers, definitions, key concepts
   - "What is backpropagation?" (from PDF section title)
   - "Explain gradient descent" (from PDF paragraph header)

2. **From WordNet**: Convert synsets to questions
   - 117K ready-made definition questions
   - "Define: neural_network.n.01"
   - "What is the meaning of machine_learning?"

3. **Cross-References**: Link related concepts
   - "How does backpropagation relate to gradient descent?"
   - "Compare supervised vs unsupervised learning"

4. **Difficulty Levels**: Easy → Hard
   - Easy: Definition questions (WordNet)
   - Medium: Explanation questions (PDF summaries)
   - Hard: Reasoning questions (cross-references)

**Output**: `questions_from_k3d_knowledge.jsonl`
```json
{
  "question": "What is backpropagation?",
  "source": "pdf:328/page:42",
  "context": "Backpropagation is an algorithm...",
  "difficulty": "easy",
  "trigrams": ["back", "prop", "agat", "gati", "atio", ...],
  "expected_answer_length": 50
}
```

---

### Phase 2: Student Attempt (TRM Baseline)
**File**: `knowledge3d/training/rlwhf/student_attempt_trm.py`

**Process**:
```python
def trm_student_attempt(question: str, rpn_engine, trm, weights) -> dict:
    """TRM attempts to answer question."""
    # 1. Embed question
    q_emb_128 = rpn_engine.embed_sentence(question)
    q_emb_512 = expand_embedding_to_trm(q_emb_128)

    # 2. TRM reasoning (6 recursions)
    y = np.zeros(512, dtype=np.float32)
    z = np.zeros(512, dtype=np.float32)
    y_out, z_out = trm.refine(q_emb_512, y, z,
                               weights['W1'], weights['W2'],
                               weights['W3'], weights['W4'],
                               n_steps=6)

    # 3. Compute answer metrics
    output_norm = np.linalg.norm(y_out)
    confidence = sigmoid(output_norm / 100.0)  # Normalize to [0, 1]

    return {
        'question': question,
        'answer_embedding': y_out,
        'output_norm': output_norm,
        'confidence': confidence,
        'converged': output_norm > 1.0
    }
```

---

### Phase 3: Teacher Evaluation (Ollama)
**File**: `knowledge3d/training/rlwhf/teacher_eval_ollama.py`

**Teacher Models** (with thinking tags):
- `deepseek-r1:latest` (reasoning specialist)
- `qwen2.5:14b` (multilingual, strong reasoning)
- `exaone3.5:latest` (honesty-focused)

**Prompt Template**:
```python
TEACHER_PROMPT = """You are an expert teacher evaluating a student AI's answer.

Question: {question}

Ground Truth Context (from K3D knowledge base):
{context}

Student's Answer (embedding-based):
- Output norm: {output_norm:.2f}
- Confidence: {confidence:.2%}
- Converged: {converged}

Your task:
1. <think> Analyze the student's reasoning process. What did they try to do?
2. <think> Compare to ground truth. Is it accurate? Partially correct? Wrong?
3. <think> Assess honesty. Did they admit uncertainty or hallucinate?
4. Rate the answer: 'good', 'partial', 'bad', or 'dishonest'
5. Provide the CORRECT answer if the student was wrong
6. Give specific feedback on how to improve

Format your response as:
<think>
[Your reasoning about the student's attempt]
</think>

Rating: [good|partial|bad|dishonest]
Correct Answer: [if student was wrong]
Feedback: [specific improvements]
"""
```

**Execution**:
```python
def evaluate_with_teacher(student_attempt: dict,
                         question_data: dict,
                         ollama_url: str,
                         model: str = "deepseek-r1:latest") -> dict:
    """Get teacher evaluation via Ollama."""
    prompt = TEACHER_PROMPT.format(
        question=question_data['question'],
        context=question_data['context'],
        output_norm=student_attempt['output_norm'],
        confidence=student_attempt['confidence'],
        converged=student_attempt['converged']
    )

    # Call Ollama
    response = ollama_generate(ollama_url, model, prompt)

    # Parse thinking tags
    thinking_parser = ThinkingTagsParser()
    analysis = thinking_parser.parse_and_analyze(response)

    # Extract rating
    rating = extract_rating(response)  # good/partial/bad/dishonest

    # Extract corrected answer
    corrected = extract_corrected_answer(response)

    # Extract feedback
    feedback = extract_feedback(response)

    return {
        'rating': rating,
        'corrected_answer': corrected,
        'feedback': feedback,
        'thinking_analysis': analysis,
        'thinking_segments': [seg.content for seg in analysis.segments],
        'honesty_score': analysis.overall_honesty,
        'reasoning_depth': analysis.reasoning_depth
    }
```

---

### Phase 4: Thinking Tag Harvesting
**File**: `knowledge3d/training/rlwhf/harvest_thinking_tags.py`

**Process**:
```python
def harvest_thinking_embeddings(teacher_response: dict,
                               rpn_engine) -> np.ndarray:
    """
    Convert teacher's thinking tags to TRM-trainable embeddings.

    This is KEY: We're teaching TRM to REASON like the teacher!
    """
    thinking_segments = teacher_response['thinking_segments']

    # Embed each thinking segment
    thinking_embs = []
    for segment in thinking_segments:
        emb_128 = rpn_engine.embed_sentence(segment)
        emb_512 = expand_embedding_to_trm(emb_128)
        thinking_embs.append(emb_512)

    # Aggregate into single reasoning pattern
    if not thinking_embs:
        return np.zeros(512, dtype=np.float32)

    # Weighted average (later segments = more refined reasoning)
    weights = np.linspace(0.5, 1.0, len(thinking_embs))
    weights /= weights.sum()

    thinking_pattern = np.average(thinking_embs, axis=0, weights=weights)

    return thinking_pattern.astype(np.float32)
```

**What This Teaches TRM**:
- **Reasoning chains**: "Because X, therefore Y"
- **Uncertainty modeling**: "I'm not sure, but..."
- **Self-correction**: "Wait, actually..."
- **Question decomposition**: "First, let's consider..."

---

### Phase 5: Dataset Construction
**File**: `knowledge3d/training/rlwhf/build_semantic_reasoning_dataset.py`

**Output Format** (`rlwhf_semantic_reasoning.npz`):
```python
{
    'questions': np.ndarray,      # (N, 512) - Question embeddings
    'student_answers': np.ndarray, # (N, 512) - TRM baseline attempts
    'target_answers': np.ndarray,  # (N, 512) - Teacher-corrected
    'thinking_patterns': np.ndarray, # (N, 512) - Reasoning embeddings
    'rewards': np.ndarray,         # (N,) - Training rewards [-2, +2]
    'honesty_scores': np.ndarray,  # (N,) - Honesty [0, 1]
    'difficulty': np.ndarray,      # (N,) - Question difficulty [1, 3]
    'metadata': {
        'n_samples': int,
        'teacher_model': str,
        'creation_date': str,
        'k3d_knowledge_version': str
    }
}
```

---

### Phase 6: Reward-Weighted TRM Training
**File**: `scripts/train_trm_semantic_rlwhf.py`

**Training Loop**:
```python
def train_trm_with_rlwhf(dataset_path: Path,
                         epochs: int = 100,
                         batch_size: int = 128) -> dict:
    """
    Train TRM on semantic reasoning with RLWHF.
    """
    # Load dataset
    data = np.load(dataset_path)
    questions = data['questions']
    targets = data['target_answers']
    thinking = data['thinking_patterns']
    rewards = data['rewards']
    honesty = data['honesty_scores']

    # Convert to PyTorch
    ds = TensorDataset(
        torch.from_numpy(questions),
        torch.from_numpy(targets),
        torch.from_numpy(thinking),
        torch.from_numpy(rewards),
        torch.from_numpy(honesty)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Initialize TRM
    W1, W2, W3, W4 = load_initial_weights(trm_weights_path)
    trm = TRMPyTorch(W1, W2, W3, W4).to(device)
    optimizer = torch.optim.AdamW(trm.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0

        for q, target, think, reward, hon in loader:
            q, target, think = q.to(device), target.to(device), think.to(device)
            reward, hon = reward.to(device), hon.to(device)

            # Forward pass (6 Tesla recursions)
            predicted = trm(q, n_steps=6)

            # Multi-objective loss
            loss_answer = F.mse_loss(predicted, target, reduction='none').mean(dim=1)
            loss_thinking = F.mse_loss(predicted, think, reduction='none').mean(dim=1)
            loss_combined = loss_answer + 0.3 * loss_thinking

            # Reward weighting (KEY: prioritize high-quality examples)
            weights = reward_to_weight(reward, hon)
            loss_weighted = (loss_combined * weights).mean()

            # Backward pass
            optimizer.zero_grad()
            loss_weighted.backward()
            torch.nn.utils.clip_grad_norm_(trm.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss_weighted)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")

    # Save trained weights
    save_weights(output_path, trm.W1, trm.W2, trm.W3, trm.W4)

    return {'epochs': epochs, 'final_loss': epoch_loss/len(loader)}
```

---

## Expected Results

### After RLWHF Training (100-200 epochs):

| Metric | Baseline (RPN-init) | After ARC | After RLWHF | Improvement |
|--------|-------------------|-----------|-------------|-------------|
| **Semantic Activation** | 0.29 | 0.29 | **0.6-0.7** | **100-140%** |
| **Semantic Clustering** | 0.088 (8.8%) | 0.088 | **0.5-0.6 (50-60%)** | **570-680%** |
| **Domain Transfer** | 0.062 (6.2%) | 0.062 | **0.4-0.5 (40-50%)** | **650-800%** |
| **Honesty Score** | N/A | N/A | **0.7-0.8** | **NEW** |
| **Reasoning Depth** | Random | Grid-focused | **Semantic chains** | **NEW** |

### What TRM Will Learn:

1. **Answer Accuracy**: Transform question embeddings → correct answer embeddings
2. **Reasoning Patterns**: Embed reasoning chains from teacher's thinking tags
3. **Honesty Modeling**: Express uncertainty when knowledge is insufficient
4. **Grounding**: Activate relevant trigrams from Galaxy/House (290K vocab)
5. **Quality Preference**: Prioritize high-reward examples (aimed training)

---

## Validation Strategy

### Test Set:
- 1,000 questions held out from dataset
- Mix of easy/medium/hard difficulty
- Covers all topics in K3D knowledge base

### Metrics:
1. **Answer Accuracy**: Cosine similarity to ground truth
2. **Knowledge Activation**: How well TRM accesses Galaxy/House
3. **Honesty**: Does TRM admit "I don't know" when appropriate?
4. **Reasoning Depth**: Number of thinking steps (inferred from output)
5. **Reward Distribution**: Average reward on test set

### Comparison:
- Baseline (RPN-init)
- After ARC training (grid reasoning)
- After RLWHF training (semantic reasoning)

**Expected**:
- RLWHF >> ARC on semantic tasks (50-100× improvement)
- ARC >> RLWHF on grid tasks (domain-specific training)
- Both >> Baseline (proof that training works)

---

## Timeline Estimate

| Phase | Duration | Artifact |
|-------|----------|----------|
| 1. Question Generation | 1-2 hours | 10K-50K questions from K3D |
| 2. Student Baseline | 30 min | TRM attempts on all questions |
| 3. Teacher Evaluation | 3-6 hours | Ollama feedback (depends on # questions) |
| 4. Thinking Harvesting | 15 min | Extract & embed thinking tags |
| 5. Dataset Construction | 15 min | Build training dataset |
| 6. RLWHF Training | 10-20 sec | Train TRM (100-200 epochs) |
| 7. Validation | 5 min | Test on held-out set |
| **TOTAL** | **5-9 hours** | **Chat-capable K3D** |

**Bottleneck**: Teacher evaluation (Ollama inference)
**Optimization**: Run multiple Ollama models in parallel, batch processing

---

## Key Advantages of K3D RLWHF

### 1. **Accuracy over Volume**
- Traditional: Train on all data equally
- K3D RLWHF: Train more on high-quality examples (reward weighting)

### 2. **Reasoning Transfer**
- Traditional: Learn input-output mapping
- K3D RLWHF: Learn HOW to reason (thinking tag embeddings)

### 3. **Honesty Built-In**
- Traditional: Hallucinations are common
- K3D RLWHF: Reward honesty, penalize fabrication

### 4. **Knowledge Grounding**
- Traditional: Knowledge in weights (opaque)
- K3D RLWHF: Knowledge in embeddings (inspectable, updatable)

### 5. **Fast Iteration**
- Traditional: Hours/days to retrain
- K3D RLWHF: 10-20 seconds to train TRM on new feedback

---

## Next Steps

1. **Implement Phase 1**: Question generation from K3D knowledge
2. **Run Phase 2**: TRM baseline attempts
3. **Set up Ollama**: Install thinking models (deepseek-r1, qwen2.5)
4. **Run Phase 3-4**: Teacher evaluation + thinking harvesting
5. **Execute Phase 5-6**: Build dataset + train TRM
6. **Validate**: Test semantic Q&A capability
7. **Iterate**: Collect user feedback → Re-train with RLWHF

---

## Success Criteria

**K3D is chat-capable when**:
- ✅ Can answer questions about ingested PDFs (semantic activation >0.5)
- ✅ Admits "I don't know" when knowledge is missing (honesty >0.7)
- ✅ Reasons through multi-step queries (thinking depth >3)
- ✅ Outperforms baseline by 50-100× on semantic tasks
- ✅ Training completes in <1 minute (fast iteration)

**User Experience**:
```python
>>> k3d = K3D()
>>> k3d.chat("What is backpropagation?")
"Backpropagation is an algorithm for training neural networks by
computing gradients via the chain rule... (sources: PDF_328_p42)"

>>> k3d.chat("How does quantum computing work?")
"I don't know - this topic isn't in my current knowledge base.
You might want to ingest PDFs on quantum computing first."
```

---

**Status**: Design complete, ready for implementation
**Paradigm Validated**: ARC training proved TRM can learn (62,000× improvement)
**Next**: Build RLWHF pipeline for semantic reasoning

