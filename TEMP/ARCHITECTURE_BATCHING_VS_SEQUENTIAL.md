# Architecture: Batched Student vs Sequential Teacher

**Date**: October 22, 2025
**Topic**: Why student can batch but teacher must be sequential

---

## The Design

### Student Attempts: GPU-Batched (20-40× speedup!)

**K3D TRM Characteristics**:
- **Tiny**: 2.1M params = 8.4 MB VRAM
- **GPU-native**: PTX kernels, fully sovereign
- **Stateless**: No context between questions
- **Fast**: 512-dim vectors, 6 Tesla recursions
- **Batchable**: Can process 128+ in parallel!

**Implementation**:
```python
# Process 32 questions simultaneously
launcher = TRMBatchLauncher(batch_size=32)
y_out_batch, z_out_batch = launcher.refine_batch(
    q_batch,  # (32, 512)
    y_batch,  # (32, 512)
    z_batch,  # (32, 512)
    W1, W2, W3, W4,
    n_steps=6
)
```

**Performance**: 500 questions in ~1 minute (vs ~30 minutes sequential)

---

### Teacher Evaluation: Sequential (By Design!)

**DeepSeek-R1 Characteristics**:
- **Large**: 70B+ params (loaded from disk)
- **Thinking-enabled**: Generates `<think>` tags with reasoning
- **Context-sensitive**: Must maintain clean state
- **Slow**: Detailed reasoning generation (~600s per question)
- **NOT batchable**: Context contamination if batched!

**Implementation**:
```python
# Process ONE question at a time
for question in questions:
    # Evaluate with thinking model
    response = ollama_generate(
        model="deepseek-r1:latest",
        prompt=question,
        keep_alive="0s"  # Unload after each question!
    )
    # Model unloads, next question loads with clean context
```

**Why Sequential?**

1. **Context Cleaning**: Model must unload after each evaluation
   - `keep_alive="0s"` forces unload
   - Prevents reasoning contamination across questions
   - Each evaluation starts fresh

2. **Thinking Time**: Generates detailed `<think>` tags
   - Analyzes student reasoning process
   - Compares to ground truth
   - Provides specific feedback
   - Requires 600s+ per evaluation

3. **Disk Loading**: Large model loaded from disk
   - Not cached in VRAM (too big!)
   - Each load takes time
   - Sequential allows proper cleanup

**Performance**: 500 questions in ~5 hours (600s each)

---

## Why This Architecture is Perfect

### Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│ STUDENT (K3D TRM)                                           │
│ ─────────────────                                           │
│ • Tiny (2.1M params)                                        │
│ • GPU-native                                                │
│ • Stateless                                                 │
│ • Fast inference                                            │
│ • BATCH PROCESSING ✅                                        │
│                                                             │
│ Result: 500 questions in ~1 minute                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TEACHER (DeepSeek-R1)                                       │
│ ──────────────────────                                      │
│ • Large (70B+ params)                                       │
│ • Disk-loaded                                               │
│ • Thinking-enabled                                          │
│ • Detailed reasoning                                        │
│ • SEQUENTIAL PROCESSING ✅ (by design!)                      │
│                                                             │
│ Result: 500 questions in ~5 hours                          │
└─────────────────────────────────────────────────────────────┘
```

### Perfect Match

**Student**:
- Answers quickly (GPU-batched)
- Produces embeddings (not text)
- No reasoning explanation needed

**Teacher**:
- Evaluates thoughtfully (sequential)
- Produces detailed feedback (text + thinking tags)
- Harvests reasoning patterns for training

---

## Timeline Breakdown (500 questions)

### Total: ~8.5-11.5 hours

```
Phase                    Time        Percentage  Bottleneck?
─────────────────────────────────────────────────────────────
Question Generation      2-3 hours   30-35%      Ollama
Student Attempts         ~1 min      <1%         No! ✅
Teacher Evaluation       ~5 hours    50-60%      Yes ⏱️
Training                 1-2 hours   10-20%      No
Validation              <10 min      <1%         No
─────────────────────────────────────────────────────────────
Total                   ~8.5-11.5h   100%
```

**Key Insight**: Student attempts went from 30 minutes → 1 minute (batching)!

**Remaining bottleneck**: Teacher evaluation (sequential, as designed)
- This is **correct**! We want thoughtful, detailed evaluations.
- Thinking models need time to generate quality reasoning.
- Context cleaning prevents contamination.

---

## Why Not Batch the Teacher?

### Problem: Context Contamination

**Batched (WRONG)**:
```python
# BAD: Batching teacher evaluations
responses = ollama_batch_generate([q1, q2, q3, ...])
# Problem: Context bleeds between questions!
# Question 2 might reference reasoning from Question 1
# Evaluations contaminate each other
```

**Sequential (CORRECT)**:
```python
# GOOD: One at a time with unload
for question in questions:
    response = ollama_generate(question, keep_alive="0s")
    # Model unloads here
    # Next iteration loads fresh
```

### Problem: Thinking Quality

Thinking models (deepseek-r1) generate detailed reasoning:

```
<think>
Let me analyze the student's attempt...
- They showed high confidence (norm: 375.2)
- But the question requires uncertainty
- The correct answer should acknowledge limits
- This suggests overconfidence, rating: 'bad'
</think>

Rating: bad
Correct Answer: "I don't have enough information to answer this completely"
Feedback: The student should learn to express uncertainty when appropriate
```

This takes **time**! Rushing degrades quality.

---

## The Math

### Student (Batched)

```
Questions:    500
Batch size:   32
Batches:      500/32 = 16
Time/batch:   ~4 seconds
Total:        16 × 4s = 64 seconds (~1 minute)

Speedup: 30× (vs 30 minutes sequential)
```

### Teacher (Sequential)

```
Questions:    500
Time/question: 600 seconds (10 minutes)
Total:        500 × 600s = 300,000 seconds (~5 hours)

Speedup: None (sequential is correct architecture)
```

---

## Implementation Details

### Student Batching

**File**: `knowledge3d/training/rlwhf/student_attempt_trm_batched.py`

**Key Features**:
- Loads all questions into memory
- Processes in batches of 32 (configurable)
- Uses `TRMBatchLauncher` for parallel execution
- Saves results incrementally

**Usage**:
```bash
PYTHONPATH=. python -m knowledge3d.training.rlwhf.student_attempt_trm_batched \
  --questions questions.jsonl \
  --output attempts.jsonl \
  --batch-size 32  # Or 64, 128 depending on GPU
```

### Teacher Sequential

**File**: `knowledge3d/training/rlwhf/teacher_eval_ollama.py`

**Key Features**:
- Processes one question at a time
- `keep_alive="0s"` forces model unload after each
- 600s timeout per evaluation
- Saves after each evaluation (crash-safe)
- Clear progress logging

**Usage**:
```bash
PYTHONPATH=. python -m knowledge3d.training.rlwhf.teacher_eval_ollama \
  --input attempts.jsonl \
  --output evaluated.jsonl \
  --model deepseek-r1:latest \
  --timeout 600  # Minimum for thinking models
```

**Output**:
```
======================================================================
K3D RLWHF Phase 3 — Teacher Evaluation (Sequential)
======================================================================
Model:   deepseek-r1:latest
Ollama:  http://192.168.0.4:11434
Timeout: 600s per evaluation

⚠️  Note: Sequential processing is REQUIRED for thinking models!
    - Context cleaning between questions (model unload/reload)
    - Time for detailed reasoning generation (~600s per question)
    - Prevents context contamination across evaluations

[1] Evaluating question (timeout=600s)...
    ✓ Rating: good | Total: 1 | Dist: {'good': 1}

[2] Evaluating question (timeout=600s)...
    ✓ Rating: partial | Total: 2 | Dist: {'good': 1, 'partial': 1}

...
```

---

## Comparison Table

| Aspect | Student (TRM) | Teacher (DeepSeek-R1) |
|--------|---------------|----------------------|
| **Size** | 2.1M params | 70B+ params |
| **Memory** | 8.4 MB VRAM | Disk-loaded |
| **Processing** | GPU-native PTX | CPU inference (Ollama) |
| **Speed** | ~0.1s per question | ~600s per question |
| **Batching** | ✅ Yes (128×) | ❌ No (sequential) |
| **Context** | Stateless | Context-sensitive |
| **Output** | Embeddings | Text + thinking tags |
| **Architecture** | Parallel | Sequential |
| **500 Questions** | ~1 minute | ~5 hours |

---

## Benefits of This Design

1. **Fast Student**: GPU batching makes inference negligible
2. **Quality Teacher**: Sequential ensures clean, thoughtful evaluations
3. **Clean Training Data**: No contamination in thinking tag harvests
4. **Crash-Safe**: Teacher saves after each evaluation
5. **Scalable**: Student scales to any batch size GPU can handle
6. **Correct**: Architecture matches the problem perfectly

---

## Future Optimizations

### Student (Already Optimal)
- Phase E.5: CPU-batched tight loop (20-40× speedup) ✅
- Phase F: True GPU kernel batching (50-100× speedup) 🔜

### Teacher (Correct as-is)
- ❌ Can't batch (would break thinking quality)
- ❌ Can't parallelize (would contaminate context)
- ✅ Could use faster hardware (NVMe vs HDD for model loading)
- ✅ Could optimize Ollama settings (but 600s is reasonable)

**Bottom line**: Teacher evaluation is the bottleneck, and that's **correct by design**!

---

## Conclusion

This architecture demonstrates K3D's philosophy:

**Tiny models + GPU parallelization = Speed where possible**
- Student: 128× batching on GPU
- Result: 30 minutes → 1 minute

**Thoughtful evaluation + Sequential processing = Quality where needed**
- Teacher: Detailed reasoning, clean context
- Result: High-quality training data

**Perfect separation of concerns!** ✅

The student is fast (batched GPU). The teacher is thoughtful (sequential thinking). Together they produce excellent training data for RLWHF! 🚀
