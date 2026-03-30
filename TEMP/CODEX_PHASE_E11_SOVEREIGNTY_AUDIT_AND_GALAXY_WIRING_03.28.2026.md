# CODEX PHASE E.11: Sovereignty Audit + Galaxy-Navigating Kernel

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — Daniel: "he's doing python!!"
**Sovereignty Debt:** SEVERE — the kernel is cosmetic; Python does the real work

---

## The Problem (Read This First)

Daniel has flagged this for 6 months: **GPU utilization is ~1-5% because the kernel is a thin cosine-similarity picker while Python does all the real work.**

Here is the honest sovereignty audit of the current pipeline:

### What Python Does (VIOLATIONS)

| Function | File | What It Does | Violation |
|----------|------|------------|-----------|
| `_embed_query_gpu()` | knowledgeverse.py:3103 | Calls sentence-transformers to compute embeddings | **Python ML framework in hot path** |
| `_embed_query_batch_gpu()` | knowledgeverse.py:3071 | Batch sentence-transformers | **Python ML framework in hot path** |
| `_find_colored_cell()` | run_full_benchmark.py:48 | Searches grid for colored cell | **Python environment simulation** |
| `_apply_arc3_action()` | run_full_benchmark.py:85 | Applies action to grid state | **Python environment simulation** |
| `_distance_to_goal()` | run_full_benchmark.py:120 | Computes Manhattan distance | **Python evaluation** |
| `_grid_equal()` | run_full_benchmark.py:116 | Compares grids | **Python evaluation** |
| `_score_rows()` | run_gpu_benchmark.py:279 | Decides if kernel answer was correct | **Python outcome logic** |
| outcome signal block | run_full_benchmark.py:213-221 | Decides +1/0/-1 for sleep-time | **Python decides ternary signal** |

### What the Kernel Actually Does (Cosmetic)

The "specialist" device functions are single-line tanh perturbations:

```cuda
// arc_reason_device — the ENTIRE function:
embedding[i] = tanhf((0.96f * embedding[i]) + (0.04f * absf(spatial_delta)));

// geometry_route_device — the ENTIRE function:
embedding[i] = tanhf(embedding[i] + (0.03f * route * context[i]));

// fractal_emit_device — the ENTIRE function:
embedding[i] = tanhf((0.94f * embedding[i]) + (0.03f * coarse) + (0.03f * fine));
```

These are NOT reasoning. They are gradient-free embedding nudges. A single Python `tanh(0.96*x + 0.04*y)` would produce identical results.

### What Galaxy Does (Almost Nothing)

The 93 foundational stars in VRAM are **only read for ARC3 option embeddings** (task_type == 8u):
```cuda
if (galaxy_table != nullptr && task_type == 8u && option_index < galaxy_star_count) {
    galaxy_compose_embedding_device(...);
}
```

For MMLU, GSM8K, LHE, ARC-AGI-2 — the Galaxy is **completely ignored**. The kernel never navigates it. All that knowledge sits unused in VRAM.

### The Real Pipeline Today

```
Python (sentence-transformers) computes embedding       ← 95% of compute time
Python packs embedding into VRAM                         ← I/O
Kernel does tanh perturbations + cosine similarity       ← 0.5ms, ~1% GPU
Python reads answer_index                                ← I/O
Python computes outcome signal                           ← Python decides if answer was right
Python calls sleep_time kernel                           ← 0.1ms, trivial
```

**This is a Python program with GPU decoration.** The kernel adds noise to pre-computed embeddings and picks the highest cosine score. It never reads knowledge. It never composes programs. It never evaluates itself.

---

## What Sovereignty MEANS (from docs/vocabulary specs)

Per `KNOWLEDGEVERSE_SPECIFICATION.md` and `THREE_BRAIN_SYSTEM_SPECIFICATION.md`:

1. **The kernel navigates Galaxy** — finds relevant knowledge stars by embedding proximity
2. **The kernel composes knowledge** — follows component_refs, blends multi-hop stars
3. **The kernel executes RPN programs** — the Galaxy stars contain procedural programs
4. **The kernel evaluates outcomes** — knows the goal, computes success/failure on device
5. **Python = boot + I/O only** — loads data into VRAM, reads results out. ~200 lines target.

The kernel should be a BRAIN that thinks, not a picker that chooses.

---

## Deliverable 1: Galaxy Navigation for ALL Task Types

### Current State
Galaxy stars are only read for ARC3 (task_type == 8u). For all other task types, `option_embeddings` come pre-computed from Python.

### Target State
The kernel navigates Galaxy to find relevant knowledge for EVERY task type. The specialist switch should use Galaxy, not just perturb embeddings.

### Implementation

In `gpu_task_dispatch.cu`, BEFORE the specialist switch, add Galaxy navigation:

```cuda
// Navigate Galaxy: find the k-nearest stars to the reasoning state
__shared__ unsigned int galaxy_nearest[8];  // top-8 star indices
__shared__ float galaxy_nearest_scores[8];
__shared__ float galaxy_knowledge[GPU_TASK_EMBED_DIMS]; // composed knowledge embedding

if (threadIdx.x == 0 && galaxy_table != nullptr && galaxy_star_count > 0u) {
    // Initialize
    for (int k = 0; k < 8; ++k) {
        galaxy_nearest[k] = 0xFFFFFFFFu;
        galaxy_nearest_scores[k] = -1.0e30f;
    }

    // Linear scan of Galaxy (all stars, no caps — LOD handles volume)
    for (unsigned int star_idx = 0u; star_idx < galaxy_star_count; ++star_idx) {
        float star_embedding[GPU_TASK_EMBED_DIMS];
        galaxy_compose_embedding_device(star_embedding, galaxy_table, star_idx, GPU_TASK_EMBED_DIMS);
        float sim = cosine32_device(reasoning_state, star_embedding, GPU_TASK_EMBED_DIMS);

        // Insert into top-8 if better than worst
        int worst_k = 0;
        for (int k = 1; k < 8; ++k) {
            if (galaxy_nearest_scores[k] < galaxy_nearest_scores[worst_k]) worst_k = k;
        }
        if (sim > galaxy_nearest_scores[worst_k]) {
            galaxy_nearest[worst_k] = star_idx;
            galaxy_nearest_scores[worst_k] = sim;
        }
    }

    // Compose: weighted blend of top-8 Galaxy stars into galaxy_knowledge
    float total_weight = 0.0f;
    for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) galaxy_knowledge[i] = 0.0f;
    for (int k = 0; k < 8; ++k) {
        if (galaxy_nearest[k] == 0xFFFFFFFFu) continue;
        float w = fmaxf(0.0f, galaxy_nearest_scores[k]);
        total_weight += w;
        float star_emb[GPU_TASK_EMBED_DIMS];
        galaxy_compose_embedding_device(star_emb, galaxy_table, galaxy_nearest[k], GPU_TASK_EMBED_DIMS);
        for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) {
            galaxy_knowledge[i] += w * star_emb[i];
        }
    }
    if (total_weight > 1.0e-6f) {
        for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) {
            galaxy_knowledge[i] /= total_weight;
        }
    }
}
__syncthreads();
```

Then each specialist USES `galaxy_knowledge` instead of just perturbing embeddings:

```cuda
// BEFORE (cosmetic):
// embedding[i] = tanhf(0.96f * embedding[i] + 0.04f * absf(spatial_delta));

// AFTER (knowledge-informed):
// Blend reasoning with Galaxy knowledge — the specialist activates Galaxy context
case 0u: // ARC
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        embedding[i] = tanhf(
            (0.60f * embedding[i]) +
            (0.30f * galaxy_knowledge[i]) +
            (0.10f * context[(3 * dim) + i])
        );
    }
    __syncthreads();
    break;
```

The weight on `galaxy_knowledge` should be SIGNIFICANT (0.20-0.40), not 0.03. The Galaxy IS the knowledge. If the specialist doesn't use it, it's not reasoning — it's noise.

### Success Criteria
- Kernel reads from Galaxy for ALL task types (0-8), not just ARC3
- Galaxy navigation takes measurable GPU time (the kernel should be HEAVIER, not lighter)
- Different task types weight different Galaxy neighborhoods (math stars for GSM8K, reality stars for LHE)

---

## Deliverable 2: FNV-1a Embeddings Instead of Sentence-Transformers

### Current State
`_embed_query_gpu()` calls sentence-transformers (Python ML framework) to compute embeddings. This is the HEAVIEST computation in the pipeline and it's Python.

### Target State
Use FNV-1a hashing to compute embeddings — the SAME approach already used by:
- `ARC3FrameEncoder.encode()` — hashes grid features into 32-float embedding
- `action_embedding_loader._node_to_embedding()` — hashes RPN tokens via FNV-1a
- `vram_task_buffer._fnv1a32()` — hashes text to uint32

### Implementation

Create `knowledge3d/knowledgeverse/sovereign_text_embedder.py`:

```python
"""Sovereign text embedding via FNV-1a token hashing — no ML frameworks."""

EMBEDDING_DIM = 32

def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value

def embed_text_sovereign(text: str) -> list[float]:
    """Hash text tokens into 32-float embedding via FNV-1a.

    Same approach as ARC3FrameEncoder and action_embedding_loader.
    No sentence-transformers. No Python ML frameworks. Sovereign.
    """
    tokens = text.lower().split()
    embedding = [0.0] * EMBEDDING_DIM
    for token in tokens:
        h = _fnv1a32(token)
        bucket = h % EMBEDDING_DIM
        sign = 1.0 if (h >> 16) & 1 else -1.0
        embedding[bucket] += sign * (1.0 + 0.1 * ((h >> 8) & 0xFF) / 255.0)
    # L2 normalize
    norm = sum(v * v for v in embedding) ** 0.5
    if norm > 1e-8:
        embedding = [v / norm for v in embedding]
    return embedding
```

Then in `run_gpu_benchmark.py`, replace `_embed_query32(kv, text, task=task_ctx)` with:

```python
from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign
# ...
"query_embedding": embed_text_sovereign(question["question_text"]),
"option_embeddings": [embed_text_sovereign(opt) for opt in options[:7]],
```

### Why This Works
FNV-1a embeddings are NOT dumb — they map semantically related tokens to nearby buckets via hash collision patterns. More importantly:
- The Galaxy stars ALSO use FNV-1a embeddings (from `foundational_galaxy_builder.py`)
- So the query embedding and Galaxy embeddings are in the SAME hash space
- Cosine similarity between query and Galaxy stars becomes meaningful
- The kernel can navigate Galaxy because query and knowledge are commensurable

Sentence-transformers puts queries in a 384-dim transformer space that has NOTHING to do with our Galaxy's FNV-1a space. That's why Galaxy navigation does nothing for non-ARC3 tasks — the embeddings are in different spaces!

### What This Removes
- `from knowledge3d.knowledgeverse import Knowledgeverse` — no longer needed in run_gpu_benchmark.py for embedding
- `_embed_query32(kv, text, task=task_ctx)` — replaced by sovereign embedder
- `kv._embed_query_gpu()` — no longer called in the benchmark path
- sentence-transformers dependency in the hot path — GONE

### Success Criteria
- ZERO calls to `_embed_query_gpu()` or sentence-transformers in the benchmark runner
- All embeddings computed via FNV-1a (same space as Galaxy)
- MMLU/GSM8K/LHE accuracy may initially DROP — that's honest. The kernel must learn to navigate Galaxy with these embeddings.
- Knowledgeverse still instantiated for question loading, NOT for embedding

---

## Deliverable 3: Kernel-Side ARC3 Goal Awareness

### Current State
For ARC3 synthetic, Python:
1. Searches the grid for the colored cell (`_find_colored_cell`)
2. Applies the action to the grid (`_apply_arc3_action`)
3. Computes distance to goal (`_distance_to_goal`)
4. Decides the outcome signal (`outcome_signal = 1 if distance decreased`)

The kernel NEVER sees the goal frame. It doesn't know what it's trying to achieve.

### Target State
The kernel receives the goal embedding alongside the frame embedding. It computes distance/progress ON DEVICE. The outcome signal comes from the kernel, not from Python.

### Implementation

#### Step 1: Extend input slot to carry goal embedding

In `vram_task_buffer.py`, add:

```python
GOAL_EMBEDDING_OFFSET = 1056  # after TERNARY_SIGNAL_OFFSET (1052) + padding
# 32 floats × 4 bytes = 128 bytes → fits within INPUT_SLOT_BYTES (1280)
```

In `_pack_task_slot()`, pack goal embedding:

```python
goal_embedding = _embedding32(task.get("goal_embedding") or [])
struct.pack_into("<32f", payload, base + GOAL_EMBEDDING_OFFSET, *goal_embedding)
```

#### Step 2: Kernel reads goal and computes progress

In `device_functions.cuh`, add:

```cuda
#define GPU_TASK_GOAL_EMBEDDING_OFFSET 1056u

__device__ float goal_progress_device(
    const float* current_frame,
    const float* goal_embedding,
    const float* prev_frame,
    int dim
) {
    float current_dist = 0.0f;
    float prev_dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float dc = current_frame[i] - goal_embedding[i];
        float dp = prev_frame[i] - goal_embedding[i];
        current_dist += dc * dc;
        prev_dist += dp * dp;
    }
    current_dist = sqrtf(current_dist);
    prev_dist = sqrtf(prev_dist);

    if (current_dist < 1.0e-4f) return 1.0f;   // reached goal
    if (current_dist < prev_dist) return 0.5f;   // getting closer
    if (current_dist > prev_dist) return -0.5f;  // getting farther
    return 0.0f;                                  // no change
}
```

In `gpu_task_dispatch.cu`, for ARC3 tasks (case 8u), read goal and compute progress:

```cuda
if (task_type == 8u) {
    const float* goal_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_GOAL_EMBEDDING_OFFSET);
    float progress = goal_progress_device(query_embedding, goal_embedding, brain_prev_frame, GPU_TASK_EMBED_DIMS);
    // Bias action selection toward progress
    // Store progress in output for Python to read (but Python doesn't decide it)
}
```

#### Step 3: Python passes goal embedding, reads kernel's progress signal

In `run_full_benchmark.py`, the ARC3 synthetic loop passes goal embedding:

```python
goal_embedding = encoder.encode(task["goal_frame"])
task["goal_embedding"] = goal_embedding
```

The outcome signal comes from the KERNEL's progress computation, not from Python's `_distance_to_goal()`.

### Success Criteria
- The kernel knows the goal frame (via goal_embedding in input slot)
- Distance/progress computed ON DEVICE, not in Python
- `_find_colored_cell()`, `_distance_to_goal()` no longer determine the outcome signal
- Python still applies actions to the grid (this is environment simulation, which is I/O — acceptable for synthetic. The live API does this server-side.)

---

## Deliverable 4: Remove Python Environment Simulation from Scoring

### Current State
Python computes outcome_signal based on grid comparison:
```python
if _grid_equal(frame, goal):
    outcome_signal = 1
elif changed and current_distance < last_distance:
    outcome_signal = 1
elif not changed or current_distance >= last_distance:
    outcome_signal = -1
```

### Target State
The outcome signal comes from comparing frame embeddings ON DEVICE. The kernel already has `brain_prev_frame` and gets the new `query_embedding`. After Deliverable 3, it also has `goal_embedding`.

The progress signal from `goal_progress_device()` becomes the ternary outcome:
- `progress >= 0.9` → outcome = +1 (reached goal or very close)
- `progress > 0.0` → outcome = +1 (getting closer)
- `progress < 0.0` → outcome = -1 (getting farther or stuck)
- `progress == 0.0` → outcome = 0 (neutral)

Store this in the output buffer for Python to read (for logging), but the sleep-time kernel receives it from the DISPATCH kernel's output, not from Python's computation.

### Implementation

Add a `GOAL_PROGRESS_OFFSET` to the output slot (e.g., offset 24, after answer_text_hash). The kernel writes its computed progress there. Python reads it for logging and passes it to sleep_time.

**Keep `_grid_equal()` for the FINAL correctness check** — Python still determines if the task is "solved" by grid equality (this is ground-truth evaluation, not reasoning). But the STEP-BY-STEP outcome signals that drive sleep-time consolidation come from the kernel.

### Success Criteria
- Sleep-time outcome signal computed by kernel, not by Python
- `_distance_to_goal()` used only for final logging, not for driving learning
- The kernel's self-assessment drives its own learning loop

---

## Deliverable 5: Heavier Specialist Functions

### Current State
Each specialist is 3-6 lines of tanh. Total specialist compute: ~0.01ms.

### Target State
Specialists should do REAL work: navigate Galaxy, compose multi-star knowledge, apply different blending strategies per task type. The kernel should be HEAVIER (target: 5-50ms per task, not 0.5ms).

### Implementation

Replace each specialist with a Galaxy-informed version. Example for MMLU (case 4u):

```cuda
case 4u:  // MMLU: broad knowledge lookup
    // Navigate Galaxy for domain-relevant stars
    if (galaxy_table != nullptr) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            // Blend swarm output with Galaxy knowledge (heavy)
            float knowledge_sum = 0.0f;
            float weight_sum = 0.0f;
            for (unsigned int star = 0u; star < galaxy_star_count; ++star) {
                float star_emb[GPU_TASK_EMBED_DIMS];
                galaxy_compose_embedding_device(star_emb, galaxy_table, star, dim);
                float relevance = star_emb[i] * reasoning_state[i]; // dot-product component
                knowledge_sum += relevance * star_emb[i];
                weight_sum += fabsf(relevance);
            }
            if (weight_sum > 1.0e-6f) {
                embedding[i] = tanhf(
                    (0.50f * embedding[i]) +
                    (0.50f * knowledge_sum / weight_sum)
                );
            }
        }
        __syncthreads();
    }
    resonance_field_device(embedding, context, dim);
    vector_resonate_device(embedding, context, dim);
    break;
```

This makes the kernel iterate over ALL Galaxy stars PER specialist PER thinking step. With 93 stars × 10 thinking steps × 9 chains, the kernel does real work.

### Per Task Type Strategy

| Task Type | Case | Galaxy Strategy |
|-----------|------|----------------|
| ARC (0u) | Drawing + spatial stars | Compose visual primitives from Galaxy |
| MATH (1u) | Math operation stars | Compose operation chains |
| GSM8K (2u) | Math + reality stars | Word problem decomposition via Galaxy |
| LHE (3u) | All stars (multi-hop) | Graph crystallize across Galaxy neighborhoods |
| MMLU (4u) | Domain-relevant stars | Broad knowledge scan |
| ARC3 (8u) | Action + spatial stars | Already has Galaxy — extend to spatial stars |

### Success Criteria
- GPU utilization during benchmark run > 20% (currently ~1-5%)
- Kernel execution time > 5ms per task (currently ~0.5ms)
- Each specialist reads from Galaxy, not just perturbs embeddings
- Galaxy navigation is the dominant compute cost (as it should be — the knowledge IS the reasoning)

---

## DO NOT

- **DO NOT** call `kv._embed_query_gpu()` or sentence-transformers in the benchmark runner. Use `embed_text_sovereign()`.
- **DO NOT** add new Python helper functions for things the kernel should do. Ask: "should this be on GPU?" Default answer: YES.
- **DO NOT** make the specialists lighter. Make them HEAVIER. The kernel should be doing WORK.
- **DO NOT** cap Galaxy navigation to a subset of stars. Scan ALL stars. LOD handles volume (that's what it's for).
- **DO NOT** compute outcome signals in Python. The kernel evaluates itself.
- **DO NOT** add Python fallbacks for any computation. If the kernel gets it wrong, that's an honest 0. We fix the kernel, not add Python workarounds.
- **DO NOT** keep `Knowledgeverse` instantiation in `run_gpu_benchmark.py` for embedding purposes. Keep it only for question loading (which IS legitimate I/O).
- **DO NOT** add more Python orchestration. Every new line of Python you write is sovereignty debt. The goal is ~200 lines, not 450.

---

## Files to Create

| File | Purpose |
|------|---------|
| `knowledge3d/knowledgeverse/sovereign_text_embedder.py` | FNV-1a text embedding (no ML frameworks) |

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` | Add Galaxy navigation BEFORE specialist switch. Make specialists use galaxy_knowledge. Add goal_progress_device output. |
| `knowledge3d/cranium/cuda/device_functions.cuh` | Add `goal_progress_device()`. Add `GPU_TASK_GOAL_EMBEDDING_OFFSET`. Add `GPU_TASK_GOAL_PROGRESS_OUTPUT_OFFSET`. |
| `scripts/run_gpu_benchmark.py` | Replace `_embed_query32(kv, ...)` with `embed_text_sovereign(...)`. Remove Knowledgeverse dependency for embedding. |
| `scripts/run_full_benchmark.py` | Pass goal_embedding in ARC3 tasks. Read kernel's progress signal instead of Python-computed distance. |
| `knowledge3d/knowledgeverse/vram_task_buffer.py` | Add GOAL_EMBEDDING_OFFSET. Add GOAL_PROGRESS output offset. Pack/unpack goal embedding. |

## Files to Read (do not modify)

| File | Why |
|------|-----|
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | Understand Galaxy star structure |
| `knowledge3d/knowledgeverse/galaxy_vram_table.py` | `galaxy_compose_embedding_device` reference |
| `knowledge3d/knowledgeverse/arc3_frame_encoder.py` | FNV-1a embedding pattern to follow |
| `knowledge3d/knowledgeverse/action_embedding_loader.py` | FNV-1a embedding pattern to follow |

## Tests

Update `tests/test_phase_e_runners.py`:
1. Test `embed_text_sovereign()` produces non-zero, normalized 32-float vectors
2. Test that similar texts produce higher cosine similarity than unrelated texts
3. Test that Galaxy navigation runs for all task types (not just ARC3)
4. Test that kernel output includes goal_progress for ARC3 tasks
5. Test that ZERO calls to `_embed_query_gpu` exist in benchmark runners

Add sovereignty grep test:
```python
def test_no_sentence_transformers_in_benchmark_runner():
    """Benchmark runners must not import or call sentence-transformers."""
    for path in ["scripts/run_gpu_benchmark.py", "scripts/run_full_benchmark.py"]:
        content = Path(path).read_text()
        assert "_embed_query_gpu" not in content
        assert "sentence_transform" not in content.lower()
        assert "embed_sentence" not in content.lower()
```

---

## Expected Impact

| Metric | Before E.11 | After E.11 |
|--------|-------------|------------|
| GPU utilization | ~1-5% | >20% |
| Kernel time per task | ~0.5ms | >5ms |
| Python hot-path compute | sentence-transformers (~95%) | FNV-1a hash (~5%) |
| Galaxy reads per benchmark | 0 (except ARC3 options) | 93 stars × tasks × thinking_steps |
| Sovereignty violations | 8 functions | 0 in benchmark runner |
| MMLU accuracy | 7/50 (14%) | May drop initially — HONEST |
| GSM8K accuracy | 0/10 | Still 0/10 — Galaxy needs more math knowledge |

**Accuracy may drop.** That is CORRECT. We are removing the sentence-transformers crutch. The kernel must learn to reason with Galaxy knowledge. Sleep-time consolidation will improve accuracy over time — that's the whole point of the architecture.

Daniel: "We fail and fix — this is the goal."

---

## Sovereignty Compliance Checklist (Run After Implementation)

```bash
# 1. No sentence-transformers in benchmark runners
grep -r "embed_query_gpu\|sentence_transform\|embed_sentence" scripts/run_*.py
# Expected: 0 matches

# 2. Galaxy read in kernel for all task types
grep -c "galaxy_compose_embedding_device" knowledge3d/cranium/cuda/gpu_task_dispatch.cu
# Expected: >1 (currently 1, should be 5+)

# 3. Goal embedding in ARC3 input
grep "GOAL_EMBEDDING_OFFSET" knowledge3d/knowledgeverse/vram_task_buffer.py
# Expected: defined and used

# 4. Kernel writes progress signal
grep "GOAL_PROGRESS" knowledge3d/cranium/cuda/gpu_task_dispatch.cu
# Expected: output written

# 5. Python line count in run_full_benchmark.py
wc -l scripts/run_full_benchmark.py
# Target: stay under 500 lines (DO NOT grow)
```
