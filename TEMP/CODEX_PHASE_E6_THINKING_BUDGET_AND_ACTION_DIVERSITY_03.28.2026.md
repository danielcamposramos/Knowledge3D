# Codex — Phase E.6: Thinking Budget + Action Diversity Fix

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.5 DONE. Live ARC-AGI-3 ran 8 games. Pipeline proven end-to-end. **0 levels completed — ACTION5 fixation must be fixed.**

---

## DIAGNOSIS: WHY ARC-AGI-3 SCORES ZERO

### Problem 1: No Thinking Budget (GPU idle)
The kernel runs the swarm for 3 iterations, then does ONE pass through specialists and cosine scoring. Total GPU time per frame: ~80µs. The Python API call, frame packing, embedding loading, and memcpy dominate. **The GPU is doing almost no work.**

Per the Adaptive Reasoning Budget Specification (`docs/vocabulary/ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md` §4):
- `B(q) = B_base × 2^(1−σ(q))`
- B_base = 5 iterations
- For uncertain knowledge (σ=0): 10 iterations
- For contradictory/deep (σ=−1): 20 iterations
- Currently: **3 iterations, no outer loop, no halting check between rounds**

Per the Avatar Embodiment Specification (`docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md` §7.2), the game loop is:
```
PERCEIVE → NAVIGATE → REASON → DECIDE (if not converged: iterate) → ACT → LEARN
```
The "if not converged: iterate" part does not exist in the current kernel.

### Problem 2: ACTION5 (Perform) Always Wins
All 7 action embeddings for non-movement actions have displacement=[0,0], producing near-identical base embeddings. The RPN hash bucketing (±0.125) is too weak to overcome the swarm's output bias. `arc3_action_select_device` adds only 4% frame influence — not enough to differentiate actions.

**Result:** cosine similarity between swarm output and Perform embedding wins by ~0.001 on every single frame across all 8 games.

### Problem 3: No Exploration
Pure greedy cosine scoring with no history, no temperature, no repetition penalty. Once Perform wins frame 1, it wins forever.

---

## DO NOT:
- Add Python decision logic (no action selection in Python)
- Import external frameworks
- Modify the slot size (INPUT_SLOT_BYTES stays 1280)
- Add randomness/epsilon-greedy (that's NOT thinking — use the halting gate properly)

## DO:
- Add thinking budget loop to `gpu_task_dispatch.cu`
- Fix action embedding differentiation in `action_embedding_loader.py`
- Add frame-to-action directional mapping in `arc3_action_select_device`
- Add action history suppression in the kernel
- Pack new fields into the FREE space in the input slot (offsets 1040-1279)

---

## ORDER 1: Add Thinking Budget to Input Slot

### 1A: New constants in `device_functions.cuh`

```c
#define GPU_TASK_THINKING_BUDGET_OFFSET 1040
#define GPU_TASK_ACTION_HISTORY_OFFSET  1044
#define GPU_TASK_ACTION_HISTORY_LEN_OFFSET 1051
#define GPU_TASK_TERNARY_SIGNAL_OFFSET  1052
#define GPU_TASK_DEFAULT_THINKING_BUDGET 10
#define GPU_TASK_MIN_THINKING_BUDGET 5
#define GPU_TASK_MAX_THINKING_BUDGET 20
```

### 1B: Pack fields in `vram_task_buffer.py`

Add these constants:
```python
THINKING_BUDGET_OFFSET = 1040
ACTION_HISTORY_OFFSET = 1044
ACTION_HISTORY_LEN_OFFSET = 1051
TERNARY_SIGNAL_OFFSET = 1052
```

In `_pack_task_slot`, pack the new fields from the task dict:
```python
# Thinking budget: how many outer-loop iterations the kernel should run
thinking_budget = max(5, min(20, int(task.get("thinking_budget", 10))))
struct.pack_into("<I", payload, base + THINKING_BUDGET_OFFSET, thinking_budget)

# Action history: last N actions taken (for repetition suppression)
action_history = task.get("action_history", [])
for i, action_idx in enumerate(action_history[:7]):
    struct.pack_into("<B", payload, base + ACTION_HISTORY_OFFSET + i, int(action_idx) & 0xFF)
struct.pack_into("<B", payload, base + ACTION_HISTORY_LEN_OFFSET, min(len(action_history), 7))

# Ternary signal: -1, 0, or +1 (default 0 = uncertain = moderate budget)
ternary_signal = max(-1, min(1, int(task.get("ternary_signal", 0))))
struct.pack_into("<b", payload, base + TERNARY_SIGNAL_OFFSET, ternary_signal)
```

In `_unpack_task_slot`, unpack them:
```python
thinking_budget = struct.unpack_from("<I", payload, base + THINKING_BUDGET_OFFSET)[0]
action_history_len = struct.unpack_from("<B", payload, base + ACTION_HISTORY_LEN_OFFSET)[0]
action_history = [
    struct.unpack_from("<B", payload, base + ACTION_HISTORY_OFFSET + i)[0]
    for i in range(min(action_history_len, 7))
]
ternary_signal = struct.unpack_from("<b", payload, base + TERNARY_SIGNAL_OFFSET)[0]
```

---

## ORDER 2: Rewrite `gpu_task_dispatch.cu` with Thinking Loop

This is the critical change. The kernel must implement the ARB spec's outer loop:

```c
#include "device_functions.cuh"

extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count
) {
    const unsigned int task_id = blockIdx.x;
    if (task_id >= task_count) return;

    const unsigned int input_base = task_id * GPU_TASK_INPUT_SLOT_BYTES;
    const unsigned int output_base = task_id * GPU_TASK_OUTPUT_SLOT_BYTES;

    const float* query_embedding =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_QUERY_EMBEDDING_OFFSET);
    const unsigned int task_type =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_TYPE_OFFSET);
    const unsigned int option_count =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_OPTION_COUNT_OFFSET);
    const float* option_embeddings =
        reinterpret_cast<const float*>(input_buffer + input_base + GPU_TASK_OPTION_EMBEDDINGS_OFFSET);

    /* --- NEW: Read thinking budget and action history from slot --- */
    const unsigned int raw_budget =
        *reinterpret_cast<const unsigned int*>(input_buffer + input_base + GPU_TASK_THINKING_BUDGET_OFFSET);
    const unsigned int thinking_budget =
        (raw_budget >= GPU_TASK_MIN_THINKING_BUDGET && raw_budget <= GPU_TASK_MAX_THINKING_BUDGET)
        ? raw_budget : GPU_TASK_DEFAULT_THINKING_BUDGET;
    const unsigned char* action_history =
        input_buffer + input_base + GPU_TASK_ACTION_HISTORY_OFFSET;
    const unsigned int action_history_len =
        static_cast<unsigned int>(input_buffer[input_base + GPU_TASK_ACTION_HISTORY_LEN_OFFSET]);
    const int ternary_signal =
        static_cast<int>(*reinterpret_cast<const signed char*>(input_buffer + input_base + GPU_TASK_TERNARY_SIGNAL_OFFSET));

    __shared__ float chain_states[GPU_TASK_NUM_CHAINS * GPU_TASK_EMBED_DIMS];
    __shared__ float swarm_output[GPU_TASK_EMBED_DIMS];
    __shared__ float resonance_scores[GPU_TASK_NUM_CHAINS];
    __shared__ float candidate_scores[GPU_TASK_MAX_OPTIONS];
    __shared__ unsigned int best_index;
    __shared__ float best_score;
    __shared__ int converged;
    __shared__ unsigned int bounded_options;
    __shared__ unsigned int iterations_used;

    if (threadIdx.x == 0) {
        bounded_options = option_count > GPU_TASK_MAX_OPTIONS ? GPU_TASK_MAX_OPTIONS : option_count;
        best_index = 0u;
        best_score = 0.0f;
        converged = 0;
        iterations_used = 0u;
    }
    __syncthreads();

    /* ================================================================
       THINKING LOOP — implements ARB spec §4 and Avatar Embodiment §7.2
       Each iteration = one cognitive cycle:
         REASON (swarm) → SPECIALIZE → EXECUTIVE → SCORE → DECIDE (halt?)
       Budget: B(q) = B_base × 2^(1−σ(q)), packed by Python as thinking_budget
       ================================================================ */
    for (unsigned int think_step = 0u; think_step < thinking_budget; ++think_step) {

        /* --- REASON: Nine-Chain Swarm ---
           First iteration: 3 internal swarm rounds (cold start).
           Subsequent iterations: 2 rounds (warm continuation). */
        const int swarm_rounds = (think_step == 0u) ? 3 : 2;
        nine_chain_swarm_device(query_embedding, chain_states, swarm_output, resonance_scores, swarm_rounds);

        /* --- SPECIALIZE: Task-type specialist activation --- */
        switch (task_type) {
            case 0u:
                arc_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                geometry_route_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                fractal_emit_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 1u:
                atomic_fission_fusion_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                geometry_route_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 2u:
                atomic_fission_fusion_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                temporal_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 3u:
                graph_crystallize_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 4u:
                resonance_field_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                vector_resonate_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                break;
            case 8u:
                arc_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                geometry_route_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                fractal_emit_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
                arc3_action_select_device(swarm_output, chain_states, query_embedding, GPU_TASK_EMBED_DIMS);
                break;
            default:
                __syncthreads();
                break;
        }

        /* --- EXECUTIVE: Trust-weighted chain aggregation --- */
        cognitive_executive_device(resonance_scores, chain_states, swarm_output, GPU_TASK_EMBED_DIMS);

        /* --- DECIDE: Check halting gate --- */
        if (threadIdx.x == 0) {
            iterations_used = think_step + 1u;

            /* Score candidates */
            if (bounded_options > 0u) {
                for (unsigned int oi = 0u; oi < bounded_options; ++oi) {
                    const float* opt = option_embeddings + (oi * GPU_TASK_EMBED_DIMS);
                    float score = cosine32_device(swarm_output, opt, GPU_TASK_EMBED_DIMS);

                    /* --- Action history suppression (GPU-side exploration) ---
                       Decay score for recently repeated actions.
                       Most recent action: 0.85x, second most recent: 0.90x, etc.
                       This is NOT randomness — it's memory-informed preference. */
                    for (unsigned int hi = 0u; hi < action_history_len && hi < 7u; ++hi) {
                        if (static_cast<unsigned int>(action_history[hi]) == oi) {
                            const float recency = 1.0f - (0.15f / (1.0f + static_cast<float>(hi)));
                            score *= recency;
                            break;
                        }
                    }

                    candidate_scores[oi] = score;
                }
                for (unsigned int oi = bounded_options; oi < GPU_TASK_MAX_OPTIONS; ++oi) {
                    candidate_scores[oi] = -1.0e30f;
                }

                /* Find best */
                best_index = 0u;
                best_score = candidate_scores[0];
                for (unsigned int oi = 1u; oi < bounded_options; ++oi) {
                    if (candidate_scores[oi] > best_score) {
                        best_score = candidate_scores[oi];
                        best_index = oi;
                    }
                }

                /* Halting gate: check convergence */
                converged = halting_gate_device(
                    resonance_scores,
                    GPU_TASK_NUM_CHAINS,
                    0.1f,   /* min_threshold */
                    0.15f,  /* gap_threshold */
                    0.7f    /* agreement_threshold */
                );
            }
        }
        __syncthreads();

        /* If converged AND we've run at least the minimum budget, stop thinking */
        if (converged && iterations_used >= GPU_TASK_MIN_THINKING_BUDGET) {
            break;
        }
    }

    /* --- OUTPUT --- */
    if (threadIdx.x == 0) {
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + 0u) = best_index;
        *reinterpret_cast<float*>(output_buffer + output_base + 4u) = best_score;
        *reinterpret_cast<signed char*>(output_buffer + output_base + 8u) =
            bounded_options > 0u ? static_cast<signed char>(converged ? 1 : 0) : static_cast<signed char>(0);
        *reinterpret_cast<unsigned int*>(output_buffer + output_base + 12u) = iterations_used;
        const unsigned long long answer_hash =
            (static_cast<unsigned long long>(task_type) << 32) |
            static_cast<unsigned long long>(best_index);
        *reinterpret_cast<unsigned long long*>(output_buffer + output_base + 16u) = answer_hash;
    }
}
```

**Key architectural points:**
- The outer loop (think_step) IS the "if not converged: iterate" from Avatar Embodiment §7.2
- Halting gate checked EVERY iteration — early exit on convergence (anytime algorithm per ARB §2.5)
- Minimum budget enforced (GPU_TASK_MIN_THINKING_BUDGET = 5 per ARB §5)
- iterations_used written to output so Python can log actual thinking depth
- Action history suppression is memory-informed (NOT random — it's the avatar remembering what it already tried)

---

## ORDER 3: Fix Action Embedding Differentiation

### 3A: Strengthen `_node_to_embedding` in `action_embedding_loader.py`

The current RPN hash bucketing (±0.125) is too weak. Each action TYPE needs a dedicated signature in specific embedding dimensions:

```python
def _node_to_embedding(node: Any) -> list[float]:
    displacement = _get_displacement(None, getattr(node, "node_id", ""))
    embedding = _displacement_to_embedding(displacement)

    # --- ACTION ROLE SIGNATURE (dims 2-3) ---
    # These dimensions encode WHAT KIND of action this is,
    # not just WHERE it moves. Critical for differentiation.
    metadata = getattr(node, "metadata", {}) or {}
    action_type = metadata.get("action_type", "")

    # dim[2]: action category (orthogonal to displacement dims 0-1)
    ACTION_TYPE_SIGNATURES = {
        "spatial_translation": 0.9,       # cardinal moves
        "spatial_translation_composed": 0.8,  # diagonal, walk_to
        "spatial_navigation": 0.7,        # teleport
        "spatial_navigation_composed": 0.7,
        "spatial_orientation": 0.6,       # look_at
        "spatial_interaction": 0.3,       # perform
        "spatial_selection": 0.2,         # click
        "temporal_reversal": -0.9,        # undo
        "object_interaction": 0.1,        # reach/grab/hold/release/use
    }
    embedding[2] = ACTION_TYPE_SIGNATURES.get(action_type, 0.0)

    # dim[3]: parametric flag (actions that take coordinates)
    embedding[3] = 1.0 if metadata.get("parameterized") else 0.0

    # --- RPN TOKEN HASHING (dims 8+) --- stronger weight
    token_stream = " ".join([
        str(getattr(node, "visual_rpn", "")),
        str(getattr(node, "behavior_rpn", "")),
        str(getattr(node, "law_rpn", "")),
    ]).strip()
    tokens = [t for t in token_stream.split() if t]
    for token in tokens:
        bucket = 8 + (_fnv1a32(token) % max(1, EMBEDDING32_DIMS - 8))
        embedding[bucket] += 0.25  # was 0.125 — doubled for stronger differentiation

    # dim[16]: temporal reversal marker (existing, keep)
    if action_type == "temporal_reversal":
        embedding[16] += 1.0

    # dim[17]: parametric marker (existing, keep)
    if metadata.get("parameterized"):
        embedding[17] += 1.0

    # dim[18]: inverse relationship indicator
    if metadata.get("inverse"):
        embedding[18] = 0.5

    return embedding
```

### 3B: Verify Galaxy-sourced embeddings are actually used

In `benchmarks/arc_agi_3.py`, the agent currently builds `_action_embeddings` from `load_action_embeddings_from_galaxy()`. Verify that this path uses `_node_to_embedding` (which applies the role signatures) rather than just `_displacement_to_embedding` (which only uses dx/dy). The Galaxy must have the atoms loaded with full metadata for `_node_to_embedding` to fire.

---

## ORDER 4: Strengthen `arc3_action_select_device` with Directional Frame Mapping

The frame encoder (`arc3_frame_encoder.cu`) encodes spatial features into specific dimensions:
- dims 0-9: color histogram (10 color channels)
- dims 10-11: centroid_x, centroid_y (normalized 0-1)
- dims 12-13: spread_x, spread_y
- dims 14-17: symmetry (horiz, vert, diag_lr, diag_rl)
- dims 18-21: adjacency features

The current `arc3_action_select_device` treats all frame dims equally (just adds 4% frame_bias). It should extract DIRECTIONAL signals:

```c
__device__ void arc3_action_select_device(
    float* embedding,
    const float* context,
    const float* frame_data,
    int dim
) {
    /* Frame feature layout from arc3_frame_encoder.cu:
       [0-9]   color histogram
       [10-11] centroid_x, centroid_y (normalized 0..1, center = 0.5)
       [12-13] spread_x, spread_y
       [14-17] symmetry scores
       [18-21] adjacency features
    */

    if (threadIdx.x == 0) {
        /* Extract directional signal from frame centroid.
           If centroid is off-center, the "interesting" area is offset.
           Movement TOWARD the centroid is more promising.
           Centroid at 0.5 = center, <0.5 = left/up, >0.5 = right/down. */
        const float cx = frame_data[10] - 0.5f;  /* negative = object left of center */
        const float cy = frame_data[11] - 0.5f;  /* negative = object above center */
        const float sx = frame_data[12];          /* spread_x: wide = explore, narrow = act */
        const float sy = frame_data[13];          /* spread_y */

        /* Symmetry signal: high symmetry → pattern might be solvable with "perform" */
        const float symmetry_avg = (frame_data[14] + frame_data[15] +
                                     frame_data[16] + frame_data[17]) * 0.25f;

        /* Color complexity: more colors = more complex scene = explore more */
        float color_count = 0.0f;
        for (int c = 0; c < 10; ++c) {
            if (frame_data[c] > 0.01f) color_count += 1.0f;
        }
        const float complexity = color_count / 10.0f;

        /* Directional bias vector (dims 0-1 correspond to dx, dy in action embeddings):
           Push swarm output toward the direction of interest. */
        const float dir_strength = 0.15f;  /* 15% directional influence */

        /* dim[0] = dx direction: positive cx means object is right → bias right (positive) */
        embedding[0] = tanhf(embedding[0] + dir_strength * cx);

        /* dim[1] = dy direction: positive cy means object is below → bias down (positive) */
        embedding[1] = tanhf(embedding[1] + dir_strength * cy);

        /* dim[2] = action type: high complexity → prefer movement (0.9),
           high symmetry → prefer interaction (0.3) */
        const float type_bias = (complexity > 0.3f) ? 0.8f : (symmetry_avg > 0.5f ? 0.3f : 0.5f);
        embedding[2] = tanhf(embedding[2] + 0.10f * (type_bias - 0.5f));

        /* dim[4] = magnitude: spread indicates whether to move far or act locally */
        const float spread_mag = sqrtf(sx * sx + sy * sy);
        embedding[4] = tanhf(embedding[4] + 0.10f * spread_mag);

        /* General frame influence on remaining dims (keep existing behavior, weaker) */
        for (int i = 5; i < dim; ++i) {
            const float spatial_delta = context[(3 * dim) + i] - context[(4 * dim) + i];
            embedding[i] = tanhf((0.94f * embedding[i]) + (0.03f * device_absf(spatial_delta)) + (0.03f * frame_data[i]));
        }
    }
    __syncthreads();
}
```

**Why this works:**
- Frame centroid directly biases the swarm output toward the direction of visual interest
- This directional bias aligns with movement action embeddings (which have non-zero displacement in dims 0-1)
- Symmetry and complexity signals influence the action-type dimension (dim 2)
- The cosine scoring then naturally favors the action whose embedding aligns with the frame's directional signal

---

## ORDER 5: Wire Action History in the Agent

### 5A: Update `K3DARC3Agent.choose_action` in `benchmarks/arc_agi_3.py`

The agent must pass action history to the task buffer so the kernel can suppress repetition:

```python
def choose_action(self, frame: list[list[int]]) -> dict[str, Any]:
    frame_embedding = self.encoder.encode(frame)
    self.frame_history.append(frame_embedding)

    # Build action history from last N actions (most recent first)
    recent_actions = [
        r["action_index"] for r in self.action_history[-7:]
    ][::-1]  # reverse: index 0 = most recent

    # Compute thinking budget per ARB spec:
    # B(q) = B_base × 2^(1−σ(q)), B_base=5
    # σ defaults to 0 (uncertain) = 10 iterations
    # If we've been repeating the same action, signal = -1 (contradictory) = 20 iterations
    ternary_signal = 0
    if len(recent_actions) >= 3 and len(set(recent_actions[:3])) == 1:
        ternary_signal = -1  # stuck! think harder
    thinking_budget = int(5 * (2 ** (1 - ternary_signal)))
    thinking_budget = max(5, min(20, thinking_budget))

    task = {
        "type": "ARC3_TASK",
        "query_embedding": frame_embedding,
        "option_embeddings": self._action_embeddings,
        "subject": "arc3_game",
        "domain_hint": "arc3_interactive",
        "thinking_budget": thinking_budget,
        "action_history": recent_actions,
        "ternary_signal": ternary_signal,
    }
    loaded = self.task_buffer.bulk_load([task])
    self.dispatcher.launch(self.task_buffer, loaded)
    results = self.task_buffer.read_results(loaded)
    result = results[0] if results else {"answer_index": 0, "confidence": 0.0, "convergence_signal": 0}
    action_index = int(result.get("answer_index", 0))
    action_index = max(0, min(action_index, len(ACTION_NAMES) - 1))
    action_record = {
        "action": ACTION_NAMES[action_index],
        "action_index": action_index,
        "label": ACTION_LABELS[action_index],
        "confidence": float(result.get("confidence", 0.0)),
        "converged": int(result.get("convergence_signal", 0)),
        "iterations_used": int(result.get("iterations_used", 0)),
        "thinking_budget": thinking_budget,
        "ternary_signal": ternary_signal,
        "frame_number": len(self.frame_history),
    }
    self.action_history.append(action_record)
    return action_record
```

---

## ORDER 6: Update `gpu_task_dispatch.py` CPU Reference

The CPU reference in `gpu_task_dispatch.py` (`cpu_reference_dispatch`) must mirror the thinking loop for test parity. Add the same outer loop structure:
- Read thinking_budget from task
- Loop up to thinking_budget times
- Check halting per iteration
- Apply action history suppression
- Write iterations_used to result

---

## ORDER 7: Test

### 7A: Existing tests must still pass
```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py -v
```

### 7B: New test — verify action diversity
Add a test in `tests/test_arc3_agent.py` that:
1. Runs the agent on 20 synthetic frames with varying centroid positions
2. Asserts that at least 3 different actions are chosen (not all ACTION5)
3. Verifies `iterations_used` >= 5 for all results

### 7C: New test — verify thinking budget
Add a test in `tests/test_gpu_task_dispatch.py` that:
1. Packs a task with `thinking_budget=10`
2. Dispatches it
3. Reads `iterations_used` from result
4. Asserts `iterations_used` >= 5 and <= 10

### 7D: Run ARC-AGI-3 synthetic to verify diversity
```bash
python scripts/run_full_benchmark.py --arc3-count 20
```
Check that `action_distribution` shows at least 3 different actions.

---

## FILE INVENTORY

Files you MODIFY:
- `knowledge3d/cranium/cuda/device_functions.cuh` — add new offset constants
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — rewrite with thinking loop
- `knowledge3d/knowledgeverse/vram_task_buffer.py` — pack/unpack thinking_budget, action_history, ternary_signal
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — update CPU reference with thinking loop
- `knowledge3d/knowledgeverse/action_embedding_loader.py` — strengthen `_node_to_embedding` action role signatures
- `benchmarks/arc_agi_3.py` — wire action history + thinking budget + ternary signal
- `tests/test_arc3_agent.py` — add action diversity test
- `tests/test_gpu_task_dispatch.py` — add thinking budget test

Files you DO NOT TOUCH:
- `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` — frame encoder layout is correct
- `knowledge3d/cranium/action_primitives_bootstrap.py` — atoms are correct
- `scripts/run_full_benchmark.py` — runner is correct
- GPU kernels other than `gpu_task_dispatch.cu` and `device_functions.cuh`

---

## EXECUTION SEQUENCE

1. Add new constants to `device_functions.cuh`
2. Update `vram_task_buffer.py` with new pack/unpack fields
3. Rewrite `gpu_task_dispatch.cu` with thinking loop
4. Strengthen `_node_to_embedding` in `action_embedding_loader.py`
5. Rewrite `arc3_action_select_device` in `device_functions.cuh`
6. Wire action history + thinking budget in `benchmarks/arc_agi_3.py`
7. Update CPU reference in `gpu_task_dispatch.py`
8. Run tests → all green
9. Run `scripts/run_full_benchmark.py --arc3-count 20` → verify action diversity
10. Report: action distribution, iterations_used histogram, GPU kernel time

---

## SUCCESS CRITERIA

- **Thinking budget**: `iterations_used` >= 5 for all tasks (minimum budget enforced)
- **Action diversity**: ARC-AGI-3 synthetic shows >= 3 different actions in 20 frames
- **No ACTION5 fixation**: Perform is NOT 100% of actions
- **Directional sensitivity**: Frames with off-center centroid produce movement actions toward the centroid
- **GPU utilization**: Kernel time per frame increases from ~80µs to ~500µs+ (actual thinking)
- **Existing benchmarks hold**: Synthetic 10/10, MMLU >= 30%
- **Spec compliance**:
  - ARB spec §4: `B(q) = B_base × 2^(1−σ(q))` implemented
  - Avatar Embodiment §7.2: PERCEIVE→NAVIGATE→REASON→DECIDE(loop)→ACT cycle implemented
  - Three Brain: Python = boot + I/O only, all reasoning on GPU

**Build it.**
