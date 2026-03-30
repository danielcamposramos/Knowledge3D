# Codex — Phase E.7: Living Agent — Persistent Brain State + Sleep-Time

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.6 DONE. Action diversity fixed. Thinking budget loop works. **BUT: GPU at ~0.25% utilization. Python-driven burst pattern. No persistent brain state. No sleep-time. Not a living AI — still a script calling functions.**

---

## THE FUNDAMENTAL PROBLEM (6 MONTHS OF CORRECTION)

Daniel has been saying this since the beginning. Read the specs:

**Three Brain System Specification §3.1:**
> "TRM IS the avatar — not a function Python calls, but the AI entity that lives in the House and thinks in the Galaxy. Runs as a **continuous game loop** via `trm_step_fused.ptx`"

**Avatar Embodiment Specification §7.2:**
> The game loop runs continuously: PERCEIVE → NAVIGATE → REASON → DECIDE → ACT → **LEARN**

**Knowledgeverse Specification §1.1:**
> "**one persistent CUDA/PTX execution domain** [...] Enables Shadow Copy learning and SleepTime consolidation within GPU context"

**Hyper-Parallel Processing:**
> "Session resets are anti-pattern — one living mind"

**What we built instead:**
```python
while state in ACTIVE_STATES and action_count < max_actions:
    action = agent.choose_action(frame)      # cold-start kernel, ~0.5ms
    response = session.post(api_url, ...)     # GPU IDLE FOR 200ms
    frame = normalize_frame(response)         # Python string ops
```

This is a Python script. The GPU is a calculator it pokes. Between pokes, the brain DIES and is reborn from scratch. No memory. No learning. No consolidation. The "thinking budget" made the 0.5ms kernel richer, but it's still 0.5ms in a 200ms Python loop.

**The fix is not "add more iterations." The fix is: the brain must be alive between frames.**

---

## DO NOT:
- Add Python decision logic
- Import external ML frameworks
- Create new Python orchestration classes
- Make the Python loop smarter (make the GPU loop persistent instead)

## DO:
- Add persistent brain state in VRAM (survives across kernel launches)
- Add frame-delta encoding (what changed after my last action?)
- Wire sleep-time consolidation between API calls
- Wire Shadow Copy event recording for action traces
- Keep the Galaxy loaded warm for the entire session

---

## THE ARCHITECTURE: PERSISTENT VRAM BRAIN

### Current Pattern (WRONG):
```
Frame 1: Python packs slot → GPU kernel cold-starts → GPU writes result → Python reads → GPU DIES
         [--- 200ms API wait, GPU idle ---]
Frame 2: Python packs slot → GPU kernel cold-starts → GPU writes result → Python reads → GPU DIES
```

The kernel at `gpu_task_dispatch.cu` line 48-49 does: `reasoning_state[index] = query_embedding[index]`. Every frame starts from the raw frame embedding. The swarm has amnesia.

### Target Pattern (LIVING AI):
```
Session init: Allocate brain_state in VRAM (persistent across all frames)
              Load Galaxy warm (stays allocated)

Frame 1: Python writes frame to brain input region
          → GPU kernel reads brain_state (warm from init/previous)
          → Swarm reasons with persistent chain states
          → Kernel writes updated brain_state back to VRAM
          → Kernel writes action to output
          Python reads action, calls API

         [--- API wait: Python launches SLEEP-TIME CONSOLIDATION KERNEL ---]
         → Shadow Copy records action trace into VRAM audit region
         → Consolidation kernel strengthens successful swarm paths
         → Specialist weights updated based on ternary outcome signal
         [--- GPU is ACTIVE during the wait ---]

Frame 2: Python writes new frame + frame-delta to brain input region
          → GPU kernel reads brain_state (WARM from frame 1 + consolidation)
          → Swarm starts from where it left off, not from scratch
          → Kernel integrates frame-delta (what changed?) into reasoning
          → Repeat
```

---

## ORDER 1: Create Persistent Brain Buffer

### 1A: New class `PersistentBrainState` in `knowledge3d/knowledgeverse/persistent_brain.py`

This allocates and manages VRAM that persists across kernel launches. Per Knowledgeverse Specification §3 (7 Memory Regions), this lives in the TRM Region.

```python
"""Persistent VRAM brain state for the living AI entity.

Per Three Brain System Spec §3.1: TRM runs as continuous game loop.
Per Knowledgeverse Spec §3: TRM Region holds weights + reasoning state.
This buffer persists across kernel launches — the brain stays alive.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader


# Brain layout (all in VRAM, persistent across frames):
BRAIN_REASONING_STATE_BYTES = 32 * 4            # 32 floats: last reasoning output
BRAIN_CHAIN_STATES_BYTES = 9 * 32 * 4           # 9 chains × 32 floats: swarm state
BRAIN_PREV_FRAME_BYTES = 32 * 4                  # 32 floats: previous frame embedding
BRAIN_ACTION_RING_BYTES = 7                       # 7 bytes: action history ring
BRAIN_ACTION_RING_LEN_BYTES = 1                   # 1 byte: ring length
BRAIN_TERNARY_SIGNAL_BYTES = 1                    # 1 byte: current ternary state
BRAIN_FRAME_COUNT_BYTES = 4                       # uint32: frames processed
BRAIN_SPECIALIST_TRACE_BYTES = 9 * 4             # 9 floats: specialist activation trace
BRAIN_PAD = 3                                      # alignment padding

BRAIN_TOTAL_BYTES = (
    BRAIN_REASONING_STATE_BYTES +     # 128
    BRAIN_CHAIN_STATES_BYTES +        # 1152
    BRAIN_PREV_FRAME_BYTES +          # 128
    BRAIN_ACTION_RING_BYTES +         # 7
    BRAIN_ACTION_RING_LEN_BYTES +     # 1
    BRAIN_TERNARY_SIGNAL_BYTES +      # 1
    BRAIN_FRAME_COUNT_BYTES +         # 4
    BRAIN_SPECIALIST_TRACE_BYTES +    # 36
    BRAIN_PAD                          # 3
)  # = 1460 bytes, fits in 2KB aligned

# Offsets within the brain buffer
BRAIN_REASONING_OFFSET = 0
BRAIN_CHAINS_OFFSET = BRAIN_REASONING_STATE_BYTES
BRAIN_PREV_FRAME_OFFSET = BRAIN_CHAINS_OFFSET + BRAIN_CHAIN_STATES_BYTES
BRAIN_ACTION_RING_OFFSET = BRAIN_PREV_FRAME_OFFSET + BRAIN_PREV_FRAME_BYTES
BRAIN_ACTION_RING_LEN_OFFSET = BRAIN_ACTION_RING_OFFSET + BRAIN_ACTION_RING_BYTES
BRAIN_TERNARY_OFFSET = BRAIN_ACTION_RING_LEN_OFFSET + BRAIN_ACTION_RING_LEN_BYTES
BRAIN_FRAME_COUNT_OFFSET = BRAIN_TERNARY_OFFSET + BRAIN_TERNARY_SIGNAL_BYTES
BRAIN_SPECIALIST_TRACE_OFFSET = BRAIN_FRAME_COUNT_OFFSET + BRAIN_FRAME_COUNT_BYTES


class PersistentBrainState:
    """VRAM-resident brain state that persists across kernel launches.

    This IS the avatar's cognitive state. It does not die between frames.
    """

    def __init__(self) -> None:
        self.gpu_ptr = loader.gpu_malloc(BRAIN_TOTAL_BYTES)
        # Zero-initialize: clean slate at session start (not per frame)
        payload = bytearray(BRAIN_TOTAL_BYTES)
        self._upload(payload)

    def close(self) -> None:
        if getattr(self, "gpu_ptr", None):
            loader.gpu_free(self.gpu_ptr)
            self.gpu_ptr = None

    def _upload(self, payload: bytearray) -> None:
        ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))
        loader.memcpy_htod(self.gpu_ptr, ptr, len(payload))

    def _download(self) -> bytearray:
        payload = bytearray(BRAIN_TOTAL_BYTES)
        ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))
        loader.memcpy_dtoh(ptr, self.gpu_ptr, BRAIN_TOTAL_BYTES)
        return payload

    def read_state(self) -> dict[str, Any]:
        """Read current brain state from VRAM (for logging/diagnostics)."""
        data = self._download()
        reasoning = list(struct.unpack_from("<32f", data, BRAIN_REASONING_OFFSET))
        frame_count = struct.unpack_from("<I", data, BRAIN_FRAME_COUNT_OFFSET)[0]
        ring_len = data[BRAIN_ACTION_RING_LEN_OFFSET]
        action_ring = list(data[BRAIN_ACTION_RING_OFFSET:BRAIN_ACTION_RING_OFFSET + min(ring_len, 7)])
        ternary = struct.unpack_from("<b", data, BRAIN_TERNARY_OFFSET)[0]
        specialist_trace = list(struct.unpack_from("<9f", data, BRAIN_SPECIALIST_TRACE_OFFSET))
        return {
            "reasoning_norm": sum(v * v for v in reasoning) ** 0.5,
            "frame_count": frame_count,
            "action_ring": action_ring,
            "ternary_signal": ternary,
            "specialist_trace": specialist_trace,
        }
```

---

## ORDER 2: Modify `gpu_task_dispatch.cu` to Read/Write Brain State

The kernel gains a FOURTH parameter: the persistent brain buffer. Instead of cold-starting `reasoning_state` from `query_embedding`, it reads from brain VRAM. After reasoning, it writes back.

### 2A: New kernel signature

```c
extern "C" __global__ void gpu_task_dispatch(
    const unsigned char* __restrict__ input_buffer,
    unsigned char* __restrict__ output_buffer,
    unsigned int task_count,
    unsigned char* __restrict__ brain_state    /* NEW: persistent VRAM brain */
)
```

### 2B: Replace cold-start with warm-start

**Current** (line 48-49, WRONG — amnesia):
```c
reasoning_state[index] = query_embedding[index];
swarm_output[index] = query_embedding[index];
```

**New** (warm start from persistent brain):
```c
/* Read persistent reasoning state from brain VRAM.
   If frame_count == 0 (first frame), use query_embedding as seed.
   Otherwise, BLEND persistent state with new frame embedding.
   This is "single mind" — the avatar remembers what it was thinking. */

const unsigned int frame_count = *reinterpret_cast<const unsigned int*>(
    brain_state + BRAIN_FRAME_COUNT_OFFSET);
const float* brain_reasoning = reinterpret_cast<const float*>(
    brain_state + BRAIN_REASONING_OFFSET);
const float* prev_frame = reinterpret_cast<const float*>(
    brain_state + BRAIN_PREV_FRAME_OFFSET);

for (int i = threadIdx.x; i < GPU_TASK_EMBED_DIMS; i += blockDim.x) {
    if (frame_count == 0u) {
        /* First frame: seed from query (frame embedding) */
        reasoning_state[i] = query_embedding[i];
    } else {
        /* Warm start: blend persistent brain state with new perception.
           70% memory (what I was thinking) + 30% new perception (what I see now).
           Per Avatar Embodiment §7.2 step 1 (PERCEIVE): new stimulus
           integrates with ongoing cognitive state, does not replace it. */
        reasoning_state[i] = tanhf(
            (0.70f * brain_reasoning[i]) +
            (0.30f * query_embedding[i])
        );
    }
    swarm_output[i] = reasoning_state[i];
}
```

### 2C: Read frame-delta from brain

```c
/* Frame delta: what changed since my last action?
   Per Avatar Embodiment §7.2 step 1: perception includes change detection. */
__shared__ float frame_delta[GPU_TASK_EMBED_DIMS];
for (int i = threadIdx.x; i < GPU_TASK_EMBED_DIMS; i += blockDim.x) {
    frame_delta[i] = (frame_count > 0u)
        ? (query_embedding[i] - prev_frame[i])
        : 0.0f;
}
__syncthreads();
```

Inject `frame_delta` into the ARC3 specialist path (case 8):
```c
case 8u:
    arc_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    geometry_route_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    fractal_emit_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    arc3_action_select_device(swarm_output, chain_states, query_embedding, GPU_TASK_EMBED_DIMS);
    /* NEW: integrate frame delta — what changed? */
    arc3_frame_delta_device(swarm_output, frame_delta, GPU_TASK_EMBED_DIMS);
    break;
```

### 2D: Write brain state back after reasoning

After the thinking loop completes (before writing output):
```c
/* LEARN (Avatar Embodiment §7.2 step 6): persist cognitive state.
   The brain stays alive for the next frame. */
for (int i = threadIdx.x; i < GPU_TASK_EMBED_DIMS; i += blockDim.x) {
    /* Write reasoning result to persistent brain */
    reinterpret_cast<float*>(brain_state + BRAIN_REASONING_OFFSET)[i] = swarm_output[i];
    /* Save current frame as "previous" for next frame's delta */
    reinterpret_cast<float*>(brain_state + BRAIN_PREV_FRAME_OFFSET)[i] = query_embedding[i];
}
if (threadIdx.x == 0) {
    /* Update frame count */
    *reinterpret_cast<unsigned int*>(brain_state + BRAIN_FRAME_COUNT_OFFSET) = frame_count + 1u;
    /* Write action to ring buffer */
    unsigned int ring_len = static_cast<unsigned int>(brain_state[BRAIN_ACTION_RING_LEN_OFFSET]);
    if (ring_len < 7u) {
        brain_state[BRAIN_ACTION_RING_OFFSET + ring_len] = static_cast<unsigned char>(best_index);
        brain_state[BRAIN_ACTION_RING_LEN_OFFSET] = static_cast<unsigned char>(ring_len + 1u);
    } else {
        /* Shift ring left, append new */
        for (unsigned int r = 0u; r < 6u; ++r) {
            brain_state[BRAIN_ACTION_RING_OFFSET + r] = brain_state[BRAIN_ACTION_RING_OFFSET + r + 1u];
        }
        brain_state[BRAIN_ACTION_RING_OFFSET + 6u] = static_cast<unsigned char>(best_index);
    }
    /* Write specialist activation trace (resonance scores) */
    for (int c = 0; c < GPU_TASK_NUM_CHAINS; ++c) {
        reinterpret_cast<float*>(brain_state + BRAIN_SPECIALIST_TRACE_OFFSET)[c] = resonance_scores[c];
    }
}
__syncthreads();
```

### 2E: Read action history from brain instead of input slot

Replace the current pattern (reading action_history from input slot) with reading from the persistent brain ring buffer:
```c
const unsigned char* action_history = brain_state + BRAIN_ACTION_RING_OFFSET;
const unsigned int action_history_len =
    static_cast<unsigned int>(brain_state[BRAIN_ACTION_RING_LEN_OFFSET]);
```

This means Python no longer needs to pack action history per frame — the brain maintains its own memory.

---

## ORDER 3: Add `arc3_frame_delta_device` to `device_functions.cuh`

```c
__device__ void arc3_frame_delta_device(
    float* embedding,
    const float* frame_delta,
    int dim
) {
    /* Frame delta captures what changed after the avatar's last action.
       Large delta = action had effect → reinforce similar actions.
       Zero delta = action had no effect → suppress and explore.
       Per Avatar Embodiment §7.2: PERCEIVE includes change detection. */
    if (threadIdx.x == 0) {
        float delta_magnitude = 0.0f;
        for (int i = 0; i < dim; ++i) {
            delta_magnitude += frame_delta[i] * frame_delta[i];
        }
        delta_magnitude = sqrtf(delta_magnitude + 1.0e-12f);

        /* delta_signal: high = action changed the world, low = action did nothing */
        const float delta_signal = device_clamp01(delta_magnitude * 2.0f);

        for (int i = 0; i < dim; ++i) {
            /* Blend: if action had effect, lean into the direction of change.
               If action had no effect, add exploratory perturbation. */
            const float explore = (delta_signal < 0.1f)
                ? (0.08f * pseudo_random_device(i, dim))
                : 0.0f;
            embedding[i] = tanhf(
                (0.92f * embedding[i]) +
                (0.06f * delta_signal * frame_delta[i]) +
                explore
            );
        }
    }
    __syncthreads();
}
```

---

## ORDER 4: Wire Sleep-Time Consolidation Between Frames

### 4A: Create `knowledge3d/cranium/cuda/sleep_time_micro.cu`

A lightweight consolidation kernel that runs during API wait time. Per Knowledgeverse Specification §8 (SleepTime Consolidation):

```c
#include "device_functions.cuh"

/* Micro sleep-time consolidation: runs between game frames.
   Not the full two-phase commit (that's for idle periods).
   This is "micro-nap" consolidation: strengthen successful paths,
   weaken failed paths, based on the ternary outcome signal.

   Per Knowledgeverse Spec §8: "SleepTime consolidation happens in two stages"
   This is a fast Stage-A-only pass suitable for inter-frame gaps. */

extern "C" __global__ void sleep_time_micro(
    unsigned char* __restrict__ brain_state,
    int outcome_signal    /* +1 = action helped, 0 = neutral, -1 = action failed */
) {
    float* reasoning = reinterpret_cast<float*>(brain_state + BRAIN_REASONING_OFFSET);
    float* chains = reinterpret_cast<float*>(brain_state + BRAIN_CHAINS_OFFSET);
    float* specialist_trace = reinterpret_cast<float*>(brain_state + BRAIN_SPECIALIST_TRACE_OFFSET);

    const float outcome = static_cast<float>(outcome_signal);

    /* Strengthen/weaken reasoning state based on outcome.
       Per Three Brain Spec §3.1: Shadow Copy records successful traces.
       +1 outcome: amplify current reasoning direction (reinforce)
       -1 outcome: dampen current reasoning direction (explore alternatives)
       0 outcome: gentle decay toward neutral (don't commit either way) */
    for (int i = threadIdx.x; i < GPU_TASK_EMBED_DIMS; i += blockDim.x) {
        if (outcome > 0.5f) {
            /* Reinforce: sharpen the reasoning state */
            reasoning[i] = tanhf(reasoning[i] * 1.05f);
        } else if (outcome < -0.5f) {
            /* Weaken: decay toward zero, encouraging exploration */
            reasoning[i] *= 0.85f;
        } else {
            /* Neutral: very gentle decay */
            reasoning[i] *= 0.98f;
        }
    }
    __syncthreads();

    /* Update specialist trace: boost chains that agree with outcome */
    if (threadIdx.x == 0) {
        for (int c = 0; c < GPU_TASK_NUM_CHAINS; ++c) {
            const float resonance = specialist_trace[c];
            if (outcome > 0.5f) {
                /* Successful: boost resonant chains */
                specialist_trace[c] = device_clamp01(resonance + 0.05f);
            } else if (outcome < -0.5f) {
                /* Failed: penalize high-resonance chains (they led to bad action) */
                specialist_trace[c] = device_clamp01(resonance - 0.03f);
            }
        }
    }
    __syncthreads();

    /* Cross-pollinate chain states: best chain seeds weakest.
       This implements the "superdotados" model (Hyper-Parallel Processing Spec):
       successful cognitive channels strengthen weaker ones. */
    if (threadIdx.x == 0 && outcome > 0.5f) {
        int best_chain = 0;
        int worst_chain = 0;
        float best_res = specialist_trace[0];
        float worst_res = specialist_trace[0];
        for (int c = 1; c < GPU_TASK_NUM_CHAINS; ++c) {
            if (specialist_trace[c] > best_res) { best_res = specialist_trace[c]; best_chain = c; }
            if (specialist_trace[c] < worst_res) { worst_res = specialist_trace[c]; worst_chain = c; }
        }
        if (best_chain != worst_chain) {
            float* src = chains + (best_chain * GPU_TASK_EMBED_DIMS);
            float* dst = chains + (worst_chain * GPU_TASK_EMBED_DIMS);
            for (int i = 0; i < GPU_TASK_EMBED_DIMS; ++i) {
                dst[i] = tanhf((0.70f * dst[i]) + (0.30f * src[i]));
            }
        }
    }
    __syncthreads();
}
```

### 4B: Create `knowledge3d/knowledgeverse/sleep_time_micro.py`

Python launcher for the micro-consolidation kernel:

```python
"""Micro sleep-time consolidation launcher.

Runs between game frames during API wait time.
Per Knowledgeverse Spec §8: fast Stage-A-only pass.
"""

from __future__ import annotations
from knowledge3d.cranium.sovereign import loader


class SleepTimeMicro:
    """Launch micro-consolidation kernel on the persistent brain state."""

    def __init__(self) -> None:
        self._module = None
        self._kernel = None

    def _ensure_compiled(self) -> None:
        if self._kernel is not None:
            return
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "cranium" / "cuda" / "sleep_time_micro.cu"
        header_dir = str(src.parent)
        self._module = loader.compile_ptx(str(src), extra_flags=[f"-I{header_dir}"])
        self._kernel = loader.get_function(self._module, "sleep_time_micro")

    def consolidate(self, brain_gpu_ptr: int, outcome_signal: int) -> None:
        """Run micro-consolidation on the brain state.

        Args:
            brain_gpu_ptr: VRAM pointer to PersistentBrainState
            outcome_signal: +1 (action helped), 0 (neutral), -1 (action failed)
        """
        self._ensure_compiled()
        import ctypes
        outcome = max(-1, min(1, int(outcome_signal)))
        self._kernel(
            ctypes.c_void_p(brain_gpu_ptr),
            ctypes.c_int(outcome),
            block=(128, 1, 1),
            grid=(1, 1, 1),
        )
```

---

## ORDER 5: Rewrite `K3DARC3Agent` as Living Agent

The agent class must use the persistent brain state, wire sleep-time between frames, and track frame deltas. Per Avatar Embodiment §7.2: the full 6-step cycle.

```python
class K3DARC3Agent:
    """Sovereign GPU agent — a LIVING entity, not a script.

    Per Three Brain System Spec §3.1: TRM IS the avatar.
    Per Avatar Embodiment §7.2: continuous game loop with LEARN step.
    """

    def __init__(
        self,
        max_actions: int = 80,
        log_path: str | Path | None = None,
        galaxy: Any | None = None,
    ) -> None:
        self.max_actions = int(max_actions)
        self.log_path = Path(log_path) if log_path else None
        self.encoder = ARC3FrameEncoder()
        self.dispatcher = GPUTaskDispatch()
        self.task_buffer = VRAMTaskBuffer(max_tasks=1)
        self.brain = PersistentBrainState()          # PERSISTENT VRAM brain
        self.sleeper = SleepTimeMicro()               # Inter-frame consolidation
        self.action_history: list[dict[str, Any]] = []
        self.frame_history: list[list[float]] = []
        self.prev_levels: int = 0                     # Track level progress
        self.reality_galaxy = galaxy if galaxy is not None else build_default_action_galaxy()
        self._action_embeddings = (
            load_action_embeddings_from_galaxy(self.reality_galaxy, ARC3_EXTENDED_ACTION_ATOM_IDS)
            if self.reality_galaxy is not None
            else ACTION_EMBEDDINGS
        )

    def choose_action(self, frame: list[list[int]]) -> dict[str, Any]:
        """PERCEIVE → NAVIGATE → REASON → DECIDE → ACT (Avatar Embodiment §7.2 steps 1-5)"""
        frame_embedding = self.encoder.encode(frame)
        self.frame_history.append(frame_embedding)

        # Ternary signal from brain state (not recomputed in Python)
        brain_state = self.brain.read_state()
        recent_actions = brain_state["action_ring"]
        ternary_signal = 0
        if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
            ternary_signal = -1
        thinking_budget = int(5 * (2 ** (1 - ternary_signal)))
        thinking_budget = max(5, min(20, thinking_budget))

        task = {
            "type": "ARC3_TASK",
            "query_embedding": frame_embedding,
            "option_embeddings": self._action_embeddings,
            "subject": "arc3_game",
            "domain_hint": "arc3_interactive",
            "thinking_budget": thinking_budget,
            "ternary_signal": ternary_signal,
            # action_history no longer packed — brain maintains its own memory
        }
        loaded = self.task_buffer.bulk_load([task])
        self.dispatcher.launch(self.task_buffer, loaded, brain_ptr=self.brain.gpu_ptr)
        results = self.task_buffer.read_results(loaded)
        result = results[0] if results else {"answer_index": 0, "confidence": 0.0}
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

    def learn_from_outcome(self, levels_completed: int) -> None:
        """LEARN step (Avatar Embodiment §7.2 step 6).

        Called BETWEEN frames, during the API wait.
        This IS sleep-time micro-consolidation.
        """
        if levels_completed > self.prev_levels:
            outcome = 1   # action progressed! reinforce
        elif levels_completed < self.prev_levels:
            outcome = -1  # regressed! weaken
        else:
            outcome = 0   # neutral

        # Launch consolidation kernel (GPU work during API wait)
        self.sleeper.consolidate(self.brain.gpu_ptr, outcome)
        self.prev_levels = levels_completed

    def close(self) -> None:
        self.task_buffer.close()
        self.brain.close()
        if self.log_path and self.action_history:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in self.action_history))
                handle.write("\n")
```

---

## ORDER 6: Wire Persistent Brain into `run_arc3_agent.py`

The live runner's game loop must call `learn_from_outcome` between frames:

```python
while state in ACTIVE_STATES and action_count < max_actions:
    action = agent.choose_action(frame)

    # --- API CALL (GPU consolidates during this wait) ---
    response = session.post(...)
    levels_completed = level_progress(response.get("levels_completed", ...))

    # --- LEARN: Sleep-time micro-consolidation (Avatar Embodiment §7.2 step 6) ---
    agent.learn_from_outcome(levels_completed)

    frame = normalize_frame(response.get("frame", ...))
    # ... rest unchanged
```

---

## ORDER 7: Wire Brain Pointer Through `GPUTaskDispatch.launch`

### 7A: Update `gpu_task_dispatch.py`

The `launch` method must accept and pass the brain pointer as the 4th kernel argument:

```python
def launch(self, task_buffer, task_count, *, brain_ptr=None):
    # ... existing compilation logic ...
    kernel_args = [task_buffer.input_buffer, task_buffer.output_buffer, task_count]
    if brain_ptr is not None:
        kernel_args.append(brain_ptr)
    # Launch with updated args
```

### 7B: Update CPU reference

The `cpu_reference_dispatch` must also accept brain_state for test parity. If brain_state is None, fall back to cold-start (backward compatible with non-ARC3 tasks).

---

## ORDER 8: Add Brain State Constants to `device_functions.cuh`

```c
/* Persistent brain state layout (VRAM, survives across kernel launches) */
#define BRAIN_REASONING_OFFSET         0
#define BRAIN_CHAINS_OFFSET            128     /* 32 * 4 */
#define BRAIN_PREV_FRAME_OFFSET        1280    /* 128 + 9*32*4 */
#define BRAIN_ACTION_RING_OFFSET       1408    /* 1280 + 32*4 */
#define BRAIN_ACTION_RING_LEN_OFFSET   1415    /* 1408 + 7 */
#define BRAIN_TERNARY_OFFSET           1416
#define BRAIN_FRAME_COUNT_OFFSET       1417
#define BRAIN_SPECIALIST_TRACE_OFFSET  1421    /* 1417 + 4 */
#define BRAIN_TOTAL_BYTES              1460    /* 1421 + 9*4 + 3 pad */
```

---

## ORDER 9: Test

### 9A: Existing tests pass
```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py -v
```

### 9B: New test — persistent brain state
Add `tests/test_persistent_brain.py`:
1. Create PersistentBrainState
2. Run agent.choose_action on frame 1
3. Read brain state → frame_count == 1, reasoning_norm > 0
4. Run agent.choose_action on frame 2
5. Read brain state → frame_count == 2, action_ring has 2 entries
6. Close brain → no crash

### 9C: New test — sleep-time micro
1. Create brain, run one action
2. Call sleeper.consolidate(brain.gpu_ptr, +1)
3. Read brain → specialist_trace changed
4. Call sleeper.consolidate(brain.gpu_ptr, -1)
5. Read brain → specialist_trace weakened

### 9D: GPU utilization check
Run ARC-AGI-3 synthetic (20 frames) and observe:
- kernel launches per frame: >= 2 (reasoning + consolidation)
- brain state persists: frame_count == 20 at end
- action diversity maintained

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/knowledgeverse/persistent_brain.py` — VRAM brain state
- `knowledge3d/cranium/cuda/sleep_time_micro.cu` — inter-frame consolidation kernel
- `knowledge3d/knowledgeverse/sleep_time_micro.py` — consolidation launcher
- `tests/test_persistent_brain.py` — brain + consolidation tests

Files you MODIFY:
- `knowledge3d/cranium/cuda/device_functions.cuh` — brain offsets + `arc3_frame_delta_device`
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — 4th parameter, warm start, brain write-back
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — pass brain_ptr, update CPU reference
- `benchmarks/arc_agi_3.py` — PersistentBrainState, learn_from_outcome
- `scripts/run_arc3_agent.py` — wire learn_from_outcome between frames

Files you DO NOT TOUCH:
- `knowledge3d/cranium/cuda/arc3_frame_encoder.cu`
- `knowledge3d/cranium/action_primitives_bootstrap.py`
- `knowledge3d/knowledgeverse/vram_task_buffer.py` (slot layout unchanged)
- `scripts/run_full_benchmark.py`

---

## EXECUTION SEQUENCE

1. Add brain offset constants to `device_functions.cuh`
2. Add `arc3_frame_delta_device` to `device_functions.cuh`
3. Create `persistent_brain.py`
4. Create `sleep_time_micro.cu` + `sleep_time_micro.py`
5. Modify `gpu_task_dispatch.cu` — 4th param, warm start, brain write-back, frame delta
6. Modify `gpu_task_dispatch.py` — pass brain_ptr
7. Rewrite `benchmarks/arc_agi_3.py` — PersistentBrainState + learn_from_outcome
8. Modify `scripts/run_arc3_agent.py` — wire learn_from_outcome
9. Run tests → all green
10. Run ARC-AGI-3 synthetic → verify persistent state + action diversity
11. Report: brain_state diagnostics, GPU kernel count per frame, action distribution

---

## SUCCESS CRITERIA

- **Persistent brain**: `frame_count` accumulates across frames (not reset to 0)
- **Warm start**: reasoning_state at frame N+1 carries information from frame N
- **Sleep-time active**: consolidation kernel fires between every frame pair
- **GPU active between frames**: 2+ kernel launches per frame (reasoning + consolidation)
- **Frame-delta awareness**: agent responds differently when frame changes vs stays same
- **Action diversity maintained**: still >= 3 different actions in 20 synthetic frames
- **Existing benchmarks hold**: Synthetic 10/10, MMLU >= 30%
- **Spec compliance**:
  - Avatar Embodiment §7.2: all 6 steps (PERCEIVE → NAVIGATE → REASON → DECIDE → ACT → **LEARN**)
  - Three Brain §3.1: "continuous game loop", persistent cognitive state
  - Knowledgeverse §8: SleepTime consolidation active
  - Hyper-Parallel Processing: "one living mind, not session resets"

**This is the avatar coming alive. Build it.**
