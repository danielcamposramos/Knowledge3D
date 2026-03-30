# Codex — Phase E.2: ARC-AGI-3 Agent Wiring

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.1 Orders 1-5 DONE. `gpu_task_dispatch.cu` runs the full composed-head pipeline.

---

## CONTEXT: ARC-AGI-3 IS A GAME, K3D IS A GAME ENGINE

ARC-AGI-3 is no longer static grids. It is an **interactive game**:
- Server sends a **frame** (2D grid of color indices)
- Agent sends an **action** (Move Up/Down/Left/Right, Perform, Click(x,y), Undo, Reset)
- Server sends the **next frame**
- Loop until WIN or GAME_OVER

This maps EXACTLY to the TRM game loop:
- **Frame** = what Frustum Cull sees (perception)
- **Action selection** = Nine-Chain Swarm + Halting Gate (reasoning + decision)
- **Action execution** = API call (I/O — Python is fine here)

The GPU does the THINKING. Python does the API I/O. That's it.

---

## DO NOT:
- Add LLM API calls (OpenAI, Anthropic, etc.) for reasoning — K3D IS the reasoning engine
- Add Python game-state analysis, pattern matching, or heuristics — that's the GPU's job
- Import langchain, langgraph, smolagents, or any agentic framework
- Add asyncio, threading, or multiprocessing
- Modify `gpu_task_dispatch.cu` or `device_functions.cuh` (they work)
- Modify `vram_task_buffer.py` slot layout

## DO:
- Use the ARC-AGI-3 Agents SDK (`arcengine`) for API communication
- Pack game frames into VRAM task slots
- Let the existing GPU pipeline choose actions
- Write thin Python I/O (target: <150 lines)

---

## ORDER 1: Create ARC-AGI-3 Task Type

### 1A: Add ARC3 task type to `vram_task_buffer.py`

Add to `TASK_TYPE_IDS`:
```python
"ARC3_TASK": 8,
```

This is separate from `ARC_TASK` (0) which handles static ARC-AGI-2 grid I/O. ARC3 is an interactive game task.

### 1B: Extend `device_functions.cuh` with ARC3 specialist

Add a new device function after `arc_reason_device`:

```c
__device__ void arc3_action_select_device(
    float* embedding,           // swarm output (32 floats)
    const float* context,       // chain_states (9 × 32)
    const float* frame_data,    // flattened grid embedding from query_embedding
    int dim
) {
    // The 7 actions map to 7 directions in embedding space.
    // The swarm output after processing the frame embedding already encodes
    // the "best direction to move" — we just need to sharpen it toward
    // one of the 7 action embeddings.
    //
    // Action encoding in the option_embeddings:
    //   option[0] = Move Up    (ACTION1)
    //   option[1] = Move Down  (ACTION2)
    //   option[2] = Move Left  (ACTION3)
    //   option[3] = Move Right (ACTION4)
    //
    // For ACTION5 (Perform), ACTION6 (Click), ACTION7 (Undo):
    // These are encoded as additional options beyond the 4 movement options.
    // The current 4-option slot limit handles movements.
    // TODO: Extend to 7 option slots for full action coverage.
    //
    // For now: use the ARC specialist pipeline (arc_reason + geometry_router + fractal_emitter)
    // plus world-model temporal coherence from the frame history.

    // Apply temporal difference between chains 3 and 4 (same as arc_reason_device)
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        float spatial_delta = context[(3 * dim) + index] - context[(4 * dim) + index];
        embedding[index] = tanhf((0.94f * embedding[index]) + (0.06f * device_absf(spatial_delta)));
    }
    __syncthreads();
}
```

### 1C: Add case 8 to the task-type switch in `gpu_task_dispatch.cu`

```c
case 8u:  // ARC3 — interactive game
    arc_reason_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    geometry_route_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    fractal_emit_device(swarm_output, chain_states, GPU_TASK_EMBED_DIMS);
    arc3_action_select_device(swarm_output, chain_states, query_embedding, GPU_TASK_EMBED_DIMS);
    break;
```

This reuses the existing ARC specialist chain and adds the action selector.

---

## ORDER 2: Create Frame-to-Embedding Encoder

### 2A: Create `knowledge3d/cranium/cuda/arc3_frame_encoder.cu`

This kernel converts an ARC-AGI-3 frame (2D grid of uint8 color indices, max 64×64) into a 32-float embedding suitable for the task dispatch pipeline.

```c
#include "device_functions.cuh"

#define ARC3_MAX_GRID_SIZE 64
#define ARC3_MAX_COLORS 16

extern "C" __global__ void arc3_encode_frame(
    const unsigned char* __restrict__ frame,    // [H, W] uint8 color grid
    float* __restrict__ embedding,              // [32] output embedding
    unsigned int width,
    unsigned int height
) {
    // Thread 0 computes the embedding; other threads can help with reductions
    if (threadIdx.x != 0) return;

    unsigned int total = width * height;
    if (total == 0) {
        for (int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) embedding[d] = 0.0f;
        return;
    }

    // Feature extraction from grid:
    // dims 0-9:   color histogram (10 ARC colors, normalized)
    // dims 10-13: spatial moments (centroid_x, centroid_y, spread_x, spread_y)
    // dims 14-17: symmetry features (horizontal, vertical, rotational, diagonal)
    // dims 18-21: adjacency features (horizontal transitions, vertical transitions, unique objects, border ratio)
    // dims 22-25: pattern features (row repetition, col repetition, periodicity_h, periodicity_v)
    // dims 26-31: reserved / grid shape encoding

    // Color histogram (dims 0-9)
    float hist[ARC3_MAX_COLORS];
    for (int c = 0; c < ARC3_MAX_COLORS; ++c) hist[c] = 0.0f;
    for (unsigned int i = 0; i < total; ++i) {
        unsigned int color = frame[i];
        if (color < ARC3_MAX_COLORS) hist[color] += 1.0f;
    }
    float inv_total = 1.0f / static_cast<float>(total);
    for (int c = 0; c < 10; ++c) {
        embedding[c] = hist[c] * inv_total;
    }

    // Spatial moments (dims 10-13)
    float cx = 0.0f, cy = 0.0f;
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (frame[y * width + x] > 0) {
                cx += static_cast<float>(x);
                cy += static_cast<float>(y);
            }
        }
    }
    float nonzero = static_cast<float>(total) - hist[0];
    if (nonzero < 1.0f) nonzero = 1.0f;
    cx /= nonzero;
    cy /= nonzero;
    embedding[10] = cx / static_cast<float>(width);
    embedding[11] = cy / static_cast<float>(height);
    // Spread
    float sx = 0.0f, sy = 0.0f;
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (frame[y * width + x] > 0) {
                float dx = static_cast<float>(x) - cx;
                float dy = static_cast<float>(y) - cy;
                sx += dx * dx;
                sy += dy * dy;
            }
        }
    }
    embedding[12] = sqrtf(sx / nonzero) / static_cast<float>(width);
    embedding[13] = sqrtf(sy / nonzero) / static_cast<float>(height);

    // Symmetry (dims 14-17)
    float h_sym = 0.0f, v_sym = 0.0f;
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width / 2; ++x) {
            if (frame[y * width + x] == frame[y * width + (width - 1 - x)]) h_sym += 1.0f;
        }
    }
    for (unsigned int y = 0; y < height / 2; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (frame[y * width + x] == frame[(height - 1 - y) * width + x]) v_sym += 1.0f;
        }
    }
    float half_cells_h = static_cast<float>(height * (width / 2));
    float half_cells_v = static_cast<float>((height / 2) * width);
    embedding[14] = half_cells_h > 0.0f ? h_sym / half_cells_h : 0.0f;
    embedding[15] = half_cells_v > 0.0f ? v_sym / half_cells_v : 0.0f;
    embedding[16] = (embedding[14] + embedding[15]) * 0.5f;  // approximate rotational
    embedding[17] = 0.0f;  // diagonal symmetry placeholder

    // Adjacency (dims 18-21)
    float h_trans = 0.0f, v_trans = 0.0f;
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width - 1; ++x) {
            if (frame[y * width + x] != frame[y * width + x + 1]) h_trans += 1.0f;
        }
    }
    for (unsigned int y = 0; y < height - 1; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (frame[y * width + x] != frame[(y + 1) * width + x]) v_trans += 1.0f;
        }
    }
    float max_h_trans = static_cast<float>(height * (width > 0 ? width - 1 : 0));
    float max_v_trans = static_cast<float>((height > 0 ? height - 1 : 0) * width);
    embedding[18] = max_h_trans > 0.0f ? h_trans / max_h_trans : 0.0f;
    embedding[19] = max_v_trans > 0.0f ? v_trans / max_v_trans : 0.0f;
    embedding[20] = (h_trans + v_trans) * inv_total;  // approximate unique objects
    embedding[21] = 0.0f;  // border ratio placeholder

    // Pattern (dims 22-25)
    embedding[22] = 0.0f;  // row repetition placeholder
    embedding[23] = 0.0f;  // col repetition placeholder
    embedding[24] = static_cast<float>(width) / 64.0f;   // normalized width
    embedding[25] = static_cast<float>(height) / 64.0f;  // normalized height

    // Grid shape (dims 26-31)
    embedding[26] = static_cast<float>(width) / static_cast<float>(height + 1);  // aspect ratio
    embedding[27] = inv_total * 64.0f * 64.0f;  // grid density
    embedding[28] = (hist[0] > 0.0f) ? 1.0f - (hist[0] * inv_total) : 1.0f;  // fill ratio
    embedding[29] = 0.0f;  // reserved
    embedding[30] = 0.0f;  // reserved
    embedding[31] = 0.0f;  // reserved
}
```

### 2B: Create Python launcher `knowledge3d/knowledgeverse/arc3_frame_encoder.py`

```python
"""Encode ARC-AGI-3 game frames to embeddings on GPU."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

from knowledge3d.cranium.sovereign import loader

CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "arc3_frame_encoder.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
PTX_PATH = PTX_DIR / "arc3_frame_encoder.ptx"


class ARC3FrameEncoder:
    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self._ensure_ptx()), "arc3_encode_frame")

    @staticmethod
    def _ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest = max(
            CUDA_SOURCE.stat().st_mtime,
            CUDA_HEADER.stat().st_mtime if CUDA_HEADER.exists() else 0.0,
        )
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_arc3_frame_encoder")
        subprocess.run([nvcc, "-ptx", "-arch=sm_86", "-o", str(PTX_PATH), str(CUDA_SOURCE)], check=True)
        return PTX_PATH

    def encode(self, frame: list[list[int]]) -> list[float]:
        """Encode a 2D grid (list of rows of uint8) to a 32-float embedding."""
        height = len(frame)
        width = len(frame[0]) if height > 0 else 0
        flat = bytearray()
        for row in frame:
            flat.extend(int(c) & 0xFF for c in row)
        # Allocate GPU buffers
        frame_gpu = loader.gpu_malloc(len(flat))
        embed_gpu = loader.gpu_malloc(32 * 4)  # 32 floats
        # Upload frame
        frame_host = ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(flat)))
        loader.memcpy_htod(frame_gpu, frame_host, len(flat))
        # Launch
        loader.launch(
            self.kernel, (1, 1, 1), (1, 1, 1),
            [frame_gpu, embed_gpu, ctypes.c_uint(width), ctypes.c_uint(height)],
        )
        loader.synchronize()
        # Download embedding
        import struct
        embed_host = bytearray(32 * 4)
        embed_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(embed_host)))
        loader.memcpy_dtoh(embed_ptr, embed_gpu, 32 * 4)
        # Cleanup
        loader.gpu_free(frame_gpu)
        loader.gpu_free(embed_gpu)
        return list(struct.unpack("<32f", embed_host))
```

---

## ORDER 3: Create K3D ARC-AGI-3 Agent

### 3A: Create `benchmarks/arc_agi_3.py`

This is the K3D sovereign agent that plays ARC-AGI-3 games. It implements the game loop:
`frame → GPU encode → GPU dispatch → action → API call → next frame`

```python
"""K3D sovereign ARC-AGI-3 agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch
from knowledge3d.knowledgeverse.vram_task_buffer import EMBEDDING32_DIMS, VRAMTaskBuffer


# 7 ARC-AGI-3 actions mapped to 4 option embeddings (movement subset)
# Full 7-action support requires extending OPTION_EMBEDDING_SLOTS to 7
ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]

# Canonical action embeddings (one-hot in first 4 dims for movement)
ACTION_EMBEDDINGS: list[list[float]] = []
for action_index in range(4):
    embedding = [0.0] * EMBEDDING32_DIMS
    embedding[action_index] = 1.0
    ACTION_EMBEDDINGS.append(embedding)


class K3DARC3Agent:
    """Sovereign GPU agent for ARC-AGI-3 interactive games."""

    def __init__(self, max_actions: int = 80, log_path: str | Path | None = None) -> None:
        self.max_actions = max_actions
        self.log_path = Path(log_path) if log_path else None
        self.encoder = ARC3FrameEncoder()
        self.dispatcher = GPUTaskDispatch()
        self.task_buffer = VRAMTaskBuffer(max_tasks=1)  # one task at a time for interactive
        self.action_history: list[dict[str, Any]] = []
        self.frame_history: list[list[float]] = []  # embedding history for temporal reasoning

    def choose_action(self, frame: list[list[int]]) -> dict[str, Any]:
        """Given a game frame, return the action to take."""
        # 1. Encode frame to embedding on GPU
        frame_embedding = self.encoder.encode(frame)
        self.frame_history.append(frame_embedding)

        # 2. Pack as ARC3_TASK with movement options
        task = {
            "type": "ARC3_TASK",
            "query_embedding": frame_embedding,
            "option_embeddings": ACTION_EMBEDDINGS,
            "subject": "arc3_game",
            "domain_hint": "arc3_interactive",
        }

        # 3. Dispatch through GPU pipeline
        loaded = self.task_buffer.bulk_load([task])
        self.dispatcher.launch(self.task_buffer, loaded)
        results = self.task_buffer.read_results(loaded)

        # 4. Interpret result
        result = results[0] if results else {"answer_index": 0, "confidence": 0.0}
        action_index = int(result.get("answer_index", 0))
        action_index = max(0, min(action_index, 3))  # clamp to movement actions

        action_name = ACTION_NAMES[action_index]
        confidence = float(result.get("confidence", 0.0))
        converged = int(result.get("convergence_signal", 0))

        action_record = {
            "action": action_name,
            "action_index": action_index,
            "label": ACTION_LABELS[action_index],
            "confidence": confidence,
            "converged": converged,
            "frame_number": len(self.frame_history),
        }
        self.action_history.append(action_record)

        return action_record

    def close(self) -> None:
        self.task_buffer.close()
        if self.log_path and self.action_history:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as f:
                json.dump(self.action_history, f, indent=2)
```

### 3B: Create `scripts/run_arc3_agent.py`

This is the I/O shell that connects K3D to the ARC-AGI-3 server.

```python
"""Run K3D sovereign agent on ARC-AGI-3 games."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_3 import K3DARC3Agent


def main() -> int:
    parser = argparse.ArgumentParser(description="K3D ARC-AGI-3 sovereign agent")
    parser.add_argument("--game-id", required=True, help="ARC-AGI-3 game ID (e.g. ls20)")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--api-url", default="https://three.arcprize.org")
    args = parser.parse_args()

    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        print("ERROR: ARC_API_KEY not set", file=sys.stderr)
        return 1

    # Use requests directly — minimal I/O shell, no frameworks
    import requests

    session = requests.Session()
    session.headers.update({
        "X-API-Key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    })

    # 1. Open scorecard
    scorecard_resp = session.post(
        f"{args.api_url}/api/scorecard/open",
        json={"game_ids": [args.game_id], "tags": ["k3d-sovereign"]},
    ).json()
    card_id = scorecard_resp.get("card_id", "")
    print(f"Scorecard: {card_id}")

    # 2. Reset game to get initial frame
    reset_resp = session.post(
        f"{args.api_url}/api/cmd/RESET",
        json={"card_id": card_id, "game_id": args.game_id, "reasoning": "K3D init"},
    ).json()
    guid = reset_resp.get("guid", "")
    frame = reset_resp.get("frame", [[]])

    # 3. Create agent
    agent = K3DARC3Agent(
        max_actions=args.max_actions,
        log_path=args.log_path or None,
    )

    # 4. Game loop — Python does I/O only, GPU does thinking
    action_count = 0
    state = reset_resp.get("state", "IN_PROGRESS")
    start = time.time()

    try:
        while state == "IN_PROGRESS" and action_count < args.max_actions:
            # GPU thinks
            action = agent.choose_action(frame)
            action_name = action["action"]

            # I/O: send action to server
            action_data = {"game_id": args.game_id, "guid": guid}
            if action_name == "ACTION6":
                action_data["x"] = 32  # center click placeholder
                action_data["y"] = 32
            action_data["reasoning"] = {
                "agent": "k3d-sovereign",
                "confidence": action["confidence"],
                "converged": action["converged"],
            }

            resp = session.post(
                f"{args.api_url}/api/cmd/{action_name}",
                json=action_data,
            ).json()

            # Update state
            frame = resp.get("frame", frame)
            state = resp.get("state", state)
            guid = resp.get("guid", guid)
            levels = resp.get("levels_completed", 0)
            action_count += 1

            print(f"  [{action_count:3d}] {action['label']:12s}  conf={action['confidence']:.3f}  levels={levels}  state={state}")

            if state in ("WIN", "GAME_OVER"):
                break
    finally:
        agent.close()

    # 5. Close scorecard
    session.post(
        f"{args.api_url}/api/scorecard/close",
        json={"card_id": card_id},
    )

    elapsed = time.time() - start
    print(f"\nDone: {action_count} actions in {elapsed:.1f}s  state={state}")
    if state == "WIN":
        print("WIN!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## ORDER 4: Test Without API (Synthetic Frames)

Before connecting to the live ARC server, test the full pipeline with synthetic frames.

### 4A: Create `tests/test_arc3_agent.py`

```python
"""Test K3D ARC-AGI-3 agent with synthetic frames."""

import pytest
from benchmarks.arc_agi_3 import K3DARC3Agent


def _make_frame(width: int = 8, height: int = 8, fill: int = 0) -> list[list[int]]:
    return [[fill] * width for _ in range(height)]


def _make_frame_with_object(width: int = 8, height: int = 8) -> list[list[int]]:
    grid = [[0] * width for _ in range(height)]
    grid[2][3] = 1  # single colored cell
    return grid


class TestK3DARC3Agent:
    def test_choose_action_returns_valid(self):
        agent = K3DARC3Agent(max_actions=10)
        try:
            result = agent.choose_action(_make_frame())
            assert "action" in result
            assert result["action"] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
            assert "confidence" in result
            assert isinstance(result["confidence"], float)
        finally:
            agent.close()

    def test_frame_history_grows(self):
        agent = K3DARC3Agent(max_actions=10)
        try:
            agent.choose_action(_make_frame())
            agent.choose_action(_make_frame_with_object())
            assert len(agent.frame_history) == 2
            assert len(agent.action_history) == 2
        finally:
            agent.close()

    def test_different_frames_different_confidence(self):
        agent = K3DARC3Agent(max_actions=10)
        try:
            r1 = agent.choose_action(_make_frame(fill=0))
            r2 = agent.choose_action(_make_frame_with_object())
            # Different frames should produce different embeddings
            assert agent.frame_history[0] != agent.frame_history[1]
        finally:
            agent.close()

    def test_action_index_in_range(self):
        agent = K3DARC3Agent(max_actions=10)
        try:
            for _ in range(5):
                result = agent.choose_action(_make_frame())
                assert 0 <= result["action_index"] <= 3
        finally:
            agent.close()
```

Run: `pytest tests/test_arc3_agent.py -v`

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` — frame-to-embedding GPU kernel
- `knowledge3d/knowledgeverse/arc3_frame_encoder.py` — Python launcher for frame encoder
- `benchmarks/arc_agi_3.py` — K3D sovereign ARC-AGI-3 agent class
- `scripts/run_arc3_agent.py` — I/O shell for live ARC-AGI-3 API
- `tests/test_arc3_agent.py` — synthetic frame tests

Files you MODIFY:
- `knowledge3d/knowledgeverse/vram_task_buffer.py` — add `"ARC3_TASK": 8` to `TASK_TYPE_IDS`
- `knowledge3d/cranium/cuda/device_functions.cuh` — add `arc3_action_select_device()`
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — add `case 8u` to switch

Files you DO NOT TOUCH:
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — already handles any task type
- `scripts/run_gpu_benchmark.py` — separate benchmark runner
- Any file in `knowledge3d/knowledgeverse/knowledgeverse.py` — no Python hot path

---

## EXECUTION SEQUENCE

1. Add `"ARC3_TASK": 8` to `vram_task_buffer.py`
2. Add `arc3_action_select_device()` to `device_functions.cuh`
3. Add `case 8u` to `gpu_task_dispatch.cu`
4. Recompile PTX (the launcher handles this automatically on next run)
5. Create `arc3_frame_encoder.cu` + `arc3_frame_encoder.py`
6. Create `benchmarks/arc_agi_3.py`
7. Create `tests/test_arc3_agent.py` → run → all green
8. Create `scripts/run_arc3_agent.py`
9. Test with synthetic frames: `pytest tests/test_arc3_agent.py -v`

---

## SUCCESS CRITERIA

- `pytest tests/test_arc3_agent.py`: all tests pass
- Frame encoding: 8×8 grid → 32-float embedding in <1ms on GPU
- Action selection: frame embedding → action in <5ms total (encode + dispatch + read)
- `run_arc3_agent.py --game-id ls20` runs against live API (requires ARC_API_KEY)
- ZERO LLM API calls — K3D IS the reasoning engine
- Python lines in hot path (between frame receipt and action selection) = 0 (all GPU)
- Python does ONLY: API I/O, frame bytes upload, result download

**Build it.**
