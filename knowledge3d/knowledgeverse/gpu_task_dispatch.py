"""Sovereign launcher for the Phase E GPU task dispatch kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
from typing import Any

from knowledge3d.cranium.sovereign import loader

from .lesson_vram_ring import VRAMLessonRing
from .vram_task_buffer import EMBEDDING_DIMS, VRAMTaskBuffer


CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "gpu_task_dispatch.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
TRM_CORE_HEADER = CUDA_DIR / "trm_recursive_core.cuh"
PTX_PATH = PTX_DIR / "gpu_task_dispatch.ptx"


class GPUTaskDispatch:
    """Thin sovereign wrapper around the E1 batch dispatch kernel."""

    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self.ensure_ptx()), "gpu_task_dispatch")

    @staticmethod
    def ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest_source_mtime = max(
            CUDA_SOURCE.stat().st_mtime,
            CUDA_HEADER.stat().st_mtime if CUDA_HEADER.exists() else 0.0,
            TRM_CORE_HEADER.stat().st_mtime if TRM_CORE_HEADER.exists() else 0.0,
        )
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest_source_mtime:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_gpu_task_dispatch")
        subprocess.run(
            [
                nvcc,
                "-ptx",
                "-arch=sm_86",
                "--compiler-bindir",
                "/usr/bin/gcc-13",
                "-o",
                str(PTX_PATH),
                str(CUDA_SOURCE),
            ],
            check=True,
        )
        return PTX_PATH

    def launch(
        self,
        task_buffer: VRAMTaskBuffer,
        task_count: int,
        *,
        block_size: int = 128,
        brain_ptr=None,
        star_table=None,
        lesson_ring: VRAMLessonRing | None = None,
        trm_weight_buffers: dict[str, Any] | None = None,
    ) -> None:
        total = max(0, int(task_count))
        if total <= 0:
            return
        if brain_ptr is not None and total != 1:
            raise RuntimeError("persistent_brain_requires_single_task_dispatch")
        galaxy_ptr = getattr(star_table, "gpu_ptr", None) if star_table is not None else None
        galaxy_star_count = int(getattr(star_table, "star_count", 0) or 0) if star_table is not None else 0
        loader.launch(
            self.kernel,
            (total, 1, 1),
            (int(block_size), 1, 1),
            [
                task_buffer.input_buffer,
                task_buffer.output_buffer,
                ctypes.c_uint(total),
                brain_ptr if brain_ptr is not None else ctypes.c_void_p(),
                galaxy_ptr if galaxy_ptr is not None else ctypes.c_void_p(),
                ctypes.c_uint(int(max(0, galaxy_star_count))),
                getattr(star_table, "router_offsets_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "router_counts_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "executor_offsets_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "executor_counts_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "validator_offsets_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "validator_counts_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "anti_pattern_offsets_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "anti_pattern_counts_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                getattr(star_table, "ref_indices_ptr", ctypes.c_void_p()) if star_table is not None else ctypes.c_void_p(),
                ctypes.c_void_p(getattr((trm_weight_buffers or {}).get("W1"), "value", 0)),
                ctypes.c_void_p(getattr((trm_weight_buffers or {}).get("W2"), "value", 0)),
                ctypes.c_void_p(getattr((trm_weight_buffers or {}).get("W3"), "value", 0)),
                ctypes.c_void_p(getattr((trm_weight_buffers or {}).get("W4"), "value", 0)),
                lesson_ring.buffer if lesson_ring is not None else ctypes.c_void_p(),
                lesson_ring.counter if lesson_ring is not None else ctypes.c_void_p(),
                ctypes.c_uint(int(lesson_ring.capacity if lesson_ring is not None else 0)),
            ],
        )
        loader.synchronize()


def cpu_reference_dispatch(
    tasks: list[dict[str, Any]],
    *,
    brain_state: dict[str, Any] | None = None,
    galaxy_stars: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Reference implementation matching the current kernel semantics."""
    rows: list[dict[str, Any]] = []
    state = _coerce_brain_state(brain_state)
    for task in tasks:
        query = _pad_embedding(task.get("query_embedding") or [])
        goal_embedding = _pad_embedding(task.get("goal_embedding") or [])
        option_embeddings = [_pad_embedding(option) for option in list(task.get("option_embeddings") or [])[:7]]
        task_type = VRAMTaskBuffer.task_type_id(str(task.get("type", "")))
        task_family = VRAMTaskBuffer.task_type_name(task_type)
        enable_game2d_control = task_family == "GAME_2D" and _is_live_game2d_packet(query)
        thinking_budget = max(5, min(20, int(task.get("thinking_budget", 10))))
        ternary_signal = max(-1, min(1, int(task.get("ternary_signal", 0))))
        if state is not None and state["frame_count"] > 0:
            action_history = list(state["action_ring"])
            ternary_signal = int(state["ternary_signal"]) if int(state["ternary_signal"]) != 0 else ternary_signal
        else:
            action_history = [int(value) for value in list(task.get("action_history") or [])[:7]]

        bounded_options = min(len(option_embeddings), 7)
        if state is not None and state["frame_count"] > 0:
            reasoning_state = [
                _tanh((0.70 * float(state["reasoning"][index])) + (0.30 * query[index]))
                for index in range(EMBEDDING_DIMS)
            ]
            frame_delta = [query[index] - float(state["prev_frame"][index]) for index in range(EMBEDDING_DIMS)]
        else:
            reasoning_state = list(query)
            frame_delta = [0.0] * EMBEDDING_DIMS
        best_index = 0
        best_score = 0.0
        converged = 0
        iterations_used = 0
        goal_progress = 0.0

        for think_step in range(thinking_budget):
            chain_states, swarm_output, resonance_scores = _nine_chain_swarm_ref(
                reasoning_state,
                3 if think_step == 0 else 2,
            )
            if state is not None and state["frame_count"] > 0:
                for chain_index in range(9):
                    chain_states[chain_index] = [
                        _tanh((0.70 * float(state["chains"][chain_index][dim])) + (0.30 * chain_states[chain_index][dim]))
                        for dim in range(EMBEDDING_DIMS)
                    ]
                    resonance_scores[chain_index] = _cosine(chain_states[chain_index], reasoning_state)
            galaxy_knowledge, top_galaxy_star_indices, top_galaxy_star_scores = _navigate_galaxy_ref(
                reasoning_state,
                galaxy_stars,
                route_family=task_family,
            )
            swarm_output = _specialize_task_ref(
                task_type,
                swarm_output,
                chain_states,
                query,
                galaxy_knowledge,
                enable_game2d_control=enable_game2d_control,
            )
            if enable_game2d_control:
                swarm_output = _arc3_frame_delta_ref(swarm_output, frame_delta)
                swarm_output = _blend_with_galaxy_ref(
                    swarm_output,
                    galaxy_knowledge,
                    chain_states,
                    query,
                    0.55,
                    0.30,
                    0.05,
                    0.10,
                    [3, 4],
                )
            swarm_output = _cognitive_executive_ref(resonance_scores, chain_states, swarm_output)

            iterations_used = think_step + 1
            if bounded_options > 0:
                candidate_scores: list[float] = []
                for option_index, option_embedding in enumerate(option_embeddings[:bounded_options]):
                    score = _cosine(swarm_output, option_embedding)
                    if enable_game2d_control:
                        score += _arc3_action_prior_ref(option_index, query, ternary_signal)
                    for history_index, history_action in enumerate(action_history[:7]):
                        if int(history_action) == option_index:
                            score *= 0.25 + (0.12 * float(history_index))
                            break
                    candidate_scores.append(score)

                best_index = max(range(len(candidate_scores)), key=lambda index: candidate_scores[index])
                best_score = candidate_scores[best_index]
                min_threshold = 0.05 if ternary_signal < 0 else 0.1
                gap_threshold = 0.10 if ternary_signal < 0 else 0.15
                converged = int(_halting_gate_ref(resonance_scores, min_threshold, gap_threshold, 0.7))
            else:
                best_index = 0
                best_score = 0.0
                converged = 0
            reasoning_state = list(swarm_output)
            if converged and iterations_used >= 5:
                break

        if any(abs(value) > 1.0e-8 for value in goal_embedding):
            goal_progress = _goal_progress_ref(reasoning_state, goal_embedding, query)

        rows.append(
            {
                "answer_index": int(best_index),
                "confidence": float(best_score),
                "convergence_signal": int(converged),
                "iterations_used": int(iterations_used),
                "answer_text_hash": (task_type << 32) | int(best_index),
                "goal_progress": float(goal_progress),
                "top_galaxy_star_indices": [int(index) for index in top_galaxy_star_indices if int(index) >= 0],
                "top_galaxy_star_scores": [float(score) for score in top_galaxy_star_scores if float(score) > -1.0e29],
            }
        )

        if state is not None:
            state["reasoning"] = list(reasoning_state)
            state["prev_frame"] = list(query)
            state["chains"] = [list(row) for row in chain_states]
            state["specialist_trace"] = [float(value) for value in resonance_scores[:9]]
            state["frame_count"] = int(state["frame_count"]) + 1
            state["ternary_signal"] = _quantize_goal_progress_ref(goal_progress) if task_type == 8 else ternary_signal
            ring = list(state["action_ring"])
            ring.append(int(best_index))
            state["action_ring"] = ring[-7:]
    if state is not None and isinstance(brain_state, dict):
        brain_state.clear()
        brain_state.update(state)
    return rows


def _pad_embedding(values: Any) -> list[float]:
    row = [float(value) for value in list(values or [])[:EMBEDDING_DIMS]]
    if len(row) < EMBEDDING_DIMS:
        row.extend([0.0] * (EMBEDDING_DIMS - len(row)))
    return row


def _blank_brain_state() -> dict[str, Any]:
    return {
        "reasoning": [0.0] * EMBEDDING_DIMS,
        "chains": [[0.0] * EMBEDDING_DIMS for _ in range(9)],
        "prev_frame": [0.0] * EMBEDDING_DIMS,
        "action_ring": [],
        "ternary_signal": 0,
        "frame_count": 0,
        "specialist_trace": [0.0] * 9,
    }


def _is_live_game2d_packet(query: list[float]) -> bool:
    if len(query) < EMBEDDING_DIMS:
        return False
    reserved_signal = any(abs(float(query[index])) > 1.0e-6 for index in (10, 11, 12, 13, 28, 29, 31))
    high_lane_signal = any(abs(float(value)) > 1.0e-6 for value in query[32:])
    return bool(reserved_signal and high_lane_signal)


def _coerce_brain_state(brain_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if brain_state is None:
        return None
    state = _blank_brain_state()
    if isinstance(brain_state, dict):
        state["reasoning"] = _pad_embedding(brain_state.get("reasoning") or state["reasoning"])
        raw_chains = list(brain_state.get("chains") or state["chains"])[:9]
        state["chains"] = [_pad_embedding(row) for row in raw_chains] + ([[0.0] * EMBEDDING_DIMS] * max(0, 9 - len(raw_chains)))
        state["prev_frame"] = _pad_embedding(brain_state.get("prev_frame") or state["prev_frame"])
        state["action_ring"] = [int(value) for value in list(brain_state.get("action_ring") or [])[:7]]
        state["ternary_signal"] = max(-1, min(1, int(brain_state.get("ternary_signal", 0))))
        state["frame_count"] = max(0, int(brain_state.get("frame_count", 0)))
        raw_trace = [float(value) for value in list(brain_state.get("specialist_trace") or [])[:9]]
        raw_trace.extend([0.0] * (9 - len(raw_trace)))
        state["specialist_trace"] = raw_trace[:9]
    return state


def _tanh(value: float) -> float:
    import math

    return math.tanh(float(value))


def _blend_with_galaxy_ref(
    embedding: list[float],
    galaxy_knowledge: list[float],
    chain_states: list[list[float]],
    query: list[float],
    self_weight: float,
    galaxy_weight: float,
    context_weight: float,
    query_weight: float,
    chain_ids: list[int],
) -> list[float]:
    import math

    output: list[float] = []
    chain_count = max(1, len(chain_ids))
    for index in range(EMBEDDING_DIMS):
        context_mean = sum(chain_states[chain_id % 9][index] for chain_id in chain_ids) / float(chain_count)
        neighbor_mix = 0.5 * (galaxy_knowledge[(index - 1) % EMBEDDING_DIMS] + galaxy_knowledge[(index + 1) % EMBEDDING_DIMS])
        output.append(
            math.tanh(
                (self_weight * embedding[index]) +
                (galaxy_weight * galaxy_knowledge[index]) +
                (context_weight * context_mean) +
                (query_weight * query[index]) +
                (0.05 * neighbor_mix)
            )
        )
    return output


def _navigate_galaxy_ref(
    reasoning_state: list[float],
    galaxy_stars: list[dict[str, Any]] | None,
    *,
    route_family: str | None = None,
) -> tuple[list[float], list[int], list[float]]:
    if not galaxy_stars:
        return [0.0] * EMBEDDING_DIMS, [], []
    normalized_family = str(route_family or "").strip().upper()
    best: list[tuple[float, int]] = [(-1.0e30, -1) for _ in range(8)]
    for star_index in range(len(galaxy_stars)):
        star_family = str(galaxy_stars[star_index].get("route_family") or "").strip().upper()
        if normalized_family and star_family != normalized_family:
            continue
        star_embedding = _compose_galaxy_embedding_ref(galaxy_stars, star_index)
        similarity = _cosine(reasoning_state, star_embedding)
        worst_slot = min(range(8), key=lambda slot: best[slot][0])
        if similarity > best[worst_slot][0]:
            best[worst_slot] = (similarity, star_index)
    knowledge = [0.0] * EMBEDDING_DIMS
    total_weight = 0.0
    for similarity, star_index in best:
        if star_index < 0:
            continue
        weight = max(0.0, similarity)
        if weight <= 1.0e-8:
            continue
        total_weight += weight
        star_embedding = _compose_galaxy_embedding_ref(galaxy_stars, star_index)
        for index in range(EMBEDDING_DIMS):
            knowledge[index] += weight * star_embedding[index]
    if total_weight > 1.0e-6:
        knowledge = [value / total_weight for value in knowledge]
    ordered = sorted(
        ((float(similarity), int(star_index)) for similarity, star_index in best if int(star_index) >= 0),
        reverse=True,
    )
    return knowledge, [star_index for _, star_index in ordered], [similarity for similarity, _ in ordered]


def _goal_progress_ref(current_frame: list[float], goal_embedding: list[float], prev_frame: list[float]) -> float:
    import math

    current_dist = math.sqrt(sum((current_frame[index] - goal_embedding[index]) ** 2 for index in range(EMBEDDING_DIMS)))
    prev_dist = math.sqrt(sum((prev_frame[index] - goal_embedding[index]) ** 2 for index in range(EMBEDDING_DIMS)))
    if current_dist < 1.0e-4:
        return 1.0
    if current_dist < prev_dist:
        return 0.5
    if current_dist > prev_dist:
        return -0.5
    return 0.0


def _quantize_goal_progress_ref(progress: float) -> int:
    if progress > 1.0e-6:
        return 1
    if progress < -1.0e-6:
        return -1
    return 0


def _pseudo_random(chain_id: int, dim: int) -> float:
    seed = ((chain_id * 73856093) ^ (dim * 19349663)) & 0xFFFFFFFF
    seed ^= seed >> 13
    seed = (seed * 1274126177) & 0xFFFFFFFF
    seed ^= seed >> 16
    return ((seed & 0xFFFF) / 65535.0) - 0.5


def _nine_chain_swarm_ref(query: list[float], rounds: int) -> tuple[list[list[float]], list[float], list[float]]:
    import math

    chain_states: list[list[float]] = []
    for chain in range(9):
        if chain == 0:
            chain_states.append(list(query))
        else:
            chain_states.append([(0.90 * float(value)) + (0.05 * _pseudo_random(chain, dim)) for dim, value in enumerate(query)])

    resonance_scores = [0.0] * 9
    swarm_output = list(query)
    for _ in range(rounds):
        chain_states = [[math.tanh(value) for value in row] for row in chain_states]
        consensus = [sum(chain_states[c][d] for c in range(9)) / 9.0 for d in range(EMBEDDING_DIMS)]
        consensus_norm = math.sqrt(sum(value * value for value in consensus) + 1.0e-12)
        weight_sum = 0.0
        for chain in range(9):
            dot = sum(chain_states[chain][d] * consensus[d] for d in range(EMBEDDING_DIMS))
            state_norm = math.sqrt(sum(value * value for value in chain_states[chain]) + 1.0e-12)
            resonance = dot / ((state_norm * consensus_norm) + 1.0e-12)
            resonance_scores[chain] = resonance
            blend = 0.18 if chain == 8 else 0.12
            chain_states[chain] = [
                ((1.0 - blend) * chain_states[chain][d]) + (blend * consensus[d]) for d in range(EMBEDDING_DIMS)
            ]
            weight_sum += abs(resonance) + 1.0e-4
        if weight_sum <= 1.0e-12:
            weight_sum = 1.0
        swarm_output = [
            sum(((abs(resonance_scores[c]) + 1.0e-4) / weight_sum) * chain_states[c][d] for c in range(9))
            for d in range(EMBEDDING_DIMS)
        ]
    return chain_states, swarm_output, resonance_scores


def _specialize_task_ref(
    task_type: int,
    swarm_output: list[float],
    chain_states: list[list[float]],
    frame_query: list[float],
    galaxy_knowledge: list[float],
    *,
    enable_game2d_control: bool,
) -> list[float]:
    import math

    output = list(swarm_output)
    task_family = VRAMTaskBuffer.task_type_name(task_type)
    if task_family == "INTERACTION":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.60, 0.30, 0.10, 0.0, [3, 4])
        route = _cosine(output, galaxy_knowledge)
        output = [math.tanh(output[d] + (0.03 * route * galaxy_knowledge[d])) for d in range(EMBEDDING_DIMS)]
        output = [math.tanh((0.94 * output[d]) + (0.03 * output[d // 2]) + (0.03 * output[(d * 2) % EMBEDDING_DIMS])) for d in range(EMBEDDING_DIMS)]
    elif task_family == "GAME_2D" and enable_game2d_control:
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.45, 0.40, 0.10, 0.05, [3, 4])
        output = _arc3_action_select_ref(output, chain_states, frame_query)
    elif task_family == "GAME_2D":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.55, 0.30, 0.10, 0.05, list(range(9)))
    elif task_family == "MATH":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.50, 0.40, 0.10, 0.0, [0, 1, 2])
        output = [math.tanh((0.92 * output[d]) + (0.08 * ((chain_states[0][d] + chain_states[1][d] + chain_states[2][d]) / 3.0))) for d in range(EMBEDDING_DIMS)]
        route = _cosine(output, galaxy_knowledge)
        output = [math.tanh(output[d] + (0.03 * route * galaxy_knowledge[d])) for d in range(EMBEDDING_DIMS)]
    elif task_family == "QUESTION":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.48, 0.40, 0.08, 0.04, [0, 1, 2, 3])
        output = [math.tanh((0.92 * output[d]) + (0.08 * ((chain_states[0][d] + chain_states[1][d] + chain_states[2][d]) / 3.0))) for d in range(EMBEDDING_DIMS)]
        output = [math.tanh(output[d] + (0.05 * (output[d] - output[(d + EMBEDDING_DIMS - 1) % EMBEDDING_DIMS]))) for d in range(EMBEDDING_DIMS)]
    elif task_family == "CHAT":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.45, 0.40, 0.15, 0.0, list(range(9)))
        output = [math.tanh((0.90 * output[d]) + (0.10 * (sum(chain_states[c][d] for c in range(9)) / 9.0))) for d in range(EMBEDDING_DIMS)]
    elif task_family == "GENERAL":
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.50, 0.35, 0.10, 0.05, [7, 8])
        energy = sum(value * value for value in output)
        boost = 1.0 + (0.04 * min(1.0, max(0.0, energy / float(EMBEDDING_DIMS))))
        output = [value * boost for value in output]
        creative = chain_states[7]
        synthesis = chain_states[8]
        creative_score = _cosine(output, creative)
        synthesis_score = _cosine(output, synthesis)
        total = math.exp(creative_score) + math.exp(synthesis_score) + 1.0e-6
        creative_weight = math.exp(creative_score) / total
        synthesis_weight = math.exp(synthesis_score) / total
        output = [
            math.tanh((0.75 * output[d]) + (0.25 * ((creative_weight * creative[d]) + (synthesis_weight * synthesis[d]))))
            for d in range(EMBEDDING_DIMS)
        ]
    else:
        output = _blend_with_galaxy_ref(output, galaxy_knowledge, chain_states, frame_query, 0.55, 0.30, 0.10, 0.05, list(range(9)))
    return output


def _arc3_action_select_ref(embedding: list[float], chain_states: list[list[float]], frame_data: list[float]) -> list[float]:
    import math

    output = list(embedding)
    cx = frame_data[10] - 0.5
    cy = frame_data[11] - 0.5
    sx = frame_data[12]
    sy = frame_data[13]
    occupancy = max(0.0, min(1.0, frame_data[28]))
    spread_mag = math.sqrt((sx * sx) + (sy * sy))
    centeredness = 1.0 - max(0.0, min(1.0, (abs(cx) + abs(cy)) * 1.25))
    movement_need = max(0.0, min(1.0, (abs(cx) + abs(cy)) * 1.6 + (0.35 * spread_mag)))
    interaction_readiness = max(0.0, min(1.0, centeredness * occupancy * (0.45 + (0.55 * frame_data[31]))))
    click_readiness = max(0.0, min(1.0, interaction_readiness * (1.0 - spread_mag) * frame_data[29]))

    output[0] = math.tanh((0.35 * output[0]) + (1.75 * cx))
    output[1] = math.tanh((0.35 * output[1]) + (1.75 * cy))
    output[2] = math.tanh((0.20 * output[2]) + (1.10 * movement_need) - (0.65 * interaction_readiness))
    output[3] = math.tanh((0.20 * output[3]) + (0.95 * click_readiness) - (0.55 * movement_need))
    output[4] = math.tanh((0.30 * output[4]) + (0.90 * movement_need) + (0.15 * spread_mag))
    output[5] = math.tanh((0.30 * output[5]) + (1.25 * cx))
    output[6] = math.tanh((0.30 * output[6]) + (1.25 * cy))
    output[7] = math.tanh((0.40 * output[7]) + (0.80 * occupancy))
    for index in range(8, EMBEDDING_DIMS):
        spatial_delta = chain_states[3][index] - chain_states[4][index]
        output[index] = math.tanh((0.82 * output[index]) + (0.02 * abs(spatial_delta)) + (0.02 * frame_data[index]))
    output[10] = math.tanh((0.15 * output[10]) - (0.40 * movement_need))
    output[11] = math.tanh((0.15 * output[11]) + (0.30 * interaction_readiness))
    return output


def _arc3_action_prior_ref(option_index: int, frame_data: list[float], ternary_signal: int) -> float:
    import math

    cx = frame_data[10] - 0.5
    cy = frame_data[11] - 0.5
    sx = frame_data[12]
    sy = frame_data[13]
    spread_mag = math.sqrt((sx * sx) + (sy * sy))
    occupancy = max(0.0, min(1.0, frame_data[28]))
    centeredness = 1.0 - max(0.0, min(1.0, (abs(cx) + abs(cy)) * 1.25))
    movement_need = max(0.0, min(1.0, (abs(cx) + abs(cy)) * 1.6 + (0.35 * spread_mag)))
    interaction_readiness = max(0.0, min(1.0, centeredness * occupancy * (0.45 + (0.55 * frame_data[31]))))
    click_readiness = max(0.0, min(1.0, interaction_readiness * (1.0 - spread_mag) * frame_data[29]))
    undo_readiness = 0.85 if ternary_signal < 0 else 0.0

    if option_index == 0:
        return (0.55 * movement_need * max(0.0, -cy)) + (0.15 * movement_need)
    if option_index == 1:
        return (0.55 * movement_need * max(0.0, cy)) + (0.15 * movement_need)
    if option_index == 2:
        return (0.55 * movement_need * max(0.0, -cx)) + (0.15 * movement_need)
    if option_index == 3:
        return (0.55 * movement_need * max(0.0, cx)) + (0.15 * movement_need)
    if option_index == 4:
        return (0.35 * interaction_readiness) - (0.25 * movement_need)
    if option_index == 5:
        return (0.45 * click_readiness) - (0.20 * movement_need)
    if option_index == 6:
        return (0.70 * undo_readiness) - (0.0 if ternary_signal < 0 else 0.40)
    return 0.0


def _arc3_frame_delta_ref(embedding: list[float], frame_delta: list[float]) -> list[float]:
    import math

    output = list(embedding)
    delta_magnitude = math.sqrt(sum(float(value) * float(value) for value in frame_delta) + 1.0e-12)
    delta_signal = max(0.0, min(1.0, delta_magnitude * 2.0))
    for index in range(EMBEDDING_DIMS):
        explore = (0.08 * _pseudo_random(index, EMBEDDING_DIMS)) if delta_signal < 0.1 else 0.0
        output[index] = math.tanh(
            (0.92 * output[index]) +
            (0.06 * delta_signal * frame_delta[index]) +
            explore
        )
    return output


def _compose_galaxy_embedding_ref(galaxy_stars: list[dict[str, Any]], star_index: int) -> list[float]:
    if star_index < 0 or star_index >= len(galaxy_stars):
        return [0.0] * EMBEDDING_DIMS
    star = galaxy_stars[star_index]
    output = _pad_embedding(star.get("embedding") or [])
    refs = [int(value) for value in list(star.get("component_refs") or [])[:4] if int(value) >= 0]
    if not refs:
        return output
    base_weight = 0.60
    ref_weight = 0.40 / float(len(refs))
    output = [value * base_weight for value in output]
    for ref_index in refs:
        if ref_index >= len(galaxy_stars):
            continue
        ref_embedding = _pad_embedding(galaxy_stars[ref_index].get("embedding") or [])
        for index in range(EMBEDDING_DIMS):
            output[index] += ref_weight * ref_embedding[index]
    norm = sum(value * value for value in output) ** 0.5
    if norm > 1.0e-6:
        output = [value / norm for value in output]
    return output


def _cognitive_executive_ref(resonance_scores: list[float], chain_states: list[list[float]], embedding: list[float]) -> list[float]:
    import math

    logits = []
    for chain in range(9):
        chain_norm = math.sqrt(sum(value * value for value in chain_states[chain]) + 1.0e-12)
        logits.append(resonance_scores[chain] * (1.0 + math.log(chain_norm + 1.0)))
    max_logit = max(logits)
    weights = [math.exp(value - max_logit) for value in logits]
    denom = sum(weights) or 1.0
    return [
        math.tanh((0.80 * embedding[d]) + (0.20 * sum((weights[c] / denom) * chain_states[c][d] for c in range(9))))
        for d in range(EMBEDDING_DIMS)
    ]


def _halting_gate_ref(scores: list[float], min_threshold: float, gap_threshold: float, agreement_threshold: float) -> bool:
    import math

    top = scores[0]
    second = -1.0e30
    for score in scores[1:]:
        if score >= top:
            second = top
            top = score
        elif score > second:
            second = score
    agreement = sum(1 for value in scores if (top - value) <= 0.15)
    required_agreement = int(math.ceil(agreement_threshold * len(scores))) if agreement_threshold > 0 else 1
    return bool(top >= min_threshold and (top - second) >= gap_threshold and agreement >= required_agreement)


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a) + 1.0e-12)
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b) + 1.0e-12)
    if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["GPUTaskDispatch", "cpu_reference_dispatch"]
