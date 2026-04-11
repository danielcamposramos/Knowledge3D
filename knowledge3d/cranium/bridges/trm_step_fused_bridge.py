from __future__ import annotations

import ctypes
from pathlib import Path
import threading
import time
from typing import Any

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign.loader import (
    CUdeviceptr,
    get_function,
    gpu_free,
    gpu_malloc,
    launch,
    load_module_from_file,
    memcpy_dtoh,
    memcpy_htod,
    synchronize,
)


TRM_EVENT_PERCEPTION_STIMULUS = 0
TRM_EVENT_COLLISION = 1
TRM_EVENT_INTERACTION = 2
TRM_EVENT_TIMER = 3
TRM_EVENT_IO = 4
TRM_EVENT_INTERNAL = 5
TRM_EVENT_WAKEUP = 6

TRM_STATE_SLEEP = 0
TRM_STATE_IDLE = 1
TRM_STATE_PERCEIVING = 2
TRM_STATE_NAVIGATING = 3
TRM_STATE_REASONING = 4
TRM_STATE_ACTING = 5
TRM_STATE_HANDLING_QUERY = 6

TRM_EVENT_RING_CAPACITY = 4096
TRM_DEFAULT_DELTA_TIME = 0.02
TRM_DEFAULT_TICK_HZ = 50.0
TRM_DIMS = 512
TRM_HIDDEN_DIMS = 1024
TRM_WORKSPACE_FLOATS_PER_ENTITY = 4096
GALAXY_STAR_RECORD_BYTES = 400
GALAXY_EMBEDDING_DIMS = 64
GALAXY_INVALID_STAR_INDEX = 0xFFFFFFFF
ACTION_BUFFER_BYTES = 288
ACTION_BUFFER_WORDS = 72
ACTION_NAV_MOVE = 0x00
ACTION_NAV_LOOK = 0x01
ACTION_DIALOGUE = 0x02
ACTION_WRITE_MEM = 0x03
ACTION_UPDATE_TABLET = 0x04
ACTION_NO_ACTION = 0xFF
ACTION_WORD_OFFSET_MUTATION_TYPE = 60
ACTION_WORD_OFFSET_TABLET_DATA = 61


class _GPUEventStruct(ctypes.Structure):
    _fields_ = [
        ("entity_id", ctypes.c_uint32),
        ("event_type", ctypes.c_uint8),
        ("priority", ctypes.c_uint8),
        ("pad", ctypes.c_uint16),
        ("payload", ctypes.c_uint64),
    ]


class _TRMStateMachineStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("state_stack", ctypes.c_uint8 * 4),
        ("stack_depth", ctypes.c_uint8),
        ("current_state", ctypes.c_uint8),
        ("state_flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("idle_accumulator", ctypes.c_float),
        ("state_entry_tick", ctypes.c_uint64),
        ("deferred_event_mask", ctypes.c_uint32),
        ("interrupt_priority_level", ctypes.c_uint32),
        ("last_tick", ctypes.c_uint32),
    ]


class _EntityHotPathStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("star_table_idx", ctypes.c_uint32),
        ("physics_body_id", ctypes.c_uint32),
        ("behavior_rpn_addr", ctypes.c_uint64),
        ("house_x", ctypes.c_float),
        ("house_y", ctypes.c_float),
        ("house_z", ctypes.c_float),
        ("sleep_state", ctypes.c_uint8),
        ("faction", ctypes.c_uint8),
        ("ai_tier", ctypes.c_uint8),
        ("perception_flags", ctypes.c_uint8),
        ("perception_radius", ctypes.c_float),
        ("last_player_dist", ctypes.c_float),
        ("awareness", ctypes.c_float),
        ("blackboard_star_id", ctypes.c_uint32),
        ("meta_rule_addr", ctypes.c_uint32),
        ("cranial_origin", ctypes.c_float * 3),
        ("gaze_yaw", ctypes.c_float),
        ("gaze_pitch", ctypes.c_float),
        ("gaze_fov", ctypes.c_float),
        ("attention_entity_id", ctypes.c_uint32),
        ("motor_output", ctypes.c_float * 3),
        ("current_goal_star", ctypes.c_uint32),
    ]


assert ctypes.sizeof(_GPUEventStruct) == 16
assert ctypes.sizeof(_TRMStateMachineStruct) == 32
assert ctypes.sizeof(_EntityHotPathStruct) == 96


def _ptr_value(ptr: Any) -> int:
    return int(ptr.value) if hasattr(ptr, "value") else int(ptr)


def _device_offset(ptr: CUdeviceptr, offset_bytes: int) -> CUdeviceptr:
    return CUdeviceptr(_ptr_value(ptr) + int(offset_bytes))


def _coerce_galaxy_id(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        text = str(value or "").strip()
        if not text:
            return 0
        return abs(hash(text)) & 0xFFFFFFFF


def _default_entity() -> dict[str, Any]:
    return {
        "star_id": "entity:trm:primary",
        "star_table_idx": 0,
        "physics_body_id": 0,
        "behavior_rpn_addr": 0,
        "house_x": 0.0,
        "house_y": 1.75,
        "house_z": 0.0,
        "sleep_state": TRM_STATE_IDLE,
        "faction": 0,
        "ai_tier": 0,
        "perception_flags": 0x1,
        "perception_radius": 30.0,
        "last_player_dist": 999.0,
        "awareness": 0.0,
        "blackboard_star_id": 0,
        "meta_rule_addr": 0,
        "cranial_origin": [0.0, 1.6, 0.0],
        "gaze_yaw": 0.0,
        "gaze_pitch": 0.0,
        "gaze_fov": 0.7853981633974483,
        "attention_entity_id": 0,
        "motor_output": [0.0, 0.0, 0.0],
        "current_goal_star": 0,
    }


class TRMStepFusedBridge:
    """Single fused tick bridge for embodied TRM execution."""

    def __init__(self, *, arch: str = "sm_86") -> None:
        self.arch = str(arch)
        self._base_dir = Path(__file__).resolve().parent.parent
        self._ptx_dir = self._base_dir / "ptx"
        self._cuda_dir = self._base_dir / "cuda"

        self._step_module = load_module_from_file(str(self._ensure_ptx(self._ptx_dir / "trm_step_fused.cu")))
        self._step_kernel = get_function(self._step_module, "trm_step_fused")

        self._state_module = load_module_from_file(str(self._ensure_ptx(self._ptx_dir / "trm_state_machine.cu")))
        self._state_kernel = get_function(self._state_module, "trm_state_machine_step")

        self._queue_module = load_module_from_file(str(self._ensure_ptx(self._cuda_dir / "gpu_event_queue.cu")))
        self._queue_reset_kernel = get_function(self._queue_module, "gpu_event_queue_reset")
        self._queue_enqueue_stress_kernel = get_function(self._queue_module, "gpu_event_queue_enqueue_stress")
        self._queue_enqueue_host_batch_kernel = get_function(self._queue_module, "gpu_event_queue_enqueue_host_batch")
        self._queue_dequeue_kernel = get_function(self._queue_module, "gpu_event_queue_dequeue_all")

        self._galaxy_decode_module = load_module_from_file(str(self._ensure_ptx(self._cuda_dir / "galaxy_answer_decode.cu")))
        self._galaxy_decode_top1_kernel = get_function(self._galaxy_decode_module, "galaxy_answer_decode_top1")

        self._d_event_ring = gpu_malloc(TRM_EVENT_RING_CAPACITY * ctypes.sizeof(_GPUEventStruct))
        self._d_event_head = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        self._d_event_tail = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        if _ptr_value(self._d_event_ring) % 16 != 0:
            raise AssertionError("GPU event ring base pointer is not 16-byte aligned")
        self._d_entities: CUdeviceptr | None = None
        self._d_state_machines: CUdeviceptr | None = None
        self._d_action_buffer: CUdeviceptr | None = None
        self._d_query_action_buffer_in: CUdeviceptr | None = None
        self._action_buffer_count = 0
        self._default_tick_buffers: dict[str, CUdeviceptr] = {}
        self._default_workspace_entity_count = 0
        self._query_runtime_buffers: dict[str, Any] = {}
        self._query_q_input_ptr: Any | None = None
        self._query_matryoshka_bridge: Any | None = None
        self._query_matryoshka_weight_ptr: Any | None = None
        self._galaxy_table_ptr: Any | None = None
        self._galaxy_star_count = 0
        self._galaxy_embedding_dims = GALAXY_EMBEDDING_DIMS
        self._galaxy_host_stars: list[dict[str, Any]] = []
        self._program_table_ptr: Any | None = None
        self._program_table_size_bytes = 0
        self._entity_count = 0
        self._tick_counter = 0
        self._tick_loop_count = 0
        self._tick_loop_last_error = ""
        self._ticking = False
        self._tick_stop_event = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self._launch_lock = threading.RLock()
        self._gpu_producers_active = False

        self.reset_event_ring()
        self.bind_entity_hot_paths([_default_entity()])
        self.bind_state_machines()

    def _ensure_ptx(self, source_path: Path) -> Path:
        target_path = self._ptx_dir / f"{source_path.stem}.ptx"
        dependencies = [
            source_path,
            self._cuda_dir / "trm_game_loop.cuh",
            self._cuda_dir / "trm_recursive_core.cuh",
            self._cuda_dir / "rpn_execute_device.cuh",
            self._cuda_dir / "device_functions.cuh",
            self._base_dir / "kernels" / "entity_hot_path.h",
            self._base_dir / "kernels" / "physics_body_soa.h",
            Path(__file__).resolve().parent.parent / "kernels" / "ptx_compiler.py",
        ]
        target_mtime = target_path.stat().st_mtime if target_path.exists() else 0.0
        newest_source = max(path.stat().st_mtime for path in dependencies if path.exists())
        if target_mtime >= newest_source:
            return target_path
        use_fast_math = source_path.name not in {"trm_step_fused.cu", "trm_recursive_fused.cu"}
        ptx_text = compile_cuda_file(
            source_path,
            arch=self.arch,
            use_fast_math=use_fast_math,
        )
        target_path.write_text(ptx_text, encoding="utf-8")
        return target_path

    def bind_entity_hot_paths(self, entity_hot_path_array: list[dict[str, Any]] | None = None) -> dict[str, int]:
        rows = list(entity_hot_path_array or []) or [_default_entity()]
        if self._d_entities is not None:
            gpu_free(self._d_entities)
            self._d_entities = None

        host_array_type = _EntityHotPathStruct * len(rows)
        host = host_array_type()
        for index, entry in enumerate(rows):
            cranial_origin = list(entry.get("cranial_origin", [0.0, 1.6, 0.0]) or [0.0, 1.6, 0.0])[:3]
            while len(cranial_origin) < 3:
                cranial_origin.append(0.0)
            motor_output = list(entry.get("motor_output", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])[:3]
            while len(motor_output) < 3:
                motor_output.append(0.0)
            host[index] = _EntityHotPathStruct(
                star_table_idx=int(entry.get("star_table_idx", index)),
                physics_body_id=int(entry.get("physics_body_id", 0)),
                behavior_rpn_addr=int(entry.get("behavior_rpn_addr", 0) or 0),
                house_x=float(entry.get("house_x", 0.0)),
                house_y=float(entry.get("house_y", 1.75)),
                house_z=float(entry.get("house_z", 0.0)),
                sleep_state=int(entry.get("sleep_state", TRM_STATE_IDLE)),
                faction=int(entry.get("faction", 0)),
                ai_tier=int(entry.get("ai_tier", 0)),
                perception_flags=int(entry.get("perception_flags", 0x1)),
                perception_radius=float(entry.get("perception_radius", 30.0)),
                last_player_dist=float(entry.get("last_player_dist", 999.0)),
                awareness=float(entry.get("awareness", 0.0)),
                blackboard_star_id=int(entry.get("blackboard_star_id", 0)),
                meta_rule_addr=int(entry.get("meta_rule_addr", 0)),
                cranial_origin=(ctypes.c_float * 3)(*map(float, cranial_origin)),
                gaze_yaw=float(entry.get("gaze_yaw", 0.0)),
                gaze_pitch=float(entry.get("gaze_pitch", 0.0)),
                gaze_fov=float(entry.get("gaze_fov", 0.7853981633974483)),
                attention_entity_id=int(entry.get("attention_entity_id", 0)),
                motor_output=(ctypes.c_float * 3)(*map(float, motor_output)),
                current_goal_star=int(entry.get("current_goal_star", 0)),
            )

        nbytes = ctypes.sizeof(host)
        self._d_entities = gpu_malloc(nbytes)
        memcpy_htod(self._d_entities, ctypes.cast(host, ctypes.c_void_p), nbytes)
        self._entity_count = len(rows)
        self._ensure_action_buffer_capacity(self._entity_count)
        return {"entity_count": self._entity_count, "stride_bytes": ctypes.sizeof(_EntityHotPathStruct)}

    def _ensure_action_buffer_capacity(self, entity_count: int) -> None:
        count = max(1, int(entity_count))
        if self._d_action_buffer is not None and self._action_buffer_count == count:
            self._initialize_action_buffer(count)
            return
        if self._d_action_buffer is not None:
            gpu_free(self._d_action_buffer)
            self._d_action_buffer = None
            self._action_buffer_count = 0

        self._d_action_buffer = gpu_malloc(count * ACTION_BUFFER_BYTES)
        self._action_buffer_count = count
        self._initialize_action_buffer(count)

    def _initialize_action_buffer(self, entity_count: int | None = None) -> None:
        if self._d_action_buffer is None:
            return
        count = max(1, int(entity_count if entity_count is not None else self._action_buffer_count))
        host_type = ctypes.c_uint32 * (count * ACTION_BUFFER_WORDS)
        host = host_type()
        for index in range(count):
            host[index * ACTION_BUFFER_WORDS] = ACTION_NO_ACTION
        memcpy_htod(self._d_action_buffer, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))

    @staticmethod
    def _zeroed_float_buffer(float_count: int) -> CUdeviceptr:
        count = max(1, int(float_count))
        host_type = ctypes.c_float * count
        host = host_type()
        ptr = gpu_malloc(count * ctypes.sizeof(ctypes.c_float))
        memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
        return ptr

    @staticmethod
    def _prepare_query_host_vector(query_embedding: Any) -> tuple[Any, bool]:
        raw_values = [] if query_embedding is None else list(query_embedding)
        prepared = (ctypes.c_float * TRM_DIMS)()
        limit = min(len(raw_values), TRM_DIMS)
        for index in range(limit):
            prepared[index] = ctypes.c_float(float(raw_values[index]))
        return prepared, len(raw_values) != TRM_DIMS

    @staticmethod
    def _zero_device_float_buffer(ptr: Any, float_count: int) -> None:
        count = max(1, int(float_count))
        host_type = ctypes.c_float * count
        host = host_type()
        memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))

    def _query_buffers_or_default(self) -> dict[str, Any]:
        required = ("q", "y", "z", "W1", "W2", "W3", "W4", "z_new", "y_new", "workspace")
        if all(name in self._query_runtime_buffers for name in required):
            return dict(self._query_runtime_buffers)
        self._ensure_default_tick_buffers(max(1, self._entity_count))
        return {
            "q": self._default_tick_buffers["q"],
            "y": self._default_tick_buffers["y"],
            "z": self._default_tick_buffers["z"],
            "W1": self._default_tick_buffers["W1"],
            "W2": self._default_tick_buffers["W2"],
            "W3": self._default_tick_buffers["W3"],
            "W4": self._default_tick_buffers["W4"],
            "z_new": self._default_tick_buffers["z_new"],
            "y_new": self._default_tick_buffers["y_new"],
            "workspace": self._default_tick_buffers["workspace"],
        }

    def bind_query_runtime_buffers(
        self,
        *,
        q_ptr: Any,
        y_ptr: Any,
        z_ptr: Any,
        W1_ptr: Any,
        W2_ptr: Any,
        W3_ptr: Any,
        W4_ptr: Any,
        z_new_ptr: Any,
        y_new_ptr: Any,
        workspace_ptr: Any,
        q_input_ptr: Any | None = None,
        matryoshka_bridge: Any | None = None,
        matryoshka_weight_ptr: Any | None = None,
    ) -> dict[str, int]:
        self._query_runtime_buffers = {
            "q": q_ptr,
            "y": y_ptr,
            "z": z_ptr,
            "W1": W1_ptr,
            "W2": W2_ptr,
            "W3": W3_ptr,
            "W4": W4_ptr,
            "z_new": z_new_ptr,
            "y_new": y_new_ptr,
            "workspace": workspace_ptr,
        }
        self._query_q_input_ptr = q_input_ptr
        self._query_matryoshka_bridge = matryoshka_bridge
        self._query_matryoshka_weight_ptr = matryoshka_weight_ptr
        return {"bound": 1, "vector_dim": TRM_DIMS, "workspace_floats": TRM_WORKSPACE_FLOATS_PER_ENTITY}

    def bind_galaxy_table(
        self,
        gpu_ptr: Any,
        star_count: int,
        *,
        embedding_dims: int = GALAXY_EMBEDDING_DIMS,
        host_stars: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        self._galaxy_table_ptr = gpu_ptr
        self._galaxy_star_count = max(0, int(star_count))
        self._galaxy_embedding_dims = max(1, min(GALAXY_EMBEDDING_DIMS, int(embedding_dims)))
        self._galaxy_host_stars = [dict(star) for star in list(host_stars or [])[: self._galaxy_star_count]]
        return {
            "bound": 1 if self._galaxy_table_ptr is not None and self._galaxy_star_count > 0 else 0,
            "star_count": int(self._galaxy_star_count),
            "embedding_dims": int(self._galaxy_embedding_dims),
        }

    def bind_program_table(
        self,
        gpu_ptr: Any,
        size_bytes: int,
    ) -> dict[str, int]:
        self._program_table_ptr = gpu_ptr
        self._program_table_size_bytes = max(0, int(size_bytes))
        return {
            "bound": 1 if self._program_table_ptr is not None and self._program_table_size_bytes > 0 else 0,
            "size_bytes": int(self._program_table_size_bytes),
        }

    def reset_query_state(self) -> None:
        buffers = self._query_buffers_or_default()
        for name, count in (
            ("y", TRM_DIMS),
            ("z", TRM_DIMS),
            ("z_new", TRM_DIMS),
            ("y_new", TRM_DIMS),
            ("workspace", TRM_WORKSPACE_FLOATS_PER_ENTITY),
        ):
            self._zero_device_float_buffer(buffers[name], count)
        if self._query_q_input_ptr is not None:
            self._zero_device_float_buffer(self._query_q_input_ptr, TRM_DIMS)

    def prepare_query(self, query_embedding: Any, *, readback: bool = False) -> list[float] | None:
        with self._launch_lock:
            return self._prepare_query_unlocked(query_embedding, readback=readback)

    def _prepare_query_unlocked(self, query_embedding: Any, *, readback: bool = False) -> list[float] | None:
        buffers = self._query_buffers_or_default()
        host, needs_projection = self._prepare_query_host_vector(query_embedding)
        if (
            needs_projection
            and self._query_q_input_ptr is not None
            and self._query_matryoshka_bridge is not None
            and self._query_matryoshka_weight_ptr is not None
        ):
            memcpy_htod(self._query_q_input_ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
            self._query_matryoshka_bridge.project_device(
                self._query_matryoshka_weight_ptr,
                self._query_q_input_ptr,
                buffers["q"],
                target_dim=TRM_DIMS,
                stride=TRM_DIMS,
            )
        else:
            memcpy_htod(buffers["q"], ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
        if readback:
            return self._read_device_float_vector(buffers["q"], TRM_DIMS)
        return None

    @staticmethod
    def _read_device_float_vector(ptr: Any, float_count: int) -> list[float]:
        count = max(1, int(float_count))
        host_type = ctypes.c_float * count
        host = host_type()
        memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), ptr, ctypes.sizeof(host))
        return [float(value) for value in host]

    def _ensure_query_action_buffer_in(self) -> CUdeviceptr:
        if self._d_query_action_buffer_in is None:
            self._d_query_action_buffer_in = gpu_malloc(ACTION_BUFFER_BYTES)
        return self._d_query_action_buffer_in

    def _copy_query_action_buffer_in(self, action_buffer_words: Any | None) -> int:
        if action_buffer_words is None:
            return 0
        words = list(action_buffer_words)
        if not words:
            return 0
        host_type = ctypes.c_uint32 * ACTION_BUFFER_WORDS
        host = host_type()
        for index, value in enumerate(words[:ACTION_BUFFER_WORDS]):
            host[index] = ctypes.c_uint32(int(value) & 0xFFFFFFFF)
        ptr = self._ensure_query_action_buffer_in()
        memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
        return _ptr_value(ptr)

    def _host_star_metadata(self, star_index: int) -> dict[str, Any]:
        index = int(star_index)
        if 0 <= index < len(self._galaxy_host_stars):
            return dict(self._galaxy_host_stars[index])
        return {}

    def _answer_decode_from_action_buffer(self, action_buffers: list[list[int]]) -> dict[str, Any]:
        first = list(action_buffers[0]) if action_buffers else []
        if len(first) < ACTION_BUFFER_WORDS:
            return {}
        if int(first[0]) != ACTION_UPDATE_TABLET or int(first[ACTION_WORD_OFFSET_TABLET_DATA + 5]) != 1:
            return {}
        top_index = int(first[ACTION_WORD_OFFSET_TABLET_DATA + 1])
        top_star = self._host_star_metadata(top_index)
        role_value = top_star.get("selection_role_id", top_star.get("selection_role", 0))
        try:
            role_id = int(role_value)
        except Exception:
            role_map = {
                "unknown": 0,
                "router": 1,
                "executor": 2,
                "validator": 3,
                "answer": 4,
                "anti_pattern": 5,
            }
            role_id = int(role_map.get(str(role_value).strip().lower(), 0))
        star_hash = (
            int(first[ACTION_WORD_OFFSET_TABLET_DATA + 2])
            | (int(first[ACTION_WORD_OFFSET_TABLET_DATA + 3]) << 32)
        )
        return {
            "answer_materialized": True,
            "failure_code": "",
            "top_star_idx": top_index,
            "top_star_score": 1.0,
            "top_star_galaxy_id": _coerce_galaxy_id(top_star.get("galaxy_id", 0)),
            "top_star_role": role_id,
            "top_star_hash": int(star_hash),
            "top_star": top_star,
            "tablet_result_value": int(ctypes.c_int32(first[ACTION_WORD_OFFSET_TABLET_DATA]).value),
        }

    def _decode_top_galaxy_star(self, y_new_ptr: Any) -> dict[str, Any]:
        if self._galaxy_table_ptr is None or self._galaxy_star_count <= 0:
            return {
                "answer_materialized": False,
                "failure_code": "galaxy_table_unbound",
            }
        d_index = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        d_score = gpu_malloc(ctypes.sizeof(ctypes.c_float))
        d_galaxy_id = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        d_role_id = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        d_star_hash = gpu_malloc(ctypes.sizeof(ctypes.c_uint64))
        try:
            launch(
                self._galaxy_decode_top1_kernel,
                grid=(1, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(_ptr_value(y_new_ptr)),
                    ctypes.c_uint64(_ptr_value(self._galaxy_table_ptr)),
                    ctypes.c_uint32(int(self._galaxy_star_count)),
                    ctypes.c_uint32(int(self._galaxy_embedding_dims)),
                    ctypes.c_uint32(1),
                    ctypes.c_uint64(_ptr_value(d_index)),
                    ctypes.c_uint64(_ptr_value(d_score)),
                    ctypes.c_uint64(_ptr_value(d_galaxy_id)),
                    ctypes.c_uint64(_ptr_value(d_role_id)),
                    ctypes.c_uint64(_ptr_value(d_star_hash)),
                ],
            )
            synchronize()

            host_index = ctypes.c_uint32()
            host_score = ctypes.c_float()
            host_galaxy_id = ctypes.c_uint32()
            host_role_id = ctypes.c_uint32()
            host_star_hash = ctypes.c_uint64()
            memcpy_dtoh(ctypes.byref(host_index), d_index, ctypes.sizeof(host_index))
            memcpy_dtoh(ctypes.byref(host_score), d_score, ctypes.sizeof(host_score))
            memcpy_dtoh(ctypes.byref(host_galaxy_id), d_galaxy_id, ctypes.sizeof(host_galaxy_id))
            memcpy_dtoh(ctypes.byref(host_role_id), d_role_id, ctypes.sizeof(host_role_id))
            memcpy_dtoh(ctypes.byref(host_star_hash), d_star_hash, ctypes.sizeof(host_star_hash))

            top_index = int(host_index.value)
            if top_index == GALAXY_INVALID_STAR_INDEX or top_index >= self._galaxy_star_count:
                return {
                    "answer_materialized": False,
                    "failure_code": "no_answer_eligible_star",
                    "top_star_idx": -1,
                    "top_star_score": float(host_score.value),
                }
            return {
                "answer_materialized": True,
                "failure_code": "",
                "top_star_idx": top_index,
                "top_star_score": float(host_score.value),
                "top_star_galaxy_id": int(host_galaxy_id.value),
                "top_star_role": int(host_role_id.value),
                "top_star_hash": int(host_star_hash.value),
                "top_star": self._host_star_metadata(top_index),
            }
        finally:
            gpu_free(d_index)
            gpu_free(d_score)
            gpu_free(d_galaxy_id)
            gpu_free(d_role_id)
            gpu_free(d_star_hash)

    def decode_y_new_top_star(self, y_new_ptr: Any | None = None) -> dict[str, Any]:
        buffers = self._query_buffers_or_default()
        return self._decode_top_galaxy_star(y_new_ptr if y_new_ptr is not None else buffers["y_new"])

    def _free_default_tick_buffers(self) -> None:
        for ptr in self._default_tick_buffers.values():
            gpu_free(ptr)
        self._default_tick_buffers = {}
        self._default_workspace_entity_count = 0

    def _ensure_default_tick_buffers(self, entity_count: int | None = None) -> None:
        count = max(1, int(entity_count if entity_count is not None else self._entity_count))
        if self._default_tick_buffers and self._default_workspace_entity_count >= count:
            return
        self._free_default_tick_buffers()
        matrix_floats = TRM_DIMS * TRM_HIDDEN_DIMS
        self._default_tick_buffers = {
            "q": self._zeroed_float_buffer(TRM_DIMS),
            "y": self._zeroed_float_buffer(TRM_DIMS),
            "z": self._zeroed_float_buffer(TRM_DIMS),
            "W1": self._zeroed_float_buffer(matrix_floats),
            "W2": self._zeroed_float_buffer(matrix_floats),
            "W3": self._zeroed_float_buffer(matrix_floats),
            "W4": self._zeroed_float_buffer(matrix_floats),
            "z_new": self._zeroed_float_buffer(TRM_DIMS),
            "y_new": self._zeroed_float_buffer(TRM_DIMS),
            "workspace": self._zeroed_float_buffer(count * TRM_WORKSPACE_FLOATS_PER_ENTITY),
        }
        self._default_workspace_entity_count = count

    def _resolve_tick_pointers(
        self,
        *,
        q_ptr: Any | None,
        y_ptr: Any | None,
        z_ptr: Any | None,
        W1_ptr: Any | None,
        W2_ptr: Any | None,
        W3_ptr: Any | None,
        W4_ptr: Any | None,
        z_new_ptr: Any | None,
        y_new_ptr: Any | None,
        workspace_ptr: Any | None,
        entity_count: int,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        values = [q_ptr, y_ptr, z_ptr, W1_ptr, W2_ptr, W3_ptr, W4_ptr, z_new_ptr, y_new_ptr, workspace_ptr]
        if all(value is None for value in values):
            self._ensure_default_tick_buffers(entity_count)
            return (
                self._default_tick_buffers["q"],
                self._default_tick_buffers["y"],
                self._default_tick_buffers["z"],
                self._default_tick_buffers["W1"],
                self._default_tick_buffers["W2"],
                self._default_tick_buffers["W3"],
                self._default_tick_buffers["W4"],
                self._default_tick_buffers["z_new"],
                self._default_tick_buffers["y_new"],
                self._default_tick_buffers["workspace"],
            )
        if any(value is None for value in values):
            raise ValueError("trm_step_fused_tick_requires_all_or_no_tensor_pointers")
        return (q_ptr, y_ptr, z_ptr, W1_ptr, W2_ptr, W3_ptr, W4_ptr, z_new_ptr, y_new_ptr, workspace_ptr)

    def bind_state_machines(self, state_machines: list[dict[str, Any]] | None = None) -> dict[str, int]:
        count = max(1, int(self._entity_count))
        rows = list(state_machines or [])
        if self._d_state_machines is not None:
            gpu_free(self._d_state_machines)
            self._d_state_machines = None

        host_array_type = _TRMStateMachineStruct * count
        host = host_array_type()
        for index in range(count):
            row = rows[index] if index < len(rows) else {}
            current_state = int(row.get("current_state", TRM_STATE_IDLE))
            stack = list(row.get("state_stack", []))[:4]
            while len(stack) < 4:
                stack.append(0)
            host[index] = _TRMStateMachineStruct(
                state_stack=(ctypes.c_uint8 * 4)(*map(int, stack)),
                stack_depth=int(row.get("stack_depth", 0)),
                current_state=current_state,
                state_flags=int(row.get("state_flags", 0)),
                reserved=int(row.get("owner_entity_id", index)) & 0xFF,
                idle_accumulator=float(row.get("idle_accumulator", 0.0)),
                state_entry_tick=int(row.get("state_entry_tick", 0)),
                deferred_event_mask=int(row.get("deferred_event_mask", 0)),
                interrupt_priority_level=int(row.get("interrupt_priority_level", 0)),
                last_tick=int(row.get("last_tick", 0)),
            )

        nbytes = ctypes.sizeof(host)
        self._d_state_machines = gpu_malloc(nbytes)
        memcpy_htod(self._d_state_machines, ctypes.cast(host, ctypes.c_void_p), nbytes)
        return {"entity_count": count, "stride_bytes": ctypes.sizeof(_TRMStateMachineStruct)}

    def reset_event_ring(self) -> None:
        launch(
            self._queue_reset_kernel,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            params=[
                ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                ctypes.c_uint64(_ptr_value(self._d_event_head)),
                ctypes.c_uint64(_ptr_value(self._d_event_tail)),
            ],
        )
        synchronize()

    def reset_runtime(self, *, current_state: int = TRM_STATE_IDLE) -> None:
        entities = self.read_entity_hot_paths() or [_default_entity()]
        for entity in entities:
            entity["sleep_state"] = int(current_state)
        self.reset_event_ring()
        self.bind_entity_hot_paths(entities)
        self.bind_state_machines(
            [
                {
                    "current_state": int(current_state),
                    "stack_depth": 0,
                    "idle_accumulator": 0.0,
                    "state_entry_tick": 0,
                    "deferred_event_mask": 0,
                    "interrupt_priority_level": 0,
                    "last_tick": 0,
                    "owner_entity_id": index,
                }
                for index in range(max(1, len(entities)))
            ]
        )

    def enqueue_event(
        self,
        *,
        entity_id: int,
        event_type: int,
        priority: int = 0,
        payload: int = 0,
    ) -> None:
        self.enqueue_events(
            [
                {
                    "entity_id": int(entity_id),
                    "event_type": int(event_type),
                    "priority": int(priority),
                    "payload": int(payload),
                }
            ]
        )

    def enqueue_query(self, *, entity_id: int = 0, payload: int = 0) -> None:
        self.enqueue_event(
            entity_id=entity_id,
            event_type=TRM_EVENT_IO,
            priority=255,
            payload=payload,
        )

    def enqueue_events(self, events: list[dict[str, Any]]) -> list[int]:
        if self._gpu_producers_active:
            raise AssertionError("Host event injection is disabled while GPU producers are active")
        batch_size = len(events)
        if batch_size <= 0:
            return []

        host_array_type = _GPUEventStruct * batch_size
        host = host_array_type()
        for index, event in enumerate(events):
            host[index] = _GPUEventStruct(
                entity_id=int(event.get("entity_id", 0)),
                event_type=int(event.get("event_type", TRM_EVENT_INTERNAL)),
                priority=int(event.get("priority", 0)),
                pad=0,
                payload=int(event.get("payload", 0)),
            )

        d_batch = gpu_malloc(ctypes.sizeof(host))
        d_results = gpu_malloc(batch_size * ctypes.sizeof(ctypes.c_uint32))
        try:
            memcpy_htod(d_batch, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
            launch(
                self._queue_enqueue_host_batch_kernel,
                grid=((batch_size + 31) // 32, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                    ctypes.c_uint64(_ptr_value(self._d_event_head)),
                    ctypes.c_uint64(_ptr_value(self._d_event_tail)),
                    ctypes.c_uint64(_ptr_value(d_batch)),
                    ctypes.c_uint32(batch_size),
                    ctypes.c_uint64(_ptr_value(d_results)),
                ],
            )
            synchronize()

            host_results = (ctypes.c_uint32 * batch_size)()
            memcpy_dtoh(ctypes.cast(host_results, ctypes.c_void_p), d_results, ctypes.sizeof(host_results))
            pushed = [int(value) for value in list(host_results)]
            if not all(value == 1 for value in pushed):
                raise RuntimeError(f"trm_event_ring_full: only {sum(pushed)}/{batch_size} events enqueued")
            return pushed
        finally:
            gpu_free(d_batch)
            gpu_free(d_results)

    def set_gpu_producers_active(self, active: bool = True) -> None:
        self._gpu_producers_active = bool(active)

    def read_state_machines(self) -> list[dict[str, Any]]:
        if self._d_state_machines is None:
            return []
        count = max(1, int(self._entity_count))
        host_array_type = _TRMStateMachineStruct * count
        host = host_array_type()
        memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), self._d_state_machines, ctypes.sizeof(host))
        rows: list[dict[str, Any]] = []
        for row in host:
            rows.append(
                {
                    "state_stack": [int(value) for value in list(row.state_stack)],
                    "stack_depth": int(row.stack_depth),
                    "current_state": int(row.current_state),
                    "state_flags": int(row.state_flags),
                    "owner_entity_id": int(row.reserved),
                    "idle_accumulator": float(row.idle_accumulator),
                    "state_entry_tick": int(row.state_entry_tick),
                    "deferred_event_mask": int(row.deferred_event_mask),
                    "interrupt_priority_level": int(row.interrupt_priority_level),
                    "last_tick": int(row.last_tick),
                }
            )
        return rows

    def read_entity_hot_paths(self) -> list[dict[str, Any]]:
        if self._d_entities is None or self._entity_count <= 0:
            return []
        host_array_type = _EntityHotPathStruct * self._entity_count
        host = host_array_type()
        memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), self._d_entities, ctypes.sizeof(host))
        rows: list[dict[str, Any]] = []
        for row in host:
            rows.append(
                {
                    "star_table_idx": int(row.star_table_idx),
                    "physics_body_id": int(row.physics_body_id),
                    "behavior_rpn_addr": int(row.behavior_rpn_addr),
                    "house_x": float(row.house_x),
                    "house_y": float(row.house_y),
                    "house_z": float(row.house_z),
                    "sleep_state": int(row.sleep_state),
                    "faction": int(row.faction),
                    "ai_tier": int(row.ai_tier),
                    "perception_flags": int(row.perception_flags),
                    "perception_radius": float(row.perception_radius),
                    "last_player_dist": float(row.last_player_dist),
                    "awareness": float(row.awareness),
                    "blackboard_star_id": int(row.blackboard_star_id),
                    "meta_rule_addr": int(row.meta_rule_addr),
                    "cranial_origin": [float(value) for value in list(row.cranial_origin)],
                    "gaze_yaw": float(row.gaze_yaw),
                    "gaze_pitch": float(row.gaze_pitch),
                    "gaze_fov": float(row.gaze_fov),
                    "attention_entity_id": int(row.attention_entity_id),
                    "motor_output": [float(value) for value in list(row.motor_output)],
                    "current_goal_star": int(row.current_goal_star),
                }
            )
        return rows

    def step_state_machine(
        self,
        *,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick: int | None = None,
    ) -> list[dict[str, Any]]:
        tick_value = self._next_tick(tick)
        if self._d_entities is None or self._d_state_machines is None:
            raise RuntimeError("trm_state_machine_unbound")
        launch(
            self._state_kernel,
            grid=(max(1, int(self._entity_count)), 1, 1),
            block=(1, 1, 1),
            params=[
                ctypes.c_uint64(_ptr_value(self._d_entities)),
                ctypes.c_uint64(_ptr_value(self._d_state_machines)),
                ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                ctypes.c_uint64(_ptr_value(self._d_event_head)),
                ctypes.c_uint64(_ptr_value(self._d_event_tail)),
                ctypes.c_uint32(int(self._entity_count)),
                ctypes.c_float(float(delta_time)),
                ctypes.c_uint64(int(tick_value)),
            ],
        )
        synchronize()
        return self.read_state_machines()

    def launch_tick(
        self,
        *,
        q_ptr: Any | None = None,
        y_ptr: Any | None = None,
        z_ptr: Any | None = None,
        W1_ptr: Any | None = None,
        W2_ptr: Any | None = None,
        W3_ptr: Any | None = None,
        W4_ptr: Any | None = None,
        z_new_ptr: Any | None = None,
        y_new_ptr: Any | None = None,
        workspace_ptr: Any | None = None,
        body_soa_ptr: Any | None = None,
        contact_soa_ptr: Any | None = None,
        body_count: int = 0,
        solver_iterations: int = 1,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick: int | None = None,
        max_steps: int = 6,
        epsilon: float = 1e-4,
        grid_x_override: int | None = None,
        entity_count_override: int | None = None,
    ) -> dict[str, float | int]:
        with self._launch_lock:
            return self._launch_tick_unlocked(
                q_ptr=q_ptr,
                y_ptr=y_ptr,
                z_ptr=z_ptr,
                W1_ptr=W1_ptr,
                W2_ptr=W2_ptr,
                W3_ptr=W3_ptr,
                W4_ptr=W4_ptr,
                z_new_ptr=z_new_ptr,
                y_new_ptr=y_new_ptr,
                workspace_ptr=workspace_ptr,
                body_soa_ptr=body_soa_ptr,
                contact_soa_ptr=contact_soa_ptr,
                body_count=body_count,
                solver_iterations=solver_iterations,
                delta_time=delta_time,
                tick=tick,
                max_steps=max_steps,
                epsilon=epsilon,
                grid_x_override=grid_x_override,
                entity_count_override=entity_count_override,
            )

    def _launch_tick_unlocked(
        self,
        *,
        q_ptr: Any | None = None,
        y_ptr: Any | None = None,
        z_ptr: Any | None = None,
        W1_ptr: Any | None = None,
        W2_ptr: Any | None = None,
        W3_ptr: Any | None = None,
        W4_ptr: Any | None = None,
        z_new_ptr: Any | None = None,
        y_new_ptr: Any | None = None,
        workspace_ptr: Any | None = None,
        body_soa_ptr: Any | None = None,
        contact_soa_ptr: Any | None = None,
        body_count: int = 0,
        solver_iterations: int = 1,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick: int | None = None,
        max_steps: int = 6,
        epsilon: float = 1e-4,
        grid_x_override: int | None = None,
        entity_count_override: int | None = None,
    ) -> dict[str, float | int]:
        if self._d_entities is None or self._d_state_machines is None:
            raise RuntimeError("trm_step_fused_unbound")

        tick_value = self._next_tick(tick)
        kernel_entity_count = max(1, int(entity_count_override if entity_count_override is not None else self._entity_count))
        grid_x = max(1, int(grid_x_override if grid_x_override is not None else kernel_entity_count))
        (
            q_ptr,
            y_ptr,
            z_ptr,
            W1_ptr,
            W2_ptr,
            W3_ptr,
            W4_ptr,
            z_new_ptr,
            y_new_ptr,
            workspace_ptr,
        ) = self._resolve_tick_pointers(
            q_ptr=q_ptr,
            y_ptr=y_ptr,
            z_ptr=z_ptr,
            W1_ptr=W1_ptr,
            W2_ptr=W2_ptr,
            W3_ptr=W3_ptr,
            W4_ptr=W4_ptr,
            z_new_ptr=z_new_ptr,
            y_new_ptr=y_new_ptr,
            workspace_ptr=workspace_ptr,
            entity_count=kernel_entity_count,
        )
        self._ensure_action_buffer_capacity(kernel_entity_count)
        d_steps = gpu_malloc(kernel_entity_count * ctypes.sizeof(ctypes.c_int32))
        d_drift = gpu_malloc(kernel_entity_count * ctypes.sizeof(ctypes.c_float))
        try:
            launch(
                self._step_kernel,
                grid=(grid_x, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(_ptr_value(q_ptr)),
                    ctypes.c_uint64(_ptr_value(y_ptr)),
                    ctypes.c_uint64(_ptr_value(z_ptr)),
                    ctypes.c_uint64(_ptr_value(W1_ptr)),
                    ctypes.c_uint64(_ptr_value(W2_ptr)),
                    ctypes.c_uint64(_ptr_value(W3_ptr)),
                    ctypes.c_uint64(_ptr_value(W4_ptr)),
                    ctypes.c_uint64(_ptr_value(z_new_ptr)),
                    ctypes.c_uint64(_ptr_value(y_new_ptr)),
                    ctypes.c_uint64(_ptr_value(workspace_ptr)),
                    ctypes.c_uint64(_ptr_value(body_soa_ptr) if body_soa_ptr is not None else 0),
                    ctypes.c_uint64(_ptr_value(contact_soa_ptr) if contact_soa_ptr is not None else 0),
                    ctypes.c_uint32(int(body_count)),
                    ctypes.c_uint32(int(solver_iterations)),
                    ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                    ctypes.c_uint64(_ptr_value(self._d_event_head)),
                    ctypes.c_uint64(_ptr_value(self._d_event_tail)),
                    ctypes.c_uint64(_ptr_value(self._d_state_machines)),
                    ctypes.c_uint64(_ptr_value(self._d_entities)),
                    ctypes.c_uint32(kernel_entity_count),
                    ctypes.c_float(float(delta_time)),
                    ctypes.c_uint64(int(tick_value)),
                    ctypes.c_int32(int(max_steps)),
                    ctypes.c_float(float(epsilon)),
                    ctypes.c_uint64(_ptr_value(d_steps)),
                    ctypes.c_uint64(_ptr_value(d_drift)),
                    ctypes.c_uint64(_ptr_value(self._galaxy_table_ptr) if self._galaxy_table_ptr is not None else 0),
                    ctypes.c_uint32(int(self._galaxy_star_count)),
                    ctypes.c_uint64(_ptr_value(self._d_action_buffer) if self._d_action_buffer is not None else 0),
                    ctypes.c_uint64(_ptr_value(self._program_table_ptr) if self._program_table_ptr is not None else 0),
                    ctypes.c_uint64(_ptr_value(self._d_query_action_buffer_in) if self._d_query_action_buffer_in is not None else 0),
                ],
            )
            synchronize()

            host_steps = (ctypes.c_int32 * kernel_entity_count)()
            host_drift = (ctypes.c_float * kernel_entity_count)()
            memcpy_dtoh(ctypes.cast(host_steps, ctypes.c_void_p), d_steps, ctypes.sizeof(host_steps))
            memcpy_dtoh(ctypes.cast(host_drift, ctypes.c_void_p), d_drift, ctypes.sizeof(host_drift))
            states = self.read_state_machines()
            entities = self.read_entity_hot_paths()
            first_state = states[0]
            first_entity = entities[0]
            entity_results = []
            for index in range(min(kernel_entity_count, len(states), len(entities))):
                entity_results.append(
                    {
                        "entity_idx": index,
                        "steps": int(host_steps[index]),
                        "drift": float(host_drift[index]),
                        "current_state": int(states[index]["current_state"]),
                        "sleep_state": int(entities[index]["sleep_state"]),
                    }
                )
            return {
                "tick": int(tick_value),
                "steps": int(host_steps[0]),
                "drift": float(host_drift[0]),
                "current_state": int(first_state["current_state"]),
                "sleep_state": int(first_entity["sleep_state"]),
                "entity_results": entity_results,
            }
        finally:
            gpu_free(d_steps)
            gpu_free(d_drift)

    @property
    def action_buffer_ptr(self) -> CUdeviceptr | None:
        return self._d_action_buffer

    def read_action_buffers_words(self, entity_count: int | None = None) -> list[list[int]]:
        if self._d_action_buffer is None:
            return []
        count = max(1, int(entity_count if entity_count is not None else self._entity_count))
        if count > self._action_buffer_count:
            raise ValueError(f"action_buffer_count_exceeded: requested {count}, allocated {self._action_buffer_count}")
        host_type = ctypes.c_uint32 * (count * ACTION_BUFFER_WORDS)
        host = host_type()
        memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), self._d_action_buffer, ctypes.sizeof(host))
        rows: list[list[int]] = []
        for index in range(count):
            start = index * ACTION_BUFFER_WORDS
            rows.append([int(value) for value in host[start : start + ACTION_BUFFER_WORDS]])
        return rows

    def run_query_tick(
        self,
        *,
        q_ptr: Any | None = None,
        y_ptr: Any | None = None,
        z_ptr: Any | None = None,
        W1_ptr: Any | None = None,
        W2_ptr: Any | None = None,
        W3_ptr: Any | None = None,
        W4_ptr: Any | None = None,
        z_new_ptr: Any | None = None,
        y_new_ptr: Any | None = None,
        workspace_ptr: Any | None = None,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick: int | None = None,
        max_steps: int = 6,
        epsilon: float = 1e-4,
        reset_runtime: bool = False,
    ) -> dict[str, float | int]:
        with self._launch_lock:
            if reset_runtime:
                self.reset_runtime(current_state=TRM_STATE_IDLE)
            self.enqueue_query(entity_id=0)
            return self._launch_tick_unlocked(
                q_ptr=q_ptr,
                y_ptr=y_ptr,
                z_ptr=z_ptr,
                W1_ptr=W1_ptr,
                W2_ptr=W2_ptr,
                W3_ptr=W3_ptr,
                W4_ptr=W4_ptr,
                z_new_ptr=z_new_ptr,
                y_new_ptr=y_new_ptr,
                workspace_ptr=workspace_ptr,
                delta_time=delta_time,
                tick=tick,
                max_steps=max_steps,
                epsilon=epsilon,
                grid_x_override=1,
                entity_count_override=1,
            )

    def submit_query(
        self,
        query_embedding: Any,
        *,
        action_buffer_words: Any | None = None,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick: int | None = None,
        max_steps: int = 6,
        epsilon: float = 1e-4,
        reset_runtime: bool = True,
    ) -> dict[str, Any]:
        with self._launch_lock:
            buffers = self._query_buffers_or_default()
            if reset_runtime:
                self.reset_runtime(current_state=TRM_STATE_IDLE)
            self.reset_query_state()
            projected_query = self._prepare_query_unlocked(query_embedding, readback=True) or []
            event_payload = self._copy_query_action_buffer_in(action_buffer_words)
            self.enqueue_query(entity_id=0, payload=event_payload)
            started = time.perf_counter()
            tick_result = self._launch_tick_unlocked(
                q_ptr=buffers["q"],
                y_ptr=buffers["y"],
                z_ptr=buffers["z"],
                W1_ptr=buffers["W1"],
                W2_ptr=buffers["W2"],
                W3_ptr=buffers["W3"],
                W4_ptr=buffers["W4"],
                z_new_ptr=buffers["z_new"],
                y_new_ptr=buffers["y_new"],
                workspace_ptr=buffers["workspace"],
                delta_time=delta_time,
                tick=tick,
                max_steps=max_steps,
                epsilon=epsilon,
                grid_x_override=1,
                entity_count_override=1,
            )
            y_new = self._read_device_float_vector(buffers["y_new"], TRM_DIMS)
            z_new = self._read_device_float_vector(buffers["z_new"], TRM_DIMS)
            action_buffers = self.read_action_buffers_words(entity_count=1)
            answer_decode = self._answer_decode_from_action_buffer(action_buffers) or self._decode_top_galaxy_star(buffers["y_new"])
            return {
                "status": "ok",
                "mode": "submit_query",
                "tick": int(tick_result.get("tick", 0)),
                "steps": int(tick_result.get("steps", 0)),
                "drift": float(tick_result.get("drift", 0.0)),
                "current_state": int(tick_result.get("current_state", 0)),
                "sleep_state": int(tick_result.get("sleep_state", 0)),
                "query_embedding_512": list(projected_query),
                "y_new_vector_512": list(y_new),
                "z_new_vector_512": list(z_new),
                "trm_latency_us": float((time.perf_counter() - started) * 1_000_000.0),
                "action_buffers": action_buffers,
                "ring_event_payload": int(event_payload),
                "tick_result": dict(tick_result),
                **answer_decode,
            }

    @property
    def tick_count(self) -> int:
        return int(self._tick_loop_count)

    @property
    def tick_loop_last_error(self) -> str:
        return str(self._tick_loop_last_error)

    def start_tick_loop(
        self,
        *,
        delta_time: float = TRM_DEFAULT_DELTA_TIME,
        tick_hz: float = TRM_DEFAULT_TICK_HZ,
    ) -> dict[str, Any]:
        if self._tick_thread is not None and self._tick_thread.is_alive():
            return self.tick_loop_status()
        self._tick_loop_last_error = ""
        self._ticking = True
        self._tick_stop_event.clear()
        self._tick_thread = threading.Thread(
            target=self._tick_loop_main,
            kwargs={"delta_time": float(delta_time), "tick_hz": float(tick_hz)},
            daemon=True,
            name="k3d-trm-fused-tick",
        )
        self._tick_thread.start()
        return self.tick_loop_status()

    def stop_tick_loop(self, *, timeout: float = 1.0) -> dict[str, Any]:
        self._ticking = False
        self._tick_stop_event.set()
        thread = self._tick_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        if thread is not None and not thread.is_alive():
            self._tick_thread = None
        return self.tick_loop_status()

    def tick_loop_status(self) -> dict[str, Any]:
        thread = self._tick_thread
        return {
            "ticking": bool(self._ticking and thread is not None and thread.is_alive()),
            "tick_count": int(self._tick_loop_count),
            "tick_counter": int(self._tick_counter),
            "last_error": str(self._tick_loop_last_error),
        }

    def _tick_loop_main(self, *, delta_time: float, tick_hz: float) -> None:
        period = 1.0 / max(1.0, float(tick_hz))
        while self._ticking and not self._tick_stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self.launch_tick(delta_time=float(delta_time))
                self._tick_loop_count += 1
            except Exception as exc:
                self._tick_loop_last_error = f"{type(exc).__name__}: {exc}"
                self._ticking = False
                self._tick_stop_event.set()
                break
            elapsed = time.perf_counter() - t0
            sleep_remaining = max(0.0, period - elapsed)
            if self._tick_stop_event.wait(sleep_remaining):
                break

    def stress_enqueue(
        self,
        *,
        thread_count: int,
        total_events: int,
        entity_id: int = 0,
        payload_base: int = 0,
    ) -> list[int]:
        total_event_count = int(total_events)
        if total_event_count <= 0:
            return []
        d_results = gpu_malloc(total_event_count * ctypes.sizeof(ctypes.c_uint32))
        try:
            launch(
                self._queue_enqueue_stress_kernel,
                grid=((int(thread_count) + 31) // 32, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                    ctypes.c_uint64(_ptr_value(self._d_event_head)),
                    ctypes.c_uint64(_ptr_value(self._d_event_tail)),
                    ctypes.c_uint32(int(entity_id)),
                    ctypes.c_uint32(total_event_count),
                    ctypes.c_uint64(int(payload_base)),
                    ctypes.c_uint64(_ptr_value(d_results)),
                ],
            )
            synchronize()
            host = (ctypes.c_uint32 * total_event_count)()
            memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), d_results, ctypes.sizeof(host))
            return [int(value) for value in list(host)]
        finally:
            gpu_free(d_results)

    def drain_events(self, *, max_events: int) -> list[dict[str, Any]]:
        if max_events <= 0:
            return []
        d_output = gpu_malloc(max_events * ctypes.sizeof(_GPUEventStruct))
        d_count = gpu_malloc(ctypes.sizeof(ctypes.c_uint32))
        try:
            launch(
                self._queue_dequeue_kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_uint64(_ptr_value(self._d_event_ring)),
                    ctypes.c_uint64(_ptr_value(self._d_event_head)),
                    ctypes.c_uint64(_ptr_value(self._d_event_tail)),
                    ctypes.c_uint64(_ptr_value(d_output)),
                    ctypes.c_uint32(int(max_events)),
                    ctypes.c_uint64(_ptr_value(d_count)),
                ],
            )
            synchronize()

            count = ctypes.c_uint32()
            memcpy_dtoh(ctypes.byref(count), d_count, ctypes.sizeof(count))
            output_array_type = _GPUEventStruct * int(count.value)
            host = output_array_type()
            if int(count.value) > 0:
                memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), d_output, ctypes.sizeof(host))
            rows: list[dict[str, Any]] = []
            for row in host:
                rows.append(
                    {
                        "entity_id": int(row.entity_id),
                        "event_type": int(row.event_type),
                        "priority": int(row.priority),
                        "payload": int(row.payload),
                    }
                )
            return rows
        finally:
            gpu_free(d_output)
            gpu_free(d_count)

    def _next_tick(self, tick: int | None) -> int:
        if tick is None:
            self._tick_counter += 1
            return int(self._tick_counter)
        self._tick_counter = max(int(self._tick_counter), int(tick))
        return int(tick)

    def cleanup(self) -> None:
        self.stop_tick_loop(timeout=1.0)
        self._free_default_tick_buffers()
        if self._d_state_machines is not None:
            gpu_free(self._d_state_machines)
            self._d_state_machines = None
        if self._d_entities is not None:
            gpu_free(self._d_entities)
            self._d_entities = None
        if self._d_action_buffer is not None:
            gpu_free(self._d_action_buffer)
            self._d_action_buffer = None
            self._action_buffer_count = 0
        if self._d_query_action_buffer_in is not None:
            gpu_free(self._d_query_action_buffer_in)
            self._d_query_action_buffer_in = None
        if self._d_event_ring is not None:
            gpu_free(self._d_event_ring)
            self._d_event_ring = None
        if self._d_event_head is not None:
            gpu_free(self._d_event_head)
            self._d_event_head = None
        if self._d_event_tail is not None:
            gpu_free(self._d_event_tail)
            self._d_event_tail = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = [
    "ACTION_BUFFER_BYTES",
    "ACTION_BUFFER_WORDS",
    "ACTION_DIALOGUE",
    "ACTION_NAV_LOOK",
    "ACTION_NAV_MOVE",
    "ACTION_NO_ACTION",
    "ACTION_UPDATE_TABLET",
    "ACTION_WRITE_MEM",
    "TRMStepFusedBridge",
    "TRM_EVENT_COLLISION",
    "TRM_EVENT_INTERACTION",
    "TRM_EVENT_INTERNAL",
    "TRM_EVENT_IO",
    "TRM_EVENT_PERCEPTION_STIMULUS",
    "TRM_EVENT_TIMER",
    "TRM_EVENT_WAKEUP",
    "TRM_STATE_ACTING",
    "TRM_STATE_HANDLING_QUERY",
    "TRM_STATE_IDLE",
    "TRM_STATE_NAVIGATING",
    "TRM_STATE_PERCEIVING",
    "TRM_STATE_REASONING",
    "TRM_STATE_SLEEP",
]
