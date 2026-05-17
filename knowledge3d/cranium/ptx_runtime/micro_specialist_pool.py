"""Tier-1 micro-specialist pool for the sovereign swarm.

This pool pre-allocates LightweightRPNEngine shards and exposes fixed slot ids
that swarm workers can claim for cheap bounded sub-computations. The pool is a
GPU service: Python only manages slot ownership and batches work submission.
"""

from __future__ import annotations

import ctypes
import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from knowledge3d.cranium.sovereign import loader

from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine


_TOKEN_TO_OPCODE: dict[str, int] = {
    "+": 10,
    "add": 10,
    "-": 11,
    "sub": 11,
    "*": 12,
    "mul": 12,
    "/": 13,
    "div": 13,
    "neg": 15,
    "sqrt": 20,
    "exp": 21,
    "log": 22,
    "sin": 24,
    "cos": 25,
    "tan": 26,
    "gt": 40,
    "lt": 42,
    "eq": 44,
    "max": 46,
    "min": 47,
    "dup": 50,
    "swap": 51,
    "drop": 52,
}


@dataclass(frozen=True)
class MicroSlotBinding:
    slot_id: int
    engine_index: int
    instance_id: int


class MicroSpecialistPool:
    """Thread-safe Tier-1 slot pool sized from GPU SM capacity."""

    DEFAULT_RESERVE_RATIO = 0.66
    CORES_PER_SM = 10
    FALLBACK_SM_COUNT = 46

    def __init__(
        self,
        *,
        pool_size: int,
        engine_factory: type[LightweightRPNEngine] = LightweightRPNEngine,
        reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    ) -> None:
        self.pool_size = max(1, int(pool_size))
        self.reserve_ratio = float(reserve_ratio)
        self.engine_factory = engine_factory
        self.slots_per_engine = int(engine_factory.MAX_INSTANCES)
        self._lock = threading.Lock()
        self._peak_utilization = 0
        self._engines: list[LightweightRPNEngine] = []
        self._bindings: list[MicroSlotBinding] = []
        self._free_slots: list[int] = []
        self._in_use: set[int] = set()
        self._sequential_engine: LightweightRPNEngine | None = None
        self._tiered_fallback_engine: Any = None
        self._tier1_empty_stack_fallbacks = 0

        engine_count = max(1, math.ceil(float(self.pool_size) / float(self.slots_per_engine)))
        for engine_index in range(engine_count):
            engine = engine_factory()
            self._engines.append(engine)
            for instance_id in range(self.slots_per_engine):
                slot_id = len(self._bindings)
                self._bindings.append(
                    MicroSlotBinding(
                        slot_id=slot_id,
                        engine_index=engine_index,
                        instance_id=instance_id,
                    )
                )
                if len(self._free_slots) < self.pool_size:
                    self._free_slots.append(slot_id)
        self._sequential_engine = engine_factory()

    @classmethod
    def query_sm_count(cls, gpu_id: int = 0) -> int:
        """Query GPU SM count via libcuda without pulling CuPy into the hot path."""
        try:
            try:
                nvcuda = ctypes.CDLL("libcuda.so.1")
            except OSError:
                nvcuda = ctypes.CDLL("libcuda.so")
            # Sovereign invariant: single CUDA context from loader.
            # See TEMP/CLAUDE_SINGLE_CONTEXT_LIVING_AI_SPEC_04.18.2026.md
            loader._ensure_init()
            device = ctypes.c_int()
            if int(nvcuda.cuDeviceGet(ctypes.byref(device), int(gpu_id))) != 0:
                raise RuntimeError("cuDeviceGet_failed")
            sm_count = ctypes.c_int()
            if int(nvcuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, device)) != 0:
                raise RuntimeError("cuDeviceGetAttribute_failed")
            return max(1, int(sm_count.value))
        except Exception:
            return int(cls.FALLBACK_SM_COUNT)

    @classmethod
    def recommended_pool_size(
        cls,
        *,
        gpu_id: int = 0,
        reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    ) -> int:
        sm_count = int(cls.query_sm_count(gpu_id=gpu_id))
        return max(1, int(sm_count * cls.CORES_PER_SM * float(reserve_ratio)))

    @classmethod
    def create_for_current_gpu(
        cls,
        *,
        gpu_id: int = 0,
        reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    ) -> "MicroSpecialistPool":
        return cls(
            pool_size=cls.recommended_pool_size(gpu_id=gpu_id, reserve_ratio=reserve_ratio),
            reserve_ratio=reserve_ratio,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            used = len(self._in_use)
            free = len(self._free_slots)
            return {
                "pool_size": int(self.pool_size),
                "engine_count": len(self._engines),
                "slots_used": int(used),
                "slots_free": int(free),
                "peak_utilization": int(self._peak_utilization),
                "reserve_ratio": float(self.reserve_ratio),
                "tier1_empty_stack_fallbacks": int(self._tier1_empty_stack_fallbacks),
            }

    def acquire(self, count: int) -> list[int]:
        requested = max(0, int(count))
        if requested <= 0:
            return []
        with self._lock:
            granted = min(requested, len(self._free_slots))
            slots = [self._free_slots.pop(0) for _ in range(granted)]
            for slot_id in slots:
                self._in_use.add(int(slot_id))
            self._peak_utilization = max(self._peak_utilization, len(self._in_use))
            return slots

    def release(self, slots: Iterable[int]) -> None:
        with self._lock:
            for raw_slot in slots:
                slot_id = int(raw_slot)
                if slot_id not in self._in_use:
                    continue
                self._in_use.remove(slot_id)
                self._free_slots.append(slot_id)
            self._free_slots.sort()

    def run_batch(self, slots: Sequence[int], programs: Sequence[Any]) -> list[float]:
        if len(slots) != len(programs):
            raise ValueError("slot_program_length_mismatch")
        results: list[float] = []
        for raw_slot, program in zip(slots, programs):
            binding = self._binding(int(raw_slot))
            compiled = self._compile_program(program)
            results.append(float(self._execute_binding(binding, compiled)))
        return results

    # Sovereign cut (TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §8):
    # run_overflow_sequential was the Python fallback for when the GPU slot
    # pool overflowed. Deleted. Overflow is now dropped silently; the pool
    # itself is off the hot path (see decomposer deletion in knowledgeverse.py).

    def run_with_graceful_degradation(self, programs: Sequence[Any]) -> tuple[list[float], dict[str, Any]]:
        requested = len(programs)
        slots = self.acquire(requested)
        granted = len(slots)
        overflow = max(0, requested - granted)
        try:
            results = self.run_batch(slots, programs[:granted]) if granted else []
            stats = self.snapshot()
            stats.update(
                {
                    "requested_slots": int(requested),
                    "granted_slots": int(granted),
                    "overflow_sequential": int(overflow),
                }
            )
            return results, stats
        finally:
            self.release(slots)

    def cleanup(self) -> None:
        for engine in self._engines:
            try:
                engine.cleanup()
            except Exception:
                pass
        if self._sequential_engine is not None:
            try:
                self._sequential_engine.cleanup()
            except Exception:
                pass
        if self._tiered_fallback_engine is not None:
            try:
                self._tiered_fallback_engine.cleanup()
            except Exception:
                pass

    def _execute_binding(
        self,
        binding: MicroSlotBinding,
        compiled: dict[str, list[Any]],
        *,
        sequential: bool = False,
    ) -> float:
        engine = self._sequential_engine if sequential else self._engines[binding.engine_index]
        if engine is None:
            raise RuntimeError("micro_specialist_engine_unavailable")
        try:
            engine.reset_instance(binding.instance_id)
            return float(
                engine.execute_single(
                    binding.instance_id,
                    compiled["op_codes"],
                    compiled["scalars"],
                    compiled["vectors"],
                )
            )
        except RuntimeError as exc:
            if "empty stack" not in str(exc).lower():
                raise
            self._tier1_empty_stack_fallbacks += 1
            return float(self._execute_tiered_fallback(binding.instance_id, compiled))

    def _execute_tiered_fallback(
        self,
        instance_id: int,
        compiled: dict[str, list[Any]],
    ) -> float:
        if self._tiered_fallback_engine is None:
            from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine

            self._tiered_fallback_engine = TieredRPNEngine()
        safe_instance = int(instance_id) % int(getattr(self._tiered_fallback_engine, "MAX_INSTANCES", 18))
        return float(
            self._tiered_fallback_engine.execute_single(
                safe_instance,
                compiled["op_codes"],
                compiled["scalars"],
                compiled["vectors"],
            )
        )

    def _binding(self, slot_id: int) -> MicroSlotBinding:
        if not (0 <= int(slot_id) < len(self._bindings)):
            raise ValueError(f"invalid_micro_slot:{slot_id}")
        return self._bindings[int(slot_id)]

    def _compile_program(self, program: Any) -> dict[str, list[Any]]:
        if isinstance(program, dict):
            op_codes = [int(value) for value in list(program.get("op_codes", []))]
            scalars = [float(value) for value in list(program.get("scalars", []))]
            vectors = [
                [float(component) for component in list(vector)]
                for vector in list(program.get("vectors", []))
            ]
            while len(scalars) < len(op_codes):
                scalars.append(0.0)
            while len(vectors) < len(op_codes):
                vectors.append([0.0, 0.0, 0.0])
            return {
                "op_codes": list(op_codes),
                "scalars": list(scalars[: len(op_codes)]),
                "vectors": [list(vector) for vector in vectors[: len(op_codes)]],
            }
        tokens = list(program) if isinstance(program, (list, tuple)) else str(program).split()
        op_codes: list[int] = []
        scalars: list[float] = []
        vectors: list[list[float]] = []
        for token in tokens:
            text = str(token).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in _TOKEN_TO_OPCODE:
                op_codes.append(int(_TOKEN_TO_OPCODE[lowered]))
                scalars.append(0.0)
                vectors.append([0.0, 0.0, 0.0])
                continue
            if text.startswith("[") and text.endswith("]"):
                parts = [float(value.strip()) for value in text[1:-1].split(",") if value.strip()]
                if len(parts) != 3:
                    raise ValueError(f"micro_vector_literal_requires_three_components:{text}")
                op_codes.append(1)
                scalars.append(0.0)
                vectors.append(list(parts))
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"unsupported_micro_token:{text}") from exc
            op_codes.append(0)
            scalars.append(float(value))
            vectors.append([0.0, 0.0, 0.0])
        return {"op_codes": op_codes, "scalars": scalars, "vectors": vectors}


_GLOBAL_MICRO_POOL: MicroSpecialistPool | None = None


def configure_global_micro_specialist_pool(pool: MicroSpecialistPool | None) -> MicroSpecialistPool | None:
    global _GLOBAL_MICRO_POOL
    _GLOBAL_MICRO_POOL = pool
    return _GLOBAL_MICRO_POOL


def get_global_micro_specialist_pool() -> MicroSpecialistPool:
    global _GLOBAL_MICRO_POOL
    if _GLOBAL_MICRO_POOL is None:
        _GLOBAL_MICRO_POOL = MicroSpecialistPool.create_for_current_gpu()
    return _GLOBAL_MICRO_POOL


__all__ = [
    "MicroSlotBinding",
    "MicroSpecialistPool",
    "configure_global_micro_specialist_pool",
    "get_global_micro_specialist_pool",
]
