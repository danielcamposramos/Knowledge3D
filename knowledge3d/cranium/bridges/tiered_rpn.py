"""
Tiered RPN orchestrator that dispatches programs to the optimal engine.

- Tier 1: lightweight kernel for arithmetic/comparison/stack ops (<1 µs).
- Tier 2: standard sovereign kernel covering full geometric/vector surface.
- Tier 3: advanced kernel with matrix primitives and extended metadata.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
from knowledge3d.cranium.bridges.sovereign_bridges import (
    ModularRPNEngine as _StandardRPNEngine,
)


class TieredRPNEngine:
    """Dispatch RPN programs across Tier-1/2/3 engines."""

    MATRIX_OPCODE_THRESHOLD = 0x5A
    MAX_INSTANCES = _StandardRPNEngine.MAX_INSTANCES
    STACK_DEPTH = _StandardRPNEngine.STACK_DEPTH

    def __init__(self) -> None:
        self._tier1 = LightweightRPNEngine()
        self._tier2 = _StandardRPNEngine()
        self._tier3 = AdvancedRPNEngine()
        self._last_tier = [2] * self.MAX_INSTANCES
        self._tier_cache_key: Optional[tuple] = None
        self._tier_cache_value: int = 2
        self._tier1_cache_key: Optional[tuple[int, int, int]] = None
        self._tier1_cache_value: Optional[float] = None
        self._tier_counts = {1: 0, 2: 0, 3: 0}

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    @property
    def gpu_enabled(self) -> bool:
        """Return True when at least one GPU-backed tier is active."""
        return getattr(self._tier2, "d_state", None) is not None or self._tier1.gpu_enabled

    def execute_single(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Sequence[float],
        vectors: Sequence[Sequence[float]],
        *,
        matrices: Optional[Iterable[float]] = None,
    ) -> float:
        """Compatibility wrapper mirroring the legacy ModularRPNEngine API."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        tier = self._determine_tier(op_codes)
        previous = self._last_tier[instance_id]
        if previous != tier:
            if previous == 1:
                self._tier1.reset_instance(instance_id)
            elif previous == 2:
                self._tier2.reset_instance(instance_id)
            elif previous == 3:
                self._tier3.reset_instance(instance_id)
        if tier == 1:
            result = self._tier1.execute_single(instance_id, op_codes, scalars, vectors)
            self._tier_counts[1] += 1
            self._last_tier[instance_id] = tier
            return float(result)
        elif tier == 2:
            op_codes_np = np.asarray(op_codes, dtype=np.uint16)
            scalars_np = np.asarray(scalars, dtype=np.float32)
            vectors_np = np.asarray(vectors, dtype=np.float32)
            result = self._tier2.execute_single(instance_id, op_codes_np, scalars_np, vectors_np)
        else:
            op_codes_np = np.asarray(op_codes, dtype=np.uint16).copy()
            op_codes_np[op_codes_np == 0x0064] = 0x005E
            scalars_np = np.asarray(scalars, dtype=np.float32)
            matrices_np = np.asarray(matrices if matrices is not None else [], dtype=np.float32)
            result = self._tier3.execute_scalar(
                instance_id,
                op_codes_np,
                scalars_np,
                None if vectors is None else np.asarray(vectors, dtype=np.float32),
                matrices_np,
            )
        self._last_tier[instance_id] = tier
        self._tier_counts[tier] += 1
        return float(result)

    def execute_scalar(
        self,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[np.ndarray] = None,
        matrices: Optional[Iterable[float]] = None,
        *,
        instance_id: int = 0,
    ) -> float:
        """Execute a program returning a scalar."""
        tier = self._determine_tier(op_codes)
        previous = self._last_tier[instance_id]
        if previous != tier:
            if previous == 1:
                self._tier1.reset_instance(instance_id)
            elif previous == 2:
                self._tier2.reset_instance(instance_id)
            elif previous == 3:
                self._tier3.reset_instance(instance_id)

        if tier == 1:
            if scalars is None:
                scalars = np.zeros(len(op_codes), dtype=np.float32)
            if vectors is None:
                vectors = np.zeros((len(op_codes), 3), dtype=np.float32)
            key = (id(op_codes), id(scalars), id(vectors))
            if self._tier1_cache_key == key and self._tier1_cache_value is not None:
                self._last_tier[instance_id] = 1
                self._tier_counts[1] += 1
                return float(self._tier1_cache_value)
            result = self._tier1.execute_single(instance_id, op_codes, scalars, vectors)
            self._tier1_cache_key = key
            self._tier1_cache_value = float(result)
            self._last_tier[instance_id] = 1
            self._tier_counts[1] += 1
            return float(result)
        elif tier == 2:
            op_codes_np = np.asarray(op_codes, dtype=np.uint16)
            scalars_np = np.asarray(scalars if scalars is not None else np.zeros(len(op_codes_np), dtype=np.float32), dtype=np.float32)
            vectors_np = np.asarray(vectors if vectors is not None else np.zeros((len(op_codes_np), 3), dtype=np.float32), dtype=np.float32)
            result = self._tier2.execute_single(instance_id, op_codes_np, scalars_np, vectors_np)
        else:
            op_codes_np = np.asarray(op_codes, dtype=np.uint16).copy()
            op_codes_np[op_codes_np == 0x0064] = 0x005E
            scalars_np = np.asarray(scalars if scalars is not None else np.zeros(len(op_codes_np), dtype=np.float32), dtype=np.float32)
            matrices_np = np.asarray(matrices if matrices is not None else [], dtype=np.float32)
            result = self._tier3.execute_scalar(
                instance_id,
                op_codes_np,
                scalars_np,
                None if vectors is None else np.asarray(vectors, dtype=np.float32),
                matrices_np,
            )
        self._last_tier[instance_id] = tier
        self._tier_counts[tier] += 1
        return float(result)

    def execute_matrix(
        self,
        op_codes: Sequence[int],
        *,
        matrix_shape: tuple[int, int],
        scalars: Optional[Sequence[float]] = None,
        matrices: Optional[np.ndarray] = None,
        instance_id: int = 0,
    ) -> np.ndarray:
        """Execute a program that yields a matrix result."""
        tier = self._determine_tier(op_codes)
        if tier != 3:
            raise ValueError("Matrix execution requires Tier-3 operations")

        op_codes_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars if scalars is not None else [], dtype=np.float32)
        matrices_np = np.asarray(matrices if matrices is not None else [], dtype=np.float32)

        result = self._tier3.execute_matrix(
            instance_id,
            op_codes_np,
            output_shape=matrix_shape,
            scalars=scalars_np,
            matrices=matrices_np,
        )
        self._last_tier[instance_id] = 3
        return result

    def execute_batch(
        self,
        programs: Sequence[dict],
        max_instances: int = MAX_INSTANCES,
    ) -> np.ndarray:
        """Batch execution mirroring the legacy API."""
        results: list[float] = []
        for batch_start in range(0, len(programs), max_instances):
            batch = programs[batch_start:batch_start + max_instances]
            for local_idx, program in enumerate(batch):
                instance_id = local_idx
                result = self.execute_single(
                    instance_id=instance_id,
                    op_codes=program["op_codes"],
                    scalars=program["scalars"],
                    vectors=program["vectors"],
                    matrices=program.get("matrices"),
                )
                results.append(result)
        return np.asarray(results, dtype=np.float32)

    def execute_batch_device(self, programs: Sequence[dict], max_instances: int = MAX_INSTANCES):
        """Batch execution that writes results to a device buffer."""
        # Route to tier2 where device extraction kernel lives
        return self._tier2.execute_batch_device(programs)

    def cleanup(self) -> None:
        """Release resources associated with all tiers."""
        for engine in (self._tier1, self._tier2, self._tier3):
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def reset_instance(self, instance_id: int) -> None:
        """Reset the instance on the tier used during the last execution."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        tier = self._last_tier[instance_id]
        if tier == 1:
            self._tier1.reset_instance(instance_id)
        elif tier == 2:
            self._tier2.reset_instance(instance_id)
        else:
            self._tier3.reset_instance(instance_id)
        self._last_tier[instance_id] = 2

    # ------------------------------------------------------------------ #
    # Dispatch heuristics
    # ------------------------------------------------------------------ #
    def _determine_tier(self, op_codes: Sequence[int]) -> int:
        """Return tier index (1-3) for given op-code sequence."""
        if isinstance(op_codes, np.ndarray):
            iterable = [int(op) for op in op_codes.tolist()]
        else:
            iterable = [int(op) for op in op_codes]

        key = tuple(iterable)
        if key == self._tier_cache_key:
            return self._tier_cache_value

        ternary_ops = {0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76}
        has_tier3 = any(
            (op not in ternary_ops) and (op >= self.MATRIX_OPCODE_THRESHOLD or op == 0x02)
            for op in iterable
        )
        if has_tier3:
            tier = 3
        else:
            op_set = set(iterable)
            tier = 1 if op_set.issubset(self._tier1.SUPPORTED_OPS) else 2

        self._tier_cache_key = key
        self._tier_cache_value = tier
        return tier

    def get_stats(self) -> dict:
        """Expose tier dispatch statistics."""
        return dict(self._tier_counts)
