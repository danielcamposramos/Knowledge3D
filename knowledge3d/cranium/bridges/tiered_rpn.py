"""
Tiered RPN orchestrator that dispatches programs to the optimal engine.

- Tier 1: lightweight kernel for arithmetic/comparison/stack ops (<1 µs).
- Tier 2: standard sovereign kernel covering full geometric/vector surface.
- Tier 3: advanced kernel with matrix primitives and extended metadata.
"""
from __future__ import annotations

import ctypes
import numbers
from pathlib import Path
from typing import Iterable, Optional, Sequence

from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
from knowledge3d.cranium.bridges.sovereign_bridges import (
    ModularRPNEngine as _StandardRPNEngine,
)
from knowledge3d.cranium.sovereign import loader

_CODEC_TOKEN_MAP = {
    "TERNARY_QUANT": "quant",
    "TERNARY_DEQUANT": "dequant",
    "DCT8": "dct8",
    "DCT8X8": "dct8",
    "DCT8X8_FORWARD": "dct8",
    "IDCT8": "idct8",
    "IDCT8X8": "idct8",
    "IDCT8X8_INVERSE": "idct8",
    "MDCT": "mdct",
    "MDCT_FORWARD": "mdct",
    "IMDCT": "imdct",
    "IMDCT_INVERSE": "imdct",
    "BATCH_MDCT": "batch_mdct",
    "BATCH_DCT": "batch_dct",
    "RESHAPE_TO_BLOCKS": "reshape_blocks",
    "BLOCKS_TO_GRID": "blocks_to_grid",
    "TERNARY_ADD": "tadd",
    "TERNARY_MUL": "tmul",
}

_CAS_TIER2_RANGE = range(0x220, 0x238)
_SAS_TIER1_OPS = {0x239}
_SAS_TIER2_OPS = {0x23A, 0x23B, 0x23D}
_SAS_TIER3_OPS = {0x238, 0x23C}
_CAS_SAS_GPU_KERNEL_OPS = set(_CAS_TIER2_RANGE) | _SAS_TIER1_OPS | _SAS_TIER2_OPS | _SAS_TIER3_OPS


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
        self._tier1_fallback_count = 0  # Count of Tier-1 -> Tier-2 fallbacks due to empty stack
        self._codec_ops: Optional[TernaryCodecOps] = None
        self._ternary_kernels: Optional[dict] = None

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
            try:
                result = self._tier1.execute_single(instance_id, op_codes, scalars, vectors)
                self._tier_counts[1] += 1
                self._last_tier[instance_id] = tier
                return float(result)
            except RuntimeError as e:
                if "empty stack" in str(e).lower():
                    # Tier-1 failed with empty stack - retry on Tier-2
                    self._tier1.reset_instance(instance_id)
                    result = self._tier2.execute_single(
                        instance_id, list(op_codes), list(scalars), list(vectors)
                    )
                    self._last_tier[instance_id] = 2
                    self._tier_counts[2] += 1
                    self._tier1_fallback_count += 1
                    return float(result)
                raise
        elif tier == 2:
            result = self._tier2.execute_single(instance_id, list(op_codes), list(scalars), list(vectors))
        else:
            remapped = [0x005E if op == 0x0064 else int(op) for op in op_codes]
            matrices_seq = list(matrices) if matrices is not None else []
            result = self._tier3.execute_scalar(
                instance_id,
                remapped,
                list(scalars),
                vectors,
                matrices_seq,
            )
        self._last_tier[instance_id] = tier
        self._tier_counts[tier] += 1
        return float(result)

    def execute_scalar(
        self,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[Sequence[Sequence[float]]] = None,
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
                scalars = [0.0] * len(op_codes)
            if vectors is None:
                vectors = [[0.0, 0.0, 0.0] for _ in op_codes]
            key = (id(op_codes), id(scalars), id(vectors))
            if self._tier1_cache_key == key and self._tier1_cache_value is not None:
                self._last_tier[instance_id] = 1
                self._tier_counts[1] += 1
                return float(self._tier1_cache_value)
            try:
                result = self._tier1.execute_single(instance_id, op_codes, scalars, vectors)
                self._tier1_cache_key = key
                self._tier1_cache_value = float(result)
                self._last_tier[instance_id] = 1
                self._tier_counts[1] += 1
                return float(result)
            except RuntimeError as e:
                if "empty stack" in str(e).lower():
                    # Tier-1 failed with empty stack - retry on Tier-2
                    self._tier1.reset_instance(instance_id)
                    scalars_seq = list(scalars)
                    vectors_seq = list(vectors)
                    result = self._tier2.execute_single(instance_id, list(op_codes), scalars_seq, vectors_seq)
                    self._last_tier[instance_id] = 2
                    self._tier_counts[2] += 1
                    self._tier1_fallback_count += 1
                    return float(result)
                raise
        elif tier == 2:
            scalars_seq = list(scalars) if scalars is not None else [0.0] * len(op_codes)
            vectors_seq = list(vectors) if vectors is not None else [[0.0, 0.0, 0.0] for _ in op_codes]
            result = self._tier2.execute_single(instance_id, list(op_codes), scalars_seq, vectors_seq)
        else:
            remapped = [0x005E if op == 0x0064 else int(op) for op in op_codes]
            scalars_seq = list(scalars) if scalars is not None else [0.0] * len(op_codes)
            matrices_seq = list(matrices) if matrices is not None else []
            result = self._tier3.execute_scalar(
                instance_id,
                remapped,
                scalars_seq,
                vectors,
                matrices_seq,
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
        matrices: Optional[Sequence[float]] = None,
        instance_id: int = 0,
    ):
        """Execute a program that yields a matrix result."""
        tier = self._determine_tier(op_codes)
        if tier != 3:
            raise ValueError("Matrix execution requires Tier-3 operations")

        op_codes_seq = [int(o) for o in op_codes]
        scalars_seq = list(scalars) if scalars is not None else []
        matrices_seq = list(matrices) if matrices is not None else []
        result = self._tier3.execute_matrix(
            instance_id,
            op_codes_seq,
            output_shape=matrix_shape,
            scalars=scalars_seq,
            matrices=matrices_seq,
        )
        self._last_tier[instance_id] = 3
        return result

    def execute_batch(
        self,
        programs: Sequence[dict],
        max_instances: int = MAX_INSTANCES,
    ) -> list[float]:
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
        return results

    def execute_batch_device(self, programs: Sequence[dict], max_instances: int = MAX_INSTANCES):
        """Batch execution that writes results to a device buffer."""
        # Route to tier2 where device extraction kernel lives
        return self._tier2.execute_batch_device(programs)

    def bind_galaxy_buffer(
        self,
        flat_entries: Sequence[float],
        *,
        entry_count: int,
        entry_stride: int = 19,
        embedding_offset: int = 3,
        embedding_dim: int = 16,
    ) -> dict[str, int]:
        """Bind a flattened Galaxy table into the Tier-2 sovereign runtime."""
        return self._tier2.bind_galaxy_buffer(
            flat_entries,
            entry_count=entry_count,
            entry_stride=entry_stride,
            embedding_offset=embedding_offset,
            embedding_dim=embedding_dim,
        )

    def read_instance_stack_scalars(self, instance_id: int) -> list[float]:
        """Expose the Tier-2 stack surface for frontier-aware navigation."""
        return self._tier2.read_instance_stack_scalars(instance_id)

    def store_embedding(
        self,
        *,
        instance_id: int,
        embedding: Sequence[float],
        slot: int = 0,
    ) -> None:
        """Upload a query embedding for a specific instance."""
        self._tier2.store_embedding(instance_id=instance_id, embedding=embedding, slot=slot)

    # ------------------------------------------------------------------ #
    # Codec-aware execution (GPU kernels orchestrated via TernaryCodecOps)
    # ------------------------------------------------------------------ #
    def execute_codec(
        self,
        tokens: Sequence[str],
        *,
        data=None,
        return_vector: bool = False,
    ):
        """Execute codec-focused RPN programs using GPU codec kernels."""
        self._ensure_codec_ops()
        stack: list = []
        if data is not None:
            stack.append(self._to_native(data))

        for token in tokens:
            if token in _CODEC_TOKEN_MAP:
                op = _CODEC_TOKEN_MAP[token]
                if op == "dct8":
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    transformed = self._codec_ops.dct8_forward(values)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(transformed, shape)))
                elif op == "idct8":
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    transformed = self._codec_ops.dct8_inverse(values)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(transformed, shape)))
                elif op == "quant":
                    threshold = self._pop_number(stack, default=self._codec_ops.threshold)
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    q = self._codec_ops.quantize(values, threshold=threshold)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(q, shape)))
                elif op == "dequant":
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    dq = self._codec_ops.dequantize(values)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(dq, shape)))
                elif op == "tadd":
                    b_source = self._pop_any(stack)
                    a_source = self._pop_any(stack)
                    b_vals, b_shape = self._flatten_with_shape(b_source)
                    a_vals, a_shape = self._flatten_with_shape(a_source)
                    if len(a_vals) != len(b_vals):
                        raise ValueError("TERNARY_ADD requires operands of equal length")
                    result = self._ternary_add_gpu(a_vals, b_vals)
                    target_shape = a_shape or b_shape
                    stack.append(
                        self._preserve_layout(a_source, self._reshape_from_flat(result, target_shape))
                    )
                elif op == "tmul":
                    b_source = self._pop_any(stack)
                    a_source = self._pop_any(stack)
                    b_vals, b_shape = self._flatten_with_shape(b_source)
                    a_vals, a_shape = self._flatten_with_shape(a_source)
                    if len(a_vals) != len(b_vals):
                        raise ValueError("TERNARY_MUL requires operands of equal length")
                    result = self._ternary_mul_gpu(a_vals, b_vals)
                    target_shape = a_shape or b_shape
                    stack.append(
                        self._preserve_layout(a_source, self._reshape_from_flat(result, target_shape))
                    )
                elif op == "mdct":
                    frame_size_val = stack.pop() if stack and self._is_number(stack[-1]) else None
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    frame_size_int = int(frame_size_val) if frame_size_val is not None else len(values)
                    coeffs = self._codec_ops.batch_mdct(values, frame_size=frame_size_int)
                    frame_count = len(values) // frame_size_int if frame_size_int else 0
                    out_shape = (frame_count, frame_size_int // 2) if frame_count > 1 else (frame_size_int // 2,)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(coeffs, out_shape)))
                elif op == "imdct":
                    frame_size_val = stack.pop() if stack and self._is_number(stack[-1]) else None
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    inferred_size = len(values) * 2
                    frame_size_int = int(frame_size_val) if frame_size_val is not None else inferred_size
                    reconstructed = self._codec_ops.batch_imdct(values, frame_size=frame_size_int)
                    half = frame_size_int // 2
                    frame_count = len(values) // half if half else 0
                    out_shape = (frame_count, frame_size_int) if frame_count > 1 else (frame_size_int,)
                    stack.append(
                        self._preserve_layout(source, self._reshape_from_flat(reconstructed, out_shape))
                    )
                elif op == "batch_mdct":
                    frame_size = self._pop_number(stack)
                    frame_size_int = int(frame_size)
                    # Optional frame-count scalar on stack (used for validation only)
                    if stack and self._is_number(stack[-1]):
                        stack.pop()
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    total = len(values)
                    if frame_size_int <= 0 or total % frame_size_int != 0:
                        raise ValueError("Input length must be multiple of frame_size for BATCH_MDCT")
                    frame_count = total // frame_size_int
                    coeffs = self._codec_ops.batch_mdct(values, frame_size=frame_size_int)
                    out_shape = (frame_count, frame_size_int // 2) if frame_count > 1 else (frame_size_int // 2,)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(coeffs, out_shape)))
                elif op == "batch_dct":
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    transformed = self._codec_ops.dct8_forward(values)
                    stack.append(self._preserve_layout(source, self._reshape_from_flat(transformed, shape)))
                elif op == "reshape_blocks":
                    source = self._pop_any(stack)
                    values, shape = self._flatten_with_shape(source)
                    rows, cols = self._infer_grid_shape(shape)
                    is_integer = self._all_integral(values)
                    transformed = self._codec_ops.reshape_to_blocks(
                        values,
                        rows=rows,
                        cols=cols,
                        block_h=8,
                        block_w=8,
                        integer=is_integer,
                    )
                    blocks_per_grid = (rows // 8) * (cols // 8)
                    block_data = self._reshape_from_flat(transformed, (blocks_per_grid, 64))
                    stack.append(
                        self._make_block_packet(
                            data=block_data,
                            rows=rows,
                            cols=cols,
                            block_h=8,
                            block_w=8,
                            integer=is_integer,
                        )
                    )
                elif op == "blocks_to_grid":
                    source = self._pop_any(stack)
                    if not self._is_block_packet(source):
                        raise ValueError("BLOCKS_TO_GRID requires block layout packet from RESHAPE_TO_BLOCKS")
                    rows = int(source["rows"])
                    cols = int(source["cols"])
                    block_h = int(source["block_h"])
                    block_w = int(source["block_w"])
                    is_integer = bool(source["integer"])
                    values, shape = self._flatten_with_shape(source)
                    transformed = self._codec_ops.blocks_to_grid(
                        values,
                        rows=rows,
                        cols=cols,
                        block_h=block_h,
                        block_w=block_w,
                        integer=is_integer,
                    )
                    stack.append(self._reshape_from_flat(transformed, (rows, cols)))
                else:
                    raise ValueError(f"Unsupported codec op {op}")
            elif token.startswith("[") and token.endswith("]"):
                vec_str = token[1:-1]
                components = [float(x.strip()) for x in vec_str.split(",") if x.strip()]
                stack.append(components)
            else:
                try:
                    stack.append(float(token))
                except ValueError as exc:
                    raise ValueError(f"Unknown token in codec execution: {token}") from exc

        if not stack:
            raise ValueError("Codec execution produced no result")
        result = stack[-1]
        if not return_vector and self._is_number(result):
            return float(result)
        return result

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

    def bind_cas_pool(self):
        """Expose the sovereign CAS pool through the canonical tiered surface."""
        return self._tier2.bind_cas_pool()

    def bind_sas_symbol_table(self, values: list[float], star_ids: list[int]) -> None:
        """Expose SAS symbol upload through the canonical tiered surface."""
        self._tier2.bind_sas_symbol_table(values, star_ids)

    def read_cas_node(self, node_idx: int):
        return self._tier2.read_cas_node(node_idx)

    def read_cas_pool_top(self) -> int:
        return self._tier2.read_cas_pool_top()

    # ------------------------------------------------------------------ #
    # Dispatch heuristics
    # ------------------------------------------------------------------ #
    def _reset_tier(self, tier: int, instance_id: int) -> None:
        """Reset a specific tier instance."""
        if tier == 1:
            self._tier1.reset_instance(instance_id)
        elif tier == 2:
            self._tier2.reset_instance(instance_id)
        elif tier == 3:
            self._tier3.reset_instance(instance_id)

    def _determine_tier(self, op_codes: Sequence[int]) -> int:
        """Return tier index (1-3) for given op-code sequence."""
        iterable = [int(op) for op in op_codes]

        key = tuple(iterable)
        if key == self._tier_cache_key:
            return self._tier_cache_value

        tier = self._select_tier(set(iterable), iterable)

        self._tier_cache_key = key
        self._tier_cache_value = tier
        return tier

    def select_tier(self, op_codes: Sequence[int]) -> int:
        """Public tier query used by higher-level math-core allocation code."""
        return self._determine_tier(op_codes)

    def _select_tier(self, op_set: set[int], iterable: Sequence[int]) -> int:
        """Return 1/2/3 using explicit CAS/SAS routing before generic thresholds."""
        if op_set & _SAS_TIER3_OPS:
            return 3
        if (op_set & _SAS_TIER2_OPS) or any(op in _CAS_TIER2_RANGE for op in iterable):
            return 2
        if op_set & _SAS_TIER1_OPS:
            return 1
        return self._select_tier_base(op_set, iterable)

    def _select_tier_base(self, op_set: set[int], iterable: Sequence[int]) -> int:
        """Legacy tier heuristic for the non-CAS/SAS opcode surface."""

        ternary_ops = {0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76}
        tier2_forced_ops = {0xE0, 0xE1, 0xE2}
        has_tier3 = any(
            (op not in ternary_ops)
            and (op not in tier2_forced_ops)
            and (op >= self.MATRIX_OPCODE_THRESHOLD or op == 0x02)
            for op in iterable
        )
        if has_tier3:
            tier = 3
        elif op_set & tier2_forced_ops:
            tier = 2
        else:
            # Tier-1 for simple programs, Tier-2 for stack-heavy programs.
            # Tier-1 has a known bug with empty stack - fallback to Tier-2 is automatic.
            stack_ops = {50, 51, 52, 53, 54, 55}  # DUP, SWAP, DROP, OVER, ROT, NIP
            tier1_supported = getattr(self._tier1, "SUPPORTED_OPS", set())
            needs_tier2 = any(
                (op not in tier1_supported) and (op not in ternary_ops) and (op < self.MATRIX_OPCODE_THRESHOLD)
                for op in iterable
            )
            if needs_tier2 or (op_set & stack_ops):
                tier = 2  # Route stack-heavy to Tier-2 for stability
            else:
                tier = 1  # Simple arithmetic tries Tier-1 (with fallback)
        return tier

    def _execute_on_tier(
        self,
        tier: int,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Sequence[float],
        vectors: Sequence[Sequence[float]],
        matrices: Sequence[float] | None,
    ) -> float:
        if tier == 3 and any(int(op) in _CAS_SAS_GPU_KERNEL_OPS for op in op_codes):
            # CAS/SAS currently executes on the sovereign modular kernel surface.
            # Tier selection still reserves the caller on the Tier-3 math-core lane.
            return float(self._tier2.execute_single(instance_id, list(op_codes), list(scalars), list(vectors)))
        if tier == 1:
            return float(self._tier1.execute_single(instance_id, op_codes, scalars, vectors))
        if tier == 2:
            return float(self._tier2.execute_single(instance_id, list(op_codes), list(scalars), list(vectors)))
        remapped = [0x005E if op == 0x0064 else int(op) for op in op_codes]
        matrices_seq = list(matrices) if matrices is not None else []
        return float(self._tier3.execute_scalar(instance_id, remapped, list(scalars), vectors, matrices_seq))

    def execute_single(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Sequence[float],
        vectors: Sequence[Sequence[float]],
        *,
        matrices: Sequence[float] | None = None,
    ) -> float:
        """Execute with automatic Tier-1 retry fallback to Tier-2 on empty-stack."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        tier = self._determine_tier(op_codes)
        previous = self._last_tier[instance_id]
        if previous != tier:
            self._reset_tier(previous, instance_id)

        try:
            result = self._execute_on_tier(tier, instance_id, op_codes, scalars, vectors, matrices)
            self._last_tier[instance_id] = tier
            self._tier_counts[tier] += 1
            return result
        except RuntimeError as exc:
            msg = str(exc)
            if "empty stack" in msg and tier == 1:
                # Retry on Tier-2 for robustness while Tier-1 is unstable.
                self._tier1.reset_instance(instance_id)
                result = self._execute_on_tier(2, instance_id, op_codes, scalars, vectors, matrices)
                self._last_tier[instance_id] = 2
                self._tier_counts[2] += 1
                self._tier1_fallback_count = getattr(self, "_tier1_fallback_count", 0) + 1
                return result
            raise

    # ------------------------------------------------------------------ #
    # Helpers for codec-aware execution
    # ------------------------------------------------------------------ #
    def _ensure_codec_ops(self) -> None:
        if self._codec_ops is None:
            self._codec_ops = TernaryCodecOps()

    def _ensure_ternary_kernels(self) -> None:
        if self._ternary_kernels is None:
            ptx_path = Path(__file__).parent.parent / "ptx" / "ternary_ops.ptx"
            if not ptx_path.exists():
                raise FileNotFoundError(f"Ternary ops PTX not found at {ptx_path}")
            module = loader.load_module_from_file(str(ptx_path))
            self._ternary_kernels = {
                "module": module,
                "add": loader.get_function(module, "ternary_add_kernel"),
                "mul": loader.get_function(module, "ternary_mul_kernel"),
            }

    def _ternary_add_gpu(self, a_vals: list, b_vals: list) -> list[int]:
        self._ensure_ternary_kernels()
        n = len(a_vals)
        if n == 0:
            return []
        IntArray = ctypes.c_int8 * n
        buf_a = IntArray(*[int(v) for v in a_vals])
        buf_b = IntArray(*[int(v) for v in b_vals])
        d_a = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        d_b = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        try:
            loader.memcpy_htod(d_a, ctypes.cast(buf_a, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int8))
            loader.memcpy_htod(d_b, ctypes.cast(buf_b, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int8))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self._ternary_kernels["add"],
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_a.value),
                    ctypes.c_uint64(d_b.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                ],
            )
            loader.synchronize()
            host_out = IntArray()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_int8))
            return [int(v) for v in host_out]
        finally:
            loader.gpu_free(d_a)
            loader.gpu_free(d_b)
            loader.gpu_free(d_out)

    def _ternary_mul_gpu(self, a_vals: list, b_vals: list) -> list[int]:
        self._ensure_ternary_kernels()
        n = len(a_vals)
        if n == 0:
            return []
        IntArray = ctypes.c_int8 * n
        buf_a = IntArray(*[int(v) for v in a_vals])
        buf_b = IntArray(*[int(v) for v in b_vals])
        d_a = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        d_b = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int8))
        try:
            loader.memcpy_htod(d_a, ctypes.cast(buf_a, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int8))
            loader.memcpy_htod(d_b, ctypes.cast(buf_b, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int8))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self._ternary_kernels["mul"],
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_a.value),
                    ctypes.c_uint64(d_b.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                ],
            )
            loader.synchronize()
            host_out = IntArray()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_int8))
            return [int(v) for v in host_out]
        finally:
            loader.gpu_free(d_a)
            loader.gpu_free(d_b)
            loader.gpu_free(d_out)

    def _to_native(self, value):
        if hasattr(value, "to_python"):
            try:
                return value.to_python()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return value

    def _is_number(self, value) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool)

    def _pop_number(self, stack: list, default=None) -> float:
        if stack and self._is_number(stack[-1]):
            return float(stack.pop())
        if default is not None:
            return float(default)
        raise ValueError("Expected numeric literal on stack for codec op")

    def _pop_any(self, stack: list):
        if not stack:
            raise ValueError("Stack underflow during codec execution")
        return stack.pop()

    def _flatten_with_shape(self, value) -> tuple[list, tuple[int, ...]]:
        if self._is_block_packet(value):
            native = value.get("data", [])
        else:
            native = self._to_native(value)
        shape = self._infer_shape(native)
        flat: list = []
        for item in self._flatten(native):
            flat.append(item)
        return flat, shape

    def _flatten(self, value):
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self._flatten(item)
        else:
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                yield int(value)
            else:
                yield float(value)

    def _infer_shape(self, value) -> tuple[int, ...]:
        if isinstance(value, (list, tuple)) and value:
            return (len(value),) + self._infer_shape(value[0])
        return ()

    def _reshape_from_flat(self, flat: list, shape: tuple[int, ...]):
        if not shape:
            return flat[0] if flat else 0.0
        rebuilt, _ = self._reshape_recursive(flat, shape, 0)
        return rebuilt

    def _reshape_recursive(self, flat: list, shape: tuple[int, ...], offset: int):
        if not shape:
            return flat[offset], offset + 1
        size = shape[0]
        out = []
        cursor = offset
        for _ in range(size):
            val, cursor = self._reshape_recursive(flat, shape[1:], cursor)
            out.append(val)
        return out, cursor

    def _shape_with_new_last(self, base_shape: tuple[int, ...], new_last: int) -> tuple[int, ...]:
        if not base_shape:
            return (new_last,)
        if len(base_shape) == 1:
            return (new_last,)
        return tuple(base_shape[:-1]) + (new_last,)

    def _infer_grid_shape(self, shape: tuple[int, ...]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError("RESHAPE_TO_BLOCKS currently requires a 2D grid input")
        rows, cols = int(shape[0]), int(shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("grid shape must be positive")
        if rows % 8 != 0 or cols % 8 != 0:
            raise ValueError("RESHAPE_TO_BLOCKS requires 2D grid dimensions divisible by 8")
        return rows, cols

    def _all_integral(self, values: Sequence[object]) -> bool:
        return all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values)

    def _is_block_packet(self, value) -> bool:
        return isinstance(value, dict) and value.get("__k3d_layout__") == "blocks8x8_v1"

    def _make_block_packet(
        self,
        *,
        data,
        rows: int,
        cols: int,
        block_h: int,
        block_w: int,
        integer: bool,
    ) -> dict:
        return {
            "__k3d_layout__": "blocks8x8_v1",
            "rows": int(rows),
            "cols": int(cols),
            "block_h": int(block_h),
            "block_w": int(block_w),
            "integer": bool(integer),
            "data": data,
        }

    def _preserve_layout(self, source, rebuilt):
        if self._is_block_packet(source):
            packet = dict(source)
            packet["data"] = rebuilt
            return packet
        return rebuilt

    def get_stats(self) -> dict:
        """Expose tier dispatch statistics."""
        stats = dict(self._tier_counts)
        stats["tier1_fallbacks"] = self._tier1_fallback_count
        return stats
