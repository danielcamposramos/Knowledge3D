"""GPU-resident modular RPN engine using sovereign PTX architecture.

This module provides a high-level RPN calculator interface that leverages our
sovereign ModularRPNEngine bridge (pure ctypes + hand-authored PTX).

Python is used ONLY for:
- Entry point (tokenize, compile expressions)
- I/O (reading results, formatting output)
- High-level orchestration

All computation happens on GPU via modular_rpn_kernel.ptx.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from knowledge3d.cranium.ptx_runtime.math_core_pool import (
    MathCorePool,
    get_global_math_core_pool,
)
from .rpn_opcodes import (
    OP_ENTROPY_SUM,
    OP_SIGMOID_APPROX,
    OP_SMAV,
    OP_SPARSE_LOAD,
    OP_POINTER_LITERAL,
    OP_TRM_MATVEC_1024x512,
    OP_TRM_MATVEC_512x1024,
    OP_TRM_SWIGLU_1024,
    OP_TRM_SWIGLU_512,
    OP_TRM_VEC_ADD3_512,
)
from .codec_opcodes import (
    OP_TERNARY_QUANT,
    OP_TERNARY_DEQUANT,
    OP_TERNARY_ADD,
    OP_TERNARY_MUL,
    OP_DCT8X8,
    OP_IDCT8X8,
    OP_MDCT_FRAME,
    OP_IMDCT_FRAME,
    OP_BATCH_DCT,
    OP_BATCH_MDCT,
    OP_RESHAPE_TO_BLOCKS,
    OP_BLOCKS_TO_GRID,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as SovereignRPNEngine


class ModularRPNEngine:
    """High-level RPN calculator using sovereign PTX architecture.

    This is a thin Python wrapper around the sovereign ModularRPNEngine bridge.
    All GPU computation happens via hand-authored PTX kernels.

    Example:
        engine = ModularRPNEngine()
        result = engine.evaluate("2 3 + 5 *")  # (2 + 3) * 5 = 25.0
        print(result)  # 25.0
    """

    _INSTANCE_COUNT = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
    _STACK_MAX = 69       # Tesla 6-9: 6+9=15→6, 6×9=54→9, contains literal 6&9

    OP_LITERAL = 0
    OP_LITERAL_VEC = 1

    OPCODES: Dict[str, int] = {
        "+": 10,
        "-": 11,
        "*": 12,
        "/": 13,
        "^": 14,
        "pow": 14,
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
        "over": 53,
        "rot": 54,
        "clear": 55,
        "dot": 60,
        "cross": 61,
        "mag": 62,
        "norm": 63,
        "rotate": 70,
        "scale": 71,
        "translate": 72,
        "ifelse": 80,
        # Extended scalar ops used by physics:
        "abs": 0x27,
        # Procedural drawing opcodes (host parser may also consume)
        "MOVE": 0x64,
        "LINE": 0x65,
        "QUAD": 0x66,
        "CUBIC": 0x67,
        "ARC": 0x68,
        "CLOSE": 0x69,
        "STROKE": 0x6A,
        "FILL": 0x6B,
        "PUSH_STATE": 0x70,
        "POP_STATE": 0x71,
        "TRANSLATE": 0x72,
        "ROTATE": 0x73,
        "SCALE": 0x74,
        "SET_STROKE_COLOR": 0x75,
        "SET_FILL_COLOR": 0x76,
        "SET_LINE_WIDTH": 0x77,
        "SET_TERNARY_HINT": 0x78,
        # Lower-case aliases for convenience
        "move": 0x64,
        "line": 0x65,
        "quad": 0x66,
        "cubic": 0x67,
        "arc": 0x68,
        "close": 0x69,
        "stroke": 0x6A,
        "fill": 0x6B,
        "push_state": 0x70,
        "pop_state": 0x71,
        "translate_draw": 0x72,
        "rotate_draw": 0x73,
        "scale_draw": 0x74,
        "set_stroke_color": 0x75,
        "set_fill_color": 0x76,
        "set_line_width": 0x77,
        "set_ternary_hint": 0x78,
        "tadd": 112,
        "tmul": 113,
        "tnot": 114,
        "tcomp": 115,
        "tquant": 116,
        "tpack": 117,
        "tunpack": 118,
        "tfuse": 83,
        # Ternary codec ops
        "TERNARY_QUANT": OP_TERNARY_QUANT,
        "TERNARY_DEQUANT": OP_TERNARY_DEQUANT,
        "TERNARY_ADD": OP_TERNARY_ADD,
        "TERNARY_MUL": OP_TERNARY_MUL,
        "DCT8": OP_DCT8X8,
        "IDCT8": OP_IDCT8X8,
        "MDCT": OP_MDCT_FRAME,
        "IMDCT": OP_IMDCT_FRAME,
        "MDCT_FORWARD": OP_MDCT_FRAME,
        "IMDCT_INVERSE": OP_IMDCT_FRAME,
        "BATCH_DCT": OP_BATCH_DCT,
        "BATCH_MDCT": OP_BATCH_MDCT,
        "RESHAPE_TO_BLOCKS": OP_RESHAPE_TO_BLOCKS,
        "BLOCKS_TO_GRID": OP_BLOCKS_TO_GRID,
    }

    CONSTANTS: Dict[str, float] = {
        "pi": math.pi,
        "π": math.pi,
        "tau": math.tau,
        "phi": (1.0 + math.sqrt(5.0)) / 2.0,
        "φ": (1.0 + math.sqrt(5.0)) / 2.0,
        "e": math.e,
    }
    CODEC_TOKENS = {
        "TERNARY_QUANT",
        "TERNARY_DEQUANT",
        "DCT8",
        "DCT8X8",
        "DCT8X8_FORWARD",
        "IDCT8",
        "IDCT8X8",
        "IDCT8X8_INVERSE",
        "MDCT",
        "MDCT_FORWARD",
        "IMDCT",
        "IMDCT_INVERSE",
        "BATCH_MDCT",
        "BATCH_DCT",
        "TERNARY_ADD",
        "TERNARY_MUL",
    }

    def __init__(
        self,
        max_instances: int = _INSTANCE_COUNT,
        *,
        pool: Optional[MathCorePool] = None,
        instance_id: Optional[int] = None,
    ) -> None:
        """Initialize RPN engine with sovereign PTX backend.

        Args:
            max_instances: Maximum parallel instances (default 18, Tesla 3-6-9 resonance)
            pool: Optional shared MathCorePool for dynamic allocation
            instance_id: Optional pre-allocated math core to bind to
        """
        if max_instances > self._INSTANCE_COUNT:
            raise ValueError(f"Maximum supported instances is {self._INSTANCE_COUNT}")

        self.max_instances = max_instances
        self.pool = pool or get_global_math_core_pool()
        self.instance_id: Optional[int] = instance_id
        self._owned = instance_id is None
        # Lazy import keeps module importable in environments without CUDA bindings
        from knowledge3d.cranium.bridges.tiered_rpn import (
            TieredRPNEngine as SovereignRPNEngine,
        )

        self._sovereign_engine = SovereignRPNEngine()

    def tokenize_rpn(self, expression: str) -> List[str]:
        """Tokenize RPN expression into operators and operands.

        Args:
            expression: RPN expression string (e.g., "2 3 + 5 *")

        Returns:
            List of tokens
        """
        # Replace special unicode symbols
        expression = expression.replace("×", "*").replace("÷", "/")

        # Split on whitespace
        tokens = expression.split()

        # Expand constants
        expanded = []
        for token in tokens:
            if token in self.CONSTANTS:
                expanded.append(str(self.CONSTANTS[token]))
            else:
                expanded.append(token)

        return expanded

    def compile_tokens(
        self,
        tokens: List[str],
        instance_id: int = 0
    ) -> Tuple[List[int], List[float], List[Tuple[float, float, float]]]:
        """Compile tokens into op_codes, scalars, and vectors for GPU execution.

        Args:
            tokens: List of RPN tokens
            instance_id: Instance slot (0-14)

        Returns:
            (op_codes, scalars, vectors) ready for GPU
        """
        op_codes: list[int] = []
        scalar_literals: list[float] = []
        vector_literals: list[tuple[float, float, float]] = []

        for token in tokens:
            # Check if it's a vector literal [x,y,z]
            if token.startswith("[") and token.endswith("]"):
                # Vector literal
                vec_str = token[1:-1]  # Strip brackets
                components = [float(x.strip()) for x in vec_str.split(",")]

                if len(components) != 3:
                    raise ValueError(f"Vector must have exactly 3 components, got {len(components)}")

                op_codes.append(self.OP_LITERAL_VEC)
                vector_literals.append((components[0], components[1], components[2]))

            # Check if it's an operator
            elif token in self.OPCODES:
                op_codes.append(self.OPCODES[token])

            # Otherwise, treat as scalar literal
            else:
                try:
                    value = float(token)
                    op_codes.append(self.OP_LITERAL)
                    scalar_literals.append(value)
                except ValueError:
                    raise ValueError(f"Unknown token: {token}")

        return op_codes, scalar_literals, vector_literals

    def evaluate(
        self,
        expression: str,
        instance_id: Optional[int] = None,
        return_vector: bool = False,
        data=None,
    ) -> float:
        """Evaluate RPN expression on GPU.

        Args:
            expression: RPN expression string (e.g., "2 3 +")
            instance_id: Optional math core instance. When None, allocate via pool.
            return_vector: If True, return full float4 (NOT IMPLEMENTED - returns scalar only)
            data: Optional vector/tensor payload for codec ops (DCT/MDCT paths)

        Returns:
            Result from top of stack (scalar)

        Example:
            >>> engine = ModularRPNEngine()
            >>> engine.evaluate("2 3 + 5 *")
            25.0
            >>> engine.evaluate("[1,0,0] [0,1,0] dot")  # Dot product
            0.0
            >>> engine.evaluate("[1,0,0] [0,1,0] cross mag")  # Cross product magnitude
            1.0
        """
        tokens = self.tokenize_rpn(expression)
        if any(token in self.CODEC_TOKENS for token in tokens):
            # Codec ops are orchestrated directly through TieredRPNEngine to GPU kernels
            return self._sovereign_engine.execute_codec(tokens, data=data, return_vector=return_vector)

        core_id = self._ensure_core(tier=1, override_instance=instance_id)
        # Reset instance to clear stack before evaluation
        self._sovereign_engine.reset_instance(core_id)

        # Tokenize expression
        op_codes, scalars, vectors = self.compile_tokens(tokens, instance_id)

        # Execute on GPU via sovereign bridge
        result = self._sovereign_engine.execute_single(
            instance_id=core_id,
            op_codes=op_codes,
            scalars=scalars,
            vectors=vectors
        )

        return float(result)

    def evaluate_batch(
        self,
        expressions: List[str],
        max_parallel: int = 18  # Tesla 3-6-9 resonance
    ) -> List[float]:
        """Evaluate multiple RPN expressions in parallel.

        Args:
            expressions: List of RPN expression strings
            max_parallel: Maximum parallel instances (default 18, Tesla 3-6-9 resonance)

        Returns:
            NumPy array of results

        Example:
            >>> engine = ModularRPNEngine()
            >>> results = engine.evaluate_batch([
            ...     "2 3 +",
            ...     "5 4 *",
            ...     "10 2 /"
            ... ])
            >>> print(results)  # [5.0, 20.0, 5.0]
        """
        programs = []

        for expr in expressions:
            tokens = self.tokenize_rpn(expr)
            if any(token in self.CODEC_TOKENS for token in tokens):
                raise ValueError("Codec opcodes are not supported in evaluate_batch; use evaluate with data.")
            op_codes, scalars, vectors = self.compile_tokens(tokens)
            programs.append({
                'op_codes': op_codes,
                'scalars': scalars,
                'vectors': vectors
            })

        # Execute batch via sovereign bridge
        results = self._sovereign_engine.execute_batch(programs, max_instances=max_parallel)

        return list(results)

    def evaluate_batch_device(
        self,
        expressions: List[str],
    ) -> tuple[loader.CUdeviceptr, int]:
        """Evaluate multiple RPN expressions and return device buffer pointer + count."""
        programs = []
        for expr in expressions:
            tokens = self.tokenize_rpn(expr)
            op_codes, scalars, vectors = self.compile_tokens(tokens)
            programs.append({
                "op_codes": op_codes,
                "scalars": scalars,
                "vectors": vectors,
            })
        return self._sovereign_engine.execute_batch_device(programs)

    def reset(self, instance_id: int = 0) -> None:
        """Reset instance state (clear stack).

        Args:
            instance_id: Instance slot to reset (0-14)
        """
        self._sovereign_engine.reset_instance(instance_id)

    def close(self) -> None:
        """Clean up GPU resources."""
        if self._owned and self.instance_id is not None:
            try:
                self.pool.release_core(self.instance_id)
            except Exception:
                pass
        self._sovereign_engine.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except:
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_core(self, tier: int = 1, *, override_instance: Optional[int] = None) -> int:
        """Ensure a math core is available and return its instance id."""
        if override_instance is not None:
            self.instance_id = override_instance
            self._owned = False
            return override_instance

        if self.instance_id is None:
            self.instance_id = self.pool.spawn_core(tier=tier)
        else:
            self.pool.touch(self.instance_id)
        return self.instance_id


# Convenience API for quick calculations
def rpn_eval(expression: str) -> float:
    """Quick RPN evaluation (creates temporary engine).

    Args:
        expression: RPN expression string

    Returns:
        Result

    Example:
        >>> from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import rpn_eval
        >>> rpn_eval("2 3 + 5 *")
        25.0
    """
    engine = ModularRPNEngine()
    result = engine.evaluate(expression)
    engine.close()
    return result


# Opcode constants for thinking tag system (GLM's extended opcodes)
# These match the CUDA kernel definitions in modular_rpn_kernel.cu

class RPNProgram:
    """Low-level RPN bytecode builder for thinking tag inference.

    This class provides a builder pattern for constructing RPN programs
    with precise control over opcodes, particularly for sparse operations
    and temporal reasoning. Used by ThinkingTagBridge.

    Example:
        p = RPNProgram()
        p.u32(OP_SPARSE_LOAD)
        p.ptr(weight_buffer)
        p.u32(OP_SMAV)
        p.u8(0x0A)  # MAX opcode
    """

    def __init__(self):
        self.bytecode = bytearray()
        self._ptrs = []  # Track pointer positions for relocation

    def u8(self, val: int):
        """Append uint8 opcode"""
        self.bytecode.append(val & 0xFF)

    def u32(self, val: int):
        """Append uint32 value (little-endian)"""
        self.bytecode.extend(val.to_bytes(4, byteorder='little'))

    def f32(self, val: float):
        """Append float32 value"""
        import struct
        self.bytecode.extend(struct.pack('f', val))

    def ptr(self, device_ptr, *, rows: Optional[int] = None, cols: Optional[int] = None):
        """Append device pointer literal.

        When ``rows`` and ``cols`` are provided the opcode stream emits the
        tensor-aware literal (0x03, rows, cols, ptr_lo, ptr_hi) used by the
        Tier‑3 TRM kernels.  Otherwise it falls back to the legacy behaviour of
        writing an opaque pointer placeholder that is resolved during
        ``to_bytes``.
        """
        if rows is not None or cols is not None:
            if rows is None or cols is None:
                raise ValueError("rows and cols must be provided together")
            self.u8(OP_POINTER_LITERAL)
            self.f32(float(rows))
            self.f32(float(cols))
            pos = len(self.bytecode)
            self.bytecode.extend(b'\x00' * 8)
            self._ptrs.append((pos, device_ptr, "tensor"))
        else:
            pos = len(self.bytecode)
            self._ptrs.append((pos, device_ptr, "raw"))
            self.bytecode.extend(b'\x00' * 8)

    def to_bytes(self) -> bytes:
        """Convert to final bytecode"""
        self._resolve_ptrs()
        return bytes(self.bytecode)

    def to_uint32_array(self) -> list[int]:
        """Convert to uint32 array for GPU execution"""
        self._resolve_ptrs()
        # Pad to uint32 boundary
        while len(self.bytecode) % 4 != 0:
            self.bytecode.append(0)
        return [
            int.from_bytes(self.bytecode[i:i + 4], byteorder="little", signed=False)
            for i in range(0, len(self.bytecode), 4)
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_ptrs(self) -> None:
        """Write recorded device pointers into the bytecode buffer."""
        if not self._ptrs:
            return
        for entry in self._ptrs:
            if len(entry) == 3:
                offset, ptr, _mode = entry
            else:
                offset, ptr = entry  # Backward compatibility
            # Accept objects providing .value (ctypes-style) or plain ints
            if hasattr(ptr, "value"):
                ptr_value = int(ptr.value)  # type: ignore[attr-defined]
            else:
                ptr_value = int(ptr)
            ptr_bytes = ptr_value.to_bytes(8, byteorder="little", signed=False)
            self.bytecode[offset:offset + 8] = ptr_bytes
        # Prevent duplicate writes on subsequent conversions
        self._ptrs.clear()


__all__ = [
    "ModularRPNEngine",
    "rpn_eval",
    "RPNProgram",
    "OP_SPARSE_LOAD",
    "OP_SMAV",
    "OP_ENTROPY_SUM",
    "OP_SIGMOID_APPROX",
    "OP_POINTER_LITERAL",
    "OP_TRM_MATVEC_512x1024",
    "OP_TRM_MATVEC_1024x512",
    "OP_TRM_VEC_ADD3_512",
    "OP_TRM_SWIGLU_512",
    "OP_TRM_SWIGLU_1024",
]
