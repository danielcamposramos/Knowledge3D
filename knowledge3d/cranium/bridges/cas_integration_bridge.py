"""Sovereign CAS bridge backed by the live RPN/PTX execution surface.

This module keeps CAS honest: only operations that can be compiled to the
existing sovereign RPN kernels are exposed as live GPU execution. Higher-order
symbolic ambitions remain future work and must not silently fall back to Python
pattern matching in the hot path.
"""

from __future__ import annotations

import ast
import ctypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.bridges.rpn_config import (
    RPN_GRID_DIM,
    TIER2_BLOCK_DIM,
    TIER3_BLOCK_DIM,
)
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_ABS,
    OP_ADD,
    OP_COS,
    OP_DIV,
    OP_EXP,
    OP_LOG,
    OP_MAX,
    OP_MIN,
    OP_MUL,
    OP_NEGATE,
    OP_POWER,
    OP_SIN,
    OP_SQRT,
    OP_SUB,
    OP_SWAP,
    OP_TAN,
)
from knowledge3d.cranium.sovereign import loader


@dataclass
class CASExpression:
    expression: str
    variables: List[str]
    operation_type: str
    constraints: Optional[List[str]] = None
    domain: Optional[str] = None


@dataclass
class CASRPNProgram:
    op_codes: List[int]
    scalars: List[float]
    vectors: List[List[float]]
    matrices: List[float]
    complexity_tier: int
    cas_operation: str
    metadata: Dict[str, Any]


class _CASASTCompiler:
    """Compile a constrained arithmetic/logic subset to live RPN opcodes."""

    _UNARY_FUNCTIONS = {
        "sin": OP_SIN,
        "cos": OP_COS,
        "tan": OP_TAN,
        "sqrt": OP_SQRT,
        "exp": OP_EXP,
        "log": OP_LOG,
        "abs": OP_ABS,
    }
    _BINARY_FUNCTIONS = {
        "max": OP_MAX,
        "min": OP_MIN,
    }
    _BINOP_MAP = {
        ast.Add: OP_ADD,
        ast.Sub: OP_SUB,
        ast.Mult: OP_MUL,
        ast.Div: OP_DIV,
        ast.Pow: OP_POWER,
    }

    def compile(self, expr: CASExpression) -> CASRPNProgram:
        normalized = self._normalize_expression(expr.expression, expr.operation_type)
        tree = ast.parse(normalized, mode="eval")
        variable_values = self._parse_constraints(expr.constraints)

        op_codes: List[int] = []
        scalars: List[float] = []
        self._emit(tree.body, op_codes, scalars, variable_values)

        tier = 2
        if len(op_codes) > 32:
            tier = 3

        return CASRPNProgram(
            op_codes=op_codes,
            scalars=scalars,
            vectors=[],
            matrices=[],
            complexity_tier=tier,
            cas_operation=expr.operation_type,
            metadata={
                "normalized_expression": normalized,
                "variable_values": variable_values,
                "ast_root": tree.body.__class__.__name__,
            },
        )

    def compile_matrix(self, expr: CASExpression, output_shape: tuple[int, int]) -> CASRPNProgram:
        rows, cols = int(output_shape[0]), int(output_shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid matrix shape {output_shape}")
        normalized = self._normalize_expression(expr.expression, expr.operation_type)
        literal = self._try_parse_matrix_expression(normalized, rows, cols)
        if literal is not None:
            flattened, metadata = literal
            cas_operation = "matrix_scaled_literal" if "matrix_scale" in metadata else "matrix_literal"
            return CASRPNProgram(
                op_codes=[],
                scalars=[float(rows), float(cols)],
                vectors=[],
                matrices=flattened,
                complexity_tier=3,
                cas_operation=cas_operation,
                metadata={"shape": (rows, cols), **metadata},
            )

        scalar_program = self.compile(expr)
        return CASRPNProgram(
            op_codes=list(scalar_program.op_codes),
            scalars=list(scalar_program.scalars) + [float(rows), float(cols)],
            vectors=list(scalar_program.vectors),
            matrices=[0.0] * (rows * cols),
            complexity_tier=3,
            cas_operation="matrix_scalar_fill",
            metadata={"shape": (rows, cols), "matrix_mode": "scalar_fill", **scalar_program.metadata},
        )

    def _try_parse_matrix_expression(
        self,
        expression: str,
        rows: int,
        cols: int,
    ) -> Optional[tuple[List[float], Dict[str, Any]]]:
        text = expression.strip()
        scale = 1.0
        scaled = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*\*\s*(.+)", text)
        if scaled:
            scale = float(scaled.group(1))
            text = scaled.group(2).strip()

        lowered = text.lower()
        flattened: Optional[List[float]] = None
        metadata: Dict[str, Any] = {}

        if lowered in {"identity", "eye"} or re.fullmatch(r"(identity|eye)\(\d+\)", lowered):
            flattened = [0.0] * (rows * cols)
            for idx in range(min(rows, cols)):
                flattened[idx * cols + idx] = 1.0
            metadata["matrix_mode"] = "identity"
        elif lowered == "zeros" or re.fullmatch(r"zeros\(\d+\s*,\s*\d+\)", lowered):
            flattened = [0.0] * (rows * cols)
            metadata["matrix_mode"] = "zeros"
        elif lowered == "ones" or re.fullmatch(r"ones\(\d+\s*,\s*\d+\)", lowered):
            flattened = [1.0] * (rows * cols)
            metadata["matrix_mode"] = "ones"
        elif lowered.startswith("diag(") and lowered.endswith(")"):
            diag_values = [float(value.strip()) for value in text[5:-1].split(",") if value.strip()]
            if len(diag_values) != min(rows, cols):
                raise ValueError(f"diag() length {len(diag_values)} does not match requested shape {rows}x{cols}")
            flattened = [0.0] * (rows * cols)
            for idx, value in enumerate(diag_values):
                flattened[idx * cols + idx] = value
            metadata["matrix_mode"] = "diag"
        else:
            try:
                matrix_obj = ast.literal_eval(text)
            except Exception:
                return None
            if not isinstance(matrix_obj, (list, tuple)) or not matrix_obj:
                raise ValueError("Matrix expressions must be non-empty nested lists/tuples")
            matrix_rows = [list(row) for row in matrix_obj]
            row_count = len(matrix_rows)
            col_count = len(matrix_rows[0]) if matrix_rows else 0
            if row_count != rows or col_count != cols:
                raise ValueError(
                    f"Matrix literal shape {(row_count, col_count)} does not match requested {(rows, cols)}"
                )
            if any(len(row) != col_count for row in matrix_rows):
                raise ValueError("Matrix literal must be rectangular")
            flattened = [float(value) for row in matrix_rows for value in row]
            metadata["matrix_mode"] = "literal"

        if flattened is None:
            return None
        if scale != 1.0:
            metadata["matrix_scale"] = scale
        return flattened, metadata

    def _normalize_expression(self, expression: str, operation_type: str) -> str:
        text = (expression or "").strip().replace("^", "**")
        if operation_type == "solve" and "=" in text:
            lhs, rhs = text.split("=", 1)
            return f"(({lhs.strip()}) - ({rhs.strip()}))"
        return text

    def _parse_constraints(self, constraints: Optional[Sequence[str]]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for constraint in constraints or ():
            if "=" not in constraint:
                continue
            name, raw = constraint.split("=", 1)
            try:
                values[name.strip()] = float(ast.literal_eval(raw.strip()))
            except Exception:
                try:
                    values[name.strip()] = float(raw.strip())
                except Exception:
                    continue
        return values

    def _emit(
        self,
        node: ast.AST,
        op_codes: List[int],
        scalars: List[float],
        variable_values: Dict[str, float],
    ) -> None:
        if isinstance(node, ast.Constant):
            op_codes.append(0)
            scalars.append(float(node.value))
            return
        if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compatibility shape
            op_codes.append(0)
            scalars.append(float(node.n))
            return
        if isinstance(node, ast.Name):
            op_codes.append(0)
            scalars.append(float(variable_values.get(node.id, 1.0)))
            return
        if isinstance(node, ast.BinOp):
            self._emit(node.left, op_codes, scalars, variable_values)
            self._emit(node.right, op_codes, scalars, variable_values)
            opcode = self._BINOP_MAP.get(type(node.op))
            if opcode is None:
                raise ValueError(f"Unsupported CAS binary operator: {ast.dump(node.op)}")
            op_codes.append(opcode)
            return
        if isinstance(node, ast.UnaryOp):
            self._emit(node.operand, op_codes, scalars, variable_values)
            if isinstance(node.op, ast.USub):
                op_codes.append(OP_NEGATE)
                return
            if isinstance(node.op, ast.Not):
                # Boolean-style NOT on the live surface: 1 x SWAP SUB -> 1 - x
                op_codes.extend([0, OP_SWAP, OP_SUB])
                scalars.append(1.0)
                return
            raise ValueError(f"Unsupported CAS unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self._UNARY_FUNCTIONS and len(node.args) == 1:
                self._emit(node.args[0], op_codes, scalars, variable_values)
                op_codes.append(self._UNARY_FUNCTIONS[func_name])
                return
            if func_name in self._BINARY_FUNCTIONS and len(node.args) == 2:
                self._emit(node.args[0], op_codes, scalars, variable_values)
                self._emit(node.args[1], op_codes, scalars, variable_values)
                op_codes.append(self._BINARY_FUNCTIONS[func_name])
                return
            raise ValueError(f"Unsupported CAS function call: {func_name}")
        if isinstance(node, ast.BoolOp):
            if len(node.values) < 2:
                raise ValueError("Boolean CAS expressions require at least two operands")
            op = OP_MIN if isinstance(node.op, ast.And) else OP_MAX if isinstance(node.op, ast.Or) else None
            if op is None:
                raise ValueError(f"Unsupported CAS boolean operator: {ast.dump(node.op)}")
            self._emit(node.values[0], op_codes, scalars, variable_values)
            for value in node.values[1:]:
                self._emit(value, op_codes, scalars, variable_values)
                op_codes.append(op)
            return
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only single comparisons are supported in live CAS")
            self._emit(node.left, op_codes, scalars, variable_values)
            self._emit(node.comparators[0], op_codes, scalars, variable_values)
            op_codes.append(OP_SUB)
            return
        raise ValueError(f"Unsupported CAS expression node: {ast.dump(node)}")


class SovereignRPNCAS:
    """GPU-first CAS bridge using the live tiered RPN execution surface."""

    MAX_INSTANCES = 18
    STACK_DEPTH = 69

    def __init__(self):
        self.tier1_engine = LightweightRPNEngine()
        self.tier2_engine = self._load_kernel(
            Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel.ptx",
            "modular_rpn_geometric_kernel",
        )
        self.tier3_engine = self._load_kernel(
            Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel_extended.ptx",
            "modular_rpn_kernel_extended",
        )
        self.matrix_module, self.matrix_copy_kernel, self.matrix_scale_kernel = self._load_matrix_kernels()
        self.compiler = _CASASTCompiler()
        self.cas_executions = 0
        self.tier_distribution = {1: 0, 2: 0, 3: 0}

    def _load_kernel(self, ptx_path: Path, entry_name: str) -> tuple[Any, loader.CUdeviceptr]:
        if not ptx_path.exists():
            raise FileNotFoundError(f"CAS PTX kernel missing: {ptx_path}")
        kernel = loader.load_ptx_file(str(ptx_path), entry_name)
        device_state = loader.gpu_malloc(self.MAX_INSTANCES * 1040)
        zeros = (ctypes.c_uint8 * (self.MAX_INSTANCES * 1040))()
        loader.memcpy_htod(device_state, ctypes.cast(zeros, ctypes.c_void_p), ctypes.sizeof(zeros))
        return kernel, device_state

    def _load_matrix_kernels(self):
        source_path = Path(__file__).parent.parent / "kernels" / "cas_matrix_ops.cu"
        ptx_text = compile_cuda_file(source_path)
        module = loader.load_module(ptx_text.encode("utf-8"))
        return (
            module,
            loader.get_function(module, "matrix_literal_copy_kernel"),
            loader.get_function(module, "matrix_scale_kernel"),
        )

    def evaluate_expression(self, expr: CASExpression, instance_id: int = 0) -> float:
        self._validate_instance(instance_id)
        program = self.compiler.compile(expr)
        result = self._execute_program(program, instance_id)
        self.cas_executions += 1
        self.tier_distribution[program.complexity_tier] += 1
        return float(result)

    def evaluate_matrix(
        self,
        expr: CASExpression,
        output_shape: tuple[int, int],
        instance_id: int = 0,
    ) -> List[float]:
        self._validate_instance(instance_id)
        program = self.compiler.compile_matrix(expr, output_shape)
        result = self._execute_matrix_program(program, instance_id)
        self.cas_executions += 1
        self.tier_distribution[3] += 1
        return result

    def _validate_instance(self, instance_id: int) -> None:
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id}")

    def _execute_program(self, program: CASRPNProgram, instance_id: int) -> float:
        if program.complexity_tier == 2:
            kernel, state = self.tier2_engine
            return self._execute_kernel_program(kernel, state, program, instance_id, tier=2)
        kernel, state = self.tier3_engine
        return self._execute_kernel_program(kernel, state, program, instance_id, tier=3)

    def _execute_matrix_program(self, program: CASRPNProgram, instance_id: int) -> List[float]:
        if program.cas_operation in {"matrix_literal", "matrix_scaled_literal"}:
            return self._gpu_materialize_matrix(
                program.matrices,
                scale=float(program.metadata.get("matrix_scale", 1.0)),
            )
        kernel, state = self.tier3_engine
        scalar_value = self._execute_kernel_program(kernel, state, program, instance_id, tier=3)
        if program.cas_operation == "matrix_scalar_fill":
            return self._gpu_materialize_matrix([float(scalar_value)] * len(program.matrices))
        return self._gpu_materialize_matrix(program.matrices)

    def _gpu_materialize_matrix(self, values: Sequence[float], *, scale: float = 1.0) -> List[float]:
        matrix = list(float(value) for value in values)
        element_count = len(matrix)
        if element_count == 0:
            return []

        MatrixArray = ctypes.c_float * element_count
        input_arr = MatrixArray(*matrix)
        output_arr = MatrixArray(*([0.0] * element_count))
        d_input = loader.gpu_malloc(ctypes.sizeof(input_arr))
        d_output = loader.gpu_malloc(ctypes.sizeof(output_arr))
        try:
            loader.memcpy_htod(d_input, ctypes.cast(input_arr, ctypes.c_void_p), ctypes.sizeof(input_arr))
            kernel = self.matrix_scale_kernel if scale != 1.0 else self.matrix_copy_kernel
            params = [
                ctypes.c_uint64(int(d_input.value)),
                ctypes.c_uint64(int(d_output.value)),
            ]
            if scale != 1.0:
                params.append(ctypes.c_float(scale))
            params.append(ctypes.c_int32(element_count))
            loader.launch(
                kernel,
                grid=((element_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=params,
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(output_arr, ctypes.c_void_p), d_output, ctypes.sizeof(output_arr))
            return list(output_arr)
        finally:
            loader.gpu_free(d_input)
            loader.gpu_free(d_output)

    def _execute_kernel_program(
        self,
        kernel: Any,
        device_state: loader.CUdeviceptr,
        program: CASRPNProgram,
        instance_id: int,
        *,
        tier: int,
    ) -> float:
        op_arr = (ctypes.c_uint16 * len(program.op_codes))(*[int(op) for op in program.op_codes])
        scalar_values = [float(value) for value in program.scalars]
        scalar_arr = (ctypes.c_float * max(1, len(scalar_values)))(*scalar_values) if scalar_values else (ctypes.c_float * 1)(0.0)
        flat_vec = [float(value) for vec in program.vectors for value in vec]
        vec_arr = (ctypes.c_float * max(1, len(flat_vec)))(*flat_vec) if flat_vec else (ctypes.c_float * 1)(0.0)
        flat_mat = [float(value) for value in program.matrices]
        mat_arr = (ctypes.c_float * max(1, len(flat_mat)))(*flat_mat) if flat_mat else (ctypes.c_float * 1)(0.0)

        d_op_codes = loader.gpu_malloc(ctypes.sizeof(op_arr))
        d_scalars = loader.gpu_malloc(ctypes.sizeof(scalar_arr))
        d_vectors = loader.gpu_malloc(ctypes.sizeof(vec_arr))
        d_matrices = loader.gpu_malloc(ctypes.sizeof(mat_arr)) if tier == 3 else None

        try:
            loader.memcpy_htod(d_op_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
            loader.memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
            loader.memcpy_htod(d_vectors, ctypes.cast(vec_arr, ctypes.c_void_p), ctypes.sizeof(vec_arr))
            if d_matrices is not None:
                loader.memcpy_htod(d_matrices, ctypes.cast(mat_arr, ctypes.c_void_p), ctypes.sizeof(mat_arr))

            params = [
                ctypes.c_uint32(instance_id),
                ctypes.c_uint64(int(d_op_codes.value)),
                ctypes.c_uint64(int(d_scalars.value)),
                ctypes.c_uint64(int(d_vectors.value)),
            ]
            if d_matrices is not None:
                params.append(ctypes.c_uint64(int(d_matrices.value)))
            params.extend(
                [
                    ctypes.c_uint64(int(device_state.value)),
                    ctypes.c_uint32(len(program.op_codes)),
                ]
            )
            loader.launch(
                kernel,
                grid=(RPN_GRID_DIM, 1, 1),
                block=((TIER2_BLOCK_DIM if tier == 2 else TIER3_BLOCK_DIM), 1, 1),
                params=params,
            )
            loader.synchronize()
            return self._extract_result(device_state, instance_id)
        finally:
            loader.gpu_free(d_op_codes)
            loader.gpu_free(d_scalars)
            loader.gpu_free(d_vectors)
            if d_matrices is not None:
                loader.gpu_free(d_matrices)

    def _extract_result(self, device_state: loader.CUdeviceptr, instance_id: int) -> float:
        header = (ctypes.c_uint32 * 4)()
        instance_offset = instance_id * 1040
        loader.memcpy_dtoh(
            ctypes.cast(header, ctypes.c_void_p),
            loader.CUdeviceptr(int(device_state.value + instance_offset)),
            ctypes.sizeof(header),
        )
        size = int(header[1])
        if size == 0:
            raise RuntimeError("CAS GPU execution produced empty stack")
        stack_top = (header[0] + size - 1) & 63
        element_offset = instance_offset + 16 + stack_top * 16
        result_vec = (ctypes.c_float * 4)()
        loader.memcpy_dtoh(
            ctypes.cast(result_vec, ctypes.c_void_p),
            loader.CUdeviceptr(int(device_state.value + element_offset)),
            ctypes.sizeof(result_vec),
        )
        return float(result_vec[0])

    def spawn_math_core_units(self, count: int = 1) -> List["SovereignRPNCAS"]:
        return [SovereignRPNCAS() for _ in range(max(0, int(count)))]

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "cas_executions": self.cas_executions,
            "tier_distribution": dict(self.tier_distribution),
            "gpu_execution_mode": "gpu_first_rpn_ptx",
            "sovereign_gpu_execution": True,
            "matrix_kernel_surface": True,
            "max_instances": self.MAX_INSTANCES,
            "stack_depth": self.STACK_DEPTH,
        }

    def reset_instance(self, instance_id: int) -> None:
        self._validate_instance(instance_id)
        self.tier1_engine.reset_instance(instance_id)
        header_zero = (ctypes.c_uint32 * 4)()
        for _, device_state in (self.tier2_engine, self.tier3_engine):
            offset = instance_id * 1040
            loader.memcpy_htod(
                loader.CUdeviceptr(int(device_state.value + offset)),
                ctypes.cast(header_zero, ctypes.c_void_p),
                ctypes.sizeof(header_zero),
            )

    def cleanup(self) -> None:
        self.tier1_engine.cleanup()
        for _, device_state in (self.tier2_engine, self.tier3_engine):
            loader.gpu_free(device_state)


def integrate_cas_with_rpn() -> SovereignRPNCAS:
    return SovereignRPNCAS()


def demonstrate_sovereign_cas() -> None:
    cas = SovereignRPNCAS()
    try:
        samples = [
            CASExpression("2 + 3 * 4", [], "evaluate"),
            CASExpression("sin(0.5) + cos(0.5)", [], "evaluate"),
            CASExpression("1 and (0 or not 0)", [], "ternary_logic"),
            CASExpression("x**2 + 2*x + 1", ["x"], "evaluate", constraints=["x=2.5"]),
        ]
        for expr in samples:
            result = cas.evaluate_expression(expr)
            print(f"{expr.operation_type}: {expr.expression} -> {result}")
        print(cas.get_performance_stats())
    finally:
        cas.cleanup()


if __name__ == "__main__":
    demonstrate_sovereign_cas()
