"""ThinkingTag-specific RPN bridge built on the parallel Tier-2 interpreter."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc

_KNOWN_RPN_OPCODES = {
    int(value)
    for value in vars(ropc).values()
    if isinstance(value, int) and 0 <= value <= 0xFFFF
}


def _encode_pointer_literal(ptr_value: int, rows: int, cols: int = 1) -> Tuple[float, float, float, float]:
    """Encode a 64-bit device pointer into four float scalars for OP_POINTER_LITERAL."""
    lo_bits = np.array([ptr_value & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
    hi_bits = np.array([(ptr_value >> 32) & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]
    return float(rows), float(cols), float(lo_bits), float(hi_bits)


@dataclass
class _DeviceTensor:
    ptr: loader.CUdeviceptr
    rows: int
    cols: int
    nbytes: int


class ThinkingTagRPNBridge:
    """High-performance RPN bridge specialised for ThinkingTag inference."""

    def __init__(self, tier: int = 2):
        if tier != 2:
            raise ValueError("ThinkingTagRPNBridge currently supports Tier-2 execution only")

        self.engine = AdvancedRPNEngine()

        # Weight cache: key -> _DeviceTensor
        self._weight_cache: Dict[int, _DeviceTensor] = {}

        # Vector workspaces keyed by length
        self._vector_buffers: Dict[int, Dict[int, loader.CUdeviceptr]] = {}
        # Matrix workspaces keyed by (rows, cols)
        self._matrix_buffers: Dict[Tuple[int, int], loader.CUdeviceptr] = {}

        # Constant vector cache (by object id)
        self._constant_vectors: Dict[int, loader.CUdeviceptr] = {}

        # Host buffers reused for readback
        self._host_buffer: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_temporal(
        self,
        input_vec: np.ndarray,
        weights: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Execute the temporal (full) ThinkingTag pipeline using parallel RPN.

        Args:
            input_vec: Input embedding (shape [D])
            weights: Dict containing dense weight matrices W1, W2, W3
            mask: Optional gating vector for the second layer (broadcast to shape of W2 output)

        Returns:
            (fused_output, entropy)
        """
        if input_vec.ndim != 1:
            raise ValueError("input_vec must be a flat vector")

        # Prepare device tensors
        d_input = self._upload_constant_vector(input_vec)
        d_w1 = self._upload_matrix(weights['W1'])
        d_w2 = self._upload_matrix(weights['W2'])
        d_w3 = self._upload_matrix(weights['W3'])

        hidden1, input_dim = d_w1.rows, d_w1.cols
        hidden2, _ = d_w2.rows, d_w2.cols
        output_dim, _ = d_w3.rows, d_w3.cols

        d_hidden1 = self._get_vector_buffer(hidden1, slot=0)
        d_hidden2 = self._get_vector_buffer(hidden2, slot=1)
        d_output = self._get_vector_buffer(output_dim, slot=2)

        # Optional mask (defaults to ones)
        if mask is None:
            mask_arr = np.ones(hidden2, dtype=np.float32)
        else:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.size != hidden2:
                raise ValueError(f"mask length {mask_arr.size} != hidden2 {hidden2}")
        d_mask = self._upload_constant_vector(mask_arr, length=hidden2)

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows: int, cols: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows, cols))

        # Layer 1: hidden1 = ReLU(W1 @ input)
        append_pointer(d_hidden1, hidden1, 1)
        append_pointer(d_w1.ptr, hidden1, input_dim)
        append_pointer(d_input, input_dim, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_hidden1, hidden1, 1)
        op_codes.append(ropc.OP_VECTOR_RELU)

        # Layer 2: hidden2 = ReLU((W2 @ hidden1) * mask)
        append_pointer(d_hidden2, hidden2, 1)
        append_pointer(d_w2.ptr, hidden2, hidden1)
        append_pointer(d_hidden1, hidden1, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_hidden2, hidden2, 1)
        append_pointer(d_mask, hidden2, 1)
        op_codes.append(ropc.OP_VECTOR_MUL_F32)

        append_pointer(d_hidden2, hidden2, 1)
        op_codes.append(ropc.OP_VECTOR_RELU)

        # Layer 3: output = sigmoid(W3 @ hidden2)
        append_pointer(d_output, output_dim, 1)
        append_pointer(d_w3.ptr, output_dim, hidden2)
        append_pointer(d_hidden2, hidden2, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_output, output_dim, 1)
        op_codes.append(ropc.OP_VECTOR_SIGMOID)

        append_pointer(d_output, output_dim, 1)
        op_codes.append(ropc.OP_ENTROPY_SUM)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        stack = self.engine.execute_program(0, op_np, scalars_np)

        # Read back output vector
        host_output = self._fetch_vector(d_output, output_dim)

        # Entropy is top of stack (scalar)
        entropy = float(stack[-1, 0]) if stack.size else float(self._compute_entropy_cpu(host_output))

        return host_output, entropy

    def execute_spatial(
        self,
        input_vec: np.ndarray,
        weights: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Spatial-only execution (no temporal mask)."""
        if input_vec.ndim != 1:
            raise ValueError("input_vec must be a flat vector")

        d_input = self._upload_vector(input_vec)
        d_w1 = self._upload_matrix(weights['W1'])
        d_w2 = self._upload_matrix(weights['W2'])
        d_w3 = self._upload_matrix(weights['W3'])

        hidden1, input_dim = d_w1.rows, d_w1.cols
        hidden2, _ = d_w2.rows, d_w2.cols
        output_dim, _ = d_w3.rows, d_w3.cols

        d_hidden1 = self._get_vector_buffer(hidden1, slot=0)
        d_hidden2 = self._get_vector_buffer(hidden2, slot=1)
        d_output = self._get_vector_buffer(output_dim, slot=2)

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows: int, cols: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows, cols))

        # Layer 1
        append_pointer(d_hidden1, hidden1, 1)
        append_pointer(d_w1.ptr, hidden1, input_dim)
        append_pointer(d_input, input_dim, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_hidden1, hidden1, 1)
        op_codes.append(ropc.OP_VECTOR_RELU)

        # Layer 2
        append_pointer(d_hidden2, hidden2, 1)
        append_pointer(d_w2.ptr, hidden2, hidden1)
        append_pointer(d_hidden1, hidden1, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_hidden2, hidden2, 1)
        op_codes.append(ropc.OP_VECTOR_RELU)

        # Layer 3
        append_pointer(d_output, output_dim, 1)
        append_pointer(d_w3.ptr, output_dim, hidden2)
        append_pointer(d_hidden2, hidden2, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        append_pointer(d_output, output_dim, 1)
        op_codes.append(ropc.OP_VECTOR_SIGMOID)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        self.engine.execute_program(0, op_np, scalars_np)
        return self._fetch_vector(d_output, output_dim)

    def compute_temporal_mask(
        self,
        context: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute temporal mask, coherence, and activity signals on the GPU.

        Args:
            context: Temporal context matrix shaped (T, D).
            threshold: Optional coherence threshold. When None, derived from context magnitude.

        Returns:
            Tuple of (mask, coherence, activity), each length-D float32 arrays.
        """
        ctx = np.asarray(context, dtype=np.float32)
        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]
        if ctx.ndim != 2:
            raise ValueError("context must be 1D or 2D")

        time_steps, feature_dim = ctx.shape
        ctx_ptr = self._upload_temp_matrix(ctx)

        coherence_ptr = self._get_vector_buffer(feature_dim, slot=0)
        mask_ptr = self._get_vector_buffer(feature_dim, slot=1)
        activity_ptr = self._get_vector_buffer(feature_dim, slot=2)

        if threshold is None:
            threshold = float(np.mean(np.abs(ctx)))

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows: int, cols: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows, cols))

        # Coherence scores
        append_pointer(coherence_ptr, feature_dim, 1)
        append_pointer(ctx_ptr, time_steps, feature_dim)
        op_codes.append(ropc.OP_TEMPORAL_COHERENCE)

        # Activity proxy (mean abs context)
        append_pointer(activity_ptr, feature_dim, 1)
        append_pointer(ctx_ptr, time_steps, feature_dim)
        op_codes.append(ropc.OP_TEMPORAL_AGGREGATE)

        # Mask derivation from coherence
        append_pointer(mask_ptr, feature_dim, 1)
        append_pointer(coherence_ptr, feature_dim, 1)
        op_codes.append(0x00)  # literal scalar
        scalars.append(float(threshold))
        op_codes.append(ropc.OP_TEMPORAL_MASK)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        self.engine.execute_program(
            0,
            op_np,
            scalars_np,
        )

        mask_host = self._fetch_vector(mask_ptr, feature_dim)
        coherence_host = self._fetch_vector(coherence_ptr, feature_dim)
        activity_host = self._fetch_vector(activity_ptr, feature_dim)
        return mask_host, coherence_host, activity_host

    def _execute_rpn_program(
        self,
        program: Iterable[Union[int, float]],
    ) -> np.ndarray:
        """
        Execute an ad-hoc Tier-2 RPN program for testing.

        Scalars push via OP_LITERAL (0x00); recognised opcode integers are emitted
        directly, while any other integer token is treated as a scalar literal.
        Returns the scalar lane (X component) of the resulting stack.
        """
        op_codes: list[int] = []
        scalars: list[float] = []

        for token in program:
            if isinstance(token, (float, np.floating)):
                op_codes.append(0x00)
                scalars.append(float(token))
            elif isinstance(token, (int, np.integer)):
                opcode = int(token)
                if opcode in _KNOWN_RPN_OPCODES:
                    op_codes.append(opcode)
                else:
                    op_codes.append(0x00)
                    scalars.append(float(opcode))
            else:
                raise TypeError("program tokens must be ints or floats")

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        stack = self.engine.execute_program(0, op_np, scalars_np)
        if stack.size == 0:
            return np.zeros(0, dtype=np.float32)
        return stack[:, 0].copy()

    def cleanup(self) -> None:
        """Release GPU resources."""
        for tensor in self._weight_cache.values():
            loader.gpu_free(tensor.ptr)
        self._weight_cache.clear()

        for slot_map in self._vector_buffers.values():
            for ptr in slot_map.values():
                loader.gpu_free(ptr)
        self._vector_buffers.clear()

        for ptr in self._matrix_buffers.values():
            loader.gpu_free(ptr)
        self._matrix_buffers.clear()

        for ptr in self._constant_vectors.values():
            loader.gpu_free(ptr)
        self._constant_vectors.clear()

        self.engine.cleanup()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _upload_matrix(self, matrix: np.ndarray) -> _DeviceTensor:
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Weights must be numpy arrays")
        matrix = np.asarray(matrix, dtype=np.float32)

        key = id(matrix)
        cached = self._weight_cache.get(key)
        if cached and cached.nbytes == matrix.nbytes:
            loader.memcpy_htod(
                cached.ptr,
                matrix.ctypes.data_as(ctypes.c_void_p),
                matrix.nbytes,
            )
            return cached

        if cached:
            loader.gpu_free(cached.ptr)

        ptr = loader.gpu_malloc(matrix.nbytes)
        loader.memcpy_htod(ptr, matrix.ctypes.data_as(ctypes.c_void_p), matrix.nbytes)
        tensor = _DeviceTensor(ptr=ptr, rows=matrix.shape[0], cols=matrix.shape[1], nbytes=matrix.nbytes)
        self._weight_cache[key] = tensor
        return tensor

    def _get_vector_buffer(self, length: int, slot: int = 0) -> loader.CUdeviceptr:
        slots = self._vector_buffers.setdefault(length, {})
        ptr = slots.get(slot)
        if ptr is None:
            ptr = loader.gpu_malloc(length * 4)
            slots[slot] = ptr
        return ptr

    def _get_matrix_buffer(self, rows: int, cols: int) -> loader.CUdeviceptr:
        key = (rows, cols)
        ptr = self._matrix_buffers.get(key)
        required = rows * cols * 4
        if ptr is None:
            ptr = loader.gpu_malloc(required)
            self._matrix_buffers[key] = ptr
        return ptr

    def _upload_temp_matrix(self, matrix: np.ndarray) -> loader.CUdeviceptr:
        mat = np.ascontiguousarray(matrix, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("Temporary matrix upload expects 2D array")
        rows, cols = mat.shape
        ptr = self._get_matrix_buffer(rows, cols)
        loader.memcpy_htod(ptr, mat.ctypes.data_as(ctypes.c_void_p), rows * cols * 4)
        return ptr

    def _upload_vector(self, vector: np.ndarray, length_override: Optional[int] = None) -> loader.CUdeviceptr:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        length = length_override or vec.size
        if vec.size != length:
            vec = np.resize(vec, length)
        ptr = self._get_vector_buffer(length)
        loader.memcpy_htod(ptr, vec.ctypes.data_as(ctypes.c_void_p), length * 4)
        return ptr

    def _upload_constant_vector(self, vector: np.ndarray, length: Optional[int] = None) -> loader.CUdeviceptr:
        arr = np.asarray(vector, dtype=np.float32).flatten()
        length = length or arr.size
        if arr.size != length:
            arr = np.resize(arr, length)

        key = id(vector)
        ptr = self._constant_vectors.get(key)
        if ptr is None:
            ptr = loader.gpu_malloc(length * 4)
            self._constant_vectors[key] = ptr
        loader.memcpy_htod(ptr, arr.ctypes.data_as(ctypes.c_void_p), length * 4)
        return ptr

    def _fetch_vector(self, ptr: loader.CUdeviceptr, length: int) -> np.ndarray:
        host = self._host_buffer.get(length)
        if host is None:
            host = np.zeros(length, dtype=np.float32)
            self._host_buffer[length] = host
        loader.memcpy_dtoh(host.ctypes.data_as(ctypes.c_void_p), ptr, length * 4)
        return host.copy()

    def _fetch_matrix(self, ptr: loader.CUdeviceptr, rows: int, cols: int) -> np.ndarray:
        host = np.zeros((rows, cols), dtype=np.float32)
        loader.memcpy_dtoh(host.ctypes.data_as(ctypes.c_void_p), ptr, rows * cols * 4)
        return host

    def _test_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        mat = np.ascontiguousarray(matrix, dtype=np.float32)
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if mat.ndim != 2 or vec.ndim != 1:
            raise ValueError("matvec test expects 2D matrix and 1D vector")
        rows, cols = mat.shape
        if vec.size != cols:
            raise ValueError(f"Vector length {vec.size} != matrix cols {cols}")

        d_matrix = self._upload_temp_matrix(mat)
        d_vector = self._upload_constant_vector(vec, length=vec.size)
        d_output = self._get_vector_buffer(rows)

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows_val: int, cols_val: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows_val, cols_val))

        append_pointer(d_output, rows, 1)
        append_pointer(d_matrix, rows, cols)
        append_pointer(d_vector, vec.size, 1)
        op_codes.append(ropc.OP_MATVEC_F32)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        self.engine.execute_program(0, op_np, scalars_np)
        return self._fetch_vector(d_output, rows)

    def _test_matmul_small(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mat_a = np.ascontiguousarray(a, dtype=np.float32)
        mat_b = np.ascontiguousarray(b, dtype=np.float32)
        if mat_a.ndim != 2 or mat_b.ndim != 2:
            raise ValueError("matmul test expects two 2D matrices")
        if mat_a.shape[1] != mat_b.shape[0]:
            raise ValueError("Incompatible shapes for matmul test")

        tensor_a = self._upload_matrix(mat_a)
        tensor_b = self._upload_matrix(mat_b)
        rows, cols = mat_a.shape[0], mat_b.shape[1]
        dest_ptr = self._get_matrix_buffer(rows, cols)

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows_val: int, cols_val: int) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows_val, cols_val))

        append_pointer(dest_ptr, rows, cols)
        append_pointer(tensor_a.ptr, mat_a.shape[0], mat_a.shape[1])
        append_pointer(tensor_b.ptr, mat_b.shape[0], mat_b.shape[1])
        op_codes.append(ropc.OP_MATMUL_SMALL)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        self.engine.execute_program(0, op_np, scalars_np)
        return self._fetch_matrix(dest_ptr, rows, cols)

    def _test_dot_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        vec_query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        mat_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if mat_vectors.ndim != 2:
            raise ValueError("vectors must be 2D matrix")
        if vec_query.size != mat_vectors.shape[1]:
            raise ValueError("Query dimension mismatch in dot batch test")

        tensor_vectors = self._upload_matrix(mat_vectors)
        query_ptr = self._upload_constant_vector(vec_query, length=vec_query.size)
        result_ptr = self._get_vector_buffer(mat_vectors.shape[0])

        op_codes: list[int] = []
        scalars: list[float] = []

        def append_pointer(ptr: loader.CUdeviceptr, rows_val: int, cols_val: int = 1) -> None:
            op_codes.append(ropc.OP_POINTER_LITERAL)
            scalars.extend(_encode_pointer_literal(int(ptr.value), rows_val, cols_val))

        append_pointer(result_ptr, mat_vectors.shape[0], 1)
        append_pointer(tensor_vectors.ptr, mat_vectors.shape[0], mat_vectors.shape[1])
        append_pointer(query_ptr, vec_query.size, 1)
        op_codes.append(ropc.OP_DOT_BATCH)

        op_np = np.asarray(op_codes, dtype=np.uint16)
        scalars_np = np.asarray(scalars, dtype=np.float32)

        self.engine.reset_instance(0)
        self.engine.execute_program(0, op_np, scalars_np)
        return self._fetch_vector(result_ptr, mat_vectors.shape[0])

    @staticmethod
    def _compute_entropy_cpu(probs: Iterable[float]) -> float:
        p = np.clip(np.asarray(list(probs), dtype=np.float32), 1e-6, 1.0)
        return float(-np.sum(p * np.log(p)))
