"""
Sovereign RPN Executor - Zero CuPy Version
==========================================

Python bridge to modular RPN PTX kernel using sovereign loader only.
Executes compiled RPN programs on GPU with zero dependencies on CuPy.

Uses existing modular_rpn_kernel.ptx (15 instances, 64-deep stacks).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Optional

import numpy as np

from knowledge3d.cranium.sovereign import loader


class SovereignRPNExecutor:
    """
    Sovereign RPN executor using PTX loader (no CuPy).
    Executes compiled RPN programs on GPU with zero-copy semantics.
    """

    # RPN Kernel Constants
    MAX_INSTANCES = 15
    STACK_DEPTH = 64
    INSTANCE_STRIDE = 1040  # bytes per instance state

    def __init__(self, ptx_path: Optional[Path] = None):
        """
        Initialize sovereign RPN executor.

        Args:
            ptx_path: Path to modular_rpn_kernel.ptx (default: auto-detect)
        """
        if ptx_path is None:
            ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel.ptx"

        if not ptx_path.exists():
            raise FileNotFoundError(f"RPN kernel not found: {ptx_path}")

        # Load PTX kernel via sovereign loader
        self.module = loader.load_module_from_file(str(ptx_path))
        self.kernel = loader.get_function(self.module, "modular_rpn_geometric_kernel")

        # Allocate persistent state buffer (15 instances × 1040 bytes)
        state_size = self.MAX_INSTANCES * self.INSTANCE_STRIDE
        self.state_buffer = loader.gpu_malloc(state_size)

        # Zero-initialize state buffer
        zeros = np.zeros(state_size, dtype=np.uint8)
        loader.memcpy_htod(self.state_buffer, zeros.ctypes.data, state_size)

        # Performance tracking
        self.total_executions = 0

    def execute_single(
        self,
        instance_id: int,
        op_codes: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray,
    ) -> float:
        """
        Execute single RPN program on specified instance.

        Args:
            instance_id: Instance slot (0-14)
            op_codes: RPN operation codes (uint16 array)
            scalars: Scalar literal pool (float32 array)
            vectors: Vector literal pool (float32 array, flat)

        Returns:
            Result from top of stack (float32)
        """
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id} (must be 0-14)")

        # Ensure contiguous arrays
        op_codes = np.ascontiguousarray(op_codes, dtype=np.uint16)
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Allocate GPU memory for inputs
        op_codes_gpu = loader.gpu_malloc(op_codes.nbytes)
        scalars_gpu = loader.gpu_malloc(scalars.nbytes)
        vectors_gpu = loader.gpu_malloc(vectors.nbytes)

        # Copy to GPU
        loader.memcpy_htod(op_codes_gpu, op_codes.ctypes.data, op_codes.nbytes)
        loader.memcpy_htod(scalars_gpu, scalars.ctypes.data, scalars.nbytes)
        loader.memcpy_htod(vectors_gpu, vectors.ctypes.data, vectors.nbytes)

        # Launch kernel
        loader.launch(
            self.kernel,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            params=[
                ctypes.c_uint32(instance_id),
                ctypes.c_uint64(op_codes_gpu.value),
                ctypes.c_uint64(scalars_gpu.value),
                ctypes.c_uint64(vectors_gpu.value),
                ctypes.c_uint64(self.state_buffer.value),
                ctypes.c_uint32(len(op_codes)),
            ],
        )

        loader.synchronize()

        # Read result from state buffer
        # State layout: head(4) + size(4) + error(4) + reserved(4) + stack[64][4]
        result_offset = instance_id * self.INSTANCE_STRIDE + 16  # Skip header
        result_bytes = np.empty(4, dtype=np.uint8)
        result_ptr = loader.CUdeviceptr(self.state_buffer.value + result_offset)
        loader.memcpy_dtoh(result_bytes.ctypes.data, result_ptr, 4)

        result = np.frombuffer(result_bytes, dtype=np.float32)[0]

        # Cleanup
        loader.gpu_free(op_codes_gpu)
        loader.gpu_free(scalars_gpu)
        loader.gpu_free(vectors_gpu)

        self.total_executions += 1
        return float(result)

    def execute_batch(
        self,
        programs: List[tuple],
        max_instances: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute multiple RPN programs in parallel across instances.

        Args:
            programs: List of (op_codes, scalars, vectors) tuples
            max_instances: Max parallel instances (default: MAX_INSTANCES)

        Returns:
            Results array (float32)
        """
        if max_instances is None:
            max_instances = self.MAX_INSTANCES

        max_instances = min(max_instances, self.MAX_INSTANCES, len(programs))
        results = np.zeros(len(programs), dtype=np.float32)

        # Process in batches
        for batch_start in range(0, len(programs), max_instances):
            batch_end = min(batch_start + max_instances, len(programs))
            batch_size = batch_end - batch_start

            # Execute batch programs
            for i in range(batch_size):
                program_idx = batch_start + i
                op_codes, scalars, vectors = programs[program_idx]
                results[program_idx] = self.execute_single(i, op_codes, scalars, vectors)

        return results

    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, "state_buffer"):
            try:
                loader.gpu_free(self.state_buffer)
            except Exception:
                pass


# Singleton accessor
_sovereign_rpn_executor: Optional[SovereignRPNExecutor] = None


def get_sovereign_rpn_executor() -> SovereignRPNExecutor:
    """Get or create singleton sovereign RPN executor."""
    global _sovereign_rpn_executor
    if _sovereign_rpn_executor is None:
        _sovereign_rpn_executor = SovereignRPNExecutor()
    return _sovereign_rpn_executor
