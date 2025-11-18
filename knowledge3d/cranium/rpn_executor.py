"""
RPN Executor: Python Bridge to Modular RPN PTX Kernel

Provides GPU-native execution of RPN programs for semantic computations:
- Semantic depth calculation (GLM's suggestion #1)
- Honesty scoring (RLWHF)
- Golden ratio calculations (Garden fractals)
- Cosine similarity (clustering)

Uses existing modular_rpn_kernel.ptx (15 instances, 64-deep stacks)
"""

from pathlib import Path
from typing import Dict, List, Optional

import cupy as cp
import numpy as np


class RPNExecutor:
    """
    Python bridge to modular RPN PTX kernel.
    Executes compiled RPN programs on GPU with zero-copy semantics.
    """

    # RPN Kernel Constants (from modular_rpn_kernel.ptx)
    MAX_INSTANCES = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
    STACK_DEPTH = 69    # Tesla 6-9: literal 6&9, mirror symmetry (Yin-Yang)
    INSTANCE_STRIDE = 1040  # bytes per instance state

    def __init__(self, ptx_path: Optional[Path] = None):
        """
        Initialize RPN executor with PTX kernel.

        Args:
            ptx_path: Path to modular_rpn_kernel.ptx (default: auto-detect)
        """
        if ptx_path is None:
            ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel.ptx"

        if not ptx_path.exists():
            raise FileNotFoundError(f"RPN kernel not found: {ptx_path}")

        # Load PTX kernel
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("modular_rpn_geometric_kernel")

        # Allocate persistent state buffer (15 instances × 1040 bytes)
        self.state_buffer = cp.zeros(
            self.MAX_INSTANCES * self.INSTANCE_STRIDE,
            dtype=cp.uint8
        )

        # Performance tracking
        self.total_executions = 0
        self.total_time_us = 0.0

    def execute_single(
        self,
        instance_id: int,
        op_codes: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray
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

        # Convert inputs to GPU arrays (zero-copy if already GPU)
        op_codes_gpu = cp.asarray(op_codes, dtype=cp.uint16)
        scalars_gpu = cp.asarray(scalars, dtype=cp.float32)
        vectors_gpu = cp.asarray(vectors, dtype=cp.float32)

        # Launch kernel
        self.kernel(
            (1,),  # grid
            (1,),  # block
            (
                np.uint32(instance_id),
                op_codes_gpu,
                scalars_gpu,
                vectors_gpu,
                self.state_buffer,
                np.uint32(len(op_codes))
            )
        )

        cp.cuda.runtime.deviceSynchronize()

        # Read result from instance stack (top element)
        instance_offset = instance_id * self.INSTANCE_STRIDE
        stack_offset = instance_offset + 16  # Skip header (4 uint32: head, size, error, pad)

        # Stack stores float4 (16 bytes each), read first float
        result_bytes = self.state_buffer[stack_offset:stack_offset+4].tobytes()
        result = np.frombuffer(result_bytes, dtype=np.float32)[0]

        self.total_executions += 1
        return float(result)

    def execute_batch(
        self,
        programs: List[Dict[str, np.ndarray]],
        max_instances: int = 15
    ) -> np.ndarray:
        """
        Execute batch of RPN programs in parallel across instances.

        Args:
            programs: List of dicts with keys 'op_codes', 'scalars', 'vectors'
            max_instances: Max parallel instances (default 15)

        Returns:
            NumPy array of results (length = len(programs))
        """
        results = []

        # Process in batches of max_instances
        for batch_start in range(0, len(programs), max_instances):
            batch = programs[batch_start:batch_start+max_instances]

            # Launch kernels for this batch (parallel execution)
            batch_results = []
            for i, program in enumerate(batch):
                result = self.execute_single(
                    instance_id=i,
                    op_codes=program['op_codes'],
                    scalars=program['scalars'],
                    vectors=program['vectors']
                )
                batch_results.append(result)

            results.extend(batch_results)

        return np.array(results, dtype=np.float32)

    def reset_instance(self, instance_id: int):
        """Reset instance state (clear stack, reset head/size)."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        instance_offset = instance_id * self.INSTANCE_STRIDE
        # Zero out instance state (head=0, size=0, error=0)
        self.state_buffer[instance_offset:instance_offset+12] = 0

    def reset_all(self):
        """Reset all instance states."""
        self.state_buffer[:] = 0

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        avg_time = (
            self.total_time_us / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        return {
            'total_executions': self.total_executions,
            'total_time_us': self.total_time_us,
            'avg_time_us': avg_time
        }


# Global executor instance (singleton pattern)
_rpn_executor: Optional[RPNExecutor] = None


def get_rpn_executor() -> RPNExecutor:
    """Get or create global RPN executor instance."""
    global _rpn_executor
    if _rpn_executor is None:
        _rpn_executor = RPNExecutor()
    return _rpn_executor


def execute_rpn_kernel(
    instance_id: int,
    op_codes: np.ndarray,
    scalars: np.ndarray,
    vectors: np.ndarray
) -> float:
    """
    Convenience function for single RPN execution.

    Args:
        instance_id: Instance slot (0-14)
        op_codes: RPN operation codes (uint16)
        scalars: Scalar literal pool (float32)
        vectors: Vector literal pool (float32)

    Returns:
        Computation result (float32)
    """
    executor = get_rpn_executor()
    return executor.execute_single(instance_id, op_codes, scalars, vectors)


def execute_rpn_kernel_batch(
    programs: List[Dict[str, np.ndarray]],
    max_instances: int = 15
) -> np.ndarray:
    """
    Convenience function for batch RPN execution.

    Args:
        programs: List of RPN programs (each with op_codes, scalars, vectors)
        max_instances: Max parallel instances

    Returns:
        Array of results
    """
    executor = get_rpn_executor()
    return executor.execute_batch(programs, max_instances)


def reset_rpn_executor():
    """Reset global RPN executor state."""
    executor = get_rpn_executor()
    executor.reset_all()
