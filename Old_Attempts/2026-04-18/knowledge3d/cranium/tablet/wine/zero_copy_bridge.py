"""
Zero-Copy Bridge
Provides cuMemHostAlloc interface for zero-copy memory management.
This bridge maintains architectural sovereignty by using only GPU operations
and integrating with the existing K3D memory management system.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, Any
from ...kernels.zero_copy_memory_manager import (
    cuMemHostAlloc_wrapper,
    cuMemFreeHost_wrapper,
    zero_copy_memcpy_async
)


class ZeroCopyBridge:
    """
    Bridge that provides zero-copy memory interface for external 3D models.
    Uses cuMemHostAlloc for pinned host memory that can be accessed by GPU.
    """
    
    def __init__(self):
        self._allocated_buffers = {}  # Track allocated buffers
        self._buffer_sizes = {}       # Track buffer sizes
        self._next_buffer_id = 1      # Next buffer ID counter
    
    def allocate_zero_copy_buffer(self, size: int) -> int:
        """
        Allocate zero-copy buffer using cuMemHostAlloc.
        
        Args:
            size: Size of buffer in bytes
            
        Returns:
            Buffer ID that can be used for GPU operations
        """
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        
        # Allocate pinned host memory using cuMemHostAlloc
        buffer_ptr = cuMemHostAlloc_wrapper(size)
        
        if buffer_ptr is None:
            raise RuntimeError(f"Failed to allocate zero-copy buffer of size {size}")
        
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1
        
        self._allocated_buffers[buffer_id] = buffer_ptr
        self._buffer_sizes[buffer_id] = size
        
        return buffer_id
    
    def free_zero_copy_buffer(self, buffer_id: int) -> bool:
        """
        Free zero-copy buffer using cuMemFreeHost.
        
        Args:
            buffer_id: ID of buffer to free
            
        Returns:
            True if successful, False otherwise
        """
        if buffer_id not in self._allocated_buffers:
            return False
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        
        # Free pinned host memory
        success = cuMemFreeHost_wrapper(buffer_ptr)
        
        if success:
            del self._allocated_buffers[buffer_id]
            del self._buffer_sizes[buffer_id]
        
        return success
    
    def copy_to_zero_copy(self, buffer_id: int, data: np.ndarray) -> bool:
        """
        Copy data to zero-copy buffer asynchronously.
        
        Args:
            buffer_id: ID of destination buffer
            data: NumPy array to copy
            
        Returns:
            True if successful, False otherwise
        """
        if buffer_id not in self._allocated_buffers:
            return False
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        buffer_size = self._buffer_sizes[buffer_id]
        
        # Check data size
        data_size = data.nbytes
        if data_size > buffer_size:
            raise ValueError(f"Data size {data_size} exceeds buffer size {buffer_size}")
        
        # Perform asynchronous copy
        success = zero_copy_memcpy_async(buffer_ptr, data.ctypes.data, data_size)
        
        return success
    
    def copy_from_zero_copy(self, buffer_id: int, size: int) -> Optional[np.ndarray]:
        """
        Copy data from zero-copy buffer.
        
        Args:
            buffer_id: ID of source buffer
            size: Number of bytes to copy
            
        Returns:
            NumPy array with copied data, or None if failed
        """
        if buffer_id not in self._allocated_buffers:
            return None
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        buffer_size = self._buffer_sizes[buffer_id]
        
        if size > buffer_size:
            raise ValueError(f"Copy size {size} exceeds buffer size {buffer_size}")
        
        # Create NumPy array and copy data
        result = np.empty(size, dtype=np.uint8)
        
        # Copy from zero-copy buffer
        ctypes.memmove(result.ctypes.data, buffer_ptr, size)
        
        return result
    
    def get_buffer_info(self, buffer_id: int) -> Optional[dict]:
        """
        Get information about a zero-copy buffer.
        
        Args:
            buffer_id: ID of buffer
            
        Returns:
            Dictionary with buffer information, or None if not found
        """
        if buffer_id not in self._allocated_buffers:
            return None
        
        return {
            'buffer_id': buffer_id,
            'size': self._buffer_sizes[buffer_id],
            'pointer': self._allocated_buffers[buffer_id],
            'type': 'zero_copy',
            'pinned': True
        }
    
    def create_zero_copy_view(self, buffer_id: int, dtype: np.dtype, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Create a NumPy view of zero-copy buffer without copying.
        
        Args:
            buffer_id: ID of buffer
            dtype: NumPy data type
            shape: Shape of the view
            
        Returns:
            NumPy array view, or None if failed
        """
        if buffer_id not in self._allocated_buffers:
            return None
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        buffer_size = self._buffer_sizes[buffer_id]
        
        # Calculate required size
        dtype_size = np.dtype(dtype).itemsize
        required_size = np.prod(shape) * dtype_size
        
        if required_size > buffer_size:
            raise ValueError(f"View size {required_size} exceeds buffer size {buffer_size}")
        
        # Create view using the buffer pointer
        # This creates a zero-copy view of the pinned memory
        view = np.ctypeslib.as_array(
            ctypes.cast(buffer_ptr, ctypes.POINTER(dtype)),
            shape=shape
        )
        
        return view
    
    def transfer_to_gpu(self, buffer_id: int, gpu_ptr: int, size: int) -> bool:
        """
        Transfer data from zero-copy buffer to GPU memory.
        
        Args:
            buffer_id: ID of source buffer
            gpu_ptr: GPU memory pointer
            size: Number of bytes to transfer
            
        Returns:
            True if successful, False otherwise
        """
        if buffer_id not in self._allocated_buffers:
            return False
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        
        # Use the async copy function for GPU transfer
        success = zero_copy_memcpy_async(gpu_ptr, buffer_ptr, size)
        
        return success
    
    def transfer_from_gpu(self, buffer_id: int, gpu_ptr: int, size: int) -> bool:
        """
        Transfer data from GPU memory to zero-copy buffer.
        
        Args:
            buffer_id: ID of destination buffer
            gpu_ptr: GPU memory pointer
            size: Number of bytes to transfer
            
        Returns:
            True if successful, False otherwise
        """
        if buffer_id not in self._allocated_buffers:
            return False
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        
        # Use the async copy function for GPU transfer
        success = zero_copy_memcpy_async(buffer_ptr, gpu_ptr, size)
        
        return success
    
    def get_memory_stats(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        total_allocated = sum(self._buffer_sizes.values())
        buffer_count = len(self._allocated_buffers)
        
        return {
            'total_allocated': total_allocated,
            'buffer_count': buffer_count,
            'largest_buffer': max(self._buffer_sizes.values()) if self._buffer_sizes else 0,
            'smallest_buffer': min(self._buffer_sizes.values()) if self._buffer_sizes else 0,
            'average_size': total_allocated / buffer_count if buffer_count > 0 else 0
        }
    
    def cleanup_all_buffers(self) -> int:
        """
        Free all allocated zero-copy buffers.
        
        Returns:
            Number of buffers freed
        """
        freed_count = 0
        buffer_ids = list(self._allocated_buffers.keys())
        
        for buffer_id in buffer_ids:
            if self.free_zero_copy_buffer(buffer_id):
                freed_count += 1
        
        return freed_count
    
    def validate_buffer_alignment(self, buffer_id: int, alignment: int = 256) -> bool:
        """
        Validate that buffer is properly aligned for GPU operations.
        
        Args:
            buffer_id: ID of buffer to check
            alignment: Required alignment in bytes
            
        Returns:
            True if properly aligned, False otherwise
        """
        if buffer_id not in self._allocated_buffers:
            return False
        
        buffer_ptr = self._allocated_buffers[buffer_id]
        
        # Check alignment
        return (buffer_ptr % alignment) == 0
    
    def get_capabilities(self) -> dict:
        """
        Get capabilities of the zero-copy bridge.
        
        Returns:
            Dictionary with capability information
        """
        return {
            'supports_async_copy': True,
            'supports_gpu_transfer': True,
            'supports_zero_copy_views': True,
            'memory_type': 'pinned_host',
            'alignment_requirement': 256,
            'max_buffer_size': 2**32 - 1,  # 4GB limit
            'api_version': '1.0'
        }