"""Thin re-export wrapper for sovereign loader in ptx_runtime context.

This allows ptx_runtime modules to use relative imports like:
    from .sovereign.loader import gpu_malloc, launch_kernel

While maintaining the actual implementation in knowledge3d.cranium.sovereign
"""
