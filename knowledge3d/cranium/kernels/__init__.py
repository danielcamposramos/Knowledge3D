"""Python utility surface for sovereign kernel support modules.

This package intentionally exposes only small host-side helpers used by tests,
bootstrap code, and ingestion-side bridges. Hot-path reasoning remains in PTX.
"""

from . import kernel_loader, ptx_compiler

__all__ = ["kernel_loader", "ptx_compiler"]
