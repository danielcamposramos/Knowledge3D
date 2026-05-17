# Archived: transfer_yard_tiered.py

**Original Path**: `knowledge3d/cranium/bridges/transfer_yard_tiered.py`  
**Archive Date**: 2026-04-18  
**Reason**: Python dataclass wrapper superseded by modular_rpn_kernel_lite_transfer_yard.ptx (GPU-resident)

## Why Archived

This module implements TransferYardStack dataclass + 3 TierEngine classes (731 LoC) as a Python abstraction over the Transfer Yard matrix stack. 

**K3D Paradigm Shift**: Transfer Yard is now exclusively GPU-resident:
- `modular_rpn_kernel_lite_transfer_yard.ptx` manages the addressable matrix stack on GPU
- No Python-side TransferYardStack needed; GPU kernels handle tier dispatch
- Python sees only the PTX kernel interface (load/execute/result retrieval)

This Python wrapper represented a pre-GPU-migration design; it is now redundant.

## Note on .ptx Files

The kernel variants (`modular_rpn_kernel.ptx`, `modular_rpn_kernel_lite.ptx`, `modular_rpn_kernel_lite_transfer_yard.ptx`) are NOT archived—all are generation-specific and active.

Only the Python wrapper class is archived.

---

*This file was moved as part of GPU-resident migration. See feedback_transfer_yard_is_the_addressable_matrix.md.*
