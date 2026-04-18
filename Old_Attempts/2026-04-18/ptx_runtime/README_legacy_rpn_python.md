# Archived: legacy_rpn_python.py

**Original Path**: `knowledge3d/core/legacy_rpn_python.py`  
**Archive Date**: 2026-04-18  
**Reason**: Named "legacy"; implements Python-side RPN (superseded by GPU kernels)

## Why Archived

This module is explicitly named "legacy" and implements RPN operations in Python. K3D architecture mandates:
- RPN execution lives on GPU via PTX kernels (modular_rpn_kernel.ptx variants)
- Python serves bootstrap + I/O only (~200 lines target)
- No Python-side RPN interpreter in hot path

All RPN programs are now composed and executed via GPU kernels, rendering this Python implementation obsolete.

## Replacement

Use `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` and PTX kernel dispatch for all RPN execution.

---

*This file was moved as part of legacy cleanup. See CLAUDE.md "Python = boot + I/O only".*
