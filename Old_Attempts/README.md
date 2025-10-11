# Old Attempts - Deprecated Implementations

This directory contains deprecated implementations that have been superseded by the sovereign architecture.

## Why These Were Deprecated

**CuPy Dependency Issues**:
- Required CuPy + cuda-python version matching
- CUDA header conflicts (cuda_fp16.h compilation errors)
- Library versioning hell (cuda.bindings API changes)
- Not aligned with sovereign mandate (pure ctypes + libcuda.so)

**Codex's Implementations**:
- Many were conceptual/placeholder code
- PTX syntax errors (CUDA_ERROR_INVALID_PTX)
- Mixed pseudo-code with real implementations
- Difficult to distinguish real vs fake implementations

## What Replaced Them

**Sovereign Architecture** (Current):
- Pure ctypes + CUDA Driver API (libcuda.so)
- Zero external dependencies (only Python stdlib)
- Hand-authored CUDA C++ → compiled to PTX
- All 15 Step8 kernels operational
- Latency validated (29.7µs < 95µs mandate)

## Directory Structure

```
Old_Attempts/
├── bridges/           # CuPy-based bridges (deprecated)
├── tests/             # Old test scripts using CuPy
├── scripts/           # Deprecated utility scripts
└── README.md          # This file
```

## Current Active Code

**Location**: `knowledge3d/cranium/`

**Active Components**:
- `sovereign/loader.py` - Pure ctypes CUDA Driver wrapper
- `sovereign/trm_launcher.py` - TRM recursive refinement
- `bridges/sovereign_bridges.py` - All 15 Step8 bridges (pure ctypes)
- `kernels/*.cu` - CUDA C++ source for all kernels
- `kernels/*.ptx` - Compiled PTX kernels (all valid)
- `ptx/modular_rpn_kernel.ptx` - RPN gem (787 lines)
- `ptx/trm_extensions.ptx` - TRM extensions (488 lines)

## Date Deprecated

2025-10-11 - Session with Claude implementing sovereign architecture

## Team Decision

**Daniel's Directive**: "Move anything deprecated to Old_Attempts folder, keep only what we're using in actual folders"
