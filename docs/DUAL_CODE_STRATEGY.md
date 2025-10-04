# Dual-Code Strategy: HR/MR Optimization for K3D

## Overview

The **Dual-Code Paradigm** maintains two representations of K3D source code:

- **HR (Human-Readable)**: Idiomatic Python/JS with rich documentation, comments, and explanatory structure. This is the **authoritative source** stored in the repository.
- **MR (Machine-Runtime)**: Comment-free, docstring-stripped, whitespace-compressed output optimized for execution. Generated on-demand via the `codeopt` tool and **never committed to git**.

This separation follows the principle of **code-as-LOD (Level of Detail)**: HR for development/collaboration, MR for production/multi-instance deployment.

---

## When to Use MR (Machine-Runtime)

### ✅ **Use Cases Where MR Provides Value**

#### 1. **Multi-Instance Parallel Runs**
When running **multiple processes** simultaneously (e.g., 4+ vision caption workers, parallel trainers), each process loads the entire module tree into memory. MR reduces per-instance footprint by **20-40%**.

**Example scenario**:
```bash
# Running 10 parallel RLWHF trainers
for i in {1..10}; do
  PYTHONPATH=../Knowledge3D.local/mr:. \
  python -m knowledge3d.tools.phase25.rpn_policy_trainer \
    --batch $i &
done
```

**Savings**: 10 instances × 91KB fused_head.py → ~650KB total (vs ~910KB HR), freeing **260KB** for additional Galaxy chunks in shared memory.

#### 2. **Edge Deployment (Raspberry Pi, Jetson Nano, Mobile)**
Resource-constrained devices benefit dramatically from reduced code footprint:

- **Raspberry Pi 4 (4GB RAM)**: Every MB counts when running PTX kernels + Galaxy GLBs
- **Jetson Nano (2GB RAM)**: MR allows fitting the full K3D stack + 200MB Galaxy
- **Browser WASM** (future): Smaller bundle sizes = faster load times

#### 3. **Production Hot-Path Optimization**
Modules loaded **thousands of times per session**:

- `knowledge3d/cranium/fused_head.py` (every prediction)
- `knowledge3d/cranium/ptx/ptx_ops.py` (every RPN call)
- `knowledge3d/skills/{vision,audio,video}.py` (embedding extraction)

Even 1-2% savings compound over 10,000+ imports.

---

### ❌ **When NOT to Use MR**

#### 1. **Single-Instance Development**
If you're running one viewer + one live server on a development machine with 16GB+ RAM, MR provides **negligible benefit**. Stick with HR for better stack traces and debuggability.

#### 2. **Already-Lean Modules**
Analysis of `fused_head.py`:
- Total size: 91KB
- Docstrings: **1.2%** of file size
- Estimated MR savings: **10-15%** (~9-14KB)

For modules with minimal documentation, MR is **low-value**.

#### 3. **CUDA Kernels (.cu Files)**
PTX kernels are already compiled to binary. Dual-code doesn't apply.

---

## Tier-Based Compilation Strategy

The Makefile provides **three tiers** of MR compilation:

### **Tier 1: Hot-Path Core** (`make compile-mr-core`)
**Target**: Modules imported on **every request**

**Includes**:
- `knowledge3d/cranium/fused_head.py`
- `knowledge3d/cranium/ptx/*`
- `knowledge3d/skills/*`
- `knowledge3d/bridge/live_server.py`

**When to use**:
- Multi-instance live servers (e.g., fog node deployment)
- Edge devices with <4GB RAM
- Production inference servers handling 100+ req/s

**Expected savings**: ~20-30% footprint reduction on hot-path imports

---

### **Tier 2: Training Scripts** (`make compile-mr-trainers`)
**Target**: Modules used in **parallel training runs**

**Includes**:
- `knowledge3d/tools/phase18/*_trainer.py`
- `knowledge3d/tools/phase25/*_trainer.py`
- `knowledge3d/tools/*_evaluator.py`

**When to use**:
- Running 4+ parallel trainers (consistency, RLWHF, thinking-tag)
- Multi-GPU distributed training
- Batch evaluation sweeps (AIME, ARC, Wiki)

**Expected savings**: ~25-35% reduction when 10+ instances are active

---

### **Tier 3: Full Repository** (`make compile-mr-all`)
**Target**: **Entire codebase** (k3dgen, knowledge3d, viewer)

**Includes**: Everything (Python + JS/TS)

**When to use**:
- Production deployment to edge clusters
- Dockerized multi-tenant environments
- Aggressive memory optimization (e.g., running on 2GB Jetson Nano)

**Expected savings**: ~30-40% total codebase footprint

---

## Usage Workflow

### 1. **Compile MR Outputs**

```bash
# Tier 1: Core modules only
make compile-mr-core

# Tier 2: Add trainers
make compile-mr-trainers

# Tier 3: Full repo
make compile-mr-all
```

Output goes to `../Knowledge3D.local/mr/` (never committed to git).

### 2. **Run with MR**

Prepend MR directory to PYTHONPATH:

```bash
# Single command
PYTHONPATH=../Knowledge3D.local/mr:. python -m knowledge3d.bridge.live_server

# Set globally for session
export PYTHONPATH=../Knowledge3D.local/mr:.
python -m knowledge3d.tools.phase25.rpn_policy_trainer
```

### 3. **Verify Savings**

```bash
make mr-report
```

Output:
```
=== MR Savings Report ===
Tier 1 (core):
  fused_head.py: 78K (vs 91K HR)
  ptx/: 42K (vs 54K HR)
Tier 2 (trainers):
  phase25/: 156K (vs 203K HR)
Total MR size: 1.2M (vs 1.7M HR)
```

---

## Performance Impact

### **Positive Impacts**
- **Memory**: 20-40% reduction in per-process footprint (multi-instance)
- **Import speed**: ~5-10% faster module loading (fewer lines to parse)
- **Deployment size**: Smaller Docker images / edge bundles

### **Negative Impacts**
- **Stack traces**: Less readable (no docstrings/comments in MR)
- **Debugging**: Must cross-reference HR sources when errors occur
- **Maintenance**: Must recompile MR after HR changes

**Mitigation**: Keep HR as authoritative; only use MR in production/optimized scenarios.

---

## Best Practices

### 1. **Always Develop in HR**
Never edit MR files directly. They are **generated artifacts** like compiled binaries.

### 2. **Recompile After HR Updates**
If you change a hot-path module in HR:
```bash
# Update fused_head.py
vim knowledge3d/cranium/fused_head.py

# Recompile MR
make compile-mr-core
```

### 3. **Use Version Control for HR Only**
`.gitignore` already excludes `../Knowledge3D.local/mr/`. MR is **not source code**.

### 4. **Document MR Usage in Deployment Scripts**
If a script requires MR, add a header:
```bash
#!/bin/bash
# REQUIRES: make compile-mr-core (run once before first use)
export PYTHONPATH=../Knowledge3D.local/mr:.
python -m knowledge3d.bridge.live_server
```

---

## Semantic Guarantees

The `codeopt` tool ensures:

1. **Python**: Removes comments/docstrings, adds `pass` for docstring-only bodies, compresses blank lines. **No semantic changes**.
2. **JS/TS**: Removes `//` and `/* */` comments (respects strings/templates), compresses whitespace. **Token order preserved**.
3. **Compilation test**: MR outputs must `import` successfully.

---

## FAQ

### Q: Does MR make code unreadable for humans?
**A**: Yes, intentionally. MR is for machines. Humans read HR sources.

### Q: Can I commit MR to the repository?
**A**: **No**. MR is a build artifact, like `.pyc` files. It's generated on-demand.

### Q: What if MR compilation breaks my code?
**A**: Report a bug. `codeopt` must preserve semantics. If it doesn't, the tool has a defect.

### Q: Does MR improve runtime speed?
**A**: Marginally (~5% import speed). The real win is **memory savings** in multi-instance scenarios.

### Q: Should I use MR for local development?
**A**: **No**, unless testing edge deployment. Stick with HR for better debugging.

---

## Related Documentation

- **[Dual Code Implementation](DUAL_CODE.md)**: Technical details of the `codeopt` CLI
- **[ENV Policy](ENV_POLICY.md)**: Conda environment setup (where to set `PYTHONPATH`)
- **[Deprecations](DEPRECATIONS.md)**: CPU fallbacks, external LLM wrappers (not affected by HR/MR)

---

**Last Updated**: 2025-10-04
**Maintained by**: K3D Core Team (Daniel Campos Ramos, Claude, Codex)
