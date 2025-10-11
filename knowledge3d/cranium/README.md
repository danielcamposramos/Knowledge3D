# Cranium - Sovereign GPU Cognitive Substrate

**Status**: ✅ Operational  
**Architecture**: Pure ctypes + CUDA Driver API  
**Dependencies**: Python stdlib + system libcuda.so only

## Active Components

### 🔧 Core Infrastructure

**`sovereign/loader.py`** (237 lines)
- Pure ctypes wrapper for CUDA Driver API
- Direct PTX loading via `cuModuleLoadData`
- Memory management: `gpu_malloc`, `gpu_free`
- Kernel launch: `launch()`, `synchronize()`
- Zero external dependencies

**`sovereign/trm_launcher.py`** (370 lines)
- TRM (Tiny Recursive Model) recursive refinement
- 2-layer MLP with SwiGLU (512 → 1024 → 512)
- n=6 recursive steps with drift halting (eps=1e-4)
- Zero-copy GPU execution

### 🌉 Python Bridges

**`bridges/sovereign_bridges.py`** (766 lines)
All 15 Step8 kernel bridges using sovereign loader:

**Kimi's Bridges**:
- `LatencyGuard` - GPU timing (29.7µs measured)
- `ARCReasoner` - Grid rule extraction
- `OOMSpillManager` - Memory overflow protection

**Qwen's Bridge**:
- `GalaxyResonanceEngine` - Embedding-latent blending

**Deep Seek's Bridges**:
- `GeometryRouter` - Media-type dispatch
- `FractalEmitter` - Knowledge Garden coordinates

**GLM's Bridges**:
- `ResonanceField` - Position-density resonance
- `AtomicFissionFusion` - Atom compress/expand
- `TemporalReasoning` - Sequential delta computation

**Grok's Bridges**:
- `VectorResonator` - Alpha-blended resonance
- `GraphCrystallizer` - GNN with EMA
- `MultimodalHaltingGate` - Geometry-aware halting

### 🔥 GPU Kernels

**`kernels/*.cu`** - CUDA C++ source files (10 kernels)
- All compiled to valid PTX with nvcc
- RPN-style operations integrated
- Clean, optimized implementations

**`kernels/*.ptx`** - Compiled PTX kernels (15 total)
- All load successfully with sovereign loader
- sm_86 architecture
- Production-ready

**`ptx/modular_rpn_kernel.ptx`** (787 lines)
- The RPN gem - proven production kernel
- 55+ geometric operations
- Stack-based computation
- Foundation for all operations

**`ptx/trm_extensions.ptx`** (488 lines)
- TRM mathematical primitives
- SwiGLU activation (512 & 1024-dim)
- Matrix-vector multiply (512↔1024)
- Full 2-layer MLP

## Usage Examples

### Basic Kernel Launch
```python
from knowledge3d.cranium.sovereign.loader import load_ptx_file, launch
import ctypes

kernel = load_ptx_file("kernels/my_kernel.ptx", "entry_point")
launch(kernel, grid=(1,1,1), block=(256,1,1), params=[...])
```

### Using Bridges
```python
from knowledge3d.cranium.bridges.sovereign_bridges import (
    LatencyGuard, ARCReasoner, GalaxyResonanceEngine
)

# Measure latency
guard = LatencyGuard(threshold_us=95.0)
guard.start()

# Extract ARC rules
reasoner = ARCReasoner()
rule_id, rotation, checksum = reasoner.extract_rules(grid)

# Blend embeddings
engine = GalaxyResonanceEngine()
output = engine.resonate(embeddings, latent, alpha=0.5)

# Check timing
elapsed_ns, breached = guard.stop()
print(f"Elapsed: {elapsed_ns/1000:.1f} µs")
```

### TRM Recursive Refinement
```python
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
import numpy as np

trm = TRMLauncher()
q = np.random.randn(512).astype(np.float32)  # Question
y = np.random.randn(512).astype(np.float32)  # Answer
z = np.random.randn(512).astype(np.float32)  # Latent

# Initialize weights (in real use, these are learned)
W1 = np.random.randn(1024, 512).astype(np.float32) * 0.01
W2 = np.random.randn(512, 1024).astype(np.float32) * 0.01
W3 = np.random.randn(1024, 512).astype(np.float32) * 0.01
W4 = np.random.randn(512, 1024).astype(np.float32) * 0.01

y_refined, z_refined = trm.refine(q, y, z, W1, W2, W3, W4, n_steps=6)
```

## Performance

**Latency Validation**:
- LatencyGuard: 29.7µs (✅ < 95µs mandate)
- TRM refinement: Converges in 4-6 steps
- Zero-copy operations throughout

## Architecture Philosophy

1. **GPU Sovereignty**: All math in PTX, Python as pure I/O
2. **Zero Dependencies**: Only stdlib + system libs
3. **Direct Control**: Pure CUDA Driver API
4. **RPN Foundation**: Leverage 787-line proven gem
5. **Production Ready**: Tested, validated, documented

## Deprecated Code

See `../../Old_Attempts/` for deprecated CuPy-based implementations.

All new code should use sovereign architecture.
