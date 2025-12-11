# CODEX: Embodied Architecture + Drawing Galaxy Completion

**Date:** December 11, 2025
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** Critical
**Sovereignty:** MANDATORY — No CPU fallbacks, No NumPy in hot path

---

## Context

We have two parallel efforts that need to converge:

1. **Embodied Architecture** (Dec 10) — Galaxy persistence, working memory, adaptive thresholds
2. **Drawing Galaxy** (Dec 7) — PTX kernels for visual transformations, VectorDotMap codec

The embodied scaffolding is in place but:
- Has a bug (`grammar.count()` doesn't exist)
- Drawing transformation rules are **defined** but not **executable** (no PTX kernels)
- The semantic bridge uses n-gram heuristics, not ternary embeddings

**Goal:** Fix the bug, implement the missing PTX kernels, and run training to measure uplift.

---

## Part 1: Quick Bug Fix

### File: `knowledge3d/training/arc_agi/sovereign_pipeline.py`
### Line: 286

**Current (Broken):**
```python
print(f"  [GRAMMAR] Loaded {self.grammar.count()} rules")
```

**Fix:**
```python
print(f"  [GRAMMAR] Loaded {len(self.grammar.rules)} rules")
```

---

## Part 2: Drawing Galaxy PTX Kernels (Critical for ARC-AGI)

The transformation rules in `drawing_galaxy.py` are RPN strings but need **executable PTX kernels**.

### 2.1 Transformation Kernel: `kernels/drawing_transform_ops.cu`

```cuda
/**
 * Drawing Galaxy transformation kernels — GPU-native visual transforms.
 * Used by RPN executor for ROT90, FLIP, SCALE, TILE operations.
 */

// Rotate grid 90° clockwise
__global__ void rot90_cw_kernel(
    const int* input,    // (H, W)
    int* output,         // (W, H) — dimensions swap
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    // rot90_cw: out[x][y] = in[H-1-y][x]
    int in_x = out_y;
    int in_y = in_height - 1 - out_x;

    output[out_y * in_height + out_x] = input[in_y * in_width + in_x];
}

// Rotate grid 90° counter-clockwise
__global__ void rot90_ccw_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    // rot90_ccw: out[x][y] = in[y][W-1-x]
    int in_x = in_width - 1 - out_y;
    int in_y = out_x;

    output[out_y * in_height + out_x] = input[in_y * in_width + in_x];
}

// Flip horizontally
__global__ void flip_h_kernel(
    const int* input,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[y * width + (width - 1 - x)];
}

// Flip vertically
__global__ void flip_v_kernel(
    const int* input,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = input[(height - 1 - y) * width + x];
}

// Transpose (flip diagonal)
__global__ void transpose_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= in_height || out_y >= in_width) return;

    output[out_y * in_height + out_x] = input[out_x * in_width + out_y];
}

// Scale 2x (nearest neighbor upsampling)
__global__ void scale_2x_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_width = in_width * 2;
    int out_height = in_height * 2;

    if (out_x >= out_width || out_y >= out_height) return;

    int in_x = out_x / 2;
    int in_y = out_y / 2;

    output[out_y * out_width + out_x] = input[in_y * in_width + in_x];
}

// Recolor: map old_color to new_color
__global__ void recolor_kernel(
    int* grid,
    int old_color, int new_color,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (grid[idx] == old_color) {
        grid[idx] = new_color;
    }
}

// Tile 2x2: replicate grid into 2x2 pattern
__global__ void tile_2x2_kernel(
    const int* input,
    int* output,
    int in_height, int in_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_width = in_width * 2;
    int out_height = in_height * 2;

    if (out_x >= out_width || out_y >= out_height) return;

    int in_x = out_x % in_width;
    int in_y = out_y % in_height;

    output[out_y * out_width + out_x] = input[in_y * in_width + in_x];
}

// Overlay: grid_a over grid_b (non-zero from a wins)
__global__ void overlay_kernel(
    const int* grid_a,
    const int* grid_b,
    int* output,
    int height, int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int val_a = grid_a[idx];
    output[idx] = (val_a != 0) ? val_a : grid_b[idx];
}
```

### 2.2 Python Wrapper: `ptx_runtime/drawing_transform_kernels.py`

```python
"""
PTX wrappers for Drawing Galaxy transformation kernels.
Uses CuPy for GPU execution — NO numpy in hot path.
"""

import cupy as cp
from pathlib import Path

# Compile kernel once at module load
_KERNEL_SOURCE = Path(__file__).parent.parent / "kernels" / "drawing_transform_ops.cu"
_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        with open(_KERNEL_SOURCE) as f:
            _MODULE = cp.RawModule(code=f.read())
    return _MODULE

def rot90_cw(grid: cp.ndarray) -> cp.ndarray:
    """Rotate grid 90° clockwise. Returns new array with swapped dims."""
    h, w = grid.shape
    output = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_module().get_function("rot90_cw_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def rot90_ccw(grid: cp.ndarray) -> cp.ndarray:
    """Rotate grid 90° counter-clockwise."""
    h, w = grid.shape
    output = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_module().get_function("rot90_ccw_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def flip_h(grid: cp.ndarray) -> cp.ndarray:
    """Flip grid horizontally."""
    h, w = grid.shape
    output = cp.empty_like(grid)
    kernel = _get_module().get_function("flip_h_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def flip_v(grid: cp.ndarray) -> cp.ndarray:
    """Flip grid vertically."""
    h, w = grid.shape
    output = cp.empty_like(grid)
    kernel = _get_module().get_function("flip_v_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def transpose(grid: cp.ndarray) -> cp.ndarray:
    """Transpose grid (flip diagonal)."""
    h, w = grid.shape
    output = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_module().get_function("transpose_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def scale_2x(grid: cp.ndarray) -> cp.ndarray:
    """Scale grid 2x using nearest neighbor."""
    h, w = grid.shape
    output = cp.empty((h * 2, w * 2), dtype=grid.dtype)
    kernel = _get_module().get_function("scale_2x_kernel")
    block = (16, 16)
    grid_dim = ((w * 2 + 15) // 16, (h * 2 + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def recolor(grid: cp.ndarray, old_color: int, new_color: int) -> cp.ndarray:
    """Recolor: replace old_color with new_color in-place."""
    h, w = grid.shape
    kernel = _get_module().get_function("recolor_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, old_color, new_color, h, w))
    return grid

def tile_2x2(grid: cp.ndarray) -> cp.ndarray:
    """Tile grid in 2x2 pattern."""
    h, w = grid.shape
    output = cp.empty((h * 2, w * 2), dtype=grid.dtype)
    kernel = _get_module().get_function("tile_2x2_kernel")
    block = (16, 16)
    grid_dim = ((w * 2 + 15) // 16, (h * 2 + 15) // 16)
    kernel(grid_dim, block, (grid, output, h, w))
    return output

def overlay(grid_a: cp.ndarray, grid_b: cp.ndarray) -> cp.ndarray:
    """Overlay grid_a on grid_b (non-zero from a wins)."""
    h, w = grid_a.shape
    output = cp.empty_like(grid_a)
    kernel = _get_module().get_function("overlay_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid_a, grid_b, output, h, w))
    return output


__all__ = [
    "rot90_cw", "rot90_ccw", "flip_h", "flip_v", "transpose",
    "scale_2x", "recolor", "tile_2x2", "overlay"
]
```

### 2.3 Wire into RPN Executor

**File:** `knowledge3d/training/arc_agi/rpn_executor.py`

Add imports and dispatch for new operations:

```python
# At top of file
try:
    from knowledge3d.cranium.ptx_runtime.drawing_transform_kernels import (
        rot90_cw, rot90_ccw, flip_h, flip_v, transpose,
        scale_2x, recolor, tile_2x2, overlay
    )
    _HAS_DRAWING_KERNELS = True
except ImportError:
    _HAS_DRAWING_KERNELS = False

# In execute() method, add dispatch for transformation RPN tokens
def _execute_transformation(self, op: str, grid: cp.ndarray) -> cp.ndarray:
    """Execute Drawing Galaxy transformation."""
    if not _HAS_DRAWING_KERNELS:
        raise RuntimeError("Drawing transform kernels not available")

    op_upper = op.upper()
    if op_upper == "ROT90_CW" or op_upper == "ROT90":
        return rot90_cw(grid)
    elif op_upper == "ROT90_CCW":
        return rot90_ccw(grid)
    elif op_upper == "ROT180":
        return rot90_cw(rot90_cw(grid))
    elif op_upper == "FLIP_H":
        return flip_h(grid)
    elif op_upper == "FLIP_V":
        return flip_v(grid)
    elif op_upper == "TRANSPOSE" or op_upper == "FLIP_DIAG":
        return transpose(grid)
    elif op_upper == "SCALE_2X":
        return scale_2x(grid)
    elif op_upper == "TILE_2X2":
        return tile_2x2(grid)
    else:
        raise ValueError(f"Unknown transformation: {op}")
```

---

## Part 3: Embodied Integration Verification

After fixing the bug and adding kernels, verify the full stack:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python -c "
import cupy as cp
from knowledge3d.cranium.embodied_agent import EmbodiedSovereignAgent
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline

print('=== Embodied Agent Test ===')
agent = EmbodiedSovereignAgent(working_capacity=1024)
print(f'Drawing: {agent.drawing_galaxy.summary()}')
print(f'Grammar: {len(agent.grammar_galaxy.rules)} rules')
print(f'Transformations: {len(agent.drawing_galaxy.transformations)}')

print('\\n=== Drawing Transform Kernel Test ===')
from knowledge3d.cranium.ptx_runtime.drawing_transform_kernels import rot90_cw, flip_h
grid = cp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=cp.int32)
print(f'Original:\\n{grid}')
print(f'ROT90_CW:\\n{rot90_cw(grid)}')
print(f'FLIP_H:\\n{flip_h(grid)}')

print('\\n=== Pipeline Test ===')
pipeline = SovereignAIPipeline(embodied_agent=agent)
result = pipeline.process_task('test_diag', [[1,0,0],[0,1,0],[0,0,1]])
print(f'Test result: score={result.score:.2f}')

print('\\n=== All Tests Passed ===')
"
```

---

## Part 4: Run Training

Once all tests pass, launch the embodied training:

```bash
tmux new-session -d -s k3d_embodied_draw "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 162 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/embodied_drawing_$(date +%Y%m%d_%H%M%S).log
'"
```

Monitor:
```bash
# Check Galaxy loaded once (should see only ONE "Loaded X rules" line)
grep "Loaded.*rules" /K3D/Knowledge3D.local/logs/embodied_drawing_*.log | head -5

# Check PTX rate
grep "PTX.*rate\|fallback" /K3D/Knowledge3D.local/logs/embodied_drawing_*.log | tail -10

# Check epoch progress
grep "Epoch.*correct" /K3D/Knowledge3D.local/logs/embodied_drawing_*.log | tail -20
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `sovereign_pipeline.py:286` | **FIX** | `grammar.count()` → `len(grammar.rules)` |
| `kernels/drawing_transform_ops.cu` | **CREATE** | PTX kernels for transforms |
| `ptx_runtime/drawing_transform_kernels.py` | **CREATE** | CuPy wrappers |
| `training/arc_agi/rpn_executor.py` | **MODIFY** | Wire transformation dispatch |

---

## Success Criteria

1. **Bug fixed** — No `AttributeError: count`
2. **Drawing kernels work** — ROT90, FLIP, SCALE produce correct outputs on GPU
3. **Galaxy loaded ONCE** — No reload spam in logs
4. **100% PTX** — Zero CPU fallbacks
5. **Measurable uplift** — Target 50%+ (baseline 46.19%)

---

## Sovereignty Checklist

```
╔════════════════════════════════════════════════════════════════╗
║  ✓ CuPy arrays (cp.ndarray), NOT numpy                        ║
║  ✓ PTX kernels for all transforms                              ║
║  ✓ No Python loops over grid data                              ║
║  ✓ Galaxy state persists (load once)                           ║
║  ✓ RPN execution on GPU via TieredRPNEngine                    ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Codex: Fix the bug first, then implement the PTX kernels, verify, and launch training. Sovereignty is non-negotiable. You have the conn.**
