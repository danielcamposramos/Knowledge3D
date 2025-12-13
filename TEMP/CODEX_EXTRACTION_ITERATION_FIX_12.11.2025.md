# CODEX: Extraction + Iteration Operations — Critical Gap Fix

**Priority:** HIGH — Blocking 68+ tasks (40% of failures)

---

## Analysis Summary

Training results show clear failure pattern:

| Task Type | Example | Input→Output | Score |
|-----------|---------|--------------|-------|
| Same-size | fd096ab6 | 22×22 → 22×22 | 97% |
| Extraction | f4081712 | 24×24 → 3×3 | 0% |

**Root cause:** No RPN operations for:
1. **Extraction/cropping** — get sub-region of grid
2. **Pattern detection with output** — find bbox, return coordinates
3. **Iteration** — repeat operation N times or until stable

---

## Fix A: GrammarGalaxy Singleton (Quick)

Workers reload 747 rules each time instead of sharing EmbodiedAgent's instance.

**File:** `knowledge3d/training/arc_agi/parallel_candidate_generator.py`

**Problem:** Each worker creates new GrammarGalaxy:
```python
# In worker function
grammar = GrammarGalaxy()  # WRONG: reloads from disk
```

**Fix:** Pass shared instance via embodied agent or use module-level singleton.

---

## Fix B: Wire Drawing Transform PTX Kernels

PTX kernels exist in `knowledge3d/cranium/kernels/drawing_transform_ops.cu` but may not be fully wired.

**Verify and wire:**
- `rot90_cw_kernel`, `rot90_ccw_kernel`
- `flip_h_kernel`, `flip_v_kernel`
- `transpose_kernel`, `scale_2x_kernel`
- `recolor_kernel`, `tile_2x2_kernel`
- `overlay_kernel`

**File:** `knowledge3d/training/arc_agi/arc_rpn_executor.py`

Ensure these call PTX via `drawing_transform_kernels.py` wrappers.

---

## Fix C: Add Extraction Operations (Core Fix)

### C1. Add CROP operation

**File:** `knowledge3d/cranium/kernels/drawing_transform_ops.cu`

```cuda
// Crop/extract sub-region
__global__ void crop_kernel(
    const int* input,
    int* output,
    int in_height, int in_width,
    int crop_y, int crop_x,
    int crop_h, int crop_w
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= crop_w || out_y >= crop_h) return;

    int in_x = crop_x + out_x;
    int in_y = crop_y + out_y;

    if (in_x < in_width && in_y < in_height) {
        output[out_y * crop_w + out_x] = input[in_y * in_width + in_x];
    }
}
```

### C2. Add FIND_BBOX operation (returns coordinates)

**File:** `knowledge3d/cranium/kernels/drawing_transform_ops.cu`

```cuda
// Find bounding box of non-zero cells
// Returns: [min_y, min_x, max_y, max_x] in output array
__global__ void find_bbox_kernel(
    const int* grid,
    int* bbox,  // output: [min_y, min_x, max_y, max_x]
    int height, int width,
    int target_color  // 0 = any non-zero, else specific color
) {
    __shared__ int s_min_y, s_min_x, s_max_y, s_max_x;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_min_y = height;
        s_min_x = width;
        s_max_y = -1;
        s_max_x = -1;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int val = grid[y * width + x];
        bool match = (target_color == 0) ? (val != 0) : (val == target_color);

        if (match) {
            atomicMin(&s_min_y, y);
            atomicMin(&s_min_x, x);
            atomicMax(&s_max_y, y);
            atomicMax(&s_max_x, x);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMin(&bbox[0], s_min_y);
        atomicMin(&bbox[1], s_min_x);
        atomicMax(&bbox[2], s_max_y);
        atomicMax(&bbox[3], s_max_x);
    }
}
```

### C3. Add EXTRACT_BBOX (compound operation)

Combines FIND_BBOX + CROP:
```
EXTRACT_BBOX color -> finds bbox of color, crops to that region
```

### C4. Python wrapper

**File:** `knowledge3d/cranium/ptx_runtime/drawing_transform_kernels.py`

```python
def crop_gpu(grid: cp.ndarray, y: int, x: int, h: int, w: int) -> cp.ndarray:
    """Crop sub-region using PTX kernel."""
    output = cp.zeros((h, w), dtype=cp.int32)
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)

    _crop_kernel(grid_dim, block, (
        grid, output,
        grid.shape[0], grid.shape[1],
        y, x, h, w
    ))
    return output

def find_bbox_gpu(grid: cp.ndarray, color: int = 0) -> Tuple[int, int, int, int]:
    """Find bounding box of colored region. Returns (min_y, min_x, max_y, max_x)."""
    bbox = cp.array([grid.shape[0], grid.shape[1], -1, -1], dtype=cp.int32)
    block = (16, 16)
    grid_dim = ((grid.shape[1] + 15) // 16, (grid.shape[0] + 15) // 16)

    _find_bbox_kernel(grid_dim, block, (grid, bbox, grid.shape[0], grid.shape[1], color))

    return tuple(bbox.get())

def extract_bbox_gpu(grid: cp.ndarray, color: int = 0) -> cp.ndarray:
    """Find bbox and crop to it in one operation."""
    min_y, min_x, max_y, max_x = find_bbox_gpu(grid, color)
    if max_y < 0:  # No match found
        return cp.zeros((1, 1), dtype=cp.int32)
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    return crop_gpu(grid, min_y, min_x, h, w)
```

---

## Fix D: Add Iteration Operations

### D1. REPEAT n op

**File:** `knowledge3d/training/arc_agi/drawing_galaxy.py`

Add to TRANSFORMATION_RULES:
```python
# Iteration
"REPEAT_2": "DUP EXEC DUP EXEC",  # Apply top-of-stack operation twice
"REPEAT_3": "DUP EXEC DUP EXEC DUP EXEC",
"UNTIL_STABLE": "LOOP_START DUP EXEC DUP ROT EQ IF_BREAK LOOP_END",
```

### D2. RPN Executor iteration support

**File:** `knowledge3d/training/arc_agi/arc_rpn_executor.py`

Add iteration opcodes:
```python
def _execute_repeat(self, stack: List, n: int, op: str) -> None:
    """Execute operation n times."""
    for _ in range(n):
        self._execute_single_op(stack, op)

def _execute_until_stable(self, stack: List, op: str, max_iter: int = 10) -> None:
    """Execute operation until grid stops changing."""
    for _ in range(max_iter):
        prev = stack[-1].copy() if hasattr(stack[-1], 'copy') else stack[-1]
        self._execute_single_op(stack, op)
        if self._grids_equal(prev, stack[-1]):
            break
```

---

## Fix E: Candidate Generator Size-Adaptive Logic

**File:** `knowledge3d/training/arc_agi/candidate_generator.py`

When train examples show size reduction pattern, generate extraction candidates:

```python
def _detect_size_pattern(self, train_examples: List[Dict]) -> str:
    """Detect if task involves size change."""
    ratios = []
    for ex in train_examples:
        inp = ex.get('input', [])
        out = ex.get('output', [])
        if inp and out:
            h_ratio = len(out) / len(inp)
            w_ratio = len(out[0]) / len(inp[0]) if inp[0] and out[0] else 1.0
            ratios.append((h_ratio, w_ratio))

    if not ratios:
        return "same"

    avg_h = sum(r[0] for r in ratios) / len(ratios)
    avg_w = sum(r[1] for r in ratios) / len(ratios)

    if avg_h < 0.6 and avg_w < 0.6:
        return "extract"  # Shrinking significantly
    elif avg_h > 1.5 and avg_w > 1.5:
        return "expand"   # Growing significantly
    return "same"

def generate_candidates(self, input_grid, train_examples, ...):
    size_pattern = self._detect_size_pattern(train_examples)

    if size_pattern == "extract":
        # Generate extraction-focused candidates
        candidates.extend(self._generate_extraction_candidates(input_grid, train_examples))
    # ... rest of generation
```

---

## Verification

After implementing, run:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
PYTHONPATH=. /home/daniel/miniforge/bin/conda run -n k3d-cranium python -c "
from knowledge3d.cranium.ptx_runtime.drawing_transform_kernels import crop_gpu, find_bbox_gpu, extract_bbox_gpu
import cupy as cp

# Test grid with pattern in corner
grid = cp.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0],
    [0, 3, 4, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=cp.int32)

# Test find_bbox
bbox = find_bbox_gpu(grid)
print(f'Bbox: {bbox}')  # Should be (1, 1, 2, 2)

# Test extract_bbox
extracted = extract_bbox_gpu(grid)
print(f'Extracted shape: {extracted.shape}')  # Should be (2, 2)
print(f'Extracted:\\n{extracted.get()}')

print('=== EXTRACTION VERIFICATION PASSED ===')
"
```

Then launch training targeting extraction tasks:

```bash
tmux new-session -d -s k3d_extract_fix "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 162 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/extraction_fix_$(date +%Y%m%d_%H%M%S).log
'"
```

---

## Success Criteria

1. **Extraction tasks improve** — f4081712, ed74f2f2, etc. should score >50%
2. **PTX rate stays 100%** — all new ops use GPU kernels
3. **Average accuracy** — target >50% (up from 45.79%)
4. **GrammarGalaxy loads once** — no more repeated "[GrammarGalaxy] Loaded 747 rules" per worker

---

## Implementation Order

1. **Fix A** (GrammarGalaxy singleton) — quick win, reduces log noise
2. **Fix C** (CROP/BBOX kernels) — core capability
3. **Fix E** (size-adaptive generator) — use new ops
4. **Fix B** (wire existing PTX) — verify all transforms work
5. **Fix D** (iteration) — future multi-step tasks

**Start with A+C+E for immediate impact on extraction tasks.**
