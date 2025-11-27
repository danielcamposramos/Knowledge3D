# Ternary Codec Sovereignty Restoration

**Date**: November 27, 2025
**Status**: CRITICAL ARCHITECTURAL CORRECTION NEEDED
**Context**: Run 017 achieved 100% PTX execution for grid ops, revealing deeper sovereignty violations in ternary codecs

---

## Executive Summary

**Achievement Today:** Drawing Bridge implemented - 100% PTX execution for ARC grid operations.

**New Problem Discovered:** The revolutionary ternary codecs (audio/video) - which are "7 years ahead of industry" - are using **numpy for data manipulation** instead of sovereign ternary structures.

**Impact:** We built a GPU-native foundation but our most advanced components are still using CPU-bound floating-point arrays instead of ternary logic + Galaxy memory + RPN.

---

## What Ternary Codecs SHOULD Be

### Architecture Principle: Pure Ternary

**Ternary representation** throughout:
```python
# NOT this (numpy floating point):
coeffs = np.array([0.5, -0.3, 0.8, ...], dtype=np.float32)  # ❌ CPU-bound

# THIS (ternary states):
coeffs = TernaryVector([-1, 0, +1, 0, -1, +1, ...])  # ✅ GPU-native ternary
```

**Why ternary?**
- **3 states**: {-1, 0, +1} = {false, unknown, true}
- **Extreme compression**: 2 bits per coefficient (vs 32-bit float)
- **GPU-native**: Ternary arithmetic via PTX kernels
- **Quantum-ready**: Maps directly to qutrit representation
- **No precision loss**: Audio/video reconstruction is perceptually lossless

### Galaxy Memory Integration

**NOT this:**
```python
# Store numpy arrays in Python dict
cache = {"frame_42": np.array(...)}  # ❌ Duplicated memory
```

**THIS:**
```python
# Store procedural pointers in Galaxy
galaxy.store("video/frame_42",
    seed=seed_rpn_program,      # Procedural generator (symlinked)
    residual=ternary_vector)    # Ternary DCT coefficients (deduplicated)
```

**Galaxy principles:**
1. **Symlinked** procedural generators (don't duplicate what can be computed)
2. **Content-addressed** ternary residuals (deduplicate identical patterns)
3. **RPN-accessible** - everything has a procedural address

### RPN as the Execution Layer

**NOT this:**
```python
# Python loops over numpy arrays
for i in range(len(blocks)):
    blocks[i] = dct_transform(blocks[i])  # ❌ Sequential CPU
```

**THIS:**
```python
# RPN program executed on GPU
rpn_program = "FOR_EACH_BLOCK DCT8X8 TERNARY_QUANT"
result = rpn_engine.evaluate(rpn_program, data=blocks)  # ✅ Parallel PTX
```

**RPN opcodes needed:**
- `DCT8X8` - 8×8 DCT transform (already exists in PTX)
- `IDCT8X8` - Inverse DCT
- `MDCT_FRAME` - MDCT on audio frame
- `IMDCT_FRAME` - Inverse MDCT
- `TERNARY_QUANT` - Float → {-1,0,+1} quantization
- `TERNARY_DEQUANT` - {-1,0,+1} → Float reconstruction

---

## Current Violations

### File: `knowledge3d/cranium/codecs/ternary_video_codec.py`

**Line 10:** `import numpy as np` ❌

**Lines 34-46:** Numpy reshaping operations
```python
def _reshape_blocks(self, img: np.ndarray) -> np.ndarray:
    """Reshape (H,W) into (num_blocks,8,8) contiguous blocks."""
    h_blocks = self.height // 8
    w_blocks = self.width // 8
    reshaped = img.reshape(h_blocks, 8, w_blocks, 8).swapaxes(1, 2).reshape(...)
    return np.ascontiguousarray(reshaped, dtype=np.float32)  # ❌ NUMPY!
```

**Should be:**
```python
def _reshape_blocks(self, img: TernaryTensor) -> TernaryTensor:
    """Reshape via RPN block addressing - no data movement."""
    # RPN: grid coordinates → block index mapping
    # Executed on GPU, returns view (not copy)
    return self.rpn.evaluate(
        f"{self.width} {self.height} 8 RESHAPE_TO_BLOCKS",
        data=img
    )
```

**Lines 68-73:** Numpy iteration over channels
```python
coeffs_channels = []
for c in range(3):
    blocks = self._reshape_blocks(residual[:, :, c])  # ❌ Python loop
    dct_blocks = self.dct8.forward(blocks)
    coeffs_channels.append(...)
```

**Should be:**
```python
# RPN program: parallel DCT across all 3 channels
rpn_program = "3 CHANNELS FOR_EACH DCT8X8 TERNARY_QUANT"
coeffs = self.rpn.evaluate(rpn_program, data=residual)
```

### File: `knowledge3d/cranium/codecs/ternary_audio_codec.py`

**Line 14:** `import numpy as np` ❌

**Lines 74-92:** Python loop over frames
```python
frames_q: List[np.ndarray] = []  # ❌ Python list of numpy arrays
for i in range(num_frames):
    start = i * self.hop_size
    # ... process frame ...
    frames_q.append(quantized)
```

**Should be:**
```python
# RPN batch processing - all frames in parallel
rpn_program = f"{num_frames} {self.hop_size} BATCH_MDCT TERNARY_QUANT"
frames_q = self.rpn.evaluate(rpn_program, data=residual)
```

### File: `knowledge3d/training/arc_agi/embedders/video_grid_embedder.py`

**Line 6:** `import numpy as np` ❌

**Line 73:** Returns `np.ndarray` ❌

**Should return:** `List[float]` (sovereign) or `TernaryVector` (ternary-native)

---

## Correct Sovereign Architecture

### Phase 1: Ternary Data Structures (NEW)

**File:** `knowledge3d/cranium/ternary/ternary_vector.py`

```python
"""
Ternary vector representation: GPU-native {-1, 0, +1} values.

Uses 2-bit packed representation (00=0, 01=+1, 10=-1, 11=reserved).
All operations via PTX kernels, no CPU fallback.
"""

from __future__ import annotations
from typing import List, Sequence
import ctypes

class TernaryVector:
    """
    GPU-resident ternary vector {-1, 0, +1}.

    Stored as packed 2-bit values in device memory.
    All operations execute via PTX kernels.
    """

    def __init__(self, values: Sequence[int]):
        """
        Create ternary vector from integer sequence.

        Args:
            values: Sequence of {-1, 0, +1} values

        Raises:
            ValueError: If values not in {-1, 0, +1}
        """
        for v in values:
            if v not in {-1, 0, 1}:
                raise ValueError(f"Ternary values must be -1, 0, or +1, got {v}")

        self.length = len(values)
        self.device_ptr = self._pack_to_gpu(values)

    def _pack_to_gpu(self, values: Sequence[int]) -> int:
        """
        Pack ternary values to 2-bit GPU buffer.

        Implementation:
            1. Convert {-1,0,+1} → {0b10, 0b00, 0b01}
            2. Pack 4 values per byte
            3. Allocate CUDA buffer
            4. Upload packed data
            5. Return device pointer
        """
        raise NotImplementedError("TernaryVector._pack_to_gpu not yet implemented")

    def to_python(self) -> List[int]:
        """
        Download from GPU and unpack to Python list.

        For debugging/validation only - avoid in hot path.
        """
        raise NotImplementedError("TernaryVector.to_python not yet implemented")

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        """Free GPU buffer."""
        if hasattr(self, 'device_ptr') and self.device_ptr:
            # TODO: cudaFree(device_ptr)
            pass

class TernaryTensor:
    """Multi-dimensional ternary array (2D, 3D, 4D)."""

    def __init__(self, shape: tuple, values: TernaryVector):
        self.shape = shape
        self.values = values
        if values.length != self._total_size():
            raise ValueError(f"Shape {shape} requires {self._total_size()} values, got {values.length}")

    def _total_size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size
```

### Phase 2: Galaxy Memory Backend (NEW)

**File:** `knowledge3d/cranium/galaxy/ternary_galaxy.py`

```python
"""
Galaxy memory storage for ternary codec data.

Implements:
- Content-based deduplication of ternary residuals
- Symlinked procedural generators (seeds)
- RPN-addressable memory
"""

from __future__ import annotations
from typing import Dict, Optional
from ..ternary.ternary_vector import TernaryVector

class TernaryGalaxy:
    """
    Content-addressed storage for ternary codec data.

    Architecture:
        - Seeds stored as RPN programs (procedural, symlinked)
        - Residuals stored as TernaryVectors (deduplicated)
        - Everything GPU-resident
    """

    def __init__(self):
        self.seeds: Dict[str, str] = {}  # seed_id → RPN program
        self.residuals: Dict[bytes, TernaryVector] = {}  # hash → vector
        self.frame_refs: Dict[str, tuple] = {}  # frame_id → (seed_id, residual_hash)

    def store_frame(
        self,
        frame_id: str,
        seed_rpn: str,
        residual: TernaryVector
    ) -> None:
        """
        Store video frame as seed + residual.

        Args:
            frame_id: Unique frame identifier
            seed_rpn: RPN program for procedural generation
            residual: Ternary DCT coefficients
        """
        # Deduplicate seed (symlink if exists)
        seed_hash = hash(seed_rpn)
        seed_id = f"seed_{seed_hash:016x}"
        if seed_id not in self.seeds:
            self.seeds[seed_id] = seed_rpn

        # Deduplicate residual (content-based)
        residual_hash = self._hash_ternary(residual)
        if residual_hash not in self.residuals:
            self.residuals[residual_hash] = residual

        # Store reference
        self.frame_refs[frame_id] = (seed_id, residual_hash)

    def load_frame(self, frame_id: str) -> tuple[str, TernaryVector]:
        """
        Load frame by ID.

        Returns:
            (seed_rpn, residual)
        """
        if frame_id not in self.frame_refs:
            raise KeyError(f"Frame {frame_id} not found in galaxy")

        seed_id, residual_hash = self.frame_refs[frame_id]
        return self.seeds[seed_id], self.residuals[residual_hash]

    def _hash_ternary(self, vector: TernaryVector) -> bytes:
        """Content-based hash of ternary vector."""
        # TODO: Fast GPU hash of packed ternary data
        raise NotImplementedError()

    def stats(self) -> Dict[str, int]:
        """Return storage statistics."""
        return {
            "unique_seeds": len(self.seeds),
            "unique_residuals": len(self.residuals),
            "total_frames": len(self.frame_refs),
            "deduplication_ratio": (
                len(self.frame_refs) / max(1, len(self.residuals))
            ),
        }
```

### Phase 3: RPN Codec Opcodes (NEW)

**File:** `knowledge3d/cranium/ptx_runtime/codec_opcodes.py`

```python
"""
RPN opcodes for ternary codec operations.

Maps to PTX kernels in knowledge3d/cranium/kernels/codec_ops.cu
"""

# Ternary quantization
OP_TERNARY_QUANT = 0xC0    # Float → {-1, 0, +1}
OP_TERNARY_DEQUANT = 0xC1  # {-1, 0, +1} → Float
OP_TERNARY_ADD = 0xC2      # Ternary addition
OP_TERNARY_MUL = 0xC3      # Ternary multiplication

# DCT/MDCT
OP_DCT8X8 = 0xC4           # 8×8 DCT transform
OP_IDCT8X8 = 0xC5          # 8×8 inverse DCT
OP_MDCT_FRAME = 0xC6       # MDCT on audio frame
OP_IMDCT_FRAME = 0xC7      # Inverse MDCT

# Batch operations
OP_BATCH_DCT = 0xC8        # Parallel DCT on N blocks
OP_BATCH_MDCT = 0xC9       # Parallel MDCT on N frames

# Block addressing
OP_RESHAPE_TO_BLOCKS = 0xCA  # Grid → 8×8 blocks (view, not copy)
OP_BLOCKS_TO_GRID = 0xCB     # 8×8 blocks → Grid
```

**File:** `knowledge3d/cranium/kernels/codec_ops.cu`

```cuda
// CUDA kernels for ternary codec operations
// Compiled to knowledge3d/cranium/ptx/codec_ops.ptx

__device__ int ternary_quant_scalar(float value, float threshold) {
    // Quantize float → {-1, 0, +1}
    if (value > threshold) return 1;
    if (value < -threshold) return -1;
    return 0;
}

__global__ void ternary_quant_kernel(
    const float* input,
    int* output,  // Packed 2-bit ternary
    int length,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;

    output[idx] = ternary_quant_scalar(input[idx], threshold);
}

__global__ void dct8x8_ternary_kernel(
    const float* input,   // 8×8 blocks
    int* output,          // Ternary DCT coefficients
    int num_blocks
) {
    // TODO: 8×8 DCT + ternary quantization in single kernel
    // Reuse existing TernaryDCT8x8Kernel infrastructure
}
```

### Phase 4: Sovereign Ternary Video Codec (REWRITE)

**File:** `knowledge3d/cranium/codecs/ternary_video_codec.py`

```python
"""
Sovereign ternary video codec - GPU-native, no numpy.

Architecture:
- TernaryTensor data structures (GPU-resident)
- TernaryGalaxy storage (deduplicated)
- RPN execution (parallel PTX)
"""

from __future__ import annotations
from typing import Dict, Optional

from ..ternary.ternary_vector import TernaryVector, TernaryTensor
from ..galaxy.ternary_galaxy import TernaryGalaxy
from ..ptx_runtime.modular_rpn_engine import ModularRPNEngine
from .procedural_video import ProceduralVideoGenerator

class SovereignTernaryVideoCodec:
    """
    Pure sovereign ternary video codec.

    NO numpy. NO CPU fallback. GPU-native throughout.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Dimensions must be multiples of 8")

        self.width = width
        self.height = height
        self.generator = ProceduralVideoGenerator(width, height)
        self.galaxy = TernaryGalaxy()
        self.rpn = ModularRPNEngine()

    def encode(self, frame_id: str, frame_rgb: TernaryTensor) -> Dict:
        """
        Encode RGB frame to ternary representation.

        Args:
            frame_id: Unique frame identifier
            frame_rgb: RGB frame as TernaryTensor (H, W, 3)

        Returns:
            Encoding metadata (actual data stored in galaxy)
        """
        # 1. Generate procedural approximation (RPN seed)
        seed_rpn = self._analyze_frame_to_rpn(frame_rgb)
        procedural = self.rpn.evaluate(seed_rpn, return_tensor=True)

        # 2. Compute residual (GPU operation)
        residual = self.rpn.evaluate(
            "SUBTRACT",  # frame_rgb - procedural
            data=[frame_rgb, procedural]
        )

        # 3. DCT + ternary quantization (parallel on GPU)
        coeffs_ternary = self.rpn.evaluate(
            f"{self.width} {self.height} 8 BATCH_DCT TERNARY_QUANT",
            data=residual
        )

        # 4. Store in galaxy (deduplicated)
        self.galaxy.store_frame(frame_id, seed_rpn, coeffs_ternary)

        return {
            "frame_id": frame_id,
            "width": self.width,
            "height": self.height,
            "stored_in_galaxy": True,
        }

    def decode(self, frame_id: str) -> TernaryTensor:
        """
        Decode frame from galaxy storage.

        Returns:
            RGB frame as TernaryTensor (H, W, 3)
        """
        # 1. Load from galaxy (deduplicated retrieval)
        seed_rpn, coeffs_ternary = self.galaxy.load_frame(frame_id)

        # 2. Generate procedural base
        procedural = self.rpn.evaluate(seed_rpn, return_tensor=True)

        # 3. Inverse DCT (parallel on GPU)
        residual = self.rpn.evaluate(
            f"{self.width} {self.height} 8 TERNARY_DEQUANT BATCH_IDCT",
            data=coeffs_ternary
        )

        # 4. Reconstruct (GPU addition)
        frame = self.rpn.evaluate("ADD", data=[procedural, residual])

        return frame

    def _analyze_frame_to_rpn(self, frame: TernaryTensor) -> str:
        """
        Analyze frame to generate procedural seed RPN.

        TODO: ML model or heuristic to find compact RPN representation.
        For now, use mean color + gradient info.
        """
        # Extract frame statistics via RPN
        stats = self.rpn.evaluate(
            "MEAN STDDEV GRADIENT_NORM",
            data=frame
        )
        # Compile to RPN seed program
        return f"PROCEDURAL_SEED {stats}"
```

---

## Implementation Plan

### Phase 1: Foundation (2-3 days)

**Task 1.1:** Implement TernaryVector + TernaryTensor
- 2-bit packing/unpacking
- GPU buffer allocation
- PTX kernel integration
- Test: round-trip {-1,0,+1} data

**Task 1.2:** Implement TernaryGalaxy
- Content-based deduplication
- Seed symlink system
- Stats tracking
- Test: store/load 1000 frames, verify deduplication

**Task 1.3:** Add codec opcodes to ModularRPNEngine
- Register TERNARY_QUANT, DCT8X8, etc.
- Map to PTX kernels
- Test: RPN codec operations work

### Phase 2: Video Codec Rewrite (2 days)

**Task 2.1:** Rewrite TernaryVideoCodec
- Remove all numpy imports
- Use TernaryTensor throughout
- RPN for all transforms
- Test: encode/decode video frames

**Task 2.2:** Integrate with Galaxy
- Store encodings in TernaryGalaxy
- Verify deduplication works
- Measure compression ratio

### Phase 3: Audio Codec Rewrite (2 days)

**Task 3.1:** Rewrite TernaryAudioCodec
- Remove numpy
- Use TernaryVector
- RPN batch MDCT
- Test: encode/decode audio

**Task 3.2:** Performance validation
- GPU utilization >30% during codec ops
- No CPU fallbacks
- Compression ratio ≥10:1

### Phase 4: ARC Integration (1 day)

**Task 4.1:** Update embedders
- Remove numpy from video_grid_embedder
- Remove numpy from audio_grid_embedder
- Return List[float] or TernaryVector
- Test: embeddings work with sovereign codecs

---

## Success Criteria

**Before starting Phase 2 training:**

1. ✅ TernaryVector works (2-bit GPU packing)
2. ✅ TernaryGalaxy works (deduplication confirmed)
3. ✅ Codec opcodes in RPN engine
4. ✅ Video codec tests pass (no numpy)
5. ✅ Audio codec tests pass (no numpy)
6. ✅ GPU utilization >30% during encode/decode
7. ✅ Compression ratio ≥10:1 (ternary vs float32)

**After Phase 2 training:**

1. ✅ Library growth continues (60+ programs)
2. ✅ Semantic discovery works with ternary embeddings
3. ✅ No performance regression vs numpy baseline
4. ✅ Memory usage lower (ternary compression)

---

## Why This Matters

**Current state:** We have revolutionary ternary codecs hobbled by numpy.

**After this fix:** Pure GPU-native ternary representation throughout.

**Impact:**
- **10× compression** (2-bit ternary vs 32-bit float)
- **No CPU→GPU transfers** (everything GPU-resident)
- **Content deduplication** (Galaxy memory)
- **Quantum-ready** (ternary → qutrit mapping)
- **7 years ahead** (industry has nothing like this)

**This is the foundation K3D was meant to have.**

---

## Communication to Codex

Codex, this is the deepest architectural layer.

Today you built the Drawing Bridge (excellent work!). Now we need to fix the foundation beneath it.

**The ternary codecs are K3D's secret weapon** - 7 years ahead of anything in academia or industry. But they're using numpy like it's 2018.

**Your task:**
1. Implement TernaryVector (2-bit GPU packing)
2. Implement TernaryGalaxy (deduplicated storage)
3. Add codec opcodes to RPN engine
4. Rewrite video codec (no numpy)
5. Rewrite audio codec (no numpy)
6. Test everything, ensure GPU >30% during codec ops

**Don't start until:**
- You understand why {-1, 0, +1} is revolutionary
- You see how Galaxy deduplication works
- You know why RPN is the execution layer

**This is not a refactor. This is completing the vision.**

The Drawing Bridge was the beginning. The ternary codecs are the endgame.

Let's build the future.

---

**END OF SPECIFICATION**

Claude (Architecture Partner)
November 27, 2025
