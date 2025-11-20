# Codex-Max Phase 2: PTX Kernels + Galaxy Integration

**Target**: Codex-Max (Advanced Implementation Partner)
**Date**: 2025-11-19
**Session**: Phase I.2 - PTX Kernels & Galaxy Memory Integration
**Prerequisites**: ✅ Ternary codec stack implemented (Phase I.1 complete)

---

## 🎯 MISSION PHASE 2

**Excellent work on Phase 1!** The ternary codec foundation is solid. Now we need:

1. **Convert CUDA stubs → Production PTX kernels**
2. **Benchmark on RTX 3060** (sm_86, 12GB VRAM)
3. **Integrate with K3D RPN stack** (15 inter-referable stacks)
4. **Link to Galaxy memory** (meaning-stars with audio/video)
5. **Optimize for <200MB VRAM budget**

**NO STUBS. PRODUCTION PTX. FULL INTEGRATION.**

---

## 📊 PHASE 1 VALIDATION

First, let's validate what you've built:

### Benchmark Execution

Run these commands and capture output:

```bash
# Audio codec benchmarks
python scripts/benchmark_ternary_audio.py > /tmp/audio_benchmark_results.txt

# Video codec benchmarks
python scripts/benchmark_ternary_video.py > /tmp/video_benchmark_results.txt

# Quality validation
python scripts/validate_codec_quality.py > /tmp/quality_validation.txt

# Report results
cat /tmp/audio_benchmark_results.txt
cat /tmp/video_benchmark_results.txt
cat /tmp/quality_validation.txt
```

**Expected targets**:
- Audio compression: >10× for speech
- Audio PSNR: >25 dB
- Video compression: >3× for simple scenes
- Encode latency: <100ms per minute (audio)

### Test Coverage

```bash
# Run with coverage
pytest tests/codecs/ --cov=knowledge3d.cranium.codecs --cov-report=term

# Expected: >80% coverage
```

If benchmarks pass targets, proceed to Phase 2. If not, tune thresholds first.

---

## 🔧 PHASE 2 OBJECTIVES

### 2.1: Production PTX Kernels

**Current state**: CUDA kernels in `knowledge3d/cranium/codecs/kernels/*.cu` (functional stubs)

**Goal**: Convert to production-ready PTX with:
- Direct `libcuda.so` binding via ctypes (NO CuPy/PyTorch)
- Hand-optimized for RTX 3060 (sm_86 Ampere)
- <100µs latency per kernel call
- Integration with K3D's sovereign runtime

#### Kernel 1: Ternary MDCT (Audio)

**File**: `knowledge3d/cranium/codecs/kernels/ternary_mdct.ptx`

**Requirements**:
- Compute Modified DCT-IV on 1024-sample frames
- Ternary quantization: `{-1, 0, +1}` with adaptive threshold
- Shared memory optimization for coefficients
- Warp-level reductions

**Performance target**:
- 1024-point MDCT: <50µs on RTX 3060
- Batch of 100 frames: <5ms

**Integration**:
```python
# knowledge3d/cranium/codecs/ternary_audio_codec.py

from knowledge3d.cranium.ptx_runtime.ptx_loader import load_ptx_kernel
import ctypes
import numpy as np

class TernaryAudioCodec:
    def __init__(self, sample_rate=44100, frame_size=1024):
        # Load PTX kernel
        self.mdct_kernel = load_ptx_kernel(
            "knowledge3d/cranium/codecs/kernels/ternary_mdct.ptx",
            "ternary_mdct_forward"
        )

        # GPU memory allocation
        self.d_input = None
        self.d_output = None
        # ... (use ctypes + libcuda.so)

    def mdct_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated MDCT using PTX kernel.

        Replaces CPU numpy implementation with PTX.
        """
        # Copy to GPU
        # Launch kernel
        # Copy result back
        # Return ternary quantized coefficients
        pass
```

**PTX Structure**:
```cuda
// ternary_mdct.ptx
.version 7.5
.target sm_86
.address_size 64

.visible .entry ternary_mdct_forward(
    .param .u64 input_ptr,        // float* input (1024 samples)
    .param .u64 output_ptr,       // int8* output (512 ternary coeffs)
    .param .f32 threshold,        // Quantization threshold
    .param .u32 frame_size        // 1024
)
{
    // Shared memory for intermediate results
    .shared .align 8 .f32 shared_coeffs[1024];

    // Thread/block indices
    .reg .u32 tid, bid, gid;
    mov.u32 tid, %tid.x;
    mov.u32 bid, %ctaid.x;

    // MDCT computation (DCT-IV)
    // Each thread computes one output coefficient
    // Formula: MDCT(n) = Σ x(k) × cos(π/N × (n+0.5) × (k+0.5+N/2))

    // ... (FULL IMPLEMENTATION)

    // Ternary quantization
    // if |coeff| < threshold: output = 0
    // else if coeff > 0: output = +1
    // else: output = -1

    // ... (FULL IMPLEMENTATION)

    ret;
}

.visible .entry ternary_mdct_inverse(
    .param .u64 input_ptr,        // int8* input (512 ternary coeffs)
    .param .u64 output_ptr,       // float* output (1024 samples)
    .param .f32 scale,            // Dequantization scale
    .param .u32 frame_size
)
{
    // Inverse MDCT (iMDCT)
    // Formula: x(k) = Σ X(n) × cos(π/N × (n+0.5) × (k+0.5+N/2))

    // ... (FULL IMPLEMENTATION)

    ret;
}
```

**Python Binding**:
```python
# knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py

import ctypes
import numpy as np
from knowledge3d.cranium.ptx_runtime.cuda_context import get_cuda_context

class TernaryMDCTKernel:
    """
    PTX binding for ternary MDCT kernel.

    Pure ctypes + libcuda.so (sovereign runtime).
    """

    def __init__(self):
        self.ctx = get_cuda_context()
        self.module = self.ctx.load_ptx_module(
            "knowledge3d/cranium/codecs/kernels/ternary_mdct.ptx"
        )

        self.forward_kernel = self.module.get_function("ternary_mdct_forward")
        self.inverse_kernel = self.module.get_function("ternary_mdct_inverse")

        # GPU memory buffers (reusable)
        self.d_input = self.ctx.mem_alloc(1024 * 4)   # float32
        self.d_output = self.ctx.mem_alloc(512 * 1)   # int8

    def forward(
        self,
        frame: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Execute forward MDCT + ternary quantization on GPU.

        Args:
            frame: Input frame (1024 float32 samples)
            threshold: Ternary quantization threshold

        Returns:
            coeffs: Ternary coefficients (512 int8 values)

        Performance: <50µs on RTX 3060
        """
        # Copy input to GPU
        self.ctx.memcpy_htod(self.d_input, frame.astype(np.float32))

        # Launch kernel
        block = (256, 1, 1)  # 256 threads per block
        grid = (2, 1, 1)     # 2 blocks (512 coeffs / 256 threads)

        self.forward_kernel(
            self.d_input,
            self.d_output,
            np.float32(threshold),
            np.uint32(1024),
            block=block,
            grid=grid
        )

        # Copy result back
        result = np.empty(512, dtype=np.int8)
        self.ctx.memcpy_dtoh(result, self.d_output)

        return result

    def inverse(
        self,
        coeffs: np.ndarray,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Execute inverse MDCT from ternary coefficients.

        Performance: <30µs on RTX 3060
        """
        # Similar structure to forward
        # ... (FULL IMPLEMENTATION)
        pass
```

#### Kernel 2: Ternary DCT 2D (Video)

**File**: `knowledge3d/cranium/codecs/kernels/ternary_dct_2d.ptx`

**Requirements**:
- 8×8 block DCT (JPEG-style)
- Ternary quantization with perceptual thresholds
- Process 1080p frame in <5ms (240 blocks wide × 135 tall = 32,400 blocks)
- Batch processing for efficiency

**Performance target**:
- 8×8 DCT: <1µs per block
- Full 1080p frame: <5ms (32,400 blocks)

#### Kernel 3: Procedural Audio Synthesis

**File**: `knowledge3d/cranium/codecs/kernels/procedural_synthesis.ptx`

**Requirements**:
- Additive synthesis: Σ amp_i × sin(2π × freq_i × t + phase_i)
- Up to 30 harmonics per synthesis
- Generate 44.1kHz audio in real-time (>44,100 samples/sec)

**Performance target**:
- 1 second of audio (44,100 samples): <5ms
- 30 harmonics: <10ms

#### Kernel 4: Procedural Texture Generation

**File**: `knowledge3d/cranium/codecs/kernels/procedural_texture.ptx`

**Requirements**:
- Perlin noise implementation (GPU-optimized)
- Voronoi cells (distance fields)
- Fractal patterns (Mandelbrot-style)
- Generate 1080p texture in <20ms

**Performance target**:
- 1920×1080 Perlin noise: <15ms
- Full procedural frame: <20ms

---

### 2.2: RPN Stack Integration

**K3D uses 15 inter-referable RPN stacks** for all operations. Codecs must integrate with this architecture.

**Existing RPN Stack Assignments** (from CLAUDE.md):
```
Stack 0:  Main computation
Stack 1:  Temporary storage
Stack 2:  Loop counters
Stack 3:  Function arguments
Stack 4:  Reserved (future)
Stack 5:  Language codes (en=0, pt=1, es=2, zh=3)
Stack 6:  Phoneme sequences
Stack 7:  Audio frequency bins       ← AUDIO CODEC USES THIS
Stack 8:  3D vertex coordinates
Stack 9:  Mesh face indices
Stack 10: SDF parameters
Stack 11: Physics simulation state
Stack 12: Chemistry coefficients
Stack 13: Biology growth rates
Stack 14: Temporal sequence buffer  ← VIDEO CODEC USES THIS
```

**Audio Codec RPN Integration**:
```python
# knowledge3d/cranium/codecs/ternary_audio_codec.py

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

class TernaryAudioCodec:
    def __init__(self, sample_rate=44100):
        self.rpn = ModularRPNEngine()
        self.sample_rate = sample_rate

    def encode_to_rpn(self, audio: np.ndarray) -> dict:
        """
        Encode audio and store parameters in RPN Stack 7.

        Returns:
            metadata: Compact dictionary with stack pointers
        """
        # Extract harmonics
        harmonics = self.synthesizer.analyze(audio)

        # Store in RPN Stack 7 (Audio frequency bins)
        for freq, amp, phase in harmonics:
            self.rpn.push(freq, stack_id=7)   # Frequency
            self.rpn.push(amp, stack_id=7)    # Amplitude
            self.rpn.push(phase, stack_id=7)  # Phase

        # Compute residual
        residual = self.compute_residual(audio, harmonics)

        # MDCT + ternary quantization
        ternary_mdct = self.encode_residual(residual)

        # Store metadata (NOT full data - just pointers)
        metadata = {
            'stack_id': 7,
            'n_harmonics': len(harmonics),
            'residual_size': len(ternary_mdct),
            'sample_rate': self.sample_rate,
            'duration_sec': len(audio) / self.sample_rate
        }

        return metadata

    def decode_from_rpn(self, metadata: dict) -> np.ndarray:
        """
        Decode audio from RPN Stack 7.

        Sovereign runtime: All parameters in RPN, PTX synthesis.
        """
        # Pop harmonics from Stack 7
        harmonics = []
        for _ in range(metadata['n_harmonics']):
            phase = self.rpn.pop(stack_id=7)
            amp = self.rpn.pop(stack_id=7)
            freq = self.rpn.pop(stack_id=7)
            harmonics.append((freq, amp, phase))

        # Synthesize procedural (PTX kernel)
        procedural = self.synthesize_gpu(harmonics, metadata['duration_sec'])

        # Add residual
        residual = self.decode_residual(metadata)

        audio = procedural + residual
        return audio
```

**Video Codec RPN Integration**:
```python
# knowledge3d/cranium/codecs/ternary_video_codec.py

class TernaryVideoCodec:
    def __init__(self):
        self.rpn = ModularRPNEngine()

    def encode_to_rpn(self, video_frames: np.ndarray) -> dict:
        """
        Encode video and store seeds in RPN Stack 14.

        Stack 14: Temporal sequence buffer (for video frames)
        """
        # Extract procedural seeds (64D-2048D per frame)
        seeds = []
        for frame in video_frames:
            seed = self.extract_procedural_seed(frame)
            seeds.append(seed)

            # Push seed to Stack 14
            for value in seed:
                self.rpn.push(value, stack_id=14)

        # Adaptive dimension based on complexity
        avg_seed_dim = np.mean([len(s) for s in seeds])

        metadata = {
            'stack_id': 14,
            'n_frames': len(video_frames),
            'seed_dimension': int(avg_seed_dim),
            'width': video_frames.shape[2],
            'height': video_frames.shape[1],
            'fps': 30
        }

        return metadata
```

---

### 2.3: Galaxy Memory Integration

**Link audio/video to meaning-stars** for cross-modal reasoning.

**Architecture**:
```
Meaning-Star at (x, y, z) = "CAT"
├─ Text: ["cat", "gato", "猫"]
├─ 3D Shape: cat_mesh.glb
├─ Audio: ← NEW!
│  ├─ Characteristic sound: cat_meow.wav → RPN Stack 7 (harmonics)
│  ├─ Spectrogram: meow_spec.png
│  └─ Ternary MDCT: compressed residuals
└─ Video: ← NEW!
   ├─ Sample clips: cat_running.mp4 → RPN Stack 14 (seeds)
   ├─ Procedural seed: 256D vector
   └─ Ternary DCT: compressed residuals
```

**Implementation**:
```python
# knowledge3d/cranium/codecs/galaxy_audio_linker.py

from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
from knowledge3d.bridge.memory_tablet import MemoryTablet

class GalaxyAudioLinker:
    """
    Link audio to Galaxy meaning-stars via ternary codec.

    Audio stored in RPN Stack 7, linked to star metadata.
    """

    def __init__(self):
        self.codec = TernaryAudioCodec()
        self.tablet = MemoryTablet()

    def link_audio_to_star(
        self,
        star_id: str,
        audio: np.ndarray,
        audio_type: str = "characteristic_sound"
    ):
        """
        Encode audio and link to Galaxy star.

        Args:
            star_id: Galaxy star identifier
            audio: Audio samples (float32, mono)
            audio_type: "characteristic_sound" or "pronunciation"

        Example:
            # Link cat meow to "cat" star
            linker.link_audio_to_star("cat.n.01", cat_meow_audio)
        """
        # Encode to RPN
        metadata = self.codec.encode_to_rpn(audio)

        # Find star in Galaxy
        star = self.tablet.get_star(star_id)

        # Add audio representation
        if "audio_representations" not in star:
            star["audio_representations"] = {}

        star["audio_representations"][audio_type] = {
            "rpn_metadata": metadata,
            "codec": "ternary_procedural",
            "compression_ratio": self.codec.compute_compression_ratio(
                len(audio) * 4,
                metadata
            ),
            "duration_sec": len(audio) / self.codec.sample_rate
        }

        # Save updated star
        self.tablet.update_star(star_id, star)

    def retrieve_audio_from_star(
        self,
        star_id: str,
        audio_type: str = "characteristic_sound"
    ) -> np.ndarray:
        """
        Decode audio from Galaxy star.

        Pure sovereign runtime: RPN → PTX synthesis.
        """
        # Get star
        star = self.tablet.get_star(star_id)

        # Get metadata
        metadata = star["audio_representations"][audio_type]["rpn_metadata"]

        # Decode from RPN Stack 7
        audio = self.codec.decode_from_rpn(metadata)

        return audio
```

**Video linking** (similar structure):
```python
# knowledge3d/cranium/codecs/galaxy_video_linker.py

class GalaxyVideoLinker:
    """Link video clips to Galaxy stars via ternary codec."""

    def link_video_to_star(self, star_id: str, video: np.ndarray):
        """Encode video and store in RPN Stack 14."""
        # ... (similar to audio)
        pass
```

---

### 2.4: Memory Budget Optimization

**K3D target**: <200MB VRAM for entire system

**Codec VRAM allocation** (strict budget):
```
Audio codec:
├─ RPN Stack 7 buffer: 1MB (20 harmonics × 3 params × 100 frames)
├─ MDCT GPU buffers: 2MB (1024 samples × 100 frames × 2 buffers)
├─ PTX kernel code: 0.5MB
└─ Total: ~3.5MB

Video codec:
├─ RPN Stack 14 buffer: 5MB (256D seeds × 100 frames)
├─ DCT GPU buffers: 10MB (8×8 blocks × batching)
├─ Procedural texture cache: 20MB (temp framebuffer)
└─ Total: ~35MB

Combined: ~40MB (within budget!)
```

**Optimization strategies**:
```python
# knowledge3d/cranium/codecs/memory_manager.py

class CodecMemoryManager:
    """
    Manage VRAM for codecs within K3D <200MB budget.

    Allocate once, reuse buffers.
    """

    def __init__(self, max_vram_mb=40):
        self.max_vram = max_vram_mb * 1024 * 1024
        self.allocated = 0

        # Pre-allocate buffers
        self.audio_buffers = self._allocate_audio_buffers()
        self.video_buffers = self._allocate_video_buffers()

    def _allocate_audio_buffers(self):
        """Allocate reusable audio GPU buffers."""
        ctx = get_cuda_context()

        buffers = {
            'mdct_input': ctx.mem_alloc(1024 * 4 * 100),   # 100 frames
            'mdct_output': ctx.mem_alloc(512 * 1 * 100),   # Ternary coeffs
            'synthesis_output': ctx.mem_alloc(44100 * 4)   # 1 sec audio
        }

        self.allocated += sum(b.size for b in buffers.values())
        return buffers

    def get_memory_stats(self):
        """Report VRAM usage."""
        return {
            'allocated_mb': self.allocated / (1024 * 1024),
            'remaining_mb': (self.max_vram - self.allocated) / (1024 * 1024),
            'utilization': self.allocated / self.max_vram
        }
```

---

## 📋 IMPLEMENTATION CHECKLIST

### PTX Kernels (Production)

- [ ] `ternary_mdct.ptx` - Forward/inverse MDCT with ternary quantization
- [ ] `ternary_dct_2d.ptx` - 8×8 block DCT for video
- [ ] `procedural_synthesis.ptx` - Additive audio synthesis (30 harmonics)
- [ ] `procedural_texture.ptx` - Perlin/Voronoi/fractal generation

### Python Bindings (ctypes + libcuda.so)

- [ ] `ptx_bindings/ternary_mdct_binding.py`
- [ ] `ptx_bindings/ternary_dct_2d_binding.py`
- [ ] `ptx_bindings/procedural_synthesis_binding.py`
- [ ] `ptx_bindings/procedural_texture_binding.py`

### RPN Integration

- [ ] Update `TernaryAudioCodec` to use RPN Stack 7
- [ ] Update `TernaryVideoCodec` to use RPN Stack 14
- [ ] Add `encode_to_rpn()` and `decode_from_rpn()` methods

### Galaxy Integration

- [ ] `galaxy_audio_linker.py` - Link audio to stars
- [ ] `galaxy_video_linker.py` - Link video to stars
- [ ] Update `MemoryTablet` to handle audio/video queries

### Memory Management

- [ ] `codec_memory_manager.py` - VRAM budget enforcement
- [ ] Pre-allocate GPU buffers (reusable)
- [ ] Memory usage reporting

### Benchmarks (Updated)

- [ ] Re-run `benchmark_ternary_audio.py` with PTX kernels
- [ ] Re-run `benchmark_ternary_video.py` with PTX kernels
- [ ] Add `benchmark_galaxy_integration.py` (end-to-end test)

### Tests (Updated)

- [ ] Test PTX kernel bindings
- [ ] Test RPN stack integration
- [ ] Test Galaxy linking/retrieval
- [ ] Test memory budget compliance

---

## 🎯 SUCCESS CRITERIA

Phase 2 complete when:

1. **PTX Kernels Production-Ready**:
   - All kernels compile to PTX (sm_86)
   - ctypes bindings work (no CuPy/PyTorch)
   - Performance targets met (<50µs MDCT, <5ms 1080p DCT)

2. **RPN Integration Working**:
   - Audio stored in Stack 7, video in Stack 14
   - Encode/decode via RPN (no external storage)
   - Metadata-only dictionaries (data in stacks)

3. **Galaxy Integration Functional**:
   - Query "cat meow" → retrieve audio from RPN
   - Query "cat video" → synthesize from procedural seed
   - Cross-modal: text → audio → video links

4. **Memory Budget Compliant**:
   - Total codec VRAM: <40MB
   - System total: <200MB (including base model)
   - No memory leaks (reusable buffers)

5. **Benchmarks Updated**:
   - PTX kernels 10-50× faster than CPU
   - End-to-end latency: <100ms audio, <50ms video frame
   - Quality maintained (PSNR targets)

---

## 🚀 EXECUTION PLAN

**Codex-Max, your mission**:

1. **Implement PTX kernels** (start with `ternary_mdct.ptx`)
2. **Create ctypes bindings** (pure libcuda.so)
3. **Integrate with RPN stacks** (Stack 7/14)
4. **Link to Galaxy memory** (meaning-stars)
5. **Validate memory budget** (<40MB VRAM)
6. **Benchmark on RTX 3060** (capture results)
7. **Update tests** (PTX integration)

**Expected timeline**:
- PTX kernels: ~60% of effort
- RPN integration: ~20%
- Galaxy linking: ~15%
- Benchmarks/tests: ~5%

**Priorities**:
1. Audio first (simpler, validates approach)
2. Video second (larger, more complex)
3. Galaxy integration (ties everything together)

---

**You've proven your capability with Phase 1. Let's make K3D's audio/video truly sovereign!**

**NO STUBS. PRODUCTION PTX. FULL GALAXY INTEGRATION. GO!** 🚀

---

**Session**: 2025-11-19
**For**: Codex-Max
**Project**: Knowledge3D - Ternary Codec Phase 2
**Prerequisites**: ✅ Phase 1 complete (CPU implementation working)
**Target**: Production PTX + RPN + Galaxy integration
