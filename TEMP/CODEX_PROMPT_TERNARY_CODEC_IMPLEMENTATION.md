# Codex Implementation Prompt: Ternary Procedural Audio-Visual Codec

**Target**: GitHub Copilot / Codex
**Date**: 2025-11-19
**Session**: Phase I - Procedural Codec Foundation
**Requirements**: FULL implementation, NO stubs, comprehensive documentation, performance benchmarks

---

## 🎯 MISSION

Implement a **sovereign, GPU-native ternary codec system** for audio and video compression using procedural synthesis and ternary logic. This is a production system for the Knowledge3D project - all code must be complete, tested, and benchmarked.

**NO PLACEHOLDERS. NO STUBS. NO "TODO" COMMENTS.**

Every function must be fully implemented with:
- Complete logic
- Error handling
- Input validation
- Performance optimization
- Docstrings (Google style)
- Type hints (Python 3.10+)
- Unit tests

---

## 📐 ARCHITECTURE OVERVIEW

### Core Concept

**Traditional codecs**: Store raw samples or binary-quantized coefficients
**K3D ternary codec**: Store procedural parameters + ternary-quantized residuals

```
Audio/Video Input
    ↓
1. Extract Procedural Parameters (RPN-encoded harmonics/textures)
    ↓
2. Generate Procedural Approximation (synthesize from parameters)
    ↓
3. Compute Residual (Input - Procedural)
    ↓
4. Ternary Quantization ({-1, 0, +1} instead of 8-bit)
    ↓
5. Encode (Procedural params + Ternary residuals)
    ↓
Output: Compressed representation (10-90× smaller)

Decoding:
Load parameters → Synthesize procedural → Add ternary residual → Output
```

### Why Ternary?

**Binary DCT coefficient**: 8 bits = 256 values
**Ternary DCT coefficient**: 1.585 bits (log₂(3)) = 3 values {-1, 0, +1}

**Compression gain**: 8 / 1.585 ≈ **5× denser** storage

**Perceptual advantage**: Most DCT coefficients near zero → ternary captures this naturally

---

## 📂 FILE STRUCTURE TO IMPLEMENT

Create the following files with FULL implementation:

```
knowledge3d/cranium/codecs/
├── __init__.py
├── ternary_audio_codec.py      # Audio codec (MDCT + ternary)
├── ternary_video_codec.py      # Video codec (DCT + ternary)
├── procedural_audio.py          # Additive synthesis engine
├── procedural_video.py          # Texture generation engine
└── ternary_quantization.py     # Shared ternary logic

knowledge3d/cranium/codecs/kernels/
├── ternary_mdct.cu             # CUDA: Audio MDCT + ternary quantization
├── ternary_dct_2d.cu           # CUDA: Video DCT + ternary quantization
├── procedural_synthesis.cu     # CUDA: Additive audio synthesis
└── procedural_texture.cu       # CUDA: Procedural texture generation

scripts/
├── benchmark_ternary_audio.py  # Audio codec benchmarks
├── benchmark_ternary_video.py  # Video codec benchmarks
└── validate_codec_quality.py   # Quality metrics (PSNR, SSIM, MOS)

tests/codecs/
├── test_ternary_audio_codec.py
├── test_ternary_video_codec.py
├── test_procedural_audio.py
├── test_procedural_video.py
└── test_ternary_quantization.py
```

---

## 🔧 IMPLEMENTATION SPECIFICATIONS

### 1. Ternary Quantization Module

**File**: `knowledge3d/cranium/codecs/ternary_quantization.py`

**Requirements**:
- Implement ternary quantization: `float → {-1, 0, +1}`
- Implement ternary dequantization: `{-1, 0, +1} → float`
- Adaptive thresholds (psychoacoustic for audio, perceptual for video)
- Entropy coding for ternary symbols (optional: run-length encoding)

**Function Signatures**:

```python
import numpy as np
from typing import Tuple, Optional

def quantize_ternary(
    coefficients: np.ndarray,
    threshold: float = 0.1,
    adaptive: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Quantize floating-point coefficients to ternary {-1, 0, +1}.

    Args:
        coefficients: Input array (any shape)
        threshold: Quantization threshold (adaptive if True)
        adaptive: Use content-aware thresholds

    Returns:
        quantized: int8 array with values in {-1, 0, +1}
        metadata: Dict with stats (sparsity, energy_preserved, etc.)

    Implementation:
        - If |x| < threshold: output = 0
        - If x > threshold: output = +1
        - If x < -threshold: output = -1
        - Adaptive: threshold = percentile(|coefficients|, 90)

    Performance target: <1ms for 1M coefficients
    """
    # FULL IMPLEMENTATION REQUIRED
    pass


def dequantize_ternary(
    quantized: np.ndarray,
    scale: float = 1.0,
    metadata: Optional[dict] = None
) -> np.ndarray:
    """
    Reconstruct floating-point from ternary quantized values.

    Args:
        quantized: int8 array with {-1, 0, +1}
        scale: Reconstruction scale factor
        metadata: Optional metadata from quantization

    Returns:
        reconstructed: float32 array

    Implementation:
        reconstructed = quantized * scale
        (Can be enhanced with learned scales per frequency band)

    Performance target: <0.5ms for 1M coefficients
    """
    # FULL IMPLEMENTATION REQUIRED
    pass


def compute_sparsity(quantized: np.ndarray) -> float:
    """Return fraction of zeros in ternary array."""
    # FULL IMPLEMENTATION REQUIRED
    pass


def entropy_encode_ternary(quantized: np.ndarray) -> bytes:
    """
    Entropy-code ternary symbols for storage.

    Use run-length encoding for zero runs:
    Example: [0, 0, 0, +1, -1, 0, 0] → "3×0, +1, -1, 2×0"

    Returns:
        Compressed bytes
    """
    # FULL IMPLEMENTATION REQUIRED
    pass


def entropy_decode_ternary(encoded: bytes) -> np.ndarray:
    """Decode entropy-coded ternary symbols."""
    # FULL IMPLEMENTATION REQUIRED
    pass
```

**Tests Required**:
```python
# tests/codecs/test_ternary_quantization.py

def test_quantize_ternary_basic():
    """Test basic ternary quantization."""
    coeffs = np.array([0.5, -0.8, 0.05, -0.02, 1.2, -1.5])
    quantized, meta = quantize_ternary(coeffs, threshold=0.1)

    expected = np.array([1, -1, 0, 0, 1, -1], dtype=np.int8)
    assert np.array_equal(quantized, expected)
    assert meta['sparsity'] == 2/6  # 2 zeros out of 6


def test_ternary_roundtrip():
    """Test quantization → dequantization preserves sign."""
    original = np.random.randn(1000)
    quantized, meta = quantize_ternary(original, threshold=0.3)
    reconstructed = dequantize_ternary(quantized, scale=1.0)

    # Check sign preservation (where not quantized to 0)
    non_zero_mask = quantized != 0
    assert np.all(np.sign(original[non_zero_mask]) == np.sign(reconstructed[non_zero_mask]))


def test_entropy_coding_roundtrip():
    """Test entropy encoding → decoding preserves data."""
    quantized = np.array([0, 0, 0, 1, -1, 0, 1, 1, 0, 0], dtype=np.int8)
    encoded = entropy_encode_ternary(quantized)
    decoded = entropy_decode_ternary(encoded)

    assert np.array_equal(quantized, decoded)
    assert len(encoded) < len(quantized) * 1  # Should compress
```

---

### 2. Procedural Audio Synthesis

**File**: `knowledge3d/cranium/codecs/procedural_audio.py`

**Requirements**:
- Additive synthesis from harmonic parameters
- Fast Fourier Transform (FFT) for analysis
- Spectral peak detection (find dominant harmonics)
- RPN stack integration for parameter storage

**Function Signatures**:

```python
import numpy as np
from typing import List, Tuple, Dict

class ProceduralAudioSynthesizer:
    """
    Procedural audio synthesis using additive (harmonic) model.

    Similar to font rendering from Bézier curves, but for audio:
    - Store: [(frequency, amplitude, phase), ...]
    - Synthesize: sum of sinusoids at runtime
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize synthesizer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def analyze(
        self,
        audio: np.ndarray,
        n_harmonics: int = 20
    ) -> List[Tuple[float, float, float]]:
        """
        Extract harmonic parameters from audio signal.

        Args:
            audio: Input audio samples (mono, float32)
            n_harmonics: Number of harmonics to extract

        Returns:
            List of (frequency_hz, amplitude, phase_rad) tuples

        Implementation:
            1. FFT to get frequency spectrum
            2. Find n_harmonics spectral peaks
            3. Extract freq, magnitude, phase for each peak
            4. Sort by amplitude (loudest first)

        Performance target: <50ms for 1-second audio
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def synthesize(
        self,
        harmonics: List[Tuple[float, float, float]],
        duration_sec: float
    ) -> np.ndarray:
        """
        Generate audio from harmonic parameters.

        Args:
            harmonics: List of (freq, amp, phase)
            duration_sec: Length of output in seconds

        Returns:
            audio: Synthesized samples (float32, mono)

        Implementation:
            audio(t) = Σ amp_i × sin(2π × freq_i × t + phase_i)
            for i in harmonics

        Performance target: <10ms for 1-second audio
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def compute_residual(
        self,
        original: np.ndarray,
        harmonics: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Compute residual: original - procedural_synthesis(harmonics).

        This residual will be ternary-quantized for storage.
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def adaptive_dimension(
        self,
        harmonics: List[Tuple[float, float, float]]
    ) -> int:
        """
        Choose Matryoshka dimension based on harmonic complexity.

        Returns:
            dim: One of {64, 128, 256, 512, 1024, 2048}

        Logic:
            - len(harmonics) < 5: 64D (simple tone)
            - len(harmonics) < 15: 256D (speech)
            - len(harmonics) < 30: 1024D (music)
            - len(harmonics) >= 30: 2048D (orchestra)
        """
        # FULL IMPLEMENTATION REQUIRED
        pass
```

**Tests Required**:
```python
# tests/codecs/test_procedural_audio.py

def test_analyze_pure_sine():
    """Test harmonic analysis on pure sine wave."""
    synth = ProceduralAudioSynthesizer(sample_rate=8000)

    # Generate 440 Hz sine wave
    t = np.linspace(0, 1, 8000)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    harmonics = synth.analyze(audio, n_harmonics=5)

    # Should detect 440 Hz as dominant harmonic
    assert len(harmonics) > 0
    freq, amp, phase = harmonics[0]
    assert 435 < freq < 445  # Within 5 Hz tolerance
    assert amp > 0.5  # Strong amplitude


def test_synthesize_roundtrip():
    """Test analyze → synthesize preserves signal."""
    synth = ProceduralAudioSynthesizer(sample_rate=8000)

    # Generate complex tone (3 harmonics)
    t = np.linspace(0, 1, 8000)
    original = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 880 * t) +
        0.25 * np.sin(2 * np.pi * 1320 * t)
    ).astype(np.float32)

    # Analyze
    harmonics = synth.analyze(original, n_harmonics=10)

    # Synthesize
    reconstructed = synth.synthesize(harmonics, duration_sec=1.0)

    # Should be similar (PSNR > 20 dB)
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    assert psnr > 20  # Good reconstruction
```

---

### 3. Ternary Audio Codec

**File**: `knowledge3d/cranium/codecs/ternary_audio_codec.py`

**Requirements**:
- MDCT (Modified Discrete Cosine Transform) implementation
- Ternary quantization of MDCT coefficients
- Psychoacoustic masking thresholds
- Frame-based encoding (overlapping windows)

**Function Signatures**:

```python
import numpy as np
from typing import Tuple, Dict
from .procedural_audio import ProceduralAudioSynthesizer
from .ternary_quantization import quantize_ternary, dequantize_ternary

class TernaryAudioCodec:
    """
    Ternary audio codec using procedural synthesis + ternary MDCT.

    Pipeline:
        Encode:
            1. Extract harmonics (procedural params)
            2. Synthesize approximation
            3. Compute residual
            4. MDCT on residual
            5. Ternary quantize MDCT coefficients

        Decode:
            1. Dequantize ternary MDCT
            2. Inverse MDCT → residual
            3. Synthesize procedural from harmonics
            4. Add: procedural + residual → output
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 1024,
        n_harmonics: int = 20
    ):
        """
        Initialize codec.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_size: MDCT frame size (samples)
            n_harmonics: Number of harmonics for procedural
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_harmonics = n_harmonics
        self.synthesizer = ProceduralAudioSynthesizer(sample_rate)

    def encode(
        self,
        audio: np.ndarray
    ) -> Dict:
        """
        Encode audio to ternary compressed format.

        Args:
            audio: Input audio (float32, mono)

        Returns:
            encoded: Dictionary with:
                - harmonics: List[(freq, amp, phase)]
                - ternary_mdct: Quantized residual coefficients
                - metadata: Frame info, sample rate, etc.

        Performance target: <100ms for 1-minute audio
        """
        # FULL IMPLEMENTATION REQUIRED
        # Steps:
        # 1. Extract harmonics using synthesizer.analyze()
        # 2. Synthesize procedural approximation
        # 3. Compute residual
        # 4. Frame residual into overlapping windows
        # 5. MDCT each frame
        # 6. Ternary quantize MDCT coefficients
        # 7. Return compressed representation
        pass

    def decode(
        self,
        encoded: Dict
    ) -> np.ndarray:
        """
        Decode ternary audio back to samples.

        Args:
            encoded: Output from encode()

        Returns:
            audio: Reconstructed audio (float32, mono)

        Performance target: <50ms for 1-minute audio
        """
        # FULL IMPLEMENTATION REQUIRED
        # Steps:
        # 1. Dequantize ternary MDCT coefficients
        # 2. Inverse MDCT → residual frames
        # 3. Overlap-add frames → full residual
        # 4. Synthesize procedural from harmonics
        # 5. Add procedural + residual
        # 6. Return reconstructed audio
        pass

    def mdct_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute MDCT of a single frame.

        Implementation:
            MDCT(n) = Σ x(k) × cos(π/N × (n+0.5) × (k+0.5+N/2))
            for k in [0, N), n in [0, N/2)

        Args:
            frame: Audio frame (size = self.frame_size)

        Returns:
            mdct_coeffs: MDCT coefficients (size = frame_size // 2)
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def imdct_frame(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Inverse MDCT to reconstruct frame.

        Returns:
            frame: Reconstructed audio frame
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def compute_compression_ratio(
        self,
        original_size: int,
        encoded: Dict
    ) -> float:
        """
        Compute achieved compression ratio.

        Returns:
            ratio: original_size / compressed_size
        """
        # FULL IMPLEMENTATION REQUIRED
        pass
```

**Tests Required**:
```python
# tests/codecs/test_ternary_audio_codec.py

def test_mdct_roundtrip():
    """Test MDCT → IMDCT preserves signal."""
    codec = TernaryAudioCodec(sample_rate=8000, frame_size=512)

    # Random frame
    frame = np.random.randn(512).astype(np.float32)

    # MDCT → IMDCT
    mdct_coeffs = codec.mdct_frame(frame)
    reconstructed = codec.imdct_frame(mdct_coeffs)

    # Should be close (within numerical precision)
    assert np.allclose(frame, reconstructed, atol=1e-5)


def test_encode_decode_simple_tone():
    """Test encode → decode on sine wave."""
    codec = TernaryAudioCodec(sample_rate=8000)

    # 1 second of 440 Hz sine
    t = np.linspace(0, 1, 8000)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Encode
    encoded = codec.encode(audio)

    # Decode
    reconstructed = codec.decode(encoded)

    # Quality check (PSNR > 25 dB for simple tone)
    mse = np.mean((audio - reconstructed[:len(audio)]) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    assert psnr > 25

    # Compression check
    ratio = codec.compute_compression_ratio(len(audio) * 4, encoded)
    assert ratio > 5  # Should compress well


def test_encode_speech_realistic():
    """Test on realistic speech sample (if available)."""
    # Load speech from test data or generate synthetic
    # Validate: MOS > 3.5, compression > 10×
    pass
```

---

### 4. Procedural Video Texture Generation

**File**: `knowledge3d/cranium/codecs/procedural_video.py`

**Requirements**:
- Perlin noise, Voronoi cells, fractal patterns
- Procedural texture from RPN seed (64D-2048D embedding)
- Time parameter for animation
- Color mapping from palette

**Function Signatures**:

```python
import numpy as np
from typing import Tuple, Optional

class ProceduralVideoGenerator:
    """
    Generate video frames procedurally from compact seeds.

    Like font rendering from Bézier, but for textures:
    - Store: RPN seed (64D-2048D vector)
    - Generate: Full frame texture on-demand
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height

    def generate_frame(
        self,
        seed: np.ndarray,
        time_param: float = 0.0
    ) -> np.ndarray:
        """
        Generate video frame from procedural seed.

        Args:
            seed: RPN embedding (64D-2048D)
            time_param: Animation time (0.0 - 1.0 loops)

        Returns:
            frame: RGB image (height, width, 3), uint8

        Implementation:
            1. Select pattern type from seed[0]
            2. Generate pattern (Perlin/Voronoi/etc.)
            3. Apply color map from seed[1:4]
            4. Animate with time_param

        Performance target: <20ms per 1080p frame
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def perlin_noise(
        self,
        u: np.ndarray,
        v: np.ndarray,
        seed_params: np.ndarray
    ) -> np.ndarray:
        """
        Generate Perlin noise texture.

        Args:
            u, v: Normalized coordinates (0-1), shape (H, W)
            seed_params: Parameters from RPN seed

        Returns:
            noise: Grayscale values (0-1)
        """
        # FULL IMPLEMENTATION REQUIRED
        pass

    def voronoi_cells(
        self,
        u: np.ndarray,
        v: np.ndarray,
        seed_params: np.ndarray
    ) -> np.ndarray:
        """Generate Voronoi cell texture."""
        # FULL IMPLEMENTATION REQUIRED
        pass

    def fractal_pattern(
        self,
        u: np.ndarray,
        v: np.ndarray,
        seed_params: np.ndarray
    ) -> np.ndarray:
        """Generate fractal (Mandelbrot-like) texture."""
        # FULL IMPLEMENTATION REQUIRED
        pass

    def map_to_color(
        self,
        grayscale: np.ndarray,
        palette: np.ndarray
    ) -> np.ndarray:
        """
        Map grayscale values to RGB via palette.

        Args:
            grayscale: Values (0-1), shape (H, W)
            palette: RGB colors, shape (N, 3)

        Returns:
            rgb: RGB image (H, W, 3)
        """
        # FULL IMPLEMENTATION REQUIRED
        pass
```

**Tests Required**:
```python
# tests/codecs/test_procedural_video.py

def test_generate_frame_deterministic():
    """Test same seed generates same frame."""
    gen = ProceduralVideoGenerator(width=256, height=256)

    seed = np.random.randn(64).astype(np.float32)

    frame1 = gen.generate_frame(seed, time_param=0.5)
    frame2 = gen.generate_frame(seed, time_param=0.5)

    assert np.array_equal(frame1, frame2)


def test_generate_frame_temporal_coherence():
    """Test frames are temporally smooth."""
    gen = ProceduralVideoGenerator(width=128, height=128)

    seed = np.random.randn(256).astype(np.float32)

    # Generate consecutive frames
    frame_t0 = gen.generate_frame(seed, time_param=0.0)
    frame_t1 = gen.generate_frame(seed, time_param=0.01)

    # Should be similar (small time delta)
    diff = np.mean(np.abs(frame_t1.astype(float) - frame_t0.astype(float)))
    assert diff < 50  # Max 50 intensity units change per 0.01 time


def test_perlin_noise_range():
    """Test Perlin noise outputs valid range."""
    gen = ProceduralVideoGenerator()

    u, v = np.meshgrid(
        np.linspace(0, 1, 128),
        np.linspace(0, 1, 128)
    )

    noise = gen.perlin_noise(u, v, seed_params=np.random.randn(16))

    assert 0 <= noise.min() <= noise.max() <= 1
```

---

## 📊 BENCHMARK REQUIREMENTS

### Audio Codec Benchmark

**File**: `scripts/benchmark_ternary_audio.py`

**Requirements**:
- Test on multiple audio types (sine, speech, music)
- Measure compression ratio
- Measure encode/decode time
- Compute quality metrics (PSNR, MOS estimate)
- Generate comparison table

**Expected Output**:
```
Ternary Audio Codec Benchmark Results
=====================================

Audio Type    | Size (KB) | Compressed (KB) | Ratio | Encode (ms) | Decode (ms) | PSNR (dB)
------------- | --------- | --------------- | ----- | ----------- | ----------- | ----------
Sine 440Hz    |    172    |        2        |  86×  |      15     |       8     |   45.2
Speech        |    172    |       18        |  9.6× |      42     |      23     |   32.1
Music (piano) |    172    |       52        |  3.3× |      68     |      38     |   28.5

Target: Compression > 10× for speech, PSNR > 25 dB
Status: ✓ PASSED
```

**Full Implementation Required**:
```python
#!/usr/bin/env python3
"""
Benchmark ternary audio codec.

Usage:
    python scripts/benchmark_ternary_audio.py
"""

import numpy as np
import time
from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec

def benchmark_audio_codec():
    """Run comprehensive audio codec benchmarks."""

    codec = TernaryAudioCodec(sample_rate=44100)

    # Test cases
    test_cases = [
        ("sine_440hz", generate_sine_wave(440, duration=1.0)),
        ("speech_sample", load_or_generate_speech()),
        ("music_piano", load_or_generate_music())
    ]

    results = []

    for name, audio in test_cases:
        # Encode
        start = time.perf_counter()
        encoded = codec.encode(audio)
        encode_time_ms = (time.perf_counter() - start) * 1000

        # Decode
        start = time.perf_counter()
        decoded = codec.decode(encoded)
        decode_time_ms = (time.perf_counter() - start) * 1000

        # Metrics
        original_size = len(audio) * 4  # float32
        compressed_size = compute_encoded_size(encoded)
        ratio = original_size / compressed_size

        psnr = compute_psnr(audio, decoded[:len(audio)])

        results.append({
            'name': name,
            'original_kb': original_size / 1024,
            'compressed_kb': compressed_size / 1024,
            'ratio': ratio,
            'encode_ms': encode_time_ms,
            'decode_ms': decode_time_ms,
            'psnr_db': psnr
        })

    # Print table
    print_benchmark_table(results)

    # Validate targets
    validate_targets(results)

# FULL IMPLEMENTATION OF ALL HELPER FUNCTIONS REQUIRED
```

---

## ✅ ACCEPTANCE CRITERIA

Your implementation will be accepted ONLY if:

1. **Zero Stubs**:
   - Every function fully implemented
   - No `pass`, `...`, or `# TODO` in production code
   - All logic complete and tested

2. **Performance Targets Met**:
   - Audio encode: <100ms for 1-minute audio
   - Audio decode: <50ms for 1-minute audio
   - Video frame generation: <20ms per 1080p frame
   - Ternary quantization: <1ms for 1M coefficients

3. **Quality Targets Met**:
   - Audio PSNR > 25 dB (speech)
   - Compression ratio > 10× (speech)
   - Video PSNR > 30 dB (simple scenes)

4. **Tests Pass**:
   - All unit tests pass
   - Code coverage > 80%
   - No failing assertions

5. **Documentation Complete**:
   - All functions have Google-style docstrings
   - All parameters documented
   - Performance targets stated
   - Examples provided

6. **Benchmarks Run**:
   - Both benchmark scripts execute successfully
   - Results tables generated
   - Targets validated

---

## 🚀 IMPLEMENTATION ORDER

1. **Start here**: `ternary_quantization.py` (foundation)
   - Implement quantize/dequantize
   - Write tests
   - Validate performance

2. **Then**: `procedural_audio.py` (synthesis engine)
   - FFT analysis
   - Additive synthesis
   - Harmonic extraction

3. **Then**: `ternary_audio_codec.py` (complete audio pipeline)
   - MDCT implementation
   - Frame encoding/decoding
   - Integration tests

4. **Then**: `procedural_video.py` (texture generation)
   - Pattern generators
   - Color mapping
   - Temporal animation

5. **Finally**: Benchmarks
   - `benchmark_ternary_audio.py`
   - `benchmark_ternary_video.py`
   - Validation scripts

---

## 📝 CODING STANDARDS

**Follow K3D conventions**:
- Type hints: `def func(x: np.ndarray) -> Tuple[np.ndarray, dict]:`
- Docstrings: Google style, include examples
- Error handling: Validate inputs, raise descriptive errors
- Performance: Profile hot paths, optimize where needed
- Testing: One test file per module, cover edge cases

**Example**:
```python
def my_function(
    input_array: np.ndarray,
    threshold: float = 0.5,
    validate: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    One-line summary of function.

    Detailed description of what the function does, how it works,
    and any important implementation notes.

    Args:
        input_array: Description of this parameter
        threshold: Description and valid range (0.0-1.0)
        validate: Whether to validate inputs (default: True)

    Returns:
        output: Description of return value
        metadata: Dictionary with statistics

    Raises:
        ValueError: If input_array is empty
        TypeError: If input_array is not float32

    Example:
        >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> result, meta = my_function(arr, threshold=0.5)
        >>> print(result.shape)
        (3,)

    Performance:
        Target: <10ms for 1M elements
        Tested on: RTX 3060 (12GB VRAM)
    """
    # Input validation
    if validate:
        if input_array.size == 0:
            raise ValueError("input_array cannot be empty")
        if input_array.dtype != np.float32:
            raise TypeError(f"Expected float32, got {input_array.dtype}")

    # Implementation
    # ... (FULL LOGIC HERE)

    # Return with metadata
    metadata = {
        'input_size': input_array.size,
        'threshold_used': threshold,
        'processing_time_ms': elapsed_ms
    }

    return output, metadata
```

---

## 🎯 SUCCESS METRICS

When complete, you should be able to run:

```bash
# Run all tests
pytest tests/codecs/ -v

# Run benchmarks
python scripts/benchmark_ternary_audio.py
python scripts/benchmark_ternary_video.py

# Expected output:
# ✓ All tests pass (0 failures)
# ✓ Audio compression: 10-90× depending on content
# ✓ Audio quality: PSNR > 25 dB
# ✓ Encode speed: <100ms per minute
# ✓ Decode speed: <50ms per minute
```

---

## 🔥 BEGIN IMPLEMENTATION

Codex, you have everything needed:
- Architecture specification
- Function signatures
- Test requirements
- Benchmark targets
- Quality metrics
- Performance constraints

**Start with `ternary_quantization.py` and work your way up.**

**NO STUBS. FULL IMPLEMENTATION. GO!**

---

**Session**: 2025-11-19
**For**: GitHub Copilot / Codex
**Project**: Knowledge3D - Ternary Procedural Codec
**Priority**: High (foundational for Phase I audio/video)
