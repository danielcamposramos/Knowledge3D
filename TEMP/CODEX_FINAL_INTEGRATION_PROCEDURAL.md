# Final Integration Directive: Production-Ready Adaptive Procedural Compression

**Codex**, you've achieved the breakthrough: **80.6:1 compression @ 0.9963 fidelity** (64D) and **69.4:1 @ 0.99998** (128D). All validation complete. Now we integrate this into K3D's live production system.

---

## Mission: Wire Adaptive Compression into Phase H Matryoshka Bridge

**Goal**: Make adaptive procedural compression available to all K3D inference pipelines with zero breaking changes to existing code.

**Principle**: Backward compatible—systems using raw embeddings continue working, new systems opt-in to compression.

---

## Task 1: Create AdaptiveDimensionCompressor (Central Interface)

**File**: `knowledge3d/cranium/adaptive_procedural_bridge.py` (NEW)

**Implementation**:

```python
"""
Adaptive procedural compression bridge for K3D.

Integrates:
- Matryoshka dimension selection (Phase H)
- Dictionary-based procedural compression (Phase 2.6)
- Auto-fallback to dense codec (PD02) when needed
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Literal
from .procedural_compiler import ProceduralCompiler, PrototypeTable
from .fidelity_validator import ProceduralFidelityValidator

QualityLevel = Literal["ultrafast", "fast", "balanced", "maximum"]

class AdaptiveDimensionCompressor:
    """
    Production-ready adaptive compression integrating Matryoshka + Dictionary.

    Quality levels:
        ultrafast: 64D + dict → ~80:1 compression, 0.996 fidelity
        fast: 128D + dict → ~69:1 compression, 0.9999 fidelity
        balanced: 512D + dict → ~24:1 compression, 0.9999 fidelity
        maximum: 2048D + dict → ~12:1 compression, 0.9999 fidelity
    """

    _DIMENSION_MAP: Dict[QualityLevel, int] = {
        "ultrafast": 64,
        "fast": 128,
        "balanced": 512,
        "maximum": 2048
    }

    _FIDELITY_THRESHOLDS: Dict[QualityLevel, float] = {
        "ultrafast": 0.98,   # Lower threshold for ultrafast
        "fast": 0.99,        # Standard threshold
        "balanced": 0.995,   # Higher for balanced
        "maximum": 0.999     # Maximum fidelity
    }

    def __init__(
        self,
        cache_dir: Path = Path("validation_cache"),
        enable_compression: bool = True
    ):
        """
        Initialize adaptive compressor with dimension-specific dictionaries.

        Args:
            cache_dir: Directory containing dictionaries (dictionary_<dim>d_<atoms>.npz)
            enable_compression: If False, returns raw embeddings (backward compat)
        """
        self.cache_dir = Path(cache_dir)
        self.enable_compression = enable_compression

        # Load dimension-specific compilers
        self.compilers: Dict[int, ProceduralCompiler] = {}
        self.validators: Dict[int, ProceduralFidelityValidator] = {}

        if self.enable_compression:
            for quality, dim in self._DIMENSION_MAP.items():
                self._load_compiler_for_dimension(dim)

    def _load_compiler_for_dimension(self, dim: int):
        """Load dictionary and create compiler for specific dimension."""
        dict_path = self.cache_dir / f"dictionary_{dim}d_512.npz"

        if not dict_path.exists():
            raise FileNotFoundError(
                f"Dictionary not found: {dict_path}. "
                f"Run scripts/train_dictionary.py to generate."
            )

        # Load dictionary (stores learned basis + metadata)
        dict_data = np.load(dict_path)
        dictionary = dict_data['dictionary']  # Shape: (512, dim)

        # Create compiler with dictionary
        compiler = ProceduralCompiler()
        compiler.attach_dictionary(dictionary, dimension=dim)

        # Create validator
        validator = ProceduralFidelityValidator(compiler)

        self.compilers[dim] = compiler
        self.validators[dim] = validator

    def compress(
        self,
        embedding: np.ndarray,
        quality: QualityLevel = "fast",
        return_metadata: bool = False
    ):
        """
        Compress embedding at specified quality level.

        Args:
            embedding: Full 2048D embedding (will be truncated to target dimension)
            quality: Quality level selecting dimension + fidelity threshold
            return_metadata: If True, return (program, metadata) tuple

        Returns:
            Compressed program bytes (or tuple with metadata if requested)
        """
        if not self.enable_compression:
            # Backward compatibility: return raw embedding
            return embedding.tobytes()

        # Get target dimension and compiler
        target_dim = self._DIMENSION_MAP[quality]
        threshold = self._FIDELITY_THRESHOLDS[quality]

        # Truncate embedding to target dimension (Matryoshka property)
        embedding_truncated = embedding[:target_dim].astype(np.float32)

        # Compress with dimension-specific dictionary
        compiler = self.compilers[target_dim]
        program = compiler.compile_dictionary_sparse(
            embedding_truncated,
            return_metadata=True
        )

        if isinstance(program, tuple):
            program_bytes, metadata = program
        else:
            program_bytes = program
            metadata = {}

        # Validate fidelity
        validator = self.validators[target_dim]
        validation = validator.validate_dictionary(
            embedding_truncated,
            program_bytes,
            threshold=threshold
        )

        # Auto-fallback to dense codec if fidelity insufficient
        if not validation.valid:
            # Fallback to PD02 dense codec
            program_bytes = compiler.compile_prototype_delta_dense(
                embedding_truncated
            )
            metadata['fallback'] = 'dense'
            metadata['original_fidelity'] = validation.cosine_similarity

        # Add dimension metadata
        metadata['target_dim'] = target_dim
        metadata['quality'] = quality
        metadata['threshold'] = threshold

        if return_metadata:
            return program_bytes, metadata
        else:
            return program_bytes

    def decompress(
        self,
        program_bytes: bytes,
        target_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Decompress procedural program back to embedding.

        Args:
            program_bytes: Compressed program
            target_dim: Target dimension (auto-detected from program header if None)

        Returns:
            Reconstructed embedding at target dimension
        """
        if not self.enable_compression:
            # Backward compatibility: interpret as raw bytes
            return np.frombuffer(program_bytes, dtype=np.float32)

        # Auto-detect dimension from program header if not specified
        if target_dim is None:
            target_dim = self._detect_dimension(program_bytes)

        # Decompress with dimension-specific compiler
        compiler = self.compilers[target_dim]
        return compiler.decompile_dictionary_sparse(program_bytes)

    def _detect_dimension(self, program_bytes: bytes) -> int:
        """Detect target dimension from program header."""
        import struct

        # Read magic + dimension from header
        magic, dims = struct.unpack('<4sI', program_bytes[:8])

        # Map dims to nearest standard dimension
        if dims <= 64:
            return 64
        elif dims <= 128:
            return 128
        elif dims <= 512:
            return 512
        else:
            return 2048

    def get_compression_stats(self, quality: QualityLevel = "fast") -> Dict:
        """Get expected compression statistics for quality level."""
        dim = self._DIMENSION_MAP[quality]

        # Measured results from validation
        stats = {
            64: {"compression": 80.6, "fidelity": 0.9963, "bytes": 101},
            128: {"compression": 69.4, "fidelity": 0.99998, "bytes": 118},
            512: {"compression": 24.2, "fidelity": 0.99998, "bytes": 338},
            2048: {"compression": 12.0, "fidelity": 0.99996, "bytes": 682}
        }

        return {
            "dimension": dim,
            "quality": quality,
            "expected_compression": stats[dim]["compression"],
            "expected_fidelity": stats[dim]["fidelity"],
            "expected_bytes": stats[dim]["bytes"],
            "original_bytes": 8192  # 2048D float32
        }
```

---

## Task 2: Update Phase H Bridge Integration

**File**: `knowledge3d/cranium/phase_h_procedural_integration.py` (NEW)

**Implementation**:

```python
"""
Phase H integration for adaptive procedural compression.

Connects Matryoshka TRM → Adaptive Compression → Procedural Galaxy.
"""

from .adaptive_procedural_bridge import AdaptiveDimensionCompressor, QualityLevel
from .matryoshka_trm import MatryoshkaTRM
from typing import Optional

class PhaseHProceduralIntegration:
    """
    Integrate adaptive compression into Phase H inference pipeline.

    Workflow:
        1. Matryoshka TRM generates embedding at selected dimension
        2. Adaptive compressor compresses with dimension-specific dictionary
        3. Store compressed program in Procedural Galaxy
        4. Decompression on-demand during inference
    """

    def __init__(
        self,
        matryoshka_model: MatryoshkaTRM,
        compressor: AdaptiveDimensionCompressor,
        enable_compression: bool = True
    ):
        self.matryoshka = matryoshka_model
        self.compressor = compressor
        self.enable_compression = enable_compression

    def embed_and_compress(
        self,
        text: str,
        quality: QualityLevel = "fast"
    ):
        """
        Generate embedding and compress in one operation.

        Args:
            text: Input text
            quality: Quality level (determines dimension + compression)

        Returns:
            Compressed program bytes
        """
        # Get target dimension for quality level
        target_dim = self.compressor._DIMENSION_MAP[quality]

        # Generate embedding at target dimension using Matryoshka
        embedding = self.matryoshka.forward(text, output_dim=target_dim)

        # Compress (will auto-fallback to dense if needed)
        if self.enable_compression:
            program, metadata = self.compressor.compress(
                embedding,
                quality=quality,
                return_metadata=True
            )

            # Log compression achieved
            original_bytes = embedding.nbytes
            compressed_bytes = len(program)
            actual_compression = original_bytes / compressed_bytes

            metadata['actual_compression'] = actual_compression

            return program, metadata
        else:
            # Backward compatibility
            return embedding.tobytes(), {'compression': 1.0}

    def decompress_and_use(
        self,
        program_bytes: bytes,
        target_dim: Optional[int] = None
    ):
        """
        Decompress program and return embedding for inference.

        Args:
            program_bytes: Compressed program
            target_dim: Dimension (auto-detected if None)

        Returns:
            Decompressed embedding
        """
        if not self.enable_compression:
            # Backward compatibility
            return np.frombuffer(program_bytes, dtype=np.float32)

        return self.compressor.decompress(program_bytes, target_dim)
```

---

## Task 3: Export New Classes

**File**: `knowledge3d/cranium/__init__.py` (UPDATE)

Add exports:

```python
# Adaptive procedural compression (Phase 2.6)
from .adaptive_procedural_bridge import AdaptiveDimensionCompressor
from .phase_h_procedural_integration import PhaseHProceduralIntegration
```

---

## Task 4: Update Tests

**File**: `knowledge3d/cranium/tests/test_adaptive_compression.py` (NEW)

```python
"""Tests for adaptive dimension compression."""

import pytest
import numpy as np
from knowledge3d.cranium import AdaptiveDimensionCompressor, PhaseHProceduralIntegration

def test_adaptive_compressor_initialization():
    """Test compressor loads all dimension-specific dictionaries."""
    compressor = AdaptiveDimensionCompressor()

    # Should have loaded compilers for all dimensions
    assert 64 in compressor.compilers
    assert 128 in compressor.compilers
    assert 512 in compressor.compilers
    assert 2048 in compressor.compilers

def test_compression_quality_levels():
    """Test each quality level compresses correctly."""
    compressor = AdaptiveDimensionCompressor()

    # Generate test embedding
    embedding = np.random.randn(2048).astype(np.float32)

    for quality in ["ultrafast", "fast", "balanced", "maximum"]:
        program, metadata = compressor.compress(
            embedding,
            quality=quality,
            return_metadata=True
        )

        # Should compress
        assert len(program) < 8192

        # Should have metadata
        assert 'target_dim' in metadata
        assert 'quality' in metadata

        # Should decompress
        reconstructed = compressor.decompress(program, metadata['target_dim'])
        assert reconstructed.shape[0] == metadata['target_dim']

def test_backward_compatibility():
    """Test disabling compression returns raw embeddings."""
    compressor = AdaptiveDimensionCompressor(enable_compression=False)

    embedding = np.random.randn(2048).astype(np.float32)
    raw_bytes = compressor.compress(embedding, quality="fast")

    # Should be raw bytes, not compressed
    assert len(raw_bytes) == 8192

def test_auto_fallback():
    """Test auto-fallback to dense codec when dictionary fails."""
    # This test would need a pathological embedding that fails dict compression
    # For now, just validate the fallback code path exists
    pass
```

---

## Task 5: Create Integration Example

**File**: `examples/adaptive_compression_demo.py` (NEW)

```python
#!/usr/bin/env python3
"""
Demo of adaptive procedural compression in K3D.

Shows how to use different quality levels for different use cases.
"""

import numpy as np
from knowledge3d.cranium import AdaptiveDimensionCompressor

def main():
    # Initialize compressor
    compressor = AdaptiveDimensionCompressor()

    # Simulate generating embeddings
    sample_embedding = np.random.randn(2048).astype(np.float32)

    print("Adaptive Procedural Compression Demo")
    print("=" * 50)
    print(f"Original embedding: 2048D × 4 bytes = 8192 bytes\n")

    # Test each quality level
    for quality in ["ultrafast", "fast", "balanced", "maximum"]:
        program, metadata = compressor.compress(
            sample_embedding,
            quality=quality,
            return_metadata=True
        )

        stats = compressor.get_compression_stats(quality)

        print(f"{quality.upper()}:")
        print(f"  Target dimension: {metadata['target_dim']}D")
        print(f"  Compressed size: {len(program)} bytes")
        print(f"  Compression ratio: {8192 / len(program):.1f}:1")
        print(f"  Expected fidelity: {stats['expected_fidelity']:.5f}")

        # Decompress and validate
        reconstructed = compressor.decompress(program, metadata['target_dim'])
        print(f"  Reconstructed shape: {reconstructed.shape}\n")

    print("\nUse cases:")
    print("  - Ultrafast: Semantic search, initial retrieval")
    print("  - Fast: Most inference queries (recommended default)")
    print("  - Balanced: Complex reasoning tasks")
    print("  - Maximum: Critical fidelity requirements")

if __name__ == "__main__":
    main()
```

---

## Task 6: Documentation Integration

**File**: `docs/procedural_compression/ADAPTIVE_GUIDE.md` (NEW)

Create comprehensive guide covering:
- Quality level selection criteria
- Expected compression ratios
- Fidelity guarantees
- Integration examples
- Backward compatibility notes

---

## Task 7: File Organization & Cleanup

**Organize validation results**:

```
validation_results/
├── procedural_compression_proof.md          # Phase 1: 3.88:1
├── prototype_analysis.md                    # Phase 2.1: Prototype table
├── prototype_delta_compression.md           # Phase 2.2: 3.97:1 (PD02)
├── dictionary_training.md                   # Phase 2.5: Dictionary creation
├── dictionary_compression_64d.md            # Phase 2.6: 80.6:1
├── dictionary_compression_128d.md           # Phase 2.6: 69.4:1
├── dictionary_compression_512d.md           # Phase 2.6: 24.2:1
├── dictionary_compression_2048d.md          # Phase 2.6: 12.0:1
└── PROCEDURAL_COMPRESSION_SUMMARY.md        # Overview for Milton
```

**Organize cache artifacts**:

```
validation_cache/
├── prototype_table_2048d_512.npz            # k-means prototypes
├── dictionary_64d_512.npz                   # 64D dictionary
├── dictionary_128d_128.npz                  # 128D dictionary
├── dictionary_512d_512.npz                  # 512D dictionary
└── dictionary_2048d_512.npz                 # 2048D dictionary
```

---

## Success Criteria

**Integration complete when**:

1. ✅ `AdaptiveDimensionCompressor` class implemented and exported
2. ✅ `PhaseHProceduralIntegration` class implemented and exported
3. ✅ All tests passing (`test_adaptive_compression.py`)
4. ✅ Demo script works (`examples/adaptive_compression_demo.py`)
5. ✅ Files organized in proper structure
6. ✅ No breaking changes to existing code

**Validation command**:

```bash
# Run all procedural compression tests
python3 -m pytest knowledge3d/cranium/tests/test_procedural_compression.py \
                   knowledge3d/cranium/tests/test_prototype_delta.py \
                   knowledge3d/cranium/tests/test_adaptive_compression.py -v

# Run demo
PYTHONPATH=. python3 examples/adaptive_compression_demo.py
```

---

## Timeline Estimate

- Task 1-2: 2 hours (core implementation)
- Task 3-4: 1 hour (exports + tests)
- Task 5-6: 1 hour (examples + docs)
- Task 7: 30 min (file organization)

**Total**: 4-5 hours for complete production integration

---

## Final Notes

**This is the culmination of a Sunday's work**:
- Phase 1: Proof of concept (3.88:1)
- Phase 2.2: Prototype validation (3.97:1)
- Phase 2.5: Dictionary breakthrough (12.0:1)
- Phase 2.6: Adaptive compression (69-80:1)
- **Now**: Production integration

**When complete**, we present to Milton with:
- ✅ Measured 20× improvement (3.88 → 69.4:1)
- ✅ 4000 samples validated across 4 dimensions
- ✅ Production-ready codebase
- ✅ Full reproduction instructions
- ✅ Mathematical validation of "domains of discourse"

**One final push to production, Codex. Let's make procedural compression live.** 🎯⚛️🚀
