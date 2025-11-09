import numpy as np
import pytest
import tempfile
from pathlib import Path

from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor


def _create_dictionary(path: Path, dim: int, atoms: int) -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(atoms, dim)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, atoms=data)


def _create_prototype_table(path: Path, dim: int, count: int) -> None:
    rng = np.random.default_rng(11)
    prototypes = rng.normal(size=(count, dim)).astype(np.float32)
    metadata = {"num_prototypes": count, "dimension": dim}
    np.savez_compressed(path, prototypes=prototypes, metadata=np.array(["{}"]))


def test_compressor_initialisation_custom_map():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        proto_path = cache / "prototype_table_32d_8.npz"
        _create_prototype_table(proto_path, dim=32, count=4)
        _create_dictionary(cache / "dictionary_8d_8.npz", dim=8, atoms=8)
        _create_dictionary(cache / "dictionary_16d_16.npz", dim=16, atoms=16)

        dimension_map = {"fast": 8, "maximum": 16}  # type: ignore[assignment]

        compressor = AdaptiveDimensionCompressor(
            cache_dir=cache,
            prototype_table_path=proto_path,
            enable_compression=True,
            dimension_map=dimension_map,
        )

        assert set(compressor.compilers.keys()) == {8, 16}


def test_quality_levels_compress_and_decompress():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        proto_path = cache / "prototype_table_64d_8.npz"
        _create_prototype_table(proto_path, dim=64, count=4)
        _create_dictionary(cache / "dictionary_8d_8.npz", dim=8, atoms=8)
        _create_dictionary(cache / "dictionary_16d_16.npz", dim=16, atoms=16)

        dimension_map = {"fast": 8, "maximum": 16}  # type: ignore[assignment]

        compressor = AdaptiveDimensionCompressor(
            cache_dir=cache,
            prototype_table_path=proto_path,
            dimension_map=dimension_map,
            enable_compression=True,
        )

        embedding = np.random.randn(32).astype(np.float32)
        for quality in ["fast", "maximum"]:
            program, metadata = compressor.compress(embedding, quality=quality, return_metadata=True)
            assert len(program) <= metadata["target_dim"] * 4 + 64
            reconstructed = compressor.decompress(program, metadata["target_dim"])
            assert reconstructed.shape[0] == metadata["target_dim"]


def test_backward_compatibility_returns_raw_bytes():
    compressor = AdaptiveDimensionCompressor(enable_compression=False)
    embedding = np.arange(16, dtype=np.float32)
    payload = compressor.compress(embedding, return_metadata=False)
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) == embedding.nbytes
