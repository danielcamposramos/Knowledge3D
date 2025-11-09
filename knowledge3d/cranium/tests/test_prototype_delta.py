import numpy as np

from knowledge3d.cranium.procedural_compiler import ProceduralCompiler, PrototypeTable


def _build_table(dim: int = 32, count: int = 4) -> PrototypeTable:
    rng = np.random.default_rng(42)
    prototypes = rng.normal(size=(count, dim)).astype(np.float32)
    return PrototypeTable(prototypes)


def test_prototype_delta_round_trip_cosine():
    table = _build_table()
    compiler = ProceduralCompiler(
        prototype_table=table,
        prototype_topk=8,
        prototype_topk_step=4,
        prototype_topk_cap=16,
        prototype_cosine_threshold=0.98,
    )
    rng = np.random.default_rng(7)
    embedding = rng.normal(size=table.dimension).astype(np.float32)

    payload, meta = compiler.compile_prototype_delta(embedding, return_metadata=True)
    reconstructed = compiler.decompile_prototype_delta(payload)

    cosine = float(np.dot(embedding, reconstructed) / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed)))
    assert cosine >= 0.98
    assert meta["nnz"] <= compiler.prototype_topk_cap
    assert len(payload) < embedding.nbytes


def test_prototype_delta_dynamic_topk_expands():
    table = _build_table()
    compiler = ProceduralCompiler(
        prototype_table=table,
        prototype_topk=2,
        prototype_topk_step=2,
        prototype_topk_cap=12,
        prototype_cosine_threshold=0.999,
    )
    rng = np.random.default_rng(123)
    embedding = rng.normal(size=table.dimension).astype(np.float32)
    _, meta = compiler.compile_prototype_delta(embedding, return_metadata=True)
    assert meta["nnz"] > 2  # expanded beyond initial top-k
    assert meta["similarity"] >= 0.9


def test_prototype_delta_dense_round_trip():
    table = _build_table(dim=64)
    compiler = ProceduralCompiler(
        prototype_table=table,
        prototype_cosine_threshold=0.0,
        use_prototype_basis=False,
    )
    rng = np.random.default_rng(99)
    embedding = rng.normal(size=table.dimension).astype(np.float32)
    payload, meta = compiler.compile_prototype_delta_dense(embedding, return_metadata=True)
    reconstructed = compiler.decompile_prototype_delta_dense(payload)
    cosine = float(np.dot(embedding, reconstructed) / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed)))
    assert cosine > 0.99
    assert meta["codec"] == "dense"
    assert len(payload) < embedding.nbytes


def test_prototype_delta_dense_fallback_to_simple():
    compiler = ProceduralCompiler(prototype_table=None)
    embedding = np.arange(16, dtype=np.float32)
    payload_sparse, meta = compiler.compile_prototype_delta_dense(embedding, return_metadata=True)
    assert meta["codec"] == "simple_fallback"
    # should decode with simple path (length equals simple header +)
    reconstructed = compiler.decompile_simple(payload_sparse)
    assert np.allclose(reconstructed, embedding, atol=1e-1)


def test_prototype_delta_multi_round_trip():
    rng = np.random.default_rng(21)
    prototypes = rng.normal(size=(5, 32)).astype(np.float32)
    table = PrototypeTable(prototypes)
    compiler = ProceduralCompiler(
        prototype_table=table,
        multi_max_prototypes=3,
        multi_candidate_count=5,
        multi_residual_topk=8,
        multi_similarity_threshold=0.9,
    )
    weights_true = np.array([0.8, -0.3, 0.5], dtype=np.float32)
    embedding = weights_true @ table.prototypes[:3] + rng.normal(scale=0.01, size=32).astype(np.float32)
    payload, meta = compiler.compile_prototype_multi(embedding, return_metadata=True)
    assert meta["codec"] in {"multi", "dense"}  # dense if fallback kicks in
    reconstructed = compiler.decompile_prototype_multi(payload) if meta["codec"] == "multi" else compiler.decompile_prototype_delta_dense(payload)
    cosine = float(np.dot(embedding, reconstructed) / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed)))
    assert cosine > 0.95


def test_dictionary_sparse_round_trip():
    dim = 16
    atoms = np.eye(dim, dtype=np.float32)
    compiler = ProceduralCompiler(
        dictionary_atoms=atoms,
        dictionary_max_coeffs=4,
        dictionary_residual_topk=4,
        dictionary_similarity_threshold=0.9,
    )
    rng = np.random.default_rng(5)
    embedding = rng.normal(size=dim).astype(np.float32)
    payload, meta = compiler.compile_dictionary_sparse(embedding, return_metadata=True)
    assert meta["codec"] in {"dict", "dense"}
    reconstructed = (
        compiler.decompile_dictionary_sparse(payload)
        if meta["codec"] == "dict"
        else compiler.decompile_prototype_delta_dense(payload)
    )
    cosine = float(np.dot(embedding, reconstructed) / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed)))
    assert cosine > 0.9
