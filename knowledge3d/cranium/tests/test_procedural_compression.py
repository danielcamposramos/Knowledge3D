import numpy as np

from knowledge3d.cranium.fidelity_validator import ProceduralFidelityValidator
from knowledge3d.cranium.procedural_compiler import ProceduralCompiler, PrototypeTable
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def test_simple_round_trip_cosine_similarity():
    compiler = ProceduralCompiler()
    rng = np.random.default_rng(seed=7)
    embedding = rng.normal(size=2048).astype(np.float32)

    program = compiler.compile_embedding_simple(embedding)
    reconstructed = compiler.decompile_simple(program)

    cosine = float(np.dot(embedding, reconstructed) / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed)))

    assert len(program) < embedding.nbytes  # compression happens
    assert cosine > 0.999  # high-fidelity reconstruction


def test_fidelity_validator_summary_metrics():
    rpn_engine = RPNEmbeddingEngine(embedding_dim=256)
    validator = ProceduralFidelityValidator(rpn_engine=rpn_engine, similarity_threshold=0.95)

    tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
    results = validator.batch_validate(tokens)
    summary = validator.summarize(results)

    assert summary["count"] == len(tokens)
    assert 0.0 < summary["average_similarity"] <= 1.0
    assert summary["min_similarity"] <= summary["max_similarity"]
    assert 0.0 <= summary["valid_ratio"] <= 1.0


def test_fidelity_validator_prototype_delta_mode():
    rng = np.random.default_rng(11)
    prototypes = rng.normal(size=(4, 64)).astype(np.float32)
    table = PrototypeTable(prototypes)
    compiler = ProceduralCompiler(prototype_table=table, prototype_topk=4, prototype_topk_cap=8, prototype_cosine_threshold=0.0, use_prototype_basis=False)
    rpn_engine = RPNEmbeddingEngine(embedding_dim=64)
    validator = ProceduralFidelityValidator(rpn_engine=rpn_engine, compiler=compiler, similarity_threshold=0.0)
    result = validator.validate_prototype_sparse("alpha")
    assert result.extra["mode"] == "prototype_delta_sparse"
    assert result.extra["codec"] == "sparse"
    assert result.compression_ratio > 0


def test_fidelity_validator_dictionary_mode():
    dim = 32
    atoms = np.eye(dim, dtype=np.float32)
    compiler = ProceduralCompiler(dictionary_atoms=atoms, dictionary_max_coeffs=4, dictionary_residual_topk=4, dictionary_similarity_threshold=0.5)
    rpn_engine = RPNEmbeddingEngine(embedding_dim=dim)
    validator = ProceduralFidelityValidator(rpn_engine=rpn_engine, compiler=compiler, similarity_threshold=0.0)
    result = validator.validate_dictionary_sparse("beta")
    assert result.extra["mode"] == "dictionary_sparse"
    assert result.compression_ratio > 0
