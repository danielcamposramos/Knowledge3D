import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def test_rpn_word_embedding_determinism_and_norm(tmp_path):
    engine = RPNEmbeddingEngine(embedding_dim=128)

    hello_emb = engine.embed_word("hello")
    world_emb = engine.embed_word("world")
    hello_again = engine.embed_word("hello")

    assert hello_emb.shape == (128,)
    assert world_emb.shape == (128,)

    assert np.isclose(np.linalg.norm(hello_emb), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(world_emb), 1.0, atol=1e-6)

    # Deterministic across calls
    assert np.allclose(hello_emb, hello_again)

    # Different tokens should not collapse to identical vectors
    similarity = float(np.dot(hello_emb, world_emb))
    assert similarity < 0.99

    # Persistence preserves learned table
    save_path = tmp_path / "rpn_embeddings.pkl"
    engine.save_embeddings(save_path)

    restored = RPNEmbeddingEngine(embedding_dim=128)
    restored.load_embeddings(save_path)
    assert np.allclose(restored.embed_word("hello"), hello_emb)


def test_rpn_semantic_clustering():
    engine = RPNEmbeddingEngine(embedding_dim=128)

    cat = engine.embed_word("cat")
    cats = engine.embed_word("cats")
    kitten = engine.embed_word("kitten")
    computer = engine.embed_word("computer")

    cat_cats_sim = float(np.dot(cat, cats))
    cat_kitten_sim = float(np.dot(cat, kitten))
    cat_computer_sim = float(np.dot(cat, computer))

    assert cat_cats_sim > cat_computer_sim
    assert cat_kitten_sim > cat_computer_sim

    # Ensure similarities stay within cosine bounds
    for sim in (cat_cats_sim, cat_kitten_sim, cat_computer_sim):
        assert -1.01 <= sim <= 1.01
