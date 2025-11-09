import numpy as np

from knowledge3d.cranium.procedural_compiler import PrototypeTable


def test_build_and_persist_prototype_table(tmp_path):
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(512, 16)).astype(np.float32)
    table, stats = PrototypeTable.build_from_embeddings(embeddings, num_prototypes=8, max_iters=10, seed=7, batch_size=128)

    assert table.count == 8
    assert table.dimension == 16
    assert stats["num_embeddings"] == 512
    assert stats["num_prototypes"] == 8
    assert stats["avg_distance"] <= stats["max_distance"]

    idx, proto, dist = table.nearest(embeddings[0])
    assert 0 <= idx < table.count
    assert proto.shape == (16,)
    assert dist >= 0

    eval_metrics = table.evaluate_embeddings(embeddings[:64])
    assert "avg_distance" in eval_metrics
    assert eval_metrics["num_embeddings"] == 64

    target = tmp_path / "prototypes.npz"
    table.save(target)
    loaded = PrototypeTable.load(target)
    assert np.allclose(loaded.prototypes, table.prototypes)
    assert loaded.metadata["num_prototypes"] == 8
